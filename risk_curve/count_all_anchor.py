"""Canonical label-free Count-all anchors derived from integer A-window counts.

The archive never stores trusted floating-point risk curves for this policy.
Every pixel/component rate is derived at the point of use from validated integer
counts and its integer exposure.  This keeps selection independent of future-E
labels and makes malformed or semantically inconsistent archives fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np

from .curve_dataset import validate_count_all_adaptation_contract
from .representation import (
    LOGIT_REPRESENTATION,
    logit_threshold_grid_sha256,
    validate_logit_threshold_grid,
)


COUNT_ALL_ANCHOR_SCHEMA_VERSION = "rc-v4-count-all-anchor-v1"
PIXEL_ANCHOR_LOG_EPSILON = 1.0e-12
COMPONENT_ANCHOR_LOG_EPSILON = 1.0e-6


def _scalar_text(value: Any, field: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{field} must be scalar")
    return str(array.item())


def _strict_integer_array(
    archive: Mapping[str, np.ndarray],
    field: str,
    *,
    shape: tuple[int, ...],
    minimum: int = 0,
) -> np.ndarray:
    if field not in archive:
        raise ValueError(f"Count-all anchor archive is missing {field}")
    raw = np.asarray(archive[field])
    if raw.shape != shape:
        raise ValueError(f"{field} must have shape {shape}")
    if raw.dtype.kind not in "iu":
        raise ValueError(f"{field} must use an integer dtype")
    values = raw.astype(np.int64, copy=True)
    if np.any(values < minimum):
        raise ValueError(f"{field} contains a value below {minimum}")
    values.setflags(write=False)
    return values


def _semantic_sha256(
    *,
    grid_sha256: str,
    thresholds: np.ndarray,
    pixel_counts: np.ndarray,
    component_counts_raw: np.ndarray,
    component_counts_upper: np.ndarray,
    total_pixels: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    header = {
        "schema_version": COUNT_ALL_ANCHOR_SCHEMA_VERSION,
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_sha256": grid_sha256,
        "shape": list(pixel_counts.shape),
    }
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    for array in (
        np.asarray(thresholds, dtype="<f4"),
        np.asarray(pixel_counts, dtype="<i8"),
        np.asarray(component_counts_raw, dtype="<i8"),
        np.asarray(component_counts_upper, dtype="<i8"),
        np.asarray(total_pixels, dtype="<i8"),
    ):
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class CountAllAnchor:
    """One episode's immutable integer sufficient statistics."""

    row: int
    thresholds: np.ndarray
    threshold_grid_sha256: str
    pixel_counts: np.ndarray
    component_counts_raw: np.ndarray
    component_counts_upper: np.ndarray
    total_pixels: int
    archive_semantic_sha256: str
    adaptation_masks_read: bool = False


@dataclass(frozen=True)
class CountAllAnchorBatch:
    """Validated immutable Count-all A-window count batch."""

    thresholds: np.ndarray
    threshold_grid_sha256: str
    pixel_counts: np.ndarray
    component_counts_raw: np.ndarray
    component_counts_upper: np.ndarray
    total_pixels: np.ndarray
    semantic_sha256: str
    contract: Mapping[str, object]

    @property
    def num_episodes(self) -> int:
        return int(self.pixel_counts.shape[0])

    def episode(self, row: int) -> CountAllAnchor:
        if isinstance(row, bool) or not isinstance(row, (int, np.integer)):
            raise TypeError("row must be an integer")
        index = int(row)
        if not 0 <= index < self.num_episodes:
            raise IndexError("Count-all anchor row is out of range")
        return CountAllAnchor(
            row=index,
            thresholds=self.thresholds,
            threshold_grid_sha256=self.threshold_grid_sha256,
            pixel_counts=self.pixel_counts[index],
            component_counts_raw=self.component_counts_raw[index],
            component_counts_upper=self.component_counts_upper[index],
            total_pixels=int(self.total_pixels[index]),
            archive_semantic_sha256=self.semantic_sha256,
            adaptation_masks_read=False,
        )


def validate_count_all_anchor_archive(
    archive: Mapping[str, np.ndarray],
    *,
    expected_grid_sha256: str | None = None,
) -> CountAllAnchorBatch:
    """Validate the full A-count contract and freeze its canonical integers."""

    if not isinstance(archive, Mapping):
        raise TypeError("archive must be a mapping")
    # The shared validator binds schema, representation, provenance masks=false,
    # component suffix envelope, monotonicity, and exposure/count consistency.
    contract = validate_count_all_adaptation_contract(dict(archive), required=True)
    if not bool(contract.get("verified", False)):
        raise ValueError("Count-all adaptation contract is not verified")
    if contract.get("adaptation_masks_read") is not False:
        raise ValueError("Count-all anchor selection requires adaptation_masks_read=false")

    thresholds = validate_logit_threshold_grid(
        np.asarray(archive["thresholds"], dtype=np.float32)
    ).copy()
    thresholds.setflags(write=False)
    semantic_grid_hash = logit_threshold_grid_sha256(thresholds)
    recorded_grid_hash = _scalar_text(
        archive["threshold_grid_sha256"], "threshold_grid_sha256"
    )
    if recorded_grid_hash != semantic_grid_hash:
        raise ValueError("Count-all anchor threshold-grid hash mismatch")
    if expected_grid_sha256 is not None and semantic_grid_hash != expected_grid_sha256:
        raise ValueError("Count-all anchor does not use the required threshold grid")

    statistics = np.asarray(archive["statistics"])
    if statistics.ndim != 2 or statistics.shape[0] <= 0:
        raise ValueError("statistics must have shape [N,D] with N > 0")
    rows = int(statistics.shape[0])
    shape = (rows, int(thresholds.size))
    pixel_counts = _strict_integer_array(
        archive, "adaptation_predicted_pixel_counts", shape=shape
    )
    component_raw = _strict_integer_array(
        archive, "adaptation_predicted_component_counts_raw", shape=shape
    )
    component_upper = _strict_integer_array(
        archive, "adaptation_predicted_component_counts_upper", shape=shape
    )
    total_pixels = _strict_integer_array(
        archive, "adaptation_total_pixels", shape=(rows,), minimum=1
    )
    if np.any(pixel_counts > total_pixels[:, None]):
        raise ValueError("A-window pixel counts exceed integer exposure")
    if np.any(component_raw > pixel_counts):
        raise ValueError("A-window component counts exceed retained pixel counts")
    if np.any(np.diff(pixel_counts, axis=1) > 0):
        raise ValueError("A-window pixel count curves are not monotone")
    expected_upper = np.maximum.accumulate(component_raw[:, ::-1], axis=1)[:, ::-1]
    if not np.array_equal(component_upper, expected_upper):
        raise ValueError("A-window component upper counts were tampered or malformed")
    if np.any(np.diff(component_upper, axis=1) > 0):
        raise ValueError("A-window component upper curves are not monotone")

    semantic_hash = _semantic_sha256(
        grid_sha256=semantic_grid_hash,
        thresholds=thresholds,
        pixel_counts=pixel_counts,
        component_counts_raw=component_raw,
        component_counts_upper=component_upper,
        total_pixels=total_pixels,
    )
    return CountAllAnchorBatch(
        thresholds=thresholds,
        threshold_grid_sha256=semantic_grid_hash,
        pixel_counts=pixel_counts,
        component_counts_raw=component_raw,
        component_counts_upper=component_upper,
        total_pixels=total_pixels,
        semantic_sha256=semantic_hash,
        contract=dict(contract),
    )


def derive_anchor_rates(anchor: CountAllAnchor) -> tuple[np.ndarray, np.ndarray]:
    """Derive pixel/component upper rates directly from integer A counts."""

    if anchor.adaptation_masks_read is not False:
        raise ValueError("Count-all anchor must declare adaptation_masks_read=false")
    if anchor.total_pixels <= 0:
        raise ValueError("Count-all anchor exposure must be positive")
    pixel_counts = np.asarray(anchor.pixel_counts)
    component_upper = np.asarray(anchor.component_counts_upper)
    if pixel_counts.dtype.kind not in "iu" or component_upper.dtype.kind not in "iu":
        raise ValueError("Count-all anchor rates require integer counts")
    if pixel_counts.ndim != 1 or component_upper.shape != pixel_counts.shape:
        raise ValueError("Count-all anchor count curves have incompatible shapes")
    if np.any(pixel_counts < 0) or np.any(component_upper < 0):
        raise ValueError("Count-all anchor counts must be non-negative")
    if np.any(np.diff(pixel_counts.astype(np.int64)) > 0):
        raise ValueError("Count-all anchor pixel counts are not monotone")
    if np.any(np.diff(component_upper.astype(np.int64)) > 0):
        raise ValueError("Count-all anchor component counts are not monotone")
    pixel_rates = pixel_counts.astype(np.float64) / float(anchor.total_pixels)
    component_rates = component_upper.astype(np.float64) / (
        float(anchor.total_pixels) / 1_000_000.0
    )
    if not np.isfinite(pixel_rates).all() or not np.isfinite(component_rates).all():
        raise ValueError("Count-all anchor rates are non-finite")
    pixel_rates.setflags(write=False)
    component_rates.setflags(write=False)
    return pixel_rates, component_rates


def derive_anchor_log_curves(
    anchors: CountAllAnchorBatch,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive canonical vectorized log-risk anchors for model inference.

    The inputs remain the validated integer A-window sufficient statistics.
    No future-E counts or masks are consulted.  The two epsilons are the
    registered physical floors used by the risk heads: ``-12`` for the pixel
    rate and ``-6`` for the component rate per million pixels.
    """

    if not isinstance(anchors, CountAllAnchorBatch):
        raise TypeError("anchors must be a CountAllAnchorBatch")
    if anchors.contract.get("adaptation_masks_read") is not False:
        raise ValueError("Count-all log anchors require adaptation_masks_read=false")
    exposure = np.asarray(anchors.total_pixels, dtype=np.float64)[:, None]
    if exposure.ndim != 2 or exposure.shape[0] != anchors.num_episodes:
        raise ValueError("Count-all anchor exposure shape is invalid")
    pixel_rate = np.asarray(anchors.pixel_counts, dtype=np.float64) / exposure
    component_rate = np.asarray(
        anchors.component_counts_upper, dtype=np.float64
    ) / (exposure / 1_000_000.0)
    pixel_log = np.log10(pixel_rate + PIXEL_ANCHOR_LOG_EPSILON).astype(
        np.float32
    )
    component_log = np.log10(
        component_rate + COMPONENT_ANCHOR_LOG_EPSILON
    ).astype(np.float32)
    expected = anchors.pixel_counts.shape
    for name, values, floor in (
        ("pixel", pixel_log, -12.0),
        ("component", component_log, -6.0),
    ):
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"Count-all {name} log anchor is invalid")
        if np.any(values < floor) or np.any(np.diff(values, axis=1) > 0.0):
            raise ValueError(
                f"Count-all {name} log anchor violates its floor/monotonicity"
            )
    pixel_log = np.ascontiguousarray(pixel_log)
    component_log = np.ascontiguousarray(component_log)
    pixel_log.setflags(write=False)
    component_log.setflags(write=False)
    return pixel_log, component_log


__all__ = [
    "COMPONENT_ANCHOR_LOG_EPSILON",
    "COUNT_ALL_ANCHOR_SCHEMA_VERSION",
    "CountAllAnchor",
    "CountAllAnchorBatch",
    "PIXEL_ANCHOR_LOG_EPSILON",
    "derive_anchor_log_curves",
    "derive_anchor_rates",
    "validate_count_all_anchor_archive",
]
