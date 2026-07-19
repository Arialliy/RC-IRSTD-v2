"""Build a source-only dense raw-logit threshold grid for RC-IRSTD-v2.

This builder has a deliberately different provenance contract from held-out
evaluation.  For every inner source-only detector fold, that detector scores
its own official training domain; all such self-score artifacts are pooled
into one outer-fold grid.  The outer target must be absent from every scored
domain, detector source set, path, and record provenance chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

from evaluation.artifact_integrity import (
    PROBABILITY_DTYPE,
    RAW_LOGIT_DTYPE,
    RAW_LOGIT_SCORE_REPRESENTATION,
    SCORE_MANIFEST_SCHEMA_VERSION,
    SCORE_RECORD_INTEGRITY_SCHEMA,
    file_sha256,
    verify_score_map_directory,
)
from evaluation.threshold_sweep import domain_key

from .representation import (
    GRID_DETECTOR_PROTOCOL,
    DEFAULT_LOGIT_GRID_POINTS,
    LOGIT_GRID_ARTIFACT_TYPE,
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_PREDICTION_RULE,
    LOGIT_REPRESENTATION,
    MAX_MODEL_GRID_POINTS,
    canonical_json_sha256,
    empty_action_contract,
    logit_threshold_grid_sha256,
    validate_logit_threshold_grid,
)


GRID_FILENAME = "threshold_grid.npy"
GRID_MANIFEST_FILENAME = "threshold_grid.json"
GRID_DIGEST_FILENAME = "threshold_grid.sha256"
GRID_BUILDER_VERSION = (
    "rc-v4-source-dense-tail-builder-v4-deterministic-midpoint-fallback"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class DenseTailGridSpec:
    max_grid_points: int = DEFAULT_LOGIT_GRID_POINTS
    bulk_points: int = 128
    upper_points: int = 256
    extreme_points: int = 512
    candidate_points: int = 128
    candidate_window: int = 3
    candidate_connectivity: int = 8
    candidate_quantile_min: float = 0.90
    bulk_quantile_min: float = 0.001
    bulk_quantile_max: float = 0.90
    upper_quantile_max: float = 0.99
    extreme_quantile_max: float = 0.9999
    quantile_method: str = "linear"

    def validate(self) -> None:
        if (
            isinstance(self.max_grid_points, bool)
            or not isinstance(self.max_grid_points, int)
            or self.max_grid_points < 2
            or self.max_grid_points > MAX_MODEL_GRID_POINTS
        ):
            raise ValueError(
                f"max_grid_points must lie in [2, {MAX_MODEL_GRID_POINTS}]"
            )
        point_counts = (
            self.bulk_points,
            self.upper_points,
            self.extreme_points,
            self.candidate_points,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in point_counts
        ):
            raise ValueError("dense-tail point counts must be non-negative integers")
        if sum(point_counts) < 2:
            raise ValueError("dense-tail construction requires at least two points")
        if sum(point_counts) > self.max_grid_points:
            raise ValueError(
                "bulk+upper+extreme+candidate point caps exceed max_grid_points"
            )
        if (
            isinstance(self.candidate_window, bool)
            or not isinstance(self.candidate_window, int)
            or self.candidate_window < 1
            or self.candidate_window % 2 == 0
        ):
            raise ValueError("candidate_window must be a positive odd integer")
        if self.candidate_connectivity not in {4, 8}:
            raise ValueError("candidate_connectivity must be 4 or 8")
        quantiles = (
            self.bulk_quantile_min,
            self.bulk_quantile_max,
            self.upper_quantile_max,
            self.extreme_quantile_max,
            self.candidate_quantile_min,
        )
        if not all(np.isfinite(value) for value in quantiles):
            raise ValueError("dense-tail quantiles must be finite")
        if not (
            0.0
            <= self.bulk_quantile_min
            < self.bulk_quantile_max
            < self.upper_quantile_max
            < self.extreme_quantile_max
            <= 1.0
        ):
            raise ValueError("dense-tail quantile regions are not strictly ordered")
        if not 0.0 <= self.candidate_quantile_min < 1.0:
            raise ValueError("candidate_quantile_min must lie in [0, 1)")
        if self.quantile_method != "linear":
            raise ValueError("the formal v1 grid contract requires quantile_method='linear'")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "max_grid_points": self.max_grid_points,
            "bulk": {
                "points": self.bulk_points,
                "quantile_min": self.bulk_quantile_min,
                "quantile_max": self.bulk_quantile_max,
                "upper_endpoint_included": False,
            },
            "upper_tail": {
                "points": self.upper_points,
                "quantile_min": self.bulk_quantile_max,
                "quantile_max": self.upper_quantile_max,
                "upper_endpoint_included": False,
            },
            "extreme_tail": {
                "points": self.extreme_points,
                "quantile_min": self.upper_quantile_max,
                "quantile_max": self.extreme_quantile_max,
                "upper_endpoint_included": True,
            },
            "background_candidates": {
                "points_cap": self.candidate_points,
                "selection": "upper_tail_quantiles",
                "quantile_min": self.candidate_quantile_min,
                "quantile_max": 1.0,
                "window": self.candidate_window,
                "connectivity": self.candidate_connectivity,
                "plateau_reduction": "one_maximum_per_connected_plateau",
            },
            "quantile_method": self.quantile_method,
            "float_rounding_before_unique": "float32",
        }


@dataclass(frozen=True)
class GridSourceInput:
    root: Path
    manifest: dict[str, Any]
    paths: tuple[Path, ...]
    integrity: dict[str, Any]
    target_dataset: str
    target_domain_key: str
    detector_source_datasets: tuple[str, ...]
    detector_source_domain_keys: tuple[str, ...]
    detector_weight_sha256: str


def _normalise_domains(values: Sequence[str], *, field: str) -> dict[str, str]:
    if not values:
        raise ValueError(f"{field} must contain at least one domain")
    result: dict[str, str] = {}
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must contain non-empty strings")
        display = value.strip()
        key = domain_key(display, field=field)
        if key in result:
            raise ValueError(f"{field} contains duplicate domain aliases")
        result[key] = display
    return result


def _manifest_source_domains(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    raw = manifest.get("source_datasets")
    if raw is None and manifest.get("source_dataset") is not None:
        raw = [manifest.get("source_dataset")]
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(value, str) or not value.strip() for value in raw)
    ):
        raise ValueError("Grid source manifest requires non-empty source_datasets")
    values = tuple(str(value).strip() for value in raw)
    keys = [domain_key(value, field="source_datasets entry") for value in values]
    if len(set(keys)) != len(keys):
        raise ValueError("Grid source manifest contains duplicate source-domain aliases")
    return values


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def validate_grid_source_manifest(
    manifest: Mapping[str, Any] | None,
    integrity: Mapping[str, Any],
    *,
    expected_source_domains: Sequence[str],
    outer_target: str,
) -> dict[str, Any]:
    """Validate one self-scored source-train raw-logit artifact.

    Unlike held-out evaluation, membership of ``target_dataset`` in
    ``source_datasets`` is required here: each fold-specific detector is
    self-scored on its own official source train solely to define a fixed
    threshold support.  With two source domains the complete global grid uses
    three checkpoints: the final detector trained on both sources and the two
    inner leave-one-source-out detectors.  Each checkpoint must self-score all
    of its own official training domains, yielding four input artifacts.
    """

    expected = _normalise_domains(expected_source_domains, field="expected sources")
    outer_key = domain_key(outer_target, field="outer target")
    if outer_key in expected:
        raise ValueError("Outer target appears in the expected source set")
    if manifest is None:
        raise ValueError("Grid construction requires a version-3 score manifest")
    if manifest.get("schema_version") != SCORE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Grid construction requires score manifest schema version 3")
    if manifest.get("record_integrity_schema") != SCORE_RECORD_INTEGRITY_SCHEMA:
        raise ValueError("Grid construction requires the v3 record hash schema")
    if integrity.get("verified") is not True:
        raise ValueError("Grid construction requires verified score-map integrity")
    if integrity.get("mask_alignment_verified") is not True:
        raise ValueError("Grid construction requires verified mask alignment")

    exact_fields = {
        "labels_loaded": True,
        "spatial_mode": "native",
        "split_role": "train",
        "requested_split": "train",
        "split_authority_verified": True,
        "checkpoint_selection_rule": "fixed_last",
        "checkpoint_diagnostic_only": False,
        "score_type": "sigmoid_probability",
        "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
        "probability_dtype": PROBABILITY_DTYPE,
        "logit_dtype": RAW_LOGIT_DTYPE,
        "probability_transform": "sigmoid",
        "probability_clipping": "none",
        "inference_autocast_enabled": False,
    }
    for field, required in exact_fields.items():
        if manifest.get(field) != required:
            raise ValueError(
                f"Grid source manifest {field} must equal {required!r}"
            )
    # RC-MSHNET-PATCH: formal detector backends
    model_backend = manifest.get("model_backend")
    if model_backend not in {"canonical", "rc_mshnet"}:
        raise ValueError(
            "Grid source manifest model_backend must be canonical or "
            "rc_mshnet"
        )
    for field in ("diagnostic_only", "non_strict_state_loading"):
        value = manifest.get(field, False)
        if not isinstance(value, bool) or value:
            raise ValueError(f"Grid source manifest rejects {field}=true/non-boolean")
    if manifest.get("formal_protocol_eligible") is False:
        raise ValueError("Grid source manifest is explicitly non-formal")

    target = manifest.get("target_dataset")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("Grid source manifest requires target_dataset")
    target = target.strip()
    target_key = domain_key(target, field="target_dataset")
    if target_key not in expected:
        raise ValueError(
            f"Scored source domain {target!r} is not in the expected source set"
        )
    if target_key == outer_key:
        raise ValueError("Outer target appears as a scored grid-source domain")

    detector_sources = _manifest_source_domains(manifest)
    detector_source_keys = tuple(
        domain_key(value, field="source_datasets entry")
        for value in detector_sources
    )
    detector_key_set = set(detector_source_keys)
    if not detector_key_set.issubset(set(expected)):
        raise ValueError(
            "A grid detector source_datasets entry is outside expected sources"
        )
    if outer_key in detector_key_set:
        raise ValueError("Outer target appears in final detector source_datasets")
    if target_key not in detector_key_set:
        raise ValueError(
            "Grid protocol requires detector self-scored official source train"
        )

    weight_sha = _require_sha256(
        manifest.get("weight_sha256"), field="weight_sha256"
    )
    warm_flag = manifest.get("warm_flag")
    if not isinstance(warm_flag, bool):
        raise ValueError("Grid source manifest warm_flag must be boolean")
    for field in (
        "split_file_sha256",
        "split_ordered_ids_sha256",
    ):
        _require_sha256(manifest.get(field), field=field)
    return {
        "target_dataset": target,
        "target_domain_key": target_key,
        "detector_source_datasets": list(detector_sources),
        "detector_source_domain_keys": list(detector_source_keys),
        "detector_weight_sha256": weight_sha,
        "warm_flag": warm_flag,
        "score_manifest_sha256": _require_sha256(
            integrity.get("manifest_sha256"), field="score manifest sha256"
        ),
        "score_records_sha256": _require_sha256(
            integrity.get("records_sha256"), field="score records sha256"
        ),
        "score_ordered_image_ids_sha256": _require_sha256(
            integrity.get("ordered_image_ids_sha256"),
            field="score ordered image IDs sha256",
        ),
        "score_num_records": int(integrity.get("num_records", 0)),
        "split_file_sha256": str(manifest["split_file_sha256"]),
        "split_ordered_ids_sha256": str(manifest["split_ordered_ids_sha256"]),
    }


def load_grid_source_input(
    score_dir: str | Path,
    *,
    expected_source_domains: Sequence[str],
    outer_target: str,
) -> GridSourceInput:
    root = Path(score_dir).expanduser().resolve()
    outer_key = domain_key(outer_target, field="outer target")

    def _normalised_text(value: object) -> str:
        return "".join(
            character
            for character in str(value).casefold()
            if character.isalnum()
        )

    if outer_key in _normalised_text(root):
        raise ValueError(
            "Outer target appears in a grid source artifact path"
        )
    manifest, paths, integrity = verify_score_map_directory(
        root,
        require_integrity=True,
        require_masks=True,
    )
    contract = validate_grid_source_manifest(
        manifest,
        integrity,
        expected_source_domains=expected_source_domains,
        outer_target=outer_target,
    )
    assert manifest is not None
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("Grid source manifest records must be a list")
    leaked_ids = [
        str(record.get("image_id"))
        for record in records
        if isinstance(record, dict)
        and outer_key in _normalised_text(record.get("image_id", ""))
    ]
    if leaked_ids:
        raise ValueError(
            "Outer target appears in grid source record IDs: "
            + ", ".join(leaked_ids[:3])
        )
    for field in (
        "weight_path",
        "checkpoint_path",
        "dataset_dir",
        "split_file",
        "output_dir",
    ):
        value = manifest.get(field)
        if value is not None and outer_key in _normalised_text(value):
            raise ValueError(
                f"Outer target appears in grid source manifest {field}"
            )
    return GridSourceInput(
        root=root,
        manifest=dict(manifest),
        paths=tuple(paths),
        integrity=dict(integrity),
        target_dataset=str(contract["target_dataset"]),
        target_domain_key=str(contract["target_domain_key"]),
        detector_source_datasets=tuple(contract["detector_source_datasets"]),
        detector_source_domain_keys=tuple(contract["detector_source_domain_keys"]),
        detector_weight_sha256=str(contract["detector_weight_sha256"]),
    )


def _background_candidate_logits(
    logits: np.ndarray,
    mask: np.ndarray,
    *,
    window: int,
    connectivity: int,
) -> np.ndarray:
    background = np.where(mask, -np.inf, logits)
    pooled = ndimage.maximum_filter(
        background,
        size=window,
        mode="constant",
        cval=-np.inf,
    )
    peak_mask = (~mask) & (background == pooled)
    structure = (
        ndimage.generate_binary_structure(2, 1)
        if connectivity == 4
        else np.ones((3, 3), dtype=bool)
    )
    labels, count = ndimage.label(peak_mask, structure=structure)
    if count == 0:
        return np.empty(0, dtype=np.float32)
    values = ndimage.maximum(
        logits,
        labels=labels,
        index=np.arange(1, count + 1, dtype=np.int32),
    )
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _source_value_stream_sha256(
    ordered_records: Sequence[tuple[str, np.ndarray, np.ndarray]],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"rc-v4-grid-source-logit-mask-stream-v1\0")
    for image_id, logits, mask in ordered_records:
        identifier = image_id.encode("utf-8")
        digest.update(len(identifier).to_bytes(8, "little", signed=False))
        digest.update(identifier)
        digest.update(np.asarray(logits.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(logits, dtype="<f4").tobytes())
        digest.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def collect_source_tail_values(
    source: GridSourceInput,
    *,
    spec: DenseTailGridSpec,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Collect source-background pixels and deterministic local candidates."""

    background_parts: list[np.ndarray] = []
    candidate_parts: list[np.ndarray] = []
    stream_records: list[tuple[str, np.ndarray, np.ndarray]] = []
    manifest_records = source.manifest.get("records")
    if not isinstance(manifest_records, list) or len(manifest_records) != len(source.paths):
        raise ValueError("Grid source manifest/path record counts differ")
    for index, (record, path) in enumerate(zip(manifest_records, source.paths)):
        with np.load(path, allow_pickle=False) as payload:
            logits = np.asarray(payload["logit"])
            mask_raw = np.asarray(payload["mask"])
            image_id = str(np.asarray(payload["image_id"]).item())
        if not isinstance(record, dict) or image_id != record.get("image_id"):
            raise ValueError(f"Grid source record {index} image identity mismatch")
        if logits.dtype != np.float32 or logits.ndim != 2 or not np.isfinite(logits).all():
            raise ValueError(f"Grid source record {index} has invalid raw logits")
        if mask_raw.shape != logits.shape or not np.isin(
            np.unique(mask_raw), (0, 1, False, True)
        ).all():
            raise ValueError(f"Grid source record {index} has invalid mask")
        mask = mask_raw.astype(bool, copy=False)
        background = np.asarray(logits[~mask], dtype=np.float32)
        if background.size:
            background_parts.append(background)
        candidates = _background_candidate_logits(
            logits,
            mask,
            window=spec.candidate_window,
            connectivity=spec.candidate_connectivity,
        )
        if candidates.size:
            candidate_parts.append(candidates)
        stream_records.append((image_id, logits, mask))
    background_values = (
        np.concatenate(background_parts).astype(np.float32, copy=False)
        if background_parts
        else np.empty(0, dtype=np.float32)
    )
    candidate_values = (
        np.concatenate(candidate_parts).astype(np.float32, copy=False)
        if candidate_parts
        else np.empty(0, dtype=np.float32)
    )
    if background_values.size == 0:
        raise ValueError(
            f"Source domain {source.target_dataset!r} has no background logits"
        )
    return background_values, candidate_values, {
        "num_records": len(source.paths),
        "num_background_pixels": int(background_values.size),
        "num_background_candidates": int(candidate_values.size),
        "source_value_stream_sha256": _source_value_stream_sha256(stream_records),
    }


def _quantile_segment(
    values: np.ndarray,
    lower: float,
    upper: float,
    points: int,
    *,
    method: str,
    include_upper_endpoint: bool,
) -> np.ndarray:
    if points == 0:
        return np.empty(0, dtype=np.float32)
    quantiles = np.linspace(
        lower,
        upper,
        num=points,
        endpoint=include_upper_endpoint,
        dtype=np.float64,
    )
    return np.asarray(
        np.quantile(values, quantiles, method=method),
        dtype=np.float32,
    )


def _maximum_adjacent_logit_gap(grid: np.ndarray) -> float:
    """Return the largest adjacent gap in raw-logit value space."""

    values = np.asarray(grid, dtype=np.float32)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("grid must contain at least two float32 points")
    return float(np.max(np.diff(values.astype(np.float64))))


def _refine_largest_float32_gaps(
    grid: np.ndarray,
    *,
    max_points: int,
) -> tuple[np.ndarray, int, str]:
    """Bisect the largest representable raw-logit gap deterministically.

    Ties are resolved by the lower (left-most) interval.  Midpoints are
    computed in float64 and rounded once to float32, exactly matching the
    model-grid representation.  An interval is retired when that rounded
    midpoint equals an endpoint; refinement stops only when the requested cap
    is reached or no interval has a representable float32 midpoint.
    """

    values = np.asarray(grid, dtype=np.float32)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("grid must contain at least two float32 points")
    if max_points < values.size:
        raise ValueError("max_points cannot be smaller than the input grid")

    refined = np.ascontiguousarray(values)
    points_added = 0
    while refined.size < max_points:
        left = refined[:-1]
        right = refined[1:]
        midpoints = np.asarray(
            (left.astype(np.float64) + right.astype(np.float64)) * 0.5,
            dtype=np.float32,
        )
        representable = (midpoints > left) & (midpoints < right)
        if not bool(np.any(representable)):
            return refined, points_added, "no_representable_float32_midpoint"

        gaps = right.astype(np.float64) - left.astype(np.float64)
        gaps[~representable] = -np.inf
        # np.argmax returns the first maximum, so equal-width gaps are split
        # from low to high logit without relying on input traversal order.
        gap_index = int(np.argmax(gaps))
        refined = np.insert(
            refined,
            gap_index + 1,
            midpoints[gap_index],
        ).astype(np.float32, copy=False)
        points_added += 1

    return refined, points_added, "max_grid_points_reached"


def build_dense_tail_grid(
    background_logits: np.ndarray,
    candidate_logits: np.ndarray,
    *,
    spec: DenseTailGridSpec | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct the finite model grid from pooled source-only logits."""

    resolved = spec or DenseTailGridSpec()
    resolved.validate()
    background = np.asarray(background_logits)
    candidates = np.asarray(candidate_logits)
    if background.dtype != np.float32 or background.ndim != 1:
        raise ValueError("background_logits must be a one-dimensional float32 array")
    if candidates.dtype != np.float32 or candidates.ndim != 1:
        raise ValueError("candidate_logits must be a one-dimensional float32 array")
    if background.size == 0 or not np.isfinite(background).all():
        raise ValueError("background_logits must be non-empty and finite")
    if not np.isfinite(candidates).all():
        raise ValueError("candidate_logits must be finite")

    pieces = [
        _quantile_segment(
            background,
            resolved.bulk_quantile_min,
            resolved.bulk_quantile_max,
            resolved.bulk_points,
            method=resolved.quantile_method,
            include_upper_endpoint=False,
        ),
        _quantile_segment(
            background,
            resolved.bulk_quantile_max,
            resolved.upper_quantile_max,
            resolved.upper_points,
            method=resolved.quantile_method,
            include_upper_endpoint=False,
        ),
        _quantile_segment(
            background,
            resolved.upper_quantile_max,
            resolved.extreme_quantile_max,
            resolved.extreme_points,
            method=resolved.quantile_method,
            include_upper_endpoint=True,
        ),
    ]
    unique_candidates = np.unique(candidates)
    selected_candidates = (
        _quantile_segment(
            candidates,
            resolved.candidate_quantile_min,
            1.0,
            resolved.candidate_points,
            method=resolved.quantile_method,
            include_upper_endpoint=True,
        )
        if candidates.size and resolved.candidate_points > 0
        else np.empty(0, dtype=np.float32)
    )
    pieces.append(np.asarray(selected_candidates, dtype=np.float32))
    grid = np.unique(np.concatenate(pieces).astype(np.float32, copy=False))
    initial_grid_points = int(grid.size)
    synthetic_neighbor_added = False
    if grid.size == 1:
        value = np.float32(grid[0])
        neighbor = np.nextafter(value, np.float32(np.inf))
        if not np.isfinite(neighbor):
            neighbor = np.nextafter(value, np.float32(-np.inf))
        if not np.isfinite(neighbor) or neighbor == value:
            raise ValueError("Cannot construct two finite float32 thresholds")
        grid = np.unique(np.asarray([value, neighbor], dtype=np.float32))
        synthetic_neighbor_added = True
    maximum_gap_before_refinement = _maximum_adjacent_logit_gap(grid)
    refinement_enabled = bool(
        resolved.max_grid_points > DEFAULT_LOGIT_GRID_POINTS
    )
    refinement_points_added = 0
    if refinement_enabled and grid.size < resolved.max_grid_points:
        grid, refinement_points_added, refinement_stop_reason = (
            _refine_largest_float32_gaps(
                grid,
                max_points=resolved.max_grid_points,
            )
        )
    elif grid.size >= resolved.max_grid_points:
        refinement_stop_reason = "initial_grid_at_limit"
    else:
        refinement_stop_reason = "disabled_for_default_1024_contract"
    maximum_gap_after_refinement = _maximum_adjacent_logit_gap(grid)
    grid = validate_logit_threshold_grid(
        np.asarray(grid, dtype=np.float32),
        max_points=resolved.max_grid_points,
    )
    return grid, {
        "actual_grid_points": int(grid.size),
        "initial_grid_points": initial_grid_points,
        "candidate_unique_points": int(unique_candidates.size),
        "candidate_selected_points": int(selected_candidates.size),
        "synthetic_float32_neighbor_added": synthetic_neighbor_added,
        "refinement_enabled": refinement_enabled,
        "refinement_strategy": (
            "largest_adjacent_raw_logit_gap_float32_midpoint"
        ),
        "refinement_tie_break": "lowest_left_endpoint",
        "refinement_points_added": int(refinement_points_added),
        "refinement_stop_reason": refinement_stop_reason,
        "max_adjacent_logit_gap_before": maximum_gap_before_refinement,
        "max_adjacent_logit_gap_after": maximum_gap_after_refinement,
        "observed_background_logit_min": float(background.min()),
        "observed_background_logit_max": float(background.max()),
        "observed_candidate_logit_min": (
            float(candidates.min()) if candidates.size else None
        ),
        "observed_candidate_logit_max": (
            float(candidates.max()) if candidates.size else None
        ),
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _npy_bytes(values: np.ndarray) -> bytes:
    import io

    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    return buffer.getvalue()


def build_logit_threshold_grid_artifact(
    source_score_dirs: Sequence[str | Path],
    *,
    expected_source_domains: Sequence[str],
    outer_target: str,
    output_dir: str | Path,
    spec: DenseTailGridSpec | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build and atomically write the formal three-file grid artifact."""

    resolved = spec or DenseTailGridSpec()
    resolved.validate()
    if not source_score_dirs:
        raise ValueError("At least one --source-score-dir is required")
    expected = _normalise_domains(expected_source_domains, field="expected sources")
    if len(expected) < 2:
        raise ValueError("Global source-only grid requires at least two source domains")
    outer_display = str(outer_target).strip()
    outer_key = domain_key(outer_display, field="outer target")
    if outer_key in expected:
        raise ValueError("Outer target appears in the expected source set")

    loaded = [
        load_grid_source_input(
            directory,
            expected_source_domains=tuple(expected.values()),
            outer_target=outer_display,
        )
        for directory in source_score_dirs
    ]
    loaded.sort(
        key=lambda item: (item.detector_weight_sha256, item.target_domain_key)
    )
    # RC-MSHNET-PATCH: backend consistency
    model_backends = {
        str(item.manifest.get("model_backend")) for item in loaded
    }
    if len(model_backends) != 1:
        raise ValueError(
            "All source-only grid artifacts must use one detector backend"
        )
    model_backend = next(iter(model_backends))
    input_pairs = [
        (item.detector_weight_sha256, item.target_domain_key) for item in loaded
    ]
    if len(set(input_pairs)) != len(input_pairs):
        raise ValueError(
            "Multiple grid inputs declare the same detector/source-domain pair"
        )
    target_keys = [item.target_domain_key for item in loaded]
    if set(target_keys) != set(expected):
        missing = sorted(set(expected).difference(target_keys))
        extra = sorted(set(target_keys).difference(expected))
        raise ValueError(
            f"Scored grid-source set differs from expected sources: missing={missing}, extra={extra}"
        )
    by_checkpoint: dict[str, list[GridSourceInput]] = {}
    for item in loaded:
        by_checkpoint.setdefault(item.detector_weight_sha256, []).append(item)
    detector_source_sets: dict[str, frozenset[str]] = {}
    detector_folds: list[dict[str, Any]] = []
    for checkpoint_hash, fold_inputs in by_checkpoint.items():
        declared_sets = {
            frozenset(item.detector_source_domain_keys) for item in fold_inputs
        }
        if len(declared_sets) != 1:
            raise ValueError(
                "One detector checkpoint declares inconsistent source-domain sets"
            )
        declared = next(iter(declared_sets))
        scored = {item.target_domain_key for item in fold_inputs}
        if scored != set(declared):
            raise ValueError(
                "Every source-only detector must self-score all and only its "
                "official training domains"
            )
        detector_source_sets[checkpoint_hash] = declared

    full_source_set = frozenset(expected)
    expected_fold_sets = {full_source_set}
    expected_fold_sets.update(
        frozenset(set(expected).difference({held_out}))
        for held_out in expected
    )
    if frozenset() in expected_fold_sets:
        expected_fold_sets.remove(frozenset())
    actual_fold_sets = set(detector_source_sets.values())
    if actual_fold_sets != expected_fold_sets:
        raise ValueError(
            "Grid inputs must cover the final full-source detector and every "
            "inner leave-one-source-out detector exactly once"
        )
    if len(actual_fold_sets) != len(detector_source_sets):
        raise ValueError(
            "Multiple detector checkpoints declare the same source-only fold"
        )
    checkpoint_hashes = set(detector_source_sets)
    outer_detector_hash = next(
        checkpoint_hash
        for checkpoint_hash, source_set in detector_source_sets.items()
        if source_set == full_source_set
    )
    episode_detector_hashes = sorted(
        checkpoint_hash
        for checkpoint_hash, source_set in detector_source_sets.items()
        if source_set != full_source_set
    )
    for checkpoint_hash, source_set in sorted(
        detector_source_sets.items(), key=lambda item: (sorted(item[1]), item[0])
    ):
        held_out = sorted(set(expected).difference(source_set))
        detector_folds.append(
            {
                "detector_checkpoint_sha256": checkpoint_hash,
                "source_domain_keys": sorted(source_set),
                "scored_official_train_domain_keys": sorted(source_set),
                "held_out_pseudo_target_keys": held_out,
                "role": (
                    "outer_final_detector"
                    if source_set == full_source_set
                    else "inner_pseudo_target_detector"
                ),
            }
        )
    warm_flags = {item.manifest.get("warm_flag") for item in loaded}
    if len(warm_flags) != 1:
        raise ValueError("All grid sources must use the same detector inference head")

    background_parts: list[np.ndarray] = []
    candidate_parts: list[np.ndarray] = []
    input_provenance: list[dict[str, Any]] = []
    for source in loaded:
        background, candidates, counts = collect_source_tail_values(
            source, spec=resolved
        )
        background_parts.append(background)
        candidate_parts.append(candidates)
        input_provenance.append(
            {
                "target_dataset": source.target_dataset,
                "target_domain_key": source.target_domain_key,
                "score_dir": str(source.root),
                "score_manifest": str(source.root / "manifest.json"),
                "score_manifest_sha256": str(
                    source.integrity["manifest_sha256"]
                ),
                "score_records_sha256": str(source.integrity["records_sha256"]),
                "score_ordered_image_ids_sha256": str(
                    source.integrity["ordered_image_ids_sha256"]
                ),
                "split_file_sha256": str(source.manifest["split_file_sha256"]),
                "split_ordered_ids_sha256": str(
                    source.manifest["split_ordered_ids_sha256"]
                ),
                "detector_source_datasets": list(
                    source.detector_source_datasets
                ),
                "detector_source_domain_keys": list(
                    source.detector_source_domain_keys
                ),
                "detector_weight_sha256": source.detector_weight_sha256,
                "warm_flag": bool(source.manifest["warm_flag"]),
                **counts,
            }
        )
    pooled_background = np.concatenate(background_parts).astype(
        np.float32, copy=False
    )
    pooled_candidates = (
        np.concatenate(candidate_parts).astype(np.float32, copy=False)
        if any(part.size for part in candidate_parts)
        else np.empty(0, dtype=np.float32)
    )
    grid, construction_audit = build_dense_tail_grid(
        pooled_background,
        pooled_candidates,
        spec=resolved,
    )
    semantic_sha = logit_threshold_grid_sha256(grid)

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    grid_path = output / GRID_FILENAME
    manifest_path = output / GRID_MANIFEST_FILENAME
    digest_path = output / GRID_DIGEST_FILENAME
    existing = [path for path in (grid_path, manifest_path, digest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Grid artifact already exists; pass overwrite=True/--force: "
            + ", ".join(str(path) for path in existing)
        )

    npy_payload = _npy_bytes(grid)
    grid_file_sha = hashlib.sha256(npy_payload).hexdigest()
    source_keys = sorted(expected)
    source_domains = [expected[key] for key in source_keys]
    manifest: dict[str, Any] = {
        "schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "artifact_type": LOGIT_GRID_ARTIFACT_TYPE,
        "builder_version": GRID_BUILDER_VERSION,
        "representation": LOGIT_REPRESENTATION,
        "dtype": "float32",
        "prediction_rule": LOGIT_PREDICTION_RULE,
        "grid_source": "source_official_train_only",
        "grid_file": GRID_FILENAME,
        "digest_file": GRID_DIGEST_FILENAME,
        "grid_points": int(grid.size),
        "finite_grid_points": int(grid.size),
        "max_model_grid_points": MAX_MODEL_GRID_POINTS,
        "grid_sha256": semantic_sha,
        "grid_file_sha256": grid_file_sha,
        "empty_action": empty_action_contract(),
        "source_domains": source_domains,
        "source_domain_keys": source_keys,
        "expected_source_domains": [expected[key] for key in sorted(expected)],
        "expected_source_domain_keys": sorted(expected),
        "outer_target": outer_display,
        "outer_target_key": outer_key,
        "outer_target_excluded": True,
        "outer_target_labels_used": False,
        "source_train_masks_used": True,
        "source_train_mask_use": (
            "select background pixels and background local-maximum candidates"
        ),
        "grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        # RC-MSHNET-PATCH: record grid backend
        "model_backend": model_backend,
        "detector_checkpoint_count": len(checkpoint_hashes),
        "detector_checkpoint_sha256s": sorted(checkpoint_hashes),
        "outer_detector_checkpoint_sha256": outer_detector_hash,
        "episode_detector_checkpoint_sha256s": episode_detector_hashes,
        "detector_folds": detector_folds,
        "detector_warm_flag": next(iter(warm_flags)),
        "input_score_artifacts": input_provenance,
        "source_provenance_sha256": canonical_json_sha256(input_provenance),
        "construction": resolved.to_dict(),
        "construction_audit": {
            **construction_audit,
            "pooled_background_pixels": int(pooled_background.size),
            "pooled_background_candidates": int(pooled_candidates.size),
        },
        "formal_protocol_eligible": True,
    }
    manifest_payload = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(grid_path, npy_payload)
    _atomic_write_bytes(manifest_path, manifest_payload)
    _atomic_write_bytes(digest_path, f"{semantic_sha}\n".encode("ascii"))
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-score-dir", action="append", required=True)
    parser.add_argument("--expected-source-domain", action="append", required=True)
    parser.add_argument("--outer-target", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--max-grid-points", type=int, default=DEFAULT_LOGIT_GRID_POINTS
    )
    parser.add_argument("--bulk-points", type=int, default=128)
    parser.add_argument("--upper-points", type=int, default=256)
    parser.add_argument("--extreme-points", type=int, default=512)
    parser.add_argument("--candidate-points", type=int, default=128)
    parser.add_argument("--candidate-window", type=int, default=3)
    parser.add_argument(
        "--candidate-connectivity", type=int, choices=(4, 8), default=8
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    spec = DenseTailGridSpec(
        max_grid_points=args.max_grid_points,
        bulk_points=args.bulk_points,
        upper_points=args.upper_points,
        extreme_points=args.extreme_points,
        candidate_points=args.candidate_points,
        candidate_window=args.candidate_window,
        candidate_connectivity=args.candidate_connectivity,
    )
    manifest = build_logit_threshold_grid_artifact(
        args.source_score_dir,
        expected_source_domains=args.expected_source_domain,
        outer_target=args.outer_target,
        output_dir=args.output_dir,
        spec=spec,
        overwrite=args.force,
    )
    print(
        json.dumps(
            {
                "grid_points": manifest["grid_points"],
                "grid_sha256": manifest["grid_sha256"],
                "manifest": str(
                    Path(args.output_dir).expanduser().resolve()
                    / GRID_MANIFEST_FILENAME
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DenseTailGridSpec",
    "GRID_BUILDER_VERSION",
    "GRID_DIGEST_FILENAME",
    "GRID_FILENAME",
    "GRID_MANIFEST_FILENAME",
    "GridSourceInput",
    "build_argument_parser",
    "build_dense_tail_grid",
    "build_logit_threshold_grid_artifact",
    "collect_source_tail_values",
    "load_grid_source_input",
    "main",
    "validate_grid_source_manifest",
]
