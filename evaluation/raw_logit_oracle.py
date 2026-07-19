"""Exact diagnostic Oracle over integrity-checked FP32 raw-logit score maps.

This tool is deliberately label-derived and therefore can never select a
deployment threshold.  Its purpose is to distinguish genuine model ranking
from float32 sigmoid saturation: every distinct raw-logit prediction state is
considered, with only states that are mathematically impossible under the
pixel budget omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifact_integrity import (
    PROBABILITY_DTYPE,
    RAW_LOGIT_DTYPE,
    RAW_LOGIT_SCORE_REPRESENTATION,
    SCORE_MANIFEST_SCHEMA_VERSION,
    SCORE_RECORD_INTEGRITY_SCHEMA,
    verify_score_map_directory,
)
from .budget_metrics import is_budget_satisfied
from .component_matching import MatchResult, connected_components, match_components
from .threshold_sweep import validate_formal_score_manifest, write_json_atomic


RAW_LOGIT_ORACLE_SCHEMA_VERSION = 1
RAW_LOGIT_ORACLE_ARTIFACT_TYPE = "rc-irstd-exact-raw-logit-global-oracle"


@dataclass(frozen=True)
class RawLogitSample:
    """One native-resolution labeled raw-logit record."""

    image_id: str
    logits: np.ndarray
    probability: np.ndarray
    mask: np.ndarray


def _validate_budget(value: float, *, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return number


def _boolean_scalar(value: Any, *, name: str) -> bool:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean scalar")
    return bool(array.item())


def _string_scalar(value: Any, *, name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be a string scalar")
    return str(array.item())


def _validated_sample(sample: RawLogitSample, index: int) -> RawLogitSample:
    if not isinstance(sample.image_id, str) or not sample.image_id:
        raise ValueError(f"raw-logit sample {index} has no valid image_id")
    logits = np.asarray(sample.logits)
    probability = np.asarray(sample.probability)
    mask = np.asarray(sample.mask)
    if logits.dtype != np.float32:
        raise ValueError(f"raw-logit sample {index} logits must be float32")
    if probability.dtype != np.float32:
        raise ValueError(f"raw-logit sample {index} probability must be float32")
    if logits.ndim != 2 or probability.ndim != 2 or mask.ndim != 2:
        raise ValueError(f"raw-logit sample {index} arrays must be 2-D")
    if logits.size == 0:
        raise ValueError(f"raw-logit sample {index} must not be empty")
    if logits.shape != probability.shape or logits.shape != mask.shape:
        raise ValueError(f"raw-logit sample {index} arrays have different shapes")
    if not np.isfinite(logits).all():
        raise ValueError(f"raw-logit sample {index} contains non-finite logits")
    if not np.isfinite(probability).all() or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError(f"raw-logit sample {index} has invalid probabilities")
    if (
        (not np.issubdtype(mask.dtype, np.number) and mask.dtype != np.bool_)
        or not np.isfinite(mask).all()
        or not np.isin(np.unique(mask), (0, 1, False, True)).all()
    ):
        raise ValueError(f"raw-logit sample {index} mask must be finite and binary")
    return RawLogitSample(
        image_id=sample.image_id,
        logits=np.ascontiguousarray(logits),
        probability=np.ascontiguousarray(probability),
        mask=np.ascontiguousarray(mask.astype(bool, copy=False)),
    )


def validate_formal_raw_logit_manifest(
    manifest: Mapping[str, Any] | None,
    integrity: Mapping[str, Any],
    *,
    expected_split_role: str = "test",
) -> dict[str, Any]:
    """Fail closed on the full held-out raw-logit Oracle input contract."""

    if manifest is None:
        raise ValueError("Raw-logit Oracle requires a v3 score-map manifest")
    if manifest.get("schema_version") != SCORE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Raw-logit Oracle requires score manifest schema version 3")
    if manifest.get("record_integrity_schema") != SCORE_RECORD_INTEGRITY_SCHEMA:
        raise ValueError("Raw-logit Oracle requires the complete v3 record hash schema")
    if integrity.get("verified") is not True:
        raise ValueError("Raw-logit Oracle requires verified v3 score-map integrity")
    if integrity.get("mask_alignment_verified") is not True:
        raise ValueError("Raw-logit Oracle requires verified mask/alignment evidence")

    # Reuse the formal contract for masks, native resolution, split authority,
    # fixed-last strict checkpoint loading, an approved detector backend, and
    # source/held-out-target exclusion.  Oracle selection remains diagnostic.
    contract = validate_formal_score_manifest(
        manifest,
        integrity,
        expected_split_role=expected_split_role,
    )
    expected = {
        "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
        "probability_dtype": PROBABILITY_DTYPE,
        "logit_dtype": RAW_LOGIT_DTYPE,
        "probability_transform": "sigmoid",
        "probability_clipping": "none",
        "inference_autocast_enabled": False,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"Raw-logit Oracle requires manifest {field}={value!r}"
            )
    return {**contract, **expected}


def load_formal_raw_logit_directory(
    score_dir: str | Path,
    *,
    expected_split_role: str = "test",
) -> tuple[
    list[RawLogitSample],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Load ordered raw logits only after the complete formal input audit."""

    manifest, paths, integrity = verify_score_map_directory(
        score_dir,
        require_integrity=True,
        require_masks=True,
    )
    contract = validate_formal_raw_logit_manifest(
        manifest,
        integrity,
        expected_split_role=expected_split_role,
    )
    assert manifest is not None  # guarded above
    records = manifest["records"]
    samples: list[RawLogitSample] = []
    for index, (record, path) in enumerate(zip(records, paths)):
        with np.load(path, allow_pickle=False) as payload:
            required = {
                "logit",
                "prob",
                "mask",
                "image_id",
                "labels_loaded",
                "spatial_mode",
                "score_representation",
                "probability_dtype",
                "logit_dtype",
                "probability_transform",
                "probability_clipping",
                "inference_autocast_enabled",
            }
            missing = required.difference(payload.files)
            if missing:
                raise ValueError(
                    f"Raw-logit record {index} lacks fields: "
                    + ", ".join(sorted(missing))
                )
            image_id = _string_scalar(payload["image_id"], name="image_id")
            if image_id != record["image_id"]:
                raise ValueError(f"Raw-logit record {index} image_id mismatch")
            if not _boolean_scalar(payload["labels_loaded"], name="labels_loaded"):
                raise ValueError(f"Raw-logit record {index} is label-free")
            if _string_scalar(payload["spatial_mode"], name="spatial_mode") != "native":
                raise ValueError(f"Raw-logit record {index} is not native resolution")
            embedded_contract = {
                "score_representation": _string_scalar(
                    payload["score_representation"], name="score_representation"
                ),
                "probability_dtype": _string_scalar(
                    payload["probability_dtype"], name="probability_dtype"
                ),
                "logit_dtype": _string_scalar(
                    payload["logit_dtype"], name="logit_dtype"
                ),
                "probability_transform": _string_scalar(
                    payload["probability_transform"], name="probability_transform"
                ),
                "probability_clipping": _string_scalar(
                    payload["probability_clipping"], name="probability_clipping"
                ),
                "inference_autocast_enabled": _boolean_scalar(
                    payload["inference_autocast_enabled"],
                    name="inference_autocast_enabled",
                ),
            }
            for field in embedded_contract:
                if embedded_contract[field] != contract[field]:
                    raise ValueError(
                        f"Raw-logit record {index} has inconsistent {field}"
                    )
            samples.append(
                _validated_sample(
                    RawLogitSample(
                        image_id=image_id,
                        logits=np.asarray(payload["logit"]),
                        probability=np.asarray(payload["prob"]),
                        mask=np.asarray(payload["mask"]),
                    ),
                    index,
                )
            )
    if len(samples) != len(paths):
        raise AssertionError("raw-logit loader lost manifest records")
    return samples, manifest, integrity, contract


def compare_probability_with_reference(
    raw_score_dir: str | Path,
    reference_probability_dir: str | Path,
) -> dict[str, Any]:
    """Require bitwise-equal probabilities against an existing formal v3 export."""

    raw_manifest, raw_paths, raw_integrity = verify_score_map_directory(
        raw_score_dir,
        require_integrity=True,
        require_masks=True,
    )
    raw_contract = validate_formal_raw_logit_manifest(raw_manifest, raw_integrity)
    reference_manifest, reference_paths, reference_integrity = (
        verify_score_map_directory(
            reference_probability_dir,
            require_integrity=True,
            require_masks=True,
        )
    )
    reference_contract = validate_formal_score_manifest(
        reference_manifest,
        reference_integrity,
        expected_split_role="test",
    )
    if reference_manifest is None:  # guarded by formal validation
        raise AssertionError("formal probability reference has no manifest")
    provenance_fields = (
        "target_dataset",
        "source_datasets",
        "detector_weight_sha256",
        "requested_split",
        "split_role",
        "split_file_sha256",
        "split_ordered_ids_sha256",
        "score_ordered_image_ids_sha256",
        "score_num_records",
    )
    for field in provenance_fields:
        if raw_contract[field] != reference_contract[field]:
            raise ValueError(
                f"Raw/reference probability provenance differs at {field}"
            )
    if len(raw_paths) != len(reference_paths):
        raise ValueError("Raw/reference probability record counts differ")

    total_pixels = 0
    for index, (raw_path, reference_path) in enumerate(
        zip(raw_paths, reference_paths)
    ):
        with np.load(raw_path, allow_pickle=False) as raw_payload, np.load(
            reference_path, allow_pickle=False
        ) as reference_payload:
            raw_probability = np.asarray(raw_payload["prob"])
            reference_probability = np.asarray(reference_payload["prob"])
            if raw_probability.shape != reference_probability.shape:
                raise ValueError(
                    f"Raw/reference probability shape differs at record {index}"
                )
            if not np.array_equal(raw_probability, reference_probability):
                differing = int(
                    np.count_nonzero(raw_probability != reference_probability)
                )
                raise ValueError(
                    "Raw/reference probability values are not bitwise equal at "
                    f"record {index}; differing_pixels={differing}"
                )
            total_pixels += int(raw_probability.size)
    return {
        "provided": True,
        "bitwise_equal": True,
        "comparison_dtype": PROBABILITY_DTYPE,
        "num_records": len(raw_paths),
        "num_pixels": total_pixels,
        "raw_score_manifest_sha256": raw_contract["score_manifest_sha256"],
        "raw_score_records_sha256": raw_contract["score_records_sha256"],
        "reference_score_dir": str(
            Path(reference_probability_dir).expanduser().resolve()
        ),
        "reference_score_manifest_sha256": reference_contract[
            "score_manifest_sha256"
        ],
        "reference_score_records_sha256": reference_contract[
            "score_records_sha256"
        ],
    }


def raw_logit_stream_sha256(samples: Sequence[RawLogitSample]) -> str:
    """Hash ordered image IDs, shapes, and canonical little-endian FP32 logits."""

    if not samples:
        raise ValueError("at least one raw-logit sample is required")
    digest = hashlib.sha256()
    digest.update(b"rc-irstd-ordered-raw-logit-f32-v1\0")
    for index, raw_sample in enumerate(samples):
        sample = _validated_sample(raw_sample, index)
        identifier = sample.image_id.encode("utf-8")
        digest.update(len(identifier).to_bytes(8, "little", signed=False))
        digest.update(identifier)
        digest.update(np.asarray(sample.logits.shape, dtype="<i8").tobytes())
        digest.update(
            np.ascontiguousarray(sample.logits, dtype="<f4").tobytes(order="C")
        )
    return digest.hexdigest()


def audit_exact_one_saturation(
    samples: Sequence[RawLogitSample],
    *,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
) -> dict[str, Any]:
    """Audit every probability pixel that rounded to exactly float32 one."""

    if not samples:
        raise ValueError("at least one raw-logit sample is required")
    per_image: list[dict[str, Any]] = []
    for index, raw_sample in enumerate(samples):
        sample = _validated_sample(raw_sample, index)
        exact_one = sample.probability == np.float32(1.0)
        result = match_components(
            exact_one,
            sample.mask,
            rule=matching_rule,
            centroid_distance=centroid_distance,
            connectivity=connectivity,
            min_component_area=min_component_area,
        )
        target_pixels = int(np.count_nonzero(sample.mask))
        exact_target = int(np.count_nonzero(exact_one & sample.mask))
        exact_background = int(np.count_nonzero(exact_one & ~sample.mask))
        per_image.append(
            {
                "image_id": sample.image_id,
                "total_pixels": int(sample.mask.size),
                "target_pixels": target_pixels,
                "background_pixels": int(sample.mask.size - target_pixels),
                "exact_one_pixels": int(np.count_nonzero(exact_one)),
                "exact_one_target_pixels": exact_target,
                "exact_one_background_pixels": exact_background,
                "candidate_components": int(
                    result.num_tp_objects + result.num_fp_components
                ),
                "matched_target_candidate_components": int(result.num_tp_objects),
                "false_candidate_components": int(result.num_fp_components),
                "false_candidate_pixels": int(result.num_fp_pixels),
                "gt_objects": int(result.num_gt),
            }
        )

    summed_fields = (
        "total_pixels",
        "target_pixels",
        "background_pixels",
        "exact_one_pixels",
        "exact_one_target_pixels",
        "exact_one_background_pixels",
        "candidate_components",
        "matched_target_candidate_components",
        "false_candidate_components",
        "false_candidate_pixels",
        "gt_objects",
    )
    totals = {
        field: int(sum(int(row[field]) for row in per_image))
        for field in summed_fields
    }
    return {
        "probability_value": 1.0,
        "probability_dtype": PROBABILITY_DTYPE,
        "probability_clipping": "none",
        "num_images": len(per_image),
        "num_images_with_exact_one": int(
            sum(int(row["exact_one_pixels"]) > 0 for row in per_image)
        ),
        **totals,
        "exact_one_fraction_all_pixels": float(
            totals["exact_one_pixels"] / totals["total_pixels"]
        ),
        "exact_one_fraction_target_pixels": float(
            totals["exact_one_target_pixels"] / totals["target_pixels"]
        )
        if totals["target_pixels"]
        else 0.0,
        "exact_one_fraction_background_pixels": float(
            totals["exact_one_background_pixels"] / totals["background_pixels"]
        )
        if totals["background_pixels"]
        else 0.0,
        "per_image": per_image,
    }


def _local_background_peak_values(
    sample: RawLogitSample,
    *,
    connectivity: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one deterministic representative per 3x3 background plateau."""

    background = np.where(sample.mask, -np.inf, sample.logits)
    padded = np.pad(background, 1, mode="constant", constant_values=-np.inf)
    neighborhoods = [
        padded[row : row + background.shape[0], col : col + background.shape[1]]
        for row in range(3)
        for col in range(3)
    ]
    pooled = np.maximum.reduce(neighborhoods)
    peak_mask = (~sample.mask) & (background == pooled)
    labels, count = connected_components(
        peak_mask,
        connectivity=connectivity,
        min_component_area=1,
    )
    if count == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    flat_labels = labels.reshape(-1)
    indices = np.flatnonzero(flat_labels)
    component_ids = flat_labels[indices]
    values = sample.logits.reshape(-1)[indices]
    # Group by plateau, prefer maximum raw logit, then the lowest flat index.
    order = np.lexsort((indices, -values, component_ids))
    ordered_ids = component_ids[order]
    first = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.flatnonzero(np.diff(ordered_ids)) + 1)
    )
    representatives = indices[order[first]]
    if representatives.size != count:
        raise AssertionError("background peak plateau extraction is inconsistent")
    return (
        np.asarray(sample.logits.reshape(-1)[representatives], dtype=np.float32),
        np.asarray(sample.probability.reshape(-1)[representatives], dtype=np.float32),
    )


def _quantiles(values: np.ndarray, quantiles: Sequence[float]) -> dict[str, float]:
    if values.size == 0:
        return {}
    result = np.quantile(values.astype(np.float64), np.asarray(quantiles))
    return {
        f"q{100.0 * float(quantile):g}": float(value)
        for quantile, value in zip(quantiles, result.tolist())
    }


def audit_target_background_ranking(
    samples: Sequence[RawLogitSample],
    *,
    operating_threshold_logit: float,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
    top_k: int = 100,
) -> dict[str, Any]:
    """Measure GT/background ordering directly in the unsaturated logit domain."""

    if not samples:
        raise ValueError("at least one raw-logit sample is required")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    threshold = float(operating_threshold_logit)
    if not np.isfinite(threshold):
        raise ValueError("operating_threshold_logit must be finite")
    validated = [
        _validated_sample(sample, index) for index, sample in enumerate(samples)
    ]

    gt_records: list[dict[str, Any]] = []
    background_parts: list[np.ndarray] = []
    false_peak_parts: list[np.ndarray] = []
    false_peak_probability_parts: list[np.ndarray] = []
    false_component_records: list[dict[str, Any]] = []
    for sample in validated:
        gt_labels, gt_count = connected_components(
            sample.mask,
            connectivity=connectivity,
            min_component_area=1,
        )
        for component_id in range(1, gt_count + 1):
            selected = gt_labels == component_id
            target_logits = sample.logits[selected]
            target_probability = sample.probability[selected]
            gt_records.append(
                {
                    "image_id": sample.image_id,
                    "gt_component_id": int(component_id),
                    "num_pixels": int(target_logits.size),
                    "max_target_logit": float(np.max(target_logits)),
                    "mean_target_logit": float(np.mean(target_logits, dtype=np.float64)),
                    "target_component_max_logit": float(np.max(target_logits)),
                    "max_target_probability_float32": float(
                        np.max(target_probability)
                    ),
                }
            )
        background_parts.append(sample.logits[~sample.mask].reshape(-1))
        peak_logits, peak_probability = _local_background_peak_values(
            sample,
            connectivity=connectivity,
        )
        false_peak_parts.append(peak_logits)
        false_peak_probability_parts.append(peak_probability)

        # ``reject_all_threshold_logit`` is a float64 sentinel immediately
        # above the largest float32 logit.  NumPy's scalar promotion can round
        # that sentinel back to float32 during comparison, so compare in
        # float64 here to preserve the advertised prediction state.
        prediction = sample.logits.astype(np.float64) >= threshold
        prediction_labels, _ = connected_components(
            prediction,
            connectivity=connectivity,
            min_component_area=min_component_area,
        )
        matched = match_components(
            prediction,
            sample.mask,
            rule=matching_rule,
            centroid_distance=centroid_distance,
            connectivity=connectivity,
            min_component_area=min_component_area,
        )
        matched_prediction_ids = {
            int(prediction_id) for prediction_id, _ in matched.matched_pairs
        }
        for component_id in range(1, int(prediction_labels.max()) + 1):
            if component_id in matched_prediction_ids:
                continue
            component = prediction_labels == component_id
            false_component_records.append(
                {
                    "image_id": sample.image_id,
                    "prediction_component_id": int(component_id),
                    "num_pixels": int(np.count_nonzero(component)),
                    "max_logit": float(np.max(sample.logits[component])),
                }
            )

    background = (
        np.concatenate(background_parts).astype(np.float32, copy=False)
        if background_parts
        else np.empty(0, dtype=np.float32)
    )
    false_peaks = (
        np.concatenate(false_peak_parts).astype(np.float32, copy=False)
        if false_peak_parts
        else np.empty(0, dtype=np.float32)
    )
    false_peak_probabilities = (
        np.concatenate(false_peak_probability_parts).astype(np.float32, copy=False)
        if false_peak_probability_parts
        else np.empty(0, dtype=np.float32)
    )
    gt_max = np.asarray(
        [record["max_target_logit"] for record in gt_records], dtype=np.float32
    )
    false_component_max = np.asarray(
        [record["max_logit"] for record in false_component_records],
        dtype=np.float32,
    )

    curve_breakpoints: list[np.ndarray] = []
    if gt_max.size:
        curve_breakpoints.append(gt_max)
    if false_peaks.size:
        curve_breakpoints.append(
            np.quantile(
                false_peaks.astype(np.float64),
                np.asarray([0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, 1.0]),
            ).astype(np.float32)
        )
    if background.size:
        curve_breakpoints.append(
            np.quantile(
                background.astype(np.float64),
                np.asarray([0.99, 0.999, 0.9999, 1.0]),
            ).astype(np.float32)
        )
    breakpoints = (
        np.unique(np.concatenate(curve_breakpoints))[::-1]
        if curve_breakpoints
        else np.empty(0, dtype=np.float32)
    )
    recall_survival_curve = [
        {
            "threshold_logit": float(value),
            "gt_max_recalled": int(np.count_nonzero(gt_max >= value)),
            "gt_max_recall": float(np.mean(gt_max >= value))
            if gt_max.size
            else 0.0,
            "false_peaks_surviving": int(np.count_nonzero(false_peaks >= value)),
            "false_peak_survival": float(np.mean(false_peaks >= value))
            if false_peaks.size
            else 0.0,
        }
        for value in breakpoints.tolist()
    ]

    pairwise: dict[str, Any]
    if gt_max.size and false_peaks.size:
        sorted_peaks = np.sort(false_peaks)
        lower = np.searchsorted(sorted_peaks, gt_max, side="left").astype(np.int64)
        upper = np.searchsorted(sorted_peaks, gt_max, side="right").astype(np.int64)
        ties = upper - lower
        total_pairs = int(gt_max.size * false_peaks.size)
        strict_wins = int(np.sum(lower, dtype=np.int64))
        tied_pairs = int(np.sum(ties, dtype=np.int64))
        strict_losses = int(total_pairs - strict_wins - tied_pairs)
        pairwise = {
            "num_pairs": total_pairs,
            "strict_target_win_fraction": float(strict_wins / total_pairs),
            "tie_fraction": float(tied_pairs / total_pairs),
            "strict_target_loss_fraction": float(strict_losses / total_pairs),
            "auc_with_half_credit_for_ties": float(
                (strict_wins + 0.5 * tied_pairs) / total_pairs
            ),
        }
    else:
        pairwise = {
            "num_pairs": 0,
            "strict_target_win_fraction": None,
            "tie_fraction": None,
            "strict_target_loss_fraction": None,
            "auc_with_half_credit_for_ties": None,
        }

    highest_false_peak = float(np.max(false_peaks)) if false_peaks.size else None
    margins = (
        gt_max.astype(np.float64) - highest_false_peak
        if highest_false_peak is not None
        else np.empty(0, dtype=np.float64)
    )
    false_peak_order = np.sort(false_peaks.astype(np.float64))[::-1]
    false_component_order = sorted(
        false_component_records,
        key=lambda row: (-float(row["max_logit"]), str(row["image_id"])),
    )
    saturated_background_peak_count = int(
        np.count_nonzero(false_peak_probabilities == np.float32(1.0))
    )
    gt_saturated_count = int(
        sum(
            float(record["max_target_probability_float32"]) == 1.0
            for record in gt_records
        )
    )
    return {
        "threshold_domain": "raw_logit_float32",
        "operating_threshold_logit": threshold,
        "gt_objects": len(gt_records),
        "gt_object_statistics": gt_records,
        "background_pixels": int(background.size),
        "background_logit_quantiles": _quantiles(
            background, [0.99, 0.999, 0.9999]
        ),
        "false_peaks": int(false_peaks.size),
        "false_peak_logit_quantiles": _quantiles(
            false_peaks, [0.5, 0.9, 0.99, 0.999, 0.9999, 1.0]
        ),
        "top_false_peak_logits": [
            float(value) for value in false_peak_order[:top_k]
        ],
        "false_components_at_operating_threshold": len(false_component_records),
        "false_component_max_logit_quantiles": _quantiles(
            false_component_max, [0.5, 0.9, 0.99, 1.0]
        ),
        "top_false_components": false_component_order[:top_k],
        "gt_max_recall_false_peak_survival_curve": recall_survival_curve,
        "curve_breakpoint_rule": (
            "all_unique_gt_max_logits plus false-peak/background tail quantiles"
        ),
        "target_vs_false_peak_pairwise_ranking": pairwise,
        "gt_max_minus_highest_false_peak_margin_quantiles": _quantiles(
            margins, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]
        ),
        "highest_background_false_peak_logit": highest_false_peak,
        "gt_max_below_highest_background_false_peak": int(
            np.count_nonzero(gt_max < highest_false_peak)
        )
        if highest_false_peak is not None
        else 0,
        "gt_max_equal_highest_background_false_peak": int(
            np.count_nonzero(gt_max == highest_false_peak)
        )
        if highest_false_peak is not None
        else 0,
        "gt_objects_saturated_at_probability_one": gt_saturated_count,
        "background_false_peaks_saturated_at_probability_one": (
            saturated_background_peak_count
        ),
        "gt_objects_and_background_false_peak_both_saturated": (
            gt_saturated_count if saturated_background_peak_count > 0 else 0
        ),
    }


def _metrics_row(
    threshold: float,
    results: Sequence[MatchResult],
    *,
    total_pixels: int,
) -> dict[str, float | int | str]:
    totals = {
        "tp_objects": int(sum(result.num_tp_objects for result in results)),
        "gt_objects": int(sum(result.num_gt for result in results)),
        "fp_components": int(sum(result.num_fp_components for result in results)),
        "fp_pixels": int(sum(result.num_fp_pixels for result in results)),
        "total_pixels": int(total_pixels),
    }
    probability64 = _sigmoid_scalar(threshold)
    return {
        "threshold": float(threshold),
        "threshold_domain": "raw_logit",
        "threshold_logit": float(threshold),
        "threshold_probability_float64": probability64,
        "threshold_probability_float32": float(np.float32(probability64)),
        "pd": float(totals["tp_objects"] / totals["gt_objects"])
        if totals["gt_objects"]
        else 0.0,
        "fa_pixel": float(totals["fp_pixels"] / total_pixels),
        "fa_component_mp": float(
            totals["fp_components"] / (total_pixels / 1_000_000.0)
        ),
        **totals,
    }


def _probability_metrics_row(
    threshold: float,
    results: Sequence[MatchResult],
    *,
    total_pixels: int,
) -> dict[str, float | int | str]:
    """Build one aggregate row for an exact stored-FP32 probability state."""

    totals = {
        "tp_objects": int(sum(result.num_tp_objects for result in results)),
        "gt_objects": int(sum(result.num_gt for result in results)),
        "fp_components": int(sum(result.num_fp_components for result in results)),
        "fp_pixels": int(sum(result.num_fp_pixels for result in results)),
        "total_pixels": int(total_pixels),
    }
    return {
        "threshold": float(threshold),
        "threshold_domain": "probability",
        "threshold_probability_float32": float(np.float32(threshold)),
        "pd": float(totals["tp_objects"] / totals["gt_objects"])
        if totals["gt_objects"]
        else 0.0,
        "fa_pixel": float(totals["fp_pixels"] / total_pixels),
        "fa_component_mp": float(
            totals["fp_components"] / (total_pixels / 1_000_000.0)
        ),
        **totals,
    }


def _sigmoid_scalar(value: float) -> float:
    if value >= 0.0:
        return float(1.0 / (1.0 + np.exp(-value)))
    exp_value = float(np.exp(value))
    return float(exp_value / (1.0 + exp_value))


def _provably_infeasible_pixel_cutoff(
    samples: Sequence[RawLogitSample],
    *,
    pixel_budget: float,
    min_component_area: int,
) -> tuple[float | None, dict[str, Any]]:
    """Return a lossless lower-tail cutoff when raw FP pixels are monotone.

    With ``min_component_area == 1``, every predicted background pixel is
    counted by the pixel false-alarm metric.  Once a tied background-logit group
    makes that budget fail, its threshold and every lower state are impossible
    and need not undergo connected-component evaluation.
    """

    total_pixels = int(sum(sample.mask.size for sample in samples))
    if min_component_area != 1:
        return None, {
            "applied": False,
            "reason": "min_component_area_not_one",
            "provably_infeasible_at_or_below_logit": None,
        }
    background = np.concatenate(
        [sample.logits[~sample.mask].reshape(-1) for sample in samples]
    )
    if background.size == 0:
        return None, {
            "applied": False,
            "reason": "no_background_pixels",
            "provably_infeasible_at_or_below_logit": None,
        }
    values, counts = np.unique(background, return_counts=True)
    feasible_count = 0
    cutoff: float | None = None
    for value, count in zip(values[::-1], counts[::-1]):
        proposed = feasible_count + int(count)
        if proposed / total_pixels <= pixel_budget:
            feasible_count = proposed
        else:
            cutoff = float(value)
            break
    return cutoff, {
        "applied": cutoff is not None,
        "reason": (
            "pixel_budget_proves_all_lower_states_infeasible"
            if cutoff is not None
            else "pixel_budget_allows_every_background_pixel"
        ),
        "provably_infeasible_at_or_below_logit": cutoff,
        "background_pixels_in_lowest_retained_state": int(feasible_count),
        "background_pixels_total": int(background.size),
    }


def _provably_infeasible_probability_cutoff(
    samples: Sequence[RawLogitSample],
    *,
    pixel_budget: float,
    min_component_area: int,
) -> tuple[float | None, dict[str, Any]]:
    """Probability-domain counterpart of the lossless raw-logit pruning proof."""

    total_pixels = int(sum(sample.mask.size for sample in samples))
    if min_component_area != 1:
        return None, {
            "applied": False,
            "reason": "min_component_area_not_one",
            "provably_infeasible_at_or_below_probability": None,
        }
    background = np.concatenate(
        [sample.probability[~sample.mask].reshape(-1) for sample in samples]
    )
    if background.size == 0:
        return None, {
            "applied": False,
            "reason": "no_background_pixels",
            "provably_infeasible_at_or_below_probability": None,
        }
    values, counts = np.unique(background, return_counts=True)
    feasible_count = 0
    cutoff: float | None = None
    for value, count in zip(values[::-1], counts[::-1]):
        proposed = feasible_count + int(count)
        if proposed / total_pixels <= pixel_budget:
            feasible_count = proposed
        else:
            cutoff = float(value)
            break
    return cutoff, {
        "applied": cutoff is not None,
        "reason": (
            "pixel_budget_proves_all_lower_states_infeasible"
            if cutoff is not None
            else "pixel_budget_allows_every_background_pixel"
        ),
        "provably_infeasible_at_or_below_probability": cutoff,
        "background_pixels_in_lowest_retained_state": int(feasible_count),
        "background_pixels_total": int(background.size),
    }


def select_exact_global_oracle(
    samples: Sequence[RawLogitSample],
    *,
    pixel_budget: float,
    component_budget: float,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
) -> dict[str, Any]:
    """Select max-Pd over every lossless global raw-logit prediction state."""

    pixel_budget = _validate_budget(pixel_budget, name="pixel_budget")
    component_budget = _validate_budget(
        component_budget, name="component_budget"
    )
    if not samples:
        raise ValueError("at least one raw-logit sample is required")
    validated = [
        _validated_sample(sample, index) for index, sample in enumerate(samples)
    ]
    total_pixels = int(sum(sample.mask.size for sample in validated))
    all_logits = np.concatenate([sample.logits.reshape(-1) for sample in validated])
    unique_logit_count = int(np.unique(all_logits).size)
    max_logit = float(np.max(all_logits))
    reject_all_threshold = float(np.nextafter(np.float64(max_logit), np.inf))
    cutoff, pruning = _provably_infeasible_pixel_cutoff(
        validated,
        pixel_budget=pixel_budget,
        min_component_area=min_component_area,
    )

    events: dict[float, list[int]] = {}
    for sample_index, sample in enumerate(validated):
        local_values = np.unique(sample.logits)
        if cutoff is not None:
            local_values = local_values[local_values > cutoff]
        for value in local_values.tolist():
            events.setdefault(float(value), []).append(sample_index)

    empty = np.zeros(validated[0].mask.shape, dtype=bool)
    cached: list[MatchResult] = []
    for sample in validated:
        if empty.shape == sample.mask.shape:
            prediction = empty
        else:
            prediction = np.zeros(sample.mask.shape, dtype=bool)
        cached.append(
            match_components(
                prediction,
                sample.mask,
                rule=matching_rule,
                centroid_distance=centroid_distance,
                connectivity=connectivity,
                min_component_area=min_component_area,
            )
        )

    selected: dict[str, Any] | None = None

    def consider(row: dict[str, Any]) -> None:
        nonlocal selected
        if not is_budget_satisfied(
            row,
            pixel_budget=pixel_budget,
            component_budget=component_budget,
        ):
            return
        if selected is None or (
            int(row["tp_objects"]) > int(selected["tp_objects"])
            or (
                int(row["tp_objects"]) == int(selected["tp_objects"])
                and float(row["threshold_logit"])
                < float(selected["threshold_logit"])
            )
        ):
            selected = dict(row)

    consider(
        _metrics_row(
            reject_all_threshold,
            cached,
            total_pixels=total_pixels,
        )
    )
    for threshold in sorted(events, reverse=True):
        for sample_index in events[threshold]:
            sample = validated[sample_index]
            cached[sample_index] = match_components(
                sample.logits >= threshold,
                sample.mask,
                rule=matching_rule,
                centroid_distance=centroid_distance,
                connectivity=connectivity,
                min_component_area=min_component_area,
            )
        consider(_metrics_row(threshold, cached, total_pixels=total_pixels))

    if selected is None:  # reject-all has zero risk and positive budgets
        raise AssertionError("exact Oracle failed to retain its reject-all state")
    evaluated_unique = len(events)
    return {
        "found": True,
        "strategy": "maximize_global_pd_then_lowest_raw_logit_threshold",
        "tie_break": "lowest_raw_logit_threshold",
        "operating_point": selected,
        "search": {
            "exact": True,
            "threshold_domain": "raw_logit_float32",
            "prediction_rule": "raw_logit >= threshold",
            "breakpoint_rule": (
                "all_unique_logits_plus_reject_all; only pixel-budget-proven-"
                "infeasible lower states may be omitted"
            ),
            "num_unique_logits_total": unique_logit_count,
            "num_unique_logits_evaluated": evaluated_unique,
            "num_unique_logits_proven_infeasible": int(
                unique_logit_count - evaluated_unique
            ),
            "num_prediction_states_evaluated": int(evaluated_unique + 1),
            "reject_all_threshold_logit": reject_all_threshold,
            "lossless_pruning": pruning,
        },
    }


def select_exact_probability_global_oracle(
    samples: Sequence[RawLogitSample],
    *,
    pixel_budget: float,
    component_budget: float,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
) -> dict[str, Any]:
    """Select max-Pd over every stored float32-probability prediction state.

    Every unique probability value is a breakpoint under the repository's
    ``probability >= threshold`` rule.  The only omitted lower states are those
    proven infeasible by monotone false-positive pixels when
    ``min_component_area == 1``.  A float32 ``nextafter(1, +inf)`` state is
    always evaluated so that reject-all is represented even when probability
    one occurs in the artifact.
    """

    pixel_budget = _validate_budget(pixel_budget, name="pixel_budget")
    component_budget = _validate_budget(
        component_budget, name="component_budget"
    )
    if not samples:
        raise ValueError("at least one raw-logit sample is required")
    validated = [
        _validated_sample(sample, index) for index, sample in enumerate(samples)
    ]
    total_pixels = int(sum(sample.mask.size for sample in validated))
    all_probabilities = np.concatenate(
        [sample.probability.reshape(-1) for sample in validated]
    )
    all_logits = np.concatenate([sample.logits.reshape(-1) for sample in validated])
    unique_logit_count = int(np.unique(all_logits).size)
    unique_probabilities, probability_counts = np.unique(
        all_probabilities, return_counts=True
    )
    unique_probability_count = int(unique_probabilities.size)
    reject_all_threshold = float(
        np.nextafter(
            np.float32(1.0),
            np.float32(np.inf),
            dtype=np.float32,
        )
    )
    cutoff, pruning = _provably_infeasible_probability_cutoff(
        validated,
        pixel_budget=pixel_budget,
        min_component_area=min_component_area,
    )

    events: dict[float, list[int]] = {}
    for sample_index, sample in enumerate(validated):
        local_values = np.unique(sample.probability)
        if cutoff is not None:
            local_values = local_values[local_values > cutoff]
        for value in local_values.tolist():
            events.setdefault(float(value), []).append(sample_index)

    cached: list[MatchResult] = []
    for sample in validated:
        cached.append(
            match_components(
                np.zeros(sample.mask.shape, dtype=bool),
                sample.mask,
                rule=matching_rule,
                centroid_distance=centroid_distance,
                connectivity=connectivity,
                min_component_area=min_component_area,
            )
        )

    selected: dict[str, Any] | None = None

    def consider(row: dict[str, Any]) -> None:
        nonlocal selected
        if not is_budget_satisfied(
            row,
            pixel_budget=pixel_budget,
            component_budget=component_budget,
        ):
            return
        if selected is None or (
            int(row["tp_objects"]) > int(selected["tp_objects"])
            or (
                int(row["tp_objects"]) == int(selected["tp_objects"])
                and float(row["threshold_probability_float32"])
                < float(selected["threshold_probability_float32"])
            )
        ):
            selected = dict(row)

    consider(
        _probability_metrics_row(
            reject_all_threshold,
            cached,
            total_pixels=total_pixels,
        )
    )
    for threshold in sorted(events, reverse=True):
        for sample_index in events[threshold]:
            sample = validated[sample_index]
            cached[sample_index] = match_components(
                sample.probability >= np.float32(threshold),
                sample.mask,
                rule=matching_rule,
                centroid_distance=centroid_distance,
                connectivity=connectivity,
                min_component_area=min_component_area,
            )
        consider(
            _probability_metrics_row(
                threshold,
                cached,
                total_pixels=total_pixels,
            )
        )

    if selected is None:  # reject-all has zero risk and positive budgets
        raise AssertionError(
            "exact probability Oracle failed to retain its reject-all state"
        )
    evaluated_unique = len(events)
    tied = probability_counts > 1
    exact_one = all_probabilities == np.float32(1.0)
    exact_one_logits = all_logits[exact_one]
    return {
        "found": True,
        "strategy": "maximize_global_pd_then_lowest_float32_probability_threshold",
        "tie_break": "lowest_float32_probability_threshold",
        "operating_point": selected,
        "search": {
            "exact": True,
            "threshold_domain": "sigmoid_probability_float32",
            "prediction_rule": "stored float32 probability >= threshold",
            "breakpoint_rule": (
                "all_unique_stored_float32_probabilities_plus_float32_"
                "nextafter(1,+inf)_reject_all; only pixel-budget-proven-"
                "infeasible lower states may be omitted"
            ),
            "num_unique_probabilities_total": unique_probability_count,
            "num_unique_probabilities_evaluated": evaluated_unique,
            "num_unique_probabilities_proven_infeasible": int(
                unique_probability_count - evaluated_unique
            ),
            "num_prediction_states_evaluated": int(evaluated_unique + 1),
            "reject_all_threshold_probability_float32": reject_all_threshold,
            "lossless_pruning": pruning,
            "tie_audit": {
                "num_probability_tie_groups": int(np.count_nonzero(tied)),
                "num_pixels_in_probability_tie_groups": int(
                    probability_counts[tied].sum(dtype=np.int64)
                ),
                "largest_probability_tie_group_pixels": int(
                    probability_counts.max(initial=0)
                ),
                "num_unique_logits_total": unique_logit_count,
                "num_unique_probability_values_total": unique_probability_count,
                "unique_state_collapse_count": int(
                    unique_logit_count - unique_probability_count
                ),
            },
            "saturation_audit": {
                "exact_one_pixels": int(np.count_nonzero(exact_one)),
                "exact_one_unique_logits": int(np.unique(exact_one_logits).size),
                "exact_one_logit_min": float(np.min(exact_one_logits))
                if exact_one_logits.size
                else None,
                "exact_one_logit_max": float(np.max(exact_one_logits))
                if exact_one_logits.size
                else None,
            },
        },
    }


def _false_alarm_concentration(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, Any]:
    values = np.asarray([int(row[key]) for row in rows], dtype=np.int64)
    order = np.argsort(-values, kind="stable")
    total = int(values.sum(dtype=np.int64))
    specifications = {
        "top_1": 1,
        "top_5": 5,
        "top_10": 10,
        "top_1_percent": max(1, int(math.ceil(len(rows) * 0.01))),
        "top_5_percent": max(1, int(math.ceil(len(rows) * 0.05))),
    }
    output: dict[str, Any] = {"total": total}
    for name, requested in specifications.items():
        count = min(int(requested), len(rows))
        selected_indices = order[:count]
        subtotal = int(values[selected_indices].sum(dtype=np.int64))
        output[name] = {
            "num_images": count,
            "value": subtotal,
            "fraction": float(subtotal / total) if total else 0.0,
            "image_ids": [str(rows[index]["image_id"]) for index in selected_indices],
        }
    return output


def _selected_threshold_image_diagnostics(
    samples: Sequence[RawLogitSample],
    *,
    threshold: float,
    threshold_domain: str,
    matching_rule: str,
    centroid_distance: float,
    connectivity: int,
    min_component_area: int,
) -> dict[str, Any]:
    if threshold_domain not in {"raw_logit", "probability"}:
        raise ValueError("threshold_domain must be raw_logit or probability")
    rows: list[dict[str, Any]] = []
    for sample in samples:
        prediction = (
            sample.logits.astype(np.float64) >= float(threshold)
            if threshold_domain == "raw_logit"
            else sample.probability >= np.float32(threshold)
        )
        matched = match_components(
            prediction,
            sample.mask,
            rule=matching_rule,
            centroid_distance=centroid_distance,
            connectivity=connectivity,
            min_component_area=min_component_area,
        )
        rows.append(
            {
                "image_id": sample.image_id,
                "total_pixels": int(sample.mask.size),
                "tp_objects": int(matched.num_tp_objects),
                "gt_objects": int(matched.num_gt),
                "fp_pixels": int(matched.num_fp_pixels),
                "fp_components": int(matched.num_fp_components),
            }
        )
    return {
        "threshold_domain": threshold_domain,
        "selected_threshold": float(threshold),
        "per_image_raw_counts": rows,
        "aggregate_raw_counts": {
            "tp_objects": int(sum(row["tp_objects"] for row in rows)),
            "gt_objects": int(sum(row["gt_objects"] for row in rows)),
            "fp_pixels": int(sum(row["fp_pixels"] for row in rows)),
            "fp_components": int(sum(row["fp_components"] for row in rows)),
            "total_pixels": int(sum(row["total_pixels"] for row in rows)),
        },
        "false_alarm_concentration": {
            "fp_pixels": _false_alarm_concentration(rows, "fp_pixels"),
            "fp_components": _false_alarm_concentration(rows, "fp_components"),
        },
    }


def _compare_selected_prediction_states(
    samples: Sequence[RawLogitSample],
    *,
    raw_logit_threshold: float,
    probability_threshold: float,
) -> dict[str, Any]:
    differing_pixels = 0
    differing_images: list[str] = []
    for sample in samples:
        raw_prediction = sample.logits.astype(np.float64) >= float(
            raw_logit_threshold
        )
        probability_prediction = sample.probability >= np.float32(
            probability_threshold
        )
        count = int(np.count_nonzero(raw_prediction != probability_prediction))
        differing_pixels += count
        if count:
            differing_images.append(sample.image_id)
    return {
        "selected_prediction_states_equal": differing_pixels == 0,
        "selected_prediction_state_differing_pixels": differing_pixels,
        "selected_prediction_state_differing_images": len(differing_images),
        "selected_prediction_state_differing_image_ids": differing_images,
    }


def build_raw_logit_oracle_payload(
    score_dir: str | Path,
    *,
    pixel_budget: float,
    component_budget: float,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
    reference_probability_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Audit one score artifact and build the exact diagnostic Oracle JSON."""

    samples, manifest, integrity, contract = load_formal_raw_logit_directory(
        score_dir
    )
    selection = select_exact_global_oracle(
        samples,
        pixel_budget=pixel_budget,
        component_budget=component_budget,
        matching_rule=matching_rule,
        centroid_distance=centroid_distance,
        connectivity=connectivity,
        min_component_area=min_component_area,
    )
    probability_selection = select_exact_probability_global_oracle(
        samples,
        pixel_budget=pixel_budget,
        component_budget=component_budget,
        matching_rule=matching_rule,
        centroid_distance=centroid_distance,
        connectivity=connectivity,
        min_component_area=min_component_area,
    )
    saturation = audit_exact_one_saturation(
        samples,
        matching_rule=matching_rule,
        centroid_distance=centroid_distance,
        connectivity=connectivity,
        min_component_area=min_component_area,
    )
    operating_point = selection["operating_point"]
    ranking = audit_target_background_ranking(
        samples,
        operating_threshold_logit=float(operating_point["threshold_logit"]),
        matching_rule=matching_rule,
        centroid_distance=centroid_distance,
        connectivity=connectivity,
        min_component_area=min_component_area,
    )
    probability_reference = (
        compare_probability_with_reference(score_dir, reference_probability_dir)
        if reference_probability_dir is not None
        else {"provided": False, "bitwise_equal": None}
    )
    probability_operating_point = probability_selection["operating_point"]
    raw_image_diagnostics = _selected_threshold_image_diagnostics(
        samples,
        threshold=float(operating_point["threshold_logit"]),
        threshold_domain="raw_logit",
        matching_rule=matching_rule,
        centroid_distance=centroid_distance,
        connectivity=connectivity,
        min_component_area=min_component_area,
    )
    probability_image_diagnostics = _selected_threshold_image_diagnostics(
        samples,
        threshold=float(
            probability_operating_point["threshold_probability_float32"]
        ),
        threshold_domain="probability",
        matching_rule=matching_rule,
        centroid_distance=centroid_distance,
        connectivity=connectivity,
        min_component_area=min_component_area,
    )
    for domain, point, diagnostics in (
        ("raw_logit_float32", operating_point, raw_image_diagnostics),
        (
            "sigmoid_probability_float32",
            probability_operating_point,
            probability_image_diagnostics,
        ),
    ):
        aggregate = diagnostics["aggregate_raw_counts"]
        for field in (
            "tp_objects",
            "gt_objects",
            "fp_pixels",
            "fp_components",
            "total_pixels",
        ):
            if int(aggregate[field]) != int(point[field]):
                raise AssertionError(
                    f"{domain} selected-point per-image {field} does not "
                    "match the exact Oracle aggregate"
                )
    selected_state_comparison = _compare_selected_prediction_states(
        samples,
        raw_logit_threshold=float(operating_point["threshold_logit"]),
        probability_threshold=float(
            probability_operating_point["threshold_probability_float32"]
        ),
    )
    domain_comparison = {
        "raw_logit_selected_threshold": float(
            operating_point["threshold_logit"]
        ),
        "probability_selected_threshold_float32": float(
            probability_operating_point["threshold_probability_float32"]
        ),
        "raw_logit_pd": float(operating_point["pd"]),
        "probability_pd": float(probability_operating_point["pd"]),
        "pd_difference_raw_logit_minus_probability": float(
            float(operating_point["pd"])
            - float(probability_operating_point["pd"])
        ),
        "raw_logit_tp_objects": int(operating_point["tp_objects"]),
        "probability_tp_objects": int(probability_operating_point["tp_objects"]),
        "raw_logit_fp_pixels": int(operating_point["fp_pixels"]),
        "probability_fp_pixels": int(probability_operating_point["fp_pixels"]),
        "raw_logit_fp_components": int(operating_point["fp_components"]),
        "probability_fp_components": int(
            probability_operating_point["fp_components"]
        ),
        "control_semantics": {
            "exact_probability_vs_fixed_probability_grid": (
                "threshold_grid_resolution control"
            ),
            "exact_raw_logit_vs_exact_probability": (
                "stored float32 sigmoid quantization/saturation control"
            ),
        },
        **selected_state_comparison,
    }
    return {
        "schema_version": RAW_LOGIT_ORACLE_SCHEMA_VERSION,
        "artifact_type": RAW_LOGIT_ORACLE_ARTIFACT_TYPE,
        "found": selection["found"],
        "pixel_budget": float(pixel_budget),
        "component_budget": float(component_budget),
        "strategy": selection["strategy"],
        "tie_break": selection["tie_break"],
        "operating_point": operating_point,
        "search": selection["search"],
        "exact_oracles": {
            "raw_logit_float32": selection,
            "sigmoid_probability_float32": probability_selection,
        },
        "exact_domain_comparison": domain_comparison,
        "selected_operating_point_image_diagnostics": {
            "raw_logit_float32": raw_image_diagnostics,
            "sigmoid_probability_float32": probability_image_diagnostics,
        },
        "exact_one_saturation_audit": saturation,
        "target_background_ranking_audit": ranking,
        "probability_reference_equivalence": probability_reference,
        "matching_protocol": {
            "matching_rule": matching_rule,
            "centroid_distance": float(centroid_distance),
            "connectivity": int(connectivity),
            "min_component_area": int(min_component_area),
        },
        "input_artifact_formal_contract_verified": True,
        "score_dir": str(Path(score_dir).expanduser().resolve()),
        "score_manifest_schema_version": int(manifest["schema_version"]),
        "score_manifest_sha256": contract["score_manifest_sha256"],
        "score_records_sha256": contract["score_records_sha256"],
        "score_ordered_image_ids_sha256": contract[
            "score_ordered_image_ids_sha256"
        ],
        "raw_logit_stream_sha256": raw_logit_stream_sha256(samples),
        "checkpoint_sha256": contract["detector_weight_sha256"],
        "checkpoint_selection_rule": contract["checkpoint_selection_rule"],
        "model_backend": contract["model_backend"],
        "target_dataset": contract["target_dataset"],
        "source_datasets": contract["source_datasets"],
        "requested_split": contract["requested_split"],
        "split_role": contract["split_role"],
        "split_file_sha256": contract["split_file_sha256"],
        "split_ordered_ids_sha256": contract["split_ordered_ids_sha256"],
        "score_representation": contract["score_representation"],
        "probability_dtype": contract["probability_dtype"],
        "logit_dtype": contract["logit_dtype"],
        "probability_transform": contract["probability_transform"],
        "probability_clipping": contract["probability_clipping"],
        "inference_autocast_enabled": contract["inference_autocast_enabled"],
        "test_labels_used": True,
        "oracle_only": True,
        "formal_protocol_eligible": False,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pixel-budget", type=float, required=True)
    parser.add_argument("--component-budget", type=float, required=True)
    parser.add_argument(
        "--reference-probability-dir",
        help=(
            "Existing formal v3 probability-only export from the same checkpoint; "
            "when supplied, every probability pixel must be bitwise identical"
        ),
    )
    parser.add_argument(
        "--matching-rule", choices=("overlap", "centroid"), default="overlap"
    )
    parser.add_argument("--centroid-distance", type=float, default=3.0)
    parser.add_argument("--connectivity", type=int, default=2)
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument(
        "--oracle-diagnostic",
        action="store_true",
        help=(
            "Required acknowledgement: this exact search reads held-out masks "
            "and cannot select a formal/deployment threshold"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if not args.oracle_diagnostic:
        raise ValueError(
            "raw_logit_oracle reads held-out masks; pass --oracle-diagnostic "
            "to acknowledge diagnostic-only use"
        )
    payload = build_raw_logit_oracle_payload(
        args.score_dir,
        pixel_budget=args.pixel_budget,
        component_budget=args.component_budget,
        matching_rule=args.matching_rule,
        centroid_distance=args.centroid_distance,
        connectivity=args.connectivity,
        min_component_area=args.min_component_area,
        reference_probability_dir=args.reference_probability_dir,
    )
    write_json_atomic(args.output, payload)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI tests
    raise SystemExit(main())


__all__ = [
    "RAW_LOGIT_ORACLE_ARTIFACT_TYPE",
    "RAW_LOGIT_ORACLE_SCHEMA_VERSION",
    "RawLogitSample",
    "audit_exact_one_saturation",
    "audit_target_background_ranking",
    "build_raw_logit_oracle_payload",
    "compare_probability_with_reference",
    "load_formal_raw_logit_directory",
    "raw_logit_stream_sha256",
    "select_exact_global_oracle",
    "select_exact_probability_global_oracle",
    "validate_formal_raw_logit_manifest",
]
