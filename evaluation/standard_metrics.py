"""Formal native-resolution metrics at one frozen probability threshold.

This command is intentionally stricter than the diagnostic threshold sweep.  It
accepts only an integrity-verified v3 score-map export for an official native
test split with embedded masks and a non-diagnostic detector checkpoint.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rc_irstd.utils.io import atomic_write_json

from .artifact_integrity import file_sha256, verify_score_map_directory
from .component_matching import match_components


STANDARD_METRICS_SCHEMA_VERSION = "rc-v2-standard-native-fixed-threshold-v1"
DEFAULT_NEAR_CONSTANT_STD_THRESHOLD = 1e-6
FORMAL_MODEL_BACKENDS = frozenset({"canonical", "rc_mshnet"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_threshold(value: float) -> float:
    threshold = float(value)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and lie in [0, 1]")
    return threshold


def _validate_near_constant_threshold(value: float) -> float:
    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            "near_constant_std_threshold must be finite and non-negative"
        )
    return threshold


def _domain_key(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty domain name")
    key = "".join(character for character in value.casefold() if character.isalnum())
    if key.endswith("sirst") and len(key) > len("sirst"):
        key = key[: -len("sirst")]
    if not key:
        raise ValueError(f"{field} must contain an alphanumeric domain name")
    return key


def _require_formal_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed unless the score export declares the formal test protocol."""

    required_values = {
        "labels_loaded": True,
        "spatial_mode": "native",
        "split_role": "test",
        "split_authority_verified": True,
        "checkpoint_diagnostic_only": False,
        "checkpoint_selection_rule": "fixed_last",
        "score_type": "sigmoid_probability",
    }
    for field, expected in required_values.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"Formal fixed-threshold evaluation requires {field}={expected!r}"
            )
    model_backend = manifest.get("model_backend")
    if model_backend not in FORMAL_MODEL_BACKENDS:
        allowed = ", ".join(sorted(FORMAL_MODEL_BACKENDS))
        raise ValueError(
            "Formal fixed-threshold evaluation requires model_backend in "
            f"{{{allowed}}}"
        )
    non_strict = manifest.get("non_strict_state_loading", False)
    if not isinstance(non_strict, bool):
        raise ValueError("non_strict_state_loading must be boolean when present")
    if non_strict:
        raise ValueError(
            "Formal fixed-threshold evaluation forbids non-strict checkpoint loading"
        )
    detector_sha = manifest.get("weight_sha256")
    if not isinstance(detector_sha, str) or not _SHA256_RE.fullmatch(detector_sha):
        raise ValueError(
            "Formal fixed-threshold evaluation requires detector weight_sha256"
        )
    for field in ("split_file", "split_file_sha256", "split_ordered_ids_sha256"):
        value = manifest.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(
                "Formal fixed-threshold evaluation requires complete split-file "
                f"provenance; missing {field}"
            )
    target_dataset = manifest.get("target_dataset")
    target_key = _domain_key(target_dataset, "target_dataset")
    source_datasets = manifest.get("source_datasets")
    if not isinstance(source_datasets, list) or not source_datasets:
        raise ValueError(
            "Formal fixed-threshold evaluation requires non-empty source_datasets"
        )
    source_keys = [
        _domain_key(value, "source_datasets entry") for value in source_datasets
    ]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("source_datasets must identify unique domains")
    if target_key in set(source_keys):
        raise ValueError(
            "Formal fixed-threshold evaluation requires target_dataset to be "
            "excluded from detector source_datasets"
        )


def _validate_pair(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    sample_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(probability)
    target = np.asarray(mask)
    if score.ndim == 3 and score.shape[0] == 1:
        score = score[0]
    if target.ndim == 3 and target.shape[0] == 1:
        target = target[0]
    if score.ndim != 2 or target.ndim != 2:
        raise ValueError(f"score/mask pair {sample_index} must contain 2-D arrays")
    if score.shape != target.shape:
        raise ValueError(
            f"score/mask pair {sample_index} has mismatched shapes: "
            f"{score.shape} vs {target.shape}"
        )
    if score.size == 0:
        raise ValueError(f"score/mask pair {sample_index} must not be empty")
    if not np.issubdtype(score.dtype, np.number) or not np.isfinite(score).all():
        raise ValueError(f"probability map {sample_index} must be finite and numeric")
    if np.any((score < 0.0) | (score > 1.0)):
        raise ValueError(f"probability map {sample_index} contains values outside [0, 1]")
    if not np.issubdtype(target.dtype, np.number) and target.dtype != np.bool_:
        raise TypeError(f"mask {sample_index} must be numeric or boolean")
    if not np.isfinite(target).all():
        raise ValueError(f"mask {sample_index} contains NaN or infinity")
    return (
        np.ascontiguousarray(score, dtype=np.float64),
        np.ascontiguousarray(target > 0, dtype=bool),
    )


def compute_standard_metrics(
    probabilities: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    threshold: float,
    *,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
    near_constant_std_threshold: float = DEFAULT_NEAR_CONSTANT_STD_THRESHOLD,
) -> dict[str, Any]:
    """Compute publication metrics using the single rule ``prob >= threshold``.

    Pixel metrics use the raw thresholded mask.  ``min_component_area`` affects
    predicted-component matching only; ground-truth objects are never filtered.
    Per-image IoU assigns 0.0 to an empty-prediction/empty-target pair, matching
    the BasicIRSTD/RDIAN ``SamplewiseSigmoidMetric`` convention and preventing
    negative-only samples from increasing nIoU.
    """

    threshold = _validate_threshold(threshold)
    near_constant_std_threshold = _validate_near_constant_threshold(
        near_constant_std_threshold
    )
    if len(probabilities) != len(masks):
        raise ValueError("probabilities and masks must have the same length")
    if not probabilities:
        raise ValueError("at least one score/mask pair is required")

    pixel_tp = 0
    pixel_fp = 0
    pixel_fn = 0
    pixel_tn = 0
    object_tp = 0
    object_gt = 0
    false_positive_components = 0
    per_image_ious: list[float] = []
    empty_empty_images = 0
    image_score_means: list[float] = []
    within_image_score_stds: list[float] = []
    global_score_min = float("inf")
    global_score_max = float("-inf")
    exact_constant_images = 0

    for sample_index, (raw_probability, raw_mask) in enumerate(
        zip(probabilities, masks)
    ):
        probability, target = _validate_pair(
            raw_probability,
            raw_mask,
            sample_index=sample_index,
        )
        prediction = probability >= threshold
        image_mean = float(np.mean(probability))
        within_image_std = float(np.std(probability, ddof=0))
        image_min = float(np.min(probability))
        image_max = float(np.max(probability))
        image_score_means.append(image_mean)
        within_image_score_stds.append(within_image_std)
        global_score_min = min(global_score_min, image_min)
        global_score_max = max(global_score_max, image_max)
        exact_constant_images += int(image_max == image_min)
        tp = int(np.count_nonzero(prediction & target))
        fp = int(np.count_nonzero(prediction & ~target))
        fn = int(np.count_nonzero(~prediction & target))
        tn = int(np.count_nonzero(~prediction & ~target))
        union = tp + fp + fn
        if union == 0:
            per_image_ious.append(0.0)
            empty_empty_images += 1
        else:
            per_image_ious.append(float(tp / union))
        pixel_tp += tp
        pixel_fp += fp
        pixel_fn += fn
        pixel_tn += tn

        match = match_components(
            prediction,
            target,
            rule=matching_rule,
            centroid_distance=centroid_distance,
            connectivity=connectivity,
            min_component_area=min_component_area,
        )
        object_tp += int(match.num_tp_objects)
        object_gt += int(match.num_gt)
        false_positive_components += int(match.num_fp_components)

    total_pixels = pixel_tp + pixel_fp + pixel_fn + pixel_tn
    global_union = pixel_tp + pixel_fp + pixel_fn
    pixel_iou = float(pixel_tp / global_union) if global_union else 0.0
    precision_denominator = pixel_tp + pixel_fp
    recall_denominator = pixel_tp + pixel_fn
    f1_denominator = 2 * pixel_tp + pixel_fp + pixel_fn
    pixel_precision = (
        float(pixel_tp / precision_denominator) if precision_denominator else 0.0
    )
    pixel_recall = float(pixel_tp / recall_denominator) if recall_denominator else 0.0
    pixel_f1 = float(2 * pixel_tp / f1_denominator) if f1_denominator else 0.0
    object_pd = float(object_tp / object_gt) if object_gt else 0.0
    pixel_fa = float(pixel_fp / total_pixels)
    component_fa_per_megapixel = float(
        false_positive_components / (total_pixels / 1_000_000.0)
    )
    image_means = np.asarray(image_score_means, dtype=np.float64)
    within_image_stds = np.asarray(within_image_score_stds, dtype=np.float64)
    score_summary_values = np.concatenate(
        [
            np.asarray([global_score_min, global_score_max], dtype=np.float64),
            image_means,
            within_image_stds,
        ]
    )
    if not np.isfinite(score_summary_values).all():
        raise ValueError("score statistics contain NaN or infinity")
    near_constant_images = int(
        np.count_nonzero(within_image_stds <= near_constant_std_threshold)
    )

    return {
        "metrics": {
            "pixel_iou": pixel_iou,
            "nIoU": float(np.mean(per_image_ious)),
            "pixel_precision": pixel_precision,
            "pixel_recall": pixel_recall,
            "pixel_f1": pixel_f1,
            "object_pd": object_pd,
            "component_fa_per_megapixel": component_fa_per_megapixel,
            "pixel_fa": pixel_fa,
        },
        "counts": {
            "num_samples": len(probabilities),
            "num_pixels": total_pixels,
            "num_target_pixels": pixel_tp + pixel_fn,
            "num_predicted_positive_pixels": pixel_tp + pixel_fp,
            "pixel_tp": pixel_tp,
            "pixel_fp": pixel_fp,
            "pixel_fn": pixel_fn,
            "pixel_tn": pixel_tn,
            "num_gt_objects": object_gt,
            "num_tp_objects": object_tp,
            "num_fn_objects": object_gt - object_tp,
            "num_fp_components": false_positive_components,
            "num_predicted_components": object_tp + false_positive_components,
            "num_empty_empty_images": empty_empty_images,
        },
        "score_statistics": {
            "global_min": global_score_min,
            "global_max": global_score_max,
            "image_mean_std": float(np.std(image_means, ddof=0)),
            "min_within_image_std": float(np.min(within_image_stds)),
            "median_within_image_std": float(np.median(within_image_stds)),
            "exact_constant_images": exact_constant_images,
            "near_constant_images": near_constant_images,
            "exact_constant_fraction": float(exact_constant_images / len(probabilities)),
            "near_constant_fraction": float(near_constant_images / len(probabilities)),
            "near_constant_std_threshold": near_constant_std_threshold,
        },
    }


def evaluate_standard_score_directory(
    score_dir: str | Path,
    threshold: float,
    *,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
    near_constant_std_threshold: float = DEFAULT_NEAR_CONSTANT_STD_THRESHOLD,
) -> dict[str, Any]:
    """Validate one formal v3 score directory and compute fixed-threshold metrics."""

    threshold = _validate_threshold(threshold)
    root = Path(score_dir).expanduser().resolve()
    manifest, paths, integrity = verify_score_map_directory(
        root,
        require_integrity=True,
        require_masks=True,
    )
    if not isinstance(manifest, Mapping):  # require_integrity already fails closed
        raise ValueError("Formal score-map input requires a v3 manifest")
    _require_formal_manifest(manifest)

    probabilities: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for sample_index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as payload:
            if "prob" not in payload or "mask" not in payload:
                raise ValueError(f"Formal score map lacks prob/mask arrays: {path}")
            if "spatial_mode" not in payload or str(payload["spatial_mode"].item()) != "native":
                raise ValueError(f"Score-map record {sample_index} is not native resolution")
            if "labels_loaded" not in payload or payload["labels_loaded"].ndim != 0:
                raise ValueError(f"Score-map record {sample_index} lacks labels_loaded evidence")
            if payload["labels_loaded"].dtype.kind != "b" or not bool(
                payload["labels_loaded"].item()
            ):
                raise ValueError(f"Score-map record {sample_index} is label-free")
            probabilities.append(np.asarray(payload["prob"]))
            masks.append(np.asarray(payload["mask"]))

    computed = compute_standard_metrics(
        probabilities,
        masks,
        threshold,
        matching_rule=matching_rule,
        centroid_distance=centroid_distance,
        connectivity=connectivity,
        min_component_area=min_component_area,
        near_constant_std_threshold=near_constant_std_threshold,
    )
    manifest_path = root / "manifest.json"
    protocol = {
        "evaluation_mode": "formal_native_fixed_threshold",
        "model_backend": manifest.get("model_backend"),
        "threshold_rule": "prediction = (probability >= threshold)",
        "threshold": threshold,
        "spatial_mode": "native",
        "pixel_iou_definition": "sum_i intersection_i / sum_i union_i",
        "pixel_iou_empty_global_union_value": 0.0,
        "nIoU_definition": "mean_i(intersection_i / union_i)",
        "nIoU_empty_empty_image_value": 0.0,
        "zero_denominator_convention_source": (
            "BasicIRSTD/RDIAN SamplewiseSigmoidMetric-compatible; return 0"
        ),
        "pixel_precision_zero_denominator_value": 0.0,
        "pixel_recall_zero_denominator_value": 0.0,
        "pixel_f1_zero_denominator_value": 0.0,
        "object_pd_zero_gt_value": 0.0,
        "pixel_fa_unit": "false_positive_pixels_per_evaluated_pixel",
        "component_fa_unit": "false_positive_components_per_megapixel",
        "pixel_metrics_component_filtering": "none",
        "component_matching": {
            "rule": matching_rule,
            "centroid_distance": float(centroid_distance),
            "connectivity": int(connectivity),
            "predicted_min_component_area": int(min_component_area),
            "ground_truth_min_component_area": 1,
            "assignment": "maximum_cardinality_one_to_one",
        },
        "score_collapse_statistics": {
            "global_min_definition": "minimum probability over every evaluated pixel",
            "global_max_definition": "maximum probability over every evaluated pixel",
            "image_mean_std_definition": (
                "population standard deviation (ddof=0) of per-image mean probabilities"
            ),
            "within_image_std_definition": (
                "population standard deviation (ddof=0) over native pixels of one image"
            ),
            "min_within_image_std_definition": (
                "minimum of the per-image population standard deviations"
            ),
            "median_within_image_std_definition": (
                "median of the per-image population standard deviations"
            ),
            "exact_constant_images_definition": (
                "count of images with max(probability) == min(probability) exactly"
            ),
            "near_constant_images_definition": (
                "count of images whose within-image population std is <= "
                "near_constant_std_threshold; includes exact-constant images"
            ),
            "near_constant_std_threshold": float(near_constant_std_threshold),
        },
        "hiou_status": "not_defined_not_reported",
    }
    return {
        "schema_version": STANDARD_METRICS_SCHEMA_VERSION,
        "formal_protocol_eligible": True,
        "hiou_status": "not_defined_not_reported",
        "protocol": protocol,
        **computed,
        "provenance": {
            "score_dir": str(root),
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "manifest_schema_version": manifest.get("schema_version"),
            "records_sha256": manifest.get("records_sha256"),
            "ordered_image_ids_sha256": manifest.get("ordered_image_ids_sha256"),
            "num_manifest_records": manifest.get("num_images"),
            "target_dataset": manifest.get("target_dataset"),
            "requested_split": manifest.get("requested_split"),
            "split_role": manifest.get("split_role"),
            "split_authority_verified": manifest.get("split_authority_verified"),
            "split_file": manifest.get("split_file"),
            "split_file_sha256": manifest.get("split_file_sha256"),
            "split_ordered_ids_sha256": manifest.get("split_ordered_ids_sha256"),
            "detector_weight_sha256": manifest.get("weight_sha256"),
            "checkpoint_selection_rule": manifest.get("checkpoint_selection_rule"),
            "checkpoint_diagnostic_only": manifest.get("checkpoint_diagnostic_only"),
            "model_backend": manifest.get("model_backend"),
            "score_integrity_audit": integrity,
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--matching-rule", choices=("overlap", "centroid"), default="overlap"
    )
    parser.add_argument("--centroid-distance", type=float, default=3.0)
    parser.add_argument("--connectivity", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument(
        "--near-constant-std-threshold",
        type=float,
        default=DEFAULT_NEAR_CONSTANT_STD_THRESHOLD,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = evaluate_standard_score_directory(
        args.score_dir,
        args.threshold,
        matching_rule=args.matching_rule,
        centroid_distance=args.centroid_distance,
        connectivity=args.connectivity,
        min_component_area=args.min_component_area,
        near_constant_std_threshold=args.near_constant_std_threshold,
    )
    atomic_write_json(args.output, result)
    print(Path(args.output))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())


__all__ = [
    "STANDARD_METRICS_SCHEMA_VERSION",
    "DEFAULT_NEAR_CONSTANT_STD_THRESHOLD",
    "FORMAL_MODEL_BACKENDS",
    "build_argument_parser",
    "compute_standard_metrics",
    "evaluate_standard_score_directory",
    "main",
]
