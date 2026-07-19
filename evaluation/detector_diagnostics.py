"""Label-derived detector diagnostics for an integrity-bound formal curve.

This module does not select a deployable threshold.  It decomposes a frozen
target-test Oracle into pixel/component constraints, localises concentrated
false alarms, and measures target/background ordering in probability space.
Every result is therefore explicitly diagnostic and label-derived.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

from .artifact_integrity import verify_score_map_directory
from .component_matching import connected_components, match_components
from .operating_point import select_operating_point
from .source_operating_point import FormalCurve, load_formal_curve
from .threshold_sweep import EMPTY_SET_THRESHOLD, write_json_atomic


DETECTOR_DIAGNOSTICS_SCHEMA_VERSION = (
    "rc-v2-detector-probability-diagnostics-v1"
)


@dataclass(frozen=True)
class BudgetPair:
    name: str
    pixel: float
    component: float


def parse_budget(value: str) -> BudgetPair:
    """Parse ``NAME:PIXEL:COMPONENT`` with positive finite budgets."""

    parts = value.split(":")
    if len(parts) != 3 or not parts[0].strip():
        raise argparse.ArgumentTypeError(
            "budget must use NAME:PIXEL:COMPONENT"
        )
    try:
        pixel = float(parts[1])
        component = float(parts[2])
    except ValueError as error:
        raise argparse.ArgumentTypeError("budget values must be numbers") from error
    if not math.isfinite(pixel) or pixel <= 0.0:
        raise argparse.ArgumentTypeError("pixel budget must be finite and positive")
    if not math.isfinite(component) or component <= 0.0:
        raise argparse.ArgumentTypeError(
            "component budget must be finite and positive"
        )
    return BudgetPair(parts[0].strip(), pixel, component)


def _selected_payload(row: Mapping[str, object] | None) -> dict[str, Any]:
    if row is None:
        return {"found": False, "nonempty": False, "operating_point": None}
    payload = dict(row)
    nonempty = bool(
        float(payload["threshold"]) <= 1.0
        and (
            int(payload["tp_objects"]) > 0
            or int(payload["fp_components"]) > 0
            or int(payload["fp_pixels"]) > 0
        )
    )
    return {"found": True, "nonempty": nonempty, "operating_point": payload}


def decompose_constraints(
    rows: Sequence[Mapping[str, object]],
    budgets: Sequence[BudgetPair],
) -> dict[str, dict[str, Any]]:
    """Return pixel-only, component-only, and joint probability-grid Oracles."""

    if not rows:
        raise ValueError("curve rows must be non-empty")
    if not budgets:
        raise ValueError("at least one budget pair is required")
    output: dict[str, dict[str, Any]] = {}
    for budget in budgets:
        if budget.name in output:
            raise ValueError(f"duplicate budget name: {budget.name}")
        selections = {
            "pixel_only": select_operating_point(
                rows, pixel_budget=budget.pixel, strategy="max_pd"
            ),
            "component_only": select_operating_point(
                rows, component_budget=budget.component, strategy="max_pd"
            ),
            "joint": select_operating_point(
                rows,
                pixel_budget=budget.pixel,
                component_budget=budget.component,
                strategy="max_pd",
            ),
        }
        output[budget.name] = {
            "pixel_budget": float(budget.pixel),
            "component_budget": float(budget.component),
            "oracle": {
                name: _selected_payload(row) for name, row in selections.items()
            },
        }
    return output


def _load_arrays(paths: Sequence[Path]) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    ids: list[str] = []
    probabilities: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            probability = np.asarray(payload["prob"], dtype=np.float32).squeeze()
            mask = np.asarray(payload["mask"]).squeeze() > 0
            image_id = str(np.asarray(payload["image_id"]).item())
        if probability.ndim != 2 or mask.ndim != 2 or probability.shape != mask.shape:
            raise ValueError(f"invalid probability/mask pair: {path}")
        if not np.isfinite(probability).all() or np.any(
            (probability < 0.0) | (probability > 1.0)
        ):
            raise ValueError(f"invalid probability values: {path}")
        ids.append(image_id)
        probabilities.append(np.ascontiguousarray(probability))
        masks.append(np.ascontiguousarray(mask))
    if len(ids) != len(set(ids)):
        raise ValueError("score records contain duplicate image IDs")
    return ids, probabilities, masks


def _per_image_at_threshold(
    image_ids: Sequence[str],
    probabilities: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    threshold: float,
    *,
    matching_rule: str,
    centroid_distance: float,
    connectivity: int,
    min_component_area: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image_id, probability, mask in zip(image_ids, probabilities, masks):
        prediction = probability >= float(threshold)
        prediction_labels, num_predictions = connected_components(
            prediction,
            connectivity=connectivity,
            min_component_area=min_component_area,
        )
        matched = match_components(
            prediction,
            mask,
            rule=matching_rule,
            centroid_distance=centroid_distance,
            connectivity=connectivity,
            min_component_area=min_component_area,
        )
        matched_prediction_labels = {
            int(prediction_label)
            for prediction_label, _ in matched.matched_pairs
        }
        false_labels = [
            label
            for label in range(1, num_predictions + 1)
            if label not in matched_prediction_labels
        ]
        false_component_maxima = [
            float(np.max(probability[prediction_labels == label]))
            for label in false_labels
        ]
        false_peak_scores = _background_false_peaks(
            probability,
            mask,
            min_score=float(threshold),
        )
        rows.append(
            {
                "image_id": str(image_id),
                "height": int(probability.shape[0]),
                "width": int(probability.shape[1]),
                "total_pixels": int(probability.size),
                "gt_objects": int(matched.num_gt),
                "matched_targets": int(matched.num_tp_objects),
                "false_pixels": int(matched.num_fp_pixels),
                "false_components": int(matched.num_fp_components),
                # A false component can contain more than one local maximum;
                # keep the two counts separate for the concentration audit.
                "false_peaks": int(false_peak_scores.size),
                "false_peak_max_score": (
                    float(np.max(false_peak_scores))
                    if false_peak_scores.size
                    else None
                ),
                "false_component_max_score": (
                    float(max(false_component_maxima))
                    if false_component_maxima
                    else None
                ),
                "exact_one_background_pixels": int(
                    np.count_nonzero((probability == 1.0) & ~mask)
                ),
                "exact_one_target_pixels": int(
                    np.count_nonzero((probability == 1.0) & mask)
                ),
            }
        )
    return rows


def _concentration(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[str, Any]:
    values = np.asarray([int(row[key]) for row in rows], dtype=np.int64)
    order = np.argsort(-values, kind="stable")
    total = int(values.sum())
    specs = {
        "top_1": 1,
        "top_5": 5,
        "top_10": 10,
        "top_1_percent": max(1, int(math.ceil(len(rows) * 0.01))),
        "top_5_percent": max(1, int(math.ceil(len(rows) * 0.05))),
    }
    result: dict[str, Any] = {"total": total, "ranking": []}
    for rank, index in enumerate(order.tolist(), start=1):
        result["ranking"].append(
            {
                "rank": rank,
                "image_id": str(rows[index]["image_id"]),
                "value": int(values[index]),
            }
        )
    for name, requested in specs.items():
        count = min(requested, len(rows))
        subtotal = int(values[order[:count]].sum())
        result[name] = {
            "num_images": count,
            "value": subtotal,
            "fraction": float(subtotal / total) if total else 0.0,
            "image_ids": [str(rows[index]["image_id"]) for index in order[:count]],
        }
    return result


def _quantiles(values: Sequence[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            name: None
            for name in ("min", "q01", "q05", "q25", "median", "q75", "q95", "q99", "max")
        }
    levels = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    names = ("min", "q01", "q05", "q25", "median", "q75", "q95", "q99", "max")
    return {
        name: float(value)
        for name, value in zip(names, np.quantile(array, levels).tolist())
    }


def _background_false_peaks(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    min_score: float = 0.05,
) -> np.ndarray:
    # Define peaks on the background field itself. A nearby true target must
    # not suppress an otherwise extreme background response in this audit.
    background = np.where(mask, -np.inf, probability)
    pooled = ndimage.maximum_filter(background, size=3, mode="nearest")
    peak_mask = (
        (~mask)
        & (background >= pooled - 1e-7)
        & (background >= min_score)
    )
    labels, count = ndimage.label(
        peak_mask, structure=np.ones((3, 3), dtype=np.uint8)
    )
    values: list[float] = []
    for label in range(1, int(count) + 1):
        plateau = labels == label
        values.append(float(np.max(background[plateau])))
    return np.asarray(values, dtype=np.float32)


def ranking_diagnostics(
    probabilities: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    thresholds: Sequence[float],
    *,
    connectivity: int,
) -> dict[str, Any]:
    """Measure probability-domain target/background ordering without selection."""

    if not probabilities or len(probabilities) != len(masks):
        raise ValueError("probabilities and masks must be non-empty paired sequences")

    target_component_maxima: list[float] = []
    target_component_means: list[float] = []
    target_margins_to_image_background_max: list[float] = []
    background_values: list[np.ndarray] = []
    false_peak_values: list[np.ndarray] = []
    gt_with_exact_one_and_image_background_exact_one = 0
    total_target_pixels = 0
    total_background_pixels = 0
    exact_one_target_pixels = 0
    exact_one_background_pixels = 0

    for probability, mask in zip(probabilities, masks):
        target_labels, count = connected_components(mask, connectivity=connectivity)
        background = probability[~mask]
        background_values.append(background)
        peaks = _background_false_peaks(probability, mask)
        false_peak_values.append(peaks)
        background_max = float(np.max(background)) if background.size else float("-inf")
        image_background_exact_one = bool(np.any(background == 1.0))
        total_target_pixels += int(np.count_nonzero(mask))
        total_background_pixels += int(np.count_nonzero(~mask))
        exact_one_target_pixels += int(np.count_nonzero((probability == 1.0) & mask))
        exact_one_background_pixels += int(
            np.count_nonzero((probability == 1.0) & ~mask)
        )
        for label in range(1, count + 1):
            values = probability[target_labels == label]
            maximum = float(np.max(values))
            target_component_maxima.append(maximum)
            target_component_means.append(float(np.mean(values)))
            target_margins_to_image_background_max.append(maximum - background_max)
            if maximum == 1.0 and image_background_exact_one:
                gt_with_exact_one_and_image_background_exact_one += 1

    all_background = np.concatenate(background_values)
    if all_background.size == 0:
        raise ValueError("ranking diagnostics require at least one background pixel")
    all_false_peaks = (
        np.concatenate([values for values in false_peak_values if values.size])
        if any(values.size for values in false_peak_values)
        else np.empty(0, dtype=np.float32)
    )
    target_max = np.asarray(target_component_maxima, dtype=np.float64)
    sorted_peaks = np.sort(np.asarray(all_false_peaks, dtype=np.float64))
    if target_max.size and sorted_peaks.size:
        lower = np.searchsorted(sorted_peaks, target_max, side="left")
        upper = np.searchsorted(sorted_peaks, target_max, side="right")
        pairwise = float(np.mean((lower + 0.5 * (upper - lower)) / sorted_peaks.size))
    else:
        pairwise = 0.0

    grid = np.asarray(thresholds, dtype=np.float64)
    gt_recall = [
        float(np.mean(target_max >= threshold)) if target_max.size else 0.0
        for threshold in grid
    ]
    peak_survival = [
        float(np.mean(all_false_peaks >= threshold))
        if all_false_peaks.size
        else 0.0
        for threshold in grid
    ]
    return {
        "score_representation": "sigmoid_probability_float32",
        "exact_one": {
            "target_pixels": exact_one_target_pixels,
            "background_pixels": exact_one_background_pixels,
            "target_pixel_fraction": (
                float(exact_one_target_pixels / total_target_pixels)
                if total_target_pixels
                else 0.0
            ),
            "background_pixel_fraction": (
                float(exact_one_background_pixels / total_background_pixels)
                if total_background_pixels
                else 0.0
            ),
            "gt_components_saturated_with_saturated_image_background": (
                gt_with_exact_one_and_image_background_exact_one
            ),
        },
        "background_probability_quantiles": {
            "q99": float(np.quantile(all_background, 0.99)),
            "q999": float(np.quantile(all_background, 0.999)),
            "q9999": float(np.quantile(all_background, 0.9999)),
            "max": float(np.max(all_background)),
        },
        "target_component_max_quantiles": _quantiles(target_component_maxima),
        "target_component_mean_quantiles": _quantiles(target_component_means),
        "false_peak_score_quantiles": _quantiles(all_false_peaks),
        "target_margin_to_same_image_background_max_quantiles": _quantiles(
            target_margins_to_image_background_max
        ),
        "gt_component_count": int(target_max.size),
        "false_peak_count": int(all_false_peaks.size),
        "gt_max_below_global_background_max": int(
            np.count_nonzero(target_max < float(np.max(all_background)))
        ),
        "target_vs_false_peak_pairwise_auc_with_half_ties": pairwise,
        "curves": {
            "thresholds": grid.tolist(),
            "gt_max_score_recall": gt_recall,
            "false_peak_survival": peak_survival,
        },
    }


def _key_thresholds(
    decomposition: Mapping[str, Mapping[str, Any]],
    fixed_thresholds: Sequence[float],
) -> list[float]:
    values = {float(value) for value in fixed_thresholds}
    values.add(1.0)
    for budget in decomposition.values():
        for selection in budget["oracle"].values():
            row = selection.get("operating_point")
            if row is not None and float(row["threshold"]) <= 1.0:
                values.add(float(row["threshold"]))
    return sorted(values)


def evaluate_detector_diagnostics(
    curve: FormalCurve,
    budgets: Sequence[BudgetPair],
    *,
    fixed_thresholds: Sequence[float] = (0.5,),
) -> dict[str, Any]:
    metadata = curve.metadata
    score_dir = Path(str(metadata["score_dir"])).resolve()
    manifest, paths, integrity = verify_score_map_directory(
        score_dir, require_integrity=True, require_masks=True
    )
    if manifest is None or integrity.get("verified") is not True:
        raise ValueError("detector diagnostics require verified v3 labeled scores")
    image_ids, probabilities, masks = _load_arrays(paths)
    decomposition = decompose_constraints(curve.rows, budgets)
    key_thresholds = _key_thresholds(decomposition, fixed_thresholds)
    protocol = {
        "matching_rule": str(metadata["matching_rule"]),
        "centroid_distance": float(metadata["centroid_distance"]),
        "connectivity": int(metadata["connectivity"]),
        "min_component_area": int(metadata["min_component_area"]),
        "threshold_rule": "probability >= threshold",
    }
    threshold_diagnostics: dict[str, Any] = {}
    for threshold in key_thresholds:
        per_image = _per_image_at_threshold(
            image_ids,
            probabilities,
            masks,
            threshold,
            matching_rule=protocol["matching_rule"],
            centroid_distance=protocol["centroid_distance"],
            connectivity=protocol["connectivity"],
            min_component_area=protocol["min_component_area"],
        )
        threshold_diagnostics[format(threshold, ".17g")] = {
            "threshold": threshold,
            "per_image": per_image,
            "concentration": {
                "false_pixels": _concentration(per_image, "false_pixels"),
                "false_components": _concentration(per_image, "false_components"),
                "false_peaks": _concentration(per_image, "false_peaks"),
                "exact_one_background_pixels": _concentration(
                    per_image, "exact_one_background_pixels"
                ),
            },
        }
    thresholds = [float(row["threshold"]) for row in curve.rows]
    return {
        "schema_version": DETECTOR_DIAGNOSTICS_SCHEMA_VERSION,
        "artifact_type": "rc-irstd-label-derived-detector-diagnostics",
        "diagnostic_only": True,
        "formal_protocol_eligible": False,
        "test_labels_used": True,
        "threshold_selection_is_oracle": True,
        "constraint_decomposition": decomposition,
        "key_threshold_diagnostics": threshold_diagnostics,
        "ranking_diagnostics": ranking_diagnostics(
            probabilities,
            masks,
            thresholds,
            connectivity=protocol["connectivity"],
        ),
        "protocol": protocol,
        "provenance": {
            "curve": str(curve.path),
            "curve_sha256": metadata["curve_sha256"],
            "curve_metadata": str(curve.metadata_path),
            "curve_metadata_sha256": curve.metadata_sha256,
            "threshold_grid_sha256": metadata["threshold_grid_sha256"],
            "score_dir": str(score_dir),
            "score_manifest_sha256": integrity["manifest_sha256"],
            "score_records_sha256": integrity["records_sha256"],
            "score_ordered_image_ids_sha256": integrity[
                "ordered_image_ids_sha256"
            ],
            "detector_weight_sha256": metadata["detector_weight_sha256"],
            "target_dataset": metadata["target_dataset"],
            "source_datasets": metadata["source_datasets"],
            "num_images": len(paths),
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", required=True)
    parser.add_argument(
        "--budget",
        action="append",
        type=parse_budget,
        required=True,
        help="Repeat NAME:PIXEL:COMPONENT, e.g. loose:1e-5:5",
    )
    parser.add_argument(
        "--fixed-threshold", action="append", type=float, default=None
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    budgets: list[BudgetPair] = list(args.budget)
    if len({budget.name for budget in budgets}) != len(budgets):
        raise ValueError("budget names must be unique")
    fixed = [0.5] if args.fixed_threshold is None else args.fixed_threshold
    if any(not np.isfinite(value) or not 0.0 <= value <= EMPTY_SET_THRESHOLD for value in fixed):
        raise ValueError("fixed thresholds must lie in the evaluation threshold range")
    curve = load_formal_curve(args.curve, expected_split_role="test")
    result = evaluate_detector_diagnostics(
        curve, budgets, fixed_thresholds=fixed
    )
    write_json_atomic(args.output, result)
    print(Path(args.output))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BudgetPair",
    "DETECTOR_DIAGNOSTICS_SCHEMA_VERSION",
    "decompose_constraints",
    "evaluate_detector_diagnostics",
    "main",
    "parse_budget",
    "ranking_diagnostics",
]
