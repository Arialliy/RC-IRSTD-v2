"""Policy-matched, label-derived oracle diagnostics on formal score artifacts.

Unlike a single global target oracle, this diagnostic matches the deployment
decision granularity.  A static-fold action is selected only from that fold's
query labels; a causal action is selected only from its future evaluation
labels; and an image action is selected only from that image.  Adaptation IDs
are recorded for policy identity but are never passed to threshold selection.

Every result is label-derived, oracle-only, and ineligible for a deployment or
formal guarantee.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifact_integrity import file_sha256, verify_score_map_directory
from .operating_point import select_operating_point
from .threshold_sweep import (
    CURVE_METADATA_ARTIFACT_TYPE,
    EMPTY_SET_THRESHOLD,
    build_default_thresholds,
    curve_metadata_path,
    evaluation_threshold_grid_sha256,
    read_curve_csv,
    sweep_thresholds,
    validate_formal_score_manifest,
    validate_thresholds,
    write_json_atomic,
)


POLICY_MATCHED_ORACLE_SCHEMA_VERSION = "rc-v2-policy-matched-oracle-v1"
DEFAULT_STATIC_FOLDS = 5
DEFAULT_SEED = 42
DEFAULT_ADAPTATION_WINDOW = 32
DEFAULT_EVALUATION_WINDOW = 1
DEFAULT_CAUSAL_STRIDE = 33


@dataclass(frozen=True)
class DecisionUnit:
    unit_id: str
    unit_index: int
    adaptation_indices: tuple[int, ...]
    evaluation_indices: tuple[int, ...]
    adaptation_ids: tuple[str, ...]
    evaluation_ids: tuple[str, ...]
    adaptation_complement_size: int | None = None
    adaptation_sampling_rule: str | None = None
    adaptation_sampling_seed_components: tuple[int, int] | None = None


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a positive integer") from error
    if result <= 0 or result != value:
        raise ValueError(f"{field} must be a positive integer")
    return result


def _unit_id(
    policy: str,
    unit_index: int,
    adaptation_ids: Sequence[str],
    evaluation_ids: Sequence[str],
) -> str:
    payload = json.dumps(
        {
            "schema": POLICY_MATCHED_ORACLE_SCHEMA_VERSION,
            "policy": policy,
            "unit_index": int(unit_index),
            "adaptation_ids": list(adaptation_ids),
            "evaluation_ids": list(evaluation_ids),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{policy}-{unit_index:06d}-{hashlib.sha256(payload).hexdigest()[:16]}"


def _make_unit(
    policy: str,
    unit_index: int,
    adaptation_indices: Sequence[int],
    evaluation_indices: Sequence[int],
    image_ids: Sequence[str],
    *,
    adaptation_complement_size: int | None = None,
    adaptation_sampling_rule: str | None = None,
    adaptation_sampling_seed_components: tuple[int, int] | None = None,
) -> DecisionUnit:
    adaptation = tuple(int(value) for value in adaptation_indices)
    evaluation = tuple(int(value) for value in evaluation_indices)
    if not adaptation and policy in {"static", "causal"}:
        raise ValueError(f"{policy} decision units require adaptation images")
    if not evaluation:
        raise ValueError("Every decision unit requires at least one evaluation image")
    if len(adaptation) != len(set(adaptation)):
        raise ValueError("A decision unit contains duplicate adaptation indices")
    if len(evaluation) != len(set(evaluation)):
        raise ValueError("A decision unit contains duplicate evaluation indices")
    if set(adaptation).intersection(evaluation):
        raise ValueError("A decision unit reuses an image in adaptation and evaluation")
    if any(index < 0 or index >= len(image_ids) for index in adaptation + evaluation):
        raise ValueError("A decision-unit index lies outside the score artifact")
    if adaptation_complement_size is not None:
        if adaptation_complement_size < len(adaptation):
            raise ValueError("adaptation complement cannot be smaller than its sample")
        if not adaptation_sampling_rule:
            raise ValueError("sampled adaptation requires an explicit sampling rule")
        if adaptation_sampling_seed_components is None:
            raise ValueError("sampled adaptation requires fold-local seed components")
    adaptation_ids = tuple(str(image_ids[index]) for index in adaptation)
    evaluation_ids = tuple(str(image_ids[index]) for index in evaluation)
    return DecisionUnit(
        unit_id=_unit_id(policy, unit_index, adaptation_ids, evaluation_ids),
        unit_index=int(unit_index),
        adaptation_indices=adaptation,
        evaluation_indices=evaluation,
        adaptation_ids=adaptation_ids,
        evaluation_ids=evaluation_ids,
        adaptation_complement_size=adaptation_complement_size,
        adaptation_sampling_rule=adaptation_sampling_rule,
        adaptation_sampling_seed_components=adaptation_sampling_seed_components,
    )


def build_decision_units(
    image_ids: Sequence[str],
    policy: str,
    *,
    folds: int = DEFAULT_STATIC_FOLDS,
    seed: int = DEFAULT_SEED,
    adaptation_window: int = DEFAULT_ADAPTATION_WINDOW,
    evaluation_window: int = DEFAULT_EVALUATION_WINDOW,
    stride: int = DEFAULT_CAUSAL_STRIDE,
) -> list[DecisionUnit]:
    """Build deterministic decision units without opening labels."""

    ids = [str(value) for value in image_ids]
    if not ids or any(not value for value in ids):
        raise ValueError("image_ids must be non-empty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("image_ids must be globally unique")
    if policy not in {"global", "static", "causal", "image"}:
        raise ValueError("policy must be global, static, causal, or image")

    if policy == "global":
        units = [_make_unit(policy, 0, (), range(len(ids)), ids)]
    elif policy == "image":
        units = [_make_unit(policy, index, (), (index,), ids) for index in range(len(ids))]
    elif policy == "static":
        folds = _positive_integer(folds, "folds")
        adaptation_window = _positive_integer(
            adaptation_window, "adaptation_window"
        )
        if folds < 2 or folds > len(ids):
            raise ValueError("static folds must lie in [2, num_images]")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        permutation = np.random.default_rng(int(seed)).permutation(len(ids))
        query_folds = [
            tuple(int(value) for value in fold.tolist())
            for fold in np.array_split(permutation, folds)
        ]
        all_indices = tuple(range(len(ids)))
        units = []
        for fold_index, query in enumerate(query_folds):
            query_set = set(query)
            complement = tuple(index for index in all_indices if index not in query_set)
            complement_size = len(complement)
            if complement_size < adaptation_window:
                raise ValueError(
                    "Static cross-fit complement is smaller than the requested "
                    f"adaptation window in fold {fold_index}: "
                    f"complement_size={complement_size}, "
                    f"adaptation_window={adaptation_window}"
                )
            fold_rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(fold_index)])
            )
            selected_positions = fold_rng.choice(
                complement_size,
                size=adaptation_window,
                replace=False,
            )
            adaptation = tuple(complement[int(position)] for position in selected_positions)
            units.append(
                _make_unit(
                    policy,
                    fold_index,
                    adaptation,
                    query,
                    ids,
                    adaptation_complement_size=complement_size,
                    adaptation_sampling_rule=(
                        "seedsequence(seed,fold_index)_without_replacement_from_complement"
                    ),
                    adaptation_sampling_seed_components=(int(seed), int(fold_index)),
                )
            )
    else:
        adaptation_window = _positive_integer(adaptation_window, "adaptation_window")
        evaluation_window = _positive_integer(evaluation_window, "evaluation_window")
        stride = _positive_integer(stride, "stride")
        if stride != adaptation_window + evaluation_window:
            raise ValueError(
                "formal causal policy requires stride=adaptation_window+evaluation_window"
            )
        block_size = adaptation_window + evaluation_window
        starts = range(0, len(ids) - block_size + 1, stride)
        units = [
            _make_unit(
                policy,
                block_index,
                range(start, start + adaptation_window),
                range(start + adaptation_window, start + block_size),
                ids,
            )
            for block_index, start in enumerate(starts)
        ]
        if not units:
            raise ValueError(
                "No complete causal A->E block fits the ordered score artifact"
            )

    flattened_evaluation = [
        index for unit in units for index in unit.evaluation_indices
    ]
    if len(flattened_evaluation) != len(set(flattened_evaluation)):
        raise ValueError("An image belongs to more than one evaluation decision unit")
    if policy in {"global", "static", "image"} and set(flattened_evaluation) != set(
        range(len(ids))
    ):
        raise ValueError(f"{policy} policy does not cover every score-map image")
    if policy == "static" and len(flattened_evaluation) != len(ids):
        raise ValueError("static policy does not provide exactly-once query coverage")
    if policy == "causal":
        all_adaptation = {
            index for unit in units for index in unit.adaptation_indices
        }
        all_evaluation = set(flattened_evaluation)
        role_reuse = sorted(all_adaptation.intersection(all_evaluation))
        if role_reuse:
            raise ValueError(
                "Causal policy reuses images across adaptation/evaluation roles: "
                + ", ".join(ids[index] for index in role_reuse[:10])
            )
    return units


def _require_canonical_grid(thresholds: Sequence[float] | np.ndarray) -> np.ndarray:
    grid = validate_thresholds(thresholds)
    canonical = build_default_thresholds()
    if not np.array_equal(grid, canonical):
        raise ValueError(
            "Policy-matched oracle requires the canonical 653-point evaluation grid"
        )
    if grid.size != 653 or grid[-2] != 1.0 or grid[-1] != EMPTY_SET_THRESHOLD:
        raise ValueError("Canonical evaluation-grid endpoints are missing")
    return grid


def _bound_curve_grid(
    curve_path: str | Path,
    *,
    score_contract: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    curve = Path(curve_path).expanduser().resolve()
    sidecar_path = curve_metadata_path(curve)
    if not curve.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("A manifest-bound curve requires its metadata sidecar")
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Curve metadata must decode to an object")
    if metadata.get("artifact_type") != CURVE_METADATA_ARTIFACT_TYPE:
        raise ValueError("Curve metadata artifact_type is unsupported")
    if metadata.get("formal_protocol_eligible") is not True:
        raise ValueError("Curve metadata is not formal-protocol eligible")
    if metadata.get("curve_file") != curve.name:
        raise ValueError("Curve metadata is bound to a different filename")
    if metadata.get("curve_sha256") != file_sha256(curve):
        raise ValueError("Curve SHA-256 does not match its metadata")
    bindings = {
        "score_manifest_sha256": "score_manifest_sha256",
        "score_records_sha256": "score_records_sha256",
        "score_ordered_image_ids_sha256": "score_ordered_image_ids_sha256",
        "detector_weight_sha256": "detector_weight_sha256",
        "split_role": "split_role",
    }
    for metadata_field, contract_field in bindings.items():
        if metadata.get(metadata_field) != score_contract.get(contract_field):
            raise ValueError(
                f"Curve/score artifact binding differs in {metadata_field}"
            )
    if metadata.get("score_num_records") != score_contract.get("score_num_records"):
        raise ValueError("Curve/score artifact image counts differ")
    rows = read_curve_csv(curve)
    grid = _require_canonical_grid(
        np.asarray([float(row["threshold"]) for row in rows], dtype=np.float64)
    )
    if metadata.get("threshold_grid_size") != int(grid.size):
        raise ValueError("Curve metadata threshold_grid_size mismatch")
    grid_hash = evaluation_threshold_grid_sha256(grid)
    if metadata.get("threshold_grid_sha256") != grid_hash:
        raise ValueError("Curve metadata threshold-grid hash mismatch")
    return grid, {
        "source_type": "manifest_bound_formal_curve",
        "curve_path": str(curve),
        "curve_sha256": file_sha256(curve),
        "curve_metadata_path": str(sidecar_path.resolve()),
        "curve_metadata_sha256": file_sha256(sidecar_path),
    }


def load_policy_grid(
    *,
    score_contract: Mapping[str, Any],
    threshold_grid_path: str | Path | None = None,
    curve_path: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if threshold_grid_path is not None and curve_path is not None:
        raise ValueError("Choose either threshold_grid_path or curve_path")
    if curve_path is not None:
        return _bound_curve_grid(curve_path, score_contract=score_contract)
    if threshold_grid_path is None:
        return _require_canonical_grid(build_default_thresholds()), {
            "source_type": "built_in_canonical_653_grid"
        }
    path = Path(threshold_grid_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    grid = _require_canonical_grid(np.load(path, allow_pickle=False))
    return grid, {
        "source_type": "explicit_canonical_grid_file",
        "path": str(path),
        "file_sha256": file_sha256(path),
    }


def _load_formal_scores(
    score_dir: str | Path,
    *,
    expected_split_role: str,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[str],
    dict[str, Any],
    dict[str, Any],
]:
    root = Path(score_dir).expanduser().resolve()
    manifest, paths, integrity = verify_score_map_directory(
        root, require_integrity=True, require_masks=True
    )
    contract = validate_formal_score_manifest(
        manifest, integrity, expected_split_role=expected_split_role
    )
    assert manifest is not None
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != len(paths):
        raise ValueError("Formal score manifest records do not align with files")
    probabilities: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    image_ids: list[str] = []
    for index, (record, path) in enumerate(zip(records, paths)):
        if not isinstance(record, Mapping):
            raise ValueError(f"Score record {index} must be an object")
        with np.load(path, allow_pickle=False) as payload:
            required = {"prob", "mask", "image_id", "labels_loaded", "spatial_mode"}
            missing = sorted(required.difference(payload.files))
            if missing:
                raise ValueError(f"Formal score record is missing: {', '.join(missing)}")
            if payload["labels_loaded"].ndim != 0 or payload["labels_loaded"].dtype.kind != "b":
                raise ValueError("labels_loaded must be a boolean scalar")
            if not bool(payload["labels_loaded"].item()):
                raise ValueError("Policy-matched oracle requires embedded labels")
            if str(np.asarray(payload["spatial_mode"]).item()) != "native":
                raise ValueError("Policy-matched oracle requires native score maps")
            image_id = str(np.asarray(payload["image_id"]).item())
            if image_id != str(record.get("image_id", "")):
                raise ValueError("Embedded image ID differs from the score manifest")
            probabilities.append(np.asarray(payload["prob"]))
            masks.append(np.asarray(payload["mask"]))
            image_ids.append(image_id)
    return probabilities, masks, image_ids, contract, {
        "manifest": manifest,
        "integrity": integrity,
        "score_dir": str(root),
    }


def _aggregate_selected_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("At least one selected row is required")
    totals = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "tp_objects",
            "gt_objects",
            "fp_components",
            "fp_pixels",
            "total_pixels",
        )
    }
    total_pixels = totals["total_pixels"]
    if total_pixels <= 0:
        raise ValueError("Selected rows contain no evaluated pixels")
    totals.update(
        {
            "pd": (
                float(totals["tp_objects"] / totals["gt_objects"])
                if totals["gt_objects"] > 0
                else 0.0
            ),
            "fa_pixel": float(totals["fp_pixels"] / total_pixels),
            "fa_component_mp": float(
                totals["fp_components"] / (total_pixels / 1_000_000.0)
            ),
        }
    )
    return totals


def _pixel_budget_pruned_grid(
    probabilities: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    grid: np.ndarray,
    *,
    pixel_budget: float,
    min_component_area: int,
) -> tuple[np.ndarray, np.ndarray | None, int | None, dict[str, Any]]:
    """Losslessly discard thresholds that already violate the pixel budget.

    With ``min_component_area=1``, component matching evaluates every raw
    thresholded prediction pixel.  Consequently, background score order
    statistics give the exact FP-pixel count at every threshold without
    connected-component work.  A threshold that fails the pixel budget cannot
    satisfy the joint pixel/component constraint and cannot be the selected
    operating point.
    """

    original_size = int(grid.size)
    grid_hash = evaluation_threshold_grid_sha256(grid)
    if min_component_area != 1:
        return grid, None, None, {
            "enabled": False,
            "lossless": True,
            "method": "disabled_full_grid_component_matching",
            "original_threshold_grid_size": original_size,
            "original_threshold_grid_sha256": grid_hash,
            "evaluated_threshold_count": original_size,
            "pruned_threshold_count": 0,
            "reason": "pixel-count shortcut requires min_component_area=1",
            "proof": (
                "No thresholds were pruned; evaluating the original grid is "
                "identical by construction."
            ),
            "runtime_exact_fp_pixel_cross_check": False,
        }

    background_chunks: list[np.ndarray] = []
    total_pixels = 0
    for pair_index, (probability, mask) in enumerate(zip(probabilities, masks)):
        score = np.asarray(probability)
        target = np.asarray(mask)
        if score.ndim == 3 and score.shape[0] == 1:
            score = score[0]
        if target.ndim == 3 and target.shape[0] == 1:
            target = target[0]
        if score.ndim != 2 or target.ndim != 2:
            raise ValueError(
                f"score/mask pair {pair_index} must contain 2-D arrays"
            )
        if score.shape != target.shape:
            raise ValueError(
                f"score/mask pair {pair_index} has mismatched shapes: "
                f"{score.shape} vs {target.shape}"
            )
        if not np.issubdtype(score.dtype, np.number) or not np.isfinite(score).all():
            raise ValueError(
                f"probability map {pair_index} must be finite and numeric"
            )
        if np.any((score < 0.0) | (score > 1.0)):
            raise ValueError(
                f"probability map {pair_index} contains values outside [0, 1]"
            )
        if not np.issubdtype(target.dtype, np.number) and target.dtype != np.bool_:
            raise TypeError(f"mask {pair_index} must be numeric or boolean")
        if not np.isfinite(target).all():
            raise ValueError(f"mask {pair_index} contains NaN or infinity")
        score64 = np.ascontiguousarray(score, dtype=np.float64)
        target_bool = np.ascontiguousarray(target > 0)
        background_chunks.append(score64[~target_bool])
        total_pixels += int(target_bool.size)
    if total_pixels <= 0:
        raise ValueError("query/evaluation unit contains no pixels")

    nonempty_chunks = [chunk for chunk in background_chunks if chunk.size]
    if nonempty_chunks:
        sorted_background = np.sort(np.concatenate(nonempty_chunks))
    else:
        sorted_background = np.empty(0, dtype=np.float64)
    first_ge = np.searchsorted(sorted_background, grid, side="left")
    fp_pixel_counts = (
        int(sorted_background.size) - first_ge.astype(np.int64, copy=False)
    )
    if np.any(np.diff(fp_pixel_counts) > 0):
        raise RuntimeError("background FP-pixel counts are not threshold-monotone")
    pixel_rates = fp_pixel_counts.astype(np.float64) / float(total_pixels)
    retained_mask = pixel_rates <= float(pixel_budget)
    reject_mask = grid == EMPTY_SET_THRESHOLD
    if int(np.count_nonzero(reject_mask)) != 1:
        raise ValueError("Oracle grid must contain exactly one reject-all sentinel")
    retained_mask = retained_mask | reject_mask
    retained_grid = grid[retained_mask]
    retained_counts = fp_pixel_counts[retained_mask]
    if retained_grid.size == 0 or retained_grid[-1] != EMPTY_SET_THRESHOLD:
        raise RuntimeError("lossless pruning removed the reject-all sentinel")
    return retained_grid, retained_counts, total_pixels, {
        "enabled": True,
        "lossless": True,
        "method": "sorted_background_scores_vectorized_searchsorted",
        "original_threshold_grid_size": original_size,
        "original_threshold_grid_sha256": grid_hash,
        "evaluated_threshold_count": int(retained_grid.size),
        "pruned_threshold_count": int(original_size - retained_grid.size),
        "pixel_feasible_threshold_count": int(np.count_nonzero(retained_mask)),
        "background_pixel_count": int(sorted_background.size),
        "total_pixel_count": int(total_pixels),
        "pixel_budget": float(pixel_budget),
        "reject_all_explicitly_retained": True,
        "fp_pixel_counts_monotone_nonincreasing": True,
        "proof": (
            "For min_component_area=1, sorted background scores give the exact "
            "FP-pixel count for score>=threshold. Any removed threshold already "
            "violates the pixel budget, so it cannot satisfy the joint pixel and "
            "component budgets or win max-Pd selection. All pixel-feasible "
            "thresholds and the explicit reject-all sentinel are retained."
        ),
        "runtime_exact_fp_pixel_cross_check": True,
    }


def evaluate_policy_oracle(
    probabilities: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    image_ids: Sequence[str],
    units: Sequence[DecisionUnit],
    thresholds: Sequence[float] | np.ndarray,
    *,
    pixel_budget: float,
    component_budget: float,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
) -> dict[str, Any]:
    """Select one query-label oracle action per precomputed decision unit."""

    if len(probabilities) != len(masks) or len(probabilities) != len(image_ids):
        raise ValueError("scores, masks, and image IDs must have identical lengths")
    grid = validate_thresholds(thresholds)
    if grid[-1] != EMPTY_SET_THRESHOLD:
        raise ValueError("Oracle grid must end with the explicit empty-set sentinel")
    if not np.isfinite(pixel_budget) or pixel_budget <= 0.0:
        raise ValueError("pixel_budget must be finite and positive")
    if not np.isfinite(component_budget) or component_budget <= 0.0:
        raise ValueError("component_budget must be finite and positive")

    unit_results: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for unit in units:
        # This slice is the central leakage boundary: adaptation_indices are
        # intentionally not referenced by score/mask selection.
        query_probabilities = [probabilities[index] for index in unit.evaluation_indices]
        query_masks = [masks[index] for index in unit.evaluation_indices]
        retained_grid, expected_fp_pixels, expected_total_pixels, pruning_audit = (
            _pixel_budget_pruned_grid(
                query_probabilities,
                query_masks,
                grid,
                pixel_budget=pixel_budget,
                min_component_area=min_component_area,
            )
        )
        rows = sweep_thresholds(
            query_probabilities,
            query_masks,
            retained_grid,
            matching_rule=matching_rule,
            centroid_distance=centroid_distance,
            connectivity=connectivity,
            min_component_area=min_component_area,
        )
        if expected_fp_pixels is not None:
            observed_fp_pixels = np.asarray(
                [int(row["fp_pixels"]) for row in rows], dtype=np.int64
            )
            observed_total_pixels = {
                int(row["total_pixels"]) for row in rows
            }
            if not np.array_equal(observed_fp_pixels, expected_fp_pixels):
                raise RuntimeError(
                    "vectorized FP-pixel pruning disagrees with component matching"
                )
            if observed_total_pixels != {int(expected_total_pixels)}:
                raise RuntimeError(
                    "vectorized pruning/component matching pixel exposures differ"
                )
        selected = select_operating_point(
            rows,
            pixel_budget=pixel_budget,
            component_budget=component_budget,
            strategy="max_pd",
        )
        if selected is None:  # the required empty-set sentinel should make this impossible
            raise RuntimeError("No feasible query-label oracle threshold was found")
        selected_rows.append(selected)
        unit_results.append(
            {
                "unit_id": unit.unit_id,
                "unit_index": unit.unit_index,
                "adaptation_indices": list(unit.adaptation_indices),
                "evaluation_indices": list(unit.evaluation_indices),
                "adaptation_ids": list(unit.adaptation_ids),
                "evaluation_ids": list(unit.evaluation_ids),
                "num_adaptation_images": len(unit.adaptation_ids),
                "num_evaluation_images": len(unit.evaluation_ids),
                "adaptation_complement_size": unit.adaptation_complement_size,
                "adaptation_sampling_rule": unit.adaptation_sampling_rule,
                "adaptation_sampling_seed_components": (
                    list(unit.adaptation_sampling_seed_components)
                    if unit.adaptation_sampling_seed_components is not None
                    else None
                ),
                "adaptation_evaluation_role_overlap_ids": sorted(
                    set(unit.adaptation_ids).intersection(unit.evaluation_ids)
                ),
                "labels_used_for_selection_ids": list(unit.evaluation_ids),
                "adaptation_labels_used_for_selection": False,
                "threshold_pruning": pruning_audit,
                "threshold": float(selected["threshold"]),
                "pd": float(selected["pd"]),
                "fa_pixel": float(selected["fa_pixel"]),
                "fa_component_mp": float(selected["fa_component_mp"]),
                "tp_objects": int(selected["tp_objects"]),
                "gt_objects": int(selected["gt_objects"]),
                "fp_components": int(selected["fp_components"]),
                "fp_pixels": int(selected["fp_pixels"]),
                "total_pixels": int(selected["total_pixels"]),
                "individual_budget_satisfied": bool(
                    float(selected["fa_pixel"]) <= pixel_budget
                    and float(selected["fa_component_mp"]) <= component_budget
                ),
                "budget_feasibility": {
                    "pixel_budget_satisfied": bool(
                        float(selected["fa_pixel"]) <= pixel_budget
                    ),
                    "component_budget_satisfied": bool(
                        float(selected["fa_component_mp"]) <= component_budget
                    ),
                    "joint_budget_satisfied": bool(
                        float(selected["fa_pixel"]) <= pixel_budget
                        and float(selected["fa_component_mp"]) <= component_budget
                    ),
                },
            }
        )
    aggregate = _aggregate_selected_rows(selected_rows)
    aggregate["global_aggregate_budget_satisfied"] = bool(
        aggregate["fa_pixel"] <= pixel_budget
        and aggregate["fa_component_mp"] <= component_budget
    )
    pruning_rows = [result["threshold_pruning"] for result in unit_results]
    return {
        "units": unit_results,
        "aggregate": aggregate,
        "all_units_individually_budget_satisfied": all(
            bool(result["individual_budget_satisfied"]) for result in unit_results
        ),
        "threshold_pruning": {
            "original_threshold_grid_size": int(grid.size),
            "original_threshold_grid_sha256": evaluation_threshold_grid_sha256(
                grid
            ),
            "num_decision_units": len(unit_results),
            "evaluated_threshold_counts_by_unit": [
                int(row["evaluated_threshold_count"]) for row in pruning_rows
            ],
            "total_evaluated_unit_thresholds": sum(
                int(row["evaluated_threshold_count"]) for row in pruning_rows
            ),
            "total_original_unit_thresholds": int(grid.size) * len(unit_results),
            "total_pruned_unit_thresholds": sum(
                int(row["pruned_threshold_count"]) for row in pruning_rows
            ),
            "all_units_lossless": all(bool(row["lossless"]) for row in pruning_rows),
            "all_units_vectorized_pixel_pruning_enabled": all(
                bool(row["enabled"]) for row in pruning_rows
            ),
            "proof": (
                "Each enabled unit removes only thresholds whose exact FP-pixel "
                "rate already exceeds that unit's pixel budget. Such thresholds "
                "cannot be jointly budget-feasible; retained counts are checked "
                "against component matching. Disabled units evaluate the full grid."
            ),
        },
        "selection_label_ids": [
            image_id
            for result in unit_results
            for image_id in result["labels_used_for_selection_ids"]
        ],
    }


def run_policy_matched_oracle(
    *,
    score_dir: str | Path,
    policy: str,
    pixel_budget: float,
    component_budget: float,
    expected_split_role: str = "test",
    threshold_grid_path: str | Path | None = None,
    curve_path: str | Path | None = None,
    folds: int = DEFAULT_STATIC_FOLDS,
    seed: int = DEFAULT_SEED,
    adaptation_window: int = DEFAULT_ADAPTATION_WINDOW,
    evaluation_window: int = DEFAULT_EVALUATION_WINDOW,
    stride: int = DEFAULT_CAUSAL_STRIDE,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
) -> dict[str, Any]:
    probabilities, masks, image_ids, contract, score_evidence = _load_formal_scores(
        score_dir, expected_split_role=expected_split_role
    )
    grid, grid_source = load_policy_grid(
        score_contract=contract,
        threshold_grid_path=threshold_grid_path,
        curve_path=curve_path,
    )
    units = build_decision_units(
        image_ids,
        policy,
        folds=folds,
        seed=seed,
        adaptation_window=adaptation_window,
        evaluation_window=evaluation_window,
        stride=stride,
    )
    evaluated = evaluate_policy_oracle(
        probabilities,
        masks,
        image_ids,
        units,
        grid,
        pixel_budget=pixel_budget,
        component_budget=component_budget,
        matching_rule=matching_rule,
        centroid_distance=centroid_distance,
        connectivity=connectivity,
        min_component_area=min_component_area,
    )
    evaluation_ids = [
        image_id for unit in units for image_id in unit.evaluation_ids
    ]
    adaptation_ids = [
        image_id for unit in units for image_id in unit.adaptation_ids
    ]
    full_coverage = set(evaluation_ids) == set(image_ids) and len(evaluation_ids) == len(
        image_ids
    )
    within_unit_role_overlap = sorted(
        {
            image_id
            for unit in units
            for image_id in set(unit.adaptation_ids).intersection(unit.evaluation_ids)
        }
    )
    policy_interpretations = {
        "global": {
            "evidence_role": "globally_pooled_target_oracle_upper_bound",
            "budget_enforcement_unit": "single_global_evaluation_pool",
            "aggregate_metric_scope": "all_score_images",
            "interpretation": (
                "One threshold is selected from all target labels; this is a pooled "
                "oracle upper bound, not a deployable threshold."
            ),
        },
        "static": {
            "evidence_role": "primary_policy_matched_oracle_evidence",
            "budget_enforcement_unit": "each_seeded_static_query_fold",
            "aggregate_metric_scope": (
                "raw_count_micro_aggregate_across_exactly_once_query_folds"
            ),
            "interpretation": (
                "The seeded non-overlapping query folds cover the full score artifact "
                "exactly once. Each fold uses a fixed-size, fold-seeded sample from its "
                "complement, while its oracle threshold is selected only from that "
                "fold's query labels. This is still label-derived oracle evidence."
            ),
        },
        "causal": {
            "evidence_role": "causal_policy_matched_oracle_sensitivity_evidence",
            "budget_enforcement_unit": "each_causal_evaluation_window",
            "aggregate_metric_scope": (
                "raw_count_micro_aggregate_over_complete_causal_evaluation_windows_only"
            ),
            "interpretation": (
                "Only E frames in complete, disjoint A->E blocks are evaluated. Unless "
                "coverage is 100%, the aggregate Pd is not the full-test-set Pd and "
                "must not be compared as if it used every score image."
            ),
        },
        "image": {
            "evidence_role": "extremely_permissive_per_image_oracle_upper_bound",
            "budget_enforcement_unit": "each_individual_image",
            "aggregate_metric_scope": (
                "raw_count_micro_aggregate_after_per_image_oracle_selection"
            ),
            "interpretation": (
                "Every image uses its own label-selected threshold; this is an extremely "
                "permissive upper bound, not evidence for a realizable policy."
            ),
        },
    }
    policy_contract: dict[str, Any] = {
        "name": policy,
        "num_decision_units": len(units),
        "query_labels_only": True,
        "adaptation_labels_used_for_selection": False,
        "evaluation_ids_globally_unique": len(evaluation_ids)
        == len(set(evaluation_ids)),
        "full_evaluation_coverage": full_coverage,
        **policy_interpretations[policy],
    }
    if policy == "static":
        policy_contract.update(
            {
                "folds": int(folds),
                "seed": int(seed),
                "adaptation_window": int(adaptation_window),
                "adaptation_sampling_rule": (
                    "seedsequence(seed,fold_index)_without_replacement_from_complement"
                ),
                "selected_adaptation_sizes": [
                    len(unit.adaptation_indices) for unit in units
                ],
                "complement_sizes": [
                    int(unit.adaptation_complement_size)
                    for unit in units
                    if unit.adaptation_complement_size is not None
                ],
                "query_fold_sizes": [
                    len(unit.evaluation_indices) for unit in units
                ],
                "query_folds_pairwise_disjoint": len(evaluation_ids)
                == len(set(evaluation_ids)),
                "exactly_once_query_coverage": full_coverage,
            }
        )
    elif policy == "causal":
        policy_contract.update(
            {
                "adaptation_window": int(adaptation_window),
                "evaluation_window": int(evaluation_window),
                "stride": int(stride),
                "num_complete_causal_blocks": len(units),
                "num_adaptation_assignments": sum(
                    len(unit.adaptation_indices) for unit in units
                ),
                "num_evaluation_assignments": len(evaluation_ids),
                "global_adaptation_evaluation_role_overlap": sorted(
                    set(adaptation_ids).intersection(evaluation_ids)
                ),
            }
        )
    result = {
        "schema_version": POLICY_MATCHED_ORACLE_SCHEMA_VERSION,
        "mode": "policy_matched_target_oracle_diagnostic",
        "oracle_only": True,
        "labels_used_for_threshold_selection": True,
        "formal_protocol_eligible": False,
        "guarantee": "none; target query/evaluation labels select each oracle action",
        "policy": policy_contract,
        "budgets": {
            "pixel": float(pixel_budget),
            "component_per_megapixel": float(component_budget),
        },
        "budget_semantics": {
            "selection_enforcement": "each_decision_unit_independently_satisfies_both_budgets",
            "enforcement_unit": policy_interpretations[policy][
                "budget_enforcement_unit"
            ],
            "global_aggregate_role": "post_selection_audit_not_a_joint_selection_constraint",
            "cross_unit_label_pooling_for_selection": policy == "global",
            "note": (
                "For adaptive policies, jointly optimizing a global aggregate budget "
                "would let other units' labels influence an action and is intentionally "
                "not called policy-matched. Use policy=global for a globally pooled oracle."
            ),
        },
        "threshold_grid": {
            "size": int(grid.size),
            "sha256": evaluation_threshold_grid_sha256(grid),
            "contains_exact_one": bool(np.any(grid == 1.0)),
            "empty_set_threshold": float(EMPTY_SET_THRESHOLD),
            **grid_source,
        },
        "threshold_pruning": evaluated["threshold_pruning"],
        "coverage": {
            "num_score_images": len(image_ids),
            "num_evaluated_images": len(evaluation_ids),
            "num_unique_evaluated_images": len(set(evaluation_ids)),
            "evaluated_fraction": float(len(evaluation_ids) / len(image_ids)),
            "full_evaluation_coverage": full_coverage,
            "aggregate_metrics_cover_evaluated_images_only": True,
            "aggregate_pd_is_full_score_artifact_pd": full_coverage,
            "aggregate_metric_scope": policy_interpretations[policy][
                "aggregate_metric_scope"
            ],
            "unevaluated_image_ids": [
                image_id for image_id in image_ids if image_id not in set(evaluation_ids)
            ],
        },
        "partition_audit": {
            "evaluation_ids_globally_unique": len(evaluation_ids)
            == len(set(evaluation_ids)),
            "within_unit_adaptation_evaluation_role_overlap_ids": (
                within_unit_role_overlap
            ),
            "within_unit_role_disjoint": not within_unit_role_overlap,
            "num_evaluation_assignments": len(evaluation_ids),
            "num_unique_evaluation_ids": len(set(evaluation_ids)),
        },
        "units": evaluated["units"],
        "aggregate": evaluated["aggregate"],
        "all_units_individually_budget_satisfied": evaluated[
            "all_units_individually_budget_satisfied"
        ],
        "provenance": {
            "score_dir": score_evidence["score_dir"],
            "score_manifest_sha256": contract["score_manifest_sha256"],
            "score_records_sha256": contract["score_records_sha256"],
            "score_ordered_image_ids_sha256": contract[
                "score_ordered_image_ids_sha256"
            ],
            "detector_weight_sha256": contract["detector_weight_sha256"],
            "checkpoint_selection_rule": contract["checkpoint_selection_rule"],
            "target_dataset": contract["target_dataset"],
            "source_datasets": contract["source_datasets"],
            "split_role": contract["split_role"],
            "split_file_sha256": contract["split_file_sha256"],
        },
        "matching": {
            "rule": matching_rule,
            "centroid_distance": float(centroid_distance),
            "connectivity": int(connectivity),
            "min_component_area": int(min_component_area),
        },
    }
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument(
        "--policy", choices=("global", "static", "causal", "image"), default="static"
    )
    parser.add_argument("--pixel-budget", required=True, type=float)
    parser.add_argument("--component-budget", required=True, type=float)
    parser.add_argument("--expected-split-role", choices=("train", "test"), default="test")
    grid = parser.add_mutually_exclusive_group()
    grid.add_argument("--threshold-grid")
    grid.add_argument("--curve")
    parser.add_argument("--folds", type=int, default=DEFAULT_STATIC_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--adaptation-window", type=int, default=DEFAULT_ADAPTATION_WINDOW)
    parser.add_argument("--evaluation-window", type=int, default=DEFAULT_EVALUATION_WINDOW)
    parser.add_argument("--stride", type=int, default=DEFAULT_CAUSAL_STRIDE)
    parser.add_argument("--matching-rule", choices=("overlap", "centroid"), default="overlap")
    parser.add_argument("--centroid-distance", type=float, default=3.0)
    parser.add_argument("--connectivity", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--oracle-diagnostic",
        action="store_true",
        help="Required acknowledgement that target labels select every reported action",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if not args.oracle_diagnostic:
        raise ValueError(
            "policy_matched_oracle reads target labels; pass --oracle-diagnostic"
        )
    result = run_policy_matched_oracle(
        score_dir=args.score_dir,
        policy=args.policy,
        pixel_budget=args.pixel_budget,
        component_budget=args.component_budget,
        expected_split_role=args.expected_split_role,
        threshold_grid_path=args.threshold_grid,
        curve_path=args.curve,
        folds=args.folds,
        seed=args.seed,
        adaptation_window=args.adaptation_window,
        evaluation_window=args.evaluation_window,
        stride=args.stride,
        matching_rule=args.matching_rule,
        centroid_distance=args.centroid_distance,
        connectivity=args.connectivity,
        min_component_area=args.min_component_area,
    )
    write_json_atomic(args.output, result)
    print(Path(args.output))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ADAPTATION_WINDOW",
    "DEFAULT_CAUSAL_STRIDE",
    "DEFAULT_EVALUATION_WINDOW",
    "DEFAULT_SEED",
    "DEFAULT_STATIC_FOLDS",
    "DecisionUnit",
    "POLICY_MATCHED_ORACLE_SCHEMA_VERSION",
    "build_argument_parser",
    "build_decision_units",
    "evaluate_policy_oracle",
    "load_policy_grid",
    "main",
    "run_policy_matched_oracle",
]
