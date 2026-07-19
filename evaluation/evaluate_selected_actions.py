"""Evaluate frozen per-image raw-logit actions on a labeled score export.

The formal path is deliberately fail closed.  It binds a frozen zero-label
selection to (1) an integrity-checked labeled count-curve archive and (2) the
verified post-freeze labeled replay of the unlabeled target score export.  It
then recomputes every selected count from the raw logits and masks before
reporting segmentation and object-detection metrics.

Reject is always scored as an all-zero prediction in the all-image metrics.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from certification.build_calibration_losses import (
    LOSS_MODE_BUDGET_VIOLATION,
    build_calibration_losses,
    load_count_curve_archive,
    load_count_curve_provenance,
    validate_formal_protocol,
    verify_count_curve_archive_integrity,
)
from rc_irstd.utils.io import atomic_write_json
from risk_curve.evaluate_zero_label import (
    _actions_for_ids,
    validate_count_curve_binding,
    validate_zero_result_selection_contract,
)
from risk_curve.representation import (
    GRID_DETECTOR_PROTOCOL,
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_PREDICTION_RULE,
    LOGIT_REPRESENTATION,
    canonical_json_sha256,
    empty_action_contract,
    logit_threshold_grid_sha256,
    validate_logit_threshold_grid,
)

from .artifact_integrity import (
    RAW_LOGIT_SCORE_REPRESENTATION,
    file_sha256,
    verify_score_map_directory,
)
from .standard_metrics import (
    FORMAL_MODEL_BACKENDS,
    _require_formal_manifest,
    compute_standard_metrics,
)


SELECTED_ACTION_METRICS_SCHEMA_VERSION = "rc-irstd-selected-action-metrics-v1"
RAW_LOGIT_REPRESENTATION = LOGIT_REPRESENTATION
METRIC_NAMES = (
    "pixel_iou",
    "nIoU",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "object_pd",
    "pixel_fa",
    "component_fa_per_megapixel",
)


def _load_json(path: Path, *, role: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{role} must decode to a JSON object")
    return payload


def _manifest_ids(manifest: Mapping[str, Any]) -> list[str]:
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("score manifest records must be a non-empty list")
    image_ids: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"score manifest record {index} is not an object")
        image_id = record.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError(f"score manifest record {index} has no image_id")
        image_ids.append(image_id)
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("score manifest image IDs are not unique")
    return image_ids


def _normalise_actions(
    zero_result: Mapping[str, Any],
    image_ids: Sequence[str],
    num_thresholds: int,
) -> list[int | None]:
    actions, missing, extras = _actions_for_ids(
        zero_result,
        image_ids,
        allow_unmapped_as_reject=False,
    )
    if missing:
        raise ValueError("selection does not cover every image ID: " + ", ".join(missing[:10]))
    if extras:
        raise ValueError(
            "selection references image IDs absent from the score manifest: "
            + ", ".join(extras[:10])
        )
    normalised: list[int | None] = []
    for value in actions:
        if value is None:
            normalised.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("threshold indices must be integers or null")
        index = int(value)
        if not 0 <= index < num_thresholds:
            raise ValueError("a selected threshold index lies outside the frozen grid")
        normalised.append(index)
    return normalised


def _empty_active_result() -> tuple[dict[str, None], dict[str, int]]:
    return (
        {name: None for name in METRIC_NAMES},
        {
            "num_samples": 0,
            "num_pixels": 0,
            "num_target_pixels": 0,
            "num_predicted_positive_pixels": 0,
            "pixel_tp": 0,
            "pixel_fp": 0,
            "pixel_fn": 0,
            "pixel_tn": 0,
            "num_gt_objects": 0,
            "num_tp_objects": 0,
            "num_fn_objects": 0,
            "num_fp_components": 0,
            "num_predicted_components": 0,
            "num_empty_empty_images": 0,
        },
    )


def _exact_nonnegative_int(value: Any, *, field: str) -> int:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{field} must be scalar")
    scalar = array.item()
    if isinstance(scalar, bool):
        raise ValueError(f"{field} must be a non-negative integer")
    try:
        integer = int(scalar)
        numeric = float(scalar)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a non-negative integer") from error
    if not np.isfinite(numeric) or numeric != integer or integer < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return integer


def _selection_grid(selection: Mapping[str, Any]) -> np.ndarray:
    raw = selection.get("thresholds")
    if not isinstance(raw, list) or not raw:
        raise ValueError("selection lacks a non-empty frozen threshold grid")
    try:
        values64 = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("selection threshold grid is not numeric") from error
    if values64.ndim != 1 or not np.isfinite(values64).all():
        raise ValueError("selection threshold grid must be finite and one-dimensional")
    values32 = np.asarray(values64, dtype=np.float32)
    if not np.array_equal(values64, values32.astype(np.float64)):
        raise ValueError("selection threshold grid is not an exact FP32 grid")
    return validate_logit_threshold_grid(values32)


def _require_equal(owner: str, field: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{owner} {field} differs from the bound count protocol")


def _diagnostic_score_binding(
    selection: Mapping[str, Any], score_integrity: Mapping[str, Any]
) -> dict[str, Any]:
    artifact = selection.get("statistics_artifact")
    provenance = artifact.get("provenance") if isinstance(artifact, Mapping) else None
    if not isinstance(provenance, Mapping):
        return {
            "verified": False,
            "binding_mode": "diagnostic_without_statistics_provenance",
        }
    comparisons = {
        "manifest_sha256": "score_manifest_sha256",
        "records_sha256": "score_records_sha256",
        "ordered_image_ids_sha256": "score_ordered_image_ids_sha256",
        "num_records": "score_num_records",
    }
    for score_field, selection_field in comparisons.items():
        if score_integrity.get(score_field) != provenance.get(selection_field):
            raise ValueError(
                "labeled score artifact and selection statistics differ in "
                + selection_field
            )
    return {
        "verified": True,
        "binding_mode": "diagnostic_identical_score_artifact_without_count_protocol",
    }


def _load_bound_count_contract(
    count_curves: str | Path,
    selection: Mapping[str, Any],
    image_ids: Sequence[str],
    *,
    selection_sha256: str | None,
    pair_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    count_path = Path(count_curves).expanduser().resolve()
    arrays = load_count_curve_archive(count_path, require_integrity=True)
    provenance = load_count_curve_provenance(count_path, require_integrity=True)
    integrity = verify_count_curve_archive_integrity(
        count_path, require_integrity=True
    )
    count_ids = [str(value) for value in np.asarray(arrays["image_ids"]).tolist()]
    if count_ids != list(image_ids):
        raise ValueError(
            "count curves and labeled score manifest have different ordered image IDs"
        )

    try:
        pixel_budget = float(selection["pixel_budget"])
        component_budget = float(selection["component_budget"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "selection must bind finite positive pixel/component budgets"
        ) from error
    validated_losses = build_calibration_losses(
        **arrays,
        pixel_budget=pixel_budget,
        component_budget=component_budget,
        loss_mode=LOSS_MODE_BUDGET_VIOLATION,
    )

    raw_protocol = provenance.get("protocol")
    raw_fingerprint = provenance.get("protocol_fingerprint")
    if not isinstance(raw_fingerprint, str):
        raise ValueError("count provenance lacks a recorded protocol_fingerprint")
    protocol, fingerprint = validate_formal_protocol(
        raw_protocol, raw_fingerprint
    )
    selection_protocol = selection.get("protocol")
    selection_fingerprint = selection.get("protocol_fingerprint")
    if not isinstance(selection_fingerprint, str):
        raise ValueError("selection lacks a recorded protocol_fingerprint")
    validated_selection_protocol, validated_selection_fingerprint = (
        validate_formal_protocol(
            selection_protocol,
            selection_fingerprint,
        )
    )
    if validated_selection_protocol != protocol or validated_selection_fingerprint != fingerprint:
        raise ValueError("selection and count archive use different canonical protocols")

    binding = validate_count_curve_binding(
        selection,
        provenance,
        count_ids,
        pair_audit=pair_audit,
        zero_result_sha256=selection_sha256 if pair_audit is not None else None,
    )

    count_grid = validate_logit_threshold_grid(validated_losses.thresholds)
    selected_grid = _selection_grid(selection)
    if not np.array_equal(selected_grid, count_grid):
        raise ValueError("selection and count archive threshold grids differ")
    semantic_grid_sha = logit_threshold_grid_sha256(count_grid)

    count_contract = {
        "representation": arrays.get("representation"),
        "threshold_grid_schema_version": arrays.get(
            "threshold_grid_schema_version"
        ),
        "threshold_grid_sha256": arrays.get("recorded_threshold_grid_sha256"),
        "threshold_grid_manifest_sha256": arrays.get(
            "threshold_grid_manifest_sha256"
        ),
        "threshold_grid_detector_protocol": arrays.get(
            "threshold_grid_detector_protocol"
        ),
        "threshold_grid_detector_checkpoint_sha256s": arrays.get(
            "threshold_grid_detector_checkpoint_sha256s"
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": arrays.get(
            "threshold_grid_outer_detector_checkpoint_sha256"
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": arrays.get(
            "threshold_grid_episode_detector_checkpoint_sha256s"
        ),
    }
    expected_count_contract = {
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": semantic_grid_sha,
        "threshold_grid_manifest_sha256": provenance.get(
            "threshold_grid_manifest_sha256"
        ),
        "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": protocol.get(
            "threshold_grid_detector_checkpoint_sha256s"
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": protocol.get(
            "threshold_grid_outer_detector_checkpoint_sha256"
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": protocol.get(
            "threshold_grid_episode_detector_checkpoint_sha256s"
        ),
    }
    for field, expected in expected_count_contract.items():
        _require_equal("count archive", field, count_contract.get(field), expected)
        if field in provenance:
            _require_equal("count provenance", field, provenance.get(field), expected)

    selection_contract = {
        **expected_count_contract,
        "prediction_rule": LOGIT_PREDICTION_RULE,
        "empty_action": empty_action_contract(),
    }
    for owner, payload in (
        ("selection", selection),
        ("selection_data_contract", selection.get("selection_data_contract")),
    ):
        if not isinstance(payload, Mapping):
            raise ValueError(f"{owner} must be an object")
        for field, expected in selection_contract.items():
            _require_equal(owner, field, payload.get(field), expected)

    protocol_arguments = {
        "matching_rule": protocol["matching_rule"],
        "centroid_distance": float(protocol["centroid_distance"]),
        "connectivity": int(protocol["connectivity"]),
        "min_component_area": int(protocol["min_component_area"]),
    }
    if protocol_arguments["min_component_area"] != 1:
        raise ValueError(
            "selected-action v1 requires min_component_area=1 so pixel-FP and "
            "count-curve semantics are identical"
        )
    return {
        "path": count_path,
        "arrays": arrays,
        "provenance": provenance,
        "integrity": integrity,
        "binding": binding,
        "thresholds": count_grid,
        "threshold_grid_sha256": semantic_grid_sha,
        "protocol": protocol,
        "protocol_arguments": protocol_arguments,
    }


def _assert_selected_count_consistency(
    *,
    image_id: str,
    row_index: int,
    threshold_index: int | None,
    single_counts: Mapping[str, Any],
    count_arrays: Mapping[str, Any],
) -> dict[str, Any]:
    if "tp_object_counts" not in count_arrays or "gt_object_counts" not in count_arrays:
        raise ValueError(
            "formal selected-action evaluation requires TP/GT object count curves"
        )
    if threshold_index is None:
        expected_fp_pixels = 0
        expected_fp_components = 0
        expected_tp_objects = 0
    else:
        expected_fp_pixels = _exact_nonnegative_int(
            np.asarray(count_arrays["false_positive_pixels"])[row_index, threshold_index],
            field=f"false_positive_pixels[{image_id}]",
        )
        expected_fp_components = _exact_nonnegative_int(
            np.asarray(count_arrays["false_positive_components"])[
                row_index, threshold_index
            ],
            field=f"false_positive_components[{image_id}]",
        )
        expected_tp_objects = _exact_nonnegative_int(
            np.asarray(count_arrays["tp_object_counts"])[row_index, threshold_index],
            field=f"tp_object_counts[{image_id}]",
        )
    expected = {
        "pixel_fp": expected_fp_pixels,
        "num_fp_components": expected_fp_components,
        "num_tp_objects": expected_tp_objects,
        "num_gt_objects": _exact_nonnegative_int(
            np.asarray(count_arrays["gt_object_counts"])[row_index],
            field=f"gt_object_counts[{image_id}]",
        ),
        "num_pixels": _exact_nonnegative_int(
            np.asarray(count_arrays["total_pixels"])[row_index],
            field=f"total_pixels[{image_id}]",
        ),
    }
    actual = {field: int(single_counts[field]) for field in expected}
    if actual != expected:
        differences = [
            f"{field}: actual={actual[field]}, expected={expected[field]}"
            for field in expected
            if actual[field] != expected[field]
        ]
        raise ValueError(
            f"selected-action count consistency mismatch for {image_id}: "
            + "; ".join(differences)
        )
    return {"verified": True, "actual": actual, "expected": expected}


def evaluate_selected_actions(
    score_dir: str | Path,
    selection: Mapping[str, Any],
    *,
    count_curves: str | Path | None = None,
    selection_sha256: str | None = None,
    target_stage_pair_audit: Mapping[str, Any] | None = None,
    representation: str = RAW_LOGIT_REPRESENTATION,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
    reject_as_empty: bool = True,
    allow_unverified_diagnostic: bool = False,
) -> dict[str, Any]:
    if representation != RAW_LOGIT_REPRESENTATION:
        raise ValueError("selected-action formal evaluation requires raw_logit_float32")
    if not reject_as_empty:
        raise ValueError("formal selected-action evaluation requires reject_as_empty=true")
    if not allow_unverified_diagnostic:
        if count_curves is None:
            raise ValueError("formal selected-action evaluation requires count_curves")
        if target_stage_pair_audit is None:
            raise ValueError(
                "formal selected-action evaluation requires target_stage_pair_audit"
            )
        if selection_sha256 is None:
            raise ValueError("formal selected-action evaluation requires selection_sha256")

    try:
        selection_contract_audit = validate_zero_result_selection_contract(selection)
    except ValueError as error:
        if not allow_unverified_diagnostic:
            raise
        selection_contract_audit = {
            "verified": False,
            "error": str(error),
            "adaptation_protocol": selection.get("adaptation_protocol"),
        }

    root = Path(score_dir).expanduser().resolve()
    manifest, paths, score_integrity = verify_score_map_directory(
        root,
        require_integrity=True,
        require_masks=True,
    )
    if not isinstance(manifest, Mapping):
        raise ValueError("selected-action evaluation requires a formal score manifest")
    _require_formal_manifest(manifest)
    if manifest.get("model_backend") not in FORMAL_MODEL_BACKENDS:
        raise ValueError("selected-action evaluation received an unsupported model backend")
    raw_contract = {
        "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
        "probability_dtype": "float32",
        "logit_dtype": "float32",
        "probability_transform": "sigmoid",
        "probability_clipping": "none",
        "inference_autocast_enabled": False,
    }
    for field, expected in raw_contract.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"selected-action raw-logit manifest requires {field}={expected!r}"
            )

    image_ids = _manifest_ids(manifest)
    count_contract: dict[str, Any] | None = None
    if count_curves is not None:
        count_contract = _load_bound_count_contract(
            count_curves,
            selection,
            image_ids,
            selection_sha256=selection_sha256,
            pair_audit=target_stage_pair_audit,
        )
        protocol_arguments = count_contract["protocol_arguments"]
        requested_arguments = {
            "matching_rule": matching_rule,
            "centroid_distance": float(centroid_distance),
            "connectivity": int(connectivity),
            "min_component_area": int(min_component_area),
        }
        if requested_arguments != protocol_arguments:
            raise ValueError(
                "selected-action metric arguments differ from the bound count protocol"
            )
        thresholds = count_contract["thresholds"]
        score_binding_audit = count_contract["binding"]
    else:
        thresholds = _selection_grid(selection)
        score_binding_audit = _diagnostic_score_binding(selection, score_integrity)

    actions = _normalise_actions(selection, image_ids, int(thresholds.size))
    predictions: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    active_predictions: list[np.ndarray] = []
    active_masks: list[np.ndarray] = []
    selected_actions: list[dict[str, Any]] = []
    per_image_count_audits: list[dict[str, Any]] = []
    for row_index, (image_id, path, action) in enumerate(zip(image_ids, paths, actions)):
        with np.load(path, allow_pickle=False) as payload:
            if not {"image_id", "logit", "mask", "labels_loaded"}.issubset(payload):
                raise ValueError(f"selected-action score record is incomplete: {path}")
            embedded_id = str(np.asarray(payload["image_id"]).item())
            if embedded_id != image_id:
                raise ValueError(
                    f"score record image ID {embedded_id!r} differs from manifest {image_id!r}"
                )
            if not bool(np.asarray(payload["labels_loaded"]).item()):
                raise ValueError(f"selected-action score record is label-free: {path}")
            logits = np.asarray(payload["logit"])
            mask = np.asarray(payload["mask"])
        if logits.dtype != np.float32 or logits.ndim != 2 or not np.isfinite(logits).all():
            raise ValueError(f"raw logits must be finite 2-D float32: {path}")
        if mask.shape != logits.shape:
            raise ValueError(f"raw logits and mask shapes differ: {path}")
        target = np.ascontiguousarray(mask > 0, dtype=np.uint8)
        if action is None:
            prediction = np.zeros_like(target, dtype=np.float32)
            threshold: float | None = None
            action_name = "no_detection_reject"
        else:
            threshold = float(thresholds[action])
            prediction = np.ascontiguousarray(logits >= threshold, dtype=np.float32)
            active_predictions.append(prediction)
            active_masks.append(target)
            action_name = "threshold"
        predictions.append(prediction)
        masks.append(target)

        count_audit: dict[str, Any] | None = None
        if count_contract is not None:
            single = compute_standard_metrics(
                [prediction],
                [target],
                0.5,
                matching_rule=matching_rule,
                centroid_distance=centroid_distance,
                connectivity=connectivity,
                min_component_area=min_component_area,
            )
            count_audit = _assert_selected_count_consistency(
                image_id=image_id,
                row_index=row_index,
                threshold_index=action,
                single_counts=single["counts"],
                count_arrays=count_contract["arrays"],
            )
            per_image_count_audits.append(
                {"image_id": image_id, "threshold_index": action, **count_audit}
            )
        selected_actions.append(
            {
                "image_id": image_id,
                "threshold_index": action,
                "threshold": threshold,
                "action": action_name,
                "count_consistency_verified": bool(
                    count_audit and count_audit.get("verified") is True
                ),
            }
        )

    all_result = compute_standard_metrics(
        predictions,
        masks,
        0.5,
        matching_rule=matching_rule,
        centroid_distance=centroid_distance,
        connectivity=connectivity,
        min_component_area=min_component_area,
    )
    if active_predictions:
        active_result = compute_standard_metrics(
            active_predictions,
            active_masks,
            0.5,
            matching_rule=matching_rule,
            centroid_distance=centroid_distance,
            connectivity=connectivity,
            min_component_area=min_component_area,
        )
        metrics_active = active_result["metrics"]
        counts_active = active_result["counts"]
    else:
        metrics_active, counts_active = _empty_active_result()
    active_count = len(active_predictions)
    num_images = len(image_ids)
    coverage = float(active_count / num_images)
    count_consistency_verified = bool(
        count_contract is not None and len(per_image_count_audits) == num_images
    )
    formal_eligible = bool(
        not allow_unverified_diagnostic
        and selection_contract_audit.get("verified") is True
        and score_binding_audit.get("verified") is True
        and score_integrity.get("verified") is True
        and count_consistency_verified
    )
    selected_actions_sha256 = canonical_json_sha256(selected_actions)

    return {
        "schema_version": SELECTED_ACTION_METRICS_SCHEMA_VERSION,
        "mode": "independent_labeled_audit_of_frozen_selected_actions",
        "formal_protocol_eligible": formal_eligible,
        "metrics_all": all_result["metrics"],
        "metrics_active_only": metrics_active,
        "counts_all": all_result["counts"],
        "counts_active_only": counts_active,
        "num_images": num_images,
        "active_image_count": active_count,
        "rejected_image_count": num_images - active_count,
        "coverage_rate": coverage,
        "reject_rate": float(1.0 - coverage),
        "selection_sha256": selection_sha256,
        "score_manifest_sha256": score_integrity.get("manifest_sha256"),
        "score_records_sha256": score_integrity.get("records_sha256"),
        "ordered_image_ids_sha256": score_integrity.get("ordered_image_ids_sha256"),
        "count_curves": str(count_contract["path"]) if count_contract else None,
        "count_curves_sha256": (
            count_contract["integrity"].get("file_sha256") if count_contract else None
        ),
        "count_curves_file_sha256": (
            count_contract["integrity"].get("file_sha256") if count_contract else None
        ),
        "count_archive_payload_sha256": (
            count_contract["integrity"].get("payload_sha256")
            if count_contract
            else None
        ),
        "threshold_grid_sha256": (
            count_contract["threshold_grid_sha256"]
            if count_contract
            else logit_threshold_grid_sha256(thresholds)
        ),
        "action_coverage_verified": True,
        "selected_count_consistency_verified": count_consistency_verified,
        "selected_actions_sha256": selected_actions_sha256,
        "formal_empirical_evaluation_eligible": formal_eligible,
        "formal_crc_eligible": bool(
            selection_contract_audit.get("formal_crc_eligible", False)
        ),
        "selection_contract_audit": selection_contract_audit,
        "score_binding_audit": score_binding_audit,
        "per_image_count_audits": per_image_count_audits,
        "selected_actions": selected_actions,
        "protocol": {
            "representation": representation,
            "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
            "prediction_rule": LOGIT_PREDICTION_RULE,
            "reject_rule": "reject is scored as an all-zero prediction",
            "empty_action": empty_action_contract(),
            "reject_included_in_all_metrics": True,
            "model_backend": manifest.get("model_backend"),
            "matching_rule": matching_rule,
            "centroid_distance": float(centroid_distance),
            "connectivity": int(connectivity),
            "min_component_area": int(min_component_area),
            "pixel_iou_definition": "sum_i TP_i / sum_i(TP_i+FP_i+FN_i)",
            "nIoU_definition": "mean_i(TP_i/(TP_i+FP_i+FN_i))",
            "empty_prediction_empty_target_nIoU": 0.0,
        },
        "provenance": {
            "score_dir": str(root),
            "detector_weight_sha256": manifest.get("weight_sha256"),
            "target_dataset": manifest.get("target_dataset"),
            "split_file_sha256": manifest.get("split_file_sha256"),
            "protocol_fingerprint": (
                count_contract["provenance"].get("protocol_fingerprint")
                if count_contract
                else selection.get("protocol_fingerprint")
            ),
            "threshold_grid_manifest_sha256": (
                count_contract["arrays"].get("threshold_grid_manifest_sha256")
                if count_contract
                else selection.get("threshold_grid_manifest_sha256")
            ),
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument(
        "--count-curves",
        help="Integrity-checked labeled count curves bound to the frozen grid",
    )
    parser.add_argument(
        "--target-stage-pair-audit",
        help="Verified unlabeled/labeled score-pair audit for physically separated stages",
    )
    parser.add_argument(
        "--representation",
        default=RAW_LOGIT_REPRESENTATION,
        choices=[RAW_LOGIT_REPRESENTATION],
    )
    parser.add_argument("--matching-rule", choices=["overlap", "centroid"], default="overlap")
    parser.add_argument("--centroid-distance", type=float, default=3.0)
    parser.add_argument("--connectivity", type=int, choices=[1, 2], default=2)
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument("--reject-as-empty", action="store_true")
    parser.add_argument("--allow-unverified-diagnostic", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if not args.allow_unverified_diagnostic and not args.count_curves:
        raise ValueError("formal selected-action CLI requires --count-curves")
    if not args.allow_unverified_diagnostic and not args.target_stage_pair_audit:
        raise ValueError(
            "formal selected-action CLI requires --target-stage-pair-audit"
        )
    selection_path = Path(args.selection).expanduser().resolve()
    selection = _load_json(selection_path, role="selection")
    selection_sha256 = file_sha256(selection_path)
    pair_audit = (
        _load_json(
            Path(args.target_stage_pair_audit).expanduser().resolve(),
            role="target-stage pair audit",
        )
        if args.target_stage_pair_audit
        else None
    )
    result = evaluate_selected_actions(
        args.score_dir,
        selection,
        count_curves=args.count_curves,
        selection_sha256=selection_sha256,
        target_stage_pair_audit=pair_audit,
        representation=args.representation,
        matching_rule=args.matching_rule,
        centroid_distance=args.centroid_distance,
        connectivity=args.connectivity,
        min_component_area=args.min_component_area,
        reject_as_empty=args.reject_as_empty,
        allow_unverified_diagnostic=args.allow_unverified_diagnostic,
    )
    result["selection"] = str(selection_path)
    result["target_stage_pair_audit"] = (
        str(Path(args.target_stage_pair_audit).expanduser().resolve())
        if args.target_stage_pair_audit
        else None
    )
    result["target_stage_pair_audit_sha256"] = (
        file_sha256(args.target_stage_pair_audit)
        if args.target_stage_pair_audit
        else None
    )
    atomic_write_json(args.output, result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "RAW_LOGIT_REPRESENTATION",
    "SELECTED_ACTION_METRICS_SCHEMA_VERSION",
    "build_argument_parser",
    "evaluate_selected_actions",
    "main",
]
