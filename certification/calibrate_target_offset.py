"""Calibrate a non-negative threshold-grid rank offset on a target domain.

The CLI consumes per-image count curves for a labelled calibration split, an
independent test split (for mandatory ID leakage checks only), and a zero-label
selection JSON.  Test labels and test count curves are never used to choose the
offset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .build_calibration_losses import (
    COMPONENT_BUDGET_UNIT,
    LOSS_SCHEMA_VERSION,
    RAW_LOGIT_LOSS_SCHEMA_VERSION,
    LOSS_MODE_BUDGET_VIOLATION,
    PIXEL_BUDGET_UNIT,
    CalibrationLosses,
    build_calibration_losses,
    load_count_curve_archive,
    load_count_curve_provenance,
    score_map_protocol,
    validate_formal_protocol,
    verify_count_curve_archive_integrity,
)
from evaluation.artifact_integrity import ordered_ids_sha256, verify_score_map_directory
from .conformal_offset import OffsetSelection, select_conformal_offset
from risk_curve.build_deployment_statistics import (
    DEPLOYMENT_STATISTICS_SCHEMA_VERSION,
    LOGIT_DEPLOYMENT_STATISTICS_SCHEMA_VERSION,
)
from risk_curve.deployment_contract import audit_checkpoint_deployment_contract
from risk_curve.domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    STATISTICS_SCHEMA_VERSION,
)
from risk_curve.representation import (
    GRID_DETECTOR_PROTOCOL,
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_PREDICTION_RULE,
    LOGIT_REPRESENTATION,
    PROBABILITY_REPRESENTATION,
    empty_action_contract,
    logit_threshold_grid_sha256,
    validate_logit_threshold_grid,
)
from risk_curve.threshold_grid import threshold_grid_sha256


RESULT_SCHEMA_VERSION = "rc-v2-target-offset-result-v2"
RAW_LOGIT_RESULT_SCHEMA_VERSION = "rc-v4-target-offset-result-v1-raw-logit"
ZERO_RESULT_SCHEMA_VERSION = "rc-v2-zero-label-result-v2-formal"
SELECTION_DATA_CONTRACT_SCHEMA_VERSION = "rc-v2-zero-selection-data-v1"
FORMAL_COUNT_CURVE_SOURCE_TYPE = "exported_score_map_directory"


def _ids(image_ids: Sequence[str] | np.ndarray, split_name: str) -> tuple[str, ...]:
    values = tuple(str(value) for value in np.asarray(image_ids).reshape(-1).tolist())
    if not values or any(not value for value in values):
        raise ValueError(f"{split_name} image IDs must be non-empty strings")
    unique, counts = np.unique(np.asarray(values), return_counts=True)
    duplicates = unique[counts > 1]
    if duplicates.size:
        preview = ", ".join(duplicates[:5].tolist())
        raise ValueError(f"Duplicate IDs within {split_name} split: {preview}")
    return values


def assert_disjoint_image_ids(
    calibration_ids: Sequence[str] | np.ndarray,
    test_ids: Sequence[str] | np.ndarray,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate uniqueness within each split and disjointness across splits."""

    calibration = _ids(calibration_ids, "calibration")
    test = _ids(test_ids, "test")
    overlap = sorted(set(calibration).intersection(test))
    if overlap:
        preview = ", ".join(overlap[:5])
        raise ValueError(f"Calibration/test ID leakage detected: {preview}")
    return calibration, test


def assert_three_way_disjoint_image_ids(
    adaptation_ids: Sequence[str] | np.ndarray,
    calibration_ids: Sequence[str] | np.ndarray,
    test_ids: Sequence[str] | np.ndarray,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Validate the formal adaptation/calibration/future-test partition."""

    adaptation = _ids(adaptation_ids, "adaptation")
    calibration, test = assert_disjoint_image_ids(calibration_ids, test_ids)
    for left_name, left, right_name, right in (
        ("adaptation", adaptation, "calibration", calibration),
        ("adaptation", adaptation, "test", test),
    ):
        overlap = sorted(set(left).intersection(right))
        if overlap:
            preview = ", ".join(overlap[:5])
            raise ValueError(
                f"{left_name}/{right_name} ID leakage detected: {preview}"
            )
    return adaptation, calibration, test


def _ids_digest(values: Sequence[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _grid_digest(thresholds: np.ndarray) -> str:
    return threshold_grid_sha256(np.asarray(thresholds, dtype=np.float32))


def _semantic_grid_digest(thresholds: np.ndarray, representation: str) -> str:
    if representation == LOGIT_REPRESENTATION:
        return logit_threshold_grid_sha256(
            validate_logit_threshold_grid(np.asarray(thresholds))
        )
    if representation == PROBABILITY_REPRESENTATION:
        return _grid_digest(thresholds)
    raise ValueError(f"Unsupported score representation: {representation!r}")


def _sigmoid_display(value: float) -> float:
    number = float(value)
    if number >= 0.0:
        return float(1.0 / (1.0 + np.exp(-number)))
    exponential = float(np.exp(number))
    return float(exponential / (1.0 + exponential))


def _file_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _same_grid(left: np.ndarray, right: np.ndarray) -> bool:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=0.0, atol=1e-8))


def _load_test_identity(
    path: str | Path,
    *,
    require_integrity: bool = False,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
    """Read only test IDs, the public grid, and its top-level contract.

    In particular, this function deliberately does not load any test false-
    positive or true-positive count arrays.  The returned contract contains
    only representation/grid/checkpoint identities needed to fail closed
    before a formal calibration result is written.
    """

    verify_count_curve_archive_integrity(
        path, require_integrity=require_integrity
    )
    with np.load(Path(path), allow_pickle=False) as archive:
        id_key = "image_ids" if "image_ids" in archive else "ids"
        grid_key = "thresholds" if "thresholds" in archive else "threshold_grid"
        if id_key not in archive or grid_key not in archive:
            raise ValueError("Test archive must contain image_ids and thresholds")
        image_ids = np.asarray(archive[id_key])
        thresholds = np.asarray(archive[grid_key])
        representation = (
            str(np.asarray(archive["representation"]).item())
            if "representation" in archive
            else PROBABILITY_REPRESENTATION
        )
        contract: dict[str, Any] = {"representation": representation}
        for field in (
            "threshold_grid_schema_version",
            "threshold_grid_sha256",
            "threshold_grid_manifest_sha256",
            "threshold_grid_detector_protocol",
            "threshold_grid_outer_detector_checkpoint_sha256",
        ):
            if field in archive:
                raw = np.asarray(archive[field])
                if raw.ndim != 0:
                    raise ValueError(f"Test archive {field} must be a scalar string")
                contract[field] = str(raw.item())
        for field in (
            "threshold_grid_detector_checkpoint_sha256s",
            "threshold_grid_episode_detector_checkpoint_sha256s",
        ):
            if field in archive:
                raw = np.asarray(archive[field])
                if raw.ndim != 1:
                    raise ValueError(f"Test archive {field} must be one-dimensional")
                contract[field] = [str(value) for value in raw.tolist()]
    return image_ids, thresholds, representation, contract


_RAW_COUNT_BINDING_FIELDS = (
    "representation",
    "threshold_grid_schema_version",
    "threshold_grid_sha256",
    "threshold_grid_manifest_sha256",
    "threshold_grid_detector_protocol",
    "threshold_grid_detector_checkpoint_sha256s",
    "threshold_grid_outer_detector_checkpoint_sha256",
    "threshold_grid_episode_detector_checkpoint_sha256s",
)


def _raw_count_archive_provenance_binding(
    archive_contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
    thresholds: np.ndarray,
    *,
    split_name: str,
) -> dict[str, Any]:
    """Bind a raw-logit count archive's public contract to provenance exactly."""

    if archive_contract.get("representation") != LOGIT_REPRESENTATION:
        return {
            "required": False,
            "verified": True,
            "representation": archive_contract.get("representation"),
            "errors": {},
        }

    errors: dict[str, str] = {}
    expected_semantic = logit_threshold_grid_sha256(
        validate_logit_threshold_grid(np.asarray(thresholds))
    )
    if archive_contract.get("threshold_grid_schema_version") != LOGIT_GRID_SCHEMA_VERSION:
        errors["threshold_grid_schema_version"] = (
            f"{split_name} archive does not use the v4 raw-logit grid schema"
        )
    if archive_contract.get("threshold_grid_sha256") != expected_semantic:
        errors["threshold_grid_sha256"] = (
            f"{split_name} archive semantic grid hash differs from its FP32 grid"
        )
    try:
        _sha256_text(
            archive_contract.get("threshold_grid_manifest_sha256"),
            f"{split_name} archive grid-manifest sha256",
        )
    except ValueError as error:
        errors["threshold_grid_manifest_sha256"] = str(error)
    if archive_contract.get("threshold_grid_detector_protocol") != GRID_DETECTOR_PROTOCOL:
        errors["threshold_grid_detector_protocol"] = (
            f"{split_name} archive detector protocol is unsupported"
        )

    all_hashes = archive_contract.get("threshold_grid_detector_checkpoint_sha256s")
    outer_hash = archive_contract.get(
        "threshold_grid_outer_detector_checkpoint_sha256"
    )
    episode_hashes = archive_contract.get(
        "threshold_grid_episode_detector_checkpoint_sha256s"
    )
    if not isinstance(all_hashes, list) or not all_hashes:
        errors["threshold_grid_detector_checkpoint_sha256s"] = (
            f"{split_name} archive detector hashes must be a non-empty list"
        )
    elif len(set(all_hashes)) != len(all_hashes):
        errors["threshold_grid_detector_checkpoint_sha256s"] = (
            f"{split_name} archive detector hashes must be distinct"
        )
    else:
        try:
            for position, value in enumerate(all_hashes):
                _sha256_text(value, f"{split_name} detector hash {position}")
        except ValueError as error:
            errors["threshold_grid_detector_checkpoint_sha256s"] = str(error)
    if not isinstance(episode_hashes, list) or not episode_hashes:
        errors["threshold_grid_episode_detector_checkpoint_sha256s"] = (
            f"{split_name} archive episode-detector hashes must be a non-empty list"
        )
    elif len(set(episode_hashes)) != len(episode_hashes):
        errors["threshold_grid_episode_detector_checkpoint_sha256s"] = (
            f"{split_name} archive episode-detector hashes must be distinct"
        )
    else:
        try:
            for position, value in enumerate(episode_hashes):
                _sha256_text(value, f"{split_name} episode detector hash {position}")
        except ValueError as error:
            errors["threshold_grid_episode_detector_checkpoint_sha256s"] = str(error)
    try:
        checked_outer = _sha256_text(
            outer_hash, f"{split_name} outer detector hash"
        )
    except ValueError as error:
        checked_outer = None
        errors["threshold_grid_outer_detector_checkpoint_sha256"] = str(error)
    if (
        isinstance(all_hashes, list)
        and isinstance(episode_hashes, list)
        and checked_outer is not None
        and (
            checked_outer in episode_hashes
            or set(all_hashes) != set(episode_hashes).union({checked_outer})
        )
    ):
        errors["detector_roles"] = (
            f"{split_name} detector hashes must equal distinct inner hashes plus "
            "one disjoint outer hash"
        )

    field_matches: dict[str, bool] = {}
    for field in _RAW_COUNT_BINDING_FIELDS:
        field_matches[field] = archive_contract.get(field) == provenance.get(field)
        if not field_matches[field]:
            errors[f"provenance.{field}"] = (
                f"{split_name} archive top-level {field} differs from provenance"
            )
    return {
        "required": True,
        "verified": not errors,
        "representation": LOGIT_REPRESENTATION,
        "semantic_grid_sha256": expected_semantic,
        "field_matches": field_matches,
        "errors": errors,
    }


def _selection_assumptions() -> list[str]:
    return [
        "calibration and future test images are exchangeable for the bounded loss",
        "adaptation, calibration, and future-test image IDs are unique and pairwise disjoint",
        "the detector, score-map preprocessing, budgets, alpha, and threshold grid were fixed before target calibration",
        "candidate actions are non-negative ranks on one fixed threshold grid",
        "per-image joint loss is bounded in [0, 1] and non-increasing over rank",
        "component risk uses a conservative per-image suffix-max envelope",
        "the terminal action emits no detections and is reported as rejection, not as a certified threshold",
    ]


def _adaptive_base_indices(
    values: Sequence[int | None] | np.ndarray,
    *,
    expected_count: int,
    num_thresholds: int,
    split_name: str,
) -> np.ndarray:
    """Map a zero-stage reject (None) to the terminal base index T."""

    raw = np.asarray(values, dtype=object).reshape(-1)
    if raw.shape != (expected_count,):
        raise ValueError(f"{split_name}_zero_indices have the wrong shape")
    converted: list[int] = []
    for value in raw.tolist():
        if value is None:
            converted.append(num_thresholds)
            continue
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{split_name}_zero_indices must contain integers or null")
        index = int(value)
        if not 0 <= index < num_thresholds:
            raise ValueError(f"{split_name}_zero_indices contain an invalid grid index")
        converted.append(index)
    return np.asarray(converted, dtype=np.int64)


def calibrate_target_offset(
    calibration_losses: CalibrationLosses,
    *,
    alpha: float,
    test_image_ids: Sequence[str] | np.ndarray,
    zero_index: int | None = None,
    calibration_zero_indices: Sequence[int | None] | np.ndarray | None = None,
    test_zero_indices: Sequence[int | None] | np.ndarray | None = None,
    adaptation_image_ids: Sequence[str] | np.ndarray | None = None,
    candidate_offset_ranks: Sequence[int] | None = None,
    curve_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Select an offset and return a JSON-serialisable audit record."""

    representation = calibration_losses.representation
    if representation == LOGIT_REPRESENTATION:
        if not isinstance(curve_checkpoint_sha256, str) or (
            len(curve_checkpoint_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in curve_checkpoint_sha256
            )
        ):
            raise ValueError(
                "Raw-logit CRC must bind a curve checkpoint SHA-256"
            )
    elif representation != PROBABILITY_REPRESENTATION:
        raise ValueError(f"Unsupported score representation: {representation!r}")
    if adaptation_image_ids is None:
        calibration_ids, test_ids = assert_disjoint_image_ids(
            calibration_losses.image_ids, test_image_ids
        )
        adaptation_ids: tuple[str, ...] = ()
    else:
        adaptation_ids, calibration_ids, test_ids = assert_three_way_disjoint_image_ids(
            adaptation_image_ids, calibration_losses.image_ids, test_image_ids
        )
    if (zero_index is None) == (calibration_zero_indices is None):
        raise ValueError("Provide exactly one of zero_index or calibration_zero_indices")
    if calibration_zero_indices is None:
        calibration_bases = np.full(
            calibration_losses.num_images, int(zero_index), dtype=np.int64
        )
        test_bases = np.full(len(test_ids), int(zero_index), dtype=np.int64)
        adaptation_mode = "global_zero_threshold"
    else:
        if test_zero_indices is None:
            raise ValueError("test_zero_indices are required for sample-adaptive calibration")
        calibration_bases = _adaptive_base_indices(
            calibration_zero_indices,
            expected_count=calibration_losses.num_images,
            num_thresholds=calibration_losses.num_thresholds,
            split_name="calibration",
        )
        test_bases = _adaptive_base_indices(
            test_zero_indices,
            expected_count=len(test_ids),
            num_thresholds=calibration_losses.num_thresholds,
            split_name="test",
        )
        adaptation_mode = "sample_adaptive_zero_plus_shared_offset"
    selection: OffsetSelection = select_conformal_offset(
        calibration_losses.joint_loss,
        zero_index=int(zero_index) if calibration_zero_indices is None else None,
        zero_indices=calibration_bases if calibration_zero_indices is not None else None,
        alpha=alpha,
        candidate_offset_ranks=candidate_offset_ranks,
        include_terminal_reject=True,
    )
    selected_index = selection.selected_threshold_index
    selected_threshold = (
        float(calibration_losses.thresholds[selected_index])
        if selected_index is not None
        else None
    )
    selected_probability_threshold = (
        _sigmoid_display(selected_threshold)
        if representation == LOGIT_REPRESENTATION
        and selected_threshold is not None
        else selected_threshold
    )
    test_selected_indices: list[int | None] = []
    if selection.success and selection.offset_rank is not None:
        for base_index in np.asarray(test_bases, dtype=np.int64).tolist():
            candidate = int(base_index) + int(selection.offset_rank)
            test_selected_indices.append(
                candidate if candidate < calibration_losses.num_thresholds else None
            )
    test_reject_rate = (
        float(np.mean([value is None for value in test_selected_indices]))
        if test_selected_indices
        else 1.0
    )
    loss_definition = calibration_losses.metadata()["loss_definition"]
    raw_logit_mode = representation == LOGIT_REPRESENTATION
    result: dict[str, Any] = {
        "schema_version": (
            RAW_LOGIT_RESULT_SCHEMA_VERSION
            if raw_logit_mode
            else RESULT_SCHEMA_VERSION
        ),
        "mode": (
            f"few_shot_{adaptation_mode}_grid_rank_crc"
            if selection.success
            else "rejected_no_operating_point"
        ),
        "adaptation_mode": adaptation_mode,
        "formal_artifact_chain_verified": False,
        "success": selection.success,
        "reject": selection.reject,
        "reason": selection.reason,
        "zero_threshold_index": int(zero_index) if zero_index is not None else None,
        "zero_threshold": (
            float(calibration_losses.thresholds[int(zero_index)])
            if zero_index is not None
            else None
        ),
        "zero_logit_threshold": (
            float(calibration_losses.thresholds[int(zero_index)])
            if raw_logit_mode and zero_index is not None
            else None
        ),
        "zero_probability_threshold": (
            _sigmoid_display(
                float(calibration_losses.thresholds[int(zero_index)])
            )
            if raw_logit_mode and zero_index is not None
            else None
        ),
        "calibration_zero_threshold_indices": [
            int(value) if value < calibration_losses.num_thresholds else None
            for value in calibration_bases
        ],
        "test_zero_threshold_indices": [
            int(value) if value < calibration_losses.num_thresholds else None
            for value in np.asarray(test_bases, dtype=int)
        ],
        "selected_threshold_index": selected_index,
        "selected_threshold": selected_threshold,
        "selected_logit_threshold": (
            selected_threshold if raw_logit_mode else None
        ),
        "selected_probability_threshold": selected_probability_threshold,
        "offset_rank": selection.offset_rank,
        "selected_test_threshold_indices": test_selected_indices,
        "selected_test_logit_thresholds": (
            [
                (
                    float(calibration_losses.thresholds[index])
                    if index is not None
                    else None
                )
                for index in test_selected_indices
            ]
            if raw_logit_mode
            else None
        ),
        "selected_test_probability_thresholds": (
            [
                (
                    _sigmoid_display(
                        float(calibration_losses.thresholds[index])
                    )
                    if index is not None
                    else None
                )
                for index in test_selected_indices
            ]
            if raw_logit_mode
            else None
        ),
        "calibration_reject_rate": (
            selection.candidate_trace[-1]["reject_rate"]
            if selection.candidate_trace and selection.success
            else 1.0
        ),
        "test_reject_rate": test_reject_rate,
        "alpha": float(alpha),
        "budgets": {
            "pixel": {
                "value": calibration_losses.pixel_budget,
                "unit": PIXEL_BUDGET_UNIT,
            },
            "component": {
                "value": calibration_losses.component_budget,
                "unit": COMPONENT_BUDGET_UNIT,
            },
        },
        "loss": {
            "schema_version": (
                RAW_LOGIT_LOSS_SCHEMA_VERSION
                if raw_logit_mode
                else LOSS_SCHEMA_VERSION
            ),
            "mode": calibration_losses.loss_mode,
            "definition": loss_definition,
            "range": [0.0, 1.0],
            "maps_directly_to_joint_bsr": (
                calibration_losses.loss_mode == LOSS_MODE_BUDGET_VIOLATION
            ),
        },
        "selection": selection.to_dict(),
        "thresholds": calibration_losses.thresholds.tolist(),
        "representation": representation,
        "threshold_grid_schema_version": (
            calibration_losses.threshold_grid_schema_version
        ),
        "threshold_grid_sha256": calibration_losses.threshold_grid_sha256_value,
        "threshold_grid_manifest_sha256": (
            calibration_losses.threshold_grid_manifest_sha256
        ),
        "threshold_grid_detector_protocol": (
            calibration_losses.threshold_grid_detector_protocol
        ),
        "threshold_grid_detector_checkpoint_sha256s": list(
            calibration_losses.threshold_grid_detector_checkpoint_sha256s
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": (
            calibration_losses.threshold_grid_outer_detector_checkpoint_sha256
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": list(
            calibration_losses.threshold_grid_episode_detector_checkpoint_sha256s
        ),
        "curve_checkpoint_sha256": curve_checkpoint_sha256,
        "prediction_rule": (
            LOGIT_PREDICTION_RULE
            if raw_logit_mode
            else "prediction = (sigmoid_probability >= threshold)"
        ),
        "empty_action": (
            empty_action_contract()
            if raw_logit_mode
            else {"threshold": 1.0, "threshold_index": None}
        ),
        "calibration_image_ids": list(calibration_ids),
        "test_image_ids_checked": list(test_ids),
        "adaptation_image_ids": list(adaptation_ids),
        "split_audit": {
            "adaptation_count": len(adaptation_ids),
            "calibration_count": len(calibration_ids),
            "test_count": len(test_ids),
            "overlap_count": 0,
            "three_way_disjoint_verified": bool(adaptation_ids),
            "adaptation_ids_sha256": (
                _ids_digest(adaptation_ids) if adaptation_ids else None
            ),
            "calibration_ids_sha256": _ids_digest(calibration_ids),
            "test_ids_sha256": _ids_digest(test_ids),
        },
        "assumptions": calibration_losses.assumptions() + _selection_assumptions(),
    }
    if selection.success:
        result["guarantee_scope"] = (
            "none: the complete detector/protocol/zero-statistics artifact chain "
            "was not supplied to this API call"
        )
    else:
        result["guarantee_scope"] = (
            "none: no non-reject threshold-grid action passed the finite-sample check"
        )
    return result


def _budget(
    explicit: float | None, zero_result: dict[str, Any], name: str
) -> float:
    if explicit is not None:
        return float(explicit)
    direct_key = f"{name}_budget"
    if direct_key in zero_result:
        return float(zero_result[direct_key])
    budgets = zero_result.get("budgets", {})
    value = budgets.get(name)
    if isinstance(value, dict):
        value = value.get("value")
    if value is None:
        raise ValueError(f"No {name} budget found; pass --{name}-budget")
    return float(value)


def _offset_ranks(text: str | None) -> list[int] | None:
    if text is None:
        return None
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--candidate-offset-ranks must not be empty")
    return [int(value) for value in values]


def _zero_indices_for_splits(
    zero_result: dict[str, Any],
    calibration_ids: Sequence[str],
    test_ids: Sequence[str],
) -> tuple[int | None, np.ndarray | None, np.ndarray | None]:
    """Resolve either one global base index or per-image adaptive base indices."""

    mapping = zero_result.get("threshold_indices_by_image")
    if mapping is None and "image_ids" in zero_result and "threshold_indices" in zero_result:
        image_ids = [str(value) for value in zero_result["image_ids"]]
        indices = list(zero_result["threshold_indices"])
        if len(image_ids) != len(indices):
            raise ValueError("zero-result image_ids/threshold_indices lengths differ")
        mapping = dict(zip(image_ids, indices))
    if mapping is not None:
        if not isinstance(mapping, dict):
            raise ValueError("threshold_indices_by_image must be a JSON object")
        missing = [
            image_id
            for image_id in list(calibration_ids) + list(test_ids)
            if image_id not in mapping
        ]
        if missing:
            raise ValueError(
                "Zero-label adaptive result lacks indices for: " + ", ".join(missing[:5])
            )
        calibration = np.asarray([mapping[item] for item in calibration_ids])
        test = np.asarray([mapping[item] for item in test_ids])
        return None, calibration, test
    zero_index = zero_result.get("threshold_index")
    if zero_index is None:
        raise ValueError(
            "Zero-label result must contain threshold_index or threshold_indices_by_image"
        )
    return int(zero_index), None, None


def audit_protocol_bundle(
    zero_result: Mapping[str, Any],
    calibration_provenance: Mapping[str, Any],
    test_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute all protocol fingerprints and compare complete protocols."""

    records = {
        "zero": zero_result,
        "calibration": calibration_provenance,
        "test": test_provenance,
    }
    representations = {
        name: str(record.get("representation", PROBABILITY_REPRESENTATION))
        for name, record in records.items()
    }
    if len(set(representations.values())) != 1:
        return {
            "verified": False,
            "fingerprints": {
                name: record.get("protocol_fingerprint")
                for name, record in records.items()
            },
            "recomputed_from_complete_protocol": False,
            "canonical_protocol": None,
            "representation": None,
            "errors": {
                "representation": "zero/calibration/test score representations differ"
            },
        }
    representation = next(iter(representations.values()))
    if representation not in {
        PROBABILITY_REPRESENTATION,
        LOGIT_REPRESENTATION,
    }:
        return {
            "verified": False,
            "fingerprints": {},
            "recomputed_from_complete_protocol": False,
            "canonical_protocol": None,
            "representation": representation,
            "errors": {"representation": "unsupported CRC score representation"},
        }
    fingerprints = {
        name: record.get("protocol_fingerprint") for name, record in records.items()
    }
    canonical: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for name, record in records.items():
        if name != "zero" and record.get("source_type") != FORMAL_COUNT_CURVE_SOURCE_TYPE:
            errors[name] = (
                "formal calibration requires count curves built directly from an "
                "exported score-map directory; precomputed/legacy artifacts are diagnostic"
            )
            continue
        if name != "zero":
            try:
                score_dir = Path(str(record.get("score_dir", "")))
                manifest_path = score_dir / "manifest.json"
                manifest_sha = _sha256_text(
                    record.get("manifest_sha256"), f"{name} manifest sha256"
                )
                if (
                    not manifest_path.is_file()
                    or _file_digest(manifest_path) != manifest_sha
                ):
                    raise ValueError(f"{name} score-map manifest artifact mismatch")
                score_manifest, _, score_integrity = verify_score_map_directory(
                    score_dir,
                    require_integrity=True,
                    require_masks=True,
                )
                assert score_manifest is not None
                if record.get("score_manifest_schema_version") != score_manifest.get(
                    "schema_version"
                ):
                    raise ValueError(f"{name} score-manifest schema mismatch")
                if record.get("score_records_sha256") != score_integrity.get(
                    "records_sha256"
                ):
                    raise ValueError(f"{name} score-record aggregate hash mismatch")
                if record.get("score_ordered_image_ids_sha256") != score_integrity.get(
                    "ordered_image_ids_sha256"
                ):
                    raise ValueError(f"{name} ordered score-map IDs hash mismatch")
                if record.get("score_num_records") != score_integrity.get("num_records"):
                    raise ValueError(f"{name} score-record count mismatch")
                for provenance_key, manifest_key in (
                    ("split_file", "split_file"),
                    ("split_file_sha256", "split_file_sha256"),
                    ("split_ordered_ids_sha256", "split_ordered_ids_sha256"),
                ):
                    if record.get(provenance_key) != score_manifest.get(manifest_key):
                        raise ValueError(f"{name} split provenance mismatch")
                grid_path = Path(str(record.get("threshold_grid", "")))
                grid_file_sha = _sha256_text(
                    record.get("threshold_grid_file_sha256"),
                    f"{name} threshold-grid file sha256",
                )
                if not grid_path.is_file() or _file_digest(grid_path) != grid_file_sha:
                    raise ValueError(f"{name} threshold-grid file artifact mismatch")
                grid_values = np.load(grid_path, allow_pickle=False)
                resolved_grid_sha = _semantic_grid_digest(
                    grid_values, representation
                )
                if representation == LOGIT_REPRESENTATION:
                    grid_manifest_path = Path(
                        str(record.get("threshold_grid_manifest", ""))
                    )
                    grid_manifest_sha = _sha256_text(
                        record.get("threshold_grid_manifest_sha256"),
                        f"{name} threshold-grid manifest sha256",
                    )
                    if (
                        not grid_manifest_path.is_file()
                        or _file_digest(grid_manifest_path) != grid_manifest_sha
                    ):
                        raise ValueError(
                            f"{name} threshold-grid manifest artifact mismatch"
                        )
                protocol_candidate = record.get("protocol")
                protocol_grid_sha = (
                    protocol_candidate.get("threshold_grid_sha256")
                    if isinstance(protocol_candidate, Mapping)
                    else None
                )
                if (
                    record.get("threshold_grid_sha256") != resolved_grid_sha
                    or protocol_grid_sha != resolved_grid_sha
                ):
                    raise ValueError(f"{name} threshold-grid provenance mismatch")
                if not isinstance(protocol_candidate, Mapping):
                    raise ValueError(f"{name} complete protocol object is missing")
                reconstructed, reconstructed_fingerprint = score_map_protocol(
                    score_dir,
                    grid_values,
                    matching_rule=str(protocol_candidate.get("matching_rule")),
                    centroid_distance=float(
                        protocol_candidate.get("centroid_distance")
                    ),
                    connectivity=int(protocol_candidate.get("connectivity")),
                    min_component_area=int(
                        protocol_candidate.get("min_component_area")
                    ),
                    representation=representation,
                    threshold_grid_detector_protocol=(
                        str(record.get("threshold_grid_detector_protocol"))
                        if representation == LOGIT_REPRESENTATION
                        else None
                    ),
                    threshold_grid_detector_checkpoint_sha256s=(
                        record.get(
                            "threshold_grid_detector_checkpoint_sha256s"
                        )
                        if representation == LOGIT_REPRESENTATION
                        else None
                    ),
                    threshold_grid_outer_detector_checkpoint_sha256=(
                        record.get(
                            "threshold_grid_outer_detector_checkpoint_sha256"
                        )
                        if representation == LOGIT_REPRESENTATION
                        else None
                    ),
                    threshold_grid_episode_detector_checkpoint_sha256s=(
                        record.get(
                            "threshold_grid_episode_detector_checkpoint_sha256s"
                        )
                        if representation == LOGIT_REPRESENTATION
                        else None
                    ),
                )
                if (
                    reconstructed != dict(protocol_candidate)
                    or reconstructed_fingerprint != record.get("protocol_fingerprint")
                ):
                    raise ValueError(
                        f"{name} protocol is not reproducible from its score manifest"
                    )
            except (OSError, TypeError, ValueError) as error:
                errors[name] = str(error)
                continue
        protocol = record.get("protocol")
        if not isinstance(protocol, Mapping):
            errors[name] = "missing complete canonical protocol object"
            continue
        try:
            canonical[name], computed = validate_formal_protocol(
                protocol, fingerprints[name]
            )
        except (TypeError, ValueError) as error:
            errors[name] = str(error)
            continue
        if computed != fingerprints[name]:
            errors[name] = "recomputed protocol fingerprint differs from recorded value"
    if not errors and not (
        canonical["zero"] == canonical["calibration"] == canonical["test"]
    ):
        errors["cross_split"] = (
            "zero/calibration/test canonical protocol objects differ"
        )
    if not errors and representation == LOGIT_REPRESENTATION:
        binding_fields = (
            "representation",
            "threshold_grid_schema_version",
            "threshold_grid_sha256",
            "threshold_grid_manifest_sha256",
            "threshold_grid_detector_protocol",
            "threshold_grid_detector_checkpoint_sha256s",
            "threshold_grid_outer_detector_checkpoint_sha256",
            "threshold_grid_episode_detector_checkpoint_sha256s",
        )
        expected_binding = {
            field: zero_result.get(field) for field in binding_fields
        }
        if expected_binding["representation"] != LOGIT_REPRESENTATION:
            errors["raw_logit_binding"] = "v4 CRC requires raw_logit_float32"
        elif expected_binding["threshold_grid_schema_version"] != LOGIT_GRID_SCHEMA_VERSION:
            errors["raw_logit_binding"] = "v4 CRC grid schema is unsupported"
        elif not isinstance(
            expected_binding["threshold_grid_manifest_sha256"], str
        ) or len(expected_binding["threshold_grid_manifest_sha256"]) != 64:
            errors["raw_logit_binding"] = "v4 CRC grid manifest hash is invalid"
        elif expected_binding["threshold_grid_detector_protocol"] != GRID_DETECTOR_PROTOCOL:
            errors["raw_logit_binding"] = "v4 CRC grid detector protocol is invalid"
        else:
            hashes = expected_binding[
                "threshold_grid_detector_checkpoint_sha256s"
            ]
            if (
                not isinstance(hashes, list)
                or not hashes
                or len(set(hashes)) != len(hashes)
                or any(
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)
                    for value in hashes
                )
            ):
                errors["raw_logit_binding"] = (
                    "v4 CRC detector checkpoint hashes must be distinct SHA-256s"
                )
            else:
                outer_hash = expected_binding[
                    "threshold_grid_outer_detector_checkpoint_sha256"
                ]
                episode_hashes = expected_binding[
                    "threshold_grid_episode_detector_checkpoint_sha256s"
                ]
                if (
                    not isinstance(outer_hash, str)
                    or outer_hash not in hashes
                    or not isinstance(episode_hashes, list)
                    or not episode_hashes
                    or len(set(episode_hashes)) != len(episode_hashes)
                    or outer_hash in episode_hashes
                    or set(hashes) != set(episode_hashes).union({outer_hash})
                ):
                    errors["raw_logit_binding"] = (
                        "v4 CRC outer/episode detector roles are invalid"
                    )
        if not errors:
            for name, record in records.items():
                observed = {field: record.get(field) for field in binding_fields}
                if observed != expected_binding:
                    errors[name] = "raw-logit CRC semantic binding mismatch"
    return {
        "verified": not errors,
        "fingerprints": fingerprints,
        "recomputed_from_complete_protocol": not errors,
        "canonical_protocol": canonical.get("zero") if not errors else None,
        "representation": representation,
        "raw_logit_binding": (
            expected_binding
            if representation == LOGIT_REPRESENTATION and not errors
            else None
        ),
        "errors": errors,
    }


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256_text(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be 64 lowercase hex digits")
    return text


def _id_rows(value: Any, name: str, expected_rows: int) -> list[list[str]]:
    if not isinstance(value, list) or len(value) != expected_rows:
        raise ValueError(f"{name} must contain one ID list per causal block")
    rows: list[list[str]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or not row:
            raise ValueError(f"{name} row {row_index} must be a non-empty ID list")
        ids = [str(item) for item in row]
        if any(not item for item in ids):
            raise ValueError(f"{name} row {row_index} contains an empty ID")
        rows.append(ids)
    return rows


def _verify_zero_protocol_against_manifest(
    zero_result: Mapping[str, Any], score_map_dir: Path
) -> None:
    verify_score_map_directory(
        score_map_dir,
        require_integrity=True,
        require_masks=False,
    )
    protocol = zero_result.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("zero result lacks a complete protocol object")
    thresholds = np.asarray(zero_result.get("thresholds"), dtype=np.float32)
    representation = str(
        zero_result.get("representation", PROBABILITY_REPRESENTATION)
    )
    if _semantic_grid_digest(thresholds, representation) != protocol.get(
        "threshold_grid_sha256"
    ):
        raise ValueError("zero threshold grid does not match its protocol")
    try:
        reconstructed, fingerprint = score_map_protocol(
            score_map_dir,
            thresholds,
            matching_rule=str(protocol.get("matching_rule")),
            centroid_distance=float(protocol.get("centroid_distance")),
            connectivity=int(protocol.get("connectivity")),
            min_component_area=int(protocol.get("min_component_area")),
            representation=representation,
            threshold_grid_detector_protocol=(
                str(zero_result.get("threshold_grid_detector_protocol"))
                if representation == LOGIT_REPRESENTATION
                else None
            ),
            threshold_grid_detector_checkpoint_sha256s=(
                zero_result.get(
                    "threshold_grid_detector_checkpoint_sha256s"
                )
                if representation == LOGIT_REPRESENTATION
                else None
            ),
            threshold_grid_outer_detector_checkpoint_sha256=(
                zero_result.get(
                    "threshold_grid_outer_detector_checkpoint_sha256"
                )
                if representation == LOGIT_REPRESENTATION
                else None
            ),
            threshold_grid_episode_detector_checkpoint_sha256s=(
                zero_result.get(
                    "threshold_grid_episode_detector_checkpoint_sha256s"
                )
                if representation == LOGIT_REPRESENTATION
                else None
            ),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("zero protocol cannot be reconstructed from score manifest") from error
    if (
        reconstructed != dict(protocol)
        or fingerprint != zero_result.get("protocol_fingerprint")
    ):
        raise ValueError("zero protocol is not reproducible from its score manifest")


def _validate_sample_adaptive_artifact(
    zero_result: Mapping[str, Any],
    calibration_ids: Sequence[str],
    test_ids: Sequence[str],
) -> dict[str, Any]:
    if zero_result.get("adaptation_protocol") != "external_causal_statistics":
        raise ValueError(
            "sample-adaptive formal calibration requires external_causal_statistics"
        )
    artifact = zero_result.get("statistics_artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("sample-adaptive zero result lacks statistics_artifact")
    if artifact.get("source_type") != "deployment_statistics_archive":
        raise ValueError("statistics_artifact source_type is not deployment statistics")
    representation = str(
        zero_result.get("representation", PROBABILITY_REPRESENTATION)
    )
    expected_deployment_schema = (
        LOGIT_DEPLOYMENT_STATISTICS_SCHEMA_VERSION
        if representation == LOGIT_REPRESENTATION
        else DEPLOYMENT_STATISTICS_SCHEMA_VERSION
    )
    expected_statistics_schema = (
        LOGIT_STATISTICS_SCHEMA_VERSION
        if representation == LOGIT_REPRESENTATION
        else STATISTICS_SCHEMA_VERSION
    )
    if (
        artifact.get("deployment_statistics_schema_version")
        != expected_deployment_schema
    ):
        raise ValueError("deployment statistics schema is missing or unsupported")
    if (
        artifact.get("statistics_schema_version") != expected_statistics_schema
        or zero_result.get("statistics_schema_version") != expected_statistics_schema
    ):
        raise ValueError("zero/deployment statistics feature schema is unsupported")
    artifact_path = Path(str(artifact.get("path", "")))
    if not artifact_path.is_file():
        raise ValueError("deployment statistics artifact is not readable")
    recorded_sha = _sha256_text(artifact.get("sha256"), "statistics artifact sha256")
    if _file_digest(artifact_path) != recorded_sha:
        raise ValueError("deployment statistics artifact sha256 mismatch")
    if (
        zero_result.get("statistics_file") != str(artifact_path)
        or zero_result.get("statistics_file_sha256") != recorded_sha
    ):
        raise ValueError("zero result statistics-file evidence is internally inconsistent")
    artifact_grid_sha = _sha256_text(
        artifact.get("threshold_grid_sha256"), "statistics threshold-grid sha256"
    )
    protocol = zero_result.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("threshold_grid_sha256") != artifact_grid_sha:
        raise ValueError("statistics artifact and zero protocol threshold grids differ")
    provenance = artifact.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("deployment statistics provenance is missing")
    if provenance.get("masks_read") is not False:
        raise ValueError("formal zero-label statistics must record masks_read=false")
    if provenance.get("threshold_grid_sha256") != artifact_grid_sha:
        raise ValueError("deployment provenance threshold-grid sha256 mismatch")
    if representation == LOGIT_REPRESENTATION:
        bound_fields = {
            "representation": LOGIT_REPRESENTATION,
            "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
            "threshold_grid_sha256": artifact_grid_sha,
            "threshold_grid_manifest_sha256": zero_result.get(
                "threshold_grid_manifest_sha256"
            ),
            "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
            "threshold_grid_detector_checkpoint_sha256s": zero_result.get(
                "threshold_grid_detector_checkpoint_sha256s"
            ),
            "threshold_grid_outer_detector_checkpoint_sha256": zero_result.get(
                "threshold_grid_outer_detector_checkpoint_sha256"
            ),
            "threshold_grid_episode_detector_checkpoint_sha256s": zero_result.get(
                "threshold_grid_episode_detector_checkpoint_sha256s"
            ),
        }
        if not isinstance(
            bound_fields["threshold_grid_manifest_sha256"], str
        ) or len(bound_fields["threshold_grid_manifest_sha256"]) != 64:
            raise ValueError("raw-logit grid-manifest hash is invalid")
        detector_hashes = bound_fields[
            "threshold_grid_detector_checkpoint_sha256s"
        ]
        if (
            not isinstance(detector_hashes, list)
            or not detector_hashes
            or len(set(detector_hashes)) != len(detector_hashes)
        ):
            raise ValueError("raw-logit detector checkpoint hashes are not distinct")
        outer_detector_hash = bound_fields[
            "threshold_grid_outer_detector_checkpoint_sha256"
        ]
        episode_detector_hashes = bound_fields[
            "threshold_grid_episode_detector_checkpoint_sha256s"
        ]
        if (
            not isinstance(outer_detector_hash, str)
            or outer_detector_hash not in detector_hashes
            or not isinstance(episode_detector_hashes, list)
            or not episode_detector_hashes
            or outer_detector_hash in episode_detector_hashes
            or len(set(episode_detector_hashes)) != len(episode_detector_hashes)
            or set(detector_hashes)
            != set(episode_detector_hashes).union({outer_detector_hash})
        ):
            raise ValueError("raw-logit outer/episode detector roles are invalid")
        for owner, record in (
            ("zero result", zero_result),
            ("statistics artifact", artifact),
            ("deployment provenance", provenance),
        ):
            for field, expected in bound_fields.items():
                if record.get(field) != expected:
                    raise ValueError(
                        f"{owner} raw-logit CRC binding differs in {field}"
                    )
        if zero_result.get("empty_action") != empty_action_contract():
            raise ValueError("raw-logit zero result lacks external +inf reject action")
        selection_contract = zero_result.get("selection_data_contract")
        if not isinstance(selection_contract, Mapping) or (
            selection_contract.get("empty_action") != empty_action_contract()
        ):
            raise ValueError("raw-logit selection contract lacks external reject action")
    score_map_dir = Path(str(provenance.get("score_map_dir", "")))
    score_manifest_path = score_map_dir / "manifest.json"
    score_manifest_sha = _sha256_text(
        provenance.get("score_manifest_sha256"), "score manifest sha256"
    )
    if (
        not score_manifest_path.is_file()
        or _file_digest(score_manifest_path) != score_manifest_sha
    ):
        raise ValueError("deployment score-map manifest artifact sha256 mismatch")
    _verify_zero_protocol_against_manifest(zero_result, score_map_dir)
    adaptation_window = _positive_int(
        provenance.get("adaptation_window"), "adaptation_window"
    )
    evaluation_window = _positive_int(
        provenance.get("evaluation_window"), "evaluation_window"
    )
    stride = _positive_int(provenance.get("stride"), "stride")
    if evaluation_window != 1:
        raise ValueError("formal sample-adaptive calibration requires evaluation_window=1")
    if stride != adaptation_window + evaluation_window:
        raise ValueError("formal causal blocks require stride=adaptation_window+evaluation_window")
    if provenance.get("allow_role_reuse") is not False:
        raise ValueError("formal causal blocks forbid role reuse")
    if provenance.get("global_role_overlap") != []:
        raise ValueError("formal causal blocks contain global A/E role overlap")
    num_windows = _positive_int(zero_result.get("num_windows"), "num_windows")
    if provenance.get("num_windows") != num_windows:
        raise ValueError("deployment provenance num_windows differs from zero result")
    adaptation_rows = _id_rows(
        zero_result.get("adaptation_ids"), "adaptation_ids", num_windows
    )
    evaluation_rows = _id_rows(
        zero_result.get("evaluation_ids"), "evaluation_ids", num_windows
    )
    if any(len(row) != adaptation_window for row in adaptation_rows):
        raise ValueError("adaptation block size differs from adaptation_window")
    if any(len(row) != 1 for row in evaluation_rows):
        raise ValueError("formal mapping requires exactly one evaluation image per block")
    all_block_ids = [
        image_id
        for adaptation_row, evaluation_row in zip(adaptation_rows, evaluation_rows)
        for image_id in adaptation_row + evaluation_row
    ]
    if len(set(all_block_ids)) != len(all_block_ids):
        raise ValueError("formal causal A/E blocks must be globally disjoint")
    flattened_adaptation = [item for row in adaptation_rows for item in row]
    if list(zero_result.get("window_ids") or []) != flattened_adaptation:
        raise ValueError("window_ids do not exactly match ordered adaptation blocks")
    threshold_indices = zero_result.get("threshold_indices")
    if not isinstance(threshold_indices, list) or len(threshold_indices) != num_windows:
        raise ValueError("threshold_indices must contain one value per causal block")
    mapping = zero_result.get("threshold_indices_by_image")
    if not isinstance(mapping, Mapping):
        raise ValueError("sample-adaptive zero result lacks per-image threshold mapping")
    expected_mapping = {
        evaluation_row[0]: threshold_index
        for evaluation_row, threshold_index in zip(evaluation_rows, threshold_indices)
    }
    if dict(mapping) != expected_mapping:
        raise ValueError(
            "threshold_indices_by_image is not the one-to-one causal block mapping"
        )
    expected_evaluation_ids = set(calibration_ids).union(test_ids)
    if set(expected_mapping) != expected_evaluation_ids:
        raise ValueError(
            "causal evaluation blocks must map one-to-one onto calibration and test IDs"
        )
    return {
        "mode": "sample_adaptive_disjoint_blocks",
        "num_blocks": num_windows,
        "adaptation_window": adaptation_window,
        "evaluation_window": evaluation_window,
        "stride": stride,
        "statistics_artifact_sha256": recorded_sha,
    }


def audit_zero_artifact_contract(
    zero_result: Mapping[str, Any],
    calibration_ids: Sequence[str],
    test_ids: Sequence[str],
) -> dict[str, Any]:
    """Audit the mask-free causal evidence chain behind zero-label indices."""

    try:
        if zero_result.get("schema_version") != ZERO_RESULT_SCHEMA_VERSION:
            raise ValueError("zero result schema is missing or unsupported")
        if zero_result.get("mode") != "zero_label_empirical_adaptation":
            raise ValueError("zero result mode is not zero_label_empirical_adaptation")
        data_contract = zero_result.get("selection_data_contract")
        if not isinstance(data_contract, Mapping):
            raise ValueError("zero result lacks selection_data_contract")
        if (
            data_contract.get("schema_version")
            != SELECTION_DATA_CONTRACT_SCHEMA_VERSION
        ):
            raise ValueError("zero selection data-contract schema is unsupported")
        if data_contract.get("masks_read") is not False:
            raise ValueError("zero selection must explicitly record masks_read=false")
        if data_contract.get("checkpoint_training_contract_verified") is not True:
            raise ValueError(
                "zero selection does not record a verified checkpoint training contract"
            )
        representation = str(
            zero_result.get("representation", PROBABILITY_REPRESENTATION)
        )
        if representation == LOGIT_REPRESENTATION:
            required_raw_bindings = {
                "representation": LOGIT_REPRESENTATION,
                "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
                "threshold_grid_sha256": zero_result.get(
                    "threshold_grid_sha256"
                ),
                "threshold_grid_manifest_sha256": zero_result.get(
                    "threshold_grid_manifest_sha256"
                ),
                "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
                "threshold_grid_detector_checkpoint_sha256s": zero_result.get(
                    "threshold_grid_detector_checkpoint_sha256s"
                ),
                "threshold_grid_outer_detector_checkpoint_sha256": (
                    zero_result.get(
                        "threshold_grid_outer_detector_checkpoint_sha256"
                    )
                ),
                "threshold_grid_episode_detector_checkpoint_sha256s": (
                    zero_result.get(
                        "threshold_grid_episode_detector_checkpoint_sha256s"
                    )
                ),
            }
            for field, expected in required_raw_bindings.items():
                if data_contract.get(field) != expected:
                    raise ValueError(
                        f"raw-logit selection data contract differs in {field}"
                    )
            if data_contract.get("empty_action") != empty_action_contract():
                raise ValueError(
                    "raw-logit selection data contract lacks external +inf reject"
                )
            if zero_result.get("empty_action") != empty_action_contract():
                raise ValueError("raw-logit zero result lacks external +inf reject")
            if zero_result.get("prediction_rule") != LOGIT_PREDICTION_RULE:
                raise ValueError("raw-logit zero result prediction rule is invalid")
        elif representation != PROBABILITY_REPRESENTATION:
            raise ValueError("zero result uses an unsupported representation")
        checkpoint_path = Path(str(zero_result.get("curve_checkpoint", "")))
        checkpoint_sha = _sha256_text(
            zero_result.get("curve_checkpoint_sha256"), "curve checkpoint sha256"
        )
        if not checkpoint_path.is_file() or _file_digest(checkpoint_path) != checkpoint_sha:
            raise ValueError("curve checkpoint artifact sha256 mismatch")
        adaptive = zero_result.get("threshold_indices_by_image") is not None
        protocol = zero_result.get("protocol")
        deployment_provenance: Mapping[str, Any] | None = None
        if adaptive:
            statistics_artifact = zero_result.get("statistics_artifact")
            if isinstance(statistics_artifact, Mapping):
                candidate_provenance = statistics_artifact.get("provenance")
                if isinstance(candidate_provenance, Mapping):
                    deployment_provenance = candidate_provenance
        else:
            candidate_provenance = zero_result.get("score_map_provenance")
            if isinstance(candidate_provenance, Mapping):
                deployment_provenance = candidate_provenance
        checkpoint_contract_audit = audit_checkpoint_deployment_contract(
            checkpoint_path,
            deployment_provenance=deployment_provenance,
            target_dataset=(
                str(protocol.get("target_dataset", ""))
                if isinstance(protocol, Mapping)
                else ""
            ),
            expected_threshold_grid_sha256=(
                str(protocol.get("threshold_grid_sha256", ""))
                if isinstance(protocol, Mapping)
                else ""
            ),
            expected_representation=(
                LOGIT_REPRESENTATION
                if representation == LOGIT_REPRESENTATION
                else None
            ),
            expected_threshold_grid_schema_version=(
                str(zero_result.get("threshold_grid_schema_version"))
                if representation == LOGIT_REPRESENTATION
                else None
            ),
            expected_threshold_grid_manifest_sha256=(
                str(zero_result.get("threshold_grid_manifest_sha256"))
                if representation == LOGIT_REPRESENTATION
                else None
            ),
            expected_threshold_grid_detector_protocol=(
                str(zero_result.get("threshold_grid_detector_protocol"))
                if representation == LOGIT_REPRESENTATION
                else None
            ),
            expected_threshold_grid_detector_checkpoint_sha256s=(
                zero_result.get(
                    "threshold_grid_detector_checkpoint_sha256s"
                )
                if representation == LOGIT_REPRESENTATION
                else None
            ),
            expected_threshold_grid_outer_detector_checkpoint_sha256=(
                zero_result.get(
                    "threshold_grid_outer_detector_checkpoint_sha256"
                )
                if representation == LOGIT_REPRESENTATION
                else None
            ),
            expected_threshold_grid_episode_detector_checkpoint_sha256s=(
                zero_result.get(
                    "threshold_grid_episode_detector_checkpoint_sha256s"
                )
                if representation == LOGIT_REPRESENTATION
                else None
            ),
            expected_curve_checkpoint_sha256=checkpoint_sha,
        )
        if not checkpoint_contract_audit["verified"]:
            raise ValueError(
                "curve checkpoint/deployment contract failed: "
                + json.dumps(checkpoint_contract_audit["errors"], sort_keys=True)
            )
        if adaptive:
            if (
                data_contract.get("threshold_mapping_rule")
                != "one_A_block_prediction_to_its_future_E_identity"
            ):
                raise ValueError("sample-adaptive threshold mapping rule is unsupported")
            details = _validate_sample_adaptive_artifact(
                zero_result, calibration_ids, test_ids
            )
        else:
            if zero_result.get("adaptation_protocol") != "causal_warmup":
                raise ValueError("formal global zero selection requires causal_warmup")
            provenance = zero_result.get("score_map_provenance")
            if not isinstance(provenance, Mapping):
                raise ValueError("global zero result lacks score-map provenance")
            if data_contract.get("threshold_mapping_rule") != "one_global_warmup_prediction":
                raise ValueError("global zero threshold mapping rule is unsupported")
            manifest_sha = _sha256_text(
                provenance.get("manifest_sha256"), "score manifest sha256"
            )
            manifest_path = Path(str(provenance.get("manifest_path", "")))
            if not manifest_path.is_file() or _file_digest(manifest_path) != manifest_sha:
                raise ValueError("score-map manifest artifact sha256 mismatch")
            _verify_zero_protocol_against_manifest(
                zero_result, manifest_path.parent
            )
            details = {"mode": "global_causal_warmup"}
        return {
            "verified": True,
            "errors": {},
            "curve_checkpoint_contract": checkpoint_contract_audit,
            **details,
        }
    except (TypeError, ValueError) as error:
        return {"verified": False, "errors": {"zero_artifact": str(error)}}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-curves", required=True)
    parser.add_argument(
        "--test-curves",
        required=True,
        help="Independent test count curves; only IDs/grid are inspected during calibration",
    )
    parser.add_argument("--zero-result", required=True)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--pixel-budget", type=float)
    parser.add_argument("--component-budget", type=float)
    parser.add_argument(
        "--loss-mode",
        choices=("budget_violation", "risk_ratio"),
        default="budget_violation",
        help="Formal JointBSR claims require budget_violation (the default)",
    )
    parser.add_argument(
        "--candidate-offset-ranks",
        help="Comma-separated rank offsets; default is the complete suffix plus reject",
    )
    parser.add_argument(
        "--allow-unverified-protocol",
        action="store_true",
        help=(
            "Diagnostic only: permit incomplete legacy protocol/artifact evidence "
            "and disable formal claims"
        ),
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    calibration_path = Path(args.calibration_curves)
    test_path = Path(args.test_curves)
    zero_path = Path(args.zero_result)
    zero_result = json.loads(zero_path.read_text(encoding="utf-8"))
    require_integrity = not args.allow_unverified_protocol
    calibration_data = load_count_curve_archive(
        calibration_path, require_integrity=require_integrity
    )
    calibration_provenance = load_count_curve_provenance(
        calibration_path, require_integrity=require_integrity
    )
    test_provenance = load_count_curve_provenance(
        test_path, require_integrity=require_integrity
    )
    (
        test_image_ids,
        test_grid,
        test_representation,
        test_archive_contract,
    ) = _load_test_identity(test_path, require_integrity=require_integrity)
    calibration_archive_audit = verify_count_curve_archive_integrity(
        calibration_path, require_integrity=require_integrity
    )
    test_archive_audit = verify_count_curve_archive_integrity(
        test_path, require_integrity=require_integrity
    )
    calibration_ids = _ids(calibration_data["image_ids"], "calibration")
    test_ids = _ids(test_image_ids, "test")
    count_source_binding = {
        "calibration": (
            calibration_provenance.get("score_ordered_image_ids_sha256")
            == ordered_ids_sha256(calibration_ids)
        ),
        "test": (
            test_provenance.get("score_ordered_image_ids_sha256")
            == ordered_ids_sha256(test_ids)
        ),
    }
    if not all(count_source_binding.values()) and require_integrity:
        raise ValueError(
            "Count archive image ID/order is not bound to its score-map manifest: "
            + json.dumps(count_source_binding, sort_keys=True)
        )

    calibration_representation = str(
        calibration_data.get("representation", PROBABILITY_REPRESENTATION)
    )
    zero_representation = str(
        zero_result.get("representation", PROBABILITY_REPRESENTATION)
    )
    if not (
        calibration_representation
        == test_representation
        == zero_representation
    ):
        raise ValueError(
            "Zero-label, calibration, and test score representations differ"
        )
    calibration_grid = np.asarray(calibration_data["thresholds"])
    if not _same_grid(calibration_grid, test_grid):
        raise ValueError("Calibration and test threshold grids differ")
    if "thresholds" in zero_result and not _same_grid(
        calibration_grid, np.asarray(zero_result["thresholds"])
    ):
        raise ValueError("Zero-label and calibration threshold grids differ")
    calibration_archive_contract = {
        "representation": calibration_representation,
        "threshold_grid_schema_version": calibration_data.get(
            "threshold_grid_schema_version"
        ),
        "threshold_grid_sha256": calibration_data.get(
            "recorded_threshold_grid_sha256"
        ),
        "threshold_grid_manifest_sha256": calibration_data.get(
            "threshold_grid_manifest_sha256"
        ),
        "threshold_grid_detector_protocol": calibration_data.get(
            "threshold_grid_detector_protocol"
        ),
        "threshold_grid_detector_checkpoint_sha256s": calibration_data.get(
            "threshold_grid_detector_checkpoint_sha256s"
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": calibration_data.get(
            "threshold_grid_outer_detector_checkpoint_sha256"
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": calibration_data.get(
            "threshold_grid_episode_detector_checkpoint_sha256s"
        ),
    }
    raw_count_binding = {
        "calibration": _raw_count_archive_provenance_binding(
            calibration_archive_contract,
            calibration_provenance,
            calibration_grid,
            split_name="calibration",
        ),
        "test": _raw_count_archive_provenance_binding(
            test_archive_contract,
            test_provenance,
            test_grid,
            split_name="test",
        ),
    }
    raw_count_binding_verified = all(
        audit["verified"] for audit in raw_count_binding.values()
    )
    if not raw_count_binding_verified and require_integrity:
        raise ValueError(
            "Raw-logit count archive top-level contract differs from provenance: "
            + json.dumps(
                {
                    split: audit["errors"]
                    for split, audit in raw_count_binding.items()
                    if not audit["verified"]
                },
                sort_keys=True,
            )
        )
    has_adaptive_mapping = (
        zero_result.get("threshold_indices_by_image") is not None
        or (
            "image_ids" in zero_result
            and "threshold_indices" in zero_result
        )
    )
    if zero_result.get("reject") and not has_adaptive_mapping:
        raise ValueError("Zero-label stage rejected; there is no threshold to calibrate")
    if zero_result.get("adaptation_protocol") == "transductive":
        raise ValueError(
            "Transductive zero-label adaptation is empirical only and cannot enter "
            "the formal three-way-disjoint calibration CLI"
        )
    adaptation_ids = zero_result.get("window_ids") or zero_result.get(
        "adaptation_image_ids"
    )
    if not adaptation_ids:
        raise ValueError(
            "Formal calibration requires non-empty zero-result window_ids so "
            "adaptation/calibration/test disjointness can be verified"
        )
    zero_index, calibration_zero_indices, test_zero_indices = _zero_indices_for_splits(
        zero_result, calibration_ids, test_ids
    )
    if zero_index is not None and not 0 <= zero_index < calibration_grid.size:
        raise ValueError("Zero-label threshold index lies outside the shared grid")

    pixel_budget = _budget(args.pixel_budget, zero_result, "pixel")
    component_budget = _budget(args.component_budget, zero_result, "component")
    protocol_audit = audit_protocol_bundle(
        zero_result, calibration_provenance, test_provenance
    )
    if (
        protocol_audit["verified"]
        and protocol_audit["canonical_protocol"]["threshold_grid_sha256"]
        != _semantic_grid_digest(calibration_grid, calibration_representation)
    ):
        protocol_audit["verified"] = False
        protocol_audit["recomputed_from_complete_protocol"] = False
        protocol_audit["errors"] = {
            "threshold_grid": (
                "canonical protocol threshold_grid_sha256 differs from the "
                "calibration/test archive grid"
            )
        }
        protocol_audit["canonical_protocol"] = None
    zero_artifact_audit = audit_zero_artifact_contract(
        zero_result, calibration_ids, test_ids
    )
    formal_artifact_chain_verified = bool(
        protocol_audit["verified"]
        and zero_artifact_audit["verified"]
        and calibration_archive_audit["verified"]
        and test_archive_audit["verified"]
        and all(count_source_binding.values())
        and raw_count_binding_verified
    )
    if not formal_artifact_chain_verified and not args.allow_unverified_protocol:
        raise ValueError(
            "Formal calibration protocol/artifact contract failed: "
            + json.dumps(
                {
                    "protocol": protocol_audit["errors"],
                    "zero_artifact": zero_artifact_audit["errors"],
                    "raw_logit_count_archive_binding": {
                        split: audit["errors"]
                        for split, audit in raw_count_binding.items()
                        if not audit["verified"]
                    },
                },
                sort_keys=True,
            )
        )
    losses = build_calibration_losses(
        **calibration_data,
        pixel_budget=pixel_budget,
        component_budget=component_budget,
        loss_mode=args.loss_mode,
    )
    result = calibrate_target_offset(
        losses,
        zero_index=zero_index,
        calibration_zero_indices=calibration_zero_indices,
        test_zero_indices=test_zero_indices,
        alpha=args.alpha,
        test_image_ids=test_image_ids,
        adaptation_image_ids=adaptation_ids,
        candidate_offset_ranks=_offset_ranks(args.candidate_offset_ranks),
        curve_checkpoint_sha256=(
            str(zero_result.get("curve_checkpoint_sha256"))
            if calibration_representation == LOGIT_REPRESENTATION
            else None
        ),
    )
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    protocol_audit["allow_unverified_protocol"] = bool(
        args.allow_unverified_protocol
    )
    result["protocol_audit"] = protocol_audit
    result["zero_artifact_audit"] = zero_artifact_audit
    result["count_archive_audit"] = {
        "calibration": calibration_archive_audit,
        "test": test_archive_audit,
        "score_source_id_order_binding": count_source_binding,
        "raw_logit_top_level_provenance_binding": raw_count_binding,
    }
    result["formal_artifact_chain_verified"] = formal_artifact_chain_verified
    if (
        formal_artifact_chain_verified
        and result["success"]
        and losses.loss_mode == LOSS_MODE_BUDGET_VIOLATION
    ):
        result["guarantee_scope"] = (
            "finite-sample corrected control of the marginal joint physical-budget "
            "violation probability (equivalently, a JointBSR lower bound of 1-alpha), "
            "conditional on the verified artifact chain, exchangeability, "
            "monotonicity, and pre-specification assumptions"
        )
    elif formal_artifact_chain_verified and result["success"]:
        result["guarantee_scope"] = (
            "finite-sample corrected control of an expected bounded risk-ratio "
            "surrogate; this does not imply a JointBSR guarantee"
        )
    elif not formal_artifact_chain_verified:
        result["guarantee_scope"] = (
            "none: the complete detector/protocol/zero-statistics artifact chain "
            "was not verified; this run is diagnostic only"
        )
    result["provenance"] = {
        "command": shlex.join(sys.argv),
        "calibration_curves": str(calibration_path.resolve()),
        "calibration_curves_sha256": _file_digest(calibration_path),
        "calibration_count_payload_sha256": calibration_archive_audit[
            "payload_sha256"
        ],
        "test_curves": str(test_path.resolve()),
        "test_curves_sha256": _file_digest(test_path),
        "test_count_payload_sha256": test_archive_audit["payload_sha256"],
        "zero_result": str(zero_path.resolve()),
        "zero_result_sha256": _file_digest(zero_path),
        "representation": calibration_representation,
        "threshold_grid_schema_version": result.get(
            "threshold_grid_schema_version"
        ),
        "threshold_grid_sha256": _semantic_grid_digest(
            calibration_grid, calibration_representation
        ),
        "threshold_grid_manifest_sha256": result.get(
            "threshold_grid_manifest_sha256"
        ),
        "threshold_grid_detector_protocol": result.get(
            "threshold_grid_detector_protocol"
        ),
        "threshold_grid_detector_checkpoint_sha256s": result.get(
            "threshold_grid_detector_checkpoint_sha256s"
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": result.get(
            "threshold_grid_outer_detector_checkpoint_sha256"
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": result.get(
            "threshold_grid_episode_detector_checkpoint_sha256s"
        ),
        "curve_checkpoint_sha256": result.get("curve_checkpoint_sha256"),
        "three_way_disjoint_verified": True,
        "protocol_fingerprint_verified": bool(protocol_audit["verified"]),
        "formal_artifact_chain_verified": formal_artifact_chain_verified,
        "test_labels_used_for_selection": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "success": result["success"],
                "reject": result["reject"],
                "reason": result["reason"],
                "selected_threshold_index": result["selected_threshold_index"],
                "selected_threshold": result["selected_threshold"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
