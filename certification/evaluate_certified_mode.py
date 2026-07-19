"""Audit a selected target-domain operating point on an independent test split.

This evaluator never re-selects a threshold from test outcomes.  It verifies
the recorded grid, budgets, and calibration/test ID separation, then reports
physical false-alarm risks and bounded-loss diagnostics at the already selected
grid index.  Per-image ``None`` actions are explicit no-detection rejections:
they contribute zero detections while remaining in overall exposure and
ground-truth denominators.  A rejected calibration result remains rejected and
is never relabeled as certified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .build_calibration_losses import (
    COMPONENT_BUDGET_UNIT,
    LOSS_SCHEMA_VERSION,
    RAW_LOGIT_LOSS_SCHEMA_VERSION,
    PIXEL_BUDGET_UNIT,
    CalibrationLosses,
    build_calibration_losses,
    load_count_curve_archive,
)
from .calibrate_target_offset import (
    RAW_LOGIT_RESULT_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    assert_disjoint_image_ids,
    assert_three_way_disjoint_image_ids,
)
from .conformal_offset import SELECTION_SCHEMA_VERSION
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


EVALUATION_SCHEMA_VERSION = "rc-v2-independent-test-audit-v4-action-bound"
RAW_LOGIT_EVALUATION_SCHEMA_VERSION = (
    "rc-v4-independent-test-audit-v1-raw-logit-action-bound"
)

_RAW_BINDING_FIELDS = (
    "representation",
    "threshold_grid_schema_version",
    "threshold_grid_sha256",
    "threshold_grid_manifest_sha256",
    "threshold_grid_detector_protocol",
    "threshold_grid_detector_checkpoint_sha256s",
    "threshold_grid_outer_detector_checkpoint_sha256",
    "threshold_grid_episode_detector_checkpoint_sha256s",
    "curve_checkpoint_sha256",
)


def _file_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256_text(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _detector_role_hashes(
    record: Mapping[str, Any],
) -> tuple[list[str], str, list[str]]:
    all_value = record.get("threshold_grid_detector_checkpoint_sha256s")
    episode_value = record.get(
        "threshold_grid_episode_detector_checkpoint_sha256s"
    )
    if not isinstance(all_value, list) or not all_value:
        raise ValueError("raw-logit detector hashes must be a non-empty list")
    if not isinstance(episode_value, list) or not episode_value:
        raise ValueError("raw-logit episode-detector hashes must be a non-empty list")
    all_hashes = [
        _sha256_text(value, field=f"detector hash {position}")
        for position, value in enumerate(all_value)
    ]
    episode_hashes = [
        _sha256_text(value, field=f"episode detector hash {position}")
        for position, value in enumerate(episode_value)
    ]
    if len(set(all_hashes)) != len(all_hashes):
        raise ValueError("raw-logit detector checkpoint hashes must be distinct")
    if len(set(episode_hashes)) != len(episode_hashes):
        raise ValueError("raw-logit episode-detector checkpoint hashes must be distinct")
    outer_hash = _sha256_text(
        record.get("threshold_grid_outer_detector_checkpoint_sha256"),
        field="outer detector hash",
    )
    if outer_hash in episode_hashes:
        raise ValueError("raw-logit outer detector must be disjoint from episode detectors")
    if set(all_hashes) != set(episode_hashes).union({outer_hash}):
        raise ValueError(
            "raw-logit detector hashes must equal outer plus episode detectors"
        )
    return all_hashes, outer_hash, episode_hashes


def _evaluation_contract(
    selection_result: Mapping[str, Any],
    test_losses: CalibrationLosses,
) -> dict[str, Any]:
    """Validate the representation-specific result/loss binding.

    v2 probability records retain their compatibility behaviour.  A v4 record
    is accepted only when its FP32 logit grid and every source/checkpoint role
    identity agree with the independently loaded test-loss archive.
    """

    schema = selection_result.get("schema_version")
    if schema is None:
        return {
            "verified": False,
            "mode": "legacy_unversioned",
            "representation": test_losses.representation,
            "raw_binding": None,
            "result_schema": None,
            "loss_schema": None,
        }

    representation = str(
        selection_result.get("representation", PROBABILITY_REPRESENTATION)
    )
    loss = selection_result.get("loss")
    loss_schema = loss.get("schema_version") if isinstance(loss, Mapping) else None
    if schema == RESULT_SCHEMA_VERSION:
        if representation != PROBABILITY_REPRESENTATION:
            raise ValueError("v2 calibration result must use sigmoid probability")
        if test_losses.representation != PROBABILITY_REPRESENTATION:
            raise ValueError("v2 calibration result cannot evaluate raw-logit losses")
        if loss_schema != LOSS_SCHEMA_VERSION:
            raise ValueError("Calibration loss schema is missing or unsupported")
        return {
            "verified": True,
            "mode": "v2_probability_contract",
            "representation": representation,
            "raw_binding": None,
            "result_schema": schema,
            "loss_schema": loss_schema,
        }
    if schema != RAW_LOGIT_RESULT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported calibration result schema: {schema!r}")
    if representation != LOGIT_REPRESENTATION:
        raise ValueError("v4 calibration result must use raw_logit_float32")
    if test_losses.representation != LOGIT_REPRESENTATION:
        raise ValueError("v4 calibration result and test-loss representation differ")
    if loss_schema != RAW_LOGIT_LOSS_SCHEMA_VERSION:
        raise ValueError("v4 raw-logit calibration loss schema is missing or unsupported")

    test_grid = validate_logit_threshold_grid(np.asarray(test_losses.thresholds))
    recorded_grid = selection_result.get("thresholds")
    try:
        recorded_values = np.asarray(recorded_grid, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("v4 result threshold grid is malformed") from error
    if (
        recorded_values.ndim != 1
        or recorded_values.shape != test_grid.shape
        or not np.isfinite(recorded_values).all()
        or not np.array_equal(recorded_values, test_grid.astype(np.float64))
    ):
        raise ValueError("v4 result and test archive use different FP32 logit grids")
    semantic_hash = logit_threshold_grid_sha256(test_grid)
    if selection_result.get("threshold_grid_schema_version") != LOGIT_GRID_SCHEMA_VERSION:
        raise ValueError("v4 result uses an unsupported logit-grid schema")
    if selection_result.get("threshold_grid_sha256") != semantic_hash:
        raise ValueError("v4 result semantic logit-grid hash mismatch")
    _sha256_text(
        selection_result.get("threshold_grid_manifest_sha256"),
        field="threshold-grid manifest hash",
    )
    if selection_result.get("threshold_grid_detector_protocol") != GRID_DETECTOR_PROTOCOL:
        raise ValueError("v4 result grid detector protocol is unsupported")
    all_hashes, outer_hash, episode_hashes = _detector_role_hashes(selection_result)
    curve_hash = _sha256_text(
        selection_result.get("curve_checkpoint_sha256"),
        field="curve checkpoint hash",
    )
    if selection_result.get("prediction_rule") != LOGIT_PREDICTION_RULE:
        raise ValueError("v4 result prediction rule is not raw-logit thresholding")
    if selection_result.get("empty_action") != empty_action_contract():
        raise ValueError("v4 result external +inf empty action is missing or altered")

    raw_binding = {
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": semantic_hash,
        "threshold_grid_manifest_sha256": selection_result.get(
            "threshold_grid_manifest_sha256"
        ),
        "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": all_hashes,
        "threshold_grid_outer_detector_checkpoint_sha256": outer_hash,
        "threshold_grid_episode_detector_checkpoint_sha256s": episode_hashes,
        "curve_checkpoint_sha256": curve_hash,
    }
    loss_metadata = test_losses.metadata()
    for field in _RAW_BINDING_FIELDS[:-1]:
        if loss_metadata.get(field) != raw_binding[field]:
            raise ValueError(
                f"v4 test-loss archive differs from calibration result in {field}"
            )

    provenance = selection_result.get("provenance")
    formal = selection_result.get("formal_artifact_chain_verified") is True
    if formal or provenance is not None:
        if not isinstance(provenance, Mapping):
            raise ValueError("v4 formal result lacks bound provenance")
        for field in _RAW_BINDING_FIELDS:
            if provenance.get(field) != raw_binding[field]:
                raise ValueError(
                    f"v4 calibration provenance differs from result in {field}"
                )
    return {
        "verified": True,
        "mode": "v4_raw_logit_contract",
        "representation": LOGIT_REPRESENTATION,
        "raw_binding": raw_binding,
        "result_schema": schema,
        "loss_schema": loss_schema,
        "semantic_grid_sha256_verified": True,
        "detector_role_partition_verified": True,
        "external_empty_action_verified": True,
        "provenance_binding_verified": bool(formal or provenance is not None),
    }


def _same_grid(left: np.ndarray, right: np.ndarray) -> bool:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=0.0, atol=1e-8))


def _budget_value(result: dict[str, Any], name: str) -> float:
    value = result.get("budgets", {}).get(name)
    if isinstance(value, dict):
        value = value.get("value")
    if value is None:
        raise ValueError(f"Certification result is missing the {name} budget")
    return float(value)


def _mean(values: np.ndarray) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    return float(np.mean(selected)) if selected.size else None


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0.0 else None


def _required_bool(record: Mapping[str, Any], name: str) -> bool:
    value = record.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"Calibration result field {name} must be boolean")
    return value


def _optional_index(
    value: Any,
    *,
    name: str,
    num_thresholds: int,
    allow_none: bool,
    allow_terminal: bool = False,
) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer" + (" or null" if allow_none else ""))
    index = int(value)
    upper = num_thresholds if allow_terminal else num_thresholds - 1
    if not 0 <= index <= upper:
        suffix = " or the terminal reject rank" if allow_terminal else ""
        raise ValueError(f"{name} lies outside the threshold grid{suffix}")
    return index


def _index_sequence(
    values: Any,
    *,
    name: str,
    expected_count: int | None,
    num_thresholds: int,
    allow_none: bool,
    allow_terminal: bool = False,
) -> tuple[int | None, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an ordered index list")
    normalised = tuple(
        _optional_index(
            value,
            name=f"{name}[{position}]",
            num_thresholds=num_thresholds,
            allow_none=allow_none,
            allow_terminal=allow_terminal,
        )
        for position, value in enumerate(values)
    )
    if expected_count is not None and len(normalised) != expected_count:
        raise ValueError(
            f"{name} length {len(normalised)} differs from expected {expected_count}"
        )
    return normalised


def _same_optional_float(left: Any, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (bool, np.bool_)):
        return False
    try:
        value = float(left)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value) and np.isclose(value, right, rtol=0.0, atol=1e-8))


def _same_optional_float_sequence(
    recorded: Any,
    expected: Sequence[float | None],
) -> bool:
    if not isinstance(recorded, Sequence) or isinstance(recorded, (str, bytes)):
        return False
    return len(recorded) == len(expected) and all(
        _same_optional_float(left, right)
        for left, right in zip(recorded, expected)
    )


def _sigmoid_display(value: float) -> float:
    number = float(value)
    if number >= 0.0:
        return float(1.0 / (1.0 + np.exp(-number)))
    exponential = float(np.exp(number))
    return float(exponential / (1.0 + exponential))


def _first_action_difference(
    recorded: Sequence[int | None], expected: Sequence[int | None]
) -> int | None:
    return next(
        (
            position
            for position, (recorded_value, expected_value) in enumerate(
                zip(recorded, expected)
            )
            if recorded_value != expected_value
        ),
        None,
    )


def _formal_zero_bases(
    selection_result: Mapping[str, Any],
    *,
    calibration_ids: Sequence[str],
    test_ids: Sequence[str],
    num_thresholds: int,
    raw_binding: Mapping[str, Any] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Recover formal base actions from the hash-bound zero-label artifact."""

    provenance = selection_result.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Formal calibration result lacks provenance")
    zero_path_value = provenance.get("zero_result")
    zero_digest = provenance.get("zero_result_sha256")
    if not isinstance(zero_path_value, str) or not zero_path_value:
        raise ValueError("Formal calibration provenance lacks zero_result")
    if not isinstance(zero_digest, str) or len(zero_digest) != 64:
        raise ValueError("Formal calibration provenance has an invalid zero-result digest")
    zero_path = Path(zero_path_value)
    if not zero_path.is_file() or _file_digest(zero_path) != zero_digest:
        raise ValueError("Formal zero-label artifact is missing or its digest differs")
    try:
        zero_result = json.loads(zero_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Formal zero-label artifact is not readable JSON") from error
    if not isinstance(zero_result, Mapping):
        raise ValueError("Formal zero-label artifact must be a JSON object")
    if raw_binding is not None:
        for field in _RAW_BINDING_FIELDS:
            if zero_result.get(field) != raw_binding[field]:
                raise ValueError(
                    f"Formal zero-label artifact differs from v4 calibration in {field}"
                )
        if zero_result.get("prediction_rule") != LOGIT_PREDICTION_RULE:
            raise ValueError("Formal v4 zero-label prediction rule is invalid")
        if zero_result.get("empty_action") != empty_action_contract():
            raise ValueError("Formal v4 zero-label external +inf action is invalid")
        curve_path_value = zero_result.get("curve_checkpoint")
        if not isinstance(curve_path_value, str) or not curve_path_value:
            raise ValueError("Formal v4 zero-label artifact lacks curve_checkpoint")
        curve_path = Path(curve_path_value)
        if (
            not curve_path.is_file()
            or _file_digest(curve_path) != raw_binding["curve_checkpoint_sha256"]
        ):
            raise ValueError(
                "Formal v4 curve checkpoint is missing or its digest differs"
            )

    mapping = zero_result.get("threshold_indices_by_image")
    if mapping is None and "image_ids" in zero_result and "threshold_indices" in zero_result:
        zero_ids = zero_result.get("image_ids")
        zero_indices = zero_result.get("threshold_indices")
        if (
            not isinstance(zero_ids, Sequence)
            or isinstance(zero_ids, (str, bytes))
            or not isinstance(zero_indices, Sequence)
            or isinstance(zero_indices, (str, bytes))
            or len(zero_ids) != len(zero_indices)
        ):
            raise ValueError("Formal zero-label image IDs and indices are malformed")
        mapping = dict(zip(map(str, zero_ids), zero_indices))
    if mapping is not None:
        if not isinstance(mapping, Mapping):
            raise ValueError("Formal zero-label threshold mapping is malformed")
        missing = [
            image_id
            for image_id in list(calibration_ids) + list(test_ids)
            if image_id not in mapping
        ]
        if missing:
            raise ValueError(
                "Formal zero-label artifact lacks a base action for " + missing[0]
            )

        def mapped_bases(ids: Sequence[str], split: str) -> tuple[int, ...]:
            values = _index_sequence(
                [mapping[image_id] for image_id in ids],
                name=f"formal_{split}_zero_threshold_indices",
                expected_count=len(ids),
                num_thresholds=num_thresholds,
                allow_none=True,
            )
            return tuple(
                num_thresholds if value is None else int(value) for value in values
            )

        return mapped_bases(calibration_ids, "calibration"), mapped_bases(
            test_ids, "test"
        )

    zero_index = _optional_index(
        zero_result.get("threshold_index"),
        name="formal zero-label threshold_index",
        num_thresholds=num_thresholds,
        allow_none=False,
    )
    assert zero_index is not None
    return (
        (zero_index,) * len(calibration_ids),
        (zero_index,) * len(test_ids),
    )


def _audit_formal_flag(
    selection_result: Mapping[str, Any],
    *,
    formal: bool,
) -> None:
    protocol_audit = selection_result.get("protocol_audit")
    zero_audit = selection_result.get("zero_artifact_audit")
    provenance = selection_result.get("provenance")
    protocol_verified = (
        protocol_audit.get("verified")
        if isinstance(protocol_audit, Mapping)
        else None
    )
    zero_verified = zero_audit.get("verified") if isinstance(zero_audit, Mapping) else None
    provenance_formal = (
        provenance.get("formal_artifact_chain_verified")
        if isinstance(provenance, Mapping)
        else None
    )
    if formal:
        if protocol_verified is not True or zero_verified is not True:
            raise ValueError(
                "formal_artifact_chain_verified=true conflicts with artifact audits"
            )
        if provenance_formal is not True:
            raise ValueError(
                "formal_artifact_chain_verified=true conflicts with provenance"
            )
        if protocol_audit.get("allow_unverified_protocol") is not False:
            raise ValueError("A formal calibration cannot allow an unverified protocol")
        if provenance.get("protocol_fingerprint_verified") is not True:
            raise ValueError("Formal provenance lacks protocol fingerprint verification")
        if selection_result.get("success") is True and str(
            selection_result.get("guarantee_scope", "")
        ).lower().startswith("none:"):
            raise ValueError("A successful formal calibration has a null guarantee scope")
        return

    # A single Boolean must not be able to downgrade an otherwise-recorded formal run.
    if provenance_formal is True or (
        protocol_verified is True and zero_verified is True
    ):
        raise ValueError(
            "formal_artifact_chain_verified=false conflicts with the recorded formal evidence"
        )
    if selection_result.get("success") is True and not str(
        selection_result.get("guarantee_scope", "")
    ).lower().startswith("none:"):
        raise ValueError(
            "A diagnostic calibration cannot retain a formal guarantee scope"
        )


def _bound_test_actions(
    selection_result: Mapping[str, Any],
    *,
    calibration_ids: Sequence[str],
    test_ids: Sequence[str],
    thresholds: np.ndarray,
    evaluation_contract: Mapping[str, Any],
) -> tuple[tuple[int | None, ...] | None, dict[str, Any]]:
    """Validate a versioned calibration record and derive its test actions.

    A schema-less historical diagnostic record is retained as an explicitly
    unverified compatibility path.  Versioned records never use their stored
    ``selected_test_threshold_indices`` as the source of truth.
    """

    schema = selection_result.get("schema_version")
    if schema is None:
        if selection_result.get("formal_artifact_chain_verified") is True:
            raise ValueError("Formal calibration result lacks a supported schema")
        if "selection" in selection_result:
            raise ValueError("Calibration result with a selection record lacks schema_version")
        return None, {
            "verified": False,
            "mode": "legacy_unversioned_actions",
            "reason": "no versioned calibration selection record was available",
        }
    if schema not in (RESULT_SCHEMA_VERSION, RAW_LOGIT_RESULT_SCHEMA_VERSION):
        raise ValueError(f"Unsupported calibration result schema: {schema!r}")

    formal = _required_bool(selection_result, "formal_artifact_chain_verified")
    _audit_formal_flag(selection_result, formal=formal)
    selection = selection_result.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Versioned calibration result lacks its selection record")
    if selection.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError("Calibration selection schema is missing or unsupported")
    loss = selection_result.get("loss")
    expected_loss_schema = (
        RAW_LOGIT_LOSS_SCHEMA_VERSION
        if schema == RAW_LOGIT_RESULT_SCHEMA_VERSION
        else LOSS_SCHEMA_VERSION
    )
    raw_logit_mode = schema == RAW_LOGIT_RESULT_SCHEMA_VERSION
    if not isinstance(loss, Mapping) or loss.get("schema_version") != expected_loss_schema:
        raise ValueError("Calibration loss schema is missing or unsupported")

    success = _required_bool(selection_result, "success")
    reject = _required_bool(selection_result, "reject")
    nested_success = _required_bool(selection, "success")
    nested_reject = _required_bool(selection, "reject")
    if success != nested_success or reject != nested_reject:
        raise ValueError("Top-level success/reject flags differ from calibration selection")
    if success == reject:
        raise ValueError("Calibration success/reject state is internally inconsistent")
    expected_status = "selected_operating_point" if success else "reject"
    if selection.get("status") != expected_status:
        raise ValueError("Calibration selection status conflicts with success/reject")
    if selection_result.get("reason") != selection.get("reason"):
        raise ValueError("Top-level reason differs from calibration selection")

    num_thresholds = int(np.asarray(thresholds).size)
    num_calibration = len(calibration_ids)
    calibration_bases_public = _index_sequence(
        selection_result.get("calibration_zero_threshold_indices"),
        name="calibration_zero_threshold_indices",
        expected_count=num_calibration,
        num_thresholds=num_thresholds,
        allow_none=True,
    )
    calibration_bases = tuple(
        num_thresholds if value is None else int(value)
        for value in calibration_bases_public
    )
    nested_bases_raw = _index_sequence(
        selection.get("zero_threshold_indices"),
        name="selection.zero_threshold_indices",
        expected_count=num_calibration,
        num_thresholds=num_thresholds,
        allow_none=False,
        allow_terminal=True,
    )
    nested_bases = tuple(int(value) for value in nested_bases_raw if value is not None)
    if calibration_bases != nested_bases:
        raise ValueError(
            "Calibration base indices differ from the embedded selection record"
        )

    test_bases_public = _index_sequence(
        selection_result.get("test_zero_threshold_indices"),
        name="test_zero_threshold_indices",
        expected_count=len(test_ids),
        num_thresholds=num_thresholds,
        allow_none=True,
    )
    test_bases = tuple(
        num_thresholds if value is None else int(value) for value in test_bases_public
    )

    adaptation_mode = selection_result.get("adaptation_mode")
    if adaptation_mode == "global_zero_threshold":
        zero_index = _optional_index(
            selection_result.get("zero_threshold_index"),
            name="zero_threshold_index",
            num_thresholds=num_thresholds,
            allow_none=False,
        )
        assert zero_index is not None
        if calibration_bases != (zero_index,) * num_calibration or test_bases != (
            zero_index,
        ) * len(test_ids):
            raise ValueError("Global zero threshold does not match all recorded base actions")
        expected_zero_threshold = float(thresholds[zero_index])
        if not _same_optional_float(
            selection_result.get("zero_threshold"), expected_zero_threshold
        ):
            raise ValueError("Recorded zero-threshold value does not match its grid index")
        if raw_logit_mode and (
            not _same_optional_float(
                selection_result.get("zero_logit_threshold"),
                expected_zero_threshold,
            )
            or not _same_optional_float(
                selection_result.get("zero_probability_threshold"),
                _sigmoid_display(expected_zero_threshold),
            )
        ):
            raise ValueError("Recorded raw-logit zero-threshold displays are inconsistent")
        expected_nested_zero = zero_index
    elif adaptation_mode == "sample_adaptive_zero_plus_shared_offset":
        if selection_result.get("zero_threshold_index") is not None:
            raise ValueError("Sample-adaptive calibration cannot record a scalar zero index")
        if selection_result.get("zero_threshold") is not None:
            raise ValueError("Sample-adaptive calibration cannot record a scalar zero threshold")
        if raw_logit_mode and (
            selection_result.get("zero_logit_threshold") is not None
            or selection_result.get("zero_probability_threshold") is not None
        ):
            raise ValueError(
                "Sample-adaptive raw-logit calibration cannot record scalar zero displays"
            )
        expected_nested_zero = (
            calibration_bases[0]
            if calibration_bases and len(set(calibration_bases)) == 1
            else None
        )
    else:
        raise ValueError("Calibration result has an unsupported adaptation_mode")
    if selection.get("zero_index") != expected_nested_zero:
        raise ValueError("Calibration selection zero_index differs from its base indices")

    top_offset = _optional_index(
        selection_result.get("offset_rank"),
        name="offset_rank",
        num_thresholds=num_thresholds,
        allow_none=True,
        allow_terminal=True,
    )
    nested_offset = _optional_index(
        selection.get("offset_rank"),
        name="selection.offset_rank",
        num_thresholds=num_thresholds,
        allow_none=True,
        allow_terminal=True,
    )
    if top_offset != nested_offset:
        raise ValueError("Top-level offset_rank differs from calibration selection")
    if success and nested_offset is None:
        raise ValueError("Successful calibration selection lacks offset_rank")

    expected_calibration_actions: tuple[int | None, ...]
    if nested_offset is None:
        expected_calibration_actions = ()
    else:
        expected_calibration_actions = tuple(
            base + nested_offset if base + nested_offset < num_thresholds else None
            for base in calibration_bases
        )
    recorded_calibration_actions = _index_sequence(
        selection.get("selected_threshold_indices"),
        name="selection.selected_threshold_indices",
        expected_count=(
            num_calibration if nested_offset is not None else 0
        ),
        num_thresholds=num_thresholds,
        allow_none=True,
    )
    if recorded_calibration_actions != expected_calibration_actions:
        raise ValueError("Calibration selected actions do not equal base + offset_rank")

    expected_scalar_index = None
    if (
        success
        and expected_calibration_actions
        and all(value is not None for value in expected_calibration_actions)
        and len(set(expected_calibration_actions)) == 1
    ):
        expected_scalar_index = int(expected_calibration_actions[0])
    nested_scalar = _optional_index(
        selection.get("selected_threshold_index"),
        name="selection.selected_threshold_index",
        num_thresholds=num_thresholds,
        allow_none=True,
    )
    top_scalar = _optional_index(
        selection_result.get("selected_threshold_index"),
        name="selected_threshold_index",
        num_thresholds=num_thresholds,
        allow_none=True,
    )
    if nested_scalar != expected_scalar_index or top_scalar != expected_scalar_index:
        raise ValueError("Recorded scalar selected index differs from calibrated actions")
    expected_threshold = (
        float(thresholds[expected_scalar_index])
        if expected_scalar_index is not None
        else None
    )
    if not _same_optional_float(
        selection_result.get("selected_threshold"), expected_threshold
    ):
        raise ValueError("Recorded selected-threshold value does not match its grid index")
    if raw_logit_mode and (
        not _same_optional_float(
            selection_result.get("selected_logit_threshold"), expected_threshold
        )
        or not _same_optional_float(
            selection_result.get("selected_probability_threshold"),
            _sigmoid_display(expected_threshold)
            if expected_threshold is not None
            else None,
        )
    ):
        raise ValueError("Recorded raw-logit selected-threshold displays are inconsistent")

    candidate_ranks = _index_sequence(
        selection.get("candidate_offset_ranks"),
        name="selection.candidate_offset_ranks",
        expected_count=None,
        num_thresholds=num_thresholds,
        allow_none=False,
        allow_terminal=True,
    )
    if tuple(sorted(set(candidate_ranks))) != candidate_ranks:
        raise ValueError("Calibration candidate offset ranks are not sorted and unique")
    trace = selection.get("candidate_trace")
    if not isinstance(trace, Sequence) or isinstance(trace, (str, bytes)):
        raise ValueError("Calibration candidate trace is malformed")
    if nested_offset is not None:
        if nested_offset not in candidate_ranks:
            raise ValueError("Selected offset_rank is absent from the candidate ranks")
        matching_trace = [
            item
            for item in trace
            if isinstance(item, Mapping) and item.get("offset_rank") == nested_offset
        ]
        if len(matching_trace) != 1:
            raise ValueError("Selected offset_rank is not uniquely recorded in candidate trace")
        selected_trace = matching_trace[0]
        if selected_trace.get("threshold_index") != expected_scalar_index:
            raise ValueError("Candidate trace threshold index differs from selected action")
        if success and not (
            selected_trace.get("feasible") is True
            and selected_trace.get("eligible_for_formal_success") is True
            and selected_trace.get("terminal_reject") is False
            and selected_trace.get("reject_count") == 0
        ):
            raise ValueError("Successful selected offset is not feasible in candidate trace")

    if success:
        assert nested_offset is not None
        expected_test_actions = tuple(
            base + nested_offset if base + nested_offset < num_thresholds else None
            for base in test_bases
        )
    else:
        expected_test_actions = ()
    recorded_test_actions = _index_sequence(
        selection_result.get("selected_test_threshold_indices"),
        name="selected_test_threshold_indices",
        expected_count=(len(test_ids) if success else 0),
        num_thresholds=num_thresholds,
        allow_none=True,
    )
    if recorded_test_actions != expected_test_actions:
        position = _first_action_difference(recorded_test_actions, expected_test_actions)
        suffix = f" at test position {position}" if position is not None else ""
        raise ValueError(
            "Recorded test action differs from test base + calibrated offset_rank"
            + suffix
        )
    if raw_logit_mode:
        expected_test_logits = [
            float(thresholds[value]) if value is not None else None
            for value in expected_test_actions
        ]
        expected_test_probabilities = [
            _sigmoid_display(value) if value is not None else None
            for value in expected_test_logits
        ]
        if not _same_optional_float_sequence(
            selection_result.get("selected_test_logit_thresholds"),
            expected_test_logits,
        ):
            raise ValueError("Recorded per-test raw-logit thresholds are inconsistent")
        if not _same_optional_float_sequence(
            selection_result.get("selected_test_probability_thresholds"),
            expected_test_probabilities,
        ):
            raise ValueError("Recorded per-test probability displays are inconsistent")

    expected_mode = (
        f"few_shot_{adaptation_mode}_grid_rank_crc"
        if success
        else "rejected_no_operating_point"
    )
    if selection_result.get("mode") != expected_mode:
        raise ValueError("Calibration mode conflicts with its success/adaptation state")
    expected_test_reject_rate = (
        float(np.mean([value is None for value in expected_test_actions]))
        if expected_test_actions
        else 1.0
    )
    if not _same_optional_float(
        selection_result.get("test_reject_rate"), expected_test_reject_rate
    ):
        raise ValueError("Recorded test_reject_rate differs from calibrated actions")

    if formal:
        formal_calibration_bases, formal_test_bases = _formal_zero_bases(
            selection_result,
            calibration_ids=calibration_ids,
            test_ids=test_ids,
            num_thresholds=num_thresholds,
            raw_binding=evaluation_contract.get("raw_binding"),
        )
        if formal_calibration_bases != calibration_bases:
            raise ValueError(
                "Calibration base actions differ from the formal zero-label artifact"
            )
        if formal_test_bases != test_bases:
            raise ValueError("Test base actions differ from the formal zero-label artifact")

    return expected_test_actions, {
        "verified": True,
        "mode": "base_plus_calibration_offset",
        "calibration_result_schema": schema,
        "selection_schema": SELECTION_SCHEMA_VERSION,
        "loss_schema": expected_loss_schema,
        "representation_contract": {
            key: value
            for key, value in evaluation_contract.items()
            if key != "raw_binding"
        },
        "formal_artifact_chain_verified": formal,
        "test_base_count": len(test_bases),
        "offset_rank": nested_offset,
        "recorded_actions_match_recomputation": True,
    }


def _audit_guarantee_scope(selection_result: dict[str, Any]) -> str | None:
    """Qualify component-envelope control without overstating raw-risk identity."""

    recorded = selection_result.get("guarantee_scope")
    if recorded is None:
        return None
    recorded_text = str(recorded)
    if recorded_text.lower().startswith("none:"):
        return recorded_text
    loss_mode = selection_result.get("loss", {}).get("mode")
    if loss_mode == "budget_violation":
        return (
            "finite-sample corrected control of the marginal joint violation "
            "loss formed from pixel risk and the conservative per-image "
            "suffix-max component-risk envelope; because that envelope "
            "dominates realised raw component risk at the selected threshold, "
            "the result conservatively implies raw component-budget control "
            "under the recorded assumptions, but the two risks are not an identity; "
            "this risk statement does not guarantee deployment coverage, which is "
            "reported separately"
        )
    if loss_mode == "risk_ratio":
        return (
            "finite-sample corrected control of an expected bounded risk-ratio "
            "surrogate, not a JointBSR guarantee; its component term uses a "
            "conservative suffix-max envelope that dominates raw component risk, "
            "and it does not guarantee deployment coverage"
        )
    return (
        "the recorded non-null guarantee requires an explicit loss mode before "
        "it can be restated; any component claim is limited to conservative "
        "suffix-max-envelope control that implies raw component-budget control"
    )


def evaluate_selected_operating_point(
    selection_result: dict[str, Any], test_losses: CalibrationLosses
) -> dict[str, Any]:
    """Evaluate, but never tune, one recorded grid operating point."""

    calibration_ids = selection_result.get("calibration_image_ids")
    if calibration_ids is None:
        raise ValueError("Selection result lacks calibration_image_ids for leakage audit")
    adaptation_ids = selection_result.get("adaptation_image_ids") or []
    if adaptation_ids:
        _, calibration_ids, test_ids = assert_three_way_disjoint_image_ids(
            adaptation_ids, calibration_ids, test_losses.image_ids
        )
    else:
        calibration_ids, test_ids = assert_disjoint_image_ids(
            calibration_ids, test_losses.image_ids
        )
    checked_test_ids = selection_result.get("test_image_ids_checked")
    if checked_test_ids is not None:
        checked_test_ids = tuple(map(str, checked_test_ids))
        if checked_test_ids != tuple(test_ids):
            if set(checked_test_ids) == set(test_ids):
                raise ValueError(
                    "The evaluated test ID order differs from the order frozen at "
                    "calibration; positional per-image actions cannot be remapped safely"
                )
            raise ValueError(
                "The evaluated test split differs from the split whose IDs were "
                "checked at calibration"
            )
    representation_contract = _evaluation_contract(selection_result, test_losses)
    recorded_grid = selection_result.get("thresholds")
    if recorded_grid is None or not _same_grid(recorded_grid, test_losses.thresholds):
        raise ValueError("Selection result and test archive use different threshold grids")
    if not np.isclose(
        _budget_value(selection_result, "pixel"), test_losses.pixel_budget, rtol=0.0, atol=0.0
    ):
        raise ValueError("Pixel budget differs from the calibration result")
    if not np.isclose(
        _budget_value(selection_result, "component"),
        test_losses.component_budget,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("Component budget differs from the calibration result")

    bound_actions, action_contract_audit = _bound_test_actions(
        selection_result,
        calibration_ids=calibration_ids,
        test_ids=test_ids,
        thresholds=test_losses.thresholds,
        evaluation_contract=representation_contract,
    )

    base: dict[str, Any] = {
        "schema_version": (
            RAW_LOGIT_EVALUATION_SCHEMA_VERSION
            if representation_contract.get("representation")
            == LOGIT_REPRESENTATION
            else EVALUATION_SCHEMA_VERSION
        ),
        "representation": representation_contract.get("representation"),
        "selection_success": bool(selection_result.get("success", False)),
        "reject": bool(selection_result.get("reject", True)),
        "num_test_images": test_losses.num_images,
        "split_audit": {
            "calibration_count": len(calibration_ids),
            "test_count": len(test_ids),
            "overlap_count": 0,
            "ordered_test_ids_verified": checked_test_ids is not None,
        },
        "budgets": selection_result["budgets"],
        "test_labels_used_for_selection": False,
        "test_action_contract_audit": action_contract_audit,
        "representation_contract_audit": representation_contract,
    }
    if not selection_result.get("success", False) or selection_result.get("reject", True):
        base.update(
            {
                "mode": "rejected_no_test_operating_point",
                "success": False,
                "reason": selection_result.get("reason", "selection_rejected"),
                "selected_threshold_index": None,
                "selected_threshold": None,
                "guarantee_scope": (
                    "none: calibration did not select a non-reject operating point"
                ),
                "metrics": None,
            }
        )
        return base

    selected_per_image = (
        list(bound_actions)
        if bound_actions is not None
        else selection_result.get("selected_test_threshold_indices")
    )
    if selected_per_image is None:
        index = selection_result.get("selected_threshold_index")
        if index is None:
            raise ValueError(
                "Successful selection result lacks scalar or per-image threshold indices"
            )
        selected_per_image = [int(index)] * test_losses.num_images
    elif checked_test_ids is None:
        raise ValueError(
            "Per-image test actions require ordered test_image_ids_checked for "
            "identity-safe alignment"
        )
    if len(selected_per_image) != test_losses.num_images:
        raise ValueError("selected_test_threshold_indices length differs from test split")
    normalised_actions: list[int | None] = []
    for value in selected_per_image:
        if value is None:
            normalised_actions.append(None)
            continue
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(
                "selected_test_threshold_indices must contain grid-rank integers or null"
            )
        normalised_actions.append(int(value))
    selected_per_image = normalised_actions
    active = np.asarray([value is not None for value in selected_per_image], dtype=bool)
    indices = np.asarray(
        [value if value is not None else 0 for value in selected_per_image],
        dtype=np.int64,
    )
    if np.any((indices[active] < 0) | (indices[active] >= test_losses.num_thresholds)):
        raise ValueError("A selected per-image threshold index lies outside the test grid")
    rows = np.arange(test_losses.num_images, dtype=np.int64)

    def gather(curves: np.ndarray) -> np.ndarray:
        values = np.zeros(test_losses.num_images, dtype=np.float64)
        values[active] = np.asarray(curves)[rows[active], indices[active]]
        return values

    num_active = int(np.sum(active))
    num_rejected = int(test_losses.num_images - num_active)
    coverage_rate = float(num_active / test_losses.num_images)
    pixel_risk = gather(test_losses.pixel_risk)
    component_raw = gather(test_losses.component_risk_raw)
    component_envelope = gather(test_losses.component_risk_envelope)
    pixel_loss = gather(test_losses.pixel_loss)
    component_loss = gather(test_losses.component_loss)
    joint_loss = gather(test_losses.joint_loss)
    pixel_ok = pixel_risk <= test_losses.pixel_budget
    component_ok = component_raw <= test_losses.component_budget
    conservative_component_ok = component_envelope <= test_losses.component_budget
    selected_fp_pixels = gather(test_losses.false_positive_pixels)
    selected_fp_components = gather(test_losses.false_positive_components)
    total_exposure = float(np.sum(test_losses.total_pixels))
    active_exposure = float(np.sum(test_losses.total_pixels[active]))
    aggregate_pixel_risk = float(
        np.sum(selected_fp_pixels) / total_exposure
    )
    aggregate_component_risk = float(
        np.sum(selected_fp_components) / (total_exposure / 1_000_000.0)
    )
    aggregate_pixel_risk_active_only = (
        float(np.sum(selected_fp_pixels[active]) / active_exposure)
        if active_exposure > 0.0
        else None
    )
    aggregate_component_risk_active_only = (
        float(
            np.sum(selected_fp_components[active])
            / (active_exposure / 1_000_000.0)
        )
        if active_exposure > 0.0
        else None
    )
    unique_active_indices = np.unique(indices[active]) if np.any(active) else np.asarray([])
    scalar_index = (
        int(unique_active_indices[0])
        if np.all(active) and unique_active_indices.size == 1
        else None
    )
    threshold = (
        float(test_losses.thresholds[scalar_index]) if scalar_index is not None else None
    )
    recorded_index = selection_result.get("selected_threshold_index")
    recorded_threshold = selection_result.get("selected_threshold")
    if scalar_index is not None:
        if recorded_index is not None and int(recorded_index) != scalar_index:
            raise ValueError("Recorded scalar threshold index differs from test actions")
        if recorded_threshold is not None and not np.isclose(
            float(recorded_threshold), threshold, rtol=0.0, atol=1e-8
        ):
            raise ValueError("Recorded threshold value does not match its grid index")
    metrics = {
        "test_action_semantics": (
            "null threshold indices are explicit no-detection rejections; they "
            "remain in overall image/exposure/ground-truth denominators"
        ),
        "metric_scope_interpretation": (
            "overall budget metrics include no-detection rejects as zero-risk "
            "actions; active-only budget and Pd metrics are conditional diagnostics "
            "and are not a substitute for reporting coverage"
        ),
        "active_image_count": num_active,
        "rejected_image_count": num_rejected,
        "no_detection_action_count": num_rejected,
        "coverage_rate": coverage_rate,
        "reject_rate": float(1.0 - coverage_rate),
        "mean_pixel_risk": _mean(pixel_risk),
        "mean_pixel_risk_active_only": _masked_mean(pixel_risk, active),
        "aggregate_pixel_risk": aggregate_pixel_risk,
        "aggregate_pixel_risk_active_only": aggregate_pixel_risk_active_only,
        "pixel_risk_unit": PIXEL_BUDGET_UNIT,
        "mean_component_risk_raw": _mean(component_raw),
        "mean_component_risk_raw_active_only": _masked_mean(component_raw, active),
        "mean_component_risk_suffix_max": _mean(component_envelope),
        "mean_component_risk_suffix_max_active_only": _masked_mean(
            component_envelope, active
        ),
        "aggregate_component_risk_raw": aggregate_component_risk,
        "aggregate_component_risk_raw_active_only": (
            aggregate_component_risk_active_only
        ),
        "component_risk_unit": COMPONENT_BUDGET_UNIT,
        "mean_pixel_loss": _mean(pixel_loss),
        "mean_pixel_loss_active_only": _masked_mean(pixel_loss, active),
        "mean_component_loss": _mean(component_loss),
        "mean_component_loss_active_only": _masked_mean(component_loss, active),
        "mean_joint_bounded_loss": _mean(joint_loss),
        "mean_joint_bounded_loss_active_only": _masked_mean(joint_loss, active),
        "max_joint_bounded_loss": float(np.max(joint_loss)),
        "max_joint_bounded_loss_active_only": (
            float(np.max(joint_loss[active])) if num_active else None
        ),
        "pixel_budget_satisfaction_rate_per_image": _mean(pixel_ok),
        "pixel_budget_satisfaction_rate_per_image_active_only": _masked_mean(
            pixel_ok, active
        ),
        "component_budget_satisfaction_rate_per_image_raw": _mean(component_ok),
        "component_budget_satisfaction_rate_per_image_raw_active_only": _masked_mean(
            component_ok, active
        ),
        "component_budget_satisfaction_rate_per_image_suffix_max": _mean(
            conservative_component_ok
        ),
        "component_budget_satisfaction_rate_per_image_suffix_max_active_only": (
            _masked_mean(conservative_component_ok, active)
        ),
        "dual_budget_satisfaction_rate_per_image_raw": _mean(pixel_ok & component_ok),
        "dual_budget_satisfaction_rate_per_image_raw_active_only": _masked_mean(
            pixel_ok & component_ok, active
        ),
        "joint_budget_satisfaction_rate_per_image_suffix_max": _mean(
            pixel_ok & conservative_component_ok
        ),
        "joint_budget_satisfaction_rate_per_image_suffix_max_including_rejects_as_no_detection": _mean(
            pixel_ok & conservative_component_ok
        ),
        "joint_budget_satisfaction_rate_per_image_suffix_max_active_only": _masked_mean(
            pixel_ok & conservative_component_ok, active
        ),
        "joint_budget_violation_rate_per_image_suffix_max": _mean(
            ~(pixel_ok & conservative_component_ok)
        ),
        "joint_budget_violation_rate_per_image_suffix_max_active_only": _masked_mean(
            ~(pixel_ok & conservative_component_ok), active
        ),
        "component_control_interpretation": (
            "suffix-max component risk is a conservative majorant; control of "
            "its violation implies conservative control of realised raw component "
            "risk and is not an identity with the raw curve"
        ),
    }
    if test_losses.tp_object_counts is not None and test_losses.gt_object_counts is not None:
        selected_tp = gather(test_losses.tp_object_counts)
        gt = np.asarray(test_losses.gt_object_counts, dtype=np.float64)
        total_tp = float(np.sum(selected_tp))
        total_gt = float(np.sum(gt))
        active_tp = float(np.sum(selected_tp[active]))
        active_gt = float(np.sum(gt[active]))
        rejected_gt = float(np.sum(gt[~active]))
        metrics.update(
            {
                "true_positive_objects": int(total_tp),
                "ground_truth_objects": int(total_gt),
                "ground_truth_objects_active_only": int(active_gt),
                "ground_truth_objects_in_rejected_images": int(rejected_gt),
                "true_positive_objects_active_only": int(active_tp),
                "pd_object_aggregate": _ratio(total_tp, total_gt),
                "pd_object_aggregate_including_rejects_as_no_detection": _ratio(
                    total_tp, total_gt
                ),
                "pd_object_aggregate_active_only": _ratio(active_tp, active_gt),
            }
        )
    else:
        metrics.update(
            {
                "true_positive_objects": None,
                "ground_truth_objects": None,
                "ground_truth_objects_active_only": None,
                "ground_truth_objects_in_rejected_images": None,
                "true_positive_objects_active_only": None,
                "pd_object_aggregate": None,
                "pd_object_aggregate_including_rejects_as_no_detection": None,
                "pd_object_aggregate_active_only": None,
            }
        )
    selected_test_actions = [
        {
            "image_id": str(test_ids[row_index]),
            "action": (
                "threshold_grid_action" if value is not None else "no_detection_reject"
            ),
            "threshold_index": value,
            "threshold": (
                float(test_losses.thresholds[value]) if value is not None else None
            ),
        }
        for row_index, value in enumerate(selected_per_image)
    ]
    base.update(
        {
            "mode": "independent_test_audit_of_selected_operating_point",
            "success": True,
            "reason": "evaluated_without_test_time_reselection",
            "selected_threshold_index": scalar_index,
            "selected_threshold": threshold,
            "selected_test_threshold_indices": list(selected_per_image),
            "selected_test_thresholds": [
                float(test_losses.thresholds[int(value)]) if value is not None else None
                for value in selected_per_image
            ],
            "selected_test_actions": selected_test_actions,
            "has_test_rejections": bool(num_rejected),
            "all_test_actions_rejected": bool(num_active == 0),
            "test_action_audit": {
                "active_count": num_active,
                "no_detection_reject_count": num_rejected,
                "coverage_rate": coverage_rate,
                "reject_rate": float(1.0 - coverage_rate),
                "ordered_id_alignment_verified": checked_test_ids is not None,
            },
            "offset_rank": selection_result.get("offset_rank"),
            "guarantee_scope": _audit_guarantee_scope(selection_result),
            "metrics": metrics,
        }
    )
    return base


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-result", required=True)
    parser.add_argument("--test-curves", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result_path = Path(args.selection_result)
    test_path = Path(args.test_curves)
    selection_result = json.loads(result_path.read_text(encoding="utf-8"))
    recorded_test_digest = selection_result.get("provenance", {}).get(
        "test_curves_sha256"
    )
    actual_test_digest = _file_digest(test_path)
    if recorded_test_digest is not None and recorded_test_digest != actual_test_digest:
        raise ValueError(
            "The test archive differs byte-for-byte from the split frozen during calibration"
        )
    test_data = load_count_curve_archive(test_path)
    test_losses = build_calibration_losses(
        **test_data,
        pixel_budget=_budget_value(selection_result, "pixel"),
        component_budget=_budget_value(selection_result, "component"),
        loss_mode=selection_result.get("loss", {}).get("mode", "budget_violation"),
    )
    audit = evaluate_selected_operating_point(selection_result, test_losses)
    audit["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    audit["assumptions"] = [
        "the test split is independent of and exchangeable with the calibration split",
        "test image IDs are unique and disjoint from all calibration image IDs",
        "the selected rank, budgets, loss, detector, and preprocessing were frozen before test evaluation",
        "test outcomes are used only for audit metrics and never for threshold reselection",
        "per-image null actions are no-detection rejections retained in overall exposure and ground-truth denominators",
        "the component suffix-max envelope conservatively dominates realised raw component risk at each selected threshold",
    ]
    audit["provenance"] = {
        "command": shlex.join(sys.argv),
        "selection_result": str(result_path.resolve()),
        "selection_result_sha256": _file_digest(result_path),
        "test_curves": str(test_path.resolve()),
        "test_curves_sha256": _file_digest(test_path),
        "test_labels_used_for_selection": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "success": audit["success"],
                "reject": audit["reject"],
                "selected_threshold": audit["selected_threshold"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
