"""Audit zero-label threshold actions on independent labelled count curves.

This evaluator does not select or change an operating point.  It binds the
already-frozen scalar/per-image actions in a zero-label result to image IDs in
an integrity-checked count-curve archive and reports risk, budget satisfaction,
coverage, rejection, and object-level detection probability.  No formal claim
is attached to the empirical zero-label mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from certification.build_calibration_losses import (
    LOSS_MODE_BUDGET_VIOLATION,
    build_calibration_losses,
    load_count_curve_archive,
    load_count_curve_provenance,
)
from evaluation.artifact_integrity import file_sha256, verify_score_map_directory
from evaluation.target_stage_separation import (
    PAIR_AUDIT_SCHEMA_VERSION,
    verify_selection_freeze,
)
from rc_irstd.utils.io import atomic_write_json
from .select_zero_label_threshold import (
    SELECTION_DATA_CONTRACT_SCHEMA_VERSION,
    ZERO_RESULT_SCHEMA_VERSION,
)


ZERO_LABEL_EVALUATION_SCHEMA_VERSION = "rc-v2-zero-label-evaluation-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PAIR_SPATIAL_FIELDS = frozenset(
    {
        "target_dataset",
        "requested_split",
        "split_role",
        "split_authority_verified",
        "spatial_mode",
        "base_hw",
        "pad_multiple",
        "score_representation",
        "probability_dtype",
        "logit_dtype",
        "probability_transform",
        "probability_clipping",
        "inference_autocast_enabled",
        "warm_flag",
        "model_backend",
        "source_datasets",
    }
)
_PAIR_REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "target_dataset",
        "requested_split",
        "split_role",
        "split_authority_verified",
        "spatial_mode",
        "pad_multiple",
        "score_representation",
        "probability_dtype",
        "logit_dtype",
        "probability_transform",
        "probability_clipping",
        "inference_autocast_enabled",
        "warm_flag",
        "model_backend",
        "source_datasets",
    }
)


def _selection_contract_error(message: str) -> ValueError:
    return ValueError(f"Unverified zero-label selection contract: {message}")


def _validate_action_identity_contract(zero_result: Mapping[str, Any]) -> None:
    protocol = zero_result.get("adaptation_protocol")
    if protocol not in {"static_cross_fit", "external_causal_statistics"}:
        return
    evaluation_rows = zero_result.get("evaluation_ids")
    row_indices = zero_result.get("threshold_indices")
    mapping = zero_result.get("threshold_indices_by_image")
    if not isinstance(evaluation_rows, list) or not evaluation_rows:
        raise _selection_contract_error("adaptive result lacks evaluation_ids")
    if not isinstance(row_indices, list) or len(row_indices) != len(evaluation_rows):
        raise _selection_contract_error(
            "threshold_indices do not align with evaluation rows"
        )
    if not isinstance(mapping, Mapping):
        raise _selection_contract_error(
            "adaptive result lacks threshold_indices_by_image"
        )
    expected: dict[str, Any] = {}
    for row_index, (raw_ids, threshold_index) in enumerate(
        zip(evaluation_rows, row_indices)
    ):
        if not isinstance(raw_ids, list) or not raw_ids:
            raise _selection_contract_error(
                f"evaluation_ids row {row_index} must be a non-empty list"
            )
        row_ids = [str(value) for value in raw_ids]
        if any(not value.strip() for value in row_ids) or len(row_ids) != len(set(row_ids)):
            raise _selection_contract_error(
                f"evaluation_ids row {row_index} contains invalid or duplicate IDs"
            )
        for image_id in row_ids:
            if image_id in expected:
                raise _selection_contract_error(
                    "evaluation IDs are not globally unique"
                )
            expected[image_id] = threshold_index
    normalised_mapping = {str(key): value for key, value in mapping.items()}
    if set(normalised_mapping) != set(expected):
        raise _selection_contract_error(
            "threshold mapping does not exactly cover evaluation IDs"
        )
    for image_id, expected_index in expected.items():
        if normalised_mapping[image_id] != expected_index:
            raise _selection_contract_error(
                f"threshold mapping disagrees with row action for {image_id!r}"
            )
    if zero_result.get("num_windows") != len(evaluation_rows):
        raise _selection_contract_error("num_windows differs from evaluation rows")


def validate_zero_result_selection_contract(
    zero_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the no-label selection statement before reproducing it in an audit."""

    if zero_result.get("schema_version") != ZERO_RESULT_SCHEMA_VERSION:
        raise _selection_contract_error("zero-result schema is missing or unsupported")
    if zero_result.get("mode") != "zero_label_empirical_adaptation":
        raise _selection_contract_error("mode is not zero_label_empirical_adaptation")
    data_contract = zero_result.get("selection_data_contract")
    if not isinstance(data_contract, Mapping):
        raise _selection_contract_error("selection_data_contract must be an object")
    if data_contract.get("schema_version") != SELECTION_DATA_CONTRACT_SCHEMA_VERSION:
        raise _selection_contract_error("selection data-contract schema is unsupported")
    if data_contract.get("masks_read") is not False:
        raise _selection_contract_error("selection contract must record masks_read=false")
    if zero_result.get("masks_read") is not False:
        raise _selection_contract_error("zero result must record masks_read=false")
    if data_contract.get("evaluation_labels_or_masks_used") is not False:
        raise _selection_contract_error(
            "selection contract must record evaluation_labels_or_masks_used=false"
        )

    adaptation_protocol = zero_result.get("adaptation_protocol")
    expected_contracts = {
        "static_cross_fit": (
            "static_cross_fit_complement_folds",
            "one_complement_fold_prediction_to_held_out_fold_ids",
        ),
        "external_causal_statistics": (
            "causal_adaptation_blocks_A",
            "one_A_block_prediction_to_its_future_E_identity",
        ),
        "causal_warmup": (
            "causal_warmup_score_maps",
            "one_global_warmup_prediction",
        ),
        "transductive": (
            "transductive_score_maps",
            "one_global_transductive_prediction",
        ),
    }
    if adaptation_protocol not in expected_contracts:
        raise _selection_contract_error(
            f"unsupported adaptation_protocol {adaptation_protocol!r}"
        )
    expected_statistics, expected_mapping = expected_contracts[adaptation_protocol]
    if data_contract.get("statistics_computed_from") != expected_statistics:
        raise _selection_contract_error(
            "statistics_computed_from is inconsistent with adaptation_protocol"
        )
    if data_contract.get("threshold_mapping_rule") != expected_mapping:
        raise _selection_contract_error(
            "threshold_mapping_rule is inconsistent with adaptation_protocol"
        )

    if adaptation_protocol in {"static_cross_fit", "external_causal_statistics"}:
        if data_contract.get("deployment_identity_contract_verified") is not True:
            raise _selection_contract_error(
                "deployment identity contract is not verified"
            )
        artifact = zero_result.get("statistics_artifact")
        if not isinstance(artifact, Mapping):
            raise _selection_contract_error("adaptive result lacks statistics_artifact")
        identity = artifact.get("identity_contract")
        if not isinstance(identity, Mapping) or identity.get("verified") is not True:
            raise _selection_contract_error(
                "statistics artifact identity contract is not verified"
            )
        provenance = artifact.get("provenance")
        if not isinstance(provenance, Mapping) or provenance.get("masks_read") is not False:
            raise _selection_contract_error(
                "statistics artifact does not prove masks_read=false"
            )
        _validate_action_identity_contract(zero_result)

    if adaptation_protocol == "static_cross_fit":
        if data_contract.get("formal_crc_eligible") is not False:
            raise _selection_contract_error("static cross-fit cannot be CRC-formal")
        if data_contract.get("static_checkpoint_compatibility_verified") is not True:
            raise _selection_contract_error(
                "static checkpoint compatibility is not verified"
            )
        static_audit = zero_result.get("static_checkpoint_compatibility_audit")
        if not isinstance(static_audit, Mapping) or static_audit.get("verified") is not True:
            raise _selection_contract_error(
                "static checkpoint compatibility audit is missing"
            )
        provenance = zero_result["statistics_artifact"]["provenance"]
        for field, expected in (
            ("masks_read", False),
            ("full_test_coverage", True),
            ("formal_crc_eligible", False),
            ("score_integrity_verified", True),
        ):
            if provenance.get(field) is not expected:
                raise _selection_contract_error(
                    f"static provenance must record {field}={expected!r}"
                )
    elif adaptation_protocol == "external_causal_statistics":
        formal = data_contract.get("formal_crc_eligible")
        if not isinstance(formal, bool):
            raise _selection_contract_error("formal_crc_eligible must be boolean")
        if formal:
            provenance_audit = zero_result.get("causal_formal_provenance_audit")
            checkpoint_audit = zero_result.get("curve_checkpoint_deployment_audit")
            if (
                not isinstance(provenance_audit, Mapping)
                or provenance_audit.get("verified") is not True
            ):
                raise _selection_contract_error(
                    "formal causal provenance audit is not verified"
                )
            if (
                not isinstance(checkpoint_audit, Mapping)
                or checkpoint_audit.get("verified") is not True
            ):
                raise _selection_contract_error(
                    "formal checkpoint/deployment audit is not verified"
                )
    return {
        "verified": True,
        "schema_version": SELECTION_DATA_CONTRACT_SCHEMA_VERSION,
        "adaptation_protocol": adaptation_protocol,
        "masks_read": False,
        "evaluation_labels_or_masks_used": False,
        "formal_crc_eligible": bool(data_contract.get("formal_crc_eligible", False)),
    }


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _manifest_image_ids(manifest: Mapping[str, Any], *, role: str) -> list[str]:
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{role} score manifest contains no records")
    image_ids: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"{role} score manifest record is not an object")
        image_id = record.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError(f"{role} score manifest record lacks image_id")
        image_ids.append(image_id)
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"{role} score manifest image IDs are not unique")
    return image_ids


def _protocol_detector_sha256(value: Any, *, role: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} raw-logit protocol is missing")
    return _require_sha256(
        value.get("detector_weight_sha256"),
        field=f"{role} protocol detector_weight_sha256",
    )


def _validate_paired_score_binding(
    zero_result: Mapping[str, Any],
    statistics_provenance: Mapping[str, Any],
    count_provenance: Mapping[str, Any],
    ids: Sequence[str],
    pair_audit: Mapping[str, Any],
    *,
    zero_result_sha256: str | None,
) -> dict[str, Any]:
    """Revalidate and bind distinct label-free and labelled score exports."""

    if pair_audit.get("schema_version") != PAIR_AUDIT_SCHEMA_VERSION:
        raise ValueError("Target-stage pair audit schema is missing or unsupported")
    if pair_audit.get("verified") is not True:
        raise ValueError("Target-stage pair audit is not verified")
    for field, expected in (
        ("labels_loaded_during_selection", False),
        ("labels_loaded_during_audit", True),
        ("labeled_audit_created_after_freeze", True),
    ):
        if pair_audit.get(field) is not expected:
            raise ValueError(f"Target-stage pair audit requires {field}={expected!r}")

    reported_spatial_fields = pair_audit.get("spatial_protocol_fields_verified")
    if not isinstance(reported_spatial_fields, list) or any(
        not isinstance(field, str) for field in reported_spatial_fields
    ):
        raise ValueError("Target-stage pair audit lacks spatial protocol evidence")
    missing_spatial_evidence = sorted(
        _PAIR_SPATIAL_FIELDS.difference(reported_spatial_fields)
    )
    if missing_spatial_evidence:
        raise ValueError(
            "Target-stage pair audit omits spatial protocol fields: "
            + ", ".join(missing_spatial_evidence)
        )

    freeze_path = Path(str(pair_audit.get("selection_freeze_record", ""))).expanduser()
    freeze = verify_selection_freeze(freeze_path)
    if file_sha256(freeze_path) != _require_sha256(
        pair_audit.get("selection_freeze_record_sha256"),
        field="selection_freeze_record_sha256",
    ):
        raise ValueError("Target-stage pair audit freeze-record SHA mismatch")
    zero_sha = _require_sha256(
        zero_result_sha256,
        field="current zero_result_sha256",
    )
    frozen_results = pair_audit.get("frozen_zero_results")
    if not isinstance(frozen_results, list) or not frozen_results:
        raise ValueError("Target-stage pair audit lacks frozen zero-result evidence")
    frozen_hashes = {
        str(item.get("sha256"))
        for item in frozen_results
        if isinstance(item, Mapping)
    }
    if zero_sha not in frozen_hashes:
        raise ValueError("Current zero result is not bound by the selection freeze")
    freeze_records = freeze.get("records")
    if not isinstance(freeze_records, list) or zero_sha not in {
        str(item.get("sha256"))
        for item in freeze_records
        if isinstance(item, Mapping) and item.get("role") == "zero_label_action"
    }:
        raise ValueError("Selection freeze does not bind the current zero result")

    unlabeled_root = Path(str(pair_audit.get("unlabeled_score_dir", ""))).expanduser().resolve()
    labeled_root = Path(str(pair_audit.get("labeled_score_dir", ""))).expanduser().resolve()
    if unlabeled_root == labeled_root:
        raise ValueError("Paired target score artifacts must use distinct directories")
    if Path(str(statistics_provenance.get("score_map_dir", ""))).expanduser().resolve() != unlabeled_root:
        raise ValueError("Pair audit unlabeled directory differs from zero-label statistics")
    if Path(str(count_provenance.get("score_dir", ""))).expanduser().resolve() != labeled_root:
        raise ValueError("Pair audit labeled directory differs from count curves")

    unlabeled, unlabeled_paths, unlabeled_integrity = verify_score_map_directory(
        unlabeled_root,
        require_integrity=True,
        require_masks=False,
    )
    labeled, labeled_paths, labeled_integrity = verify_score_map_directory(
        labeled_root,
        require_integrity=True,
        require_masks=True,
    )
    if unlabeled is None or labeled is None:
        raise ValueError("Paired target score manifests are missing")
    if unlabeled.get("labels_loaded") is not False:
        raise ValueError("Selection score artifact must record labels_loaded=false")
    if labeled.get("labels_loaded") is not True:
        raise ValueError("Labeled audit score artifact must record labels_loaded=true")
    for field in _PAIR_REQUIRED_MANIFEST_FIELDS:
        if field not in unlabeled or field not in labeled:
            raise ValueError(f"Paired target score manifests lack required field {field}")
    for field in _PAIR_SPATIAL_FIELDS:
        if unlabeled.get(field) != labeled.get(field):
            raise ValueError(f"Paired target score artifacts differ in {field}")

    unlabeled_manifest_sha = _require_sha256(
        unlabeled_integrity.get("manifest_sha256"),
        field="unlabeled manifest SHA",
    )
    labeled_manifest_sha = _require_sha256(
        labeled_integrity.get("manifest_sha256"),
        field="labeled manifest SHA",
    )
    unlabeled_records_sha = _require_sha256(
        unlabeled_integrity.get("records_sha256"),
        field="unlabeled records SHA",
    )
    labeled_records_sha = _require_sha256(
        labeled_integrity.get("records_sha256"),
        field="labeled records SHA",
    )
    pair_hash_bindings = {
        "unlabeled_manifest_sha256": unlabeled_manifest_sha,
        "unlabeled_records_sha256": unlabeled_records_sha,
        "labeled_manifest_sha256": labeled_manifest_sha,
        "labeled_records_sha256": labeled_records_sha,
    }
    for field, actual in pair_hash_bindings.items():
        if pair_audit.get(field) != actual:
            raise ValueError(f"Target-stage pair audit {field} mismatch")
    if statistics_provenance.get("score_manifest_sha256") != unlabeled_manifest_sha:
        raise ValueError("Zero-label statistics are not bound to the unlabeled export")
    if statistics_provenance.get("score_records_sha256") != unlabeled_records_sha:
        raise ValueError("Zero-label statistics records differ from unlabeled export")
    if count_provenance.get("manifest_sha256") != labeled_manifest_sha:
        raise ValueError("Count curves are not bound to the labeled audit export")
    if count_provenance.get("score_records_sha256") != labeled_records_sha:
        raise ValueError("Count-curve records differ from labeled audit export")

    unlabeled_ids = _manifest_image_ids(unlabeled, role="unlabeled")
    labeled_ids = _manifest_image_ids(labeled, role="labeled")
    if unlabeled_ids != labeled_ids or labeled_ids != list(ids):
        raise ValueError("Paired exports and count curves have different ordered IDs")
    ordered_ids_sha = _require_sha256(
        pair_audit.get("ordered_image_ids_sha256"),
        field="pair ordered_image_ids_sha256",
    )
    for role, value in (
        ("unlabeled statistics", statistics_provenance.get("score_ordered_image_ids_sha256")),
        ("labeled counts", count_provenance.get("score_ordered_image_ids_sha256")),
        ("unlabeled manifest", unlabeled_integrity.get("ordered_image_ids_sha256")),
        ("labeled manifest", labeled_integrity.get("ordered_image_ids_sha256")),
    ):
        if value != ordered_ids_sha:
            raise ValueError(f"{role} ordered image-ID SHA differs from pair audit")

    pair_detector_sha = _require_sha256(
        pair_audit.get("detector_weight_sha256"),
        field="pair detector_weight_sha256",
    )
    if unlabeled.get("weight_sha256") != pair_detector_sha or labeled.get(
        "weight_sha256"
    ) != pair_detector_sha:
        raise ValueError("Paired target score artifacts use different checkpoints")
    if _protocol_detector_sha256(
        zero_result.get("protocol"), role="zero-label selection"
    ) != pair_detector_sha:
        raise ValueError("Zero-label protocol checkpoint differs from pair audit")
    if _protocol_detector_sha256(
        count_provenance.get("protocol"), role="labeled count"
    ) != pair_detector_sha:
        raise ValueError("Count-curve protocol checkpoint differs from pair audit")

    num_records = pair_audit.get("num_records")
    if isinstance(num_records, bool) or not isinstance(num_records, int):
        raise ValueError("Target-stage pair audit num_records must be an integer")
    if num_records != len(ids) or statistics_provenance.get(
        "score_num_records"
    ) != num_records or count_provenance.get("score_num_records") != num_records:
        raise ValueError("Paired target artifacts disagree on record count")

    raw_logit_digest = hashlib.sha256()
    if len(unlabeled_paths) != len(labeled_paths) or len(unlabeled_paths) != len(ids):
        raise ValueError("Paired target score record counts differ")
    for image_id, first_path, second_path in zip(
        ids, unlabeled_paths, labeled_paths
    ):
        with np.load(first_path, allow_pickle=False) as first, np.load(
            second_path, allow_pickle=False
        ) as second:
            if "logit" not in first or "logit" not in second:
                raise ValueError("Paired target score artifacts must contain raw logits")
            first_logit = np.asarray(first["logit"])
            second_logit = np.asarray(second["logit"])
        if first_logit.dtype != np.float32 or second_logit.dtype != np.float32:
            raise ValueError("Paired target raw logits must be float32")
        if not np.isfinite(first_logit).all() or not np.array_equal(
            first_logit, second_logit
        ):
            raise ValueError(f"Paired target raw logits differ for {image_id}")
        raw_logit_digest.update(image_id.encode("utf-8"))
        raw_logit_digest.update(b"\0")
        raw_logit_digest.update(first_logit.tobytes(order="C"))
    raw_logit_sha = raw_logit_digest.hexdigest()
    if pair_audit.get("raw_logit_stream_sha256") != raw_logit_sha:
        raise ValueError("Target-stage pair audit raw-logit stream SHA mismatch")

    frozen_at_ns = int(freeze.get("frozen_at_unix_ns", 0))
    labeled_manifest_path = labeled_root / "manifest.json"
    if frozen_at_ns <= 0 or labeled_manifest_path.stat().st_mtime_ns <= frozen_at_ns:
        raise ValueError("Labeled audit export was not created after selection freeze")
    return {
        "binding_mode": "paired_unlabeled_selection_and_labeled_audit",
        "selection_manifest_sha256": unlabeled_manifest_sha,
        "selection_records_sha256": unlabeled_records_sha,
        "labeled_manifest_sha256": labeled_manifest_sha,
        "labeled_records_sha256": labeled_records_sha,
        "ordered_image_ids_sha256": ordered_ids_sha,
        "detector_weight_sha256": pair_detector_sha,
        "raw_logit_stream_sha256": raw_logit_sha,
        "selection_freeze_record_sha256": pair_audit[
            "selection_freeze_record_sha256"
        ],
    }


def validate_count_curve_binding(
    zero_result: Mapping[str, Any],
    count_provenance: Mapping[str, Any],
    count_image_ids: Sequence[str],
    *,
    pair_audit: Mapping[str, Any] | None = None,
    zero_result_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind a labelled audit to either the same or a verified paired export."""

    if count_provenance.get("source_type") != "exported_score_map_directory":
        raise ValueError(
            "Formal zero-label evaluation requires count curves rebuilt from the "
            "exported score-map directory"
        )
    artifact = zero_result.get("statistics_artifact")
    statistics_provenance = (
        artifact.get("provenance") if isinstance(artifact, Mapping) else None
    )
    if not isinstance(statistics_provenance, Mapping):
        raise ValueError("Zero-label statistics provenance is missing")
    if count_provenance.get("protocol_fingerprint") != zero_result.get(
        "protocol_fingerprint"
    ):
        raise ValueError(
            "Count curves and zero-label statistics use different score protocols"
        )
    ids = [str(value) for value in count_image_ids]
    if (
        not ids
        or any(not value.strip() for value in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("Count-curve image IDs must be unique non-empty strings")
    if count_provenance.get("score_num_records") != len(ids):
        raise ValueError("Count-curve provenance record count differs from its image IDs")

    if pair_audit is None:
        comparisons = {
            "manifest_sha256": "score_manifest_sha256",
            "score_records_sha256": "score_records_sha256",
            "score_ordered_image_ids_sha256": "score_ordered_image_ids_sha256",
            "score_num_records": "score_num_records",
        }
        for count_field, selection_field in comparisons.items():
            if count_provenance.get(count_field) != statistics_provenance.get(
                selection_field
            ):
                raise ValueError(
                    "Count curves and zero-label statistics differ in " + count_field
                )
        binding: dict[str, Any] = {
            "binding_mode": "identical_score_artifact",
            "manifest_sha256": count_provenance.get("manifest_sha256"),
            "score_records_sha256": count_provenance.get("score_records_sha256"),
            "score_ordered_image_ids_sha256": count_provenance.get(
                "score_ordered_image_ids_sha256"
            ),
        }
    else:
        binding = _validate_paired_score_binding(
            zero_result,
            statistics_provenance,
            count_provenance,
            ids,
            pair_audit,
            zero_result_sha256=zero_result_sha256,
        )
    if zero_result.get("adaptation_protocol") == "static_cross_fit":
        mapping = zero_result.get("threshold_indices_by_image")
        if not isinstance(mapping, Mapping) or set(map(str, mapping)) != set(ids):
            raise ValueError(
                "Static zero-label actions do not exactly cover the bound count archive"
            )
    return {
        "verified": True,
        **binding,
        "num_images": len(ids),
        "protocol_fingerprint": count_provenance.get("protocol_fingerprint"),
    }


def _mean(values: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    array = np.asarray(values)
    if mask is not None:
        array = array[np.asarray(mask, dtype=bool)]
    return float(np.mean(array)) if array.size else None


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0.0 else None


def _budget(zero_result: Mapping[str, Any], name: str) -> float:
    direct = zero_result.get(f"{name}_budget")
    if direct is None and isinstance(zero_result.get("budgets"), Mapping):
        direct = zero_result["budgets"].get(name)
    try:
        value = float(direct)
    except (TypeError, ValueError) as error:
        raise ValueError(f"zero-label result lacks a valid {name} budget") from error
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} budget must be finite and positive")
    return value


def _actions_for_ids(
    zero_result: Mapping[str, Any],
    image_ids: Sequence[str],
    *,
    allow_unmapped_as_reject: bool,
) -> tuple[list[int | None], list[str], list[str]]:
    ids = [str(value) for value in image_ids]
    mapping = zero_result.get("threshold_indices_by_image")
    if mapping is not None:
        if not isinstance(mapping, Mapping):
            raise ValueError("threshold_indices_by_image must be a JSON object")
        normalised = {str(key): value for key, value in mapping.items()}
        missing = [image_id for image_id in ids if image_id not in normalised]
        if missing and not allow_unmapped_as_reject:
            raise ValueError(
                "zero-label result does not cover every evaluated image ID: "
                + ", ".join(missing[:10])
            )
        actions = [normalised.get(image_id) for image_id in ids]
        extras = sorted(set(normalised).difference(ids))
        return actions, missing, extras
    scalar = zero_result.get("threshold_index")
    if scalar is None:
        if not bool(zero_result.get("reject", False)):
            raise ValueError("zero-label result has neither scalar nor per-image actions")
        return [None] * len(ids), [], []
    return [scalar] * len(ids), [], []


def _count_curves_for_mapped_actions(
    zero_result: Mapping[str, Any],
    count_curves: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Restrict a labelled audit to causal E/query IDs with frozen actions."""

    if zero_result.get("adaptation_protocol") != "external_causal_statistics":
        raise ValueError(
            "mapped-actions-only evaluation is valid only for external causal "
            "deployment statistics"
        )
    mapping = zero_result.get("threshold_indices_by_image")
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("mapped-actions-only evaluation requires per-image actions")
    if "image_ids" not in count_curves:
        raise ValueError("count curves lack image_ids")
    image_ids = np.asarray(count_curves["image_ids"]).astype(str)
    if image_ids.ndim != 1 or image_ids.size == 0:
        raise ValueError("count-curve image_ids must be a non-empty vector")
    if len(set(image_ids.tolist())) != image_ids.size:
        raise ValueError("count-curve image_ids must be unique")
    mapped_ids = {str(value) for value in mapping}
    extras = sorted(mapped_ids.difference(image_ids.tolist()))
    if extras:
        raise ValueError(
            "zero-label actions reference IDs absent from count curves: "
            + ", ".join(extras[:10])
        )
    keep = np.asarray([value in mapped_ids for value in image_ids], dtype=bool)
    if not np.any(keep):
        raise ValueError("No mapped causal evaluation IDs occur in count curves")
    excluded = image_ids[~keep].tolist()
    per_image_fields = {
        "image_ids",
        "false_positive_pixels",
        "false_positive_components",
        "total_pixels",
        "tp_object_counts",
        "gt_object_counts",
    }
    subset: dict[str, np.ndarray] = {}
    for name, raw in count_curves.items():
        values = np.asarray(raw)
        if name in per_image_fields:
            if values.ndim < 1 or values.shape[0] != image_ids.size:
                raise ValueError(
                    f"count-curve field {name!r} does not align with image_ids"
                )
            subset[name] = values[keep]
        else:
            subset[name] = values
    return subset, excluded


def evaluate_zero_label_actions(
    zero_result: Mapping[str, Any],
    count_curves: Mapping[str, np.ndarray],
    *,
    allow_unmapped_as_reject: bool = False,
    allow_unverified_diagnostic: bool = True,
    mapped_actions_only: bool = False,
) -> dict[str, Any]:
    """Evaluate frozen actions; legacy direct API calls are diagnostic only.

    The command-line entry point explicitly disables this compatibility mode
    unless ``--allow-unverified-diagnostic`` is supplied, so persisted formal
    evaluations reject an incomplete selection contract.
    """

    try:
        selection_contract_audit = validate_zero_result_selection_contract(zero_result)
    except ValueError as error:
        if not allow_unverified_diagnostic:
            raise
        selection_contract_audit = {
            "verified": False,
            "error": str(error),
            "adaptation_protocol": zero_result.get("adaptation_protocol"),
        }
    if mapped_actions_only and allow_unmapped_as_reject:
        raise ValueError(
            "mapped_actions_only and allow_unmapped_as_reject are mutually exclusive"
        )
    excluded_unmapped_ids: list[str] = []
    effective_count_curves = dict(count_curves)
    if mapped_actions_only:
        effective_count_curves, excluded_unmapped_ids = (
            _count_curves_for_mapped_actions(zero_result, count_curves)
        )
    pixel_budget = _budget(zero_result, "pixel")
    component_budget = _budget(zero_result, "component")
    losses = build_calibration_losses(
        **effective_count_curves,
        pixel_budget=pixel_budget,
        component_budget=component_budget,
        loss_mode=LOSS_MODE_BUDGET_VIOLATION,
    )
    recorded_grid = np.asarray(zero_result.get("thresholds"), dtype=np.float64)
    if recorded_grid.shape != losses.thresholds.shape or not np.allclose(
        recorded_grid, losses.thresholds, rtol=0.0, atol=1e-8
    ):
        raise ValueError("zero-label result and count curves use different grids")
    actions, missing_ids, extra_ids = _actions_for_ids(
        zero_result,
        losses.image_ids.astype(str).tolist(),
        allow_unmapped_as_reject=allow_unmapped_as_reject,
    )
    if (
        selection_contract_audit.get("verified") is True
        and zero_result.get("adaptation_protocol") == "static_cross_fit"
        and extra_ids
    ):
        raise ValueError(
            "Static full-coverage actions reference image IDs absent from the "
            "count-curve archive: "
            + ", ".join(extra_ids[:10])
        )
    normalised: list[int | None] = []
    for value in actions:
        if value is None:
            normalised.append(None)
            continue
        if isinstance(value, bool):
            raise ValueError("threshold indices must be integers or null")
        try:
            index = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError("threshold indices must be integers or null") from error
        if index != value or not 0 <= index < losses.num_thresholds:
            raise ValueError("A zero-label threshold index lies outside the grid")
        normalised.append(index)
    active = np.asarray([value is not None for value in normalised], dtype=bool)
    indices = np.asarray(
        [0 if value is None else value for value in normalised], dtype=np.int64
    )
    rows = np.arange(losses.num_images, dtype=np.int64)

    def gather(curves: np.ndarray) -> np.ndarray:
        values = np.zeros(losses.num_images, dtype=np.float64)
        values[active] = np.asarray(curves)[rows[active], indices[active]]
        return values

    pixel_risk = gather(losses.pixel_risk)
    component_raw = gather(losses.component_risk_raw)
    component_upper = gather(losses.component_risk_envelope)
    selected_fp_pixels = gather(losses.false_positive_pixels)
    selected_fp_components = gather(losses.false_positive_components)
    pixel_ok = pixel_risk <= pixel_budget
    component_raw_ok = component_raw <= component_budget
    component_upper_ok = component_upper <= component_budget
    pixel_excess = np.maximum(pixel_risk / pixel_budget - 1.0, 0.0)
    component_excess = np.maximum(component_upper / component_budget - 1.0, 0.0)
    joint_excess = np.maximum(pixel_excess, component_excess)
    total_exposure = float(np.sum(losses.total_pixels))
    active_exposure = float(np.sum(losses.total_pixels[active]))
    metrics: dict[str, Any] = {
        "num_images": losses.num_images,
        "active_image_count": int(np.sum(active)),
        "rejected_image_count": int(np.sum(~active)),
        "coverage_rate": float(np.mean(active)),
        "reject_rate": float(np.mean(~active)),
        "mean_pixel_risk": _mean(pixel_risk),
        "mean_pixel_risk_active_only": _mean(pixel_risk, active),
        "aggregate_pixel_risk": _ratio(float(np.sum(selected_fp_pixels)), total_exposure),
        "aggregate_pixel_risk_active_only": _ratio(
            float(np.sum(selected_fp_pixels[active])), active_exposure
        ),
        "mean_component_risk_raw": _mean(component_raw),
        "mean_component_risk_raw_active_only": _mean(component_raw, active),
        "mean_component_risk_upper": _mean(component_upper),
        "mean_component_risk_upper_active_only": _mean(component_upper, active),
        "aggregate_component_risk_raw": _ratio(
            float(np.sum(selected_fp_components)), total_exposure / 1_000_000.0
        ),
        "aggregate_component_risk_raw_active_only": _ratio(
            float(np.sum(selected_fp_components[active])),
            active_exposure / 1_000_000.0,
        ),
        "pixel_budget_satisfaction_rate": _mean(pixel_ok),
        "pixel_budget_satisfaction_rate_active_only": _mean(pixel_ok, active),
        "component_budget_satisfaction_rate_raw": _mean(component_raw_ok),
        "component_budget_satisfaction_rate_raw_active_only": _mean(
            component_raw_ok, active
        ),
        "component_budget_satisfaction_rate_upper": _mean(component_upper_ok),
        "component_budget_satisfaction_rate_upper_active_only": _mean(
            component_upper_ok, active
        ),
        "joint_budget_satisfaction_rate_raw": _mean(pixel_ok & component_raw_ok),
        "joint_budget_satisfaction_rate_upper": _mean(pixel_ok & component_upper_ok),
        "joint_budget_satisfaction_rate_upper_active_only": _mean(
            pixel_ok & component_upper_ok, active
        ),
        "mean_relative_excess": _mean(joint_excess),
        "mean_relative_excess_active_only": _mean(joint_excess, active),
        "max_relative_excess": float(np.max(joint_excess)),
        "max_relative_excess_active_only": (
            float(np.max(joint_excess[active])) if np.any(active) else None
        ),
    }
    if losses.tp_object_counts is not None and losses.gt_object_counts is not None:
        selected_tp = gather(losses.tp_object_counts)
        gt = np.asarray(losses.gt_object_counts, dtype=np.float64)
        metrics.update(
            {
                "true_positive_objects": int(np.sum(selected_tp)),
                "ground_truth_objects": int(np.sum(gt)),
                "true_positive_objects_active_only": int(np.sum(selected_tp[active])),
                "ground_truth_objects_active_only": int(np.sum(gt[active])),
                "pd_object_aggregate": _ratio(
                    float(np.sum(selected_tp)), float(np.sum(gt))
                ),
                "pd_object_aggregate_active_only": _ratio(
                    float(np.sum(selected_tp[active])), float(np.sum(gt[active]))
                ),
            }
        )
    else:
        metrics.update(
            {
                "true_positive_objects": None,
                "ground_truth_objects": None,
                "true_positive_objects_active_only": None,
                "ground_truth_objects_active_only": None,
                "pd_object_aggregate": None,
                "pd_object_aggregate_active_only": None,
            }
        )
    selected_actions = [
        {
            "image_id": str(image_id),
            "threshold_index": value,
            "threshold": (
                float(losses.thresholds[value]) if value is not None else None
            ),
            "action": "threshold" if value is not None else "no_detection_reject",
        }
        for image_id, value in zip(losses.image_ids.astype(str), normalised)
    ]
    return {
        "schema_version": ZERO_LABEL_EVALUATION_SCHEMA_VERSION,
        "mode": "independent_labelled_audit_of_frozen_zero_label_actions",
        "guarantee": "none; zero-label risk prediction is empirical",
        "test_labels_used_for_selection": (
            False if selection_contract_audit["verified"] else None
        ),
        "selection_contract_audit": selection_contract_audit,
        "adaptation_protocol": zero_result.get("adaptation_protocol"),
        "evaluation_scope": (
            "mapped_causal_evaluation_ids_only"
            if mapped_actions_only
            else "complete_count_curve_archive"
        ),
        "excluded_unmapped_image_ids": excluded_unmapped_ids,
        "budgets": {"pixel": pixel_budget, "component": component_budget},
        "unmapped_image_ids_treated_as_reject": missing_ids,
        "extra_action_image_ids_not_evaluated": extra_ids,
        "metrics": metrics,
        "selected_actions": selected_actions,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-result", required=True)
    parser.add_argument("--count-curves", required=True)
    parser.add_argument(
        "--target-stage-pair-audit",
        help=(
            "Verified audit JSON binding a label-free selection export to a "
            "distinct post-freeze labelled export"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-unmapped-as-reject",
        action="store_true",
        help="Explicitly treat images without an action as no-detection rejects",
    )
    parser.add_argument(
        "--allow-unverified-diagnostic",
        action="store_true",
        help=(
            "Evaluate a legacy/unverified action artifact diagnostically; the output "
            "will not claim that test labels were unused for selection"
        ),
    )
    parser.add_argument(
        "--mapped-actions-only",
        action="store_true",
        help=(
            "For causal A->E artifacts, evaluate only E/query image IDs that "
            "received frozen actions; standard full-test metrics remain separate"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    zero_path = Path(args.zero_result)
    count_path = Path(args.count_curves)
    zero_result = json.loads(zero_path.read_text(encoding="utf-8"))
    if not isinstance(zero_result, Mapping):
        raise ValueError("zero-result must decode to a JSON object")
    counts = load_count_curve_archive(count_path, require_integrity=True)
    count_provenance = load_count_curve_provenance(
        count_path, require_integrity=True
    )
    zero_result_sha256 = hashlib.sha256(zero_path.read_bytes()).hexdigest()
    pair_path: Path | None = None
    pair_audit: Mapping[str, Any] | None = None
    pair_audit_sha256: str | None = None
    if args.target_stage_pair_audit:
        pair_path = Path(args.target_stage_pair_audit)
        pair_payload = json.loads(pair_path.read_text(encoding="utf-8"))
        if not isinstance(pair_payload, Mapping):
            raise ValueError("target-stage pair audit must decode to a JSON object")
        pair_audit = pair_payload
        pair_audit_sha256 = hashlib.sha256(pair_path.read_bytes()).hexdigest()
    try:
        count_binding_audit = validate_count_curve_binding(
            zero_result,
            count_provenance,
            np.asarray(counts["image_ids"]).astype(str).tolist(),
            pair_audit=pair_audit,
            zero_result_sha256=(
                zero_result_sha256 if pair_audit is not None else None
            ),
        )
    except ValueError as error:
        if not args.allow_unverified_diagnostic:
            raise
        count_binding_audit = {
            "verified": False,
            "error": str(error),
        }
    result = evaluate_zero_label_actions(
        zero_result,
        counts,
        allow_unmapped_as_reject=args.allow_unmapped_as_reject,
        allow_unverified_diagnostic=args.allow_unverified_diagnostic,
        mapped_actions_only=args.mapped_actions_only,
    )
    result["provenance"] = {
        "zero_result": str(zero_path.resolve()),
        "zero_result_sha256": zero_result_sha256,
        "count_curves": str(count_path.resolve()),
        "count_curves_sha256": hashlib.sha256(count_path.read_bytes()).hexdigest(),
        "target_stage_pair_audit": (
            str(pair_path.resolve()) if pair_path is not None else None
        ),
        "target_stage_pair_audit_sha256": pair_audit_sha256,
        "count_curve_binding_audit": count_binding_audit,
    }
    atomic_write_json(args.output, result)
    print(Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ZERO_LABEL_EVALUATION_SCHEMA_VERSION",
    "build_argument_parser",
    "evaluate_zero_label_actions",
    "validate_count_curve_binding",
    "validate_zero_result_selection_contract",
    "_count_curves_for_mapped_actions",
    "main",
]
