from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from risk_curve.build_deployment_statistics import build_static_cross_fit_statistics
from risk_curve.evaluate_zero_label import (
    evaluate_zero_label_actions,
    validate_count_curve_binding,
    validate_zero_result_selection_contract,
)
from risk_curve.select_zero_label_threshold import (
    SELECTION_DATA_CONTRACT_SCHEMA_VERSION,
    ZERO_RESULT_SCHEMA_VERSION,
    _statistics_from_archive,
    _validate_static_checkpoint_compatibility,
)
from risk_curve.train_curve_predictor import (
    TRAINING_EPISODE_CONTRACT_SCHEMA_VERSION,
)


def _score_directory(root: Path, count: int = 6) -> Path:
    root.mkdir()
    files: list[str] = []
    for index in range(count):
        filename = f"{index:03d}.npz"
        probability = np.full((3, 3), index / count, dtype=np.float32)
        np.savez_compressed(
            root / filename,
            prob=probability,
            gray=probability,
            image_id=np.asarray(f"image-{index}"),
        )
        files.append(filename)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "files": files,
                "score_type": "sigmoid_probability",
                "warm_flag": True,
                "spatial_mode": "native",
                "pad_multiple": 16,
                "target_dataset": "static-target",
                "source_datasets": ["source-a", "source-b"],
                "weight_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    return root


def _static_arrays(tmp_path: Path) -> dict[str, object]:
    return build_static_cross_fit_statistics(
        _score_directory(tmp_path / "scores"),
        np.asarray([0.0, 0.5], dtype=np.float32),
        folds=3,
        seed=7,
        adaptation_window=3,
    )


def test_static_archive_loader_rejects_tampered_no_mask_provenance(
    tmp_path: Path,
) -> None:
    arrays = _static_arrays(tmp_path)
    valid_path = tmp_path / "valid.npz"
    np.savez_compressed(valid_path, **arrays)
    evidence = _statistics_from_archive(valid_path)[-1]
    assert evidence["identity_contract"]["verified"] is True

    provenance = json.loads(str(np.asarray(arrays["provenance_json"]).item()))
    provenance["masks_read"] = True
    tampered = dict(arrays)
    tampered["provenance_json"] = np.asarray(json.dumps(provenance, sort_keys=True))
    tampered_path = tmp_path / "tampered-provenance.npz"
    np.savez_compressed(tampered_path, **tampered)
    with pytest.raises(ValueError, match="masks_read=False"):
        _statistics_from_archive(tampered_path)


def test_static_archive_loader_rejects_duplicate_evaluation_identity(
    tmp_path: Path,
) -> None:
    arrays = _static_arrays(tmp_path)
    evaluation = [json.loads(str(value)) for value in arrays["evaluation_ids"]]
    evaluation[1][0] = evaluation[0][0]
    tampered = dict(arrays)
    tampered["evaluation_ids"] = np.asarray(
        [json.dumps(row) for row in evaluation], dtype=str
    )
    path = tmp_path / "duplicate-evaluation.npz"
    np.savez_compressed(path, **tampered)
    with pytest.raises(ValueError, match="reuses IDs|globally unique"):
        _statistics_from_archive(path)


def test_static_archive_loader_binds_fixed_adaptation_sampling_contract(
    tmp_path: Path,
) -> None:
    arrays = _static_arrays(tmp_path)
    provenance = json.loads(str(np.asarray(arrays["provenance_json"]).item()))
    provenance["adaptation_window"] = 2
    tampered = dict(arrays)
    tampered["provenance_json"] = np.asarray(json.dumps(provenance, sort_keys=True))
    path = tmp_path / "wrong-adaptation-window.npz"
    np.savez_compressed(path, **tampered)
    with pytest.raises(ValueError, match="adaptation rows"):
        _statistics_from_archive(path)


def _static_checkpoint_contract(*, evaluation_window: int = 1) -> dict[str, object]:
    grid_hash = "a" * 64
    source_reference = {
        "sha256": "b" * 64,
        "domain_names": ["source-a", "source-b"],
        "statistics_names_sha256": "c" * 64,
    }
    protocol_fields = {
        "adaptation_window": 32,
        "evaluation_window": evaluation_window,
        "stride": 32 + evaluation_window,
        "threshold_grid_sha256": grid_hash,
        "pseudo_targets": ["source-a", "source-b"],
        "source_reference_sha256": source_reference["sha256"],
        "source_reference_domain_names": source_reference["domain_names"],
        "source_reference_statistics_names_sha256": source_reference[
            "statistics_names_sha256"
        ],
    }
    return {
        "checkpoint": {
            "threshold_grid_sha256": grid_hash,
            "episode_contract": {
                "schema_version": TRAINING_EPISODE_CONTRACT_SCHEMA_VERSION,
                "verified": True,
                "formal_protocol_eligible": True,
                "adaptation_window": 32,
                "evaluation_window": evaluation_window,
                "stride": 32 + evaluation_window,
                "one_to_one_future_target": evaluation_window == 1,
                "protocol_fields": protocol_fields,
            },
        },
        "provenance": {
            "adaptation_window": 32,
            "threshold_grid_sha256": grid_hash,
            "source_reference_sha256": source_reference["sha256"],
            "source_reference_domain_names": source_reference["domain_names"],
            "source_reference_statistics_names_sha256": source_reference[
                "statistics_names_sha256"
            ],
            "score_integrity_verified": True,
            "score_num_records": 100,
            "num_images": 100,
            "score_manifest_sha256": "d" * 64,
            "score_records_sha256": "e" * 64,
            "score_ordered_image_ids_sha256": "f" * 64,
        },
        "protocol": {
            "threshold_grid_sha256": grid_hash,
            "target_dataset": "target",
        },
        "grid_hash": grid_hash,
    }


def test_static_checkpoint_contract_rejects_non_image_level_training() -> None:
    contract = _static_checkpoint_contract(evaluation_window=2)
    with pytest.raises(ValueError, match="E=1"):
        _validate_static_checkpoint_compatibility(
            contract["checkpoint"],
            provenance=contract["provenance"],
            protocol=contract["protocol"],
            expected_threshold_grid_sha256=contract["grid_hash"],
        )


def test_static_checkpoint_contract_accepts_formal_a32_e1_stride33() -> None:
    contract = _static_checkpoint_contract()
    audit = _validate_static_checkpoint_compatibility(
        contract["checkpoint"],
        provenance=contract["provenance"],
        protocol=contract["protocol"],
        expected_threshold_grid_sha256=contract["grid_hash"],
    )
    assert audit["verified"] is True
    assert audit["adaptation_window"] == 32
    assert audit["evaluation_window"] == 1
    assert audit["stride"] == 33


def _valid_static_zero_result() -> dict[str, object]:
    identity = {"verified": True, "scope": "static_cross_fit_full_coverage_empirical"}
    provenance = {
        "masks_read": False,
        "full_test_coverage": True,
        "formal_crc_eligible": False,
        "score_integrity_verified": True,
    }
    return {
        "schema_version": ZERO_RESULT_SCHEMA_VERSION,
        "mode": "zero_label_empirical_adaptation",
        "adaptation_protocol": "static_cross_fit",
        "masks_read": False,
        "selection_data_contract": {
            "schema_version": SELECTION_DATA_CONTRACT_SCHEMA_VERSION,
            "masks_read": False,
            "evaluation_labels_or_masks_used": False,
            "statistics_computed_from": "static_cross_fit_complement_folds",
            "threshold_mapping_rule": (
                "one_complement_fold_prediction_to_held_out_fold_ids"
            ),
            "formal_crc_eligible": False,
            "deployment_identity_contract_verified": True,
            "static_checkpoint_compatibility_verified": True,
        },
        "statistics_artifact": {
            "identity_contract": identity,
            "provenance": provenance,
        },
        "static_checkpoint_compatibility_audit": {"verified": True},
        "num_windows": 1,
        "evaluation_ids": [["a", "b"]],
        "threshold_indices": [1],
        "threshold_indices_by_image": {"a": 1, "b": 1},
        "pixel_budget": 0.1,
        "component_budget": 100.0,
        "thresholds": [0.0, 0.5],
    }


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value.pop("schema_version"), "schema"),
        (
            lambda value: value["selection_data_contract"].update(
                {"evaluation_labels_or_masks_used": True}
            ),
            "evaluation_labels_or_masks_used=false",
        ),
        (
            lambda value: value["selection_data_contract"].update(
                {"statistics_computed_from": "causal_adaptation_blocks_A"}
            ),
            "inconsistent with adaptation_protocol",
        ),
        (
            lambda value: value.update(
                {"threshold_indices_by_image": {"a": 1}}
            ),
            "does not exactly cover evaluation IDs",
        ),
    ],
)
def test_zero_result_contract_rejects_tampering(mutator, message: str) -> None:
    value = _valid_static_zero_result()
    mutator(value)
    with pytest.raises(ValueError, match=message):
        validate_zero_result_selection_contract(value)


def test_legacy_diagnostic_evaluation_never_claims_no_label_selection() -> None:
    zero = {
        "mode": "zero_label_empirical_adaptation",
        "pixel_budget": 0.1,
        "component_budget": 100.0,
        "thresholds": [0.0, 0.5],
        "threshold_index": 1,
    }
    curves = {
        "image_ids": np.asarray(["a"]),
        "thresholds": np.asarray([0.0, 0.5], dtype=np.float32),
        "false_positive_pixels": np.asarray([[1.0, 0.0]]),
        "false_positive_components": np.asarray([[1.0, 0.0]]),
        "total_pixels": np.asarray([100.0]),
    }
    result = evaluate_zero_label_actions(
        zero,
        curves,
        allow_unverified_diagnostic=True,
    )
    assert result["selection_contract_audit"]["verified"] is False
    assert result["test_labels_used_for_selection"] is None


def test_verified_static_evaluation_rejects_a_subset_count_archive() -> None:
    zero = _valid_static_zero_result()
    curves = {
        "image_ids": np.asarray(["a"]),
        "thresholds": np.asarray([0.0, 0.5], dtype=np.float32),
        "false_positive_pixels": np.asarray([[1.0, 0.0]]),
        "false_positive_components": np.asarray([[1.0, 0.0]]),
        "total_pixels": np.asarray([100.0]),
    }
    with pytest.raises(ValueError, match="absent from the count-curve archive"):
        evaluate_zero_label_actions(
            zero,
            curves,
            allow_unverified_diagnostic=False,
        )


def test_count_curve_binding_rejects_a_different_score_manifest() -> None:
    zero = _valid_static_zero_result()
    zero["protocol_fingerprint"] = "1" * 64
    zero["statistics_artifact"]["provenance"].update(
        {
            "score_manifest_sha256": "2" * 64,
            "score_records_sha256": "3" * 64,
            "score_ordered_image_ids_sha256": "4" * 64,
            "score_num_records": 2,
        }
    )
    count_provenance = {
        "source_type": "exported_score_map_directory",
        "manifest_sha256": "9" * 64,
        "score_records_sha256": "3" * 64,
        "score_ordered_image_ids_sha256": "4" * 64,
        "score_num_records": 2,
        "protocol_fingerprint": "1" * 64,
    }
    with pytest.raises(ValueError, match="manifest_sha256"):
        validate_count_curve_binding(zero, count_provenance, ["a", "b"])


def test_causal_audit_can_restrict_metrics_to_frozen_e_query_ids() -> None:
    zero = {
        "schema_version": ZERO_RESULT_SCHEMA_VERSION,
        "mode": "zero_label_empirical_adaptation",
        "adaptation_protocol": "external_causal_statistics",
        "masks_read": False,
        "selection_data_contract": {
            "schema_version": SELECTION_DATA_CONTRACT_SCHEMA_VERSION,
            "masks_read": False,
            "evaluation_labels_or_masks_used": False,
            "statistics_computed_from": "causal_adaptation_blocks_A",
            "threshold_mapping_rule": (
                "one_A_block_prediction_to_its_future_E_identity"
            ),
            "formal_crc_eligible": False,
            "deployment_identity_contract_verified": True,
        },
        "statistics_artifact": {
            "identity_contract": {"verified": True, "scope": "causal_blocks"},
            "provenance": {"masks_read": False},
        },
        "num_windows": 1,
        "evaluation_ids": [["b"]],
        "threshold_indices": [1],
        "threshold_indices_by_image": {"b": 1},
        "pixel_budget": 0.1,
        "component_budget": 100.0,
        "thresholds": [0.0, 0.5],
    }
    curves = {
        "image_ids": np.asarray(["a", "b", "c"]),
        "thresholds": np.asarray([0.0, 0.5], dtype=np.float32),
        "false_positive_pixels": np.asarray(
            [[10.0, 0.0], [10.0, 0.0], [10.0, 0.0]]
        ),
        "false_positive_components": np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        ),
        "total_pixels": np.asarray([100.0, 100.0, 100.0]),
        "tp_object_counts": np.asarray(
            [[1.0, 0.0], [1.0, 1.0], [1.0, 0.0]]
        ),
        "gt_object_counts": np.asarray([1.0, 1.0, 1.0]),
    }

    result = evaluate_zero_label_actions(
        zero,
        curves,
        allow_unverified_diagnostic=False,
        mapped_actions_only=True,
    )

    assert result["selection_contract_audit"]["verified"] is True
    assert result["evaluation_scope"] == "mapped_causal_evaluation_ids_only"
    assert result["excluded_unmapped_image_ids"] == ["a", "c"]
    assert result["metrics"]["num_images"] == 1
    assert result["metrics"]["pd_object_aggregate"] == pytest.approx(1.0)
