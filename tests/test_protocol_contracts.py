import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from certification.build_calibration_losses import (
    COUNT_ARCHIVE_INTEGRITY_SCHEMA,
    PROTOCOL_SCHEMA_VERSION,
    build_calibration_losses,
    count_archive_payload_sha256,
    load_count_curve_archive,
    protocol_fingerprint,
    validate_formal_protocol,
)
from certification.calibrate_target_offset import (
    DEPLOYMENT_STATISTICS_SCHEMA_VERSION,
    SELECTION_DATA_CONTRACT_SCHEMA_VERSION,
    STATISTICS_SCHEMA_VERSION,
    ZERO_RESULT_SCHEMA_VERSION,
    audit_protocol_bundle,
    audit_zero_artifact_contract,
    main as calibrate_main,
)
from certification.evaluate_certified_mode import evaluate_selected_operating_point
from evaluation.artifact_integrity import (
    SCORE_MANIFEST_SCHEMA_VERSION,
    SCORE_MASK_ALIGNMENT_SCHEMA,
    SCORE_RECORD_INTEGRITY_SCHEMA,
    file_sha256,
    ordered_ids_sha256,
    score_records_sha256,
)
from data_ext.mask_alignment import (
    MASK_ALIGNMENT_NOT_LOADED_POLICY,
    MASK_ALIGNMENT_POLICY,
)
from risk_curve.threshold_grid import threshold_grid_sha256
from risk_curve.deployment_contract import validate_checkpoint_deployment_contract
from risk_curve.train_curve_predictor import (
    TRAINING_EPISODE_CONTRACT_SCHEMA_VERSION,
)


def _formal_protocol() -> dict:
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "detector_weight_sha256": "a" * 64,
        "score_type": "sigmoid_probability",
        "warm_flag": True,
        "spatial_mode": "native",
        "pad_multiple": 16,
        "base_hw": None,
        "target_dataset": "target-domain",
        "source_datasets": ["source-a", "source-b"],
        "threshold_grid_sha256": "b" * 64,
        "matching_rule": "overlap",
        "centroid_distance": 3.0,
        "connectivity": 2,
        "min_component_area": 1,
        "component_monotone_transform": "per_image_suffix_max",
    }


def _write_hashed_score_manifest(
    score_dir: Path,
    protocol: dict,
    *,
    image_ids: list[str],
    labels_loaded: bool,
) -> tuple[Path, dict]:
    records = []
    for index, image_id in enumerate(image_ids):
        filename = f"{index:03d}.npz"
        arrays = {
            "prob": np.zeros((2, 2), dtype=np.float32),
            "gray": np.zeros((2, 2), dtype=np.float32),
            "image_id": np.asarray(image_id),
            "labels_loaded": np.asarray(labels_loaded),
            "original_hw": np.asarray([2, 2], dtype=np.int32),
            "mask_alignment_applied": np.asarray(False),
            "mask_original_hw": np.asarray(
                [2, 2] if labels_loaded else [0, 0], dtype=np.int32
            ),
            "mask_aspect_relative_error": np.asarray(
                0.0 if labels_loaded else -1.0
            ),
            "mask_alignment_policy": np.asarray(
                MASK_ALIGNMENT_POLICY
                if labels_loaded
                else MASK_ALIGNMENT_NOT_LOADED_POLICY
            ),
        }
        if labels_loaded:
            arrays["mask"] = np.zeros((2, 2), dtype=np.uint8)
        np.savez_compressed(score_dir / filename, **arrays)
        records.append(
            {
                "image_id": image_id,
                "file": filename,
                "shape": [2, 2],
                "sha256": file_sha256(score_dir / filename),
                "mask_alignment_applied": False,
                "mask_original_hw": [2, 2] if labels_loaded else [0, 0],
                "mask_aspect_relative_error": 0.0 if labels_loaded else -1.0,
            }
        )
    manifest = {
        "schema_version": SCORE_MANIFEST_SCHEMA_VERSION,
        "record_integrity_schema": SCORE_RECORD_INTEGRITY_SCHEMA,
        "score_type": protocol["score_type"],
        "warm_flag": protocol["warm_flag"],
        "labels_loaded": labels_loaded,
        "spatial_mode": protocol["spatial_mode"],
        "pad_multiple": protocol["pad_multiple"],
        "target_dataset": protocol["target_dataset"],
        "source_datasets": protocol["source_datasets"],
        "weight_sha256": protocol["detector_weight_sha256"],
        "num_images": len(records),
        "records": records,
        "records_sha256": score_records_sha256(records),
        "ordered_image_ids_sha256": ordered_ids_sha256(image_ids),
        "mask_alignment_schema": SCORE_MASK_ALIGNMENT_SCHEMA,
        "mask_alignment_policy": (
            MASK_ALIGNMENT_POLICY
            if labels_loaded
            else MASK_ALIGNMENT_NOT_LOADED_POLICY
        ),
        "mask_alignment_count": 0,
        "mask_aligned_sample_ids": [],
    }
    manifest_path = score_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest


def _write_hashed_count_archive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    payload = dict(arrays)
    payload["count_archive_integrity_schema"] = np.asarray(
        COUNT_ARCHIVE_INTEGRITY_SCHEMA
    )
    payload["count_archive_payload_sha256"] = np.asarray(
        count_archive_payload_sha256(payload)
    )
    np.savez_compressed(path, **payload)


def _protocol_records(protocol: dict, tmp_path: Path) -> tuple[dict, dict, dict]:
    protocol = copy.deepcopy(protocol)
    grid = np.asarray([0.0, 0.5, 0.9], dtype=np.float32)
    protocol["threshold_grid_sha256"] = threshold_grid_sha256(grid)
    fingerprint = protocol_fingerprint(protocol)
    zero = {"protocol": copy.deepcopy(protocol), "protocol_fingerprint": fingerprint}
    records = []
    for split in ("calibration", "test"):
        score_dir = tmp_path / f"{split}-scores"
        score_dir.mkdir()
        manifest_path, manifest = _write_hashed_score_manifest(
            score_dir,
            protocol,
            image_ids=["cal-0" if split == "calibration" else "test-0"],
            labels_loaded=True,
        )
        grid_path = tmp_path / f"{split}-grid.npy"
        np.save(grid_path, grid)
        records.append(
            {
                "source_type": "exported_score_map_directory",
                "score_dir": str(score_dir),
                "manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
                "score_manifest_schema_version": manifest["schema_version"],
                "score_records_sha256": manifest["records_sha256"],
                "score_ordered_image_ids_sha256": manifest[
                    "ordered_image_ids_sha256"
                ],
                "score_num_records": manifest["num_images"],
                "split_file": None,
                "split_file_sha256": None,
                "split_ordered_ids_sha256": None,
                "threshold_grid": str(grid_path),
                "threshold_grid_file_sha256": hashlib.sha256(
                    grid_path.read_bytes()
                ).hexdigest(),
                "threshold_grid_sha256": threshold_grid_sha256(grid),
                "protocol": copy.deepcopy(protocol),
                "protocol_fingerprint": fingerprint,
            }
        )
    calibration, test = records
    return zero, calibration, test


def _adaptive_zero_result(tmp_path: Path) -> dict:
    artifact_path = tmp_path / "deployment-statistics.npz"
    artifact_path.write_bytes(b"immutable deployment statistics fixture")
    artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    checkpoint_path = tmp_path / "curve.pt"
    score_map_dir = tmp_path / "scores"
    score_map_dir.mkdir()
    protocol = _formal_protocol()
    grid = np.asarray([0.0, 0.5, 0.9], dtype=np.float32)
    protocol["threshold_grid_sha256"] = threshold_grid_sha256(grid)
    split_contract = {
        "verified": True,
        "formal_protocol_eligible": True,
        "adaptation_window": 1,
        "evaluation_window": 1,
        "stride": 2,
    }
    torch.save(
        {
            "threshold_grid_sha256": protocol["threshold_grid_sha256"],
            "episode_contract": {
                "schema_version": TRAINING_EPISODE_CONTRACT_SCHEMA_VERSION,
                "verified": True,
                "formal_protocol_eligible": True,
                "ineligibility_reasons": [],
                "adaptation_window": 1,
                "evaluation_window": 1,
                "stride": 2,
                "risk_target_unit": "aggregate_risk_over_1_future_images",
                "one_to_one_future_target": True,
                "protocol_fields": {
                    "protocol": "causal_adaptation_then_future_evaluation",
                    "adaptation_window": 1,
                    "evaluation_window": 1,
                    "stride": 2,
                    "pseudo_targets": ["source-a", "source-b"],
                    "threshold_grid_sha256": protocol["threshold_grid_sha256"],
                },
                "train": dict(split_contract),
                "validation": dict(split_contract),
            },
        },
        checkpoint_path,
    )
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest_path, _ = _write_hashed_score_manifest(
        score_map_dir,
        protocol,
        image_ids=["warm-0", "warm-1"],
        labels_loaded=False,
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    fingerprint = protocol_fingerprint(protocol)
    provenance = {
        "adaptation_window": 1,
        "evaluation_window": 1,
        "stride": 2,
        "num_windows": 2,
        "masks_read": False,
        "score_map_dir": str(score_map_dir),
        "score_manifest_sha256": manifest_sha,
        "threshold_grid_sha256": protocol["threshold_grid_sha256"],
        "global_role_overlap": [],
        "allow_role_reuse": False,
    }
    return {
        "schema_version": ZERO_RESULT_SCHEMA_VERSION,
        "mode": "zero_label_empirical_adaptation",
        "adaptation_protocol": "external_causal_statistics",
        "selection_data_contract": {
            "schema_version": SELECTION_DATA_CONTRACT_SCHEMA_VERSION,
            "masks_read": False,
            "statistics_computed_from": "causal_adaptation_blocks_A",
            "evaluation_labels_or_masks_used": False,
            "threshold_mapping_rule": "one_A_block_prediction_to_its_future_E_identity",
            "checkpoint_training_contract_verified": True,
        },
        "statistics_schema_version": STATISTICS_SCHEMA_VERSION,
        "statistics_file": str(artifact_path),
        "statistics_file_sha256": artifact_sha,
        "statistics_artifact": {
            "source_type": "deployment_statistics_archive",
            "path": str(artifact_path),
            "sha256": artifact_sha,
            "deployment_statistics_schema_version": DEPLOYMENT_STATISTICS_SCHEMA_VERSION,
            "statistics_schema_version": STATISTICS_SCHEMA_VERSION,
            "threshold_grid_sha256": protocol["threshold_grid_sha256"],
            "provenance": provenance,
        },
        "protocol": protocol,
        "protocol_fingerprint": fingerprint,
        "thresholds": grid.tolist(),
        "curve_checkpoint": str(checkpoint_path),
        "curve_checkpoint_sha256": checkpoint_sha,
        "num_windows": 2,
        "adaptation_ids": [["warm-0"], ["warm-1"]],
        "evaluation_ids": [["cal-0"], ["test-0"]],
        "window_ids": ["warm-0", "warm-1"],
        "threshold_indices": [0, 0],
        "threshold_indices_by_image": {"cal-0": 0, "test-0": 0},
    }


def test_forged_fingerprint_and_precomputed_artifact_are_not_formal(tmp_path: Path):
    protocol = _formal_protocol()
    zero, calibration, test = _protocol_records(protocol, tmp_path)
    assert audit_protocol_bundle(zero, calibration, test)["verified"] is True

    zero["protocol_fingerprint"] = "f" * 64
    forged = audit_protocol_bundle(zero, calibration, test)
    assert forged["verified"] is False
    assert "does not match" in forged["errors"]["zero"]

    fresh_path = tmp_path / "fresh"
    fresh_path.mkdir()
    zero, calibration, test = _protocol_records(protocol, fresh_path)
    calibration["source_type"] = "precomputed_count_curve_archive"
    legacy = audit_protocol_bundle(zero, calibration, test)
    assert legacy["verified"] is False
    assert "diagnostic" in legacy["errors"]["calibration"]


def test_target_domain_in_detector_sources_is_rejected():
    protocol = _formal_protocol()
    protocol["source_datasets"] = ["source-a", "target-domain"]
    with pytest.raises(ValueError, match="contains target_dataset"):
        validate_formal_protocol(protocol, protocol_fingerprint(protocol))


def test_formal_sample_adaptive_contract_rejects_e_greater_than_one(tmp_path: Path):
    zero = _adaptive_zero_result(tmp_path)
    zero["statistics_artifact"]["provenance"]["evaluation_window"] = 2
    zero["statistics_artifact"]["provenance"]["stride"] = 3
    audit = audit_zero_artifact_contract(zero, ["cal-0"], ["test-0"])
    assert audit["verified"] is False
    assert "differ in evaluation_window" in audit["errors"]["zero_artifact"]


def _rewrite_checkpoint(zero: dict, mutate) -> None:
    checkpoint_path = Path(zero["curve_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    mutate(checkpoint)
    torch.save(checkpoint, checkpoint_path)
    zero["curve_checkpoint_sha256"] = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()


def test_formal_chain_rejects_checkpoint_deployment_unit_mismatch(tmp_path: Path):
    zero = _adaptive_zero_result(tmp_path)

    def make_e2(checkpoint: dict) -> None:
        contract = checkpoint["episode_contract"]
        contract["evaluation_window"] = 2
        contract["stride"] = 3
        contract["risk_target_unit"] = "aggregate_risk_over_2_future_images"
        contract["one_to_one_future_target"] = False
        contract["protocol_fields"]["evaluation_window"] = 2
        contract["protocol_fields"]["stride"] = 3
        for split in ("train", "validation"):
            contract[split]["evaluation_window"] = 2
            contract[split]["stride"] = 3

    _rewrite_checkpoint(zero, make_e2)
    audit = audit_zero_artifact_contract(zero, ["cal-0"], ["test-0"])
    assert audit["verified"] is False
    assert "evaluation_window=1" in audit["errors"]["zero_artifact"]


def test_formal_chain_rejects_risk_predictor_target_domain_leakage(tmp_path: Path):
    zero = _adaptive_zero_result(tmp_path)

    def leak_target(checkpoint: dict) -> None:
        checkpoint["episode_contract"]["protocol_fields"]["pseudo_targets"].append(
            "target_domain"
        )

    _rewrite_checkpoint(zero, leak_target)
    audit = audit_zero_artifact_contract(zero, ["cal-0"], ["test-0"])
    assert audit["verified"] is False
    assert "appears in risk-predictor pseudo-target" in audit["errors"][
        "zero_artifact"
    ]


def test_checkpoint_deployment_contract_binds_source_reference(tmp_path: Path):
    zero = _adaptive_zero_result(tmp_path)
    checkpoint = torch.load(
        zero["curve_checkpoint"], map_location="cpu", weights_only=True
    )
    provenance = dict(zero["statistics_artifact"]["provenance"])
    baseline = validate_checkpoint_deployment_contract(
        checkpoint,
        deployment_provenance=provenance,
        target_dataset=zero["protocol"]["target_dataset"],
        expected_threshold_grid_sha256=zero["protocol"]["threshold_grid_sha256"],
    )
    assert baseline["source_reference_contract_match"] is True

    provenance["source_reference_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="source-reference contracts differ"):
        validate_checkpoint_deployment_contract(
            checkpoint,
            deployment_provenance=provenance,
            target_dataset=zero["protocol"]["target_dataset"],
            expected_threshold_grid_sha256=zero["protocol"]["threshold_grid_sha256"],
        )


def test_label_posterior_mapping_not_derived_from_blocks_is_rejected(tmp_path: Path):
    zero = _adaptive_zero_result(tmp_path)
    baseline = audit_zero_artifact_contract(zero, ["cal-0"], ["test-0"])
    assert baseline["verified"] is True

    zero["threshold_indices_by_image"] = {"cal-0": 1, "test-0": 0}
    audit = audit_zero_artifact_contract(zero, ["cal-0"], ["test-0"])
    assert audit["verified"] is False
    assert "one-to-one causal block mapping" in audit["errors"]["zero_artifact"]


def test_complete_reproducible_chain_can_enter_formal_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    zero_root = tmp_path / "zero-chain"
    zero_root.mkdir()
    zero = _adaptive_zero_result(zero_root)
    zero.update(
        {
            "reject": False,
            "pixel_budget": 1.0,
            "component_budget": 1.0,
        }
    )
    count_root = tmp_path / "count-chain"
    count_root.mkdir()
    _, calibration_provenance, test_provenance = _protocol_records(
        _formal_protocol(), count_root
    )
    thresholds = np.asarray(zero["thresholds"], dtype=np.float32)

    def write_count_archive(path: Path, image_id: str, provenance: dict) -> None:
        _write_hashed_count_archive(
            path,
            {
                "image_ids": np.asarray([image_id]),
                "thresholds": thresholds,
                "false_positive_pixels": np.zeros(
                    (1, thresholds.size), dtype=np.int64
                ),
                "false_positive_components": np.zeros(
                    (1, thresholds.size), dtype=np.int64
                ),
                "total_pixels": np.asarray([64], dtype=np.int64),
                "provenance_json": np.asarray(
                    json.dumps(provenance, sort_keys=True)
                ),
            },
        )

    calibration_path = tmp_path / "calibration.npz"
    test_path = tmp_path / "test.npz"
    write_count_archive(calibration_path, "cal-0", calibration_provenance)
    write_count_archive(test_path, "test-0", test_provenance)
    zero_path = tmp_path / "zero.json"
    zero_path.write_text(json.dumps(zero), encoding="utf-8")
    output_path = tmp_path / "selection.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate_target_offset",
            "--calibration-curves",
            str(calibration_path),
            "--test-curves",
            str(test_path),
            "--zero-result",
            str(zero_path),
            "--alpha",
            "0.5",
            "--output",
            str(output_path),
        ],
    )
    calibrate_main()
    selection = json.loads(output_path.read_text(encoding="utf-8"))
    assert selection["formal_artifact_chain_verified"] is True
    assert selection["protocol_audit"]["verified"] is True
    assert selection["zero_artifact_audit"]["verified"] is True
    assert "JointBSR" in selection["guarantee_scope"]

    test_losses = build_calibration_losses(
        **load_count_curve_archive(test_path),
        pixel_budget=1.0,
        component_budget=1.0,
    )
    audit = evaluate_selected_operating_point(selection, test_losses)
    assert audit["success"] is True
    assert audit["test_action_contract_audit"]["verified"] is True
    assert audit["test_action_contract_audit"]["formal_artifact_chain_verified"] is True

    forged = copy.deepcopy(selection)
    forged["test_zero_threshold_indices"][0] = 1
    forged["selected_test_threshold_indices"][0] = 1 + forged["offset_rank"]
    with pytest.raises(ValueError, match="formal zero-label artifact"):
        evaluate_selected_operating_point(forged, test_losses)
