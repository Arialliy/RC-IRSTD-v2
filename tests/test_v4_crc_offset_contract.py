import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import certification.build_calibration_losses as loss_builder
from certification.build_calibration_losses import (
    RAW_LOGIT_LOSS_SCHEMA_VERSION,
    build_calibration_losses,
    build_count_curves_from_score_maps,
    score_map_protocol,
    validate_formal_protocol,
)
from certification.calibrate_target_offset import (
    RAW_LOGIT_RESULT_SCHEMA_VERSION,
    SELECTION_DATA_CONTRACT_SCHEMA_VERSION,
    ZERO_RESULT_SCHEMA_VERSION,
    _raw_count_archive_provenance_binding,
    audit_zero_artifact_contract,
    calibrate_target_offset,
)
import certification.calibrate_target_offset as offset_calibration
from certification.evaluate_certified_mode import (
    RAW_LOGIT_EVALUATION_SCHEMA_VERSION,
    evaluate_selected_operating_point,
)
from evaluation.artifact_integrity import (
    PROBABILITY_DTYPE,
    RAW_LOGIT_DTYPE,
    RAW_LOGIT_SCORE_REPRESENTATION,
    SCORE_MANIFEST_SCHEMA_VERSION,
    SCORE_RECORD_INTEGRITY_SCHEMA,
)
from risk_curve.deployment_contract import (
    audit_checkpoint_deployment_contract,
    validate_checkpoint_deployment_contract,
)
from risk_curve.domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    feature_schema_sha256,
    statistics_names_sha256,
)
from risk_curve.monotone_curve_predictor import RISK_CURVE_ARCHITECTURE_VERSION
from risk_curve.representation import (
    GRID_DETECTOR_PROTOCOL,
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_REPRESENTATION,
    empty_action_contract,
    logit_threshold_grid_sha256,
)
from risk_curve.train_curve_predictor import (
    TRAINING_EPISODE_CONTRACT_SCHEMA_VERSION,
)


GRID = np.asarray([-2.0, 0.0, 2.0], dtype=np.float32)
GRID_SHA = logit_threshold_grid_sha256(GRID)
GRID_MANIFEST_SHA = "a" * 64
DETECTOR_HASHES = ["b" * 64, "c" * 64]
OUTER_DETECTOR_HASH = DETECTOR_HASHES[0]
EPISODE_DETECTOR_HASHES = [DETECTOR_HASHES[1]]
CURVE_HASH = "d" * 64


def _raw_losses(num_images: int = 20, prefix: str = "cal"):
    fp_pixels = np.tile(np.asarray([2, 1, 0]), (num_images, 1))
    fp_components = fp_pixels.copy()
    return build_calibration_losses(
        image_ids=[f"{prefix}-{index}" for index in range(num_images)],
        thresholds=GRID,
        false_positive_pixels=fp_pixels,
        false_positive_components=fp_components,
        total_pixels=np.full(num_images, 1_000_000),
        pixel_budget=1e-6,
        component_budget=1.0,
        representation=LOGIT_REPRESENTATION,
        threshold_grid_schema_version=LOGIT_GRID_SCHEMA_VERSION,
        recorded_threshold_grid_sha256=GRID_SHA,
        threshold_grid_manifest_sha256=GRID_MANIFEST_SHA,
        threshold_grid_detector_protocol=GRID_DETECTOR_PROTOCOL,
        threshold_grid_detector_checkpoint_sha256s=DETECTOR_HASHES,
        threshold_grid_outer_detector_checkpoint_sha256=OUTER_DETECTOR_HASH,
        threshold_grid_episode_detector_checkpoint_sha256s=(
            EPISODE_DETECTOR_HASHES
        ),
    )


def _raw_checkpoint() -> tuple[dict, dict]:
    names = ("feature-a", "feature-b")
    feature_hash = feature_schema_sha256(
        LOGIT_STATISTICS_SCHEMA_VERSION,
        statistics_names=names,
    )
    split = {
        "verified": True,
        "formal_protocol_eligible": True,
        "adaptation_window": 1,
        "evaluation_window": 1,
        "stride": 2,
    }
    protocol_fields = {
        "protocol": "causal_adaptation_then_future_evaluation",
        "adaptation_window": 1,
        "evaluation_window": 1,
        "stride": 2,
        "pseudo_targets": ["source-a", "source-b"],
        "source_reference_sha256": None,
        "source_reference_domain_names": [],
        "source_reference_statistics_names_sha256": None,
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": GRID_SHA,
        "threshold_grid_manifest_sha256": GRID_MANIFEST_SHA,
        "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": DETECTOR_HASHES,
        "threshold_grid_outer_detector_checkpoint_sha256": OUTER_DETECTOR_HASH,
        "threshold_grid_episode_detector_checkpoint_sha256s": (
            EPISODE_DETECTOR_HASHES
        ),
        "feature_schema_sha256": feature_hash,
    }
    episode_contract = {
        "schema_version": TRAINING_EPISODE_CONTRACT_SCHEMA_VERSION,
        "verified": True,
        "formal_protocol_eligible": True,
        "ineligibility_reasons": [],
        "adaptation_window": 1,
        "evaluation_window": 1,
        "stride": 2,
        "risk_target_unit": "aggregate_risk_over_1_future_images",
        "one_to_one_future_target": True,
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": GRID_SHA,
        "threshold_grid_manifest_sha256": GRID_MANIFEST_SHA,
        "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": DETECTOR_HASHES,
        "threshold_grid_outer_detector_checkpoint_sha256": OUTER_DETECTOR_HASH,
        "threshold_grid_episode_detector_checkpoint_sha256s": (
            EPISODE_DETECTOR_HASHES
        ),
        "feature_schema_sha256": feature_hash,
        "protocol_fields": protocol_fields,
        "train": dict(split),
        "validation": dict(split),
    }
    checkpoint = {
        "representation": LOGIT_REPRESENTATION,
        "thresholds": GRID.tolist(),
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": GRID_SHA,
        "threshold_grid_manifest_sha256": GRID_MANIFEST_SHA,
        "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": DETECTOR_HASHES,
        "threshold_grid_outer_detector_checkpoint_sha256": OUTER_DETECTOR_HASH,
        "threshold_grid_episode_detector_checkpoint_sha256s": (
            EPISODE_DETECTOR_HASHES
        ),
        "statistics_schema_version": LOGIT_STATISTICS_SCHEMA_VERSION,
        "statistics_names": names,
        "statistics_names_sha256": statistics_names_sha256(names),
        "feature_schema_sha256": feature_hash,
        "model_architecture_version": RISK_CURVE_ARCHITECTURE_VERSION,
        "model_config": {
            "input_dim": len(names),
            "num_thresholds": len(GRID),
            "architecture_version": RISK_CURVE_ARCHITECTURE_VERSION,
        },
        "episode_contract": episode_contract,
    }
    deployment = {
        "adaptation_window": 1,
        "evaluation_window": 1,
        "stride": 2,
        "source_reference_sha256": None,
        "source_reference_domain_names": [],
        "source_reference_statistics_names_sha256": None,
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": GRID_SHA,
        "threshold_grid_manifest_sha256": GRID_MANIFEST_SHA,
        "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": DETECTOR_HASHES,
        "threshold_grid_outer_detector_checkpoint_sha256": OUTER_DETECTOR_HASH,
        "threshold_grid_episode_detector_checkpoint_sha256s": (
            EPISODE_DETECTOR_HASHES
        ),
    }
    return checkpoint, deployment


def test_raw_logit_count_curves_never_sigmoid_or_reindex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    record_path = tmp_path / "sample.npz"
    np.savez_compressed(
        record_path,
        logit=np.asarray([[-2.0, 0.0], [2.0, 4.0]], dtype=np.float32),
        mask=np.zeros((2, 2), dtype=np.uint8),
    )
    manifest = {
        "schema_version": SCORE_MANIFEST_SCHEMA_VERSION,
        "record_integrity_schema": SCORE_RECORD_INTEGRITY_SCHEMA,
        "labels_loaded": True,
        "records": [{"image_id": "sample"}],
        "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
        "probability_dtype": PROBABILITY_DTYPE,
        "logit_dtype": RAW_LOGIT_DTYPE,
        "probability_transform": "sigmoid",
        "probability_clipping": "none",
        "inference_autocast_enabled": False,
    }
    monkeypatch.setattr(
        loss_builder,
        "verify_score_map_directory",
        lambda *args, **kwargs: (
            manifest,
            [record_path],
            {"verified": True},
        ),
    )
    raw_grid = np.asarray([0.0, 3.0], dtype=np.float32)
    counts = build_count_curves_from_score_maps(
        tmp_path,
        raw_grid,
        representation=LOGIT_REPRESENTATION,
    )
    # raw >= 0 selects three pixels; raw >= 3 selects one.  Applying sigmoid
    # first would produce [4, 0], so this is a representation-sensitive test.
    np.testing.assert_array_equal(counts["false_positive_pixels"], [[3, 1]])
    assert counts["thresholds"].dtype == np.float32


def test_raw_logit_score_protocol_binds_detector_roles(tmp_path: Path):
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    (score_dir / "manifest.json").write_text(
        json.dumps(
            {
                "score_type": "sigmoid_probability",
                "warm_flag": False,
                "spatial_mode": "native",
                "pad_multiple": 16,
                "target_dataset": "target-domain",
                "source_datasets": ["source-a", "source-b"],
                "weight_sha256": OUTER_DETECTOR_HASH,
                "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
                "probability_dtype": PROBABILITY_DTYPE,
                "logit_dtype": RAW_LOGIT_DTYPE,
                "probability_transform": "sigmoid",
                "probability_clipping": "none",
                "inference_autocast_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    protocol, fingerprint = score_map_protocol(
        score_dir,
        GRID,
        matching_rule="overlap",
        centroid_distance=3.0,
        connectivity=2,
        min_component_area=1,
        representation=LOGIT_REPRESENTATION,
        threshold_grid_detector_protocol=GRID_DETECTOR_PROTOCOL,
        threshold_grid_detector_checkpoint_sha256s=DETECTOR_HASHES,
        threshold_grid_outer_detector_checkpoint_sha256=(
            OUTER_DETECTOR_HASH
        ),
        threshold_grid_episode_detector_checkpoint_sha256s=(
            EPISODE_DETECTOR_HASHES
        ),
    )
    canonical, recomputed = validate_formal_protocol(protocol, fingerprint)
    assert canonical == protocol
    assert recomputed == fingerprint
    assert protocol["threshold_grid_outer_detector_checkpoint_sha256"] == (
        OUTER_DETECTOR_HASH
    )
    assert protocol["threshold_grid_episode_detector_checkpoint_sha256s"] == (
        EPISODE_DETECTOR_HASHES
    )


def test_raw_logit_crc_binds_grid_checkpoint_and_external_reject():
    losses = _raw_losses()
    result = calibrate_target_offset(
        losses,
        zero_index=0,
        alpha=0.1,
        test_image_ids=[f"test-{index}" for index in range(20)],
        curve_checkpoint_sha256=CURVE_HASH,
    )
    assert result["schema_version"] == RAW_LOGIT_RESULT_SCHEMA_VERSION
    assert result["loss"]["schema_version"] == RAW_LOGIT_LOSS_SCHEMA_VERSION
    assert result["representation"] == LOGIT_REPRESENTATION
    assert result["selected_threshold_index"] == 1
    assert result["selected_logit_threshold"] == 0.0
    assert result["selected_probability_threshold"] == pytest.approx(0.5)
    assert result["threshold_grid_sha256"] == GRID_SHA
    assert result["threshold_grid_manifest_sha256"] == GRID_MANIFEST_SHA
    assert result["threshold_grid_detector_checkpoint_sha256s"] == DETECTOR_HASHES
    assert result["threshold_grid_outer_detector_checkpoint_sha256"] == (
        OUTER_DETECTOR_HASH
    )
    assert result["threshold_grid_episode_detector_checkpoint_sha256s"] == (
        EPISODE_DETECTOR_HASHES
    )
    assert result["curve_checkpoint_sha256"] == CURVE_HASH
    assert result["empty_action"] == empty_action_contract()
    assert result["empty_action"]["threshold_index"] is None
    assert result["empty_action"]["threshold"] == "+inf"


def test_raw_logit_crc_fails_closed_on_missing_or_mismatched_binding():
    with pytest.raises(ValueError, match="curve checkpoint SHA-256"):
        calibrate_target_offset(
            _raw_losses(),
            zero_index=0,
            alpha=0.1,
            test_image_ids=[f"test-{index}" for index in range(20)],
        )
    with pytest.raises(ValueError, match="semantic grid hash mismatch"):
        build_calibration_losses(
            image_ids=["a"],
            thresholds=GRID,
            false_positive_pixels=np.zeros((1, 3)),
            false_positive_components=np.zeros((1, 3)),
            total_pixels=np.ones(1),
            pixel_budget=1.0,
            component_budget=1.0,
            representation=LOGIT_REPRESENTATION,
            threshold_grid_schema_version=LOGIT_GRID_SCHEMA_VERSION,
            recorded_threshold_grid_sha256="f" * 64,
            threshold_grid_manifest_sha256=GRID_MANIFEST_SHA,
            threshold_grid_detector_protocol=GRID_DETECTOR_PROTOCOL,
            threshold_grid_detector_checkpoint_sha256s=DETECTOR_HASHES,
            threshold_grid_outer_detector_checkpoint_sha256=(
                OUTER_DETECTOR_HASH
            ),
            threshold_grid_episode_detector_checkpoint_sha256s=(
                EPISODE_DETECTOR_HASHES
            ),
        )


def test_raw_logit_checkpoint_deployment_contract_is_fail_closed(tmp_path: Path):
    checkpoint, deployment = _raw_checkpoint()
    audit = validate_checkpoint_deployment_contract(
        checkpoint,
        deployment_provenance=deployment,
        target_dataset="target-domain",
        expected_threshold_grid_sha256=GRID_SHA,
        expected_representation=LOGIT_REPRESENTATION,
        expected_threshold_grid_schema_version=LOGIT_GRID_SCHEMA_VERSION,
        expected_threshold_grid_manifest_sha256=GRID_MANIFEST_SHA,
        expected_threshold_grid_detector_protocol=GRID_DETECTOR_PROTOCOL,
        expected_threshold_grid_detector_checkpoint_sha256s=DETECTOR_HASHES,
        expected_threshold_grid_outer_detector_checkpoint_sha256=(
            OUTER_DETECTOR_HASH
        ),
        expected_threshold_grid_episode_detector_checkpoint_sha256s=(
            EPISODE_DETECTOR_HASHES
        ),
    )
    assert audit["raw_logit_crc_contract_verified"] is True

    forged = dict(deployment)
    forged["threshold_grid_manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="threshold_grid_manifest_sha256"):
        validate_checkpoint_deployment_contract(
            checkpoint,
            deployment_provenance=forged,
            target_dataset="target-domain",
            expected_threshold_grid_sha256=GRID_SHA,
            expected_representation=LOGIT_REPRESENTATION,
        )

    duplicated = dict(checkpoint)
    duplicated["threshold_grid_detector_checkpoint_sha256s"] = [
        DETECTOR_HASHES[0],
        DETECTOR_HASHES[0],
    ]
    duplicated["threshold_grid_outer_detector_checkpoint_sha256"] = (
        DETECTOR_HASHES[0]
    )
    duplicated["threshold_grid_episode_detector_checkpoint_sha256s"] = [
        DETECTOR_HASHES[0]
    ]
    with pytest.raises(ValueError, match="distinct"):
        validate_checkpoint_deployment_contract(
            duplicated,
            deployment_provenance=deployment,
            target_dataset="target-domain",
            expected_threshold_grid_sha256=GRID_SHA,
            expected_representation=LOGIT_REPRESENTATION,
        )

    checkpoint_path = tmp_path / "curve.pt"
    torch.save(checkpoint, checkpoint_path)
    actual_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    verified = audit_checkpoint_deployment_contract(
        checkpoint_path,
        deployment_provenance=deployment,
        target_dataset="target-domain",
        expected_threshold_grid_sha256=GRID_SHA,
        expected_representation=LOGIT_REPRESENTATION,
        expected_threshold_grid_schema_version=LOGIT_GRID_SCHEMA_VERSION,
        expected_threshold_grid_manifest_sha256=GRID_MANIFEST_SHA,
        expected_threshold_grid_detector_protocol=GRID_DETECTOR_PROTOCOL,
        expected_threshold_grid_detector_checkpoint_sha256s=DETECTOR_HASHES,
        expected_threshold_grid_outer_detector_checkpoint_sha256=(
            OUTER_DETECTOR_HASH
        ),
        expected_threshold_grid_episode_detector_checkpoint_sha256s=(
            EPISODE_DETECTOR_HASHES
        ),
        expected_curve_checkpoint_sha256=actual_sha,
    )
    assert verified["verified"] is True
    assert verified["curve_checkpoint_sha256_verified"] is True
    rejected = audit_checkpoint_deployment_contract(
        checkpoint_path,
        deployment_provenance=deployment,
        target_dataset="target-domain",
        expected_threshold_grid_sha256=GRID_SHA,
        expected_representation=LOGIT_REPRESENTATION,
        expected_curve_checkpoint_sha256="e" * 64,
    )
    assert rejected["verified"] is False
    assert "SHA-256 mismatch" in rejected["errors"]["curve_checkpoint_contract"]


def _result_raw_binding(result: dict) -> dict:
    fields = (
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
    return {field: copy.deepcopy(result[field]) for field in fields}


def test_v4_evaluator_accepts_raw_logit_schema_and_binds_test_loss_contract():
    calibration = _raw_losses(prefix="cal")
    test = _raw_losses(prefix="test")
    result = calibrate_target_offset(
        calibration,
        zero_index=0,
        alpha=0.1,
        test_image_ids=test.image_ids,
        curve_checkpoint_sha256=CURVE_HASH,
    )

    audit = evaluate_selected_operating_point(result, test)

    assert audit["schema_version"] == RAW_LOGIT_EVALUATION_SCHEMA_VERSION
    assert audit["representation"] == LOGIT_REPRESENTATION
    binding = audit["representation_contract_audit"]
    assert binding["verified"] is True
    assert binding["semantic_grid_sha256_verified"] is True
    assert binding["detector_role_partition_verified"] is True
    assert audit["test_action_contract_audit"]["loss_schema"] == (
        RAW_LOGIT_LOSS_SCHEMA_VERSION
    )


@pytest.mark.parametrize(
    ("field", "forged"),
    [
        ("representation", "sigmoid_probability_float32"),
        ("threshold_grid_schema_version", "forged-grid-schema"),
        ("threshold_grid_sha256", "e" * 64),
        ("threshold_grid_manifest_sha256", "e" * 64),
        ("threshold_grid_detector_checkpoint_sha256s", list(reversed(DETECTOR_HASHES))),
        ("threshold_grid_outer_detector_checkpoint_sha256", EPISODE_DETECTOR_HASHES[0]),
        ("threshold_grid_episode_detector_checkpoint_sha256s", [OUTER_DETECTOR_HASH]),
        ("curve_checkpoint_sha256", "e" * 64),
        ("thresholds", [-2.0, 0.25, 2.0]),
        ("selected_probability_threshold", 0.75),
    ],
)
def test_v4_evaluator_rejects_raw_contract_tampering(field: str, forged):
    calibration = _raw_losses(prefix="cal")
    test = _raw_losses(prefix="test")
    result = calibrate_target_offset(
        calibration,
        zero_index=0,
        alpha=0.1,
        test_image_ids=test.image_ids,
        curve_checkpoint_sha256=CURVE_HASH,
    )
    result["provenance"] = _result_raw_binding(result)
    tampered = copy.deepcopy(result)
    tampered[field] = forged

    with pytest.raises(ValueError):
        evaluate_selected_operating_point(tampered, test)


def test_v4_evaluator_rejects_loss_and_provenance_tampering():
    calibration = _raw_losses(prefix="cal")
    test = _raw_losses(prefix="test")
    result = calibrate_target_offset(
        calibration,
        zero_index=0,
        alpha=0.1,
        test_image_ids=test.image_ids,
        curve_checkpoint_sha256=CURVE_HASH,
    )
    result["provenance"] = _result_raw_binding(result)

    bad_loss = copy.deepcopy(result)
    bad_loss["loss"]["schema_version"] = RAW_LOGIT_LOSS_SCHEMA_VERSION + "-forged"
    with pytest.raises(ValueError, match="loss schema"):
        evaluate_selected_operating_point(bad_loss, test)

    bad_provenance = copy.deepcopy(result)
    bad_provenance["provenance"]["threshold_grid_manifest_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="provenance differs"):
        evaluate_selected_operating_point(bad_provenance, test)


def test_v4_formal_evaluator_rehashes_curve_checkpoint(tmp_path: Path):
    calibration = _raw_losses(prefix="cal")
    test = _raw_losses(prefix="test")
    checkpoint_path = tmp_path / "curve.pt"
    checkpoint_path.write_bytes(b"formal curve checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    result = calibrate_target_offset(
        calibration,
        zero_index=0,
        alpha=0.1,
        test_image_ids=test.image_ids,
        curve_checkpoint_sha256=checkpoint_sha,
    )
    zero_result = {
        **_result_raw_binding(result),
        "prediction_rule": result["prediction_rule"],
        "empty_action": result["empty_action"],
        "curve_checkpoint": str(checkpoint_path),
        "threshold_index": 0,
    }
    zero_path = tmp_path / "zero.json"
    zero_path.write_text(json.dumps(zero_result), encoding="utf-8")
    result["formal_artifact_chain_verified"] = True
    result["protocol_audit"] = {
        "verified": True,
        "allow_unverified_protocol": False,
    }
    result["zero_artifact_audit"] = {"verified": True}
    result["guarantee_scope"] = "formal finite-sample control"
    result["provenance"] = {
        **_result_raw_binding(result),
        "zero_result": str(zero_path),
        "zero_result_sha256": hashlib.sha256(zero_path.read_bytes()).hexdigest(),
        "formal_artifact_chain_verified": True,
        "protocol_fingerprint_verified": True,
    }
    assert evaluate_selected_operating_point(result, test)["success"] is True

    checkpoint_path.write_bytes(b"tampered checkpoint")
    with pytest.raises(ValueError, match="curve checkpoint"):
        evaluate_selected_operating_point(result, test)


def test_raw_count_top_level_contract_must_equal_provenance():
    losses = _raw_losses(num_images=2)
    metadata = losses.metadata()
    fields = (
        "representation",
        "threshold_grid_schema_version",
        "threshold_grid_sha256",
        "threshold_grid_manifest_sha256",
        "threshold_grid_detector_protocol",
        "threshold_grid_detector_checkpoint_sha256s",
        "threshold_grid_outer_detector_checkpoint_sha256",
        "threshold_grid_episode_detector_checkpoint_sha256s",
    )
    archive_contract = {field: copy.deepcopy(metadata[field]) for field in fields}
    provenance = copy.deepcopy(archive_contract)
    verified = _raw_count_archive_provenance_binding(
        archive_contract,
        provenance,
        losses.thresholds,
        split_name="test",
    )
    assert verified["verified"] is True

    provenance["threshold_grid_manifest_sha256"] = "e" * 64
    rejected = _raw_count_archive_provenance_binding(
        archive_contract,
        provenance,
        losses.thresholds,
        split_name="test",
    )
    assert rejected["verified"] is False
    assert rejected["field_matches"]["threshold_grid_manifest_sha256"] is False


def test_global_raw_zero_audit_uses_score_provenance_before_checkpoint_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint_path = tmp_path / "curve.pt"
    checkpoint_path.write_bytes(b"curve")
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    score_provenance = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "sentinel": "global-causal-provenance",
    }
    raw_fields = {
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": GRID_SHA,
        "threshold_grid_manifest_sha256": GRID_MANIFEST_SHA,
        "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": DETECTOR_HASHES,
        "threshold_grid_outer_detector_checkpoint_sha256": OUTER_DETECTOR_HASH,
        "threshold_grid_episode_detector_checkpoint_sha256s": (
            EPISODE_DETECTOR_HASHES
        ),
    }
    zero_result = {
        "schema_version": ZERO_RESULT_SCHEMA_VERSION,
        "mode": "zero_label_empirical_adaptation",
        **raw_fields,
        "curve_checkpoint": str(checkpoint_path),
        "curve_checkpoint_sha256": checkpoint_sha,
        "prediction_rule": "prediction = (raw_logits >= threshold)",
        "empty_action": empty_action_contract(),
        "adaptation_protocol": "causal_warmup",
        "score_map_provenance": score_provenance,
        "protocol": {
            "target_dataset": "target-domain",
            "threshold_grid_sha256": GRID_SHA,
        },
        "selection_data_contract": {
            "schema_version": SELECTION_DATA_CONTRACT_SCHEMA_VERSION,
            "masks_read": False,
            "checkpoint_training_contract_verified": True,
            "threshold_mapping_rule": "one_global_warmup_prediction",
            "empty_action": empty_action_contract(),
            **raw_fields,
        },
    }
    captured: dict = {}

    def fake_checkpoint_audit(path, *, deployment_provenance, **kwargs):
        captured["deployment_provenance"] = deployment_provenance
        return {"verified": True, "errors": {}}

    monkeypatch.setattr(
        offset_calibration,
        "audit_checkpoint_deployment_contract",
        fake_checkpoint_audit,
    )
    monkeypatch.setattr(
        offset_calibration,
        "_verify_zero_protocol_against_manifest",
        lambda *args, **kwargs: None,
    )

    audit = audit_zero_artifact_contract(zero_result, ["cal-0"], ["test-0"])

    assert audit["verified"] is True
    assert captured["deployment_provenance"] is score_provenance
    assert audit["mode"] == "global_causal_warmup"
