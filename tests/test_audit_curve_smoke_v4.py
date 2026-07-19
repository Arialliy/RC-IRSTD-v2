import json
from pathlib import Path

import numpy as np
import pytest
import torch

from risk_curve.audit_curve_smoke_v4 import audit_curve_smoke
from risk_curve.build_curve_episodes import (
    COMPONENT_RISK_SCHEMA_VERSION,
    LOGIT_EPISODE_SCHEMA_VERSION,
)
from risk_curve.curve_dataset import load_curve_archive
from risk_curve.domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    feature_schema_sha256,
    statistics_names_sha256,
)
from risk_curve.monotone_curve_predictor import (
    RISK_CURVE_ARCHITECTURE_VERSION,
    RiskCurvePredictor,
)
from risk_curve.representation import (
    GRID_DETECTOR_PROTOCOL,
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_REPRESENTATION,
    logit_threshold_grid_sha256,
)
from risk_curve.train_curve_predictor import (
    TRAINING_EPISODE_CONTRACT_SCHEMA_VERSION,
    _archive_episode_contract,
)


NAMES = ("feature-a", "feature-b", "feature-c")
DETECTOR_HASHES = ("b" * 64, "c" * 64, "d" * 64)
EPISODE_DETECTOR_HASHES = DETECTOR_HASHES[:2]
OUTER_DETECTOR_HASH = DETECTOR_HASHES[-1]
MANIFEST_HASH = "a" * 64


def _write_validation_archive(path: Path) -> None:
    thresholds = np.linspace(-8.0, 8.0, 16, dtype=np.float32)
    statistics = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [2.0, -1.0, 0.5],
            [-2.0, 3.0, 1.0],
            [4.0, 0.2, -3.0],
        ],
        dtype=np.float32,
    )
    rows = statistics.shape[0]
    grid_hash = logit_threshold_grid_sha256(thresholds)
    feature_hash = feature_schema_sha256(
        LOGIT_STATISTICS_SCHEMA_VERSION,
        statistics_names=NAMES,
    )
    pixel = np.tile(
        np.linspace(-0.5, -8.0, thresholds.size, dtype=np.float32),
        (rows, 1),
    )
    component = np.tile(
        np.linspace(0.0, -4.0, thresholds.size, dtype=np.float32),
        (rows, 1),
    )
    decreasing_counts = np.tile(
        np.arange(thresholds.size, 0, -1, dtype=np.int64),
        (rows, 1),
    )
    adaptation_ids = [[f"adapt-{index}"] for index in range(rows)]
    evaluation_ids = [[f"future-{index}"] for index in range(rows)]
    provenance = {
        "archive_split": "validation",
        "protocol": "causal_adaptation_then_future_evaluation",
        "representation": LOGIT_REPRESENTATION,
        "adaptation_window": 1,
        "evaluation_window": 1,
        "stride": 2,
        "matching_rule": "overlap",
        "centroid_distance": 3.0,
        "connectivity": 2,
        "min_component_area": 1,
        "source_reference": None,
        "source_reference_sha256": None,
        "source_reference_domain_names": [],
        "source_reference_statistics_names_sha256": None,
        "pseudo_targets": ["NUDT-SIRST", "IRSTD-1K"],
        "validation_domain": "IRSTD-1K",
        "paired_lodo_validation_domains": ["IRSTD-1K", "NUDT-SIRST"],
        "pseudo_target_split": "train",
        "expected_split_role": "train",
        "statistics_sample_role": "adaptation_window_A_label_free",
        "risk_label_sample_role": "immediately_following_evaluation_window_E",
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": grid_hash,
        "threshold_grid_manifest_sha256": MANIFEST_HASH,
        "threshold_grid_outer_target_excluded": True,
        "threshold_grid_outer_target_key": "nuaa",
        "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": list(DETECTOR_HASHES),
        "threshold_grid_outer_detector_checkpoint_sha256": OUTER_DETECTOR_HASH,
        "threshold_grid_episode_detector_checkpoint_sha256s": list(
            EPISODE_DETECTOR_HASHES
        ),
        "threshold_grid_source_domains": ["NUDT-SIRST", "IRSTD-1K"],
        "feature_schema_sha256": feature_hash,
        "fold_provenance_audits": [
            {
                "verified": True,
                "pseudo_target": target,
                "detector_weight_sha256": digest,
            }
            for target, digest in zip(
                ("NUDT-SIRST", "IRSTD-1K"), EPISODE_DETECTOR_HASHES
            )
        ],
        "fold_provenance_verified": True,
        "allow_unverified_fold_provenance": False,
        "allow_cross_episode_role_reuse": False,
        "cross_episode_role_reuse_detected": False,
        "cross_episode_role_reuse_ids": [],
        "formal_causal_contract_verified": True,
        "protocol_scope": "formal_causal",
    }
    np.savez_compressed(
        path,
        statistics=statistics,
        statistics_names=np.asarray(NAMES),
        statistics_names_sha256=np.asarray(statistics_names_sha256(NAMES)),
        statistics_schema_version=np.asarray(LOGIT_STATISTICS_SCHEMA_VERSION),
        feature_schema_sha256=np.asarray(feature_hash),
        pixel_log_risk=pixel,
        component_log_risk=component,
        component_log_risk_raw=component,
        component_log_risk_upper=component,
        component_risk_schema_version=np.asarray(COMPONENT_RISK_SCHEMA_VERSION),
        component_log_risk_alias=np.asarray("component_log_risk_upper"),
        pd_curve=np.ones_like(pixel),
        pixel_fp_counts=decreasing_counts,
        component_fp_counts=decreasing_counts,
        tp_object_counts=np.ones_like(decreasing_counts),
        gt_object_counts=np.ones(rows, dtype=np.int64),
        total_pixels=np.full(rows, 256, dtype=np.int64),
        thresholds=thresholds,
        representation=np.asarray(LOGIT_REPRESENTATION),
        threshold_grid_schema_version=np.asarray(LOGIT_GRID_SCHEMA_VERSION),
        threshold_grid_sha256=np.asarray(grid_hash),
        threshold_grid_manifest_sha256=np.asarray(MANIFEST_HASH),
        threshold_grid_detector_protocol=np.asarray(GRID_DETECTOR_PROTOCOL),
        threshold_grid_detector_checkpoint_sha256s=np.asarray(DETECTOR_HASHES),
        threshold_grid_outer_detector_checkpoint_sha256=np.asarray(
            OUTER_DETECTOR_HASH
        ),
        threshold_grid_episode_detector_checkpoint_sha256s=np.asarray(
            EPISODE_DETECTOR_HASHES
        ),
        episode_schema_version=np.asarray(LOGIT_EPISODE_SCHEMA_VERSION),
        adaptation_sizes=np.ones(rows, dtype=np.int64),
        evaluation_sizes=np.ones(rows, dtype=np.int64),
        adaptation_ids=np.asarray([json.dumps(row) for row in adaptation_ids]),
        evaluation_ids=np.asarray([json.dumps(row) for row in evaluation_ids]),
        pseudo_targets=np.asarray(["IRSTD-1K"] * rows),
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )


def _episode_contract(validation_path: Path) -> dict[str, object]:
    validation = _archive_episode_contract(
        load_curve_archive(validation_path),
        archive_path=validation_path,
        split_name="validation",
    )
    fields = (
        "episode_schema_version",
        "representation",
        "threshold_grid_schema_version",
        "threshold_grid_sha256",
        "feature_schema_sha256",
        "threshold_grid_manifest_sha256",
        "threshold_grid_detector_protocol",
        "threshold_grid_detector_checkpoint_sha256s",
        "threshold_grid_outer_detector_checkpoint_sha256",
        "threshold_grid_episode_detector_checkpoint_sha256s",
        "adaptation_window",
        "evaluation_window",
        "stride",
        "risk_target_unit",
        "one_to_one_future_target",
        "deployment_compatibility_rule",
        "protocol_fields",
    )
    contract: dict[str, object] = {
        "schema_version": TRAINING_EPISODE_CONTRACT_SCHEMA_VERSION,
        "verified": True,
        "formal_protocol_eligible": True,
        "ineligibility_reasons": [],
        "train_validation_image_id_overlap": [],
        "validation": validation,
    }
    contract.update({field: validation[field] for field in fields})
    return contract


def _write_checkpoint_bundle(
    checkpoint_path: Path,
    validation_path: Path,
    *,
    constant_model: bool = False,
    seed: int = 1,
) -> None:
    archive = load_curve_archive(validation_path)
    thresholds = np.asarray(archive["thresholds"], dtype=np.float32)
    torch.manual_seed(1)
    model = RiskCurvePredictor(
        input_dim=len(NAMES),
        num_thresholds=thresholds.size,
        hidden_dim=8,
        dropout=0.0,
    )
    with torch.no_grad():
        model.pixel_head.total_drop.bias.fill_(20.0)
        model.component_head.total_drop.bias.fill_(20.0)
        if constant_model:
            for parameter in model.parameters():
                parameter.zero_()
            model.pixel_head.total_drop.bias.fill_(20.0)
            model.component_head.total_drop.bias.fill_(20.0)
    episode_contract = _episode_contract(validation_path)
    checkpoint = {
        "method_name": "risk_curve",
        "model_class": "RiskCurvePredictor",
        "model_architecture_version": RISK_CURVE_ARCHITECTURE_VERSION,
        "role": "proposed_method",
        "state_dict": model.state_dict(),
        "model_config": model.config(),
        "thresholds": thresholds.tolist(),
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": str(archive["threshold_grid_sha256"].item()),
        "feature_schema_sha256": str(archive["feature_schema_sha256"].item()),
        "threshold_grid_manifest_sha256": MANIFEST_HASH,
        "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": list(DETECTOR_HASHES),
        "threshold_grid_outer_detector_checkpoint_sha256": OUTER_DETECTOR_HASH,
        "threshold_grid_episode_detector_checkpoint_sha256s": list(
            EPISODE_DETECTOR_HASHES
        ),
        "statistics_mean": [0.0] * len(NAMES),
        "statistics_std": [1.0] * len(NAMES),
        "statistics_schema_version": LOGIT_STATISTICS_SCHEMA_VERSION,
        "statistics_names": list(NAMES),
        "statistics_names_sha256": statistics_names_sha256(NAMES),
        "quantile": 0.9,
        "lambda_component": 1.0,
        "seed": seed,
        "best_epoch": 2,
        "validation_metrics": {"quantile_pinball_objective": 1.0},
        "episode_contract": episode_contract,
    }
    torch.save(checkpoint, checkpoint_path)
    metrics = {
        "method_name": "risk_curve",
        "model_class": "RiskCurvePredictor",
        "model_architecture_version": RISK_CURVE_ARCHITECTURE_VERSION,
        "role": "proposed_method",
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": checkpoint["threshold_grid_sha256"],
        "feature_schema_sha256": checkpoint["feature_schema_sha256"],
        "threshold_grid_manifest_sha256": MANIFEST_HASH,
        "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": list(DETECTOR_HASHES),
        "threshold_grid_outer_detector_checkpoint_sha256": OUTER_DETECTOR_HASH,
        "threshold_grid_episode_detector_checkpoint_sha256s": list(
            EPISODE_DETECTOR_HASHES
        ),
        "best_objective": 1.0,
        "selection_objective": "validation_quantile_pinball",
        "quantile": 0.9,
        "lambda_component": 1.0,
        "episode_contract": episode_contract,
        "history": [
            {"epoch": 0, "train_loss": 3.0, "val_objective": 3.0},
            {"epoch": 1, "train_loss": 2.0, "val_objective": 2.0},
            {"epoch": 2, "train_loss": 1.0, "val_objective": 1.0},
        ],
    }
    checkpoint_path.with_suffix(checkpoint_path.suffix + ".metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def test_gate_b_audit_passes_and_repeat_is_semantically_reproducible(
    tmp_path: Path,
) -> None:
    validation = tmp_path / "validation.npz"
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    output = tmp_path / "audit.json"
    _write_validation_archive(validation)
    _write_checkpoint_bundle(first, validation)
    _write_checkpoint_bundle(second, validation)

    audit_curve_smoke(
        validation_file=validation,
        checkpoint=first,
        repeat_checkpoint=second,
        output=output,
        device="cpu",
        batch_size=2,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["gate_b_pass"] is True
    assert payload["primary"]["curve_geometry"]["pixel"][
        "monotonic_violation_rate"
    ] == 0.0
    assert payload["primary"]["budget_selection"][
        "all_budgets_not_all_reject"
    ] is True
    assert payload["primary"]["budget_selection"][
        "selected_index_variation_observed"
    ] is True
    assert payload["repeat"]["semantic_reproducibility_pass"] is True
    assert payload["repeat"]["semantic_reproducibility_checks"] == {
        "best_epoch_equal": True,
        "history_equal": True,
        "predictions_equal": True,
        "seed_equal": True,
        "state_tensors_equal": True,
    }


def test_gate_b_audit_rejects_checkpoint_with_wrong_validation_archive_hash(
    tmp_path: Path,
) -> None:
    validation = tmp_path / "validation.npz"
    checkpoint_path = tmp_path / "model.pt"
    _write_validation_archive(validation)
    _write_checkpoint_bundle(checkpoint_path, validation)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["episode_contract"]["validation"]["archive_sha256"] = "e" * 64
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="validation archive SHA-256"):
        audit_curve_smoke(
            validation_file=validation,
            checkpoint=checkpoint_path,
            output=tmp_path / "audit.json",
            device="cpu",
        )


def test_gate_b_audit_reports_non_degenerate_and_selection_failure(
    tmp_path: Path,
) -> None:
    validation = tmp_path / "validation.npz"
    checkpoint_path = tmp_path / "constant.pt"
    output = tmp_path / "audit.json"
    _write_validation_archive(validation)
    _write_checkpoint_bundle(checkpoint_path, validation, constant_model=True)

    audit_curve_smoke(
        validation_file=validation,
        checkpoint=checkpoint_path,
        output=output,
        device="cpu",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["gate_b_pass"] is False
    assert payload["primary"]["gate_b_checks"]["curves_non_degenerate"] is False
    assert payload["primary"]["gate_b_checks"][
        "selected_index_variation_observed"
    ] is False


def test_optional_repeat_detects_seed_semantic_mismatch(tmp_path: Path) -> None:
    validation = tmp_path / "validation.npz"
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    output = tmp_path / "audit.json"
    _write_validation_archive(validation)
    _write_checkpoint_bundle(first, validation, seed=1)
    _write_checkpoint_bundle(second, validation, seed=2)

    audit_curve_smoke(
        validation_file=validation,
        checkpoint=first,
        repeat_checkpoint=second,
        output=output,
        device="cpu",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["gate_b_pass"] is False
    assert payload["repeat"]["semantic_reproducibility_checks"]["seed_equal"] is False
    assert payload["repeat"]["semantic_reproducibility_checks"][
        "state_tensors_equal"
    ] is True
    assert payload["repeat"]["semantic_reproducibility_pass"] is False


def test_metrics_sidecar_rejects_non_finite_json_constant(tmp_path: Path) -> None:
    validation = tmp_path / "validation.npz"
    checkpoint_path = tmp_path / "model.pt"
    _write_validation_archive(validation)
    _write_checkpoint_bundle(checkpoint_path, validation)
    metrics_path = checkpoint_path.with_suffix(".pt.metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["history"][0]["train_loss"] = float("nan")
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite constant NaN"):
        audit_curve_smoke(
            validation_file=validation,
            checkpoint=checkpoint_path,
            output=tmp_path / "audit.json",
            device="cpu",
        )
