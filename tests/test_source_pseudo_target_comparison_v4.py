from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rc_irstd.models.calibrator import (
    RC_DIRECT_ARCHITECTURE_VERSION,
    MonotoneBudgetCalibrator,
)
from risk_curve.build_curve_episodes import COMPONENT_RISK_SCHEMA_VERSION
from risk_curve.curve_dataset import LOGIT_EPISODE_SCHEMA_VERSION
from risk_curve.direct_calibrator import (
    ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL,
    RC_DIRECT_BUDGET_SCHEMA_VERSION,
    RC_DIRECT_CHECKPOINT_SCHEMA_VERSION,
)
from risk_curve.domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    feature_schema_sha256,
    statistics_names_sha256,
)
from risk_curve.evaluate_source_pseudo_target_v4 import (
    SOURCE_PSEUDO_TARGET_COMPARISON_SCHEMA_VERSION,
    evaluate_source_pseudo_target_comparison,
)
from risk_curve.monotone_curve_predictor import (
    RISK_CURVE_ARCHITECTURE_VERSION,
    RiskCurvePredictor,
)
from risk_curve.representation import (
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_REPRESENTATION,
    logit_threshold_grid_sha256,
)


GRID = np.asarray([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
NAMES = ("logit_feature_a", "logit_feature_b")
GRID_HASH = logit_threshold_grid_sha256(GRID)
MANIFEST_HASH = "a" * 64
FEATURE_HASH = feature_schema_sha256(
    LOGIT_STATISTICS_SCHEMA_VERSION, statistics_names=NAMES
)
OUTER_HASH = "1" * 64
INNER_HASHES = ("2" * 64, "3" * 64)
ALL_HASHES = (OUTER_HASH, *INNER_HASHES)
PIXEL_BUDGETS = [10.0, 1.0]
COMPONENT_BUDGETS = [100.0, 10.0]
TRAIN_ARCHIVE_HASH = "4" * 64
SEED = 42


def _episode_contract(validation_archive_sha256: str) -> dict[str, object]:
    return {
        "formal_protocol_eligible": True,
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": GRID_HASH,
        "threshold_grid_manifest_sha256": MANIFEST_HASH,
        "threshold_grid_detector_protocol": ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": list(ALL_HASHES),
        "threshold_grid_outer_detector_checkpoint_sha256": OUTER_HASH,
        "threshold_grid_episode_detector_checkpoint_sha256s": list(INNER_HASHES),
        "feature_schema_sha256": FEATURE_HASH,
        "adaptation_window": 1,
        "evaluation_window": 1,
        "stride": 2,
        "protocol_fields": {
            "pseudo_targets": ["NUDT-SIRST", "IRSTD-1K"],
            "threshold_grid_source_domains": ["irstd1k", "nudt"],
            "validation_domain": "IRSTD-1K",
        },
        "train": {"archive_sha256": TRAIN_ARCHIVE_HASH},
        "validation": {"archive_sha256": validation_archive_sha256},
    }


def _write_archive(path: Path) -> None:
    rows = 3
    statistics = np.asarray(
        [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]], dtype=np.float32
    )
    pixel_fp = np.asarray(
        [[100, 20, 2, 0], [80, 12, 1, 0], [120, 25, 3, 0]],
        dtype=np.int64,
    )
    component_fp = np.asarray(
        [[10, 3, 1, 0], [8, 2, 0, 0], [12, 4, 1, 0]],
        dtype=np.int64,
    )
    tp = np.asarray(
        [[5, 5, 4, 0], [4, 4, 3, 0], [6, 6, 5, 0]], dtype=np.int64
    )
    gt = np.asarray([5, 4, 6], dtype=np.int64)
    total_pixels = np.asarray([1_000_000] * rows, dtype=np.int64)
    pixel_log = np.log10(pixel_fp / total_pixels[:, None] + 1e-12).astype(
        np.float32
    )
    component_log = np.log10(
        component_fp / (total_pixels[:, None] / 1_000_000.0) + 1e-6
    ).astype(np.float32)
    component_upper = np.maximum.accumulate(component_log[:, ::-1], axis=1)[:, ::-1]
    pd = (tp / np.maximum(gt[:, None], 1)).astype(np.float32)
    provenance = {
        "protocol": "causal_adaptation_then_future_evaluation",
        "representation": LOGIT_REPRESENTATION,
        "adaptation_window": 1,
        "evaluation_window": 1,
        "stride": 2,
        "pseudo_targets": ["NUDT-SIRST", "IRSTD-1K"],
        "validation_domain": "IRSTD-1K",
        "archive_split": "validation",
        "threshold_grid_source_domains": ["irstd1k", "nudt"],
        "paired_lodo_validation_domains": ["IRSTD-1K", "NUDT-SIRST"],
        "pseudo_target_split": "train",
        "expected_split_role": "train",
        "threshold_grid_outer_target_key": "NUAA-SIRST",
        "threshold_grid_outer_target_excluded": True,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": GRID_HASH,
        "threshold_grid_manifest_sha256": MANIFEST_HASH,
        "threshold_grid_detector_protocol": ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": list(ALL_HASHES),
        "threshold_grid_outer_detector_checkpoint_sha256": OUTER_HASH,
        "threshold_grid_episode_detector_checkpoint_sha256s": list(INNER_HASHES),
        "fold_provenance_audits": [
            {
                "verified": True,
                "pseudo_target": "NUDT-SIRST",
                "detector_weight_sha256": INNER_HASHES[0],
            },
            {
                "verified": True,
                "pseudo_target": "IRSTD-1K",
                "detector_weight_sha256": INNER_HASHES[1],
            },
        ],
        "feature_schema_sha256": FEATURE_HASH,
        "fold_provenance_verified": True,
        "allow_unverified_fold_provenance": False,
        "allow_cross_episode_role_reuse": False,
        "cross_episode_role_reuse_detected": False,
        "cross_episode_role_reuse_ids": [],
        "formal_causal_contract_verified": True,
        "protocol_scope": "formal_causal",
        "statistics_sample_role": "adaptation_window_A_label_free",
        "risk_label_sample_role": "immediately_following_evaluation_window_E",
    }
    np.savez_compressed(
        path,
        statistics=statistics,
        statistics_names=np.asarray(NAMES),
        statistics_names_sha256=np.asarray(statistics_names_sha256(NAMES)),
        statistics_schema_version=np.asarray(LOGIT_STATISTICS_SCHEMA_VERSION),
        feature_schema_sha256=np.asarray(FEATURE_HASH),
        pixel_log_risk=pixel_log,
        component_log_risk=component_upper,
        component_log_risk_raw=component_log,
        component_log_risk_upper=component_upper,
        component_risk_schema_version=np.asarray(COMPONENT_RISK_SCHEMA_VERSION),
        component_log_risk_alias=np.asarray("component_log_risk_upper"),
        pd_curve=pd,
        thresholds=GRID,
        pixel_fp_counts=pixel_fp,
        component_fp_counts=component_fp,
        tp_object_counts=tp,
        gt_object_counts=gt,
        total_pixels=total_pixels,
        representation=np.asarray(LOGIT_REPRESENTATION),
        threshold_grid_schema_version=np.asarray(LOGIT_GRID_SCHEMA_VERSION),
        threshold_grid_sha256=np.asarray(GRID_HASH),
        threshold_grid_manifest_sha256=np.asarray(MANIFEST_HASH),
        threshold_grid_detector_protocol=np.asarray(
            ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL
        ),
        threshold_grid_detector_checkpoint_sha256s=np.asarray(ALL_HASHES),
        threshold_grid_outer_detector_checkpoint_sha256=np.asarray(OUTER_HASH),
        threshold_grid_episode_detector_checkpoint_sha256s=np.asarray(INNER_HASHES),
        episode_schema_version=np.asarray(LOGIT_EPISODE_SCHEMA_VERSION),
        adaptation_sizes=np.ones(rows, dtype=np.int64),
        evaluation_sizes=np.ones(rows, dtype=np.int64),
        adaptation_ids=np.asarray(
            [json.dumps([f"adapt-{index}"]) for index in range(rows)]
        ),
        evaluation_ids=np.asarray(
            [json.dumps([f"eval-{index}"]) for index in range(rows)]
        ),
        pseudo_targets=np.asarray(["IRSTD-1K"] * rows),
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )


def _write_checkpoints(
    risk_path: Path,
    direct_path: Path,
    *,
    validation_archive_sha256: str,
) -> None:
    episode_contract = _episode_contract(validation_archive_sha256)
    risk_model = RiskCurvePredictor(
        input_dim=2,
        num_thresholds=GRID.size,
        hidden_dim=8,
        dropout=0.0,
    )
    for parameter in risk_model.parameters():
        if parameter.ndim > 0:
            torch.nn.init.zeros_(parameter)
    risk_checkpoint = {
        "method_name": "risk_curve",
        "model_class": "RiskCurvePredictor",
        "model_architecture_version": RISK_CURVE_ARCHITECTURE_VERSION,
        "role": "proposed_method",
        "state_dict": risk_model.state_dict(),
        "model_config": risk_model.config(),
        "thresholds": GRID.tolist(),
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": GRID_HASH,
        "threshold_grid_manifest_sha256": MANIFEST_HASH,
        "threshold_grid_detector_protocol": ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": list(ALL_HASHES),
        "threshold_grid_outer_detector_checkpoint_sha256": OUTER_HASH,
        "threshold_grid_episode_detector_checkpoint_sha256s": list(INNER_HASHES),
        "feature_schema_sha256": FEATURE_HASH,
        "statistics_schema_version": LOGIT_STATISTICS_SCHEMA_VERSION,
        "statistics_names": list(NAMES),
        "statistics_names_sha256": statistics_names_sha256(NAMES),
        "statistics_mean": [1.0, 1.5],
        "statistics_std": [1.0, 1.0],
        "episode_contract": episode_contract,
        "seed": SEED,
    }
    torch.save(risk_checkpoint, risk_path)

    direct_model = MonotoneBudgetCalibrator(
        feature_dim=2,
        budget_grid=PIXEL_BUDGETS,
        hidden_dims=(8,),
        dropout=0.0,
        representation=LOGIT_REPRESENTATION,
        threshold_grid=GRID.tolist(),
        architecture_version=RC_DIRECT_ARCHITECTURE_VERSION,
    )
    direct_model.normalizer.fit(
        torch.asarray([[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]])
    )
    direct_checkpoint = {
        "checkpoint_schema_version": RC_DIRECT_CHECKPOINT_SCHEMA_VERSION,
        "format_version": 4,
        "kind": "calibrator",
        "method_name": "direct_threshold",
        "model_class": "MonotoneBudgetCalibrator",
        "role": "baseline",
        "representation": LOGIT_REPRESENTATION,
        "thresholds": torch.from_numpy(GRID.copy()),
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": GRID_HASH,
        "threshold_grid_manifest_sha256": MANIFEST_HASH,
        "threshold_grid_detector_protocol": ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": list(ALL_HASHES),
        "threshold_grid_outer_detector_checkpoint_sha256": OUTER_HASH,
        "threshold_grid_episode_detector_checkpoint_sha256s": list(INNER_HASHES),
        "statistics_schema_version": LOGIT_STATISTICS_SCHEMA_VERSION,
        "statistics_names": list(NAMES),
        "statistics_names_sha256": statistics_names_sha256(NAMES),
        "feature_schema_sha256": FEATURE_HASH,
        "statistics_mean": torch.asarray([1.0, 1.5]),
        "statistics_std": torch.asarray([1.0, 1.0]),
        "budget_schema_version": RC_DIRECT_BUDGET_SCHEMA_VERSION,
        "pixel_budgets": PIXEL_BUDGETS,
        "component_budgets": COMPONENT_BUDGETS,
        "model_architecture_version": RC_DIRECT_ARCHITECTURE_VERSION,
        "model_config": direct_model.export_config(),
        "state_dict": direct_model.state_dict(),
        "episode_contract": episode_contract,
        "seed": SEED,
        "target_label_policy": {
            "model_inputs": "adaptation_window_A_label_free_statistics_only",
            "supervision": "source_official_train_future_E_risk_only",
            "outer_target_labels_used_for_features": False,
            "outer_target_labels_used_for_checkpoint_selection": False,
        },
    }
    torch.save(direct_checkpoint, direct_path)


def _fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    archive = tmp_path / "validation.npz"
    risk = tmp_path / "risk.pt"
    direct = tmp_path / "direct.pt"
    _write_archive(archive)
    _write_checkpoints(
        risk,
        direct,
        validation_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    return archive, risk, direct


def test_fair_comparison_emits_counts_metrics_and_finite_actions(
    tmp_path: Path,
) -> None:
    archive, risk, direct = _fixtures(tmp_path)
    output = tmp_path / "comparison.json"
    evaluate_source_pseudo_target_comparison(
        episode_file=archive,
        risk_curve_checkpoint=risk,
        rc_direct_checkpoint=direct,
        output=output,
        pixel_budgets=PIXEL_BUDGETS,
        component_budgets=COMPONENT_BUDGETS,
        device="cpu",
        batch_size=2,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SOURCE_PSEUDO_TARGET_COMPARISON_SCHEMA_VERSION
    assert payload["labels_used_for_action_selection"] is False
    assert payload["outer_target_labels_used"] is False
    assert payload["formal_source_domains"] == ["IRSTD-1K", "NUDT-SIRST"]
    assert payload["excluded_outer_target"] == "NUAA-SIRST"
    assert payload["archive_split"] == "validation"
    assert payload["seed"] == SEED
    assert payload["train_archive_sha256"] == TRAIN_ARCHIVE_HASH
    assert payload["validation_archive_sha256"] == hashlib.sha256(
        archive.read_bytes()
    ).hexdigest()
    assert payload["episode_archive_sha256"] == payload[
        "validation_archive_sha256"
    ]
    assert payload["threshold_grid_sha256"] == GRID_HASH
    assert payload["threshold_grid_outer_detector_checkpoint_sha256"] == OUTER_HASH
    assert payload["threshold_grid_episode_detector_checkpoint_sha256s"] == list(
        INNER_HASHES
    )
    assert len(payload["budgets"]) == 2
    assert payload["gate"]["decision"] in {"GO", "HOLD"}
    for budget in payload["budgets"]:
        for method in ("risk_curve", "rc_direct"):
            metrics = budget["methods"][method]
            assert 0.0 <= metrics["pd"] <= 1.0
            assert metrics["pixel_risk"] >= 0.0
            assert metrics["component_risk"] >= 0.0
            assert 0.0 <= metrics["joint_violation_rate"] <= 1.0
            assert metrics["mean_relative_excess"] >= 0.0
            assert metrics["max_relative_excess"] >= 0.0
    finite = [
        action
        for episode in payload["per_episode"]
        for budget in episode["actions"]
        for action in budget["methods"].values()
        if not action["reject"]
    ]
    assert finite
    assert all(isinstance(action["threshold_index"], int) for action in finite)
    assert all("pixel_fp_count" in action for action in finite)


def test_comparison_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    archive, risk, direct = _fixtures(tmp_path)
    checkpoint = torch.load(direct, map_location="cpu", weights_only=True)
    checkpoint["threshold_grid_manifest_sha256"] = "b" * 64
    checkpoint["episode_contract"] = dict(checkpoint["episode_contract"])
    checkpoint["episode_contract"]["threshold_grid_manifest_sha256"] = "b" * 64
    torch.save(checkpoint, direct)
    with pytest.raises(ValueError, match="RC-Direct/archive.*manifest"):
        evaluate_source_pseudo_target_comparison(
            episode_file=archive,
            risk_curve_checkpoint=risk,
            rc_direct_checkpoint=direct,
            output=tmp_path / "must-not-exist.json",
            pixel_budgets=PIXEL_BUDGETS,
            component_budgets=COMPONENT_BUDGETS,
            device="cpu",
        )


def test_comparison_rejects_unregistered_joint_budget_pair(tmp_path: Path) -> None:
    archive, risk, direct = _fixtures(tmp_path)
    with pytest.raises(ValueError, match="registered RC-Direct"):
        evaluate_source_pseudo_target_comparison(
            episode_file=archive,
            risk_curve_checkpoint=risk,
            rc_direct_checkpoint=direct,
            output=tmp_path / "must-not-exist.json",
            pixel_budgets=[10.0, 0.5],
            component_budgets=[100.0, 10.0],
            device="cpu",
        )


@pytest.mark.parametrize(
    ("field", "expected_error"),
    [
        ("seed", "checkpoint seed mismatch"),
        ("train_archive_sha256", "train_archive_sha256 mismatch"),
        ("validation_archive_sha256", "validation_archive_sha256 mismatch"),
    ],
)
def test_comparison_rejects_unpaired_checkpoint_training_bindings(
    tmp_path: Path,
    field: str,
    expected_error: str,
) -> None:
    archive, risk, direct = _fixtures(tmp_path)
    checkpoint = torch.load(direct, map_location="cpu", weights_only=True)
    if field == "seed":
        checkpoint["seed"] = SEED + 1
    else:
        split = "train" if field.startswith("train") else "validation"
        checkpoint["episode_contract"] = dict(checkpoint["episode_contract"])
        checkpoint["episode_contract"][split] = dict(
            checkpoint["episode_contract"][split]
        )
        checkpoint["episode_contract"][split]["archive_sha256"] = "5" * 64
    torch.save(checkpoint, direct)

    with pytest.raises(ValueError, match=expected_error):
        evaluate_source_pseudo_target_comparison(
            episode_file=archive,
            risk_curve_checkpoint=risk,
            rc_direct_checkpoint=direct,
            output=tmp_path / "must-not-exist.json",
            pixel_budgets=PIXEL_BUDGETS,
            component_budgets=COMPONENT_BUDGETS,
            device="cpu",
        )


def test_comparison_rejects_episode_file_not_bound_as_validation_archive(
    tmp_path: Path,
) -> None:
    archive, risk, direct = _fixtures(tmp_path)
    for checkpoint_path in (risk, direct):
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        checkpoint["episode_contract"] = dict(checkpoint["episode_contract"])
        checkpoint["episode_contract"]["validation"] = dict(
            checkpoint["episode_contract"]["validation"]
        )
        checkpoint["episode_contract"]["validation"]["archive_sha256"] = "5" * 64
        torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="episode_file SHA-256"):
        evaluate_source_pseudo_target_comparison(
            episode_file=archive,
            risk_curve_checkpoint=risk,
            rc_direct_checkpoint=direct,
            output=tmp_path / "must-not-exist.json",
            pixel_budgets=PIXEL_BUDGETS,
            component_budgets=COMPONENT_BUDGETS,
            device="cpu",
        )
