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
from risk_curve.build_curve_episodes import (
    COMPONENT_RISK_SCHEMA_VERSION,
    LOGIT_EPISODE_SCHEMA_VERSION,
)
from risk_curve.build_deployment_statistics import (
    LOGIT_DEPLOYMENT_STATISTICS_SCHEMA_VERSION,
)
from risk_curve.direct_calibrator import (
    ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL,
    derive_direct_threshold_targets,
    load_direct_training_pair,
    quantize_direct_logit_threshold,
    validate_direct_checkpoint_contract,
)
from risk_curve.domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    feature_schema_sha256,
    statistics_names_sha256,
)
from risk_curve.representation import (
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_REPRESENTATION,
    logit_threshold_grid_sha256,
)
from risk_curve.select_direct_threshold_v4 import select_direct_thresholds
from risk_curve.train_direct_calibrator_v4 import train_direct_calibrator


GRID = np.asarray([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
NAMES = ("logit_feature_a", "logit_feature_b")
GRID_HASH = logit_threshold_grid_sha256(GRID)
MANIFEST_HASH = "a" * 64
FEATURE_HASH = feature_schema_sha256(
    LOGIT_STATISTICS_SCHEMA_VERSION, statistics_names=NAMES
)
OUTER_DETECTOR_CHECKPOINT_SHA256 = "1" * 64
EPISODE_DETECTOR_CHECKPOINT_SHA256S = ("2" * 64, "3" * 64)
DETECTOR_CHECKPOINT_SHA256S = (
    OUTER_DETECTOR_CHECKPOINT_SHA256,
    *EPISODE_DETECTOR_CHECKPOINT_SHA256S,
)


def _write_archive(
    path: Path,
    *,
    split: str,
    outer: str = "nuaa",
    outer_detector_checkpoint_sha256: str = OUTER_DETECTOR_CHECKPOINT_SHA256,
    episode_detector_checkpoint_sha256s: tuple[str, str] = (
        EPISODE_DETECTOR_CHECKPOINT_SHA256S
    ),
) -> None:
    detector_checkpoint_sha256s = (
        outer_detector_checkpoint_sha256,
        *episode_detector_checkpoint_sha256s,
    )
    num_rows = 4 if split == "train" else 2
    statistics = np.stack(
        [np.asarray([index, index + 0.5], dtype=np.float32) for index in range(num_rows)]
    )
    pixel = np.tile(
        np.asarray([-1.0, -2.0, -3.0, -4.0], dtype=np.float32),
        (num_rows, 1),
    )
    component = np.tile(
        np.asarray([0.5, 0.0, -1.0, -2.0], dtype=np.float32),
        (num_rows, 1),
    )
    adaptation_ids = [[f"{split}-a-{index}"] for index in range(num_rows)]
    evaluation_ids = [[f"{split}-e-{index}"] for index in range(num_rows)]
    provenance = {
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
        "pseudo_target_split": "train",
        "expected_split_role": "train",
        "validation_domain": "IRSTD-1K",
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": GRID_HASH,
        "threshold_grid_manifest_sha256": MANIFEST_HASH,
        "threshold_grid_outer_target_key": outer,
        "threshold_grid_outer_target_excluded": True,
        "threshold_grid_detector_protocol": ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": list(
            detector_checkpoint_sha256s
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": (
            outer_detector_checkpoint_sha256
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": list(
            episode_detector_checkpoint_sha256s
        ),
        "threshold_grid_source_domains": ["nudt", "irstd1k"],
        "fold_provenance_audits": [
            {
                "verified": True,
                "pseudo_target": "NUDT-SIRST",
                "detector_weight_sha256": (
                    episode_detector_checkpoint_sha256s[0]
                ),
            },
            {
                "verified": True,
                "pseudo_target": "IRSTD-1K",
                "detector_weight_sha256": (
                    episode_detector_checkpoint_sha256s[1]
                ),
            },
        ],
        "feature_schema_sha256": FEATURE_HASH,
        "fold_provenance_verified": True,
        "allow_unverified_fold_provenance": False,
        "allow_cross_episode_role_reuse": False,
        "cross_episode_role_reuse_detected": False,
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
        pixel_log_risk=pixel,
        component_log_risk=component,
        component_log_risk_raw=component,
        component_log_risk_upper=component,
        component_risk_schema_version=np.asarray(COMPONENT_RISK_SCHEMA_VERSION),
        component_log_risk_alias=np.asarray("component_log_risk_upper"),
        pd_curve=np.ones_like(pixel),
        thresholds=GRID,
        representation=np.asarray(LOGIT_REPRESENTATION),
        threshold_grid_schema_version=np.asarray(LOGIT_GRID_SCHEMA_VERSION),
        threshold_grid_sha256=np.asarray(GRID_HASH),
        threshold_grid_manifest_sha256=np.asarray(MANIFEST_HASH),
        threshold_grid_detector_protocol=np.asarray(
            ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL
        ),
        threshold_grid_detector_checkpoint_sha256s=np.asarray(
            detector_checkpoint_sha256s
        ),
        threshold_grid_outer_detector_checkpoint_sha256=np.asarray(
            outer_detector_checkpoint_sha256
        ),
        threshold_grid_episode_detector_checkpoint_sha256s=np.asarray(
            episode_detector_checkpoint_sha256s
        ),
        episode_schema_version=np.asarray(LOGIT_EPISODE_SCHEMA_VERSION),
        adaptation_sizes=np.ones(num_rows, dtype=np.int64),
        evaluation_sizes=np.ones(num_rows, dtype=np.int64),
        adaptation_ids=np.asarray([json.dumps(row) for row in adaptation_ids]),
        evaluation_ids=np.asarray([json.dumps(row) for row in evaluation_ids]),
        pseudo_targets=np.asarray(
            ["NUDT-SIRST" if split == "train" else "IRSTD-1K"] * num_rows
        ),
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )


def _write_deployment_statistics(
    path: Path,
    *,
    score_detector_checkpoint_sha256: str = OUTER_DETECTOR_CHECKPOINT_SHA256,
) -> None:
    provenance = {
        "masks_read": False,
        "adaptation_window": 1,
        "evaluation_window": 1,
        "stride": 2,
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": GRID_HASH,
        "threshold_grid_manifest_sha256": MANIFEST_HASH,
        "feature_schema_sha256": FEATURE_HASH,
        "source_reference_sha256": None,
        "source_reference_statistics_names_sha256": None,
        "threshold_grid_detector_protocol": ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": list(
            DETECTOR_CHECKPOINT_SHA256S
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": (
            OUTER_DETECTOR_CHECKPOINT_SHA256
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": list(
            EPISODE_DETECTOR_CHECKPOINT_SHA256S
        ),
    }
    protocol = {
        "detector_weight_sha256": score_detector_checkpoint_sha256,
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": GRID_HASH,
        "threshold_grid_detector_protocol": ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": list(
            DETECTOR_CHECKPOINT_SHA256S
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": (
            OUTER_DETECTOR_CHECKPOINT_SHA256
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": list(
            EPISODE_DETECTOR_CHECKPOINT_SHA256S
        ),
    }
    protocol_text = json.dumps(protocol, sort_keys=True)
    protocol_fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    np.savez_compressed(
        path,
        statistics=np.asarray([[0.5, 1.0]], dtype=np.float32),
        statistics_names=np.asarray(NAMES),
        statistics_names_sha256=np.asarray(statistics_names_sha256(NAMES)),
        statistics_schema_version=np.asarray(LOGIT_STATISTICS_SCHEMA_VERSION),
        feature_schema_sha256=np.asarray(FEATURE_HASH),
        representation=np.asarray(LOGIT_REPRESENTATION),
        thresholds=GRID,
        threshold_grid_schema_version=np.asarray(LOGIT_GRID_SCHEMA_VERSION),
        threshold_grid_sha256=np.asarray(GRID_HASH),
        threshold_grid_manifest_sha256=np.asarray(MANIFEST_HASH),
        threshold_grid_detector_protocol=np.asarray(
            ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL
        ),
        threshold_grid_detector_checkpoint_sha256s=np.asarray(
            DETECTOR_CHECKPOINT_SHA256S
        ),
        threshold_grid_outer_detector_checkpoint_sha256=np.asarray(
            OUTER_DETECTOR_CHECKPOINT_SHA256
        ),
        threshold_grid_episode_detector_checkpoint_sha256s=np.asarray(
            EPISODE_DETECTOR_CHECKPOINT_SHA256S
        ),
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
        protocol_json=np.asarray(protocol_text),
        protocol_fingerprint=np.asarray(protocol_fingerprint),
        deployment_statistics_schema_version=np.asarray(
            LOGIT_DEPLOYMENT_STATISTICS_SCHEMA_VERSION
        ),
        block_ids=np.asarray(["block-0"]),
    )


def test_raw_logit_calibrator_round_trip_and_probability_compatibility() -> None:
    raw = MonotoneBudgetCalibrator(
        feature_dim=2,
        budget_grid=[1e-1, 1e-3],
        hidden_dims=(8,),
        dropout=0.0,
        representation=LOGIT_REPRESENTATION,
        threshold_grid=GRID.tolist(),
    )
    assert raw.architecture_version == RC_DIRECT_ARCHITECTURE_VERSION
    features = torch.asarray([[0.0, 1.0], [1.0, 2.0]])
    raw.normalizer.fit(features)
    output = raw(features)
    assert output.representation == LOGIT_REPRESENTATION
    assert torch.equal(output.grid_thresholds, output.grid_logits)
    assert torch.all(output.grid_logits[:, 1:] >= output.grid_logits[:, :-1])
    restored = MonotoneBudgetCalibrator(**raw.export_config())
    restored.load_state_dict(raw.state_dict())
    assert restored.representation == LOGIT_REPRESENTATION

    probability = MonotoneBudgetCalibrator(
        feature_dim=2,
        budget_grid=[1e-1, 1e-3],
        hidden_dims=(8,),
        dropout=0.0,
    )
    probability.normalizer.fit(features)
    assert torch.all((probability(features).grid_thresholds >= 0.0))
    assert probability.threshold_grid is None


def test_joint_targets_and_external_reject_action() -> None:
    pixel = np.asarray([[-1.0, -2.0, -3.0, -4.0]], dtype=np.float32)
    component = np.asarray([[0.5, 0.0, -1.0, -1.5]], dtype=np.float32)
    targets = derive_direct_threshold_targets(
        pixel,
        component,
        GRID,
        [1e-1, 1e-3],
        [1.0, 1e-3],
    )
    assert targets.indices.tolist() == [[1, 4]]
    assert targets.logits[0, 0] == GRID[1]
    assert targets.logits[0, 1] > GRID[-1]
    finite = quantize_direct_logit_threshold(-1.5, GRID)
    reject = quantize_direct_logit_threshold(targets.reject_code, GRID)
    assert (finite.selected_logit_threshold, finite.threshold_index, finite.reject) == (
        -1.0,
        1,
        False,
    )
    assert reject.selected_logit_threshold == float("inf")
    assert reject.threshold_index is None and reject.reject is True


def test_source_only_fairness_contract_fails_closed(tmp_path: Path) -> None:
    train = tmp_path / "train.npz"
    validation = tmp_path / "validation.npz"
    _write_archive(train, split="train")
    _write_archive(validation, split="validation")
    pair = load_direct_training_pair(train, validation)
    assert pair.statistics_names == NAMES
    assert pair.episode_contract["formal_protocol_eligible"] is True
    assert pair.episode_contract["representation"] == LOGIT_REPRESENTATION
    assert pair.outer_detector_checkpoint_sha256 == (
        OUTER_DETECTOR_CHECKPOINT_SHA256
    )
    assert pair.episode_detector_checkpoint_sha256s == (
        EPISODE_DETECTOR_CHECKPOINT_SHA256S
    )

    leaked = tmp_path / "leaked.npz"
    _write_archive(leaked, split="validation", outer="NUDT-SIRST")
    with pytest.raises(ValueError, match="include the excluded outer target"):
        load_direct_training_pair(train, leaked)

    mismatched_outer = tmp_path / "mismatched-outer.npz"
    _write_archive(
        mismatched_outer,
        split="validation",
        outer_detector_checkpoint_sha256="4" * 64,
    )
    with pytest.raises(ValueError, match="detector role/checkpoint"):
        load_direct_training_pair(train, mismatched_outer)


def test_checkpoint_and_selection_round_trip_on_same_v4_contract(tmp_path: Path) -> None:
    train = tmp_path / "train.npz"
    validation = tmp_path / "validation.npz"
    checkpoint_path = tmp_path / "direct.pt"
    statistics_path = tmp_path / "deployment.npz"
    selection_path = tmp_path / "selection.json"
    _write_archive(train, split="train")
    _write_archive(validation, split="validation")
    _write_deployment_statistics(statistics_path)
    train_direct_calibrator(
        train_file=train,
        validation_file=validation,
        output=checkpoint_path,
        pixel_budgets=[1e-1, 1e-3],
        component_budgets=[1.0, 1e-2],
        hidden_dims=(8,),
        dropout=0.0,
        epochs=1,
        batch_size=2,
        patience=1,
        num_workers=0,
        seed=7,
        device="cpu",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    contract = validate_direct_checkpoint_contract(checkpoint)
    assert contract["formal_v4_eligible"] is True
    assert tuple(contract["threshold_grid_detector_checkpoint_sha256s"]) == (
        DETECTOR_CHECKPOINT_SHA256S
    )
    assert contract["threshold_grid_outer_detector_checkpoint_sha256"] == (
        OUTER_DETECTOR_CHECKPOINT_SHA256
    )
    assert tuple(
        contract["threshold_grid_episode_detector_checkpoint_sha256s"]
    ) == EPISODE_DETECTOR_CHECKPOINT_SHA256S
    assert checkpoint["target_label_policy"][
        "outer_target_labels_used_for_features"
    ] is False
    assert checkpoint["target_label_policy"][
        "outer_target_labels_used_for_checkpoint_selection"
    ] is False
    select_direct_thresholds(
        checkpoint_path=checkpoint_path,
        statistics_file=statistics_path,
        output=selection_path,
        pixel_budget=1e-1,
        component_budget=1.0,
        device="cpu",
    )
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    assert payload["representation"] == LOGIT_REPRESENTATION
    assert payload["threshold_grid_sha256"] == GRID_HASH
    assert payload["threshold_grid_detector_protocol"] == (
        ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL
    )
    assert payload["threshold_grid_outer_detector_checkpoint_sha256"] == (
        OUTER_DETECTOR_CHECKPOINT_SHA256
    )
    assert payload["threshold_grid_episode_detector_checkpoint_sha256s"] == list(
        EPISODE_DETECTOR_CHECKPOINT_SHA256S
    )
    assert payload["deployment_score_detector_checkpoint_sha256"] == (
        OUTER_DETECTOR_CHECKPOINT_SHA256
    )
    assert payload["target_labels_used"] is False
    assert payload["records"][0]["threshold_index"] in {0, 1, 2, 3, None}
    if payload["records"][0]["reject"]:
        assert payload["records"][0]["selected_logit_threshold"] == "+inf"
    else:
        assert isinstance(
            payload["records"][0]["selected_logit_threshold"], float
        )

    tampered = dict(checkpoint)
    tampered["feature_schema_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="feature-schema"):
        validate_direct_checkpoint_contract(tampered)

    bad_roles = dict(checkpoint)
    bad_roles["threshold_grid_episode_detector_checkpoint_sha256s"] = [
        OUTER_DETECTOR_CHECKPOINT_SHA256,
        EPISODE_DETECTOR_CHECKPOINT_SHA256S[0],
    ]
    with pytest.raises(ValueError, match="Outer detector cannot supervise"):
        validate_direct_checkpoint_contract(bad_roles)

    wrong_detector_statistics = tmp_path / "wrong-detector-deployment.npz"
    _write_deployment_statistics(
        wrong_detector_statistics,
        score_detector_checkpoint_sha256="4" * 64,
    )
    with pytest.raises(ValueError, match="must equal the frozen outer detector"):
        select_direct_thresholds(
            checkpoint_path=checkpoint_path,
            statistics_file=wrong_detector_statistics,
            output=tmp_path / "must-not-exist.json",
            pixel_budget=1e-1,
            component_budget=1.0,
            device="cpu",
        )
