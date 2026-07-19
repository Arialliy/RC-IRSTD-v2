import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from risk_curve.build_curve_episodes import (
    COMPONENT_RISK_SCHEMA_VERSION,
    EPISODE_SCHEMA_VERSION,
    LOGIT_EPISODE_SCHEMA_VERSION,
)
from risk_curve.curve_dataset import (
    load_curve_archive,
    validate_archive_compatibility,
)
from risk_curve.domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    STATISTICS_SCHEMA_VERSION,
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
    PROBABILITY_REPRESENTATION,
    logit_threshold_grid_sha256,
)
from risk_curve.threshold_grid import threshold_grid_sha256, threshold_grid_version
from risk_curve.train_curve_predictor import (
    validate_curve_checkpoint_contract,
    validate_training_episode_contract,
)


def _write_v4_archive(
    path: Path,
    *,
    split: str,
    thresholds: np.ndarray | None = None,
    names: tuple[str, ...] = ("logit-a", "logit-b"),
    episode_schema: str = LOGIT_EPISODE_SCHEMA_VERSION,
    grid_hash: str | None = None,
    feature_hash: str | None = None,
    manifest_hash: str = "a" * 64,
    detector_hashes: tuple[str, ...] = ("b" * 64, "c" * 64, "d" * 64),
    outer_detector_hash: str = "d" * 64,
    episode_detector_hashes: tuple[str, ...] = ("b" * 64, "c" * 64),
    detector_protocol: str = GRID_DETECTOR_PROTOCOL,
) -> None:
    grid = np.asarray(
        [-3.0, -0.25, 0.5, 4.0] if thresholds is None else thresholds,
        dtype=np.float32,
    )
    num_episodes = 2
    statistics = np.asarray([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    if len(names) != statistics.shape[1]:
        raise ValueError("test feature names must match the statistics dimension")
    pixel = np.tile(
        np.linspace(-1.0, -4.0, grid.size, dtype=np.float32),
        (num_episodes, 1),
    )
    component = np.tile(
        np.linspace(0.0, -3.0, grid.size, dtype=np.float32),
        (num_episodes, 1),
    )
    semantic_hash = grid_hash or logit_threshold_grid_sha256(grid)
    semantic_feature_hash = feature_hash or feature_schema_sha256(
        LOGIT_STATISTICS_SCHEMA_VERSION,
        statistics_names=names,
    )
    adaptation_rows = [[f"{split}-a-{index}"] for index in range(num_episodes)]
    evaluation_rows = [[f"{split}-e-{index}"] for index in range(num_episodes)]
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
        "validation_domain": "IRSTD-1K",
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": semantic_hash,
        "threshold_grid_manifest_sha256": manifest_hash,
        "threshold_grid_detector_protocol": detector_protocol,
        "threshold_grid_detector_checkpoint_sha256s": list(detector_hashes),
        "threshold_grid_outer_detector_checkpoint_sha256": outer_detector_hash,
        "threshold_grid_episode_detector_checkpoint_sha256s": list(
            episode_detector_hashes
        ),
        "threshold_grid_source_domains": ["NUDT-SIRST", "IRSTD-1K"],
        "feature_schema_sha256": semantic_feature_hash,
        "fold_provenance_verified": True,
        "allow_unverified_fold_provenance": False,
        "allow_cross_episode_role_reuse": False,
        "cross_episode_role_reuse_detected": False,
        "formal_causal_contract_verified": True,
        "protocol_scope": "formal_causal",
    }
    np.savez_compressed(
        path,
        statistics=statistics,
        statistics_names=np.asarray(names),
        statistics_names_sha256=np.asarray(statistics_names_sha256(names)),
        statistics_schema_version=np.asarray(LOGIT_STATISTICS_SCHEMA_VERSION),
        feature_schema_sha256=np.asarray(semantic_feature_hash),
        pixel_log_risk=pixel,
        component_log_risk=component,
        component_log_risk_raw=component,
        component_log_risk_upper=component,
        component_risk_schema_version=np.asarray(COMPONENT_RISK_SCHEMA_VERSION),
        component_log_risk_alias=np.asarray("component_log_risk_upper"),
        pd_curve=np.ones_like(pixel),
        thresholds=grid,
        representation=np.asarray(LOGIT_REPRESENTATION),
        threshold_grid_schema_version=np.asarray(LOGIT_GRID_SCHEMA_VERSION),
        threshold_grid_sha256=np.asarray(semantic_hash),
        threshold_grid_manifest_sha256=np.asarray(manifest_hash),
        threshold_grid_detector_protocol=np.asarray(detector_protocol),
        threshold_grid_detector_checkpoint_sha256s=np.asarray(
            detector_hashes,
            dtype=str,
        ),
        threshold_grid_outer_detector_checkpoint_sha256=np.asarray(
            outer_detector_hash
        ),
        threshold_grid_episode_detector_checkpoint_sha256s=np.asarray(
            episode_detector_hashes,
            dtype=str,
        ),
        episode_schema_version=np.asarray(episode_schema),
        adaptation_sizes=np.ones(num_episodes, dtype=np.int64),
        evaluation_sizes=np.ones(num_episodes, dtype=np.int64),
        adaptation_ids=np.asarray([json.dumps(row) for row in adaptation_rows]),
        evaluation_ids=np.asarray([json.dumps(row) for row in evaluation_rows]),
        pseudo_targets=np.asarray(["NUDT-SIRST"] * num_episodes),
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )


def _write_probability_archive(path: Path, thresholds: np.ndarray) -> None:
    grid = np.asarray(thresholds, dtype=np.float32)
    np.savez_compressed(
        path,
        statistics=np.asarray([[0.0, 1.0]], dtype=np.float32),
        statistics_names=np.asarray(["logit-a", "logit-b"]),
        statistics_schema_version=np.asarray(STATISTICS_SCHEMA_VERSION),
        pixel_log_risk=np.asarray([[-1.0, -2.0, -3.0]], dtype=np.float32),
        component_log_risk=np.asarray([[0.0, -1.0, -2.0]], dtype=np.float32),
        pd_curve=np.ones((1, 3), dtype=np.float32),
        thresholds=grid,
        representation=np.asarray(PROBABILITY_REPRESENTATION),
        threshold_grid_schema_version=np.asarray(threshold_grid_version(grid)),
        threshold_grid_sha256=np.asarray(threshold_grid_sha256(grid)),
    )


def test_v4_raw_logit_archive_contract_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "v4.npz"
    _write_v4_archive(path, split="train")
    archive = load_curve_archive(path)
    assert str(archive["representation"].item()) == LOGIT_REPRESENTATION
    assert str(archive["threshold_grid_schema_version"].item()) == (
        LOGIT_GRID_SCHEMA_VERSION
    )
    assert str(archive["threshold_grid_sha256"].item()) == (
        logit_threshold_grid_sha256(archive["thresholds"])
    )
    assert str(archive["feature_schema_sha256"].item()) == feature_schema_sha256(
        LOGIT_STATISTICS_SCHEMA_VERSION,
        statistics_names=archive["statistics_names"],
    )
    assert str(archive["threshold_grid_detector_protocol"].item()) == (
        GRID_DETECTOR_PROTOCOL
    )
    assert archive["threshold_grid_detector_checkpoint_sha256s"].tolist() == [
        "b" * 64,
        "c" * 64,
        "d" * 64,
    ]
    assert str(
        archive["threshold_grid_outer_detector_checkpoint_sha256"].item()
    ) == "d" * 64
    assert archive[
        "threshold_grid_episode_detector_checkpoint_sha256s"
    ].tolist() == [
        "b" * 64,
        "c" * 64,
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("threshold_grid_sha256", "b" * 64, "semantic grid"),
        ("feature_schema_sha256", "b" * 64, "ordered feature schema"),
        ("threshold_grid_manifest_sha256", "not-a-hash", "lowercase SHA-256"),
        (
            "threshold_grid_detector_protocol",
            "single_detector_checkpoint",
            "detector_protocol",
        ),
        (
            "threshold_grid_detector_checkpoint_sha256s",
            ["b" * 64, "b" * 64],
            "distinct checkpoint hashes",
        ),
        (
            "threshold_grid_outer_detector_checkpoint_sha256",
            "e" * 64,
            "outer detector plus all episode detector hashes",
        ),
    ],
)
def test_v4_archive_rejects_corrupt_semantic_contract(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    valid = tmp_path / "valid.npz"
    _write_v4_archive(valid, split="train")
    with np.load(valid, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload[field] = np.asarray(value)
    corrupt = tmp_path / f"bad-{field}.npz"
    np.savez_compressed(corrupt, **payload)
    with pytest.raises(ValueError, match=message):
        load_curve_archive(corrupt)


def test_representation_mismatch_fails_before_training(tmp_path: Path) -> None:
    grid = np.asarray([0.0, 0.2, 0.8], dtype=np.float32)
    raw_path = tmp_path / "raw.npz"
    probability_path = tmp_path / "probability.npz"
    _write_v4_archive(raw_path, split="raw", thresholds=grid)
    _write_probability_archive(probability_path, grid)
    raw = load_curve_archive(raw_path)
    probability = load_curve_archive(probability_path)
    with pytest.raises(ValueError, match="representations differ"):
        validate_archive_compatibility(raw, probability)


def test_v4_train_validation_detector_fold_contract_must_match(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.npz"
    validation_path = tmp_path / "validation.npz"
    _write_v4_archive(train_path, split="train")
    _write_v4_archive(
        validation_path,
        split="validation",
        detector_hashes=("e" * 64, "f" * 64, "0" * 64),
        outer_detector_hash="0" * 64,
        episode_detector_hashes=("e" * 64, "f" * 64),
    )
    with pytest.raises(ValueError, match="detector role/checkpoint hashes differ"):
        validate_archive_compatibility(
            load_curve_archive(train_path),
            load_curve_archive(validation_path),
        )


def test_v4_causal_contract_binds_detector_folds_to_provenance(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.npz"
    validation_path = tmp_path / "validation.npz"
    _write_v4_archive(train_path, split="train")
    _write_v4_archive(validation_path, split="validation")
    train = load_curve_archive(train_path)
    validation = load_curve_archive(validation_path)
    provenance = json.loads(str(validation["provenance_json"].item()))
    provenance["threshold_grid_detector_checkpoint_sha256s"] = [
        "e" * 64,
        "f" * 64,
        "0" * 64,
    ]
    validation["provenance_json"] = np.asarray(
        json.dumps(provenance, sort_keys=True)
    )
    with pytest.raises(ValueError, match="provenance grid detector checkpoint hashes"):
        validate_training_episode_contract(
            train,
            validation,
            train_path=train_path,
            validation_path=validation_path,
        )


def test_v4_causal_contract_requires_one_inner_detector_per_pseudo_target(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.npz"
    validation_path = tmp_path / "validation.npz"
    _write_v4_archive(train_path, split="train")
    _write_v4_archive(
        validation_path,
        split="validation",
        detector_hashes=("b" * 64, "d" * 64),
        outer_detector_hash="d" * 64,
        episode_detector_hashes=("b" * 64,),
    )
    train = load_curve_archive(train_path)
    validation = load_curve_archive(validation_path)
    with pytest.raises(ValueError, match="per source/pseudo-target domain"):
        validate_training_episode_contract(
            train,
            validation,
            train_path=train_path,
            validation_path=validation_path,
        )


def test_training_causal_contract_rejects_v2_v4_episode_schema_mix(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.npz"
    validation_path = tmp_path / "validation.npz"
    _write_v4_archive(train_path, split="train")
    _write_v4_archive(validation_path, split="validation")
    train = load_curve_archive(train_path)
    validation = load_curve_archive(validation_path)
    # Simulate an in-memory caller bypassing the archive loader.  The training
    # contract independently remains fail-closed.
    validation["episode_schema_version"] = np.asarray(EPISODE_SCHEMA_VERSION)
    with pytest.raises(ValueError, match="incompatible with representation"):
        validate_training_episode_contract(
            train,
            validation,
            train_path=train_path,
            validation_path=validation_path,
        )


def test_v4_training_checkpoint_contract_and_model_round_trip(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    train_path = tmp_path / "train.npz"
    validation_path = tmp_path / "validation.npz"
    checkpoint_path = tmp_path / "curve-v4.pt"
    _write_v4_archive(train_path, split="train")
    _write_v4_archive(validation_path, split="validation")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "risk_curve.train_curve_predictor",
            "--train-file",
            str(train_path),
            "--val-file",
            str(validation_path),
            "--output",
            str(checkpoint_path),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--hidden-dim",
            "4",
            "--dropout",
            "0",
            "--patience",
            "1",
            "--device",
            "cpu",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    audit = validate_curve_checkpoint_contract(checkpoint)
    assert audit["formal_v4_eligible"] is True
    assert checkpoint["representation"] == LOGIT_REPRESENTATION
    assert checkpoint["threshold_grid_schema_version"] == LOGIT_GRID_SCHEMA_VERSION
    assert checkpoint["threshold_grid_manifest_sha256"] == "a" * 64
    assert checkpoint["threshold_grid_detector_protocol"] == (
        GRID_DETECTOR_PROTOCOL
    )
    assert checkpoint["threshold_grid_detector_checkpoint_sha256s"] == [
        "b" * 64,
        "c" * 64,
        "d" * 64,
    ]
    assert checkpoint["threshold_grid_outer_detector_checkpoint_sha256"] == (
        "d" * 64
    )
    assert checkpoint["threshold_grid_episode_detector_checkpoint_sha256s"] == [
        "b" * 64,
        "c" * 64,
    ]
    assert audit["threshold_grid_detector_protocol"] == GRID_DETECTOR_PROTOCOL
    assert audit["threshold_grid_detector_checkpoint_sha256s"] == [
        "b" * 64,
        "c" * 64,
        "d" * 64,
    ]
    assert audit["threshold_grid_outer_detector_checkpoint_sha256"] == "d" * 64
    assert audit["threshold_grid_episode_detector_checkpoint_sha256s"] == [
        "b" * 64,
        "c" * 64,
    ]
    assert checkpoint["model_architecture_version"] == (
        RISK_CURVE_ARCHITECTURE_VERSION
    )
    assert checkpoint["model_config"]["architecture_version"] == (
        RISK_CURVE_ARCHITECTURE_VERSION
    )
    restored = RiskCurvePredictor(**checkpoint["model_config"])
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    output = restored(torch.zeros(1, 2))
    assert output["pixel_log_risk"].shape == (1, 4)
    assert torch.isfinite(output["pixel_log_risk"]).all()

    obsolete_architecture = dict(checkpoint)
    obsolete_architecture["model_architecture_version"] = (
        "controlled-total-drop-v1"
    )
    obsolete_architecture["model_config"] = dict(checkpoint["model_config"])
    obsolete_architecture["model_config"]["architecture_version"] = (
        "controlled-total-drop-v1"
    )
    with pytest.raises(ValueError, match="Unsupported curve model architecture"):
        validate_curve_checkpoint_contract(obsolete_architecture)

    missing_protocol = dict(checkpoint)
    missing_protocol.pop("threshold_grid_detector_protocol")
    with pytest.raises(ValueError, match="threshold_grid_detector_protocol"):
        validate_curve_checkpoint_contract(missing_protocol)

    mismatched_folds = dict(checkpoint)
    mismatched_folds["threshold_grid_detector_checkpoint_sha256s"] = [
        "e" * 64,
        "f" * 64,
        "0" * 64,
    ]
    mismatched_folds["threshold_grid_outer_detector_checkpoint_sha256"] = "0" * 64
    mismatched_folds["threshold_grid_episode_detector_checkpoint_sha256s"] = [
        "e" * 64,
        "f" * 64,
    ]
    with pytest.raises(ValueError, match="episode contract differ in"):
        validate_curve_checkpoint_contract(mismatched_folds)


def test_legacy_probability_checkpoint_is_diagnostic_only() -> None:
    thresholds = np.asarray([0.1, 0.5, 0.9], dtype=np.float32)
    legacy = {
        "thresholds": thresholds.tolist(),
        "threshold_grid_sha256": threshold_grid_sha256(thresholds),
        "statistics_schema_version": STATISTICS_SCHEMA_VERSION,
    }
    audit = validate_curve_checkpoint_contract(legacy)
    assert audit["legacy_checkpoint"] is True
    assert audit["diagnostic_compatible"] is True
    assert audit["formal_v4_eligible"] is False
    assert audit["representation"] == PROBABILITY_REPRESENTATION
