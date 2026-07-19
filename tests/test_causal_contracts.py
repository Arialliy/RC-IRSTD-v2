import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from data_ext.mask_alignment import MASK_ALIGNMENT_POLICY
from evaluation.artifact_integrity import (
    SCORE_MANIFEST_SCHEMA_VERSION,
    SCORE_MASK_ALIGNMENT_SCHEMA,
    SCORE_RECORD_INTEGRITY_SCHEMA,
    file_sha256,
    ordered_ids_sha256,
    score_records_sha256,
)
from risk_curve.build_curve_episodes import (
    EPISODE_SCHEMA_VERSION,
    _resolve_window_config,
    main as build_curve_episodes_main,
)
from risk_curve.build_deployment_statistics import build_deployment_statistics
from risk_curve.curve_dataset import load_curve_archive
from risk_curve.domain_statistics import (
    STATISTICS_SCHEMA_VERSION,
    statistics_names_sha256,
)
from risk_curve.threshold_grid import threshold_grid_sha256
from risk_curve.train_curve_predictor import validate_training_episode_contract


def _write_score_directory(root: Path, target: str, count: int = 6) -> None:
    root.mkdir()
    records: list[dict[str, object]] = []
    image_ids: list[str] = []
    for index in range(count):
        image_id = f"{target}-image-{index}"
        filename = f"{index:03d}.npz"
        probability = np.linspace(0.0, 0.9, 64, dtype=np.float32).reshape(8, 8)
        probability = np.roll(probability, index, axis=1)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[3, 3] = 1
        np.savez_compressed(
            root / filename,
            prob=probability,
            gray=np.zeros_like(probability),
            mask=mask,
            image_id=np.asarray(image_id),
            labels_loaded=np.asarray(True),
            original_hw=np.asarray([8, 8], dtype=np.int32),
            mask_alignment_applied=np.asarray(False),
            mask_original_hw=np.asarray([8, 8], dtype=np.int32),
            mask_aspect_relative_error=np.asarray(0.0, dtype=np.float64),
            mask_alignment_policy=np.asarray(MASK_ALIGNMENT_POLICY),
        )
        records.append(
            {
                "file": filename,
                "image_id": image_id,
                "shape": [8, 8],
                "sha256": file_sha256(root / filename),
                "mask_alignment_applied": False,
                "mask_original_hw": [8, 8],
                "mask_aspect_relative_error": 0.0,
            }
        )
        image_ids.append(image_id)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "score_type": "sigmoid_probability",
                "schema_version": SCORE_MANIFEST_SCHEMA_VERSION,
                "record_integrity_schema": SCORE_RECORD_INTEGRITY_SCHEMA,
                "warm_flag": True,
                "labels_loaded": True,
                "spatial_mode": "native",
                "pad_multiple": 16,
                "target_dataset": target,
                "source_datasets": ["independent-source"],
                "weight_sha256": "a" * 64,
                # Curve-episode construction now independently verifies the
                # leakage-free source-train artifact contract.  Keep this
                # shared synthetic manifest representative of a formal export.
                "split_role": "train",
                "split_authority_verified": True,
                "checkpoint_diagnostic_only": False,
                "non_strict_state_loading": False,
                "num_images": len(records),
                "records": records,
                "records_sha256": score_records_sha256(records),
                "ordered_image_ids_sha256": ordered_ids_sha256(image_ids),
                "mask_alignment_schema": SCORE_MASK_ALIGNMENT_SCHEMA,
                "mask_alignment_policy": MASK_ALIGNMENT_POLICY,
                "mask_alignment_count": 0,
                "mask_aligned_sample_ids": [],
            }
        ),
        encoding="utf-8",
    )


def test_formal_episode_stride_gate_and_diagnostic_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    formal_args = argparse.Namespace(
        adaptation_window=2,
        evaluation_window=1,
        window_size=None,
        stride=2,
        allow_cross_episode_role_reuse=False,
    )
    with pytest.raises(ValueError, match=r"stride >= A\+E"):
        _resolve_window_config(formal_args)

    diagnostic_args = argparse.Namespace(
        adaptation_window=2,
        evaluation_window=1,
        window_size=None,
        stride=2,
        allow_cross_episode_role_reuse=True,
    )
    _resolve_window_config(diagnostic_args)

    train_scores = tmp_path / "train-scores"
    validation_scores = tmp_path / "validation-scores"
    _write_score_directory(train_scores, "train-domain", count=4)
    _write_score_directory(validation_scores, "validation-domain", count=4)
    output = tmp_path / "episodes"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_curve_episodes",
            "--score-map-dir",
            str(train_scores),
            "--pseudo-target",
            "train-domain",
            "--score-map-dir",
            str(validation_scores),
            "--pseudo-target",
            "validation-domain",
            "--validation-domain",
            "validation-domain",
            "--adaptation-window",
            "1",
            "--evaluation-window",
            "1",
            "--stride",
            "1",
            "--allow-cross-episode-role-reuse",
            "--output-dir",
            str(output),
        ],
    )
    build_curve_episodes_main()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete_diagnostic_only"
    assert manifest["formal_protocol_eligible"] is False
    assert manifest["allow_cross_episode_role_reuse"] is True
    assert manifest["cross_episode_role_reuse_detected"] is True
    assert manifest["cross_episode_role_reuse_ids"]

    formal_output = tmp_path / "formal-episodes"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_curve_episodes",
            "--score-map-dir",
            str(train_scores),
            "--pseudo-target",
            "train-domain",
            "--score-map-dir",
            str(validation_scores),
            "--pseudo-target",
            "validation-domain",
            "--validation-domain",
            "validation-domain",
            "--adaptation-window",
            "1",
            "--evaluation-window",
            "1",
            "--stride",
            "2",
            "--output-dir",
            str(formal_output),
        ],
    )
    build_curve_episodes_main()
    formal_manifest = json.loads(
        (formal_output / "manifest.json").read_text(encoding="utf-8")
    )
    assert formal_manifest["status"] == "complete"
    assert formal_manifest["formal_protocol_eligible"] is True
    formal_contract = validate_training_episode_contract(
        load_curve_archive(formal_output / "train.npz"),
        load_curve_archive(formal_output / "val.npz"),
        train_path=formal_output / "train.npz",
        validation_path=formal_output / "val.npz",
    )
    assert formal_contract["formal_protocol_eligible"] is True
    assert formal_contract["adaptation_window"] == 1
    assert formal_contract["evaluation_window"] == 1


def test_deployment_statistics_store_stable_block_associations(tmp_path: Path):
    score_dir = tmp_path / "scores"
    _write_score_directory(score_dir, "target-domain", count=6)
    thresholds = np.asarray([0.0, 0.5, 0.9], dtype=np.float32)

    first = build_deployment_statistics(
        score_dir,
        thresholds,
        adaptation_window=2,
        evaluation_window=1,
        stride=3,
    )
    second = build_deployment_statistics(
        score_dir,
        thresholds,
        adaptation_window=2,
        evaluation_window=1,
        stride=3,
    )
    np.testing.assert_array_equal(first["block_ids"], second["block_ids"])
    assert bool(np.asarray(first["one_to_one_evaluation"]).item()) is True
    assert str(np.asarray(first["exchangeability_unit"]).item()) == "causal_A_to_E_block"
    records = [json.loads(str(value)) for value in np.asarray(first["block_records_json"])]
    adaptation_rows = [json.loads(str(value)) for value in np.asarray(first["adaptation_ids"])]
    evaluation_rows = [json.loads(str(value)) for value in np.asarray(first["evaluation_ids"])]
    assert len(records) == 2
    for index, record in enumerate(records):
        assert record["block_id"] == np.asarray(first["block_ids"])[index]
        assert record["adaptation_ids"] == adaptation_rows[index]
        assert record["evaluation_ids"] == evaluation_rows[index]
        assert set(record["adaptation_ids"]).isdisjoint(record["evaluation_ids"])

    grouped = build_deployment_statistics(
        score_dir,
        thresholds,
        adaptation_window=1,
        evaluation_window=2,
        stride=3,
    )
    assert bool(np.asarray(grouped["one_to_one_evaluation"]).item()) is False


def _write_curve_archive(
    path: Path,
    *,
    split: str,
    adaptation_window: int = 2,
    evaluation_window: int = 1,
) -> None:
    num_episodes = 4
    thresholds = np.asarray([0.0, 0.5, 0.9], dtype=np.float32)
    names = ("feature-a", "feature-b")
    statistics = np.arange(num_episodes * 2, dtype=np.float32).reshape(num_episodes, 2)
    pixel = np.tile(np.asarray([-1.0, -2.0, -3.0], dtype=np.float32), (num_episodes, 1))
    component = np.tile(
        np.asarray([1.0, 0.0, -1.0], dtype=np.float32), (num_episodes, 1)
    )
    adaptation_rows = [
        [f"{split}-a-{episode}-{index}" for index in range(adaptation_window)]
        for episode in range(num_episodes)
    ]
    evaluation_rows = [
        [f"{split}-e-{episode}-{index}" for index in range(evaluation_window)]
        for episode in range(num_episodes)
    ]
    provenance = {
        "protocol": "causal_adaptation_then_future_evaluation",
        "adaptation_window": adaptation_window,
        "evaluation_window": evaluation_window,
        "stride": adaptation_window + evaluation_window,
        "matching_rule": "overlap",
        "centroid_distance": 3.0,
        "connectivity": 2,
        "min_component_area": 1,
        "source_reference": None,
        "pseudo_targets": ["source-a", "source-b"],
        "validation_domain": "source-b",
        "threshold_grid_sha256": threshold_grid_sha256(thresholds),
        "fold_provenance_verified": True,
        "allow_unverified_fold_provenance": False,
        "allow_cross_episode_role_reuse": False,
        "cross_episode_role_reuse_detected": False,
        "formal_causal_contract_verified": True,
    }
    np.savez_compressed(
        path,
        statistics=statistics,
        statistics_names=np.asarray(names),
        statistics_names_sha256=np.asarray(statistics_names_sha256(names)),
        statistics_schema_version=np.asarray(STATISTICS_SCHEMA_VERSION),
        pixel_log_risk=pixel,
        component_log_risk=component,
        pd_curve=np.ones_like(pixel),
        thresholds=thresholds,
        episode_schema_version=np.asarray(EPISODE_SCHEMA_VERSION),
        adaptation_sizes=np.full(num_episodes, adaptation_window, dtype=np.int64),
        evaluation_sizes=np.full(num_episodes, evaluation_window, dtype=np.int64),
        adaptation_ids=np.asarray([json.dumps(row) for row in adaptation_rows]),
        evaluation_ids=np.asarray([json.dumps(row) for row in evaluation_rows]),
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )


def test_training_checkpoint_persists_and_validates_causal_contract(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    train_path = tmp_path / "train.npz"
    validation_path = tmp_path / "validation.npz"
    _write_curve_archive(train_path, split="train")
    _write_curve_archive(validation_path, split="validation")
    output = tmp_path / "curve.pt"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "risk_curve.train_curve_predictor",
            "--train-file",
            str(train_path),
            "--val-file",
            str(validation_path),
            "--output",
            str(output),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--hidden-dim",
            "4",
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
    assert '"per_threshold_metrics_recorded": true' in completed.stdout
    assert "coverage_by_threshold" not in completed.stdout
    checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    contract = checkpoint["episode_contract"]
    assert contract["verified"] is True
    assert contract["formal_protocol_eligible"] is True
    assert contract["adaptation_window"] == 2
    assert contract["evaluation_window"] == 1
    assert contract["risk_target_unit"] == "aggregate_risk_over_1_future_images"
    assert contract["one_to_one_future_target"] is True
    metrics = json.loads(
        output.with_suffix(output.suffix + ".metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["episode_contract"]["train"]["archive_sha256"] == contract["train"][
        "archive_sha256"
    ]

    mismatched_path = tmp_path / "validation-e2.npz"
    _write_curve_archive(
        mismatched_path,
        split="validation-e2",
        evaluation_window=2,
    )
    with pytest.raises(ValueError, match="differ in evaluation_window"):
        validate_training_episode_contract(
            load_curve_archive(train_path),
            load_curve_archive(mismatched_path),
            train_path=train_path,
            validation_path=mismatched_path,
        )


def test_training_contract_rejects_train_validation_image_id_overlap(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.npz"
    validation_path = tmp_path / "validation.npz"
    _write_curve_archive(train_path, split="train")
    _write_curve_archive(validation_path, split="validation")
    train = load_curve_archive(train_path)
    validation = load_curve_archive(validation_path)
    validation["adaptation_ids"] = np.asarray(
        validation["adaptation_ids"]
    ).copy()
    validation["adaptation_ids"][0] = train["adaptation_ids"][0]
    with pytest.raises(ValueError, match="reuse image IDs"):
        validate_training_episode_contract(
            train,
            validation,
            train_path=train_path,
            validation_path=validation_path,
        )
