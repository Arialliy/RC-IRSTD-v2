from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from data_ext.eval_dataset import IRSTDEvalDataset
from evaluation.export_score_maps import export_dataset_score_maps
from risk_curve.build_deployment_statistics import (
    LOGIT_DEPLOYMENT_STATISTICS_SCHEMA_VERSION,
    build_deployment_statistics,
)
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
    empty_action_contract,
    logit_threshold_grid_sha256,
)
from risk_curve.select_zero_label_threshold import (
    _score_map_protocol_provenance,
    _statistics_from_archive,
    apply_selected_threshold,
    main as select_main,
    select_dual_budget_threshold,
)


OUTER_DETECTOR_SHA256 = "a" * 64
EPISODE_DETECTOR_SHA256S = ["c" * 64, "d" * 64]
ALL_DETECTOR_SHA256S = [
    OUTER_DETECTOR_SHA256,
    *EPISODE_DETECTOR_SHA256S,
]


class _ConstantLogitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.probe = torch.nn.Conv2d(3, 1, kernel_size=1, bias=False)
        torch.nn.init.zeros_(self.probe.weight)

    def forward(self, images: torch.Tensor, warm_flag: bool) -> torch.Tensor:
        del warm_flag
        logits = self.probe(images)
        logits[..., 0, 0] = 4.0
        logits[..., 1:, 1:] = -3.0
        return logits


def _raw_score_directory(
    tmp_path: Path,
    count: int = 3,
    *,
    detector_sha256: str = OUTER_DETECTOR_SHA256,
) -> Path:
    dataset_root = tmp_path / "target"
    images = dataset_root / "images"
    images.mkdir(parents=True)
    ids = [f"sample-{index}" for index in range(count)]
    for index, image_id in enumerate(ids):
        array = np.full((4, 4, 3), index * 10, dtype=np.uint8)
        Image.fromarray(array).save(images / f"{image_id}.png")
    (dataset_root / "train.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    dataset = IRSTDEvalDataset(
        dataset_root,
        split="train",
        spatial_mode="native",
        pad_multiple=4,
        dataset_name="target-domain",
        load_masks=False,
    )
    output = tmp_path / "raw-scores"
    export_dataset_score_maps(
        _ConstantLogitModel(),
        dataset,
        output,
        labels_loaded=False,
        export_raw_logits=True,
        manifest_metadata={
            "weight_sha256": detector_sha256,
            "source_datasets": ["source-a", "source-b"],
            "checkpoint_selection_rule": "fixed_last",
            "checkpoint_diagnostic_only": False,
            "diagnostic_only": False,
            "non_strict_state_loading": False,
            "formal_protocol_eligible": True,
            "model_backend": "canonical",
        },
    )
    return output


def test_raw_logit_selector_has_external_empty_action_and_nonreject_action() -> None:
    grid = np.asarray([-2.0, 0.0, 3.0], dtype=np.float32)
    pixel = np.asarray([-1.0, -4.0, -8.0], dtype=np.float64)
    component = np.asarray([1.0, -1.0, -4.0], dtype=np.float64)

    threshold, reject, index = select_dual_budget_threshold(
        grid,
        pixel,
        component,
        1e-3,
        1.0,
        representation=LOGIT_REPRESENTATION,
    )
    assert (threshold, reject, index) == (0.0, False, 1)
    prediction = apply_selected_threshold(
        np.asarray([[-1.0, 0.0, 2.0]], dtype=np.float32),
        threshold,
        representation=LOGIT_REPRESENTATION,
    )
    assert prediction.tolist() == [[False, True, True]]

    empty_threshold, reject, index = select_dual_budget_threshold(
        grid,
        pixel,
        component,
        1e-20,
        1e-20,
        representation=LOGIT_REPRESENTATION,
    )
    assert reject is True and index is None and np.isposinf(empty_threshold)
    assert empty_action_contract()["included_in_model_grid"] is False
    assert not apply_selected_threshold(
        np.asarray([[1e6, -1e6]], dtype=np.float32),
        empty_threshold,
        representation=LOGIT_REPRESENTATION,
    ).any()


def test_raw_logit_deployment_statistics_are_contract_bound_and_label_free(
    tmp_path: Path,
) -> None:
    root = _raw_score_directory(tmp_path)
    grid = np.asarray([-4.0, 0.0, 4.0], dtype=np.float32)
    arrays = build_deployment_statistics(
        root,
        grid,
        adaptation_window=2,
        evaluation_window=1,
        stride=3,
        representation=LOGIT_REPRESENTATION,
        threshold_grid_manifest_sha256="b" * 64,
        threshold_grid_detector_protocol=GRID_DETECTOR_PROTOCOL,
        threshold_grid_detector_checkpoint_sha256s=ALL_DETECTOR_SHA256S,
        threshold_grid_outer_detector_checkpoint_sha256=OUTER_DETECTOR_SHA256,
        threshold_grid_episode_detector_checkpoint_sha256s=(
            EPISODE_DETECTOR_SHA256S
        ),
    )

    assert str(np.asarray(arrays["representation"]).item()) == LOGIT_REPRESENTATION
    assert str(np.asarray(arrays["statistics_schema_version"]).item()) == (
        LOGIT_STATISTICS_SCHEMA_VERSION
    )
    assert str(np.asarray(arrays["threshold_grid_schema_version"]).item()) == (
        LOGIT_GRID_SCHEMA_VERSION
    )
    assert str(np.asarray(arrays["threshold_grid_sha256"]).item()) == (
        logit_threshold_grid_sha256(grid)
    )
    assert str(
        np.asarray(arrays["deployment_statistics_schema_version"]).item()
    ) == LOGIT_DEPLOYMENT_STATISTICS_SCHEMA_VERSION
    provenance = json.loads(str(np.asarray(arrays["provenance_json"]).item()))
    assert provenance["representation"] == LOGIT_REPRESENTATION
    assert provenance["masks_read"] is False
    assert provenance["score_integrity_verified"] is True


def test_raw_logit_statistics_archive_fails_closed_on_feature_hash_mismatch(
    tmp_path: Path,
) -> None:
    root = _raw_score_directory(tmp_path)
    grid = np.asarray([-4.0, 0.0, 4.0], dtype=np.float32)
    arrays = build_deployment_statistics(
        root,
        grid,
        adaptation_window=2,
        evaluation_window=1,
        stride=3,
        representation=LOGIT_REPRESENTATION,
        threshold_grid_manifest_sha256="b" * 64,
        threshold_grid_detector_protocol=GRID_DETECTOR_PROTOCOL,
        threshold_grid_detector_checkpoint_sha256s=ALL_DETECTOR_SHA256S,
        threshold_grid_outer_detector_checkpoint_sha256=OUTER_DETECTOR_SHA256,
        threshold_grid_episode_detector_checkpoint_sha256s=(
            EPISODE_DETECTOR_SHA256S
        ),
    )
    valid = tmp_path / "valid.npz"
    np.savez_compressed(valid, **arrays)
    evidence = _statistics_from_archive(valid)[-1]
    names = tuple(str(value) for value in arrays["statistics_names"])
    assert evidence["feature_schema_sha256"] == feature_schema_sha256(
        LOGIT_STATISTICS_SCHEMA_VERSION, statistics_names=names
    )

    tampered = dict(arrays)
    tampered["feature_schema_sha256"] = np.asarray("0" * 64)
    invalid = tmp_path / "invalid.npz"
    np.savez_compressed(invalid, **tampered)
    with pytest.raises(ValueError, match="feature-schema hash mismatch"):
        _statistics_from_archive(invalid)

    detector_tampered = dict(arrays)
    detector_tampered["threshold_grid_detector_checkpoint_sha256s"] = np.asarray(
        [OUTER_DETECTOR_SHA256, "e" * 64, "f" * 64], dtype=str
    )
    detector_invalid = tmp_path / "invalid-detectors.npz"
    np.savez_compressed(detector_invalid, **detector_tampered)
    with pytest.raises(ValueError, match="detector roles are invalid"):
        _statistics_from_archive(detector_invalid)

    outer_tampered = dict(arrays)
    outer_tampered[
        "threshold_grid_outer_detector_checkpoint_sha256"
    ] = np.asarray("e" * 64)
    outer_invalid = tmp_path / "invalid-outer.npz"
    np.savez_compressed(outer_invalid, **outer_tampered)
    with pytest.raises(ValueError, match="outer-final detector checkpoint"):
        _statistics_from_archive(outer_invalid)

    protocol_tampered = dict(arrays)
    protocol = json.loads(str(np.asarray(arrays["protocol_json"]).item()))
    protocol["threshold_grid_episode_detector_checkpoint_sha256s"] = [
        "c" * 64,
        "e" * 64,
    ]
    protocol_tampered["protocol_json"] = np.asarray(
        json.dumps(protocol, sort_keys=True)
    )
    from certification.build_calibration_losses import protocol_fingerprint

    protocol_tampered["protocol_fingerprint"] = np.asarray(
        protocol_fingerprint(protocol)
    )
    protocol_invalid = tmp_path / "invalid-protocol-inner.npz"
    np.savez_compressed(protocol_invalid, **protocol_tampered)
    with pytest.raises(
        ValueError,
        match="protocol threshold_grid_episode_detector_checkpoint_sha256s mismatch",
    ):
        _statistics_from_archive(protocol_invalid)


def test_raw_logit_score_map_detector_must_be_outer_final(tmp_path: Path) -> None:
    root = _raw_score_directory(
        tmp_path,
        detector_sha256="e" * 64,
    )
    with pytest.raises(ValueError, match="outer-final detector"):
        _score_map_protocol_provenance(
            root,
            np.asarray([-4.0, 0.0, 4.0], dtype=np.float32),
            representation=LOGIT_REPRESENTATION,
            threshold_grid_detector_protocol=GRID_DETECTOR_PROTOCOL,
            threshold_grid_detector_checkpoint_sha256s=ALL_DETECTOR_SHA256S,
            threshold_grid_outer_detector_checkpoint_sha256=(
                OUTER_DETECTOR_SHA256
            ),
            threshold_grid_episode_detector_checkpoint_sha256s=(
                EPISODE_DETECTOR_SHA256S
            ),
        )


def test_raw_logit_zero_label_cli_emits_native_and_display_thresholds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _raw_score_directory(tmp_path)
    grid = np.asarray([-4.0, 0.0, 4.0], dtype=np.float32)
    arrays = build_deployment_statistics(
        root,
        grid,
        adaptation_window=2,
        evaluation_window=1,
        stride=3,
        representation=LOGIT_REPRESENTATION,
        threshold_grid_manifest_sha256="b" * 64,
        threshold_grid_detector_protocol=GRID_DETECTOR_PROTOCOL,
        threshold_grid_detector_checkpoint_sha256s=ALL_DETECTOR_SHA256S,
        threshold_grid_outer_detector_checkpoint_sha256=OUTER_DETECTOR_SHA256,
        threshold_grid_episode_detector_checkpoint_sha256s=(
            EPISODE_DETECTOR_SHA256S
        ),
    )
    statistics = np.asarray(arrays["statistics"])[0]
    names = tuple(str(value) for value in arrays["statistics_names"])
    model = RiskCurvePredictor(
        input_dim=statistics.size,
        num_thresholds=grid.size,
        hidden_dim=4,
        dropout=0.0,
    )
    checkpoint = {
        "model_config": model.config(),
        "model_architecture_version": RISK_CURVE_ARCHITECTURE_VERSION,
        "state_dict": model.state_dict(),
        "thresholds": grid.tolist(),
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": logit_threshold_grid_sha256(grid),
        "statistics_mean": statistics.tolist(),
        "statistics_std": np.ones_like(statistics).tolist(),
        "statistics_names": list(names),
        "statistics_names_sha256": statistics_names_sha256(names),
        "statistics_schema_version": LOGIT_STATISTICS_SCHEMA_VERSION,
        "feature_schema_sha256": feature_schema_sha256(
            LOGIT_STATISTICS_SCHEMA_VERSION, statistics_names=names
        ),
        "threshold_grid_manifest_sha256": "b" * 64,
        "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": ALL_DETECTOR_SHA256S,
        "threshold_grid_outer_detector_checkpoint_sha256": (
            OUTER_DETECTOR_SHA256
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": (
            EPISODE_DETECTOR_SHA256S
        ),
    }
    checkpoint["episode_contract"] = {
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": logit_threshold_grid_sha256(grid),
        "threshold_grid_manifest_sha256": "b" * 64,
        "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": ALL_DETECTOR_SHA256S,
        "threshold_grid_outer_detector_checkpoint_sha256": (
            OUTER_DETECTOR_SHA256
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": (
            EPISODE_DETECTOR_SHA256S
        ),
        "feature_schema_sha256": checkpoint["feature_schema_sha256"],
    }
    checkpoint_path = tmp_path / "curve.pt"
    torch.save(checkpoint, checkpoint_path)
    output = tmp_path / "selection.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_zero_label_threshold",
            "--score-map-dir",
            str(root),
            "--warmup-window",
            "2",
            "--curve-checkpoint",
            str(checkpoint_path),
            "--pixel-budget",
            "1e12",
            "--component-budget",
            "1e12",
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
    )
    select_main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["representation"] == LOGIT_REPRESENTATION
    assert result["threshold_index"] == 0
    assert result["selected_logit_threshold"] == -4.0
    assert result["selected_probability_threshold"] == pytest.approx(
        1.0 / (1.0 + np.exp(4.0))
    )
    assert result["reject"] is False
    assert result["prediction_rule"] == "prediction = (raw_logits >= threshold)"
    assert result["protocol"]["detector_weight_sha256"] == OUTER_DETECTOR_SHA256
    assert result["threshold_grid_outer_detector_checkpoint_sha256"] == (
        OUTER_DETECTOR_SHA256
    )
    assert result["threshold_grid_episode_detector_checkpoint_sha256s"] == (
        EPISODE_DETECTOR_SHA256S
    )
    assert all(np.isfinite(result["thresholds"]))
    assert result["empty_action"]["threshold"] == "+inf"
