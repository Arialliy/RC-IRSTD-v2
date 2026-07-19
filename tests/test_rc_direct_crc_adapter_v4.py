from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from certification.build_calibration_losses import build_calibration_losses
from certification.calibrate_target_offset import (
    RAW_LOGIT_RESULT_SCHEMA_VERSION,
    audit_zero_artifact_contract,
    calibrate_target_offset,
)
from data_ext.eval_dataset import IRSTDEvalDataset
from evaluation.export_score_maps import export_dataset_score_maps
from rc_irstd.models.calibrator import (
    RC_DIRECT_ARCHITECTURE_VERSION,
    MonotoneBudgetCalibrator,
)
from risk_curve.adapt_direct_selection_for_crc_v4 import (
    DIRECT_CRC_ADAPTER_SCHEMA_VERSION,
    build_direct_crc_zero_result,
)
from risk_curve.build_deployment_statistics import build_deployment_statistics
from risk_curve.direct_calibrator import (
    RC_DIRECT_BUDGET_SCHEMA_VERSION,
    RC_DIRECT_CHECKPOINT_SCHEMA_VERSION,
)
from risk_curve.domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    statistics_names_sha256,
)
from risk_curve.representation import (
    GRID_DETECTOR_PROTOCOL,
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_REPRESENTATION,
    logit_threshold_grid_sha256,
)
from risk_curve.select_zero_label_threshold import ZERO_RESULT_SCHEMA_VERSION
from risk_curve.train_curve_predictor import (
    TRAINING_EPISODE_CONTRACT_SCHEMA_VERSION,
)


GRID = np.asarray([-4.0, 0.0, 4.0], dtype=np.float32)
GRID_HASH = logit_threshold_grid_sha256(GRID)
GRID_MANIFEST_HASH = "b" * 64
OUTER_HASH = "a" * 64
EPISODE_HASHES = ["c" * 64, "d" * 64]
ALL_HASHES = [OUTER_HASH, *EPISODE_HASHES]


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


def _score_directory(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "target"
    images = dataset_root / "images"
    images.mkdir(parents=True)
    image_ids = [f"sample-{index}" for index in range(6)]
    for index, image_id in enumerate(image_ids):
        pixels = np.full((4, 4, 3), index * 10, dtype=np.uint8)
        Image.fromarray(pixels).save(images / f"{image_id}.png")
    (dataset_root / "train.txt").write_text(
        "\n".join(image_ids) + "\n", encoding="utf-8"
    )
    dataset = IRSTDEvalDataset(
        dataset_root,
        split="train",
        spatial_mode="native",
        pad_multiple=4,
        dataset_name="target-domain",
        load_masks=False,
    )
    output = tmp_path / "scores"
    export_dataset_score_maps(
        _ConstantLogitModel(),
        dataset,
        output,
        labels_loaded=False,
        export_raw_logits=True,
        manifest_metadata={
            "weight_sha256": OUTER_HASH,
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


def _episode_contract(feature_hash: str) -> dict[str, object]:
    split = {
        "verified": True,
        "formal_protocol_eligible": True,
        "adaptation_window": 2,
        "evaluation_window": 1,
        "stride": 3,
    }
    protocol_fields = {
        "protocol": "causal_adaptation_then_future_evaluation",
        "adaptation_window": 2,
        "evaluation_window": 1,
        "stride": 3,
        "pseudo_targets": ["source-a", "source-b"],
        "source_reference_sha256": None,
        "source_reference_domain_names": [],
        "source_reference_statistics_names_sha256": None,
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": GRID_HASH,
        "threshold_grid_manifest_sha256": GRID_MANIFEST_HASH,
        "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": ALL_HASHES,
        "threshold_grid_outer_detector_checkpoint_sha256": OUTER_HASH,
        "threshold_grid_episode_detector_checkpoint_sha256s": EPISODE_HASHES,
        "feature_schema_sha256": feature_hash,
    }
    return {
        "schema_version": TRAINING_EPISODE_CONTRACT_SCHEMA_VERSION,
        "verified": True,
        "formal_protocol_eligible": True,
        "ineligibility_reasons": [],
        "adaptation_window": 2,
        "evaluation_window": 1,
        "stride": 3,
        "risk_target_unit": "aggregate_risk_over_1_future_images",
        "one_to_one_future_target": True,
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": GRID_HASH,
        "threshold_grid_manifest_sha256": GRID_MANIFEST_HASH,
        "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": ALL_HASHES,
        "threshold_grid_outer_detector_checkpoint_sha256": OUTER_HASH,
        "threshold_grid_episode_detector_checkpoint_sha256s": EPISODE_HASHES,
        "feature_schema_sha256": feature_hash,
        "protocol_fields": protocol_fields,
        "train": dict(split),
        "validation": dict(split),
    }


def _direct_checkpoint(path: Path, arrays: dict[str, object]) -> None:
    names = tuple(str(value) for value in np.asarray(arrays["statistics_names"]))
    feature_hash = str(np.asarray(arrays["feature_schema_sha256"]).item())
    statistics = np.asarray(arrays["statistics"], dtype=np.float32)
    model = MonotoneBudgetCalibrator(
        feature_dim=len(names),
        budget_grid=[1.0, 0.1],
        hidden_dims=(4,),
        dropout=0.0,
        representation=LOGIT_REPRESENTATION,
        threshold_grid=GRID.tolist(),
        architecture_version=RC_DIRECT_ARCHITECTURE_VERSION,
    )
    model.normalizer.fit(torch.from_numpy(statistics))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    checkpoint = {
        "checkpoint_schema_version": RC_DIRECT_CHECKPOINT_SCHEMA_VERSION,
        "kind": "calibrator",
        "method_name": "direct_threshold",
        "model_class": "MonotoneBudgetCalibrator",
        "role": "baseline",
        "representation": LOGIT_REPRESENTATION,
        "thresholds": torch.from_numpy(GRID.copy()),
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": GRID_HASH,
        "threshold_grid_manifest_sha256": GRID_MANIFEST_HASH,
        "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": ALL_HASHES,
        "threshold_grid_outer_detector_checkpoint_sha256": OUTER_HASH,
        "threshold_grid_episode_detector_checkpoint_sha256s": EPISODE_HASHES,
        "statistics_schema_version": LOGIT_STATISTICS_SCHEMA_VERSION,
        "statistics_names": list(names),
        "statistics_names_sha256": statistics_names_sha256(names),
        "feature_schema_sha256": feature_hash,
        "statistics_mean": torch.from_numpy(statistics.mean(axis=0)),
        "statistics_std": torch.ones(len(names), dtype=torch.float32),
        "budget_schema_version": RC_DIRECT_BUDGET_SCHEMA_VERSION,
        "pixel_budgets": [1.0, 0.1],
        "component_budgets": [1.0, 0.1],
        "model_architecture_version": RC_DIRECT_ARCHITECTURE_VERSION,
        "model_config": model.export_config(),
        "state_dict": model.state_dict(),
        "episode_contract": _episode_contract(feature_hash),
        "target_label_policy": {
            "model_inputs": "adaptation_window_A_label_free_statistics_only",
            "supervision": "source_official_train_future_E_risk_only",
            "outer_target_labels_used_for_features": False,
            "outer_target_labels_used_for_checkpoint_selection": False,
        },
    }
    torch.save(checkpoint, path)


def _artifacts(tmp_path: Path) -> tuple[Path, Path]:
    score_dir = _score_directory(tmp_path)
    arrays = build_deployment_statistics(
        score_dir,
        GRID,
        adaptation_window=2,
        evaluation_window=1,
        stride=3,
        representation=LOGIT_REPRESENTATION,
        threshold_grid_manifest_sha256=GRID_MANIFEST_HASH,
        threshold_grid_detector_protocol=GRID_DETECTOR_PROTOCOL,
        threshold_grid_detector_checkpoint_sha256s=ALL_HASHES,
        threshold_grid_outer_detector_checkpoint_sha256=OUTER_HASH,
        threshold_grid_episode_detector_checkpoint_sha256s=EPISODE_HASHES,
    )
    statistics_path = tmp_path / "deployment.npz"
    np.savez_compressed(statistics_path, **arrays)
    checkpoint_path = tmp_path / "direct.pt"
    _direct_checkpoint(checkpoint_path, arrays)
    return checkpoint_path, statistics_path


def test_direct_adapter_enters_formal_crc_zero_artifact_gate(tmp_path: Path) -> None:
    checkpoint_path, statistics_path = _artifacts(tmp_path)
    output = tmp_path / "direct-zero.json"
    build_direct_crc_zero_result(
        checkpoint_path=checkpoint_path,
        statistics_file=statistics_path,
        output=output,
        pixel_budget=1.0,
        component_budget=1.0,
        device="cpu",
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["schema_version"] == ZERO_RESULT_SCHEMA_VERSION
    assert result["adapter_schema_version"] == DIRECT_CRC_ADAPTER_SCHEMA_VERSION
    assert result["method_name"] == "direct_threshold"
    assert result["threshold_grid_detector_checkpoint_sha256s"] == ALL_HASHES
    assert result["threshold_grid_outer_detector_checkpoint_sha256"] == OUTER_HASH
    assert result["threshold_grid_episode_detector_checkpoint_sha256s"] == (
        EPISODE_HASHES
    )
    assert result["direct_checkpoint_sha256"] == result["curve_checkpoint_sha256"]
    assert result["selection_data_contract"]["formal_crc_eligible"] is True
    audit = audit_zero_artifact_contract(result, ["sample-2"], ["sample-5"])
    assert audit["verified"] is True
    assert audit["curve_checkpoint_contract"]["checkpoint_method_name"] == (
        "direct_threshold"
    )
    calibration_losses = build_calibration_losses(
        image_ids=["sample-2"],
        thresholds=GRID,
        false_positive_pixels=np.zeros((1, GRID.size), dtype=np.int64),
        false_positive_components=np.zeros((1, GRID.size), dtype=np.int64),
        total_pixels=np.asarray([16], dtype=np.int64),
        pixel_budget=1.0,
        component_budget=1.0,
        representation=LOGIT_REPRESENTATION,
        threshold_grid_schema_version=LOGIT_GRID_SCHEMA_VERSION,
        recorded_threshold_grid_sha256=GRID_HASH,
        threshold_grid_manifest_sha256=GRID_MANIFEST_HASH,
        threshold_grid_detector_protocol=GRID_DETECTOR_PROTOCOL,
        threshold_grid_detector_checkpoint_sha256s=ALL_HASHES,
        threshold_grid_outer_detector_checkpoint_sha256=OUTER_HASH,
        threshold_grid_episode_detector_checkpoint_sha256s=EPISODE_HASHES,
    )
    crc = calibrate_target_offset(
        calibration_losses,
        alpha=0.5,
        test_image_ids=["sample-5"],
        calibration_zero_indices=[
            result["threshold_indices_by_image"]["sample-2"]
        ],
        test_zero_indices=[result["threshold_indices_by_image"]["sample-5"]],
        adaptation_image_ids=result["adaptation_image_ids"],
        curve_checkpoint_sha256=result["direct_checkpoint_sha256"],
    )
    assert crc["schema_version"] == RAW_LOGIT_RESULT_SCHEMA_VERSION
    assert crc["adaptation_mode"] == "sample_adaptive_zero_plus_shared_offset"
    assert crc["success"] is True


def test_direct_crc_zero_result_rejects_role_tampering(tmp_path: Path) -> None:
    checkpoint_path, statistics_path = _artifacts(tmp_path)
    output = tmp_path / "direct-zero.json"
    build_direct_crc_zero_result(
        checkpoint_path=checkpoint_path,
        statistics_file=statistics_path,
        output=output,
        pixel_budget=1.0,
        component_budget=1.0,
        device="cpu",
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    result["threshold_grid_outer_detector_checkpoint_sha256"] = "e" * 64
    audit = audit_zero_artifact_contract(result, ["sample-2"], ["sample-5"])
    assert audit["verified"] is False
    assert "differs" in audit["errors"]["zero_artifact"]


def test_direct_crc_adapter_rejects_manifest_binding_tamper(tmp_path: Path) -> None:
    checkpoint_path, statistics_path = _artifacts(tmp_path)
    with np.load(statistics_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["threshold_grid_manifest_sha256"] = np.asarray("f" * 64)
    tampered = tmp_path / "tampered-deployment.npz"
    np.savez_compressed(tampered, **arrays)
    with pytest.raises(ValueError, match="grid-manifest hashes differ"):
        build_direct_crc_zero_result(
            checkpoint_path=checkpoint_path,
            statistics_file=tampered,
            output=tmp_path / "should-not-exist.json",
            pixel_budget=1.0,
            component_budget=1.0,
            device="cpu",
        )
