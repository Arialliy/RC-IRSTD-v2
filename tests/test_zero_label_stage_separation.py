"""Executable tests for zero-label selection and post-freeze labelled audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from data_ext.eval_dataset import IRSTDEvalDataset
from evaluation.artifact_integrity import file_sha256, score_records_sha256
from evaluation.export_score_maps import export_dataset_score_maps
from evaluation.target_stage_separation import (
    PAIR_AUDIT_SCHEMA_VERSION,
    audit_target_score_stage_pair,
    freeze_zero_label_actions,
)


ROOT = Path(__file__).resolve().parents[1]


class _DeterministicLogitModel(torch.nn.Module):
    def forward(self, images: torch.Tensor, warm_flag: bool) -> torch.Tensor:
        del warm_flag
        rows = torch.arange(images.shape[-2], device=images.device).view(1, 1, -1, 1)
        cols = torch.arange(images.shape[-1], device=images.device).view(1, 1, 1, -1)
        return (rows.float() - cols.float()).expand(images.shape[0], 1, -1, -1)


def _dataset(root: Path, *, labels_loaded: bool) -> IRSTDEvalDataset:
    return IRSTDEvalDataset(
        root,
        split="test",
        spatial_mode="native",
        pad_multiple=4,
        dataset_name="target-domain",
        load_masks=labels_loaded,
    )


def _make_dataset(root: Path) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 1] = 255
    Image.fromarray(image).save(root / "images" / "sample.png")
    Image.fromarray(mask).save(root / "masks" / "sample.png")
    (root / "test.txt").write_text("sample\n", encoding="utf-8")
    return root


def _metadata() -> dict[str, object]:
    return {
        "weight_sha256": "a" * 64,
        "checkpoint_selection_rule": "fixed_last",
        "checkpoint_diagnostic_only": False,
        "diagnostic_only": False,
        "non_strict_state_loading": False,
        "formal_protocol_eligible": True,
        "model_backend": "rc_mshnet",
        "source_datasets": ["source-a", "source-b"],
    }


def _export(root: Path, destination: Path, *, labels_loaded: bool) -> None:
    export_dataset_score_maps(
        _DeterministicLogitModel(),
        _dataset(root, labels_loaded=labels_loaded),
        destination,
        labels_loaded=labels_loaded,
        manifest_metadata=_metadata(),
        export_raw_logits=True,
    )


def _freeze(tmp_path: Path, zero_result: Path) -> Path:
    bindings = []
    for name in ("statistics.npz", "curve.pt", "grid.npy", "grid.json", "grid.sha256"):
        path = tmp_path / name
        path.write_bytes(name.encode("utf-8"))
        bindings.append(path)
    return freeze_zero_label_actions(
        [zero_result],
        bound_artifacts=bindings,
        output_dir=tmp_path / "freeze",
    )


def test_post_freeze_pair_audit_proves_identity_and_logit_equivalence(
    tmp_path: Path,
) -> None:
    dataset_root = _make_dataset(tmp_path / "dataset")
    unlabeled = tmp_path / "scores_unlabeled"
    labeled = tmp_path / "scores_labeled_audit"
    _export(dataset_root, unlabeled, labels_loaded=False)
    zero_result = tmp_path / "zero_selection.json"
    zero_result.write_text(json.dumps({"threshold_index": 3}), encoding="utf-8")
    freeze_record = _freeze(tmp_path, zero_result)
    _export(dataset_root, labeled, labels_loaded=True)

    output = tmp_path / "pair_audit.json"
    result = audit_target_score_stage_pair(
        unlabeled,
        labeled,
        freeze_record=freeze_record,
        output=output,
    )
    assert result["schema_version"] == PAIR_AUDIT_SCHEMA_VERSION
    assert result["verified"] is True
    assert result["labels_loaded_during_selection"] is False
    assert result["labels_loaded_during_audit"] is True
    assert result["labeled_audit_created_after_freeze"] is True
    assert result["frozen_zero_results"] == [
        {"path": str(zero_result.resolve()), "sha256": result["frozen_zero_results"][0]["sha256"]}
    ]
    assert len(result["raw_logit_stream_sha256"]) == 64
    assert output.is_file()

    unlabeled_manifest = json.loads(
        (unlabeled / "manifest.json").read_text(encoding="utf-8")
    )
    unlabeled_record = unlabeled / unlabeled_manifest["records"][0]["file"]
    with np.load(unlabeled_record, allow_pickle=False) as payload:
        assert "mask" not in payload
    labeled_manifest = json.loads(
        (labeled / "manifest.json").read_text(encoding="utf-8")
    )
    labeled_record = labeled / labeled_manifest["records"][0]["file"]
    with np.load(labeled_record, allow_pickle=False) as payload:
        assert "mask" in payload


def test_labeled_export_created_before_freeze_is_rejected(tmp_path: Path) -> None:
    dataset_root = _make_dataset(tmp_path / "dataset")
    unlabeled = tmp_path / "scores_unlabeled"
    labeled = tmp_path / "scores_labeled_audit"
    _export(dataset_root, unlabeled, labels_loaded=False)
    _export(dataset_root, labeled, labels_loaded=True)
    zero_result = tmp_path / "zero_selection.json"
    zero_result.write_text("{}", encoding="utf-8")
    freeze_record = _freeze(tmp_path, zero_result)
    with pytest.raises(ValueError, match="created after action freeze"):
        audit_target_score_stage_pair(
            unlabeled,
            labeled,
            freeze_record=freeze_record,
            output=tmp_path / "audit.json",
        )


def test_unlabeled_record_cannot_hide_a_mask_even_with_rehashed_manifest(
    tmp_path: Path,
) -> None:
    dataset_root = _make_dataset(tmp_path / "dataset")
    unlabeled = tmp_path / "scores_unlabeled"
    labeled = tmp_path / "scores_labeled_audit"
    _export(dataset_root, unlabeled, labels_loaded=False)
    zero_result = tmp_path / "zero_selection.json"
    zero_result.write_text("{}", encoding="utf-8")
    freeze_record = _freeze(tmp_path, zero_result)
    _export(dataset_root, labeled, labels_loaded=True)

    manifest_path = unlabeled / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record_path = unlabeled / manifest["records"][0]["file"]
    with np.load(record_path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays["mask"] = np.zeros_like(arrays["prob"], dtype=np.uint8)
    np.savez_compressed(record_path, **arrays)
    manifest["records"][0]["sha256"] = file_sha256(record_path)
    manifest["records_sha256"] = score_records_sha256(manifest["records"])
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpectedly embeds a mask"):
        audit_target_score_stage_pair(
            unlabeled,
            labeled,
            freeze_record=freeze_record,
            output=tmp_path / "audit.json",
        )


def test_pipeline_orders_freeze_before_labeled_export_and_audit() -> None:
    runner = (ROOT / "rc_irstd" / "cli" / "run_pipeline.py").read_text(
        encoding="utf-8"
    )
    freeze = runner.index("freeze_record = freeze_zero_label_actions")
    labeled = runner.index('audit_score_dir = target_root / "scores_labeled_audit"')
    pair = runner.index("audit_target_score_stage_pair(")
    counts = runner.index('"certification.build_calibration_losses"', pair)
    selected = runner.index('"evaluation.evaluate_selected_actions"', counts)
    assert freeze < labeled < pair < counts < selected
    assert 'labels_loaded=False if raw_logit_target else evaluate_target' in runner
    assert 'labels_loaded=True,' in runner[labeled:pair]
    assert '"--target-stage-pair-audit"' in runner
    selected_block = runner[selected : selected + 2200]
    assert '"--count-curves"' in selected_block
    assert '"--target-stage-pair-audit"' in selected_block
    assert '"--reject-as-empty"' in selected_block
    assert 'budget_root / "selected_action_metrics.json"' in runner
