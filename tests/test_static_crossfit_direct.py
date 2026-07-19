from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import rc_irstd.cli.evaluate_static_crossfit_direct as direct_cli
from data_ext.mask_alignment import MASK_ALIGNMENT_POLICY
from evaluation.artifact_integrity import (
    SCORE_MANIFEST_SCHEMA_VERSION,
    SCORE_MASK_ALIGNMENT_SCHEMA,
    SCORE_RECORD_INTEGRITY_SCHEMA,
    file_sha256,
    ordered_ids_sha256,
    score_records_sha256,
)
from rc_irstd.features import FeatureSpec, feature_names
from rc_irstd.models import MonotoneBudgetCalibrator


def _write_formal_scores(root: Path, count: int = 40) -> Path:
    root.mkdir()
    image_ids = [f"target-{index:03d}" for index in range(count)]
    split_file = root / "official_test.txt"
    split_file.write_text("\n".join(image_ids) + "\n", encoding="utf-8")
    records: list[dict[str, object]] = []
    for index, image_id in enumerate(image_ids):
        filename = f"{index:03d}.npz"
        probability = np.full((4, 4), 0.1, dtype=np.float32)
        probability[1, 1] = 0.9
        gray = np.full((4, 4), index / max(count - 1, 1), dtype=np.float32)
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1, 1] = 1
        np.savez_compressed(
            root / filename,
            prob=probability,
            gray=gray,
            mask=mask,
            image_id=np.asarray(image_id),
            labels_loaded=np.asarray(True),
            original_hw=np.asarray([4, 4], dtype=np.int32),
            mask_alignment_applied=np.asarray(False),
            mask_original_hw=np.asarray([4, 4], dtype=np.int32),
            mask_aspect_relative_error=np.asarray(0.0),
            mask_alignment_policy=np.asarray(MASK_ALIGNMENT_POLICY),
        )
        records.append(
            {
                "image_id": image_id,
                "file": filename,
                "shape": [4, 4],
                "sha256": file_sha256(root / filename),
                "mask_alignment_applied": False,
                "mask_original_hw": [4, 4],
                "mask_aspect_relative_error": 0.0,
            }
        )
    manifest = {
        "schema_version": SCORE_MANIFEST_SCHEMA_VERSION,
        "record_integrity_schema": SCORE_RECORD_INTEGRITY_SCHEMA,
        "score_type": "sigmoid_probability",
        "warm_flag": True,
        "labels_loaded": True,
        "spatial_mode": "native",
        "base_hw": None,
        "pad_multiple": 16,
        "target_dataset": "unseen-target",
        "source_datasets": ["detector-source-a", "detector-source-b"],
        "weight_sha256": "a" * 64,
        "checkpoint_diagnostic_only": False,
        "non_strict_state_loading": False,
        "split_role": "test",
        "split_authority_verified": True,
        "split_file": str(split_file.resolve()),
        "split_file_sha256": file_sha256(split_file),
        "split_ordered_ids_sha256": ordered_ids_sha256(image_ids),
        "num_images": len(records),
        "records": records,
        "records_sha256": score_records_sha256(records),
        "ordered_image_ids_sha256": ordered_ids_sha256(image_ids),
        "mask_alignment_schema": SCORE_MASK_ALIGNMENT_SCHEMA,
        "mask_alignment_policy": MASK_ALIGNMENT_POLICY,
        "mask_alignment_count": 0,
        "mask_aligned_sample_ids": [],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return root


def _write_formal_checkpoint(path: Path) -> Path:
    spec = FeatureSpec()
    names = feature_names(spec)
    model = MonotoneBudgetCalibrator(
        feature_dim=len(names),
        budget_grid=[0.5, 0.1],
        hidden_dims=(8,),
        dropout=0.0,
    )
    model.normalizer.fit(torch.zeros(2, len(names)))
    with torch.no_grad():
        model.spacing_head.weight.zero_()
        model.spacing_head.bias.zero_()
    torch.save(
        {
            "format_version": 2,
            "kind": "calibrator",
            "method_name": "direct_threshold",
            "model_class": "MonotoneBudgetCalibrator",
            "role": "baseline",
            "diagnostic_only": False,
            "formal_causal_contract": True,
            "formal_paper_checkpoint": True,
            "model_state": model.state_dict(),
            "model_config": model.export_config(),
            "budgets": [0.5, 0.1],
            "feature_spec": spec.to_dict(),
            "feature_names": names,
            "episode_metadata": {
                "support_size": 32,
                "query_size": 1,
                "stride": 33,
                "mode": "causal",
                "domain_names": ["meta-source-a", "meta-source-b"],
                "formal_causal_contract": True,
                "diagnostic_only": False,
            },
        },
        path,
    )
    return path


def test_static_direct_is_deterministic_mask_blind_until_freeze_and_full_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score_dir = _write_formal_scores(tmp_path / "scores")
    checkpoint = _write_formal_checkpoint(tmp_path / "direct.pt")
    events: list[str] = []
    original_support_loader = direct_cli.load_score_sample
    original_query_loader = direct_cli.load_score_item

    def support_loader(path, *, require_mask=True):
        assert require_mask is False
        events.append("support_prob_gray_only")
        return original_support_loader(path, require_mask=require_mask)

    def query_loader(path):
        events.append("query_mask")
        return original_query_loader(path)

    monkeypatch.setattr(direct_cli, "load_score_sample", support_loader)
    monkeypatch.setattr(direct_cli, "load_score_item", query_loader)
    output = tmp_path / "result.json"
    pairs = [
        direct_cli.BudgetPair("loose", pixel=0.5, component=1.0),
        direct_cli.BudgetPair("strict", pixel=0.1, component=1.0),
    ]
    result = direct_cli.evaluate_static_crossfit_direct(
        score_dir=score_dir,
        calibrator_path=checkpoint,
        output_path=output,
        budget_pairs=pairs,
        seed=19,
        device_name="cpu",
    )

    assert events[: 5 * 32] == ["support_prob_gray_only"] * (5 * 32)
    assert events[5 * 32 :] == ["query_mask"] * 40
    assert result["support_masks_parsed"] is False
    assert result["query_masks_loaded_after_threshold_freeze"] is True
    assert result["test_labels_used_for_threshold_selection"] is False
    assert result["test_labels_used_for_labelled_audit"] is True
    assert result["full_test_coverage"] is True
    assert result["every_query_evaluated_exactly_once"] is True
    assert result["num_evaluated_query_images"] == 40
    assert result["transductive"] is True
    assert result["causal_claim"] is False
    assert result["formal_certification"] is False
    assert result["unseen_target_contract_verified"] is True
    assert len(result["folds"]) == 5
    assert all(fold["support_size"] == 32 for fold in result["folds"])
    assert all(fold["query_size"] == 8 for fold in result["folds"])
    assert all(
        not set(fold["support_ids"]).intersection(fold["query_ids"])
        for fold in result["folds"]
    )
    query_ids = [
        image_id for fold in result["folds"] for image_id in fold["query_ids"]
    ]
    assert len(query_ids) == len(set(query_ids)) == 40
    by_name = {record["name"]: record for record in result["results"]}
    assert set(by_name) == {"loose", "strict"}
    assert by_name["loose"]["pd"] == pytest.approx(1.0)
    assert by_name["loose"]["pixel_risk"] == pytest.approx(0.0)
    assert by_name["loose"]["component_risk_raw"] == pytest.approx(0.0)
    assert by_name["loose"]["joint_bsr"] == pytest.approx(1.0)
    assert by_name["loose"]["mean_relative_excess"] == pytest.approx(0.0)
    assert result["provenance"]["thresholds_frozen_before_query_mask_loading"] is True
    assert len(result["provenance"]["frozen_actions_sha256"]) == 64
    assert json.loads(output.read_text(encoding="utf-8")) == result

    manifest = json.loads((score_dir / "manifest.json").read_text(encoding="utf-8"))
    ids = [record["image_id"] for record in manifest["records"]]
    assert direct_cli._build_static_folds(ids, folds=5, seed=19) == (
        direct_cli._build_static_folds(ids, folds=5, seed=19)
    )


@pytest.mark.parametrize(
    ("field", "value"), [("support_size", 31), ("query_size", 2)]
)
def test_static_direct_rejects_checkpoint_outside_A32_E1_contract(
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    checkpoint_path = _write_formal_checkpoint(tmp_path / "direct.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["episode_metadata"][field] = value
    with pytest.raises(ValueError, match="A=32/E=1"):
        direct_cli._strict_checkpoint_contract(checkpoint)


def test_static_direct_rejects_checkpoint_outside_stride33_contract(
    tmp_path: Path,
) -> None:
    checkpoint_path = _write_formal_checkpoint(tmp_path / "direct.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["episode_metadata"]["stride"] = 34
    with pytest.raises(ValueError, match="stride=33"):
        direct_cli._strict_checkpoint_contract(checkpoint)


def test_static_direct_rejects_seen_target_and_bad_budget_contract(tmp_path: Path) -> None:
    score_dir = _write_formal_scores(tmp_path / "scores")
    checkpoint_path = _write_formal_checkpoint(tmp_path / "direct.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["episode_metadata"]["domain_names"].append("unseen-target")
    seen_path = tmp_path / "seen.pt"
    torch.save(checkpoint, seen_path)
    with pytest.raises(ValueError, match="target_seen_by_calibrator"):
        direct_cli.evaluate_static_crossfit_direct(
            score_dir=score_dir,
            calibrator_path=seen_path,
            output_path=tmp_path / "seen.json",
            budget_pairs=[direct_cli.BudgetPair("b", 0.5, 1.0)],
            device_name="cpu",
        )

    with pytest.raises(ValueError, match="outside the trained grid"):
        direct_cli.evaluate_static_crossfit_direct(
            score_dir=score_dir,
            calibrator_path=checkpoint_path,
            output_path=tmp_path / "bad-budget.json",
            budget_pairs=[direct_cli.BudgetPair("too_strict", 0.01, 1.0)],
            device_name="cpu",
        )


def test_static_direct_cli_budget_pair_parser_is_strict() -> None:
    assert direct_cli.parse_budget_pair("strict:1e-6:1").name == "strict"
    with pytest.raises(argparse.ArgumentTypeError):
        direct_cli.parse_budget_pair("missing-component:1e-6")
    with pytest.raises(argparse.ArgumentTypeError):
        direct_cli.parse_budget_pair("bad name:1e-6:1")
