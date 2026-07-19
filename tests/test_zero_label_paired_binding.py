from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from data_ext.eval_dataset import IRSTDEvalDataset
from evaluation.artifact_integrity import file_sha256, verify_score_map_directory
from evaluation.export_score_maps import export_dataset_score_maps
from evaluation.target_stage_separation import (
    audit_target_score_stage_pair,
    freeze_zero_label_actions,
)
from risk_curve.evaluate_zero_label import validate_count_curve_binding


DETECTOR_SHA256 = "a" * 64
PROTOCOL_FINGERPRINT = "b" * 64


class _DeterministicLogitModel(torch.nn.Module):
    def forward(
        self,
        images: torch.Tensor,
        warm_flag: bool,
    ) -> torch.Tensor:
        del warm_flag
        return images[:, :1]


def _dataset(root: Path, *, labels_loaded: bool) -> IRSTDEvalDataset:
    return IRSTDEvalDataset(
        root,
        split="test",
        spatial_mode="native",
        pad_multiple=4,
        dataset_name="target-domain",
        load_masks=labels_loaded,
    )


def _export(
    root: Path,
    dataset: IRSTDEvalDataset,
    *,
    labels_loaded: bool,
) -> None:
    export_dataset_score_maps(
        _DeterministicLogitModel(),
        dataset,
        root,
        labels_loaded=labels_loaded,
        export_raw_logits=True,
        manifest_metadata={
            "weight_sha256": DETECTOR_SHA256,
            "source_datasets": ["source-a", "source-b"],
            "checkpoint_selection_rule": "fixed_last",
            "checkpoint_diagnostic_only": False,
            "diagnostic_only": False,
            "non_strict_state_loading": False,
            "formal_protocol_eligible": True,
            "model_backend": "rc_mshnet",
        },
    )


def _paired_contract(tmp_path: Path) -> tuple[
    dict[str, object],
    dict[str, object],
    list[str],
    dict[str, object],
    str,
]:
    dataset_root = tmp_path / "dataset"
    images = dataset_root / "images"
    masks = dataset_root / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    image_ids = ["sample-0", "sample-1"]
    for index, image_id in enumerate(image_ids):
        image = np.full((4, 4, 3), 32 + index, dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[index, index] = 255
        Image.fromarray(image).save(images / f"{image_id}.png")
        Image.fromarray(mask).save(masks / f"{image_id}.png")
    (dataset_root / "test.txt").write_text(
        "\n".join(image_ids) + "\n",
        encoding="utf-8",
    )

    unlabeled_dir = tmp_path / "scores-unlabeled"
    _export(
        unlabeled_dir,
        _dataset(dataset_root, labels_loaded=False),
        labels_loaded=False,
    )
    unlabeled_manifest, _, unlabeled_integrity = verify_score_map_directory(
        unlabeled_dir,
        require_integrity=True,
        require_masks=False,
    )
    assert unlabeled_manifest is not None
    statistics_provenance = {
        "score_map_dir": str(unlabeled_dir.resolve()),
        "score_manifest_sha256": unlabeled_integrity["manifest_sha256"],
        "score_records_sha256": unlabeled_integrity["records_sha256"],
        "score_ordered_image_ids_sha256": unlabeled_integrity[
            "ordered_image_ids_sha256"
        ],
        "score_num_records": unlabeled_integrity["num_records"],
    }
    zero_result: dict[str, object] = {
        "adaptation_protocol": "static_cross_fit",
        "threshold_indices_by_image": {image_id: 0 for image_id in image_ids},
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "protocol": {"detector_weight_sha256": DETECTOR_SHA256},
        "statistics_artifact": {"provenance": statistics_provenance},
    }
    zero_path = tmp_path / "zero-result.json"
    zero_path.write_text(
        json.dumps(zero_result, sort_keys=True),
        encoding="utf-8",
    )
    freeze_path = freeze_zero_label_actions(
        [zero_path],
        bound_artifacts=[unlabeled_dir / "manifest.json"],
        output_dir=tmp_path / "freeze",
    )

    labeled_dir = tmp_path / "scores-labeled-audit"
    _export(
        labeled_dir,
        _dataset(dataset_root, labels_loaded=True),
        labels_loaded=True,
    )
    _, _, labeled_integrity = verify_score_map_directory(
        labeled_dir,
        require_integrity=True,
        require_masks=True,
    )
    pair_audit = audit_target_score_stage_pair(
        unlabeled_dir,
        labeled_dir,
        freeze_record=freeze_path,
        output=tmp_path / "target-stage-pair-audit.json",
    )
    count_provenance: dict[str, object] = {
        "source_type": "exported_score_map_directory",
        "score_dir": str(labeled_dir.resolve()),
        "manifest_sha256": labeled_integrity["manifest_sha256"],
        "score_records_sha256": labeled_integrity["records_sha256"],
        "score_ordered_image_ids_sha256": labeled_integrity[
            "ordered_image_ids_sha256"
        ],
        "score_num_records": labeled_integrity["num_records"],
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "protocol": {"detector_weight_sha256": DETECTOR_SHA256},
    }
    return (
        zero_result,
        count_provenance,
        image_ids,
        pair_audit,
        file_sha256(zero_path),
    )


def test_paired_binding_accepts_distinct_verified_score_artifacts(
    tmp_path: Path,
) -> None:
    zero, counts, image_ids, pair, zero_sha = _paired_contract(tmp_path)

    result = validate_count_curve_binding(
        zero,
        counts,
        image_ids,
        pair_audit=pair,
        zero_result_sha256=zero_sha,
    )

    assert result["verified"] is True
    assert result["binding_mode"] == (
        "paired_unlabeled_selection_and_labeled_audit"
    )
    assert result["selection_manifest_sha256"] != result[
        "labeled_manifest_sha256"
    ]
    assert result["selection_records_sha256"] != result[
        "labeled_records_sha256"
    ]
    assert result["detector_weight_sha256"] == DETECTOR_SHA256


def test_paired_binding_fails_closed_on_pair_contract_tampering(
    tmp_path: Path,
) -> None:
    zero, counts, image_ids, pair, zero_sha = _paired_contract(tmp_path)
    cases: list[tuple[dict[str, object], str]] = []

    wrong_ids = dict(pair)
    wrong_ids["ordered_image_ids_sha256"] = "c" * 64
    cases.append((wrong_ids, "ordered image-ID SHA"))

    wrong_checkpoint = dict(pair)
    wrong_checkpoint["detector_weight_sha256"] = "d" * 64
    cases.append((wrong_checkpoint, "checkpoints|checkpoint"))

    wrong_logits = dict(pair)
    wrong_logits["raw_logit_stream_sha256"] = "e" * 64
    cases.append((wrong_logits, "raw-logit stream SHA"))

    missing_spatial = dict(pair)
    missing_spatial["spatial_protocol_fields_verified"] = [
        field
        for field in pair["spatial_protocol_fields_verified"]
        if field != "spatial_mode"
    ]
    cases.append((missing_spatial, "omits spatial protocol fields"))

    wrong_label_stage = dict(pair)
    wrong_label_stage["labels_loaded_during_selection"] = True
    cases.append((wrong_label_stage, "labels_loaded_during_selection"))

    for corrupted, message in cases:
        with pytest.raises(ValueError, match=message):
            validate_count_curve_binding(
                zero,
                counts,
                image_ids,
                pair_audit=corrupted,
                zero_result_sha256=zero_sha,
            )

    with pytest.raises(ValueError, match="not bound by the selection freeze"):
        validate_count_curve_binding(
            zero,
            counts,
            image_ids,
            pair_audit=pair,
            zero_result_sha256="f" * 64,
        )
