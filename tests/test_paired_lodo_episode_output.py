from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import risk_curve.build_curve_episodes as episode_module
from data_ext.mask_alignment import MASK_ALIGNMENT_POLICY
from evaluation.artifact_integrity import (
    SCORE_MANIFEST_SCHEMA_VERSION,
    SCORE_MASK_ALIGNMENT_SCHEMA,
    SCORE_RECORD_INTEGRITY_SCHEMA,
    file_sha256,
    ordered_ids_sha256,
    score_records_sha256,
)


def _write_scores(root: Path, target: str) -> None:
    root.mkdir(parents=True)
    records: list[dict[str, object]] = []
    image_ids: list[str] = []
    for index in range(4):
        image_id = f"{target}-{index}"
        filename = f"{index:03d}.npz"
        probability = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
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
    manifest = {
        "schema_version": SCORE_MANIFEST_SCHEMA_VERSION,
        "record_integrity_schema": SCORE_RECORD_INTEGRITY_SCHEMA,
        "target_dataset": target,
        "source_datasets": ["independent-source"],
        "weight_sha256": "a" * 64 if target == "pseudo-a" else "b" * 64,
        "split_role": "train",
        "split_authority_verified": True,
        "spatial_mode": "native",
        "checkpoint_diagnostic_only": False,
        "non_strict_state_loading": False,
        "labels_loaded": True,
        "num_images": len(records),
        "records": records,
        "records_sha256": score_records_sha256(records),
        "ordered_image_ids_sha256": ordered_ids_sha256(image_ids),
        "mask_alignment_schema": SCORE_MASK_ALIGNMENT_SCHEMA,
        "mask_alignment_policy": MASK_ALIGNMENT_POLICY,
        "mask_alignment_count": 0,
        "mask_aligned_sample_ids": [],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _archive_provenance(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    provenance = json.loads(str(arrays["provenance_json"].item()))
    return arrays, provenance


def test_paired_lodo_builds_each_domain_once_and_writes_both_directions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores_a = tmp_path / "pseudo-a"
    scores_b = tmp_path / "pseudo-b"
    _write_scores(scores_a, "pseudo-a")
    _write_scores(scores_b, "pseudo-b")
    grid = tmp_path / "thresholds.npy"
    np.save(grid, np.asarray([0.25, 0.50, 0.75], dtype=np.float32))
    primary = tmp_path / "val-b"
    reverse = tmp_path / "val-a"

    calls: list[str] = []
    original = episode_module._episodes_for_files

    def counted(*args, **kwargs):
        calls.append(str(args[1]))
        return original(*args, **kwargs)

    monkeypatch.setattr(episode_module, "_episodes_for_files", counted)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_curve_episodes",
            "--score-map-dir",
            str(scores_a),
            "--pseudo-target",
            "pseudo-a",
            "--score-map-dir",
            str(scores_b),
            "--pseudo-target",
            "pseudo-b",
            "--validation-domain",
            "pseudo-b",
            "--output-dir",
            str(primary),
            "--paired-output-dir",
            str(reverse),
            "--threshold-grid",
            str(grid),
            "--adaptation-window",
            "1",
            "--evaluation-window",
            "1",
            "--stride",
            "2",
        ],
    )

    episode_module.main()

    assert calls == ["pseudo-a", "pseudo-b"]
    primary_manifest = json.loads(
        (primary / "manifest.json").read_text(encoding="utf-8")
    )
    reverse_manifest = json.loads(
        (reverse / "manifest.json").read_text(encoding="utf-8")
    )
    assert primary_manifest["validation_domain"] == "pseudo-b"
    assert reverse_manifest["validation_domain"] == "pseudo-a"
    assert primary_manifest["num_train_episodes"] == 2
    assert primary_manifest["num_val_episodes"] == 2
    assert reverse_manifest["num_train_episodes"] == 2
    assert reverse_manifest["num_val_episodes"] == 2
    assert primary_manifest["split_summary"]["pseudo-a"]["split"] == "train_domain"
    assert primary_manifest["split_summary"]["pseudo-a"]["train_episodes"] == 2
    assert primary_manifest["split_summary"]["pseudo-a"]["val_episodes"] == 0
    assert primary_manifest["split_summary"]["pseudo-b"]["split"] == "validation_domain"
    assert reverse_manifest["split_summary"]["pseudo-a"]["split"] == "validation_domain"
    assert reverse_manifest["split_summary"]["pseudo-b"]["split"] == "train_domain"
    assert primary_manifest["paired_lodo"] is True
    assert primary_manifest["formal_protocol_eligible"] is True
    assert reverse_manifest["formal_protocol_eligible"] is True
    assert primary_manifest["paired_lodo_role"] == "primary"
    assert reverse_manifest["paired_lodo_role"] == "reverse"
    assert primary_manifest["paired_lodo_peer_output_dir"] == str(reverse.resolve())
    assert reverse_manifest["paired_lodo_peer_output_dir"] == str(primary.resolve())

    primary_train, primary_train_provenance = _archive_provenance(
        primary / "train.npz"
    )
    primary_val, primary_val_provenance = _archive_provenance(primary / "val.npz")
    reverse_train, reverse_train_provenance = _archive_provenance(
        reverse / "train.npz"
    )
    reverse_val, reverse_val_provenance = _archive_provenance(reverse / "val.npz")
    assert primary_train["pseudo_targets"].tolist() == ["pseudo-a", "pseudo-a"]
    assert primary_val["pseudo_targets"].tolist() == ["pseudo-b", "pseudo-b"]
    assert reverse_train["pseudo_targets"].tolist() == ["pseudo-b", "pseudo-b"]
    assert reverse_val["pseudo_targets"].tolist() == ["pseudo-a", "pseudo-a"]
    np.testing.assert_array_equal(primary_train["statistics"], reverse_val["statistics"])
    np.testing.assert_array_equal(primary_val["statistics"], reverse_train["statistics"])
    np.testing.assert_array_equal(primary_train["pixel_fp_counts"], reverse_val["pixel_fp_counts"])
    np.testing.assert_array_equal(primary_val["component_fp_counts"], reverse_train["component_fp_counts"])

    assert primary_train_provenance["validation_domain"] == "pseudo-b"
    assert primary_train_provenance["archive_split"] == "train"
    assert primary_train_provenance["num_archive_episodes"] == 2
    assert primary_train_provenance["num_train_episodes"] == 2
    assert primary_train_provenance["num_val_episodes"] == 2
    assert primary_val_provenance["archive_split"] == "validation"
    assert reverse_train_provenance["validation_domain"] == "pseudo-a"
    assert reverse_train_provenance["archive_split"] == "train"
    assert reverse_val_provenance["archive_split"] == "validation"
    assert reverse_val_provenance["split_summary"] == reverse_manifest["split_summary"]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            [
                "--score-map-dir",
                "pseudo-a",
                "--score-map-dir",
                "pseudo-b",
                "--output-dir",
                "out-a",
                "--paired-output-dir",
                "out-b",
            ],
            "requires --validation-domain",
        ),
        (
            [
                "--score-map-dir",
                "pseudo-a",
                "--validation-domain",
                "pseudo-a",
                "--output-dir",
                "out-a",
                "--paired-output-dir",
                "out-b",
            ],
            "requires exactly two pseudo targets",
        ),
        (
            [
                "--score-map-dir",
                "nudt-a",
                "--pseudo-target",
                "NUDT",
                "--score-map-dir",
                "nudt-b",
                "--pseudo-target",
                "NUDT-SIRST",
                "--validation-domain",
                "NUDT",
                "--output-dir",
                "out-a",
                "--paired-output-dir",
                "out-b",
            ],
            "requires two distinct pseudo-target domains",
        ),
        (
            [
                "--score-map-dir",
                "pseudo-a",
                "--score-map-dir",
                "pseudo-b",
                "--score-map-dir",
                "pseudo-c",
                "--validation-domain",
                "pseudo-b",
                "--output-dir",
                "out-a",
                "--paired-output-dir",
                "out-b",
            ],
            "requires exactly two pseudo targets",
        ),
        (
            [
                "--score-map-dir",
                "pseudo-a",
                "--score-map-dir",
                "pseudo-b",
                "--validation-domain",
                "pseudo-b",
                "--output-dir",
                "same-output",
                "--paired-output-dir",
                "same-output",
            ],
            "must differ from --output-dir",
        ),
    ],
)
def test_paired_lodo_cli_fails_closed(
    argv: list[str],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["build_curve_episodes", *argv])

    with pytest.raises(ValueError, match=message):
        episode_module.main()
