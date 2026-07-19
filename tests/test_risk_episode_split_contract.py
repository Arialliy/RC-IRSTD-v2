from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import risk_curve.build_curve_episodes as curve_episode_module
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
    audit_fold_score_manifest,
    main as build_curve_episodes_main,
)


def _manifest(
    *,
    target: str = "pseudo-a",
    split_role: str = "train",
) -> dict[str, object]:
    return {
        "target_dataset": target,
        "source_datasets": ["independent-source"],
        "weight_sha256": "a" * 64,
        "split_role": split_role,
        "split_authority_verified": True,
        "spatial_mode": "native",
        "checkpoint_diagnostic_only": False,
        "non_strict_state_loading": False,
    }


def _write_manifest(root: Path, manifest: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _write_score_directory(root: Path, *, target: str, split_role: str) -> None:
    manifest = _manifest(target=target, split_role=split_role)
    records: list[dict[str, object]] = []
    image_ids: list[str] = []
    for index in range(4):
        image_id = f"{target}-{index}"
        filename = f"{index:03d}.npz"
        probability = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
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
    manifest.update(
        {
            "schema_version": SCORE_MANIFEST_SCHEMA_VERSION,
            "record_integrity_schema": SCORE_RECORD_INTEGRITY_SCHEMA,
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
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_library_audit_without_expected_split_keeps_legacy_compatibility(
    tmp_path: Path,
) -> None:
    root = tmp_path / "scores"
    legacy = _manifest()
    for field in (
        "split_role",
        "split_authority_verified",
        "spatial_mode",
        "checkpoint_diagnostic_only",
        "non_strict_state_loading",
    ):
        legacy.pop(field)
    _write_manifest(root, legacy)

    audit = audit_fold_score_manifest(root, "pseudo-a")

    assert audit["verified"] is True
    assert audit["expected_split_role"] is None
    assert audit["split_role"] is None


def test_explicit_train_split_requires_strict_native_artifact(tmp_path: Path) -> None:
    root = tmp_path / "scores"
    _write_manifest(root, _manifest(split_role="train"))

    audit = audit_fold_score_manifest(
        root, "pseudo-a", expected_split_role="train"
    )

    assert audit["verified"] is True
    assert audit["expected_split_role"] == "train"
    assert audit["split_role"] == "train"
    assert audit["split_authority_verified"] is True
    assert audit["spatial_mode"] == "native"
    assert audit["checkpoint_diagnostic_only"] is False
    assert audit["non_strict_state_loading"] is False


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("split_role", "test", "split_role_mismatch"),
        ("split_authority_verified", False, "split_authority_unverified"),
        ("spatial_mode", "resize", "non_native_spatial_mode"),
        ("checkpoint_diagnostic_only", True, "checkpoint_diagnostic_only"),
        ("non_strict_state_loading", True, "non_strict_state_loading"),
    ],
)
def test_explicit_split_audit_marks_nonformal_artifacts_unverified(
    tmp_path: Path,
    field: str,
    value: object,
    reason: str,
) -> None:
    root = tmp_path / field
    manifest = _manifest()
    manifest[field] = value
    _write_manifest(root, manifest)

    audit = audit_fold_score_manifest(
        root, "pseudo-a", expected_split_role="train"
    )

    assert audit["verified"] is False
    assert audit["reason"] == reason
    assert audit["detector_weight_sha256"] == "a" * 64


def test_main_defaults_to_source_train_and_records_split_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_scores = tmp_path / "pseudo-a"
    validation_scores = tmp_path / "pseudo-b"
    train_scores.mkdir()
    validation_scores.mkdir()
    _write_score_directory(train_scores, target="pseudo-a", split_role="train")
    _write_score_directory(
        validation_scores, target="pseudo-b", split_role="train"
    )
    output = tmp_path / "episodes"
    monkeypatch.setattr(
        curve_episode_module,
        "score_files",
        lambda _root: (_ for _ in ()).throw(
            AssertionError("formal path must use verifier-returned paths")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_curve_episodes",
            "--score-map-dir",
            str(train_scores),
            "--pseudo-target",
            "pseudo-a",
            "--score-map-dir",
            str(validation_scores),
            "--pseudo-target",
            "pseudo-b",
            "--validation-domain",
            "pseudo-b",
            "--adaptation-window",
            "1",
            "--evaluation-window",
            "1",
            "--stride",
            "2",
            "--output-dir",
            str(output),
        ],
    )

    build_curve_episodes_main()

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["expected_split_role"] == "train"
    assert manifest["pseudo_target_split"] == "train"
    assert manifest["pseudo_target_splits"] == {
        "pseudo-a": "train",
        "pseudo-b": "train",
    }
    assert manifest["fold_provenance_verified"] is True
    assert manifest["score_artifact_integrity_verified"] is True
    assert all(
        audit["mask_alignment_verified"]
        for audit in manifest["score_artifact_integrity_audits"]
    )
    assert manifest["formal_protocol_eligible"] is True
    with np.load(output / "train.npz", allow_pickle=False) as archive:
        provenance = json.loads(str(np.asarray(archive["provenance_json"]).item()))
    assert provenance["expected_split_role"] == "train"
    assert provenance["pseudo_target_splits"]["pseudo-a"] == "train"


def test_cli_split_mismatch_is_strictly_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scores = tmp_path / "scores"
    scores.mkdir()
    _write_score_directory(scores, target="pseudo-a", split_role="test")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_curve_episodes",
            "--score-map-dir",
            str(scores),
            "--pseudo-target",
            "pseudo-a",
            "--output-dir",
            str(tmp_path / "episodes"),
        ],
    )

    with pytest.raises(ValueError, match="pseudo-a:split_role_mismatch"):
        build_curve_episodes_main()


def test_formal_episode_builder_rejects_tampered_record_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scores = tmp_path / "scores"
    scores.mkdir()
    _write_score_directory(scores, target="pseudo-a", split_role="train")
    record = scores / "000.npz"
    record.write_bytes(record.read_bytes() + b"tampered")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_curve_episodes",
            "--score-map-dir",
            str(scores),
            "--pseudo-target",
            "pseudo-a",
            "--output-dir",
            str(tmp_path / "episodes"),
        ],
    )

    with pytest.raises(ValueError, match="sha256 mismatch"):
        build_curve_episodes_main()


def test_formal_episode_builder_rejects_missing_alignment_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scores = tmp_path / "scores"
    scores.mkdir()
    _write_score_directory(scores, target="pseudo-a", split_role="train")
    record_path = scores / "000.npz"
    with np.load(record_path, allow_pickle=False) as archive:
        payload = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key != "mask_alignment_policy"
        }
    np.savez_compressed(record_path, **payload)
    manifest_path = scores / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"][0]["sha256"] = file_sha256(record_path)
    manifest["records_sha256"] = score_records_sha256(manifest["records"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_curve_episodes",
            "--score-map-dir",
            str(scores),
            "--pseudo-target",
            "pseudo-a",
            "--output-dir",
            str(tmp_path / "episodes"),
        ],
    )

    with pytest.raises(ValueError, match="lacks alignment evidence"):
        build_curve_episodes_main()


def test_formal_episode_builder_rejects_unlabeled_score_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scores = tmp_path / "scores"
    scores.mkdir()
    _write_score_directory(scores, target="pseudo-a", split_role="train")
    manifest_path = scores / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["labels_loaded"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_curve_episodes",
            "--score-map-dir",
            str(scores),
            "--pseudo-target",
            "pseudo-a",
            "--output-dir",
            str(tmp_path / "episodes"),
        ],
    )

    with pytest.raises(ValueError, match="label mode does not match"):
        build_curve_episodes_main()
