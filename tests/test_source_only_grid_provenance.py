from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data_ext.mask_alignment import MASK_ALIGNMENT_POLICY
from evaluation.artifact_integrity import (
    RAW_LOGIT_SCORE_REPRESENTATION,
    SCORE_MANIFEST_SCHEMA_VERSION,
    SCORE_MASK_ALIGNMENT_SCHEMA,
    SCORE_RECORD_INTEGRITY_SCHEMA,
    file_sha256,
    ordered_ids_sha256,
    score_records_sha256,
)
from risk_curve.build_logit_threshold_grid import (
    DenseTailGridSpec,
    build_logit_threshold_grid_artifact,
)
from risk_curve.representation import (
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_REPRESENTATION,
    empty_action_contract,
    load_logit_grid_artifact,
)


SOURCE_DOMAINS = ("NUDT-SIRST", "IRSTD-1K")
OUTER_TARGET = "NUAA-SIRST"
CHECKPOINT_A_SHA = "a" * 64
CHECKPOINT_B_SHA = "b" * 64
CHECKPOINT_C_SHA = "c" * 64


def _sigmoid_float32(logits: np.ndarray) -> np.ndarray:
    values = logits.astype(np.float64)
    result = np.empty_like(values)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_values = np.exp(values[~nonnegative])
    result[~nonnegative] = exp_values / (1.0 + exp_values)
    return result.astype(np.float32)


def _write_source_score_dir(
    root: Path,
    *,
    target_dataset: str,
    checkpoint_sha: str = CHECKPOINT_A_SHA,
    detector_sources: tuple[str, ...] | None = None,
    image_id_prefix: str | None = None,
) -> Path:
    if detector_sources is None:
        detector_sources = (target_dataset,)
    root.mkdir(parents=True)
    prefix = target_dataset if image_id_prefix is None else image_id_prefix
    image_ids = [f"{prefix}-{index}" for index in range(3)]
    split_path = root / "train.txt"
    split_path.write_text("\n".join(image_ids) + "\n", encoding="utf-8")
    records: list[dict[str, object]] = []
    target_offset = 0.25 if target_dataset.startswith("IRSTD") else 0.0
    for index, image_id in enumerate(image_ids):
        logits = np.linspace(-10.0, 15.0, 36, dtype=np.float32).reshape(6, 6)
        logits = (logits + np.float32(target_offset + index * 0.1)).astype(
            np.float32
        )
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[2, 2] = 1
        probability = _sigmoid_float32(logits)
        filename = f"{index:03d}.npz"
        record_path = root / filename
        np.savez_compressed(
            record_path,
            prob=probability,
            logit=logits,
            gray=np.zeros((6, 6), dtype=np.float32),
            mask=mask,
            image_id=np.asarray(image_id),
            labels_loaded=np.asarray(True),
            spatial_mode=np.asarray("native"),
            original_hw=np.asarray([6, 6], dtype=np.int32),
            mask_alignment_applied=np.asarray(False),
            mask_original_hw=np.asarray([6, 6], dtype=np.int32),
            mask_aspect_relative_error=np.asarray(0.0, dtype=np.float64),
            mask_alignment_policy=np.asarray(MASK_ALIGNMENT_POLICY),
            score_representation=np.asarray(RAW_LOGIT_SCORE_REPRESENTATION),
            probability_dtype=np.asarray("float32"),
            logit_dtype=np.asarray("float32"),
            probability_transform=np.asarray("sigmoid"),
            probability_clipping=np.asarray("none"),
            inference_autocast_enabled=np.asarray(False),
        )
        records.append(
            {
                "image_id": image_id,
                "file": filename,
                "shape": [6, 6],
                "sha256": file_sha256(record_path),
                "mask_alignment_applied": False,
                "mask_original_hw": [6, 6],
                "mask_aspect_relative_error": 0.0,
                "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
                "probability_dtype": "float32",
                "logit_dtype": "float32",
                "probability_transform": "sigmoid",
                "probability_clipping": "none",
                "inference_autocast_enabled": False,
            }
        )
    manifest = {
        "schema_version": SCORE_MANIFEST_SCHEMA_VERSION,
        "record_integrity_schema": SCORE_RECORD_INTEGRITY_SCHEMA,
        "score_type": "sigmoid_probability",
        "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
        "probability_dtype": "float32",
        "logit_dtype": "float32",
        "probability_transform": "sigmoid",
        "probability_clipping": "none",
        "inference_autocast_enabled": False,
        "warm_flag": True,
        "labels_loaded": True,
        "num_images": len(records),
        "records": records,
        "records_sha256": score_records_sha256(records),
        "ordered_image_ids_sha256": ordered_ids_sha256(image_ids),
        "mask_alignment_schema": SCORE_MASK_ALIGNMENT_SCHEMA,
        "mask_alignment_policy": MASK_ALIGNMENT_POLICY,
        "mask_alignment_count": 0,
        "mask_aligned_sample_ids": [],
        "target_dataset": target_dataset,
        "source_datasets": list(detector_sources),
        "weight_sha256": checkpoint_sha,
        "checkpoint_selection_rule": "fixed_last",
        "checkpoint_diagnostic_only": False,
        "diagnostic_only": False,
        "non_strict_state_loading": False,
        "formal_protocol_eligible": True,
        "model_backend": "canonical",
        "split_file": str(split_path.resolve()),
        "split_file_sha256": file_sha256(split_path),
        "split_ordered_ids_sha256": ordered_ids_sha256(image_ids),
        "requested_split": "train",
        "split_role": "train",
        "split_authority_verified": True,
        "spatial_mode": "native",
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return root


def _source_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    nudt = _write_source_score_dir(
        tmp_path / "inner-nudt-scores",
        target_dataset="NUDT-SIRST",
        checkpoint_sha=CHECKPOINT_A_SHA,
    )
    irstd = _write_source_score_dir(
        tmp_path / "inner-irstd-scores",
        target_dataset="IRSTD-1K",
        checkpoint_sha=CHECKPOINT_B_SHA,
    )
    outer_nudt = _write_source_score_dir(
        tmp_path / "outer-nudt-scores",
        target_dataset="NUDT-SIRST",
        checkpoint_sha=CHECKPOINT_C_SHA,
        detector_sources=SOURCE_DOMAINS,
    )
    outer_irstd = _write_source_score_dir(
        tmp_path / "outer-irstd-scores",
        target_dataset="IRSTD-1K",
        checkpoint_sha=CHECKPOINT_C_SHA,
        detector_sources=SOURCE_DOMAINS,
    )
    return nudt, irstd, outer_nudt, outer_irstd


def _small_spec() -> DenseTailGridSpec:
    return DenseTailGridSpec(
        max_grid_points=32,
        bulk_points=8,
        upper_points=8,
        extreme_points=8,
        candidate_points=8,
    )


def _rewrite_manifest(root: Path, **changes: object) -> None:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(changes)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_source_only_grid_pools_all_fold_detectors_self_scored_train_and_roundtrips(
    tmp_path: Path,
) -> None:
    inputs = list(_source_inputs(tmp_path))
    output = tmp_path / "grid"
    manifest = build_logit_threshold_grid_artifact(
        list(reversed(inputs)),
        expected_source_domains=["IRSTD-1K", "NUDT"],
        outer_target=OUTER_TARGET,
        output_dir=output,
        spec=_small_spec(),
    )
    artifact = load_logit_grid_artifact(output)

    assert manifest["schema_version"] == LOGIT_GRID_SCHEMA_VERSION
    assert manifest["representation"] == LOGIT_REPRESENTATION
    assert manifest["grid_source"] == "source_official_train_only"
    assert manifest["builder_version"].endswith(
        "deterministic-midpoint-fallback"
    )
    assert manifest["source_domain_keys"] == ["irstd1k", "nudt"]
    assert manifest["outer_target_key"] == "nuaa"
    assert manifest["outer_target_excluded"] is True
    assert manifest["outer_target_labels_used"] is False
    assert manifest["source_train_masks_used"] is True
    assert manifest["grid_detector_protocol"] == (
        "all_source_only_detector_folds"
    )
    assert manifest["detector_checkpoint_count"] == 3
    assert manifest["detector_checkpoint_sha256s"] == [
        CHECKPOINT_A_SHA,
        CHECKPOINT_B_SHA,
        CHECKPOINT_C_SHA,
    ]
    assert manifest["outer_detector_checkpoint_sha256"] == CHECKPOINT_C_SHA
    assert manifest["episode_detector_checkpoint_sha256s"] == [
        CHECKPOINT_A_SHA,
        CHECKPOINT_B_SHA,
    ]
    assert len(manifest["detector_folds"]) == 3
    assert manifest["empty_action"] == empty_action_contract()
    assert artifact.thresholds.dtype == np.float32
    assert artifact.thresholds.size <= 32
    assert np.isfinite(artifact.thresholds).all()
    assert np.all(np.diff(artifact.thresholds.astype(np.float64)) > 0.0)
    assert artifact.semantic_sha256 == manifest["grid_sha256"]
    assert (output / "threshold_grid.sha256").read_text().strip() == manifest[
        "grid_sha256"
    ]
    assert "Infinity" not in (output / "threshold_grid.json").read_text()
    construction_audit = manifest["construction_audit"]
    assert construction_audit["initial_grid_points"] <= manifest["grid_points"]
    assert construction_audit["refinement_strategy"] == (
        "largest_adjacent_raw_logit_gap_float32_midpoint"
    )
    assert construction_audit["refinement_points_added"] == 0
    assert construction_audit["max_adjacent_logit_gap_before"] == (
        construction_audit["max_adjacent_logit_gap_after"]
    )

    # Each detector self-scores every official source domain on which it was
    # trained. The full-source outer detector contributes two artifacts and
    # each inner fold contributes one.
    for item in manifest["input_score_artifacts"]:
        assert item["target_dataset"] in item["detector_source_datasets"]


def test_grid_is_independent_of_cli_source_directory_order(tmp_path: Path) -> None:
    inputs = list(_source_inputs(tmp_path))
    first = build_logit_threshold_grid_artifact(
        inputs,
        expected_source_domains=SOURCE_DOMAINS,
        outer_target=OUTER_TARGET,
        output_dir=tmp_path / "grid-a",
        spec=_small_spec(),
    )
    second = build_logit_threshold_grid_artifact(
        list(reversed(inputs)),
        expected_source_domains=tuple(reversed(SOURCE_DOMAINS)),
        outer_target=OUTER_TARGET,
        output_dir=tmp_path / "grid-b",
        spec=_small_spec(),
    )
    assert first["grid_sha256"] == second["grid_sha256"]
    assert np.array_equal(
        np.load(tmp_path / "grid-a" / "threshold_grid.npy"),
        np.load(tmp_path / "grid-b" / "threshold_grid.npy"),
    )
    assert first["source_provenance_sha256"] == second["source_provenance_sha256"]


def test_outer_target_must_be_excluded_from_expected_and_scored_sources(
    tmp_path: Path,
) -> None:
    inputs = list(_source_inputs(tmp_path))
    with pytest.raises(ValueError, match="Outer target appears"):
        build_logit_threshold_grid_artifact(
            inputs,
            expected_source_domains=[*SOURCE_DOMAINS, "NUAA"],
            outer_target=OUTER_TARGET,
            output_dir=tmp_path / "bad-grid",
            spec=_small_spec(),
        )


def test_expected_source_set_must_exactly_match_scored_domains(tmp_path: Path) -> None:
    inputs = list(_source_inputs(tmp_path))
    nudt = inputs[0]
    with pytest.raises(ValueError, match="Scored grid-source set differs"):
        build_logit_threshold_grid_artifact(
            [nudt],
            expected_source_domains=SOURCE_DOMAINS,
            outer_target=OUTER_TARGET,
            output_dir=tmp_path / "missing-grid",
            spec=_small_spec(),
        )
    with pytest.raises(
        ValueError,
        match="not in the expected source set|outside expected sources",
    ):
        build_logit_threshold_grid_artifact(
            inputs,
            expected_source_domains=["NUDT-SIRST", "OtherSource"],
            outer_target=OUTER_TARGET,
            output_dir=tmp_path / "wrong-grid",
            spec=_small_spec(),
        )


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"split_role": "test"}, "split_role"),
        ({"requested_split": "test"}, "requested_split"),
        ({"spatial_mode": "resize"}, "spatial_mode"),
        ({"checkpoint_selection_rule": "best"}, "checkpoint_selection_rule"),
        ({"model_backend": "compact"}, "model_backend"),
        ({"checkpoint_diagnostic_only": True}, "checkpoint_diagnostic_only"),
        ({"non_strict_state_loading": True}, "non_strict_state_loading"),
    ],
)
def test_grid_source_contract_rejects_nonformal_modes(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    inputs = list(_source_inputs(tmp_path))
    nudt = inputs[0]
    _rewrite_manifest(nudt, **changes)
    with pytest.raises(ValueError, match=message):
        build_logit_threshold_grid_artifact(
            inputs,
            expected_source_domains=SOURCE_DOMAINS,
            outer_target=OUTER_TARGET,
            output_dir=tmp_path / "bad-grid",
            spec=_small_spec(),
        )


def test_probability_only_v3_input_cannot_enter_logit_grid_builder(
    tmp_path: Path,
) -> None:
    inputs = list(_source_inputs(tmp_path))
    nudt = inputs[0]
    path = nudt / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for field in (
        "score_representation",
        "probability_dtype",
        "logit_dtype",
        "probability_transform",
        "probability_clipping",
        "inference_autocast_enabled",
    ):
        manifest.pop(field)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="score_representation"):
        build_logit_threshold_grid_artifact(
            inputs,
            expected_source_domains=SOURCE_DOMAINS,
            outer_target=OUTER_TARGET,
            output_dir=tmp_path / "probability-grid",
            spec=_small_spec(),
        )


def test_all_source_exports_require_distinct_fold_detector_checkpoints(
    tmp_path: Path,
) -> None:
    inputs = list(_source_inputs(tmp_path))
    _rewrite_manifest(inputs[1], weight_sha256=CHECKPOINT_A_SHA)
    with pytest.raises(
        ValueError,
        match="inconsistent source-domain sets|same source-only fold",
    ):
        build_logit_threshold_grid_artifact(
            inputs,
            expected_source_domains=SOURCE_DOMAINS,
            outer_target=OUTER_TARGET,
            output_dir=tmp_path / "mixed-checkpoint-grid",
            spec=_small_spec(),
        )


def test_detector_source_set_cannot_include_outer_target(tmp_path: Path) -> None:
    inputs = list(_source_inputs(tmp_path))
    _rewrite_manifest(
        inputs[2],
        source_datasets=("NUDT-SIRST", "IRSTD-1K", "NUAA-SIRST"),
    )
    with pytest.raises(ValueError, match="outside expected sources|Outer target"):
        build_logit_threshold_grid_artifact(
            inputs,
            expected_source_domains=SOURCE_DOMAINS,
            outer_target=OUTER_TARGET,
            output_dir=tmp_path / "outer-leak-grid",
            spec=_small_spec(),
        )


def test_outer_target_cannot_appear_in_source_path_or_record_ids(
    tmp_path: Path,
) -> None:
    path_inputs = list(_source_inputs(tmp_path / "path-case"))
    leaked_path = _write_source_score_dir(
        tmp_path / "nuaa-source-scores",
        target_dataset="NUDT-SIRST",
        checkpoint_sha=CHECKPOINT_A_SHA,
    )
    path_inputs[0] = leaked_path
    with pytest.raises(ValueError, match="artifact path"):
        build_logit_threshold_grid_artifact(
            path_inputs,
            expected_source_domains=SOURCE_DOMAINS,
            outer_target=OUTER_TARGET,
            output_dir=tmp_path / "path-leak-grid",
            spec=_small_spec(),
        )

    id_inputs = list(_source_inputs(tmp_path / "id-case"))
    clean_nudt = _write_source_score_dir(
        tmp_path / "clean-nudt-scores",
        target_dataset="NUDT-SIRST",
        checkpoint_sha=CHECKPOINT_A_SHA,
        image_id_prefix="NUAA-leaked-id",
    )
    id_inputs[0] = clean_nudt
    with pytest.raises(ValueError, match="record IDs"):
        build_logit_threshold_grid_artifact(
            id_inputs,
            expected_source_domains=SOURCE_DOMAINS,
            outer_target=OUTER_TARGET,
            output_dir=tmp_path / "id-leak-grid",
            spec=_small_spec(),
        )


def test_grid_loader_rejects_tampered_npy(tmp_path: Path) -> None:
    inputs = list(_source_inputs(tmp_path))
    output = tmp_path / "grid"
    build_logit_threshold_grid_artifact(
        inputs,
        expected_source_domains=SOURCE_DOMAINS,
        outer_target=OUTER_TARGET,
        output_dir=output,
        spec=_small_spec(),
    )
    grid_path = output / "threshold_grid.npy"
    values = np.load(grid_path, allow_pickle=False)
    values = values.copy()
    values[0] = np.nextafter(values[0], np.float32(-np.inf))
    np.save(grid_path, values, allow_pickle=False)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_logit_grid_artifact(output)
