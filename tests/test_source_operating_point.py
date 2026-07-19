from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data_ext.mask_alignment import MASK_ALIGNMENT_POLICY
from evaluation.artifact_integrity import (
    SCORE_MANIFEST_SCHEMA_VERSION,
    SCORE_MASK_ALIGNMENT_SCHEMA,
    SCORE_RECORD_INTEGRITY_SCHEMA,
    file_sha256,
    ordered_ids_sha256,
    score_records_sha256,
)
from evaluation.source_operating_point import (
    load_formal_curve,
    main as source_operating_point_main,
    select_source_operating_points,
)
from evaluation.threshold_sweep import (
    CURVE_METADATA_ARTIFACT_TYPE,
    curve_metadata_path,
    main as threshold_sweep_main,
)


def _write_formal_score_artifact(
    root: Path,
    *,
    target_dataset: str,
    source_datasets: list[str],
    split_role: str,
    probability: np.ndarray,
    mask: np.ndarray,
    weight_digit: str,
    model_backend: str = "canonical",
) -> Path:
    root.mkdir(parents=True)
    image_id = f"{target_dataset}-sample"
    record_path = root / "sample.npz"
    np.savez_compressed(
        record_path,
        prob=np.asarray(probability, dtype=np.float32),
        gray=np.zeros_like(probability, dtype=np.float32),
        mask=np.asarray(mask > 0, dtype=np.uint8),
        image_id=np.asarray(image_id),
        labels_loaded=np.asarray(True),
        original_hw=np.asarray(probability.shape, dtype=np.int32),
        input_hw=np.asarray(probability.shape, dtype=np.int32),
        valid_hw=np.asarray(probability.shape, dtype=np.int32),
        padding_ltrb=np.asarray([0, 0, 0, 0], dtype=np.int32),
        spatial_mode=np.asarray("native"),
        mask_alignment_applied=np.asarray(False),
        mask_original_hw=np.asarray(mask.shape, dtype=np.int32),
        mask_aspect_relative_error=np.asarray(0.0),
        mask_alignment_policy=np.asarray(MASK_ALIGNMENT_POLICY),
    )
    record = {
        "image_id": image_id,
        "file": record_path.name,
        "shape": list(probability.shape),
        "sha256": file_sha256(record_path),
        "mask_alignment_applied": False,
        "mask_original_hw": list(mask.shape),
        "mask_aspect_relative_error": 0.0,
    }
    split_file = root / "frozen_split.txt"
    split_file.write_text(image_id + "\n", encoding="utf-8")
    ids_sha = ordered_ids_sha256([image_id])
    requested_split = split_role
    manifest = {
        "schema_version": SCORE_MANIFEST_SCHEMA_VERSION,
        "record_integrity_schema": SCORE_RECORD_INTEGRITY_SCHEMA,
        "score_type": "sigmoid_probability",
        "warm_flag": True,
        "labels_loaded": True,
        "num_images": 1,
        "records": [record],
        "records_sha256": score_records_sha256([record]),
        "ordered_image_ids_sha256": ids_sha,
        "mask_alignment_schema": SCORE_MASK_ALIGNMENT_SCHEMA,
        "mask_alignment_policy": MASK_ALIGNMENT_POLICY,
        "mask_alignment_count": 0,
        "mask_aligned_sample_ids": [],
        "target_dataset": target_dataset,
        "source_datasets": source_datasets,
        "weight_sha256": weight_digit * 64,
        "checkpoint_selection_rule": "fixed_last",
        "checkpoint_diagnostic_only": False,
        "non_strict_state_loading": False,
        "model_backend": model_backend,
        "requested_split": requested_split,
        "split_role": split_role,
        "split_authority_verified": True,
        "split_file": str(split_file.resolve()),
        "split_file_sha256": file_sha256(split_file),
        "split_ordered_ids_sha256": ids_sha,
        "spatial_mode": "native",
        "pad_multiple": 16,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return root


def _curve_from_score_artifact(
    tmp_path: Path,
    *,
    name: str,
    target_dataset: str,
    source_datasets: list[str],
    split_role: str,
    target_score: float,
    false_score: float | None,
    thresholds: str = "0.5,0.7",
    weight_digit: str = "a",
) -> Path:
    probability = np.zeros((10, 10), dtype=np.float32)
    probability[0, 0] = target_score
    if false_score is not None:
        probability[9, 9] = false_score
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, 0] = 1
    score_dir = _write_formal_score_artifact(
        tmp_path / f"{name}-scores",
        target_dataset=target_dataset,
        source_datasets=source_datasets,
        split_role=split_role,
        probability=probability,
        mask=mask,
        weight_digit=weight_digit,
    )
    curve = tmp_path / f"{name}.csv"
    assert threshold_sweep_main(
        [
            "--score-dir",
            str(score_dir),
            "--output",
            str(curve),
            "--thresholds",
            thresholds,
            "--formal",
            "--expected-split-role",
            split_role,
        ]
    ) == 0
    return curve


def _source_curves(tmp_path: Path) -> tuple[Path, Path]:
    source_a = _curve_from_score_artifact(
        tmp_path,
        name="source-a",
        target_dataset="NUAA-SIRST",
        source_datasets=["NUDT-SIRST"],
        split_role="train",
        target_score=0.8,
        false_score=0.6,
        weight_digit="a",
    )
    source_b = _curve_from_score_artifact(
        tmp_path,
        name="source-b",
        target_dataset="NUDT-SIRST",
        source_datasets=["NUAA-SIRST"],
        split_role="train",
        target_score=0.65,
        false_score=None,
        weight_digit="b",
    )
    return source_a, source_b


def test_formal_sweep_writes_hash_bound_atomic_sidecar(tmp_path: Path) -> None:
    curve, _ = _source_curves(tmp_path)
    sidecar = curve_metadata_path(curve)
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metadata["artifact_type"] == CURVE_METADATA_ARTIFACT_TYPE
    assert metadata["formal_protocol_eligible"] is True
    assert metadata["curve_sha256"] == file_sha256(curve)
    assert metadata["score_manifest_sha256"] == file_sha256(
        tmp_path / "source-a-scores" / "manifest.json"
    )
    assert metadata["score_records_sha256"]
    assert metadata["target_domain_key"] == "nuaa"
    assert metadata["source_domain_keys"] == ["nudt"]
    assert metadata["checkpoint_selection_rule"] == "fixed_last"
    assert metadata["model_backend"] == "canonical"
    assert metadata["split_role"] == "train"
    assert not list(tmp_path.rglob("*.tmp"))


def test_formal_sweep_preserves_rc_mshnet_backend_and_rejects_compact(
    tmp_path: Path,
) -> None:
    probability = np.asarray([[0.8, 0.1], [0.1, 0.1]], dtype=np.float32)
    mask = np.asarray([[1, 0], [0, 0]], dtype=np.uint8)
    rc_scores = _write_formal_score_artifact(
        tmp_path / "rc-scores",
        target_dataset="NUAA-SIRST",
        source_datasets=["NUDT-SIRST"],
        split_role="train",
        probability=probability,
        mask=mask,
        weight_digit="d",
        model_backend="rc_mshnet",
    )
    curve = tmp_path / "rc.csv"
    assert threshold_sweep_main(
        [
            "--score-dir",
            str(rc_scores),
            "--output",
            str(curve),
            "--thresholds",
            "0.5",
            "--formal",
            "--expected-split-role",
            "train",
        ]
    ) == 0
    metadata = json.loads(
        curve_metadata_path(curve).read_text(encoding="utf-8")
    )
    assert metadata["model_backend"] == "rc_mshnet"

    compact_scores = _write_formal_score_artifact(
        tmp_path / "compact-scores",
        target_dataset="IRSTD-1K",
        source_datasets=["NUDT-SIRST"],
        split_role="train",
        probability=probability,
        mask=mask,
        weight_digit="e",
        model_backend="compact",
    )
    with pytest.raises(ValueError, match="model_backend"):
        threshold_sweep_main(
            [
                "--score-dir",
                str(compact_scores),
                "--output",
                str(tmp_path / "compact.csv"),
                "--thresholds",
                "0.5",
                "--formal",
                "--expected-split-role",
                "train",
            ]
        )


def test_formal_sweep_rejects_alias_seen_target_and_diagnostics(tmp_path: Path) -> None:
    probability = np.zeros((2, 2), dtype=np.float32)
    mask = np.zeros_like(probability, dtype=np.uint8)
    seen = _write_formal_score_artifact(
        tmp_path / "seen",
        target_dataset="NUAA-SIRST",
        source_datasets=["nuaa"],
        split_role="train",
        probability=probability,
        mask=mask,
        weight_digit="c",
    )
    with pytest.raises(ValueError, match="held out"):
        threshold_sweep_main(
            [
                "--score-dir",
                str(seen),
                "--output",
                str(tmp_path / "seen.csv"),
                "--thresholds",
                "0.5",
                "--formal",
                "--expected-split-role",
                "train",
            ]
        )

    clean_manifest_path = seen / "manifest.json"
    manifest = json.loads(clean_manifest_path.read_text(encoding="utf-8"))
    manifest["target_dataset"] = "IRSTD-1K"
    manifest["checkpoint_diagnostic_only"] = True
    clean_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint_diagnostic_only"):
        threshold_sweep_main(
            [
                "--score-dir",
                str(seen),
                "--output",
                str(tmp_path / "diagnostic.csv"),
                "--thresholds",
                "0.5",
                "--formal",
                "--expected-split-role",
                "train",
            ]
        )


def test_source_selection_pooled_worst_and_target_free_contract(tmp_path: Path) -> None:
    source_a, source_b = _source_curves(tmp_path)
    no_target_output = tmp_path / "source-only.json"
    base_args = [
        "--source-curve",
        f"NUAA={source_a}",
        "--source-curve",
        "NUDT-SIRST",
        str(source_b),
        "--pixel-budget",
        "0.005",
        "--component-budget",
        "5000",
    ]
    assert source_operating_point_main(
        [*base_args, "--output", str(no_target_output)]
    ) == 0
    source_only = json.loads(no_target_output.read_text(encoding="utf-8"))
    assert source_only["pseudo_target_lodo_closure_verified"] is True
    assert source_only["selection_is_source_only"] is True
    assert source_only["results"]["source_pooled"]["operating_point"][
        "threshold"
    ] == pytest.approx(0.5)
    assert source_only["results"]["source_worst"]["operating_point"][
        "threshold"
    ] == pytest.approx(0.7)
    assert source_only["results"]["source_worst"]["worst_domain_pd"] == 0.0

    target_one = _curve_from_score_artifact(
        tmp_path,
        name="target-one",
        target_dataset="IRSTD-1K",
        source_datasets=["NUAA-SIRST", "NUDT-SIRST"],
        split_role="test",
        target_score=0.9,
        false_score=0.8,
        weight_digit="c",
    )
    target_two = _curve_from_score_artifact(
        tmp_path,
        name="target-two",
        target_dataset="IRSTD-1K",
        source_datasets=["nuaa", "nudt"],
        split_role="test",
        target_score=0.55,
        false_score=None,
        weight_digit="d",
    )
    payloads = []
    for index, target in enumerate((target_one, target_two)):
        output = tmp_path / f"with-target-{index}.json"
        assert source_operating_point_main(
            [
                *base_args,
                "--target-curve",
                str(target),
                "--output",
                str(output),
            ]
        ) == 0
        payloads.append(json.loads(output.read_text(encoding="utf-8")))
    assert payloads[0]["results"] == source_only["results"]
    assert payloads[1]["results"] == source_only["results"]
    assert payloads[0]["target_evaluation"]["target_labels_used"] is True
    assert (
        payloads[0]["target_evaluation"]["target_labels_used_for_selection"]
        is False
    )
    assert payloads[0]["target_evaluation"][
        "selection_frozen_before_target_load"
    ] is True
    assert payloads[0]["target_evaluation"][
        "rows_at_source_selected_thresholds"
    ] != payloads[1]["target_evaluation"]["rows_at_source_selected_thresholds"]


def test_source_selection_rejects_tampering_grid_mismatch_and_bad_budget(
    tmp_path: Path,
) -> None:
    source_a, source_b = _source_curves(tmp_path)
    source_a.write_bytes(source_a.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="CSV sha256 mismatch"):
        load_formal_curve(source_a, expected_split_role="train")

    source_b_record = tmp_path / "source-b-scores" / "sample.npz"
    source_b_record.write_bytes(source_b_record.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_formal_curve(source_b, expected_split_role="train")

    grid_root = tmp_path / "grid-mismatch"
    good_a, _ = _source_curves(grid_root)
    mismatch_b = _curve_from_score_artifact(
        grid_root,
        name="mismatch-b",
        target_dataset="NUDT-SIRST",
        source_datasets=["NUAA-SIRST"],
        split_role="train",
        target_score=0.65,
        false_score=None,
        thresholds="0.5,0.8",
        weight_digit="e",
    )
    with pytest.raises(ValueError, match="identical threshold grid"):
        source_operating_point_main(
            [
                "--source-curve",
                f"NUAA={good_a}",
                "--source-curve",
                f"NUDT={mismatch_b}",
                "--pixel-budget",
                "0.005",
                "--component-budget",
                "5000",
                "--output",
                str(tmp_path / "bad-grid.json"),
            ]
        )

    valid_rows = load_formal_curve(good_a, expected_split_role="train").rows
    with pytest.raises(ValueError, match="strictly positive"):
        select_source_operating_points(
            {"a": valid_rows, "b": valid_rows},
            pixel_budget=0.0,
            component_budget=1.0,
        )


def test_source_selection_requires_complete_lodo_closure_and_unique_aliases(
    tmp_path: Path,
) -> None:
    source_a, _ = _source_curves(tmp_path)
    incomplete = _curve_from_score_artifact(
        tmp_path,
        name="incomplete-b",
        target_dataset="NUDT-SIRST",
        source_datasets=["IRSTD-1K"],
        split_role="train",
        target_score=0.65,
        false_score=None,
        weight_digit="f",
    )
    with pytest.raises(ValueError, match="LODO closure"):
        source_operating_point_main(
            [
                "--source-curve",
                f"NUAA={source_a}",
                "--source-curve",
                f"NUDT={incomplete}",
                "--pixel-budget",
                "0.005",
                "--component-budget",
                "5000",
                "--output",
                str(tmp_path / "incomplete.json"),
            ]
        )

    with pytest.raises(ValueError, match="Duplicate source curve domain alias"):
        source_operating_point_main(
            [
                "--source-curve",
                f"NUAA={source_a}",
                "--source-curve",
                f"nuaa-sirst={source_a}",
                "--pixel-budget",
                "0.005",
                "--component-budget",
                "5000",
                "--output",
                str(tmp_path / "duplicate.json"),
            ]
        )
