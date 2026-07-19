import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import evaluation.count_all_baseline as count_all_module
from data_ext.mask_alignment import (
    MASK_ALIGNMENT_NOT_LOADED_POLICY,
    MASK_ALIGNMENT_POLICY,
)
from evaluation.artifact_integrity import (
    SCORE_MANIFEST_SCHEMA_VERSION,
    SCORE_MASK_ALIGNMENT_SCHEMA,
    SCORE_RECORD_INTEGRITY_SCHEMA,
    file_sha256,
    ordered_ids_sha256,
    score_records_sha256,
)
from evaluation.count_all_baseline import (
    build_count_all_curves,
    load_probability_records,
    run_count_all_baseline,
)


def _write_score_dir(
    root: Path,
    probabilities: list[np.ndarray],
    *,
    masks: list[np.ndarray] | None = None,
    prefix: str = "image",
) -> list[str]:
    root.mkdir()
    image_ids: list[str] = []
    records = []
    for index, probability in enumerate(probabilities):
        image_id = f"{prefix}-{index}"
        filename = f"record-{index}.npz"
        payload = {
            "prob": np.asarray(probability, dtype=np.float32),
            "image_id": np.asarray(image_id),
        }
        if masks is not None:
            payload["mask"] = np.asarray(masks[index], dtype=np.uint8)
        np.savez_compressed(root / filename, **payload)
        image_ids.append(image_id)
        records.append(
            {
                "image_id": image_id,
                "file": filename,
                "shape": list(np.asarray(probability).shape),
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "score_type": "sigmoid_probability",
                "spatial_mode": "native",
                "target_dataset": "synthetic-native",
                "weight_path": "/frozen/detector/weight.pkl",
                "num_images": len(records),
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    return image_ids


def _write_formal_score_dir(
    root: Path,
    probabilities: list[np.ndarray],
    *,
    masks: list[np.ndarray] | None = None,
    prefix: str = "image",
) -> list[str]:
    root.mkdir()
    labels_loaded = masks is not None
    image_ids: list[str] = []
    records: list[dict[str, object]] = []
    for index, raw_probability in enumerate(probabilities):
        probability = np.asarray(raw_probability, dtype=np.float32)
        height, width = probability.shape
        image_id = f"{prefix}-{index}"
        filename = f"record-{index}.npz"
        payload: dict[str, np.ndarray] = {
            "prob": probability,
            "gray": np.zeros_like(probability),
            "image_id": np.asarray(image_id),
            "labels_loaded": np.asarray(labels_loaded),
            "original_hw": np.asarray([height, width], dtype=np.int32),
            "mask_alignment_applied": np.asarray(False),
            "mask_original_hw": np.asarray(
                [height, width] if labels_loaded else [0, 0], dtype=np.int32
            ),
            "mask_aspect_relative_error": np.asarray(
                0.0 if labels_loaded else -1.0, dtype=np.float64
            ),
            "mask_alignment_policy": np.asarray(
                MASK_ALIGNMENT_POLICY
                if labels_loaded
                else MASK_ALIGNMENT_NOT_LOADED_POLICY
            ),
        }
        if masks is not None:
            payload["mask"] = np.asarray(masks[index], dtype=np.uint8)
        record_path = root / filename
        np.savez_compressed(record_path, **payload)
        records.append(
            {
                "image_id": image_id,
                "file": filename,
                "shape": [height, width],
                "sha256": file_sha256(record_path),
                "mask_alignment_applied": False,
                "mask_original_hw": (
                    [height, width] if labels_loaded else [0, 0]
                ),
                "mask_aspect_relative_error": 0.0 if labels_loaded else -1.0,
            }
        )
        image_ids.append(image_id)
    manifest = {
        "schema_version": SCORE_MANIFEST_SCHEMA_VERSION,
        "record_integrity_schema": SCORE_RECORD_INTEGRITY_SCHEMA,
        "score_type": "sigmoid_probability",
        "labels_loaded": labels_loaded,
        "spatial_mode": "native",
        "target_dataset": "synthetic-native",
        "num_images": len(records),
        "records": records,
        "records_sha256": score_records_sha256(records),
        "ordered_image_ids_sha256": ordered_ids_sha256(image_ids),
        "mask_alignment_schema": SCORE_MASK_ALIGNMENT_SCHEMA,
        "mask_alignment_policy": (
            MASK_ALIGNMENT_POLICY
            if labels_loaded
            else MASK_ALIGNMENT_NOT_LOADED_POLICY
        ),
        "mask_alignment_count": 0,
        "mask_aligned_sample_ids": [],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return image_ids


def test_saturated_probability_one_is_not_treated_as_an_empty_endpoint(tmp_path):
    warmup = tmp_path / "warmup"
    _write_score_dir(
        warmup,
        [np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)],
    )
    result = run_count_all_baseline(
        warmup,
        np.asarray([0.5, 1.0]),
        pixel_budget=0.20,
        component_budget=200_000.0,
    )
    assert result["warmup_curve"]["predicted_pixel_counts"] == [1, 1]
    assert result["warmup_curve"]["component_counts_suffix_max"] == [1, 1]
    assert result["reject"] is True
    assert result["threshold"] is None


def test_no_feasible_grid_point_returns_explicit_reject(tmp_path):
    warmup = tmp_path / "warmup"
    _write_score_dir(warmup, [np.full((3, 3), 0.95, dtype=np.float32)])
    result = run_count_all_baseline(
        warmup,
        np.asarray([0.2, 0.8, 0.9]),
        pixel_budget=1e-3,
        component_budget=1.0,
    )
    assert result["success"] is False
    assert result["reject"] is True
    assert result["threshold_index"] is None
    assert result["reason"] == "no_threshold_satisfies_count_all_upper_bounds"


def test_warmup_selection_does_not_read_or_depend_on_masks_and_supports_native_shapes(
    tmp_path,
):
    probabilities = [
        np.asarray([[0.9, 0.1, 0.1], [0.1, 0.1, 0.1]], dtype=np.float32),
        np.asarray(
            [[0.8, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]],
            dtype=np.float32,
        ),
    ]
    without_masks = tmp_path / "without-masks"
    all_target_masks = tmp_path / "all-target-masks"
    _write_score_dir(without_masks, probabilities)
    _write_score_dir(
        all_target_masks,
        probabilities,
        masks=[np.ones_like(value, dtype=np.uint8) for value in probabilities],
    )
    kwargs = {
        "pixel_budget": 0.12,
        "component_budget": 60_000.0,
    }
    result_without = run_count_all_baseline(
        without_masks, np.asarray([0.2, 0.85]), **kwargs
    )
    result_with = run_count_all_baseline(
        all_target_masks, np.asarray([0.2, 0.85]), **kwargs
    )
    assert result_without["threshold_index"] == result_with["threshold_index"]
    assert result_without["threshold"] == result_with["threshold"]
    assert result_without["warmup_curve"] == result_with["warmup_curve"]
    assert result_without["warmup_curve"]["total_pixels"] == 18
    assert result_without["protocol"]["warmup_masks_read_for_selection"] is False
    assert result_without["provenance"]["warmup"]["manifest_metadata"]["spatial_mode"] == "native"


def test_component_counts_use_per_image_suffix_max_envelope(tmp_path):
    warmup = tmp_path / "warmup"
    # The low-score bridge makes one component at 0.3; removing it produces two
    # components at 0.8.  The conservative curve must therefore be [2, 2].
    _write_score_dir(
        warmup,
        [np.asarray([[0.9, 0.4, 0.9]], dtype=np.float32)],
    )
    records, _ = load_probability_records(warmup)
    curves = build_count_all_curves(
        records,
        np.asarray([0.3, 0.8]),
        connectivity=1,
    )
    np.testing.assert_array_equal(curves["component_counts_raw"], [1, 2])
    np.testing.assert_array_equal(curves["component_counts_suffix_max"], [2, 2])


def test_future_labels_are_audited_after_freeze_and_never_reselect_threshold(tmp_path):
    warmup = tmp_path / "warmup"
    future = tmp_path / "future"
    warmup_ids = _write_score_dir(
        warmup,
        [np.asarray([[0.9, 0.5], [0.0, 0.0]], dtype=np.float32)],
        prefix="warm",
    )
    future_ids = _write_score_dir(
        future,
        [np.full((2, 2), 0.1, dtype=np.float32)],
        masks=[np.asarray([[1, 0], [0, 0]], dtype=np.uint8)],
        prefix="future",
    )
    grid = np.asarray([0.2, 0.8])
    result = run_count_all_baseline(
        warmup,
        grid,
        pixel_budget=0.25,
        component_budget=300_000.0,
        future_score_dir=future,
    )
    future_as_warmup = run_count_all_baseline(
        future,
        grid,
        pixel_budget=0.25,
        component_budget=300_000.0,
    )
    assert result["threshold"] == 0.8
    assert future_as_warmup["threshold"] == 0.2
    assert result["future_audit"]["threshold"] == 0.8
    assert result["future_audit"]["threshold_reselected_on_future"] is False
    assert result["warmup_image_ids"] == warmup_ids
    assert result["future_image_ids"] == future_ids
    assert result["bsr"] == 1.0
    assert result["pd"] == 0.0
    assert result["protocol"]["future_labels_used_for_selection"] is False


def test_formal_mode_records_integrity_and_allows_mask_free_warmup(tmp_path):
    warmup = tmp_path / "warmup"
    future = tmp_path / "future"
    _write_formal_score_dir(
        warmup,
        [np.asarray([[0.9, 0.0], [0.0, 0.0]], dtype=np.float32)],
        prefix="warm",
    )
    _write_formal_score_dir(
        future,
        [np.zeros((2, 2), dtype=np.float32)],
        masks=[np.zeros((2, 2), dtype=np.uint8)],
        prefix="future",
    )

    result = run_count_all_baseline(
        warmup,
        np.asarray([0.5, 1.0]),
        pixel_budget=0.5,
        component_budget=500_000.0,
        future_score_dir=future,
        formal=True,
    )

    assert result["formal_protocol_eligible"] is True
    assert result["integrity_audit"]["verified"] is True
    assert result["integrity_audit"]["warmup"]["labels_loaded"] is False
    assert result["integrity_audit"]["warmup"]["mask_alignment_verified"] is True
    assert result["integrity_audit"]["future"]["labels_loaded"] is True
    assert result["integrity_audit"]["future"]["mask_alignment_verified"] is True
    assert result["protocol"]["warmup_masks_read_for_selection"] is False


def test_formal_warmup_selection_never_loads_mask_array(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    warmup = tmp_path / "warmup"
    _write_formal_score_dir(
        warmup,
        [np.zeros((2, 2), dtype=np.float32)],
        masks=[np.zeros((2, 2), dtype=np.uint8)],
        prefix="warm",
    )
    monkeypatch.setattr(
        count_all_module,
        "_validate_mask",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("warm-up selection loaded a mask")
        ),
    )

    result = run_count_all_baseline(
        warmup,
        np.asarray([0.5, 1.0]),
        pixel_budget=0.5,
        component_budget=500_000.0,
        formal=True,
    )

    assert result["formal_protocol_eligible"] is True
    assert result["protocol"]["warmup_masks_read_for_selection"] is False


def test_formal_count_all_rejects_missing_alignment_evidence(tmp_path):
    warmup = tmp_path / "warmup"
    _write_formal_score_dir(
        warmup,
        [np.zeros((2, 2), dtype=np.float32)],
        prefix="warm",
    )
    manifest_path = warmup / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("mask_alignment_policy")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="mask-alignment provenance is incomplete"):
        run_count_all_baseline(
            warmup,
            np.asarray([0.5, 1.0]),
            pixel_budget=0.5,
            component_budget=500_000.0,
            formal=True,
        )


def test_formal_count_all_rejects_tampered_future_hash(tmp_path):
    warmup = tmp_path / "warmup"
    future = tmp_path / "future"
    _write_formal_score_dir(
        warmup,
        [np.zeros((2, 2), dtype=np.float32)],
        prefix="warm",
    )
    _write_formal_score_dir(
        future,
        [np.zeros((2, 2), dtype=np.float32)],
        masks=[np.zeros((2, 2), dtype=np.uint8)],
        prefix="future",
    )
    record = future / "record-0.npz"
    record.write_bytes(record.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        run_count_all_baseline(
            warmup,
            np.asarray([0.5, 1.0]),
            pixel_budget=0.5,
            component_budget=500_000.0,
            future_score_dir=future,
            formal=True,
        )


def test_formal_count_all_requires_labeled_future_artifact(tmp_path):
    warmup = tmp_path / "warmup"
    future = tmp_path / "future"
    _write_formal_score_dir(
        warmup,
        [np.zeros((2, 2), dtype=np.float32)],
        prefix="warm",
    )
    _write_formal_score_dir(
        future,
        [np.zeros((2, 2), dtype=np.float32)],
        prefix="future",
    )

    with pytest.raises(ValueError, match="label mode does not match"):
        run_count_all_baseline(
            warmup,
            np.asarray([0.5, 1.0]),
            pixel_budget=0.5,
            component_budget=500_000.0,
            future_score_dir=future,
            formal=True,
        )


def test_cli_writes_protocol_ids_metrics_and_manifest_provenance(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    warmup = tmp_path / "warmup"
    future = tmp_path / "future"
    _write_formal_score_dir(
        warmup, [np.zeros((2, 3), dtype=np.float32)], prefix="warm"
    )
    _write_formal_score_dir(
        future,
        [np.zeros((3, 2), dtype=np.float32)],
        masks=[np.zeros((3, 2), dtype=np.uint8)],
        prefix="future",
    )
    grid_path = tmp_path / "grid.npy"
    np.save(grid_path, np.asarray([0.5, 1.0], dtype=np.float32))
    output = tmp_path / "count-all.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.count_all_baseline",
            "--warmup-score-dir",
            str(warmup),
            "--future-score-dir",
            str(future),
            "--formal",
            "--threshold-grid",
            str(grid_path),
            "--pixel-budget",
            "0.1",
            "--component-budget",
            "1",
            "--output",
            str(output),
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["warmup_image_ids"] == ["warm-0"]
    assert result["future_image_ids"] == ["future-0"]
    assert result["bsr"] == 1.0
    assert result["pd"] is None
    assert result["protocol"]["future_threshold_reselection"] is False
    assert result["formal_protocol_eligible"] is True
    assert result["integrity_audit"]["verified"] is True
    assert result["integrity_audit"]["warmup"]["mask_alignment_verified"] is True
    assert result["integrity_audit"]["future"]["mask_alignment_verified"] is True
    assert len(result["provenance"]["warmup"]["manifest_sha256"]) == 64
    assert len(result["provenance"]["future"]["manifest_sha256"]) == 64
    assert len(result["provenance"]["threshold_grid"]["file_sha256"]) == 64
    assert len(
        result["provenance"]["threshold_grid"]["threshold_grid_sha256"]
    ) == 64
