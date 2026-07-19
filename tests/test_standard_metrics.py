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
from evaluation.standard_metrics import (
    compute_standard_metrics,
    evaluate_standard_score_directory,
    main,
)


def _formal_score_directory(
    root: Path,
    probabilities: list[np.ndarray],
    masks: list[np.ndarray],
    *,
    model_backend: str = "canonical",
) -> Path:
    root.mkdir()
    image_ids = [f"image-{index}" for index in range(len(probabilities))]
    records = []
    for image_id, probability, mask in zip(image_ids, probabilities, masks):
        filename = f"{image_id}.npz"
        height, width = probability.shape
        np.savez_compressed(
            root / filename,
            prob=np.asarray(probability, dtype=np.float32),
            gray=np.zeros((height, width), dtype=np.float32),
            mask=np.asarray(mask, dtype=np.uint8),
            image_id=np.asarray(image_id),
            labels_loaded=np.asarray(True),
            spatial_mode=np.asarray("native"),
            original_hw=np.asarray([height, width], dtype=np.int32),
            mask_alignment_applied=np.asarray(False),
            mask_original_hw=np.asarray([height, width], dtype=np.int32),
            mask_aspect_relative_error=np.asarray(0.0, dtype=np.float64),
            mask_alignment_policy=np.asarray(MASK_ALIGNMENT_POLICY),
        )
        records.append(
            {
                "image_id": image_id,
                "file": filename,
                "shape": [height, width],
                "sha256": file_sha256(root / filename),
                "mask_alignment_applied": False,
                "mask_original_hw": [height, width],
                "mask_aspect_relative_error": 0.0,
            }
        )
    split_file = root / "official-test.txt"
    split_file.write_text("\n".join(image_ids) + "\n", encoding="utf-8")
    ids_sha = ordered_ids_sha256(image_ids)
    manifest = {
        "schema_version": SCORE_MANIFEST_SCHEMA_VERSION,
        "record_integrity_schema": SCORE_RECORD_INTEGRITY_SCHEMA,
        "score_type": "sigmoid_probability",
        "warm_flag": True,
        "labels_loaded": True,
        "num_images": len(records),
        "records": records,
        "records_sha256": score_records_sha256(records),
        "ordered_image_ids_sha256": ids_sha,
        "target_dataset": "held-out-target",
        "source_datasets": ["source-a", "source-b"],
        "split_file": str(split_file.resolve()),
        "split_file_sha256": file_sha256(split_file),
        "split_ordered_ids_sha256": ids_sha,
        "requested_split": "test",
        "split_role": "test",
        "split_authority_verified": True,
        "spatial_mode": "native",
        "base_hw": None,
        "pad_multiple": 16,
        "mask_alignment_schema": SCORE_MASK_ALIGNMENT_SCHEMA,
        "mask_alignment_policy": MASK_ALIGNMENT_POLICY,
        "mask_alignment_count": 0,
        "mask_aligned_sample_ids": [],
        "weight_sha256": "a" * 64,
        "checkpoint_selection_rule": "fixed_last",
        "checkpoint_diagnostic_only": False,
        "model_backend": model_backend,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return root


def _metric_example() -> tuple[list[np.ndarray], list[np.ndarray]]:
    first_probability = np.asarray(
        [[0.5, 0.1, 0.1], [0.1, 0.1, 0.9]], dtype=np.float32
    )
    first_mask = np.asarray([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    empty_probability = np.full((2, 3), 0.1, dtype=np.float32)
    empty_mask = np.zeros((2, 3), dtype=np.uint8)
    return [first_probability, empty_probability], [first_mask, empty_mask]


def test_standard_metrics_use_greater_equal_and_define_empty_empty_niou() -> None:
    probabilities, masks = _metric_example()
    result = compute_standard_metrics(probabilities, masks, 0.5)
    metrics = result["metrics"]
    counts = result["counts"]
    assert counts["pixel_tp"] == 1  # the true-positive probability equals 0.5
    assert counts["pixel_fp"] == 1
    assert counts["pixel_fn"] == 1
    assert counts["pixel_tn"] == 9
    assert counts["num_samples"] == 2
    assert counts["num_pixels"] == 12
    assert counts["num_target_pixels"] == 2
    assert counts["num_gt_objects"] == 1
    assert counts["num_tp_objects"] == 1
    assert counts["num_fp_components"] == 1
    assert counts["num_empty_empty_images"] == 1
    assert metrics["pixel_iou"] == pytest.approx(1.0 / 3.0)
    assert metrics["nIoU"] == pytest.approx(1.0 / 6.0)
    assert metrics["pixel_precision"] == pytest.approx(0.5)
    assert metrics["pixel_recall"] == pytest.approx(0.5)
    assert metrics["pixel_f1"] == pytest.approx(0.5)
    assert metrics["object_pd"] == pytest.approx(1.0)
    assert metrics["pixel_fa"] == pytest.approx(1.0 / 12.0)
    assert metrics["component_fa_per_megapixel"] == pytest.approx(1e6 / 12.0)
    score_statistics = result["score_statistics"]
    assert score_statistics["global_min"] == pytest.approx(0.1)
    assert score_statistics["global_max"] == pytest.approx(0.9)
    assert score_statistics["image_mean_std"] == pytest.approx(0.1)
    assert score_statistics["min_within_image_std"] == 0.0
    assert score_statistics["median_within_image_std"] > 0.0
    assert score_statistics["exact_constant_images"] == 1
    assert score_statistics["near_constant_images"] == 1


def test_all_empty_zero_denominators_follow_basicirstd_convention() -> None:
    result = compute_standard_metrics(
        [np.zeros((2, 2), dtype=np.float32)],
        [np.zeros((2, 2), dtype=np.uint8)],
        0.5,
    )
    metrics = result["metrics"]
    assert metrics["pixel_iou"] == 0.0
    assert metrics["nIoU"] == 0.0
    assert metrics["pixel_precision"] == 0.0
    assert metrics["pixel_recall"] == 0.0
    assert metrics["pixel_f1"] == 0.0
    assert metrics["object_pd"] == 0.0


def test_formal_cli_writes_protocol_and_hash_provenance(tmp_path: Path) -> None:
    probabilities, masks = _metric_example()
    score_dir = _formal_score_directory(tmp_path / "scores", probabilities, masks)
    output = tmp_path / "metrics.json"
    assert main(
        [
            "--score-dir",
            str(score_dir),
            "--threshold",
            "0.5",
            "--output",
            str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["formal_protocol_eligible"] is True
    assert payload["protocol"]["threshold_rule"] == (
        "prediction = (probability >= threshold)"
    )
    assert payload["protocol"]["nIoU_definition"] == (
        "mean_i(intersection_i / union_i)"
    )
    assert payload["protocol"]["nIoU_empty_empty_image_value"] == 0.0
    collapse_protocol = payload["protocol"]["score_collapse_statistics"]
    assert collapse_protocol["near_constant_std_threshold"] == pytest.approx(1e-6)
    assert "population standard deviation" in collapse_protocol[
        "within_image_std_definition"
    ]
    assert payload["score_statistics"]["exact_constant_images"] == 1
    assert payload["score_statistics"]["near_constant_images"] == 1
    assert payload["hiou_status"] == "not_defined_not_reported"
    assert payload["provenance"]["manifest_sha256"] == file_sha256(
        score_dir / "manifest.json"
    )
    assert payload["provenance"]["score_integrity_audit"]["verified"] is True
    assert payload["provenance"]["num_manifest_records"] == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_formal_evaluation_accepts_and_preserves_rc_mshnet_backend(
    tmp_path: Path,
) -> None:
    probabilities, masks = _metric_example()
    score_dir = _formal_score_directory(
        tmp_path / "rc-scores",
        probabilities,
        masks,
        model_backend="rc_mshnet",
    )

    result = evaluate_standard_score_directory(score_dir, 0.5)

    assert result["protocol"]["model_backend"] == "rc_mshnet"
    assert result["provenance"]["model_backend"] == "rc_mshnet"


def _mutate_manifest(score_dir: Path, **updates: object) -> None:
    path = score_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(updates)
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def test_formal_evaluation_fails_closed_for_legacy_manifest(tmp_path: Path) -> None:
    probabilities, masks = _metric_example()
    score_dir = _formal_score_directory(tmp_path / "scores", probabilities, masks)
    _mutate_manifest(score_dir, schema_version=2)
    with pytest.raises(ValueError, match="complete version-3"):
        evaluate_standard_score_directory(score_dir, 0.5)


def test_formal_evaluation_fails_closed_for_resized_scores(tmp_path: Path) -> None:
    probabilities, masks = _metric_example()
    score_dir = _formal_score_directory(tmp_path / "scores", probabilities, masks)
    _mutate_manifest(score_dir, spatial_mode="resize", base_hw=[2, 3])
    with pytest.raises(ValueError, match="spatial_mode='native'"):
        evaluate_standard_score_directory(score_dir, 0.5)


def test_formal_evaluation_fails_closed_for_label_free_scores(tmp_path: Path) -> None:
    probabilities, masks = _metric_example()
    score_dir = _formal_score_directory(tmp_path / "scores", probabilities, masks)
    _mutate_manifest(score_dir, labels_loaded=False)
    with pytest.raises(ValueError, match="label mode|labels_loaded"):
        evaluate_standard_score_directory(score_dir, 0.5)


def test_formal_evaluation_requires_split_hash_provenance(tmp_path: Path) -> None:
    probabilities, masks = _metric_example()
    score_dir = _formal_score_directory(tmp_path / "scores", probabilities, masks)
    manifest_path = score_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("split_file_sha256")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="split provenance|split-file provenance"):
        evaluate_standard_score_directory(score_dir, 0.5)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"checkpoint_diagnostic_only": True}, "checkpoint_diagnostic_only=False"),
        ({"non_strict_state_loading": True}, "non-strict checkpoint"),
        ({"split_role": "train"}, "split_role='test'"),
        ({"split_authority_verified": False}, "split_authority_verified=True"),
        ({"checkpoint_selection_rule": "best_iou"}, "fixed_last"),
        ({"model_backend": "compact"}, "model_backend"),
        ({"source_datasets": []}, "non-empty source_datasets"),
        (
            {"source_datasets": ["source-a", "held-out-target"]},
            "excluded from detector source_datasets",
        ),
    ],
)
def test_formal_evaluation_rejects_nonformal_manifest_fields(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    probabilities, masks = _metric_example()
    score_dir = _formal_score_directory(tmp_path / "scores", probabilities, masks)
    _mutate_manifest(score_dir, **updates)
    with pytest.raises(ValueError, match=message):
        evaluate_standard_score_directory(score_dir, 0.5)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_standard_metrics_reject_invalid_threshold(threshold: float) -> None:
    probabilities, masks = _metric_example()
    with pytest.raises(ValueError, match="threshold"):
        compute_standard_metrics(probabilities, masks, threshold)


@pytest.mark.parametrize("threshold", [-1e-6, float("inf"), float("nan")])
def test_score_collapse_threshold_must_be_finite_and_nonnegative(
    threshold: float,
) -> None:
    probabilities, masks = _metric_example()
    with pytest.raises(ValueError, match="near_constant_std_threshold"):
        compute_standard_metrics(
            probabilities,
            masks,
            0.5,
            near_constant_std_threshold=threshold,
        )


def test_near_constant_image_count_uses_recorded_inclusive_threshold() -> None:
    probability = np.asarray([[0.5, 0.500002]], dtype=np.float64)
    mask = np.zeros_like(probability, dtype=np.uint8)
    below = compute_standard_metrics(
        [probability],
        [mask],
        0.5,
        near_constant_std_threshold=0.9e-6,
    )
    at_boundary = compute_standard_metrics(
        [probability],
        [mask],
        0.5,
        near_constant_std_threshold=1.0e-6,
    )
    assert below["score_statistics"]["near_constant_images"] == 0
    assert at_boundary["score_statistics"]["near_constant_images"] == 1
    assert at_boundary["score_statistics"]["exact_constant_images"] == 0
