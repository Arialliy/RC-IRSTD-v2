from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from risk_curve.build_deployment_statistics import (
    STATIC_CROSS_FIT_STATISTICS_SCHEMA_VERSION,
    build_static_cross_fit_statistics,
)
from risk_curve.evaluate_zero_label import evaluate_zero_label_actions


def _score_directory(root: Path, count: int = 6) -> Path:
    root.mkdir()
    files = []
    for index in range(count):
        filename = f"{index:03d}.npz"
        probability = np.full((4, 4), index / max(count - 1, 1), dtype=np.float32)
        np.savez_compressed(
            root / filename,
            prob=probability,
            gray=probability,
            image_id=np.asarray(f"image-{index}"),
        )
        files.append(filename)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "files": files,
                "score_type": "sigmoid_probability",
                "warm_flag": True,
                "spatial_mode": "native",
                "pad_multiple": 16,
                "target_dataset": "static-target",
                "source_datasets": ["source-a", "source-b"],
                "weight_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_static_crossfit_is_mask_free_disjoint_per_fold_and_full_coverage(
    tmp_path: Path,
) -> None:
    score_dir = _score_directory(tmp_path / "scores")
    arrays = build_static_cross_fit_statistics(
        score_dir,
        np.asarray([0.0, 0.5, 0.9], dtype=np.float32),
        folds=3,
        seed=7,
    )
    assert str(np.asarray(arrays["deployment_statistics_schema_version"]).item()) == (
        STATIC_CROSS_FIT_STATISTICS_SCHEMA_VERSION
    )
    assert np.asarray(arrays["statistics"]).shape[0] == 3
    provenance = json.loads(str(np.asarray(arrays["provenance_json"]).item()))
    assert provenance["mode"] == "static_cross_fit"
    assert provenance["masks_read"] is False
    assert provenance["full_test_coverage"] is True
    assert provenance["formal_crc_eligible"] is False
    adaptation = [json.loads(str(value)) for value in arrays["adaptation_ids"]]
    evaluation = [json.loads(str(value)) for value in arrays["evaluation_ids"]]
    assert all(len(row) == 4 for row in adaptation)
    assert all(len(row) == 2 for row in evaluation)
    assert all(not set(a).intersection(e) for a, e in zip(adaptation, evaluation))
    evaluated = [image_id for row in evaluation for image_id in row]
    assert len(evaluated) == len(set(evaluated)) == 6


def test_static_crossfit_samples_fixed_adaptation_window_deterministically(
    tmp_path: Path,
) -> None:
    score_dir = _score_directory(tmp_path / "scores", count=10)
    kwargs = {
        "folds": 5,
        "seed": 19,
        "adaptation_window": 3,
    }
    first = build_static_cross_fit_statistics(
        score_dir,
        np.asarray([0.0, 0.5, 0.9], dtype=np.float32),
        **kwargs,
    )
    second = build_static_cross_fit_statistics(
        score_dir,
        np.asarray([0.0, 0.5, 0.9], dtype=np.float32),
        **kwargs,
    )

    adaptation = [json.loads(str(value)) for value in first["adaptation_ids"]]
    repeated_adaptation = [
        json.loads(str(value)) for value in second["adaptation_ids"]
    ]
    evaluation = [json.loads(str(value)) for value in first["evaluation_ids"]]
    assert adaptation == repeated_adaptation
    assert all(len(row) == 3 for row in adaptation)
    assert all(not set(a).intersection(e) for a, e in zip(adaptation, evaluation))
    evaluated = [image_id for row in evaluation for image_id in row]
    assert len(evaluated) == len(set(evaluated)) == 10

    provenance = json.loads(str(np.asarray(first["provenance_json"]).item()))
    assert provenance["adaptation_window"] == 3
    assert provenance["selected_adaptation_ids_by_fold"] == adaptation
    assert provenance["selected_adaptation_sizes"] == [3] * 5
    assert provenance["complement_sizes"] == [8] * 5
    records = [json.loads(str(value)) for value in first["block_records_json"]]
    for record, selected_ids in zip(records, adaptation):
        assert record["complement_size"] == 8
        assert record["selected_adaptation_size"] == 3
        assert record["selected_adaptation_ids"] == selected_ids
        assert record["adaptation_window"] == 3
        assert "without_replacement" in record["adaptation_sampling_rule"]


def test_static_crossfit_rejects_complement_smaller_than_adaptation_window(
    tmp_path: Path,
) -> None:
    score_dir = _score_directory(tmp_path / "scores", count=6)
    with pytest.raises(ValueError, match="complement_size=4, adaptation_window=5"):
        build_static_cross_fit_statistics(
            score_dir,
            np.asarray([0.0, 0.5], dtype=np.float32),
            folds=3,
            seed=7,
            adaptation_window=5,
        )


def _count_curves() -> dict[str, np.ndarray]:
    return {
        "image_ids": np.asarray(["a", "b"]),
        "thresholds": np.asarray([0.0, 0.5], dtype=np.float32),
        "false_positive_pixels": np.asarray([[10, 0], [10, 0]], dtype=np.float64),
        "false_positive_components": np.asarray([[2, 0], [2, 0]], dtype=np.float64),
        "total_pixels": np.asarray([100, 100], dtype=np.float64),
        "tp_object_counts": np.asarray([[1, 1], [1, 0]], dtype=np.float64),
        "gt_object_counts": np.asarray([1, 1], dtype=np.float64),
    }


def test_zero_label_audit_binds_actions_by_identity_and_keeps_rejects() -> None:
    zero = {
        "mode": "zero_label_empirical_adaptation",
        "adaptation_protocol": "static_cross_fit",
        "pixel_budget": 0.01,
        "component_budget": 100.0,
        "thresholds": [0.0, 0.5],
        "threshold_indices_by_image": {"a": 1, "b": None},
    }
    result = evaluate_zero_label_actions(zero, _count_curves())
    metrics = result["metrics"]
    assert result["guarantee"].startswith("none")
    assert metrics["coverage_rate"] == pytest.approx(0.5)
    assert metrics["reject_rate"] == pytest.approx(0.5)
    assert metrics["joint_budget_satisfaction_rate_upper"] == pytest.approx(1.0)
    # Rejected image b remains in the GT denominator as a no-detection action.
    assert metrics["pd_object_aggregate"] == pytest.approx(0.5)
    assert metrics["pd_object_aggregate_active_only"] == pytest.approx(1.0)


def test_zero_label_audit_refuses_silent_partial_coverage() -> None:
    zero = {
        "mode": "zero_label_empirical_adaptation",
        "pixel_budget": 0.01,
        "component_budget": 100.0,
        "thresholds": [0.0, 0.5],
        "threshold_indices_by_image": {"a": 1},
    }
    with pytest.raises(ValueError, match="does not cover every evaluated image"):
        evaluate_zero_label_actions(zero, _count_curves())
