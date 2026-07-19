from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from certification.build_calibration_losses import build_calibration_losses
from certification.conformal_offset import select_conformal_offset
from certification.evaluate_certified_mode import evaluate_selected_operating_point


def _test_losses(image_ids: list[str] | None = None):
    ids = image_ids or ["test-0", "test-1", "test-2", "test-3"]
    return build_calibration_losses(
        image_ids=ids,
        thresholds=np.asarray([0.1, 0.9]),
        false_positive_pixels=np.asarray(
            [
                [0, 0],
                [0, 0],
                [10, 0],
                [0, 0],
            ],
            dtype=np.float64,
        ),
        false_positive_components=np.zeros((4, 2), dtype=np.float64),
        total_pixels=np.full(4, 100, dtype=np.float64),
        pixel_budget=0.05,
        component_budget=1.0,
        tp_object_counts=np.asarray(
            [
                [2, 0],
                [10, 0],
                [2, 0],
                [10, 0],
            ],
            dtype=np.float64,
        ),
        gt_object_counts=np.asarray([2, 10, 2, 10], dtype=np.float64),
    )


def _selection(test_ids: list[str] | None = None) -> dict[str, object]:
    ids = test_ids or ["test-0", "test-1", "test-2", "test-3"]
    return {
        "success": True,
        "reject": False,
        "reason": "corrected_bound_satisfies_alpha",
        "calibration_image_ids": ["cal-0", "cal-1"],
        "adaptation_image_ids": [],
        "test_image_ids_checked": ids,
        "thresholds": [0.1, 0.9],
        "budgets": {
            "pixel": {"value": 0.05},
            "component": {"value": 1.0},
        },
        "loss": {"mode": "budget_violation"},
        "selected_threshold_index": None,
        "selected_threshold": None,
        "selected_test_threshold_indices": [0, None, 0, None],
        "offset_rank": 0,
        "guarantee_scope": (
            "finite-sample joint control, equivalently a raw-component JointBSR bound"
        ),
    }


def test_partial_calibration_reject_is_traced_but_cannot_certify() -> None:
    curves = np.ones((20, 6), dtype=np.float64)
    curves[:, 4:] = 0.0
    bases = np.asarray([0] * 10 + [3] * 10)

    result = select_conformal_offset(curves, zero_indices=bases, alpha=0.1)

    assert result.success is False
    assert result.reject is True
    assert result.reason == "only_terminal_reject_is_feasible"
    assert result.selected_threshold_index is None
    assert result.selected_threshold_indices == (None,) * 20
    partial = next(item for item in result.candidate_trace if item["offset_rank"] == 4)
    assert partial["partial_reject"] is True
    assert partial["reject_count"] == 10
    assert partial["corrected_bound_satisfies_alpha"] is True
    assert partial["eligible_for_formal_success"] is False
    assert partial["feasible"] is False


def test_full_calibration_coverage_can_still_select_adaptive_rank() -> None:
    curves = np.ones((20, 6), dtype=np.float64)
    curves[:, 4:] = 0.0
    bases = np.asarray([0] * 10 + [1] * 10)

    result = select_conformal_offset(curves, zero_indices=bases, alpha=0.1)

    assert result.success is True
    assert result.reject is False
    assert result.offset_rank == 4
    assert all(value is not None for value in result.selected_threshold_indices)
    assert result.candidate_trace[-1]["calibration_coverage_rate"] == 1.0
    assert result.candidate_trace[-1]["eligible_for_formal_success"] is True


def test_test_rejects_are_no_detection_actions_with_transparent_metrics() -> None:
    losses = _test_losses()
    audit = evaluate_selected_operating_point(_selection(), losses)
    metrics = audit["metrics"]

    assert audit["success"] is True
    assert audit["has_test_rejections"] is True
    assert audit["all_test_actions_rejected"] is False
    assert audit["test_action_audit"] == {
        "active_count": 2,
        "no_detection_reject_count": 2,
        "coverage_rate": 0.5,
        "reject_rate": 0.5,
        "ordered_id_alignment_verified": True,
    }
    assert [item["action"] for item in audit["selected_test_actions"]] == [
        "threshold_grid_action",
        "no_detection_reject",
        "threshold_grid_action",
        "no_detection_reject",
    ]
    assert audit["selected_test_actions"][1]["threshold"] is None
    assert metrics["joint_budget_satisfaction_rate_per_image_suffix_max"] == 0.75
    assert (
        metrics[
            "joint_budget_satisfaction_rate_per_image_suffix_max_active_only"
        ]
        == 0.5
    )
    assert metrics["ground_truth_objects"] == 24
    assert metrics["ground_truth_objects_active_only"] == 4
    assert metrics["ground_truth_objects_in_rejected_images"] == 20
    assert metrics["pd_object_aggregate"] == pytest.approx(1.0 / 6.0)
    assert metrics["pd_object_aggregate_active_only"] == 1.0
    assert "conservative" in audit["guarantee_scope"]
    assert "equivalently" not in audit["guarantee_scope"]
    assert "conservative" in metrics["component_control_interpretation"]


def test_all_test_rejects_keep_full_gt_denominator_and_zero_coverage() -> None:
    losses = _test_losses()
    selection = _selection()
    selection["selected_test_threshold_indices"] = [None, None, None, None]

    audit = evaluate_selected_operating_point(selection, losses)
    metrics = audit["metrics"]

    assert audit["all_test_actions_rejected"] is True
    assert audit["test_action_audit"]["coverage_rate"] == 0.0
    assert audit["test_action_audit"]["no_detection_reject_count"] == 4
    assert metrics["ground_truth_objects"] == 24
    assert metrics["pd_object_aggregate"] == 0.0
    assert metrics["pd_object_aggregate_active_only"] is None
    assert (
        metrics[
            "joint_budget_satisfaction_rate_per_image_suffix_max_including_rejects_as_no_detection"
        ]
        == 1.0
    )
    assert (
        metrics[
            "joint_budget_satisfaction_rate_per_image_suffix_max_active_only"
        ]
        is None
    )


def test_per_image_actions_reject_reordered_or_unidentified_test_ids() -> None:
    losses = _test_losses()
    reordered = replace(
        losses,
        image_ids=np.asarray(
            ["test-1", "test-0", "test-2", "test-3"], dtype=str
        ),
    )
    with pytest.raises(ValueError, match="ID order differs"):
        evaluate_selected_operating_point(_selection(), reordered)

    selection = _selection()
    selection.pop("test_image_ids_checked")
    with pytest.raises(ValueError, match="ordered test_image_ids_checked"):
        evaluate_selected_operating_point(selection, losses)
