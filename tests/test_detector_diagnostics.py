from __future__ import annotations

import numpy as np
import pytest

from evaluation.detector_diagnostics import (
    BudgetPair,
    _concentration,
    _per_image_at_threshold,
    decompose_constraints,
    parse_budget,
    ranking_diagnostics,
)


def _curve_rows() -> list[dict[str, float | int]]:
    return [
        {
            "threshold": 0.5,
            "pd": 1.0,
            "fa_pixel": 2e-5,
            "fa_component_mp": 4.0,
            "tp_objects": 2,
            "gt_objects": 2,
            "fp_components": 4,
            "fp_pixels": 20,
            "total_pixels": 1_000_000,
        },
        {
            "threshold": 0.9,
            "pd": 0.5,
            "fa_pixel": 1e-6,
            "fa_component_mp": 2.0,
            "tp_objects": 1,
            "gt_objects": 2,
            "fp_components": 2,
            "fp_pixels": 1,
            "total_pixels": 1_000_000,
        },
        {
            "threshold": 1.0,
            "pd": 0.0,
            "fa_pixel": 0.0,
            "fa_component_mp": 0.0,
            "tp_objects": 0,
            "gt_objects": 2,
            "fp_components": 0,
            "fp_pixels": 0,
            "total_pixels": 1_000_000,
        },
    ]


def test_constraint_decomposition_identifies_binding_risk() -> None:
    result = decompose_constraints(
        _curve_rows(), [BudgetPair("strict", 1e-6, 1.0)]
    )["strict"]["oracle"]
    assert result["pixel_only"]["operating_point"]["threshold"] == 0.9
    assert result["pixel_only"]["operating_point"]["pd"] == 0.5
    assert result["component_only"]["operating_point"]["threshold"] == 1.0
    assert result["joint"]["operating_point"]["threshold"] == 1.0
    assert result["joint"]["nonempty"] is False


def test_budget_parser_rejects_invalid_contract() -> None:
    assert parse_budget("loose:1e-5:5") == BudgetPair("loose", 1e-5, 5.0)
    with pytest.raises(Exception, match="NAME:PIXEL:COMPONENT"):
        parse_budget("bad")
    with pytest.raises(Exception, match="positive"):
        parse_budget("bad:0:1")


def test_per_image_false_alarm_counts_and_concentration() -> None:
    probability = np.asarray(
        [[0.9, 0.0, 0.8], [0.0, 0.95, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    rows = _per_image_at_threshold(
        ["a"],
        [probability],
        [mask],
        0.5,
        matching_rule="overlap",
        centroid_distance=3.0,
        connectivity=1,
        min_component_area=1,
    )
    assert rows[0]["matched_targets"] == 1
    assert rows[0]["false_pixels"] == 2
    assert rows[0]["false_components"] == 2
    assert rows[0]["false_peaks"] == 2
    concentration = _concentration(rows, "false_pixels")
    assert concentration["top_1"]["fraction"] == 1.0


def test_probability_ranking_reports_saturation_and_pairwise_order() -> None:
    probability = np.asarray(
        [[1.0, 0.2, 0.9], [0.1, 1.0, 0.2], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    result = ranking_diagnostics(
        [probability], [mask], [0.5, 1.0], connectivity=1
    )
    assert result["exact_one"]["target_pixels"] == 1
    assert result["exact_one"]["background_pixels"] == 1
    assert (
        result["exact_one"][
            "gt_components_saturated_with_saturated_image_background"
        ]
        == 1
    )
    assert result["curves"]["gt_max_score_recall"] == [1.0, 1.0]
    assert 0.0 <= result["target_vs_false_peak_pairwise_auc_with_half_ties"] <= 1.0


def test_false_peak_count_is_not_an_alias_for_component_count() -> None:
    probability = np.asarray(
        [[0.9, 0.6, 0.8, 0.0], [0.0, 0.0, 0.0, 0.95]],
        dtype=np.float32,
    )
    mask = np.zeros_like(probability, dtype=bool)
    mask[1, 3] = True
    rows = _per_image_at_threshold(
        ["two-peaks"],
        [probability],
        [mask],
        0.5,
        matching_rule="overlap",
        centroid_distance=3.0,
        connectivity=1,
        min_component_area=1,
    )
    assert rows[0]["false_components"] == 1
    assert rows[0]["false_peaks"] == 2
