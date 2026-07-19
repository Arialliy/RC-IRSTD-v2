import numpy as np

from evaluation.raw_logit_oracle import RawLogitSample
from risk_curve.evaluate_dense_grid_gap import (
    compare_exact_and_dense,
    select_dense_grid_oracle,
)


def _sample() -> RawLogitSample:
    logits = np.full((4, 4), -8.0, dtype=np.float32)
    logits[1, 1] = 10.0
    logits[3, 3] = 9.0
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    probability = (1.0 / (1.0 + np.exp(-logits.astype(np.float64)))).astype(
        np.float32
    )
    return RawLogitSample("sample", logits, probability, mask)


def test_dense_grid_gap_is_zero_when_exact_breakpoint_is_present():
    result = compare_exact_and_dense(
        [_sample()],
        np.asarray([0.0, 10.0, 11.0], dtype=np.float32),
        pixel_budget=1e-3,
        component_budget=1.0,
    )
    assert result["exact_pd"] == 1.0
    assert result["dense_pd"] == 1.0
    assert result["grid_gap"] == 0.0
    assert result["dense"]["operating_point"]["threshold_index"] == 1


def test_lossless_pixel_cutoff_prunes_only_infeasible_dense_states():
    grid = np.asarray([0.0, 9.0, 10.0, 11.0], dtype=np.float32)
    full = select_dense_grid_oracle(
        [_sample()],
        grid,
        pixel_budget=1e-3,
        component_budget=1.0,
    )
    pruned = select_dense_grid_oracle(
        [_sample()],
        grid,
        pixel_budget=1e-3,
        component_budget=1.0,
        provably_infeasible_at_or_below_logit=9.0,
    )
    assert pruned["operating_point"] == full["operating_point"]
    assert pruned["search"]["finite_grid_points_evaluated"] == 2
    assert (
        pruned["search"]["finite_grid_points_pixel_budget_proven_infeasible"]
        == 2
    )


def test_dense_grid_worker_threads_are_result_equivalent():
    grid = np.asarray([0.0, 8.0, 9.0, 10.0, 11.0], dtype=np.float32)
    sequential = select_dense_grid_oracle(
        [_sample(), _sample()],
        grid,
        pixel_budget=1e-3,
        component_budget=1.0,
        workers=1,
    )
    threaded = select_dense_grid_oracle(
        [_sample(), _sample()],
        grid,
        pixel_budget=1e-3,
        component_budget=1.0,
        workers=2,
    )
    assert sequential["operating_point"] == threaded["operating_point"]
    assert sequential["search"] | {"workers": 2} == threaded["search"]


def test_dense_grid_rejects_invalid_worker_count():
    with np.testing.assert_raises_regex(ValueError, "positive integer"):
        select_dense_grid_oracle(
            [_sample()],
            np.asarray([0.0, 11.0], dtype=np.float32),
            pixel_budget=1e-3,
            component_budget=1.0,
            workers=0,
        )


def test_dense_grid_gap_detects_missing_extreme_tail_state():
    result = compare_exact_and_dense(
        [_sample()],
        np.asarray([0.0, 11.0], dtype=np.float32),
        pixel_budget=1e-3,
        component_budget=1.0,
    )
    assert result["exact_pd"] == 1.0
    assert result["dense_pd"] == 0.0
    assert result["grid_gap"] == 1.0


def test_external_empty_action_is_not_part_of_finite_model_grid():
    result = select_dense_grid_oracle(
        [_sample()],
        np.asarray([11.0, 12.0], dtype=np.float32),
        pixel_budget=1e-6,
        component_budget=1e-6,
    )
    point = result["operating_point"]
    # A finite zero-Pd state is preferred on the exact tie-break, but reject is
    # still independently evaluated and never appended to the model grid.
    assert point["threshold_index"] == 0
    assert point["threshold_logit"] == 11.0
    assert result["search"]["external_empty_action_evaluated"] is True


def test_reject_is_selected_when_no_finite_grid_point_is_feasible():
    logits = np.full((3, 3), 5.0, dtype=np.float32)
    sample = RawLogitSample(
        "all-background",
        logits,
        np.full((3, 3), 1.0 / (1.0 + np.exp(-5.0)), dtype=np.float32),
        np.zeros((3, 3), dtype=bool),
    )
    result = select_dense_grid_oracle(
        [sample],
        np.asarray([0.0, 4.0], dtype=np.float32),
        pixel_budget=1e-6,
        component_budget=1e-6,
    )
    point = result["operating_point"]
    assert point["empty_action"] is True
    assert point["threshold_index"] is None
    assert point["threshold_logit"] == "+inf"
