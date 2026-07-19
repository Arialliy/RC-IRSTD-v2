from __future__ import annotations

import hashlib

import numpy as np
import pytest

from risk_curve.build_logit_threshold_grid import (
    DenseTailGridSpec,
    build_dense_tail_grid,
)
from risk_curve.representation import (
    DEFAULT_LOGIT_GRID_POINTS,
    EMPTY_ACTION_THRESHOLD,
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_REPRESENTATION,
    MAX_MODEL_GRID_POINTS,
    empty_action_contract,
    logit_threshold_grid_sha256,
    validate_logit_threshold_grid,
)
from risk_curve.threshold_grid import (
    build_threshold_grid,
    validate_threshold_grid,
)


def test_logit_grid_is_finite_float32_strict_and_bounded() -> None:
    values = np.asarray([-8.0, 0.0, 9.5], dtype=np.float32)
    validated = validate_logit_threshold_grid(values)
    assert validated.dtype == np.float32
    assert validated.flags.c_contiguous
    assert np.array_equal(validated, values)
    assert np.isfinite(validated).all()
    assert EMPTY_ACTION_THRESHOLD == float("inf")
    assert empty_action_contract() == {
        "external": True,
        "threshold": "+inf",
        "threshold_index": None,
        "prediction": "empty_mask",
        "included_in_model_grid": False,
    }


@pytest.mark.parametrize(
    "values, message",
    [
        (np.asarray([0.0, 1.0]), "float32"),
        (np.asarray([[0.0, 1.0]], dtype=np.float32), "one-dimensional"),
        (np.asarray([0.0], dtype=np.float32), "at least two"),
        (np.asarray([0.0, np.nan], dtype=np.float32), "must all be finite"),
        (np.asarray([-np.inf, 0.0], dtype=np.float32), "must all be finite"),
        (np.asarray([0.0, np.inf], dtype=np.float32), "external action"),
        (np.asarray([0.0, 0.0], dtype=np.float32), "strictly increasing"),
        (np.asarray([1.0, 0.0], dtype=np.float32), "strictly increasing"),
    ],
)
def test_logit_grid_rejects_implicit_or_invalid_states(
    values: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_logit_threshold_grid(values)


def test_logit_grid_rejects_more_than_model_contract() -> None:
    values = np.linspace(-20.0, 20.0, MAX_MODEL_GRID_POINTS + 1).astype(
        np.float32
    )
    with pytest.raises(ValueError, match="maximum"):
        validate_logit_threshold_grid(values)


def test_default_grid_is_1024_but_contract_allows_source_only_2048_fallback() -> None:
    assert DenseTailGridSpec().max_grid_points == DEFAULT_LOGIT_GRID_POINTS == 1024
    fallback = np.linspace(-20.0, 20.0, MAX_MODEL_GRID_POINTS).astype(np.float32)
    assert validate_logit_threshold_grid(fallback).size == 2048


def test_semantic_hash_binds_schema_representation_and_little_endian_values() -> None:
    values = np.asarray([-3.0, 0.25, 12.0], dtype=np.float32)
    digest = logit_threshold_grid_sha256(values)
    raw_only = hashlib.sha256(
        np.ascontiguousarray(values, dtype="<f4").tobytes()
    ).hexdigest()
    assert digest != raw_only
    assert digest == logit_threshold_grid_sha256(values.copy())
    assert digest != logit_threshold_grid_sha256(
        values,
        schema_version=LOGIT_GRID_SCHEMA_VERSION + "-different",
    )
    assert digest != logit_threshold_grid_sha256(
        values,
        representation=LOGIT_REPRESENTATION + "-different",
    )


def test_dense_tail_builder_is_deterministic_finite_and_keeps_candidate_extreme() -> None:
    background = np.linspace(-12.0, 14.0, 20_000, dtype=np.float32)
    candidates = np.asarray([4.0, 8.0, 15.0, 16.0], dtype=np.float32)
    spec = DenseTailGridSpec(
        max_grid_points=32,
        bulk_points=8,
        upper_points=8,
        extreme_points=8,
        candidate_points=4,
    )
    first, audit = build_dense_tail_grid(background, candidates, spec=spec)
    second, _ = build_dense_tail_grid(background.copy(), candidates[::-1], spec=spec)
    assert np.array_equal(first, second)
    assert first.dtype == np.float32
    assert first.size <= 32
    assert np.isfinite(first).all()
    assert np.all(np.diff(first.astype(np.float64)) > 0.0)
    assert first[-1] == np.float32(16.0)
    assert audit["candidate_selected_points"] == 4
    assert audit["refinement_enabled"] is False
    assert audit["refinement_points_added"] == 0
    assert audit["max_adjacent_logit_gap_before"] == audit[
        "max_adjacent_logit_gap_after"
    ]


def test_2048_fallback_deterministically_refines_largest_float32_gaps() -> None:
    background = np.linspace(-8.0, 8.0, 4096, dtype=np.float32)
    candidates = np.asarray([16.0, 32.0], dtype=np.float32)
    spec = DenseTailGridSpec(
        max_grid_points=MAX_MODEL_GRID_POINTS,
        bulk_points=2,
        upper_points=2,
        extreme_points=2,
        candidate_points=2,
    )

    first, audit = build_dense_tail_grid(background, candidates, spec=spec)
    second, second_audit = build_dense_tail_grid(
        background[::-1].copy(), candidates[::-1].copy(), spec=spec
    )

    assert np.array_equal(first, second)
    assert first.size == MAX_MODEL_GRID_POINTS
    assert audit == second_audit
    assert audit["initial_grid_points"] < MAX_MODEL_GRID_POINTS
    assert audit["refinement_enabled"] is True
    assert audit["refinement_strategy"] == (
        "largest_adjacent_raw_logit_gap_float32_midpoint"
    )
    assert audit["refinement_points_added"] == (
        MAX_MODEL_GRID_POINTS - audit["initial_grid_points"]
    )
    assert audit["refinement_stop_reason"] == "max_grid_points_reached"
    assert audit["max_adjacent_logit_gap_after"] < audit[
        "max_adjacent_logit_gap_before"
    ]


def test_default_1024_cap_does_not_fill_unused_quantile_capacity() -> None:
    background = np.linspace(-4.0, 4.0, 128, dtype=np.float32)
    grid, audit = build_dense_tail_grid(
        background,
        np.asarray([12.0], dtype=np.float32),
        spec=DenseTailGridSpec(
            max_grid_points=DEFAULT_LOGIT_GRID_POINTS,
            bulk_points=2,
            upper_points=2,
            extreme_points=2,
            candidate_points=1,
        ),
    )

    assert grid.size == audit["initial_grid_points"] < DEFAULT_LOGIT_GRID_POINTS
    assert audit["refinement_enabled"] is False
    assert audit["refinement_points_added"] == 0
    assert audit["refinement_stop_reason"] == (
        "disabled_for_default_1024_contract"
    )


def test_constant_source_gets_two_distinct_finite_float32_model_points() -> None:
    background = np.full(64, 2.0, dtype=np.float32)
    grid, audit = build_dense_tail_grid(
        background,
        np.empty(0, dtype=np.float32),
        spec=DenseTailGridSpec(
            max_grid_points=4,
            bulk_points=2,
            upper_points=0,
            extreme_points=0,
            candidate_points=0,
        ),
    )
    assert grid.size == 2
    assert np.isfinite(grid).all()
    assert np.diff(grid.astype(np.float64))[0] > 0.0
    assert audit["synthetic_float32_neighbor_added"] is True


def test_legacy_probability_grid_contract_is_unchanged() -> None:
    legacy = build_threshold_grid()
    assert np.array_equal(validate_threshold_grid(legacy), legacy)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_threshold_grid(np.asarray([-2.0, 2.0], dtype=np.float32))
