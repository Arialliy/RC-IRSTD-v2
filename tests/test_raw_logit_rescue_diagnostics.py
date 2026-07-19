from __future__ import annotations

import copy

import pytest

from evaluation.raw_logit_rescue_diagnostics import (
    build_realized_fa_sensitivity,
    deterministic_dense_state_indices,
    select_dense_operating_points,
    summarize_false_alarm_concentration,
)


def _row(
    tp: int,
    *,
    fp_pixels: int,
    fp_components: int,
    gt: int = 10,
    total_pixels: int = 1_000_000,
) -> dict[str, int]:
    return {
        "tp_objects": tp,
        "gt_objects": gt,
        "fp_pixels": fp_pixels,
        "fp_components": fp_components,
        "total_pixels": total_pixels,
    }


def _state(
    threshold: float | str,
    a: dict[str, int],
    b: dict[str, int],
    *,
    reject: bool = False,
) -> dict[str, object]:
    pooled = {
        key: int(a[key]) + int(b[key])
        for key in (
            "tp_objects",
            "gt_objects",
            "fp_pixels",
            "fp_components",
            "total_pixels",
        )
    }
    return {
        "threshold_logit_float32": threshold,
        "all_reject_sentinel": reject,
        "pooled": pooled,
        "per_domain": {"a": a, "b": b},
    }


def _selection_states() -> list[dict[str, object]]:
    return [
        _state(
            "+inf",
            _row(0, fp_pixels=0, fp_components=0),
            _row(0, fp_pixels=0, fp_components=0),
            reject=True,
        ),
        # Both domains meet a 1 pixel / 1 component per-MP budget.
        _state(
            9.0,
            _row(5, fp_pixels=1, fp_components=1),
            _row(5, fp_pixels=1, fp_components=1),
        ),
        # Pooled rates meet the same budget, but domain a alone violates it.
        _state(
            8.0,
            _row(8, fp_pixels=2, fp_components=2),
            _row(4, fp_pixels=0, fp_components=0),
        ),
    ]


def test_dense_state_indices_include_both_endpoints_without_duplicates() -> None:
    assert deterministic_dense_state_indices(1) == [0]
    assert deterministic_dense_state_indices(4, grid_size=10) == [0, 1, 2, 3]
    assert deterministic_dense_state_indices(10, grid_size=4) == [0, 3, 6, 9]
    indices = deterministic_dense_state_indices(10_000, grid_size=1024)
    assert len(indices) == len(set(indices)) == 1024
    assert indices == sorted(indices)
    assert (indices[0], indices[-1]) == (0, 9_999)


def test_dense_state_indices_reject_impossible_one_point_grid() -> None:
    with pytest.raises(ValueError, match="at least two"):
        deterministic_dense_state_indices(2, grid_size=1)
    with pytest.raises(TypeError, match="integer"):
        deterministic_dense_state_indices(True)


def test_select_dense_points_preserves_pooled_and_worst_shared_thresholds() -> None:
    selected = select_dense_operating_points(
        _selection_states(),
        {"strict": {"pixel_budget": 1e-6, "component_budget": 1.0}},
    )["strict"]

    pooled = selected["source_pooled"]
    assert pooled["found"] is True
    assert pooled["threshold_logit_float32"] == 8.0
    assert pooled["operating_point"]["tp_objects"] == 12
    assert pooled["source_rows"]["a"]["fa_pixel"] == pytest.approx(2e-6)

    worst = selected["source_worst"]
    assert worst["found"] is True
    assert worst["threshold_logit_float32"] == 9.0
    assert worst["worst_domain_pd"] == pytest.approx(0.5)
    assert set(worst["source_rows"]) == {"a", "b"}


def test_select_dense_points_uses_lowest_threshold_after_pd_ties() -> None:
    states = _selection_states()
    states.append(
        _state(
            7.0,
            _row(8, fp_pixels=2, fp_components=2),
            _row(4, fp_pixels=0, fp_components=0),
        )
    )
    selected = select_dense_operating_points(
        states,
        {"strict": (1e-6, 1.0)},
    )["strict"]["source_pooled"]
    assert selected["threshold_logit_float32"] == 7.0


def test_select_dense_points_checks_pooled_raw_count_identity() -> None:
    states = _selection_states()
    states[1] = copy.deepcopy(states[1])
    states[1]["pooled"]["fp_pixels"] += 1  # type: ignore[index]
    with pytest.raises(ValueError, match="does not equal domain sum"):
        select_dense_operating_points(
            states,
            {"strict": {"pixel_budget": 1e-6, "component_budget": 1.0}},
        )


def test_realized_fa_sensitivity_uses_both_raw_caps() -> None:
    control = select_dense_operating_points(
        _selection_states(),
        {"strict": {"pixel_budget": 1e-6, "component_budget": 1.0}},
    )
    full_states = [
        _state(
            "+inf",
            _row(0, fp_pixels=0, fp_components=0),
            _row(0, fp_pixels=0, fp_components=0),
            reject=True,
        ),
        # Highest Pd, but exceeds control's pooled pixel cap and a's domain cap.
        _state(
            10.0,
            _row(8, fp_pixels=3, fp_components=1),
            _row(7, fp_pixels=0, fp_components=0),
        ),
        # Fits pooled caps, but not the worst-domain cap for domain a.
        _state(
            9.0,
            _row(7, fp_pixels=2, fp_components=2),
            _row(6, fp_pixels=0, fp_components=0),
        ),
        # Fits both pooled caps and every control worst-domain cap.
        _state(
            8.0,
            _row(6, fp_pixels=1, fp_components=1),
            _row(6, fp_pixels=1, fp_components=1),
        ),
    ]

    sensitivity = build_realized_fa_sensitivity(control, full_states)["strict"]
    pooled = sensitivity["source_pooled"]
    assert pooled["control_realized_fa_cap"] == {
        "fp_pixels": 2,
        "fp_components": 2,
    }
    assert pooled["threshold_logit_float32"] == 9.0
    assert pooled["operating_point"]["tp_objects"] == 13
    assert pooled["pd_delta_full_minus_control"] == pytest.approx(0.05)

    worst = sensitivity["source_worst"]
    assert worst["threshold_logit_float32"] == 8.0
    assert worst["control_realized_fa_caps_per_domain"]["a"] == {
        "fp_pixels": 1,
        "fp_components": 1,
    }
    assert worst["full_worst_domain_pd"] == pytest.approx(0.6)
    assert worst["worst_pd_delta_full_minus_control"] == pytest.approx(0.1)


def test_false_alarm_concentration_summarises_counts_and_optional_anatomy() -> None:
    summary = summarize_false_alarm_concentration(
        [
            {
                "image_id": "b",
                "fp_pixels": 3,
                "fp_components": 2,
                "false_component_areas": [1, 4],
                "unmatched_background_fp_pixels": 2,
                "matched_spillover_fp_pixels": 1,
            },
            {
                "image_id": "a",
                "fp_pixels": 1,
                "fp_components": 1,
                "false_component_areas": [1],
                "unmatched_background_fp_pixels": 1,
                "matched_spillover_fp_pixels": 0,
            },
        ]
    )

    pixels = summary["fp_pixels"]
    assert pixels["total"] == 4
    assert pixels["top_1"]["image_ids"] == ["b"]
    assert pixels["top_1"]["fraction"] == pytest.approx(0.75)
    assert pixels["top_10_percent"]["num_images"] == 1
    assert pixels["median"] == 2.0
    assert pixels["nonzero_image_fraction"] == 1.0
    assert pixels["hhi"] == pytest.approx(0.625)
    assert pixels["gini"] == pytest.approx(0.25)

    components = summary["fp_components"]
    assert components["total"] == 3
    assert components["max"] == 2
    areas = summary["false_component_area_distribution"]
    assert areas["count"] == 3
    assert areas["single_pixel_count"] == 2
    assert areas["single_pixel_fraction"] == pytest.approx(2 / 3)
    attribution = summary["fp_pixel_attribution"]
    assert attribution["total_fp_pixels"] == 4
    assert attribution["unmatched_background_fp_pixels"]["fraction"] == 0.75
    assert attribution["matched_spillover_fp_pixels"]["fraction"] == 0.25


def test_false_alarm_concentration_handles_zero_false_alarms() -> None:
    summary = summarize_false_alarm_concentration(
        [
            {"image_id": "a", "fp_pixels": 0, "fp_components": 0},
            {"image_id": "b", "fp_pixels": 0, "fp_components": 0},
        ]
    )
    for key in ("fp_pixels", "fp_components"):
        assert summary[key]["total"] == 0
        assert summary[key]["hhi"] == 0.0
        assert summary[key]["gini"] == 0.0
        assert summary[key]["nonzero_image_fraction"] == 0.0


def test_false_alarm_concentration_fails_closed_on_bad_attribution() -> None:
    with pytest.raises(ValueError, match="does not sum"):
        summarize_false_alarm_concentration(
            [
                {
                    "image_id": "x",
                    "fp_pixels": 2,
                    "fp_components": 0,
                    "unmatched_background_fp_pixels": 1,
                    "matched_spillover_fp_pixels": 0,
                }
            ]
        )
