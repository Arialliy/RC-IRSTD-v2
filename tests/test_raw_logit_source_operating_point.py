from __future__ import annotations

import numpy as np
import pytest

from evaluation.raw_logit_oracle import RawLogitSample
from evaluation.raw_logit_source_operating_point import (
    build_cross_domain_calibration_gap,
    enumerate_exact_shared_states,
    evaluate_domains_at_threshold,
    select_domain_oracles,
    select_exact_shared_source_operating_points,
)


def _sample(image_id: str, logits: list[list[float]], target: tuple[int, int]):
    raw = np.asarray(logits, dtype=np.float32)
    probability = (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)
    mask = np.zeros(raw.shape, dtype=bool)
    mask[target] = True
    return RawLogitSample(image_id, raw, probability, mask)


def _domains():
    return {
        "a": [
            _sample(
                "a1",
                [[-3, -2, -1], [-4, 10, 8], [-5, -6, -7]],
                (1, 1),
            )
        ],
        "b": [
            _sample(
                "b1",
                [[-8, -7, 9], [-6, 6, -5], [-4, -3, -2]],
                (1, 1),
            )
        ],
    }


def test_sparse_exact_states_match_legacy_full_image_evaluation() -> None:
    domains = _domains()
    exact = enumerate_exact_shared_states(domains, loose_pixel_budget=0.20)

    assert exact["exact_state_enumeration"] is True
    assert exact["states"][0]["all_reject_sentinel"] is True
    assert exact["states"][0]["threshold_logit_float32"] is None
    assert all(
        state["num_distinct_logit_states"] == len(exact["states"])
        for state in exact["states"]
    )
    for state in exact["states"]:
        legacy = evaluate_domains_at_threshold(
            domains, state["threshold_logit_float32"]
        )
        for name in ("a", "b"):
            for field in (
                "tp_objects",
                "gt_objects",
                "fp_components",
                "fp_pixels",
                "total_pixels",
            ):
                assert legacy["per_domain"][name][field] == state["per_domain"][name][field]


def test_exact_selector_uses_one_shared_threshold_and_raw_count_budgets() -> None:
    exact = enumerate_exact_shared_states(_domains(), loose_pixel_budget=0.20)
    selected = select_exact_shared_source_operating_points(
        exact,
        pixel_budget=0.20,
        component_budget=1_000_000.0,
    )

    pooled = selected["source_pooled"]
    worst = selected["source_worst"]
    assert pooled["found"] is True
    assert worst["found"] is True
    assert all(
        row["threshold_logit_float32"] == pooled["threshold_logit_float32"]
        for row in pooled["source_rows"].values()
    )
    assert all(
        row["threshold_logit_float32"] == worst["threshold_logit_float32"]
        for row in worst["source_rows"].values()
    )


def test_domain_oracles_are_diagnostic_and_cross_applied() -> None:
    domains = _domains()
    exact = enumerate_exact_shared_states(domains, loose_pixel_budget=0.20)
    shared = select_exact_shared_source_operating_points(
        exact,
        pixel_budget=0.20,
        component_budget=1_000_000.0,
    )
    oracles = select_domain_oracles(
        exact,
        pixel_budget=0.20,
        component_budget=1_000_000.0,
    )
    gap = build_cross_domain_calibration_gap(
        exact,
        shared,
        oracles,
        samples_by_domain=domains,
    )

    assert oracles["diagnostic_only"] is True
    assert set(oracles["domains"]) == {"a", "b"}
    assert gap["per_domain_oracles_not_used_for_formal_gate"] is True
    assert set(gap["cross_application"]) == {
        "a_threshold_on_b",
        "b_threshold_on_a",
    }
    assert set(gap["tail_rank_and_quantile"]) == {"a", "b"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"matching_rule": "centroid"}, "overlap"),
        ({"connectivity": 1}, "8-neighbor"),
        ({"min_component_area": 2}, "min_component_area=1"),
    ],
)
def test_protocol_drift_fails_closed(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        enumerate_exact_shared_states(_domains(), **kwargs)


def test_invalid_float64_logit_input_is_rejected() -> None:
    sample = _domains()["a"][0]
    invalid = RawLogitSample(
        sample.image_id,
        sample.logits.astype(np.float64),
        sample.probability,
        sample.mask,
    )
    with pytest.raises(ValueError, match="float32"):
        enumerate_exact_shared_states({"a": [invalid], "b": _domains()["b"]})
