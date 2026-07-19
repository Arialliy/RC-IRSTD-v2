from __future__ import annotations

import copy

import pytest

from evaluation.raw_logit_rescue_gate import (
    RESCUE_GO_TIER2,
    RESCUE_HOLD_MODEL_GATE,
    RESCUE_HOLD_PROTOCOL,
    evaluate_rescue_decision,
)


def _point(nudt: float, irstd: float, *, pooled: float | None = None, worst: float | None = None):
    macro = (nudt + irstd) / 2.0
    return {
        "found": True,
        "pooled_pd": macro if pooled is None else pooled,
        "worst_pd": min(nudt, irstd) if worst is None else worst,
        "macro_pd": macro,
        "domain_pd": {"nudt": nudt, "irstd1k": irstd},
        "threshold_logit": 19.0,
        "threshold_state_rank": 3,
    }


def _evidence(control_value: float = 0.60, full_value: float = 0.62):
    control = {
        budget: _point(control_value, control_value)
        for budget in ("strict", "medium", "loose")
    }
    full = {
        budget: _point(full_value, full_value)
        for budget in ("strict", "medium", "loose")
    }
    return control, full


def _condition(result: dict, condition_id: str) -> dict:
    return next(
        item for item in result["conditions"] if item["condition_id"] == condition_id
    )


def test_all_preregistered_conditions_pass_and_only_tier2_is_authorized() -> None:
    control, full = _evidence()
    result = evaluate_rescue_decision(control, full)

    assert result["decision"] == RESCUE_GO_TIER2
    assert result["gate_valid"] is True
    assert result["authorizes_tier2"] is True
    assert result["authorizes_outer_target_image_access"] is False
    assert result["authorizes_outer_target_label_access"] is False
    assert result["failed_conditions"] == []
    assert result["protocol_errors"] == []
    assert len(result["conditions"]) == 9
    assert all(item["passed"] is True for item in result["conditions"])
    assert result["evidence"]["strict"]["macro_pd_delta"] == pytest.approx(0.02)


def test_strict_macro_gain_exactly_one_percentage_point_passes() -> None:
    control, full = _evidence(control_value=0.60, full_value=0.61)
    result = evaluate_rescue_decision(control, full, numeric_atol=0.0)

    assert result["decision"] == RESCUE_GO_TIER2
    condition = _condition(result, "strict_macro_pd_gain_minimum")
    assert condition["actual_delta"] == pytest.approx(0.01)
    assert condition["passed"] is True


def test_numeric_tolerance_is_applied_to_the_strict_boundary() -> None:
    control, full = _evidence(control_value=0.60, full_value=0.6095)

    within = evaluate_rescue_decision(control, full, numeric_atol=5.0e-4)
    outside = evaluate_rescue_decision(control, full, numeric_atol=4.0e-4)

    assert within["decision"] == RESCUE_GO_TIER2
    assert outside["decision"] == RESCUE_HOLD_MODEL_GATE
    assert outside["failed_conditions"] == ["strict_macro_pd_gain_minimum"]


def test_valid_evidence_below_strict_macro_gain_is_model_hold() -> None:
    control, full = _evidence(control_value=0.60, full_value=0.609)
    result = evaluate_rescue_decision(control, full)

    assert result["decision"] == RESCUE_HOLD_MODEL_GATE
    assert result["gate_valid"] is True
    assert result["authorizes_tier2"] is False
    assert result["protocol_errors"] == []
    assert result["failed_conditions"] == ["strict_macro_pd_gain_minimum"]


def test_strict_domain_degradation_is_model_hold_even_when_macro_gain_passes() -> None:
    control, full = _evidence()
    full["strict"] = _point(0.599, 0.641)
    result = evaluate_rescue_decision(control, full)

    assert result["decision"] == RESCUE_HOLD_MODEL_GATE
    assert "strict_nudt_pd_non_degraded" in result["failed_conditions"]
    assert _condition(result, "strict_macro_pd_gain_minimum")["passed"] is True


@pytest.mark.parametrize("budget", ["strict", "medium", "loose"])
def test_each_budget_pooled_degradation_is_model_hold(budget: str) -> None:
    control, full = _evidence()
    full[budget]["pooled_pd"] = control[budget]["pooled_pd"] - 0.001
    result = evaluate_rescue_decision(control, full)

    assert result["decision"] == RESCUE_HOLD_MODEL_GATE
    assert f"{budget}_pooled_pd_non_degraded" in result["failed_conditions"]


@pytest.mark.parametrize("budget", ["strict", "medium", "loose"])
def test_each_budget_worst_degradation_is_model_hold(budget: str) -> None:
    control, full = _evidence()
    full[budget]["worst_pd"] = control[budget]["worst_pd"] - 0.001
    result = evaluate_rescue_decision(control, full)

    assert result["decision"] == RESCUE_HOLD_MODEL_GATE
    assert f"{budget}_worst_pd_non_degraded" in result["failed_conditions"]


@pytest.mark.parametrize(
    "mutator, expected_fragment",
    [
        (lambda control, full: full.pop("loose"), "budget schema mismatch"),
        (
            lambda control, full: full["strict"].__setitem__("found", False),
            "missing or infeasible",
        ),
        (
            lambda control, full: full["strict"].__setitem__("pooled_pd", float("nan")),
            "must be finite",
        ),
        (
            lambda control, full: full["strict"].__setitem__("pooled_pd", 1.1),
            "must lie in [0, 1]",
        ),
        (
            lambda control, full: full["strict"]["domain_pd"].pop("nudt"),
            "domain_pd schema mismatch",
        ),
        (
            lambda control, full: full["strict"].__setitem__("macro_pd", 0.9),
            "disagrees with the mean domain Pd",
        ),
    ],
)
def test_malformed_or_incomplete_evidence_is_protocol_hold(
    mutator, expected_fragment: str
) -> None:
    control, full = _evidence()
    mutator(control, full)
    result = evaluate_rescue_decision(control, full)

    assert result["decision"] == RESCUE_HOLD_PROTOCOL
    assert result["gate_valid"] is False
    assert result["authorizes_tier2"] is False
    assert result["conditions"] == []
    assert any(expected_fragment in error for error in result["protocol_errors"])


@pytest.mark.parametrize("numeric_atol", [-1.0, float("nan"), True, "1e-12"])
def test_invalid_numeric_tolerance_is_protocol_hold(numeric_atol) -> None:
    control, full = _evidence()
    result = evaluate_rescue_decision(
        control, full, numeric_atol=numeric_atol
    )

    assert result["decision"] == RESCUE_HOLD_PROTOCOL
    assert result["numeric_atol"] is None
    assert result["conditions"] == []


def test_input_evidence_is_not_mutated() -> None:
    control, full = _evidence()
    before = copy.deepcopy((control, full))

    evaluate_rescue_decision(control, full)

    assert (control, full) == before
