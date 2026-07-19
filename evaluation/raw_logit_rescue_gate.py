"""Pure decision logic for the source-only raw-logit rescue gate.

This module deliberately performs no file or dataset I/O.  Callers must first
verify and hash-bind the source-only evidence, then pass one normalized point
per preregistered budget for the matched control and full RC-MSHNet.

Each normalized point has this shape::

    {
        "found": True,
        "pooled_pd": 0.70,
        "worst_pd": 0.65,
        "macro_pd": 0.69,
        "domain_pd": {"nudt": 0.68, "irstd1k": 0.70},
    }

Additional point fields (for example the selected threshold and state rank)
are permitted but ignored by this pure gate.  Missing/infeasible points,
non-finite values, or malformed normalized evidence fail closed as protocol
errors; scientifically valid evidence that misses a preregistered condition is
reported separately as a model-gate HOLD.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any


RESCUE_DECISION_SCHEMA = "rc-irstd-aaai27-raw-logit-rescue-decision-v1"
RESCUE_GO_TIER2 = "RESCUE_GO_TIER2"
RESCUE_HOLD_MODEL_GATE = "RESCUE_HOLD_MODEL_GATE"
RESCUE_HOLD_PROTOCOL = "RESCUE_HOLD_PROTOCOL"

REQUIRED_BUDGETS = ("strict", "medium", "loose")
REQUIRED_DOMAINS = ("irstd1k", "nudt")
STRICT_MACRO_PD_GAIN_MINIMUM = 0.01
DEFAULT_NUMERIC_ATOL = 1.0e-12


def _protocol_result(
    errors: list[str],
    *,
    numeric_atol: float | None,
) -> dict[str, Any]:
    """Build a non-authorizing, auditable protocol-HOLD result."""

    return {
        "schema_version": RESCUE_DECISION_SCHEMA,
        "decision": RESCUE_HOLD_PROTOCOL,
        "gate_valid": False,
        "authorizes_tier2": False,
        "authorizes_outer_target_image_access": False,
        "authorizes_outer_target_label_access": False,
        "required_budgets": list(REQUIRED_BUDGETS),
        "required_domains": list(REQUIRED_DOMAINS),
        "numeric_atol": numeric_atol,
        "strict_macro_pd_gain_minimum": STRICT_MACRO_PD_GAIN_MINIMUM,
        "evidence": {},
        "conditions": [],
        "failed_conditions": [],
        "protocol_errors": errors,
    }


def _finite_pd(value: Any, *, field: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        errors.append(f"{field} must be a real number")
        return None
    number = float(value)
    if not math.isfinite(number):
        errors.append(f"{field} must be finite")
        return None
    if not 0.0 <= number <= 1.0:
        errors.append(f"{field} must lie in [0, 1]")
        return None
    return number


def _normalize_points(
    value: Any,
    *,
    role: str,
    numeric_atol: float,
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        errors.append(f"{role}_points must be a mapping")
        return {}

    budget_keys = set(value)
    expected_budget_keys = set(REQUIRED_BUDGETS)
    if budget_keys != expected_budget_keys:
        missing = sorted(expected_budget_keys - budget_keys)
        extra = sorted(budget_keys - expected_budget_keys, key=str)
        errors.append(
            f"{role}_points budget schema mismatch: missing={missing}, extra={extra}"
        )

    normalized: dict[str, dict[str, Any]] = {}
    for budget in REQUIRED_BUDGETS:
        if budget not in value:
            continue
        point = value[budget]
        prefix = f"{role}_points.{budget}"
        if not isinstance(point, Mapping):
            errors.append(f"{prefix} must be a mapping")
            continue
        found = point.get("found")
        if found is not True:
            if found is False:
                errors.append(f"{prefix} is missing or infeasible (found=false)")
            else:
                errors.append(f"{prefix}.found must be boolean true")
            continue

        pooled_pd = _finite_pd(
            point.get("pooled_pd"), field=f"{prefix}.pooled_pd", errors=errors
        )
        worst_pd = _finite_pd(
            point.get("worst_pd"), field=f"{prefix}.worst_pd", errors=errors
        )
        macro_pd = _finite_pd(
            point.get("macro_pd"), field=f"{prefix}.macro_pd", errors=errors
        )

        raw_domain_pd = point.get("domain_pd")
        domain_pd: dict[str, float] = {}
        if not isinstance(raw_domain_pd, Mapping):
            errors.append(f"{prefix}.domain_pd must be a mapping")
        else:
            domain_keys = set(raw_domain_pd)
            expected_domain_keys = set(REQUIRED_DOMAINS)
            if domain_keys != expected_domain_keys:
                missing = sorted(expected_domain_keys - domain_keys)
                extra = sorted(domain_keys - expected_domain_keys, key=str)
                errors.append(
                    f"{prefix}.domain_pd schema mismatch: "
                    f"missing={missing}, extra={extra}"
                )
            for domain in REQUIRED_DOMAINS:
                if domain not in raw_domain_pd:
                    continue
                parsed = _finite_pd(
                    raw_domain_pd[domain],
                    field=f"{prefix}.domain_pd.{domain}",
                    errors=errors,
                )
                if parsed is not None:
                    domain_pd[domain] = parsed

        complete = (
            pooled_pd is not None
            and worst_pd is not None
            and macro_pd is not None
            and set(domain_pd) == set(REQUIRED_DOMAINS)
        )
        if not complete:
            continue
        derived_macro = sum(domain_pd.values()) / len(REQUIRED_DOMAINS)
        if abs(macro_pd - derived_macro) > numeric_atol:
            errors.append(
                f"{prefix}.macro_pd disagrees with the mean domain Pd: "
                f"reported={macro_pd}, derived={derived_macro}"
            )
            continue
        normalized[budget] = {
            "found": True,
            "pooled_pd": pooled_pd,
            "worst_pd": worst_pd,
            "macro_pd": macro_pd,
            "domain_pd": {
                domain: domain_pd[domain] for domain in REQUIRED_DOMAINS
            },
        }
    return normalized


def _condition(
    condition_id: str,
    *,
    budget: str,
    metric: str,
    actual_delta: float,
    minimum_delta: float,
    numeric_atol: float,
    domain: str | None = None,
) -> dict[str, Any]:
    passed = actual_delta >= minimum_delta - numeric_atol
    result: dict[str, Any] = {
        "condition_id": condition_id,
        "budget": budget,
        "metric": metric,
        "operator": "greater_than_or_equal",
        "minimum_delta": minimum_delta,
        "actual_delta": actual_delta,
        "numeric_atol": numeric_atol,
        "passed": passed,
    }
    if domain is not None:
        result["domain"] = domain
    return result


def evaluate_rescue_decision(
    control_points: Mapping[str, Mapping[str, Any]],
    full_points: Mapping[str, Mapping[str, Any]],
    *,
    numeric_atol: float = DEFAULT_NUMERIC_ATOL,
) -> dict[str, Any]:
    """Evaluate the preregistered raw-logit rescue decision conditions.

    The function returns one of ``RESCUE_GO_TIER2``,
    ``RESCUE_HOLD_MODEL_GATE``, or ``RESCUE_HOLD_PROTOCOL`` and never grants
    outer-target image or label access.
    """

    if (
        isinstance(numeric_atol, bool)
        or not isinstance(numeric_atol, Real)
        or not math.isfinite(float(numeric_atol))
        or float(numeric_atol) < 0.0
    ):
        return _protocol_result(
            ["numeric_atol must be a finite non-negative real number"],
            numeric_atol=None,
        )
    tolerance = float(numeric_atol)
    errors: list[str] = []
    control = _normalize_points(
        control_points,
        role="control",
        numeric_atol=tolerance,
        errors=errors,
    )
    full = _normalize_points(
        full_points,
        role="full",
        numeric_atol=tolerance,
        errors=errors,
    )
    if errors or set(control) != set(REQUIRED_BUDGETS) or set(full) != set(
        REQUIRED_BUDGETS
    ):
        if not errors:
            errors.append("normalized evidence does not cover every required point")
        return _protocol_result(errors, numeric_atol=tolerance)

    evidence: dict[str, Any] = {}
    conditions: list[dict[str, Any]] = []
    for budget in REQUIRED_BUDGETS:
        control_point = control[budget]
        full_point = full[budget]
        pooled_delta = full_point["pooled_pd"] - control_point["pooled_pd"]
        worst_delta = full_point["worst_pd"] - control_point["worst_pd"]
        macro_delta = full_point["macro_pd"] - control_point["macro_pd"]
        domain_delta = {
            domain: (
                full_point["domain_pd"][domain]
                - control_point["domain_pd"][domain]
            )
            for domain in REQUIRED_DOMAINS
        }
        evidence[budget] = {
            "control": control_point,
            "full": full_point,
            "pooled_pd_delta": pooled_delta,
            "worst_pd_delta": worst_delta,
            "macro_pd_delta": macro_delta,
            "domain_pd_delta": domain_delta,
        }
        conditions.append(
            _condition(
                f"{budget}_pooled_pd_non_degraded",
                budget=budget,
                metric="pooled_pd_delta",
                actual_delta=pooled_delta,
                minimum_delta=0.0,
                numeric_atol=tolerance,
            )
        )
        conditions.append(
            _condition(
                f"{budget}_worst_pd_non_degraded",
                budget=budget,
                metric="worst_pd_delta",
                actual_delta=worst_delta,
                minimum_delta=0.0,
                numeric_atol=tolerance,
            )
        )

    strict = evidence["strict"]
    conditions.append(
        _condition(
            "strict_macro_pd_gain_minimum",
            budget="strict",
            metric="macro_pd_delta",
            actual_delta=strict["macro_pd_delta"],
            minimum_delta=STRICT_MACRO_PD_GAIN_MINIMUM,
            numeric_atol=tolerance,
        )
    )
    for domain in REQUIRED_DOMAINS:
        conditions.append(
            _condition(
                f"strict_{domain}_pd_non_degraded",
                budget="strict",
                domain=domain,
                metric="domain_pd_delta",
                actual_delta=strict["domain_pd_delta"][domain],
                minimum_delta=0.0,
                numeric_atol=tolerance,
            )
        )

    failed = [
        condition["condition_id"]
        for condition in conditions
        if condition["passed"] is not True
    ]
    decision = RESCUE_GO_TIER2 if not failed else RESCUE_HOLD_MODEL_GATE
    return {
        "schema_version": RESCUE_DECISION_SCHEMA,
        "decision": decision,
        "gate_valid": True,
        "authorizes_tier2": decision == RESCUE_GO_TIER2,
        "authorizes_outer_target_image_access": False,
        "authorizes_outer_target_label_access": False,
        "required_budgets": list(REQUIRED_BUDGETS),
        "required_domains": list(REQUIRED_DOMAINS),
        "numeric_atol": tolerance,
        "strict_macro_pd_gain_minimum": STRICT_MACRO_PD_GAIN_MINIMUM,
        "evidence": evidence,
        "conditions": conditions,
        "failed_conditions": failed,
        "protocol_errors": [],
    }


__all__ = [
    "DEFAULT_NUMERIC_ATOL",
    "REQUIRED_BUDGETS",
    "REQUIRED_DOMAINS",
    "RESCUE_DECISION_SCHEMA",
    "RESCUE_GO_TIER2",
    "RESCUE_HOLD_MODEL_GATE",
    "RESCUE_HOLD_PROTOCOL",
    "STRICT_MACRO_PD_GAIN_MINIMUM",
    "evaluate_rescue_decision",
]
