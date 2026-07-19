"""Pure diagnostics for the source-only raw-logit rescue audit.

The functions in this module do not read score maps, choose deployment
actions, or write authorization artifacts.  They operate only on exact-state
raw counts supplied by the rescue evaluator.  In particular, all false-alarm
constraints are checked from integer counts rather than rounded rates.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Mapping, Sequence

import numpy as np


_COUNT_FIELDS = (
    "tp_objects",
    "gt_objects",
    "fp_components",
    "fp_pixels",
    "total_pixels",
)


def deterministic_dense_state_indices(
    num_states: int,
    grid_size: int = 1024,
) -> list[int]:
    """Return an increasing deterministic subsample of exact-state indices.

    The caller's state sequence is required to place reject-all at index zero
    and the lowest retained finite tail state at ``num_states - 1``.  Both are
    always included.  Integer arithmetic avoids platform-dependent rounding,
    and the final de-duplication is defensive for future sampling policies.
    """

    num_states = _positive_int(num_states, name="num_states")
    grid_size = _positive_int(grid_size, name="grid_size")
    if num_states == 1:
        return [0]
    if grid_size < 2:
        raise ValueError("grid_size must be at least two when num_states > 1")
    if num_states <= grid_size:
        return list(range(num_states))

    # Here grid_size <= num_states, so adjacent values are already distinct.
    # Keep the explicit de-duplication to make the endpoint contract obvious.
    sampled = [
        (position * (num_states - 1)) // (grid_size - 1)
        for position in range(grid_size)
    ]
    sampled.extend((0, num_states - 1))
    return sorted(set(sampled))


def select_dense_operating_points(
    states: Sequence[Mapping[str, Any]],
    budgets: Mapping[str, Any] | Sequence[Any],
) -> dict[str, dict[str, Any]]:
    """Select pooled and worst-domain points on a shared-threshold state set.

    Each state must contain ``pooled`` and ``per_domain`` raw-count rows plus
    ``threshold_logit_float32`` and ``all_reject_sentinel``.  Pooled selection
    aggregates raw counts before checking both budgets.  Worst-domain
    selection requires every domain to satisfy both budgets at the same raw
    logit threshold.
    """

    normalised = _normalise_states(states)
    results: dict[str, dict[str, Any]] = {}
    for name, pixel_budget, component_budget in _normalise_budgets(budgets):
        pooled_candidates = [
            state
            for state in normalised
            if _within_rate_budget(
                state["pooled"],
                pixel_budget=pixel_budget,
                component_budget=component_budget,
            )
        ]
        worst_candidates = [
            state
            for state in normalised
            if all(
                _within_rate_budget(
                    row,
                    pixel_budget=pixel_budget,
                    component_budget=component_budget,
                )
                for row in state["per_domain"].values()
            )
        ]
        results[name] = {
            "pixel_budget": float(pixel_budget),
            "component_budget": float(component_budget),
            "source_pooled": _pooled_selection_payload(pooled_candidates),
            "source_worst": _worst_selection_payload(worst_candidates),
        }
    return results


def build_realized_fa_sensitivity(
    control_points: Mapping[str, Any],
    full_states: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Match full-model operating points to control's realised raw FA counts.

    For pooled sensitivity, the control pooled point supplies one FP-pixel and
    one FP-component cap.  For worst-domain sensitivity, every domain supplies
    its own two caps, and all of them must hold at one shared threshold.
    """

    states = _normalise_states(full_states)
    domain_names = tuple(states[0]["per_domain"])
    raw_control = control_points.get("results", control_points)
    if not isinstance(raw_control, Mapping) or not raw_control:
        raise ValueError("control_points must contain at least one budget")

    output: dict[str, dict[str, Any]] = {}
    for budget_name, raw_budget_points in raw_control.items():
        if not isinstance(budget_name, str) or not budget_name:
            raise ValueError("control point names must be non-empty strings")
        if not isinstance(raw_budget_points, Mapping):
            raise TypeError(f"control_points[{budget_name!r}] must be a mapping")
        pooled_control = _require_selected_point(
            raw_budget_points.get("source_pooled"),
            name=f"{budget_name}.source_pooled",
        )
        worst_control = _require_selected_point(
            raw_budget_points.get("source_worst"),
            name=f"{budget_name}.source_worst",
        )

        pooled_cap = {
            "fp_pixels": _count(pooled_control["operating_point"], "fp_pixels"),
            "fp_components": _count(
                pooled_control["operating_point"], "fp_components"
            ),
        }
        pooled_candidates = [
            state
            for state in states
            if _within_raw_cap(state["pooled"], pooled_cap)
        ]
        pooled_result = _pooled_selection_payload(pooled_candidates)
        pooled_result["control_realized_fa_cap"] = dict(pooled_cap)
        pooled_result["control_pd"] = float(
            _pd_fraction(pooled_control["operating_point"])
        )
        if pooled_result["found"]:
            pooled_result["full_pd"] = float(
                _pd_fraction(pooled_result["operating_point"])
            )
            pooled_result["pd_delta_full_minus_control"] = float(
                _pd_fraction(pooled_result["operating_point"])
                - _pd_fraction(pooled_control["operating_point"])
            )

        control_domain_rows = worst_control.get(
            "source_rows", worst_control.get("per_domain")
        )
        if not isinstance(control_domain_rows, Mapping):
            raise ValueError(
                f"{budget_name}.source_worst lacks source_rows/per_domain"
            )
        if set(control_domain_rows) != set(domain_names):
            raise ValueError(
                f"{budget_name}.source_worst domain set differs from full states"
            )
        per_domain_caps = {
            domain: {
                "fp_pixels": _count(control_domain_rows[domain], "fp_pixels"),
                "fp_components": _count(
                    control_domain_rows[domain], "fp_components"
                ),
            }
            for domain in domain_names
        }
        worst_candidates = [
            state
            for state in states
            if all(
                _within_raw_cap(state["per_domain"][domain], per_domain_caps[domain])
                for domain in domain_names
            )
        ]
        worst_result = _worst_selection_payload(worst_candidates)
        worst_result["control_realized_fa_caps_per_domain"] = per_domain_caps
        control_worst_pd = min(
            _pd_fraction(control_domain_rows[domain]) for domain in domain_names
        )
        worst_result["control_worst_domain_pd"] = float(control_worst_pd)
        if worst_result["found"]:
            full_worst_pd = min(
                _pd_fraction(row) for row in worst_result["source_rows"].values()
            )
            worst_result["full_worst_domain_pd"] = float(full_worst_pd)
            worst_result["worst_pd_delta_full_minus_control"] = float(
                full_worst_pd - control_worst_pd
            )

        output[budget_name] = {
            "source_pooled": pooled_result,
            "source_worst": worst_result,
        }
    return output


def summarize_false_alarm_concentration(
    per_image_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarise per-image false-alarm concentration and optional anatomy."""

    if not isinstance(per_image_rows, Sequence) or isinstance(
        per_image_rows, (str, bytes)
    ):
        raise TypeError("per_image_rows must be a sequence of mappings")
    if not per_image_rows:
        raise ValueError("per_image_rows must not be empty")

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_row in enumerate(per_image_rows):
        if not isinstance(raw_row, Mapping):
            raise TypeError(f"per_image_rows[{index}] must be a mapping")
        image_id = str(raw_row.get("image_id", index))
        if not image_id:
            raise ValueError(f"per_image_rows[{index}] has an empty image_id")
        if image_id in seen_ids:
            raise ValueError(f"duplicate image_id in per_image_rows: {image_id}")
        seen_ids.add(image_id)
        rows.append(
            {
                **dict(raw_row),
                "image_id": image_id,
                "fp_pixels": _count(raw_row, "fp_pixels"),
                "fp_components": _count(raw_row, "fp_components"),
            }
        )

    output: dict[str, Any] = {
        "num_images": len(rows),
        "fp_pixels": _concentration_for_key(rows, "fp_pixels"),
        "fp_components": _concentration_for_key(rows, "fp_components"),
    }

    area_presence = ["false_component_areas" in row for row in rows]
    if any(area_presence):
        if not all(area_presence):
            raise ValueError(
                "false_component_areas must be present for every image or none"
            )
        areas: list[int] = []
        for row in rows:
            raw_areas = row["false_component_areas"]
            if not isinstance(raw_areas, Sequence) or isinstance(
                raw_areas, (str, bytes)
            ):
                raise TypeError("false_component_areas must be a sequence")
            local = [
                _positive_int(value, name="false component area")
                for value in raw_areas
            ]
            if len(local) != int(row["fp_components"]):
                raise ValueError(
                    "false_component_areas length must equal fp_components "
                    f"for image {row['image_id']}"
                )
            areas.extend(local)
        output["false_component_area_distribution"] = _area_distribution(areas)

    attribution_keys = (
        "unmatched_background_fp_pixels",
        "matched_spillover_fp_pixels",
    )
    attribution_presence = [
        any(key in row for key in attribution_keys) for row in rows
    ]
    if any(attribution_presence):
        if not all(
            all(key in row for key in attribution_keys) for row in rows
        ):
            raise ValueError(
                "both FP-pixel attribution fields must be present for every image"
            )
        totals = {key: 0 for key in attribution_keys}
        for row in rows:
            local = {key: _count(row, key) for key in attribution_keys}
            if sum(local.values()) != int(row["fp_pixels"]):
                raise ValueError(
                    "FP-pixel attribution does not sum to fp_pixels for image "
                    f"{row['image_id']}"
                )
            for key, value in local.items():
                totals[key] += value
        total_fp = int(sum(int(row["fp_pixels"]) for row in rows))
        output["fp_pixel_attribution"] = {
            key: {
                "value": value,
                "fraction": float(value / total_fp) if total_fp else 0.0,
            }
            for key, value in totals.items()
        }
        output["fp_pixel_attribution"]["total_fp_pixels"] = total_fp

    return output


def _normalise_states(
    states: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
        raise TypeError("states must be a sequence of mappings")
    if not states:
        raise ValueError("states must not be empty")

    normalised: list[dict[str, Any]] = []
    expected_domains: tuple[str, ...] | None = None
    expected_denominators: dict[str, tuple[int, int]] | None = None
    reject_count = 0
    finite_thresholds: set[float] = set()
    for index, raw_state in enumerate(states):
        if not isinstance(raw_state, Mapping):
            raise TypeError(f"states[{index}] must be a mapping")
        reject = raw_state.get("all_reject_sentinel")
        if not isinstance(reject, bool):
            raise TypeError(
                f"states[{index}].all_reject_sentinel must be boolean"
            )
        if reject:
            reject_count += 1
            threshold: float | str = "+inf"
        else:
            raw_threshold = raw_state.get("threshold_logit_float32")
            if isinstance(raw_threshold, bool) or not isinstance(
                raw_threshold, (int, float, np.integer, np.floating)
            ):
                raise TypeError(
                    f"states[{index}].threshold_logit_float32 must be numeric"
                )
            threshold = float(np.float32(raw_threshold))
            if not math.isfinite(threshold):
                raise ValueError(f"states[{index}] has a non-finite threshold")
            if threshold in finite_thresholds:
                raise ValueError(f"duplicate finite threshold state: {threshold}")
            finite_thresholds.add(threshold)

        raw_domains = raw_state.get("per_domain")
        if not isinstance(raw_domains, Mapping) or not raw_domains:
            raise ValueError(f"states[{index}].per_domain must be a non-empty mapping")
        domains = tuple(sorted(raw_domains))
        if any(not isinstance(name, str) or not name for name in domains):
            raise ValueError("domain names must be non-empty strings")
        if expected_domains is None:
            expected_domains = domains
        elif domains != expected_domains:
            raise ValueError("all states must contain the same domain set")

        per_domain = {
            domain: _normalise_count_row(
                raw_domains[domain], name=f"states[{index}].per_domain[{domain!r}]"
            )
            for domain in domains
        }
        pooled = _normalise_count_row(
            raw_state.get("pooled"), name=f"states[{index}].pooled"
        )
        for field in _COUNT_FIELDS:
            expected = sum(row[field] for row in per_domain.values())
            if pooled[field] != expected:
                raise ValueError(
                    f"states[{index}].pooled.{field} does not equal domain sum"
                )

        denominators = {
            domain: (row["gt_objects"], row["total_pixels"])
            for domain, row in per_domain.items()
        }
        if expected_denominators is None:
            expected_denominators = denominators
        elif denominators != expected_denominators:
            raise ValueError("domain gt_objects/total_pixels must be state-invariant")

        if reject and any(
            pooled[field] != 0
            for field in ("tp_objects", "fp_components", "fp_pixels")
        ):
            raise ValueError("reject-all state must have zero TP and false alarms")
        normalised.append(
            {
                "state_index": index,
                "threshold_logit_float32": threshold,
                "all_reject_sentinel": reject,
                "pooled": pooled,
                "per_domain": per_domain,
            }
        )
    if reject_count != 1:
        raise ValueError("states must contain exactly one reject-all sentinel")
    return normalised


def _normalise_count_row(raw_row: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(raw_row, Mapping):
        raise TypeError(f"{name} must be a mapping")
    row = {field: _count(raw_row, field) for field in _COUNT_FIELDS}
    if row["total_pixels"] <= 0:
        raise ValueError(f"{name}.total_pixels must be positive")
    if row["tp_objects"] > row["gt_objects"]:
        raise ValueError(f"{name}.tp_objects exceeds gt_objects")
    row.update(_derived_metrics(row))
    return row


def _derived_metrics(row: Mapping[str, int]) -> dict[str, float]:
    total_pixels = int(row["total_pixels"])
    gt_objects = int(row["gt_objects"])
    return {
        "pd": float(int(row["tp_objects"]) / gt_objects) if gt_objects else 0.0,
        "fa_pixel": float(int(row["fp_pixels"]) / total_pixels),
        "fa_component_mp": float(
            int(row["fp_components"]) / (total_pixels / 1_000_000.0)
        ),
    }


def _normalise_budgets(
    budgets: Mapping[str, Any] | Sequence[Any],
) -> list[tuple[str, Fraction, Fraction]]:
    if isinstance(budgets, Mapping):
        raw_items = list(budgets.items())
    elif isinstance(budgets, Sequence) and not isinstance(budgets, (str, bytes)):
        raw_items = []
        for index, item in enumerate(budgets):
            if isinstance(item, Mapping):
                name = item.get("name")
                raw_items.append((name, item))
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                if len(item) != 3:
                    raise ValueError(
                        f"budgets[{index}] must be (name, pixel, component)"
                    )
                raw_items.append((item[0], (item[1], item[2])))
            else:
                raise TypeError(f"budgets[{index}] has an unsupported form")
    else:
        raise TypeError("budgets must be a mapping or sequence")
    if not raw_items:
        raise ValueError("budgets must not be empty")

    output: list[tuple[str, Fraction, Fraction]] = []
    seen: set[str] = set()
    for raw_name, raw_spec in raw_items:
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("budget names must be non-empty strings")
        if raw_name in seen:
            raise ValueError(f"duplicate budget name: {raw_name}")
        seen.add(raw_name)
        if isinstance(raw_spec, Mapping):
            pixel = _first_present(
                raw_spec,
                ("pixel_budget", "fa_pixel", "pixel_fa", "pixel"),
                name=f"{raw_name} pixel budget",
            )
            component = _first_present(
                raw_spec,
                (
                    "component_budget",
                    "fa_component_mp",
                    "component_fa_mp",
                    "component",
                ),
                name=f"{raw_name} component budget",
            )
        elif isinstance(raw_spec, Sequence) and not isinstance(
            raw_spec, (str, bytes)
        ):
            if len(raw_spec) != 2:
                raise ValueError(f"budget {raw_name!r} must contain two values")
            pixel, component = raw_spec
        else:
            raise TypeError(f"budget {raw_name!r} has an unsupported form")
        output.append(
            (
                raw_name,
                _positive_fraction(pixel, name=f"{raw_name} pixel budget"),
                _positive_fraction(
                    component, name=f"{raw_name} component budget"
                ),
            )
        )
    return output


def _within_rate_budget(
    row: Mapping[str, int],
    *,
    pixel_budget: Fraction,
    component_budget: Fraction,
) -> bool:
    total_pixels = int(row["total_pixels"])
    pixel_ok = (
        int(row["fp_pixels"]) * pixel_budget.denominator
        <= pixel_budget.numerator * total_pixels
    )
    component_ok = (
        int(row["fp_components"])
        * 1_000_000
        * component_budget.denominator
        <= component_budget.numerator * total_pixels
    )
    return pixel_ok and component_ok


def _within_raw_cap(row: Mapping[str, int], cap: Mapping[str, int]) -> bool:
    return int(row["fp_pixels"]) <= int(cap["fp_pixels"]) and int(
        row["fp_components"]
    ) <= int(cap["fp_components"])


def _pooled_selection_payload(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not candidates:
        return {
            "found": False,
            "strategy": "maximize_pooled_pd",
            "tie_break": ["lowest_threshold"],
            "state_index": None,
            "operating_point": None,
            "source_rows": None,
        }
    selected = min(
        candidates,
        key=lambda state: (
            -_pd_fraction(state["pooled"]),
            _threshold_sort_key(state),
        ),
    )
    return {
        "found": True,
        "strategy": "aggregate_raw_counts_then_maximize_pooled_pd",
        "tie_break": ["lowest_threshold"],
        **_selected_state_payload(selected),
    }


def _worst_selection_payload(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not candidates:
        return {
            "found": False,
            "strategy": "require_every_domain_cap_then_maximize_worst_domain_pd",
            "tie_break": ["maximize_pooled_pd", "lowest_threshold"],
            "state_index": None,
            "operating_point": None,
            "source_rows": None,
            "worst_domain_pd": None,
            "worst_domain_names": [],
        }

    def key(state: Mapping[str, Any]) -> tuple[Fraction, Fraction, float]:
        worst = min(_pd_fraction(row) for row in state["per_domain"].values())
        return (-worst, -_pd_fraction(state["pooled"]), _threshold_sort_key(state))

    selected = min(candidates, key=key)
    domain_pd = {
        name: _pd_fraction(row) for name, row in selected["per_domain"].items()
    }
    worst_pd = min(domain_pd.values())
    return {
        "found": True,
        "strategy": "require_every_source_cap_then_maximize_worst_domain_pd",
        "tie_break": ["maximize_pooled_pd", "lowest_threshold"],
        **_selected_state_payload(selected),
        "worst_domain_pd": float(worst_pd),
        "worst_domain_names": [
            name for name, value in domain_pd.items() if value == worst_pd
        ],
    }


def _selected_state_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    threshold = state["threshold_logit_float32"]
    reject = bool(state["all_reject_sentinel"])

    def decorate(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **dict(row),
            "threshold_logit_float32": threshold,
            "all_reject_sentinel": reject,
        }

    return {
        "state_index": int(state["state_index"]),
        "threshold_logit_float32": threshold,
        "all_reject_sentinel": reject,
        "operating_point": decorate(state["pooled"]),
        "source_rows": {
            name: decorate(row) for name, row in state["per_domain"].items()
        },
    }


def _threshold_sort_key(state: Mapping[str, Any]) -> float:
    if state["all_reject_sentinel"]:
        return math.inf
    return float(state["threshold_logit_float32"])


def _require_selected_point(raw: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} is missing")
    if raw.get("found") is not True:
        raise ValueError(f"{name} has no selected point")
    if not isinstance(raw.get("operating_point"), Mapping):
        raise ValueError(f"{name} lacks operating_point")
    return raw


def _concentration_for_key(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, Any]:
    values = np.asarray([int(row[key]) for row in rows], dtype=np.int64)
    total = int(values.sum(dtype=np.int64))
    order = sorted(
        range(len(rows)),
        key=lambda index: (-int(values[index]), str(rows[index]["image_id"])),
    )
    specs = {
        "top_1": 1,
        "top_5": 5,
        "top_10": 10,
        "top_1_percent": max(1, math.ceil(len(rows) * 0.01)),
        "top_5_percent": max(1, math.ceil(len(rows) * 0.05)),
        "top_10_percent": max(1, math.ceil(len(rows) * 0.10)),
    }
    output: dict[str, Any] = {"total": total}
    for label, requested in specs.items():
        selected = order[: min(requested, len(rows))]
        subtotal = int(values[selected].sum(dtype=np.int64))
        output[label] = {
            "num_images": len(selected),
            "value": subtotal,
            "fraction": float(subtotal / total) if total else 0.0,
            "image_ids": [str(rows[index]["image_id"]) for index in selected],
        }

    quantiles = np.quantile(values.astype(np.float64), [0.5, 0.90, 0.95, 0.99])
    output.update(
        {
            "median": float(quantiles[0]),
            "p90": float(quantiles[1]),
            "p95": float(quantiles[2]),
            "p99": float(quantiles[3]),
            "max": int(values.max(initial=0)),
            "nonzero_image_fraction": float(np.count_nonzero(values) / len(values)),
            "hhi": _hhi(values),
            "gini": _gini(values),
        }
    )
    return output


def _area_distribution(areas: Sequence[int]) -> dict[str, Any]:
    values = np.asarray(areas, dtype=np.int64)
    if values.size == 0:
        return {
            "count": 0,
            "total_area": 0,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "single_pixel_count": 0,
            "single_pixel_fraction": 0.0,
        }
    quantiles = np.quantile(values.astype(np.float64), [0.5, 0.90, 0.95, 0.99])
    single = int(np.count_nonzero(values == 1))
    return {
        "count": int(values.size),
        "total_area": int(values.sum(dtype=np.int64)),
        "median": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "max": int(values.max()),
        "single_pixel_count": single,
        "single_pixel_fraction": float(single / values.size),
    }


def _hhi(values: np.ndarray) -> float:
    total = int(values.sum(dtype=np.int64))
    if total == 0:
        return 0.0
    shares = values.astype(np.float64) / total
    return float(np.dot(shares, shares))


def _gini(values: np.ndarray) -> float:
    total = int(values.sum(dtype=np.int64))
    if total == 0:
        return 0.0
    ordered = np.sort(values.astype(np.float64))
    n = ordered.size
    indices = np.arange(1, n + 1, dtype=np.float64)
    return float(np.dot(2.0 * indices - n - 1.0, ordered) / (n * total))


def _pd_fraction(row: Mapping[str, Any]) -> Fraction:
    gt = _count(row, "gt_objects")
    return Fraction(_count(row, "tp_objects"), gt) if gt else Fraction(0, 1)


def _count(row: Mapping[str, Any], field: str) -> int:
    if not isinstance(row, Mapping) or field not in row:
        raise ValueError(f"missing raw count field: {field}")
    value = row[field]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_fraction(value: Any, *, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating, str)
    ):
        raise TypeError(f"{name} must be numeric")
    try:
        number = float(value)
        fraction = Fraction(str(value))
    except (ValueError, TypeError, ZeroDivisionError) as error:
        raise ValueError(f"{name} is not numeric") from error
    if not math.isfinite(number) or fraction <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return fraction


def _first_present(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
    *,
    name: str,
) -> Any:
    present = [key for key in keys if key in mapping]
    if not present:
        raise ValueError(f"missing {name}")
    if len(present) > 1:
        values = [mapping[key] for key in present]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"conflicting aliases for {name}")
    return mapping[present[0]]


__all__ = [
    "build_realized_fa_sensitivity",
    "deterministic_dense_state_indices",
    "select_dense_operating_points",
    "summarize_false_alarm_concentration",
]
