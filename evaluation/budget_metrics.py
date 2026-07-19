"""Budget-satisfaction and excess-risk summaries without pandas."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _validate_budget(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return number


def _risk_value(row: Mapping[str, object], key: str) -> float:
    if key not in row:
        raise KeyError(f"result row is missing required field {key!r}")
    value = float(row[key])
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{key} must be finite and non-negative, got {value}")
    return value


def is_budget_satisfied(
    row: Mapping[str, object],
    *,
    pixel_budget: float | None = None,
    component_budget: float | None = None,
) -> bool:
    """Return whether all supplied budgets are satisfied by one result row."""

    pixel_budget = _validate_budget(pixel_budget, "pixel_budget")
    component_budget = _validate_budget(component_budget, "component_budget")
    if pixel_budget is None and component_budget is None:
        raise ValueError("at least one budget must be supplied")
    if pixel_budget is not None and _risk_value(row, "fa_pixel") > pixel_budget:
        return False
    if (
        component_budget is not None
        and _risk_value(row, "fa_component_mp") > component_budget
    ):
        return False
    return True


def compute_budget_excess(
    row: Mapping[str, object],
    *,
    pixel_budget: float | None = None,
    component_budget: float | None = None,
) -> dict[str, float]:
    """Compute absolute and normalized positive excess for one row."""

    pixel_budget = _validate_budget(pixel_budget, "pixel_budget")
    component_budget = _validate_budget(component_budget, "component_budget")
    if pixel_budget is None and component_budget is None:
        raise ValueError("at least one budget must be supplied")
    pixel_excess = 0.0
    pixel_relative = 0.0
    component_excess = 0.0
    component_relative = 0.0
    if pixel_budget is not None:
        pixel_risk = _risk_value(row, "fa_pixel")
        pixel_excess = max(pixel_risk - pixel_budget, 0.0)
        pixel_relative = pixel_excess / pixel_budget
    if component_budget is not None:
        component_risk = _risk_value(row, "fa_component_mp")
        component_excess = max(component_risk - component_budget, 0.0)
        component_relative = component_excess / component_budget
    return {
        "pixel_excess": float(pixel_excess),
        "pixel_relative_excess": float(pixel_relative),
        "component_excess": float(component_excess),
        "component_relative_excess": float(component_relative),
        "max_relative_excess": float(max(pixel_relative, component_relative)),
    }


def summarize_budget_results(
    rows: Sequence[Mapping[str, object]],
    *,
    pixel_budget: float | None = None,
    component_budget: float | None = None,
) -> dict[str, float | int | None]:
    """Summarize BSR and excess over windows, domains, or repeated trials."""

    if not rows:
        raise ValueError("rows must be non-empty")
    pixel_budget = _validate_budget(pixel_budget, "pixel_budget")
    component_budget = _validate_budget(component_budget, "component_budget")
    if pixel_budget is None and component_budget is None:
        raise ValueError("at least one budget must be supplied")
    satisfied = [
        is_budget_satisfied(
            row,
            pixel_budget=pixel_budget,
            component_budget=component_budget,
        )
        for row in rows
    ]
    excesses = [
        compute_budget_excess(
            row,
            pixel_budget=pixel_budget,
            component_budget=component_budget,
        )
        for row in rows
    ]

    def aggregate(key: str, reducer) -> float:
        return float(reducer([entry[key] for entry in excesses]))

    return {
        "num_results": len(rows),
        "num_satisfied": int(sum(satisfied)),
        "bsr": float(np.mean(satisfied)),
        "pixel_budget": pixel_budget,
        "component_budget": component_budget,
        "mean_pixel_excess": aggregate("pixel_excess", np.mean),
        "max_pixel_excess": aggregate("pixel_excess", np.max),
        "mean_component_excess": aggregate("component_excess", np.mean),
        "max_component_excess": aggregate("component_excess", np.max),
        "mean_relative_excess": aggregate("max_relative_excess", np.mean),
        "max_relative_excess": aggregate("max_relative_excess", np.max),
    }


def summarize_budget_by_group(
    rows: Sequence[Mapping[str, object]],
    group_key: str,
    *,
    pixel_budget: float | None = None,
    component_budget: float | None = None,
) -> dict[str, dict[str, float | int | None]]:
    """Compute the same transparent summary independently for each group."""

    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        if group_key not in row:
            raise KeyError(f"result row is missing group field {group_key!r}")
        grouped.setdefault(str(row[group_key]), []).append(row)
    if not grouped:
        raise ValueError("rows must be non-empty")
    return {
        group: summarize_budget_results(
            group_rows,
            pixel_budget=pixel_budget,
            component_budget=component_budget,
        )
        for group, group_rows in sorted(grouped.items())
    }


# Short, discoverable alias used by downstream experiment scripts.
compute_budget_metrics = summarize_budget_results


def _read_rows(path: str | Path) -> list[dict[str, object]]:
    input_path = Path(path).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input results file does not exist: {input_path}")
    if input_path.suffix.lower() == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("JSON input must be a row list or an object containing 'rows'")
        return rows
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pixel-budget", type=float)
    parser.add_argument("--component-budget", type=float)
    parser.add_argument("--group-key")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    rows = _read_rows(args.input)
    if args.group_key:
        summary: object = summarize_budget_by_group(
            rows,
            args.group_key,
            pixel_budget=args.pixel_budget,
            component_budget=args.component_budget,
        )
    else:
        summary = summarize_budget_results(
            rows,
            pixel_budget=args.pixel_budget,
            component_budget=args.component_budget,
        )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "compute_budget_excess",
    "compute_budget_metrics",
    "is_budget_satisfied",
    "summarize_budget_by_group",
    "summarize_budget_results",
]

