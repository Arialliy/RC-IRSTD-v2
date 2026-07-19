"""Select feasible operating points from threshold-curve rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .budget_metrics import is_budget_satisfied
from .threshold_sweep import EMPTY_SET_THRESHOLD, read_curve_csv


def _validate_optional_budget(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return value


def _finite_field(row: Mapping[str, object], key: str) -> float:
    if key not in row:
        raise KeyError(f"curve row is missing required field {key!r}")
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"curve row field {key!r} must be finite")
    return value


def select_operating_point(
    rows: Sequence[Mapping[str, object]],
    pixel_budget: float | None = None,
    component_budget: float | None = None,
    *,
    strategy: str = "max_pd",
) -> dict[str, object] | None:
    """Select a feasible row.

    ``max_pd`` is the oracle diagnostic protocol from the design document and
    breaks ties toward the lower threshold.  ``lowest_threshold`` implements the
    first-feasible rule used by monotone deployed risk curves.
    """

    if not rows:
        raise ValueError("rows must be non-empty")
    pixel_budget = _validate_optional_budget(pixel_budget, "pixel_budget")
    component_budget = _validate_optional_budget(component_budget, "component_budget")
    if strategy not in {"max_pd", "lowest_threshold"}:
        raise ValueError("strategy must be 'max_pd' or 'lowest_threshold'")

    feasible: list[Mapping[str, object]] = []
    for row in rows:
        threshold = _finite_field(row, "threshold")
        pd = _finite_field(row, "pd")
        if not (
            0.0 <= threshold <= 1.0
            or threshold == EMPTY_SET_THRESHOLD
        ):
            raise ValueError(
                "threshold values must lie in [0, 1] or equal the "
                "evaluation-only empty-set sentinel nextafter(1,+inf)"
            )
        if not 0.0 <= pd <= 1.0:
            raise ValueError("pd values must lie in [0, 1]")
        if pixel_budget is None and component_budget is None:
            within_budget = True
        else:
            within_budget = is_budget_satisfied(
                row,
                pixel_budget=pixel_budget,
                component_budget=component_budget,
            )
        if within_budget:
            feasible.append(row)
    if not feasible:
        return None
    if strategy == "max_pd":
        selected = min(
            feasible,
            key=lambda row: (-float(row["pd"]), float(row["threshold"])),
        )
    else:
        selected = min(
            feasible,
            key=lambda row: (float(row["threshold"]), -float(row["pd"])),
        )
    return dict(selected)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pixel-budget", type=float)
    parser.add_argument("--component-budget", type=float)
    parser.add_argument(
        "--strategy", choices=("max_pd", "lowest_threshold"), default="max_pd"
    )
    parser.add_argument(
        "--oracle-diagnostic",
        action="store_true",
        help=(
            "Required acknowledgement: this command selects from a realised "
            "ground-truth curve and cannot produce a deployment/formal threshold"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if not args.oracle_diagnostic:
        raise ValueError(
            "operating_point reads realised label-derived curves; pass "
            "--oracle-diagnostic to acknowledge diagnostic-only use"
        )
    rows = read_curve_csv(args.curve)
    selected = select_operating_point(
        rows,
        pixel_budget=args.pixel_budget,
        component_budget=args.component_budget,
        strategy=args.strategy,
    )
    payload = {
        "found": selected is not None,
        "pixel_budget": args.pixel_budget,
        "component_budget": args.component_budget,
        "strategy": args.strategy,
        "operating_point": selected,
        "test_labels_used": True,
        "oracle_only": True,
        "formal_protocol_eligible": False,
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["select_operating_point"]
