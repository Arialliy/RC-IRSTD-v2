"""Shared low-false-alarm evaluation core for RC-IRSTD-v2.

The public helpers are loaded lazily so ``python -m evaluation.<tool>`` does not
pre-import the tool module and emit a runpy warning.
"""

from __future__ import annotations

from importlib import import_module

from .component_matching import MatchResult, connected_components, match_components

__all__ = [
    "MatchResult",
    "build_default_thresholds",
    "compute_budget_excess",
    "compute_budget_metrics",
    "connected_components",
    "is_budget_satisfied",
    "match_components",
    "pixel_fa_is_monotone",
    "compute_standard_metrics",
    "evaluate_standard_score_directory",
    "evaluate_selected_actions",
    "select_operating_point",
    "summarize_budget_results",
    "sweep_thresholds",
]


_LAZY_EXPORTS = {
    "build_default_thresholds": (".threshold_sweep", "build_default_thresholds"),
    "pixel_fa_is_monotone": (".threshold_sweep", "pixel_fa_is_monotone"),
    "sweep_thresholds": (".threshold_sweep", "sweep_thresholds"),
    "compute_standard_metrics": (".standard_metrics", "compute_standard_metrics"),
    "evaluate_standard_score_directory": (
        ".standard_metrics",
        "evaluate_standard_score_directory",
    ),
    "evaluate_selected_actions": (
        ".evaluate_selected_actions",
        "evaluate_selected_actions",
    ),
    "compute_budget_excess": (".budget_metrics", "compute_budget_excess"),
    "compute_budget_metrics": (".budget_metrics", "compute_budget_metrics"),
    "is_budget_satisfied": (".budget_metrics", "is_budget_satisfied"),
    "summarize_budget_results": (".budget_metrics", "summarize_budget_results"),
    "select_operating_point": (".operating_point", "select_operating_point"),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
