"""Few-shot grid-rank risk-control utilities for RC-IRSTD-v2.

Exports are loaded lazily so ``python -m certification.<module>`` does not
pre-import the module being executed and trigger a runpy warning.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "COMPONENT_BUDGET_UNIT": (".build_calibration_losses", "COMPONENT_BUDGET_UNIT"),
    "PIXEL_BUDGET_UNIT": (".build_calibration_losses", "PIXEL_BUDGET_UNIT"),
    "LOSS_MODE_BUDGET_VIOLATION": (
        ".build_calibration_losses",
        "LOSS_MODE_BUDGET_VIOLATION",
    ),
    "LOSS_MODE_RISK_RATIO": (".build_calibration_losses", "LOSS_MODE_RISK_RATIO"),
    "CalibrationLosses": (".build_calibration_losses", "CalibrationLosses"),
    "OffsetSelection": (".conformal_offset", "OffsetSelection"),
    "assert_disjoint_image_ids": (
        ".calibrate_target_offset",
        "assert_disjoint_image_ids",
    ),
    "assert_three_way_disjoint_image_ids": (
        ".calibrate_target_offset",
        "assert_three_way_disjoint_image_ids",
    ),
    "build_calibration_losses": (
        ".build_calibration_losses",
        "build_calibration_losses",
    ),
    "build_count_curves_from_score_maps": (
        ".build_calibration_losses",
        "build_count_curves_from_score_maps",
    ),
    "build_grid_rank_candidates": (
        ".conformal_offset",
        "build_grid_rank_candidates",
    ),
    "build_adaptive_grid_rank_candidates": (
        ".conformal_offset",
        "build_adaptive_grid_rank_candidates",
    ),
    "calibrate_target_offset": (
        ".calibrate_target_offset",
        "calibrate_target_offset",
    ),
    "conservative_suffix_max": (
        ".build_calibration_losses",
        "conservative_suffix_max",
    ),
    "evaluate_selected_operating_point": (
        ".evaluate_certified_mode",
        "evaluate_selected_operating_point",
    ),
    "finite_sample_feasibility": (
        ".conformal_offset",
        "finite_sample_feasibility",
    ),
    "select_conformal_offset": (".conformal_offset", "select_conformal_offset"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "COMPONENT_BUDGET_UNIT",
    "PIXEL_BUDGET_UNIT",
    "LOSS_MODE_BUDGET_VIOLATION",
    "LOSS_MODE_RISK_RATIO",
    "CalibrationLosses",
    "OffsetSelection",
    "assert_disjoint_image_ids",
    "assert_three_way_disjoint_image_ids",
    "build_calibration_losses",
    "build_count_curves_from_score_maps",
    "build_grid_rank_candidates",
    "build_adaptive_grid_rank_candidates",
    "calibrate_target_offset",
    "conservative_suffix_max",
    "evaluate_selected_operating_point",
    "finite_sample_feasibility",
    "select_conformal_offset",
]
