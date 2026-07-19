"""Metrics for predicted false-alarm risk curves and budget selection."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _paired_arrays(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {true.shape}")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("Risk curves must contain only finite values")
    return pred, true


def monotonic_violation_rate(curves: np.ndarray, tolerance: float = 1e-8) -> float:
    values = np.asarray(curves, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("curves must have shape [N, T] with T >= 2")
    return float(np.mean(np.diff(values, axis=1) > tolerance))


def curve_regression_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, float | list[float] | int]:
    pred, true = _paired_arrays(prediction, target)
    if pred.ndim == 1:
        pred = pred[None, :]
        true = true[None, :]
    if pred.ndim != 2 or pred.shape[1] < 2:
        raise ValueError("risk curves must have shape [N, T] with T >= 2")
    gap = pred - true
    coverage_by_threshold = np.mean(true <= pred, axis=0)
    mae_by_threshold = np.mean(np.abs(gap), axis=0)
    worst_index = int(np.argmin(coverage_by_threshold))
    return {
        "log_risk_mae": float(np.mean(np.abs(gap))),
        "upper_bound_coverage": float(np.mean(true <= pred)),
        "upper_bound_coverage_by_threshold": coverage_by_threshold.tolist(),
        "minimum_threshold_coverage": float(coverage_by_threshold[worst_index]),
        "worst_coverage_threshold_index": worst_index,
        "log_risk_mae_by_threshold": mae_by_threshold.tolist(),
        "mean_conservative_gap": float(np.mean(gap)),
        "monotonic_violation_rate": monotonic_violation_rate(pred),
    }


def budget_control_metrics(
    realised_risk: Sequence[float] | np.ndarray,
    budgets: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    risk = np.asarray(realised_risk, dtype=np.float64)
    budget = np.asarray(budgets, dtype=np.float64)
    if risk.shape != budget.shape:
        raise ValueError("realised_risk and budgets must have identical shapes")
    if np.any(budget <= 0.0):
        raise ValueError("budgets must be positive")
    excess = np.maximum(risk - budget, 0.0)
    return {
        "budget_satisfaction_rate": float(np.mean(risk <= budget)),
        "mean_excess": float(np.mean(excess)),
        "max_excess": float(np.max(excess)) if excess.size else 0.0,
    }


def worst_domain_bsr(
    realised_risk: Sequence[float] | np.ndarray,
    budgets: Sequence[float] | np.ndarray,
    domain_ids: Sequence[str] | np.ndarray,
) -> float:
    risk = np.asarray(realised_risk, dtype=np.float64)
    budget = np.asarray(budgets, dtype=np.float64)
    domains = np.asarray(domain_ids)
    if risk.shape != budget.shape or risk.shape != domains.shape:
        raise ValueError("risk, budgets, and domain_ids must have identical shapes")
    if risk.size == 0:
        return 0.0
    scores = [np.mean(risk[domains == key] <= budget[domains == key]) for key in np.unique(domains)]
    return float(np.min(scores))
