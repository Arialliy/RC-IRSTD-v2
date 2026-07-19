"""Structurally monotone dual-risk curve predictor."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


PIXEL_LOG_RISK_FLOOR = -12.0
COMPONENT_LOG_RISK_FLOOR = -6.0
INITIAL_TOTAL_DROP_FRACTION = 0.95
RISK_CURVE_ARCHITECTURE_VERSION = "controlled-total-drop-v2-budget-ready-init"


class MonotoneRiskHead(nn.Module):
    """Predict a monotone curve using one bounded total decrease.

    The historical head applied ``softplus`` independently at every threshold
    and accumulated all resulting decrements.  Its total decrease therefore
    grew with grid length and eventually relied on a hard floor clamp.  This
    parameterisation instead predicts a positive headroom above ``risk_floor``,
    a total decrease bounded by that headroom, and a softmax allocation of the
    decrease across threshold intervals.  The lower bound is structural; no
    post-hoc curve clamp is required.

    ``decrements`` intentionally keeps its historical attribute name for
    source-level compatibility.  Its outputs are allocation logits, not
    independent positive decrements, so historical state dictionaries remain
    (correctly) shape/key incompatible because of the new ``total_drop`` layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_thresholds: int,
        decrement_bias: float = -6.0,
        risk_floor: float = PIXEL_LOG_RISK_FLOOR,
        initial_total_drop_fraction: float = INITIAL_TOTAL_DROP_FRACTION,
    ) -> None:
        super().__init__()
        if num_thresholds < 2:
            raise ValueError("num_thresholds must be at least 2")
        if not torch.isfinite(torch.tensor(float(risk_floor))):
            raise ValueError("risk_floor must be finite")
        if not 0.0 < float(initial_total_drop_fraction) < 1.0:
            raise ValueError("initial_total_drop_fraction must lie in (0, 1)")
        self.num_thresholds = int(num_thresholds)
        self.risk_floor = float(risk_floor)
        self.initial_total_drop_fraction = float(initial_total_drop_fraction)
        self.start = nn.Linear(hidden_dim, 1)
        self.decrements = nn.Linear(hidden_dim, num_thresholds - 1)
        self.total_drop = nn.Linear(hidden_dim, 1)
        # Empirical log-risk at the permissive end is commonly near zero.
        # Initialising ``softplus(start)`` near ``-risk_floor`` avoids starting
        # pixel/component predictions around -11/-5, which would otherwise
        # waste a short smoke run merely translating the whole curve upward.
        initial_headroom = max(-self.risk_floor, 1e-3)
        inverse_softplus = torch.log(torch.expm1(torch.tensor(initial_headroom)))
        nn.init.constant_(self.start.bias, float(inverse_softplus))
        nn.init.constant_(self.decrements.bias, decrement_bias)
        # ``drop_fraction = softplus(z) / (1 + softplus(z))`` below.  Solving
        # this map for the registered initial fraction gives a raw magnitude
        # of f / (1 - f), followed by the inverse softplus.  Initialising near
        # the physical floor makes the finite-grid endpoint budget-feasible
        # before a short smoke run, while still leaving five percent headroom
        # and therefore a non-degenerate, trainable curve.  A zero weight makes
        # the registered initial drop fraction independent of the randomly
        # initialised encoder before the first optimisation step.
        initial_raw_drop = self.initial_total_drop_fraction / (
            1.0 - self.initial_total_drop_fraction
        )
        inverse_drop_softplus = math.log(math.expm1(initial_raw_drop))
        nn.init.zeros_(self.total_drop.weight)
        nn.init.constant_(self.total_drop.bias, inverse_drop_softplus)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        headroom = F.softplus(self.start(hidden))
        start = headroom + self.risk_floor

        # Map a positive raw magnitude into [0, 1) and retain at least one dtype
        # epsilon of headroom.  The latter prevents a rounded fraction of exactly
        # one from placing the endpoint below the physical floor in float32.
        raw_total_drop = F.softplus(self.total_drop(hidden))
        drop_fraction = raw_total_drop / (1.0 + raw_total_drop)
        drop_fraction = drop_fraction.clamp_max(
            1.0 - torch.finfo(drop_fraction.dtype).eps
        )
        total_drop = headroom * drop_fraction

        allocation = F.softmax(self.decrements(hidden), dim=1)
        cumulative_allocation = torch.cumsum(allocation, dim=1)
        # Normalising by the accumulated final mass makes the endpoint exactly
        # use ``total_drop`` even if a long float32 softmax sums a few ulps away
        # from one.
        cumulative_allocation = cumulative_allocation / cumulative_allocation[:, -1:]
        tail = start - total_drop * cumulative_allocation
        return torch.cat([start, tail], dim=1)


class RiskCurvePredictor(nn.Module):
    """Predict pixel and component log-risk curves from unlabeled statistics."""

    def __init__(
        self,
        input_dim: int,
        num_thresholds: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        pixel_log_risk_floor: float = PIXEL_LOG_RISK_FLOOR,
        component_log_risk_floor: float = COMPONENT_LOG_RISK_FLOOR,
        architecture_version: str = RISK_CURVE_ARCHITECTURE_VERSION,
        initial_total_drop_fraction: float = INITIAL_TOTAL_DROP_FRACTION,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if architecture_version != RISK_CURVE_ARCHITECTURE_VERSION:
            raise ValueError(
                "Unsupported RiskCurvePredictor architecture_version "
                f"{architecture_version!r}; expected "
                f"{RISK_CURVE_ARCHITECTURE_VERSION!r}"
            )
        if float(initial_total_drop_fraction) != INITIAL_TOTAL_DROP_FRACTION:
            raise ValueError(
                "This RiskCurvePredictor architecture requires "
                f"initial_total_drop_fraction={INITIAL_TOTAL_DROP_FRACTION}"
            )
        self.input_dim = int(input_dim)
        self.num_thresholds = int(num_thresholds)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.pixel_log_risk_floor = float(pixel_log_risk_floor)
        self.component_log_risk_floor = float(component_log_risk_floor)
        self.architecture_version = architecture_version
        self.initial_total_drop_fraction = float(initial_total_drop_fraction)
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.pixel_head = MonotoneRiskHead(
            hidden_dim,
            num_thresholds,
            risk_floor=self.pixel_log_risk_floor,
            initial_total_drop_fraction=self.initial_total_drop_fraction,
        )
        self.component_head = MonotoneRiskHead(
            hidden_dim,
            num_thresholds,
            risk_floor=self.component_log_risk_floor,
            initial_total_drop_fraction=self.initial_total_drop_fraction,
        )

    def forward(self, statistics: torch.Tensor) -> dict[str, torch.Tensor]:
        if statistics.ndim != 2 or statistics.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected statistics [N, {self.input_dim}], got {tuple(statistics.shape)}"
            )
        hidden = self.encoder(statistics)
        return {
            "pixel_log_risk": self.pixel_head(hidden),
            "component_log_risk": self.component_head(hidden),
        }

    def config(self) -> dict[str, int | float | str]:
        return {
            "input_dim": self.input_dim,
            "num_thresholds": self.num_thresholds,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "pixel_log_risk_floor": self.pixel_log_risk_floor,
            "component_log_risk_floor": self.component_log_risk_floor,
            "architecture_version": self.architecture_version,
            "initial_total_drop_fraction": self.initial_total_drop_fraction,
        }


__all__ = [
    "COMPONENT_LOG_RISK_FLOOR",
    "INITIAL_TOTAL_DROP_FRACTION",
    "PIXEL_LOG_RISK_FLOOR",
    "RISK_CURVE_ARCHITECTURE_VERSION",
    "MonotoneRiskHead",
    "RiskCurvePredictor",
]
