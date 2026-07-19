from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from rc_irstd.evaluation.metrics import (
    REJECT_ALL_LATENT_LOGIT,
    REJECT_ALL_THRESHOLD,
)
from risk_curve.representation import (
    LOGIT_REPRESENTATION,
    PROBABILITY_REPRESENTATION,
    validate_logit_threshold_grid,
)


RC_DIRECT_ARCHITECTURE_VERSION = "rc-direct-ordered-budget-v4-raw-logit-v1"


@dataclass
class CalibratorOutput:
    grid_logits: torch.Tensor
    grid_thresholds: torch.Tensor
    requested_logits: torch.Tensor | None = None
    requested_thresholds: torch.Tensor | None = None
    representation: str = PROBABILITY_REPRESENTATION


class FeatureNormalizer(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("std", torch.ones(dimension))
        self.register_buffer("is_fitted", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def fit(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.mean.numel():
            raise ValueError("Feature matrix has an incompatible shape")
        mean = features.mean(dim=0)
        std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
        self.mean.copy_(mean)
        self.std.copy_(std)
        self.is_fitted.fill_(True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not bool(self.is_fitted.item()):
            raise RuntimeError("FeatureNormalizer must be fitted before inference")
        return (features - self.mean) / self.std


class MonotoneBudgetCalibrator(nn.Module):
    """Maps unlabeled support statistics to a monotone inverse risk curve.

    `budget_grid` must be strictly descending from loose to strict, e.g.
    ``[1e-4, 1e-5, 1e-6]``. Positive spacings make the predicted threshold
    logits non-decreasing as the budget becomes stricter.
    """

    def __init__(
        self,
        feature_dim: int,
        budget_grid: Sequence[float],
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.1,
        min_logit: float | None = None,
        max_logit: float | None = None,
        representation: str = PROBABILITY_REPRESENTATION,
        threshold_grid: Sequence[float] | None = None,
        architecture_version: str = RC_DIRECT_ARCHITECTURE_VERSION,
    ) -> None:
        super().__init__()
        budgets = torch.as_tensor(list(budget_grid), dtype=torch.float32)
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if budgets.ndim != 1 or budgets.numel() < 2:
            raise ValueError("budget_grid must have at least two values")
        if (
            not torch.isfinite(budgets).all()
            or torch.any(budgets <= 0)
            or not torch.all(budgets[:-1] > budgets[1:])
        ):
            raise ValueError("budget_grid must be positive and strictly descending")
        if representation not in {
            PROBABILITY_REPRESENTATION,
            LOGIT_REPRESENTATION,
        }:
            raise ValueError(f"Unsupported calibrator representation {representation!r}")
        if architecture_version != RC_DIRECT_ARCHITECTURE_VERSION:
            raise ValueError(
                f"Unsupported RC-Direct architecture {architecture_version!r}"
            )
        raw_grid: tuple[float, ...] | None = None
        reject_code: float | None = None
        if representation == LOGIT_REPRESENTATION:
            if threshold_grid is None:
                raise ValueError("Raw-logit RC-Direct requires the finite threshold_grid")
            validated_grid = validate_logit_threshold_grid(
                torch.as_tensor(list(threshold_grid), dtype=torch.float32)
                .cpu()
                .numpy()
            )
            raw_grid = tuple(float(value) for value in validated_grid.tolist())
            final_spacing = max(
                float(validated_grid[-1] - validated_grid[-2]),
                float(torch.finfo(torch.float32).eps)
                * max(abs(float(validated_grid[-1])), 1.0)
                * 16.0,
                1e-4,
            )
            reject_code = float(validated_grid[-1]) + final_spacing
            if min_logit is None:
                min_logit = float(validated_grid[0]) - final_spacing
            if max_logit is None:
                max_logit = reject_code + final_spacing
        else:
            if threshold_grid is not None:
                raise ValueError(
                    "Probability RC-Direct must not attach a raw-logit threshold_grid"
                )
            min_logit = -10.0 if min_logit is None else min_logit
            max_logit = 18.0 if max_logit is None else max_logit
        assert min_logit is not None and max_logit is not None
        if not math.isfinite(float(min_logit)) or not math.isfinite(float(max_logit)):
            raise ValueError("min_logit and max_logit must be finite")
        if min_logit >= max_logit:
            raise ValueError("min_logit must be lower than max_logit")
        if (
            representation == PROBABILITY_REPRESENTATION
            and max_logit <= REJECT_ALL_LATENT_LOGIT
        ):
            raise ValueError(
                f"max_logit must exceed {REJECT_ALL_LATENT_LOGIT:g} so the "
                "model can represent the reject-all threshold"
            )
        if reject_code is not None and float(max_logit) <= reject_code:
            raise ValueError(
                "Raw-logit max_logit must exceed the finite internal reject code"
            )

        self.feature_dim = int(feature_dim)
        self.register_buffer("budget_grid", budgets)
        self.register_buffer("log_budget_grid", torch.log10(budgets))
        self.min_logit = float(min_logit)
        self.max_logit = float(max_logit)
        self.representation = str(representation)
        self.threshold_grid = raw_grid
        self.reject_code = reject_code
        self.architecture_version = str(architecture_version)
        self.normalizer = FeatureNormalizer(self.feature_dim)

        if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be finite and lie in [0,1)")
        widths = tuple(int(width) for width in hidden_dims)
        if any(width <= 0 for width in widths):
            raise ValueError("hidden_dims must contain positive widths")
        layers: list[nn.Module] = []
        previous = self.feature_dim
        for width in widths:
            layers.extend(
                [
                    nn.Linear(previous, int(width)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            previous = int(width)
        self.encoder = nn.Sequential(*layers)
        # J+1 positive intervals place J ordered points inside [min_logit, max_logit].
        self.spacing_head = nn.Linear(previous, budgets.numel() + 1)

    @property
    def num_budgets(self) -> int:
        return int(self.budget_grid.numel())

    def _ordered_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        spacings = F.softplus(self.spacing_head(hidden)) + 1e-4
        cumulative = torch.cumsum(spacings[:, :-1], dim=1)
        positions = cumulative / spacings.sum(dim=1, keepdim=True)
        return self.min_logit + positions * (self.max_logit - self.min_logit)

    @staticmethod
    def logits_to_thresholds(logits: torch.Tensor) -> torch.Tensor:
        sentinel = torch.as_tensor(
            REJECT_ALL_THRESHOLD,
            dtype=logits.dtype,
            device=logits.device,
        )
        thresholds = torch.sigmoid(logits) * sentinel
        return torch.where(
            logits >= REJECT_ALL_LATENT_LOGIT,
            sentinel,
            thresholds,
        )

    def interpolate_logits(
        self,
        grid_logits: torch.Tensor,
        budgets: torch.Tensor,
    ) -> torch.Tensor:
        """Piecewise-linear interpolation in log10-budget space."""
        request = budgets.to(device=grid_logits.device, dtype=grid_logits.dtype)
        if request.ndim == 0:
            request = request[None, None].expand(grid_logits.shape[0], 1)
        elif request.ndim == 1:
            if request.numel() == grid_logits.shape[0]:
                request = request[:, None]
            else:
                request = request[None, :].expand(grid_logits.shape[0], -1)
        elif request.ndim == 2 and request.shape[0] == 1:
            request = request.expand(grid_logits.shape[0], -1)
        if request.ndim != 2 or request.shape[0] != grid_logits.shape[0]:
            raise ValueError("budgets cannot be broadcast to the batch dimension")
        if not torch.isfinite(request).all() or torch.any(request <= 0):
            raise ValueError("Requested budgets must be finite and positive")

        lower_budget = self.budget_grid[-1].to(device=request.device, dtype=request.dtype)
        upper_budget = self.budget_grid[0].to(device=request.device, dtype=request.dtype)
        tolerance = torch.finfo(request.dtype).eps * 16
        if torch.any(request < lower_budget * (1.0 - tolerance)) or torch.any(
            request > upper_budget * (1.0 + tolerance)
        ):
            raise ValueError(
                "Requested budgets must stay inside the trained budget grid "
                f"[{float(lower_budget):.6g}, {float(upper_budget):.6g}]"
            )

        x = torch.flip(self.log_budget_grid.to(grid_logits.dtype), dims=[0])
        y = torch.flip(grid_logits, dims=[1])
        query = torch.log10(request.clamp_min(torch.finfo(request.dtype).tiny))
        # Clamp only numerical roundoff after explicit range validation.
        query = query.clamp(min=x[0], max=x[-1])
        right = torch.searchsorted(x, query, right=True).clamp(1, x.numel() - 1)
        left = right - 1
        x0 = x[left]
        x1 = x[right]
        batch_index = torch.arange(grid_logits.shape[0], device=grid_logits.device)[:, None]
        y0 = y[batch_index, left]
        y1 = y[batch_index, right]
        weight = (query - x0) / (x1 - x0).clamp_min(1e-8)
        return y0 + weight * (y1 - y0)

    def forward(
        self,
        features: torch.Tensor,
        budgets: torch.Tensor | None = None,
    ) -> CalibratorOutput:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected features [B,{self.feature_dim}], received {tuple(features.shape)}"
            )
        normalized = self.normalizer(features)
        hidden = self.encoder(normalized)
        grid_logits = self._ordered_logits(hidden)
        grid_thresholds = (
            grid_logits
            if self.representation == LOGIT_REPRESENTATION
            else self.logits_to_thresholds(grid_logits)
        )
        if budgets is None:
            return CalibratorOutput(
                grid_logits,
                grid_thresholds,
                representation=self.representation,
            )
        requested_logits = self.interpolate_logits(grid_logits, budgets)
        requested_thresholds = (
            requested_logits
            if self.representation == LOGIT_REPRESENTATION
            else self.logits_to_thresholds(requested_logits)
        )
        return CalibratorOutput(
            grid_logits=grid_logits,
            grid_thresholds=grid_thresholds,
            requested_logits=requested_logits,
            requested_thresholds=requested_thresholds,
            representation=self.representation,
        )

    def export_config(self) -> dict[str, object]:
        linear_widths = [
            module.out_features for module in self.encoder if isinstance(module, nn.Linear)
        ]
        dropout = next(
            (module.p for module in self.encoder if isinstance(module, nn.Dropout)), 0.0
        )
        return {
            "feature_dim": self.feature_dim,
            "budget_grid": self.budget_grid.detach().cpu().tolist(),
            "hidden_dims": linear_widths,
            "dropout": dropout,
            "min_logit": self.min_logit,
            "max_logit": self.max_logit,
            "representation": self.representation,
            "threshold_grid": (
                list(self.threshold_grid) if self.threshold_grid is not None else None
            ),
            "architecture_version": self.architecture_version,
        }
