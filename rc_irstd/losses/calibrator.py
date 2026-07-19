from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from rc_irstd.evaluation.metrics import (
    REJECT_ALL_LATENT_LOGIT,
    REJECT_ALL_THRESHOLD,
)


@dataclass
class CalibratorLossOutput:
    total: torch.Tensor
    oracle: torch.Tensor
    violation: torch.Tensor
    utility: torch.Tensor
    smoothness: torch.Tensor
    soft_fa: torch.Tensor
    soft_pd: torch.Tensor


def asymmetric_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    under_weight: float = 4.0,
    beta: float = 0.5,
) -> torch.Tensor:
    per_item = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    weights = torch.where(
        prediction < target,
        torch.full_like(prediction, float(under_weight)),
        torch.ones_like(prediction),
    )
    return (per_item * weights).mean()


def soft_operating_metrics(
    predicted_logits: torch.Tensor,
    background_histogram: torch.Tensor,
    object_histogram: torch.Tensor,
    bin_centers: torch.Tensor,
    total_pixels: torch.Tensor | None = None,
    *,
    temperature: float = 0.20,
) -> tuple[torch.Tensor, torch.Tensor]:
    centers = bin_centers.to(
        device=predicted_logits.device, dtype=predicted_logits.dtype
    )[None, None, :]
    sentinel = torch.as_tensor(
        REJECT_ALL_THRESHOLD,
        device=predicted_logits.device,
        dtype=predicted_logits.dtype,
    )
    threshold_probability = torch.sigmoid(predicted_logits) * sentinel
    probability_eps = torch.finfo(predicted_logits.dtype).eps
    clipped = threshold_probability.clamp(
        min=probability_eps,
        max=1.0 - probability_eps,
    )
    threshold_score_logits = torch.log(clipped) - torch.log1p(-clipped)
    # A latent at/above the reserved code means no probability can pass.
    threshold_score_logits = torch.where(
        predicted_logits >= REJECT_ALL_LATENT_LOGIT,
        torch.full_like(threshold_score_logits, 32.0),
        threshold_score_logits,
    )
    gate = torch.sigmoid(
        (centers - threshold_score_logits[:, :, None]) / temperature
    )
    background = background_histogram.to(predicted_logits.dtype)
    objects = object_histogram.to(predicted_logits.dtype)
    if total_pixels is None:
        background_total = background.sum(dim=1, keepdim=True).clamp_min(1.0)
    else:
        background_total = total_pixels.to(
            device=predicted_logits.device, dtype=predicted_logits.dtype
        ).reshape(-1, 1).clamp_min(1.0)
    object_total = objects.sum(dim=1, keepdim=True).clamp_min(1.0)
    soft_fa = (background[:, None, :] * gate).sum(dim=2) / background_total
    soft_pd = (objects[:, None, :] * gate).sum(dim=2) / object_total
    has_objects = (objects.sum(dim=1, keepdim=True) > 0).to(predicted_logits.dtype)
    soft_pd = soft_pd * has_objects
    return soft_fa, soft_pd


def calibrator_objective(
    predicted_logits: torch.Tensor,
    target_logits: torch.Tensor,
    budgets: torch.Tensor,
    background_histogram: torch.Tensor,
    object_histogram: torch.Tensor,
    bin_centers: torch.Tensor,
    total_pixels: torch.Tensor | None = None,
    *,
    lambda_oracle: float = 1.0,
    lambda_violation: float = 1.0,
    lambda_utility: float = 0.05,
    lambda_smoothness: float = 0.01,
    under_weight: float = 4.0,
    soft_temperature: float = 0.20,
) -> CalibratorLossOutput:
    oracle = asymmetric_smooth_l1(
        predicted_logits,
        target_logits,
        under_weight=under_weight,
    )
    soft_fa, soft_pd = soft_operating_metrics(
        predicted_logits,
        background_histogram,
        object_histogram,
        bin_centers,
        total_pixels=total_pixels,
        temperature=soft_temperature,
    )
    budget_matrix = budgets.to(
        device=predicted_logits.device, dtype=predicted_logits.dtype
    )[None, :]
    violation = F.relu(
        torch.log((soft_fa + 1e-12) / (budget_matrix + 1e-12))
    ).mean()
    has_objects = object_histogram.sum(dim=1) > 0
    if torch.any(has_objects):
        utility = (1.0 - soft_pd[has_objects]).mean()
    else:
        utility = predicted_logits.sum() * 0.0
    if predicted_logits.shape[1] >= 3:
        second_difference = (
            predicted_logits[:, 2:]
            - 2.0 * predicted_logits[:, 1:-1]
            + predicted_logits[:, :-2]
        )
        smoothness = second_difference.square().mean()
    else:
        smoothness = predicted_logits.sum() * 0.0
    total = (
        lambda_oracle * oracle
        + lambda_violation * violation
        + lambda_utility * utility
        + lambda_smoothness * smoothness
    )
    return CalibratorLossOutput(
        total=total,
        oracle=oracle,
        violation=violation,
        utility=utility,
        smoothness=smoothness,
        soft_fa=soft_fa,
        soft_pd=soft_pd,
    )
