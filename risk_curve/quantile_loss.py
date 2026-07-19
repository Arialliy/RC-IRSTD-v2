"""Losses used to fit conservative risk quantiles."""

from __future__ import annotations

import torch


def pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantile: float = 0.9,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute quantile (pinball) loss with ``target - prediction`` errors."""

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between 0 and 1")
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {prediction.shape} vs {target.shape}"
        )
    error = target - prediction
    loss = torch.maximum(quantile * error, (quantile - 1.0) * error)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")
