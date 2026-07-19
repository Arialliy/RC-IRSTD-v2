"""Smooth worst-domain aggregation."""

from __future__ import annotations

import torch


def smooth_max(
    values: torch.Tensor,
    *,
    gamma: float = 10.0,
    dim: int = 0,
    keepdim: bool = False,
    normalize: bool = False,
) -> torch.Tensor:
    """Log-sum-exp approximation of a maximum.

    normalize=False implements the RC-IRSTD objective.  normalize=True
    optionally subtracts the constant log(n) / gamma.
    """

    if not isinstance(values, torch.Tensor):
        raise TypeError("values must be a torch tensor")
    if values.numel() == 0:
        raise ValueError("values cannot be empty")
    gamma = float(gamma)
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if values.ndim == 0:
        return values

    result = torch.logsumexp(gamma * values, dim=dim, keepdim=keepdim) / gamma
    if normalize:
        count = values.shape[dim]
        correction = torch.log(values.new_tensor(float(count))) / gamma
        result = result - correction
    return result


def smooth_worst_domain(
    domain_risks: torch.Tensor,
    *,
    gamma: float = 10.0,
) -> torch.Tensor:
    return smooth_max(domain_risks, gamma=gamma, dim=0)
