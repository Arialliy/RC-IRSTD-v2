"""Connected-component hard-target Miss-CVaR loss."""

from __future__ import annotations

import torch
from skimage import measure

from .local_peak_cvar import top_fraction_mean


def _validate_fraction(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return value


def target_object_scores(
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    response_fraction: float = 0.25,
    connectivity: int = 2,
) -> torch.Tensor:
    """Aggregate a differentiable score for every GT connected component.

    Connected components are identified on detached binary masks.  Component
    membership is then moved back to the logits device, so probability
    aggregation and all subsequent risk operations retain gradients.
    """

    if not isinstance(logits, torch.Tensor) or not isinstance(masks, torch.Tensor):
        raise TypeError("logits and masks must be torch tensors")
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"logits must have shape [N, 1, H, W], got {tuple(logits.shape)}")
    if masks.shape != logits.shape:
        raise ValueError(
            f"masks must match logits shape, got {tuple(masks.shape)} and "
            f"{tuple(logits.shape)}"
        )
    response_fraction = _validate_fraction(response_fraction, "response_fraction")
    if connectivity not in (1, 2):
        raise ValueError("connectivity must be 1 or 2")

    probability = torch.sigmoid(logits)
    object_scores: list[torch.Tensor] = []
    for batch_index in range(logits.shape[0]):
        binary_mask = (
            masks[batch_index, 0].detach().to(device="cpu").numpy() > 0.5
        )
        labels = measure.label(binary_mask, connectivity=connectivity)
        if labels.max() == 0:
            continue
        labels_tensor = torch.from_numpy(labels).to(device=logits.device)
        for component_id in range(1, int(labels.max()) + 1):
            component_values = probability[batch_index, 0].masked_select(
                labels_tensor == component_id
            )
            object_scores.append(
                top_fraction_mean(component_values, response_fraction)
            )

    if not object_scores:
        # An empty view preserves a gradient connection to the model output.
        return probability.reshape(-1)[:0]
    return torch.stack(object_scores)


def hard_target_miss_from_scores(
    object_scores: torch.Tensor,
    *,
    miss_fraction: float = 0.2,
) -> torch.Tensor:
    """Apply upper-tail CVaR to per-object miss scores: 1 - response."""

    if not isinstance(object_scores, torch.Tensor):
        raise TypeError("object_scores must be a torch tensor")
    miss_fraction = _validate_fraction(miss_fraction, "miss_fraction")
    object_scores = object_scores.reshape(-1)
    if object_scores.numel() == 0:
        return object_scores.sum() * 0.0
    return top_fraction_mean(1.0 - object_scores, miss_fraction)


def hard_target_miss_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    miss_fraction: float = 0.2,
    response_fraction: float = 0.25,
    connectivity: int = 2,
) -> torch.Tensor:
    scores = target_object_scores(
        logits,
        masks,
        response_fraction=response_fraction,
        connectivity=connectivity,
    )
    return hard_target_miss_from_scores(scores, miss_fraction=miss_fraction)
