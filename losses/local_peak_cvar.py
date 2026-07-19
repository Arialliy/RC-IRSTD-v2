"""Differentiable local-background-peak Tail-CVaR objectives."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as functional
from skimage import measure


def _validate_fraction(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return value


def _validate_prediction_pair(
    logits: torch.Tensor,
    masks: torch.Tensor,
) -> None:
    if not isinstance(logits, torch.Tensor) or not isinstance(masks, torch.Tensor):
        raise TypeError("logits and masks must be torch tensors")
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"logits must have shape [N, 1, H, W], got {tuple(logits.shape)}")
    if masks.shape != logits.shape:
        raise ValueError(
            f"masks must match logits shape, got {tuple(masks.shape)} and "
            f"{tuple(logits.shape)}"
        )
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")


def top_fraction_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    """Mean of the largest ceil(fraction * n) values.

    The empty-input result is a differentiable zero tied to values.  This
    matters when a clean image has no background peak above min_score.
    """

    if not isinstance(values, torch.Tensor):
        raise TypeError("values must be a torch tensor")
    fraction = _validate_fraction(fraction, "fraction")
    values = values.reshape(-1)
    if values.numel() == 0:
        return values.sum() * 0.0
    k = max(1, int(math.ceil(fraction * values.numel())))
    return torch.topk(values, k=k, largest=True, sorted=False).values.mean()


def local_background_peak_scores(
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    kernel_size: int = 3,
    min_score: float = 0.05,
    tolerance: float = 1e-7,
) -> list[torch.Tensor]:
    """Return one differentiable score per 8-connected peak plateau.

    Plateau membership is discrete and therefore identified on a detached CPU
    mask.  The representative maximum is selected from the original
    probability tensor, so every returned score remains connected to logits.
    This matches the one-value-per-plateau contract used by domain statistics.
    """

    _validate_prediction_pair(logits, masks)
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
        raise TypeError("kernel_size must be an integer")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if not 0.0 <= float(min_score) <= 1.0:
        raise ValueError("min_score must be in [0, 1]")
    if float(tolerance) < 0.0:
        raise ValueError("tolerance must be non-negative")

    probability = torch.sigmoid(logits)
    background_mask = masks < 0.5
    background = probability * background_mask.to(probability.dtype)
    pooled = functional.max_pool2d(
        background,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    peak_mask = (
        background_mask
        & (background >= pooled - float(tolerance))
        & (background >= float(min_score))
    )
    per_image: list[torch.Tensor] = []
    for index in range(background.shape[0]):
        plateau_labels = measure.label(
            peak_mask[index, 0].detach().to(device="cpu").numpy(),
            connectivity=2,
        )
        plateau_count = int(plateau_labels.max())
        if plateau_count == 0:
            per_image.append(background[index].reshape(-1)[:0])
            continue

        # Select one representative maximum for every plateau on the detached
        # CPU map, then gather all differentiable values from the original GPU
        # tensor in one operation.  The previous implementation launched one
        # ``masked_select`` kernel per plateau (often thousands per image),
        # making the auxiliary loss substantially slower than the detector.
        flat_labels = plateau_labels.reshape(-1)
        candidate_indices = np.flatnonzero(flat_labels)
        candidate_labels = flat_labels[candidate_indices]
        detached_values = (
            background[index, 0]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .numpy()
            .reshape(-1)[candidate_indices]
        )
        # np.lexsort uses the last key as primary: group by label, prefer the
        # largest value, and use the lowest flat index for deterministic ties.
        order = np.lexsort(
            (candidate_indices, -detached_values, candidate_labels)
        )
        ordered_labels = candidate_labels[order]
        first_in_plateau = np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.flatnonzero(np.diff(ordered_labels)) + 1,
            )
        )
        representative_indices = candidate_indices[order[first_in_plateau]]
        if representative_indices.size != plateau_count:
            raise RuntimeError(
                "connected-component representative extraction is inconsistent"
            )
        gather_indices = torch.as_tensor(
            representative_indices,
            dtype=torch.long,
            device=background.device,
        )
        per_image.append(
            background[index, 0].reshape(-1).index_select(0, gather_indices)
        )
    return per_image


def local_peak_tail_risk(
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    tail_fraction: float = 0.01,
    kernel_size: int = 3,
    min_score: float = 0.05,
) -> torch.Tensor:
    """Compute candidate-level upper-tail risk for one batch/domain."""

    tail_fraction = _validate_fraction(tail_fraction, "tail_fraction")
    if logits.shape[0] == 0:
        raise ValueError("the batch cannot be empty")
    per_image = local_background_peak_scores(
        logits,
        masks,
        kernel_size=kernel_size,
        min_score=min_score,
    )
    values = torch.cat(per_image, dim=0)
    return top_fraction_mean(values, tail_fraction)


def domain_local_peak_tail_risks(
    logits: torch.Tensor,
    masks: torch.Tensor,
    domain_ids: torch.Tensor,
    *,
    tail_fraction: float = 0.01,
    kernel_size: int = 3,
    min_score: float = 0.05,
) -> dict[int, torch.Tensor]:
    """Compute one local-peak Tail-CVaR value for every domain in a batch."""

    _validate_prediction_pair(logits, masks)
    if not isinstance(domain_ids, torch.Tensor):
        raise TypeError("domain_ids must be a torch tensor")
    if domain_ids.ndim != 1 or domain_ids.shape[0] != logits.shape[0]:
        raise ValueError(
            f"domain_ids must have shape [{logits.shape[0]}], got "
            f"{tuple(domain_ids.shape)}"
        )
    if domain_ids.dtype == torch.bool or domain_ids.is_floating_point():
        raise TypeError("domain_ids must use an integer dtype")

    risks: dict[int, torch.Tensor] = {}
    for domain_tensor in torch.unique(domain_ids.detach(), sorted=True):
        domain_id = int(domain_tensor.item())
        selected = domain_ids == domain_tensor
        risks[domain_id] = local_peak_tail_risk(
            logits[selected],
            masks[selected],
            tail_fraction=tail_fraction,
            kernel_size=kernel_size,
            min_score=min_score,
        )
    if not risks:
        raise ValueError("domain_ids cannot be empty")
    return risks


def stack_domain_risks(
    risks: Mapping[int, torch.Tensor],
) -> tuple[list[int], torch.Tensor]:
    """Return sorted ids and a stacked tensor suitable for smooth-max."""

    if not risks:
        raise ValueError("risks cannot be empty")
    domain_ids = sorted(int(domain_id) for domain_id in risks)
    values = [risks[domain_id].reshape(()) for domain_id in domain_ids]
    return domain_ids, torch.stack(values)
