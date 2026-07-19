"""Raw-logit TailRank/Margin objective for preregistered detector Candidate A."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from skimage import measure
from torch.nn import functional as F


RAW_LOGIT_TAILRANK_MODE = "raw_logit_tailrank_margin_v1"
RAW_LOGIT_TAILRANK_LAMBDA_TAIL = 0.05
RAW_LOGIT_TAILRANK_LAMBDA_MISS = 0.10
RAW_LOGIT_TAILRANK_LAMBDA_MARGIN = 0.05
RAW_LOGIT_TAILRANK_MARGIN = 1.0
RAW_LOGIT_TAILRANK_CONNECTIVITY = 8


@dataclass(frozen=True)
class RawLogitTailRankComponents:
    background_tail: torch.Tensor
    hard_miss: torch.Tensor
    separation_margin: torch.Tensor
    worst_background_logit: torch.Tensor
    hard_target_logit: torch.Tensor
    num_background_peaks: int
    num_target_objects: int


def _validate_fraction(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 < number <= 1.0:
        raise ValueError(f"{name} must be finite and lie in (0, 1]")
    return number


def _validate_inputs(
    logits: torch.Tensor,
    target: torch.Tensor,
    domain_ids: torch.Tensor,
) -> None:
    if not isinstance(logits, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("logits and target must be torch tensors")
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"logits must have shape [N,1,H,W], got {tuple(logits.shape)}")
    if target.shape != logits.shape:
        raise ValueError("target must have the same shape as logits")
    if not logits.is_floating_point() or not target.is_floating_point():
        raise TypeError("logits and target must be floating point")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("logits and target must be finite")
    if not isinstance(domain_ids, torch.Tensor):
        raise TypeError("domain_ids must be a torch tensor")
    if domain_ids.ndim != 1 or domain_ids.shape[0] != logits.shape[0]:
        raise ValueError(f"domain_ids must have shape [{logits.shape[0]}]")
    if domain_ids.dtype == torch.bool or domain_ids.is_floating_point():
        raise TypeError("domain_ids must use an integer dtype")


def _fraction_mean(
    values: torch.Tensor,
    fraction: float,
    *,
    largest: bool,
) -> torch.Tensor:
    flat = values.reshape(-1)
    fraction = _validate_fraction(fraction, "fraction")
    if flat.numel() == 0:
        return flat.sum() * 0.0
    count = max(1, int(math.ceil(fraction * flat.numel())))
    return torch.topk(flat, count, largest=largest, sorted=False).values.mean()


def raw_logit_background_peaks(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    kernel_size: int = 3,
    exclusion_radius: int = 2,
    min_peak_score: float = 0.05,
    tolerance: float = 1e-6,
) -> list[torch.Tensor]:
    """Gather one raw logit per 8-connected local-maximum plateau.

    ``min_peak_score`` is retained only as the fixed candidate-membership rule
    used by the historical Tail/Miss branch.  It is converted once to a logit;
    all differentiable score arithmetic remains in raw-logit space.
    """

    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
        raise TypeError("kernel_size must be an integer")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if isinstance(exclusion_radius, bool) or int(exclusion_radius) < 0:
        raise ValueError("exclusion_radius must be a non-negative integer")
    if not math.isfinite(float(min_peak_score)) or not 0.0 < float(
        min_peak_score
    ) < 1.0:
        raise ValueError("min_peak_score must lie strictly inside (0, 1)")
    if not math.isfinite(float(tolerance)) or float(tolerance) < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    work_logits = (
        logits
        if logits.dtype in (torch.float32, torch.float64)
        else logits.float()
    )
    binary_target = target >= 0.5
    if int(exclusion_radius) > 0:
        width = 2 * int(exclusion_radius) + 1
        excluded = F.max_pool2d(
            binary_target.to(work_logits.dtype),
            kernel_size=width,
            stride=1,
            padding=int(exclusion_radius),
        ) >= 0.5
    else:
        excluded = binary_target
    background_mask = ~excluded
    negative_infinity = torch.full_like(work_logits, float("-inf"))
    background_logits = torch.where(background_mask, work_logits, negative_infinity)
    pooled = F.max_pool2d(
        background_logits,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    minimum_logit = math.log(float(min_peak_score) / (1.0 - float(min_peak_score)))
    peak_mask = (
        background_mask
        & (background_logits >= pooled - float(tolerance))
        & (background_logits >= minimum_logit)
    )

    per_image: list[torch.Tensor] = []
    for batch_index in range(logits.shape[0]):
        labels = measure.label(
            peak_mask[batch_index, 0].detach().cpu().numpy(),
            connectivity=2,
        )
        plateau_count = int(labels.max())
        if plateau_count == 0:
            per_image.append(work_logits[batch_index].reshape(-1)[:0])
            continue
        flat_labels = labels.reshape(-1)
        candidate_indices = np.flatnonzero(flat_labels)
        candidate_labels = flat_labels[candidate_indices]
        detached_values = (
            work_logits[batch_index, 0]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)[candidate_indices]
        )
        order = np.lexsort((candidate_indices, -detached_values, candidate_labels))
        ordered_labels = candidate_labels[order]
        first = np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.flatnonzero(np.diff(ordered_labels)) + 1,
            )
        )
        representatives = candidate_indices[order[first]]
        if representatives.size != plateau_count:
            raise RuntimeError("Raw-logit peak plateau extraction is inconsistent")
        indices = torch.as_tensor(
            representatives,
            dtype=torch.long,
            device=work_logits.device,
        )
        per_image.append(
            work_logits[batch_index, 0].reshape(-1).index_select(0, indices)
        )
    return per_image


def raw_logit_target_object_scores(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    response_fraction: float = 0.25,
    connectivity: int = RAW_LOGIT_TAILRANK_CONNECTIVITY,
) -> torch.Tensor:
    """Aggregate one differentiable raw-logit score per GT component."""

    response_fraction = _validate_fraction(response_fraction, "response_fraction")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    work_logits = (
        logits
        if logits.dtype in (torch.float32, torch.float64)
        else logits.float()
    )
    scores: list[torch.Tensor] = []
    skimage_connectivity = 1 if connectivity == 4 else 2
    for batch_index in range(logits.shape[0]):
        binary = target[batch_index, 0].detach().cpu().numpy() >= 0.5
        labels = measure.label(binary, connectivity=skimage_connectivity)
        label_count = int(labels.max())
        if label_count == 0:
            continue
        labels_tensor = torch.from_numpy(labels).to(device=work_logits.device)
        for component_id in range(1, label_count + 1):
            values = work_logits[batch_index, 0].masked_select(
                labels_tensor == component_id
            )
            scores.append(
                _fraction_mean(values, response_fraction, largest=True)
            )
    if not scores:
        return work_logits.reshape(-1)[:0]
    return torch.stack(scores)


def compute_raw_logit_tailrank_margin(
    logits: torch.Tensor,
    target: torch.Tensor,
    domain_ids: torch.Tensor,
    *,
    background_fraction: float = 0.01,
    miss_fraction: float = 0.2,
    object_top_fraction: float = 0.25,
    gamma: float = 10.0,
    margin: float = RAW_LOGIT_TAILRANK_MARGIN,
    min_peak_score: float = 0.05,
    exclusion_radius: int = 2,
    target_connectivity: int = RAW_LOGIT_TAILRANK_CONNECTIVITY,
) -> RawLogitTailRankComponents:
    """Compute Candidate A losses without differentiable sigmoid scores."""

    _validate_inputs(logits, target, domain_ids)
    background_fraction = _validate_fraction(
        background_fraction, "background_fraction"
    )
    miss_fraction = _validate_fraction(miss_fraction, "miss_fraction")
    object_top_fraction = _validate_fraction(
        object_top_fraction, "object_top_fraction"
    )
    if not math.isfinite(float(gamma)) or float(gamma) <= 0.0:
        raise ValueError("gamma must be finite and positive")
    if not math.isfinite(float(margin)) or float(margin) < 0.0:
        raise ValueError("margin must be finite and non-negative")
    if target_connectivity not in (4, 8):
        raise ValueError("target_connectivity must be 4 or 8")

    work_logits = (
        logits
        if logits.dtype in (torch.float32, torch.float64)
        else logits.float()
    )
    per_image_peaks = raw_logit_background_peaks(
        work_logits,
        target,
        min_peak_score=min_peak_score,
        exclusion_radius=exclusion_radius,
    )
    domain_tail_logits: list[torch.Tensor] = []
    total_peaks = 0
    for domain_tensor in torch.unique(domain_ids.detach(), sorted=True):
        domain_values: list[torch.Tensor] = []
        selected_indices = torch.nonzero(
            domain_ids == domain_tensor, as_tuple=False
        ).flatten()
        for index_tensor in selected_indices:
            values = per_image_peaks[int(index_tensor.item())]
            total_peaks += int(values.numel())
            if values.numel() > 0:
                domain_values.append(values)
        if domain_values:
            domain_tail_logits.append(
                _fraction_mean(
                    torch.cat(domain_values),
                    background_fraction,
                    largest=True,
                )
            )

    connected_zero = work_logits.sum() * 0.0
    if domain_tail_logits:
        stacked = torch.stack(domain_tail_logits)
        worst_background_logit = torch.logsumexp(
            stacked * float(gamma), dim=0
        ) / float(gamma)
        # Unlike sigmoid(logit), softplus(logit) retains derivative ~1 for a
        # catastrophic positive background logit of 10--20.
        background_tail = F.softplus(worst_background_logit)
    else:
        worst_background_logit = connected_zero
        background_tail = connected_zero

    target_scores = raw_logit_target_object_scores(
        work_logits,
        target,
        response_fraction=object_top_fraction,
        connectivity=target_connectivity,
    )
    if target_scores.numel() > 0:
        miss_scores = F.softplus(-target_scores)
        hard_miss = _fraction_mean(miss_scores, miss_fraction, largest=True)
        hard_target_logit = _fraction_mean(
            target_scores, miss_fraction, largest=False
        )
        separation_margin = (
            F.relu(float(margin) + worst_background_logit - hard_target_logit)
            if domain_tail_logits
            else connected_zero
        )
    else:
        hard_miss = connected_zero
        hard_target_logit = connected_zero
        separation_margin = connected_zero

    return RawLogitTailRankComponents(
        background_tail=background_tail,
        hard_miss=hard_miss,
        separation_margin=separation_margin,
        worst_background_logit=worst_background_logit,
        hard_target_logit=hard_target_logit,
        num_background_peaks=total_peaks,
        num_target_objects=int(target_scores.numel()),
    )


__all__ = [
    "RAW_LOGIT_TAILRANK_CONNECTIVITY",
    "RAW_LOGIT_TAILRANK_LAMBDA_MARGIN",
    "RAW_LOGIT_TAILRANK_LAMBDA_MISS",
    "RAW_LOGIT_TAILRANK_LAMBDA_TAIL",
    "RAW_LOGIT_TAILRANK_MARGIN",
    "RAW_LOGIT_TAILRANK_MODE",
    "RawLogitTailRankComponents",
    "compute_raw_logit_tailrank_margin",
    "raw_logit_background_peaks",
    "raw_logit_target_object_scores",
]
