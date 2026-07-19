from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F

from losses.local_peak_cvar import local_background_peak_scores


@dataclass
class TailRiskComponents:
    background_tail: torch.Tensor
    hard_miss: torch.Tensor
    separation_margin: torch.Tensor
    worst_domain_tail: torch.Tensor
    num_background_peaks: int
    num_target_objects: int


def top_fraction_mean(values: torch.Tensor, fraction: float, *, largest: bool = True) -> torch.Tensor:
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return values.sum() * 0.0
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    count = max(1, int(math.ceil(fraction * flat.numel())))
    selected = torch.topk(flat, k=count, largest=largest).values
    return selected.mean()


def local_background_peaks(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    kernel_size: int = 3,
    exclusion_radius: int = 2,
    min_score: float = 0.05,
) -> list[torch.Tensor]:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    if exclusion_radius > 0:
        dilation_kernel = 2 * exclusion_radius + 1
        excluded = F.max_pool2d(
            target,
            kernel_size=dilation_kernel,
            stride=1,
            padding=exclusion_radius,
        )
    else:
        excluded = target
    # Delegate plateau extraction to the canonical optimized implementation:
    # one CPU connected-component pass and one differentiable GPU gather per
    # image, rather than one kernel for every plateau pixel/component.
    return local_background_peak_scores(
        logits,
        (excluded >= 0.5).to(dtype=target.dtype),
        kernel_size=kernel_size,
        min_score=min_score,
    )


def object_scores(
    probability: torch.Tensor,
    target: torch.Tensor,
    *,
    top_fraction: float = 0.25,
) -> list[torch.Tensor]:
    """Return differentiable scores for each GT connected component."""
    scores: list[torch.Tensor] = []
    target_cpu = (target.detach().cpu().numpy() >= 0.5).astype(np.uint8)
    for batch_index in range(target.shape[0]):
        labeled, count = ndimage.label(target_cpu[batch_index, 0])
        for component_id in range(1, count + 1):
            component_numpy = labeled == component_id
            component = torch.from_numpy(component_numpy).to(device=probability.device)
            values = probability[batch_index, 0][component]
            if values.numel() == 0:
                continue
            scores.append(top_fraction_mean(values, top_fraction, largest=True))
    return scores


def smooth_max(values: torch.Tensor, gamma: float) -> torch.Tensor:
    if values.numel() == 0:
        return values.sum() * 0.0
    return torch.logsumexp(values * gamma, dim=0) / gamma


def compute_tail_risk(
    logits: torch.Tensor,
    target: torch.Tensor,
    domain_ids: torch.Tensor,
    *,
    background_fraction: float = 0.01,
    miss_fraction: float = 0.2,
    object_top_fraction: float = 0.25,
    gamma: float = 10.0,
    margin: float = 0.15,
    min_peak_score: float = 0.05,
    exclusion_radius: int = 2,
) -> TailRiskComponents:
    probability = torch.sigmoid(logits)
    per_image_peaks = local_background_peaks(
        logits,
        target,
        min_score=min_peak_score,
        exclusion_radius=exclusion_radius,
    )
    domain_risks: list[torch.Tensor] = []
    total_peaks = 0
    for domain_id in torch.unique(domain_ids).tolist():
        selected: list[torch.Tensor] = []
        indices = torch.nonzero(domain_ids == int(domain_id), as_tuple=False).flatten().tolist()
        for index in indices:
            values = per_image_peaks[index]
            total_peaks += int(values.numel())
            if values.numel() > 0:
                selected.append(values)
        if selected:
            domain_values = torch.cat(selected)
            domain_risks.append(top_fraction_mean(domain_values, background_fraction))
        else:
            domain_risks.append(logits.sum() * 0.0)
    stacked_domain_risks = torch.stack(domain_risks)
    worst_domain_tail = smooth_max(stacked_domain_risks, gamma)
    background_tail = stacked_domain_risks.mean()

    target_scores = object_scores(
        probability,
        target,
        top_fraction=object_top_fraction,
    )
    if target_scores:
        target_vector = torch.stack(target_scores)
        hard_miss = top_fraction_mean(1.0 - target_vector, miss_fraction)
        hard_target_score = top_fraction_mean(target_vector, miss_fraction, largest=False)
        separation_margin = F.relu(margin + worst_domain_tail - hard_target_score)
    else:
        hard_miss = logits.sum() * 0.0
        separation_margin = logits.sum() * 0.0

    return TailRiskComponents(
        background_tail=background_tail,
        hard_miss=hard_miss,
        separation_margin=separation_margin,
        worst_domain_tail=worst_domain_tail,
        num_background_peaks=total_peaks,
        num_target_objects=len(target_scores),
    )
