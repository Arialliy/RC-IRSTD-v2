from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SLSComponents:
    total: torch.Tensor
    bce: torch.Tensor
    scale_iou: torch.Tensor
    location: torch.Tensor


def _soft_centroid(probability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, _, height, width = probability.shape
    y = torch.linspace(0.0, 1.0, height, device=probability.device, dtype=probability.dtype)
    x = torch.linspace(0.0, 1.0, width, device=probability.device, dtype=probability.dtype)
    mass = probability.sum(dim=(1, 2, 3)).clamp_min(1e-8)
    center_y = (probability * y[None, None, :, None]).sum(dim=(1, 2, 3)) / mass
    center_x = (probability * x[None, None, None, :]).sum(dim=(1, 2, 3)) / mass
    return center_y, center_x, mass


class StableSLSLoss(nn.Module):
    """Numerically stable scale- and location-sensitive segmentation loss.

    This keeps the two ideas of SLS—scale-aware overlap and soft-center
    localization—while adding a bounded class-balanced BCE term so that empty
    and extremely sparse masks remain trainable.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        iou_weight: float = 1.0,
        location_weight: float = 0.25,
        max_positive_weight: float = 50.0,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.iou_weight = float(iou_weight)
        self.location_weight = float(location_weight)
        self.max_positive_weight = float(max_positive_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> SLSComponents:
        if logits.shape != target.shape:
            raise ValueError(f"logits and target must match: {logits.shape} vs {target.shape}")
        # Ordinary 256x256 pixel-count reductions overflow float16 (65536),
        # turning the scale term into inf/inf under CUDA AMP. Keep the graph
        # connected to the autocast logits but evaluate all loss arithmetic in
        # float32, matching the hardened canonical SLS implementation.
        work_logits = (
            logits.float()
            if logits.dtype in (torch.float16, torch.bfloat16)
            else logits
        )
        target = target.to(dtype=work_logits.dtype)
        positive = target.sum()
        negative = target.numel() - positive
        positive_weight = (negative / positive.clamp_min(1.0)).clamp(
            min=1.0, max=self.max_positive_weight
        )
        bce = F.binary_cross_entropy_with_logits(
            work_logits,
            target,
            pos_weight=positive_weight.detach(),
        )

        probability = torch.sigmoid(work_logits)
        intersection = (probability * target).sum(dim=(1, 2, 3))
        predicted_area = probability.sum(dim=(1, 2, 3))
        target_area = target.sum(dim=(1, 2, 3))
        union = predicted_area + target_area - intersection
        iou = (intersection + 1.0) / (union + 1.0)
        area_gap = ((predicted_area - target_area) * 0.5).square()
        scale_factor = (
            torch.minimum(predicted_area, target_area) + area_gap + 1.0
        ) / (torch.maximum(predicted_area, target_area) + area_gap + 1.0)
        scale_iou = (1.0 - scale_factor * iou).mean()

        pred_y, pred_x, _ = _soft_centroid(probability)
        target_y, target_x, target_mass = _soft_centroid(target)
        center_distance = (pred_y - target_y).square() + (pred_x - target_x).square()
        location = torch.where(target_mass > 1e-6, center_distance, torch.zeros_like(center_distance))
        location = location.mean()

        total = (
            self.bce_weight * bce
            + self.iou_weight * scale_iou
            + self.location_weight * location
        )
        return SLSComponents(total=total, bce=bce, scale_iou=scale_iou, location=location)
