from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from rc_irstd.models.mshnet import MSHNetOutput

from .sls import StableSLSLoss
from .tail import compute_tail_risk
from .tail_rank import (
    RAW_LOGIT_TAILRANK_CONNECTIVITY,
    RAW_LOGIT_TAILRANK_LAMBDA_MARGIN,
    RAW_LOGIT_TAILRANK_LAMBDA_MISS,
    RAW_LOGIT_TAILRANK_LAMBDA_TAIL,
    RAW_LOGIT_TAILRANK_MARGIN,
    RAW_LOGIT_TAILRANK_MODE,
    compute_raw_logit_tailrank_margin,
)


LEGACY_PROBABILITY_TAIL_MODE = "probability_tail_miss_v1"


@dataclass
class DetectorLossOutput:
    total: torch.Tensor
    metrics: dict[str, float]


class DetectorObjective(nn.Module):
    def __init__(
        self,
        *,
        lambda_tail: float = 0.1,
        lambda_miss: float = 0.1,
        lambda_margin: float = 0.1,
        auxiliary_weight: float = 0.25,
        background_fraction: float = 0.01,
        miss_fraction: float = 0.2,
        object_top_fraction: float = 0.25,
        gamma: float = 10.0,
        margin: float = 0.15,
        min_peak_score: float = 0.05,
        exclusion_radius: int = 2,
        tail_mode: str = LEGACY_PROBABILITY_TAIL_MODE,
        target_connectivity: int = 4,
        sls_kwargs: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.sls = StableSLSLoss(**(sls_kwargs or {}))
        self.lambda_tail = float(lambda_tail)
        self.lambda_miss = float(lambda_miss)
        self.lambda_margin = float(lambda_margin)
        self.auxiliary_weight = float(auxiliary_weight)
        self.background_fraction = float(background_fraction)
        self.miss_fraction = float(miss_fraction)
        self.object_top_fraction = float(object_top_fraction)
        self.gamma = float(gamma)
        self.margin = float(margin)
        self.min_peak_score = float(min_peak_score)
        self.exclusion_radius = int(exclusion_radius)
        self.tail_mode = str(tail_mode)
        self.target_connectivity = int(target_connectivity)
        if self.tail_mode not in {
            LEGACY_PROBABILITY_TAIL_MODE,
            RAW_LOGIT_TAILRANK_MODE,
        }:
            raise ValueError(f"Unsupported detector tail_mode: {self.tail_mode!r}")
        if self.tail_mode == LEGACY_PROBABILITY_TAIL_MODE:
            if self.target_connectivity != 4:
                raise ValueError(
                    "Legacy probability Tail/Miss has fixed 4-neighbor target "
                    "semantics; use the raw-logit Candidate A mode for 8-neighbor"
                )
        else:
            required = {
                "lambda_tail": (
                    self.lambda_tail,
                    RAW_LOGIT_TAILRANK_LAMBDA_TAIL,
                ),
                "lambda_miss": (
                    self.lambda_miss,
                    RAW_LOGIT_TAILRANK_LAMBDA_MISS,
                ),
                "lambda_margin": (
                    self.lambda_margin,
                    RAW_LOGIT_TAILRANK_LAMBDA_MARGIN,
                ),
                "margin": (self.margin, RAW_LOGIT_TAILRANK_MARGIN),
            }
            for field, (actual, expected) in required.items():
                if actual != expected:
                    raise ValueError(
                        f"Raw-logit TailRank Candidate A requires "
                        f"{field}={expected}"
                    )
            if self.target_connectivity != RAW_LOGIT_TAILRANK_CONNECTIVITY:
                raise ValueError(
                    "Raw-logit TailRank Candidate A requires "
                    f"target_connectivity={RAW_LOGIT_TAILRANK_CONNECTIVITY}"
                )

    def forward(
        self,
        output: MSHNetOutput,
        target: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> DetectorLossOutput:
        base = self.sls(output.logits, target)
        auxiliary_total = output.logits.sum() * 0.0
        if output.auxiliary_logits:
            auxiliary_losses: list[torch.Tensor] = []
            for auxiliary in output.auxiliary_logits:
                resized_target = F.interpolate(target, size=auxiliary.shape[-2:], mode="nearest")
                auxiliary_losses.append(self.sls(auxiliary, resized_target).total)
            auxiliary_total = torch.stack(auxiliary_losses).mean()

        # RC-MSHNET-PATCH: SLS-only fast path
        if (
            self.lambda_tail == 0.0
            and self.lambda_miss == 0.0
            and self.lambda_margin == 0.0
        ):
            total = base.total + self.auxiliary_weight * auxiliary_total
            metrics = {
                "loss_total": float(total.detach().cpu()),
                "loss_sls": float(base.total.detach().cpu()),
                "loss_bce": float(base.bce.detach().cpu()),
                "loss_scale_iou": float(base.scale_iou.detach().cpu()),
                "loss_location": float(base.location.detach().cpu()),
                "loss_auxiliary": float(auxiliary_total.detach().cpu()),
                "loss_tail": 0.0,
                "loss_miss": 0.0,
                "loss_margin": 0.0,
                "num_background_peaks": 0.0,
                "num_target_objects": 0.0,
            }
            return DetectorLossOutput(total=total, metrics=metrics)

        if self.tail_mode == RAW_LOGIT_TAILRANK_MODE:
            tail = compute_raw_logit_tailrank_margin(
                output.logits,
                target,
                domain_ids,
                background_fraction=self.background_fraction,
                miss_fraction=self.miss_fraction,
                object_top_fraction=self.object_top_fraction,
                gamma=self.gamma,
                margin=self.margin,
                min_peak_score=self.min_peak_score,
                exclusion_radius=self.exclusion_radius,
                target_connectivity=self.target_connectivity,
            )
            tail_loss = tail.background_tail
        else:
            tail = compute_tail_risk(
                output.logits,
                target,
                domain_ids,
                background_fraction=self.background_fraction,
                miss_fraction=self.miss_fraction,
                object_top_fraction=self.object_top_fraction,
                gamma=self.gamma,
                margin=self.margin,
                min_peak_score=self.min_peak_score,
                exclusion_radius=self.exclusion_radius,
            )
            tail_loss = tail.worst_domain_tail
        total = (
            base.total
            + self.auxiliary_weight * auxiliary_total
            + self.lambda_tail * tail_loss
            + self.lambda_miss * tail.hard_miss
            + self.lambda_margin * tail.separation_margin
        )
        metrics = {
            "loss_total": float(total.detach().cpu()),
            "loss_sls": float(base.total.detach().cpu()),
            "loss_bce": float(base.bce.detach().cpu()),
            "loss_scale_iou": float(base.scale_iou.detach().cpu()),
            "loss_location": float(base.location.detach().cpu()),
            "loss_auxiliary": float(auxiliary_total.detach().cpu()),
            "loss_tail": float(tail_loss.detach().cpu()),
            "loss_miss": float(tail.hard_miss.detach().cpu()),
            "loss_margin": float(tail.separation_margin.detach().cpu()),
            "num_background_peaks": float(tail.num_background_peaks),
            "num_target_objects": float(tail.num_target_objects),
        }
        return DetectorLossOutput(total=total, metrics=metrics)
