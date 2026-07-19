from __future__ import annotations

import torch

from rc_irstd.losses import DetectorObjective, calibrator_objective
from rc_irstd.losses.sls import StableSLSLoss
from rc_irstd.models import build_mshnet


def test_complete_detector_objective_backpropagates() -> None:
    model = build_mshnet(
        {
            "backend": "complete_compat",
            "input_channels": 3,
            "channels": [2, 4, 8, 16, 32],
            "block_counts": [1, 1, 1, 1],
        }
    )
    images = torch.randn(2, 3, 32, 32)
    masks = torch.zeros(2, 1, 32, 32)
    masks[0, 0, 10:13, 14:17] = 1
    masks[1, 0, 20:24, 5:9] = 1
    domain_ids = torch.tensor([0, 1])
    output = model(images, multi_scale=True)
    objective = DetectorObjective(
        lambda_tail=0.05,
        lambda_miss=0.05,
        lambda_margin=0.05,
        background_fraction=0.1,
    )
    result = objective(output, masks, domain_ids)
    assert torch.isfinite(result.total)
    result.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_complete_calibrator_objective_is_finite() -> None:
    prediction = torch.tensor([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]], requires_grad=True)
    target = prediction.detach() + 0.2
    background_hist = torch.ones(2, 16)
    object_hist = torch.ones(2, 16)
    centers = torch.linspace(-8, 8, 16)
    budgets = torch.tensor([1e-2, 5e-3, 1e-3])
    result = calibrator_objective(
        prediction,
        target,
        budgets,
        background_hist,
        object_hist,
        centers,
    )
    assert torch.isfinite(result.total)
    result.total.backward()
    assert prediction.grad is not None


def test_stable_sls_half_precision_large_empty_mask_is_finite() -> None:
    logits = torch.zeros(1, 1, 256, 256, dtype=torch.float16, requires_grad=True)
    target = torch.zeros_like(logits)
    output = StableSLSLoss()(logits, target)
    assert output.total.dtype == torch.float32
    assert torch.isfinite(output.total)
    assert torch.isfinite(output.scale_iou)
    output.total.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
