from __future__ import annotations

import pytest
import torch

from rc_irstd.models import MonotoneBudgetCalibrator, build_mshnet
from rc_irstd.evaluation.metrics import REJECT_ALL_LATENT_LOGIT, REJECT_ALL_THRESHOLD


def _tiny_complete_model():
    # The configurable reference-package detector is retained only as an
    # explicit compatibility/smoke backend. Formal runs default to canonical
    # checkpoint-compatible MSHNet.
    return build_mshnet(
        {
            "backend": "complete_compat",
            "input_channels": 3,
            "channels": [2, 4, 8, 16, 32],
            "block_counts": [1, 1, 1, 1],
        }
    )


def test_complete_mshnet_output_shapes() -> None:
    model = _tiny_complete_model()
    inputs = torch.randn(1, 3, 32, 32)
    output = model(inputs, multi_scale=True)
    assert output.logits.shape == (1, 1, 32, 32)
    assert len(output.auxiliary_logits) == 4
    assert output.auxiliary_logits[0].shape[-2:] == (32, 32)
    warm = model(inputs, multi_scale=False)
    assert warm.logits.shape == (1, 1, 32, 32)
    assert warm.auxiliary_logits == ()


def test_formal_model_rejects_silent_width_changes() -> None:
    with pytest.raises(ValueError, match="complete_compat"):
        build_mshnet(
            {
                "backend": "canonical",
                "channels": [2, 4, 8, 16, 32],
                "block_counts": [1, 1, 1, 1],
            }
        )


def test_calibrator_is_budget_monotone() -> None:
    model = MonotoneBudgetCalibrator(
        feature_dim=12,
        budget_grid=[1e-4, 1e-5, 1e-6],
        hidden_dims=(16, 8),
        dropout=0.0,
    )
    features = torch.randn(5, 12)
    model.normalizer.fit(features)
    output = model(features)
    assert output.grid_logits.shape == (5, 3)
    assert torch.all(output.grid_logits[:, 1:] >= output.grid_logits[:, :-1])
    requested = model(features, torch.tensor([5e-5, 5e-6]))
    assert requested.requested_thresholds is not None
    assert requested.requested_thresholds.shape == (5, 2)


def test_calibrator_rejects_out_of_grid_budgets() -> None:
    model = MonotoneBudgetCalibrator(
        feature_dim=4,
        budget_grid=[1e-4, 1e-5, 1e-6],
        hidden_dims=(8,),
        dropout=0.0,
    )
    features = torch.randn(3, 4)
    model.normalizer.fit(features)
    with pytest.raises(ValueError, match="inside the trained budget grid"):
        model(features, torch.tensor([1e-7]))


def test_calibrator_can_emit_reject_all_sentinel() -> None:
    latent = torch.tensor([[REJECT_ALL_LATENT_LOGIT]])
    threshold = MonotoneBudgetCalibrator.logits_to_thresholds(latent)
    assert float(threshold.item()) == REJECT_ALL_THRESHOLD
    assert float(threshold.item()) > 1.0
