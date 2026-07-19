from __future__ import annotations

from pathlib import Path

import torch

from model.MSHNet import MSHNet
from rc_irstd.models import build_mshnet, forward_mshnet
from rc_irstd.models.rc_mshnet import (
    RCMSHNet,
    initialize_rc_mshnet_from_checkpoint,
)


def _input(batch: int = 2) -> torch.Tensor:
    torch.manual_seed(17)
    return torch.randn(batch, 3, 32, 32)


def test_rc_mshnet_builder_and_output_contract() -> None:
    model = build_mshnet(
        {
            "backend": "rc_mshnet",
            "fusion_channels": 8,
            "contrast_windows": [3, 7, 15],
            "context_dilations": [1, 2],
        }
    )
    assert isinstance(model, RCMSHNet)
    output = forward_mshnet(model.eval(), _input(), warm_flag=True)
    assert output.logits.shape == (2, 1, 32, 32)
    # Four original MSHNet scales plus two residual-correction auxiliaries.
    assert len(output.auxiliary_logits) == 6
    assert model.export_config()["baseline_identity"] == "canonical_mshnet"


def test_zero_residual_initialization_is_exact_mshnet() -> None:
    torch.manual_seed(5)
    baseline = MSHNet(3).eval()
    proposed = RCMSHNet(fusion_channels=8).eval()
    incompatible = proposed.load_state_dict(baseline.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert all(
        any(key.startswith(prefix) for prefix in proposed.extension_prefixes)
        for key in incompatible.missing_keys
    )
    inputs = _input()
    with torch.no_grad():
        _, baseline_logits = baseline(inputs, True)
        proposed_logits = proposed(inputs, multi_scale=True).logits
    torch.testing.assert_close(proposed_logits, baseline_logits, rtol=0.0, atol=0.0)


def test_rc_mshnet_correction_heads_receive_gradient() -> None:
    model = RCMSHNet(fusion_channels=8).train()
    logits = model(_input()).logits
    logits.square().mean().backward()
    contrast_gradient = model.fusion_head.contrast_delta[-1].weight.grad
    component_gradient = model.fusion_head.component_delta[-1].weight.grad
    assert contrast_gradient is not None
    assert component_gradient is not None
    assert float(contrast_gradient.abs().sum()) > 0.0
    assert float(component_gradient.abs().sum()) > 0.0


def test_all_extensions_disabled_reduce_to_mshnet() -> None:
    torch.manual_seed(31)
    baseline = MSHNet(3).eval()
    proposed = RCMSHNet(
        fusion_channels=8,
        use_contrast=False,
        use_component_context=False,
        expose_branch_auxiliary=False,
    ).eval()
    proposed.load_state_dict(baseline.state_dict(), strict=False)
    inputs = _input()
    with torch.no_grad():
        baseline_auxiliary, baseline_logits = baseline(inputs, True)
        output = proposed(inputs, multi_scale=True)
    torch.testing.assert_close(output.logits, baseline_logits, rtol=0.0, atol=0.0)
    assert len(output.auxiliary_logits) == len(baseline_auxiliary) == 4


def test_checkpoint_initializer_requires_complete_mshnet_backbone(
    tmp_path: Path,
) -> None:
    baseline = MSHNet(3)
    checkpoint = tmp_path / "mshnet.pt"
    torch.save({"model_state": baseline.state_dict()}, checkpoint)
    model = RCMSHNet(fusion_channels=8)
    report = initialize_rc_mshnet_from_checkpoint(model, checkpoint)
    assert report["backbone_fully_loaded"] is True
    assert report["zero_residual_identity_preserved"] is True
    assert report["unexpected_keys"] == []
    assert len(report["source_sha256"]) == 64


def test_parameter_overhead_is_lightweight() -> None:
    baseline = MSHNet(3)
    proposed = RCMSHNet(fusion_channels=16)
    baseline_parameters = sum(value.numel() for value in baseline.parameters())
    proposed_parameters = sum(value.numel() for value in proposed.parameters())
    overhead = (proposed_parameters - baseline_parameters) / baseline_parameters
    assert 0.0 < overhead < 0.02
