from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from model.MSHNet import MSHNet
from rc_irstd.models.rc_mshnet import (
    RCMSHNet,
    RC_MSHNET_ARCHITECTURE_VERSION_V1,
    RC_MSHNET_ARCHITECTURE_VERSION_V2,
    build_rc_mshnet,
    initialize_rc_mshnet_from_checkpoint,
    rc_mshnet_extension_state_sha256,
)


def _small_v2(**overrides: object) -> RCMSHNet:
    values: dict[str, object] = {
        "fusion_channels": 8,
        "contrast_windows": (3,),
        "context_dilations": (1,),
        "architecture_version": RC_MSHNET_ARCHITECTURE_VERSION_V2,
        "use_component_context": True,
        "use_component_expert": False,
        "expose_branch_auxiliary": False,
    }
    values.update(overrides)
    return RCMSHNet(**values)


def _inputs() -> torch.Tensor:
    generator = torch.Generator().manual_seed(187)
    return torch.randn(1, 3, 32, 32, generator=generator)


def test_v1_defaults_and_export_remain_frozen() -> None:
    model = RCMSHNet(fusion_channels=8)
    assert model.architecture_version == RC_MSHNET_ARCHITECTURE_VERSION_V1
    assert model.use_component_expert is model.use_component_context is True
    assert model.expose_branch_auxiliary is True
    assert model.export_config() == {
        "architecture_version": "rc-mshnet-v1",
        "backend": "rc_mshnet",
        "input_channels": 3,
        "channels": [16, 32, 64, 128, 256],
        "block_counts": [2, 2, 2, 2],
        "fusion_channels": 8,
        "contrast_windows": [3, 7, 15],
        "context_dilations": [1, 2, 4],
        "component_support_window": 3,
        "component_ring_window": 9,
        "use_contrast": True,
        "use_component_context": True,
        "use_risk_gate": True,
        "expose_branch_auxiliary": True,
        "baseline_identity": "canonical_mshnet",
        "initialization_contract": "zero_residual_exact_mshnet",
    }
    rebuilt = build_rc_mshnet(model.export_config())
    assert rebuilt.architecture_version == RC_MSHNET_ARCHITECTURE_VERSION_V1
    assert rebuilt.use_component_expert is True


def test_v2_version_is_authoritative_and_boole_are_strict() -> None:
    config = {
        "architecture_version": RC_MSHNET_ARCHITECTURE_VERSION_V2,
        "fusion_channels": 8,
        "use_component_context": True,
        "use_component_expert": False,
        "expose_branch_auxiliary": False,
    }
    model = build_rc_mshnet(config)
    assert model.architecture_version == RC_MSHNET_ARCHITECTURE_VERSION_V2
    assert model.export_config()["use_component_expert"] is False
    rebuilt = build_rc_mshnet(model.export_config())
    assert rebuilt.use_component_expert is False

    with pytest.raises(ValueError, match="Unsupported.*architecture_version"):
        build_rc_mshnet({"architecture_version": "rc-mshnet-v0"})
    with pytest.raises(ValueError, match="not valid for rc-mshnet-v1"):
        build_rc_mshnet({"use_component_expert": False})
    with pytest.raises(ValueError, match="requires the explicit"):
        build_rc_mshnet(
            {"architecture_version": RC_MSHNET_ARCHITECTURE_VERSION_V2}
        )
    with pytest.raises(TypeError, match="use_component_expert must be a boolean"):
        build_rc_mshnet(
            {
                "architecture_version": RC_MSHNET_ARCHITECTURE_VERSION_V2,
                "use_component_expert": 0,
            }
        )
    with pytest.raises(TypeError, match="use_contrast must be a boolean"):
        build_rc_mshnet({"use_contrast": "false"})


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {
                "use_component_context": False,
                "use_component_expert": True,
            },
            "requires use_component_context=true",
        ),
        (
            {
                "use_contrast": False,
                "use_component_context": True,
                "use_component_expert": False,
            },
            "dead branch",
        ),
        (
            {
                "use_risk_gate": False,
                "use_component_context": True,
                "use_component_expert": False,
            },
            "dead branch",
        ),
    ],
)
def test_illegal_or_silent_dead_component_roles_fail_closed(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _small_v2(**overrides)


def test_gate_context_reaches_gate_but_masks_component_expert_and_aux() -> None:
    model = _small_v2(expose_branch_auxiliary=True).eval()
    captured: dict[str, torch.Tensor] = {}
    expert_calls: list[bool] = []

    def hook(
        module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
        kwargs: dict[str, object],
        output: tuple[torch.Tensor, ...],
    ) -> None:
        del module, kwargs
        captured["component_feature"] = args[2]
        captured["component_delta"] = output[2]
        captured["gates"] = output[3]

    expert_handle = model.fusion_head.component_delta.register_forward_hook(
        lambda module, args, output: expert_calls.append(True)
    )
    handle = model.fusion_head.register_forward_hook(hook, with_kwargs=True)
    with torch.no_grad():
        output = model(_inputs())
    handle.remove()
    expert_handle.remove()

    assert expert_calls == []
    assert torch.count_nonzero(captured["component_feature"]).item() > 0
    assert torch.count_nonzero(captured["component_delta"]).item() == 0
    assert captured["gates"][:, 1:2].abs().max().item() < 1.0e-6
    # Four canonical scales plus contrast auxiliary; component has no auxiliary.
    assert len(output.auxiliary_logits) == 5


def test_contrast_only_and_full_v2_expert_wiring() -> None:
    contrast_only = _small_v2(
        use_component_context=False,
        use_component_expert=False,
        expose_branch_auxiliary=True,
    ).eval()
    c_context_calls: list[bool] = []
    c_expert_calls: list[bool] = []
    c_gates: list[torch.Tensor] = []
    c_handles = (
        contrast_only.component_context.register_forward_hook(
            lambda module, args, output: c_context_calls.append(True)
        ),
        contrast_only.fusion_head.component_delta.register_forward_hook(
            lambda module, args, output: c_expert_calls.append(True)
        ),
        contrast_only.fusion_head.register_forward_hook(
            lambda module, args, output: c_gates.append(output[3])
        ),
    )
    with torch.no_grad():
        c_output = contrast_only(_inputs())
    for handle in c_handles:
        handle.remove()
    assert c_context_calls == []
    assert c_expert_calls == []
    assert c_gates[0][:, 1:2].abs().max().item() < 1.0e-6
    assert len(c_output.auxiliary_logits) == 5

    full = _small_v2(
        use_component_context=True,
        use_component_expert=True,
        expose_branch_auxiliary=True,
    ).eval()
    full_context_calls: list[bool] = []
    full_expert_calls: list[bool] = []
    full_gates: list[torch.Tensor] = []
    full_handles = (
        full.component_context.register_forward_hook(
            lambda module, args, output: full_context_calls.append(True)
        ),
        full.fusion_head.component_delta.register_forward_hook(
            lambda module, args, output: full_expert_calls.append(True)
        ),
        full.fusion_head.register_forward_hook(
            lambda module, args, output: full_gates.append(output[3])
        ),
    )
    with torch.no_grad():
        full_output = full(_inputs())
    for handle in full_handles:
        handle.remove()
    assert full_context_calls == [True]
    assert full_expert_calls == [True]
    assert full_gates[0][:, 1:2].max().item() > 0.0
    assert len(full_output.auxiliary_logits) == 6



def test_gate_context_initial_output_is_exact_canonical_identity() -> None:
    torch.manual_seed(211)
    baseline = MSHNet(3).eval()
    candidate = _small_v2().eval()
    incompatible = candidate.load_state_dict(baseline.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    inputs = _inputs()
    with torch.no_grad():
        _, baseline_logits = baseline(inputs, True)
        candidate_logits = candidate(inputs).logits
    torch.testing.assert_close(candidate_logits, baseline_logits, rtol=0.0, atol=0.0)


def test_complete_extension_state_hash_and_checkpoint_identity(
    tmp_path: Path,
) -> None:
    torch.manual_seed(307)
    model = _small_v2()
    before = rc_mshnet_extension_state_sha256(model)
    extension_keys = sorted(
        key
        for key in model.state_dict()
        if any(key.startswith(prefix) for prefix in model.extension_prefixes)
    )

    torch.manual_seed(307)
    same = _small_v2()
    assert rc_mshnet_extension_state_sha256(same) == before
    with torch.no_grad():
        next(same.contrast_pyramid.parameters()).add_(1.0)
    assert rc_mshnet_extension_state_sha256(same) != before

    checkpoint = tmp_path / "canonical.pt"
    baseline = MSHNet(3)
    torch.save(
        {
            "model_state": baseline.state_dict(),
            "model_config": {
                "backend": "canonical",
                "input_channels": 3,
                "channels": [16, 32, 64, 128, 256],
                "block_counts": [2, 2, 2, 2],
            },
        },
        checkpoint,
    )
    report = initialize_rc_mshnet_from_checkpoint(model, checkpoint)
    assert report["source_checkpoint_identity"]["resolved_identity"] == (
        "canonical_mshnet"
    )
    assert report["initial_extension_key_count"] == len(extension_keys)
    assert report["initial_extension_state_sha256"] == before
    assert report["extension_state_preserved"] is True
    assert rc_mshnet_extension_state_sha256(model) == before

    disguised_rc = tmp_path / "disguised_rc.pt"
    torch.save(
        {
            "model_state": baseline.state_dict(),
            "model_config": {
                "backend": "rc_mshnet",
                "architecture_version": RC_MSHNET_ARCHITECTURE_VERSION_V2,
            },
        },
        disguised_rc,
    )
    with pytest.raises(RuntimeError, match="rejects checkpoint model_config"):
        initialize_rc_mshnet_from_checkpoint(_small_v2(), disguised_rc)


def test_gate_context_receives_gradient_after_three_backward_steps() -> None:
    torch.manual_seed(401)
    model = _small_v2().train()
    for name, parameter in model.named_parameters():
        if not any(name.startswith(prefix) for prefix in model.extension_prefixes):
            parameter.requires_grad_(False)
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.05,
    )
    inputs = _inputs()
    target = torch.zeros(1, 1, 32, 32)
    context_gradient_sums: list[float] = []

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(inputs).logits, target)
        loss.backward()
        context_gradients = [
            parameter.grad
            for parameter in model.component_context.parameters()
            if parameter.grad is not None
        ]
        assert all(torch.isfinite(gradient).all() for gradient in context_gradients)
        context_gradient_sums.append(
            sum(float(gradient.abs().sum()) for gradient in context_gradients)
        )
        optimizer.step()

    assert len(context_gradient_sums) == 3
    assert context_gradient_sums[-1] > 0.0
