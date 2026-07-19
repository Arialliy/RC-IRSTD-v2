import pickle

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from data_ext.balanced_domain_loader import BalancedDomainLoader
from data_ext.multi_source_dataset import DomainDataset
from losses.hard_target_loss import hard_target_miss_loss, target_object_scores
from losses.local_peak_cvar import (
    domain_local_peak_tail_risks,
    local_background_peak_scores,
    stack_domain_risks,
    top_fraction_mean,
)
from losses.smooth_worst_domain import smooth_max
from main import (
    capture_rng_state,
    load_model_state_dict,
    resolve_inference_warm_flag,
    torch_load_compat,
    validate_input_size,
)
from model.loss import SLSIoULoss
from scripts.train_multisource_tail import (
    RESUME_CRITICAL_CONFIG_KEYS,
    validate_resume_config,
)
from utils.data import IRSTD_Dataset


def _logits(probability: torch.Tensor) -> torch.Tensor:
    return torch.logit(probability.clamp(1e-5, 1.0 - 1e-5)).requires_grad_()


def test_top_fraction_mean_uses_ceiling_and_largest_values():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    result = top_fraction_mean(values, 0.26)
    assert result.item() == pytest.approx(3.5)
    result.backward()
    assert values.grad.tolist() == pytest.approx([0.0, 0.0, 0.5, 0.5])


def test_local_peak_tail_is_per_domain_and_differentiable():
    probability = torch.full((2, 1, 5, 5), 0.01)
    probability[0, 0, 1, 1] = 0.9
    probability[0, 0, 4, 4] = 0.99
    probability[1, 0, 2, 3] = 0.7
    logits = _logits(probability)
    masks = torch.zeros_like(logits)
    masks[0, 0, 4, 4] = 1.0
    domain_ids = torch.tensor([0, 1], dtype=torch.long)

    risks = domain_local_peak_tail_risks(
        logits,
        masks,
        domain_ids,
        tail_fraction=0.1,
        min_score=0.05,
    )
    ids, stacked = stack_domain_risks(risks)
    assert ids == [0, 1]
    assert stacked.detach().tolist() == pytest.approx([0.9, 0.7], abs=1e-6)
    worst = smooth_max(stacked, gamma=10.0)
    assert 0.9 < worst.item() < 0.9 + np.log(2.0) / 10.0
    worst.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0.0


def test_local_peak_plateaus_are_deduplicated_and_keep_gradient():
    probability = torch.full((1, 1, 6, 6), 0.01)
    probability[0, 0, 1, 1:3] = 0.8
    probability[0, 0, 4, 4] = 0.9
    logits = _logits(probability)
    masks = torch.zeros_like(logits)

    scores = local_background_peak_scores(logits, masks, min_score=0.05)[0]

    assert scores.numel() == 2
    assert sorted(scores.detach().tolist()) == pytest.approx([0.8, 0.9])
    scores.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0.0


def test_flat_local_peak_plateau_produces_one_candidate():
    logits = torch.zeros((1, 1, 16, 16), requires_grad=True)
    masks = torch.zeros_like(logits)

    scores = local_background_peak_scores(logits, masks, min_score=0.05)[0]

    assert scores.shape == (1,)
    assert scores.item() == pytest.approx(0.5)
    scores.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0.0


@pytest.mark.parametrize("epoch", [0, 6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sls_empty_target_with_extreme_negative_logits_is_finite(epoch, dtype):
    logits = torch.full(
        (2, 1, 16, 16),
        -100.0,
        dtype=dtype,
        requires_grad=True,
    )
    masks = torch.zeros_like(logits)

    loss = SLSIoULoss()(logits, masks, warm_epoch=5, epoch=epoch)

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("epoch", [0, 6])
def test_sls_empty_target_penalizes_mean_foreground_probability(epoch):
    logits = torch.zeros((2, 1, 8, 8), requires_grad=True)
    masks = torch.zeros_like(logits)

    loss = SLSIoULoss()(logits, masks, warm_epoch=5, epoch=epoch)

    assert loss.item() == pytest.approx(0.5)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0.0


def test_hard_target_miss_selects_the_weakest_object_and_keeps_gradient():
    probability = torch.full((1, 1, 6, 6), 0.05)
    masks = torch.zeros_like(probability)
    masks[0, 0, 0:2, 0:2] = 1.0
    masks[0, 0, 4:6, 4:6] = 1.0
    probability[masks.bool()] = 0.9
    probability[0, 0, 4:6, 4:6] = 0.2
    logits = _logits(probability)

    scores = target_object_scores(logits, masks, response_fraction=1.0)
    assert sorted(scores.detach().tolist()) == pytest.approx([0.2, 0.9])
    loss = hard_target_miss_loss(
        logits,
        masks,
        miss_fraction=0.5,
        response_fraction=1.0,
    )
    assert loss.item() == pytest.approx(0.8, abs=1e-6)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0.0


def test_hard_target_miss_is_differentiable_zero_without_targets():
    logits = torch.zeros((2, 1, 4, 4), requires_grad=True)
    masks = torch.zeros_like(logits)
    loss = hard_target_miss_loss(logits, masks)
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() == 0.0


def test_balanced_loader_has_equal_domain_counts_on_every_step():
    first = TensorDataset(
        torch.randn(4, 3, 4, 4),
        torch.zeros(4, 1, 4, 4),
    )
    second = TensorDataset(
        torch.randn(6, 3, 4, 4),
        torch.zeros(6, 1, 4, 4),
    )
    loader = BalancedDomainLoader(
        [
            DomainDataset(first, 0, "first"),
            DomainDataset(second, 1, "second"),
        ],
        batch_size_per_domain=2,
        steps_per_epoch=3,
        seed=7,
    )
    assert len(loader) == 3
    assert loader.batch_size == 4
    for batch in loader:
        assert torch.bincount(batch["domain_id"], minlength=2).tolist() == [2, 2]
        assert batch["domain_name"].count("first") == 2
        assert batch["domain_name"].count("second") == 2


def test_legacy_resolver_uses_nuaa_pixels_mask_and_ignores_xml(tmp_path):
    masks = tmp_path / "masks"
    masks.mkdir()
    (masks / "Misc_1.xml").write_text("<annotation/>", encoding="utf-8")
    expected = masks / "Misc_1_pixels0.png"
    expected.write_bytes(b"not-decoded-in-this-resolver-test")
    resolved = IRSTD_Dataset._resolve_image_path(
        str(masks),
        "Misc_1",
        is_mask=True,
    )
    assert resolved == str(expected)


def test_checkpoint_helpers_cover_legacy_prefix_and_warm_head_override(tmp_path):
    source = nn.Linear(3, 2)
    target = nn.Linear(3, 2)
    prefixed = {
        "module.weight": source.weight.detach().clone(),
        "module.bias": source.bias.detach().clone(),
    }
    load_model_state_dict(target, prefixed)
    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)

    assert resolve_inference_warm_flag("auto", prefixed) is True
    assert resolve_inference_warm_flag("auto", {"warm_flag": False}) is False
    assert resolve_inference_warm_flag("false", prefixed) is False
    assert resolve_inference_warm_flag("true", {"warm_flag": False}) is True

    legacy_checkpoint = tmp_path / "legacy.pkl"
    torch.save({"net": source.state_dict(), "iou": np.float64(0.5)}, legacy_checkpoint)
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        torch_load_compat(
            legacy_checkpoint,
            map_location="cpu",
            full_checkpoint=True,
        )
    with pytest.warns(RuntimeWarning, match="trusted legacy resume"):
        loaded = torch_load_compat(
            legacy_checkpoint,
            map_location="cpu",
            full_checkpoint=True,
            allow_unsafe_legacy=True,
        )
    assert float(loaded["iou"]) == 0.5

    safe_checkpoint = tmp_path / "safe-full.pkl"
    torch.save({"net": source.state_dict(), "rng_state": capture_rng_state()}, safe_checkpoint)
    safely_loaded = torch_load_compat(
        safe_checkpoint,
        map_location="cpu",
        full_checkpoint=True,
    )
    assert safely_loaded["rng_state"]["schema_version"] == 2
    assert isinstance(safely_loaded["rng_state"]["numpy"]["state"], torch.Tensor)


def test_mshnet_input_size_must_be_a_multiple_of_sixteen():
    validate_input_size("crop_size", 256)
    with pytest.raises(ValueError, match="divisible by 16"):
        validate_input_size("crop_size", 250)


def test_tail_resume_rejects_changed_critical_config():
    config = {
        key: 1
        for key in RESUME_CRITICAL_CONFIG_KEYS
    }
    config["source_dirs"] = ["/source/a", "/source/b"]
    config["domain_names"] = ["a", "b"]
    config["steps_per_epoch"] = None
    checkpoint = {"config": dict(config)}
    validate_resume_config(config, checkpoint)

    changed = dict(config)
    changed["tail_q"] = 0.25
    with pytest.raises(ValueError, match="tail_q"):
        validate_resume_config(changed, checkpoint)

    changed = dict(config)
    changed["source_dirs"] = ["/source/b", "/source/a"]
    with pytest.raises(ValueError, match="source_dirs"):
        validate_resume_config(changed, checkpoint)
