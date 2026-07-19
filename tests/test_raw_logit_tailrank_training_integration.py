from __future__ import annotations

import copy

import pytest
import torch

from rc_irstd.losses.detector import DetectorObjective
from rc_irstd.losses.tail_rank import (
    RAW_LOGIT_TAILRANK_LAMBDA_MARGIN,
    RAW_LOGIT_TAILRANK_LAMBDA_MISS,
    RAW_LOGIT_TAILRANK_LAMBDA_TAIL,
    RAW_LOGIT_TAILRANK_MARGIN,
    RAW_LOGIT_TAILRANK_MODE,
    compute_raw_logit_tailrank_margin,
)
from rc_irstd.models import build_mshnet, forward_mshnet
from rc_irstd.training.detector_trainer import DetectorTrainer


CANDIDATE_LOSS = {
    "tail_mode": RAW_LOGIT_TAILRANK_MODE,
    "lambda_tail": RAW_LOGIT_TAILRANK_LAMBDA_TAIL,
    "lambda_miss": RAW_LOGIT_TAILRANK_LAMBDA_MISS,
    "lambda_margin": RAW_LOGIT_TAILRANK_LAMBDA_MARGIN,
    "auxiliary_weight": 0.25,
    "background_fraction": 0.01,
    "miss_fraction": 0.2,
    "object_top_fraction": 0.25,
    "gamma": 10.0,
    "margin": RAW_LOGIT_TAILRANK_MARGIN,
    "min_peak_score": 0.05,
    "exclusion_radius": 2,
    "target_connectivity": 8,
    "sls_kwargs": {
        "bce_weight": 0.5,
        "iou_weight": 1.0,
        "location_weight": 0.25,
        "max_positive_weight": 50.0,
    },
}


def test_all_empty_batch_keeps_background_tail_gradient() -> None:
    logits = torch.full((2, 1, 9, 9), -10.0)
    logits[0, 0, 2, 2] = 20.0
    logits[1, 0, 6, 6] = 10.0
    logits.requires_grad_()
    target = torch.zeros_like(logits)

    result = compute_raw_logit_tailrank_margin(
        logits,
        target,
        torch.tensor([0, 1]),
        background_fraction=1.0,
        miss_fraction=1.0,
        object_top_fraction=1.0,
        margin=RAW_LOGIT_TAILRANK_MARGIN,
        exclusion_radius=2,
        target_connectivity=8,
    )

    assert result.num_target_objects == 0
    assert float(result.hard_miss.detach()) == 0.0
    assert float(result.separation_margin.detach()) == 0.0
    assert float(result.background_tail.detach()) > 19.0
    (RAW_LOGIT_TAILRANK_LAMBDA_TAIL * result.background_tail).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad[0, 0, 2, 2]) > 0.049


def test_no_eligible_background_peak_keeps_hard_target_gradient() -> None:
    logits = torch.full((1, 1, 9, 9), -10.0)
    logits[0, 0, 4, 4] = -20.0
    logits.requires_grad_()
    target = torch.zeros_like(logits)
    target[0, 0, 4, 4] = 1.0

    result = compute_raw_logit_tailrank_margin(
        logits,
        target,
        torch.tensor([0]),
        background_fraction=1.0,
        miss_fraction=1.0,
        object_top_fraction=1.0,
        margin=RAW_LOGIT_TAILRANK_MARGIN,
        exclusion_radius=2,
        target_connectivity=8,
    )

    assert result.num_background_peaks == 0
    assert float(result.background_tail.detach()) == 0.0
    assert float(result.separation_margin.detach()) == 0.0
    assert float(result.hard_miss.detach()) > 19.0
    (RAW_LOGIT_TAILRANK_LAMBDA_MISS * result.hard_miss).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad[0, 0, 4, 4]) < -0.099


def test_canonical_multiscale_candidate_backpropagates_to_every_output_head() -> None:
    torch.manual_seed(42)
    model = build_mshnet(
        {
            "backend": "canonical",
            "input_channels": 3,
            "channels": [16, 32, 64, 128, 256],
            "block_counts": [2, 2, 2, 2],
        }
    )
    objective = DetectorObjective(**copy.deepcopy(CANDIDATE_LOSS))
    images = torch.randn(2, 3, 32, 32)
    masks = torch.zeros(2, 1, 32, 32)
    # The first image is deliberately empty. The second has two targets; the
    # diagonal pair is one component under Candidate A's locked 8-connectivity.
    masks[1, 0, 5, 5] = 1.0
    masks[1, 0, 6, 6] = 1.0
    masks[1, 0, 24:26, 24:26] = 1.0

    output = forward_mshnet(model, images, warm_flag=True)
    result = objective(output, masks, torch.tensor([0, 1]))
    assert result.metrics["num_target_objects"] == 2.0
    assert result.metrics["num_background_peaks"] > 0.0
    assert torch.isfinite(result.total)
    result.total.backward()

    parameters = dict(model.named_parameters())
    for prefix in ("output_0", "output_1", "output_2", "output_3", "final"):
        gradients = [
            parameter.grad
            for name, parameter in parameters.items()
            if name.startswith(prefix + ".")
        ]
        assert gradients and all(gradient is not None for gradient in gradients)
        assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


class _Stateful:
    def __init__(self, value: object) -> None:
        self.value = value

    def state_dict(self) -> dict[str, object]:
        return {"value": self.value}


def test_final_checkpoint_payload_binds_candidate_loss_and_source_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = object.__new__(DetectorTrainer)
    trainer.model = torch.nn.Conv2d(3, 1, kernel_size=1)
    trainer.warmup_epochs = 5
    trainer.diagnostic_test_eval = False
    trainer.diagnostic_only = False
    trainer.model_config = {
        "backend": "canonical",
        "input_channels": 3,
        "channels": [16, 32, 64, 128, 256],
        "block_counts": [2, 2, 2, 2],
    }
    trainer.optimizer = _Stateful("optimizer")
    trainer.scheduler = _Stateful("scheduler")
    trainer.scaler = _Stateful("scaler")
    trainer.source_names = ["NUDT-SIRST", "IRSTD-1K"]
    trainer.source_split_records = [
        {"name": "NUDT-SIRST", "train_test_id_overlap": False},
        {"name": "IRSTD-1K", "train_test_id_overlap": False},
    ]
    trainer.train_batches = _Stateful("balanced")
    trainer.resume_contract = {
        "loss": copy.deepcopy(CANDIDATE_LOSS),
        "determinism_contract": {"torch_deterministic_algorithms": True},
    }
    trainer.config = {
        "loss": copy.deepcopy(CANDIDATE_LOSS),
        "data": {
            "sources": [
                {"name": "NUDT-SIRST", "path": "../datasets/NUDT-SIRST"},
                {"name": "IRSTD-1K", "path": "../datasets/IRSTD-1K"},
            ],
            "train_split": "train",
            "val_split": None,
            "diagnostic_test_eval": False,
        },
        "training": {"checkpoint_selection": "fixed_last", "epochs": 20},
        "_config_path": "must-not-be-serialized",
    }
    monkeypatch.setattr(
        "rc_irstd.training.detector_trainer.capture_rng_state",
        lambda: {"schema_version": 2},
    )

    payload = trainer._checkpoint_payload(19)

    assert payload["epoch"] == 19
    assert payload["warm_flag"] is True
    assert payload["inference_head"] == "multi_scale_fused"
    assert payload["checkpoint_selection"] == "fixed_last"
    assert payload["formal_paper_checkpoint"] is True
    assert payload["source_names"] == ["NUDT-SIRST", "IRSTD-1K"]
    assert payload["config"]["loss"] == CANDIDATE_LOSS
    assert payload["resume_contract"]["loss"] == CANDIDATE_LOSS
    assert "_config_path" not in payload["config"]
