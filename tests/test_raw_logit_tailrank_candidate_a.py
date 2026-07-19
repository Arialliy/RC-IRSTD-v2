from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from rc_irstd.losses.detector import (
    LEGACY_PROBABILITY_TAIL_MODE,
    DetectorObjective,
)
from rc_irstd.losses.tail_rank import (
    RAW_LOGIT_TAILRANK_CONNECTIVITY,
    RAW_LOGIT_TAILRANK_LAMBDA_MARGIN,
    RAW_LOGIT_TAILRANK_LAMBDA_MISS,
    RAW_LOGIT_TAILRANK_LAMBDA_TAIL,
    RAW_LOGIT_TAILRANK_MARGIN,
    RAW_LOGIT_TAILRANK_MODE,
    compute_raw_logit_tailrank_margin,
    raw_logit_target_object_scores,
)
from rc_irstd.models.mshnet import MSHNetOutput


ROOT = Path(__file__).resolve().parents[1]
FULL_SOURCE_CONFIG = "stage4_full_sources_tailrank_margin_a_20ep.yaml"
FULL_SOURCE_ORDER_SHA256 = (
    "d7da61f2ad9b6ead6a7149470d99a614a2ddb4f3113e19700d392a42a8f373c4"
)
CANDIDATE_LOSS_SHA256 = (
    "3f0976641ff045c1341e2bd72150f8e198cdd13acd9bd8e05fca93e446463497"
)
CANDIDATE_CONFIGS = (
    (
        "stage4_inner_nudt_from_irstd_tailrank_margin_a_20ep.yaml",
        "IRSTD-1K",
        "../datasets/IRSTD-1K",
        "NUDT-SIRST",
        "cuda:0",
    ),
    (
        "stage4_inner_irstd_from_nudt_tailrank_margin_a_20ep.yaml",
        "NUDT-SIRST",
        "../datasets/NUDT-SIRST",
        "IRSTD-1K",
        "cuda:1",
    ),
)


def _candidate_components(
    background_logit: float,
    target_logit: float = -20.0,
) -> tuple[torch.Tensor, object]:
    logits = torch.full((1, 1, 7, 7), -10.0, dtype=torch.float64)
    logits[0, 0, 1, 1] = float(background_logit)
    logits[0, 0, 5, 5] = float(target_logit)
    logits.requires_grad_()
    target = torch.zeros_like(logits)
    target[0, 0, 5, 5] = 1.0
    components = compute_raw_logit_tailrank_margin(
        logits,
        target,
        torch.tensor([0]),
        background_fraction=1.0,
        miss_fraction=1.0,
        object_top_fraction=1.0,
        gamma=10.0,
        margin=1.0,
        min_peak_score=0.05,
        exclusion_radius=0,
        target_connectivity=8,
    )
    return logits, components


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("background_logit", [10.0, 20.0])
def test_raw_logit_tail_retains_large_positive_logit_gradient(
    background_logit: float,
) -> None:
    logits, components = _candidate_components(background_logit)
    loss = RAW_LOGIT_TAILRANK_LAMBDA_TAIL * components.background_tail
    loss.backward()
    gradient = float(logits.grad[0, 0, 1, 1])
    # d softplus(x) / dx tends to one, so the weighted gradient tends to .05
    # rather than the near-zero sigmoid derivative of the legacy score.
    assert gradient == pytest.approx(0.05, rel=1e-4, abs=1e-6)


def test_candidate_weighted_gradient_matches_central_finite_difference() -> None:
    logits, components = _candidate_components(20.0, target_logit=-20.0)
    loss = (
        RAW_LOGIT_TAILRANK_LAMBDA_TAIL * components.background_tail
        + RAW_LOGIT_TAILRANK_LAMBDA_MISS * components.hard_miss
        + RAW_LOGIT_TAILRANK_LAMBDA_MARGIN * components.separation_margin
    )
    loss.backward()
    analytic_background = float(logits.grad[0, 0, 1, 1])
    analytic_target = float(logits.grad[0, 0, 5, 5])

    def value(background: float, target_value: float) -> float:
        _logits, result = _candidate_components(background, target_value)
        candidate_loss = (
            RAW_LOGIT_TAILRANK_LAMBDA_TAIL * result.background_tail
            + RAW_LOGIT_TAILRANK_LAMBDA_MISS * result.hard_miss
            + RAW_LOGIT_TAILRANK_LAMBDA_MARGIN * result.separation_margin
        )
        return float(candidate_loss.detach())

    epsilon = 1e-4
    numeric_background = (
        value(20.0 + epsilon, -20.0) - value(20.0 - epsilon, -20.0)
    ) / (2.0 * epsilon)
    numeric_target = (
        value(20.0, -20.0 + epsilon) - value(20.0, -20.0 - epsilon)
    ) / (2.0 * epsilon)
    assert analytic_background == pytest.approx(numeric_background, rel=1e-5)
    assert analytic_target == pytest.approx(numeric_target, rel=1e-5)
    assert analytic_background > 0.09
    assert analytic_target < -0.14


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_candidate_extreme_logits_have_finite_amp_compatible_gradients(
    dtype: torch.dtype,
) -> None:
    logits = torch.full((1, 1, 7, 7), -10.0, dtype=dtype)
    logits[0, 0, 1, 1] = 20.0
    logits[0, 0, 5, 5] = -20.0
    logits.requires_grad_()
    target = torch.zeros_like(logits)
    target[0, 0, 5, 5] = 1.0
    result = compute_raw_logit_tailrank_margin(
        logits,
        target,
        torch.tensor([0]),
        background_fraction=1.0,
        miss_fraction=1.0,
        object_top_fraction=1.0,
        margin=1.0,
        exclusion_radius=0,
        target_connectivity=8,
    )
    loss = (
        RAW_LOGIT_TAILRANK_LAMBDA_TAIL * result.background_tail
        + RAW_LOGIT_TAILRANK_LAMBDA_MISS * result.hard_miss
        + RAW_LOGIT_TAILRANK_LAMBDA_MARGIN * result.separation_margin
    )
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad[0, 0, 1, 1]) > 0.09
    assert float(logits.grad[0, 0, 5, 5]) < -0.14


def test_target_components_use_explicit_eight_neighbor_semantics() -> None:
    logits = torch.zeros((1, 1, 5, 5), requires_grad=True)
    logits.data[0, 0, 1, 1] = -2.0
    logits.data[0, 0, 2, 2] = 2.0
    target = torch.zeros_like(logits)
    target[0, 0, 1, 1] = 1.0
    target[0, 0, 2, 2] = 1.0

    four_neighbor = raw_logit_target_object_scores(
        logits, target, response_fraction=1.0, connectivity=4
    )
    eight_neighbor = raw_logit_target_object_scores(
        logits, target, response_fraction=1.0, connectivity=8
    )
    assert four_neighbor.numel() == 2
    assert eight_neighbor.numel() == 1
    assert float(eight_neighbor[0].detach()) == pytest.approx(0.0)
    eight_neighbor.sum().backward()
    assert logits.grad is not None
    assert float(logits.grad[0, 0, 1, 1]) == pytest.approx(0.5)
    assert float(logits.grad[0, 0, 2, 2]) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("lambda_tail", 0.051),
        ("lambda_miss", 0.11),
        ("lambda_margin", 0.0),
        ("margin", 0.9),
        ("target_connectivity", 4),
    ],
)
def test_candidate_a_objective_contract_rejects_weight_or_semantic_drift(
    field: str,
    bad_value: float | int,
) -> None:
    kwargs: dict[str, float | int | str] = {
        "tail_mode": RAW_LOGIT_TAILRANK_MODE,
        "lambda_tail": RAW_LOGIT_TAILRANK_LAMBDA_TAIL,
        "lambda_miss": RAW_LOGIT_TAILRANK_LAMBDA_MISS,
        "lambda_margin": RAW_LOGIT_TAILRANK_LAMBDA_MARGIN,
        "margin": RAW_LOGIT_TAILRANK_MARGIN,
        "target_connectivity": RAW_LOGIT_TAILRANK_CONNECTIVITY,
    }
    kwargs[field] = bad_value
    with pytest.raises(ValueError, match=field):
        DetectorObjective(**kwargs)


@pytest.mark.parametrize(
    "filename,source_name,source_path,held_name,device",
    CANDIDATE_CONFIGS,
)
def test_candidate_a_configs_are_fixed_last_source_train_only(
    filename: str,
    source_name: str,
    source_path: str,
    held_name: str,
    device: str,
) -> None:
    path = ROOT / "configs" / filename
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["seed"] == 42
    assert config["deterministic"] is True
    assert config["device"] == device
    data = config["data"]
    assert data["sources"] == [{"name": source_name, "path": source_path}]
    assert held_name not in str(data)
    assert data["train_split"] == "train"
    assert data["val_split"] is None
    assert data["diagnostic_test_eval"] is False
    loss = config["loss"]
    assert loss["tail_mode"] == RAW_LOGIT_TAILRANK_MODE
    assert loss["lambda_tail"] == RAW_LOGIT_TAILRANK_LAMBDA_TAIL
    assert loss["lambda_miss"] == RAW_LOGIT_TAILRANK_LAMBDA_MISS
    assert loss["lambda_margin"] == RAW_LOGIT_TAILRANK_LAMBDA_MARGIN
    assert loss["margin"] == RAW_LOGIT_TAILRANK_MARGIN
    assert loss["target_connectivity"] == RAW_LOGIT_TAILRANK_CONNECTIVITY
    DetectorObjective(**loss)
    training = config["training"]
    assert training["checkpoint_selection"] == "fixed_last"
    assert training["epochs"] == 20
    assert training["resume"] is None


def test_three_checkpoint_candidate_a_config_contract_is_identical_and_complete() -> None:
    filenames = [case[0] for case in CANDIDATE_CONFIGS] + [FULL_SOURCE_CONFIG]
    configs = [
        yaml.safe_load((ROOT / "configs" / filename).read_text(encoding="utf-8"))
        for filename in filenames
    ]
    full = configs[-1]
    expected_sources = [
        {"name": "NUDT-SIRST", "path": "../datasets/NUDT-SIRST"},
        {"name": "IRSTD-1K", "path": "../datasets/IRSTD-1K"},
    ]
    assert full["seed"] == 42
    assert full["deterministic"] is True
    assert full["device"] == "cuda:2"
    assert full["output_dir"] == (
        "../outputs/stage4_full_sources_tailrank_margin_a_20ep"
    )
    assert full["data"]["sources"] == expected_sources
    # This digest locks both canonical source identities and their order.  The
    # DetectorTrainer preserves this order in source_names/source_split_records,
    # where each official train manifest receives its runtime file/order hash.
    assert _canonical_sha256(full["data"]["sources"]) == FULL_SOURCE_ORDER_SHA256
    assert full["data"]["train_split"] == "train"
    assert full["data"]["val_split"] is None
    assert full["data"]["diagnostic_test_eval"] is False
    assert full.get("diagnostic_only", False) is False

    reference_loss = configs[0]["loss"]
    assert _canonical_sha256(reference_loss) == CANDIDATE_LOSS_SHA256
    for filename, config in zip(filenames, configs):
        assert config["loss"] == reference_loss, filename
        assert _canonical_sha256(config["loss"]) == CANDIDATE_LOSS_SHA256
        assert config["training"]["checkpoint_selection"] == "fixed_last"
        assert config["training"]["epochs"] == 20
        assert config["training"]["resume"] is None
        assert config["data"]["train_split"] == "train"
        assert config["data"]["val_split"] is None
        assert config["data"]["diagnostic_test_eval"] is False
        assert "nuaa" not in json.dumps(config, ensure_ascii=False).casefold()
        DetectorObjective(**config["loss"])

    source_sets = [
        tuple(source["name"] for source in config["data"]["sources"])
        for config in configs
    ]
    assert source_sets == [
        ("IRSTD-1K",),
        ("NUDT-SIRST",),
        ("NUDT-SIRST", "IRSTD-1K"),
    ]


def test_legacy_probability_tail_default_remains_byte_for_value_equivalent() -> None:
    torch.manual_seed(17)
    logits = torch.randn(2, 1, 8, 8)
    target = torch.zeros_like(logits)
    target[0, 0, 1:3, 1:3] = 1.0
    target[1, 0, 5:7, 5:7] = 1.0
    domains = torch.tensor([0, 1])
    output = MSHNetOutput(auxiliary_logits=(), logits=logits)
    implicit = DetectorObjective()
    explicit = DetectorObjective(
        tail_mode=LEGACY_PROBABILITY_TAIL_MODE,
        target_connectivity=4,
    )
    implicit_result = implicit(output, target, domains)
    explicit_result = explicit(output, target, domains)
    assert torch.equal(implicit_result.total, explicit_result.total)
    assert implicit_result.metrics == explicit_result.metrics

    for filename in (
        "stage4_inner_nudt_from_irstd_tailmiss_20ep.yaml",
        "stage4_inner_irstd_from_nudt_tailmiss_20ep.yaml",
    ):
        config = yaml.safe_load(
            (ROOT / "configs" / filename).read_text(encoding="utf-8")
        )
        original_loss = copy.deepcopy(config["loss"])
        assert "tail_mode" not in original_loss
        assert original_loss["lambda_tail"] == 0.10
        assert original_loss["lambda_miss"] == 0.10
        assert original_loss["lambda_margin"] == 0.0
        DetectorObjective(**original_loss)
