from __future__ import annotations

import math

import pytest
import torch

from risk_curve.monotone_curve_predictor import (
    COMPONENT_LOG_RISK_FLOOR,
    INITIAL_TOTAL_DROP_FRACTION,
    PIXEL_LOG_RISK_FLOOR,
    RISK_CURVE_ARCHITECTURE_VERSION,
    MonotoneRiskHead,
    RiskCurvePredictor,
)


STRICT_PIXEL_BUDGET = 1e-6
STRICT_COMPONENT_BUDGET = 1.0


@pytest.mark.parametrize("num_thresholds", [1024, 2048])
def test_long_controlled_total_drop_curves_have_finite_forward_and_backward(
    num_thresholds: int,
) -> None:
    torch.manual_seed(17)
    model = RiskCurvePredictor(
        input_dim=7,
        num_thresholds=num_thresholds,
        hidden_dim=24,
        dropout=0.0,
    )
    statistics = torch.randn(4, 7, requires_grad=True)
    predictions = model(statistics)

    for name, floor in (
        ("pixel_log_risk", PIXEL_LOG_RISK_FLOOR),
        ("component_log_risk", COMPONENT_LOG_RISK_FLOOR),
    ):
        curve = predictions[name]
        assert curve.shape == (4, num_thresholds)
        assert torch.isfinite(curve).all()
        assert torch.all(curve >= floor)
        assert torch.all(torch.diff(curve, dim=1) <= 0.0)
        # A normal long-grid initialisation must not collapse into the repeated
        # hard-floor plateau produced by the historical clamp-based head.
        assert not torch.any(curve == floor)
        assert torch.all(curve[:, 0] > curve[:, -1])

    loss = sum(curve.square().mean() for curve in predictions.values())
    loss.backward()
    assert statistics.grad is not None
    assert torch.isfinite(statistics.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    ("risk_floor", "num_thresholds"),
    [(PIXEL_LOG_RISK_FLOOR, 1024), (COMPONENT_LOG_RISK_FLOOR, 2048)],
)
def test_total_drop_is_structurally_bounded_by_start_headroom(
    risk_floor: float,
    num_thresholds: int,
) -> None:
    head = MonotoneRiskHead(
        hidden_dim=3,
        num_thresholds=num_thresholds,
        risk_floor=risk_floor,
    )
    with torch.no_grad():
        head.start.weight.zero_()
        head.start.bias.zero_()
        head.total_drop.weight.zero_()
        head.total_drop.bias.fill_(100.0)
        head.decrements.weight.zero_()
        head.decrements.bias.zero_()

    curve = head(torch.zeros(2, 3))
    assert torch.isfinite(curve).all()
    assert torch.all(curve >= risk_floor)
    assert not torch.any(curve == risk_floor)
    assert torch.all(torch.diff(curve, dim=1) < 0.0)
    headroom = curve[:, 0] - risk_floor
    realised_drop = curve[:, 0] - curve[:, -1]
    assert torch.all(realised_drop < headroom)


def test_architecture_version_is_validated_and_round_trips_in_config() -> None:
    model = RiskCurvePredictor(
        input_dim=5,
        num_thresholds=33,
        hidden_dim=8,
        dropout=0.0,
    )
    config = model.config()
    assert config["architecture_version"] == RISK_CURVE_ARCHITECTURE_VERSION
    assert config["initial_total_drop_fraction"] == INITIAL_TOTAL_DROP_FRACTION
    restored = RiskCurvePredictor(**config)
    assert restored.config() == config

    for obsolete_architecture in (
        "controlled-total-drop-v1",
        "legacy-independent-decrements",
    ):
        with pytest.raises(ValueError, match="architecture_version"):
            RiskCurvePredictor(
                input_dim=5,
                num_thresholds=33,
                architecture_version=obsolete_architecture,
            )

    with pytest.raises(ValueError, match="initial_total_drop_fraction"):
        RiskCurvePredictor(
            input_dim=5,
            num_thresholds=33,
            initial_total_drop_fraction=0.9,
        )


def test_budget_ready_initialisation_is_strict_feasible_on_2048_point_grid() -> None:
    pixel_head = MonotoneRiskHead(
        hidden_dim=4,
        num_thresholds=2048,
        risk_floor=PIXEL_LOG_RISK_FLOOR,
    )
    component_head = MonotoneRiskHead(
        hidden_dim=4,
        num_thresholds=2048,
        risk_floor=COMPONENT_LOG_RISK_FLOOR,
    )

    hidden = torch.zeros(3, 4)
    for head, budget in (
        (pixel_head, STRICT_PIXEL_BUDGET),
        (component_head, STRICT_COMPONENT_BUDGET),
    ):
        curve = head(hidden)
        assert curve.shape == (3, 2048)
        assert torch.isfinite(curve).all()
        assert torch.all(curve >= head.risk_floor)
        assert not torch.any(curve == head.risk_floor)
        assert torch.all(torch.diff(curve, dim=1) <= 0.0)
        assert torch.all(curve[:, 0] > curve[:, -1])
        assert torch.all(curve[:, -1] <= math.log10(budget))

        raw_drop = torch.nn.functional.softplus(head.total_drop.bias)
        realised_fraction = raw_drop / (1.0 + raw_drop)
        assert float(realised_fraction.detach().item()) == pytest.approx(
            INITIAL_TOTAL_DROP_FRACTION,
            abs=1e-7,
        )
        assert torch.count_nonzero(head.total_drop.weight) == 0


@pytest.mark.parametrize(
    "risk_floor", [PIXEL_LOG_RISK_FLOOR, COMPONENT_LOG_RISK_FLOOR]
)
def test_permissive_endpoint_initialises_near_zero_risk(risk_floor: float) -> None:
    head = MonotoneRiskHead(
        hidden_dim=4,
        num_thresholds=16,
        risk_floor=risk_floor,
    )
    with torch.no_grad():
        head.start.weight.zero_()
    curve = head(torch.zeros(3, 4))
    assert torch.allclose(curve[:, 0], torch.zeros(3), atol=2e-5)


def test_checkpoint_without_total_drop_parameters_is_incompatible() -> None:
    model = RiskCurvePredictor(
        input_dim=3,
        num_thresholds=9,
        hidden_dim=4,
        dropout=0.0,
    )
    legacy_like_state = {
        name: value
        for name, value in model.state_dict().items()
        if ".total_drop." not in name
    }
    with pytest.raises(RuntimeError, match="total_drop"):
        model.load_state_dict(legacy_like_state, strict=True)
