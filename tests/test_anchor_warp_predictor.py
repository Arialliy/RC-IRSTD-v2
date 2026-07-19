from __future__ import annotations

import pytest
import torch

from risk_curve.anchor_warp_predictor import (
    ANCHOR_WARP_ARCHITECTURE_VERSION,
    ANCHOR_WARP_BETA_RADIUS,
    ANCHOR_WARP_DELTA_LIMIT,
    ANCHOR_WARP_PARAMETER_COUNT,
    ANCHOR_WARP_SEGMENTS,
    CountAllAnchorWarpRiskCurve,
)
from risk_curve.monotone_curve_predictor import (
    COMPONENT_LOG_RISK_FLOOR,
    PIXEL_LOG_RISK_FLOOR,
)


def _anchors(
    batch_size: int = 3, num_thresholds: int = 65
) -> tuple[torch.Tensor, torch.Tensor]:
    pixel_base = torch.linspace(0.0, PIXEL_LOG_RISK_FLOOR, num_thresholds)
    component_base = torch.linspace(3.0, COMPONENT_LOG_RISK_FLOOR, num_thresholds)
    # Include plateaus: Count-all log curves need only be non-increasing.
    pixel_start = num_thresholds // 3
    component_start = 2 * num_thresholds // 3
    pixel_base[pixel_start : pixel_start + 3] = pixel_base[pixel_start]
    component_base[component_start : component_start + 3] = component_base[
        component_start
    ]
    pixel = pixel_base.unsqueeze(0).repeat(batch_size, 1)
    component = component_base.unsqueeze(0).repeat(batch_size, 1)
    return pixel, component


def _model(num_thresholds: int = 65) -> CountAllAnchorWarpRiskCurve:
    torch.manual_seed(3407)
    return CountAllAnchorWarpRiskCurve(num_thresholds=num_thresholds)


def _activate_controller(model: CountAllAnchorWarpRiskCurve) -> None:
    with torch.no_grad():
        model.controller.weight.normal_(mean=0.0, std=0.02)
        model.controller.bias.normal_(mean=0.0, std=0.02)


def test_exact_parameter_budget_and_head_output_contract() -> None:
    model = _model()
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_284
    assert ANCHOR_WARP_PARAMETER_COUNT == 1_284
    assert model.controller.in_features == 119
    assert model.controller.out_features == 8
    assert model.pixel_head.projection.out_features == ANCHOR_WARP_SEGMENTS + 2
    assert model.component_head.projection.out_features == ANCHOR_WARP_SEGMENTS + 2
    assert model.config()["architecture_version"] == ANCHOR_WARP_ARCHITECTURE_VERSION
    restored = CountAllAnchorWarpRiskCurve(**model.config())
    assert restored.config() == model.config()


def test_default_zero_controller_is_identity_for_both_count_all_anchors() -> None:
    model = _model()
    assert torch.count_nonzero(model.controller.weight) == 0
    assert torch.count_nonzero(model.controller.bias) == 0
    assert torch.count_nonzero(model.pixel_head.projection.weight) > 0
    assert torch.count_nonzero(model.component_head.projection.weight) > 0

    statistics = torch.randn(3, 119)
    pixel, component = _anchors()
    parameters = model.adaptation_parameters(statistics)
    uniform = torch.linspace(0.0, 1.0, ANCHOR_WARP_SEGMENTS + 1).repeat(3, 1)
    assert torch.equal(parameters["pixel_warp_knots"], uniform)
    assert torch.equal(parameters["component_warp_knots"], uniform)
    assert torch.equal(parameters["pixel_beta"], torch.ones(3, 1))
    assert torch.equal(parameters["component_beta"], torch.ones(3, 1))
    assert torch.equal(parameters["pixel_delta"], torch.zeros(3, 1))
    assert torch.equal(parameters["component_delta"], torch.zeros(3, 1))

    predictions = model(statistics, pixel, component)
    assert torch.equal(predictions["pixel_log_risk"], pixel)
    assert torch.equal(predictions["component_log_risk"], component)


@pytest.mark.parametrize("num_thresholds", [17, 65, 2048])
def test_predictions_are_finite_non_increasing_and_above_floor(
    num_thresholds: int,
) -> None:
    model = _model(num_thresholds)
    _activate_controller(model)
    statistics = torch.randn(3, 119)
    pixel, component = _anchors(num_thresholds=num_thresholds)
    predictions = model(statistics, pixel, component)
    for key, floor in (
        ("pixel_log_risk", PIXEL_LOG_RISK_FLOOR),
        ("component_log_risk", COMPONENT_LOG_RISK_FLOOR),
    ):
        curve = predictions[key]
        assert curve.shape == (3, num_thresholds)
        assert torch.isfinite(curve).all()
        assert torch.all(curve >= floor)
        assert torch.all(torch.diff(curve, dim=1) <= 8.0 * torch.finfo(curve.dtype).eps)

    parameters = model.adaptation_parameters(statistics)
    for prefix in ("pixel", "component"):
        knots = parameters[f"{prefix}_warp_knots"]
        assert torch.equal(knots[:, 0], torch.zeros(3))
        assert torch.equal(knots[:, -1], torch.ones(3))
        assert torch.all(torch.diff(knots, dim=1) >= 0.0)


def test_bounded_controller_and_head_parameters_remain_safe_for_ood_statistics() -> None:
    model = _model()
    _activate_controller(model)
    statistics = torch.empty(4, 119)
    statistics[0].fill_(1e30)
    statistics[1].fill_(-1e30)
    statistics[2] = torch.linspace(-1e30, 1e30, 119)
    statistics[3].zero_()
    pixel, component = _anchors(batch_size=4)

    parameters = model.adaptation_parameters(statistics)
    assert torch.isfinite(parameters["controller_state"]).all()
    assert torch.all(parameters["controller_state"].abs() <= 1.0)
    for prefix in ("pixel", "component"):
        beta = parameters[f"{prefix}_beta"]
        delta = parameters[f"{prefix}_delta"]
        assert torch.all(beta >= 1.0 - ANCHOR_WARP_BETA_RADIUS)
        assert torch.all(beta <= 1.0 + ANCHOR_WARP_BETA_RADIUS)
        assert torch.all(delta >= -ANCHOR_WARP_DELTA_LIMIT)
        assert torch.all(delta <= ANCHOR_WARP_DELTA_LIMIT)
    for curve in model(statistics, pixel, component).values():
        assert torch.isfinite(curve).all()


def test_forward_backward_has_finite_gradients_for_model_statistics_and_anchors() -> None:
    model = _model()
    _activate_controller(model)
    statistics = torch.randn(3, 119, requires_grad=True)
    pixel, component = _anchors()
    pixel.requires_grad_()
    component.requires_grad_()

    predictions = model(statistics, pixel, component)
    loss = predictions["pixel_log_risk"].square().mean()
    loss = loss + predictions["component_log_risk"].square().mean()
    loss.backward()

    for tensor in (statistics, pixel, component):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_identity_initialisation_still_backpropagates_into_controller() -> None:
    model = _model(num_thresholds=33)
    statistics = torch.randn(2, 119, requires_grad=True)
    pixel, component = _anchors(batch_size=2, num_thresholds=33)
    predictions = model(statistics, pixel, component)
    loss = predictions["pixel_log_risk"].square().mean()
    loss = loss + predictions["component_log_risk"].square().mean()
    loss.backward()

    assert model.controller.weight.grad is not None
    assert model.controller.bias.grad is not None
    assert torch.any(model.controller.weight.grad != 0.0)
    assert torch.any(model.controller.bias.grad != 0.0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("input_dim", 118, "input_dim"),
        ("controller_dim", 9, "controller_dim"),
        ("num_warp_segments", 15, "num_warp_segments"),
        ("num_thresholds", 1, "num_thresholds"),
        ("pixel_log_risk_floor", -20.0, "pixel_log_risk_floor"),
        ("component_log_risk_floor", -20.0, "component_log_risk_floor"),
        ("architecture_version", "tampered-v0", "architecture_version"),
    ],
)
def test_architecture_contract_tampering_fails_closed(
    field: str, value: int | str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CountAllAnchorWarpRiskCurve(**{field: value})


def test_non_finite_state_dict_tampering_fails_closed() -> None:
    model = _model()
    statistics = torch.randn(3, 119)
    pixel, component = _anchors()
    with torch.no_grad():
        model.pixel_head.projection.weight[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        model(statistics, pixel, component)

    fresh = _model()
    tampered = dict(fresh.state_dict())
    tampered["controller.weight"] = torch.zeros(7, 119)
    with pytest.raises(RuntimeError, match="size mismatch"):
        fresh.load_state_dict(tampered, strict=True)


def test_input_shape_dtype_value_and_anchor_order_tampering_fails_closed() -> None:
    model = _model()
    statistics = torch.randn(3, 119)
    pixel, component = _anchors()

    with pytest.raises(ValueError, match="statistics"):
        model(torch.randn(3, 118), pixel, component)
    with pytest.raises(TypeError, match="floating-point"):
        model(torch.ones(3, 119, dtype=torch.int64), pixel, component)
    non_finite_statistics = statistics.clone()
    non_finite_statistics[0, 0] = float("inf")
    with pytest.raises(ValueError, match="NaN or infinite"):
        model(non_finite_statistics, pixel, component)

    with pytest.raises(ValueError, match="shape"):
        model(statistics, pixel[:2], component)
    with pytest.raises(ValueError, match="shape"):
        model(statistics, pixel[:, :-1], component)
    with pytest.raises(ValueError, match="same device and dtype"):
        model(statistics, pixel.double(), component)

    non_finite_anchor = pixel.clone()
    non_finite_anchor[0, 4] = float("nan")
    with pytest.raises(ValueError, match="NaN or infinite"):
        model(statistics, non_finite_anchor, component)
    below_floor = pixel.clone()
    below_floor[0, -1] = PIXEL_LOG_RISK_FLOOR - 0.1
    with pytest.raises(ValueError, match="physical floor"):
        model(statistics, below_floor, component)
    non_monotone = component.clone()
    non_monotone[0, 20] = non_monotone[0, 19] + 0.1
    with pytest.raises(ValueError, match="non-increasing"):
        model(statistics, pixel, non_monotone)
