"""Compact Count-all-anchor warp predictor for dual risk curves.

The model deliberately learns only a small, bounded correction to two
label-free Count-all anchor curves.  A controller maps the 119 registered
window statistics to eight bounded coordinates.  Each risk head then emits
16 horizontal-warp logits plus two vertical-calibration scalars.  The
resulting 1,284 trainable parameters do not depend on the threshold-grid
length.

For a non-increasing anchor ``a`` with physical floor ``f``, a head predicts
an endpoint-fixed, monotone piecewise-linear warp ``w`` and applies

    h = a(w(x)) - f
    y = f + exp(delta) * h * (1 + h) ** (beta - 1),

where ``beta`` is in ``[0.5, 1.5]`` and ``delta`` is in ``[-1, 1]``.  The
vertical map is increasing for every admissible ``beta`` and therefore
preserves monotonicity and the floor.  Zero controller output gives uniform
warp increments, ``beta=1`` and ``delta=0``, hence exactly the anchor (up to
floating-point interpolation round-off).
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn

from .monotone_curve_predictor import (
    COMPONENT_LOG_RISK_FLOOR,
    PIXEL_LOG_RISK_FLOOR,
)


ANCHOR_WARP_ARCHITECTURE_VERSION: Final = "count-all-anchor-warp-v1"
ANCHOR_WARP_INPUT_DIM: Final = 119
ANCHOR_WARP_CONTROLLER_DIM: Final = 8
ANCHOR_WARP_SEGMENTS: Final = 16
ANCHOR_WARP_BETA_RADIUS: Final = 0.5
ANCHOR_WARP_DELTA_LIMIT: Final = 1.0
ANCHOR_WARP_STATISTICS_CLIP: Final = 32.0
ANCHOR_WARP_PARAMETER_COUNT: Final = 1_284


class _AnchorWarpHead(nn.Module):
    """One 18-output bounded warp/calibration head."""

    def __init__(self, controller_dim: int, num_segments: int, floor: float) -> None:
        super().__init__()
        self.controller_dim = int(controller_dim)
        self.num_segments = int(num_segments)
        self.floor = float(floor)
        self.projection = nn.Linear(controller_dim, num_segments + 2)
        # A zero controller state must be a verifiable identity policy even
        # though the head weights remain trainable at ordinary states.
        nn.init.zeros_(self.projection.bias)

    def decode(self, controller_state: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.projection(controller_state)
        if not bool(torch.isfinite(raw).all()):
            raise RuntimeError("Anchor-warp head produced non-finite parameters")

        increment_logits = raw[:, : self.num_segments]
        increments = torch.softmax(increment_logits, dim=1)
        cumulative = torch.cumsum(increments, dim=1)
        # Construct the endpoints explicitly.  In particular, the right
        # endpoint is exactly one even if a low-precision sum is a few ulps off.
        zero = torch.zeros_like(cumulative[:, :1])
        one = torch.ones_like(cumulative[:, :1])
        warp_knots = torch.cat([zero, cumulative[:, :-1], one], dim=1)

        beta = 1.0 + ANCHOR_WARP_BETA_RADIUS * torch.tanh(
            raw[:, self.num_segments : self.num_segments + 1]
        )
        delta = ANCHOR_WARP_DELTA_LIMIT * torch.tanh(
            raw[:, self.num_segments + 1 :]
        )
        return {
            "warp_knots": warp_knots,
            "beta": beta,
            "delta": delta,
        }

    def warp_and_calibrate(
        self,
        anchor: torch.Tensor,
        parameters: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, num_thresholds = anchor.shape
        del batch_size
        dtype = anchor.dtype
        device = anchor.device

        # Interpolate deviations from the identity warp.  With uniform
        # increments all deviations are exactly zero for K=16, so integer base
        # positions are retained and the anchor is recovered point-for-point.
        base_positions = torch.arange(num_thresholds, dtype=dtype, device=device)
        uniform_knots = torch.arange(
            self.num_segments + 1, dtype=dtype, device=device
        ) / float(self.num_segments)
        deviations = parameters["warp_knots"] - uniform_knots.unsqueeze(0)

        segment_coordinate = (
            base_positions * float(self.num_segments) / float(num_thresholds - 1)
        )
        segment_index = torch.floor(segment_coordinate).to(torch.long)
        segment_index = segment_index.clamp(max=self.num_segments - 1)
        local_coordinate = segment_coordinate - segment_index.to(dtype)
        left_deviation = deviations[:, segment_index]
        right_deviation = deviations[:, segment_index + 1]
        interpolated_deviation = left_deviation + local_coordinate.unsqueeze(0) * (
            right_deviation - left_deviation
        )
        warped_positions = base_positions.unsqueeze(0) + interpolated_deviation * float(
            num_thresholds - 1
        )
        warped_positions = warped_positions.clamp(0.0, float(num_thresholds - 1))

        left_index = torch.floor(warped_positions).to(torch.long)
        right_index = (left_index + 1).clamp(max=num_thresholds - 1)
        interpolation_weight = warped_positions - left_index.to(dtype)
        left_value = torch.gather(anchor, 1, left_index)
        right_value = torch.gather(anchor, 1, right_index)
        warped_anchor = left_value + interpolation_weight * (right_value - left_value)

        headroom = warped_anchor - self.floor
        log_scale = parameters["delta"] + (parameters["beta"] - 1.0) * torch.log1p(
            headroom
        )
        # Residual form makes the identity case pointwise exact when the warp
        # positions are integers, rather than subtracting and re-adding floor.
        calibrated = warped_anchor + headroom * torch.expm1(log_scale)
        if not bool(torch.isfinite(calibrated).all()):
            raise RuntimeError("Anchor-warp calibration produced non-finite risks")
        if bool(torch.any(calibrated < self.floor)):
            raise RuntimeError("Anchor-warp calibration violated its physical floor")
        # This is an invariant check, not a repair: the horizontal and vertical
        # maps are both monotone by construction.
        tolerance = 8.0 * torch.finfo(dtype).eps
        if bool(torch.any(torch.diff(calibrated, dim=1) > tolerance)):
            raise RuntimeError("Anchor-warp calibration violated monotonicity")
        return calibrated


class CountAllAnchorWarpRiskCurve(nn.Module):
    """Predict dual risks by bounded warping of two Count-all anchor curves.

    Parameters other than ``num_thresholds`` are frozen architecture-contract
    fields.  Rejecting altered dimensions prevents a checkpoint/config edit
    from silently changing the registered 1,284-parameter model.
    """

    def __init__(
        self,
        input_dim: int = ANCHOR_WARP_INPUT_DIM,
        num_thresholds: int = 2048,
        controller_dim: int = ANCHOR_WARP_CONTROLLER_DIM,
        num_warp_segments: int = ANCHOR_WARP_SEGMENTS,
        pixel_log_risk_floor: float = PIXEL_LOG_RISK_FLOOR,
        component_log_risk_floor: float = COMPONENT_LOG_RISK_FLOOR,
        architecture_version: str = ANCHOR_WARP_ARCHITECTURE_VERSION,
    ) -> None:
        super().__init__()
        if int(input_dim) != ANCHOR_WARP_INPUT_DIM:
            raise ValueError(f"input_dim must be {ANCHOR_WARP_INPUT_DIM}")
        if int(controller_dim) != ANCHOR_WARP_CONTROLLER_DIM:
            raise ValueError(f"controller_dim must be {ANCHOR_WARP_CONTROLLER_DIM}")
        if int(num_warp_segments) != ANCHOR_WARP_SEGMENTS:
            raise ValueError(f"num_warp_segments must be {ANCHOR_WARP_SEGMENTS}")
        if int(num_thresholds) < 2:
            raise ValueError("num_thresholds must be at least 2")
        if architecture_version != ANCHOR_WARP_ARCHITECTURE_VERSION:
            raise ValueError(
                "Unsupported anchor-warp architecture_version "
                f"{architecture_version!r}; expected "
                f"{ANCHOR_WARP_ARCHITECTURE_VERSION!r}"
            )
        for name, value, required in (
            ("pixel_log_risk_floor", pixel_log_risk_floor, PIXEL_LOG_RISK_FLOOR),
            (
                "component_log_risk_floor",
                component_log_risk_floor,
                COMPONENT_LOG_RISK_FLOOR,
            ),
        ):
            scalar = torch.tensor(float(value))
            if not bool(torch.isfinite(scalar)):
                raise ValueError(f"{name} must be finite")
            if float(value) != float(required):
                raise ValueError(f"{name} must be {float(required)}")

        self.input_dim = ANCHOR_WARP_INPUT_DIM
        self.num_thresholds = int(num_thresholds)
        self.controller_dim = ANCHOR_WARP_CONTROLLER_DIM
        self.num_warp_segments = ANCHOR_WARP_SEGMENTS
        self.pixel_log_risk_floor = float(pixel_log_risk_floor)
        self.component_log_risk_floor = float(component_log_risk_floor)
        self.architecture_version = architecture_version

        self.controller = nn.Linear(self.input_dim, self.controller_dim)
        # Start from the trusted Count-all estimator for every window.  The
        # head weights intentionally retain their non-zero initialisation, so
        # the first backward pass still sends a learning signal into this
        # zero-initialised controller; only the head biases are zero below.
        nn.init.zeros_(self.controller.weight)
        nn.init.zeros_(self.controller.bias)
        self.pixel_head = _AnchorWarpHead(
            self.controller_dim,
            self.num_warp_segments,
            self.pixel_log_risk_floor,
        )
        self.component_head = _AnchorWarpHead(
            self.controller_dim,
            self.num_warp_segments,
            self.component_log_risk_floor,
        )

        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if parameter_count != ANCHOR_WARP_PARAMETER_COUNT:
            raise RuntimeError(
                "Anchor-warp architecture parameter count changed: "
                f"{parameter_count} != {ANCHOR_WARP_PARAMETER_COUNT}"
            )

    def _validate_statistics(self, statistics: torch.Tensor) -> None:
        if not isinstance(statistics, torch.Tensor):
            raise TypeError("statistics must be a torch.Tensor")
        if statistics.ndim != 2 or statistics.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected statistics [N, {self.input_dim}], got "
                f"{tuple(statistics.shape)}"
            )
        if not statistics.is_floating_point():
            raise TypeError("statistics must have a floating-point dtype")
        if not bool(torch.isfinite(statistics).all()):
            raise ValueError("statistics contain NaN or infinite values")

        reference = self.controller.weight
        if statistics.device != reference.device or statistics.dtype != reference.dtype:
            raise ValueError(
                "statistics must have the same device and dtype as the model"
            )

    def _validate_anchor(
        self,
        name: str,
        anchor: torch.Tensor,
        statistics: torch.Tensor,
        floor: float,
    ) -> None:
        if not isinstance(anchor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        expected = (statistics.shape[0], self.num_thresholds)
        if anchor.ndim != 2 or tuple(anchor.shape) != expected:
            raise ValueError(f"Expected {name} shape {expected}, got {tuple(anchor.shape)}")
        if not anchor.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype")
        if anchor.device != statistics.device or anchor.dtype != statistics.dtype:
            raise ValueError(
                f"{name} must have the same device and dtype as statistics"
            )
        if not bool(torch.isfinite(anchor).all()):
            raise ValueError(f"{name} contains NaN or infinite values")
        if bool(torch.any(anchor < floor)):
            raise ValueError(f"{name} falls below its physical floor {floor}")
        if bool(torch.any(torch.diff(anchor, dim=1) > 0.0)):
            raise ValueError(f"{name} must be non-increasing")

    def _validate_parameters(self) -> None:
        for name, parameter in self.named_parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise RuntimeError(f"Model parameter {name!r} is non-finite")

    def adaptation_parameters(
        self, statistics: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return auditable bounded warp knots, beta and delta for both heads."""

        self._validate_statistics(statistics)
        self._validate_parameters()
        bounded_statistics = statistics.clamp(
            -ANCHOR_WARP_STATISTICS_CLIP, ANCHOR_WARP_STATISTICS_CLIP
        )
        controller_state = torch.tanh(self.controller(bounded_statistics))
        if not bool(torch.isfinite(controller_state).all()):
            raise RuntimeError("Anchor-warp controller produced a non-finite state")
        pixel = self.pixel_head.decode(controller_state)
        component = self.component_head.decode(controller_state)
        return {
            "controller_state": controller_state,
            "pixel_warp_knots": pixel["warp_knots"],
            "pixel_beta": pixel["beta"],
            "pixel_delta": pixel["delta"],
            "component_warp_knots": component["warp_knots"],
            "component_beta": component["beta"],
            "component_delta": component["delta"],
        }

    def forward(
        self,
        statistics: torch.Tensor,
        pixel_anchor_log_curve: torch.Tensor,
        component_anchor_log_curve: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._validate_statistics(statistics)
        self._validate_anchor(
            "pixel_anchor_log_curve",
            pixel_anchor_log_curve,
            statistics,
            self.pixel_log_risk_floor,
        )
        self._validate_anchor(
            "component_anchor_log_curve",
            component_anchor_log_curve,
            statistics,
            self.component_log_risk_floor,
        )
        parameters = self.adaptation_parameters(statistics)
        pixel_parameters = {
            "warp_knots": parameters["pixel_warp_knots"],
            "beta": parameters["pixel_beta"],
            "delta": parameters["pixel_delta"],
        }
        component_parameters = {
            "warp_knots": parameters["component_warp_knots"],
            "beta": parameters["component_beta"],
            "delta": parameters["component_delta"],
        }
        return {
            "pixel_log_risk": self.pixel_head.warp_and_calibrate(
                pixel_anchor_log_curve, pixel_parameters
            ),
            "component_log_risk": self.component_head.warp_and_calibrate(
                component_anchor_log_curve, component_parameters
            ),
        }

    def config(self) -> dict[str, int | float | str]:
        return {
            "input_dim": self.input_dim,
            "num_thresholds": self.num_thresholds,
            "controller_dim": self.controller_dim,
            "num_warp_segments": self.num_warp_segments,
            "pixel_log_risk_floor": self.pixel_log_risk_floor,
            "component_log_risk_floor": self.component_log_risk_floor,
            "architecture_version": self.architecture_version,
        }


__all__ = [
    "ANCHOR_WARP_ARCHITECTURE_VERSION",
    "ANCHOR_WARP_BETA_RADIUS",
    "ANCHOR_WARP_CONTROLLER_DIM",
    "ANCHOR_WARP_DELTA_LIMIT",
    "ANCHOR_WARP_INPUT_DIM",
    "ANCHOR_WARP_PARAMETER_COUNT",
    "ANCHOR_WARP_SEGMENTS",
    "CountAllAnchorWarpRiskCurve",
]
