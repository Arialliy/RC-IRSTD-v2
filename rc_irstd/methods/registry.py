"""Canonical registry for proposed and baseline deployment methods."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

from torch import nn

from rc_irstd.models.calibrator import MonotoneBudgetCalibrator
from risk_curve.monotone_curve_predictor import RiskCurvePredictor


RISK_CURVE_METHOD_NAME = "risk_curve"
DIRECT_THRESHOLD_METHOD_NAME = "direct_threshold"
DEFAULT_METHOD_NAME = RISK_CURVE_METHOD_NAME

MethodRole = Literal["proposed_method", "baseline"]


@dataclass(frozen=True)
class MethodSpec:
    """Immutable method identity saved alongside experimental artifacts."""

    name: str
    model_class: type[nn.Module]
    display_name: str
    role: MethodRole
    output_contract: str

    @property
    def is_proposed_method(self) -> bool:
        return self.role == "proposed_method"

    def metadata(self) -> dict[str, str | bool]:
        """Return JSON-serializable identity metadata for checkpoints/logs."""

        return {
            "method_name": self.name,
            "model_class": self.model_class.__name__,
            "canonical_import": (
                f"{self.model_class.__module__}.{self.model_class.__qualname__}"
            ),
            "display_name": self.display_name,
            "role": self.role,
            "is_proposed_method": self.is_proposed_method,
            "output_contract": self.output_contract,
        }


_METHOD_SPECS = {
    RISK_CURVE_METHOD_NAME: MethodSpec(
        name=RISK_CURVE_METHOD_NAME,
        model_class=RiskCurvePredictor,
        display_name="RC-IRSTD-v2",
        role="proposed_method",
        output_contract="dual_monotone_log_risk_curves",
    ),
    DIRECT_THRESHOLD_METHOD_NAME: MethodSpec(
        name=DIRECT_THRESHOLD_METHOD_NAME,
        model_class=MonotoneBudgetCalibrator,
        display_name="RC-Direct",
        role="baseline",
        output_contract="ordered_budget_thresholds",
    ),
}

# Public mappings are read-only: experiment code may inspect, but not mutate,
# the method identity halfway through a run.
METHOD_SPECS: Mapping[str, MethodSpec] = MappingProxyType(_METHOD_SPECS)
METHOD_REGISTRY: Mapping[str, type[nn.Module]] = MappingProxyType(
    {name: spec.model_class for name, spec in _METHOD_SPECS.items()}
)


def resolve_method_name(method_name: str | None = None) -> str:
    """Resolve ``None`` to the paper method and validate explicit names."""

    resolved = DEFAULT_METHOD_NAME if method_name is None else method_name
    if not isinstance(resolved, str) or not resolved:
        raise TypeError("method_name must be a non-empty string or None")
    if resolved not in METHOD_SPECS:
        available = ", ".join(sorted(METHOD_SPECS))
        raise ValueError(f"Unknown method {resolved!r}; available methods: {available}")
    return resolved


def get_method_spec(method_name: str | None = None) -> MethodSpec:
    return METHOD_SPECS[resolve_method_name(method_name)]


def get_method_class(method_name: str | None = None) -> type[nn.Module]:
    return get_method_spec(method_name).model_class


def get_method_metadata(method_name: str | None = None) -> dict[str, str | bool]:
    return get_method_spec(method_name).metadata()


def build_method(method_name: str | None = None, **kwargs: Any) -> nn.Module:
    """Build a registered method; omitted name means ``risk_curve``."""

    return get_method_class(method_name)(**kwargs)


def assert_main_method(method_name: str | None, model: nn.Module) -> RiskCurvePredictor:
    """Fail loudly if a formal main-method run is wired to a baseline."""

    resolved = resolve_method_name(method_name)
    if resolved != RISK_CURVE_METHOD_NAME:
        raise ValueError(
            "Formal main-method runs must use 'risk_curve'; "
            f"received {resolved!r}, whose registered role is "
            f"{get_method_spec(resolved).role!r}"
        )
    if not isinstance(model, RiskCurvePredictor):
        raise TypeError(
            "The 'risk_curve' main method must be the canonical "
            "risk_curve.monotone_curve_predictor.RiskCurvePredictor; "
            f"received {type(model).__module__}.{type(model).__qualname__}"
        )
    return model


__all__ = [
    "DEFAULT_METHOD_NAME",
    "DIRECT_THRESHOLD_METHOD_NAME",
    "METHOD_REGISTRY",
    "METHOD_SPECS",
    "RISK_CURVE_METHOD_NAME",
    "MethodSpec",
    "assert_main_method",
    "build_method",
    "get_method_class",
    "get_method_metadata",
    "get_method_spec",
    "resolve_method_name",
]
