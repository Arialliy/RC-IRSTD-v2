"""Explicit method identities for the RC-IRSTD-v2 experiments.

The proposed risk-curve model and the direct-threshold baseline deliberately
live behind different registry entries so that a configuration cannot silently
turn the baseline into the paper's main method.
"""

from .registry import (
    DEFAULT_METHOD_NAME,
    DIRECT_THRESHOLD_METHOD_NAME,
    METHOD_REGISTRY,
    METHOD_SPECS,
    RISK_CURVE_METHOD_NAME,
    MethodSpec,
    assert_main_method,
    build_method,
    get_method_class,
    get_method_metadata,
    get_method_spec,
    resolve_method_name,
)

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
