"""Regression contracts for the AAAI-27 executable pipeline dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rc_irstd.cli.run_pipeline import (
    DIRECT_THRESHOLD_METHOD,
    FORMAL_RISK_ADAPTATION_WINDOW,
    FORMAL_RISK_EVALUATION_WINDOW,
    FORMAL_RISK_STRIDE,
    RISK_CURVE_METHOD,
    _method_name,
    _risk_window_contract,
    _static_cross_fit_statistics_arguments,
    _target_risk_mode,
    _validated_meta_split,
)


ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIGS = (
    "pipeline.yaml",
    "pipeline_outer_nuaa.yaml",
    "pipeline_outer_nudt.yaml",
    "pipeline_outer_irstd.yaml",
)


def _load(name: str) -> dict[str, object]:
    payload = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.mark.parametrize("name", FORMAL_CONFIGS)
def test_formal_configs_select_source_train_risk_curve(name: str) -> None:
    config = _load(name)
    assert config["method"]["name"] == RISK_CURVE_METHOD
    assert config["meta"]["split"] == "train"
    assert _risk_window_contract(config["meta"], diagnostic_only=False) == (
        FORMAL_RISK_ADAPTATION_WINDOW,
        FORMAL_RISK_EVALUATION_WINDOW,
        FORMAL_RISK_STRIDE,
    )
    assert config["risk_curve"]["quantile"] == pytest.approx(0.90)
    assert config["baseline"]["direct_threshold"]["enabled"] is True


@pytest.mark.parametrize(
    "name",
    ("pipeline_outer_nuaa.yaml", "pipeline_outer_nudt.yaml", "pipeline_outer_irstd.yaml"),
)
def test_outer_targets_use_static_cross_fit_and_full_test_sweep(name: str) -> None:
    config = _load(name)
    target = config["final_targets"][0]
    assert target["split"] == "test"
    assert target["evaluate"] is True
    assert _target_risk_mode(target) == "static-cross-fit"
    assert target["cross_fit_folds"] == 5

    runner = (ROOT / "rc_irstd" / "cli" / "run_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert '"risk_curve.build_deployment_statistics"' in runner
    assert '"risk_curve.select_zero_label_threshold"' in runner
    assert '"certification.build_calibration_losses"' in runner
    assert '"risk_curve.evaluate_zero_label"' in runner
    assert '"rc_irstd.cli.evaluate_static_crossfit_direct"' in runner
    assert '"evaluation.threshold_sweep"' in runner
    assert '"evaluation.standard_metrics"' in runner
    assert '"--formal", "--expected-split-role", "test"' in runner


def test_legacy_smoke_is_explicit_direct_threshold() -> None:
    config = _load("smoke_pipeline.yaml")
    assert config["diagnostic_only"] is True
    assert config["method"]["name"] == DIRECT_THRESHOLD_METHOD
    assert config["meta"]["split"] == "test"


def test_method_defaults_to_proposed_risk_curve() -> None:
    assert _method_name({}) == RISK_CURVE_METHOD


def test_risk_curve_never_accepts_source_test_even_for_diagnostics() -> None:
    with pytest.raises(ValueError, match="official train"):
        _validated_meta_split(
            {"split": "test"},
            method_name=RISK_CURVE_METHOD,
            diagnostic_only=True,
        )


def test_formal_direct_threshold_also_rejects_non_train_meta_split() -> None:
    with pytest.raises(ValueError, match="official train"):
        _validated_meta_split(
            {"split": "test"},
            method_name=DIRECT_THRESHOLD_METHOD,
            diagnostic_only=False,
        )
    assert (
        _validated_meta_split(
            {"split": "test"},
            method_name=DIRECT_THRESHOLD_METHOD,
            diagnostic_only=True,
        )
        == "test"
    )


def test_temporal_mode_is_never_the_static_default() -> None:
    assert _target_risk_mode({}) == "static-cross-fit"
    assert _target_risk_mode({"risk_curve_mode": "causal"}) == "causal"
    runner = (ROOT / "rc_irstd" / "cli" / "run_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert 'evaluate_zero_command.append("--mapped-actions-only")' in runner
    assert 'evaluate_zero_command.append("--allow-unmapped-as-reject")' not in runner


def test_static_cross_fit_command_uses_meta_adaptation_window() -> None:
    arguments = _static_cross_fit_statistics_arguments(
        {"cross_fit_folds": 5, "cross_fit_seed": 71},
        seed=42,
        adaptation_window=FORMAL_RISK_ADAPTATION_WINDOW,
    )
    assert arguments == [
        "--folds",
        "5",
        "--seed",
        "71",
        "--adaptation-window",
        "32",
    ]
    diagnostic_arguments = _static_cross_fit_statistics_arguments(
        {}, seed=42, adaptation_window=1
    )
    assert diagnostic_arguments[-2:] == ["--adaptation-window", "1"]
