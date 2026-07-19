from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from rc_irstd.methods import (
    DEFAULT_METHOD_NAME,
    METHOD_REGISTRY,
    assert_main_method,
    build_method,
    get_method_metadata,
    resolve_method_name,
)
from rc_irstd.models.calibrator import MonotoneBudgetCalibrator
from rc_irstd.models.system import RCIRSTDSystem
from risk_curve.monotone_curve_predictor import RiskCurvePredictor


def _direct_threshold(feature_dim: int = 4) -> MonotoneBudgetCalibrator:
    model = MonotoneBudgetCalibrator(
        feature_dim=feature_dim,
        budget_grid=[1e-4, 1e-5, 1e-6],
        hidden_dims=(8,),
        dropout=0.0,
    )
    model.normalizer.fit(torch.zeros(2, feature_dim))
    return model


def test_registry_defaults_to_canonical_risk_curve_predictor() -> None:
    assert DEFAULT_METHOD_NAME == "risk_curve"
    assert resolve_method_name() == "risk_curve"
    assert METHOD_REGISTRY["risk_curve"] is RiskCurvePredictor
    assert METHOD_REGISTRY["direct_threshold"] is MonotoneBudgetCalibrator

    model = build_method(input_dim=5, num_thresholds=7, hidden_dim=8, dropout=0.0)
    assert type(model) is RiskCurvePredictor
    assert assert_main_method(None, model) is model

    explicitly_named = build_method(
        method_name="risk_curve",
        input_dim=5,
        num_thresholds=7,
        hidden_dim=8,
        dropout=0.0,
    )
    assert type(explicitly_named) is RiskCurvePredictor


def test_registry_marks_direct_threshold_as_baseline_only() -> None:
    proposed = get_method_metadata("risk_curve")
    baseline = get_method_metadata("direct_threshold")

    assert proposed["role"] == "proposed_method"
    assert proposed["is_proposed_method"] is True
    assert proposed["output_contract"] == "dual_monotone_log_risk_curves"
    assert baseline == {
        "method_name": "direct_threshold",
        "model_class": "MonotoneBudgetCalibrator",
        "canonical_import": (
            "rc_irstd.models.calibrator.MonotoneBudgetCalibrator"
        ),
        "display_name": "RC-Direct",
        "role": "baseline",
        "is_proposed_method": False,
        "output_contract": "ordered_budget_thresholds",
    }

    direct = build_method(
        "direct_threshold",
        feature_dim=4,
        budget_grid=[1e-4, 1e-5],
        hidden_dims=(8,),
        dropout=0.0,
    )
    assert type(direct) is MonotoneBudgetCalibrator
    with pytest.raises(ValueError, match="must use 'risk_curve'"):
        assert_main_method("direct_threshold", direct)


def test_registry_rejects_unknown_or_wrong_main_model() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        resolve_method_name("not_a_method")
    with pytest.raises(TypeError, match="canonical"):
        assert_main_method("risk_curve", nn.Identity())


def test_system_exposes_isolated_proposed_and_baseline_slots() -> None:
    detector = nn.Identity()
    predictor = RiskCurvePredictor(
        input_dim=4,
        num_thresholds=5,
        hidden_dim=8,
        dropout=0.0,
    )
    direct = _direct_threshold()
    system = RCIRSTDSystem(
        detector,
        risk_curve_predictor=predictor,
        direct_threshold_baseline=direct,
    )

    assert system.risk_curve_predictor is predictor
    assert system.direct_threshold_baseline is direct
    assert system.calibrator is direct
    assert system.method_metadata == {
        "container_class": "RCIRSTDSystem",
        "detector_class": "Identity",
        "main_method_name": "risk_curve",
        "main_model_class": "RiskCurvePredictor",
        "baseline_method_name": "direct_threshold",
        "baseline_model_class": "MonotoneBudgetCalibrator",
    }
    assert "risk_curve_predictor.encoder.1.weight" in system.state_dict()
    assert "direct_threshold_baseline.encoder.0.weight" in system.state_dict()
    assert not any(key.startswith("calibrator.") for key in system.state_dict())


def test_system_preserves_legacy_calibrator_constructor_and_calibrate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = _direct_threshold()
    # Positional (detector, calibrator) is the pre-registry constructor contract.
    system = RCIRSTDSystem(nn.Identity(), direct)
    monkeypatch.setattr(
        "rc_irstd.models.system.extract_window_features",
        lambda support_items, feature_spec: np.zeros(4, dtype=np.float32),
    )

    prediction = system.calibrate(
        [object()],
        budgets=[1e-4, 1e-5, 1e-6],
        device="cpu",
    )
    assert system.calibrator is direct
    assert prediction.budgets.tolist() == pytest.approx([1e-4, 1e-5, 1e-6])
    assert prediction.thresholds.shape == (3,)

    replacement = _direct_threshold()
    system.calibrator = replacement
    assert system.calibrator is replacement
    assert system.direct_threshold_baseline is replacement
    assert "calibrator" not in system._modules


def test_system_rejects_ambiguous_alias_wiring() -> None:
    first = _direct_threshold()
    second = _direct_threshold()
    with pytest.raises(ValueError, match="are aliases"):
        RCIRSTDSystem(
            nn.Identity(),
            calibrator=first,
            direct_threshold_baseline=second,
        )
    with pytest.raises(TypeError, match="canonical RiskCurvePredictor"):
        RCIRSTDSystem(
            nn.Identity(),
            risk_curve_predictor=first,
            direct_threshold_baseline=first,
        )


def test_system_proposed_slot_accepts_only_canonical_predictor_or_subclass() -> None:
    class DerivedRiskCurvePredictor(RiskCurvePredictor):
        pass

    derived = DerivedRiskCurvePredictor(
        input_dim=4,
        num_thresholds=5,
        hidden_dim=8,
        dropout=0.0,
    )
    system = RCIRSTDSystem(nn.Identity(), risk_curve_predictor=derived)
    assert system.risk_curve_predictor is derived
    assert system.method_metadata["main_method_name"] == "risk_curve"
    assert system.method_metadata["main_model_class"] == (
        "DerivedRiskCurvePredictor"
    )

    with pytest.raises(TypeError, match="canonical RiskCurvePredictor"):
        RCIRSTDSystem(nn.Identity(), risk_curve_predictor=nn.Identity())
    with pytest.raises(TypeError, match="canonical RiskCurvePredictor"):
        RCIRSTDSystem(
            nn.Identity(), risk_curve_predictor=_direct_threshold()
        )

    empty = RCIRSTDSystem(nn.Identity())
    with pytest.raises(TypeError, match="canonical RiskCurvePredictor"):
        empty.risk_curve_predictor = nn.Identity()
    assert empty.risk_curve_predictor is None
    assert empty.method_metadata["main_method_name"] is None

    # Metadata is also fail-closed if someone bypasses ``__setattr__`` and
    # tampers with torch's internal module registry directly.
    system._modules["risk_curve_predictor"] = nn.Identity()
    with pytest.raises(RuntimeError, match="invariant violated"):
        _ = system.method_metadata


def test_system_without_direct_baseline_fails_calibrate_explicitly() -> None:
    system = RCIRSTDSystem(
        nn.Identity(),
        risk_curve_predictor=RiskCurvePredictor(
            input_dim=4,
            num_thresholds=5,
            hidden_dim=8,
            dropout=0.0,
        ),
    )
    with pytest.raises(RuntimeError, match="direct_threshold_baseline"):
        system.calibrate([object()], budgets=[1e-4], device="cpu")
