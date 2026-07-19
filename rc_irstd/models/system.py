from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from rc_irstd.evaluation.score_store import ScoreItem
from rc_irstd.features.domain_statistics import FeatureSpec, extract_window_features
from risk_curve.monotone_curve_predictor import RiskCurvePredictor

from .calibrator import MonotoneBudgetCalibrator
from .mshnet import MSHNet, MSHNetOutput, forward_mshnet


@dataclass
class CalibratedPrediction:
    budgets: np.ndarray
    thresholds: np.ndarray


class RCIRSTDSystem(nn.Module):
    """Composition container for the detector and deployment-time methods.

    ``risk_curve_predictor`` is the proposed method.  The direct-threshold
    model occupies the explicitly named ``direct_threshold_baseline`` slot.
    ``calibrator`` remains a constructor/property alias for compatibility with
    earlier code, but never denotes the proposed model.
    """

    def __setattr__(self, name: str, value: object) -> None:
        # ``nn.Module.__setattr__`` registers child modules before Python's
        # property setter machinery runs.  Redirect legacy post-construction
        # assignments explicitly so no duplicate ``calibrator.*`` state-dict
        # namespace can be created.
        if name == "calibrator":
            if value is not None and not isinstance(value, MonotoneBudgetCalibrator):
                raise TypeError(
                    "calibrator must be a MonotoneBudgetCalibrator or None"
                )
            name = "direct_threshold_baseline"
        if name == "risk_curve_predictor" and value is not None and not isinstance(
            value, RiskCurvePredictor
        ):
            raise TypeError(
                "risk_curve_predictor must be the canonical RiskCurvePredictor "
                "(or a subclass) or None"
            )
        super().__setattr__(name, value)

    def __init__(
        self,
        detector: MSHNet,
        calibrator: MonotoneBudgetCalibrator | None = None,
        feature_spec: FeatureSpec | None = None,
        *,
        risk_curve_predictor: RiskCurvePredictor | None = None,
        direct_threshold_baseline: MonotoneBudgetCalibrator | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(detector, nn.Module):
            raise TypeError("detector must be a torch.nn.Module")
        if risk_curve_predictor is not None and not isinstance(
            risk_curve_predictor, RiskCurvePredictor
        ):
            raise TypeError(
                "risk_curve_predictor must be the canonical RiskCurvePredictor "
                "(or a subclass) or None"
            )
        if calibrator is not None and not isinstance(
            calibrator, MonotoneBudgetCalibrator
        ):
            raise TypeError("calibrator must be a MonotoneBudgetCalibrator or None")
        if direct_threshold_baseline is not None and not isinstance(
            direct_threshold_baseline, MonotoneBudgetCalibrator
        ):
            raise TypeError(
                "direct_threshold_baseline must be a "
                "MonotoneBudgetCalibrator or None"
            )
        if (
            calibrator is not None
            and direct_threshold_baseline is not None
            and calibrator is not direct_threshold_baseline
        ):
            raise ValueError(
                "calibrator and direct_threshold_baseline are aliases; when both "
                "are supplied they must reference the same module"
            )

        direct_baseline = (
            direct_threshold_baseline
            if direct_threshold_baseline is not None
            else calibrator
        )
        if (
            risk_curve_predictor is not None
            and direct_baseline is risk_curve_predictor
        ):
            raise ValueError(
                "risk_curve_predictor and direct_threshold_baseline must be "
                "different modules with isolated outputs/checkpoints"
            )

        self.detector = detector
        self.risk_curve_predictor = risk_curve_predictor
        self.direct_threshold_baseline = direct_baseline
        self.feature_spec = feature_spec or FeatureSpec()

    @property
    def calibrator(self) -> MonotoneBudgetCalibrator | None:
        """Legacy read alias for :attr:`direct_threshold_baseline`."""

        return self.direct_threshold_baseline

    @property
    def method_metadata(self) -> dict[str, str | None]:
        """Describe container wiring without presenting it as a new network."""

        predictor = self.risk_curve_predictor
        baseline = self.direct_threshold_baseline
        if predictor is not None and not isinstance(predictor, RiskCurvePredictor):
            # Assignment is guarded above.  Keep metadata fail-closed as well so
            # even manual corruption of ``_modules`` cannot label an arbitrary
            # module as the paper method.
            raise RuntimeError(
                "RCIRSTDSystem invariant violated: the proposed-method slot does "
                "not contain a canonical RiskCurvePredictor"
            )
        return {
            "container_class": type(self).__name__,
            "detector_class": type(self.detector).__name__,
            "main_method_name": "risk_curve" if predictor is not None else None,
            "main_model_class": type(predictor).__name__ if predictor is not None else None,
            "baseline_method_name": (
                "direct_threshold" if baseline is not None else None
            ),
            "baseline_model_class": (
                type(baseline).__name__ if baseline is not None else None
            ),
        }

    def forward(self, images: torch.Tensor, multi_scale: bool = True) -> MSHNetOutput:
        return forward_mshnet(self.detector, images, warm_flag=bool(multi_scale))

    @torch.no_grad()
    def calibrate(
        self,
        support_items: Sequence[ScoreItem],
        budgets: Sequence[float],
        device: torch.device | str | None = None,
    ) -> CalibratedPrediction:
        if not support_items:
            raise ValueError("support_items must not be empty")
        calibrator = self.direct_threshold_baseline
        if calibrator is None:
            raise RuntimeError(
                "calibrate() is available only when the direct_threshold_baseline "
                "(legacy calibrator) slot is configured"
            )
        target_device = torch.device(device) if device is not None else next(self.parameters()).device
        features = torch.from_numpy(
            extract_window_features(support_items, self.feature_spec)
        )[None].to(target_device)
        budget_tensor = torch.as_tensor(list(budgets), dtype=torch.float32, device=target_device)
        output = calibrator(features, budget_tensor)
        if output.requested_thresholds is None:
            raise RuntimeError("Calibrator did not return requested thresholds")
        return CalibratedPrediction(
            budgets=budget_tensor.detach().cpu().numpy(),
            thresholds=output.requested_thresholds[0].detach().cpu().numpy(),
        )
