"""Single shared preprocessing and inference path for AnchorWarp policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from .anchor_warp_predictor import CountAllAnchorWarpRiskCurve
from .anchor_warp_training import RobustMedianMADScaler
from .bind_anchor_warp_validation_v4 import (
    ANCHOR_WARP_BOUND_PACKAGE_SCHEMA_VERSION,
    validate_anchor_warp_bound_package,
)
from .count_all_anchor import (
    derive_anchor_log_curves,
    validate_count_all_anchor_archive,
)
from .domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    feature_schema_sha256,
    statistics_names_sha256,
    validate_statistics_names,
)
from .representation import (
    LOGIT_REPRESENTATION,
    logit_threshold_grid_sha256,
    validate_logit_threshold_grid,
)
from .train_anchor_warp_predictor_v4 import (
    validate_anchor_warp_train_only_checkpoint,
)


@dataclass(frozen=True)
class AnchorWarpPolicy:
    checkpoint: Mapping[str, Any]
    model: CountAllAnchorWarpRiskCurve
    scaler: RobustMedianMADScaler
    pixel_oof_inflation: float
    component_oof_inflation: float
    policy_semantic_sha256: str
    validation_binding: Mapping[str, Any] | None
    device: torch.device


@dataclass(frozen=True)
class AnchorWarpInputs:
    statistics: np.ndarray
    normalized_statistics: np.ndarray
    pixel_anchor: np.ndarray
    component_anchor: np.ndarray
    anchor_semantic_sha256: str
    preprocessing_audit: Mapping[str, Any]


def load_anchor_warp_policy(
    payload: Mapping[str, Any], *, device: str | torch.device = "cpu"
) -> AnchorWarpPolicy:
    if not isinstance(payload, Mapping):
        raise TypeError("AnchorWarp policy payload must be a mapping")
    binding: Mapping[str, Any] | None
    if payload.get("package_schema_version") == ANCHOR_WARP_BOUND_PACKAGE_SCHEMA_VERSION:
        validate_anchor_warp_bound_package(payload)
        checkpoint = payload["frozen_checkpoint"]
        binding = payload["validation_binding"]
    else:
        validate_anchor_warp_train_only_checkpoint(payload)
        checkpoint = payload
        binding = None
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = CountAllAnchorWarpRiskCurve(**dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(torch_device).eval()
    scaler = RobustMedianMADScaler.from_payload(checkpoint["preprocessing"])
    return AnchorWarpPolicy(
        checkpoint=checkpoint,
        model=model,
        scaler=scaler,
        pixel_oof_inflation=float(checkpoint["pixel_oof_inflation"]),
        component_oof_inflation=float(checkpoint["component_oof_inflation"]),
        policy_semantic_sha256=str(checkpoint["policy_semantic_sha256"]),
        validation_binding=binding,
        device=torch_device,
    )


def prepare_anchor_warp_inputs(
    archive: Mapping[str, np.ndarray], policy: AnchorWarpPolicy
) -> AnchorWarpInputs:
    if not isinstance(archive, Mapping):
        raise TypeError("AnchorWarp input archive must be a mapping")
    checkpoint = policy.checkpoint
    representation = str(np.asarray(archive["representation"]).item())
    if representation != LOGIT_REPRESENTATION:
        raise ValueError("AnchorWarp inference requires raw-logit inputs")
    thresholds = validate_logit_threshold_grid(
        np.asarray(archive["thresholds"], dtype=np.float32)
    )
    grid_hash = logit_threshold_grid_sha256(thresholds)
    if grid_hash != checkpoint["threshold_grid_sha256"] or not np.array_equal(
        thresholds,
        validate_logit_threshold_grid(
            np.asarray(checkpoint["thresholds"], dtype=np.float32)
        ),
    ):
        raise ValueError("AnchorWarp archive/checkpoint threshold-grid mismatch")
    names = validate_statistics_names(archive["statistics_names"], expected_dim=119)
    if str(np.asarray(archive["statistics_schema_version"]).item()) != (
        LOGIT_STATISTICS_SCHEMA_VERSION
    ):
        raise ValueError("AnchorWarp archive statistics schema mismatch")
    if str(np.asarray(archive["statistics_names_sha256"]).item()) != (
        statistics_names_sha256(names)
    ):
        raise ValueError("AnchorWarp archive statistics-name digest mismatch")
    feature_hash = feature_schema_sha256(
        LOGIT_STATISTICS_SCHEMA_VERSION, statistics_names=names
    )
    if (
        str(np.asarray(archive["feature_schema_sha256"]).item()) != feature_hash
        or checkpoint["feature_schema_sha256"] != feature_hash
        or tuple(checkpoint["statistics_names"]) != tuple(names)
    ):
        raise ValueError("AnchorWarp archive/checkpoint feature contract mismatch")
    anchors = validate_count_all_anchor_archive(
        archive, expected_grid_sha256=grid_hash
    )
    pixel_anchor, component_anchor = derive_anchor_log_curves(anchors)
    if policy.validation_binding is not None:
        expected_anchor_hash = policy.validation_binding.get(
            "validation_anchor_semantic_sha256"
        )
        if expected_anchor_hash != anchors.semantic_sha256:
            raise ValueError("AnchorWarp validation anchor semantic hash mismatch")
    statistics = np.asarray(archive["statistics"], dtype=np.float32)
    if statistics.shape != (anchors.num_episodes, 119) or not np.isfinite(statistics).all():
        raise ValueError("AnchorWarp statistics shape/finiteness is invalid")
    raw_z = (statistics - policy.scaler.median[None, :]) / policy.scaler.scale[None, :]
    if not np.isfinite(raw_z).all():
        raise ValueError("AnchorWarp robust z values are non-finite")
    normalized = policy.scaler.transform(statistics)
    clipped = np.abs(raw_z) > policy.scaler.clip
    audit = {
        "preprocessing_schema_version": policy.scaler.schema_version,
        "ood_policy": "clip_and_audit_no_hard_reject",
        "hard_reject_applied": False,
        "clip": policy.scaler.clip,
        "max_abs_unclipped_robust_z": float(np.max(np.abs(raw_z))),
        "num_clipped_values": int(clipped.sum()),
        "num_rows_with_clipping": int(np.any(clipped, axis=1).sum()),
    }
    return AnchorWarpInputs(
        statistics=np.ascontiguousarray(statistics),
        normalized_statistics=np.ascontiguousarray(normalized),
        pixel_anchor=pixel_anchor,
        component_anchor=component_anchor,
        anchor_semantic_sha256=anchors.semantic_sha256,
        preprocessing_audit=audit,
    )


def predict_anchor_warp_curves(
    policy: AnchorWarpPolicy,
    inputs: AnchorWarpInputs,
    *,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows = inputs.normalized_statistics.shape[0]
    pixel_batches: list[np.ndarray] = []
    component_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, rows, int(batch_size)):
            stop = min(rows, start + int(batch_size))
            prediction = policy.model(
                torch.from_numpy(
                    np.array(
                        inputs.normalized_statistics[start:stop],
                        dtype=np.float32,
                        copy=True,
                        order="C",
                    )
                ).to(policy.device),
                torch.from_numpy(
                    np.array(
                        inputs.pixel_anchor[start:stop],
                        dtype=np.float32,
                        copy=True,
                        order="C",
                    )
                ).to(policy.device),
                torch.from_numpy(
                    np.array(
                        inputs.component_anchor[start:stop],
                        dtype=np.float32,
                        copy=True,
                        order="C",
                    )
                ).to(policy.device),
            )
            pixel_batches.append(prediction["pixel_log_risk"].cpu().numpy())
            component_batches.append(
                prediction["component_log_risk"].cpu().numpy()
            )
    pixel = np.concatenate(pixel_batches, axis=0).astype(np.float32)
    component = np.concatenate(component_batches, axis=0).astype(np.float32)
    pixel += np.float32(policy.pixel_oof_inflation)
    component += np.float32(policy.component_oof_inflation)
    for name, values in (("pixel", pixel), ("component", component)):
        if not np.isfinite(values).all() or np.any(np.diff(values, axis=1) > 1.0e-6):
            raise RuntimeError(f"AnchorWarp {name} prediction violated its curve contract")
    return {
        "pixel_log_risk": pixel,
        "component_log_risk": component,
    }


__all__ = [
    "AnchorWarpInputs",
    "AnchorWarpPolicy",
    "load_anchor_warp_policy",
    "predict_anchor_warp_curves",
    "prepare_anchor_warp_inputs",
]
