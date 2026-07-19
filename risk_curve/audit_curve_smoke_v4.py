"""Audit a formal raw-logit RiskCurve smoke run against Gate B.

The audit is deliberately read-only with respect to the validation episode and
model checkpoint.  It validates the complete v4 source-only contract before it
uses the checkpoint, restores the model with ``strict=True``, and evaluates the
two pre-registered joint budgets without consulting episode labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .curve_dataset import load_curve_archive
from .curve_metrics import monotonic_violation_rate
from .direct_calibrator import ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL
from .evaluate_source_pseudo_target_v4 import (
    _archive_contract,
    _checkpoint_contract,
    _validate_episode_contract_binding,
    _validate_formal_archive,
)
from .monotone_curve_predictor import RiskCurvePredictor
from .representation import LOGIT_REPRESENTATION
from .select_zero_label_threshold import select_dual_budget_threshold
from .train_curve_predictor import (
    _archive_episode_contract,
    validate_curve_checkpoint_contract,
)


CURVE_SMOKE_AUDIT_SCHEMA_VERSION = "rc-v4-risk-curve-smoke-gate-b-v1"
FIXED_JOINT_BUDGETS: tuple[tuple[float, float], ...] = (
    (1e-5, 5.0),
    (1e-6, 1.0),
)
DEFAULT_MONOTONIC_TOLERANCE = 1e-8
DEFAULT_NONDEGENERATE_DROP_TOLERANCE = 1e-8
DEFAULT_CROSS_EPISODE_VARIANCE_TOLERANCE = 1e-12


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_device(name: str) -> torch.device:
    resolved = "cuda" if name == "auto" and torch.cuda.is_available() else name
    if resolved == "auto":
        resolved = "cpu"
    if str(resolved).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(resolved)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains the non-finite constant {value}")


def _load_json_mapping(path: Path, *, kind: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{kind} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{kind} must contain a JSON object")
    return payload


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"RiskCurve checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("RiskCurve checkpoint must contain a mapping")
    return dict(payload)


def _default_metrics_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(checkpoint_path.suffix + ".metrics.json")


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _validate_metrics_contract(
    metrics: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> None:
    expected_checkpoint_identity = {
        "method_name": "risk_curve",
        "model_class": "RiskCurvePredictor",
        "role": "proposed_method",
    }
    for field, expected in expected_checkpoint_identity.items():
        if checkpoint.get(field) != expected:
            raise ValueError(f"Checkpoint {field} must equal {expected!r}")
    if metrics.get("selection_objective") != "validation_quantile_pinball":
        raise ValueError("Metrics selection_objective is incompatible with Gate B")
    scalar_fields = (
        "method_name",
        "model_class",
        "model_architecture_version",
        "representation",
        "threshold_grid_schema_version",
        "threshold_grid_sha256",
        "feature_schema_sha256",
        "threshold_grid_manifest_sha256",
        "threshold_grid_detector_protocol",
        "threshold_grid_outer_detector_checkpoint_sha256",
    )
    for field in scalar_fields:
        if metrics.get(field) != checkpoint.get(field):
            raise ValueError(f"Metrics/checkpoint {field} mismatch")
    sequence_fields = (
        "threshold_grid_detector_checkpoint_sha256s",
        "threshold_grid_episode_detector_checkpoint_sha256s",
    )
    for field in sequence_fields:
        if tuple(metrics.get(field, [])) != tuple(checkpoint.get(field, [])):
            raise ValueError(f"Metrics/checkpoint {field} mismatch")
    if metrics.get("episode_contract") != checkpoint.get("episode_contract"):
        raise ValueError("Metrics/checkpoint episode_contract mismatch")
    for field in ("quantile", "lambda_component"):
        metric_value = _finite_number(metrics.get(field), f"metrics.{field}")
        checkpoint_value = _finite_number(
            checkpoint.get(field), f"checkpoint.{field}"
        )
        if metric_value != checkpoint_value:
            raise ValueError(f"Metrics/checkpoint {field} mismatch")


def _audit_loss_history(
    metrics: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    raw_history = metrics.get("history")
    if not isinstance(raw_history, list) or len(raw_history) < 2:
        raise ValueError("Gate B requires at least two training-history records")
    history: list[Mapping[str, Any]] = []
    seen_epochs: set[int] = set()
    all_loss_fields: set[str] = set()
    for index, raw_record in enumerate(raw_history):
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"history[{index}] must be an object")
        epoch_value = _finite_number(raw_record.get("epoch"), f"history[{index}].epoch")
        epoch = int(epoch_value)
        if epoch_value != epoch or epoch in seen_epochs:
            raise ValueError("History epochs must be distinct integers")
        seen_epochs.add(epoch)
        for field in raw_record:
            if "loss" in field or "objective" in field:
                _finite_number(raw_record[field], f"history[{index}].{field}")
                all_loss_fields.add(str(field))
        _finite_number(raw_record.get("train_loss"), f"history[{index}].train_loss")
        _finite_number(
            raw_record.get("val_objective"),
            f"history[{index}].val_objective",
        )
        history.append(raw_record)
    epoch_sequence = [int(record["epoch"]) for record in history]
    if epoch_sequence != sorted(epoch_sequence):
        raise ValueError("Training history epochs must be strictly increasing")

    train_losses = np.asarray(
        [float(record["train_loss"]) for record in history], dtype=np.float64
    )
    validation_objectives = np.asarray(
        [float(record["val_objective"]) for record in history], dtype=np.float64
    )
    # Mirror the trainer's material-improvement rule exactly.  A numerically
    # smaller objective within 1e-8 does not replace the persisted checkpoint.
    best_position = 0
    selected_best_objective = float(validation_objectives[0])
    for position, objective in enumerate(validation_objectives[1:], start=1):
        if float(objective) < selected_best_objective - 1e-8:
            selected_best_objective = float(objective)
            best_position = position
    expected_best_epoch = epoch_sequence[best_position]
    recorded_best_epoch = int(checkpoint.get("best_epoch", -1))
    if recorded_best_epoch != expected_best_epoch:
        raise ValueError("Checkpoint best_epoch does not match validation history")
    best_objective = _finite_number(metrics.get("best_objective"), "best_objective")
    if not np.isclose(
        best_objective,
        selected_best_objective,
        rtol=1e-7,
        atol=1e-12,
    ):
        raise ValueError("Metrics best_objective does not match validation history")
    validation_metrics = checkpoint.get("validation_metrics")
    if not isinstance(validation_metrics, Mapping):
        raise ValueError("Checkpoint validation_metrics must be an object")
    checkpoint_objective = _finite_number(
        validation_metrics.get("quantile_pinball_objective"),
        "checkpoint.validation_metrics.quantile_pinball_objective",
    )
    if not np.isclose(
        checkpoint_objective,
        best_objective,
        rtol=1e-7,
        atol=1e-12,
    ):
        raise ValueError("Checkpoint validation objective does not match best history")

    train_decreased = bool(train_losses[-1] < train_losses[0])
    validation_decreased = bool(
        np.min(validation_objectives[1:]) < validation_objectives[0]
    )
    return {
        "num_epochs_recorded": len(history),
        "epoch_sequence": epoch_sequence,
        "finite_loss_fields": sorted(all_loss_fields),
        "all_losses_finite": True,
        "train_loss": {
            "first": float(train_losses[0]),
            "last": float(train_losses[-1]),
            "minimum": float(train_losses.min()),
            "decreased_first_to_last": train_decreased,
        },
        "validation_objective": {
            "first": float(validation_objectives[0]),
            "last": float(validation_objectives[-1]),
            "minimum": float(validation_objectives.min()),
            "best_epoch": expected_best_epoch,
            "improved_after_first_epoch": validation_decreased,
        },
        "loss_decrease_gate_pass": train_decreased and validation_decreased,
    }


def _require_archive_checkpoint_binding(
    *,
    archive: Mapping[str, np.ndarray],
    archive_path: Path,
    archive_contract: Mapping[str, Any],
    archive_episode_contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> None:
    validated = validate_curve_checkpoint_contract(checkpoint)
    if not bool(validated.get("formal_v4_eligible", False)):
        raise ValueError("RiskCurve checkpoint is not formal v4 eligible")
    if validated.get("representation") != LOGIT_REPRESENTATION:
        raise ValueError("Gate B requires a raw-logit RiskCurve checkpoint")
    checkpoint_contract = _checkpoint_contract(checkpoint)
    for field, archive_value in archive_contract.items():
        if checkpoint_contract.get(field) != archive_value:
            raise ValueError(f"RiskCurve/archive {field} mismatch")
    if archive_contract["threshold_grid_detector_protocol"] != (
        ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL
    ):
        raise ValueError("Validation archive does not bind all source-only folds")
    _validate_episode_contract_binding(
        checkpoint,
        archive_contract,
        provenance,
        method="RiskCurve",
    )

    episode_contract = checkpoint.get("episode_contract")
    if not isinstance(episode_contract, Mapping):
        raise ValueError("RiskCurve checkpoint lacks episode_contract")
    if not bool(episode_contract.get("formal_protocol_eligible", False)):
        raise ValueError("RiskCurve episode contract is not formally eligible")
    validation_binding = episode_contract.get("validation")
    if not isinstance(validation_binding, Mapping):
        raise ValueError("RiskCurve episode contract lacks validation binding")
    actual_archive_hash = _file_sha256(archive_path)
    if validation_binding.get("archive_sha256") != actual_archive_hash:
        raise ValueError("Checkpoint validation archive SHA-256 mismatch")
    if validation_binding.get("provenance_sha256") != archive_episode_contract.get(
        "provenance_sha256"
    ):
        raise ValueError("Checkpoint validation provenance SHA-256 mismatch")
    for field in (
        "episode_schema_version",
        "representation",
        "threshold_grid_schema_version",
        "threshold_grid_sha256",
        "feature_schema_sha256",
        "threshold_grid_manifest_sha256",
        "threshold_grid_detector_protocol",
        "threshold_grid_outer_detector_checkpoint_sha256",
        "adaptation_window",
        "evaluation_window",
        "stride",
        "num_episodes",
    ):
        if validation_binding.get(field) != archive_episode_contract.get(field):
            raise ValueError(f"Checkpoint validation {field} binding mismatch")
    for field in (
        "threshold_grid_detector_checkpoint_sha256s",
        "threshold_grid_episode_detector_checkpoint_sha256s",
    ):
        if tuple(validation_binding.get(field, [])) != tuple(
            archive_episode_contract.get(field, [])
        ):
            raise ValueError(f"Checkpoint validation {field} binding mismatch")
    if int(np.asarray(archive["statistics"]).shape[0]) != int(
        validation_binding["num_episodes"]
    ):
        raise ValueError("Checkpoint validation episode count mismatch")


def _restore_and_predict(
    checkpoint: Mapping[str, Any],
    statistics: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    config = checkpoint.get("model_config")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(config, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint model_config/state_dict must be mappings")
    input_dim = int(config.get("input_dim", -1))
    num_thresholds = int(config.get("num_thresholds", -1))
    if input_dim != statistics.shape[1]:
        raise ValueError("Checkpoint input dimension differs from validation features")
    thresholds = np.asarray(checkpoint["thresholds"], dtype=np.float32)
    if num_thresholds != thresholds.size:
        raise ValueError("Checkpoint output dimension differs from threshold grid")
    mean = np.asarray(checkpoint.get("statistics_mean"), dtype=np.float32)
    std = np.asarray(checkpoint.get("statistics_std"), dtype=np.float32)
    if mean.shape != (input_dim,) or std.shape != (input_dim,):
        raise ValueError("Checkpoint normalisation dimension mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0):
        raise ValueError("Checkpoint normalisation contains invalid values")
    normalised = (statistics - mean) / np.maximum(std, 1e-6)
    if not np.isfinite(normalised).all():
        raise ValueError("Normalised validation statistics are not finite")

    def restore() -> RiskCurvePredictor:
        model = RiskCurvePredictor(**dict(config)).to(device)
        incompatible = model.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("Strict state_dict restoration returned incompatible keys")
        return model.eval()

    def predict(model: RiskCurvePredictor) -> dict[str, np.ndarray]:
        pixel: list[np.ndarray] = []
        component: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, normalised.shape[0], batch_size):
                tensor = torch.from_numpy(normalised[start : start + batch_size]).to(
                    device
                )
                output = model(tensor)
                pixel.append(output["pixel_log_risk"].detach().cpu().numpy())
                component.append(output["component_log_risk"].detach().cpu().numpy())
        return {
            "pixel_log_risk": np.concatenate(pixel, axis=0),
            "component_log_risk": np.concatenate(component, axis=0),
        }

    first = predict(restore())
    second = predict(restore())
    expected_shape = (statistics.shape[0], num_thresholds)
    for field in ("pixel_log_risk", "component_log_risk"):
        if first[field].shape != expected_shape or not np.isfinite(first[field]).all():
            raise ValueError(f"Restored model emitted invalid {field}")
    roundtrip_exact = all(
        np.array_equal(first[field], second[field])
        for field in ("pixel_log_risk", "component_log_risk")
    )
    return first, {
        "strict_load_state_dict": True,
        "forward_shape": list(expected_shape),
        "forward_predictions_finite": True,
        "independent_restore_forward_roundtrip_exact": roundtrip_exact,
    }


def _audit_curve_geometry(
    predictions: Mapping[str, np.ndarray],
    *,
    monotonic_tolerance: float,
    drop_tolerance: float,
    variance_tolerance: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind in ("pixel", "component"):
        curves = np.asarray(predictions[f"{kind}_log_risk"], dtype=np.float64)
        drops = curves[:, 0] - curves[:, -1]
        per_row_non_degenerate = drops > drop_tolerance
        cross_episode_max_variance = float(np.max(np.var(curves, axis=0)))
        result[kind] = {
            "monotonic_violation_rate": monotonic_violation_rate(
                curves, tolerance=monotonic_tolerance
            ),
            "first_to_last_drop_min": float(drops.min()),
            "first_to_last_drop_max": float(drops.max()),
            "all_rows_first_to_last_decrease": bool(
                np.all(per_row_non_degenerate)
            ),
            "non_degenerate_row_count": int(np.count_nonzero(per_row_non_degenerate)),
            "num_rows": int(curves.shape[0]),
            "cross_episode_max_variance": cross_episode_max_variance,
            "cross_episode_variation_present": bool(
                curves.shape[0] >= 2
                and cross_episode_max_variance > variance_tolerance
            ),
        }
    monotonic = all(result[kind]["monotonic_violation_rate"] == 0.0 for kind in result)
    non_degenerate = all(
        result[kind]["all_rows_first_to_last_decrease"]
        and result[kind]["cross_episode_variation_present"]
        for kind in result
    )
    result["monotonic_gate_pass"] = monotonic
    result["non_degenerate_gate_pass"] = non_degenerate
    return result


def _audit_budget_selections(
    thresholds: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    budgets: Sequence[tuple[float, float]],
) -> dict[str, Any]:
    pixel = np.asarray(predictions["pixel_log_risk"])
    component = np.asarray(predictions["component_log_risk"])
    results: list[dict[str, Any]] = []
    for pixel_budget, component_budget in budgets:
        selected_indices: list[int | None] = []
        distribution: dict[str, int] = {}
        for pixel_curve, component_curve in zip(pixel, component):
            _threshold, reject, index = select_dual_budget_threshold(
                thresholds,
                pixel_curve,
                component_curve,
                float(pixel_budget),
                float(component_budget),
                representation=LOGIT_REPRESENTATION,
            )
            selected = None if reject else int(index)  # type: ignore[arg-type]
            selected_indices.append(selected)
            key = "reject" if selected is None else str(selected)
            distribution[key] = distribution.get(key, 0) + 1
        reject_count = sum(index is None for index in selected_indices)
        finite_indices = {index for index in selected_indices if index is not None}
        unique_actions = set(selected_indices)
        results.append(
            {
                "pixel_budget": float(pixel_budget),
                "component_budget": float(component_budget),
                "selected_indices": selected_indices,
                "selected_index_distribution": dict(
                    sorted(distribution.items(), key=lambda item: item[0])
                ),
                "unique_selected_action_count": len(unique_actions),
                "unique_finite_index_count": len(finite_indices),
                "reject_count": reject_count,
                "reject_rate": reject_count / float(len(selected_indices)),
                "all_reject": reject_count == len(selected_indices),
                "not_all_reject": reject_count < len(selected_indices),
                "episode_action_variation_present": len(unique_actions) >= 2,
            }
        )
    not_all_reject = all(item["not_all_reject"] for item in results)
    variation = any(item["episode_action_variation_present"] for item in results)
    return {
        "fixed_joint_budgets": results,
        "all_budgets_not_all_reject": not_all_reject,
        "selected_index_variation_observed": variation,
        "selection_gate_pass": not_all_reject and variation,
    }


def _state_dict_semantically_equal(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    keys = sorted(set(first).union(second))
    mismatches: list[str] = []
    for key in keys:
        if key not in first or key not in second:
            mismatches.append(key)
            continue
        left, right = first[key], second[key]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            if left != right:
                mismatches.append(key)
        elif (
            left.dtype != right.dtype
            or left.shape != right.shape
            or not torch.equal(left.cpu(), right.cpu())
        ):
            mismatches.append(key)
    return not mismatches, mismatches


def _audit_one_checkpoint(
    *,
    checkpoint_path: Path,
    metrics_path: Path,
    archive: dict[str, np.ndarray],
    archive_path: Path,
    archive_contract: Mapping[str, Any],
    archive_episode_contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
    thresholds: np.ndarray,
    statistics: np.ndarray,
    names: Sequence[str],
    device: torch.device,
    batch_size: int,
    monotonic_tolerance: float,
    drop_tolerance: float,
    variance_tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    checkpoint = _load_checkpoint(checkpoint_path)
    metrics = _load_json_mapping(metrics_path, kind="RiskCurve metrics sidecar")
    _require_archive_checkpoint_binding(
        archive=archive,
        archive_path=archive_path,
        archive_contract=archive_contract,
        archive_episode_contract=archive_episode_contract,
        provenance=provenance,
        checkpoint=checkpoint,
    )
    if not np.array_equal(
        thresholds, np.asarray(checkpoint["thresholds"], dtype=np.float32)
    ):
        raise ValueError("RiskCurve/archive threshold arrays differ")
    if tuple(names) != tuple(str(item) for item in checkpoint["statistics_names"]):
        raise ValueError("RiskCurve/archive ordered feature names differ")
    _validate_metrics_contract(metrics, checkpoint)
    loss = _audit_loss_history(metrics, checkpoint)
    predictions, restoration = _restore_and_predict(
        checkpoint,
        statistics,
        device=device,
        batch_size=batch_size,
    )
    geometry = _audit_curve_geometry(
        predictions,
        monotonic_tolerance=monotonic_tolerance,
        drop_tolerance=drop_tolerance,
        variance_tolerance=variance_tolerance,
    )
    selection = _audit_budget_selections(
        thresholds, predictions, FIXED_JOINT_BUDGETS
    )
    checks = {
        "formal_raw_logit_archive_checkpoint_contract": True,
        "archive_sha256_bound_to_checkpoint": True,
        "metrics_contract_bound_to_checkpoint": True,
        "strict_checkpoint_restore": bool(restoration["strict_load_state_dict"]),
        "forward_roundtrip_exact": bool(
            restoration["independent_restore_forward_roundtrip_exact"]
        ),
        "losses_finite_and_decreased": bool(loss["loss_decrease_gate_pass"]),
        "zero_monotonic_violation_rate": bool(geometry["monotonic_gate_pass"]),
        "curves_non_degenerate": bool(geometry["non_degenerate_gate_pass"]),
        "fixed_budget_selections_not_all_reject": bool(
            selection["all_budgets_not_all_reject"]
        ),
        "selected_index_variation_observed": bool(
            selection["selected_index_variation_observed"]
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "metrics_sidecar": str(metrics_path),
        "metrics_sidecar_sha256": _file_sha256(metrics_path),
        "seed": int(checkpoint.get("seed", -1)),
        "best_epoch": int(checkpoint["best_epoch"]),
        "contract": {
            "representation": checkpoint["representation"],
            "threshold_grid_schema_version": checkpoint[
                "threshold_grid_schema_version"
            ],
            "threshold_grid_sha256": checkpoint["threshold_grid_sha256"],
            "feature_schema_sha256": checkpoint["feature_schema_sha256"],
            "threshold_grid_manifest_sha256": checkpoint[
                "threshold_grid_manifest_sha256"
            ],
            "threshold_grid_detector_protocol": checkpoint[
                "threshold_grid_detector_protocol"
            ],
            "threshold_grid_detector_checkpoint_sha256s": list(
                checkpoint["threshold_grid_detector_checkpoint_sha256s"]
            ),
            "threshold_grid_outer_detector_checkpoint_sha256": checkpoint[
                "threshold_grid_outer_detector_checkpoint_sha256"
            ],
            "threshold_grid_episode_detector_checkpoint_sha256s": list(
                checkpoint["threshold_grid_episode_detector_checkpoint_sha256s"]
            ),
        },
        "loss_history": loss,
        "checkpoint_restoration": restoration,
        "curve_geometry": geometry,
        "budget_selection": selection,
        "gate_b_checks": checks,
        "gate_b_failures": failures,
        "gate_b_pass": not failures,
    }
    return result, checkpoint, predictions, metrics


def audit_curve_smoke(
    *,
    validation_file: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    metrics_file: str | Path | None = None,
    repeat_checkpoint: str | Path | None = None,
    repeat_metrics_file: str | Path | None = None,
    device: str = "auto",
    batch_size: int = 64,
    monotonic_tolerance: float = DEFAULT_MONOTONIC_TOLERANCE,
    drop_tolerance: float = DEFAULT_NONDEGENERATE_DROP_TOLERANCE,
    variance_tolerance: float = DEFAULT_CROSS_EPISODE_VARIANCE_TOLERANCE,
) -> Path:
    """Audit one smoke checkpoint and optionally one repeated-seed checkpoint."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for value, name in (
        (monotonic_tolerance, "monotonic_tolerance"),
        (drop_tolerance, "drop_tolerance"),
        (variance_tolerance, "variance_tolerance"),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    validation_path = Path(validation_file).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    metrics_path = (
        Path(metrics_file).expanduser().resolve()
        if metrics_file is not None
        else _default_metrics_path(checkpoint_path)
    )
    torch_device = _resolve_device(device)

    archive = load_curve_archive(validation_path)
    thresholds, statistics, names, provenance = _validate_formal_archive(archive)
    archive_contract = _archive_contract(archive)
    archive_episode_contract = _archive_episode_contract(
        archive,
        archive_path=validation_path,
        split_name="validation",
    )
    if not bool(archive_episode_contract.get("formal_protocol_eligible", False)):
        raise ValueError("Validation episode archive is not formally eligible")

    primary, primary_checkpoint, primary_predictions, primary_metrics = (
        _audit_one_checkpoint(
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
            archive=archive,
            archive_path=validation_path,
            archive_contract=archive_contract,
            archive_episode_contract=archive_episode_contract,
            provenance=provenance,
            thresholds=thresholds,
            statistics=statistics,
            names=names,
            device=torch_device,
            batch_size=batch_size,
            monotonic_tolerance=monotonic_tolerance,
            drop_tolerance=drop_tolerance,
            variance_tolerance=variance_tolerance,
        )
    )

    repeat_result: dict[str, Any] | None = None
    repeat_gate_pass = True
    if repeat_checkpoint is not None:
        repeat_path = Path(repeat_checkpoint).expanduser().resolve()
        if repeat_path == checkpoint_path:
            raise ValueError("repeat_checkpoint must be a distinct file")
        repeat_metrics_path = (
            Path(repeat_metrics_file).expanduser().resolve()
            if repeat_metrics_file is not None
            else _default_metrics_path(repeat_path)
        )
        repeated, repeated_checkpoint, repeated_predictions, repeated_metrics = (
            _audit_one_checkpoint(
                checkpoint_path=repeat_path,
                metrics_path=repeat_metrics_path,
                archive=archive,
                archive_path=validation_path,
                archive_contract=archive_contract,
                archive_episode_contract=archive_episode_contract,
                provenance=provenance,
                thresholds=thresholds,
                statistics=statistics,
                names=names,
                device=torch_device,
                batch_size=batch_size,
                monotonic_tolerance=monotonic_tolerance,
                drop_tolerance=drop_tolerance,
                variance_tolerance=variance_tolerance,
            )
        )
        state_equal, state_mismatches = _state_dict_semantically_equal(
            primary_checkpoint["state_dict"], repeated_checkpoint["state_dict"]
        )
        prediction_equal = all(
            np.array_equal(primary_predictions[field], repeated_predictions[field])
            for field in ("pixel_log_risk", "component_log_risk")
        )
        semantic_checks = {
            "seed_equal": primary_checkpoint.get("seed")
            == repeated_checkpoint.get("seed"),
            "history_equal": _canonical_json(primary_metrics.get("history"))
            == _canonical_json(repeated_metrics.get("history")),
            "best_epoch_equal": primary_checkpoint.get("best_epoch")
            == repeated_checkpoint.get("best_epoch"),
            "state_tensors_equal": state_equal,
            "predictions_equal": prediction_equal,
        }
        repeat_gate_pass = bool(repeated["gate_b_pass"]) and all(
            semantic_checks.values()
        )
        repeat_result = {
            "checkpoint_audit": repeated,
            "semantic_reproducibility_checks": semantic_checks,
            "state_tensor_mismatch_keys": state_mismatches,
            "checkpoint_file_sha256_equal": primary["checkpoint_sha256"]
            == repeated["checkpoint_sha256"],
            "metrics_file_sha256_equal": primary["metrics_sidecar_sha256"]
            == repeated["metrics_sidecar_sha256"],
            "semantic_reproducibility_pass": all(semantic_checks.values()),
            "repeat_gate_pass": repeat_gate_pass,
        }
    elif repeat_metrics_file is not None:
        raise ValueError("repeat_metrics_file requires repeat_checkpoint")

    final_checks = {
        "primary_gate_b_pass": bool(primary["gate_b_pass"]),
        "optional_repeat_gate_pass": repeat_gate_pass,
    }
    payload = {
        "schema_version": CURVE_SMOKE_AUDIT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_inputs": True,
        "validation_episode": {
            "path": str(validation_path),
            "sha256": _file_sha256(validation_path),
            "num_episodes": int(statistics.shape[0]),
            "num_features": int(statistics.shape[1]),
            "num_thresholds": int(thresholds.size),
            "representation": archive_contract["representation"],
            "formal_protocol_eligible": True,
        },
        "audit_tolerances": {
            "monotonic": float(monotonic_tolerance),
            "first_to_last_drop": float(drop_tolerance),
            "cross_episode_variance": float(variance_tolerance),
        },
        "primary": primary,
        "repeat": repeat_result,
        "gate_b_checks": final_checks,
        "gate_b_failures": [
            name for name, passed in final_checks.items() if not passed
        ],
        "gate_b_pass": all(final_checks.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-file")
    parser.add_argument("--repeat-checkpoint")
    parser.add_argument("--repeat-metrics-file")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--monotonic-tolerance",
        type=float,
        default=DEFAULT_MONOTONIC_TOLERANCE,
    )
    parser.add_argument(
        "--drop-tolerance",
        type=float,
        default=DEFAULT_NONDEGENERATE_DROP_TOLERANCE,
    )
    parser.add_argument(
        "--variance-tolerance",
        type=float,
        default=DEFAULT_CROSS_EPISODE_VARIANCE_TOLERANCE,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = audit_curve_smoke(
        validation_file=args.validation_file,
        checkpoint=args.checkpoint,
        output=args.output,
        metrics_file=args.metrics_file,
        repeat_checkpoint=args.repeat_checkpoint,
        repeat_metrics_file=args.repeat_metrics_file,
        device=args.device,
        batch_size=args.batch_size,
        monotonic_tolerance=args.monotonic_tolerance,
        drop_tolerance=args.drop_tolerance,
        variance_tolerance=args.variance_tolerance,
    )
    print(path)


if __name__ == "__main__":
    main()


__all__ = [
    "CURVE_SMOKE_AUDIT_SCHEMA_VERSION",
    "FIXED_JOINT_BUDGETS",
    "audit_curve_smoke",
]
