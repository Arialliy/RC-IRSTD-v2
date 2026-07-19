"""Train-only utilities for the compact Count-all anchor-warp risk model.

This module deliberately has no held-source validation-file argument.  It
constructs every model-selection decision from one source-official-train
episode archive: deterministic episode cross-validation selects an epoch,
then a fresh model is refit on every training episode for exactly that many
epochs.  A caller may bind a held-source evaluation archive only after the
returned :class:`TrainOnlyAnchorWarpResult` has been frozen and hashed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .anchor_warp_predictor import (
    ANCHOR_WARP_ARCHITECTURE_VERSION,
    CountAllAnchorWarpRiskCurve,
)
from .count_all_anchor import (
    derive_anchor_log_curves,
    validate_count_all_anchor_archive,
)
from .monotone_curve_predictor import (
    COMPONENT_LOG_RISK_FLOOR,
    PIXEL_LOG_RISK_FLOOR,
)


ANCHOR_WARP_TRAINING_SCHEMA_VERSION = "rc-v4-anchor-warp-train-only-v1"
ANCHOR_WARP_NORMALIZATION_SCHEMA_VERSION = "robust-median-mad-v1-train-only"
ANCHOR_WARP_SELECTION_SCHEMA_VERSION = (
    "episode-5fold-cv-median-epoch-all-train-refit-v1"
)

DEFAULT_PIXEL_BUDGETS = (1.0e-5, 1.0e-6)
DEFAULT_COMPONENT_BUDGETS = (5.0, 1.0)
DEFAULT_QUANTILE = 0.90
DEFAULT_BASE_WEIGHT = 0.10
DEFAULT_BUDGET_WEIGHT = 0.45
DEFAULT_LEFT_RADIUS = 512
DEFAULT_RIGHT_RADIUS = 128
DEFAULT_STATISTICS_CLIP = 4.0
def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _validate_indices(indices: Sequence[int], size: int, field: str) -> np.ndarray:
    raw = np.asarray(indices)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iu":
        raise ValueError(f"{field} must be a non-empty one-dimensional integer list")
    values = raw.astype(np.int64, copy=True)
    if np.any(values < 0) or np.any(values >= size):
        raise ValueError(f"{field} contains an out-of-range episode index")
    if np.unique(values).size != values.size:
        raise ValueError(f"{field} contains a duplicate episode index")
    return values


@dataclass(frozen=True)
class RobustMedianMADScaler:
    """A train-subset-only robust scaler with a fixed bounded deployment map."""

    median: np.ndarray
    scale: np.ndarray
    clip: float = DEFAULT_STATISTICS_CLIP
    schema_version: str = ANCHOR_WARP_NORMALIZATION_SCHEMA_VERSION

    @classmethod
    def fit(
        cls,
        statistics: np.ndarray,
        indices: Sequence[int] | None = None,
        *,
        clip: float = DEFAULT_STATISTICS_CLIP,
    ) -> "RobustMedianMADScaler":
        values = np.asarray(statistics, dtype=np.float32)
        if values.ndim != 2 or min(values.shape) <= 0 or not np.isfinite(values).all():
            raise ValueError("statistics must be a non-empty finite [N,D] array")
        if not np.isfinite(clip) or float(clip) <= 0.0:
            raise ValueError("clip must be finite and positive")
        if indices is None:
            subset = values
        else:
            subset = values[_validate_indices(indices, values.shape[0], "indices")]
        median64 = np.median(subset.astype(np.float64), axis=0)
        mad64 = np.median(np.abs(subset.astype(np.float64) - median64), axis=0)
        scale64 = np.maximum(1.4826 * mad64, 1.0e-6)
        if not np.isfinite(median64).all() or not np.isfinite(scale64).all():
            raise ValueError("robust statistic normalization is non-finite")
        return cls(
            median=_readonly(median64.astype(np.float32)),
            scale=_readonly(scale64.astype(np.float32)),
            clip=float(clip),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RobustMedianMADScaler":
        if not isinstance(payload, Mapping):
            raise TypeError("normalization payload must be a mapping")
        if payload.get("schema_version") != ANCHOR_WARP_NORMALIZATION_SCHEMA_VERSION:
            raise ValueError("unsupported anchor-warp normalization schema")
        median = np.asarray(payload.get("median"), dtype=np.float32)
        scale = np.asarray(payload.get("scale"), dtype=np.float32)
        clip = float(payload.get("clip", float("nan")))
        if (
            median.ndim != 1
            or median.size == 0
            or scale.shape != median.shape
            or not np.isfinite(median).all()
            or not np.isfinite(scale).all()
            or np.any(scale < 1.0e-6)
            or not np.isfinite(clip)
            or clip <= 0.0
        ):
            raise ValueError("invalid anchor-warp normalization payload")
        return cls(_readonly(median), _readonly(scale), clip=clip)

    def transform(self, statistics: np.ndarray) -> np.ndarray:
        values = np.asarray(statistics, dtype=np.float32)
        if values.ndim != 2 or values.shape[1:] != self.median.shape:
            raise ValueError("statistics and robust scaler dimensions disagree")
        if not np.isfinite(values).all():
            raise ValueError("statistics contain NaN or infinite values")
        normalized = (values - self.median[None, :]) / self.scale[None, :]
        normalized = np.clip(normalized, -self.clip, self.clip).astype(
            np.float32, copy=False
        )
        if not np.isfinite(normalized).all():
            raise ValueError("normalized statistics are non-finite")
        return np.ascontiguousarray(normalized)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fit_scope": "training_episode_subset_only",
            "center": "featurewise_median",
            "scale_rule": "max(1.4826*MAD,1e-6)",
            "clip": self.clip,
            "median": self.median.tolist(),
            "scale": self.scale.tolist(),
        }


@dataclass(frozen=True)
class AnchorWarpTrainingData:
    statistics: np.ndarray
    pixel_anchor: np.ndarray
    component_anchor: np.ndarray
    pixel_target: np.ndarray
    component_target: np.ndarray
    threshold_weights: np.ndarray
    focus_mask: np.ndarray
    crossing_indices: np.ndarray
    episode_keys: tuple[str, ...]
    anchor_semantic_sha256: str
    threshold_grid_sha256: str

    @property
    def num_episodes(self) -> int:
        return int(self.statistics.shape[0])

    @property
    def num_thresholds(self) -> int:
        return int(self.pixel_anchor.shape[1])


@dataclass(frozen=True)
class FoldTrainingResult:
    fold_index: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    best_epoch: int
    best_objective: float
    history: tuple[Mapping[str, float | int], ...]
    state_dict: Mapping[str, torch.Tensor]
    scaler: RobustMedianMADScaler


@dataclass(frozen=True)
class TrainOnlyAnchorWarpResult:
    model_config: Mapping[str, int | float | str]
    state_dict: Mapping[str, torch.Tensor]
    scaler: RobustMedianMADScaler
    pixel_oof_inflation: float
    component_oof_inflation: float
    fixed_epoch: int
    folds: tuple[FoldTrainingResult, ...]
    cv_aggregate_history: tuple[Mapping[str, float | int], ...]
    final_history: tuple[Mapping[str, float | int], ...]
    oof_prediction_sha256: str
    frozen_model_semantic_sha256: str
    anchor_semantic_sha256: str
    threshold_grid_sha256: str
    training_schema_version: str = ANCHOR_WARP_TRAINING_SCHEMA_VERSION
    selection_schema_version: str = ANCHOR_WARP_SELECTION_SCHEMA_VERSION


def _episode_keys(archive: Mapping[str, np.ndarray], rows: int) -> tuple[str, ...]:
    required = ("pseudo_targets", "adaptation_ids", "evaluation_ids")
    for field in required:
        values = np.asarray(archive.get(field))
        if values.ndim != 1 or values.shape[0] != rows:
            raise ValueError(f"{field} must contain one value per episode")
    seen_ids: dict[str, tuple[str, int]] = {}
    keys_list: list[str] = []
    for index in range(rows):
        decoded_rows: dict[str, list[str]] = {}
        for field in ("adaptation_ids", "evaluation_ids"):
            raw_value = str(np.asarray(archive[field])[index])
            try:
                decoded = json.loads(raw_value)
            except json.JSONDecodeError as error:
                raise ValueError(f"{field}[{index}] is not valid JSON") from error
            if (
                not isinstance(decoded, list)
                or not decoded
                or any(not isinstance(item, str) or not item for item in decoded)
                or len(set(decoded)) != len(decoded)
            ):
                raise ValueError(f"{field}[{index}] must be a distinct non-empty ID list")
            for image_id in decoded:
                previous = seen_ids.get(image_id)
                if previous is not None:
                    raise ValueError(
                        "train-only CV requires globally unique A/E image IDs; "
                        f"{image_id!r} is reused by {previous} and {(field, index)}"
                    )
                seen_ids[image_id] = (field, index)
            decoded_rows[field] = decoded
        keys_list.append(
            "\x1f".join(
                (
                    str(np.asarray(archive["pseudo_targets"])[index]),
                    json.dumps(decoded_rows["adaptation_ids"], separators=(",", ":")),
                    json.dumps(decoded_rows["evaluation_ids"], separators=(",", ":")),
                )
            )
        )
    keys = tuple(keys_list)
    if len(set(keys)) != len(keys):
        raise ValueError("training archive contains duplicate episode identities")
    return keys


def _budget_focus_weights(
    pixel_anchor: np.ndarray,
    component_anchor: np.ndarray,
    *,
    pixel_budgets: Sequence[float],
    component_budgets: Sequence[float],
    base_weight: float,
    budget_weight: float,
    left_radius: int,
    right_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixel = np.asarray(pixel_budgets, dtype=np.float64)
    component = np.asarray(component_budgets, dtype=np.float64)
    if (
        pixel.ndim != 1
        or component.shape != pixel.shape
        or pixel.size == 0
        or not np.isfinite(pixel).all()
        or not np.isfinite(component).all()
        or np.any(pixel <= 0.0)
        or np.any(component <= 0.0)
    ):
        raise ValueError("joint risk budgets must be aligned, finite, and positive")
    if not np.isfinite(base_weight) or base_weight <= 0.0:
        raise ValueError("base_weight must be finite and positive")
    if not np.isfinite(budget_weight) or budget_weight < 0.0:
        raise ValueError("budget_weight must be finite and non-negative")
    if left_radius < 0 or right_radius < 0:
        raise ValueError("budget neighborhood radii must be non-negative")
    rows, thresholds = pixel_anchor.shape
    weights = np.full((rows, thresholds), float(base_weight), dtype=np.float64)
    focus = np.zeros((rows, thresholds), dtype=bool)
    crossings = np.empty((rows, pixel.size), dtype=np.int64)
    log_pixel = np.log10(pixel)
    log_component = np.log10(component)
    for row in range(rows):
        for budget_index, (pixel_limit, component_limit) in enumerate(
            zip(log_pixel, log_component)
        ):
            feasible = np.flatnonzero(
                (pixel_anchor[row] <= pixel_limit)
                & (component_anchor[row] <= component_limit)
            )
            crossing = int(feasible[0]) if feasible.size else thresholds - 1
            crossings[row, budget_index] = crossing
            start = max(0, crossing - int(left_radius))
            stop = min(thresholds, crossing + int(right_radius) + 1)
            weights[row, start:stop] += float(budget_weight)
            focus[row, start:stop] = True
        weights[row] /= weights[row].mean()
    if not focus.any(axis=1).all():
        raise RuntimeError("every training episode must have a budget focus region")
    return (
        _readonly(weights.astype(np.float32)),
        _readonly(focus),
        _readonly(crossings),
    )


def prepare_anchor_warp_training_data(
    archive: Mapping[str, np.ndarray],
    *,
    pixel_budgets: Sequence[float] = DEFAULT_PIXEL_BUDGETS,
    component_budgets: Sequence[float] = DEFAULT_COMPONENT_BUDGETS,
    base_weight: float = DEFAULT_BASE_WEIGHT,
    budget_weight: float = DEFAULT_BUDGET_WEIGHT,
    left_radius: int = DEFAULT_LEFT_RADIUS,
    right_radius: int = DEFAULT_RIGHT_RADIUS,
) -> AnchorWarpTrainingData:
    """Validate one train archive and derive its label-free A-window anchors."""

    if not isinstance(archive, Mapping):
        raise TypeError("archive must be a mapping")
    anchors = validate_count_all_anchor_archive(archive)
    statistics = np.asarray(archive["statistics"], dtype=np.float32)
    if statistics.ndim != 2 or statistics.shape[1] != 119:
        raise ValueError("anchor-warp training requires statistics shape [N,119]")
    if statistics.shape[0] != anchors.num_episodes or not np.isfinite(statistics).all():
        raise ValueError("statistics do not align with Count-all anchor episodes")
    pixel_anchor, component_anchor = derive_anchor_log_curves(anchors)
    target_pixel = np.asarray(archive["pixel_log_risk"], dtype=np.float32)
    target_component = np.asarray(
        archive["component_log_risk_upper"], dtype=np.float32
    )
    expected = pixel_anchor.shape
    for name, values, floor in (
        ("pixel_anchor", pixel_anchor, PIXEL_LOG_RISK_FLOOR),
        ("component_anchor", component_anchor, COMPONENT_LOG_RISK_FLOOR),
        ("pixel_log_risk", target_pixel, PIXEL_LOG_RISK_FLOOR),
        ("component_log_risk_upper", target_component, COMPONENT_LOG_RISK_FLOOR),
    ):
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite with shape {expected}")
        if np.any(values < floor - 8.0 * np.finfo(np.float32).eps):
            raise ValueError(f"{name} falls below its registered physical floor")
        if np.any(np.diff(values.astype(np.float64), axis=1) > 0.0):
            raise ValueError(f"{name} must be non-increasing")
    weights, focus, crossings = _budget_focus_weights(
        pixel_anchor,
        component_anchor,
        pixel_budgets=pixel_budgets,
        component_budgets=component_budgets,
        base_weight=base_weight,
        budget_weight=budget_weight,
        left_radius=left_radius,
        right_radius=right_radius,
    )
    return AnchorWarpTrainingData(
        statistics=_readonly(statistics),
        pixel_anchor=_readonly(pixel_anchor),
        component_anchor=_readonly(component_anchor),
        pixel_target=_readonly(target_pixel),
        component_target=_readonly(target_component),
        threshold_weights=weights,
        focus_mask=focus,
        crossing_indices=crossings,
        episode_keys=_episode_keys(archive, statistics.shape[0]),
        anchor_semantic_sha256=anchors.semantic_sha256,
        threshold_grid_sha256=anchors.threshold_grid_sha256,
    )


def deterministic_episode_folds(
    episode_keys: Sequence[str], *, num_folds: int = 5, seed: int = 42
) -> tuple[tuple[int, ...], ...]:
    keys = tuple(str(value) for value in episode_keys)
    if len(keys) < 2 or len(set(keys)) != len(keys):
        raise ValueError("episode keys must be distinct and contain at least two rows")
    if isinstance(num_folds, bool) or not 2 <= int(num_folds) <= len(keys):
        raise ValueError("num_folds must lie in [2, num_episodes]")
    ranked = sorted(
        range(len(keys)),
        key=lambda index: hashlib.sha256(
            f"{int(seed)}\x00{keys[index]}".encode("utf-8")
        ).digest(),
    )
    folds: list[list[int]] = [[] for _ in range(int(num_folds))]
    for position, index in enumerate(ranked):
        folds[position % int(num_folds)].append(index)
    result = tuple(tuple(sorted(fold)) for fold in folds)
    flattened = [index for fold in result for index in fold]
    if sorted(flattened) != list(range(len(keys))) or any(not fold for fold in result):
        raise RuntimeError("deterministic episode folds are not a full partition")
    return result


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }


def _tensor_subset(
    data: AnchorWarpTrainingData,
    normalized_statistics: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "statistics": torch.from_numpy(normalized_statistics[indices]).to(device),
        "pixel_anchor": torch.from_numpy(data.pixel_anchor[indices]).to(device),
        "component_anchor": torch.from_numpy(data.component_anchor[indices]).to(device),
        "pixel_target": torch.from_numpy(data.pixel_target[indices]).to(device),
        "component_target": torch.from_numpy(data.component_target[indices]).to(device),
        "weights": torch.from_numpy(data.threshold_weights[indices]).to(device),
        "focus": torch.from_numpy(data.focus_mask[indices]).to(device),
    }


def _weighted_pinball(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    error = target - prediction
    loss = torch.maximum(quantile * error, (quantile - 1.0) * error)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def anchor_warp_data_objective(
    prediction: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    quantile: float = DEFAULT_QUANTILE,
    lambda_component: float = 1.0,
    max_underestimation_weight: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")
    if lambda_component < 0.0 or max_underestimation_weight < 0.0:
        raise ValueError("loss weights must be non-negative")
    pixel = _weighted_pinball(
        prediction["pixel_log_risk"],
        batch["pixel_target"],
        batch["weights"],
        float(quantile),
    )
    component = _weighted_pinball(
        prediction["component_log_risk"],
        batch["component_target"],
        batch["weights"],
        float(quantile),
    )
    focus = batch["focus"]
    pixel_under = torch.where(
        focus,
        torch.relu(batch["pixel_target"] - prediction["pixel_log_risk"]),
        torch.zeros_like(batch["pixel_target"]),
    ).amax(dim=1).mean()
    component_under = torch.where(
        focus,
        torch.relu(
            batch["component_target"] - prediction["component_log_risk"]
        ),
        torch.zeros_like(batch["component_target"]),
    ).amax(dim=1).mean()
    hinge = 0.5 * (pixel_under + component_under)
    total = pixel + float(lambda_component) * component
    total = total + float(max_underestimation_weight) * hinge
    return total, {
        "pixel_pinball": pixel,
        "component_pinball": component,
        "max_underestimation": hinge,
    }


def _regularization(
    model: CountAllAnchorWarpRiskCurve,
    statistics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    parameters = model.adaptation_parameters(statistics)
    dtype = statistics.dtype
    device = statistics.device
    uniform = torch.linspace(
        0.0,
        1.0,
        model.num_warp_segments + 1,
        dtype=dtype,
        device=device,
    ).unsqueeze(0)
    smooth_terms: list[torch.Tensor] = []
    identity_terms: list[torch.Tensor] = []
    for prefix in ("pixel", "component"):
        knots = parameters[f"{prefix}_warp_knots"]
        smooth_terms.append(torch.diff(knots, n=2, dim=1).square().mean())
        identity_terms.extend(
            (
                (knots - uniform).square().mean(),
                (parameters[f"{prefix}_beta"] - 1.0).square().mean(),
                parameters[f"{prefix}_delta"].square().mean(),
            )
        )
    return torch.stack(smooth_terms).mean(), torch.stack(identity_terms).mean()


def _forward(
    model: CountAllAnchorWarpRiskCurve,
    batch: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    return model(
        batch["statistics"], batch["pixel_anchor"], batch["component_anchor"]
    )


def _fit_with_inner_validation(
    data: AnchorWarpTrainingData,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    scaler: RobustMedianMADScaler,
    *,
    fold_index: int,
    seed: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    quantile: float,
    lambda_component: float,
    max_underestimation_weight: float,
    smoothness_weight: float,
    identity_weight: float,
    grad_clip_norm: float,
    device: torch.device,
) -> tuple[FoldTrainingResult, np.ndarray, np.ndarray]:
    _seed_everything(seed)
    normalized = scaler.transform(data.statistics)
    train_batch = _tensor_subset(data, normalized, train_indices, device)
    validation_batch = _tensor_subset(data, normalized, validation_indices, device)
    model = CountAllAnchorWarpRiskCurve(num_thresholds=data.num_thresholds).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_objective = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        prediction = _forward(model, train_batch)
        data_loss, _ = anchor_warp_data_objective(
            prediction,
            train_batch,
            quantile=quantile,
            lambda_component=lambda_component,
            max_underestimation_weight=max_underestimation_weight,
        )
        smoothness, identity = _regularization(model, train_batch["statistics"])
        loss = data_loss + smoothness_weight * smoothness + identity_weight * identity
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("anchor-warp training loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_prediction = _forward(model, validation_batch)
            validation_loss, _ = anchor_warp_data_objective(
                validation_prediction,
                validation_batch,
                quantile=quantile,
                lambda_component=lambda_component,
                max_underestimation_weight=max_underestimation_weight,
            )
        objective = float(validation_loss.detach().cpu())
        history.append(
            {
                "epoch": epoch,
                "train_objective": float(loss.detach().cpu()),
                "inner_validation_objective": objective,
            }
        )
        if objective < best_objective - 1.0e-8:
            best_objective = objective
            best_epoch = epoch
            best_state = _state_dict_cpu(model)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None or best_epoch <= 0:
        raise RuntimeError("inner fold did not produce a finite checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.inference_mode():
        oof = _forward(model, validation_batch)
    return (
        FoldTrainingResult(
            fold_index=fold_index,
            train_indices=tuple(int(value) for value in train_indices.tolist()),
            validation_indices=tuple(
                int(value) for value in validation_indices.tolist()
            ),
            best_epoch=best_epoch,
            best_objective=best_objective,
            history=tuple(history),
            state_dict=best_state,
            scaler=scaler,
        ),
        oof["pixel_log_risk"].detach().cpu().numpy(),
        oof["component_log_risk"].detach().cpu().numpy(),
    )


def _fit_fixed_epochs(
    data: AnchorWarpTrainingData,
    scaler: RobustMedianMADScaler,
    *,
    train_indices: np.ndarray | None = None,
    epochs: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    quantile: float,
    lambda_component: float,
    max_underestimation_weight: float,
    smoothness_weight: float,
    identity_weight: float,
    grad_clip_norm: float,
    device: torch.device,
) -> tuple[CountAllAnchorWarpRiskCurve, tuple[Mapping[str, float | int], ...]]:
    _seed_everything(seed)
    indices = (
        np.arange(data.num_episodes, dtype=np.int64)
        if train_indices is None
        else _validate_indices(train_indices, data.num_episodes, "train_indices")
    )
    batch = _tensor_subset(data, scaler.transform(data.statistics), indices, device)
    model = CountAllAnchorWarpRiskCurve(num_thresholds=data.num_thresholds).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        prediction = _forward(model, batch)
        data_loss, parts = anchor_warp_data_objective(
            prediction,
            batch,
            quantile=quantile,
            lambda_component=lambda_component,
            max_underestimation_weight=max_underestimation_weight,
        )
        smoothness, identity = _regularization(model, batch["statistics"])
        loss = data_loss + smoothness_weight * smoothness + identity_weight * identity
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("all-train refit loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip_norm
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("all-train refit gradient is non-finite")
        optimizer.step()
        history.append(
            {
                "epoch": epoch,
                "train_objective": float(loss.detach().cpu()),
                "pixel_pinball": float(parts["pixel_pinball"].detach().cpu()),
                "component_pinball": float(
                    parts["component_pinball"].detach().cpu()
                ),
                "max_underestimation": float(
                    parts["max_underestimation"].detach().cpu()
                ),
            }
        )
    return model, tuple(history)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    mass = np.asarray(weights, dtype=np.float64).reshape(-1)
    if (
        raw.shape != mass.shape
        or raw.size == 0
        or not np.isfinite(raw).all()
        or not np.isfinite(mass).all()
        or np.any(mass < 0.0)
        or mass.sum() <= 0.0
        or not 0.0 < q < 1.0
    ):
        raise ValueError("invalid weighted quantile inputs")
    order = np.argsort(raw, kind="mergesort")
    cumulative = np.cumsum(mass[order])
    target = q * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, target, side="left")), raw.size - 1)
    return float(raw[order[index]])


def _oof_prediction_sha256(pixel: np.ndarray, component: np.ndarray) -> str:
    digest = hashlib.sha256(b"rc-v4-anchor-warp-oof-predictions-v1\0")
    for name, values in (("pixel", pixel), ("component", component)):
        array = np.ascontiguousarray(values, dtype="<f4")
        digest.update(name.encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def frozen_anchor_warp_semantic_sha256(
    *,
    state_dict: Mapping[str, torch.Tensor],
    model_config: Mapping[str, Any],
    scaler: RobustMedianMADScaler,
    pixel_oof_inflation: float,
    component_oof_inflation: float,
    fixed_epoch: int,
) -> str:
    state_hash = state_dict_semantic_sha256(state_dict)
    header = {
        "architecture_version": ANCHOR_WARP_ARCHITECTURE_VERSION,
        "training_schema_version": ANCHOR_WARP_TRAINING_SCHEMA_VERSION,
        "selection_schema_version": ANCHOR_WARP_SELECTION_SCHEMA_VERSION,
        "model_config": dict(model_config),
        "normalization": scaler.payload(),
        "pixel_oof_inflation": float(pixel_oof_inflation),
        "component_oof_inflation": float(component_oof_inflation),
        "fixed_epoch": int(fixed_epoch),
        "state_dict_semantic_sha256": state_hash,
    }
    return hashlib.sha256(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def state_dict_semantic_sha256(
    state_dict: Mapping[str, torch.Tensor],
) -> str:
    """Hash ordered tensor names, dtypes, shapes and exact CPU bytes."""

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("state_dict must be a non-empty mapping")
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor) or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"state_dict tensor {name!r} is invalid")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def train_anchor_warp_train_only(
    data: AnchorWarpTrainingData,
    *,
    seed: int = 42,
    num_folds: int = 5,
    max_epochs: int = 400,
    patience: int = 40,
    learning_rate: float = 3.0e-3,
    weight_decay: float = 1.0e-3,
    quantile: float = DEFAULT_QUANTILE,
    lambda_component: float = 1.0,
    max_underestimation_weight: float = 0.10,
    smoothness_weight: float = 1.0e-3,
    identity_weight: float = 1.0e-3,
    grad_clip_norm: float = 1.0,
    oof_inflation_quantile: float | None = 0.90,
    device: str | torch.device = "cpu",
) -> TrainOnlyAnchorWarpResult:
    """Select and refit an anchor-warp model without any external val labels."""

    if not isinstance(data, AnchorWarpTrainingData):
        raise TypeError("data must be AnchorWarpTrainingData")
    if max_epochs <= 0 or patience <= 0 or patience > max_epochs:
        raise ValueError("max_epochs/patience must be positive with patience <= max")
    if learning_rate <= 0.0 or weight_decay < 0.0 or grad_clip_norm <= 0.0:
        raise ValueError("optimizer hyperparameters are invalid")
    if oof_inflation_quantile is not None and not (
        0.0 < float(oof_inflation_quantile) < 1.0
    ):
        raise ValueError("oof_inflation_quantile must be None or lie in (0,1)")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    folds = deterministic_episode_folds(
        data.episode_keys, num_folds=num_folds, seed=seed
    )
    all_indices = np.arange(data.num_episodes, dtype=np.int64)
    oof_pixel = np.full_like(data.pixel_target, np.nan, dtype=np.float32)
    oof_component = np.full_like(data.component_target, np.nan, dtype=np.float32)
    # Phase 1: trace every fold for the same maximum epoch horizon.  A single
    # pooled CV history selects one global epoch; fold-specific best epochs are
    # never combined post hoc.
    trace_results: list[FoldTrainingResult] = []
    fold_train_indices: list[np.ndarray] = []
    fold_validation_indices: list[np.ndarray] = []
    fold_scalers: list[RobustMedianMADScaler] = []
    for fold_index, validation_tuple in enumerate(folds):
        validation_indices = np.asarray(validation_tuple, dtype=np.int64)
        train_indices = np.setdiff1d(all_indices, validation_indices, assume_unique=True)
        scaler = RobustMedianMADScaler.fit(data.statistics, train_indices)
        trace, _unused_pixel, _unused_component = _fit_with_inner_validation(
            data,
            train_indices,
            validation_indices,
            scaler,
            fold_index=fold_index,
            seed=int(seed) + fold_index,
            max_epochs=int(max_epochs),
            # Fold-local stopping would make later epochs incomparable across
            # folds.  The registered patience is applied to the pooled curve.
            patience=int(max_epochs),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            quantile=float(quantile),
            lambda_component=float(lambda_component),
            max_underestimation_weight=float(max_underestimation_weight),
            smoothness_weight=float(smoothness_weight),
            identity_weight=float(identity_weight),
            grad_clip_norm=float(grad_clip_norm),
            device=torch_device,
        )
        if len(trace.history) != int(max_epochs):
            raise RuntimeError("a CV fold did not trace the complete epoch horizon")
        trace_results.append(trace)
        fold_train_indices.append(train_indices)
        fold_validation_indices.append(validation_indices)
        fold_scalers.append(scaler)

    aggregate_history: list[dict[str, float | int]] = []
    best_cv_objective = float("inf")
    fixed_epoch = 0
    stale = 0
    total_validation_rows = sum(values.size for values in fold_validation_indices)
    for epoch_offset in range(int(max_epochs)):
        objective = sum(
            float(trace.history[epoch_offset]["inner_validation_objective"])
            * fold_validation_indices[fold_index].size
            for fold_index, trace in enumerate(trace_results)
        ) / float(total_validation_rows)
        aggregate_history.append(
            {"epoch": epoch_offset + 1, "pooled_inner_cv_objective": objective}
        )
        if objective < best_cv_objective - 1.0e-8:
            best_cv_objective = objective
            fixed_epoch = epoch_offset + 1
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    if fixed_epoch <= 0:
        raise RuntimeError("pooled inner CV did not select a fixed epoch")

    # Phase 2: rerun every fold from its registered seed for exactly the one
    # selected epoch and produce genuine OOF predictions.  Each row is scored
    # by a model that did not train on that row.
    fold_results: list[FoldTrainingResult] = []
    for fold_index, trace in enumerate(trace_results):
        train_indices = fold_train_indices[fold_index]
        validation_indices = fold_validation_indices[fold_index]
        scaler = fold_scalers[fold_index]
        fold_model, _fold_refit_history = _fit_fixed_epochs(
            data,
            scaler,
            train_indices=train_indices,
            epochs=fixed_epoch,
            seed=int(seed) + fold_index,
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            quantile=float(quantile),
            lambda_component=float(lambda_component),
            max_underestimation_weight=float(max_underestimation_weight),
            smoothness_weight=float(smoothness_weight),
            identity_weight=float(identity_weight),
            grad_clip_norm=float(grad_clip_norm),
            device=torch_device,
        )
        validation_batch = _tensor_subset(
            data,
            scaler.transform(data.statistics),
            validation_indices,
            torch_device,
        )
        fold_model.eval()
        with torch.inference_mode():
            prediction = _forward(fold_model, validation_batch)
            fold_objective, _ = anchor_warp_data_objective(
                prediction,
                validation_batch,
                quantile=float(quantile),
                lambda_component=float(lambda_component),
                max_underestimation_weight=float(max_underestimation_weight),
            )
        oof_pixel[validation_indices] = (
            prediction["pixel_log_risk"].detach().cpu().numpy()
        )
        oof_component[validation_indices] = (
            prediction["component_log_risk"].detach().cpu().numpy()
        )
        fold_results.append(
            FoldTrainingResult(
                fold_index=fold_index,
                train_indices=tuple(int(value) for value in train_indices.tolist()),
                validation_indices=tuple(
                    int(value) for value in validation_indices.tolist()
                ),
                best_epoch=fixed_epoch,
                best_objective=float(fold_objective.detach().cpu()),
                history=trace.history[: len(aggregate_history)],
                state_dict=_state_dict_cpu(fold_model),
                scaler=scaler,
            )
        )
    if not np.isfinite(oof_pixel).all() or not np.isfinite(oof_component).all():
        raise RuntimeError("OOF prediction matrix was not filled exactly once")
    oof_prediction_hash = _oof_prediction_sha256(oof_pixel, oof_component)
    if oof_inflation_quantile is None:
        pixel_inflation = component_inflation = 0.0
    else:
        oof_focus_weights = np.where(
            data.focus_mask, data.threshold_weights, 0.0
        ).astype(np.float32)
        pixel_inflation = max(
            0.0,
            _weighted_quantile(
                data.pixel_target - oof_pixel,
                oof_focus_weights,
                float(oof_inflation_quantile),
            ),
        )
        component_inflation = max(
            0.0,
            _weighted_quantile(
                data.component_target - oof_component,
                oof_focus_weights,
                float(oof_inflation_quantile),
            ),
        )
    final_scaler = RobustMedianMADScaler.fit(data.statistics)
    final_model, final_history = _fit_fixed_epochs(
        data,
        final_scaler,
        epochs=fixed_epoch,
        seed=int(seed) + 10_000,
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        quantile=float(quantile),
        lambda_component=float(lambda_component),
        max_underestimation_weight=float(max_underestimation_weight),
        smoothness_weight=float(smoothness_weight),
        identity_weight=float(identity_weight),
        grad_clip_norm=float(grad_clip_norm),
        device=torch_device,
    )
    state = _state_dict_cpu(final_model)
    config = final_model.config()
    frozen_hash = frozen_anchor_warp_semantic_sha256(
        state_dict=state,
        model_config=config,
        scaler=final_scaler,
        pixel_oof_inflation=pixel_inflation,
        component_oof_inflation=component_inflation,
        fixed_epoch=fixed_epoch,
    )
    return TrainOnlyAnchorWarpResult(
        model_config=config,
        state_dict=state,
        scaler=final_scaler,
        pixel_oof_inflation=float(pixel_inflation),
        component_oof_inflation=float(component_inflation),
        fixed_epoch=fixed_epoch,
        folds=tuple(fold_results),
        cv_aggregate_history=tuple(aggregate_history),
        final_history=final_history,
        oof_prediction_sha256=oof_prediction_hash,
        frozen_model_semantic_sha256=frozen_hash,
        anchor_semantic_sha256=data.anchor_semantic_sha256,
        threshold_grid_sha256=data.threshold_grid_sha256,
    )


def predict_frozen_anchor_warp(
    result: TrainOnlyAnchorWarpResult,
    *,
    statistics: np.ndarray,
    pixel_anchor: np.ndarray,
    component_anchor: np.ndarray,
    device: str | torch.device = "cpu",
) -> dict[str, np.ndarray]:
    """Shared train/evaluation inference including frozen OOF inflation."""

    torch_device = torch.device(device)
    model = CountAllAnchorWarpRiskCurve(**dict(result.model_config)).to(torch_device)
    model.load_state_dict(result.state_dict, strict=True)
    model.eval()
    normalized = result.scaler.transform(statistics)
    pixel = np.array(pixel_anchor, dtype=np.float32, order="C", copy=True)
    component = np.array(component_anchor, dtype=np.float32, order="C", copy=True)
    with torch.inference_mode():
        prediction = model(
            torch.from_numpy(normalized).to(torch_device),
            torch.from_numpy(pixel).to(torch_device),
            torch.from_numpy(component).to(torch_device),
        )
    return {
        "pixel_log_risk": (
            prediction["pixel_log_risk"].cpu().numpy()
            + result.pixel_oof_inflation
        ).astype(np.float32),
        "component_log_risk": (
            prediction["component_log_risk"].cpu().numpy()
            + result.component_oof_inflation
        ).astype(np.float32),
    }


__all__ = [
    "ANCHOR_WARP_NORMALIZATION_SCHEMA_VERSION",
    "ANCHOR_WARP_SELECTION_SCHEMA_VERSION",
    "ANCHOR_WARP_TRAINING_SCHEMA_VERSION",
    "AnchorWarpTrainingData",
    "FoldTrainingResult",
    "RobustMedianMADScaler",
    "TrainOnlyAnchorWarpResult",
    "anchor_warp_data_objective",
    "deterministic_episode_folds",
    "frozen_anchor_warp_semantic_sha256",
    "predict_frozen_anchor_warp",
    "prepare_anchor_warp_training_data",
    "state_dict_semantic_sha256",
    "train_anchor_warp_train_only",
]
