"""Train and freeze CountAllAnchorWarpRiskCurve without a held val archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .anchor_warp_predictor import (
    ANCHOR_WARP_ARCHITECTURE_VERSION,
    ANCHOR_WARP_PARAMETER_COUNT,
    CountAllAnchorWarpRiskCurve,
)
from .anchor_warp_training import (
    ANCHOR_WARP_NORMALIZATION_SCHEMA_VERSION,
    ANCHOR_WARP_SELECTION_SCHEMA_VERSION,
    ANCHOR_WARP_TRAINING_SCHEMA_VERSION,
    DEFAULT_COMPONENT_BUDGETS,
    DEFAULT_PIXEL_BUDGETS,
    RobustMedianMADScaler,
    frozen_anchor_warp_semantic_sha256,
    prepare_anchor_warp_training_data,
    state_dict_semantic_sha256,
    train_anchor_warp_train_only,
)
from .count_all_anchor import (
    COMPONENT_ANCHOR_LOG_EPSILON,
    COUNT_ALL_ANCHOR_SCHEMA_VERSION,
    PIXEL_ANCHOR_LOG_EPSILON,
    validate_count_all_anchor_archive,
)
from .curve_dataset import (
    COUNT_ALL_ADAPTATION_SCHEMA_VERSION,
    load_curve_archive,
)
from .domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    feature_schema_sha256,
    statistics_names_sha256,
    validate_statistics_names,
)
from .representation import (
    GRID_DETECTOR_PROTOCOL,
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_REPRESENTATION,
    logit_threshold_grid_sha256,
    validate_logit_threshold_grid,
)


ANCHOR_WARP_CHECKPOINT_SCHEMA_VERSION = (
    "rc-v4-count-all-anchor-warp-train-only-checkpoint-v1"
)
ANCHOR_WARP_TARGET_LABEL_POLICY = {
    "model_inputs": (
        "adaptation_window_A_label_free_statistics_and_count_all_curves_only"
    ),
    "supervision": "source_official_train_future_E_risk_only",
    "held_validation_archive_opened_during_checkpoint_selection": False,
    "held_validation_labels_used_for_checkpoint_selection": False,
    "outer_target_labels_used_for_features": False,
    "outer_target_labels_used_for_checkpoint_selection": False,
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )


def _scalar(archive: Mapping[str, np.ndarray], field: str) -> str:
    if field not in archive:
        raise ValueError(f"training archive is missing {field}")
    value = np.asarray(archive[field])
    if value.ndim != 0:
        raise ValueError(f"{field} must be scalar")
    return str(value.item())


def _sha256_scalar(archive: Mapping[str, np.ndarray], field: str) -> str:
    value = _scalar(archive, field)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _detector_hashes(
    archive: Mapping[str, np.ndarray],
) -> tuple[tuple[str, ...], str, tuple[str, ...]]:
    all_hashes = tuple(
        str(value)
        for value in np.asarray(
            archive["threshold_grid_detector_checkpoint_sha256s"]
        ).tolist()
    )
    episode_hashes = tuple(
        str(value)
        for value in np.asarray(
            archive["threshold_grid_episode_detector_checkpoint_sha256s"]
        ).tolist()
    )
    outer = _sha256_scalar(
        archive, "threshold_grid_outer_detector_checkpoint_sha256"
    )
    if (
        len(all_hashes) != 3
        or len(set(all_hashes)) != 3
        or len(episode_hashes) != 2
        or len(set(episode_hashes)) != 2
        or outer in episode_hashes
        or set(all_hashes) != set(episode_hashes).union({outer})
    ):
        raise ValueError("training archive detector role contract is not 2-inner+1-outer")
    for digest in all_hashes:
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("detector checkpoint hash is invalid")
    return all_hashes, outer, episode_hashes


def _fold_metadata(result: Any) -> list[dict[str, Any]]:
    return [
        {
            "fold_index": fold.fold_index,
            "train_indices": list(fold.train_indices),
            "validation_indices": list(fold.validation_indices),
            "selected_epoch": fold.best_epoch,
            "selected_epoch_objective": fold.best_objective,
            "scaler": fold.scaler.payload(),
        }
        for fold in result.folds
    ]


def _policy_hash_from_checkpoint(checkpoint: Mapping[str, Any]) -> str:
    scaler = RobustMedianMADScaler.from_payload(checkpoint["preprocessing"])
    return frozen_anchor_warp_semantic_sha256(
        state_dict=checkpoint["state_dict"],
        model_config=checkpoint["model_config"],
        scaler=scaler,
        pixel_oof_inflation=float(checkpoint["pixel_oof_inflation"]),
        component_oof_inflation=float(checkpoint["component_oof_inflation"]),
        fixed_epoch=int(checkpoint["final_refit_epochs"]),
    )


def validate_anchor_warp_train_only_checkpoint(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on the immutable train-only AnchorWarp checkpoint schema."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError("anchor-warp checkpoint must be a mapping")
    required = {
        "checkpoint_schema_version",
        "artifact_stage",
        "method_name",
        "model_class",
        "model_architecture_version",
        "model_config",
        "state_dict",
        "state_dict_semantic_sha256",
        "policy_semantic_sha256",
        "parameter_count",
        "representation",
        "thresholds",
        "threshold_grid_schema_version",
        "threshold_grid_sha256",
        "threshold_grid_manifest_sha256",
        "threshold_grid_detector_protocol",
        "threshold_grid_detector_checkpoint_sha256s",
        "threshold_grid_outer_detector_checkpoint_sha256",
        "threshold_grid_episode_detector_checkpoint_sha256s",
        "statistics_schema_version",
        "statistics_names",
        "statistics_names_sha256",
        "feature_schema_sha256",
        "preprocessing",
        "anchor_contract",
        "training_schema_version",
        "selection_protocol_schema_version",
        "selection_scope",
        "target_label_policy",
        "train_archive_sha256",
        "train_pseudo_targets",
        "formal_source_domains",
        "excluded_outer_target",
        "train_episode_keys_sha256",
        "fold_assignment_sha256",
        "cv_history_sha256",
        "oof_prediction_sha256",
        "cv_folds",
        "selected_epoch",
        "final_refit_epochs",
        "oof_inflation_enabled",
        "oof_quantile",
        "pixel_oof_inflation",
        "component_oof_inflation",
        "state_frozen_before_validation_binding",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError("anchor-warp checkpoint is missing: " + ", ".join(missing))
    expected = {
        "checkpoint_schema_version": ANCHOR_WARP_CHECKPOINT_SCHEMA_VERSION,
        "artifact_stage": "train_only_frozen",
        "method_name": "count_all_anchor_warp_risk_curve",
        "model_class": "CountAllAnchorWarpRiskCurve",
        "model_architecture_version": ANCHOR_WARP_ARCHITECTURE_VERSION,
        "parameter_count": ANCHOR_WARP_PARAMETER_COUNT,
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "statistics_schema_version": LOGIT_STATISTICS_SCHEMA_VERSION,
        "training_schema_version": ANCHOR_WARP_TRAINING_SCHEMA_VERSION,
        "selection_protocol_schema_version": ANCHOR_WARP_SELECTION_SCHEMA_VERSION,
        "selection_scope": "train_archive_only",
        "target_label_policy": ANCHOR_WARP_TARGET_LABEL_POLICY,
        "state_frozen_before_validation_binding": True,
        "formal_source_domains": ["IRSTD-1K", "NUDT-SIRST"],
        "excluded_outer_target": "NUAA-SIRST",
    }
    for field, value in expected.items():
        if checkpoint.get(field) != value:
            raise ValueError(f"anchor-warp checkpoint {field} must equal {value!r}")
    train_targets = checkpoint["train_pseudo_targets"]
    if (
        not isinstance(train_targets, list)
        or len(train_targets) != 1
        or train_targets[0] not in {"IRSTD-1K", "NUDT-SIRST"}
    ):
        raise ValueError("anchor-warp checkpoint must train on one canonical source")
    if "validation_binding" in checkpoint or "validation_archive_sha256" in checkpoint:
        raise ValueError("train-only frozen checkpoint must not bind a validation archive")
    thresholds = validate_logit_threshold_grid(
        np.asarray(checkpoint["thresholds"], dtype=np.float32)
    )
    if checkpoint["threshold_grid_sha256"] != logit_threshold_grid_sha256(thresholds):
        raise ValueError("anchor-warp checkpoint threshold-grid hash mismatch")
    for field in ("threshold_grid_manifest_sha256", "train_archive_sha256"):
        value = str(checkpoint[field])
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"anchor-warp checkpoint {field} is invalid")
    names = validate_statistics_names(checkpoint["statistics_names"], expected_dim=119)
    if checkpoint["statistics_names_sha256"] != statistics_names_sha256(names):
        raise ValueError("anchor-warp checkpoint statistics-name hash mismatch")
    if checkpoint["feature_schema_sha256"] != feature_schema_sha256(
        LOGIT_STATISTICS_SCHEMA_VERSION, statistics_names=names
    ):
        raise ValueError("anchor-warp checkpoint feature-schema hash mismatch")
    scaler = RobustMedianMADScaler.from_payload(checkpoint["preprocessing"])
    if scaler.median.shape != (119,) or checkpoint["preprocessing"].get(
        "ood_policy"
    ) != "clip_and_audit_no_hard_reject":
        raise ValueError("anchor-warp checkpoint preprocessing contract is invalid")
    anchor = checkpoint["anchor_contract"]
    expected_anchor = {
        "anchor_schema_version": COUNT_ALL_ANCHOR_SCHEMA_VERSION,
        "count_all_adaptation_schema_version": COUNT_ALL_ADAPTATION_SCHEMA_VERSION,
        "pixel_epsilon": PIXEL_ANCHOR_LOG_EPSILON,
        "component_epsilon": COMPONENT_ANCHOR_LOG_EPSILON,
        "component_curve": "suffix_upper",
        "connectivity_argument": 2,
        "connectivity_semantics": "8_neighbor",
        "min_component_area": 1,
        "adaptation_masks_read": False,
    }
    if not isinstance(anchor, Mapping):
        raise ValueError("anchor-warp checkpoint anchor_contract must be a mapping")
    for field, value in expected_anchor.items():
        if anchor.get(field) != value:
            raise ValueError(f"anchor-warp anchor contract {field} mismatch")
    if checkpoint["selected_epoch"] != checkpoint["final_refit_epochs"]:
        raise ValueError("selected epoch and final refit epoch count differ")
    if not isinstance(checkpoint["selected_epoch"], int) or checkpoint["selected_epoch"] <= 0:
        raise ValueError("anchor-warp selected epoch must be positive")
    folds = checkpoint["cv_folds"]
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("anchor-warp formal checkpoint requires five CV folds")
    seen: list[int] = []
    for expected_index, fold in enumerate(folds):
        if not isinstance(fold, Mapping) or fold.get("fold_index") != expected_index:
            raise ValueError("anchor-warp CV fold metadata is malformed")
        train = set(fold.get("train_indices", []))
        validation = set(fold.get("validation_indices", []))
        if not train or not validation or train.intersection(validation):
            raise ValueError("anchor-warp CV fold train/validation overlap")
        if fold.get("selected_epoch") != checkpoint["selected_epoch"]:
            raise ValueError("anchor-warp fold did not use the global selected epoch")
        seen.extend(int(value) for value in validation)
    if sorted(seen) != list(range(len(seen))):
        raise ValueError("anchor-warp OOF folds are not an exact row partition")
    model = CountAllAnchorWarpRiskCurve(**dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != ANCHOR_WARP_PARAMETER_COUNT:
        raise ValueError("anchor-warp checkpoint parameter count changed")
    state_hash = state_dict_semantic_sha256(checkpoint["state_dict"])
    if checkpoint["state_dict_semantic_sha256"] != state_hash:
        raise ValueError("anchor-warp checkpoint state hash mismatch")
    policy_hash = _policy_hash_from_checkpoint(checkpoint)
    if checkpoint["policy_semantic_sha256"] != policy_hash:
        raise ValueError("anchor-warp checkpoint policy hash mismatch")
    for field in (
        "train_episode_keys_sha256",
        "fold_assignment_sha256",
        "cv_history_sha256",
        "oof_prediction_sha256",
    ):
        value = str(checkpoint[field])
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"anchor-warp checkpoint {field} is invalid")
    return {
        "formal_train_only_eligible": True,
        "state_dict_semantic_sha256": state_hash,
        "policy_semantic_sha256": policy_hash,
        "threshold_grid_sha256": checkpoint["threshold_grid_sha256"],
        "selected_epoch": checkpoint["selected_epoch"],
    }


def train_anchor_warp_checkpoint(
    *,
    train_file: str | Path,
    output: str | Path,
    seed: int = 42,
    num_folds: int = 5,
    max_epochs: int = 400,
    patience: int = 40,
    learning_rate: float = 3.0e-3,
    weight_decay: float = 1.0e-3,
    quantile: float = 0.90,
    lambda_component: float = 1.0,
    max_underestimation_weight: float = 0.10,
    smoothness_weight: float = 1.0e-3,
    identity_weight: float = 1.0e-3,
    grad_clip_norm: float = 1.0,
    oof_inflation_quantile: float | None = 0.90,
    pixel_budgets: Sequence[float] = DEFAULT_PIXEL_BUDGETS,
    component_budgets: Sequence[float] = DEFAULT_COMPONENT_BUDGETS,
    left_radius: int = 512,
    right_radius: int = 128,
    device: str = "cpu",
) -> Path:
    train_path = Path(train_file).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not train_path.is_file():
        raise FileNotFoundError(f"training archive does not exist: {train_path}")
    if output_path == train_path:
        raise ValueError("output must not overwrite the training archive")
    train_bytes = train_path.read_bytes()
    train_sha256 = _sha256_bytes(train_bytes)
    archive = load_curve_archive(io.BytesIO(train_bytes))
    if _scalar(archive, "representation") != LOGIT_REPRESENTATION:
        raise ValueError("anchor-warp formal training requires raw-logit archives")
    if _scalar(archive, "statistics_schema_version") != LOGIT_STATISTICS_SCHEMA_VERSION:
        raise ValueError("anchor-warp statistics schema mismatch")
    if _scalar(archive, "threshold_grid_schema_version") != LOGIT_GRID_SCHEMA_VERSION:
        raise ValueError("anchor-warp threshold-grid schema mismatch")
    thresholds = validate_logit_threshold_grid(
        np.asarray(archive["thresholds"], dtype=np.float32)
    )
    grid_hash = logit_threshold_grid_sha256(thresholds)
    if _sha256_scalar(archive, "threshold_grid_sha256") != grid_hash:
        raise ValueError("training archive threshold-grid semantic hash mismatch")
    if _scalar(archive, "threshold_grid_detector_protocol") != GRID_DETECTOR_PROTOCOL:
        raise ValueError("training archive detector-grid protocol mismatch")
    all_detector_hashes, outer_detector_hash, episode_detector_hashes = (
        _detector_hashes(archive)
    )
    names = validate_statistics_names(archive["statistics_names"], expected_dim=119)
    if _sha256_scalar(archive, "statistics_names_sha256") != statistics_names_sha256(names):
        raise ValueError("training archive statistics-name hash mismatch")
    feature_hash = feature_schema_sha256(
        LOGIT_STATISTICS_SCHEMA_VERSION, statistics_names=names
    )
    if _sha256_scalar(archive, "feature_schema_sha256") != feature_hash:
        raise ValueError("training archive feature-schema hash mismatch")
    anchors = validate_count_all_anchor_archive(archive, expected_grid_sha256=grid_hash)
    if anchors.contract.get("connectivity") != 2 or anchors.contract.get(
        "min_component_area"
    ) != 1:
        raise ValueError("anchor-warp formal training requires 8-neighbor/area-1 counts")
    data = prepare_anchor_warp_training_data(
        archive,
        pixel_budgets=pixel_budgets,
        component_budgets=component_budgets,
        left_radius=left_radius,
        right_radius=right_radius,
    )
    result = train_anchor_warp_train_only(
        data,
        seed=seed,
        num_folds=num_folds,
        max_epochs=max_epochs,
        patience=patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        quantile=quantile,
        lambda_component=lambda_component,
        max_underestimation_weight=max_underestimation_weight,
        smoothness_weight=smoothness_weight,
        identity_weight=identity_weight,
        grad_clip_norm=grad_clip_norm,
        oof_inflation_quantile=oof_inflation_quantile,
        device=device,
    )
    fold_metadata = _fold_metadata(result)
    training_hyperparameters = {
        "seed": int(seed),
        "num_folds": int(num_folds),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "quantile": float(quantile),
        "lambda_component": float(lambda_component),
        "max_underestimation_weight": float(max_underestimation_weight),
        "smoothness_weight": float(smoothness_weight),
        "identity_weight": float(identity_weight),
        "grad_clip_norm": float(grad_clip_norm),
        "pixel_budgets": [float(value) for value in pixel_budgets],
        "component_budgets": [float(value) for value in component_budgets],
        "budget_neighborhood_left": int(left_radius),
        "budget_neighborhood_right": int(right_radius),
        "oof_inflation_quantile": (
            None
            if oof_inflation_quantile is None
            else float(oof_inflation_quantile)
        ),
        "optimizer": "AdamW",
        "fold_seed_schedule": [int(seed) + index for index in range(num_folds)],
        "final_refit_seed": int(seed) + 10_000,
    }
    preprocessing = result.scaler.payload()
    preprocessing.update(
        {
            "clip_min": -result.scaler.clip,
            "clip_max": result.scaler.clip,
            "output_dtype": "float32",
            "ood_policy": "clip_and_audit_no_hard_reject",
            "scaler_fit_archive_sha256": train_sha256,
            "scaler_fit_episode_keys_sha256": _json_sha256(data.episode_keys),
        }
    )
    anchor_contract = {
        "anchor_schema_version": COUNT_ALL_ANCHOR_SCHEMA_VERSION,
        "count_all_adaptation_schema_version": COUNT_ALL_ADAPTATION_SCHEMA_VERSION,
        "pixel_formula": "log10(A_pixel_count/A_total_pixels + 1e-12)",
        "component_formula": (
            "log10(A_component_suffix_upper/(A_total_pixels/1e6) + 1e-6)"
        ),
        "pixel_epsilon": PIXEL_ANCHOR_LOG_EPSILON,
        "component_epsilon": COMPONENT_ANCHOR_LOG_EPSILON,
        "component_curve": "suffix_upper",
        "connectivity_argument": 2,
        "connectivity_semantics": "8_neighbor",
        "min_component_area": 1,
        "adaptation_masks_read": False,
        "training_anchor_semantic_sha256": result.anchor_semantic_sha256,
    }
    state_hash = state_dict_semantic_sha256(result.state_dict)
    train_pseudo_targets = sorted(
        {str(value) for value in np.asarray(archive["pseudo_targets"]).tolist()}
    )
    checkpoint: dict[str, Any] = {
        "checkpoint_schema_version": ANCHOR_WARP_CHECKPOINT_SCHEMA_VERSION,
        "artifact_stage": "train_only_frozen",
        "method_name": "count_all_anchor_warp_risk_curve",
        "model_class": "CountAllAnchorWarpRiskCurve",
        "role": "proposed_method",
        "model_architecture_version": ANCHOR_WARP_ARCHITECTURE_VERSION,
        "model_config": dict(result.model_config),
        "state_dict": dict(result.state_dict),
        "state_dict_semantic_sha256": state_hash,
        "policy_semantic_sha256": result.frozen_model_semantic_sha256,
        "parameter_count": ANCHOR_WARP_PARAMETER_COUNT,
        "representation": LOGIT_REPRESENTATION,
        "thresholds": torch.from_numpy(thresholds.copy()),
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": grid_hash,
        "threshold_grid_manifest_sha256": _sha256_scalar(
            archive, "threshold_grid_manifest_sha256"
        ),
        "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": list(all_detector_hashes),
        "threshold_grid_outer_detector_checkpoint_sha256": outer_detector_hash,
        "threshold_grid_episode_detector_checkpoint_sha256s": list(
            episode_detector_hashes
        ),
        "statistics_schema_version": LOGIT_STATISTICS_SCHEMA_VERSION,
        "statistics_names": list(names),
        "statistics_names_sha256": statistics_names_sha256(names),
        "feature_schema_sha256": feature_hash,
        "preprocessing": preprocessing,
        "anchor_contract": anchor_contract,
        "training_schema_version": ANCHOR_WARP_TRAINING_SCHEMA_VERSION,
        "selection_protocol_schema_version": ANCHOR_WARP_SELECTION_SCHEMA_VERSION,
        "selection_scope": "train_archive_only",
        "selection_objective": (
            "pooled_5fold_weighted_q90_pinball_plus_budget_underestimation"
        ),
        "target_label_policy": dict(ANCHOR_WARP_TARGET_LABEL_POLICY),
        "train_archive": str(train_path),
        "train_archive_sha256": train_sha256,
        "train_pseudo_targets": train_pseudo_targets,
        "formal_source_domains": ["IRSTD-1K", "NUDT-SIRST"],
        "excluded_outer_target": "NUAA-SIRST",
        "train_episode_keys_sha256": _json_sha256(data.episode_keys),
        "fold_assignment_sha256": _json_sha256(fold_metadata),
        "cv_history_sha256": _json_sha256(result.cv_aggregate_history),
        "oof_prediction_sha256": result.oof_prediction_sha256,
        "cv_folds": fold_metadata,
        "selected_epoch": result.fixed_epoch,
        "final_refit_epochs": result.fixed_epoch,
        "oof_inflation_enabled": oof_inflation_quantile is not None,
        "oof_quantile": oof_inflation_quantile,
        "oof_quantile_method": "higher",
        "oof_residual_scope": "registered_A_anchor_crossing_neighborhoods",
        "pixel_oof_inflation": result.pixel_oof_inflation,
        "component_oof_inflation": result.component_oof_inflation,
        "training_hyperparameters": training_hyperparameters,
        "state_frozen_before_validation_binding": True,
        "held_validation_archive_first_read_phase": "not_read_by_trainer",
        "seed": int(seed),
    }
    validate_anchor_warp_train_only_checkpoint(checkpoint)
    if _sha256_bytes(train_path.read_bytes()) != train_sha256:
        raise RuntimeError("training archive changed after its immutable input snapshot")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(checkpoint, temporary_path)
        reloaded = torch.load(temporary_path, map_location="cpu", weights_only=True)
        validate_anchor_warp_train_only_checkpoint(reloaded)
        if _sha256_bytes(train_path.read_bytes()) != train_sha256:
            raise RuntimeError("training archive drifted before checkpoint publication")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    metrics = {
        "checkpoint_schema_version": ANCHOR_WARP_CHECKPOINT_SCHEMA_VERSION,
        "artifact_stage": "train_only_frozen",
        "checkpoint_path": str(output_path),
        "checkpoint_sha256": _sha256_bytes(output_path.read_bytes()),
        "policy_semantic_sha256": result.frozen_model_semantic_sha256,
        "state_dict_semantic_sha256": state_hash,
        "selected_epoch": result.fixed_epoch,
        "pixel_oof_inflation": result.pixel_oof_inflation,
        "component_oof_inflation": result.component_oof_inflation,
        "cv_folds": fold_metadata,
        "cv_aggregate_history": list(result.cv_aggregate_history),
        "final_history": list(result.final_history),
        "training_hyperparameters": training_hyperparameters,
        "held_validation_archive_opened": False,
    }
    output_path.with_suffix(output_path.suffix + ".metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--disable-oof-inflation",
        action="store_true",
        help="Disable the preregistered non-negative q90 OOF correction.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = train_anchor_warp_checkpoint(
        train_file=args.train_file,
        output=args.output,
        seed=args.seed,
        num_folds=args.num_folds,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        quantile=args.quantile,
        oof_inflation_quantile=(None if args.disable_oof_inflation else 0.90),
        device=args.device,
    )
    print(path)


if __name__ == "__main__":
    main()


__all__ = [
    "ANCHOR_WARP_CHECKPOINT_SCHEMA_VERSION",
    "ANCHOR_WARP_TARGET_LABEL_POLICY",
    "train_anchor_warp_checkpoint",
    "validate_anchor_warp_train_only_checkpoint",
]
