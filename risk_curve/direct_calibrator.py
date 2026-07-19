"""Fair v4 contracts for the RC-Direct threshold baseline.

RC-Direct is trained from the *same* causal source-only curve archives used by
``RiskCurvePredictor``.  The label-free adaptation statistics are model inputs;
future-E risk curves are used only to construct supervised direct-threshold
targets.  Untouched outer-target labels are never an input or a checkpoint
selection signal.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from rc_irstd.models.calibrator import RC_DIRECT_ARCHITECTURE_VERSION

from .curve_dataset import load_curve_archive, validate_archive_compatibility
from .domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    feature_schema_sha256,
    statistics_names_sha256,
    validate_statistics_names,
)
from .representation import (
    EMPTY_ACTION_THRESHOLD,
    GRID_DETECTOR_PROTOCOL,
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_REPRESENTATION,
    logit_threshold_grid_sha256,
    validate_logit_threshold_grid,
)
from .train_curve_predictor import validate_training_episode_contract


RC_DIRECT_CHECKPOINT_SCHEMA_VERSION = "rc-direct-v4-checkpoint-v1"
RC_DIRECT_BUDGET_SCHEMA_VERSION = "rc-direct-v4-joint-budget-pairs-v1"
RC_DIRECT_SELECTION_SCHEMA_VERSION = "rc-direct-v4-selection-v1"
ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL = GRID_DETECTOR_PROTOCOL
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class DirectTrainingPair:
    train_archive: dict[str, np.ndarray]
    validation_archive: dict[str, np.ndarray]
    statistics_names: tuple[str, ...]
    statistics_mean: np.ndarray
    statistics_std: np.ndarray
    episode_contract: dict[str, object]
    outer_detector_checkpoint_sha256: str
    episode_detector_checkpoint_sha256s: tuple[str, ...]


@dataclass(frozen=True)
class DirectThresholdTargets:
    logits: np.ndarray
    indices: np.ndarray
    reject_code: float


@dataclass(frozen=True)
class DirectThresholdSelection:
    selected_logit_threshold: float
    threshold_index: int | None
    reject: bool


def _scalar_text(value: Any, field: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{field} must be a scalar string")
    return str(array.item())


def _require_sha256(value: Any, field: str) -> str:
    text = str(value)
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def normalise_detector_checkpoint_sha256s(
    value: Any,
    *,
    field: str = "threshold_grid_detector_checkpoint_sha256s",
) -> tuple[str, ...]:
    """Normalise the NPZ/JSON encodings of the grid's detector-fold digest set."""

    raw = np.asarray(value)
    if raw.ndim == 0:
        scalar = raw.item()
        if isinstance(scalar, str):
            try:
                decoded = json.loads(scalar)
            except json.JSONDecodeError:
                decoded = [scalar]
        else:
            decoded = [scalar]
    elif raw.ndim == 1:
        decoded = raw.tolist()
    else:
        raise ValueError(f"{field} must be a scalar JSON list or one-dimensional array")
    if not isinstance(decoded, list) or not decoded:
        raise ValueError(f"{field} must contain at least one detector checkpoint")
    digests = tuple(_require_sha256(item, field) for item in decoded)
    if len(set(digests)) != len(digests):
        raise ValueError(f"{field} must not contain duplicate checkpoint digests")
    return digests


def validate_detector_role_contract(
    all_checkpoint_sha256s: Any,
    outer_checkpoint_sha256: Any,
    episode_checkpoint_sha256s: Any,
) -> tuple[tuple[str, ...], str, tuple[str, ...]]:
    """Validate the frozen outer-final/inner-episode detector role partition."""

    all_digests = normalise_detector_checkpoint_sha256s(
        all_checkpoint_sha256s,
        field="threshold_grid_detector_checkpoint_sha256s",
    )
    outer_digest = _require_sha256(
        outer_checkpoint_sha256,
        "threshold_grid_outer_detector_checkpoint_sha256",
    )
    episode_digests = normalise_detector_checkpoint_sha256s(
        episode_checkpoint_sha256s,
        field="threshold_grid_episode_detector_checkpoint_sha256s",
    )
    if len(all_digests) != 3 or len(episode_digests) != 2:
        raise ValueError(
            "RC-Direct requires exactly three grid detectors: one outer-final "
            "and two inner episode detectors"
        )
    if outer_digest not in all_digests:
        raise ValueError("Outer detector checkpoint is absent from the global grid")
    if outer_digest in episode_digests:
        raise ValueError("Outer detector cannot supervise source pseudo-target episodes")
    if set(episode_digests) != set(all_digests).difference({outer_digest}):
        raise ValueError(
            "Episode detector checkpoints must be exactly the two non-outer grid detectors"
        )
    return all_digests, outer_digest, episode_digests


def validate_joint_budget_pairs(
    pixel_budgets: Sequence[float],
    component_budgets: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate ordered joint budgets shared by RC-Direct and RiskCurve."""

    pixel = np.asarray(list(pixel_budgets), dtype=np.float64)
    component = np.asarray(list(component_budgets), dtype=np.float64)
    if pixel.ndim != 1 or component.ndim != 1 or pixel.size < 2:
        raise ValueError("At least two one-dimensional joint budget pairs are required")
    if pixel.shape != component.shape:
        raise ValueError("Pixel and component budget lists must have equal length")
    if (
        not np.isfinite(pixel).all()
        or not np.isfinite(component).all()
        or np.any(pixel <= 0.0)
        or np.any(component <= 0.0)
    ):
        raise ValueError("All joint budgets must be finite and positive")
    # The calibrator architecture orders thresholds from loose to strict.  Both
    # constraints therefore cannot loosen in the declared order.  A fixed
    # component budget across several strictly tighter pixel budgets is valid.
    if np.any(pixel[:-1] <= pixel[1:]) or np.any(component[:-1] < component[1:]):
        raise ValueError(
            "Pixel budgets must be strictly descending and component budgets "
            "must be non-increasing"
        )
    return pixel.astype(np.float32), component.astype(np.float32)


def direct_reject_code(thresholds: np.ndarray) -> float:
    """Return a finite training-only code for the external ``+inf`` action."""

    grid = validate_logit_threshold_grid(np.asarray(thresholds))
    spacing = max(
        float(grid[-1] - grid[-2]),
        float(np.finfo(np.float32).eps) * max(abs(float(grid[-1])), 1.0) * 16.0,
        1e-4,
    )
    code = float(grid[-1]) + spacing
    if not math.isfinite(code) or code <= float(grid[-1]):
        raise ValueError("Could not construct a finite RC-Direct reject code")
    return code


def derive_direct_threshold_targets(
    pixel_log_risk: np.ndarray,
    component_log_risk: np.ndarray,
    thresholds: np.ndarray,
    pixel_budgets: Sequence[float],
    component_budgets: Sequence[float],
) -> DirectThresholdTargets:
    """Convert future-E risk evidence into direct joint-budget supervision.

    Only the returned thresholds/indices are supervision.  Neither risk curve
    nor a mask is exposed to the model feature path.
    """

    grid = validate_logit_threshold_grid(np.asarray(thresholds))
    pixel_budget, component_budget = validate_joint_budget_pairs(
        pixel_budgets, component_budgets
    )
    pixel = np.asarray(pixel_log_risk, dtype=np.float64)
    component = np.asarray(component_log_risk, dtype=np.float64)
    expected_tail = (grid.size,)
    if pixel.ndim != 2 or pixel.shape[1:] != expected_tail:
        raise ValueError("pixel_log_risk must have shape [N,T]")
    if component.shape != pixel.shape:
        raise ValueError("pixel/component risk curves must have identical shape")
    if not np.isfinite(pixel).all() or not np.isfinite(component).all():
        raise ValueError("Direct supervision risk curves must be finite")
    if np.any(np.diff(pixel, axis=1) > 1e-6) or np.any(
        np.diff(component, axis=1) > 1e-6
    ):
        raise ValueError("Direct supervision risk curves must be non-increasing")

    num_rows = int(pixel.shape[0])
    num_budgets = int(pixel_budget.size)
    reject_code = direct_reject_code(grid)
    indices = np.full((num_rows, num_budgets), grid.size, dtype=np.int64)
    logits = np.full((num_rows, num_budgets), reject_code, dtype=np.float32)
    for budget_index, (pixel_limit, component_limit) in enumerate(
        zip(pixel_budget, component_budget)
    ):
        feasible = (pixel <= np.log10(float(pixel_limit))) & (
            component <= np.log10(float(component_limit))
        )
        has_feasible = feasible.any(axis=1)
        first = np.argmax(feasible, axis=1)
        indices[has_feasible, budget_index] = first[has_feasible]
        logits[has_feasible, budget_index] = grid[first[has_feasible]]
    if np.any(np.diff(indices, axis=1) < 0):
        raise ValueError(
            "Joint-budget oracle targets are not monotone from loose to strict"
        )
    return DirectThresholdTargets(logits=logits, indices=indices, reject_code=reject_code)


def quantize_direct_logit_threshold(
    predicted_logit: float,
    thresholds: np.ndarray,
) -> DirectThresholdSelection:
    """Conservatively snap a direct prediction to the shared finite grid."""

    grid = validate_logit_threshold_grid(np.asarray(thresholds))
    value = float(predicted_logit)
    if not math.isfinite(value):
        raise ValueError("RC-Direct prediction must be finite before action decoding")
    index = int(np.searchsorted(grid.astype(np.float64), value, side="left"))
    if index >= grid.size:
        return DirectThresholdSelection(
            selected_logit_threshold=EMPTY_ACTION_THRESHOLD,
            threshold_index=None,
            reject=True,
        )
    return DirectThresholdSelection(
        selected_logit_threshold=float(grid[index]),
        threshold_index=index,
        reject=False,
    )


def _load_provenance(archive: Mapping[str, np.ndarray], split: str) -> dict[str, Any]:
    if "provenance_json" not in archive:
        raise ValueError(f"{split} archive lacks provenance_json")
    try:
        payload = json.loads(_scalar_text(archive["provenance_json"], "provenance_json"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{split} provenance_json is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{split} provenance_json must decode to an object")
    return payload


def _audit_source_only_archive(
    archive: Mapping[str, np.ndarray],
    *,
    split: str,
) -> None:
    provenance = _load_provenance(archive, split)
    expected = {
        "protocol": "causal_adaptation_then_future_evaluation",
        "representation": LOGIT_REPRESENTATION,
        "pseudo_target_split": "train",
        "expected_split_role": "train",
        "fold_provenance_verified": True,
        "formal_causal_contract_verified": True,
        "protocol_scope": "formal_causal",
        "statistics_sample_role": "adaptation_window_A_label_free",
        "risk_label_sample_role": "immediately_following_evaluation_window_E",
        "threshold_grid_outer_target_excluded": True,
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ValueError(
                f"{split} RC-Direct source-only contract requires "
                f"{field}={value!r}"
            )
    if provenance.get("allow_unverified_fold_provenance") is True:
        raise ValueError(f"{split} permits unverified detector-fold provenance")
    if provenance.get("allow_cross_episode_role_reuse") is True:
        raise ValueError(f"{split} permits cross-episode A/E role reuse")
    pseudo_targets = provenance.get("pseudo_targets")
    outer_target = provenance.get("threshold_grid_outer_target_key")
    if (
        not isinstance(pseudo_targets, list)
        or not pseudo_targets
        or any(not isinstance(item, str) or not item for item in pseudo_targets)
    ):
        raise ValueError(f"{split} has no valid source pseudo-target declaration")
    if not isinstance(outer_target, str) or not outer_target:
        raise ValueError(f"{split} has no declared excluded outer target")
    normalise = lambda value: "".join(
        character for character in str(value).casefold() if character.isalnum()
    ).removesuffix("sirst")
    if normalise(outer_target) in {normalise(item) for item in pseudo_targets}:
        raise ValueError(f"{split} pseudo targets include the excluded outer target")
    if "pseudo_targets" not in archive:
        raise ValueError(f"{split} archive lacks row-level pseudo_targets")
    row_targets = tuple(
        str(item) for item in np.asarray(archive["pseudo_targets"]).reshape(-1).tolist()
    )
    if not row_targets or any(not item.strip() for item in row_targets):
        raise ValueError(f"{split} archive has invalid row-level pseudo targets")
    declared_keys = {normalise(item) for item in pseudo_targets}
    if any(normalise(item) not in declared_keys for item in row_targets):
        raise ValueError(f"{split} row-level pseudo target is undeclared")
    if any(normalise(item) == normalise(outer_target) for item in row_targets):
        raise ValueError(f"{split} row-level supervision includes the outer target")


def load_direct_training_pair(
    train_path: str | Path,
    validation_path: str | Path,
) -> DirectTrainingPair:
    """Load the exact v4 archives used by RiskCurve and bind fairness fields."""

    train_archive = load_curve_archive(train_path)
    validation_archive = load_curve_archive(validation_path)
    names = validate_archive_compatibility(train_archive, validation_archive)
    episode_contract = validate_training_episode_contract(
        train_archive,
        validation_archive,
        train_path=train_path,
        validation_path=validation_path,
    )
    if not bool(episode_contract.get("formal_protocol_eligible", False)):
        raise ValueError("RC-Direct v4 requires formal source-only causal episodes")
    detector_roles: dict[
        str, tuple[tuple[str, ...], str, tuple[str, ...]]
    ] = {}
    for split, archive in (
        ("train", train_archive),
        ("validation", validation_archive),
    ):
        representation = _scalar_text(archive["representation"], "representation")
        if representation != LOGIT_REPRESENTATION:
            raise ValueError("RC-Direct v4 requires raw_logit_float32 archives")
        detector_protocol = _scalar_text(
            archive.get("threshold_grid_detector_protocol"),
            "threshold_grid_detector_protocol",
        )
        if detector_protocol != ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL:
            raise ValueError(
                "RC-Direct v4 requires a grid fitted from all source-only detector folds"
            )
        role_contract = validate_detector_role_contract(
            archive.get("threshold_grid_detector_checkpoint_sha256s"),
            archive.get("threshold_grid_outer_detector_checkpoint_sha256"),
            archive.get("threshold_grid_episode_detector_checkpoint_sha256s"),
        )
        detector_digests, outer_detector_digest, episode_detector_digests = (
            role_contract
        )
        detector_roles[split] = role_contract
        provenance = _load_provenance(archive, split)
        if provenance.get("threshold_grid_detector_protocol") != detector_protocol:
            raise ValueError(f"{split} detector-grid protocol provenance mismatch")
        if tuple(
            provenance.get("threshold_grid_detector_checkpoint_sha256s", [])
        ) != detector_digests:
            raise ValueError(f"{split} detector-grid checkpoint provenance mismatch")
        if provenance.get(
            "threshold_grid_outer_detector_checkpoint_sha256"
        ) != outer_detector_digest:
            raise ValueError(f"{split} outer-detector provenance mismatch")
        if tuple(
            provenance.get(
                "threshold_grid_episode_detector_checkpoint_sha256s", []
            )
        ) != episode_detector_digests:
            raise ValueError(f"{split} episode-detector provenance mismatch")
        fold_audits = provenance.get("fold_provenance_audits")
        if not isinstance(fold_audits, list) or not fold_audits:
            raise ValueError(f"{split} lacks detector-fold episode provenance")
        observed_episode_digests = {
            str(audit.get("detector_weight_sha256"))
            for audit in fold_audits
            if isinstance(audit, dict) and audit.get("verified") is True
        }
        if observed_episode_digests != set(episode_detector_digests):
            raise ValueError(
                f"{split} episode artifacts do not use exactly the two inner detectors"
            )
        _audit_source_only_archive(archive, split=split)
    train_detector_protocol = _scalar_text(
        train_archive["threshold_grid_detector_protocol"],
        "train.threshold_grid_detector_protocol",
    )
    validation_detector_protocol = _scalar_text(
        validation_archive["threshold_grid_detector_protocol"],
        "validation.threshold_grid_detector_protocol",
    )
    train_detector_digests, train_outer_digest, train_episode_digests = (
        detector_roles["train"]
    )
    (
        validation_detector_digests,
        validation_outer_digest,
        validation_episode_digests,
    ) = detector_roles["validation"]
    if train_detector_protocol != validation_detector_protocol:
        raise ValueError("Train/validation detector-grid protocols differ")
    if train_detector_digests != validation_detector_digests:
        raise ValueError("Train/validation detector-grid checkpoint sets differ")
    if train_outer_digest != validation_outer_digest:
        raise ValueError("Train/validation outer detector checkpoints differ")
    if train_episode_digests != validation_episode_digests:
        raise ValueError("Train/validation episode detector checkpoints differ")
    episode_contract = dict(episode_contract)
    episode_contract.update(
        {
            "threshold_grid_detector_checkpoint_sha256s": list(
                train_detector_digests
            ),
            "threshold_grid_outer_detector_checkpoint_sha256": (
                train_outer_digest
            ),
            "threshold_grid_episode_detector_checkpoint_sha256s": list(
                train_episode_digests
            ),
        }
    )
    train_statistics = np.asarray(train_archive["statistics"], dtype=np.float32)
    mean = train_statistics.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_statistics.std(axis=0, dtype=np.float64).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("RC-Direct statistic normalisation is non-finite")
    return DirectTrainingPair(
        train_archive=train_archive,
        validation_archive=validation_archive,
        statistics_names=names,
        statistics_mean=mean,
        statistics_std=std,
        episode_contract=episode_contract,
        outer_detector_checkpoint_sha256=train_outer_digest,
        episode_detector_checkpoint_sha256s=train_episode_digests,
    )


def validate_direct_checkpoint_contract(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on every semantic field required by formal RC-Direct v4."""

    if not isinstance(checkpoint, Mapping):
        raise ValueError("RC-Direct checkpoint must be a mapping")
    required = {
        "checkpoint_schema_version",
        "kind",
        "method_name",
        "model_class",
        "role",
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
        "statistics_mean",
        "statistics_std",
        "budget_schema_version",
        "pixel_budgets",
        "component_budgets",
        "model_architecture_version",
        "model_config",
        "state_dict",
        "episode_contract",
        "target_label_policy",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError("RC-Direct v4 checkpoint is missing: " + ", ".join(missing))
    expected_scalars = {
        "checkpoint_schema_version": RC_DIRECT_CHECKPOINT_SCHEMA_VERSION,
        "kind": "calibrator",
        "method_name": "direct_threshold",
        "model_class": "MonotoneBudgetCalibrator",
        "role": "baseline",
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "statistics_schema_version": LOGIT_STATISTICS_SCHEMA_VERSION,
        "budget_schema_version": RC_DIRECT_BUDGET_SCHEMA_VERSION,
        "model_architecture_version": RC_DIRECT_ARCHITECTURE_VERSION,
        "threshold_grid_detector_protocol": ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL,
    }
    for field, expected in expected_scalars.items():
        if checkpoint.get(field) != expected:
            raise ValueError(f"RC-Direct checkpoint {field} must equal {expected!r}")
    grid = validate_logit_threshold_grid(
        np.asarray(checkpoint["thresholds"], dtype=np.float32)
    )
    semantic_hash = logit_threshold_grid_sha256(grid)
    if checkpoint["threshold_grid_sha256"] != semantic_hash:
        raise ValueError("RC-Direct checkpoint semantic threshold-grid hash mismatch")
    _require_sha256(
        checkpoint["threshold_grid_manifest_sha256"],
        "threshold_grid_manifest_sha256",
    )
    detector_digests, outer_detector_digest, episode_detector_digests = (
        validate_detector_role_contract(
            checkpoint["threshold_grid_detector_checkpoint_sha256s"],
            checkpoint["threshold_grid_outer_detector_checkpoint_sha256"],
            checkpoint["threshold_grid_episode_detector_checkpoint_sha256s"],
        )
    )
    names = validate_statistics_names(checkpoint["statistics_names"])
    if checkpoint["statistics_names_sha256"] != statistics_names_sha256(names):
        raise ValueError("RC-Direct checkpoint statistics_names hash mismatch")
    expected_feature_hash = feature_schema_sha256(
        LOGIT_STATISTICS_SCHEMA_VERSION, statistics_names=names
    )
    if checkpoint["feature_schema_sha256"] != expected_feature_hash:
        raise ValueError("RC-Direct checkpoint feature-schema hash mismatch")
    mean = np.asarray(checkpoint["statistics_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["statistics_std"], dtype=np.float32)
    if mean.shape != (len(names),) or std.shape != mean.shape:
        raise ValueError("RC-Direct checkpoint normalisation dimension mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0.0):
        raise ValueError("RC-Direct checkpoint normalisation must be finite")
    pixel_budget, component_budget = validate_joint_budget_pairs(
        checkpoint["pixel_budgets"], checkpoint["component_budgets"]
    )
    config = checkpoint["model_config"]
    if not isinstance(config, Mapping):
        raise ValueError("RC-Direct checkpoint model_config must be a mapping")
    if config.get("architecture_version") != RC_DIRECT_ARCHITECTURE_VERSION:
        raise ValueError("RC-Direct model/checkpoint architecture versions disagree")
    if config.get("representation") != LOGIT_REPRESENTATION:
        raise ValueError("RC-Direct model_config representation mismatch")
    config_grid = validate_logit_threshold_grid(
        np.asarray(config.get("threshold_grid"), dtype=np.float32)
    )
    if not np.array_equal(config_grid, grid):
        raise ValueError("RC-Direct model_config threshold grid mismatch")
    if not np.array_equal(
        np.asarray(config.get("budget_grid"), dtype=np.float32), pixel_budget
    ):
        raise ValueError("RC-Direct model_config budget grid mismatch")
    if int(config.get("feature_dim", -1)) != len(names):
        raise ValueError("RC-Direct model_config feature dimension mismatch")
    policy = checkpoint["target_label_policy"]
    expected_policy = {
        "model_inputs": "adaptation_window_A_label_free_statistics_only",
        "supervision": "source_official_train_future_E_risk_only",
        "outer_target_labels_used_for_features": False,
        "outer_target_labels_used_for_checkpoint_selection": False,
    }
    if policy != expected_policy:
        raise ValueError("RC-Direct checkpoint target-label policy mismatch")
    episode_contract = checkpoint["episode_contract"]
    if not isinstance(episode_contract, Mapping) or not bool(
        episode_contract.get("formal_protocol_eligible", False)
    ):
        raise ValueError("RC-Direct checkpoint lacks a formal causal episode contract")
    bound_episode_fields = {
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": semantic_hash,
        "threshold_grid_manifest_sha256": checkpoint[
            "threshold_grid_manifest_sha256"
        ],
        "threshold_grid_detector_protocol": ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL,
        "feature_schema_sha256": expected_feature_hash,
    }
    for field, expected in bound_episode_fields.items():
        if episode_contract.get(field) != expected:
            raise ValueError(
                f"RC-Direct checkpoint and episode contract differ in {field}"
            )
    if tuple(
        episode_contract.get("threshold_grid_detector_checkpoint_sha256s", [])
    ) != detector_digests:
        raise ValueError(
            "RC-Direct checkpoint and episode contract differ in detector hashes"
        )
    if episode_contract.get(
        "threshold_grid_outer_detector_checkpoint_sha256"
    ) != outer_detector_digest:
        raise ValueError(
            "RC-Direct checkpoint and episode contract differ in outer detector"
        )
    if tuple(
        episode_contract.get(
            "threshold_grid_episode_detector_checkpoint_sha256s", []
        )
    ) != episode_detector_digests:
        raise ValueError(
            "RC-Direct checkpoint and episode contract differ in episode detectors"
        )
    return {
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_sha256": semantic_hash,
        "threshold_grid_manifest_sha256": checkpoint[
            "threshold_grid_manifest_sha256"
        ],
        "threshold_grid_detector_protocol": ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": detector_digests,
        "threshold_grid_outer_detector_checkpoint_sha256": (
            outer_detector_digest
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": (
            episode_detector_digests
        ),
        "feature_schema_sha256": expected_feature_hash,
        "model_architecture_version": RC_DIRECT_ARCHITECTURE_VERSION,
        "pixel_budgets": pixel_budget,
        "component_budgets": component_budget,
        "formal_v4_eligible": True,
    }


__all__ = [
    "RC_DIRECT_BUDGET_SCHEMA_VERSION",
    "RC_DIRECT_CHECKPOINT_SCHEMA_VERSION",
    "RC_DIRECT_SELECTION_SCHEMA_VERSION",
    "DirectThresholdSelection",
    "DirectThresholdTargets",
    "DirectTrainingPair",
    "ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL",
    "derive_direct_threshold_targets",
    "direct_reject_code",
    "load_direct_training_pair",
    "normalise_detector_checkpoint_sha256s",
    "quantize_direct_logit_threshold",
    "validate_direct_checkpoint_contract",
    "validate_detector_role_contract",
    "validate_joint_budget_pairs",
]
