"""Fair source-only pseudo-target comparison for RiskCurve and RC-Direct v4.

Both methods consume the same label-free adaptation statistics and are decoded
onto the same finite raw-logit threshold grid (plus the external ``+inf``
reject action).  Labels enter only after action selection, through the
sufficient counts stored in the formal future-E episode archive.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from rc_irstd.models.calibrator import MonotoneBudgetCalibrator

from .curve_dataset import LOGIT_EPISODE_SCHEMA_VERSION, load_curve_archive
from .direct_calibrator import (
    ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL,
    quantize_direct_logit_threshold,
    validate_direct_checkpoint_contract,
    validate_detector_role_contract,
)
from .domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    statistics_names_sha256,
    validate_statistics_names,
)
from .monotone_curve_predictor import RiskCurvePredictor
from .representation import (
    EMPTY_ACTION_THRESHOLD,
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_REPRESENTATION,
    logit_threshold_grid_sha256,
    validate_logit_threshold_grid,
)
from .select_zero_label_threshold import select_dual_budget_threshold
from .train_curve_predictor import validate_curve_checkpoint_contract


SOURCE_PSEUDO_TARGET_COMPARISON_SCHEMA_VERSION = (
    "rc-v4-source-pseudo-target-comparison-v3-canonical-domain-scope"
)
FORMAL_SOURCE_DOMAIN_KEYS = frozenset(("irstd1k", "nudt"))
FORMAL_SOURCE_DOMAIN_NAMES = ("IRSTD-1K", "NUDT-SIRST")
FORMAL_OUTER_DOMAIN_KEY = "nuaa"
FORMAL_OUTER_DOMAIN_NAME = "NUAA-SIRST"


def _domain_key(value: Any) -> str:
    text = "".join(
        character for character in str(value).casefold() if character.isalnum()
    )
    if text.endswith("sirst"):
        text = text[: -len("sirst")]
    if not text:
        raise ValueError("Domain name normalises to an empty key")
    return text


def _scalar_text(value: Any, field: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{field} must be scalar")
    return str(array.item())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_device(device: str) -> torch.device:
    name = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if name == "auto":
        name = "cpu"
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _load_torch_mapping(path: Path, *, kind: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"{kind} checkpoint does not exist: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{kind} checkpoint must contain a mapping")
    return dict(payload), digest


def _load_provenance(archive: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if "provenance_json" not in archive:
        raise ValueError("Formal v4 episode archive is missing provenance_json")
    try:
        payload = json.loads(
            _scalar_text(archive["provenance_json"], "provenance_json")
        )
    except json.JSONDecodeError as error:
        raise ValueError("Episode archive provenance_json is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError("Episode archive provenance_json must decode to an object")
    return payload


def _validate_formal_archive(
    archive: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]:
    required = {
        "statistics",
        "statistics_names",
        "statistics_names_sha256",
        "statistics_schema_version",
        "feature_schema_sha256",
        "thresholds",
        "representation",
        "threshold_grid_schema_version",
        "threshold_grid_sha256",
        "threshold_grid_manifest_sha256",
        "threshold_grid_detector_protocol",
        "threshold_grid_detector_checkpoint_sha256s",
        "threshold_grid_outer_detector_checkpoint_sha256",
        "threshold_grid_episode_detector_checkpoint_sha256s",
        "episode_schema_version",
        "pixel_fp_counts",
        "component_fp_counts",
        "tp_object_counts",
        "gt_object_counts",
        "total_pixels",
        "pseudo_targets",
        "adaptation_ids",
        "evaluation_ids",
        "adaptation_sizes",
        "evaluation_sizes",
        "provenance_json",
    }
    missing = sorted(required.difference(archive))
    if missing:
        raise ValueError(
            "Formal v4 comparison archive is missing: " + ", ".join(missing)
        )
    if _scalar_text(archive["representation"], "representation") != LOGIT_REPRESENTATION:
        raise ValueError("Source pseudo-target comparison requires raw_logit_float32")
    if _scalar_text(
        archive["statistics_schema_version"], "statistics_schema_version"
    ) != LOGIT_STATISTICS_SCHEMA_VERSION:
        raise ValueError("Source pseudo-target statistics schema is incompatible")
    if _scalar_text(
        archive["threshold_grid_schema_version"],
        "threshold_grid_schema_version",
    ) != LOGIT_GRID_SCHEMA_VERSION:
        raise ValueError("Source pseudo-target threshold-grid schema is incompatible")
    if _scalar_text(
        archive["episode_schema_version"], "episode_schema_version"
    ) != LOGIT_EPISODE_SCHEMA_VERSION:
        raise ValueError("Source pseudo-target episode schema is not formal v4")
    thresholds = validate_logit_threshold_grid(
        np.asarray(archive["thresholds"], dtype=np.float32)
    )
    if _scalar_text(
        archive["threshold_grid_sha256"], "threshold_grid_sha256"
    ) != logit_threshold_grid_sha256(thresholds):
        raise ValueError("Episode archive semantic threshold-grid hash mismatch")
    names = validate_statistics_names(
        archive["statistics_names"],
        expected_dim=int(np.asarray(archive["statistics"]).shape[-1]),
    )
    if _scalar_text(
        archive["statistics_names_sha256"], "statistics_names_sha256"
    ) != statistics_names_sha256(names):
        raise ValueError("Episode archive ordered statistics-name hash mismatch")
    statistics = np.asarray(archive["statistics"], dtype=np.float32)
    if statistics.ndim != 2 or statistics.shape[0] == 0:
        raise ValueError("Episode statistics must have shape [N,D] with N > 0")
    if not np.isfinite(statistics).all():
        raise ValueError("Episode statistics contain NaN or infinity")

    rows, grid_size = statistics.shape[0], thresholds.size
    count_arrays: dict[str, np.ndarray] = {}
    for field in ("pixel_fp_counts", "component_fp_counts", "tp_object_counts"):
        raw = np.asarray(archive[field])
        if raw.shape != (rows, grid_size):
            raise ValueError(f"{field} must have shape [{rows},{grid_size}]")
        if raw.dtype.kind not in "iu" and not np.all(np.equal(raw, np.floor(raw))):
            raise ValueError(f"{field} must contain integer counts")
        values = raw.astype(np.int64)
        if np.any(values < 0):
            raise ValueError(f"{field} must contain non-negative counts")
        count_arrays[field] = values
    for field in ("gt_object_counts", "total_pixels"):
        raw = np.asarray(archive[field])
        if raw.shape != (rows,):
            raise ValueError(f"{field} must have shape [{rows}]")
        if raw.dtype.kind not in "iu" and not np.all(np.equal(raw, np.floor(raw))):
            raise ValueError(f"{field} must contain integer counts")
        values = raw.astype(np.int64)
        minimum = 1 if field == "total_pixels" else 0
        if np.any(values < minimum):
            raise ValueError(f"{field} contains an invalid exposure/count")
        count_arrays[field] = values
    if np.any(
        count_arrays["tp_object_counts"]
        > count_arrays["gt_object_counts"][:, None]
    ):
        raise ValueError("tp_object_counts cannot exceed gt_object_counts")

    pseudo_targets = np.asarray(archive["pseudo_targets"])
    adaptation_ids = np.asarray(archive["adaptation_ids"])
    evaluation_ids = np.asarray(archive["evaluation_ids"])
    if (
        pseudo_targets.shape != (rows,)
        or adaptation_ids.shape != (rows,)
        or evaluation_ids.shape != (rows,)
    ):
        raise ValueError(
            "pseudo_targets/adaptation_ids/evaluation_ids must have one value "
            "per episode"
        )
    if any(not str(item).strip() for item in pseudo_targets.tolist()):
        raise ValueError("Every episode must declare a source pseudo-target")

    provenance = _load_provenance(archive)
    if provenance.get("archive_split") != "validation":
        raise ValueError(
            "formal validation archive must declare archive_split='validation'"
        )
    expected_provenance = {
        "protocol": "causal_adaptation_then_future_evaluation",
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "pseudo_target_split": "train",
        "expected_split_role": "train",
        "fold_provenance_verified": True,
        "formal_causal_contract_verified": True,
        "protocol_scope": "formal_causal",
        "statistics_sample_role": "adaptation_window_A_label_free",
        "risk_label_sample_role": "immediately_following_evaluation_window_E",
        "threshold_grid_outer_target_excluded": True,
    }
    for field, expected in expected_provenance.items():
        if provenance.get(field) != expected:
            raise ValueError(
                f"Formal source-only archive requires provenance {field}={expected!r}"
            )
    decoded_adaptation = [
        _episode_identifier(value, index, field="adaptation_ids")
        for index, value in enumerate(adaptation_ids.tolist())
    ]
    decoded_evaluation = [
        _episode_identifier(value, index, field="evaluation_ids")
        for index, value in enumerate(evaluation_ids.tolist())
    ]
    size_contracts = (
        (
            "adaptation_sizes",
            decoded_adaptation,
            int(provenance["adaptation_window"]),
        ),
        (
            "evaluation_sizes",
            decoded_evaluation,
            int(provenance["evaluation_window"]),
        ),
    )
    for field, decoded, expected_size in size_contracts:
        raw_sizes = np.asarray(archive[field])
        if raw_sizes.shape != (rows,) or raw_sizes.dtype.kind not in "iu":
            raise ValueError(f"{field} must be an integer vector with one row per episode")
        if np.any(raw_sizes != expected_size):
            raise ValueError(f"{field} does not match the formal window contract")
        if any(len(identifiers) != expected_size for identifiers in decoded):
            raise ValueError(f"{field} does not match the decoded identifier counts")
    all_role_ids: set[str] = set()
    for row, (adaptation, evaluation) in enumerate(
        zip(decoded_adaptation, decoded_evaluation)
    ):
        adaptation_set = set(adaptation)
        evaluation_set = set(evaluation)
        if len(adaptation_set) != len(adaptation) or len(evaluation_set) != len(
            evaluation
        ):
            raise ValueError(f"Episode {row} contains duplicate A/E identifiers")
        if adaptation_set & evaluation_set:
            raise ValueError(f"Episode {row} reuses an identifier between A and E")
        row_ids = adaptation_set | evaluation_set
        if all_role_ids & row_ids:
            raise ValueError(
                "Formal non-overlapping causal episodes reuse an A/E identifier "
                "across episodes"
            )
        all_role_ids.update(row_ids)
    if provenance.get("cross_episode_role_reuse_detected") is not False:
        raise ValueError("Formal provenance must declare no cross-episode role reuse")
    if provenance.get("cross_episode_role_reuse_ids") != []:
        raise ValueError("Formal provenance contains cross-episode role-reuse IDs")
    if provenance.get("allow_unverified_fold_provenance") is True:
        raise ValueError("Formal source-only archive permits unverified fold provenance")
    if provenance.get("allow_cross_episode_role_reuse") is True:
        raise ValueError("Formal source-only archive permits A/E role reuse")
    outer_target = str(provenance.get("threshold_grid_outer_target_key", "")).strip()
    declared_targets = provenance.get("pseudo_targets")
    if not outer_target or not isinstance(declared_targets, list) or not declared_targets:
        raise ValueError("Formal source-only archive lacks target-scope provenance")
    declared_target_keys = {_domain_key(item) for item in declared_targets}
    if declared_target_keys != FORMAL_SOURCE_DOMAIN_KEYS:
        raise ValueError(
            "Formal Gate C archive source domains must be exactly IRSTD-1K and "
            "NUDT-SIRST"
        )
    if _domain_key(outer_target) != FORMAL_OUTER_DOMAIN_KEY:
        raise ValueError(
            "Formal Gate C excluded outer target must be canonical NUAA-SIRST"
        )
    source_domain_keys = {
        _domain_key(item)
        for item in provenance.get("threshold_grid_source_domains", [])
    }
    if source_domain_keys != FORMAL_SOURCE_DOMAIN_KEYS:
        raise ValueError(
            "Formal Gate C threshold grid must declare exactly the two canonical "
            "source domains"
        )
    paired_validation_keys = {
        _domain_key(item)
        for item in provenance.get("paired_lodo_validation_domains", [])
    }
    if paired_validation_keys != FORMAL_SOURCE_DOMAIN_KEYS:
        raise ValueError(
            "Formal Gate C archive must declare the canonical paired-LODO domain set"
        )
    if _domain_key(outer_target) in declared_target_keys:
        raise ValueError("Source pseudo-targets include the excluded outer target")
    if any(_domain_key(item) == _domain_key(outer_target) for item in pseudo_targets):
        raise ValueError("A row-level pseudo-target equals the excluded outer target")
    if any(_domain_key(item) not in declared_target_keys for item in pseudo_targets):
        raise ValueError("A row-level pseudo-target is absent from provenance")
    validation_domain = str(provenance.get("validation_domain", "")).strip()
    if _domain_key(validation_domain) not in FORMAL_SOURCE_DOMAIN_KEYS:
        raise ValueError("Formal validation domain is outside the canonical sources")
    if not validation_domain or any(
        _domain_key(item) != _domain_key(validation_domain) for item in pseudo_targets
    ):
        raise ValueError(
            "Formal validation archive rows must all equal provenance validation_domain"
        )

    all_hashes, outer_hash, episode_hashes = validate_detector_role_contract(
        archive["threshold_grid_detector_checkpoint_sha256s"],
        archive["threshold_grid_outer_detector_checkpoint_sha256"],
        archive["threshold_grid_episode_detector_checkpoint_sha256s"],
    )
    provenance_contract = {
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": _scalar_text(
            archive["threshold_grid_sha256"], "threshold_grid_sha256"
        ),
        "threshold_grid_manifest_sha256": _scalar_text(
            archive["threshold_grid_manifest_sha256"],
            "threshold_grid_manifest_sha256",
        ),
        "threshold_grid_detector_protocol": _scalar_text(
            archive["threshold_grid_detector_protocol"],
            "threshold_grid_detector_protocol",
        ),
        "threshold_grid_detector_checkpoint_sha256s": list(all_hashes),
        "threshold_grid_outer_detector_checkpoint_sha256": outer_hash,
        "threshold_grid_episode_detector_checkpoint_sha256s": list(episode_hashes),
        "feature_schema_sha256": _scalar_text(
            archive["feature_schema_sha256"], "feature_schema_sha256"
        ),
    }
    for field, expected in provenance_contract.items():
        if provenance.get(field) != expected:
            raise ValueError(f"Episode archive provenance {field} mismatch")
    fold_audits = provenance.get("fold_provenance_audits")
    if not isinstance(fold_audits, list) or len(fold_audits) != 2:
        raise ValueError("Formal validation archive requires exactly two detector-fold audits")
    if any(
        not isinstance(audit, dict) or audit.get("verified") is not True
        for audit in fold_audits
    ):
        raise ValueError("Every formal detector-fold audit must be verified")
    audit_domains = {
        _domain_key(audit.get("pseudo_target", "")) for audit in fold_audits
    }
    if audit_domains != FORMAL_SOURCE_DOMAIN_KEYS:
        raise ValueError(
            "Detector-fold audits must cover each canonical source domain exactly once"
        )
    observed_episode_hashes = {
        str(audit.get("detector_weight_sha256")) for audit in fold_audits
    }
    if observed_episode_hashes != set(episode_hashes):
        raise ValueError(
            "Formal validation archive fold audits do not cover both inner detectors"
        )
    if len(observed_episode_hashes) != len(fold_audits):
        raise ValueError("Detector-fold audit checkpoint hashes must be one-to-one")

    return thresholds, statistics, names, provenance


def _archive_contract(archive: Mapping[str, np.ndarray]) -> dict[str, Any]:
    all_hashes, outer_hash, episode_hashes = validate_detector_role_contract(
        archive["threshold_grid_detector_checkpoint_sha256s"],
        archive["threshold_grid_outer_detector_checkpoint_sha256"],
        archive["threshold_grid_episode_detector_checkpoint_sha256s"],
    )
    return {
        "representation": _scalar_text(archive["representation"], "representation"),
        "threshold_grid_schema_version": _scalar_text(
            archive["threshold_grid_schema_version"],
            "threshold_grid_schema_version",
        ),
        "threshold_grid_sha256": _scalar_text(
            archive["threshold_grid_sha256"], "threshold_grid_sha256"
        ),
        "threshold_grid_manifest_sha256": _scalar_text(
            archive["threshold_grid_manifest_sha256"],
            "threshold_grid_manifest_sha256",
        ),
        "threshold_grid_detector_protocol": _scalar_text(
            archive["threshold_grid_detector_protocol"],
            "threshold_grid_detector_protocol",
        ),
        "threshold_grid_detector_checkpoint_sha256s": all_hashes,
        "threshold_grid_outer_detector_checkpoint_sha256": outer_hash,
        "threshold_grid_episode_detector_checkpoint_sha256s": episode_hashes,
        "statistics_schema_version": _scalar_text(
            archive["statistics_schema_version"], "statistics_schema_version"
        ),
        "statistics_names": tuple(str(item) for item in archive["statistics_names"]),
        "statistics_names_sha256": _scalar_text(
            archive["statistics_names_sha256"], "statistics_names_sha256"
        ),
        "feature_schema_sha256": _scalar_text(
            archive["feature_schema_sha256"], "feature_schema_sha256"
        ),
    }


def _checkpoint_contract(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    all_hashes, outer_hash, episode_hashes = validate_detector_role_contract(
        checkpoint["threshold_grid_detector_checkpoint_sha256s"],
        checkpoint["threshold_grid_outer_detector_checkpoint_sha256"],
        checkpoint["threshold_grid_episode_detector_checkpoint_sha256s"],
    )
    return {
        "representation": str(checkpoint["representation"]),
        "threshold_grid_schema_version": str(
            checkpoint["threshold_grid_schema_version"]
        ),
        "threshold_grid_sha256": str(checkpoint["threshold_grid_sha256"]),
        "threshold_grid_manifest_sha256": str(
            checkpoint["threshold_grid_manifest_sha256"]
        ),
        "threshold_grid_detector_protocol": str(
            checkpoint["threshold_grid_detector_protocol"]
        ),
        "threshold_grid_detector_checkpoint_sha256s": all_hashes,
        "threshold_grid_outer_detector_checkpoint_sha256": outer_hash,
        "threshold_grid_episode_detector_checkpoint_sha256s": episode_hashes,
        "statistics_schema_version": str(checkpoint["statistics_schema_version"]),
        "statistics_names": tuple(str(item) for item in checkpoint["statistics_names"]),
        "statistics_names_sha256": str(checkpoint["statistics_names_sha256"]),
        "feature_schema_sha256": str(checkpoint["feature_schema_sha256"]),
    }


def _require_identical_contracts(
    archive_contract: Mapping[str, Any],
    risk_contract: Mapping[str, Any],
    direct_contract: Mapping[str, Any],
) -> None:
    fields = tuple(archive_contract)
    for field in fields:
        archive_value = archive_contract[field]
        if risk_contract.get(field) != archive_value:
            raise ValueError(f"RiskCurve/archive {field} mismatch")
        if direct_contract.get(field) != archive_value:
            raise ValueError(f"RC-Direct/archive {field} mismatch")
    if archive_contract["threshold_grid_detector_protocol"] != (
        ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL
    ):
        raise ValueError("Comparison archive does not use all source-only detector folds")


def _validate_episode_contract_binding(
    checkpoint: Mapping[str, Any],
    archive_contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    method: str,
) -> None:
    episode_contract = checkpoint.get("episode_contract")
    if not isinstance(episode_contract, Mapping):
        raise ValueError(f"{method} checkpoint lacks episode_contract")
    if not bool(episode_contract.get("formal_protocol_eligible", False)):
        raise ValueError(f"{method} checkpoint is not formal-protocol eligible")
    for field in (
        "representation",
        "threshold_grid_schema_version",
        "threshold_grid_sha256",
        "threshold_grid_manifest_sha256",
        "threshold_grid_detector_protocol",
        "feature_schema_sha256",
        "threshold_grid_outer_detector_checkpoint_sha256",
    ):
        if episode_contract.get(field) != archive_contract[field]:
            raise ValueError(f"{method} episode/archive {field} mismatch")
    for field in (
        "threshold_grid_detector_checkpoint_sha256s",
        "threshold_grid_episode_detector_checkpoint_sha256s",
    ):
        if tuple(episode_contract.get(field, [])) != tuple(archive_contract[field]):
            raise ValueError(f"{method} episode/archive {field} mismatch")
    for field in ("adaptation_window", "evaluation_window", "stride"):
        if field in episode_contract and int(episode_contract[field]) != int(
            provenance.get(field, -1)
        ):
            raise ValueError(f"{method} episode/archive {field} mismatch")
    protocol_fields = episode_contract.get("protocol_fields")
    if not isinstance(protocol_fields, Mapping):
        raise ValueError(f"{method} episode contract lacks protocol_fields")
    if {
        _domain_key(item) for item in protocol_fields.get("pseudo_targets", [])
    } != FORMAL_SOURCE_DOMAIN_KEYS:
        raise ValueError(f"{method} checkpoint source-domain scope is not canonical")
    if {
        _domain_key(item)
        for item in protocol_fields.get("threshold_grid_source_domains", [])
    } != FORMAL_SOURCE_DOMAIN_KEYS:
        raise ValueError(f"{method} checkpoint threshold-grid source scope is not canonical")
    if _domain_key(protocol_fields.get("validation_domain", "")) != _domain_key(
        provenance.get("validation_domain", "")
    ):
        raise ValueError(f"{method} checkpoint validation-domain binding mismatch")


def _checkpoint_training_binding(
    checkpoint: Mapping[str, Any],
    *,
    method: str,
) -> dict[str, int | str]:
    """Extract the reproducibility fields that make a comparison paired."""

    raw_seed = checkpoint.get("seed")
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, (int, np.integer)):
        raise ValueError(f"{method} checkpoint seed must be an integer")
    episode_contract = checkpoint.get("episode_contract")
    if not isinstance(episode_contract, Mapping):
        raise ValueError(f"{method} checkpoint lacks episode_contract")

    binding: dict[str, int | str] = {"seed": int(raw_seed)}
    for split in ("train", "validation"):
        split_contract = episode_contract.get(split)
        if not isinstance(split_contract, Mapping):
            raise ValueError(
                f"{method} checkpoint episode_contract lacks {split} binding"
            )
        digest = str(split_contract.get("archive_sha256", ""))
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                f"{method} checkpoint {split}.archive_sha256 is invalid"
            )
        binding[f"{split}_archive_sha256"] = digest
    return binding


def _require_identical_training_bindings(
    risk_checkpoint: Mapping[str, Any],
    direct_checkpoint: Mapping[str, Any],
    *,
    episode_archive_sha256: str,
) -> dict[str, int | str]:
    """Fail closed unless both methods are the same seeded train/val trial."""

    risk = _checkpoint_training_binding(risk_checkpoint, method="RiskCurve")
    direct = _checkpoint_training_binding(direct_checkpoint, method="RC-Direct")
    if risk["seed"] != direct["seed"]:
        raise ValueError("RiskCurve/RC-Direct checkpoint seed mismatch")
    for field in ("train_archive_sha256", "validation_archive_sha256"):
        if risk[field] != direct[field]:
            raise ValueError(
                f"RiskCurve/RC-Direct checkpoint {field} mismatch"
            )
    if risk["validation_archive_sha256"] != episode_archive_sha256:
        raise ValueError(
            "episode_file SHA-256 does not match the validation archive bound "
            "by both checkpoints"
        )
    return risk


def _registered_budget_indices(
    checkpoint_pixel: np.ndarray,
    checkpoint_component: np.ndarray,
    requested_pixel: np.ndarray,
    requested_component: np.ndarray,
) -> list[int]:
    indices: list[int] = []
    for pixel, component in zip(requested_pixel, requested_component):
        match = np.flatnonzero(
            np.isclose(checkpoint_pixel, pixel, rtol=1e-6, atol=0.0)
            & np.isclose(checkpoint_component, component, rtol=1e-6, atol=0.0)
        )
        if match.size != 1:
            raise ValueError(
                "Every requested joint budget pair must match exactly one "
                "registered RC-Direct checkpoint pair"
            )
        indices.append(int(match[0]))
    return indices


def _validate_requested_budget_pairs(
    pixel_budgets: Sequence[float],
    component_budgets: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    pixel = np.asarray(list(pixel_budgets), dtype=np.float64)
    component = np.asarray(list(component_budgets), dtype=np.float64)
    if pixel.ndim != 1 or component.ndim != 1 or pixel.size == 0:
        raise ValueError("At least one one-dimensional joint budget pair is required")
    if pixel.shape != component.shape:
        raise ValueError("Pixel and component budget lists must have equal length")
    if (
        not np.isfinite(pixel).all()
        or not np.isfinite(component).all()
        or np.any(pixel <= 0.0)
        or np.any(component <= 0.0)
    ):
        raise ValueError("All requested joint budgets must be finite and positive")
    if pixel.size > 1 and (
        np.any(pixel[:-1] <= pixel[1:])
        or np.any(component[:-1] < component[1:])
    ):
        raise ValueError(
            "Requested pixel budgets must be strictly descending and component "
            "budgets must be non-increasing"
        )
    return pixel.astype(np.float32), component.astype(np.float32)


def _episode_identifier(
    raw: Any, index: int, *, field: str = "evaluation_ids"
) -> list[str]:
    try:
        decoded = json.loads(str(raw))
    except json.JSONDecodeError as error:
        raise ValueError(f"{field}[{index}] is invalid JSON") from error
    if not isinstance(decoded, list) or not decoded or any(
        not isinstance(item, str) or not item for item in decoded
    ):
        raise ValueError(f"{field}[{index}] must be a non-empty string list")
    return decoded


def _action_evidence(
    archive: Mapping[str, np.ndarray],
    *,
    row: int,
    index: int | None,
    threshold: float,
    pixel_budget: float,
    component_budget: float,
) -> dict[str, Any]:
    total_pixels = int(np.asarray(archive["total_pixels"])[row])
    gt_objects = int(np.asarray(archive["gt_object_counts"])[row])
    reject = index is None
    if reject:
        pixel_fp = component_fp = tp_objects = 0
    else:
        pixel_fp = int(np.asarray(archive["pixel_fp_counts"])[row, index])
        component_fp = int(np.asarray(archive["component_fp_counts"])[row, index])
        tp_objects = int(np.asarray(archive["tp_object_counts"])[row, index])
    pixel_risk = pixel_fp / float(total_pixels)
    component_exposure = total_pixels / 1_000_000.0
    component_risk = component_fp / component_exposure
    pd = tp_objects / float(max(gt_objects, 1))
    pixel_excess = max(pixel_risk / pixel_budget - 1.0, 0.0)
    component_excess = max(component_risk / component_budget - 1.0, 0.0)
    return {
        "threshold_index": index,
        "selected_logit_threshold": "+inf" if reject else float(threshold),
        "reject": reject,
        "pixel_fp_count": pixel_fp,
        "component_fp_count": component_fp,
        "tp_object_count": tp_objects,
        "gt_object_count": gt_objects,
        "total_pixels": total_pixels,
        "pd": pd,
        "pixel_risk": pixel_risk,
        "component_risk": component_risk,
        "pixel_budget_violated": bool(pixel_risk > pixel_budget),
        "component_budget_violated": bool(component_risk > component_budget),
        "joint_budget_violated": bool(
            pixel_risk > pixel_budget or component_risk > component_budget
        ),
        "pixel_relative_excess": pixel_excess,
        "component_relative_excess": component_excess,
        "joint_relative_excess": max(pixel_excess, component_excess),
    }


def _aggregate_actions(actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not actions:
        raise ValueError("Cannot aggregate an empty action sequence")
    total_pixels = sum(int(action["total_pixels"]) for action in actions)
    total_gt = sum(int(action["gt_object_count"]) for action in actions)
    total_tp = sum(int(action["tp_object_count"]) for action in actions)
    total_pixel_fp = sum(int(action["pixel_fp_count"]) for action in actions)
    total_component_fp = sum(
        int(action["component_fp_count"]) for action in actions
    )
    relative_excess = np.asarray(
        [float(action["joint_relative_excess"]) for action in actions],
        dtype=np.float64,
    )
    finite_indices = sorted(
        {int(action["threshold_index"]) for action in actions if not action["reject"]}
    )
    return {
        "num_episodes": len(actions),
        "pd": total_tp / float(max(total_gt, 1)),
        "pixel_risk": total_pixel_fp / float(total_pixels),
        "component_risk": total_component_fp / (total_pixels / 1_000_000.0),
        "mean_episode_pd": float(np.mean([action["pd"] for action in actions])),
        "mean_episode_pixel_risk": float(
            np.mean([action["pixel_risk"] for action in actions])
        ),
        "mean_episode_component_risk": float(
            np.mean([action["component_risk"] for action in actions])
        ),
        "joint_violation_rate": float(
            np.mean([action["joint_budget_violated"] for action in actions])
        ),
        "mean_relative_excess": float(relative_excess.mean()),
        "max_relative_excess": float(relative_excess.max()),
        "reject_rate": float(np.mean([action["reject"] for action in actions])),
        "unique_finite_indices": finite_indices,
        "num_unique_finite_indices": len(finite_indices),
    }


def _relative_reduction(proposed: float, baseline: float) -> float | None:
    if baseline <= 0.0:
        return None
    return (baseline - proposed) / baseline


def _gate_decision(
    budgets: list[dict[str, Any]],
    *,
    risk_monotonic_violation_rate: float,
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    benefit = False
    pd_non_degraded = True
    for item in budgets:
        risk = item["methods"]["risk_curve"]
        direct = item["methods"]["rc_direct"]
        violation_reduction = _relative_reduction(
            risk["joint_violation_rate"], direct["joint_violation_rate"]
        )
        excess_reduction = _relative_reduction(
            risk["mean_relative_excess"], direct["mean_relative_excess"]
        )
        pd_delta = risk["pd"] - direct["pd"]
        benefit = benefit or (
            violation_reduction is not None and violation_reduction >= 0.20
        ) or (excess_reduction is not None and excess_reduction >= 0.20)
        pd_non_degraded = pd_non_degraded and pd_delta >= -0.02
        comparisons.append(
            {
                "pixel_budget": item["pixel_budget"],
                "component_budget": item["component_budget"],
                "risk_curve_minus_rc_direct_pd": pd_delta,
                "joint_violation_relative_reduction": violation_reduction,
                "mean_excess_relative_reduction": excess_reduction,
            }
        )
    strict_pd_gain = comparisons[-1]["risk_curve_minus_rc_direct_pd"] >= 0.03
    benefit = benefit or strict_pd_gain
    maximum_reject_rate = max(
        item["methods"]["risk_curve"]["reject_rate"] for item in budgets
    )
    reject_rate_acceptable = maximum_reject_rate < 0.20
    monotonic = risk_monotonic_violation_rate == 0.0
    go = benefit and pd_non_degraded and reject_rate_acceptable and monotonic
    return {
        "decision": "GO" if go else "HOLD",
        "scope": "source_only_pseudo_target_method_comparison",
        "not_an_outer_target_claim": True,
        "criteria": {
            "benefit": benefit,
            "benefit_rule": (
                "violation relative reduction >=20% OR mean excess relative "
                "reduction >=20% OR strictest-budget Pd gain >=3 percentage points"
            ),
            "pd_non_degraded": pd_non_degraded,
            "pd_non_degradation_rule": "RiskCurve Pd drop <=2 percentage points at every pair",
            "risk_curve_reject_rate_acceptable": reject_rate_acceptable,
            "risk_curve_reject_rate_rule": "maximum per-budget reject rate <20%",
            "risk_curve_mvr_zero": monotonic,
        },
        "maximum_risk_curve_reject_rate": maximum_reject_rate,
        "risk_curve_monotonic_violation_rate": risk_monotonic_violation_rate,
        "strictest_budget_pd_gain": comparisons[-1][
            "risk_curve_minus_rc_direct_pd"
        ],
        "budget_comparisons": comparisons,
    }


def evaluate_source_pseudo_target_comparison(
    *,
    episode_file: str | Path,
    risk_curve_checkpoint: str | Path,
    rc_direct_checkpoint: str | Path,
    output: str | Path,
    pixel_budgets: Sequence[float] | None = None,
    component_budgets: Sequence[float] | None = None,
    device: str = "auto",
    batch_size: int = 64,
) -> Path:
    """Run a fair, label-after-selection comparison and write its JSON record."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    torch_device = _resolve_device(device)
    episode_path = Path(episode_file).expanduser().resolve()
    risk_path = Path(risk_curve_checkpoint).expanduser().resolve()
    direct_path = Path(rc_direct_checkpoint).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not episode_path.is_file():
        raise FileNotFoundError(f"Episode archive does not exist: {episode_path}")

    episode_raw = episode_path.read_bytes()
    episode_archive_sha256 = hashlib.sha256(episode_raw).hexdigest()
    archive = load_curve_archive(io.BytesIO(episode_raw))
    thresholds, statistics, names, provenance = _validate_formal_archive(archive)
    risk_checkpoint, risk_checkpoint_sha256 = _load_torch_mapping(
        risk_path, kind="RiskCurve"
    )
    direct_checkpoint, direct_checkpoint_sha256 = _load_torch_mapping(
        direct_path, kind="RC-Direct"
    )
    risk_validated = validate_curve_checkpoint_contract(risk_checkpoint)
    direct_validated = validate_direct_checkpoint_contract(direct_checkpoint)
    if not bool(risk_validated.get("formal_v4_eligible", False)):
        raise ValueError("RiskCurve checkpoint is not formal v4 eligible")
    if not bool(direct_validated.get("formal_v4_eligible", False)):
        raise ValueError("RC-Direct checkpoint is not formal v4 eligible")
    training_binding = _require_identical_training_bindings(
        risk_checkpoint,
        direct_checkpoint,
        episode_archive_sha256=episode_archive_sha256,
    )

    archive_contract = _archive_contract(archive)
    risk_contract = _checkpoint_contract(risk_checkpoint)
    direct_contract = _checkpoint_contract(direct_checkpoint)
    _require_identical_contracts(archive_contract, risk_contract, direct_contract)
    _validate_episode_contract_binding(
        risk_checkpoint,
        archive_contract,
        provenance,
        method="RiskCurve",
    )
    _validate_episode_contract_binding(
        direct_checkpoint,
        archive_contract,
        provenance,
        method="RC-Direct",
    )
    if not np.array_equal(
        thresholds,
        validate_logit_threshold_grid(
            np.asarray(risk_checkpoint["thresholds"], dtype=np.float32)
        ),
    ) or not np.array_equal(
        thresholds,
        validate_logit_threshold_grid(
            np.asarray(direct_checkpoint["thresholds"], dtype=np.float32)
        ),
    ):
        raise ValueError("Archive and checkpoints do not contain the identical grid")
    if tuple(names) != tuple(risk_checkpoint["statistics_names"]) or tuple(
        names
    ) != tuple(direct_checkpoint["statistics_names"]):
        raise ValueError("Archive and checkpoints have different ordered features")

    checkpoint_pixel = np.asarray(
        direct_validated["pixel_budgets"], dtype=np.float32
    )
    checkpoint_component = np.asarray(
        direct_validated["component_budgets"], dtype=np.float32
    )
    if pixel_budgets is None and component_budgets is None:
        requested_pixel = checkpoint_pixel
        requested_component = checkpoint_component
    elif pixel_budgets is None or component_budgets is None:
        raise ValueError("pixel_budgets and component_budgets must be supplied together")
    else:
        requested_pixel, requested_component = _validate_requested_budget_pairs(
            pixel_budgets, component_budgets
        )
    direct_budget_indices = _registered_budget_indices(
        checkpoint_pixel,
        checkpoint_component,
        requested_pixel,
        requested_component,
    )

    risk_model = RiskCurvePredictor(**risk_checkpoint["model_config"])
    risk_model.load_state_dict(risk_checkpoint["state_dict"])
    risk_model.to(torch_device).eval()
    direct_model = MonotoneBudgetCalibrator(**direct_checkpoint["model_config"])
    direct_model.load_state_dict(direct_checkpoint["state_dict"])
    direct_model.to(torch_device).eval()
    risk_mean = np.asarray(risk_checkpoint["statistics_mean"], dtype=np.float32)
    risk_std = np.asarray(risk_checkpoint["statistics_std"], dtype=np.float32)
    if risk_mean.shape != (statistics.shape[1],) or risk_std.shape != risk_mean.shape:
        raise ValueError("RiskCurve checkpoint normalisation dimension mismatch")
    if (
        not np.isfinite(risk_mean).all()
        or not np.isfinite(risk_std).all()
        or np.any(risk_std < 0.0)
    ):
        raise ValueError("RiskCurve checkpoint normalisation is invalid")
    normalised_statistics = (statistics - risk_mean) / np.maximum(risk_std, 1e-6)

    risk_pixel_batches: list[np.ndarray] = []
    risk_component_batches: list[np.ndarray] = []
    direct_logit_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, statistics.shape[0], batch_size):
            stop = min(start + batch_size, statistics.shape[0])
            risk_output = risk_model(
                torch.from_numpy(normalised_statistics[start:stop]).to(torch_device)
            )
            direct_output = direct_model(
                torch.from_numpy(statistics[start:stop]).to(torch_device)
            )
            risk_pixel_batches.append(
                risk_output["pixel_log_risk"].detach().cpu().numpy()
            )
            risk_component_batches.append(
                risk_output["component_log_risk"].detach().cpu().numpy()
            )
            direct_logit_batches.append(
                direct_output.grid_logits.detach().cpu().numpy()
            )
    risk_pixel = np.concatenate(risk_pixel_batches, axis=0)
    risk_component = np.concatenate(risk_component_batches, axis=0)
    direct_logits = np.concatenate(direct_logit_batches, axis=0)
    if (
        risk_pixel.shape != (statistics.shape[0], thresholds.size)
        or risk_component.shape != risk_pixel.shape
        or direct_logits.shape[0] != statistics.shape[0]
        or not np.isfinite(risk_pixel).all()
        or not np.isfinite(risk_component).all()
        or not np.isfinite(direct_logits).all()
    ):
        raise ValueError("A comparison model emitted invalid predictions")

    adaptation_ids = [
        _episode_identifier(value, index, field="adaptation_ids")
        for index, value in enumerate(np.asarray(archive["adaptation_ids"]).tolist())
    ]
    evaluation_ids = [
        _episode_identifier(value, index)
        for index, value in enumerate(np.asarray(archive["evaluation_ids"]).tolist())
    ]
    row_targets = [str(value) for value in np.asarray(archive["pseudo_targets"]).tolist()]
    per_episode: list[dict[str, Any]] = [
        {
            "episode_index": row,
            "pseudo_target": row_targets[row],
            "adaptation_ids": adaptation_ids[row],
            "evaluation_ids": evaluation_ids[row],
            "actions": [],
        }
        for row in range(statistics.shape[0])
    ]
    budget_results: list[dict[str, Any]] = []
    action_codes: dict[str, list[list[int]]] = {
        "risk_curve": [[] for _ in range(statistics.shape[0])],
        "rc_direct": [[] for _ in range(statistics.shape[0])],
    }
    for budget_position, (pixel_budget, component_budget, direct_budget_index) in enumerate(
        zip(requested_pixel, requested_component, direct_budget_indices)
    ):
        method_actions: dict[str, list[dict[str, Any]]] = {
            "risk_curve": [],
            "rc_direct": [],
        }
        for row in range(statistics.shape[0]):
            risk_threshold, risk_reject, risk_index = select_dual_budget_threshold(
                thresholds,
                risk_pixel[row],
                risk_component[row],
                float(pixel_budget),
                float(component_budget),
                representation=LOGIT_REPRESENTATION,
            )
            direct_selection = quantize_direct_logit_threshold(
                float(direct_logits[row, direct_budget_index]), thresholds
            )
            selections = {
                "risk_curve": (
                    risk_index,
                    EMPTY_ACTION_THRESHOLD if risk_reject else risk_threshold,
                ),
                "rc_direct": (
                    direct_selection.threshold_index,
                    direct_selection.selected_logit_threshold,
                ),
            }
            episode_budget_actions: dict[str, Any] = {
                "budget_position": budget_position,
                "pixel_budget": float(pixel_budget),
                "component_budget": float(component_budget),
                "methods": {},
            }
            for method, (index, threshold) in selections.items():
                action = _action_evidence(
                    archive,
                    row=row,
                    index=index,
                    threshold=float(threshold),
                    pixel_budget=float(pixel_budget),
                    component_budget=float(component_budget),
                )
                if method == "risk_curve":
                    action.update(
                        {
                            "selection_rule": "earliest_jointly_feasible_grid_index",
                            "predicted_pixel_log_risk_at_action": (
                                None
                                if index is None
                                else float(risk_pixel[row, index])
                            ),
                            "predicted_component_log_risk_at_action": (
                                None
                                if index is None
                                else float(risk_component[row, index])
                            ),
                        }
                    )
                else:
                    action.update(
                        {
                            "selection_rule": "conservative_left_grid_quantization",
                            "predicted_logit_before_quantization": float(
                                direct_logits[row, direct_budget_index]
                            ),
                            "registered_budget_index": direct_budget_index,
                        }
                    )
                method_actions[method].append(action)
                episode_budget_actions["methods"][method] = action
                action_codes[method][row].append(
                    thresholds.size if index is None else int(index)
                )
            per_episode[row]["actions"].append(episode_budget_actions)
        budget_results.append(
            {
                "budget_position": budget_position,
                "pixel_budget": float(pixel_budget),
                "component_budget": float(component_budget),
                "methods": {
                    method: _aggregate_actions(actions)
                    for method, actions in method_actions.items()
                },
            }
        )

    monotonic_violation_rates: dict[str, float] = {}
    for method, rows in action_codes.items():
        violations = sum(
            int(any(right < left for left, right in zip(row, row[1:])))
            for row in rows
        )
        monotonic_violation_rates[method] = violations / float(len(rows))
    gate = _gate_decision(
        budget_results,
        risk_monotonic_violation_rate=monotonic_violation_rates["risk_curve"],
    )
    payload = {
        "schema_version": SOURCE_PSEUDO_TARGET_COMPARISON_SCHEMA_VERSION,
        "protocol": "source_only_pseudo_target_fair_comparison",
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "labels_used_for_action_selection": False,
        "source_pseudo_target_labels_used_for_post_selection_evaluation": True,
        "outer_target_labels_used": False,
        "formal_source_domains": list(FORMAL_SOURCE_DOMAIN_NAMES),
        "excluded_outer_target": FORMAL_OUTER_DOMAIN_NAME,
        "validation_pseudo_target": str(provenance["validation_domain"]),
        "archive_split": str(provenance["archive_split"]),
        "episode_archive": str(episode_path),
        "episode_archive_sha256": episode_archive_sha256,
        "seed": training_binding["seed"],
        "train_archive_sha256": training_binding["train_archive_sha256"],
        "validation_archive_sha256": training_binding[
            "validation_archive_sha256"
        ],
        "risk_curve_checkpoint": str(risk_path),
        "risk_curve_checkpoint_sha256": risk_checkpoint_sha256,
        "rc_direct_checkpoint": str(direct_path),
        "rc_direct_checkpoint_sha256": direct_checkpoint_sha256,
        "device": str(torch_device),
        "num_episodes": int(statistics.shape[0]),
        "pseudo_targets": sorted(set(row_targets)),
        "threshold_grid_size": int(thresholds.size),
        "threshold_grid_sha256": archive_contract["threshold_grid_sha256"],
        "threshold_grid_manifest_sha256": archive_contract[
            "threshold_grid_manifest_sha256"
        ],
        "feature_schema_sha256": archive_contract["feature_schema_sha256"],
        "threshold_grid_detector_protocol": archive_contract[
            "threshold_grid_detector_protocol"
        ],
        "threshold_grid_detector_checkpoint_sha256s": list(
            archive_contract["threshold_grid_detector_checkpoint_sha256s"]
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": archive_contract[
            "threshold_grid_outer_detector_checkpoint_sha256"
        ],
        "threshold_grid_episode_detector_checkpoint_sha256s": list(
            archive_contract[
                "threshold_grid_episode_detector_checkpoint_sha256s"
            ]
        ),
        "monotonic_violation_rates": monotonic_violation_rates,
        "budgets": budget_results,
        "per_episode": per_episode,
        "gate": gate,
    }
    if _sha256_file(episode_path) != episode_archive_sha256:
        raise ValueError("Episode archive changed after its immutable byte snapshot")
    if _sha256_file(risk_path) != risk_checkpoint_sha256:
        raise ValueError("RiskCurve checkpoint changed after its immutable byte snapshot")
    if _sha256_file(direct_path) != direct_checkpoint_sha256:
        raise ValueError("RC-Direct checkpoint changed after its immutable byte snapshot")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-file", required=True)
    parser.add_argument("--risk-curve-checkpoint", required=True)
    parser.add_argument("--rc-direct-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pixel-budgets", nargs="+", type=float)
    parser.add_argument("--component-budgets", nargs="+", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = evaluate_source_pseudo_target_comparison(
        episode_file=args.episode_file,
        risk_curve_checkpoint=args.risk_curve_checkpoint,
        rc_direct_checkpoint=args.rc_direct_checkpoint,
        output=args.output,
        pixel_budgets=args.pixel_budgets,
        component_budgets=args.component_budgets,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps({"output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "SOURCE_PSEUDO_TARGET_COMPARISON_SCHEMA_VERSION",
    "build_parser",
    "evaluate_source_pseudo_target_comparison",
]
