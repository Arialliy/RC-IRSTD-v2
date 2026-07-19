"""Train a structurally monotone upper-quantile dual-risk predictor."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import warnings
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from .build_curve_episodes import EPISODE_SCHEMA_VERSION
from .curve_dataset import (
    CurveDataset,
    load_curve_archive,
    validate_archive_compatibility,
    validate_count_all_adaptation_contract,
)
from .curve_metrics import curve_regression_metrics
from .domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    STATISTICS_SCHEMA_VERSION,
    feature_schema_sha256,
    statistics_names_sha256,
)
from .monotone_curve_predictor import (
    RISK_CURVE_ARCHITECTURE_VERSION,
    RiskCurvePredictor,
)
from .quantile_loss import pinball_loss
from .representation import (
    GRID_DETECTOR_PROTOCOL,
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_REPRESENTATION,
    PROBABILITY_REPRESENTATION,
    logit_threshold_grid_sha256,
    validate_logit_threshold_grid,
)
from .threshold_grid import (
    threshold_grid_sha256,
    threshold_grid_version,
    validate_threshold_grid,
)


TRAINING_EPISODE_CONTRACT_SCHEMA_VERSION = "rc-v2-training-episode-contract-v1"
LOGIT_EPISODE_SCHEMA_VERSION = "rc-v4-curve-episode-v1-raw-logit-causal"
_EPISODE_CONTRACT_KEYS = {
    "episode_schema_version",
    "adaptation_sizes",
    "evaluation_sizes",
    "adaptation_ids",
    "evaluation_ids",
    "provenance_json",
}


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scalar_text(value: np.ndarray, field: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{field} must be a scalar string")
    return str(array.item())


def _distinct_checkpoint_hashes(value: Any, field: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{field} must be a non-empty one-dimensional sequence")
    hashes = tuple(str(item) for item in array.tolist())
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"{field} must contain distinct checkpoint hashes")
    for index, digest in enumerate(hashes):
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                f"{field}[{index}] must be a lowercase SHA-256 digest"
            )
    return hashes


def _checkpoint_sha256(value: Any, field: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _detector_role_checkpoint_hashes(
    values: Mapping[str, Any],
    *,
    field_prefix: str = "",
) -> tuple[tuple[str, ...], str, tuple[str, ...]]:
    all_hashes = _distinct_checkpoint_hashes(
        values["threshold_grid_detector_checkpoint_sha256s"],
        f"{field_prefix}threshold_grid_detector_checkpoint_sha256s",
    )
    outer_hash = _checkpoint_sha256(
        values["threshold_grid_outer_detector_checkpoint_sha256"],
        f"{field_prefix}threshold_grid_outer_detector_checkpoint_sha256",
    )
    episode_hashes = _distinct_checkpoint_hashes(
        values["threshold_grid_episode_detector_checkpoint_sha256s"],
        f"{field_prefix}threshold_grid_episode_detector_checkpoint_sha256s",
    )
    if outer_hash in set(episode_hashes):
        raise ValueError("outer detector checkpoint must not be an episode detector")
    if set(all_hashes) != set(episode_hashes).union({outer_hash}):
        raise ValueError(
            "Global-grid detector hashes must equal the outer detector plus all "
            "episode detector hashes"
        )
    if len(episode_hashes) >= len(all_hashes):
        raise ValueError(
            "Episode detector checkpoint hashes must be a true subset of the "
            "global-grid detector hashes"
        )
    return all_hashes, outer_hash, episode_hashes


def validate_curve_checkpoint_contract(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate representation-bound checkpoint metadata.

    Checkpoints created before the v4 migration did not declare a score
    representation.  Such checkpoints remain identifiable as probability-grid
    diagnostics, but they are never upgraded to formal v4 eligibility by
    inference.  An explicitly raw-logit checkpoint, in contrast, must contain
    every schema/hash field and is checked fail-closed.
    """

    if not isinstance(checkpoint, Mapping):
        raise ValueError("Curve checkpoint must be a mapping")
    raw_thresholds = checkpoint.get("thresholds")
    if raw_thresholds is None:
        raise ValueError("Curve checkpoint is missing thresholds")

    explicit_representation = "representation" in checkpoint
    representation = str(
        checkpoint.get("representation", PROBABILITY_REPRESENTATION)
    )
    if representation not in {PROBABILITY_REPRESENTATION, LOGIT_REPRESENTATION}:
        raise ValueError(f"Unsupported checkpoint representation {representation!r}")

    if not explicit_representation:
        schema = checkpoint.get("statistics_schema_version")
        if schema == LOGIT_STATISTICS_SCHEMA_VERSION:
            raise ValueError(
                "Raw-logit statistics cannot be inferred for a legacy checkpoint"
            )
        thresholds = validate_threshold_grid(np.asarray(raw_thresholds))
        recorded_hash = checkpoint.get("threshold_grid_sha256")
        if recorded_hash is not None and str(recorded_hash) != threshold_grid_sha256(
            thresholds
        ):
            raise ValueError("Legacy checkpoint threshold-grid hash mismatch")
        return {
            "representation": PROBABILITY_REPRESENTATION,
            "legacy_checkpoint": True,
            "diagnostic_compatible": True,
            "formal_v4_eligible": False,
            "threshold_grid_schema_version": threshold_grid_version(thresholds),
            "threshold_grid_sha256": threshold_grid_sha256(thresholds),
        }

    required = {
        "threshold_grid_schema_version",
        "threshold_grid_sha256",
        "statistics_schema_version",
        "statistics_names",
        "statistics_names_sha256",
        "model_config",
        "model_architecture_version",
    }
    if representation == LOGIT_REPRESENTATION:
        required.add("feature_schema_sha256")
        required.add("threshold_grid_manifest_sha256")
        required.add("threshold_grid_detector_protocol")
        required.add("threshold_grid_detector_checkpoint_sha256s")
        required.add("threshold_grid_outer_detector_checkpoint_sha256")
        required.add("threshold_grid_episode_detector_checkpoint_sha256s")
        required.add("episode_contract")
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(
            "Representation-bound curve checkpoint is missing: "
            + ", ".join(missing)
        )

    names = tuple(str(item) for item in checkpoint["statistics_names"])
    if str(checkpoint["statistics_names_sha256"]) != statistics_names_sha256(names):
        raise ValueError("Curve checkpoint statistics_names hash mismatch")
    model_config = checkpoint["model_config"]
    if not isinstance(model_config, Mapping):
        raise ValueError("Curve checkpoint model_config must be a mapping")
    architecture = str(checkpoint["model_architecture_version"])
    if architecture != str(model_config.get("architecture_version")):
        raise ValueError("Curve checkpoint architecture versions disagree")
    if architecture != RISK_CURVE_ARCHITECTURE_VERSION:
        raise ValueError(f"Unsupported curve model architecture {architecture!r}")

    if representation == LOGIT_REPRESENTATION:
        thresholds = validate_logit_threshold_grid(
            np.asarray(raw_thresholds, dtype=np.float32)
        )
        grid_schema = str(checkpoint["threshold_grid_schema_version"])
        if grid_schema != LOGIT_GRID_SCHEMA_VERSION:
            raise ValueError("Raw-logit checkpoint grid schema is incompatible")
        semantic_hash = logit_threshold_grid_sha256(
            thresholds,
            schema_version=grid_schema,
            representation=representation,
        )
        if str(checkpoint["statistics_schema_version"]) != (
            LOGIT_STATISTICS_SCHEMA_VERSION
        ):
            raise ValueError("Raw-logit checkpoint statistics schema is incompatible")
        if str(checkpoint["feature_schema_sha256"]) != feature_schema_sha256(
            LOGIT_STATISTICS_SCHEMA_VERSION,
            statistics_names=names,
        ):
            raise ValueError("Raw-logit checkpoint feature-schema hash mismatch")
        manifest_hash = str(checkpoint["threshold_grid_manifest_sha256"])
        if len(manifest_hash) != 64 or any(
            character not in "0123456789abcdef" for character in manifest_hash
        ):
            raise ValueError("Raw-logit checkpoint grid-manifest hash is invalid")
        detector_protocol = str(checkpoint["threshold_grid_detector_protocol"])
        if detector_protocol != GRID_DETECTOR_PROTOCOL:
            raise ValueError("Raw-logit checkpoint grid detector protocol is invalid")
        detector_hashes, outer_detector_hash, episode_detector_hashes = (
            _detector_role_checkpoint_hashes(checkpoint)
        )
        episode_contract = checkpoint["episode_contract"]
        if not isinstance(episode_contract, Mapping):
            raise ValueError("Raw-logit checkpoint episode_contract must be a mapping")
        bound_fields: dict[str, Any] = {
            "representation": representation,
            "threshold_grid_schema_version": grid_schema,
            "threshold_grid_sha256": semantic_hash,
            "threshold_grid_manifest_sha256": manifest_hash,
            "threshold_grid_detector_protocol": detector_protocol,
            "threshold_grid_outer_detector_checkpoint_sha256": (
                outer_detector_hash
            ),
            "feature_schema_sha256": str(checkpoint["feature_schema_sha256"]),
        }
        for field, expected in bound_fields.items():
            if episode_contract.get(field) != expected:
                raise ValueError(
                    f"Raw-logit checkpoint and episode contract differ in {field}"
                )
        if tuple(
            episode_contract.get(
                "threshold_grid_detector_checkpoint_sha256s", []
            )
        ) != detector_hashes:
            raise ValueError(
                "Raw-logit checkpoint and episode contract differ in detector hashes"
            )
        if tuple(
            episode_contract.get(
                "threshold_grid_episode_detector_checkpoint_sha256s", []
            )
        ) != episode_detector_hashes:
            raise ValueError(
                "Raw-logit checkpoint and episode contract differ in episode "
                "detector hashes"
            )
    else:
        thresholds = validate_threshold_grid(np.asarray(raw_thresholds))
        grid_schema = str(checkpoint["threshold_grid_schema_version"])
        expected_grid_schema = threshold_grid_version(thresholds)
        if grid_schema != expected_grid_schema:
            raise ValueError("Probability checkpoint grid schema is incompatible")
        semantic_hash = threshold_grid_sha256(thresholds)
        if str(checkpoint["statistics_schema_version"]) != STATISTICS_SCHEMA_VERSION:
            raise ValueError("Probability checkpoint statistics schema is incompatible")
    if str(checkpoint["threshold_grid_sha256"]) != semantic_hash:
        raise ValueError("Curve checkpoint semantic threshold-grid hash mismatch")
    if int(model_config.get("num_thresholds", -1)) != int(thresholds.size):
        raise ValueError("Curve checkpoint model/grid dimensions disagree")
    return {
        "representation": representation,
        "legacy_checkpoint": False,
        "diagnostic_compatible": True,
        "formal_v4_eligible": representation == LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": grid_schema,
        "threshold_grid_sha256": semantic_hash,
        "threshold_grid_detector_protocol": (
            detector_protocol if representation == LOGIT_REPRESENTATION else None
        ),
        "threshold_grid_detector_checkpoint_sha256s": (
            list(detector_hashes) if representation == LOGIT_REPRESENTATION else None
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": (
            outer_detector_hash if representation == LOGIT_REPRESENTATION else None
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": (
            list(episode_detector_hashes)
            if representation == LOGIT_REPRESENTATION
            else None
        ),
        "model_architecture_version": architecture,
    }


def _constant_positive_size(
    value: np.ndarray,
    *,
    expected_rows: int,
    field: str,
) -> tuple[int, np.ndarray]:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.shape != (expected_rows,):
        raise ValueError(f"{field} must have shape [{expected_rows}]")
    if raw.dtype.kind not in "iu" and not np.all(np.equal(raw, np.floor(raw))):
        raise ValueError(f"{field} must contain integers")
    sizes = raw.astype(np.int64)
    if np.any(sizes <= 0):
        raise ValueError(f"{field} must contain positive values")
    unique = np.unique(sizes)
    if unique.size != 1:
        raise ValueError(
            f"{field} must be constant within one training archive; got {unique.tolist()}"
        )
    return int(unique[0]), sizes


def _decode_id_rows(
    value: np.ndarray,
    *,
    expected_rows: int,
    field: str,
) -> list[list[str]]:
    encoded = np.asarray(value)
    if encoded.ndim != 1 or encoded.shape != (expected_rows,):
        raise ValueError(f"{field} must contain one JSON list per episode")
    rows: list[list[str]] = []
    for row_index, item in enumerate(encoded.tolist()):
        try:
            decoded = json.loads(str(item))
        except json.JSONDecodeError as error:
            raise ValueError(f"{field}[{row_index}] is not valid JSON") from error
        if not isinstance(decoded, list) or not decoded:
            raise ValueError(f"{field}[{row_index}] must be a non-empty ID list")
        ids = [str(image_id) for image_id in decoded]
        if any(not image_id for image_id in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{field}[{row_index}] contains empty or duplicate IDs")
        rows.append(ids)
    return rows


def _archive_episode_contract(
    archive: dict[str, np.ndarray],
    *,
    archive_path: str | Path,
    split_name: str,
) -> dict[str, object]:
    """Validate one causal episode archive and return a JSON-safe contract."""

    present = _EPISODE_CONTRACT_KEYS.intersection(archive)
    missing = sorted(_EPISODE_CONTRACT_KEYS.difference(archive))
    if not present:
        return {
            "contract_present": False,
            "verified": False,
            "formal_protocol_eligible": False,
            "ineligibility_reasons": ["legacy_archive_missing_causal_episode_contract"],
            "missing_fields": missing,
            "archive": str(Path(archive_path).resolve()),
            "archive_sha256": _file_sha256(archive_path),
            "split": split_name,
        }
    if missing:
        raise ValueError(
            f"{split_name} archive has an incomplete causal contract; missing: "
            + ", ".join(missing)
        )

    representation = _scalar_text(archive["representation"], "representation")
    expected_episode_schema = (
        LOGIT_EPISODE_SCHEMA_VERSION
        if representation == LOGIT_REPRESENTATION
        else EPISODE_SCHEMA_VERSION
    )
    schema = _scalar_text(archive["episode_schema_version"], "episode_schema_version")
    if schema != expected_episode_schema:
        raise ValueError(
            f"{split_name} episode schema {schema!r} is incompatible with "
            f"representation {representation!r}; expected "
            f"{expected_episode_schema!r}"
        )
    num_episodes = int(np.asarray(archive["statistics"]).shape[0])
    adaptation_window, adaptation_sizes = _constant_positive_size(
        archive["adaptation_sizes"],
        expected_rows=num_episodes,
        field=f"{split_name}.adaptation_sizes",
    )
    evaluation_window, evaluation_sizes = _constant_positive_size(
        archive["evaluation_sizes"],
        expected_rows=num_episodes,
        field=f"{split_name}.evaluation_sizes",
    )
    adaptation_rows = _decode_id_rows(
        archive["adaptation_ids"],
        expected_rows=num_episodes,
        field=f"{split_name}.adaptation_ids",
    )
    evaluation_rows = _decode_id_rows(
        archive["evaluation_ids"],
        expected_rows=num_episodes,
        field=f"{split_name}.evaluation_ids",
    )
    for row_index, (adaptation_ids, evaluation_ids) in enumerate(
        zip(adaptation_rows, evaluation_rows)
    ):
        if len(adaptation_ids) != int(adaptation_sizes[row_index]):
            raise ValueError(f"{split_name} adaptation ID/size mismatch at episode {row_index}")
        if len(evaluation_ids) != int(evaluation_sizes[row_index]):
            raise ValueError(f"{split_name} evaluation ID/size mismatch at episode {row_index}")
        if set(adaptation_ids).intersection(evaluation_ids):
            raise ValueError(f"{split_name} episode {row_index} reuses an ID in A and E")

    provenance_text = _scalar_text(archive["provenance_json"], "provenance_json")
    try:
        provenance = json.loads(provenance_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{split_name} provenance_json is invalid") from error
    if not isinstance(provenance, dict):
        raise ValueError(f"{split_name} provenance_json must decode to an object")
    count_all_adaptation_contract = validate_count_all_adaptation_contract(
        archive, required=False
    )
    if provenance.get("protocol") != "causal_adaptation_then_future_evaluation":
        raise ValueError(f"{split_name} archive does not declare the causal A->E protocol")
    if int(provenance.get("adaptation_window", -1)) != adaptation_window:
        raise ValueError(f"{split_name} adaptation size disagrees with provenance")
    if int(provenance.get("evaluation_window", -1)) != evaluation_window:
        raise ValueError(f"{split_name} evaluation size disagrees with provenance")
    stride = int(provenance.get("stride", -1))
    if stride <= 0:
        raise ValueError(f"{split_name} provenance has an invalid stride")
    actual_grid_hash = _scalar_text(
        archive["threshold_grid_sha256"], "threshold_grid_sha256"
    )
    recorded_grid_hash = provenance.get("threshold_grid_sha256")
    if recorded_grid_hash != actual_grid_hash:
        raise ValueError(f"{split_name} provenance threshold-grid hash mismatch")
    grid_schema = _scalar_text(
        archive["threshold_grid_schema_version"],
        "threshold_grid_schema_version",
    )
    feature_hash = (
        _scalar_text(archive["feature_schema_sha256"], "feature_schema_sha256")
        if "feature_schema_sha256" in archive
        else None
    )
    grid_manifest_hash = (
        _scalar_text(
            archive["threshold_grid_manifest_sha256"],
            "threshold_grid_manifest_sha256",
        )
        if "threshold_grid_manifest_sha256" in archive
        else None
    )
    grid_detector_protocol = (
        _scalar_text(
            archive["threshold_grid_detector_protocol"],
            "threshold_grid_detector_protocol",
        )
        if representation == LOGIT_REPRESENTATION
        else None
    )
    grid_detector_roles = (
        _detector_role_checkpoint_hashes(archive)
        if representation == LOGIT_REPRESENTATION
        else ((), None, ())
    )
    grid_detector_hashes, grid_outer_detector_hash, grid_episode_detector_hashes = (
        grid_detector_roles
    )
    if representation == LOGIT_REPRESENTATION:
        if provenance.get("representation") != representation:
            raise ValueError(f"{split_name} provenance representation mismatch")
        if provenance.get("threshold_grid_schema_version") != grid_schema:
            raise ValueError(f"{split_name} provenance threshold-grid schema mismatch")
        if provenance.get("feature_schema_sha256") != feature_hash:
            raise ValueError(f"{split_name} provenance feature-schema hash mismatch")
        if provenance.get("threshold_grid_manifest_sha256") != grid_manifest_hash:
            raise ValueError(f"{split_name} provenance grid-manifest hash mismatch")
        if provenance.get("threshold_grid_detector_protocol") != (
            grid_detector_protocol
        ):
            raise ValueError(f"{split_name} provenance grid detector protocol mismatch")
        provenance_detector_hashes = provenance.get(
            "threshold_grid_detector_checkpoint_sha256s"
        )
        if not isinstance(provenance_detector_hashes, list) or tuple(
            provenance_detector_hashes
        ) != grid_detector_hashes:
            raise ValueError(
                f"{split_name} provenance grid detector checkpoint hashes mismatch"
            )
        if provenance.get(
            "threshold_grid_outer_detector_checkpoint_sha256"
        ) != grid_outer_detector_hash:
            raise ValueError(
                f"{split_name} provenance outer detector checkpoint hash mismatch"
            )
        provenance_episode_hashes = provenance.get(
            "threshold_grid_episode_detector_checkpoint_sha256s"
        )
        if not isinstance(provenance_episode_hashes, list) or tuple(
            provenance_episode_hashes
        ) != grid_episode_detector_hashes:
            raise ValueError(
                f"{split_name} provenance episode detector checkpoint hashes mismatch"
            )
        grid_source_domains = provenance.get("threshold_grid_source_domains")
        provenance_pseudo_targets = provenance.get("pseudo_targets")
        if (
            not isinstance(grid_source_domains, list)
            or len(grid_source_domains) != len(grid_episode_detector_hashes)
            or len(set(grid_source_domains)) != len(grid_source_domains)
            or any(
                not isinstance(value, str) or not value.strip()
                for value in grid_source_domains
            )
            or not isinstance(provenance_pseudo_targets, list)
            or len(provenance_pseudo_targets) != len(grid_episode_detector_hashes)
            or len(set(provenance_pseudo_targets))
            != len(provenance_pseudo_targets)
            or any(
                not isinstance(value, str) or not value.strip()
                for value in provenance_pseudo_targets
            )
        ):
            raise ValueError(
                f"{split_name} provenance must bind one distinct detector hash "
                "per source/pseudo-target domain"
            )

    all_adaptation_ids = {item for row in adaptation_rows for item in row}
    all_evaluation_ids = {item for row in evaluation_rows for item in row}
    role_overlap = sorted(all_adaptation_ids.intersection(all_evaluation_ids))
    diagnostic_reuse = bool(provenance.get("allow_cross_episode_role_reuse", False))
    if role_overlap and not diagnostic_reuse:
        raise ValueError(
            f"{split_name} has undeclared cross-episode A/E role reuse: "
            + ", ".join(role_overlap[:5])
        )
    if stride < adaptation_window + evaluation_window and not diagnostic_reuse:
        raise ValueError(
            f"{split_name} stride is shorter than A+E without diagnostic provenance"
        )
    if "pseudo_targets" in archive:
        raw_targets = np.asarray(archive["pseudo_targets"])
        if raw_targets.ndim != 1 or raw_targets.shape != (num_episodes,):
            raise ValueError(
                f"{split_name}.pseudo_targets must have shape [{num_episodes}]"
            )
        row_targets = [str(item) for item in raw_targets.tolist()]
        if any(not target.strip() for target in row_targets):
            raise ValueError(f"{split_name}.pseudo_targets contains an empty name")
    else:
        # Legacy archives lack row-level target names.  Use one conservative
        # namespace so duplicate IDs across train/validation cannot evade the
        # split guard merely because the archive labels differ.
        row_targets = [""] * num_episodes

    reasons: list[str] = []
    if diagnostic_reuse:
        reasons.append("cross_episode_role_reuse_explicitly_allowed")
    if role_overlap:
        reasons.append("cross_episode_role_reuse_detected")
    if stride < adaptation_window + evaluation_window:
        reasons.append("stride_shorter_than_A_plus_E")
    if not bool(provenance.get("fold_provenance_verified", False)):
        reasons.append("detector_fold_provenance_unverified")
    if bool(provenance.get("allow_unverified_fold_provenance", False)):
        reasons.append("unverified_fold_provenance_flag_used")
    if provenance.get("formal_causal_contract_verified") is False:
        reasons.append("episode_builder_marked_causal_contract_unverified")
    if bool(provenance.get("cross_episode_role_reuse_detected", False)):
        reasons.append("episode_builder_recorded_cross_episode_role_reuse")
    if provenance.get("protocol_scope") == "diagnostic_only":
        reasons.append("episode_builder_protocol_scope_is_diagnostic_only")
    reasons = list(dict.fromkeys(reasons))
    formal_eligible = not reasons
    protocol_fields = {
        key: provenance.get(key)
        for key in (
            "protocol",
            "adaptation_window",
            "evaluation_window",
            "stride",
            "matching_rule",
            "centroid_distance",
            "connectivity",
            "min_component_area",
            "source_reference",
            "source_reference_sha256",
            "source_reference_domain_names",
            "source_reference_statistics_names_sha256",
            "pseudo_targets",
            "validation_domain",
            "threshold_grid_sha256",
            "threshold_grid_schema_version",
            "representation",
            "feature_schema_sha256",
            "threshold_grid_manifest_sha256",
            "threshold_grid_detector_protocol",
            "threshold_grid_detector_checkpoint_sha256s",
            "threshold_grid_outer_detector_checkpoint_sha256",
            "threshold_grid_episode_detector_checkpoint_sha256s",
            "threshold_grid_source_domains",
            "count_all_adaptation_schema_version",
            "count_all_adaptation_sample_role",
            "count_all_adaptation_masks_read",
            "count_all_adaptation_prediction_rule",
            "count_all_adaptation_pixel_count_semantics",
            "count_all_adaptation_component_count_semantics",
            "count_all_adaptation_component_envelope",
        )
    }
    return {
        "contract_present": True,
        "verified": True,
        "formal_protocol_eligible": formal_eligible,
        "ineligibility_reasons": reasons,
        "episode_schema_version": schema,
        "representation": representation,
        "threshold_grid_schema_version": grid_schema,
        "threshold_grid_sha256": actual_grid_hash,
        "feature_schema_sha256": feature_hash,
        "threshold_grid_manifest_sha256": grid_manifest_hash,
        "threshold_grid_detector_protocol": grid_detector_protocol,
        "threshold_grid_detector_checkpoint_sha256s": list(grid_detector_hashes),
        "threshold_grid_outer_detector_checkpoint_sha256": (
            grid_outer_detector_hash
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": list(
            grid_episode_detector_hashes
        ),
        "adaptation_window": adaptation_window,
        "evaluation_window": evaluation_window,
        "stride": stride,
        "num_episodes": num_episodes,
        "risk_target_unit": f"aggregate_risk_over_{evaluation_window}_future_images",
        "one_to_one_future_target": evaluation_window == 1,
        "deployment_compatibility_rule": (
            "Direct risk-curve semantics require deployment A/E window sizes to "
            "match this training contract"
        ),
        "cross_episode_role_reuse_ids": role_overlap,
        "all_qualified_image_ids": sorted(
            {
                f"{target}:{image_id}"
                for target, adaptation_ids, evaluation_ids in zip(
                    row_targets,
                    adaptation_rows,
                    evaluation_rows,
                )
                for image_id in adaptation_ids + evaluation_ids
            }
        ),
        "protocol_fields": protocol_fields,
        "count_all_adaptation_contract": count_all_adaptation_contract,
        "provenance": provenance,
        "provenance_sha256": hashlib.sha256(provenance_text.encode("utf-8")).hexdigest(),
        "archive": str(Path(archive_path).resolve()),
        "archive_sha256": _file_sha256(archive_path),
        "split": split_name,
    }


def validate_training_episode_contract(
    train_archive: dict[str, np.ndarray],
    validation_archive: dict[str, np.ndarray],
    *,
    train_path: str | Path,
    validation_path: str | Path,
) -> dict[str, object]:
    """Validate and combine the train/validation causal data contracts."""

    train_representation = _scalar_text(
        train_archive["representation"], "train.representation"
    )
    validation_representation = _scalar_text(
        validation_archive["representation"], "validation.representation"
    )
    if train_representation != validation_representation:
        raise ValueError(
            "Train/validation causal contracts differ in representation"
        )
    train = _archive_episode_contract(
        train_archive, archive_path=train_path, split_name="train"
    )
    validation = _archive_episode_contract(
        validation_archive,
        archive_path=validation_path,
        split_name="validation",
    )
    if bool(train["contract_present"]) != bool(validation["contract_present"]):
        raise ValueError("Train and validation archives differ in causal-contract presence")
    if not train["contract_present"]:
        warnings.warn(
            "Curve archives lack the causal episode contract; training continues for "
            "legacy diagnostics, but the checkpoint is not formal-protocol eligible",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            "schema_version": TRAINING_EPISODE_CONTRACT_SCHEMA_VERSION,
            "verified": False,
            "formal_protocol_eligible": False,
            "ineligibility_reasons": ["legacy_archives_missing_causal_episode_contract"],
            "train": train,
            "validation": validation,
        }
    for field in (
        "episode_schema_version",
        "representation",
        "threshold_grid_schema_version",
        "threshold_grid_sha256",
        "feature_schema_sha256",
        "threshold_grid_manifest_sha256",
        "threshold_grid_detector_protocol",
        "threshold_grid_detector_checkpoint_sha256s",
        "threshold_grid_outer_detector_checkpoint_sha256",
        "threshold_grid_episode_detector_checkpoint_sha256s",
        "adaptation_window",
        "evaluation_window",
        "stride",
        "protocol_fields",
    ):
        if train[field] != validation[field]:
            raise ValueError(f"Train/validation causal contracts differ in {field}")
    count_all_semantic_fields = (
        "present",
        "verified",
        "schema_version",
        "representation",
        "threshold_grid_sha256",
        "num_thresholds",
        "connectivity",
        "min_component_area",
        "adaptation_masks_read",
        "prediction_rule",
    )
    for field in count_all_semantic_fields:
        if train["count_all_adaptation_contract"].get(field) != validation[
            "count_all_adaptation_contract"
        ].get(field):
            raise ValueError(
                "Train/validation Count-all adaptation contracts differ in "
                f"{field}"
            )
    train_ids = set(train["all_qualified_image_ids"])
    validation_ids = set(validation["all_qualified_image_ids"])
    split_overlap = sorted(train_ids.intersection(validation_ids))
    if split_overlap:
        raise ValueError(
            "Train/validation causal archives reuse image IDs: "
            + ", ".join(split_overlap[:10])
        )
    reasons = list(
        dict.fromkeys(
            list(train["ineligibility_reasons"])
            + list(validation["ineligibility_reasons"])
        )
    )
    return {
        "schema_version": TRAINING_EPISODE_CONTRACT_SCHEMA_VERSION,
        "verified": True,
        "formal_protocol_eligible": not reasons,
        "ineligibility_reasons": reasons,
        "episode_schema_version": train["episode_schema_version"],
        "representation": train["representation"],
        "threshold_grid_schema_version": train["threshold_grid_schema_version"],
        "threshold_grid_sha256": train["threshold_grid_sha256"],
        "feature_schema_sha256": train["feature_schema_sha256"],
        "threshold_grid_manifest_sha256": train[
            "threshold_grid_manifest_sha256"
        ],
        "threshold_grid_detector_protocol": train[
            "threshold_grid_detector_protocol"
        ],
        "threshold_grid_detector_checkpoint_sha256s": train[
            "threshold_grid_detector_checkpoint_sha256s"
        ],
        "threshold_grid_outer_detector_checkpoint_sha256": train[
            "threshold_grid_outer_detector_checkpoint_sha256"
        ],
        "threshold_grid_episode_detector_checkpoint_sha256s": train[
            "threshold_grid_episode_detector_checkpoint_sha256s"
        ],
        "adaptation_window": train["adaptation_window"],
        "evaluation_window": train["evaluation_window"],
        "stride": train["stride"],
        "risk_target_unit": train["risk_target_unit"],
        "one_to_one_future_target": train["one_to_one_future_target"],
        "deployment_compatibility_rule": train["deployment_compatibility_rule"],
        "protocol_fields": train["protocol_fields"],
        "count_all_adaptation_contract": {
            field: train["count_all_adaptation_contract"].get(field)
            for field in count_all_semantic_fields
        },
        "train_validation_image_id_overlap": [],
        "train": train,
        "validation": validation,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate(
    model,
    loader,
    device,
    *,
    quantile: float,
    lambda_component: float,
) -> tuple[float, dict[str, float]]:
    model.eval()
    pixel_predictions: list[np.ndarray] = []
    component_predictions: list[np.ndarray] = []
    pixel_targets: list[np.ndarray] = []
    component_targets: list[np.ndarray] = []
    pixel_pinball_sum = 0.0
    component_pinball_sum = 0.0
    pixel_elements = 0
    component_elements = 0
    with torch.no_grad():
        for batch in loader:
            statistics = batch["statistics"].to(device)
            output = model(statistics)
            pixel_target_tensor = batch["pixel_log_risk"].to(device)
            component_target_tensor = batch["component_log_risk"].to(device)
            pixel_pinball_sum += float(
                pinball_loss(
                    output["pixel_log_risk"],
                    pixel_target_tensor,
                    quantile,
                    reduction="sum",
                ).cpu()
            )
            component_pinball_sum += float(
                pinball_loss(
                    output["component_log_risk"],
                    component_target_tensor,
                    quantile,
                    reduction="sum",
                ).cpu()
            )
            pixel_elements += int(pixel_target_tensor.numel())
            component_elements += int(component_target_tensor.numel())
            pixel_predictions.append(output["pixel_log_risk"].cpu().numpy())
            component_predictions.append(output["component_log_risk"].cpu().numpy())
            pixel_targets.append(batch["pixel_log_risk"].numpy())
            component_targets.append(batch["component_log_risk"].numpy())
    pixel_prediction = np.concatenate(pixel_predictions)
    component_prediction = np.concatenate(component_predictions)
    pixel_target = np.concatenate(pixel_targets)
    component_target = np.concatenate(component_targets)
    pixel_metrics = curve_regression_metrics(pixel_prediction, pixel_target)
    component_metrics = curve_regression_metrics(component_prediction, component_target)
    metrics = {f"pixel_{key}": value for key, value in pixel_metrics.items()}
    metrics.update({f"component_{key}": value for key, value in component_metrics.items()})
    pixel_pinball = pixel_pinball_sum / pixel_elements
    component_pinball = component_pinball_sum / component_elements
    objective = pixel_pinball + lambda_component * component_pinball
    metrics.update(
        {
            "pixel_pinball_loss": pixel_pinball,
            "component_pinball_loss": component_pinball,
            "quantile_pinball_objective": objective,
        }
    )
    return objective, metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--lambda-component", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device (for example auto, cpu, cuda, or cuda:1)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0.0 < args.quantile < 1.0:
        raise ValueError("--quantile must lie strictly between 0 and 1")
    if args.lambda_component < 0.0:
        raise ValueError("--lambda-component must be non-negative")
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("--epochs, --batch-size, and --patience must be positive")
    if args.lr <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("--lr must be positive and --weight-decay non-negative")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    _seed_everything(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)

    train_archive = load_curve_archive(args.train_file)
    val_archive = load_curve_archive(args.val_file)
    statistics_names = validate_archive_compatibility(train_archive, val_archive)
    episode_contract = validate_training_episode_contract(
        train_archive,
        val_archive,
        train_path=args.train_file,
        validation_path=args.val_file,
    )
    thresholds = np.asarray(train_archive["thresholds"], dtype=np.float32)
    representation = _scalar_text(train_archive["representation"], "representation")
    threshold_grid_schema_version = _scalar_text(
        train_archive["threshold_grid_schema_version"],
        "threshold_grid_schema_version",
    )
    semantic_grid_hash = _scalar_text(
        train_archive["threshold_grid_sha256"], "threshold_grid_sha256"
    )
    feature_schema_hash = (
        _scalar_text(train_archive["feature_schema_sha256"], "feature_schema_sha256")
        if "feature_schema_sha256" in train_archive
        else None
    )
    threshold_grid_manifest_hash = (
        _scalar_text(
            train_archive["threshold_grid_manifest_sha256"],
            "threshold_grid_manifest_sha256",
        )
        if "threshold_grid_manifest_sha256" in train_archive
        else None
    )
    threshold_grid_detector_protocol = (
        _scalar_text(
            train_archive["threshold_grid_detector_protocol"],
            "threshold_grid_detector_protocol",
        )
        if representation == LOGIT_REPRESENTATION
        else None
    )
    threshold_grid_detector_roles = (
        _detector_role_checkpoint_hashes(train_archive)
        if representation == LOGIT_REPRESENTATION
        else ((), None, ())
    )
    (
        threshold_grid_detector_checkpoint_hashes,
        threshold_grid_outer_detector_checkpoint_hash,
        threshold_grid_episode_detector_checkpoint_hashes,
    ) = threshold_grid_detector_roles
    statistics_schema_version = _scalar_text(
        train_archive["statistics_schema_version"], "statistics_schema_version"
    )
    train_statistics = np.asarray(train_archive["statistics"], dtype=np.float32)
    statistics_mean = train_statistics.mean(axis=0)
    statistics_std = train_statistics.std(axis=0)
    train_set = CurveDataset(args.train_file, statistics_mean, statistics_std)
    val_set = CurveDataset(args.val_file, statistics_mean, statistics_std)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = RiskCurvePredictor(
        input_dim=train_statistics.shape[1],
        num_thresholds=thresholds.size,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_objective = float("inf")
    stale_epochs = 0
    history: list[dict[str, object]] = []

    for epoch in range(args.epochs):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            statistics = batch["statistics"].to(device)
            target_pixel = batch["pixel_log_risk"].to(device)
            target_component = batch["component_log_risk"].to(device)
            prediction = model(statistics)
            pixel_loss = pinball_loss(prediction["pixel_log_risk"], target_pixel, args.quantile)
            component_loss = pinball_loss(
                prediction["component_log_risk"], target_component, args.quantile
            )
            loss = pixel_loss + args.lambda_component * component_loss
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach().cpu()))
        objective, metrics = _evaluate(
            model,
            val_loader,
            device,
            quantile=args.quantile,
            lambda_component=args.lambda_component,
        )
        record: dict[str, object] = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_objective": objective,
            **metrics,
        }
        history.append(record)
        # Keep the complete per-threshold diagnostics in the metrics artifact,
        # but avoid emitting hundreds of array entries on every training epoch.
        console_record = {
            key: value for key, value in record.items() if not isinstance(value, list)
        }
        console_record["per_threshold_metrics_recorded"] = any(
            isinstance(value, list) for value in record.values()
        )
        print(json.dumps(console_record, sort_keys=True))
        if objective < best_objective - 1e-8:
            best_objective = objective
            stale_epochs = 0
            checkpoint = {
                "method_name": "risk_curve",
                "model_class": "RiskCurvePredictor",
                "model_architecture_version": RISK_CURVE_ARCHITECTURE_VERSION,
                "role": "proposed_method",
                "state_dict": model.state_dict(),
                "model_config": model.config(),
                "thresholds": thresholds.tolist(),
                "representation": representation,
                "threshold_grid_schema_version": threshold_grid_schema_version,
                "threshold_grid_version": (
                    LOGIT_GRID_SCHEMA_VERSION
                    if representation == LOGIT_REPRESENTATION
                    else threshold_grid_version(thresholds)
                ),
                "threshold_grid_sha256": semantic_grid_hash,
                "feature_schema_sha256": feature_schema_hash,
                "threshold_grid_manifest_sha256": threshold_grid_manifest_hash,
                "threshold_grid_detector_protocol": (
                    threshold_grid_detector_protocol
                ),
                "threshold_grid_detector_checkpoint_sha256s": list(
                    threshold_grid_detector_checkpoint_hashes
                ),
                "threshold_grid_outer_detector_checkpoint_sha256": (
                    threshold_grid_outer_detector_checkpoint_hash
                ),
                "threshold_grid_episode_detector_checkpoint_sha256s": list(
                    threshold_grid_episode_detector_checkpoint_hashes
                ),
                "statistics_mean": statistics_mean.tolist(),
                "statistics_std": statistics_std.tolist(),
                "statistics_schema_version": statistics_schema_version,
                "statistics_names": list(statistics_names),
                "statistics_names_sha256": statistics_names_sha256(statistics_names),
                "quantile": args.quantile,
                "lambda_component": args.lambda_component,
                "seed": args.seed,
                "best_epoch": epoch,
                "validation_metrics": metrics,
                "episode_contract": episode_contract,
            }
            validate_curve_checkpoint_contract(checkpoint)
            torch.save(checkpoint, output)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    output.with_suffix(output.suffix + ".metrics.json").write_text(
        json.dumps(
            {
                "method_name": "risk_curve",
                "model_class": "RiskCurvePredictor",
                "model_architecture_version": RISK_CURVE_ARCHITECTURE_VERSION,
                "role": "proposed_method",
                "representation": representation,
                "threshold_grid_schema_version": threshold_grid_schema_version,
                "threshold_grid_sha256": semantic_grid_hash,
                "feature_schema_sha256": feature_schema_hash,
                "threshold_grid_manifest_sha256": threshold_grid_manifest_hash,
                "threshold_grid_detector_protocol": (
                    threshold_grid_detector_protocol
                ),
                "threshold_grid_detector_checkpoint_sha256s": list(
                    threshold_grid_detector_checkpoint_hashes
                ),
                "threshold_grid_outer_detector_checkpoint_sha256": (
                    threshold_grid_outer_detector_checkpoint_hash
                ),
                "threshold_grid_episode_detector_checkpoint_sha256s": list(
                    threshold_grid_episode_detector_checkpoint_hashes
                ),
                "best_objective": best_objective,
                "selection_objective": "validation_quantile_pinball",
                "quantile": args.quantile,
                "lambda_component": args.lambda_component,
                "episode_contract": episode_contract,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
