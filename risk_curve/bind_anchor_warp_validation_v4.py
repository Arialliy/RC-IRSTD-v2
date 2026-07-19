"""Bind held-source metadata to an already frozen AnchorWarp policy.

The binder never loads future-E risk curves, detection counts, masks, or Pd
targets.  It only validates the label-free A inputs, episode identities, and
the shared representation/provenance contract.  The frozen policy is nested
unchanged inside the resulting package.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from .count_all_anchor import (
    derive_anchor_log_curves,
    validate_count_all_anchor_archive,
)
from .curve_dataset import LOGIT_EPISODE_SCHEMA_VERSION
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
from .train_anchor_warp_predictor_v4 import (
    validate_anchor_warp_train_only_checkpoint,
)


ANCHOR_WARP_VALIDATION_BINDING_SCHEMA_VERSION = (
    "rc-v4-anchor-warp-source-evaluation-binding-v1"
)
ANCHOR_WARP_BOUND_PACKAGE_SCHEMA_VERSION = (
    "rc-v4-anchor-warp-source-evaluation-package-v1"
)

LABEL_FREE_BINDING_FIELDS = (
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
    "adaptation_sizes",
    "evaluation_sizes",
    "adaptation_ids",
    "evaluation_ids",
    "pseudo_targets",
    "provenance_json",
    "adaptation_predicted_pixel_counts",
    "adaptation_predicted_component_counts_raw",
    "adaptation_predicted_component_counts_upper",
    "adaptation_total_pixels",
    "count_all_adaptation_schema_version",
)

FORBIDDEN_BINDER_FIELDS = (
    "pixel_log_risk",
    "component_log_risk",
    "component_log_risk_raw",
    "component_log_risk_upper",
    "pd_curve",
    "pixel_fp_counts",
    "component_fp_counts",
    "component_fp_counts_raw",
    "component_fp_counts_upper",
    "tp_object_counts",
    "gt_object_counts",
    "mask",
    "masks",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )


def _load_label_free_fields(raw: bytes, *, role: str) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        missing = sorted(set(LABEL_FREE_BINDING_FIELDS).difference(archive.files))
        if missing:
            raise ValueError(
                f"{role} archive lacks label-free binding fields: " + ", ".join(missing)
            )
        # Deliberately index only the allowlist.  NpzFile is lazy, so future-E
        # supervision arrays are not decompressed or interpreted by this binder.
        return {field: np.asarray(archive[field]) for field in LABEL_FREE_BINDING_FIELDS}


def _scalar(archive: Mapping[str, np.ndarray], field: str) -> str:
    value = np.asarray(archive[field])
    if value.ndim != 0:
        raise ValueError(f"{field} must be scalar")
    return str(value.item())


def _parse_id_rows(
    archive: Mapping[str, np.ndarray], *, role: str
) -> tuple[list[list[str]], list[list[str]], list[str]]:
    rows = int(np.asarray(archive["statistics"]).shape[0])
    adaptation: list[list[str]] = []
    evaluation: list[list[str]] = []
    seen: set[str] = set()
    for field, destination in (
        ("adaptation_ids", adaptation),
        ("evaluation_ids", evaluation),
    ):
        values = np.asarray(archive[field])
        if values.ndim != 1 or values.shape[0] != rows:
            raise ValueError(f"{role}.{field} must align with statistics rows")
        for row_index, raw_value in enumerate(values.tolist()):
            try:
                decoded = json.loads(str(raw_value))
            except json.JSONDecodeError as error:
                raise ValueError(f"{role}.{field}[{row_index}] is invalid JSON") from error
            if (
                not isinstance(decoded, list)
                or not decoded
                or any(not isinstance(item, str) or not item for item in decoded)
                or len(set(decoded)) != len(decoded)
            ):
                raise ValueError(f"{role}.{field}[{row_index}] is not a distinct ID list")
            overlap = sorted(seen.intersection(decoded))
            if overlap:
                raise ValueError(f"{role} archive reuses A/E image IDs: {overlap[:3]}")
            seen.update(decoded)
            destination.append(decoded)
    return adaptation, evaluation, sorted(seen)


def _domain_key(value: str) -> str:
    key = "".join(character for character in str(value).casefold() if character.isalnum())
    if key.endswith("sirst"):
        key = key[: -len("sirst")]
    return key


def _validate_label_free_archive(
    archive: Mapping[str, np.ndarray],
    checkpoint: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    if _scalar(archive, "representation") != LOGIT_REPRESENTATION:
        raise ValueError(f"{role} archive is not raw-logit float32")
    if _scalar(archive, "threshold_grid_schema_version") != LOGIT_GRID_SCHEMA_VERSION:
        raise ValueError(f"{role} archive grid schema mismatch")
    if _scalar(archive, "statistics_schema_version") != LOGIT_STATISTICS_SCHEMA_VERSION:
        raise ValueError(f"{role} archive statistics schema mismatch")
    if _scalar(archive, "episode_schema_version") != LOGIT_EPISODE_SCHEMA_VERSION:
        raise ValueError(f"{role} archive episode schema mismatch")
    if _scalar(archive, "threshold_grid_detector_protocol") != GRID_DETECTOR_PROTOCOL:
        raise ValueError(f"{role} archive detector-grid protocol mismatch")
    thresholds = validate_logit_threshold_grid(
        np.asarray(archive["thresholds"], dtype=np.float32)
    )
    grid_hash = logit_threshold_grid_sha256(thresholds)
    if _scalar(archive, "threshold_grid_sha256") != grid_hash:
        raise ValueError(f"{role} archive threshold-grid hash mismatch")
    if grid_hash != checkpoint["threshold_grid_sha256"] or not np.array_equal(
        thresholds,
        validate_logit_threshold_grid(
            np.asarray(checkpoint["thresholds"], dtype=np.float32)
        ),
    ):
        raise ValueError(f"{role} archive and frozen policy use different grids")
    names = validate_statistics_names(archive["statistics_names"], expected_dim=119)
    if _scalar(archive, "statistics_names_sha256") != statistics_names_sha256(names):
        raise ValueError(f"{role} statistics-name digest mismatch")
    expected_feature_hash = feature_schema_sha256(
        LOGIT_STATISTICS_SCHEMA_VERSION, statistics_names=names
    )
    if (
        _scalar(archive, "feature_schema_sha256") != expected_feature_hash
        or checkpoint["feature_schema_sha256"] != expected_feature_hash
        or tuple(checkpoint["statistics_names"]) != tuple(names)
    ):
        raise ValueError(f"{role} archive feature contract differs from policy")
    scalar_fields = (
        "threshold_grid_manifest_sha256",
        "threshold_grid_outer_detector_checkpoint_sha256",
    )
    for field in scalar_fields:
        if _scalar(archive, field) != checkpoint[field]:
            raise ValueError(f"{role} archive/checkpoint {field} mismatch")
    sequence_fields = (
        "threshold_grid_detector_checkpoint_sha256s",
        "threshold_grid_episode_detector_checkpoint_sha256s",
    )
    for field in sequence_fields:
        if tuple(str(value) for value in np.asarray(archive[field]).tolist()) != tuple(
            checkpoint[field]
        ):
            raise ValueError(f"{role} archive/checkpoint {field} mismatch")
    anchors = validate_count_all_anchor_archive(
        archive, expected_grid_sha256=grid_hash
    )
    pixel_anchor, component_anchor = derive_anchor_log_curves(anchors)
    statistics = np.asarray(archive["statistics"], dtype=np.float32)
    if statistics.shape != (anchors.num_episodes, 119) or not np.isfinite(statistics).all():
        raise ValueError(f"{role} statistics shape/finiteness is invalid")
    adaptation, evaluation, all_ids = _parse_id_rows(archive, role=role)
    adaptation_sizes = np.asarray(archive["adaptation_sizes"])
    evaluation_sizes = np.asarray(archive["evaluation_sizes"])
    if (
        adaptation_sizes.shape != (anchors.num_episodes,)
        or evaluation_sizes.shape != adaptation_sizes.shape
        or any(len(row) != int(size) for row, size in zip(adaptation, adaptation_sizes))
        or any(len(row) != int(size) for row, size in zip(evaluation, evaluation_sizes))
    ):
        raise ValueError(f"{role} A/E sizes do not match their identity lists")
    pseudo_targets = sorted(
        {str(value) for value in np.asarray(archive["pseudo_targets"]).tolist()}
    )
    if len(pseudo_targets) != 1 or pseudo_targets[0] not in {
        "IRSTD-1K",
        "NUDT-SIRST",
    }:
        raise ValueError(f"{role} archive must contain one canonical source domain")
    provenance = json.loads(_scalar(archive, "provenance_json"))
    expected_provenance = {
        "formal_causal_contract_verified": True,
        "allow_cross_episode_role_reuse": False,
        "cross_episode_role_reuse_detected": False,
        "threshold_grid_outer_target_excluded": True,
        "threshold_grid_outer_target_key": "nuaa",
        "count_all_adaptation_masks_read": False,
    }
    for field, expected in expected_provenance.items():
        if provenance.get(field) != expected:
            raise ValueError(f"{role} provenance {field} mismatch")
    provenance_validation_domain = str(provenance.get("validation_domain", ""))
    if _domain_key(provenance_validation_domain) not in {"irstd1k", "nudt"}:
        raise ValueError(f"{role} validation-domain provenance is not canonical")
    if role == "validation" and _domain_key(provenance_validation_domain) != _domain_key(
        pseudo_targets[0]
    ):
        raise ValueError(f"{role} validation-domain provenance mismatch")
    if role == "train" and _domain_key(provenance_validation_domain) == _domain_key(
        pseudo_targets[0]
    ):
        raise ValueError("train archive pseudo-target equals its held validation domain")
    semantic = hashlib.sha256(b"rc-v4-anchor-warp-label-free-input-v1\0")
    for name, values in (
        ("statistics", statistics),
        ("pixel_anchor", pixel_anchor),
        ("component_anchor", component_anchor),
    ):
        array = np.ascontiguousarray(values, dtype="<f4")
        semantic.update(name.encode("ascii"))
        semantic.update(json.dumps(list(array.shape)).encode("ascii"))
        semantic.update(array.tobytes(order="C"))
    semantic.update(_json_sha256(adaptation).encode("ascii"))
    semantic.update(_json_sha256(evaluation).encode("ascii"))
    return {
        "domain": pseudo_targets[0],
        "provenance_validation_domain": provenance_validation_domain,
        "num_episodes": anchors.num_episodes,
        "adaptation_ids_sha256": _json_sha256(adaptation),
        "evaluation_ids_sha256": _json_sha256(evaluation),
        "all_image_ids": all_ids,
        "all_image_ids_sha256": _json_sha256(all_ids),
        "anchor_semantic_sha256": anchors.semantic_sha256,
        "label_free_input_semantic_sha256": semantic.hexdigest(),
        "threshold_grid_sha256": grid_hash,
        "feature_schema_sha256": expected_feature_hash,
    }


def validate_anchor_warp_bound_package(package: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(package, Mapping):
        raise TypeError("anchor-warp bound package must be a mapping")
    if package.get("package_schema_version") != ANCHOR_WARP_BOUND_PACKAGE_SCHEMA_VERSION:
        raise ValueError("anchor-warp bound package schema mismatch")
    if package.get("artifact_stage") != "source_evaluation_bound":
        raise ValueError("anchor-warp bound package stage mismatch")
    frozen = package.get("frozen_checkpoint")
    frozen_contract = validate_anchor_warp_train_only_checkpoint(frozen)
    binding = package.get("validation_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("anchor-warp package lacks validation_binding")
    expected = {
        "schema_version": ANCHOR_WARP_VALIDATION_BINDING_SCHEMA_VERSION,
        "validation_labels_read_by_binder": False,
        "state_frozen_before_validation_binding": True,
        "post_binding_state_dict_semantic_sha256": frozen_contract[
            "state_dict_semantic_sha256"
        ],
        "post_binding_policy_semantic_sha256": frozen_contract[
            "policy_semantic_sha256"
        ],
        "accessed_npz_fields": list(LABEL_FREE_BINDING_FIELDS),
        "forbidden_npz_fields_accessed": [],
    }
    for field, value in expected.items():
        if binding.get(field) != value:
            raise ValueError(f"anchor-warp validation binding {field} mismatch")
    parent = str(package.get("parent_frozen_checkpoint_sha256", ""))
    validation_sha = str(binding.get("validation_archive_sha256", ""))
    for field, value in (
        ("parent_frozen_checkpoint_sha256", parent),
        ("validation_archive_sha256", validation_sha),
    ):
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"anchor-warp package {field} is invalid")
    held_domain = binding.get("validation_domain")
    if held_domain not in {"IRSTD-1K", "NUDT-SIRST"} or held_domain in frozen[
        "train_pseudo_targets"
    ]:
        raise ValueError("anchor-warp binding held domain is not complementary")
    if set(frozen["train_pseudo_targets"] + [held_domain]) != {
        "IRSTD-1K",
        "NUDT-SIRST",
    }:
        raise ValueError("anchor-warp package does not cover both source domains")
    return {
        "formal_source_evaluation_bound": True,
        "validation_domain": held_domain,
        "validation_archive_sha256": validation_sha,
        **frozen_contract,
    }


def bind_anchor_warp_validation(
    *,
    frozen_checkpoint: str | Path,
    validation_file: str | Path,
    output: str | Path,
) -> Path:
    frozen_path = Path(frozen_checkpoint).expanduser().resolve()
    validation_path = Path(validation_file).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not frozen_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("frozen checkpoint and validation archive must exist")
    frozen_bytes = frozen_path.read_bytes()
    frozen_sha = _sha256_bytes(frozen_bytes)
    frozen = torch.load(io.BytesIO(frozen_bytes), map_location="cpu", weights_only=True)
    frozen_contract = validate_anchor_warp_train_only_checkpoint(frozen)
    # The first held-source read occurs only after the immutable train-only
    # checkpoint above has passed its state/policy hash validation.
    validation_bytes = validation_path.read_bytes()
    validation_sha = _sha256_bytes(validation_bytes)
    validation_archive = _load_label_free_fields(
        validation_bytes, role="validation"
    )
    validation_contract = _validate_label_free_archive(
        validation_archive, frozen, role="validation"
    )
    train_path = Path(frozen["train_archive"]).expanduser().resolve()
    train_bytes = train_path.read_bytes()
    if _sha256_bytes(train_bytes) != frozen["train_archive_sha256"]:
        raise RuntimeError("frozen checkpoint training archive has drifted")
    train_archive = _load_label_free_fields(train_bytes, role="train")
    train_contract = _validate_label_free_archive(train_archive, frozen, role="train")
    if train_contract["domain"] != frozen["train_pseudo_targets"][0]:
        raise ValueError("frozen checkpoint train-domain metadata mismatch")
    if _domain_key(train_contract["provenance_validation_domain"]) != _domain_key(
        validation_contract["domain"]
    ):
        raise ValueError("train/validation archives bind different held source domains")
    overlap = sorted(
        set(train_contract["all_image_ids"]).intersection(
            validation_contract["all_image_ids"]
        )
    )
    if overlap:
        raise ValueError(f"train/held source episodes reuse image IDs: {overlap[:3]}")
    package: dict[str, Any] = {
        "package_schema_version": ANCHOR_WARP_BOUND_PACKAGE_SCHEMA_VERSION,
        "artifact_stage": "source_evaluation_bound",
        "parent_frozen_checkpoint_path": str(frozen_path),
        "parent_frozen_checkpoint_sha256": frozen_sha,
        "frozen_checkpoint": frozen,
        "validation_binding": {
            "schema_version": ANCHOR_WARP_VALIDATION_BINDING_SCHEMA_VERSION,
            "validation_archive": str(validation_path),
            "validation_archive_sha256": validation_sha,
            "validation_domain": validation_contract["domain"],
            "validation_num_episodes": validation_contract["num_episodes"],
            "validation_adaptation_ids_sha256": validation_contract[
                "adaptation_ids_sha256"
            ],
            "validation_evaluation_ids_sha256": validation_contract[
                "evaluation_ids_sha256"
            ],
            "validation_anchor_semantic_sha256": validation_contract[
                "anchor_semantic_sha256"
            ],
            "validation_label_free_input_semantic_sha256": validation_contract[
                "label_free_input_semantic_sha256"
            ],
            "train_all_image_ids_sha256": train_contract["all_image_ids_sha256"],
            "train_validation_image_id_overlap": [],
            "validation_labels_read_by_binder": False,
            "accessed_npz_fields": list(LABEL_FREE_BINDING_FIELDS),
            "forbidden_npz_fields_accessed": [],
            "state_frozen_before_validation_binding": True,
            "post_binding_state_dict_semantic_sha256": frozen_contract[
                "state_dict_semantic_sha256"
            ],
            "post_binding_policy_semantic_sha256": frozen_contract[
                "policy_semantic_sha256"
            ],
        },
    }
    validate_anchor_warp_bound_package(package)
    if _sha256_bytes(frozen_path.read_bytes()) != frozen_sha:
        raise RuntimeError("frozen checkpoint changed during validation binding")
    if _sha256_bytes(validation_path.read_bytes()) != validation_sha:
        raise RuntimeError("validation archive changed during metadata binding")
    if _sha256_bytes(train_path.read_bytes()) != frozen["train_archive_sha256"]:
        raise RuntimeError("training archive changed during validation binding")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(package, temporary)
        replay = torch.load(temporary, map_location="cpu", weights_only=True)
        validate_anchor_warp_bound_package(replay)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-checkpoint", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        bind_anchor_warp_validation(
            frozen_checkpoint=args.frozen_checkpoint,
            validation_file=args.validation_file,
            output=args.output,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ANCHOR_WARP_BOUND_PACKAGE_SCHEMA_VERSION",
    "ANCHOR_WARP_VALIDATION_BINDING_SCHEMA_VERSION",
    "FORBIDDEN_BINDER_FIELDS",
    "LABEL_FREE_BINDING_FIELDS",
    "bind_anchor_warp_validation",
    "validate_anchor_warp_bound_package",
]
