"""Formal representation contract for source-derived raw-logit grids.

The learned risk-curve model consumes only the finite thresholds stored in the
grid array.  Rejecting every pixel is an external deployment action represented
by ``+inf`` and ``threshold_index=None``; it is deliberately not appended to the
model output grid.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROBABILITY_REPRESENTATION = "sigmoid_probability_float32"
LOGIT_REPRESENTATION = "raw_logit_float32"
SUPPORTED_REPRESENTATIONS = (
    PROBABILITY_REPRESENTATION,
    LOGIT_REPRESENTATION,
)
LOGIT_GRID_SCHEMA_VERSION = "rc-v4-logit-dense-tail-grid-v1"
LOGIT_GRID_ARTIFACT_TYPE = "rc-irstd-logit-threshold-grid"
LOGIT_PREDICTION_RULE = "prediction = (raw_logits >= threshold)"
GRID_DETECTOR_PROTOCOL = "all_source_only_detector_folds"
DEFAULT_LOGIT_GRID_POINTS = 1024
MAX_MODEL_GRID_POINTS = 2048
EMPTY_ACTION_THRESHOLD = float("inf")

_GRID_FILENAME = "threshold_grid.npy"
_MANIFEST_FILENAME = "threshold_grid.json"
_DIGEST_FILENAME = "threshold_grid.sha256"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_sha256(payload: Any) -> str:
    """Hash a JSON-safe value using one canonical UTF-8 encoding."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def empty_action_contract() -> dict[str, Any]:
    """Return the external, model-independent reject-all action contract."""

    return {
        "external": True,
        "threshold": "+inf",
        "threshold_index": None,
        "prediction": "empty_mask",
        "included_in_model_grid": False,
    }


def validate_logit_threshold_grid(
    thresholds: np.ndarray,
    *,
    max_points: int = MAX_MODEL_GRID_POINTS,
) -> np.ndarray:
    """Validate a finite FP32 raw-logit model grid without coercion.

    Refusing implicit float64-to-float32 conversion is intentional: otherwise
    the semantic digest could describe different threshold states before and
    after persistence.
    """

    if isinstance(max_points, bool) or not isinstance(max_points, int):
        raise TypeError("max_points must be an integer")
    if max_points < 2 or max_points > MAX_MODEL_GRID_POINTS:
        raise ValueError(
            f"max_points must lie in [2, {MAX_MODEL_GRID_POINTS}]"
        )
    values = np.asarray(thresholds)
    if values.dtype != np.float32:
        raise ValueError("Raw-logit threshold grid must use float32 dtype")
    if values.ndim != 1:
        raise ValueError("Raw-logit threshold grid must be one-dimensional")
    if values.size < 2:
        raise ValueError("Raw-logit threshold grid must contain at least two points")
    if values.size > max_points:
        raise ValueError(
            f"Raw-logit threshold grid has {values.size} points; maximum is {max_points}"
        )
    if not np.isfinite(values).all():
        raise ValueError(
            "Raw-logit model thresholds must all be finite; +inf is an external action"
        )
    if np.any(np.diff(values.astype(np.float64)) <= 0.0):
        raise ValueError("Raw-logit threshold grid must be strictly increasing")
    return np.ascontiguousarray(values, dtype=np.float32)


def logit_threshold_grid_sha256(
    thresholds: np.ndarray,
    *,
    schema_version: str = LOGIT_GRID_SCHEMA_VERSION,
    representation: str = LOGIT_REPRESENTATION,
) -> str:
    """Return a semantic digest bound to schema, representation, and FP32 values."""

    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError("schema_version must be a non-empty string")
    if not isinstance(representation, str) or not representation:
        raise ValueError("representation must be a non-empty string")
    values = validate_logit_threshold_grid(thresholds)
    header = json.dumps(
        {
            "representation": representation,
            "schema_version": schema_version,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(values, dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _domain_key(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("domain name must be a non-empty string")
    key = "".join(
        character for character in value.casefold() if character.isalnum()
    )
    if key.endswith("sirst") and len(key) > len("sirst"):
        key = key[: -len("sirst")]
    if not key:
        raise ValueError("domain name contains no alphanumeric characters")
    return key


@dataclass(frozen=True)
class LogitGridArtifact:
    thresholds: np.ndarray
    manifest: dict[str, Any]
    manifest_path: Path
    semantic_sha256: str


def load_logit_grid_artifact(path: str | Path) -> LogitGridArtifact:
    """Load and fail-closed verify the three-file v4 grid artifact."""

    requested = Path(path).expanduser().resolve()
    manifest_path = (
        requested / _MANIFEST_FILENAME if requested.is_dir() else requested
    )
    if manifest_path.name != _MANIFEST_FILENAME or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Expected {_MANIFEST_FILENAME} or its containing directory: {requested}"
        )
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, dict):
        raise ValueError("Logit-grid manifest must decode to a JSON object")
    manifest: dict[str, Any] = dict(raw_manifest)
    expected_scalars = {
        "schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "artifact_type": LOGIT_GRID_ARTIFACT_TYPE,
        "representation": LOGIT_REPRESENTATION,
        "dtype": "float32",
        "prediction_rule": LOGIT_PREDICTION_RULE,
        "grid_source": "source_official_train_only",
        "grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "grid_file": _GRID_FILENAME,
        "digest_file": _DIGEST_FILENAME,
    }
    for field, expected in expected_scalars.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"Logit-grid manifest {field} must equal {expected!r}"
            )
    if manifest.get("formal_protocol_eligible") is not True:
        raise ValueError("Logit-grid manifest is not formal-protocol eligible")

    grid_path = manifest_path.parent / _GRID_FILENAME
    digest_path = manifest_path.parent / _DIGEST_FILENAME
    if not grid_path.is_file() or not digest_path.is_file():
        raise FileNotFoundError("Logit-grid artifact is missing its NPY or digest file")
    thresholds = validate_logit_threshold_grid(
        np.load(grid_path, allow_pickle=False)
    )
    if manifest.get("grid_points") != int(thresholds.size):
        raise ValueError("Logit-grid manifest grid_points mismatch")
    if manifest.get("finite_grid_points") != int(thresholds.size):
        raise ValueError("Logit-grid manifest finite_grid_points mismatch")
    if manifest.get("max_model_grid_points") != MAX_MODEL_GRID_POINTS:
        raise ValueError("Logit-grid manifest maximum-size contract mismatch")

    semantic = logit_threshold_grid_sha256(thresholds)
    recorded_semantic = _require_sha256(
        manifest.get("grid_sha256"), field="grid_sha256"
    )
    if recorded_semantic != semantic:
        raise ValueError("Logit-grid semantic SHA-256 mismatch")
    recorded_file_sha = _require_sha256(
        manifest.get("grid_file_sha256"), field="grid_file_sha256"
    )
    if recorded_file_sha != _file_sha256(grid_path):
        raise ValueError("Logit-grid NPY file SHA-256 mismatch")
    digest_text = digest_path.read_text(encoding="ascii").strip()
    if digest_text != semantic:
        raise ValueError("Logit-grid digest sidecar mismatch")

    if manifest.get("empty_action") != empty_action_contract():
        raise ValueError("Logit-grid external empty-action contract mismatch")
    source_keys = manifest.get("source_domain_keys")
    outer_key = manifest.get("outer_target_key")
    if (
        not isinstance(source_keys, list)
        or not source_keys
        or any(not isinstance(item, str) or not item for item in source_keys)
        or len(set(source_keys)) != len(source_keys)
    ):
        raise ValueError("Logit-grid manifest has invalid source_domain_keys")
    if not isinstance(outer_key, str) or not outer_key:
        raise ValueError("Logit-grid manifest has no valid outer_target_key")
    if outer_key in set(source_keys) or manifest.get("outer_target_excluded") is not True:
        raise ValueError("Logit-grid manifest does not prove outer-target exclusion")
    if manifest.get("outer_target_labels_used") is not False:
        raise ValueError("Logit-grid manifest permits outer-target labels")
    checkpoint_hashes = manifest.get("detector_checkpoint_sha256s")
    checkpoint_count = manifest.get("detector_checkpoint_count")
    if (
        not isinstance(checkpoint_hashes, list)
        or len(checkpoint_hashes) != len(source_keys) + 1
        or len(set(checkpoint_hashes)) != len(checkpoint_hashes)
        or any(
            not isinstance(value, str) or not _SHA256_RE.fullmatch(value)
            for value in checkpoint_hashes
        )
        or checkpoint_count != len(checkpoint_hashes)
    ):
        raise ValueError(
            "Logit-grid manifest must bind one distinct detector checkpoint "
            "for the outer-final and every inner source-only fold"
        )
    outer_detector_hash = manifest.get("outer_detector_checkpoint_sha256")
    episode_detector_hashes = manifest.get(
        "episode_detector_checkpoint_sha256s"
    )
    if (
        not isinstance(outer_detector_hash, str)
        or outer_detector_hash not in checkpoint_hashes
        or not isinstance(episode_detector_hashes, list)
        or len(episode_detector_hashes) != len(source_keys)
        or len(set(episode_detector_hashes)) != len(episode_detector_hashes)
        or set(episode_detector_hashes)
        != set(checkpoint_hashes).difference({outer_detector_hash})
    ):
        raise ValueError(
            "Logit-grid outer/episode detector checkpoint roles are invalid"
        )
    inputs = manifest.get("input_score_artifacts")
    if not isinstance(inputs, list) or len(inputs) != len(source_keys) ** 2:
        raise ValueError("Logit-grid manifest has incomplete input provenance")
    observed_input_hashes = {
        str(item.get("detector_weight_sha256"))
        for item in inputs
        if isinstance(item, dict)
    }
    if observed_input_hashes != set(checkpoint_hashes):
        raise ValueError("Logit-grid detector checkpoint provenance mismatch")
    if manifest.get("source_provenance_sha256") != canonical_json_sha256(inputs):
        raise ValueError("Logit-grid source provenance SHA-256 mismatch")
    observed_input_targets: set[str] = set()
    inputs_by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    for item in inputs:
        assert isinstance(item, dict)
        target_key = item.get("target_domain_key")
        detector_sources = item.get("detector_source_datasets")
        detector_source_keys = item.get("detector_source_domain_keys")
        checkpoint_hash = item.get("detector_weight_sha256")
        if (
            not isinstance(target_key, str)
            or target_key not in set(source_keys)
            or not isinstance(detector_sources, list)
            or not detector_sources
            or not isinstance(detector_source_keys, list)
            or sorted(_domain_key(value) for value in detector_sources)
            != sorted(map(str, detector_source_keys))
            or target_key not in set(map(str, detector_source_keys))
            or checkpoint_hash not in set(checkpoint_hashes)
        ):
            raise ValueError(
                "Logit-grid input is not a source-only detector self-score fold"
            )
        observed_input_targets.add(target_key)
        inputs_by_checkpoint.setdefault(str(checkpoint_hash), []).append(item)
    if observed_input_targets != set(source_keys):
        raise ValueError("Logit-grid input source-domain provenance mismatch")
    folds = manifest.get("detector_folds")
    if not isinstance(folds, list) or len(folds) != len(checkpoint_hashes):
        raise ValueError("Logit-grid detector-fold provenance is incomplete")
    expected_fold_sets = {frozenset(source_keys)}
    expected_fold_sets.update(
        frozenset(set(source_keys).difference({held_out}))
        for held_out in source_keys
    )
    observed_fold_sets: set[frozenset[str]] = set()
    fold_by_checkpoint: dict[str, frozenset[str]] = {}
    for fold in folds:
        if not isinstance(fold, dict):
            raise ValueError("Logit-grid detector fold must be an object")
        checkpoint_hash = fold.get("detector_checkpoint_sha256")
        source_set = frozenset(map(str, fold.get("source_domain_keys", [])))
        if (
            checkpoint_hash not in set(checkpoint_hashes)
            or source_set not in expected_fold_sets
            or checkpoint_hash in fold_by_checkpoint
        ):
            raise ValueError("Logit-grid detector-fold roles are inconsistent")
        fold_by_checkpoint[str(checkpoint_hash)] = source_set
        observed_fold_sets.add(source_set)
        scored = set(
            map(str, fold.get("scored_official_train_domain_keys", []))
        )
        if scored != set(source_set):
            raise ValueError("Logit-grid detector fold did not self-score all sources")
    if observed_fold_sets != expected_fold_sets:
        raise ValueError("Logit-grid detector-fold source sets are incomplete")
    if fold_by_checkpoint.get(outer_detector_hash) != frozenset(source_keys):
        raise ValueError("Logit-grid outer detector is not the full-source fold")
    if set(episode_detector_hashes) != {
        checkpoint_hash
        for checkpoint_hash, source_set in fold_by_checkpoint.items()
        if source_set != frozenset(source_keys)
    }:
        raise ValueError("Logit-grid episode detector roles are inconsistent")
    for checkpoint_hash, fold_inputs in inputs_by_checkpoint.items():
        scored_targets = {str(item["target_domain_key"]) for item in fold_inputs}
        if scored_targets != set(fold_by_checkpoint[checkpoint_hash]):
            raise ValueError(
                "Logit-grid input artifacts do not cover the detector training fold"
            )

    return LogitGridArtifact(
        thresholds=thresholds,
        manifest=manifest,
        manifest_path=manifest_path,
        semantic_sha256=semantic,
    )


__all__ = [
    "EMPTY_ACTION_THRESHOLD",
    "DEFAULT_LOGIT_GRID_POINTS",
    "LOGIT_GRID_ARTIFACT_TYPE",
    "GRID_DETECTOR_PROTOCOL",
    "LOGIT_GRID_SCHEMA_VERSION",
    "LOGIT_PREDICTION_RULE",
    "LOGIT_REPRESENTATION",
    "MAX_MODEL_GRID_POINTS",
    "PROBABILITY_REPRESENTATION",
    "SUPPORTED_REPRESENTATIONS",
    "LogitGridArtifact",
    "canonical_json_sha256",
    "empty_action_contract",
    "load_logit_grid_artifact",
    "logit_threshold_grid_sha256",
    "validate_logit_threshold_grid",
]
