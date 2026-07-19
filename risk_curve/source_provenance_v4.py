"""Transitive source-only provenance verification for the formal v4 Gate C.

The ordinary episode and threshold-grid loaders validate the semantics of one
artifact at a time.  This module closes the remaining provenance gap: it
walks from the two paired-LODO validation archives to the score exports,
official source-train split files, threshold-grid inputs, and the three real
detector checkpoints.  Every parsed object is derived from an entry-time byte
snapshot, and every logical path is re-hashed before a successful return.

The verifier is deliberately source-only.  It accepts exactly IRSTD-1K and
NUDT-SIRST as sources, records NUAA-SIRST only as the excluded outer target,
and has no argument through which an outer-target split or label path can be
provided.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import resource
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from data_ext.split_utils import ensure_unique_sample_ids
from evaluation.artifact_integrity import (
    SCORE_MANIFEST_SCHEMA_VERSION,
    SCORE_RECORD_INTEGRITY_SCHEMA,
    ordered_ids_sha256,
    score_records_sha256,
)

from .curve_dataset import load_curve_archive
from .representation import (
    GRID_DETECTOR_PROTOCOL,
    LOGIT_REPRESENTATION,
    canonical_json_sha256,
    load_logit_grid_artifact,
)


SOURCE_PROVENANCE_SCHEMA_VERSION = (
    "rc-v4-source-only-transitive-provenance-v2-checkpoint-metadata"
)
SOURCE_PROVENANCE_RUN_SCHEMA_VERSION = (
    "rc-v4-source-only-transitive-provenance-run-v2-checkpoint-metadata"
)
FORMAL_SOURCE_DOMAIN_NAMES = ("IRSTD-1K", "NUDT-SIRST")
FORMAL_SOURCE_DOMAIN_KEYS = frozenset(("irstd1k", "nudt"))
FORMAL_OUTER_DOMAIN_NAME = "NUAA-SIRST"
FORMAL_OUTER_DOMAIN_KEY = "nuaa"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _domain_key(value: Any) -> str:
    text = "".join(
        character for character in str(value).casefold() if character.isalnum()
    )
    if text.endswith("sirst"):
        text = text[: -len("sirst")]
    if not text:
        raise ValueError("Domain name normalises to an empty key")
    return text


def _normalised_text(value: Any) -> str:
    return "".join(
        character for character in str(value).casefold() if character.isalnum()
    )


def _reject_outer_reference(value: Any, *, field: str) -> None:
    if FORMAL_OUTER_DOMAIN_KEY in _normalised_text(value):
        raise ValueError(f"Outer target appears in source-only {field}: {value!r}")


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _formal_checkpoint_exact_metadata() -> dict[str, Any]:
    return {
        "kind": "detector",
        "checkpoint_selection": "fixed_last",
        "selection_rule": "fixed_last",
        "test_labels_used_for_selection": False,
        "diagnostic_test_eval": False,
        "diagnostic_only": False,
        "formal_paper_checkpoint": True,
        "format_version": 2,
        "epoch": 19,
        "warm_flag": True,
        "inference_head": "multi_scale_fused",
    }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_bytes(payload: bytes, *, field: str) -> dict[str, Any]:
    def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{field} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is not valid UTF-8 JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"{field} must decode to a JSON object")
    return decoded


@dataclass(frozen=True)
class _FileStamp:
    logical_path: Path
    resolved_path: Path
    sha256: str
    size: int


@dataclass(frozen=True)
class _DirectoryStamp:
    logical_path: Path
    resolved_path: Path
    npz_names: tuple[str, ...]


class _SnapshotRegistry:
    """Capture bytes once for parsing and later detect path/content drift."""

    def __init__(self, project_root: str | Path) -> None:
        root = Path(project_root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(f"Project root is not a directory: {root}")
        self.root = root
        self._files: dict[Path, _FileStamp] = {}
        self._directories: dict[Path, _DirectoryStamp] = {}

    def _logical_path(
        self, value: str | Path, *, base: Path | None = None
    ) -> Path:
        requested = Path(value).expanduser()
        if not requested.is_absolute():
            requested = (base or self.root) / requested
        return Path(os.path.abspath(requested))

    def resolve_file(
        self, value: str | Path, *, base: Path | None = None
    ) -> tuple[Path, Path]:
        logical = self._logical_path(value, base=base)
        try:
            resolved = logical.resolve(strict=True)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Provenance file does not exist: {logical}") from error
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise ValueError(
                f"Provenance path escapes project_root: {logical} -> {resolved}"
            ) from error
        if not resolved.is_file():
            raise FileNotFoundError(f"Provenance path is not a regular file: {resolved}")
        return logical, resolved

    def resolve_directory(
        self, value: str | Path, *, base: Path | None = None
    ) -> tuple[Path, Path]:
        logical = self._logical_path(value, base=base)
        try:
            resolved = logical.resolve(strict=True)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Provenance directory does not exist: {logical}"
            ) from error
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise ValueError(
                f"Provenance directory escapes project_root: {logical} -> {resolved}"
            ) from error
        if not resolved.is_dir():
            raise NotADirectoryError(resolved)
        return logical, resolved

    def capture(
        self, value: str | Path, *, base: Path | None = None
    ) -> tuple[bytes, _FileStamp]:
        logical, resolved = self.resolve_file(value, base=base)
        if logical in self._files:
            raise ValueError(f"A provenance path was captured more than once: {logical}")
        payload = resolved.read_bytes()
        stamp = _FileStamp(
            logical_path=logical,
            resolved_path=resolved,
            sha256=_sha256_bytes(payload),
            size=len(payload),
        )
        self._files[logical] = stamp
        return payload, stamp

    def capture_npz_listing(self, value: str | Path) -> _DirectoryStamp:
        logical, resolved = self.resolve_directory(value)
        if logical in self._directories:
            raise ValueError(f"A provenance directory was captured twice: {logical}")
        names = tuple(sorted(path.name for path in resolved.glob("*.npz")))
        stamp = _DirectoryStamp(logical, resolved, names)
        self._directories[logical] = stamp
        return stamp

    def assert_unchanged(self) -> None:
        for logical, stamp in self._files.items():
            try:
                current_resolved = logical.resolve(strict=True)
            except FileNotFoundError as error:
                raise ValueError(
                    f"Provenance path disappeared after its byte snapshot: {logical}"
                ) from error
            if current_resolved != stamp.resolved_path:
                raise ValueError(
                    f"Provenance path drifted after its byte snapshot: {logical}"
                )
            if _sha256_file(current_resolved) != stamp.sha256:
                raise ValueError(
                    f"Provenance file changed after its byte snapshot: {logical}"
                )
        for logical, stamp in self._directories.items():
            try:
                current_resolved = logical.resolve(strict=True)
            except FileNotFoundError as error:
                raise ValueError(
                    f"Score directory disappeared after snapshot: {logical}"
                ) from error
            current_names = tuple(
                sorted(path.name for path in current_resolved.glob("*.npz"))
            )
            if (
                current_resolved != stamp.resolved_path
                or current_names != stamp.npz_names
            ):
                raise ValueError(
                    f"Score directory contents/path drifted after snapshot: {logical}"
                )

    @property
    def num_files(self) -> int:
        return len(self._files)


@dataclass(frozen=True)
class _SplitEvidence:
    domain: str
    path: Path
    sha256: str
    ordered_ids_sha256: str
    ids: tuple[str, ...]


@dataclass(frozen=True)
class _ScoreEvidence:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    records_sha256: str
    ordered_ids_sha256: str
    image_ids: tuple[str, ...]
    target_domain: str
    source_domains: tuple[str, ...]
    checkpoint_sha256: str


@dataclass(frozen=True)
class _CheckpointEvidence:
    path: Path
    sha256: str
    source_domains: tuple[str, ...]
    metadata_sha256: str
    format_version: int
    epoch: int
    warm_flag: bool
    inference_head: str


def _parse_split_snapshot(payload: bytes, *, path: Path) -> tuple[str, ...]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"Official train split is not UTF-8: {path}") from error
    entries = [line.strip() for line in text.splitlines() if line.strip()]
    if not entries:
        raise ValueError(f"Official train split is empty: {path}")
    if len(set(entries)) != len(entries):
        raise ValueError(f"Official train split contains duplicate entries: {path}")
    ids = tuple(ensure_unique_sample_ids(entries))
    for image_id in ids:
        _reject_outer_reference(image_id, field="official-train image ID")
    return ids


def _verify_detector_checkpoint_snapshot(
    payload: bytes,
    *,
    stamp: _FileStamp,
    registry: _SnapshotRegistry,
    splits: Mapping[str, _SplitEvidence],
) -> tuple[_CheckpointEvidence, dict[str, Any]]:
    """Safely verify formal detector metadata from the captured PT bytes.

    ``weights_only=True`` is mandatory.  There is deliberately no unsafe
    pickle fallback: an unsupported checkpoint is ineligible for this formal
    source chain rather than being loaded with arbitrary-code semantics.
    Test-split paths embedded for historical bookkeeping are never opened.
    """

    try:
        checkpoint = torch.load(
            io.BytesIO(payload), map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise ValueError(
            f"Detector checkpoint cannot be safely loaded with weights_only=True: "
            f"{stamp.resolved_path}"
        ) from error
    if not isinstance(checkpoint, dict):
        raise ValueError("Formal detector checkpoint must decode to a dictionary")
    exact_metadata = _formal_checkpoint_exact_metadata()
    for field, expected in exact_metadata.items():
        observed = checkpoint.get(field)
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(
                f"Detector checkpoint {field} must equal {expected!r}: "
                f"{stamp.resolved_path}"
            )
    source_names = checkpoint.get("source_names")
    if (
        not isinstance(source_names, list)
        or not source_names
        or any(value not in FORMAL_SOURCE_DOMAIN_NAMES for value in source_names)
        or len(set(source_names)) != len(source_names)
        or len(source_names) not in {1, 2}
    ):
        raise ValueError(
            "Detector checkpoint source_names must be one canonical source or "
            "the exact canonical source pair"
        )
    if len(source_names) == 2 and set(source_names) != set(
        FORMAL_SOURCE_DOMAIN_NAMES
    ):
        raise ValueError("Two-source checkpoint must contain exactly IRSTD and NUDT")

    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Detector checkpoint lacks config metadata")
    data_config = config.get("data")
    training_config = config.get("training")
    if not isinstance(data_config, dict) or not isinstance(training_config, dict):
        raise ValueError("Detector checkpoint config data/training metadata is invalid")
    config_sources = data_config.get("sources")
    if (
        not isinstance(config_sources, list)
        or any(not isinstance(item, dict) for item in config_sources)
        or [item.get("name") for item in config_sources] != source_names
        or data_config.get("train_split") != "train"
        or data_config.get("val_split") is not None
        or data_config.get("diagnostic_test_eval") is not False
        or training_config.get("checkpoint_selection") != "fixed_last"
    ):
        raise ValueError(
            "Detector checkpoint config does not corroborate fixed-last, "
            "train-only canonical source metadata"
        )

    records = checkpoint.get("source_split_records")
    if not isinstance(records, list) or len(records) != len(source_names):
        raise ValueError("Detector checkpoint source_split_records cardinality is invalid")
    records_by_name: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Detector checkpoint source_split_records must be objects")
        name = record.get("name")
        if name not in source_names or name in records_by_name:
            raise ValueError("Detector checkpoint source_split_records names are invalid")
        records_by_name[str(name)] = record
    if set(records_by_name) != set(source_names):
        raise ValueError("Detector checkpoint split records do not cover source_names")

    split_audit: list[dict[str, Any]] = []
    for name in source_names:
        record = records_by_name[name]
        _reject_outer_reference(name, field="checkpoint source name")
        for field in ("path", "train_split_file"):
            value = record.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Checkpoint split record lacks {field}")
            _reject_outer_reference(
                value, field=f"checkpoint {name} source split {field}"
            )
        split = splits[name]
        _logical_dataset, resolved_dataset = registry.resolve_directory(
            str(record["path"])
        )
        expected_dataset = split.path.parent.parent
        if resolved_dataset != expected_dataset:
            raise ValueError(
                f"Checkpoint {name} dataset path is not the canonical split root"
            )
        _logical_split, resolved_split = registry.resolve_file(
            str(record["train_split_file"])
        )
        if resolved_split != split.path:
            raise ValueError(
                f"Checkpoint {name} train split path is not the supplied official split"
            )
        if record.get("train_split_file_sha256") != split.sha256:
            raise ValueError(f"Checkpoint {name} train split byte SHA mismatch")
        if record.get("train_ordered_ids_sha256") != split.ordered_ids_sha256:
            raise ValueError(f"Checkpoint {name} train ordered-ID SHA mismatch")
        if record.get("num_train_samples") != len(split.ids):
            raise ValueError(f"Checkpoint {name} train sample count mismatch")
        if record.get("train_test_id_overlap") is not False:
            raise ValueError(f"Checkpoint {name} does not prove train/test disjointness")
        # Do not resolve or read test_split_file.  Its presence in old training
        # bookkeeping does not make it an input to this source-only verifier.
        for field in ("test_split_file",):
            if record.get(field) is not None:
                _reject_outer_reference(
                    record[field], field=f"checkpoint {name} bookkeeping {field}"
                )
        split_audit.append(
            {
                "name": name,
                "dataset_root": str(resolved_dataset),
                "train_split_file": str(split.path),
                "train_split_file_sha256": split.sha256,
                "train_ordered_ids_sha256": split.ordered_ids_sha256,
                "num_train_samples": len(split.ids),
                "train_test_id_overlap": False,
            }
        )

    selected_metadata = {
        **exact_metadata,
        "source_names": list(source_names),
        "source_split_records": split_audit,
        "source_split_records_only_canonical_official_train": True,
        "test_split_artifacts_read": False,
        "safe_load": {
            "weights_only": True,
            "map_location": "cpu",
            "unsafe_pickle_fallback_used": False,
        },
    }
    metadata_sha = canonical_json_sha256(selected_metadata)
    evidence = _CheckpointEvidence(
        path=stamp.resolved_path,
        sha256=stamp.sha256,
        source_domains=tuple(map(str, source_names)),
        metadata_sha256=metadata_sha,
        format_version=2,
        epoch=19,
        warm_flag=True,
        inference_head="multi_scale_fused",
    )
    audit = {
        "path": str(stamp.resolved_path),
        "sha256": stamp.sha256,
        "size_bytes": stamp.size,
        "metadata_sha256": metadata_sha,
        **selected_metadata,
    }
    del checkpoint
    return evidence, audit


def _embedded_score_identity(payload: bytes, *, path: Path) -> str:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if "image_id" not in archive:
                raise ValueError(f"Score record lacks image_id: {path}")
            image_array = np.asarray(archive["image_id"])
            if image_array.ndim != 0 or image_array.dtype.kind not in {"U", "S"}:
                raise ValueError(f"Score record image_id must be a string scalar: {path}")
            image_id = str(image_array.item())
            if "labels_loaded" not in archive:
                raise ValueError(f"Formal source score lacks labels_loaded: {path}")
            labels_loaded = np.asarray(archive["labels_loaded"])
            if (
                labels_loaded.ndim != 0
                or labels_loaded.dtype.kind != "b"
                or not bool(labels_loaded.item())
            ):
                raise ValueError(
                    f"Formal source score must declare labels_loaded=true: {path}"
                )
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError(f"Unreadable score record snapshot: {path}") from error
    if not image_id:
        raise ValueError(f"Score record has an empty image_id: {path}")
    return image_id


def _safe_record_path(root: Path, filename: Any) -> Path:
    if not isinstance(filename, str) or not filename:
        raise ValueError("Score record file must be a non-empty string")
    relative = Path(filename)
    if (
        relative.is_absolute()
        or len(relative.parts) != 1
        or relative.name != filename
        or relative.suffix.lower() != ".npz"
    ):
        raise ValueError(f"Unsafe score record path: {filename!r}")
    return root / relative


def _score_reference_matches(
    evidence: _ScoreEvidence, reference: Mapping[str, Any], *, field: str
) -> None:
    expected = {
        "score_manifest_sha256": evidence.manifest_sha256,
        "score_records_sha256": evidence.records_sha256,
        "score_ordered_image_ids_sha256": evidence.ordered_ids_sha256,
    }
    aliases = {
        "score_manifest_sha256": ("score_manifest_sha256", "manifest_sha256"),
        "score_records_sha256": ("score_records_sha256", "records_sha256"),
        "score_ordered_image_ids_sha256": (
            "score_ordered_image_ids_sha256",
            "ordered_image_ids_sha256",
        ),
    }
    for semantic, names in aliases.items():
        present = [name for name in names if name in reference]
        if not present:
            raise ValueError(f"{field} lacks {semantic}")
        for name in present:
            if reference.get(name) != expected[semantic]:
                raise ValueError(f"{field}.{name} does not match current bytes")
    count = reference.get("score_num_records", reference.get("num_records"))
    if count != len(evidence.image_ids):
        raise ValueError(f"{field} record count does not match current bytes")


def _verify_score_artifact(
    score_dir_value: str | Path,
    *,
    registry: _SnapshotRegistry,
    splits: Mapping[str, _SplitEvidence],
    checkpoints_by_sha: Mapping[str, _CheckpointEvidence],
    cache: dict[Path, _ScoreEvidence],
    expected_reference: Mapping[str, Any],
    role: str,
) -> _ScoreEvidence:
    logical_root, root = registry.resolve_directory(score_dir_value)
    _reject_outer_reference(logical_root, field=f"{role} score directory")
    _reject_outer_reference(root, field=f"{role} resolved score directory")
    manifest_path = root / "manifest.json"
    cache_key = manifest_path.resolve(strict=True)
    if cache_key in cache:
        evidence = cache[cache_key]
        _score_reference_matches(evidence, expected_reference, field=role)
        return evidence

    listing = registry.capture_npz_listing(logical_root)
    manifest_payload, manifest_stamp = registry.capture(manifest_path)
    manifest = _strict_json_bytes(
        manifest_payload, field=f"{role} score manifest"
    )
    if manifest.get("schema_version") != SCORE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"{role} score manifest must use schema version 3")
    if manifest.get("record_integrity_schema") != SCORE_RECORD_INTEGRITY_SCHEMA:
        raise ValueError(f"{role} score manifest has the wrong record hash schema")
    exact = {
        "labels_loaded": True,
        "requested_split": "train",
        "split_role": "train",
        "split_authority_verified": True,
        "spatial_mode": "native",
        "checkpoint_selection_rule": "fixed_last",
        "checkpoint_diagnostic_only": False,
    }
    for name, expected in exact.items():
        if manifest.get(name) != expected:
            raise ValueError(
                f"{role} score manifest {name} must equal {expected!r}"
            )
    for name in ("non_strict_state_loading", "diagnostic_only"):
        value = manifest.get(name, False)
        if not isinstance(value, bool) or value:
            raise ValueError(f"{role} score manifest rejects {name}=true/non-boolean")
    target = manifest.get("target_dataset")
    sources = manifest.get("source_datasets")
    if target not in FORMAL_SOURCE_DOMAIN_NAMES:
        raise ValueError(f"{role} target_dataset is not a canonical formal source")
    if (
        not isinstance(sources, list)
        or not sources
        or any(value not in FORMAL_SOURCE_DOMAIN_NAMES for value in sources)
        or len(set(sources)) != len(sources)
    ):
        raise ValueError(f"{role} source_datasets are not canonical formal sources")
    for name in ("dataset_dir", "output_dir", "weight_path", "checkpoint_path", "split_file"):
        if manifest.get(name) is not None:
            _reject_outer_reference(manifest[name], field=f"{role} manifest {name}")

    checkpoint_sha = _require_sha256(
        manifest.get("weight_sha256"), field=f"{role}.weight_sha256"
    )
    if checkpoint_sha not in checkpoints_by_sha:
        raise ValueError(f"{role} refers to an unbound detector checkpoint")
    weight_path = manifest.get("weight_path", manifest.get("checkpoint_path"))
    if not isinstance(weight_path, str) or not weight_path:
        raise ValueError(f"{role} lacks a detector checkpoint path")
    _logical_weight, resolved_weight = registry.resolve_file(weight_path)
    checkpoint_evidence = checkpoints_by_sha[checkpoint_sha]
    if resolved_weight != checkpoint_evidence.path:
        raise ValueError(f"{role} checkpoint path/SHA binding is inconsistent")
    if set(map(str, sources)) != set(checkpoint_evidence.source_domains):
        raise ValueError(
            f"{role} source_datasets disagree with source_names inside the "
            "safely loaded checkpoint bytes"
        )
    checkpoint_manifest_fields = {
        "checkpoint_epoch": checkpoint_evidence.epoch,
        "warm_flag": checkpoint_evidence.warm_flag,
        "checkpoint_inference_head": checkpoint_evidence.inference_head,
    }
    for field, expected in checkpoint_manifest_fields.items():
        observed = manifest.get(field)
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(
                f"{role} {field} disagrees with the safely loaded checkpoint bytes"
            )

    split = splits[str(target)]
    split_value = manifest.get("split_file")
    if not isinstance(split_value, str) or not split_value:
        raise ValueError(f"{role} lacks split_file")
    _logical_split, resolved_split = registry.resolve_file(split_value)
    if resolved_split != split.path:
        raise ValueError(f"{role} does not use the supplied official train split")
    if manifest.get("split_file_sha256") != split.sha256:
        raise ValueError(f"{role} split_file_sha256 mismatch")
    if manifest.get("split_ordered_ids_sha256") != split.ordered_ids_sha256:
        raise ValueError(f"{role} split_ordered_ids_sha256 mismatch")

    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{role} records must be a non-empty list")
    if manifest.get("num_images") != len(records):
        raise ValueError(f"{role} num_images differs from records length")
    if manifest.get("records_sha256") != score_records_sha256(records):
        raise ValueError(f"{role} records_sha256 mismatch")
    image_ids: list[str] = []
    filenames: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{role} record {index} must be an object")
        image_id = record.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError(f"{role} record {index} has no valid image_id")
        _reject_outer_reference(image_id, field=f"{role} score-record ID")
        record_path = _safe_record_path(root, record.get("file"))
        _reject_outer_reference(record_path, field=f"{role} score-record path")
        record_payload, record_stamp = registry.capture(record_path)
        declared_sha = _require_sha256(
            record.get("sha256"), field=f"{role}.records[{index}].sha256"
        )
        if record_stamp.sha256 != declared_sha:
            raise ValueError(f"{role} score-record SHA mismatch: {record_path}")
        embedded_id = _embedded_score_identity(record_payload, path=record_path)
        del record_payload
        if embedded_id != image_id:
            raise ValueError(f"{role} manifest/NPZ image_id mismatch: {record_path}")
        image_ids.append(image_id)
        filenames.append(record_path.name)
    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"{role} score manifest contains duplicate image IDs")
    if len(set(filenames)) != len(filenames):
        raise ValueError(f"{role} score manifest contains duplicate filenames")
    if tuple(filenames) != listing.npz_names:
        if set(filenames) != set(listing.npz_names):
            raise ValueError(f"{role} score directory/manifest NPZ set differs")
    ids_hash = ordered_ids_sha256(image_ids)
    if manifest.get("ordered_image_ids_sha256") != ids_hash:
        raise ValueError(f"{role} ordered_image_ids_sha256 mismatch")
    if tuple(image_ids) != split.ids:
        raise ValueError(
            f"{role} record IDs/order do not equal the official source-train split"
        )

    evidence = _ScoreEvidence(
        root=root,
        manifest_path=manifest_stamp.resolved_path,
        manifest_sha256=manifest_stamp.sha256,
        records_sha256=str(manifest["records_sha256"]),
        ordered_ids_sha256=ids_hash,
        image_ids=tuple(image_ids),
        target_domain=str(target),
        source_domains=tuple(map(str, sources)),
        checkpoint_sha256=checkpoint_sha,
    )
    _score_reference_matches(evidence, expected_reference, field=role)
    cache[cache_key] = evidence
    return evidence


def _load_grid_from_snapshots(
    manifest_value: str | Path, registry: _SnapshotRegistry
) -> tuple[Any, str, Path]:
    _reject_outer_reference(manifest_value, field="threshold-grid manifest path")
    manifest_payload, manifest_stamp = registry.capture(manifest_value)
    _reject_outer_reference(
        manifest_stamp.resolved_path, field="resolved threshold-grid manifest path"
    )
    manifest = _strict_json_bytes(
        manifest_payload, field="threshold-grid manifest"
    )
    grid_file = manifest.get("grid_file")
    digest_file = manifest.get("digest_file")
    if grid_file != "threshold_grid.npy" or digest_file != "threshold_grid.sha256":
        raise ValueError("Threshold-grid child filenames are not the formal fixed names")
    base = manifest_stamp.logical_path.parent
    grid_payload, _grid_stamp = registry.capture(str(grid_file), base=base)
    digest_payload, _digest_stamp = registry.capture(str(digest_file), base=base)
    with tempfile.TemporaryDirectory(prefix="rc-v4-source-provenance-grid-") as raw:
        temporary = Path(raw)
        os.chmod(temporary, 0o700)
        (temporary / "threshold_grid.json").write_bytes(manifest_payload)
        (temporary / "threshold_grid.npy").write_bytes(grid_payload)
        (temporary / "threshold_grid.sha256").write_bytes(digest_payload)
        artifact = load_logit_grid_artifact(temporary)
    return artifact, manifest_stamp.sha256, manifest_stamp.resolved_path


def _decode_episode_ids(value: Any, *, field: str, row: int) -> tuple[str, ...]:
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(f"{field}[{row}] is not valid JSON") from error
    if (
        not isinstance(decoded, list)
        or not decoded
        or any(not isinstance(item, str) or not item for item in decoded)
    ):
        raise ValueError(f"{field}[{row}] must be a non-empty string list")
    if len(set(decoded)) != len(decoded):
        raise ValueError(f"{field}[{row}] contains duplicate IDs")
    for image_id in decoded:
        _reject_outer_reference(image_id, field=f"episode {field} ID")
    return tuple(decoded)


def _scalar_text(value: Any, *, field: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{field} must be scalar")
    return str(array.item())


def verify_source_only_provenance_v4(
    *,
    project_root: str | Path,
    threshold_grid_manifest: str | Path,
    official_train_split_manifests: Mapping[str, str | Path],
    detector_checkpoints: Sequence[str | Path],
    episode_archives: Sequence[str | Path],
) -> dict[str, Any]:
    """Verify the complete source-only evidence chain for one Gate-C seed.

    ``episode_archives`` must contain exactly the two validation archives from
    the paired source LODO directions (one held IRSTD-1K and one held
    NUDT-SIRST).  Passing training archives as well would duplicate the same
    source episodes and is intentionally rejected by the global A/E identity
    check.
    """

    registry = _SnapshotRegistry(project_root)
    if set(official_train_split_manifests) != set(FORMAL_SOURCE_DOMAIN_NAMES):
        raise ValueError(
            "official_train_split_manifests must contain exactly IRSTD-1K and "
            "NUDT-SIRST"
        )
    if len(detector_checkpoints) != 3:
        raise ValueError("Formal source-only provenance requires exactly 3 checkpoints")
    if len(episode_archives) != 2:
        raise ValueError(
            "Formal paired-LODO Gate C requires exactly 2 validation archives"
        )

    splits: dict[str, _SplitEvidence] = {}
    split_audit: list[dict[str, Any]] = []
    for domain in FORMAL_SOURCE_DOMAIN_NAMES:
        value = official_train_split_manifests[domain]
        _reject_outer_reference(value, field=f"{domain} official split path")
        payload, stamp = registry.capture(value)
        _reject_outer_reference(
            stamp.resolved_path, field=f"{domain} resolved official split path"
        )
        ids = _parse_split_snapshot(payload, path=stamp.resolved_path)
        del payload
        evidence = _SplitEvidence(
            domain=domain,
            path=stamp.resolved_path,
            sha256=stamp.sha256,
            ordered_ids_sha256=ordered_ids_sha256(ids),
            ids=ids,
        )
        splits[domain] = evidence
        split_audit.append(
            {
                "domain": domain,
                "path": str(stamp.resolved_path),
                "sha256": stamp.sha256,
                "ordered_ids_sha256": evidence.ordered_ids_sha256,
                "num_ids": len(ids),
            }
        )

    grid, grid_manifest_sha, grid_manifest_path = _load_grid_from_snapshots(
        threshold_grid_manifest, registry
    )
    grid_manifest = grid.manifest
    if set(grid_manifest.get("source_domains", [])) != set(
        FORMAL_SOURCE_DOMAIN_NAMES
    ):
        raise ValueError("Threshold grid sources are not exactly the canonical pair")
    if set(grid_manifest.get("expected_source_domains", [])) != set(
        FORMAL_SOURCE_DOMAIN_NAMES
    ):
        raise ValueError("Threshold grid expected sources are not canonical")
    if set(grid_manifest.get("source_domain_keys", [])) != set(
        FORMAL_SOURCE_DOMAIN_KEYS
    ):
        raise ValueError("Threshold grid source keys are not canonical")
    if (
        grid_manifest.get("outer_target") != FORMAL_OUTER_DOMAIN_NAME
        or grid_manifest.get("outer_target_key") != FORMAL_OUTER_DOMAIN_KEY
        or grid_manifest.get("outer_target_excluded") is not True
        or grid_manifest.get("outer_target_labels_used") is not False
    ):
        raise ValueError("Threshold grid does not exactly exclude canonical NUAA-SIRST")
    if grid_manifest.get("grid_detector_protocol") != GRID_DETECTOR_PROTOCOL:
        raise ValueError("Threshold grid detector protocol is not source-only")

    checkpoint_audit: list[dict[str, Any]] = []
    checkpoints_by_sha: dict[str, _CheckpointEvidence] = {}
    for value in detector_checkpoints:
        _reject_outer_reference(value, field="detector checkpoint path")
        payload, stamp = registry.capture(value)
        _reject_outer_reference(
            stamp.resolved_path, field="resolved detector checkpoint path"
        )
        if stamp.sha256 in checkpoints_by_sha:
            raise ValueError("Detector checkpoint files are not byte-distinct")
        checkpoint_evidence, metadata_audit = (
            _verify_detector_checkpoint_snapshot(
                payload,
                stamp=stamp,
                registry=registry,
                splits=splits,
            )
        )
        del payload
        checkpoints_by_sha[stamp.sha256] = checkpoint_evidence
        checkpoint_audit.append(metadata_audit)
    grid_checkpoint_hashes = set(
        map(str, grid_manifest.get("detector_checkpoint_sha256s", []))
    )
    if set(checkpoints_by_sha) != grid_checkpoint_hashes:
        raise ValueError(
            "The three real checkpoint path/SHA values do not match the grid manifest"
        )
    observed_checkpoint_source_sets = {
        frozenset(value.source_domains) for value in checkpoints_by_sha.values()
    }
    expected_checkpoint_source_sets = {
        frozenset(("IRSTD-1K",)),
        frozenset(("NUDT-SIRST",)),
        frozenset(FORMAL_SOURCE_DOMAIN_NAMES),
    }
    if observed_checkpoint_source_sets != expected_checkpoint_source_sets:
        raise ValueError(
            "Safely loaded checkpoint bytes must comprise one IRSTD-only inner, "
            "one NUDT-only inner, and one exact two-source outer detector"
        )
    key_to_name = {
        "irstd1k": "IRSTD-1K",
        "nudt": "NUDT-SIRST",
    }
    grid_folds = grid_manifest.get("detector_folds")
    assert isinstance(grid_folds, list)  # load_logit_grid_artifact verified it
    for fold in grid_folds:
        assert isinstance(fold, dict)
        checkpoint_sha = str(fold["detector_checkpoint_sha256"])
        try:
            fold_sources = {
                key_to_name[str(value)] for value in fold["source_domain_keys"]
            }
        except (KeyError, TypeError) as error:
            raise ValueError("Grid detector fold contains a non-canonical source") from error
        if fold_sources != set(checkpoints_by_sha[checkpoint_sha].source_domains):
            raise ValueError(
                "Grid detector-fold source claims disagree with source_names "
                "inside checkpoint bytes"
            )
    outer_checkpoint_sha = str(
        grid_manifest["outer_detector_checkpoint_sha256"]
    )
    if set(checkpoints_by_sha[outer_checkpoint_sha].source_domains) != set(
        FORMAL_SOURCE_DOMAIN_NAMES
    ):
        raise ValueError("Grid outer detector is not the exact two-source checkpoint")
    for checkpoint_sha in grid_manifest["episode_detector_checkpoint_sha256s"]:
        if len(checkpoints_by_sha[str(checkpoint_sha)].source_domains) != 1:
            raise ValueError("Grid episode detector is not a one-source inner checkpoint")

    score_cache: dict[Path, _ScoreEvidence] = {}
    grid_score_audit: list[dict[str, Any]] = []
    grid_inputs = grid_manifest.get("input_score_artifacts")
    assert isinstance(grid_inputs, list)  # enforced by load_logit_grid_artifact
    for index, reference in enumerate(grid_inputs):
        assert isinstance(reference, dict)
        score_dir = reference.get("score_dir")
        score_manifest_value = reference.get("score_manifest")
        if not isinstance(score_dir, str) or not isinstance(score_manifest_value, str):
            raise ValueError("Grid input lacks score_dir/score_manifest path")
        logical_dir, resolved_dir = registry.resolve_directory(score_dir)
        _logical_manifest, resolved_manifest = registry.resolve_file(
            score_manifest_value
        )
        if resolved_manifest != resolved_dir / "manifest.json":
            raise ValueError("Grid score_manifest is not score_dir/manifest.json")
        evidence = _verify_score_artifact(
            logical_dir,
            registry=registry,
            splits=splits,
            checkpoints_by_sha=checkpoints_by_sha,
            cache=score_cache,
            expected_reference=reference,
            role=f"threshold_grid.input_score_artifacts[{index}]",
        )
        if evidence.target_domain != reference.get("target_dataset"):
            raise ValueError("Grid score target differs from its transitive manifest")
        if evidence.checkpoint_sha256 != reference.get("detector_weight_sha256"):
            raise ValueError("Grid score checkpoint differs from grid provenance")
        if set(evidence.source_domains) != set(
            map(str, reference.get("detector_source_datasets", []))
        ):
            raise ValueError("Grid score detector sources differ from grid provenance")
        if evidence.target_domain not in set(evidence.source_domains):
            raise ValueError("Threshold-grid score must be a detector self-score")
        grid_score_audit.append(
            {
                "target_domain": evidence.target_domain,
                "manifest_path": str(evidence.manifest_path),
                "manifest_sha256": evidence.manifest_sha256,
                "records_sha256": evidence.records_sha256,
                "num_records": len(evidence.image_ids),
                "checkpoint_sha256": evidence.checkpoint_sha256,
            }
        )

    global_episode_ids: set[str] = set()
    validation_domains: set[str] = set()
    episode_audit: list[dict[str, Any]] = []
    episode_score_audit_by_manifest: dict[str, dict[str, Any]] = {}
    expected_grid_detector_hashes = set(checkpoints_by_sha)
    for archive_index, archive_value in enumerate(episode_archives):
        _reject_outer_reference(archive_value, field="episode archive path")
        archive_payload, archive_stamp = registry.capture(archive_value)
        _reject_outer_reference(
            archive_stamp.resolved_path, field="resolved episode archive path"
        )
        archive = load_curve_archive(io.BytesIO(archive_payload))
        del archive_payload
        provenance = _strict_json_bytes(
            _scalar_text(
                archive.get("provenance_json"), field="provenance_json"
            ).encode("utf-8"),
            field="episode provenance_json",
        )
        expected_provenance = {
            "protocol": "causal_adaptation_then_future_evaluation",
            "archive_split": "validation",
            "representation": LOGIT_REPRESENTATION,
            "pseudo_target_split": "train",
            "expected_split_role": "train",
            "fold_provenance_verified": True,
            "score_artifact_integrity_verified": True,
            "formal_causal_contract_verified": True,
            "protocol_scope": "formal_causal",
            "threshold_grid_outer_target_excluded": True,
            "allow_unverified_fold_provenance": False,
            "allow_cross_episode_role_reuse": False,
            "cross_episode_role_reuse_detected": False,
        }
        for field, expected in expected_provenance.items():
            if provenance.get(field) != expected:
                raise ValueError(
                    f"Episode provenance {field} must equal {expected!r}"
                )
        if provenance.get("cross_episode_role_reuse_ids") != []:
            raise ValueError("Episode provenance reports cross-role ID reuse")
        if set(provenance.get("pseudo_targets", [])) != set(
            FORMAL_SOURCE_DOMAIN_NAMES
        ):
            raise ValueError("Episode provenance sources are not canonical")
        if set(provenance.get("paired_lodo_validation_domains", [])) != set(
            FORMAL_SOURCE_DOMAIN_NAMES
        ):
            raise ValueError("Episode provenance paired-LODO domains are incomplete")
        if set(provenance.get("threshold_grid_source_domains", [])) != set(
            FORMAL_SOURCE_DOMAIN_KEYS
        ):
            raise ValueError("Episode provenance threshold-grid sources are not canonical")
        if _domain_key(provenance.get("threshold_grid_outer_target_key")) != (
            FORMAL_OUTER_DOMAIN_KEY
        ):
            raise ValueError("Episode provenance does not exclude canonical NUAA-SIRST")
        if provenance.get("threshold_grid_manifest_sha256") != grid_manifest_sha:
            raise ValueError("Episode provenance threshold-grid manifest SHA mismatch")
        if _scalar_text(
            archive["threshold_grid_manifest_sha256"],
            field="threshold_grid_manifest_sha256",
        ) != grid_manifest_sha:
            raise ValueError("Episode archive threshold-grid manifest SHA mismatch")
        if not np.array_equal(
            np.asarray(archive["thresholds"], dtype=np.float32), grid.thresholds
        ):
            raise ValueError("Episode archive thresholds differ from the real grid")
        archive_detector_hashes = set(
            map(
                str,
                np.asarray(
                    archive["threshold_grid_detector_checkpoint_sha256s"]
                ).tolist(),
            )
        )
        if archive_detector_hashes != expected_grid_detector_hashes:
            raise ValueError("Episode archive detector checkpoint set is not transitive")

        integrity_audits = provenance.get("score_artifact_integrity_audits")
        score_map_dirs = provenance.get("score_map_dirs")
        fold_audits = provenance.get("fold_provenance_audits")
        if (
            not isinstance(integrity_audits, list)
            or len(integrity_audits) != 2
            or not isinstance(score_map_dirs, list)
            or len(score_map_dirs) != 2
            or not isinstance(fold_audits, list)
            or len(fold_audits) != 2
        ):
            raise ValueError("Episode provenance lacks the two source score chains")
        referenced_score_dirs = {
            str(Path(value).expanduser()) for value in score_map_dirs
        }
        audit_score_dirs = {
            str(Path(str(item.get("score_dir"))).expanduser())
            for item in integrity_audits
            if isinstance(item, dict)
        }
        if referenced_score_dirs != audit_score_dirs:
            # Resolve below for the authoritative comparison; this early check
            # catches missing/duplicate declared entries without path I/O.
            if len(referenced_score_dirs) != 2 or len(audit_score_dirs) != 2:
                raise ValueError("Episode score_map_dirs/audits are not one-to-one")

        source_scores: dict[str, _ScoreEvidence] = {}
        for score_index, reference in enumerate(integrity_audits):
            if not isinstance(reference, dict):
                raise ValueError("Episode score integrity audit must be an object")
            if (
                reference.get("verified") is not True
                or reference.get("mask_alignment_verified") is not True
                or reference.get("labels_loaded") is not True
            ):
                raise ValueError("Episode source score audit is not formally verified")
            score_dir = reference.get("score_dir")
            if not isinstance(score_dir, str) or not score_dir:
                raise ValueError("Episode source score audit lacks score_dir")
            evidence = _verify_score_artifact(
                score_dir,
                registry=registry,
                splits=splits,
                checkpoints_by_sha=checkpoints_by_sha,
                cache=score_cache,
                expected_reference=reference,
                role=f"episode[{archive_index}].score_artifacts[{score_index}]",
            )
            if evidence.target_domain in source_scores:
                raise ValueError("Episode score chain duplicates a source target")
            if evidence.target_domain in set(evidence.source_domains):
                raise ValueError("Episode score must come from the held-source detector")
            if set(evidence.source_domains) != set(FORMAL_SOURCE_DOMAIN_NAMES).difference(
                {evidence.target_domain}
            ):
                raise ValueError("Episode score detector is not the canonical LODO fold")
            source_scores[evidence.target_domain] = evidence
            episode_score_audit_by_manifest[evidence.manifest_sha256] = {
                "target_domain": evidence.target_domain,
                "manifest_path": str(evidence.manifest_path),
                "manifest_sha256": evidence.manifest_sha256,
                "records_sha256": evidence.records_sha256,
                "num_records": len(evidence.image_ids),
                "checkpoint_sha256": evidence.checkpoint_sha256,
            }
        if set(source_scores) != set(FORMAL_SOURCE_DOMAIN_NAMES):
            raise ValueError("Episode score chain does not cover both formal sources")
        resolved_declared_dirs = {
            registry.resolve_directory(value)[1] for value in score_map_dirs
        }
        if resolved_declared_dirs != {value.root for value in source_scores.values()}:
            raise ValueError("Episode score_map_dirs differ from integrity audits")

        fold_by_target: dict[str, Mapping[str, Any]] = {}
        for fold in fold_audits:
            if not isinstance(fold, dict) or fold.get("verified") is not True:
                raise ValueError("Episode fold provenance audit is not verified")
            target = fold.get("pseudo_target")
            if target not in FORMAL_SOURCE_DOMAIN_NAMES or target in fold_by_target:
                raise ValueError("Episode fold provenance target set is invalid")
            evidence = source_scores[str(target)]
            if (
                fold.get("manifest_sha256") != evidence.manifest_sha256
                or fold.get("detector_weight_sha256")
                != evidence.checkpoint_sha256
                or set(fold.get("source_datasets", []))
                != set(evidence.source_domains)
            ):
                raise ValueError("Episode fold audit is not bound to current score bytes")
            fold_by_target[str(target)] = fold
        if set(fold_by_target) != set(FORMAL_SOURCE_DOMAIN_NAMES):
            raise ValueError("Episode fold audits do not cover both sources")

        targets = np.asarray(archive.get("pseudo_targets"))
        adaptation = np.asarray(archive.get("adaptation_ids"))
        evaluation = np.asarray(archive.get("evaluation_ids"))
        rows = int(np.asarray(archive["statistics"]).shape[0])
        if targets.shape != (rows,) or adaptation.shape != (rows,) or evaluation.shape != (rows,):
            raise ValueError("Episode target/A/E vectors must have one entry per row")
        target_set = set(map(str, targets.tolist()))
        validation_domain = provenance.get("validation_domain")
        if target_set != {validation_domain} or validation_domain not in (
            FORMAL_SOURCE_DOMAIN_NAMES
        ):
            raise ValueError("Validation archive rows do not match validation_domain")
        if validation_domain in validation_domains:
            raise ValueError("Paired validation archives repeat a validation domain")
        validation_domains.add(str(validation_domain))
        adaptation_sizes = np.asarray(archive.get("adaptation_sizes"))
        evaluation_sizes = np.asarray(archive.get("evaluation_sizes"))
        if adaptation_sizes.shape != (rows,) or evaluation_sizes.shape != (rows,):
            raise ValueError("Episode archive lacks A/E size vectors")
        source_score_ids = set(source_scores[str(validation_domain)].image_ids)
        source_split_ids = set(splits[str(validation_domain)].ids)
        num_episode_ids = 0
        for row in range(rows):
            row_a = _decode_episode_ids(
                adaptation[row], field="adaptation_ids", row=row
            )
            row_e = _decode_episode_ids(
                evaluation[row], field="evaluation_ids", row=row
            )
            if len(row_a) != int(adaptation_sizes[row]) or len(row_e) != int(
                evaluation_sizes[row]
            ):
                raise ValueError("Decoded episode IDs differ from stored A/E sizes")
            row_ids = set(row_a).union(row_e)
            if set(row_a).intersection(row_e):
                raise ValueError("Episode reuses an ID between A and E")
            if not row_ids.issubset(source_split_ids):
                raise ValueError("Episode A/E ID is outside the official source train split")
            if not row_ids.issubset(source_score_ids):
                raise ValueError("Episode A/E ID is absent from the bound source score records")
            if global_episode_ids.intersection(row_ids):
                raise ValueError("Episode A/E IDs are not globally unique")
            global_episode_ids.update(row_ids)
            num_episode_ids += len(row_ids)
        episode_audit.append(
            {
                "validation_domain": validation_domain,
                "archive_path": str(archive_stamp.resolved_path),
                "archive_sha256": archive_stamp.sha256,
                "num_episodes": rows,
                "num_unique_a_e_ids": num_episode_ids,
                "score_manifest_sha256": source_scores[
                    str(validation_domain)
                ].manifest_sha256,
            }
        )

    if validation_domains != set(FORMAL_SOURCE_DOMAIN_NAMES):
        raise ValueError("Paired LODO validation archives do not cover both sources")

    # This expensive final pass is intentional: successful publication-facing
    # evidence must be tied to paths that remained byte-identical throughout
    # the transitive verification, including every listed score NPZ.
    registry.assert_unchanged()
    return {
        "schema_version": SOURCE_PROVENANCE_SCHEMA_VERSION,
        "verified": True,
        "formal_source_domains": list(FORMAL_SOURCE_DOMAIN_NAMES),
        "excluded_outer_target": FORMAL_OUTER_DOMAIN_NAME,
        "outer_target_labels_read": False,
        "path_containment_verified": True,
        "immutable_byte_snapshot_verified": True,
        "all_paths_unchanged_after_snapshot": True,
        "checkpoint_metadata_verified_from_bytes": True,
        "checkpoint_safe_load_weights_only": True,
        "checkpoint_test_split_artifacts_read": False,
        "official_train_splits": split_audit,
        "threshold_grid": {
            "manifest_path": str(grid_manifest_path),
            "manifest_sha256": grid_manifest_sha,
            "semantic_sha256": grid.semantic_sha256,
            "detector_protocol": grid_manifest["grid_detector_protocol"],
        },
        "detector_checkpoints": sorted(
            checkpoint_audit, key=lambda item: str(item["sha256"])
        ),
        "grid_score_artifacts": grid_score_audit,
        "episode_score_artifacts": sorted(
            episode_score_audit_by_manifest.values(),
            key=lambda item: str(item["target_domain"]),
        ),
        "validation_archives": sorted(
            episode_audit, key=lambda item: str(item["validation_domain"])
        ),
        "global_unique_episode_a_e_ids": len(global_episode_ids),
        "captured_file_count": registry.num_files,
        "source_chain_sha256": canonical_json_sha256(
            {
                "splits": split_audit,
                "grid_manifest_sha256": grid_manifest_sha,
                "checkpoints": sorted(
                    checkpoint_audit, key=lambda item: str(item["sha256"])
                ),
                "grid_scores": grid_score_audit,
                "episode_scores": sorted(
                    episode_score_audit_by_manifest.values(),
                    key=lambda item: str(item["target_domain"]),
                ),
                "episodes": sorted(
                    episode_audit,
                    key=lambda item: str(item["validation_domain"]),
                ),
            }
        ),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _resolve_cli_input(value: str | Path, *, project_root: Path) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        requested = project_root / requested
    resolved = requested.resolve(strict=True)
    try:
        resolved.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"CLI input escapes project_root: {value!r}") from error
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _resolve_cli_output(value: str | Path, *, project_root: Path) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        requested = project_root / requested
    lexical = Path(os.path.abspath(requested))
    try:
        lexical.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"CLI output escapes project_root: {value!r}") from error
    lexical.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = lexical.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"CLI output parent escapes project_root: {value!r}") from error
    output = resolved_parent / lexical.name
    if output.exists() and output.resolve(strict=True) != output:
        raise ValueError("CLI output must not be a symbolic link")
    return output


def _parse_domain_paths(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        domain, separator, path = value.partition("=")
        if not separator or domain not in FORMAL_SOURCE_DOMAIN_NAMES or not path:
            raise ValueError(
                "--official-train-split must use canonical DOMAIN=PATH with "
                "DOMAIN equal to IRSTD-1K or NUDT-SIRST"
            )
        if domain in result:
            raise ValueError(f"Duplicate official train split for {domain}")
        result[domain] = path
    if set(result) != set(FORMAL_SOURCE_DOMAIN_NAMES):
        raise ValueError(
            "Exactly one official train split is required for each canonical source"
        )
    return result


def validate_source_provenance_run_evidence(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail-closed structural self-check for one published replay record."""

    if not isinstance(payload, Mapping):
        raise ValueError("Source provenance run evidence must be an object")
    if payload.get("schema_version") != SOURCE_PROVENANCE_RUN_SCHEMA_VERSION:
        raise ValueError("Source provenance run evidence schema is unsupported")
    verifier = payload.get("verifier_module")
    execution = payload.get("execution")
    declaration = payload.get("outer_target_access_declaration")
    verification = payload.get("verification")
    if not isinstance(verifier, Mapping):
        raise ValueError("Source provenance evidence lacks verifier_module")
    if not isinstance(execution, Mapping):
        raise ValueError("Source provenance evidence lacks execution")
    if not isinstance(declaration, Mapping):
        raise ValueError("Source provenance evidence lacks outer-target declaration")
    if not isinstance(verification, Mapping):
        raise ValueError("Source provenance evidence lacks verification result")
    _require_sha256(verifier.get("sha256"), field="verifier_module.sha256")
    if (
        not isinstance(verifier.get("path"), str)
        or not verifier.get("path")
        or not isinstance(verifier.get("size_bytes"), int)
        or isinstance(verifier.get("size_bytes"), bool)
        or int(verifier["size_bytes"]) <= 0
    ):
        raise ValueError("verifier_module path/size evidence is invalid")
    command = execution.get("command_argv")
    parameters = execution.get("parameters")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(value, str) or not value for value in command)
        or not isinstance(parameters, Mapping)
    ):
        raise ValueError("Execution command/parameters are invalid")
    for field in ("started_utc", "finished_utc"):
        value = execution.get(field)
        if not isinstance(value, str) or not value.endswith("Z"):
            raise ValueError(f"execution.{field} must be a UTC timestamp")
    elapsed = execution.get("elapsed_seconds")
    peak_rss = execution.get("peak_rss_kib")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not np.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise ValueError("execution.elapsed_seconds is invalid")
    if (
        isinstance(peak_rss, bool)
        or not isinstance(peak_rss, int)
        or peak_rss <= 0
        or execution.get("peak_rss_bytes") != peak_rss * 1024
    ):
        raise ValueError("execution peak-RSS evidence is invalid")
    if declaration != {
        "excluded_outer_target": FORMAL_OUTER_DOMAIN_NAME,
        "outer_target_split_read": False,
        "outer_target_labels_read": False,
        "outer_target_masks_read": False,
        "statement": (
            "This replay accepts no NUAA path argument and did not read the "
            "NUAA split, labels, or masks."
        ),
    }:
        raise ValueError("Outer-target non-access declaration is not exact")
    if (
        verification.get("schema_version") != SOURCE_PROVENANCE_SCHEMA_VERSION
        or verification.get("verified") is not True
        or verification.get("excluded_outer_target")
        != FORMAL_OUTER_DOMAIN_NAME
        or verification.get("outer_target_labels_read") is not False
        or verification.get("all_paths_unchanged_after_snapshot") is not True
        or verification.get("checkpoint_metadata_verified_from_bytes") is not True
        or verification.get("checkpoint_safe_load_weights_only") is not True
        or verification.get("checkpoint_test_split_artifacts_read") is not False
    ):
        raise ValueError("Nested source provenance verification is not valid")
    split_rows = verification.get("official_train_splits")
    checkpoints = verification.get("detector_checkpoints")
    threshold_grid = verification.get("threshold_grid")
    grid_scores = verification.get("grid_score_artifacts")
    episode_scores = verification.get("episode_score_artifacts")
    validation_archives = verification.get("validation_archives")
    if (
        not isinstance(split_rows, list)
        or len(split_rows) != 2
        or any(not isinstance(item, dict) for item in split_rows)
        or not isinstance(checkpoints, list)
        or len(checkpoints) != 3
        or any(not isinstance(item, dict) for item in checkpoints)
        or not isinstance(threshold_grid, Mapping)
        or not isinstance(grid_scores, list)
        or not isinstance(episode_scores, list)
        or not isinstance(validation_archives, list)
    ):
        raise ValueError("Nested source-chain evidence lists are incomplete")
    split_by_domain = {
        str(item.get("domain")): item for item in split_rows
    }
    if set(split_by_domain) != set(FORMAL_SOURCE_DOMAIN_NAMES):
        raise ValueError("Evidence official-train split domains are not canonical")
    for domain, split in split_by_domain.items():
        if (
            not isinstance(split.get("path"), str)
            or not split.get("path")
            or not isinstance(split.get("num_ids"), int)
            or isinstance(split.get("num_ids"), bool)
            or split["num_ids"] <= 0
        ):
            raise ValueError(f"Evidence official split for {domain} is invalid")
        _require_sha256(split.get("sha256"), field=f"split[{domain}].sha256")
        _require_sha256(
            split.get("ordered_ids_sha256"),
            field=f"split[{domain}].ordered_ids_sha256",
        )

    exact_checkpoint_metadata = _formal_checkpoint_exact_metadata()
    observed_source_sets: set[frozenset[str]] = set()
    for index, checkpoint in enumerate(checkpoints):
        for field, expected in exact_checkpoint_metadata.items():
            observed = checkpoint.get(field)
            if type(observed) is not type(expected) or observed != expected:
                raise ValueError(
                    f"Published checkpoint evidence {index}.{field} is invalid"
                )
        _require_sha256(
            checkpoint.get("sha256"), field=f"checkpoint[{index}].sha256"
        )
        metadata_sha = _require_sha256(
            checkpoint.get("metadata_sha256"),
            field=f"checkpoint[{index}].metadata_sha256",
        )
        if (
            not isinstance(checkpoint.get("path"), str)
            or not checkpoint.get("path")
            or not isinstance(checkpoint.get("size_bytes"), int)
            or isinstance(checkpoint.get("size_bytes"), bool)
            or checkpoint["size_bytes"] <= 0
        ):
            raise ValueError(f"Published checkpoint evidence {index} path/size invalid")
        source_names = checkpoint.get("source_names")
        records = checkpoint.get("source_split_records")
        if (
            not isinstance(source_names, list)
            or not source_names
            or any(value not in FORMAL_SOURCE_DOMAIN_NAMES for value in source_names)
            or len(set(source_names)) != len(source_names)
            or not isinstance(records, list)
            or len(records) != len(source_names)
            or any(not isinstance(item, dict) for item in records)
        ):
            raise ValueError(f"Published checkpoint evidence {index} source scope invalid")
        observed_source_sets.add(frozenset(map(str, source_names)))
        if [record.get("name") for record in records] != source_names:
            raise ValueError(
                f"Published checkpoint evidence {index} split-record order invalid"
            )
        for record in records:
            name = str(record["name"])
            split = split_by_domain[name]
            expected_dataset_root = str(Path(str(split["path"])).parent.parent)
            if (
                record.get("dataset_root") != expected_dataset_root
                or record.get("train_split_file") != split["path"]
                or record.get("train_split_file_sha256") != split["sha256"]
                or record.get("train_ordered_ids_sha256")
                != split["ordered_ids_sha256"]
                or record.get("num_train_samples") != split["num_ids"]
                or record.get("train_test_id_overlap") is not False
            ):
                raise ValueError(
                    f"Published checkpoint {index} official-train binding is invalid"
                )
        safe_load = {
            "weights_only": True,
            "map_location": "cpu",
            "unsafe_pickle_fallback_used": False,
        }
        if (
            checkpoint.get("safe_load") != safe_load
            or checkpoint.get(
                "source_split_records_only_canonical_official_train"
            )
            is not True
            or checkpoint.get("test_split_artifacts_read") is not False
        ):
            raise ValueError(f"Published checkpoint evidence {index} safety invalid")
        selected_metadata = {
            **exact_checkpoint_metadata,
            "source_names": source_names,
            "source_split_records": records,
            "source_split_records_only_canonical_official_train": True,
            "test_split_artifacts_read": False,
            "safe_load": safe_load,
        }
        if canonical_json_sha256(selected_metadata) != metadata_sha:
            raise ValueError(
                f"Published checkpoint evidence {index} metadata SHA mismatch"
            )
    if observed_source_sets != {
        frozenset(("IRSTD-1K",)),
        frozenset(("NUDT-SIRST",)),
        frozenset(FORMAL_SOURCE_DOMAIN_NAMES),
    }:
        raise ValueError("Published checkpoint source roles are incomplete")

    grid_manifest_sha = _require_sha256(
        threshold_grid.get("manifest_sha256"),
        field="verification.threshold_grid.manifest_sha256",
    )
    source_chain = {
        "splits": split_rows,
        "grid_manifest_sha256": grid_manifest_sha,
        "checkpoints": sorted(checkpoints, key=lambda item: str(item["sha256"])),
        "grid_scores": grid_scores,
        "episode_scores": sorted(
            episode_scores, key=lambda item: str(item["target_domain"])
        ),
        "episodes": sorted(
            validation_archives,
            key=lambda item: str(item["validation_domain"]),
        ),
    }
    recorded_source_chain_sha = _require_sha256(
        verification.get("source_chain_sha256"),
        field="verification.source_chain_sha256",
    )
    if canonical_json_sha256(source_chain) != recorded_source_chain_sha:
        raise ValueError("Published source_chain_sha256 does not match its evidence")
    return dict(payload)


def load_source_provenance_run_evidence(
    path: str | Path,
) -> tuple[dict[str, Any], str]:
    """Load one published run from exact bytes and return its file SHA-256."""

    source = Path(path).expanduser().resolve(strict=True)
    raw = source.read_bytes()
    decoded = _strict_json_bytes(raw, field="source provenance run evidence")
    return validate_source_provenance_run_evidence(decoded), _sha256_bytes(raw)


def _atomic_write_json(
    path: Path, payload: Mapping[str, Any], *, overwrite: bool
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Evidence output exists; pass --overwrite to replace it: {path}"
        )
    serialized = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def build_source_provenance_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the formal v4 source-only provenance chain and atomically "
            "publish a self-describing JSON evidence record."
        )
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--threshold-grid-manifest", required=True)
    parser.add_argument(
        "--official-train-split",
        action="append",
        required=True,
        metavar="DOMAIN=PATH",
    )
    parser.add_argument(
        "--detector-checkpoint", action="append", required=True
    )
    parser.add_argument("--episode-archive", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_source_provenance_argument_parser().parse_args(raw_argv)
    project_root = Path(args.project_root).expanduser().resolve(strict=True)
    if not project_root.is_dir():
        raise NotADirectoryError(project_root)
    raw_splits = _parse_domain_paths(args.official_train_split)
    split_paths = {
        domain: _resolve_cli_input(value, project_root=project_root)
        for domain, value in raw_splits.items()
    }
    grid_path = _resolve_cli_input(
        args.threshold_grid_manifest, project_root=project_root
    )
    checkpoint_paths = [
        _resolve_cli_input(value, project_root=project_root)
        for value in args.detector_checkpoint
    ]
    episode_paths = [
        _resolve_cli_input(value, project_root=project_root)
        for value in args.episode_archive
    ]
    output_path = _resolve_cli_output(args.output, project_root=project_root)
    module_path = Path(__file__).resolve(strict=True)
    module_snapshot = module_path.read_bytes()
    module_sha = _sha256_bytes(module_snapshot)
    started_utc = _utc_now()
    started_monotonic = time.perf_counter()
    verification = verify_source_only_provenance_v4(
        project_root=project_root,
        threshold_grid_manifest=grid_path,
        official_train_split_manifests=split_paths,
        detector_checkpoints=checkpoint_paths,
        episode_archives=episode_paths,
    )
    elapsed = time.perf_counter() - started_monotonic
    finished_utc = _utc_now()
    if _sha256_file(module_path) != module_sha:
        raise ValueError("Verifier module changed during the provenance replay")
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    parameters = {
        "project_root": str(project_root),
        "threshold_grid_manifest": str(grid_path),
        "official_train_split_manifests": {
            domain: str(split_paths[domain]) for domain in FORMAL_SOURCE_DOMAIN_NAMES
        },
        "detector_checkpoints": [str(value) for value in checkpoint_paths],
        "episode_archives": [str(value) for value in episode_paths],
        "output": str(output_path),
        "overwrite": bool(args.overwrite),
    }
    evidence = {
        "schema_version": SOURCE_PROVENANCE_RUN_SCHEMA_VERSION,
        "verifier_module": {
            "path": str(module_path),
            "sha256": module_sha,
            "size_bytes": len(module_snapshot),
        },
        "execution": {
            "command_argv": [
                sys.executable,
                "-m",
                "risk_curve.source_provenance_v4",
                *raw_argv,
            ],
            "parameters": parameters,
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "elapsed_seconds": float(elapsed),
            "peak_rss_kib": peak_rss_kib,
            "peak_rss_bytes": peak_rss_kib * 1024,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_only_requested": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        },
        "outer_target_access_declaration": {
            "excluded_outer_target": FORMAL_OUTER_DOMAIN_NAME,
            "outer_target_split_read": False,
            "outer_target_labels_read": False,
            "outer_target_masks_read": False,
            "statement": (
                "This replay accepts no NUAA path argument and did not read the "
                "NUAA split, labels, or masks."
            ),
        },
        "verification": verification,
    }
    validate_source_provenance_run_evidence(evidence)
    _atomic_write_json(output_path, evidence, overwrite=bool(args.overwrite))
    published, file_sha = load_source_provenance_run_evidence(output_path)
    if published != evidence:
        raise ValueError("Published provenance JSON differs from the validated payload")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "output_sha256": file_sha,
                "schema_version": published["schema_version"],
                "source_chain_sha256": published["verification"][
                    "source_chain_sha256"
                ],
                "verified": True,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "FORMAL_OUTER_DOMAIN_KEY",
    "FORMAL_OUTER_DOMAIN_NAME",
    "FORMAL_SOURCE_DOMAIN_KEYS",
    "FORMAL_SOURCE_DOMAIN_NAMES",
    "SOURCE_PROVENANCE_SCHEMA_VERSION",
    "SOURCE_PROVENANCE_RUN_SCHEMA_VERSION",
    "build_source_provenance_argument_parser",
    "load_source_provenance_run_evidence",
    "main",
    "validate_source_provenance_run_evidence",
    "verify_source_only_provenance_v4",
]


if __name__ == "__main__":
    raise SystemExit(main())
