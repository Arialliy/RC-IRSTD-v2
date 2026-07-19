"""Cryptographic integrity helpers for exported score-map artifacts.

The manifest is the ordering authority.  Version-3 manifests bind every NPZ
byte-for-byte, bind the ordered record list, and reject unlisted NPZ files.
They also require a validated mask-alignment evidence sub-chain.
Legacy manifests remain readable in diagnostic paths, but callers making a
formal claim must pass ``require_integrity=True``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from data_ext.mask_alignment import (
    MASK_ALIGNMENT_NOT_LOADED_POLICY,
    MASK_ALIGNMENT_POLICY,
    validate_mask_alignment_evidence,
)


SCORE_MANIFEST_SCHEMA_VERSION = 3
SCORE_RECORD_INTEGRITY_SCHEMA = "rc-v3-score-records-sha256-v1"
SCORE_MASK_ALIGNMENT_SCHEMA = "rc-v3-mask-alignment-audit-v1"
RAW_LOGIT_SCORE_REPRESENTATION = (
    "raw_logit_float32+sigmoid_probability_float32"
)
RAW_LOGIT_DTYPE = "float32"
PROBABILITY_DTYPE = "float32"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one regular file."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_ids_sha256(image_ids: Sequence[str]) -> str:
    """Hash an ordered ID sequence with an unambiguous canonical encoding."""

    values = [str(value) for value in image_ids]
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def score_records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash the ordered record identity/file/content triples canonically."""

    canonical = []
    for position, record in enumerate(records):
        canonical.append(
            {
                "position": position,
                "image_id": str(record.get("image_id", "")),
                "file": str(record.get("file", "")),
                "sha256": str(record.get("sha256", "")),
            }
        )
    payload = json.dumps(
        {"schema": SCORE_RECORD_INTEGRITY_SCHEMA, "records": canonical},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _record_path(root: Path, value: Any) -> Path:
    filename = str(value)
    relative = Path(filename)
    if (
        not filename
        or relative.is_absolute()
        or len(relative.parts) != 1
        or relative.name != filename
        or relative.suffix.lower() != ".npz"
    ):
        raise ValueError(f"Unsafe score-map record path: {filename!r}")
    return root / relative


def _boolean_scalar(value: Any, *, name: str) -> bool:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean scalar")
    return bool(array.item())


def _integer_pair(value: Any, *, name: str) -> tuple[int, int]:
    array = np.asarray(value)
    if array.shape != (2,) or array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be a two-element integer array")
    return int(array[0]), int(array[1])


def _float_scalar(value: Any, *, name: str) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{name} must be a numeric scalar")
    result = float(array.item())
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _string_scalar(value: Any, *, name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be a string scalar")
    return str(array.item())


def _validate_formal_numeric_arrays(
    payload: Mapping[str, Any],
    *,
    path: Path,
    expected_masks: bool | None,
) -> None:
    """Validate the writer's numeric NPZ contract for formal v3 consumers."""

    if not isinstance(expected_masks, bool):
        raise ValueError("Formal score map requires boolean labels_loaded evidence")
    probability = np.asarray(payload["prob"])
    if probability.dtype != np.float32:
        raise ValueError(f"Formal probability map must be float32: {path}")
    if probability.ndim != 2 or probability.size == 0:
        raise ValueError(f"Formal probability map must be a non-empty 2-D array: {path}")
    if not np.isfinite(probability).all() or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError(f"Formal probability map must be finite and in [0, 1]: {path}")

    if "gray" not in payload:
        raise ValueError(f"Formal score map lacks gray array: {path}")
    gray = np.asarray(payload["gray"])
    if gray.dtype != np.float32:
        raise ValueError(f"Formal gray map must be float32: {path}")
    if gray.ndim != 2 or gray.shape != probability.shape:
        raise ValueError(f"Formal gray/probability shape mismatch: {path}")
    if not np.isfinite(gray).all() or np.any((gray < 0.0) | (gray > 1.0)):
        raise ValueError(f"Formal gray map must be finite and in [0, 1]: {path}")

    if expected_masks:
        mask = np.asarray(payload["mask"])
        if mask.dtype not in {np.dtype(np.uint8), np.dtype(np.bool_)}:
            raise ValueError(f"Formal mask must use uint8 or bool dtype: {path}")
        if mask.ndim != 2 or mask.shape != probability.shape:
            raise ValueError(f"Formal mask/probability shape mismatch: {path}")
        if not np.isin(np.unique(mask), (0, 1, False, True)).all():
            raise ValueError(f"Formal mask must be binary: {path}")


def verify_score_map_directory(
    score_dir: str | Path,
    *,
    require_integrity: bool = False,
    require_masks: bool | None = None,
) -> tuple[dict[str, Any] | None, list[Path], dict[str, Any]]:
    """Resolve and verify the ordered score-map record set.

    When a manifest exists, its order is authoritative.  Any recorded digest is
    always enforced.  ``require_integrity`` additionally requires the complete
    version-3 chain and therefore fails closed for legacy manifests/directories.
    """

    root = Path(score_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Score-map directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        if require_integrity:
            raise ValueError("Formal score-map input requires manifest.json")
        paths = sorted(root.glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No .npz score maps found under {root}")
        return None, paths, {
            "verified": False,
            "diagnostic_reason": "legacy_directory_without_manifest",
            "records_sha256": None,
            "ordered_image_ids_sha256": None,
        }

    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, dict):
        raise ValueError("Score-map manifest must decode to a JSON object")
    records = raw_manifest.get("records")
    if (
        records is None
        and not require_integrity
        and raw_manifest.get("schema_version") != SCORE_MANIFEST_SCHEMA_VERSION
    ):
        paths = sorted(root.glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No .npz score maps found under {root}")
        return raw_manifest, paths, {
            "verified": False,
            "diagnostic_reason": "legacy_manifest_without_record_index",
            "manifest_sha256": file_sha256(manifest_path),
            "records_sha256": None,
            "ordered_image_ids_sha256": None,
            "num_records": len(paths),
        }
    if not isinstance(records, list):
        raise ValueError("Score-map manifest records must be a list")
    if not records:
        if (
            not require_integrity
            and raw_manifest.get("schema_version") != SCORE_MANIFEST_SCHEMA_VERSION
        ):
            paths = sorted(root.glob("*.npz"))
            if paths:
                return raw_manifest, paths, {
                    "verified": False,
                    "diagnostic_reason": "legacy_manifest_with_empty_record_index",
                    "manifest_sha256": file_sha256(manifest_path),
                    "records_sha256": None,
                    "ordered_image_ids_sha256": None,
                    "num_records": len(paths),
                }
        raise ValueError("Score-map manifest records must be non-empty")
    if raw_manifest.get("num_images") != len(records):
        raise ValueError("Score-map manifest num_images differs from records length")
    manifest_labels_loaded = raw_manifest.get("labels_loaded")
    if manifest_labels_loaded is not None and not isinstance(manifest_labels_loaded, bool):
        raise ValueError("Score-map manifest labels_loaded must be boolean")
    expected_masks = (
        require_masks if require_masks is not None else manifest_labels_loaded
    )
    if (
        require_masks is not None
        and manifest_labels_loaded is not None
        and require_masks != manifest_labels_loaded
    ):
        raise ValueError("Score-map label mode does not match the caller's requirement")

    # Raw-logit exports are an optional, additive v3 representation.  Legacy
    # probability-only v3 manifests deliberately omit these fields and remain
    # valid.  Once any raw-representation field is declared, however, the
    # entire precision contract is mandatory and is checked in every NPZ.
    raw_precision_fields = (
        "score_representation",
        "probability_dtype",
        "logit_dtype",
        "probability_transform",
        "probability_clipping",
        "inference_autocast_enabled",
    )
    raw_precision_values = {
        name: raw_manifest.get(name) for name in raw_precision_fields
    }
    raw_precision_present = [
        name for name, value in raw_precision_values.items() if value is not None
    ]
    if raw_precision_present and len(raw_precision_present) != len(
        raw_precision_fields
    ):
        missing = sorted(set(raw_precision_fields).difference(raw_precision_present))
        raise ValueError(
            "Score-map manifest raw-logit precision contract is incomplete: "
            + ", ".join(missing)
        )
    raw_logit_complete = bool(
        len(raw_precision_present) == len(raw_precision_fields)
    )
    if raw_logit_complete:
        if (
            raw_precision_values["score_representation"]
            != RAW_LOGIT_SCORE_REPRESENTATION
        ):
            raise ValueError("Score-map manifest score_representation is unsupported")
        if raw_precision_values["probability_dtype"] != PROBABILITY_DTYPE:
            raise ValueError("Raw-logit manifest probability_dtype must be float32")
        if raw_precision_values["logit_dtype"] != RAW_LOGIT_DTYPE:
            raise ValueError("Raw-logit manifest logit_dtype must be float32")
        if raw_precision_values["probability_transform"] != "sigmoid":
            raise ValueError("Raw-logit manifest probability_transform must be sigmoid")
        if raw_precision_values["probability_clipping"] != "none":
            raise ValueError("Raw-logit manifest probability_clipping must be none")
        if raw_precision_values["inference_autocast_enabled"] is not False:
            raise ValueError("Raw-logit inference must explicitly disable autocast")

    alignment_field_names = (
        "mask_alignment_schema",
        "mask_alignment_policy",
        "mask_alignment_count",
        "mask_aligned_sample_ids",
    )
    alignment_values = {
        name: raw_manifest.get(name) for name in alignment_field_names
    }
    alignment_fields_present = [
        name for name, value in alignment_values.items() if value is not None
    ]
    if alignment_fields_present and len(alignment_fields_present) != len(
        alignment_field_names
    ):
        missing = sorted(set(alignment_field_names).difference(alignment_fields_present))
        raise ValueError(
            "Score-map manifest mask-alignment provenance is incomplete: "
            + ", ".join(missing)
        )
    alignment_complete = bool(
        len(alignment_fields_present) == len(alignment_field_names)
        and alignment_values["mask_alignment_schema"] == SCORE_MASK_ALIGNMENT_SCHEMA
    )
    if alignment_fields_present and not alignment_complete:
        raise ValueError("Score-map manifest mask_alignment_schema is unsupported")
    if alignment_complete:
        alignment_policy = alignment_values["mask_alignment_policy"]
        if not isinstance(alignment_policy, str):
            raise ValueError("Score-map manifest mask_alignment_policy must be a string")
        alignment_count = alignment_values["mask_alignment_count"]
        if (
            isinstance(alignment_count, bool)
            or not isinstance(alignment_count, int)
            or alignment_count < 0
        ):
            raise ValueError(
                "Score-map manifest mask_alignment_count must be non-negative integer"
            )
        alignment_ids = alignment_values["mask_aligned_sample_ids"]
        if (
            not isinstance(alignment_ids, list)
            or any(not isinstance(value, str) or not value for value in alignment_ids)
            or len(set(alignment_ids)) != len(alignment_ids)
        ):
            raise ValueError(
                "Score-map manifest mask_aligned_sample_ids must be unique strings"
            )
        if manifest_labels_loaded is True and alignment_policy != MASK_ALIGNMENT_POLICY:
            raise ValueError("Labeled score-map manifest has an invalid alignment policy")
        if (
            manifest_labels_loaded is False
            and alignment_policy != MASK_ALIGNMENT_NOT_LOADED_POLICY
        ):
            raise ValueError("Mask-free score-map manifest has an invalid alignment policy")

    paths: list[Path] = []
    ids: list[str] = []
    filenames: list[str] = []
    has_complete_record_hashes = True
    embedded_aligned_ids: list[str] = []
    for index, raw_record in enumerate(records):
        if not isinstance(raw_record, dict):
            raise ValueError(f"Score-map manifest record {index} must be an object")
        image_id = raw_record.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError(f"Score-map manifest record {index} has no valid image_id")
        path = _record_path(root, raw_record.get("file"))
        if not path.is_file():
            raise FileNotFoundError(f"Manifest score-map file does not exist: {path}")
        if path.resolve().parent != root:
            raise ValueError(f"Score-map record escapes its artifact directory: {path}")
        recorded_sha = raw_record.get("sha256")
        if recorded_sha is None:
            has_complete_record_hashes = False
        else:
            if not _valid_sha256(recorded_sha):
                raise ValueError(f"Score-map record {index} has an invalid sha256")
            if file_sha256(path) != recorded_sha:
                raise ValueError(f"Score-map record sha256 mismatch: {path}")
        try:
            with np.load(path, allow_pickle=False) as payload:
                if "prob" not in payload:
                    raise ValueError(f"Score map lacks prob array: {path}")
                if expected_masks is True and "mask" not in payload:
                    raise ValueError(f"Labeled score map lacks mask array: {path}")
                if expected_masks is False and "mask" in payload:
                    raise ValueError(f"Mask-free score map unexpectedly embeds a mask: {path}")
                if require_integrity:
                    _validate_formal_numeric_arrays(
                        payload,
                        path=path,
                        expected_masks=expected_masks,
                    )
                if "labels_loaded" in payload:
                    embedded_labels = _boolean_scalar(
                        payload["labels_loaded"], name="labels_loaded"
                    )
                    if (
                        manifest_labels_loaded is not None
                        and embedded_labels != manifest_labels_loaded
                    ):
                        raise ValueError(f"Manifest/NPZ labels_loaded mismatch: {path}")
                elif require_integrity:
                    raise ValueError(f"Formal score map lacks labels_loaded evidence: {path}")
                if "image_id" not in payload:
                    if require_integrity:
                        raise ValueError(f"Formal score map lacks embedded image_id: {path}")
                elif str(np.asarray(payload["image_id"]).item()) != image_id:
                    raise ValueError(
                        f"Manifest/NPZ image_id mismatch for record {index}: {path}"
                    )
                if "shape" in raw_record:
                    shape = [int(value) for value in np.asarray(payload["prob"]).shape]
                    if list(raw_record["shape"]) != shape:
                        raise ValueError(f"Manifest/NPZ shape mismatch: {path}")
                if raw_logit_complete:
                    raw_record_fields = {
                        "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
                        "probability_dtype": PROBABILITY_DTYPE,
                        "logit_dtype": RAW_LOGIT_DTYPE,
                        "probability_transform": "sigmoid",
                        "probability_clipping": "none",
                        "inference_autocast_enabled": False,
                    }
                    for field, expected in raw_record_fields.items():
                        if raw_record.get(field) != expected:
                            raise ValueError(
                                f"Raw-logit manifest record {index} has invalid {field}"
                            )
                    required_raw_payload = {
                        "logit",
                        "score_representation",
                        "probability_dtype",
                        "logit_dtype",
                        "probability_transform",
                        "probability_clipping",
                        "inference_autocast_enabled",
                    }
                    missing_raw_payload = required_raw_payload.difference(payload.files)
                    if missing_raw_payload:
                        raise ValueError(
                            f"Raw-logit score map lacks precision evidence: {path}; missing="
                            + ", ".join(sorted(missing_raw_payload))
                        )
                    probability = np.asarray(payload["prob"])
                    logits = np.asarray(payload["logit"])
                    if probability.dtype != np.float32:
                        raise ValueError(f"Raw-logit probability is not float32: {path}")
                    if logits.dtype != np.float32:
                        raise ValueError(f"Raw-logit array is not float32: {path}")
                    if logits.shape != probability.shape:
                        raise ValueError(f"Raw-logit/probability shape mismatch: {path}")
                    if not np.isfinite(logits).all():
                        raise ValueError(f"Raw-logit array contains NaN or infinity: {path}")
                    embedded_raw_fields = {
                        "score_representation": _string_scalar(
                            payload["score_representation"],
                            name="score_representation",
                        ),
                        "probability_dtype": _string_scalar(
                            payload["probability_dtype"], name="probability_dtype"
                        ),
                        "logit_dtype": _string_scalar(
                            payload["logit_dtype"], name="logit_dtype"
                        ),
                        "probability_transform": _string_scalar(
                            payload["probability_transform"],
                            name="probability_transform",
                        ),
                        "probability_clipping": _string_scalar(
                            payload["probability_clipping"],
                            name="probability_clipping",
                        ),
                        "inference_autocast_enabled": _boolean_scalar(
                            payload["inference_autocast_enabled"],
                            name="inference_autocast_enabled",
                        ),
                    }
                    if embedded_raw_fields != raw_record_fields:
                        raise ValueError(
                            f"Manifest/NPZ raw-logit precision evidence mismatch: {path}"
                        )
                    # This is a semantic synchronization check in addition to
                    # the byte-level record digest.  Use a float64 reference
                    # sigmoid followed by float32 rounding and allow only the
                    # tiny libm/backend variation expected at one ULP.
                    logits64 = logits.astype(np.float64)
                    expected_probability = np.empty(logits.shape, dtype=np.float64)
                    nonnegative = logits64 >= 0.0
                    expected_probability[nonnegative] = 1.0 / (
                        1.0 + np.exp(-logits64[nonnegative])
                    )
                    exp_logits = np.exp(logits64[~nonnegative])
                    expected_probability[~nonnegative] = exp_logits / (
                        1.0 + exp_logits
                    )
                    expected_probability = expected_probability.astype(np.float32)
                    if not np.allclose(
                        probability,
                        expected_probability,
                        rtol=1e-6,
                        atol=1e-7,
                    ):
                        raise ValueError(
                            f"Raw-logit probability is not synchronized with logits: {path}"
                        )
                if alignment_complete:
                    if not isinstance(manifest_labels_loaded, bool):
                        raise ValueError(
                            "Mask-alignment provenance requires boolean labels_loaded"
                        )
                    record_alignment_keys = {
                        "mask_alignment_applied",
                        "mask_original_hw",
                        "mask_aspect_relative_error",
                    }
                    missing_record_keys = record_alignment_keys.difference(raw_record)
                    if missing_record_keys:
                        raise ValueError(
                            f"Score-map record {index} lacks alignment fields: "
                            + ", ".join(sorted(missing_record_keys))
                        )
                    record_applied = raw_record["mask_alignment_applied"]
                    if not isinstance(record_applied, bool):
                        raise ValueError(
                            f"Score-map record {index} alignment flag must be boolean"
                        )
                    raw_mask_hw = raw_record["mask_original_hw"]
                    if (
                        not isinstance(raw_mask_hw, list)
                        or len(raw_mask_hw) != 2
                        or any(
                            isinstance(value, bool) or not isinstance(value, int)
                            for value in raw_mask_hw
                        )
                    ):
                        raise ValueError(
                            f"Score-map record {index} mask_original_hw is invalid"
                        )
                    record_mask_hw = (int(raw_mask_hw[0]), int(raw_mask_hw[1]))
                    record_error = raw_record["mask_aspect_relative_error"]
                    if isinstance(record_error, bool) or not isinstance(
                        record_error, (int, float)
                    ):
                        raise ValueError(
                            f"Score-map record {index} mask aspect error is invalid"
                        )
                    record_error = float(record_error)
                    required_payload_keys = {
                        "original_hw",
                        "mask_alignment_applied",
                        "mask_original_hw",
                        "mask_aspect_relative_error",
                        "mask_alignment_policy",
                    }
                    missing_payload_keys = required_payload_keys.difference(payload.files)
                    if missing_payload_keys:
                        raise ValueError(
                            f"Formal score map lacks alignment evidence: {path}; missing="
                            + ", ".join(sorted(missing_payload_keys))
                        )
                    embedded_image_hw = _integer_pair(
                        payload["original_hw"], name="original_hw"
                    )
                    embedded_applied = _boolean_scalar(
                        payload["mask_alignment_applied"],
                        name="mask_alignment_applied",
                    )
                    embedded_mask_hw = _integer_pair(
                        payload["mask_original_hw"], name="mask_original_hw"
                    )
                    embedded_error = _float_scalar(
                        payload["mask_aspect_relative_error"],
                        name="mask_aspect_relative_error",
                    )
                    embedded_policy = _string_scalar(
                        payload["mask_alignment_policy"],
                        name="mask_alignment_policy",
                    )
                    if (
                        record_applied != embedded_applied
                        or record_mask_hw != embedded_mask_hw
                        or not np.isclose(
                            record_error,
                            embedded_error,
                            rtol=1e-12,
                            atol=1e-15,
                        )
                    ):
                        raise ValueError(
                            f"Manifest/NPZ mask-alignment evidence mismatch: {path}"
                        )
                    validate_mask_alignment_evidence(
                        labels_loaded=manifest_labels_loaded,
                        image_hw=embedded_image_hw,
                        original_mask_hw=embedded_mask_hw,
                        applied=embedded_applied,
                        relative_error=embedded_error,
                        policy=embedded_policy,
                    )
                    if embedded_applied:
                        embedded_aligned_ids.append(image_id)
        except (OSError, ValueError) as error:
            if isinstance(error, ValueError):
                raise
            raise ValueError(f"Unreadable score-map NPZ: {path}") from error
        paths.append(path)
        ids.append(image_id)
        filenames.append(path.name)

    if len(set(ids)) != len(ids):
        raise ValueError("Score-map manifest contains duplicate image IDs")
    if len(set(filenames)) != len(filenames):
        raise ValueError("Score-map manifest contains duplicate filenames")
    if alignment_complete:
        if int(alignment_values["mask_alignment_count"]) != len(
            embedded_aligned_ids
        ):
            raise ValueError("Score-map manifest mask_alignment_count mismatch")
        if list(alignment_values["mask_aligned_sample_ids"]) != embedded_aligned_ids:
            raise ValueError("Score-map manifest mask_aligned_sample_ids mismatch")
    on_disk = {path.name for path in root.glob("*.npz")}
    if on_disk != set(filenames):
        missing = sorted(set(filenames).difference(on_disk))
        extra = sorted(on_disk.difference(filenames))
        raise ValueError(
            "Score-map directory/manifest NPZ set differs: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    computed_records_sha = score_records_sha256(records)
    recorded_records_sha = raw_manifest.get("records_sha256")
    if recorded_records_sha is not None:
        if not _valid_sha256(recorded_records_sha) or recorded_records_sha != computed_records_sha:
            raise ValueError("Score-map manifest records_sha256 mismatch")
    computed_ids_sha = ordered_ids_sha256(ids)
    recorded_ids_sha = raw_manifest.get("ordered_image_ids_sha256")
    if recorded_ids_sha is not None:
        if not _valid_sha256(recorded_ids_sha) or recorded_ids_sha != computed_ids_sha:
            raise ValueError("Score-map manifest ordered_image_ids_sha256 mismatch")

    split_path_value = raw_manifest.get("split_file")
    split_sha_value = raw_manifest.get("split_file_sha256")
    split_ids_sha_value = raw_manifest.get("split_ordered_ids_sha256")
    split_fields = (split_path_value, split_sha_value, split_ids_sha_value)
    if any(value is not None for value in split_fields):
        if not all(value is not None for value in split_fields):
            raise ValueError("Score-map manifest split provenance is incomplete")
        split_path = Path(str(split_path_value)).expanduser()
        if not split_path.is_file():
            raise FileNotFoundError(f"Score-map split artifact does not exist: {split_path}")
        if not _valid_sha256(split_sha_value) or file_sha256(split_path) != split_sha_value:
            raise ValueError("Score-map split_file_sha256 mismatch")
        if split_ids_sha_value != computed_ids_sha:
            raise ValueError("Score-map split order differs from exported record order")
        from data_ext.split_utils import ensure_unique_sample_ids, read_split_file

        split_ids = ensure_unique_sample_ids(read_split_file(split_path))
        if ordered_ids_sha256(split_ids) != computed_ids_sha:
            raise ValueError("Current split contents/order differ from score-map records")

    complete = bool(
        raw_manifest.get("schema_version") == SCORE_MANIFEST_SCHEMA_VERSION
        and raw_manifest.get("record_integrity_schema") == SCORE_RECORD_INTEGRITY_SCHEMA
        and isinstance(manifest_labels_loaded, bool)
        and has_complete_record_hashes
        and alignment_complete
        and _valid_sha256(recorded_records_sha)
        and _valid_sha256(recorded_ids_sha)
    )
    if require_integrity and not complete:
        raise ValueError(
            "Formal score-map input requires a complete version-3 hash and "
            "mask-alignment evidence chain"
        )
    return raw_manifest, paths, {
        "verified": complete,
        "mask_alignment_verified": bool(complete and alignment_complete),
        "diagnostic_reason": None if complete else "legacy_manifest_without_complete_hash_chain",
        "manifest_sha256": file_sha256(manifest_path),
        "records_sha256": computed_records_sha if complete else None,
        "ordered_image_ids_sha256": computed_ids_sha,
        "num_records": len(records),
    }


__all__ = [
    "PROBABILITY_DTYPE",
    "RAW_LOGIT_DTYPE",
    "RAW_LOGIT_SCORE_REPRESENTATION",
    "SCORE_MANIFEST_SCHEMA_VERSION",
    "SCORE_MASK_ALIGNMENT_SCHEMA",
    "SCORE_RECORD_INTEGRITY_SCHEMA",
    "file_sha256",
    "ordered_ids_sha256",
    "score_records_sha256",
    "verify_score_map_directory",
]
