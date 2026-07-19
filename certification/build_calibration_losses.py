"""Build bounded per-image false-alarm loss curves for conformal control.

The two physical risks have deliberately different units:

* pixel risk: false-positive pixels / evaluated pixels;
* component risk: false-positive connected components / megapixel.

The default loss is the binary joint budget-violation indicator.  Its expected
value is exactly the marginal probability of violating either budget, so the
finite-sample corrected bound maps directly to JointBSR.  The older clipped
risk-ratio loss remains available as an explicitly diagnostic mode; it must
not be reported as a budget-satisfaction guarantee.  Component counts can
increase when a component splits as the threshold rises, so their curve is
replaced by a conservative suffix-max envelope before either loss is formed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.artifact_integrity import (
    PROBABILITY_DTYPE,
    RAW_LOGIT_DTYPE,
    RAW_LOGIT_SCORE_REPRESENTATION,
    SCORE_MANIFEST_SCHEMA_VERSION,
    SCORE_RECORD_INTEGRITY_SCHEMA,
    file_sha256,
    verify_score_map_directory,
)
from risk_curve.representation import (
    GRID_DETECTOR_PROTOCOL,
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_PREDICTION_RULE,
    LOGIT_REPRESENTATION,
    PROBABILITY_REPRESENTATION,
    empty_action_contract,
    load_logit_grid_artifact,
    logit_threshold_grid_sha256,
    validate_logit_threshold_grid,
)
from risk_curve.threshold_grid import threshold_grid_sha256


LOSS_SCHEMA_VERSION = "rc-v2-calibration-loss-v2"
RAW_LOGIT_LOSS_SCHEMA_VERSION = "rc-v4-calibration-loss-v1-raw-logit"
LOSS_MODE_BUDGET_VIOLATION = "budget_violation"
LOSS_MODE_RISK_RATIO = "risk_ratio"
PIXEL_BUDGET_UNIT = "false_positive_pixels_per_evaluated_pixel"
COMPONENT_BUDGET_UNIT = "false_positive_components_per_megapixel"
PROTOCOL_SCHEMA_VERSION = "rc-v2-score-map-protocol-v1"
RAW_LOGIT_PROTOCOL_SCHEMA_VERSION = "rc-v4-score-map-protocol-v1-raw-logit"
COUNT_ARCHIVE_INTEGRITY_SCHEMA = "rc-v2-count-archive-sha256-v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_PROTOCOL_KEYS = frozenset(
    {
        "schema_version",
        "detector_weight_sha256",
        "score_type",
        "warm_flag",
        "spatial_mode",
        "pad_multiple",
        "base_hw",
        "target_dataset",
        "source_datasets",
        "threshold_grid_sha256",
        "matching_rule",
        "centroid_distance",
        "connectivity",
        "min_component_area",
        "component_monotone_transform",
    }
)
_RAW_LOGIT_CANONICAL_PROTOCOL_KEYS = frozenset(
    {
        "schema_version",
        "detector_weight_sha256",
        "score_type",
        "representation",
        "score_representation",
        "warm_flag",
        "spatial_mode",
        "pad_multiple",
        "base_hw",
        "target_dataset",
        "source_datasets",
        "threshold_grid_schema_version",
        "threshold_grid_sha256",
        "threshold_grid_detector_protocol",
        "threshold_grid_detector_checkpoint_sha256s",
        "threshold_grid_outer_detector_checkpoint_sha256",
        "threshold_grid_episode_detector_checkpoint_sha256s",
        "prediction_rule",
        "empty_action",
        "matching_rule",
        "centroid_distance",
        "connectivity",
        "min_component_area",
        "component_monotone_transform",
    }
)


def protocol_fingerprint(protocol: Mapping[str, Any]) -> str:
    """Hash the frozen detector/preprocessing/matching protocol canonically."""

    payload = json.dumps(dict(protocol), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_PATTERN.fullmatch(value))


def _distinct_detector_hashes(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    hashes = [str(item) for item in value]
    if any(not _valid_sha256(item) for item in hashes):
        raise ValueError(f"{field} must contain lowercase SHA-256 digests")
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"{field} must contain distinct detector checkpoints")
    return hashes


def _detector_role_hashes(
    all_hashes: Any,
    outer_hash: Any,
    episode_hashes: Any,
) -> tuple[list[str], str, list[str]]:
    hashes = _distinct_detector_hashes(
        all_hashes, field="threshold_grid_detector_checkpoint_sha256s"
    )
    outer = str(outer_hash)
    if not _valid_sha256(outer) or outer not in hashes:
        raise ValueError("outer detector checkpoint hash is invalid")
    episodes = _distinct_detector_hashes(
        episode_hashes,
        field="threshold_grid_episode_detector_checkpoint_sha256s",
    )
    if outer in episodes:
        raise ValueError("outer detector checkpoint must not be an episode detector")
    if set(hashes) != set(episodes).union({outer}):
        raise ValueError(
            "detector hashes must equal outer detector plus episode detectors"
        )
    return hashes, outer, episodes


def _validate_raw_logit_formal_protocol(
    protocol: Mapping[str, Any],
    recorded_fingerprint: str | None,
) -> tuple[dict[str, Any], str]:
    """Validate the v4 score/count protocol without probability re-indexing."""

    keys = frozenset(protocol)
    missing = sorted(_RAW_LOGIT_CANONICAL_PROTOCOL_KEYS.difference(keys))
    extra = sorted(keys.difference(_RAW_LOGIT_CANONICAL_PROTOCOL_KEYS))
    if missing or extra:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise ValueError(
            "Formal raw-logit protocol is not complete/canonical: "
            + "; ".join(details)
        )
    if protocol.get("schema_version") != RAW_LOGIT_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("Formal raw-logit protocol schema_version is unsupported")
    detector_sha = str(protocol.get("detector_weight_sha256"))
    if not _valid_sha256(detector_sha):
        raise ValueError(
            "Formal raw-logit protocol detector_weight_sha256 is invalid"
        )
    if protocol.get("score_type") != "sigmoid_probability":
        raise ValueError(
            "Raw-logit exports must retain the canonical probability companion"
        )
    if protocol.get("representation") != LOGIT_REPRESENTATION:
        raise ValueError("Formal v4 protocol requires raw_logit_float32")
    if protocol.get("score_representation") != RAW_LOGIT_SCORE_REPRESENTATION:
        raise ValueError("Formal v4 score representation contract is unsupported")
    if not isinstance(protocol.get("warm_flag"), bool):
        raise ValueError("Formal raw-logit protocol warm_flag must be boolean")
    spatial_mode = str(protocol.get("spatial_mode", "")).strip()
    if spatial_mode not in {"native", "resize"}:
        raise ValueError("Formal raw-logit spatial_mode must be native or resize")
    pad_multiple = protocol.get("pad_multiple")
    if (
        isinstance(pad_multiple, bool)
        or not isinstance(pad_multiple, int)
        or pad_multiple < 1
    ):
        raise ValueError("Formal raw-logit pad_multiple must be positive")
    base_hw = protocol.get("base_hw")
    if spatial_mode == "native":
        if base_hw is not None:
            raise ValueError("Formal native raw-logit protocol requires base_hw=null")
    elif (
        not isinstance(base_hw, list)
        or len(base_hw) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in base_hw
        )
    ):
        raise ValueError("Formal resize raw-logit base_hw is invalid")
    target_dataset = str(protocol.get("target_dataset", "")).strip()
    raw_sources = protocol.get("source_datasets")
    if not target_dataset:
        raise ValueError("Formal raw-logit target_dataset must be non-empty")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("Formal raw-logit source_datasets must be non-empty")
    sources = [str(value).strip() for value in raw_sources]
    if any(not value for value in sources):
        raise ValueError("Formal raw-logit source_datasets contains an empty name")
    if len({value.casefold() for value in sources}) != len(sources):
        raise ValueError("Formal raw-logit source_datasets must be unique")
    canonical_sources = sorted(sources, key=str.casefold)
    if sources != canonical_sources:
        raise ValueError("Formal raw-logit source_datasets must be sorted")
    if target_dataset.casefold() in {value.casefold() for value in sources}:
        raise ValueError("Formal raw-logit detector sources contain target_dataset")
    if protocol.get("threshold_grid_schema_version") != LOGIT_GRID_SCHEMA_VERSION:
        raise ValueError("Formal raw-logit grid schema is unsupported")
    grid_sha = str(protocol.get("threshold_grid_sha256"))
    if not _valid_sha256(grid_sha):
        raise ValueError("Formal raw-logit semantic grid hash is invalid")
    if protocol.get("threshold_grid_detector_protocol") != GRID_DETECTOR_PROTOCOL:
        raise ValueError("Formal raw-logit grid detector protocol is unsupported")
    detector_hashes, outer_detector_hash, episode_detector_hashes = _detector_role_hashes(
        protocol.get("threshold_grid_detector_checkpoint_sha256s"),
        protocol.get("threshold_grid_outer_detector_checkpoint_sha256"),
        protocol.get("threshold_grid_episode_detector_checkpoint_sha256s"),
    )
    if protocol.get("prediction_rule") != LOGIT_PREDICTION_RULE:
        raise ValueError("Formal raw-logit prediction rule is unsupported")
    if protocol.get("empty_action") != empty_action_contract():
        raise ValueError("Formal raw-logit reject action must be external +inf/index null")
    matching_rule = str(protocol.get("matching_rule"))
    if matching_rule not in {"overlap", "centroid"}:
        raise ValueError("Formal raw-logit matching_rule is invalid")
    try:
        centroid_distance = float(protocol.get("centroid_distance"))
    except (TypeError, ValueError) as error:
        raise ValueError("Formal raw-logit centroid_distance is invalid") from error
    if not np.isfinite(centroid_distance) or centroid_distance < 0.0:
        raise ValueError("Formal raw-logit centroid_distance is invalid")
    connectivity = protocol.get("connectivity")
    if isinstance(connectivity, bool) or connectivity not in {1, 2, 4, 8}:
        raise ValueError("Formal raw-logit connectivity is invalid")
    min_component_area = protocol.get("min_component_area")
    if (
        isinstance(min_component_area, bool)
        or not isinstance(min_component_area, int)
        or min_component_area < 1
    ):
        raise ValueError("Formal raw-logit min_component_area is invalid")
    if protocol.get("component_monotone_transform") != "per_image_suffix_max":
        raise ValueError("Formal raw-logit component transform is invalid")
    canonical = {
        "schema_version": RAW_LOGIT_PROTOCOL_SCHEMA_VERSION,
        "detector_weight_sha256": detector_sha,
        "score_type": "sigmoid_probability",
        "representation": LOGIT_REPRESENTATION,
        "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
        "warm_flag": protocol["warm_flag"],
        "spatial_mode": spatial_mode,
        "pad_multiple": pad_multiple,
        "base_hw": base_hw,
        "target_dataset": target_dataset,
        "source_datasets": canonical_sources,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": grid_sha,
        "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": detector_hashes,
        "threshold_grid_outer_detector_checkpoint_sha256": outer_detector_hash,
        "threshold_grid_episode_detector_checkpoint_sha256s": (
            episode_detector_hashes
        ),
        "prediction_rule": LOGIT_PREDICTION_RULE,
        "empty_action": empty_action_contract(),
        "matching_rule": matching_rule,
        "centroid_distance": centroid_distance,
        "connectivity": int(connectivity),
        "min_component_area": min_component_area,
        "component_monotone_transform": "per_image_suffix_max",
    }
    if dict(protocol) != canonical:
        raise ValueError("Formal raw-logit protocol values are not canonical")
    computed = protocol_fingerprint(canonical)
    if recorded_fingerprint is not None and (
        not _valid_sha256(recorded_fingerprint)
        or recorded_fingerprint != computed
    ):
        raise ValueError("Recorded protocol_fingerprint does not match raw-logit protocol")
    return canonical, computed


def _hash_array(digest: "hashlib._Hash", name: str, value: np.ndarray) -> None:
    """Update a digest with one named ndarray using a portable encoding."""

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError(f"Count archive array {name!r} must not use object dtype")
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    if array.dtype.kind in {"U", "S"}:
        values = [str(item) for item in array.reshape(-1).tolist()]
        digest.update(
            json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
    else:
        canonical = np.ascontiguousarray(array)
        if canonical.dtype.byteorder == ">" or (
            canonical.dtype.byteorder == "=" and not np.little_endian
        ):
            canonical = canonical.byteswap().view(canonical.dtype.newbyteorder("<"))
        digest.update(canonical.dtype.str.replace(">", "<").encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical.tobytes(order="C"))
    digest.update(b"\0")


def count_archive_payload_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash all count-archive arrays except the digest field itself."""

    digest = hashlib.sha256()
    digest.update(COUNT_ARCHIVE_INTEGRITY_SCHEMA.encode("ascii"))
    digest.update(b"\0")
    for name in sorted(arrays):
        if name == "count_archive_payload_sha256":
            continue
        _hash_array(digest, name, np.asarray(arrays[name]))
    return digest.hexdigest()


def verify_count_curve_archive_integrity(
    path: str | Path,
    *,
    require_integrity: bool = False,
) -> dict[str, Any]:
    """Verify an archive's embedded digest; fail closed for formal callers."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Count-curve archive does not exist: {source}")
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    raw_schema = arrays.get("count_archive_integrity_schema")
    raw_digest = arrays.get("count_archive_payload_sha256")
    schema = str(raw_schema.item()) if raw_schema is not None and raw_schema.ndim == 0 else None
    recorded = str(raw_digest.item()) if raw_digest is not None and raw_digest.ndim == 0 else None
    complete = schema == COUNT_ARCHIVE_INTEGRITY_SCHEMA and _valid_sha256(recorded)
    if raw_schema is not None and schema != COUNT_ARCHIVE_INTEGRITY_SCHEMA:
        raise ValueError("Count archive integrity schema is invalid")
    if raw_digest is not None:
        if not _valid_sha256(recorded):
            raise ValueError("Count archive payload digest is invalid")
        computed = count_archive_payload_sha256(arrays)
        if computed != recorded:
            raise ValueError("Count archive payload sha256 mismatch")
    else:
        computed = None
    if require_integrity and not complete:
        raise ValueError(
            "Formal calibration requires a count archive with an embedded payload hash"
        )
    return {
        "verified": bool(complete),
        "payload_sha256": recorded if complete else None,
        "file_sha256": file_sha256(source),
        "diagnostic_reason": None if complete else "legacy_archive_without_payload_hash",
    }


def validate_formal_protocol(
    protocol: Mapping[str, Any],
    recorded_fingerprint: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate and return the exact canonical protocol used for formal runs.

    Fingerprints are only integrity checks over this complete object.  A bare
    fingerprint, an incomplete legacy object, or a detector trained on the
    target domain is never sufficient for a formal claim.
    """

    if not isinstance(protocol, Mapping):
        raise ValueError("Formal protocol must be a JSON object")
    if protocol.get("schema_version") == RAW_LOGIT_PROTOCOL_SCHEMA_VERSION:
        return _validate_raw_logit_formal_protocol(
            protocol, recorded_fingerprint
        )
    keys = frozenset(protocol)
    missing = sorted(_CANONICAL_PROTOCOL_KEYS.difference(keys))
    extra = sorted(keys.difference(_CANONICAL_PROTOCOL_KEYS))
    if missing or extra:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise ValueError("Formal protocol is not complete/canonical: " + "; ".join(details))
    if protocol["schema_version"] != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("Formal protocol schema_version is unsupported")
    detector_sha = str(protocol["detector_weight_sha256"])
    if not _valid_sha256(detector_sha):
        raise ValueError("Formal protocol detector_weight_sha256 must be 64 lowercase hex digits")
    score_type = str(protocol["score_type"]).strip()
    if score_type != "sigmoid_probability":
        raise ValueError("Formal protocol requires sigmoid_probability scores")
    if not isinstance(protocol["warm_flag"], bool):
        raise ValueError("Formal protocol warm_flag must be boolean")
    spatial_mode = str(protocol["spatial_mode"]).strip()
    if spatial_mode not in {"native", "resize"}:
        raise ValueError("Formal protocol spatial_mode must be native or resize")
    pad_multiple = protocol["pad_multiple"]
    if isinstance(pad_multiple, bool) or not isinstance(pad_multiple, int) or pad_multiple < 1:
        raise ValueError("Formal protocol pad_multiple must be a positive integer")
    base_hw = protocol["base_hw"]
    if spatial_mode == "native":
        if base_hw is not None:
            raise ValueError("Formal native protocol must use base_hw=null")
    else:
        if (
            not isinstance(base_hw, list)
            or len(base_hw) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in base_hw)
        ):
            raise ValueError("Formal resize protocol base_hw must contain two positive integers")
    target_dataset = str(protocol["target_dataset"]).strip()
    if not target_dataset:
        raise ValueError("Formal protocol target_dataset must be non-empty")
    raw_sources = protocol["source_datasets"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("Formal protocol source_datasets must be a non-empty list")
    source_datasets = [str(value).strip() for value in raw_sources]
    if any(not value for value in source_datasets):
        raise ValueError("Formal protocol source_datasets contains an empty name")
    source_keys = [value.casefold() for value in source_datasets]
    if len(set(source_keys)) != len(source_keys):
        raise ValueError("Formal protocol source_datasets must be unique")
    canonical_sources = sorted(source_datasets, key=str.casefold)
    if source_datasets != canonical_sources:
        raise ValueError("Formal protocol source_datasets must use canonical sorted order")
    if target_dataset.casefold() in set(source_keys):
        raise ValueError("Formal protocol detector source_datasets contains target_dataset")
    grid_sha = str(protocol["threshold_grid_sha256"])
    if not _valid_sha256(grid_sha):
        raise ValueError("Formal protocol threshold_grid_sha256 is invalid")
    matching_rule = str(protocol["matching_rule"])
    if matching_rule not in {"overlap", "centroid"}:
        raise ValueError("Formal protocol matching_rule is invalid")
    centroid_distance = protocol["centroid_distance"]
    if isinstance(centroid_distance, bool):
        raise ValueError("Formal protocol centroid_distance must be finite and non-negative")
    centroid_distance = float(centroid_distance)
    if not np.isfinite(centroid_distance) or centroid_distance < 0.0:
        raise ValueError("Formal protocol centroid_distance must be finite and non-negative")
    connectivity = protocol["connectivity"]
    if isinstance(connectivity, bool) or connectivity not in {1, 2, 4, 8}:
        raise ValueError("Formal protocol connectivity is invalid")
    min_component_area = protocol["min_component_area"]
    if (
        isinstance(min_component_area, bool)
        or not isinstance(min_component_area, int)
        or min_component_area < 1
    ):
        raise ValueError("Formal protocol min_component_area must be a positive integer")
    if protocol["component_monotone_transform"] != "per_image_suffix_max":
        raise ValueError("Formal protocol component monotone transform is invalid")
    canonical = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "detector_weight_sha256": detector_sha,
        "score_type": score_type,
        "warm_flag": protocol["warm_flag"],
        "spatial_mode": spatial_mode,
        "pad_multiple": pad_multiple,
        "base_hw": base_hw,
        "target_dataset": target_dataset,
        "source_datasets": canonical_sources,
        "threshold_grid_sha256": grid_sha,
        "matching_rule": matching_rule,
        "centroid_distance": centroid_distance,
        "connectivity": int(connectivity),
        "min_component_area": min_component_area,
        "component_monotone_transform": "per_image_suffix_max",
    }
    if dict(protocol) != canonical:
        raise ValueError("Formal protocol values are not in canonical representation")
    computed = protocol_fingerprint(canonical)
    if recorded_fingerprint is not None:
        if not _valid_sha256(recorded_fingerprint) or recorded_fingerprint != computed:
            raise ValueError("Recorded protocol_fingerprint does not match canonical protocol")
    return canonical, computed


def score_map_protocol(
    score_dir: str | Path,
    thresholds: np.ndarray,
    *,
    matching_rule: str,
    centroid_distance: float,
    connectivity: int,
    min_component_area: int,
    representation: str = PROBABILITY_REPRESENTATION,
    threshold_grid_detector_protocol: str | None = None,
    threshold_grid_detector_checkpoint_sha256s: Sequence[str] | None = None,
    threshold_grid_outer_detector_checkpoint_sha256: str | None = None,
    threshold_grid_episode_detector_checkpoint_sha256s: Sequence[str] | None = None,
) -> tuple[dict[str, Any], str]:
    """Resolve protocol fields shared by warm-up, calibration, and test splits."""

    manifest_path = Path(score_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            "Formal certification requires score-map manifest.json provenance"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = ("score_type", "warm_flag", "spatial_mode", "target_dataset")
    missing = [name for name in required if name not in manifest]
    if missing:
        raise ValueError(
            "Score-map manifest lacks protocol fields: " + ", ".join(missing)
        )
    detector_sha = manifest.get("weight_sha256")
    weight_path = manifest.get("weight_path")
    if detector_sha is None and weight_path and Path(weight_path).is_file():
        detector_sha = hashlib.sha256(Path(weight_path).read_bytes()).hexdigest()
    if detector_sha is None:
        raise ValueError(
            "Score-map manifest must record weight_sha256 (or a readable weight_path)"
        )
    detector_sha = str(detector_sha).strip().lower()
    if not _valid_sha256(detector_sha):
        raise ValueError("Score-map detector weight_sha256 must be 64 lowercase hex digits")
    source_datasets = manifest.get("source_datasets")
    if source_datasets is None and manifest.get("source_dataset") is not None:
        source_datasets = [manifest["source_dataset"]]
    if source_datasets is not None:
        if not isinstance(source_datasets, list) or any(
            not str(value).strip() for value in source_datasets
        ):
            raise ValueError("source_datasets must be a list of non-empty names")
        cleaned_sources = [str(value).strip() for value in source_datasets]
        if len({value.casefold() for value in cleaned_sources}) != len(cleaned_sources):
            raise ValueError("source_datasets must be unique")
        source_datasets = sorted(cleaned_sources, key=str.casefold)
    target_dataset = str(manifest["target_dataset"]).strip()
    if not target_dataset:
        raise ValueError("target_dataset must be non-empty")
    if source_datasets and target_dataset.casefold() in {
        value.casefold() for value in source_datasets
    }:
        raise ValueError("Detector source_datasets contains the target_dataset")
    if representation == LOGIT_REPRESENTATION:
        expected_raw = {
            "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
            "probability_dtype": PROBABILITY_DTYPE,
            "logit_dtype": RAW_LOGIT_DTYPE,
            "probability_transform": "sigmoid",
            "probability_clipping": "none",
            "inference_autocast_enabled": False,
        }
        for field, expected in expected_raw.items():
            if manifest.get(field) != expected:
                raise ValueError(
                    f"Raw-logit score manifest requires {field}={expected!r}"
                )
        grid = validate_logit_threshold_grid(np.asarray(thresholds))
        if threshold_grid_detector_protocol != GRID_DETECTOR_PROTOCOL:
            raise ValueError("Raw-logit score protocol requires the source-only grid protocol")
        detector_hashes, outer_detector_hash, episode_detector_hashes = (
            _detector_role_hashes(
            threshold_grid_detector_checkpoint_sha256s,
            threshold_grid_outer_detector_checkpoint_sha256,
            threshold_grid_episode_detector_checkpoint_sha256s,
            )
        )
        if detector_sha != outer_detector_hash:
            raise ValueError(
                "Raw-logit count score maps must use the outer-final detector"
            )
        protocol = {
            "schema_version": RAW_LOGIT_PROTOCOL_SCHEMA_VERSION,
            "detector_weight_sha256": detector_sha,
            "score_type": str(manifest["score_type"]),
            "representation": LOGIT_REPRESENTATION,
            "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
            "warm_flag": bool(manifest["warm_flag"]),
            "spatial_mode": str(manifest["spatial_mode"]),
            "pad_multiple": manifest.get("pad_multiple"),
            "base_hw": (
                manifest.get("base_hw")
                if manifest["spatial_mode"] == "resize"
                else None
            ),
            "target_dataset": target_dataset,
            "source_datasets": source_datasets,
            "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
            "threshold_grid_sha256": logit_threshold_grid_sha256(grid),
            "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
            "threshold_grid_detector_checkpoint_sha256s": detector_hashes,
            "threshold_grid_outer_detector_checkpoint_sha256": (
                outer_detector_hash
            ),
            "threshold_grid_episode_detector_checkpoint_sha256s": (
                episode_detector_hashes
            ),
            "prediction_rule": LOGIT_PREDICTION_RULE,
            "empty_action": empty_action_contract(),
            "matching_rule": matching_rule,
            "centroid_distance": float(centroid_distance),
            "connectivity": int(connectivity),
            "min_component_area": int(min_component_area),
            "component_monotone_transform": "per_image_suffix_max",
        }
        canonical, fingerprint = validate_formal_protocol(protocol)
        return canonical, fingerprint
    if representation != PROBABILITY_REPRESENTATION:
        raise ValueError(f"Unsupported score representation: {representation!r}")
    protocol = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "detector_weight_sha256": detector_sha,
        "score_type": str(manifest["score_type"]),
        "warm_flag": bool(manifest["warm_flag"]),
        "spatial_mode": str(manifest["spatial_mode"]),
        "pad_multiple": manifest.get("pad_multiple"),
        "base_hw": manifest.get("base_hw") if manifest["spatial_mode"] == "resize" else None,
        "target_dataset": target_dataset,
        "source_datasets": source_datasets,
        "threshold_grid_sha256": threshold_grid_sha256(thresholds),
        "matching_rule": matching_rule,
        "centroid_distance": float(centroid_distance),
        "connectivity": int(connectivity),
        "min_component_area": int(min_component_area),
        "component_monotone_transform": "per_image_suffix_max",
    }
    return protocol, protocol_fingerprint(protocol)


def conservative_suffix_max(curves: np.ndarray) -> np.ndarray:
    """Return the least suffix-max majorant of one or more risk curves.

    Thresholds are assumed to increase from left to right.  The returned
    curves are therefore non-increasing and never smaller than the input.
    """

    values = np.asarray(curves, dtype=np.float64)
    if values.ndim not in (1, 2):
        raise ValueError("curves must have shape [T] or [N, T]")
    if values.shape[-1] == 0:
        raise ValueError("curves must contain at least one threshold")
    if not np.isfinite(values).all():
        raise ValueError("curves contain NaN or infinite values")
    return np.maximum.accumulate(values[..., ::-1], axis=-1)[..., ::-1]


def _normalise_image_ids(image_ids: Sequence[str] | np.ndarray) -> np.ndarray:
    ids = np.asarray(image_ids).astype(str).reshape(-1)
    if ids.size == 0:
        raise ValueError("image_ids must not be empty")
    if any(not item for item in ids.tolist()):
        raise ValueError("image_ids must not contain empty values")
    unique, counts = np.unique(ids, return_counts=True)
    duplicates = unique[counts > 1]
    if duplicates.size:
        preview = ", ".join(duplicates[:5].tolist())
        raise ValueError(f"Duplicate image IDs in one split: {preview}")
    return ids


def _curve_matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape [N, T]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return array


def _pixel_exposure(total_pixels: np.ndarray, num_images: int) -> np.ndarray:
    exposure = np.asarray(total_pixels, dtype=np.float64)
    if exposure.ndim == 0:
        exposure = np.full(num_images, float(exposure), dtype=np.float64)
    exposure = exposure.reshape(-1)
    if exposure.shape != (num_images,):
        raise ValueError("total_pixels must be a scalar or have shape [N]")
    if not np.isfinite(exposure).all() or np.any(exposure <= 0.0):
        raise ValueError("total_pixels must contain finite positive exposures")
    return exposure


def _threshold_array(
    thresholds: np.ndarray,
    num_thresholds: int,
    *,
    representation: str,
) -> np.ndarray:
    if representation == LOGIT_REPRESENTATION:
        raw = validate_logit_threshold_grid(np.asarray(thresholds))
        grid = raw
    elif representation == PROBABILITY_REPRESENTATION:
        grid = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    else:
        raise ValueError(f"Unsupported score representation: {representation!r}")
    if grid.shape != (num_thresholds,):
        raise ValueError(f"thresholds must have shape [{num_thresholds}]")
    if not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0.0):
        raise ValueError("thresholds must be finite and strictly increasing")
    if representation == PROBABILITY_REPRESENTATION and np.any(
        (grid < 0.0) | (grid > 1.0)
    ):
        raise ValueError("probability thresholds must lie in [0, 1]")
    return grid


@dataclass(frozen=True)
class CalibrationLosses:
    """Physical risk curves and their bounded per-image loss curves."""

    image_ids: np.ndarray
    thresholds: np.ndarray
    false_positive_pixels: np.ndarray
    false_positive_components: np.ndarray
    total_pixels: np.ndarray
    pixel_risk: np.ndarray
    component_risk_raw: np.ndarray
    component_risk_envelope: np.ndarray
    pixel_loss: np.ndarray
    component_loss: np.ndarray
    joint_loss: np.ndarray
    pixel_budget: float
    component_budget: float
    loss_mode: str
    tp_object_counts: np.ndarray | None = None
    gt_object_counts: np.ndarray | None = None
    representation: str = PROBABILITY_REPRESENTATION
    threshold_grid_schema_version: str | None = None
    threshold_grid_sha256_value: str | None = None
    threshold_grid_manifest_sha256: str | None = None
    threshold_grid_detector_protocol: str | None = None
    threshold_grid_detector_checkpoint_sha256s: tuple[str, ...] = ()
    threshold_grid_outer_detector_checkpoint_sha256: str | None = None
    threshold_grid_episode_detector_checkpoint_sha256s: tuple[str, ...] = ()

    @property
    def num_images(self) -> int:
        return int(self.joint_loss.shape[0])

    @property
    def num_thresholds(self) -> int:
        return int(self.joint_loss.shape[1])

    def assumptions(self) -> list[str]:
        assumptions = [
            "pixel risk is measured as false-positive pixels per evaluated pixel",
            "component risk is measured as false-positive connected components per megapixel",
            "pixel false-positive counts are non-increasing over the threshold grid",
            "component risk uses a conservative per-image suffix-max envelope",
            "joint loss lies in [0, 1] and is non-increasing over the fixed grid",
        ]
        if self.loss_mode == LOSS_MODE_BUDGET_VIOLATION:
            assumptions.extend(
                [
                    "each marginal loss is the indicator that realised risk strictly exceeds its budget",
                    "joint loss is the indicator that either physical budget is violated",
                ]
            )
        else:
            assumptions.extend(
                [
                    "each marginal diagnostic loss is clip(realised_risk / physical_budget, 0, 1)",
                    "risk-ratio mode controls an expected bounded surrogate, not JointBSR",
                ]
            )
        return assumptions

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": (
                RAW_LOGIT_LOSS_SCHEMA_VERSION
                if self.representation == LOGIT_REPRESENTATION
                else LOSS_SCHEMA_VERSION
            ),
            "representation": self.representation,
            "threshold_grid_schema_version": self.threshold_grid_schema_version,
            "threshold_grid_sha256": self.threshold_grid_sha256_value,
            "threshold_grid_manifest_sha256": self.threshold_grid_manifest_sha256,
            "threshold_grid_detector_protocol": self.threshold_grid_detector_protocol,
            "threshold_grid_detector_checkpoint_sha256s": list(
                self.threshold_grid_detector_checkpoint_sha256s
            ),
            "threshold_grid_outer_detector_checkpoint_sha256": (
                self.threshold_grid_outer_detector_checkpoint_sha256
            ),
            "threshold_grid_episode_detector_checkpoint_sha256s": list(
                self.threshold_grid_episode_detector_checkpoint_sha256s
            ),
            "num_images": self.num_images,
            "num_thresholds": self.num_thresholds,
            "pixel_budget": {
                "value": self.pixel_budget,
                "unit": PIXEL_BUDGET_UNIT,
            },
            "component_budget": {
                "value": self.component_budget,
                "unit": COMPONENT_BUDGET_UNIT,
            },
            "loss_mode": self.loss_mode,
            "loss_definition": (
                "indicator(pixel_risk>pixel_budget or "
                "component_suffix_max_risk>component_budget)"
                if self.loss_mode == LOSS_MODE_BUDGET_VIOLATION
                else "max(clip(pixel_risk/pixel_budget,0,1),"
                "clip(component_suffix_max_risk/component_budget,0,1))"
            ),
            "joint_bsr_interpretation": self.loss_mode == LOSS_MODE_BUDGET_VIOLATION,
            "has_detection_utility_counts": self.tp_object_counts is not None,
            "assumptions": self.assumptions(),
        }


def build_calibration_losses(
    *,
    image_ids: Sequence[str] | np.ndarray,
    thresholds: np.ndarray,
    false_positive_pixels: np.ndarray,
    false_positive_components: np.ndarray,
    total_pixels: np.ndarray,
    pixel_budget: float,
    component_budget: float,
    loss_mode: str = LOSS_MODE_BUDGET_VIOLATION,
    tp_object_counts: np.ndarray | None = None,
    gt_object_counts: np.ndarray | None = None,
    monotonic_tolerance: float = 1e-9,
    representation: str = PROBABILITY_REPRESENTATION,
    threshold_grid_schema_version: str | None = None,
    recorded_threshold_grid_sha256: str | None = None,
    threshold_grid_manifest_sha256: str | None = None,
    threshold_grid_detector_protocol: str | None = None,
    threshold_grid_detector_checkpoint_sha256s: Sequence[str] | None = None,
    threshold_grid_outer_detector_checkpoint_sha256: str | None = None,
    threshold_grid_episode_detector_checkpoint_sha256s: Sequence[str] | None = None,
) -> CalibrationLosses:
    """Construct bounded joint loss curves from per-image count curves.

    ``false_positive_pixels`` and ``false_positive_components`` must have
    shape ``[N, T]``.  ``total_pixels`` is the original-resolution evaluated
    exposure per image, not a resized surrogate unless that protocol is
    explicitly intended by the caller.
    """

    if loss_mode not in (LOSS_MODE_BUDGET_VIOLATION, LOSS_MODE_RISK_RATIO):
        raise ValueError(
            f"loss_mode must be {LOSS_MODE_BUDGET_VIOLATION!r} or {LOSS_MODE_RISK_RATIO!r}"
        )
    if not np.isfinite(pixel_budget) or pixel_budget <= 0.0:
        raise ValueError("pixel_budget must be finite and positive")
    if not np.isfinite(component_budget) or component_budget <= 0.0:
        raise ValueError("component_budget must be finite and positive")
    ids = _normalise_image_ids(image_ids)
    fp_pixels = _curve_matrix(false_positive_pixels, "false_positive_pixels")
    fp_components = _curve_matrix(
        false_positive_components, "false_positive_components"
    )
    if fp_pixels.shape != fp_components.shape:
        raise ValueError("pixel and component count curves must have identical shapes")
    if fp_pixels.shape[0] != ids.size:
        raise ValueError("image_ids length does not match count curves")
    grid = _threshold_array(
        thresholds,
        fp_pixels.shape[1],
        representation=representation,
    )
    if representation == LOGIT_REPRESENTATION:
        semantic_grid_hash = logit_threshold_grid_sha256(
            np.asarray(thresholds)
        )
        if threshold_grid_schema_version != LOGIT_GRID_SCHEMA_VERSION:
            raise ValueError("Raw-logit calibration requires the v4 dense-grid schema")
        if recorded_threshold_grid_sha256 != semantic_grid_hash:
            raise ValueError("Raw-logit calibration semantic grid hash mismatch")
        if not _valid_sha256(threshold_grid_manifest_sha256):
            raise ValueError("Raw-logit calibration grid-manifest hash is invalid")
        if threshold_grid_detector_protocol != GRID_DETECTOR_PROTOCOL:
            raise ValueError("Raw-logit calibration grid detector protocol is invalid")
        raw_hashes, outer_detector_hash, raw_episode_hashes = _detector_role_hashes(
                threshold_grid_detector_checkpoint_sha256s,
                threshold_grid_outer_detector_checkpoint_sha256,
                threshold_grid_episode_detector_checkpoint_sha256s,
        )
        detector_hashes = tuple(raw_hashes)
        episode_detector_hashes = tuple(raw_episode_hashes)
    else:
        semantic_grid_hash = threshold_grid_sha256(
            np.asarray(thresholds, dtype=np.float32)
        )
        if recorded_threshold_grid_sha256 is not None and (
            recorded_threshold_grid_sha256 != semantic_grid_hash
        ):
            raise ValueError("Probability calibration semantic grid hash mismatch")
        detector_hashes = ()
        outer_detector_hash = None
        episode_detector_hashes = ()
    exposure_pixels = _pixel_exposure(total_pixels, ids.size)
    if np.any(fp_pixels > exposure_pixels[:, None] + monotonic_tolerance):
        raise ValueError("false_positive_pixels cannot exceed total_pixels")
    if np.any(np.diff(fp_pixels, axis=1) > monotonic_tolerance):
        raise ValueError(
            "false-positive pixel counts must be non-increasing as thresholds rise"
        )

    pixel_risk = fp_pixels / exposure_pixels[:, None]
    exposure_megapixels = exposure_pixels / 1_000_000.0
    component_risk_raw = fp_components / exposure_megapixels[:, None]
    component_risk_envelope = conservative_suffix_max(component_risk_raw)

    if loss_mode == LOSS_MODE_BUDGET_VIOLATION:
        pixel_loss = (pixel_risk > float(pixel_budget)).astype(np.float64)
        component_loss = (
            component_risk_envelope > float(component_budget)
        ).astype(np.float64)
    else:
        pixel_loss = np.clip(pixel_risk / float(pixel_budget), 0.0, 1.0)
        component_loss = np.clip(
            component_risk_envelope / float(component_budget), 0.0, 1.0
        )
    joint_loss = np.maximum(pixel_loss, component_loss)
    if np.any(np.diff(joint_loss, axis=1) > monotonic_tolerance):
        raise AssertionError("Internal error: joint calibration loss is not monotone")

    tp_counts: np.ndarray | None = None
    gt_counts: np.ndarray | None = None
    if (tp_object_counts is None) != (gt_object_counts is None):
        raise ValueError("tp_object_counts and gt_object_counts must be provided together")
    if tp_object_counts is not None:
        tp_counts = _curve_matrix(tp_object_counts, "tp_object_counts")
        if tp_counts.shape != fp_pixels.shape:
            raise ValueError("tp_object_counts must match the false-positive curve shape")
        gt_counts = np.asarray(gt_object_counts, dtype=np.float64).reshape(-1)
        if gt_counts.shape != (ids.size,):
            raise ValueError("gt_object_counts must have shape [N]")
        if not np.isfinite(gt_counts).all() or np.any(gt_counts < 0.0):
            raise ValueError("gt_object_counts must contain finite non-negative values")
        if np.any(tp_counts > gt_counts[:, None] + monotonic_tolerance):
            raise ValueError("tp_object_counts cannot exceed gt_object_counts")

    return CalibrationLosses(
        image_ids=ids,
        thresholds=grid,
        false_positive_pixels=fp_pixels,
        false_positive_components=fp_components,
        total_pixels=exposure_pixels,
        pixel_risk=pixel_risk,
        component_risk_raw=component_risk_raw,
        component_risk_envelope=component_risk_envelope,
        pixel_loss=pixel_loss,
        component_loss=component_loss,
        joint_loss=joint_loss,
        pixel_budget=float(pixel_budget),
        component_budget=float(component_budget),
        loss_mode=loss_mode,
        tp_object_counts=tp_counts,
        gt_object_counts=gt_counts,
        representation=representation,
        threshold_grid_schema_version=threshold_grid_schema_version,
        threshold_grid_sha256_value=semantic_grid_hash,
        threshold_grid_manifest_sha256=threshold_grid_manifest_sha256,
        threshold_grid_detector_protocol=threshold_grid_detector_protocol,
        threshold_grid_detector_checkpoint_sha256s=detector_hashes,
        threshold_grid_outer_detector_checkpoint_sha256=outer_detector_hash,
        threshold_grid_episode_detector_checkpoint_sha256s=(
            episode_detector_hashes
        ),
    )


def _first_key(archive: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    for name in names:
        if name in archive:
            return archive[name]
    raise ValueError(f"Missing required array; expected one of: {', '.join(names)}")


def load_count_curve_archive(
    path: str | Path,
    *,
    require_integrity: bool = False,
) -> dict[str, np.ndarray]:
    """Load the portable count-curve schema and verify any embedded digest."""

    source = Path(path)
    verify_count_curve_archive_integrity(
        source, require_integrity=require_integrity
    )
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    result = {
        "image_ids": _first_key(arrays, ("image_ids", "ids")),
        "thresholds": _first_key(arrays, ("thresholds", "threshold_grid")),
        "false_positive_pixels": _first_key(
            arrays, ("false_positive_pixels", "fp_pixels", "num_fp_pixels")
        ),
        "false_positive_components": _first_key(
            arrays,
            ("false_positive_components", "fp_components", "num_fp_components"),
        ),
        "total_pixels": _first_key(
            arrays, ("total_pixels", "evaluated_pixels", "pixel_exposure")
        ),
        "representation": (
            str(np.asarray(arrays["representation"]).item())
            if "representation" in arrays
            else PROBABILITY_REPRESENTATION
        ),
    }
    optional_scalar_contract = {
        "threshold_grid_schema_version": "threshold_grid_schema_version",
        "threshold_grid_sha256": "recorded_threshold_grid_sha256",
        "threshold_grid_manifest_sha256": "threshold_grid_manifest_sha256",
        "threshold_grid_detector_protocol": "threshold_grid_detector_protocol",
    }
    for archive_key, result_key in optional_scalar_contract.items():
        if archive_key in arrays:
            raw = np.asarray(arrays[archive_key])
            if raw.ndim != 0:
                raise ValueError(f"{archive_key} must be a scalar string")
            result[result_key] = str(raw.item())
    if "threshold_grid_detector_checkpoint_sha256s" in arrays:
        raw_hashes = np.asarray(
            arrays["threshold_grid_detector_checkpoint_sha256s"]
        )
        if raw_hashes.ndim != 1:
            raise ValueError(
                "threshold_grid_detector_checkpoint_sha256s must be one-dimensional"
            )
        result["threshold_grid_detector_checkpoint_sha256s"] = [
            str(value) for value in raw_hashes.tolist()
        ]
    if "threshold_grid_outer_detector_checkpoint_sha256" in arrays:
        raw_outer = np.asarray(
            arrays["threshold_grid_outer_detector_checkpoint_sha256"]
        )
        if raw_outer.ndim != 0:
            raise ValueError(
                "threshold_grid_outer_detector_checkpoint_sha256 must be scalar"
            )
        result["threshold_grid_outer_detector_checkpoint_sha256"] = str(
            raw_outer.item()
        )
    if "threshold_grid_episode_detector_checkpoint_sha256s" in arrays:
        raw_episode_hashes = np.asarray(
            arrays["threshold_grid_episode_detector_checkpoint_sha256s"]
        )
        if raw_episode_hashes.ndim != 1:
            raise ValueError(
                "threshold_grid_episode_detector_checkpoint_sha256s must be one-dimensional"
            )
        result["threshold_grid_episode_detector_checkpoint_sha256s"] = [
            str(value) for value in raw_episode_hashes.tolist()
        ]
    tp_key = next(
        (name for name in ("tp_object_counts", "true_positive_objects") if name in arrays),
        None,
    )
    gt_key = next(
        (name for name in ("gt_object_counts", "ground_truth_object_counts") if name in arrays),
        None,
    )
    if (tp_key is None) != (gt_key is None):
        raise ValueError(
            "Count archive must contain both tp_object_counts and gt_object_counts or neither"
        )
    if tp_key is not None and gt_key is not None:
        result["tp_object_counts"] = arrays[tp_key]
        result["gt_object_counts"] = arrays[gt_key]
    return result


def load_count_curve_provenance(
    path: str | Path,
    *,
    require_integrity: bool = False,
) -> dict[str, Any]:
    """Load optional JSON provenance without enabling pickle deserialisation."""

    verify_count_curve_archive_integrity(
        path, require_integrity=require_integrity
    )
    with np.load(Path(path), allow_pickle=False) as archive:
        if "provenance_json" not in archive:
            return {}
        raw = np.asarray(archive["provenance_json"])
        if raw.ndim != 0:
            raise ValueError("provenance_json must be a scalar JSON string")
        value = json.loads(str(raw.item()))
    if not isinstance(value, dict):
        raise ValueError("provenance_json must decode to an object")
    return value


def _score_record_ids(
    score_dir: str | Path,
    expected_count: int,
    *,
    require_integrity: bool,
) -> np.ndarray:
    root = Path(score_dir).expanduser()
    manifest, paths, _ = verify_score_map_directory(
        root, require_integrity=require_integrity
    )
    if manifest is not None:
        records = manifest.get("records")
        assert isinstance(records, list)  # validated by verify_score_map_directory
        ids = [str(record["image_id"]) for record in records]
    else:
        ids = []
        for record_path in paths:
            with np.load(record_path, allow_pickle=False) as payload:
                image_id = (
                    payload["image_id"].item()
                    if "image_id" in payload
                    else record_path.stem
                )
            ids.append(str(image_id))
    if len(ids) != expected_count:
        raise ValueError(
            f"Loaded {expected_count} score maps but resolved {len(ids)} image IDs"
        )
    return np.asarray(ids)


def build_count_curves_from_score_maps(
    score_dir: str | Path,
    thresholds: np.ndarray,
    *,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
    require_integrity: bool = True,
    representation: str = PROBABILITY_REPRESENTATION,
) -> dict[str, np.ndarray]:
    """Generate per-image FP count curves from exported NPZ score maps.

    Record order follows ``manifest.json`` when present, matching the shared
    evaluation loader.  Counts use the same one-to-one component matcher and
    pixel definition as the common evaluation core.
    """

    from evaluation.component_matching import match_components
    from evaluation.threshold_sweep import load_score_map_directory

    if representation == LOGIT_REPRESENTATION:
        if not require_integrity:
            raise ValueError("Raw-logit count curves require integrity verification")
        manifest, paths, integrity = verify_score_map_directory(
            score_dir,
            require_integrity=True,
            require_masks=True,
        )
        if manifest is None:
            raise ValueError("Raw-logit count curves require manifest.json")
        if manifest.get("schema_version") != SCORE_MANIFEST_SCHEMA_VERSION:
            raise ValueError("Raw-logit count curves require score manifest v3")
        if manifest.get("record_integrity_schema") != SCORE_RECORD_INTEGRITY_SCHEMA:
            raise ValueError("Raw-logit count curves require complete record hashes")
        if integrity.get("verified") is not True:
            raise ValueError("Raw-logit count curves require verified score integrity")
        if manifest.get("labels_loaded") is not True:
            raise ValueError("Raw-logit calibration count curves require labels")
        expected_raw = {
            "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
            "probability_dtype": PROBABILITY_DTYPE,
            "logit_dtype": RAW_LOGIT_DTYPE,
            "probability_transform": "sigmoid",
            "probability_clipping": "none",
            "inference_autocast_enabled": False,
        }
        for field, expected in expected_raw.items():
            if manifest.get(field) != expected:
                raise ValueError(
                    f"Raw-logit count curves require manifest {field}={expected!r}"
                )
        scores: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for index, path in enumerate(paths):
            with np.load(path, allow_pickle=False) as payload:
                if "logit" not in payload or "mask" not in payload:
                    raise ValueError(
                        f"Raw-logit count record {index} lacks logit/mask"
                    )
                logit = np.asarray(payload["logit"])
                if logit.dtype != np.float32:
                    raise ValueError(
                        f"Raw-logit count record {index} is not float32"
                    )
                scores.append(logit)
                masks.append(np.asarray(payload["mask"]))
        grid = validate_logit_threshold_grid(np.asarray(thresholds))
        image_ids = np.asarray(
            [str(record["image_id"]) for record in manifest["records"]]
        )
    elif representation == PROBABILITY_REPRESENTATION:
        scores, masks = load_score_map_directory(
            score_dir, require_integrity=require_integrity
        )
        grid = np.asarray(thresholds, dtype=np.float64).reshape(-1)
        if grid.size == 0 or not np.isfinite(grid).all():
            raise ValueError("thresholds must be a finite non-empty array")
        if np.any((grid < 0.0) | (grid > 1.0)) or np.any(np.diff(grid) <= 0.0):
            raise ValueError("thresholds must be strictly increasing in [0, 1]")
        image_ids = _score_record_ids(
            score_dir,
            len(scores),
            require_integrity=require_integrity,
        )
    else:
        raise ValueError(f"Unsupported score representation: {representation!r}")
    if not scores:
        raise ValueError("score_dir did not yield any score maps")
    num_images = len(scores)
    num_thresholds = grid.size
    fp_pixels = np.zeros((num_images, num_thresholds), dtype=np.int64)
    fp_components = np.zeros((num_images, num_thresholds), dtype=np.int64)
    tp_objects = np.zeros((num_images, num_thresholds), dtype=np.int64)
    gt_objects = np.zeros(num_images, dtype=np.int64)
    total_pixels = np.zeros(num_images, dtype=np.int64)
    for image_index, (score_value, mask) in enumerate(zip(scores, masks)):
        score = np.asarray(score_value)
        target = np.asarray(mask)
        if score.ndim == 3 and score.shape[0] == 1:
            score = score[0]
        if target.ndim == 3 and target.shape[0] == 1:
            target = target[0]
        if score.ndim != 2 or target.ndim != 2 or score.shape != target.shape:
            raise ValueError(
                f"score-map record {image_index} must contain same-shaped 2-D prob/mask"
            )
        if not np.isfinite(score).all():
            raise ValueError(f"score-map record {image_index} has non-finite scores")
        if representation == PROBABILITY_REPRESENTATION and np.any(
            (score < 0.0) | (score > 1.0)
        ):
            raise ValueError(f"score-map record {image_index} has invalid probabilities")
        target = target > 0
        total_pixels[image_index] = score.size
        for threshold_index, threshold in enumerate(grid):
            match = match_components(
                score >= threshold,
                target,
                rule=matching_rule,
                centroid_distance=centroid_distance,
                connectivity=connectivity,
                min_component_area=min_component_area,
            )
            fp_pixels[image_index, threshold_index] = match.num_fp_pixels
            fp_components[image_index, threshold_index] = match.num_fp_components
            tp_objects[image_index, threshold_index] = match.num_tp_objects
            if threshold_index == 0:
                gt_objects[image_index] = match.num_gt
            elif gt_objects[image_index] != match.num_gt:
                raise RuntimeError("Ground-truth object count changed across thresholds")
    return {
        "image_ids": image_ids,
        "thresholds": grid,
        "false_positive_pixels": fp_pixels,
        "false_positive_components": fp_components,
        "total_pixels": total_pixels,
        "tp_object_counts": tp_objects,
        "gt_object_counts": gt_objects,
    }


def save_calibration_losses(
    path: str | Path,
    losses: CalibrationLosses,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "image_ids": losses.image_ids.astype(str),
        "thresholds": losses.thresholds,
        "false_positive_pixels": losses.false_positive_pixels,
        "false_positive_components": losses.false_positive_components,
        "total_pixels": losses.total_pixels,
        "pixel_risk": losses.pixel_risk,
        "component_risk_raw": losses.component_risk_raw,
        "component_risk_envelope": losses.component_risk_envelope,
        "pixel_loss": losses.pixel_loss,
        "component_loss": losses.component_loss,
        "joint_loss": losses.joint_loss,
        "metadata_json": np.asarray(json.dumps(losses.metadata(), sort_keys=True)),
        "representation": np.asarray(losses.representation),
        "threshold_grid_sha256": np.asarray(
            losses.threshold_grid_sha256_value
        ),
        "provenance_json": np.asarray(
            json.dumps(dict(provenance or {}), sort_keys=True)
        ),
        "count_archive_integrity_schema": np.asarray(
            COUNT_ARCHIVE_INTEGRITY_SCHEMA
        ),
    }
    if losses.threshold_grid_schema_version is not None:
        arrays["threshold_grid_schema_version"] = np.asarray(
            losses.threshold_grid_schema_version
        )
    if losses.threshold_grid_manifest_sha256 is not None:
        arrays["threshold_grid_manifest_sha256"] = np.asarray(
            losses.threshold_grid_manifest_sha256
        )
    if losses.threshold_grid_detector_protocol is not None:
        arrays["threshold_grid_detector_protocol"] = np.asarray(
            losses.threshold_grid_detector_protocol
        )
    if losses.threshold_grid_detector_checkpoint_sha256s:
        arrays["threshold_grid_detector_checkpoint_sha256s"] = np.asarray(
            losses.threshold_grid_detector_checkpoint_sha256s
        )
    if losses.threshold_grid_outer_detector_checkpoint_sha256 is not None:
        arrays["threshold_grid_outer_detector_checkpoint_sha256"] = np.asarray(
            losses.threshold_grid_outer_detector_checkpoint_sha256
        )
    if losses.threshold_grid_episode_detector_checkpoint_sha256s:
        arrays["threshold_grid_episode_detector_checkpoint_sha256s"] = np.asarray(
            losses.threshold_grid_episode_detector_checkpoint_sha256s
        )
    if losses.tp_object_counts is not None and losses.gt_object_counts is not None:
        arrays["tp_object_counts"] = losses.tp_object_counts
        arrays["gt_object_counts"] = losses.gt_object_counts
    arrays["count_archive_payload_sha256"] = np.asarray(
        count_archive_payload_sha256(arrays)
    )
    np.savez_compressed(output, **arrays)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--count-curves")
    source.add_argument("--score-dir")
    parser.add_argument(
        "--threshold-grid",
        help="Required with --score-dir; .npy predictor/deployment threshold grid",
    )
    parser.add_argument(
        "--representation",
        choices=(PROBABILITY_REPRESENTATION, LOGIT_REPRESENTATION),
        default=PROBABILITY_REPRESENTATION,
        help="Score domain used to build counts; v4 CRC requires raw_logit_float32",
    )
    parser.add_argument(
        "--threshold-grid-manifest",
        help="Required for raw-logit counts; threshold_grid.json or its directory",
    )
    parser.add_argument("--matching-rule", choices=("overlap", "centroid"), default="overlap")
    parser.add_argument("--centroid-distance", type=float, default=3.0)
    parser.add_argument("--connectivity", type=int, default=2)
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument("--pixel-budget", required=True, type=float)
    parser.add_argument("--component-budget", required=True, type=float)
    parser.add_argument(
        "--loss-mode",
        choices=(LOSS_MODE_BUDGET_VIOLATION, LOSS_MODE_RISK_RATIO),
        default=LOSS_MODE_BUDGET_VIOLATION,
        help="Default binary mode maps the CRC bound directly to JointBSR",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    provenance: dict[str, Any]
    if args.score_dir:
        if not args.threshold_grid:
            raise ValueError("--threshold-grid is required with --score-dir")
        grid_path = Path(args.threshold_grid)
        thresholds = np.load(grid_path, allow_pickle=False)
        grid_manifest_sha256: str | None = None
        grid_manifest_path: str | None = None
        grid_detector_protocol: str | None = None
        grid_detector_hashes: list[str] = []
        grid_outer_detector_hash: str | None = None
        grid_episode_detector_hashes: list[str] = []
        if args.representation == LOGIT_REPRESENTATION:
            if not args.threshold_grid_manifest:
                raise ValueError(
                    "Raw-logit count curves require --threshold-grid-manifest"
                )
            grid_artifact = load_logit_grid_artifact(
                args.threshold_grid_manifest
            )
            if not np.array_equal(thresholds, grid_artifact.thresholds):
                raise ValueError(
                    "--threshold-grid does not match its raw-logit manifest"
                )
            thresholds = grid_artifact.thresholds
            grid_manifest_sha256 = file_sha256(grid_artifact.manifest_path)
            grid_manifest_path = str(grid_artifact.manifest_path.resolve())
            grid_detector_protocol = str(
                grid_artifact.manifest["grid_detector_protocol"]
            )
            grid_detector_hashes = list(
                grid_artifact.manifest["detector_checkpoint_sha256s"]
            )
            grid_outer_detector_hash = str(
                grid_artifact.manifest["outer_detector_checkpoint_sha256"]
            )
            grid_episode_detector_hashes = list(
                grid_artifact.manifest[
                    "episode_detector_checkpoint_sha256s"
                ]
            )
        elif args.threshold_grid_manifest:
            raise ValueError(
                "--threshold-grid-manifest is only valid for raw-logit counts"
            )
        data = build_count_curves_from_score_maps(
            args.score_dir,
            thresholds,
            matching_rule=args.matching_rule,
            centroid_distance=args.centroid_distance,
            connectivity=args.connectivity,
            min_component_area=args.min_component_area,
            representation=args.representation,
        )
        data["representation"] = args.representation
        if args.representation == LOGIT_REPRESENTATION:
            data.update(
                {
                    "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
                    "recorded_threshold_grid_sha256": (
                        logit_threshold_grid_sha256(thresholds)
                    ),
                    "threshold_grid_manifest_sha256": grid_manifest_sha256,
                    "threshold_grid_detector_protocol": grid_detector_protocol,
                    "threshold_grid_detector_checkpoint_sha256s": (
                        grid_detector_hashes
                    ),
                    "threshold_grid_outer_detector_checkpoint_sha256": (
                        grid_outer_detector_hash
                    ),
                    "threshold_grid_episode_detector_checkpoint_sha256s": (
                        grid_episode_detector_hashes
                    ),
                }
            )
        manifest_path = Path(args.score_dir) / "manifest.json"
        score_manifest, _, score_integrity = verify_score_map_directory(
            args.score_dir, require_integrity=True
        )
        assert score_manifest is not None
        protocol, fingerprint = score_map_protocol(
            args.score_dir,
            thresholds,
            matching_rule=args.matching_rule,
            centroid_distance=args.centroid_distance,
            connectivity=args.connectivity,
            min_component_area=args.min_component_area,
            representation=args.representation,
            threshold_grid_detector_protocol=grid_detector_protocol,
            threshold_grid_detector_checkpoint_sha256s=grid_detector_hashes,
            threshold_grid_outer_detector_checkpoint_sha256=(
                grid_outer_detector_hash
            ),
            threshold_grid_episode_detector_checkpoint_sha256s=(
                grid_episode_detector_hashes
            ),
        )
        provenance = {
            "source_type": "exported_score_map_directory",
            "score_dir": str(Path(args.score_dir).resolve()),
            "manifest_sha256": (
                hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                if manifest_path.is_file()
                else None
            ),
            "score_manifest_schema_version": score_manifest.get("schema_version"),
            "score_records_sha256": score_integrity["records_sha256"],
            "score_ordered_image_ids_sha256": score_integrity[
                "ordered_image_ids_sha256"
            ],
            "score_num_records": score_integrity["num_records"],
            "split_file": score_manifest.get("split_file"),
            "split_file_sha256": score_manifest.get("split_file_sha256"),
            "split_ordered_ids_sha256": score_manifest.get(
                "split_ordered_ids_sha256"
            ),
            "threshold_grid": str(grid_path.resolve()),
            "threshold_grid_file_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
            "representation": args.representation,
            "threshold_grid_schema_version": (
                LOGIT_GRID_SCHEMA_VERSION
                if args.representation == LOGIT_REPRESENTATION
                else None
            ),
            "threshold_grid_sha256": (
                logit_threshold_grid_sha256(thresholds)
                if args.representation == LOGIT_REPRESENTATION
                else threshold_grid_sha256(thresholds)
            ),
            "threshold_grid_manifest_sha256": grid_manifest_sha256,
            "threshold_grid_manifest": grid_manifest_path,
            "threshold_grid_detector_protocol": grid_detector_protocol,
            "threshold_grid_detector_checkpoint_sha256s": grid_detector_hashes,
            "threshold_grid_outer_detector_checkpoint_sha256": (
                grid_outer_detector_hash
            ),
            "threshold_grid_episode_detector_checkpoint_sha256s": (
                grid_episode_detector_hashes
            ),
            "matching_rule": args.matching_rule,
            "centroid_distance": args.centroid_distance,
            "connectivity": args.connectivity,
            "min_component_area": args.min_component_area,
            "protocol": protocol,
            "protocol_fingerprint": fingerprint,
        }
    else:
        if args.threshold_grid:
            raise ValueError("--threshold-grid is only used with --score-dir")
        if args.threshold_grid_manifest:
            raise ValueError(
                "--threshold-grid-manifest is only used with --score-dir"
            )
        data = load_count_curve_archive(args.count_curves)
        source_path = Path(args.count_curves)
        upstream_provenance = load_count_curve_provenance(source_path)
        provenance = {
            "source_type": "precomputed_count_curve_archive",
            "count_curves": str(source_path.resolve()),
            "count_curves_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "upstream_provenance": upstream_provenance,
        }
        for key in ("protocol", "protocol_fingerprint", "threshold_grid_sha256"):
            if key in upstream_provenance:
                provenance[key] = upstream_provenance[key]
    losses = build_calibration_losses(
        **data,
        pixel_budget=args.pixel_budget,
        component_budget=args.component_budget,
        loss_mode=args.loss_mode,
    )
    path = save_calibration_losses(args.output, losses, provenance=provenance)
    print(
        json.dumps(
            {"output": str(path), **losses.metadata(), "provenance": provenance},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
