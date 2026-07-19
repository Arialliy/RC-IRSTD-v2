"""Validated NPZ dataset for risk-curve training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from torch.utils.data import Dataset

from .domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    STATISTICS_SCHEMA_VERSION,
    feature_schema_sha256,
    statistics_names_sha256,
    validate_statistics_names,
)
from .representation import (
    GRID_DETECTOR_PROTOCOL,
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_REPRESENTATION,
    PROBABILITY_REPRESENTATION,
    SUPPORTED_REPRESENTATIONS,
    logit_threshold_grid_sha256,
    validate_logit_threshold_grid,
)
from .threshold_grid import (
    threshold_grid_sha256,
    threshold_grid_version,
    validate_threshold_grid,
)


REQUIRED_KEYS = (
    "statistics",
    "statistics_names",
    "statistics_schema_version",
    "pixel_log_risk",
    "pd_curve",
    "thresholds",
)

COMPONENT_RISK_SCHEMA_VERSION = "rc-v2-component-risk-v1-raw-upper"
LOGIT_EPISODE_SCHEMA_VERSION = "rc-v4-curve-episode-v1-raw-logit-causal"
COUNT_ALL_ADAPTATION_SCHEMA_VERSION = "rc-v4-count-all-adaptation-curves-v1"
COUNT_ALL_ADAPTATION_ARCHIVE_FIELDS = (
    "adaptation_predicted_pixel_counts",
    "adaptation_predicted_component_counts_raw",
    "adaptation_predicted_component_counts_upper",
    "adaptation_total_pixels",
    "count_all_adaptation_schema_version",
)


def _require_sha256_scalar(value: np.ndarray, field: str) -> str:
    digest = _scalar_string(value, field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _require_distinct_sha256_sequence(
    value: np.ndarray,
    field: str,
) -> tuple[str, ...]:
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


def _detector_role_hashes(
    data: dict[str, np.ndarray],
    *,
    field_prefix: str = "",
) -> tuple[tuple[str, ...], str, tuple[str, ...]]:
    all_hashes = _require_distinct_sha256_sequence(
        data["threshold_grid_detector_checkpoint_sha256s"],
        f"{field_prefix}threshold_grid_detector_checkpoint_sha256s",
    )
    outer_hash = _require_sha256_scalar(
        data["threshold_grid_outer_detector_checkpoint_sha256"],
        f"{field_prefix}threshold_grid_outer_detector_checkpoint_sha256",
    )
    episode_hashes = _require_distinct_sha256_sequence(
        data["threshold_grid_episode_detector_checkpoint_sha256s"],
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


def _representation_contract(
    data: dict[str, np.ndarray],
    *,
    source: Path,
) -> tuple[str, str, str]:
    """Validate and normalise an archive's threshold representation contract.

    Historical v2 archives did not persist representation metadata.  They are
    unambiguously interpreted as sigmoid-probability archives, while every v4
    raw-logit archive is required to carry the complete semantic contract.
    """

    schema_version = _scalar_string(
        data["statistics_schema_version"], "statistics_schema_version"
    )
    representation_field = data.get("representation")
    representation = (
        _scalar_string(representation_field, "representation")
        if representation_field is not None
        else PROBABILITY_REPRESENTATION
    )
    if representation not in SUPPORTED_REPRESENTATIONS:
        raise ValueError(
            f"Unsupported threshold representation {representation!r} in {source}"
        )

    raw_logit = representation == LOGIT_REPRESENTATION
    expected_statistics_schema = (
        LOGIT_STATISTICS_SCHEMA_VERSION if raw_logit else STATISTICS_SCHEMA_VERSION
    )
    if schema_version != expected_statistics_schema:
        raise ValueError(
            "Threshold representation/statistics schema mismatch: "
            f"{representation!r} requires {expected_statistics_schema!r}, got "
            f"{schema_version!r}"
        )
    if representation_field is None and raw_logit:
        # Kept as an explicit invariant even though the default above currently
        # makes it unreachable; it documents that raw logits are never inferred.
        raise ValueError("Raw-logit archives must explicitly declare representation")

    raw_thresholds = np.asarray(data["thresholds"])
    if raw_logit:
        required = {
            "representation",
            "threshold_grid_schema_version",
            "threshold_grid_sha256",
            "threshold_grid_manifest_sha256",
            "threshold_grid_detector_protocol",
            "threshold_grid_detector_checkpoint_sha256s",
            "threshold_grid_outer_detector_checkpoint_sha256",
            "threshold_grid_episode_detector_checkpoint_sha256s",
            "feature_schema_sha256",
            "statistics_names_sha256",
            "episode_schema_version",
        }
        missing = sorted(required.difference(data))
        if missing:
            raise ValueError(
                f"v4 raw-logit archive {source} is missing contract fields: "
                + ", ".join(missing)
            )
        grid_schema = _scalar_string(
            data["threshold_grid_schema_version"],
            "threshold_grid_schema_version",
        )
        if grid_schema != LOGIT_GRID_SCHEMA_VERSION:
            raise ValueError(
                f"Raw-logit threshold grid schema must be "
                f"{LOGIT_GRID_SCHEMA_VERSION!r}, got {grid_schema!r}"
            )
        thresholds = validate_logit_threshold_grid(raw_thresholds)
        semantic_grid_hash = logit_threshold_grid_sha256(
            thresholds,
            schema_version=grid_schema,
            representation=representation,
        )
        recorded_grid_hash = _scalar_string(
            data["threshold_grid_sha256"], "threshold_grid_sha256"
        )
        if recorded_grid_hash != semantic_grid_hash:
            raise ValueError(
                "Raw-logit threshold_grid_sha256 does not match the semantic grid"
            )
        _require_sha256_scalar(
            data["threshold_grid_manifest_sha256"],
            "threshold_grid_manifest_sha256",
        )
        detector_protocol = _scalar_string(
            data["threshold_grid_detector_protocol"],
            "threshold_grid_detector_protocol",
        )
        if detector_protocol != GRID_DETECTOR_PROTOCOL:
            raise ValueError(
                "Raw-logit threshold_grid_detector_protocol must be "
                f"{GRID_DETECTOR_PROTOCOL!r}, got {detector_protocol!r}"
            )
        detector_hashes, outer_detector_hash, episode_detector_hashes = (
            _detector_role_hashes(data)
        )
        data["threshold_grid_detector_checkpoint_sha256s"] = np.asarray(
            detector_hashes,
            dtype=str,
        )
        data["threshold_grid_outer_detector_checkpoint_sha256"] = np.asarray(
            outer_detector_hash
        )
        data["threshold_grid_episode_detector_checkpoint_sha256s"] = np.asarray(
            episode_detector_hashes,
            dtype=str,
        )
        episode_schema = _scalar_string(
            data["episode_schema_version"], "episode_schema_version"
        )
        if episode_schema != LOGIT_EPISODE_SCHEMA_VERSION:
            raise ValueError(
                "Raw-logit archive episode schema must be "
                f"{LOGIT_EPISODE_SCHEMA_VERSION!r}, got {episode_schema!r}"
            )
    else:
        thresholds = validate_threshold_grid(raw_thresholds)
        grid_schema_field = data.get("threshold_grid_schema_version")
        expected_grid_schema = threshold_grid_version(thresholds)
        grid_schema = (
            _scalar_string(grid_schema_field, "threshold_grid_schema_version")
            if grid_schema_field is not None
            else expected_grid_schema
        )
        if grid_schema != expected_grid_schema:
            raise ValueError(
                f"Probability threshold grid schema must be {expected_grid_schema!r}, "
                f"got {grid_schema!r}"
            )
        semantic_grid_hash = threshold_grid_sha256(thresholds)
        if "threshold_grid_sha256" in data:
            recorded_grid_hash = _scalar_string(
                data["threshold_grid_sha256"], "threshold_grid_sha256"
            )
            if recorded_grid_hash != semantic_grid_hash:
                raise ValueError(
                    "Probability threshold_grid_sha256 does not match thresholds"
                )

    data["thresholds"] = thresholds
    data["representation"] = np.asarray(representation)
    data["threshold_grid_schema_version"] = np.asarray(grid_schema)
    data["threshold_grid_sha256"] = np.asarray(semantic_grid_hash)
    return representation, grid_schema, semantic_grid_hash


def _component_upper_envelope(curves: np.ndarray) -> np.ndarray:
    values = np.asarray(curves)
    if values.ndim != 2:
        raise ValueError("component risk curves must have shape [N, T]")
    return np.maximum.accumulate(values[:, ::-1], axis=1)[:, ::-1]


def _scalar_string(value: np.ndarray, field: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{field} must be a scalar string")
    return str(array.item())


def _validated_nonnegative_integer_array(
    value: np.ndarray,
    *,
    field: str,
    shape: tuple[int, ...],
    minimum: int = 0,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape:
        raise ValueError(f"{field} must have shape {shape}")
    if raw.dtype.kind not in "iu":
        if not np.isfinite(raw).all() or not np.all(np.equal(raw, np.floor(raw))):
            raise ValueError(f"{field} must contain finite integer counts")
    counts = raw.astype(np.int64)
    if np.any(counts < minimum):
        raise ValueError(f"{field} contains a value below {minimum}")
    return counts


def validate_count_all_adaptation_contract(
    data: dict[str, np.ndarray],
    *,
    required: bool = False,
) -> dict[str, object]:
    """Validate the optional v4 label-free A-window Count-all sub-contract."""

    present = set(COUNT_ALL_ADAPTATION_ARCHIVE_FIELDS).intersection(data)
    if not present:
        if required:
            raise ValueError(
                "Raw-logit Count-all requires complete adaptation-window count "
                "curves; this archive predates that sub-contract"
            )
        return {
            "present": False,
            "verified": False,
            "schema_version": None,
            "missing_fields": list(COUNT_ALL_ADAPTATION_ARCHIVE_FIELDS),
        }
    missing = sorted(set(COUNT_ALL_ADAPTATION_ARCHIVE_FIELDS).difference(data))
    if missing:
        raise ValueError(
            "Incomplete Count-all adaptation-window contract; missing: "
            + ", ".join(missing)
        )
    schema = _scalar_string(
        data["count_all_adaptation_schema_version"],
        "count_all_adaptation_schema_version",
    )
    if schema != COUNT_ALL_ADAPTATION_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported Count-all adaptation schema; expected "
            f"{COUNT_ALL_ADAPTATION_SCHEMA_VERSION!r}, got {schema!r}"
        )
    representation = _scalar_string(data["representation"], "representation")
    if representation != LOGIT_REPRESENTATION:
        raise ValueError("Count-all adaptation curves require raw_logit_float32")
    rows = int(np.asarray(data["statistics"]).shape[0])
    thresholds = validate_logit_threshold_grid(
        np.asarray(data["thresholds"], dtype=np.float32)
    )
    shape = (rows, int(thresholds.size))
    pixels = _validated_nonnegative_integer_array(
        data["adaptation_predicted_pixel_counts"],
        field="adaptation_predicted_pixel_counts",
        shape=shape,
    )
    components_raw = _validated_nonnegative_integer_array(
        data["adaptation_predicted_component_counts_raw"],
        field="adaptation_predicted_component_counts_raw",
        shape=shape,
    )
    components_upper = _validated_nonnegative_integer_array(
        data["adaptation_predicted_component_counts_upper"],
        field="adaptation_predicted_component_counts_upper",
        shape=shape,
    )
    total_pixels = _validated_nonnegative_integer_array(
        data["adaptation_total_pixels"],
        field="adaptation_total_pixels",
        shape=(rows,),
        minimum=1,
    )
    if np.any(pixels > total_pixels[:, None]):
        raise ValueError(
            "adaptation_predicted_pixel_counts exceed adaptation_total_pixels"
        )
    if np.any(components_raw > pixels):
        raise ValueError(
            "adaptation predicted component counts exceed retained pixel counts"
        )
    if np.any(np.diff(pixels, axis=1) > 0):
        raise ValueError("adaptation predicted-pixel count curves are not monotone")
    expected_upper = np.maximum.accumulate(
        components_raw[:, ::-1], axis=1
    )[:, ::-1]
    if not np.array_equal(components_upper, expected_upper):
        raise ValueError(
            "adaptation_predicted_component_counts_upper must equal "
            "suffix_max(raw counts)"
        )
    if np.any(np.diff(components_upper, axis=1) > 0):
        raise ValueError("adaptation component upper curves are not monotone")

    if "provenance_json" not in data:
        raise ValueError("Count-all adaptation contract requires provenance_json")
    try:
        provenance = json.loads(
            _scalar_string(data["provenance_json"], "provenance_json")
        )
    except json.JSONDecodeError as error:
        raise ValueError("Count-all adaptation provenance_json is invalid") from error
    if not isinstance(provenance, dict):
        raise ValueError("Count-all adaptation provenance must be an object")
    expected_provenance = {
        "representation": LOGIT_REPRESENTATION,
        "count_all_adaptation_schema_version": (
            COUNT_ALL_ADAPTATION_SCHEMA_VERSION
        ),
        "count_all_adaptation_sample_role": "adaptation_window_A_label_free",
        "count_all_adaptation_masks_read": False,
        "count_all_adaptation_prediction_rule": (
            "prediction = (raw_logits >= threshold)"
        ),
        "count_all_adaptation_pixel_count_semantics": (
            "pixels retained after connectivity/min_component_area filtering"
        ),
        "count_all_adaptation_component_count_semantics": (
            "connected components retained after min_component_area filtering"
        ),
        "count_all_adaptation_component_envelope": (
            "suffix_max_of_window_aggregate_raw_component_counts"
        ),
    }
    for field, expected in expected_provenance.items():
        if provenance.get(field) != expected:
            raise ValueError(
                f"Count-all adaptation provenance {field} must equal {expected!r}"
            )
    connectivity = provenance.get("connectivity")
    min_component_area = provenance.get("min_component_area")
    if connectivity not in {1, 2, 4, 8}:
        raise ValueError("Count-all adaptation provenance connectivity is invalid")
    if (
        isinstance(min_component_area, bool)
        or not isinstance(min_component_area, int)
        or min_component_area <= 0
    ):
        raise ValueError(
            "Count-all adaptation provenance min_component_area is invalid"
        )
    return {
        "present": True,
        "verified": True,
        "schema_version": schema,
        "representation": representation,
        "threshold_grid_sha256": _scalar_string(
            data["threshold_grid_sha256"], "threshold_grid_sha256"
        ),
        "num_episodes": rows,
        "num_thresholds": int(thresholds.size),
        "connectivity": int(connectivity),
        "min_component_area": int(min_component_area),
        "adaptation_masks_read": False,
        "prediction_rule": expected_provenance[
            "count_all_adaptation_prediction_rule"
        ],
    }


def load_curve_archive(path: str | Path | BinaryIO) -> dict[str, np.ndarray]:
    if hasattr(path, "read"):
        archive_source = path
        source = Path(str(getattr(path, "name", "<immutable-byte-snapshot>")))
        if hasattr(archive_source, "seek"):
            archive_source.seek(0)
    else:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Curve archive does not exist: {source}")
        archive_source = source
    with np.load(archive_source, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_KEYS if key not in archive]
        if missing:
            raise ValueError(f"{source} is missing keys: {', '.join(missing)}")
        data = {key: archive[key] for key in archive.files}
    if "component_log_risk" not in data:
        if "comp_log_risk" in data:
            data["component_log_risk"] = data["comp_log_risk"]
        elif "component_log_risk_upper" in data:
            # Be liberal when loading an otherwise complete new-style archive,
            # while writers continue to persist the historical supervision key.
            data["component_log_risk"] = data["component_log_risk_upper"]
        else:
            raise ValueError(f"{source} is missing component_log_risk")
    statistics = np.asarray(data["statistics"])
    if statistics.ndim != 2 or min(statistics.shape) <= 0:
        raise ValueError("statistics must have shape [N, D] with N,D > 0")
    representation, _grid_schema, _grid_hash = _representation_contract(
        data, source=source
    )
    thresholds = np.asarray(data["thresholds"], dtype=np.float32)
    names = validate_statistics_names(
        data["statistics_names"], expected_dim=statistics.shape[1]
    )
    data["statistics_names"] = np.asarray(names, dtype=str)
    schema_version = _scalar_string(
        data["statistics_schema_version"], "statistics_schema_version"
    )
    data["statistics_schema_version"] = np.asarray(schema_version)
    if "statistics_names_sha256" in data:
        recorded_hash = str(np.asarray(data["statistics_names_sha256"]).item())
        if recorded_hash != statistics_names_sha256(names):
            raise ValueError("statistics_names_sha256 does not match statistics_names")
    elif representation == LOGIT_REPRESENTATION:
        raise ValueError("v4 raw-logit archives require statistics_names_sha256")
    if "feature_schema_sha256" in data:
        recorded_feature_hash = _scalar_string(
            data["feature_schema_sha256"], "feature_schema_sha256"
        )
        expected_feature_hash = feature_schema_sha256(
            schema_version,
            statistics_names=names,
        )
        if recorded_feature_hash != expected_feature_hash:
            raise ValueError(
                "feature_schema_sha256 does not match the ordered feature schema"
            )
    expected = (statistics.shape[0], thresholds.size)

    component_schema = data.get("component_risk_schema_version")
    if component_schema is not None:
        component_schema_text = _scalar_string(
            component_schema, "component_risk_schema_version"
        )
        if component_schema_text != COMPONENT_RISK_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported component risk schema {component_schema_text!r}; "
                f"expected {COMPONENT_RISK_SCHEMA_VERSION!r}"
            )
        missing_component_evidence = [
            key
            for key in ("component_log_risk_raw", "component_log_risk_upper")
            if key not in data
        ]
        if missing_component_evidence:
            raise ValueError(
                "Component risk schema declares incomplete evidence; missing: "
                + ", ".join(missing_component_evidence)
            )

    if "component_log_risk_raw" in data and "component_log_risk_upper" not in data:
        data["component_log_risk_upper"] = _component_upper_envelope(
            np.asarray(data["component_log_risk_raw"])
        )
    elif "component_log_risk_upper" not in data:
        # Legacy archives stored only the supervised/enveloped curve.  Keep
        # them readable and expose that known curve as upper evidence, but do
        # not fabricate the irrecoverable pre-envelope raw counts curve.
        data["component_log_risk_upper"] = data["component_log_risk"]

    curve_keys = [
        "pixel_log_risk",
        "component_log_risk",
        "component_log_risk_upper",
        "pd_curve",
    ]
    if "component_log_risk_raw" in data:
        curve_keys.append("component_log_risk_raw")
    for key in curve_keys:
        if np.asarray(data[key]).shape != expected:
            raise ValueError(f"{key} must have shape {expected}")
    for key in ("statistics", "thresholds", *curve_keys):
        if not np.isfinite(np.asarray(data[key])).all():
            raise ValueError(f"{key} contains NaN or infinite values")

    if "component_log_risk_raw" in data:
        expected_upper = _component_upper_envelope(
            np.asarray(data["component_log_risk_raw"], dtype=np.float64)
        )
        if not np.allclose(
            np.asarray(data["component_log_risk_upper"], dtype=np.float64),
            expected_upper,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError(
                "component_log_risk_upper must be the suffix-maximum envelope "
                "of component_log_risk_raw"
            )

    component_alias = data.get("component_log_risk_alias")
    if component_alias is not None:
        alias_target = _scalar_string(component_alias, "component_log_risk_alias")
        if alias_target not in {
            "component_log_risk_raw",
            "component_log_risk_upper",
        }:
            raise ValueError("component_log_risk_alias names an unsupported target")
        if alias_target not in data:
            raise ValueError(f"component_log_risk_alias target {alias_target!r} is missing")
        if not np.allclose(
            np.asarray(data["component_log_risk"], dtype=np.float64),
            np.asarray(data[alias_target], dtype=np.float64),
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError(f"component_log_risk does not match {alias_target}")
    pd_curve = np.asarray(data["pd_curve"])
    if np.any((pd_curve < 0.0) | (pd_curve > 1.0)):
        raise ValueError("pd_curve values must lie in [0, 1]")
    validate_count_all_adaptation_contract(data, required=False)
    return data


def validate_archive_compatibility(
    train_archive: dict[str, np.ndarray],
    validation_archive: dict[str, np.ndarray],
) -> tuple[str, ...]:
    """Require identical threshold and ordered feature schemas across splits."""

    train_thresholds = np.asarray(train_archive["thresholds"], dtype=np.float32)
    validation_thresholds = np.asarray(
        validation_archive["thresholds"], dtype=np.float32
    )
    if not np.array_equal(train_thresholds, validation_thresholds):
        raise ValueError("Train and validation threshold grids differ")
    train_representation = _scalar_string(
        train_archive["representation"], "train.representation"
    )
    validation_representation = _scalar_string(
        validation_archive["representation"], "validation.representation"
    )
    if train_representation != validation_representation:
        raise ValueError("Train and validation threshold representations differ")
    train_grid_schema = _scalar_string(
        train_archive["threshold_grid_schema_version"],
        "train.threshold_grid_schema_version",
    )
    validation_grid_schema = _scalar_string(
        validation_archive["threshold_grid_schema_version"],
        "validation.threshold_grid_schema_version",
    )
    if train_grid_schema != validation_grid_schema:
        raise ValueError("Train and validation threshold-grid schemas differ")
    train_grid_hash = _scalar_string(
        train_archive["threshold_grid_sha256"], "train.threshold_grid_sha256"
    )
    validation_grid_hash = _scalar_string(
        validation_archive["threshold_grid_sha256"],
        "validation.threshold_grid_sha256",
    )
    if train_grid_hash != validation_grid_hash:
        raise ValueError("Train and validation semantic threshold-grid hashes differ")
    train_detector_protocol = (
        _scalar_string(
            train_archive["threshold_grid_detector_protocol"],
            "train.threshold_grid_detector_protocol",
        )
        if train_representation == LOGIT_REPRESENTATION
        else None
    )
    validation_detector_protocol = (
        _scalar_string(
            validation_archive["threshold_grid_detector_protocol"],
            "validation.threshold_grid_detector_protocol",
        )
        if validation_representation == LOGIT_REPRESENTATION
        else None
    )
    if train_detector_protocol != validation_detector_protocol:
        raise ValueError("Train and validation grid detector protocols differ")
    train_detector_roles = (
        _detector_role_hashes(train_archive, field_prefix="train.")
        if train_representation == LOGIT_REPRESENTATION
        else None
    )
    validation_detector_roles = (
        _detector_role_hashes(validation_archive, field_prefix="validation.")
        if validation_representation == LOGIT_REPRESENTATION
        else None
    )
    if train_detector_roles != validation_detector_roles:
        raise ValueError(
            "Train and validation grid detector role/checkpoint hashes differ"
        )
    train_grid_manifest_hash = (
        _scalar_string(
            train_archive["threshold_grid_manifest_sha256"],
            "train.threshold_grid_manifest_sha256",
        )
        if "threshold_grid_manifest_sha256" in train_archive
        else None
    )
    validation_grid_manifest_hash = (
        _scalar_string(
            validation_archive["threshold_grid_manifest_sha256"],
            "validation.threshold_grid_manifest_sha256",
        )
        if "threshold_grid_manifest_sha256" in validation_archive
        else None
    )
    if train_grid_manifest_hash != validation_grid_manifest_hash:
        raise ValueError("Train and validation threshold-grid manifest hashes differ")
    if (
        train_representation == LOGIT_REPRESENTATION
        and train_grid_manifest_hash is None
    ):
        raise ValueError("v4 raw-logit training requires threshold_grid_manifest_sha256")
    train_names = validate_statistics_names(
        train_archive["statistics_names"],
        expected_dim=np.asarray(train_archive["statistics"]).shape[1],
    )
    validation_names = validate_statistics_names(
        validation_archive["statistics_names"],
        expected_dim=np.asarray(validation_archive["statistics"]).shape[1],
    )
    if train_names != validation_names:
        raise ValueError(
            "Train and validation statistics_names must match exactly in order"
        )
    train_version = str(np.asarray(train_archive["statistics_schema_version"]).item())
    validation_version = str(
        np.asarray(validation_archive["statistics_schema_version"]).item()
    )
    if train_version != validation_version:
        raise ValueError("Train and validation statistics schema versions differ")

    train_feature_hash = (
        _scalar_string(
            train_archive["feature_schema_sha256"], "train.feature_schema_sha256"
        )
        if "feature_schema_sha256" in train_archive
        else None
    )
    validation_feature_hash = (
        _scalar_string(
            validation_archive["feature_schema_sha256"],
            "validation.feature_schema_sha256",
        )
        if "feature_schema_sha256" in validation_archive
        else None
    )
    if train_feature_hash != validation_feature_hash:
        raise ValueError("Train and validation feature-schema hashes differ")
    if train_representation == LOGIT_REPRESENTATION and train_feature_hash is None:
        raise ValueError("v4 raw-logit training requires feature_schema_sha256")

    train_component_version = (
        _scalar_string(
            train_archive["component_risk_schema_version"],
            "train.component_risk_schema_version",
        )
        if "component_risk_schema_version" in train_archive
        else None
    )
    validation_component_version = (
        _scalar_string(
            validation_archive["component_risk_schema_version"],
            "validation.component_risk_schema_version",
        )
        if "component_risk_schema_version" in validation_archive
        else None
    )
    if train_component_version != validation_component_version:
        raise ValueError(
            "Train and validation component risk schema versions differ; "
            "versioned and legacy component-risk archives cannot be mixed"
        )

    train_component_alias = (
        _scalar_string(
            train_archive["component_log_risk_alias"],
            "train.component_log_risk_alias",
        )
        if "component_log_risk_alias" in train_archive
        else None
    )
    validation_component_alias = (
        _scalar_string(
            validation_archive["component_log_risk_alias"],
            "validation.component_log_risk_alias",
        )
        if "component_log_risk_alias" in validation_archive
        else None
    )
    if train_component_alias != validation_component_alias:
        raise ValueError(
            "Train and validation component_log_risk_alias values differ"
        )

    if train_component_version is None:
        # Historical archives predate both metadata fields and remain usable as
        # a matched pair.  An alias without a schema is ambiguous and therefore
        # is not accepted as a training contract.
        if train_component_alias is not None:
            raise ValueError(
                "Unversioned component-risk archives used for training must not "
                "declare component_log_risk_alias"
            )
    else:
        if train_component_version != COMPONENT_RISK_SCHEMA_VERSION:
            # ``load_curve_archive`` normally catches this first; retain the
            # compatibility guard for callers supplying archive dictionaries.
            raise ValueError(
                f"Unsupported training component risk schema "
                f"{train_component_version!r}"
            )
        if train_component_alias != "component_log_risk_upper":
            if train_component_alias == "component_log_risk_raw":
                raise ValueError(
                    "component_log_risk_raw is diagnostic evidence only; formal "
                    "risk-curve training requires component_log_risk_alias="
                    "'component_log_risk_upper'"
                )
            raise ValueError(
                "Versioned component-risk archives used for training must declare "
                "component_log_risk_alias='component_log_risk_upper'"
            )
    return train_names


class CurveDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        statistics_mean: np.ndarray | None = None,
        statistics_std: np.ndarray | None = None,
    ) -> None:
        data = load_curve_archive(path)
        statistics = np.asarray(data["statistics"], dtype=np.float32)
        if statistics_mean is None:
            statistics_mean = np.zeros(statistics.shape[1], dtype=np.float32)
        if statistics_std is None:
            statistics_std = np.ones(statistics.shape[1], dtype=np.float32)
        mean = np.asarray(statistics_mean, dtype=np.float32)
        std = np.asarray(statistics_std, dtype=np.float32)
        if mean.shape != (statistics.shape[1],) or std.shape != (statistics.shape[1],):
            raise ValueError("Statistic normalisation has the wrong feature dimension")
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std < 0.0)
        ):
            raise ValueError("Statistic normalisation must be finite with non-negative std")
        normalised = (statistics - mean) / np.maximum(std, 1e-6)
        if not np.isfinite(normalised).all():
            raise ValueError("Normalised statistics contain NaN or infinite values")
        self.statistics = torch.from_numpy(normalised)
        self.pixel = torch.from_numpy(np.asarray(data["pixel_log_risk"], dtype=np.float32))
        self.component = torch.from_numpy(np.asarray(data["component_log_risk"], dtype=np.float32))
        self.component_upper = torch.from_numpy(
            np.asarray(data["component_log_risk_upper"], dtype=np.float32)
        )
        self.component_raw = (
            torch.from_numpy(
                np.asarray(data["component_log_risk_raw"], dtype=np.float32)
            )
            if "component_log_risk_raw" in data
            else None
        )
        self.thresholds = np.asarray(data["thresholds"], dtype=np.float32)
        self.statistics_names = validate_statistics_names(
            data["statistics_names"], expected_dim=statistics.shape[1]
        )
        self.statistics_schema_version = str(
            np.asarray(data["statistics_schema_version"]).item()
        )
        self.representation = _scalar_string(data["representation"], "representation")
        self.threshold_grid_schema_version = _scalar_string(
            data["threshold_grid_schema_version"], "threshold_grid_schema_version"
        )
        self.threshold_grid_sha256 = _scalar_string(
            data["threshold_grid_sha256"], "threshold_grid_sha256"
        )
        self.threshold_grid_manifest_sha256 = (
            _scalar_string(
                data["threshold_grid_manifest_sha256"],
                "threshold_grid_manifest_sha256",
            )
            if "threshold_grid_manifest_sha256" in data
            else None
        )
        self.threshold_grid_detector_protocol = (
            _scalar_string(
                data["threshold_grid_detector_protocol"],
                "threshold_grid_detector_protocol",
            )
            if self.representation == LOGIT_REPRESENTATION
            else None
        )
        self.threshold_grid_detector_checkpoint_sha256s = (
            _detector_role_hashes(data)[0]
            if self.representation == LOGIT_REPRESENTATION
            else None
        )
        self.threshold_grid_outer_detector_checkpoint_sha256 = (
            _detector_role_hashes(data)[1]
            if self.representation == LOGIT_REPRESENTATION
            else None
        )
        self.threshold_grid_episode_detector_checkpoint_sha256s = (
            _detector_role_hashes(data)[2]
            if self.representation == LOGIT_REPRESENTATION
            else None
        )
        self.feature_schema_sha256 = (
            _scalar_string(data["feature_schema_sha256"], "feature_schema_sha256")
            if "feature_schema_sha256" in data
            else None
        )

    def __len__(self) -> int:
        return int(self.statistics.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = {
            "statistics": self.statistics[index],
            "pixel_log_risk": self.pixel[index],
            "component_log_risk": self.component[index],
            "component_log_risk_upper": self.component_upper[index],
        }
        if self.component_raw is not None:
            sample["component_log_risk_raw"] = self.component_raw[index]
        return sample
