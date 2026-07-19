"""Select a zero-label threshold satisfying predicted pixel and component budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .build_deployment_statistics import (
    DEPLOYMENT_STATISTICS_SCHEMA_VERSION,
    LOGIT_DEPLOYMENT_STATISTICS_SCHEMA_VERSION,
    LOGIT_STATIC_CROSS_FIT_STATISTICS_SCHEMA_VERSION,
    STATIC_CROSS_FIT_STATISTICS_SCHEMA_VERSION,
    _raw_logit_protocol,
    _require_raw_logit_manifest_contract,
    _validate_grid_detector_contract,
    stable_block_id,
    stable_cross_fit_fold_id,
)
from .deployment_contract import audit_checkpoint_deployment_contract
from .domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    STATISTICS_SCHEMA_VERSION,
    extract_logit_window_statistics,
    extract_window_statistics,
    feature_schema_sha256,
    load_source_reference,
    statistics_names_sha256,
    validate_statistics_names,
)
from .monotone_curve_predictor import (
    COMPONENT_LOG_RISK_FLOOR,
    PIXEL_LOG_RISK_FLOOR,
    RiskCurvePredictor,
)
from .train_curve_predictor import (
    TRAINING_EPISODE_CONTRACT_SCHEMA_VERSION,
    validate_curve_checkpoint_contract,
)
from .threshold_grid import (
    threshold_grid_sha256,
    threshold_grid_version,
    validate_threshold_grid,
)
from .representation import (
    EMPTY_ACTION_THRESHOLD,
    GRID_DETECTOR_PROTOCOL,
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_PREDICTION_RULE,
    LOGIT_REPRESENTATION,
    PROBABILITY_REPRESENTATION,
    empty_action_contract,
    logit_threshold_grid_sha256,
    validate_logit_threshold_grid,
)


DEFAULT_OOD_Z_THRESHOLD = 8.0
DEFAULT_MATCHING_RULE = "overlap"
DEFAULT_CENTROID_DISTANCE = 3.0
DEFAULT_CONNECTIVITY = 2
DEFAULT_MIN_COMPONENT_AREA = 1
ZERO_RESULT_SCHEMA_VERSION = "rc-v2-zero-label-result-v2-formal"
SELECTION_DATA_CONTRACT_SCHEMA_VERSION = "rc-v2-zero-selection-data-v1"


def select_dual_budget_threshold(
    thresholds: np.ndarray,
    pixel_log_risk: np.ndarray,
    component_log_risk: np.ndarray,
    pixel_budget: float,
    component_budget: float,
    *,
    representation: str = PROBABILITY_REPRESENTATION,
) -> tuple[float, bool, int | None]:
    if representation == PROBABILITY_REPRESENTATION:
        grid = validate_threshold_grid(np.asarray(thresholds)).astype(np.float64)
        empty_threshold = 1.0
    elif representation == LOGIT_REPRESENTATION:
        grid = validate_logit_threshold_grid(np.asarray(thresholds)).astype(np.float64)
        empty_threshold = EMPTY_ACTION_THRESHOLD
    else:
        raise ValueError(f"Unsupported score representation: {representation!r}")
    pixel = np.asarray(pixel_log_risk, dtype=np.float64).reshape(-1)
    component = np.asarray(component_log_risk, dtype=np.float64).reshape(-1)
    if grid.shape != pixel.shape or grid.shape != component.shape:
        raise ValueError("threshold and risk curves must have identical lengths")
    if not np.isfinite(pixel).all() or not np.isfinite(component).all():
        raise ValueError("Predicted risk curves must contain only finite values")
    if np.any(np.diff(pixel) > 1e-6) or np.any(np.diff(component) > 1e-6):
        raise ValueError("Predicted risk curves must be non-increasing")
    if pixel.min() < PIXEL_LOG_RISK_FLOOR - 1e-6:
        raise ValueError("Pixel risk prediction falls below its physical epsilon floor")
    if component.min() < COMPONENT_LOG_RISK_FLOOR - 1e-6:
        raise ValueError("Component risk prediction falls below its physical epsilon floor")
    if (
        not np.isfinite(pixel_budget)
        or not np.isfinite(component_budget)
        or pixel_budget <= 0.0
        or component_budget <= 0.0
    ):
        raise ValueError("Budgets must be finite and positive")
    feasible = np.flatnonzero(
        (pixel <= np.log10(pixel_budget))
        & (component <= np.log10(component_budget))
    )
    if feasible.size == 0:
        return empty_threshold, True, None
    index = int(feasible[0])
    return float(grid[index]), False, index


def assess_ood_statistics(
    normalised_statistics: np.ndarray,
    statistics_names: tuple[str, ...] | list[str],
    z_threshold: float = DEFAULT_OOD_Z_THRESHOLD,
) -> dict[str, Any]:
    """Conservatively reject feature vectors far outside train normalisation."""

    values = np.asarray(normalised_statistics, dtype=np.float64).reshape(-1)
    names = validate_statistics_names(statistics_names, expected_dim=values.size)
    if not np.isfinite(values).all():
        raise ValueError("Normalised statistics contain NaN or infinite values")
    if not np.isfinite(z_threshold) or z_threshold <= 0.0:
        raise ValueError("OOD z threshold must be finite and positive")
    absolute = np.abs(values)
    offending = np.flatnonzero(absolute > z_threshold)
    ranked = sorted(offending.tolist(), key=lambda index: -absolute[index])
    return {
        "is_ood": bool(offending.size),
        "z_threshold": float(z_threshold),
        "max_abs_z": float(absolute.max(initial=0.0)),
        "num_features_exceeding": int(offending.size),
        "top_exceeding_features": [
            {"name": names[index], "abs_z": float(absolute[index])}
            for index in ranked[:10]
        ],
    }


def _sigmoid_display(logit_threshold: float) -> float:
    value = float(logit_threshold)
    if value >= 0.0:
        return float(1.0 / (1.0 + np.exp(-value)))
    exponential = float(np.exp(value))
    return float(exponential / (1.0 + exponential))


def apply_selected_threshold(
    scores: np.ndarray,
    threshold: float,
    *,
    representation: str,
) -> np.ndarray:
    """Apply the selected action in its native score domain."""

    values = np.asarray(scores)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Score array must be non-empty and finite")
    if representation == LOGIT_REPRESENTATION:
        if values.dtype != np.float32:
            raise ValueError("Raw-logit score array must use float32")
        if not (np.isfinite(threshold) or float(threshold) == EMPTY_ACTION_THRESHOLD):
            raise ValueError("Raw-logit threshold must be finite or the +inf empty action")
        return values >= float(threshold)
    if representation == PROBABILITY_REPRESENTATION:
        if not np.isfinite(threshold) or not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("Probability threshold must lie in [0, 1]")
        return values >= float(threshold)
    raise ValueError(f"Unsupported score representation: {representation!r}")


def _load_checkpoint(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"Curve checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Curve checkpoint must contain a mapping")
    required = {
        "model_config",
        "state_dict",
        "thresholds",
        "threshold_grid_version",
        "threshold_grid_sha256",
        "statistics_mean",
        "statistics_std",
        "statistics_names",
        "statistics_names_sha256",
        "statistics_schema_version",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"Curve checkpoint is missing: {', '.join(missing)}")
    representation_contract = validate_curve_checkpoint_contract(checkpoint)
    representation = str(representation_contract["representation"])
    expected_statistics_schema = (
        LOGIT_STATISTICS_SCHEMA_VERSION
        if representation == LOGIT_REPRESENTATION
        else STATISTICS_SCHEMA_VERSION
    )
    if str(checkpoint["statistics_schema_version"]) != expected_statistics_schema:
        raise ValueError("Curve checkpoint uses an incompatible statistics schema")
    checkpoint_names = validate_statistics_names(checkpoint["statistics_names"])
    if str(checkpoint["statistics_names_sha256"]) != statistics_names_sha256(
        checkpoint_names
    ):
        raise ValueError("Curve checkpoint statistics_names hash mismatch")
    if representation == LOGIT_REPRESENTATION:
        thresholds = validate_logit_threshold_grid(
            np.asarray(checkpoint["thresholds"], dtype=np.float32)
        )
        if str(checkpoint.get("threshold_grid_schema_version")) != LOGIT_GRID_SCHEMA_VERSION:
            raise ValueError("Curve checkpoint raw-logit grid schema mismatch")
        if str(checkpoint["threshold_grid_sha256"]) != logit_threshold_grid_sha256(
            thresholds
        ):
            raise ValueError("Curve checkpoint raw-logit grid semantic hash mismatch")
    else:
        thresholds = validate_threshold_grid(np.asarray(checkpoint["thresholds"]))
        if str(checkpoint["threshold_grid_sha256"]) != threshold_grid_sha256(thresholds):
            raise ValueError("Curve checkpoint threshold-grid hash mismatch")
        if str(checkpoint["threshold_grid_version"]) != threshold_grid_version(thresholds):
            raise ValueError("Curve checkpoint threshold-grid version mismatch")
    config = checkpoint["model_config"]
    model = RiskCurvePredictor(**config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def _score_map_protocol_provenance(
    score_map_dir: Path,
    thresholds: np.ndarray,
    *,
    representation: str = PROBABILITY_REPRESENTATION,
    threshold_grid_detector_protocol: str | None = None,
    threshold_grid_detector_checkpoint_sha256s: list[str] | tuple[str, ...] | None = None,
    threshold_grid_outer_detector_checkpoint_sha256: str | None = None,
    threshold_grid_episode_detector_checkpoint_sha256s: (
        list[str] | tuple[str, ...] | None
    ) = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Freeze score-map provenance using the certification protocol schema."""

    # Keep this import local so deployment from a precomputed statistics file
    # does not acquire an unnecessary certification dependency.
    from certification.build_calibration_losses import score_map_protocol

    manifest_path = score_map_dir / "manifest.json"
    if representation == LOGIT_REPRESENTATION:
        from evaluation.artifact_integrity import verify_score_map_directory

        manifest, _files, integrity = verify_score_map_directory(
            score_map_dir, require_integrity=True
        )
        _require_raw_logit_manifest_contract(manifest, integrity)
        assert manifest is not None
        protocol, fingerprint = _raw_logit_protocol(
            score_map_dir,
            manifest,
            thresholds,
            matching_rule=DEFAULT_MATCHING_RULE,
            centroid_distance=DEFAULT_CENTROID_DISTANCE,
            connectivity=DEFAULT_CONNECTIVITY,
            min_component_area=DEFAULT_MIN_COMPONENT_AREA,
            grid_detector_protocol=str(threshold_grid_detector_protocol),
            grid_detector_checkpoint_sha256s=(
                threshold_grid_detector_checkpoint_sha256s or ()
            ),
            grid_outer_detector_checkpoint_sha256=str(
                threshold_grid_outer_detector_checkpoint_sha256 or ""
            ),
            grid_episode_detector_checkpoint_sha256s=(
                threshold_grid_episode_detector_checkpoint_sha256s or ()
            ),
        )
    elif representation == PROBABILITY_REPRESENTATION:
        protocol, fingerprint = score_map_protocol(
            score_map_dir,
            thresholds,
            matching_rule=DEFAULT_MATCHING_RULE,
            centroid_distance=DEFAULT_CENTROID_DISTANCE,
            connectivity=DEFAULT_CONNECTIVITY,
            min_component_area=DEFAULT_MIN_COMPONENT_AREA,
        )
    else:
        raise ValueError(f"Unsupported score representation: {representation!r}")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    provenance = {
        "source_type": "exported_score_map_directory",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "detector_weight_sha256": protocol["detector_weight_sha256"],
        "warm_flag": protocol["warm_flag"],
        "spatial_mode": protocol["spatial_mode"],
        "target_dataset": protocol["target_dataset"],
        "masks_read": False,
        "representation": representation,
        "threshold_grid_schema_version": protocol.get(
            "threshold_grid_schema_version"
        ),
        "threshold_grid_sha256": protocol.get("threshold_grid_sha256"),
        "threshold_grid_detector_protocol": protocol.get(
            "threshold_grid_detector_protocol"
        ),
        "threshold_grid_detector_checkpoint_sha256s": protocol.get(
            "threshold_grid_detector_checkpoint_sha256s"
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": protocol.get(
            "threshold_grid_outer_detector_checkpoint_sha256"
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": protocol.get(
            "threshold_grid_episode_detector_checkpoint_sha256s"
        ),
    }
    return protocol, fingerprint, provenance


def _statistics_from_score_dir(
    path: Path,
    source_reference_path: str | None = None,
    window_size: int | None = 32,
    *,
    representation: str = PROBABILITY_REPRESENTATION,
) -> tuple[np.ndarray, tuple[str, ...], list[str], str]:
    manifest_path = path / "manifest.json"
    if representation == LOGIT_REPRESENTATION:
        from evaluation.artifact_integrity import verify_score_map_directory

        manifest, files, integrity = verify_score_map_directory(
            path, require_integrity=True
        )
        _require_raw_logit_manifest_contract(manifest, integrity)
    else:
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = manifest.get("records")
            files = (
                [path / str(record["file"]) for record in records]
                if records
                else sorted(path.glob("*.npz"))
            )
        else:
            files = sorted(path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz score maps found under {path}")
    if window_size is not None:
        if window_size < 1:
            raise ValueError("warmup window must be positive")
        if len(files) < window_size:
            raise ValueError(f"Warmup needs {window_size} maps, but {path} contains {len(files)}")
        files = files[:window_size]
    scores: list[np.ndarray] = []
    grays: list[np.ndarray | None] = []
    image_ids: list[str] = []
    for filename in files:
        with np.load(filename, allow_pickle=False) as sample:
            score_field = "logit" if representation == LOGIT_REPRESENTATION else "prob"
            if score_field not in sample:
                raise ValueError(
                    f"Score map {filename.name} lacks required {score_field!r} array"
                )
            score = np.asarray(sample[score_field])
            if representation == LOGIT_REPRESENTATION and score.dtype != np.float32:
                raise ValueError("Raw-logit deployment arrays must use float32")
            scores.append(score)
            grays.append(sample["gray"] if "gray" in sample else None)
            image_ids.append(_scalar_np_string(sample["image_id"]) if "image_id" in sample else filename.stem)
    reference = load_source_reference(source_reference_path) if source_reference_path else None
    if representation == LOGIT_REPRESENTATION:
        result = extract_logit_window_statistics(
            scores, grays, source_reference=reference
        )
    else:
        result = extract_window_statistics(scores, grays, source_reference=reference)
    return result.values, result.names, image_ids, result.schema_version


def _statistics_from_archive(
    path: Path,
) -> tuple[
    np.ndarray,
    tuple[str, ...],
    str,
    list[list[str]],
    list[list[str]],
    dict[str, Any] | None,
    str | None,
    dict[str, Any] | None,
]:
    if not path.is_file():
        raise FileNotFoundError(f"Statistics archive does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"statistics", "statistics_names", "statistics_schema_version"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"Statistics archive is missing: {', '.join(missing)}")
        statistics = np.asarray(archive["statistics"], dtype=np.float32)
        names = validate_statistics_names(archive["statistics_names"])
        schema_version = str(np.asarray(archive["statistics_schema_version"]).item())
        recorded_names_hash = (
            str(np.asarray(archive["statistics_names_sha256"]).item())
            if "statistics_names_sha256" in archive
            else None
        )
        if statistics.ndim == 1:
            statistics = statistics[None, :]
        if statistics.ndim != 2 or min(statistics.shape) <= 0:
            raise ValueError(
                "statistics-file must contain statistics with shape [D] or [N,D]"
            )
        num_windows = int(statistics.shape[0])
        if "evaluation_ids" in archive:
            evaluation_ids = _json_id_rows(
                archive["evaluation_ids"], num_windows, "evaluation_ids"
            )
        elif "image_ids" in archive:
            evaluation_ids = _image_id_rows(
                archive["image_ids"], num_windows, "image_ids"
            )
        else:
            evaluation_ids = []
        adaptation_ids = (
            _json_id_rows(archive["adaptation_ids"], num_windows, "adaptation_ids")
            if "adaptation_ids" in archive
            else []
        )
        protocol = (
            json.loads(str(np.asarray(archive["protocol_json"]).item()))
            if "protocol_json" in archive
            else None
        )
        protocol_fingerprint = (
            str(np.asarray(archive["protocol_fingerprint"]).item())
            if "protocol_fingerprint" in archive
            else None
        )
        archive_provenance = (
            json.loads(str(np.asarray(archive["provenance_json"]).item()))
            if "provenance_json" in archive
            else None
        )
        deployment_schema_version = (
            str(np.asarray(archive["deployment_statistics_schema_version"]).item())
            if "deployment_statistics_schema_version" in archive
            else None
        )
        archive_grid_sha256 = (
            str(np.asarray(archive["threshold_grid_sha256"]).item())
            if "threshold_grid_sha256" in archive
            else None
        )
        archive_representation = (
            str(np.asarray(archive["representation"]).item())
            if "representation" in archive
            else PROBABILITY_REPRESENTATION
        )
        archive_grid_schema_version = (
            str(np.asarray(archive["threshold_grid_schema_version"]).item())
            if "threshold_grid_schema_version" in archive
            else None
        )
        archive_feature_schema_sha256 = (
            str(np.asarray(archive["feature_schema_sha256"]).item())
            if "feature_schema_sha256" in archive
            else None
        )
        archive_grid_manifest_sha256 = (
            str(np.asarray(archive["threshold_grid_manifest_sha256"]).item())
            if "threshold_grid_manifest_sha256" in archive
            else None
        )
        archive_grid_detector_protocol = (
            str(np.asarray(archive["threshold_grid_detector_protocol"]).item())
            if "threshold_grid_detector_protocol" in archive
            else None
        )
        archive_grid_detector_checkpoint_sha256s = (
            [
                str(value)
                for value in np.asarray(
                    archive["threshold_grid_detector_checkpoint_sha256s"]
                ).reshape(-1)
            ]
            if "threshold_grid_detector_checkpoint_sha256s" in archive
            else []
        )
        archive_grid_outer_detector_checkpoint_sha256 = (
            str(
                np.asarray(
                    archive["threshold_grid_outer_detector_checkpoint_sha256"]
                ).item()
            )
            if "threshold_grid_outer_detector_checkpoint_sha256" in archive
            else None
        )
        archive_grid_episode_detector_checkpoint_sha256s = (
            [
                str(value)
                for value in np.asarray(
                    archive[
                        "threshold_grid_episode_detector_checkpoint_sha256s"
                    ]
                ).reshape(-1)
            ]
            if "threshold_grid_episode_detector_checkpoint_sha256s" in archive
            else []
        )
        archive_thresholds = (
            np.asarray(archive["thresholds"])
            if "thresholds" in archive
            else None
        )
        block_ids = (
            [_scalar_np_string(value) for value in np.asarray(archive["block_ids"])]
            if "block_ids" in archive
            else None
        )
        block_records = (
            _json_object_rows(
                archive["block_records_json"], num_windows, "block_records_json"
            )
            if "block_records_json" in archive
            else None
        )
        recorded_block_ids_sha256 = (
            str(np.asarray(archive["block_ids_sha256"]).item())
            if "block_ids_sha256" in archive
            else None
        )
        recorded_one_to_one = (
            _strict_scalar_bool(archive["one_to_one_evaluation"], "one_to_one_evaluation")
            if "one_to_one_evaluation" in archive
            else None
        )
        score_image_ids = (
            [_scalar_np_string(value) for value in np.asarray(archive["score_image_ids"])]
            if "score_image_ids" in archive
            else None
        )
    names = validate_statistics_names(names, expected_dim=statistics.shape[1])
    if recorded_names_hash is not None and recorded_names_hash != statistics_names_sha256(
        names
    ):
        raise ValueError("statistics-file statistics_names hash mismatch")
    if archive_representation == LOGIT_REPRESENTATION:
        if deployment_schema_version not in {
            LOGIT_DEPLOYMENT_STATISTICS_SCHEMA_VERSION,
            LOGIT_STATIC_CROSS_FIT_STATISTICS_SCHEMA_VERSION,
        }:
            raise ValueError("Raw-logit statistics-file has an incompatible deployment schema")
        if schema_version != LOGIT_STATISTICS_SCHEMA_VERSION:
            raise ValueError("Raw-logit statistics-file has an incompatible feature schema")
        if archive_grid_schema_version != LOGIT_GRID_SCHEMA_VERSION:
            raise ValueError("Raw-logit statistics-file grid schema mismatch")
        expected_feature_hash = feature_schema_sha256(
            LOGIT_STATISTICS_SCHEMA_VERSION, statistics_names=names
        )
        if archive_feature_schema_sha256 != expected_feature_hash:
            raise ValueError("Raw-logit statistics-file feature-schema hash mismatch")
        if (
            not isinstance(archive_grid_manifest_sha256, str)
            or len(archive_grid_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in archive_grid_manifest_sha256
            )
        ):
            raise ValueError("Raw-logit statistics-file grid-manifest hash is invalid")
        if archive_grid_detector_protocol != GRID_DETECTOR_PROTOCOL:
            raise ValueError("Raw-logit statistics-file grid detector protocol mismatch")
        (
            _validated_detector_protocol,
            validated_detector_hashes,
            validated_outer_detector_hash,
            validated_episode_detector_hashes,
        ) = _validate_grid_detector_contract(
            archive_grid_detector_protocol,
            archive_grid_detector_checkpoint_sha256s,
            archive_grid_outer_detector_checkpoint_sha256,
            archive_grid_episode_detector_checkpoint_sha256s,
            representation=archive_representation,
        )
        archive_grid_detector_checkpoint_sha256s = list(validated_detector_hashes)
        archive_grid_outer_detector_checkpoint_sha256 = (
            validated_outer_detector_hash
        )
        archive_grid_episode_detector_checkpoint_sha256s = list(
            validated_episode_detector_hashes
        )
        if archive_thresholds is None:
            raise ValueError("Raw-logit statistics-file lacks its finite threshold grid")
        raw_grid = validate_logit_threshold_grid(archive_thresholds)
        if archive_grid_sha256 != logit_threshold_grid_sha256(raw_grid):
            raise ValueError("Raw-logit statistics-file grid semantic hash mismatch")
    elif archive_representation != PROBABILITY_REPRESENTATION:
        raise ValueError(
            f"statistics-file uses unsupported representation {archive_representation!r}"
        )
    if statistics.shape[0] > 1 and not evaluation_ids:
        raise ValueError(
            "A batched statistics-file must contain evaluation_ids or image_ids"
        )
    if (protocol is None) != (protocol_fingerprint is None):
        raise ValueError(
            "statistics-file must contain both protocol_json and protocol_fingerprint"
        )
    if protocol is not None:
        from certification.build_calibration_losses import protocol_fingerprint as hash_protocol

        if not isinstance(protocol, dict) or hash_protocol(protocol) != protocol_fingerprint:
            raise ValueError("statistics-file protocol fingerprint mismatch")
    if archive_provenance is not None and not isinstance(archive_provenance, dict):
        raise ValueError("statistics-file provenance_json must decode to an object")
    if archive_representation == LOGIT_REPRESENTATION:
        if not isinstance(archive_provenance, dict):
            raise ValueError("Raw-logit statistics-file requires object provenance")
        expected_provenance_fields = {
            "representation": archive_representation,
            "threshold_grid_schema_version": archive_grid_schema_version,
            "threshold_grid_sha256": archive_grid_sha256,
            "threshold_grid_manifest_sha256": archive_grid_manifest_sha256,
            "feature_schema_sha256": archive_feature_schema_sha256,
            "threshold_grid_detector_protocol": archive_grid_detector_protocol,
            "threshold_grid_detector_checkpoint_sha256s": (
                archive_grid_detector_checkpoint_sha256s
            ),
            "threshold_grid_outer_detector_checkpoint_sha256": (
                archive_grid_outer_detector_checkpoint_sha256
            ),
            "threshold_grid_episode_detector_checkpoint_sha256s": (
                archive_grid_episode_detector_checkpoint_sha256s
            ),
        }
        for field, expected in expected_provenance_fields.items():
            if archive_provenance.get(field) != expected:
                raise ValueError(
                    f"Raw-logit statistics-file provenance {field} mismatch"
                )
        if not isinstance(protocol, dict):
            raise ValueError("Raw-logit statistics-file requires a bound protocol")
        for field, expected in {
            "representation": archive_representation,
            "threshold_grid_schema_version": archive_grid_schema_version,
            "threshold_grid_sha256": archive_grid_sha256,
            "threshold_grid_detector_protocol": archive_grid_detector_protocol,
            "threshold_grid_detector_checkpoint_sha256s": (
                archive_grid_detector_checkpoint_sha256s
            ),
            "threshold_grid_outer_detector_checkpoint_sha256": (
                archive_grid_outer_detector_checkpoint_sha256
            ),
            "threshold_grid_episode_detector_checkpoint_sha256s": (
                archive_grid_episode_detector_checkpoint_sha256s
            ),
        }.items():
            if protocol.get(field) != expected:
                raise ValueError(
                    f"Raw-logit statistics-file protocol {field} mismatch"
                )
        if protocol.get("detector_weight_sha256") != (
            archive_grid_outer_detector_checkpoint_sha256
        ):
            raise ValueError(
                "Raw-logit statistics-file protocol detector weight is not the "
                "outer-final detector"
            )
    identity_contract = _validate_archive_identity_contract(
        deployment_schema_version=deployment_schema_version,
        num_windows=int(statistics.shape[0]),
        adaptation_ids=adaptation_ids,
        evaluation_ids=evaluation_ids,
        block_ids=block_ids,
        block_records=block_records,
        recorded_block_ids_sha256=recorded_block_ids_sha256,
        recorded_one_to_one=recorded_one_to_one,
        score_image_ids=score_image_ids,
        provenance=archive_provenance,
    )
    archive_evidence = {
        "deployment_statistics_schema_version": deployment_schema_version,
        "threshold_grid_sha256": archive_grid_sha256,
        "representation": archive_representation,
        "threshold_grid_schema_version": archive_grid_schema_version,
        "feature_schema_sha256": archive_feature_schema_sha256,
        "threshold_grid_manifest_sha256": archive_grid_manifest_sha256,
        "threshold_grid_detector_protocol": archive_grid_detector_protocol,
        "threshold_grid_detector_checkpoint_sha256s": (
            archive_grid_detector_checkpoint_sha256s
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": (
            archive_grid_outer_detector_checkpoint_sha256
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": (
            archive_grid_episode_detector_checkpoint_sha256s
        ),
        "provenance": archive_provenance,
        "identity_contract": identity_contract,
    }
    return (
        statistics,
        names,
        schema_version,
        evaluation_ids,
        adaptation_ids,
        protocol,
        protocol_fingerprint,
        archive_evidence,
    )


def _validate_id_row(values: Any, field: str, row_index: int) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{field} row {row_index} must be a JSON ID list")
    ids = [str(value) for value in values]
    if not ids or any(not value.strip() for value in ids):
        raise ValueError(f"{field} row {row_index} must contain non-empty IDs")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{field} row {row_index} contains duplicate IDs")
    return ids


def _strict_scalar_bool(value: Any, field: str) -> bool:
    array = np.asarray(value)
    if array.ndim != 0 or not isinstance(array.item(), (bool, np.bool_)):
        raise ValueError(f"{field} must be a scalar boolean")
    return bool(array.item())


def _json_object_rows(
    values: np.ndarray,
    expected_rows: int,
    field: str,
) -> list[dict[str, Any]]:
    encoded = np.asarray(values)
    if encoded.ndim == 0 and expected_rows == 1:
        encoded = encoded.reshape(1)
    if encoded.ndim != 1 or encoded.shape[0] != expected_rows:
        raise ValueError(f"{field} must contain one JSON object per statistics row")
    rows: list[dict[str, Any]] = []
    for row_index, value in enumerate(encoded):
        try:
            decoded = json.loads(_scalar_np_string(value))
        except json.JSONDecodeError as error:
            raise ValueError(f"{field} row {row_index} is not valid JSON") from error
        if not isinstance(decoded, dict):
            raise ValueError(f"{field} row {row_index} must decode to an object")
        rows.append(decoded)
    return rows


def _validate_archive_identity_contract(
    *,
    deployment_schema_version: str | None,
    num_windows: int,
    adaptation_ids: list[list[str]],
    evaluation_ids: list[list[str]],
    block_ids: list[str] | None,
    block_records: list[dict[str, Any]] | None,
    recorded_block_ids_sha256: str | None,
    recorded_one_to_one: bool | None,
    score_image_ids: list[str] | None,
    provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate block identities before an archive can drive adaptive actions.

    Legacy statistics arrays without a deployment schema remain readable for
    diagnostic/global use.  Known causal and static schemas, however, are
    treated as identity-bearing artifacts and must be complete and internally
    consistent; silently accepting a partial block table would make an action
    appear to cover images that never received an independent prediction.
    """

    known_schema = deployment_schema_version in {
        DEPLOYMENT_STATISTICS_SCHEMA_VERSION,
        STATIC_CROSS_FIT_STATISTICS_SCHEMA_VERSION,
        LOGIT_DEPLOYMENT_STATISTICS_SCHEMA_VERSION,
        LOGIT_STATIC_CROSS_FIT_STATISTICS_SCHEMA_VERSION,
    }
    if not known_schema:
        return {
            "verified": False,
            "scope": "legacy_or_unknown_schema_diagnostic",
            "schema_version": deployment_schema_version,
        }
    if not adaptation_ids or not evaluation_ids:
        raise ValueError(
            "Deployment statistics must contain adaptation_ids and evaluation_ids"
        )
    if len(adaptation_ids) != num_windows or len(evaluation_ids) != num_windows:
        raise ValueError("Deployment ID rows do not match statistics row count")
    for row_index, (adaptation, evaluation) in enumerate(
        zip(adaptation_ids, evaluation_ids)
    ):
        overlap = sorted(set(adaptation).intersection(evaluation))
        if overlap:
            raise ValueError(
                f"Deployment row {row_index} reuses IDs in adaptation and evaluation: "
                + ", ".join(overlap[:10])
            )
    flattened_evaluation = [value for row in evaluation_ids for value in row]
    if len(flattened_evaluation) != len(set(flattened_evaluation)):
        raise ValueError("Evaluation IDs must be globally unique across deployment rows")
    if block_ids is None or block_records is None:
        raise ValueError("Deployment statistics lack block IDs or block records")
    if len(block_ids) != num_windows or len(block_records) != num_windows:
        raise ValueError("Deployment block rows do not match statistics row count")
    if any(not value.strip() for value in block_ids) or len(block_ids) != len(set(block_ids)):
        raise ValueError("Deployment block IDs must be non-empty and globally unique")
    actual_block_hash = hashlib.sha256("\n".join(block_ids).encode("utf-8")).hexdigest()
    if recorded_block_ids_sha256 != actual_block_hash:
        raise ValueError("Deployment block_ids_sha256 mismatch")
    if not isinstance(provenance, dict):
        raise ValueError("Deployment statistics require object provenance")
    if provenance.get("block_ids_sha256") != actual_block_hash:
        raise ValueError("Deployment provenance block_ids_sha256 mismatch")
    if provenance.get("num_windows") != num_windows:
        raise ValueError("Deployment provenance num_windows differs from block rows")

    static = deployment_schema_version in {
        STATIC_CROSS_FIT_STATISTICS_SCHEMA_VERSION,
        LOGIT_STATIC_CROSS_FIT_STATISTICS_SCHEMA_VERSION,
    }
    index_field = "fold_index" if static else "block_index"
    for row_index, (block_id, record, adaptation, evaluation) in enumerate(
        zip(block_ids, block_records, adaptation_ids, evaluation_ids)
    ):
        if record.get("block_id") != block_id:
            raise ValueError(f"Deployment block record {row_index} has a mismatched block_id")
        if record.get(index_field) != row_index:
            raise ValueError(
                f"Deployment block record {row_index} has a mismatched {index_field}"
            )
        if record.get("adaptation_ids") != adaptation:
            raise ValueError(
                f"Deployment block record {row_index} has mismatched adaptation_ids"
            )
        if record.get("evaluation_ids") != evaluation:
            raise ValueError(
                f"Deployment block record {row_index} has mismatched evaluation_ids"
            )
        if record.get("adaptation_size") != len(adaptation):
            raise ValueError(
                f"Deployment block record {row_index} has mismatched adaptation_size"
            )
        if record.get("evaluation_size") != len(evaluation):
            raise ValueError(
                f"Deployment block record {row_index} has mismatched evaluation_size"
            )
        expected_block_id = (
            stable_cross_fit_fold_id(
                row_index,
                adaptation,
                evaluation,
                schema_version=deployment_schema_version,
            )
            if static
            else stable_block_id(
                row_index,
                adaptation,
                evaluation,
                schema_version=deployment_schema_version,
            )
        )
        if block_id != expected_block_id:
            raise ValueError(
                f"Deployment block record {row_index} has a non-canonical block_id"
            )

    computed_one_to_one = all(len(row) == 1 for row in evaluation_ids)
    if static:
        if provenance.get("mode") != "static_cross_fit":
            raise ValueError(
                "Static cross-fit provenance must record mode='static_cross_fit'"
            )
        for field, expected in (
            ("masks_read", False),
            ("full_test_coverage", True),
            ("formal_crc_eligible", False),
        ):
            if provenance.get(field) is not expected:
                raise ValueError(
                    f"Static cross-fit provenance must record {field}={expected!r}"
                )
        if provenance.get("num_images") != len(flattened_evaluation):
            raise ValueError(
                "Static cross-fit provenance num_images differs from evaluation IDs"
            )
        if score_image_ids is None:
            raise ValueError("Static cross-fit archive lacks score_image_ids")
        if (
            not score_image_ids
            or any(not value.strip() for value in score_image_ids)
            or len(score_image_ids) != len(set(score_image_ids))
        ):
            raise ValueError("Static cross-fit score_image_ids must be unique non-empty IDs")
        if len(score_image_ids) != len(flattened_evaluation) or set(
            score_image_ids
        ) != set(flattened_evaluation):
            raise ValueError(
                "Static cross-fit evaluation IDs do not exactly cover score_image_ids"
            )
        from evaluation.artifact_integrity import ordered_ids_sha256

        if provenance.get("score_image_ids_sha256") != ordered_ids_sha256(
            score_image_ids
        ):
            raise ValueError("Static cross-fit score_image_ids_sha256 mismatch")
        adaptation_window = provenance.get("adaptation_window")
        if (
            isinstance(adaptation_window, bool)
            or not isinstance(adaptation_window, int)
            or adaptation_window <= 0
        ):
            raise ValueError(
                "Static cross-fit provenance adaptation_window must be positive"
            )
        if any(len(row) != adaptation_window for row in adaptation_ids):
            raise ValueError(
                "Static cross-fit adaptation rows do not match adaptation_window"
            )
        if provenance.get("selected_adaptation_ids_by_fold") != adaptation_ids:
            raise ValueError(
                "Static cross-fit selected adaptation IDs differ from archive rows"
            )
        if provenance.get("selected_adaptation_sizes") != [
            len(row) for row in adaptation_ids
        ]:
            raise ValueError("Static cross-fit selected adaptation sizes mismatch")
        complement_sizes = provenance.get("complement_sizes")
        if (
            not isinstance(complement_sizes, list)
            or len(complement_sizes) != num_windows
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < adaptation_window
                for value in complement_sizes
            )
        ):
            raise ValueError("Static cross-fit complement sizes are invalid")
        for row_index, (record, adaptation, complement_size) in enumerate(
            zip(block_records, adaptation_ids, complement_sizes)
        ):
            if (
                record.get("adaptation_window") != adaptation_window
                or record.get("selected_adaptation_ids") != adaptation
                or record.get("selected_adaptation_size") != len(adaptation)
                or record.get("complement_size") != complement_size
            ):
                raise ValueError(
                    f"Static cross-fit block record {row_index} sampling contract mismatch"
                )
        if recorded_one_to_one is not False:
            raise ValueError(
                "Static cross-fit must explicitly remain outside one-to-one causal CRC"
            )
        if any(record.get("protocol") != "static_cross_fit" for record in block_records):
            raise ValueError("Static cross-fit block records have an invalid protocol")
        scope = "static_cross_fit_full_coverage_empirical"
    else:
        if provenance.get("masks_read") is not False:
            raise ValueError("Causal deployment provenance must record masks_read=false")
        if recorded_one_to_one is not computed_one_to_one:
            raise ValueError(
                "Causal one_to_one_evaluation disagrees with evaluation block sizes"
            )
        if provenance.get("one_to_one_evaluation") is not recorded_one_to_one:
            raise ValueError(
                "Causal provenance one_to_one_evaluation disagrees with archive"
            )
        scope = "causal_blocks"
    return {
        "verified": True,
        "scope": scope,
        "schema_version": deployment_schema_version,
        "num_windows": num_windows,
        "num_evaluation_images": len(flattened_evaluation),
        "evaluation_ids_globally_unique": True,
        "adaptation_evaluation_disjoint_per_row": True,
        "block_ids_sha256": actual_block_hash,
        "computed_one_to_one_evaluation": computed_one_to_one,
    }


def _domain_identity(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty domain name")
    identity = "".join(character for character in value.casefold() if character.isalnum())
    if not identity:
        raise ValueError(f"{field} must contain an alphanumeric domain name")
    return identity


def _positive_contract_integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a positive integer") from error
    if integer <= 0 or integer != value:
        raise ValueError(f"{field} must be a positive integer")
    return integer


def _validate_static_checkpoint_compatibility(
    checkpoint: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any] | None,
    protocol: Mapping[str, Any] | None,
    expected_threshold_grid_sha256: str,
    expected_representation: str | None = None,
    expected_threshold_grid_schema_version: str | None = None,
    expected_feature_schema_sha256: str | None = None,
    expected_threshold_grid_manifest_sha256: str | None = None,
    expected_threshold_grid_detector_protocol: str | None = None,
    expected_threshold_grid_detector_checkpoint_sha256s: (
        list[str] | tuple[str, ...] | None
    ) = None,
    expected_threshold_grid_outer_detector_checkpoint_sha256: str | None = None,
    expected_threshold_grid_episode_detector_checkpoint_sha256s: (
        list[str] | tuple[str, ...] | None
    ) = None,
) -> dict[str, Any]:
    """Bind an empirical static fold artifact to its trained curve semantics."""

    if not isinstance(provenance, Mapping):
        raise ValueError("Static cross-fit deployment provenance is required")
    if not isinstance(protocol, Mapping):
        raise ValueError("Static cross-fit score-map protocol is required")
    contract = checkpoint.get("episode_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Curve checkpoint lacks episode_contract")
    if contract.get("schema_version") != TRAINING_EPISODE_CONTRACT_SCHEMA_VERSION:
        raise ValueError("Curve checkpoint training-contract schema is unsupported")
    if contract.get("verified") is not True:
        raise ValueError("Curve checkpoint training contract is not verified")
    if contract.get("formal_protocol_eligible") is not True:
        raise ValueError(
            "Static cross-fit requires a formally eligible curve-training contract"
        )
    adaptation_window = _positive_contract_integer(
        contract.get("adaptation_window"), "checkpoint adaptation_window"
    )
    evaluation_window = _positive_contract_integer(
        contract.get("evaluation_window"), "checkpoint evaluation_window"
    )
    stride = _positive_contract_integer(contract.get("stride"), "checkpoint stride")
    if evaluation_window != 1 or contract.get("one_to_one_future_target") is not True:
        raise ValueError(
            "Static image-level evaluation requires checkpoint E=1 one-to-one targets"
        )
    if stride != adaptation_window + evaluation_window:
        raise ValueError(
            "Static image-level evaluation requires disjoint A+E training stride"
        )
    deployed_adaptation_window = _positive_contract_integer(
        provenance.get("adaptation_window"), "static provenance adaptation_window"
    )
    if deployed_adaptation_window != adaptation_window:
        raise ValueError(
            "Static cross-fit adaptation_window differs from checkpoint semantics"
        )
    protocol_fields = contract.get("protocol_fields")
    if not isinstance(protocol_fields, Mapping):
        raise ValueError("Curve checkpoint lacks training protocol_fields")
    if protocol_fields.get("adaptation_window") != adaptation_window:
        raise ValueError(
            "Curve checkpoint protocol_fields disagree in adaptation_window"
        )
    if protocol_fields.get("evaluation_window") != evaluation_window:
        raise ValueError(
            "Curve checkpoint protocol_fields disagree in evaluation_window"
        )
    if protocol_fields.get("stride") != stride:
        raise ValueError("Curve checkpoint protocol_fields disagree in stride")

    if provenance.get("score_integrity_verified") is not True:
        raise ValueError(
            "Static cross-fit requires an integrity-verified target score artifact"
        )
    if provenance.get("score_num_records") != provenance.get("num_images"):
        raise ValueError("Static score integrity record count differs from num_images")
    for field in (
        "score_manifest_sha256",
        "score_records_sha256",
        "score_ordered_image_ids_sha256",
    ):
        value = provenance.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"Static provenance lacks a valid {field}")

    if checkpoint.get("threshold_grid_sha256") != expected_threshold_grid_sha256:
        raise ValueError("Curve checkpoint and static deployment threshold grids differ")
    if protocol_fields.get("threshold_grid_sha256") != expected_threshold_grid_sha256:
        raise ValueError("Curve training protocol and static threshold grids differ")
    if provenance.get("threshold_grid_sha256") != expected_threshold_grid_sha256:
        raise ValueError("Static provenance and curve checkpoint threshold grids differ")
    if protocol.get("threshold_grid_sha256") != expected_threshold_grid_sha256:
        raise ValueError("Static score-map protocol and curve checkpoint grids differ")
    if expected_representation is not None:
        for owner, value in (
            ("checkpoint", checkpoint.get("representation")),
            ("training protocol", protocol_fields.get("representation")),
            ("static provenance", provenance.get("representation")),
            ("score-map protocol", protocol.get("representation")),
        ):
            if value != expected_representation:
                raise ValueError(f"{owner} representation mismatch")
    if expected_threshold_grid_schema_version is not None:
        for owner, value in (
            ("checkpoint", checkpoint.get("threshold_grid_schema_version")),
            (
                "training protocol",
                protocol_fields.get("threshold_grid_schema_version"),
            ),
            (
                "static provenance",
                provenance.get("threshold_grid_schema_version"),
            ),
            (
                "score-map protocol",
                protocol.get("threshold_grid_schema_version"),
            ),
        ):
            if value != expected_threshold_grid_schema_version:
                raise ValueError(f"{owner} threshold-grid schema mismatch")
    if expected_feature_schema_sha256 is not None:
        for owner, value in (
            ("checkpoint", checkpoint.get("feature_schema_sha256")),
            ("training protocol", protocol_fields.get("feature_schema_sha256")),
            ("static provenance", provenance.get("feature_schema_sha256")),
        ):
            if value != expected_feature_schema_sha256:
                raise ValueError(f"{owner} feature-schema hash mismatch")
    if expected_threshold_grid_manifest_sha256 is not None:
        for owner, value in (
            (
                "checkpoint",
                checkpoint.get("threshold_grid_manifest_sha256"),
            ),
            (
                "training protocol",
                protocol_fields.get("threshold_grid_manifest_sha256"),
            ),
            (
                "static provenance",
                provenance.get("threshold_grid_manifest_sha256"),
            ),
        ):
            if value != expected_threshold_grid_manifest_sha256:
                raise ValueError(f"{owner} threshold-grid manifest hash mismatch")
    if expected_threshold_grid_detector_protocol is not None:
        for owner, value in (
            (
                "checkpoint",
                checkpoint.get("threshold_grid_detector_protocol"),
            ),
            (
                "training protocol",
                protocol_fields.get("threshold_grid_detector_protocol"),
            ),
            (
                "static provenance",
                provenance.get("threshold_grid_detector_protocol"),
            ),
            (
                "score-map protocol",
                protocol.get("threshold_grid_detector_protocol"),
            ),
        ):
            if value != expected_threshold_grid_detector_protocol:
                raise ValueError(f"{owner} grid detector protocol mismatch")
    if expected_threshold_grid_detector_checkpoint_sha256s is not None:
        expected_detector_hashes = list(
            expected_threshold_grid_detector_checkpoint_sha256s
        )
        for owner, value in (
            (
                "checkpoint",
                checkpoint.get("threshold_grid_detector_checkpoint_sha256s"),
            ),
            (
                "training protocol",
                protocol_fields.get(
                    "threshold_grid_detector_checkpoint_sha256s"
                ),
            ),
            (
                "static provenance",
                provenance.get("threshold_grid_detector_checkpoint_sha256s"),
            ),
            (
                "score-map protocol",
                protocol.get("threshold_grid_detector_checkpoint_sha256s"),
            ),
        ):
            if list(value or []) != expected_detector_hashes:
                raise ValueError(f"{owner} grid detector hashes mismatch")
    if expected_threshold_grid_outer_detector_checkpoint_sha256 is not None:
        for owner, value in (
            (
                "checkpoint",
                checkpoint.get(
                    "threshold_grid_outer_detector_checkpoint_sha256"
                ),
            ),
            (
                "training protocol",
                protocol_fields.get(
                    "threshold_grid_outer_detector_checkpoint_sha256"
                ),
            ),
            (
                "static provenance",
                provenance.get(
                    "threshold_grid_outer_detector_checkpoint_sha256"
                ),
            ),
            (
                "score-map protocol",
                protocol.get(
                    "threshold_grid_outer_detector_checkpoint_sha256"
                ),
            ),
        ):
            if value != expected_threshold_grid_outer_detector_checkpoint_sha256:
                raise ValueError(f"{owner} outer detector hash mismatch")
        if protocol.get("detector_weight_sha256") != (
            expected_threshold_grid_outer_detector_checkpoint_sha256
        ):
            raise ValueError(
                "Static score-map detector weight is not the outer-final detector"
            )
    if expected_threshold_grid_episode_detector_checkpoint_sha256s is not None:
        expected_episode_hashes = list(
            expected_threshold_grid_episode_detector_checkpoint_sha256s
        )
        for owner, value in (
            (
                "checkpoint",
                checkpoint.get(
                    "threshold_grid_episode_detector_checkpoint_sha256s"
                ),
            ),
            (
                "training protocol",
                protocol_fields.get(
                    "threshold_grid_episode_detector_checkpoint_sha256s"
                ),
            ),
            (
                "static provenance",
                provenance.get(
                    "threshold_grid_episode_detector_checkpoint_sha256s"
                ),
            ),
            (
                "score-map protocol",
                protocol.get(
                    "threshold_grid_episode_detector_checkpoint_sha256s"
                ),
            ),
        ):
            if list(value or []) != expected_episode_hashes:
                raise ValueError(f"{owner} episode detector hashes mismatch")

    pseudo_targets = protocol_fields.get("pseudo_targets")
    if not isinstance(pseudo_targets, (list, tuple)) or not pseudo_targets:
        raise ValueError("Curve checkpoint does not identify pseudo-target domains")
    pseudo_target_names = [str(value) for value in pseudo_targets]
    pseudo_target_identities = {
        _domain_identity(value, "checkpoint pseudo-target")
        for value in pseudo_target_names
    }
    target_dataset = str(protocol.get("target_dataset", ""))
    if _domain_identity(target_dataset, "static target_dataset") in pseudo_target_identities:
        raise ValueError(
            "Static target_dataset appears in risk-predictor pseudo-target training"
        )

    training_reference = {
        "sha256": protocol_fields.get("source_reference_sha256"),
        "domain_names": protocol_fields.get("source_reference_domain_names") or [],
        "statistics_names_sha256": protocol_fields.get(
            "source_reference_statistics_names_sha256"
        ),
    }
    deployment_reference = {
        "sha256": provenance.get("source_reference_sha256"),
        "domain_names": provenance.get("source_reference_domain_names") or [],
        "statistics_names_sha256": provenance.get(
            "source_reference_statistics_names_sha256"
        ),
    }
    if training_reference != deployment_reference:
        raise ValueError(
            "Curve checkpoint and static deployment source-reference contracts differ"
        )
    return {
        "verified": True,
        "scope": "static_cross_fit_empirical_compatibility",
        "adaptation_window": adaptation_window,
        "evaluation_window": evaluation_window,
        "stride": stride,
        "target_dataset": target_dataset,
        "pseudo_target_training_domains": pseudo_target_names,
        "target_domain_excluded_from_pseudo_targets": True,
        "threshold_grid_sha256": expected_threshold_grid_sha256,
        "representation": expected_representation,
        "threshold_grid_schema_version": expected_threshold_grid_schema_version,
        "feature_schema_sha256": expected_feature_schema_sha256,
        "threshold_grid_manifest_sha256": expected_threshold_grid_manifest_sha256,
        "threshold_grid_detector_protocol": (
            expected_threshold_grid_detector_protocol
        ),
        "threshold_grid_detector_checkpoint_sha256s": list(
            expected_threshold_grid_detector_checkpoint_sha256s or []
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": (
            expected_threshold_grid_outer_detector_checkpoint_sha256
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": list(
            expected_threshold_grid_episode_detector_checkpoint_sha256s or []
        ),
        "source_reference_contract_match": True,
        "score_integrity_verified": True,
        "formal_crc_eligible": False,
    }


def _audit_causal_formal_provenance(
    provenance: Mapping[str, Any] | None,
    identity_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(provenance, Mapping):
        errors.append("causal deployment provenance is missing")
    else:
        if provenance.get("allow_role_reuse") is not False:
            errors.append("allow_role_reuse must be false")
        overlap = provenance.get("global_role_overlap")
        if not isinstance(overlap, list) or overlap:
            errors.append("global_role_overlap must be an empty list")
        if provenance.get("masks_read") is not False:
            errors.append("masks_read must be false")
        if provenance.get("one_to_one_evaluation") is not True:
            errors.append("one_to_one_evaluation must be true")
        if provenance.get("score_integrity_verified") is not True:
            errors.append("score_integrity_verified must be true")
        if provenance.get("score_num_records") != provenance.get("num_images"):
            errors.append("score integrity record count must match num_images")
    if not isinstance(identity_contract, Mapping) or identity_contract.get("verified") is not True:
        errors.append("deployment block identity contract is not verified")
    elif identity_contract.get("scope") != "causal_blocks":
        errors.append("deployment identity contract is not causal")
    return {
        "verified": not errors,
        "errors": errors,
        "allow_role_reuse": (
            provenance.get("allow_role_reuse") if isinstance(provenance, Mapping) else None
        ),
        "global_role_overlap": (
            provenance.get("global_role_overlap") if isinstance(provenance, Mapping) else None
        ),
        "masks_read": (
            provenance.get("masks_read") if isinstance(provenance, Mapping) else None
        ),
        "one_to_one_evaluation": (
            provenance.get("one_to_one_evaluation")
            if isinstance(provenance, Mapping)
            else None
        ),
    }


def _json_id_rows(
    values: np.ndarray,
    expected_rows: int,
    field: str,
) -> list[list[str]]:
    encoded = np.asarray(values)
    if encoded.ndim == 0 and expected_rows == 1:
        encoded = encoded.reshape(1)
    if encoded.ndim != 1 or encoded.shape[0] != expected_rows:
        raise ValueError(f"{field} must contain one JSON ID list per statistics row")
    rows: list[list[str]] = []
    for row_index, value in enumerate(encoded):
        try:
            decoded = json.loads(_scalar_np_string(value))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{field} row {row_index} is not a valid JSON ID list"
            ) from error
        rows.append(_validate_id_row(decoded, field, row_index))
    return rows


def _image_id_rows(
    values: np.ndarray,
    expected_rows: int,
    field: str,
) -> list[list[str]]:
    image_ids = np.asarray(values)
    if image_ids.ndim == 0:
        if expected_rows != 1:
            raise ValueError(f"{field} count differs from statistics rows")
        rows = [[_scalar_np_string(image_ids)]]
    elif image_ids.ndim == 1:
        if image_ids.size == expected_rows:
            rows = [[str(value)] for value in image_ids.tolist()]
        elif expected_rows == 1:
            rows = [[str(value) for value in image_ids.tolist()]]
        else:
            raise ValueError(f"{field} count differs from statistics rows")
    elif image_ids.ndim == 2 and image_ids.shape[0] == expected_rows:
        rows = [[str(value) for value in row.tolist()] for row in image_ids]
    else:
        raise ValueError(f"{field} must align one row with each statistics row")
    return [_validate_id_row(row, field, index) for index, row in enumerate(rows)]


def _flatten_unique_id_rows(rows: list[list[str]]) -> list[str]:
    return list(dict.fromkeys(image_id for row in rows for image_id in row))


def _threshold_indices_by_image(
    evaluation_ids: list[list[str]],
    threshold_indices: list[int | None],
) -> dict[str, int | None]:
    if not evaluation_ids:
        return {}
    if len(evaluation_ids) != len(threshold_indices):
        raise ValueError("evaluation_ids and threshold_indices row counts differ")
    mapping: dict[str, int | None] = {}
    for row_ids, threshold_index in zip(evaluation_ids, threshold_indices):
        for image_id in row_ids:
            if image_id in mapping and mapping[image_id] != threshold_index:
                raise ValueError(
                    f"Duplicate image ID {image_id!r} maps to conflicting threshold indices"
                )
            mapping[image_id] = threshold_index
    return mapping


def _scalar_np_string(value: np.ndarray) -> str:
    array = np.asarray(value)
    return str(array.item() if array.ndim == 0 else array.reshape(-1)[0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--statistics-file")
    source.add_argument("--score-map-dir")
    parser.add_argument("--curve-checkpoint", required=True)
    parser.add_argument("--source-reference")
    parser.add_argument("--warmup-window", type=int, default=32)
    parser.add_argument(
        "--transductive",
        action="store_true",
        help="Use every exported target image instead of the first causal warmup window",
    )
    parser.add_argument("--pixel-budget", type=float, required=True)
    parser.add_argument("--component-budget", type=float, required=True)
    parser.add_argument(
        "--ood-z-threshold",
        type=float,
        default=DEFAULT_OOD_Z_THRESHOLD,
        help="Reject if any normalised statistic exceeds this absolute z value",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device (for example auto, cpu, cuda, or cuda:1)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    checkpoint_path = Path(args.curve_checkpoint)
    model, checkpoint = _load_checkpoint(checkpoint_path, device)
    checkpoint_representation_contract = validate_curve_checkpoint_contract(checkpoint)
    representation = str(checkpoint_representation_contract["representation"])
    if representation == LOGIT_REPRESENTATION:
        thresholds = validate_logit_threshold_grid(
            np.asarray(checkpoint["thresholds"], dtype=np.float32)
        )
    else:
        thresholds = validate_threshold_grid(np.asarray(checkpoint["thresholds"]))
    expected_grid_sha256 = str(
        checkpoint_representation_contract["threshold_grid_sha256"]
    )
    expected_grid_schema_version = str(
        checkpoint_representation_contract["threshold_grid_schema_version"]
    )
    expected_feature_schema_sha256 = (
        str(checkpoint["feature_schema_sha256"])
        if representation == LOGIT_REPRESENTATION
        else checkpoint.get("feature_schema_sha256")
    )
    expected_grid_manifest_sha256 = (
        str(checkpoint["threshold_grid_manifest_sha256"])
        if representation == LOGIT_REPRESENTATION
        else None
    )
    expected_grid_detector_protocol = (
        str(checkpoint_representation_contract["threshold_grid_detector_protocol"])
        if representation == LOGIT_REPRESENTATION
        else None
    )
    expected_grid_detector_checkpoint_sha256s = (
        list(
            checkpoint_representation_contract[
                "threshold_grid_detector_checkpoint_sha256s"
            ]
        )
        if representation == LOGIT_REPRESENTATION
        else []
    )
    expected_grid_outer_detector_checkpoint_sha256 = (
        str(
            checkpoint_representation_contract[
                "threshold_grid_outer_detector_checkpoint_sha256"
            ]
        )
        if representation == LOGIT_REPRESENTATION
        else None
    )
    expected_grid_episode_detector_checkpoint_sha256s = (
        list(
            checkpoint_representation_contract[
                "threshold_grid_episode_detector_checkpoint_sha256s"
            ]
        )
        if representation == LOGIT_REPRESENTATION
        else []
    )
    window_ids: list[str] = []
    protocol: dict[str, Any] | None = None
    protocol_fingerprint: str | None = None
    score_map_provenance: dict[str, Any] | None = None
    statistics_artifact: dict[str, Any] | None = None
    archive_identity_contract: dict[str, Any] | None = None
    evaluation_id_rows: list[list[str]] = []
    adaptation_id_rows: list[list[str]] = []
    if args.statistics_file:
        statistics_path = Path(args.statistics_file)
        (
            statistics,
            statistics_names,
            statistics_schema_version,
            evaluation_id_rows,
            adaptation_id_rows,
            protocol,
            protocol_fingerprint,
            archive_evidence,
        ) = _statistics_from_archive(statistics_path)
        score_map_provenance = archive_evidence["provenance"]
        archive_identity_contract = archive_evidence["identity_contract"]
        statistics_artifact = {
            "source_type": "deployment_statistics_archive",
            "path": str(statistics_path.resolve()),
            "sha256": hashlib.sha256(statistics_path.read_bytes()).hexdigest(),
            "deployment_statistics_schema_version": archive_evidence[
                "deployment_statistics_schema_version"
            ],
            "statistics_schema_version": statistics_schema_version,
            "threshold_grid_sha256": archive_evidence["threshold_grid_sha256"],
            "representation": archive_evidence["representation"],
            "threshold_grid_schema_version": archive_evidence[
                "threshold_grid_schema_version"
            ],
            "feature_schema_sha256": archive_evidence[
                "feature_schema_sha256"
            ],
            "threshold_grid_manifest_sha256": archive_evidence[
                "threshold_grid_manifest_sha256"
            ],
            "threshold_grid_detector_protocol": archive_evidence[
                "threshold_grid_detector_protocol"
            ],
            "threshold_grid_detector_checkpoint_sha256s": archive_evidence[
                "threshold_grid_detector_checkpoint_sha256s"
            ],
            "threshold_grid_outer_detector_checkpoint_sha256": archive_evidence[
                "threshold_grid_outer_detector_checkpoint_sha256"
            ],
            "threshold_grid_episode_detector_checkpoint_sha256s": archive_evidence[
                "threshold_grid_episode_detector_checkpoint_sha256s"
            ],
            "provenance": score_map_provenance,
            "identity_contract": archive_identity_contract,
        }
        window_ids = _flatten_unique_id_rows(adaptation_id_rows)
    else:
        score_map_dir = Path(args.score_map_dir)
        (
            statistics,
            statistics_names,
            window_ids,
            statistics_schema_version,
        ) = _statistics_from_score_dir(
            score_map_dir,
            args.source_reference,
            None if args.transductive else args.warmup_window,
            representation=representation,
        )
        (
            protocol,
            protocol_fingerprint,
            score_map_provenance,
        ) = _score_map_protocol_provenance(
            score_map_dir,
            thresholds,
            representation=representation,
            threshold_grid_detector_protocol=expected_grid_detector_protocol,
            threshold_grid_detector_checkpoint_sha256s=(
                expected_grid_detector_checkpoint_sha256s
            ),
            threshold_grid_outer_detector_checkpoint_sha256=(
                expected_grid_outer_detector_checkpoint_sha256
            ),
            threshold_grid_episode_detector_checkpoint_sha256s=(
                expected_grid_episode_detector_checkpoint_sha256s
            ),
        )
        statistics = np.asarray(statistics, dtype=np.float32)[None, :]
        reference = (
            load_source_reference(args.source_reference)
            if args.source_reference
            else None
        )
        effective_adaptation_window = (
            len(window_ids) if args.transductive else int(args.warmup_window)
        )
        score_map_provenance.update(
            {
                "adaptation_window": effective_adaptation_window,
                "evaluation_window": 1,
                "stride": effective_adaptation_window + 1,
                "representation": representation,
                "threshold_grid_schema_version": expected_grid_schema_version,
                "threshold_grid_sha256": expected_grid_sha256,
                "feature_schema_sha256": expected_feature_schema_sha256,
                "threshold_grid_manifest_sha256": expected_grid_manifest_sha256,
                "threshold_grid_detector_protocol": (
                    expected_grid_detector_protocol
                ),
                "threshold_grid_detector_checkpoint_sha256s": (
                    expected_grid_detector_checkpoint_sha256s
                ),
                "threshold_grid_outer_detector_checkpoint_sha256": (
                    expected_grid_outer_detector_checkpoint_sha256
                ),
                "threshold_grid_episode_detector_checkpoint_sha256s": (
                    expected_grid_episode_detector_checkpoint_sha256s
                ),
                "source_reference_sha256": (
                    hashlib.sha256(Path(args.source_reference).read_bytes()).hexdigest()
                    if args.source_reference
                    else None
                ),
                "source_reference_domain_names": (
                    list(reference.domain_names) if reference is not None else []
                ),
                "source_reference_statistics_names_sha256": (
                    statistics_names_sha256(reference.statistics_names)
                    if reference is not None
                    else None
                ),
                "transductive": bool(args.transductive),
            }
        )

    if statistics_artifact is not None:
        if statistics_artifact.get("representation") != representation:
            raise ValueError(
                "Deployment statistics and curve checkpoint representations differ"
            )
        if representation == LOGIT_REPRESENTATION:
            if (
                statistics_artifact.get("threshold_grid_schema_version")
                != expected_grid_schema_version
            ):
                raise ValueError(
                    "Deployment statistics and curve checkpoint grid schemas differ"
                )
            if (
                statistics_artifact.get("feature_schema_sha256")
                != expected_feature_schema_sha256
            ):
                raise ValueError(
                    "Deployment statistics and curve checkpoint feature schemas differ"
                )
            if (
                statistics_artifact.get("threshold_grid_manifest_sha256")
                != expected_grid_manifest_sha256
            ):
                raise ValueError(
                    "Deployment statistics and curve checkpoint grid manifests differ"
                )
            if (
                statistics_artifact.get("threshold_grid_detector_protocol")
                != expected_grid_detector_protocol
            ):
                raise ValueError(
                    "Deployment statistics and checkpoint grid detector protocols differ"
                )
            if list(
                statistics_artifact.get(
                    "threshold_grid_detector_checkpoint_sha256s"
                )
                or []
            ) != expected_grid_detector_checkpoint_sha256s:
                raise ValueError(
                    "Deployment statistics and checkpoint grid detector hashes differ"
                )
            if statistics_artifact.get(
                "threshold_grid_outer_detector_checkpoint_sha256"
            ) != expected_grid_outer_detector_checkpoint_sha256:
                raise ValueError(
                    "Deployment statistics and checkpoint outer detector hashes differ"
                )
            if list(
                statistics_artifact.get(
                    "threshold_grid_episode_detector_checkpoint_sha256s"
                )
                or []
            ) != expected_grid_episode_detector_checkpoint_sha256s:
                raise ValueError(
                    "Deployment statistics and checkpoint episode detector hashes differ"
                )
        recorded_grid_sha256 = statistics_artifact.get("threshold_grid_sha256")
        if (
            recorded_grid_sha256 is not None
            and recorded_grid_sha256 != expected_grid_sha256
        ):
            raise ValueError(
                "Deployment statistics and curve checkpoint threshold grids differ"
            )
    if protocol is not None and protocol.get("threshold_grid_sha256") != expected_grid_sha256:
        raise ValueError("Deployment protocol and curve checkpoint threshold grids differ")
    if representation == LOGIT_REPRESENTATION and protocol is not None:
        if protocol.get("representation") != representation:
            raise ValueError("Deployment protocol representation mismatch")
        if protocol.get("threshold_grid_schema_version") != expected_grid_schema_version:
            raise ValueError("Deployment protocol raw-logit grid schema mismatch")
        if protocol.get("threshold_grid_detector_protocol") != expected_grid_detector_protocol:
            raise ValueError("Deployment protocol grid detector protocol mismatch")
        if list(protocol.get("threshold_grid_detector_checkpoint_sha256s") or []) != (
            expected_grid_detector_checkpoint_sha256s
        ):
            raise ValueError("Deployment protocol grid detector hashes mismatch")
        if protocol.get("threshold_grid_outer_detector_checkpoint_sha256") != (
            expected_grid_outer_detector_checkpoint_sha256
        ):
            raise ValueError("Deployment protocol outer detector hash mismatch")
        if list(
            protocol.get("threshold_grid_episode_detector_checkpoint_sha256s")
            or []
        ) != expected_grid_episode_detector_checkpoint_sha256s:
            raise ValueError("Deployment protocol episode detector hashes mismatch")
        if protocol.get("detector_weight_sha256") != (
            expected_grid_outer_detector_checkpoint_sha256
        ):
            raise ValueError(
                "Deployment protocol detector weight is not the outer-final detector"
            )

    target_dataset = (
        str(protocol.get("target_dataset", ""))
        if isinstance(protocol, Mapping)
        else ""
    )
    checkpoint_deployment_audit = audit_checkpoint_deployment_contract(
        checkpoint_path,
        deployment_provenance=score_map_provenance,
        target_dataset=target_dataset,
        expected_threshold_grid_sha256=expected_grid_sha256,
    )
    deployment_statistics_schema = (
        statistics_artifact.get("deployment_statistics_schema_version")
        if statistics_artifact is not None
        else None
    )
    static_cross_fit_statistics = bool(
        deployment_statistics_schema
        in {
            STATIC_CROSS_FIT_STATISTICS_SCHEMA_VERSION,
            LOGIT_STATIC_CROSS_FIT_STATISTICS_SCHEMA_VERSION,
        }
    )
    causal_statistics_artifact = bool(
        statistics_artifact is not None
        and deployment_statistics_schema
        in {
            DEPLOYMENT_STATISTICS_SCHEMA_VERSION,
            LOGIT_DEPLOYMENT_STATISTICS_SCHEMA_VERSION,
        }
    )
    causal_provenance_audit = _audit_causal_formal_provenance(
        score_map_provenance,
        archive_identity_contract,
    )
    formal_statistics_artifact = bool(
        causal_statistics_artifact
        and causal_provenance_audit["verified"]
        and checkpoint_deployment_audit["verified"]
    )
    if (
        causal_statistics_artifact
        and causal_provenance_audit["verified"]
        and not checkpoint_deployment_audit["verified"]
    ):
        raise ValueError(
            "Curve checkpoint/deployment contract failed: "
            + json.dumps(checkpoint_deployment_audit["errors"], sort_keys=True)
        )
    static_checkpoint_compatibility: dict[str, Any] | None = None
    if static_cross_fit_statistics:
        static_checkpoint_compatibility = _validate_static_checkpoint_compatibility(
            checkpoint,
            provenance=score_map_provenance,
            protocol=protocol,
            expected_threshold_grid_sha256=expected_grid_sha256,
            expected_representation=(
                representation
                if representation == LOGIT_REPRESENTATION
                else None
            ),
            expected_threshold_grid_schema_version=(
                expected_grid_schema_version
                if representation == LOGIT_REPRESENTATION
                else None
            ),
            expected_feature_schema_sha256=(
                str(expected_feature_schema_sha256)
                if representation == LOGIT_REPRESENTATION
                else None
            ),
            expected_threshold_grid_manifest_sha256=(
                expected_grid_manifest_sha256
                if representation == LOGIT_REPRESENTATION
                else None
            ),
            expected_threshold_grid_detector_protocol=(
                expected_grid_detector_protocol
                if representation == LOGIT_REPRESENTATION
                else None
            ),
            expected_threshold_grid_detector_checkpoint_sha256s=(
                expected_grid_detector_checkpoint_sha256s
                if representation == LOGIT_REPRESENTATION
                else None
            ),
            expected_threshold_grid_outer_detector_checkpoint_sha256=(
                expected_grid_outer_detector_checkpoint_sha256
                if representation == LOGIT_REPRESENTATION
                else None
            ),
            expected_threshold_grid_episode_detector_checkpoint_sha256s=(
                expected_grid_episode_detector_checkpoint_sha256s
                if representation == LOGIT_REPRESENTATION
                else None
            ),
        )

    expected_statistics_schema_version = (
        LOGIT_STATISTICS_SCHEMA_VERSION
        if representation == LOGIT_REPRESENTATION
        else STATISTICS_SCHEMA_VERSION
    )
    if statistics_schema_version != expected_statistics_schema_version:
        raise ValueError("Deployment statistics use an incompatible schema version")
    statistics = np.asarray(statistics, dtype=np.float32)
    if statistics.ndim != 2 or min(statistics.shape) <= 0:
        raise ValueError("Deployment statistics must have shape [N,D] with N,D > 0")
    checkpoint_names = validate_statistics_names(checkpoint["statistics_names"])
    actual_names = validate_statistics_names(
        statistics_names, expected_dim=statistics.shape[1]
    )
    if actual_names != checkpoint_names:
        raise ValueError(
            "Deployment statistics_names do not exactly match checkpoint feature order"
        )
    actual_feature_schema_sha256 = feature_schema_sha256(
        statistics_schema_version, statistics_names=actual_names
    )
    if representation == LOGIT_REPRESENTATION and (
        actual_feature_schema_sha256 != expected_feature_schema_sha256
    ):
        raise ValueError(
            "Deployment statistics feature schema does not match the checkpoint"
        )
    mean = np.asarray(checkpoint["statistics_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["statistics_std"], dtype=np.float32)
    if mean.shape != (statistics.shape[1],) or std.shape != mean.shape:
        raise ValueError(
            f"Expected {mean.size} statistics per row, got {statistics.shape[1]}"
        )
    if (
        not np.isfinite(statistics).all()
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std < 0.0)
    ):
        raise ValueError("Statistics normalisation contains invalid values")
    if (
        not np.isfinite(args.pixel_budget)
        or not np.isfinite(args.component_budget)
        or args.pixel_budget <= 0.0
        or args.component_budget <= 0.0
    ):
        raise ValueError("Budgets must be finite and positive")
    normalised = (statistics - mean[None, :]) / np.maximum(std[None, :], 1e-6)
    ood_audits: list[dict[str, Any]] = []
    pixel_curves: list[np.ndarray | None] = []
    component_curves: list[np.ndarray | None] = []
    selected_thresholds: list[float | None] = []
    selected_logit_thresholds: list[float | str | None] = []
    selected_probability_thresholds: list[float | None] = []
    threshold_indices: list[int | None] = []
    rejects: list[bool] = []
    reject_reasons: list[str | None] = []
    for row_index, row in enumerate(normalised):
        row_ood_audit = assess_ood_statistics(
            row, actual_names, z_threshold=args.ood_z_threshold
        )
        ood_audits.append(row_ood_audit)
        pixel: np.ndarray | None
        component: np.ndarray | None
        if row_ood_audit["is_ood"]:
            pixel = None
            component = None
            threshold = (
                EMPTY_ACTION_THRESHOLD
                if representation == LOGIT_REPRESENTATION
                else None
            )
            reject, index = True, None
            reject_reason = "ood_statistics"
        else:
            with torch.inference_mode():
                prediction = model(
                    torch.from_numpy(normalised[row_index : row_index + 1]).to(device)
                )
            pixel = prediction["pixel_log_risk"][0].cpu().numpy()
            component = prediction["component_log_risk"][0].cpu().numpy()
            threshold, reject, index = select_dual_budget_threshold(
                thresholds,
                pixel,
                component,
                args.pixel_budget,
                args.component_budget,
                representation=representation,
            )
            reject_reason = "no_predicted_feasible_threshold" if reject else None
        pixel_curves.append(pixel)
        component_curves.append(component)
        selected_thresholds.append(None if reject else float(threshold))
        if representation == LOGIT_REPRESENTATION:
            selected_logit_thresholds.append(
                "+inf" if reject else float(threshold)
            )
            selected_probability_thresholds.append(
                None if reject else _sigmoid_display(float(threshold))
            )
        else:
            selected_logit_thresholds.append(None)
            selected_probability_thresholds.append(
                None if reject else float(threshold)
            )
        threshold_indices.append(index)
        rejects.append(reject)
        reject_reasons.append(reject_reason)

    threshold_indices_by_image = _threshold_indices_by_image(
        evaluation_id_rows, threshold_indices
    )
    if archive_identity_contract and archive_identity_contract.get("verified") is True:
        expected_action_ids = {
            image_id for row in evaluation_id_rows for image_id in row
        }
        if set(threshold_indices_by_image) != expected_action_ids:
            raise ValueError(
                "Threshold mapping does not completely cover deployment evaluation IDs"
            )
        if len(threshold_indices_by_image) != sum(map(len, evaluation_id_rows)):
            raise ValueError(
                "Threshold mapping is not one complete action per evaluation image"
            )
    num_windows = int(statistics.shape[0])
    reject_rate = float(np.mean(rejects))
    any_reject = any(rejects)
    if num_windows == 1:
        scalar_threshold = selected_thresholds[0]
        scalar_logit_threshold = selected_logit_thresholds[0]
        scalar_probability_threshold = selected_probability_thresholds[0]
        scalar_index = threshold_indices[0]
        scalar_reject_reason = reject_reasons[0]
        ood_audit: dict[str, Any] = ood_audits[0]
        predicted_pixel: Any = (
            pixel_curves[0].tolist() if pixel_curves[0] is not None else None
        )
        predicted_component: Any = (
            component_curves[0].tolist() if component_curves[0] is not None else None
        )
    else:
        scalar_threshold = None
        scalar_logit_threshold = None
        scalar_probability_threshold = None
        scalar_index = None
        scalar_reject_reason = "one_or_more_windows_rejected" if any_reject else None
        ood_audit = {
            "is_ood": any(audit["is_ood"] for audit in ood_audits),
            "z_threshold": float(args.ood_z_threshold),
            "num_ood_windows": sum(bool(audit["is_ood"]) for audit in ood_audits),
            "num_windows": num_windows,
            "max_abs_z": max(float(audit["max_abs_z"]) for audit in ood_audits),
        }
        predicted_pixel = [
            curve.tolist() if curve is not None else None for curve in pixel_curves
        ]
        predicted_component = [
            curve.tolist() if curve is not None else None
            for curve in component_curves
        ]
    row_results = [
        {
            "row_index": row_index,
            "evaluation_ids": (
                evaluation_id_rows[row_index] if evaluation_id_rows else []
            ),
            "threshold": selected_thresholds[row_index],
            "selected_logit_threshold": selected_logit_thresholds[row_index],
            "selected_probability_threshold": selected_probability_thresholds[
                row_index
            ],
            "threshold_index": threshold_indices[row_index],
            "reject": rejects[row_index],
            "reject_reason": reject_reasons[row_index],
            "ood_audit": ood_audits[row_index],
        }
        for row_index in range(num_windows)
    ]
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if static_cross_fit_statistics:
        statistics_computed_from = "static_cross_fit_complement_folds"
        threshold_mapping_rule = "one_complement_fold_prediction_to_held_out_fold_ids"
    elif args.statistics_file:
        statistics_computed_from = "causal_adaptation_blocks_A"
        threshold_mapping_rule = "one_A_block_prediction_to_its_future_E_identity"
    elif args.transductive:
        statistics_computed_from = "transductive_score_maps"
        threshold_mapping_rule = "one_global_transductive_prediction"
    else:
        statistics_computed_from = "causal_warmup_score_maps"
        threshold_mapping_rule = "one_global_warmup_prediction"
    selection_data_contract = {
        "schema_version": SELECTION_DATA_CONTRACT_SCHEMA_VERSION,
        "masks_read": False,
        "statistics_computed_from": statistics_computed_from,
        "evaluation_labels_or_masks_used": False,
        "threshold_mapping_rule": threshold_mapping_rule,
        "checkpoint_training_contract_verified": bool(
            checkpoint_deployment_audit["verified"]
        ),
        "formal_crc_eligible": bool(formal_statistics_artifact),
        "deployment_identity_contract_verified": bool(
            archive_identity_contract
            and archive_identity_contract.get("verified") is True
        ),
        "static_checkpoint_compatibility_verified": bool(
            static_checkpoint_compatibility
            and static_checkpoint_compatibility.get("verified") is True
        ),
        "representation": representation,
        "threshold_grid_schema_version": expected_grid_schema_version,
        "threshold_grid_sha256": expected_grid_sha256,
        "feature_schema_sha256": expected_feature_schema_sha256,
        "threshold_grid_manifest_sha256": expected_grid_manifest_sha256,
        "threshold_grid_detector_protocol": expected_grid_detector_protocol,
        "threshold_grid_detector_checkpoint_sha256s": (
            expected_grid_detector_checkpoint_sha256s
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": (
            expected_grid_outer_detector_checkpoint_sha256
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": (
            expected_grid_episode_detector_checkpoint_sha256s
        ),
        "prediction_rule": (
            LOGIT_PREDICTION_RULE
            if representation == LOGIT_REPRESENTATION
            else "prediction = (sigmoid_probability >= threshold)"
        ),
        "empty_action": (
            empty_action_contract()
            if representation == LOGIT_REPRESENTATION
            else {"threshold": 1.0, "threshold_index": None}
        ),
    }
    result = {
        "schema_version": ZERO_RESULT_SCHEMA_VERSION,
        "mode": "zero_label_empirical_adaptation",
        "threshold": scalar_threshold,
        "selected_logit_threshold": scalar_logit_threshold,
        "selected_probability_threshold": scalar_probability_threshold,
        "threshold_index": scalar_index,
        "reject": any_reject,
        "reject_reason": scalar_reject_reason,
        "num_windows": num_windows,
        "selected_thresholds": selected_thresholds,
        "selected_logit_thresholds": selected_logit_thresholds,
        "selected_probability_thresholds": selected_probability_thresholds,
        "threshold_indices": threshold_indices,
        # ``null`` preserves legacy scalar-index consumption when no per-image
        # identity metadata is available (notably direct score-map warm-up).
        "threshold_indices_by_image": threshold_indices_by_image or None,
        "rejects": rejects,
        "reject_reasons": reject_reasons,
        "reject_rate": reject_rate,
        "ood_audits": ood_audits,
        "row_results": row_results,
        "pixel_budget": args.pixel_budget,
        "component_budget": args.component_budget,
        "curve_checkpoint": str(checkpoint_path.resolve()),
        "curve_checkpoint_sha256": digest,
        "curve_checkpoint_deployment_audit": checkpoint_deployment_audit,
        "static_checkpoint_compatibility_audit": static_checkpoint_compatibility,
        "causal_formal_provenance_audit": causal_provenance_audit,
        "source_reference": args.source_reference,
        "selection_data_contract": selection_data_contract,
        "masks_read": False,
        "statistics_file": (
            statistics_artifact["path"] if statistics_artifact is not None else None
        ),
        "statistics_file_sha256": (
            statistics_artifact["sha256"] if statistics_artifact is not None else None
        ),
        "deployment_statistics_schema_version": (
            statistics_artifact["deployment_statistics_schema_version"]
            if statistics_artifact is not None
            else None
        ),
        "adaptation_window": (
            score_map_provenance.get("adaptation_window")
            if args.statistics_file and score_map_provenance is not None
            else None
        ),
        "evaluation_window": (
            score_map_provenance.get("evaluation_window")
            if args.statistics_file and score_map_provenance is not None
            else None
        ),
        "stride": (
            score_map_provenance.get("stride")
            if args.statistics_file and score_map_provenance is not None
            else None
        ),
        "statistics_artifact": statistics_artifact,
        "protocol": protocol,
        "protocol_fingerprint": protocol_fingerprint,
        "score_map_provenance": score_map_provenance,
        "statistics_schema_version": statistics_schema_version,
        "statistics_names_sha256": statistics_names_sha256(actual_names),
        "feature_schema_sha256": actual_feature_schema_sha256,
        "representation": representation,
        "threshold_grid_schema_version": expected_grid_schema_version,
        "threshold_grid_sha256": expected_grid_sha256,
        "threshold_grid_manifest_sha256": expected_grid_manifest_sha256,
        "threshold_grid_detector_protocol": expected_grid_detector_protocol,
        "threshold_grid_detector_checkpoint_sha256s": (
            expected_grid_detector_checkpoint_sha256s
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": (
            expected_grid_outer_detector_checkpoint_sha256
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": (
            expected_grid_episode_detector_checkpoint_sha256s
        ),
        "prediction_rule": (
            LOGIT_PREDICTION_RULE
            if representation == LOGIT_REPRESENTATION
            else "prediction = (sigmoid_probability >= threshold)"
        ),
        "empty_action": (
            empty_action_contract()
            if representation == LOGIT_REPRESENTATION
            else {"threshold": 1.0, "threshold_index": None}
        ),
        "ood_audit": ood_audit,
        "adaptation_protocol": (
            "static_cross_fit"
            if static_cross_fit_statistics
            else (
                "external_causal_statistics"
                if args.statistics_file
                else ("transductive" if args.transductive else "causal_warmup")
            )
        ),
        "warmup_window": len(window_ids) if window_ids else None,
        "window_ids": window_ids,
        "adaptation_image_ids": window_ids,
        "adaptation_ids": adaptation_id_rows,
        "evaluation_ids": evaluation_id_rows,
        "thresholds": thresholds.tolist(),
        "predicted_pixel_log_risk": predicted_pixel,
        "predicted_component_log_risk": predicted_component,
        "guarantee": "none; upper-quantile risk prediction is empirical",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("threshold", "threshold_index", "reject", "reject_rate")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
