"""Build mask-free statistics for sample-adaptive zero-label selection.

Two explicit protocols are supported.  ``causal`` interprets manifest order as
disjoint ``A -> E`` blocks.  ``static-cross-fit`` partitions an unordered test
set into deterministic folds, computes one label-free statistic row from the
complement of each fold, and maps that row's action to every image in the held
out fold.  Cross-fit output is empirical/transductive and is deliberately not
eligible for the causal conformal certificate.  Ground-truth masks are never
read by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from certification.build_calibration_losses import score_map_protocol
from evaluation.artifact_integrity import (
    PROBABILITY_DTYPE,
    RAW_LOGIT_DTYPE,
    RAW_LOGIT_SCORE_REPRESENTATION,
    SCORE_MANIFEST_SCHEMA_VERSION,
    SCORE_RECORD_INTEGRITY_SCHEMA,
    ordered_ids_sha256,
    verify_score_map_directory,
)

from .build_curve_episodes import build_causal_windows, load_score_sample, score_files
from .domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    STATISTICS_SCHEMA_VERSION,
    extract_logit_window_statistics,
    extract_window_statistics,
    feature_schema_sha256,
    load_source_reference,
    statistics_names_sha256,
)
from .threshold_grid import (
    load_threshold_grid,
    threshold_grid_sha256,
    threshold_grid_version,
)
from .representation import (
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


DEPLOYMENT_STATISTICS_SCHEMA_VERSION = "rc-v2-deployment-statistics-v2-causal-blocks"
STATIC_CROSS_FIT_STATISTICS_SCHEMA_VERSION = (
    "rc-v2-deployment-statistics-v2-static-cross-fit-fixed-adaptation"
)
LOGIT_DEPLOYMENT_STATISTICS_SCHEMA_VERSION = (
    "rc-v4-deployment-statistics-v1-causal-blocks-raw-logit"
)
LOGIT_STATIC_CROSS_FIT_STATISTICS_SCHEMA_VERSION = (
    "rc-v4-deployment-statistics-v1-static-cross-fit-raw-logit"
)

RAW_LOGIT_REPRESENTATION = LOGIT_REPRESENTATION
LOGIT_THRESHOLD_GRID_SCHEMA_VERSION = LOGIT_GRID_SCHEMA_VERSION


def _normalise_representation(value: str) -> str:
    representation = str(value).strip()
    aliases = {
        "probability": PROBABILITY_REPRESENTATION,
        "sigmoid_probability": PROBABILITY_REPRESENTATION,
        PROBABILITY_REPRESENTATION: PROBABILITY_REPRESENTATION,
        RAW_LOGIT_REPRESENTATION: RAW_LOGIT_REPRESENTATION,
    }
    try:
        return aliases[representation]
    except KeyError as error:
        raise ValueError(f"Unsupported score representation: {value!r}") from error


def _validate_grid_for_representation(
    thresholds: np.ndarray,
    representation: str,
) -> np.ndarray:
    representation = _normalise_representation(representation)
    if representation == PROBABILITY_REPRESENTATION:
        # ``threshold_grid_sha256`` performs the established v2 validation.
        threshold_grid_sha256(thresholds)
        return np.asarray(thresholds, dtype=np.float32).reshape(-1)
    return validate_logit_threshold_grid(np.asarray(thresholds))


def _grid_semantic_sha256(thresholds: np.ndarray, representation: str) -> str:
    representation = _normalise_representation(representation)
    values = _validate_grid_for_representation(thresholds, representation)
    if representation == PROBABILITY_REPRESENTATION:
        # Preserve every existing v2 artifact byte-for-byte at the hash layer.
        return threshold_grid_sha256(values)
    return logit_threshold_grid_sha256(values)


def _grid_schema_version(
    representation: str,
    thresholds: np.ndarray,
) -> str:
    return (
        LOGIT_THRESHOLD_GRID_SCHEMA_VERSION
        if _normalise_representation(representation) == RAW_LOGIT_REPRESENTATION
        else threshold_grid_version(thresholds)
    )


def _validate_grid_manifest_sha256(
    value: str | None,
    *,
    representation: str,
) -> str | None:
    if representation == PROBABILITY_REPRESENTATION:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            "Raw-logit deployment requires threshold_grid_manifest_sha256"
        )
    return value


def _validate_grid_detector_contract(
    protocol: str | None,
    checkpoint_sha256s: Sequence[str] | None,
    outer_checkpoint_sha256: str | None,
    episode_checkpoint_sha256s: Sequence[str] | None,
    *,
    representation: str,
) -> tuple[str | None, tuple[str, ...], str | None, tuple[str, ...]]:
    if representation == PROBABILITY_REPRESENTATION:
        return None, (), None, ()
    if protocol != GRID_DETECTOR_PROTOCOL:
        raise ValueError(
            f"Raw-logit deployment requires grid detector protocol "
            f"{GRID_DETECTOR_PROTOCOL!r}"
        )
    if not isinstance(checkpoint_sha256s, (list, tuple)) or not checkpoint_sha256s:
        raise ValueError("Raw-logit deployment requires detector checkpoint hashes")
    values = tuple(str(value) for value in checkpoint_sha256s)
    if len(values) != len(set(values)):
        raise ValueError("Grid detector checkpoint hashes must be distinct")
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in values
    ):
        raise ValueError("Grid detector checkpoint hashes must be SHA-256 digests")
    outer = str(outer_checkpoint_sha256 or "")
    if outer not in values:
        raise ValueError(
            "Raw-logit deployment requires the outer-final detector checkpoint"
        )
    if not isinstance(episode_checkpoint_sha256s, (list, tuple)):
        raise ValueError(
            "Raw-logit deployment requires inner pseudo-target detector checkpoints"
        )
    episode_values = tuple(str(value) for value in episode_checkpoint_sha256s)
    if (
        not episode_values
        or len(episode_values) != len(set(episode_values))
        or outer in episode_values
        or set(values) != set(episode_values).union({outer})
    ):
        raise ValueError(
            "Grid outer-final and inner pseudo-target detector roles are invalid"
        )
    return protocol, values, outer, episode_values


def _require_raw_logit_manifest_contract(
    manifest: dict[str, object] | None,
    integrity: dict[str, object],
) -> dict[str, object]:
    """Fail closed before any raw-logit deployment statistic is extracted."""

    if manifest is None:
        raise ValueError("Raw-logit deployment requires manifest.json")
    if manifest.get("schema_version") != SCORE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Raw-logit deployment requires score manifest schema version 3")
    if manifest.get("record_integrity_schema") != SCORE_RECORD_INTEGRITY_SCHEMA:
        raise ValueError("Raw-logit deployment requires the v3 record-integrity schema")
    if integrity.get("verified") is not True:
        raise ValueError("Raw-logit deployment requires verified score-map integrity")
    expected: dict[str, object] = {
        "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
        "probability_dtype": PROBABILITY_DTYPE,
        "logit_dtype": RAW_LOGIT_DTYPE,
        "probability_transform": "sigmoid",
        "probability_clipping": "none",
        "inference_autocast_enabled": False,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"Raw-logit deployment requires manifest {field}={value!r}"
            )
    return expected


def _load_label_free_sample(
    path: Path,
    *,
    representation: str,
) -> tuple[str, np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"image_id", "gray"}
        score_field = (
            "logit"
            if _normalise_representation(representation) == RAW_LOGIT_REPRESENTATION
            else "prob"
        )
        required.add(score_field)
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(
                f"Deployment score record {path.name} lacks: {', '.join(missing)}"
            )
        image_id = str(np.asarray(payload["image_id"]).item())
        score = np.asarray(payload[score_field], dtype=np.float32)
        gray = np.asarray(payload["gray"], dtype=np.float32)
    if not image_id.strip():
        raise ValueError(f"Deployment score record has an empty image_id: {path}")
    if score.ndim != 2 or score.size == 0 or not np.isfinite(score).all():
        raise ValueError(f"Deployment {score_field} map must be finite 2-D: {path}")
    if gray.shape != score.shape or not np.isfinite(gray).all():
        raise ValueError(f"Deployment gray/score shape or finiteness mismatch: {path}")
    return image_id, score, gray


def _raw_logit_protocol(
    root: Path,
    manifest: dict[str, object],
    thresholds: np.ndarray,
    *,
    matching_rule: str,
    centroid_distance: float,
    connectivity: int,
    min_component_area: int,
    grid_detector_protocol: str,
    grid_detector_checkpoint_sha256s: Sequence[str],
    grid_outer_detector_checkpoint_sha256: str,
    grid_episode_detector_checkpoint_sha256s: Sequence[str],
) -> tuple[dict[str, object], str]:
    detector_sha = str(manifest.get("weight_sha256", "")).strip().lower()
    if len(detector_sha) != 64 or any(c not in "0123456789abcdef" for c in detector_sha):
        raise ValueError("Raw-logit manifest requires a valid weight_sha256")
    if detector_sha != grid_outer_detector_checkpoint_sha256:
        raise ValueError(
            "Raw-logit deployment score maps must come from the outer-final detector"
        )
    target_dataset = str(manifest.get("target_dataset", "")).strip()
    if not target_dataset:
        raise ValueError("Raw-logit manifest requires target_dataset")
    sources = manifest.get("source_datasets")
    if not isinstance(sources, list) or any(not str(value).strip() for value in sources):
        raise ValueError("Raw-logit manifest requires non-empty source_datasets")
    if target_dataset.casefold() in {str(value).strip().casefold() for value in sources}:
        raise ValueError("Raw-logit detector source_datasets contains target_dataset")
    grid = _validate_grid_for_representation(thresholds, RAW_LOGIT_REPRESENTATION)
    semantic_hash = _grid_semantic_sha256(grid, RAW_LOGIT_REPRESENTATION)
    protocol: dict[str, object] = {
        "schema_version": "rc-v4-score-map-protocol-v1-raw-logit",
        "detector_weight_sha256": detector_sha,
        "score_type": manifest.get("score_type"),
        "representation": RAW_LOGIT_REPRESENTATION,
        "score_representation": manifest.get("score_representation"),
        "warm_flag": bool(manifest.get("warm_flag")),
        "spatial_mode": manifest.get("spatial_mode"),
        "pad_multiple": manifest.get("pad_multiple"),
        "base_hw": manifest.get("base_hw") if manifest.get("spatial_mode") == "resize" else None,
        "target_dataset": target_dataset,
        "source_datasets": sorted([str(value).strip() for value in sources], key=str.casefold),
        "threshold_grid_schema_version": LOGIT_THRESHOLD_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": semantic_hash,
        "threshold_grid_detector_protocol": grid_detector_protocol,
        "threshold_grid_detector_checkpoint_sha256s": list(
            grid_detector_checkpoint_sha256s
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": (
            grid_outer_detector_checkpoint_sha256
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": list(
            grid_episode_detector_checkpoint_sha256s
        ),
        "prediction_rule": LOGIT_PREDICTION_RULE,
        "empty_action": empty_action_contract(),
        "matching_rule": matching_rule,
        "centroid_distance": float(centroid_distance),
        "connectivity": int(connectivity),
        "min_component_area": int(min_component_area),
        "component_monotone_transform": "per_image_suffix_max",
    }
    fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return protocol, fingerprint


def stable_block_id(
    block_index: int,
    adaptation_ids: Sequence[str],
    evaluation_ids: Sequence[str],
    *,
    schema_version: str = DEPLOYMENT_STATISTICS_SCHEMA_VERSION,
) -> str:
    """Return a deterministic ID for one ordered causal ``A -> E`` block."""

    payload = json.dumps(
        {
            "schema": schema_version,
            "block_index": int(block_index),
            "adaptation_ids": list(map(str, adaptation_ids)),
            "evaluation_ids": list(map(str, evaluation_ids)),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"block-{block_index:06d}-{hashlib.sha256(payload).hexdigest()[:16]}"


def stable_cross_fit_fold_id(
    fold_index: int,
    adaptation_ids: Sequence[str],
    evaluation_ids: Sequence[str],
    *,
    schema_version: str = STATIC_CROSS_FIT_STATISTICS_SCHEMA_VERSION,
) -> str:
    """Return a deterministic identity for one static cross-fit fold."""

    payload = json.dumps(
        {
            "schema": schema_version,
            "fold_index": int(fold_index),
            "adaptation_ids": list(map(str, adaptation_ids)),
            "evaluation_ids": list(map(str, evaluation_ids)),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"fold-{fold_index:03d}-{hashlib.sha256(payload).hexdigest()[:16]}"


@dataclass(frozen=True)
class _DeploymentSample:
    image_id: str
    score: np.ndarray
    gray: np.ndarray | None


def _deployment_sample(path: Path, representation: str) -> _DeploymentSample:
    if representation == PROBABILITY_REPRESENTATION:
        sample = load_score_sample(path, require_mask=False)
        return _DeploymentSample(sample.image_id, sample.probability, sample.gray)
    image_id, score, gray = _load_label_free_sample(
        path, representation=representation
    )
    return _DeploymentSample(image_id, score, gray)


def _window_statistics(
    samples: Sequence[_DeploymentSample],
    representation: str,
    source_reference,
):
    if representation == RAW_LOGIT_REPRESENTATION:
        return extract_logit_window_statistics(
            [sample.score for sample in samples],
            [sample.gray for sample in samples],
            source_reference=source_reference,
        )
    return extract_window_statistics(
        [sample.score for sample in samples],
        [sample.gray for sample in samples],
        source_reference=source_reference,
    )


def build_static_cross_fit_statistics(
    score_map_dir: str | Path,
    thresholds: np.ndarray,
    *,
    folds: int = 5,
    seed: int = 42,
    adaptation_window: int | None = None,
    source_reference_path: str | None = None,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
    require_score_integrity: bool = False,
    representation: str = PROBABILITY_REPRESENTATION,
    threshold_grid_manifest_sha256: str | None = None,
    threshold_grid_detector_protocol: str | None = None,
    threshold_grid_detector_checkpoint_sha256s: Sequence[str] | None = None,
    threshold_grid_outer_detector_checkpoint_sha256: str | None = None,
    threshold_grid_episode_detector_checkpoint_sha256s: Sequence[str] | None = None,
) -> dict[str, object]:
    """Build deterministic full-coverage statistics for an unordered dataset.

    Every image is an evaluation item exactly once.  Its fold's statistic is
    computed only from images outside that fold, without opening any mask.  If
    ``adaptation_window`` is set, exactly that many complement images are
    sampled without replacement using a fold-local deterministic RNG.  ``None``
    retains the backwards-compatible behaviour of using the whole complement.
    An image necessarily may appear in adaptation sets for other folds, so this
    is a static transductive protocol rather than a causal or CRC-formal
    protocol.
    """

    if (
        isinstance(folds, bool)
        or not isinstance(folds, (int, np.integer))
        or int(folds) < 2
    ):
        raise ValueError("folds must be an integer of at least 2")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if adaptation_window is not None and (
        isinstance(adaptation_window, bool)
        or not isinstance(adaptation_window, (int, np.integer))
        or int(adaptation_window) < 1
    ):
        raise ValueError("adaptation_window must be a positive integer or None")
    requested_adaptation_window = (
        None if adaptation_window is None else int(adaptation_window)
    )
    representation = _normalise_representation(representation)
    grid_manifest_sha256 = _validate_grid_manifest_sha256(
        threshold_grid_manifest_sha256, representation=representation
    )
    (
        grid_detector_protocol,
        grid_detector_hashes,
        grid_outer_detector_hash,
        grid_episode_detector_hashes,
    ) = _validate_grid_detector_contract(
        threshold_grid_detector_protocol,
        threshold_grid_detector_checkpoint_sha256s,
        threshold_grid_outer_detector_checkpoint_sha256,
        threshold_grid_episode_detector_checkpoint_sha256s,
        representation=representation,
    )
    root = Path(score_map_dir)
    score_manifest: dict[str, object] | None = None
    score_integrity: dict[str, object] = {
        "verified": False,
        "records_sha256": None,
        "ordered_image_ids_sha256": None,
        "num_records": None,
    }
    if require_score_integrity or representation == RAW_LOGIT_REPRESENTATION:
        score_manifest, files, score_integrity = verify_score_map_directory(
            root,
            require_integrity=True,
        )
        if score_manifest is None or score_integrity.get("verified") is not True:
            raise ValueError(
                "Formal static cross-fit requires an integrity-verified score manifest"
            )
        if representation == RAW_LOGIT_REPRESENTATION:
            _require_raw_logit_manifest_contract(score_manifest, score_integrity)
    else:
        files = score_files(root)
    if folds > len(files):
        raise ValueError(
            f"static cross-fit needs at least one image per fold: "
            f"folds={folds}, images={len(files)}"
        )
    source_reference = (
        load_source_reference(source_reference_path) if source_reference_path else None
    )
    permutation = np.random.default_rng(int(seed)).permutation(len(files))
    evaluation_folds = [
        np.asarray(indices, dtype=np.int64)
        for indices in np.array_split(permutation, int(folds))
    ]
    rows: list[np.ndarray] = []
    adaptation_rows: list[list[str]] = []
    evaluation_rows: list[list[str]] = []
    block_ids: list[str] = []
    block_records: list[dict[str, object]] = []
    names: tuple[str, ...] | None = None
    all_indices = np.arange(len(files), dtype=np.int64)
    for fold_index, evaluation_indices in enumerate(evaluation_folds):
        evaluation_set = set(int(value) for value in evaluation_indices.tolist())
        complement_indices = np.asarray(
            [index for index in all_indices.tolist() if index not in evaluation_set],
            dtype=np.int64,
        )
        complement_size = int(complement_indices.size)
        if requested_adaptation_window is None:
            adaptation_indices = complement_indices
            sampling_rule = "all_cross_fit_complement_images"
        else:
            if complement_size < requested_adaptation_window:
                raise ValueError(
                    "Static cross-fit complement is smaller than the requested "
                    f"adaptation window in fold {fold_index}: "
                    f"complement_size={complement_size}, "
                    f"adaptation_window={requested_adaptation_window}"
                )
            fold_rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(fold_index)])
            )
            selected_positions = fold_rng.choice(
                complement_size,
                size=requested_adaptation_window,
                replace=False,
            )
            adaptation_indices = complement_indices[selected_positions]
            sampling_rule = (
                "seedsequence(seed,fold_index)_without_replacement_from_complement"
            )
        adaptation = [
            _deployment_sample(files[int(index)], representation)
            for index in adaptation_indices
        ]
        evaluation = [
            _deployment_sample(files[int(index)], representation)
            for index in evaluation_indices
        ]
        result = _window_statistics(adaptation, representation, source_reference)
        if names is None:
            names = result.names
        elif result.names != names:
            raise ValueError("Static cross-fit statistic feature schemas differ")
        adaptation_ids = [sample.image_id for sample in adaptation]
        evaluation_ids = [sample.image_id for sample in evaluation]
        if set(adaptation_ids).intersection(evaluation_ids):
            raise ValueError("Static cross-fit fold contains an adaptation/evaluation overlap")
        fold_id = stable_cross_fit_fold_id(
            fold_index,
            adaptation_ids,
            evaluation_ids,
            schema_version=(
                LOGIT_STATIC_CROSS_FIT_STATISTICS_SCHEMA_VERSION
                if representation == RAW_LOGIT_REPRESENTATION
                else STATIC_CROSS_FIT_STATISTICS_SCHEMA_VERSION
            ),
        )
        rows.append(result.values)
        adaptation_rows.append(adaptation_ids)
        evaluation_rows.append(evaluation_ids)
        block_ids.append(fold_id)
        block_records.append(
            {
                "block_id": fold_id,
                "fold_index": fold_index,
                "adaptation_ids": adaptation_ids,
                "evaluation_ids": evaluation_ids,
                "adaptation_size": len(adaptation_ids),
                "selected_adaptation_ids": adaptation_ids,
                "selected_adaptation_size": len(adaptation_ids),
                "complement_size": complement_size,
                "adaptation_window": requested_adaptation_window,
                "adaptation_sampling_rule": sampling_rule,
                "adaptation_sampling_seed_components": [
                    int(seed),
                    int(fold_index),
                ],
                "evaluation_size": len(evaluation_ids),
                "protocol": "static_cross_fit",
            }
        )
    flattened_evaluation = [item for row in evaluation_rows for item in row]
    manifest_ids = [
        _deployment_sample(path, representation).image_id for path in files
    ]
    if len(flattened_evaluation) != len(set(flattened_evaluation)):
        raise ValueError("An image is evaluated by more than one static cross-fit fold")
    if set(flattened_evaluation) != set(manifest_ids):
        raise ValueError("Static cross-fit does not cover the complete score-map set")
    if names is None:
        raise RuntimeError("No static cross-fit statistics were extracted")
    grid = _validate_grid_for_representation(thresholds, representation)
    grid_sha256 = _grid_semantic_sha256(grid, representation)
    if representation == RAW_LOGIT_REPRESENTATION:
        if score_manifest is None:
            raise ValueError("Raw-logit deployment lacks a verified manifest")
        protocol, fingerprint = _raw_logit_protocol(
            root,
            score_manifest,
            grid,
            matching_rule=matching_rule,
            centroid_distance=centroid_distance,
            connectivity=connectivity,
            min_component_area=min_component_area,
            grid_detector_protocol=str(grid_detector_protocol),
            grid_detector_checkpoint_sha256s=grid_detector_hashes,
            grid_outer_detector_checkpoint_sha256=str(
                grid_outer_detector_hash
            ),
            grid_episode_detector_checkpoint_sha256s=(
                grid_episode_detector_hashes
            ),
        )
    else:
        protocol, fingerprint = score_map_protocol(
            root,
            grid,
            matching_rule=matching_rule,
            centroid_distance=centroid_distance,
            connectivity=connectivity,
            min_component_area=min_component_area,
        )
    manifest_path = root / "manifest.json"
    block_ids_sha256 = hashlib.sha256(
        "\n".join(block_ids).encode("utf-8")
    ).hexdigest()
    provenance = {
        "score_map_dir": str(root.resolve()),
        "score_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "score_integrity_verified": bool(score_integrity.get("verified", False)),
        "score_manifest_schema_version": (
            score_manifest.get("schema_version") if score_manifest is not None else None
        ),
        "score_records_sha256": score_integrity.get("records_sha256"),
        "score_ordered_image_ids_sha256": score_integrity.get(
            "ordered_image_ids_sha256"
        ),
        "score_num_records": score_integrity.get("num_records"),
        "threshold_grid_sha256": grid_sha256,
        "source_reference": source_reference_path,
        "source_reference_sha256": (
            hashlib.sha256(Path(source_reference_path).read_bytes()).hexdigest()
            if source_reference_path
            else None
        ),
        "source_reference_domain_names": (
            list(source_reference.domain_names) if source_reference is not None else []
        ),
        "source_reference_statistics_names_sha256": (
            statistics_names_sha256(source_reference.statistics_names)
            if source_reference is not None
            else None
        ),
        "mode": "static_cross_fit",
        "folds": int(folds),
        "seed": int(seed),
        "adaptation_window": requested_adaptation_window,
        "selected_adaptation_ids_by_fold": adaptation_rows,
        "selected_adaptation_sizes": [len(row) for row in adaptation_rows],
        "complement_sizes": [
            int(record["complement_size"]) for record in block_records
        ],
        "adaptation_sampling_rule": (
            "all_cross_fit_complement_images"
            if requested_adaptation_window is None
            else "seedsequence(seed,fold_index)_without_replacement_from_complement"
        ),
        "num_windows": len(rows),
        "num_images": len(files),
        "score_image_ids_sha256": ordered_ids_sha256(manifest_ids),
        "full_test_coverage": True,
        "block_ids_sha256": block_ids_sha256,
        "one_to_one_evaluation": False,
        "exchangeability_unit": "static_cross_fit_fold",
        "masks_read": False,
        "cross_fit_role_reuse": True,
        "formal_crc_eligible": False,
        "representation": representation,
        "threshold_grid_schema_version": _grid_schema_version(representation, grid),
        "threshold_grid_manifest_sha256": grid_manifest_sha256,
        "threshold_grid_detector_protocol": grid_detector_protocol,
        "threshold_grid_detector_checkpoint_sha256s": list(grid_detector_hashes),
        "threshold_grid_outer_detector_checkpoint_sha256": (
            grid_outer_detector_hash
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": list(
            grid_episode_detector_hashes
        ),
        "feature_schema_sha256": feature_schema_sha256(
            result.schema_version, statistics_names=names
        ),
    }
    arrays: dict[str, object] = {
        "statistics": np.stack(rows).astype(np.float32),
        "statistics_names": np.asarray(names, dtype=str),
        "statistics_names_sha256": np.asarray(statistics_names_sha256(names)),
        "statistics_schema_version": np.asarray(result.schema_version),
        "adaptation_ids": np.asarray(
            [json.dumps(row) for row in adaptation_rows], dtype=str
        ),
        "evaluation_ids": np.asarray(
            [json.dumps(row) for row in evaluation_rows], dtype=str
        ),
        "score_image_ids": np.asarray(manifest_ids, dtype=str),
        "block_ids": np.asarray(block_ids, dtype=str),
        "block_records_json": np.asarray(
            [json.dumps(record, sort_keys=True) for record in block_records], dtype=str
        ),
        "block_ids_sha256": np.asarray(block_ids_sha256),
        "one_to_one_evaluation": np.asarray(False),
        "exchangeability_unit": np.asarray("static_cross_fit_fold"),
        "thresholds": grid,
        "threshold_grid_sha256": np.asarray(grid_sha256),
        "protocol_json": np.asarray(json.dumps(protocol, sort_keys=True)),
        "protocol_fingerprint": np.asarray(fingerprint),
        "provenance_json": np.asarray(json.dumps(provenance, sort_keys=True)),
        "deployment_statistics_schema_version": np.asarray(
            LOGIT_STATIC_CROSS_FIT_STATISTICS_SCHEMA_VERSION
            if representation == RAW_LOGIT_REPRESENTATION
            else STATIC_CROSS_FIT_STATISTICS_SCHEMA_VERSION
        ),
    }
    if representation == RAW_LOGIT_REPRESENTATION:
        arrays.update(
            {
                "representation": np.asarray(representation),
                "threshold_grid_schema_version": np.asarray(
                    LOGIT_THRESHOLD_GRID_SCHEMA_VERSION
                ),
                "threshold_grid_manifest_sha256": np.asarray(
                    str(grid_manifest_sha256)
                ),
                "threshold_grid_detector_protocol": np.asarray(
                    str(grid_detector_protocol)
                ),
                "threshold_grid_detector_checkpoint_sha256s": np.asarray(
                    grid_detector_hashes, dtype=str
                ),
                "threshold_grid_outer_detector_checkpoint_sha256": np.asarray(
                    str(grid_outer_detector_hash)
                ),
                "threshold_grid_episode_detector_checkpoint_sha256s": np.asarray(
                    grid_episode_detector_hashes, dtype=str
                ),
                "feature_schema_sha256": np.asarray(
                    feature_schema_sha256(
                        LOGIT_STATISTICS_SCHEMA_VERSION, statistics_names=names
                    )
                ),
                "prediction_rule": np.asarray(LOGIT_PREDICTION_RULE),
                "empty_action_threshold": np.asarray("+inf"),
                "empty_action_index": np.asarray("null"),
            }
        )
    return arrays


def build_deployment_statistics(
    score_map_dir: str | Path,
    thresholds: np.ndarray,
    *,
    adaptation_window: int = 32,
    evaluation_window: int = 1,
    stride: int | None = None,
    source_reference_path: str | None = None,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
    allow_role_reuse: bool = False,
    require_score_integrity: bool = False,
    representation: str = PROBABILITY_REPRESENTATION,
    threshold_grid_manifest_sha256: str | None = None,
    threshold_grid_detector_protocol: str | None = None,
    threshold_grid_detector_checkpoint_sha256s: Sequence[str] | None = None,
    threshold_grid_outer_detector_checkpoint_sha256: str | None = None,
    threshold_grid_episode_detector_checkpoint_sha256s: Sequence[str] | None = None,
) -> dict[str, object]:
    representation = _normalise_representation(representation)
    grid_manifest_sha256 = _validate_grid_manifest_sha256(
        threshold_grid_manifest_sha256, representation=representation
    )
    (
        grid_detector_protocol,
        grid_detector_hashes,
        grid_outer_detector_hash,
        grid_episode_detector_hashes,
    ) = _validate_grid_detector_contract(
        threshold_grid_detector_protocol,
        threshold_grid_detector_checkpoint_sha256s,
        threshold_grid_outer_detector_checkpoint_sha256,
        threshold_grid_episode_detector_checkpoint_sha256s,
        representation=representation,
    )
    root = Path(score_map_dir)
    score_manifest: dict[str, object] | None = None
    score_integrity: dict[str, object] = {
        "verified": False,
        "records_sha256": None,
        "ordered_image_ids_sha256": None,
        "num_records": None,
    }
    if require_score_integrity or representation == RAW_LOGIT_REPRESENTATION:
        score_manifest, files, score_integrity = verify_score_map_directory(
            root,
            require_integrity=True,
        )
        if score_manifest is None or score_integrity.get("verified") is not True:
            raise ValueError(
                "Formal causal statistics require an integrity-verified score manifest"
            )
        if representation == RAW_LOGIT_REPRESENTATION:
            _require_raw_logit_manifest_contract(score_manifest, score_integrity)
    else:
        files = score_files(root)
    if stride is None:
        stride = adaptation_window + evaluation_window
    windows = build_causal_windows(
        files, adaptation_window, evaluation_window, stride
    )
    if not windows:
        raise ValueError(
            "No complete causal deployment-statistics window: need at least "
            f"A+E={adaptation_window + evaluation_window} score maps"
        )
    source_reference = (
        load_source_reference(source_reference_path) if source_reference_path else None
    )
    score_image_ids = [
        _deployment_sample(path, representation).image_id for path in files
    ]
    if len(score_image_ids) != len(set(score_image_ids)):
        raise ValueError("Deployment score-map image IDs must be globally unique")
    rows = []
    adaptation_rows: list[list[str]] = []
    evaluation_rows: list[list[str]] = []
    block_ids: list[str] = []
    block_records: list[dict[str, object]] = []
    names: tuple[str, ...] | None = None
    for block_index, window in enumerate(windows):
        adaptation = [
            _deployment_sample(path, representation)
            for path in window.adaptation_files
        ]
        evaluation = [
            _deployment_sample(path, representation)
            for path in window.evaluation_files
        ]
        result = _window_statistics(adaptation, representation, source_reference)
        if names is None:
            names = result.names
        elif result.names != names:
            raise ValueError("Deployment statistic feature schemas differ across windows")
        adaptation_ids = [sample.image_id for sample in adaptation]
        evaluation_ids = [sample.image_id for sample in evaluation]
        if len(set(adaptation_ids)) != len(adaptation_ids):
            raise ValueError("A causal block contains duplicate adaptation image IDs")
        if len(set(evaluation_ids)) != len(evaluation_ids):
            raise ValueError("A causal block contains duplicate evaluation image IDs")
        if set(adaptation_ids).intersection(evaluation_ids):
            raise ValueError("An image ID occurs in both A and E of one causal window")
        block_id = stable_block_id(
            block_index,
            adaptation_ids,
            evaluation_ids,
            schema_version=(
                LOGIT_DEPLOYMENT_STATISTICS_SCHEMA_VERSION
                if representation == RAW_LOGIT_REPRESENTATION
                else DEPLOYMENT_STATISTICS_SCHEMA_VERSION
            ),
        )
        rows.append(result.values)
        adaptation_rows.append(adaptation_ids)
        evaluation_rows.append(evaluation_ids)
        block_ids.append(block_id)
        block_records.append(
            {
                "block_id": block_id,
                "block_index": block_index,
                "adaptation_ids": adaptation_ids,
                "evaluation_ids": evaluation_ids,
                "adaptation_size": len(adaptation_ids),
                "evaluation_size": len(evaluation_ids),
                "causal_order": "adaptation_then_evaluation",
            }
        )
    all_adaptation_ids = {item for row in adaptation_rows for item in row}
    all_evaluation_ids = {item for row in evaluation_rows for item in row}
    global_overlap = sorted(all_adaptation_ids.intersection(all_evaluation_ids))
    if global_overlap and not allow_role_reuse:
        raise ValueError(
            "Images change roles across causal windows; use the default full-span "
            "stride or --allow-role-reuse for diagnostic-only output: "
            + ", ".join(global_overlap[:5])
        )
    flattened_evaluation_ids = [item for row in evaluation_rows for item in row]
    if len(set(flattened_evaluation_ids)) != len(flattened_evaluation_ids):
        raise ValueError(
            "An evaluation image ID belongs to more than one causal block; "
            "block-to-action alignment would be ambiguous"
        )
    if names is None:
        raise RuntimeError("No statistics were extracted")
    one_to_one_evaluation = all(len(row) == 1 for row in evaluation_rows)
    block_ids_sha256 = hashlib.sha256(
        "\n".join(block_ids).encode("utf-8")
    ).hexdigest()
    grid = _validate_grid_for_representation(thresholds, representation)
    grid_sha256 = _grid_semantic_sha256(grid, representation)
    if representation == RAW_LOGIT_REPRESENTATION:
        if score_manifest is None:
            raise ValueError("Raw-logit deployment lacks a verified manifest")
        protocol, fingerprint = _raw_logit_protocol(
            root,
            score_manifest,
            grid,
            matching_rule=matching_rule,
            centroid_distance=centroid_distance,
            connectivity=connectivity,
            min_component_area=min_component_area,
            grid_detector_protocol=str(grid_detector_protocol),
            grid_detector_checkpoint_sha256s=grid_detector_hashes,
            grid_outer_detector_checkpoint_sha256=str(
                grid_outer_detector_hash
            ),
            grid_episode_detector_checkpoint_sha256s=(
                grid_episode_detector_hashes
            ),
        )
    else:
        protocol, fingerprint = score_map_protocol(
            root,
            grid,
            matching_rule=matching_rule,
            centroid_distance=centroid_distance,
            connectivity=connectivity,
            min_component_area=min_component_area,
        )
    manifest_path = root / "manifest.json"
    provenance = {
        "score_map_dir": str(root.resolve()),
        "score_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "score_integrity_verified": bool(score_integrity.get("verified", False)),
        "score_manifest_schema_version": (
            score_manifest.get("schema_version") if score_manifest is not None else None
        ),
        "score_records_sha256": score_integrity.get("records_sha256"),
        "score_ordered_image_ids_sha256": score_integrity.get(
            "ordered_image_ids_sha256"
        ),
        "score_num_records": score_integrity.get("num_records"),
        "threshold_grid_sha256": grid_sha256,
        "source_reference": source_reference_path,
        "source_reference_sha256": (
            hashlib.sha256(Path(source_reference_path).read_bytes()).hexdigest()
            if source_reference_path
            else None
        ),
        "source_reference_domain_names": (
            list(source_reference.domain_names) if source_reference is not None else []
        ),
        "source_reference_statistics_names_sha256": (
            statistics_names_sha256(source_reference.statistics_names)
            if source_reference is not None
            else None
        ),
        "adaptation_window": adaptation_window,
        "evaluation_window": evaluation_window,
        "stride": stride,
        "num_windows": len(rows),
        "num_images": len(files),
        "score_image_ids_sha256": ordered_ids_sha256(score_image_ids),
        "block_ids_sha256": block_ids_sha256,
        "one_to_one_evaluation": one_to_one_evaluation,
        "exchangeability_unit": "causal_A_to_E_block",
        "masks_read": False,
        "global_role_overlap": global_overlap,
        "allow_role_reuse": bool(allow_role_reuse),
        "representation": representation,
        "threshold_grid_schema_version": _grid_schema_version(representation, grid),
        "threshold_grid_manifest_sha256": grid_manifest_sha256,
        "threshold_grid_detector_protocol": grid_detector_protocol,
        "threshold_grid_detector_checkpoint_sha256s": list(grid_detector_hashes),
        "threshold_grid_outer_detector_checkpoint_sha256": (
            grid_outer_detector_hash
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": list(
            grid_episode_detector_hashes
        ),
        "feature_schema_sha256": feature_schema_sha256(
            result.schema_version, statistics_names=names
        ),
    }
    arrays: dict[str, object] = {
        "statistics": np.stack(rows).astype(np.float32),
        "statistics_names": np.asarray(names, dtype=str),
        "statistics_names_sha256": np.asarray(statistics_names_sha256(names)),
        "statistics_schema_version": np.asarray(result.schema_version),
        "adaptation_ids": np.asarray(
            [json.dumps(row) for row in adaptation_rows], dtype=str
        ),
        "evaluation_ids": np.asarray(
            [json.dumps(row) for row in evaluation_rows], dtype=str
        ),
        "score_image_ids": np.asarray(score_image_ids, dtype=str),
        "block_ids": np.asarray(block_ids, dtype=str),
        "block_records_json": np.asarray(
            [json.dumps(record, sort_keys=True) for record in block_records], dtype=str
        ),
        "block_ids_sha256": np.asarray(block_ids_sha256),
        "one_to_one_evaluation": np.asarray(one_to_one_evaluation),
        "exchangeability_unit": np.asarray("causal_A_to_E_block"),
        "thresholds": grid,
        "threshold_grid_sha256": np.asarray(grid_sha256),
        "protocol_json": np.asarray(json.dumps(protocol, sort_keys=True)),
        "protocol_fingerprint": np.asarray(fingerprint),
        "provenance_json": np.asarray(json.dumps(provenance, sort_keys=True)),
        "deployment_statistics_schema_version": np.asarray(
            LOGIT_DEPLOYMENT_STATISTICS_SCHEMA_VERSION
            if representation == RAW_LOGIT_REPRESENTATION
            else DEPLOYMENT_STATISTICS_SCHEMA_VERSION
        ),
    }
    if representation == RAW_LOGIT_REPRESENTATION:
        arrays.update(
            {
                "representation": np.asarray(representation),
                "threshold_grid_schema_version": np.asarray(
                    LOGIT_THRESHOLD_GRID_SCHEMA_VERSION
                ),
                "threshold_grid_manifest_sha256": np.asarray(
                    str(grid_manifest_sha256)
                ),
                "threshold_grid_detector_protocol": np.asarray(
                    str(grid_detector_protocol)
                ),
                "threshold_grid_detector_checkpoint_sha256s": np.asarray(
                    grid_detector_hashes, dtype=str
                ),
                "threshold_grid_outer_detector_checkpoint_sha256": np.asarray(
                    str(grid_outer_detector_hash)
                ),
                "threshold_grid_episode_detector_checkpoint_sha256s": np.asarray(
                    grid_episode_detector_hashes, dtype=str
                ),
                "feature_schema_sha256": np.asarray(
                    feature_schema_sha256(
                        LOGIT_STATISTICS_SCHEMA_VERSION, statistics_names=names
                    )
                ),
                "prediction_rule": np.asarray(LOGIT_PREDICTION_RULE),
                "empty_action_threshold": np.asarray("+inf"),
                "empty_action_index": np.asarray("null"),
            }
        )
    return arrays


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-map-dir", required=True)
    parser.add_argument("--threshold-grid", required=True)
    parser.add_argument(
        "--representation",
        choices=(PROBABILITY_REPRESENTATION, RAW_LOGIT_REPRESENTATION),
        default=PROBABILITY_REPRESENTATION,
        help="Score/threshold domain; raw-logit mode requires a verified v4 grid manifest",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=("causal", "static-cross-fit"),
        default="causal",
        help="Use causal A->E blocks or full-coverage static cross-fitting",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adaptation-window", type=int, default=32)
    parser.add_argument("--evaluation-window", type=int, default=1)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--source-reference")
    parser.add_argument("--matching-rule", choices=("overlap", "centroid"), default="overlap")
    parser.add_argument("--centroid-distance", type=float, default=3.0)
    parser.add_argument("--connectivity", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument("--allow-role-reuse", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.representation == RAW_LOGIT_REPRESENTATION:
        grid_artifact = load_logit_grid_artifact(args.threshold_grid)
        grid = grid_artifact.thresholds
        grid_manifest_sha256 = hashlib.sha256(
            grid_artifact.manifest_path.read_bytes()
        ).hexdigest()
        grid_detector_protocol = str(
            grid_artifact.manifest["grid_detector_protocol"]
        )
        grid_detector_checkpoint_sha256s = list(
            grid_artifact.manifest["detector_checkpoint_sha256s"]
        )
        grid_outer_detector_checkpoint_sha256 = str(
            grid_artifact.manifest["outer_detector_checkpoint_sha256"]
        )
        grid_episode_detector_checkpoint_sha256s = list(
            grid_artifact.manifest["episode_detector_checkpoint_sha256s"]
        )
    else:
        grid = load_threshold_grid(args.threshold_grid)
        grid_manifest_sha256 = None
        grid_detector_protocol = None
        grid_detector_checkpoint_sha256s = None
        grid_outer_detector_checkpoint_sha256 = None
        grid_episode_detector_checkpoint_sha256s = None
    if args.mode == "static-cross-fit":
        if args.allow_role_reuse:
            raise ValueError("--allow-role-reuse is only valid in causal mode")
        arrays = build_static_cross_fit_statistics(
            args.score_map_dir,
            grid,
            folds=args.folds,
            seed=args.seed,
            adaptation_window=args.adaptation_window,
            source_reference_path=args.source_reference,
            matching_rule=args.matching_rule,
            centroid_distance=args.centroid_distance,
            connectivity=args.connectivity,
            min_component_area=args.min_component_area,
            # The CLI is the persisted experiment path.  Unlike the library
            # helper's legacy-compatible default, it fails closed on anything
            # other than a complete v3 score-map integrity chain.
            require_score_integrity=True,
            representation=args.representation,
            threshold_grid_manifest_sha256=grid_manifest_sha256,
            threshold_grid_detector_protocol=grid_detector_protocol,
            threshold_grid_detector_checkpoint_sha256s=(
                grid_detector_checkpoint_sha256s
            ),
            threshold_grid_outer_detector_checkpoint_sha256=(
                grid_outer_detector_checkpoint_sha256
            ),
            threshold_grid_episode_detector_checkpoint_sha256s=(
                grid_episode_detector_checkpoint_sha256s
            ),
        )
    else:
        arrays = build_deployment_statistics(
            args.score_map_dir,
            grid,
            adaptation_window=args.adaptation_window,
            evaluation_window=args.evaluation_window,
            stride=args.stride,
            source_reference_path=args.source_reference,
            matching_rule=args.matching_rule,
            centroid_distance=args.centroid_distance,
            connectivity=args.connectivity,
            min_component_area=args.min_component_area,
            allow_role_reuse=args.allow_role_reuse,
            require_score_integrity=True,
            representation=args.representation,
            threshold_grid_manifest_sha256=grid_manifest_sha256,
            threshold_grid_detector_protocol=grid_detector_protocol,
            threshold_grid_detector_checkpoint_sha256s=(
                grid_detector_checkpoint_sha256s
            ),
            threshold_grid_outer_detector_checkpoint_sha256=(
                grid_outer_detector_checkpoint_sha256
            ),
            threshold_grid_episode_detector_checkpoint_sha256s=(
                grid_episode_detector_checkpoint_sha256s
            ),
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(
        json.dumps(
            {
                "output": str(output),
                "num_windows": int(np.asarray(arrays["statistics"]).shape[0]),
                "protocol_fingerprint": str(
                    np.asarray(arrays["protocol_fingerprint"]).item()
                ),
                "representation": args.representation,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEPLOYMENT_STATISTICS_SCHEMA_VERSION",
    "LOGIT_DEPLOYMENT_STATISTICS_SCHEMA_VERSION",
    "LOGIT_STATIC_CROSS_FIT_STATISTICS_SCHEMA_VERSION",
    "STATIC_CROSS_FIT_STATISTICS_SCHEMA_VERSION",
    "build_deployment_statistics",
    "build_static_cross_fit_statistics",
    "stable_block_id",
    "stable_cross_fit_fold_id",
]
