"""Build causal, label-grounded risk-curve episodes from offline score maps.

Each episode has two temporally ordered and disjoint parts.  Label-free
statistics are extracted only from an adaptation window ``A``; ground-truth
masks are consulted only in the immediately following evaluation window ``E``
to form the risk-curve targets.  This mirrors deployment (warm up, then predict
future risk) and prevents the old same-window/transductive leakage.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from evaluation.artifact_integrity import (
    RAW_LOGIT_DTYPE,
    RAW_LOGIT_SCORE_REPRESENTATION,
    verify_score_map_directory,
)
from evaluation.component_matching import match_components

from .domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    STATISTICS_SCHEMA_VERSION,
    SourceStatisticsReference,
    WindowStatistics,
    base_feature_names,
    extract_logit_window_statistics,
    extract_window_statistics,
    feature_schema_sha256,
    load_source_reference,
    logit_feature_names,
    statistics_names_sha256,
)
from .representation import (
    GRID_DETECTOR_PROTOCOL,
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_REPRESENTATION,
    PROBABILITY_REPRESENTATION,
    SUPPORTED_REPRESENTATIONS,
    load_logit_grid_artifact,
    logit_threshold_grid_sha256,
    validate_logit_threshold_grid,
)
from .threshold_grid import (
    load_threshold_grid,
    threshold_grid_sha256,
    threshold_grid_version,
)


EPISODE_SCHEMA_VERSION = "rc-v2-curve-episode-v2-causal"
LOGIT_EPISODE_SCHEMA_VERSION = "rc-v4-curve-episode-v1-raw-logit-causal"
COMPONENT_RISK_SCHEMA_VERSION = "rc-v2-component-risk-v1-raw-upper"
COUNT_ALL_ADAPTATION_SCHEMA_VERSION = "rc-v4-count-all-adaptation-curves-v1"

_COUNT_ALL_PROVENANCE = {
    "count_all_adaptation_schema_version": COUNT_ALL_ADAPTATION_SCHEMA_VERSION,
    "count_all_adaptation_sample_role": "adaptation_window_A_label_free",
    "count_all_adaptation_masks_read": False,
    "count_all_adaptation_prediction_rule": "prediction = (raw_logits >= threshold)",
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


def _domain_key(value: str) -> str:
    """Normalise documented short aliases such as NUAA vs NUAA-SIRST."""

    key = "".join(character for character in str(value).casefold() if character.isalnum())
    if key.endswith("sirst") and len(key) > len("sirst"):
        key = key[: -len("sirst")]
    return key


@dataclass(frozen=True)
class ScoreSample:
    image_id: str
    probability: np.ndarray
    mask: np.ndarray | None
    gray: np.ndarray | None
    source_path: str
    raw_logit: np.ndarray | None = None


@dataclass(frozen=True)
class Episode:
    statistics: WindowStatistics
    pixel_log_risk: np.ndarray
    component_log_risk: np.ndarray
    component_log_risk_raw: np.ndarray
    component_log_risk_upper: np.ndarray
    pd_curve: np.ndarray
    thresholds: np.ndarray
    pixel_fp_counts: np.ndarray
    component_fp_counts: np.ndarray
    tp_object_counts: np.ndarray
    gt_object_count: int
    total_pixels: int
    pseudo_target: str
    adaptation_ids: tuple[str, ...]
    evaluation_ids: tuple[str, ...]
    representation: str = PROBABILITY_REPRESENTATION
    adaptation_predicted_pixel_counts: np.ndarray | None = None
    adaptation_predicted_component_counts_raw: np.ndarray | None = None
    adaptation_predicted_component_counts_upper: np.ndarray | None = None
    adaptation_total_pixels: int | None = None
    connectivity: int = 2
    min_component_area: int = 1

    @property
    def window_ids(self) -> tuple[str, ...]:
        """Deprecated, unambiguous alias for :attr:`evaluation_ids`."""

        return self.evaluation_ids


@dataclass(frozen=True)
class AdaptationPredictionCounts:
    predicted_pixel_counts: np.ndarray
    predicted_component_counts_raw: np.ndarray
    total_pixels: int


@dataclass(frozen=True)
class CausalWindow:
    adaptation_files: tuple[Path, ...]
    evaluation_files: tuple[Path, ...]


def cross_episode_role_reuse(episodes: Sequence[Episode]) -> tuple[str, ...]:
    """Return target-qualified IDs used as both A and E across episodes.

    Image IDs are namespaced by pseudo-target because independent datasets
    commonly reuse short numeric IDs.  Reuse within one pseudo-target breaks
    the formal disjoint-block contract even though every individual episode is
    internally causal.
    """

    adaptation_by_target: dict[str, set[str]] = {}
    evaluation_by_target: dict[str, set[str]] = {}
    for episode in episodes:
        adaptation_by_target.setdefault(episode.pseudo_target, set()).update(
            episode.adaptation_ids
        )
        evaluation_by_target.setdefault(episode.pseudo_target, set()).update(
            episode.evaluation_ids
        )
    overlap: list[str] = []
    for target in sorted(set(adaptation_by_target).union(evaluation_by_target)):
        repeated = adaptation_by_target.get(target, set()).intersection(
            evaluation_by_target.get(target, set())
        )
        overlap.extend(f"{target}:{image_id}" for image_id in sorted(repeated))
    return tuple(overlap)


def monotone_upper_envelope(risk: np.ndarray) -> np.ndarray:
    """Smallest suffix-maximum envelope that dominates a sampled curve."""

    values = np.asarray(risk, dtype=np.float64)
    if values.ndim == 1:
        return np.maximum.accumulate(values[::-1])[::-1]
    if values.ndim == 2:
        return np.maximum.accumulate(values[:, ::-1], axis=1)[:, ::-1]
    raise ValueError("risk must be one- or two-dimensional")


def _scalar_string(value: np.ndarray | str) -> str:
    array = np.asarray(value)
    return str(array.item() if array.ndim == 0 else array.reshape(-1)[0])


def load_score_sample(
    path: str | Path,
    *,
    require_mask: bool = True,
    representation: str = PROBABILITY_REPRESENTATION,
) -> ScoreSample:
    if representation not in SUPPORTED_REPRESENTATIONS:
        raise ValueError(f"Unsupported score representation: {representation!r}")
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if "prob" not in archive:
            raise ValueError(f"{source} is missing required array 'prob'")
        if require_mask and "mask" not in archive:
            raise ValueError(f"{source} is missing required array 'mask'")
        probability_raw = np.asarray(archive["prob"])
        probability = probability_raw.astype(np.float32, copy=False).squeeze()
        mask = (
            (np.asarray(archive["mask"]).squeeze() > 0).astype(np.uint8)
            if require_mask
            else None
        )
        gray = np.asarray(archive["gray"], dtype=np.float32).squeeze() if "gray" in archive else None
        image_id = _scalar_string(archive["image_id"]) if "image_id" in archive else source.stem
        raw_logit: np.ndarray | None = None
        if representation == LOGIT_REPRESENTATION:
            required_raw = {
                "logit",
                "score_representation",
                "logit_dtype",
                "probability_transform",
                "probability_clipping",
                "inference_autocast_enabled",
            }
            missing_raw = sorted(required_raw.difference(archive.files))
            if missing_raw:
                raise ValueError(
                    f"{source} lacks the v4 raw-logit record contract: "
                    + ", ".join(missing_raw)
                )
            logit_array = np.asarray(archive["logit"])
            if logit_array.dtype != np.float32:
                raise ValueError(f"Raw logits must use float32 in {source}")
            embedded = {
                "score_representation": _scalar_string(
                    archive["score_representation"]
                ),
                "logit_dtype": _scalar_string(archive["logit_dtype"]),
                "probability_transform": _scalar_string(
                    archive["probability_transform"]
                ),
                "probability_clipping": _scalar_string(
                    archive["probability_clipping"]
                ),
            }
            expected = {
                "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
                "logit_dtype": RAW_LOGIT_DTYPE,
                "probability_transform": "sigmoid",
                "probability_clipping": "none",
            }
            if embedded != expected:
                raise ValueError(f"Raw-logit precision contract mismatch in {source}")
            autocast = np.asarray(archive["inference_autocast_enabled"])
            if autocast.ndim != 0 or autocast.dtype.kind != "b" or bool(autocast):
                raise ValueError(f"Raw-logit export must disable autocast in {source}")
            raw_logit = logit_array.squeeze()
    if probability.ndim != 2:
        raise ValueError(f"Invalid score shape in {source}: {probability.shape}")
    if mask is not None and probability.shape != mask.shape:
        raise ValueError(
            f"Invalid score/mask shapes in {source}: {probability.shape}, {mask.shape}"
        )
    if not np.isfinite(probability).all() or probability.min() < 0.0 or probability.max() > 1.0:
        raise ValueError(f"Invalid probability values in {source}")
    if gray is not None and gray.shape != probability.shape:
        raise ValueError(f"gray and probability shapes differ in {source}")
    if raw_logit is not None:
        if raw_logit.ndim != 2 or raw_logit.shape != probability.shape:
            raise ValueError(f"raw-logit and probability shapes differ in {source}")
        if not np.isfinite(raw_logit).all():
            raise ValueError(f"Invalid raw-logit values in {source}")
    return ScoreSample(image_id, probability, mask, gray, str(source), raw_logit)


def _score_for_representation(
    sample: ScoreSample,
    representation: str,
) -> np.ndarray:
    if representation == PROBABILITY_REPRESENTATION:
        return sample.probability
    if representation == LOGIT_REPRESENTATION:
        if sample.raw_logit is None:
            raise ValueError(
                f"Sample {sample.image_id!r} has no float32 raw-logit map"
            )
        return sample.raw_logit
    raise ValueError(f"Unsupported score representation: {representation!r}")


def _empirical_log_risk(counts: np.ndarray, exposure: float, epsilon: float) -> np.ndarray:
    return np.log10(np.asarray(counts, dtype=np.float64) / max(float(exposure), 1e-12) + epsilon)


def _prediction_count_curves(
    score: np.ndarray,
    thresholds: np.ndarray,
    *,
    connectivity: int,
    min_component_area: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Count retained pixels/components at every threshold without a mask.

    Pixels are activated once, in descending-score order.  A disjoint-set
    forest then tracks the exact connected components of every super-level
    set.  This is equivalent to relabeling ``score >= threshold`` at each grid
    point, but avoids one full connected-component pass per threshold.
    """

    values = np.asarray(score)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("Count-all adaptation scores must be a floating 2-D array")
    if values.dtype != np.float32:
        raise ValueError("Count-all adaptation raw logits must remain float32")
    if not np.isfinite(values).all():
        raise ValueError("Count-all adaptation raw logits contain NaN or infinity")
    grid = validate_logit_threshold_grid(np.asarray(thresholds))
    if connectivity in {1, 4}:
        offsets = ((-1, 0), (0, -1), (0, 1), (1, 0))
    elif connectivity in {2, 8}:
        offsets = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    else:
        raise ValueError("connectivity must be 1/2 (skimage) or 4/8")
    if isinstance(min_component_area, bool) or not isinstance(
        min_component_area, int
    ):
        raise TypeError("min_component_area must be an integer")
    if min_component_area <= 0:
        raise ValueError("min_component_area must be positive")

    height, width = values.shape
    flat = values.reshape(-1)
    # mergesort gives deterministic ordering inside equal-logit batches.  The
    # counts are sampled only after the complete batch has been activated, so
    # the result is independent of that tie order.
    order = np.argsort(-flat, kind="stable")
    parent = np.full(flat.size, -1, dtype=np.int64)
    sizes = np.zeros(flat.size, dtype=np.int64)
    active = np.zeros(flat.size, dtype=bool)
    pixel_counts = np.zeros(grid.size, dtype=np.int64)
    component_counts = np.zeros(grid.size, dtype=np.int64)
    qualified_pixels = 0
    qualified_components = 0
    cursor = 0

    def find(index: int) -> int:
        root = index
        while int(parent[root]) != root:
            root = int(parent[root])
        while index != root:
            next_index = int(parent[index])
            parent[index] = root
            index = next_index
        return root

    def remove_qualified(root: int) -> None:
        nonlocal qualified_pixels, qualified_components
        size = int(sizes[root])
        if size >= min_component_area:
            qualified_pixels -= size
            qualified_components -= 1

    def add_qualified(root: int) -> None:
        nonlocal qualified_pixels, qualified_components
        size = int(sizes[root])
        if size >= min_component_area:
            qualified_pixels += size
            qualified_components += 1

    for threshold_index in range(grid.size - 1, -1, -1):
        threshold = float(grid[threshold_index])
        while cursor < order.size and float(flat[int(order[cursor])]) >= threshold:
            index = int(order[cursor])
            cursor += 1
            active[index] = True
            parent[index] = index
            sizes[index] = 1
            add_qualified(index)
            row, column = divmod(index, width)
            for delta_row, delta_column in offsets:
                neighbor_row = row + delta_row
                neighbor_column = column + delta_column
                if not (
                    0 <= neighbor_row < height and 0 <= neighbor_column < width
                ):
                    continue
                neighbor = neighbor_row * width + neighbor_column
                if not active[neighbor]:
                    continue
                left = find(index)
                right = find(neighbor)
                if left == right:
                    continue
                remove_qualified(left)
                remove_qualified(right)
                # Deterministic union-by-size; root index breaks equal-size ties.
                if (int(sizes[left]), -left) < (int(sizes[right]), -right):
                    left, right = right, left
                parent[right] = left
                sizes[left] += sizes[right]
                add_qualified(left)
            # The newly activated pixel may have been merged; no result is read
            # until all pixels satisfying this threshold have been processed.
        pixel_counts[threshold_index] = qualified_pixels
        component_counts[threshold_index] = qualified_components

    if np.any(np.diff(pixel_counts) > 0):
        raise AssertionError("Count-all adaptation pixel counts are not monotone")
    return pixel_counts, component_counts


_COUNT_ALL_WORKER_GRID: np.ndarray | None = None
_COUNT_ALL_WORKER_CONNECTIVITY: int | None = None
_COUNT_ALL_WORKER_MIN_COMPONENT_AREA: int | None = None


def _initialise_count_all_worker(
    thresholds: np.ndarray,
    connectivity: int,
    min_component_area: int,
) -> None:
    """Initialise one spawn worker without opening any score-map mask."""

    global _COUNT_ALL_WORKER_GRID
    global _COUNT_ALL_WORKER_CONNECTIVITY
    global _COUNT_ALL_WORKER_MIN_COMPONENT_AREA
    _COUNT_ALL_WORKER_GRID = validate_logit_threshold_grid(
        np.asarray(thresholds, dtype=np.float32)
    )
    _COUNT_ALL_WORKER_CONNECTIVITY = int(connectivity)
    _COUNT_ALL_WORKER_MIN_COMPONENT_AREA = int(min_component_area)


def _count_all_counts_from_path(
    path: str | Path,
    thresholds: np.ndarray,
    *,
    connectivity: int,
    min_component_area: int,
) -> AdaptationPredictionCounts:
    """Read only raw logits from one A record and compute its count curves."""

    sample = load_score_sample(
        path,
        require_mask=False,
        representation=LOGIT_REPRESENTATION,
    )
    score = _score_for_representation(sample, LOGIT_REPRESENTATION)
    pixels, components = _prediction_count_curves(
        score,
        thresholds,
        connectivity=connectivity,
        min_component_area=min_component_area,
    )
    return AdaptationPredictionCounts(
        predicted_pixel_counts=pixels,
        predicted_component_counts_raw=components,
        total_pixels=int(score.size),
    )


def _count_all_worker_task(path: str) -> tuple[str, AdaptationPredictionCounts]:
    if (
        _COUNT_ALL_WORKER_GRID is None
        or _COUNT_ALL_WORKER_CONNECTIVITY is None
        or _COUNT_ALL_WORKER_MIN_COMPONENT_AREA is None
    ):
        raise RuntimeError("Count-all worker was not initialised")
    counts = _count_all_counts_from_path(
        path,
        _COUNT_ALL_WORKER_GRID,
        connectivity=_COUNT_ALL_WORKER_CONNECTIVITY,
        min_component_area=_COUNT_ALL_WORKER_MIN_COMPONENT_AREA,
    )
    return str(Path(path).expanduser().resolve()), counts


def _count_all_path_key(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _precompute_adaptation_prediction_counts(
    paths: Sequence[Path],
    thresholds: np.ndarray,
    *,
    connectivity: int,
    min_component_area: int,
    executor: ProcessPoolExecutor | None = None,
) -> dict[str, AdaptationPredictionCounts]:
    """Precompute every unique A score map exactly once."""

    unique: dict[str, Path] = {}
    for path in paths:
        unique.setdefault(_count_all_path_key(path), Path(path))
    if not unique:
        return {}
    cache: dict[str, AdaptationPredictionCounts] = {}
    if executor is None:
        for key, path in unique.items():
            cache[key] = _count_all_counts_from_path(
                path,
                thresholds,
                connectivity=connectivity,
                min_component_area=min_component_area,
            )
        return cache

    futures = {
        executor.submit(_count_all_worker_task, key): key for key in unique
    }
    try:
        for future in as_completed(futures):
            expected_key = futures[future]
            returned_key, counts = future.result()
            if returned_key != expected_key:
                raise RuntimeError("Count-all worker returned the wrong score-map path")
            cache[expected_key] = counts
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    if set(cache) != set(unique):
        raise RuntimeError("Count-all worker pool returned an incomplete cache")
    return cache


def build_episode(
    adaptation_samples: Sequence[ScoreSample],
    thresholds: np.ndarray,
    pseudo_target: str,
    *,
    evaluation_samples: Sequence[ScoreSample],
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
    component_upper_envelope: bool = True,
    source_reference: SourceStatisticsReference | None = None,
    representation: str = PROBABILITY_REPRESENTATION,
    adaptation_prediction_counts: Sequence[AdaptationPredictionCounts] | None = None,
) -> Episode:
    if representation not in SUPPORTED_REPRESENTATIONS:
        raise ValueError(f"Unsupported score representation: {representation!r}")
    if not adaptation_samples:
        raise ValueError("An adaptation window must contain at least one sample")
    if not evaluation_samples:
        raise ValueError("An evaluation window must contain at least one sample")
    adaptation_ids = tuple(sample.image_id for sample in adaptation_samples)
    evaluation_ids = tuple(sample.image_id for sample in evaluation_samples)
    if len(set(adaptation_ids)) != len(adaptation_ids):
        raise ValueError("adaptation window contains duplicate image IDs")
    if len(set(evaluation_ids)) != len(evaluation_ids):
        raise ValueError("evaluation window contains duplicate image IDs")
    overlap = set(adaptation_ids).intersection(evaluation_ids)
    if overlap:
        raise ValueError(
            "Adaptation and evaluation windows must be disjoint; repeated IDs: "
            + ", ".join(sorted(overlap)[:5])
        )
    if representation == LOGIT_REPRESENTATION:
        grid = validate_logit_threshold_grid(np.asarray(thresholds))
    else:
        grid = np.asarray(thresholds, dtype=np.float32).reshape(-1)
    pixel_fp = np.zeros(grid.size, dtype=np.int64)
    component_fp = np.zeros(grid.size, dtype=np.int64)
    tp_objects = np.zeros(grid.size, dtype=np.int64)
    gt_objects_at_threshold = np.zeros(grid.size, dtype=np.int64)
    # Exposures and all supervised targets come exclusively from future E.
    if any(sample.mask is None for sample in evaluation_samples):
        raise ValueError("Every evaluation sample must provide a ground-truth mask")
    total_pixels = sum(
        int(_score_for_representation(sample, representation).size)
        for sample in evaluation_samples
    )

    adaptation_predicted_pixel_counts: np.ndarray | None = None
    adaptation_predicted_component_counts_raw: np.ndarray | None = None
    adaptation_predicted_component_counts_upper: np.ndarray | None = None
    adaptation_total_pixels: int | None = None
    if representation == LOGIT_REPRESENTATION:
        # This branch deliberately reads only A raw scores.  ``sample.mask`` is
        # neither inspected nor passed to any helper, making Count-all strictly
        # label-free by construction.
        adaptation_predicted_pixel_counts = np.zeros(grid.size, dtype=np.int64)
        adaptation_predicted_component_counts_raw = np.zeros(
            grid.size, dtype=np.int64
        )
        adaptation_total_pixels = 0
        if adaptation_prediction_counts is not None and len(
            adaptation_prediction_counts
        ) != len(adaptation_samples):
            raise ValueError(
                "adaptation_prediction_counts must align one-to-one with A samples"
            )
        for sample_index, sample in enumerate(adaptation_samples):
            score = _score_for_representation(sample, representation)
            cached = (
                None
                if adaptation_prediction_counts is None
                else adaptation_prediction_counts[sample_index]
            )
            if cached is None:
                predicted_pixels, predicted_components = _prediction_count_curves(
                    score,
                    grid,
                    connectivity=connectivity,
                    min_component_area=min_component_area,
                )
                sample_total_pixels = int(score.size)
            else:
                predicted_pixels = np.asarray(cached.predicted_pixel_counts)
                predicted_components = np.asarray(
                    cached.predicted_component_counts_raw
                )
                sample_total_pixels = int(cached.total_pixels)
                if (
                    predicted_pixels.shape != grid.shape
                    or predicted_components.shape != grid.shape
                    or predicted_pixels.dtype.kind not in "iu"
                    or predicted_components.dtype.kind not in "iu"
                    or np.any(predicted_pixels < 0)
                    or np.any(predicted_components < 0)
                    or sample_total_pixels != int(score.size)
                ):
                    raise ValueError(
                        "Cached Count-all A curves violate the score/grid contract"
                    )
            adaptation_predicted_pixel_counts += predicted_pixels
            adaptation_predicted_component_counts_raw += predicted_components
            adaptation_total_pixels += sample_total_pixels
        adaptation_predicted_component_counts_upper = monotone_upper_envelope(
            adaptation_predicted_component_counts_raw
        ).astype(np.int64)
    elif adaptation_prediction_counts is not None:
        raise ValueError(
            "adaptation_prediction_counts are only valid for raw-logit episodes"
        )

    for threshold_index, threshold in enumerate(grid):
        for sample in evaluation_samples:
            assert sample.mask is not None
            score = _score_for_representation(sample, representation)
            result = match_components(
                score >= float(threshold),
                sample.mask,
                rule=matching_rule,
                centroid_distance=centroid_distance,
                connectivity=connectivity,
                min_component_area=min_component_area,
            )
            pixel_fp[threshold_index] += int(result.num_fp_pixels)
            component_fp[threshold_index] += int(result.num_fp_components)
            tp_objects[threshold_index] += int(result.num_tp_objects)
            gt_objects_at_threshold[threshold_index] += int(result.num_gt)
    if not np.all(gt_objects_at_threshold == gt_objects_at_threshold[0]):
        raise RuntimeError("Ground-truth object count changed across thresholds")
    gt_objects = int(gt_objects_at_threshold[0])
    pixel_log_risk = _empirical_log_risk(pixel_fp, total_pixels, epsilon=1e-12)
    total_megapixels = total_pixels / 1_000_000.0
    component_log_risk_raw = _empirical_log_risk(
        component_fp, total_megapixels, epsilon=1e-6
    )
    component_log_risk_upper = monotone_upper_envelope(component_log_risk_raw)
    # Keep the historical field as the predictor's supervision alias.  The
    # explicit raw/upper fields retain the evidence needed to audit how that
    # target was constructed.  ``False`` preserves the legacy opt-out API for
    # diagnostic callers; formal episode generation always requests ``True``.
    component_log_risk = (
        component_log_risk_upper
        if component_upper_envelope
        else component_log_risk_raw
    )
    pd_curve = tp_objects.astype(np.float64) / max(gt_objects, 1)
    # Masks from A are deliberately inaccessible to the statistics extractor.
    if representation == LOGIT_REPRESENTATION:
        statistics = extract_logit_window_statistics(
            [
                _score_for_representation(sample, representation)
                for sample in adaptation_samples
            ],
            [sample.gray for sample in adaptation_samples],
            source_reference=source_reference,
        )
    else:
        statistics = extract_window_statistics(
            [sample.probability for sample in adaptation_samples],
            [sample.gray for sample in adaptation_samples],
            source_reference=source_reference,
        )
    return Episode(
        statistics=statistics,
        pixel_log_risk=pixel_log_risk.astype(np.float32),
        component_log_risk=component_log_risk.astype(np.float32),
        component_log_risk_raw=component_log_risk_raw.astype(np.float32),
        component_log_risk_upper=component_log_risk_upper.astype(np.float32),
        pd_curve=pd_curve.astype(np.float32),
        thresholds=grid,
        pixel_fp_counts=pixel_fp,
        component_fp_counts=component_fp,
        tp_object_counts=tp_objects,
        gt_object_count=gt_objects,
        total_pixels=total_pixels,
        pseudo_target=pseudo_target,
        adaptation_ids=adaptation_ids,
        evaluation_ids=evaluation_ids,
        representation=representation,
        adaptation_predicted_pixel_counts=adaptation_predicted_pixel_counts,
        adaptation_predicted_component_counts_raw=(
            adaptation_predicted_component_counts_raw
        ),
        adaptation_predicted_component_counts_upper=(
            adaptation_predicted_component_counts_upper
        ),
        adaptation_total_pixels=adaptation_total_pixels,
        connectivity=connectivity,
        min_component_area=min_component_area,
    )


def score_files(score_map_dir: str | Path) -> list[Path]:
    root = Path(score_map_dir)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        listed = manifest.get("files")
        if not listed and manifest.get("records"):
            listed = [record["file"] for record in manifest["records"]]
        if listed:
            files = [root / str(item) for item in listed]
        else:
            files = sorted(root.glob("*.npz"))
    else:
        files = sorted(root.glob("*.npz"))
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Manifest references missing score maps: {missing[:3]}")
    if not files:
        raise FileNotFoundError(f"No score maps found under {root}")
    return files


def audit_fold_score_manifest(
    score_map_dir: str | Path,
    pseudo_target: str,
    expected_split_role: str | None = None,
    *,
    verified_manifest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Verify detector-fold provenance and, optionally, its official split.

    ``expected_split_role=None`` preserves the historical library API: only
    detector-source LODO provenance and the detector-weight digest are
    audited.  Supplying ``"train"`` or ``"test"`` additionally activates the
    strict score-artifact contract used by the main risk-curve pipeline.  In
    that mode the official split must match, its authority must be verified,
    native spatial inference is mandatory, and diagnostic/non-strict detector
    artifacts are not accepted as verified.
    """

    if expected_split_role not in {None, "train", "test"}:
        raise ValueError("expected_split_role must be None, 'train', or 'test'")

    root = Path(score_map_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {
            "verified": False,
            "reason": "missing_manifest",
            "pseudo_target": pseudo_target,
            "expected_split_role": expected_split_role,
        }
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if verified_manifest is None
        else dict(verified_manifest)
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    strict_artifact_evidence = {
        "expected_split_role": expected_split_role,
        "split_role": manifest.get("split_role"),
        "split_authority_verified": manifest.get("split_authority_verified"),
        "spatial_mode": manifest.get("spatial_mode"),
        "checkpoint_diagnostic_only": bool(
            manifest.get("checkpoint_diagnostic_only", False)
        ),
        "non_strict_state_loading": bool(
            manifest.get("non_strict_state_loading", False)
        ),
    }
    declared_target = manifest.get("target_dataset")
    if not isinstance(declared_target, str) or not declared_target.strip():
        return {
            "verified": False,
            "reason": "missing_target_dataset",
            "pseudo_target": pseudo_target,
            "manifest_sha256": manifest_sha256,
            **strict_artifact_evidence,
        }
    target_domain = declared_target.strip()
    if _domain_key(target_domain) != _domain_key(pseudo_target):
        raise ValueError(
            f"Pseudo-target mismatch: CLI declares {pseudo_target!r}, but "
            f"{manifest_path} declares target_dataset={target_domain!r}"
        )
    source_domains = manifest.get("source_datasets")
    if source_domains is None and manifest.get("source_dataset") is not None:
        source_domains = [manifest["source_dataset"]]
    if not isinstance(source_domains, list) or not source_domains:
        return {
            "verified": False,
            "reason": "missing_detector_source_domains",
            "pseudo_target": pseudo_target,
            "target_dataset": target_domain,
            "manifest_sha256": manifest_sha256,
            **strict_artifact_evidence,
        }
    sources = tuple(str(value) for value in source_domains)
    if any(not value.strip() for value in sources) or len(set(sources)) != len(sources):
        raise ValueError(f"Invalid source_datasets in {manifest_path}")
    detector_sha = manifest.get("weight_sha256")
    weight_path = manifest.get("weight_path")
    if detector_sha is None and weight_path and Path(weight_path).is_file():
        detector_sha = hashlib.sha256(Path(weight_path).read_bytes()).hexdigest()
    if detector_sha is None:
        return {
            "verified": False,
            "reason": "missing_detector_weight_sha256",
            "pseudo_target": pseudo_target,
            "target_dataset": target_domain,
            "source_datasets": list(sources),
            "manifest_sha256": manifest_sha256,
            **strict_artifact_evidence,
        }
    detector_sha = str(detector_sha).strip().lower()
    if len(detector_sha) != 64 or any(
        character not in "0123456789abcdef" for character in detector_sha
    ):
        return {
            "verified": False,
            "reason": "invalid_detector_weight_sha256",
            "pseudo_target": pseudo_target,
            "target_dataset": target_domain,
            "source_datasets": list(sources),
            "manifest_sha256": manifest_sha256,
            **strict_artifact_evidence,
        }
    target_key = _domain_key(target_domain)
    source_keys = {_domain_key(value) for value in sources}
    if target_key in source_keys:
        raise ValueError(
            f"Pseudo-target leakage: detector for {target_domain!r} was trained on "
            f"that same domain according to {manifest_path}"
        )
    if expected_split_role is not None:
        strict_failures = (
            (
                manifest.get("split_role") != expected_split_role,
                "split_role_mismatch",
            ),
            (
                manifest.get("split_authority_verified") is not True,
                "split_authority_unverified",
            ),
            (manifest.get("spatial_mode") != "native", "non_native_spatial_mode"),
            (
                bool(manifest.get("checkpoint_diagnostic_only", False)),
                "checkpoint_diagnostic_only",
            ),
            (
                bool(manifest.get("non_strict_state_loading", False)),
                "non_strict_state_loading",
            ),
        )
        for failed, reason in strict_failures:
            if failed:
                return {
                    "verified": False,
                    "reason": reason,
                    "pseudo_target": pseudo_target,
                    "target_dataset": target_domain,
                    "source_datasets": list(sources),
                    "detector_weight_sha256": detector_sha,
                    "manifest_sha256": manifest_sha256,
                    **strict_artifact_evidence,
                }
    return {
        "verified": True,
        "reason": None,
        "pseudo_target": pseudo_target,
        "target_dataset": target_domain,
        "source_datasets": list(sources),
        "detector_weight_sha256": detector_sha,
        "warm_flag": manifest.get("warm_flag"),
        "manifest_sha256": manifest_sha256,
        **strict_artifact_evidence,
    }


def assert_source_reference_excludes_pseudo_targets(
    source_reference: SourceStatisticsReference,
    pseudo_targets: Sequence[str],
) -> None:
    """Prevent label-free distance centres from leaking any pseudo target.

    One shared reference is safe across nested pseudo-target folds only when
    every centre comes from an external source domain.  Fold-specific
    references require separately trained predictors because their feature
    names/order differ.
    """

    reference_by_key = {
        str(name).strip().casefold(): str(name)
        for name in source_reference.domain_names
    }
    overlap = sorted(
        {
            reference_by_key[str(target).strip().casefold()]
            for target in pseudo_targets
            if str(target).strip().casefold() in reference_by_key
        }
    )
    if overlap:
        raise ValueError(
            "A shared source reference contains pseudo-target domains: "
            + ", ".join(overlap)
            + ". Use external-only centres or train a separate predictor/reference "
            "per fold."
        )


def build_causal_windows(
    files: Sequence[Path],
    adaptation_window: int,
    evaluation_window: int,
    stride: int,
) -> list[CausalWindow]:
    """Return chronological ``A -> E`` windows without partial episodes.

    ``files`` is expected to be in acquisition/manifest order.  A trailing
    incomplete pair is intentionally skipped; the caller records that fact in
    the run manifest.
    """

    if adaptation_window < 1 or evaluation_window < 1 or stride < 1:
        raise ValueError("adaptation_window, evaluation_window, and stride must be positive")
    span = adaptation_window + evaluation_window
    if len(files) < span:
        return []
    windows: list[CausalWindow] = []
    for start in range(0, len(files) - span + 1, stride):
        boundary = start + adaptation_window
        end = boundary + evaluation_window
        adaptation = tuple(files[start:boundary])
        evaluation = tuple(files[boundary:end])
        if set(adaptation).intersection(evaluation):
            raise ValueError("A score-map path occurs in both A and E")
        windows.append(CausalWindow(adaptation, evaluation))
    return windows


def build_windows(files: Sequence[Path], window_size: int, stride: int) -> list[list[Path]]:
    """Deprecated legacy helper; it is not used by the causal episode builder."""

    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be positive")
    if len(files) < window_size:
        return []
    return [
        list(files[start : start + window_size])
        for start in range(0, len(files) - window_size + 1, stride)
    ]


def _pack_episodes(episodes: Sequence[Episode], output: Path, provenance: dict[str, object]) -> None:
    if not episodes:
        raise ValueError(f"No episodes available for {output}")
    reference_grid = episodes[0].thresholds
    reference_names = episodes[0].statistics.names
    representation = episodes[0].representation
    statistics_schema = episodes[0].statistics.schema_version
    provenance = dict(provenance)
    if representation not in SUPPORTED_REPRESENTATIONS:
        raise ValueError(f"Unsupported episode representation: {representation!r}")
    if representation == LOGIT_REPRESENTATION:
        validate_logit_threshold_grid(reference_grid)
        if statistics_schema != LOGIT_STATISTICS_SCHEMA_VERSION:
            raise ValueError("Raw-logit episodes require the v4 logit statistics schema")
        episode_schema = LOGIT_EPISODE_SCHEMA_VERSION
        grid_schema = LOGIT_GRID_SCHEMA_VERSION
        grid_hash = logit_threshold_grid_sha256(reference_grid)
        grid_version = LOGIT_GRID_SCHEMA_VERSION
        expected_feature_hash = feature_schema_sha256(
            statistics_schema, statistics_names=reference_names
        )
        required_provenance = {
            "representation": LOGIT_REPRESENTATION,
            "threshold_grid_sha256": grid_hash,
            "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
            "feature_schema_sha256": expected_feature_hash,
            "threshold_grid_outer_target_excluded": True,
            "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        }
        for field, expected in required_provenance.items():
            if provenance.get(field) != expected:
                raise ValueError(
                    f"Raw-logit episode provenance {field} must equal {expected!r}"
                )
        manifest_hash = provenance.get("threshold_grid_manifest_sha256")
        if (
            not isinstance(manifest_hash, str)
            or len(manifest_hash) != 64
            or any(character not in "0123456789abcdef" for character in manifest_hash)
        ):
            raise ValueError(
                "Raw-logit episodes require a valid threshold-grid manifest SHA-256"
            )
        detector_hashes = provenance.get(
            "threshold_grid_detector_checkpoint_sha256s"
        )
        if (
            not isinstance(detector_hashes, list)
            or not detector_hashes
            or len(set(detector_hashes)) != len(detector_hashes)
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in detector_hashes
            )
        ):
            raise ValueError(
                "Raw-logit episodes require detector checkpoint hashes from "
                "the global source grid"
            )
        outer_detector_hash = provenance.get(
            "threshold_grid_outer_detector_checkpoint_sha256"
        )
        episode_detector_hashes = provenance.get(
            "threshold_grid_episode_detector_checkpoint_sha256s"
        )
        if (
            not isinstance(outer_detector_hash, str)
            or outer_detector_hash not in detector_hashes
            or not isinstance(episode_detector_hashes, list)
            or not episode_detector_hashes
            or len(set(episode_detector_hashes)) != len(episode_detector_hashes)
            or any(value not in detector_hashes for value in episode_detector_hashes)
            or outer_detector_hash in episode_detector_hashes
            or set(detector_hashes)
            != set(episode_detector_hashes).union({outer_detector_hash})
        ):
            raise ValueError(
                "Raw-logit episodes require disjoint outer-final and inner "
                "pseudo-target detector checkpoint roles"
            )
        reference_connectivity = episodes[0].connectivity
        reference_min_component_area = episodes[0].min_component_area
        for field, expected in {
            **_COUNT_ALL_PROVENANCE,
            "connectivity": reference_connectivity,
            "min_component_area": reference_min_component_area,
        }.items():
            if field in provenance and provenance[field] != expected:
                raise ValueError(
                    f"Raw-logit Count-all provenance {field} must equal {expected!r}"
                )
            provenance[field] = expected
    else:
        if statistics_schema != STATISTICS_SCHEMA_VERSION:
            raise ValueError("Probability episodes require the v2 probability statistics schema")
        episode_schema = EPISODE_SCHEMA_VERSION
        grid_schema = threshold_grid_version(reference_grid)
        grid_hash = threshold_grid_sha256(reference_grid)
        grid_version = threshold_grid_version(reference_grid)
    supervision_aliases: set[str] = set()
    for episode in episodes:
        if not np.array_equal(episode.thresholds, reference_grid):
            raise ValueError("Episode threshold grids differ")
        if episode.statistics.names != reference_names:
            raise ValueError("Episode statistic schemas differ")
        if episode.representation != representation:
            raise ValueError("Episode score representations differ")
        if episode.statistics.schema_version != statistics_schema:
            raise ValueError("Episode statistics schema versions differ")
        if representation == LOGIT_REPRESENTATION:
            if episode.connectivity != provenance["connectivity"]:
                raise ValueError("Raw-logit episodes use inconsistent connectivity")
            if episode.min_component_area != provenance["min_component_area"]:
                raise ValueError(
                    "Raw-logit episodes use inconsistent min_component_area"
                )
            count_fields = {
                "adaptation_predicted_pixel_counts": (
                    episode.adaptation_predicted_pixel_counts
                ),
                "adaptation_predicted_component_counts_raw": (
                    episode.adaptation_predicted_component_counts_raw
                ),
                "adaptation_predicted_component_counts_upper": (
                    episode.adaptation_predicted_component_counts_upper
                ),
            }
            validated_counts: dict[str, np.ndarray] = {}
            for field, raw in count_fields.items():
                if raw is None:
                    raise ValueError(f"Raw-logit episode lacks {field}")
                array = np.asarray(raw)
                if array.shape != reference_grid.shape or array.dtype.kind not in "iu":
                    raise ValueError(
                        f"Raw-logit episode {field} must be an integer grid curve"
                    )
                if np.any(array < 0):
                    raise ValueError(f"Raw-logit episode {field} contains negatives")
                validated_counts[field] = array.astype(np.int64)
            adaptation_total = episode.adaptation_total_pixels
            if (
                isinstance(adaptation_total, bool)
                or not isinstance(adaptation_total, (int, np.integer))
                or int(adaptation_total) <= 0
            ):
                raise ValueError(
                    "Raw-logit episode adaptation_total_pixels must be positive"
                )
            pixel_counts = validated_counts["adaptation_predicted_pixel_counts"]
            component_raw = validated_counts[
                "adaptation_predicted_component_counts_raw"
            ]
            component_upper = validated_counts[
                "adaptation_predicted_component_counts_upper"
            ]
            if np.any(pixel_counts > int(adaptation_total)):
                raise ValueError(
                    "Raw-logit adaptation predicted pixels exceed A exposure"
                )
            if np.any(component_raw > pixel_counts):
                raise ValueError(
                    "Raw-logit adaptation components exceed retained pixels"
                )
            if np.any(np.diff(pixel_counts) > 0):
                raise ValueError(
                    "Raw-logit adaptation predicted-pixel curve is not monotone"
                )
            expected_component_upper = monotone_upper_envelope(
                component_raw
            ).astype(np.int64)
            if not np.array_equal(component_upper, expected_component_upper):
                raise ValueError(
                    "Raw-logit adaptation component upper curve must equal "
                    "suffix_max(raw counts)"
                )
        expected_upper = monotone_upper_envelope(episode.component_log_risk_raw)
        if not np.allclose(
            episode.component_log_risk_upper,
            expected_upper,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError(
                "component_log_risk_upper must be the suffix-maximum envelope "
                "of component_log_risk_raw"
            )
        if np.allclose(
            episode.component_log_risk,
            episode.component_log_risk_upper,
            rtol=1e-6,
            atol=1e-6,
        ):
            supervision_aliases.add("component_log_risk_upper")
        elif np.allclose(
            episode.component_log_risk,
            episode.component_log_risk_raw,
            rtol=1e-6,
            atol=1e-6,
        ):
            supervision_aliases.add("component_log_risk_raw")
        else:
            raise ValueError(
                "component_log_risk must alias either the raw or upper component curve"
            )
    if len(supervision_aliases) != 1:
        raise ValueError("Episodes use inconsistent component supervision aliases")
    component_supervision_alias = next(iter(supervision_aliases))
    count_all_payload: dict[str, np.ndarray] = {}
    if representation == LOGIT_REPRESENTATION:
        count_all_payload = {
            "adaptation_predicted_pixel_counts": np.stack(
                [episode.adaptation_predicted_pixel_counts for episode in episodes]
            ).astype(np.int64),
            "adaptation_predicted_component_counts_raw": np.stack(
                [
                    episode.adaptation_predicted_component_counts_raw
                    for episode in episodes
                ]
            ).astype(np.int64),
            "adaptation_predicted_component_counts_upper": np.stack(
                [
                    episode.adaptation_predicted_component_counts_upper
                    for episode in episodes
                ]
            ).astype(np.int64),
            "adaptation_total_pixels": np.asarray(
                [episode.adaptation_total_pixels for episode in episodes],
                dtype=np.int64,
            ),
            "count_all_adaptation_schema_version": np.asarray(
                COUNT_ALL_ADAPTATION_SCHEMA_VERSION
            ),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        statistics=np.stack([episode.statistics.values for episode in episodes]),
        statistics_names=np.asarray(reference_names, dtype=str),
        statistics_names_sha256=np.asarray(statistics_names_sha256(reference_names)),
        pixel_log_risk=np.stack([episode.pixel_log_risk for episode in episodes]),
        component_log_risk=np.stack([episode.component_log_risk for episode in episodes]),
        component_log_risk_raw=np.stack(
            [episode.component_log_risk_raw for episode in episodes]
        ),
        component_log_risk_upper=np.stack(
            [episode.component_log_risk_upper for episode in episodes]
        ),
        pd_curve=np.stack([episode.pd_curve for episode in episodes]),
        thresholds=reference_grid,
        pixel_fp_counts=np.stack([episode.pixel_fp_counts for episode in episodes]),
        component_fp_counts=np.stack([episode.component_fp_counts for episode in episodes]),
        tp_object_counts=np.stack([episode.tp_object_counts for episode in episodes]),
        gt_object_counts=np.asarray([episode.gt_object_count for episode in episodes], dtype=np.int64),
        total_pixels=np.asarray([episode.total_pixels for episode in episodes], dtype=np.int64),
        pseudo_targets=np.asarray([episode.pseudo_target for episode in episodes], dtype=str),
        adaptation_ids=np.asarray(
            [json.dumps(episode.adaptation_ids) for episode in episodes], dtype=str
        ),
        evaluation_ids=np.asarray(
            [json.dumps(episode.evaluation_ids) for episode in episodes], dtype=str
        ),
        adaptation_sizes=np.asarray(
            [len(episode.adaptation_ids) for episode in episodes], dtype=np.int64
        ),
        evaluation_sizes=np.asarray(
            [len(episode.evaluation_ids) for episode in episodes], dtype=np.int64
        ),
        # Backward-compatible, explicitly documented alias: window == future E.
        window_ids=np.asarray([json.dumps(episode.window_ids) for episode in episodes], dtype=str),
        window_sizes=np.asarray([len(episode.window_ids) for episode in episodes], dtype=np.int64),
        window_ids_alias=np.asarray("evaluation_ids"),
        support_ids=np.asarray(
            [json.dumps(episode.adaptation_ids) for episode in episodes], dtype=str
        ),
        query_ids=np.asarray(
            [json.dumps(episode.evaluation_ids) for episode in episodes], dtype=str
        ),
        support_ids_alias=np.asarray("adaptation_ids"),
        query_ids_alias=np.asarray("evaluation_ids"),
        episode_schema_version=np.asarray(episode_schema),
        statistics_schema_version=np.asarray(statistics_schema),
        feature_schema_sha256=np.asarray(
            feature_schema_sha256(
                statistics_schema, statistics_names=reference_names
            )
        ),
        representation=np.asarray(representation),
        threshold_grid_schema_version=np.asarray(grid_schema),
        threshold_grid_version=np.asarray(grid_version),
        threshold_grid_sha256=np.asarray(grid_hash),
        threshold_grid_manifest_sha256=np.asarray(
            str(provenance.get("threshold_grid_manifest_sha256") or "")
        ),
        threshold_grid_detector_protocol=np.asarray(
            str(provenance.get("threshold_grid_detector_protocol") or "")
        ),
        threshold_grid_detector_checkpoint_sha256s=np.asarray(
            provenance.get("threshold_grid_detector_checkpoint_sha256s") or [],
            dtype=str,
        ),
        threshold_grid_outer_detector_checkpoint_sha256=np.asarray(
            str(
                provenance.get(
                    "threshold_grid_outer_detector_checkpoint_sha256"
                )
                or ""
            )
        ),
        threshold_grid_episode_detector_checkpoint_sha256s=np.asarray(
            provenance.get(
                "threshold_grid_episode_detector_checkpoint_sha256s"
            )
            or [],
            dtype=str,
        ),
        component_risk_schema_version=np.asarray(COMPONENT_RISK_SCHEMA_VERSION),
        component_log_risk_alias=np.asarray(component_supervision_alias),
        component_curve_envelope=np.asarray(
            component_supervision_alias == "component_log_risk_upper"
        ),
        component_log_risk_raw_estimator=np.asarray(
            "log10(component_fp_counts / total_megapixels + 1e-6)"
        ),
        component_log_risk_upper_estimator=np.asarray(
            "suffix_max(component_log_risk_raw)"
        ),
        risk_label_estimator=np.asarray("empirical counts/exposure with stored sufficient statistics"),
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
        **count_all_payload,
    )


def _episodes_for_files(
    files: Sequence[Path],
    pseudo_target: str,
    thresholds: np.ndarray,
    args: argparse.Namespace,
    source_reference: SourceStatisticsReference | None,
    count_all_executor: ProcessPoolExecutor | None = None,
) -> tuple[list[Episode], dict[str, object]]:
    episodes: list[Episode] = []
    windows = build_causal_windows(
        files,
        args.adaptation_window,
        args.evaluation_window,
        args.stride,
    )
    count_all_cache: dict[str, AdaptationPredictionCounts] = {}
    if args.representation == LOGIT_REPRESENTATION:
        unique_adaptation_paths = [
            path for window in windows for path in window.adaptation_files
        ]
        count_all_cache = _precompute_adaptation_prediction_counts(
            unique_adaptation_paths,
            thresholds,
            connectivity=args.connectivity,
            min_component_area=args.min_component_area,
            executor=count_all_executor,
        )
    for window in windows:
        adaptation_samples = [
            load_score_sample(
                path,
                require_mask=False,
                representation=args.representation,
            )
            for path in window.adaptation_files
        ]
        evaluation_samples = [
            load_score_sample(
                path,
                require_mask=True,
                representation=args.representation,
            )
            for path in window.evaluation_files
        ]
        episodes.append(
            build_episode(
                adaptation_samples,
                thresholds,
                pseudo_target,
                evaluation_samples=evaluation_samples,
                matching_rule=args.matching_rule,
                centroid_distance=args.centroid_distance,
                connectivity=args.connectivity,
                min_component_area=args.min_component_area,
                component_upper_envelope=True,
                source_reference=source_reference,
                representation=args.representation,
                adaptation_prediction_counts=(
                    [
                        count_all_cache[_count_all_path_key(path)]
                        for path in window.adaptation_files
                    ]
                    if args.representation == LOGIT_REPRESENTATION
                    else None
                ),
            )
        )
    role_overlap = cross_episode_role_reuse(episodes)
    allow_role_reuse = bool(
        getattr(args, "allow_cross_episode_role_reuse", False)
    )
    if role_overlap and not allow_role_reuse:
        raise ValueError(
            "Cross-episode A/E role reuse violates the formal causal contract: "
            + ", ".join(role_overlap[:5])
        )
    span = args.adaptation_window + args.evaluation_window
    if windows:
        starts = list(range(0, len(files) - span + 1, args.stride))
        last_complete_end = starts[-1] + span
        trailing_images = len(files) - last_complete_end
        used_indices = {
            index
            for start in starts
            for index in range(start, start + span)
        }
        unused_images = len(files) - len(used_indices)
        skipped_reason = None
    else:
        trailing_images = len(files)
        unused_images = len(files)
        skipped_reason = (
            f"need at least {span} images for one causal episode "
            f"(A={args.adaptation_window}, E={args.evaluation_window}); found {len(files)}"
        )
    summary: dict[str, object] = {
        "total_images": len(files),
        "episodes": len(episodes),
        "required_images_per_episode": span,
        "adaptation_window": args.adaptation_window,
        "evaluation_window": args.evaluation_window,
        "stride": args.stride,
        "trailing_images_not_in_complete_episode": trailing_images,
        "images_not_used_by_any_complete_episode": unused_images,
        "cross_episode_role_reuse_count": len(role_overlap),
        "cross_episode_role_reuse_ids": list(role_overlap),
        "count_all_unique_adaptation_files_precomputed": len(count_all_cache),
        "count_all_workers": (
            int(getattr(args, "count_all_workers", 1))
            if args.representation == LOGIT_REPRESENTATION
            else None
        ),
        "skipped_reason": skipped_reason,
    }
    return episodes, summary


def _split_complete_pseudo_domains(
    episodes_by_target: Mapping[str, Sequence[Episode]],
    window_summaries: Mapping[str, Mapping[str, object]],
    validation_domain: str,
) -> tuple[list[Episode], list[Episode], dict[str, dict[str, object]]]:
    """Assign prebuilt pseudo-domain episodes to one complete-domain split.

    Paired LODO output calls this function twice with opposite validation
    domains.  Episode construction remains outside the function, so expensive
    threshold/component matching is performed exactly once per pseudo domain.
    """

    if validation_domain not in episodes_by_target:
        raise ValueError("validation_domain is not one of the pseudo targets")
    if set(episodes_by_target) != set(window_summaries):
        raise ValueError(
            "Episode and window-summary pseudo-target domains differ"
        )
    train_episodes: list[Episode] = []
    val_episodes: list[Episode] = []
    split_summary: dict[str, dict[str, object]] = {}
    for target, selected in episodes_by_target.items():
        is_validation = target == validation_domain
        destination = val_episodes if is_validation else train_episodes
        destination.extend(selected)
        split_summary[target] = {
            **dict(window_summaries[target]),
            "split": "validation_domain" if is_validation else "train_domain",
            "train_episodes": 0 if is_validation else len(selected),
            "val_episodes": len(selected) if is_validation else 0,
        }
    return train_episodes, val_episodes, split_summary


def _write_episode_output(
    *,
    output_dir: Path,
    train_episodes: Sequence[Episode],
    val_episodes: Sequence[Episode],
    provenance: Mapping[str, object],
    episode_schema: str,
    statistics_schema: str,
    feature_schema_hash: str,
    grid_version: str,
    grid_hash: str,
    grid_manifest_sha256: str | None,
    formal_protocol_ready: bool,
) -> str:
    """Write one train/validation direction with split-specific provenance."""

    output_provenance = {
        **dict(provenance),
        "num_train_episodes": len(train_episodes),
        "num_val_episodes": len(val_episodes),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if train_episodes:
        _pack_episodes(
            train_episodes,
            output_dir / "train.npz",
            {
                **output_provenance,
                "archive_split": "train",
                "num_archive_episodes": len(train_episodes),
            },
        )
    if val_episodes:
        _pack_episodes(
            val_episodes,
            output_dir / "val.npz",
            {
                **output_provenance,
                "archive_split": "validation",
                "num_archive_episodes": len(val_episodes),
            },
        )
    if not train_episodes or not val_episodes:
        status = "insufficient_complete_windows"
    elif formal_protocol_ready:
        status = "complete"
    else:
        status = "complete_diagnostic_only"
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                **output_provenance,
                "episode_schema_version": episode_schema,
                "statistics_schema_version": statistics_schema,
                "representation": output_provenance["representation"],
                "feature_schema_sha256": feature_schema_hash,
                "threshold_grid_version": grid_version,
                "threshold_grid_sha256": grid_hash,
                "threshold_grid_manifest_sha256": grid_manifest_sha256,
                "status": status,
                "formal_protocol_eligible": bool(
                    status == "complete" and formal_protocol_ready
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return status


def _incomplete_episode_error(
    train_episodes: Sequence[Episode],
    val_episodes: Sequence[Episode],
    *,
    adaptation_window: int,
    evaluation_window: int,
    output_dir: Path,
) -> ValueError | None:
    if train_episodes and val_episodes:
        return None
    missing = []
    if not train_episodes:
        missing.append("training")
    if not val_episodes:
        missing.append("validation")
    return ValueError(
        "No complete causal episodes for "
        + " and ".join(missing)
        + f" archives (each needs A+E={adaptation_window + evaluation_window} "
        + "images after the image/domain split). See "
        + f"{output_dir / 'manifest.json'} split_summary."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-map-dir", action="append", required=True)
    parser.add_argument("--pseudo-target", action="append")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--paired-output-dir",
        help=(
            "Optional reverse-LODO output directory. This reuses one episode "
            "construction pass to write both validation-domain directions and "
            "requires exactly two pseudo targets plus --validation-domain."
        ),
    )
    parser.add_argument("--threshold-grid")
    parser.add_argument(
        "--threshold-grid-manifest",
        help=(
            "v4 raw-logit threshold_grid.json (or its directory). Required "
            "with --representation raw_logit_float32"
        ),
    )
    parser.add_argument(
        "--representation",
        choices=SUPPORTED_REPRESENTATIONS,
        default=PROBABILITY_REPRESENTATION,
        help="Canonical score/threshold representation (default: legacy probability)",
    )
    parser.add_argument(
        "--expected-split-role",
        choices=("train", "test"),
        default="train",
        help=(
            "Official pseudo-target split required in every score manifest "
            "(default: train, the source-only main risk-curve protocol)"
        ),
    )
    parser.add_argument(
        "--source-reference",
        help="Optional NPZ with source centres/precision for domain-distance features",
    )
    parser.add_argument(
        "--adaptation-window",
        type=int,
        help="Number of earlier, label-free images used for statistics (default: 32)",
    )
    parser.add_argument(
        "--evaluation-window",
        type=int,
        help=(
            "Number of immediately following images used only for curve labels "
            "(default: 1, matching the formal image-level deployment unit)"
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        help=(
            "DEPRECATED compatibility alias: sets both --adaptation-window and "
            "--evaluation-window; it never enables same-window/transductive labels"
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        help="Episode start stride (default: adaptation + evaluation window sizes)",
    )
    parser.add_argument(
        "--validation-domain",
        help="Recommended formal protocol: reserve this complete pseudo-target domain for validation",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--matching-rule", choices=("overlap", "centroid"), default="overlap")
    parser.add_argument("--centroid-distance", type=float, default=3.0)
    parser.add_argument("--connectivity", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument(
        "--count-all-workers",
        type=int,
        default=1,
        help=(
            "Spawn workers used to precompute unique raw-logit A-window Count-all "
            "curves (default: 1, sequential and backward compatible)"
        ),
    )
    parser.add_argument(
        "--allow-unverified-fold-provenance",
        action="store_true",
        help="Diagnostic only: permit score maps without detector source-domain provenance",
    )
    parser.add_argument(
        "--allow-cross-episode-role-reuse",
        action="store_true",
        help=(
            "Diagnostic only: permit stride<A+E or an image changing A/E roles "
            "across episodes; the output manifest is ineligible for formal claims"
        ),
    )
    return parser.parse_args()


def _validate_count_all_workers(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("--count-all-workers must be a positive integer")
    return value


def _resolve_window_config(args: argparse.Namespace) -> None:
    """Resolve causal window CLI values in-place with strict compatibility checks."""

    legacy = args.window_size
    if legacy is not None:
        warnings.warn(
            "--window-size is deprecated; it now maps to equally sized, disjoint "
            "adaptation and evaluation windows",
            FutureWarning,
            stacklevel=2,
        )
        if args.adaptation_window is not None and args.adaptation_window != legacy:
            raise ValueError("--window-size conflicts with --adaptation-window")
        if args.evaluation_window is not None and args.evaluation_window != legacy:
            raise ValueError("--window-size conflicts with --evaluation-window")
        args.adaptation_window = legacy
        args.evaluation_window = legacy
    args.adaptation_window = 32 if args.adaptation_window is None else args.adaptation_window
    # The formal downstream selector assigns one predicted curve to one future
    # image. Keep that image-level target unit as the default so a checkpoint
    # cannot silently learn an E-image aggregate and then be read per image.
    args.evaluation_window = 1 if args.evaluation_window is None else args.evaluation_window
    if args.adaptation_window < 1 or args.evaluation_window < 1:
        raise ValueError("--adaptation-window and --evaluation-window must be positive")
    if args.stride is None:
        args.stride = args.adaptation_window + args.evaluation_window
    if args.stride < 1:
        raise ValueError("--stride must be positive")
    span = args.adaptation_window + args.evaluation_window
    if (
        args.stride < span
        and not bool(getattr(args, "allow_cross_episode_role_reuse", False))
    ):
        raise ValueError(
            f"Formal causal episodes require --stride >= A+E ({span}); "
            "use --allow-cross-episode-role-reuse only for diagnostic output"
        )


def main() -> None:
    args = _parse_args()
    _resolve_window_config(args)
    args.count_all_workers = _validate_count_all_workers(args.count_all_workers)
    if (
        args.representation != LOGIT_REPRESENTATION
        and args.count_all_workers != 1
    ):
        raise ValueError(
            "--count-all-workers > 1 is only valid with raw_logit_float32"
        )
    if not 0.0 < args.val_fraction < 1.0 and args.validation_domain is None:
        raise ValueError("--val-fraction must lie in (0, 1) without --validation-domain")
    roots = [Path(item) for item in args.score_map_dir]
    if args.pseudo_target is None:
        targets = [root.name for root in roots]
    else:
        targets = args.pseudo_target
    if len(targets) != len(roots):
        raise ValueError("Provide exactly one --pseudo-target per --score-map-dir")
    if len(set(targets)) != len(targets):
        raise ValueError("pseudo-target names must be unique")
    if args.validation_domain is not None and args.validation_domain not in targets:
        raise ValueError("--validation-domain is not one of the pseudo targets")
    if args.paired_output_dir is not None:
        if args.validation_domain is None:
            raise ValueError(
                "--paired-output-dir requires --validation-domain"
            )
        if len(targets) != 2:
            raise ValueError(
                "--paired-output-dir requires exactly two pseudo targets"
            )
        if len({_domain_key(target) for target in targets}) != 2:
            raise ValueError(
                "--paired-output-dir requires two distinct pseudo-target domains"
            )
        output_path = Path(args.output_dir).expanduser().resolve()
        paired_output_path = Path(args.paired_output_dir).expanduser().resolve()
        if output_path == paired_output_path:
            raise ValueError(
                "--paired-output-dir must differ from --output-dir"
            )
    logit_grid_artifact = None
    if args.representation == LOGIT_REPRESENTATION:
        grid_artifact_path = args.threshold_grid_manifest or args.threshold_grid
        if not grid_artifact_path:
            raise ValueError(
                "Raw-logit episode generation requires --threshold-grid-manifest"
            )
        if args.threshold_grid_manifest and args.threshold_grid:
            raise ValueError(
                "Use only --threshold-grid-manifest for a raw-logit grid artifact"
            )
        logit_grid_artifact = load_logit_grid_artifact(grid_artifact_path)
        thresholds = logit_grid_artifact.thresholds
    else:
        if args.threshold_grid_manifest:
            raise ValueError(
                "--threshold-grid-manifest is only valid for raw-logit episodes"
            )
        thresholds = load_threshold_grid(args.threshold_grid)
    source_reference = load_source_reference(args.source_reference) if args.source_reference else None
    if source_reference is not None:
        assert_source_reference_excludes_pseudo_targets(source_reference, targets)
    train_episodes: list[Episode] = []
    val_episodes: list[Episode] = []
    split_summary: dict[str, dict[str, object]] = {}
    complete_domain_episodes: dict[str, list[Episode]] = {}
    complete_domain_summaries: dict[str, dict[str, object]] = {}
    formal_artifact_mode = not bool(args.allow_unverified_fold_provenance)
    artifact_inputs: list[
        tuple[dict[str, object] | None, list[Path], dict[str, object]]
    ] = []
    for root, target in zip(roots, targets):
        if formal_artifact_mode:
            verified_manifest, verified_paths, integrity = verify_score_map_directory(
                root,
                require_integrity=True,
                require_masks=True,
            )
            if verified_manifest is None or not integrity.get("verified", False):
                raise ValueError(
                    f"Formal score artifact failed integrity verification: {root}"
                )
            if verified_manifest.get("labels_loaded") is not True:
                raise ValueError(
                    "Formal risk-curve episodes require labels_loaded=true for "
                    f"every pseudo target: {target}"
                )
            if args.representation == LOGIT_REPRESENTATION:
                raw_contract = {
                    "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
                    "logit_dtype": RAW_LOGIT_DTYPE,
                    "probability_transform": "sigmoid",
                    "probability_clipping": "none",
                    "inference_autocast_enabled": False,
                }
                mismatches = [
                    field
                    for field, expected in raw_contract.items()
                    if verified_manifest.get(field) != expected
                ]
                if mismatches:
                    raise ValueError(
                        "Raw-logit episode input lacks the complete precision contract: "
                        + ", ".join(mismatches)
                    )
            manifest_for_audit: dict[str, object] | None = dict(verified_manifest)
            paths_for_episodes = list(verified_paths)
            integrity_audit = {
                "pseudo_target": target,
                "score_dir": str(root.expanduser().resolve()),
                "labels_loaded": True,
                **integrity,
            }
        else:
            # Preserve legacy score directories only behind the explicitly
            # diagnostic flag.  Formal code must never resolve files again via
            # score_files after the verifier has established authoritative order.
            manifest_for_audit = None
            paths_for_episodes = score_files(root)
            integrity_audit = {
                "pseudo_target": target,
                "score_dir": str(root.expanduser().resolve()),
                "labels_loaded": None,
                "verified": False,
                "mask_alignment_verified": False,
                "diagnostic_reason": "allow_unverified_fold_provenance",
                "num_records": len(paths_for_episodes),
            }
        artifact_inputs.append(
            (manifest_for_audit, paths_for_episodes, integrity_audit)
        )

    fold_audits = [
        audit_fold_score_manifest(
            root,
            target,
            expected_split_role=args.expected_split_role,
            verified_manifest=artifact[0],
        )
        for root, target, artifact in zip(roots, targets, artifact_inputs)
    ]
    unverified = [audit for audit in fold_audits if not audit["verified"]]
    if unverified and not args.allow_unverified_fold_provenance:
        reasons = ", ".join(
            f"{audit['pseudo_target']}:{audit['reason']}" for audit in unverified
        )
        raise ValueError(
            "LODO episode construction requires verifiable detector fold provenance "
            f"({reasons}); re-export with repeated --source-dataset or use "
            "--allow-unverified-fold-provenance for diagnostic-only runs"
        )
    if args.representation == LOGIT_REPRESENTATION:
        assert logit_grid_artifact is not None
        grid_source_keys = set(logit_grid_artifact.manifest["source_domain_keys"])
        episode_target_keys = {_domain_key(target) for target in targets}
        if episode_target_keys != grid_source_keys:
            raise ValueError(
                "Raw-logit episode pseudo-target domains do not exactly match "
                "the global source-grid domains"
            )
        grid_checkpoint_hashes = set(
            logit_grid_artifact.manifest[
                "episode_detector_checkpoint_sha256s"
            ]
        )
        episode_checkpoint_hashes = {
            str(audit.get("detector_weight_sha256")) for audit in fold_audits
        }
        if episode_checkpoint_hashes != grid_checkpoint_hashes:
            raise ValueError(
                "Raw-logit episode detector checkpoints do not exactly match "
                "the inner pseudo-target detector folds bound by the global "
                "source grid"
            )
        checkpoint_to_source_set = {
            str(item["detector_checkpoint_sha256"]): set(
                map(str, item["source_domain_keys"])
            )
            for item in logit_grid_artifact.manifest["detector_folds"]
            if item["role"] == "inner_pseudo_target_detector"
        }
        for target, audit in zip(targets, fold_audits):
            checkpoint_hash = str(audit["detector_weight_sha256"])
            expected_source_keys = checkpoint_to_source_set.get(checkpoint_hash)
            actual_source_keys = {
                _domain_key(str(value))
                for value in audit.get("source_datasets", [])
            }
            expected_held_out = set(grid_source_keys).difference(
                expected_source_keys or set()
            )
            if (
                expected_source_keys is None
                or actual_source_keys != expected_source_keys
                or expected_held_out != {_domain_key(target)}
            ):
                raise ValueError(
                    "Pseudo-target detector source domain does not match the "
                    "inner fold bound by the global source grid"
                )

    count_all_executor: ProcessPoolExecutor | None = None
    if (
        args.representation == LOGIT_REPRESENTATION
        and args.count_all_workers > 1
    ):
        count_all_executor = ProcessPoolExecutor(
            max_workers=args.count_all_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialise_count_all_worker,
            initargs=(thresholds, args.connectivity, args.min_component_area),
        )
    try:
        for root, target, (_, files, _) in zip(roots, targets, artifact_inputs):
            if args.validation_domain is not None:
                selected, window_summary = _episodes_for_files(
                    files,
                    target,
                    thresholds,
                    args,
                    source_reference,
                    count_all_executor,
                )
                complete_domain_episodes[target] = selected
                complete_domain_summaries[target] = window_summary
                continue
            boundary = int(round(len(files) * (1.0 - args.val_fraction)))
            train_files = files[:boundary]
            val_files = files[boundary:]
            # Windows are created only after the image-level split, preventing overlap leakage.
            train_selected, train_summary = _episodes_for_files(
                train_files,
                target,
                thresholds,
                args,
                source_reference,
                count_all_executor,
            )
            val_selected, val_summary = _episodes_for_files(
                val_files,
                target,
                thresholds,
                args,
                source_reference,
                count_all_executor,
            )
            train_episodes.extend(train_selected)
            val_episodes.extend(val_selected)
            split_summary[target] = {
                "train_images": len(train_files),
                "val_images": len(val_files),
                "train_episodes": len(train_selected),
                "val_episodes": len(val_selected),
                "train_windowing": train_summary,
                "val_windowing": val_summary,
            }
    finally:
        if count_all_executor is not None:
            count_all_executor.shutdown(wait=True, cancel_futures=True)
    if args.validation_domain is not None:
        train_episodes, val_episodes, split_summary = (
            _split_complete_pseudo_domains(
                complete_domain_episodes,
                complete_domain_summaries,
                args.validation_domain,
            )
        )
    global_role_overlap = cross_episode_role_reuse(train_episodes + val_episodes)
    diagnostic_role_reuse = bool(args.allow_cross_episode_role_reuse)
    if global_role_overlap and not diagnostic_role_reuse:
        raise ValueError(
            "Cross-split episode A/E role reuse violates the formal causal contract: "
            + ", ".join(global_role_overlap[:5])
        )
    formal_causal_contract = bool(
        not diagnostic_role_reuse
        and args.stride >= args.adaptation_window + args.evaluation_window
        and not global_role_overlap
    )
    score_artifact_integrity_audits = [artifact[2] for artifact in artifact_inputs]
    score_artifact_integrity_verified = bool(
        formal_artifact_mode
        and score_artifact_integrity_audits
        and all(audit.get("verified", False) for audit in score_artifact_integrity_audits)
    )
    logit_grid_verified = bool(
        args.representation != LOGIT_REPRESENTATION
        or logit_grid_artifact is not None
    )
    formal_protocol_ready = bool(
        formal_causal_contract
        and score_artifact_integrity_verified
        and logit_grid_verified
        and not unverified
        and not args.allow_unverified_fold_provenance
    )
    pseudo_target_splits = {
        target: audit.get("split_role")
        for target, audit in zip(targets, fold_audits)
    }
    observed_split_roles = set(pseudo_target_splits.values())
    pseudo_target_split = (
        next(iter(observed_split_roles)) if len(observed_split_roles) == 1 else None
    )
    if args.representation == LOGIT_REPRESENTATION:
        assert logit_grid_artifact is not None
        grid_hash = logit_grid_artifact.semantic_sha256
        grid_version = LOGIT_GRID_SCHEMA_VERSION
        grid_manifest_sha256 = hashlib.sha256(
            logit_grid_artifact.manifest_path.read_bytes()
        ).hexdigest()
        grid_manifest_path: str | None = str(logit_grid_artifact.manifest_path)
        grid_source_domains = list(
            logit_grid_artifact.manifest["source_domain_keys"]
        )
        grid_outer_target_key: str | None = str(
            logit_grid_artifact.manifest["outer_target_key"]
        )
        grid_detector_protocol: str | None = str(
            logit_grid_artifact.manifest["grid_detector_protocol"]
        )
        grid_detector_checkpoint_hashes = list(
            logit_grid_artifact.manifest["detector_checkpoint_sha256s"]
        )
        grid_outer_detector_checkpoint_hash: str | None = str(
            logit_grid_artifact.manifest[
                "outer_detector_checkpoint_sha256"
            ]
        )
        grid_episode_detector_checkpoint_hashes = list(
            logit_grid_artifact.manifest[
                "episode_detector_checkpoint_sha256s"
            ]
        )
    else:
        grid_hash = threshold_grid_sha256(thresholds)
        grid_version = threshold_grid_version(thresholds)
        grid_manifest_sha256 = None
        grid_manifest_path = None
        grid_source_domains = []
        grid_outer_target_key = None
        grid_detector_protocol = None
        grid_detector_checkpoint_hashes = []
        grid_outer_detector_checkpoint_hash = None
        grid_episode_detector_checkpoint_hashes = []
    statistics_schema = (
        LOGIT_STATISTICS_SCHEMA_VERSION
        if args.representation == LOGIT_REPRESENTATION
        else STATISTICS_SCHEMA_VERSION
    )
    episode_schema = (
        LOGIT_EPISODE_SCHEMA_VERSION
        if args.representation == LOGIT_REPRESENTATION
        else EPISODE_SCHEMA_VERSION
    )
    if train_episodes:
        episode_feature_names = train_episodes[0].statistics.names
    elif val_episodes:
        episode_feature_names = val_episodes[0].statistics.names
    else:
        episode_feature_names = tuple(
            logit_feature_names()
            if statistics_schema == LOGIT_STATISTICS_SCHEMA_VERSION
            else base_feature_names()
        )
    episode_feature_hash = feature_schema_sha256(
        statistics_schema,
        statistics_names=episode_feature_names,
    )
    provenance: dict[str, object] = {
        "score_map_dirs": [str(root) for root in roots],
        "pseudo_targets": targets,
        "expected_split_role": args.expected_split_role,
        "pseudo_target_split": pseudo_target_split,
        "pseudo_target_splits": pseudo_target_splits,
        "validation_domain": args.validation_domain,
        "protocol": "causal_adaptation_then_future_evaluation",
        "representation": args.representation,
        "prediction_rule": (
            "prediction = (raw_logits >= threshold)"
            if args.representation == LOGIT_REPRESENTATION
            else "prediction = (sigmoid_probability >= threshold)"
        ),
        "adaptation_window": args.adaptation_window,
        "evaluation_window": args.evaluation_window,
        "deprecated_window_size_alias": args.window_size,
        "stride": args.stride,
        "matching_rule": args.matching_rule,
        "centroid_distance": args.centroid_distance,
        "connectivity": args.connectivity,
        "min_component_area": args.min_component_area,
        "source_reference": args.source_reference,
        "source_reference_sha256": (
            hashlib.sha256(Path(args.source_reference).read_bytes()).hexdigest()
            if args.source_reference
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
        "fold_provenance_audits": fold_audits,
        "fold_provenance_verified": not unverified,
        "score_artifact_integrity_audits": score_artifact_integrity_audits,
        "score_artifact_integrity_verified": score_artifact_integrity_verified,
        "allow_unverified_fold_provenance": bool(
            args.allow_unverified_fold_provenance
        ),
        "allow_cross_episode_role_reuse": diagnostic_role_reuse,
        "cross_episode_role_reuse_detected": bool(global_role_overlap),
        "cross_episode_role_reuse_ids": list(global_role_overlap),
        "formal_causal_contract_verified": formal_causal_contract,
        "protocol_scope": (
            "formal_causal" if formal_protocol_ready else "diagnostic_only"
        ),
        "split_summary": split_summary,
        "label_leakage_guard": (
            "statistics use only A scores/gray in the declared representation; "
            "labels use only future E masks; "
            "A and E IDs are disjoint; train/val image/domain split precedes episode construction"
        ),
        "statistics_sample_role": "adaptation_window_A_label_free",
        "risk_label_sample_role": "immediately_following_evaluation_window_E",
        "window_ids_compatibility_alias": "evaluation_ids",
        "threshold_grid_sha256": grid_hash,
        "threshold_grid_version": grid_version,
        "threshold_grid_schema_version": grid_version,
        "threshold_grid_manifest": grid_manifest_path,
        "threshold_grid_manifest_sha256": grid_manifest_sha256,
        "threshold_grid_source_domains": grid_source_domains,
        "threshold_grid_outer_target_key": grid_outer_target_key,
        "threshold_grid_outer_target_excluded": (
            bool(logit_grid_artifact.manifest["outer_target_excluded"])
            if logit_grid_artifact is not None
            else None
        ),
        "threshold_grid_detector_protocol": grid_detector_protocol,
        "threshold_grid_detector_checkpoint_sha256s": (
            grid_detector_checkpoint_hashes
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": (
            grid_outer_detector_checkpoint_hash
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": (
            grid_episode_detector_checkpoint_hashes
        ),
        "feature_schema_sha256": episode_feature_hash,
    }
    if args.representation == LOGIT_REPRESENTATION:
        provenance.update(_COUNT_ALL_PROVENANCE)
        provenance.update(
            {
                "count_all_workers": args.count_all_workers,
                "count_all_worker_start_method": (
                    "spawn" if args.count_all_workers > 1 else "sequential"
                ),
                "count_all_parallelism_semantic": (
                    "execution_only_not_part_of_grid_feature_or_count_contract"
                ),
            }
        )
    output_dir = Path(args.output_dir)
    paired_output_dir = (
        Path(args.paired_output_dir)
        if args.paired_output_dir is not None
        else None
    )
    if paired_output_dir is None:
        primary_provenance = {
            **provenance,
            "paired_lodo": False,
            "paired_lodo_role": None,
            "paired_lodo_peer_output_dir": None,
            "paired_lodo_validation_domains": [],
        }
    else:
        assert args.validation_domain is not None
        reverse_validation_domain = next(
            target for target in targets if target != args.validation_domain
        )
        paired_validation_domains = [
            args.validation_domain,
            reverse_validation_domain,
        ]
        primary_provenance = {
            **provenance,
            "paired_lodo": True,
            "paired_lodo_role": "primary",
            "paired_lodo_peer_output_dir": str(
                paired_output_dir.expanduser().resolve()
            ),
            "paired_lodo_validation_domains": paired_validation_domains,
        }

    _write_episode_output(
        output_dir=output_dir,
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        provenance=primary_provenance,
        episode_schema=episode_schema,
        statistics_schema=statistics_schema,
        feature_schema_hash=episode_feature_hash,
        grid_version=grid_version,
        grid_hash=grid_hash,
        grid_manifest_sha256=grid_manifest_sha256,
        formal_protocol_ready=formal_protocol_ready,
    )
    incomplete_errors: list[ValueError] = []
    primary_incomplete = _incomplete_episode_error(
        train_episodes,
        val_episodes,
        adaptation_window=args.adaptation_window,
        evaluation_window=args.evaluation_window,
        output_dir=output_dir,
    )
    if primary_incomplete is not None:
        incomplete_errors.append(primary_incomplete)
    print(
        f"wrote {len(train_episodes)} train and {len(val_episodes)} "
        f"validation episodes to {output_dir}"
    )

    if paired_output_dir is not None:
        reverse_train_episodes, reverse_val_episodes, reverse_split_summary = (
            _split_complete_pseudo_domains(
                complete_domain_episodes,
                complete_domain_summaries,
                reverse_validation_domain,
            )
        )
        reverse_provenance = {
            **provenance,
            "validation_domain": reverse_validation_domain,
            "split_summary": reverse_split_summary,
            "paired_lodo": True,
            "paired_lodo_role": "reverse",
            "paired_lodo_peer_output_dir": str(
                output_dir.expanduser().resolve()
            ),
            "paired_lodo_validation_domains": paired_validation_domains,
        }
        _write_episode_output(
            output_dir=paired_output_dir,
            train_episodes=reverse_train_episodes,
            val_episodes=reverse_val_episodes,
            provenance=reverse_provenance,
            episode_schema=episode_schema,
            statistics_schema=statistics_schema,
            feature_schema_hash=episode_feature_hash,
            grid_version=grid_version,
            grid_hash=grid_hash,
            grid_manifest_sha256=grid_manifest_sha256,
            formal_protocol_ready=formal_protocol_ready,
        )
        reverse_incomplete = _incomplete_episode_error(
            reverse_train_episodes,
            reverse_val_episodes,
            adaptation_window=args.adaptation_window,
            evaluation_window=args.evaluation_window,
            output_dir=paired_output_dir,
        )
        if reverse_incomplete is not None:
            incomplete_errors.append(reverse_incomplete)
        print(
            f"wrote {len(reverse_train_episodes)} train and "
            f"{len(reverse_val_episodes)} validation episodes to "
            f"{paired_output_dir}"
        )

    if incomplete_errors:
        if len(incomplete_errors) == 1:
            raise incomplete_errors[0]
        raise ValueError(
            "Paired LODO outputs are incomplete: "
            + "; ".join(str(error) for error in incomplete_errors)
        )


if __name__ == "__main__":
    main()
