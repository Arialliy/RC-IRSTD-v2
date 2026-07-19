"""Deterministic connected-component matching for IRSTD evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # SciPy is a project dependency, but keep source-only fallback portability.
    from scipy.ndimage import label as _SCIPY_LABEL
except ImportError:  # pragma: no cover - exercised by the explicit fallback test.
    _SCIPY_LABEL = None


@dataclass(frozen=True)
class MatchResult:
    """Counts for one thresholded prediction/ground-truth pair.

    ``matched_pairs`` stores ``(prediction_label, ground_truth_label)`` using
    one-based connected-component labels.
    """

    num_gt: int
    num_tp_objects: int
    num_fp_components: int
    num_fp_pixels: int
    matched_pairs: list[tuple[int, int]]


def _as_binary_2d(array: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array (or 1xHxW), got {value.shape}")
    if not np.issubdtype(value.dtype, np.number) and value.dtype != np.bool_:
        raise TypeError(f"{name} must have a numeric or boolean dtype")
    # Thresholded predictions and masks are already boolean in all formal
    # evaluators.  Avoid sorting every HxW boolean array with ``np.unique`` on
    # this hot path; non-boolean callers retain the full finite/binary audit.
    if value.dtype != np.bool_:
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or infinity")
        unique = np.unique(value)
        if not np.isin(unique, (0, 1, False, True)).all():
            raise ValueError(
                f"{name} must already be binary; threshold continuous scores first"
            )
    return np.ascontiguousarray(value.astype(bool, copy=False))


def _neighbor_offsets(connectivity: int) -> tuple[tuple[int, int], ...]:
    # Accept both skimage convention (1/2) and explicit neighbor counts (4/8).
    if connectivity in {1, 4}:
        return ((-1, 0), (0, -1), (0, 1), (1, 0))
    if connectivity in {2, 8}:
        return (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    raise ValueError("connectivity must be 1/2 (skimage) or 4/8")


def _connected_components_python(
    image: np.ndarray,
    *,
    offsets: tuple[tuple[int, int], ...],
    min_component_area: int,
) -> tuple[np.ndarray, int]:
    """Reference flood-fill implementation for environments without SciPy."""

    height, width = image.shape
    visited = np.zeros_like(image, dtype=bool)
    labels = np.zeros(image.shape, dtype=np.int32)
    next_label = 0

    for row, col in np.argwhere(image):
        row, col = int(row), int(col)
        if visited[row, col]:
            continue
        stack = [(row, col)]
        visited[row, col] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            current_row, current_col = stack.pop()
            pixels.append((current_row, current_col))
            for delta_row, delta_col in offsets:
                neighbor_row = current_row + delta_row
                neighbor_col = current_col + delta_col
                if (
                    0 <= neighbor_row < height
                    and 0 <= neighbor_col < width
                    and image[neighbor_row, neighbor_col]
                    and not visited[neighbor_row, neighbor_col]
                ):
                    visited[neighbor_row, neighbor_col] = True
                    stack.append((neighbor_row, neighbor_col))
        if len(pixels) < min_component_area:
            continue
        next_label += 1
        rows, cols = zip(*pixels)
        labels[np.asarray(rows), np.asarray(cols)] = next_label
    return labels, next_label


def _connected_components_scipy(
    image: np.ndarray,
    *,
    connectivity: int,
    min_component_area: int,
) -> tuple[np.ndarray, int]:
    """Fast labeling with deterministic post-filter label compaction."""

    if connectivity in {1, 4}:
        structure = np.asarray(
            ((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8
        )
    else:
        structure = np.ones((3, 3), dtype=np.uint8)

    raw_labels, raw_count = _SCIPY_LABEL(image, structure=structure)
    labels = np.asarray(raw_labels, dtype=np.int32)
    raw_count = int(raw_count)
    if raw_count == 0:
        return np.ascontiguousarray(labels), 0

    component_areas = np.bincount(labels.ravel(), minlength=raw_count + 1)
    retained = component_areas >= min_component_area
    retained[0] = False
    retained_count = int(np.count_nonzero(retained))
    if retained_count == raw_count:
        return np.ascontiguousarray(labels), raw_count

    # scipy.ndimage.label assigns component IDs in forward scan order.  Compact
    # those IDs after area filtering so labels retain the reference contract:
    # one-based, consecutive, and ordered by each component's first pixel.
    remap = np.zeros(raw_count + 1, dtype=np.int32)
    remap[retained] = np.arange(1, retained_count + 1, dtype=np.int32)
    return np.ascontiguousarray(remap[labels]), retained_count


def connected_components(
    binary: np.ndarray,
    *,
    connectivity: int = 2,
    min_component_area: int = 1,
) -> tuple[np.ndarray, int]:
    """Label a binary array with deterministic one-based component IDs.

    SciPy supplies the normal fast path.  The original pure-Python flood fill
    remains the reference and portability fallback, with identical 4/8
    connectivity and ``min_component_area`` semantics.
    """

    image = _as_binary_2d(binary, "binary")
    if isinstance(min_component_area, bool) or not isinstance(min_component_area, int):
        raise TypeError("min_component_area must be an integer")
    if min_component_area <= 0:
        raise ValueError("min_component_area must be positive")
    offsets = _neighbor_offsets(connectivity)
    if _SCIPY_LABEL is not None:
        return _connected_components_scipy(
            image,
            connectivity=connectivity,
            min_component_area=min_component_area,
        )
    return _connected_components_python(
        image,
        offsets=offsets,
        min_component_area=min_component_area,
    )


def _component_centroids(labels: np.ndarray, count: int) -> dict[int, np.ndarray]:
    centroids: dict[int, np.ndarray] = {}
    for label in range(1, count + 1):
        coordinates = np.argwhere(labels == label)
        centroids[label] = coordinates.mean(axis=0)
    return centroids


def _maximum_cardinality_pairs(
    adjacency: dict[int, list[int]],
) -> list[tuple[int, int]]:
    """Return a deterministic maximum-cardinality one-to-one matching."""

    ground_truth_to_prediction: dict[int, int] = {}

    def augment(prediction_label: int, seen: set[int]) -> bool:
        for ground_truth_label in adjacency.get(prediction_label, []):
            if ground_truth_label in seen:
                continue
            seen.add(ground_truth_label)
            previous = ground_truth_to_prediction.get(ground_truth_label)
            if previous is None or augment(previous, seen):
                ground_truth_to_prediction[ground_truth_label] = prediction_label
                return True
        return False

    for prediction_label in sorted(adjacency):
        augment(prediction_label, set())
    return sorted(
        (prediction_label, ground_truth_label)
        for ground_truth_label, prediction_label in ground_truth_to_prediction.items()
    )


def match_components(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
) -> MatchResult:
    """Match thresholded components with overlap or centroid-distance rules.

    Matching is one-to-one and maximum-cardinality.  Overlap candidates are
    preferred by larger intersection; centroid candidates by shorter distance.
    ``num_fp_pixels`` is the pixel-level false-positive count ``prediction & ~GT``
    and therefore remains monotone under increasing score thresholds.
    """

    predicted = _as_binary_2d(prediction, "prediction")
    target = _as_binary_2d(ground_truth, "ground_truth")
    if predicted.shape != target.shape:
        raise ValueError(
            f"prediction and ground_truth shapes differ: {predicted.shape} vs {target.shape}"
        )
    if rule not in {"overlap", "centroid"}:
        raise ValueError("rule must be 'overlap' or 'centroid'")
    if not np.isfinite(centroid_distance) or centroid_distance < 0:
        raise ValueError("centroid_distance must be finite and non-negative")

    prediction_labels, num_predictions = connected_components(
        predicted,
        connectivity=connectivity,
        min_component_area=min_component_area,
    )
    # Never filter tiny ground-truth targets when suppressing predicted speckles.
    target_labels, num_ground_truth = connected_components(
        target,
        connectivity=connectivity,
        min_component_area=1,
    )

    scored_adjacency: dict[int, list[tuple[float, int]]] = {
        label: [] for label in range(1, num_predictions + 1)
    }
    if rule == "overlap":
        joint = (prediction_labels > 0) & (target_labels > 0)
        if joint.any():
            encoded = (
                prediction_labels[joint].astype(np.int64) * (num_ground_truth + 1)
                + target_labels[joint].astype(np.int64)
            )
            keys, counts = np.unique(encoded, return_counts=True)
            for key, count in zip(keys.tolist(), counts.tolist()):
                prediction_label = int(key // (num_ground_truth + 1))
                ground_truth_label = int(key % (num_ground_truth + 1))
                scored_adjacency[prediction_label].append(
                    (-float(count), ground_truth_label)
                )
    else:
        prediction_centroids = _component_centroids(prediction_labels, num_predictions)
        target_centroids = _component_centroids(target_labels, num_ground_truth)
        for prediction_label, prediction_centroid in prediction_centroids.items():
            for ground_truth_label, target_centroid in target_centroids.items():
                distance = float(np.linalg.norm(prediction_centroid - target_centroid))
                if distance <= centroid_distance:
                    scored_adjacency[prediction_label].append(
                        (distance, ground_truth_label)
                    )

    adjacency = {
        prediction_label: [
            ground_truth_label
            for _, ground_truth_label in sorted(candidates, key=lambda item: (item[0], item[1]))
        ]
        for prediction_label, candidates in scored_adjacency.items()
    }
    matched_pairs = _maximum_cardinality_pairs(adjacency)
    num_matches = len(matched_pairs)
    evaluated_prediction = prediction_labels > 0
    return MatchResult(
        num_gt=num_ground_truth,
        num_tp_objects=num_matches,
        num_fp_components=num_predictions - num_matches,
        num_fp_pixels=int(np.count_nonzero(evaluated_prediction & ~target)),
        matched_pairs=matched_pairs,
    )


__all__ = ["MatchResult", "connected_components", "match_components"]
