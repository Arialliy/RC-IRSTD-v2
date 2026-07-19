from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage

from evaluation.component_matching import connected_components, match_components

from .score_store import ScoreItem


# One float32 ULP above the largest valid sigmoid probability. Under the
# repository-wide ``score >= threshold`` rule this is the only finite scalar
# that faithfully represents a reject-all decision when a score equals 1.0.
REJECT_ALL_THRESHOLD = float(
    np.nextafter(np.float32(1.0), np.float32(np.inf), dtype=np.float32)
)
# Extended-logit code used by the calibrator for the reject-all sentinel. A
# probability threshold of exactly 1.0 maps below this value, so the two cases
# remain distinguishable.
REJECT_ALL_LATENT_LOGIT = 16.5


@dataclass
class OperatingMetrics:
    threshold: float
    pd: float
    fa_pixel: float
    fa_component_mp: float
    detected_objects: int
    total_objects: int
    false_positive_pixels: int
    false_positive_components: int
    total_pixels: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def component_max_scores(probability: np.ndarray, mask: np.ndarray) -> np.ndarray:
    labeled, count = connected_components(mask.astype(bool), connectivity=2)
    if count == 0:
        return np.empty(0, dtype=np.float32)
    indices = np.arange(1, count + 1, dtype=np.int32)
    maxima = ndimage.maximum(probability, labels=labeled, index=indices)
    return np.asarray(maxima, dtype=np.float32)


def aggregate_query_scores(
    items: Sequence[ScoreItem],
) -> tuple[np.ndarray, np.ndarray, int]:
    if not items:
        raise ValueError("Query window is empty")
    background_scores: list[np.ndarray] = []
    object_scores: list[np.ndarray] = []
    total_pixels = 0
    for item in items:
        if not item.has_mask:
            raise ValueError(f"Evaluation requires labels: {item.image_id}")
        background_scores.append(item.probability[item.mask == 0].reshape(-1))
        object_scores.append(component_max_scores(item.probability, item.mask))
        total_pixels += int(item.probability.size)
    background = np.concatenate(background_scores).astype(np.float32)
    objects = (
        np.concatenate(object_scores).astype(np.float32)
        if any(scores.size for scores in object_scores)
        else np.empty(0, dtype=np.float32)
    )
    return background, objects, total_pixels


def exact_safe_threshold(background_scores: np.ndarray, budget: float, total_pixels: int) -> float:
    """Safe threshold under the repository-wide ``score >= threshold`` rule."""
    if budget < 0:
        raise ValueError("budget must be non-negative")
    allowed = int(np.floor(float(budget) * int(total_pixels)))
    if background_scores.size == 0:
        return 0.0
    if allowed >= background_scores.size:
        return 0.0
    descending = np.sort(background_scores)[::-1]
    # Advance one float32 ULP above the first disallowed score.  This excludes
    # that value and all ties under >= while retaining an auditable reject-all
    # sentinel above 1.0 when an exported score equals exactly 1.0.
    boundary = np.float32(descending[max(allowed, 0)])
    return float(np.nextafter(boundary, np.float32(np.inf), dtype=np.float32))


def oracle_thresholds(
    items: Sequence[ScoreItem],
    budgets: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    background, objects, total_pixels = aggregate_query_scores(items)
    thresholds: list[float] = []
    pds: list[float] = []
    fas: list[float] = []
    for budget in budgets:
        threshold = exact_safe_threshold(background, float(budget), total_pixels)
        fa = float(np.count_nonzero(background >= threshold) / max(total_pixels, 1))
        pd = float(np.count_nonzero(objects >= threshold) / max(objects.size, 1)) if objects.size else 0.0
        thresholds.append(threshold)
        fas.append(fa)
        pds.append(pd)
    return (
        np.asarray(thresholds, dtype=np.float32),
        np.asarray(pds, dtype=np.float32),
        np.asarray(fas, dtype=np.float32),
    )


def evaluate_threshold(items: Sequence[ScoreItem], threshold: float) -> OperatingMetrics:
    if not items:
        raise ValueError("Evaluation requires at least one query item")
    detected_objects = 0
    total_objects = 0
    false_positive_pixels = 0
    false_positive_components = 0
    total_pixels = 0
    for item in items:
        if not item.has_mask:
            raise ValueError(f"Evaluation requires labels: {item.image_id}")
        prediction = item.probability >= float(threshold)
        ground_truth = item.mask.astype(bool)
        total_pixels += prediction.size
        matched = match_components(
            prediction,
            ground_truth,
            rule="overlap",
            connectivity=2,
            min_component_area=1,
        )
        total_objects += int(matched.num_gt)
        detected_objects += int(matched.num_tp_objects)
        false_positive_pixels += int(matched.num_fp_pixels)
        false_positive_components += int(matched.num_fp_components)

    pd = detected_objects / max(total_objects, 1)
    fa_pixel = false_positive_pixels / max(total_pixels, 1)
    fa_component_mp = false_positive_components / max(total_pixels / 1_000_000.0, 1e-12)
    return OperatingMetrics(
        threshold=float(threshold),
        pd=float(pd),
        fa_pixel=float(fa_pixel),
        fa_component_mp=float(fa_component_mp),
        detected_objects=detected_objects,
        total_objects=total_objects,
        false_positive_pixels=false_positive_pixels,
        false_positive_components=false_positive_components,
        total_pixels=total_pixels,
    )


def risk_histograms(
    items: Sequence[ScoreItem],
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    background, objects, total_pixels = aggregate_query_scores(items)
    eps = 1e-6
    background_logits = np.log(np.clip(background, eps, 1 - eps)) - np.log1p(
        -np.clip(background, eps, 1 - eps)
    )
    object_logits = np.log(np.clip(objects, eps, 1 - eps)) - np.log1p(
        -np.clip(objects, eps, 1 - eps)
    )
    lower = float(bin_edges[0])
    upper = float(bin_edges[-1])
    margin = max((upper - lower) * 1e-7, np.finfo(np.float32).eps)
    background_logits = np.clip(background_logits, lower + margin, upper - margin)
    object_logits = np.clip(object_logits, lower + margin, upper - margin)
    background_hist, _ = np.histogram(background_logits, bins=bin_edges)
    object_hist, _ = np.histogram(object_logits, bins=bin_edges)
    return background_hist.astype(np.float32), object_hist.astype(np.float32), total_pixels
