from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage

from rc_irstd.evaluation.score_store import ScoreItem


@dataclass(frozen=True)
class FeatureSpec:
    probability_bins: int = 32
    logit_bins: int = 32
    peak_bins: int = 32
    probability_quantiles: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99, 0.995, 0.999)
    peak_quantiles: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99, 0.995, 0.999)
    peak_min_score: float = 0.05
    peak_window: int = 3
    logit_min: float = -12.0
    logit_max: float = 18.0

    def to_dict(self) -> dict[str, object]:
        return {
            "probability_bins": self.probability_bins,
            "logit_bins": self.logit_bins,
            "peak_bins": self.peak_bins,
            "probability_quantiles": list(self.probability_quantiles),
            "peak_quantiles": list(self.peak_quantiles),
            "peak_min_score": self.peak_min_score,
            "peak_window": self.peak_window,
            "logit_min": self.logit_min,
            "logit_max": self.logit_max,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "FeatureSpec":
        return cls(
            probability_bins=int(payload.get("probability_bins", 32)),
            logit_bins=int(payload.get("logit_bins", 32)),
            peak_bins=int(payload.get("peak_bins", 32)),
            probability_quantiles=tuple(payload.get("probability_quantiles", cls.probability_quantiles)),
            peak_quantiles=tuple(payload.get("peak_quantiles", cls.peak_quantiles)),
            peak_min_score=float(payload.get("peak_min_score", 0.05)),
            peak_window=int(payload.get("peak_window", 3)),
            logit_min=float(payload.get("logit_min", -12.0)),
            logit_max=float(payload.get("logit_max", 18.0)),
        )


def probability_to_logit(probability: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(probability, eps, 1.0 - eps)
    return np.log(clipped) - np.log1p(-clipped)


def _normalized_histogram(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    if values.size == 0:
        return np.zeros(bins, dtype=np.float32)
    lower, upper = value_range
    # Keep every observation in the histogram. Values outside the modeled
    # range are assigned to the boundary bins instead of being silently lost.
    margin = max((upper - lower) * 1e-7, np.finfo(np.float32).eps)
    clipped = np.clip(values, lower + margin, upper - margin)
    histogram, _ = np.histogram(clipped, bins=bins, range=value_range)
    histogram = histogram.astype(np.float32)
    return histogram / max(float(histogram.sum()), 1.0)


def _safe_quantiles(values: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    if values.size == 0:
        return np.zeros(len(quantiles), dtype=np.float32)
    return np.quantile(values, quantiles).astype(np.float32)


def extract_peak_scores(
    probability: np.ndarray,
    *,
    window: int = 3,
    min_score: float = 0.05,
) -> np.ndarray:
    maximum = ndimage.maximum_filter(probability, size=window, mode="nearest")
    peak_mask = (probability >= maximum - 1e-7) & (probability >= min_score)
    labeled, count = ndimage.label(peak_mask)
    if count == 0:
        return np.empty(0, dtype=np.float32)
    # Vectorized plateau reduction: one representative maximum per connected
    # local-maximum plateau. This avoids O(num_peaks * num_pixels) scans.
    indices = np.arange(1, count + 1, dtype=np.int32)
    peaks = ndimage.maximum(probability, labels=labeled, index=indices)
    return np.asarray(peaks, dtype=np.float32)


def feature_names(spec: FeatureSpec) -> list[str]:
    names = [f"prob_hist_{index}" for index in range(spec.probability_bins)]
    names.extend(f"logit_hist_{index}" for index in range(spec.logit_bins))
    names.extend(f"prob_q_{value:g}" for value in spec.probability_quantiles)
    names.extend(f"peak_hist_{index}" for index in range(spec.peak_bins))
    names.extend(f"peak_q_{value:g}" for value in spec.peak_quantiles)
    names.extend(
        [
            "peak_density_mp",
            "peak_top1",
            "peak_top5_mean",
            "gray_mean",
            "gray_std",
            "gray_mad",
            "gradient_mean",
            "gradient_q95",
            "laplacian_std",
            "high_frequency_ratio",
            "score_mean",
            "score_std",
            "score_skew",
            "score_kurtosis",
        ]
    )
    return names


def extract_window_features(
    items: Sequence[ScoreItem],
    spec: FeatureSpec | None = None,
) -> np.ndarray:
    if not items:
        raise ValueError("At least one support item is required")
    spec = spec or FeatureSpec()
    probabilities = np.concatenate([item.probability.reshape(-1) for item in items]).astype(np.float32)
    logits = probability_to_logit(probabilities)
    peak_arrays = [
        extract_peak_scores(
            item.probability,
            window=spec.peak_window,
            min_score=spec.peak_min_score,
        )
        for item in items
    ]
    peaks = np.concatenate(peak_arrays) if any(array.size for array in peak_arrays) else np.empty(0, np.float32)
    gray = np.concatenate([item.gray.reshape(-1) for item in items]).astype(np.float32)
    gradient_values: list[np.ndarray] = []
    laplacian_values: list[np.ndarray] = []
    high_frequency_energy = 0.0
    total_energy = 0.0
    for item in items:
        gy, gx = np.gradient(item.gray.astype(np.float32))
        gradient_values.append(np.hypot(gx, gy).reshape(-1))
        laplacian_values.append(ndimage.laplace(item.gray.astype(np.float32)).reshape(-1))
        low = ndimage.gaussian_filter(item.gray.astype(np.float32), sigma=1.5)
        high = item.gray.astype(np.float32) - low
        high_frequency_energy += float(np.square(high).sum())
        total_energy += float(np.square(item.gray).sum())
    gradients = np.concatenate(gradient_values)
    laplacian = np.concatenate(laplacian_values)

    centered = probabilities - probabilities.mean()
    score_std = float(probabilities.std())
    score_skew = float(np.mean(centered**3) / (score_std**3 + 1e-8))
    score_kurtosis = float(np.mean(centered**4) / (score_std**4 + 1e-8))
    total_pixels = sum(item.probability.size for item in items)
    sorted_peaks = np.sort(peaks)[::-1]
    peak_top1 = float(sorted_peaks[0]) if sorted_peaks.size else 0.0
    peak_top5 = float(sorted_peaks[:5].mean()) if sorted_peaks.size else 0.0
    gray_median = np.median(gray)

    parts = [
        _normalized_histogram(probabilities, spec.probability_bins, (0.0, 1.0)),
        _normalized_histogram(logits, spec.logit_bins, (spec.logit_min, spec.logit_max)),
        _safe_quantiles(probabilities, spec.probability_quantiles),
        _normalized_histogram(peaks, spec.peak_bins, (0.0, 1.0)),
        _safe_quantiles(peaks, spec.peak_quantiles),
        np.asarray(
            [
                peaks.size / max(total_pixels / 1_000_000.0, 1e-12),
                peak_top1,
                peak_top5,
                float(gray.mean()),
                float(gray.std()),
                float(np.median(np.abs(gray - gray_median))),
                float(gradients.mean()),
                float(np.quantile(gradients, 0.95)),
                float(laplacian.std()),
                high_frequency_energy / max(total_energy, 1e-8),
                float(probabilities.mean()),
                score_std,
                score_skew,
                score_kurtosis,
            ],
            dtype=np.float32,
        ),
    ]
    features = np.concatenate(parts).astype(np.float32)
    expected = len(feature_names(spec))
    if features.size != expected:
        raise RuntimeError(f"Feature dimension mismatch: {features.size} != {expected}")
    if not np.isfinite(features).all():
        raise ValueError("Non-finite domain statistics were produced")
    return features
