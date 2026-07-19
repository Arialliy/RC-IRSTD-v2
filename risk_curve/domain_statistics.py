"""Versioned, label-free window statistics for risk-curve prediction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from scipy import ndimage


STATISTICS_SCHEMA_VERSION = "rc-v2-stats-v2-plateau-nms"
LOGIT_STATISTICS_SCHEMA_VERSION = "rc-v4-stats-v1-raw-logit"
SUPPORTED_STATISTICS_SCHEMA_VERSIONS = frozenset(
    {STATISTICS_SCHEMA_VERSION, LOGIT_STATISTICS_SCHEMA_VERSION}
)
_QUANTILES = np.asarray([0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999])
_LOGIT_QUANTILES = np.asarray(
    [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 0.9999]
)
_LOGIT_HISTOGRAM_RANGE = (-16.0, 16.0)
_LOGIT_SURVIVAL_ANCHORS = (-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0, 12.0)
_LOGIT_CANDIDATE_MINIMUM = -8.0


def _safe_moments(values: np.ndarray) -> tuple[float, float, float, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    mean = float(flat.mean()) if flat.size else 0.0
    std = float(flat.std()) if flat.size else 0.0
    if std < 1e-12:
        return mean, std, 0.0, 0.0
    centred = (flat - mean) / std
    return mean, std, float(np.mean(centred**3)), float(np.mean(centred**4) - 3.0)


def _normalised_hist(values: np.ndarray, bins: int = 32) -> np.ndarray:
    hist, _ = np.histogram(np.clip(values, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
    return hist.astype(np.float64) / max(int(hist.sum()), 1)


def _normalised_logit_hist(values: np.ndarray, bins: int = 32) -> np.ndarray:
    """Fixed-range logit histogram with explicit saturation at both edges.

    The fixed range is part of the v4 feature schema.  Clipping only affects
    histogram membership; raw quantiles and moments below retain the actual
    finite logit values, including the extreme tail.
    """

    low, high = _LOGIT_HISTOGRAM_RANGE
    clipped = np.clip(np.asarray(values, dtype=np.float64), low, high)
    hist, _ = np.histogram(clipped, bins=bins, range=(low, high))
    return hist.astype(np.float64) / max(int(hist.sum()), 1)


def _peak_values(probability: np.ndarray, min_score: float = 0.05) -> np.ndarray:
    pooled = ndimage.maximum_filter(probability, size=3, mode="nearest")
    mask = (probability >= pooled - 1e-7) & (probability >= min_score)
    # One value per connected plateau, not one value per plateau pixel.  The
    # old implementation turned a uniform/quantised background into up to one
    # million artificial peaks and corrupted the survival statistics.
    labels, num_labels = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if num_labels == 0:
        return np.empty(0, dtype=np.asarray(probability).dtype)
    values = ndimage.maximum(
        probability,
        labels=labels,
        index=np.arange(1, num_labels + 1, dtype=np.int32),
    )
    return np.asarray(values, dtype=np.asarray(probability).dtype).reshape(-1)


def _image_features(gray: np.ndarray) -> np.ndarray:
    image = np.asarray(gray, dtype=np.float64)
    if image.max(initial=0.0) > 1.0:
        image = image / 255.0
    image = np.clip(image, 0.0, 1.0)
    mean = float(image.mean())
    std = float(image.std())
    mad = float(np.median(np.abs(image - np.median(image))))
    gy, gx = np.gradient(image)
    grad = np.hypot(gx, gy)
    grad_mean = float(grad.mean())
    grad_q95 = float(np.quantile(grad, 0.95))
    grad_density = float(np.mean(grad > max(grad_q95 * 0.5, 1e-6)))
    laplacian_noise = float(ndimage.laplace(image).std())

    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(image - mean))) ** 2
    height, width = image.shape
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt(((yy - height / 2) / max(height, 1)) ** 2 + ((xx - width / 2) / max(width, 1)) ** 2)
    high_frequency = float(spectrum[radius >= 0.25].sum() / max(spectrum.sum(), 1e-12))
    gx_energy = float(np.mean(gx**2))
    gy_energy = float(np.mean(gy**2))
    stripe_directionality = abs(gx_energy - gy_energy) / max(gx_energy + gy_energy, 1e-12)
    local_mean = ndimage.uniform_filter(image, size=7, mode="reflect")
    contrast = np.abs(image - local_mean)
    return np.asarray(
        [
            mean,
            std,
            mad,
            grad_mean,
            grad_q95,
            grad_density,
            laplacian_noise,
            high_frequency,
            stripe_directionality,
            float(contrast.mean()),
            float(np.quantile(contrast, 0.95)),
        ],
        dtype=np.float64,
    )


def base_feature_names(histogram_bins: int = 32) -> list[str]:
    names = [f"prob_hist_{index:02d}" for index in range(histogram_bins)]
    names += [f"prob_q_{value:g}" for value in _QUANTILES]
    names += ["prob_ratio_0.5", "prob_ratio_0.9", "prob_ratio_0.99", "prob_ratio_0.999"]
    names += ["prob_mean", "prob_std", "prob_skew", "prob_excess_kurtosis"]
    names += ["between_image_mean_std", "between_image_q99_std"]
    names += [f"peak_hist_{index:02d}" for index in range(histogram_bins)]
    names += [f"peak_q_{value:g}" for value in _QUANTILES]
    names += ["peaks_per_mp_mean", "peaks_per_mp_std", "peak_max", "peak_mean", "peak_top10_mean"]
    names += [
        "gray_mean",
        "gray_std",
        "gray_mad",
        "gradient_mean",
        "gradient_q95",
        "gradient_density",
        "laplacian_noise",
        "high_frequency_energy",
        "stripe_directionality",
        "local_contrast_mean",
        "local_contrast_q95",
        "gray_available_fraction",
    ]
    return names


def logit_feature_names(histogram_bins: int = 32) -> list[str]:
    """Ordered, label-free feature names for the v4 raw-logit contract."""

    names = [f"logit_hist_{index:02d}" for index in range(histogram_bins)]
    names += [f"logit_q_{value:g}" for value in _LOGIT_QUANTILES]
    names += [f"logit_survival_{value:g}" for value in _LOGIT_SURVIVAL_ANCHORS]
    names += ["logit_mean", "logit_std", "logit_skew", "logit_excess_kurtosis"]
    names += ["between_image_logit_mean_std", "between_image_logit_q99_std"]
    names += [f"candidate_logit_hist_{index:02d}" for index in range(histogram_bins)]
    names += [f"candidate_logit_q_{value:g}" for value in _LOGIT_QUANTILES]
    names += [
        "candidates_per_mp_mean",
        "candidates_per_mp_std",
        "candidate_logit_max",
        "candidate_logit_mean",
        "candidate_logit_top10_mean",
    ]
    names += [
        "gray_mean",
        "gray_std",
        "gray_mad",
        "gradient_mean",
        "gradient_q95",
        "gradient_density",
        "laplacian_noise",
        "high_frequency_energy",
        "stripe_directionality",
        "local_contrast_mean",
        "local_contrast_q95",
        "gray_available_fraction",
    ]
    return names


def feature_schema_sha256(
    statistics_schema_version: str,
    *,
    histogram_bins: int = 32,
    statistics_names: Iterable[str] | np.ndarray | None = None,
) -> str:
    """Hash feature order *and* all numeric extraction parameters.

    A names-only digest cannot detect a silent change to histogram bounds,
    quantiles, survival anchors, or the local-candidate rule.  This digest is
    therefore the canonical v4 feature contract recorded by episodes and
    checkpoints.  The probability schema remains supported for old artifacts.
    """

    if histogram_bins < 1:
        raise ValueError("histogram_bins must be positive")
    if statistics_schema_version == STATISTICS_SCHEMA_VERSION:
        payload: dict[str, object] = {
            "schema_version": STATISTICS_SCHEMA_VERSION,
            "names": base_feature_names(histogram_bins),
            "histogram_bins": histogram_bins,
            "histogram_range": [0.0, 1.0],
            "quantiles": _QUANTILES.tolist(),
            "peak_rule": {
                "maximum_filter_size": 3,
                "minimum_probability": 0.05,
                "connected_plateau_reduction": "maximum",
            },
        }
    elif statistics_schema_version == LOGIT_STATISTICS_SCHEMA_VERSION:
        payload = {
            "schema_version": LOGIT_STATISTICS_SCHEMA_VERSION,
            "names": logit_feature_names(histogram_bins),
            "histogram_bins": histogram_bins,
            "histogram_range": list(_LOGIT_HISTOGRAM_RANGE),
            "quantiles": _LOGIT_QUANTILES.tolist(),
            "survival_anchors": list(_LOGIT_SURVIVAL_ANCHORS),
            "candidate_rule": {
                "maximum_filter_size": 3,
                "minimum_logit": _LOGIT_CANDIDATE_MINIMUM,
                "connected_plateau_reduction": "maximum",
            },
        }
    else:
        raise ValueError(
            f"Unsupported statistics schema for feature hashing: "
            f"{statistics_schema_version!r}"
        )
    if statistics_names is not None:
        payload["names"] = list(validate_statistics_names(statistics_names))
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_statistics_names(
    names: Iterable[str] | np.ndarray,
    *,
    expected_dim: int | None = None,
) -> tuple[str, ...]:
    """Validate the ordered feature schema used by archives and checkpoints."""

    values = np.asarray(list(names) if not isinstance(names, np.ndarray) else names)
    if values.ndim != 1:
        raise ValueError("statistics_names must be a one-dimensional sequence")
    result = tuple(str(item) for item in values.tolist())
    if expected_dim is not None and len(result) != int(expected_dim):
        raise ValueError(
            f"statistics_names has {len(result)} entries; expected {int(expected_dim)}"
        )
    if not result or any(not item.strip() for item in result):
        raise ValueError("statistics_names must contain non-empty names")
    if len(set(result)) != len(result):
        raise ValueError("statistics_names must be unique")
    return result


def statistics_names_sha256(names: Iterable[str] | np.ndarray) -> str:
    """Hash an ordered feature-name sequence with an unambiguous encoding."""

    validated = validate_statistics_names(names)
    payload = json.dumps(validated, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class WindowStatistics:
    values: np.ndarray
    names: tuple[str, ...]
    schema_version: str = STATISTICS_SCHEMA_VERSION


@dataclass(frozen=True)
class SourceStatisticsReference:
    """Source-domain centres and a regularised shared precision matrix."""

    domain_names: tuple[str, ...]
    centers: np.ndarray
    precision: np.ndarray
    statistics_names: tuple[str, ...]
    statistics_schema_version: str = STATISTICS_SCHEMA_VERSION

    def validate(
        self,
        feature_dim: int,
        statistics_names: Iterable[str] | np.ndarray | None = None,
        statistics_schema_version: str | None = None,
    ) -> None:
        if self.statistics_schema_version not in SUPPORTED_STATISTICS_SCHEMA_VERSIONS:
            raise ValueError(
                "Source reference uses an unsupported statistics schema: "
                f"{self.statistics_schema_version!r}"
            )
        if (
            statistics_schema_version is not None
            and self.statistics_schema_version != statistics_schema_version
        ):
            raise ValueError(
                "Source reference statistics schema differs from the window: "
                f"{self.statistics_schema_version!r} != "
                f"{statistics_schema_version!r}"
            )
        if not self.domain_names or any(not name.strip() for name in self.domain_names):
            raise ValueError("Source reference domain names must be non-empty")
        if len(set(self.domain_names)) != len(self.domain_names):
            raise ValueError("Source reference domain names must be unique")
        reference_names = validate_statistics_names(
            self.statistics_names, expected_dim=feature_dim
        )
        if statistics_names is not None:
            actual_names = validate_statistics_names(
                statistics_names, expected_dim=feature_dim
            )
            if actual_names != reference_names:
                raise ValueError(
                    "Source reference statistics_names do not exactly match the "
                    "window feature order"
                )
        if self.centers.shape != (len(self.domain_names), feature_dim):
            raise ValueError(
                f"Source centers have shape {self.centers.shape}; expected "
                f"({len(self.domain_names)}, {feature_dim})"
            )
        if self.precision.shape != (feature_dim, feature_dim):
            raise ValueError("Source precision matrix has the wrong shape")
        if not np.isfinite(self.centers).all() or not np.isfinite(self.precision).all():
            raise ValueError("Source reference contains NaN or infinite values")


def fit_source_reference(
    domain_statistics: Mapping[str, np.ndarray],
    regularization: float = 1e-3,
    *,
    statistics_names: Iterable[str] | np.ndarray | None = None,
    statistics_schema_version: str = STATISTICS_SCHEMA_VERSION,
) -> SourceStatisticsReference:
    """Fit source centres without using target-domain labels."""

    if regularization <= 0.0:
        raise ValueError("regularization must be positive")
    if not domain_statistics:
        raise ValueError("At least one source domain is required")
    if statistics_schema_version not in SUPPORTED_STATISTICS_SCHEMA_VERSIONS:
        raise ValueError(
            "Cannot fit a source reference from an incompatible statistics schema: "
            f"{statistics_schema_version!r}"
        )
    raw_names = tuple(domain_statistics)
    if any(not isinstance(name, str) or not name.strip() for name in raw_names):
        raise ValueError("Source domain names must be non-empty strings")
    if len(set(raw_names)) != len(raw_names):
        raise ValueError("Source domain names must be unique")
    names = tuple(sorted(raw_names))
    arrays = [np.asarray(domain_statistics[name], dtype=np.float64) for name in names]
    if arrays[0].ndim != 2 or arrays[0].shape[0] == 0:
        raise ValueError(f"Domain {names[0]} statistics must have shape [N, D]")
    feature_dim = arrays[0].shape[1]
    if statistics_names is None:
        inferred = tuple(
            base_feature_names()
            if statistics_schema_version == STATISTICS_SCHEMA_VERSION
            else logit_feature_names()
        )
        if len(inferred) != feature_dim:
            raise ValueError(
                "statistics_names are required when feature dimensions do not match "
                "the canonical base schema"
            )
        feature_names = inferred
    else:
        feature_names = validate_statistics_names(
            statistics_names, expected_dim=feature_dim
        )
    for name, array in zip(names, arrays):
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != feature_dim:
            raise ValueError(f"Domain {name} statistics must have shape [N, {feature_dim}]")
        if not np.isfinite(array).all():
            raise ValueError(f"Domain {name} statistics contain invalid values")
    centers = np.stack([array.mean(axis=0) for array in arrays])
    pooled = np.concatenate([array - array.mean(axis=0, keepdims=True) for array in arrays], axis=0)
    covariance = pooled.T @ pooled / max(pooled.shape[0] - len(arrays), 1)
    scale = max(float(np.trace(covariance) / feature_dim), 1e-8)
    covariance = covariance + regularization * scale * np.eye(feature_dim)
    precision = np.linalg.pinv(covariance, hermitian=True)
    reference = SourceStatisticsReference(
        names,
        centers.astype(np.float32),
        precision.astype(np.float32),
        feature_names,
        statistics_schema_version,
    )
    reference.validate(feature_dim, feature_names)
    return reference


def save_source_reference(reference: SourceStatisticsReference, path: str | Path) -> Path:
    feature_dim = int(np.asarray(reference.centers).shape[-1])
    reference.validate(feature_dim, reference.statistics_names)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        domain_names=np.asarray(reference.domain_names, dtype=str),
        centers=np.asarray(reference.centers, dtype=np.float32),
        precision=np.asarray(reference.precision, dtype=np.float32),
        statistics_names=np.asarray(reference.statistics_names, dtype=str),
        statistics_names_sha256=np.asarray(
            statistics_names_sha256(reference.statistics_names)
        ),
        feature_schema_sha256=np.asarray(
            feature_schema_sha256(
                reference.statistics_schema_version,
                statistics_names=reference.statistics_names,
            )
        ),
        statistics_schema_version=np.asarray(reference.statistics_schema_version),
    )
    return output


def load_source_reference(path: str | Path) -> SourceStatisticsReference:
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "domain_names",
            "centers",
            "precision",
            "statistics_names",
            "statistics_schema_version",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(
                f"Source reference {source} is missing: {', '.join(missing)}"
            )
        schema_version = str(np.asarray(archive["statistics_schema_version"]).item())
        reference = SourceStatisticsReference(
            tuple(str(item) for item in archive["domain_names"]),
            np.asarray(archive["centers"], dtype=np.float32),
            np.asarray(archive["precision"], dtype=np.float32),
            tuple(str(item) for item in archive["statistics_names"]),
            schema_version,
        )
        expected_hash = (
            str(np.asarray(archive["statistics_names_sha256"]).item())
            if "statistics_names_sha256" in archive
            else None
        )
        expected_feature_hash = (
            str(np.asarray(archive["feature_schema_sha256"]).item())
            if "feature_schema_sha256" in archive
            else None
        )
    if reference.centers.ndim != 2:
        raise ValueError("Source reference centers must have shape [K, D]")
    reference.validate(reference.centers.shape[1], reference.statistics_names)
    if expected_hash is not None and expected_hash != statistics_names_sha256(
        reference.statistics_names
    ):
        raise ValueError("Source reference statistics_names hash mismatch")
    if expected_feature_hash is not None and expected_feature_hash != (
        feature_schema_sha256(
            reference.statistics_schema_version,
            statistics_names=reference.statistics_names,
        )
    ):
        raise ValueError("Source reference feature schema hash mismatch")
    return reference


def append_source_distances(
    statistics: WindowStatistics, reference: SourceStatisticsReference
) -> WindowStatistics:
    reference.validate(
        statistics.values.size,
        statistics.names,
        statistics.schema_version,
    )
    difference = reference.centers.astype(np.float64) - statistics.values.astype(np.float64)[None, :]
    euclidean = np.linalg.norm(difference, axis=1)
    mahalanobis_squared = np.einsum("ki,ij,kj->k", difference, reference.precision, difference)
    mahalanobis = np.sqrt(np.maximum(mahalanobis_squared, 0.0))
    values = np.concatenate([statistics.values, euclidean, mahalanobis]).astype(np.float32)
    names = list(statistics.names)
    names += [f"source_euclidean_{name}" for name in reference.domain_names]
    names += [f"source_mahalanobis_{name}" for name in reference.domain_names]
    return WindowStatistics(values, tuple(names), statistics.schema_version)


def extract_window_statistics(
    probabilities: Iterable[np.ndarray],
    gray_images: Iterable[np.ndarray | None] | None = None,
    histogram_bins: int = 32,
    source_reference: SourceStatisticsReference | None = None,
) -> WindowStatistics:
    """Extract label-free features in a stable, documented order."""

    probs = [np.asarray(item, dtype=np.float32).squeeze() for item in probabilities]
    if not probs:
        raise ValueError("At least one probability map is required")
    for item in probs:
        if item.ndim != 2 or not np.isfinite(item).all():
            raise ValueError("Each probability map must be a finite 2-D array")
        if item.min() < 0.0 or item.max() > 1.0:
            raise ValueError("Probability maps must lie in [0, 1]")

    all_prob = np.concatenate([item.reshape(-1) for item in probs])
    probability_features: list[float] = []
    probability_features.extend(_normalised_hist(all_prob, histogram_bins))
    probability_features.extend(np.quantile(all_prob, _QUANTILES))
    probability_features.extend(float(np.mean(all_prob >= value)) for value in (0.5, 0.9, 0.99, 0.999))
    probability_features.extend(_safe_moments(all_prob))
    probability_features.append(float(np.std([item.mean() for item in probs])))
    probability_features.append(float(np.std([np.quantile(item, 0.99) for item in probs])))

    peak_arrays = [_peak_values(item) for item in probs]
    nonempty_peaks = [item for item in peak_arrays if item.size]
    all_peaks = np.concatenate(nonempty_peaks) if nonempty_peaks else np.zeros(1, dtype=np.float32)
    peak_features: list[float] = []
    peak_features.extend(_normalised_hist(all_peaks, histogram_bins))
    peak_features.extend(np.quantile(all_peaks, _QUANTILES))
    per_mp = [peaks.size / (prob.size / 1_000_000.0) for peaks, prob in zip(peak_arrays, probs)]
    peak_features.extend([float(np.mean(per_mp)), float(np.std(per_mp))])
    top_count = min(10, all_peaks.size)
    peak_features.extend(
        [float(all_peaks.max()), float(all_peaks.mean()), float(np.partition(all_peaks, -top_count)[-top_count:].mean())]
    )

    if gray_images is None:
        grays: list[np.ndarray | None] = [None] * len(probs)
    else:
        grays = list(gray_images)
        if len(grays) != len(probs):
            raise ValueError("gray_images and probabilities must have the same length")
    available = [gray for gray in grays if gray is not None]
    if available:
        image_features = np.mean([_image_features(np.asarray(gray).squeeze()) for gray in available], axis=0)
    else:
        image_features = np.zeros(11, dtype=np.float64)
    image_features = np.concatenate([image_features, [len(available) / len(probs)]])

    values = np.asarray(probability_features + peak_features + image_features.tolist(), dtype=np.float32)
    names = base_feature_names(histogram_bins)
    if values.size != len(names):
        raise RuntimeError(f"Feature schema mismatch: {values.size} values vs {len(names)} names")
    if not np.isfinite(values).all():
        raise RuntimeError("Extracted statistics contain NaN or infinite values")
    result = WindowStatistics(values=values, names=tuple(names))
    return append_source_distances(result, source_reference) if source_reference is not None else result


def extract_logit_window_statistics(
    raw_logits: Iterable[np.ndarray],
    gray_images: Iterable[np.ndarray | None] | None = None,
    histogram_bins: int = 32,
    source_reference: SourceStatisticsReference | None = None,
) -> WindowStatistics:
    """Extract the v4 label-free statistics directly from float32 raw logits.

    No sigmoid is applied anywhere in this path.  In particular, positive-tail
    survival and local-candidate values retain distinctions that probability
    saturation would erase.  Masks are neither accepted nor consulted.
    """

    if histogram_bins < 1:
        raise ValueError("histogram_bins must be positive")
    logits = [np.asarray(item, dtype=np.float32).squeeze() for item in raw_logits]
    if not logits:
        raise ValueError("At least one raw-logit map is required")
    for item in logits:
        if item.ndim != 2 or not np.isfinite(item).all():
            raise ValueError("Each raw-logit map must be a finite 2-D array")

    all_logits = np.concatenate([item.reshape(-1) for item in logits])
    logit_features: list[float] = []
    logit_features.extend(_normalised_logit_hist(all_logits, histogram_bins))
    logit_features.extend(np.quantile(all_logits, _LOGIT_QUANTILES))
    logit_features.extend(
        float(np.mean(all_logits >= value))
        for value in _LOGIT_SURVIVAL_ANCHORS
    )
    logit_features.extend(_safe_moments(all_logits))
    logit_features.append(float(np.std([item.mean() for item in logits])))
    logit_features.append(
        float(np.std([np.quantile(item, 0.99) for item in logits]))
    )

    candidate_arrays = [
        _peak_values(item, min_score=_LOGIT_CANDIDATE_MINIMUM) for item in logits
    ]
    nonempty_candidates = [item for item in candidate_arrays if item.size]
    # A fixed low-end value makes the empty-candidate case deterministic while
    # candidate density still records that no maxima passed the candidate cut.
    all_candidates = (
        np.concatenate(nonempty_candidates)
        if nonempty_candidates
        else np.asarray([_LOGIT_CANDIDATE_MINIMUM], dtype=np.float32)
    )
    candidate_features: list[float] = []
    candidate_features.extend(
        _normalised_logit_hist(all_candidates, histogram_bins)
    )
    candidate_features.extend(np.quantile(all_candidates, _LOGIT_QUANTILES))
    per_mp = [
        candidates.size / (logit.size / 1_000_000.0)
        for candidates, logit in zip(candidate_arrays, logits)
    ]
    candidate_features.extend([float(np.mean(per_mp)), float(np.std(per_mp))])
    top_count = min(10, all_candidates.size)
    candidate_features.extend(
        [
            float(all_candidates.max()),
            float(all_candidates.mean()),
            float(
                np.partition(all_candidates, -top_count)[-top_count:].mean()
            ),
        ]
    )

    if gray_images is None:
        grays: list[np.ndarray | None] = [None] * len(logits)
    else:
        grays = list(gray_images)
        if len(grays) != len(logits):
            raise ValueError("gray_images and raw_logits must have the same length")
    available = [gray for gray in grays if gray is not None]
    if available:
        image_features = np.mean(
            [_image_features(np.asarray(gray).squeeze()) for gray in available],
            axis=0,
        )
    else:
        image_features = np.zeros(11, dtype=np.float64)
    image_features = np.concatenate(
        [image_features, [len(available) / len(logits)]]
    )

    values = np.asarray(
        logit_features + candidate_features + image_features.tolist(),
        dtype=np.float32,
    )
    names = logit_feature_names(histogram_bins)
    if values.size != len(names):
        raise RuntimeError(
            f"Feature schema mismatch: {values.size} values vs {len(names)} names"
        )
    if not np.isfinite(values).all():
        raise RuntimeError("Extracted logit statistics contain NaN or infinite values")
    result = WindowStatistics(
        values=values,
        names=tuple(names),
        schema_version=LOGIT_STATISTICS_SCHEMA_VERSION,
    )
    return (
        append_source_distances(result, source_reference)
        if source_reference is not None
        else result
    )
