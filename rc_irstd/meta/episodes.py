from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from rc_irstd.evaluation.metrics import (
    REJECT_ALL_LATENT_LOGIT,
    REJECT_ALL_THRESHOLD,
    oracle_thresholds,
    risk_histograms,
)
from rc_irstd.evaluation.score_store import ScoreItem, ScoreStore
from rc_irstd.features.domain_statistics import FeatureSpec, extract_window_features, feature_names
from rc_irstd.utils.io import atomic_write_json, ensure_dir, read_json, save_npz_atomic
from evaluation.artifact_integrity import file_sha256


@dataclass(frozen=True)
class EpisodeWindow:
    support_indices: tuple[int, ...]
    query_indices: tuple[int, ...]


def probability_to_logit_scalar(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0) or np.any(
        values > REJECT_ALL_THRESHOLD
    ):
        raise ValueError("threshold values must lie in [0, reject-all sentinel]")
    # Encode the extended threshold interval [0, nextafter(1,+inf)] with a
    # logistic latent. This preserves a genuine >1 reject-all value instead of
    # clipping it back to 1-eps.
    normalized = values.astype(np.float64) / float(REJECT_ALL_THRESHOLD)
    clipped = np.clip(normalized, eps, 1.0 - eps)
    logits = np.log(clipped) - np.log1p(-clipped)
    logits[values > 1.0] = REJECT_ALL_LATENT_LOGIT
    return logits.astype(np.float32)


def make_episode_windows(
    length: int,
    support_size: int,
    query_size: int,
    *,
    stride: int | None = None,
    max_episodes: int | None = None,
    mode: str = "causal",
    seed: int = 0,
) -> list[EpisodeWindow]:
    if support_size <= 0 or query_size <= 0:
        raise ValueError("support_size and query_size must be positive")
    if length < support_size + query_size:
        raise ValueError(
            f"Need at least {support_size + query_size} images, received {length}"
        )
    span = support_size + query_size
    resolved_stride = span if stride is None else int(stride)
    if resolved_stride <= 0:
        raise ValueError("stride must be positive")
    rng = np.random.default_rng(seed)
    windows: list[EpisodeWindow] = []
    if mode == "causal":
        starts = list(range(0, length - support_size - query_size + 1, resolved_stride))
        if max_episodes is not None and len(starts) > max_episodes:
            selected = np.linspace(0, len(starts) - 1, max_episodes, dtype=int)
            starts = [starts[index] for index in selected]
        for start in starts:
            support = tuple(range(start, start + support_size))
            query = tuple(range(start + support_size, start + support_size + query_size))
            windows.append(EpisodeWindow(support, query))
    elif mode == "random":
        count = max_episodes or max(1, length // (support_size + query_size))
        for _ in range(count):
            selected = rng.choice(length, size=support_size + query_size, replace=False)
            support = tuple(int(value) for value in selected[:support_size])
            query = tuple(int(value) for value in selected[support_size:])
            windows.append(EpisodeWindow(support, query))
    else:
        raise ValueError("mode must be 'causal' or 'random'")
    return windows


def build_episode_archive(
    score_directories: Sequence[str | Path],
    output_dir: str | Path,
    *,
    budgets: Sequence[float],
    support_size: int,
    query_size: int,
    stride: int | None = None,
    max_episodes_per_domain: int | None = None,
    mode: str = "causal",
    seed: int = 0,
    feature_spec: FeatureSpec | None = None,
    risk_bins: int = 256,
    risk_logit_min: float = -12.0,
    risk_logit_max: float = 18.0,
) -> Path:
    budgets_array = np.asarray(list(budgets), dtype=np.float32)
    if budgets_array.ndim != 1 or budgets_array.size < 2:
        raise ValueError("At least two budgets are required")
    if not np.all(budgets_array[:-1] > budgets_array[1:]):
        raise ValueError("Budgets must be strictly descending from loose to strict")
    spec = feature_spec or FeatureSpec()
    output = ensure_dir(output_dir)
    bin_edges = np.linspace(risk_logit_min, risk_logit_max, risk_bins + 1, dtype=np.float32)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5

    features: list[np.ndarray] = []
    thresholds: list[np.ndarray] = []
    threshold_logits: list[np.ndarray] = []
    oracle_pd_values: list[np.ndarray] = []
    oracle_fa_values: list[np.ndarray] = []
    background_histograms: list[np.ndarray] = []
    object_histograms: list[np.ndarray] = []
    total_pixel_counts: list[int] = []
    domain_indices: list[int] = []
    episode_metadata: list[dict[str, Any]] = []
    domain_names: list[str] = []
    score_store_provenance: list[dict[str, Any]] = []
    resolved_stride = support_size + query_size if stride is None else int(stride)
    if resolved_stride <= 0:
        raise ValueError("stride must be positive")

    for domain_index, directory in enumerate(score_directories):
        store = ScoreStore(directory)
        if store.dataset_name in domain_names:
            raise ValueError(f"Duplicate episode domain name: {store.dataset_name}")
        domain_names.append(store.dataset_name)
        score_store_provenance.append(
            {
                "dataset_name": store.dataset_name,
                "manifest": str((store.root / "manifest.json").resolve()),
                "manifest_sha256": file_sha256(store.root / "manifest.json"),
                "records_sha256": store.integrity.get("records_sha256"),
                "ordered_image_ids_sha256": store.integrity.get(
                    "ordered_image_ids_sha256"
                ),
                "integrity_verified": bool(store.integrity.get("verified", False)),
            }
        )
        items = list(store)
        windows = make_episode_windows(
            len(items),
            support_size,
            query_size,
            stride=resolved_stride,
            max_episodes=max_episodes_per_domain,
            mode=mode,
            seed=seed + domain_index * 1009,
        )
        for episode_index, window in enumerate(windows):
            support_items = [items[index] for index in window.support_indices]
            query_items = [items[index] for index in window.query_indices]
            if not all(item.has_mask for item in query_items):
                raise ValueError(
                    f"Pseudo-target query labels are required for {store.dataset_name}"
                )
            feature_vector = extract_window_features(support_items, spec)
            oracle_threshold, oracle_pd, oracle_fa = oracle_thresholds(query_items, budgets_array)
            background_hist, object_hist, total_pixels = risk_histograms(query_items, bin_edges)
            features.append(feature_vector)
            thresholds.append(oracle_threshold)
            threshold_logits.append(probability_to_logit_scalar(oracle_threshold))
            oracle_pd_values.append(oracle_pd)
            oracle_fa_values.append(oracle_fa)
            background_histograms.append(background_hist)
            object_histograms.append(object_hist)
            total_pixel_counts.append(total_pixels)
            domain_indices.append(domain_index)
            episode_metadata.append(
                {
                    "episode_index": episode_index,
                    "domain": store.dataset_name,
                    "support_ids": [item.image_id for item in support_items],
                    "query_ids": [item.image_id for item in query_items],
                }
            )

    if not features:
        raise ValueError("No episodes were created")
    archive_path = output / "episodes.npz"
    save_npz_atomic(
        archive_path,
        features=np.stack(features).astype(np.float32),
        thresholds=np.stack(thresholds).astype(np.float32),
        threshold_logits=np.stack(threshold_logits).astype(np.float32),
        oracle_pd=np.stack(oracle_pd_values).astype(np.float32),
        oracle_fa=np.stack(oracle_fa_values).astype(np.float32),
        background_histograms=np.stack(background_histograms).astype(np.float32),
        object_histograms=np.stack(object_histograms).astype(np.float32),
        total_pixels=np.asarray(total_pixel_counts, dtype=np.float32),
        domain_indices=np.asarray(domain_indices, dtype=np.int64),
        budgets=budgets_array,
        bin_edges=bin_edges,
        bin_centers=bin_centers,
    )
    atomic_write_json(
        output / "metadata.json",
        {
            "format_version": 2,
            "archive": archive_path.name,
            "archive_sha256": file_sha256(archive_path),
            "num_episodes": len(features),
            "support_size": support_size,
            "query_size": query_size,
            "stride": resolved_stride,
            "mode": mode,
            "budgets": budgets_array.tolist(),
            "domain_names": domain_names,
            "feature_spec": spec.to_dict(),
            "feature_names": feature_names(spec),
            "feature_dim": len(feature_names(spec)),
            "risk_bins": risk_bins,
            "risk_logit_min": risk_logit_min,
            "risk_logit_max": risk_logit_max,
            "episodes": episode_metadata,
            "score_stores": score_store_provenance,
            "formal_causal_contract": False,
            "diagnostic_only": True,
            "diagnostic_reason": "library build pending strict CLI contract annotation",
        },
    )
    return archive_path


class EpisodeDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, directory: str | Path) -> None:
        self.root = Path(directory).expanduser().resolve()
        metadata_path = self.root / "metadata.json"
        self.metadata = read_json(metadata_path)
        if not isinstance(self.metadata, dict):
            raise ValueError("Episode metadata must be a JSON object")
        archive_name = self.metadata.get("archive", "episodes.npz")
        if archive_name != "episodes.npz":
            raise ValueError("Episode metadata archive must be exactly episodes.npz")
        archive_path = self.root / archive_name
        recorded_sha = self.metadata.get("archive_sha256")
        if recorded_sha is not None and file_sha256(archive_path) != recorded_sha:
            raise ValueError("Episode archive sha256 differs from metadata")
        with np.load(archive_path, allow_pickle=False) as payload:
            required_arrays = {
                "features",
                "thresholds",
                "threshold_logits",
                "oracle_pd",
                "oracle_fa",
                "background_histograms",
                "object_histograms",
                "total_pixels",
                "domain_indices",
                "budgets",
                "bin_edges",
                "bin_centers",
            }
            missing = required_arrays.difference(payload.files)
            if missing:
                raise ValueError(
                    "Episode archive is missing arrays: " + ", ".join(sorted(missing))
                )
            self.features = torch.from_numpy(np.asarray(payload["features"], dtype=np.float32))
            self.thresholds = torch.from_numpy(np.asarray(payload["thresholds"], dtype=np.float32))
            self.threshold_logits = torch.from_numpy(
                np.asarray(payload["threshold_logits"], dtype=np.float32)
            )
            self.oracle_pd = torch.from_numpy(np.asarray(payload["oracle_pd"], dtype=np.float32))
            self.oracle_fa = torch.from_numpy(np.asarray(payload["oracle_fa"], dtype=np.float32))
            self.background_histograms = torch.from_numpy(
                np.asarray(payload["background_histograms"], dtype=np.float32)
            )
            self.object_histograms = torch.from_numpy(
                np.asarray(payload["object_histograms"], dtype=np.float32)
            )
            self.total_pixels = torch.from_numpy(
                np.asarray(payload["total_pixels"], dtype=np.float32)
            )
            self.domain_indices = torch.from_numpy(
                np.asarray(payload["domain_indices"], dtype=np.int64)
            )
            self.budgets = torch.from_numpy(np.asarray(payload["budgets"], dtype=np.float32))
            self.bin_centers = torch.from_numpy(
                np.asarray(payload["bin_centers"], dtype=np.float32)
            )
            self.bin_edges = torch.from_numpy(
                np.asarray(payload["bin_edges"], dtype=np.float32)
            )
        length = self.features.shape[0]
        if self.features.ndim != 2 or length <= 0:
            raise ValueError("Episode features must have shape [N,F] with N,F > 0")
        for tensor in (
            self.thresholds,
            self.threshold_logits,
            self.oracle_pd,
            self.oracle_fa,
            self.background_histograms,
            self.object_histograms,
            self.total_pixels,
            self.domain_indices,
        ):
            if tensor.shape[0] != length:
                raise ValueError("Episode archive arrays have inconsistent lengths")
        budget_count = int(self.budgets.numel())
        if self.budgets.ndim != 1 or budget_count < 2 or not torch.all(
            self.budgets[:-1] > self.budgets[1:]
        ):
            raise ValueError("Episode budgets must be positive and strictly descending")
        for name, tensor in (
            ("thresholds", self.thresholds),
            ("threshold_logits", self.threshold_logits),
            ("oracle_pd", self.oracle_pd),
            ("oracle_fa", self.oracle_fa),
        ):
            if tensor.shape != (length, budget_count):
                raise ValueError(f"Episode {name} must have shape [N,num_budgets]")
        if self.background_histograms.ndim != 2 or self.object_histograms.shape != self.background_histograms.shape:
            raise ValueError("Episode risk histograms must have the same [N,K] shape")
        bin_count = int(self.background_histograms.shape[1])
        if self.bin_centers.shape != (bin_count,) or self.bin_edges.shape != (bin_count + 1,):
            raise ValueError("Episode histogram bin arrays have inconsistent shapes")
        tensors_to_check = (
            self.features,
            self.thresholds,
            self.threshold_logits,
            self.oracle_pd,
            self.oracle_fa,
            self.background_histograms,
            self.object_histograms,
            self.total_pixels,
            self.budgets,
            self.bin_edges,
            self.bin_centers,
        )
        if any(not torch.isfinite(tensor).all() for tensor in tensors_to_check):
            raise ValueError("Episode archive contains NaN or infinity")
        if torch.any(self.budgets <= 0) or torch.any(self.total_pixels <= 0):
            raise ValueError("Episode budgets and total pixel counts must be positive")
        if torch.any(self.thresholds < 0) or torch.any(
            self.thresholds > REJECT_ALL_THRESHOLD
        ):
            raise ValueError("Episode thresholds exceed the supported interval")
        if torch.any(self.thresholds[:, 1:] < self.thresholds[:, :-1]):
            raise ValueError("Episode thresholds must tighten monotonically with budget")
        if torch.any((self.oracle_pd < 0) | (self.oracle_pd > 1)) or torch.any(
            (self.oracle_fa < 0) | (self.oracle_fa > 1)
        ):
            raise ValueError("Episode oracle metrics must lie in [0,1]")
        if torch.any(
            self.oracle_fa
            > self.budgets[None, :] + 8 * torch.finfo(self.oracle_fa.dtype).eps
        ):
            raise ValueError("Episode oracle false-alarm values violate their budgets")
        if torch.any(self.background_histograms < 0) or torch.any(
            self.object_histograms < 0
        ):
            raise ValueError("Episode histograms cannot contain negative counts")
        domain_names = self.metadata.get("domain_names")
        if not isinstance(domain_names, list) or not domain_names or len(set(domain_names)) != len(domain_names):
            raise ValueError("Episode metadata domain_names must be unique and non-empty")
        if torch.any(self.domain_indices < 0) or torch.any(
            self.domain_indices >= len(domain_names)
        ):
            raise ValueError("Episode domain index is outside metadata domain_names")
        if int(self.metadata.get("num_episodes", -1)) != length:
            raise ValueError("Episode metadata num_episodes differs from archive")
        if int(self.metadata.get("feature_dim", -1)) != self.features.shape[1]:
            raise ValueError("Episode metadata feature_dim differs from archive")

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[index],
            "thresholds": self.thresholds[index],
            "threshold_logits": self.threshold_logits[index],
            "oracle_pd": self.oracle_pd[index],
            "oracle_fa": self.oracle_fa[index],
            "background_histogram": self.background_histograms[index],
            "object_histogram": self.object_histograms[index],
            "total_pixels": self.total_pixels[index],
            "domain_index": self.domain_indices[index],
        }
