from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from rc_irstd.evaluation.metrics import (
    REJECT_ALL_LATENT_LOGIT,
    REJECT_ALL_THRESHOLD,
    evaluate_threshold,
    oracle_thresholds,
    risk_histograms,
)
from rc_irstd.evaluation.score_store import ScoreItem, ScoreStore, save_score_item
from rc_irstd.features import FeatureSpec, extract_window_features, feature_names
from rc_irstd.meta import probability_to_logit_scalar


def make_item() -> ScoreItem:
    probability = np.asarray(
        [
            [0.95, 0.90, 0.20, 0.10],
            [0.80, 0.70, 0.30, 0.10],
            [0.60, 0.50, 0.40, 0.20],
            [0.20, 0.10, 0.05, 0.01],
        ],
        dtype=np.float32,
    )
    mask = np.zeros_like(probability, dtype=np.uint8)
    mask[0:2, 0:2] = 1
    gray = probability.copy()
    return ScoreItem(
        probability=probability,
        mask=mask,
        gray=gray,
        image_id="sample",
        dataset_name="domain",
        sequence_id="sequence",
        original_hw=probability.shape,
        has_mask=True,
    )


def test_oracle_respects_pixel_budget() -> None:
    item = make_item()
    thresholds, pds, fas = oracle_thresholds([item], [0.25, 0.0])
    assert thresholds.shape == (2,)
    assert np.all(fas <= np.asarray([0.25, 0.0]) + 1e-12)
    assert np.all(pds >= 0)
    metrics = evaluate_threshold([item], float(thresholds[0]))
    assert metrics.fa_pixel <= 0.25


def test_unlabeled_feature_dimension_and_finiteness() -> None:
    item = make_item()
    spec = FeatureSpec(probability_bins=8, logit_bins=8, peak_bins=8)
    features = extract_window_features([item, item], spec)
    assert features.shape == (len(feature_names(spec)),)
    assert np.isfinite(features).all()


def test_risk_histograms_preserve_all_background_and_object_counts() -> None:
    item = make_item()
    # Deliberately narrow limits force values into both boundary bins.
    edges = np.linspace(-1.0, 1.0, 9, dtype=np.float32)
    background_hist, object_hist, total_pixels = risk_histograms([item], edges)
    assert int(background_hist.sum()) == int(np.count_nonzero(item.mask == 0))
    assert int(object_hist.sum()) == 1
    assert total_pixels == item.probability.size


def test_component_metrics_count_only_non_overlapping_predictions_as_false() -> None:
    probability = np.zeros((6, 6), dtype=np.float32)
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    probability[1:3, 1:3] = 0.9  # one true-positive component
    probability[4:6, 4:6] = 0.8  # one false-positive component
    item = ScoreItem(
        probability=probability,
        mask=mask,
        gray=probability.copy(),
        image_id="components",
        dataset_name="domain",
        sequence_id="sequence",
        original_hw=probability.shape,
        has_mask=True,
    )
    metrics = evaluate_threshold([item], 0.5)
    assert metrics.detected_objects == 1
    assert metrics.false_positive_components == 1
    assert metrics.false_positive_pixels == 4


def test_evaluate_threshold_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="at least one query item"):
        evaluate_threshold([], 0.5)


def test_score_one_uses_a_real_reject_all_threshold() -> None:
    probability = np.ones((2, 2), dtype=np.float32)
    item = ScoreItem(
        probability=probability,
        mask=np.zeros_like(probability, dtype=np.uint8),
        gray=probability.copy(),
        image_id="saturated",
        dataset_name="domain",
        sequence_id="sequence",
        original_hw=probability.shape,
        has_mask=True,
    )
    thresholds, _, false_alarm = oracle_thresholds([item], [0.0])
    assert float(thresholds[0]) == REJECT_ALL_THRESHOLD
    assert float(thresholds[0]) > 1.0
    assert float(false_alarm[0]) == 0.0
    latent = probability_to_logit_scalar(thresholds)
    assert float(latent[0]) == REJECT_ALL_LATENT_LOGIT


def test_legacy_score_item_save_rejects_traversal_image_id(tmp_path) -> None:
    item = replace(make_item(), image_id="../outside")
    with pytest.raises(ValueError, match="Unsafe legacy score-map filename"):
        save_score_item(tmp_path / "store", item)
    assert not (tmp_path / "outside.npz").exists()


def test_legacy_score_store_rejects_traversal_and_accepts_one_level_file(tmp_path) -> None:
    store_root = tmp_path / "store"
    path = save_score_item(store_root, make_item())
    manifest_path = store_root / "manifest.json"
    manifest_path.write_text(
        json.dumps({"dataset_name": "domain", "entries": [{"file": path.name}]}),
        encoding="utf-8",
    )
    store = ScoreStore(store_root)
    assert store[0].image_id == "sample"

    manifest_path.write_text(
        json.dumps({"dataset_name": "domain", "entries": [{"file": "../outside.npz"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsafe legacy score-map filename"):
        ScoreStore(store_root)
