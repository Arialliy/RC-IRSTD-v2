from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from risk_curve.anchor_warp_training import (
    ANCHOR_WARP_NORMALIZATION_SCHEMA_VERSION,
    RobustMedianMADScaler,
    deterministic_episode_folds,
    predict_frozen_anchor_warp,
    prepare_anchor_warp_training_data,
    train_anchor_warp_train_only,
)
from risk_curve.count_all_anchor import (
    derive_anchor_log_curves,
    validate_count_all_anchor_archive,
)
from risk_curve.curve_dataset import COUNT_ALL_ADAPTATION_SCHEMA_VERSION
from risk_curve.representation import (
    LOGIT_REPRESENTATION,
    logit_threshold_grid_sha256,
)


def _archive(rows: int = 10, thresholds: int = 17) -> dict[str, np.ndarray]:
    grid = np.linspace(-4.0, 20.0, thresholds, dtype=np.float32)
    grid_hash = logit_threshold_grid_sha256(grid)
    exposure = np.full(rows, 1_000_000, dtype=np.int64)
    pixel_counts = np.empty((rows, thresholds), dtype=np.int64)
    component_raw = np.empty_like(pixel_counts)
    for row in range(rows):
        pixel_counts[row] = np.maximum(
            0, (thresholds - 1 - np.arange(thresholds)) * (20 + row)
        )
        component_raw[row] = np.maximum(
            0, (thresholds - 1 - np.arange(thresholds)) * (2 + row % 3)
        )
    component_upper = np.maximum.accumulate(component_raw[:, ::-1], axis=1)[
        :, ::-1
    ]
    statistics = np.stack(
        [
            np.linspace(-1.0 + feature * 0.01, 1.0 + feature * 0.01, rows)
            for feature in range(119)
        ],
        axis=1,
    ).astype(np.float32)
    provenance = {
        "representation": LOGIT_REPRESENTATION,
        "connectivity": 2,
        "min_component_area": 1,
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
    archive: dict[str, np.ndarray] = {
        "statistics": statistics,
        "thresholds": grid,
        "representation": np.asarray(LOGIT_REPRESENTATION),
        "threshold_grid_sha256": np.asarray(grid_hash),
        "adaptation_predicted_pixel_counts": pixel_counts,
        "adaptation_predicted_component_counts_raw": component_raw,
        "adaptation_predicted_component_counts_upper": component_upper,
        "adaptation_total_pixels": exposure,
        "count_all_adaptation_schema_version": np.asarray(
            COUNT_ALL_ADAPTATION_SCHEMA_VERSION
        ),
        "provenance_json": np.asarray(json.dumps(provenance, sort_keys=True)),
        "pseudo_targets": np.asarray(["NUDT-SIRST"] * rows),
        "adaptation_ids": np.asarray(
            [json.dumps([f"a-{row}"]) for row in range(rows)]
        ),
        "evaluation_ids": np.asarray(
            [json.dumps([f"e-{row}"]) for row in range(rows)]
        ),
    }
    anchors = validate_count_all_anchor_archive(archive)
    pixel_anchor, component_anchor = derive_anchor_log_curves(anchors)
    # A small episode-specific conservative translation is learnable while
    # preserving both monotonicity and the registered physical floors.
    shift = np.linspace(0.02, 0.20, rows, dtype=np.float32)[:, None]
    archive["pixel_log_risk"] = pixel_anchor + shift
    archive["component_log_risk_upper"] = component_anchor + 0.5 * shift
    return archive


def test_vectorized_log_anchor_formula_is_exact_and_read_only() -> None:
    archive = _archive(rows=2)
    anchors = validate_count_all_anchor_archive(archive)
    pixel, component = derive_anchor_log_curves(anchors)
    expected_pixel = np.log10(
        anchors.pixel_counts.astype(np.float64)
        / anchors.total_pixels[:, None]
        + 1.0e-12
    ).astype(np.float32)
    expected_component = np.log10(
        anchors.component_counts_upper.astype(np.float64)
        / (anchors.total_pixels[:, None] / 1_000_000.0)
        + 1.0e-6
    ).astype(np.float32)
    np.testing.assert_array_equal(pixel, expected_pixel)
    np.testing.assert_array_equal(component, expected_component)
    assert pixel[0, -1] == np.float32(-12.0)
    assert component[0, -1] == np.float32(-6.0)
    assert not pixel.flags.writeable and not component.flags.writeable


def test_robust_scaler_uses_median_mad_floor_and_fixed_clip() -> None:
    statistics = np.asarray(
        [[0.0, 2.0], [1.0, 2.0], [100.0, 2.0]], dtype=np.float32
    )
    scaler = RobustMedianMADScaler.fit(statistics, [0, 1])
    assert scaler.schema_version == ANCHOR_WARP_NORMALIZATION_SCHEMA_VERSION
    np.testing.assert_array_equal(scaler.median, [0.5, 2.0])
    assert scaler.scale[1] == np.float32(1.0e-6)
    transformed = scaler.transform(statistics)
    assert transformed[2, 0] == pytest.approx(4.0)
    assert np.all(np.abs(transformed) <= 4.0)
    restored = RobustMedianMADScaler.from_payload(scaler.payload())
    np.testing.assert_array_equal(restored.transform(statistics), transformed)


def test_episode_folds_are_balanced_complete_and_row_order_invariant() -> None:
    keys = tuple(f"episode-{index}" for index in range(13))
    folds = deterministic_episode_folds(keys, num_folds=5, seed=3407)
    assert sorted(index for fold in folds for index in fold) == list(range(13))
    assert max(map(len, folds)) - min(map(len, folds)) <= 1
    assignment = {
        keys[index]: fold_index
        for fold_index, fold in enumerate(folds)
        for index in fold
    }
    order = [8, 2, 12, 0, 7, 4, 10, 1, 5, 11, 3, 9, 6]
    shuffled = tuple(keys[index] for index in order)
    shuffled_folds = deterministic_episode_folds(
        shuffled, num_folds=5, seed=3407
    )
    shuffled_assignment = {
        shuffled[index]: fold_index
        for fold_index, fold in enumerate(shuffled_folds)
        for index in fold
    }
    assert shuffled_assignment == assignment


def test_prepare_training_data_rejects_any_cross_episode_id_reuse() -> None:
    archive = _archive()
    archive["evaluation_ids"] = archive["evaluation_ids"].copy()
    archive["evaluation_ids"][1] = archive["evaluation_ids"][0]
    with pytest.raises(ValueError, match="globally unique"):
        prepare_anchor_warp_training_data(archive, left_radius=3, right_radius=2)


def test_train_only_cv_is_deterministic_and_every_oof_row_is_excluded() -> None:
    torch.set_num_threads(1)
    data = prepare_anchor_warp_training_data(
        _archive(), left_radius=3, right_radius=2
    )
    kwargs = {
        "seed": 17,
        "num_folds": 5,
        "max_epochs": 6,
        "patience": 3,
        "learning_rate": 3.0e-3,
        "oof_inflation_quantile": 0.90,
        "device": "cpu",
    }
    first = train_anchor_warp_train_only(data, **kwargs)
    second = train_anchor_warp_train_only(data, **kwargs)
    assert first.fixed_epoch == second.fixed_epoch
    assert first.frozen_model_semantic_sha256 == (
        second.frozen_model_semantic_sha256
    )
    assert first.pixel_oof_inflation >= 0.0
    assert first.component_oof_inflation >= 0.0
    assert len(first.cv_aggregate_history) <= kwargs["max_epochs"]
    seen: list[int] = []
    for fold in first.folds:
        assert set(fold.train_indices).isdisjoint(fold.validation_indices)
        assert fold.best_epoch == first.fixed_epoch
        seen.extend(fold.validation_indices)
    assert sorted(seen) == list(range(data.num_episodes))
    prediction = predict_frozen_anchor_warp(
        first,
        statistics=data.statistics,
        pixel_anchor=data.pixel_anchor,
        component_anchor=data.component_anchor,
    )
    for key in ("pixel_log_risk", "component_log_risk"):
        curve = prediction[key]
        assert curve.shape == (data.num_episodes, data.num_thresholds)
        assert np.isfinite(curve).all()
        assert np.all(np.diff(curve, axis=1) <= 1.0e-6)
