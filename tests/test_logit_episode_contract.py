import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pytest

import risk_curve.build_curve_episodes as episode_module
from evaluation.artifact_integrity import RAW_LOGIT_SCORE_REPRESENTATION
from evaluation.component_matching import connected_components
from risk_curve.build_curve_episodes import (
    COUNT_ALL_ADAPTATION_SCHEMA_VERSION,
    LOGIT_EPISODE_SCHEMA_VERSION,
    ScoreSample,
    _initialise_count_all_worker,
    _episodes_for_files,
    _pack_episodes,
    _precompute_adaptation_prediction_counts,
    _prediction_count_curves,
    _validate_count_all_workers,
    build_episode,
    load_score_sample,
)
from risk_curve.domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    feature_schema_sha256,
)
from risk_curve.curve_dataset import (
    load_curve_archive,
    validate_count_all_adaptation_contract,
)
from risk_curve.representation import (
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_REPRESENTATION,
    logit_threshold_grid_sha256,
)


def _sample(image_id: str, logits: np.ndarray, mask: np.ndarray | None) -> ScoreSample:
    raw = np.asarray(logits, dtype=np.float32)
    probability = (1.0 / (1.0 + np.exp(-raw.astype(np.float64)))).astype(
        np.float32
    )
    return ScoreSample(
        image_id=image_id,
        probability=probability,
        mask=None if mask is None else np.asarray(mask, dtype=np.uint8),
        gray=np.zeros(raw.shape, dtype=np.float32),
        source_path=f"/{image_id}.npz",
        raw_logit=raw,
    )


def _write_raw_record(path: Path, image_id: str, logits: np.ndarray) -> None:
    raw = np.asarray(logits, dtype=np.float32)
    probability = (1.0 / (1.0 + np.exp(-raw.astype(np.float64)))).astype(
        np.float32
    )
    np.savez_compressed(
        path,
        image_id=np.asarray(image_id),
        logit=raw,
        prob=probability,
        gray=np.zeros_like(raw),
        # Deliberately present and nonzero: precomputation must not read it.
        mask=np.ones(raw.shape, dtype=np.uint8),
        score_representation=np.asarray(RAW_LOGIT_SCORE_REPRESENTATION),
        logit_dtype=np.asarray("float32"),
        probability_transform=np.asarray("sigmoid"),
        probability_clipping=np.asarray("none"),
        inference_autocast_enabled=np.asarray(False),
    )


def _episode():
    adaptation = np.linspace(-12.0, 28.0, 64, dtype=np.float32).reshape(8, 8)
    evaluation = np.full((8, 8), -10.0, dtype=np.float32)
    evaluation[2, 3] = 22.0
    evaluation[5, 6] = 16.0
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2, 3] = 1
    grid = np.asarray([-8.0, 0.0, 12.0, 18.0, 24.0], dtype=np.float32)
    return build_episode(
        [_sample("support", adaptation, None)],
        grid,
        "pseudo-domain",
        evaluation_samples=[_sample("query", evaluation, mask)],
        representation=LOGIT_REPRESENTATION,
    )


def test_logit_episode_uses_raw_scores_end_to_end():
    episode = _episode()
    assert episode.representation == LOGIT_REPRESENTATION
    assert episode.statistics.schema_version == LOGIT_STATISTICS_SCHEMA_VERSION
    np.testing.assert_array_equal(episode.thresholds, [-8.0, 0.0, 12.0, 18.0, 24.0])
    # The false 16-logit candidate survives through threshold 12, then vanishes.
    np.testing.assert_array_equal(episode.pixel_fp_counts, [1, 1, 1, 0, 0])
    np.testing.assert_array_equal(episode.tp_object_counts, [1, 1, 1, 1, 0])
    np.testing.assert_array_equal(
        episode.adaptation_predicted_pixel_counts, [57, 45, 26, 16, 7]
    )
    np.testing.assert_array_equal(
        episode.adaptation_predicted_component_counts_raw, [1, 1, 1, 1, 1]
    )
    np.testing.assert_array_equal(
        episode.adaptation_predicted_component_counts_upper, [1, 1, 1, 1, 1]
    )
    assert episode.adaptation_total_pixels == 64


@pytest.mark.parametrize("connectivity", [1, 2, 4, 8])
@pytest.mark.parametrize("min_component_area", [1, 2, 3])
def test_fast_adaptation_counts_match_explicit_connected_components(
    connectivity: int, min_component_area: int
) -> None:
    rng = np.random.default_rng(19)
    logits = rng.normal(size=(7, 9)).astype(np.float32)
    grid = np.asarray([-1.5, -0.5, 0.0, 0.75, 1.5], dtype=np.float32)
    pixels, components = _prediction_count_curves(
        logits,
        grid,
        connectivity=connectivity,
        min_component_area=min_component_area,
    )
    expected_pixels: list[int] = []
    expected_components: list[int] = []
    for threshold in grid:
        labels, count = connected_components(
            logits >= threshold,
            connectivity=connectivity,
            min_component_area=min_component_area,
        )
        expected_pixels.append(int(np.count_nonzero(labels)))
        expected_components.append(int(count))
    np.testing.assert_array_equal(pixels, expected_pixels)
    np.testing.assert_array_equal(components, expected_components)


def test_count_all_adaptation_curves_do_not_depend_on_adaptation_mask() -> None:
    logits = np.asarray(
        [[3.0, -2.0, 3.0], [-2.0, 1.0, -2.0], [3.0, -2.0, 3.0]],
        dtype=np.float32,
    )
    grid = np.asarray([-1.0, 0.0, 2.0], dtype=np.float32)
    evaluation_logits = np.zeros((3, 3), dtype=np.float32)
    evaluation_mask = np.zeros((3, 3), dtype=np.uint8)
    episodes = []
    for adaptation_mask in (
        np.zeros((3, 3), dtype=np.uint8),
        np.ones((3, 3), dtype=np.uint8),
    ):
        episodes.append(
            build_episode(
                [_sample("a", logits, adaptation_mask)],
                grid,
                "domain",
                evaluation_samples=[
                    _sample("e", evaluation_logits, evaluation_mask)
                ],
                representation=LOGIT_REPRESENTATION,
            )
        )
    for field in (
        "adaptation_predicted_pixel_counts",
        "adaptation_predicted_component_counts_raw",
        "adaptation_predicted_component_counts_upper",
    ):
        np.testing.assert_array_equal(
            getattr(episodes[0], field), getattr(episodes[1], field)
        )
    assert episodes[0].adaptation_total_pixels == episodes[1].adaptation_total_pixels


def test_spawn_workers_match_sequential_counts_and_deduplicate_A_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_raw_record(
        first,
        "first",
        np.asarray([[2.0, -2.0], [-2.0, 2.0]], dtype=np.float32),
    )
    _write_raw_record(
        second,
        "second",
        np.asarray([[1.5, 0.5], [-0.5, -1.5]], dtype=np.float32),
    )
    paths = [first, second, first, second]
    calls: list[str] = []
    original = episode_module._count_all_counts_from_path

    def counted(path, thresholds, *, connectivity, min_component_area):
        calls.append(str(Path(path).resolve()))
        return original(
            path,
            thresholds,
            connectivity=connectivity,
            min_component_area=min_component_area,
        )

    monkeypatch.setattr(episode_module, "_count_all_counts_from_path", counted)
    sequential = _precompute_adaptation_prediction_counts(
        paths,
        grid,
        connectivity=2,
        min_component_area=1,
    )
    assert len(calls) == 2
    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialise_count_all_worker,
        initargs=(grid, 2, 1),
    ) as executor:
        parallel = _precompute_adaptation_prediction_counts(
            paths,
            grid,
            connectivity=2,
            min_component_area=1,
            executor=executor,
        )
    assert len(sequential) == len(parallel) == 2
    assert set(sequential) == set(parallel)
    for key in sequential:
        np.testing.assert_array_equal(
            sequential[key].predicted_pixel_counts,
            parallel[key].predicted_pixel_counts,
        )
        np.testing.assert_array_equal(
            sequential[key].predicted_component_counts_raw,
            parallel[key].predicted_component_counts_raw,
        )
        assert sequential[key].total_pixels == parallel[key].total_pixels


def test_episode_arrays_are_identical_for_one_and_two_count_all_workers(
    tmp_path: Path,
) -> None:
    grid = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    files: list[Path] = []
    for index in range(6):
        path = tmp_path / f"{index}.npz"
        logits = np.asarray(
            [[index - 2.0, -1.0], [0.5, 2.0 - index]], dtype=np.float32
        )
        _write_raw_record(path, f"image-{index}", logits)
        files.append(path)

    def args(workers: int) -> argparse.Namespace:
        return argparse.Namespace(
            adaptation_window=2,
            evaluation_window=1,
            stride=3,
            representation=LOGIT_REPRESENTATION,
            connectivity=2,
            min_component_area=1,
            matching_rule="overlap",
            centroid_distance=3.0,
            allow_cross_episode_role_reuse=False,
            count_all_workers=workers,
        )

    sequential, sequential_summary = _episodes_for_files(
        files, "pseudo", grid, args(1), None
    )
    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialise_count_all_worker,
        initargs=(grid, 2, 1),
    ) as executor:
        parallel, parallel_summary = _episodes_for_files(
            files, "pseudo", grid, args(2), None, executor
        )
    assert len(sequential) == len(parallel) == 2
    assert sequential_summary["count_all_unique_adaptation_files_precomputed"] == 4
    assert parallel_summary["count_all_unique_adaptation_files_precomputed"] == 4
    for left, right in zip(sequential, parallel):
        for field in (
            "pixel_fp_counts",
            "component_fp_counts",
            "tp_object_counts",
            "adaptation_predicted_pixel_counts",
            "adaptation_predicted_component_counts_raw",
            "adaptation_predicted_component_counts_upper",
        ):
            np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
        np.testing.assert_array_equal(left.statistics.values, right.statistics.values)
        assert left.adaptation_total_pixels == right.adaptation_total_pixels


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_count_all_workers_must_be_a_positive_integer(workers) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _validate_count_all_workers(workers)
    assert _validate_count_all_workers(1) == 1
    assert _validate_count_all_workers(2) == 2


def test_logit_episode_rejects_missing_logits_and_inf_in_model_grid():
    probability_only = ScoreSample(
        image_id="prob-only-a",
        probability=np.zeros((4, 4), dtype=np.float32),
        mask=np.zeros((4, 4), dtype=np.uint8),
        gray=None,
        source_path="/prob-only-a.npz",
    )
    probability_only_e = ScoreSample(
        image_id="prob-only-e",
        probability=probability_only.probability,
        mask=probability_only.mask,
        gray=None,
        source_path="/prob-only-e.npz",
    )
    with pytest.raises(ValueError, match="no float32 raw-logit"):
        build_episode(
            [probability_only],
            np.asarray([-1.0, 1.0], dtype=np.float32),
            "domain",
            evaluation_samples=[probability_only_e],
            representation=LOGIT_REPRESENTATION,
        )
    with pytest.raises(ValueError, match="all be finite"):
        build_episode(
            [_sample("a", np.zeros((4, 4)), None)],
            np.asarray([-1.0, np.inf], dtype=np.float32),
            "domain",
            evaluation_samples=[
                _sample("e", np.zeros((4, 4)), np.zeros((4, 4)))
            ],
            representation=LOGIT_REPRESENTATION,
        )


def test_logit_episode_pack_records_representation_and_semantic_hash(tmp_path: Path):
    episode = _episode()
    grid_hash = logit_threshold_grid_sha256(episode.thresholds)
    feature_hash = feature_schema_sha256(
        LOGIT_STATISTICS_SCHEMA_VERSION,
        statistics_names=episode.statistics.names,
    )
    provenance = {
        "protocol": "causal_adaptation_then_future_evaluation",
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_sha256": grid_hash,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_manifest_sha256": "a" * 64,
        "threshold_grid_outer_target_excluded": True,
        "threshold_grid_detector_protocol": "all_source_only_detector_folds",
        "threshold_grid_detector_checkpoint_sha256s": [
            "b" * 64,
            "c" * 64,
            "d" * 64,
        ],
        "threshold_grid_outer_detector_checkpoint_sha256": "d" * 64,
        "threshold_grid_episode_detector_checkpoint_sha256s": [
            "b" * 64,
            "c" * 64,
        ],
        "feature_schema_sha256": feature_hash,
    }
    output = tmp_path / "logit-episodes.npz"
    _pack_episodes([episode], output, provenance)

    with np.load(output, allow_pickle=False) as archive:
        assert str(archive["episode_schema_version"].item()) == (
            LOGIT_EPISODE_SCHEMA_VERSION
        )
        assert str(archive["representation"].item()) == LOGIT_REPRESENTATION
        assert str(archive["threshold_grid_sha256"].item()) == grid_hash
        assert str(archive["feature_schema_sha256"].item()) == feature_hash
        assert str(
            archive["threshold_grid_outer_detector_checkpoint_sha256"].item()
        ) == "d" * 64
        assert archive[
            "threshold_grid_episode_detector_checkpoint_sha256s"
        ].tolist() == ["b" * 64, "c" * 64]
        assert str(archive["count_all_adaptation_schema_version"].item()) == (
            COUNT_ALL_ADAPTATION_SCHEMA_VERSION
        )
        np.testing.assert_array_equal(
            archive["adaptation_predicted_pixel_counts"][0],
            episode.adaptation_predicted_pixel_counts,
        )
        np.testing.assert_array_equal(
            archive["adaptation_predicted_component_counts_raw"][0],
            episode.adaptation_predicted_component_counts_raw,
        )
        np.testing.assert_array_equal(
            archive["adaptation_predicted_component_counts_upper"][0],
            episode.adaptation_predicted_component_counts_upper,
        )
        assert int(archive["adaptation_total_pixels"][0]) == 64
        provenance = json.loads(str(archive["provenance_json"].item()))
        assert provenance["count_all_adaptation_masks_read"] is False
        assert json.loads(str(archive["support_ids"][0])) == ["support"]
        assert json.loads(str(archive["query_ids"][0])) == ["query"]
    loaded = load_curve_archive(output)
    contract = validate_count_all_adaptation_contract(loaded, required=True)
    assert contract["verified"] is True
    assert contract["adaptation_masks_read"] is False


def test_count_all_worker_provenance_does_not_change_semantic_contract(
    tmp_path: Path,
) -> None:
    episode = _episode()
    grid_hash = logit_threshold_grid_sha256(episode.thresholds)
    feature_hash = feature_schema_sha256(
        LOGIT_STATISTICS_SCHEMA_VERSION,
        statistics_names=episode.statistics.names,
    )
    base = {
        "protocol": "causal_adaptation_then_future_evaluation",
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_sha256": grid_hash,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_manifest_sha256": "a" * 64,
        "threshold_grid_outer_target_excluded": True,
        "threshold_grid_detector_protocol": "all_source_only_detector_folds",
        "threshold_grid_detector_checkpoint_sha256s": [
            "b" * 64,
            "c" * 64,
            "d" * 64,
        ],
        "threshold_grid_outer_detector_checkpoint_sha256": "d" * 64,
        "threshold_grid_episode_detector_checkpoint_sha256s": [
            "b" * 64,
            "c" * 64,
        ],
        "feature_schema_sha256": feature_hash,
    }
    contracts = []
    provenance_records = []
    for workers in (1, 2):
        output = tmp_path / f"workers-{workers}.npz"
        _pack_episodes(
            [episode],
            output,
            {
                **base,
                "count_all_workers": workers,
                "count_all_worker_start_method": (
                    "spawn" if workers > 1 else "sequential"
                ),
            },
        )
        archive = load_curve_archive(output)
        contracts.append(
            validate_count_all_adaptation_contract(archive, required=True)
        )
        provenance_records.append(
            json.loads(str(archive["provenance_json"].item()))
        )
    assert contracts[0] == contracts[1]
    assert provenance_records[0]["count_all_workers"] == 1
    assert provenance_records[1]["count_all_workers"] == 2
    assert provenance_records[0]["threshold_grid_sha256"] == (
        provenance_records[1]["threshold_grid_sha256"]
    )


def test_raw_logit_record_loader_fails_closed_on_embedded_contract(tmp_path: Path):
    path = tmp_path / "record.npz"
    logits = np.asarray([[-2.0, 3.0], [7.0, 20.0]], dtype=np.float32)
    probability = (1.0 / (1.0 + np.exp(-logits.astype(np.float64)))).astype(
        np.float32
    )
    np.savez_compressed(
        path,
        image_id=np.asarray("x"),
        logit=logits,
        prob=probability,
        gray=np.zeros((2, 2), dtype=np.float32),
        mask=np.zeros((2, 2), dtype=np.uint8),
        score_representation=np.asarray(RAW_LOGIT_SCORE_REPRESENTATION),
        logit_dtype=np.asarray("float32"),
        probability_transform=np.asarray("sigmoid"),
        probability_clipping=np.asarray("none"),
        inference_autocast_enabled=np.asarray(False),
    )
    sample = load_score_sample(path, representation=LOGIT_REPRESENTATION)
    np.testing.assert_array_equal(sample.raw_logit, logits)

    broken = tmp_path / "broken.npz"
    np.savez_compressed(broken, image_id=np.asarray("x"), prob=probability)
    with pytest.raises(ValueError, match="raw-logit record contract"):
        load_score_sample(
            broken,
            require_mask=False,
            representation=LOGIT_REPRESENTATION,
        )
