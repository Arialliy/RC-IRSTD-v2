import numpy as np
import pytest

from risk_curve.build_curve_episodes import (
    COMPONENT_RISK_SCHEMA_VERSION,
    ScoreSample,
    _pack_episodes,
    build_episode,
    monotone_upper_envelope,
)
from risk_curve.curve_dataset import (
    CurveDataset,
    load_curve_archive,
    validate_archive_compatibility,
)
from risk_curve.domain_statistics import STATISTICS_SCHEMA_VERSION


def _score_sample(
    image_id: str,
    probability: np.ndarray,
    mask: np.ndarray | None,
) -> ScoreSample:
    return ScoreSample(
        image_id=image_id,
        probability=np.asarray(probability, dtype=np.float32),
        mask=None if mask is None else np.asarray(mask, dtype=np.uint8),
        gray=np.asarray(probability, dtype=np.float32),
        source_path=f"/{image_id}.npz",
    )


def _nonmonotone_component_episode():
    shape = (5, 7)
    adaptation_probability = np.linspace(0.0, 1.0, np.prod(shape), dtype=np.float32).reshape(
        shape
    )
    evaluation_probability = np.zeros(shape, dtype=np.float32)
    # One low-threshold component splits into two, then only the left peak
    # survives.  Component counts are therefore [1, 2, 1].
    evaluation_probability[2, 1:6] = np.asarray(
        [0.95, 0.30, 0.30, 0.30, 0.70], dtype=np.float32
    )
    return build_episode(
        [_score_sample("adaptation", adaptation_probability, None)],
        np.asarray([0.2, 0.5, 0.9], dtype=np.float32),
        "test-domain",
        evaluation_samples=[
            _score_sample(
                "evaluation",
                evaluation_probability,
                np.zeros(shape, dtype=np.uint8),
            )
        ],
    )


def test_component_raw_upper_and_supervision_alias_round_trip(tmp_path):
    episode = _nonmonotone_component_episode()
    np.testing.assert_array_equal(episode.component_fp_counts, [1, 2, 1])

    total_megapixels = episode.total_pixels / 1_000_000.0
    expected_raw = np.log10(
        episode.component_fp_counts.astype(np.float64) / total_megapixels + 1e-6
    )
    expected_upper = monotone_upper_envelope(expected_raw)
    np.testing.assert_allclose(episode.component_log_risk_raw, expected_raw, rtol=1e-6)
    np.testing.assert_allclose(
        episode.component_log_risk_upper, expected_upper, rtol=1e-6
    )
    np.testing.assert_array_equal(
        episode.component_log_risk, episode.component_log_risk_upper
    )
    assert episode.component_log_risk_raw[0] < episode.component_log_risk_raw[1]
    assert np.all(np.diff(episode.component_log_risk_upper) <= 0.0)

    archive_path = tmp_path / "episodes.npz"
    _pack_episodes([episode], archive_path, {"protocol": "test"})
    archive = load_curve_archive(archive_path)

    assert str(archive["component_risk_schema_version"].item()) == (
        COMPONENT_RISK_SCHEMA_VERSION
    )
    assert str(archive["component_log_risk_alias"].item()) == (
        "component_log_risk_upper"
    )
    np.testing.assert_array_equal(
        archive["component_log_risk_raw"][0], episode.component_log_risk_raw
    )
    np.testing.assert_array_equal(
        archive["component_log_risk_upper"][0], episode.component_log_risk_upper
    )
    np.testing.assert_array_equal(
        archive["component_log_risk"], archive["component_log_risk_upper"]
    )

    sample = CurveDataset(archive_path)[0]
    assert set(sample) == {
        "statistics",
        "pixel_log_risk",
        "component_log_risk",
        "component_log_risk_raw",
        "component_log_risk_upper",
    }
    np.testing.assert_array_equal(
        sample["component_log_risk"].numpy(),
        sample["component_log_risk_upper"].numpy(),
    )


def test_legacy_alias_only_archive_remains_readable_without_fabricated_raw(tmp_path):
    path = tmp_path / "legacy.npz"
    thresholds = np.asarray([0.2, 0.5, 0.9], dtype=np.float32)
    component = np.asarray([[3.0, 2.0, 1.0]], dtype=np.float32)
    np.savez_compressed(
        path,
        statistics=np.asarray([[0.0, 1.0]], dtype=np.float32),
        statistics_names=np.asarray(["a", "b"]),
        statistics_schema_version=np.asarray(STATISTICS_SCHEMA_VERSION),
        pixel_log_risk=np.asarray([[-1.0, -2.0, -3.0]], dtype=np.float32),
        component_log_risk=component,
        pd_curve=np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32),
        thresholds=thresholds,
    )

    archive = load_curve_archive(path)
    assert "component_log_risk_raw" not in archive
    np.testing.assert_array_equal(archive["component_log_risk_upper"], component)
    sample = CurveDataset(path)[0]
    assert "component_log_risk_raw" not in sample
    np.testing.assert_array_equal(
        sample["component_log_risk_upper"].numpy(), component[0]
    )


def test_declared_component_schema_rejects_a_non_suffix_max_upper_curve(tmp_path):
    episode = _nonmonotone_component_episode()
    valid_path = tmp_path / "valid.npz"
    _pack_episodes([episode], valid_path, {"protocol": "test"})
    with np.load(valid_path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["component_log_risk_upper"] = np.asarray(
        payload["component_log_risk_raw"], dtype=np.float32
    )
    corrupt_path = tmp_path / "corrupt.npz"
    np.savez_compressed(corrupt_path, **payload)

    with pytest.raises(ValueError, match="suffix-maximum envelope"):
        load_curve_archive(corrupt_path)


def test_training_compatibility_accepts_matching_upper_and_legacy_contracts(tmp_path):
    episode = _nonmonotone_component_episode()
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    _pack_episodes([episode], first_path, {"protocol": "test"})
    _pack_episodes([episode], second_path, {"protocol": "test"})
    first = load_curve_archive(first_path)
    second = load_curve_archive(second_path)
    assert validate_archive_compatibility(first, second) == tuple(
        first["statistics_names"].tolist()
    )

    legacy_path = tmp_path / "legacy-compatible.npz"
    np.savez_compressed(
        legacy_path,
        statistics=first["statistics"],
        statistics_names=first["statistics_names"],
        statistics_schema_version=first["statistics_schema_version"],
        pixel_log_risk=first["pixel_log_risk"],
        component_log_risk=first["component_log_risk_upper"],
        pd_curve=first["pd_curve"],
        thresholds=first["thresholds"],
    )
    legacy = load_curve_archive(legacy_path)
    assert validate_archive_compatibility(legacy, legacy) == tuple(
        legacy["statistics_names"].tolist()
    )


def test_training_compatibility_rejects_new_legacy_mix_and_raw_alias(tmp_path):
    episode = _nonmonotone_component_episode()
    upper_path = tmp_path / "upper.npz"
    _pack_episodes([episode], upper_path, {"protocol": "test"})
    with np.load(upper_path, allow_pickle=False) as archive:
        upper_payload = {key: archive[key] for key in archive.files}
    upper = load_curve_archive(upper_path)

    legacy_path = tmp_path / "legacy.npz"
    legacy_payload = {
        key: value
        for key, value in upper_payload.items()
        if key
        not in {
            "component_risk_schema_version",
            "component_log_risk_alias",
            "component_log_risk_raw",
            "component_log_risk_upper",
        }
    }
    np.savez_compressed(legacy_path, **legacy_payload)
    legacy = load_curve_archive(legacy_path)
    with pytest.raises(ValueError, match="versioned and legacy"):
        validate_archive_compatibility(upper, legacy)

    # Raw supervision remains loadable for diagnostics, but is intentionally
    # ineligible as the main risk-curve training target.
    raw_path = tmp_path / "raw-diagnostic.npz"
    raw_payload = dict(upper_payload)
    raw_payload["component_log_risk"] = raw_payload["component_log_risk_raw"]
    raw_payload["component_log_risk_alias"] = np.asarray(
        "component_log_risk_raw"
    )
    np.savez_compressed(raw_path, **raw_payload)
    raw = load_curve_archive(raw_path)
    raw_sample = CurveDataset(raw_path)[0]
    np.testing.assert_array_equal(
        raw_sample["component_log_risk"].numpy(),
        raw_sample["component_log_risk_raw"].numpy(),
    )
    with pytest.raises(ValueError, match="diagnostic evidence only"):
        validate_archive_compatibility(raw, raw)
    with pytest.raises(ValueError, match="alias values differ"):
        validate_archive_compatibility(upper, raw)
