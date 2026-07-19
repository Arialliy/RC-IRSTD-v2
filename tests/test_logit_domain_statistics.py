import numpy as np
import pytest

from risk_curve.domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    SourceStatisticsReference,
    append_source_distances,
    extract_logit_window_statistics,
    feature_schema_sha256,
    fit_source_reference,
    logit_feature_names,
)


def test_logit_statistics_are_finite_label_free_and_versioned():
    first = np.linspace(-30.0, 40.0, 17 * 19, dtype=np.float32).reshape(17, 19)
    second = np.flip(first, axis=1).copy()
    gray = np.linspace(0.0, 1.0, first.size, dtype=np.float32).reshape(first.shape)

    result = extract_logit_window_statistics([first, second], [gray, None])

    assert result.schema_version == LOGIT_STATISTICS_SCHEMA_VERSION
    assert result.names == tuple(logit_feature_names())
    assert result.values.shape == (len(result.names),)
    assert np.isfinite(result.values).all()
    assert len(feature_schema_sha256(LOGIT_STATISTICS_SCHEMA_VERSION)) == 64


def test_logit_statistics_retain_extreme_tail_lost_by_float32_sigmoid():
    base = np.full((16, 16), 18.0, dtype=np.float32)
    shifted = np.full((16, 16), 36.0, dtype=np.float32)
    # Both maps saturate after a float32 sigmoid, but raw-logit quantiles and
    # moments must still distinguish them.
    prob_base = (1.0 / (1.0 + np.exp(-base))).astype(np.float32)
    prob_shifted = (1.0 / (1.0 + np.exp(-shifted))).astype(np.float32)
    np.testing.assert_array_equal(prob_base, prob_shifted)

    base_stats = extract_logit_window_statistics([base])
    shifted_stats = extract_logit_window_statistics([shifted])
    assert not np.array_equal(base_stats.values, shifted_stats.values)
    q50 = base_stats.names.index("logit_q_0.5")
    assert base_stats.values[q50] == pytest.approx(18.0)
    assert shifted_stats.values[q50] == pytest.approx(36.0)


def test_logit_source_reference_is_schema_bound():
    sample = extract_logit_window_statistics(
        [np.linspace(-4.0, 8.0, 64, dtype=np.float32).reshape(8, 8)]
    )
    reference = fit_source_reference(
        {
            "source-a": np.stack([sample.values, sample.values + 0.01]),
            "source-b": np.stack([sample.values + 0.1, sample.values + 0.2]),
        },
        statistics_names=sample.names,
        statistics_schema_version=LOGIT_STATISTICS_SCHEMA_VERSION,
    )
    augmented = append_source_distances(sample, reference)
    assert augmented.values.size == sample.values.size + 4
    assert augmented.schema_version == LOGIT_STATISTICS_SCHEMA_VERSION

    wrong = SourceStatisticsReference(
        domain_names=reference.domain_names,
        centers=reference.centers,
        precision=reference.precision,
        statistics_names=reference.statistics_names,
    )
    with pytest.raises(ValueError, match="differs from the window"):
        append_source_distances(sample, wrong)


def test_logit_statistics_reject_nonfinite_maps():
    invalid = np.zeros((4, 4), dtype=np.float32)
    invalid[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite 2-D"):
        extract_logit_window_statistics([invalid])
