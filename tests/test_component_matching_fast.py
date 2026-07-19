from __future__ import annotations

import numpy as np
import pytest

import evaluation.component_matching as component_matching


def _reference_components(
    binary: np.ndarray,
    *,
    connectivity: int,
    min_component_area: int,
) -> tuple[np.ndarray, int]:
    image = component_matching._as_binary_2d(binary, "binary")
    return component_matching._connected_components_python(
        image,
        offsets=component_matching._neighbor_offsets(connectivity),
        min_component_area=min_component_area,
    )


@pytest.mark.skipif(
    component_matching._SCIPY_LABEL is None, reason="SciPy fast path unavailable"
)
@pytest.mark.parametrize("connectivity", [1, 2, 4, 8])
@pytest.mark.parametrize("min_component_area", [1, 2, 5, 17])
def test_scipy_components_exactly_match_reference_on_random_binary_maps(
    connectivity: int,
    min_component_area: int,
) -> None:
    rng = np.random.default_rng(20260715 + 31 * connectivity + min_component_area)
    cases = [
        np.zeros((19, 23), dtype=np.uint8),
        np.ones((7, 11), dtype=np.uint8),
        (rng.random((31, 37)) < 0.01).astype(np.uint8),
        (rng.random((31, 37)) < 0.10).astype(np.uint8),
        (rng.random((31, 37)) < 0.50).astype(np.uint8),
        (rng.random((1, 31, 37)) < 0.20).astype(np.uint8),
    ]
    for binary in cases:
        expected_labels, expected_count = _reference_components(
            binary,
            connectivity=connectivity,
            min_component_area=min_component_area,
        )
        actual_labels, actual_count = component_matching.connected_components(
            binary,
            connectivity=connectivity,
            min_component_area=min_component_area,
        )
        assert actual_count == expected_count
        np.testing.assert_array_equal(actual_labels, expected_labels)
        assert actual_labels.dtype == np.int32
        assert actual_labels.flags.c_contiguous


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("min_component_area", [1, 3, 9])
def test_public_fallback_exactly_matches_python_reference(
    monkeypatch: pytest.MonkeyPatch,
    connectivity: int,
    min_component_area: int,
) -> None:
    rng = np.random.default_rng(991 + connectivity + min_component_area)
    binary = rng.random((29, 43)) < 0.16
    expected = _reference_components(
        binary,
        connectivity=connectivity,
        min_component_area=min_component_area,
    )

    monkeypatch.setattr(component_matching, "_SCIPY_LABEL", None)
    actual = component_matching.connected_components(
        binary,
        connectivity=connectivity,
        min_component_area=min_component_area,
    )

    assert actual[1] == expected[1]
    np.testing.assert_array_equal(actual[0], expected[0])


@pytest.mark.skipif(
    component_matching._SCIPY_LABEL is None, reason="SciPy fast path unavailable"
)
def test_area_filtering_compacts_labels_in_first_pixel_order() -> None:
    binary = np.zeros((6, 9), dtype=np.uint8)
    binary[0, 0] = 1  # Removed component appears first in scan order.
    binary[0:2, 4] = 1
    binary[3, 1:4] = 1

    labels, count = component_matching.connected_components(
        binary, connectivity=4, min_component_area=2
    )

    assert count == 2
    assert labels[0, 0] == 0
    assert np.all(labels[0:2, 4] == 1)
    assert np.all(labels[3, 1:4] == 2)


@pytest.mark.skipif(
    component_matching._SCIPY_LABEL is None, reason="SciPy fast path unavailable"
)
@pytest.mark.parametrize("rule", ["overlap", "centroid"])
@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("min_component_area", [1, 4])
def test_match_components_fast_and_fallback_results_are_identical(
    monkeypatch: pytest.MonkeyPatch,
    rule: str,
    connectivity: int,
    min_component_area: int,
) -> None:
    rng = np.random.default_rng(
        1701 + connectivity + min_component_area + (0 if rule == "overlap" else 100)
    )
    prediction = rng.random((41, 47)) < 0.12
    ground_truth = rng.random((41, 47)) < 0.08

    fast = component_matching.match_components(
        prediction,
        ground_truth,
        rule=rule,
        centroid_distance=3.0,
        connectivity=connectivity,
        min_component_area=min_component_area,
    )
    monkeypatch.setattr(component_matching, "_SCIPY_LABEL", None)
    fallback = component_matching.match_components(
        prediction,
        ground_truth,
        rule=rule,
        centroid_distance=3.0,
        connectivity=connectivity,
        min_component_area=min_component_area,
    )

    assert fast == fallback


@pytest.mark.skipif(
    component_matching._SCIPY_LABEL is None, reason="SciPy fast path unavailable"
)
@pytest.mark.parametrize("connectivity", [4, 8])
def test_empty_map_fast_reference_and_fallback_are_identical(
    monkeypatch: pytest.MonkeyPatch,
    connectivity: int,
) -> None:
    binary = np.zeros((0, 9), dtype=np.uint8)
    expected = _reference_components(
        binary, connectivity=connectivity, min_component_area=3
    )
    fast = component_matching.connected_components(
        binary, connectivity=connectivity, min_component_area=3
    )
    monkeypatch.setattr(component_matching, "_SCIPY_LABEL", None)
    fallback = component_matching.connected_components(
        binary, connectivity=connectivity, min_component_area=3
    )

    for actual in (fast, fallback):
        assert actual[1] == expected[1] == 0
        np.testing.assert_array_equal(actual[0], expected[0])
