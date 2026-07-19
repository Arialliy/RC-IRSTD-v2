import pytest

from data_ext.lodo_split import build_lodo_folds


def test_nested_lodo_excludes_outer_and_pseudo_targets():
    folds = build_lodo_folds(["A", "B", "C", "D"], outer_target="D")
    assert {fold.pseudo_target for fold in folds} == {"A", "B", "C"}
    for fold in folds:
        assert "D" not in fold.detector_sources
        assert fold.pseudo_target not in fold.detector_sources
        assert len(fold.detector_sources) == 2


def test_nested_lodo_requires_enough_unique_domains():
    with pytest.raises(ValueError):
        build_lodo_folds(["A", "B"], outer_target="B")
    with pytest.raises(ValueError):
        build_lodo_folds(["A", "A", "B"])
