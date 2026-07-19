from __future__ import annotations

import numpy as np
import pytest

from evaluation.counterfactual_tail_signal import (
    matched_tail_pair_metrics,
    robust_cross_scale_evidence,
)


def test_robust_cross_scale_evidence_uses_median_minus_mad() -> None:
    drops = np.asarray([[3.0, 2.0, 1.0], [-1.0, 1.0, 3.0]])
    evidence, consensus = robust_cross_scale_evidence(drops)
    np.testing.assert_allclose(evidence, [1.0, -1.0])
    np.testing.assert_allclose(consensus, [1.0, 2.0 / 3.0])


def test_positive_affine_scaling_cannot_change_matched_pair_order() -> None:
    target = np.asarray([2.0, 1.0, 0.0])
    clutter = np.asarray([1.0, 1.0, 1.0])
    metrics = matched_tail_pair_metrics(
        3.0 * target + 2.0,
        3.0 * clutter + 2.0,
        5.0 * target - 4.0,
        5.0 * clutter - 4.0,
    )
    assert metrics.raw_tail_pair_accuracy == pytest.approx(1.0 / 3.0)
    assert metrics.evidence_tail_pair_accuracy == pytest.approx(1.0 / 3.0)
    assert metrics.tail_pair_delta == pytest.approx(0.0)
    assert metrics.net_repair == pytest.approx(0.0)
    assert metrics.matched_pair_order_inversion_fraction == pytest.approx(0.0)


def test_nonmonotone_evidence_reports_repairs_harms_and_net_delta() -> None:
    metrics = matched_tail_pair_metrics(
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.0, -1.0],
    )
    assert metrics.num_pairs == 3
    assert metrics.raw_tail_pair_accuracy == pytest.approx(1.0 / 3.0)
    assert metrics.evidence_tail_pair_accuracy == pytest.approx(2.0 / 3.0)
    assert metrics.tail_pair_delta == pytest.approx(1.0 / 3.0)
    assert metrics.baseline_inversion_repair_rate == pytest.approx(2.0 / 3.0)
    assert metrics.harmful_inversion_rate == pytest.approx(1.0 / 3.0)
    assert metrics.net_repair == pytest.approx(1.0 / 3.0)
    assert metrics.matched_pair_order_inversion_fraction == pytest.approx(1.0)
    assert metrics.factual_counterfactual_margin_median == pytest.approx(1.0)


@pytest.mark.parametrize("drops", (np.empty((0, 3)), [[1.0, np.nan]]))
def test_cross_scale_evidence_rejects_empty_or_nonfinite(drops: object) -> None:
    with pytest.raises(ValueError):
        robust_cross_scale_evidence(drops)


def test_matched_pair_metrics_reject_mismatched_or_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="identical length"):
        matched_tail_pair_metrics([1.0], [0.0, 1.0], [1.0], [0.0])
    with pytest.raises(ValueError, match="finite"):
        matched_tail_pair_metrics([np.inf], [0.0], [1.0], [0.0])
    with pytest.raises(ValueError, match="non-empty"):
        matched_tail_pair_metrics([], [], [], [])


def _disk_mask(
    height: int,
    width: int,
    *,
    center_yx: tuple[float, float],
    radius: float,
) -> np.ndarray:
    yy, xx = np.indices((height, width), dtype=np.float64)
    return np.hypot(yy - center_yx[0], xx - center_yx[1]) <= radius


def test_opposite_ray_fill_reconstructs_planar_background_and_preserves_outside() -> None:
    from evaluation.counterfactual_tail_signal import (
        opposite_ray_annular_core_fill,
    )

    height = 17
    width = 17
    center = (8.0, 8.0)
    yy, xx = np.indices((height, width), dtype=np.float32)
    plane = 2.0 + 0.25 * yy - 0.5 * xx
    background = np.stack((plane, -2.0 * plane + 3.0), axis=0).astype(
        np.float32
    )
    core = _disk_mask(height, width, center_yx=center, radius=2.25)
    factual = background.copy()
    factual[:, core] += np.asarray([[100.0], [50.0]], dtype=np.float32)
    factual_before = factual.copy()

    counterfactual = opposite_ray_annular_core_fill(
        factual,
        center_yx=center,
        core_mask=core,
        annulus_inner_radius=4.0,
        annulus_outer_radius=6.0,
        radial_samples=5,
        center_angular_samples=32,
    )

    assert counterfactual.dtype == np.float32
    np.testing.assert_array_equal(factual, factual_before)
    np.testing.assert_array_equal(counterfactual[:, ~core], factual[:, ~core])
    np.testing.assert_allclose(
        counterfactual[:, core],
        background[:, core],
        rtol=0.0,
        atol=2.0e-6,
    )


def test_opposite_ray_fill_keeps_constant_field_and_returns_copy() -> None:
    from evaluation.counterfactual_tail_signal import (
        opposite_ray_annular_core_fill,
    )

    image = np.full((13, 13), 7.25, dtype=np.float32)
    core = _disk_mask(13, 13, center_yx=(6.0, 6.0), radius=1.5)
    result = opposite_ray_annular_core_fill(
        image,
        center_yx=(6.0, 6.0),
        core_mask=core,
        annulus_inner_radius=3.0,
        annulus_outer_radius=5.0,
    )
    np.testing.assert_array_equal(result, image)
    assert result is not image


def test_opposite_ray_fill_fails_closed_on_unobservable_or_invalid_inputs() -> None:
    from evaluation.counterfactual_tail_signal import (
        opposite_ray_annular_core_fill,
    )

    image = np.zeros((13, 13), dtype=np.float32)
    core = _disk_mask(13, 13, center_yx=(6.0, 6.0), radius=1.5)
    arguments = {
        "center_yx": (6.0, 6.0),
        "core_mask": core,
        "annulus_inner_radius": 3.0,
        "annulus_outer_radius": 5.0,
    }
    with pytest.raises(ValueError, match="float32"):
        opposite_ray_annular_core_fill(image.astype(np.float64), **arguments)
    nonfinite = image.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        opposite_ray_annular_core_fill(nonfinite, **arguments)
    with pytest.raises(ValueError, match="boolean"):
        opposite_ray_annular_core_fill(
            image,
            **{**arguments, "core_mask": core.astype(np.uint8)},
        )
    with pytest.raises(ValueError, match="at least one"):
        opposite_ray_annular_core_fill(
            image,
            **{**arguments, "core_mask": np.zeros_like(core)},
        )
    with pytest.raises(ValueError, match="full annulus"):
        opposite_ray_annular_core_fill(
            image,
            **{
                **arguments,
                "center_yx": (2.0, 2.0),
                "core_mask": _disk_mask(
                    13,
                    13,
                    center_yx=(2.0, 2.0),
                    radius=1.0,
                ),
            },
        )
    oversized_core = _disk_mask(13, 13, center_yx=(6.0, 6.0), radius=3.0)
    with pytest.raises(ValueError, match="strictly inside"):
        opposite_ray_annular_core_fill(
            image,
            **{**arguments, "core_mask": oversized_core},
        )
    with pytest.raises(ValueError, match="0 < inner < outer"):
        opposite_ray_annular_core_fill(
            image,
            **{**arguments, "annulus_inner_radius": 5.0},
        )
    with pytest.raises(ValueError, match="radial_samples"):
        opposite_ray_annular_core_fill(
            image,
            **arguments,
            radial_samples=1,
        )


def test_strict_rank_inversion_is_zero_for_positive_affine_transform() -> None:
    from evaluation.counterfactual_tail_signal import (
        strict_pairwise_rank_inversion_metrics,
    )

    raw = np.asarray([3.0, 2.0, 1.0])
    metrics = strict_pairwise_rank_inversion_metrics(raw, 4.0 * raw - 7.0)
    assert metrics.num_candidates == 3
    assert metrics.num_comparable_pairs == 3
    assert metrics.num_strict_inversions == 0
    assert metrics.strict_inversion_fraction == pytest.approx(0.0)


def test_strict_rank_inversion_counts_reversals_and_excludes_ties() -> None:
    from evaluation.counterfactual_tail_signal import (
        strict_pairwise_rank_inversion_metrics,
    )

    metrics = strict_pairwise_rank_inversion_metrics(
        [2.0, 2.0, 1.0],
        [0.0, 3.0, 1.0],
    )
    assert metrics.num_candidates == 3
    assert metrics.num_comparable_pairs == 2
    assert metrics.num_strict_inversions == 1
    assert metrics.strict_inversion_fraction == pytest.approx(0.5)


def test_strict_rank_inversion_fails_closed_without_comparable_pairs() -> None:
    from evaluation.counterfactual_tail_signal import (
        strict_pairwise_rank_inversion_metrics,
    )

    with pytest.raises(ValueError, match="identical length"):
        strict_pairwise_rank_inversion_metrics([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="at least two"):
        strict_pairwise_rank_inversion_metrics([1.0], [2.0])
    with pytest.raises(ValueError, match="no strict non-tied"):
        strict_pairwise_rank_inversion_metrics([1.0, 1.0], [2.0, 3.0])


def _match_id_signature(matches: object) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (match.target_id, match.clutter_id, match.match_slot)
        for match in matches
    )


def test_stratified_match_is_input_order_invariant_and_ties_break_by_id() -> None:
    from evaluation.counterfactual_tail_signal import (
        deterministic_stratified_greedy_match,
    )

    canonical = deterministic_stratified_greedy_match(
        target_ids=["t2", "t1"],
        clutter_ids=["c4", "c2", "c3", "c1"],
        target_strata=["tail", "tail"],
        clutter_strata=["tail", "tail", "tail", "tail"],
        target_features=[[10.0], [0.0]],
        clutter_features=[[11.0], [1.0], [9.0], [-1.0]],
        clutter_per_target=2,
    )
    permuted = deterministic_stratified_greedy_match(
        target_ids=["t1", "t2"],
        clutter_ids=["c1", "c3", "c2", "c4"],
        target_strata=["tail", "tail"],
        clutter_strata=["tail", "tail", "tail", "tail"],
        target_features=[[0.0], [10.0]],
        clutter_features=[[-1.0], [9.0], [1.0], [11.0]],
        clutter_per_target=2,
    )
    expected = (
        ("t1", "c1", 0),
        ("t1", "c2", 1),
        ("t2", "c3", 0),
        ("t2", "c4", 1),
    )
    assert _match_id_signature(canonical) == expected
    assert _match_id_signature(permuted) == expected
    assert len({match.clutter_id for match in canonical}) == 4


def test_stratified_match_never_crosses_registered_strata() -> None:
    from evaluation.counterfactual_tail_signal import (
        deterministic_stratified_greedy_match,
    )

    matches = deterministic_stratified_greedy_match(
        target_ids=["target-a", "target-b"],
        clutter_ids=["clutter-b", "clutter-a"],
        target_strata=["a", "b"],
        clutter_strata=["b", "a"],
        target_features=[[0.0, 0.0], [0.0, 0.0]],
        clutter_features=[[0.0, 0.0], [0.0, 0.0]],
    )
    assert tuple((match.target_id, match.clutter_id) for match in matches) == (
        ("target-a", "clutter-a"),
        ("target-b", "clutter-b"),
    )
    assert all(
        match.stratum == match.target_id.removeprefix("target-")
        for match in matches
    )


def test_stratified_match_fails_closed_on_invalid_or_incomplete_matrix() -> None:
    from evaluation.counterfactual_tail_signal import (
        deterministic_stratified_greedy_match,
    )

    base = {
        "target_ids": ["t1", "t2"],
        "clutter_ids": ["c1"],
        "target_strata": ["a", "a"],
        "clutter_strata": ["a"],
        "target_features": [[0.0], [1.0]],
        "clutter_features": [[0.0]],
    }
    with pytest.raises(ValueError, match="insufficient clutter coverage"):
        deterministic_stratified_greedy_match(**base)
    with pytest.raises(ValueError, match="unique"):
        deterministic_stratified_greedy_match(
            **{**base, "target_ids": ["t1", "t1"]},
        )
    with pytest.raises(ValueError, match="finite"):
        deterministic_stratified_greedy_match(
            **{
                **base,
                "target_ids": ["t1"],
                "target_strata": ["a"],
                "target_features": [[np.nan]],
            },
        )
    with pytest.raises(ValueError, match="positive integer"):
        deterministic_stratified_greedy_match(
            **base,
            clutter_per_target=False,
        )
    with pytest.raises(ValueError, match="feature matrices"):
        deterministic_stratified_greedy_match(
            **{**base, "target_features": [0.0, 1.0]},
        )


def test_counterfactual_signal_module_remains_pure_and_io_free() -> None:
    import ast
    from pathlib import Path

    module_path = (
        Path(__file__).resolve().parents[1]
        / "evaluation/counterfactual_tail_signal.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", maxsplit=1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])
    assert imported_roots == {"__future__", "dataclasses", "numpy"}

    forbidden_names = {"open", "exec", "eval", "compile", "__import__"}
    forbidden_attributes = {
        "open",
        "read_text",
        "read_bytes",
        "write_text",
        "write_bytes",
        "load",
        "save",
        "run",
        "Popen",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_names
        elif isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden_attributes


def test_zero_training_draft_and_literature_preserve_governance_boundaries() -> None:
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    draft = (
        project_root
        / "docs/aaai27/zero_training_cross_scale_counterfactual_tail_signal_draft_v0.md"
    ).read_text(encoding="utf-8")
    literature = (
        project_root
        / "docs/aaai27/literature-search-20260718-counterfactual-tail-preliminary/papers.md"
    ).read_text(encoding="utf-8")
    notes = (
        project_root
        / "docs/aaai27/literature-search-20260718-counterfactual-tail-preliminary/search-notes.md"
    ).read_text(encoding="utf-8")
    assert "UNREGISTERED DRAFT" in draft
    assert "DO NOT RUN ON SOURCE DATA BEFORE TIER2S" in draft
    assert "NUAA is forbidden" in draft
    assert "fresh seeds 46–50 remain untouched" in draft
    assert "No experimental result has been generated" in draft
    assert "AC-SLSIoU" in draft
    assert "global" in draft and "minimum-total-L1" in draft
    assert "bilinear sampling stencil" in draft
    assert "CHAL" in draft
    assert "PRELIMINARY_NOT_FROZEN" in literature
    assert "Loddis" in literature
    assert "not a novelty certificate" in literature
    assert "Unresolved items required before preregistration" in draft
    assert "HOLD and forbids" in draft
    assert "Conditional single-core V3 translation — NOT AUTHORIZED" in draft
    assert "supervision machinery, not separate contributions" in draft
    assert "this sketch is discarded" in draft
    assert "Preliminary deployable-V3 query expansion executed" in notes
    assert "CFKD" in notes and "DeFeat" in notes and "OED" in notes
    assert "Post-signal frozen search still required" in notes
    assert "exact final queries" in notes and "remain queued" in notes
    assert "authorized novelty claim" in notes


def test_counterfactual_terms_cannot_conflate_model_path_and_input_intervention() -> None:
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    draft = (
        project_root
        / "docs/aaai27/zero_training_cross_scale_counterfactual_tail_signal_draft_v0.md"
    ).read_text(encoding="utf-8")
    legacy = (
        project_root / "scripts/diagnose_component_fusion.py"
    ).read_text(encoding="utf-8")

    assert "`fusion_path_counterfactual`" in draft
    assert "`input_core_intervention`" in draft
    assert "model-path causal attribution" in draft
    assert "target-absent scene" in draft
    assert "distinct schema names" in draft
    assert "drop_component_action_fixed_full_gate" in legacy
    assert "context_zero_component_expert_off_reforward" in legacy


def test_opposite_ray_fill_rejects_bilinear_support_touching_core() -> None:
    from evaluation.counterfactual_tail_signal import (
        opposite_ray_annular_core_fill,
    )

    image = np.zeros((15, 15), dtype=np.float32)
    core = _disk_mask(15, 15, center_yx=(7.0, 7.0), radius=2.0)
    with pytest.raises(ValueError, match="bilinear support intersects"):
        opposite_ray_annular_core_fill(
            image,
            center_yx=(7.0, 7.0),
            core_mask=core,
            annulus_inner_radius=2.01,
            annulus_outer_radius=4.0,
            radial_samples=3,
            center_angular_samples=8,
        )


def test_zero_training_diagnostic_cannot_substitute_for_deployable_v3() -> None:
    from pathlib import Path

    draft = (
        Path(__file__).resolve().parents[1]
        / "docs/aaai27/zero_training_cross_scale_counterfactual_tail_signal_draft_v0.md"
    ).read_text(encoding="utf-8")
    assert "double forward is not a deployable V3" in draft
    assert "without test-time target" in draft
    assert "second full detector forward" in draft
    assert "median-latency ratio of at most" in draft
    assert "Counterfactual supervision or distillation alone" in draft



def test_rectangular_min_cost_assignment_matches_exhaustive_optimum() -> None:
    from itertools import permutations

    from evaluation.counterfactual_tail_signal import (
        _rectangular_minimum_cost_assignment,
    )

    generator = np.random.default_rng(731)
    for row_count, column_count in ((2, 3), (3, 4)):
        for _ in range(20):
            cost = generator.integers(
                0,
                8,
                size=(row_count, column_count),
            ).astype(np.float64)
            assignment = _rectangular_minimum_cost_assignment(cost)
            observed = float(cost[np.arange(row_count), assignment].sum())
            optimum = min(
                float(cost[np.arange(row_count), columns].sum())
                for columns in permutations(range(column_count), row_count)
            )
            assert observed == pytest.approx(optimum, rel=0.0, abs=0.0)
            assert len(set(assignment.tolist())) == row_count


def test_stratified_min_cost_match_improves_order_biased_greedy_case() -> None:
    from evaluation.counterfactual_tail_signal import (
        deterministic_stratified_greedy_match,
        deterministic_stratified_min_cost_match,
    )

    problem = {
        "target_ids": ["a", "b"],
        "clutter_ids": ["c1", "c2"],
        "target_strata": ["s", "s"],
        "clutter_strata": ["s", "s"],
        "target_features": [[5.0], [0.0]],
        "clutter_features": [[4.0], [100.0]],
    }
    greedy = deterministic_stratified_greedy_match(**problem)
    optimal = deterministic_stratified_min_cost_match(**problem)
    assert sum(match.l1_distance for match in greedy) == pytest.approx(101.0)
    assert sum(match.l1_distance for match in optimal) == pytest.approx(99.0)
    assert tuple((match.target_id, match.clutter_id) for match in optimal) == (
        ("a", "c2"),
        ("b", "c1"),
    )


def test_stratified_min_cost_match_is_permutation_invariant_and_no_reuse() -> None:
    from evaluation.counterfactual_tail_signal import (
        deterministic_stratified_min_cost_match,
    )

    first = deterministic_stratified_min_cost_match(
        target_ids=["a", "b"],
        clutter_ids=["c1", "c2", "c3", "c4"],
        target_strata=["s", "s"],
        clutter_strata=["s", "s", "s", "s"],
        target_features=[[0.0], [10.0]],
        clutter_features=[[1.0], [2.0], [9.0], [11.0]],
        clutter_per_target=2,
    )
    permuted = deterministic_stratified_min_cost_match(
        target_ids=["b", "a"],
        clutter_ids=["c4", "c2", "c1", "c3"],
        target_strata=["s", "s"],
        clutter_strata=["s", "s", "s", "s"],
        target_features=[[10.0], [0.0]],
        clutter_features=[[11.0], [2.0], [1.0], [9.0]],
        clutter_per_target=2,
    )
    first_pairs = tuple(
        (match.target_id, match.match_slot, match.clutter_id)
        for match in first
    )
    permuted_pairs = tuple(
        (match.target_id, match.match_slot, match.clutter_id)
        for match in permuted
    )
    assert first_pairs == permuted_pairs
    assert len({match.clutter_id for match in first}) == len(first)


def test_stratified_min_cost_match_fails_closed_on_shortage() -> None:
    from evaluation.counterfactual_tail_signal import (
        deterministic_stratified_min_cost_match,
    )

    with pytest.raises(ValueError, match="insufficient clutter coverage"):
        deterministic_stratified_min_cost_match(
            target_ids=["a", "b"],
            clutter_ids=["c1"],
            target_strata=["s", "s"],
            clutter_strata=["s"],
            target_features=[[0.0], [1.0]],
            clutter_features=[[0.0]],
        )



def test_intervention_generator_exposes_no_ground_truth_interface() -> None:
    import inspect

    from evaluation.counterfactual_tail_signal import (
        opposite_ray_annular_core_fill,
    )

    parameters = tuple(
        inspect.signature(opposite_ray_annular_core_fill).parameters
    )
    assert parameters == (
        "image",
        "center_yx",
        "core_mask",
        "annulus_inner_radius",
        "annulus_outer_radius",
        "radial_samples",
        "center_angular_samples",
    )
    assert all(
        token not in name.lower()
        for name in parameters
        for token in ("ground_truth", "label", "target")
    )
