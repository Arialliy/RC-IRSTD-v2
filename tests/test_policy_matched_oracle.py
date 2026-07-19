from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data_ext.mask_alignment import MASK_ALIGNMENT_POLICY
from evaluation import policy_matched_oracle as oracle
from evaluation.artifact_integrity import (
    SCORE_MANIFEST_SCHEMA_VERSION,
    SCORE_MASK_ALIGNMENT_SCHEMA,
    SCORE_RECORD_INTEGRITY_SCHEMA,
    file_sha256,
    ordered_ids_sha256,
    score_records_sha256,
    verify_score_map_directory,
)
from evaluation.threshold_sweep import (
    EMPTY_SET_THRESHOLD,
    build_default_thresholds,
    evaluation_threshold_grid_sha256,
    main as threshold_sweep_main,
    validate_formal_score_manifest,
)
from risk_curve.build_deployment_statistics import build_static_cross_fit_statistics


CANONICAL_GRID_SHA256 = (
    "4290e3ef607694da82fdf9022e6a65cb8472a60b6b743df336e6c0e2a33cc203"
)


def _write_formal_score_artifact(
    root: Path,
    probabilities: list[np.ndarray],
    masks: list[np.ndarray],
    *,
    split_role: str = "test",
    weight_digit: str = "a",
) -> tuple[Path, list[str]]:
    if len(probabilities) != len(masks) or not probabilities:
        raise ValueError("The synthetic score artifact must contain paired records")
    root.mkdir(parents=True)
    image_ids = [f"target-domain-sample-{index:03d}" for index in range(len(masks))]
    records: list[dict[str, object]] = []
    for index, (probability, mask) in enumerate(zip(probabilities, masks)):
        probability = np.asarray(probability, dtype=np.float32)
        mask = np.asarray(mask > 0, dtype=np.uint8)
        if probability.shape != mask.shape:
            raise ValueError("Synthetic probability/mask shapes must match")
        path = root / f"sample-{index:03d}.npz"
        np.savez_compressed(
            path,
            prob=probability,
            gray=np.zeros_like(probability, dtype=np.float32),
            mask=mask,
            image_id=np.asarray(image_ids[index]),
            labels_loaded=np.asarray(True),
            original_hw=np.asarray(probability.shape, dtype=np.int32),
            input_hw=np.asarray(probability.shape, dtype=np.int32),
            valid_hw=np.asarray(probability.shape, dtype=np.int32),
            padding_ltrb=np.asarray([0, 0, 0, 0], dtype=np.int32),
            spatial_mode=np.asarray("native"),
            mask_alignment_applied=np.asarray(False),
            mask_original_hw=np.asarray(mask.shape, dtype=np.int32),
            mask_aspect_relative_error=np.asarray(0.0),
            mask_alignment_policy=np.asarray(MASK_ALIGNMENT_POLICY),
        )
        records.append(
            {
                "image_id": image_ids[index],
                "file": path.name,
                "shape": list(probability.shape),
                "sha256": file_sha256(path),
                "mask_alignment_applied": False,
                "mask_original_hw": list(mask.shape),
                "mask_aspect_relative_error": 0.0,
            }
        )

    split_file = root / "frozen_split.txt"
    split_file.write_text("\n".join(image_ids) + "\n", encoding="utf-8")
    ids_hash = ordered_ids_sha256(image_ids)
    manifest = {
        "schema_version": SCORE_MANIFEST_SCHEMA_VERSION,
        "record_integrity_schema": SCORE_RECORD_INTEGRITY_SCHEMA,
        "score_type": "sigmoid_probability",
        "warm_flag": True,
        "labels_loaded": True,
        "num_images": len(records),
        "records": records,
        "records_sha256": score_records_sha256(records),
        "ordered_image_ids_sha256": ids_hash,
        "mask_alignment_schema": SCORE_MASK_ALIGNMENT_SCHEMA,
        "mask_alignment_policy": MASK_ALIGNMENT_POLICY,
        "mask_alignment_count": 0,
        "mask_aligned_sample_ids": [],
        "target_dataset": "target-domain",
        "source_datasets": ["source-domain"],
        "weight_sha256": weight_digit * 64,
        "checkpoint_selection_rule": "fixed_last",
        "checkpoint_diagnostic_only": False,
        "non_strict_state_loading": False,
        "model_backend": "canonical",
        "requested_split": split_role,
        "split_role": split_role,
        "split_authority_verified": True,
        "split_file": str(split_file.resolve()),
        "split_file_sha256": file_sha256(split_file),
        "split_ordered_ids_sha256": ids_hash,
        "spatial_mode": "native",
        "pad_multiple": 16,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return root, image_ids


def _simple_pairs(count: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    probabilities: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for index in range(count):
        probability = np.zeros((3, 3), dtype=np.float32)
        probability[1, 1] = np.float32(0.90 - 0.01 * index)
        probability[0, 0] = np.float32(0.60)
        mask = np.zeros((3, 3), dtype=np.uint8)
        mask[1, 1] = 1
        probabilities.append(probability)
        masks.append(mask)
    return probabilities, masks


def test_canonical_grid_has_exact_one_reject_endpoint_and_stable_hash() -> None:
    grid, source = oracle.load_policy_grid(score_contract={})
    assert np.array_equal(grid, build_default_thresholds())
    assert grid.size == 653
    assert grid[-2] == 1.0
    assert grid[-1] == EMPTY_SET_THRESHOLD
    assert evaluation_threshold_grid_sha256(grid) == CANONICAL_GRID_SHA256
    assert source == {"source_type": "built_in_canonical_653_grid"}


def test_static_units_are_seeded_nonoverlapping_and_exactly_cover_queries() -> None:
    image_ids = [f"image-{index}" for index in range(11)]
    first = oracle.build_decision_units(
        image_ids, "static", folds=5, seed=17, adaptation_window=4
    )
    repeated = oracle.build_decision_units(
        image_ids, "static", folds=5, seed=17, adaptation_window=4
    )
    changed_seed = oracle.build_decision_units(
        image_ids, "static", folds=5, seed=18, adaptation_window=4
    )

    assert first == repeated
    assert [unit.evaluation_indices for unit in first] != [
        unit.evaluation_indices for unit in changed_seed
    ]
    flattened = [index for unit in first for index in unit.evaluation_indices]
    assert sorted(flattened) == list(range(len(image_ids)))
    assert len(flattened) == len(set(flattened))
    for unit in first:
        assert set(unit.adaptation_indices).isdisjoint(unit.evaluation_indices)
        complement = set(range(len(image_ids))).difference(unit.evaluation_indices)
        assert len(unit.adaptation_indices) == 4
        assert set(unit.adaptation_indices).issubset(complement)
        assert unit.adaptation_complement_size == len(complement)
        assert unit.adaptation_sampling_rule == (
            "seedsequence(seed,fold_index)_without_replacement_from_complement"
        )
        assert unit.adaptation_sampling_seed_components == (17, unit.unit_index)


def test_static_default_support_matches_existing_cross_fit_builder(
    tmp_path: Path,
) -> None:
    probabilities, masks = _simple_pairs(40)
    score_dir, image_ids = _write_formal_score_artifact(
        tmp_path / "static-builder-scores", probabilities, masks
    )
    units = oracle.build_decision_units(image_ids, "static", folds=5, seed=23)
    existing = build_static_cross_fit_statistics(
        score_dir,
        np.asarray([0.0, 0.5], dtype=np.float32),
        folds=5,
        seed=23,
        adaptation_window=32,
        require_score_integrity=True,
    )
    existing_adaptation = [
        json.loads(str(value)) for value in existing["adaptation_ids"]
    ]
    existing_evaluation = [
        json.loads(str(value)) for value in existing["evaluation_ids"]
    ]

    assert [list(unit.adaptation_ids) for unit in units] == existing_adaptation
    assert [list(unit.evaluation_ids) for unit in units] == existing_evaluation
    assert all(len(unit.adaptation_indices) == 32 for unit in units)
    assert all(unit.adaptation_complement_size == 32 for unit in units)
    assert all(
        unit.adaptation_sampling_seed_components == (23, unit.unit_index)
        for unit in units
    )


def test_static_default_support_fails_closed_when_a_complement_is_too_small() -> None:
    image_ids = [f"image-{index}" for index in range(39)]
    with pytest.raises(
        ValueError, match="complement_size=31, adaptation_window=32"
    ):
        oracle.build_decision_units(image_ids, "static", folds=5, seed=42)


def test_causal_units_enforce_disjoint_a_e_roles_and_formal_stride() -> None:
    image_ids = [f"frame-{index:03d}" for index in range(70)]
    units = oracle.build_decision_units(image_ids, "causal")

    assert [unit.evaluation_indices for unit in units] == [(32,), (65,)]
    all_adaptation = {
        index for unit in units for index in unit.adaptation_indices
    }
    all_evaluation = {
        index for unit in units for index in unit.evaluation_indices
    }
    assert all_adaptation.isdisjoint(all_evaluation)
    assert all(
        max(unit.adaptation_indices) < min(unit.evaluation_indices) for unit in units
    )

    with pytest.raises(
        ValueError, match=r"stride=adaptation_window\+evaluation_window"
    ):
        oracle.build_decision_units(
            image_ids,
            "causal",
            adaptation_window=32,
            evaluation_window=1,
            stride=32,
        )


def test_threshold_selection_never_receives_adaptation_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probabilities, masks = _simple_pairs(6)
    image_ids = [f"sample-{index}" for index in range(6)]
    units = oracle.build_decision_units(
        image_ids, "static", folds=3, seed=9, adaptation_window=4
    )
    thresholds = np.asarray([0.0, 0.5, 0.8, EMPTY_SET_THRESHOLD])

    calls: list[tuple[list[int], list[int]]] = []
    original_sweep = oracle.sweep_thresholds

    def _record_query_only(
        query_probabilities: list[np.ndarray],
        query_masks: list[np.ndarray],
        query_thresholds: np.ndarray,
        **kwargs: object,
    ) -> list[dict[str, float | int]]:
        calls.append(
            (
                [id(value) for value in query_probabilities],
                [id(value) for value in query_masks],
            )
        )
        return original_sweep(
            query_probabilities, query_masks, query_thresholds, **kwargs
        )

    monkeypatch.setattr(oracle, "sweep_thresholds", _record_query_only)
    before = oracle.evaluate_policy_oracle(
        probabilities,
        masks,
        image_ids,
        units,
        thresholds,
        pixel_budget=0.2,
        component_budget=1_000_000.0,
    )
    assert len(calls) == len(units)
    for call, unit in zip(calls, units):
        assert call[0] == [id(probabilities[index]) for index in unit.evaluation_indices]
        assert call[1] == [id(masks[index]) for index in unit.evaluation_indices]

    protected = units[0]
    adaptation_index = protected.adaptation_indices[0]
    probabilities[adaptation_index] = np.ones((3, 3), dtype=np.float32)
    masks[adaptation_index] = np.zeros((3, 3), dtype=np.uint8)
    calls.clear()
    after = oracle.evaluate_policy_oracle(
        probabilities,
        masks,
        image_ids,
        units,
        thresholds,
        pixel_budget=0.2,
        component_budget=1_000_000.0,
    )
    before_unit = next(row for row in before["units"] if row["unit_id"] == protected.unit_id)
    after_unit = next(row for row in after["units"] if row["unit_id"] == protected.unit_id)
    assert before_unit == after_unit
    assert before_unit["labels_used_for_selection_ids"] == list(
        protected.evaluation_ids
    )
    assert before_unit["adaptation_labels_used_for_selection"] is False


def test_image_policy_distinguishes_exact_one_from_empty_set_endpoint() -> None:
    probabilities = [
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[0.9, 1.0]], dtype=np.float32),
    ]
    masks = [
        np.asarray([[1, 0]], dtype=np.uint8),
        np.asarray([[1, 0]], dtype=np.uint8),
    ]
    image_ids = ["clean-one", "false-one"]
    units = oracle.build_decision_units(image_ids, "image")
    result = oracle.evaluate_policy_oracle(
        probabilities,
        masks,
        image_ids,
        units,
        np.asarray([1.0, EMPTY_SET_THRESHOLD]),
        pixel_budget=0.1,
        component_budget=1.0,
    )
    assert [row["threshold"] for row in result["units"]] == [
        1.0,
        EMPTY_SET_THRESHOLD,
    ]
    assert [row["pd"] for row in result["units"]] == [1.0, 0.0]


def test_vectorized_pixel_pruning_is_exactly_equivalent_to_full_grid_selection() -> None:
    probabilities = [
        np.asarray(
            [
                [0.75, 0.10, 0.20, 0.30],
                [0.10, 0.90, 0.40, 0.20],
                [0.10, 0.20, 0.30, 0.10],
                [0.20, 0.10, 0.20, 0.10],
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [
                [0.65, 0.10, 0.20, 0.10],
                [0.10, 0.70, 0.30, 0.20],
                [0.10, 0.10, 0.20, 0.10],
                [0.20, 0.10, 0.10, 0.20],
            ],
            dtype=np.float32,
        ),
    ]
    masks = [np.zeros((4, 4), dtype=np.uint8) for _ in probabilities]
    for mask in masks:
        mask[1, 1] = 1
    image_ids = ["first", "second"]
    grid = np.asarray(
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, EMPTY_SET_THRESHOLD],
        dtype=np.float64,
    )
    pixel_budget = 0.05
    component_budget = 1_000_000.0
    units = oracle.build_decision_units(image_ids, "global")

    pruned = oracle.evaluate_policy_oracle(
        probabilities,
        masks,
        image_ids,
        units,
        grid,
        pixel_budget=pixel_budget,
        component_budget=component_budget,
        min_component_area=1,
    )
    full_rows = oracle.sweep_thresholds(
        probabilities, masks, grid, min_component_area=1
    )
    full_selected = oracle.select_operating_point(
        full_rows,
        pixel_budget=pixel_budget,
        component_budget=component_budget,
        strategy="max_pd",
    )
    assert full_selected is not None
    selected = pruned["units"][0]
    for field in (
        "threshold",
        "pd",
        "fa_pixel",
        "fa_component_mp",
        "tp_objects",
        "gt_objects",
        "fp_components",
        "fp_pixels",
        "total_pixels",
    ):
        assert selected[field] == full_selected[field]

    audit = selected["threshold_pruning"]
    assert audit["enabled"] is True
    assert audit["lossless"] is True
    assert audit["original_threshold_grid_size"] == len(grid)
    assert audit["original_threshold_grid_sha256"] == (
        evaluation_threshold_grid_sha256(grid)
    )
    assert 0 < audit["evaluated_threshold_count"] < len(grid)
    assert audit["pruned_threshold_count"] == (
        len(grid) - audit["evaluated_threshold_count"]
    )
    assert audit["runtime_exact_fp_pixel_cross_check"] is True
    assert "joint pixel and component budgets" in audit["proof"]
    assert pruned["threshold_pruning"]["all_units_lossless"] is True
    assert pruned["threshold_pruning"]["total_pruned_unit_thresholds"] > 0
    assert "cannot be jointly budget-feasible" in pruned["threshold_pruning"]["proof"]


def test_cli_requires_oracle_ack_and_emits_hash_bound_per_unit_audit(
    tmp_path: Path,
) -> None:
    probabilities, masks = _simple_pairs(5)
    score_dir, image_ids = _write_formal_score_artifact(
        tmp_path / "scores", probabilities, masks
    )
    output = tmp_path / "policy-oracle.json"
    args = [
        "--score-dir",
        str(score_dir),
        "--policy",
        "static",
        "--pixel-budget",
        "0.2",
        "--component-budget",
        "1000000",
        "--adaptation-window",
        "3",
        "--output",
        str(output),
    ]
    with pytest.raises(ValueError, match="--oracle-diagnostic"):
        oracle.main(args)
    assert oracle.main([*args, "--oracle-diagnostic"]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["oracle_only"] is True
    assert payload["formal_protocol_eligible"] is False
    assert payload["policy"]["folds"] == 5
    assert payload["policy"]["adaptation_window"] == 3
    assert payload["policy"]["selected_adaptation_sizes"] == [3] * 5
    assert payload["policy"]["complement_sizes"] == [4] * 5
    assert payload["policy"]["adaptation_sampling_rule"] == (
        "seedsequence(seed,fold_index)_without_replacement_from_complement"
    )
    assert payload["policy"]["evidence_role"] == (
        "primary_policy_matched_oracle_evidence"
    )
    assert payload["policy"]["full_evaluation_coverage"] is True
    assert payload["policy"]["num_decision_units"] == 5
    assert payload["policy"]["query_folds_pairwise_disjoint"] is True
    assert payload["policy"]["exactly_once_query_coverage"] is True
    assert payload["policy"]["query_fold_sizes"] == [1, 1, 1, 1, 1]
    assert payload["coverage"]["num_evaluated_images"] == 5
    assert payload["coverage"]["evaluated_fraction"] == 1.0
    assert payload["coverage"]["aggregate_pd_is_full_score_artifact_pd"] is True
    assert payload["partition_audit"]["within_unit_role_disjoint"] is True
    assert sorted(
        image_id for unit in payload["units"] for image_id in unit["evaluation_ids"]
    ) == sorted(image_ids)
    assert all(
        unit["labels_used_for_selection_ids"] == unit["evaluation_ids"]
        and unit["adaptation_labels_used_for_selection"] is False
        and unit["num_adaptation_images"] == 3
        and unit["num_evaluation_images"] == 1
        and unit["adaptation_complement_size"] == 4
        and "without_replacement" in unit["adaptation_sampling_rule"]
        and unit["adaptation_sampling_seed_components"]
        == [42, unit["unit_index"]]
        and not unit["adaptation_evaluation_role_overlap_ids"]
        and unit["budget_feasibility"]["joint_budget_satisfied"] is True
        for unit in payload["units"]
    )
    required_selected_fields = {
        "threshold",
        "pd",
        "fa_pixel",
        "fa_component_mp",
        "tp_objects",
        "gt_objects",
        "fp_components",
        "fp_pixels",
        "total_pixels",
        "budget_feasibility",
    }
    assert all(required_selected_fields.issubset(unit) for unit in payload["units"])
    assert {
        "tp_objects",
        "gt_objects",
        "fp_components",
        "fp_pixels",
        "total_pixels",
        "pd",
        "fa_pixel",
        "fa_component_mp",
        "global_aggregate_budget_satisfied",
    }.issubset(payload["aggregate"])
    assert payload["all_units_individually_budget_satisfied"] is True
    assert (
        payload["budget_semantics"]["global_aggregate_role"]
        == "post_selection_audit_not_a_joint_selection_constraint"
    )
    assert payload["threshold_grid"]["size"] == 653
    assert payload["threshold_grid"]["sha256"] == CANONICAL_GRID_SHA256
    assert payload["threshold_grid"]["contains_exact_one"] is True
    assert payload["threshold_pruning"]["original_threshold_grid_size"] == 653
    assert payload["threshold_pruning"]["original_threshold_grid_sha256"] == (
        CANONICAL_GRID_SHA256
    )
    assert payload["threshold_pruning"]["all_units_lossless"] is True
    assert payload["threshold_pruning"][
        "all_units_vectorized_pixel_pruning_enabled"
    ] is True
    assert all(
        0 < count <= 653
        for count in payload["threshold_pruning"][
            "evaluated_threshold_counts_by_unit"
        ]
    )
    assert payload["provenance"]["score_manifest_sha256"] == file_sha256(
        score_dir / "manifest.json"
    )
    assert payload["provenance"]["detector_weight_sha256"] == "a" * 64
    assert not list(tmp_path.rglob("*.tmp"))


def test_causal_and_image_outputs_have_explicit_evidence_and_coverage_scope(
    tmp_path: Path,
) -> None:
    causal_probabilities, causal_masks = _simple_pairs(34)
    causal_score_dir, _ = _write_formal_score_artifact(
        tmp_path / "causal-scores", causal_probabilities, causal_masks
    )
    causal = oracle.run_policy_matched_oracle(
        score_dir=causal_score_dir,
        policy="causal",
        pixel_budget=0.2,
        component_budget=1_000_000.0,
    )
    assert causal["policy"]["evidence_role"] == (
        "causal_policy_matched_oracle_sensitivity_evidence"
    )
    assert causal["policy"]["adaptation_window"] == 32
    assert causal["policy"]["evaluation_window"] == 1
    assert causal["policy"]["stride"] == 33
    assert causal["policy"]["num_complete_causal_blocks"] == 1
    assert causal["policy"]["global_adaptation_evaluation_role_overlap"] == []
    assert causal["coverage"]["num_evaluated_images"] == 1
    assert causal["coverage"]["num_score_images"] == 34
    assert causal["coverage"]["evaluated_fraction"] == pytest.approx(1.0 / 34.0)
    assert causal["coverage"]["full_evaluation_coverage"] is False
    assert causal["coverage"]["aggregate_pd_is_full_score_artifact_pd"] is False
    assert causal["coverage"]["aggregate_metric_scope"] == (
        "raw_count_micro_aggregate_over_complete_causal_evaluation_windows_only"
    )
    assert causal["partition_audit"]["within_unit_role_disjoint"] is True
    assert causal["units"][0]["adaptation_indices"] == list(range(32))
    assert causal["units"][0]["evaluation_indices"] == [32]

    image_probabilities, image_masks = _simple_pairs(2)
    image_score_dir, _ = _write_formal_score_artifact(
        tmp_path / "image-scores", image_probabilities, image_masks, weight_digit="b"
    )
    image = oracle.run_policy_matched_oracle(
        score_dir=image_score_dir,
        policy="image",
        pixel_budget=0.2,
        component_budget=1_000_000.0,
    )
    assert image["policy"]["evidence_role"] == (
        "extremely_permissive_per_image_oracle_upper_bound"
    )
    assert image["policy"]["budget_enforcement_unit"] == "each_individual_image"
    assert image["coverage"]["evaluated_fraction"] == 1.0
    assert all(unit["num_evaluation_images"] == 1 for unit in image["units"])


def test_manifest_bound_curve_rejects_post_sidecar_tampering(tmp_path: Path) -> None:
    probabilities, masks = _simple_pairs(2)
    score_dir, _ = _write_formal_score_artifact(
        tmp_path / "scores", probabilities, masks
    )
    curve = tmp_path / "curve.csv"
    assert threshold_sweep_main(
        [
            "--score-dir",
            str(score_dir),
            "--output",
            str(curve),
            "--formal",
            "--expected-split-role",
            "test",
        ]
    ) == 0
    manifest, _, integrity = verify_score_map_directory(
        score_dir, require_integrity=True, require_masks=True
    )
    contract = validate_formal_score_manifest(
        manifest, integrity, expected_split_role="test"
    )
    grid, source = oracle.load_policy_grid(
        score_contract=contract, curve_path=curve
    )
    assert evaluation_threshold_grid_sha256(grid) == CANONICAL_GRID_SHA256
    assert source["curve_sha256"] == file_sha256(curve)

    curve.write_text(curve.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Curve SHA-256"):
        oracle.load_policy_grid(score_contract=contract, curve_path=curve)
