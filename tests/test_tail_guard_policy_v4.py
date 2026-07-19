from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from risk_curve.count_all_anchor import (
    COUNT_ALL_ANCHOR_SCHEMA_VERSION,
    derive_anchor_rates,
    validate_count_all_anchor_archive,
)
from risk_curve.curve_dataset import COUNT_ALL_ADAPTATION_SCHEMA_VERSION
from risk_curve.representation import (
    LOGIT_REPRESENTATION,
    logit_threshold_grid_sha256,
)
from risk_curve.tail_guard_policy_v4 import (
    CANONICAL_GRID_SHA256,
    CANONICAL_TAIL_INDEX,
    CANONICAL_TAIL_LOGIT,
    LOOSE_NONZERO_TAIL_FACTOR,
    LOOSE_ZERO_TAIL_FACTOR,
    STRICT_FACTOR,
    TailGuardContract,
    policy_contract_record,
    select_tail_guard_action,
)
import risk_curve.evaluate_tail_guard_v4 as tail_evaluator


GRID = np.asarray([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
GRID_HASH = logit_threshold_grid_sha256(GRID)
TEST_CONTRACT = TailGuardContract(
    grid_sha256=GRID_HASH,
    tail_index=2,
    tail_logit=float(GRID[2]),
)


def _provenance(*, masks_read: bool = False) -> dict[str, object]:
    return {
        "representation": LOGIT_REPRESENTATION,
        "connectivity": 2,
        "min_component_area": 1,
        "count_all_adaptation_schema_version": COUNT_ALL_ADAPTATION_SCHEMA_VERSION,
        "count_all_adaptation_sample_role": "adaptation_window_A_label_free",
        "count_all_adaptation_masks_read": masks_read,
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


def _archive() -> dict[str, np.ndarray]:
    pixel = np.asarray(
        [[100, 20, 0, 0], [100, 20, 10, 0]], dtype=np.int64
    )
    component_raw = np.asarray(
        [[10, 5, 0, 0], [10, 5, 1, 0]], dtype=np.int64
    )
    component_upper = np.maximum.accumulate(
        component_raw[:, ::-1], axis=1
    )[:, ::-1]
    return {
        "statistics": np.zeros((2, 1), dtype=np.float32),
        "thresholds": GRID.copy(),
        "representation": np.asarray(LOGIT_REPRESENTATION),
        "threshold_grid_sha256": np.asarray(GRID_HASH),
        "adaptation_predicted_pixel_counts": pixel,
        "adaptation_predicted_component_counts_raw": component_raw,
        "adaptation_predicted_component_counts_upper": component_upper,
        "adaptation_total_pixels": np.full(2, 1_000_000, dtype=np.int64),
        "count_all_adaptation_schema_version": np.asarray(
            COUNT_ALL_ADAPTATION_SCHEMA_VERSION
        ),
        "provenance_json": np.asarray(json.dumps(_provenance(), sort_keys=True)),
    }


def test_canonical_policy_constants_are_frozen() -> None:
    assert CANONICAL_GRID_SHA256 == (
        "1675cae81dace9ee92f1cf23fe5742f2d5ba0ce7ecea6e00c9f9648fa1825ced"
    )
    assert CANONICAL_TAIL_INDEX == 1750
    assert np.float32(CANONICAL_TAIL_LOGIT).tobytes() == np.float32(
        17.8231925964
    ).tobytes()
    record = policy_contract_record()
    assert record["loose_zero_tail_factor"] == 0.35
    assert record["loose_nonzero_tail_factor"] == 1.0
    assert record["strict_factor"] == 3.0
    assert record["adaptation_masks_read"] is False
    assert record["post_hoc_source_pseudo_target_development_selection"] is True
    assert record["confirmatory_gate_c_eligible"] is False


def test_anchor_rates_are_derived_from_integer_counts_and_exposure() -> None:
    batch = validate_count_all_anchor_archive(_archive())
    assert batch.semantic_sha256 != GRID_HASH
    assert batch.semantic_sha256 == batch.episode(0).archive_semantic_sha256
    assert COUNT_ALL_ANCHOR_SCHEMA_VERSION == "rc-v4-count-all-anchor-v1"
    pixel, component = derive_anchor_rates(batch.episode(0))
    np.testing.assert_allclose(pixel, [1e-4, 2e-5, 0.0, 0.0])
    np.testing.assert_allclose(component, [10.0, 5.0, 0.0, 0.0])
    assert not pixel.flags.writeable
    assert not component.flags.writeable


def test_anchor_requires_integer_dtype_and_all_fields() -> None:
    floating = _archive()
    floating["adaptation_predicted_pixel_counts"] = floating[
        "adaptation_predicted_pixel_counts"
    ].astype(np.float32)
    with pytest.raises(ValueError, match="integer dtype"):
        validate_count_all_anchor_archive(floating)

    missing = _archive()
    del missing["adaptation_total_pixels"]
    with pytest.raises(ValueError, match="missing"):
        validate_count_all_anchor_archive(missing)


def test_anchor_fails_closed_on_envelope_or_mask_tamper() -> None:
    envelope = _archive()
    envelope["adaptation_predicted_component_counts_upper"] = envelope[
        "adaptation_predicted_component_counts_upper"
    ].copy()
    envelope["adaptation_predicted_component_counts_upper"][0, 1] += 1
    with pytest.raises(ValueError, match="suffix_max|tampered"):
        validate_count_all_anchor_archive(envelope)

    masks = _archive()
    masks["provenance_json"] = np.asarray(
        json.dumps(_provenance(masks_read=True), sort_keys=True)
    )
    with pytest.raises(ValueError, match="masks_read"):
        validate_count_all_anchor_archive(masks)


def test_loose_tail_zero_and_nonzero_factors_are_applied_to_rates() -> None:
    batch = validate_count_all_anchor_archive(
        _archive(), expected_grid_sha256=GRID_HASH
    )
    zero = select_tail_guard_action(
        batch.episode(0),
        pixel_budget=1e-5,
        component_budget=5.0,
        contract=TEST_CONTRACT,
    )
    assert zero["tail_both_counts_zero"] is True
    assert zero["factor"] == LOOSE_ZERO_TAIL_FACTOR
    assert zero["threshold_index"] == 1
    assert zero["guarded_pixel_risk_at_action"] == pytest.approx(7e-6)
    assert zero["guarded_component_risk_at_action"] == pytest.approx(1.75)
    assert zero["future_e_counts_used_for_selection"] is False

    nonzero = select_tail_guard_action(
        batch.episode(1),
        pixel_budget=1e-5,
        component_budget=5.0,
        contract=TEST_CONTRACT,
    )
    assert nonzero["tail_both_counts_zero"] is False
    assert nonzero["factor"] == LOOSE_NONZERO_TAIL_FACTOR
    assert nonzero["threshold_index"] == 2


def test_strict_factor_and_registered_budget_fail_close() -> None:
    batch = validate_count_all_anchor_archive(_archive())
    strict = select_tail_guard_action(
        batch.episode(1),
        pixel_budget=1e-6,
        component_budget=1.0,
        contract=TEST_CONTRACT,
    )
    assert strict["factor"] == STRICT_FACTOR
    assert strict["threshold_index"] == 3
    with pytest.raises(ValueError, match="two registered budget"):
        select_tail_guard_action(
            batch.episode(1),
            pixel_budget=2e-6,
            component_budget=1.0,
            contract=TEST_CONTRACT,
        )


def test_policy_rejects_grid_and_count_tampering() -> None:
    anchor = validate_count_all_anchor_archive(_archive()).episode(0)
    with pytest.raises(ValueError, match="grid/hash"):
        select_tail_guard_action(
            replace(anchor, threshold_grid_sha256="0" * 64),
            pixel_budget=1e-5,
            component_budget=5.0,
            contract=TEST_CONTRACT,
        )

    bad_upper = np.asarray(anchor.component_counts_upper).copy()
    bad_upper[1] += 1
    with pytest.raises(ValueError, match="envelope"):
        select_tail_guard_action(
            replace(anchor, component_counts_upper=bad_upper),
            pixel_budget=1e-5,
            component_budget=5.0,
            contract=TEST_CONTRACT,
        )


def test_policy_rejects_noncanonical_tail_binding() -> None:
    anchor = validate_count_all_anchor_archive(_archive()).episode(0)
    wrong_tail = TailGuardContract(
        grid_sha256=GRID_HASH,
        tail_index=2,
        tail_logit=1.5,
    )
    with pytest.raises(ValueError, match="tail logit"):
        select_tail_guard_action(
            anchor,
            pixel_budget=1e-5,
            component_budget=5.0,
            contract=wrong_tail,
        )


def test_evaluator_freezes_all_A_actions_before_E_audit_and_matches_metric_shape(
    tmp_path, monkeypatch
) -> None:
    archive = _archive()
    archive.update(
        {
            "threshold_grid_manifest_sha256": np.asarray("a" * 64),
            "feature_schema_sha256": np.asarray("b" * 64),
            "threshold_grid_detector_protocol": np.asarray("all_source_detector_folds"),
            "threshold_grid_detector_checkpoint_sha256s": np.asarray(["1" * 64]),
            "threshold_grid_outer_detector_checkpoint_sha256": np.asarray("1" * 64),
            "threshold_grid_episode_detector_checkpoint_sha256s": np.asarray(["2" * 64]),
            "adaptation_ids": np.asarray(
                [json.dumps(["a0"]), json.dumps(["a1"])]
            ),
            "evaluation_ids": np.asarray(
                [json.dumps(["e0"]), json.dumps(["e1"])]
            ),
        }
    )
    episode_path = tmp_path / "episode.npz"
    episode_path.write_bytes(b"fixture-bound-by-monkeypatch")
    output = tmp_path / "tail.json"
    selection_calls: list[tuple[int, int]] = []
    original_select = tail_evaluator.select_tail_guard_action

    def counted_select(anchor, **kwargs):
        selection_calls.append((anchor.row, len(selection_calls)))
        return original_select(anchor, **kwargs)

    def future_e_after_selection(_archive, *, split):
        assert split == "validation"
        assert len(selection_calls) == 4
        return {
            "thresholds": GRID,
            "pixel_fp_counts": np.asarray([[10, 5, 0, 0], [10, 5, 1, 0]]),
            "component_fp_counts": np.asarray([[2, 1, 0, 0], [2, 1, 1, 0]]),
            "tp_object_counts": np.asarray([[2, 2, 1, 0], [2, 2, 1, 0]]),
            "gt_object_counts": np.asarray([2, 2]),
            "total_pixels": np.asarray([1_000_000, 1_000_000]),
            "pseudo_targets": np.asarray(["source-a", "source-a"]),
        }

    monkeypatch.setattr(tail_evaluator, "load_curve_archive", lambda _path: archive)
    monkeypatch.setattr(
        tail_evaluator, "_load_selection_archive", lambda _path: archive
    )
    monkeypatch.setattr(tail_evaluator, "select_tail_guard_action", counted_select)
    monkeypatch.setattr(tail_evaluator, "_validate_count_archive", future_e_after_selection)
    monkeypatch.setattr(tail_evaluator, "_sha256_file", lambda _path: "f" * 64)
    tail_evaluator.evaluate_tail_guard(
        episode_file=episode_path,
        output=output,
        contract=TEST_CONTRACT,
    )
    payload = json.loads(output.read_text())
    assert payload["selection_finalized_before_future_e_audit"] is True
    assert payload["labels_used_for_action_selection"] is False
    assert payload["outer_target_labels_used"] is False
    assert payload["adaptation_masks_read"] is False
    assert payload["monotonic_violation_rates"]["tail_guard"] == 0.0
    assert payload["monotonic_violation_counts"]["tail_guard"] == 0
    assert payload["post_hoc_source_pseudo_target_development_selection"] is True
    assert payload["confirmatory_gate_c_eligible"] is False
    assert payload["status"] == "DEVELOPMENT_COMPLETE_NOT_CONFIRMATORY"
    assert len(payload["budgets"]) == 2
    for item in payload["budgets"]:
        metrics = item["methods"]["tail_guard"]
        for field in (
            "pd",
            "pixel_risk",
            "component_risk",
            "joint_violation_rate",
            "mean_relative_excess",
            "max_relative_excess",
            "reject_rate",
        ):
            assert field in metrics


def test_evaluator_fails_closed_if_archive_changes_between_selection_and_audit(
    tmp_path, monkeypatch
) -> None:
    episode_path = tmp_path / "episode.npz"
    episode_path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        tail_evaluator, "_load_selection_archive", lambda _path: _archive()
    )
    hashes = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(tail_evaluator, "_sha256_file", lambda _path: next(hashes))
    with pytest.raises(ValueError, match="changed during A-only selection"):
        tail_evaluator.evaluate_tail_guard(
            episode_file=episode_path,
            output=tmp_path / "unused.json",
            contract=TEST_CONTRACT,
        )
