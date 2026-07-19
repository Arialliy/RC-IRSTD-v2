from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import risk_curve.aggregate_gate_c_v4 as gate_c_module
import risk_curve.semantic_preflight_gate_c_v4 as preflight_module

from risk_curve.aggregate_gate_c_v4 import (
    AGGREGATE_GATE_C_SCHEMA_VERSION,
    GATE_C_INPUT_SEAL_SCHEMA_VERSION,
    GATE_C_SEMANTIC_PREFLIGHT_SCHEMA_VERSION,
    REQUIRED_METHODS,
    SEMANTIC_PREFLIGHT_TOOL_VERSION,
    SEMANTIC_PREFLIGHT_VALIDATOR_FILES,
    _aggregate_actions,
    _relation,
    aggregate_gate_c,
)
from risk_curve.evaluate_gate_c_baselines_v4 import GATE_C_BASELINES_SCHEMA_VERSION
from risk_curve.evaluate_source_pseudo_target_v4 import (
    SOURCE_PSEUDO_TARGET_COMPARISON_SCHEMA_VERSION,
)
from risk_curve.representation import LOGIT_GRID_SCHEMA_VERSION, LOGIT_REPRESENTATION
from risk_curve.semantic_preflight_gate_c_v4 import (
    build_gate_c_semantic_preflight,
)


BUDGETS = ((1e-5, 5.0), (1e-6, 1.0))
GRID_HASH = "a" * 64
MANIFEST_HASH = "b" * 64
FEATURE_HASH = "c" * 64
OUTER_DETECTOR_HASH = "d" * 64
INNER_HASHES = ("e" * 64, "f" * 64)
ALL_DETECTOR_HASHES = (OUTER_DETECTOR_HASH, *INNER_HASHES)


def _file(path: Path, content: str) -> tuple[str, str]:
    path.write_text(content, encoding="utf-8")
    return str(path.resolve()), hashlib.sha256(path.read_bytes()).hexdigest()


def _action(
    *,
    index: int,
    pixel_fp: int,
    component_fp: int,
    tp: int,
    gt: int,
    pixel_budget: float,
    component_budget: float,
) -> dict[str, object]:
    total_pixels = 1_000_000
    pixel_risk = pixel_fp / total_pixels
    component_risk = float(component_fp)
    pd = tp / max(gt, 1)
    pixel_excess = max(pixel_risk / pixel_budget - 1.0, 0.0)
    component_excess = max(component_risk / component_budget - 1.0, 0.0)
    return {
        "threshold_index": index,
        "selected_logit_threshold": float(index),
        "reject": False,
        "pixel_fp_count": pixel_fp,
        "component_fp_count": component_fp,
        "tp_object_count": tp,
        "gt_object_count": gt,
        "total_pixels": total_pixels,
        "pd": pd,
        "pixel_risk": pixel_risk,
        "component_risk": component_risk,
        "pixel_budget_violated": pixel_risk > pixel_budget,
        "component_budget_violated": component_risk > component_budget,
        "joint_budget_violated": (
            pixel_risk > pixel_budget or component_risk > component_budget
        ),
        "pixel_relative_excess": pixel_excess,
        "component_relative_excess": component_excess,
        "joint_relative_excess": max(pixel_excess, component_excess),
    }


def _shared_contract() -> dict[str, object]:
    return {
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_size": 4,
        "threshold_grid_sha256": GRID_HASH,
        "threshold_grid_manifest_sha256": MANIFEST_HASH,
        "feature_schema_sha256": FEATURE_HASH,
        "threshold_grid_detector_protocol": "all_source_only_detector_folds",
        "threshold_grid_detector_checkpoint_sha256s": list(ALL_DETECTOR_HASHES),
        "threshold_grid_outer_detector_checkpoint_sha256": OUTER_DETECTOR_HASH,
        "threshold_grid_episode_detector_checkpoint_sha256s": list(INNER_HASHES),
    }


def _method_actions(
    *, scenario: str, budget_position: int, fold_id: str
) -> dict[str, list[dict[str, object]]]:
    pixel_budget, component_budget = BUDGETS[budget_position]
    index = budget_position + 1
    if scenario == "controlled":
        risk_counts = [(0, 0, 5), (0, 0, 5)]
        direct_counts = (
            [(20, 10, 4), (0, 0, 4)]
            if budget_position == 0
            else [(2, 2, 4), (0, 0, 4)]
        )
    elif scenario == "pd_only":
        risk_counts = (
            [(30, 15, 5), (30, 15, 5)]
            if budget_position == 0
            else [(3, 3, 5), (3, 3, 5)]
        )
        direct_counts = [(0, 0, 2), (0, 0, 2)]
    elif scenario == "unsafe_direct_high_pd":
        risk_counts = [(0, 0, 2), (0, 0, 2)]
        direct_counts = (
            [(30, 15, 5), (30, 15, 5)]
            if budget_position == 0
            else [(3, 3, 5), (3, 3, 5)]
        )
    elif scenario == "no_feasible_comparator":
        risk_counts = [(0, 0, 2), (0, 0, 2)]
        direct_counts = (
            [(30, 15, 5), (30, 15, 5)]
            if budget_position == 0
            else [(3, 3, 5), (3, 3, 5)]
        )
    elif scenario == "single_cell_feasibility":
        if budget_position == 0 and fold_id == "val_irstd":
            # Both methods violate the same one episode, so violation reduction
            # is zero.  The 21 -> 20 pixel change only moves aggregate RC-Direct
            # from just above to exactly on budget; excess reduction is <20%.
            risk_counts = [(20, 0, 2), (0, 0, 2)]
            direct_counts = [(21, 0, 2), (0, 0, 2)]
        elif budget_position == 0:
            risk_counts = [(0, 0, 2), (0, 0, 2)]
            direct_counts = [(0, 0, 2), (0, 0, 2)]
        else:
            risk_counts = [(0, 0, 3), (0, 0, 3)]
            direct_counts = [(0, 0, 2), (0, 0, 2)]
    else:  # pragma: no cover - fixture guard
        raise AssertionError(scenario)

    def make(rows: list[tuple[int, int, int]]) -> list[dict[str, object]]:
        return [
            _action(
                index=index,
                pixel_fp=pixel_fp,
                component_fp=component_fp,
                tp=tp,
                gt=5,
                pixel_budget=pixel_budget,
                component_budget=component_budget,
            )
            for pixel_fp, component_fp, tp in rows
        ]

    risk = make(risk_counts)
    direct = make(direct_counts)
    if scenario == "no_feasible_comparator":
        count_all = direct
        source_static = direct
        source_worst = direct
    elif scenario == "single_cell_feasibility":
        zero_pd = make([(0, 0, 0), (0, 0, 0)])
        count_all = risk
        source_static = zero_pd
        source_worst = zero_pd
    else:
        count_all = risk
        source_static = direct
        source_worst = direct
    return {
        "risk_curve": risk,
        "rc_direct": direct,
        "source_static": source_static,
        "source_worst": source_worst,
        "count_all": count_all,
    }


def _write_fold(
    root: Path,
    *,
    fold_id: str,
    validation_target: str,
    train_target: str,
    scenario: str,
) -> tuple[Path, Path]:
    fold_root = root / fold_id
    fold_root.mkdir(parents=True)
    train_file, train_hash = _file(fold_root / "train.npz", f"{fold_id}-train")
    validation_file, validation_hash = _file(
        fold_root / "validation.npz", f"{fold_id}-validation"
    )
    risk_file, risk_hash = _file(fold_root / "risk.pt", f"{fold_id}-risk")
    direct_file, direct_hash = _file(fold_root / "direct.pt", f"{fold_id}-direct")
    identities = [
        {
            "episode_index": index,
            "pseudo_target": validation_target,
            "adaptation_ids": [f"{fold_id}-adapt-{index}"],
            "evaluation_ids": [f"{fold_id}-eval-{index}"],
        }
        for index in range(2)
    ]
    all_actions = {
        position: _method_actions(
            scenario=scenario,
            budget_position=position,
            fold_id=fold_id,
        )
        for position in range(len(BUDGETS))
    }
    comparison_episodes: list[dict[str, object]] = []
    for row, identity in enumerate(identities):
        budget_actions = []
        for position, (pixel_budget, component_budget) in enumerate(BUDGETS):
            budget_actions.append(
                {
                    "budget_position": position,
                    "pixel_budget": pixel_budget,
                    "component_budget": component_budget,
                    "methods": {
                        method: all_actions[position][method][row]
                        for method in ("risk_curve", "rc_direct")
                    },
                }
            )
        comparison_episodes.append({**identity, "actions": budget_actions})
    comparison_budgets = []
    for position, (pixel_budget, component_budget) in enumerate(BUDGETS):
        comparison_budgets.append(
            {
                "budget_position": position,
                "pixel_budget": pixel_budget,
                "component_budget": component_budget,
                "methods": {
                    method: _aggregate_actions(all_actions[position][method])
                    for method in ("risk_curve", "rc_direct")
                },
            }
        )
    comparison = {
        "schema_version": SOURCE_PSEUDO_TARGET_COMPARISON_SCHEMA_VERSION,
        "protocol": "source_only_pseudo_target_fair_comparison",
        **_shared_contract(),
        "labels_used_for_action_selection": False,
        "source_pseudo_target_labels_used_for_post_selection_evaluation": True,
        "outer_target_labels_used": False,
        "formal_source_domains": ["IRSTD-1K", "NUDT-SIRST"],
        "excluded_outer_target": "NUAA-SIRST",
        "validation_pseudo_target": validation_target,
        "archive_split": "validation",
        "episode_archive": validation_file,
        "episode_archive_sha256": validation_hash,
        "train_archive_sha256": train_hash,
        "validation_archive_sha256": validation_hash,
        "risk_curve_checkpoint": risk_file,
        "risk_curve_checkpoint_sha256": risk_hash,
        "rc_direct_checkpoint": direct_file,
        "rc_direct_checkpoint_sha256": direct_hash,
        "seed": 42,
        "num_episodes": len(identities),
        "pseudo_targets": [validation_target],
        "budgets": comparison_budgets,
        "per_episode": comparison_episodes,
        "monotonic_violation_rates": {"risk_curve": 0.0, "rc_direct": 0.0},
        # Deliberately GO: the aggregate owns the actual decision.
        "gate": {"decision": "GO"},
    }
    comparison_path = fold_root / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, sort_keys=True), encoding="utf-8")

    baseline_budgets = []
    for position, (pixel_budget, component_budget) in enumerate(BUDGETS):
        methods: dict[str, object] = {}
        for method in ("source_static", "source_worst", "count_all"):
            per_episode = []
            for row, identity in enumerate(identities):
                record: dict[str, object] = {
                    **identity,
                    "action": all_actions[position][method][row],
                }
                if method == "count_all":
                    record["selection"] = {
                        "found": True,
                        "reject": False,
                        "threshold_index": position + 1,
                    }
                per_episode.append(record)
            evaluation: dict[str, object] = {
                "per_episode": per_episode,
                "aggregate": _aggregate_actions(all_actions[position][method]),
                "validation_labels_used_for_selection": False,
            }
            if method == "count_all":
                evaluation.update(
                    {
                        "adaptation_masks_read": False,
                        "future_e_counts_used_for_selection": False,
                        "selection_source": "validation_episode_adaptation_window_A_counts_only",
                    }
                )
                selection: object = "episode_specific_from_label_free_A_count_curves"
            else:
                evaluation.update(
                    {
                        "selection_function_received_training_counts_only": True,
                        "validation_counts_used_for_selection": False,
                    }
                )
                selection = {
                    "found": True,
                    "reject": False,
                    "threshold_index": position + 1,
                    "degenerate_k1": method == "source_worst",
                }
            methods[method] = {
                "selection": selection,
                "validation_evaluation": evaluation,
            }
        baseline_budgets.append(
            {
                "budget_position": position,
                "pixel_budget": pixel_budget,
                "component_budget": component_budget,
                "methods": methods,
            }
        )
    baselines = {
        "schema_version": GATE_C_BASELINES_SCHEMA_VERSION,
        "protocol": "source_only_pseudo_target_gate_c_frozen_baselines",
        **_shared_contract(),
        "threshold_grid_schema_version": "rc-v4-logit-dense-tail-grid-v1",
        "train_archive": train_file,
        "train_archive_sha256": train_hash,
        "validation_archive": validation_file,
        "validation_archive_sha256": validation_hash,
        "train_pseudo_targets": [train_target],
        "validation_pseudo_targets": [validation_target],
        "excluded_outer_target": "NUAA-SIRST",
        "labels_policy": {
            "train_source_future_e_labels_used_for_static_selection": True,
            "validation_source_future_e_labels_used_only_after_selection": True,
            "validation_labels_used_for_selection": False,
            "count_all_validation_A_masks_read_for_selection": False,
            "count_all_validation_A_raw_logits_used_for_selection": True,
            "outer_target_labels_used": False,
        },
        "external_reject_action": {
            "threshold": "+inf",
            "threshold_index": None,
            "inside_finite_model_grid": False,
        },
        "count_all": {
            "status": "AVAILABLE_AND_EVALUATED",
            "available": True,
            "evaluated": True,
            "formal_protocol_eligible": True,
            "selection_source": "adaptation_window_A_label_free_counts_only",
            "adaptation_masks_read": False,
            "future_e_counts_used_for_selection": False,
        },
        "budgets": baseline_budgets,
        "complete_required_baseline_matrix_ready": True,
        "status": "COMPLETE",
    }
    baseline_path = fold_root / "baselines.json"
    baseline_path.write_text(json.dumps(baselines, sort_keys=True), encoding="utf-8")
    return comparison_path, baseline_path


def _inputs(tmp_path: Path, *, scenario: str) -> list[tuple[str, Path, Path]]:
    first = _write_fold(
        tmp_path,
        fold_id="val_irstd",
        validation_target="IRSTD-1K",
        train_target="NUDT-SIRST",
        scenario=scenario,
    )
    second = _write_fold(
        tmp_path,
        fold_id="val_nudt",
        validation_target="NUDT-SIRST",
        train_target="IRSTD-1K",
        scenario=scenario,
    )
    return [
        ("val_irstd", *first),
        ("val_nudt", *second),
    ]


def _input_seal(
    folds: list[tuple[str, Path, Path]],
) -> dict[str, object]:
    repository_root = Path(gate_c_module.__file__).resolve().parents[1]
    report_path = folds[0][1].parents[1] / "semantic_preflight.json"
    fold_hashes = {
        fold_id: {
            "comparison_sha256": hashlib.sha256(
                comparison.read_bytes()
            ).hexdigest(),
            "baselines_sha256": hashlib.sha256(baselines.read_bytes()).hexdigest(),
            "reproduced_comparison_semantic_sha256": "1" * 64,
            "reproduced_baselines_semantic_sha256": "2" * 64,
            "comparison_replay_exact": True,
            "baselines_replay_exact": True,
            "referenced_artifacts": {
                "validation_episode_archive": {
                    "path": json.loads(comparison.read_text())["episode_archive"],
                    "sha256": json.loads(comparison.read_text())[
                        "episode_archive_sha256"
                    ],
                },
                "risk_curve_checkpoint": {
                    "path": json.loads(comparison.read_text())[
                        "risk_curve_checkpoint"
                    ],
                    "sha256": json.loads(comparison.read_text())[
                        "risk_curve_checkpoint_sha256"
                    ],
                },
                "rc_direct_checkpoint": {
                    "path": json.loads(comparison.read_text())[
                        "rc_direct_checkpoint"
                    ],
                    "sha256": json.loads(comparison.read_text())[
                        "rc_direct_checkpoint_sha256"
                    ],
                },
                "train_episode_archive": {
                    "path": json.loads(baselines.read_text())["train_archive"],
                    "sha256": json.loads(baselines.read_text())[
                        "train_archive_sha256"
                    ],
                },
            },
        }
        for fold_id, comparison, baselines in folds
    }
    report = {
        "schema_version": GATE_C_SEMANTIC_PREFLIGHT_SCHEMA_VERSION,
        "tool_version": SEMANTIC_PREFLIGHT_TOOL_VERSION,
        "status": "PASS",
        "deep_archive_checkpoint_revalidation_complete": True,
        "submitted_decision_evidence_exactly_reproduced": True,
        "outer_target_labels_used": False,
        "formal_source_domains": ["IRSTD-1K", "NUDT-SIRST"],
        "excluded_outer_target": "NUAA-SIRST",
        "validator_code_sha256s": {
            relative_path: hashlib.sha256(
                (repository_root / relative_path).read_bytes()
            ).hexdigest()
            for relative_path in SEMANTIC_PREFLIGHT_VALIDATOR_FILES
        },
        "folds": fold_hashes,
    }
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return {
        "schema_version": GATE_C_INPUT_SEAL_SCHEMA_VERSION,
        "upstream_semantic_validation_complete": True,
        "outer_target_labels_used": False,
        "semantic_preflight_report": str(report_path.resolve()),
        "semantic_preflight_report_sha256": hashlib.sha256(
            report_path.read_bytes()
        ).hexdigest(),
        "folds": {
            fold_id: {
                "comparison_sha256": fold_hashes[fold_id]["comparison_sha256"],
                "baselines_sha256": fold_hashes[fold_id]["baselines_sha256"],
            }
            for fold_id, comparison, baselines in folds
        },
    }


@pytest.fixture
def synthetic_runtime_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate aggregate-policy unit tests from the real NPZ/PT replay layer."""

    def replay(*, fold_files: object, normalised_input_seal: object) -> dict[str, object]:
        seal = dict(normalised_input_seal)  # type: ignore[arg-type]
        rows = dict(seal["folds"])
        return {
            "performed_inside_aggregate": True,
            "device": "cpu",
            "batch_size": 64,
            "folds": {
                fold_id: {
                    "comparison_sha256": row["comparison_sha256"],
                    "baselines_sha256": row["baselines_sha256"],
                    "reproduced_comparison_semantic_sha256": row[
                        "reproduced_comparison_semantic_sha256"
                    ],
                    "reproduced_baselines_semantic_sha256": row[
                        "reproduced_baselines_semantic_sha256"
                    ],
                    "comparison_replay_exact": True,
                    "baselines_replay_exact": True,
                }
                for fold_id, row in rows.items()
            },
        }

    monkeypatch.setattr(gate_c_module, "_runtime_replay_sealed_inputs", replay)


def test_strict_aggregate_recomputes_complete_matrix_and_can_go(
    tmp_path: Path, synthetic_runtime_replay: None
) -> None:
    output = tmp_path / "aggregate.json"
    folds = _inputs(tmp_path, scenario="controlled")
    aggregate_gate_c(folds=folds, output=output, input_seal=_input_seal(folds))
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == AGGREGATE_GATE_C_SCHEMA_VERSION
    assert payload["decision"] == "GO"
    assert payload["required_methods"] == list(REQUIRED_METHODS)
    assert payload["single_fold_decisions_non_authoritative"] is True
    assert all(payload["criteria"].values())
    assert payload["micro_by_budget"]["0"]["risk_curve"]["aggregate_counts"][
        "num_episodes"
    ] == 4


def test_proposed_still_noncompliant_pd_only_is_hard_hold(
    tmp_path: Path, synthetic_runtime_replay: None
) -> None:
    output = tmp_path / "aggregate.json"
    folds = _inputs(tmp_path, scenario="pd_only")
    aggregate_gate_c(folds=folds, output=output, input_seal=_input_seal(folds))
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["decision"] == "HOLD"
    assert payload["benefit_evidence"]["literal_strict_pd_gain_vs_rc_direct"] is True
    assert payload["benefit_evidence"]["pd_only_can_never_pass"] is True
    assert payload["criteria"]["c2_absolute_risk_budget_control"] is False
    assert payload["criteria"]["c3_risk_non_adverse_to_rc_direct"] is False
    assert payload["criteria"]["c4_stable_benefit_not_pd_only"] is False
    strict = payload["risk_curve_vs_rc_direct"]["by_budget"]["1"]
    assert strict["macro"]["joint_violation_rate"]["status"] == "worse_from_zero"
    assert strict["micro"]["mean_relative_excess"]["status"] == "worse_from_zero"
    assert all(
        fold["single_fold_decision"] == "GO"
        and fold["single_fold_decision_authoritative"] is False
        for fold in payload["folds"].values()
    )


def test_unsafe_direct_high_pd_constrained_comparator_is_diagnostic_only(
    tmp_path: Path, synthetic_runtime_replay: None,
) -> None:
    output = tmp_path / "aggregate.json"
    folds = _inputs(tmp_path, scenario="unsafe_direct_high_pd")
    aggregate_gate_c(
        folds=folds,
        output=output,
        input_seal=_input_seal(folds),
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["decision"] == "HOLD"
    assert payload["criteria"]["c2_absolute_risk_budget_control"] is True
    assert payload["criteria"]["c5_literal_pd_non_degraded_vs_rc_direct"] is False
    assert payload["authoritative_c5_comparator"] == "literal_rc_direct"
    assert payload["constrained_pd_comparator_authoritative"] is False
    assert payload["benefit_evidence"]["feasibility_benefit"] is True
    fold_checks = [
        row
        for row in payload[
            "constrained_pd_vs_best_feasible_baseline_diagnostic"
        ]
        if row["scope"].startswith("fold:")
    ]
    assert fold_checks
    assert all(row["rc_direct_absolute_feasible"] is False for row in fold_checks)
    assert all(row["comparator_method"] == "count_all" for row in fold_checks)
    assert all(row["pd_delta"] == pytest.approx(0.0) for row in fold_checks)
    assert all(row["non_degraded"] is True for row in fold_checks)
    literal_fold_checks = [
        row
        for row in payload["literal_pd_vs_rc_direct_checks"]
        if row["scope"].startswith("fold:")
    ]
    assert all(row["pd_delta"] < -0.02 for row in literal_fold_checks)
    assert all(row["non_degraded"] is False for row in literal_fold_checks)


def test_single_cell_feasibility_plus_pd_gain_cannot_go(
    tmp_path: Path, synthetic_runtime_replay: None
) -> None:
    folds = _inputs(tmp_path, scenario="single_cell_feasibility")
    output = tmp_path / "aggregate.json"
    aggregate_gate_c(folds=folds, output=output, input_seal=_input_seal(folds))
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["decision"] == "HOLD"
    assert payload["criteria"]["c2_absolute_risk_budget_control"] is True
    assert payload["criteria"]["c3_risk_non_adverse_to_rc_direct"] is True
    assert payload["criteria"]["c4_stable_benefit_not_pd_only"] is False
    assert payload["criteria"]["c5_literal_pd_non_degraded_vs_rc_direct"] is True
    evidence = payload["benefit_evidence"]
    assert evidence["violation_benefit"] is False
    assert evidence["excess_benefit"] is False
    assert evidence["literal_strict_pd_gain_vs_rc_direct"] is True
    assert evidence["feasibility_benefit"] is True
    assert evidence["feasibility_benefit_diagnostic_only"] is True
    assert evidence["authoritative_success_criterion_count"] == 1
    assert len(evidence["feasibility_improved_fold_budget_cells"]) == 1


def test_no_feasible_constrained_comparator_fails_closed_diagnostically(
    tmp_path: Path, synthetic_runtime_replay: None,
) -> None:
    folds = _inputs(tmp_path, scenario="no_feasible_comparator")
    output = tmp_path / "aggregate.json"
    aggregate_gate_c(folds=folds, output=output, input_seal=_input_seal(folds))
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["decision"] == "HOLD"
    checks = payload["constrained_pd_vs_best_feasible_baseline_diagnostic"]
    assert checks
    assert all(row["comparator_method"] is None for row in checks)
    assert all(row["pd_delta"] is None for row in checks)
    assert all(row["non_degraded"] is False for row in checks)
    assert payload["criteria"]["c5_literal_pd_non_degraded_vs_rc_direct"] is False


def test_missing_or_mismatched_input_seal_cannot_authorise_go(tmp_path: Path) -> None:
    folds = _inputs(tmp_path, scenario="controlled")
    output = tmp_path / "aggregate.json"
    aggregate_gate_c(folds=folds, output=output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["decision"] == "HOLD"
    assert payload["criteria"]["c0_integrity_and_source_boundary"] is False
    assert payload["provenance_validation"]["input_seal_required_for_go"] is True
    assert payload["provenance_validation"]["input_seal_verified"] is False

    seal = _input_seal(folds)
    seal["folds"]["val_irstd"]["comparison_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="comparison SHA-256 mismatch"):
        aggregate_gate_c(
            folds=folds,
            output=tmp_path / "must-not-exist.json",
            input_seal=seal,
        )


def test_input_replacement_after_seal_validation_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folds = _inputs(tmp_path / "sealed", scenario="unsafe_direct_high_pd")
    replacements = _inputs(tmp_path / "replacement", scenario="controlled")
    seal = _input_seal(folds)
    original_validate = gate_c_module._validate_input_seal

    def replace_after_validation(*args: object, **kwargs: object):
        result = original_validate(*args, **kwargs)
        for (_, comparison, baselines), (
            _replacement_id,
            replacement_comparison,
            replacement_baselines,
        ) in zip(folds, replacements):
            comparison.write_bytes(replacement_comparison.read_bytes())
            baselines.write_bytes(replacement_baselines.read_bytes())
        return result

    monkeypatch.setattr(
        gate_c_module, "_validate_input_seal", replace_after_validation
    )
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError, match="changed after its immutable byte snapshot"):
        aggregate_gate_c(folds=folds, output=output, input_seal=seal)
    assert not output.exists()


def test_machine_generated_preflight_orchestrates_replay_and_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folds = _inputs(tmp_path / "inputs", scenario="controlled")
    comparison_by_episode: dict[str, Path] = {}
    baselines_by_train: dict[str, Path] = {}
    for _fold_id, comparison_path, baseline_path in folds:
        comparison_payload = json.loads(
            comparison_path.read_text(encoding="utf-8")
        )
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        comparison_by_episode[
            str(Path(comparison_payload["episode_archive"]).resolve())
        ] = comparison_path
        baselines_by_train[
            str(Path(baseline_payload["train_archive"]).resolve())
        ] = baseline_path

    def replay_comparison(**kwargs: object) -> Path:
        source = comparison_by_episode[
            str(Path(str(kwargs["episode_file"])).resolve())
        ]
        output = Path(str(kwargs["output"]))
        output.write_bytes(source.read_bytes())
        return output

    def replay_baselines(**kwargs: object) -> Path:
        source = baselines_by_train[
            str(Path(str(kwargs["train_file"])).resolve())
        ]
        output = Path(str(kwargs["output"]))
        output.write_bytes(source.read_bytes())
        return output

    monkeypatch.setattr(
        preflight_module,
        "evaluate_source_pseudo_target_comparison",
        replay_comparison,
    )
    monkeypatch.setattr(
        preflight_module, "evaluate_gate_c_baselines", replay_baselines
    )
    report_path = tmp_path / "semantic_report.json"
    seal_path = tmp_path / "input_seal.json"
    build_gate_c_semantic_preflight(
        folds=folds,
        report_output=report_path,
        seal_output=seal_path,
        device="cpu",
        batch_size=8,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["deep_archive_checkpoint_revalidation_complete"] is True
    assert report["submitted_decision_evidence_exactly_reproduced"] is True
    assert all(
        row["comparison_replay_exact"] and row["baselines_replay_exact"]
        for row in report["folds"].values()
    )
    output = tmp_path / "aggregate.json"
    aggregate_gate_c(
        folds=folds,
        output=output,
        input_seal=json.loads(seal_path.read_text(encoding="utf-8")),
    )
    aggregate = json.loads(output.read_text(encoding="utf-8"))
    assert aggregate["decision"] == "GO"
    assert aggregate["provenance_validation"][
        "machine_generated_semantic_preflight_verified"
    ] is True


def test_nonregistered_budget_fails_closed(tmp_path: Path) -> None:
    folds = _inputs(tmp_path, scenario="controlled")
    comparison_path = folds[0][1]
    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    payload["budgets"][0]["pixel_budget"] = 2e-5
    comparison_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="registered formal Gate C budget"):
        aggregate_gate_c(folds=folds, output=tmp_path / "must-not-exist.json")


def test_noncanonical_source_or_outer_domain_fails_closed(tmp_path: Path) -> None:
    first = _write_fold(
        tmp_path / "source",
        fold_id="val_source_a",
        validation_target="source-a",
        train_target="source-b",
        scenario="controlled",
    )
    second = _write_fold(
        tmp_path / "source",
        fold_id="val_source_b",
        validation_target="source-b",
        train_target="source-a",
        scenario="controlled",
    )
    with pytest.raises(ValueError, match="exactly IRSTD-1K and NUDT-SIRST"):
        aggregate_gate_c(
            folds=[("val_source_a", *first), ("val_source_b", *second)],
            output=tmp_path / "must-not-exist-source.json",
        )

    folds = _inputs(tmp_path / "outer", scenario="controlled")
    for _, _, baseline_path in folds:
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        payload["excluded_outer_target"] = "Other-SIRST"
        baseline_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="excluded outer target must be NUAA-SIRST"):
        aggregate_gate_c(
            folds=folds,
            output=tmp_path / "must-not-exist-outer.json",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "threshold_grid_detector_protocol",
            "outer_inclusive_detector_grid",
            "must use detector protocol",
        ),
        (
            "threshold_grid_detector_checkpoint_sha256s",
            [OUTER_DETECTOR_HASH, OUTER_DETECTOR_HASH, INNER_HASHES[0]],
            "must not contain duplicate",
        ),
        (
            "threshold_grid_episode_detector_checkpoint_sha256s",
            [OUTER_DETECTOR_HASH, INNER_HASHES[0]],
            "Outer detector cannot supervise",
        ),
    ],
)
def test_invalid_detector_role_contract_fails_closed(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    folds = _inputs(tmp_path, scenario="controlled")
    comparison_path = folds[0][1]
    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    payload[field] = value
    comparison_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        aggregate_gate_c(folds=folds, output=tmp_path / "must-not-exist.json")


def test_zero_baseline_relation_distinguishes_tie_from_worse() -> None:
    assert _relation(0.0, 0.0) == {
        "risk_curve": 0.0,
        "rc_direct": 0.0,
        "delta": 0.0,
        "relative_reduction": 0.0,
        "status": "tie_at_zero",
        "non_adverse": True,
    }
    worse = _relation(0.1, 0.0)
    assert worse["status"] == "worse_from_zero"
    assert worse["relative_reduction"] is None
    assert worse["non_adverse"] is False


def test_outer_label_selection_tamper_fails_closed(tmp_path: Path) -> None:
    folds = _inputs(tmp_path, scenario="controlled")
    baseline_path = folds[0][2]
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    payload["labels_policy"]["outer_target_labels_used"] = True
    baseline_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="outer_target_labels_used must be False"):
        aggregate_gate_c(folds=folds, output=tmp_path / "must-not-exist.json")


def test_missing_method_and_episode_identity_tamper_fail_closed(tmp_path: Path) -> None:
    folds = _inputs(tmp_path, scenario="controlled")
    baseline_path = folds[0][2]
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    del payload["budgets"][0]["methods"]["count_all"]
    baseline_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="method matrix is incomplete"):
        aggregate_gate_c(folds=folds, output=tmp_path / "must-not-exist.json")

    folds = _inputs(tmp_path / "identity", scenario="controlled")
    baseline_path = folds[0][2]
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    payload["budgets"][0]["methods"]["count_all"]["validation_evaluation"][
        "per_episode"
    ][0]["evaluation_ids"] = ["different-evaluation-id"]
    baseline_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="episode identity mismatch"):
        aggregate_gate_c(folds=folds, output=tmp_path / "must-not-exist-2.json")
