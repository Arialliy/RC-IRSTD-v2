from __future__ import annotations

from copy import deepcopy

import pytest

from scripts import run_phase3_tier2r_exact_gate as gate


def _point(value: float) -> dict:
    return {
        "found": True,
        "macro_pd": value,
        "pooled_pd": value,
        "worst_pd": value,
        "domain_pd": {"nudt": value, "irstd1k": value},
    }


def _points(control: float = 0.50, c: float = 0.52, cv: float = 0.54) -> dict:
    values = {"control": control, "c": c, "cv": cv}
    return {
        role: {
            seed: {budget: _point(value) for budget, _, _ in gate.BUDGETS}
            for seed in gate.SEEDS
        }
        for role, value in values.items()
    }


def test_two_level_go_selects_cv_and_retains_component_claim() -> None:
    result = gate.evaluate_two_level_decision(_points())
    assert result["decision"] == gate.TIER2R_GO
    assert result["selected_candidate"] == "cv"
    assert result["component_claim_retained"] is True


def test_level_b_hold_falls_back_to_c_and_drops_component_claim() -> None:
    result = gate.evaluate_two_level_decision(_points(cv=0.51))
    assert result["decision"] == gate.TIER2R_GO
    assert result["selected_candidate"] == "c"
    assert result["component_claim_retained"] is False


def test_level_a_hold_selects_nothing() -> None:
    result = gate.evaluate_two_level_decision(_points(c=0.49, cv=0.55))
    assert result["decision"] == gate.TIER2R_HOLD
    assert result["selected_candidate"] is None
    assert result["component_claim_retained"] is False


def test_strict_macro_requires_two_positive_seeds() -> None:
    points = _points()
    for seed, delta in zip(gate.SEEDS, (0.020, -0.002, -0.002)):
        points["c"][seed]["strict"]["macro_pd"] = 0.50 + delta
    result = gate.evaluate_two_level_decision(points)
    criterion = result["levels"]["contrast_vs_control"]["criteria"][0]
    assert criterion["paired_mean_delta"] >= 0.005
    assert criterion["worst_seed_delta"] >= -0.005
    assert criterion["positive_seed_count"] == 1
    assert criterion["passed"] is False
    assert result["decision"] == gate.TIER2R_HOLD


def test_worst_seed_floor_and_nonfinite_fail_closed() -> None:
    points = _points()
    for seed, delta in zip(gate.SEEDS, (0.020, 0.020, -0.010)):
        points["c"][seed]["strict"]["macro_pd"] = 0.50 + delta
    result = gate.evaluate_two_level_decision(points)
    criterion = result["levels"]["contrast_vs_control"]["criteria"][0]
    assert criterion["worst_seed_passed"] is False
    assert result["decision"] == gate.TIER2R_HOLD

    points = _points()
    points["c"][43]["strict"]["macro_pd"] = float("nan")
    result = gate.evaluate_two_level_decision(points)
    criterion = result["levels"]["contrast_vs_control"]["criteria"][0]
    assert criterion["protocol_error"] is not None
    assert result["decision"] == gate.TIER2R_HOLD


def test_missing_point_and_extra_role_fail_closed() -> None:
    points = _points()
    points["c"][43]["strict"] = {"found": False}
    result = gate.evaluate_two_level_decision(points)
    assert result["decision"] == gate.TIER2R_HOLD

    points = _points()
    points["extra"] = deepcopy(points["control"])
    with pytest.raises(ValueError, match="exactly the three roles"):
        gate.evaluate_two_level_decision(points)


def test_selected_checkpoint_set_is_null_on_hold() -> None:
    decision = gate.evaluate_two_level_decision(_points(c=0.49))
    assert gate._selected_checkpoint_set(decision, {}) is None


def _fake_run_bindings() -> dict:
    result = {}
    for spec in gate.RUN_SPECS:
        result[spec.run_id] = {
            "role": spec.role,
            "checkpoint": f"/tmp/{spec.run_id}/last.pt",
            "checkpoint_sha256": "1" * 64,
            "run_identity": f"/tmp/{spec.run_id}/run.json",
            "run_identity_sha256": "2" * 64,
            "export_identity": f"/tmp/{spec.run_id}/export.json",
            "export_identity_sha256": "3" * 64,
            "score_manifest": f"/tmp/{spec.run_id}/manifest.json",
            "score_manifest_sha256": "4" * 64,
        }
    return result


def test_selected_checkpoint_set_follows_fallback_candidate() -> None:
    runs = _fake_run_bindings()
    cv_decision = gate.evaluate_two_level_decision(_points())
    cv_set = gate._selected_checkpoint_set(cv_decision, runs)
    assert cv_set["selected_role"] == "cv"
    assert cv_set["num_checkpoints"] == 6
    assert all(run_id.split("_")[1] == "cv" for run_id in cv_set["runs"])

    c_decision = gate.evaluate_two_level_decision(_points(cv=0.51))
    c_set = gate._selected_checkpoint_set(c_decision, runs)
    assert c_set["selected_role"] == "c"
    assert c_set["num_checkpoints"] == 6
    assert c_decision["component_claim_retained"] is False


def test_finalize_recovers_authorization_and_completion_from_frozen_decision(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "gate"
    output.mkdir()
    handoff = tmp_path / "TIER2R_HANDOFF.json"
    handoff.write_bytes(b"handoff\n")
    evidence_path = output / "evidence_manifest.json"
    gate._write_once_json(
        evidence_path,
        {
            "exact_raw_logit_states_are_primary": True,
            "dense_grid_used_for_decision": False,
        },
    )
    decision = gate.evaluate_two_level_decision(_points(c=0.49))
    decision.update(
        {
            "handoff_sha256": gate.file_sha256(handoff),
            "evidence_manifest_sha256": gate.file_sha256(evidence_path),
        }
    )
    gate._write_once_json(output / "COMPONENT_RESCUE_DECISION.json", decision)
    monkeypatch.setattr(
        gate,
        "validate_handoff",
        lambda *args, **kwargs: ({}, _fake_run_bindings(), {}),
    )

    gate._finalize_gate_chain(tmp_path, output, handoff)
    gate._finalize_gate_chain(tmp_path, output, handoff)

    source = gate._verify_frozen_json(
        output / "SOURCE_TIER3_DESIGN_AUTHORIZATION.json"
    )
    completion = gate._verify_frozen_json(output / gate.COMPLETION_NAME)
    assert source["selected_candidate"] is None
    assert source["selected_checkpoint_set"] is None
    assert source["source_tier3_design_authorized"] is False
    assert completion["completed"] is True
    assert completion["decision_sha256"] == gate.file_sha256(
        output / "COMPONENT_RESCUE_DECISION.json"
    )
