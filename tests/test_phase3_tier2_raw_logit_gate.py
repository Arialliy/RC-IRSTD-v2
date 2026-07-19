from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import pytest

import scripts.run_phase3_tier2_raw_logit_gate as gate
from evaluation.artifact_integrity import file_sha256


REGISTERED_AT = "2026-07-16T12:30:00+08:00"


def _point(value: float) -> dict[str, Any]:
    return {
        "found": True,
        "pooled_pd": value,
        "worst_pd": value,
        "macro_pd": value,
        "domain_pd": {"nudt": value, "irstd1k": value},
    }


def _points(value: float) -> dict[str, dict[str, Any]]:
    return {name: _point(value) for name, _, _ in gate.BUDGETS}


def test_tier2_decision_uses_full_minus_ablation_and_all_pairs_must_pass() -> None:
    decision = gate.evaluate_tier2_decision(
        _points(0.70),
        {
            "no_contrast": _points(0.60),
            "no_component": _points(0.65),
        },
    )
    assert decision["decision"] == gate.TIER2_GO_TIER3
    assert decision["comparison_direction"] == "full_minus_ablation"
    assert decision["authorizes_tier3"] is True
    assert len(decision["conditions"]) == 18
    assert all(condition["passed"] for condition in decision["conditions"])
    delta = decision["evidence"]["no_contrast"]["strict"][
        "deltas_full_minus_ablation"
    ]["macro_pd"]
    assert delta == pytest.approx(0.10)


def test_strict_macro_requires_positive_beyond_tolerance() -> None:
    decision = gate.evaluate_tier2_decision(
        _points(0.70),
        {
            "no_contrast": _points(0.70),
            "no_component": _points(0.65),
        },
    )
    assert decision["decision"] == gate.TIER2_HOLD
    failed = [item for item in decision["conditions"] if not item["passed"]]
    assert any(
        item["role"] == "no_contrast" and item["metric"] == "macro_pd"
        for item in failed
    )
    assert decision["authorizes_tier3"] is False


def _counts(tp: int, fp_pixels: int, fp_components: int) -> dict[str, Any]:
    return {
        "tp_objects": tp,
        "gt_objects": 10,
        "fp_pixels": fp_pixels,
        "fp_components": fp_components,
        "total_pixels": 1_000_000,
        "pd": tp / 10,
    }


def _selected(threshold: float) -> dict[str, Any]:
    a = _counts(5, 1, 1)
    b = _counts(6, 2, 2)
    pooled = {
        "tp_objects": 11,
        "gt_objects": 20,
        "fp_pixels": 3,
        "fp_components": 3,
        "total_pixels": 2_000_000,
        "pd": 0.55,
    }
    point = {
        "found": True,
        "threshold_logit_float32": threshold,
        "state_index": 2,
        "operating_point": pooled,
        "source_rows": {"a": a, "b": b},
    }
    return {"source_pooled": point, "source_worst": dict(point)}


def test_every_selected_threshold_is_legacy_full_image_rechecked() -> None:
    calls: list[float | None] = []

    def evaluator(_samples, threshold, **protocol):
        assert protocol == gate.MATCHING_PROTOCOL
        calls.append(threshold)
        return {
            "threshold_logit_float32": threshold,
            "per_domain": {"a": _counts(5, 1, 1), "b": _counts(6, 2, 2)},
            "pooled": {
                "tp_objects": 11,
                "gt_objects": 20,
                "fp_pixels": 3,
                "fp_components": 3,
                "total_pixels": 2_000_000,
            },
        }

    result = gate.verify_selected_operating_points(
        {"a": (), "b": ()},
        {"strict": _selected(5.0), "medium": _selected(4.0)},
        evaluator=evaluator,
    )
    assert result["all_selected_points_verified"] is True
    assert result["num_unique_thresholds_rechecked"] == 2
    assert sorted(calls) == [4.0, 5.0]


def test_legacy_raw_count_mismatch_fails_closed() -> None:
    def evaluator(_samples, threshold, **_protocol):
        return {
            "threshold_logit_float32": threshold,
            "per_domain": {"a": _counts(4, 1, 1), "b": _counts(6, 2, 2)},
            "pooled": {
                "tp_objects": 10,
                "gt_objects": 20,
                "fp_pixels": 3,
                "fp_components": 3,
                "total_pixels": 2_000_000,
            },
        }

    with pytest.raises(RuntimeError, match="legacy full-image count mismatch"):
        gate.verify_selected_operating_points(
            {"a": (), "b": ()},
            {"strict": _selected(5.0)},
            evaluator=evaluator,
        )


def _fake_rescue(project: Path) -> dict[str, Any]:
    full_curve = project / "frozen-full.json"
    full_curve.write_text("{}\n", encoding="utf-8")
    full_bindings: dict[str, Any] = {}
    for name, _, _ in gate.BUDGETS:
        point = project / f"full-{name}.json"
        point.write_text("{}\n", encoding="utf-8")
        full_bindings[name] = {"path": str(point), "sha256": file_sha256(point)}
    return {
        "full_points": _points(0.70),
        "full_point_bindings": full_bindings,
        "full_curve_binding": {
            "path": str(full_curve),
            "sha256": file_sha256(full_curve),
        },
        "rescue_decision_sha256": "a" * 64,
        "rescue_authorization_sha256": "b" * 64,
        "rescue_evidence_manifest_sha256": "c" * 64,
        "rescue_input_manifest": {"runs": {}},
        "rescue_input_hashes": {"runs": {}},
    }


def _fake_record(run_id: str) -> dict[str, Any]:
    return {
        "role": run_id.split("_heldout_")[0],
        "fold": "heldout_" + run_id.split("_heldout_")[1],
        "held_out_source": "NUDT-SIRST",
        "training_source": "IRSTD-1K",
        "score_dir": f"/scores/{run_id}",
        "score_manifest_sha256": "1" * 64,
        "score_records_sha256": "2" * 64,
        "score_ordered_image_ids_sha256": "3" * 64,
        "score_num_records": 1,
        "raw_logit_stream_sha256": "4" * 64,
        "checkpoint_sha256": "5" * 64,
        "phase3_identity_sha256": "6" * 64,
        "export_identity_sha256": "7" * 64,
        "split_file_sha256": "8" * 64,
        "split_ordered_ids_sha256": "9" * 64,
    }


def _fake_compute_factory():
    values = iter((0.60, 0.65))

    def compute(_samples, *, api):
        del api
        value = next(values)
        selections = {
            name: {"source_pooled": {"found": True}, "source_worst": {"found": True}}
            for name, _, _ in gate.BUDGETS
        }
        return (
            {
                "exact_state_enumeration": True,
                "shared_threshold_across_domains": True,
                "states": [0, 1],
            },
            selections,
            {
                "gate_points": _points(value),
                "all_selected_points_verified": True,
                "num_unique_thresholds_rechecked": 2,
                "entries": [],
            },
        )

    return compute


def _prepare_mock_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    project = tmp_path / "project"
    output = project / gate.TIER2_RELATIVE
    (project / "outputs/phase_state").mkdir(parents=True)
    (project / "outputs/phase_state/HOLD_PHASE3_TARGET_LABEL_ACCESS").write_text(
        "HOLD\n", encoding="utf-8"
    )
    output.mkdir(parents=True)
    handoff = output / gate.DEFAULT_HANDOFF_NAME
    gate._write_once_json(handoff, {})
    runs = {
        spec.run_id: {
            "role": spec.role,
            "fold": spec.fold,
            "held_out_source": spec.held_out_source,
            "training_source": spec.training_source,
        }
        for spec in gate.RUN_SPECS
    }
    monkeypatch.setattr(
        gate,
        "validate_handoff",
        lambda *_args, **_kwargs: (
            {
                "registered_at": REGISTERED_AT,
                "source_only": True,
                "outer_target_images_used": False,
                "outer_target_labels_used": False,
            },
            runs,
        ),
    )
    rescue = _fake_rescue(project)
    monkeypatch.setattr(gate, "_verify_rescue_chain", lambda _root: rescue)
    records = {spec.run_id: _fake_record(spec.run_id) for spec in gate.RUN_SPECS}
    loaded = {"no_contrast": {"nudt": (), "irstd1k": ()}, "no_component": {"nudt": (), "irstd1k": ()}}
    monkeypatch.setattr(
        gate,
        "_load_tier2_inputs",
        lambda *_args, **_kwargs: (loaded, records),
    )
    api = gate.AlgorithmAPI(lambda *_a, **_k: None, lambda *_a, **_k: None, lambda *_a, **_k: None)
    monkeypatch.setattr(gate, "_load_algorithm_api", lambda: api)
    monkeypatch.setattr(gate, "_code_bindings", lambda *_args: {"fake.py": "d" * 64})
    monkeypatch.setattr(gate, "_compute_role_evidence", _fake_compute_factory())
    return project, output, handoff


def test_runner_rejects_noncanonical_output_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _output, handoff = _prepare_mock_run(tmp_path, monkeypatch)
    escaped = tmp_path / "escaped-tier2"
    with pytest.raises(RuntimeError, match="canonical audit directory"):
        gate.run_tier2_gate(
            project_root=project,
            output_root=escaped,
            handoff_path=handoff,
        )
    assert not escaped.exists()


def test_runner_freezes_go_artifacts_and_never_authorizes_outer_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, output, handoff = _prepare_mock_run(tmp_path, monkeypatch)
    result = gate.run_tier2_gate(
        project_root=project,
        output_root=output,
        handoff_path=handoff,
    )
    assert result["decision"] == gate.TIER2_GO_TIER3
    decision = json.loads((output / "tier2_decision.json").read_text())
    authorization = json.loads((output / "effective_authorization.json").read_text())
    assert decision["comparison_direction"] == "full_minus_ablation"
    assert authorization["tier3_source_lodo_authorized"] is True
    assert authorization["outer_target_image_access_authorized"] is False
    assert authorization["outer_target_label_access_authorized"] is False
    assert (
        project / "outputs/phase_state/HOLD_PHASE3_TARGET_LABEL_ACCESS"
    ).read_text() == "HOLD\n"
    for path in output.rglob("*.json"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
        sidecar = path.with_suffix(".sha256")
        assert stat.S_IMODE(sidecar.stat().st_mode) == 0o444
        assert sidecar.read_text() == f"{file_sha256(path)}  {path.name}\n"


def test_algorithm_failure_never_writes_decision_or_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, output, handoff = _prepare_mock_run(tmp_path, monkeypatch)
    monkeypatch.setattr(
        gate,
        "_compute_role_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad exact")),
    )
    with pytest.raises(RuntimeError, match="bad exact"):
        gate.run_tier2_gate(
            project_root=project,
            output_root=output,
            handoff_path=handoff,
        )
    assert not (output / "tier2_decision.json").exists()
    assert not (output / "effective_authorization.json").exists()


def test_restart_repairs_decision_without_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, output, handoff = _prepare_mock_run(tmp_path, monkeypatch)
    first = gate.run_tier2_gate(
        project_root=project,
        output_root=output,
        handoff_path=handoff,
    )
    assert first["decision"] == gate.TIER2_GO_TIER3
    authorization = output / "effective_authorization.json"
    authorization.with_suffix(".sha256").unlink()
    authorization.unlink()

    monkeypatch.setattr(gate, "_compute_role_evidence", _fake_compute_factory())
    resumed = gate.run_tier2_gate(
        project_root=project,
        output_root=output,
        handoff_path=handoff,
    )
    assert resumed["decision"] == gate.TIER2_GO_TIER3
    assert authorization.is_file()
    assert authorization.with_suffix(".sha256").is_file()
    assert stat.S_IMODE(authorization.stat().st_mode) == 0o444
