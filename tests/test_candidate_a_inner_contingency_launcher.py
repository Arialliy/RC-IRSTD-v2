from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts import launch_candidate_a_inner_after_tailmiss_hold as launcher


def _gate_payload(decision: str) -> dict[str, object]:
    criteria = {f"c{index}": True for index in range(8)}
    if decision == "HOLD":
        criteria["c5"] = False
    return {
        "decision": decision,
        "criteria": criteria,
        "aggregation_contract": {
            "criteria_not_lowered": True,
            "pd_non_degradation_floor": 0.0,
        },
        "outer_target_labels_used": False,
    }


def test_tailmiss_gate_requires_valid_strict_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate_dir = tmp_path / "gate_c"
    comparisons = (
        gate_dir / "val_irstd/comparison_seed42.json",
        gate_dir / "val_nudt/comparison_seed42.json",
    )
    for path in comparisons:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    gate_path = tmp_path / "aggregate.json"
    gate_path.write_text(json.dumps(_gate_payload("HOLD")) + "\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "GATE_PATH", gate_path)
    monkeypatch.setattr(launcher, "GATE_COMPARISON_DIR", gate_dir)
    monkeypatch.setattr(
        launcher.phase2, "_json_artifact_complete", lambda step: (True, None)
    )

    evidence = launcher._tailmiss_hold_evidence()

    assert evidence["decision"] == "HOLD"
    assert evidence["path"] == str(gate_path)
    assert set(evidence["comparison_sha256"]) == {str(path) for path in comparisons}


def test_tailmiss_go_cannot_unlock_contingency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate_dir = tmp_path / "gate_c"
    for relative in (
        "val_irstd/comparison_seed42.json",
        "val_nudt/comparison_seed42.json",
    ):
        path = gate_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    gate_path = tmp_path / "aggregate.json"
    gate_path.write_text(json.dumps(_gate_payload("GO")) + "\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "GATE_PATH", gate_path)
    monkeypatch.setattr(launcher, "GATE_COMPARISON_DIR", gate_dir)
    monkeypatch.setattr(
        launcher.phase2, "_json_artifact_complete", lambda step: (True, None)
    )

    with pytest.raises(RuntimeError, match="decision=HOLD"):
        launcher._tailmiss_hold_evidence()


def test_candidate_full_checkpoint_must_pass_epoch19_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = launcher.phase2.CheckpointSpec(
        "full_sources",
        launcher.FULL_CHECKPOINT,
        ("NUDT-SIRST", "IRSTD-1K"),
        None,
    )
    spec = launcher.phase2.BranchSpec(
        "candidate_a", "candidate", Path("/tmp/out"), 2, (checkpoint,), {}
    )
    monkeypatch.setattr(
        launcher.phase2, "_audit_checkpoint", lambda spec, value: {
            "valid": False,
            "problems": ["metadata_mismatch:epoch"],
        }
    )

    with pytest.raises(RuntimeError, match="epoch-19"):
        launcher._candidate_full_checkpoint_evidence(spec)


def test_gpu_observer_is_limited_to_physical_gpu_zero_and_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[int, ...]] = []

    def fake_query(indices: tuple[int, ...]) -> tuple[dict[str, object], ...]:
        observed.append(indices)
        return ()

    monkeypatch.setattr(launcher.safety, "_gpu_compute_processes", fake_query)
    assert launcher._gpu_compute_processes() == ()
    assert observed == [(0, 1)]


def test_candidate_inner_configs_are_exact_source_only_gpu_pair() -> None:
    expected = (
        ("cuda:0", "IRSTD-1K", "../datasets/IRSTD-1K"),
        ("cuda:1", "NUDT-SIRST", "../datasets/NUDT-SIRST"),
    )
    for run, (device, source, source_path) in zip(launcher.RUNS, expected, strict=True):
        config = yaml.safe_load(Path(run["config"]).read_text(encoding="utf-8"))
        assert config["device"] == device
        assert config["data"]["sources"] == [
            {"name": source, "path": source_path}
        ]
        assert config["data"]["val_split"] is None
        assert config["data"]["diagnostic_test_eval"] is False
        assert "nuaa" not in json.dumps(config).casefold()
        assert config["training"]["checkpoint_selection"] == "fixed_last"
        assert config["training"]["epochs"] == 20


def test_frozen_snapshot_rejects_same_size_drift(tmp_path: Path) -> None:
    path = tmp_path / "input.py"
    path.write_text("alpha\n", encoding="utf-8")
    snapshot = launcher._snapshot((path,))
    path.write_text("omega\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="drifted"):
        launcher._verify_snapshot(snapshot)


def test_source_only_path_guard_rejects_outer_dataset_path() -> None:
    with pytest.raises(RuntimeError, match="Outer-domain path"):
        launcher._assert_source_only_paths((Path("/datasets/NUAA-SIRST/masks"),))


def test_failed_dry_run_never_queries_gpu_or_starts_process(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        launcher,
        "audit_readiness",
        lambda: (_ for _ in ()).throw(RuntimeError("Tail/Miss gate absent")),
    )
    monkeypatch.setattr(
        launcher,
        "_gpu_compute_processes",
        lambda: (_ for _ in ()).throw(AssertionError("GPU queried")),
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("process started")),
    )

    assert launcher.main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["ready"] is False
    assert payload["gpu_processes_queried"] is False
    assert payload["gpu_work_started"] is False
    assert payload["outer_target_label_paths_opened"] is False


def test_parser_is_audit_only_by_default() -> None:
    args = launcher.build_parser().parse_args([])
    assert args.execute is False
    assert args.clear_observations == 3
    assert args.poll_seconds == 30.0
