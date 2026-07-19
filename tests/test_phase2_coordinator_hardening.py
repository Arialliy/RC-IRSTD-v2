from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml

import scripts.coordinate_phase2_rc_mshnet_gate as coordinator


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _temporary_run(tmp_path: Path) -> tuple[coordinator.GateRun, Path, Path]:
    config = yaml.safe_load(
        (ROOT / "configs/phase2_rc_mshnet_full_outer_nuaa.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(config, dict)
    config["output_dir"] = "run"
    for index, source in enumerate(config["data"]["sources"]):
        source_root = tmp_path / f"source-{index}"
        source_root.mkdir(parents=True)
        (source_root / "train.txt").write_text(
            f"train-{index}-0\ntrain-{index}-1\n", encoding="utf-8"
        )
        (source_root / "test.txt").write_text(
            f"test-{index}-0\n", encoding="utf-8"
        )
        source["path"] = str(source_root)
    config_path = tmp_path / "phase2.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    initializer = tmp_path / "initializer.pt"
    initializer.write_bytes(b"frozen-initializer")
    run = coordinator.GateRun("rc_mshnet_full", config_path, 1)
    run_dir = coordinator.load_gate_output_dir(config_path)
    run_dir.mkdir(parents=True)
    return run, initializer, run_dir


def _launcher_payload(
    run: coordinator.GateRun,
    initializer: Path,
    *,
    pid: int = 987_654_321,
) -> dict[str, Any]:
    run_dir = coordinator.load_gate_output_dir(run.config_path)
    launch_id = "a" * 32
    process_identity = {
        "pid": pid,
        "state": "S",
        "start_time_ticks": 101,
        "argv": coordinator._phase2_command(run, initializer),
        "cwd": str(coordinator.ROOT),
        "process_group_id": pid,
        "session_id": pid,
        "launch_id": launch_id,
    }
    return {
        "schema_version": coordinator.PHASE2_LAUNCH_SCHEMA,
        "role": run.role,
        "pid": pid,
        "launch_id": launch_id,
        "physical_gpu": run.gpu,
        "visible_device": "cuda:0",
        "command": coordinator._phase2_command(run, initializer),
        "config_path": str(run.config_path.resolve()),
        "config_sha256": coordinator.sha256_file(run.config_path),
        "initializer_path": str(initializer.resolve()),
        "initializer_sha256": coordinator.sha256_file(initializer),
        "expected_output_dir": str(run_dir),
        "process_identity": process_identity,
        "launched_at": "2026-07-15T00:00:00+08:00",
    }


def _write_complete_checkpoint(
    run: coordinator.GateRun,
    initializer: Path,
) -> Path:
    run_dir = coordinator.load_gate_output_dir(run.config_path)
    effective = coordinator._expected_effective_config(run, initializer)
    _write_json(run_dir / "config.json", effective)
    initialization = {
        "source_path": str(initializer.resolve()),
        "source_sha256": coordinator.sha256_file(initializer),
        "backbone_fully_loaded": True,
        "zero_residual_identity_preserved": True,
    }
    _write_json(run_dir / "initialization_report.json", initialization)
    (run_dir / "history.csv").write_text(
        "epoch,lr,loss_total\n"
        + "".join(f"{epoch},0.0002,0.0\n" for epoch in range(80)),
        encoding="utf-8",
    )
    source_names = [
        str(source["name"]) for source in effective["data"]["sources"]
    ]
    source_split_records = []
    for source in effective["data"]["sources"]:
        source_root = Path(str(source["path"])).resolve()
        train_path = source_root / "train.txt"
        test_path = source_root / "test.txt"
        train_ids = coordinator.ensure_unique_sample_ids(
            coordinator.read_split_file(train_path)
        )
        test_ids = coordinator.ensure_unique_sample_ids(
            coordinator.read_split_file(test_path)
        )
        source_split_records.append(
            {
                "name": source["name"],
                "path": str(source_root),
                "train_split_file": str(train_path),
                "train_split_file_sha256": coordinator.sha256_file(train_path),
                "train_ordered_ids_sha256": coordinator.ordered_ids_sha256(
                    train_ids
                ),
                "num_train_samples": len(train_ids),
                "test_split_file": str(test_path),
                "test_split_file_sha256": coordinator.sha256_file(test_path),
                "test_ordered_ids_sha256": coordinator.ordered_ids_sha256(test_ids),
                "num_test_samples": len(test_ids),
                "train_test_id_overlap": False,
            }
        )
    model = coordinator.build_mshnet(effective["model"])
    model_config = dict(model.export_config())
    model_state = model.state_dict()
    payload = {
        "format_version": 2,
        "kind": "detector",
        "epoch": 79,
        "checkpoint_selection": "fixed_last",
        "test_labels_used_for_selection": False,
        "diagnostic_test_eval": False,
        "diagnostic_only": False,
        "formal_paper_checkpoint": True,
        "inference_head": "multi_scale_fused",
        "warm_flag": True,
        "initialization": initialization,
        "config": effective,
        "source_names": source_names,
        "source_split_records": source_split_records,
        "model_config": model_config,
        "model_state": model_state,
        "optimizer_state": {},
        "scheduler_state": {},
        "scaler_state": {},
        "rng_state": {"torch_cuda": [torch.zeros(8, dtype=torch.uint8)]},
        "balanced_batcher_state": {},
        "resume_contract": {
            "model": model_config,
            "training": {"epochs": 80, "warmup_epochs": 0},
        },
    }
    checkpoint = run_dir / "last.pt"
    torch.save(payload, checkpoint)
    (run_dir / "train.stdout_stderr.log").write_text(
        str(checkpoint.resolve()) + "\n", encoding="utf-8"
    )
    return checkpoint


def _write_partial_checkpoint(
    run: coordinator.GateRun,
    initializer: Path,
    *,
    history_rows: int = 3,
) -> Path:
    assert 1 <= history_rows < 80
    checkpoint = _write_complete_checkpoint(run, initializer)
    run_dir = coordinator.load_gate_output_dir(run.config_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["epoch"] = history_rows - 1
    payload.update(
        {
            "optimizer_state": {},
            "scheduler_state": {},
            "scaler_state": {},
            "rng_state": {"torch_cuda": [torch.zeros(8, dtype=torch.uint8)]},
            "balanced_batcher_state": {},
        }
    )
    torch.save(payload, checkpoint)
    (run_dir / "history.csv").write_text(
        "epoch,lr,loss_total\n"
        + "".join(
            f"{epoch},0.0002,{1.0 / (epoch + 1):.8f}\n"
            for epoch in range(history_rows)
        ),
        encoding="utf-8",
    )
    (run_dir / "train.stdout_stderr.log").write_text(
        "interrupted before successful trailer\n", encoding="utf-8"
    )
    return checkpoint


def _patch_state_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    state = tmp_path / "phase_state"
    audit = tmp_path / "audit"
    initializers = tmp_path / "initializers"
    monkeypatch.setattr(coordinator, "STATE_DIR", state)
    monkeypatch.setattr(coordinator, "AUDIT_DIR", audit)
    monkeypatch.setattr(coordinator, "INITIALIZER_DIR", initializers)
    monkeypatch.setattr(coordinator, "HOLD", state / "HOLD_RC_MSHNET_GATE")
    monkeypatch.setattr(coordinator, "ALLOW", state / "ALLOW_RC_MSHNET_GATE")
    monkeypatch.setattr(coordinator, "LOG_PATH", state / "coordinator.log")
    monkeypatch.setattr(coordinator, "LOCK_PATH", state / "coordinator.lock")
    monkeypatch.setattr(coordinator, "PID_PATH", state / "coordinator.pid")
    return state


def test_phase2_schedule_uses_only_physical_gpu_two_and_three() -> None:
    rounds = (coordinator.ROUND1, coordinator.ROUND2, coordinator.ROUND3)
    runs = [run for round_runs in rounds for run in round_runs]
    assert len(runs) == 6
    assert len({run.role for run in runs}) == 6
    assert all(len(round_runs) == 2 for round_runs in rounds)
    assert all(
        {run.gpu for run in round_runs} == {2, 3} for round_runs in rounds
    )


def test_launcher_binding_rejects_pid_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, initializer, run_dir = _temporary_run(tmp_path)
    launcher = _launcher_payload(run, initializer)
    _write_json(run_dir / "launcher.json", launcher)
    live_identity = dict(launcher["process_identity"])
    monkeypatch.setattr(
        coordinator, "_read_process_identity", lambda _pid: live_identity
    )
    _, live = coordinator.validate_launcher_binding(
        run, initializer, require_live=True
    )
    assert live is True

    reused_identity = dict(live_identity)
    reused_identity["start_time_ticks"] = 202
    monkeypatch.setattr(
        coordinator, "_read_process_identity", lambda _pid: reused_identity
    )
    with pytest.raises(RuntimeError, match="PID identity mismatch"):
        coordinator.validate_launcher_binding(run, initializer, require_live=True)


def test_launch_identity_wait_tolerates_transient_pre_exec_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch_id = "c" * 32
    identities = [
        {
            "pid": 123,
            "state": "R",
            "argv": [],
            "cwd": str(coordinator.ROOT),
            "process_group_id": 123,
            "session_id": 123,
            "launch_id": None,
        },
        {
            "pid": 123,
            "state": "R",
            "argv": ["python", "train"],
            "cwd": str(coordinator.ROOT),
            "process_group_id": 123,
            "session_id": 123,
            "launch_id": launch_id,
        },
    ]
    monkeypatch.setattr(
        coordinator,
        "_read_process_identity",
        lambda _pid: identities.pop(0),
    )
    monkeypatch.setattr(coordinator.time, "sleep", lambda _seconds: None)
    identity = coordinator._wait_for_launch_identity(
        123,
        launch_id,
        expected_argv=["python", "train"],
        expected_cwd=coordinator.ROOT,
        timeout_seconds=1.0,
    )
    assert identity["launch_id"] == launch_id
    assert identities == []


def test_launch_identity_wait_rejects_foreign_nonempty_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        coordinator,
        "_read_process_identity",
        lambda _pid: {
            "pid": 123,
            "state": "R",
            "process_group_id": 123,
            "session_id": 123,
            "launch_id": "d" * 32,
        },
    )
    with pytest.raises(RuntimeError, match="foreign launch ID"):
        coordinator._wait_for_launch_identity(
            123, "c" * 32, timeout_seconds=1.0
        )


def test_complete_checkpoint_is_bound_to_config_launcher_and_initializer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, initializer, run_dir = _temporary_run(tmp_path)
    _write_json(run_dir / "launcher.json", _launcher_payload(run, initializer))
    checkpoint = _write_complete_checkpoint(run, initializer)
    monkeypatch.setattr(
        coordinator,
        "_read_process_identity",
        lambda _pid: (_ for _ in ()).throw(FileNotFoundError()),
    )
    report = coordinator.verify_gate_checkpoint(
        run, coordinator.sha256_file(initializer), initializer
    )
    assert report["initializer_sha256"] == coordinator.sha256_file(initializer)
    assert (run_dir / "PHASE2_IDENTITY.json").is_file()

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["config"]["training"]["initialize_from"] = "/foreign/initializer.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="embedded config"):
        coordinator.verify_gate_checkpoint(
            run, coordinator.sha256_file(initializer), initializer
        )


def test_completed_resume_checkpoint_requires_bound_completed_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, initializer, run_dir = _temporary_run(tmp_path)
    _patch_state_paths(monkeypatch, tmp_path / "state-root")
    checkpoint = _write_complete_checkpoint(run, initializer)
    launch_id = "f" * 32
    previous_launch_id = "a" * 32
    pid = 987_654_321
    snapshot_bytes = b"immutable partial checkpoint"
    source_seed = tmp_path / "source-seed.pt"
    source_seed.write_bytes(snapshot_bytes)
    source_sha = coordinator.sha256_file(source_seed)
    snapshot = (
        run_dir
        / "resume_snapshots"
        / f"epoch_0002_{source_sha[:16]}.pt"
    )
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(snapshot_bytes)
    audit_path = coordinator._resume_audit_path(run, launch_id)
    resume = {
        "source_checkpoint_path": str(checkpoint.resolve()),
        "source_checkpoint_sha256": source_sha,
        "source_epoch": 2,
        "history_rows_before_resume": 3,
        "history_sha256_before_resume": "b" * 64,
        "previous_launch_id": previous_launch_id,
        "previous_launcher_sha256": "c" * 64,
        "audit_path": str(audit_path),
    }
    command = coordinator._phase2_command(
        run,
        initializer,
        resume_checkpoint=checkpoint,
        resume_audit=audit_path,
    )
    launcher = _launcher_payload(run, initializer, pid=pid)
    launcher.update(
        {
            "schema_version": coordinator.PHASE2_RESUME_LAUNCH_SCHEMA,
            "launch_id": launch_id,
            "launch_mode": "exact_resume",
            "command": command,
            "resume": resume,
        }
    )
    launcher["process_identity"].update(
        {"argv": command, "launch_id": launch_id}
    )
    _write_json(run_dir / "launcher.json", launcher)
    final_sha = coordinator.sha256_file(checkpoint)
    event = {
        "owner_pid": pid,
        "process_group_id": pid,
        "session_id": pid,
        "source_checkpoint_path": str(checkpoint.resolve()),
        "immutable_snapshot_path": str(snapshot.resolve()),
        "source_checkpoint_sha256": source_sha,
        "source_epoch": 2,
        "resume_start_epoch": 3,
        "history_rows_before_resume": 3,
        "history_sha256_before_resume": "b" * 64,
        "cuda_visible_devices": str(run.gpu),
        "physical_gpu_index": run.gpu,
        "logical_device": "cuda:0",
        "formal_config_sha256": coordinator.sha256_file(run_dir / "config.json"),
        "final_checkpoint_path": str(checkpoint.resolve()),
        "final_checkpoint_sha256": final_sha,
        "status": "completed",
    }
    _write_json(
        audit_path,
        {
            "schema_version": coordinator.FORMAL_RESUME_AUDIT_SCHEMA,
            "status": "completed",
            "events": [event],
        },
    )
    monkeypatch.setattr(
        coordinator,
        "_read_process_identity",
        lambda _pid: (_ for _ in ()).throw(FileNotFoundError()),
    )

    initializer_sha = coordinator.sha256_file(initializer)
    report = coordinator.verify_gate_checkpoint(run, initializer_sha, initializer)

    assert report["launch_mode"] == "exact_resume"
    assert report["resume_audit"]["sha256"] == coordinator.sha256_file(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["status"] = "running"
    _write_json(audit_path, audit)
    with pytest.raises(RuntimeError, match="resume audit is not completed"):
        coordinator.verify_gate_checkpoint(run, initializer_sha, initializer)


def test_round_poll_failure_stops_every_registered_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, initializer, _ = _temporary_run(tmp_path)

    class FakeProcess:
        returncode = None

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    checks = 0

    def fail_on_poll(_initializer: Path, _binding: dict[str, Any]) -> None:
        nonlocal checks
        checks += 1
        if checks >= 2:
            raise RuntimeError("sentinel changed")

    stopped: list[int] = []
    monkeypatch.setattr(coordinator, "_capture_round_binding", lambda _path: {})
    monkeypatch.setattr(coordinator, "_assert_round_binding", fail_on_poll)
    monkeypatch.setattr(
        coordinator, "start_or_attach", lambda _run, _path: (FakeProcess(), 444)
    )
    monkeypatch.setattr(
        coordinator, "stop_jobs", lambda pids, **_kwargs: stopped.extend(pids)
    )
    with pytest.raises(RuntimeError, match="sentinel changed"):
        coordinator.run_round("test", (run,), initializer)
    assert stopped == [444]


def test_launch_metadata_failure_cannot_leave_an_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, initializer, _ = _temporary_run(tmp_path)
    state = _patch_state_paths(monkeypatch, tmp_path / "state-root")
    state.mkdir(parents=True)
    coordinator.ALLOW.touch()
    launch_id = "b" * 32

    class FakeProcess:
        pid = 555

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    monkeypatch.setattr(coordinator.uuid, "uuid4", lambda: SimpleNamespace(hex=launch_id))
    monkeypatch.setattr(coordinator.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(
        coordinator,
        "_read_process_identity",
        lambda _pid: {
            "pid": 555,
            "state": "S",
            "start_time_ticks": 1,
            "argv": coordinator._phase2_command(run, initializer),
            "cwd": str(coordinator.ROOT),
            "process_group_id": 555,
            "session_id": 555,
            "launch_id": launch_id,
        },
    )
    stopped: list[int] = []
    monkeypatch.setattr(
        coordinator, "stop_jobs", lambda pids, **_kwargs: stopped.extend(pids)
    )
    monkeypatch.setattr(
        coordinator,
        "atomic_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("audit write failed")),
    )
    with pytest.raises(OSError, match="audit write failed"):
        coordinator.start_or_attach(run, initializer)
    assert stopped == [555]
    assert not (coordinator.load_gate_output_dir(run.config_path) / "launcher.json").exists()


def test_dead_identity_verified_zero_epoch_launcher_is_archived_before_relaunch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, initializer, run_dir = _temporary_run(tmp_path)
    state = _patch_state_paths(monkeypatch, tmp_path / "state-root")
    state.mkdir(parents=True)
    coordinator.ALLOW.touch()
    stale = _launcher_payload(run, initializer)
    _write_json(run_dir / "launcher.json", stale)
    _write_json(
        run_dir / "config.json",
        coordinator._expected_effective_config(run, initializer),
    )

    launch_id = "e" * 32

    class FakeProcess:
        pid = 777

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    monkeypatch.setattr(
        coordinator,
        "validate_launcher_binding",
        lambda *_args, **_kwargs: (stale, False),
    )
    monkeypatch.setattr(coordinator.uuid, "uuid4", lambda: SimpleNamespace(hex=launch_id))
    monkeypatch.setattr(coordinator.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(
        coordinator,
        "_wait_for_launch_identity",
        lambda _pid, _launch_id, **_kwargs: {
            "pid": 777,
            "state": "S",
            "start_time_ticks": 2,
            "argv": coordinator._phase2_command(run, initializer),
            "cwd": str(coordinator.ROOT),
            "process_group_id": 777,
            "session_id": 777,
            "launch_id": launch_id,
        },
    )
    coordinator.HOLD.touch()
    coordinator.ALLOW.unlink()
    archived = coordinator.recover_uncommitted_phase2_attempts(
        initializer, runs=(run,)
    )
    assert len(archived) == 1
    coordinator.HOLD.unlink()
    coordinator.ALLOW.touch()
    process, pid = coordinator.start_or_attach(run, initializer)
    assert process is not None
    assert pid == 777
    archive = (
        coordinator.AUDIT_DIR
        / "phase2_failed_launches"
        / f"{run.role}_{stale['launch_id']}"
    )
    assert (archive / "launcher.json").is_file()
    recovery = json.loads((archive / "RECOVERY.json").read_text(encoding="utf-8"))
    assert recovery["status"] == "archived_zero_epoch_attempt"
    assert (coordinator.load_gate_output_dir(run.config_path) / "launcher.json").is_file()


def test_zero_epoch_recovery_refuses_any_history_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, initializer, run_dir = _temporary_run(tmp_path)
    state = _patch_state_paths(monkeypatch, tmp_path / "state-root")
    state.mkdir(parents=True)
    coordinator.HOLD.touch()
    launcher = _launcher_payload(run, initializer)
    _write_json(run_dir / "launcher.json", launcher)
    (run_dir / "history.csv").write_text(
        "epoch,lr,loss_total\n0,0.001,1.0\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="training state exists"):
        coordinator._archive_zero_epoch_attempt(
            run,
            initializer,
            reason="test",
            launcher=launcher,
        )


def test_valid_dead_partial_is_resume_candidate_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, initializer, run_dir = _temporary_run(tmp_path)
    _write_partial_checkpoint(run, initializer, history_rows=3)
    _write_json(run_dir / "launcher.json", _launcher_payload(run, initializer))
    monkeypatch.setattr(
        coordinator,
        "_read_process_identity",
        lambda _pid: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monkeypatch.setattr(coordinator, "_process_group_alive", lambda _pgid: False)
    evidence = [
        run_dir / "config.json",
        run_dir / "history.csv",
        run_dir / "last.pt",
        run_dir / "launcher.json",
    ]
    before = {path: coordinator.sha256_file(path) for path in evidence}

    plan = coordinator.inspect_partial_resume(run, initializer)

    assert plan.checkpoint_epoch == 2
    assert plan.history_rows == 3
    assert plan.checkpoint_sha256 == before[run_dir / "last.pt"]
    assert {path: coordinator.sha256_file(path) for path in evidence} == before
    assert not list(run_dir.glob("*.tmp"))


@pytest.mark.parametrize(
    "tamper,match",
    [
        ("history_ahead", "checkpoint/history epoch mismatch"),
        ("checkpoint_tmp", "incomplete checkpoint temporary"),
        ("missing_rng", "partial checkpoint is missing"),
        ("config_drift", "partial config.json drifted"),
        ("initializer_drift", "partial initializer binding mismatch"),
    ],
)
def test_partial_resume_inconsistent_evidence_fails_closed(
    tamper: str,
    match: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, initializer, run_dir = _temporary_run(tmp_path)
    checkpoint = _write_partial_checkpoint(run, initializer, history_rows=3)
    _write_json(run_dir / "launcher.json", _launcher_payload(run, initializer))
    if tamper == "history_ahead":
        with (run_dir / "history.csv").open("a", encoding="utf-8") as handle:
            handle.write("3,0.0002,0.1\n")
    elif tamper == "checkpoint_tmp":
        (run_dir / "last.pt.tmp").write_bytes(b"incomplete")
    elif tamper == "missing_rng":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        payload.pop("rng_state")
        torch.save(payload, checkpoint)
    elif tamper == "config_drift":
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        config["seed"] = 999
        _write_json(run_dir / "config.json", config)
    elif tamper == "initializer_drift":
        report = json.loads(
            (run_dir / "initialization_report.json").read_text(encoding="utf-8")
        )
        report["source_sha256"] = "0" * 64
        _write_json(run_dir / "initialization_report.json", report)
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(tamper)
    monkeypatch.setattr(
        coordinator,
        "_read_process_identity",
        lambda _pid: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monkeypatch.setattr(coordinator, "_process_group_alive", lambda _pgid: False)
    evidence = [
        path
        for path in (
            run_dir / "history.csv",
            run_dir / "last.pt",
            run_dir / "last.pt.tmp",
            run_dir / "config.json",
            run_dir / "initialization_report.json",
            run_dir / "launcher.json",
        )
        if path.exists()
    ]
    before = {path: coordinator.sha256_file(path) for path in evidence}

    with pytest.raises(RuntimeError, match=match):
        coordinator.inspect_partial_resume(run, initializer)
    assert {path: coordinator.sha256_file(path) for path in evidence} == before


def test_recovery_preserves_valid_partial_and_archives_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, initializer, run_dir = _temporary_run(tmp_path)
    state = _patch_state_paths(monkeypatch, tmp_path / "state-root")
    state.mkdir(parents=True)
    coordinator.HOLD.touch()
    _write_partial_checkpoint(run, initializer, history_rows=3)
    _write_json(run_dir / "launcher.json", _launcher_payload(run, initializer))
    monkeypatch.setattr(
        coordinator,
        "_read_process_identity",
        lambda _pid: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monkeypatch.setattr(coordinator, "_process_group_alive", lambda _pgid: False)
    evidence = [run_dir / "history.csv", run_dir / "last.pt", run_dir / "launcher.json"]
    before = {path: coordinator.sha256_file(path) for path in evidence}

    assert coordinator.recover_uncommitted_phase2_attempts(
        initializer, runs=(run,)
    ) == []

    assert {path: coordinator.sha256_file(path) for path in evidence} == before
    assert not (coordinator.AUDIT_DIR / "phase2_failed_launches").exists()


def test_cleanly_failed_prior_resume_is_not_retried_automatically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, initializer, run_dir = _temporary_run(tmp_path)
    _patch_state_paths(monkeypatch, tmp_path / "state-root")
    checkpoint = _write_partial_checkpoint(run, initializer, history_rows=3)
    launch_id = "d" * 32
    audit_path = coordinator._resume_audit_path(run, launch_id)
    resume = {
        "source_checkpoint_path": str(checkpoint.resolve()),
        "source_checkpoint_sha256": coordinator.sha256_file(checkpoint),
        "source_epoch": 2,
        "history_rows_before_resume": 3,
        "history_sha256_before_resume": coordinator.sha256_file(
            run_dir / "history.csv"
        ),
        "previous_launch_id": "a" * 32,
        "previous_launcher_sha256": "b" * 64,
        "audit_path": str(audit_path),
    }
    command = coordinator._phase2_command(
        run,
        initializer,
        resume_checkpoint=checkpoint,
        resume_audit=audit_path,
    )
    launcher = _launcher_payload(run, initializer)
    launcher.update(
        {
            "schema_version": coordinator.PHASE2_RESUME_LAUNCH_SCHEMA,
            "launch_id": launch_id,
            "launch_mode": "exact_resume",
            "command": command,
            "resume": resume,
        }
    )
    launcher["process_identity"].update(
        {"argv": command, "launch_id": launch_id}
    )
    _write_json(run_dir / "launcher.json", launcher)
    _write_json(
        audit_path,
        {
            "schema_version": coordinator.FORMAL_RESUME_AUDIT_SCHEMA,
            "status": "failed",
            "events": [
                {
                    "source_checkpoint_path": str(checkpoint.resolve()),
                    "source_checkpoint_sha256": coordinator.sha256_file(checkpoint),
                    "source_epoch": 2,
                    "history_rows_before_resume": 3,
                    "history_sha256_before_resume": coordinator.sha256_file(
                        run_dir / "history.csv"
                    ),
                    "physical_gpu_index": run.gpu,
                    "status": "failed",
                }
            ],
        },
    )
    monkeypatch.setattr(
        coordinator,
        "_read_process_identity",
        lambda _pid: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monkeypatch.setattr(coordinator, "_process_group_alive", lambda _pgid: False)

    with pytest.raises(RuntimeError, match="audit status 'failed'"):
        coordinator.inspect_partial_resume(run, initializer)


def test_start_or_attach_dead_partial_launches_runtime_exact_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, initializer, run_dir = _temporary_run(tmp_path)
    state = _patch_state_paths(monkeypatch, tmp_path / "state-root")
    state.mkdir(parents=True)
    coordinator.ALLOW.touch()
    _write_partial_checkpoint(run, initializer, history_rows=3)
    old_launcher = _launcher_payload(run, initializer)
    _write_json(run_dir / "launcher.json", old_launcher)
    old_launcher_sha = coordinator.sha256_file(run_dir / "launcher.json")
    launch_id = "e" * 32
    captured: dict[str, Any] = {}

    class FakeProcess:
        pid = 777

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    def fake_popen(command: list[str], **kwargs: Any) -> FakeProcess:
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return FakeProcess()

    monkeypatch.setattr(
        coordinator,
        "_read_process_identity",
        lambda _pid: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monkeypatch.setattr(coordinator, "_process_group_alive", lambda _pgid: False)
    monkeypatch.setattr(coordinator.uuid, "uuid4", lambda: SimpleNamespace(hex=launch_id))
    monkeypatch.setattr(coordinator.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        coordinator,
        "_wait_for_launch_identity",
        lambda _pid, _launch_id, **_kwargs: {
            "pid": 777,
            "state": "S",
            "start_time_ticks": 2,
            "argv": captured["command"],
            "cwd": str(coordinator.ROOT),
            "process_group_id": 777,
            "session_id": 777,
            "launch_id": launch_id,
        },
    )

    process, pid = coordinator.start_or_attach(run, initializer)

    assert process is not None and pid == 777
    assert captured["environment"]["CUDA_VISIBLE_DEVICES"] == str(run.gpu)
    command = captured["command"]
    assert "--resume-checkpoint" in command
    assert command[command.index("--resume-checkpoint") + 1] == str(
        (run_dir / "last.pt").resolve()
    )
    assert "--resume-audit" in command
    launcher = json.loads((run_dir / "launcher.json").read_text(encoding="utf-8"))
    assert launcher["schema_version"] == coordinator.PHASE2_RESUME_LAUNCH_SCHEMA
    assert launcher["launch_mode"] == "exact_resume"
    assert launcher["resume"]["previous_launch_id"] == old_launcher["launch_id"]
    assert launcher["resume"]["previous_launcher_sha256"] == old_launcher_sha
    archived = (
        coordinator.AUDIT_DIR
        / "phase2_superseded_launchers"
        / f"{run.role}_{old_launcher['launch_id']}"
        / "launcher.json"
    )
    assert coordinator.sha256_file(archived) == old_launcher_sha
    assert (run_dir / "train.stdout_stderr.log").read_text(encoding="utf-8") == (
        "interrupted before successful trailer\n"
    )


def test_dead_epoch79_checkpoint_can_exact_resume_finalization_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, initializer, run_dir = _temporary_run(tmp_path)
    state = _patch_state_paths(monkeypatch, tmp_path / "state-root")
    state.mkdir(parents=True)
    coordinator.ALLOW.touch()
    _write_complete_checkpoint(run, initializer)
    (run_dir / "train.stdout_stderr.log").write_text(
        "final checkpoint committed before interrupted trailer\n", encoding="utf-8"
    )
    old_launcher = _launcher_payload(run, initializer)
    _write_json(run_dir / "launcher.json", old_launcher)
    launch_id = "d" * 32
    captured: dict[str, Any] = {}

    class FakeProcess:
        pid = 888

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    def fake_popen(command: list[str], **_kwargs: Any) -> FakeProcess:
        captured["command"] = command
        return FakeProcess()

    monkeypatch.setattr(
        coordinator,
        "_read_process_identity",
        lambda _pid: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monkeypatch.setattr(coordinator, "_process_group_alive", lambda _pgid: False)
    monkeypatch.setattr(coordinator.uuid, "uuid4", lambda: SimpleNamespace(hex=launch_id))
    monkeypatch.setattr(coordinator.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        coordinator,
        "_wait_for_launch_identity",
        lambda _pid, _launch_id, **_kwargs: {
            "pid": 888,
            "state": "S",
            "start_time_ticks": 3,
            "argv": captured["command"],
            "cwd": str(coordinator.ROOT),
            "process_group_id": 888,
            "session_id": 888,
            "launch_id": launch_id,
        },
    )

    process, pid = coordinator.start_or_attach(run, initializer)

    assert process is not None and pid == 888
    launcher = json.loads((run_dir / "launcher.json").read_text(encoding="utf-8"))
    assert launcher["launch_mode"] == "exact_resume"
    assert launcher["resume"]["source_epoch"] == 79
    assert launcher["resume"]["history_rows_before_resume"] == 80
    assert "--resume-checkpoint" in captured["command"]


def test_progress_watchdog_uses_safe_attach_baseline_and_monotonic_time() -> None:
    run = coordinator.GateRun("role", Path("/tmp/config.yaml"), 2)
    first = (3, (1, 2, 3), (4, 5, 6))
    watch = coordinator._check_progress_watchdog(
        run,
        None,
        first,
        alive=True,
        checkpoint_ready=False,
        now_monotonic=10_000.0,
        no_progress_timeout_seconds=1_200.0,
        finalizing_timeout_seconds=300.0,
    )
    assert watch.last_progress_at == 10_000.0
    watch = coordinator._check_progress_watchdog(
        run,
        watch,
        (4, (7, 8, 9), (10, 11, 12)),
        alive=True,
        checkpoint_ready=False,
        now_monotonic=11_199.0,
        no_progress_timeout_seconds=1_200.0,
        finalizing_timeout_seconds=300.0,
    )
    assert watch.last_progress_at == 11_199.0
    with pytest.raises(RuntimeError, match="no committed history/checkpoint progress"):
        coordinator._check_progress_watchdog(
            run,
            watch,
            watch.fingerprint,
            alive=True,
            checkpoint_ready=False,
            now_monotonic=12_399.0,
            no_progress_timeout_seconds=1_200.0,
            finalizing_timeout_seconds=300.0,
        )


def test_progress_watchdog_bounds_finalizing_from_first_observation() -> None:
    run = coordinator.GateRun("role", Path("/tmp/config.yaml"), 3)
    fingerprint = (80, (1, 2, 3), (4, 5, 6))
    watch = coordinator._check_progress_watchdog(
        run,
        None,
        fingerprint,
        alive=True,
        checkpoint_ready=True,
        now_monotonic=50_000.0,
        no_progress_timeout_seconds=1_200.0,
        finalizing_timeout_seconds=300.0,
    )
    assert watch.finalizing_since == 50_000.0
    with pytest.raises(RuntimeError, match="finalizing exceeded"):
        coordinator._check_progress_watchdog(
            run,
            watch,
            fingerprint,
            alive=True,
            checkpoint_ready=True,
            now_monotonic=50_300.0,
            no_progress_timeout_seconds=1_200.0,
            finalizing_timeout_seconds=300.0,
        )


def test_main_invalid_start_state_restores_hold_and_removes_owned_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _patch_state_paths(monkeypatch, tmp_path)
    state.mkdir(parents=True)
    coordinator.ALLOW.touch()
    with pytest.raises(RuntimeError, match="fail-closed"):
        coordinator.main()
    assert coordinator.HOLD.is_file()
    assert not coordinator.ALLOW.exists()
    assert not coordinator.PID_PATH.exists()


def test_main_success_closes_gate_before_reporting_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _patch_state_paths(monkeypatch, tmp_path)
    state.mkdir(parents=True)
    coordinator.HOLD.touch()
    initializer = tmp_path / "initializer.pt"
    initializer.write_bytes(b"initializer")
    monkeypatch.setattr(coordinator, "wait_for_baselines", lambda: None)
    monkeypatch.setattr(coordinator, "finalize_baselines", lambda: {})
    monkeypatch.setattr(coordinator, "extract_initializers", lambda: initializer)
    monkeypatch.setattr(
        coordinator, "recover_uncommitted_phase2_attempts", lambda _path: []
    )

    def fake_preflight(_initializer: Path) -> dict[str, Any]:
        coordinator.unlock_gate()
        return {}

    monkeypatch.setattr(coordinator, "preflight_and_unlock", fake_preflight)
    monkeypatch.setattr(coordinator, "run_round", lambda *_args: {})
    monkeypatch.setattr(coordinator, "write_gate_sha_manifest", lambda _reports: None)
    assert coordinator.main() == 0
    assert coordinator.HOLD.is_file()
    assert not coordinator.ALLOW.exists()
    assert not coordinator.PID_PATH.exists()
    status = json.loads(
        (coordinator.AUDIT_DIR / "phase2_status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "completed"
    assert status["gate_state"] == "HOLD"
