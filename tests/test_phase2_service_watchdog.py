from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import pytest
import torch

from scripts import watch_phase2_rc_mshnet_service as watchdog


SERVICE = watchdog.ServiceState(
    active_state="active",
    sub_state="running",
    main_pid=123,
    control_group="/test/phase2.service",
)


def _run(
    role: str,
    *,
    committed: int,
    complete: bool = False,
    live: bool = True,
    launcher: bool = True,
    wall_time: float = 1_000.0,
    pending: str | None = None,
) -> watchdog.RunObservation:
    checkpoint_epoch = committed - 1 if committed else None
    return watchdog.RunObservation(
        role=role,
        physical_gpu=2 if role == "left" else 3,
        committed_epochs=committed,
        history_rows=committed,
        checkpoint_epoch=checkpoint_epoch,
        complete=complete,
        launcher_present=launcher,
        launcher_live=live,
        artifacts_present=launcher or committed > 0,
        last_commit_wall_time=wall_time if committed else None,
        launch_wall_time=wall_time if launcher else None,
        pending_evidence=pending,
        pending_since_wall_time=wall_time if pending else None,
    )


def _open_snapshot(*runs: watchdog.RunObservation) -> watchdog.Snapshot:
    return watchdog.Snapshot(
        gate_state="ALLOW",
        service=SERVICE,
        current_round="round1",
        runs=tuple(runs),
        all_rounds_complete=False,
        phase_status=None,
    )


def _hold_snapshot() -> watchdog.Snapshot:
    return watchdog.Snapshot(
        gate_state="HOLD",
        service=None,
        current_round=None,
        runs=(),
        all_rounds_complete=False,
        phase_status=None,
    )


class _Clock:
    def __init__(self) -> None:
        self.monotonic_value = 100.0
        self.wall_value = 1_000.0

    def monotonic(self) -> float:
        return self.monotonic_value

    def wall(self) -> float:
        return self.wall_value

    def sleep(self, seconds: float) -> None:
        self.monotonic_value += seconds
        self.wall_value += seconds


def _monitor(
    provider: Callable[[], watchdog.Snapshot],
    restart: Callable[[], None],
    *,
    state: watchdog.Phase2Watchdog,
    clock: _Clock,
    poll_seconds: float,
) -> str:
    return watchdog.monitor(
        provider,
        restart,
        watchdog=state,
        poll_seconds=poll_seconds,
        monotonic_clock=clock.monotonic,
        wall_clock=clock.wall,
        sleeper=clock.sleep,
        emit=lambda _line: None,
    )


def test_hold_exits_cleanly_without_querying_or_restarting() -> None:
    restart_calls: list[bool] = []
    result = _monitor(
        _hold_snapshot,
        lambda: restart_calls.append(True),
        state=watchdog.Phase2Watchdog(
            stall_seconds=20,
            finalizing_seconds=10,
            evidence_grace_seconds=5,
        ),
        clock=_Clock(),
        poll_seconds=5,
    )
    assert result == "safe_exit"
    assert restart_calls == []


def test_collect_snapshot_on_hold_never_queries_systemd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = tmp_path / "outputs/phase_state"
    state.mkdir(parents=True)
    (state / watchdog.HOLD_NAME).touch()
    monkeypatch.setattr(
        watchdog,
        "_probe_service",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("HOLD must not query systemd")
        ),
    )
    snapshot = watchdog.collect_snapshot(tmp_path, rounds=())
    assert snapshot.gate_state == "HOLD"
    assert snapshot.service is None


def test_twenty_minute_equivalent_no_commit_requests_exactly_one_restart() -> None:
    clock = _Clock()
    snapshot = _open_snapshot(
        _run("left", committed=4),
        _run("right", committed=3),
    )
    restart_calls: list[float] = []
    result = _monitor(
        lambda: snapshot,
        lambda: restart_calls.append(clock.monotonic()),
        state=watchdog.Phase2Watchdog(
            stall_seconds=20,
            finalizing_seconds=10,
            evidence_grace_seconds=5,
        ),
        clock=clock,
        poll_seconds=5,
    )
    assert result == "restart_requested"
    assert restart_calls == [120.0]


def test_progress_of_one_role_does_not_mask_other_role_stall() -> None:
    clock = _Clock()

    def provider() -> watchdog.Snapshot:
        left_progress = 1 + int((clock.monotonic() - 100.0) // 5)
        return _open_snapshot(
            _run(
                "left",
                committed=left_progress,
                wall_time=clock.wall(),
            ),
            _run("right", committed=1),
        )

    restart_calls: list[float] = []
    result = _monitor(
        provider,
        lambda: restart_calls.append(clock.monotonic()),
        state=watchdog.Phase2Watchdog(
            stall_seconds=20,
            finalizing_seconds=10,
            evidence_grace_seconds=5,
        ),
        clock=clock,
        poll_seconds=5,
    )
    assert result == "restart_requested"
    assert restart_calls == [120.0]


def test_five_minute_equivalent_finalizing_timeout_requests_restart() -> None:
    clock = _Clock()
    snapshot = _open_snapshot(
        _run("left", committed=80, complete=True, live=True),
        _run("right", committed=79),
    )
    restart_calls: list[float] = []
    result = _monitor(
        lambda: snapshot,
        lambda: restart_calls.append(clock.monotonic()),
        state=watchdog.Phase2Watchdog(
            stall_seconds=100,
            finalizing_seconds=10,
            evidence_grace_seconds=5,
        ),
        clock=clock,
        poll_seconds=5,
    )
    assert result == "restart_requested"
    assert restart_calls == [110.0]


def test_stable_evidence_mismatch_fails_closed_without_restart() -> None:
    clock = _Clock()
    snapshot = _open_snapshot(
        _run(
            "left",
            committed=4,
            pending="history_is_one_row_ahead_of_checkpoint",
        ),
        _run("right", committed=4),
    )
    restart_calls: list[bool] = []
    with pytest.raises(watchdog.EvidenceError, match="stable inconsistent evidence"):
        _monitor(
            lambda: snapshot,
            lambda: restart_calls.append(True),
            state=watchdog.Phase2Watchdog(
                stall_seconds=20,
                finalizing_seconds=10,
                evidence_grace_seconds=5,
            ),
            clock=clock,
            poll_seconds=5,
        )
    assert restart_calls == []


def test_dead_incomplete_launcher_fails_closed_without_restart() -> None:
    state = watchdog.Phase2Watchdog(
        stall_seconds=20,
        finalizing_seconds=10,
        evidence_grace_seconds=5,
    )
    with pytest.raises(watchdog.EvidenceError, match="launcher is dead"):
        state.step(
            _open_snapshot(
                _run("left", committed=4, live=False),
                _run("right", committed=4),
            ),
            monotonic_now=100.0,
            wall_now=1_000.0,
        )


def test_committed_progress_resets_timer_then_hold_exits() -> None:
    clock = _Clock()
    snapshots = iter(
        (
            _open_snapshot(_run("left", committed=1), _run("right", committed=1)),
            _open_snapshot(
                _run("left", committed=2, wall_time=1_009.0),
                _run("right", committed=2, wall_time=1_009.0),
            ),
            _hold_snapshot(),
        )
    )
    restart_calls: list[bool] = []
    result = _monitor(
        lambda: next(snapshots),
        lambda: restart_calls.append(True),
        state=watchdog.Phase2Watchdog(
            stall_seconds=10,
            finalizing_seconds=5,
            evidence_grace_seconds=3,
        ),
        clock=clock,
        poll_seconds=9,
    )
    assert result == "safe_exit"
    assert restart_calls == []


def test_restart_callback_uses_only_the_expected_systemctl_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], bool]] = []

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        calls.append((command, check))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(watchdog.subprocess, "run", fake_run)
    watchdog.restart_service("test-phase2.service")
    assert calls == [
        (["systemctl", "--user", "restart", "test-phase2.service"], True)
    ]


def test_live_launcher_with_wrong_physical_gpu_binding_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    config_path = root / "configs/phase2.yaml"
    output_dir = root / "outputs/run"
    initializer = root / watchdog.EXPECTED_INITIALIZER_RELATIVE
    config_path.parent.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    initializer.parent.mkdir(parents=True)
    initializer.write_bytes(b"initializer")
    config_path.write_text(
        f"output_dir: {output_dir}\ntraining:\n  epochs: 80\n",
        encoding="utf-8",
    )
    spec = watchdog.RunSpec("left", "configs/phase2.yaml", 2)
    cache = watchdog.FileHashCache()
    command = [
        sys.executable,
        "-m",
        "rc_irstd.cli.train_detector",
        "--config",
        str(config_path),
        "--set",
        "device=cuda:0",
        "--set",
        f"training.initialize_from={initializer}",
        "--set",
        f"output_dir={output_dir}",
    ]
    launch_id = "a" * 32
    identity = {
        "pid": 456,
        "state": "S",
        "start_time_ticks": 99,
        "argv": command,
        "cwd": str(root),
        "process_group_id": 456,
        "session_id": 456,
        "launch_id": launch_id,
    }
    launcher = {
        "schema_version": watchdog.LAUNCH_SCHEMA,
        "role": "left",
        "pid": 456,
        "launch_id": launch_id,
        "physical_gpu": 2,
        "visible_device": "cuda:0",
        "command": command,
        "config_path": str(config_path),
        "config_sha256": cache.sha256(config_path),
        "initializer_path": str(initializer),
        "initializer_sha256": cache.sha256(initializer),
        "expected_output_dir": str(output_dir),
        "process_identity": identity,
        "launched_at": "2026-07-16T00:00:00+08:00",
    }
    (output_dir / "launcher.json").write_text(
        json.dumps(launcher), encoding="utf-8"
    )
    current = {
        **identity,
        "cuda_visible_devices": "3",
        "control_group": SERVICE.control_group,
    }
    monkeypatch.setattr(watchdog, "_read_process_identity", lambda _pid: current)
    with pytest.raises(watchdog.EvidenceError, match="CUDA_VISIBLE_DEVICES"):
        watchdog._observe_run(
            root,
            spec,
            service=SERVICE,
            hash_cache=cache,
        )


def test_launcher_command_rejects_duplicate_overrides(tmp_path: Path) -> None:
    config = tmp_path / "phase2.yaml"
    output = tmp_path / "run"
    initializer = tmp_path / "initializer.pt"
    command = [
        sys.executable,
        "-m",
        "rc_irstd.cli.train_detector",
        "--config",
        str(config),
        "--set",
        "device=cuda:0",
        "--set",
        "device=cuda:1",
        "--set",
        f"training.initialize_from={initializer}",
        "--set",
        f"output_dir={output}",
    ]
    with pytest.raises(watchdog.EvidenceError, match="duplicates override device"):
        watchdog._validate_command(
            command,
            config_path=config,
            output_dir=output,
            initializer=initializer,
        )


def test_process_identity_accepts_normal_zombie_reap_window() -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        deadline = time.monotonic() + 5.0
        state = ""
        while time.monotonic() < deadline:
            stat_text = (Path("/proc") / str(child.pid) / "stat").read_text(
                encoding="utf-8"
            )
            state = stat_text[stat_text.rfind(")") + 2 :].split()[0]
            if state == "Z":
                break
            time.sleep(0.01)
        assert state == "Z"
        assert watchdog._read_process_identity(child.pid) == {
            "pid": child.pid,
            "state": "Z",
        }
    finally:
        child.wait(timeout=5)


def test_history_and_checkpoint_reader_accept_only_an_aligned_formal_commit(
    tmp_path: Path,
) -> None:
    history = tmp_path / "history.csv"
    with history.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "lr", "loss_total"])
        writer.writeheader()
        writer.writerow({"epoch": 0, "lr": 0.001, "loss_total": 1.0})
    checkpoint = tmp_path / "last.pt"
    torch.save(
        {
            "format_version": 2,
            "kind": "detector",
            "epoch": 0,
            "config": {"training": {"epochs": 80, "resume": None}},
        },
        checkpoint,
    )
    assert watchdog._read_history(history)[0] == 1
    assert watchdog._read_checkpoint_epoch(checkpoint)[0] == 0

    with history.open("a", encoding="utf-8", newline="") as handle:
        handle.write("3,0.001,1.0\n")
    with pytest.raises(watchdog.EvidenceError, match="not exactly contiguous"):
        watchdog._read_history(history)


def test_timeout_defaults_are_twenty_and_five_minutes() -> None:
    assert watchdog.DEFAULT_STALL_SECONDS == 20 * 60
    assert watchdog.DEFAULT_FINALIZING_SECONDS == 5 * 60
