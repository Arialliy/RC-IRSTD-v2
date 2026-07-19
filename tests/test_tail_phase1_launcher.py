from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import wait_and_launch_tail_phase1 as launcher


def _attachment(pid: int, gpu_index: int = 0) -> dict[str, object]:
    return {
        "gpu_index": gpu_index,
        "gpu_uuid": f"GPU-{gpu_index}",
        "pid": pid,
        "process_name": "python",
        "used_gpu_memory_mib": "100",
        "command": f"python train-{pid}.py",
        "command_status": "observed",
    }


def test_gpu_compute_processes_filters_to_physical_0_1_2_and_records_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries: list[str] = []

    def fake_query(query: str) -> list[list[str]]:
        queries.append(query)
        if query == "--query-gpu=index,uuid":
            return [["0", "GPU-A"], ["1", "GPU-B"], ["2", "GPU-C"], ["3", "GPU-D"]]
        return [
            ["GPU-A", "71", "/python", "120"],
            ["GPU-C", "71", "/python", "110"],
            ["GPU-D", "99", "/unrelated", "900"],
        ]

    command_reads: list[int] = []

    def fake_command(pid: int, proc_root: Path = Path("/proc")) -> tuple[str, str]:
        command_reads.append(pid)
        return f"/python train.py --pid {pid}", "observed"

    monkeypatch.setattr(launcher, "_nvidia_query", fake_query)
    monkeypatch.setattr(launcher, "_read_process_command", fake_command)

    records = launcher._gpu_compute_processes()

    assert queries == [
        "--query-gpu=index,uuid",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
    ]
    assert [(record["gpu_index"], record["pid"]) for record in records] == [
        (0, 71),
        (2, 71),
    ]
    assert command_reads == [71]
    assert all(record["command"] == "/python train.py --pid 71" for record in records)


def test_wait_for_gpu_clear_resets_on_any_pid_and_confirms_after_verification() -> None:
    occupied_a = (_attachment(101, 0),)
    occupied_b = (_attachment(202, 2),)
    occupied_race = (_attachment(303, 1),)
    sequence = iter(
        [
            occupied_a,
            (),
            occupied_b,
            (),
            (),
            (),
            occupied_race,
            (),
            (),
            (),
            (),
        ]
    )
    observations: list[tuple[list[int], int, str]] = []
    sleeps: list[float] = []
    verifies: list[str] = []

    launcher._wait_for_gpu_clear(
        required_clear_observations=3,
        poll_seconds=7.0,
        query=lambda: next(sequence),
        observe=lambda records, count, phase: observations.append(
            ([int(record["pid"]) for record in records], count, phase)
        ),
        verify_ready=lambda: verifies.append("verified"),
        sleeper=sleeps.append,
    )

    assert verifies == ["verified", "verified"]
    assert ([303], 0, "post_verify_confirmation") in observations
    assert observations[-1] == ([], 3, "post_verify_confirmation")
    assert all(duration == 7.0 for duration in sleeps)
    # A busy sample at any point, including the post-verification race check,
    # forces the spaced-clear counter back to zero.
    assert ([101], 0, "spaced_poll") in observations
    assert ([202], 0, "spaced_poll") in observations


def test_acquire_reclaims_dead_legacy_lock_and_release_matches_nonce(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "launcher.lock"
    guard_path = tmp_path / "launcher.lock.guard"
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    lock_path.write_text("99999999\n", encoding="utf-8")

    owned, stale = launcher._acquire_launcher_lock(
        lock_path=lock_path,
        guard_path=guard_path,
        proc_root=proc_root,
    )

    assert stale == {
        "lock_format": "legacy_pid_v0",
        "pid": 99999999,
        "process_state_at_takeover": "absent_from_proc",
    }
    published = json.loads(lock_path.read_text(encoding="utf-8"))
    assert published["pid"] == os.getpid()
    assert published["nonce"] == owned["nonce"]
    assert launcher._release_launcher_lock(owned, lock_path, guard_path) is True
    assert not lock_path.exists()


def test_acquire_never_takes_over_lock_whose_pid_is_live(tmp_path: Path) -> None:
    lock_path = tmp_path / "launcher.lock"
    guard_path = tmp_path / "launcher.lock.guard"
    lock_path.write_text(f"{os.getpid()}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="live PID"):
        launcher._acquire_launcher_lock(
            lock_path=lock_path,
            guard_path=guard_path,
        )

    assert lock_path.read_text(encoding="utf-8") == f"{os.getpid()}\n"


def test_acquire_reclaims_json_lock_only_when_start_time_proves_pid_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "launcher.lock"
    guard_path = tmp_path / "launcher.lock.guard"
    lock_path.write_text(
        json.dumps({"pid": 12345, "nonce": "old", "proc_start_time_ticks": 100})
        + "\n",
        encoding="utf-8",
    )

    def fake_process(pid: int, proc_root: Path = Path("/proc")) -> dict[str, object]:
        return {
            "pid": pid,
            "command": "unrelated live replacement",
            "command_status": "observed",
            "proc_start_time_ticks": 200 if pid == 12345 else 300,
        }

    monkeypatch.setattr(launcher, "_proc_process_record", fake_process)
    owned, stale = launcher._acquire_launcher_lock(lock_path, guard_path)

    assert stale is not None
    assert stale["process_state_at_takeover"] == (
        "pid_reused_with_different_start_time"
    )
    assert stale["observed_replacement_process"]["proc_start_time_ticks"] == 200
    assert launcher._release_launcher_lock(owned, lock_path, guard_path) is True


def test_read_lock_owner_rejects_unsafe_json_pid(tmp_path: Path) -> None:
    lock_path = tmp_path / "launcher.lock"
    lock_path.write_text('{"pid": 1, "nonce": "invalid"}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="unsafe PID"):
        launcher._read_lock_owner(lock_path)


def test_release_refuses_a_replaced_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "launcher.lock"
    guard_path = tmp_path / "launcher.lock.guard"
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    owned, _ = launcher._acquire_launcher_lock(lock_path, guard_path, proc_root)
    replacement = {**owned, "nonce": "replacement-owner"}
    lock_path.write_text(json.dumps(replacement) + "\n", encoding="utf-8")

    assert launcher._release_launcher_lock(owned, lock_path, guard_path) is False
    assert json.loads(lock_path.read_text(encoding="utf-8"))["nonce"] == "replacement-owner"


def test_verify_snapshot_rejects_same_size_content_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = tmp_path / "frozen.py"
    frozen.write_text("alpha\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "_training_code_paths", lambda: (frozen,))
    expected = launcher._snapshot((frozen,))

    frozen.write_text("omega\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="drifted while waiting"):
        launcher._verify_snapshot(expected)


def test_parser_defaults_to_dynamic_any_compute_gate() -> None:
    args = launcher.build_parser().parse_args([])
    assert args.wait_pid is None
    assert args.clear_observations == 3
    assert args.poll_seconds == 30.0


def test_run_preserves_virtual_environment_python_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_python = tmp_path / "system-python"
    real_python.write_bytes(b"binary")
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(real_python)
    observed: list[Path] = []

    def fake_prerequisites(python_bin: Path) -> dict[str, dict[str, object]]:
        observed.append(python_bin)
        return {}

    monkeypatch.setattr(launcher, "_validate_prerequisites", fake_prerequisites)
    args = launcher.build_parser().parse_args(
        ["--python", str(venv_python), "--dry-run"]
    )

    assert launcher._run(args) == 0
    assert observed == [venv_python]
    assert observed[0] != real_python
