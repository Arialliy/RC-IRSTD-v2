"""Wait for the foreign three-GPU job, then launch frozen source-only phase 1.

This is deliberately narrow: it launches exactly two preregistered Tail/Miss
inner detectors and the preregistered Candidate-A full-source detector.  It
never signals the foreign process, refuses any code/config drift, waits until
all three GPUs have no compute process, and atomically records its state.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import tempfile
import time
from typing import Any, Callable, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
PREREGISTRATION = (
    ROOT
    / "outputs"
    / "v4_source_only"
    / "preregistration"
    / "detector_tail_branches_seed42.json"
)
PREREGISTRATION_SHA256 = (
    "ac65d2d60b90a308c687aacb23d8612a57406134b1782e15d5dcc910798756be"
)
STATUS_PATH = PREREGISTRATION.with_name("tail_phase1_launcher_status.json")
LOCK_PATH = PREREGISTRATION.with_name("tail_phase1_launcher.lock")
LOCK_GUARD_PATH = PREREGISTRATION.with_name("tail_phase1_launcher.lock.guard")
LOG_DIR = PREREGISTRATION.with_name("tail_phase1_logs")
TARGET_GPU_INDICES = (0, 1, 2)
VISIBLE_GPUS = ",".join(str(index) for index in TARGET_GPU_INDICES)

RUNS = (
    {
        "run_id": "tailmiss_irstd_only_inner",
        "config": ROOT
        / "configs/stage4_inner_nudt_from_irstd_tailmiss_20ep.yaml",
        "output_dir": ROOT / "outputs/stage4_inner_nudt_from_irstd_tailmiss_20ep",
        "source_names": ["IRSTD-1K"],
    },
    {
        "run_id": "tailmiss_nudt_only_inner",
        "config": ROOT
        / "configs/stage4_inner_irstd_from_nudt_tailmiss_20ep.yaml",
        "output_dir": ROOT / "outputs/stage4_inner_irstd_from_nudt_tailmiss_20ep",
        "source_names": ["NUDT-SIRST"],
    },
    {
        "run_id": "tailrank_a_full_sources",
        "config": ROOT / "configs/stage4_full_sources_tailrank_margin_a_20ep.yaml",
        "output_dir": ROOT / "outputs/stage4_full_sources_tailrank_margin_a_20ep",
        "source_names": ["NUDT-SIRST", "IRSTD-1K"],
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _training_code_paths() -> tuple[Path, ...]:
    paths: set[Path] = {Path(__file__).resolve()}
    for relative in ("rc_irstd", "data_ext"):
        paths.update(path.resolve() for path in (ROOT / relative).rglob("*.py"))
    for relative in (
        "evaluation/artifact_integrity.py",
        "losses/local_peak_cvar.py",
        "model/MSHNet.py",
        "utils/data.py",
    ):
        paths.add((ROOT / relative).resolve())
    paths.add(PREREGISTRATION.resolve())
    paths.update(Path(run["config"]).resolve() for run in RUNS)
    result = tuple(sorted(paths))
    if not result or any(not path.is_file() for path in result):
        raise FileNotFoundError("A frozen phase-1 code/config input is missing")
    return result


def _snapshot(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(ROOT)): {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    }


def _verify_snapshot(expected: Mapping[str, Mapping[str, Any]]) -> None:
    current_paths = _training_code_paths()
    current_names = {str(path.relative_to(ROOT)) for path in current_paths}
    if current_names != set(expected):
        raise RuntimeError("Frozen phase-1 code/config file set changed while waiting")
    for relative, record in expected.items():
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Frozen phase-1 input disappeared: {path}")
        if path.stat().st_size != record["size_bytes"] or _sha256(path) != record[
            "sha256"
        ]:
            raise RuntimeError(f"Frozen phase-1 input drifted while waiting: {path}")


def _read_process_command(pid: int, proc_root: Path = Path("/proc")) -> tuple[str, str]:
    command_path = proc_root / str(pid) / "cmdline"
    try:
        raw = command_path.read_bytes()
    except FileNotFoundError:
        return "", "exited_before_cmdline_read"
    except PermissionError:
        return "", "permission_denied"
    command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    return command, "observed" if command else "empty"


def _nvidia_query(query: str) -> list[list[str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            query,
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    rows: list[list[str]] = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        values = [value.strip() for value in row]
        if not values or not any(values):
            continue
        if values[0].casefold().startswith("no running"):
            continue
        rows.append(values)
    return rows


def _gpu_compute_processes(
    gpu_indices: Sequence[int] = TARGET_GPU_INDICES,
) -> tuple[dict[str, Any], ...]:
    """Return every compute-app attachment on the requested physical GPUs.

    A data-parallel process legitimately appears once per attached GPU.  Those
    rows are retained so the status evidence shows exactly which of GPU0/1/2
    was occupied; callers can additionally inspect the unique PID set.
    """

    index_rows = _nvidia_query("--query-gpu=index,uuid")
    uuid_to_index: dict[str, int] = {}
    for row in index_rows:
        if len(row) != 2:
            raise RuntimeError(f"Unexpected nvidia-smi GPU row: {row!r}")
        try:
            index = int(row[0])
        except ValueError as error:
            raise RuntimeError(f"Unexpected nvidia-smi GPU index: {row[0]!r}") from error
        uuid_to_index[row[1]] = index

    requested = set(gpu_indices)
    missing = requested.difference(uuid_to_index.values())
    if missing:
        raise RuntimeError(f"Requested physical GPUs are absent: {sorted(missing)}")

    app_rows = _nvidia_query(
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"
    )
    command_cache: dict[int, tuple[str, str]] = {}
    records: list[dict[str, Any]] = []
    for row in app_rows:
        if len(row) != 4:
            raise RuntimeError(f"Unexpected nvidia-smi compute-app row: {row!r}")
        gpu_uuid, raw_pid, process_name, used_memory = row
        if gpu_uuid not in uuid_to_index:
            raise RuntimeError(f"Compute app reported an unknown GPU UUID: {gpu_uuid}")
        gpu_index = uuid_to_index[gpu_uuid]
        if gpu_index not in requested:
            continue
        try:
            pid = int(raw_pid)
        except ValueError as error:
            raise RuntimeError(f"Unexpected nvidia-smi compute PID: {raw_pid!r}") from error
        if pid not in command_cache:
            command_cache[pid] = _read_process_command(pid)
        command, command_status = command_cache[pid]
        records.append(
            {
                "gpu_index": gpu_index,
                "gpu_uuid": gpu_uuid,
                "pid": pid,
                "process_name": process_name,
                "used_gpu_memory_mib": used_memory,
                "command": command,
                "command_status": command_status,
            }
        )
    return tuple(sorted(records, key=lambda item: (item["gpu_index"], item["pid"])))


def _proc_process_record(
    pid: int, proc_root: Path = Path("/proc")
) -> dict[str, Any] | None:
    """Inspect a PID without issuing signal 0 (or any other signal)."""

    process_dir = proc_root / str(pid)
    try:
        stat_raw = (process_dir / "stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except PermissionError:
        stat_raw = ""
    command, command_status = _read_process_command(pid, proc_root)
    start_time_ticks: int | None = None
    if stat_raw:
        close_parenthesis = stat_raw.rfind(")")
        remaining = stat_raw[close_parenthesis + 1 :].split()
        if close_parenthesis >= 0 and len(remaining) > 19:
            try:
                start_time_ticks = int(remaining[19])
            except ValueError:
                pass
    return {
        "pid": pid,
        "command": command,
        "command_status": command_status,
        "proc_start_time_ticks": start_time_ticks,
    }


def _read_lock_owner(lock_path: Path) -> dict[str, Any]:
    raw = lock_path.read_text(encoding="utf-8").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and type(payload.get("pid")) is int:
        if payload["pid"] <= 1:
            raise RuntimeError(
                f"Launcher lock has an unsafe PID; refusing takeover: {lock_path}"
            )
        return {"lock_format": "json_v2", **payload}
    try:
        pid = int(raw)
    except ValueError as error:
        raise RuntimeError(
            f"Launcher lock is corrupt; refusing unsafe takeover: {lock_path}"
        ) from error
    if pid <= 1:
        raise RuntimeError(
            f"Launcher lock has an unsafe PID; refusing takeover: {lock_path}"
        )
    return {"lock_format": "legacy_pid_v0", "pid": pid}


@contextmanager
def _lock_guard(guard_path: Path) -> Iterator[None]:
    guard_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(guard_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _link_complete_lock(lock_path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a fully written lock atomically; never leave a partial new lock."""

    raw = (
        json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=lock_path.parent, prefix=f".{lock_path.name}.", delete=False
        ) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, lock_path)
        directory_fd = os.open(lock_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _acquire_launcher_lock(
    lock_path: Path = LOCK_PATH,
    guard_path: Path = LOCK_GUARD_PATH,
    proc_root: Path = Path("/proc"),
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Acquire the project launcher lock, reclaiming only a provably dead PID."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stale_owner: dict[str, Any] | None = None
    with _lock_guard(guard_path):
        if lock_path.exists():
            owner = _read_lock_owner(lock_path)
            live_process = _proc_process_record(owner["pid"], proc_root)
            if live_process is not None:
                recorded_start = owner.get("proc_start_time_ticks")
                observed_start = live_process.get("proc_start_time_ticks")
                pid_was_reused = (
                    type(recorded_start) is int
                    and type(observed_start) is int
                    and recorded_start != observed_start
                )
                if not pid_was_reused:
                    raise FileExistsError(
                        "Phase-1 launcher lock belongs to a live PID; no takeover was "
                        f"attempted: owner={owner!r}, process={live_process!r}"
                    )
                stale_owner = {
                    **owner,
                    "process_state_at_takeover": "pid_reused_with_different_start_time",
                    "observed_replacement_process": live_process,
                }
            else:
                stale_owner = {
                    **owner,
                    "process_state_at_takeover": "absent_from_proc",
                }
            lock_path.unlink()

        current_process = _proc_process_record(os.getpid(), proc_root)
        record = {
            "schema_version": "rc-v4-tail-phase1-launcher-lock-v2",
            "pid": os.getpid(),
            "nonce": secrets.token_hex(16),
            "created_utc": _utc_now(),
            "project_root": str(ROOT),
            "script_path": str(Path(__file__).resolve()),
            "proc_start_time_ticks": (
                None
                if current_process is None
                else current_process.get("proc_start_time_ticks")
            ),
        }
        _link_complete_lock(lock_path, record)
    return record, stale_owner


def _release_launcher_lock(
    owned_record: Mapping[str, Any],
    lock_path: Path = LOCK_PATH,
    guard_path: Path = LOCK_GUARD_PATH,
) -> bool:
    """Release only the exact nonce acquired by this launcher."""

    with _lock_guard(guard_path):
        if not lock_path.exists():
            return False
        try:
            current = _read_lock_owner(lock_path)
        except (OSError, RuntimeError):
            # A replacement or corrupt lock is never ours to remove.  Refusing
            # release is safer than masking an earlier launcher failure.
            return False
        if (
            current.get("pid") != owned_record.get("pid")
            or current.get("nonce") != owned_record.get("nonce")
        ):
            return False
        lock_path.unlink()
        return True


def _wait_for_gpu_clear(
    *,
    required_clear_observations: int,
    poll_seconds: float,
    observe: Callable[[Sequence[Mapping[str, Any]], int, str], None],
    verify_ready: Callable[[], None],
    query: Callable[[], Sequence[Mapping[str, Any]]] = _gpu_compute_processes,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    """Wait for spaced clear samples and one post-verification clear sample."""

    clear_observations = 0
    while True:
        processes = tuple(query())
        clear_observations = clear_observations + 1 if not processes else 0
        observe(processes, clear_observations, "spaced_poll")
        if clear_observations >= required_clear_observations:
            verify_ready()
            confirmation = tuple(query())
            if confirmation:
                clear_observations = 0
                observe(confirmation, clear_observations, "post_verify_confirmation")
            else:
                observe(confirmation, clear_observations, "post_verify_confirmation")
                return
        sleeper(poll_seconds)


def _validate_prerequisites(python_bin: Path) -> dict[str, dict[str, Any]]:
    if ROOT != Path("/home/ly/RC-IRSTD-v2"):
        raise RuntimeError(f"Unexpected project root: {ROOT}")
    if not python_bin.is_file():
        raise FileNotFoundError(python_bin)
    if _sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("Detector-tail preregistration bytes have drifted")
    _assert_outputs_absent()
    return _snapshot(_training_code_paths())


def _assert_outputs_absent() -> None:
    for run in RUNS:
        if Path(run["output_dir"]).exists():
            raise FileExistsError(
                f"Refusing to overwrite an existing phase-1 run: {run['output_dir']}"
            )


def _validate_checkpoint(path: Path, expected_sources: Sequence[str]) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    exact = {
        "kind": "detector",
        "format_version": 2,
        "epoch": 19,
        "checkpoint_selection": "fixed_last",
        "selection_rule": "fixed_last",
        "test_labels_used_for_selection": False,
        "diagnostic_test_eval": False,
        "diagnostic_only": False,
        "formal_paper_checkpoint": True,
        "warm_flag": True,
        "inference_head": "multi_scale_fused",
    }
    for field, expected in exact.items():
        if type(checkpoint.get(field)) is not type(expected) or checkpoint.get(
            field
        ) != expected:
            raise RuntimeError(f"Completed checkpoint violates {field}: {path}")
    if checkpoint.get("source_names") != list(expected_sources):
        raise RuntimeError(f"Completed checkpoint source_names mismatch: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "source_names": list(expected_sources),
        "epoch": 19,
        "selection_rule": "fixed_last",
    }


def _terminate_owned(children: Mapping[str, subprocess.Popen[bytes]]) -> None:
    """Stop only subprocesses created by this launcher after a sibling failure."""

    for process in children.values():
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline and any(
        process.poll() is None for process in children.values()
    ):
        time.sleep(0.5)
    for process in children.values():
        if process.poll() is None:
            process.kill()


def _run(args: argparse.Namespace) -> int:
    # Keep the virtual-environment entry point itself.  Resolving this symlink
    # to /usr/bin/python would discard pyvenv.cfg discovery and launch the
    # system interpreter without the experiment dependencies (notably torch).
    python_bin = Path(os.path.abspath(os.path.expanduser(args.python)))
    entry_snapshot = _validate_prerequisites(python_bin)
    snapshot_hash = _canonical_sha256(entry_snapshot)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "preregistration_sha256": PREREGISTRATION_SHA256,
                    "frozen_file_count": len(entry_snapshot),
                    "frozen_tree_sha256": snapshot_hash,
                    "run_ids": [str(run["run_id"]) for run in RUNS],
                },
                sort_keys=True,
            )
        )
        return 0

    lock_record, stale_lock_owner = _acquire_launcher_lock()
    started = _utc_now()
    base_status: dict[str, Any] = {
        "schema_version": "rc-v4-tail-phase1-launcher-status-v2",
        "launcher_pid": os.getpid(),
        "started_utc": started,
        "lock_record": lock_record,
        "stale_lock_reclaimed": stale_lock_owner is not None,
        "stale_lock_owner": stale_lock_owner,
        "legacy_wait_pid_hint_ignored_for_gating": args.wait_pid,
        "gpu_wait_policy": {
            "physical_gpu_indices": list(TARGET_GPU_INDICES),
            "block_on_any_compute_process": True,
            "required_consecutive_clear_observations": args.clear_observations,
            "poll_seconds": args.poll_seconds,
            "post_snapshot_clear_confirmation": True,
            "pid_specific_gating": False,
        },
        "foreign_process_signals_sent": False,
        "external_process_signals_sent": False,
        "visible_gpus": list(TARGET_GPU_INDICES),
        "preregistration_path": str(PREREGISTRATION),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "entry_frozen_file_count": len(entry_snapshot),
        "entry_frozen_tree_sha256": snapshot_hash,
        "entry_snapshot": entry_snapshot,
        "commands": {},
        "gpu_compute_observation_count": 0,
        "gpu_compute_process_registry": {},
    }
    children: dict[str, subprocess.Popen[bytes]] = {}
    log_handles: list[Any] = []
    try:
        base_status.update(
            {
                "state": "waiting_for_gpu_compute_quiescence",
                "updated_utc": _utc_now(),
            }
        )
        _atomic_json(STATUS_PATH, base_status)

        def record_gpu_observation(
            processes: Sequence[Mapping[str, Any]],
            clear_observations: int,
            observation_phase: str,
        ) -> None:
            observed_utc = _utc_now()
            unique_pids = sorted({int(process["pid"]) for process in processes})
            registry = base_status["gpu_compute_process_registry"]
            for pid in unique_pids:
                attachments = [
                    process for process in processes if int(process["pid"]) == pid
                ]
                registry_key = str(pid)
                record = registry.setdefault(
                    registry_key,
                    {
                        "pid": pid,
                        "first_seen_utc": observed_utc,
                        "observation_count": 0,
                        "command_history": [],
                        "command_status_history": [],
                        "process_name_history": [],
                        "gpu_indices_seen": [],
                    },
                )
                record["last_seen_utc"] = observed_utc
                record["observation_count"] += 1
                for attachment in attachments:
                    for field, history_field in (
                        ("command", "command_history"),
                        ("command_status", "command_status_history"),
                        ("process_name", "process_name_history"),
                    ):
                        value = attachment.get(field, "")
                        if value not in record[history_field]:
                            record[history_field].append(value)
                record["gpu_indices_seen"] = sorted(
                    set(record["gpu_indices_seen"])
                    | {int(attachment["gpu_index"]) for attachment in attachments}
                )

            base_status["gpu_compute_observation_count"] += 1
            base_status.update(
                {
                    "state": "waiting_for_gpu_compute_quiescence",
                    "updated_utc": _utc_now(),
                    "observation_phase": observation_phase,
                    "observed_gpu_compute_pids": unique_pids,
                    "observed_gpu_compute_processes": list(processes),
                    "consecutive_clear_observations": clear_observations,
                }
            )
            _atomic_json(STATUS_PATH, base_status)

        def verify_launch_inputs() -> None:
            _verify_snapshot(entry_snapshot)
            _assert_outputs_absent()

        _wait_for_gpu_clear(
            required_clear_observations=args.clear_observations,
            poll_seconds=args.poll_seconds,
            observe=record_gpu_observation,
            verify_ready=verify_launch_inputs,
        )

        # Recheck once more at the narrow launch boundary.  This cannot replace
        # a cluster scheduler's exclusive allocation, but it closes the local
        # work done between the spaced-clear gate and the first Popen call.
        verify_launch_inputs()
        immediate_processes = tuple(_gpu_compute_processes())
        record_gpu_observation(
            immediate_processes,
            args.clear_observations + 1 if not immediate_processes else 0,
            "immediate_prelaunch_confirmation",
        )
        if immediate_processes:
            raise RuntimeError(
                "GPU compute activity appeared at the immediate launch boundary; "
                "no detector subprocess was started"
            )

        # No external PID is ever signalled.  Any later cleanup is restricted
        # to Popen objects created below and retained in ``children``.
        LOG_DIR.mkdir(parents=True, exist_ok=False)
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": VISIBLE_GPUS,
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTHONUNBUFFERED": "1",
            }
        )
        for run in RUNS:
            run_id = str(run["run_id"])
            command = [
                str(python_bin),
                "-m",
                "rc_irstd.cli.train_detector",
                "--config",
                str(run["config"]),
            ]
            log_path = LOG_DIR / f"{run_id}.stdout_stderr.log"
            handle = log_path.open("wb")
            log_handles.append(handle)
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children[run_id] = process
            base_status["commands"][run_id] = {
                "argv": command,
                "config_sha256": _sha256(Path(run["config"])),
                "log_path": str(log_path),
                "pid": process.pid,
                "expected_output_dir": str(run["output_dir"]),
            }

        base_status.update({"state": "running", "launched_utc": _utc_now()})
        _atomic_json(STATUS_PATH, base_status)
        while True:
            returncodes = {
                run_id: process.poll() for run_id, process in children.items()
            }
            failures = {
                run_id: code
                for run_id, code in returncodes.items()
                if code is not None and code != 0
            }
            base_status.update(
                {
                    "state": "running" if not failures else "terminating_after_failure",
                    "updated_utc": _utc_now(),
                    "returncodes": returncodes,
                }
            )
            _atomic_json(STATUS_PATH, base_status)
            if failures:
                _terminate_owned(children)
                raise RuntimeError(f"Phase-1 detector subprocess failed: {failures}")
            if all(code == 0 for code in returncodes.values()):
                break
            time.sleep(args.poll_seconds)

        completed: dict[str, Any] = {}
        for run in RUNS:
            checkpoint_path = Path(run["output_dir"]) / "last.pt"
            completed[str(run["run_id"])] = _validate_checkpoint(
                checkpoint_path, run["source_names"]
            )
        _verify_snapshot(entry_snapshot)
        base_status.update(
            {
                "state": "completed",
                "updated_utc": _utc_now(),
                "completed_utc": _utc_now(),
                "returncodes": {run_id: 0 for run_id in children},
                "completed_checkpoints": completed,
                "foreign_process_signals_sent": False,
                "external_process_signals_sent": False,
                "all_entry_files_unchanged_at_completion": True,
            }
        )
        _atomic_json(STATUS_PATH, base_status)
        return 0
    except BaseException as error:
        termination_error: str | None = None
        if children:
            try:
                _terminate_owned(children)
            except BaseException as cleanup_error:  # pragma: no cover - OS failure
                termination_error = (
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        base_status.update(
            {
                "state": "failed",
                "updated_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "owned_children_termination_attempted": bool(children),
                "owned_children_termination_error": termination_error,
                "foreign_process_signals_sent": False,
                "external_process_signals_sent": False,
            }
        )
        _atomic_json(STATUS_PATH, base_status)
        raise
    finally:
        for handle in log_handles:
            handle.close()
        _release_launcher_lock(lock_record)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wait-pid",
        type=int,
        default=None,
        help=(
            "Deprecated observation hint retained for command compatibility; "
            "never used as a launch gate"
        ),
    )
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--clear-observations", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.wait_pid is not None and args.wait_pid <= 1:
        raise ValueError("--wait-pid must identify a non-system process")
    if not 5.0 <= args.poll_seconds <= 300.0:
        raise ValueError("--poll-seconds must lie in [5, 300]")
    if not 2 <= args.clear_observations <= 20:
        raise ValueError("--clear-observations must lie in [2, 20]")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
