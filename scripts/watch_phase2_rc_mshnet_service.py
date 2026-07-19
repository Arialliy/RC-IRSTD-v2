#!/usr/bin/env python3
"""Read-only watchdog for the RC-MSHNet Phase-2 coordinator service.

The watchdog never repairs, truncates, archives, or removes experiment artifacts.
It requests one service restart only after a stable no-commit timeout or a stable
finalization timeout.  Identity or artifact inconsistencies fail closed instead.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
SERVICE_NAME = "rc-irstd-phase2-coordinator.service"
COORDINATOR_SCRIPT = "coordinate_phase2_rc_mshnet_gate.py"
HOLD_NAME = "HOLD_RC_MSHNET_GATE"
ALLOW_NAME = "ALLOW_RC_MSHNET_GATE"
LAUNCH_ID_ENV = "RC_IRSTD_PHASE2_LAUNCH_ID"
LAUNCH_SCHEMA = "rc-irstd-phase2-launch-v2"
EXPECTED_INITIALIZER_RELATIVE = (
    "artifacts/aaai27/initializers/"
    "mshnet_seed42_train_nudt_irstd_tensor_only.pt"
)
EXPECTED_EPOCHS = 80
DEFAULT_STALL_SECONDS = 20 * 60.0
DEFAULT_FINALIZING_SECONDS = 5 * 60.0
DEFAULT_EVIDENCE_GRACE_SECONDS = 60.0
DEFAULT_POLL_SECONDS = 30.0


class EvidenceError(RuntimeError):
    """Raised when observed evidence is unsafe or internally inconsistent."""


class Decision(str, Enum):
    CONTINUE = "continue"
    EXIT_SAFE = "exit_safe"
    RESTART = "restart"


@dataclass(frozen=True)
class RunSpec:
    role: str
    config_relative: str
    physical_gpu: int


@dataclass(frozen=True)
class RoundSpec:
    name: str
    runs: tuple[RunSpec, ...]


ROUNDS = (
    RoundSpec(
        "round1",
        (
            RunSpec(
                "mshnet_ft_matched_control",
                "configs/phase2_mshnet_ft_outer_nuaa.yaml",
                2,
            ),
            RunSpec(
                "rc_mshnet_full",
                "configs/phase2_rc_mshnet_full_outer_nuaa.yaml",
                3,
            ),
        ),
    ),
    RoundSpec(
        "round2",
        (
            RunSpec(
                "rc_mshnet_no_contrast",
                "configs/phase2_rc_mshnet_no_contrast_outer_nuaa.yaml",
                2,
            ),
            RunSpec(
                "rc_mshnet_no_component",
                "configs/phase2_rc_mshnet_no_component_outer_nuaa.yaml",
                3,
            ),
        ),
    ),
    RoundSpec(
        "round3",
        (
            RunSpec(
                "rc_mshnet_no_gate_fixed_average",
                "configs/phase2_rc_mshnet_no_gate_outer_nuaa.yaml",
                2,
            ),
            RunSpec(
                "rc_mshnet_branch_aux",
                "configs/phase2_rc_mshnet_branch_aux_outer_nuaa.yaml",
                3,
            ),
        ),
    ),
)


@dataclass(frozen=True)
class ServiceState:
    active_state: str
    sub_state: str
    main_pid: int
    control_group: str

    @property
    def running(self) -> bool:
        return (
            self.active_state == "active"
            and self.sub_state == "running"
            and self.main_pid > 0
        )


@dataclass(frozen=True)
class RunObservation:
    role: str
    physical_gpu: int
    committed_epochs: int
    history_rows: int
    checkpoint_epoch: int | None
    complete: bool
    launcher_present: bool
    launcher_live: bool
    artifacts_present: bool
    last_commit_wall_time: float | None
    launch_wall_time: float | None
    pending_evidence: str | None = None
    pending_since_wall_time: float | None = None


@dataclass(frozen=True)
class Snapshot:
    gate_state: str
    service: ServiceState | None
    current_round: str | None
    runs: tuple[RunObservation, ...]
    all_rounds_complete: bool
    phase_status: str | None


@dataclass
class _RoleTimer:
    committed_epochs: int
    last_commit_monotonic: float
    finalizing_since_monotonic: float | None = None
    pending_since_monotonic: float | None = None


class FileHashCache:
    """Cache immutable-input hashes without trusting path alone."""

    def __init__(self) -> None:
        self._values: dict[Path, tuple[int, int, str]] = {}
        self._checkpoint_values: dict[Path, tuple[int, int, int, float]] = {}

    def sha256(self, path: Path) -> str:
        resolved = path.resolve()
        stat = resolved.stat()
        cached = self._values.get(resolved)
        identity = (stat.st_size, stat.st_mtime_ns)
        if cached is not None and cached[:2] == identity:
            return cached[2]
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        value = digest.hexdigest()
        self._values[resolved] = (*identity, value)
        return value

    def checkpoint_epoch(self, path: Path) -> tuple[int, float] | None:
        resolved = path.resolve()
        stat = resolved.stat()
        cached = self._checkpoint_values.get(resolved)
        if cached is None or cached[:2] != (stat.st_size, stat.st_mtime_ns):
            return None
        return cached[2], cached[3]

    def store_checkpoint_epoch(self, path: Path, epoch: int) -> float:
        resolved = path.resolve()
        stat = resolved.stat()
        self._checkpoint_values[resolved] = (
            stat.st_size,
            stat.st_mtime_ns,
            epoch,
            stat.st_mtime,
        )
        return stat.st_mtime


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read valid JSON evidence: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise EvidenceError(f"JSON evidence root is not a mapping: {path}")
    return payload


def _stable_bytes(path: Path, *, attempts: int = 3) -> bytes:
    """Avoid treating a short append-in-progress window as corrupt evidence."""

    last: bytes | None = None
    for attempt in range(attempts):
        before = path.stat()
        value = path.read_bytes()
        after = path.stat()
        if (
            before.st_size == after.st_size == len(value)
            and before.st_mtime_ns == after.st_mtime_ns
        ):
            return value
        last = value
        if attempt + 1 < attempts:
            time.sleep(0.02)
    if last is None:
        raise EvidenceError(f"could not read stable evidence: {path}")
    raise EvidenceError(f"evidence kept changing while sampled: {path}")


def _read_history(path: Path) -> tuple[int, float | None]:
    if not path.is_file():
        return 0, None
    raw = _stable_bytes(path)
    try:
        text = raw.decode("utf-8")
        reader = csv.DictReader(text.splitlines())
        fieldnames = reader.fieldnames
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as error:
        raise EvidenceError(f"invalid history CSV: {path}: {error}") from error
    if (
        not isinstance(fieldnames, list)
        or len(fieldnames) != len(set(fieldnames))
        or not {"epoch", "lr", "loss_total"}.issubset(fieldnames)
    ):
        raise EvidenceError(f"history has an invalid schema: {path}")
    epochs: list[int] = []
    for row_index, row in enumerate(rows):
        try:
            epoch = int(row["epoch"])
        except (KeyError, TypeError, ValueError) as error:
            raise EvidenceError(
                f"history row {row_index} has an invalid epoch: {path}"
            ) from error
        epochs.append(epoch)
        for field in fieldnames:
            raw_value = row.get(field)
            if raw_value is None or not raw_value.strip():
                raise EvidenceError(
                    f"history row {row_index} has an empty {field}: {path}"
                )
            if field == "epoch":
                continue
            try:
                value = float(raw_value)
            except ValueError as error:
                raise EvidenceError(
                    f"history row {row_index} has non-numeric {field}: {path}"
                ) from error
            if not math.isfinite(value):
                raise EvidenceError(
                    f"history row {row_index} has non-finite {field}: {path}"
                )
    if epochs != list(range(len(rows))):
        raise EvidenceError(f"history epochs are not exactly contiguous 0..N: {path}")
    if len(rows) > EXPECTED_EPOCHS:
        raise EvidenceError(f"history exceeds {EXPECTED_EPOCHS} epochs: {path}")
    return len(rows), path.stat().st_mtime


def _read_checkpoint_epoch(
    path: Path,
    *,
    cache: FileHashCache | None = None,
) -> tuple[int | None, float | None]:
    if not path.is_file():
        return None, None
    if cache is not None:
        cached = cache.checkpoint_epoch(path)
        if cached is not None:
            return cached
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise EvidenceError(f"checkpoint is not safely loadable: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise EvidenceError(f"checkpoint root is not a mapping: {path}")
    if payload.get("format_version") != 2 or payload.get("kind") != "detector":
        raise EvidenceError(f"checkpoint does not satisfy detector format v2: {path}")
    try:
        epoch = int(payload.get("epoch", -1))
    except (TypeError, ValueError) as error:
        raise EvidenceError(f"checkpoint epoch is invalid: {path}") from error
    if not 0 <= epoch < EXPECTED_EPOCHS:
        raise EvidenceError(f"checkpoint epoch is outside 0..79: {path}")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise EvidenceError(f"checkpoint has no embedded formal config: {path}")
    training = config.get("training")
    if not isinstance(training, dict) or int(training.get("epochs", -1)) != EXPECTED_EPOCHS:
        raise EvidenceError(f"checkpoint has the wrong training schedule: {path}")
    if training.get("resume") is not None:
        raise EvidenceError(f"checkpoint leaked a runtime resume path: {path}")
    mtime = (
        cache.store_checkpoint_epoch(path, epoch)
        if cache is not None
        else path.stat().st_mtime
    )
    return epoch, mtime


def _load_output_dir(root: Path, spec: RunSpec) -> tuple[Path, Path]:
    config_path = (root / spec.config_relative).resolve()
    if not config_path.is_file():
        raise EvidenceError(f"missing frozen Phase-2 config: {config_path}")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise EvidenceError(f"cannot load frozen config {config_path}: {error}") from error
    if not isinstance(config, dict):
        raise EvidenceError(f"frozen config root is not a mapping: {config_path}")
    training = config.get("training")
    if not isinstance(training, dict) or int(training.get("epochs", -1)) != EXPECTED_EPOCHS:
        raise EvidenceError(f"frozen config does not specify 80 epochs: {config_path}")
    configured = Path(str(config.get("output_dir", "")))
    if not str(configured):
        raise EvidenceError(f"frozen config has no output_dir: {config_path}")
    output_dir = (
        configured.resolve()
        if configured.is_absolute()
        else (config_path.parent / configured).resolve()
    )
    return config_path, output_dir


def _read_process_identity(pid: int) -> dict[str, Any] | None:
    if pid <= 0:
        return None
    proc = Path("/proc") / str(pid)
    try:
        stat_text = (proc / "stat").read_text(encoding="utf-8")
        close_paren = stat_text.rfind(")")
        if close_paren < 0:
            raise EvidenceError(f"malformed /proc/{pid}/stat")
        fields = stat_text[close_paren + 2 :].split()
        if len(fields) <= 19:
            raise EvidenceError(f"incomplete /proc/{pid}/stat")
        state = fields[0]
        # A normally exited child can remain as a zombie until its coordinator
        # reaches the next poll and reaps it.  Linux may reject dereferencing
        # cwd/exe for a zombie; the state itself is sufficient for callers to
        # classify this launcher as no longer live.
        if state in {"Z", "X", "x"}:
            return {"pid": pid, "state": state}
        argv = [
            item.decode("utf-8", errors="surrogateescape")
            for item in (proc / "cmdline").read_bytes().split(b"\0")
            if item
        ]
        environment: dict[str, str] = {}
        for item in (proc / "environ").read_bytes().split(b"\0"):
            if not item or b"=" not in item:
                continue
            key, value = item.split(b"=", 1)
            decoded_key = key.decode("utf-8", errors="surrogateescape")
            if decoded_key in {LAUNCH_ID_ENV, "CUDA_VISIBLE_DEVICES"}:
                environment[decoded_key] = value.decode(
                    "utf-8", errors="surrogateescape"
                )
        cgroups = (proc / "cgroup").read_text(encoding="utf-8").splitlines()
        unified = [line.split("::", 1)[1] for line in cgroups if "::" in line]
        return {
            "pid": pid,
            "state": state,
            "ppid": int(fields[1]),
            "start_time_ticks": int(fields[19]),
            "argv": argv,
            "cwd": str((proc / "cwd").resolve()),
            "process_group_id": os.getpgid(pid),
            "session_id": os.getsid(pid),
            "launch_id": environment.get(LAUNCH_ID_ENV),
            "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
            "control_group": unified[0] if len(unified) == 1 else None,
        }
    except (FileNotFoundError, ProcessLookupError):
        return None
    except PermissionError as error:
        raise EvidenceError(f"cannot inspect process identity for pid={pid}") from error


def _validate_command(
    command: Any,
    *,
    config_path: Path,
    output_dir: Path,
    initializer: Path,
) -> list[str]:
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise EvidenceError("launcher command is not a string list")
    if len(command) < 5 or command[1:3] != ["-m", "rc_irstd.cli.train_detector"]:
        raise EvidenceError("launcher command is not rc_irstd.cli.train_detector")
    try:
        config_indices = [index for index, value in enumerate(command) if value == "--config"]
        if len(config_indices) != 1:
            raise ValueError("config count")
        recorded_config = command[config_indices[0] + 1]
    except (IndexError, ValueError) as error:
        raise EvidenceError("launcher command has no unique --config value") from error
    if recorded_config != str(config_path):
        raise EvidenceError("launcher command config path mismatch")
    overrides: dict[str, str] = {}
    for index, value in enumerate(command):
        if value == "--set":
            if index + 1 >= len(command):
                raise EvidenceError("launcher command ends after --set")
            raw_override = command[index + 1]
            if "=" not in raw_override:
                raise EvidenceError("launcher command has a malformed --set value")
            key, value = raw_override.split("=", 1)
            if key in overrides:
                raise EvidenceError(f"launcher command duplicates override {key}")
            overrides[key] = value
    required = {
        "device": "cuda:0",
        "training.initialize_from": str(initializer),
        "output_dir": str(output_dir),
    }
    if any(overrides.get(key) != value for key, value in required.items()):
        raise EvidenceError("launcher command lacks a frozen Phase-2 override")
    return command


def _launcher_wall_time(launcher: Mapping[str, Any], launcher_path: Path) -> float:
    raw = launcher.get("launched_at")
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw).timestamp()
        except ValueError:
            pass
    return launcher_path.stat().st_mtime


def _observe_run(
    root: Path,
    spec: RunSpec,
    *,
    service: ServiceState,
    hash_cache: FileHashCache,
) -> RunObservation:
    config_path, output_dir = _load_output_dir(root, spec)
    if not output_dir.exists():
        return RunObservation(
            role=spec.role,
            physical_gpu=spec.physical_gpu,
            committed_epochs=0,
            history_rows=0,
            checkpoint_epoch=None,
            complete=False,
            launcher_present=False,
            launcher_live=False,
            artifacts_present=False,
            last_commit_wall_time=None,
            launch_wall_time=None,
        )
    if not output_dir.is_dir():
        raise EvidenceError(f"Phase-2 output path is not a directory: {output_dir}")
    artifacts_present = any(output_dir.iterdir())
    history_rows, history_mtime = _read_history(output_dir / "history.csv")
    checkpoint_epoch, checkpoint_mtime = _read_checkpoint_epoch(
        output_dir / "last.pt",
        cache=hash_cache,
    )

    pending: str | None = None
    pending_since: float | None = None
    if checkpoint_epoch is None:
        if history_rows == 0:
            committed = 0
        elif history_rows == 1:
            committed = 0
            pending = "history_has_one_uncommitted_row_without_checkpoint"
            pending_since = history_mtime
        else:
            raise EvidenceError(
                f"{spec.role} has {history_rows} history rows without a checkpoint"
            )
    elif history_rows == checkpoint_epoch + 1:
        committed = history_rows
    elif history_rows == checkpoint_epoch + 2:
        committed = checkpoint_epoch + 1
        pending = "history_is_one_row_ahead_of_checkpoint"
        pending_since = history_mtime
    else:
        raise EvidenceError(
            f"{spec.role} history/checkpoint mismatch: rows={history_rows}, "
            f"checkpoint_epoch={checkpoint_epoch}"
        )
    temporary_paths = (
        output_dir / "last.pt.tmp",
        output_dir / "history.csv.tmp",
    )
    existing_temporaries = [path for path in temporary_paths if path.exists()]
    if existing_temporaries and pending is None:
        pending = "temporary_training_artifact_exists"
        pending_since = min(path.stat().st_mtime for path in existing_temporaries)

    launcher_path = output_dir / "launcher.json"
    launcher_present = launcher_path.is_file()
    launcher_live = False
    launch_wall_time: float | None = None
    if launcher_present:
        launcher = _read_json_mapping(launcher_path)
        expected_scalars = {
            "schema_version": LAUNCH_SCHEMA,
            "role": spec.role,
            "physical_gpu": spec.physical_gpu,
            "visible_device": "cuda:0",
            "config_path": str(config_path),
            "config_sha256": hash_cache.sha256(config_path),
            "expected_output_dir": str(output_dir),
        }
        for key, expected in expected_scalars.items():
            if launcher.get(key) != expected:
                raise EvidenceError(f"{spec.role} launcher {key} mismatch")
        launch_id = launcher.get("launch_id")
        if (
            not isinstance(launch_id, str)
            or len(launch_id) != 32
            or any(character not in "0123456789abcdef" for character in launch_id)
        ):
            raise EvidenceError(f"{spec.role} launcher has an invalid launch ID")
        initializer_raw = launcher.get("initializer_path")
        if not isinstance(initializer_raw, str):
            raise EvidenceError(f"{spec.role} launcher has no initializer path")
        initializer = Path(initializer_raw).resolve()
        expected_initializer = (root / EXPECTED_INITIALIZER_RELATIVE).resolve()
        if initializer != expected_initializer:
            raise EvidenceError(f"{spec.role} launcher initializer path mismatch")
        if not initializer.is_file():
            raise EvidenceError(f"{spec.role} launcher initializer is missing")
        if launcher.get("initializer_sha256") != hash_cache.sha256(initializer):
            raise EvidenceError(f"{spec.role} launcher initializer SHA mismatch")
        command = _validate_command(
            launcher.get("command"),
            config_path=config_path,
            output_dir=output_dir,
            initializer=initializer,
        )
        pid = launcher.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            raise EvidenceError(f"{spec.role} launcher PID is invalid")
        recorded = launcher.get("process_identity")
        if not isinstance(recorded, dict):
            raise EvidenceError(f"{spec.role} launcher has no process identity")
        current = _read_process_identity(pid)
        launcher_live = bool(
            current is not None and str(current.get("state")) not in {"Z", "X", "x"}
        )
        if launcher_live and current is not None:
            for key in (
                "pid",
                "start_time_ticks",
                "argv",
                "cwd",
                "process_group_id",
                "session_id",
                "launch_id",
            ):
                if current.get(key) != recorded.get(key):
                    raise EvidenceError(
                        f"{spec.role} live process identity mismatch for {key}"
                    )
            if current.get("argv") != command:
                raise EvidenceError(f"{spec.role} live argv differs from launcher")
            if current.get("cwd") != str(root.resolve()):
                raise EvidenceError(f"{spec.role} live process has the wrong cwd")
            if current.get("process_group_id") != pid or current.get("session_id") != pid:
                raise EvidenceError(f"{spec.role} live process is not session-isolated")
            if current.get("cuda_visible_devices") != str(spec.physical_gpu):
                raise EvidenceError(
                    f"{spec.role} live CUDA_VISIBLE_DEVICES is not {spec.physical_gpu}"
                )
            if current.get("control_group") != service.control_group:
                raise EvidenceError(f"{spec.role} is outside the coordinator cgroup")
        launch_wall_time = _launcher_wall_time(launcher, launcher_path)
    elif committed > 0 or checkpoint_epoch is not None:
        raise EvidenceError(f"{spec.role} has training state without launcher evidence")

    complete = (
        pending is None
        and history_rows == EXPECTED_EPOCHS
        and checkpoint_epoch == EXPECTED_EPOCHS - 1
    )
    last_commit_wall = checkpoint_mtime if committed > 0 else None
    return RunObservation(
        role=spec.role,
        physical_gpu=spec.physical_gpu,
        committed_epochs=committed,
        history_rows=history_rows,
        checkpoint_epoch=checkpoint_epoch,
        complete=complete,
        launcher_present=launcher_present,
        launcher_live=launcher_live,
        artifacts_present=artifacts_present,
        last_commit_wall_time=last_commit_wall,
        launch_wall_time=launch_wall_time,
        pending_evidence=pending,
        pending_since_wall_time=pending_since,
    )


def _probe_service(service_name: str = SERVICE_NAME) -> ServiceState:
    completed = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            service_name,
            "--no-pager",
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "MainPID",
            "-p",
            "ControlGroup",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise EvidenceError(
            f"cannot query {service_name}: {completed.stderr.strip()}"
        )
    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    try:
        main_pid = int(values.get("MainPID", "0"))
    except ValueError as error:
        raise EvidenceError(f"{service_name} returned an invalid MainPID") from error
    return ServiceState(
        active_state=values.get("ActiveState", ""),
        sub_state=values.get("SubState", ""),
        main_pid=main_pid,
        control_group=values.get("ControlGroup", ""),
    )


def _validate_coordinator_process(root: Path, service: ServiceState) -> None:
    if not service.running:
        raise EvidenceError(
            f"coordinator service is not active/running: "
            f"{service.active_state}/{service.sub_state}"
        )
    identity = _read_process_identity(service.main_pid)
    if identity is None or str(identity.get("state")) in {"Z", "X", "x"}:
        raise EvidenceError("coordinator MainPID is not live")
    if identity.get("cwd") != str(root.resolve()):
        raise EvidenceError("coordinator MainPID has the wrong cwd")
    argv = identity.get("argv")
    if not isinstance(argv, list) or not any(
        str(value).endswith(COORDINATOR_SCRIPT) for value in argv
    ):
        raise EvidenceError("coordinator MainPID argv does not name the coordinator")
    if identity.get("control_group") != service.control_group:
        raise EvidenceError("coordinator MainPID is outside its service cgroup")


def _read_gate_state(root: Path) -> str:
    state_dir = root / "outputs/phase_state"
    hold = (state_dir / HOLD_NAME).is_file()
    allow = (state_dir / ALLOW_NAME).is_file()
    if hold and allow:
        # Both files are momentarily present during safe HOLD restoration.  Re-read
        # once so that the deliberate close ordering is not treated as corruption.
        time.sleep(0.1)
        hold = (state_dir / HOLD_NAME).is_file()
        allow = (state_dir / ALLOW_NAME).is_file()
    if hold and not allow:
        return "HOLD"
    if allow and not hold:
        return "ALLOW"
    if hold and allow:
        return "BOTH"
    return "NEITHER"


def _read_phase_status(root: Path) -> str | None:
    path = root / "artifacts/aaai27/audit/phase2_status.json"
    if not path.is_file():
        return None
    payload = _read_json_mapping(path)
    status = payload.get("status")
    return str(status) if status is not None else None


def collect_snapshot(
    root: Path,
    *,
    rounds: tuple[RoundSpec, ...] = ROUNDS,
    service_name: str = SERVICE_NAME,
    hash_cache: FileHashCache | None = None,
) -> Snapshot:
    resolved_root = root.resolve()
    gate_state = _read_gate_state(resolved_root)
    phase_status = _read_phase_status(resolved_root)
    if gate_state == "HOLD":
        return Snapshot(
            gate_state=gate_state,
            service=None,
            current_round=None,
            runs=(),
            all_rounds_complete=phase_status == "completed",
            phase_status=phase_status,
        )
    if gate_state != "ALLOW":
        raise EvidenceError(f"invalid stable Phase-2 gate state: {gate_state}")
    if phase_status == "completed":
        raise EvidenceError("phase2_status is completed while the gate remains ALLOW")
    service = _probe_service(service_name)
    _validate_coordinator_process(resolved_root, service)
    cache = hash_cache or FileHashCache()
    observed_rounds: list[tuple[RoundSpec, tuple[RunObservation, ...]]] = []
    for round_spec in rounds:
        observations = tuple(
            _observe_run(
                resolved_root,
                run,
                service=service,
                hash_cache=cache,
            )
            for run in round_spec.runs
        )
        observed_rounds.append((round_spec, observations))

    current_index: int | None = None
    for index, (_, observations) in enumerate(observed_rounds):
        fully_quiescent = all(run.complete and not run.launcher_live for run in observations)
        if not fully_quiescent:
            current_index = index
            break
    if current_index is None:
        final_runs = observed_rounds[-1][1] if observed_rounds else ()
        return Snapshot(
            gate_state=gate_state,
            service=service,
            current_round=None,
            runs=final_runs,
            all_rounds_complete=True,
            phase_status=phase_status,
        )

    current_spec, current_runs = observed_rounds[current_index]
    for _, later_runs in observed_rounds[current_index + 1 :]:
        for run in later_runs:
            if run.launcher_present or run.committed_epochs > 0:
                raise EvidenceError(
                    f"later-round artifacts exist before {current_spec.name} is quiescent"
                )
    for run in current_runs:
        if run.launcher_present and not run.launcher_live and not run.complete:
            raise EvidenceError(
                f"{run.role} launcher is dead before an epoch-79 checkpoint"
            )
    return Snapshot(
        gate_state=gate_state,
        service=service,
        current_round=current_spec.name,
        runs=current_runs,
        all_rounds_complete=False,
        phase_status=phase_status,
    )


class Phase2Watchdog:
    def __init__(
        self,
        *,
        stall_seconds: float = DEFAULT_STALL_SECONDS,
        finalizing_seconds: float = DEFAULT_FINALIZING_SECONDS,
        evidence_grace_seconds: float = DEFAULT_EVIDENCE_GRACE_SECONDS,
    ) -> None:
        if min(stall_seconds, finalizing_seconds, evidence_grace_seconds) <= 0:
            raise ValueError("watchdog timeouts must be positive")
        self.stall_seconds = float(stall_seconds)
        self.finalizing_seconds = float(finalizing_seconds)
        self.evidence_grace_seconds = float(evidence_grace_seconds)
        self._round: str | None = None
        self._roles: dict[str, _RoleTimer] = {}
        self._transition_since: float | None = None
        self.reason: str | None = None

    @staticmethod
    def _wall_to_monotonic(
        wall_timestamp: float | None,
        *,
        monotonic_now: float,
        wall_now: float,
    ) -> float:
        if wall_timestamp is None:
            return monotonic_now
        elapsed = max(0.0, wall_now - wall_timestamp)
        return monotonic_now - elapsed

    def step(
        self,
        snapshot: Snapshot,
        *,
        monotonic_now: float,
        wall_now: float,
    ) -> Decision:
        self.reason = None
        if snapshot.gate_state == "HOLD":
            self.reason = "gate_is_hold"
            return Decision.EXIT_SAFE
        if snapshot.gate_state != "ALLOW":
            raise EvidenceError(f"unsafe gate state: {snapshot.gate_state}")
        if snapshot.service is None or not snapshot.service.running:
            raise EvidenceError("ALLOW is open without a running coordinator service")

        if snapshot.all_rounds_complete:
            if self._transition_since is None:
                completed_times = [
                    run.last_commit_wall_time
                    for run in snapshot.runs
                    if run.last_commit_wall_time is not None
                ]
                latest = max(completed_times) if completed_times else None
                self._transition_since = self._wall_to_monotonic(
                    latest,
                    monotonic_now=monotonic_now,
                    wall_now=wall_now,
                )
            if monotonic_now - self._transition_since >= self.stall_seconds:
                self.reason = "all_rounds_complete_but_gate_remained_allow"
                return Decision.RESTART
            return Decision.CONTINUE

        if snapshot.current_round is None:
            raise EvidenceError("incomplete Phase-2 snapshot has no current round")
        if self._round != snapshot.current_round:
            self._round = snapshot.current_round
            self._roles.clear()
            self._transition_since = None

        observed_roles = {run.role for run in snapshot.runs}
        if len(observed_roles) != len(snapshot.runs):
            raise EvidenceError("current round contains duplicate roles")
        for role in list(self._roles):
            if role not in observed_roles:
                del self._roles[role]

        for run in snapshot.runs:
            if run.launcher_present and not run.launcher_live and not run.complete:
                raise EvidenceError(
                    f"{run.role} launcher is dead before an epoch-79 checkpoint"
                )
            timer = self._roles.get(run.role)
            if timer is None:
                activity_wall = run.last_commit_wall_time or run.launch_wall_time
                timer = _RoleTimer(
                    committed_epochs=run.committed_epochs,
                    last_commit_monotonic=self._wall_to_monotonic(
                        activity_wall,
                        monotonic_now=monotonic_now,
                        wall_now=wall_now,
                    ),
                )
                self._roles[run.role] = timer
            elif run.committed_epochs < timer.committed_epochs:
                raise EvidenceError(
                    f"{run.role} committed epoch count regressed from "
                    f"{timer.committed_epochs} to {run.committed_epochs}"
                )
            elif run.committed_epochs > timer.committed_epochs:
                timer.committed_epochs = run.committed_epochs
                timer.last_commit_monotonic = monotonic_now
                timer.pending_since_monotonic = None

            if run.pending_evidence is not None:
                if timer.pending_since_monotonic is None:
                    timer.pending_since_monotonic = self._wall_to_monotonic(
                        run.pending_since_wall_time,
                        monotonic_now=monotonic_now,
                        wall_now=wall_now,
                    )
                if (
                    monotonic_now - timer.pending_since_monotonic
                    >= self.evidence_grace_seconds
                ):
                    raise EvidenceError(
                        f"{run.role} has stable inconsistent evidence: "
                        f"{run.pending_evidence}"
                    )
            else:
                timer.pending_since_monotonic = None

            if run.complete and run.launcher_live:
                if timer.finalizing_since_monotonic is None:
                    timer.finalizing_since_monotonic = self._wall_to_monotonic(
                        run.last_commit_wall_time,
                        monotonic_now=monotonic_now,
                        wall_now=wall_now,
                    )
                if (
                    monotonic_now - timer.finalizing_since_monotonic
                    >= self.finalizing_seconds
                ):
                    self.reason = f"{run.role}_finalizing_timeout"
                    return Decision.RESTART
            else:
                timer.finalizing_since_monotonic = None

            if not run.complete:
                if (
                    monotonic_now - timer.last_commit_monotonic
                    >= self.stall_seconds
                ):
                    self.reason = f"{run.role}_no_committed_epoch_timeout"
                    return Decision.RESTART
        return Decision.CONTINUE


def _render_snapshot(snapshot: Snapshot, *, decision: Decision, reason: str | None) -> str:
    payload = {
        "at": _now(),
        "decision": decision.value,
        "reason": reason,
        "gate_state": snapshot.gate_state,
        "service": (
            None
            if snapshot.service is None
            else {
                "active_state": snapshot.service.active_state,
                "sub_state": snapshot.service.sub_state,
                "main_pid": snapshot.service.main_pid,
            }
        ),
        "current_round": snapshot.current_round,
        "all_rounds_complete": snapshot.all_rounds_complete,
        "runs": [
            {
                "role": run.role,
                "gpu": run.physical_gpu,
                "committed_epochs": run.committed_epochs,
                "history_rows": run.history_rows,
                "checkpoint_epoch": run.checkpoint_epoch,
                "complete": run.complete,
                "launcher_live": run.launcher_live,
                "pending_evidence": run.pending_evidence,
            }
            for run in snapshot.runs
        ],
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def monitor(
    snapshot_provider: Callable[[], Snapshot],
    restart_callback: Callable[[], None],
    *,
    watchdog: Phase2Watchdog | None = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    monotonic_clock: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], None] = time.sleep,
    emit: Callable[[str], None] = print,
) -> str:
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    state = watchdog or Phase2Watchdog()
    while True:
        snapshot = snapshot_provider()
        decision = state.step(
            snapshot,
            monotonic_now=monotonic_clock(),
            wall_now=wall_clock(),
        )
        emit(_render_snapshot(snapshot, decision=decision, reason=state.reason))
        if decision is Decision.EXIT_SAFE:
            return "safe_exit"
        if decision is Decision.RESTART:
            restart_callback()
            return "restart_requested"
        sleeper(poll_seconds)


def restart_service(service_name: str = SERVICE_NAME) -> None:
    """Request one complete cgroup restart; no direct PID signaling is used."""

    subprocess.run(
        ["systemctl", "--user", "restart", service_name],
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--service", default=SERVICE_NAME)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--stall-seconds", type=float, default=DEFAULT_STALL_SECONDS)
    parser.add_argument(
        "--finalizing-seconds", type=float, default=DEFAULT_FINALIZING_SECONDS
    )
    parser.add_argument(
        "--evidence-grace-seconds",
        type=float,
        default=DEFAULT_EVIDENCE_GRACE_SECONDS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    cache = FileHashCache()
    watchdog = Phase2Watchdog(
        stall_seconds=args.stall_seconds,
        finalizing_seconds=args.finalizing_seconds,
        evidence_grace_seconds=args.evidence_grace_seconds,
    )
    try:
        monitor(
            lambda: collect_snapshot(
                root,
                service_name=args.service,
                hash_cache=cache,
            ),
            lambda: restart_service(args.service),
            watchdog=watchdog,
            poll_seconds=args.poll_seconds,
        )
    except EvidenceError as error:
        print(
            json.dumps(
                {
                    "at": _now(),
                    "status": "failed_closed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    except (OSError, subprocess.SubprocessError) as error:
        print(
            json.dumps(
                {
                    "at": _now(),
                    "status": "restart_or_service_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
