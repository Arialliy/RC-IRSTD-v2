#!/usr/bin/env python3
"""Wait for StableSLS baselines, freeze identity, then run Phase-2 fail-closed."""

from __future__ import annotations

import csv
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.artifact_integrity import ordered_ids_sha256
from rc_irstd.data import ensure_unique_sample_ids, read_split_file
from rc_irstd.models import build_mshnet
from scripts.phase2_gatekeeper import (
    assert_sentinel_allows,
    validate_gate_ready,
    validate_phase2_configs,
)
from scripts.phase2_identity import sha256_file, write_identity


STATE_DIR = ROOT / "outputs/phase_state"
AUDIT_DIR = ROOT / "artifacts/aaai27/audit"
INITIALIZER_DIR = ROOT / "artifacts/aaai27/initializers"
HOLD = STATE_DIR / "HOLD_RC_MSHNET_GATE"
ALLOW = STATE_DIR / "ALLOW_RC_MSHNET_GATE"
LOG_PATH = STATE_DIR / "phase2_coordinator.log"
LOCK_PATH = STATE_DIR / "phase2_coordinator.lock"
PID_PATH = STATE_DIR / "phase2_coordinator.pid"
SOURCE_MANIFEST = (
    ROOT / "outputs/aaai27/detectors/nuaa/mshnet_sls_seed42/IDENTITY_MANIFEST.json"
)
PHASE2_LAUNCH_SCHEMA = "rc-irstd-phase2-launch-v2"
PHASE2_RESUME_LAUNCH_SCHEMA = "rc-irstd-phase2-launch-v3"
PHASE2_LAUNCH_SCHEMAS = frozenset(
    {PHASE2_LAUNCH_SCHEMA, PHASE2_RESUME_LAUNCH_SCHEMA}
)
FORMAL_RESUME_AUDIT_SCHEMA = "rc-irstd-formal-resume-v1"
PHASE2_LAUNCH_ID_ENV = "RC_IRSTD_PHASE2_LAUNCH_ID"
PHASE2_RECOVERY_SCHEMA = "rc-irstd-phase2-zero-epoch-recovery-v1"
LAUNCH_IDENTITY_TIMEOUT_SECONDS = 5.0
LAUNCH_IDENTITY_POLL_SECONDS = 0.01
NO_PROGRESS_TIMEOUT_SECONDS = 20.0 * 60.0
FINALIZING_TIMEOUT_SECONDS = 5.0 * 60.0


@dataclass(frozen=True)
class BaselineRun:
    key: str
    run_dir: Path
    pid_file: Path
    config_path: Path
    expected_sources: tuple[str, ...]


@dataclass(frozen=True)
class GateRun:
    role: str
    config_path: Path
    gpu: int


@dataclass(frozen=True)
class PartialResumePlan:
    """Immutable evidence required to resume one incomplete Phase-2 run."""

    checkpoint: Path
    checkpoint_sha256: str
    checkpoint_epoch: int
    history_rows: int
    history_sha256: str
    previous_launch_id: str
    previous_launcher_sha256: str


@dataclass
class ProgressWatch:
    fingerprint: tuple[Any, ...]
    last_progress_at: float
    finalizing_since: float | None = None


BASELINES = (
    BaselineRun(
        "pair",
        ROOT / "outputs/aaai27/detectors/nuaa/mshnet_sls_seed42",
        ROOT / "outputs/aaai27/detectors/pair_baseline.pid",
        ROOT / "configs/detector_mshnet_outer_nuaa_sls.yaml",
        ("NUDT-SIRST", "IRSTD-1K"),
    ),
    BaselineRun(
        "nuaa",
        ROOT / "outputs/aaai27/detectors/source_initializers/nuaa_sls_seed42",
        ROOT / "outputs/aaai27/detectors/nuaa_baseline.pid",
        ROOT / "configs/detector_mshnet_source_nuaa_sls.yaml",
        ("NUAA-SIRST",),
    ),
    BaselineRun(
        "nudt",
        ROOT / "outputs/aaai27/detectors/source_initializers/nudt_sls_seed42",
        ROOT / "outputs/aaai27/detectors/nudt_baseline.pid",
        ROOT / "configs/detector_mshnet_source_nudt_sls.yaml",
        ("NUDT-SIRST",),
    ),
    BaselineRun(
        "irstd",
        ROOT / "outputs/aaai27/detectors/source_initializers/irstd_sls_seed42",
        ROOT / "outputs/aaai27/detectors/irstd_baseline.pid",
        ROOT / "configs/detector_mshnet_source_irstd_sls.yaml",
        ("IRSTD-1K",),
    ),
)
ROUND1 = (
    GateRun(
        "mshnet_ft_matched_control",
        ROOT / "configs/phase2_mshnet_ft_outer_nuaa.yaml",
        2,
    ),
    GateRun(
        "rc_mshnet_full",
        ROOT / "configs/phase2_rc_mshnet_full_outer_nuaa.yaml",
        3,
    ),
)
ROUND2 = (
    GateRun(
        "rc_mshnet_no_contrast",
        ROOT / "configs/phase2_rc_mshnet_no_contrast_outer_nuaa.yaml",
        2,
    ),
    GateRun(
        "rc_mshnet_no_component",
        ROOT / "configs/phase2_rc_mshnet_no_component_outer_nuaa.yaml",
        3,
    ),
)
ROUND3 = (
    GateRun(
        "rc_mshnet_no_gate_fixed_average",
        ROOT / "configs/phase2_rc_mshnet_no_gate_outer_nuaa.yaml",
        2,
    ),
    GateRun(
        "rc_mshnet_branch_aux",
        ROOT / "configs/phase2_rc_mshnet_branch_aux_outer_nuaa.yaml",
        3,
    ),
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_process_identity(pid: int) -> dict[str, Any]:
    """Read enough immutable Linux process data to detect PID reuse/spoofing."""

    if pid <= 0:
        raise ProcessLookupError(pid)
    proc = Path("/proc") / str(pid)
    stat_text = (proc / "stat").read_text(encoding="utf-8")
    close_paren = stat_text.rfind(")")
    if close_paren < 0:
        raise RuntimeError(f"malformed /proc/{pid}/stat")
    stat_fields = stat_text[close_paren + 2 :].split()
    if len(stat_fields) <= 19:
        raise RuntimeError(f"incomplete /proc/{pid}/stat")
    command_bytes = (proc / "cmdline").read_bytes()
    argv = [
        item.decode("utf-8", errors="surrogateescape")
        for item in command_bytes.split(b"\0")
        if item
    ]
    environment: dict[str, str] = {}
    for item in (proc / "environ").read_bytes().split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        decoded_key = key.decode("utf-8", errors="surrogateescape")
        if decoded_key == PHASE2_LAUNCH_ID_ENV:
            environment[decoded_key] = value.decode(
                "utf-8", errors="surrogateescape"
            )
    return {
        "pid": pid,
        "state": stat_fields[0],
        "start_time_ticks": int(stat_fields[19]),
        "argv": argv,
        "cwd": str((proc / "cwd").resolve()),
        "process_group_id": os.getpgid(pid),
        "session_id": os.getsid(pid),
        "launch_id": environment.get(PHASE2_LAUNCH_ID_ENV),
    }


def _process_identity_is_live(identity: dict[str, Any]) -> bool:
    return str(identity.get("state")) not in {"Z", "X", "x"}


def _wait_for_launch_identity(
    pid: int,
    launch_id: str,
    *,
    process: subprocess.Popen[bytes] | None = None,
    expected_argv: list[str] | None = None,
    expected_cwd: Path | None = None,
    timeout_seconds: float = LAUNCH_IDENTITY_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Wait for exec-time environment identity to become visible in ``/proc``.

    ``subprocess.Popen`` can return while a freshly spawned process still exposes
    its pre-exec environment through ``/proc/<pid>/environ``.  Treat a missing
    launch ID as transient for a short bounded interval, while still rejecting a
    wrong non-empty ID immediately as PID reuse or foreign-process evidence.
    """

    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    while True:
        return_code = process.poll() if process is not None else None
        if return_code is not None:
            raise RuntimeError(
                f"Phase-2 child pid={pid} exited before publishing its complete "
                f"launch identity, return_code={return_code}"
            )
        try:
            identity = _read_process_identity(pid)
        except (FileNotFoundError, ProcessLookupError) as error:
            return_code = process.poll() if process is not None else None
            if return_code is not None:
                raise RuntimeError(
                    f"Phase-2 child pid={pid} exited before publishing its complete "
                    f"launch identity, return_code={return_code}"
                ) from error
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Phase-2 child pid={pid} did not publish a readable identity"
                ) from error
            time.sleep(LAUNCH_IDENTITY_POLL_SECONDS)
            continue
        if not _process_identity_is_live(identity):
            return_code = process.poll() if process is not None else None
            raise RuntimeError(
                f"Phase-2 child pid={pid} exited before publishing its complete "
                f"launch identity, return_code={return_code}"
            )
        observed = identity.get("launch_id")
        complete = (
            observed == launch_id
            and (expected_argv is None or identity.get("argv") == expected_argv)
            and (
                expected_cwd is None
                or identity.get("cwd") == str(expected_cwd.resolve())
            )
            and identity.get("process_group_id") == pid
            and identity.get("session_id") == pid
        )
        if complete:
            return identity
        if observed is not None:
            if observed != launch_id:
                raise RuntimeError(
                    f"Phase-2 child pid={pid} published a foreign launch ID"
                )
        return_code = process.poll() if process is not None else None
        if return_code is not None:
            raise RuntimeError(
                f"Phase-2 child pid={pid} exited before publishing its complete "
                f"launch identity, return_code={return_code}"
            )
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Phase-2 child pid={pid} did not publish its complete launch identity"
            )
        time.sleep(LAUNCH_IDENTITY_POLL_SECONDS)


def require_preflight_hold() -> None:
    if not HOLD.is_file() or ALLOW.exists():
        raise RuntimeError(
            "Phase-2 preflight is fail-closed: HOLD must exist and ALLOW must be absent"
        )


def restore_hold() -> None:
    """Close the gate in the safe order; HOLD becomes visible before ALLOW vanishes."""

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    HOLD.touch(exist_ok=True)
    ALLOW.unlink(missing_ok=True)


def unlock_gate() -> None:
    """Open the gate only from the exact fail-closed preflight state."""

    require_preflight_hold()
    ALLOW.touch(exist_ok=False)
    HOLD.unlink()
    assert_sentinel_allows(STATE_DIR)


def _phase2_command(
    run: GateRun,
    initializer: Path,
    *,
    resume_checkpoint: Path | None = None,
    resume_audit: Path | None = None,
) -> list[str]:
    if (resume_checkpoint is None) != (resume_audit is None):
        raise ValueError(
            "resume_checkpoint and resume_audit must be provided together"
        )
    run_dir = load_gate_output_dir(run.config_path)
    command = [
        sys.executable,
        "-m",
        "rc_irstd.cli.train_detector",
        "--config",
        str(run.config_path),
        "--set",
        "device=cuda:0",
        "--set",
        f"training.initialize_from={initializer.resolve()}",
        "--set",
        f"output_dir={run_dir}",
    ]
    if resume_checkpoint is not None and resume_audit is not None:
        command.extend(
            [
                "--resume-checkpoint",
                str(resume_checkpoint.resolve()),
                "--resume-audit",
                str(resume_audit.resolve()),
            ]
        )
    return command


def _expected_effective_config(run: GateRun, initializer: Path) -> dict[str, Any]:
    payload = yaml.safe_load(run.config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping: {run.config_path}")
    config = deepcopy(payload)
    config["device"] = "cuda:0"
    config["output_dir"] = str(load_gate_output_dir(run.config_path))
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError(f"{run.config_path} has no training mapping")
    training["initialize_from"] = str(initializer.resolve())
    return config


def _load_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return payload


def read_pid(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    return int(path.read_text(encoding="utf-8").strip())


def history_epochs(run_dir: Path) -> int:
    path = run_dir / "history.csv"
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _require_baseline_process_identity(run: BaselineRun, pid: int) -> None:
    identity = _read_process_identity(pid)
    if not _process_identity_is_live(identity):
        raise RuntimeError(f"baseline {run.key} pid={pid} is a zombie")
    if Path(str(identity["cwd"])).resolve() != ROOT:
        raise RuntimeError(f"baseline {run.key} pid={pid} has the wrong cwd")
    command = " ".join(str(value) for value in identity["argv"])
    required_fragments = (
        "rc_irstd.cli.train_detector",
        run.config_path.name,
        str(run.run_dir.resolve()),
    )
    if any(fragment not in command for fragment in required_fragments):
        raise RuntimeError(
            f"baseline {run.key} pid={pid} does not match its registered launcher"
        )


def wait_for_baselines() -> None:
    previous = ""
    while True:
        require_preflight_hold()
        report: list[str] = []
        pending = False
        for run in BASELINES:
            epochs = history_epochs(run.run_dir)
            pid = read_pid(run.pid_file)
            alive = pid_alive(pid)
            if alive:
                _require_baseline_process_identity(run, pid)
            report.append(f"{run.key}={epochs}/400:{'alive' if alive else 'exited'}")
            if epochs < 400:
                pending = True
                if not alive:
                    raise RuntimeError(
                        f"baseline {run.key} exited early at {epochs}/400 (pid={pid})"
                    )
            elif alive:
                pending = True
        rendered = " ".join(report)
        if rendered != previous:
            log(rendered)
            previous = rendered
        if not pending:
            break
        time.sleep(30)

    require_preflight_hold()
    for run in BASELINES:
        output_log = run.run_dir / "train.stdout_stderr.log"
        lines = [
            line.strip()
            for line in output_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        expected_trailer = str((run.run_dir / "last.pt").resolve())
        if not lines or lines[-1] != expected_trailer:
            raise RuntimeError(
                f"baseline {run.key} launcher exited without the successful last.pt trailer"
            )


def verify_expected_sources(run: BaselineRun, manifest: dict[str, Any]) -> None:
    sources = manifest["data"]["sources"]
    actual = tuple(str(item["name"]) for item in sources)
    if actual != run.expected_sources:
        raise ValueError(
            f"baseline {run.key} sources {actual} != {run.expected_sources}"
        )


def write_run_sha_manifest(run_dir: Path) -> None:
    filenames = (
        "last.pt",
        "config.json",
        "history.csv",
        "detector_train.log",
        "train.stdout_stderr.log",
        "checkpoint.sha256",
        "LOSS_IDENTITY.txt",
        "MODEL_IDENTITY.txt",
        "DATA_IDENTITY.txt",
        "IDENTITY_MANIFEST.json",
    )
    lines: list[str] = []
    for filename in filenames:
        path = run_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        lines.append(f"{sha256_file(path)}  {filename}")
    (run_dir / "ARTIFACT_SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_baselines() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for run in BASELINES:
        require_preflight_hold()
        manifest = write_identity(run.run_dir, finalize=True)
        verify_expected_sources(run, manifest)
        write_run_sha_manifest(run.run_dir)
        result[run.key] = manifest
        log(f"finalized baseline {run.key} sha={manifest['checkpoint_sha256']}")
    atomic_json(
        AUDIT_DIR / "phase2_baseline_identity_summary.json",
        {
            "schema_version": "rc-irstd-phase2-baseline-summary-v1",
            "status": "finalized",
            "runs": result,
        },
    )
    require_preflight_hold()
    return result


def run_checked(command: list[str], *, log_path: Path | None = None) -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    handle: TextIO | None = None
    process: subprocess.Popen[bytes] | None = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT if handle is not None else None,
            start_new_session=True,
        )
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
    except BaseException:
        if process is not None and process.poll() is None:
            stop_jobs([process.pid])
            try:
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ChildProcessError):
                pass
        raise
    finally:
        if handle is not None:
            handle.close()


def extract_initializers() -> Path:
    require_preflight_hold()
    INITIALIZER_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pair": INITIALIZER_DIR / "mshnet_seed42_train_nudt_irstd_tensor_only.pt",
        "nuaa": INITIALIZER_DIR / "mshnet_seed42_train_nuaa_tensor_only.pt",
        "nudt": INITIALIZER_DIR / "mshnet_seed42_train_nudt_tensor_only.pt",
        "irstd": INITIALIZER_DIR / "mshnet_seed42_train_irstd_tensor_only.pt",
    }
    by_key = {run.key: run for run in BASELINES}
    for key, target in outputs.items():
        require_preflight_hold()
        run_checked(
            [
                sys.executable,
                str(ROOT / "scripts/extract_mshnet_weights.py"),
                "--input",
                str(by_key[key].run_dir / "last.pt"),
                "--output",
                str(target),
                "--require-stable-sls-v1",
                "--force",
            ]
        )
        require_preflight_hold()
    alias = INITIALIZER_DIR / "nuaa_mshnet_seed42_tensor_only.pt"
    shutil.copyfile(outputs["pair"], alias)
    if sha256_file(alias) != sha256_file(outputs["pair"]):
        raise RuntimeError("outer=NUAA initializer alias is not byte-identical")
    initializer_files = sorted(INITIALIZER_DIR.glob("*.pt"))
    (INITIALIZER_DIR / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in initializer_files),
        encoding="utf-8",
    )
    require_preflight_hold()
    return outputs["pair"]


def preflight_and_unlock(initializer: Path) -> dict[str, Any]:
    require_preflight_hold()
    validate_phase2_configs()
    test_log = AUDIT_DIR / "phase2_preflight_pytest.log"
    run_checked(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            "tests/test_rc_mshnet.py",
            "tests/test_rc_mshnet_main_contract.py",
            "tests/test_phase2_rc_mshnet_gate_contract.py",
        ],
        log_path=test_log,
    )
    require_preflight_hold()
    smoke_log = AUDIT_DIR / "phase2_real_initializer_smoke.json"
    run_checked(
        [
            sys.executable,
            "scripts/smoke_rc_mshnet.py",
            "--device",
            "cpu",
            "--checkpoint",
            str(initializer),
        ],
        log_path=smoke_log,
    )
    require_preflight_hold()
    unlock_gate()
    report = validate_gate_ready(
        state_dir=STATE_DIR,
        initializer=initializer,
        source_manifest=SOURCE_MANIFEST,
    )
    report.update(
        {
            "schema_version": "rc-irstd-phase2-preflight-v1",
            "unlocked_at": now(),
            "pytest_log_sha256": sha256_file(test_log),
            "smoke_report_sha256": sha256_file(smoke_log),
            "ccfa_yaml_required": False,
        }
    )
    atomic_json(AUDIT_DIR / "phase2_preflight.json", report)
    return report


def load_gate_output_dir(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(config_path)
    configured = Path(str(config["output_dir"]))
    return (
        configured.resolve()
        if configured.is_absolute()
        else (config_path.parent / configured).resolve()
    )


def _resume_audit_path(run: GateRun, launch_id: str) -> Path:
    return (
        AUDIT_DIR
        / "phase2_resume_audits"
        / f"{run.role}_{launch_id}.json"
    ).resolve()


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _expected_launcher_command(
    run: GateRun,
    initializer: Path,
    launcher: dict[str, Any],
) -> list[str]:
    """Derive the only command permitted by fresh-v2 or resume-v3 metadata."""

    schema = launcher.get("schema_version")
    if schema not in PHASE2_LAUNCH_SCHEMAS:
        raise RuntimeError(f"{run.role} launcher schema is unsupported")
    launch_id = launcher.get("launch_id")
    if not _is_lower_hex(launch_id, 32):
        raise RuntimeError(f"{run.role} launcher has no valid launch ID")
    if schema == PHASE2_LAUNCH_SCHEMA:
        if launcher.get("launch_mode") not in (None, "fresh"):
            raise RuntimeError(f"{run.role} v2 launcher cannot claim resume mode")
        if launcher.get("resume") is not None:
            raise RuntimeError(f"{run.role} v2 launcher cannot contain resume metadata")
        return _phase2_command(run, initializer)

    if launcher.get("launch_mode") != "exact_resume":
        raise RuntimeError(f"{run.role} v3 launcher must be an exact resume")
    resume = launcher.get("resume")
    if not isinstance(resume, dict):
        raise RuntimeError(f"{run.role} resume launcher has no resume metadata")
    run_dir = load_gate_output_dir(run.config_path)
    checkpoint = (run_dir / "last.pt").resolve()
    audit_path = _resume_audit_path(run, launch_id)
    expected_paths = {
        "source_checkpoint_path": str(checkpoint),
        "audit_path": str(audit_path),
    }
    for key, expected in expected_paths.items():
        if resume.get(key) != expected:
            raise RuntimeError(f"{run.role} resume launcher {key} mismatch")
    source_epoch = resume.get("source_epoch")
    history_rows = resume.get("history_rows_before_resume")
    if (
        not isinstance(source_epoch, int)
        or isinstance(source_epoch, bool)
        or not 0 <= source_epoch < 80
        or history_rows != source_epoch + 1
    ):
        raise RuntimeError(f"{run.role} resume launcher epoch/history mismatch")
    for key in (
        "source_checkpoint_sha256",
        "history_sha256_before_resume",
        "previous_launcher_sha256",
    ):
        if not _is_lower_hex(resume.get(key), 64):
            raise RuntimeError(f"{run.role} resume launcher has invalid {key}")
    previous_launch_id = resume.get("previous_launch_id")
    if not _is_lower_hex(previous_launch_id, 32) or previous_launch_id == launch_id:
        raise RuntimeError(f"{run.role} resume launcher parent ID is invalid")
    return _phase2_command(
        run,
        initializer,
        resume_checkpoint=checkpoint,
        resume_audit=audit_path,
    )


def validate_launcher_binding(
    run: GateRun,
    initializer: Path,
    *,
    require_live: bool,
) -> tuple[dict[str, Any], bool]:
    """Validate launcher metadata and, while live, immutable /proc identity."""

    run_dir = load_gate_output_dir(run.config_path)
    launcher_path = run_dir / "launcher.json"
    launcher = _load_json_mapping(launcher_path)
    schema = launcher.get("schema_version")
    if schema not in PHASE2_LAUNCH_SCHEMAS:
        raise RuntimeError(f"{run.role} launcher schema is unsupported")
    expected_scalars = {
        "role": run.role,
        "physical_gpu": run.gpu,
        "visible_device": "cuda:0",
        "config_path": str(run.config_path.resolve()),
        "config_sha256": sha256_file(run.config_path),
        "initializer_path": str(initializer.resolve()),
        "initializer_sha256": sha256_file(initializer),
        "expected_output_dir": str(run_dir),
    }
    for key, expected in expected_scalars.items():
        if launcher.get(key) != expected:
            raise RuntimeError(
                f"{run.role} launcher {key} does not match the current frozen input"
            )
    launch_id = launcher.get("launch_id")
    if not _is_lower_hex(launch_id, 32):
        raise RuntimeError(f"{run.role} launcher has no valid launch ID")
    expected_command = _expected_launcher_command(run, initializer, launcher)
    if launcher.get("command") != expected_command:
        raise RuntimeError(f"{run.role} launcher command mismatch")
    pid = launcher.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        raise RuntimeError(f"{run.role} launcher has no valid PID")
    recorded_identity = launcher.get("process_identity")
    if not isinstance(recorded_identity, dict):
        raise RuntimeError(f"{run.role} launcher has no process identity")

    try:
        current_identity = _read_process_identity(pid)
    except (FileNotFoundError, ProcessLookupError):
        current_identity = None
    live = bool(
        current_identity is not None and _process_identity_is_live(current_identity)
    )
    if live:
        immutable_keys = (
            "pid",
            "start_time_ticks",
            "argv",
            "cwd",
            "process_group_id",
            "session_id",
            "launch_id",
        )
        for key in immutable_keys:
            if current_identity.get(key) != recorded_identity.get(key):
                raise RuntimeError(
                    f"{run.role} live PID identity mismatch for {key}; "
                    "refusing PID reuse or a foreign launcher"
                )
        if current_identity.get("argv") != expected_command:
            raise RuntimeError(f"{run.role} live process argv mismatch")
        if current_identity.get("cwd") != str(ROOT):
            raise RuntimeError(f"{run.role} live process cwd mismatch")
        if current_identity.get("process_group_id") != pid:
            raise RuntimeError(f"{run.role} is not the leader of its process group")
        if current_identity.get("session_id") != pid:
            raise RuntimeError(f"{run.role} is not isolated in its own session")
        if current_identity.get("launch_id") != launch_id:
            raise RuntimeError(f"{run.role} live process launch ID mismatch")
    elif require_live:
        raise RuntimeError(f"{run.role} launcher PID {pid} is not live")
    return launcher, live


def _read_exact_history_epochs(run_dir: Path, expected_rows: int) -> list[int]:
    path = run_dir / "history.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if (
        not isinstance(fieldnames, list)
        or len(fieldnames) != len(set(fieldnames))
        or not {"epoch", "lr", "loss_total"}.issubset(fieldnames)
    ):
        raise ValueError(f"{path} has an incomplete or duplicate CSV header")
    try:
        epochs = [int(row["epoch"]) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid epoch column in {path}") from error
    expected = list(range(expected_rows))
    if epochs != expected:
        raise ValueError(
            f"{path} epochs must be exactly 0..{expected_rows - 1}, received "
            f"{epochs[:5]}...{epochs[-5:]}"
        )
    for row_index, row in enumerate(rows):
        for field in fieldnames:
            raw = row.get(field)
            if raw is None or not raw.strip():
                raise ValueError(f"{path} row {row_index} has an empty {field}")
            if field == "epoch":
                continue
            try:
                value = float(raw)
            except ValueError as error:
                raise ValueError(
                    f"{path} row {row_index} has a non-numeric {field}"
                ) from error
            if not math.isfinite(value):
                raise ValueError(
                    f"{path} row {row_index} has a non-finite {field}"
                )
    return epochs


def _validate_checkpoint_source_splits(
    run: GateRun,
    expected_config: dict[str, Any],
    records: Any,
) -> list[str]:
    sources = expected_config.get("data", {}).get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"{run.role} frozen config has no source domains")
    expected_names = [str(source.get("name", "")) for source in sources]
    if expected_names != ["NUDT-SIRST", "IRSTD-1K"]:
        raise ValueError(
            f"{run.role} must train on exactly NUDT-SIRST + IRSTD-1K"
        )
    if not isinstance(records, list) or len(records) != len(sources):
        raise ValueError(f"{run.role} frozen source-split record count mismatch")
    for source, record, expected_name in zip(sources, records, expected_names):
        if not isinstance(source, dict) or not isinstance(record, dict):
            raise ValueError(f"{run.role} source/split entry is not a mapping")
        source_path = Path(str(source.get("path", ""))).expanduser()
        if not source_path.is_absolute():
            source_path = (run.config_path.parent / source_path).resolve()
        else:
            source_path = source_path.resolve()
        if record.get("name") != expected_name:
            raise ValueError(f"{run.role} source record name mismatch")
        if Path(str(record.get("path", ""))).resolve() != source_path:
            raise ValueError(f"{run.role} source record path mismatch")
        train_path = Path(str(record.get("train_split_file", ""))).resolve()
        test_path = Path(str(record.get("test_split_file", ""))).resolve()
        if not train_path.is_file() or not test_path.is_file():
            raise FileNotFoundError(
                f"{run.role} frozen train/test split file is missing"
            )
        if sha256_file(train_path) != record.get("train_split_file_sha256"):
            raise ValueError(f"{run.role} train split SHA mismatch")
        if sha256_file(test_path) != record.get("test_split_file_sha256"):
            raise ValueError(f"{run.role} test split SHA mismatch")
        train_ids = ensure_unique_sample_ids(read_split_file(train_path))
        test_ids = ensure_unique_sample_ids(read_split_file(test_path))
        if ordered_ids_sha256(train_ids) != record.get("train_ordered_ids_sha256"):
            raise ValueError(f"{run.role} train ordered-ID SHA mismatch")
        if ordered_ids_sha256(test_ids) != record.get("test_ordered_ids_sha256"):
            raise ValueError(f"{run.role} test ordered-ID SHA mismatch")
        if int(record.get("num_train_samples", -1)) != len(train_ids):
            raise ValueError(f"{run.role} train sample count mismatch")
        if int(record.get("num_test_samples", -1)) != len(test_ids):
            raise ValueError(f"{run.role} test sample count mismatch")
        if record.get("train_test_id_overlap") is not False:
            raise ValueError(f"{run.role} split record permits train/test overlap")
        overlap = set(train_ids).intersection(test_ids)
        if overlap:
            raise ValueError(f"{run.role} frozen train/test IDs overlap")
    return expected_names


def _validate_interrupted_resume_audit(
    run: GateRun,
    launcher: dict[str, Any],
    *,
    checkpoint_sha256: str,
    checkpoint_epoch: int,
    history_rows: int,
    history_sha256: str,
) -> None:
    """Reject a cleanly failed prior resume; only abrupt interruption is retryable."""

    if launcher.get("schema_version") != PHASE2_RESUME_LAUNCH_SCHEMA:
        return
    resume = launcher.get("resume")
    if not isinstance(resume, dict):
        raise RuntimeError(f"{run.role} resume launcher has no resume metadata")
    audit_path = Path(str(resume["audit_path"]))
    if not audit_path.is_file():
        # The child may have died between exec identity publication and the
        # resume CLI's first atomic audit write.  In that case no committed
        # training state may differ from the launch source.
        if (
            checkpoint_sha256 != resume["source_checkpoint_sha256"]
            or checkpoint_epoch != resume["source_epoch"]
            or history_rows != resume["history_rows_before_resume"]
            or history_sha256 != resume["history_sha256_before_resume"]
        ):
            raise RuntimeError(
                f"{run.role} progressed without its required resume audit"
            )
        return
    audit = _load_json_mapping(audit_path)
    if audit.get("schema_version") != FORMAL_RESUME_AUDIT_SCHEMA:
        raise RuntimeError(f"{run.role} prior resume audit schema mismatch")
    audit_status = audit.get("status")
    if audit_status not in {"prepared", "running", "completed"}:
        raise RuntimeError(
            f"{run.role} prior resume ended with audit status "
            f"{audit.get('status')!r}; refusing automatic retry"
        )
    events = audit.get("events")
    if not isinstance(events, list) or len(events) != 1:
        raise RuntimeError(f"{run.role} prior resume audit event count mismatch")
    event = events[0]
    if not isinstance(event, dict):
        raise RuntimeError(f"{run.role} prior resume audit event is invalid")
    expected = {
        "source_checkpoint_path": resume["source_checkpoint_path"],
        "source_checkpoint_sha256": resume["source_checkpoint_sha256"],
        "source_epoch": resume["source_epoch"],
        "history_rows_before_resume": resume["history_rows_before_resume"],
        "history_sha256_before_resume": resume["history_sha256_before_resume"],
        "cuda_visible_devices": str(run.gpu),
        "physical_gpu_index": run.gpu,
    }
    for key, value in expected.items():
        if event.get(key) != value:
            raise RuntimeError(f"{run.role} prior resume audit {key} mismatch")
    event_status = event.get("status")
    if (audit_status == "completed") != (event_status == "completed"):
        raise RuntimeError(f"{run.role} prior resume audit status is inconsistent")
    if event_status == "completed":
        if (
            audit_status != "completed"
            or checkpoint_epoch != 79
            or history_rows != 80
            or event.get("final_checkpoint_sha256") != checkpoint_sha256
            or event.get("final_checkpoint_path")
            != str((load_gate_output_dir(run.config_path) / "last.pt").resolve())
        ):
            raise RuntimeError(
                f"{run.role} completed prior resume audit does not bind the final state"
            )
    elif event_status not in {"prepared", "running"}:
        raise RuntimeError(f"{run.role} prior resume audit event is not interrupted")


def inspect_partial_resume(
    run: GateRun,
    initializer: Path,
) -> PartialResumePlan:
    """Validate, without mutation, an exact epoch-boundary Phase-2 resume."""

    run_dir = load_gate_output_dir(run.config_path)
    launcher_path = run_dir / "launcher.json"
    launcher, live = validate_launcher_binding(
        run, initializer, require_live=False
    )
    process_group_id = int(
        launcher.get("process_identity", {}).get("process_group_id", 0)
    )
    if live or _process_group_alive(process_group_id):
        raise RuntimeError(f"refusing to resume live {run.role} launcher/process group")
    if (run_dir / "last.pt.tmp").exists():
        raise RuntimeError(f"{run.role} has an incomplete checkpoint temporary file")
    checkpoint = (run_dir / "last.pt").resolve()
    history_path = run_dir / "history.csv"
    if not checkpoint.is_file() or not history_path.is_file():
        raise RuntimeError(
            f"{run.role} partial resume requires both history.csv and last.pt"
        )
    history_rows = history_epochs(run_dir)
    if not 1 <= history_rows <= 80:
        raise RuntimeError(
            f"{run.role} partial resume requires 1..80 rows, got {history_rows}"
        )
    _read_exact_history_epochs(run_dir, history_rows)

    expected_config = _expected_effective_config(run, initializer)
    recorded_config = _load_json_mapping(run_dir / "config.json")
    if recorded_config != expected_config:
        raise RuntimeError(f"{run.role} partial config.json drifted")
    initialization_report = _load_json_mapping(
        run_dir / "initialization_report.json"
    )
    if (
        initialization_report.get("source_path") != str(initializer.resolve())
        or initialization_report.get("source_sha256") != sha256_file(initializer)
        or initialization_report.get("backbone_fully_loaded") is not True
        or initialization_report.get("zero_residual_identity_preserved") is not True
    ):
        raise RuntimeError(f"{run.role} partial initializer binding mismatch")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{run.role} partial checkpoint root is not a mapping")
    epoch = payload.get("epoch")
    if (
        payload.get("format_version") != 2
        or payload.get("kind") != "detector"
        or not isinstance(epoch, int)
        or isinstance(epoch, bool)
        or epoch != history_rows - 1
    ):
        raise RuntimeError(f"{run.role} partial checkpoint/history epoch mismatch")
    if payload.get("config") != expected_config:
        raise RuntimeError(f"{run.role} partial embedded config drifted")
    training = expected_config.get("training")
    if (
        not isinstance(training, dict)
        or training.get("resume") is not None
        or training.get("initialize_from") != str(initializer.resolve())
        or int(training.get("epochs", -1)) != 80
        or int(training.get("warmup_epochs", -1)) != 0
    ):
        raise RuntimeError(f"{run.role} partial formal training contract drifted")
    if payload.get("checkpoint_selection") != "fixed_last":
        raise RuntimeError(f"{run.role} partial checkpoint is not fixed-last")
    if (
        bool(payload.get("test_labels_used_for_selection", True))
        or bool(payload.get("diagnostic_test_eval", True))
        or bool(payload.get("diagnostic_only", True))
        or payload.get("formal_paper_checkpoint") is not True
    ):
        raise RuntimeError(f"{run.role} partial checkpoint is not formal-causal")
    if (
        payload.get("inference_head") != "multi_scale_fused"
        or payload.get("warm_flag") is not True
    ):
        raise RuntimeError(f"{run.role} partial checkpoint has the wrong inference head")
    if payload.get("initialization") != initialization_report:
        raise RuntimeError(f"{run.role} partial initialization record drifted")
    required = {
        "model_state",
        "model_config",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "rng_state",
        "balanced_batcher_state",
        "resume_contract",
        "source_split_records",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise RuntimeError(
            f"{run.role} partial checkpoint is missing: {', '.join(missing)}"
        )
    rng_state = payload.get("rng_state")
    if not isinstance(rng_state, dict) or len(rng_state.get("torch_cuda", [])) != 1:
        raise RuntimeError(f"{run.role} partial checkpoint CUDA RNG state mismatch")
    expected_source_names = _validate_checkpoint_source_splits(
        run, expected_config, payload.get("source_split_records")
    )
    if payload.get("source_names") != expected_source_names:
        raise RuntimeError(f"{run.role} partial source-domain identity mismatch")
    expected_model = build_mshnet(expected_config["model"])
    expected_model_config = dict(expected_model.export_config())
    if payload.get("model_config") != expected_model_config:
        raise RuntimeError(f"{run.role} partial model contract drifted")
    try:
        expected_model.load_state_dict(payload["model_state"], strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"{run.role} partial model state does not strict-load"
        ) from error
    finally:
        del expected_model
    resume_contract = payload.get("resume_contract")
    if (
        not isinstance(resume_contract, dict)
        or resume_contract.get("model") != expected_model_config
    ):
        raise RuntimeError(f"{run.role} partial resume/model contract mismatch")
    resume_training = resume_contract.get("training")
    if (
        not isinstance(resume_training, dict)
        or int(resume_training.get("epochs", -1)) != 80
        or int(resume_training.get("warmup_epochs", -1)) != 0
    ):
        raise RuntimeError(f"{run.role} partial resume schedule mismatch")

    checkpoint_sha256 = sha256_file(checkpoint)
    history_sha256 = sha256_file(history_path)
    launcher_sha256 = sha256_file(launcher_path)
    _validate_interrupted_resume_audit(
        run,
        launcher,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_epoch=epoch,
        history_rows=history_rows,
        history_sha256=history_sha256,
    )
    # Re-read the small mutable identities last.  A mismatch indicates a TOCTOU
    # or an unregistered writer and must never be papered over by a resume.
    if (
        checkpoint_sha256 != sha256_file(checkpoint)
        or history_sha256 != sha256_file(history_path)
        or launcher_sha256 != sha256_file(launcher_path)
    ):
        raise RuntimeError(f"{run.role} partial evidence changed during inspection")
    return PartialResumePlan(
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_epoch=epoch,
        history_rows=history_rows,
        history_sha256=history_sha256,
        previous_launch_id=str(launcher["launch_id"]),
        previous_launcher_sha256=launcher_sha256,
    )


def gate_checkpoint_complete(run: GateRun) -> bool:
    run_dir = load_gate_output_dir(run.config_path)
    if history_epochs(run_dir) != 80 or not (run_dir / "last.pt").is_file():
        return False
    payload = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    return isinstance(payload, dict) and int(payload.get("epoch", -1)) == 79


def _archive_zero_epoch_attempt(
    run: GateRun,
    initializer: Path,
    *,
    reason: str,
    launcher: dict[str, Any] | None,
) -> Path:
    """Preserve and remove a provably zero-epoch failed launch from the run path.

    Any directory containing a checkpoint or even one history row remains
    fail-closed.  A launcher-less directory is recoverable only when it is the
    empty log created before launcher metadata is committed.  Identity-verified
    dead launchers may contain initialization sidecars, all of which are archived
    intact before a fresh attempt is allowed.
    """

    require_preflight_hold()
    run_dir = load_gate_output_dir(run.config_path)
    if not run_dir.is_dir() or not any(run_dir.iterdir()):
        raise RuntimeError(f"no failed Phase-2 artifacts to archive: {run_dir}")
    if (
        (run_dir / "history.csv").exists()
        or (run_dir / "last.pt").exists()
        or (run_dir / "last.pt.tmp").exists()
        or (run_dir / "PHASE2_IDENTITY.json").exists()
    ):
        raise RuntimeError(
            f"refusing zero-epoch recovery for {run.role}: training state exists"
        )

    directories = sorted(path for path in run_dir.rglob("*") if path.is_dir())
    if directories:
        raise RuntimeError(
            f"refusing zero-epoch recovery for {run.role}: unknown directories exist"
        )
    files = sorted(path for path in run_dir.rglob("*") if path.is_file())
    if launcher is None:
        relative_files = [str(path.relative_to(run_dir)) for path in files]
        if relative_files != ["train.stdout_stderr.log"] or files[0].stat().st_size != 0:
            raise RuntimeError(
                f"refusing unbound Phase-2 recovery for {run.role}: "
                "artifacts are not the empty pre-launch log"
            )
        attempt_id = f"unbound_{uuid.uuid4().hex}"
        launcher_summary: dict[str, Any] | None = None
    else:
        allowed = {
            "launcher.json",
            "config.json",
            "initialization_report.json",
            "detector_train.log",
            "train.stdout_stderr.log",
        }
        relative_files = {str(path.relative_to(run_dir)) for path in files}
        if not relative_files.issubset(allowed) or "launcher.json" not in relative_files:
            raise RuntimeError(
                f"refusing bound Phase-2 recovery for {run.role}: unknown artifacts exist"
            )
        config_path = run_dir / "config.json"
        if config_path.is_file() and _load_json_mapping(config_path) != _expected_effective_config(
            run, initializer
        ):
            raise RuntimeError(f"{run.role} zero-epoch config binding mismatch")
        initialization_path = run_dir / "initialization_report.json"
        if initialization_path.is_file():
            initialization = _load_json_mapping(initialization_path)
            if (
                initialization.get("source_path") != str(initializer.resolve())
                or initialization.get("source_sha256") != sha256_file(initializer)
                or initialization.get("backbone_fully_loaded") is not True
                or initialization.get("zero_residual_identity_preserved") is not True
            ):
                raise RuntimeError(
                    f"{run.role} zero-epoch initializer binding mismatch"
                )
        launch_id = launcher.get("launch_id")
        if not isinstance(launch_id, str) or len(launch_id) != 32:
            raise RuntimeError(f"{run.role} recovery launcher has no valid launch ID")
        attempt_id = launch_id
        launcher_summary = {
            "launch_id": launch_id,
            "pid": launcher.get("pid"),
            "launcher_sha256": sha256_file(run_dir / "launcher.json"),
        }

    recovery = {
        "schema_version": PHASE2_RECOVERY_SCHEMA,
        "status": "archived_zero_epoch_attempt",
        "role": run.role,
        "reason": reason,
        "archived_at": now(),
        "config_path": str(run.config_path.resolve()),
        "config_sha256": sha256_file(run.config_path),
        "initializer_path": str(initializer.resolve()),
        "initializer_sha256": sha256_file(initializer),
        "launcher": launcher_summary,
        "files_before_archive": {
            str(path.relative_to(run_dir)): sha256_file(path) for path in files
        },
    }
    archive_root = AUDIT_DIR / "phase2_failed_launches"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_dir = archive_root / f"{run.role}_{attempt_id}"
    if archive_dir.exists():
        raise RuntimeError(f"refusing to overwrite recovery archive: {archive_dir}")
    atomic_json(run_dir / "RECOVERY.json", recovery)
    require_preflight_hold()
    run_dir.rename(archive_dir)
    log(f"archived zero-epoch {run.role} attempt at {archive_dir}")
    require_preflight_hold()
    return archive_dir


def recover_uncommitted_phase2_attempts(
    initializer: Path,
    *,
    runs: tuple[GateRun, ...] | None = None,
) -> list[Path]:
    """Archive strictly proven zero-epoch attempts while the gate is closed."""

    require_preflight_hold()
    selected = runs or (*ROUND1, *ROUND2, *ROUND3)
    archived: list[Path] = []
    for run in selected:
        require_preflight_hold()
        run_dir = load_gate_output_dir(run.config_path)
        if not run_dir.exists() or not any(run_dir.iterdir()):
            continue
        if gate_checkpoint_complete(run):
            continue
        launcher_path = run_dir / "launcher.json"
        if launcher_path.is_file():
            launcher, live = validate_launcher_binding(
                run, initializer, require_live=False
            )
            process_group_id = int(
                launcher.get("process_identity", {}).get("process_group_id", 0)
            )
            if live or _process_group_alive(process_group_id):
                log(
                    f"preserved identity-verified live {run.role} launcher "
                    "for attach"
                )
                continue
            if (run_dir / "history.csv").exists() or (run_dir / "last.pt").exists():
                plan = inspect_partial_resume(run, initializer)
                log(
                    f"validated exact-resume candidate {run.role} "
                    f"epoch={plan.checkpoint_epoch} sha={plan.checkpoint_sha256}"
                )
                continue
            archived.append(
                _archive_zero_epoch_attempt(
                    run,
                    initializer,
                    reason="identity_verified_launcher_pid_not_live",
                    launcher=launcher,
                )
            )
        else:
            archived.append(
                _archive_zero_epoch_attempt(
                    run,
                    initializer,
                    reason="unbound_empty_prelaunch_log",
                    launcher=None,
                )
            )
    require_preflight_hold()
    return archived


def _assert_resume_plan_unchanged(
    run: GateRun,
    plan: PartialResumePlan,
) -> None:
    run_dir = load_gate_output_dir(run.config_path)
    launcher_path = run_dir / "launcher.json"
    history_path = run_dir / "history.csv"
    if plan.checkpoint != (run_dir / "last.pt").resolve():
        raise RuntimeError(f"{run.role} resume checkpoint path changed")
    if (
        sha256_file(plan.checkpoint) != plan.checkpoint_sha256
        or sha256_file(history_path) != plan.history_sha256
        or sha256_file(launcher_path) != plan.previous_launcher_sha256
        or history_epochs(run_dir) != plan.history_rows
        or (run_dir / "last.pt.tmp").exists()
    ):
        raise RuntimeError(f"{run.role} partial evidence changed before resume launch")
    launcher = _load_json_mapping(launcher_path)
    if launcher.get("launch_id") != plan.previous_launch_id:
        raise RuntimeError(f"{run.role} previous launch identity changed")


def _archive_superseded_launcher(
    run: GateRun,
    plan: PartialResumePlan,
) -> Path:
    """Copy, but never remove, the launcher superseded by an exact resume."""

    _assert_resume_plan_unchanged(run, plan)
    launcher_path = load_gate_output_dir(run.config_path) / "launcher.json"
    archive_dir = (
        AUDIT_DIR
        / "phase2_superseded_launchers"
        / f"{run.role}_{plan.previous_launch_id}"
    )
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived = archive_dir / "launcher.json"
    if archived.is_file():
        if sha256_file(archived) != plan.previous_launcher_sha256:
            raise RuntimeError(f"refusing to overwrite launcher archive: {archived}")
        return archived
    temporary = archived.with_suffix(archived.suffix + ".tmp")
    shutil.copyfile(launcher_path, temporary)
    if sha256_file(temporary) != plan.previous_launcher_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"failed to preserve superseded launcher: {launcher_path}")
    temporary.replace(archived)
    return archived


def _launch_phase2(
    run: GateRun,
    initializer: Path,
    *,
    resume: PartialResumePlan | None,
) -> tuple[subprocess.Popen[bytes], int]:
    assert_sentinel_allows(STATE_DIR)
    run_dir = load_gate_output_dir(run.config_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    launch_id = uuid.uuid4().hex
    if resume is None:
        command = _phase2_command(run, initializer)
        schema = PHASE2_LAUNCH_SCHEMA
        launch_mode: str | None = None
        resume_metadata: dict[str, Any] | None = None
        log_mode = "w"
    else:
        _assert_resume_plan_unchanged(run, resume)
        _archive_superseded_launcher(run, resume)
        resume_audit = _resume_audit_path(run, launch_id)
        command = _phase2_command(
            run,
            initializer,
            resume_checkpoint=resume.checkpoint,
            resume_audit=resume_audit,
        )
        schema = PHASE2_RESUME_LAUNCH_SCHEMA
        launch_mode = "exact_resume"
        resume_metadata = {
            "source_checkpoint_path": str(resume.checkpoint.resolve()),
            "source_checkpoint_sha256": resume.checkpoint_sha256,
            "source_epoch": resume.checkpoint_epoch,
            "history_rows_before_resume": resume.history_rows,
            "history_sha256_before_resume": resume.history_sha256,
            "previous_launch_id": resume.previous_launch_id,
            "previous_launcher_sha256": resume.previous_launcher_sha256,
            "audit_path": str(resume_audit),
        }
        log_mode = "a"
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(run.gpu),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            PHASE2_LAUNCH_ID_ENV: launch_id,
        }
    )
    log_handle: TextIO = (run_dir / "train.stdout_stderr.log").open(
        log_mode, encoding="utf-8"
    )
    process: subprocess.Popen[bytes] | None = None
    launcher_committed = False
    launcher_path = run_dir / "launcher.json"
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        process_identity = _wait_for_launch_identity(
            process.pid,
            launch_id,
            process=process,
            expected_argv=command,
            expected_cwd=ROOT,
        )
        if process_identity.get("process_group_id") != process.pid:
            raise RuntimeError(f"{run.role} child has no isolated process group")
        if process_identity.get("session_id") != process.pid:
            raise RuntimeError(f"{run.role} child has no isolated session")
        launcher_payload: dict[str, Any] = {
            "schema_version": schema,
            "role": run.role,
            "pid": process.pid,
            "launch_id": launch_id,
            "physical_gpu": run.gpu,
            "visible_device": "cuda:0",
            "command": command,
            "config_path": str(run.config_path.resolve()),
            "config_sha256": sha256_file(run.config_path),
            "initializer_path": str(initializer.resolve()),
            "initializer_sha256": sha256_file(initializer),
            "expected_output_dir": str(run_dir),
            "process_identity": process_identity,
            "launched_at": now(),
        }
        if launch_mode is not None:
            launcher_payload["launch_mode"] = launch_mode
            launcher_payload["resume"] = resume_metadata
        atomic_json(launcher_path, launcher_payload)
        launcher_committed = True
        log(
            f"launched {run.role} pid={process.pid} gpu={run.gpu} "
            f"mode={launch_mode or 'fresh'}"
        )
        return process, process.pid
    except BaseException:
        if process is not None:
            stop_jobs([process.pid])
            try:
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ChildProcessError):
                pass
        if not launcher_committed and resume is None:
            launcher_path.unlink(missing_ok=True)
        raise
    finally:
        log_handle.close()


def start_or_attach(run: GateRun, initializer: Path) -> tuple[subprocess.Popen[bytes] | None, int]:
    assert_sentinel_allows(STATE_DIR)
    run_dir = load_gate_output_dir(run.config_path)
    if gate_checkpoint_complete(run):
        launcher, live = validate_launcher_binding(
            run, initializer, require_live=False
        )
        if live:
            pid = int(launcher["pid"])
            try:
                log(f"attached to finalizing identity-verified {run.role} pid={pid}")
            except BaseException:
                stop_jobs([pid])
                raise
            return None, pid
        try:
            verify_gate_checkpoint(run, sha256_file(initializer), initializer)
        except (FileNotFoundError, RuntimeError, ValueError) as verification_error:
            plan = inspect_partial_resume(run, initializer)
            if plan.checkpoint_epoch != 79 or plan.history_rows != 80:
                raise RuntimeError(
                    f"{run.role} final checkpoint verification failed and is not "
                    "an exact finalization candidate"
                ) from verification_error
            log(
                f"resuming finalization-only {run.role} after verification error: "
                f"{verification_error}"
            )
            return _launch_phase2(run, initializer, resume=plan)
        return None, 0
    launcher_path = run_dir / "launcher.json"
    if launcher_path.is_file():
        launcher, live = validate_launcher_binding(
            run, initializer, require_live=False
        )
        if live:
            pid = int(launcher["pid"])
            try:
                log(f"attached to identity-verified {run.role} pid={pid}")
            except BaseException:
                stop_jobs([pid])
                raise
            return None, pid
        plan = inspect_partial_resume(run, initializer)
        return _launch_phase2(run, initializer, resume=plan)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty Phase-2 directory: {run_dir}")
    return _launch_phase2(run, initializer, resume=None)


def _process_group_alive(process_group_id: int) -> bool:
    if process_group_id <= 0:
        return False
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_jobs(pids: list[int], *, grace_seconds: float = 5.0) -> None:
    """Terminate complete isolated process groups and escalate if needed."""

    process_groups = sorted({pid for pid in pids if pid > 0})
    for process_group_id in process_groups:
        if _process_group_alive(process_group_id):
            try:
                os.killpg(process_group_id, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + max(grace_seconds, 0.0)
    while any(_process_group_alive(group) for group in process_groups):
        if time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    for process_group_id in process_groups:
        if _process_group_alive(process_group_id):
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass


def validate_resume_audit(
    run: GateRun,
    launcher: dict[str, Any],
    final_checkpoint_sha256: str,
) -> dict[str, Any] | None:
    """Bind a completed resume-v3 launcher to its immutable CLI audit."""

    if launcher.get("schema_version") == PHASE2_LAUNCH_SCHEMA:
        return None
    if launcher.get("schema_version") != PHASE2_RESUME_LAUNCH_SCHEMA:
        raise RuntimeError(f"{run.role} completed with an unsupported launcher")
    resume = launcher.get("resume")
    if not isinstance(resume, dict):
        raise RuntimeError(f"{run.role} completed resume has no metadata")
    audit_path = Path(str(resume.get("audit_path", ""))).resolve()
    if audit_path != _resume_audit_path(run, str(launcher["launch_id"])):
        raise RuntimeError(f"{run.role} completed resume audit path mismatch")
    audit = _load_json_mapping(audit_path)
    if (
        audit.get("schema_version") != FORMAL_RESUME_AUDIT_SCHEMA
        or audit.get("status") != "completed"
    ):
        raise RuntimeError(f"{run.role} resume audit is not completed")
    events = audit.get("events")
    if not isinstance(events, list) or len(events) != 1:
        raise RuntimeError(f"{run.role} resume audit event count mismatch")
    event = events[0]
    if not isinstance(event, dict) or event.get("status") != "completed":
        raise RuntimeError(f"{run.role} resume audit event is not completed")
    run_dir = load_gate_output_dir(run.config_path)
    expected = {
        "owner_pid": launcher["pid"],
        "process_group_id": launcher["pid"],
        "session_id": launcher["pid"],
        "source_checkpoint_path": resume["source_checkpoint_path"],
        "source_checkpoint_sha256": resume["source_checkpoint_sha256"],
        "source_epoch": resume["source_epoch"],
        "resume_start_epoch": int(resume["source_epoch"]) + 1,
        "history_rows_before_resume": resume["history_rows_before_resume"],
        "history_sha256_before_resume": resume["history_sha256_before_resume"],
        "physical_gpu_index": run.gpu,
        "logical_device": "cuda:0",
        "final_checkpoint_path": str((run_dir / "last.pt").resolve()),
        "final_checkpoint_sha256": final_checkpoint_sha256,
        "formal_config_sha256": sha256_file(run_dir / "config.json"),
    }
    for key, value in expected.items():
        if event.get(key) != value:
            raise RuntimeError(f"{run.role} resume audit {key} mismatch")
    snapshot = Path(str(event.get("immutable_snapshot_path", ""))).resolve()
    expected_snapshot = (
        run_dir
        / "resume_snapshots"
        / (
            f"epoch_{int(resume['source_epoch']):04d}_"
            f"{str(resume['source_checkpoint_sha256'])[:16]}.pt"
        )
    ).resolve()
    if (
        snapshot != expected_snapshot
        or not snapshot.is_file()
        or sha256_file(snapshot) != resume["source_checkpoint_sha256"]
    ):
        raise RuntimeError(f"{run.role} immutable resume snapshot mismatch")
    return {
        "path": str(audit_path),
        "sha256": sha256_file(audit_path),
        "source_checkpoint_sha256": resume["source_checkpoint_sha256"],
        "source_epoch": resume["source_epoch"],
        "immutable_snapshot_path": str(snapshot),
    }


def verify_gate_checkpoint(
    run: GateRun,
    initializer_sha256: str,
    initializer: Path,
) -> dict[str, Any]:
    run_dir = load_gate_output_dir(run.config_path)
    _read_exact_history_epochs(run_dir, 80)
    launcher, _ = validate_launcher_binding(
        run, initializer, require_live=False
    )
    expected_config = _expected_effective_config(run, initializer)
    recorded_config = _load_json_mapping(run_dir / "config.json")
    if recorded_config != expected_config:
        raise ValueError(f"{run.role} config.json does not match the frozen config")
    initialization_report = _load_json_mapping(
        run_dir / "initialization_report.json"
    )
    expected_initializer_path = str(initializer.resolve())
    if initialization_report.get("source_sha256") != initializer_sha256:
        raise ValueError(f"{run.role} initialization report SHA mismatch")
    if initialization_report.get("source_path") != expected_initializer_path:
        raise ValueError(f"{run.role} initialization report path mismatch")
    if initialization_report.get("backbone_fully_loaded") is not True:
        raise ValueError(f"{run.role} did not fully initialize the MSHNet backbone")
    if initialization_report.get("zero_residual_identity_preserved") is not True:
        raise ValueError(f"{run.role} did not preserve zero-residual identity")

    checkpoint = run_dir / "last.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or int(payload.get("epoch", -1)) != 79:
        raise ValueError(f"{run.role} has no epoch-79 checkpoint")
    if payload.get("format_version") != 2:
        raise ValueError(f"{run.role} checkpoint format_version is unsupported")
    if payload.get("kind") != "detector":
        raise ValueError(f"{run.role} artifact is not a detector checkpoint")
    if payload.get("checkpoint_selection") != "fixed_last":
        raise ValueError(f"{run.role} is not fixed-last")
    if bool(payload.get("test_labels_used_for_selection", True)):
        raise ValueError(f"{run.role} used test labels for checkpoint selection")
    if bool(payload.get("diagnostic_test_eval", True)):
        raise ValueError(f"{run.role} enabled diagnostic target-label evaluation")
    if bool(payload.get("diagnostic_only", True)):
        raise ValueError(f"{run.role} is marked diagnostic-only")
    if payload.get("formal_paper_checkpoint") is not True:
        raise ValueError(f"{run.role} is not marked as a formal checkpoint")
    if (
        payload.get("inference_head") != "multi_scale_fused"
        or payload.get("warm_flag") is not True
    ):
        raise ValueError(f"{run.role} has the wrong final inference head")
    initialization = payload.get("initialization")
    if not isinstance(initialization, dict):
        raise ValueError(f"{run.role} checkpoint has no initialization record")
    if initialization != initialization_report:
        raise ValueError(f"{run.role} checkpoint initialization record drifted")
    if initialization.get("source_sha256") != initializer_sha256:
        raise ValueError(f"{run.role} does not use the shared initializer")
    if initialization.get("source_path") != expected_initializer_path:
        raise ValueError(f"{run.role} checkpoint uses a different initializer path")
    config = payload.get("config")
    if not isinstance(config, dict) or config != expected_config:
        raise ValueError(f"{run.role} embedded config does not match the frozen config")
    identity = config.get("experiment_identity")
    if not isinstance(identity, dict) or identity.get("run_role") != run.role:
        raise ValueError(f"{run.role} checkpoint identity mismatch")
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError(f"{run.role} embedded config has no training mapping")
    if training.get("resume") is not None:
        raise ValueError(f"{run.role} unexpectedly resumed another checkpoint")
    if training.get("initialize_from") != expected_initializer_path:
        raise ValueError(f"{run.role} embedded initializer path mismatch")
    if int(training.get("epochs", -1)) != 80 or int(training.get("warmup_epochs", -1)) != 0:
        raise ValueError(f"{run.role} training schedule mismatch")
    expected_source_names = _validate_checkpoint_source_splits(
        run,
        expected_config,
        payload.get("source_split_records"),
    )
    if payload.get("source_names") != expected_source_names:
        raise ValueError(f"{run.role} source-domain identity mismatch")
    resume_contract = payload.get("resume_contract")
    if not isinstance(resume_contract, dict):
        raise ValueError(f"{run.role} has no resume/training contract")
    if resume_contract.get("model") != payload.get("model_config"):
        raise ValueError(f"{run.role} model and resume contracts disagree")
    expected_model = build_mshnet(expected_config["model"])
    expected_model_config = dict(expected_model.export_config())
    if payload.get("model_config") != expected_model_config:
        raise ValueError(f"{run.role} model_config differs from the frozen architecture")
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError(f"{run.role} checkpoint has no model_state")
    try:
        expected_model.load_state_dict(model_state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"{run.role} model_state does not strict-load") from error
    del expected_model
    resume_training = resume_contract.get("training")
    if not isinstance(resume_training, dict):
        raise ValueError(f"{run.role} resume contract has no training mapping")
    if int(resume_training.get("epochs", -1)) != 80 or int(
        resume_training.get("warmup_epochs", -1)
    ) != 0:
        raise ValueError(f"{run.role} resume training contract mismatch")
    output_log = run_dir / "train.stdout_stderr.log"
    output_lines = [
        line.strip()
        for line in output_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not output_lines or output_lines[-1] != str(checkpoint.resolve()):
        raise ValueError(f"{run.role} has no exact successful last.pt trailer")
    checkpoint_sha256 = sha256_file(checkpoint)
    resume_audit = validate_resume_audit(
        run, launcher, checkpoint_sha256
    )
    (run_dir / "checkpoint.sha256").write_text(
        f"{checkpoint_sha256}  last.pt\n", encoding="utf-8"
    )
    report = {
        "schema_version": "rc-irstd-phase2-checkpoint-identity-v1",
        "role": run.role,
        "checkpoint_sha256": checkpoint_sha256,
        "initializer_sha256": initializer_sha256,
        "config_sha256": sha256_file(run.config_path),
        "effective_config_sha256": sha256_file(run_dir / "config.json"),
        "launcher_sha256": sha256_file(run_dir / "launcher.json"),
        "launcher_pid": launcher["pid"],
        "launcher_launch_id": launcher["launch_id"],
        "launch_mode": launcher.get("launch_mode", "fresh"),
        "resume_audit": resume_audit,
        "loss_id": identity.get("base_loss_id"),
        "model_flags": {key: config["model"][key] for key in (
            "use_contrast",
            "use_component_context",
            "use_risk_gate",
            "expose_branch_auxiliary",
        )},
    }
    atomic_json(run_dir / "PHASE2_IDENTITY.json", report)
    return report


def _capture_round_binding(initializer: Path) -> dict[str, Any]:
    report = validate_gate_ready(
        state_dir=STATE_DIR,
        initializer=initializer,
        source_manifest=SOURCE_MANIFEST,
    )
    return {
        "config_sha256": dict(report["config_sha256"]),
        "initializer_sha256": str(report["initializer_sha256"]),
        "source_checkpoint_sha256": str(report["source_checkpoint_sha256"]),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
    }


def _assert_round_binding(initializer: Path, expected: dict[str, Any]) -> None:
    assert_sentinel_allows(STATE_DIR)
    if validate_phase2_configs() != expected["config_sha256"]:
        raise RuntimeError("Phase-2 configuration changed while the gate was open")
    if sha256_file(initializer) != expected["initializer_sha256"]:
        raise RuntimeError("Phase-2 initializer changed while the gate was open")
    if sha256_file(SOURCE_MANIFEST) != expected["source_manifest_sha256"]:
        raise RuntimeError("Phase-2 source identity manifest changed while the gate was open")
    source_manifest = _load_json_mapping(SOURCE_MANIFEST)
    if source_manifest.get("checkpoint_sha256") != expected["source_checkpoint_sha256"]:
        raise RuntimeError("Phase-2 source checkpoint binding changed while the gate was open")


def _committed_artifact_token(path: Path) -> tuple[int, int, int] | None:
    try:
        status = path.stat()
    except FileNotFoundError:
        return None
    return (status.st_ino, status.st_size, status.st_mtime_ns)


def _progress_fingerprint(run_dir: Path, history_rows: int) -> tuple[Any, ...]:
    return (
        history_rows,
        _committed_artifact_token(run_dir / "history.csv"),
        _committed_artifact_token(run_dir / "last.pt"),
    )


def _check_progress_watchdog(
    run: GateRun,
    watch: ProgressWatch | None,
    fingerprint: tuple[Any, ...],
    *,
    alive: bool,
    checkpoint_ready: bool,
    now_monotonic: float,
    no_progress_timeout_seconds: float,
    finalizing_timeout_seconds: float,
) -> ProgressWatch:
    """Fail closed on a live job that stops committing epoch-boundary state."""

    if watch is None:
        # A newly attached live process receives a full timeout window.  We do
        # not infer elapsed stall time from wall-clock mtimes.
        watch = ProgressWatch(
            fingerprint=fingerprint,
            last_progress_at=now_monotonic,
        )
    elif fingerprint != watch.fingerprint:
        watch.fingerprint = fingerprint
        watch.last_progress_at = now_monotonic

    if checkpoint_ready and alive:
        if watch.finalizing_since is None:
            watch.finalizing_since = now_monotonic
        elif now_monotonic - watch.finalizing_since >= finalizing_timeout_seconds:
            raise RuntimeError(
                f"{run.role} finalizing exceeded "
                f"{finalizing_timeout_seconds:.0f}s"
            )
    else:
        watch.finalizing_since = None

    if (
        alive
        and not checkpoint_ready
        and now_monotonic - watch.last_progress_at >= no_progress_timeout_seconds
    ):
        raise RuntimeError(
            f"{run.role} made no committed history/checkpoint progress for "
            f"{no_progress_timeout_seconds:.0f}s"
        )
    return watch


def run_round(
    name: str,
    runs: tuple[GateRun, ...],
    initializer: Path,
    *,
    no_progress_timeout_seconds: float = NO_PROGRESS_TIMEOUT_SECONDS,
    finalizing_timeout_seconds: float = FINALIZING_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if no_progress_timeout_seconds <= 0 or finalizing_timeout_seconds <= 0:
        raise ValueError("Phase-2 watchdog timeouts must be positive")
    frozen_binding = _capture_round_binding(initializer)
    handles: dict[str, tuple[subprocess.Popen[bytes] | None, int]] = {}
    watches: dict[str, ProgressWatch] = {}
    try:
        for run in runs:
            _assert_round_binding(initializer, frozen_binding)
            handles[run.role] = start_or_attach(run, initializer)
        previous = ""
        while True:
            _assert_round_binding(initializer, frozen_binding)
            pending = False
            parts: list[str] = []
            for run in runs:
                process, pid = handles[run.role]
                epochs = history_epochs(load_gate_output_dir(run.config_path))
                checkpoint_ready = gate_checkpoint_complete(run)
                return_code: int | None = None
                if process is not None:
                    return_code = process.poll()
                    alive = return_code is None
                    validate_launcher_binding(
                        run, initializer, require_live=alive
                    )
                elif pid > 0:
                    _, alive = validate_launcher_binding(
                        run, initializer, require_live=False
                    )
                else:
                    alive = False

                if checkpoint_ready and not alive:
                    if return_code not in (None, 0):
                        raise RuntimeError(
                            f"{run.role} returned {return_code} despite writing last.pt"
                        )
                    status = "complete"
                elif checkpoint_ready:
                    pending = True
                    status = "finalizing"
                elif alive:
                    pending = True
                    status = "alive"
                else:
                    raise RuntimeError(
                        f"{run.role} exited at {epochs}/80, return_code={return_code}"
                    )
                fingerprint = _progress_fingerprint(
                    load_gate_output_dir(run.config_path), epochs
                )
                watches[run.role] = _check_progress_watchdog(
                    run,
                    watches.get(run.role),
                    fingerprint,
                    alive=alive,
                    checkpoint_ready=checkpoint_ready,
                    now_monotonic=time.monotonic(),
                    no_progress_timeout_seconds=no_progress_timeout_seconds,
                    finalizing_timeout_seconds=finalizing_timeout_seconds,
                )
                parts.append(f"{run.role}={epochs}/80:{status}")
            rendered = " ".join(parts)
            if rendered != previous:
                log(f"{name} {rendered}")
                previous = rendered
            if not pending:
                break
            time.sleep(30)
        _assert_round_binding(initializer, frozen_binding)
    except BaseException:
        stop_jobs([pid for _, pid in handles.values()])
        for process, _ in handles.values():
            if process is None:
                continue
            try:
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ChildProcessError):
                pass
        raise

    initializer_sha256 = sha256_file(initializer)
    reports = {
        run.role: verify_gate_checkpoint(run, initializer_sha256, initializer)
        for run in runs
    }
    log(f"{name} completed")
    return reports


def write_gate_sha_manifest(reports: dict[str, Any]) -> None:
    lines = [
        f"{payload['checkpoint_sha256']}  {role}/last.pt"
        for role, payload in sorted(reports.items())
    ]
    (AUDIT_DIR / "detector_gate_checkpoint_SHA256SUMS").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


class CoordinatorTermination(RuntimeError):
    """Raised by termination-signal handlers so fail-closed cleanup runs."""


def _termination_handler(signum: int, _frame: Any) -> None:
    raise CoordinatorTermination(f"coordinator received signal {signum}")


def _remove_owned_pid_file(pid: int) -> None:
    try:
        recorded = read_pid(PID_PATH)
    except (FileNotFoundError, TypeError, ValueError):
        return
    if recorded == pid:
        PID_PATH.unlink(missing_ok=True)


def main() -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    INITIALIZER_DIR.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK_PATH.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError("another Phase-2 coordinator holds the lock") from error
    coordinator_pid = os.getpid()
    old_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGTERM, signal.SIGHUP)
    }
    for signum in old_handlers:
        signal.signal(signum, _termination_handler)
    try:
        PID_PATH.write_text(f"{coordinator_pid}\n", encoding="utf-8")
        require_preflight_hold()
        log(f"coordinator started pid={coordinator_pid} python={sys.executable}")
        wait_for_baselines()
        finalize_baselines()
        initializer = extract_initializers()
        recover_uncommitted_phase2_attempts(initializer)
        preflight_and_unlock(initializer)
        round1 = run_round("round1", ROUND1, initializer)
        round2 = run_round("round2", ROUND2, initializer)
        round3 = run_round("round3", ROUND3, initializer)
        reports = {**round1, **round2, **round3}
        write_gate_sha_manifest(reports)
        assert_sentinel_allows(STATE_DIR)
        restore_hold()
        atomic_json(
            AUDIT_DIR / "phase2_status.json",
            {
                "schema_version": "rc-irstd-aaai27-phase2-v2",
                "status": "completed",
                "completed_at": now(),
                "next_gate": "detector_matched_fa_evaluation_and_go_no_go",
                "risk_curve_started": False,
                "gate_state": "HOLD",
                "physical_gpus": [2, 3],
                "schedule": {
                    "round1": [run.role for run in ROUND1],
                    "round2": [run.role for run in ROUND2],
                    "round3": [run.role for run in ROUND3],
                },
                "runs": reports,
            },
        )
        log("PHASE2_COMPLETED; stopped before matched-FA evaluation and RiskCurve")
        return 0
    except BaseException as error:
        restore_hold()
        try:
            atomic_json(
                AUDIT_DIR / "phase2_failure.json",
                {
                    "schema_version": "rc-irstd-aaai27-phase2-failure-v1",
                    "status": "failed_closed",
                    "failed_at": now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "gate_state": "HOLD",
                },
            )
            log(f"FAILED_CLOSED {type(error).__name__}: {error}")
        except Exception:
            # Gate/PID cleanup is safety-critical; audit I/O is best effort.
            pass
        raise
    finally:
        _remove_owned_pid_file(coordinator_pid)
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
