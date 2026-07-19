"""Train the multi-source detector with fixed-last checkpoint selection."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch

from rc_irstd.config import (
    apply_overrides,
    load_yaml,
    public_config,
    resolve_config_path,
)
from rc_irstd.training import DetectorTrainer
from rc_irstd.utils.io import atomic_write_json


RESUME_AUDIT_SCHEMA = "rc-irstd-formal-resume-v1"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _history_epochs(path: Path) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    try:
        epochs = [int(row["epoch"]) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid epoch column in {path}") from error
    if epochs != list(range(len(epochs))):
        raise ValueError(f"Resume history must contain contiguous epochs 0..N: {path}")
    return epochs


def _visible_gpu_record() -> dict[str, object]:
    raw_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible = [item.strip() for item in raw_visible.split(",") if item.strip()]
    if len(visible) != 1 or not visible[0].isdigit():
        raise RuntimeError(
            "Formal resume requires exactly one numeric physical GPU in "
            "CUDA_VISIBLE_DEVICES"
        )
    physical_index = int(visible[0])
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    matches = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4 and fields[0] == str(physical_index):
            matches.append(fields)
    if len(matches) != 1:
        raise RuntimeError(f"Could not resolve physical GPU {physical_index}")
    _, uuid, name, driver = matches[0]
    return {
        "cuda_visible_devices": raw_visible,
        "physical_gpu_index": physical_index,
        "physical_gpu_uuid": uuid,
        "physical_gpu_name": name,
        "driver_version": driver,
        "logical_device": "cuda:0",
    }


def _resume_snapshot(
    checkpoint: Path,
    output_dir: Path,
    *,
    epoch: int,
    checkpoint_sha256: str,
) -> Path:
    snapshot_dir = output_dir / "resume_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_dir / (
        f"epoch_{epoch:04d}_{checkpoint_sha256[:16]}.pt"
    )
    if snapshot.is_file():
        if _sha256_file(snapshot) != checkpoint_sha256:
            raise RuntimeError(f"Existing resume snapshot SHA mismatch: {snapshot}")
        return snapshot
    temporary = snapshot.with_suffix(snapshot.suffix + ".tmp")
    shutil.copyfile(checkpoint, temporary)
    if _sha256_file(temporary) != checkpoint_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Resume snapshot copy failed its SHA256 check")
    temporary.replace(snapshot)
    return snapshot


def _prepare_formal_resume(
    config: dict[str, object],
    resume_checkpoint: Path,
) -> tuple[Path, Path, dict[str, object]]:
    formal = public_config(config)
    training = formal.get("training")
    if not isinstance(training, dict) or training.get("resume") is not None:
        raise ValueError(
            "--resume-checkpoint is runtime-only and requires formal "
            "training.resume=null"
        )
    output_dir = resolve_config_path(config, str(config.get("output_dir", "")))
    config_path = output_dir / "config.json"
    if _load_json(config_path) != formal:
        raise ValueError("On-disk formal config differs from the requested resume config")
    checkpoint = resume_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if (output_dir / "last.pt.tmp").exists():
        raise RuntimeError(f"Incomplete checkpoint temporary file in {output_dir}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Resume checkpoint root must be a mapping")
    if payload.get("format_version") != 2 or payload.get("kind") != "detector":
        raise ValueError("Resume checkpoint is not a format-v2 detector checkpoint")
    if payload.get("config") != formal:
        raise ValueError("Resume checkpoint embedded config differs from formal config")
    epoch = int(payload.get("epoch", -1))
    epochs = _history_epochs(output_dir / "history.csv")
    if epoch < 0 or epochs != list(range(epoch + 1)):
        raise ValueError("Resume history and checkpoint epoch are not exactly aligned")
    required = {
        "model_state",
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
        raise ValueError("Resume checkpoint is missing: " + ", ".join(missing))
    rng_state = payload["rng_state"]
    if not isinstance(rng_state, dict) or len(rng_state.get("torch_cuda", [])) != 1:
        raise ValueError("Resume checkpoint must contain exactly one CUDA RNG state")
    checkpoint_sha256 = _sha256_file(checkpoint)
    snapshot = _resume_snapshot(
        checkpoint,
        output_dir,
        epoch=epoch,
        checkpoint_sha256=checkpoint_sha256,
    )
    event: dict[str, object] = {
        "owner_pid": os.getpid(),
        "process_group_id": os.getpgid(0),
        "session_id": os.getsid(0),
        "source_checkpoint_path": str(checkpoint),
        "immutable_snapshot_path": str(snapshot),
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_checkpoint_size": checkpoint.stat().st_size,
        "source_epoch": epoch,
        "resume_start_epoch": epoch + 1,
        "history_rows_before_resume": len(epochs),
        "history_sha256_before_resume": _sha256_file(output_dir / "history.csv"),
        "formal_config_sha256": _sha256_file(config_path),
        "resume_cli_sha256": _sha256_file(Path(__file__).resolve()),
        "torch_version": torch.__version__,
        "started_at": _now(),
        "interruption_reason": "external_termination_without_in_process_failure_record",
        "status": "prepared",
        **_visible_gpu_record(),
    }
    return output_dir, snapshot, event


def _load_or_create_audit(path: Path) -> dict[str, object]:
    if path.is_file():
        audit = _load_json(path)
        if audit.get("schema_version") != RESUME_AUDIT_SCHEMA:
            raise ValueError(f"Unexpected resume audit schema: {path}")
        events = audit.get("events")
        if not isinstance(events, list):
            raise ValueError(f"Resume audit has no events list: {path}")
        return audit
    return {
        "schema_version": RESUME_AUDIT_SCHEMA,
        "status": "prepared",
        "events": [],
    }


def _run_formal_resume(
    config: dict[str, object],
    *,
    resume_checkpoint: Path,
    pid_file: Path | None,
    audit_path: Path | None,
) -> Path:
    output_dir = resolve_config_path(config, str(config.get("output_dir", "")))
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".formal_resume.lock"
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError(f"Another formal resume owns {output_dir}") from error

    formal_public = public_config(config)
    resolved_audit = (audit_path or output_dir / "RESUME_AUDIT.json").resolve()
    event: dict[str, object] | None = None
    audit: dict[str, object] | None = None
    try:
        output_dir, snapshot, event = _prepare_formal_resume(
            config,
            resume_checkpoint,
        )
        audit = _load_or_create_audit(resolved_audit)
        events = audit["events"]
        if not isinstance(events, list):
            raise TypeError("Resume audit events must be a list")
        events.append(event)
        event["status"] = "running"
        audit["status"] = "running"
        audit["updated_at"] = _now()
        atomic_write_json(resolved_audit, audit)
        if pid_file is not None:
            _atomic_write_text(pid_file.resolve(), f"{os.getpid()}\n")

        runtime_config = deepcopy(config)
        runtime_training = runtime_config.get("training")
        if not isinstance(runtime_training, dict):
            raise TypeError("Runtime training config must be a mapping")
        formal_initialize_from = runtime_training.get("initialize_from")
        # A resumed checkpoint already contains the exact initialized weights
        # and initialization report.  Keep ``initialize_from`` in the frozen
        # public config, but clear it only while constructing the runtime
        # trainer so the mutually-exclusive resume path can restore that state.
        runtime_training["initialize_from"] = None
        runtime_training["resume"] = str(snapshot)
        trainer: DetectorTrainer | None = None
        try:
            trainer = DetectorTrainer(runtime_config)
        finally:
            runtime_training["resume"] = None
            runtime_training["initialize_from"] = formal_initialize_from
            atomic_write_json(output_dir / "config.json", formal_public)
        if trainer is None:  # pragma: no cover - defensive after constructor failure
            raise RuntimeError("DetectorTrainer construction did not complete")
        trainer.config = config
        checkpoint = trainer.run()
        final_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        expected_epochs = int(formal_public["training"]["epochs"])
        if (
            final_payload.get("epoch") != expected_epochs - 1
            or final_payload.get("config") != formal_public
            or _history_epochs(output_dir / "history.csv") != list(range(expected_epochs))
        ):
            raise RuntimeError("Formal resume completed without an exact final artifact")
        event["status"] = "completed"
        event["completed_at"] = _now()
        event["final_checkpoint_path"] = str(checkpoint.resolve())
        event["final_checkpoint_sha256"] = _sha256_file(checkpoint)
        audit["status"] = "completed"
        audit["updated_at"] = _now()
        atomic_write_json(resolved_audit, audit)
        return checkpoint
    except BaseException as error:
        atomic_write_json(output_dir / "config.json", formal_public)
        if event is not None and audit is not None:
            event["status"] = "failed"
            event["failed_at"] = _now()
            event["error_type"] = type(error).__name__
            event["error"] = str(error)
            audit["status"] = "failed"
            audit["error_type"] = type(error).__name__
            audit["error"] = str(error)
            audit["updated_at"] = _now()
            atomic_write_json(resolved_audit, audit)
        raise
    finally:
        lock_handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Detector YAML configuration")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a nested key, e.g. --set training.epochs=10",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help=(
            "Runtime-only exact resume checkpoint. The formal serialized config "
            "must retain training.resume=null."
        ),
    )
    parser.add_argument(
        "--pid-file",
        type=Path,
        default=None,
        help="Atomically record the actual trainer process PID.",
    )
    parser.add_argument(
        "--resume-audit",
        type=Path,
        default=None,
        help="Resume provenance JSON (defaults to OUTPUT_DIR/RESUME_AUDIT.json).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = apply_overrides(load_yaml(args.config), args.overrides)
    training = config.setdefault("training", {})
    selection = str(training.get("checkpoint_selection", "fixed_last"))
    if selection != "fixed_last":
        raise ValueError("Formal detector training requires checkpoint_selection=fixed_last")
    if args.resume_checkpoint is not None:
        checkpoint = _run_formal_resume(
            config,
            resume_checkpoint=args.resume_checkpoint,
            pid_file=args.pid_file,
            audit_path=args.resume_audit,
        )
    else:
        if args.pid_file is not None:
            _atomic_write_text(args.pid_file.resolve(), f"{os.getpid()}\n")
        checkpoint = DetectorTrainer(config).run()
    if checkpoint.name != "last.pt":
        raise RuntimeError(f"Fixed-last trainer returned an unexpected artifact: {checkpoint}")
    print(checkpoint)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
