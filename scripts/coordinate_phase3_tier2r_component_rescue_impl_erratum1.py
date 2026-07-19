#!/usr/bin/env python3
"""Resume Tier2R through an immutable implementation-only erratum chain."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO

from evaluation.artifact_integrity import file_sha256
from scripts import coordinate_phase3_tier2r_component_rescue as c
from scripts import register_phase3_tier2r_startup_fix1 as startup_fix1
from scripts import run_phase3_tier2r_exact_gate_impl_erratum1 as erratum_gate
from scripts import tier2r_impl_erratum1 as erratum


PROJECT_ROOT = erratum.PROJECT_ROOT
NEW_AUDIT_ROOT = PROJECT_ROOT / erratum.NEW_AUDIT_RELATIVE
OLD_AUDIT_ROOT = PROJECT_ROOT / erratum.OLD_AUDIT_RELATIVE
OLD_OUTPUT_ROOT = PROJECT_ROOT / erratum.OLD_OUTPUT_RELATIVE

_ORIGINAL_VERIFY_PREREQUISITES = c.verify_prerequisites
_ORIGINAL_PREREGISTRATION_PAYLOAD = c._preregistration_payload
_ORIGINAL_HANDOFF_PAYLOAD = c._handoff_payload
_ORIGINAL_RUN_TRAINING_ROUND = c._run_training_round

_STARTUP_BINDING_KEYS = ("startup_fix1", "startup_fix1_validation")
_ACTIVE_STARTUP_BINDINGS: dict[str, dict[str, str]] | None = None
_ACTIVE_STATUS_DELEGATE: Any | None = None


def _normalize_startup_bindings(
    bindings: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Copy and validate the two immutable startup-evidence bindings."""

    if set(bindings) != set(_STARTUP_BINDING_KEYS):
        raise RuntimeError("startup-fix1 binding set drift")
    normalized: dict[str, dict[str, str]] = {}
    for key in _STARTUP_BINDING_KEYS:
        binding = bindings.get(key)
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise RuntimeError(f"{key} binding shape drift")
        path = binding.get("path")
        digest = binding.get("sha256")
        if (
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise RuntimeError(f"{key} binding value drift")
        try:
            int(digest, 16)
        except ValueError as error:
            raise RuntimeError(f"{key} SHA-256 binding drift") from error
        normalized[key] = {"path": path, "sha256": digest}
    return normalized


def _active_startup_bindings() -> dict[str, dict[str, str]]:
    if _ACTIVE_STARTUP_BINDINGS is None:
        raise RuntimeError("frozen startup-fix1 evidence was not verified before execution")
    return _normalize_startup_bindings(_ACTIVE_STARTUP_BINDINGS)


def _packet_startup_bindings(
    packet: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    return _normalize_startup_bindings(
        {key: packet.get(key) for key in _STARTUP_BINDING_KEYS}
    )


def _verify_prerequisites_impl() -> dict[str, Any]:
    """Extend the old read-only verification with the recovery code closure."""

    packet = _ORIGINAL_VERIFY_PREREQUISITES()
    code_bindings = packet.get("code_bindings")
    if not isinstance(code_bindings, dict):
        raise RuntimeError("Tier2R prerequisite code bindings are absent")
    for relative in erratum.RECOVERY_CODE_RELATIVES:
        path = PROJECT_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"implementation-erratum code is absent: {path}")
        code_bindings[relative] = file_sha256(path)
    packet["implementation_erratum_plan"] = erratum.implementation_erratum_plan()
    if erratum.ERRATUM_PATH.exists():
        erratum.ensure_implementation_erratum(create=False)
        packet["implementation_erratum"] = erratum.implementation_erratum_binding()
    else:
        packet["implementation_erratum"] = erratum.ensure_implementation_erratum(
            create=False
        )
    if _ACTIVE_STARTUP_BINDINGS is not None:
        packet.update(_active_startup_bindings())
    return packet


def _preregistration_payload(
    packet: Mapping[str, Any], target_authorization_path: Path, path: Path
) -> dict[str, Any]:
    payload = _ORIGINAL_PREREGISTRATION_PAYLOAD(
        packet, target_authorization_path, path
    )
    payload["execution_instance"] = erratum.EXECUTION_INSTANCE
    payload["implementation_erratum"] = erratum.implementation_erratum_binding()
    payload["implementation_erratum_plan"] = erratum.implementation_erratum_plan()
    payload.update(_packet_startup_bindings(packet))
    return payload


def _handoff_payload(
    packet: Mapping[str, Any],
    preregistration_path: Path,
    target_auth_path: Path,
    path: Path,
) -> dict[str, Any]:
    payload = _ORIGINAL_HANDOFF_PAYLOAD(
        packet, preregistration_path, target_auth_path, path
    )
    payload["execution_instance"] = erratum.EXECUTION_INSTANCE
    payload["implementation_erratum"] = erratum.implementation_erratum_binding()
    payload["implementation_erratum_plan"] = erratum.implementation_erratum_plan()
    payload.update(_packet_startup_bindings(packet))
    return payload


def _write_status(status: str, **fields: Any) -> None:
    """Attach startup evidence to every formal status write in this scope."""

    if _ACTIVE_STATUS_DELEGATE is None:
        raise RuntimeError("startup-fix1 status delegate is outside its scope")
    bindings = _active_startup_bindings()
    for key, binding in bindings.items():
        if key in fields and fields[key] != binding:
            raise RuntimeError(f"{key} status binding drift")
    _ACTIVE_STATUS_DELEGATE(status, **{**fields, **bindings})


def _run_training_round(
    jobs: Sequence[c.JobSpec], packet: Mapping[str, Any]
) -> None:
    """Make complete-run reuse explicit and prove that no trainer is spawned."""

    inspections = [c.inspect_run(job, packet) for job in jobs]
    if all(inspection.state == "complete" for inspection in inspections):
        c._write_status(
            "tier2r_reusing_frozen_complete_runs",
            run_ids=[job.run_id for job in jobs],
            checkpoint_epochs=[inspection.epoch for inspection in inspections],
            trainer_processes_launched=0,
            implementation_erratum=erratum.implementation_erratum_binding(),
        )
        return
    _ORIGINAL_RUN_TRAINING_ROUND(jobs, packet)


def _run_exact_gate(handoff: Path) -> dict[str, Any]:
    output = PROJECT_ROOT / erratum.NEW_GATE_RELATIVE
    command = [
        str(c._python_executable()),
        str(
            PROJECT_ROOT
            / "scripts/run_phase3_tier2r_exact_gate_impl_erratum1.py"
        ),
        "--handoff",
        str(handoff.resolve()),
        "--output-root",
        str(output.resolve()),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    log_path = NEW_AUDIT_ROOT / "tier2r_exact_gate.log"
    with log_path.open("ab") as log_handle:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_handle.flush()
        os.fsync(log_handle.fileno())
    if completed.returncode != 0:
        raise RuntimeError(
            f"Tier2R implementation-erratum gate failed with exit code "
            f"{completed.returncode}"
        )
    return erratum_gate.verify_frozen_gate(
        handoff_path=handoff,
        output_root=output,
        project_root=PROJECT_ROOT,
    )


@contextmanager
def _patched_coordinator(
    startup_bindings: Mapping[str, Any] | None = None,
) -> Any:
    """Install coordinator recovery hooks only for one scoped operation."""

    global _ACTIVE_STARTUP_BINDINGS, _ACTIVE_STATUS_DELEGATE

    original_audit = c.AUDIT_ROOT
    original_output = c.OUTPUT_ROOT
    original_gate_audit = c.exact_gate.AUDIT_RELATIVE
    original_gate_output = c.exact_gate.GATE_RELATIVE
    original_verify = c.verify_prerequisites
    original_preregistration = c._preregistration_payload
    original_handoff = c._handoff_payload
    original_training_round = c._run_training_round
    original_exact_gate = c._run_exact_gate
    original_write_status = c._write_status
    previous_startup_bindings = _ACTIVE_STARTUP_BINDINGS
    previous_status_delegate = _ACTIVE_STATUS_DELEGATE
    _ACTIVE_STARTUP_BINDINGS = (
        None
        if startup_bindings is None
        else _normalize_startup_bindings(startup_bindings)
    )
    _ACTIVE_STATUS_DELEGATE = original_write_status
    c.AUDIT_ROOT = NEW_AUDIT_ROOT
    c.OUTPUT_ROOT = OLD_OUTPUT_ROOT
    c.exact_gate.AUDIT_RELATIVE = erratum.NEW_AUDIT_RELATIVE
    c.exact_gate.GATE_RELATIVE = erratum.NEW_GATE_RELATIVE
    c.verify_prerequisites = _verify_prerequisites_impl
    c._preregistration_payload = _preregistration_payload
    c._handoff_payload = _handoff_payload
    c._run_training_round = _run_training_round
    c._run_exact_gate = _run_exact_gate
    c._write_status = _write_status
    try:
        yield
    finally:
        c._write_status = original_write_status
        c._run_exact_gate = original_exact_gate
        c._run_training_round = original_training_round
        c._handoff_payload = original_handoff
        c._preregistration_payload = original_preregistration
        c.verify_prerequisites = original_verify
        c.exact_gate.GATE_RELATIVE = original_gate_output
        c.exact_gate.AUDIT_RELATIVE = original_gate_audit
        c.OUTPUT_ROOT = original_output
        c.AUDIT_ROOT = original_audit
        _ACTIVE_STATUS_DELEGATE = previous_status_delegate
        _ACTIVE_STARTUP_BINDINGS = previous_startup_bindings


def verify_prerequisites(
    *, startup_bindings: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    with _patched_coordinator(startup_bindings=startup_bindings):
        return _verify_prerequisites_impl()


def _exclusive_existing_readonly_lock(path: Path) -> BinaryIO:
    """Lock an existing historical lock inode without write access or creation."""

    if path.is_symlink() or not path.is_file():
        raise RuntimeError(
            f"historical Tier2R lock is absent, non-regular, or a symlink: {path}"
        )
    handle = path.open("rb", buffering=0)
    try:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise RuntimeError(f"historical Tier2R lock is not regular: {path}")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(
            "historical Tier2R coordinator lock is already owned"
        ) from error
    except BaseException:
        handle.close()
        raise
    return handle


def run(*, verify_only: bool = False) -> dict[str, Any]:
    if verify_only:
        with _patched_coordinator():
            result = c.run(verify_only=True)
            result["implementation_erratum_plan"] = (
                erratum.implementation_erratum_plan()
            )
            return result

    # This Docker-free verifier must succeed before the historical lock is
    # acquired, before the implementation erratum is registered, and before
    # any base-coordinator execution can create a formal artifact.
    startup_bindings = startup_fix1.verify_frozen_startup_fix1()
    with _patched_coordinator(startup_bindings=startup_bindings):
        old_lock = _exclusive_existing_readonly_lock(
            OLD_AUDIT_ROOT / ".tier2r_component_rescue.lock"
        )
        try:
            erratum.ensure_implementation_erratum(create=True)
            return c.run(verify_only=False)
        finally:
            fcntl.flock(old_lock, fcntl.LOCK_UN)
            old_lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(verify_only=args.verify_only)
    except BaseException as error:
        print(f"FAILED_CLOSED {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run", "verify_prerequisites"]
