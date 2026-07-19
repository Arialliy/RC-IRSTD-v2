#!/usr/bin/env python3
"""Isolated launcher for the Tier2R implementation erratum-1 execution.

The scientific protocol remains ``tier2r_c_v1``.  This launcher creates a
separate, auditable execution instance for the checkpoint-loader compatibility
erratum while reusing the frozen detector output root.  The superseded v1
container and artifacts are retained as evidence and must already be inert.

The default invocation is a non-mutating dry run.  ``--verify-only`` may create
and remove an empty canonical mountpoint, but it never registers a formal
erratum artifact.  ``--execute`` is the only mode that may register a new
launch intent and start the named formal container.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import functools
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import threading
from typing import Any, Iterator, Mapping, Sequence

PROJECT_ROOT = Path("/home/ly/RC-IRSTD-v2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import launch_phase3_tier2r_isolated as base  # noqa: E402


SCIENTIFIC_PROTOCOL_ID = "tier2r_c_v1"
EXECUTION_INSTANCE_ID = "tier2r_c_v1_impl_erratum1"
ERRATUM_ID = "impl_erratum1"

OLD_LAUNCHER = PROJECT_ROOT / "scripts/launch_phase3_tier2r_isolated.py"
OLD_COORDINATOR = (
    PROJECT_ROOT / "scripts/coordinate_phase3_tier2r_component_rescue.py"
)
OLD_EXACT_GATE = PROJECT_ROOT / "scripts/run_phase3_tier2r_exact_gate.py"
OLD_CONTAINER_PROBE = base.CONTAINER_PROBE
HELPER = PROJECT_ROOT / "scripts/tier2r_impl_erratum1.py"
COORDINATOR = (
    PROJECT_ROOT
    / "scripts/coordinate_phase3_tier2r_component_rescue_impl_erratum1.py"
)
EXACT_GATE = (
    PROJECT_ROOT / "scripts/run_phase3_tier2r_exact_gate_impl_erratum1.py"
)
CONTAINER_PROBE = (
    PROJECT_ROOT / "scripts/probe_phase3_tier2r_impl_erratum1_container.py"
)

OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs/aaai27/detectors/component_rescue/tier2r_c_v1"
)
AUDIT_ROOT = (
    PROJECT_ROOT
    / "artifacts/aaai27/audit/component_rescue/tier2r_c_v1_impl_erratum1"
)
INTENT_PATH = AUDIT_ROOT / "LAUNCH_INTENT.json"
INTENT_SHA256_PATH = AUDIT_ROOT / "LAUNCH_INTENT.json.sha256"
CONTAINER_NAME = "rc-irstd-tier2r-component-rescue-v1-impl-erratum1"
SCHEMA = "rc-irstd-aaai27-tier2r-isolated-launch-intent-impl-erratum1-v1"

OLD_AUDIT_ROOT = (
    PROJECT_ROOT / "artifacts/aaai27/audit/component_rescue/tier2r_c_v1"
)
OLD_INTENT_PATH = OLD_AUDIT_ROOT / "LAUNCH_INTENT.json"
OLD_INTENT_SHA256_PATH = OLD_AUDIT_ROOT / "LAUNCH_INTENT.json.sha256"
OLD_STATUS_PATH = OLD_AUDIT_ROOT / "COMPONENT_RESCUE_STATUS.json"
OLD_CONTAINER_NAME = "rc-irstd-tier2r-component-rescue-v1"

RUNTIME_LABELS = {
    "org.rc-irstd.protocol": SCIENTIFIC_PROTOCOL_ID,
    "org.rc-irstd.scientific-protocol": SCIENTIFIC_PROTOCOL_ID,
    "org.rc-irstd.execution-instance": EXECUTION_INSTANCE_ID,
    "org.rc-irstd.implementation-erratum": ERRATUM_ID,
    "org.rc-irstd.source-only": "true",
    "org.rc-irstd.outer-target-access": "denied",
}

CODE_PATHS = (
    OLD_LAUNCHER,
    Path(__file__).resolve(),
    HELPER,
    COORDINATOR,
    EXACT_GATE,
    PROJECT_ROOT / "evaluation/export_score_maps.py",
    OLD_COORDINATOR,
    OLD_EXACT_GATE,
    OLD_CONTAINER_PROBE,
    CONTAINER_PROBE,
    PROJECT_ROOT / "rc_irstd/cli/export_scores.py",
    PROJECT_ROOT / "rc_irstd/cli/train_detector.py",
    PROJECT_ROOT / "rc_irstd/training/detector_trainer.py",
    PROJECT_ROOT / "rc_irstd/models/rc_mshnet.py",
    PROJECT_ROOT / "rc_irstd/models/__init__.py",
    PROJECT_ROOT / "evaluation/raw_logit_source_operating_point.py",
    PROJECT_ROOT / "evaluation/component_matching.py",
)

FIXED_IMPLEMENTATION_PLAN = {
    "scientific_protocol_id": SCIENTIFIC_PROTOCOL_ID,
    "execution_instance_id": EXECUTION_INSTANCE_ID,
    "implementation_erratum_id": ERRATUM_ID,
    "scope": "implementation_only",
    "scientific_protocol_changed": False,
    "correction": "numpy_checkpoint_safe_global_resolution",
    "reuse_completed_round1_epoch79_checkpoints": True,
    "retrain_completed_round1": False,
    "output_root": str(OUTPUT_ROOT),
    "audit_root": str(AUDIT_ROOT),
    "outer_target_access_authorized": False,
}


# Capture the original implementations before rebinding the base module.  The
# imported functions resolve module globals dynamically, which lets this thin
# wrapper retain the already-audited isolation machinery without copying it.
_BASE_CONTAINER_SPEC = base.container_spec
_BASE_BUILD_FORMAL_COMMAND = base.build_formal_command
_BASE_VALIDATE_EXISTING_CONTAINER = base._validate_existing_container
_BASE_START_OR_RECONCILE = base._start_or_reconcile
_BASE_LAUNCH_INTENT_PAYLOAD = base._launch_intent_payload


_VERIFY_AUDIT_SOURCE: Path | None = None


def bind_mount_contract() -> tuple[dict[str, Any], ...]:
    """Expose only the project, frozen output root, and new audit root."""

    audit_source = _VERIFY_AUDIT_SOURCE or AUDIT_ROOT
    return (
        {
            "source": str(PROJECT_ROOT),
            "destination": str(PROJECT_ROOT),
            "readonly": True,
        },
        {
            "source": str(OUTPUT_ROOT),
            "destination": str(OUTPUT_ROOT),
            "readonly": False,
        },
        {
            "source": str(audit_source),
            "destination": str(AUDIT_ROOT),
            "readonly": False,
        },
    )


def container_spec(*, formal: bool) -> dict[str, Any]:
    spec = _BASE_CONTAINER_SPEC(formal=formal)
    spec["restart"] = "on-failure:3" if formal else None
    spec["restart_policy"] = (
        {"Name": "on-failure", "MaximumRetryCount": 3} if formal else None
    )
    spec["scientific_protocol_id"] = SCIENTIFIC_PROTOCOL_ID
    spec["execution_instance_id"] = EXECUTION_INSTANCE_ID
    spec["implementation_erratum_id"] = ERRATUM_ID
    return spec


def build_formal_command(intent_sha256: str) -> list[str]:
    if len(intent_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in intent_sha256
    ):
        raise ValueError("launch-intent SHA-256 must be lowercase hexadecimal")
    labels = {**RUNTIME_LABELS, base.INTENT_LABEL: intent_sha256}
    return [
        *base._common_docker_args(labels=labels),
        "--detach",
        "--name",
        CONTAINER_NAME,
        "--restart",
        "on-failure:3",
        base.IMAGE_ID,
        str(COORDINATOR),
    ]


def _validate_existing_container(
    container: Mapping[str, Any], intent_sha256: str
) -> str:
    status = _BASE_VALIDATE_EXISTING_CONTAINER(container, intent_sha256)
    host = container.get("HostConfig")
    if not isinstance(host, Mapping):
        raise RuntimeError("existing container HostConfig is absent")
    restart = host.get("RestartPolicy")
    if (
        not isinstance(restart, Mapping)
        or restart.get("Name") != "on-failure"
        or restart.get("MaximumRetryCount") != 3
    ):
        raise RuntimeError(
            "existing named container restart policy must be exactly on-failure:3"
        )
    return status


def _start_or_reconcile(
    intent_sha256: str, *, runner: base.Runner | None = None
) -> dict[str, Any]:
    result = _BASE_START_OR_RECONCILE(intent_sha256, runner=runner)
    observed = base._inspect_existing_container(runner=runner)
    if observed is None:
        raise RuntimeError("formal container disappeared after launch")
    status = _validate_existing_container(observed, intent_sha256)
    result["observed_status"] = status
    result["restart_policy_verified"] = {
        "Name": "on-failure",
        "MaximumRetryCount": 3,
    }
    return result


_BASE_SCOPE_LOCK = threading.RLock()
_BASE_SCOPE_DEPTH = 0
_BASE_SCOPE_SAVED: dict[str, Any] = {}


def _base_overrides() -> dict[str, Any]:
    return {
        "CONTAINER_PROBE": CONTAINER_PROBE,
        "COORDINATOR": COORDINATOR,
        "EXACT_GATE": EXACT_GATE,
        "OUTPUT_ROOT": OUTPUT_ROOT,
        "AUDIT_ROOT": AUDIT_ROOT,
        "INTENT_PATH": INTENT_PATH,
        "INTENT_SHA256_PATH": INTENT_SHA256_PATH,
        "CONTAINER_NAME": CONTAINER_NAME,
        "SCHEMA": SCHEMA,
        "RUNTIME_LABELS": RUNTIME_LABELS,
        "CODE_PATHS": CODE_PATHS,
        "bind_mount_contract": bind_mount_contract,
        "container_spec": container_spec,
        "build_formal_command": build_formal_command,
        "_validate_existing_container": _validate_existing_container,
        "_start_or_reconcile": _start_or_reconcile,
    }


@contextmanager
def _base_erratum_scope() -> Iterator[None]:
    """Install erratum globals only while an operation is in flight."""

    global _BASE_SCOPE_DEPTH, _BASE_SCOPE_SAVED
    with _BASE_SCOPE_LOCK:
        if _BASE_SCOPE_DEPTH == 0:
            overrides = _base_overrides()
            _BASE_SCOPE_SAVED = {
                name: getattr(base, name) for name in overrides
            }
            for name, value in overrides.items():
                setattr(base, name, value)
        _BASE_SCOPE_DEPTH += 1
        try:
            yield
        finally:
            _BASE_SCOPE_DEPTH -= 1
            if _BASE_SCOPE_DEPTH == 0:
                for name, value in _BASE_SCOPE_SAVED.items():
                    setattr(base, name, value)
                _BASE_SCOPE_SAVED = {}


def _scoped(function: Any) -> Any:
    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with _base_erratum_scope():
            return function(*args, **kwargs)

    return wrapped


def _load_regular_json(path: Path, *, immutable: bool = False) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required evidence is absent or a symlink: {path}")
    if immutable and stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise RuntimeError(f"immutable evidence mode drift: {path}")
    return base._load_json_bytes(path.read_text(encoding="utf-8"), context=str(path))


def _verify_old_intent_sidecar() -> str:
    intent = _load_regular_json(OLD_INTENT_PATH, immutable=True)
    del intent
    if OLD_INTENT_SHA256_PATH.is_symlink() or not OLD_INTENT_SHA256_PATH.is_file():
        raise RuntimeError("superseded launch-intent sidecar is absent or a symlink")
    if stat.S_IMODE(OLD_INTENT_SHA256_PATH.stat().st_mode) != 0o444:
        raise RuntimeError("superseded launch-intent sidecar mode drift")
    raw = OLD_INTENT_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = f"{digest}  {OLD_INTENT_PATH.name}\n".encode("ascii")
    if OLD_INTENT_SHA256_PATH.read_bytes() != expected:
        raise RuntimeError("superseded launch-intent sidecar digest drift")
    return digest


def _superseded_execution_snapshot(
    *, runner: base.Runner | None = None
) -> dict[str, Any]:
    old_intent_sha256 = _verify_old_intent_sidecar()
    old_intent = _load_regular_json(OLD_INTENT_PATH, immutable=True)
    old_status = _load_regular_json(OLD_STATUS_PATH)
    if old_intent.get("container_name") != OLD_CONTAINER_NAME:
        raise RuntimeError("superseded launch intent container identity drift")
    if old_status.get("status") != "failed_closed":
        raise RuntimeError("superseded execution is not frozen at failed_closed")
    completed = base._run(
        ["docker", "container", "inspect", OLD_CONTAINER_NAME],
        runner=runner,
    )
    inspected = json.loads(completed.stdout)
    if (
        not isinstance(inspected, list)
        or len(inspected) != 1
        or not isinstance(inspected[0], Mapping)
    ):
        raise RuntimeError("superseded container inspect returned malformed JSON")
    container = inspected[0]
    state = container.get("State")
    host = container.get("HostConfig")
    if not isinstance(state, Mapping) or not isinstance(host, Mapping):
        raise RuntimeError("superseded container inspect sections are incomplete")
    restart = host.get("RestartPolicy")
    if (
        state.get("Status") != "exited"
        or state.get("Running") is not False
        or state.get("Restarting") is not False
        or not isinstance(restart, Mapping)
        or restart.get("Name") != "no"
        or restart.get("MaximumRetryCount") != 0
    ):
        raise RuntimeError(
            "superseded container must be exited with restart policy exactly no"
        )
    return {
        "scientific_protocol_id": SCIENTIFIC_PROTOCOL_ID,
        "execution_instance_id": SCIENTIFIC_PROTOCOL_ID,
        "launch_intent": str(OLD_INTENT_PATH),
        "launch_intent_sha256": old_intent_sha256,
        "status_artifact": str(OLD_STATUS_PATH),
        "status_artifact_sha256": base._sha256(OLD_STATUS_PATH),
        "status": old_status.get("status"),
        "error_type": old_status.get("error_type"),
        "container": {
            "name": OLD_CONTAINER_NAME,
            "id": container.get("Id"),
            "status": state.get("Status"),
            "exit_code": state.get("ExitCode"),
            "oom_killed": state.get("OOMKilled"),
            "restart_count": container.get("RestartCount"),
            "restart_policy": {
                "Name": restart.get("Name"),
                "MaximumRetryCount": restart.get("MaximumRetryCount"),
            },
        },
    }


def _validated_coordinator_plan(verification: Mapping[str, Any]) -> dict[str, Any]:
    plan = verification.get("implementation_erratum_plan")
    if not isinstance(plan, Mapping):
        raise RuntimeError(
            "coordinator verify-only omitted implementation_erratum_plan"
        )
    expected = {
        "scientific_protocol_id": SCIENTIFIC_PROTOCOL_ID,
        "execution_instance": EXECUTION_INSTANCE_ID,
        "change_scope": "checkpoint_loader_numpy_1x_2x_compatibility_only",
        "reuse_frozen_round1_checkpoints": True,
        "training_unchanged": True,
        "model_unchanged": True,
        "config_unchanged": True,
        "data_unchanged": True,
        "decision_protocol_unchanged": True,
        "outer_target_access_authorized": False,
    }
    if dict(plan) != expected:
        raise RuntimeError("coordinator implementation_erratum_plan contract drift")
    return dict(plan)


@contextmanager
def _verification_audit_source() -> Iterator[None]:
    """Provide a non-formal bind source for a cold-start verification.

    Docker needs the nested bind destination to exist inside the outer
    read-only project bind. Create only that empty mountpoint, direct the
    writable verification bind to ``/tmp``, and remove the empty mountpoint on
    every exit path.
    """

    global _VERIFY_AUDIT_SOURCE
    previous = _VERIFY_AUDIT_SOURCE
    if AUDIT_ROOT.exists():
        if AUDIT_ROOT.is_symlink() or not AUDIT_ROOT.is_dir():
            raise RuntimeError(f"erratum audit root is not canonical: {AUDIT_ROOT}")
        _VERIFY_AUDIT_SOURCE = AUDIT_ROOT
        try:
            yield
        finally:
            _VERIFY_AUDIT_SOURCE = previous
        return
    AUDIT_ROOT.mkdir(parents=True, exist_ok=False)
    try:
        with tempfile.TemporaryDirectory(
            prefix="tier2r_impl_erratum1_verify_", dir="/tmp"
        ) as raw:
            _VERIFY_AUDIT_SOURCE = Path(raw)
            try:
                yield
            finally:
                _VERIFY_AUDIT_SOURCE = previous
    finally:
        try:
            AUDIT_ROOT.rmdir()
        except OSError as error:
            raise RuntimeError(
                "verify-only mountpoint is not empty; refusing silent cleanup"
            ) from error


def _validate_nonmutating_host_sources() -> None:
    if not OUTPUT_ROOT.is_dir() or OUTPUT_ROOT.is_symlink():
        raise RuntimeError(f"frozen output root is absent or invalid: {OUTPUT_ROOT}")
    base._validate_host_sources()


def verify_only(*, runner: base.Runner | None = None) -> dict[str, Any]:
    """Run isolated preflight without registering a formal erratum artifact."""

    audit_existed = AUDIT_ROOT.exists()
    with _verification_audit_source():
        _validate_nonmutating_host_sources()
        image_id = base._inspect_image(runner=runner)
        inventory = base._gpu_inventory(runner=runner)
        verification = base._verify_in_container(runner=runner)
        coordinator_plan = _validated_coordinator_plan(verification)
        attestation = base._verify_container_attestation(inventory, runner=runner)
        superseded = _superseded_execution_snapshot(runner=runner)
    if AUDIT_ROOT.exists() is not audit_existed:
        raise RuntimeError("verify-only mutated the formal erratum audit root")
    return {
        "schema_version": SCHEMA,
        "verified": True,
        "formal_container_started": False,
        "formal_artifact_registered": False,
        "scientific_protocol_id": SCIENTIFIC_PROTOCOL_ID,
        "execution_instance_id": EXECUTION_INSTANCE_ID,
        "implementation_erratum_id": ERRATUM_ID,
        "source_only": True,
        "outer_target_access_authorized": False,
        "image_id": image_id,
        "host_gpu_inventory": list(inventory),
        "coordinator_assigned_host_gpus": [
            dict(value) for value in inventory
            if value["index"] in base.TARGET_GPU_INDICES
        ],
        "container_attestation": attestation,
        "verification_container_spec": container_spec(formal=False),
        "formal_container_spec": container_spec(formal=True),
        "coordinator_verification": verification,
        "implementation_erratum_plan": coordinator_plan,
        "launcher_fixed_implementation_plan": dict(FIXED_IMPLEMENTATION_PLAN),
        "superseded_execution": superseded,
    }


def _launch_intent_payload(
    *,
    image_id: str,
    gpu_inventory: Sequence[Mapping[str, Any]],
    attestation: Mapping[str, Any],
    verification: Mapping[str, Any],
    coordinator_plan: Mapping[str, Any],
    superseded: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _BASE_LAUNCH_INTENT_PAYLOAD(
        image_id=image_id,
        gpu_inventory=gpu_inventory,
        attestation=attestation,
        verification=verification,
    )
    payload.update(
        {
            "schema_version": SCHEMA,
            "decision": "LAUNCH_TIER2R_IMPL_ERRATUM1_SOURCE_ONLY",
            "scientific_protocol_id": SCIENTIFIC_PROTOCOL_ID,
            "scientific_protocol_changed": False,
            "execution_instance_id": EXECUTION_INSTANCE_ID,
            "implementation_erratum_id": ERRATUM_ID,
            "implementation_erratum_plan": dict(coordinator_plan),
            "launcher_fixed_implementation_plan": dict(FIXED_IMPLEMENTATION_PLAN),
            "supersedes": dict(superseded),
            "restart_policy": {
                "Name": "on-failure",
                "MaximumRetryCount": 3,
            },
        }
    )
    return payload


def _ensure_formal_audit_root() -> None:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    if (
        AUDIT_ROOT.is_symlink()
        or not AUDIT_ROOT.is_dir()
        or AUDIT_ROOT.resolve() != AUDIT_ROOT
    ):
        raise RuntimeError(f"formal erratum audit root is not exact: {AUDIT_ROOT}")


def execute_launch(*, runner: base.Runner | None = None) -> dict[str, Any]:
    evidence = verify_only(runner=runner)
    _ensure_formal_audit_root()
    current_superseded = _superseded_execution_snapshot(runner=runner)
    if base._canonical_bytes(current_superseded) != base._canonical_bytes(
        evidence["superseded_execution"]
    ):
        raise RuntimeError("superseded execution changed between preflight and registration")
    intent = _launch_intent_payload(
        image_id=str(evidence["image_id"]),
        gpu_inventory=evidence["host_gpu_inventory"],
        attestation=evidence["container_attestation"],
        verification=evidence["coordinator_verification"],
        coordinator_plan=evidence["implementation_erratum_plan"],
        superseded=current_superseded,
    )
    intent_sha256 = base._register_launch_intent(intent)
    # Recheck immediately before Docker state changes.
    if base._canonical_bytes(_superseded_execution_snapshot(runner=runner)) != base._canonical_bytes(
        current_superseded
    ):
        raise RuntimeError("superseded execution changed after intent registration")
    launch = _start_or_reconcile(intent_sha256, runner=runner)
    return {
        "schema_version": SCHEMA,
        "verified_before_launch": True,
        "scientific_protocol_id": SCIENTIFIC_PROTOCOL_ID,
        "scientific_protocol_changed": False,
        "execution_instance_id": EXECUTION_INSTANCE_ID,
        "implementation_erratum_id": ERRATUM_ID,
        "source_only": True,
        "outer_target_access_authorized": False,
        "launch_intent": str(INTENT_PATH),
        "launch_intent_sha256": intent_sha256,
        "container": launch,
    }


def dry_run_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "dry_run": True,
        "docker_invoked": False,
        "filesystem_mutated": False,
        "scientific_protocol_id": SCIENTIFIC_PROTOCOL_ID,
        "scientific_protocol_changed": False,
        "execution_instance_id": EXECUTION_INSTANCE_ID,
        "implementation_erratum_id": ERRATUM_ID,
        "container_name": CONTAINER_NAME,
        "image": base.IMAGE_ID,
        "source_only": True,
        "outer_target_access_authorized": False,
        "coordinator_physical_gpus": list(base.TARGET_GPU_INDICES),
        "wait_for_idle_gpu": False,
        "verify_command": base.build_verify_command(),
        "formal_command_template": build_formal_command("0" * 64),
        "formal_container_spec": container_spec(formal=True),
        "superseded_container_required_state": {
            "name": OLD_CONTAINER_NAME,
            "status": "exited",
            "restart_policy": {"Name": "no", "MaximumRetryCount": 0},
        },
        "implementation_erratum_plan": dict(FIXED_IMPLEMENTATION_PLAN),
        "execute_with": f"python3 {Path(__file__).resolve()} --execute",
    }


container_spec = _scoped(container_spec)
build_formal_command = _scoped(build_formal_command)
_validate_existing_container = _scoped(_validate_existing_container)
_start_or_reconcile = _scoped(_start_or_reconcile)
verify_only = _scoped(verify_only)
execute_launch = _scoped(execute_launch)
dry_run_payload = _scoped(dry_run_payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.execute:
            payload = execute_launch()
        elif args.verify_only:
            payload = verify_only()
        else:
            payload = dry_run_payload()
    except BaseException as error:
        print(f"FAILED_CLOSED {type(error).__name__}: {error}", file=os.sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "bind_mount_contract",
    "build_formal_command",
    "container_spec",
    "dry_run_payload",
    "execute_launch",
    "verify_only",
]
