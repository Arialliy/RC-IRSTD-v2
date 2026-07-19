#!/usr/bin/env python3
"""Fail-closed Docker launcher for the Tier2R component-rescue protocol.

The launcher is deliberately separate from the experiment coordinator.  It
first executes the coordinator with ``--verify-only`` in an ephemeral
container that has the same bind mounts, tmpfs target denial, GPU exposure,
and read-only root filesystem as the formal container.  Only a successful,
machine-readable verification can register the immutable launch intent and
start (or idempotently reconcile) the named formal container.

The default invocation is non-mutating.  Pass ``--execute`` to run Docker,
register ``LAUNCH_INTENT.json``, and start/restart the formal container.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path("/home/ly/RC-IRSTD-v2")
CONTAINER_PYTHON = "python"

COORDINATOR = PROJECT_ROOT / "scripts/coordinate_phase3_tier2r_component_rescue.py"
CONTAINER_PROBE = PROJECT_ROOT / "scripts/probe_phase3_tier2r_container.py"
EXACT_GATE = PROJECT_ROOT / "scripts/run_phase3_tier2r_exact_gate.py"
PROTOCOL = PROJECT_ROOT / "configs/tier2r_component_rescue_protocol.json"

OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs/aaai27/detectors/component_rescue/tier2r_c_v1"
)
AUDIT_ROOT = (
    PROJECT_ROOT / "artifacts/aaai27/audit/component_rescue/tier2r_c_v1"
)
FORBIDDEN_TARGET = PROJECT_ROOT / "datasets/NUAA-SIRST"
INTENT_PATH = AUDIT_ROOT / "LAUNCH_INTENT.json"
INTENT_SHA256_PATH = AUDIT_ROOT / "LAUNCH_INTENT.json.sha256"

IMAGE_ID = "sha256:42e03b9c7e2628bf7622008c79a6be33bca666a1dcb6d56a7d4d1e28f0c91fe3"
CONTAINER_NAME = "rc-irstd-tier2r-component-rescue-v1"
CONTAINER_USER = "1004:1004"
TARGET_GPU_INDICES = (2, 3)
SCHEMA = "rc-irstd-aaai27-tier2r-isolated-launch-intent-v1"
INTENT_LABEL = "org.rc-irstd.launch-intent-sha256"

ENVIRONMENT = {
    "HOME": "/tmp",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONPATH": "/home/ly/RC-IRSTD-v2",
    "NVIDIA_VISIBLE_DEVICES": "all",
}

RUNTIME_LABELS = {
    "org.rc-irstd.protocol": "tier2r_c_v1",
    "org.rc-irstd.source-only": "true",
    "org.rc-irstd.outer-target-access": "denied",
}

CODE_PATHS = (
    Path(__file__).resolve(),
    COORDINATOR,
    CONTAINER_PROBE,
    EXACT_GATE,
    PROJECT_ROOT / "rc_irstd/models/rc_mshnet.py",
    PROJECT_ROOT / "rc_irstd/models/__init__.py",
    PROJECT_ROOT / "rc_irstd/cli/train_detector.py",
    PROJECT_ROOT / "rc_irstd/cli/export_scores.py",
    PROJECT_ROOT / "rc_irstd/training/detector_trainer.py",
    PROJECT_ROOT / "evaluation/raw_logit_source_operating_point.py",
    PROJECT_ROOT / "evaluation/component_matching.py",
)

CONFIG_PATHS = (
    PROTOCOL,
    PROJECT_ROOT / "configs/tier2r_control.yaml",
    PROJECT_ROOT / "configs/tier2r_rc_mshnet_c.yaml",
    PROJECT_ROOT / "configs/tier2r_rc_mshnet_cv_v1.yaml",
)

Runner = Callable[..., subprocess.CompletedProcess[str]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _load_json_bytes(raw: str, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{context} did not emit one JSON document") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} JSON root is not an object")
    return payload


def _run(
    command: Sequence[str],
    *,
    runner: Runner | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    execute = subprocess.run if runner is None else runner
    completed = execute(
        list(command),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"command failed ({completed.returncode}): {command[0]}: {detail}"
        )
    return completed


def _bind_mount(source: Path, destination: Path, *, readonly: bool) -> list[str]:
    fields = [
        "type=bind",
        f"src={source}",
        f"dst={destination}",
    ]
    if readonly:
        fields.append("readonly")
    return ["--mount", ",".join(fields)]


def bind_mount_contract() -> tuple[dict[str, Any], ...]:
    """Return the complete host-bind contract without touching NUAA."""

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
            "source": str(AUDIT_ROOT),
            "destination": str(AUDIT_ROOT),
            "readonly": False,
        },
    )


def tmpfs_contract() -> tuple[dict[str, str], ...]:
    return (
        {
            "destination": "/tmp",
            "options": "rw,nosuid,nodev,noexec,mode=1777,size=8g",
        },
        {
            "destination": str(FORBIDDEN_TARGET),
            "options": "rw,nosuid,nodev,noexec,mode=000,size=4k",
        },
    )


def container_spec(*, formal: bool) -> dict[str, Any]:
    return {
        "image": IMAGE_ID,
        "name": CONTAINER_NAME if formal else None,
        "entrypoint": CONTAINER_PYTHON,
        "command": [str(COORDINATOR)] + ([] if formal else ["--verify-only"]),
        "working_directory": str(PROJECT_ROOT),
        "user": CONTAINER_USER,
        "network": "none",
        "read_only_rootfs": True,
        "init": True,
        "restart": "on-failure" if formal else None,
        "shm_size": "16g",
        "gpu_exposure": "all",
        "coordinator_physical_gpu_indices": list(TARGET_GPU_INDICES),
        "environment": dict(ENVIRONMENT),
        "labels": dict(RUNTIME_LABELS),
        "bind_mounts": list(bind_mount_contract()),
        "tmpfs": list(tmpfs_contract()),
        "cap_drop": ["ALL"],
        "security_options": ["no-new-privileges:true"],
    }


def _common_docker_args(*, labels: Mapping[str, str]) -> list[str]:
    command = [
        "docker",
        "run",
        "--network",
        "none",
        "--read-only",
        "--init",
        "--shm-size",
        "16g",
        "--user",
        CONTAINER_USER,
        "--gpus",
        "all",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--workdir",
        str(PROJECT_ROOT),
        "--entrypoint",
        CONTAINER_PYTHON,
    ]
    for key, value in sorted(ENVIRONMENT.items()):
        command.extend(["--env", f"{key}={value}"])
    for key, value in sorted(labels.items()):
        command.extend(["--label", f"{key}={value}"])
    for mount in bind_mount_contract():
        command.extend(
            _bind_mount(
                Path(mount["source"]),
                Path(mount["destination"]),
                readonly=bool(mount["readonly"]),
            )
        )
    for mount in tmpfs_contract():
        command.extend(
            ["--tmpfs", f"{mount['destination']}:{mount['options']}"]
        )
    return command


def build_verify_command() -> list[str]:
    labels = {**RUNTIME_LABELS, "org.rc-irstd.launch-purpose": "verify-only"}
    return [
        *_common_docker_args(labels=labels),
        "--rm",
        IMAGE_ID,
        str(COORDINATOR),
        "--verify-only",
    ]


def build_probe_command() -> list[str]:
    labels = {**RUNTIME_LABELS, "org.rc-irstd.launch-purpose": "isolation-probe"}
    return [
        *_common_docker_args(labels=labels),
        "--rm",
        IMAGE_ID,
        str(CONTAINER_PROBE),
    ]



def build_formal_command(intent_sha256: str) -> list[str]:
    if len(intent_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in intent_sha256
    ):
        raise ValueError("launch-intent SHA-256 must be lowercase hexadecimal")
    labels = {**RUNTIME_LABELS, INTENT_LABEL: intent_sha256}
    return [
        *_common_docker_args(labels=labels),
        "--detach",
        "--name",
        CONTAINER_NAME,
        "--restart",
        "on-failure",
        IMAGE_ID,
        str(COORDINATOR),
    ]


def _validate_host_sources() -> None:
    if Path.cwd().resolve() != PROJECT_ROOT.resolve():
        raise RuntimeError(f"launcher must run from {PROJECT_ROOT}")
    if os.getuid() != 1004 or os.getgid() != 1004:
        raise RuntimeError("launcher requires host uid/gid 1004:1004")
    for path in (*CODE_PATHS, *CONFIG_PATHS):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"immutable launch input is absent or a symlink: {path}")
    for path in (PROJECT_ROOT,):
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError(f"read-only mount source is invalid: {path}")
    home_root = Path("/home/ly")
    for mount in bind_mount_contract():
        source = Path(str(mount["source"]))
        if source.is_relative_to(home_root) and not source.is_relative_to(PROJECT_ROOT):
            raise RuntimeError(
                f"external project dependency is forbidden: {source}"
            )


def _ensure_exact_rw_roots() -> None:
    for path in (OUTPUT_ROOT, AUDIT_ROOT):
        path.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() or not path.is_dir() or path.resolve() != path:
            raise RuntimeError(f"writable root is not an exact canonical directory: {path}")
        path.relative_to(PROJECT_ROOT)


def _inspect_image(*, runner: Runner | None = None) -> str:
    completed = _run(
        ["docker", "image", "inspect", IMAGE_ID, "--format", "{{.Id}}"],
        runner=runner,
    )
    observed = completed.stdout.strip()
    if observed != IMAGE_ID:
        raise RuntimeError(f"pinned Docker image drift: expected {IMAGE_ID}, got {observed}")
    return observed


def _gpu_inventory(*, runner: Runner | None = None) -> tuple[dict[str, Any], ...]:
    completed = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        runner=runner,
    )
    inventory: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [value.strip() for value in line.split(",", 2)]
        if len(fields) != 3:
            raise RuntimeError(f"malformed nvidia-smi inventory line: {line!r}")
        inventory.append(
            {"index": int(fields[0]), "uuid": fields[1], "name": fields[2]}
        )
    if [value["index"] for value in inventory] != list(range(len(inventory))):
        raise RuntimeError("host GPU indices are not contiguous and stable")
    by_index = {value["index"]: value for value in inventory}
    if any(index not in by_index for index in TARGET_GPU_INDICES):
        raise RuntimeError("physical GPU 2/3 is absent")
    if len({value["uuid"] for value in inventory}) != len(inventory):
        raise RuntimeError("host GPU UUIDs are not unique")
    return tuple(inventory)


def _verify_in_container(*, runner: Runner | None = None) -> dict[str, Any]:
    completed = _run(build_verify_command(), runner=runner)
    payload = _load_json_bytes(completed.stdout, context="Tier2R verify-only container")
    schedule = payload.get("schedule")
    if (
        payload.get("verified") is not True
        or payload.get("source_only") is not True
        or payload.get("outer_target_access_authorized") is not False
        or not isinstance(schedule, list)
        or len(schedule) != 18
        or {value.get("physical_gpu") for value in schedule} != {2, 3}
    ):
        raise RuntimeError("Tier2R container verification contract failed closed")
    return payload

def _verify_container_attestation(
    host_inventory: Sequence[Mapping[str, Any]], *, runner: Runner | None = None
) -> dict[str, Any]:
    completed = _run(build_probe_command(), runner=runner)
    payload = _load_json_bytes(completed.stdout, context="Tier2R container probe")
    problems: list[str] = []
    if payload.get("home_ly_entries") != [PROJECT_ROOT.name]:
        problems.append("/home/ly contains a project other than RC-IRSTD-v2")
    target = payload.get("target")
    if (
        not isinstance(target, Mapping)
        or target.get("path") != str(FORBIDDEN_TARGET)
        or target.get("mode") != 0
        or target.get("list_error") != "PermissionError"
    ):
        problems.append("RC outer-target tmpfs mode=000 denial")
    mounts = payload.get("mounts")
    if not isinstance(mounts, Mapping):
        problems.append("mountinfo evidence absent")
        mounts = {}
    for path, writable in (
        (PROJECT_ROOT, False),
        (OUTPUT_ROOT, True),
        (AUDIT_ROOT, True),
        (FORBIDDEN_TARGET, True),
    ):
        entry = mounts.get(str(path))
        options = set(entry.get("mount_options", [])) if isinstance(entry, Mapping) else set()
        if not isinstance(entry, Mapping) or (("rw" in options) is not writable):
            problems.append(f"mount mode drift:{path}")
    target_mount = mounts.get(str(FORBIDDEN_TARGET))
    if (
        not isinstance(target_mount, Mapping)
        or target_mount.get("filesystem_type") != "tmpfs"
    ):
        problems.append("outer-target mount is not tmpfs")
    expected_nvidia = [dict(value) for value in host_inventory]
    if payload.get("nvidia_inventory") != expected_nvidia:
        problems.append("container nvidia-smi index/UUID mapping")
    observed_torch = payload.get("torch_inventory")
    if not isinstance(observed_torch, list):
        problems.append("Torch CUDA inventory absent")
    else:
        projected = [
            {"index": value.get("ordinal"), "uuid": value.get("uuid"), "name": value.get("name")}
            for value in observed_torch
            if isinstance(value, Mapping)
        ]
        if projected != expected_nvidia:
            problems.append("Torch ordinal/UUID mapping")
    environment = payload.get("environment")
    if (
        not isinstance(environment, Mapping)
        or environment.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
        or environment.get("CUDA_VISIBLE_DEVICES") is not None
        or environment.get("NVIDIA_VISIBLE_DEVICES") != "all"
    ):
        problems.append("container CUDA visibility environment")
    runtime = payload.get("runtime")
    exact_runtime = {
        "python": "3.11.12",
        "torch": "2.7.0+cu128",
        "torch_cuda": "12.8",
        "numpy": "1.26.4",
        "scipy": "1.17.1",
        "yaml": "6.0.2",
        "pandas": "3.0.2",
        "tqdm": "4.67.1",
    }
    if not isinstance(runtime, Mapping):
        problems.append("container runtime inventory absent")
    else:
        for key, expected in exact_runtime.items():
            if runtime.get(key) != expected:
                problems.append(f"runtime drift:{key}")
        for key, prefix in (("torchvision", "0.22."), ("PIL", "11."), ("skimage", "0.26.")):
            if not str(runtime.get(key, "")).startswith(prefix):
                problems.append(f"runtime drift:{key}")
    if problems:
        raise RuntimeError("container attestation failed: " + "; ".join(problems))
    payload["verified"] = True
    payload["host_gpu_uuid_mapping_verified"] = True
    payload["outer_target_alias_absent"] = True
    return payload



def _registered_at(path: Path) -> str:
    if not path.exists():
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"launch intent is not a regular file: {path}")
    payload = _load_json_bytes(path.read_text(encoding="utf-8"), context=str(path))
    value = payload.get("registered_at")
    if not isinstance(value, str) or not value:
        raise RuntimeError("existing launch intent lacks registered_at")
    return value


def _launch_intent_payload(
    *,
    image_id: str,
    gpu_inventory: Sequence[Mapping[str, Any]],
    attestation: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    assigned = [
        dict(value)
        for value in gpu_inventory
        if value.get("index") in TARGET_GPU_INDICES
    ]
    return {
        "schema_version": SCHEMA,
        "registered_at": _registered_at(INTENT_PATH),
        "decision": "LAUNCH_TIER2R_SOURCE_ONLY_IN_ISOLATED_CONTAINER",
        "source_only": True,
        "outer_target_access_authorized": False,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "container_name": CONTAINER_NAME,
        "image": {"requested": IMAGE_ID, "observed_id": image_id},
        "verification_container_spec": container_spec(formal=False),
        "formal_container_spec": container_spec(formal=True),
        "formal_container_intent_label": {
            "key": INTENT_LABEL,
            "value_contract": "sha256_of_canonical_LAUNCH_INTENT.json_bytes",
        },
        "host_gpu_inventory": [dict(value) for value in gpu_inventory],
        "coordinator_assigned_host_gpus": assigned,
        "container_attestation": dict(attestation),
        "container_attestation_canonical_sha256": hashlib.sha256(
            _canonical_bytes(dict(attestation))
        ).hexdigest(),
        "code_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in CODE_PATHS
        },
        "config_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in CONFIG_PATHS
        },
        "coordinator_verify_only": dict(verification),
        "coordinator_verify_only_canonical_sha256": hashlib.sha256(
            _canonical_bytes(dict(verification))
        ).hexdigest(),
        "failure_policy": "fail_closed_without_outer_target_access",
    }


def _register_launch_intent(payload: Mapping[str, Any]) -> str:
    raw = _canonical_bytes(payload)
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = f"{digest}  {INTENT_PATH.name}\n".encode("ascii")
    if INTENT_PATH.exists() or INTENT_SHA256_PATH.exists():
        if (
            INTENT_PATH.is_symlink()
            or INTENT_SHA256_PATH.is_symlink()
            or not INTENT_PATH.is_file()
            or not INTENT_SHA256_PATH.is_file()
            or INTENT_PATH.read_bytes() != raw
            or INTENT_SHA256_PATH.read_bytes() != sidecar
        ):
            raise RuntimeError("immutable LAUNCH_INTENT artifact drift")
        return digest
    _atomic_write(INTENT_PATH, raw)
    _atomic_write(INTENT_SHA256_PATH, sidecar)
    INTENT_PATH.chmod(0o444)
    INTENT_SHA256_PATH.chmod(0o444)
    return digest


def _inspect_existing_container(
    *, runner: Runner | None = None
) -> dict[str, Any] | None:
    completed = _run(
        ["docker", "container", "inspect", CONTAINER_NAME],
        runner=runner,
        check=False,
    )
    if completed.returncode != 0:
        if "No such" in completed.stderr or "No such" in completed.stdout:
            return None
        raise RuntimeError(
            "cannot determine existing container state: "
            + (completed.stderr or completed.stdout).strip()
        )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise RuntimeError("docker container inspect returned malformed JSON")
    return payload[0]


def _tmpfs_options(value: str) -> frozenset[str]:
    return frozenset(part.strip() for part in value.split(",") if part.strip())


def _validate_existing_container(container: Mapping[str, Any], intent_sha256: str) -> str:
    config = container.get("Config")
    host = container.get("HostConfig")
    state = container.get("State")
    mounts = container.get("Mounts")
    if not all(isinstance(value, Mapping) for value in (config, host, state)):
        raise RuntimeError("existing container inspect sections are incomplete")
    if not isinstance(mounts, list):
        raise RuntimeError("existing container mount inspection is absent")
    assert isinstance(config, Mapping) and isinstance(host, Mapping)
    assert isinstance(state, Mapping)
    labels = config.get("Labels") or {}
    environment = config.get("Env") or []
    restart = host.get("RestartPolicy") or {}
    expected_simple = (
        (container.get("Image") == IMAGE_ID, "image"),
        (config.get("User") == CONTAINER_USER, "user"),
        (config.get("WorkingDir") == str(PROJECT_ROOT), "working directory"),
        (config.get("Entrypoint") == [CONTAINER_PYTHON], "entrypoint"),
        (config.get("Cmd") == [str(COORDINATOR)], "command"),
        (host.get("NetworkMode") == "none", "network"),
        (host.get("ReadonlyRootfs") is True, "read-only rootfs"),
        (host.get("Init") is True, "init"),
        (host.get("ShmSize") == 16 * 1024**3, "shared memory"),
        (restart.get("Name") == "on-failure", "restart policy"),
        (labels.get(INTENT_LABEL) == intent_sha256, "intent label"),
    )
    drift = [name for valid, name in expected_simple if not valid]
    for key, value in ENVIRONMENT.items():
        if f"{key}={value}" not in environment:
            drift.append(f"environment:{key}")
    for key, value in RUNTIME_LABELS.items():
        if labels.get(key) != value:
            drift.append(f"label:{key}")
    observed_mounts = {
        (value.get("Source"), value.get("Destination")): value.get("RW")
        for value in mounts
        if isinstance(value, Mapping) and value.get("Type") == "bind"
    }
    expected_mounts = {
        (value["source"], value["destination"]): not value["readonly"]
        for value in bind_mount_contract()
    }
    if observed_mounts != expected_mounts:
        drift.append("bind mounts")
    observed_tmpfs = host.get("Tmpfs") or {}
    expected_tmpfs = {
        value["destination"]: _tmpfs_options(value["options"])
        for value in tmpfs_contract()
    }
    if set(observed_tmpfs) != set(expected_tmpfs) or any(
        _tmpfs_options(str(observed_tmpfs.get(path, ""))) != options
        for path, options in expected_tmpfs.items()
    ):
        drift.append("tmpfs")
    requests = host.get("DeviceRequests") or []
    if not any(
        isinstance(value, Mapping)
        and value.get("Count") == -1
        and ["gpu"] in (value.get("Capabilities") or [])
        for value in requests
    ):
        drift.append("GPU exposure")
    if drift:
        raise RuntimeError("existing named container contract drift: " + ", ".join(drift))
    status = state.get("Status")
    if status not in {"created", "running", "restarting", "exited"}:
        raise RuntimeError(f"existing named container is not safely reconcilable: {status}")
    return str(status)


def _start_or_reconcile(
    intent_sha256: str, *, runner: Runner | None = None
) -> dict[str, Any]:
    existing = _inspect_existing_container(runner=runner)
    if existing is None:
        completed = _run(build_formal_command(intent_sha256), runner=runner)
        container_id = completed.stdout.strip()
        if not container_id:
            raise RuntimeError("docker run did not return a container id")
        return {"action": "created_and_started", "container_id": container_id}
    status = _validate_existing_container(existing, intent_sha256)
    container_id = str(existing.get("Id", ""))
    if status in {"running", "restarting"}:
        return {"action": "already_active", "container_id": container_id, "status": status}
    completed = _run(
        ["docker", "container", "start", CONTAINER_NAME], runner=runner
    )
    return {
        "action": "existing_container_started",
        "container_id": completed.stdout.strip() or container_id,
        "previous_status": status,
    }


def execute_launch(*, runner: Runner | None = None) -> dict[str, Any]:
    """Verify, freeze intent, and idempotently start the formal container."""

    evidence = verify_only(runner=runner)
    intent = _launch_intent_payload(
        image_id=str(evidence["image_id"]),
        gpu_inventory=evidence["host_gpu_inventory"],
        attestation=evidence["container_attestation"],
        verification=evidence["coordinator_verification"],
    )
    intent_sha256 = _register_launch_intent(intent)
    launch = _start_or_reconcile(intent_sha256, runner=runner)
    return {
        "schema_version": SCHEMA,
        "verified_before_launch": True,
        "source_only": True,
        "outer_target_access_authorized": False,
        "launch_intent": str(INTENT_PATH),
        "launch_intent_sha256": intent_sha256,
        "container": launch,
    }


def verify_only(*, runner: Runner | None = None) -> dict[str, Any]:
    """Run the complete isolated preflight without registering or launching."""

    _validate_host_sources()
    _ensure_exact_rw_roots()
    image_id = _inspect_image(runner=runner)
    inventory = _gpu_inventory(runner=runner)
    verification = _verify_in_container(runner=runner)
    attestation = _verify_container_attestation(inventory, runner=runner)
    return {
        "schema_version": SCHEMA,
        "verified": True,
        "formal_container_started": False,
        "source_only": True,
        "outer_target_access_authorized": False,
        "image_id": image_id,
        "host_gpu_inventory": list(inventory),
        "coordinator_assigned_host_gpus": [
            dict(value) for value in inventory if value["index"] in TARGET_GPU_INDICES
        ],
        "container_attestation": attestation,
        "verification_container_spec": container_spec(formal=False),
        "formal_container_spec": container_spec(formal=True),
        "coordinator_verification": verification,
    }


def dry_run_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "dry_run": True,
        "docker_invoked": False,
        "filesystem_mutated": False,
        "container_name": CONTAINER_NAME,
        "image": IMAGE_ID,
        "source_only": True,
        "outer_target_access_authorized": False,
        "coordinator_physical_gpus": list(TARGET_GPU_INDICES),
        "verify_command": build_verify_command(),
        "formal_container_spec": container_spec(formal=True),
        "execute_with": f"python3 {Path(__file__).resolve()} --execute",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="run isolated verification, freeze launch intent, and start Docker",
    )
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
    "build_verify_command",
    "container_spec",
    "dry_run_payload",
    "execute_launch",
    "tmpfs_contract",
    "verify_only",
]
