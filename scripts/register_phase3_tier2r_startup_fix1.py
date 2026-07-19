#!/usr/bin/env python3
"""Register the Tier2R erratum-1 startup failure and its narrow fix.

This command never starts, restarts, recreates, reconfigures, or removes the
existing formal Docker container.  The default mode is a non-mutating
preflight.  ``--register`` runs exactly one fixed-image, no-GPU, read-only
``docker run --rm`` validation container and
writes two immutable evidence documents into the existing erratum audit root:
``STARTUP_FIX1.json`` and ``STARTUP_FIX1_VALIDATION.json``, each with a
``.json.sha256`` sidecar.  The module also exposes a Docker-free verifier for
the formal coordinator to consume after the failed container is restarted.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import fcntl
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any, BinaryIO


PROJECT_ROOT = Path("/home/ly/RC-IRSTD-v2")
AUDIT_ROOT = (
    PROJECT_ROOT
    / "artifacts/aaai27/audit/component_rescue/tier2r_c_v1_impl_erratum1"
)
OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs/aaai27/detectors/component_rescue/tier2r_c_v1"
)
OLD_AUDIT_ROOT = (
    PROJECT_ROOT / "artifacts/aaai27/audit/component_rescue/tier2r_c_v1"
)

INTENT_PATH = AUDIT_ROOT / "LAUNCH_INTENT.json"
INTENT_SIDECAR = AUDIT_ROOT / "LAUNCH_INTENT.json.sha256"
STARTUP_FIX1_PATH = AUDIT_ROOT / "STARTUP_FIX1.json"
STARTUP_FIX1_SIDECAR = AUDIT_ROOT / "STARTUP_FIX1.json.sha256"
VALIDATION_PATH = AUDIT_ROOT / "STARTUP_FIX1_VALIDATION.json"
VALIDATION_SIDECAR = AUDIT_ROOT / "STARTUP_FIX1_VALIDATION.json.sha256"

COORDINATOR_RELATIVE = (
    "scripts/coordinate_phase3_tier2r_component_rescue_impl_erratum1.py"
)
COORDINATOR_PATH = PROJECT_ROOT / COORDINATOR_RELATIVE
REGISTRAR_RELATIVE = "scripts/register_phase3_tier2r_startup_fix1.py"
REGISTRAR_PATH = PROJECT_ROOT / REGISTRAR_RELATIVE
OLD_LOCK_PATH = OLD_AUDIT_ROOT / ".tier2r_component_rescue.lock"

CONTAINER_NAME = "rc-irstd-tier2r-component-rescue-v1-impl-erratum1"
EXPECTED_CONTAINER_ID = (
    "137a82d8bae3c551b509accc473210379100d7d0bb117146a3a2d6418337a560"
)
EXPECTED_IMAGE_ID = (
    "sha256:42e03b9c7e2628bf7622008c79a6be33bca666a1dcb6d56a7d4d1e28f0c91fe3"
)
EXPECTED_INTENT_SHA256 = (
    "86ae3564dcace850a0f3ffdda966a98d4fee1be71b89f509e5134c86ca72c04f"
)
EXPECTED_OLD_COORDINATOR_SHA256 = (
    "bc0516c40231f83de49922469ad096c5de40d5145b74e04c8ebd95bc8530c37b"
)
EXPECTED_NEW_COORDINATOR_SHA256 = (
    "7d7243f608d0783392e173bd9f02107d4cafdd1e1c09543367683741690ebb61"
)
EXPECTED_FAILURE_LINE = (
    "FAILED_CLOSED OSError: [Errno 30] Read-only file system: "
    "'/home/ly/RC-IRSTD-v2/artifacts/aaai27/audit/component_rescue/"
    "tier2r_c_v1/.tier2r_component_rescue.lock'"
)
EXPECTED_DOCKER_RUNTIME = "runc"
TARGET_PATH = PROJECT_ROOT / "datasets/NUAA-SIRST"
TRANSIENT_VALIDATION_CONTAINER = (
    "rc-irstd-tier2r-startup-fix1-validation-ephemeral"
)
IMAGE_VALIDATION_SCHEMA = "rc-irstd-tier2r-startup-fix1-image-native-v1"
VALIDATION_PYTHON = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
EXPECTED_HOST_VALIDATION_RESOLVED_PYTHON = Path("/usr/bin/python3.12")
EXPECTED_HOST_VALIDATION_PYTHON_SHA256 = (
    "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
)
EXPECTED_HOST_VALIDATION_PYTHON_VERSION = "Python 3.12.3"
VALIDATION_TEST_RELATIVES = (
    "tests/test_phase3_tier2r_impl_erratum1.py",
    "tests/test_phase3_tier2r_startup_fix1.py",
)
EXPECTED_VALIDATION_TEST_COUNT = 22
EXPECTED_PYTEST_STDOUT_RE = (
    rf"\A\.{{{EXPECTED_VALIDATION_TEST_COUNT}}}\s+\[100%\]\n"
    rf"{EXPECTED_VALIDATION_TEST_COUNT} passed in "
    r"[0-9]+(?:\.[0-9]+)?s\n\Z"
)
EXPECTED_DYNAMIC_OLD_PLANNED_SHA256 = (
    "f6d3bf731d98c3cd57ac87ceb3ed08e96c0e4dd6493d40c4e9a408eeda0dd840"
)

SCIENTIFIC_PROTOCOL_ID = "tier2r_c_v1"
EXECUTION_INSTANCE = "tier2r_c_v1_impl_erratum1"
STARTUP_SCHEMA = "rc-irstd-aaai27-tier2r-startup-fix1-v2"
VALIDATION_SCHEMA = "rc-irstd-aaai27-tier2r-startup-fix1-validation-v2"

BASE_AUDIT_ENTRIES = frozenset(
    {INTENT_PATH.name, INTENT_SIDECAR.name}
)
STARTUP_ENTRIES = frozenset(
    {STARTUP_FIX1_PATH.name, STARTUP_FIX1_SIDECAR.name}
)
VALIDATION_ENTRIES = frozenset(
    {VALIDATION_PATH.name, VALIDATION_SIDECAR.name}
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def _canonical_container_inspect_sha256(
    container: Mapping[str, Any],
) -> str:
    """Hash a full Docker inspect object after normalizing unordered mounts."""

    normalized = dict(container)
    mounts = container.get("Mounts")
    if isinstance(mounts, list):
        if any(not isinstance(record, Mapping) for record in mounts):
            raise RuntimeError("Docker inspect Mounts contains a non-object")
        normalized["Mounts"] = sorted(
            (dict(record) for record in mounts),
            key=_canonical_bytes,
        )
    return hashlib.sha256(_canonical_bytes(normalized)).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return payload


def _assert_regular(path: Path, *, readonly: bool = False) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required regular file is absent or a symlink: {path}")
    if readonly and stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise RuntimeError(f"immutable evidence remains writable: {path}")


def _verify_sidecar(path: Path, sidecar: Path, *, readonly: bool = True) -> str:
    _assert_regular(path, readonly=readonly)
    _assert_regular(sidecar, readonly=readonly)
    digest = _sha256(path)
    expected = f"{digest}  {path.name}\n"
    if sidecar.read_text(encoding="ascii") != expected:
        raise RuntimeError(f"SHA-256 sidecar drift: {sidecar}")
    return digest


def _binding(path: Path, sidecar: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": _verify_sidecar(path, sidecar),
    }


_PENDING_PREFIX = ".startup-fix1-pending-"


@contextlib.contextmanager
def _registration_lock() -> Any:
    """Serialize registrars without creating a mutable lock artifact."""

    descriptor = os.open(AUDIT_ROOT, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another startup-fix1 registrar owns the audit root") from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _cleanup_pending_publications() -> None:
    """Recover only this registrar's unpublished/linked staging inodes."""

    for path in AUDIT_ROOT.glob(f"{_PENDING_PREFIX}*"):
        if path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid():
            raise RuntimeError(f"unsafe startup-fix1 pending publication: {path}")
        path.unlink()


def _publish_noreplace(path: Path, content: bytes) -> None:
    """Publish complete read-only bytes with link(2), never replacement."""

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=_PENDING_PREFIX, delete=False
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
            temporary = Path(handle.name)
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise RuntimeError(f"write-once destination already exists: {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_once_locked(
    path: Path, sidecar: Path, payload: Mapping[str, Any]
) -> str:
    raw = _canonical_bytes(payload)
    digest = hashlib.sha256(raw).hexdigest()
    sidecar_raw = f"{digest}  {path.name}\n".encode("ascii")
    if path.exists() or sidecar.exists():
        if (
            path.is_symlink()
            or sidecar.is_symlink()
            or not path.is_file()
            or not sidecar.is_file()
            or path.read_bytes() != raw
            or sidecar.read_bytes() != sidecar_raw
            or stat.S_IMODE(path.stat().st_mode) != 0o444
            or stat.S_IMODE(sidecar.stat().st_mode) != 0o444
        ):
            raise RuntimeError(f"immutable startup-fix evidence drift: {path}")
        return digest
    _publish_noreplace(path, raw)
    _publish_noreplace(sidecar, sidecar_raw)
    return digest


def _write_once(
    path: Path,
    sidecar: Path,
    payload: Mapping[str, Any],
    *,
    lock_held: bool = False,
) -> str:
    if lock_held:
        return _write_once_locked(path, sidecar, payload)
    with _registration_lock():
        _cleanup_pending_publications()
        return _write_once_locked(path, sidecar, payload)


def _recover_missing_sidecar(
    path: Path,
    sidecar: Path,
    semantic_validator: Callable[[Mapping[str, Any]], None],
) -> None:
    """Finish a path-first publication interrupted before its sidecar link."""

    if not path.exists() and sidecar.exists():
        raise RuntimeError(f"orphan startup-fix sidecar: {sidecar}")
    if not path.exists() or sidecar.exists():
        return
    _assert_regular(path, readonly=True)
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise RuntimeError(f"incomplete startup-fix artifact mode drift: {path}")
    payload = _load_object(path)
    if path.read_bytes() != _canonical_bytes(payload):
        raise RuntimeError(f"incomplete startup-fix artifact is not canonical: {path}")
    semantic_validator(payload)
    digest = _sha256(path)
    _publish_noreplace(sidecar, f"{digest}  {path.name}\n".encode("ascii"))


def _registered_at(path: Path) -> str:
    if not path.exists():
        return _now()
    payload = _load_object(path)
    value = payload.get("registered_at")
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"existing evidence lacks registered_at: {path}")
    return value


def _verify_common_semantics(payload: Mapping[str, Any], schema: str) -> None:
    registered_at = payload.get("registered_at")
    try:
        parsed = (
            datetime.fromisoformat(registered_at)
            if isinstance(registered_at, str)
            else None
        )
    except ValueError as error:
        raise RuntimeError("startup-fix registered_at drift") from error
    if parsed is None or parsed.tzinfo is None:
        raise RuntimeError("startup-fix registered_at must be timezone-aware")
    expected = _common_fields()
    if payload.get("schema_version") != schema or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("startup-fix common semantic drift")


def _verify_stream_record(record: Mapping[str, Any]) -> None:
    stdout, stderr = record.get("stdout"), record.get("stderr")
    if (
        not isinstance(record.get("command"), list)
        or not isinstance(record.get("returncode"), int)
        or not isinstance(stdout, str)
        or not isinstance(stderr, str)
        or record.get("stdout_sha256")
        != hashlib.sha256(stdout.encode()).hexdigest()
        or record.get("stderr_sha256")
        != hashlib.sha256(stderr.encode()).hexdigest()
    ):
        raise RuntimeError("frozen process stream evidence drift")


def _verify_absent_record(record: Mapping[str, Any]) -> None:
    _verify_stream_record(record)
    expected_stderr = (
        "Error response from daemon: No such container: "
        f"{TRANSIENT_VALIDATION_CONTAINER}\n"
    )
    if (
        record.get("command")
        != ["docker", "container", "inspect", TRANSIENT_VALIDATION_CONTAINER]
        or record.get("returncode") != 1
        or record.get("stdout") != "[]\n"
        or record.get("stderr") != expected_stderr
        or record.get("absent") is not True
    ):
        raise RuntimeError("transient-container absence evidence drift")


def _expected_resume_authorization() -> dict[str, Any]:
    return {
        "action": "START_EXISTING_CONTAINER_ONLY",
        "authorized": True,
        "container_name": CONTAINER_NAME,
        "container_id": EXPECTED_CONTAINER_ID,
        "image_id": EXPECTED_IMAGE_ID,
        "docker_start_only": True,
        "container_create_authorized": False,
        "container_recreate_authorized": False,
        "container_remove_authorized": False,
        "container_reconfigure_authorized": False,
        "launcher_reexecution_authorized": False,
        "effective_only_after_validation_sidecar_is_frozen": True,
    }


def _verify_intent() -> tuple[dict[str, Any], str]:
    digest = _verify_sidecar(INTENT_PATH, INTENT_SIDECAR)
    if digest != EXPECTED_INTENT_SHA256:
        raise RuntimeError("implementation-erratum launch-intent SHA-256 drift")
    intent = _load_object(INTENT_PATH)
    formal_spec = intent.get("formal_container_spec", {})
    labels = formal_spec.get("labels", {})
    attestation = intent.get("container_attestation")
    planned = (
        intent.get("coordinator_verify_only", {})
        .get("implementation_erratum", {})
        .get("planned_sha256")
    )
    if (
        intent.get("scientific_protocol_id") != SCIENTIFIC_PROTOCOL_ID
        or intent.get("execution_instance_id") != EXECUTION_INSTANCE
        or intent.get("source_only") is not True
        or intent.get("outer_target_access_authorized") is not False
        or intent.get("outer_target_images_used") is not False
        or intent.get("outer_target_labels_used") is not False
        or intent.get("container_name") != CONTAINER_NAME
        or labels.get("org.rc-irstd.outer-target-access") != "denied"
        or not isinstance(attestation, Mapping)
        or intent.get("container_attestation_canonical_sha256")
        != hashlib.sha256(_canonical_bytes(attestation)).hexdigest()
        or planned != EXPECTED_DYNAMIC_OLD_PLANNED_SHA256
    ):
        raise RuntimeError("implementation-erratum launch-intent contract drift")
    return intent, digest


def _audit_entries() -> frozenset[str]:
    if AUDIT_ROOT.is_symlink() or not AUDIT_ROOT.is_dir():
        raise RuntimeError(f"erratum audit root is absent or invalid: {AUDIT_ROOT}")
    return frozenset(path.name for path in AUDIT_ROOT.iterdir())


def _assert_registration_surface() -> dict[str, Any]:
    entries = _audit_entries()
    full_surface = BASE_AUDIT_ENTRIES | STARTUP_ENTRIES | VALIDATION_ENTRIES
    if not BASE_AUDIT_ENTRIES.issubset(entries) or not entries.issubset(full_surface):
        raise RuntimeError(
            "formal erratum artifacts exist outside the startup-fix evidence surface: "
            + ",".join(sorted(entries))
        )
    forbidden_output = sorted(
        str(path.relative_to(OUTPUT_ROOT))
        for path in OUTPUT_ROOT.rglob("*")
        if path.name == "TIER2R_EXPORT_IDENTITY.json"
        or "scores_heldout_train" in path.parts
    )
    if forbidden_output:
        raise RuntimeError("formal score exports already exist before startup-fix1")
    return {
        "audit_entries_before_startup_fix1": sorted(BASE_AUDIT_ENTRIES),
        "implementation_erratum_registered": False,
        "preregistration_registered": False,
        "status_registered": False,
        "handoff_registered": False,
        "formal_score_exports_registered": False,
    }


def _host_gpu_inventory() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "cannot inspect host GPU UUID inventory: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    result: list[dict[str, Any]] = []
    for line in completed.stdout.decode("utf-8", errors="strict").splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            raise RuntimeError("host GPU inventory output drift")
        result.append({"index": int(fields[0]), "name": fields[1], "uuid": fields[2]})
    return result


def _environment_map(values: Any) -> dict[str, str]:
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise RuntimeError("container environment is malformed")
    result: dict[str, str] = {}
    for value in values:
        key, separator, content = value.partition("=")
        if not separator or key in result:
            raise RuntimeError("container environment key drift")
        result[key] = content
    return result


def _validate_container_contract(
    container: Mapping[str, Any],
    intent: Mapping[str, Any],
    host_gpu_inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    spec = intent.get("formal_container_spec")
    attestation = intent.get("container_attestation")
    host = container.get("HostConfig")
    config = container.get("Config")
    mounts = container.get("Mounts")
    if not all(isinstance(value, Mapping) for value in (spec, attestation, host, config)):
        raise RuntimeError("container/intent contract sections are incomplete")
    if not isinstance(mounts, list):
        raise RuntimeError("container mount list is absent")

    expected_mounts = sorted(
        (
            {
                "Type": "bind",
                "Source": str(record["source"]),
                "Destination": str(record["destination"]),
                "RW": not bool(record["readonly"]),
            }
            for record in spec.get("bind_mounts", [])
        ),
        key=lambda item: item["Destination"],
    )
    observed_mounts = sorted(
        (
            {
                "Type": record.get("Type"),
                "Source": record.get("Source"),
                "Destination": record.get("Destination"),
                "RW": record.get("RW"),
            }
            for record in mounts
        ),
        key=lambda item: str(item["Destination"]),
    )
    expected_tmpfs = {
        str(record["destination"]): str(record["options"])
        for record in spec.get("tmpfs", [])
    }
    observed_tmpfs = dict(host.get("Tmpfs", {}))
    expected_environment = dict(spec.get("environment", {}))
    observed_environment = _environment_map(config.get("Env"))
    expected_device_requests = [
        {
            "Driver": "",
            "Count": -1,
            "DeviceIDs": None,
            "Capabilities": [["gpu"]],
            "Options": {},
        }
    ]
    expected_inventory = list(intent.get("host_gpu_inventory", []))
    observed_inventory = [dict(record) for record in host_gpu_inventory]
    attested_inventory = list(attestation.get("nvidia_inventory", []))
    torch_inventory = [
        {"index": record.get("ordinal"), "name": record.get("name"), "uuid": record.get("uuid")}
        for record in attestation.get("torch_inventory", [])
    ]
    assigned = list(intent.get("coordinator_assigned_host_gpus", []))
    expected_assigned = [record for record in expected_inventory if record.get("index") in (2, 3)]
    if (
        observed_mounts != expected_mounts
        or observed_tmpfs != expected_tmpfs
        or any(observed_environment.get(key) != value for key, value in expected_environment.items())
        or host.get("DeviceRequests") != expected_device_requests
        or host.get("Runtime") != EXPECTED_DOCKER_RUNTIME
        or config.get("WorkingDir") != spec.get("working_directory")
        or config.get("User") != spec.get("user")
        or config.get("Entrypoint") != [spec.get("entrypoint")]
        or host.get("Init") is not True
        or host.get("ShmSize") != 16 * 1024**3
        or sorted(host.get("CapDrop") or []) != sorted(spec.get("cap_drop", []))
        or sorted(host.get("SecurityOpt") or []) != sorted(spec.get("security_options", []))
        or observed_inventory != expected_inventory
        or attested_inventory != expected_inventory
        or torch_inventory != expected_inventory
        or assigned != expected_assigned
        or attestation.get("host_gpu_uuid_mapping_verified") is not True
        or attestation.get("outer_target_alias_absent") is not True
        or attestation.get("target", {}).get("mode") != 0
        or attestation.get("target", {}).get("list_error") != "PermissionError"
    ):
        raise RuntimeError("failed container mount/runtime/environment/GPU contract drift")
    runtime = attestation.get("runtime")
    if not isinstance(runtime, Mapping) or not runtime:
        raise RuntimeError("container package-runtime attestation is absent")
    return {
        "bind_mounts": expected_mounts,
        "tmpfs": expected_tmpfs,
        "required_environment": expected_environment,
        "full_environment_canonical_sha256": hashlib.sha256(
            _canonical_bytes(observed_environment)
        ).hexdigest(),
        "device_requests": expected_device_requests,
        "docker_runtime": EXPECTED_DOCKER_RUNTIME,
        "package_runtime": dict(runtime),
        "package_runtime_canonical_sha256": hashlib.sha256(
            _canonical_bytes(runtime)
        ).hexdigest(),
        "host_gpu_inventory": expected_inventory,
        "assigned_host_gpus": expected_assigned,
        "nuaa_tmpfs_mode": 0,
        "matches_launch_intent": True,
    }


def _inspect_container(
    intent: Mapping[str, Any], intent_sha256: str
) -> dict[str, Any]:
    completed = subprocess.run(
        ["docker", "container", "inspect", CONTAINER_NAME],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "cannot inspect failed erratum container: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    raw = completed.stdout
    inspected = json.loads(raw)
    if (
        not isinstance(inspected, list)
        or len(inspected) != 1
        or not isinstance(inspected[0], dict)
    ):
        raise RuntimeError("failed container inspect JSON is malformed")
    container = inspected[0]
    state = container.get("State")
    host = container.get("HostConfig")
    config = container.get("Config")
    if not all(isinstance(value, Mapping) for value in (state, host, config)):
        raise RuntimeError("failed container inspect sections are incomplete")
    assert isinstance(state, Mapping)
    assert isinstance(host, Mapping)
    assert isinstance(config, Mapping)
    restart = host.get("RestartPolicy")
    labels = config.get("Labels")
    contract = _validate_container_contract(
        container, intent, _host_gpu_inventory()
    )
    if (
        container.get("Id") != EXPECTED_CONTAINER_ID
        or container.get("Image") != EXPECTED_IMAGE_ID
        or state.get("Status") != "exited"
        or state.get("Running") is not False
        or state.get("Restarting") is not False
        or state.get("OOMKilled") is not False
        or state.get("ExitCode") != 1
        or container.get("RestartCount") != 3
        or not isinstance(restart, Mapping)
        or restart.get("Name") != "on-failure"
        or restart.get("MaximumRetryCount") != 3
        or not isinstance(labels, Mapping)
        or labels.get("org.rc-irstd.launch-intent-sha256") != intent_sha256
        or labels.get("org.rc-irstd.source-only") != "true"
        or labels.get("org.rc-irstd.outer-target-access") != "denied"
        or config.get("Cmd") != [str(COORDINATOR_PATH)]
        or host.get("NetworkMode") != "none"
        or host.get("ReadonlyRootfs") is not True
    ):
        raise RuntimeError("failed startup container contract drift")
    logs = subprocess.run(
        ["docker", "container", "logs", CONTAINER_NAME],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout
    decoded = logs.decode("utf-8", errors="strict").splitlines()
    if decoded != [EXPECTED_FAILURE_LINE] * 4:
        raise RuntimeError("startup failure log is not exactly four readonly-lock errors")
    return {
        "name": CONTAINER_NAME,
        "id": str(container["Id"]),
        "image_id": str(container["Image"]),
        "created_at": container.get("Created"),
        "started_at": state.get("StartedAt"),
        "finished_at": state.get("FinishedAt"),
        "status": state.get("Status"),
        "exit_code": state.get("ExitCode"),
        "oom_killed": state.get("OOMKilled"),
        "restart_count": container.get("RestartCount"),
        "restart_policy": {
            "Name": restart.get("Name"),
            "MaximumRetryCount": restart.get("MaximumRetryCount"),
        },
        # Docker exposes top-level Mounts as an unordered array and may return
        # the same bind set in a different order after any unrelated run.
        # Bind the complete normalized object; raw byte order is not semantic.
        "inspect_canonical_sha256": _canonical_container_inspect_sha256(
            container
        ),
        "failure_log_sha256": hashlib.sha256(logs).hexdigest(),
        "failure_log_num_lines": len(decoded),
        "failure_log_line": EXPECTED_FAILURE_LINE,
        "configuration_contract": contract,
    }


def _code_drift(intent: Mapping[str, Any]) -> dict[str, Any]:
    bindings = intent.get("code_sha256")
    configs = intent.get("config_sha256")
    if not isinstance(bindings, Mapping) or not isinstance(configs, Mapping):
        raise RuntimeError("launch intent code/config closure is absent")
    observed: dict[str, str] = {}
    for relative in bindings:
        path = PROJECT_ROOT / str(relative)
        _assert_regular(path)
        observed[str(relative)] = _sha256(path)
    mismatches = sorted(
        relative
        for relative, expected in bindings.items()
        if observed[str(relative)] != expected
    )
    if mismatches != [COORDINATOR_RELATIVE]:
        raise RuntimeError(
            "startup-fix code drift must be exactly the erratum coordinator: "
            + ",".join(mismatches)
        )
    if bindings.get(COORDINATOR_RELATIVE) != EXPECTED_OLD_COORDINATOR_SHA256:
        raise RuntimeError("launch-intent coordinator SHA-256 drift")
    if observed[COORDINATOR_RELATIVE] != EXPECTED_NEW_COORDINATOR_SHA256:
        raise RuntimeError("approved startup-fix1 coordinator SHA-256 drift")
    for relative, expected in configs.items():
        path = PROJECT_ROOT / str(relative)
        _assert_regular(path)
        if _sha256(path) != expected:
            raise RuntimeError(f"startup-fix config drift: {relative}")
    transition = {
        "path": COORDINATOR_RELATIVE,
        "old_sha256": str(bindings[COORDINATOR_RELATIVE]),
        "new_sha256": observed[COORDINATOR_RELATIVE],
        "approved_scope": [
            "readonly_existing_historical_lock_acquisition",
            "startup_fix1_and_validation_binding_propagation",
        ],
    }
    return {
        "allowed_changed_paths": [COORDINATOR_RELATIVE],
        "old_sha256": str(bindings[COORDINATOR_RELATIVE]),
        "new_sha256": observed[COORDINATOR_RELATIVE],
        "other_intent_code_bindings_unchanged": True,
        "all_intent_config_bindings_unchanged": True,
        "matched_other_code_bindings": len(bindings) - 1,
        "canonical_transition": transition,
        "canonical_transition_sha256": hashlib.sha256(
            _canonical_bytes(transition)
        ).hexdigest(),
    }


def _readonly_lock_contract() -> dict[str, Any]:
    _assert_regular(OLD_LOCK_PATH)
    if OLD_LOCK_PATH.stat().st_size != 0:
        raise RuntimeError("historical coordinator lock inode is not empty")
    tree = ast.parse(
        COORDINATOR_PATH.read_text(encoding="utf-8"), filename=str(COORDINATOR_PATH)
    )
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_exclusive_existing_readonly_lock"
    ]
    if len(functions) != 1:
        raise RuntimeError("readonly historical-lock helper definition drift")
    function = functions[0]
    open_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "open"
    ]
    if len(open_calls) != 1:
        raise RuntimeError("readonly historical-lock helper open-call drift")
    call = open_calls[0]
    buffering = {
        keyword.arg: keyword.value
        for keyword in call.keywords
        if keyword.arg is not None
    }.get("buffering")
    if (
        len(call.args) != 1
        or not isinstance(call.args[0], ast.Constant)
        or call.args[0].value != "rb"
        or not isinstance(buffering, ast.Constant)
        or buffering.value != 0
    ):
        raise RuntimeError("historical lock must be opened exactly rb/buffering=0")
    source = ast.unparse(function)
    if "LOCK_EX | fcntl.LOCK_NB" not in source or "a+b" in source:
        raise RuntimeError("readonly historical-lock flock semantics drift")
    handle: BinaryIO | None = None
    try:
        handle = OLD_LOCK_PATH.open("rb", buffering=0)
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        if handle is not None:
            handle.close()
        raise RuntimeError("historical coordinator lock is unexpectedly owned") from error
    finally:
        if handle is not None and not handle.closed:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
    metadata = OLD_LOCK_PATH.stat()
    return {
        "path": str(OLD_LOCK_PATH.resolve()),
        "sha256": _sha256(OLD_LOCK_PATH),
        "size_bytes": metadata.st_size,
        "inode": metadata.st_ino,
        "mode": stat.S_IMODE(metadata.st_mode),
        "open_mode": "rb",
        "buffering": 0,
        "create_if_absent": False,
        "flock": "LOCK_EX|LOCK_NB",
        "nonblocking_readonly_open_probe_passed": True,
    }


def _command_evidence(
    command: Sequence[str], *, environment: Mapping[str, str] | None = None
) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=PROJECT_ROOT,
        env=None if environment is None else dict(environment),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    evidence = {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "stdout": completed.stdout.decode("utf-8", errors="replace"),
        "stderr": completed.stderr.decode("utf-8", errors="replace"),
    }
    if completed.returncode != 0:
        raise RuntimeError(
            "startup-fix1 validation command failed: " + " ".join(command)
        )
    return evidence


def _image_native_script() -> str:
    template = r'''import glob, hashlib, json, os, py_compile, socket, stat, sys
from pathlib import Path

ROOT = Path(__PROJECT_ROOT__)
TARGET = Path(__TARGET_PATH__)
COMPILE_PATHS = [
    ROOT / "scripts/register_phase3_tier2r_startup_fix1.py",
    ROOT / "scripts/coordinate_phase3_tier2r_component_rescue_impl_erratum1.py",
    ROOT / "scripts/tier2r_impl_erratum1.py",
]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

if sys.version_info[:3] != (3, 11, 12):
    raise RuntimeError("fixed image is not Python 3.11.12")
for path in COMPILE_PATHS:
    py_compile.compile(str(path), doraise=True)

from scripts import coordinate_phase3_tier2r_component_rescue_impl_erratum1 as coordinator
from scripts import tier2r_impl_erratum1 as helper
import torch

coordinator_output = coordinator.run(verify_only=True)
if not isinstance(coordinator_output, dict):
    raise RuntimeError("coordinator verify-only output is not an object")
payload = helper._build_payload(create_snapshots=False)
stamp = payload["registered_at"]
helper._now = lambda: stamp
observed = helper.ensure_implementation_erratum(create=False)
normalized = dict(payload)
normalized.pop("registered_at", None)
bindings = payload["recovery_code_bindings"]
if observed.get("registered") is not False:
    raise RuntimeError("helper verify-only unexpectedly registered evidence")
if bindings.get("scripts/coordinate_phase3_tier2r_component_rescue_impl_erratum1.py") != __COORDINATOR_SHA__:
    raise RuntimeError("helper coordinator binding drift")
target_mode = stat.S_IMODE(TARGET.stat().st_mode)
target_denied = False
try:
    list(TARGET.iterdir())
except PermissionError:
    target_denied = True
if target_mode != 0 or not target_denied:
    raise RuntimeError("NUAA target tmpfs is not mode000/PermissionError")

def mount_record(path):
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, separator, right = line.partition(" - ")
        fields = left.split()
        if separator and len(fields) >= 6 and fields[4] == str(path):
            tail = right.split()
            return {"mount_point": fields[4], "mount_options": fields[5].split(","),
                    "filesystem_type": tail[0], "source": tail[1],
                    "super_options": tail[2].split(",")}
    raise RuntimeError("required mountinfo record absent: " + str(path))

project_mount = mount_record(ROOT)
target_mount = mount_record(TARGET)
tmp_mount = mount_record(Path("/tmp"))
interfaces = sorted(name for _, name in socket.if_nameindex())
nvidia_devices = sorted(glob.glob("/dev/nvidia*"))
torch_cuda_available = torch.cuda.is_available()
torch_cuda_device_count = torch.cuda.device_count()
if ("ro" not in project_mount["mount_options"]
        or target_mount["filesystem_type"] != "tmpfs"
        or tmp_mount["filesystem_type"] != "tmpfs"
        or interfaces != ["lo"] or nvidia_devices
        or torch_cuda_available or torch_cuda_device_count != 0
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("NVIDIA_VISIBLE_DEVICES") != "void"):
    raise RuntimeError("ephemeral mount/network/no-GPU isolation drift")
schedule = coordinator_output.get("schedule")
if (coordinator_output.get("verified") is not True
        or coordinator_output.get("source_only") is not True
        or coordinator_output.get("outer_target_images_used") is not False
        or coordinator_output.get("outer_target_labels_used") is not False
        or coordinator_output.get("outer_target_access_authorized") is not False
        or not isinstance(schedule, list) or len(schedule) != 18
        or {record.get("physical_gpu") for record in schedule} != {2, 3}
        or coordinator_output.get("implementation_erratum", {}).get("registered") is not False):
    raise RuntimeError("coordinator verify-only semantic drift")

result = {
    "schema_version": __IMAGE_SCHEMA__,
    "runtime": {
        "executable": sys.executable,
        "python_version": sys.version,
        "python_version_info": list(sys.version_info[:3]),
    },
    "py_compile": {
        "passed": True,
        "paths": [str(path) for path in COMPILE_PATHS],
        "sha256": {str(path.relative_to(ROOT)): sha256(path) for path in COMPILE_PATHS},
    },
    "coordinator_verify_only": {
        "call": "coordinator.run(verify_only=True)",
        "output": coordinator_output,
        "output_canonical_sha256": hashlib.sha256(
            (json.dumps(coordinator_output, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":")) + "\n").encode("utf-8")
        ).hexdigest(),
    },
    "helper_normalized_planning": {
        "call": "helper._build_payload(create_snapshots=False); helper.ensure_implementation_erratum(create=False)",
        "observed": observed,
        "observed_registered_at": stamp,
        "normalized_payload_sha256": hashlib.sha256(
            helper._canonical_json_bytes(normalized)
        ).hexdigest(),
        "recovery_code_bindings": bindings,
    },
    "target_lock": {
        "path": str(TARGET),
        "mode": target_mode,
        "list_error": "PermissionError",
    },
    "isolation": {
        "project_mount": project_mount, "target_mount": target_mount,
        "tmp_mount": tmp_mount, "network_interfaces": interfaces,
        "nvidia_devices": nvidia_devices,
        "torch_cuda_available": torch_cuda_available,
        "torch_cuda_device_count": torch_cuda_device_count,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    },
}
print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
'''
    return (
        template.replace("__PROJECT_ROOT__", repr(str(PROJECT_ROOT)))
        .replace("__TARGET_PATH__", repr(str(TARGET_PATH)))
        .replace("__COORDINATOR_SHA__", repr(EXPECTED_NEW_COORDINATOR_SHA256))
        .replace("__IMAGE_SCHEMA__", repr(IMAGE_VALIDATION_SCHEMA))
    )


def _image_native_command(script: str) -> list[str]:
    return [
        "docker", "run", "--rm", "--name", TRANSIENT_VALIDATION_CONTAINER,
        "--pull=never", "--network", "none", "--read-only", "--runtime", "runc",
        "--mount", f"type=bind,source={PROJECT_ROOT},target={PROJECT_ROOT},readonly",
        "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,mode=1777,size=256m",
        "--tmpfs", f"{TARGET_PATH}:rw,nosuid,nodev,noexec,mode=000,size=4k",
        "--workdir", str(PROJECT_ROOT), "--user", "1004:1004",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
        "--pids-limit", "256",
        "--env", f"PYTHONPATH={PROJECT_ROOT}",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", "PYTHONPYCACHEPREFIX=/tmp/pycache",
        "--env", "HOME=/tmp",
        "--env", "CUDA_VISIBLE_DEVICES=",
        "--env", "NVIDIA_VISIBLE_DEVICES=void",
        "--entrypoint", "python", EXPECTED_IMAGE_ID, "-c", script,
    ]


def _absent_container_evidence(name: str) -> dict[str, Any]:
    command = ["docker", "container", "inspect", name]
    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    expected_stderr = (
        f"Error response from daemon: No such container: {name}\n"
    ).encode("utf-8")
    if completed.returncode == 0:
        raise RuntimeError(f"transient validation container unexpectedly exists: {name}")
    if (
        completed.returncode != 1
        or completed.stdout != b"[]\n"
        or completed.stderr != expected_stderr
    ):
        raise RuntimeError("transient absent-container inspect contract drift")
    return {
        "command": command, "returncode": completed.returncode,
        "stdout": completed.stdout.decode("utf-8", errors="strict"),
        "stderr": completed.stderr.decode("utf-8", errors="strict"),
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "absent": True,
    }


def _verify_image_native_record(record: Mapping[str, Any]) -> dict[str, Any]:
    script = _image_native_script()
    expected_command = _image_native_command(script)
    command = record.get("command")
    if command != expected_command or any(
        value == forbidden or value.startswith(forbidden + "=")
        for value in command if isinstance(value, str)
        for forbidden in ("--gpus", "--device", "--restart")
    ):
        raise RuntimeError("image-native validation argv drift")
    stdout = record.get("stdout")
    stderr = record.get("stderr")
    if (record.get("returncode") != 0 or not isinstance(stdout, str)
            or stderr != ""
            or record.get("stdout_sha256") != hashlib.sha256(stdout.encode()).hexdigest()
            or record.get("stderr_sha256") != hashlib.sha256(stderr.encode()).hexdigest()):
        raise RuntimeError("image-native validation process evidence drift")
    payload = json.loads(stdout)
    if stdout != _canonical_bytes(payload).decode("utf-8"):
        raise RuntimeError("image-native validation output is not canonical JSON")
    compile_record = payload.get("py_compile", {})
    expected_paths = [REGISTRAR_PATH, COORDINATOR_PATH,
                      PROJECT_ROOT / "scripts/tier2r_impl_erratum1.py"]
    expected_hashes = {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in expected_paths
    }
    helper = payload.get("helper_normalized_planning", {})
    runtime = payload.get("runtime", {})
    target = payload.get("target_lock", {})
    coordinator = payload.get("coordinator_verify_only", {})
    if (payload.get("schema_version") != IMAGE_VALIDATION_SCHEMA
            or runtime.get("python_version_info") != [3, 11, 12]
            or compile_record.get("passed") is not True
            or compile_record.get("paths") != [str(path) for path in expected_paths]
            or compile_record.get("sha256") != expected_hashes
            or not isinstance(coordinator.get("output"), Mapping)
            or coordinator.get("call") != "coordinator.run(verify_only=True)"
            or helper.get("observed", {}).get("registered") is not False
            or helper.get("recovery_code_bindings", {}).get(COORDINATOR_RELATIVE)
               != EXPECTED_NEW_COORDINATOR_SHA256
            or not isinstance(helper.get("normalized_payload_sha256"), str)
            or target != {"path": str(TARGET_PATH), "mode": 0,
                          "list_error": "PermissionError"}):
        raise RuntimeError("image-native validation semantic drift")
    output = coordinator.get("output", {})
    schedule = output.get("schedule")
    isolation = payload.get("isolation", {})
    if (output.get("verified") is not True
            or output.get("source_only") is not True
            or output.get("outer_target_images_used") is not False
            or output.get("outer_target_labels_used") is not False
            or output.get("outer_target_access_authorized") is not False
            or not isinstance(schedule, list) or len(schedule) != 18
            or {item.get("physical_gpu") for item in schedule} != {2, 3}
            or output.get("implementation_erratum", {}).get("registered") is not False
            or "ro" not in isolation.get("project_mount", {}).get("mount_options", [])
            or isolation.get("target_mount", {}).get("filesystem_type") != "tmpfs"
            or isolation.get("tmp_mount", {}).get("filesystem_type") != "tmpfs"
            or isolation.get("network_interfaces") != ["lo"]
            or isolation.get("nvidia_devices") != []
            or isolation.get("torch_cuda_available") is not False
            or isolation.get("torch_cuda_device_count") != 0
            or isolation.get("CUDA_VISIBLE_DEVICES") != ""
            or isolation.get("NVIDIA_VISIBLE_DEVICES") != "void"):
        raise RuntimeError("image-native coordinator/isolation evidence drift")
    return payload


def _run_ephemeral_image_validation(
    intent: Mapping[str, Any], intent_sha: str,
) -> dict[str, Any]:
    entries_before = _audit_entries()
    if entries_before != BASE_AUDIT_ENTRIES:
        raise RuntimeError("audit surface is not the launch-intent base pair")
    formal_before = _inspect_container(intent, intent_sha)
    intent_code = intent.get("code_sha256", {})
    if not isinstance(intent_code, Mapping):
        raise RuntimeError("launch-intent code closure is absent")
    code_before = {
        str(relative): _sha256(PROJECT_ROOT / str(relative))
        for relative in intent_code
    }
    code_before[REGISTRAR_RELATIVE] = _sha256(REGISTRAR_PATH)
    absent_before = _absent_container_evidence(TRANSIENT_VALIDATION_CONTAINER)
    command_record = _command_evidence(_image_native_command(_image_native_script()))
    parsed = _verify_image_native_record(command_record)
    absent_after = _absent_container_evidence(TRANSIENT_VALIDATION_CONTAINER)
    formal_after = _inspect_container(intent, intent_sha)
    code_after = {
        relative: _sha256(PROJECT_ROOT / relative)
        for relative in code_before
    }
    entries_after = _audit_entries()
    if (formal_before != formal_after or code_before != code_after
            or entries_after != BASE_AUDIT_ENTRIES):
        raise RuntimeError("ephemeral validation changed formal container/audit state")
    return {
        "execution_surface": "fixed_image_ephemeral_validation_container",
        "image_id": EXPECTED_IMAGE_ID,
        "python_version": "3.11.12",
        "no_gpu_device_request": True,
        "project_bind_readonly": True,
        "network": "none",
        "read_only_rootfs": True,
        "command_evidence": command_record,
        "parsed_output": parsed,
        "transient_container": {
            "name": TRANSIENT_VALIDATION_CONTAINER,
            "ephemeral": True, "auto_removed": True,
            "absent_before": absent_before, "absent_after": absent_after,
        },
        "formal_existing_container": {
            "before": formal_before, "after": formal_after, "unchanged": True,
        },
        "audit_surface": {
            "before": sorted(entries_before), "after": sorted(entries_after),
            "unchanged": True,
        },
        "project_code_surface": {
            "before": code_before, "after": code_after, "unchanged": True,
        },
    }


def _validation_command_evidence() -> dict[str, Any]:
    if not VALIDATION_PYTHON.is_file() or not os.access(VALIDATION_PYTHON, os.X_OK):
        raise RuntimeError(f"validation interpreter is absent or not executable: {VALIDATION_PYTHON}")
    resolved_python = VALIDATION_PYTHON.resolve(strict=True)
    if not resolved_python.is_file() or not stat.S_ISREG(resolved_python.stat().st_mode):
        raise RuntimeError("validation interpreter does not resolve to a regular file")
    test_bindings: dict[str, dict[str, str]] = {}
    for relative in VALIDATION_TEST_RELATIVES:
        path = PROJECT_ROOT / relative
        _assert_regular(path)
        test_bindings[relative] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
        }
    with tempfile.TemporaryDirectory(prefix="tier2r-startup-fix1-validation-") as temporary:
        environment = {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HOME": temporary,
            "TMPDIR": temporary,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONPATH": str(PROJECT_ROOT),
            "PYTHONPYCACHEPREFIX": str(Path(temporary) / "pycache"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }
        # Preserve the virtual-environment argv so its site-packages remain
        # active; bind the resolved binary identity separately below.
        validation_argv0 = str(VALIDATION_PYTHON)
        compile_command = [
            validation_argv0,
            "-m",
            "py_compile",
            str(REGISTRAR_PATH),
            str(COORDINATOR_PATH),
        ]
        test_command = [
            validation_argv0,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "--tb=short",
            "-q",
            *VALIDATION_TEST_RELATIVES,
        ]
        collect_command = [
            validation_argv0,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "--collect-only",
            "-q",
            *VALIDATION_TEST_RELATIVES,
        ]
        version_command = [validation_argv0, "--version"]
        compile_evidence = _command_evidence(
            compile_command, environment=environment
        )
        collect_evidence = _command_evidence(
            collect_command, environment=environment
        )
        test_evidence = _command_evidence(test_command, environment=environment)
        version_evidence = _command_evidence(
            version_command, environment=environment
        )
    collect_lines = str(collect_evidence.get("stdout", "")).splitlines()
    collected_nodeids = [
        line for line in collect_lines[:-1] if line.strip()
    ]
    collected_summary = collect_lines[-1] if collect_lines else ""
    if (
        resolved_python != EXPECTED_HOST_VALIDATION_RESOLVED_PYTHON
        or _sha256(resolved_python) != EXPECTED_HOST_VALIDATION_PYTHON_SHA256
        or compile_evidence.get("command") != compile_command
        or version_evidence.get("command") != version_command
        or collect_evidence.get("command") != collect_command
        or collect_evidence.get("stderr") != ""
        or len(collected_nodeids) != EXPECTED_VALIDATION_TEST_COUNT
        or any("::test_" not in nodeid for nodeid in collected_nodeids)
        or re.fullmatch(
            rf"{EXPECTED_VALIDATION_TEST_COUNT} tests collected in "
            r"[0-9]+(?:\.[0-9]+)?s",
            collected_summary,
        )
        is None
        or test_evidence.get("command") != test_command
        or test_evidence.get("stderr") != ""
        or re.fullmatch(
            EXPECTED_PYTEST_STDOUT_RE,
            str(test_evidence.get("stdout", "")),
        )
        is None
        or (version_evidence.get("stdout") or version_evidence.get("stderr")).strip()
        != EXPECTED_HOST_VALIDATION_PYTHON_VERSION
    ):
        raise RuntimeError("sanitized host pytest execution/count contract drift")
    for relative, binding in test_bindings.items():
        if _sha256(PROJECT_ROOT / relative) != binding["sha256"]:
            raise RuntimeError(f"validation test changed while running: {relative}")
    return {
        "surface": "sanitized_host_unit_tests_only",
        "expected_test_count": EXPECTED_VALIDATION_TEST_COUNT,
        "collected_test_count": len(collected_nodeids),
        "result_regex": EXPECTED_PYTEST_STDOUT_RE,
        "result_regex_matched": True,
        "environment_contract": {
            "PYTEST_ADDOPTS": "absent",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "pytest_cacheprovider": "disabled",
            "plugin_autoload": "disabled",
            "temporary_home": True,
            "temporary_pycache": True,
        },
        "interpreter": {
            "path": str(VALIDATION_PYTHON),
            "resolved_path": str(resolved_python),
            "resolved_binary_sha256": _sha256(resolved_python),
            "scope": "sanitized_host_unit_tests_only",
            "version": (
                version_evidence["stdout"] or version_evidence["stderr"]
            ).strip(),
            "version_command": version_evidence,
        },
        "test_files": test_bindings,
        "py_compile": compile_evidence,
        "pytest_collect": collect_evidence,
        "pytest": test_evidence,
    }


def _implementation_erratum_planning_evidence(
    intent: Mapping[str, Any]
) -> dict[str, Any]:
    script = """
import hashlib, json
from scripts import tier2r_impl_erratum1 as helper
payload = helper._build_payload(create_snapshots=False)
stamp = payload['registered_at']
helper._now = lambda: stamp
observed = helper.ensure_implementation_erratum(create=False)
normalized = dict(payload)
normalized.pop('registered_at', None)
print(json.dumps({
    'observed': observed,
    'observed_registered_at': stamp,
    'normalized_payload_sha256': hashlib.sha256(
        helper._canonical_json_bytes(normalized)
    ).hexdigest(),
    'recovery_code_bindings': payload['recovery_code_bindings'],
}, sort_keys=True))
""".strip()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    command = [str(VALIDATION_PYTHON), "-c", script]
    evidence = _command_evidence(command, environment=environment)
    payload = json.loads(evidence["stdout"])
    observed = payload.get("observed", {})
    bindings = payload.get("recovery_code_bindings", {})
    old_planned = (
        intent.get("coordinator_verify_only", {})
        .get("implementation_erratum", {})
        .get("planned_sha256")
    )
    if (
        old_planned != EXPECTED_DYNAMIC_OLD_PLANNED_SHA256
        or observed.get("registered") is not False
        or observed.get("planned_sha256") is None
        or bindings.get(COORDINATOR_RELATIVE) != EXPECTED_NEW_COORDINATOR_SHA256
    ):
        raise RuntimeError("implementation-erratum planning evidence drift")
    return {
        "launch_intent_timestamp_bearing_planned_sha256": old_planned,
        "current_helper_timestamp_bearing_planned_sha256": observed["planned_sha256"],
        "current_helper_observed_registered_at": payload["observed_registered_at"],
        "current_helper_create": False,
        "deterministic_payload_without_registered_at_sha256": payload[
            "normalized_payload_sha256"
        ],
        "recovery_code_bindings": bindings,
        "code_binding_delta_only": [COORDINATOR_RELATIVE],
        "dynamic_planned_hashes_are_not_claimed_code_only": True,
        "supersession_reason": (
            "timestamp-bearing launch observation superseded by a current "
            "read-only helper observation plus deterministic normalized payload"
        ),
        "command_evidence": evidence,
    }


def _planning_from_image_validation(
    intent: Mapping[str, Any], image_validation: Mapping[str, Any]
) -> dict[str, Any]:
    parsed = image_validation.get("parsed_output")
    if not isinstance(parsed, Mapping):
        raise RuntimeError("image-native validation payload is absent")
    helper = parsed.get("helper_normalized_planning")
    if not isinstance(helper, Mapping):
        raise RuntimeError("image-native helper planning evidence is absent")
    observed = helper.get("observed")
    bindings = helper.get("recovery_code_bindings")
    old_planned = (
        intent.get("coordinator_verify_only", {})
        .get("implementation_erratum", {})
        .get("planned_sha256")
    )
    if (
        old_planned != EXPECTED_DYNAMIC_OLD_PLANNED_SHA256
        or not isinstance(observed, Mapping)
        or observed.get("registered") is not False
        or not isinstance(observed.get("planned_sha256"), str)
        or not isinstance(bindings, Mapping)
        or bindings.get(COORDINATOR_RELATIVE)
        != EXPECTED_NEW_COORDINATOR_SHA256
        or not isinstance(helper.get("normalized_payload_sha256"), str)
    ):
        raise RuntimeError("image-native implementation planning drift")
    command_evidence = image_validation.get("command_evidence")
    if not isinstance(command_evidence, Mapping):
        raise RuntimeError("image-native command evidence is absent")
    command = command_evidence.get("command")
    stdout_sha256 = command_evidence.get("stdout_sha256")
    if not isinstance(command, list) or not isinstance(stdout_sha256, str):
        raise RuntimeError("image-native command/stdout binding drift")
    command_bytes = (
        json.dumps(
            command,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return {
        "execution_surface": "fixed_image_python_3.11.12",
        "launch_intent_timestamp_bearing_planned_sha256": old_planned,
        "current_helper_timestamp_bearing_planned_sha256": observed[
            "planned_sha256"
        ],
        "current_helper_observed_registered_at": helper.get(
            "observed_registered_at"
        ),
        "current_helper_create": False,
        "deterministic_payload_without_registered_at_sha256": helper[
            "normalized_payload_sha256"
        ],
        "recovery_code_bindings": dict(bindings),
        "code_binding_delta_only": [COORDINATOR_RELATIVE],
        "dynamic_planned_hashes_are_not_claimed_code_only": True,
        "supersession_reason": (
            "timestamp-bearing launch observation superseded by exact-image "
            "read-only helper observation plus deterministic normalized payload"
        ),
        "image_validation_command_canonical_sha256": hashlib.sha256(
            command_bytes
        ).hexdigest(),
        "image_validation_stdout_sha256": stdout_sha256,
    }


def _common_fields() -> dict[str, Any]:
    return {
        "scientific_protocol_id": SCIENTIFIC_PROTOCOL_ID,
        "scientific_protocol_changed": False,
        "execution_instance": EXECUTION_INSTANCE,
        "source_only": True,
        "outer_target_access_authorized": False,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
    }


def _startup_payload(*, execute_image_validation: bool = False) -> dict[str, Any]:
    intent, intent_sha = _verify_intent()
    surface = _assert_registration_surface()
    if execute_image_validation:
        image_validation = _run_ephemeral_image_validation(intent, intent_sha)
        failed_container = image_validation["formal_existing_container"]["before"]
        planning = _planning_from_image_validation(intent, image_validation)
    else:
        failed_container = _inspect_container(intent, intent_sha)
        image_validation = {
            "execution_surface": "fixed_image_ephemeral_validation_container",
            "executed": False,
            "planned_exact_command": _image_native_command(_image_native_script()),
            "default_preflight_is_formal_state_non_mutating": True,
        }
        planning = _implementation_erratum_planning_evidence(intent)
        planning["execution_surface"] = "host_preflight_non_authorizing"
    return {
        "schema_version": STARTUP_SCHEMA,
        "registered_at": _registered_at(STARTUP_FIX1_PATH),
        **_common_fields(),
        "decision": (
            "REGISTER_STARTUP_FIX1_AFTER_EPHEMERAL_FIXED_IMAGE_VALIDATION_"
            "WITHOUT_FORMAL_CONTAINER_MUTATION"
        ),
        "failure_stage": "before_implementation_erratum_registration",
        "failure_cause": "historical_lock_opened_a+b_through_readonly_project_bind",
        "launch_intent": {
            "path": str(INTENT_PATH.resolve()),
            "sha256": intent_sha,
        },
        "failed_container": failed_container,
        "pre_registration_surface": surface,
        "code_drift": _code_drift(intent),
        "readonly_lock_fix": _readonly_lock_contract(),
        "implementation_erratum_planning": planning,
        "image_native_validation": image_validation,
        "formal_container_action_performed_by_registrar": False,
        "ephemeral_validation_container_action_performed": (
            execute_image_validation
        ),
        "ephemeral_validation_container_auto_removed": (
            execute_image_validation
        ),
        "container_start_authorized_by_registration_alone": False,
        "container_recreation_authorized": False,
    }


def _validation_payload(startup_binding: Mapping[str, str]) -> dict[str, Any]:
    intent, intent_sha = _verify_intent()
    drift = _code_drift(intent)
    readonly = _readonly_lock_contract()
    _assert_regular(REGISTRAR_PATH)
    startup = _load_object(STARTUP_FIX1_PATH)
    authorization = _expected_resume_authorization()
    return {
        "schema_version": VALIDATION_SCHEMA,
        "registered_at": _registered_at(VALIDATION_PATH),
        **_common_fields(),
        "validation_result": "STARTUP_FIX1_VALID",
        "startup_fix1": dict(startup_binding),
        "launch_intent": {
            "path": str(INTENT_PATH.resolve()),
            "sha256": intent_sha,
        },
        "registrar": {
            "path": str(REGISTRAR_PATH.resolve()),
            "sha256": _sha256(REGISTRAR_PATH),
        },
        "coordinator": {
            "path": str(COORDINATOR_PATH.resolve()),
            "sha256": drift["new_sha256"],
        },
        "code_drift": drift,
        "readonly_lock_fix": readonly,
        "implementation_erratum_planning": startup.get(
            "implementation_erratum_planning"
        ),
        "validation_commands": _validation_command_evidence(),
        "image_native_validation": startup.get("image_native_validation"),
        "resume_authorization": authorization,
        "checks": {
            "launch_intent_immutable": True,
            "failed_container_snapshot_bound": True,
            "four_restart_failures_bound": True,
            "no_formal_artifacts_preceded_registration": True,
            "only_coordinator_drifted_from_launch_intent": True,
            "readonly_existing_lock_no_creation": True,
            "source_only_target_lock_preserved": True,
            "fixed_image_python31112_validation_passed": True,
            "formal_existing_container_unchanged_by_ephemeral_validation": True,
        },
        "formal_container_action_performed_by_registrar": False,
        "ephemeral_validation_container_action_performed": True,
        "ephemeral_validation_container_auto_removed": True,
        "container_recreation_authorized": False,
    }


def _expected_pre_registration_surface() -> dict[str, Any]:
    return {
        "audit_entries_before_startup_fix1": sorted(BASE_AUDIT_ENTRIES),
        "implementation_erratum_registered": False,
        "preregistration_registered": False,
        "status_registered": False,
        "handoff_registered": False,
        "formal_score_exports_registered": False,
    }


def _verify_startup_payload_semantics(payload: Mapping[str, Any]) -> None:
    _verify_common_semantics(payload, STARTUP_SCHEMA)
    intent, intent_sha = _verify_intent()
    failed = payload.get("failed_container")
    image_record = payload.get("image_native_validation")
    if not isinstance(failed, Mapping) or not isinstance(image_record, Mapping):
        raise RuntimeError("startup failure/image evidence is absent")
    parsed = _verify_image_native_record(
        image_record.get("command_evidence", {})
    )
    formal = image_record.get("formal_existing_container")
    transient = image_record.get("transient_container")
    audit = image_record.get("audit_surface")
    code = image_record.get("project_code_surface")
    if not all(
        isinstance(value, Mapping)
        for value in (formal, transient, audit, code)
    ):
        raise RuntimeError("startup image lifecycle evidence is absent")
    _verify_absent_record(transient.get("absent_before", {}))
    _verify_absent_record(transient.get("absent_after", {}))
    intent_code = intent.get("code_sha256")
    if not isinstance(intent_code, Mapping):
        raise RuntimeError("launch-intent code closure is absent")
    expected_code = {
        str(relative): _sha256(PROJECT_ROOT / str(relative))
        for relative in intent_code
    }
    expected_code[REGISTRAR_RELATIVE] = _sha256(REGISTRAR_PATH)
    expected_audit = {
        "before": sorted(BASE_AUDIT_ENTRIES),
        "after": sorted(BASE_AUDIT_ENTRIES),
        "unchanged": True,
    }
    expected_planning = _planning_from_image_validation(intent, image_record)
    expected_decision = (
        "REGISTER_STARTUP_FIX1_AFTER_EPHEMERAL_FIXED_IMAGE_VALIDATION_"
        "WITHOUT_FORMAL_CONTAINER_MUTATION"
    )
    if (
        payload.get("decision") != expected_decision
        or payload.get("failure_stage")
        != "before_implementation_erratum_registration"
        or payload.get("failure_cause")
        != "historical_lock_opened_a+b_through_readonly_project_bind"
        or payload.get("launch_intent")
        != {"path": str(INTENT_PATH.resolve()), "sha256": intent_sha}
        or payload.get("pre_registration_surface")
        != _expected_pre_registration_surface()
        or payload.get("code_drift") != _code_drift(intent)
        or payload.get("readonly_lock_fix") != _readonly_lock_contract()
        or payload.get("implementation_erratum_planning")
        != expected_planning
        or image_record.get("parsed_output") != parsed
        or image_record.get("execution_surface")
        != "fixed_image_ephemeral_validation_container"
        or image_record.get("image_id") != EXPECTED_IMAGE_ID
        or image_record.get("python_version") != "3.11.12"
        or image_record.get("no_gpu_device_request") is not True
        or image_record.get("project_bind_readonly") is not True
        or image_record.get("network") != "none"
        or image_record.get("read_only_rootfs") is not True
        or formal.get("before") != failed
        or formal.get("after") != failed
        or formal.get("unchanged") is not True
        or transient.get("name") != TRANSIENT_VALIDATION_CONTAINER
        or transient.get("ephemeral") is not True
        or transient.get("auto_removed") is not True
        or audit != expected_audit
        or code.get("before") != expected_code
        or code.get("after") != expected_code
        or code.get("unchanged") is not True
        or failed.get("id") != EXPECTED_CONTAINER_ID
        or failed.get("name") != CONTAINER_NAME
        or failed.get("image_id") != EXPECTED_IMAGE_ID
        or failed.get("status") != "exited"
        or failed.get("exit_code") != 1
        or failed.get("oom_killed") is not False
        or failed.get("restart_count") != 3
        or failed.get("failure_log_num_lines") != 4
        or failed.get("failure_log_line") != EXPECTED_FAILURE_LINE
        or failed.get("configuration_contract", {}).get(
            "matches_launch_intent"
        )
        is not True
        or payload.get("formal_container_action_performed_by_registrar")
        is not False
        or payload.get("ephemeral_validation_container_action_performed")
        is not True
        or payload.get("ephemeral_validation_container_auto_removed")
        is not True
        or payload.get("container_start_authorized_by_registration_alone")
        is not False
        or payload.get("container_recreation_authorized") is not False
    ):
        raise RuntimeError("startup-fix1 complete semantic drift")


def _verify_host_validation_evidence(
    commands: Mapping[str, Any],
    *,
    verify_live_host_evidence: bool = True,
) -> None:
    expected_resolved = EXPECTED_HOST_VALIDATION_RESOLVED_PYTHON
    expected_argv0 = str(VALIDATION_PYTHON)
    expected_compile = [
        expected_argv0,
        "-m",
        "py_compile",
        str(REGISTRAR_PATH),
        str(COORDINATOR_PATH),
    ]
    expected_collect = [
        expected_argv0,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "--collect-only",
        "-q",
        *VALIDATION_TEST_RELATIVES,
    ]
    expected_pytest = [
        expected_argv0,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "--tb=short",
        "-q",
        *VALIDATION_TEST_RELATIVES,
    ]
    expected_version = [expected_argv0, "--version"]
    interpreter = commands.get("interpreter", {})
    tests = commands.get("test_files", {})
    compile_record = commands.get("py_compile", {})
    collect_record = commands.get("pytest_collect", {})
    pytest_record = commands.get("pytest", {})
    version_record = interpreter.get("version_command", {})
    for record in (
        compile_record,
        collect_record,
        pytest_record,
        version_record,
    ):
        _verify_stream_record(record)
        if record.get("returncode") != 0:
            raise RuntimeError("host validation command failed")
    collect_lines = str(collect_record.get("stdout", "")).splitlines()
    nodeids = [line for line in collect_lines[:-1] if line.strip()]
    summary = collect_lines[-1] if collect_lines else ""
    expected_environment = {
        "PYTEST_ADDOPTS": "absent",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "pytest_cacheprovider": "disabled",
        "plugin_autoload": "disabled",
        "temporary_home": True,
        "temporary_pycache": True,
    }
    if (
        commands.get("surface") != "sanitized_host_unit_tests_only"
        or commands.get("expected_test_count")
        != EXPECTED_VALIDATION_TEST_COUNT
        or commands.get("collected_test_count")
        != EXPECTED_VALIDATION_TEST_COUNT
        or commands.get("result_regex") != EXPECTED_PYTEST_STDOUT_RE
        or commands.get("result_regex_matched") is not True
        or commands.get("environment_contract") != expected_environment
        or interpreter.get("path") != str(VALIDATION_PYTHON)
        or interpreter.get("resolved_path") != str(expected_resolved)
        or interpreter.get("resolved_binary_sha256")
        != EXPECTED_HOST_VALIDATION_PYTHON_SHA256
        or interpreter.get("scope") != "sanitized_host_unit_tests_only"
        or interpreter.get("version")
        != EXPECTED_HOST_VALIDATION_PYTHON_VERSION
        or compile_record.get("command") != expected_compile
        or collect_record.get("command") != expected_collect
        or pytest_record.get("command") != expected_pytest
        or version_record.get("command") != expected_version
        or (version_record.get("stdout") or version_record.get("stderr")).strip()
        != EXPECTED_HOST_VALIDATION_PYTHON_VERSION
        or collect_record.get("stderr") != ""
        or len(nodeids) != EXPECTED_VALIDATION_TEST_COUNT
        or any("::test_" not in nodeid for nodeid in nodeids)
        or re.fullmatch(
            rf"{EXPECTED_VALIDATION_TEST_COUNT} tests collected in "
            r"[0-9]+(?:\.[0-9]+)?s",
            summary,
        )
        is None
        or pytest_record.get("stderr") != ""
        or re.fullmatch(
            EXPECTED_PYTEST_STDOUT_RE,
            str(pytest_record.get("stdout", "")),
        )
        is None
        or not isinstance(tests, Mapping)
        or set(tests) != set(VALIDATION_TEST_RELATIVES)
    ):
        raise RuntimeError("sanitized host unit-test evidence drift")
    for relative in VALIDATION_TEST_RELATIVES:
        path = PROJECT_ROOT / relative
        if tests.get(relative) != {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
        }:
            raise RuntimeError(f"sanitized host test binding drift: {relative}")
    if verify_live_host_evidence:
        if (
            not VALIDATION_PYTHON.is_file()
            or not os.access(VALIDATION_PYTHON, os.X_OK)
        ):
            raise RuntimeError("live host validation interpreter is unavailable")
        resolved = VALIDATION_PYTHON.resolve(strict=True)
        if (
            resolved != expected_resolved
            or not resolved.is_file()
            or not stat.S_ISREG(resolved.stat().st_mode)
            or _sha256(resolved) != EXPECTED_HOST_VALIDATION_PYTHON_SHA256
        ):
            raise RuntimeError("live host validation interpreter identity drift")


def _verify_validation_payload_semantics(
    payload: Mapping[str, Any],
    *,
    verify_live_host_evidence: bool = True,
) -> None:
    _verify_common_semantics(payload, VALIDATION_SCHEMA)
    _, intent_sha = _verify_intent()
    startup_binding = _binding(STARTUP_FIX1_PATH, STARTUP_FIX1_SIDECAR)
    startup = _load_object(STARTUP_FIX1_PATH)
    _verify_startup_payload_semantics(startup)
    _verify_host_validation_evidence(
        payload.get("validation_commands", {}),
        verify_live_host_evidence=verify_live_host_evidence,
    )
    expected_checks = {
        "launch_intent_immutable": True,
        "failed_container_snapshot_bound": True,
        "four_restart_failures_bound": True,
        "no_formal_artifacts_preceded_registration": True,
        "only_coordinator_drifted_from_launch_intent": True,
        "readonly_existing_lock_no_creation": True,
        "source_only_target_lock_preserved": True,
        "fixed_image_python31112_validation_passed": True,
        "formal_existing_container_unchanged_by_ephemeral_validation": True,
    }
    if (
        payload.get("validation_result") != "STARTUP_FIX1_VALID"
        or payload.get("startup_fix1") != startup_binding
        or payload.get("launch_intent")
        != {"path": str(INTENT_PATH.resolve()), "sha256": intent_sha}
        or payload.get("registrar")
        != {
            "path": str(REGISTRAR_PATH.resolve()),
            "sha256": _sha256(REGISTRAR_PATH),
        }
        or payload.get("coordinator")
        != {
            "path": str(COORDINATOR_PATH.resolve()),
            "sha256": EXPECTED_NEW_COORDINATOR_SHA256,
        }
        or payload.get("code_drift") != startup.get("code_drift")
        or payload.get("readonly_lock_fix")
        != startup.get("readonly_lock_fix")
        or payload.get("implementation_erratum_planning")
        != startup.get("implementation_erratum_planning")
        or payload.get("image_native_validation")
        != startup.get("image_native_validation")
        or payload.get("resume_authorization")
        != _expected_resume_authorization()
        or payload.get("checks") != expected_checks
        or payload.get("formal_container_action_performed_by_registrar")
        is not False
        or payload.get("ephemeral_validation_container_action_performed")
        is not True
        or payload.get("ephemeral_validation_container_auto_removed")
        is not True
        or payload.get("container_recreation_authorized") is not False
    ):
        raise RuntimeError("startup-fix1 validation complete semantic drift")


def verify_frozen_startup_fix1(
    *, verify_live_host_evidence: bool = False
) -> dict[str, dict[str, str]]:
    """Verify frozen startup evidence without Docker access or state mutation."""

    _verify_intent()
    startup_sha = _verify_sidecar(STARTUP_FIX1_PATH, STARTUP_FIX1_SIDECAR)
    validation_sha = _verify_sidecar(VALIDATION_PATH, VALIDATION_SIDECAR)
    startup = _load_object(STARTUP_FIX1_PATH)
    validation = _load_object(VALIDATION_PATH)
    startup_binding = {
        "path": str(STARTUP_FIX1_PATH.resolve()),
        "sha256": startup_sha,
    }
    _verify_startup_payload_semantics(startup)
    _verify_validation_payload_semantics(
        validation,
        verify_live_host_evidence=verify_live_host_evidence,
    )
    return {
        "startup_fix1": startup_binding,
        "startup_fix1_validation": {
            "path": str(VALIDATION_PATH.resolve()),
            "sha256": validation_sha,
        },
    }


def preflight() -> dict[str, Any]:
    entries = _audit_entries()
    if (
        STARTUP_FIX1_PATH.is_file()
        and STARTUP_FIX1_SIDECAR.is_file()
        and VALIDATION_PATH.is_file()
        and VALIDATION_SIDECAR.is_file()
    ):
        return {
            "schema_version": STARTUP_SCHEMA,
            "verified": True,
            "registered": True,
            "formal_container_action_performed": False,
            "ephemeral_validation_container_action_performed": True,
            "ephemeral_validation_container_auto_removed": True,
            **verify_frozen_startup_fix1(
                verify_live_host_evidence=True
            ),
        }
    if entries != BASE_AUDIT_ENTRIES:
        # Read-only preflight never repairs a torn path/sidecar publication.
        # It reports the closed recovery state; only explicit --register may
        # invoke the semantic validators and publish a missing sidecar.
        _assert_registration_surface()
        return {
            "schema_version": STARTUP_SCHEMA,
            "verified": False,
            "registered": False,
            "recovery_required": True,
            "recovery_action": "RUN_EXPLICIT_REGISTER",
            "audit_entries": sorted(entries),
            "resume_authorized": False,
            "formal_container_action_performed": False,
            "ephemeral_validation_container_action_performed": False,
            "ephemeral_validation_container_auto_removed": False,
        }
    payload = _startup_payload()
    return {
        "schema_version": STARTUP_SCHEMA,
        "verified": True,
        "registered": False,
        "formal_container_action_performed": False,
        "ephemeral_validation_container_action_performed": False,
        "ephemeral_validation_container_auto_removed": False,
        "candidate_startup_fix1": payload,
    }


def register() -> dict[str, Any]:
    """Freeze evidence after one ephemeral check; never mutate the formal container."""

    with _registration_lock():
        _cleanup_pending_publications()
        _recover_missing_sidecar(
            STARTUP_FIX1_PATH,
            STARTUP_FIX1_SIDECAR,
            _verify_startup_payload_semantics,
        )
        _recover_missing_sidecar(
            VALIDATION_PATH,
            VALIDATION_SIDECAR,
            lambda payload: _verify_validation_payload_semantics(
                payload,
                verify_live_host_evidence=True,
            ),
        )
        if VALIDATION_PATH.exists() or VALIDATION_SIDECAR.exists():
            bindings = verify_frozen_startup_fix1(
                verify_live_host_evidence=True
            )
            return {
                "schema_version": STARTUP_SCHEMA,
                "registered": True,
                "formal_container_action_performed": False,
                "ephemeral_validation_container_action_performed": True,
                "ephemeral_validation_container_auto_removed": True,
                "authorized_action": "START_EXISTING_CONTAINER_ONLY",
                **bindings,
            }
        if STARTUP_FIX1_PATH.exists() or STARTUP_FIX1_SIDECAR.exists():
            startup_sha = _verify_sidecar(
                STARTUP_FIX1_PATH, STARTUP_FIX1_SIDECAR
            )
            _verify_startup_payload_semantics(
                _load_object(STARTUP_FIX1_PATH)
            )
        else:
            startup = _startup_payload(execute_image_validation=True)
            _verify_startup_payload_semantics(startup)
            startup_sha = _write_once(
                STARTUP_FIX1_PATH,
                STARTUP_FIX1_SIDECAR,
                startup,
                lock_held=True,
            )
        startup_binding = {
            "path": str(STARTUP_FIX1_PATH.resolve()),
            "sha256": startup_sha,
        }
        validation = _validation_payload(startup_binding)
        _verify_validation_payload_semantics(
            validation,
            verify_live_host_evidence=True,
        )
        validation_sha = _write_once(
            VALIDATION_PATH,
            VALIDATION_SIDECAR,
            validation,
            lock_held=True,
        )
        bindings = verify_frozen_startup_fix1(
            verify_live_host_evidence=True
        )
        return {
            "schema_version": STARTUP_SCHEMA,
            "registered": True,
            "formal_container_action_performed": False,
            "ephemeral_validation_container_action_performed": True,
            "ephemeral_validation_container_auto_removed": True,
            "authorized_action": "START_EXISTING_CONTAINER_ONLY",
            "startup_fix1_sha256": startup_sha,
            "startup_fix1_validation_sha256": validation_sha,
            **bindings,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--verify-frozen", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.register:
            result = register()
        elif args.verify_frozen:
            result = {
                "schema_version": STARTUP_SCHEMA,
                "verified": True,
                "formal_container_action_performed": False,
                "ephemeral_validation_container_action_performed": True,
                "ephemeral_validation_container_auto_removed": True,
                **verify_frozen_startup_fix1(
                    verify_live_host_evidence=True
                ),
            }
        else:
            result = preflight()
    except BaseException as error:
        print(f"FAILED_CLOSED {type(error).__name__}: {error}", file=os.sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "STARTUP_FIX1_PATH",
    "VALIDATION_PATH",
    "register",
    "verify_frozen_startup_fix1",
]
