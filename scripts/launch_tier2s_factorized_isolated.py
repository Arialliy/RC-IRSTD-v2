#!/usr/bin/env python3
"""Fail-closed isolated launcher for the Tier2S factorized causal audit.

The default invocation is a non-mutating dry run.  ``--verify-only`` executes
the coordinator preflight and the isolation probe in ephemeral containers.
``--execute`` performs, in order, coordinator verification, isolated
registration, immutable host launch-intent registration, and an idempotent
start/reconcile of the named formal container.

This is a new exploratory, source-only namespace.  It neither modifies nor
continues the frozen Tier2R execution, and it can never authorize Tier3,
paper claims, or NUAA access.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import functools
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterator, Mapping, Sequence


PROJECT_ROOT = Path("/home/ly/RC-IRSTD-v2")
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from scripts import launch_phase3_tier2r_isolated as base  # noqa: E402


PROTOCOL_ID = "tier2s_factorized_causal_audit_v1"
RESEARCH_MODE = "exploratory_source_only"
COORDINATOR_SCHEMA = "rc-irstd-aaai27-tier2s-factorized-coordinator-v1"
PROBE_SCHEMA = "rc-irstd-aaai27-tier2s-factorized-container-attestation-v1"
COORDINATOR = PROJECT_ROOT / "scripts/coordinate_tier2s_factorized_audit.py"
CONTAINER_PROBE = PROJECT_ROOT / "scripts/probe_tier2s_factorized_container.py"
EXPORTER = PROJECT_ROOT / "scripts/export_tier2s_factorized_logits.py"
EVALUATOR = PROJECT_ROOT / "scripts/evaluate_tier2s_factorized_audit.py"
PROTOCOL = PROJECT_ROOT / "configs/tier2s_factorized_causal_audit_v1.json"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/aaai27/source_rescue/tier2s_factorized_causal_audit_v1"
)
AUDIT_ROOT = (
    PROJECT_ROOT
    / "artifacts/aaai27/audit/source_rescue/tier2s_factorized_causal_audit_v1"
)
FORBIDDEN_TARGET = PROJECT_ROOT / "datasets/NUAA-SIRST"
INTENT_PATH = AUDIT_ROOT / "LAUNCH_INTENT.json"
INTENT_SHA256_PATH = AUDIT_ROOT / "LAUNCH_INTENT.json.sha256"
PREREGISTRATION_PATH = AUDIT_ROOT / "PREREGISTRATION.json"

GOVERNANCE_REGISTRATION_RELATIVE = Path(
    "artifacts/aaai27/audit/governance/aaai27_model_success_contract_v1/"
    "GOVERNANCE_REGISTRATION.json"
)
GOVERNANCE_CONTRACT_RELATIVE = Path(
    "configs/aaai27_model_success_contract_v1.json"
)
FRESH_SEED_LEDGER_RELATIVE = Path(
    "artifacts/aaai27/audit/governance/aaai27_model_success_contract_v1/"
    "FRESH_SEED_LEDGER.json"
)
FRESH_SEED_LOCAL_SCAN_RELATIVE = Path(
    "artifacts/aaai27/audit/governance/aaai27_model_success_contract_v1/"
    "FRESH_SEED_LOCAL_SCAN.json"
)

IMAGE_ID = base.IMAGE_ID
CONTAINER_PYTHON = base.CONTAINER_PYTHON
CONTAINER_USER = base.CONTAINER_USER
TARGET_GPU_INDICES = (0, 1)
TARGET_GPU_DEVICE_IDS = ("0", "1")
# Docker's multi-device --gpus grammar requires the inner JSON-style quotes
# to reach the CLI as part of the single argv value.
DOCKER_GPU_REQUEST = '"device=0,1"'
NVIDIA_VISIBLE_DEVICES = "0,1"
CONTAINER_NAME = "rc-irstd-tier2s-factorized-causal-audit-v1"
SCHEMA = "rc-irstd-aaai27-tier2s-factorized-isolated-launch-intent-v1"
INTENT_LABEL = base.INTENT_LABEL

ENVIRONMENT = dict(base.ENVIRONMENT)
ENVIRONMENT["NVIDIA_VISIBLE_DEVICES"] = NVIDIA_VISIBLE_DEVICES
RUNTIME_LABELS = {
    "org.rc-irstd.protocol": PROTOCOL_ID,
    "org.rc-irstd.research-mode": RESEARCH_MODE,
    "org.rc-irstd.source-only": "true",
    "org.rc-irstd.outer-target-access": "denied",
    "org.rc-irstd.source-tier3": "denied",
    "org.rc-irstd.paper-claim": "denied",
    "org.rc-irstd.physical-gpus": NVIDIA_VISIBLE_DEVICES,
}

CODE_PATHS = (
    Path(__file__).resolve(),
    COORDINATOR,
    CONTAINER_PROBE,
    EXPORTER,
    EVALUATOR,
    PROJECT_ROOT / "rc_irstd/models/rc_mshnet.py",
    PROJECT_ROOT / "rc_irstd/models/__init__.py",
    PROJECT_ROOT / "evaluation/export_score_maps.py",
    PROJECT_ROOT / "evaluation/raw_logit_source_operating_point.py",
    PROJECT_ROOT / "evaluation/component_matching.py",
)
CONFIG_PATHS = (PROTOCOL,)

LEGACY_CONTAINER_NAME = "rc-irstd-tier2r-component-rescue-v1-impl-erratum1"
LEGACY_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs/aaai27/detectors/component_rescue/tier2r_c_v1"
)
LEGACY_AUDIT_ROOT = (
    PROJECT_ROOT
    / "artifacts/aaai27/audit/component_rescue/tier2r_c_v1_impl_erratum1"
)


_BASE_CONTAINER_SPEC = base.container_spec
_BASE_LAUNCH_INTENT_PAYLOAD = base._launch_intent_payload


def bind_mount_contract() -> tuple[dict[str, Any], ...]:
    """Return the complete bind contract; only new Tier2S roots are writable."""

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


def _assert_new_namespace() -> None:
    writable = {
        Path(str(value["destination"]))
        for value in bind_mount_contract()
        if not value["readonly"]
    }
    if writable != {OUTPUT_ROOT, AUDIT_ROOT}:
        raise RuntimeError("Tier2S writable bind namespace drift")
    if (
        CONTAINER_NAME == LEGACY_CONTAINER_NAME
        or LEGACY_OUTPUT_ROOT in writable
        or LEGACY_AUDIT_ROOT in writable
    ):
        raise RuntimeError("Tier2S attempted to reuse a frozen Tier2R namespace")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_frozen_artifact_binding(
    value: Any,
    *,
    expected_path: Path,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Tier2S {name} binding is absent")
    binding = dict(value)
    expected_sidecar = Path(str(expected_path) + ".sha256")
    if (
        set(binding)
        != {"path", "sha256", "sidecar_path", "sidecar_sha256"}
        or binding.get("path") != expected_path.as_posix()
        or binding.get("sidecar_path") != expected_sidecar.as_posix()
        or not _valid_sha256(binding.get("sha256"))
        or not _valid_sha256(binding.get("sidecar_sha256"))
    ):
        raise RuntimeError(f"Tier2S {name} binding drift")
    return binding


def _require_governance_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = payload.get("governance_binding")
    if not isinstance(value, Mapping):
        raise RuntimeError("Tier2S exact governance binding is absent")
    binding = dict(value)
    expected_keys = {
        "schema_version",
        "registration",
        "contract",
        "fresh_seed_ledger",
        "fresh_seed_local_scan",
        "code_sha256_canonical_sha256",
        "tier2s_source_only_diagnostic_authorized",
        "formal_v3_model_training_authorized",
        "source_gate_a_authorized",
        "riskcurve_authorized",
        "outer_target_access_authorized",
    }
    if (
        set(binding) != expected_keys
        or binding.get("schema_version")
        != "rc-irstd-aaai27-tier2s-governance-binding-v1"
        or not _valid_sha256(binding.get("code_sha256_canonical_sha256"))
        or binding.get("tier2s_source_only_diagnostic_authorized") is not True
        or binding.get("formal_v3_model_training_authorized") is not False
        or binding.get("source_gate_a_authorized") is not False
        or binding.get("riskcurve_authorized") is not False
        or binding.get("outer_target_access_authorized") is not False
    ):
        raise RuntimeError("Tier2S governance authorization binding drift")
    _require_frozen_artifact_binding(
        binding.get("registration"),
        expected_path=GOVERNANCE_REGISTRATION_RELATIVE,
        name="governance registration",
    )
    _require_frozen_artifact_binding(
        binding.get("contract"),
        expected_path=GOVERNANCE_CONTRACT_RELATIVE,
        name="success contract",
    )
    _require_frozen_artifact_binding(
        binding.get("fresh_seed_ledger"),
        expected_path=FRESH_SEED_LEDGER_RELATIVE,
        name="fresh-seed ledger",
    )
    _require_frozen_artifact_binding(
        binding.get("fresh_seed_local_scan"),
        expected_path=FRESH_SEED_LOCAL_SCAN_RELATIVE,
        name="fresh-seed local scan",
    )
    return json.loads(
        json.dumps(binding, sort_keys=True, separators=(",", ":"))
    )


def _require_preregistration_binding(
    payload: Mapping[str, Any],
    *,
    governance_registration_sha256: str,
) -> dict[str, Any]:
    value = payload.get("tier2s_preregistration_binding")
    if not isinstance(value, Mapping):
        raise RuntimeError("Tier2S exact preregistration binding is absent")
    binding = dict(value)
    relative = PREREGISTRATION_PATH.relative_to(PROJECT_ROOT)
    sidecar = Path(str(relative) + ".sha256")
    if (
        set(binding)
        != {
            "schema_version",
            "protocol_id",
            "path",
            "sha256",
            "sidecar_path",
            "sidecar_sha256",
            "governance_registration_sha256",
        }
        or binding.get("schema_version")
        != "rc-irstd-aaai27-tier2s-preregistration-binding-v1"
        or binding.get("protocol_id") != PROTOCOL_ID
        or binding.get("path") != relative.as_posix()
        or binding.get("sidecar_path") != sidecar.as_posix()
        or not _valid_sha256(binding.get("sha256"))
        or not _valid_sha256(binding.get("sidecar_sha256"))
        or binding.get("governance_registration_sha256")
        != governance_registration_sha256
    ):
        raise RuntimeError("Tier2S preregistration binding drift")
    return json.loads(
        json.dumps(binding, sort_keys=True, separators=(",", ":"))
    )


def _coordinator_contract(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    schedule = payload.get("schedule")
    if not isinstance(schedule, list) or not all(
        isinstance(value, Mapping) for value in schedule
    ):
        raise RuntimeError("Tier2S coordinator schedule is absent or malformed")
    lanes = payload.get("lane_lengths")
    expected_lanes = {"0": 9, "1": 9}
    if (
        payload.get("schema_version") != COORDINATOR_SCHEMA
        or payload.get("verified") is not True
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("research_mode") != RESEARCH_MODE
        or payload.get("source_only") is not True
        or payload.get("outer_target_access_authorized") is not False
        or payload.get("outer_target_images_used") is not False
        or payload.get("outer_target_labels_used") is not False
        or payload.get("source_tier3_authorized") is not False
        or payload.get("paper_claim_authorized") is not False
        or len(schedule) != 18
        or {value.get("physical_gpu") for value in schedule}
        != set(TARGET_GPU_INDICES)
        or lanes != expected_lanes
    ):
        raise RuntimeError("Tier2S exploratory/source-only coordinator contract drift")
    _require_governance_binding(payload)
    run_ids = [value.get("run_id") for value in schedule]
    if (
        any(not isinstance(value, str) or not value for value in run_ids)
        or len(set(run_ids)) != 18
        or any(
            not isinstance(value.get("physical_gpu"), int)
            or isinstance(value.get("physical_gpu"), bool)
            for value in schedule
        )
    ):
        raise RuntimeError("Tier2S coordinator run IDs are not exactly unique")
    for gpu in TARGET_GPU_INDICES:
        queue_indices = [
            value.get("queue_index")
            for value in schedule
            if value.get("physical_gpu") == gpu
        ]
        if queue_indices != list(range(9)):
            raise RuntimeError(
                f"Tier2S physical GPU {gpu} FIFO queue must be exactly 0..8"
            )
    return schedule


def _verify_in_container(
    *, runner: base.Runner | None = None
) -> dict[str, Any]:
    completed = base._run(base.build_verify_command(), runner=runner)
    payload = base._load_json_bytes(
        completed.stdout, context="Tier2S verify-only container"
    )
    _coordinator_contract(payload)
    return payload


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
        DOCKER_GPU_REQUEST,
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
            base._bind_mount(
                Path(str(mount["source"])),
                Path(str(mount["destination"])),
                readonly=bool(mount["readonly"]),
            )
        )
    for mount in tmpfs_contract():
        command.extend(
            ["--tmpfs", f"{mount['destination']}:{mount['options']}"]
        )
    return command


def _assert_exact_gpu_command(command: Sequence[str]) -> None:
    positions = [
        index for index, value in enumerate(command) if value == "--gpus"
    ]
    if (
        len(positions) != 1
        or positions[0] + 1 >= len(command)
        or command[positions[0] + 1] != DOCKER_GPU_REQUEST
        or command[positions[0] + 1] == "all"
    ):
        raise RuntimeError(
            "Tier2S Docker GPU request must be exactly device=0,1, never all"
        )
    environment_values = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--env"
    ]
    nvidia_values = [
        value
        for value in environment_values
        if value.startswith("NVIDIA_VISIBLE_DEVICES=")
    ]
    cuda_values = [
        value
        for value in environment_values
        if value.startswith("CUDA_VISIBLE_DEVICES=")
    ]
    if nvidia_values != [f"NVIDIA_VISIBLE_DEVICES={NVIDIA_VISIBLE_DEVICES}"]:
        raise RuntimeError("Tier2S NVIDIA_VISIBLE_DEVICES allowlist drift")
    if cuda_values:
        raise RuntimeError(
            "Tier2S top-level container must not set CUDA_VISIBLE_DEVICES"
        )


def _selected_host_gpus(
    host_inventory: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for raw in host_inventory:
        index = raw.get("index")
        uuid = raw.get("uuid")
        name = raw.get("name")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not isinstance(uuid, str)
            or not uuid
            or not isinstance(name, str)
            or not name
            or index in by_index
        ):
            raise RuntimeError("host GPU inventory is malformed or duplicated")
        by_index[index] = {"index": index, "uuid": uuid, "name": name}
    if any(index not in by_index for index in TARGET_GPU_INDICES):
        raise RuntimeError("required physical GPU 0/1 is absent")
    selected = [by_index[index] for index in TARGET_GPU_INDICES]
    if len({value["uuid"] for value in selected}) != len(selected):
        raise RuntimeError("selected physical GPU UUIDs are not unique")
    return selected


def _verify_container_attestation(
    host_inventory: Sequence[Mapping[str, Any]],
    *,
    runner: base.Runner | None = None,
) -> dict[str, Any]:
    completed = base._run(build_probe_command(), runner=runner)
    payload = base._load_json_bytes(
        completed.stdout, context="Tier2S container probe"
    )
    expected_nvidia = _selected_host_gpus(host_inventory)
    expected_torch = [
        {
            "ordinal": ordinal,
            "uuid": value["uuid"],
            "name": value["name"],
        }
        for ordinal, value in enumerate(expected_nvidia)
    ]
    problems: list[str] = []
    if (
        payload.get("schema_version") != PROBE_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("research_mode") != RESEARCH_MODE
    ):
        problems.append("probe identity")
    if payload.get("home_ly_entries") != [PROJECT_ROOT.name]:
        problems.append("/home/ly project isolation")
    target = payload.get("target")
    if (
        not isinstance(target, Mapping)
        or target.get("path") != str(FORBIDDEN_TARGET)
        or target.get("mode") != 0
        or target.get("list_error") != "PermissionError"
    ):
        problems.append("outer-target tmpfs mode=000 denial")
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
        options = (
            set(entry.get("mount_options", []))
            if isinstance(entry, Mapping)
            else set()
        )
        if not isinstance(entry, Mapping) or (("rw" in options) is not writable):
            problems.append(f"mount mode drift:{path}")
    target_mount = mounts.get(str(FORBIDDEN_TARGET))
    if (
        not isinstance(target_mount, Mapping)
        or target_mount.get("filesystem_type") != "tmpfs"
    ):
        problems.append("outer-target mount is not tmpfs")
    if payload.get("nvidia_inventory") != expected_nvidia:
        problems.append("container NVIDIA inventory must be exact physical 0/1")
    if payload.get("torch_inventory") != expected_torch:
        problems.append("Torch ordinals must map exactly to host physical 0/1")
    environment = payload.get("environment")
    if (
        not isinstance(environment, Mapping)
        or environment.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
        or environment.get("CUDA_VISIBLE_DEVICES") is not None
        or environment.get("NVIDIA_VISIBLE_DEVICES")
        != NVIDIA_VISIBLE_DEVICES
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
        for key, prefix in (
            ("torchvision", "0.22."),
            ("PIL", "11."),
            ("skimage", "0.26."),
        ):
            if not str(runtime.get(key, "")).startswith(prefix):
                problems.append(f"runtime drift:{key}")
    if problems:
        raise RuntimeError(
            "Tier2S container attestation failed: " + "; ".join(problems)
        )
    payload["verified"] = True
    payload["physical_gpu_allowlist_verified"] = list(TARGET_GPU_INDICES)
    payload["host_gpu_uuid_mapping_verified"] = True
    payload["torch_ordinal_mapping_verified"] = True
    payload["outer_target_alias_absent"] = True
    return payload


def _verify_selected_gpus_idle(
    host_inventory: Sequence[Mapping[str, Any]],
    *,
    runner: base.Runner | None = None,
) -> dict[str, Any]:
    expected = _selected_host_gpus(host_inventory)
    stats_command = [
        "nvidia-smi",
        "-i",
        NVIDIA_VISIBLE_DEVICES,
        "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = base._run(stats_command, runner=runner)
    stats: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [value.strip() for value in line.split(",", 3)]
        if len(fields) != 4:
            raise RuntimeError(f"malformed selected-GPU idle row: {line!r}")
        try:
            index = int(fields[0])
            memory_used_mib = int(fields[2])
            utilization_percent = int(fields[3])
        except ValueError as error:
            raise RuntimeError(
                f"non-integer selected-GPU idle row: {line!r}"
            ) from error
        stats.append(
            {
                "index": index,
                "uuid": fields[1],
                "memory_used_mib": memory_used_mib,
                "utilization_gpu_percent": utilization_percent,
            }
        )
    observed_identity = [
        {"index": value["index"], "uuid": value["uuid"]}
        for value in stats
    ]
    expected_identity = [
        {"index": value["index"], "uuid": value["uuid"]}
        for value in expected
    ]
    if observed_identity != expected_identity:
        raise RuntimeError(
            "selected-GPU idle query did not return exact physical GPU 0/1 order"
        )
    non_idle = [
        value
        for value in stats
        if value["memory_used_mib"] != 0
    ]
    process_command = [
        "nvidia-smi",
        "-i",
        NVIDIA_VISIBLE_DEVICES,
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    processes = base._run(process_command, runner=runner).stdout.splitlines()
    processes = [value.strip() for value in processes if value.strip()]
    if non_idle or processes:
        raise RuntimeError(
            "selected physical GPU 0/1 must be idle before Tier2S launch"
        )
    return {
        "schema_version": "rc-irstd-aaai27-tier2s-gpu-idle-preflight-v1",
        "verified": True,
        "physical_gpus": list(TARGET_GPU_INDICES),
        "nvidia_visible_devices": NVIDIA_VISIBLE_DEVICES,
        "stats": stats,
        "compute_processes": [],
        "idle_definition": (
            "zero_used_memory_and_no_compute_processes; instantaneous "
            "utilization_is_recorded_but_not_a_standalone_busy_signal"
        ),
    }


_BASE_SCOPE_LOCK = threading.RLock()
_BASE_SCOPE_DEPTH = 0
_BASE_SCOPE_SAVED: dict[str, Any] = {}


def _base_overrides() -> dict[str, Any]:
    return {
        "COORDINATOR": COORDINATOR,
        "CONTAINER_PROBE": CONTAINER_PROBE,
        "PROTOCOL": PROTOCOL,
        "OUTPUT_ROOT": OUTPUT_ROOT,
        "AUDIT_ROOT": AUDIT_ROOT,
        "FORBIDDEN_TARGET": FORBIDDEN_TARGET,
        "INTENT_PATH": INTENT_PATH,
        "INTENT_SHA256_PATH": INTENT_SHA256_PATH,
        "IMAGE_ID": IMAGE_ID,
        "CONTAINER_NAME": CONTAINER_NAME,
        "TARGET_GPU_INDICES": TARGET_GPU_INDICES,
        "SCHEMA": SCHEMA,
        "ENVIRONMENT": ENVIRONMENT,
        "RUNTIME_LABELS": RUNTIME_LABELS,
        "CODE_PATHS": CODE_PATHS,
        "CONFIG_PATHS": CONFIG_PATHS,
        "bind_mount_contract": bind_mount_contract,
        "tmpfs_contract": tmpfs_contract,
        "_common_docker_args": _common_docker_args,
        "build_formal_command": build_formal_command,
        "_verify_in_container": _verify_in_container,
        "_verify_container_attestation": _verify_container_attestation,
        "_validate_existing_container": _validate_existing_container,
    }


@contextmanager
def _base_scope() -> Iterator[None]:
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
        with _base_scope():
            return function(*args, **kwargs)

    return wrapped


def build_verify_command() -> list[str]:
    command = base.build_verify_command()
    _assert_exact_gpu_command(command)
    return command


def build_probe_command() -> list[str]:
    command = base.build_probe_command()
    _assert_exact_gpu_command(command)
    return command


def build_register_command() -> list[str]:
    labels = {
        **RUNTIME_LABELS,
        "org.rc-irstd.launch-purpose": "register-only",
    }
    command = [
        *base._common_docker_args(labels=labels),
        "--rm",
        IMAGE_ID,
        str(COORDINATOR),
        "--register-only",
    ]
    _assert_exact_gpu_command(command)
    return command


def build_formal_command(intent_sha256: str) -> list[str]:
    if not _valid_sha256(intent_sha256):
        raise ValueError("launch-intent SHA-256 must be lowercase hexadecimal")
    labels = {**RUNTIME_LABELS, INTENT_LABEL: intent_sha256}
    command = [
        *_common_docker_args(labels=labels),
        "--detach",
        "--name",
        CONTAINER_NAME,
        "--restart",
        "no",
        IMAGE_ID,
        str(COORDINATOR),
    ]
    _assert_exact_gpu_command(command)
    return command


def container_spec(*, formal: bool) -> dict[str, Any]:
    spec = _BASE_CONTAINER_SPEC(formal=formal)
    spec["protocol_id"] = PROTOCOL_ID
    spec["research_mode"] = RESEARCH_MODE
    spec["source_tier3_authorized"] = False
    spec["paper_claim_authorized"] = False
    spec["restart"] = "no" if formal else None
    spec["gpu_exposure"] = "device=0,1"
    spec["gpu_device_ids"] = list(TARGET_GPU_DEVICE_IDS)
    spec["nvidia_visible_devices"] = NVIDIA_VISIBLE_DEVICES
    return spec


def _register_in_container(
    *, runner: base.Runner | None = None
) -> dict[str, Any]:
    completed = base._run(build_register_command(), runner=runner)
    payload = base._load_json_bytes(
        completed.stdout, context="Tier2S register-only container"
    )
    _coordinator_contract(payload)
    if payload.get("registered") is not True:
        raise RuntimeError("Tier2S isolated registration did not complete")
    for key in (
        "preregistration_sha256",
        "queue_manifest_sha256",
        "target_lock_sha256",
    ):
        value = payload.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"Tier2S isolated registration digest drift:{key}")
    governance = _require_governance_binding(payload)
    governance_registration = governance["registration"]
    assert isinstance(governance_registration, Mapping)
    preregistration = _require_preregistration_binding(
        payload,
        governance_registration_sha256=str(
            governance_registration["sha256"]
        ),
    )
    if payload.get("preregistration_sha256") != preregistration["sha256"]:
        raise RuntimeError(
            "Tier2S isolated registration/preregistration SHA-256 drift"
        )
    return payload


def _validate_host_sources() -> None:
    _assert_new_namespace()
    base._validate_host_sources()


def _ensure_exact_rw_roots() -> None:
    _assert_new_namespace()
    base._ensure_exact_rw_roots()


def verify_only(*, runner: base.Runner | None = None) -> dict[str, Any]:
    """Run isolated preflight without coordinator or host registration."""

    _validate_host_sources()
    _ensure_exact_rw_roots()
    image_id = base._inspect_image(runner=runner)
    inventory = base._gpu_inventory(runner=runner)
    selected = _selected_host_gpus(inventory)
    idle_preflight = _verify_selected_gpus_idle(inventory, runner=runner)
    verification = _verify_in_container(runner=runner)
    attestation = _verify_container_attestation(inventory, runner=runner)
    return {
        "schema_version": SCHEMA,
        "verified": True,
        "formal_container_started": False,
        "formal_artifact_registered": False,
        "protocol_id": PROTOCOL_ID,
        "research_mode": RESEARCH_MODE,
        "source_only": True,
        "outer_target_access_authorized": False,
        "source_tier3_authorized": False,
        "paper_claim_authorized": False,
        "image_id": image_id,
        "host_gpu_inventory": list(inventory),
        "coordinator_assigned_host_gpus": selected,
        "selected_gpu_idle_preflight": idle_preflight,
        "container_attestation": attestation,
        "verification_container_spec": container_spec(formal=False),
        "formal_container_spec": container_spec(formal=True),
        "coordinator_verification": verification,
    }


def _launch_intent_payload(
    *,
    image_id: str,
    gpu_inventory: Sequence[Mapping[str, Any]],
    attestation: Mapping[str, Any],
    verification: Mapping[str, Any],
    registration: Mapping[str, Any],
    idle_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    verification_governance = _require_governance_binding(verification)
    registration_governance = _require_governance_binding(registration)
    if registration_governance != verification_governance:
        raise RuntimeError(
            "Tier2S governance binding changed between verify and register"
        )
    registration_artifact = verification_governance["registration"]
    assert isinstance(registration_artifact, Mapping)
    preregistration = _require_preregistration_binding(
        registration,
        governance_registration_sha256=str(registration_artifact["sha256"]),
    )
    if (
        attestation.get("physical_gpu_allowlist_verified")
        != list(TARGET_GPU_INDICES)
        or attestation.get("host_gpu_uuid_mapping_verified") is not True
        or attestation.get("torch_ordinal_mapping_verified") is not True
    ):
        raise RuntimeError("Tier2S exact GPU attestation is absent")
    if (
        idle_preflight.get("verified") is not True
        or idle_preflight.get("physical_gpus") != list(TARGET_GPU_INDICES)
        or idle_preflight.get("nvidia_visible_devices")
        != NVIDIA_VISIBLE_DEVICES
        or idle_preflight.get("compute_processes") != []
    ):
        raise RuntimeError("Tier2S selected-GPU idle preflight binding drift")
    payload = _BASE_LAUNCH_INTENT_PAYLOAD(
        image_id=image_id,
        gpu_inventory=gpu_inventory,
        attestation=attestation,
        verification=verification,
    )
    payload.update(
        {
            "schema_version": SCHEMA,
            "decision": "LAUNCH_TIER2S_FACTORIZED_EXPLORATORY_SOURCE_ONLY",
            "protocol_id": PROTOCOL_ID,
            "research_mode": RESEARCH_MODE,
            "source_tier3_authorized": False,
            "paper_claim_authorized": False,
            "governance_binding": verification_governance,
            "tier2s_preregistration_binding": preregistration,
            "selected_gpu_idle_preflight": dict(idle_preflight),
            "physical_gpu_allowlist": list(TARGET_GPU_INDICES),
            "docker_gpu_request": DOCKER_GPU_REQUEST,
            "nvidia_visible_devices": NVIDIA_VISIBLE_DEVICES,
            "isolated_registration": dict(registration),
            "isolated_registration_canonical_sha256": hashlib.sha256(
                base._canonical_bytes(dict(registration))
            ).hexdigest(),
            "frozen_tier2r_namespace_modified": False,
        }
    )
    return payload


def _validate_existing_container(
    container: Mapping[str, Any], intent_sha256: str
) -> str:
    config = container.get("Config")
    host = container.get("HostConfig")
    state = container.get("State")
    mounts = container.get("Mounts")
    if not all(
        isinstance(value, Mapping) for value in (config, host, state)
    ):
        raise RuntimeError("existing container inspect sections are incomplete")
    if not isinstance(mounts, list):
        raise RuntimeError("existing container mount inspection is absent")
    assert isinstance(config, Mapping)
    assert isinstance(host, Mapping)
    assert isinstance(state, Mapping)
    labels = config.get("Labels")
    environment = config.get("Env")
    restart = host.get("RestartPolicy")
    if not isinstance(labels, Mapping):
        labels = {}
    if not isinstance(environment, list):
        environment = []
    if not isinstance(restart, Mapping):
        restart = {}
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
        (
            restart.get("Name") == "no"
            and restart.get("MaximumRetryCount", 0) == 0,
            "restart policy",
        ),
        (labels.get(INTENT_LABEL) == intent_sha256, "intent label"),
    )
    drift = [name for valid, name in expected_simple if not valid]
    environment_strings = [
        value for value in environment if isinstance(value, str)
    ]
    for key, value in ENVIRONMENT.items():
        matches = [
            entry
            for entry in environment_strings
            if entry.startswith(f"{key}=")
        ]
        if matches != [f"{key}={value}"]:
            drift.append(f"environment:{key}")
    if any(
        entry.startswith("CUDA_VISIBLE_DEVICES=")
        for entry in environment_strings
    ):
        drift.append("environment:CUDA_VISIBLE_DEVICES")
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
    observed_tmpfs = host.get("Tmpfs")
    if not isinstance(observed_tmpfs, Mapping):
        observed_tmpfs = {}
    expected_tmpfs = {
        value["destination"]: base._tmpfs_options(value["options"])
        for value in tmpfs_contract()
    }
    if set(observed_tmpfs) != set(expected_tmpfs) or any(
        base._tmpfs_options(str(observed_tmpfs.get(path, ""))) != options
        for path, options in expected_tmpfs.items()
    ):
        drift.append("tmpfs")
    requests = host.get("DeviceRequests")
    exact_request = (
        isinstance(requests, list)
        and len(requests) == 1
        and isinstance(requests[0], Mapping)
        and requests[0].get("Driver") == ""
        and requests[0].get("Count") == 0
        and requests[0].get("DeviceIDs") == list(TARGET_GPU_DEVICE_IDS)
        and requests[0].get("Capabilities") == [["gpu"]]
        and requests[0].get("Options") == {}
    )
    if not exact_request:
        drift.append("GPU exposure must be exact DeviceIDs 0,1")
    if drift:
        raise RuntimeError(
            "existing named container contract drift: " + ", ".join(drift)
        )
    status = state.get("Status")
    if status not in {"created", "running", "exited"}:
        raise RuntimeError(
            f"existing named container is not safely reconcilable: {status}"
        )
    return str(status)


def _start_or_reconcile(
    intent_sha256: str, *, runner: base.Runner | None = None
) -> dict[str, Any]:
    return base._start_or_reconcile(intent_sha256, runner=runner)


def execute_launch(*, runner: base.Runner | None = None) -> dict[str, Any]:
    """Verify, register in isolation, freeze intent, then start/reconcile."""

    evidence = verify_only(runner=runner)
    registration = _register_in_container(runner=runner)
    launch_idle_preflight = _verify_selected_gpus_idle(
        evidence["host_gpu_inventory"], runner=runner
    )
    intent = _launch_intent_payload(
        image_id=str(evidence["image_id"]),
        gpu_inventory=evidence["host_gpu_inventory"],
        attestation=evidence["container_attestation"],
        verification=evidence["coordinator_verification"],
        registration=registration,
        idle_preflight=launch_idle_preflight,
    )
    intent_sha256 = base._register_launch_intent(intent)
    launch = _start_or_reconcile(intent_sha256, runner=runner)
    return {
        "schema_version": SCHEMA,
        "verified_before_registration": True,
        "registered_before_launch": True,
        "protocol_id": PROTOCOL_ID,
        "research_mode": RESEARCH_MODE,
        "source_only": True,
        "outer_target_access_authorized": False,
        "source_tier3_authorized": False,
        "paper_claim_authorized": False,
        "launch_intent": str(INTENT_PATH),
        "launch_intent_sha256": intent_sha256,
        "isolated_registration": registration,
        "selected_gpu_idle_preflight": launch_idle_preflight,
        "container": launch,
    }


def dry_run_payload() -> dict[str, Any]:
    _assert_new_namespace()
    return {
        "schema_version": SCHEMA,
        "dry_run": True,
        "docker_invoked": False,
        "filesystem_mutated": False,
        "protocol_id": PROTOCOL_ID,
        "research_mode": RESEARCH_MODE,
        "container_name": CONTAINER_NAME,
        "image": IMAGE_ID,
        "source_only": True,
        "outer_target_access_authorized": False,
        "source_tier3_authorized": False,
        "paper_claim_authorized": False,
        "coordinator_physical_gpus": list(TARGET_GPU_INDICES),
        "wait_for_idle_gpu": False,
        "require_selected_gpus_idle_at_preflight_and_launch": True,
        "docker_gpu_request": DOCKER_GPU_REQUEST,
        "nvidia_visible_devices": NVIDIA_VISIBLE_DEVICES,
        "verify_command": build_verify_command(),
        "register_command": build_register_command(),
        "formal_command_template": build_formal_command("0" * 64),
        "formal_container_spec": container_spec(formal=True),
        "execute_with": f"python3 {Path(__file__).resolve()} --execute",
    }


build_verify_command = _scoped(build_verify_command)
build_probe_command = _scoped(build_probe_command)
build_register_command = _scoped(build_register_command)
build_formal_command = _scoped(build_formal_command)
container_spec = _scoped(container_spec)
_register_in_container = _scoped(_register_in_container)
_validate_host_sources = _scoped(_validate_host_sources)
_ensure_exact_rw_roots = _scoped(_ensure_exact_rw_roots)
verify_only = _scoped(verify_only)
_launch_intent_payload = _scoped(_launch_intent_payload)
_start_or_reconcile = _scoped(_start_or_reconcile)
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
    "build_probe_command",
    "build_register_command",
    "build_verify_command",
    "container_spec",
    "dry_run_payload",
    "execute_launch",
    "tmpfs_contract",
    "verify_only",
]
