#!/usr/bin/env python3
"""Append-only Tier2S launcher correcting container-local NVIDIA indices.

Host physical GPU identity is selected by indices 2 and 3 and bound to UUIDs.
Inside the exact two-device container, NVIDIA indices and Torch ordinals must be
0 and 1 and must carry those same UUID/name pairs in order. All other Tier2S
scientific, isolation, scheduling, and idle contracts remain unchanged.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import functools
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterator, Mapping, Sequence

from scripts import launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum1 as parent
from scripts import register_tier2s_gpu23_execution_erratum2 as execution_erratum2


PROJECT_ROOT = Path("/home/ly/RC-IRSTD-v2")
PROTOCOL_ID = parent.PROTOCOL_ID
RESEARCH_MODE = parent.RESEARCH_MODE
TARGET_GPU_INDICES = parent.TARGET_GPU_INDICES
CONTAINER_ORDINAL_BY_PHYSICAL = parent.CONTAINER_ORDINAL_BY_PHYSICAL
NVIDIA_VISIBLE_DEVICES = parent.NVIDIA_VISIBLE_DEVICES
DOCKER_GPU_REQUEST = parent.DOCKER_GPU_REQUEST
DRIVER_BOOKKEEPING_MAX_MIB = parent.DRIVER_BOOKKEEPING_MAX_MIB
PROBE_SCHEMA = parent.PROBE_SCHEMA
IMAGE_ID = parent.IMAGE_ID
SCHEMA = (
    "rc-irstd-aaai27-tier2s-factorized-isolated-launch-intent-v2-gpu23-"
    "execution-erratum2"
)
CONTAINER_NAME = (
    "rc-irstd-tier2s-factorized-causal-audit-v2-gpu23-execution-erratum2"
)
AUDIT_ROOT = parent.AUDIT_ROOT
INTENT_PATH = AUDIT_ROOT / "LAUNCH_INTENT_EXECUTION_ERRATUM2.json"
INTENT_SHA256_PATH = INTENT_PATH.with_suffix(INTENT_PATH.suffix + ".sha256")
EXECUTION_ERRATUM_CONFIG = (
    PROJECT_ROOT / "configs/tier2s_gpu23_execution_erratum2.json"
)
EXECUTION_ERRATUM_REGISTRATION_RELATIVE = Path(
    "artifacts/aaai27/audit/governance/tier2s_gpu23_execution_erratum2/"
    "TIER2S_GPU23_EXECUTION_ERRATUM2.json"
)
REGISTRAR = PROJECT_ROOT / "scripts/register_tier2s_gpu23_execution_erratum2.py"
TEST_PATH = PROJECT_ROOT / "tests/test_tier2s_gpu23_execution_erratum2.py"
REGISTRATION = PROJECT_ROOT / EXECUTION_ERRATUM_REGISTRATION_RELATIVE
REGISTRATION_SIDECAR = REGISTRATION.with_suffix(REGISTRATION.suffix + ".sha256")

RUNTIME_LABELS = {
    **parent.RUNTIME_LABELS,
    "org.rc-irstd.execution-erratum": "container-index-uuid-binding-v2",
}
CODE_PATHS = tuple(
    dict.fromkeys(
        (
            *parent.CODE_PATHS,
            Path(__file__).resolve(),
            REGISTRAR,
            TEST_PATH,
            REGISTRATION,
            REGISTRATION_SIDECAR,
        )
    )
)
CONFIG_PATHS = tuple(
    dict.fromkeys((*parent.CONFIG_PATHS, EXECUTION_ERRATUM_CONFIG))
)

_PARENT_LAUNCH_INTENT = parent._launch_intent_payload
_SCOPE_LOCK = threading.RLock()
_SCOPE_DEPTH = 0
_SCOPE_SAVED: dict[str, Any] = {}


def _expected_container_inventories(
    host_inventory: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected = parent._selected_host_gpus(host_inventory)
    nvidia: list[dict[str, Any]] = []
    torch: list[dict[str, Any]] = []
    binding: list[dict[str, Any]] = []
    for ordinal, host in enumerate(selected):
        nvidia.append(
            {"index": ordinal, "uuid": host["uuid"], "name": host["name"]}
        )
        torch.append(
            {"ordinal": ordinal, "uuid": host["uuid"], "name": host["name"]}
        )
        binding.append(
            {
                "physical_index": host["index"],
                "container_nvidia_index": ordinal,
                "torch_ordinal": ordinal,
                "uuid": host["uuid"],
                "name": host["name"],
            }
        )
    return nvidia, torch, binding


def _verify_attestation_payload(
    host_inventory: Sequence[Mapping[str, Any]],
    raw_payload: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(raw_payload)
    expected_nvidia, expected_torch, uuid_binding = (
        _expected_container_inventories(host_inventory)
    )
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
        or target.get("path") != str(parent.FORBIDDEN_TARGET)
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
        (parent.OUTPUT_ROOT, True),
        (parent.AUDIT_ROOT, True),
        (parent.FORBIDDEN_TARGET, True),
    ):
        entry = mounts.get(str(path))
        options = (
            set(entry.get("mount_options", []))
            if isinstance(entry, Mapping)
            else set()
        )
        if not isinstance(entry, Mapping) or (("rw" in options) is not writable):
            problems.append(f"mount mode drift:{path}")
    target_mount = mounts.get(str(parent.FORBIDDEN_TARGET))
    if (
        not isinstance(target_mount, Mapping)
        or target_mount.get("filesystem_type") != "tmpfs"
    ):
        problems.append("outer-target mount is not tmpfs")
    if payload.get("nvidia_inventory") != expected_nvidia:
        problems.append(
            "container NVIDIA indices 0/1 must UUID-bind exactly to host physical 2/3"
        )
    if payload.get("torch_inventory") != expected_torch:
        problems.append(
            "Torch ordinals 0/1 must UUID-bind exactly to host physical 2/3"
        )
    environment = payload.get("environment")
    if (
        not isinstance(environment, Mapping)
        or environment.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
        or environment.get("CUDA_VISIBLE_DEVICES") is not None
        or environment.get("NVIDIA_VISIBLE_DEVICES") != NVIDIA_VISIBLE_DEVICES
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
            "Tier2S erratum2 container attestation failed: "
            + "; ".join(problems)
        )
    payload["verified"] = True
    payload["physical_gpu_allowlist_verified"] = list(TARGET_GPU_INDICES)
    payload["host_gpu_uuid_mapping_verified"] = True
    payload["torch_ordinal_mapping_verified"] = True
    payload["container_nvidia_indices_verified"] = [0, 1]
    payload["physical_to_container_uuid_binding"] = uuid_binding
    payload["outer_target_alias_absent"] = True
    return payload


def _verify_container_attestation(
    host_inventory: Sequence[Mapping[str, Any]],
    *,
    runner: parent.base.Runner | None = None,
) -> dict[str, Any]:
    completed = parent.base._run(parent.build_probe_command(), runner=runner)
    payload = parent.base._load_json_bytes(
        completed.stdout, context="Tier2S erratum2 container probe"
    )
    return _verify_attestation_payload(host_inventory, payload)


def _require_execution_erratum_binding() -> dict[str, Any]:
    return execution_erratum2.require_frozen_execution_erratum2()


def _launch_intent_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = _PARENT_LAUNCH_INTENT(*args, **kwargs)
    payload["decision"] = (
        "LAUNCH_TIER2S_FACTORIZED_SOURCE_ONLY_WITH_EXECUTION_ERRATUM2"
    )
    payload["container_index_namespace_attestation"] = {
        "host_physical_indices": [2, 3],
        "container_nvidia_indices": [0, 1],
        "torch_ordinals": [0, 1],
        "identity_authority": "uuid_and_name_in_host_physical_order",
    }
    return payload


def _overrides() -> dict[str, Any]:
    return {
        "SCHEMA": SCHEMA,
        "CONTAINER_NAME": CONTAINER_NAME,
        "INTENT_PATH": INTENT_PATH,
        "INTENT_SHA256_PATH": INTENT_SHA256_PATH,
        "EXECUTION_ERRATUM_CONFIG": EXECUTION_ERRATUM_CONFIG,
        "EXECUTION_ERRATUM_REGISTRATION_RELATIVE": (
            EXECUTION_ERRATUM_REGISTRATION_RELATIVE
        ),
        "RUNTIME_LABELS": RUNTIME_LABELS,
        "CODE_PATHS": CODE_PATHS,
        "CONFIG_PATHS": CONFIG_PATHS,
        "_verify_container_attestation": _verify_container_attestation,
        "_require_execution_erratum_binding": _require_execution_erratum_binding,
        "_launch_intent_payload": _launch_intent_payload,
    }


@contextmanager
def _parent_scope() -> Iterator[None]:
    global _SCOPE_DEPTH, _SCOPE_SAVED
    with _SCOPE_LOCK:
        if _SCOPE_DEPTH == 0:
            values = _overrides()
            _SCOPE_SAVED = {name: getattr(parent, name) for name in values}
            for name, value in values.items():
                setattr(parent, name, value)
        _SCOPE_DEPTH += 1
        try:
            yield
        finally:
            _SCOPE_DEPTH -= 1
            if _SCOPE_DEPTH == 0:
                for name, value in _SCOPE_SAVED.items():
                    setattr(parent, name, value)
                _SCOPE_SAVED = {}


def _scoped(function: Any) -> Any:
    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with _parent_scope():
            return function(*args, **kwargs)

    return wrapped


@_scoped
def build_verify_command() -> list[str]:
    return parent.build_verify_command()


@_scoped
def build_probe_command() -> list[str]:
    return parent.build_probe_command()


@_scoped
def build_register_command() -> list[str]:
    return parent.build_register_command()


@_scoped
def build_formal_command(intent_sha256: str) -> list[str]:
    return parent.build_formal_command(intent_sha256)


@_scoped
def container_spec(*, formal: bool) -> dict[str, Any]:
    return parent.container_spec(formal=formal)


@_scoped
def verify_only(
    *, runner: parent.base.Runner | None = None
) -> dict[str, Any]:
    return parent.verify_only(runner=runner)


@_scoped
def execute_launch(
    *, runner: parent.base.Runner | None = None
) -> dict[str, Any]:
    return parent.execute_launch(runner=runner)


@_scoped
def dry_run_payload() -> dict[str, Any]:
    payload = parent.dry_run_payload()
    payload["execute_with"] = (
        f"python3 {Path(__file__).resolve()} --execute"
    )
    return payload


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
    "build_formal_command",
    "build_probe_command",
    "build_register_command",
    "build_verify_command",
    "container_spec",
    "dry_run_payload",
    "execute_launch",
    "verify_only",
]

