#!/usr/bin/env python3
"""Append-only Tier2S outer execution relocation from busy GPU2/3 to GPU0/1.

The frozen v2 coordinator, 18-job matrix, checkpoints, source-only data rules,
and evaluator are unchanged.  Only the outer Docker device request and its
host UUID/idle attestation are rebound to physical GPU0/1.  The coordinator's
frozen legacy ``physical_gpu`` values 2/3 are retained as lane labels and are
explicitly mapped to host physical 0/1 in the launch intent.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import functools
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any, Iterator, Mapping, Sequence

PROJECT_ROOT = Path("/home/ly/RC-IRSTD-v2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum1 as engine
from scripts import launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum2 as attestation_parent
from scripts import launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum3 as frozen_parent
from scripts import register_tier2s_execution_device_relocation_gpu01_v1 as relocation_governance


HOST_PHYSICAL_GPUS = (0, 1)
HOST_DEVICE_IDS = ("0", "1")
HOST_TO_CONTAINER_ORDINAL = {0: 0, 1: 1}
FROZEN_COORDINATOR_LANES = (2, 3)
LANE_TO_HOST_PHYSICAL = {2: 0, 3: 1}
DOCKER_GPU_REQUEST = '"device=0,1"'
NVIDIA_VISIBLE_DEVICES = "0,1"
DRIVER_BOOKKEEPING_MAX_MIB = 4
SCHEMA = (
    "rc-irstd-aaai27-tier2s-factorized-isolated-launch-intent-v2-"
    "device-relocation-gpu01-v1"
)
CONTAINER_NAME = (
    "rc-irstd-tier2s-factorized-causal-audit-v2-device-relocation-gpu01-v1"
)
AUDIT_ROOT = engine.AUDIT_ROOT
INTENT_PATH = AUDIT_ROOT / "LAUNCH_INTENT_EXECUTION_DEVICE_RELOCATION_GPU01_V1.json"
INTENT_SHA256_PATH = INTENT_PATH.with_suffix(INTENT_PATH.suffix + ".sha256")
EXECUTION_ERRATUM_CONFIG = (
    PROJECT_ROOT / "configs/tier2s_execution_device_relocation_gpu01_v1.json"
)
EXECUTION_ERRATUM_REGISTRATION_RELATIVE = Path(
    "artifacts/aaai27/audit/governance/"
    "tier2s_execution_device_relocation_gpu01_v1/"
    "TIER2S_EXECUTION_DEVICE_RELOCATION_GPU01_V1.json"
)
REGISTRATION = PROJECT_ROOT / EXECUTION_ERRATUM_REGISTRATION_RELATIVE
REGISTRATION_SIDECAR = REGISTRATION.with_suffix(REGISTRATION.suffix + ".sha256")
REGISTRAR = PROJECT_ROOT / "scripts/register_tier2s_execution_device_relocation_gpu01_v1.py"
TEST_PATH = PROJECT_ROOT / "tests/test_tier2s_execution_device_relocation_gpu01_v1.py"

ENVIRONMENT = dict(engine.ENVIRONMENT)
ENVIRONMENT["NVIDIA_VISIBLE_DEVICES"] = NVIDIA_VISIBLE_DEVICES
RUNTIME_LABELS = {
    **engine.RUNTIME_LABELS,
    "org.rc-irstd.physical-gpus": NVIDIA_VISIBLE_DEVICES,
    "org.rc-irstd.frozen-coordinator-lanes": "2,3",
    "org.rc-irstd.execution-relocation": "gpu23-lanes-to-host-gpu01-v1",
}
CODE_PATHS = tuple(
    dict.fromkeys(
        (
            *frozen_parent.CODE_PATHS,
            Path(__file__).resolve(),
            REGISTRAR,
            TEST_PATH,
            REGISTRATION,
            REGISTRATION_SIDECAR,
        )
    )
)
CONFIG_PATHS = tuple(
    dict.fromkeys((*frozen_parent.CONFIG_PATHS, EXECUTION_ERRATUM_CONFIG))
)

_ENGINE_CONTAINER_SPEC = engine.container_spec
_ENGINE_LAUNCH_INTENT = engine._launch_intent_payload
_SCOPE_LOCK = threading.RLock()
_SCOPE_DEPTH = 0
_SAVED_ENGINE: dict[str, Any] = {}
_SAVED_ATTESTATION: dict[str, Any] = {}


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
    if any(index not in by_index for index in HOST_PHYSICAL_GPUS):
        raise RuntimeError("required relocated host physical GPU0/1 is absent")
    selected = [by_index[index] for index in HOST_PHYSICAL_GPUS]
    if len({value["uuid"] for value in selected}) != 2:
        raise RuntimeError("relocated host GPU UUIDs are not unique")
    return selected


def _verify_selected_gpus_idle(
    host_inventory: Sequence[Mapping[str, Any]],
    *,
    runner: engine.base.Runner | None = None,
) -> dict[str, Any]:
    expected = _selected_host_gpus(host_inventory)
    completed = engine.base._run(
        [
            "nvidia-smi",
            "-i",
            NVIDIA_VISIBLE_DEVICES,
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        runner=runner,
    )
    stats: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [value.strip() for value in line.split(",", 3)]
        if len(fields) != 4:
            raise RuntimeError(f"malformed relocated-GPU idle row: {line!r}")
        try:
            index = int(fields[0])
            memory_used_mib = int(fields[2])
            utilization_percent = int(fields[3])
        except ValueError as error:
            raise RuntimeError(f"non-integer relocated-GPU idle row: {line!r}") from error
        stats.append(
            {
                "index": index,
                "uuid": fields[1],
                "memory_used_mib": memory_used_mib,
                "utilization_gpu_percent": utilization_percent,
            }
        )
    observed = [{"index": v["index"], "uuid": v["uuid"]} for v in stats]
    wanted = [{"index": v["index"], "uuid": v["uuid"]} for v in expected]
    if observed != wanted:
        raise RuntimeError("idle query did not return exact host GPU0/1 UUID order")
    processes = engine.base._run(
        [
            "nvidia-smi",
            "-i",
            NVIDIA_VISIBLE_DEVICES,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        runner=runner,
    ).stdout.splitlines()
    processes = [value.strip() for value in processes if value.strip()]
    if any(v["memory_used_mib"] > DRIVER_BOOKKEEPING_MAX_MIB for v in stats) or processes:
        raise RuntimeError("relocated host physical GPU0/1 exceeds driver-only idle contract")
    return {
        "schema_version": "rc-irstd-aaai27-tier2s-gpu-idle-preflight-relocation-gpu01-v1",
        "verified": True,
        "physical_gpus": list(HOST_PHYSICAL_GPUS),
        "nvidia_visible_devices": NVIDIA_VISIBLE_DEVICES,
        "stats": stats,
        "compute_processes": [],
        "driver_bookkeeping_ceiling_mib": DRIVER_BOOKKEEPING_MAX_MIB,
        "no_compute_processes_required": True,
        "idle_definition": (
            "memory_used_mib_at_most_4_and_no_compute_processes; instantaneous "
            "utilization_is_recorded_but_not_a_standalone_busy_signal"
        ),
    }


def _verify_container_attestation(
    host_inventory: Sequence[Mapping[str, Any]],
    *,
    runner: engine.base.Runner | None = None,
) -> dict[str, Any]:
    completed = engine.base._run(engine.build_probe_command(), runner=runner)
    raw = engine.base._load_json_bytes(
        completed.stdout, context="Tier2S GPU0/1 relocation container probe"
    )
    return attestation_parent._verify_attestation_payload(host_inventory, raw)


def _require_execution_erratum_binding() -> dict[str, Any]:
    return relocation_governance.require_frozen_device_relocation_gpu01_v1()


def _container_spec(*, formal: bool) -> dict[str, Any]:
    spec = _ENGINE_CONTAINER_SPEC(formal=formal)
    spec["gpu_exposure"] = "device=0,1"
    spec["gpu_device_ids"] = list(HOST_DEVICE_IDS)
    spec["nvidia_visible_devices"] = NVIDIA_VISIBLE_DEVICES
    spec["host_physical_gpus"] = list(HOST_PHYSICAL_GPUS)
    spec["frozen_coordinator_lane_labels"] = list(FROZEN_COORDINATOR_LANES)
    spec["lane_to_host_physical_gpu"] = {
        str(key): value for key, value in LANE_TO_HOST_PHYSICAL.items()
    }
    return spec


@contextmanager
def _temporary_engine_target(indices: tuple[int, int]) -> Iterator[None]:
    old = engine.TARGET_GPU_INDICES
    engine.TARGET_GPU_INDICES = indices
    try:
        yield
    finally:
        engine.TARGET_GPU_INDICES = old


def _launch_intent_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    with _temporary_engine_target(HOST_PHYSICAL_GPUS):
        payload = _ENGINE_LAUNCH_INTENT(*args, **kwargs)
    inventory = kwargs.get("gpu_inventory", [])
    payload["schema_version"] = SCHEMA
    payload["decision"] = "LAUNCH_TIER2S_SOURCE_ONLY_WITH_DEVICE_RELOCATION_GPU01_V1"
    payload["coordinator_assigned_host_gpus"] = _selected_host_gpus(inventory)
    payload["outer_execution_device_relocation"] = {
        "host_physical_gpus": list(HOST_PHYSICAL_GPUS),
        "container_nvidia_indices": [0, 1],
        "torch_ordinals": [0, 1],
        "host_physical_to_container_ordinal": {"0": 0, "1": 1},
        "frozen_coordinator_lane_labels": list(FROZEN_COORDINATOR_LANES),
        "legacy_lane_to_host_physical_gpu": {"2": 0, "3": 1},
        "scientific_matrix_changed": False,
        "result_use": "failure_attribution_only",
    }
    return payload


def _engine_overrides() -> dict[str, Any]:
    return {
        "SCHEMA": SCHEMA,
        "CONTAINER_NAME": CONTAINER_NAME,
        "INTENT_PATH": INTENT_PATH,
        "INTENT_SHA256_PATH": INTENT_SHA256_PATH,
        "EXECUTION_ERRATUM_CONFIG": EXECUTION_ERRATUM_CONFIG,
        "EXECUTION_ERRATUM_REGISTRATION_RELATIVE": EXECUTION_ERRATUM_REGISTRATION_RELATIVE,
        "TARGET_GPU_DEVICE_IDS": HOST_DEVICE_IDS,
        "DOCKER_GPU_REQUEST": DOCKER_GPU_REQUEST,
        "NVIDIA_VISIBLE_DEVICES": NVIDIA_VISIBLE_DEVICES,
        "DRIVER_BOOKKEEPING_MAX_MIB": DRIVER_BOOKKEEPING_MAX_MIB,
        "ENVIRONMENT": ENVIRONMENT,
        "RUNTIME_LABELS": RUNTIME_LABELS,
        "CODE_PATHS": CODE_PATHS,
        "CONFIG_PATHS": CONFIG_PATHS,
        "_selected_host_gpus": _selected_host_gpus,
        "_verify_selected_gpus_idle": _verify_selected_gpus_idle,
        "_verify_container_attestation": _verify_container_attestation,
        "_require_execution_erratum_binding": _require_execution_erratum_binding,
        "_launch_intent_payload": _launch_intent_payload,
        "container_spec": _container_spec,
    }


def _attestation_overrides() -> dict[str, Any]:
    return {
        "TARGET_GPU_INDICES": HOST_PHYSICAL_GPUS,
        "CONTAINER_ORDINAL_BY_PHYSICAL": HOST_TO_CONTAINER_ORDINAL,
        "NVIDIA_VISIBLE_DEVICES": NVIDIA_VISIBLE_DEVICES,
        "DOCKER_GPU_REQUEST": DOCKER_GPU_REQUEST,
    }


@contextmanager
def _relocation_scope() -> Iterator[None]:
    global _SCOPE_DEPTH, _SAVED_ENGINE, _SAVED_ATTESTATION
    with _SCOPE_LOCK:
        if _SCOPE_DEPTH == 0:
            engine_values = _engine_overrides()
            attestation_values = _attestation_overrides()
            _SAVED_ENGINE = {name: getattr(engine, name) for name in engine_values}
            _SAVED_ATTESTATION = {
                name: getattr(attestation_parent, name) for name in attestation_values
            }
            for name, value in engine_values.items():
                setattr(engine, name, value)
            for name, value in attestation_values.items():
                setattr(attestation_parent, name, value)
        _SCOPE_DEPTH += 1
        try:
            yield
        finally:
            _SCOPE_DEPTH -= 1
            if _SCOPE_DEPTH == 0:
                for name, value in _SAVED_ATTESTATION.items():
                    setattr(attestation_parent, name, value)
                for name, value in _SAVED_ENGINE.items():
                    setattr(engine, name, value)
                _SAVED_ENGINE = {}
                _SAVED_ATTESTATION = {}


def _scoped(function: Any) -> Any:
    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with _relocation_scope():
            return function(*args, **kwargs)
    return wrapped


@_scoped
def build_verify_command() -> list[str]:
    return engine.build_verify_command()


@_scoped
def build_probe_command() -> list[str]:
    return engine.build_probe_command()


@_scoped
def build_register_command() -> list[str]:
    return engine.build_register_command()


@_scoped
def build_formal_command(intent_sha256: str) -> list[str]:
    return engine.build_formal_command(intent_sha256)


@_scoped
def container_spec(*, formal: bool) -> dict[str, Any]:
    return _container_spec(formal=formal)


@_scoped
def verify_only(*, runner: engine.base.Runner | None = None) -> dict[str, Any]:
    payload = engine.verify_only(runner=runner)
    payload["host_physical_gpus"] = list(HOST_PHYSICAL_GPUS)
    payload["frozen_coordinator_lane_labels"] = list(FROZEN_COORDINATOR_LANES)
    payload["legacy_lane_to_host_physical_gpu"] = {"2": 0, "3": 1}
    return payload


@_scoped
def execute_launch(*, runner: engine.base.Runner | None = None) -> dict[str, Any]:
    payload = engine.execute_launch(runner=runner)
    payload["host_physical_gpus"] = list(HOST_PHYSICAL_GPUS)
    payload["frozen_coordinator_lane_labels"] = list(FROZEN_COORDINATOR_LANES)
    payload["legacy_lane_to_host_physical_gpu"] = {"2": 0, "3": 1}
    return payload


@_scoped
def dry_run_payload() -> dict[str, Any]:
    payload = engine.dry_run_payload()
    payload["host_physical_gpus"] = list(HOST_PHYSICAL_GPUS)
    payload["frozen_coordinator_lane_labels"] = list(FROZEN_COORDINATOR_LANES)
    payload["legacy_lane_to_host_physical_gpu"] = {"2": 0, "3": 1}
    payload["execute_with"] = f"python3 {Path(__file__).resolve()} --execute"
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
