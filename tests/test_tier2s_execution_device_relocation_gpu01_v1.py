from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import launch_tier2s_factorized_isolated_v2_device_relocation_gpu01_v1 as launcher
from scripts import register_tier2s_execution_device_relocation_gpu01_v1 as registrar


ROOT = Path(__file__).resolve().parents[1]


def _inventory() -> list[dict[str, object]]:
    return [
        {"index": 0, "uuid": "GPU-0", "name": "RTX"},
        {"index": 1, "uuid": "GPU-1", "name": "RTX"},
        {"index": 2, "uuid": "GPU-2", "name": "RTX"},
        {"index": 3, "uuid": "GPU-3", "name": "RTX"},
    ]


def _runner(rows: str, processes: str = "") -> object:
    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        if any(value.startswith("--query-gpu=") for value in command):
            return subprocess.CompletedProcess(command, 0, rows, "")
        return subprocess.CompletedProcess(command, 0, processes, "")
    return runner


def test_relocation_config_is_append_only_and_science_invariant() -> None:
    config = json.loads(registrar.RELOCATION_CONFIG.read_text(encoding="utf-8"))
    assert config["reason"]["user_resource_authorization"] == "gpu0-4随便用"
    assert config["relocated_execution_contract"]["host_physical_gpus"] == [0, 1]
    assert config["unchanged_scientific_contract"]["coordinator_matrix_unchanged"] is True
    assert config["unchanged_scientific_contract"]["total_jobs"] == 18
    assert config["unchanged_scientific_contract"]["result_use"] == "failure_attribution_only"
    assert config["append_only_policy"]["parent_gpu23_files_mutated"] is False
    assert config["append_only_policy"]["scientific_protocol_replaced"] is False


def test_parent_erratum3_is_exactly_frozen_and_formal_execution_absent() -> None:
    parent = registrar.parent_erratum3.require_frozen_execution_erratum3(
        expected_registration_sha256=registrar.PARENT_REGISTRATION_SHA256
    )
    assert parent["physical_gpus"] == [2, 3]
    assert registrar._sha256(registrar.PARENT_REGISTRATION_SIDECAR) == (
        registrar.PARENT_REGISTRATION_SIDECAR_SHA256
    )
    assert registrar._sha256(registrar.PARENT_CONFIG) == registrar.PARENT_CONFIG_SHA256
    assert registrar._sha256(registrar.PARENT_LAUNCHER) == registrar.PARENT_LAUNCHER_SHA256
    assert registrar._require_parent_formal_absent()["parent_gpu23_formal_intent_absent"] is True


def test_relocation_selector_uses_host_gpu01_only() -> None:
    selected = launcher._selected_host_gpus(_inventory())
    assert [value["index"] for value in selected] == [0, 1]
    assert [value["uuid"] for value in selected] == ["GPU-0", "GPU-1"]
    with pytest.raises(RuntimeError, match="GPU0/1 is absent"):
        launcher._selected_host_gpus(_inventory()[1:])


def test_idle_relocation_accepts_driver_bookkeeping_only() -> None:
    evidence = launcher._verify_selected_gpus_idle(
        _inventory(), runner=_runner("0, GPU-0, 1, 0\n1, GPU-1, 4, 0\n")
    )
    assert evidence["physical_gpus"] == [0, 1]
    assert evidence["driver_bookkeeping_ceiling_mib"] == 4
    assert evidence["compute_processes"] == []


@pytest.mark.parametrize(
    ("rows", "processes"),
    (
        ("0, GPU-0, 5, 0\n1, GPU-1, 1, 0\n", ""),
        ("0, GPU-0, 1, 0\n1, GPU-1, 1, 0\n", "GPU-0, 123, python, 1\n"),
    ),
)
def test_idle_relocation_rejects_load_or_compute_process(rows: str, processes: str) -> None:
    with pytest.raises(RuntimeError, match="driver-only idle contract"):
        launcher._verify_selected_gpus_idle(
            _inventory(), runner=_runner(rows, processes)
        )


def test_container_uuid_binding_maps_host_gpu01_to_container_ordinal01() -> None:
    with launcher._relocation_scope():
        nvidia, torch, binding = launcher.attestation_parent._expected_container_inventories(
            _inventory()
        )
    assert nvidia == [
        {"index": 0, "uuid": "GPU-0", "name": "RTX"},
        {"index": 1, "uuid": "GPU-1", "name": "RTX"},
    ]
    assert torch == [
        {"ordinal": 0, "uuid": "GPU-0", "name": "RTX"},
        {"ordinal": 1, "uuid": "GPU-1", "name": "RTX"},
    ]
    assert [value["physical_index"] for value in binding] == [0, 1]


def test_dry_run_exposes_gpu01_but_preserves_frozen_18_job_lane_labels() -> None:
    old_request = launcher.engine.DOCKER_GPU_REQUEST
    old_name = launcher.engine.CONTAINER_NAME
    old_intent = launcher.engine.INTENT_PATH
    payload = launcher.dry_run_payload()
    assert payload["dry_run"] is True
    assert payload["filesystem_mutated"] is False
    assert payload["docker_gpu_request"] == '"device=0,1"'
    assert payload["nvidia_visible_devices"] == "0,1"
    assert payload["host_physical_gpus"] == [0, 1]
    assert payload["coordinator_physical_gpus"] == [2, 3]
    assert payload["frozen_coordinator_lane_labels"] == [2, 3]
    assert payload["legacy_lane_to_host_physical_gpu"] == {"2": 0, "3": 1}
    assert payload["formal_container_spec"]["gpu_device_ids"] == ["0", "1"]
    assert payload["container_name"] == launcher.CONTAINER_NAME
    assert launcher.engine.DOCKER_GPU_REQUEST == old_request
    assert launcher.engine.CONTAINER_NAME == old_name
    assert launcher.engine.INTENT_PATH == old_intent


def test_all_container_commands_request_exact_gpu01_without_cuda_visible_devices() -> None:
    for command in (
        launcher.build_verify_command(),
        launcher.build_probe_command(),
        launcher.build_register_command(),
        launcher.build_formal_command("0" * 64),
    ):
        assert command[command.index("--gpus") + 1] == '"device=0,1"'
        joined = "\n".join(command)
        assert "NVIDIA_VISIBLE_DEVICES=0,1" in joined
        assert "CUDA_VISIBLE_DEVICES=" not in joined
    formal = launcher.build_formal_command("0" * 64)
    assert formal[formal.index("--name") + 1] == launcher.CONTAINER_NAME


def test_relocation_registration_state_is_valid_before_and_after_write_once() -> None:
    status = registrar.verify_registration()
    assert status["verified"] is True
    payload = registrar._load_json(registrar.REGISTRATION) if status["registered"] else status["candidate"]
    execution = payload["execution_scope"]
    authorization = payload["authorization"]
    assert execution["host_physical_gpus"] == [0, 1]
    assert execution["frozen_coordinator_lane_labels"] == [2, 3]
    assert execution["legacy_lane_to_host_physical_gpu"] == {"2": 0, "3": 1}
    assert execution["coordinator_matrix_unchanged"] is True
    assert execution["total_jobs"] == 18
    assert authorization["tier2s_v2_source_only_diagnostic_authorized"] is True
    assert authorization["formal_v3_model_training_authorized"] is False
    assert authorization["outer_target_access_authorized"] is False


def test_direct_file_entrypoint_succeeds_without_inherited_pythonpath() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-B", str(launcher.Path(__file__).resolve().parents[1] / "scripts" / "launch_tier2s_factorized_isolated_v2_device_relocation_gpu01_v1.py")],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["host_physical_gpus"] == [0, 1]
    assert payload["formal_container_spec"]["gpu_device_ids"] == ["0", "1"]
    assert payload["execute_with"].endswith(
        "launch_tier2s_factorized_isolated_v2_device_relocation_gpu01_v1.py --execute"
    )
