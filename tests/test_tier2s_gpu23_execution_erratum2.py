from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum1 as parent_launcher
from scripts import launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum2 as launcher
from scripts import register_tier2s_gpu23_execution_erratum2 as registrar


ROOT = Path(__file__).resolve().parents[1]


def _host_inventory() -> list[dict[str, object]]:
    return [
        {"index": 0, "uuid": "GPU-0", "name": "RTX"},
        {"index": 1, "uuid": "GPU-1", "name": "RTX"},
        {"index": 2, "uuid": "GPU-2", "name": "RTX"},
        {"index": 3, "uuid": "GPU-3", "name": "RTX"},
    ]


def _probe_payload() -> dict[str, object]:
    return {
        "schema_version": launcher.PROBE_SCHEMA,
        "protocol_id": launcher.PROTOCOL_ID,
        "research_mode": launcher.RESEARCH_MODE,
        "home_ly_entries": [ROOT.name],
        "target": {
            "path": str(parent_launcher.FORBIDDEN_TARGET),
            "mode": 0,
            "list_error": "PermissionError",
        },
        "mounts": {
            str(ROOT): {
                "filesystem_type": "ext4",
                "mount_options": ["ro", "relatime"],
            },
            str(parent_launcher.OUTPUT_ROOT): {
                "filesystem_type": "ext4",
                "mount_options": ["rw", "relatime"],
            },
            str(parent_launcher.AUDIT_ROOT): {
                "filesystem_type": "ext4",
                "mount_options": ["rw", "relatime"],
            },
            str(parent_launcher.FORBIDDEN_TARGET): {
                "filesystem_type": "tmpfs",
                "mount_options": ["rw", "nosuid", "nodev", "noexec"],
            },
        },
        "nvidia_inventory": [
            {"index": 0, "uuid": "GPU-2", "name": "RTX"},
            {"index": 1, "uuid": "GPU-3", "name": "RTX"},
        ],
        "torch_inventory": [
            {"ordinal": 0, "uuid": "GPU-2", "name": "RTX"},
            {"ordinal": 1, "uuid": "GPU-3", "name": "RTX"},
        ],
        "environment": {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": None,
            "NVIDIA_VISIBLE_DEVICES": "2,3",
        },
        "runtime": {
            "python": "3.11.12",
            "torch": "2.7.0+cu128",
            "torch_cuda": "12.8",
            "numpy": "1.26.4",
            "scipy": "1.17.1",
            "yaml": "6.0.2",
            "pandas": "3.0.2",
            "tqdm": "4.67.1",
            "torchvision": "0.22.0+cu128",
            "PIL": "11.0.0",
            "skimage": "0.26.0",
        },
    }


def test_erratum2_preserves_exact_frozen_parent_chain() -> None:
    parent = registrar.parent_erratum1.require_frozen_execution_erratum(
        expected_registration_sha256=registrar.PARENT_REGISTRATION_SHA256
    )
    assert parent["physical_gpus"] == [2, 3]
    assert parent["physical_to_container_ordinal"] == {"2": 0, "3": 1}
    assert registrar._sha256(registrar.PARENT_CONFIG) == registrar.PARENT_CONFIG_SHA256
    assert registrar._sha256(registrar.PARENT_LAUNCHER) == registrar.PARENT_LAUNCHER_SHA256


def test_erratum2_scope_is_only_container_index_namespace() -> None:
    payload = json.loads(registrar.ERRATUM_CONFIG.read_text(encoding="utf-8"))
    assert payload["parent_execution_erratum1"]["registration_sha256"] == (
        registrar.PARENT_REGISTRATION_SHA256
    )
    assert payload["append_only_policy"]["replacement_scope"] == (
        "container_nvidia_index_namespace_attestation_only"
    )
    assert payload["append_only_policy"]["parent_execution_erratum1_mutated"] is False
    assert payload["unchanged_scientific_and_hardware_contract"]["total_jobs"] == 18
    assert payload["unchanged_scientific_and_hardware_contract"][
        "formal_v3_model_training_authorized"
    ] is False


def test_container_local_indices_uuid_bind_exact_host_physical_gpu2_and_gpu3() -> None:
    attestation = launcher._verify_attestation_payload(
        _host_inventory(), _probe_payload()
    )
    assert attestation["verified"] is True
    assert attestation["physical_gpu_allowlist_verified"] == [2, 3]
    assert attestation["container_nvidia_indices_verified"] == [0, 1]
    assert [
        row["physical_index"]
        for row in attestation["physical_to_container_uuid_binding"]
    ] == [2, 3]
    assert [
        row["container_nvidia_index"]
        for row in attestation["physical_to_container_uuid_binding"]
    ] == [0, 1]
    assert [
        row["uuid"]
        for row in attestation["physical_to_container_uuid_binding"]
    ] == ["GPU-2", "GPU-3"]


@pytest.mark.parametrize(
    "nvidia_inventory",
    (
        [
            {"index": 2, "uuid": "GPU-2", "name": "RTX"},
            {"index": 3, "uuid": "GPU-3", "name": "RTX"},
        ],
        [
            {"index": 0, "uuid": "GPU-3", "name": "RTX"},
            {"index": 1, "uuid": "GPU-2", "name": "RTX"},
        ],
        [
            {"index": 0, "uuid": "GPU-2", "name": "RTX"},
            {"index": 1, "uuid": "GPU-3", "name": "RTX"},
            {"index": 2, "uuid": "GPU-X", "name": "RTX"},
        ],
    ),
)
def test_attestation_rejects_physical_indices_wrong_uuid_order_or_extra_device(
    nvidia_inventory: list[dict[str, object]],
) -> None:
    payload = _probe_payload()
    payload["nvidia_inventory"] = nvidia_inventory
    with pytest.raises(RuntimeError, match="NVIDIA indices 0/1"):
        launcher._verify_attestation_payload(_host_inventory(), payload)


def test_attestation_rejects_wrong_torch_uuid_even_when_nvidia_is_exact() -> None:
    payload = _probe_payload()
    payload["torch_inventory"] = [
        {"ordinal": 0, "uuid": "GPU-2", "name": "RTX"},
        {"ordinal": 1, "uuid": "GPU-X", "name": "RTX"},
    ]
    with pytest.raises(RuntimeError, match="Torch ordinals 0/1"):
        launcher._verify_attestation_payload(_host_inventory(), payload)


def test_attestation_input_is_not_mutated() -> None:
    payload = _probe_payload()
    original = deepcopy(payload)
    launcher._verify_attestation_payload(_host_inventory(), payload)
    assert payload == original


def test_erratum2_registration_state_is_valid_before_and_after_write_once() -> None:
    status = registrar.verify_registration()
    assert status["verified"] is True
    payload = (
        registrar._load_json(registrar.REGISTRATION)
        if status["registered"]
        else status["candidate"]
    )
    execution = payload["execution_scope"]
    authorization = payload["authorization"]
    assert execution["physical_gpus"] == [2, 3]
    assert execution["container_nvidia_indices"] == [0, 1]
    assert execution["torch_ordinals"] == [0, 1]
    assert execution["total_jobs"] == 18
    assert authorization["tier2s_v2_source_only_diagnostic_authorized"] is True
    assert authorization["formal_v3_model_training_authorized"] is False
    assert authorization["outer_target_access_authorized"] is False


def test_erratum2_dry_run_uses_new_names_and_restores_parent_globals() -> None:
    old_name = parent_launcher.CONTAINER_NAME
    old_schema = parent_launcher.SCHEMA
    payload = launcher.dry_run_payload()
    assert payload["filesystem_mutated"] is False
    assert payload["docker_invoked"] is False
    assert payload["container_name"] == launcher.CONTAINER_NAME
    assert payload["schema_version"] == launcher.SCHEMA
    assert payload["docker_gpu_request"] == '"device=2,3"'
    assert payload["coordinator_physical_gpus"] == [2, 3]
    assert payload["formal_container_spec"]["gpu_device_ids"] == ["2", "3"]
    assert payload["execution_erratum_config"] == str(
        launcher.EXECUTION_ERRATUM_CONFIG
    )
    assert payload["execution_erratum_registration"] == str(
        launcher.REGISTRATION
    )
    assert payload["execute_with"].endswith(
        "launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum2.py --execute"
    )
    assert parent_launcher.CONTAINER_NAME == old_name
    assert parent_launcher.SCHEMA == old_schema


def test_erratum2_formal_command_keeps_exact_gpu_device_request() -> None:
    command = launcher.build_formal_command("0" * 64)
    position = command.index("--gpus")
    assert command[position + 1] == '"device=2,3"'
    assert "--name" in command
    assert command[command.index("--name") + 1] == launcher.CONTAINER_NAME
    assert "CUDA_VISIBLE_DEVICES=" not in "\n".join(command)

