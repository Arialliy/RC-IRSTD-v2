from __future__ import annotations

import json
from pathlib import Path
import stat
import subprocess
from typing import Any

import pytest

from scripts import launch_tier2s_factorized_isolated as launcher


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def _artifact_binding(path: Path, digit: str) -> dict[str, str]:
    return {
        "path": path.as_posix(),
        "sha256": digit * 64,
        "sidecar_path": f"{path.as_posix()}.sha256",
        "sidecar_sha256": digit.upper().lower() * 64,
    }


def _governance_binding() -> dict[str, object]:
    return {
        "schema_version": "rc-irstd-aaai27-tier2s-governance-binding-v1",
        "registration": _artifact_binding(
            launcher.GOVERNANCE_REGISTRATION_RELATIVE, "1"
        ),
        "contract": _artifact_binding(
            launcher.GOVERNANCE_CONTRACT_RELATIVE, "2"
        ),
        "fresh_seed_ledger": _artifact_binding(
            launcher.FRESH_SEED_LEDGER_RELATIVE, "3"
        ),
        "fresh_seed_local_scan": _artifact_binding(
            launcher.FRESH_SEED_LOCAL_SCAN_RELATIVE, "4"
        ),
        "code_sha256_canonical_sha256": "5" * 64,
        "tier2s_source_only_diagnostic_authorized": True,
        "formal_v3_model_training_authorized": False,
        "source_gate_a_authorized": False,
        "riskcurve_authorized": False,
        "outer_target_access_authorized": False,
    }


def _preregistration_binding() -> dict[str, object]:
    relative = launcher.PREREGISTRATION_PATH.relative_to(
        launcher.PROJECT_ROOT
    )
    governance = _governance_binding()["registration"]
    assert isinstance(governance, dict)
    return {
        "schema_version": (
            "rc-irstd-aaai27-tier2s-preregistration-binding-v1"
        ),
        "protocol_id": launcher.PROTOCOL_ID,
        "path": relative.as_posix(),
        "sha256": "6" * 64,
        "sidecar_path": f"{relative.as_posix()}.sha256",
        "sidecar_sha256": "7" * 64,
        "governance_registration_sha256": governance["sha256"],
    }


def _host_inventory() -> list[dict[str, object]]:
    return [
        {"index": index, "uuid": f"GPU-{index}", "name": "RTX"}
        for index in range(4)
    ]


def _schedule() -> list[dict[str, object]]:
    schedule: list[dict[str, object]] = []
    for gpu in launcher.TARGET_GPU_INDICES:
        for queue_index in range(9):
            schedule.append(
                {
                    "run_id": f"gpu{gpu}-q{queue_index}",
                    "physical_gpu": gpu,
                    "queue_index": queue_index,
                }
            )
    return schedule


def _coordinator_payload(*, registered: bool = False) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": launcher.COORDINATOR_SCHEMA,
        "verified": True,
        "protocol_id": launcher.PROTOCOL_ID,
        "research_mode": launcher.RESEARCH_MODE,
        "source_only": True,
        "outer_target_access_authorized": False,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "source_tier3_authorized": False,
        "paper_claim_authorized": False,
        "governance_binding": _governance_binding(),
        "schedule": _schedule(),
        "lane_lengths": {"0": 9, "1": 9},
    }
    if registered:
        payload.update(
            {
                "registered": True,
                "preregistration_sha256": "6" * 64,
                "queue_manifest_sha256": "2" * 64,
                "target_lock_sha256": "3" * 64,
                "tier2s_preregistration_binding": (
                    _preregistration_binding()
                ),
            }
        )
    return payload


def _probe_payload() -> dict[str, object]:
    inventory = _host_inventory()[:2]
    mounts = {}
    for path, writable, filesystem in (
        (launcher.PROJECT_ROOT, False, "ext4"),
        (launcher.OUTPUT_ROOT, True, "ext4"),
        (launcher.AUDIT_ROOT, True, "ext4"),
        (launcher.FORBIDDEN_TARGET, True, "tmpfs"),
    ):
        mounts[str(path)] = {
            "mount_options": ["rw" if writable else "ro"],
            "filesystem_type": filesystem,
            "mount_source": "tmpfs" if filesystem == "tmpfs" else "/dev/mock",
            "super_options": ["rw" if writable else "ro"],
        }
    return {
        "schema_version": launcher.PROBE_SCHEMA,
        "protocol_id": launcher.PROTOCOL_ID,
        "research_mode": launcher.RESEARCH_MODE,
        "home_ly_entries": [launcher.PROJECT_ROOT.name],
        "target": {
            "path": str(launcher.FORBIDDEN_TARGET),
            "mode": 0,
            "list_error": "PermissionError",
        },
        "mounts": mounts,
        "nvidia_inventory": inventory,
        "torch_inventory": [
            {
                "ordinal": value["index"],
                "uuid": value["uuid"],
                "name": value["name"],
            }
            for value in inventory
        ],
        "environment": {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": None,
            "NVIDIA_VISIBLE_DEVICES": launcher.NVIDIA_VISIBLE_DEVICES,
        },
        "runtime": {
            "python": "3.11.12",
            "torch": "2.7.0+cu128",
            "torch_cuda": "12.8",
            "torchvision": "0.22.0+cu128",
            "numpy": "1.26.4",
            "scipy": "1.17.1",
            "PIL": "11.3.0",
            "skimage": "0.26.0",
            "yaml": "6.0.2",
            "pandas": "3.0.2",
            "tqdm": "4.67.1",
        },
    }


def test_all_three_container_modes_share_the_fail_closed_contract() -> None:
    commands = (
        launcher.build_verify_command(),
        launcher.build_register_command(),
        launcher.build_formal_command("a" * 64),
    )
    for command in commands:
        assert launcher.IMAGE_ID in command
        assert _option_value(command, "--network") == "none"
        assert "--read-only" in command
        assert "--init" in command
        assert _option_value(command, "--gpus") == launcher.DOCKER_GPU_REQUEST
        assert "all" != _option_value(command, "--gpus")
        assert _option_value(command, "--entrypoint") == launcher.CONTAINER_PYTHON
        assert _option_value(command, "--cap-drop") == "ALL"
        assert _option_value(command, "--security-opt") == "no-new-privileges:true"
        assert "CUDA_VISIBLE_DEVICES=" not in " ".join(command)
        assert (
            f"NVIDIA_VISIBLE_DEVICES={launcher.NVIDIA_VISIBLE_DEVICES}"
            in command
        )
    assert "--verify-only" in commands[0] and "--rm" in commands[0]
    assert "--register-only" in commands[1] and "--rm" in commands[1]
    assert _option_value(commands[2], "--name") == launcher.CONTAINER_NAME
    assert _option_value(commands[2], "--restart") == "no"
    assert launcher.CONTAINER_NAME != launcher.LEGACY_CONTAINER_NAME
    formal_spec = launcher.container_spec(formal=True)
    assert formal_spec["restart"] == "no"
    assert formal_spec["gpu_exposure"] == "device=0,1"
    assert formal_spec["gpu_device_ids"] == ["0", "1"]


def test_all_gpu_docker_request_is_rejected() -> None:
    command = launcher.build_verify_command()
    command[command.index("--gpus") + 1] = "all"
    with pytest.raises(RuntimeError, match="never all"):
        launcher._assert_exact_gpu_command(command)


def test_tier2s_scope_does_not_mutate_historical_base_launcher() -> None:
    launcher.build_formal_command("b" * 64)
    assert launcher.base.TARGET_GPU_INDICES == (2, 3)
    assert launcher.base.ENVIRONMENT["NVIDIA_VISIBLE_DEVICES"] == "all"
    historical = launcher.base.build_formal_command("b" * 64)
    assert _option_value(historical, "--gpus") == "all"
    assert _option_value(historical, "--restart") == "on-failure"


def test_only_new_tier2s_output_and_audit_are_writable() -> None:
    mounts = launcher.bind_mount_contract()
    writable = {
        Path(value["destination"])
        for value in mounts
        if not value["readonly"]
    }
    assert writable == {launcher.OUTPUT_ROOT, launcher.AUDIT_ROOT}
    assert launcher.LEGACY_OUTPUT_ROOT not in writable
    assert launcher.LEGACY_AUDIT_ROOT not in writable
    assert all(value["source"] != str(launcher.FORBIDDEN_TARGET) for value in mounts)
    target = {
        value["destination"]: value for value in launcher.tmpfs_contract()
    }[str(launcher.FORBIDDEN_TARGET)]
    assert "mode=000" in target["options"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("research_mode", "confirmatory"),
        ("source_only", False),
        ("outer_target_access_authorized", True),
        ("source_tier3_authorized", True),
        ("paper_claim_authorized", True),
    ],
)
def test_verify_contract_fails_closed_on_authority_drift(
    field: str, value: object
) -> None:
    payload = _coordinator_payload()
    payload[field] = value
    with pytest.raises(RuntimeError, match="contract drift"):
        launcher._coordinator_contract(payload)


@pytest.mark.parametrize(
    "case",
    ("gpu2", "lane_count", "queue_missing", "queue_wrong_order"),
)
def test_verify_contract_requires_exact_two_lane_nine_by_nine_fifo(
    case: str,
) -> None:
    payload = _coordinator_payload()
    schedule = payload["schedule"]
    assert isinstance(schedule, list)
    if case == "gpu2":
        schedule[0]["physical_gpu"] = 2
    elif case == "lane_count":
        payload["lane_lengths"] = {"0": 8, "1": 10}
    elif case == "queue_missing":
        schedule[0]["queue_index"] = None
    else:
        schedule[0]["queue_index"] = 1
    with pytest.raises(RuntimeError, match="contract drift|FIFO queue"):
        launcher._coordinator_contract(payload)


def test_verify_contract_requires_exact_governance_binding() -> None:
    payload = _coordinator_payload()
    governance = payload["governance_binding"]
    assert isinstance(governance, dict)
    governance["outer_target_access_authorized"] = True
    with pytest.raises(RuntimeError, match="governance authorization"):
        launcher._coordinator_contract(payload)


@pytest.mark.parametrize(
    "case",
    (
        "nvidia_missing",
        "nvidia_extra",
        "nvidia_wrong_order",
        "torch_missing",
        "torch_extra",
        "torch_wrong_order",
        "nvidia_visible_all",
    ),
)
def test_attestation_rejects_missing_extra_wrong_order_or_all(
    case: str,
) -> None:
    payload = json.loads(json.dumps(_probe_payload()))
    nvidia = payload["nvidia_inventory"]
    torch_inventory = payload["torch_inventory"]
    environment = payload["environment"]
    assert isinstance(nvidia, list)
    assert isinstance(torch_inventory, list)
    assert isinstance(environment, dict)
    if case == "nvidia_missing":
        nvidia.pop()
    elif case == "nvidia_extra":
        nvidia.append({"index": 2, "uuid": "GPU-2", "name": "RTX"})
    elif case == "nvidia_wrong_order":
        nvidia.reverse()
    elif case == "torch_missing":
        torch_inventory.pop()
    elif case == "torch_extra":
        torch_inventory.append(
            {"ordinal": 2, "uuid": "GPU-2", "name": "RTX"}
        )
    elif case == "torch_wrong_order":
        torch_inventory.reverse()
    else:
        environment["NVIDIA_VISIBLE_DEVICES"] = "all"

    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    with pytest.raises(RuntimeError, match="attestation failed"):
        launcher._verify_container_attestation(
            _host_inventory(), runner=runner
        )


def test_attestation_accepts_only_exact_gpu_zero_one_mapping() -> None:
    payload = _probe_payload()

    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    attestation = launcher._verify_container_attestation(
        _host_inventory(), runner=runner
    )
    assert attestation["physical_gpu_allowlist_verified"] == [0, 1]
    assert attestation["torch_ordinal_mapping_verified"] is True


def test_selected_gpu_idle_preflight_rejects_busy_compute_process() -> None:
    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if any(value.startswith("--query-gpu=") for value in command):
            return subprocess.CompletedProcess(
                command, 0, "0, GPU-0, 0, 0\n1, GPU-1, 0, 0\n", ""
            )
        return subprocess.CompletedProcess(
            command, 0, "GPU-0, 123, python, 1024\n", ""
        )

    with pytest.raises(RuntimeError, match="must be idle"):
        launcher._verify_selected_gpus_idle(
            _host_inventory(), runner=runner
        )


@pytest.mark.parametrize(
    "rows",
    (
        "0, GPU-0, 0, 0\n",
        "1, GPU-1, 0, 0\n0, GPU-0, 0, 0\n",
        "0, GPU-0, 1, 0\n1, GPU-1, 0, 0\n",
    ),
)
def test_selected_gpu_idle_preflight_rejects_missing_wrong_order_or_load(
    rows: str,
) -> None:
    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if any(value.startswith("--query-gpu=") for value in command):
            return subprocess.CompletedProcess(command, 0, rows, "")
        return subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(RuntimeError, match="idle query|must be idle"):
        launcher._verify_selected_gpus_idle(
            _host_inventory(), runner=runner
        )


def test_selected_gpu_idle_preflight_records_transient_utilization() -> None:
    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if any(value.startswith("--query-gpu=") for value in command):
            return subprocess.CompletedProcess(
                command, 0, "0, GPU-0, 0, 2\n1, GPU-1, 0, 0\n", ""
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    evidence = launcher._verify_selected_gpus_idle(
        _host_inventory(), runner=runner
    )
    assert evidence["verified"] is True
    assert evidence["stats"][0]["utilization_gpu_percent"] == 2


def test_execute_orders_verify_probe_register_then_formal_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    intents: list[dict[str, Any]] = []
    monkeypatch.setattr(launcher, "_validate_host_sources", lambda: None)
    monkeypatch.setattr(launcher, "_ensure_exact_rw_roots", lambda: None)
    def register_intent(payload: dict[str, Any]) -> str:
        intents.append(payload)
        return "a" * 64

    monkeypatch.setattr(launcher.base, "_register_launch_intent", register_intent)

    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, launcher.IMAGE_ID + "\n", "")
        if any(value.startswith("--query-compute-apps=") for value in command):
            return subprocess.CompletedProcess(command, 0, "", "")
        if (
            "--query-gpu=index,uuid,memory.used,utilization.gpu"
            in command
        ):
            return subprocess.CompletedProcess(
                command,
                0,
                "0, GPU-0, 0, 0\n1, GPU-1, 0, 0\n",
                "",
            )
        if command[0] == "nvidia-smi":
            rows = "\n".join(
                f"{index}, GPU-{index}, RTX" for index in range(4)
            ) + "\n"
            return subprocess.CompletedProcess(command, 0, rows, "")
        if "--verify-only" in command:
            return subprocess.CompletedProcess(
                command, 0, json.dumps(_coordinator_payload()), ""
            )
        if str(launcher.CONTAINER_PROBE) in command:
            return subprocess.CompletedProcess(
                command, 0, json.dumps(_probe_payload()), ""
            )
        if "--register-only" in command:
            return subprocess.CompletedProcess(
                command, 0, json.dumps(_coordinator_payload(registered=True)), ""
            )
        if command[:3] == ["docker", "container", "inspect"]:
            return subprocess.CompletedProcess(command, 1, "", "No such container")
        if command[:2] == ["docker", "run"] and "--detach" in command:
            return subprocess.CompletedProcess(command, 0, "container-id\n", "")
        raise AssertionError(command)

    result = launcher.execute_launch(runner=runner)
    verify_index = next(i for i, value in enumerate(calls) if "--verify-only" in value)
    probe_index = next(
        i for i, value in enumerate(calls) if str(launcher.CONTAINER_PROBE) in value
    )
    register_index = next(
        i for i, value in enumerate(calls) if "--register-only" in value
    )
    start_index = next(i for i, value in enumerate(calls) if "--detach" in value)
    assert verify_index < probe_index < register_index < start_index
    assert result["container"]["action"] == "created_and_started"
    assert len(intents) == 1
    assert intents[0]["physical_gpu_allowlist"] == [0, 1]
    assert intents[0]["governance_binding"] == _governance_binding()
    assert (
        intents[0]["tier2s_preregistration_binding"]
        == _preregistration_binding()
    )
    assert result["selected_gpu_idle_preflight"]["verified"] is True


def _existing_container(intent_sha256: str = "a" * 64) -> dict[str, Any]:
    mounts = [
        {
            "Type": "bind",
            "Source": value["source"],
            "Destination": value["destination"],
            "RW": not value["readonly"],
        }
        for value in launcher.bind_mount_contract()
    ]
    return {
        "Image": launcher.IMAGE_ID,
        "Config": {
            "User": launcher.CONTAINER_USER,
            "WorkingDir": str(launcher.PROJECT_ROOT),
            "Entrypoint": [launcher.CONTAINER_PYTHON],
            "Cmd": [str(launcher.COORDINATOR)],
            "Labels": {
                **launcher.RUNTIME_LABELS,
                launcher.INTENT_LABEL: intent_sha256,
            },
            "Env": [
                f"{key}={value}"
                for key, value in launcher.ENVIRONMENT.items()
            ],
        },
        "HostConfig": {
            "NetworkMode": "none",
            "ReadonlyRootfs": True,
            "Init": True,
            "ShmSize": 16 * 1024**3,
            "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
            "Tmpfs": {
                value["destination"]: value["options"]
                for value in launcher.tmpfs_contract()
            },
            "DeviceRequests": [
                {
                    "Driver": "",
                    "Count": 0,
                    "DeviceIDs": ["0", "1"],
                    "Capabilities": [["gpu"]],
                    "Options": {},
                }
            ],
        },
        "State": {"Status": "exited"},
        "Mounts": mounts,
    }


def test_existing_container_accepts_only_exact_devices_and_restart_no() -> None:
    assert launcher._validate_existing_container(
        _existing_container(), "a" * 64
    ) == "exited"


@pytest.mark.parametrize(
    ("case", "device_ids"),
    (
        ("missing", ["0"]),
        ("extra", ["0", "1", "2"]),
        ("wrong_order", ["1", "0"]),
        ("all", None),
    ),
)
def test_existing_container_rejects_missing_extra_wrong_order_and_all(
    case: str, device_ids: list[str] | None
) -> None:
    container = _existing_container()
    request = container["HostConfig"]["DeviceRequests"][0]
    request["DeviceIDs"] = device_ids
    if case == "all":
        request["Count"] = -1
    with pytest.raises(RuntimeError, match="exact DeviceIDs"):
        launcher._validate_existing_container(container, "a" * 64)


def test_existing_container_rejects_restart_or_visibility_drift() -> None:
    restart = _existing_container()
    restart["HostConfig"]["RestartPolicy"] = {
        "Name": "on-failure",
        "MaximumRetryCount": 0,
    }
    with pytest.raises(RuntimeError, match="restart policy"):
        launcher._validate_existing_container(restart, "a" * 64)

    visibility = _existing_container()
    visibility["Config"]["Env"].append("NVIDIA_VISIBLE_DEVICES=all")
    with pytest.raises(RuntimeError, match="NVIDIA_VISIBLE_DEVICES"):
        launcher._validate_existing_container(visibility, "a" * 64)


def test_launch_intent_registration_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent = tmp_path / "LAUNCH_INTENT.json"
    sidecar = tmp_path / "LAUNCH_INTENT.json.sha256"
    with launcher._base_scope():
        monkeypatch.setattr(launcher.base, "INTENT_PATH", intent)
        monkeypatch.setattr(launcher.base, "INTENT_SHA256_PATH", sidecar)
        payload = {"schema_version": launcher.SCHEMA, "registered_at": "fixed"}
        digest = launcher.base._register_launch_intent(payload)
        assert launcher.base._register_launch_intent(payload) == digest
    assert stat.S_IMODE(intent.stat().st_mode) == 0o444
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o444
