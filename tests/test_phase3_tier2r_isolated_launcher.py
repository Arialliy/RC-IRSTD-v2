from __future__ import annotations

import json
from pathlib import Path
import stat
import subprocess

import pytest

from scripts import launch_phase3_tier2r_isolated as launcher


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]

def _probe_payload() -> dict[str, object]:
    inventory = [{"index": i, "uuid": f"GPU-{i}", "name": "RTX"} for i in range(4)]
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
        "home_ly_entries": [launcher.PROJECT_ROOT.name],
        "target": {
            "path": str(launcher.FORBIDDEN_TARGET),
            "mode": 0,
            "list_error": "PermissionError",
        },
        "mounts": mounts,
        "nvidia_inventory": inventory,
        "torch_inventory": [
            {"ordinal": value["index"], "uuid": value["uuid"], "name": value["name"]}
            for value in inventory
        ],
        "environment": {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": None,
            "NVIDIA_VISIBLE_DEVICES": "all",
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



def test_docker_contract_is_pinned_and_source_only() -> None:
    verify = launcher.build_verify_command()
    formal = launcher.build_formal_command("a" * 64)

    for command in (verify, formal):
        assert launcher.IMAGE_ID in command
        assert _option_value(command, "--network") == "none"
        assert "--read-only" in command
        assert "--init" in command
        assert _option_value(command, "--shm-size") == "16g"
        assert _option_value(command, "--user") == "1004:1004"
        assert _option_value(command, "--gpus") == "all"
        assert not any(value.startswith("CUDA_VISIBLE_DEVICES=") for value in command)
        assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in command
        assert _option_value(command, "--entrypoint") == launcher.CONTAINER_PYTHON

    assert "--rm" in verify and "--verify-only" in verify
    assert "--restart" not in verify
    assert _option_value(formal, "--name") == launcher.CONTAINER_NAME
    assert _option_value(formal, "--restart") == "on-failure"


def test_only_formal_output_and_audit_are_host_writable() -> None:
    mounts = launcher.bind_mount_contract()
    assert len(mounts) == 3
    assert all(
        Path(value["source"]).is_relative_to(launcher.PROJECT_ROOT) for value in mounts
    )
    writable = {value["destination"] for value in mounts if not value["readonly"]}
    assert writable == {str(launcher.OUTPUT_ROOT), str(launcher.AUDIT_ROOT)}
    assert all(value["source"] != str(launcher.FORBIDDEN_TARGET) for value in mounts)
    target = {value["destination"]: value for value in launcher.tmpfs_contract()}[
        str(launcher.FORBIDDEN_TARGET)
    ]
    assert "mode=000" in target["options"]


def test_execute_verifies_before_formal_start(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(launcher, "_validate_host_sources", lambda: None)
    monkeypatch.setattr(launcher, "_ensure_exact_rw_roots", lambda: None)
    monkeypatch.setattr(launcher, "_register_launch_intent", lambda payload: "a" * 64)

    schedule = [{"physical_gpu": 2 if i % 2 == 0 else 3} for i in range(18)]

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, launcher.IMAGE_ID + "\n", "")
        if command[0] == "nvidia-smi":
            rows = "\n".join(f"{i}, GPU-{i}, RTX" for i in range(4)) + "\n"
            return subprocess.CompletedProcess(command, 0, rows, "")
        if "--verify-only" in command:
            payload = {"verified": True, "source_only": True, "outer_target_access_authorized": False, "schedule": schedule}
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        if str(launcher.CONTAINER_PROBE) in command:
            return subprocess.CompletedProcess(command, 0, json.dumps(_probe_payload()), "")
        if command[:3] == ["docker", "container", "inspect"]:
            return subprocess.CompletedProcess(command, 1, "", "No such container")
        if command[:2] == ["docker", "run"] and "--detach" in command:
            return subprocess.CompletedProcess(command, 0, "container-id\n", "")
        raise AssertionError(command)

    result = launcher.execute_launch(runner=runner)
    verify_index = next(i for i, value in enumerate(calls) if "--verify-only" in value)
    probe_index = next(i for i, value in enumerate(calls) if str(launcher.CONTAINER_PROBE) in value)
    start_index = next(i for i, value in enumerate(calls) if "--detach" in value)
    assert verify_index < probe_index < start_index
    assert result["container"]["action"] == "created_and_started"


def test_running_matching_container_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    digest = "b" * 64
    mounts = [
        {"Type": "bind", "Source": value["source"], "Destination": value["destination"], "RW": not value["readonly"]}
        for value in launcher.bind_mount_contract()
    ]
    inspect = {
        "Id": "existing-id",
        "Image": launcher.IMAGE_ID,
        "Config": {
            "User": launcher.CONTAINER_USER,
            "WorkingDir": str(launcher.PROJECT_ROOT),
            "Entrypoint": [launcher.CONTAINER_PYTHON],
            "Cmd": [str(launcher.COORDINATOR)],
            "Env": [f"{k}={v}" for k, v in launcher.ENVIRONMENT.items()],
            "Labels": {**launcher.RUNTIME_LABELS, launcher.INTENT_LABEL: digest},
        },
        "HostConfig": {
            "NetworkMode": "none", "ReadonlyRootfs": True, "Init": True,
            "ShmSize": 16 * 1024**3, "RestartPolicy": {"Name": "on-failure"},
            "Tmpfs": {v["destination"]: v["options"] for v in launcher.tmpfs_contract()},
            "DeviceRequests": [{"Count": -1, "Capabilities": [["gpu"]]}],
        },
        "State": {"Status": "running"},
        "Mounts": mounts,
    }

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[:3] == ["docker", "container", "inspect"]
        return subprocess.CompletedProcess(command, 0, json.dumps([inspect]), "")

    result = launcher._start_or_reconcile(digest, runner=runner)
    assert result == {"action": "already_active", "container_id": "existing-id", "status": "running"}


def test_launch_intent_is_write_once_and_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    intent = tmp_path / "LAUNCH_INTENT.json"
    sidecar = tmp_path / "LAUNCH_INTENT.json.sha256"
    monkeypatch.setattr(launcher, "INTENT_PATH", intent)
    monkeypatch.setattr(launcher, "INTENT_SHA256_PATH", sidecar)
    payload = {"schema_version": launcher.SCHEMA, "registered_at": "fixed"}
    digest = launcher._register_launch_intent(payload)
    assert launcher._register_launch_intent(payload) == digest
    assert stat.S_IMODE(intent.stat().st_mode) == 0o444
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o444
    with pytest.raises(RuntimeError, match="drift"):
        launcher._register_launch_intent({**payload, "changed": True})
