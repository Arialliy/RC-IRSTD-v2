from __future__ import annotations

import ast
import copy
import hashlib
import json
import stat
from pathlib import Path

import pytest

from scripts import register_phase3_tier2r_startup_fix1 as registrar


def _inventory() -> list[dict[str, object]]:
    return [
        {"index": index, "name": "RTX", "uuid": f"GPU-{index}"}
        for index in range(4)
    ]


def _container_contract_fixture() -> tuple[dict, dict, list[dict[str, object]]]:
    inventory = _inventory()
    project = "/home/ly/RC-IRSTD-v2"
    output = project + "/outputs/formal"
    audit = project + "/artifacts/formal"
    target = project + "/datasets/NUAA-SIRST"
    mounts = [
        {"source": project, "destination": project, "readonly": True},
        {"source": output, "destination": output, "readonly": False},
        {"source": audit, "destination": audit, "readonly": False},
    ]
    tmpfs = [
        {"destination": "/tmp", "options": "rw,mode=1777,size=8g"},
        {"destination": target, "options": "rw,mode=000,size=4k"},
    ]
    environment = {"PYTHONPATH": project, "NVIDIA_VISIBLE_DEVICES": "all"}
    intent = {
        "formal_container_spec": {
            "bind_mounts": mounts,
            "tmpfs": tmpfs,
            "environment": environment,
            "working_directory": project,
            "user": "1004:1004",
            "entrypoint": "python",
            "cap_drop": ["ALL"],
            "security_options": ["no-new-privileges:true"],
        },
        "container_attestation": {
            "nvidia_inventory": inventory,
            "torch_inventory": [
                {"ordinal": record["index"], "name": record["name"], "uuid": record["uuid"]}
                for record in inventory
            ],
            "runtime": {"python": "3.11", "torch": "2.7"},
            "host_gpu_uuid_mapping_verified": True,
            "outer_target_alias_absent": True,
            "target": {"mode": 0, "list_error": "PermissionError"},
        },
        "host_gpu_inventory": inventory,
        "coordinator_assigned_host_gpus": inventory[2:4],
    }
    observed_mounts = [
        {
            "Type": "bind",
            "Source": record["source"],
            "Destination": record["destination"],
            "RW": not record["readonly"],
        }
        for record in mounts
    ]
    container = {
        "Mounts": observed_mounts,
        "HostConfig": {
            "Tmpfs": {record["destination"]: record["options"] for record in tmpfs},
            "DeviceRequests": [
                {
                    "Driver": "",
                    "Count": -1,
                    "DeviceIDs": None,
                    "Capabilities": [["gpu"]],
                    "Options": {},
                }
            ],
            "Runtime": "runc",
            "Init": True,
            "ShmSize": 16 * 1024**3,
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges:true"],
        },
        "Config": {
            "Env": [f"{key}={value}" for key, value in environment.items()],
            "WorkingDir": project,
            "User": "1004:1004",
            "Entrypoint": ["python"],
        },
    }
    return container, intent, inventory


def test_container_contract_binds_mounts_tmpfs_runtime_env_and_gpu_uuid() -> None:
    container, intent, inventory = _container_contract_fixture()
    result = registrar._validate_container_contract(container, intent, inventory)
    assert result["matches_launch_intent"] is True
    assert result["assigned_host_gpus"] == inventory[2:4]
    assert result["nuaa_tmpfs_mode"] == 0

    inspect_a = {
        "Id": "same-container",
        "Mounts": [
            {"Destination": "/output", "Source": "/host/output"},
            {"Destination": "/project", "Source": "/host/project"},
        ],
    }
    inspect_b = {
        "Id": "same-container",
        "Mounts": list(reversed(inspect_a["Mounts"])),
    }
    inspect_original = copy.deepcopy(inspect_a)
    hash_a = registrar._canonical_container_inspect_sha256(inspect_a)
    assert hash_a == registrar._canonical_container_inspect_sha256(inspect_b)
    assert inspect_a == inspect_original
    inspect_changed = copy.deepcopy(inspect_a)
    inspect_changed["Mounts"][0]["Source"] = "/different/source"
    assert hash_a != registrar._canonical_container_inspect_sha256(
        inspect_changed
    )

    for mutation in ("mount", "tmpfs", "runtime", "environment", "gpu"):
        bad_container = copy.deepcopy(container)
        bad_inventory = copy.deepcopy(inventory)
        if mutation == "mount":
            bad_container["Mounts"][0]["RW"] = True
        elif mutation == "tmpfs":
            bad_container["HostConfig"]["Tmpfs"].pop("/tmp")
        elif mutation == "runtime":
            bad_container["HostConfig"]["Runtime"] = "changed"
        elif mutation == "environment":
            bad_container["Config"]["Env"] = ["PYTHONPATH=wrong", "NVIDIA_VISIBLE_DEVICES=all"]
        else:
            bad_inventory[2]["uuid"] = "GPU-WRONG"
        with pytest.raises(RuntimeError, match="contract drift"):
            registrar._validate_container_contract(bad_container, intent, bad_inventory)


def test_write_once_is_noreplace_readonly_and_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registrar, "AUDIT_ROOT", tmp_path)
    path = tmp_path / "evidence.json"
    sidecar = tmp_path / "evidence.json.sha256"
    payload = {"schema_version": "test", "registered_at": "fixed"}

    digest = registrar._write_once(path, sidecar, payload)
    assert registrar._write_once(path, sidecar, payload) == digest
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o444
    assert sidecar.read_text() == f"{digest}  {path.name}\n"
    with pytest.raises(RuntimeError, match="immutable startup-fix evidence drift"):
        registrar._write_once(path, sidecar, {**payload, "drift": True})
    assert "os.replace" not in Path(registrar.__file__).read_text(encoding="utf-8")


def test_registration_lock_fails_closed_under_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registrar, "AUDIT_ROOT", tmp_path)
    with registrar._registration_lock():
        with pytest.raises(RuntimeError, match="another startup-fix1 registrar"):
            with registrar._registration_lock():
                pytest.fail("contending lock must not be entered")


def test_missing_sidecar_recovery_only_freezes_complete_path(tmp_path: Path) -> None:
    path = tmp_path / "startup.json"
    sidecar = tmp_path / "startup.json.sha256"
    payload = {
        "schema_version": registrar.STARTUP_SCHEMA,
        "registered_at": "fixed",
        "semantic_contract": "complete",
    }
    raw = registrar._canonical_bytes(payload)

    def validate_complete(candidate: dict) -> None:
        if candidate != payload:
            raise RuntimeError("semantic contract rejected")

    registrar._publish_noreplace(path, raw)
    assert not sidecar.exists()
    registrar._recover_missing_sidecar(path, sidecar, validate_complete)
    assert registrar._verify_sidecar(path, sidecar) == hashlib.sha256(raw).hexdigest()

    schema_only_path = tmp_path / "schema-only.json"
    schema_only_sidecar = tmp_path / "schema-only.json.sha256"
    registrar._publish_noreplace(
        schema_only_path,
        registrar._canonical_bytes(
            {"schema_version": registrar.STARTUP_SCHEMA, "registered_at": "fixed"}
        ),
    )
    with pytest.raises(RuntimeError, match="semantic contract rejected"):
        registrar._recover_missing_sidecar(
            schema_only_path, schema_only_sidecar, validate_complete
        )
    assert not schema_only_sidecar.exists()

    noncanonical_path = tmp_path / "noncanonical.json"
    noncanonical_sidecar = tmp_path / "noncanonical.json.sha256"
    registrar._publish_noreplace(
        noncanonical_path,
        (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        ),
    )
    with pytest.raises(RuntimeError, match="not canonical"):
        registrar._recover_missing_sidecar(
            noncanonical_path, noncanonical_sidecar, validate_complete
        )
    assert not noncanonical_sidecar.exists()

    orphan = tmp_path / "orphan.json.sha256"
    registrar._publish_noreplace(orphan, b"bad\n")
    with pytest.raises(RuntimeError, match="orphan startup-fix sidecar"):
        registrar._recover_missing_sidecar(
            tmp_path / "orphan.json", orphan, validate_complete
        )


def test_validation_payload_authorizes_only_exact_existing_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registrar_path = tmp_path / "registrar.py"
    registrar_path.write_text("# registrar\n", encoding="utf-8")
    startup_path = tmp_path / "STARTUP_FIX1.json"
    image_native_validation = {
        "execution_surface": "fixed_image_ephemeral_validation_container",
        "image_id": registrar.EXPECTED_IMAGE_ID,
        "python_version": "3.11.12",
        "formal_existing_container": {"unchanged": True},
    }
    startup_path.write_text(
        json.dumps(
            {
                "implementation_erratum_planning": {"normalized": "bound"},
                "image_native_validation": image_native_validation,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(registrar, "REGISTRAR_PATH", registrar_path)
    monkeypatch.setattr(registrar, "STARTUP_FIX1_PATH", startup_path)
    monkeypatch.setattr(registrar, "VALIDATION_PATH", tmp_path / "validation.json")
    monkeypatch.setattr(registrar, "_verify_intent", lambda: ({}, "a" * 64))
    monkeypatch.setattr(registrar, "_code_drift", lambda _intent: {"new_sha256": "b" * 64})
    monkeypatch.setattr(registrar, "_readonly_lock_contract", lambda: {"valid": True})
    monkeypatch.setattr(registrar, "_validation_command_evidence", lambda: {"passed": True})

    payload = registrar._validation_payload({"path": "/fixed", "sha256": "c" * 64})
    auth = payload["resume_authorization"]
    assert auth["action"] == "START_EXISTING_CONTAINER_ONLY"
    assert auth["container_id"] == registrar.EXPECTED_CONTAINER_ID
    assert auth["docker_start_only"] is True
    assert auth["container_create_authorized"] is False
    assert auth["container_recreate_authorized"] is False
    assert auth["container_remove_authorized"] is False
    assert auth["container_reconfigure_authorized"] is False
    assert auth["launcher_reexecution_authorized"] is False
    assert payload["image_native_validation"] == image_native_validation
    assert payload["checks"]["fixed_image_python31112_validation_passed"] is True
    assert (
        payload["checks"][
            "formal_existing_container_unchanged_by_ephemeral_validation"
        ]
        is True
    )
    assert payload["formal_container_action_performed_by_registrar"] is False
    assert payload["ephemeral_validation_container_action_performed"] is True
    assert payload["ephemeral_validation_container_auto_removed"] is True


def test_frozen_code_drift_is_exact_approved_transition() -> None:
    startup_sha = registrar._verify_sidecar(
        registrar.STARTUP_FIX1_PATH, registrar.STARTUP_FIX1_SIDECAR
    )
    validation_sha = registrar._verify_sidecar(
        registrar.VALIDATION_PATH, registrar.VALIDATION_SIDECAR
    )
    startup = registrar._load_object(registrar.STARTUP_FIX1_PATH)
    validation = registrar._load_object(registrar.VALIDATION_PATH)
    assert validation["startup_fix1"] == {
        "path": str(registrar.STARTUP_FIX1_PATH.resolve()),
        "sha256": startup_sha,
    }
    assert validation["code_drift"] == startup["code_drift"]
    assert validation_sha == registrar._sha256(registrar.VALIDATION_PATH)
    drift = validation["code_drift"]
    assert drift["old_sha256"] == registrar.EXPECTED_OLD_COORDINATOR_SHA256
    assert drift["new_sha256"] == registrar.EXPECTED_NEW_COORDINATOR_SHA256
    assert drift["allowed_changed_paths"] == [registrar.COORDINATOR_RELATIVE]
    assert registrar._sha256(registrar.COORDINATOR_PATH) == (
        registrar.EXPECTED_NEW_COORDINATOR_SHA256
    )
    transition = drift["canonical_transition"]
    assert drift["canonical_transition_sha256"] == hashlib.sha256(
        registrar._canonical_bytes(transition)
    ).hexdigest()


def test_command_evidence_binds_command_returncode_and_stream_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = registrar._command_evidence(
        [str(registrar.VALIDATION_PYTHON), "-c", "print('startup-fix1-ok')"]
    )
    assert record["returncode"] == 0
    assert record["command"][0] == str(registrar.VALIDATION_PYTHON)
    assert record["stdout_sha256"] == hashlib.sha256(
        record["stdout"].encode("utf-8")
    ).hexdigest()
    assert record["stderr_sha256"] == hashlib.sha256(
        record["stderr"].encode("utf-8")
    ).hexdigest()

    # The coordinator calls the frozen verifier inside the fixed image, where
    # the host virtual-environment entry point is intentionally not mounted.
    # Frozen evidence must therefore be fully checkable without probing that
    # external path; host-side registration still requires the live identity.
    assert registrar.verify_frozen_startup_fix1.__kwdefaults__ == {
        "verify_live_host_evidence": False
    }

    def stream(command: list[str], stdout: str = "", stderr: str = "") -> dict:
        return {
            "command": command,
            "returncode": 0,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        }

    argv0 = str(registrar.VALIDATION_PYTHON)
    compile_command = [
        argv0,
        "-m",
        "py_compile",
        str(registrar.REGISTRAR_PATH),
        str(registrar.COORDINATOR_PATH),
    ]
    collect_command = [
        argv0,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "--collect-only",
        "-q",
        *registrar.VALIDATION_TEST_RELATIVES,
    ]
    pytest_command = [
        argv0,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "--tb=short",
        "-q",
        *registrar.VALIDATION_TEST_RELATIVES,
    ]
    version_command = [argv0, "--version"]
    nodeids = [
        f"tests/synthetic.py::test_offline_{index}"
        for index in range(registrar.EXPECTED_VALIDATION_TEST_COUNT)
    ]
    collect_stdout = "\n".join(
        [
            *nodeids,
            f"{registrar.EXPECTED_VALIDATION_TEST_COUNT} tests collected in 0.01s",
        ]
    ) + "\n"
    pytest_stdout = (
        "." * registrar.EXPECTED_VALIDATION_TEST_COUNT
        + " [100%]\n"
        + f"{registrar.EXPECTED_VALIDATION_TEST_COUNT} passed in 0.01s\n"
    )
    commands = {
        "surface": "sanitized_host_unit_tests_only",
        "expected_test_count": registrar.EXPECTED_VALIDATION_TEST_COUNT,
        "collected_test_count": registrar.EXPECTED_VALIDATION_TEST_COUNT,
        "result_regex": registrar.EXPECTED_PYTEST_STDOUT_RE,
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
            "path": argv0,
            "resolved_path": str(
                registrar.EXPECTED_HOST_VALIDATION_RESOLVED_PYTHON
            ),
            "resolved_binary_sha256": (
                registrar.EXPECTED_HOST_VALIDATION_PYTHON_SHA256
            ),
            "scope": "sanitized_host_unit_tests_only",
            "version": registrar.EXPECTED_HOST_VALIDATION_PYTHON_VERSION,
            "version_command": stream(
                version_command,
                registrar.EXPECTED_HOST_VALIDATION_PYTHON_VERSION + "\n",
            ),
        },
        "test_files": {
            relative: {
                "path": str((registrar.PROJECT_ROOT / relative).resolve()),
                "sha256": registrar._sha256(registrar.PROJECT_ROOT / relative),
            }
            for relative in registrar.VALIDATION_TEST_RELATIVES
        },
        "py_compile": stream(compile_command),
        "pytest_collect": stream(collect_command, collect_stdout),
        "pytest": stream(pytest_command, pytest_stdout),
    }
    monkeypatch.setattr(registrar.os, "access", lambda *_args: False)
    registrar._verify_host_validation_evidence(
        commands, verify_live_host_evidence=False
    )
    with pytest.raises(RuntimeError, match="interpreter is unavailable"):
        registrar._verify_host_validation_evidence(
            commands, verify_live_host_evidence=True
        )


def test_registrar_ephemeral_docker_run_is_isolated_and_nonformal() -> None:
    script = "print('image-native-validation')"
    command = registrar._image_native_command(script)
    assert command[:2] == ["docker", "run"]
    assert "--rm" in command
    assert command[command.index("--name") + 1] == registrar.TRANSIENT_VALIDATION_CONTAINER
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert command[command.index("--runtime") + 1] == "runc"
    assert command[-3:] == [registrar.EXPECTED_IMAGE_ID, "-c", script]

    mount_values = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--mount"
    ]
    assert mount_values == [
        (
            f"type=bind,source={registrar.PROJECT_ROOT},"
            f"target={registrar.PROJECT_ROOT},readonly"
        )
    ]
    assert command.count(str(registrar.EXPECTED_IMAGE_ID)) == 1
    assert registrar.CONTAINER_NAME not in command
    for forbidden in ("--gpus", "--device", "--restart"):
        assert not any(
            value == forbidden or value.startswith(forbidden + "=")
            for value in command
        )

    source = Path(registrar.__file__).read_text(encoding="utf-8")
    for forbidden in (
        '"docker", "start"',
        '"docker", "rm"',
        '"docker", "create"',
        '"docker", "update"',
        '"docker", "restart"',
    ):
        assert forbidden not in source

    test_files = [
        Path(__file__),
        Path(__file__).with_name("test_phase3_tier2r_impl_erratum1.py"),
    ]
    test_function_count = sum(
        1
        for test_file in test_files
        for node in ast.walk(ast.parse(test_file.read_text(encoding="utf-8")))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )
    assert test_function_count == registrar.EXPECTED_VALIDATION_TEST_COUNT == 22
