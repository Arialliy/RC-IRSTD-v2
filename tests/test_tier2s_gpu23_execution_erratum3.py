from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from scripts import launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum2 as parent_launcher
from scripts import launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum3 as launcher
from scripts import register_tier2s_gpu23_execution_erratum3 as registrar


ROOT = Path(__file__).resolve().parents[1]


def test_erratum3_preserves_exact_frozen_erratum2_registration_and_code() -> None:
    parent = registrar.parent_erratum2.require_frozen_execution_erratum2(
        expected_registration_sha256=registrar.PARENT_REGISTRATION_SHA256
    )
    assert parent["physical_gpus"] == [2, 3]
    assert parent["container_nvidia_indices"] == [0, 1]
    assert registrar._sha256(registrar.PARENT_REGISTRATION_SIDECAR) == (
        registrar.PARENT_REGISTRATION_SIDECAR_SHA256
    )
    assert registrar._sha256(registrar.PARENT_CONFIG) == registrar.PARENT_CONFIG_SHA256
    assert registrar._sha256(registrar.PARENT_LAUNCHER) == registrar.PARENT_LAUNCHER_SHA256


def test_erratum3_scope_is_direct_file_bootstrap_only() -> None:
    payload = json.loads(registrar.ERRATUM_CONFIG.read_text(encoding="utf-8"))
    assert payload["append_only_policy"]["replacement_scope"] == (
        "direct_file_entrypoint_project_root_bootstrap_only"
    )
    assert payload["append_only_policy"]["parent_execution_erratum2_mutated"] is False
    assert payload["unchanged_execution_contract"][
        "formal_container_name_reused_from_unexecuted_erratum2"
    ] is True
    assert payload["unchanged_execution_contract"]["total_jobs"] == 18
    assert payload["unchanged_execution_contract"][
        "formal_v3_model_training_authorized"
    ] is False


def test_direct_file_entrypoint_succeeds_without_inherited_pythonpath() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(launcher.Path(__file__).resolve().parents[1] / "scripts"
                / "launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum3.py"),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["dry_run"] is True
    assert payload["docker_invoked"] is False
    assert payload["filesystem_mutated"] is False
    assert payload["execution_erratum_config"] == str(
        launcher.EXECUTION_ERRATUM_CONFIG
    )
    assert payload["execution_erratum_registration"] == str(
        launcher.REGISTRATION
    )
    assert payload["execute_with"].endswith(
        "launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum3.py --execute"
    )


def test_erratum3_dry_run_rebinds_latest_inputs_and_restores_parent() -> None:
    old_code_paths = parent_launcher.CODE_PATHS
    old_config_paths = parent_launcher.CONFIG_PATHS
    old_config = parent_launcher.EXECUTION_ERRATUM_CONFIG
    payload = launcher.dry_run_payload()
    assert payload["container_name"] == parent_launcher.CONTAINER_NAME
    assert payload["coordinator_physical_gpus"] == [2, 3]
    assert payload["formal_container_spec"]["gpu_device_ids"] == ["2", "3"]
    assert payload["docker_gpu_request"] == '"device=2,3"'
    assert payload["execution_erratum_config"] == str(
        launcher.EXECUTION_ERRATUM_CONFIG
    )
    assert payload["execution_erratum_registration"] == str(
        launcher.REGISTRATION
    )
    assert parent_launcher.CODE_PATHS == old_code_paths
    assert parent_launcher.CONFIG_PATHS == old_config_paths
    assert parent_launcher.EXECUTION_ERRATUM_CONFIG == old_config


def test_erratum3_registration_state_is_valid_before_and_after_write_once() -> None:
    status = registrar.verify_registration()
    assert status["verified"] is True
    payload = (
        registrar._load_json(registrar.REGISTRATION)
        if status["registered"]
        else status["candidate"]
    )
    execution = payload["execution_scope"]
    authorization = payload["authorization"]
    assert execution["exact_project_root_bootstrap"] == str(ROOT)
    assert execution["delegates_to_frozen_erratum2"] is True
    assert execution["physical_gpus"] == [2, 3]
    assert execution["container_nvidia_indices"] == [0, 1]
    assert execution["total_jobs"] == 18
    assert authorization["tier2s_v2_source_only_diagnostic_authorized"] is True
    assert authorization["formal_v3_model_training_authorized"] is False
    assert authorization["outer_target_access_authorized"] is False


def test_erratum3_formal_command_remains_exactly_gpu2_and_gpu3() -> None:
    command = launcher.build_formal_command("0" * 64)
    assert command[command.index("--gpus") + 1] == '"device=2,3"'
    assert command[command.index("--name") + 1] == parent_launcher.CONTAINER_NAME
    assert "CUDA_VISIBLE_DEVICES=" not in "\n".join(command)

