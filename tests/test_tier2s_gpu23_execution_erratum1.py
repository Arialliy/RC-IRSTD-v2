from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from scripts import launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum1 as launcher
from scripts import register_tier2s_gpu23_execution_erratum1 as registrar
from tests import conftest as governance_phase


ROOT = Path(__file__).resolve().parents[1]


def _host_inventory() -> list[dict[str, object]]:
    return [
        {"index": 0, "uuid": "GPU-0", "name": "RTX"},
        {"index": 1, "uuid": "GPU-1", "name": "RTX"},
        {"index": 2, "uuid": "GPU-2", "name": "RTX"},
        {"index": 3, "uuid": "GPU-3", "name": "RTX"},
    ]


def _runner(
    rows: str, processes: str = ""
) -> object:
    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        if any(value.startswith("--query-gpu=") for value in command):
            return subprocess.CompletedProcess(command, 0, rows, "")
        return subprocess.CompletedProcess(command, 0, processes, "")

    return runner


def test_frozen_parent_registration_remains_verified_and_unmodified() -> None:
    status = registrar.parent_v2.verify_registration()
    assert status["verified"] is True
    assert status["registered"] is True
    assert status["registration_sha256"] == registrar.PARENT_REGISTRATION_SHA256
    assert governance_phase.exact_parent_registration_is_frozen() is True
    assert not registrar.OLD_V2_INTENT.exists()
    assert registrar._sha256(registrar.PARENT_LAUNCHER) == registrar.PARENT_LAUNCHER_SHA256


def test_erratum_scope_is_only_idle_predicate_and_test_phase() -> None:
    payload = json.loads(registrar.ERRATUM_CONFIG.read_text(encoding="utf-8"))
    assert payload["parent_protocol"]["sha256"] == registrar.PARENT_PROTOCOL_SHA256
    assert payload["parent_governance_amendment"]["sha256"] == registrar.PARENT_REGISTRATION_SHA256
    assert payload["parent_launcher"]["sha256"] == registrar.PARENT_LAUNCHER_SHA256
    assert payload["append_only_policy"]["replacement_scope"] == (
        "host_idle_predicate_and_post_registration_test_phase_only"
    )
    assert payload["append_only_policy"]["parent_launcher_mutated"] is False
    assert governance_phase.OBSOLETE_PRE_REGISTRATION_NODEID.endswith(
        "test_amendment_verify_only_is_unregistered_and_never_authorizes_v3"
    )


def test_idle_erratum_accepts_driver_bookkeeping_without_compute_process() -> None:
    evidence = launcher._verify_selected_gpus_idle(
        _host_inventory(),
        runner=_runner("2, GPU-2, 1, 0\n3, GPU-3, 4, 0\n"),
    )
    assert evidence["schema_version"] == launcher.IDLE_SCHEMA
    assert evidence["driver_bookkeeping_ceiling_mib"] == 4
    assert evidence["no_compute_processes_required"] is True
    assert evidence["compute_processes"] == []


@pytest.mark.parametrize(
    ("rows", "processes"),
    (
        ("2, GPU-2, 5, 0\n3, GPU-3, 1, 0\n", ""),
        (
            "2, GPU-2, 1, 0\n3, GPU-3, 1, 0\n",
            "GPU-2, 123, python, 1\n",
        ),
    ),
)
def test_idle_erratum_rejects_real_memory_load_or_any_compute_process(
    rows: str, processes: str
) -> None:
    with pytest.raises(RuntimeError, match="driver-only idle contract"):
        launcher._verify_selected_gpus_idle(
            _host_inventory(), runner=_runner(rows, processes)
        )


def test_erratum_registration_state_is_valid_before_and_after_write_once() -> None:
    status = registrar.verify_registration()
    assert status["verified"] is True
    payload = (
        registrar._load_json(registrar.REGISTRATION)
        if status["registered"]
        else status["candidate"]
    )
    authorization = payload["authorization"]
    execution = payload["execution_scope"]
    assert authorization["tier2s_v2_source_only_diagnostic_authorized"] is True
    assert authorization["formal_v3_model_training_authorized"] is False
    assert authorization["outer_target_access_authorized"] is False
    assert execution["physical_gpus"] == [2, 3]
    assert execution["driver_bookkeeping_ceiling_mib"] == 4


def test_erratum_launcher_dry_run_preserves_gpu23_and_no_fallback() -> None:
    payload = launcher.dry_run_payload()
    assert payload["filesystem_mutated"] is False
    assert payload["coordinator_physical_gpus"] == [2, 3]
    assert payload["driver_bookkeeping_ceiling_mib"] == 4
    assert payload["no_compute_processes_required"] is True
    assert payload["docker_gpu_request"] == '"device=2,3"'
    assert payload["formal_container_spec"]["gpu_device_ids"] == ["2", "3"]
