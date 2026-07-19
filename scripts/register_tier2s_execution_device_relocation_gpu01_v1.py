#!/usr/bin/env python3
"""Register the append-only Tier2S outer device relocation to host GPU0/1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import register_aaai27_governance_v1 as parent_governance
from scripts import register_tier2s_gpu23_execution_erratum3 as parent_erratum3


SCHEMA = "rc-irstd-aaai27-tier2s-execution-device-relocation-gpu01-v1-registration"
BINDING_SCHEMA = "rc-irstd-aaai27-tier2s-execution-device-relocation-gpu01-v1-binding"
HOST_PHYSICAL_GPUS = (0, 1)
CONTAINER_ORDINALS = (0, 1)
FROZEN_COORDINATOR_LANES = (2, 3)

PARENT_REGISTRATION_SHA256 = "d9a6e7c96093f65facf6a9c34d7112522b7e051472ac431d6f01ffe8e06ae8e0"
PARENT_REGISTRATION_SIDECAR_SHA256 = "dce5fc9d32181da1bcbb1ab1890e9c5ac97d3c72cf173ace4fd6581904e5e1c9"
PARENT_CONFIG_SHA256 = "c99192f3a601c93a21d6204f3c51f6ad477402074b73b91e3389e1f4e4aa4181"
PARENT_LAUNCHER_SHA256 = "4fba509f933acacc45e56fec4b3297ec7547396d2d9747329ae0d5a30f161e6a"

PARENT_REGISTRATION = parent_erratum3.REGISTRATION
PARENT_REGISTRATION_SIDECAR = parent_erratum3.REGISTRATION_SIDECAR
PARENT_CONFIG = PROJECT_ROOT / "configs/tier2s_gpu23_execution_erratum3.json"
PARENT_LAUNCHER = PROJECT_ROOT / "scripts/launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum3.py"
RELOCATION_CONFIG = PROJECT_ROOT / "configs/tier2s_execution_device_relocation_gpu01_v1.json"
LAUNCHER = PROJECT_ROOT / "scripts/launch_tier2s_factorized_isolated_v2_device_relocation_gpu01_v1.py"
TEST_PATH = PROJECT_ROOT / "tests/test_tier2s_execution_device_relocation_gpu01_v1.py"

PARENT_AUDIT_ROOT = PROJECT_ROOT / "artifacts/aaai27/audit/source_rescue/tier2s_factorized_causal_audit_v2_gpu23"
PARENT_FORMAL_PATHS = (
    PARENT_AUDIT_ROOT / "LAUNCH_INTENT_EXECUTION_ERRATUM2.json",
    PARENT_AUDIT_ROOT / "LAUNCH_INTENT_EXECUTION_ERRATUM2.json.sha256",
    PARENT_AUDIT_ROOT / "PREREGISTRATION.json",
    PARENT_AUDIT_ROOT / "STATUS.json",
    PARENT_AUDIT_ROOT / "scheduler_events.jsonl",
)

AMENDMENT_ROOT = PROJECT_ROOT / "artifacts/aaai27/audit/governance/tier2s_execution_device_relocation_gpu01_v1"
REGISTRATION = AMENDMENT_ROOT / "TIER2S_EXECUTION_DEVICE_RELOCATION_GPU01_V1.json"
REGISTRATION_SIDECAR = REGISTRATION.with_suffix(REGISTRATION.suffix + ".sha256")

CODE_PATHS = (
    "configs/tier2s_execution_device_relocation_gpu01_v1.json",
    "scripts/register_tier2s_execution_device_relocation_gpu01_v1.py",
    "scripts/launch_tier2s_factorized_isolated_v2_device_relocation_gpu01_v1.py",
    "tests/test_tier2s_execution_device_relocation_gpu01_v1.py",
    "artifacts/aaai27/audit/governance/tier2s_gpu23_execution_erratum3/TIER2S_GPU23_EXECUTION_ERRATUM3.json",
    "artifacts/aaai27/audit/governance/tier2s_gpu23_execution_erratum3/TIER2S_GPU23_EXECUTION_ERRATUM3.json.sha256",
    "configs/tier2s_gpu23_execution_erratum3.json",
    "scripts/launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum3.py",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    return parent_erratum3._sha256(path)


def _load_json(path: Path) -> dict[str, Any]:
    return parent_erratum3._load_json(path)


def _require_regular(path: Path, *, expected_sha256: str | None = None) -> Path:
    return parent_erratum3._require_regular(path, expected_sha256=expected_sha256)


def _code_sha256() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in CODE_PATHS:
        path = _require_regular(PROJECT_ROOT / relative)
        result[relative] = _sha256(path)
    return result


def _require_parent_formal_absent() -> dict[str, Any]:
    present = [
        str(path.relative_to(PROJECT_ROOT))
        for path in PARENT_FORMAL_PATHS
        if path.exists() or path.is_symlink()
    ]
    if present:
        raise RuntimeError("GPU2/3 parent formal execution already started: " + ", ".join(present))
    return {
        "parent_gpu23_formal_intent_absent": True,
        "parent_gpu23_preregistration_absent": True,
        "parent_gpu23_status_absent": True,
        "parent_gpu23_scheduler_events_absent": True,
        "historical_evidence_superseded": False,
        "replacement_scope": "unexecuted_outer_execution_device_binding_only",
    }


def verify_inputs() -> dict[str, Any]:
    parent = parent_erratum3.require_frozen_execution_erratum3(
        expected_registration_sha256=PARENT_REGISTRATION_SHA256
    )
    _require_regular(PARENT_REGISTRATION_SIDECAR, expected_sha256=PARENT_REGISTRATION_SIDECAR_SHA256)
    _require_regular(PARENT_CONFIG, expected_sha256=PARENT_CONFIG_SHA256)
    _require_regular(PARENT_LAUNCHER, expected_sha256=PARENT_LAUNCHER_SHA256)
    _require_regular(RELOCATION_CONFIG)
    config = _load_json(RELOCATION_CONFIG)
    reason = config.get("reason", {})
    parent_spec = config.get("parent_execution_erratum3", {})
    execution = config.get("relocated_execution_contract", {})
    scientific = config.get("unchanged_scientific_contract", {})
    append_only = config.get("append_only_policy", {})
    if (
        config.get("schema_version") != "rc-irstd-aaai27-tier2s-execution-device-relocation-gpu01-v1"
        or config.get("relocation_id") != "tier2s_execution_device_relocation_gpu01_v1"
        or config.get("research_mode") != "exploratory_source_only"
        or reason.get("user_resource_authorization") != "gpu0-4随便用"
        or reason.get("selected_idle_host_gpus") != [0, 1]
        or reason.get("busy_host_gpus_excluded") != [2, 3]
        or parent_spec.get("registration_sha256") != PARENT_REGISTRATION_SHA256
        or parent_spec.get("registration_sidecar_sha256") != PARENT_REGISTRATION_SIDECAR_SHA256
        or parent_spec.get("config_sha256") != PARENT_CONFIG_SHA256
        or parent_spec.get("launcher_sha256") != PARENT_LAUNCHER_SHA256
        or execution.get("host_physical_gpus") != [0, 1]
        or execution.get("container_nvidia_indices") != [0, 1]
        or execution.get("torch_ordinals") != [0, 1]
        or execution.get("host_physical_to_container_ordinal") != {"0": 0, "1": 1}
        or execution.get("docker_device_ids") != ["0", "1"]
        or execution.get("driver_bookkeeping_ceiling_mib") != 4
        or execution.get("no_compute_processes_required") is not True
        or execution.get("gpu_fallback_allowed") is not False
        or scientific.get("protocol_id") != "tier2s_factorized_causal_audit_v2_gpu23"
        or scientific.get("coordinator_matrix_unchanged") is not True
        or scientific.get("coordinator_legacy_lane_labels") != [2, 3]
        or scientific.get("legacy_lane_to_host_physical_gpu") != {"2": 0, "3": 1}
        or scientific.get("legacy_lane_field_is_execution_lane_label_after_relocation") is not True
        or scientific.get("jobs_per_lane") != 9
        or scientific.get("total_jobs") != 18
        or scientific.get("source_only") is not True
        or scientific.get("result_use") != "failure_attribution_only"
        or scientific.get("formal_v3_model_training_authorized") is not False
        or scientific.get("source_gate_a_authorized") is not False
        or scientific.get("riskcurve_authorized") is not False
        or scientific.get("paper_claim_authorized") is not False
        or scientific.get("outer_target_access_authorized") is not False
        or append_only.get("parent_gpu23_files_mutated") is not False
        or append_only.get("parent_gpu23_registration_mutated") is not False
        or append_only.get("parent_gpu23_formal_execution_must_be_absent") is not True
        or append_only.get("historical_tier2r_hold_mutated") is not False
        or append_only.get("replacement_scope") != "unexecuted_outer_execution_device_binding_only"
        or append_only.get("scientific_protocol_replaced") is not False
    ):
        raise RuntimeError("Tier2S GPU0/1 relocation semantics drift")
    return {
        "verified": True,
        "parent_execution_erratum3_binding": parent,
        "append_only_state": _require_parent_formal_absent(),
        "relocation_config": {
            "path": str(RELOCATION_CONFIG.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(RELOCATION_CONFIG),
        },
        "code_sha256": _code_sha256(),
        "host_physical_gpus": [0, 1],
        "container_ordinals": [0, 1],
        "frozen_coordinator_lane_labels": [2, 3],
        "legacy_lane_to_host_physical_gpu": {"2": 0, "3": 1},
    }


def _registration_payload(verified: Mapping[str, Any], tests: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "registered_at": _now(),
        "decision": "AUTHORIZE_TIER2S_V2_SOURCE_ONLY_OUTER_DEVICE_RELOCATION_GPU01_V1",
        "parent_execution_erratum3_binding": verified["parent_execution_erratum3_binding"],
        "append_only_state": verified["append_only_state"],
        "relocation_config": verified["relocation_config"],
        "code_sha256": verified["code_sha256"],
        "execution_scope": {
            "entrypoint": "python3 scripts/launch_tier2s_factorized_isolated_v2_device_relocation_gpu01_v1.py",
            "host_physical_gpus": [0, 1],
            "container_nvidia_indices": [0, 1],
            "torch_ordinals": [0, 1],
            "host_physical_to_container_ordinal": {"0": 0, "1": 1},
            "docker_device_ids": ["0", "1"],
            "frozen_coordinator_lane_labels": [2, 3],
            "legacy_lane_to_host_physical_gpu": {"2": 0, "3": 1},
            "coordinator_matrix_unchanged": True,
            "jobs_per_lane": 9,
            "total_jobs": 18,
            "gpu_fallback_allowed": False,
            "driver_bookkeeping_ceiling_mib": 4,
            "no_compute_processes_required": True,
        },
        "authorization": {
            "tier2s_v2_source_only_diagnostic_authorized": True,
            "formal_v3_model_training_authorized": False,
            "source_gate_a_authorized": False,
            "riskcurve_authorized": False,
            "paper_claim_authorized": False,
            "outer_target_access_authorized": False,
        },
        "full_test_gate": dict(tests),
        "failure_policy": {
            "any_code_config_or_parent_sha_drift": "FAIL_CLOSED_NO_LAUNCH",
            "gpu01_not_exactly_idle": "FAIL_CLOSED_NO_LAUNCH",
            "container_uuid_or_ordinal_mismatch": "FAIL_CLOSED_NO_LAUNCH",
            "parent_gpu23_formal_artifact_appears": "FAIL_CLOSED_NEW_AMENDMENT_REQUIRED",
            "scientific_matrix_drift": "FAIL_CLOSED_NO_LAUNCH",
        },
    }


def register() -> dict[str, Any]:
    if REGISTRATION.exists() or REGISTRATION_SIDECAR.exists():
        raise RuntimeError("Tier2S GPU0/1 relocation registration already exists; never overwrite")
    before = verify_inputs()
    tests = parent_governance._run_full_tests()
    after = verify_inputs()
    if before != after:
        raise RuntimeError("Tier2S GPU0/1 relocation input drifted during tests")
    parent_erratum3.parent_erratum2._validate_test_evidence(tests)
    writer = parent_erratum3.parent_erratum2.parent_erratum1
    writer._write_once(REGISTRATION, writer._canonical_bytes(_registration_payload(after, tests)))
    digest = writer._write_sidecar(REGISTRATION, REGISTRATION_SIDECAR)
    verified = verify_registration()
    if verified["registration_sha256"] != digest:
        raise RuntimeError("Tier2S GPU0/1 relocation changed during final verification")
    return verified


def verify_registration() -> dict[str, Any]:
    present = REGISTRATION.exists() or REGISTRATION.is_symlink()
    sidecar_present = REGISTRATION_SIDECAR.exists() or REGISTRATION_SIDECAR.is_symlink()
    if not present and not sidecar_present:
        return {
            "verified": True,
            "registered": False,
            "candidate": _registration_payload(
                verify_inputs(),
                {"status": "not_run_in_verify_only", "required_before_registration": True},
            ),
        }
    if not present or not sidecar_present:
        raise RuntimeError("Tier2S GPU0/1 relocation registration pair is incomplete")
    writer = parent_erratum3.parent_erratum2.parent_erratum1
    digest = writer._verify_sidecar(REGISTRATION, REGISTRATION_SIDECAR)
    writer._require_read_only(REGISTRATION, REGISTRATION_SIDECAR)
    current = verify_inputs()
    payload = _load_json(REGISTRATION)
    execution = payload.get("execution_scope", {})
    authorization = payload.get("authorization", {})
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("decision") != "AUTHORIZE_TIER2S_V2_SOURCE_ONLY_OUTER_DEVICE_RELOCATION_GPU01_V1"
        or payload.get("parent_execution_erratum3_binding") != current["parent_execution_erratum3_binding"]
        or payload.get("append_only_state") != current["append_only_state"]
        or payload.get("relocation_config") != current["relocation_config"]
        or payload.get("code_sha256") != current["code_sha256"]
        or execution.get("host_physical_gpus") != [0, 1]
        or execution.get("frozen_coordinator_lane_labels") != [2, 3]
        or execution.get("legacy_lane_to_host_physical_gpu") != {"2": 0, "3": 1}
        or execution.get("total_jobs") != 18
        or execution.get("coordinator_matrix_unchanged") is not True
        or authorization.get("tier2s_v2_source_only_diagnostic_authorized") is not True
        or authorization.get("formal_v3_model_training_authorized") is not False
        or authorization.get("outer_target_access_authorized") is not False
    ):
        raise RuntimeError("frozen Tier2S GPU0/1 relocation registration drift")
    parent_erratum3.parent_erratum2._validate_test_evidence(payload.get("full_test_gate"))
    return {"verified": True, "registered": True, "registration_sha256": digest}


def require_frozen_device_relocation_gpu01_v1(
    *, expected_registration_sha256: str | None = None
) -> dict[str, Any]:
    status = verify_registration()
    if not status.get("registered"):
        raise RuntimeError("Tier2S GPU0/1 relocation is not registered")
    if expected_registration_sha256 is not None and status.get("registration_sha256") != expected_registration_sha256:
        raise RuntimeError("Tier2S GPU0/1 relocation registration SHA-256 drift")
    payload = _load_json(REGISTRATION)
    return {
        "schema_version": BINDING_SCHEMA,
        "verified": True,
        "registration_path": str(REGISTRATION.relative_to(PROJECT_ROOT)),
        "registration_sha256": status["registration_sha256"],
        "registration_sidecar_path": str(REGISTRATION_SIDECAR.relative_to(PROJECT_ROOT)),
        "registration_sidecar_sha256": _sha256(REGISTRATION_SIDECAR),
        "host_physical_gpus": [0, 1],
        "container_ordinals": [0, 1],
        "frozen_coordinator_lane_labels": [2, 3],
        "legacy_lane_to_host_physical_gpu": {"2": 0, "3": 1},
        "authorization": payload["authorization"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = register() if args.register else verify_registration()
    except BaseException as error:
        print(f"FAILED_CLOSED {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
