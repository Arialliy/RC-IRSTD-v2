#!/usr/bin/env python3
"""Append-only governance for Tier2S GPU2/3 execution erratum2.

The frozen protocol, GPU2/3 amendment, and execution erratum1 remain immutable.
This erratum corrects only the namespace interpretation of NVIDIA indices
inside a container that exposes host physical devices 2 and 3. Device identity
is proved by UUID; scientific authorization remains failure-attribution only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import register_aaai27_governance_v1 as parent_governance
from scripts import register_tier2s_gpu23_execution_erratum1 as parent_erratum1


SCHEMA = "rc-irstd-aaai27-tier2s-gpu23-execution-erratum2-registration"
BINDING_SCHEMA = "rc-irstd-aaai27-tier2s-gpu23-execution-erratum2-binding"
PROTOCOL_ID = "tier2s_factorized_causal_audit_v2_gpu23"
PHYSICAL_GPUS = (2, 3)
CONTAINER_LOGICAL_ORDINALS = (0, 1)

PARENT_REGISTRATION_SHA256 = (
    "802784be936d2cb0759c91a24688f6bc76f7f76daedb699ca3a4414a9d900f5f"
)
PARENT_CONFIG_SHA256 = (
    "df5df381abfaaf4205f754263fec43ab78021ba309e8c381498e9fbba9b21cd2"
)
PARENT_LAUNCHER_SHA256 = (
    "ae8755f962e919a1158a7ed44e6a8d7addfa8e6f3e8015926ed0a76e16b035b2"
)

PARENT_REGISTRATION = parent_erratum1.REGISTRATION
PARENT_REGISTRATION_SIDECAR = parent_erratum1.REGISTRATION_SIDECAR
PARENT_CONFIG = PROJECT_ROOT / "configs/tier2s_gpu23_execution_erratum1.json"
PARENT_LAUNCHER = (
    PROJECT_ROOT
    / "scripts/launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum1.py"
)
ERRATUM_CONFIG = PROJECT_ROOT / "configs/tier2s_gpu23_execution_erratum2.json"
LAUNCHER = (
    PROJECT_ROOT
    / "scripts/launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum2.py"
)
TEST_PATH = PROJECT_ROOT / "tests/test_tier2s_gpu23_execution_erratum2.py"
PARENT_AUDIT_ROOT = (
    PROJECT_ROOT
    / "artifacts/aaai27/audit/source_rescue/tier2s_factorized_causal_audit_v2_gpu23"
)
PARENT_INTENT = PARENT_AUDIT_ROOT / "LAUNCH_INTENT_EXECUTION_ERRATUM1.json"
PARENT_INTENT_SIDECAR = PARENT_INTENT.with_suffix(PARENT_INTENT.suffix + ".sha256")
PARENT_PREREGISTRATION = PARENT_AUDIT_ROOT / "PREREGISTRATION.json"
PARENT_STATUS = PARENT_AUDIT_ROOT / "STATUS.json"
PARENT_EVENTS = PARENT_AUDIT_ROOT / "scheduler_events.jsonl"

AMENDMENT_ROOT = (
    PROJECT_ROOT
    / "artifacts/aaai27/audit/governance/tier2s_gpu23_execution_erratum2"
)
REGISTRATION = AMENDMENT_ROOT / "TIER2S_GPU23_EXECUTION_ERRATUM2.json"
REGISTRATION_SIDECAR = REGISTRATION.with_suffix(REGISTRATION.suffix + ".sha256")

CODE_PATHS = (
    "configs/tier2s_gpu23_execution_erratum2.json",
    "scripts/register_tier2s_gpu23_execution_erratum2.py",
    "scripts/launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum2.py",
    "tests/test_tier2s_gpu23_execution_erratum2.py",
    "artifacts/aaai27/audit/governance/tier2s_gpu23_execution_erratum1/"
    "TIER2S_GPU23_EXECUTION_ERRATUM1.json",
    "artifacts/aaai27/audit/governance/tier2s_gpu23_execution_erratum1/"
    "TIER2S_GPU23_EXECUTION_ERRATUM1.json.sha256",
    "configs/tier2s_gpu23_execution_erratum1.json",
    "scripts/launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum1.py",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, Any]:
    return parent_erratum1._load_json(path)


def _sha256(path: Path) -> str:
    return parent_erratum1._sha256(path)


def _require_regular(path: Path, *, expected_sha256: str | None = None) -> Path:
    return parent_erratum1._require_regular(path, expected_sha256=expected_sha256)


def _code_sha256() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in CODE_PATHS:
        path = _require_regular(PROJECT_ROOT / relative)
        result[relative] = _sha256(path)
    return result


def _require_parent_execution_absent() -> dict[str, Any]:
    forbidden = (
        PARENT_INTENT,
        PARENT_INTENT_SIDECAR,
        PARENT_PREREGISTRATION,
        PARENT_STATUS,
        PARENT_EVENTS,
    )
    present = [str(path.relative_to(PROJECT_ROOT)) for path in forbidden if path.exists() or path.is_symlink()]
    if present:
        raise RuntimeError(
            "execution erratum1 already produced formal artifacts: " + ", ".join(present)
        )
    return {
        "parent_erratum1_formal_intent_absent": True,
        "parent_erratum1_preregistration_absent": True,
        "parent_erratum1_status_absent": True,
        "parent_erratum1_scheduler_events_absent": True,
        "historical_evidence_superseded": False,
        "replacement_scope": "container_nvidia_index_namespace_attestation_only",
    }


def verify_inputs() -> dict[str, Any]:
    parent_binding = parent_erratum1.require_frozen_execution_erratum(
        expected_registration_sha256=PARENT_REGISTRATION_SHA256
    )
    _require_regular(PARENT_CONFIG, expected_sha256=PARENT_CONFIG_SHA256)
    _require_regular(PARENT_LAUNCHER, expected_sha256=PARENT_LAUNCHER_SHA256)
    _require_regular(ERRATUM_CONFIG)
    config = _load_json(ERRATUM_CONFIG)
    parent = config.get("parent_execution_erratum1", {})
    mismatch = config.get("observed_post_registration_mismatch", {})
    unchanged = config.get("unchanged_scientific_and_hardware_contract", {})
    append_only = config.get("append_only_policy", {})
    if (
        config.get("schema_version")
        != "rc-irstd-aaai27-tier2s-gpu23-execution-erratum2"
        or config.get("erratum_id") != "tier2s_gpu23_execution_erratum2"
        or config.get("research_mode") != "exploratory_source_only"
        or parent.get("registration_sha256") != PARENT_REGISTRATION_SHA256
        or parent.get("config_sha256") != PARENT_CONFIG_SHA256
        or parent.get("launcher_sha256") != PARENT_LAUNCHER_SHA256
        or mismatch.get("identity_authority")
        != "host_physical_index_to_uuid_then_container_local_index_to_same_uuid"
        or mismatch.get("corrected_attestation")
        != (
            "container_nvidia_indices_and_torch_ordinals_must_be_exactly_0_1_"
            "while_uuid_and_name_pairs_must_equal_host_physical_gpu2_and_gpu3_in_order"
        )
        or mismatch.get("extra_missing_reordered_or_wrong_uuid_device")
        != "FAIL_CLOSED_NO_LAUNCH"
        or unchanged.get("physical_gpus") != list(PHYSICAL_GPUS)
        or unchanged.get("container_logical_ordinals") != {"2": 0, "3": 1}
        or unchanged.get("docker_device_ids") != ["2", "3"]
        or unchanged.get("gpu_fallback_allowed") is not False
        or unchanged.get("jobs_per_lane") != 9
        or unchanged.get("total_jobs") != 18
        or unchanged.get("result_use") != "failure_attribution_only"
        or unchanged.get("formal_v3_model_training_authorized") is not False
        or unchanged.get("source_gate_a_authorized") is not False
        or unchanged.get("riskcurve_authorized") is not False
        or unchanged.get("paper_claim_authorized") is not False
        or unchanged.get("outer_target_access_authorized") is not False
        or append_only.get("parent_execution_erratum1_mutated") is not False
        or append_only.get("historical_tier2r_hold_mutated") is not False
        or append_only.get("replacement_scope")
        != "container_nvidia_index_namespace_attestation_only"
        or append_only.get("parent_erratum1_formal_intent_must_be_absent") is not True
        or append_only.get("new_formal_container_and_intent_names_required") is not True
    ):
        raise RuntimeError("Tier2S execution erratum2 semantics drift")
    return {
        "verified": True,
        "parent_execution_erratum1_binding": parent_binding,
        "append_only_state": _require_parent_execution_absent(),
        "execution_erratum_config": {
            "path": str(ERRATUM_CONFIG.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(ERRATUM_CONFIG),
        },
        "code_sha256": _code_sha256(),
        "physical_gpus": list(PHYSICAL_GPUS),
        "container_logical_ordinals": list(CONTAINER_LOGICAL_ORDINALS),
        "physical_to_container_ordinal": {"2": 0, "3": 1},
    }


def _registration_payload(
    verified: Mapping[str, Any], test_evidence: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "registered_at": _now(),
        "decision": "AUTHORIZE_TIER2S_V2_GPU23_EXECUTION_ERRATUM2_SOURCE_ONLY",
        "parent_execution_erratum1_binding": verified[
            "parent_execution_erratum1_binding"
        ],
        "append_only_state": verified["append_only_state"],
        "execution_erratum_config": verified["execution_erratum_config"],
        "code_sha256": verified["code_sha256"],
        "execution_scope": {
            "physical_gpus": list(PHYSICAL_GPUS),
            "container_logical_ordinals": list(CONTAINER_LOGICAL_ORDINALS),
            "physical_to_container_ordinal": {"2": 0, "3": 1},
            "docker_device_ids": ["2", "3"],
            "identity_proof": "host_physical_index_to_uuid_to_container_local_index",
            "container_nvidia_indices": [0, 1],
            "torch_ordinals": [0, 1],
            "two_fixed_independent_fifo_lanes": True,
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
        "full_test_gate": dict(test_evidence),
        "failure_policy": {
            "any_code_config_or_parent_sha_drift": "FAIL_CLOSED_NO_LAUNCH",
            "container_index_not_exactly_0_1": "FAIL_CLOSED_NO_LAUNCH",
            "uuid_name_order_not_exact_host_physical_2_3": "FAIL_CLOSED_NO_LAUNCH",
            "extra_missing_or_reordered_device": "FAIL_CLOSED_NO_LAUNCH",
            "parent_erratum1_formal_artifact_appears": "FAIL_CLOSED_NEW_ERRATUM_REQUIRED",
        },
    }


def _validate_test_evidence(value: Any) -> None:
    parent_erratum1._validate_test_evidence(value)


def register() -> dict[str, Any]:
    if REGISTRATION.exists() or REGISTRATION_SIDECAR.exists():
        raise RuntimeError("Tier2S execution erratum2 already exists; never overwrite")
    before = verify_inputs()
    tests = parent_governance._run_full_tests()
    after = verify_inputs()
    if before != after:
        raise RuntimeError("Tier2S execution erratum2 input drifted during tests")
    _validate_test_evidence(tests)
    parent_erratum1._write_once(
        REGISTRATION,
        parent_erratum1._canonical_bytes(_registration_payload(after, tests)),
    )
    digest = parent_erratum1._write_sidecar(REGISTRATION, REGISTRATION_SIDECAR)
    verified = verify_registration()
    if verified["registration_sha256"] != digest:
        raise RuntimeError("Tier2S execution erratum2 changed during final verification")
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
        raise RuntimeError("Tier2S execution erratum2 registration pair is incomplete")
    digest = parent_erratum1._verify_sidecar(REGISTRATION, REGISTRATION_SIDECAR)
    parent_erratum1._require_read_only(REGISTRATION, REGISTRATION_SIDECAR)
    current = verify_inputs()
    registration = _load_json(REGISTRATION)
    _validate_test_evidence(registration.get("full_test_gate"))
    execution = registration.get("execution_scope", {})
    authorization = registration.get("authorization", {})
    if (
        registration.get("schema_version") != SCHEMA
        or registration.get("decision")
        != "AUTHORIZE_TIER2S_V2_GPU23_EXECUTION_ERRATUM2_SOURCE_ONLY"
        or not isinstance(registration.get("registered_at"), str)
        or registration.get("parent_execution_erratum1_binding")
        != current["parent_execution_erratum1_binding"]
        or registration.get("append_only_state") != current["append_only_state"]
        or registration.get("execution_erratum_config")
        != current["execution_erratum_config"]
        or registration.get("code_sha256") != current["code_sha256"]
        or execution.get("physical_gpus") != list(PHYSICAL_GPUS)
        or execution.get("container_logical_ordinals")
        != list(CONTAINER_LOGICAL_ORDINALS)
        or execution.get("physical_to_container_ordinal") != {"2": 0, "3": 1}
        or execution.get("docker_device_ids") != ["2", "3"]
        or execution.get("container_nvidia_indices") != [0, 1]
        or execution.get("torch_ordinals") != [0, 1]
        or execution.get("jobs_per_lane") != 9
        or execution.get("total_jobs") != 18
        or execution.get("gpu_fallback_allowed") is not False
        or execution.get("driver_bookkeeping_ceiling_mib") != 4
        or execution.get("no_compute_processes_required") is not True
        or authorization.get("tier2s_v2_source_only_diagnostic_authorized") is not True
        or authorization.get("formal_v3_model_training_authorized") is not False
        or authorization.get("source_gate_a_authorized") is not False
        or authorization.get("riskcurve_authorized") is not False
        or authorization.get("paper_claim_authorized") is not False
        or authorization.get("outer_target_access_authorized") is not False
    ):
        raise RuntimeError("frozen Tier2S execution erratum2 semantic drift")
    return {
        "verified": True,
        "registered": True,
        "registration_path": str(REGISTRATION),
        "registration_sha256": digest,
        "tier2s_v2_source_only_diagnostic_authorized": True,
        "formal_v3_model_training_authorized": False,
        "outer_target_access_authorized": False,
    }


def require_frozen_execution_erratum2(
    *, expected_registration_sha256: str | None = None
) -> dict[str, Any]:
    status = verify_registration()
    if status.get("registered") is not True:
        raise RuntimeError("Tier2S execution erratum2 is not registered")
    digest = str(status["registration_sha256"])
    if expected_registration_sha256 is not None and digest != expected_registration_sha256:
        raise RuntimeError("Tier2S execution erratum2 SHA-256 mismatch")
    registration = _load_json(REGISTRATION)
    parent_binding = registration["parent_execution_erratum1_binding"]
    code_map = registration["code_sha256"]
    return {
        "schema_version": BINDING_SCHEMA,
        "registration": parent_erratum1._frozen_binding(
            REGISTRATION, REGISTRATION_SIDECAR
        ),
        "parent_execution_erratum1_registration": parent_binding["registration"],
        "parent_tier2s_governance_registration": parent_binding[
            "parent_tier2s_governance_registration"
        ],
        "parent_aaai27_governance_registration": parent_binding[
            "parent_aaai27_governance_registration"
        ],
        "contract": parent_binding["contract"],
        "fresh_seed_ledger": parent_binding["fresh_seed_ledger"],
        "fresh_seed_local_scan": parent_binding["fresh_seed_local_scan"],
        "code_sha256_canonical_sha256": hashlib.sha256(
            parent_erratum1._canonical_bytes(code_map)
        ).hexdigest(),
        "physical_gpus": list(PHYSICAL_GPUS),
        "container_logical_ordinals": list(CONTAINER_LOGICAL_ORDINALS),
        "physical_to_container_ordinal": {"2": 0, "3": 1},
        "docker_device_ids": ["2", "3"],
        "container_nvidia_indices": [0, 1],
        "torch_ordinals": [0, 1],
        "gpu_fallback_allowed": False,
        "execution_erratum_config": registration["execution_erratum_config"],
        "append_only_state": registration["append_only_state"],
        "driver_bookkeeping_ceiling_mib": 4,
        "no_compute_processes_required": True,
        "tier2s_source_only_diagnostic_authorized": True,
        "formal_v3_model_training_authorized": False,
        "source_gate_a_authorized": False,
        "riskcurve_authorized": False,
        "paper_claim_authorized": False,
        "outer_target_access_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = register() if args.register else verify_registration()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

