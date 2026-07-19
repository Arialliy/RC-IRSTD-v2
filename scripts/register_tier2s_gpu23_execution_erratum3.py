#!/usr/bin/env python3
"""Append-only governance for the Tier2S erratum3 direct-file entrypoint.

The frozen erratum2 implementation is imported unchanged. This registration
authorizes only a thin wrapper that injects the exact project root before that
import and binds the wrapper plus this governance chain into launch-intent
hashes. No scientific, data, GPU, scheduling, or authorization rule changes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import register_aaai27_governance_v1 as parent_governance
from scripts import register_tier2s_gpu23_execution_erratum2 as parent_erratum2


SCHEMA = "rc-irstd-aaai27-tier2s-gpu23-execution-erratum3-registration"
BINDING_SCHEMA = "rc-irstd-aaai27-tier2s-gpu23-execution-erratum3-binding"
PHYSICAL_GPUS = (2, 3)
CONTAINER_LOGICAL_ORDINALS = (0, 1)

PARENT_REGISTRATION_SHA256 = (
    "6eb4dbcb1f7ddea180ec45b3e4d7e76ffad0cdedd44f62b165a5068de6498407"
)
PARENT_REGISTRATION_SIDECAR_SHA256 = (
    "9d41ead0f93d229a85be11bd7607a4accbdc3ba2c287d62fef3253befb2afe70"
)
PARENT_CONFIG_SHA256 = (
    "51379e15e22d398f59d7a2f482e921d039cc439ac940e5fdc562ab56469a74b0"
)
PARENT_LAUNCHER_SHA256 = (
    "3825f3922fc5008001d82d087bce2177cf99d925e6eb9d1cae7739e44d64f553"
)

PARENT_REGISTRATION = parent_erratum2.REGISTRATION
PARENT_REGISTRATION_SIDECAR = parent_erratum2.REGISTRATION_SIDECAR
PARENT_CONFIG = PROJECT_ROOT / "configs/tier2s_gpu23_execution_erratum2.json"
PARENT_LAUNCHER = (
    PROJECT_ROOT
    / "scripts/launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum2.py"
)
ERRATUM_CONFIG = PROJECT_ROOT / "configs/tier2s_gpu23_execution_erratum3.json"
LAUNCHER = (
    PROJECT_ROOT
    / "scripts/launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum3.py"
)
TEST_PATH = PROJECT_ROOT / "tests/test_tier2s_gpu23_execution_erratum3.py"
PARENT_AUDIT_ROOT = (
    PROJECT_ROOT
    / "artifacts/aaai27/audit/source_rescue/tier2s_factorized_causal_audit_v2_gpu23"
)
PARENT_INTENT = PARENT_AUDIT_ROOT / "LAUNCH_INTENT_EXECUTION_ERRATUM2.json"
PARENT_INTENT_SIDECAR = PARENT_INTENT.with_suffix(PARENT_INTENT.suffix + ".sha256")
PARENT_PREREGISTRATION = PARENT_AUDIT_ROOT / "PREREGISTRATION.json"
PARENT_STATUS = PARENT_AUDIT_ROOT / "STATUS.json"
PARENT_EVENTS = PARENT_AUDIT_ROOT / "scheduler_events.jsonl"

AMENDMENT_ROOT = (
    PROJECT_ROOT
    / "artifacts/aaai27/audit/governance/tier2s_gpu23_execution_erratum3"
)
REGISTRATION = AMENDMENT_ROOT / "TIER2S_GPU23_EXECUTION_ERRATUM3.json"
REGISTRATION_SIDECAR = REGISTRATION.with_suffix(REGISTRATION.suffix + ".sha256")

CODE_PATHS = (
    "configs/tier2s_gpu23_execution_erratum3.json",
    "scripts/register_tier2s_gpu23_execution_erratum3.py",
    "scripts/launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum3.py",
    "tests/test_tier2s_gpu23_execution_erratum3.py",
    "artifacts/aaai27/audit/governance/tier2s_gpu23_execution_erratum2/"
    "TIER2S_GPU23_EXECUTION_ERRATUM2.json",
    "artifacts/aaai27/audit/governance/tier2s_gpu23_execution_erratum2/"
    "TIER2S_GPU23_EXECUTION_ERRATUM2.json.sha256",
    "configs/tier2s_gpu23_execution_erratum2.json",
    "scripts/launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum2.py",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, Any]:
    return parent_erratum2._load_json(path)


def _sha256(path: Path) -> str:
    return parent_erratum2._sha256(path)


def _require_regular(path: Path, *, expected_sha256: str | None = None) -> Path:
    return parent_erratum2._require_regular(path, expected_sha256=expected_sha256)


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
    present = [
        str(path.relative_to(PROJECT_ROOT))
        for path in forbidden
        if path.exists() or path.is_symlink()
    ]
    if present:
        raise RuntimeError(
            "execution erratum2 already produced formal artifacts: "
            + ", ".join(present)
        )
    return {
        "parent_erratum2_formal_intent_absent": True,
        "parent_erratum2_preregistration_absent": True,
        "parent_erratum2_status_absent": True,
        "parent_erratum2_scheduler_events_absent": True,
        "historical_evidence_superseded": False,
        "replacement_scope": "direct_file_entrypoint_project_root_bootstrap_only",
    }


def verify_inputs() -> dict[str, Any]:
    parent_binding = parent_erratum2.require_frozen_execution_erratum2(
        expected_registration_sha256=PARENT_REGISTRATION_SHA256
    )
    _require_regular(
        PARENT_REGISTRATION_SIDECAR,
        expected_sha256=PARENT_REGISTRATION_SIDECAR_SHA256,
    )
    _require_regular(PARENT_CONFIG, expected_sha256=PARENT_CONFIG_SHA256)
    _require_regular(PARENT_LAUNCHER, expected_sha256=PARENT_LAUNCHER_SHA256)
    _require_regular(ERRATUM_CONFIG)
    config = _load_json(ERRATUM_CONFIG)
    parent = config.get("parent_execution_erratum2", {})
    mismatch = config.get("observed_post_registration_mismatch", {})
    unchanged = config.get("unchanged_execution_contract", {})
    append_only = config.get("append_only_policy", {})
    if (
        config.get("schema_version")
        != "rc-irstd-aaai27-tier2s-gpu23-execution-erratum3"
        or config.get("erratum_id") != "tier2s_gpu23_execution_erratum3"
        or config.get("research_mode") != "exploratory_source_only"
        or parent.get("registration_sha256") != PARENT_REGISTRATION_SHA256
        or parent.get("registration_sidecar_sha256")
        != PARENT_REGISTRATION_SIDECAR_SHA256
        or parent.get("config_sha256") != PARENT_CONFIG_SHA256
        or parent.get("launcher_sha256") != PARENT_LAUNCHER_SHA256
        or mismatch.get("corrected_entrypoint")
        != (
            "append_only_wrapper_injects_exact_project_root_then_imports_and_"
            "delegates_to_frozen_erratum2"
        )
        or mismatch.get("new_wrapper_must_be_bound_in_launch_intent_code_sha256")
        is not True
        or mismatch.get("import_failure_or_root_drift")
        != "FAIL_CLOSED_NO_LAUNCH"
        or unchanged.get("formal_container_name_reused_from_unexecuted_erratum2")
        is not True
        or unchanged.get("formal_intent_path_reused_from_unexecuted_erratum2")
        is not True
        or unchanged.get("physical_gpus") != list(PHYSICAL_GPUS)
        or unchanged.get("container_logical_ordinals") != {"2": 0, "3": 1}
        or unchanged.get("docker_device_ids") != ["2", "3"]
        or unchanged.get("container_nvidia_indices") != [0, 1]
        or unchanged.get("torch_ordinals") != [0, 1]
        or unchanged.get("gpu_fallback_allowed") is not False
        or unchanged.get("jobs_per_lane") != 9
        or unchanged.get("total_jobs") != 18
        or unchanged.get("result_use") != "failure_attribution_only"
        or unchanged.get("formal_v3_model_training_authorized") is not False
        or unchanged.get("source_gate_a_authorized") is not False
        or unchanged.get("riskcurve_authorized") is not False
        or unchanged.get("paper_claim_authorized") is not False
        or unchanged.get("outer_target_access_authorized") is not False
        or append_only.get("parent_execution_erratum2_mutated") is not False
        or append_only.get("parent_execution_erratum2_registration_mutated")
        is not False
        or append_only.get("parent_execution_erratum2_formal_intent_must_be_absent")
        is not True
        or append_only.get("historical_tier2r_hold_mutated") is not False
        or append_only.get("replacement_scope")
        != "direct_file_entrypoint_project_root_bootstrap_only"
    ):
        raise RuntimeError("Tier2S execution erratum3 semantics drift")
    return {
        "verified": True,
        "parent_execution_erratum2_binding": parent_binding,
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
        "decision": "AUTHORIZE_TIER2S_V2_GPU23_EXECUTION_ERRATUM3_SOURCE_ONLY",
        "parent_execution_erratum2_binding": verified[
            "parent_execution_erratum2_binding"
        ],
        "append_only_state": verified["append_only_state"],
        "execution_erratum_config": verified["execution_erratum_config"],
        "code_sha256": verified["code_sha256"],
        "execution_scope": {
            "entrypoint": (
                "python3 scripts/"
                "launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum3.py"
            ),
            "exact_project_root_bootstrap": str(PROJECT_ROOT),
            "delegates_to_frozen_erratum2": True,
            "physical_gpus": list(PHYSICAL_GPUS),
            "container_logical_ordinals": list(CONTAINER_LOGICAL_ORDINALS),
            "physical_to_container_ordinal": {"2": 0, "3": 1},
            "docker_device_ids": ["2", "3"],
            "container_nvidia_indices": [0, 1],
            "torch_ordinals": [0, 1],
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
            "direct_file_import_failure": "FAIL_CLOSED_NO_LAUNCH",
            "wrapper_not_bound_in_launch_intent": "FAIL_CLOSED_NO_LAUNCH",
            "parent_erratum2_formal_artifact_appears": (
                "FAIL_CLOSED_NEW_ERRATUM_REQUIRED"
            ),
        },
    }


def _validate_test_evidence(value: Any) -> None:
    parent_erratum2._validate_test_evidence(value)


def register() -> dict[str, Any]:
    if REGISTRATION.exists() or REGISTRATION_SIDECAR.exists():
        raise RuntimeError("Tier2S execution erratum3 already exists; never overwrite")
    before = verify_inputs()
    tests = parent_governance._run_full_tests()
    after = verify_inputs()
    if before != after:
        raise RuntimeError("Tier2S execution erratum3 input drifted during tests")
    _validate_test_evidence(tests)
    parent_erratum2.parent_erratum1._write_once(
        REGISTRATION,
        parent_erratum2.parent_erratum1._canonical_bytes(
            _registration_payload(after, tests)
        ),
    )
    digest = parent_erratum2.parent_erratum1._write_sidecar(
        REGISTRATION, REGISTRATION_SIDECAR
    )
    verified = verify_registration()
    if verified["registration_sha256"] != digest:
        raise RuntimeError("Tier2S execution erratum3 changed during final verification")
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
        raise RuntimeError("Tier2S execution erratum3 registration pair is incomplete")
    digest = parent_erratum2.parent_erratum1._verify_sidecar(
        REGISTRATION, REGISTRATION_SIDECAR
    )
    parent_erratum2.parent_erratum1._require_read_only(
        REGISTRATION, REGISTRATION_SIDECAR
    )
    current = verify_inputs()
    registration = _load_json(REGISTRATION)
    _validate_test_evidence(registration.get("full_test_gate"))
    execution = registration.get("execution_scope", {})
    authorization = registration.get("authorization", {})
    if (
        registration.get("schema_version") != SCHEMA
        or registration.get("decision")
        != "AUTHORIZE_TIER2S_V2_GPU23_EXECUTION_ERRATUM3_SOURCE_ONLY"
        or not isinstance(registration.get("registered_at"), str)
        or registration.get("parent_execution_erratum2_binding")
        != current["parent_execution_erratum2_binding"]
        or registration.get("append_only_state") != current["append_only_state"]
        or registration.get("execution_erratum_config")
        != current["execution_erratum_config"]
        or registration.get("code_sha256") != current["code_sha256"]
        or execution.get("exact_project_root_bootstrap") != str(PROJECT_ROOT)
        or execution.get("delegates_to_frozen_erratum2") is not True
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
        raise RuntimeError("frozen Tier2S execution erratum3 semantic drift")
    return {
        "verified": True,
        "registered": True,
        "registration_path": str(REGISTRATION),
        "registration_sha256": digest,
        "tier2s_v2_source_only_diagnostic_authorized": True,
        "formal_v3_model_training_authorized": False,
        "outer_target_access_authorized": False,
    }


def require_frozen_execution_erratum3(
    *, expected_registration_sha256: str | None = None
) -> dict[str, Any]:
    status = verify_registration()
    if status.get("registered") is not True:
        raise RuntimeError("Tier2S execution erratum3 is not registered")
    digest = str(status["registration_sha256"])
    if expected_registration_sha256 is not None and digest != expected_registration_sha256:
        raise RuntimeError("Tier2S execution erratum3 SHA-256 mismatch")
    registration = _load_json(REGISTRATION)
    parent_binding = registration["parent_execution_erratum2_binding"]
    code_map = registration["code_sha256"]
    return {
        "schema_version": BINDING_SCHEMA,
        "registration": parent_erratum2.parent_erratum1._frozen_binding(
            REGISTRATION, REGISTRATION_SIDECAR
        ),
        "parent_execution_erratum2_registration": parent_binding["registration"],
        "parent_execution_erratum1_registration": parent_binding[
            "parent_execution_erratum1_registration"
        ],
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
            parent_erratum2.parent_erratum1._canonical_bytes(code_map)
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

