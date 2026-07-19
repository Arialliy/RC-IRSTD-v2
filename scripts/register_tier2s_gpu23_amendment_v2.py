#!/usr/bin/env python3
"""Append-only governance amendment for the Tier2S GPU2/3 execution plan.

The original governance registration and the unexecuted GPU0/1 Tier2S v1
plan remain immutable.  This amendment freezes a separate protocol/code
closure that authorizes only the same source-only diagnostic on physical
GPUs 2 and 3.  It never authorizes V3 training, Gate A, RiskCurve, paper
claims, or outer-target access.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import register_aaai27_governance_v1 as parent_governance  # noqa: E402


SCHEMA = "rc-irstd-aaai27-tier2s-gpu23-governance-amendment-v2"
BINDING_SCHEMA = "rc-irstd-aaai27-tier2s-governance-binding-v2-gpu23"
PROTOCOL_ID = "tier2s_factorized_causal_audit_v2_gpu23"
PHYSICAL_GPUS = (2, 3)
CONTAINER_LOGICAL_ORDINALS = (0, 1)

PARENT_REGISTRATION_SHA256 = (
    "e177cf73db81a72d4a1acd1a50c1e87c3bfd33b295b21c0bf6d5c3f56064dedb"
)
PARENT_PROTOCOL = PROJECT_ROOT / "configs/tier2s_factorized_causal_audit_v1.json"
PARENT_PROTOCOL_SHA256 = (
    "9f0ec09e0289409b184e6dec1fe89b2a51ad5fc483f4f0ca5d27195173cdf129"
)
PARENT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/aaai27/source_rescue/tier2s_factorized_causal_audit_v1"
)
PARENT_AUDIT_ROOT = (
    PROJECT_ROOT
    / "artifacts/aaai27/audit/source_rescue/tier2s_factorized_causal_audit_v1"
)

PROTOCOL = PROJECT_ROOT / "configs/tier2s_factorized_causal_audit_v2_gpu23.json"
AMENDMENT_ROOT = (
    PROJECT_ROOT
    / "artifacts/aaai27/audit/governance/tier2s_gpu23_amendment_v2"
)
REGISTRATION = AMENDMENT_ROOT / "TIER2S_GPU23_GOVERNANCE_AMENDMENT.json"
REGISTRATION_SIDECAR = REGISTRATION.with_suffix(REGISTRATION.suffix + ".sha256")

CODE_PATHS = (
    "configs/tier2s_factorized_causal_audit_v2_gpu23.json",
    "scripts/register_tier2s_gpu23_amendment_v2.py",
    "scripts/coordinate_tier2s_factorized_audit_v2_gpu23.py",
    "scripts/evaluate_tier2s_factorized_audit_v2_gpu23.py",
    "scripts/export_tier2s_factorized_logits_v2_gpu23.py",
    "scripts/launch_tier2s_factorized_isolated_v2_gpu23.py",
    "scripts/probe_tier2s_factorized_container_v2_gpu23.py",
    "tests/test_tier2s_gpu23_v2.py",
    "configs/tier2s_factorized_causal_audit_v1.json",
    "artifacts/aaai27/audit/governance/aaai27_model_success_contract_v1/"
    "GOVERNANCE_REGISTRATION.json",
    "artifacts/aaai27/audit/governance/aaai27_model_success_contract_v1/"
    "GOVERNANCE_REGISTRATION.json.sha256",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required regular JSON is absent: {path}")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    def no_nonfinite(token: str) -> None:
        raise ValueError(f"non-finite JSON token {token!r}: {path}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=no_duplicates,
        parse_constant=no_nonfinite,
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _require_regular(path: Path, *, expected_sha256: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise RuntimeError(f"required canonical regular file is absent: {path}")
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise RuntimeError(f"SHA-256 drift: {path}")
    return path


def _require_read_only(*paths: Path) -> None:
    if any(stat.S_IMODE(path.stat().st_mode) & 0o222 for path in paths):
        raise RuntimeError("frozen Tier2S GPU2/3 amendment remains writable")


def _write_once(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"immutable amendment target already exists: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.pending.",
            delete=False,
        ) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, path)
        path.chmod(0o444)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_sidecar(path: Path, sidecar: Path) -> str:
    digest = _sha256(path)
    _write_once(sidecar, f"{digest}  {path.name}\n".encode("ascii"))
    return digest


def _verify_sidecar(path: Path, sidecar: Path) -> str:
    _require_regular(path)
    _require_regular(sidecar)
    digest = _sha256(path)
    if sidecar.read_bytes() != f"{digest}  {path.name}\n".encode("ascii"):
        raise RuntimeError(f"SHA-256 sidecar drift: {sidecar}")
    return digest


def _frozen_binding(path: Path, sidecar: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": _sha256(path),
        "sidecar_path": str(sidecar.relative_to(PROJECT_ROOT)),
        "sidecar_sha256": _sha256(sidecar),
    }


def _code_sha256() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in CODE_PATHS:
        path = _require_regular(PROJECT_ROOT / relative)
        result[relative] = _sha256(path)
    return result


def _verify_parent_plan_is_unexecuted() -> dict[str, Any]:
    if PARENT_OUTPUT_ROOT.exists() or PARENT_OUTPUT_ROOT.is_symlink():
        raise RuntimeError("GPU0/1 Tier2S v1 output namespace already exists")
    if PARENT_AUDIT_ROOT.exists() or PARENT_AUDIT_ROOT.is_symlink():
        raise RuntimeError("GPU0/1 Tier2S v1 audit namespace already exists")
    _require_regular(PARENT_PROTOCOL, expected_sha256=PARENT_PROTOCOL_SHA256)
    return {
        "protocol_path": str(PARENT_PROTOCOL.relative_to(PROJECT_ROOT)),
        "protocol_sha256": PARENT_PROTOCOL_SHA256,
        "output_namespace_absent": True,
        "audit_namespace_absent": True,
        "historical_evidence_superseded": False,
        "unexecuted_hardware_plan_replaced_only": True,
    }


def verify_inputs() -> dict[str, Any]:
    parent = parent_governance.require_frozen_tier2s_governance(
        expected_registration_sha256=PARENT_REGISTRATION_SHA256
    )
    protocol = _load_json(PROTOCOL)
    execution = protocol.get("execution", {})
    migration = protocol.get("gpu_plan_migration", {})
    if (
        protocol.get("schema_version")
        != "rc-irstd-aaai27-tier2s-factorized-causal-audit-protocol-v2-gpu23"
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("research_mode") != "exploratory_source_only"
        or execution.get("physical_gpus") != list(PHYSICAL_GPUS)
        or execution.get("container_logical_ordinals")
        != {"2": 0, "3": 1}
        or execution.get("export_jobs") != 18
        or execution.get("allow_gpu_fallback") is not False
        or migration.get("parent_protocol_sha256") != PARENT_PROTOCOL_SHA256
        or migration.get("old_v1_evidence_mutated") is not False
        or migration.get("replacement_scope")
        != "unexecuted_physical_gpu_schedule_only"
    ):
        raise RuntimeError("Tier2S GPU2/3 protocol semantics drift")
    limits = protocol.get("scientific_limits", {})
    if (
        limits.get("result_use") != "failure_attribution_only"
        or limits.get("may_authorize_v3_implementation") is not False
        or limits.get("outer_target_access_authorized") is not False
        or limits.get(
            "scaling_alpha_tail_coordinate_or_calibration_may_be_claimed_as_innovation"
        )
        is not False
    ):
        raise RuntimeError("Tier2S GPU2/3 scientific limits drift")
    return {
        "verified": True,
        "parent_governance_binding": parent,
        "superseded_execution_plan": _verify_parent_plan_is_unexecuted(),
        "protocol": {
            "path": str(PROTOCOL.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(PROTOCOL),
        },
        "code_sha256": _code_sha256(),
        "physical_gpus": list(PHYSICAL_GPUS),
        "container_logical_ordinals": list(CONTAINER_LOGICAL_ORDINALS),
    }


def _registration_payload(
    verified: Mapping[str, Any], test_evidence: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "registered_at": _now(),
        "decision": "AUTHORIZE_TIER2S_V2_SOURCE_ONLY_ON_PHYSICAL_GPU2_GPU3",
        "parent_governance_binding": verified["parent_governance_binding"],
        "superseded_execution_plan": verified["superseded_execution_plan"],
        "protocol": verified["protocol"],
        "code_sha256": verified["code_sha256"],
        "execution_scope": {
            "physical_gpus": list(PHYSICAL_GPUS),
            "container_logical_ordinals": list(CONTAINER_LOGICAL_ORDINALS),
            "physical_to_container_ordinal": {"2": 0, "3": 1},
            "two_fixed_independent_fifo_lanes": True,
            "jobs_per_lane": 9,
            "total_jobs": 18,
            "gpu_fallback_allowed": False,
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
            "any_code_protocol_or_parent_sha_drift": "FAIL_CLOSED_NO_TIER2S_V2_LAUNCH",
            "gpu2_or_gpu3_not_idle": "FAIL_CLOSED_NO_PREFLIGHT_OR_LAUNCH",
            "physical_uuid_or_container_ordinal_drift": "FAIL_CLOSED_NO_LAUNCH",
            "old_v1_namespace_appears": "FAIL_CLOSED_NEW_AMENDMENT_REQUIRED",
        },
    }


def _validate_test_evidence(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise RuntimeError("Tier2S GPU2/3 full-test evidence is absent")
    summary = value.get("summary")
    if (
        value.get("returncode") != 0
        or not isinstance(summary, str)
        or re.search(r"\b\d+ passed\b", summary) is None
        or re.search(r"\b(?:failed|error|errors)\b", summary) is not None
        or value.get("pytest_plugin_autoload_disabled") is not True
        or value.get("pytest_cacheprovider_disabled") is not True
        or value.get("temporary_pycache") is not True
    ):
        raise RuntimeError("Tier2S GPU2/3 full-test evidence drift")


def register() -> dict[str, Any]:
    if REGISTRATION.exists() or REGISTRATION_SIDECAR.exists():
        raise RuntimeError("Tier2S GPU2/3 amendment already exists; never overwrite")
    before = verify_inputs()
    tests = parent_governance._run_full_tests()
    after = verify_inputs()
    if before != after:
        raise RuntimeError("Tier2S GPU2/3 amendment input drifted during tests")
    _validate_test_evidence(tests)
    _write_once(REGISTRATION, _canonical_bytes(_registration_payload(after, tests)))
    digest = _write_sidecar(REGISTRATION, REGISTRATION_SIDECAR)
    verified = verify_registration()
    if verified["registration_sha256"] != digest:
        raise RuntimeError("Tier2S GPU2/3 amendment changed during final verification")
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
        raise RuntimeError("Tier2S GPU2/3 amendment/sidecar pair is incomplete")
    digest = _verify_sidecar(REGISTRATION, REGISTRATION_SIDECAR)
    _require_read_only(REGISTRATION, REGISTRATION_SIDECAR)
    current = verify_inputs()
    registration = _load_json(REGISTRATION)
    _validate_test_evidence(registration.get("full_test_gate"))
    expected_parent = current["parent_governance_binding"]
    authorization = registration.get("authorization", {})
    execution = registration.get("execution_scope", {})
    if (
        registration.get("schema_version") != SCHEMA
        or registration.get("decision")
        != "AUTHORIZE_TIER2S_V2_SOURCE_ONLY_ON_PHYSICAL_GPU2_GPU3"
        or not isinstance(registration.get("registered_at"), str)
        or registration.get("parent_governance_binding") != expected_parent
        or registration.get("superseded_execution_plan")
        != current["superseded_execution_plan"]
        or registration.get("protocol") != current["protocol"]
        or registration.get("code_sha256") != current["code_sha256"]
        or execution.get("physical_gpus") != list(PHYSICAL_GPUS)
        or execution.get("container_logical_ordinals")
        != list(CONTAINER_LOGICAL_ORDINALS)
        or execution.get("physical_to_container_ordinal") != {"2": 0, "3": 1}
        or execution.get("jobs_per_lane") != 9
        or execution.get("total_jobs") != 18
        or execution.get("gpu_fallback_allowed") is not False
        or authorization.get("tier2s_v2_source_only_diagnostic_authorized") is not True
        or authorization.get("formal_v3_model_training_authorized") is not False
        or authorization.get("source_gate_a_authorized") is not False
        or authorization.get("riskcurve_authorized") is not False
        or authorization.get("paper_claim_authorized") is not False
        or authorization.get("outer_target_access_authorized") is not False
    ):
        raise RuntimeError("frozen Tier2S GPU2/3 amendment semantic drift")
    return {
        "verified": True,
        "registered": True,
        "registration_path": str(REGISTRATION),
        "registration_sha256": digest,
        "tier2s_v2_source_only_diagnostic_authorized": True,
        "formal_v3_model_training_authorized": False,
        "outer_target_access_authorized": False,
    }


def require_frozen_tier2s_governance(
    *, expected_registration_sha256: str | None = None
) -> dict[str, Any]:
    status = verify_registration()
    if status.get("registered") is not True:
        raise RuntimeError("Tier2S GPU2/3 governance amendment is not registered")
    digest = str(status["registration_sha256"])
    if expected_registration_sha256 is not None and digest != expected_registration_sha256:
        raise RuntimeError("Tier2S GPU2/3 amendment SHA-256 mismatch")
    registration = _load_json(REGISTRATION)
    parent = registration["parent_governance_binding"]
    code_map = registration["code_sha256"]
    return {
        "schema_version": BINDING_SCHEMA,
        "registration": _frozen_binding(REGISTRATION, REGISTRATION_SIDECAR),
        "parent_governance_registration": parent["registration"],
        "contract": parent["contract"],
        "fresh_seed_ledger": parent["fresh_seed_ledger"],
        "fresh_seed_local_scan": parent["fresh_seed_local_scan"],
        "code_sha256_canonical_sha256": hashlib.sha256(
            _canonical_bytes(code_map)
        ).hexdigest(),
        "physical_gpus": list(PHYSICAL_GPUS),
        "container_logical_ordinals": list(CONTAINER_LOGICAL_ORDINALS),
        "physical_to_container_ordinal": {"2": 0, "3": 1},
        "gpu_fallback_allowed": False,
        "superseded_execution_plan": registration["superseded_execution_plan"],
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
