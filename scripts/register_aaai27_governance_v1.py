#!/usr/bin/env python3
"""Register the frozen AAAI-27 success criteria and Tier2S diagnostic code.

This registrar is deliberately narrower than a model-training preregistration.
It freezes the reviewed success contract, the fresh-seed reservation ledger,
the complete Tier2S/governance code closure, and a sanitized full-test result.
It authorizes only the source-only Tier2S diagnostic.  It never authorizes V3
training, source Gate A, RiskCurve, or any new NUAA access.
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
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "configs/aaai27_model_success_contract_v1.json"
CONTRACT_SIDECAR = CONTRACT_PATH.with_suffix(CONTRACT_PATH.suffix + ".sha256")
AUDIT_ROOT = (
    PROJECT_ROOT
    / "artifacts/aaai27/audit/governance/aaai27_model_success_contract_v1"
)
LEDGER_PATH = AUDIT_ROOT / "FRESH_SEED_LEDGER.json"
LEDGER_SIDECAR = LEDGER_PATH.with_suffix(LEDGER_PATH.suffix + ".sha256")
LOCAL_SCAN_PATH = AUDIT_ROOT / "FRESH_SEED_LOCAL_SCAN_V2.json"
LOCAL_SCAN_SIDECAR = LOCAL_SCAN_PATH.with_suffix(LOCAL_SCAN_PATH.suffix + ".sha256")
SUPERSEDED_LOCAL_SCAN_PATH = AUDIT_ROOT / "FRESH_SEED_LOCAL_SCAN.json"
SUPERSEDED_LOCAL_SCAN_SIDECAR = SUPERSEDED_LOCAL_SCAN_PATH.with_suffix(
    SUPERSEDED_LOCAL_SCAN_PATH.suffix + ".sha256"
)
SUPERSEDED_LOCAL_SCAN_SHA256 = (
    "64d2ed4a878a51a29ee73eb66171410dad785dfd8f975b6665b21aee10bd2cf6"
)
SUPERSEDED_LOCAL_SCAN_SIDECAR_SHA256 = (
    "e247b8556f21beb022a407929c99349ef873b9eaa29e0d3244d01c45727e994f"
)
REGISTRATION_PATH = AUDIT_ROOT / "GOVERNANCE_REGISTRATION.json"
REGISTRATION_SIDECAR = REGISTRATION_PATH.with_suffix(
    REGISTRATION_PATH.suffix + ".sha256"
)
PARENT_PREREGISTRATION = (
    PROJECT_ROOT
    / "artifacts/aaai27/audit/component_rescue/tier2r_c_v1"
    / "COMPONENT_RESCUE_PREREGISTRATION.json"
)
HISTORICAL_DECISION = (
    PROJECT_ROOT
    / "artifacts/aaai27/audit/component_rescue/tier2r_c_v1_impl_erratum1"
    / "exact_gate/COMPONENT_RESCUE_DECISION.json"
)
TARGET_EXPOSURE_REGISTRY = (
    PROJECT_ROOT / "artifacts/aaai27/audit/target_label_exposure_registry.json"
)

SCHEMA = "rc-irstd-aaai27-governance-registration-v1"
FRESH_SEEDS = (46, 47, 48, 49, 50)
DEVELOPMENT_SEEDS = (42, 43, 44, 45)
FRESH_SEED_PATH_RE = re.compile(r"(?i)(?:^|[^a-z0-9])seed[ _-]?(46|47|48|49|50)(?:[^0-9]|$)")
FRESH_SEED_SCANNER_RELATIVE = "scripts/audit_fresh_seed_reservation_v1.py"
FRESH_SEED_SCANNER_PATH = PROJECT_ROOT / FRESH_SEED_SCANNER_RELATIVE
REGISTRATION_TIME_RESCAN_NORMALIZATION = (
    "canonical_json_v1_with_generated_at_removed"
)

ADDITIONAL_CODE_PATHS = (
    "configs/aaai27_model_success_contract_v1.json",
    "configs/tier2s_factorized_causal_audit_v1.json",
    "scripts/register_aaai27_governance_v1.py",
    "scripts/audit_fresh_seed_reservation_v1.py",
    "scripts/coordinate_tier2s_factorized_audit.py",
    "scripts/evaluate_tier2s_factorized_audit.py",
    "scripts/export_tier2s_factorized_logits.py",
    "scripts/launch_tier2s_factorized_isolated.py",
    "scripts/probe_tier2s_factorized_container.py",
    "scripts/diagnose_component_fusion.py",
    "evaluation/budget_metrics.py",
    "evaluation/raw_logit_rescue_diagnostics.py",
    "evaluation/raw_logit_rescue_gate.py",
    "evaluation/__init__.py",
    "rc_irstd/__init__.py",
    "rc_irstd/evaluation/__init__.py",
    "rc_irstd/evaluation/domain_contract.py",
    "rc_irstd/evaluation/metrics.py",
    "rc_irstd/evaluation/score_store.py",
    "rc_irstd/features/__init__.py",
    "rc_irstd/features/domain_statistics.py",
    "rc_irstd/utils/__init__.py",
    "rc_irstd/utils/io.py",
    "risk_curve/__init__.py",
    "risk_curve/monotone_curve_predictor.py",
    "risk_curve/representation.py",
    "scripts/launch_phase3_tier2r_isolated.py",
    "scripts/probe_phase3_tier2r_container.py",
    "scripts/probe_phase3_tier2r_impl_erratum1_container.py",
    "tests/test_aaai27_model_success_contract.py",
    "tests/test_aaai27_governance_registration.py",
    "tests/test_fresh_seed_reservation_audit.py",
    "tests/test_phase3_tier2r_startup_fix1.py",
    "tests/test_tier2s_factorized_audit.py",
    "tests/test_tier2s_factorized_coordinator.py",
    "tests/test_tier2s_factorized_export.py",
    "tests/test_tier2s_factorized_isolated_launcher.py",
    "artifacts/aaai27/audit/governance/aaai27_model_success_contract_v1/FRESH_SEED_LEDGER.json",
    "artifacts/aaai27/audit/governance/aaai27_model_success_contract_v1/FRESH_SEED_LOCAL_SCAN.json",
    "artifacts/aaai27/audit/governance/aaai27_model_success_contract_v1/FRESH_SEED_LOCAL_SCAN_V2.json",
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
        raise RuntimeError(f"required regular JSON file is absent: {path}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {token!r}: {path}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _relative_regular(relative: str, *, expected_sha256: str | None = None) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise RuntimeError(f"path must be project-relative without traversal: {relative}")
    path = PROJECT_ROOT / raw
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise RuntimeError(f"required canonical regular file is absent: {path}")
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise RuntimeError(f"SHA-256 drift: {relative}")
    return path


def _verify_contract() -> dict[str, Any]:
    contract = _load_json(CONTRACT_PATH)
    lifecycle = contract.get("lifecycle", {})
    historical = contract.get("historical_evidence_anchor", {})
    exposure = contract.get("historical_target_exposure", {})
    fresh = contract.get("fresh5_confirmation", {})
    gate_a = contract.get("tiers", {}).get("hard", {}).get("gate_a", {})
    if (
        contract.get("schema_version")
        != "rc-irstd-aaai27-model-success-contract-v1"
        or contract.get("contract_id") != "aaai27_model_success_contract_v1"
        or lifecycle.get("state") != "reviewed_frozen_success_criteria"
        or lifecycle.get("formal_training_authorized_by_this_file") is not False
        or lifecycle.get("success_criteria_freeze_does_not_authorize_model_training")
        is not True
        or lifecycle.get(
            "formal_model_training_requires_separate_versioned_v3_preregistration"
        )
        is not True
        or historical.get("decision") != "TIER2R_HOLD"
        or historical.get("selected_candidate") is not None
        or historical.get("component_claim_retained") is not False
        or exposure.get("nuaa_official_test_labels_previously_viewed") is not True
        or exposure.get("untouched_claim_allowed") is not False
        or tuple(fresh.get("formal_seed_ids", ())) != FRESH_SEEDS
        or tuple(fresh.get("known_excluded_seed_ids", ())) != DEVELOPMENT_SEEDS
        or gate_a.get("required_result") != "9_of_9_GO"
        or gate_a.get("evaluated_on_core_model_float32_raw_logits_before_riskcurve")
        is not True
        or gate_a.get(
            "riskcurve_calibration_or_system_gain_may_compensate_gate_a_failure"
        )
        is not False
    ):
        raise RuntimeError("success-contract semantics drift")

    _relative_regular(
        str(historical["protocol_path"]),
        expected_sha256=str(historical["protocol_sha256"]),
    )
    _relative_regular(
        str(historical["decision_path"]),
        expected_sha256=str(historical["decision_sha256"]),
    )
    _relative_regular(
        str(exposure["registry_path"]),
        expected_sha256=str(exposure["registry_sha256"]),
    )
    return contract


def _reserved_seed_path_hits(
    roots: Sequence[Path] | None = None,
) -> list[str]:
    scan_roots = tuple(
        roots
        or (
            PROJECT_ROOT / "artifacts/aaai27",
            PROJECT_ROOT / "outputs/aaai27",
        )
    )
    hits: list[str] = []
    for root in scan_roots:
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError(f"seed scan root is not a canonical directory: {root}")
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = sorted(
                name
                for name in dirnames
                if not (Path(directory) / name).is_symlink()
            )
            for name in sorted(filenames):
                path = Path(directory) / name
                try:
                    relative = path.relative_to(PROJECT_ROOT)
                except ValueError:
                    relative = path
                if FRESH_SEED_PATH_RE.search(str(relative)):
                    hits.append(str(relative))
    return sorted(set(hits))


def _verify_source_split_binding(binding: Mapping[str, Any]) -> None:
    path = _relative_regular(
        str(binding["manifest"]),
        expected_sha256=str(binding["manifest_sha256"]),
    )
    ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not ids or len(ids) != len(set(ids)):
        raise RuntimeError(f"source split IDs are empty or duplicated: {path}")
    compact_json = json.dumps(
        ids, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    linefeed = "".join(f"{value}\n" for value in ids).encode("utf-8")
    expected = {
        "ordered_ids_sha256_schema":
            "sha256_utf8_compact_json_array_v1_no_trailing_newline",
        "ordered_ids_sha256": hashlib.sha256(compact_json).hexdigest(),
        "ordered_ids_linefeed_sha256_schema":
            "sha256_concatenated_utf8_id_plus_lf_v1",
        "ordered_ids_linefeed_sha256": hashlib.sha256(linefeed).hexdigest(),
        "unique_ids": True,
        "num_samples": len(ids),
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            raise RuntimeError(f"source split binding drift: {path}:{key}")


def _validate_local_scan_payload(report: Mapping[str, Any], *, context: str) -> None:
    counts = report.get("counts", {})
    checkpoint = report.get("checkpoint_audit", {})
    limitations = report.get("limitations", ())
    if (
        report.get("schema_version")
        != "rc-irstd-aaai27-fresh-seed-local-audit-v1"
        or report.get("algorithm_version")
        != "fresh-seed-local-filesystem-scan-v2"
        or tuple(report.get("fresh_seed_ids", ())) != FRESH_SEEDS
        or report.get("decision")
        != "PASS_NO_LOCAL_CONSUMPTION_EVIDENCE_FOUND"
        or report.get("local_scan_complete") is not True
        or report.get("local_scan_passed") is not True
        or report.get("freshness_certified") is not False
        or report.get("author_attestation_required") is not True
        or counts.get("consumption_like_hits") != 0
        or report.get("scan_errors") != []
        or report.get("unscanned_symlinks") != []
        or report.get("structured_json_parse_failures") != []
        or checkpoint.get("content_parsing_performed") is not False
        or checkpoint.get("path_seed_token_detection") is not True
        or not isinstance(limitations, list)
        or not any("deleted artifacts" in str(value) for value in limitations)
        or not any("other machines" in str(value) for value in limitations)
    ):
        raise RuntimeError(f"fresh-seed local scan semantics drift: {context}")


def _run_current_local_scan() -> dict[str, Any]:
    command = [
        sys.executable,
        str(FRESH_SEED_SCANNER_PATH),
        "--root",
        str(PROJECT_ROOT),
        "--reservation-path",
        str(LOCAL_SCAN_PATH.relative_to(PROJECT_ROOT)),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "current fresh-seed local scan failed closed: "
            + (completed.stderr.strip() or f"returncode={completed.returncode}")
        )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("current fresh-seed scan output is not JSON") from error
    if not isinstance(report, dict):
        raise RuntimeError("current fresh-seed scan root is not an object")
    _validate_local_scan_payload(report, context="current_workspace_rescan")
    return report


def _registration_time_rescan_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    _validate_local_scan_payload(report, context="registration_time_rescan")
    normalized = dict(report)
    normalized.pop("generated_at", None)
    counts = report.get("counts", {})
    scanner = report.get("scanner", {})
    git = report.get("git", {})
    return {
        "schema_version":
            "rc-irstd-aaai27-fresh-seed-registration-time-rescan-summary-v1",
        "decision": report["decision"],
        "local_scan_complete": True,
        "local_scan_passed": True,
        "freshness_certified": False,
        "author_attestation_required": True,
        "consumption_like_hits": counts.get("consumption_like_hits"),
        "scan_errors": len(report.get("scan_errors", ())),
        "unscanned_symlinks": len(report.get("unscanned_symlinks", ())),
        "structured_json_parse_failures": len(
            report.get("structured_json_parse_failures", ())
        ),
        "regular_files_seen": counts.get("regular_files_seen"),
        "text_files_scanned": counts.get("text_files_scanned"),
        "text_bytes_scanned": counts.get("text_bytes_scanned"),
        "checkpoint_like_files": counts.get("checkpoint_like_files"),
        "git_head_at_rescan": git.get("head"),
        "scanner_path": scanner.get("path"),
        "scanner_sha256": scanner.get("sha256"),
        "normalized_report_sha256": hashlib.sha256(
            _canonical_bytes(normalized)
        ).hexdigest(),
        "normalization": REGISTRATION_TIME_RESCAN_NORMALIZATION,
        "local_only_not_freshness_certificate": True,
        "external_deleted_author_attestation_still_required": True,
    }


def _validate_registration_time_rescan_summary(summary: Any) -> None:
    zero_count_fields = (
        "consumption_like_hits",
        "scan_errors",
        "unscanned_symlinks",
        "structured_json_parse_failures",
    )
    positive_count_fields = (
        "regular_files_seen",
        "text_files_scanned",
    )
    nonnegative_count_fields = (
        "text_bytes_scanned",
        "checkpoint_like_files",
    )
    count_fields = (
        *zero_count_fields,
        *positive_count_fields,
        *nonnegative_count_fields,
    )
    if not isinstance(summary, Mapping):
        raise RuntimeError("registration-time fresh-seed rescan evidence drift")

    counts_have_exact_int_type = all(
        type(summary.get(field)) is int for field in count_fields
    )
    if not counts_have_exact_int_type:
        raise RuntimeError("registration-time fresh-seed rescan evidence drift")

    regular_files_seen = summary["regular_files_seen"]
    text_files_scanned = summary["text_files_scanned"]
    text_bytes_scanned = summary["text_bytes_scanned"]
    checkpoint_like_files = summary["checkpoint_like_files"]
    git_head = summary.get("git_head_at_rescan")
    scanner_sha256 = summary.get("scanner_sha256")
    normalized_report_sha256 = summary.get("normalized_report_sha256")
    expected_scanner = _relative_regular(FRESH_SEED_SCANNER_RELATIVE)
    expected_scanner_sha256 = _sha256(expected_scanner)

    if (
        summary.get("schema_version")
        != "rc-irstd-aaai27-fresh-seed-registration-time-rescan-summary-v1"
        or summary.get("decision")
        != "PASS_NO_LOCAL_CONSUMPTION_EVIDENCE_FOUND"
        or summary.get("local_scan_complete") is not True
        or summary.get("local_scan_passed") is not True
        or summary.get("freshness_certified") is not False
        or summary.get("author_attestation_required") is not True
        or any(summary[field] != 0 for field in zero_count_fields)
        or regular_files_seen <= 0
        or text_files_scanned <= 0
        or text_bytes_scanned < 0
        or checkpoint_like_files < 0
        or text_files_scanned > regular_files_seen
        or checkpoint_like_files > regular_files_seen
        or type(git_head) is not str
        or re.fullmatch(r"[0-9a-f]{40,64}", git_head) is None
        or summary.get("scanner_path") != FRESH_SEED_SCANNER_RELATIVE
        or type(scanner_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", scanner_sha256) is None
        or scanner_sha256 != expected_scanner_sha256
        or type(normalized_report_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", normalized_report_sha256) is None
        or summary.get("normalization")
        != REGISTRATION_TIME_RESCAN_NORMALIZATION
        or summary.get("local_only_not_freshness_certificate") is not True
        or summary.get("external_deleted_author_attestation_still_required")
        is not True
    ):
        raise RuntimeError("registration-time fresh-seed rescan evidence drift")


def _verify_fresh_seed_local_scan(status: Mapping[str, Any]) -> dict[str, Any]:
    expected_relative = str(LOCAL_SCAN_PATH.relative_to(PROJECT_ROOT))
    expected_sidecar_relative = str(LOCAL_SCAN_SIDECAR.relative_to(PROJECT_ROOT))
    expected_sha256 = status.get("local_machine_scan_report_sha256")
    if (
        status.get("local_machine_scan_report_path") != expected_relative
        or status.get("local_machine_scan_report_sidecar_path")
        != expected_sidecar_relative
        or not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise RuntimeError("fresh-seed local scan ledger binding is invalid")
    report_path = _relative_regular(
        expected_relative, expected_sha256=expected_sha256
    )
    report_sha256 = _verify_digest_sidecar(report_path, LOCAL_SCAN_SIDECAR)
    _require_read_only(report_path, LOCAL_SCAN_SIDECAR)
    report = _load_json(report_path)
    _validate_local_scan_payload(report, context="frozen_report")
    if (
        status.get("local_machine_scan_report_sidecar_sha256")
        != _sha256(LOCAL_SCAN_SIDECAR)
        or status.get("local_machine_scan_decision") != report.get("decision")
        or status.get("local_machine_scan_generated_at")
        != report.get("generated_at")
        or status.get("local_machine_scan_git_head")
        != report.get("git", {}).get("head")
        or status.get("local_machine_scan_complete") is not True
        or status.get("local_machine_scan_consumption_like_hits") != 0
        or status.get("local_machine_scan_is_current_filesystem_snapshot_only")
        is not True
        or status.get("local_scan_report_precedes_ledger_self_binding_by_design")
        is not True
        or status.get("registration_time_current_workspace_rescan_required")
        is not True
    ):
        raise RuntimeError("fresh-seed ledger/report metadata drift")
    scanner = report.get("scanner", {})
    scanner_path = _relative_regular(
        str(scanner.get("path", "")),
        expected_sha256=str(scanner.get("sha256", "")),
    )
    if scanner_path != FRESH_SEED_SCANNER_PATH:
        raise RuntimeError("fresh-seed scanner identity drift")
    return {
        "path": expected_relative,
        "sha256": report_sha256,
        "sidecar_path": str(LOCAL_SCAN_SIDECAR.relative_to(PROJECT_ROOT)),
        "sidecar_sha256": _sha256(LOCAL_SCAN_SIDECAR),
        "decision": report["decision"],
        "frozen_scan_git_head": report.get("git", {}).get("head"),
        "local_only_not_freshness_certificate": True,
        "external_deleted_author_attestation_still_required": True,
    }


def _verify_ledger(contract: Mapping[str, Any]) -> dict[str, Any]:
    ledger = _load_json(LEDGER_PATH)
    scope = ledger.get("scope", {})
    status = ledger.get("reserved_seed_status", {})
    contract_fresh = contract.get("fresh5_confirmation", {})
    reserved = tuple(ledger.get("reserved_formal_seeds", ()))
    historical_rows = ledger.get("historical_development_seeds")
    historical = tuple(
        int(row["seed"])
        for row in historical_rows
        if isinstance(row, Mapping) and "seed" in row
    ) if isinstance(historical_rows, list) else ()
    if (
        ledger.get("schema_version") != "rc-irstd-aaai27-fresh-seed-ledger-v1"
        or ledger.get("lifecycle", {}).get("formal_model_training_authorized")
        is not False
        or scope.get("formal_model_scope")
        != "one_unique_v3_core_mechanism_with_no_component_extension_arm"
        or scope.get("gate_b_included") is not False
        or scope.get("future_gate_b_requires_new_contract_ledger_and_unseen_seeds")
        is not True
        or reserved != FRESH_SEEDS
        or historical != DEVELOPMENT_SEEDS
        or set(reserved) & set(historical)
        or status.get("observed_completed_checkpoint_metric_or_run_artifacts") != 0
        or status.get("local_machine_scan_required") is not True
        or status.get("local_machine_scan_report_revision") != 2
        or status.get("local_machine_scan_algorithm_version")
        != "fresh-seed-local-filesystem-scan-v2"
        or status.get("repository_external_deleted_artifact_coverage") is not False
        or status.get("dated_author_attestation_required_before_v3_training") is not True
        or status.get("dated_author_attestation_present") is not False
        or status.get("formal_v3_training_eligible") is not False
        or ledger.get("freshness_rules", {}).get("seed_screening_allowed")
        is not False
        or ledger.get("freshness_rules", {}).get("selective_reporting_allowed")
        is not False
        or ledger.get("freshness_rules", {}).get(
            "pretraining_smoke_or_debug_with_reserved_seeds_allowed"
        )
        is not False
        or contract_fresh.get("freshness_ledger_path")
        != str(LEDGER_PATH.relative_to(PROJECT_ROOT))
        or contract_fresh.get("gate_b_applicable_under_current_contract") is not False
        or contract_fresh.get(
            "complete_gate_a_formal_arm_matrix_must_be_frozen_before_any_reserved_seed_is_consumed"
        )
        is not True
    ):
        raise RuntimeError("fresh-seed ledger semantics drift")

    revision = ledger.get("pre_registration_local_scan_revision", {})
    superseded = revision.get("superseded_candidate", {})
    if (
        revision.get("revision") != 2
        or revision.get("replacement_algorithm_version")
        != "fresh-seed-local-filesystem-scan-v2"
        or superseded.get("path")
        != str(SUPERSEDED_LOCAL_SCAN_PATH.relative_to(PROJECT_ROOT))
        or superseded.get("sha256") != SUPERSEDED_LOCAL_SCAN_SHA256
        or superseded.get("sidecar_sha256")
        != SUPERSEDED_LOCAL_SCAN_SIDECAR_SHA256
        or superseded.get("status")
        != "REJECTED_PRE_REGISTRATION_NEVER_AUTHORIZED_TIER2S_OR_V3"
    ):
        raise RuntimeError("pre-registration fresh-seed scan revision drift")
    _relative_regular(
        str(superseded["path"]), expected_sha256=SUPERSEDED_LOCAL_SCAN_SHA256
    )
    if (
        _verify_digest_sidecar(
            SUPERSEDED_LOCAL_SCAN_PATH, SUPERSEDED_LOCAL_SCAN_SIDECAR
        )
        != SUPERSEDED_LOCAL_SCAN_SHA256
        or _sha256(SUPERSEDED_LOCAL_SCAN_SIDECAR)
        != SUPERSEDED_LOCAL_SCAN_SIDECAR_SHA256
    ):
        raise RuntimeError("superseded fresh-seed scan evidence drift")
    _require_read_only(
        SUPERSEDED_LOCAL_SCAN_PATH, SUPERSEDED_LOCAL_SCAN_SIDECAR
    )

    for row in historical_rows:
        for evidence in row.get("evidence", []):
            _relative_regular(
                str(evidence["path"]),
                expected_sha256=str(evidence["sha256"]),
            )
    for binding in ledger.get("source_split_bindings", {}).values():
        _verify_source_split_binding(binding)
    target = ledger.get("target_exposure_binding", {})
    _relative_regular(
        str(target["path"]), expected_sha256=str(target["sha256"])
    )
    hits = _reserved_seed_path_hits()
    if hits:
        raise RuntimeError(
            "reserved fresh seed already appears in an experimental artifact path: "
            + ",".join(hits)
        )
    return ledger


def _code_paths() -> tuple[str, ...]:
    parent = _load_json(PARENT_PREREGISTRATION)
    bindings = parent.get("code_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise RuntimeError("historical code closure is absent")
    paths = set(str(value) for value in bindings)
    paths.update(ADDITIONAL_CODE_PATHS)
    for relative in sorted(paths):
        _relative_regular(relative)
    return tuple(sorted(paths))


def _code_sha256() -> dict[str, str]:
    return {
        relative: _sha256(PROJECT_ROOT / relative)
        for relative in _code_paths()
    }


def verify_inputs(*, run_current_scan: bool = True) -> dict[str, Any]:
    contract = _verify_contract()
    ledger = _verify_ledger(contract)
    historical = _load_json(HISTORICAL_DECISION)
    exposure = _load_json(TARGET_EXPOSURE_REGISTRY)
    if (
        historical.get("decision") != "TIER2R_HOLD"
        or historical.get("authorizes_outer_target_access") is not False
        or exposure.get("datasets", {})
        .get("NUAA-SIRST", {})
        .get("official_test_labels_previously_viewed")
        is not True
    ):
        raise RuntimeError("historical HOLD or target exposure semantics drift")
    local_scan = _verify_fresh_seed_local_scan(
        ledger.get("reserved_seed_status", {})
    )
    result = {
        "verified": True,
        "contract_sha256": _sha256(CONTRACT_PATH),
        "ledger_sha256": _sha256(LEDGER_PATH),
        "local_scan_sha256": local_scan["sha256"],
        "local_scan_binding": local_scan,
        "code_sha256": _code_sha256(),
        "reserved_seed_path_hits": [],
        "formal_model_training_authorized": False,
        "outer_target_access_authorized": False,
        "tier2s_source_only_diagnostic_eligible": True,
        "contract": contract,
        "ledger": ledger,
    }
    if run_current_scan:
        result["registration_time_rescan_summary"] = (
            _registration_time_rescan_summary(_run_current_local_scan())
        )
    return result


def _git_snapshot() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    return {
        "head": head,
        "branch": branch,
        "worktree_clean": status == "",
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "status_num_lines": len(status.splitlines()),
        "code_identity_uses_explicit_file_sha256_not_clean_worktree": True,
    }


def _run_full_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    environment = dict(os.environ)
    environment.pop("PYTEST_ADDOPTS", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONPYCACHEPREFIX"] = "/tmp/rc_irstd_governance_registration_pycache"
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    summary_lines = [
        line.strip(" =")
        for line in combined.splitlines()
        if re.search(r"\b(?:passed|failed|error|errors)\b", line)
    ]
    summary = summary_lines[-1] if summary_lines else ""
    if (
        completed.returncode != 0
        or not re.search(r"\b\d+ passed\b", summary)
        or re.search(r"\b(?:failed|error|errors)\b", summary)
    ):
        raise RuntimeError(
            "full governance test gate failed: "
            + (summary or f"returncode={completed.returncode}")
        )
    return {
        "command": command,
        "returncode": completed.returncode,
        "summary": summary,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_sha256": hashlib.sha256(
            completed.stdout.encode("utf-8")
        ).hexdigest(),
        "stderr_sha256": hashlib.sha256(
            completed.stderr.encode("utf-8")
        ).hexdigest(),
        "pytest_plugin_autoload_disabled": True,
        "pytest_cacheprovider_disabled": True,
        "temporary_pycache": True,
    }


def build_registration(
    verification: Mapping[str, Any],
    test_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "registered_at": _now(),
        "decision": "FREEZE_GOVERNANCE_AND_AUTHORIZE_TIER2S_SOURCE_ONLY_DIAGNOSTIC",
        "scope": {
            "success_criteria_frozen": True,
            "fresh_seed_ledger_frozen": True,
            "fresh_seed_local_scan_frozen": True,
            "tier2s_code_closure_frozen_by_sha256": True,
            "tier2s_source_only_diagnostic_authorized": True,
            "formal_v3_model_training_authorized": False,
            "source_gate_a_authorized": False,
            "riskcurve_authorized": False,
            "outer_target_access_authorized": False,
            "outer_target_images_used_by_registration": False,
            "outer_target_labels_used_by_registration": False,
            "public_release_artifact_ready": False,
        },
        "contract": {
            "path": str(CONTRACT_PATH.relative_to(PROJECT_ROOT)),
            "sha256": verification["contract_sha256"],
        },
        "fresh_seed_ledger": {
            "path": str(LEDGER_PATH.relative_to(PROJECT_ROOT)),
            "sha256": verification["ledger_sha256"],
            "reserved_formal_seeds": list(FRESH_SEEDS),
            "excluded_development_seeds": list(DEVELOPMENT_SEEDS),
        },
        "fresh_seed_local_scan": dict(verification["local_scan_binding"]),
        "registration_time_fresh_seed_rescan": dict(
            verification["registration_time_rescan_summary"]
        ),
        "historical_tier2r_hold": {
            "path": str(HISTORICAL_DECISION.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(HISTORICAL_DECISION),
            "decision": "TIER2R_HOLD",
            "immutable": True,
        },
        "historical_target_exposure": {
            "path": str(TARGET_EXPOSURE_REGISTRY.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(TARGET_EXPOSURE_REGISTRY),
            "NUAA_untouched_claim_allowed": False,
        },
        "code_sha256": dict(verification["code_sha256"]),
        "git_snapshot": _git_snapshot(),
        "full_test_gate": dict(test_evidence),
        "failure_policy": {
            "any_registered_file_sha256_drift": "FAIL_CLOSED_NO_TIER2S_LAUNCH",
            "any_reserved_seed_conflict": "FAIL_CLOSED_NEW_LEDGER_VERSION_REQUIRED",
            "any_test_failure": "FAIL_CLOSED_NO_REGISTRATION",
            "registration_does_not_authorize_v3_training": True,
            "registration_does_not_authorize_NUAA_access": True,
        },
    }


def _write_once(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise RuntimeError(f"immutable artifact drift: {path}")
        return
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


def _write_digest_sidecar(path: Path, sidecar: Path) -> str:
    digest = _sha256(path)
    raw = f"{digest}  {path.name}\n".encode("ascii")
    _write_once(sidecar, raw)
    sidecar.chmod(0o444)
    return digest


def _verify_digest_sidecar(path: Path, sidecar: Path) -> str:
    if (
        path.is_symlink()
        or not path.is_file()
        or sidecar.is_symlink()
        or not sidecar.is_file()
    ):
        raise RuntimeError(f"required frozen artifact or sidecar is absent: {path}")
    digest = _sha256(path)
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if sidecar.read_bytes() != expected:
        raise RuntimeError(f"SHA-256 sidecar drift: {sidecar}")
    return digest


def _require_read_only(*paths: Path) -> None:
    writable = [
        str(path)
        for path in paths
        if stat.S_IMODE(path.stat().st_mode) & 0o222
    ]
    if writable:
        raise RuntimeError("frozen artifact remains writable: " + ",".join(writable))


def _freeze_input_sidecars() -> dict[str, str]:
    contract_sha = _write_digest_sidecar(CONTRACT_PATH, CONTRACT_SIDECAR)
    ledger_sha = _write_digest_sidecar(LEDGER_PATH, LEDGER_SIDECAR)
    CONTRACT_PATH.chmod(0o444)
    LEDGER_PATH.chmod(0o444)
    return {"contract_sha256": contract_sha, "ledger_sha256": ledger_sha}


def register() -> dict[str, Any]:
    if REGISTRATION_PATH.exists() or REGISTRATION_SIDECAR.exists():
        raise RuntimeError(
            "governance registration already exists; use --verify-only and "
            "never overwrite a frozen registration"
        )
    verification = verify_inputs()
    test_evidence = _run_full_tests()
    post_test_verification = verify_inputs()
    if post_test_verification != verification:
        raise RuntimeError("registered input drifted while the full test gate was running")
    registration = build_registration(post_test_verification, test_evidence)
    frozen = _freeze_input_sidecars()
    if (
        frozen["contract_sha256"] != verification["contract_sha256"]
        or frozen["ledger_sha256"] != verification["ledger_sha256"]
    ):
        raise RuntimeError("input changed while governance registration was running")
    raw = _canonical_bytes(registration)
    _write_once(REGISTRATION_PATH, raw)
    registration_sha = _write_digest_sidecar(
        REGISTRATION_PATH, REGISTRATION_SIDECAR
    )
    verified = verify_registration()
    if verified["registration_sha256"] != registration_sha:
        raise RuntimeError("registration changed during final verification")
    return {
        "registered": True,
        "registration_path": str(REGISTRATION_PATH),
        "registration_sha256": registration_sha,
        "formal_v3_model_training_authorized": False,
        "tier2s_source_only_diagnostic_authorized": True,
        "outer_target_access_authorized": False,
    }


def _frozen_artifact_binding(path: Path, sidecar: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": _sha256(path),
        "sidecar_path": str(sidecar.relative_to(PROJECT_ROOT)),
        "sidecar_sha256": _sha256(sidecar),
    }


def _governance_binding(
    registration: Mapping[str, Any],
    registration_sha256: str,
) -> dict[str, Any]:
    code_sha256 = registration.get("code_sha256")
    if not isinstance(code_sha256, Mapping) or not code_sha256:
        raise RuntimeError("frozen governance code closure is absent")
    if _sha256(REGISTRATION_PATH) != registration_sha256:
        raise RuntimeError("governance registration changed while binding")
    scope = registration.get("scope", {})
    return {
        "schema_version": "rc-irstd-aaai27-tier2s-governance-binding-v1",
        "registration": _frozen_artifact_binding(
            REGISTRATION_PATH, REGISTRATION_SIDECAR
        ),
        "contract": _frozen_artifact_binding(CONTRACT_PATH, CONTRACT_SIDECAR),
        "fresh_seed_ledger": _frozen_artifact_binding(
            LEDGER_PATH, LEDGER_SIDECAR
        ),
        "fresh_seed_local_scan": _frozen_artifact_binding(
            LOCAL_SCAN_PATH, LOCAL_SCAN_SIDECAR
        ),
        "code_sha256_canonical_sha256": hashlib.sha256(
            _canonical_bytes(dict(code_sha256))
        ).hexdigest(),
        "tier2s_source_only_diagnostic_authorized": scope.get(
            "tier2s_source_only_diagnostic_authorized"
        ),
        "formal_v3_model_training_authorized": scope.get(
            "formal_v3_model_training_authorized"
        ),
        "source_gate_a_authorized": scope.get("source_gate_a_authorized"),
        "riskcurve_authorized": scope.get("riskcurve_authorized"),
        "outer_target_access_authorized": scope.get(
            "outer_target_access_authorized"
        ),
    }


def verify_registration() -> dict[str, Any]:
    registration_present = REGISTRATION_PATH.exists() or REGISTRATION_PATH.is_symlink()
    sidecar_present = REGISTRATION_SIDECAR.exists() or REGISTRATION_SIDECAR.is_symlink()
    if not registration_present and not sidecar_present:
        verification = verify_inputs(run_current_scan=True)
        return {
            "verified": True,
            "registered": False,
            "candidate": build_registration(
                verification,
                {
                    "status": "not_run_in_verify_only",
                    "required_before_registration": True,
                },
            ),
        }
    if not registration_present or not sidecar_present:
        raise RuntimeError("governance registration/sidecar pair is incomplete")
    verification = verify_inputs(run_current_scan=False)
    registration_sha = _verify_digest_sidecar(
        REGISTRATION_PATH, REGISTRATION_SIDECAR
    )
    contract_sha = _verify_digest_sidecar(CONTRACT_PATH, CONTRACT_SIDECAR)
    ledger_sha = _verify_digest_sidecar(LEDGER_PATH, LEDGER_SIDECAR)
    _require_read_only(
        REGISTRATION_PATH,
        REGISTRATION_SIDECAR,
        CONTRACT_PATH,
        CONTRACT_SIDECAR,
        LEDGER_PATH,
        LEDGER_SIDECAR,
        LOCAL_SCAN_PATH,
        LOCAL_SCAN_SIDECAR,
    )
    registration = _load_json(REGISTRATION_PATH)
    _validate_registration_time_rescan_summary(
        registration.get("registration_time_fresh_seed_rescan")
    )
    scope = registration.get("scope", {})
    full_test = registration.get("full_test_gate", {})
    full_summary = str(full_test.get("summary", ""))
    full_command = full_test.get("command")
    failure_policy = registration.get("failure_policy", {})
    if (
        registration.get("schema_version") != SCHEMA
        or registration.get("decision")
        != "FREEZE_GOVERNANCE_AND_AUTHORIZE_TIER2S_SOURCE_ONLY_DIAGNOSTIC"
        or not isinstance(registration.get("registered_at"), str)
        or not registration["registered_at"]
        or registration.get("contract", {}).get("path")
        != str(CONTRACT_PATH.relative_to(PROJECT_ROOT))
        or registration.get("contract", {}).get("sha256") != contract_sha
        or registration.get("fresh_seed_ledger", {}).get("path")
        != str(LEDGER_PATH.relative_to(PROJECT_ROOT))
        or registration.get("fresh_seed_ledger", {}).get("sha256") != ledger_sha
        or tuple(
            registration.get("fresh_seed_ledger", {}).get(
                "reserved_formal_seeds", ()
            )
        )
        != FRESH_SEEDS
        or tuple(
            registration.get("fresh_seed_ledger", {}).get(
                "excluded_development_seeds", ()
            )
        )
        != DEVELOPMENT_SEEDS
        or registration.get("fresh_seed_local_scan")
        != verification["local_scan_binding"]
        or registration.get("historical_tier2r_hold", {}).get("path")
        != str(HISTORICAL_DECISION.relative_to(PROJECT_ROOT))
        or registration.get("historical_tier2r_hold", {}).get("sha256")
        != _sha256(HISTORICAL_DECISION)
        or registration.get("historical_tier2r_hold", {}).get("decision")
        != "TIER2R_HOLD"
        or registration.get("historical_tier2r_hold", {}).get("immutable")
        is not True
        or registration.get("historical_target_exposure", {}).get("path")
        != str(TARGET_EXPOSURE_REGISTRY.relative_to(PROJECT_ROOT))
        or registration.get("historical_target_exposure", {}).get("sha256")
        != _sha256(TARGET_EXPOSURE_REGISTRY)
        or registration.get("historical_target_exposure", {}).get(
            "NUAA_untouched_claim_allowed"
        )
        is not False
        or registration.get("code_sha256") != verification["code_sha256"]
        or scope.get("success_criteria_frozen") is not True
        or scope.get("fresh_seed_ledger_frozen") is not True
        or scope.get("fresh_seed_local_scan_frozen") is not True
        or scope.get("tier2s_code_closure_frozen_by_sha256") is not True
        or scope.get("tier2s_source_only_diagnostic_authorized") is not True
        or scope.get("formal_v3_model_training_authorized") is not False
        or scope.get("source_gate_a_authorized") is not False
        or scope.get("riskcurve_authorized") is not False
        or scope.get("outer_target_access_authorized") is not False
        or scope.get("outer_target_images_used_by_registration") is not False
        or scope.get("outer_target_labels_used_by_registration") is not False
        or scope.get("public_release_artifact_ready") is not False
        or not isinstance(full_command, list)
        or full_command[1:] != ["-m", "pytest", "-q", "-p", "no:cacheprovider"]
        or full_test.get("returncode") != 0
        or re.search(r"\b\d+ passed\b", full_summary) is None
        or re.search(r"\b(?:failed|error|errors)\b", full_summary) is not None
        or full_test.get("pytest_plugin_autoload_disabled") is not True
        or full_test.get("pytest_cacheprovider_disabled") is not True
        or full_test.get("temporary_pycache") is not True
        or failure_policy.get("registration_does_not_authorize_v3_training")
        is not True
        or failure_policy.get("registration_does_not_authorize_NUAA_access")
        is not True
    ):
        raise RuntimeError("frozen governance registration semantic drift")
    return {
        "verified": True,
        "registered": True,
        "registration_path": str(REGISTRATION_PATH),
        "registration_sha256": registration_sha,
        "formal_v3_model_training_authorized": False,
        "tier2s_source_only_diagnostic_authorized": True,
        "outer_target_access_authorized": False,
        "governance_binding": _governance_binding(
            registration, registration_sha
        ),
    }


def require_frozen_tier2s_governance(
    *,
    expected_registration_sha256: str | None = None,
) -> dict[str, Any]:
    verified = verify_registration()
    if verified.get("registered") is not True:
        raise RuntimeError(
            "frozen AAAI-27 governance registration is required before Tier2S"
        )
    observed = str(verified["registration_sha256"])
    if expected_registration_sha256 is not None:
        if (
            re.fullmatch(r"[0-9a-f]{64}", expected_registration_sha256) is None
            or observed != expected_registration_sha256
        ):
            raise RuntimeError("Tier2S governance registration SHA-256 drift")
    binding = verified.get("governance_binding")
    if not isinstance(binding, dict):
        raise RuntimeError("Tier2S governance binding is absent")
    return binding


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--register", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = register() if args.register else verify_registration()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
