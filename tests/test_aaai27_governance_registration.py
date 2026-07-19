from __future__ import annotations

from pathlib import Path
import stat
import subprocess

import pytest

from scripts import register_aaai27_governance_v1 as governance


def _valid_registration_time_rescan_summary() -> dict[str, object]:
    return {
        "schema_version":
            "rc-irstd-aaai27-fresh-seed-registration-time-rescan-summary-v1",
        "decision": "PASS_NO_LOCAL_CONSUMPTION_EVIDENCE_FOUND",
        "local_scan_complete": True,
        "local_scan_passed": True,
        "freshness_certified": False,
        "author_attestation_required": True,
        "consumption_like_hits": 0,
        "scan_errors": 0,
        "unscanned_symlinks": 0,
        "structured_json_parse_failures": 0,
        "regular_files_seen": 100,
        "text_files_scanned": 50,
        "text_bytes_scanned": 1000,
        "checkpoint_like_files": 2,
        "git_head_at_rescan": "7" * 40,
        "scanner_path": governance.FRESH_SEED_SCANNER_RELATIVE,
        "scanner_sha256": governance._sha256(
            governance.FRESH_SEED_SCANNER_PATH
        ),
        "normalized_report_sha256": "9" * 64,
        "normalization": governance.REGISTRATION_TIME_RESCAN_NORMALIZATION,
        "local_only_not_freshness_certificate": True,
        "external_deleted_author_attestation_still_required": True,
    }


def test_live_governance_inputs_form_a_complete_hash_closure() -> None:
    verified = governance.verify_inputs()

    assert verified["verified"] is True
    assert len(verified["contract_sha256"]) == 64
    assert len(verified["ledger_sha256"]) == 64
    assert len(verified["local_scan_sha256"]) == 64
    assert verified["local_scan_binding"]["decision"] == (
        "PASS_NO_LOCAL_CONSUMPTION_EVIDENCE_FOUND"
    )
    assert verified["local_scan_binding"][
        "external_deleted_author_attestation_still_required"
    ] is True
    assert "current_rescan_git_head" not in verified["local_scan_binding"]
    assert verified["formal_model_training_authorized"] is False
    assert verified["outer_target_access_authorized"] is False
    assert verified["tier2s_source_only_diagnostic_eligible"] is True
    assert verified["reserved_seed_path_hits"] == []

    closure = verified["code_sha256"]
    for relative in (
        "configs/aaai27_model_success_contract_v1.json",
        "configs/tier2s_factorized_causal_audit_v1.json",
        "scripts/register_aaai27_governance_v1.py",
        "scripts/audit_fresh_seed_reservation_v1.py",
        "scripts/coordinate_tier2s_factorized_audit.py",
        "scripts/evaluate_tier2s_factorized_audit.py",
        "scripts/export_tier2s_factorized_logits.py",
        "scripts/launch_tier2s_factorized_isolated.py",
        "scripts/probe_tier2s_factorized_container.py",
        "tests/test_aaai27_governance_registration.py",
        "tests/test_fresh_seed_reservation_audit.py",
        "artifacts/aaai27/audit/governance/aaai27_model_success_contract_v1/FRESH_SEED_LEDGER.json",
        "artifacts/aaai27/audit/governance/aaai27_model_success_contract_v1/FRESH_SEED_LOCAL_SCAN.json",
        "artifacts/aaai27/audit/governance/aaai27_model_success_contract_v1/FRESH_SEED_LOCAL_SCAN_V2.json",
    ):
        assert relative in closure
        assert len(closure[relative]) == 64


def test_registration_scope_is_strictly_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        governance,
        "_git_snapshot",
        lambda: {
            "head": "0" * 40,
            "branch": "test",
            "worktree_clean": False,
            "status_sha256": "1" * 64,
            "status_num_lines": 1,
            "code_identity_uses_explicit_file_sha256_not_clean_worktree": True,
        },
    )
    registration = governance.build_registration(
        {
            "contract_sha256": "2" * 64,
            "ledger_sha256": "3" * 64,
            "local_scan_binding": {
                "path": "audit/FRESH_SEED_LOCAL_SCAN.json",
                "sha256": "5" * 64,
                "sidecar_path": "audit/FRESH_SEED_LOCAL_SCAN.json.sha256",
                "sidecar_sha256": "6" * 64,
                "decision": "PASS_NO_LOCAL_CONSUMPTION_EVIDENCE_FOUND",
                "frozen_scan_git_head": "7" * 40,
                "local_only_not_freshness_certificate": True,
                "external_deleted_author_attestation_still_required": True,
            },
            "registration_time_rescan_summary": (
                _valid_registration_time_rescan_summary()
            ),
            "code_sha256": {"example.py": "4" * 64},
        },
        {"returncode": 0, "summary": "1 passed in 0.01s"},
    )

    assert registration["decision"] == (
        "FREEZE_GOVERNANCE_AND_AUTHORIZE_TIER2S_SOURCE_ONLY_DIAGNOSTIC"
    )
    scope = registration["scope"]
    assert scope["tier2s_source_only_diagnostic_authorized"] is True
    assert scope["fresh_seed_local_scan_frozen"] is True
    assert scope["formal_v3_model_training_authorized"] is False
    assert scope["source_gate_a_authorized"] is False
    assert scope["riskcurve_authorized"] is False
    assert scope["outer_target_access_authorized"] is False
    assert scope["outer_target_images_used_by_registration"] is False
    assert scope["outer_target_labels_used_by_registration"] is False
    assert scope["public_release_artifact_ready"] is False
    assert registration["fresh_seed_local_scan"][
        "local_only_not_freshness_certificate"
    ] is True
    assert registration["registration_time_fresh_seed_rescan"][
        "normalized_report_sha256"
    ] == "9" * 64


def test_reserved_seed_path_scan_fails_on_run_like_name(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "seed45_control.json").write_text("{}\n", encoding="utf-8")
    assert governance._reserved_seed_path_hits([clean]) == []

    contaminated = tmp_path / "seed46_formal_checkpoint.pth"
    contaminated.write_bytes(b"not-a-real-checkpoint")
    hits = governance._reserved_seed_path_hits([tmp_path])
    assert any("seed46_formal_checkpoint.pth" in hit for hit in hits)


def test_fresh_seed_report_binding_rejects_ledger_sha_drift() -> None:
    ledger = governance._load_json(governance.LEDGER_PATH)
    status = dict(ledger["reserved_seed_status"])
    status["local_machine_scan_report_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="SHA-256 drift"):
        governance._verify_fresh_seed_local_scan(status)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scanner_path", "scripts/other_scanner.py"),
        ("scanner_path", "/tmp/audit_fresh_seed_reservation_v1.py"),
        ("scanner_sha256", "0" * 64),
        ("normalized_report_sha256", "z" * 64),
        ("normalization", "canonical_json_v1"),
        ("git_head_at_rescan", int("7" * 40)),
        ("normalized_report_sha256", int("9" * 64)),
    ],
)
def test_registration_time_scan_rejects_identity_or_hash_drift(
    field: str,
    value: object,
) -> None:
    summary = _valid_registration_time_rescan_summary()
    summary[field] = value
    with pytest.raises(RuntimeError, match="rescan evidence drift"):
        governance._validate_registration_time_rescan_summary(summary)


@pytest.mark.parametrize(
    "field",
    [
        "consumption_like_hits",
        "scan_errors",
        "unscanned_symlinks",
        "structured_json_parse_failures",
        "regular_files_seen",
        "text_files_scanned",
        "text_bytes_scanned",
        "checkpoint_like_files",
    ],
)
@pytest.mark.parametrize("value", [-1, "1", True])
def test_registration_time_scan_rejects_noncanonical_count_types_or_values(
    field: str,
    value: object,
) -> None:
    summary = _valid_registration_time_rescan_summary()
    summary[field] = value
    with pytest.raises(RuntimeError, match="rescan evidence drift"):
        governance._validate_registration_time_rescan_summary(summary)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("consumption_like_hits", 1),
        ("scan_errors", 1),
        ("unscanned_symlinks", 1),
        ("structured_json_parse_failures", 1),
        ("regular_files_seen", 0),
        ("text_files_scanned", 0),
        ("text_files_scanned", 101),
        ("checkpoint_like_files", 101),
    ],
)
def test_registration_time_scan_rejects_out_of_range_counts(
    field: str,
    value: int,
) -> None:
    summary = _valid_registration_time_rescan_summary()
    summary[field] = value
    with pytest.raises(RuntimeError, match="rescan evidence drift"):
        governance._validate_registration_time_rescan_summary(summary)


def test_registration_time_scan_accepts_exact_current_scanner_binding() -> None:
    governance._validate_registration_time_rescan_summary(
        _valid_registration_time_rescan_summary()
    )


def test_digest_verification_is_read_only_and_exact(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    sidecar = tmp_path / "artifact.json.sha256"
    artifact.write_text('{"ok":true}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="sidecar is absent"):
        governance._verify_digest_sidecar(artifact, sidecar)
    assert not sidecar.exists()

    digest = governance._write_digest_sidecar(artifact, sidecar)
    before = sidecar.read_bytes()
    assert governance._verify_digest_sidecar(artifact, sidecar) == digest
    assert sidecar.read_bytes() == before
    assert stat.S_IMODE(sidecar.stat().st_mode) & 0o222 == 0

    sidecar.chmod(0o644)
    sidecar.write_text(f"{'0' * 64}  {artifact.name}\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="sidecar drift"):
        governance._verify_digest_sidecar(artifact, sidecar)


def test_write_once_never_replaces_drifted_artifact(tmp_path: Path) -> None:
    path = tmp_path / "frozen.json"
    governance._write_once(path, b'{"version":1}\n')
    governance._write_once(path, b'{"version":1}\n')
    assert path.read_bytes() == b'{"version":1}\n'
    assert stat.S_IMODE(path.stat().st_mode) & 0o222 == 0

    with pytest.raises(RuntimeError, match="immutable artifact drift"):
        governance._write_once(path, b'{"version":2}\n')


def test_verify_registration_rejects_an_incomplete_registration_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registration = tmp_path / "GOVERNANCE_REGISTRATION.json"
    sidecar = tmp_path / "GOVERNANCE_REGISTRATION.json.sha256"
    registration.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(governance, "REGISTRATION_PATH", registration)
    monkeypatch.setattr(governance, "REGISTRATION_SIDECAR", sidecar)
    monkeypatch.setattr(governance, "verify_inputs", lambda: {})

    with pytest.raises(RuntimeError, match="pair is incomplete"):
        governance.verify_registration()


def test_runtime_governance_api_requires_the_exact_registration_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = {"schema_version": "synthetic-binding"}
    monkeypatch.setattr(
        governance,
        "verify_registration",
        lambda: {
            "registered": True,
            "registration_sha256": "a" * 64,
            "governance_binding": binding,
        },
    )

    assert governance.require_frozen_tier2s_governance(
        expected_registration_sha256="a" * 64
    ) == binding
    with pytest.raises(RuntimeError, match="SHA-256 drift"):
        governance.require_frozen_tier2s_governance(
            expected_registration_sha256="b" * 64
        )


@pytest.mark.parametrize(
    ("stdout", "returncode", "accepted"),
    [
        ("920 passed, 26 skipped, 12 warnings in 50.94s\n", 0, True),
        ("1 failed, 919 passed in 50.94s\n", 1, False),
        ("collection completed without a test summary\n", 0, False),
    ],
)
def test_full_test_gate_requires_an_explicit_passing_summary(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    returncode: int,
    accepted: bool,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["pytest"],
            returncode=returncode,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr(governance.subprocess, "run", fake_run)
    if accepted:
        result = governance._run_full_tests()
        assert result["returncode"] == 0
        assert "920 passed" in result["summary"]
    else:
        with pytest.raises(RuntimeError, match="full governance test gate failed"):
            governance._run_full_tests()
