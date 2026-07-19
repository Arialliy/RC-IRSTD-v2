from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

from scripts import audit_fresh_seed_reservation_v1 as audit


def _consumption_paths(report: dict) -> set[str]:
    return {row["path"] for row in report["hits"]["consumption_like"]}


def test_hidden_and_git_ignored_files_are_scanned_but_git_and_datasets_are_not(
    tmp_path: Path,
) -> None:
    (tmp_path / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "config.json").write_text('{"seed":46}\n', encoding="utf-8")
    ignored = tmp_path / "ignored"
    ignored.mkdir()
    (ignored / "config.yaml").write_text("seed: 47\n", encoding="utf-8")

    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "seed48.log").write_text("seed:48\n", encoding="utf-8")
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "seed49.txt").write_text("seed49\n", encoding="utf-8")

    report = audit.scan_repository(tmp_path)
    paths = _consumption_paths(report)
    assert ".hidden/config.json" in paths
    assert "ignored/config.yaml" in paths
    assert not any(path.startswith(".git/") for path in paths)
    assert not any(path.startswith("datasets/") for path in paths)
    assert report["scan_scope"]["hidden_files_included"] is True
    assert report["scan_scope"]["git_ignored_files_included"] is True
    assert {".git", "datasets"}.issubset(
        set(report["scan_scope"]["excluded_directory_names"])
    )


def test_plain_json_seed_key_is_consumption_like(tmp_path: Path) -> None:
    run = tmp_path / "runs"
    run.mkdir()
    config = run / "config.json"
    config.write_text(
        json.dumps({"model": "candidate", "seed": 48}) + "\n",
        encoding="utf-8",
    )

    report = audit.scan_repository(tmp_path)
    hits = [
        row
        for row in report["hits"]["consumption_like"]
        if row["path"] == "runs/config.json" and row["seed"] == 48
    ]
    assert hits
    assert any(row["source"] == "structured_json" for row in hits)
    assert report["decision"] == "HOLD_LOCAL_CONSUMPTION_LIKE_EVIDENCE"
    assert report["freshness_certified"] is False
    assert report["author_attestation_required"] is True


def test_named_yaml_training_seed_is_consumption_like(tmp_path: Path) -> None:
    config = tmp_path / "training.yaml"
    config.write_text("training_seed: 50\n", encoding="utf-8")

    report = audit.scan_repository(tmp_path)
    assert any(
        row["path"] == "training.yaml" and row["seed"] == 50
        for row in report["hits"]["consumption_like"]
    )


def test_explicit_reservation_allowlist_and_test_fixture_are_separate(
    tmp_path: Path,
) -> None:
    governance = tmp_path / "governance"
    governance.mkdir()
    declaration = governance / "reservation.txt"
    declaration.write_text("reserved seed46 for future formal use\n", encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    fixture = tests / "test_seed_fixture.py"
    fixture.write_text('payload = {"seed": 47}\n', encoding="utf-8")

    report = audit.scan_repository(
        tmp_path,
        reservation_allowlist=["governance/reservation.txt"],
    )
    reservation_paths = {
        row["path"] for row in report["hits"]["reservation_declaration"]
    }
    fixture_paths = {row["path"] for row in report["hits"]["test_fixture"]}
    assert "governance/reservation.txt" in reservation_paths
    assert "tests/test_seed_fixture.py" in fixture_paths
    assert "governance/reservation.txt" not in _consumption_paths(report)
    assert "tests/test_seed_fixture.py" not in _consumption_paths(report)


def test_default_declaration_backups_are_consumption_like(tmp_path: Path) -> None:
    declaration_paths = (
        "configs/aaai27_model_success_contract_v1.json",
        "artifacts/aaai27/audit/governance/"
        "aaai27_model_success_contract_v1/FRESH_SEED_LEDGER.json",
        "artifacts/aaai27/audit/governance/"
        "aaai27_model_success_contract_v1/FRESH_SEED_LOCAL_SCAN.json",
        "artifacts/aaai27/audit/governance/"
        "aaai27_model_success_contract_v1/FRESH_SEED_LOCAL_SCAN_V2.json",
        "scripts/audit_fresh_seed_reservation_v1.py",
        "scripts/register_aaai27_governance_v1.py",
    )
    payload = json.dumps(
        {
            "seed": 46,
            "reserved_formal_seeds": [46, 47, 48, 49, 50],
        }
    )
    backup_paths: set[str] = set()
    for declaration_path in declaration_paths:
        for suffix in sorted(audit.BACKUP_SUFFIXES):
            relative = declaration_path + suffix
            backup = tmp_path / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_text(payload + "\n", encoding="utf-8")
            backup_paths.add(relative)

    report = audit.scan_repository(
        tmp_path,
        reservation_allowlist=backup_paths,
    )
    consumption = report["hits"]["consumption_like"]
    reservation_paths = {
        row["path"] for row in report["hits"]["reservation_declaration"]
    }
    for relative in backup_paths:
        assert {
            row["seed"] for row in consumption if row["path"] == relative
        } == {46, 47, 48, 49, 50}
    assert backup_paths.isdisjoint(reservation_paths)


def test_structured_reserved_seed_list_is_a_declaration(tmp_path: Path) -> None:
    declaration = tmp_path / "reservation.json"
    declaration.write_text(
        json.dumps({"reserved_formal_seeds": [46, 47, 48, 49, 50]}) + "\n",
        encoding="utf-8",
    )

    report = audit.scan_repository(tmp_path)
    seeds = {
        row["seed"]
        for row in report["hits"]["reservation_declaration"]
        if row["path"] == "reservation.json"
    }
    assert seeds == {46, 47, 48, 49, 50}
    assert "reservation.json" not in _consumption_paths(report)


def test_checkpoint_is_enumerated_and_seed_token_in_path_fails_closed(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "runs" / "seed49"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "model.pth"
    checkpoint.write_bytes(b"not-safe-to-deserialize")

    report = audit.scan_repository(tmp_path)
    rows = report["checkpoint_audit"]["files"]
    assert rows == [
        {
            "content_deserialized": False,
            "path": "runs/seed49/model.pth",
            "path_seed_tokens": [49],
            "size_bytes": len(b"not-safe-to-deserialize"),
        }
    ]
    assert "runs/seed49/model.pth" in _consumption_paths(report)
    assert report["checkpoint_audit"]["content_parsing_performed"] is False
    assert (
        report["checkpoint_audit"][
            "unsafe_content_parsing_limitation_acknowledged"
        ]
        is True
    )
    assert any("not deserialized" in line for line in report["limitations"])


def test_write_is_canonical_read_only_has_sidecar_and_never_replaces(
    tmp_path: Path,
) -> None:
    report = audit.scan_repository(tmp_path)
    output = tmp_path / "audit" / "fresh_seed_scan.json"
    written = audit.write_report_no_replace(output, report)
    sidecar = output.with_suffix(".json.sha256")

    expected_raw = audit.canonical_json_bytes(report)
    expected_digest = hashlib.sha256(expected_raw).hexdigest()
    assert output.read_bytes() == expected_raw
    assert sidecar.read_text(encoding="ascii") == (
        f"{expected_digest}  {output.name}\n"
    )
    assert written["sha256"] == expected_digest
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o444

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        audit.write_report_no_replace(output, report)


def test_report_records_scanner_identity_scope_counts_and_local_limits(
    tmp_path: Path,
) -> None:
    (tmp_path / "notes.md").write_text("no formal run here\n", encoding="utf-8")
    report = audit.scan_repository(tmp_path)

    assert audit.ALGORITHM_VERSION == "fresh-seed-local-filesystem-scan-v2"
    assert report["algorithm_version"] == audit.ALGORITHM_VERSION
    assert report["fresh_seed_ids"] == [46, 47, 48, 49, 50]
    assert len(report["scanner"]["sha256"]) == 64
    assert report["counts"]["regular_files_seen"] >= 1
    assert report["scan_scope"]["root"] == "."
    assert report["scan_scope"]["path_names_and_text_content_scanned"] is True
    assert report["git"]["head"] is None
    assert report["decision"] == "HOLD_INCOMPLETE_LOCAL_SCAN"
    assert any("deleted artifacts" in line for line in report["limitations"])


@pytest.mark.parametrize(
    "report_name",
    ["FRESH_SEED_LOCAL_SCAN.json", "FRESH_SEED_LOCAL_SCAN_V2.json"],
)
def test_canonical_scanner_report_is_audit_evidence_not_consumption(
    tmp_path: Path,
    report_name: str,
) -> None:
    prior = {
        "schema_version": audit.SCHEMA_VERSION,
        "fresh_seed_ids": [46, 47, 48, 49, 50],
        "hits": {"consumption_like": [{"seed": 46, "path": "example"}]},
    }
    relative = (
        "artifacts/aaai27/audit/governance/"
        f"aaai27_model_success_contract_v1/{report_name}"
    )
    prior_path = tmp_path / relative
    prior_path.parent.mkdir(parents=True)
    prior_path.write_bytes(audit.canonical_json_bytes(prior))

    report = audit.scan_repository(tmp_path)
    assert relative not in _consumption_paths(report)
    assert any(
        row["path"] == relative
        for row in report["hits"]["reservation_declaration"]
    )


def test_forged_scanner_schema_cannot_self_authorize(tmp_path: Path) -> None:
    forged = tmp_path / "forged_scan.json"
    forged.write_text(
        json.dumps(
            {
                "schema_version": audit.SCHEMA_VERSION,
                "seed": 46,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report = audit.scan_repository(tmp_path)
    assert any(
        row["path"] == "forged_scan.json" and row["seed"] == 46
        for row in report["hits"]["consumption_like"]
    )
    assert not any(
        row["path"] == "forged_scan.json"
        for row in report["hits"]["reservation_declaration"]
    )
