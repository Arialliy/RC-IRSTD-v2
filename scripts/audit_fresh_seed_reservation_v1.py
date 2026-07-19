#!/usr/bin/env python3
"""Audit local evidence for reserved formal seeds without deserializing checkpoints.

This scanner is deliberately narrower than a freshness certificate.  It walks the
current project tree directly (so hidden and Git-ignored files are included),
records local consumption-like evidence for seeds 46--50, and always states that
author attestation is still required for deleted or out-of-repository artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "rc-irstd-aaai27-fresh-seed-local-audit-v1"
ALGORITHM_VERSION = "fresh-seed-local-filesystem-scan-v2"
FRESH_SEEDS: tuple[int, ...] = (46, 47, 48, 49, 50)

# These directories are intentionally outside the audit scope.  In particular,
# datasets are data inputs rather than run-evidence stores, while the remaining
# names are VCS internals, environments, caches, or vendored dependency trees.
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".cache",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "datasets",
        "dist-packages",
        "env",
        "infrarenet",
        "node_modules",
        "site-packages",
        "third_party",
        "vendor",
        "venv",
    }
)

TEXT_SUFFIXES = frozenset(
    {
        ".bash",
        ".cfg",
        ".conf",
        ".csv",
        ".env",
        ".ini",
        ".json",
        ".jsonl",
        ".log",
        ".manifest",
        ".md",
        ".py",
        ".rst",
        ".sh",
        ".tex",
        ".toml",
        ".tsv",
        ".txt",
        ".yaml",
        ".yml",
        ".zsh",
    }
)
BACKUP_SUFFIXES = frozenset({".bak", ".copy", ".orig", ".rej"})
TEXT_BASENAMES = frozenset(
    {
        "dockerfile",
        "license",
        "makefile",
        "readme",
        "requirements",
    }
)
CHECKPOINT_SUFFIXES = (
    ".pth.tar",
    ".safetensors",
    ".checkpoint",
    ".pickle",
    ".ckpt",
    ".joblib",
    ".onnx",
    ".pkl",
    ".pth",
    ".pt",
    ".weights",
)

DEFAULT_RESERVATION_ALLOWLIST = frozenset(
    {
        "configs/aaai27_model_success_contract_v1.json",
        "artifacts/aaai27/audit/governance/aaai27_model_success_contract_v1/"
        "FRESH_SEED_LEDGER.json",
        "artifacts/aaai27/audit/governance/aaai27_model_success_contract_v1/"
        "FRESH_SEED_LOCAL_SCAN.json",
        "artifacts/aaai27/audit/governance/aaai27_model_success_contract_v1/"
        "FRESH_SEED_LOCAL_SCAN_V2.json",
        "scripts/audit_fresh_seed_reservation_v1.py",
        "scripts/register_aaai27_governance_v1.py",
    }
)

_SEED_ALTERNATION = "|".join(str(seed) for seed in FRESH_SEEDS)
_PATH_SEED_RE = re.compile(
    rf"(?i)(?<![a-z0-9])seed(?:[_.:= -]?)(?P<seed>{_SEED_ALTERNATION})(?![0-9])"
)
_COMPACT_SEED_RE = re.compile(
    rf"(?i)(?<![a-z0-9_])seed(?:[_. -]?)(?P<seed>{_SEED_ALTERNATION})(?![0-9])"
)
_NAMED_SEED_KEY_RE = re.compile(
    rf"(?i)(?<![a-z0-9_])[\"']?"
    rf"(?P<key>[a-z0-9_-]*seed[a-z0-9_-]*)[\"']?\s*[:=]\s*"
    rf"(?P<seed>{_SEED_ALTERNATION})(?![0-9])"
)
_RESERVATION_KEY_RE = re.compile(
    r"(?i)\b(?:reserved(?:[_ -]+formal)?[_ -]*seeds?|"
    r"fresh[_ -]*seeds?(?:[_ -]*ids)?|formal[_ -]*seed[_ -]*ids)\b"
)
_INTEGER_SEED_RE = re.compile(
    rf"(?<![0-9])(?P<seed>{_SEED_ALTERNATION})(?![0-9])"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the canonical on-disk encoding used by the audit artifact."""

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


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _normalize_allowlist(paths: Iterable[str | Path]) -> frozenset[str]:
    normalized: set[str] = set(DEFAULT_RESERVATION_ALLOWLIST)
    for raw in paths:
        text = str(raw).replace("\\", "/")
        parsed = PurePosixPath(text)
        if parsed.is_absolute() or ".." in parsed.parts or text in {"", "."}:
            raise ValueError(
                f"reservation allowlist entry must be a project-relative file: {raw}"
            )
        normalized.add(parsed.as_posix())
    return frozenset(normalized)


def _path_role(relative: str, allowlist: frozenset[str]) -> tuple[str, str]:
    parts = PurePosixPath(relative).parts
    if "tests" in parts:
        return "test_fixture", "path_is_within_tests_tree"
    if PurePosixPath(relative).suffix.casefold() in BACKUP_SUFFIXES:
        return "consumption_like", "backup_file_never_reservation_declaration"
    if relative in allowlist:
        return "reservation_declaration", "explicit_reservation_allowlist"
    return "consumption_like", "unallowlisted_seed_evidence"


def _is_text_file(path: Path) -> bool:
    name = path.name.casefold()
    suffixes = {suffix.casefold() for suffix in path.suffixes}
    if suffixes & TEXT_SUFFIXES:
        return True
    if suffixes & BACKUP_SUFFIXES and suffixes & TEXT_SUFFIXES:
        return True
    return name in TEXT_BASENAMES or any(
        name.startswith(prefix) for prefix in ("readme.", "requirements.")
    )


def _is_json_like(path: Path) -> bool:
    suffixes = [suffix.casefold() for suffix in path.suffixes]
    if not suffixes:
        return False
    if suffixes[-1] == ".json":
        return True
    return (
        len(suffixes) >= 2
        and suffixes[-2] == ".json"
        and suffixes[-1] in BACKUP_SUFFIXES
    )


def _is_checkpoint_like(path: Path) -> bool:
    name = path.name.casefold()
    return any(name.endswith(suffix) for suffix in CHECKPOINT_SUFFIXES)


def _seed_path_tokens(relative: str) -> list[int]:
    return sorted(
        {
            int(match.group("seed"))
            for match in _PATH_SEED_RE.finditer(relative)
        }
    )


def _is_reservation_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
    return bool(
        normalized.startswith("reserved_seed")
        or normalized.startswith("reserved_formal_seed")
        or normalized in {
            "fresh_seed",
            "fresh_seed_ids",
            "fresh_seeds",
            "formal_seed_ids",
            "formal_seeds",
            "reserved_formal_seeds",
            "reserved_seeds",
        }
    )


def _direct_target_seed_values(value: Any) -> list[int]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value] if value in FRESH_SEEDS else []
    if isinstance(value, float) and value.is_integer():
        integer = int(value)
        return [integer] if integer in FRESH_SEEDS else []
    if isinstance(value, str) and value.strip().isdigit():
        integer = int(value.strip())
        return [integer] if integer in FRESH_SEEDS else []
    if isinstance(value, list):
        values: list[int] = []
        for item in value:
            if isinstance(item, (dict, tuple, set)):
                continue
            values.extend(_direct_target_seed_values(item))
        return sorted(set(values))
    return []


def _json_seed_records(
    value: Any,
    *,
    relative: str,
    allowlist: frozenset[str],
    key_path: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key)
            current = (*key_path, key)
            if "seed" in key.casefold():
                for seed in _direct_target_seed_values(child):
                    path_class, path_reason = _path_role(relative, allowlist)
                    if path_class == "test_fixture":
                        classification = path_class
                        reason = path_reason
                    elif path_reason == "backup_file_never_reservation_declaration":
                        classification = "consumption_like"
                        reason = path_reason
                    elif path_class == "reservation_declaration" or _is_reservation_key(
                        key
                    ):
                        classification = "reservation_declaration"
                        reason = (
                            path_reason
                            if path_class == "reservation_declaration"
                            else "structured_reservation_key"
                        )
                    else:
                        classification = "consumption_like"
                        reason = "structured_seed_key_not_reserved"
                    records.append(
                        {
                            "classification": classification,
                            "json_path": "/" + "/".join(current),
                            "line": None,
                            "match": f"{key}={seed}",
                            "path": relative,
                            "reason": reason,
                            "seed": seed,
                            "source": "structured_json",
                        }
                    )
            records.extend(
                _json_seed_records(
                    child,
                    relative=relative,
                    allowlist=allowlist,
                    key_path=current,
                )
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            records.extend(
                _json_seed_records(
                    child,
                    relative=relative,
                    allowlist=allowlist,
                    key_path=(*key_path, str(index)),
                )
            )
    return records


def _text_seed_records(
    path: Path,
    *,
    relative: str,
    allowlist: frozenset[str],
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    bytes_scanned = 0
    path_class, path_reason = _path_role(relative, allowlist)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            bytes_scanned += len(line.encode("utf-8", errors="replace"))
            seen_spans: set[tuple[int, int, int]] = set()
            for pattern in (_COMPACT_SEED_RE, _NAMED_SEED_KEY_RE):
                for match in pattern.finditer(line):
                    seed = int(match.group("seed"))
                    marker = (match.start(), match.end(), seed)
                    if marker in seen_spans:
                        continue
                    seen_spans.add(marker)
                    classification = path_class
                    reason = path_reason
                    matched_key = match.groupdict().get("key")
                    if (
                        matched_key is not None
                        and path_class != "test_fixture"
                        and path_reason
                        != "backup_file_never_reservation_declaration"
                        and (
                            path_class == "reservation_declaration"
                            or _is_reservation_key(matched_key)
                        )
                    ):
                        classification = "reservation_declaration"
                        reason = (
                            path_reason
                            if path_class == "reservation_declaration"
                            else "text_reservation_key"
                        )
                    records.append(
                        {
                            "classification": classification,
                            "line": line_number,
                            "match": match.group(0)[:160],
                            "path": relative,
                            "reason": reason,
                            "seed": seed,
                            "source": "text_pattern",
                        }
                    )

            reservation_key = _RESERVATION_KEY_RE.search(line)
            if reservation_key is not None:
                suffix = line[reservation_key.start() :]
                for match in _INTEGER_SEED_RE.finditer(suffix):
                    seed = int(match.group("seed"))
                    if path_class == "test_fixture":
                        classification = "test_fixture"
                        reason = path_reason
                    elif path_reason == "backup_file_never_reservation_declaration":
                        classification = "consumption_like"
                        reason = path_reason
                    else:
                        classification = "reservation_declaration"
                        reason = (
                            path_reason
                            if path_class == "reservation_declaration"
                            else "text_reservation_key"
                        )
                    records.append(
                        {
                            "classification": classification,
                            "line": line_number,
                            "match": suffix[:160].rstrip("\r\n"),
                            "path": relative,
                            "reason": reason,
                            "seed": seed,
                            "source": "text_reservation_declaration",
                        }
                    )
    return records, bytes_scanned


def _git_head(root: Path) -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"git_head_unavailable:{type(error).__name__}"
    head = completed.stdout.strip()
    if completed.returncode != 0 or not re.fullmatch(r"[0-9a-fA-F]{40,64}", head):
        return None, "git_head_unavailable:not_a_git_worktree_or_no_commit"
    return head.lower(), None


def _scanner_identity(root: Path) -> dict[str, Any]:
    scanner = Path(__file__).resolve()
    try:
        display_path = _relative_posix(scanner, root)
    except ValueError:
        display_path = str(scanner)
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "path": display_path,
        "sha256": _sha256_file(scanner),
    }


def _deduplicate_and_sort(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[bytes, dict[str, Any]] = {}
    for record in records:
        normalized = dict(record)
        key = canonical_json_bytes(normalized)
        unique[key] = normalized
    return sorted(
        unique.values(),
        key=lambda row: (
            str(row.get("path", "")),
            -1 if row.get("line") is None else int(row["line"]),
            int(row.get("seed", -1)),
            str(row.get("source", "")),
            str(row.get("match", "")),
        ),
    )


def scan_repository(
    root: str | Path = PROJECT_ROOT,
    *,
    reservation_allowlist: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Scan one current filesystem snapshot and return a canonicalizable report."""

    project_root = Path(root).resolve()
    if project_root.is_symlink() or not project_root.is_dir():
        raise ValueError(f"scan root must be a canonical directory: {project_root}")
    allowlist = _normalize_allowlist(reservation_allowlist)

    all_hits: list[dict[str, Any]] = []
    checkpoint_files: list[dict[str, Any]] = []
    excluded_paths: list[str] = []
    unscanned_symlinks: list[str] = []
    scan_errors: list[dict[str, str]] = []
    structured_json_parse_failures: list[dict[str, str]] = []
    counts = {
        "checkpoint_like_files": 0,
        "directories_scanned": 0,
        "excluded_directories": 0,
        "path_names_scanned": 0,
        "regular_files_seen": 0,
        "structured_json_files_parsed": 0,
        "structured_json_parse_failures": 0,
        "symlinks_not_followed": 0,
        "text_bytes_scanned": 0,
        "text_files_scanned": 0,
    }

    def walk_error(error: OSError) -> None:
        filename = str(error.filename or "")
        try:
            relative = _relative_posix(Path(filename).resolve(), project_root)
        except (OSError, ValueError):
            relative = filename
        scan_errors.append(
            {
                "error": f"{type(error).__name__}:{error}",
                "path": relative,
            }
        )

    for directory, dirnames, filenames in os.walk(
        project_root,
        topdown=True,
        followlinks=False,
        onerror=walk_error,
    ):
        directory_path = Path(directory)
        counts["directories_scanned"] += 1
        kept_directories: list[str] = []
        for name in sorted(dirnames):
            child = directory_path / name
            relative = _relative_posix(child, project_root)
            if name.casefold() in EXCLUDED_DIRECTORY_NAMES:
                excluded_paths.append(relative)
                counts["excluded_directories"] += 1
                continue
            if child.is_symlink():
                unscanned_symlinks.append(relative)
                counts["symlinks_not_followed"] += 1
                continue
            counts["path_names_scanned"] += 1
            for seed in _seed_path_tokens(relative):
                classification, reason = _path_role(relative, allowlist)
                all_hits.append(
                    {
                        "classification": classification,
                        "line": None,
                        "match": relative,
                        "path": relative,
                        "reason": reason,
                        "seed": seed,
                        "source": "directory_path",
                    }
                )
            kept_directories.append(name)
        dirnames[:] = kept_directories

        for name in sorted(filenames):
            path = directory_path / name
            relative = _relative_posix(path, project_root)
            counts["path_names_scanned"] += 1
            if path.is_symlink():
                unscanned_symlinks.append(relative)
                counts["symlinks_not_followed"] += 1
                continue
            try:
                metadata = path.stat()
            except OSError as error:
                scan_errors.append(
                    {
                        "error": f"{type(error).__name__}:{error}",
                        "path": relative,
                    }
                )
                continue
            if not stat.S_ISREG(metadata.st_mode):
                scan_errors.append(
                    {
                        "error": "non_regular_non_symlink_entry",
                        "path": relative,
                    }
                )
                continue
            counts["regular_files_seen"] += 1

            path_seeds = _seed_path_tokens(relative)
            for seed in path_seeds:
                classification, reason = _path_role(relative, allowlist)
                all_hits.append(
                    {
                        "classification": classification,
                        "line": None,
                        "match": relative,
                        "path": relative,
                        "reason": reason,
                        "seed": seed,
                        "source": "file_path",
                    }
                )

            if _is_checkpoint_like(path):
                checkpoint_files.append(
                    {
                        "content_deserialized": False,
                        "path": relative,
                        "path_seed_tokens": path_seeds,
                        "size_bytes": metadata.st_size,
                    }
                )
                counts["checkpoint_like_files"] += 1

            structured_payload: Any | None = None
            if _is_json_like(path):
                try:
                    structured_payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as error:
                    structured_json_parse_failures.append(
                        {
                            "error": f"{type(error).__name__}:{error}",
                            "path": relative,
                        }
                    )
                    counts["structured_json_parse_failures"] += 1
                else:
                    counts["structured_json_files_parsed"] += 1

            if not _is_text_file(path):
                continue
            try:
                text_records, bytes_scanned = _text_seed_records(
                    path,
                    relative=relative,
                    allowlist=allowlist,
                )
                all_hits.extend(text_records)
                counts["text_files_scanned"] += 1
                counts["text_bytes_scanned"] += bytes_scanned
            except OSError as error:
                scan_errors.append(
                    {
                        "error": f"text_read_failed:{type(error).__name__}:{error}",
                        "path": relative,
                    }
                )
                continue

            if structured_payload is not None:
                all_hits.extend(
                    _json_seed_records(
                        structured_payload,
                        relative=relative,
                        allowlist=allowlist,
                    )
                )

    hits = _deduplicate_and_sort(all_hits)
    classified = {
        classification: [
            record
            for record in hits
            if record["classification"] == classification
        ]
        for classification in (
            "consumption_like",
            "reservation_declaration",
            "test_fixture",
        )
    }
    checkpoint_files = sorted(checkpoint_files, key=lambda row: row["path"])
    git_head, git_error = _git_head(project_root)
    scan_complete = not scan_errors and not unscanned_symlinks and git_head is not None
    local_conflict = bool(classified["consumption_like"])
    if local_conflict:
        decision = "HOLD_LOCAL_CONSUMPTION_LIKE_EVIDENCE"
    elif not scan_complete:
        decision = "HOLD_INCOMPLETE_LOCAL_SCAN"
    else:
        decision = "PASS_NO_LOCAL_CONSUMPTION_EVIDENCE_FOUND"

    limitations = [
        "The audit covers only the current filesystem snapshot under the declared scan root.",
        "No claim is made about deleted artifacts, unavailable Git history, other clones, external storage, or other machines.",
        "Checkpoint-like file contents are not deserialized because pickle- and framework-based loading is not a safe read-only audit primitive; only checkpoint paths are seed-token checked.",
        "Intent cannot be proven from a numeric seed alone; explicit reservation declarations and test fixtures are reported separately, while every other matching local hit is treated as consumption-like.",
    ]
    if structured_json_parse_failures:
        limitations.append(
            "Some JSON-like files were not valid JSON; their text was still pattern-scanned, and parse failures are reported explicitly."
        )
    if unscanned_symlinks:
        limitations.append(
            "Symlink targets were not followed; any in-scope symlink makes the local scan incomplete and fail closed."
        )
    if git_error is not None:
        limitations.append(git_error)

    return {
        "algorithm_version": ALGORITHM_VERSION,
        "author_attestation_required": True,
        "checkpoint_audit": {
            "content_parsing_performed": False,
            "files": checkpoint_files,
            "path_seed_token_detection": True,
            "unsafe_content_parsing_limitation_acknowledged": True,
        },
        "counts": {
            **counts,
            "consumption_like_hits": len(classified["consumption_like"]),
            "reservation_declaration_hits": len(
                classified["reservation_declaration"]
            ),
            "test_fixture_hits": len(classified["test_fixture"]),
        },
        "decision": decision,
        "excluded_directory_paths": sorted(excluded_paths),
        "fresh_seed_ids": list(FRESH_SEEDS),
        "freshness_certified": False,
        "generated_at": _utc_now(),
        "git": {
            "head": git_head,
            "head_error": git_error,
        },
        "hits": classified,
        "limitations": limitations,
        "local_scan_complete": scan_complete,
        "local_scan_passed": decision
        == "PASS_NO_LOCAL_CONSUMPTION_EVIDENCE_FOUND",
        "reservation_allowlist": sorted(allowlist),
        "scan_errors": sorted(scan_errors, key=lambda row: row["path"]),
        "scan_scope": {
            "excluded_directory_names": sorted(EXCLUDED_DIRECTORY_NAMES),
            "follows_symlinks": False,
            "git_ignored_files_included": True,
            "hidden_files_included": True,
            "path_names_and_text_content_scanned": True,
            "root": ".",
            "text_suffixes": sorted(TEXT_SUFFIXES),
        },
        "scanner": _scanner_identity(project_root),
        "schema_version": SCHEMA_VERSION,
        "structured_json_parse_failures": sorted(
            structured_json_parse_failures,
            key=lambda row: row["path"],
        ),
        "unscanned_symlinks": sorted(unscanned_symlinks),
    }


def _exclusive_write(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)


def write_report_no_replace(path: str | Path, report: Mapping[str, Any]) -> dict[str, str]:
    """Write report and SHA sidecar once, refusing any pre-existing target."""

    output = Path(path)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"audit output already exists; refusing overwrite: {output}")
    if sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError(
            f"audit SHA sidecar already exists; refusing overwrite: {sidecar}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(report)
    digest = hashlib.sha256(raw).hexdigest()
    sidecar_raw = f"{digest}  {output.name}\n".encode("ascii")
    _exclusive_write(output, raw)
    _exclusive_write(sidecar, sidecar_raw)
    return {
        "path": str(output),
        "sha256": digest,
        "sidecar": str(sidecar),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="project root to scan (default: this repository)",
    )
    parser.add_argument(
        "--write",
        type=Path,
        help="optional write-once canonical JSON output; default is stdout only",
    )
    parser.add_argument(
        "--reservation-path",
        action="append",
        default=[],
        help="additional project-relative reservation declaration file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    root = args.root.resolve()
    allowlist = list(args.reservation_path)
    output: Path | None = None
    if args.write is not None:
        output = args.write if args.write.is_absolute() else root / args.write
        try:
            allowlist.append(_relative_posix(output.resolve(), root))
        except ValueError as error:
            raise SystemExit("--write must remain within --root") from error
        sidecar = output.with_suffix(output.suffix + ".sha256")
        if output.exists() or output.is_symlink() or sidecar.exists() or sidecar.is_symlink():
            raise SystemExit("--write target or SHA sidecar already exists; refusing overwrite")

    report = scan_repository(root, reservation_allowlist=allowlist)
    raw = canonical_json_bytes(report)
    if output is not None:
        write_report_no_replace(output, report)
    sys.stdout.buffer.write(raw)
    return 0 if report["local_scan_passed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
