#!/usr/bin/env python3
"""Read-only audit of the repository's frozen train/test dataset manifests.

The command never creates, edits, or re-partitions dataset files.  Its only
write is an atomically replaced JSON report selected by ``--output``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

# Keep the documented ``python scripts/audit_dataset_splits.py`` invocation
# independent of the caller's PYTHONPATH.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_ext.split_utils import (
    ensure_unique_sample_ids,
    read_split_file,
    resolve_split_file,
)
from data_ext.mask_alignment import (
    MASK_ALIGNMENT_ASPECT_TOLERANCE,
    MASK_ALIGNMENT_POLICY,
    aspect_error_within_tolerance,
    relative_aspect_error,
)


_RASTER_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
_ORDERED_IDS_HASH_ALGORITHM = "sha256(utf8(id + '\\n') in manifest order)"
_RASTER_CONTENT_HASH_ALGORITHM = (
    "sha256(canonical-mode + NUL + width:uint64be + height:uint64be + pixels)"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_ids_sha256(sample_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _raster_candidates(folder: Path, sample_id: str, *, is_mask: bool) -> list[Path]:
    """Return every exact raster candidate; sidecars and prefix matches are ignored."""

    if not folder.is_dir():
        return []
    stems = {sample_id}
    if is_mask:
        stems.add(f"{sample_id}_pixels0")
    return sorted(
        (
            candidate.resolve()
            for candidate in folder.iterdir()
            if candidate.is_file()
            and candidate.stem in stems
            and candidate.suffix.lower() in _RASTER_SUFFIXES
        ),
        key=lambda candidate: str(candidate),
    )


def _raster_record(path: Path, *, canonical_mode: str) -> dict[str, Any]:
    with Image.open(path) as opened:
        frame_count = int(getattr(opened, "n_frames", 1))
        raster = opened.convert(canonical_mode)
        raster.load()
        width, height = raster.size
        digest = hashlib.sha256()
        digest.update(canonical_mode.encode("ascii"))
        digest.update(b"\0")
        digest.update(int(width).to_bytes(8, "big", signed=False))
        digest.update(int(height).to_bytes(8, "big", signed=False))
        digest.update(raster.tobytes())
    return {
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "content_sha256": digest.hexdigest(),
        "canonical_mode": canonical_mode,
        "width": width,
        "height": height,
        "frame_count": frame_count,
    }


def _resolve_raster(
    dataset_root: Path,
    sample_id: str,
    *,
    kind: str,
    issues: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    is_mask = kind == "mask"
    folder = dataset_root / ("masks" if is_mask else "images")
    candidates = _raster_candidates(folder, sample_id, is_mask=is_mask)
    record: dict[str, Any] = {
        "status": "unique" if len(candidates) == 1 else (
            "missing" if not candidates else "ambiguous"
        ),
        "candidate_count": len(candidates),
        "candidates": [str(candidate) for candidate in candidates],
    }
    if len(candidates) != 1:
        issues.append(
            {
                "code": f"{kind}_raster_{record['status']}",
                "split": split,
                "sample_id": sample_id,
                "candidate_count": len(candidates),
                "candidates": record["candidates"],
            }
        )
        return record

    try:
        record.update(
            _raster_record(candidates[0], canonical_mode="L" if is_mask else "RGB")
        )
    except Exception as error:  # Pillow exposes several format-specific exceptions.
        record["status"] = "decode_error"
        record["error"] = f"{type(error).__name__}: {error}"
        issues.append(
            {
                "code": f"{kind}_raster_decode_error",
                "split": split,
                "sample_id": sample_id,
                "path": str(candidates[0]),
                "error": record["error"],
            }
        )
    return record


def _sample_record(
    dataset_root: Path,
    split: str,
    sample_id: str,
    issues: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    image = _resolve_raster(
        dataset_root, sample_id, kind="image", issues=issues, split=split
    )
    mask = _resolve_raster(
        dataset_root, sample_id, kind="mask", issues=issues, split=split
    )
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "image": image,
        "mask": mask,
        "shape_match": None,
        "mask_alignment": None,
        "pair_content_sha256": None,
    }
    if image["status"] == "unique" and mask["status"] == "unique":
        image_shape = (int(image["height"]), int(image["width"]))
        mask_shape = (int(mask["height"]), int(mask["width"]))
        record["shape_match"] = image_shape == mask_shape
        image_size_wh = (image_shape[1], image_shape[0])
        mask_size_wh = (mask_shape[1], mask_shape[0])
        aspect_error = relative_aspect_error(image_size_wh, mask_size_wh)
        alignment_required = not record["shape_match"]
        alignment_eligible = bool(
            alignment_required
            and aspect_error_within_tolerance(
                aspect_error, MASK_ALIGNMENT_ASPECT_TOLERANCE
            )
        )
        record["mask_alignment"] = {
            "required": alignment_required,
            "eligible": alignment_eligible,
            "policy": MASK_ALIGNMENT_POLICY,
            "aspect_tolerance": MASK_ALIGNMENT_ASPECT_TOLERANCE,
            "relative_aspect_error": aspect_error,
            "image_size_wh": list(image_size_wh),
            "original_mask_size_wh": list(mask_size_wh),
        }
        if not record["shape_match"]:
            finding = {
                "split": split,
                "sample_id": sample_id,
                "image_shape_hw": list(image_shape),
                "mask_shape_hw": list(mask_shape),
                "relative_aspect_error": aspect_error,
                "aspect_tolerance": MASK_ALIGNMENT_ASPECT_TOLERANCE,
                "alignment_policy": MASK_ALIGNMENT_POLICY,
            }
            if alignment_eligible:
                diagnostics.append(
                    {
                        "code": "image_mask_guarded_alignment_required",
                        "severity": "warning",
                        **finding,
                    }
                )
            else:
                issues.append(
                    {
                        "code": "image_mask_shape_mismatch",
                        "alignment_eligible": False,
                        **finding,
                    }
                )
        pair_digest = hashlib.sha256()
        pair_digest.update(bytes.fromhex(image["content_sha256"]))
        pair_digest.update(bytes.fromhex(mask["content_sha256"]))
        record["pair_content_sha256"] = pair_digest.hexdigest()
    return record


def _cross_split_duplicate_groups(
    train_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    value_getter,
) -> list[dict[str, Any]]:
    train_by_hash: dict[str, list[str]] = defaultdict(list)
    test_by_hash: dict[str, list[str]] = defaultdict(list)
    for record in train_records:
        value = value_getter(record)
        if value:
            train_by_hash[str(value)].append(str(record["sample_id"]))
    for record in test_records:
        value = value_getter(record)
        if value:
            test_by_hash[str(value)].append(str(record["sample_id"]))
    return [
        {
            "content_sha256": digest,
            "train_sample_ids": sorted(train_by_hash[digest]),
            "test_sample_ids": sorted(test_by_hash[digest]),
        }
        for digest in sorted(train_by_hash.keys() & test_by_hash.keys())
    ]


def _read_local_split(
    dataset_root: Path,
    split: str,
    issues: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    try:
        split_path = resolve_split_file(dataset_root, split=split)
        # This is defensive: automatic resolution is expected to be local, and
        # the audit refuses to follow a protocol manifest outside the dataset.
        split_path.relative_to(dataset_root)
        entries = read_split_file(split_path)
        sample_ids = ensure_unique_sample_ids(entries)
    except Exception as error:
        issues.append(
            {
                "code": "split_manifest_error",
                "split": split,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        return {
            "status": "error",
            "path": None,
            "file_sha256": None,
            "ordered_ids_sha256": None,
            "count": 0,
            "error": f"{type(error).__name__}: {error}",
        }, []
    return {
        "status": "ok",
        "path": str(split_path),
        "file_sha256": _sha256_file(split_path),
        "ordered_ids_sha256": _ordered_ids_sha256(sample_ids),
        "count": len(sample_ids),
    }, sample_ids


def audit_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    dataset_root = Path(dataset_dir).expanduser().resolve()
    issues: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    if not dataset_root.is_dir():
        issues.append(
            {
                "code": "dataset_directory_missing",
                "path": str(dataset_root),
            }
        )
        return {
            "dataset_name": dataset_root.name,
            "dataset_dir": str(dataset_root),
            "splits": {},
            "train_test_id_overlap": {"count": 0, "sample_ids": []},
            "samples": {"train": [], "test": []},
            "train_test_exact_content_duplicates": {
                "image": [],
                "mask": [],
                "image_mask_pair": [],
            },
            "issues": issues,
            "issue_count": len(issues),
            "diagnostics": diagnostics,
            "diagnostic_count": len(diagnostics),
            "guarded_alignment_count": 0,
            "strict_pass": False,
        }

    split_records: dict[str, dict[str, Any]] = {}
    ids_by_split: dict[str, list[str]] = {}
    for split in ("train", "test"):
        split_records[split], ids_by_split[split] = _read_local_split(
            dataset_root, split, issues
        )

    overlap = sorted(set(ids_by_split["train"]) & set(ids_by_split["test"]))
    if overlap:
        issues.append(
            {
                "code": "train_test_id_overlap",
                "count": len(overlap),
                "sample_ids": overlap,
            }
        )

    sample_records = {
        split: [
            _sample_record(dataset_root, split, sample_id, issues, diagnostics)
            for sample_id in ids_by_split[split]
        ]
        for split in ("train", "test")
    }
    duplicate_groups = {
        "image": _cross_split_duplicate_groups(
            sample_records["train"],
            sample_records["test"],
            lambda record: record["image"].get("content_sha256"),
        ),
        "mask": _cross_split_duplicate_groups(
            sample_records["train"],
            sample_records["test"],
            lambda record: record["mask"].get("content_sha256"),
        ),
        "image_mask_pair": _cross_split_duplicate_groups(
            sample_records["train"],
            sample_records["test"],
            lambda record: record.get("pair_content_sha256"),
        ),
    }
    for kind, groups in duplicate_groups.items():
        # Identical masks alone are common in sparse-target data (for example,
        # two one-pixel targets at the same coordinate) and do not prove image
        # leakage.  Keep them visible in the report, but only identical images
        # or complete image/mask pairs fail the split audit.
        if kind == "mask":
            continue
        for group in groups:
            issues.append(
                {
                    "code": f"train_test_exact_{kind}_content_duplicate",
                    **group,
                }
            )

    return {
        "dataset_name": dataset_root.name,
        "dataset_dir": str(dataset_root),
        "hash_algorithms": {
            "ordered_ids_sha256": _ORDERED_IDS_HASH_ALGORITHM,
            "raster_content_sha256": _RASTER_CONTENT_HASH_ALGORITHM,
            "pair_content_sha256": "sha256(image-content-digest + mask-content-digest)",
        },
        "splits": split_records,
        "train_test_id_overlap": {"count": len(overlap), "sample_ids": overlap},
        "samples": sample_records,
        "train_test_exact_content_duplicates": duplicate_groups,
        "issues": issues,
        "issue_count": len(issues),
        "diagnostics": diagnostics,
        "diagnostic_count": len(diagnostics),
        "guarded_alignment_count": sum(
            int(bool((record.get("mask_alignment") or {}).get("eligible")))
            for split in ("train", "test")
            for record in sample_records[split]
        ),
        "strict_pass": not issues,
    }


def build_report(
    dataset_dirs: Sequence[str | Path], *, allow_known_issues: bool = False
) -> dict[str, Any]:
    datasets = [audit_dataset(dataset_dir) for dataset_dir in dataset_dirs]
    issue_count = sum(int(dataset["issue_count"]) for dataset in datasets)
    diagnostic_count = sum(int(dataset["diagnostic_count"]) for dataset in datasets)
    strict_pass = issue_count == 0
    return {
        "schema_version": 2,
        "audit_mode": "read_only",
        "allow_known_issues": bool(allow_known_issues),
        "strict_pass": strict_pass,
        "status": (
            "pass"
            if strict_pass
            else ("issues_allowed_for_diagnostics" if allow_known_issues else "fail")
        ),
        "issue_count": issue_count,
        "diagnostic_count": diagnostic_count,
        "guarded_alignment_count": sum(
            int(dataset["guarded_alignment_count"]) for dataset in datasets
        ),
        "dataset_count": len(datasets),
        "datasets": datasets,
    }


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit local frozen train/test manifests and their raster files."
    )
    parser.add_argument(
        "--dataset-dir",
        action="append",
        required=True,
        help="Local dataset directory. Repeat this option to audit multiple datasets.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="JSON report path; the file is replaced atomically.",
    )
    parser.add_argument(
        "--allow-known-issues",
        action="store_true",
        help=(
            "Diagnostic-only escape hatch: retain and mark every issue in JSON but "
            "return exit status 0. This never repairs or re-partitions data."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        args.dataset_dir, allow_known_issues=bool(args.allow_known_issues)
    )
    atomic_write_json(args.output, report)
    return 0 if report["strict_pass"] or args.allow_known_issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
