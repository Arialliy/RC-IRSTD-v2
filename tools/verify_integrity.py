#!/usr/bin/env python3
"""Verify all files listed in the package SHA256SUMS manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    failures: list[str] = []
    checked = 0
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        digest, separator, relative = raw.partition("  ")
        if not separator or len(digest) != 64:
            failures.append(f"line {line_number}: malformed")
            continue
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            failures.append(f"sha256 mismatch: {relative}")
        checked += 1
    if failures:
        for failure in failures:
            print(failure)
        return 1
    print(f"integrity=passed files={checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
