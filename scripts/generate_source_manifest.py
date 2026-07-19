#!/usr/bin/env python3
"""Generate a deterministic SHA-256 manifest for commit-eligible source files."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess


def _candidate_paths(root: Path) -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    paths: list[Path] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative_text = raw.decode("utf-8")
        if any(character in relative_text for character in "\r\n"):
            raise ValueError(f"Manifest paths cannot contain newlines: {relative_text!r}")
        relative = Path(relative_text)
        if relative.as_posix() == "MANIFEST.sha256":
            continue
        path = root / relative
        if path.is_symlink():
            raise ValueError(f"Source manifest refuses symbolic links: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(relative)
    return sorted(paths, key=lambda value: value.as_posix())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="MANIFEST.sha256")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    if output.parent != root:
        raise ValueError("The source manifest must be written in the repository root")
    lines = [
        f"{_sha256(root / relative)}  ./{relative.as_posix()}"
        for relative in _candidate_paths(root)
    ]
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(f"{output}: {len(lines)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
