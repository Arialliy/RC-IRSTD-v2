#!/usr/bin/env python3
"""Verify that a patched checkout exposes and executes RC-MSHNet."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--run-pytest", action="store_true")
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    required = [
        "rc_irstd/models/rc_mshnet.py",
        "configs/detector_rc_mshnet_fast_aaai.yaml",
        "tests/test_rc_mshnet.py",
        "scripts/smoke_rc_mshnet.py",
    ]
    missing = [path for path in required if not (repo / path).is_file()]
    if missing:
        raise FileNotFoundError("missing patched files: " + ", ".join(missing))

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    commands = [
        [sys.executable, str(repo / "scripts/smoke_rc_mshnet.py"), "--size", "32"],
    ]
    if args.run_pytest:
        commands.append(
            [sys.executable, "-m", "pytest", "-q", "tests/test_rc_mshnet.py"]
        )
    results: list[dict[str, object]] = []
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        results.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        if completed.returncode != 0:
            print(json.dumps(results, indent=2, ensure_ascii=False))
            return completed.returncode
    print(json.dumps({"status": "passed", "results": results}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
