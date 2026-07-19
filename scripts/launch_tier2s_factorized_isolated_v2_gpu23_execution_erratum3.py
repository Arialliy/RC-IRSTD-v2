#!/usr/bin/env python3
"""Direct-file Tier2S entrypoint delegating to frozen execution erratum2.

This wrapper performs only the missing exact project-root bootstrap, then
rebinds the latest append-only governance/code inputs into the frozen erratum2
launcher. Scientific, GPU, container, isolation, and scheduling behavior is
delegated unchanged.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import functools
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any, Iterator

PROJECT_ROOT = Path("/home/ly/RC-IRSTD-v2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import launch_tier2s_factorized_isolated_v2_gpu23_execution_erratum2 as parent
from scripts import register_tier2s_gpu23_execution_erratum3 as execution_erratum3


EXECUTION_ERRATUM_CONFIG = (
    PROJECT_ROOT / "configs/tier2s_gpu23_execution_erratum3.json"
)
EXECUTION_ERRATUM_REGISTRATION_RELATIVE = Path(
    "artifacts/aaai27/audit/governance/tier2s_gpu23_execution_erratum3/"
    "TIER2S_GPU23_EXECUTION_ERRATUM3.json"
)
REGISTRATION = PROJECT_ROOT / EXECUTION_ERRATUM_REGISTRATION_RELATIVE
REGISTRATION_SIDECAR = REGISTRATION.with_suffix(REGISTRATION.suffix + ".sha256")
REGISTRAR = PROJECT_ROOT / "scripts/register_tier2s_gpu23_execution_erratum3.py"
TEST_PATH = PROJECT_ROOT / "tests/test_tier2s_gpu23_execution_erratum3.py"

CODE_PATHS = tuple(
    dict.fromkeys(
        (
            *parent.CODE_PATHS,
            Path(__file__).resolve(),
            REGISTRAR,
            TEST_PATH,
            REGISTRATION,
            REGISTRATION_SIDECAR,
        )
    )
)
CONFIG_PATHS = tuple(
    dict.fromkeys((*parent.CONFIG_PATHS, EXECUTION_ERRATUM_CONFIG))
)

_SCOPE_LOCK = threading.RLock()
_SCOPE_DEPTH = 0
_SCOPE_SAVED: dict[str, Any] = {}


def _require_execution_erratum_binding() -> dict[str, Any]:
    return execution_erratum3.require_frozen_execution_erratum3()


def _overrides() -> dict[str, Any]:
    return {
        "CODE_PATHS": CODE_PATHS,
        "CONFIG_PATHS": CONFIG_PATHS,
        "EXECUTION_ERRATUM_CONFIG": EXECUTION_ERRATUM_CONFIG,
        "EXECUTION_ERRATUM_REGISTRATION_RELATIVE": (
            EXECUTION_ERRATUM_REGISTRATION_RELATIVE
        ),
        "_require_execution_erratum_binding": _require_execution_erratum_binding,
    }


@contextmanager
def _parent_scope() -> Iterator[None]:
    global _SCOPE_DEPTH, _SCOPE_SAVED
    with _SCOPE_LOCK:
        if _SCOPE_DEPTH == 0:
            values = _overrides()
            _SCOPE_SAVED = {name: getattr(parent, name) for name in values}
            for name, value in values.items():
                setattr(parent, name, value)
        _SCOPE_DEPTH += 1
        try:
            yield
        finally:
            _SCOPE_DEPTH -= 1
            if _SCOPE_DEPTH == 0:
                for name, value in _SCOPE_SAVED.items():
                    setattr(parent, name, value)
                _SCOPE_SAVED = {}


def _scoped(function: Any) -> Any:
    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with _parent_scope():
            return function(*args, **kwargs)

    return wrapped


@_scoped
def build_verify_command() -> list[str]:
    return parent.build_verify_command()


@_scoped
def build_probe_command() -> list[str]:
    return parent.build_probe_command()


@_scoped
def build_register_command() -> list[str]:
    return parent.build_register_command()


@_scoped
def build_formal_command(intent_sha256: str) -> list[str]:
    return parent.build_formal_command(intent_sha256)


@_scoped
def container_spec(*, formal: bool) -> dict[str, Any]:
    return parent.container_spec(formal=formal)


@_scoped
def verify_only(*, runner: parent.parent.base.Runner | None = None) -> dict[str, Any]:
    return parent.verify_only(runner=runner)


@_scoped
def execute_launch(*, runner: parent.parent.base.Runner | None = None) -> dict[str, Any]:
    return parent.execute_launch(runner=runner)


@_scoped
def dry_run_payload() -> dict[str, Any]:
    payload = parent.dry_run_payload()
    payload["execute_with"] = (
        f"python3 {Path(__file__).resolve()} --execute"
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.execute:
            payload = execute_launch()
        elif args.verify_only:
            payload = verify_only()
        else:
            payload = dry_run_payload()
    except BaseException as error:
        print(f"FAILED_CLOSED {type(error).__name__}: {error}", file=os.sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

