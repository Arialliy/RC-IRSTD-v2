#!/usr/bin/env python3
"""Run the unchanged Tier2R exact gate under implementation erratum 1."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evaluation.artifact_integrity import file_sha256
from scripts import run_phase3_tier2r_exact_gate as g
from scripts import tier2r_impl_erratum1 as erratum


PROJECT_ROOT = erratum.PROJECT_ROOT
AUDIT_RELATIVE = erratum.NEW_AUDIT_RELATIVE
GATE_RELATIVE = erratum.NEW_GATE_RELATIVE

_ORIGINAL_VERIFY_PREREGISTRATION = g._verify_preregistration
_ORIGINAL_VALIDATE_HANDOFF = g.validate_handoff


def _verify_erratum_binding(payload: Mapping[str, Any], *, label: str) -> None:
    erratum.ensure_implementation_erratum(create=False)
    binding = payload.get("implementation_erratum")
    expected = erratum.implementation_erratum_binding()
    if (
        payload.get("execution_instance") != erratum.EXECUTION_INSTANCE
        or payload.get("implementation_erratum_plan")
        != erratum.implementation_erratum_plan()
        or not isinstance(binding, Mapping)
        or dict(binding) != expected
        or binding.get("path") != str(erratum.ERRATUM_PATH.resolve())
        or binding.get("sha256") != file_sha256(erratum.ERRATUM_PATH)
        or payload.get("source_only") is not True
        or payload.get("outer_target_images_used") is not False
        or payload.get("outer_target_labels_used") is not False
        or payload.get("outer_target_access_authorized") is not False
    ):
        raise RuntimeError(f"{label} implementation-erratum binding drift")


def _verify_preregistration_impl(
    root: Path,
    protocol: Mapping[str, Any],
    protocol_path: Path,
) -> dict[str, Any]:
    preregistration = _ORIGINAL_VERIFY_PREREGISTRATION(
        root, protocol, protocol_path
    )
    _verify_erratum_binding(preregistration, label="preregistration")
    return preregistration


def _validate_handoff_impl(
    handoff_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]], dict[str, Any]]:
    handoff, runs, protocol = _ORIGINAL_VALIDATE_HANDOFF(
        handoff_path, project_root=project_root
    )
    _verify_erratum_binding(handoff, label="handoff")
    preregistration = g._verify_frozen_json(
        Path(project_root).resolve()
        / AUDIT_RELATIVE
        / g.PREREGISTRATION_NAME
    )
    if handoff.get("implementation_erratum") != preregistration.get(
        "implementation_erratum"
    ):
        raise RuntimeError("handoff/preregistration erratum cross-binding drift")
    return handoff, runs, protocol


@contextmanager
def _patched_gate() -> Any:
    """Install the erratum namespace only for one public gate operation."""

    original_audit = g.AUDIT_RELATIVE
    original_gate = g.GATE_RELATIVE
    original_required = g.REQUIRED_CODE_BINDINGS
    original_verify = g._verify_preregistration
    original_handoff = g.validate_handoff
    g.AUDIT_RELATIVE = AUDIT_RELATIVE
    g.GATE_RELATIVE = GATE_RELATIVE
    g.REQUIRED_CODE_BINDINGS = set(original_required) | set(
        erratum.RECOVERY_CODE_RELATIVES
    )
    g._verify_preregistration = _verify_preregistration_impl
    g.validate_handoff = _validate_handoff_impl
    try:
        yield
    finally:
        g.validate_handoff = original_handoff
        g._verify_preregistration = original_verify
        g.REQUIRED_CODE_BINDINGS = original_required
        g.GATE_RELATIVE = original_gate
        g.AUDIT_RELATIVE = original_audit


def validate_handoff(
    handoff_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]], dict[str, Any]]:
    with _patched_gate():
        return _validate_handoff_impl(
            handoff_path, project_root=project_root
        )


def run_gate(
    *,
    handoff_path: str | Path,
    output_root: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    **kwargs: Any,
) -> dict[str, Any]:
    with _patched_gate():
        return g.run_gate(
            handoff_path=handoff_path,
            output_root=output_root,
            project_root=project_root,
            **kwargs,
        )


def verify_frozen_gate(
    *,
    handoff_path: str | Path,
    output_root: str | Path,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    with _patched_gate():
        return g.verify_frozen_gate(
            handoff_path=handoff_path,
            output_root=output_root,
            project_root=project_root,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--handoff",
        default=str(PROJECT_ROOT / AUDIT_RELATIVE / g.HANDOFF_NAME),
    )
    parser.add_argument(
        "--output-root", default=str(PROJECT_ROOT / GATE_RELATIVE)
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_only:
        result = verify_frozen_gate(
            handoff_path=args.handoff,
            output_root=args.output_root,
        )
    else:
        result = run_gate(
            handoff_path=args.handoff,
            output_root=args.output_root,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_gate", "validate_handoff", "verify_frozen_gate"]
