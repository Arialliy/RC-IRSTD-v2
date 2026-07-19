#!/usr/bin/env python3
"""Emit container evidence for the Tier2R implementation-erratum execution."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Iterator


PROJECT_ROOT = Path("/home/ly/RC-IRSTD-v2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import probe_phase3_tier2r_container as base  # noqa: E402


OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs/aaai27/detectors/component_rescue/tier2r_c_v1"
)
AUDIT_ROOT = (
    PROJECT_ROOT
    / "artifacts/aaai27/audit/component_rescue/tier2r_c_v1_impl_erratum1"
)
TARGET_ROOT = PROJECT_ROOT / "datasets/NUAA-SIRST"
MOUNT_POINTS = (PROJECT_ROOT, OUTPUT_ROOT, AUDIT_ROOT, TARGET_ROOT)


@contextmanager
def _erratum_probe_scope() -> Iterator[None]:
    saved = {
        "OUTPUT_ROOT": base.OUTPUT_ROOT,
        "AUDIT_ROOT": base.AUDIT_ROOT,
        "TARGET_ROOT": base.TARGET_ROOT,
        "MOUNT_POINTS": base.MOUNT_POINTS,
    }
    try:
        base.OUTPUT_ROOT = OUTPUT_ROOT
        base.AUDIT_ROOT = AUDIT_ROOT
        base.TARGET_ROOT = TARGET_ROOT
        base.MOUNT_POINTS = MOUNT_POINTS
        yield
    finally:
        for name, value in saved.items():
            setattr(base, name, value)


def probe() -> dict[str, object]:
    with _erratum_probe_scope():
        payload = base.probe()
    payload["schema_version"] = (
        "rc-irstd-aaai27-tier2r-container-attestation-impl-erratum1-v1"
    )
    payload["scientific_protocol_id"] = "tier2r_c_v1"
    payload["execution_instance_id"] = "tier2r_c_v1_impl_erratum1"
    return payload


def main() -> int:
    print(json.dumps(probe(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
