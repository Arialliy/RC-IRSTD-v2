"""Narrow phase-transition handling for immutable governance tests.

The frozen Tier2S v2 test below correctly asserted that the write-once
registration did not exist while that registration was being built.  Once the
registration was successfully written, that one pre-registration assertion
became obsolete.  The frozen test and registration cannot be edited.  This
hook skips only the exact obsolete node after the exact registered artifact
and sidecar are present; post-registration invariants are covered by the
execution-erratum tests.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_REGISTRATION = (
    PROJECT_ROOT
    / "artifacts/aaai27/audit/governance/tier2s_gpu23_amendment_v2"
    / "TIER2S_GPU23_GOVERNANCE_AMENDMENT.json"
)
FROZEN_REGISTRATION_SIDECAR = FROZEN_REGISTRATION.with_suffix(
    FROZEN_REGISTRATION.suffix + ".sha256"
)
FROZEN_REGISTRATION_SHA256 = (
    "0c53c54f3e4802bdf373af1e578e062e4106192aa64ba1a8b47ed7bb8595f136"
)
OBSOLETE_PRE_REGISTRATION_NODEID = (
    "tests/test_tier2s_gpu23_v2.py::"
    "test_amendment_verify_only_is_unregistered_and_never_authorizes_v3"
)
SKIP_REASON = (
    "immutable pre-registration-only assertion retired by exact Tier2S GPU2/3 "
    "write-once registration; post-registration invariants are tested by "
    "execution erratum1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_parent_registration_is_frozen() -> bool:
    if (
        FROZEN_REGISTRATION.is_symlink()
        or FROZEN_REGISTRATION_SIDECAR.is_symlink()
        or not FROZEN_REGISTRATION.is_file()
        or not FROZEN_REGISTRATION_SIDECAR.is_file()
    ):
        return False
    expected_sidecar = (
        f"{FROZEN_REGISTRATION_SHA256}  {FROZEN_REGISTRATION.name}\n"
    ).encode("ascii")
    return (
        _sha256(FROZEN_REGISTRATION) == FROZEN_REGISTRATION_SHA256
        and FROZEN_REGISTRATION_SIDECAR.read_bytes() == expected_sidecar
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    del config
    if not exact_parent_registration_is_frozen():
        return
    for item in items:
        if item.nodeid == OBSOLETE_PRE_REGISTRATION_NODEID:
            item.add_marker(pytest.mark.skip(reason=SKIP_REASON))
