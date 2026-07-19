from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_phase3_tier2r_exact_gate as gate


def test_validate_source_path_accepts_only_registered_tree(tmp_path: Path) -> None:
    allowed = tmp_path / "source"
    allowed.mkdir()
    sample = allowed / "sample.bin"
    sample.write_bytes(b"source")
    forbidden = tmp_path / "target"
    forbidden.mkdir()

    assert gate.validate_source_path(
        sample, allowed_roots=[allowed], forbidden_root=forbidden
    ) == sample.resolve()

    with pytest.raises(RuntimeError, match="outside the source-root allowlist"):
        gate.validate_source_path(
            forbidden, allowed_roots=[allowed], forbidden_root=forbidden
        )


def test_validate_source_path_rejects_traversal_and_symlink(tmp_path: Path) -> None:
    allowed = tmp_path / "source"
    allowed.mkdir()
    real = allowed / "real"
    real.mkdir()
    link = allowed / "link"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises(RuntimeError, match="parent traversal"):
        gate.validate_source_path(
            allowed / ".." / "source", allowed_roots=[allowed]
        )
    with pytest.raises(RuntimeError, match="symlink path component"):
        gate.validate_source_path(link, allowed_roots=[allowed])


def test_preregistered_protocol_keeps_outer_target_locked() -> None:
    protocol = json.loads(
        (gate.PROJECT_ROOT / gate.PROTOCOL_RELATIVE).read_text(encoding="utf-8")
    )
    access = protocol["data_access"]
    assert access["allowed_source_roots"] == [
        "datasets/NUDT-SIRST",
        "datasets/IRSTD-1K",
    ]
    assert access["forbidden_outer_target_root"] == "datasets/NUAA-SIRST"
    assert access["outer_target_images_authorized"] is False
    assert access["outer_target_labels_authorized"] is False
    assert access["outer_target_images_used"] is False
    assert access["outer_target_labels_used"] is False
