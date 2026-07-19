from __future__ import annotations

import json
from pathlib import Path

import pytest

from rc_irstd.cli import build_episodes


def _manifest(*, split_role: str = "test", target: str = "target") -> dict[str, object]:
    return {
        "target_dataset": target,
        "source_datasets": ["source-a", "source-b"],
        "checkpoint_diagnostic_only": False,
        "non_strict_state_loading": False,
        "spatial_mode": "native",
        "split_role": split_role,
        "split_authority_verified": True,
    }


def _install_fake_archive_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_verify(score_dir: str, **_: object):
        manifest = json.loads(
            (Path(score_dir) / "manifest.json").read_text(encoding="utf-8")
        )
        return manifest, [], {"verified": True}

    def fake_build(
        score_directories: list[str], output_dir: str, **_: object
    ) -> Path:
        assert score_directories
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "episodes.npz"
        archive.write_bytes(b"synthetic-archive")
        (root / "metadata.json").write_text(
            json.dumps({"archive": archive.name}), encoding="utf-8"
        )
        return archive

    monkeypatch.setattr(build_episodes, "verify_score_map_directory", fake_verify)
    monkeypatch.setattr(build_episodes, "build_episode_archive", fake_build)


def _run_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: dict[str, object],
    *extra: str,
) -> dict[str, object]:
    _install_fake_archive_builder(monkeypatch)
    score_dir = tmp_path / "scores"
    score_dir.mkdir(parents=True)
    (score_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    output_dir = tmp_path / "episodes"
    result = build_episodes.main(
        [
            "--score-dir",
            str(score_dir),
            "--output-dir",
            str(output_dir),
            "--support-size",
            "1",
            "--query-size",
            "1",
            *extra,
        ]
    )
    assert result == 0
    return json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))


def test_explicit_train_split_is_formal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = _run_builder(
        tmp_path,
        monkeypatch,
        _manifest(split_role="train"),
        "--expected-split-role",
        "train",
    )

    assert metadata["formal_causal_contract"] is True
    assert metadata["diagnostic_only"] is False
    assert metadata["pseudo_target_split"] == "train"
    assert metadata["expected_split_role"] == "train"


def test_default_test_split_remains_formal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _run_builder(tmp_path, monkeypatch, _manifest(split_role="test"))

    assert metadata["formal_causal_contract"] is True
    assert metadata["diagnostic_only"] is False
    assert metadata["pseudo_target_split"] == "test"
    assert metadata["expected_split_role"] == "test"


def test_split_mismatch_is_rejected_unless_explicitly_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_archive_builder(monkeypatch)
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    (score_dir / "manifest.json").write_text(
        json.dumps(_manifest(split_role="train")), encoding="utf-8"
    )
    output_dir = tmp_path / "rejected"
    common = [
        "--score-dir",
        str(score_dir),
        "--output-dir",
        str(output_dir),
        "--support-size",
        "1",
        "--query-size",
        "1",
    ]

    with pytest.raises(ValueError, match="diagnostic detector/spatial protocol"):
        build_episodes.main(common)

    metadata = _run_builder(
        tmp_path / "allowed",
        monkeypatch,
        _manifest(split_role="train"),
        "--allow-diagnostic-detector",
    )
    assert metadata["formal_causal_contract"] is False
    assert metadata["diagnostic_only"] is True
    assert metadata["pseudo_target_split"] == "train"
    assert metadata["expected_split_role"] == "test"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_diagnostic_only", True),
        ("non_strict_state_loading", True),
        ("spatial_mode", "resize"),
        ("split_authority_verified", False),
    ],
)
def test_formal_builder_still_requires_strict_native_verified_detector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    manifest = _manifest()
    manifest[field] = value
    _install_fake_archive_builder(monkeypatch)
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    (score_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="diagnostic detector/spatial protocol"):
        build_episodes.main(
            [
                "--score-dir",
                str(score_dir),
                "--output-dir",
                str(tmp_path / "episodes"),
                "--support-size",
                "1",
                "--query-size",
                "1",
            ]
        )


def test_lodo_target_leakage_remains_a_hard_error() -> None:
    manifest = _manifest(target="source-a")
    with pytest.raises(ValueError, match="appears in detector source domains"):
        build_episodes._audit_lodo_manifest(
            manifest,
            "scores",
            expected_split_role="test",
            allow_diagnostic_detector=True,
        )
