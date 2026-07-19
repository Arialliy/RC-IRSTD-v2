from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from rc_irstd.cli import train_detector


_AUDIT_EVENT_FIELDS = {
    "owner_pid",
    "source_checkpoint_path",
    "immutable_snapshot_path",
    "source_checkpoint_sha256",
    "source_epoch",
    "resume_start_epoch",
    "cuda_visible_devices",
    "status",
}


def _formal_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "_config_path": str((tmp_path / "detector.yaml").resolve()),
        "_config_dir": str(tmp_path.resolve()),
        "output_dir": str((tmp_path / "run").resolve()),
        "training": {
            "checkpoint_selection": "fixed_last",
            "epochs": 8,
            # Formal Phase-2 runs keep their shared initializer provenance even
            # though exact runtime resume must not reload it a second time.
            "initialize_from": "/frozen/shared-initializer.pt",
            # The resume checkpoint is a launcher/runtime decision.  It must
            # never become part of the preregistered effective configuration.
            "resume": None,
        },
    }


def _write_constructor_config(config: dict[str, Any]) -> Path:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(public, sort_keys=True) + "\n", encoding="utf-8")
    return config_path


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _assert_resume_audit(path: Path, *, status: str) -> dict[str, Any]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    assert audit["schema_version"] == "rc-irstd-formal-resume-v1"
    assert audit["status"] == status
    assert isinstance(audit["events"], list) and audit["events"]
    for event in audit["events"]:
        assert _AUDIT_EVENT_FIELDS <= set(event)
        assert event["owner_pid"] == os.getpid()
        assert Path(event["source_checkpoint_path"]).is_absolute()
        assert Path(event["immutable_snapshot_path"]).is_absolute()
        assert len(event["source_checkpoint_sha256"]) == 64
        assert event["source_epoch"] == 7
        assert event["resume_start_epoch"] == 8
        assert isinstance(event["status"], str) and event["status"]
    return audit


def _install_fake_resume_preflight(
    monkeypatch: pytest.MonkeyPatch,
    *,
    output_dir: Path,
    source_checkpoint: Path,
) -> Path:
    snapshot = output_dir / "resume_snapshots" / "epoch_0007_snapshot.pt"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(source_checkpoint.read_bytes())

    def fake_prepare(
        _config: dict[str, object],
        checkpoint: Path,
    ) -> tuple[Path, Path, dict[str, object]]:
        source = checkpoint.expanduser().resolve()
        return (
            output_dir.resolve(),
            snapshot.resolve(),
            {
                "owner_pid": os.getpid(),
                "source_checkpoint_path": str(source),
                "immutable_snapshot_path": str(snapshot.resolve()),
                "source_checkpoint_sha256": "a" * 64,
                "source_epoch": 7,
                "resume_start_epoch": 8,
                "cuda_visible_devices": "2",
                "status": "prepared",
            },
        )

    monkeypatch.setattr(train_detector, "_prepare_formal_resume", fake_prepare)
    return snapshot.resolve()


def test_parser_accepts_runtime_resume_checkpoint() -> None:
    args = train_detector.build_parser().parse_args(
        [
            "--config",
            "detector.yaml",
            "--resume-checkpoint",
            "run/last.pt",
            "--pid-file",
            "run/trainer.pid",
            "--resume-audit",
            "run/resume.json",
        ]
    )

    assert args.resume_checkpoint == Path("run/last.pt")
    assert args.pid_file == Path("run/trainer.pid")
    assert args.resume_audit == Path("run/resume.json")


def test_runtime_resume_is_absolute_only_during_trainer_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal = _formal_config(tmp_path)
    formal_snapshot = copy.deepcopy(formal)
    resume_checkpoint = tmp_path / "relative" / "last.pt"
    resume_checkpoint.parent.mkdir()
    torch.save({"epoch": 7}, resume_checkpoint)
    pid_file = tmp_path / "trainer.pid"
    observed: dict[str, Any] = {}

    class FakeTrainer:
        def __init__(self, config: dict[str, Any]) -> None:
            observed["constructor_resume"] = config["training"]["resume"]
            observed["constructor_initialize_from"] = config["training"][
                "initialize_from"
            ]
            observed["trainer"] = self
            self.config = config
            self.config_path = _write_constructor_config(config)
            observed["pid_during_constructor"] = pid_file.read_text(
                encoding="utf-8"
            ).strip()

        def run(self) -> Path:
            # The real trainer serializes self.config into last.pt during run().
            # It must already have been restored to the formal value by then.
            observed["run_resume"] = self.config["training"]["resume"]
            observed["run_initialize_from"] = self.config["training"][
                "initialize_from"
            ]
            checkpoint = Path(self.config["output_dir"]) / "last.pt"
            with (checkpoint.parent / "history.csv").open(
                "w", encoding="utf-8"
            ) as handle:
                handle.write("epoch,lr,loss_total\n")
                for epoch in range(8):
                    handle.write(f"{epoch},0.001,1.0\n")
            torch.save(
                {"epoch": 7, "config": _public_config(self.config)}, checkpoint
            )
            return checkpoint

    monkeypatch.setattr(train_detector, "load_yaml", lambda _path: formal)
    monkeypatch.setattr(train_detector, "DetectorTrainer", FakeTrainer)
    snapshot = _install_fake_resume_preflight(
        monkeypatch,
        output_dir=Path(formal["output_dir"]),
        source_checkpoint=resume_checkpoint,
    )
    monkeypatch.chdir(tmp_path)

    assert (
        train_detector.main(
            [
                "--config",
                str(tmp_path / "detector.yaml"),
                "--resume-checkpoint",
                str(resume_checkpoint.relative_to(tmp_path)),
                "--pid-file",
                str(pid_file),
            ]
        )
        == 0
    )

    assert observed["constructor_resume"] == str(snapshot)
    assert observed["constructor_initialize_from"] is None
    assert observed["pid_during_constructor"] == str(os.getpid())
    assert observed["run_resume"] is None
    assert observed["run_initialize_from"] == "/frozen/shared-initializer.pt"
    assert observed["trainer"].config["training"]["resume"] is None
    assert _public_config(observed["trainer"].config) == _public_config(formal)
    assert formal == formal_snapshot
    disk_config = json.loads(
        (Path(formal["output_dir"]) / "config.json").read_text(encoding="utf-8")
    )
    assert "resume" in disk_config["training"]
    assert disk_config["training"]["resume"] is None
    assert disk_config == _public_config(formal)
    _assert_resume_audit(
        Path(formal["output_dir"]) / "RESUME_AUDIT.json", status="completed"
    )


@pytest.mark.parametrize("failure_phase", ["constructor", "run"])
def test_runtime_resume_failure_restores_config_and_disk_artifact(
    failure_phase: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal = _formal_config(tmp_path)
    formal_snapshot = copy.deepcopy(formal)
    resume_checkpoint = tmp_path / "resume" / "last.pt"
    resume_checkpoint.parent.mkdir()
    torch.save({"epoch": 7}, resume_checkpoint)
    audit_path = tmp_path / "explicit-resume-audit.json"
    observed: dict[str, Any] = {}

    class FailingTrainer:
        def __init__(self, config: dict[str, Any]) -> None:
            observed["constructor_resume"] = config["training"]["resume"]
            observed["constructor_initialize_from"] = config["training"][
                "initialize_from"
            ]
            # Keep the mutable mapping even when __init__ raises.  This checks
            # restoration of the exact mapping handed to DetectorTrainer.
            observed["trainer_config"] = config
            self.config = config
            _write_constructor_config(config)
            if failure_phase == "constructor":
                raise RuntimeError("synthetic constructor failure")

        def run(self) -> Path:
            observed["run_resume"] = self.config["training"]["resume"]
            if failure_phase == "run":
                raise RuntimeError("synthetic run failure")
            raise AssertionError("unreachable")

    monkeypatch.setattr(train_detector, "load_yaml", lambda _path: formal)
    monkeypatch.setattr(train_detector, "DetectorTrainer", FailingTrainer)
    snapshot = _install_fake_resume_preflight(
        monkeypatch,
        output_dir=Path(formal["output_dir"]),
        source_checkpoint=resume_checkpoint,
    )

    with pytest.raises(RuntimeError, match=f"synthetic {failure_phase} failure"):
        train_detector.main(
            [
                "--config",
                str(tmp_path / "detector.yaml"),
                "--resume-checkpoint",
                str(resume_checkpoint),
                "--resume-audit",
                str(audit_path),
            ]
        )

    assert observed["constructor_resume"] == str(snapshot)
    assert observed["constructor_initialize_from"] is None
    if failure_phase == "run":
        assert observed["run_resume"] is None
        assert observed["trainer_config"]["training"]["resume"] is None
        assert _public_config(observed["trainer_config"]) == _public_config(formal)
    else:
        assert observed["trainer_config"]["training"]["resume"] is None
        assert _public_config(observed["trainer_config"]) == _public_config(formal)
    assert formal == formal_snapshot
    disk_config = json.loads(
        (Path(formal["output_dir"]) / "config.json").read_text(encoding="utf-8")
    )
    assert "resume" in disk_config["training"]
    assert disk_config["training"]["resume"] is None
    assert disk_config == _public_config(formal)
    audit = _assert_resume_audit(audit_path, status="failed")
    assert audit["error_type"] == "RuntimeError"
    assert audit["error"] == f"synthetic {failure_phase} failure"
    failed_event = audit["events"][-1]
    assert failed_event["error_type"] == "RuntimeError"
    assert failed_event["error"] == f"synthetic {failure_phase} failure"
