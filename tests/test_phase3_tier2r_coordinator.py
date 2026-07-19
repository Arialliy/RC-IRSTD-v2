from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts import coordinate_phase3_tier2r_component_rescue as coordinator
from scripts import run_phase3_tier2r_exact_gate as gate


def test_three_configs_have_explicit_component_expert_and_frozen_roles() -> None:
    expected = coordinator.EXPECTED_MODEL_FLAGS
    for role, relative in coordinator.ROLE_CONFIGS.items():
        payload = yaml.safe_load((coordinator.PROJECT_ROOT / relative).read_text())
        model = payload["model"]
        assert model["architecture_version"] == "rc-mshnet-v2-component-role-split"
        for field, value in expected[role].items():
            assert type(model[field]) is bool
            assert model[field] is value
        assert payload["training"]["epochs"] == 80
        assert payload["training"]["checkpoint_selection"] == "fixed_last"


def test_schedule_is_18_jobs_seed_local_and_fixed_to_gpu_2_3() -> None:
    protocol, _ = gate._load_protocol(coordinator.PROJECT_ROOT)
    rounds = coordinator.build_schedule(protocol)
    jobs = [job for group in rounds for job in group]

    assert len(rounds) == 9
    assert len(jobs) == 18
    assert {job.run_id for job in jobs} == {spec.run_id for spec in gate.RUN_SPECS}
    assert all({job.physical_gpu for job in group} == {2, 3} for group in rounds)
    assert all(len({job.seed for job in group}) == 1 for group in rounds)
    assert all(
        job.physical_gpu
        == gate.expected_physical_gpu(job.seed, job.role, job.fold.key)
        for job in jobs
    )


def test_protocol_forbids_idle_wait_and_gpu_fallback() -> None:
    protocol, _ = gate._load_protocol(coordinator.PROJECT_ROOT)
    training = protocol["training_protocol"]
    assert training["physical_gpus"] == [2, 3]
    assert training["wait_for_idle_gpu"] is False
    assert training["allow_gpu_fallback"] is False
    assert training["checkpoint_selection"] == "fixed_last"
    assert training["checkpoint_epoch_zero_based"] == 79


@pytest.fixture(scope="module")
def prereg_packet() -> dict:
    return coordinator.verify_prerequisites()


def _first_job(packet: dict) -> coordinator.JobSpec:
    return coordinator.build_schedule(packet["protocol_payload"])[0][0]


def _write_checkpoint(
    job: coordinator.JobSpec, packet: dict, run_dir: Path, epoch: int
) -> None:
    payload = {
        "epoch": epoch,
        "config": coordinator._expected_formal_config(job, packet),
        "model_config": packet["expected_model_configs"][job.role],
        "initialization": {
            "source_sha256": packet["initializers"][job.fold.key]["sha256"],
            "initial_extension_state_sha256": "a" * 64,
            "extension_state_preserved": True,
        },
        "checkpoint_selection": "fixed_last",
        "selection_rule": "fixed_last",
        "test_labels_used_for_selection": False,
        "diagnostic_test_eval": False,
    }
    torch.save(payload, run_dir / "last.pt")


def test_history_ahead_one_is_repaired_only_after_identity_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prereg_packet: dict
) -> None:
    monkeypatch.setattr(coordinator, "OUTPUT_ROOT", tmp_path / "runs")
    job = _first_job(prereg_packet)
    coordinator._ensure_formal_config(job, prereg_packet)
    run_dir = coordinator._run_dir(job)
    _write_checkpoint(job, prereg_packet, run_dir, epoch=1)
    (run_dir / "history.csv").write_text("epoch\n0\n1\n2\n", encoding="utf-8")
    assert coordinator.inspect_run(job, prereg_packet).state == "history_ahead_one"
    archive = coordinator._repair_history_ahead_one(job, prereg_packet)
    assert archive.is_file()
    assert coordinator._history_epochs(run_dir / "history.csv") == [0, 1]
    assert coordinator.inspect_run(job, prereg_packet).state == "resume"


def test_zero_epoch_partial_is_isolated_before_clean_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prereg_packet: dict
) -> None:
    monkeypatch.setattr(coordinator, "OUTPUT_ROOT", tmp_path / "runs")
    job = _first_job(prereg_packet)
    coordinator._ensure_formal_config(job, prereg_packet)
    run_dir = coordinator._run_dir(job)
    (run_dir / "history.csv").write_text("epoch\n0\n", encoding="utf-8")
    assert coordinator.inspect_run(job, prereg_packet).state == "zero_epoch_partial"
    archive = coordinator._isolate_zero_epoch_partial(job, prereg_packet)
    assert archive.is_dir()
    assert not run_dir.exists()
    coordinator._ensure_formal_config(job, prereg_packet)
    assert coordinator.inspect_run(job, prereg_packet).state == "fresh"


def test_partial_export_is_moved_to_failed_attempt_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prereg_packet: dict
) -> None:
    monkeypatch.setattr(coordinator, "OUTPUT_ROOT", tmp_path / "runs")
    job = _first_job(prereg_packet)
    score_dir = coordinator._run_dir(job) / "scores_heldout_train"
    score_dir.mkdir(parents=True)
    (score_dir / "partial.bin").write_bytes(b"partial")
    archive = coordinator._isolate_partial_export(job)
    assert archive.is_dir()
    assert (archive / "partial.bin").read_bytes() == b"partial"
    assert not score_dir.exists()


def test_export_manifest_requires_all_frozen_raw_logit_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prereg_packet: dict
) -> None:
    monkeypatch.setattr(coordinator, "OUTPUT_ROOT", tmp_path / "runs")
    job = _first_job(prereg_packet)
    run_dir = coordinator._run_dir(job)
    score_dir = run_dir / "scores_heldout_train"
    score_dir.mkdir(parents=True)
    checkpoint = run_dir / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest_path = score_dir / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    digest = coordinator.file_sha256(checkpoint)
    payload = {
        "labels_loaded": True,
        "split_role": "train",
        "requested_split": "train",
        "target_dataset": job.fold.held_out_source,
        "source_datasets": [job.fold.training_source],
        "spatial_mode": "native",
        "score_representation": "raw_logit_float32+sigmoid_probability_float32",
        "probability_dtype": "float32",
        "logit_dtype": "float32",
        "inference_autocast_enabled": False,
        "checkpoint_warm_flag": True,
        "weight_sha256": digest,
    }
    contract = {"detector_weight_sha256": digest}
    monkeypatch.setattr(
        coordinator, "verify_score_map_directory", lambda *args, **kwargs: (payload, {}, {})
    )
    monkeypatch.setattr(
        coordinator, "validate_formal_score_manifest", lambda *args, **kwargs: contract
    )
    assert coordinator._export_manifest_valid(job, manifest_path) is True
    payload["inference_autocast_enabled"] = True
    assert coordinator._export_manifest_valid(job, manifest_path) is False
    payload["inference_autocast_enabled"] = False
    payload["score_representation"] = "probability_only"
    assert coordinator._export_manifest_valid(job, manifest_path) is False


def test_immutable_registration_restarts_idempotently_and_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "registration.json"
    monkeypatch.setattr(coordinator, "_now", lambda: "2026-07-16T00:00:00+08:00")
    payload = {"registered_at": coordinator._registered_at(path), "source_only": True}
    first = coordinator._write_once_json(path, payload)
    second = coordinator._write_once_json(path, payload)
    assert first == second
    assert coordinator._registered_at(path) == payload["registered_at"]
    with pytest.raises(RuntimeError, match="immutable Tier2R artifact drift"):
        coordinator._write_once_json(path, {**payload, "source_only": False})


def test_stale_checkpoint_temp_is_archived_only_after_resume_identity_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prereg_packet: dict
) -> None:
    monkeypatch.setattr(coordinator, "OUTPUT_ROOT", tmp_path / "runs")
    job = _first_job(prereg_packet)
    coordinator._ensure_formal_config(job, prereg_packet)
    run_dir = coordinator._run_dir(job)
    _write_checkpoint(job, prereg_packet, run_dir, epoch=1)
    (run_dir / "history.csv").write_text("epoch\n0\n1\n", encoding="utf-8")
    temporary = run_dir / "last.pt.tmp"
    temporary.write_bytes(b"interrupted-write")

    command = coordinator._training_command(job, prereg_packet, attempt=1)

    assert not temporary.exists()
    archives = list((run_dir / ".stale_checkpoint_temp_archive").glob("*.pt.tmp"))
    assert len(archives) == 1
    assert archives[0].read_bytes() == b"interrupted-write"
    assert "--resume-checkpoint" in command
    assert coordinator.inspect_run(job, prereg_packet).state == "resume"


def test_stale_checkpoint_temp_recovery_fails_closed_without_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prereg_packet: dict
) -> None:
    monkeypatch.setattr(coordinator, "OUTPUT_ROOT", tmp_path / "runs")
    job = _first_job(prereg_packet)
    coordinator._ensure_formal_config(job, prereg_packet)
    run_dir = coordinator._run_dir(job)
    (run_dir / "last.pt.tmp").write_bytes(b"unproved")
    with pytest.raises(RuntimeError, match="without proved identity"):
        coordinator._isolate_stale_checkpoint_temp(job, prereg_packet)


def test_all_children_use_current_canonical_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prereg_packet: dict
) -> None:
    expected = str(Path(sys.executable).resolve(strict=True))
    assert str(coordinator._python_executable()) == expected
    monkeypatch.setattr(coordinator, "OUTPUT_ROOT", tmp_path / "runs")
    job = _first_job(prereg_packet)

    training = coordinator._training_command(job, prereg_packet, attempt=1)
    exporting = coordinator._export_command(job)
    assert training[0] == expected
    assert exporting[0] == expected

    audit_root = tmp_path / "audit"
    audit_root.mkdir()
    monkeypatch.setattr(coordinator, "AUDIT_ROOT", audit_root)
    handoff = tmp_path / "handoff.json"
    handoff.write_text("{}\n", encoding="utf-8")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(coordinator.subprocess, "run", fake_run)
    monkeypatch.setattr(
        coordinator.exact_gate, "verify_frozen_gate", lambda **kwargs: {"verified": True}
    )
    assert coordinator._run_exact_gate(handoff) == {"verified": True}
    assert captured["command"][0] == expected

    source = Path(coordinator.__file__).read_text(encoding="utf-8")
    assert "BasicIRSTD" not in source
    assert "EXPECTED_PYTHON" not in source
