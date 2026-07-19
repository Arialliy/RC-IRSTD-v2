from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

import scripts.coordinate_phase3_source_lodo_gate as coordinator


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _temporary_layout(tmp_path: Path) -> coordinator.Layout:
    layout = coordinator.Layout(root=tmp_path)
    layout.configs.mkdir(parents=True)
    for name in coordinator.BASE_CONFIGS.values():
        shutil.copyfile(ROOT / "configs" / name, layout.configs / name)
    for name in ("NUDT-SIRST", "IRSTD-1K", "NUAA-SIRST"):
        root = layout.datasets / name
        (root / "img_idx").mkdir(parents=True)
        (root / "img_idx" / f"train_{name}.txt").write_text(
            f"{name}-train-0\n{name}-train-1\n", encoding="utf-8"
        )
        (root / "img_idx" / f"test_{name}.txt").write_text(
            f"{name}-test-0\n", encoding="utf-8"
        )
    layout.initializers.mkdir(parents=True)
    for fold in coordinator.FOLDS.values():
        (layout.initializers / fold.initializer_name).write_bytes(
            ("initializer:" + fold.key).encode("utf-8")
        )
    return layout


def _fake_model(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    model_config = {"backend": "rc_mshnet", "test_contract": 1}

    class FakeModel:
        def export_config(self) -> dict[str, Any]:
            return dict(model_config)

        def load_state_dict(self, state: Any, *, strict: bool) -> None:
            if strict is not True or state != {"weight": 1}:
                raise RuntimeError("synthetic strict-load failure")

    monkeypatch.setattr(coordinator, "build_mshnet", lambda _config: FakeModel())
    return model_config


def _write_checkpoint_run(
    layout: coordinator.Layout,
    spec: coordinator.RunSpec,
    monkeypatch: pytest.MonkeyPatch,
    *,
    epoch: int,
) -> tuple[Path, dict[str, Any]]:
    model_config = _fake_model(monkeypatch)
    config = coordinator.expected_config(layout, spec)
    formal = coordinator._ensure_formal_config(layout, spec)
    assert yaml.safe_load(formal.read_text(encoding="utf-8")) == config
    run_dir = coordinator._run_dir(layout, spec)
    _write_json(run_dir / "config.json", config)
    initializer = layout.initializers / spec.fold.initializer_name
    initialization = {
        "source_path": str(initializer.resolve()),
        "source_sha256": coordinator.file_sha256(initializer),
        "backbone_fully_loaded": True,
        "unexpected_keys": [],
        "zero_residual_identity_preserved": True,
    }
    _write_json(run_dir / "initialization_report.json", initialization)
    source_root = Path(config["data"]["sources"][0]["path"])
    train_path = coordinator.resolve_split_file(source_root, None, split="train")
    test_path = coordinator.resolve_split_file(source_root, None, split="test")
    train_ids = coordinator.ensure_unique_sample_ids(
        coordinator.read_split_file(train_path)
    )
    test_ids = coordinator.ensure_unique_sample_ids(
        coordinator.read_split_file(test_path)
    )
    split_record = {
        "name": spec.fold.train_name,
        "path": str(source_root),
        "train_split_file": str(train_path),
        "train_split_file_sha256": coordinator.file_sha256(train_path),
        "train_ordered_ids_sha256": coordinator.ordered_ids_sha256(train_ids),
        "num_train_samples": len(train_ids),
        "test_split_file": str(test_path),
        "test_split_file_sha256": coordinator.file_sha256(test_path),
        "test_ordered_ids_sha256": coordinator.ordered_ids_sha256(test_ids),
        "num_test_samples": len(test_ids),
        "train_test_id_overlap": False,
    }
    resume_contract = {
        "schema_version": 1,
        "seed": 42,
        "deterministic": True,
        "determinism_contract": {},
        "diagnostic_only": False,
        "data": {},
        "model": model_config,
        "loss": config["loss"],
        "optimizer": config["optimizer"],
        "training": {
            "epochs": 80,
            "warmup_epochs": 0,
            "grad_clip": 5.0,
            "amp": True,
            "min_lr": 0.000002,
        },
    }
    payload = {
        "format_version": 2,
        "kind": "detector",
        "epoch": epoch,
        "checkpoint_selection": "fixed_last",
        "selection_rule": "fixed_last",
        "test_labels_used_for_selection": False,
        "diagnostic_test_eval": False,
        "diagnostic_only": False,
        "formal_paper_checkpoint": True,
        "warm_flag": True,
        "inference_head": "multi_scale_fused",
        "config": config,
        "initialization": initialization,
        "source_names": [spec.fold.train_name],
        "source_split_records": [split_record],
        "model_config": model_config,
        "model_state": {"weight": 1},
        "optimizer_state": {},
        "scheduler_state": {},
        "scaler_state": {},
        "rng_state": {"torch_cuda": [torch.zeros(1, dtype=torch.uint8)]},
        "balanced_batcher_state": {},
        "resume_contract": resume_contract,
    }
    with (run_dir / "history.csv").open("w", encoding="utf-8") as handle:
        handle.write("epoch,lr,loss_total\n")
        for value in range(epoch + 1):
            handle.write(f"{value},0.0002,{1.0 / (value + 1):.8f}\n")
    torch.save(payload, run_dir / "last.pt")
    return run_dir, payload


def _selection(pd_nudt: float, pd_irstd: float, *, worst: float | None = None) -> dict[str, Any]:
    pooled_pd = (pd_nudt + pd_irstd) / 2.0
    return {
        "results": {
            "source_pooled": {
                "found": True,
                "operating_point": {"pd": pooled_pd, "threshold": 0.5},
                "source_rows": {
                    "NUDT-SIRST": {"pd": pd_nudt},
                    "IRSTD-1K": {"pd": pd_irstd},
                },
            },
            "source_worst": {
                "found": True,
                "worst_domain_pd": min(pd_nudt, pd_irstd) if worst is None else worst,
            },
        }
    }


def test_tier1_schedule_is_exactly_four_inner_lodo_runs_on_gpu_two_three() -> None:
    runs = [spec for round_specs in coordinator.TIER1_ROUNDS for spec in round_specs]
    assert len(runs) == 4
    assert {spec.role for spec in runs} == {"control", "full"}
    assert {spec.fold_key for spec in runs} == {"heldout_nudt", "heldout_irstd"}
    assert all(
        {spec.physical_gpu for spec in round_specs} == {2, 3}
        for round_specs in coordinator.TIER1_ROUNDS
    )
    assert all(spec.physical_gpu in {2, 3} for spec in runs)


def test_expected_config_is_source_only_matched_fixed_last() -> None:
    layout = coordinator.Layout()
    coordinator._validate_matched_tier1_configs(layout)
    for round_specs in coordinator.TIER1_ROUNDS:
        for spec in round_specs:
            config = coordinator.expected_config(layout, spec)
            assert config["seed"] == 42
            assert config["deterministic"] is True
            assert config["device"] == "cuda:0"
            assert config["data"]["sources"] == [
                {
                    "name": spec.fold.train_name,
                    "path": str((layout.datasets / spec.fold.train_dataset_dir).resolve()),
                }
            ]
            assert config["data"]["train_split"] == "train"
            assert config["data"]["val_split"] is None
            assert config["training"]["epochs"] == 80
            assert config["training"]["checkpoint_selection"] == "fixed_last"
            assert config["training"]["resume"] is None
            assert "NUAA-SIRST" not in json.dumps(config["data"])


def test_commands_are_source_only_and_resume_is_runtime_only(tmp_path: Path) -> None:
    layout = _temporary_layout(tmp_path)
    spec = coordinator.RunSpec("control", "heldout_nudt", 2)
    fresh = coordinator.training_command(
        layout, spec, coordinator.RunInspection("fresh")
    )
    assert "--resume-checkpoint" not in fresh
    assert fresh[fresh.index("--pid-file") + 1].endswith("TRAINER.pid")
    audit = coordinator._run_dir(layout, spec) / "audit.json"
    resume = coordinator.training_command(
        layout,
        spec,
        coordinator.RunInspection("resume", epoch=4),
        resume_audit=audit,
    )
    assert resume[resume.index("--resume-checkpoint") + 1].endswith("last.pt")
    assert resume[resume.index("--resume-audit") + 1] == str(audit.resolve())
    coordinator._assert_source_only_command(layout, resume)
    for forbidden in (
        ["python", "run_pipeline.py"],
        ["python", "-m", "rc_irstd.cli.run_pipeline"],
        ["python", "tool", "--target-curve", "target.csv"],
        ["python", "tool", "--dataset-name", "NUAA-SIRST"],
    ):
        with pytest.raises(RuntimeError, match="Forbidden"):
            coordinator._assert_source_only_command(layout, forbidden)


def test_phase2_incomplete_returns_false_before_any_source_config_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _temporary_layout(tmp_path)
    _write_json(layout.phase2_status, {"status": "running"})
    monkeypatch.setattr(
        coordinator,
        "expected_config",
        lambda *_args: (_ for _ in ()).throw(AssertionError("source config opened")),
    )
    assert coordinator.phase2_complete_and_verified(layout) is False


def test_phase2_complete_requires_hold_six_identities_and_live_hashes(tmp_path: Path) -> None:
    layout = _temporary_layout(tmp_path)
    layout.state.mkdir(parents=True)
    (layout.state / "HOLD_RC_MSHNET_GATE").write_text("HOLD\n", encoding="utf-8")
    reports: dict[str, Any] = {}
    for role, relative in coordinator.PHASE2_OUTPUTS.items():
        run_dir = layout.root / relative
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "last.pt"
        checkpoint.write_bytes(("checkpoint:" + role).encode("utf-8"))
        identity = {"role": role, "checkpoint_sha256": coordinator.file_sha256(checkpoint)}
        _write_json(run_dir / "PHASE2_IDENTITY.json", identity)
        reports[role] = identity
    status = {
        "schema_version": "rc-irstd-aaai27-phase2-v2",
        "status": "completed",
        "gate_state": "HOLD",
        "risk_curve_started": False,
        "physical_gpus": [2, 3],
        "next_gate": "detector_matched_fa_evaluation_and_go_no_go",
        "runs": reports,
    }
    _write_json(layout.phase2_status, status)
    assert coordinator.phase2_complete_and_verified(layout) is True
    first = layout.root / next(iter(coordinator.PHASE2_OUTPUTS.values())) / "last.pt"
    first.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="changed"):
        coordinator.phase2_complete_and_verified(layout)


def test_unknown_nonempty_run_fails_without_injecting_formal_config(tmp_path: Path) -> None:
    layout = _temporary_layout(tmp_path)
    spec = coordinator.RunSpec("control", "heldout_nudt", 2)
    run_dir = coordinator._run_dir(layout, spec)
    run_dir.mkdir(parents=True)
    marker = run_dir / "unknown.bin"
    marker.write_bytes(b"immutable evidence")
    before = marker.read_bytes()
    with pytest.raises(RuntimeError, match="failed closed"):
        coordinator._prepare_launch_state(layout, spec)
    assert marker.read_bytes() == before
    assert not (run_dir / "formal_config.yaml").exists()


def test_zero_epoch_half_product_is_preserved_and_never_fresh_restarted(
    tmp_path: Path,
) -> None:
    layout = _temporary_layout(tmp_path)
    spec = coordinator.RunSpec("control", "heldout_nudt", 2)
    coordinator._ensure_formal_config(layout, spec)
    marker = coordinator._run_dir(layout, spec) / "trainer-started.log"
    marker.write_bytes(b"failed before first committed epoch")
    with pytest.raises(RuntimeError, match="zero-epoch half product"):
        coordinator._prepare_launch_state(layout, spec)
    assert marker.read_bytes() == b"failed before first committed epoch"
    assert not (coordinator._run_dir(layout, spec) / "history.csv").exists()


def test_exact_checkpoint_inspection_resume_complete_and_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _temporary_layout(tmp_path)
    spec = coordinator.RunSpec("full", "heldout_irstd", 3)
    run_dir, payload = _write_checkpoint_run(
        layout, spec, monkeypatch, epoch=4
    )
    assert coordinator.inspect_run(layout, spec) == coordinator.RunInspection(
        "resume", epoch=4
    )
    payload["rng_state"] = {"torch_cuda": []}
    torch.save(payload, run_dir / "last.pt")
    inspection = coordinator.inspect_run(layout, spec)
    assert inspection.state == "corrupt"
    assert "CUDA RNG" in str(inspection.reason)

    shutil.rmtree(run_dir)
    run_dir, _ = _write_checkpoint_run(layout, spec, monkeypatch, epoch=79)
    assert coordinator.inspect_run(layout, spec) == coordinator.RunInspection(
        "complete", epoch=79
    )
    (run_dir / "config.json").write_text("{}\n", encoding="utf-8")
    assert coordinator.inspect_run(layout, spec).state == "corrupt"


def test_manual_complete_checkpoint_without_launch_provenance_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _temporary_layout(tmp_path)
    spec = coordinator.RunSpec("control", "heldout_irstd", 2)
    _write_checkpoint_run(layout, spec, monkeypatch, epoch=79)
    with pytest.raises(RuntimeError, match="successful launch provenance"):
        coordinator._write_run_identity(layout, spec)


def test_cleanly_failed_resume_audit_is_not_automatically_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _temporary_layout(tmp_path)
    spec = coordinator.RunSpec("full", "heldout_nudt", 3)
    run_dir, _ = _write_checkpoint_run(layout, spec, monkeypatch, epoch=4)
    attempts = run_dir / "launch_attempts"
    attempts.mkdir()
    attempt_id = "b" * 32
    checkpoint = run_dir / "last.pt"
    history = run_dir / "history.csv"
    audit_path = attempts / f"{attempt_id}.resume_audit.json"
    source = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": coordinator.file_sha256(checkpoint),
        "epoch": 4,
        "history_rows": 5,
        "history_sha256": coordinator.file_sha256(history),
        "audit": str(audit_path.resolve()),
    }
    intent = {
        "schema_version": coordinator.LAUNCH_SCHEMA,
        "attempt_id": attempt_id,
        "run_id": spec.run_id,
        "mode": "resume",
        "resume_source": source,
        "resume_audit": str(audit_path.resolve()),
    }
    intent_path = attempts / f"{attempt_id}.intent.json"
    _write_json(intent_path, intent)
    log_path = attempts / f"{attempt_id}.log"
    log_path.write_text("formal resume failed\n", encoding="utf-8")
    binding = {
        "attempt_id": attempt_id,
        "run_id": spec.run_id,
        "pid": 12345,
        "intent_sha256": coordinator.file_sha256(intent_path),
        "log": str(log_path.resolve()),
    }
    _write_json(attempts / f"{attempt_id}.binding.json", binding)
    final_state = {
        "checkpoint_sha256": source["checkpoint_sha256"],
        "checkpoint_epoch": 4,
        "history_rows": 5,
        "history_sha256": source["history_sha256"],
    }
    _write_json(
        attempts / f"{attempt_id}.result.json",
        {
            "attempt_id": attempt_id,
            "run_id": spec.run_id,
            "pid": 12345,
            "return_code": None,
            "log_sha256": coordinator.file_sha256(log_path),
            "final_state": final_state,
        },
    )
    _write_json(
        audit_path,
        {
            "schema_version": "rc-irstd-formal-resume-v1",
            "status": "failed",
            "events": [{"status": "failed"}],
        },
    )
    with pytest.raises(RuntimeError, match="clean failure"):
        coordinator._validate_finished_attempts(layout, spec)


def test_ambiguous_stage_intent_fails_closed_without_launching(tmp_path: Path) -> None:
    layout = _temporary_layout(tmp_path)
    spec = coordinator.RunSpec("control", "heldout_nudt", 2)
    stage_dir = coordinator._run_dir(layout, spec) / "stage_attempts" / "test_stage"
    stage_dir.mkdir(parents=True)
    _write_json(stage_dir / ("a" * 32 + ".intent.json"), {"attempt_id": "a" * 32})
    with pytest.raises(RuntimeError, match="ambiguous intent"):
        coordinator._run_checked(
            layout,
            spec,
            ["/bin/true"],
            stage="test_stage",
            uses_gpu=False,
        )


def test_registered_cpu_stage_records_intent_binding_and_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _temporary_layout(tmp_path)
    spec = coordinator.RunSpec("control", "heldout_nudt", 2)
    monkeypatch.setattr(coordinator.time, "sleep", lambda _seconds: None)
    command = [sys.executable, "-c", "import time; time.sleep(0.1)"]
    provenance = coordinator._run_checked(
        layout,
        spec,
        command,
        stage="cpu_smoke",
        uses_gpu=False,
    )
    assert len(provenance["attempt_id"]) == 32
    assert Path(provenance["binding"]).is_file()
    assert Path(provenance["result"]).is_file()
    binding = json.loads(Path(provenance["binding"]).read_text(encoding="utf-8"))
    assert binding["cuda_visible_devices"] == ""
    assert binding["logical_device"] is None
    assert binding["cwd"] == str(layout.root.resolve())
    assert not coordinator._stage_active_path(layout, spec, "cpu_smoke").exists()


def test_active_training_binding_cannot_smuggle_target_or_pipeline_command(
    tmp_path: Path,
) -> None:
    layout = _temporary_layout(tmp_path)
    spec = coordinator.RunSpec("control", "heldout_nudt", 2)
    formal = coordinator._ensure_formal_config(layout, spec)
    run_dir = coordinator._run_dir(layout, spec)
    attempts = run_dir / "launch_attempts"
    attempts.mkdir()
    attempt_id = "c" * 32
    command = ["python", "run_pipeline.py", "--dataset-name", "NUAA-SIRST"]
    intent = {
        "schema_version": coordinator.LAUNCH_SCHEMA,
        "attempt_id": attempt_id,
        "run_id": spec.run_id,
        "mode": "fresh",
        "physical_gpu": 2,
        "logical_device": "cuda:0",
        "command": command,
        "command_sha256": coordinator._canonical_json_sha256(command),
        "formal_config_sha256": coordinator.file_sha256(formal),
        "resume_audit": None,
    }
    intent_path = attempts / f"{attempt_id}.intent.json"
    _write_json(intent_path, intent)
    log_path = attempts / f"{attempt_id}.log"
    log_path.write_bytes(b"")
    binding = {
        "schema_version": coordinator.LAUNCH_SCHEMA,
        "attempt_id": attempt_id,
        "run_id": spec.run_id,
        "mode": "fresh",
        "pid": 99999999,
        "proc_start_ticks": 1,
        "physical_gpu": 2,
        "cuda_visible_devices": "2",
        "logical_device": "cuda:0",
        "command": command,
        "command_sha256": coordinator._canonical_json_sha256(command),
        "formal_config_sha256": coordinator.file_sha256(formal),
        "intent_sha256": coordinator.file_sha256(intent_path),
        "log": str(log_path.resolve()),
        "cwd": str(layout.root.resolve()),
    }
    _write_json(attempts / f"{attempt_id}.binding.json", binding)
    _write_json(run_dir / "ACTIVE_LAUNCH.json", binding)
    with pytest.raises(RuntimeError, match="Forbidden"):
        coordinator._active_launch(layout, spec)


def test_incomplete_export_is_preserved_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _temporary_layout(tmp_path)
    spec = coordinator.RunSpec("control", "heldout_nudt", 2)
    score_dir = coordinator._score_dir(layout, spec)
    score_dir.mkdir(parents=True)
    marker = score_dir / "partial.npy"
    marker.write_bytes(b"partial")
    monkeypatch.setattr(
        coordinator,
        "_run_checked",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("incomplete evidence must not relaunch")
        ),
    )
    with pytest.raises(RuntimeError, match="fails closed"):
        coordinator.ensure_export(layout, spec)
    assert marker.read_bytes() == b"partial"


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        (lambda values: None, "GO"),
        (
            lambda values: values["full"].__setitem__(
                "medium", _selection(0.60, 0.60, worst=0.40)
            ),
            "HOLD",
        ),
        (
            lambda values: values["full"].__setitem__(
                "strict", _selection(0.699, 0.721)
            ),
            "HOLD",
        ),
    ],
)
def test_tier1_decision_boundaries(mutator: Any, expected: str) -> None:
    selections = {
        "control": {
            name: _selection(0.70, 0.70) for name, _, _ in coordinator.BUDGETS
        },
        "full": {
            name: _selection(0.71, 0.71) for name, _, _ in coordinator.BUDGETS
        },
    }
    mutator(selections)
    decision = coordinator.build_tier1_decision(selections)
    assert decision["decision"] == expected
    assert decision["authorizes_outer_target_label_access"] is False


def test_preregistration_has_executable_future_rules_and_code_data_bindings() -> None:
    payload = coordinator._prereg_payload(coordinator.Layout())
    assert payload["threshold_protocol"][
        "grid_sha256"
    ] == coordinator.evaluation_threshold_grid_sha256(
        coordinator.build_default_thresholds()
    )
    assert set(payload["source_dataset_bindings"]) == {"NUDT-SIRST", "IRSTD-1K"}
    assert payload["tier2_policy"]["missing_or_infeasible_point"] == "HOLD_TIER2"
    assert payload["tier2_policy"]["numeric_tolerance"] == coordinator.NUMERIC_ATOL
    assert payload["tier3_policy"]["missing_or_infeasible_point"] == "HOLD_TIER3"
    assert payload["tier3_policy"]["branch_aux_role"].startswith("diagnostic_only")
    assert "evaluation/threshold_sweep.py" in payload["code_bindings"]


def test_freeze_is_idempotent_and_never_unlocks_target_labels(tmp_path: Path) -> None:
    layout = coordinator.Layout(root=tmp_path)
    layout.state.mkdir(parents=True)
    layout.audit.mkdir(parents=True)
    decision = {
        "schema_version": coordinator.DECISION_SCHEMA,
        "decision": "GO",
        "authorizes_outer_target_label_access": False,
    }
    path = coordinator.freeze_decision(layout, decision)
    sentinel = layout.state / "PHASE3_SOURCE_TIER1_GO"
    hold = layout.state / "HOLD_PHASE3_TARGET_LABEL_ACCESS"
    mtimes = (path.stat().st_mtime_ns, sentinel.stat().st_mtime_ns, hold.stat().st_mtime_ns)
    coordinator.freeze_decision(layout, decision)
    assert mtimes == (
        path.stat().st_mtime_ns,
        sentinel.stat().st_mtime_ns,
        hold.stat().st_mtime_ns,
    )
    assert sentinel.read_text(encoding="utf-8") == (
        f"GO {coordinator.file_sha256(path)}\n"
    )
    assert hold.read_text(encoding="utf-8") == "HOLD\n"
    conflicting = dict(decision, decision="HOLD")
    with pytest.raises(RuntimeError):
        coordinator.freeze_decision(layout, conflicting)
