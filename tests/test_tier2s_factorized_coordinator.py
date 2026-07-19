from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from scripts import coordinate_tier2s_factorized_audit as coordinator


def _bind_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, Path]:
    project = tmp_path / "project"
    audit = project / "audit"
    output = project / "outputs"
    project.mkdir()
    paths = {
        "project": project,
        "audit": audit,
        "output": output,
        "preregistration": audit / "PREREGISTRATION.json",
        "queue": audit / "QUEUE_MANIFEST.json",
        "target_lock": audit / "OUTER_TARGET_ACCESS_DENIED.json",
        "status": audit / "STATUS.json",
        "handoff": audit / "EXPORT_HANDOFF.json",
        "events": audit / "scheduler_events.jsonl",
        "events_sidecar": audit / "scheduler_events.jsonl.sha256",
        "lock": audit / ".coordinator.lock",
    }
    for name, value in (
        ("PROJECT_ROOT", paths["project"]),
        ("AUDIT_ROOT", paths["audit"]),
        ("OUTPUT_ROOT", paths["output"]),
        ("PREREGISTRATION", paths["preregistration"]),
        ("QUEUE_MANIFEST", paths["queue"]),
        ("TARGET_LOCK", paths["target_lock"]),
        ("STATUS_PATH", paths["status"]),
        ("HANDOFF_PATH", paths["handoff"]),
        ("EVENT_LOG", paths["events"]),
        ("EVENT_LOG_SIDECAR", paths["events_sidecar"]),
        ("LOCK_PATH", paths["lock"]),
    ):
        monkeypatch.setattr(coordinator, name, value)
    return paths


def _synthetic_protocol() -> dict[str, Any]:
    return {
        "folds": {
            "heldout_nudt": {
                "training_source": "IRSTD-1K",
                "held_out_source": "NUDT-SIRST",
                "training_root": "datasets/IRSTD-1K",
                "held_out_root": "datasets/NUDT-SIRST",
            },
            "heldout_irstd": {
                "training_source": "NUDT-SIRST",
                "held_out_source": "IRSTD-1K",
                "training_root": "datasets/NUDT-SIRST",
                "held_out_root": "datasets/IRSTD-1K",
            },
        }
    }


def _synthetic_parent_handoff(project: Path) -> dict[str, Any]:
    parent = project / "parent"
    parent.mkdir()
    runs: dict[str, dict[str, Any]] = {}
    for seed in coordinator.SEEDS:
        for role in coordinator.ROLES:
            for fold in coordinator.FOLDS:
                run_id = f"seed{seed}_{role}_{fold}"
                checkpoint = parent / f"{run_id}.pt"
                score_manifest = parent / f"{run_id}.scores.json"
                checkpoint.write_bytes(f"checkpoint:{run_id}\n".encode("ascii"))
                score_manifest.write_bytes(
                    f'{{"run_id":"{run_id}"}}\n'.encode("ascii")
                )
                runs[run_id] = {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": coordinator._sha256(checkpoint),
                    "score_manifest": str(score_manifest),
                    "score_manifest_sha256": coordinator._sha256(score_manifest),
                }
    return {"runs": runs}


def _governance_binding() -> dict[str, Any]:
    def artifact(name: str, token: str) -> dict[str, str]:
        return {
            "path": f"audit/{name}.json",
            "sha256": token * 64,
            "sidecar_path": f"audit/{name}.json.sha256",
            "sidecar_sha256": token.upper() * 64,
        }

    return {
        "schema_version": "rc-irstd-aaai27-tier2s-governance-binding-v1",
        "registration": artifact("registration", "a"),
        "contract": artifact("contract", "b"),
        "fresh_seed_ledger": artifact("fresh_seed_ledger", "c"),
        "fresh_seed_local_scan": artifact("fresh_seed_local_scan", "e"),
        "code_sha256_canonical_sha256": "d" * 64,
        "tier2s_source_only_diagnostic_authorized": True,
        "formal_v3_model_training_authorized": False,
        "source_gate_a_authorized": False,
        "riskcurve_authorized": False,
        "outer_target_access_authorized": False,
    }


def _synthetic_packet(project: Path) -> dict[str, Any]:
    jobs = coordinator._build_jobs(
        _synthetic_protocol(), _synthetic_parent_handoff(project)
    )
    return {
        "schema_version": coordinator.SCHEMA,
        "verified": True,
        "protocol_id": coordinator.PROTOCOL_ID,
        "source_only": True,
        "outer_target_access_authorized": False,
        "governance_binding": _governance_binding(),
        "parent_evidence": {"decision": {"sha256": "d" * 64}},
        "schedule": [asdict(job) for job in jobs],
        "lane_lengths": {
            str(gpu): sum(job.physical_gpu == gpu for job in jobs)
            for gpu in coordinator.PHYSICAL_GPUS
        },
    }


def _install_governance(
    monkeypatch: pytest.MonkeyPatch, packet: dict[str, Any]
) -> None:
    expected = packet["governance_binding"]["registration"]["sha256"]

    def require(
        *, expected_registration_sha256: str | None = None
    ) -> dict[str, Any]:
        if (
            expected_registration_sha256 is not None
            and expected_registration_sha256 != expected
        ):
            raise RuntimeError("synthetic governance SHA drift")
        return packet["governance_binding"]

    monkeypatch.setattr(coordinator, "_require_frozen_tier2s_governance", require)


def _freeze(path: Path) -> None:
    path.chmod(0o444)


def _complete_recovery_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[dict[str, Any], dict[str, Path]]:
    paths = _bind_paths(monkeypatch, tmp_path)
    packet = _synthetic_packet(paths["project"])
    _install_governance(monkeypatch, packet)
    monkeypatch.setattr(coordinator, "_now", lambda: "2026-07-17T00:00:00+00:00")
    coordinator.register(packet)

    events = coordinator.EventWriter(paths["events"])
    events.append({"event": "all_exports_completed", "completed_jobs": 18})
    event_log_binding = coordinator._freeze_event_log()

    results = []
    for item in packet["schedule"]:
        job = coordinator.ExportJob(**item)
        manifest = Path(job.output_dir) / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_bytes(f'{{"run_id":"{job.run_id}"}}\n'.encode("ascii"))
        sidecar = manifest.with_suffix(".sha256")
        sidecar.write_text(
            f"{coordinator._sha256(manifest)}  manifest.json\n", encoding="ascii"
        )
        _freeze(manifest)
        _freeze(sidecar)
        results.append({"event": "job_completed", "run_id": job.run_id})

    coordinator._write_once_json(
        paths["handoff"], coordinator._handoff(packet, results, event_log_binding)
    )
    return packet, paths


def _force_drift(path: Path) -> None:
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b"drift\n")
    path.chmod(0o444)


def test_register_twice_preserves_timestamp_hashes_and_read_only_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _bind_paths(monkeypatch, tmp_path)
    packet = {
        "governance_binding": _governance_binding(),
        "schedule": [],
        "parent_evidence": {"decision": {"sha256": "d" * 64}},
    }
    timestamps = iter(["2026-07-17T00:00:00+00:00"])
    monkeypatch.setattr(coordinator, "_now", lambda: next(timestamps))

    first = coordinator.register(packet)
    second = coordinator.register(packet)

    assert second == first
    assert coordinator._load_json(paths["preregistration"])["registered_at"] == (
        "2026-07-17T00:00:00+00:00"
    )
    artifacts = (
        paths["preregistration"],
        paths["queue"],
        paths["target_lock"],
    )
    for artifact in artifacts:
        sidecar = artifact.with_suffix(artifact.suffix + ".sha256")
        assert artifact.stat().st_mode & 0o222 == 0
        assert sidecar.stat().st_mode & 0o222 == 0


def test_synthetic_parent_builds_the_frozen_18_job_two_lane_schedule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _bind_paths(monkeypatch, tmp_path)
    packet = _synthetic_packet(paths["project"])
    jobs = [coordinator.ExportJob(**item) for item in packet["schedule"]]

    expected = {
        0: [
            "seed43_c_heldout_nudt_held_out",
            "seed43_c_heldout_nudt_held_in",
            "seed43_control_heldout_nudt_held_in",
            "seed44_c_heldout_irstd_held_out",
            "seed44_c_heldout_irstd_held_in",
            "seed44_control_heldout_irstd_held_in",
            "seed45_c_heldout_nudt_held_out",
            "seed45_c_heldout_nudt_held_in",
            "seed45_control_heldout_nudt_held_in",
        ],
        1: [
            "seed43_c_heldout_irstd_held_out",
            "seed43_c_heldout_irstd_held_in",
            "seed43_control_heldout_irstd_held_in",
            "seed44_c_heldout_nudt_held_out",
            "seed44_c_heldout_nudt_held_in",
            "seed44_control_heldout_nudt_held_in",
            "seed45_c_heldout_irstd_held_out",
            "seed45_c_heldout_irstd_held_in",
            "seed45_control_heldout_irstd_held_in",
        ],
    }
    observed = {
        gpu: [
            job.run_id
            for job in sorted(jobs, key=lambda value: value.queue_index)
            if job.physical_gpu == gpu
        ]
        for gpu in coordinator.PHYSICAL_GPUS
    }
    assert len(jobs) == 18
    assert Counter(job.physical_gpu for job in jobs) == Counter({0: 9, 1: 9})
    assert observed == expected
    assert all(
        [job.queue_index for job in jobs if job.physical_gpu == gpu]
        == list(range(len(expected[gpu])))
        for gpu in coordinator.PHYSICAL_GPUS
    )


def test_export_argv_binds_fold_and_scope_exactly_once() -> None:
    job = coordinator.ExportJob(
        run_id="seed43_c_heldout_nudt_held_out",
        checkpoint_run_id="seed43_c_heldout_nudt",
        seed=43,
        role="c",
        fold="heldout_nudt",
        scope="held_out",
        dataset_name="NUDT-SIRST",
        dataset_root="/synthetic/NUDT-SIRST",
        checkpoint="/synthetic/checkpoint.pt",
        checkpoint_sha256="c" * 64,
        parent_heldout_score_manifest="/synthetic/scores.json",
        parent_heldout_score_manifest_sha256="s" * 64,
        physical_gpu=0,
        queue_index=0,
        output_dir="/synthetic/output",
    )

    registration = {
        "tier2s_preregistration_binding": {
            "governance_registration_sha256": "a" * 64,
            "sha256": "e" * 64,
        }
    }
    command = coordinator._export_command(job, registration)

    assert command == [
        str(coordinator.PYTHON_EXECUTABLE),
        str(coordinator.EXPORTER),
        "--protocol",
        str(coordinator.PROTOCOL_PATH),
        "--governance-registration-sha256",
        "a" * 64,
        "--tier2s-preregistration",
        str(coordinator.PREREGISTRATION),
        "--tier2s-preregistration-sha256",
        "e" * 64,
        "--checkpoint",
        job.checkpoint,
        "--dataset-dir",
        job.dataset_root,
        "--dataset-name",
        job.dataset_name,
        "--split",
        "train",
        "--output-dir",
        job.output_dir,
        "--device",
        "cuda:0",
        "--expected-role",
        job.role,
        "--fold",
        job.fold,
        "--scope",
        job.scope,
    ]
    assert command.count("--fold") == command.count("--scope") == 1


def test_verify_existing_handoff_accepts_a_complete_consistent_artifact_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    packet, paths = _complete_recovery_state(monkeypatch, tmp_path)

    assert coordinator._verify_existing_handoff(packet) == coordinator._sha256(
        paths["handoff"]
    )


@pytest.mark.parametrize(
    ("drift_kind", "message"),
    [
        ("manifest", "manifest drift"),
        ("events", "freeze drift"),
        ("preregistration", "sidecar drift"),
    ],
)
def test_verify_existing_handoff_fails_closed_on_bound_sha_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    drift_kind: str,
    message: str,
) -> None:
    packet, paths = _complete_recovery_state(monkeypatch, tmp_path)
    if drift_kind == "manifest":
        target = Path(packet["schedule"][0]["output_dir"]) / "manifest.json"
    else:
        target = paths[drift_kind]
    _force_drift(target)

    with pytest.raises(RuntimeError, match=message):
        coordinator._verify_existing_handoff(packet)


def test_execute_with_existing_handoff_skips_export_lanes_and_runs_evaluator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    packet, _ = _complete_recovery_state(monkeypatch, tmp_path)
    lane_calls: list[object] = []
    evaluator_calls: list[bool] = []

    def forbidden_lane(*args: object, **kwargs: object) -> list[dict[str, Any]]:
        lane_calls.append((args, kwargs))
        raise AssertionError("an existing handoff must bypass GPU export lanes")

    def evaluator(registration: dict[str, Any]) -> dict[str, Any]:
        assert registration["tier2s_preregistration_binding"]["sha256"] == (
            coordinator._sha256(coordinator.PREREGISTRATION)
        )
        evaluator_calls.append(True)
        return {
            "outer_target_access_authorized": False,
            "source_tier3_authorized": False,
            "paper_claim_authorized": False,
            "synthetic": True,
        }

    monkeypatch.setattr(coordinator, "_run_lane", forbidden_lane)
    monkeypatch.setattr(coordinator, "_run_evaluator", evaluator)

    result = coordinator.execute(packet)

    assert lane_calls == []
    assert evaluator_calls == [True]
    assert result["status"] == "completed_exploratory_audit"
    assert result["evaluation"]["synthetic"] is True


def test_event_log_is_frozen_once_with_terminal_event_and_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _bind_paths(monkeypatch, tmp_path)
    paths["audit"].mkdir()
    events = coordinator.EventWriter(paths["events"])
    events.append({"event": "all_exports_completed", "completed_jobs": 18})

    binding = coordinator._freeze_event_log()

    assert binding["sha256"] == coordinator._sha256(paths["events"])
    assert binding["sidecar_path"] == str(paths["events_sidecar"])
    assert paths["events"].stat().st_mode & 0o222 == 0
    assert paths["events_sidecar"].stat().st_mode & 0o222 == 0
    with pytest.raises(RuntimeError, match="already exists"):
        coordinator._freeze_event_log()


def test_execute_fresh_state_dispatches_exactly_two_nine_job_lanes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _bind_paths(monkeypatch, tmp_path)
    packet = _synthetic_packet(paths["project"])
    _install_governance(monkeypatch, packet)
    monkeypatch.setattr(
        coordinator, "_now", lambda: "2026-07-17T00:00:00+00:00"
    )
    coordinator.register(packet)
    lane_calls: list[tuple[int, int, str]] = []

    def lane(
        gpu: int,
        jobs: list[coordinator.ExportJob],
        events: coordinator.EventWriter,
        registration: dict[str, Any],
    ) -> list[dict[str, Any]]:
        lane_calls.append(
            (
                gpu,
                len(jobs),
                registration["tier2s_preregistration_binding"]["sha256"],
            )
        )
        completed: list[dict[str, Any]] = []
        for job in jobs:
            manifest = Path(job.output_dir) / "manifest.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                f'{{"run_id":"{job.run_id}"}}\n', encoding="utf-8"
            )
            completed.append({"event": "job_completed", "run_id": job.run_id})
        return completed

    def evaluator(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "outer_target_access_authorized": False,
            "authorizes_outer_target_access": False,
            "source_tier3_authorized": False,
            "paper_claim_authorized": False,
        }

    monkeypatch.setattr(coordinator, "_run_lane", lane)
    monkeypatch.setattr(coordinator, "_run_evaluator", evaluator)

    result = coordinator.execute(packet)

    expected_prereg_sha = coordinator._sha256(paths["preregistration"])
    assert sorted(lane_calls) == [
        (0, 9, expected_prereg_sha),
        (1, 9, expected_prereg_sha),
    ]
    assert result["status"] == "completed_exploratory_audit"
    assert coordinator._load_json(paths["handoff"])["scheduler_event_log_sidecar"] == str(
        paths["events_sidecar"]
    )


def test_exclusive_lock_rejects_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _bind_paths(monkeypatch, tmp_path)
    paths["audit"].mkdir()
    target = paths["audit"] / "lock-target"
    target.write_bytes(b"")
    paths["lock"].symlink_to(target)

    with pytest.raises(RuntimeError, match="unsafe"):
        coordinator._exclusive_lock()
