from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import pytest

from evaluation.artifact_integrity import file_sha256
from scripts import coordinate_tier2s_factorized_audit_v2_gpu23 as coordinator
from scripts import evaluate_tier2s_factorized_audit_v2_gpu23 as evaluator
from scripts import launch_tier2s_factorized_isolated_v2_gpu23 as launcher
from scripts import register_tier2s_gpu23_amendment_v2 as amendment


ROOT = Path(__file__).resolve().parents[1]
PARENT_PROTOCOL_SHA256 = (
    "9f0ec09e0289409b184e6dec1fe89b2a51ad5fc483f4f0ca5d27195173cdf129"
)


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


def _synthetic_parent_handoff(root: Path) -> dict[str, Any]:
    parent = root / "parent"
    parent.mkdir(parents=True)
    runs: dict[str, dict[str, Any]] = {}
    for seed in coordinator.SEEDS:
        for role in coordinator.ROLES:
            for fold in coordinator.FOLDS:
                run_id = f"seed{seed}_{role}_{fold}"
                checkpoint = parent / f"{run_id}.pt"
                score_manifest = parent / f"{run_id}.scores.json"
                checkpoint.write_bytes(f"checkpoint:{run_id}\n".encode("ascii"))
                score_manifest.write_text(
                    json.dumps({"run_id": run_id}) + "\n", encoding="utf-8"
                )
                runs[run_id] = {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "score_manifest": str(score_manifest),
                    "score_manifest_sha256": file_sha256(score_manifest),
                }
    return {"runs": runs}


def _queue_jobs() -> list[dict[str, Any]]:
    return [
        {
            "run_id": f"gpu{gpu}-q{queue_index}",
            "physical_gpu": gpu,
            "container_gpu_ordinal": evaluator.CONTAINER_ORDINAL_BY_PHYSICAL[gpu],
            "queue_index": queue_index,
        }
        for gpu in evaluator.QUEUE_PHYSICAL_GPUS
        for queue_index in range(evaluator.QUEUE_JOBS_PER_LANE)
    ]


def _queue() -> dict[str, Any]:
    return {
        "schema_version": evaluator.QUEUE_SCHEMA,
        "protocol_id": evaluator.PROTOCOL_ID,
        "scheduler": evaluator.QUEUE_SCHEDULER,
        "wait_for_idle_gpu": False,
        "allow_gpu_fallback": False,
        "jobs": _queue_jobs(),
    }


def _freeze_event_log(path: Path, records: list[Mapping[str, Any]]) -> str:
    previous = "0" * 64
    lines: list[str] = []
    for raw in records:
        item = {
            "schema_version": evaluator.EVENT_SCHEMA,
            "time": "2026-07-18T00:00:00+00:00",
            "previous_event_sha256": previous,
            **dict(raw),
        }
        digest = hashlib.sha256(evaluator._canonical_json_bytes(item)).hexdigest()
        lines.append(
            json.dumps(
                {**item, "event_sha256": digest},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        previous = digest
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    file_digest = file_sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{file_digest}  {path.name}\n", encoding="ascii")
    path.chmod(0o444)
    sidecar.chmod(0o444)
    return file_digest


def _scheduler_records(*, ordinal_drift: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for queue_index in range(9):
        for gpu in (2, 3):
            ordinal = coordinator.CONTAINER_ORDINAL_BY_PHYSICAL[gpu]
            if ordinal_drift and gpu == 2 and queue_index == 0:
                ordinal = 1
            run_id = f"gpu{gpu}-q{queue_index}"
            common = {
                "run_id": run_id,
                "physical_gpu": gpu,
                "container_gpu_ordinal": ordinal,
                "queue_index": queue_index,
            }
            records.append({"event": "job_started", "logical_device": "cuda:0", **common})
            records.append({"event": "job_completed", "returncode": 0, **common})
    records.append({"event": "all_exports_completed", "completed_jobs": 18})
    return records


def _artifact(path: Path, digit: str) -> dict[str, str]:
    return {
        "path": path.as_posix(),
        "sha256": digit * 64,
        "sidecar_path": f"{path.as_posix()}.sha256",
        "sidecar_sha256": digit * 64,
    }


def _governance_binding() -> dict[str, Any]:
    return {
        "schema_version": amendment.BINDING_SCHEMA,
        "registration": _artifact(launcher.GOVERNANCE_REGISTRATION_RELATIVE, "1"),
        "parent_governance_registration": _artifact(
            launcher.PARENT_GOVERNANCE_REGISTRATION_RELATIVE, "2"
        ),
        "contract": _artifact(launcher.GOVERNANCE_CONTRACT_RELATIVE, "3"),
        "fresh_seed_ledger": _artifact(launcher.FRESH_SEED_LEDGER_RELATIVE, "4"),
        "fresh_seed_local_scan": _artifact(
            launcher.FRESH_SEED_LOCAL_SCAN_RELATIVE, "5"
        ),
        "code_sha256_canonical_sha256": "6" * 64,
        "physical_gpus": [2, 3],
        "container_logical_ordinals": [0, 1],
        "physical_to_container_ordinal": {"2": 0, "3": 1},
        "gpu_fallback_allowed": False,
        "superseded_execution_plan": {
            "protocol_path": "configs/tier2s_factorized_causal_audit_v1.json",
            "protocol_sha256": PARENT_PROTOCOL_SHA256,
            "output_namespace_absent": True,
            "audit_namespace_absent": True,
            "historical_evidence_superseded": False,
            "unexecuted_hardware_plan_replaced_only": True,
        },
        "tier2s_source_only_diagnostic_authorized": True,
        "formal_v3_model_training_authorized": False,
        "source_gate_a_authorized": False,
        "riskcurve_authorized": False,
        "paper_claim_authorized": False,
        "outer_target_access_authorized": False,
    }


def test_v1_protocol_and_absent_namespaces_remain_immutable() -> None:
    assert file_sha256(amendment.PARENT_PROTOCOL) == PARENT_PROTOCOL_SHA256
    assert not amendment.PARENT_OUTPUT_ROOT.exists()
    assert not amendment.PARENT_AUDIT_ROOT.exists()


def test_landed_v2_protocol_is_source_only_gpu23_and_non_novelty_diagnostic() -> None:
    protocol = amendment._load_json(amendment.PROTOCOL)
    assert protocol["schema_version"] == coordinator.PROTOCOL_SCHEMA
    assert protocol["protocol_id"] == coordinator.PROTOCOL_ID
    assert protocol["execution"]["physical_gpus"] == [2, 3]
    assert protocol["execution"]["container_logical_ordinals"] == {"2": 0, "3": 1}
    assert protocol["execution"]["allow_gpu_fallback"] is False
    assert protocol["execution"]["export_jobs"] == 18
    limits = protocol["scientific_limits"]
    assert limits["result_use"] == "failure_attribution_only"
    assert limits["may_authorize_v3_implementation"] is False
    assert limits["scaling_alpha_tail_coordinate_or_calibration_may_be_claimed_as_innovation"] is False
    assert limits["outer_target_access_authorized"] is False


def test_v2_schedule_is_exact_physical_9_plus_9_with_container_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(coordinator, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(coordinator, "OUTPUT_ROOT", tmp_path / "outputs")
    jobs = coordinator._build_jobs(
        _synthetic_protocol(), _synthetic_parent_handoff(tmp_path)
    )
    assert len(jobs) == 18
    assert Counter(job.physical_gpu for job in jobs) == Counter({2: 9, 3: 9})
    assert all(
        job.container_gpu_ordinal
        == coordinator.CONTAINER_ORDINAL_BY_PHYSICAL[job.physical_gpu]
        for job in jobs
    )
    assert {job.physical_gpu for job in jobs}.isdisjoint({0, 1})
    for gpu in coordinator.PHYSICAL_GPUS:
        assert [job.queue_index for job in jobs if job.physical_gpu == gpu] == list(range(9))


def test_child_process_uses_container_ordinal_not_host_index() -> None:
    base = {
        "run_id": "run",
        "checkpoint_run_id": "checkpoint",
        "seed": 43,
        "role": "c",
        "fold": "heldout_nudt",
        "scope": "held_out",
        "dataset_name": "NUDT-SIRST",
        "dataset_root": "/source/NUDT-SIRST",
        "checkpoint": "/checkpoint.pt",
        "checkpoint_sha256": "1" * 64,
        "parent_heldout_score_manifest": "/scores.json",
        "parent_heldout_score_manifest_sha256": "2" * 64,
        "queue_index": 0,
        "output_dir": "/output",
    }
    for physical, ordinal in ((2, 0), (3, 1)):
        job = coordinator.ExportJob(
            **base,
            physical_gpu=physical,
            container_gpu_ordinal=ordinal,
        )
        assert coordinator._child_env(job)["CUDA_VISIBLE_DEVICES"] == str(ordinal)
    drift = coordinator.ExportJob(
        **base, physical_gpu=2, container_gpu_ordinal=1
    )
    with pytest.raises(RuntimeError, match="mapping drift"):
        coordinator._child_env(drift)


def test_evaluator_queue_requires_exact_physical_to_container_mapping() -> None:
    jobs = evaluator._validate_fixed_two_lane_queue(_queue())
    assert len(jobs) == 18
    assert {job["physical_gpu"] for job in jobs} == {2, 3}
    drift = _queue()
    drift["jobs"][0]["container_gpu_ordinal"] = 1
    with pytest.raises(RuntimeError, match="queue"):
        evaluator._validate_fixed_two_lane_queue(drift)


def test_scheduler_fifo_event_log_rejects_container_ordinal_drift(tmp_path: Path) -> None:
    valid = tmp_path / "valid.jsonl"
    valid_sha = _freeze_event_log(valid, _scheduler_records())
    binding = evaluator._verify_scheduler_event_log(
        valid, expected_sha256=valid_sha, expected_jobs=_queue_jobs()
    )
    assert binding["num_events"] == 37

    drift = tmp_path / "drift.jsonl"
    drift_sha = _freeze_event_log(drift, _scheduler_records(ordinal_drift=True))
    with pytest.raises(RuntimeError, match="lane binding drift"):
        evaluator._verify_scheduler_event_log(
            drift, expected_sha256=drift_sha, expected_jobs=_queue_jobs()
        )


def test_launcher_dry_run_exposes_only_physical_gpu2_gpu3() -> None:
    payload = launcher.dry_run_payload()
    assert payload["filesystem_mutated"] is False
    assert payload["coordinator_physical_gpus"] == [2, 3]
    assert payload["docker_gpu_request"] == '"device=2,3"'
    assert payload["nvidia_visible_devices"] == "2,3"
    for key in ("verify_command", "register_command", "formal_command_template"):
        command = payload[key]
        assert command[command.index("--gpus") + 1] == '"device=2,3"'
        assert "NVIDIA_VISIBLE_DEVICES=2,3" in command
        assert not any(value.startswith("CUDA_VISIBLE_DEVICES=") for value in command)
    spec = payload["formal_container_spec"]
    assert spec["gpu_exposure"] == "device=2,3"
    assert spec["gpu_device_ids"] == ["2", "3"]


def test_launcher_governance_and_coordinator_contract_bind_gpu23_mapping() -> None:
    governance = _governance_binding()
    assert launcher._require_governance_binding(
        {"governance_binding": governance}
    ) == governance
    schedule = _queue_jobs()
    payload = {
        "schema_version": launcher.COORDINATOR_SCHEMA,
        "verified": True,
        "protocol_id": launcher.PROTOCOL_ID,
        "research_mode": launcher.RESEARCH_MODE,
        "source_only": True,
        "outer_target_access_authorized": False,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "source_tier3_authorized": False,
        "paper_claim_authorized": False,
        "governance_binding": governance,
        "schedule": schedule,
        "lane_lengths": {"2": 9, "3": 9},
        "physical_to_container_ordinal": {"2": 0, "3": 1},
    }
    assert len(launcher._coordinator_contract(payload)) == 18
    payload["schedule"] = [dict(value) for value in schedule]
    payload["schedule"][0]["physical_gpu"] = 0
    with pytest.raises(RuntimeError, match="coordinator contract drift"):
        launcher._coordinator_contract(payload)


def test_amendment_verify_only_is_unregistered_and_never_authorizes_v3() -> None:
    status = amendment.verify_registration()
    assert status["verified"] is True
    assert status["registered"] is False
    candidate = status["candidate"]
    assert candidate["execution_scope"]["physical_gpus"] == [2, 3]
    assert candidate["execution_scope"]["physical_to_container_ordinal"] == {"2": 0, "3": 1}
    assert candidate["authorization"]["tier2s_v2_source_only_diagnostic_authorized"] is True
    assert candidate["authorization"]["formal_v3_model_training_authorized"] is False
    assert candidate["authorization"]["paper_claim_authorized"] is False
    assert candidate["authorization"]["outer_target_access_authorized"] is False


def test_v2_cli_entrypoints_resolve_project_modules_from_outside_repo(
    tmp_path: Path,
) -> None:
    commands = (
        [
            sys.executable,
            str(ROOT / "scripts/register_tier2s_gpu23_amendment_v2.py"),
            "--verify-only",
        ],
        [
            sys.executable,
            str(ROOT / "scripts/evaluate_tier2s_factorized_audit_v2_gpu23.py"),
            "--help",
        ],
    )
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=tmp_path,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


def test_amendment_write_once_refuses_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "amendment.json"
    amendment._write_once(target, b"{}\n")
    with pytest.raises(RuntimeError, match="already exists"):
        amendment._write_once(target, b"{}\n")
    assert target.read_bytes() == b"{}\n"
    assert target.stat().st_mode & 0o222 == 0
