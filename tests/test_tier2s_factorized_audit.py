from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.artifact_integrity import (
    file_sha256,
    ordered_ids_sha256,
    score_records_sha256,
)
from scripts import evaluate_tier2s_factorized_audit as audit


def _sample(
    *,
    image_id: str = "sample",
    base: np.ndarray | None = None,
    residual: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> audit.FactorizedSample:
    base_array = np.asarray(
        base if base is not None else [[0.0, 1.0], [2.0, 3.0]],
        dtype=np.float32,
    )
    residual_array = np.asarray(
        residual if residual is not None else [[0.0, 2.0], [-2.0, 4.0]],
        dtype=np.float32,
    )
    mask_array = np.asarray(
        mask if mask is not None else [[0, 0], [0, 1]], dtype=bool
    )
    return audit.FactorizedSample(
        image_id=image_id,
        dataset_name="NUDT-SIRST",
        subset_role="held_in",
        base=base_array,
        final=np.add(base_array, residual_array, dtype=np.float32),
        residual=residual_array,
        mask=mask_array,
    )


def test_compose_alpha_uses_only_frozen_grid() -> None:
    sample = _sample()
    np.testing.assert_array_equal(audit.compose_alpha(sample, 0.0), sample.base)
    np.testing.assert_array_equal(audit.compose_alpha(sample, 1.0), sample.final)
    np.testing.assert_array_equal(
        audit.compose_alpha(sample, 0.5),
        sample.base + np.float32(0.5) * sample.residual,
    )
    with pytest.raises(ValueError, match="frozen grid"):
        audit.compose_alpha(sample, 0.1)


def test_registered_alpha_rule_uses_mean_constraint_and_small_alpha_tie() -> None:
    scores = {
        0.0: {"strict": 0.50, "medium": 0.60, "loose": 0.70},
        0.25: {"strict": 0.497, "medium": 0.61, "loose": 0.71},
        # Higher mean but infeasible because strict regresses by >0.005.
        0.5: {"strict": 0.494, "medium": 0.80, "loose": 0.80},
        0.75: {"strict": 0.497, "medium": 0.61, "loose": 0.71},
        1.0: {"strict": 0.50, "medium": 0.59, "loose": 0.69},
    }
    result = audit.select_alpha_from_held_in_scores(scores)
    assert result["selected_alpha"] == 0.25
    rows = {row["alpha"]: row for row in result["candidates"]}
    assert rows[0.5]["feasible"] is False
    assert rows[0.25]["objective_mean_pd"] == rows[0.75]["objective_mean_pd"]
    assert result["held_out_metrics_used_for_selection"] is False


def test_empirical_tail_is_registered_negative_log10_survival() -> None:
    sample = _sample(
        base=np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        residual=np.zeros((2, 2), dtype=np.float32),
        mask=np.asarray([[0, 0], [0, 1]], dtype=bool),
    )
    transform = audit.fit_empirical_background_tail([sample], lambda item: item.base)
    values = np.asarray([[-1.0, 0.0, 0.5], [1.0, 2.0, 3.0]], dtype=np.float32)
    observed = transform.apply(values)
    expected = -np.log10(
        np.asarray([[1.0, 1.0, 0.75], [0.75, 0.5, 0.25]], dtype=np.float64)
    ).astype(np.float32)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-7)
    order = audit.audit_order_preservation(values, observed)
    assert order["fp32_order_inversions"] == 0
    assert transform.summary()["monotonicity"] == "nondecreasing_in_raw_logit"


def test_hash_partition_is_deterministic_and_disjoint() -> None:
    ids = [f"image-{index}" for index in range(64)]
    first = [audit.held_in_partition_bucket("NUDT-SIRST", value) for value in ids]
    second = [audit.held_in_partition_bucket("NUDT-SIRST", value) for value in ids]
    assert first == second
    assert set(first) == {0, 1, 2, 3}
    assert not ({index for index, bucket in enumerate(first) if bucket in {0, 1}} &
                {index for index, bucket in enumerate(first) if bucket == 2})


def test_landed_protocol_matches_evaluator_contract() -> None:
    protocol, partition, old_handoff, old_decision = audit._validate_protocol(
        audit.PROTOCOL_PATH
    )
    assert protocol["protocol_id"] == audit.PROTOCOL_ID
    assert partition["tail_fit_buckets"] == [0, 1]
    assert old_handoff == audit.OLD_HANDOFF_PATH
    assert file_sha256(old_decision) == protocol["immutable_parent_evidence"]["tier2r_decision"]["sha256"]


def test_raw_final_parent_points_and_nine_criteria_replay_exactly() -> None:
    decision_path = (
        audit.PROJECT_ROOT
        / "artifacts/aaai27/audit/component_rescue/tier2r_c_v1_impl_erratum1"
        / "exact_gate/COMPONENT_RESCUE_DECISION.json"
    )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    evidence = json.loads(
        (decision_path.parent / "evidence_manifest.json").read_text(encoding="utf-8")
    )
    points: dict[str, dict[int, dict]] = {role: {} for role in audit.ROLES}
    for role in audit.ROLES:
        for seed in audit.SEEDS:
            binding = evidence["artifacts"][f"operating_points/{role}_seed{seed}"]
            payload = json.loads(Path(binding["path"]).read_text(encoding="utf-8"))
            points[role][seed] = payload["gate_points"]
    raw_result = {
        "points_by_role": points,
        "nine_criterion_replay": audit.evaluate_gate_level(
            points,
            candidate="c",
            baseline="control",
            level_name="different_name_is_allowed",
        ),
    }
    replay = audit.verify_raw_final_parent_replay(
        raw_result,
        old_decision=decision,
        old_decision_path=decision_path,
    )
    assert replay["gate_points_exactly_reproduced"] is True
    assert replay["paired_deltas_and_pass_flags_exactly_reproduced"] is True
    assert replay["num_criteria"] == 9


@pytest.mark.parametrize(
    ("passing", "expected"),
    [
        (set(), "contrast_route_unsupported"),
        ({"source_selected_alpha"}, "residual_amplitude_dominant"),
        ({"source_tail_calibrated_final"}, "cross_fold_scale_dominant"),
        ({"source_tail_calibrated_selected_alpha"}, "joint_residual_and_scale"),
        (
            {"source_selected_alpha", "source_tail_calibrated_final"},
            "report_all_supported_factors_without_posthoc_selection",
        ),
    ],
)
def test_factor_diagnosis_is_preregistered_and_never_authorizes(
    passing: set[str], expected: str
) -> None:
    routes = {
        route: {"nine_criterion_replay": {"passed": route in passing}}
        for route in audit.ROUTES
    }
    result = audit.classify_factor_diagnosis(routes)
    assert result["classification"] == expected
    assert result["posthoc_route_selection_performed"] is False
    assert result["authorizes_source_tier3_design"] is False
    assert result["authorizes_outer_target_access"] is False
    assert result["outer_target_access_authorized"] is False


def _stream_hash(factor: str, image_id: str, value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"rc-irstd-tier2s-factorized-stream-sha256-v1\0")
    digest.update(factor.encode("ascii") + b"\0")
    identity = image_id.encode("utf-8")
    digest.update(len(identity).to_bytes(8, "big"))
    digest.update(identity)
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()

def _two_lane_queue() -> dict:
    return {
        "schema_version": audit.QUEUE_SCHEMA,
        "protocol_id": audit.PROTOCOL_ID,
        "scheduler": audit.QUEUE_SCHEDULER,
        "wait_for_idle_gpu": False,
        "allow_gpu_fallback": False,
        "jobs": [
            {
                "run_id": f"gpu{gpu}-job{queue_index}",
                "physical_gpu": gpu,
                "queue_index": queue_index,
            }
            for gpu in audit.QUEUE_PHYSICAL_GPUS
            for queue_index in range(audit.QUEUE_JOBS_PER_LANE)
        ],
    }


def test_two_lane_queue_contract_accepts_exact_nine_plus_nine() -> None:
    jobs = audit._validate_fixed_two_lane_queue(_two_lane_queue())
    assert len(jobs) == 18
    assert {job["physical_gpu"] for job in jobs} == {0, 1}


@pytest.mark.parametrize(
    "drift",
    ["schema", "scheduler", "job_count", "gpu", "lane_size", "queue_index"],
)
def test_two_lane_queue_contract_rejects_every_structural_drift(
    drift: str,
) -> None:
    queue = _two_lane_queue()
    if drift == "schema":
        queue["schema_version"] = "old-four-lane-schema"
    elif drift == "scheduler":
        queue["scheduler"] = "four_fixed_independent_fifo_lanes"
    elif drift == "job_count":
        queue["jobs"].pop()
    elif drift == "gpu":
        queue["jobs"][0]["physical_gpu"] = 2
    elif drift == "lane_size":
        queue["jobs"][-1]["physical_gpu"] = 0
    elif drift == "queue_index":
        queue["jobs"][0]["queue_index"] = 1
    with pytest.raises(RuntimeError, match="queue"):
        audit._validate_fixed_two_lane_queue(queue)


def test_frozen_json_writer_is_idempotent_and_never_replaces(
    tmp_path: Path,
) -> None:
    path = tmp_path / "exact_evidence.json"
    payload = {"schema_version": "test-v1", "value": [1, 2, 3]}
    digest = audit._write_once_frozen_json(path, payload)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    original = path.read_bytes()
    assert digest == file_sha256(path)
    assert path.stat().st_mode & 0o777 == 0o444
    assert sidecar.stat().st_mode & 0o777 == 0o444
    assert sidecar.read_text(encoding="ascii") == (
        f"{digest}  {path.name}\n"
    )
    assert audit._write_once_frozen_json(path, payload) == digest
    assert path.read_bytes() == original

    with pytest.raises(RuntimeError, match="no-replace artifact drift"):
        audit._write_once_frozen_json(
            path, {"schema_version": "test-v1", "value": [1, 2, 4]}
        )
    assert path.read_bytes() == original



def test_run_audit_rechecks_strict_governance_before_any_input_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registration_sha = "7" * 64
    calls: list[str | None] = []

    def stop_after_governance(*, expected_registration_sha256=None):
        calls.append(expected_registration_sha256)
        raise RuntimeError("strict-governance-verifier-called")

    monkeypatch.setattr(
        audit.governance_registrar,
        "require_frozen_tier2s_governance",
        stop_after_governance,
    )
    with pytest.raises(RuntimeError, match="strict-governance-verifier-called"):
        audit.run_audit(
            protocol_path=tmp_path / "protocol.json",
            handoff_path=tmp_path / "handoff.json",
            output_dir=tmp_path / "result",
            governance_registration_sha256=registration_sha,
        )
    assert calls == [registration_sha]


@pytest.mark.parametrize("drift_container", ["preregistration", "queue"])
def test_handoff_prereg_queue_require_the_same_governance_binding(
    monkeypatch: pytest.MonkeyPatch, drift_container: str
) -> None:
    registration_sha = "7" * 64
    governance = {"registration": {"sha256": registration_sha}}
    preregistration_binding = {
        "schema_version": "rc-irstd-aaai27-tier2s-preregistration-binding-v1",
        "sha256": "8" * 64,
        "governance_registration_sha256": registration_sha,
    }
    handoff = {
        "schema_version": audit.HANDOFF_SCHEMA,
        "source_only": True,
        "research_mode": "exploratory_source_only",
        "outer_target_access_authorized": False,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "source_tier3_authorized": False,
        "paper_claim_authorized": False,
        "governance_binding": governance,
        "tier2s_preregistration_binding": preregistration_binding,
        "preregistration_sha256": preregistration_binding["sha256"],
        "queue_manifest_sha256": "9" * 64,
    }
    preregistration = {"governance_binding": governance}
    queue = {
        "governance_binding": governance,
        "tier2s_preregistration_binding": preregistration_binding,
    }
    if drift_container == "preregistration":
        preregistration.pop("governance_binding")
    else:
        queue["governance_binding"] = {"registration": {"sha256": "0" * 64}}

    def load(path: Path):
        if path.name == "EXPORT_HANDOFF.json":
            return handoff
        if path.name == "PREREGISTRATION.json":
            return preregistration
        if path.name == "QUEUE_MANIFEST.json":
            return queue
        raise AssertionError(path)

    monkeypatch.setattr(audit, "_load_json", load)
    monkeypatch.setattr(audit, "_verify_frozen_sidecar", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        audit.tier2s_exporter,
        "require_frozen_tier2s_consumer_bindings",
        lambda **kwargs: (governance, preregistration_binding),
    )
    with pytest.raises(RuntimeError, match=drift_container):
        audit._load_export_handoff(
            audit.tier2s_exporter.DEFAULT_PREREGISTRATION.parent
            / "EXPORT_HANDOFF.json",
            {},
            expected_governance_binding=governance,
            expected_governance_registration_sha256=registration_sha,
        )


def test_factorized_loader_binds_npz_manifest_and_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "export"
    records_root = root / "records"
    records_root.mkdir(parents=True)
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    formal_config = tmp_path / "formal.yaml"
    formal_config.write_text("model: control\n", encoding="utf-8")
    image_id = "one"
    base = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    final = base.copy()
    residual = np.zeros_like(base)
    mask = np.asarray([[0, 0], [0, 1]], dtype=np.uint8)
    record_path = records_root / "one.npz"
    np.savez_compressed(
        record_path,
        base_raw_logit_float32=base,
        final_raw_logit_float32=final,
        residual_raw_logit_float32=residual,
        mask=mask,
        image_id=np.asarray(image_id),
        dataset_name=np.asarray("IRSTD-1K"),
        subset_role=np.asarray("held_in"),
        split_role=np.asarray("train"),
        original_hw=np.asarray(base.shape, dtype=np.int64),
        input_hw=np.asarray(base.shape, dtype=np.int64),
        valid_hw=np.asarray(base.shape, dtype=np.int64),
        padding_ltrb=np.asarray([0, 0, 0, 0], dtype=np.int64),
        spatial_mode=np.asarray("native"),
        labels_loaded=np.asarray(True),
        inference_autocast_enabled=np.asarray(False),
        model_output_bitwise_equal=np.asarray(True),
    )
    record = {
        "image_id": image_id,
        "file": "records/one.npz",
        "sha256": file_sha256(record_path),
        "shape": [2, 2],
        "replay_max_abs_error": 0.0,
        "model_output_bitwise_equal": True,
        "residual_exact_zero": True,
    }
    old_run = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "formal_config_sha256": file_sha256(formal_config),
        "seed": 43,
        "role": "control",
        "fold": "heldout_nudt",
        "training_source": "IRSTD-1K",
        "held_out_source": "NUDT-SIRST",
    }
    manifest = {
        "schema_version": audit.EXPORT_SCHEMA,
        "protocol_id": audit.PROTOCOL_ID,
        "protocol_binding": {
            "path": str(audit.PROTOCOL_PATH),
            "sha256": file_sha256(audit.PROTOCOL_PATH),
            "protocol_id": audit.PROTOCOL_ID,
        },
        "governance_binding": dict(
            audit.tier2s_exporter.UNIT_TEST_UNBOUND_GOVERNANCE_BINDING
        ),
        "tier2s_preregistration_binding": dict(
            audit.tier2s_exporter.UNIT_TEST_UNBOUND_PREREGISTRATION_BINDING
        ),
        "diagnostic_only": True,
        "authorizes_go": False,
        "authorizes_source_tier3": False,
        "source_only": True,
        "outer_target_images_loaded": False,
        "outer_target_masks_loaded": False,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "outer_target_access_authorized": False,
        "architecture_version": "rc-mshnet-v2-component-role-split",
        "role": "control",
        "fold": "heldout_nudt",
        "subset_role": "held_in",
        "checkpoint_binding": {
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
            "formal_config_path": str(formal_config.resolve()),
            "formal_config_sha256": file_sha256(formal_config),
            "seed": 43,
            "role": "control",
            "fold": "heldout_nudt",
            "training_source": "IRSTD-1K",
            "held_out_source": "NUDT-SIRST",
            "checkpoint_selection": "fixed_last",
            "epoch": 79,
            "architecture_version": "rc-mshnet-v2-component-role-split",
        },
        "dataset_binding": {
            "dataset_name": "IRSTD-1K",
            "subset_role": "held_in",
            "spatial_mode": "native",
        },
        "all_model_outputs_bitwise_equal": True,
        "all_residual_exact_zero": True,
        "replay_max_abs_error": 0.0,
        "records": [record],
        "records_sha256": score_records_sha256([record]),
        "ordered_image_ids_sha256": ordered_ids_sha256([image_id]),
        "raw_logit_stream_sha256": {
            "base": _stream_hash("base", image_id, base),
            "final": _stream_hash("final", image_id, final),
            "residual": _stream_hash("residual", image_id, residual),
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    digest = file_sha256(manifest_path)
    (root / "manifest.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii"
    )
    samples, evidence = audit.load_factorized_directory(
        root,
        expected_manifest_sha256=digest,
        old_run=old_run,
        expected_subset_role="held_in",
    )
    assert len(samples) == 1
    assert evidence["checkpoint_binding_verified"] is True
    assert evidence["governance_binding_verified"] is True
    assert evidence["tier2s_preregistration_binding_verified"] is True
    np.testing.assert_array_equal(samples[0].residual, 0.0)

    manifest.pop("governance_binding")
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    stale_digest = file_sha256(manifest_path)
    (root / "manifest.sha256").write_text(
        f"{stale_digest}  manifest.json\n", encoding="ascii"
    )
    with pytest.raises(RuntimeError, match="governance binding drift"):
        audit.load_factorized_directory(
            root,
            expected_manifest_sha256=stale_digest,
            old_run=old_run,
            expected_subset_role="held_in",
        )


def _write_scheduler_log(
    path: Path,
    jobs: list[dict[str, object]],
    *,
    drift_completion: bool = False,
    payloads: list[dict[str, object]] | None = None,
) -> None:
    previous = "0" * 64
    records: list[dict[str, object]] = []

    def append(payload: dict[str, object]) -> None:
        nonlocal previous
        event = {
            "schema_version": "rc-irstd-aaai27-tier2s-scheduler-event-v1",
            "time": "2026-07-17T00:00:00+00:00",
            "previous_event_sha256": previous,
            **payload,
        }
        digest = hashlib.sha256(audit._canonical_json_bytes(event)).hexdigest()
        records.append({**event, "event_sha256": digest})
        previous = digest

    if payloads is None:
        by_lane = {
            (int(job["physical_gpu"]), int(job["queue_index"])): job
            for job in jobs
        }
        payloads = []
        for queue_index in range(audit.QUEUE_JOBS_PER_LANE):
            lane_jobs = [
                by_lane[(gpu, queue_index)] for gpu in audit.QUEUE_PHYSICAL_GPUS
            ]
            # Deliberately interleave the lanes in opposite completion order.
            # This proves cross-lane ordering is unconstrained while each lane
            # remains a strict start(q) -> completion(q) FIFO.
            for job in lane_jobs:
                payloads.append(
                    {
                        "event": "job_started",
                        "run_id": job["run_id"],
                        "physical_gpu": job["physical_gpu"],
                        "queue_index": job["queue_index"],
                    }
                )
            for job in reversed(lane_jobs):
                payloads.append(
                    {
                        "event": "job_completed",
                        "run_id": job["run_id"],
                        "physical_gpu": job["physical_gpu"],
                        "queue_index": job["queue_index"],
                    }
                )
        payloads.append({"event": "all_exports_completed", "completed_jobs": 18})
    payloads = [dict(payload) for payload in payloads]
    if drift_completion:
        first_completion = next(
            index
            for index, payload in enumerate(payloads)
            if payload.get("event") == "job_completed"
        )
        payloads[first_completion]["physical_gpu"] = (
            1 - int(payloads[first_completion]["physical_gpu"])
        )
    for payload in payloads:
        append(payload)
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    digest = file_sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    path.chmod(0o444)
    sidecar.chmod(0o444)


def test_scheduler_event_log_requires_exact_two_lane_completion_chain(
    tmp_path: Path,
) -> None:
    jobs = [
        {"run_id": f"gpu{gpu}_q{queue}", "physical_gpu": gpu, "queue_index": queue}
        for gpu in (0, 1)
        for queue in range(9)
    ]
    valid = tmp_path / "valid.jsonl"
    _write_scheduler_log(valid, jobs)

    binding = audit._verify_scheduler_event_log(
        valid, expected_sha256=file_sha256(valid), expected_jobs=jobs
    )

    assert binding["num_events"] == 37
    assert binding["sidecar_path"] == str(valid.with_suffix(".jsonl.sha256"))

    drifted = tmp_path / "drifted.jsonl"
    _write_scheduler_log(drifted, jobs, drift_completion=True)
    with pytest.raises(RuntimeError, match="completion lane binding drift"):
        audit._verify_scheduler_event_log(
            drifted,
            expected_sha256=file_sha256(drifted),
            expected_jobs=jobs,
        )


def _valid_scheduler_payloads(
    jobs: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_lane = {
        (int(job["physical_gpu"]), int(job["queue_index"])): job for job in jobs
    }
    payloads: list[dict[str, object]] = []
    for queue_index in range(audit.QUEUE_JOBS_PER_LANE):
        for gpu in audit.QUEUE_PHYSICAL_GPUS:
            job = by_lane[(gpu, queue_index)]
            payloads.extend(
                [
                    {
                        "event": "job_started",
                        "run_id": job["run_id"],
                        "physical_gpu": gpu,
                        "queue_index": queue_index,
                    },
                    {
                        "event": "job_completed",
                        "run_id": job["run_id"],
                        "physical_gpu": gpu,
                        "queue_index": queue_index,
                    },
                ]
            )
    payloads.append({"event": "all_exports_completed", "completed_jobs": 18})
    return payloads


def _scheduler_jobs() -> list[dict[str, object]]:
    return [
        {"run_id": f"gpu{gpu}_q{queue}", "physical_gpu": gpu, "queue_index": queue}
        for gpu in audit.QUEUE_PHYSICAL_GPUS
        for queue in range(audit.QUEUE_JOBS_PER_LANE)
    ]


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("completion_before_start", "completion occurred before matching start"),
        ("queue_index_out_of_order", "FIFO start order drift"),
        ("next_start_before_completion", "started next job before prior completion"),
        ("terminal_not_last", "terminal event is not last"),
    ],
)
def test_scheduler_event_log_rejects_per_lane_fifo_drift(
    tmp_path: Path, drift: str, message: str
) -> None:
    jobs = _scheduler_jobs()
    payloads = _valid_scheduler_payloads(jobs)
    if drift == "completion_before_start":
        completion = payloads.pop(1)
        payloads.insert(0, completion)
    elif drift == "queue_index_out_of_order":
        q1_start = payloads.pop(4)
        payloads.insert(0, q1_start)
    elif drift == "next_start_before_completion":
        q1_start = payloads.pop(4)
        payloads.insert(1, q1_start)
    elif drift == "terminal_not_last":
        payloads[-1], payloads[-2] = payloads[-2], payloads[-1]
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(drift)

    path = tmp_path / f"{drift}.jsonl"
    _write_scheduler_log(path, jobs, payloads=payloads)
    with pytest.raises(RuntimeError, match=message):
        audit._verify_scheduler_event_log(
            path,
            expected_sha256=file_sha256(path),
            expected_jobs=jobs,
        )
