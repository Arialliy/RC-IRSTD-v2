from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scripts.run_phase3_raw_logit_rescue_v1 as runner
from evaluation.artifact_integrity import file_sha256
from evaluation.raw_logit_oracle import RawLogitSample


REGISTERED_AT = "2026-07-16T11:30:00+08:00"


def _write_json(path: Path, payload: dict[str, Any], *, readonly: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if readonly:
        path.chmod(0o444)


def _freeze_historical_json(path: Path, payload: dict[str, Any]) -> None:
    _write_json(path, payload, readonly=True)
    sidecar = path.with_suffix(".sha256")
    sidecar.write_text(f"{file_sha256(path)}  {path.name}\n", encoding="ascii")
    sidecar.chmod(0o444)


def _point(value: float) -> dict[str, Any]:
    return {
        "found": True,
        "pooled_pd": value,
        "worst_pd": value,
        "macro_pd": value,
        "domain_pd": {"nudt": value, "irstd1k": value},
    }


def _fake_artifacts() -> dict[str, Any]:
    operating: dict[str, Any] = {}
    for name, _, _ in runner.BUDGETS:
        operating[name] = {
            "control": {"selection": {"budget": name}, "gate_point": _point(0.60)},
            "full": {"selection": {"budget": name}, "gate_point": _point(0.62)},
        }
    return {
        "exact_source_curves": {
            "control": {"schema_version": "synthetic-exact-v1", "states": [1, 2]},
            "full": {"schema_version": "synthetic-exact-v1", "states": [1, 2]},
        },
        "operating_points": operating,
        "dense_tail_grid_gap": {"diagnostic_only": True, "grid_size": 1024},
        "cross_domain_calibration_gap": {"diagnostic_only": True},
        "false_alarm_concentration": {"diagnostic_only": True},
    }


def _fake_api() -> runner.AlgorithmAPI:
    def placeholder(*_args, **_kwargs):
        raise AssertionError("unit tests mock the aggregate compute adapter")

    return runner.AlgorithmAPI(
        enumerate_exact_shared_states=placeholder,
        select_exact_shared_source_operating_points=placeholder,
        evaluate_domains_at_threshold=placeholder,
        select_domain_oracles=placeholder,
        build_cross_domain_calibration_gap=placeholder,
        deterministic_dense_state_indices=placeholder,
        select_dense_operating_points=placeholder,
        build_realized_fa_sensitivity=placeholder,
        summarize_false_alarm_concentration=placeholder,
    )


def _prepare_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    project = tmp_path / "project"
    monkeypatch.setattr(runner, "PROJECT_ROOT", project)
    historical = project / "artifacts/aaai27/audit/phase3_source_lodo_gate"
    _write_json(
        project / "artifacts/aaai27/audit/phase2_status.json",
        {"decision": "GO_PHASE3_SOURCE_ONLY"},
    )
    (project / "outputs/phase_state").mkdir(parents=True)
    (project / "outputs/phase_state/HOLD_PHASE3_TARGET_LABEL_ACCESS").write_text(
        "HOLD\n", encoding="utf-8"
    )
    (project / "RC-MSHNet_raw-logit_rescue_gate.md").write_text(
        "# disclosed post-hoc raw-logit rescue\n", encoding="utf-8"
    )

    run_bindings: dict[str, Any] = {}
    loader_records: dict[str, Any] = {}
    for index, spec in enumerate(runner.SCORE_SPECS, start=1):
        score_dir = spec.score_dir(project)
        score_dir.mkdir(parents=True)
        run_dir = score_dir.parent
        checkpoint = run_dir / "last.pt"
        checkpoint.write_bytes(f"checkpoint-{spec.run_id}".encode("utf-8"))
        checkpoint_sha = file_sha256(checkpoint)
        identity = {"run_id": spec.run_id, "checkpoint_sha256": checkpoint_sha}
        export_identity = {
            "run_id": spec.run_id,
            "checkpoint_sha256": checkpoint_sha,
        }
        _write_json(run_dir / "PHASE3_IDENTITY.json", identity)
        _write_json(run_dir / "EXPORT_IDENTITY.json", export_identity)
        manifest = {
            "schema_version": 3,
            "labels_loaded": True,
            "target_dataset": spec.target_dataset,
            "source_datasets": [spec.source_dataset],
        }
        _write_json(score_dir / "manifest.json", manifest)
        run_bindings[spec.run_id] = {
            "identity_sha256": file_sha256(run_dir / "PHASE3_IDENTITY.json"),
            "checkpoint_sha256": checkpoint_sha,
            "export_identity_sha256": file_sha256(run_dir / "EXPORT_IDENTITY.json"),
        }
        sample = RawLogitSample(
            image_id=f"sample-{index}",
            logits=np.asarray([[float(index)]], dtype=np.float32),
            probability=np.asarray([[0.5]], dtype=np.float32),
            mask=np.asarray([[1]], dtype=bool),
        )
        ids_sha = hashlib.sha256(f"ids-{spec.fold}".encode()).hexdigest()
        split_sha = hashlib.sha256(f"split-{spec.fold}".encode()).hexdigest()
        loader_records[str(score_dir.resolve())] = {
            "samples": [sample],
            "manifest": manifest,
            "integrity": {
                "manifest_sha256": file_sha256(score_dir / "manifest.json"),
                "records_sha256": hashlib.sha256(
                    f"records-{spec.run_id}".encode()
                ).hexdigest(),
                "ordered_image_ids_sha256": ids_sha,
                "num_records": 1,
            },
            "contract": {
                "target_dataset": spec.target_dataset,
                "source_datasets": [spec.source_dataset],
                "split_role": "train",
                "requested_split": "train",
                "detector_weight_sha256": checkpoint_sha,
                "split_file_sha256": split_sha,
                "split_ordered_ids_sha256": ids_sha,
                "score_representation": (
                    "raw_logit_float32+sigmoid_probability_float32"
                ),
                "probability_dtype": "float32",
                "logit_dtype": "float32",
                "probability_transform": "sigmoid",
                "probability_clipping": "none",
                "inference_autocast_enabled": False,
            },
        }

    prereg = {
        "budgets": [
            {"name": name, "pixel": pixel, "component": component}
            for name, pixel, component in runner.BUDGETS
        ],
        "threshold_protocol": dict(runner.MATCHING_PROTOCOL),
        "tier1_policy": {
            "strict_macro_pd_gain_minimum": 0.01,
            "strict_each_source_pd_non_degraded": True,
            "all_budget_pooled_pd_non_degraded": True,
            "all_budget_worst_pd_non_degraded": True,
        },
        "outer_target": "NUAA-SIRST",
        "outer_target_images_loaded": False,
        "outer_target_masks_loaded": False,
    }
    decision = {
        "decision": "HOLD",
        "authorizes_tier2": False,
        "authorizes_outer_target_label_access": False,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "input_bindings": {"runs": run_bindings},
    }
    _freeze_historical_json(
        historical / "PHASE3_SOURCE_LODO_PREREGISTRATION.json", prereg
    )
    _freeze_historical_json(historical / "tier1_decision.json", decision)

    def fake_loader(path: str | Path, *, expected_split_role: str):
        assert expected_split_role == "train"
        record = loader_records[str(Path(path).resolve())]
        return (
            record["samples"],
            record["manifest"],
            record["integrity"],
            record["contract"],
        )

    monkeypatch.setattr(runner, "load_formal_raw_logit_directory", fake_loader)
    monkeypatch.setattr(runner, "_load_algorithm_api", _fake_api)
    monkeypatch.setattr(
        runner,
        "_load_historical_probability_points",
        lambda _layout: ({}, {}),
    )
    monkeypatch.setattr(
        runner,
        "_callable_source_bindings",
        lambda _api, _layout: {"synthetic_test_algorithm.py": "a" * 64},
    )
    monkeypatch.setattr(
        runner,
        "_compute_rescue_artifacts",
        lambda *_args, **_kwargs: _fake_artifacts(),
    )
    return {
        "project": project,
        "historical": historical,
        "loader_records": loader_records,
    }


def _snapshot(paths: list[Path]) -> dict[Path, tuple[bytes, int, int]]:
    return {
        path: (
            path.read_bytes(),
            stat.S_IMODE(path.stat().st_mode),
            path.stat().st_mtime_ns,
        )
        for path in paths
    }


def test_runner_freezes_complete_layout_and_is_byte_mtime_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepare_project(tmp_path, monkeypatch)
    historical_paths = sorted(prepared["historical"].glob("*"))
    historical_before = _snapshot(historical_paths)
    output = tmp_path / "rescue"

    first = runner.run_rescue(output_root=output, registered_at=REGISTERED_AT)
    frozen_paths = sorted(output.rglob("*.json")) + sorted(output.rglob("*.sha256"))
    first_snapshot = _snapshot(frozen_paths)
    second = runner.run_rescue(output_root=output, registered_at=REGISTERED_AT)

    assert first == second
    assert _snapshot(frozen_paths) == first_snapshot
    assert _snapshot(historical_paths) == historical_before
    assert all(mode == 0o444 for _, mode, _ in first_snapshot.values())
    expected_json = {
        "protocol_amendment.json",
        "input_manifest.json",
        "input_hashes.json",
        "control.json",
        "full.json",
        "strict.json",
        "medium.json",
        "loose.json",
        "dense_tail_grid_gap.json",
        "cross_domain_calibration_gap.json",
        "false_alarm_concentration.json",
        "evidence_manifest.json",
        "rescue_decision.json",
        "effective_authorization.json",
    }
    assert {path.name for path in output.rglob("*.json")} == expected_json
    for path in output.rglob("*.json"):
        expected = f"{file_sha256(path)}  {path.name}\n"
        assert path.with_suffix(".sha256").read_text(encoding="ascii") == expected

    amendment = json.loads((output / "protocol_amendment.json").read_text())
    assert amendment["post_hoc_amendment"] is True
    assert amendment["frozen_invariants"]["new_training_allowed"] is False
    assert amendment["data_access_policy"]["outer_target_images_authorized"] is False
    assert amendment["data_access_policy"]["outer_target_labels_authorized"] is False


def test_go_authorizes_only_tier2_and_preserves_outer_target_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepare_project(tmp_path, monkeypatch)
    output = tmp_path / "rescue"

    result = runner.run_rescue(output_root=output, registered_at=REGISTERED_AT)

    assert result["decision"] == "RESCUE_GO_TIER2"
    authorization = json.loads(
        (output / "effective_authorization.json").read_text(encoding="utf-8")
    )
    assert authorization["tier2_source_lodo_authorized"] is True
    assert authorization["outer_target_image_access_authorized"] is False
    assert authorization["outer_target_label_access_authorized"] is False
    assert (
        prepared["project"]
        / "outputs/phase_state/HOLD_PHASE3_TARGET_LABEL_ACCESS"
    ).read_text(encoding="utf-8") == "HOLD\n"


def test_nuaa_input_contract_fails_before_any_claim_artifact_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepare_project(tmp_path, monkeypatch)
    first_record = next(iter(prepared["loader_records"].values()))
    first_record["manifest"]["target_dataset"] = "NUAA-SIRST"
    first_record["contract"]["target_dataset"] = "NUAA-SIRST"
    output = tmp_path / "rescue"

    with pytest.raises(RuntimeError, match="NUAA"):
        runner.run_rescue(output_root=output, registered_at=REGISTERED_AT)

    assert not list(output.rglob("*.json"))


def test_algorithm_failure_never_writes_terminal_decision_or_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_project(tmp_path, monkeypatch)
    output = tmp_path / "rescue"
    monkeypatch.setattr(
        runner,
        "_compute_rescue_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )

    with pytest.raises(RuntimeError, match="synthetic failure"):
        runner.run_rescue(output_root=output, registered_at=REGISTERED_AT)

    assert (output / "protocol_amendment.json").is_file()
    assert (output / "input_manifest.json").is_file()
    assert not (output / "rescue_decision.json").exists()
    assert not (output / "effective_authorization.json").exists()


def test_immutable_evidence_drift_fails_closed_without_touching_originals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepare_project(tmp_path, monkeypatch)
    historical_paths = sorted(prepared["historical"].glob("*"))
    historical_before = _snapshot(historical_paths)
    output = tmp_path / "rescue"
    runner.run_rescue(output_root=output, registered_at=REGISTERED_AT)
    strict = output / "operating_points/strict.json"
    strict.chmod(0o644)
    strict.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Immutable JSON content drift"):
        runner.run_rescue(output_root=output, registered_at=REGISTERED_AT)

    assert _snapshot(historical_paths) == historical_before


def test_existing_amendment_timestamp_is_reused_for_implicit_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_project(tmp_path, monkeypatch)
    output = tmp_path / "rescue"
    runner.run_rescue(output_root=output, registered_at=REGISTERED_AT)
    before = _snapshot(sorted(output.rglob("*.json")) + sorted(output.rglob("*.sha256")))

    runner.run_rescue(output_root=output)

    assert _snapshot(list(before)) == before
