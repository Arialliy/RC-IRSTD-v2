from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import coordinate_phase3_tier2_after_rescue as coordinator
from scripts import run_phase3_tier2_raw_logit_gate as tier2_gate


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _packet(root: Path) -> dict[str, Any]:
    historical_root = root / "artifacts/aaai27/audit/phase3_source_lodo_gate"
    return {
        "schema_version": coordinator.SCHEMA,
        "verified": True,
        "source_only": True,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "preregistration": {
            "path": str(historical_root / "PHASE3_SOURCE_LODO_PREREGISTRATION.json"),
            "sha256": "1" * 64,
        },
        "historical_tier1_decision": {
            "path": str(historical_root / "tier1_decision.json"),
            "sha256": "2" * 64,
        },
        "historical_coordinator": {
            "path": str(root / "scripts/coordinate_phase3_source_lodo_gate.py"),
            "sha256": "3" * 64,
        },
        "rescue_protocol_amendment": {
            "path": str(historical_root / "raw_logit_rescue_v1/protocol_amendment.json"),
            "sha256": "4" * 64,
        },
        "rescue_input_manifest": {
            "path": str(historical_root / "raw_logit_rescue_v1/input_manifest.json"),
            "sha256": "5" * 64,
        },
        "rescue_input_hashes": {
            "path": str(historical_root / "raw_logit_rescue_v1/input_hashes.json"),
            "sha256": "6" * 64,
        },
        "rescue_evidence_manifest": {
            "path": str(historical_root / "raw_logit_rescue_v1/evidence_manifest.json"),
            "sha256": "7" * 64,
        },
        "rescue_decision": {
            "path": str(historical_root / "raw_logit_rescue_v1/rescue_decision.json"),
            "sha256": "8" * 64,
        },
        "effective_authorization": {
            "path": str(historical_root / "raw_logit_rescue_v1/effective_authorization.json"),
            "sha256": "9" * 64,
        },
        "tier2_policy": coordinator.EXPECTED_TIER2_POLICY,
        "tier2_schedule": [
            {
                "run_id": spec.run_id,
                "role": spec.role,
                "fold": spec.fold_key,
                "held_out_source": spec.fold.held_out_name,
                "training_source": spec.fold.train_name,
                "physical_gpu": spec.physical_gpu,
                "logical_device": "cuda:0",
            }
            for round_specs in coordinator.historical.TIER2_PREREGISTERED_ROUNDS
            for spec in round_specs
        ],
    }


def _json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _make_product(layout: Any, spec: Any) -> None:
    run_dir = layout.output / spec.role / spec.fold_key
    score_dir = run_dir / "scores_heldout_train"
    score_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "last.pt"
    checkpoint.write_bytes((spec.run_id + " checkpoint").encode("ascii"))
    checkpoint_sha = _sha(checkpoint)
    _json(
        run_dir / "PHASE3_IDENTITY.json",
        {
            "run_id": spec.run_id,
            "checkpoint_sha256": checkpoint_sha,
            "outer_target_images_used": False,
            "outer_target_labels_used": False,
        },
    )
    _json(
        run_dir / "EXPORT_IDENTITY.json",
        {
            "run_id": spec.run_id,
            "checkpoint_sha256": checkpoint_sha,
            "outer_target_images_used": False,
            "outer_target_labels_used": False,
        },
    )
    _json(
        score_dir / "manifest.json",
        {
            "run_id": spec.run_id,
            "labels_loaded": True,
            "split_role": "train",
            "logit_dtype": "float32",
            "inference_autocast_enabled": False,
            "records_sha256": hashlib.sha256(spec.run_id.encode()).hexdigest(),
            "raw_logit_stream_sha256": hashlib.sha256(
                (spec.run_id + " logits").encode()
            ).hexdigest(),
            "outer_target_images_used": False,
            "outer_target_labels_used": False,
        },
    )


@pytest.fixture
def mocked_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = tmp_path / "project"
    root.mkdir()
    gate_runner = root / "scripts/run_phase3_tier2_raw_logit_gate.py"
    gate_runner.parent.mkdir(parents=True)
    gate_runner.write_text("# test raw-logit gate\n", encoding="utf-8")
    continuation = root / "scripts/coordinate_phase3_tier2_after_rescue.py"
    continuation.write_text("# test Tier2 continuation\n", encoding="utf-8")
    packet = _packet(root)
    for key in ("preregistration", "rescue_decision", "effective_authorization"):
        binding_path = Path(packet[key]["path"])
        coordinator._write_once_json(
            binding_path,
            {
                "binding": key,
                "tier2_policy": coordinator.EXPECTED_TIER2_POLICY,
            },
        )
        packet[key]["sha256"] = _sha(binding_path)
    calls: dict[str, list[Any]] = {
        "rounds": [],
        "exports": [],
        "gates": [],
        "main_gates": [],
        "verifications": [],
    }

    monkeypatch.setattr(coordinator, "CANONICAL_PROJECT_ROOT", root)
    monkeypatch.setattr(coordinator, "PROJECT_ROOT", root)
    monkeypatch.setattr(coordinator, "EXPECTED_PYTHON", Path(sys.executable))
    monkeypatch.setattr(coordinator, "__file__", str(continuation))
    monkeypatch.setattr(coordinator, "verify_authorization", lambda: packet)
    monkeypatch.setattr(
        coordinator,
        "_phase3_locks",
        lambda: contextlib.nullcontext(),
    )

    def train(layout: Any, name: str, specs: Any) -> None:
        calls["rounds"].append((name, tuple(spec.run_id for spec in specs)))

    def export(layout: Any, spec: Any) -> dict[str, Any]:
        calls["exports"].append(spec.run_id)
        _make_product(layout, spec)
        return {"run_id": spec.run_id}

    def probability_bomb(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the historical probability sweep must never be called")

    def fake_subprocess(command: Any, **kwargs: Any) -> Any:
        frozen_command = tuple(command)
        calls["gates"].append(frozen_command)
        output_root = Path(command[command.index("--output-root") + 1])
        if "--verify-only" in command:
            calls["verifications"].append(frozen_command)
            decision = output_root / "tier2_decision.json"
            authorization = output_root / "effective_authorization.json"
            try:
                valid = (
                    json.loads(decision.read_text(encoding="utf-8"))
                    == {"valid": True, "kind": "decision"}
                    and json.loads(authorization.read_text(encoding="utf-8"))
                    == {"valid": True, "kind": "authorization"}
                )
            except (FileNotFoundError, json.JSONDecodeError):
                valid = False
            if not valid:
                return SimpleNamespace(
                    returncode=31,
                    stdout=b"",
                    stderr=b"mock frozen gate product verification failed",
                )
            return SimpleNamespace(returncode=0, stdout=b'{"verified":true}\n', stderr=b"")
        calls["main_gates"].append(frozen_command)
        kwargs["stdout"].write(b"mock gate completed\n")
        _json(output_root / "tier2_decision.json", {"valid": True, "kind": "decision"})
        _json(
            output_root / "effective_authorization.json",
            {"valid": True, "kind": "authorization"},
        )
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(coordinator.historical, "run_training_round", train)
    monkeypatch.setattr(coordinator.historical, "ensure_export", export)
    monkeypatch.setattr(coordinator.historical, "ensure_sweep", probability_bomb)
    monkeypatch.setattr(coordinator.subprocess, "run", fake_subprocess)
    return {
        "root": root,
        "runner": gate_runner,
        "packet": packet,
        "calls": calls,
    }


def test_live_frozen_rescue_authorization_chain_passes() -> None:
    packet = coordinator.verify_authorization()
    assert packet["verified"] is True
    assert packet["rescue_decision"]["sha256"] == (
        "69756c5e7fec7d1dac549a19f11418a2ac9215d8ae3c718dc353ee8d24de253b"
    )
    assert {item["physical_gpu"] for item in packet["tier2_schedule"]} == {2, 3}
    assert len(packet["tier2_schedule"]) == 4


def test_verify_only_is_entirely_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "empty-project"
    root.mkdir()
    monkeypatch.setattr(coordinator, "CANONICAL_PROJECT_ROOT", root)
    monkeypatch.setattr(coordinator, "PROJECT_ROOT", root)
    monkeypatch.setattr(coordinator, "EXPECTED_PYTHON", Path(sys.executable))
    monkeypatch.setattr(coordinator, "verify_authorization", lambda: _packet(root))
    monkeypatch.setattr(
        coordinator, "_phase3_locks", lambda: contextlib.nullcontext()
    )
    before = list(root.rglob("*"))
    result = coordinator.run(verify_only=True)
    after = list(root.rglob("*"))
    assert result["verified"] is True
    assert before == after == []


def test_full_execution_freezes_handoff_and_never_runs_probability_sweep(
    mocked_execution: dict[str, Any],
) -> None:
    result = coordinator.run()
    root = mocked_execution["root"]
    calls = mocked_execution["calls"]
    output = (
        root
        / "artifacts/aaai27/audit/phase3_source_lodo_gate/tier2_raw_logit_gate_v1"
    )
    assert result["status"] == "tier2_raw_logit_gate_completed"
    assert len(calls["rounds"]) == 2
    assert len(calls["exports"]) == 4
    assert len(calls["main_gates"]) == 1
    assert len(calls["verifications"]) == 1
    assert len(calls["gates"]) == 2
    command = calls["main_gates"][0]
    assert command[2:4] == ("--handoff", str((output / "TIER2_HANDOFF.json").resolve()))
    assert command[4:6] == ("--output-root", str(output.resolve()))
    handoff_path = output / "TIER2_HANDOFF.json"
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    assert set(handoff["runs"]) == coordinator.TIER2_RUN_IDS
    assert handoff["exact_raw_logit_continuation"] == {
        "primary_score_domain": "float32_raw_logit",
        "prediction_rule": "float32_raw_logit >= shared_threshold_logit",
        "exact_state_enumeration_is_primary": True,
        "shared_threshold_across_nudt_and_irstd": True,
        "dense_grid_is_diagnostic_only": True,
        "probability_threshold_sweep_used": False,
    }
    assert handoff["runner"]["sha256"] == _sha(mocked_execution["runner"])
    for path in (
        handoff_path,
        handoff_path.with_suffix(".sha256"),
        output / "TIER2_EXECUTION_INTENT.json",
        output / "TIER2_EXECUTION_INTENT.sha256",
        output / "TIER2_GATE_COMPLETED.json",
        output / "TIER2_GATE_COMPLETED.sha256",
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
    completion = json.loads((output / "TIER2_GATE_COMPLETED.json").read_text())
    assert completion["frozen_product_verification"]["returncode"] == 0
    assert "--verify-only" in completion["frozen_product_verification"]["command"]
    status = json.loads((output / "tier2_status.json").read_text())
    phase3_status = json.loads(
        (
            root
            / "artifacts/aaai27/audit/phase3_source_lodo_gate/phase3_status.json"
        ).read_text()
    )
    assert status["status"] == "tier2_raw_logit_gate_completed"
    assert phase3_status["status"] == "tier2_after_rescue_completed"


def test_restart_reuses_gate_completion_idempotently(
    mocked_execution: dict[str, Any],
) -> None:
    first = coordinator.run()
    second = coordinator.run()
    calls = mocked_execution["calls"]
    assert first["handoff_sha256"] == second["handoff_sha256"]
    assert len(calls["main_gates"]) == 1
    assert len(calls["verifications"]) == 2
    assert len(calls["gates"]) == 3
    assert len(calls["rounds"]) == 4
    assert len(calls["exports"]) == 8


def test_frozen_handoff_is_accepted_by_dedicated_gate_runner(
    mocked_execution: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator.run()
    root = mocked_execution["root"]
    handoff = (
        root
        / "artifacts/aaai27/audit/phase3_source_lodo_gate"
        / "tier2_raw_logit_gate_v1/TIER2_HANDOFF.json"
    )
    monkeypatch.setattr(tier2_gate, "__file__", str(mocked_execution["runner"]))
    payload, runs = tier2_gate.validate_handoff(handoff, project_root=root)
    assert payload["tier2_source_lodo_authorized"] is True
    assert set(runs) == coordinator.TIER2_RUN_IDS


@pytest.mark.parametrize("broken_product", ["missing_decision", "bad_authorization"])
def test_completion_reuse_rejects_missing_or_corrupt_gate_products(
    mocked_execution: dict[str, Any], broken_product: str
) -> None:
    coordinator.run()
    output = (
        mocked_execution["root"]
        / "artifacts/aaai27/audit/phase3_source_lodo_gate/tier2_raw_logit_gate_v1"
    )
    log_path = output / "tier2_raw_logit_gate.log"
    original_log = log_path.read_bytes()
    if broken_product == "missing_decision":
        (output / "tier2_decision.json").unlink()
    else:
        (output / "effective_authorization.json").write_text(
            "{broken-json\n", encoding="utf-8"
        )
    with pytest.raises(RuntimeError, match="frozen-product verification failed"):
        coordinator.run()
    assert len(mocked_execution["calls"]["main_gates"]) == 1
    assert len(mocked_execution["calls"]["verifications"]) == 2
    assert log_path.read_bytes() == original_log
    assert json.loads((output / "tier2_status.json").read_text())["status"] == (
        "failed_closed"
    )


def test_missing_required_runner_fails_closed_before_training(
    mocked_execution: dict[str, Any],
) -> None:
    mocked_execution["runner"].unlink()
    with pytest.raises(FileNotFoundError, match="Required Tier2 raw-logit gate"):
        coordinator.run()
    calls = mocked_execution["calls"]
    assert calls["rounds"] == []
    output = (
        mocked_execution["root"]
        / "artifacts/aaai27/audit/phase3_source_lodo_gate/tier2_raw_logit_gate_v1"
    )
    status = json.loads((output / "tier2_status.json").read_text())
    assert status["status"] == "failed_closed"
    assert not (output / "TIER2_HANDOFF.json").exists()


def test_authorization_drift_fails_before_any_training_and_records_status(
    mocked_execution: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        coordinator,
        "verify_authorization",
        lambda: (_ for _ in ()).throw(RuntimeError("rescue sidecar drift")),
    )
    with pytest.raises(RuntimeError, match="rescue sidecar drift"):
        coordinator.run()
    assert mocked_execution["calls"]["rounds"] == []
    output = (
        mocked_execution["root"]
        / "artifacts/aaai27/audit/phase3_source_lodo_gate/tier2_raw_logit_gate_v1"
    )
    assert json.loads((output / "tier2_status.json").read_text())["status"] == (
        "failed_closed"
    )


def test_gate_failure_is_non_authorizing_and_has_no_completion(
    mocked_execution: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        coordinator.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=17),
    )
    with pytest.raises(RuntimeError, match="exit code 17"):
        coordinator.run()
    output = (
        mocked_execution["root"]
        / "artifacts/aaai27/audit/phase3_source_lodo_gate/tier2_raw_logit_gate_v1"
    )
    assert not (output / "TIER2_GATE_COMPLETED.json").exists()
    assert json.loads((output / "tier2_status.json").read_text())["status"] == (
        "failed_closed"
    )


def test_main_gate_success_without_successful_verify_never_freezes_completion(
    mocked_execution: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[tuple[str, ...]] = []

    def main_then_verify_failure(command: Any, **kwargs: Any) -> Any:
        commands.append(tuple(command))
        if "--verify-only" in command:
            return SimpleNamespace(
                returncode=29,
                stdout=b"",
                stderr=b"post-run verification rejected products",
            )
        kwargs["stdout"].write(b"main returned zero\n")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(coordinator.subprocess, "run", main_then_verify_failure)
    with pytest.raises(RuntimeError, match="frozen-product verification failed"):
        coordinator.run()
    assert len(commands) == 2
    assert "--verify-only" not in commands[0]
    assert "--verify-only" in commands[1]
    output = (
        mocked_execution["root"]
        / "artifacts/aaai27/audit/phase3_source_lodo_gate/tier2_raw_logit_gate_v1"
    )
    assert not (output / "TIER2_GATE_COMPLETED.json").exists()


def test_frozen_pair_rejects_mode_and_digest_drift(tmp_path: Path) -> None:
    path = tmp_path / "claim.json"
    coordinator._write_once_json(path, {"claim": "source-only"})
    assert coordinator._verify_frozen_pair(path) == _sha(path)
    path.chmod(0o644)
    with pytest.raises(RuntimeError, match="mode is not 0444"):
        coordinator._verify_frozen_pair(path)
    path.chmod(0o444)
    sidecar = path.with_suffix(".sha256")
    sidecar.chmod(0o644)
    sidecar.write_text("0" * 64 + "  claim.json\n", encoding="ascii")
    sidecar.chmod(0o444)
    with pytest.raises(RuntimeError, match="sidecar mismatch"):
        coordinator._verify_frozen_pair(path)


def test_same_historical_lock_contention_fails_without_launch(tmp_path: Path) -> None:
    lock = tmp_path / "phase3_source_lodo_coordinator.lock"
    lock.write_bytes(b"")
    with lock.open("r+b") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already held"):
            with coordinator._exclusive_existing_lock(lock):
                pytest.fail("contended lock was acquired")


def test_outer_target_true_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="Forbidden outer-target boolean"):
        coordinator._verify_outer_false_recursively(
            {"nested": {"outer_target_images_used": True}}
        )
