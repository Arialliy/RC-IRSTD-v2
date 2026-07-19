from __future__ import annotations

import ast
import fcntl
import importlib
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.artifact_integrity import file_sha256
from scripts import coordinate_phase3_tier2r_component_rescue as base_coordinator
from scripts import coordinate_phase3_tier2r_component_rescue_impl_erratum1 as coordinator
from scripts import launch_phase3_tier2r_impl_erratum1_isolated as launcher
from scripts import launch_phase3_tier2r_isolated as base_launcher
from scripts import register_phase3_tier2r_startup_fix1 as startup_fix1
from scripts import run_phase3_tier2r_exact_gate as base_gate
from scripts import run_phase3_tier2r_exact_gate_impl_erratum1 as gate
from scripts import tier2r_impl_erratum1 as erratum


def _fake_erratum_binding(tmp_path: Path) -> dict[str, str]:
    path = tmp_path / "IMPLEMENTATION_ERRATUM.json"
    path.write_text("{}\n", encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _fake_startup_bindings(tmp_path: Path) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for key, name in (
        ("startup_fix1", "STARTUP_FIX1.json"),
        ("startup_fix1_validation", "STARTUP_FIX1_VALIDATION.json"),
    ):
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        bindings[key] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }
    return bindings


def test_old_chain_is_unchanged_and_only_exporter_hash_may_drift() -> None:
    prereg = erratum.OLD_AUDIT_ROOT / "COMPONENT_RESCUE_PREREGISTRATION.json"
    intent = erratum.OLD_AUDIT_ROOT / "LAUNCH_INTENT.json"
    before = (prereg.read_bytes(), intent.read_bytes())

    chain = erratum._old_chain()
    bindings = chain["preregistration_payload"]["code_bindings"]
    drift = {
        relative
        for relative, digest in bindings.items()
        if digest != file_sha256(erratum.PROJECT_ROOT / relative)
    }

    assert drift == {erratum.EXPORTER_RELATIVE}
    assert chain["old_exporter_sha256"] != chain["new_exporter_sha256"]
    assert (prereg.read_bytes(), intent.read_bytes()) == before
    assert stat.S_IMODE(prereg.stat().st_mode) == 0o444
    assert stat.S_IMODE(intent.stat().st_mode) == 0o444


def test_helper_plan_weights_only_ast_and_no_manifest_precondition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = erratum.implementation_erratum_plan()
    assert plan["scientific_protocol_id"] == "tier2r_c_v1"
    assert plan["execution_instance"] == "tier2r_c_v1_impl_erratum1"
    assert plan["training_unchanged"] is True
    assert plan["outer_target_access_authorized"] is False

    erratum._assert_exporter_weights_only()
    tree = ast.parse((erratum.PROJECT_ROOT / erratum.EXPORTER_RELATIVE).read_text())
    assert any(
        isinstance(node, ast.keyword)
        and node.arg == "weights_only"
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
        for node in ast.walk(tree)
    )

    output = tmp_path / "old-output"
    output.mkdir()
    monkeypatch.setattr(erratum, "OLD_OUTPUT_ROOT", output)
    erratum._assert_no_recovery_outputs()
    manifest = output / "seed43/control/heldout_nudt/scores_heldout_train/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="precede every score manifest"):
        erratum._assert_no_recovery_outputs()


def test_verify_only_plan_writes_no_amendment_and_round1_is_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prereg = erratum.OLD_AUDIT_ROOT / "COMPONENT_RESCUE_PREREGISTRATION.json"
    intent = erratum.OLD_AUDIT_ROOT / "LAUNCH_INTENT.json"
    before = (prereg.read_bytes(), intent.read_bytes())
    amendment = tmp_path / "IMPLEMENTATION_ERRATUM.json"
    monkeypatch.setattr(erratum, "ERRATUM_PATH", amendment)
    monkeypatch.setattr(erratum, "FAILURE_EVIDENCE_ROOT", tmp_path / "evidence")
    # The temporal no-output guard is covered independently above. Isolate it
    # here because the frozen recovery outputs legitimately exist post-run.
    monkeypatch.setattr(erratum, "_assert_no_recovery_outputs", lambda: None)

    result = erratum.ensure_implementation_erratum(create=False)
    runs = erratum._round1_bindings()

    assert result["registered"] is False
    assert not amendment.exists()
    assert not (tmp_path / "evidence").exists()
    assert set(runs) == set(erratum.ROUND1_RUNS)
    assert all(value["checkpoint_epoch"] == 79 for value in runs.values())
    assert all(value["reuse_without_retraining"] is True for value in runs.values())
    assert (prereg.read_bytes(), intent.read_bytes()) == before


def test_amendment_write_once_is_read_only_and_rejects_drift(tmp_path: Path) -> None:
    path = tmp_path / "amendment.json"
    content = erratum._canonical_json_bytes({"scope": "implementation-only"})
    first = erratum._write_once(path, content)
    assert erratum._write_once(path, content) == first
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert stat.S_IMODE(erratum._sidecar(path).stat().st_mode) == 0o444
    with pytest.raises(RuntimeError, match="immutable implementation erratum drift"):
        erratum._write_once(path, b"{}\n")


def test_packet_preregistration_and_handoff_carry_erratum_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _fake_erratum_binding(tmp_path)
    startup_bindings = _fake_startup_bindings(tmp_path)
    monkeypatch.setattr(
        coordinator, "_ORIGINAL_VERIFY_PREREQUISITES", lambda: {"code_bindings": {}}
    )
    monkeypatch.setattr(erratum, "ERRATUM_PATH", Path(binding["path"]))
    monkeypatch.setattr(erratum, "ensure_implementation_erratum", lambda **_: {})
    monkeypatch.setattr(erratum, "implementation_erratum_binding", lambda: binding)

    packet = coordinator.verify_prerequisites(startup_bindings=startup_bindings)
    assert packet["implementation_erratum"] == binding
    assert set(erratum.RECOVERY_CODE_RELATIVES) <= set(packet["code_bindings"])
    assert startup_fix1.REGISTRAR_RELATIVE not in packet["code_bindings"]
    for key, value in startup_bindings.items():
        assert packet[key] == value

    monkeypatch.setattr(
        coordinator,
        "_ORIGINAL_PREREGISTRATION_PAYLOAD",
        lambda *_: {"source_only": True},
    )
    monkeypatch.setattr(
        coordinator, "_ORIGINAL_HANDOFF_PAYLOAD", lambda *_: {"source_only": True}
    )
    prereg = coordinator._preregistration_payload(packet, tmp_path / "auth", tmp_path / "p")
    handoff = coordinator._handoff_payload(
        packet, tmp_path / "p", tmp_path / "auth", tmp_path / "h"
    )
    for payload in (prereg, handoff):
        assert payload["execution_instance"] == erratum.EXECUTION_INSTANCE
        assert payload["implementation_erratum"] == binding
        assert payload["implementation_erratum_plan"] == erratum.implementation_erratum_plan()
        for key, value in startup_bindings.items():
            assert payload[key] == value


def test_status_binding_is_scoped_injected_and_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    startup_bindings = _fake_startup_bindings(tmp_path)
    statuses: list[tuple[str, dict]] = []

    def capture(status: str, **fields: object) -> None:
        statuses.append((status, fields))

    monkeypatch.setattr(base_coordinator, "_write_status", capture)
    before = base_coordinator._write_status
    with coordinator._patched_coordinator(startup_bindings=startup_bindings):
        assert base_coordinator._write_status is coordinator._write_status
        base_coordinator._write_status("bound", marker=1)
        with pytest.raises(RuntimeError, match="startup_fix1 status binding drift"):
            base_coordinator._write_status("drift", startup_fix1={})

    assert base_coordinator._write_status is before
    assert coordinator._ACTIVE_STARTUP_BINDINGS is None
    assert coordinator._ACTIVE_STATUS_DELEGATE is None
    assert statuses == [
        (
            "bound",
            {
                "marker": 1,
                **startup_bindings,
            },
        )
    ]


def test_execution_verifies_startup_before_lock_erratum_and_base_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    startup_bindings = _fake_startup_bindings(tmp_path)
    events: list[str] = []

    class FakeLock:
        def close(self) -> None:
            events.append("close")

    def verify_startup() -> dict[str, dict[str, str]]:
        events.append("verify_startup")
        return startup_bindings

    def acquire_lock(_path: Path) -> FakeLock:
        events.append("lock")
        return FakeLock()

    def ensure_erratum(*, create: bool) -> dict[str, str]:
        assert create is True
        events.append("erratum")
        return {}

    def base_run(*, verify_only: bool) -> dict[str, bool]:
        assert verify_only is False
        assert coordinator._ACTIVE_STARTUP_BINDINGS == startup_bindings
        events.append("base_run")
        return {"completed": True}

    monkeypatch.setattr(startup_fix1, "verify_frozen_startup_fix1", verify_startup)
    monkeypatch.setattr(coordinator, "_exclusive_existing_readonly_lock", acquire_lock)
    monkeypatch.setattr(erratum, "ensure_implementation_erratum", ensure_erratum)
    monkeypatch.setattr(base_coordinator, "run", base_run)
    monkeypatch.setattr(coordinator.fcntl, "flock", lambda *_: events.append("unlock"))
    original_status = base_coordinator._write_status

    assert coordinator.run(verify_only=False) == {"completed": True}
    assert events == [
        "verify_startup",
        "lock",
        "erratum",
        "base_run",
        "unlock",
        "close",
    ]
    assert base_coordinator._write_status is original_status
    assert coordinator._ACTIVE_STARTUP_BINDINGS is None
    assert coordinator._ACTIVE_STATUS_DELEGATE is None


def test_execution_fails_before_lock_or_formal_mutation_without_startup_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        startup_fix1,
        "verify_frozen_startup_fix1",
        lambda: (_ for _ in ()).throw(RuntimeError("startup evidence absent")),
    )
    monkeypatch.setattr(
        coordinator,
        "_exclusive_existing_readonly_lock",
        lambda *_: pytest.fail("lock must follow startup verification"),
    )
    monkeypatch.setattr(
        erratum,
        "ensure_implementation_erratum",
        lambda **_: pytest.fail("erratum must follow startup verification"),
    )
    monkeypatch.setattr(
        base_coordinator,
        "run",
        lambda **_: pytest.fail("base run must follow startup verification"),
    )

    with pytest.raises(RuntimeError, match="startup evidence absent"):
        coordinator.run(verify_only=False)


def test_verify_only_does_not_require_registered_startup_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        startup_fix1,
        "verify_frozen_startup_fix1",
        lambda: pytest.fail("verify-only must not require startup evidence"),
    )
    monkeypatch.setattr(
        base_coordinator,
        "run",
        lambda *, verify_only: {"verified": verify_only},
    )
    before_status = base_coordinator._write_status

    result = coordinator.run(verify_only=True)

    assert result["verified"] is True
    assert result["implementation_erratum_plan"] == erratum.implementation_erratum_plan()
    assert base_coordinator._write_status is before_status
    assert coordinator._ACTIVE_STARTUP_BINDINGS is None
    assert coordinator._ACTIVE_STATUS_DELEGATE is None


def test_complete_round_reuses_old_output_without_calling_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = [SimpleNamespace(run_id="control"), SimpleNamespace(run_id="c")]
    statuses: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        base_coordinator,
        "inspect_run",
        lambda *_: base_coordinator.RunInspection("complete", epoch=79),
    )
    monkeypatch.setattr(
        coordinator,
        "_ORIGINAL_RUN_TRAINING_ROUND",
        lambda *_: pytest.fail("complete round must not call trainer"),
    )
    monkeypatch.setattr(
        base_coordinator, "_write_status", lambda status, **fields: statuses.append((status, fields))
    )
    monkeypatch.setattr(
        erratum, "implementation_erratum_binding", lambda: _fake_erratum_binding(tmp_path)
    )

    coordinator._run_training_round(jobs, {})

    assert coordinator.OLD_OUTPUT_ROOT == erratum.OLD_OUTPUT_ROOT
    assert statuses[0][0] == "tier2r_reusing_frozen_complete_runs"
    assert statuses[0][1]["trainer_processes_launched"] == 0
    assert statuses[0][1]["checkpoint_epochs"] == [79, 79]


def test_historical_lock_is_existing_readonly_regular_and_exclusive(
    tmp_path: Path,
) -> None:
    path = tmp_path / "historical.lock"
    path.write_bytes(b"")
    path.chmod(0o444)

    handle = coordinator._exclusive_existing_readonly_lock(path)
    assert handle.readable() is True
    assert handle.writable() is False
    contender = path.open("rb", buffering=0)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        contender.close()
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()

    with pytest.raises(RuntimeError, match="absent, non-regular, or a symlink"):
        coordinator._exclusive_existing_readonly_lock(tmp_path / "missing.lock")
    symlink = tmp_path / "symlink.lock"
    symlink.symlink_to(path)
    with pytest.raises(RuntimeError, match="absent, non-regular, or a symlink"):
        coordinator._exclusive_existing_readonly_lock(symlink)


def test_gate_requires_erratum_and_cross_binds_preregistration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _fake_erratum_binding(tmp_path)
    monkeypatch.setattr(erratum, "ERRATUM_PATH", Path(binding["path"]))
    monkeypatch.setattr(erratum, "ensure_implementation_erratum", lambda **_: {})
    monkeypatch.setattr(erratum, "implementation_erratum_binding", lambda: binding)
    payload = {
        "execution_instance": erratum.EXECUTION_INSTANCE,
        "implementation_erratum": binding,
        "implementation_erratum_plan": erratum.implementation_erratum_plan(),
        "source_only": True,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "outer_target_access_authorized": False,
    }
    gate._verify_erratum_binding(payload, label="test")
    with pytest.raises(RuntimeError, match="implementation-erratum binding drift"):
        gate._verify_erratum_binding({**payload, "implementation_erratum": {}}, label="test")

    monkeypatch.setattr(gate, "_ORIGINAL_VALIDATE_HANDOFF", lambda *_args, **_kw: (payload, {}, {}))
    monkeypatch.setattr(base_gate, "_verify_frozen_json", lambda *_: payload)
    assert gate.validate_handoff(tmp_path / "handoff.json", project_root=tmp_path)[0] == payload
    monkeypatch.setattr(
        base_gate, "_verify_frozen_json", lambda *_: {**payload, "implementation_erratum": {}}
    )
    with pytest.raises(RuntimeError, match="cross-binding drift"):
        gate.validate_handoff(tmp_path / "handoff.json", project_root=tmp_path)


def test_launcher_is_new_scoped_namespace_with_locked_target_and_bounded_restart() -> None:
    names = (
        "COORDINATOR",
        "AUDIT_ROOT",
        "CONTAINER_NAME",
        "SCHEMA",
        "bind_mount_contract",
        "container_spec",
    )
    before = {name: getattr(base_launcher, name) for name in names}
    importlib.reload(launcher)
    assert {name: getattr(base_launcher, name) for name in names} == before

    spec = launcher.container_spec(formal=True)
    command = launcher.build_formal_command("a" * 64)
    mounts = {value["destination"]: value for value in launcher.bind_mount_contract()}
    target = {value["destination"]: value for value in spec["tmpfs"]}[
        str(launcher.base.FORBIDDEN_TARGET)
    ]

    assert spec["name"] == launcher.CONTAINER_NAME
    assert spec["command"] == [str(launcher.COORDINATOR)]
    assert spec["coordinator_physical_gpu_indices"] == [2, 3]
    assert spec["restart_policy"] == {"Name": "on-failure", "MaximumRetryCount": 3}
    assert command[command.index("--restart") + 1] == "on-failure:3"
    assert mounts[str(launcher.PROJECT_ROOT)]["readonly"] is True
    assert mounts[str(launcher.OUTPUT_ROOT)]["readonly"] is False
    assert mounts[str(launcher.AUDIT_ROOT)]["readonly"] is False
    assert "mode=000" in target["options"]
    assert {name: getattr(base_launcher, name) for name in names} == before


def test_launcher_verify_only_removes_cold_mountpoint_and_registers_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = tmp_path / "formal/audit"
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(launcher, "AUDIT_ROOT", audit)
    monkeypatch.setattr(launcher, "OUTPUT_ROOT", output)
    monkeypatch.setattr(launcher, "_validate_nonmutating_host_sources", lambda: None)
    inventory = tuple(
        {"index": index, "uuid": f"GPU-{index}", "name": "RTX"}
        for index in range(4)
    )
    monkeypatch.setattr(base_launcher, "_inspect_image", lambda **_: base_launcher.IMAGE_ID)
    monkeypatch.setattr(base_launcher, "_gpu_inventory", lambda **_: inventory)
    monkeypatch.setattr(
        base_launcher,
        "_verify_in_container",
        lambda **_: {"implementation_erratum_plan": erratum.implementation_erratum_plan()},
    )
    monkeypatch.setattr(base_launcher, "_verify_container_attestation", lambda *_a, **_k: {"verified": True})
    monkeypatch.setattr(
        launcher,
        "_superseded_execution_snapshot",
        lambda **_: {"status": "failed_closed", "container": {"restart_policy": {"Name": "no"}}},
    )

    result = launcher.verify_only()

    assert result["verified"] is True
    assert result["formal_artifact_registered"] is False
    assert result["coordinator_assigned_host_gpus"] == list(inventory[2:4])
    assert not audit.exists()
    assert not (audit / "IMPLEMENTATION_ERRATUM.json").exists()
