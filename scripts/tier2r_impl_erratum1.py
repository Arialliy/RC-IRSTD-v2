#!/usr/bin/env python3
"""Immutable implementation-only erratum for the Tier2R export recovery.

The scientific protocol and frozen round-1 checkpoints are inherited without
change.  This amendment records the NumPy 1.x/2.x safe-loader correction and
the exact failed-export evidence before any recovery output is created.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from evaluation.artifact_integrity import file_sha256


PROJECT_ROOT = Path("/home/ly/RC-IRSTD-v2")
OLD_AUDIT_RELATIVE = Path(
    "artifacts/aaai27/audit/component_rescue/tier2r_c_v1"
)
NEW_AUDIT_RELATIVE = Path(
    "artifacts/aaai27/audit/component_rescue/tier2r_c_v1_impl_erratum1"
)
NEW_GATE_RELATIVE = NEW_AUDIT_RELATIVE / "exact_gate"
OLD_OUTPUT_RELATIVE = Path(
    "outputs/aaai27/detectors/component_rescue/tier2r_c_v1"
)

OLD_AUDIT_ROOT = PROJECT_ROOT / OLD_AUDIT_RELATIVE
NEW_AUDIT_ROOT = PROJECT_ROOT / NEW_AUDIT_RELATIVE
OLD_OUTPUT_ROOT = PROJECT_ROOT / OLD_OUTPUT_RELATIVE
ERRATUM_PATH = NEW_AUDIT_ROOT / "IMPLEMENTATION_ERRATUM.json"
FAILURE_EVIDENCE_ROOT = NEW_AUDIT_ROOT / "failure_evidence"

SCHEMA_VERSION = "rc-irstd-aaai27-tier2r-implementation-erratum-v1"
SCIENTIFIC_PROTOCOL_ID = "tier2r_c_v1"
EXECUTION_INSTANCE = "tier2r_c_v1_impl_erratum1"
CHANGE_SCOPE = "checkpoint_loader_numpy_1x_2x_compatibility_only"
EXPECTED_FAILURE = (
    "Tier2R export failed closed: "
    "seed43_control_heldout_nudt:exit=1; "
    "seed43_c_heldout_nudt:exit=1"
)
ROUND1_RUNS = (
    "seed43_control_heldout_nudt",
    "seed43_c_heldout_nudt",
)
RECOVERY_CODE_RELATIVES = (
    "scripts/tier2r_impl_erratum1.py",
    "scripts/coordinate_phase3_tier2r_component_rescue_impl_erratum1.py",
    "scripts/run_phase3_tier2r_exact_gate_impl_erratum1.py",
)
EXPORTER_RELATIVE = "evaluation/export_score_maps.py"
REGRESSION_TEST_RELATIVE = "tests/test_export_score_maps.py"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sidecar(path: Path) -> Path:
    return path.with_suffix(".sha256")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=path.name + ".tmp.", delete=False
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _write_once(path: Path, content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()
    sidecar = _sidecar(path)
    sidecar_content = f"{digest}  {path.name}\n".encode("ascii")
    if path.exists() or sidecar.exists():
        if (
            path.is_symlink()
            or sidecar.is_symlink()
            or not path.is_file()
            or not sidecar.is_file()
            or path.read_bytes() != content
            or sidecar.read_bytes() != sidecar_content
        ):
            raise RuntimeError(f"immutable implementation erratum drift: {path}")
        return digest
    _atomic_write(path, content)
    _atomic_write(sidecar, sidecar_content)
    path.chmod(0o444)
    sidecar.chmod(0o444)
    return digest


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return payload


def _assert_regular(path: Path, *, readonly: bool = False) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required regular file is absent or a symlink: {path}")
    if readonly and stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise RuntimeError(f"immutable file remains writable: {path}")


def _verify_frozen_json(
    path: Path, *, sidecar: Path | None = None
) -> dict[str, Any]:
    digest_path = sidecar or _sidecar(path)
    _assert_regular(path, readonly=True)
    _assert_regular(digest_path, readonly=True)
    digest = file_sha256(path)
    expected = f"{digest}  {path.name}\n"
    if digest_path.read_text(encoding="ascii") != expected:
        raise RuntimeError(f"immutable sidecar drift: {digest_path}")
    return _load_object(path)


def _binding(path: Path) -> dict[str, str]:
    _assert_regular(path)
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _verify_binding(
    record: Any, *, expected: Path, label: str, readonly: bool = False
) -> Path:
    if not isinstance(record, Mapping):
        raise RuntimeError(f"{label} binding is absent")
    path = Path(str(record.get("path", "")))
    if path.resolve() != expected.resolve():
        raise RuntimeError(f"{label} path drift")
    _assert_regular(path, readonly=readonly)
    if record.get("sha256") != file_sha256(path):
        raise RuntimeError(f"{label} SHA-256 drift")
    return path


def _old_chain() -> dict[str, Any]:
    prereg_path = OLD_AUDIT_ROOT / "COMPONENT_RESCUE_PREREGISTRATION.json"
    prereg = _verify_frozen_json(prereg_path)
    intent_path = OLD_AUDIT_ROOT / "LAUNCH_INTENT.json"
    intent = _verify_frozen_json(
        intent_path, sidecar=OLD_AUDIT_ROOT / "LAUNCH_INTENT.json.sha256"
    )
    status_path = OLD_AUDIT_ROOT / "COMPONENT_RESCUE_STATUS.json"
    _assert_regular(status_path)
    status = _load_object(status_path)
    if (
        status.get("status") != "failed_closed"
        or status.get("error_type") != "RuntimeError"
        or status.get("error") != EXPECTED_FAILURE
        or status.get("source_only") is not True
        or status.get("outer_target_images_used") is not False
        or status.get("outer_target_labels_used") is not False
        or status.get("outer_target_access_authorized") is not False
    ):
        raise RuntimeError("historical Tier2R failure status drift")
    for payload, label in ((prereg, "preregistration"), (intent, "launch intent")):
        if (
            payload.get("source_only") is not True
            or payload.get("outer_target_images_used") is not False
            or payload.get("outer_target_labels_used") is not False
            or payload.get("outer_target_access_authorized") is not False
        ):
            raise RuntimeError(f"historical {label} target-lock drift")
    protocol_path = PROJECT_ROOT / "configs/tier2r_component_rescue_protocol.json"
    if prereg.get("protocol", {}).get("sha256") != file_sha256(protocol_path):
        raise RuntimeError("scientific protocol SHA-256 drift")
    protocol = _load_object(protocol_path)
    if (
        prereg.get("protocol_id") != SCIENTIFIC_PROTOCOL_ID
        or prereg.get("score_protocol") != protocol.get("score_protocol")
        or prereg.get("decision_protocol") != protocol.get("decision_protocol")
        or prereg.get("training_protocol") != protocol.get("training_protocol")
    ):
        raise RuntimeError("scientific protocol content drift")
    bindings = prereg.get("code_bindings")
    if not isinstance(bindings, Mapping) or EXPORTER_RELATIVE not in bindings:
        raise RuntimeError("historical preregistration code closure is incomplete")
    for relative, digest in bindings.items():
        candidate = PROJECT_ROOT / str(relative)
        _assert_regular(candidate)
        if relative == EXPORTER_RELATIVE:
            continue
        if digest != file_sha256(candidate):
            raise RuntimeError(f"pre-erratum code binding drift: {relative}")
    old_exporter_sha = str(bindings[EXPORTER_RELATIVE])
    exporter_path = PROJECT_ROOT / EXPORTER_RELATIVE
    new_exporter_sha = file_sha256(exporter_path)
    if old_exporter_sha == new_exporter_sha:
        raise RuntimeError("implementation erratum exporter correction is absent")
    return {
        "preregistration": _binding(prereg_path),
        "launch_intent": _binding(intent_path),
        "failed_status": _binding(status_path),
        "preregistration_payload": prereg,
        "protocol": _binding(protocol_path),
        "protocol_payload": protocol,
        "old_exporter_sha256": old_exporter_sha,
        "new_exporter_sha256": new_exporter_sha,
    }


def _assert_exporter_weights_only() -> None:
    path = PROJECT_ROOT / EXPORTER_RELATIVE
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_load_checkpoint_safely"
    ]
    if len(functions) != 1:
        raise RuntimeError("safe checkpoint loader definition drift")
    calls = [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "load"
    ]
    if len(calls) != 1:
        raise RuntimeError("safe checkpoint loader torch.load call drift")
    weights = {
        keyword.arg: keyword.value
        for keyword in calls[0].keywords
        if keyword.arg is not None
    }.get("weights_only")
    if not isinstance(weights, ast.Constant) or weights.value is not True:
        raise RuntimeError("checkpoint loader must retain weights_only=True")


def _assert_no_recovery_outputs() -> None:
    manifests = [
        path
        for path in OLD_OUTPUT_ROOT.rglob("manifest.json")
        if "scores_heldout_train" in path.parts
    ]
    identities = list(OLD_OUTPUT_ROOT.rglob("TIER2R_EXPORT_IDENTITY.json"))
    if manifests or identities:
        raise RuntimeError(
            "implementation erratum must precede every score manifest/export identity"
        )


def _round1_bindings() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role, run_id, gpu in (
        ("control", ROUND1_RUNS[0], 2),
        ("c", ROUND1_RUNS[1], 3),
    ):
        run_dir = OLD_OUTPUT_ROOT / "seed43" / role / "heldout_nudt"
        identity_path = run_dir / "TIER2R_RUN_IDENTITY.json"
        identity = _verify_frozen_json(identity_path)
        checkpoint = run_dir / "last.pt"
        _assert_regular(checkpoint, readonly=True)
        checkpoint_sha = file_sha256(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            identity.get("run_id") != run_id
            or identity.get("checkpoint_epoch") != 79
            or identity.get("checkpoint_selection") != "fixed_last"
            or identity.get("checkpoint_sha256") != checkpoint_sha
            or identity.get("physical_gpu") != gpu
            or identity.get("source_only") is not True
            or identity.get("outer_target_images_used") is not False
            or identity.get("outer_target_labels_used") is not False
            or not isinstance(payload, Mapping)
            or payload.get("epoch") != 79
            or payload.get("checkpoint_selection") != "fixed_last"
        ):
            raise RuntimeError(f"frozen round-1 checkpoint identity drift: {run_id}")
        result[run_id] = {
            "run_identity": _binding(identity_path),
            "checkpoint": _binding(checkpoint),
            "checkpoint_epoch": 79,
            "reuse_without_retraining": True,
        }
    return result


def _recovery_code_bindings() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in RECOVERY_CODE_RELATIVES:
        path = PROJECT_ROOT / relative
        _assert_regular(path)
        result[relative] = file_sha256(path)
    return result


def implementation_erratum_plan() -> dict[str, Any]:
    return {
        "scientific_protocol_id": SCIENTIFIC_PROTOCOL_ID,
        "execution_instance": EXECUTION_INSTANCE,
        "change_scope": CHANGE_SCOPE,
        "reuse_frozen_round1_checkpoints": True,
        "training_unchanged": True,
        "model_unchanged": True,
        "config_unchanged": True,
        "data_unchanged": True,
        "decision_protocol_unchanged": True,
        "outer_target_access_authorized": False,
    }


def _failure_log_paths() -> dict[str, Path]:
    return {
        run_id: OLD_OUTPUT_ROOT
        / "seed43"
        / ("control" if "control" in run_id else "c")
        / "heldout_nudt"
        / "export.log"
        for run_id in ROUND1_RUNS
    }


def _ensure_failure_snapshots(*, create: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for run_id, live_path in _failure_log_paths().items():
        _assert_regular(live_path)
        snapshot = FAILURE_EVIDENCE_ROOT / f"{run_id}.export.log"
        if create:
            _write_once(snapshot, live_path.read_bytes())
        elif not snapshot.exists() and not _sidecar(snapshot).exists():
            result[run_id] = {
                "planned_snapshot_path": str(snapshot.resolve()),
                "live_export_log_sha256": file_sha256(live_path),
            }
            continue
        _assert_regular(snapshot, readonly=True)
        sidecar = _sidecar(snapshot)
        _assert_regular(sidecar, readonly=True)
        digest = file_sha256(snapshot)
        if sidecar.read_text(encoding="ascii") != f"{digest}  {snapshot.name}\n":
            raise RuntimeError(f"failure-evidence sidecar drift: {sidecar}")
        result[run_id] = _binding(snapshot)
    return result


def _build_payload(*, create_snapshots: bool) -> dict[str, Any]:
    chain = _old_chain()
    _assert_exporter_weights_only()
    _assert_no_recovery_outputs()
    test_path = PROJECT_ROOT / REGRESSION_TEST_RELATIVE
    _assert_regular(test_path)
    snapshots = _ensure_failure_snapshots(create=create_snapshots)
    if not create_snapshots and any(
        "planned_snapshot_path" in record for record in snapshots.values()
    ):
        snapshot_bindings = {
            run_id: {
                "path": record["planned_snapshot_path"],
                "sha256": record["live_export_log_sha256"],
            }
            for run_id, record in snapshots.items()
        }
    else:
        snapshot_bindings = snapshots
    plan = implementation_erratum_plan()
    return {
        "schema_version": SCHEMA_VERSION,
        "registered_at": _now(),
        **plan,
        "source_only": True,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "historical_failure": {
            "expected_error": EXPECTED_FAILURE,
            "status": chain["failed_status"],
            "failure_log_snapshots": snapshot_bindings,
            "no_score_manifest_existed_at_registration": True,
            "no_export_identity_existed_at_registration": True,
        },
        "inherited_immutable_chain": {
            "preregistration": chain["preregistration"],
            "launch_intent": chain["launch_intent"],
        },
        "scientific_protocol": {
            **chain["protocol"],
            "score_protocol_sha256": hashlib.sha256(
                _canonical_json_bytes(chain["protocol_payload"]["score_protocol"])
            ).hexdigest(),
            "decision_protocol_sha256": hashlib.sha256(
                _canonical_json_bytes(chain["protocol_payload"]["decision_protocol"])
            ).hexdigest(),
            "training_protocol_sha256": hashlib.sha256(
                _canonical_json_bytes(chain["protocol_payload"]["training_protocol"])
            ).hexdigest(),
        },
        "implementation_change": {
            "old_exporter_sha256": chain["old_exporter_sha256"],
            "new_exporter": _binding(PROJECT_ROOT / EXPORTER_RELATIVE),
            "numpy_1x_global": "numpy.core.multiarray.scalar",
            "numpy_2x_global": "numpy._core.multiarray.scalar",
            "checkpoint_deserialization_weights_only": True,
            "regression_test": _binding(test_path),
        },
        "recovery_code_bindings": _recovery_code_bindings(),
        "reused_round1_runs": _round1_bindings(),
    }


def _validate_existing() -> dict[str, Any]:
    payload = _verify_frozen_json(ERRATUM_PATH)
    plan = implementation_erratum_plan()
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or any(payload.get(key) != value for key, value in plan.items())
        or payload.get("source_only") is not True
        or payload.get("outer_target_images_used") is not False
        or payload.get("outer_target_labels_used") is not False
    ):
        raise RuntimeError("implementation erratum semantic drift")
    chain = _old_chain()
    inherited = payload.get("inherited_immutable_chain")
    historical = payload.get("historical_failure")
    change = payload.get("implementation_change")
    if not all(isinstance(value, Mapping) for value in (inherited, historical, change)):
        raise RuntimeError("implementation erratum sections are incomplete")
    _verify_binding(
        inherited["preregistration"],
        expected=OLD_AUDIT_ROOT / "COMPONENT_RESCUE_PREREGISTRATION.json",
        label="old preregistration",
        readonly=True,
    )
    _verify_binding(
        inherited["launch_intent"],
        expected=OLD_AUDIT_ROOT / "LAUNCH_INTENT.json",
        label="old launch intent",
        readonly=True,
    )
    _verify_binding(
        historical["status"],
        expected=OLD_AUDIT_ROOT / "COMPONENT_RESCUE_STATUS.json",
        label="failed status",
    )
    if historical.get("expected_error") != EXPECTED_FAILURE:
        raise RuntimeError("implementation erratum failure reason drift")
    snapshots = historical.get("failure_log_snapshots")
    if not isinstance(snapshots, Mapping) or set(snapshots) != set(ROUND1_RUNS):
        raise RuntimeError("implementation erratum failure snapshots drift")
    for run_id in ROUND1_RUNS:
        _verify_binding(
            snapshots[run_id],
            expected=FAILURE_EVIDENCE_ROOT / f"{run_id}.export.log",
            label=f"{run_id} failure snapshot",
            readonly=True,
        )
    _assert_exporter_weights_only()
    if (
        change.get("old_exporter_sha256") != chain["old_exporter_sha256"]
        or change.get("checkpoint_deserialization_weights_only") is not True
        or change.get("numpy_1x_global") != "numpy.core.multiarray.scalar"
        or change.get("numpy_2x_global") != "numpy._core.multiarray.scalar"
    ):
        raise RuntimeError("implementation erratum loader contract drift")
    _verify_binding(
        change["new_exporter"],
        expected=PROJECT_ROOT / EXPORTER_RELATIVE,
        label="corrected exporter",
    )
    _verify_binding(
        change["regression_test"],
        expected=PROJECT_ROOT / REGRESSION_TEST_RELATIVE,
        label="exporter regression test",
    )
    if payload.get("recovery_code_bindings") != _recovery_code_bindings():
        raise RuntimeError("implementation erratum recovery-code drift")
    expected_runs = _round1_bindings()
    if payload.get("reused_round1_runs") != expected_runs:
        raise RuntimeError("implementation erratum checkpoint binding drift")
    protocol = payload.get("scientific_protocol")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("path") != chain["protocol"]["path"]
        or protocol.get("sha256") != chain["protocol"]["sha256"]
    ):
        raise RuntimeError("implementation erratum scientific-protocol drift")
    return payload


def ensure_implementation_erratum(*, create: bool) -> dict[str, Any]:
    """Validate the amendment or create it exactly once after fail-closed checks."""

    if ERRATUM_PATH.exists() or _sidecar(ERRATUM_PATH).exists():
        return _validate_existing()
    payload = _build_payload(create_snapshots=create)
    if not create:
        return {
            "registered": False,
            "path": str(ERRATUM_PATH.resolve()),
            "planned_sha256": hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
            "plan": implementation_erratum_plan(),
        }
    _write_once(ERRATUM_PATH, _canonical_json_bytes(payload))
    return _validate_existing()


def implementation_erratum_binding() -> dict[str, str]:
    _validate_existing()
    return _binding(ERRATUM_PATH)


__all__ = [
    "CHANGE_SCOPE",
    "ERRATUM_PATH",
    "EXECUTION_INSTANCE",
    "NEW_AUDIT_RELATIVE",
    "NEW_GATE_RELATIVE",
    "OLD_AUDIT_RELATIVE",
    "OLD_OUTPUT_RELATIVE",
    "RECOVERY_CODE_RELATIVES",
    "ensure_implementation_erratum",
    "implementation_erratum_binding",
    "implementation_erratum_plan",
]
