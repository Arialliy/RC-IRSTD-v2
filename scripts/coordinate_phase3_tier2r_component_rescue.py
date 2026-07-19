#!/usr/bin/env python3
"""Coordinate the corrected Tier2R source-only component-rescue protocol.

The coordinator launches the preregistered 18 detector jobs immediately on
physical GPUs 2 and 3; it never queries or waits for GPU idleness and never
falls back to another device.  It creates a new immutable audit namespace,
keeps the historical Tier2 HOLD untouched, exports FP32 raw logits on the two
source LODO folds, and delegates the decision to the exact-state gate.
"""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO

import torch
import yaml


CANONICAL_PROJECT_ROOT = Path("/home/ly/RC-IRSTD-v2")
PROJECT_ROOT = CANONICAL_PROJECT_ROOT


def _python_executable() -> Path:
    raw = Path(sys.executable)
    if not raw.is_absolute():
        raise RuntimeError(f"Python executable must be absolute: {sys.executable}")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"Python executable cannot be resolved: {raw}") from error
    if not resolved.is_file():
        raise RuntimeError(f"Python executable is not a regular file: {resolved}")
    return resolved

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.artifact_integrity import (  # noqa: E402
    file_sha256,
    ordered_ids_sha256,
    verify_score_map_directory,
)
from rc_irstd.data import (  # noqa: E402
    ensure_unique_sample_ids,
    read_split_file,
    resolve_split_file,
)
from evaluation.threshold_sweep import validate_formal_score_manifest  # noqa: E402
from scripts import run_phase3_tier2r_exact_gate as exact_gate  # noqa: E402
from rc_irstd.models import build_mshnet  # noqa: E402


SCHEMA = "rc-irstd-aaai27-tier2r-component-rescue-coordinator-v1"
PREREGISTRATION_SCHEMA = "rc-irstd-aaai27-tier2r-component-rescue-preregistration-v1"
INITIAL_TARGET_AUTHORIZATION_SCHEMA = (
    "rc-irstd-aaai27-tier2r-initial-outer-target-authorization-v1"
)
RUN_IDENTITY_SCHEMA = "rc-irstd-aaai27-tier2r-detector-run-identity-v1"
EXPORT_IDENTITY_SCHEMA = "rc-irstd-aaai27-tier2r-raw-logit-export-identity-v1"
STATUS_SCHEMA = "rc-irstd-aaai27-tier2r-component-rescue-status-v1"

PROTOCOL_PATH = PROJECT_ROOT / exact_gate.PROTOCOL_RELATIVE
AUDIT_ROOT = PROJECT_ROOT / exact_gate.AUDIT_RELATIVE
OUTPUT_ROOT = PROJECT_ROOT / "outputs/aaai27/detectors/component_rescue/tier2r_c_v1"
STATE_ROOT = PROJECT_ROOT / "outputs/phase_state"
TARGET_HOLD_SENTINEL = STATE_ROOT / "HOLD_PHASE3_TARGET_LABEL_ACCESS"

ROLE_CONFIGS = {
    "control": "configs/tier2r_control.yaml",
    "c": "configs/tier2r_rc_mshnet_c.yaml",
    "cv": "configs/tier2r_rc_mshnet_cv_v1.yaml",
}
EXPECTED_MODEL_FLAGS = {
    "control": {
        "use_contrast": False,
        "use_component_context": False,
        "use_component_expert": False,
        "use_risk_gate": False,
        "expose_branch_auxiliary": False,
    },
    "c": {
        "use_contrast": True,
        "use_component_context": False,
        "use_component_expert": False,
        "use_risk_gate": True,
        "expose_branch_auxiliary": False,
    },
    "cv": {
        "use_contrast": True,
        "use_component_context": True,
        "use_component_expert": False,
        "use_risk_gate": True,
        "expose_branch_auxiliary": False,
    },
}
MAX_RECOVERY_ATTEMPTS = 3


@dataclass(frozen=True)
class FoldSpec:
    key: str
    training_source: str
    training_root: str
    held_out_source: str
    held_out_root: str
    initializer: str


@dataclass(frozen=True)
class JobSpec:
    seed: int
    role: str
    fold: FoldSpec
    physical_gpu: int
    round_index: int

    @property
    def run_id(self) -> str:
        return f"seed{self.seed}_{self.role}_{self.fold.key}"


@dataclass(frozen=True)
class RunInspection:
    state: str
    epoch: int | None = None
    reason: str | None = None


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _registered_at(path: Path) -> str:
    if path.is_file():
        value = _load_json(path).get("registered_at")
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"immutable artifact lacks registered_at: {path}")
        return value
    return _now()


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


def _sidecar(path: Path) -> Path:
    return path.with_suffix(".sha256")


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
            raise RuntimeError(f"immutable Tier2R artifact drift: {path}")
        return digest
    _atomic_write(path, content)
    _atomic_write(sidecar, sidecar_content)
    path.chmod(0o444)
    sidecar.chmod(0o444)
    return digest


def _write_once_json(path: Path, payload: Mapping[str, Any]) -> str:
    return _write_once(path, _canonical_json_bytes(dict(payload)))


def _write_status(status: str, **fields: Any) -> None:
    _atomic_write(
        AUDIT_ROOT / "COMPONENT_RESCUE_STATUS.json",
        _canonical_json_bytes(
            {
                "schema_version": STATUS_SCHEMA,
                "updated_at": _now(),
                "status": status,
                "source_only": True,
                "outer_target_images_used": False,
                "outer_target_labels_used": False,
                "outer_target_access_authorized": False,
                **fields,
            }
        ),
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"YAML root is not an object: {path}")
    return payload


def _folds(protocol: Mapping[str, Any]) -> dict[str, FoldSpec]:
    raw = protocol.get("folds")
    if not isinstance(raw, Mapping) or set(raw) != set(exact_gate.FOLDS):
        raise RuntimeError("Tier2R fold protocol drift")
    return {
        key: FoldSpec(key=key, **dict(raw[key]))
        for key in exact_gate.FOLDS
    }


def build_schedule(protocol: Mapping[str, Any]) -> tuple[tuple[JobSpec, ...], ...]:
    """Build preregistered seed-local rounds without idle-device polling."""

    folds = _folds(protocol)
    rounds: list[tuple[JobSpec, ...]] = []
    round_index = 0
    for seed in exact_gate.SEEDS:
        assignments = (
            (("control", "heldout_nudt"), ("c", "heldout_nudt")),
            (("cv", "heldout_nudt"), ("control", "heldout_irstd")),
            (("c", "heldout_irstd"), ("cv", "heldout_irstd")),
        )
        for pair in assignments:
            round_index += 1
            group = tuple(
                JobSpec(
                    seed,
                    role,
                    folds[fold_key],
                    exact_gate.expected_physical_gpu(seed, role, fold_key),
                    round_index,
                )
                for role, fold_key in pair
            )
            rounds.append(group)
    flattened = [job for group in rounds for job in group]
    if (
        len(flattened) != 18
        or {job.run_id for job in flattened}
        != {spec.run_id for spec in exact_gate.RUN_SPECS}
        or any({job.physical_gpu for job in group} != {2, 3} for group in rounds)
    ):
        raise AssertionError("Tier2R GPU schedule construction failed")
    return tuple(rounds)


def _strip_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    value.pop("experiment_identity", None)
    value.pop("output_dir", None)
    model = value.get("model")
    if isinstance(model, dict):
        for key in EXPECTED_MODEL_FLAGS["control"]:
            model.pop(key, None)
    return value


def _validate_base_configs(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for role, relative in ROLE_CONFIGS.items():
        path = PROJECT_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Tier2R base config is absent or a symlink: {path}")
        config = _load_yaml(path)
        model = config.get("model")
        training = config.get("training")
        data = config.get("data")
        if not all(isinstance(value, Mapping) for value in (model, training, data)):
            raise RuntimeError(f"Tier2R config sections are incomplete: {role}")
        assert isinstance(model, Mapping)
        assert isinstance(training, Mapping)
        assert isinstance(data, Mapping)
        for field, expected in EXPECTED_MODEL_FLAGS[role].items():
            if type(model.get(field)) is not bool or model[field] is not expected:
                raise RuntimeError(f"Tier2R {role}.{field} must be exactly {expected}")
        if (
            model.get("backend") != "rc_mshnet"
            or model.get("architecture_version")
            != "rc-mshnet-v2-component-role-split"
            or training.get("checkpoint_selection") != "fixed_last"
            or training.get("epochs") != 80
            or training.get("initialize_from") is not None
            or training.get("resume") is not None
            or data.get("train_split") != "train"
            or data.get("val_split") is not None
            or data.get("diagnostic_test_eval") is not False
        ):
            raise RuntimeError(f"Tier2R base config contract drift: {role}")
        configs[role] = config
    c_core = copy.deepcopy(configs["c"])
    cv_core = copy.deepcopy(configs["cv"])
    for config in (c_core, cv_core):
        identity = config.pop("experiment_identity")
        if isinstance(identity, dict):
            identity.pop("run_role", None)
            identity.pop("candidate_version", None)
        config.pop("output_dir", None)
    assert isinstance(c_core["model"], dict) and isinstance(cv_core["model"], dict)
    c_core["model"].pop("use_component_context")
    cv_core["model"].pop("use_component_context")
    if c_core != cv_core:
        raise RuntimeError("C and CV configs differ outside component-context identity")
    if _strip_identity(configs["control"]) != _strip_identity(configs["c"]):
        raise RuntimeError("control and C are not matched outside RC model flags")
    protocol_roles = protocol.get("roles")
    if not isinstance(protocol_roles, Mapping):
        raise RuntimeError("protocol role bindings are absent")
    for role, relative in ROLE_CONFIGS.items():
        if protocol_roles.get(role, {}).get("config") != relative:
            raise RuntimeError(f"protocol config binding drift: {role}")
    return configs


def _validate_target_lock(protocol: Mapping[str, Any]) -> dict[str, Path]:
    if (
        not TARGET_HOLD_SENTINEL.is_file()
        or TARGET_HOLD_SENTINEL.is_symlink()
        or TARGET_HOLD_SENTINEL.read_bytes() != b"HOLD\n"
    ):
        raise RuntimeError("outer-target HOLD sentinel is absent or drifted")
    access = protocol["data_access"]
    allowed = [PROJECT_ROOT / value for value in access["allowed_source_roots"]]
    forbidden = PROJECT_ROOT / access["forbidden_outer_target_root"]
    validated = {
        str(path): exact_gate.validate_source_path(
            path,
            allowed_roots=allowed,
            forbidden_root=forbidden,
        )
        for path in allowed
    }
    if Path(os.path.abspath(forbidden)) in [Path(os.path.abspath(path)) for path in allowed]:
        raise RuntimeError("outer target leaked into source-root allowlist")
    return validated


def _source_split_binding(root: Path) -> dict[str, Any]:
    split = resolve_split_file(root, None, split="train")
    ids = ensure_unique_sample_ids(read_split_file(split))
    return {
        "root": str(root.resolve()),
        "split": "train",
        "split_file": str(split.resolve()),
        "split_file_sha256": file_sha256(split),
        "ordered_ids_sha256": ordered_ids_sha256(ids),
        "num_samples": len(ids),
    }


def verify_prerequisites() -> dict[str, Any]:
    """Read-only verification used before registration and every launch round."""

    if PROJECT_ROOT.resolve() != CANONICAL_PROJECT_ROOT.resolve():
        raise RuntimeError("Tier2R coordinator requires the canonical project root")
    protocol, protocol_path = exact_gate._load_protocol(PROJECT_ROOT)
    historical = exact_gate.verify_historical_hold(PROJECT_ROOT, protocol)
    configs = _validate_base_configs(protocol)
    expected_model_configs = {
        role: build_mshnet(config["model"]).export_config()
        for role, config in configs.items()
    }
    source_roots = _validate_target_lock(protocol)
    folds = _folds(protocol)
    initializers: dict[str, Any] = {}
    for fold in folds.values():
        initializer = PROJECT_ROOT / fold.initializer
        if initializer.is_symlink() or not initializer.is_file():
            raise RuntimeError(f"Tier2R initializer is absent or a symlink: {initializer}")
        initializers[fold.key] = {
            "path": str(initializer.resolve()),
            "sha256": file_sha256(initializer),
        }
    split_bindings = {
        name: _source_split_binding(path) for name, path in source_roots.items()
    }
    fixed_code_paths = (
        PROJECT_ROOT / "train_detector.py",
        PROJECT_ROOT / "export_scores.py",
        PROJECT_ROOT / "rc_irstd/cli/train_detector.py",
        PROJECT_ROOT / "rc_irstd/cli/export_scores.py",
        PROJECT_ROOT / "rc_irstd/training/detector_trainer.py",
        PROJECT_ROOT / "rc_irstd/losses/detector.py",
        PROJECT_ROOT / "rc_irstd/losses/sls.py",
        PROJECT_ROOT / "evaluation/artifact_integrity.py",
        PROJECT_ROOT / "evaluation/export_score_maps.py",
        PROJECT_ROOT / "evaluation/raw_logit_oracle.py",
        PROJECT_ROOT / "evaluation/raw_logit_source_operating_point.py",
        PROJECT_ROOT / "evaluation/threshold_sweep.py",
        PROJECT_ROOT / "evaluation/component_matching.py",
        PROJECT_ROOT / "scripts/run_phase3_raw_logit_rescue_v1.py",
        PROJECT_ROOT / "scripts/run_phase3_tier2_raw_logit_gate.py",
        PROJECT_ROOT / "scripts/run_phase3_tier2r_exact_gate.py",
        Path(__file__).resolve(),
    )
    code_directories = tuple(
        PROJECT_ROOT / relative
        for relative in ("model", "rc_irstd/models", "rc_irstd/data", "data_ext")
    )
    code_paths = tuple(
        dict.fromkeys(
            (*fixed_code_paths, *(
                path
                for directory in code_directories
                for path in sorted(directory.glob("*.py"))
            ))
        )
    )
    code_bindings = {}
    for path in code_paths:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Tier2R code binding is absent or a symlink: {path}")
        code_bindings[str(path.relative_to(PROJECT_ROOT))] = file_sha256(path)
    return {
        "schema_version": SCHEMA,
        "verified": True,
        "source_only": True,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "outer_target_access_authorized": False,
        "protocol": {"path": str(protocol_path), "sha256": file_sha256(protocol_path)},
        "historical_tier2_hold": historical,
        "base_configs": {
            role: {
                "path": str((PROJECT_ROOT / ROLE_CONFIGS[role]).resolve()),
                "sha256": file_sha256(PROJECT_ROOT / ROLE_CONFIGS[role]),
            }
            for role in exact_gate.ROLES
        },
        "initializers": initializers,
        "expected_model_configs": expected_model_configs,
        "source_splits": split_bindings,
        "code_bindings": code_bindings,
        "schedule": [
            {
                "round": job.round_index,
                "run_id": job.run_id,
                "seed": job.seed,
                "role": job.role,
                "fold": job.fold.key,
                "physical_gpu": job.physical_gpu,
                "logical_device": "cuda:0",
            }
            for group in build_schedule(protocol)
            for job in group
        ],
        "protocol_payload": protocol,
        "validated_configs": configs,
    }


def _initial_target_authorization_payload(
    packet: Mapping[str, Any], path: Path
) -> dict[str, Any]:
    return {
        "schema_version": INITIAL_TARGET_AUTHORIZATION_SCHEMA,
        "registered_at": _registered_at(path),
        "decision": "OUTER_TARGET_ACCESS_LOCKED_BEFORE_TIER2R",
        "protocol_sha256": packet["protocol"]["sha256"],
        "historical_tier2_hold_decision_sha256": packet["historical_tier2_hold"][
            "decision"
        ]["sha256"],
        "outer_target_access_authorized": False,
        "outer_target_image_access_authorized": False,
        "outer_target_label_access_authorized": False,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
    }


def _preregistration_payload(
    packet: Mapping[str, Any], target_authorization_path: Path, path: Path
) -> dict[str, Any]:
    protocol = packet["protocol_payload"]
    return {
        "schema_version": PREREGISTRATION_SCHEMA,
        "registered_at": _registered_at(path),
        "protocol_id": protocol["protocol_id"],
        "candidate_frozen_name": protocol["candidate_frozen_name"],
        "source_only": True,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "outer_target_access_authorized": False,
        "protocol": packet["protocol"],
        "historical_tier2_hold": packet["historical_tier2_hold"],
        "initial_outer_target_authorization": {
            "path": str(target_authorization_path.resolve()),
            "sha256": file_sha256(target_authorization_path),
        },
        "base_configs": packet["base_configs"],
        "initializers": packet["initializers"],
        "expected_model_configs": packet["expected_model_configs"],
        "source_splits": packet["source_splits"],
        "code_bindings": packet["code_bindings"],
        "seeds": list(exact_gate.SEEDS),
        "roles": list(exact_gate.ROLES),
        "folds": list(exact_gate.FOLDS),
        "schedule": packet["schedule"],
        "score_protocol": protocol["score_protocol"],
        "decision_protocol": protocol["decision_protocol"],
        "training_protocol": protocol["training_protocol"],
        "dense_grid_diagnostic": {
            "used_for_formal_decision": False,
            "formal_gate_input": False,
            "protocol_sha256": hashlib.sha256(
                _canonical_json_bytes(
                    {
                        "representation": "float32_raw_logit",
                        "status": "diagnostic_only",
                    }
                )
            ).hexdigest(),
        },
        "failure_policy": "missing_or_drifted_artifact_fails_closed_to_TIER2R_HOLD",
    }


def _verify_frozen_registration(
    packet: Mapping[str, Any], target_auth_path: Path, preregistration_path: Path
) -> str:
    _write_once_json(
        target_auth_path,
        _initial_target_authorization_payload(packet, target_auth_path),
    )
    return _write_once_json(
        preregistration_path,
        _preregistration_payload(packet, target_auth_path, preregistration_path),
    )


def _run_dir(job: JobSpec) -> Path:
    return OUTPUT_ROOT / f"seed{job.seed}" / job.role / job.fold.key


def _expected_formal_config(job: JobSpec, packet: Mapping[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(packet["validated_configs"][job.role])
    training_root = PROJECT_ROOT / job.fold.training_root
    initializer = PROJECT_ROOT / job.fold.initializer
    config["seed"] = job.seed
    config["device"] = "cuda:0"
    config["output_dir"] = str(_run_dir(job).resolve())
    config["data"]["sources"] = [
        {"name": job.fold.training_source, "path": str(training_root.resolve())}
    ]
    config["training"]["initialize_from"] = str(initializer.resolve())
    config["training"]["resume"] = None
    config["experiment_identity"].update(
        {
            "schema_version": "rc-irstd-aaai27-tier2r-component-rescue-v1",
            "stage": "tier2r_c_source_only_confirmation",
            "run_role": job.run_id,
            "outer_target": job.fold.held_out_source,
            "target_labels_used_for_training": False,
        }
    )
    config["tier2r_runtime_contract"] = {
        "protocol_id": "tier2r_c_v1",
        "seed": job.seed,
        "role": job.role,
        "fold": job.fold.key,
        "training_source": job.fold.training_source,
        "held_out_source_pseudo_target": job.fold.held_out_source,
        "physical_gpu": job.physical_gpu,
        "cuda_visible_devices": str(job.physical_gpu),
        "logical_device": "cuda:0",
        "wait_for_idle_gpu": False,
        "allow_gpu_fallback": False,
        "outer_target_dataset_loaded": False,
    }
    return config


def _ensure_formal_config(job: JobSpec, packet: Mapping[str, Any]) -> Path:
    path = _run_dir(job) / "formal_config.yaml"
    expected = _expected_formal_config(job, packet)
    content = yaml.safe_dump(expected, sort_keys=False, allow_unicode=True).encode("utf-8")
    _write_once(path, content)
    return path


def _history_epochs(path: Path) -> list[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [int(row["epoch"]) for row in rows]


def _checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint root is not an object: {path}")
    return payload


def inspect_run(job: JobSpec, packet: Mapping[str, Any]) -> RunInspection:
    run_dir = _run_dir(job)
    formal = run_dir / "formal_config.yaml"
    if not formal.exists():
        return RunInspection("fresh" if not run_dir.exists() else "corrupt", reason="formal config absent")
    if _load_yaml(formal) != _expected_formal_config(job, packet):
        return RunInspection("corrupt", reason="formal config drift")
    checkpoint = run_dir / "last.pt"
    history = run_dir / "history.csv"
    if not checkpoint.exists() and not history.exists():
        allowed = {"formal_config.yaml", "formal_config.sha256"}
        extras = {path.name for path in run_dir.iterdir()} - allowed
        return RunInspection("fresh" if not extras else "zero_epoch_partial")
    if history.exists() and not checkpoint.exists() and not history.is_symlink():
        try:
            return RunInspection(
                "zero_epoch_partial" if _history_epochs(history) == [0] else "corrupt",
                reason="history exists before first committed checkpoint",
            )
        except Exception as error:
            return RunInspection("corrupt", reason=str(error))
    if not history.exists() or checkpoint.is_symlink() or history.is_symlink():
        return RunInspection("corrupt", reason="history/checkpoint pair incomplete")
    try:
        payload = _checkpoint_payload(checkpoint)
        epochs = _history_epochs(history)
        epoch = int(payload.get("epoch", -1))
        initialization = payload.get("initialization")
        expected_config = _expected_formal_config(job, packet)
        model_config = payload.get("model_config")
        extension_sha = (
            initialization.get("initial_extension_state_sha256")
            if isinstance(initialization, Mapping) else None
        )
        if (
            payload.get("config") != expected_config
            or not isinstance(model_config, Mapping)
            or dict(model_config) != packet["expected_model_configs"][job.role]
            or model_config.get("architecture_version")
            != "rc-mshnet-v2-component-role-split"
            or any(
                type(model_config.get(field)) is not bool
                or model_config.get(field) is not expected
                for field, expected in EXPECTED_MODEL_FLAGS[job.role].items()
            )
            or not isinstance(extension_sha, str)
            or len(extension_sha) != 64
            or any(character not in "0123456789abcdef" for character in extension_sha)
            or initialization.get("extension_state_preserved") is not True
            or payload.get("checkpoint_selection") != "fixed_last"
            or payload.get("selection_rule") != "fixed_last"
            or payload.get("test_labels_used_for_selection") is not False
            or payload.get("diagnostic_test_eval") is not False
            or not isinstance(initialization, Mapping)
            or initialization.get("source_sha256")
            != packet["initializers"][job.fold.key]["sha256"]
            or epochs not in (list(range(epoch + 1)), list(range(epoch + 2)))
        ):
            return RunInspection("corrupt", epoch=epoch, reason="checkpoint identity drift")
    except Exception as error:
        return RunInspection("corrupt", reason=str(error))
    if epochs == list(range(epoch + 2)):
        return RunInspection("history_ahead_one", epoch=epoch)
    if epoch == 79:
        return RunInspection("complete", epoch=epoch)
    if 0 <= epoch < 79:
        return RunInspection("resume", epoch=epoch)
    return RunInspection("corrupt", epoch=epoch, reason="checkpoint epoch outside 0..79")


def _isolate_zero_epoch_partial(job: JobSpec, packet: Mapping[str, Any]) -> Path:
    if inspect_run(job, packet).state != "zero_epoch_partial":
        raise RuntimeError(f"refusing to isolate non-zero-epoch run: {job.run_id}")
    run_dir = _run_dir(job)
    archive_root = run_dir.parent / ".failed_zero_epoch_attempts"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = archive_root / f"{job.fold.key}_{time.time_ns()}"
    if archive.exists():
        raise RuntimeError(f"zero-epoch archive collision: {archive}")
    os.replace(run_dir, archive)
    return archive


def _repair_history_ahead_one(job: JobSpec, packet: Mapping[str, Any]) -> Path:
    inspection = inspect_run(job, packet)
    if inspection.state != "history_ahead_one" or inspection.epoch is None:
        raise RuntimeError(f"refusing unproved history repair: {job.run_id}")
    run_dir = _run_dir(job)
    history = run_dir / "history.csv"
    lines = history.read_bytes().splitlines(keepends=True)
    expected_lines = inspection.epoch + 3
    if len(lines) != expected_lines:
        raise RuntimeError(f"history repair line-count drift: {job.run_id}")
    archive_root = run_dir / ".history_repair_archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = archive_root / f"history_ahead_epoch{inspection.epoch}_{time.time_ns()}.csv"
    os.replace(history, archive)
    _atomic_write(history, b"".join(lines[: inspection.epoch + 2]))
    if inspect_run(job, packet).state != "resume":
        raise RuntimeError(f"history repair did not yield resumable state: {job.run_id}")
    return archive


def _isolate_stale_checkpoint_temp(
    job: JobSpec, packet: Mapping[str, Any]
) -> Path | None:
    run_dir = _run_dir(job)
    temporary = run_dir / "last.pt.tmp"
    if not temporary.exists() and not temporary.is_symlink():
        return None
    if temporary.is_symlink() or not temporary.is_file():
        raise RuntimeError(f"stale checkpoint temp is not a regular file: {temporary}")
    inspection = inspect_run(job, packet)
    if inspection.state not in {"resume", "complete"} or inspection.epoch is None:
        raise RuntimeError(
            f"refusing stale checkpoint-temp recovery without proved identity: {job.run_id}"
        )
    archive_root = run_dir / ".stale_checkpoint_temp_archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = archive_root / (
        f"last_epoch{inspection.epoch}_{time.time_ns()}.pt.tmp"
    )
    if archive.exists():
        raise RuntimeError(f"stale checkpoint-temp archive collision: {archive}")
    os.replace(temporary, archive)
    if inspect_run(job, packet).state != inspection.state:
        raise RuntimeError(f"stale checkpoint-temp isolation changed run state: {job.run_id}")
    return archive


def _training_command(job: JobSpec, packet: Mapping[str, Any], attempt: int) -> list[str]:
    config = _ensure_formal_config(job, packet)
    inspection = inspect_run(job, packet)
    if inspection.state == "zero_epoch_partial":
        _isolate_zero_epoch_partial(job, packet)
        config = _ensure_formal_config(job, packet)
        inspection = inspect_run(job, packet)
    if inspection.state == "history_ahead_one":
        _repair_history_ahead_one(job, packet)
        inspection = inspect_run(job, packet)
    if (_run_dir(job) / "last.pt.tmp").exists() or (_run_dir(job) / "last.pt.tmp").is_symlink():
        _isolate_stale_checkpoint_temp(job, packet)
        inspection = inspect_run(job, packet)
    command = [
        str(_python_executable()),
        str(PROJECT_ROOT / "train_detector.py"),
        "--config",
        str(config.resolve()),
        "--pid-file",
        str((_run_dir(job) / "TRAINER.pid").resolve()),
    ]
    if inspection.state == "resume":
        command.extend(
            [
                "--resume-checkpoint",
                str((_run_dir(job) / "last.pt").resolve()),
                "--resume-audit",
                str((_run_dir(job) / f"RESUME_AUDIT_{attempt}.json").resolve()),
            ]
        )
    elif inspection.state != "fresh":
        raise RuntimeError(f"cannot launch {job.run_id}: {inspection}")
    return command


def _child_environment(job: JobSpec) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(job.physical_gpu)
    environment["RC_IRSTD_TIER2R_SOURCE_ONLY"] = "1"
    environment["RC_IRSTD_ALLOWED_DATA_ROOTS"] = os.pathsep.join(
        str((PROJECT_ROOT / path).resolve())
        for path in ("datasets/NUDT-SIRST", "datasets/IRSTD-1K")
    )
    environment["RC_IRSTD_FORBIDDEN_DATA_ROOT"] = str(PROJECT_ROOT / "datasets/NUAA-SIRST")
    return environment


def _run_training_round(
    jobs: Sequence[JobSpec], packet: Mapping[str, Any]
) -> None:
    pending = list(jobs)
    attempts = {job.run_id: 0 for job in jobs}
    while pending:
        processes: list[tuple[JobSpec, subprocess.Popen[bytes], BinaryIO]] = []
        for job in pending:
            inspection = inspect_run(job, packet)
            if inspection.state == "complete":
                continue
            attempts[job.run_id] += 1
            if attempts[job.run_id] > MAX_RECOVERY_ATTEMPTS + 1:
                raise RuntimeError(f"Tier2R recovery limit exceeded: {job.run_id}")
            command = _training_command(job, packet, attempts[job.run_id])
            log_path = _run_dir(job) / f"training_attempt_{attempts[job.run_id]}.log"
            log_handle = log_path.open("ab")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=_child_environment(job),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((job, process, log_handle))
        if not processes:
            return
        failures: list[str] = []
        for job, process, log_handle in processes:
            returncode = process.wait()
            log_handle.flush()
            os.fsync(log_handle.fileno())
            log_handle.close()
            inspection = inspect_run(job, packet)
            if inspection.state == "zero_epoch_partial":
                _isolate_zero_epoch_partial(job, packet)
                continue
            if inspection.state == "history_ahead_one":
                _repair_history_ahead_one(job, packet)
                inspection = inspect_run(job, packet)
            if returncode != 0 and inspection.state not in {"resume", "complete"}:
                failures.append(f"{job.run_id}:exit={returncode}:{inspection.reason}")
            elif inspection.state not in {"resume", "complete"}:
                failures.append(f"{job.run_id}:incomplete:{inspection.reason}")
        if failures:
            raise RuntimeError("Tier2R training failed closed: " + "; ".join(failures))
        pending = [job for job in pending if inspect_run(job, packet).state != "complete"]


def _freeze_run_identity(job: JobSpec, packet: Mapping[str, Any]) -> dict[str, Any]:
    inspection = inspect_run(job, packet)
    if inspection.state != "complete":
        raise RuntimeError(f"cannot freeze incomplete Tier2R run: {job.run_id}")
    run_dir = _run_dir(job)
    checkpoint = run_dir / "last.pt"
    payload = _checkpoint_payload(checkpoint)
    identity = {
        "schema_version": RUN_IDENTITY_SCHEMA,
        "run_id": job.run_id,
        "seed": job.seed,
        "role": job.role,
        "fold": job.fold.key,
        "training_source": job.fold.training_source,
        "held_out_source": job.fold.held_out_source,
        "training_root": str((PROJECT_ROOT / job.fold.training_root).resolve()),
        "held_out_root": str((PROJECT_ROOT / job.fold.held_out_root).resolve()),
        "source_only": True,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "checkpoint_selection": "fixed_last",
        "checkpoint_epoch": 79,
        "checkpoint_sha256": file_sha256(checkpoint),
        "formal_config_sha256": file_sha256(run_dir / "formal_config.yaml"),
        "initializer_sha256": payload["initialization"]["source_sha256"],
        "initial_extension_state_sha256": payload["initialization"]["initial_extension_state_sha256"],
        "physical_gpu": job.physical_gpu,
        "logical_device": "cuda:0",
    }
    _write_once_json(run_dir / "TIER2R_RUN_IDENTITY.json", identity)
    checkpoint.chmod(0o444)
    (run_dir / "history.csv").chmod(0o444)
    return identity


def _export_command(job: JobSpec) -> list[str]:
    return [
        str(_python_executable()),
        str(PROJECT_ROOT / "export_scores.py"),
        "--checkpoint",
        str((_run_dir(job) / "last.pt").resolve()),
        "--dataset-dir",
        str((PROJECT_ROOT / job.fold.held_out_root).resolve()),
        "--dataset-name",
        job.fold.held_out_source,
        "--source-dataset",
        job.fold.training_source,
        "--split",
        "train",
        "--output-dir",
        str((_run_dir(job) / "scores_heldout_train").resolve()),
        "--device",
        "cuda:0",
        "--pad-multiple",
        "16",
        "--batch-size",
        "1",
        "--labels-loaded",
        "--export-raw-logits",
    ]


def _export_manifest_valid(job: JobSpec, manifest_path: Path) -> bool:
    try:
        payload, _, integrity = verify_score_map_directory(
            manifest_path.parent, require_integrity=True, require_masks=True
        )
        contract = validate_formal_score_manifest(
            payload, integrity, expected_split_role="train"
        )
        if payload is None:
            return False
    except Exception:
        return False
    checkpoint = _run_dir(job) / "last.pt"
    return (
        payload.get("labels_loaded") is True
        and payload.get("split_role") == "train"
        and payload.get("requested_split") == "train"
        and payload.get("target_dataset") == job.fold.held_out_source
        and payload.get("source_datasets") == [job.fold.training_source]
        and payload.get("spatial_mode") == "native"
        and payload.get("score_representation")
        == "raw_logit_float32+sigmoid_probability_float32"
        and payload.get("probability_dtype") == "float32"
        and payload.get("logit_dtype") == "float32"
        and payload.get("inference_autocast_enabled") is False
        and payload.get("checkpoint_warm_flag") is True
        and payload.get("weight_sha256") == file_sha256(checkpoint)
        and contract.get("detector_weight_sha256") == file_sha256(checkpoint)
    )


def _isolate_partial_export(job: JobSpec) -> Path:
    score_dir = _run_dir(job) / "scores_heldout_train"
    archive_root = _run_dir(job) / ".failed_export_attempts"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = archive_root / f"attempt_{time.time_ns()}"
    if archive.exists() or not score_dir.exists():
        raise RuntimeError(f"cannot isolate Tier2R export: {score_dir}")
    os.replace(score_dir, archive)
    return archive


def _ensure_exports(jobs: Sequence[JobSpec], packet: Mapping[str, Any]) -> None:
    processes: list[tuple[JobSpec, subprocess.Popen[bytes], BinaryIO]] = []
    for job in jobs:
        score_dir = _run_dir(job) / "scores_heldout_train"
        manifest = score_dir / "manifest.json"
        if manifest.is_file() and _export_manifest_valid(job, manifest):
            continue
        if score_dir.exists():
            _isolate_partial_export(job)
        command = _export_command(job)
        log_handle = (_run_dir(job) / "export.log").open("ab")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=_child_environment(job),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append((job, process, log_handle))
    failures = []
    for job, process, log_handle in processes:
        returncode = process.wait()
        log_handle.flush()
        os.fsync(log_handle.fileno())
        log_handle.close()
        if returncode != 0:
            failures.append(f"{job.run_id}:exit={returncode}")
    if failures:
        raise RuntimeError("Tier2R export failed closed: " + "; ".join(failures))
    for job in jobs:
        run_dir = _run_dir(job)
        manifest_path = run_dir / "scores_heldout_train/manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise RuntimeError(f"Tier2R export manifest is absent: {job.run_id}")
        manifest = _load_json(manifest_path)
        checkpoint = run_dir / "last.pt"
        if not _export_manifest_valid(job, manifest_path):
            raise RuntimeError(f"Tier2R export manifest contract drift: {job.run_id}")
        identity = {
            "schema_version": EXPORT_IDENTITY_SCHEMA,
            "run_id": job.run_id,
            "source_only": True,
            "outer_target_images_used": False,
            "outer_target_labels_used": False,
            "checkpoint_sha256": file_sha256(checkpoint),
            "score_manifest_sha256": file_sha256(manifest_path),
            "score_records_sha256": manifest.get("records_sha256"),
            "score_ordered_image_ids_sha256": manifest.get("ordered_image_ids_sha256"),
            "raw_logit_stream_sha256": manifest.get("raw_logit_stream_sha256"),
            "score_representation": manifest.get("score_representation"),
            "logit_dtype": manifest.get("logit_dtype"),
            "inference_autocast_enabled": manifest.get("inference_autocast_enabled"),
        }
        _write_once_json(run_dir / "TIER2R_EXPORT_IDENTITY.json", identity)
        manifest_path.chmod(0o444)


def _handoff_payload(
    packet: Mapping[str, Any],
    preregistration_path: Path,
    target_auth_path: Path,
    path: Path,
) -> dict[str, Any]:
    protocol = packet["protocol_payload"]
    runs: dict[str, Any] = {}
    for group in build_schedule(protocol):
        for job in group:
            run_dir = _run_dir(job)
            identity_path = run_dir / "TIER2R_RUN_IDENTITY.json"
            export_identity_path = run_dir / "TIER2R_EXPORT_IDENTITY.json"
            identity = _load_json(identity_path)
            export_identity = _load_json(export_identity_path)
            manifest_path = run_dir / "scores_heldout_train/manifest.json"
            runs[job.run_id] = {
                "run_id": job.run_id,
                "seed": job.seed,
                "role": job.role,
                "fold": job.fold.key,
                "training_source": job.fold.training_source,
                "held_out_source": job.fold.held_out_source,
                "training_root": str((PROJECT_ROOT / job.fold.training_root).resolve()),
                "held_out_root": str((PROJECT_ROOT / job.fold.held_out_root).resolve()),
                "source_only": True,
                "outer_target_images_used": False,
                "outer_target_labels_used": False,
                "physical_gpu": job.physical_gpu,
                "logical_device": "cuda:0",
                "checkpoint_selection": "fixed_last",
                "checkpoint_epoch": 79,
                "initializer_sha256": identity["initializer_sha256"],
                "formal_config_sha256": identity["formal_config_sha256"],
                "initial_extension_state_sha256": identity["initial_extension_state_sha256"],
                "checkpoint": str((run_dir / "last.pt").resolve()),
                "checkpoint_sha256": identity["checkpoint_sha256"],
                "run_identity": str(identity_path.resolve()),
                "run_identity_sha256": file_sha256(identity_path),
                "export_identity": str(export_identity_path.resolve()),
                "export_identity_sha256": file_sha256(export_identity_path),
                "score_dir": str((run_dir / "scores_heldout_train").resolve()),
                "score_manifest": str(manifest_path.resolve()),
                "score_manifest_sha256": export_identity["score_manifest_sha256"],
            }
    return {
        "schema_version": exact_gate.HANDOFF_SCHEMA,
        "registered_at": _registered_at(path),
        "source_only": True,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "outer_target_access_authorized": False,
        "protocol": packet["protocol"],
        "preregistration": {
            "path": str(preregistration_path.resolve()),
            "sha256": file_sha256(preregistration_path),
        },
        "initial_outer_target_authorization": {
            "path": str(target_auth_path.resolve()),
            "sha256": file_sha256(target_auth_path),
        },
        "historical_tier2_hold": packet["historical_tier2_hold"],
        "score_protocol": protocol["score_protocol"],
        "decision_protocol": protocol["decision_protocol"],
        "runs": runs,
    }


def _run_exact_gate(handoff: Path) -> dict[str, Any]:
    output = PROJECT_ROOT / exact_gate.GATE_RELATIVE
    command = [
        str(_python_executable()),
        str(PROJECT_ROOT / "scripts/run_phase3_tier2r_exact_gate.py"),
        "--handoff",
        str(handoff.resolve()),
        "--output-root",
        str(output.resolve()),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    log_path = AUDIT_ROOT / "tier2r_exact_gate.log"
    with log_path.open("ab") as log_handle:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_handle.flush()
        os.fsync(log_handle.fileno())
    if completed.returncode != 0:
        raise RuntimeError(f"Tier2R exact gate failed with exit code {completed.returncode}")
    return exact_gate.verify_frozen_gate(
        handoff_path=handoff,
        output_root=output,
        project_root=PROJECT_ROOT,
    )


def _exclusive_lock(path: Path) -> BinaryIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError(f"Tier2R coordinator lock must not be a symlink: {path}")
    handle = path.open("a+b", buffering=0)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError("another Tier2R coordinator owns the lock") from error
    return handle


def run(*, verify_only: bool = False) -> dict[str, Any]:
    """Verify the protocol or execute its recoverable source-only schedule."""

    _python_executable()
    packet = verify_prerequisites()
    if verify_only:
        return {
            key: value
            for key, value in packet.items()
            if key not in {"protocol_payload", "validated_configs"}
        }
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    lock = _exclusive_lock(AUDIT_ROOT / ".tier2r_component_rescue.lock")
    try:
        target_auth_path = AUDIT_ROOT / exact_gate.INITIAL_TARGET_AUTHORIZATION_NAME
        preregistration_path = AUDIT_ROOT / exact_gate.PREREGISTRATION_NAME
        preregistration_sha = _verify_frozen_registration(
            packet, target_auth_path, preregistration_path
        )
        _write_status(
            "tier2r_registered",
            preregistration_sha256=preregistration_sha,
            total_detector_jobs=18,
            physical_gpus=[2, 3],
            wait_for_idle_gpu=False,
        )
        schedule = build_schedule(packet["protocol_payload"])
        for round_index, jobs in enumerate(schedule, 1):
            packet = verify_prerequisites()
            _verify_frozen_registration(packet, target_auth_path, preregistration_path)
            _write_status(
                "tier2r_training",
                current_round=round_index,
                total_rounds=len(schedule),
                run_ids=[job.run_id for job in jobs],
                physical_gpus=[job.physical_gpu for job in jobs],
                wait_for_idle_gpu=False,
            )
            for job in jobs:
                _ensure_formal_config(job, packet)
            _run_training_round(jobs, packet)
            packet = verify_prerequisites()
            _verify_frozen_registration(packet, target_auth_path, preregistration_path)
            for job in jobs:
                _freeze_run_identity(job, packet)
            _write_status(
                "tier2r_exporting",
                current_round=round_index,
                total_rounds=len(schedule),
                run_ids=[job.run_id for job in jobs],
            )
            _ensure_exports(jobs, packet)
        packet = verify_prerequisites()
        _verify_frozen_registration(packet, target_auth_path, preregistration_path)
        handoff = AUDIT_ROOT / exact_gate.HANDOFF_NAME
        handoff_sha = _write_once_json(
            handoff,
            _handoff_payload(packet, preregistration_path, target_auth_path, handoff),
        )
        _write_status("tier2r_exact_gate_running", handoff_sha256=handoff_sha)
        result = _run_exact_gate(handoff)
        final = {
            "schema_version": SCHEMA,
            "status": "tier2r_exact_gate_completed",
            "source_only": True,
            "outer_target_images_used": False,
            "outer_target_labels_used": False,
            "outer_target_access_authorized": False,
            "handoff": str(handoff.resolve()),
            "handoff_sha256": handoff_sha,
            "gate_result": result,
        }
        _write_status(final["status"], handoff_sha256=handoff_sha, gate_result=result)
        return final
    except BaseException as error:
        try:
            _write_status(
                "failed_closed",
                error_type=type(error).__name__,
                error=str(error),
            )
        except Exception:
            pass
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(verify_only=args.verify_only)
    except BaseException as error:
        print(f"FAILED_CLOSED {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "JobSpec",
    "RunInspection",
    "build_schedule",
    "inspect_run",
    "run",
    "verify_prerequisites",
]
