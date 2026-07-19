#!/usr/bin/env python3
"""Recoverable, source-only Phase-3 detector gate for the AAAI-27 sprint.

The coordinator deliberately stops at a frozen Tier-1 GO/HOLD decision.  It
never opens an NUAA mask, never invokes ``run_pipeline.py``, and never trains
an outer detector.  Four inner-LODO jobs compare the matched fine-tune control
and full RC-MSHNet on the two *source official-train* pseudo targets.  Every
child sees exactly one physical GPU through ``CUDA_VISIBLE_DEVICES`` and uses
logical ``cuda:0``.

Committed detector epochs are resumed through the repository's runtime-only
formal resume path.  A zero-epoch or inconsistent half product fails closed
in place; no incomplete history/checkpoint evidence is moved or rewritten.
"""

from __future__ import annotations

import copy
import csv
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml


PROJECT_ROOT = Path("/home/ly/RC-IRSTD-v2")
EXPECTED_PYTHON = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.artifact_integrity import (  # noqa: E402
    PROBABILITY_DTYPE,
    RAW_LOGIT_DTYPE,
    RAW_LOGIT_SCORE_REPRESENTATION,
    file_sha256,
    ordered_ids_sha256,
    verify_score_map_directory,
)
from evaluation.source_operating_point import (  # noqa: E402
    load_formal_curve,
)
from evaluation.threshold_sweep import (  # noqa: E402
    build_default_thresholds,
    domain_key,
    evaluation_threshold_grid_sha256,
    validate_formal_score_manifest,
)
from rc_irstd.data import (  # noqa: E402
    ensure_unique_sample_ids,
    read_split_file,
    resolve_split_file,
)
from rc_irstd.models import build_mshnet  # noqa: E402
from rc_irstd.utils.io import atomic_write_json  # noqa: E402


PHASE3_SCHEMA = "rc-irstd-aaai27-phase3-source-lodo-v1"
PREREG_SCHEMA = "rc-irstd-aaai27-phase3-source-lodo-prereg-v1"
RUN_IDENTITY_SCHEMA = "rc-irstd-aaai27-phase3-inner-lodo-identity-v1"
DECISION_SCHEMA = "rc-irstd-aaai27-phase3-source-lodo-tier1-decision-v1"
LAUNCH_SCHEMA = "rc-irstd-aaai27-phase3-source-lodo-launch-v1"
FAILURE_SCHEMA = "rc-irstd-aaai27-phase3-source-lodo-failure-v1"

BUDGETS: tuple[tuple[str, float, float], ...] = (
    ("strict", 1.0e-6, 1.0),
    ("medium", 5.0e-6, 5.0),
    ("loose", 1.0e-5, 10.0),
)
STRICT_MACRO_PD_GAIN = 0.01
NUMERIC_ATOL = 1.0e-12
MAX_RECOVERY_ATTEMPTS = 3
TRAINING_NO_COMMIT_TIMEOUT_SECONDS = 20.0 * 60.0
TRAINING_TERMINATION_GRACE_SECONDS = 30.0

EXPECTED_FLAGS: dict[str, tuple[bool, bool, bool, bool]] = {
    "control": (False, False, False, False),
    "full": (True, True, True, False),
    "no_contrast": (False, True, True, False),
    "no_component": (True, False, True, False),
    "no_gate": (True, True, False, False),
    "branch_aux": (True, True, True, True),
}

BASE_CONFIGS: dict[str, str] = {
    "control": "phase2_mshnet_ft_outer_nuaa.yaml",
    "full": "phase2_rc_mshnet_full_outer_nuaa.yaml",
    "no_contrast": "phase2_rc_mshnet_no_contrast_outer_nuaa.yaml",
    "no_component": "phase2_rc_mshnet_no_component_outer_nuaa.yaml",
    "no_gate": "phase2_rc_mshnet_no_gate_outer_nuaa.yaml",
    "branch_aux": "phase2_rc_mshnet_branch_aux_outer_nuaa.yaml",
}

PHASE2_OUTPUTS: dict[str, str] = {
    "mshnet_ft_matched_control": (
        "outputs/aaai27/detectors/nuaa/gate_round1/"
        "mshnet_ft_matched_control"
    ),
    "rc_mshnet_full": (
        "outputs/aaai27/detectors/nuaa/gate_round1/rc_mshnet_full"
    ),
    "rc_mshnet_no_contrast": (
        "outputs/aaai27/detectors/nuaa/gate_round1/rc_mshnet_no_contrast"
    ),
    "rc_mshnet_no_component": (
        "outputs/aaai27/detectors/nuaa/gate_round2/rc_mshnet_no_component"
    ),
    "rc_mshnet_no_gate_fixed_average": (
        "outputs/aaai27/detectors/nuaa/gate_round2/"
        "rc_mshnet_no_gate_fixed_average"
    ),
    "rc_mshnet_branch_aux": (
        "outputs/aaai27/detectors/nuaa/gate_round2/rc_mshnet_branch_aux"
    ),
}


@dataclass(frozen=True)
class Layout:
    root: Path = PROJECT_ROOT

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def initializers(self) -> Path:
        return self.root / "artifacts/aaai27/initializers"

    @property
    def phase2_status(self) -> Path:
        return self.root / "artifacts/aaai27/audit/phase2_status.json"

    @property
    def audit(self) -> Path:
        return self.root / "artifacts/aaai27/audit/phase3_source_lodo_gate"

    @property
    def output(self) -> Path:
        return self.root / "outputs/aaai27/detectors/source_lodo_gate/seed42"

    @property
    def state(self) -> Path:
        return self.root / "outputs/phase_state"


@dataclass(frozen=True)
class FoldSpec:
    key: str
    held_out_name: str
    train_name: str
    train_dataset_dir: str
    held_out_dataset_dir: str
    initializer_name: str


FOLDS: dict[str, FoldSpec] = {
    "heldout_nudt": FoldSpec(
        key="heldout_nudt",
        held_out_name="NUDT-SIRST",
        train_name="IRSTD-1K",
        train_dataset_dir="IRSTD-1K",
        held_out_dataset_dir="NUDT-SIRST",
        initializer_name="mshnet_seed42_train_irstd_tensor_only.pt",
    ),
    "heldout_irstd": FoldSpec(
        key="heldout_irstd",
        held_out_name="IRSTD-1K",
        train_name="NUDT-SIRST",
        train_dataset_dir="NUDT-SIRST",
        held_out_dataset_dir="IRSTD-1K",
        initializer_name="mshnet_seed42_train_nudt_tensor_only.pt",
    ),
}


@dataclass(frozen=True)
class RunSpec:
    role: str
    fold_key: str
    physical_gpu: int

    @property
    def fold(self) -> FoldSpec:
        return FOLDS[self.fold_key]

    @property
    def run_id(self) -> str:
        return f"{self.role}_{self.fold_key}"


TIER1_ROUNDS: tuple[tuple[RunSpec, ...], ...] = (
    (
        RunSpec("control", "heldout_nudt", 2),
        RunSpec("full", "heldout_nudt", 3),
    ),
    (
        RunSpec("control", "heldout_irstd", 2),
        RunSpec("full", "heldout_irstd", 3),
    ),
)

TIER2_PREREGISTERED_ROUNDS: tuple[tuple[RunSpec, ...], ...] = (
    (
        RunSpec("no_contrast", "heldout_nudt", 2),
        RunSpec("no_component", "heldout_nudt", 3),
    ),
    (
        RunSpec("no_contrast", "heldout_irstd", 2),
        RunSpec("no_component", "heldout_irstd", 3),
    ),
)

TIER3_PREREGISTERED_ROUNDS: tuple[tuple[RunSpec, ...], ...] = (
    (
        RunSpec("no_gate", "heldout_nudt", 2),
        RunSpec("branch_aux", "heldout_nudt", 3),
    ),
    (
        RunSpec("no_gate", "heldout_irstd", 2),
        RunSpec("branch_aux", "heldout_irstd", 3),
    ),
)


@dataclass(frozen=True)
class RunInspection:
    state: str
    epoch: int | None = None
    reason: str | None = None


@dataclass
class LaunchHandle:
    spec: RunSpec
    attempt_id: str
    pid: int
    start_ticks: int
    command: tuple[str, ...]
    mode: str
    log_path: Path
    process: subprocess.Popen[bytes] | None = None
    log_handle: Any | None = None


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(layout: Layout, message: str) -> None:
    layout.state.mkdir(parents=True, exist_ok=True)
    line = f"{now()} {message}\n"
    with (layout.state / "phase3_source_lodo_coordinator.log").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(line)
        handle.flush()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _write_once_json(path: Path, payload: Mapping[str, Any], *, readonly: bool) -> None:
    expected = dict(payload)
    if path.exists():
        if _load_json(path) != expected:
            raise RuntimeError(f"Immutable JSON differs from preregistered content: {path}")
        return
    atomic_write_json(path, expected)
    if readonly:
        path.chmod(0o444)


def _history_epochs(path: Path) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        return []
    if len(fieldnames) != len(set(fieldnames)) or not {
        "epoch",
        "lr",
        "loss_total",
    }.issubset(fieldnames):
        raise ValueError(f"History has an incomplete header: {path}")
    try:
        epochs = [int(row["epoch"]) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid epoch history: {path}") from error
    if epochs != list(range(len(epochs))):
        raise ValueError(f"History is not contiguous 0..N: {path}")
    for row_index, row in enumerate(rows):
        for field in fieldnames:
            raw = row.get(field)
            if raw is None or not raw.strip():
                raise ValueError(f"History row {row_index} has an empty {field}")
            if field == "epoch":
                continue
            try:
                number = float(raw)
            except ValueError as error:
                raise ValueError(
                    f"History row {row_index} has a non-numeric {field}"
                ) from error
            if not math.isfinite(number):
                raise ValueError(f"History row {row_index} has non-finite {field}")
    return epochs


def _run_dir(layout: Layout, spec: RunSpec) -> Path:
    return layout.output / spec.role / spec.fold_key


def _score_dir(layout: Layout, spec: RunSpec) -> Path:
    return _run_dir(layout, spec) / "scores_heldout_train"


def _curve_path(layout: Layout, spec: RunSpec) -> Path:
    return _score_dir(layout, spec) / "threshold_sweep.csv"


def _selection_path(layout: Layout, role: str, budget_name: str) -> Path:
    return layout.audit / "source_operating_points" / role / f"{budget_name}.json"


def _validate_base_config(config: Mapping[str, Any], role: str) -> None:
    identity = config.get("experiment_identity")
    model = config.get("model")
    loss = config.get("loss")
    training = config.get("training")
    if not all(isinstance(value, Mapping) for value in (identity, model, loss, training)):
        raise ValueError(f"Base configuration for {role} lacks a formal section")
    assert isinstance(identity, Mapping)
    assert isinstance(model, Mapping)
    assert isinstance(loss, Mapping)
    assert isinstance(training, Mapping)
    flags = tuple(
        bool(model.get(field))
        for field in (
            "use_contrast",
            "use_component_context",
            "use_risk_gate",
            "expose_branch_auxiliary",
        )
    )
    if flags != EXPECTED_FLAGS[role]:
        raise ValueError(f"Base model flags for {role} changed: {flags}")
    if model.get("backend") != "rc_mshnet":
        raise ValueError("Phase3 inner LODO requires backend=rc_mshnet")
    if config.get("seed") != 42 or config.get("deterministic") is not True:
        raise ValueError("Phase3 inner LODO requires deterministic seed=42")
    if identity.get("base_loss_id") != "stable_sls_v1":
        raise ValueError("Phase3 inner LODO requires StableSLS-v1")
    if any(float(loss.get(name, float("nan"))) != 0.0 for name in (
        "lambda_tail", "lambda_miss", "lambda_margin"
    )):
        raise ValueError("Phase3 source gate forbids Tail/Miss/Margin losses")
    if (
        training.get("checkpoint_selection") != "fixed_last"
        or int(training.get("epochs", -1)) != 80
        or int(training.get("warmup_epochs", -1)) != 0
        or training.get("resume") is not None
    ):
        raise ValueError("Phase3 source gate requires fixed-last 80-epoch training")


def _matched_config_core(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only preregistered role-specific fields for matched comparison."""

    result = copy.deepcopy(dict(config))
    result.pop("output_dir", None)
    result.pop("experiment_identity", None)
    model = result.get("model")
    if not isinstance(model, dict):
        raise ValueError("Matched detector config has no model mapping")
    for field in (
        "use_contrast",
        "use_component_context",
        "use_risk_gate",
        "expose_branch_auxiliary",
    ):
        model.pop(field, None)
    return result


def _validate_matched_tier1_configs(layout: Layout) -> None:
    control = _load_yaml_mapping(layout.configs / BASE_CONFIGS["control"])
    full = _load_yaml_mapping(layout.configs / BASE_CONFIGS["full"])
    _validate_base_config(control, "control")
    _validate_base_config(full, "full")
    if _matched_config_core(control) != _matched_config_core(full):
        raise ValueError(
            "Tier1 control/full are not matched outside preregistered RC model flags"
        )


def _validated_dataset_root(layout: Layout, directory_name: str) -> Path:
    root = (layout.datasets / directory_name).resolve()
    target = (layout.datasets / "NUAA-SIRST").resolve()
    if root == target or target in root.parents or root in target.parents:
        raise RuntimeError("Source dataset resolves to the forbidden NUAA target")
    if not root.is_dir():
        raise FileNotFoundError(root)
    for split in ("train", "test"):
        resolve_split_file(root, None, split=split)
    return root


def expected_config(layout: Layout, spec: RunSpec) -> dict[str, Any]:
    base_path = layout.configs / BASE_CONFIGS[spec.role]
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    config = _load_yaml_mapping(base_path)
    _validate_base_config(config, spec.role)
    result = copy.deepcopy(config)
    result["device"] = "cuda:0"
    result["output_dir"] = str(_run_dir(layout, spec).resolve())
    result["data"]["sources"] = [
        {
            "name": spec.fold.train_name,
            "path": str(
                _validated_dataset_root(layout, spec.fold.train_dataset_dir)
            ),
        }
    ]
    result["data"]["train_split"] = "train"
    result["data"]["val_split"] = None
    result["data"]["diagnostic_test_eval"] = False
    result["training"]["initialize_from"] = str(
        (layout.initializers / spec.fold.initializer_name).resolve()
    )
    result["training"]["resume"] = None
    identity = result["experiment_identity"]
    identity.update(
        {
            "schema_version": "rc-irstd-phase3-source-lodo-v1",
            "stage": "phase3_source_lodo_detector_gate",
            "run_role": spec.run_id,
            "outer_target": spec.fold.held_out_name,
            "target_labels_used_for_training": False,
        }
    )
    result["phase3_runtime_contract"] = {
        "physical_gpu": spec.physical_gpu,
        "cuda_visible_devices": str(spec.physical_gpu),
        "logical_device": "cuda:0",
        "held_out_source_pseudo_target": spec.fold.held_out_name,
        "outer_target_dataset_loaded": False,
    }
    return result


def _ensure_formal_config(layout: Layout, spec: RunSpec) -> Path:
    run_dir = _run_dir(layout, spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "formal_config.yaml"
    expected = expected_config(layout, spec)
    if path.exists():
        if _load_yaml_mapping(path) != expected:
            raise RuntimeError(f"Frozen Phase3 config drifted: {path}")
        return path
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        yaml.safe_dump(expected, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint root is not a mapping: {path}")
    return payload


def _validate_source_split_records(
    spec: RunSpec,
    expected: Mapping[str, Any],
    records: Any,
) -> None:
    sources = expected.get("data", {}).get("sources")
    if not isinstance(sources, list) or len(sources) != 1:
        raise ValueError("Inner-LODO checkpoint must contain exactly one source")
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("Inner-LODO checkpoint source split record count mismatch")
    source = sources[0]
    record = records[0]
    if not isinstance(source, Mapping) or not isinstance(record, Mapping):
        raise ValueError("Inner-LODO source split entry is not a mapping")
    source_root = Path(str(source["path"])).resolve()
    if source.get("name") != spec.fold.train_name:
        raise ValueError("Inner-LODO formal source name mismatch")
    if record.get("name") != spec.fold.train_name:
        raise ValueError("Checkpoint source split name mismatch")
    if Path(str(record.get("path", ""))).resolve() != source_root:
        raise ValueError("Checkpoint source split path mismatch")
    train_path = Path(str(record.get("train_split_file", ""))).resolve()
    test_path = Path(str(record.get("test_split_file", ""))).resolve()
    if train_path != resolve_split_file(source_root, None, split="train"):
        raise ValueError("Checkpoint train split is not the frozen authority")
    if test_path != resolve_split_file(source_root, None, split="test"):
        raise ValueError("Checkpoint test split is not the frozen authority")
    train_ids = ensure_unique_sample_ids(read_split_file(train_path))
    test_ids = ensure_unique_sample_ids(read_split_file(test_path))
    expected_fields = {
        "train_split_file_sha256": file_sha256(train_path),
        "test_split_file_sha256": file_sha256(test_path),
        "train_ordered_ids_sha256": ordered_ids_sha256(train_ids),
        "test_ordered_ids_sha256": ordered_ids_sha256(test_ids),
        "num_train_samples": len(train_ids),
        "num_test_samples": len(test_ids),
        "train_test_id_overlap": False,
    }
    for field, value in expected_fields.items():
        if record.get(field) != value:
            raise ValueError(f"Checkpoint source split {field} mismatch")
    if set(train_ids).intersection(test_ids):
        raise ValueError("Checkpoint source train/test IDs overlap")


def _validate_checkpoint_contract(
    layout: Layout,
    spec: RunSpec,
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    run_dir = _run_dir(layout, spec)
    if _load_json(run_dir / "config.json") != dict(expected):
        raise ValueError("Resolved config.json differs from formal config")
    initializer = layout.initializers / spec.fold.initializer_name
    initialization = _load_json(run_dir / "initialization_report.json")
    if payload.get("initialization") != initialization:
        raise ValueError("Checkpoint initialization report drifted")
    required_initializer = {
        "source_path": str(initializer.resolve()),
        "source_sha256": file_sha256(initializer),
        "backbone_fully_loaded": True,
        "unexpected_keys": [],
        "zero_residual_identity_preserved": True,
    }
    for field, value in required_initializer.items():
        if initialization.get(field) != value:
            raise ValueError(f"Initializer identity mismatch for {field}")
    if (
        payload.get("format_version") != 2
        or payload.get("kind") != "detector"
        or payload.get("config") != dict(expected)
        or payload.get("checkpoint_selection") != "fixed_last"
        or payload.get("selection_rule") != "fixed_last"
        or payload.get("test_labels_used_for_selection") is not False
        or payload.get("diagnostic_test_eval") is not False
        or payload.get("diagnostic_only") is not False
        or payload.get("formal_paper_checkpoint") is not True
        or payload.get("warm_flag") is not True
        or payload.get("inference_head") != "multi_scale_fused"
        or payload.get("source_names") != [spec.fold.train_name]
    ):
        raise ValueError("Checkpoint formal-causal identity mismatch")
    required = {
        "model_state",
        "model_config",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "rng_state",
        "balanced_batcher_state",
        "resume_contract",
        "source_split_records",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError("Checkpoint missing " + ",".join(missing))
    rng_state = payload.get("rng_state")
    if not isinstance(rng_state, Mapping) or len(rng_state.get("torch_cuda", [])) != 1:
        raise ValueError("Checkpoint must contain exactly one CUDA RNG state")
    _validate_source_split_records(spec, expected, payload.get("source_split_records"))
    model = build_mshnet(expected["model"])
    model_config = dict(model.export_config())
    if payload.get("model_config") != model_config:
        raise ValueError("Checkpoint model contract mismatch")
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except RuntimeError as error:
        raise ValueError("Checkpoint model state does not strict-load") from error
    finally:
        del model
    resume = payload.get("resume_contract")
    if not isinstance(resume, Mapping):
        raise ValueError("Checkpoint resume contract is not a mapping")
    if (
        resume.get("schema_version") != 1
        or resume.get("seed") != 42
        or resume.get("deterministic") is not True
        or resume.get("model") != model_config
        or resume.get("loss") != expected["loss"]
        or resume.get("optimizer") != expected["optimizer"]
    ):
        raise ValueError("Checkpoint exact-resume contract mismatch")
    resume_training = resume.get("training")
    if (
        not isinstance(resume_training, Mapping)
        or resume_training.get("epochs") != 80
        or resume_training.get("warmup_epochs") != 0
    ):
        raise ValueError("Checkpoint exact-resume schedule mismatch")


def inspect_run(layout: Layout, spec: RunSpec) -> RunInspection:
    run_dir = _run_dir(layout, spec)
    formal_path = run_dir / "formal_config.yaml"
    expected = expected_config(layout, spec)
    if not formal_path.exists():
        if run_dir.exists() and any(run_dir.iterdir()):
            return RunInspection("corrupt", reason="nonempty run lacks formal_config")
        return RunInspection("fresh")
    if _load_yaml_mapping(formal_path) != expected:
        return RunInspection("corrupt", reason="formal config drift")
    if (run_dir / "last.pt.tmp").exists():
        return RunInspection("corrupt", reason="incomplete checkpoint temporary file")
    history = run_dir / "history.csv"
    checkpoint = run_dir / "last.pt"
    if not history.exists() and not checkpoint.exists():
        other = [path for path in run_dir.iterdir() if path.name != formal_path.name]
        return RunInspection("fresh" if not other else "zero_epoch_partial")
    if history.exists() != checkpoint.exists():
        return RunInspection("corrupt", reason="history/checkpoint pair is incomplete")
    try:
        epochs = _history_epochs(history)
        payload = _checkpoint_payload(checkpoint)
        _validate_checkpoint_contract(layout, spec, payload, expected)
    except Exception as error:
        return RunInspection("corrupt", reason=str(error))
    if not epochs:
        return RunInspection("corrupt", reason="empty history has a checkpoint")
    epoch = int(payload.get("epoch", -1))
    if epoch != len(epochs) - 1:
        return RunInspection("corrupt", reason="history/checkpoint epoch mismatch")
    if epoch == 79 and epochs == list(range(80)):
        return RunInspection("complete", epoch=epoch)
    if 0 <= epoch < 79:
        return RunInspection("resume", epoch=epoch)
    return RunInspection("corrupt", epoch=epoch, reason="epoch outside 0..79")


def _write_run_identity(layout: Layout, spec: RunSpec) -> dict[str, Any]:
    inspection = inspect_run(layout, spec)
    if inspection.state != "complete":
        raise RuntimeError(f"Cannot freeze incomplete run {spec.run_id}: {inspection}")
    run_dir = _run_dir(layout, spec)
    checkpoint = run_dir / "last.pt"
    payload = _checkpoint_payload(checkpoint)
    launch_provenance = _successful_launch_provenance(layout, spec)
    identity = {
        "schema_version": RUN_IDENTITY_SCHEMA,
        "run_id": spec.run_id,
        "role": spec.role,
        "held_out_pseudo_target": spec.fold.held_out_name,
        "training_source": spec.fold.train_name,
        "outer_target_labels_used": False,
        "checkpoint_selection": "fixed_last",
        "checkpoint_epoch": 79,
        "checkpoint_sha256": file_sha256(checkpoint),
        "formal_config_sha256": file_sha256(run_dir / "formal_config.yaml"),
        "resolved_config_sha256": file_sha256(run_dir / "config.json"),
        "history_sha256": file_sha256(run_dir / "history.csv"),
        "initializer_sha256": payload["initialization"]["source_sha256"],
        "physical_gpu": spec.physical_gpu,
        "logical_device": "cuda:0",
        "launch_provenance": launch_provenance,
    }
    _write_once_json(run_dir / "PHASE3_IDENTITY.json", identity, readonly=True)
    checksum = run_dir / "checkpoint.sha256"
    checksum_line = f"{identity['checkpoint_sha256']}  last.pt\n"
    if checksum.exists() and checksum.read_text(encoding="utf-8") != checksum_line:
        raise RuntimeError(f"Checkpoint digest record drifted: {checksum}")
    if not checksum.exists():
        _atomic_write_text(checksum, checksum_line)
        checksum.chmod(0o444)
    for frozen in (
        checkpoint,
        run_dir / "history.csv",
        run_dir / "config.json",
        run_dir / "initialization_report.json",
        run_dir / "formal_config.yaml",
    ):
        frozen.chmod(0o444)
    return identity


def phase2_complete_and_verified(layout: Layout) -> bool:
    if not layout.phase2_status.is_file():
        return False
    status_sha256 = file_sha256(layout.phase2_status)
    status = _load_json(layout.phase2_status)
    if status.get("status") != "completed":
        return False
    if (
        status.get("schema_version") != "rc-irstd-aaai27-phase2-v2"
        or
        status.get("gate_state") != "HOLD"
        or status.get("risk_curve_started") is not False
        or status.get("physical_gpus") != [2, 3]
        or status.get("next_gate") != "detector_matched_fa_evaluation_and_go_no_go"
    ):
        raise RuntimeError("Phase2 completion record violates the fail-closed handoff")
    phase2_pid_path = layout.state / "phase2_coordinator.pid"
    if phase2_pid_path.is_file():
        try:
            phase2_pid = int(phase2_pid_path.read_text(encoding="utf-8").strip())
        except ValueError as error:
            raise RuntimeError("Malformed Phase2 coordinator PID record") from error
        if Path(f"/proc/{phase2_pid}").exists():
            return False
    phase2_hold = layout.state / "HOLD_RC_MSHNET_GATE"
    phase2_allow = layout.state / "ALLOW_RC_MSHNET_GATE"
    if not phase2_hold.is_file() or phase2_allow.exists():
        raise RuntimeError("Phase2 live gate is not in its frozen HOLD state")
    reports = status.get("runs")
    if not isinstance(reports, Mapping) or set(reports) != set(PHASE2_OUTPUTS):
        raise RuntimeError("Phase2 completion record lacks the six frozen runs")
    for role, relative in PHASE2_OUTPUTS.items():
        run_dir = layout.root / relative
        checkpoint = run_dir / "last.pt"
        identity_path = run_dir / "PHASE2_IDENTITY.json"
        if not checkpoint.is_file() or not identity_path.is_file():
            raise RuntimeError(f"Phase2 frozen artifact is missing: {run_dir}")
        identity = _load_json(identity_path)
        report = reports[role]
        if not isinstance(report, Mapping) or identity != dict(report):
            raise RuntimeError(f"Phase2 report/identity mismatch: {role}")
        if report.get("checkpoint_sha256") != file_sha256(checkpoint):
            raise RuntimeError(f"Phase2 checkpoint changed after completion: {role}")
        launcher_path = run_dir / "launcher.json"
        if launcher_path.is_file():
            launcher = _load_json(launcher_path)
            process = launcher.get("process_identity")
            if isinstance(process, Mapping):
                pid = process.get("pid")
                ticks = process.get("start_time_ticks")
                if isinstance(pid, int) and isinstance(ticks, int):
                    try:
                        if _proc_start_ticks(pid) == ticks:
                            raise RuntimeError(
                                f"Phase2 writer is still live after completion: {role}"
                            )
                    except (FileNotFoundError, ProcessLookupError):
                        pass
    if status_sha256 != file_sha256(layout.phase2_status):
        raise RuntimeError("Phase2 status changed during handoff verification")
    return True


def _wait_for_phase2(layout: Layout, poll_seconds: float) -> None:
    while not phase2_complete_and_verified(layout):
        log(layout, "waiting for verified Phase2 completion; no source/target data opened")
        time.sleep(poll_seconds)


def _dataset_prereg_binding(layout: Layout, directory_name: str) -> dict[str, Any]:
    root = _validated_dataset_root(layout, directory_name)
    payload: dict[str, Any] = {
        "root": str(root),
        "configured_path_is_symlink": (layout.datasets / directory_name).is_symlink(),
    }
    for split in ("train", "test"):
        path = resolve_split_file(root, None, split=split)
        ids = ensure_unique_sample_ids(read_split_file(path))
        payload[split] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "ordered_ids_sha256": ordered_ids_sha256(ids),
            "num_samples": len(ids),
        }
    if set(
        read_split_file(Path(payload["train"]["path"]))
    ).intersection(read_split_file(Path(payload["test"]["path"]))):
        raise ValueError(f"Frozen source splits overlap: {directory_name}")
    return payload


def _prereg_payload(layout: Layout) -> dict[str, Any]:
    all_specs = [spec for rounds in (
        TIER1_ROUNDS,
        TIER2_PREREGISTERED_ROUNDS,
        TIER3_PREREGISTERED_ROUNDS,
    ) for round_specs in rounds for spec in round_specs]
    _validate_matched_tier1_configs(layout)
    return {
        "schema_version": PREREG_SCHEMA,
        "coordinator": str(Path(__file__).resolve()),
        "coordinator_sha256": file_sha256(Path(__file__).resolve()),
        "python_executable": str(EXPECTED_PYTHON.resolve()),
        "python_executable_sha256": file_sha256(EXPECTED_PYTHON.resolve()),
        "project_root": str(layout.root.resolve()),
        "phase2_prerequisite": str(layout.phase2_status.resolve()),
        "outer_target": "NUAA-SIRST",
        "outer_target_images_loaded": False,
        "outer_target_masks_loaded": False,
        "source_pseudo_targets": ["NUDT-SIRST", "IRSTD-1K"],
        "source_split": "train",
        "source_dataset_bindings": {
            name: _dataset_prereg_binding(layout, name)
            for name in ("NUDT-SIRST", "IRSTD-1K")
        },
        "threshold_protocol": {
            "grid": "evaluation.threshold_sweep.build_default_thresholds",
            "grid_sha256": evaluation_threshold_grid_sha256(
                build_default_thresholds()
            ),
            "matching_rule": "overlap",
            "centroid_distance": 3.0,
            "connectivity": 2,
            "min_component_area": 1,
        },
        "code_bindings": {
            str(path.relative_to(layout.root)): file_sha256(path)
            for path in (
                layout.root / "train_detector.py",
                layout.root / "export_scores.py",
                layout.root / "rc_irstd/cli/train_detector.py",
                layout.root / "rc_irstd/cli/export_scores.py",
                layout.root / "rc_irstd/training/detector_trainer.py",
                layout.root / "evaluation/threshold_sweep.py",
                layout.root / "evaluation/source_operating_point.py",
            )
        },
        "budgets": [
            {"name": name, "pixel": pixel, "component": component}
            for name, pixel, component in BUDGETS
        ],
        "tier1_policy": {
            "strict_macro_pd_gain_minimum": STRICT_MACRO_PD_GAIN,
            "strict_each_source_pd_non_degraded": True,
            "all_budget_pooled_pd_non_degraded": True,
            "all_budget_worst_pd_non_degraded": True,
            "all_control_and_full_pooled_and_worst_points_required": True,
            "matched_pd_component_fa_alternative_enabled": False,
            "training_no_commit_timeout_seconds": TRAINING_NO_COMMIT_TIMEOUT_SECONDS,
            "training_termination_grace_seconds": TRAINING_TERMINATION_GRACE_SECONDS,
            "maximum_recovery_attempts": MAX_RECOVERY_ATTEMPTS,
        },
        "tier2_policy": {
            "enabled_only_after_tier1_go": True,
            "roles": ["no_contrast", "no_component"],
            "comparison_pairs": [
                ["full", "no_contrast"],
                ["full", "no_component"],
            ],
            "pass_conditions": [
                {
                    "metric": "macro_pd_delta",
                    "budgets": ["strict"],
                    "operator": "greater_than",
                    "value": 0.0,
                },
                {
                    "metric": "each_domain_pd_delta",
                    "budgets": ["strict"],
                    "operator": "greater_than_or_equal",
                    "value": 0.0,
                },
                {
                    "metric": "pooled_pd_delta",
                    "budgets": [name for name, _, _ in BUDGETS],
                    "operator": "greater_than_or_equal",
                    "value": 0.0,
                },
                {
                    "metric": "worst_pd_delta",
                    "budgets": [name for name, _, _ in BUDGETS],
                    "operator": "greater_than_or_equal",
                    "value": 0.0,
                },
            ],
            "numeric_tolerance": NUMERIC_ATOL,
            "missing_or_infeasible_point": "HOLD_TIER2",
        },
        "tier3_policy": {
            "enabled_only_after_tier2_go": True,
            "roles": ["no_gate", "branch_aux"],
            "comparison_pairs": [["full", "no_gate"]],
            "pass_conditions": [
                {
                    "metric": "pooled_pd_delta",
                    "budgets": [name for name, _, _ in BUDGETS],
                    "operator": "greater_than_or_equal",
                    "value": 0.0,
                },
                {
                    "metric": "worst_pd_delta",
                    "budgets": [name for name, _, _ in BUDGETS],
                    "operator": "greater_than_or_equal",
                    "value": 0.0,
                },
                {
                    "metric": "each_domain_pd_delta",
                    "budgets": ["strict"],
                    "operator": "greater_than_or_equal",
                    "value": 0.0,
                },
            ],
            "branch_aux_role": "diagnostic_only_not_required_for_core_go",
            "numeric_tolerance": NUMERIC_ATOL,
            "missing_or_infeasible_point": "HOLD_TIER3",
        },
        "tier1_schedule": [
            [
                {
                    "run_id": spec.run_id,
                    "physical_gpu": spec.physical_gpu,
                    "logical_device": "cuda:0",
                }
                for spec in round_specs
            ]
            for round_specs in TIER1_ROUNDS
        ],
        "future_schedule": [
            {
                "run_id": spec.run_id,
                "base_config": BASE_CONFIGS[spec.role],
                "base_config_sha256": file_sha256(
                    layout.configs / BASE_CONFIGS[spec.role]
                ),
                "initializer": spec.fold.initializer_name,
                "initializer_sha256": file_sha256(
                    layout.initializers / spec.fold.initializer_name
                ),
            }
            for spec in all_specs
        ],
        "forbidden_commands": ["run_pipeline.py", "rc_irstd.cli.run_pipeline"],
        "target_label_hold_sentinel": str(
            (layout.state / "HOLD_PHASE3_TARGET_LABEL_ACCESS").resolve()
        ),
    }


def _ensure_preregistration(layout: Layout) -> Path:
    layout.audit.mkdir(parents=True, exist_ok=True)
    path = layout.audit / "PHASE3_SOURCE_LODO_PREREGISTRATION.json"
    _write_once_json(path, _prereg_payload(layout), readonly=True)
    digest = layout.audit / "PHASE3_SOURCE_LODO_PREREGISTRATION.sha256"
    line = f"{file_sha256(path)}  {path.name}\n"
    if digest.exists() and digest.read_text(encoding="utf-8") != line:
        raise RuntimeError("Phase3 preregistration digest record changed")
    if not digest.exists():
        _atomic_write_text(digest, line)
        digest.chmod(0o444)
    return path


def _proc_start_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    closing = raw.rfind(")")
    if closing < 0:
        raise ValueError(f"Malformed /proc stat for pid={pid}")
    fields = raw[closing + 2 :].split()
    return int(fields[19])


def _proc_command(pid: int) -> tuple[str, ...]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return tuple(value.decode("utf-8") for value in raw.split(b"\0") if value)


def _proc_visible_gpu(pid: int) -> str | None:
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    for item in raw.split(b"\0"):
        if item.startswith(b"CUDA_VISIBLE_DEVICES="):
            return item.split(b"=", 1)[1].decode("ascii")
    return None


def _proc_environment_value(pid: int, name: str) -> str | None:
    prefix = (name + "=").encode("ascii")
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    for item in raw.split(b"\0"):
        if item.startswith(prefix):
            return item.split(b"=", 1)[1].decode("utf-8")
    return None


def _proc_cwd(pid: int) -> str:
    return str(Path(f"/proc/{pid}/cwd").resolve())


def _pid_matches_launch(pid: int, start_ticks: int, command: Sequence[str], gpu: int) -> bool:
    try:
        return (
            _proc_start_ticks(pid) == start_ticks
            and _proc_command(pid) == tuple(command)
            and _proc_visible_gpu(pid) == str(gpu)
        )
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return False


def _active_launch(layout: Layout, spec: RunSpec) -> LaunchHandle | None:
    active = _run_dir(layout, spec) / "ACTIVE_LAUNCH.json"
    if not active.is_file():
        return None
    payload = _load_json(active)
    if payload.get("schema_version") != LAUNCH_SCHEMA or payload.get("run_id") != spec.run_id:
        raise RuntimeError(f"Malformed active launch record: {active}")
    attempt_id = payload.get("attempt_id")
    mode = payload.get("mode")
    if (
        not isinstance(attempt_id, str)
        or len(attempt_id) != 32
        or any(character not in "0123456789abcdef" for character in attempt_id)
        or mode not in {"fresh", "resume"}
    ):
        raise RuntimeError(f"Malformed active launch identity: {active}")
    attempts = _run_dir(layout, spec) / "launch_attempts"
    intent_path = attempts / f"{attempt_id}.intent.json"
    binding_path = attempts / f"{attempt_id}.binding.json"
    if not intent_path.is_file() or not binding_path.is_file():
        raise RuntimeError(f"Active launch lacks immutable intent/binding: {active}")
    intent = _load_json(intent_path)
    binding = _load_json(binding_path)
    if binding != payload:
        raise RuntimeError(f"Mutable ACTIVE differs from immutable binding: {active}")
    if binding.get("intent_sha256") != file_sha256(intent_path):
        raise RuntimeError(f"Active launch intent digest mismatch: {active}")
    binding_expected = {
        "attempt_id": attempt_id,
        "run_id": spec.run_id,
        "mode": mode,
        "physical_gpu": spec.physical_gpu,
        "cuda_visible_devices": str(spec.physical_gpu),
        "logical_device": "cuda:0",
        "cwd": str(layout.root.resolve()),
    }
    for field, value in binding_expected.items():
        if binding.get(field) != value:
            raise RuntimeError(f"Active launch {field} mismatch: {active}")
    if (
        intent.get("schema_version") != LAUNCH_SCHEMA
        or intent.get("attempt_id") != attempt_id
        or intent.get("run_id") != spec.run_id
        or intent.get("mode") != mode
        or intent.get("physical_gpu") != spec.physical_gpu
        or intent.get("logical_device") != "cuda:0"
    ):
        raise RuntimeError(f"Active launch intent contract mismatch: {active}")
    pid = int(payload.get("pid", -1))
    ticks = int(payload.get("proc_start_ticks", -1))
    command = tuple(str(value) for value in payload.get("command", []))
    _assert_source_only_command(layout, command)
    inspection = RunInspection(str(mode))
    resume_audit_value = intent.get("resume_audit")
    resume_audit = (
        Path(str(resume_audit_value)) if resume_audit_value is not None else None
    )
    expected_command = tuple(
        training_command(
            layout,
            spec,
            inspection,
            resume_audit=resume_audit,
        )
    )
    if command != expected_command or intent.get("command") != list(expected_command):
        raise RuntimeError(f"Active launch command mismatch: {active}")
    formal_sha256 = file_sha256(_run_dir(layout, spec) / "formal_config.yaml")
    if (
        payload.get("formal_config_sha256") != formal_sha256
        or intent.get("formal_config_sha256") != formal_sha256
        or payload.get("command_sha256") != _canonical_json_sha256(command)
        or intent.get("command_sha256") != _canonical_json_sha256(command)
    ):
        raise RuntimeError(f"Active launch formal config digest mismatch: {active}")
    log_path = Path(str(payload.get("log", ""))).resolve()
    expected_log = (attempts / f"{attempt_id}.log").resolve()
    if log_path != expected_log or not log_path.is_file():
        raise RuntimeError(f"Active launch log binding mismatch: {active}")
    try:
        current_ticks = _proc_start_ticks(pid)
    except (FileNotFoundError, ProcessLookupError):
        current_ticks = None
    if current_ticks is not None and current_ticks != ticks:
        raise RuntimeError(f"Active launch PID was reused: {active}")
    if current_ticks is not None:
        live_expected = {
            "command": _proc_command(pid) == expected_command,
            "cuda_visible_devices": _proc_visible_gpu(pid) == str(spec.physical_gpu),
            "cwd": _proc_cwd(pid) == str(layout.root.resolve()),
            "launch_id": _proc_environment_value(
                pid, "RC_IRSTD_PHASE3_LAUNCH_ID"
            )
            == attempt_id,
            "process_group": os.getpgid(pid) == pid,
            "session": os.getsid(pid) == pid,
        }
        failed = sorted(key for key, value in live_expected.items() if not value)
        if failed:
            raise RuntimeError(
                f"Active launch live identity mismatch ({','.join(failed)}): {active}"
            )
    return LaunchHandle(
        spec=spec,
        attempt_id=attempt_id,
        pid=pid,
        start_ticks=ticks,
        command=expected_command,
        mode=str(mode),
        log_path=log_path,
    )


def _assert_source_only_command(layout: Layout, command: Sequence[str]) -> None:
    rendered = " ".join(str(value) for value in command)
    forbidden = (
        "run_pipeline.py",
        "rc_irstd.cli.run_pipeline",
        str((layout.datasets / "NUAA-SIRST").resolve()),
        "--dataset-name NUAA-SIRST",
        "--target-curve",
        "NUAA-SIRST",
    )
    if any(value in rendered for value in forbidden):
        raise RuntimeError(f"Forbidden Phase3 command: {rendered}")


def training_command(
    layout: Layout,
    spec: RunSpec,
    inspection: RunInspection,
    *,
    resume_audit: Path | None = None,
) -> list[str]:
    config = _ensure_formal_config(layout, spec)
    run_dir = _run_dir(layout, spec)
    command = [
        sys.executable,
        str(layout.root / "train_detector.py"),
        "--config",
        str(config),
        "--pid-file",
        str(run_dir / "TRAINER.pid"),
    ]
    if inspection.state == "resume":
        if resume_audit is None:
            raise ValueError("Exact resume requires a unique immutable audit path")
        command.extend(
            [
                "--resume-checkpoint",
                str(run_dir / "last.pt"),
                "--resume-audit",
                str(resume_audit.resolve()),
            ]
        )
    elif inspection.state != "fresh":
        raise RuntimeError(f"Cannot launch {spec.run_id} from {inspection.state}")
    _assert_source_only_command(layout, command)
    return command


def _launch_training(layout: Layout, spec: RunSpec, inspection: RunInspection) -> LaunchHandle:
    run_dir = _run_dir(layout, spec)
    attempts = run_dir / "launch_attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    committed_attempts = len(list(attempts.glob("*.binding.json")))
    if committed_attempts >= MAX_RECOVERY_ATTEMPTS + 1:
        raise RuntimeError(f"Persistent launch-attempt limit exceeded: {spec.run_id}")
    attempt_id = uuid.uuid4().hex
    log_path = attempts / f"{attempt_id}.log"
    intent_path = attempts / f"{attempt_id}.intent.json"
    binding_path = attempts / f"{attempt_id}.binding.json"
    resume_audit = (
        attempts / f"{attempt_id}.resume_audit.json"
        if inspection.state == "resume"
        else None
    )
    command = training_command(
        layout,
        spec,
        inspection,
        resume_audit=resume_audit,
    )
    resume_source: dict[str, Any] | None = None
    if inspection.state == "resume":
        checkpoint = run_dir / "last.pt"
        history = run_dir / "history.csv"
        resume_source = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
            "epoch": inspection.epoch,
            "history_rows": len(_history_epochs(history)),
            "history_sha256": file_sha256(history),
            "audit": str(resume_audit.resolve()),
        }
    intent = {
        "schema_version": LAUNCH_SCHEMA,
        "attempt_id": attempt_id,
        "run_id": spec.run_id,
        "mode": inspection.state,
        "physical_gpu": spec.physical_gpu,
        "logical_device": "cuda:0",
        "command": command,
        "command_sha256": _canonical_json_sha256(command),
        "formal_config_sha256": file_sha256(run_dir / "formal_config.yaml"),
        "resume_audit": str(resume_audit.resolve()) if resume_audit else None,
        "resume_source": resume_source,
        "created_at": now(),
    }
    # The intent closes the spawn-before-record crash window: an intent with no
    # binding is ambiguous and a restarted coordinator must fail closed.
    _write_once_json(intent_path, intent, readonly=True)
    environment = dict(os.environ)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = str(spec.physical_gpu)
    environment["RC_IRSTD_PHASE3_LAUNCH_ID"] = attempt_id
    log_handle = log_path.open("xb")
    try:
        process = subprocess.Popen(
            command,
            cwd=layout.root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        log_handle.close()
        raise
    ticks: int | None = None
    for _ in range(100):
        try:
            ticks = _proc_start_ticks(process.pid)
            break
        except FileNotFoundError:
            if process.poll() is not None:
                break
            time.sleep(0.01)
    if ticks is None:
        process.terminate()
        process.wait(timeout=5)
        log_handle.close()
        raise RuntimeError(f"Could not bind launch identity for {spec.run_id}")
    binding = {
        "schema_version": LAUNCH_SCHEMA,
        "attempt_id": attempt_id,
        "run_id": spec.run_id,
        "mode": inspection.state,
        "pid": process.pid,
        "proc_start_ticks": ticks,
        "physical_gpu": spec.physical_gpu,
        "cuda_visible_devices": str(spec.physical_gpu),
        "logical_device": "cuda:0",
        "command": command,
        "command_sha256": _canonical_json_sha256(command),
        "formal_config_sha256": file_sha256(run_dir / "formal_config.yaml"),
        "intent_sha256": file_sha256(intent_path),
        "log": str(log_path.resolve()),
        "cwd": str(layout.root.resolve()),
        "process_group_id": process.pid,
        "session_id": process.pid,
        "started_at": now(),
    }
    _write_once_json(binding_path, binding, readonly=True)
    atomic_write_json(run_dir / "ACTIVE_LAUNCH.json", binding)
    return LaunchHandle(
        spec=spec,
        attempt_id=attempt_id,
        pid=process.pid,
        start_ticks=ticks,
        command=tuple(command),
        mode=inspection.state,
        log_path=log_path.resolve(),
        process=process,
        log_handle=log_handle,
    )


def _finish_launch(layout: Layout, handle: LaunchHandle, return_code: int | None) -> None:
    run_dir = _run_dir(layout, handle.spec)
    if handle.log_handle is not None:
        handle.log_handle.close()
        handle.log_handle = None
    attempts = run_dir / "launch_attempts"
    result_path = attempts / f"{handle.attempt_id}.result.json"
    if result_path.is_file():
        existing = _load_json(result_path)
        if (
            existing.get("attempt_id") != handle.attempt_id
            or existing.get("run_id") != handle.spec.run_id
            or existing.get("pid") != handle.pid
        ):
            raise RuntimeError(f"Existing launch result identity mismatch: {result_path}")
        active = run_dir / "ACTIVE_LAUNCH.json"
        if active.exists() and _load_json(active).get("attempt_id") == handle.attempt_id:
            active.unlink()
        return
    log_trailer = None
    if handle.log_path.is_file():
        lines = handle.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if lines:
            log_trailer = lines[-1]
    checkpoint = run_dir / "last.pt"
    history = run_dir / "history.csv"
    final_state: dict[str, Any] | None = None
    if checkpoint.is_file() and history.is_file():
        try:
            checkpoint_payload = _checkpoint_payload(checkpoint)
            history_rows = len(_history_epochs(history))
            final_state = {
                "checkpoint_sha256": file_sha256(checkpoint),
                "checkpoint_epoch": checkpoint_payload.get("epoch"),
                "history_rows": history_rows,
                "history_sha256": file_sha256(history),
            }
        except Exception as error:
            final_state = {"inspection_error": str(error)}
    _write_once_json(
        result_path,
        {
            "schema_version": "rc-irstd-phase3-launch-result-v1",
            "attempt_id": handle.attempt_id,
            "run_id": handle.spec.run_id,
            "pid": handle.pid,
            "return_code": return_code,
            "return_code_observable": return_code is not None,
            "log_sha256": file_sha256(handle.log_path),
            "log_trailer": log_trailer,
            "final_state": final_state,
            "observed_at": now(),
        },
        readonly=True,
    )
    active = run_dir / "ACTIVE_LAUNCH.json"
    if active.exists():
        payload = _load_json(active)
        if (
            int(payload.get("pid", -1)) == handle.pid
            and payload.get("attempt_id") == handle.attempt_id
        ):
            active.unlink()


def _validate_finished_attempts(layout: Layout, spec: RunSpec) -> None:
    """Validate the immutable launch ledger and prior exact-resume audits."""

    attempts = _run_dir(layout, spec) / "launch_attempts"
    if not attempts.exists():
        return
    active_path = _run_dir(layout, spec) / "ACTIVE_LAUNCH.json"
    active_id = _load_json(active_path).get("attempt_id") if active_path.is_file() else None
    for intent_path in sorted(attempts.glob("*.intent.json")):
        attempt_id = intent_path.name.removesuffix(".intent.json")
        binding_path = attempts / f"{attempt_id}.binding.json"
        result_path = attempts / f"{attempt_id}.result.json"
        if not binding_path.is_file():
            raise RuntimeError(
                f"Ambiguous spawn intent has no PID binding: {intent_path}"
            )
        intent = _load_json(intent_path)
        binding = _load_json(binding_path)
        if binding.get("intent_sha256") != file_sha256(intent_path):
            raise RuntimeError(f"Launch binding/intent digest mismatch: {binding_path}")
        if not result_path.is_file():
            if active_id == attempt_id:
                continue
            raise RuntimeError(
                f"Launch binding has neither ACTIVE ownership nor result: {binding_path}"
            )
        result = _load_json(result_path)
        if (
            result.get("attempt_id") != attempt_id
            or result.get("run_id") != spec.run_id
            or result.get("pid") != binding.get("pid")
        ):
            raise RuntimeError(f"Launch result identity mismatch: {result_path}")
        log_path = Path(str(binding.get("log", ""))).resolve()
        if not log_path.is_file() or result.get("log_sha256") != file_sha256(log_path):
            raise RuntimeError(f"Launch log changed after result freeze: {result_path}")
        return_code = result.get("return_code")
        if isinstance(return_code, int) and return_code > 0:
            raise RuntimeError(
                f"Prior training attempt failed cleanly with rc={return_code}: {result_path}"
            )
        if intent.get("mode") != "resume":
            continue
        source = intent.get("resume_source")
        if not isinstance(source, Mapping):
            raise RuntimeError(f"Resume intent lacks source binding: {intent_path}")
        audit_path = Path(str(intent.get("resume_audit", ""))).resolve()
        final_state = result.get("final_state")
        if not audit_path.is_file():
            unchanged = isinstance(final_state, Mapping) and all(
                final_state.get(result_field) == source.get(source_field)
                for result_field, source_field in (
                    ("checkpoint_sha256", "checkpoint_sha256"),
                    ("checkpoint_epoch", "epoch"),
                    ("history_rows", "history_rows"),
                    ("history_sha256", "history_sha256"),
                )
            )
            if not unchanged:
                raise RuntimeError(
                    f"Resume progressed without its required audit: {intent_path}"
                )
            continue
        audit = _load_json(audit_path)
        if audit.get("schema_version") != "rc-irstd-formal-resume-v1":
            raise RuntimeError(f"Resume audit schema mismatch: {audit_path}")
        if audit.get("status") not in {"prepared", "running", "completed"}:
            raise RuntimeError(
                f"Prior resume audit is a clean failure: {audit_path}"
            )
        events = audit.get("events")
        if not isinstance(events, list) or len(events) != 1 or not isinstance(events[0], Mapping):
            raise RuntimeError(f"Resume audit event count mismatch: {audit_path}")
        event = events[0]
        expected_event = {
            "source_checkpoint_path": source["checkpoint"],
            "source_checkpoint_sha256": source["checkpoint_sha256"],
            "source_epoch": source["epoch"],
            "history_rows_before_resume": source["history_rows"],
            "history_sha256_before_resume": source["history_sha256"],
            "cuda_visible_devices": str(spec.physical_gpu),
            "physical_gpu_index": spec.physical_gpu,
            "logical_device": "cuda:0",
        }
        for field, value in expected_event.items():
            if event.get(field) != value:
                raise RuntimeError(f"Resume audit {field} mismatch: {audit_path}")
        if not isinstance(final_state, Mapping):
            raise RuntimeError(f"Resume result has no exact final state: {result_path}")
        final_epoch = final_state.get("checkpoint_epoch")
        final_rows = final_state.get("history_rows")
        if (
            not isinstance(final_epoch, int)
            or isinstance(final_epoch, bool)
            or not source["epoch"] <= final_epoch <= 79
            or final_rows != final_epoch + 1
        ):
            raise RuntimeError(f"Resume result epoch/history mismatch: {result_path}")
        if audit.get("status") == "completed":
            if (
                event.get("status") != "completed"
                or not isinstance(final_state, Mapping)
                or final_state.get("checkpoint_epoch") != 79
                or final_state.get("history_rows") != 80
                or event.get("final_checkpoint_sha256")
                != final_state.get("checkpoint_sha256")
            ):
                raise RuntimeError(
                    f"Completed resume audit is not final-state bound: {audit_path}"
                )
        elif event.get("status") not in {"prepared", "running"}:
            raise RuntimeError(f"Interrupted resume event has invalid status: {audit_path}")


def _successful_launch_provenance(layout: Layout, spec: RunSpec) -> dict[str, Any]:
    run_dir = _run_dir(layout, spec)
    checkpoint = run_dir / "last.pt"
    attempts = run_dir / "launch_attempts"
    expected_trailer = str(checkpoint.resolve())
    candidates: list[tuple[Path, dict[str, Any], Path, dict[str, Any]]] = []
    for result_path in attempts.glob("*.result.json") if attempts.exists() else ():
        result = _load_json(result_path)
        attempt_id = result.get("attempt_id")
        if not isinstance(attempt_id, str):
            continue
        binding_path = attempts / f"{attempt_id}.binding.json"
        if not binding_path.is_file():
            continue
        binding = _load_json(binding_path)
        final_state = result.get("final_state")
        if (
            result.get("return_code") not in {0, None}
            or result.get("log_trailer") != expected_trailer
            or not isinstance(final_state, Mapping)
            or final_state.get("checkpoint_epoch") != 79
            or final_state.get("history_rows") != 80
            or final_state.get("checkpoint_sha256") != file_sha256(checkpoint)
        ):
            continue
        candidates.append((binding_path, binding, result_path, result))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Complete run {spec.run_id} requires exactly one successful launch provenance"
        )
    binding_path, binding, result_path, result = candidates[0]
    return {
        "attempt_id": binding["attempt_id"],
        "binding": str(binding_path.resolve()),
        "binding_sha256": file_sha256(binding_path),
        "result": str(result_path.resolve()),
        "result_sha256": file_sha256(result_path),
        "observed_physical_gpu": binding["physical_gpu"],
        "observed_cuda_visible_devices": binding["cuda_visible_devices"],
        "observed_logical_device": binding["logical_device"],
    }


def _prepare_launch_state(layout: Layout, spec: RunSpec) -> RunInspection:
    _validate_finished_attempts(layout, spec)
    inspection = inspect_run(layout, spec)
    # Never write a formal config into an unknown non-empty directory.  The
    # missing-config state is evidence and must fail closed.  A config is
    # created only for a genuinely fresh (absent or empty) run directory.
    if inspection.state == "fresh":
        _ensure_formal_config(layout, spec)
        inspection = inspect_run(layout, spec)
    if inspection.state == "zero_epoch_partial":
        raise RuntimeError(
            f"{spec.run_id} has a zero-epoch half product; evidence is preserved "
            "and automatic fresh restart is forbidden"
        )
    if inspection.state == "corrupt":
        raise RuntimeError(f"{spec.run_id} failed closed: {inspection.reason}")
    return inspection


def _commit_signature(layout: Layout, spec: RunSpec) -> tuple[Any, ...]:
    run_dir = _run_dir(layout, spec)
    history = run_dir / "history.csv"
    checkpoint = run_dir / "last.pt"
    try:
        rows = len(_history_epochs(history)) if history.is_file() else 0
    except Exception as error:
        return ("invalid_history", type(error).__name__, str(error))
    if not checkpoint.is_file():
        return (rows, None, None)
    stat = checkpoint.stat()
    return (rows, stat.st_size, stat.st_mtime_ns)


def _reattached_progress_state(
    layout: Layout, handle: LaunchHandle
) -> tuple[tuple[Any, ...], float]:
    run_dir = _run_dir(layout, handle.spec)
    candidates = [
        run_dir / "history.csv",
        run_dir / "last.pt",
        run_dir / "launch_attempts" / f"{handle.attempt_id}.binding.json",
    ]
    wall_times = [path.stat().st_mtime for path in candidates if path.is_file()]
    elapsed = max(0.0, time.time() - max(wall_times)) if wall_times else 0.0
    return (
        _commit_signature(layout, handle.spec),
        time.monotonic() - min(elapsed, TRAINING_NO_COMMIT_TIMEOUT_SECONDS),
    )


def _signal_owned_launch(handle: LaunchHandle, signum: int) -> None:
    if not _pid_matches_launch(
        handle.pid,
        handle.start_ticks,
        handle.command,
        handle.spec.physical_gpu,
    ):
        raise RuntimeError(
            f"Refusing to signal unbound PID for {handle.spec.run_id}"
        )
    if os.getpgid(handle.pid) != handle.pid or os.getsid(handle.pid) != handle.pid:
        raise RuntimeError(
            f"Refusing to signal non-isolated process for {handle.spec.run_id}"
        )
    os.killpg(handle.pid, signum)


def run_training_round(layout: Layout, name: str, specs: Sequence[RunSpec]) -> None:
    handles: dict[str, LaunchHandle] = {}
    retries: dict[str, int] = {spec.run_id: 0 for spec in specs}
    progress: dict[str, tuple[tuple[Any, ...], float]] = {}
    termination_requested: dict[str, float] = {}
    try:
        for spec in specs:
            active = _active_launch(layout, spec)
            if active is not None:
                handles[spec.run_id] = active
                progress[spec.run_id] = _reattached_progress_state(layout, active)
                log(layout, f"{name} attached {spec.run_id} pid={active.pid}")
                continue
            inspection = _prepare_launch_state(layout, spec)
            if inspection.state == "complete":
                _write_run_identity(layout, spec)
                log(layout, f"{name} reused complete {spec.run_id}")
                continue
            handle = _launch_training(layout, spec, inspection)
            handles[spec.run_id] = handle
            progress[spec.run_id] = (
                _commit_signature(layout, spec),
                time.monotonic(),
            )
            log(
                layout,
                f"{name} launched {spec.run_id} pid={handle.pid} physical_gpu={spec.physical_gpu}",
            )
        while handles:
            for run_id, handle in list(handles.items()):
                return_code: int | None = None
                alive = False
                if handle.process is not None:
                    return_code = handle.process.poll()
                    alive = return_code is None
                else:
                    alive = _pid_matches_launch(
                        handle.pid,
                        handle.start_ticks,
                        handle.command,
                        handle.spec.physical_gpu,
                    )
                if alive:
                    observed = _commit_signature(layout, handle.spec)
                    previous, last_progress = progress[run_id]
                    current_time = time.monotonic()
                    if observed != previous:
                        progress[run_id] = (observed, current_time)
                        termination_requested.pop(run_id, None)
                    elif run_id not in termination_requested and (
                        current_time - last_progress
                        >= TRAINING_NO_COMMIT_TIMEOUT_SECONDS
                    ):
                        _signal_owned_launch(handle, signal.SIGTERM)
                        termination_requested[run_id] = current_time
                        log(
                            layout,
                            f"{name} no-commit watchdog sent SIGTERM to "
                            f"{run_id} pid={handle.pid}",
                        )
                    elif run_id in termination_requested and (
                        current_time - termination_requested[run_id]
                        >= TRAINING_TERMINATION_GRACE_SECONDS
                    ):
                        _signal_owned_launch(handle, signal.SIGKILL)
                        termination_requested[run_id] = current_time
                    continue
                _finish_launch(layout, handle, return_code)
                inspection = _prepare_launch_state(layout, handle.spec)
                if inspection.state == "complete":
                    identity = _write_run_identity(layout, handle.spec)
                    log(
                        layout,
                        f"{name} completed {run_id} sha256={identity['checkpoint_sha256']}",
                    )
                    handles.pop(run_id)
                    progress.pop(run_id, None)
                    termination_requested.pop(run_id, None)
                    continue
                retries[run_id] += 1
                if retries[run_id] > MAX_RECOVERY_ATTEMPTS:
                    raise RuntimeError(f"Automatic recovery limit exceeded: {run_id}")
                replacement = _launch_training(layout, handle.spec, inspection)
                handles[run_id] = replacement
                progress[run_id] = (
                    _commit_signature(layout, handle.spec),
                    time.monotonic(),
                )
                termination_requested.pop(run_id, None)
                log(layout, f"{name} recovering {run_id} from {inspection.state}")
            if handles:
                time.sleep(10)
    except BaseException:
        for handle in handles.values():
            if handle.process is not None and handle.process.poll() is None:
                # These Popen objects are children created by this coordinator;
                # no PID discovered outside this launch table is ever signalled.
                handle.process.terminate()
        for handle in handles.values():
            if handle.process is not None:
                try:
                    handle.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    handle.process.kill()
                    handle.process.wait(timeout=5)
                _finish_launch(layout, handle, handle.process.returncode)
        raise


def _run_checked(
    layout: Layout,
    spec: RunSpec,
    command: Sequence[str],
    *,
    stage: str,
    uses_gpu: bool,
) -> dict[str, Any]:
    _assert_source_only_command(layout, command)
    if not stage or any(not (value.isalnum() or value in "_-") for value in stage):
        raise ValueError(f"Unsafe stage identity: {stage!r}")
    stage_dir = _run_dir(layout, spec) / "stage_attempts" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    active_path = stage_dir / "ACTIVE.json"
    handle: subprocess.Popen[bytes] | None = None
    log_handle: Any | None = None
    return_code: int | None = None
    attempt_id: str
    binding: dict[str, Any]
    if active_path.is_file():
        binding = _load_json(active_path)
        attempt_id = str(binding.get("attempt_id", ""))
        intent_path = stage_dir / f"{attempt_id}.intent.json"
        binding_path = stage_dir / f"{attempt_id}.binding.json"
        if (
            not intent_path.is_file()
            or not binding_path.is_file()
            or _load_json(binding_path) != binding
            or binding.get("intent_sha256") != file_sha256(intent_path)
        ):
            raise RuntimeError(f"{stage} ACTIVE lacks immutable intent/binding")
        intent = _load_json(intent_path)
        expected_visible = str(spec.physical_gpu) if uses_gpu else ""
        expected = {
            "schema_version": "rc-irstd-phase3-stage-launch-v1",
            "stage": stage,
            "run_id": spec.run_id,
            "command": list(command),
            "uses_gpu": uses_gpu,
            "cuda_visible_devices": expected_visible,
            "logical_device": "cuda:0" if uses_gpu else None,
        }
        for field, value in expected.items():
            if binding.get(field) != value or intent.get(field) != value:
                raise RuntimeError(f"{stage} active {field} mismatch")
        pid = int(binding.get("pid", -1))
        ticks = int(binding.get("proc_start_ticks", -1))
        try:
            current_ticks = _proc_start_ticks(pid)
        except (FileNotFoundError, ProcessLookupError):
            current_ticks = None
        if current_ticks is not None and current_ticks != ticks:
            raise RuntimeError(f"{stage} ACTIVE PID was reused")
        if current_ticks is not None:
            live_checks = (
                _proc_command(pid) == tuple(command),
                _proc_visible_gpu(pid) == expected_visible,
                _proc_cwd(pid) == str(layout.root.resolve()),
                _proc_environment_value(pid, "RC_IRSTD_PHASE3_LAUNCH_ID")
                == attempt_id,
                os.getpgid(pid) == pid,
                os.getsid(pid) == pid,
            )
            if not all(live_checks):
                raise RuntimeError(f"{stage} ACTIVE live process identity mismatch")
    else:
        intents = list(stage_dir.glob("*.intent.json"))
        for intent_path in intents:
            attempt = intent_path.name.removesuffix(".intent.json")
            binding_path = stage_dir / f"{attempt}.binding.json"
            result_path = stage_dir / f"{attempt}.result.json"
            if not binding_path.is_file():
                raise RuntimeError(f"{stage} has ambiguous intent without PID binding")
            if not result_path.is_file():
                raise RuntimeError(f"{stage} has unowned PID binding without result")
            raise RuntimeError(
                f"{stage} has a finished attempt but no valid complete artifact; fail closed"
            )
        attempt_id = uuid.uuid4().hex
        expected_visible = str(spec.physical_gpu) if uses_gpu else ""
        intent_path = stage_dir / f"{attempt_id}.intent.json"
        binding_path = stage_dir / f"{attempt_id}.binding.json"
        log_path = (stage_dir / f"{attempt_id}.log").resolve()
        intent = {
            "schema_version": "rc-irstd-phase3-stage-launch-v1",
            "attempt_id": attempt_id,
            "stage": stage,
            "run_id": spec.run_id,
            "command": list(command),
            "command_sha256": _canonical_json_sha256(command),
            "uses_gpu": uses_gpu,
            "cuda_visible_devices": expected_visible,
            "logical_device": "cuda:0" if uses_gpu else None,
            "created_at": now(),
        }
        _write_once_json(intent_path, intent, readonly=True)
        environment = dict(os.environ)
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = expected_visible
        environment["RC_IRSTD_PHASE3_LAUNCH_ID"] = attempt_id
        log_handle = log_path.open("xb")
        try:
            handle = subprocess.Popen(
                list(command),
                cwd=layout.root,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            log_handle.close()
            raise
        try:
            ticks = _proc_start_ticks(handle.pid)
        except FileNotFoundError as error:
            log_handle.close()
            raise RuntimeError(f"{stage} exited before PID identity binding") from error
        binding = {
            **intent,
            "pid": handle.pid,
            "proc_start_ticks": ticks,
            "process_group_id": handle.pid,
            "session_id": handle.pid,
            "cwd": str(layout.root.resolve()),
            "intent_sha256": file_sha256(intent_path),
            "log": str(log_path),
            "started_at": now(),
        }
        _write_once_json(binding_path, binding, readonly=True)
        atomic_write_json(active_path, binding)
    pid = int(binding["pid"])
    ticks = int(binding["proc_start_ticks"])
    while True:
        if handle is not None:
            return_code = handle.poll()
            alive = return_code is None
        else:
            if uses_gpu:
                alive = _pid_matches_launch(
                    pid, ticks, command, spec.physical_gpu
                )
            else:
                try:
                    alive = _proc_start_ticks(pid) == ticks
                except (FileNotFoundError, ProcessLookupError):
                    alive = False
        if not alive:
            break
        time.sleep(5)
    if log_handle is not None:
        log_handle.close()
    log_path = Path(str(binding["log"])).resolve()
    result_path = stage_dir / f"{attempt_id}.result.json"
    if result_path.is_file():
        result = _load_json(result_path)
        if (
            result.get("attempt_id") != attempt_id
            or result.get("stage") != stage
            or result.get("run_id") != spec.run_id
            or result.get("pid") != pid
            or result.get("log_sha256") != file_sha256(log_path)
        ):
            raise RuntimeError(f"{stage} existing result identity mismatch")
        if active_path.is_file() and _load_json(active_path).get("attempt_id") == attempt_id:
            active_path.unlink()
        if result.get("return_code") not in {0, None}:
            raise RuntimeError(f"{stage} prior attempt failed cleanly")
        return {
            "attempt_id": attempt_id,
            "binding": str((stage_dir / f"{attempt_id}.binding.json").resolve()),
            "binding_sha256": file_sha256(
                stage_dir / f"{attempt_id}.binding.json"
            ),
            "result": str(result_path.resolve()),
            "result_sha256": file_sha256(result_path),
        }
    result = {
        "schema_version": "rc-irstd-phase3-stage-result-v1",
        "attempt_id": attempt_id,
        "stage": stage,
        "run_id": spec.run_id,
        "pid": pid,
        "return_code": return_code,
        "return_code_observable": return_code is not None,
        "log_sha256": file_sha256(log_path),
        "observed_at": now(),
    }
    _write_once_json(result_path, result, readonly=True)
    if active_path.is_file() and _load_json(active_path).get("attempt_id") == attempt_id:
        active_path.unlink()
    if return_code not in {0, None}:
        raise RuntimeError(
            f"{stage} failed for {spec.run_id} with rc={return_code}; log={log_path}"
        )
    return {
        "attempt_id": attempt_id,
        "binding": str((stage_dir / f"{attempt_id}.binding.json").resolve()),
        "binding_sha256": file_sha256(stage_dir / f"{attempt_id}.binding.json"),
        "result": str(result_path.resolve()),
        "result_sha256": file_sha256(result_path),
    }


def export_command(layout: Layout, spec: RunSpec) -> list[str]:
    return [
        sys.executable,
        str(layout.root / "export_scores.py"),
        "--checkpoint",
        str(_run_dir(layout, spec) / "last.pt"),
        "--dataset-dir",
        str(_validated_dataset_root(layout, spec.fold.held_out_dataset_dir)),
        "--dataset-name",
        spec.fold.held_out_name,
        "--source-dataset",
        spec.fold.train_name,
        "--split",
        "train",
        "--output-dir",
        str(_score_dir(layout, spec)),
        "--device",
        "cuda:0",
        "--pad-multiple",
        "16",
        "--batch-size",
        "1",
        "--labels-loaded",
        "--export-raw-logits",
    ]


def _stage_active_path(layout: Layout, spec: RunSpec, stage: str) -> Path:
    return _run_dir(layout, spec) / "stage_attempts" / stage / "ACTIVE.json"


def _stage_provenance(
    layout: Layout,
    spec: RunSpec,
    stage: str,
    command: Sequence[str],
) -> dict[str, Any]:
    stage_dir = _run_dir(layout, spec) / "stage_attempts" / stage
    if _stage_active_path(layout, spec, stage).exists():
        raise RuntimeError(f"Cannot freeze {stage} while its child is ACTIVE")
    candidates: list[dict[str, Any]] = []
    for result_path in stage_dir.glob("*.result.json") if stage_dir.exists() else ():
        result = _load_json(result_path)
        attempt_id = result.get("attempt_id")
        intent_path = stage_dir / f"{attempt_id}.intent.json"
        binding_path = stage_dir / f"{attempt_id}.binding.json"
        if not intent_path.is_file() or not binding_path.is_file():
            continue
        intent = _load_json(intent_path)
        binding = _load_json(binding_path)
        log_path = Path(str(binding.get("log", ""))).resolve()
        if (
            intent.get("command") != list(command)
            or binding.get("command") != list(command)
            or binding.get("intent_sha256") != file_sha256(intent_path)
            or result.get("return_code") not in {0, None}
            or not log_path.is_file()
            or result.get("log_sha256") != file_sha256(log_path)
        ):
            continue
        candidates.append(
            {
                "attempt_id": attempt_id,
                "intent": str(intent_path.resolve()),
                "intent_sha256": file_sha256(intent_path),
                "binding": str(binding_path.resolve()),
                "binding_sha256": file_sha256(binding_path),
                "result": str(result_path.resolve()),
                "result_sha256": file_sha256(result_path),
            }
        )
    if len(candidates) != 1:
        raise RuntimeError(
            f"{stage} requires exactly one successful registered attempt"
        )
    return candidates[0]


def _validate_export(layout: Layout, spec: RunSpec) -> dict[str, Any]:
    root = _score_dir(layout, spec)
    manifest, _, integrity = verify_score_map_directory(
        root, require_integrity=True, require_masks=True
    )
    contract = validate_formal_score_manifest(
        manifest, integrity, expected_split_role="train"
    )
    if manifest is None:
        raise AssertionError("Verified export lost its manifest")
    expected = {
        "target_dataset": spec.fold.held_out_name,
        "source_datasets": [spec.fold.train_name],
        "labels_loaded": True,
        "spatial_mode": "native",
        "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
        "probability_dtype": PROBABILITY_DTYPE,
        "logit_dtype": RAW_LOGIT_DTYPE,
        "inference_autocast_enabled": False,
        "checkpoint_selection_rule": "fixed_last",
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"Export {spec.run_id} has {field}={manifest.get(field)!r}")
    identity = _load_json(_run_dir(layout, spec) / "PHASE3_IDENTITY.json")
    if contract["detector_weight_sha256"] != identity["checkpoint_sha256"]:
        raise ValueError("Export detector checkpoint binding mismatch")
    return {
        "manifest_sha256": integrity["manifest_sha256"],
        "records_sha256": integrity["records_sha256"],
        "ordered_image_ids_sha256": integrity["ordered_image_ids_sha256"],
        "detector_weight_sha256": contract["detector_weight_sha256"],
    }


def ensure_export(layout: Layout, spec: RunSpec) -> dict[str, Any]:
    score_dir = _score_dir(layout, spec)
    command = export_command(layout, spec)
    active = _stage_active_path(layout, spec, "export").is_file()
    if active or not score_dir.exists():
        _run_checked(layout, spec, command, stage="export", uses_gpu=True)
    if not (score_dir / "manifest.json").is_file():
        raise RuntimeError(
            f"Incomplete export is preserved and fails closed: {score_dir}"
        )
    validation = _validate_export(layout, spec)
    identity = {
        "schema_version": "rc-irstd-phase3-export-identity-v1",
        "run_id": spec.run_id,
        "source_only": True,
        "held_out_pseudo_target": spec.fold.held_out_name,
        "checkpoint_sha256": file_sha256(_run_dir(layout, spec) / "last.pt"),
        "command": command,
        "launch_provenance": _stage_provenance(
            layout, spec, "export", command
        ),
        "output": validation,
    }
    _write_once_json(
        _run_dir(layout, spec) / "EXPORT_IDENTITY.json",
        identity,
        readonly=True,
    )
    (score_dir / "manifest.json").chmod(0o444)
    return validation


def sweep_command(layout: Layout, spec: RunSpec) -> list[str]:
    return [
        sys.executable,
        "-m",
        "evaluation.threshold_sweep",
        "--score-dir",
        str(_score_dir(layout, spec)),
        "--formal",
        "--expected-split-role",
        "train",
        "--matching-rule",
        "overlap",
        "--connectivity",
        "2",
        "--min-component-area",
        "1",
        "--output",
        str(_curve_path(layout, spec)),
    ]


def _validate_curve(layout: Layout, spec: RunSpec) -> dict[str, Any]:
    curve = load_formal_curve(_curve_path(layout, spec), expected_split_role="train")
    identity = _load_json(_run_dir(layout, spec) / "PHASE3_IDENTITY.json")
    if curve.metadata["detector_weight_sha256"] != identity["checkpoint_sha256"]:
        raise ValueError("Threshold curve detector binding mismatch")
    if domain_key(curve.metadata["target_dataset"]) != domain_key(
        spec.fold.held_out_name
    ):
        raise ValueError("Threshold curve pseudo-target mismatch")
    if curve.metadata["source_datasets"] != [spec.fold.train_name]:
        raise ValueError("Threshold curve source-domain mismatch")
    return {
        "curve_sha256": curve.metadata["curve_sha256"],
        "metadata_sha256": file_sha256(curve.metadata_path),
        "threshold_grid_sha256": curve.metadata["threshold_grid_sha256"],
    }


def ensure_sweep(layout: Layout, spec: RunSpec) -> dict[str, Any]:
    curve = _curve_path(layout, spec)
    metadata = curve.with_name(curve.name + ".metadata.json")
    command = sweep_command(layout, spec)
    active = _stage_active_path(layout, spec, "threshold_sweep").is_file()
    if active or (not curve.exists() and not metadata.exists()):
        _run_checked(
            layout,
            spec,
            command,
            stage="threshold_sweep",
            uses_gpu=False,
        )
    if not curve.exists() or not metadata.exists():
        raise RuntimeError(
            f"Incomplete threshold sweep is preserved and fails closed: {curve}"
        )
    validation = _validate_curve(layout, spec)
    identity = {
        "schema_version": "rc-irstd-phase3-threshold-sweep-identity-v1",
        "run_id": spec.run_id,
        "source_only": True,
        "export_identity_sha256": file_sha256(
            _run_dir(layout, spec) / "EXPORT_IDENTITY.json"
        ),
        "command": command,
        "launch_provenance": _stage_provenance(
            layout, spec, "threshold_sweep", command
        ),
        "output": validation,
    }
    _write_once_json(
        _run_dir(layout, spec) / "THRESHOLD_SWEEP_IDENTITY.json",
        identity,
        readonly=True,
    )
    curve.chmod(0o444)
    metadata.chmod(0o444)
    return validation


def source_operating_command(
    layout: Layout, role: str, budget: tuple[str, float, float]
) -> list[str]:
    name, pixel, component = budget
    nudt = RunSpec(role, "heldout_nudt", 2 if role == "control" else 3)
    irstd = RunSpec(role, "heldout_irstd", 2 if role == "control" else 3)
    return [
        sys.executable,
        "-m",
        "evaluation.source_operating_point",
        "--source-curve",
        f"NUDT-SIRST={_curve_path(layout, nudt)}",
        "--source-curve",
        f"IRSTD-1K={_curve_path(layout, irstd)}",
        "--pixel-budget",
        str(pixel),
        "--component-budget",
        str(component),
        "--output",
        str(_selection_path(layout, role, name)),
    ]


def _validate_source_operating(
    layout: Layout, role: str, budget: tuple[str, float, float]
) -> dict[str, Any]:
    name, pixel, component = budget
    path = _selection_path(layout, role, name)
    payload = _load_json(path)
    expected = {
        "selection_is_source_only": True,
        "pseudo_target_lodo_closure_verified": True,
        "target_curve_used_for_selection": False,
        "pixel_budget": pixel,
        "component_budget": component,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"Source selection {role}/{name} violates {field}")
    target = payload.get("target_evaluation")
    if (
        not isinstance(target, Mapping)
        or target.get("provided") is not False
        or target.get("target_labels_used") is not False
        or target.get("target_labels_used_for_selection") is not False
    ):
        raise ValueError("Source selection unexpectedly references outer-target labels")
    if set(payload.get("meta_domain_keys", [])) != {"nudt", "irstd1k"}:
        raise ValueError("Source selection meta-domain closure changed")
    return payload


def ensure_source_operating(
    layout: Layout, role: str, budget: tuple[str, float, float]
) -> dict[str, Any]:
    name = budget[0]
    path = _selection_path(layout, role, name)
    representative = RunSpec(role, "heldout_nudt", 2 if role == "control" else 3)
    stage = f"source_operating_{name}"
    command = source_operating_command(layout, role, budget)
    active = _stage_active_path(layout, representative, stage).is_file()
    if active or not path.exists():
        _run_checked(
            layout,
            representative,
            command,
            stage=stage,
            uses_gpu=False,
        )
    if not path.is_file():
        raise RuntimeError(
            f"Incomplete source operating-point stage fails closed: {path}"
        )
    validation = _validate_source_operating(layout, role, budget)
    identity = {
        "schema_version": "rc-irstd-phase3-source-operating-identity-v1",
        "role": role,
        "budget": {"name": name, "pixel": budget[1], "component": budget[2]},
        "source_only": True,
        "curve_identity_sha256": {
            spec.fold_key: file_sha256(
                _run_dir(layout, spec) / "THRESHOLD_SWEEP_IDENTITY.json"
            )
            for spec in (
                RunSpec(role, "heldout_nudt", representative.physical_gpu),
                RunSpec(role, "heldout_irstd", representative.physical_gpu),
            )
        },
        "command": command,
        "launch_provenance": _stage_provenance(
            layout, representative, stage, command
        ),
        "output_sha256": file_sha256(path),
    }
    _write_once_json(
        path.with_name(path.name + ".identity.json"),
        identity,
        readonly=True,
    )
    path.chmod(0o444)
    return validation


def _selection_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    results = payload.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("Source selection has no results")
    pooled = results.get("source_pooled")
    worst = results.get("source_worst")
    if not isinstance(pooled, Mapping) or not isinstance(worst, Mapping):
        raise ValueError("Source selection has incomplete pooled/worst results")
    if not bool(pooled.get("found")) or not bool(worst.get("found")):
        return {"found": False}
    pooled_point = pooled.get("operating_point")
    source_rows = pooled.get("source_rows")
    if not isinstance(pooled_point, Mapping) or not isinstance(source_rows, Mapping):
        raise ValueError("Found source selection lacks operating-point rows")
    domain_pd = {
        domain_key(name): float(row["pd"])
        for name, row in source_rows.items()
        if isinstance(row, Mapping)
    }
    if set(domain_pd) != {"nudt", "irstd1k"}:
        raise ValueError("Source selection does not cover both pseudo-targets")
    return {
        "found": True,
        "pooled_pd": float(pooled_point["pd"]),
        "worst_pd": float(worst["worst_domain_pd"]),
        "macro_pd": sum(domain_pd.values()) / len(domain_pd),
        "domain_pd": domain_pd,
        "pooled_threshold": float(pooled_point["threshold"]),
    }


def build_tier1_decision(
    selections: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    failures: list[str] = []
    for name, _, _ in BUDGETS:
        control = _selection_metrics(selections["control"][name])
        full = _selection_metrics(selections["full"][name])
        record: dict[str, Any] = {"control": control, "full": full}
        if not control["found"] or not full["found"]:
            failures.append(f"{name}: missing pooled or worst feasible point")
            record["pooled_pd_delta"] = None
            record["worst_pd_delta"] = None
            record["macro_pd_delta"] = None
            record["domain_pd_delta"] = None
        else:
            record["pooled_pd_delta"] = full["pooled_pd"] - control["pooled_pd"]
            record["worst_pd_delta"] = full["worst_pd"] - control["worst_pd"]
            record["macro_pd_delta"] = full["macro_pd"] - control["macro_pd"]
            record["domain_pd_delta"] = {
                key: full["domain_pd"][key] - control["domain_pd"][key]
                for key in sorted(control["domain_pd"])
            }
            if record["pooled_pd_delta"] < -NUMERIC_ATOL:
                failures.append(f"{name}: pooled Pd degraded")
            if record["worst_pd_delta"] < -NUMERIC_ATOL:
                failures.append(f"{name}: worst-source Pd degraded")
        evidence[name] = record
    strict = evidence["strict"]
    if strict["macro_pd_delta"] is None or strict["macro_pd_delta"] < (
        STRICT_MACRO_PD_GAIN - NUMERIC_ATOL
    ):
        failures.append("strict: macro matched-FA Pd gain is below 1 percentage point")
    domain_delta = strict.get("domain_pd_delta")
    if not isinstance(domain_delta, Mapping) or any(
        float(value) < -NUMERIC_ATOL for value in domain_delta.values()
    ):
        failures.append("strict: gain is not non-degraded on both pseudo-targets")
    go = not failures
    return {
        "schema_version": DECISION_SCHEMA,
        "decision": "GO" if go else "HOLD",
        "scope": "source_only_inner_lodo_tier1",
        "authorizes_tier2": go,
        "authorizes_outer_target_label_access": False,
        "outer_target_labels_used": False,
        "outer_target_images_used": False,
        "source_official_train_labels_used_for_pseudo_target_audit": True,
        "not_an_outer_target_claim": True,
        "matched_pd_component_fa_alternative_enabled": False,
        "criteria": {
            "strict_macro_pd_gain_minimum": STRICT_MACRO_PD_GAIN,
            "strict_each_source_pd_non_degraded": True,
            "all_budget_pooled_pd_non_degraded": True,
            "all_budget_worst_pd_non_degraded": True,
        },
        "evidence": evidence,
        "failure_reasons": failures,
    }


def tier1_input_bindings(layout: Layout, preregistration: Path) -> dict[str, Any]:
    """Bind the frozen decision to every upstream file used to derive it."""

    current_coordinator_sha256 = file_sha256(Path(__file__).resolve())
    if _load_json(preregistration).get("coordinator_sha256") != current_coordinator_sha256:
        raise RuntimeError("Coordinator code changed after Phase3 preregistration")
    runs: dict[str, Any] = {}
    for round_specs in TIER1_ROUNDS:
        for spec in round_specs:
            run_dir = _run_dir(layout, spec)
            identity_path = run_dir / "PHASE3_IDENTITY.json"
            curve_path = _curve_path(layout, spec)
            curve_metadata = curve_path.with_name(curve_path.name + ".metadata.json")
            identity = _load_json(identity_path)
            if identity.get("checkpoint_sha256") != file_sha256(run_dir / "last.pt"):
                raise RuntimeError(f"Tier1 run changed before decision freeze: {spec.run_id}")
            runs[spec.run_id] = {
                "identity": str(identity_path.resolve()),
                "identity_sha256": file_sha256(identity_path),
                "checkpoint_sha256": identity["checkpoint_sha256"],
                "curve": str(curve_path.resolve()),
                "curve_sha256": file_sha256(curve_path),
                "curve_metadata": str(curve_metadata.resolve()),
                "curve_metadata_sha256": file_sha256(curve_metadata),
                "export_identity_sha256": file_sha256(
                    run_dir / "EXPORT_IDENTITY.json"
                ),
                "threshold_sweep_identity_sha256": file_sha256(
                    run_dir / "THRESHOLD_SWEEP_IDENTITY.json"
                ),
            }
    selections = {
        role: {
            name: {
                "path": str(_selection_path(layout, role, name).resolve()),
                "sha256": file_sha256(_selection_path(layout, role, name)),
                "identity_sha256": file_sha256(
                    _selection_path(layout, role, name).with_name(
                        _selection_path(layout, role, name).name
                        + ".identity.json"
                    )
                ),
            }
            for name, _, _ in BUDGETS
        }
        for role in ("control", "full")
    }
    return {
        "coordinator": str(Path(__file__).resolve()),
        "coordinator_sha256": current_coordinator_sha256,
        "phase2_status": str(layout.phase2_status.resolve()),
        "phase2_status_sha256": file_sha256(layout.phase2_status),
        "preregistration": str(preregistration.resolve()),
        "preregistration_sha256": file_sha256(preregistration),
        "runs": runs,
        "source_operating_points": selections,
    }


def freeze_decision(layout: Layout, decision: Mapping[str, Any]) -> Path:
    path = layout.audit / "tier1_decision.json"
    _write_once_json(path, decision, readonly=True)
    digest = layout.audit / "tier1_decision.sha256"
    line = f"{file_sha256(path)}  {path.name}\n"
    if digest.exists() and digest.read_text(encoding="utf-8") != line:
        raise RuntimeError("Tier1 decision digest drift")
    if not digest.exists():
        _atomic_write_text(digest, line)
        digest.chmod(0o444)
    go = decision.get("decision") == "GO"
    chosen = layout.state / (
        "PHASE3_SOURCE_TIER1_GO" if go else "PHASE3_SOURCE_TIER1_HOLD"
    )
    opposite = layout.state / (
        "PHASE3_SOURCE_TIER1_HOLD" if go else "PHASE3_SOURCE_TIER1_GO"
    )
    if opposite.exists():
        raise RuntimeError("Conflicting frozen Phase3 decision sentinel")
    expected_sentinel = f"{decision['decision']} {file_sha256(path)}\n"
    if chosen.exists():
        if chosen.read_text(encoding="utf-8") != expected_sentinel:
            raise RuntimeError("Frozen Phase3 decision sentinel content drifted")
    else:
        _atomic_write_text(chosen, expected_sentinel)
    # The outer-target hold is intentionally retained even after source-only GO.
    hold = layout.state / "HOLD_PHASE3_TARGET_LABEL_ACCESS"
    if hold.exists():
        if hold.read_text(encoding="utf-8") != "HOLD\n":
            raise RuntimeError("Outer-target HOLD sentinel content drifted")
    else:
        _atomic_write_text(hold, "HOLD\n")
    return path


def _failure_record(layout: Layout, error: BaseException) -> None:
    directory = layout.audit / "failures"
    directory.mkdir(parents=True, exist_ok=True)
    name = (
        f"failure_{datetime.now().strftime('%Y%m%dT%H%M%S')}_"
        f"{os.getpid()}_{uuid.uuid4().hex}.json"
    )
    atomic_write_json(
        directory / name,
        {
            "schema_version": FAILURE_SCHEMA,
            "status": "failed_closed",
            "error_type": type(error).__name__,
            "error": str(error),
            "outer_target_labels_used": False,
            "target_label_gate_state": "HOLD",
            "recorded_at": now(),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    if PROJECT_ROOT.resolve() != Path("/home/ly/RC-IRSTD-v2"):
        raise RuntimeError("Phase3 coordinator is pinned to /home/ly/RC-IRSTD-v2")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError(
            f"Phase3 coordinator requires {EXPECTED_PYTHON}, got {sys.executable}"
        )
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    layout = Layout()
    layout.state.mkdir(parents=True, exist_ok=True)
    layout.audit.mkdir(parents=True, exist_ok=True)
    hold = layout.state / "HOLD_PHASE3_TARGET_LABEL_ACCESS"
    if hold.exists():
        if hold.read_text(encoding="utf-8") != "HOLD\n":
            raise RuntimeError("Phase3 target-label HOLD sentinel content drifted")
    else:
        _atomic_write_text(hold, "HOLD\n")
    lock_path = layout.state / "phase3_source_lodo_coordinator.lock"
    pid_path = layout.state / "phase3_source_lodo_coordinator.pid"
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError("another Phase3 source-LODO coordinator owns the lock") from error
    try:
        _atomic_write_text(pid_path, f"{os.getpid()}\n")
        log(layout, f"coordinator started pid={os.getpid()} python={sys.executable}")
        atomic_write_json(
            layout.audit / "phase3_status.json",
            {
                "schema_version": PHASE3_SCHEMA,
                "status": "waiting_for_verified_phase2",
                "outer_target_labels_used": False,
                "outer_target_label_access": "HOLD",
                "coordinator_pid": os.getpid(),
            },
        )
        _wait_for_phase2(layout, args.poll_seconds)
        preregistration = _ensure_preregistration(layout)
        atomic_write_json(
            layout.audit / "phase3_status.json",
            {
                "schema_version": PHASE3_SCHEMA,
                "status": "tier1_running",
                "outer_target_labels_used": False,
                "outer_target_label_access": "HOLD",
                "physical_gpus": [2, 3],
                "logical_device": "cuda:0",
                "coordinator_pid": os.getpid(),
            },
        )
        for index, round_specs in enumerate(TIER1_ROUNDS, start=1):
            run_training_round(layout, f"tier1_round{index}", round_specs)
            for spec in round_specs:
                ensure_export(layout, spec)
                ensure_sweep(layout, spec)
        selections: dict[str, dict[str, dict[str, Any]]] = {
            "control": {},
            "full": {},
        }
        for role in selections:
            for budget in BUDGETS:
                selections[role][budget[0]] = ensure_source_operating(
                    layout, role, budget
                )
        decision = build_tier1_decision(selections)
        decision["input_bindings"] = tier1_input_bindings(
            layout, preregistration
        )
        decision_path = freeze_decision(layout, decision)
        atomic_write_json(
            layout.audit / "phase3_status.json",
            {
                "schema_version": PHASE3_SCHEMA,
                "status": "tier1_completed",
                "decision": decision["decision"],
                "decision_artifact": str(decision_path),
                "outer_target_labels_used": False,
                "outer_target_label_access": "HOLD",
                "next_stage": (
                    "tier2_no_contrast_no_component_source_lodo"
                    if decision["decision"] == "GO"
                    else "scientific_hold_or_rescue"
                ),
            },
        )
        log(layout, f"TIER1_{decision['decision']} frozen at {decision_path}")
        return 0
    except BaseException as error:
        if hold.exists() and hold.read_text(encoding="utf-8") != "HOLD\n":
            raise RuntimeError(
                "Phase3 target-label HOLD sentinel content drifted"
            ) from error
        if not hold.exists():
            _atomic_write_text(hold, "HOLD\n")
        try:
            _failure_record(layout, error)
            atomic_write_json(
                layout.audit / "phase3_status.json",
                {
                    "schema_version": PHASE3_SCHEMA,
                    "status": "failed_closed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "outer_target_labels_used": False,
                    "outer_target_label_access": "HOLD",
                },
            )
            log(layout, f"FAILED_CLOSED {type(error).__name__}: {error}")
        except Exception:
            pass
        raise
    finally:
        pid_path.unlink(missing_ok=True)
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
