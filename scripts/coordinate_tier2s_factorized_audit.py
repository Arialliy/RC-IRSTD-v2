#!/usr/bin/env python3
"""Coordinate the isolated source-only Tier2S factorized causal audit.

This is an exploratory audit, not a continuation of the frozen Tier2R gate.
It replays only immutable control/C checkpoints on the two source datasets,
uses two fixed FIFO GPU lanes on physical GPUs 0/1, and delegates all numerical conclusions to a
separate evaluator.  It can never authorize Tier3 or outer-target access.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import register_aaai27_governance_v1 as governance  # noqa: E402

_require_frozen_tier2s_governance = governance.require_frozen_tier2s_governance
PROTOCOL_PATH = PROJECT_ROOT / "configs/tier2s_factorized_causal_audit_v1.json"
PARENT_HANDOFF = (
    PROJECT_ROOT
    / "artifacts/aaai27/audit/component_rescue/tier2r_c_v1_impl_erratum1"
    / "TIER2R_HANDOFF.json"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/aaai27/source_rescue/tier2s_factorized_causal_audit_v1"
)
AUDIT_ROOT = (
    PROJECT_ROOT
    / "artifacts/aaai27/audit/source_rescue/tier2s_factorized_causal_audit_v1"
)
EXPORTER = PROJECT_ROOT / "scripts/export_tier2s_factorized_logits.py"
EVALUATOR = PROJECT_ROOT / "scripts/evaluate_tier2s_factorized_audit.py"
PREREGISTRATION = AUDIT_ROOT / "PREREGISTRATION.json"
QUEUE_MANIFEST = AUDIT_ROOT / "QUEUE_MANIFEST.json"
TARGET_LOCK = AUDIT_ROOT / "OUTER_TARGET_ACCESS_DENIED.json"
STATUS_PATH = AUDIT_ROOT / "STATUS.json"
HANDOFF_PATH = AUDIT_ROOT / "EXPORT_HANDOFF.json"
EVENT_LOG = AUDIT_ROOT / "scheduler_events.jsonl"
EVENT_LOG_SIDECAR = EVENT_LOG.with_suffix(EVENT_LOG.suffix + ".sha256")
LOCK_PATH = AUDIT_ROOT / ".coordinator.lock"

PROTOCOL_ID = "tier2s_factorized_causal_audit_v1"
SCHEMA = "rc-irstd-aaai27-tier2s-factorized-coordinator-v1"
PHYSICAL_GPUS = (0, 1)
SEEDS = (43, 44, 45)
FOLDS = ("heldout_nudt", "heldout_irstd")
ROLES = ("control", "c")
PYTHON_EXECUTABLE = Path(sys.executable).resolve()


@dataclass(frozen=True)
class ExportJob:
    run_id: str
    checkpoint_run_id: str
    seed: int
    role: str
    fold: str
    scope: str
    dataset_name: str
    dataset_root: str
    checkpoint: str
    checkpoint_sha256: str
    parent_heldout_score_manifest: str
    parent_heldout_score_manifest_sha256: str
    physical_gpu: int
    queue_index: int
    output_dir: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required regular JSON file is absent: {path}")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, _canonical_bytes(dict(value)))


def _write_once_json(path: Path, value: Mapping[str, Any]) -> str:
    raw = _canonical_bytes(dict(value))
    digest = hashlib.sha256(raw).hexdigest()
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise RuntimeError(f"immutable artifact drift: {path}")
    else:
        _atomic_write(path, raw)
        path.chmod(0o444)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if sidecar.exists() or sidecar.is_symlink():
        if sidecar.is_symlink() or sidecar.read_bytes() != expected:
            raise RuntimeError(f"immutable digest sidecar drift: {sidecar}")
    else:
        _atomic_write(sidecar, expected)
        sidecar.chmod(0o444)
    return digest


def _relative_regular(path_value: str, *, expected_sha256: str) -> Path:
    raw = Path(path_value)
    if raw.is_absolute() or ".." in raw.parts:
        raise RuntimeError(f"protocol evidence path is not project-relative: {raw}")
    path = PROJECT_ROOT / raw
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise RuntimeError(f"protocol evidence is absent or aliased: {path}")
    if _sha256(path) != expected_sha256:
        raise RuntimeError(f"protocol evidence SHA-256 drift: {path}")
    return path


def _verify_parent(protocol: Mapping[str, Any]) -> dict[str, Any]:
    parent = protocol.get("immutable_parent_evidence")
    if not isinstance(parent, Mapping):
        raise RuntimeError("immutable_parent_evidence is absent")
    decision_spec = parent.get("tier2r_decision")
    source_spec = parent.get("source_authorization")
    target_spec = parent.get("outer_target_authorization")
    if not all(isinstance(value, Mapping) for value in (decision_spec, source_spec, target_spec)):
        raise RuntimeError("parent evidence bindings are incomplete")
    decision_path = _relative_regular(
        str(decision_spec["path"]), expected_sha256=str(decision_spec["sha256"])
    )
    source_path = _relative_regular(
        str(source_spec["path"]), expected_sha256=str(source_spec["sha256"])
    )
    target_path = _relative_regular(
        str(target_spec["path"]), expected_sha256=str(target_spec["sha256"])
    )
    decision = _load_json(decision_path)
    source = _load_json(source_path)
    target = _load_json(target_path)
    if (
        decision.get("decision") != "TIER2R_HOLD"
        or decision.get("selected_candidate") is not None
        or decision.get("gate_valid") is not True
        or decision.get("authorizes_source_tier3_design") is not False
        or decision.get("authorizes_outer_target_access") is not False
        or source.get("source_tier3_design_authorized") is not False
        or target.get("outer_target_access_authorized") is not False
    ):
        raise RuntimeError("frozen Tier2R HOLD/authorization semantics drifted")
    return {
        "decision": {"path": str(decision_path), "sha256": _sha256(decision_path)},
        "source_authorization": {"path": str(source_path), "sha256": _sha256(source_path)},
        "outer_target_authorization": {"path": str(target_path), "sha256": _sha256(target_path)},
    }


def _expected_lane(seed: int, role: str, fold: str, scope: str) -> int:
    # Frozen balanced 9/9 assignment over the exact GPU 0/1 allowlist.
    within_seed = {
        ("c", "heldout_nudt", "held_out"): 0,
        ("c", "heldout_irstd", "held_out"): 1,
        ("c", "heldout_nudt", "held_in"): 0,
        ("c", "heldout_irstd", "held_in"): 1,
        ("control", "heldout_nudt", "held_in"): 0,
        ("control", "heldout_irstd", "held_in"): 1,
    }
    base = within_seed[(role, fold, scope)]
    return (base + (seed - 43)) % 2


def _build_jobs(protocol: Mapping[str, Any], handoff: Mapping[str, Any]) -> list[ExportJob]:
    folds = protocol.get("folds")
    runs = handoff.get("runs")
    if not isinstance(folds, Mapping) or not isinstance(runs, Mapping):
        raise RuntimeError("protocol folds or parent handoff runs are absent")
    lane_counts = {gpu: 0 for gpu in PHYSICAL_GPUS}
    jobs: list[ExportJob] = []
    for seed in SEEDS:
        specs = (
            ("c", "heldout_nudt", "held_out"),
            ("c", "heldout_irstd", "held_out"),
            ("c", "heldout_nudt", "held_in"),
            ("c", "heldout_irstd", "held_in"),
            ("control", "heldout_nudt", "held_in"),
            ("control", "heldout_irstd", "held_in"),
        )
        for role, fold, scope in specs:
            checkpoint_run_id = f"seed{seed}_{role}_{fold}"
            parent = runs.get(checkpoint_run_id)
            fold_spec = folds.get(fold)
            if not isinstance(parent, Mapping) or not isinstance(fold_spec, Mapping):
                raise RuntimeError(f"missing frozen parent binding: {checkpoint_run_id}")
            checkpoint = Path(str(parent.get("checkpoint", "")))
            score_manifest = Path(str(parent.get("score_manifest", "")))
            for path, expected, name in (
                (checkpoint, parent.get("checkpoint_sha256"), "checkpoint"),
                (score_manifest, parent.get("score_manifest_sha256"), "score manifest"),
            ):
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.resolve() != path
                    or _sha256(path) != expected
                ):
                    raise RuntimeError(f"parent {name} drift: {checkpoint_run_id}")
            dataset_name = str(
                fold_spec["training_source"] if scope == "held_in" else fold_spec["held_out_source"]
            )
            dataset_root = str(
                (PROJECT_ROOT / str(
                    fold_spec["training_root"] if scope == "held_in" else fold_spec["held_out_root"]
                )).resolve()
            )
            if dataset_name not in {"NUDT-SIRST", "IRSTD-1K"} or "NUAA" in dataset_root.upper():
                raise RuntimeError("outer target entered the Tier2S job set")
            gpu = _expected_lane(seed, role, fold, scope)
            queue_index = lane_counts[gpu]
            lane_counts[gpu] += 1
            run_id = f"seed{seed}_{role}_{fold}_{scope}"
            output = OUTPUT_ROOT / f"gpu{gpu}" / f"q{queue_index:02d}_{run_id}"
            jobs.append(
                ExportJob(
                    run_id=run_id,
                    checkpoint_run_id=checkpoint_run_id,
                    seed=seed,
                    role=role,
                    fold=fold,
                    scope=scope,
                    dataset_name=dataset_name,
                    dataset_root=dataset_root,
                    checkpoint=str(checkpoint),
                    checkpoint_sha256=str(parent["checkpoint_sha256"]),
                    parent_heldout_score_manifest=str(score_manifest),
                    parent_heldout_score_manifest_sha256=str(parent["score_manifest_sha256"]),
                    physical_gpu=gpu,
                    queue_index=queue_index,
                    output_dir=str(output),
                )
            )
    if len(jobs) != 18 or lane_counts != {0: 9, 1: 9}:
        raise AssertionError(f"frozen Tier2S lane construction failed: {lane_counts}")
    if len({job.run_id for job in jobs}) != 18:
        raise AssertionError("duplicate Tier2S run id")
    return jobs


def verify_prerequisites() -> dict[str, Any]:
    if PROJECT_ROOT.resolve() != PROJECT_ROOT or Path.cwd().resolve() != PROJECT_ROOT:
        raise RuntimeError(f"coordinator must run from canonical {PROJECT_ROOT}")
    governance_binding = _require_frozen_tier2s_governance()
    protocol = _load_json(PROTOCOL_PATH)
    if (
        protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("research_mode") != "exploratory_source_only"
        or protocol.get("scientific_limits", {}).get("outer_target_access_authorized") is not False
        or protocol.get("execution", {}).get("physical_gpus") != list(PHYSICAL_GPUS)
        or protocol.get("execution", {}).get("export_jobs") != 18
        or protocol.get("execution", {}).get("scheduler")
        != "two_fixed_independent_fifo_lanes"
        or protocol.get("scientific_limits", {}).get(
            "may_recommend_formal_training_or_fresh_seed_run"
        )
        is not False
        or protocol.get("scientific_limits", {}).get("result_use")
        != "failure_attribution_only"
        or protocol.get("governance_requirement", {}).get(
            "required_before_tier2s_preregistration_export_and_evaluation"
        )
        is not True
    ):
        raise RuntimeError("Tier2S protocol identity/lock drift")
    parent = _verify_parent(protocol)
    handoff = _load_json(PARENT_HANDOFF)
    if (
        handoff.get("source_only") is not True
        or handoff.get("outer_target_images_used") is not False
        or handoff.get("outer_target_labels_used") is not False
    ):
        raise RuntimeError("parent handoff is not source-only")
    jobs = _build_jobs(protocol, handoff)
    required_code = (
        Path(__file__).resolve(),
        EXPORTER,
        EVALUATOR,
        PROJECT_ROOT / "rc_irstd/models/rc_mshnet.py",
        PROJECT_ROOT / "evaluation/export_score_maps.py",
        PROJECT_ROOT / "evaluation/raw_logit_source_operating_point.py",
        PROJECT_ROOT / "evaluation/component_matching.py",
    )
    for path in required_code:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Tier2S code dependency missing: {path}")
    return {
        "schema_version": SCHEMA,
        "verified": True,
        "protocol_id": PROTOCOL_ID,
        "research_mode": "exploratory_source_only",
        "source_only": True,
        "outer_target_access_authorized": False,
        "governance_binding": governance_binding,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "source_tier3_authorized": False,
        "paper_claim_authorized": False,
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": _sha256(PROTOCOL_PATH)},
        "parent_evidence": parent,
        "parent_handoff": {"path": str(PARENT_HANDOFF), "sha256": _sha256(PARENT_HANDOFF)},
        "code": {str(path): _sha256(path) for path in required_code},
        "schedule": [asdict(job) for job in jobs],
        "lane_lengths": {str(gpu): sum(job.physical_gpu == gpu for job in jobs) for gpu in PHYSICAL_GPUS},
        "protocol_payload": protocol,
    }


def _public_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in packet.items() if key != "protocol_payload"}


def _registered_at() -> str:
    if not PREREGISTRATION.exists():
        return _now()
    prereg = _load_json(PREREGISTRATION)
    value = prereg.get("registered_at")
    if not isinstance(value, str) or not value:
        raise RuntimeError("existing Tier2S preregistration has no valid registered_at")
    return value


def _registration_payload(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **_public_packet(packet),
        "schema_version": "rc-irstd-aaai27-tier2s-factorized-preregistration-v1",
        "registered_at": _registered_at(),
        "decision": "REGISTER_EXPLORATORY_SOURCE_ONLY_FACTORIZED_AUDIT",
        "old_tier2r_hold_remains_immutable": True,
        "result_cannot_authorize_outer_or_tier3": True,
    }


def _tier2s_preregistration_binding(
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    sidecar = PREREGISTRATION.with_suffix(PREREGISTRATION.suffix + ".sha256")
    if (
        PREREGISTRATION.is_symlink()
        or not PREREGISTRATION.is_file()
        or sidecar.is_symlink()
        or not sidecar.is_file()
    ):
        raise RuntimeError("Tier2S preregistration binding is absent")
    digest = _sha256(PREREGISTRATION)
    expected = f"{digest}  {PREREGISTRATION.name}\n"
    if sidecar.read_text(encoding="ascii") != expected:
        raise RuntimeError("Tier2S preregistration sidecar drift")
    if PREREGISTRATION.stat().st_mode & 0o222 or sidecar.stat().st_mode & 0o222:
        raise RuntimeError("Tier2S preregistration binding is writable")
    governance_binding = packet.get("governance_binding")
    if not isinstance(governance_binding, Mapping):
        raise RuntimeError("Tier2S governance binding is absent")
    registration = governance_binding.get("registration")
    if not isinstance(registration, Mapping):
        raise RuntimeError("Tier2S governance registration binding is absent")
    return {
        "schema_version": "rc-irstd-aaai27-tier2s-preregistration-binding-v1",
        "protocol_id": PROTOCOL_ID,
        "path": str(PREREGISTRATION.relative_to(PROJECT_ROOT)),
        "sha256": digest,
        "sidecar_path": str(sidecar.relative_to(PROJECT_ROOT)),
        "sidecar_sha256": _sha256(sidecar),
        "governance_registration_sha256": registration.get("sha256"),
    }


def register(packet: Mapping[str, Any]) -> dict[str, Any]:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    prereg_sha = _write_once_json(PREREGISTRATION, _registration_payload(packet))
    tier2s_preregistration_binding = _tier2s_preregistration_binding(packet)
    queue_sha = _write_once_json(
        QUEUE_MANIFEST,
        {
            "schema_version": "rc-irstd-aaai27-tier2s-fixed-two-lane-queue-v1",
            "protocol_id": PROTOCOL_ID,
            "scheduler": "two_fixed_independent_fifo_lanes",
            "governance_binding": packet["governance_binding"],
            "tier2s_preregistration_binding": tier2s_preregistration_binding,
            "wait_for_idle_gpu": False,
            "allow_gpu_fallback": False,
            "jobs": packet["schedule"],
        },
    )
    target_sha = _write_once_json(
        TARGET_LOCK,
        {
            "schema_version": "rc-irstd-aaai27-tier2s-outer-target-denial-v1",
            "protocol_id": PROTOCOL_ID,
            "governance_binding": packet["governance_binding"],
            "tier2s_preregistration_binding": tier2s_preregistration_binding,
            "outer_target_access_authorized": False,
            "outer_target_images_used": False,
            "outer_target_labels_used": False,
            "source_tier3_authorized": False,
            "paper_claim_authorized": False,
            "parent_tier2r_hold_sha256": packet["parent_evidence"]["decision"]["sha256"],
        },
    )
    return {
        "registered": True,
        "preregistration_sha256": prereg_sha,
        "queue_manifest_sha256": queue_sha,
        "target_lock_sha256": target_sha,
        "tier2s_preregistration_binding": tier2s_preregistration_binding,
    }


def _verify_frozen_registered_json(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if (
        path.is_symlink()
        or not path.is_file()
        or sidecar.is_symlink()
        or not sidecar.is_file()
    ):
        raise RuntimeError(f"frozen Tier2S registration artifact is absent: {path}")
    digest = _sha256(path)
    if sidecar.read_text(encoding="ascii") != f"{digest}  {path.name}\n":
        raise RuntimeError(f"frozen Tier2S registration sidecar drift: {path}")
    if path.stat().st_mode & 0o222 or sidecar.stat().st_mode & 0o222:
        raise RuntimeError(f"frozen Tier2S registration became writable: {path}")
    return digest


def _verify_registration(packet: Mapping[str, Any]) -> dict[str, Any]:
    governance_binding = packet.get("governance_binding")
    if not isinstance(governance_binding, Mapping):
        raise RuntimeError("Tier2S packet governance binding is absent")
    registration = governance_binding.get("registration")
    if not isinstance(registration, Mapping):
        raise RuntimeError("Tier2S governance registration binding is absent")
    current_governance = _require_frozen_tier2s_governance(
        expected_registration_sha256=str(registration.get("sha256", ""))
    )
    if current_governance != dict(governance_binding):
        raise RuntimeError("Tier2S governance binding changed after verification")

    for path in (PREREGISTRATION, QUEUE_MANIFEST, TARGET_LOCK):
        _verify_frozen_registered_json(path)
    prereg = _load_json(PREREGISTRATION)
    queue = _load_json(QUEUE_MANIFEST)
    target = _load_json(TARGET_LOCK)
    expected_prereg = _registration_payload(packet)
    tier2s_preregistration_binding = _tier2s_preregistration_binding(packet)
    expected_queue = {
        "schema_version": "rc-irstd-aaai27-tier2s-fixed-two-lane-queue-v1",
        "protocol_id": PROTOCOL_ID,
        "scheduler": "two_fixed_independent_fifo_lanes",
        "governance_binding": dict(governance_binding),
        "tier2s_preregistration_binding": tier2s_preregistration_binding,
        "wait_for_idle_gpu": False,
        "allow_gpu_fallback": False,
        "jobs": packet["schedule"],
    }
    expected_target = {
        "schema_version": "rc-irstd-aaai27-tier2s-outer-target-denial-v1",
        "protocol_id": PROTOCOL_ID,
        "governance_binding": dict(governance_binding),
        "tier2s_preregistration_binding": tier2s_preregistration_binding,
        "outer_target_access_authorized": False,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "source_tier3_authorized": False,
        "paper_claim_authorized": False,
        "parent_tier2r_hold_sha256": packet["parent_evidence"]["decision"]["sha256"],
    }
    if prereg != expected_prereg:
        raise RuntimeError("Tier2S preregistration drift")
    if queue != expected_queue:
        raise RuntimeError("Tier2S queue manifest drift")
    if target != expected_target:
        raise RuntimeError("Tier2S target lock drift")
    return {
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "queue_manifest_sha256": _sha256(QUEUE_MANIFEST),
        "target_lock_sha256": _sha256(TARGET_LOCK),
        "tier2s_preregistration_binding": tier2s_preregistration_binding,
    }


class EventWriter:
    def __init__(self, path: Path) -> None:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise RuntimeError("scheduler event path is aliased or not a regular file")
        self.path = path
        self.lock = threading.Lock()
        self.previous = "0" * 64
        if path.exists():
            for raw in path.read_text(encoding="utf-8").splitlines():
                item = json.loads(raw)
                if item.get("previous_event_sha256") != self.previous:
                    raise RuntimeError("scheduler event hash chain is broken")
                claimed = item.pop("event_sha256", None)
                observed = hashlib.sha256(_canonical_bytes(item)).hexdigest()
                if claimed != observed:
                    raise RuntimeError("scheduler event digest is invalid")
                self.previous = claimed

    def append(self, event: Mapping[str, Any]) -> str:
        with self.lock:
            if self.path.is_symlink() or (
                self.path.exists() and not self.path.is_file()
            ):
                raise RuntimeError("scheduler event path changed identity")
            item = {
                "schema_version": "rc-irstd-aaai27-tier2s-scheduler-event-v1",
                "time": _now(),
                "previous_event_sha256": self.previous,
                **dict(event),
            }
            digest = hashlib.sha256(_canonical_bytes(item)).hexdigest()
            record = {**item, "event_sha256": digest}
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.previous = digest
            return digest


def _verify_frozen_event_log() -> dict[str, str]:
    if EVENT_LOG.is_symlink() or not EVENT_LOG.is_file():
        raise RuntimeError("Tier2S scheduler event log is absent or aliased")
    digest = _sha256(EVENT_LOG)
    expected = f"{digest}  {EVENT_LOG.name}\n".encode("ascii")
    if (
        EVENT_LOG_SIDECAR.is_symlink()
        or not EVENT_LOG_SIDECAR.is_file()
        or EVENT_LOG_SIDECAR.read_bytes() != expected
        or EVENT_LOG.stat().st_mode & 0o222
        or EVENT_LOG_SIDECAR.stat().st_mode & 0o222
    ):
        raise RuntimeError("Tier2S scheduler event freeze drift")
    EventWriter(EVENT_LOG)
    lines = EVENT_LOG.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError("Tier2S scheduler event log is empty")
    final_event = json.loads(lines[-1])
    if (
        final_event.get("event") != "all_exports_completed"
        or final_event.get("completed_jobs") != 18
    ):
        raise RuntimeError("Tier2S scheduler event log lacks its terminal event")
    return {
        "path": str(EVENT_LOG),
        "sha256": digest,
        "sidecar_path": str(EVENT_LOG_SIDECAR),
        "sidecar_sha256": _sha256(EVENT_LOG_SIDECAR),
    }


def _freeze_event_log() -> dict[str, str]:
    if EVENT_LOG.is_symlink() or not EVENT_LOG.is_file():
        raise RuntimeError("Tier2S scheduler event log is absent or aliased")
    digest = _sha256(EVENT_LOG)
    expected = f"{digest}  {EVENT_LOG.name}\n".encode("ascii")
    if EVENT_LOG_SIDECAR.exists() or EVENT_LOG_SIDECAR.is_symlink():
        raise RuntimeError("Tier2S scheduler event sidecar already exists")
    _atomic_write(EVENT_LOG_SIDECAR, expected)
    EVENT_LOG.chmod(0o444)
    EVENT_LOG_SIDECAR.chmod(0o444)
    return _verify_frozen_event_log()


def _write_status(status: str, **fields: Any) -> None:
    _write_json(
        STATUS_PATH,
        {
            "schema_version": "rc-irstd-aaai27-tier2s-factorized-status-v1",
            "protocol_id": PROTOCOL_ID,
            "status": status,
            "updated_at": _now(),
            "research_mode": "exploratory_source_only",
            "source_only": True,
            "outer_target_access_authorized": False,
            "outer_target_images_used": False,
            "outer_target_labels_used": False,
            "source_tier3_authorized": False,
            "paper_claim_authorized": False,
            **fields,
        },
    )


def _export_command(
    job: ExportJob,
    registration: Mapping[str, Any],
) -> list[str]:
    preregistration = registration.get("tier2s_preregistration_binding")
    if not isinstance(preregistration, Mapping):
        raise RuntimeError("Tier2S exporter preregistration binding is absent")
    return [
        str(PYTHON_EXECUTABLE),
        str(EXPORTER),
        "--protocol",
        str(PROTOCOL_PATH),
        "--governance-registration-sha256",
        str(preregistration.get("governance_registration_sha256", "")),
        "--tier2s-preregistration",
        str(PREREGISTRATION),
        "--tier2s-preregistration-sha256",
        str(preregistration.get("sha256", "")),
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


def _child_env(job: ExportJob) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job.physical_gpu)
    env["RC_IRSTD_TIER2S_SOURCE_ONLY"] = "1"
    env["RC_IRSTD_ALLOWED_DATA_ROOTS"] = os.pathsep.join(
        str((PROJECT_ROOT / value).resolve())
        for value in ("datasets/NUDT-SIRST", "datasets/IRSTD-1K")
    )
    env["RC_IRSTD_FORBIDDEN_DATA_ROOT"] = str(PROJECT_ROOT / "datasets/NUAA-SIRST")
    return env


def _run_lane(
    gpu: int,
    jobs: Sequence[ExportJob],
    events: EventWriter,
    registration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for job in sorted(jobs, key=lambda value: value.queue_index):
        output = Path(job.output_dir)
        if (
            not output.is_absolute()
            or output.resolve() != output
            or OUTPUT_ROOT.resolve() not in output.parents
            or output.exists()
            or output.is_symlink()
        ):
            raise RuntimeError(f"Tier2S output escaped its canonical root: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        log_path = output.parent / f"{output.name}.log"
        if log_path.exists() or log_path.is_symlink():
            raise RuntimeError(f"Tier2S lane log already exists or is aliased: {log_path}")
        command = _export_command(job, registration)
        events.append(
            {
                "event": "job_started",
                "run_id": job.run_id,
                "physical_gpu": gpu,
                "logical_device": "cuda:0",
                "queue_index": job.queue_index,
                "command_argv_sha256": hashlib.sha256(_canonical_bytes(command)).hexdigest(),
            }
        )
        with log_path.open("xb") as log:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=_child_env(job),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            log.flush()
            os.fsync(log.fileno())
        log_path.chmod(0o444)
        manifest = output / "manifest.json"
        ok = completed.returncode == 0 and manifest.is_file() and not manifest.is_symlink()
        event = {
            "event": "job_completed" if ok else "job_failed",
            "run_id": job.run_id,
            "physical_gpu": gpu,
            "queue_index": job.queue_index,
            "returncode": completed.returncode,
            "log": str(log_path),
            "log_sha256": _sha256(log_path),
            "manifest": str(manifest) if manifest.is_file() else None,
            "manifest_sha256": _sha256(manifest) if manifest.is_file() else None,
        }
        events.append(event)
        results.append(event)
        if not ok:
            break
    return results


def _handoff(
    packet: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    event_log_binding: Mapping[str, str],
) -> dict[str, Any]:
    completed = {str(item["run_id"]): item for item in results if item.get("event") == "job_completed"}
    if len(completed) != 18:
        raise RuntimeError("cannot build handoff before all 18 exports complete")
    jobs = [ExportJob(**value) for value in packet["schedule"]]
    by_checkpoint: dict[str, dict[str, Any]] = {}
    for job in jobs:
        record = by_checkpoint.setdefault(
            job.checkpoint_run_id,
            {
                "checkpoint_run_id": job.checkpoint_run_id,
                "seed": job.seed,
                "role": job.role,
                "fold": job.fold,
                "checkpoint": job.checkpoint,
                "checkpoint_sha256": job.checkpoint_sha256,
                "parent_heldout_score_manifest": job.parent_heldout_score_manifest,
                "parent_heldout_score_manifest_sha256": job.parent_heldout_score_manifest_sha256,
            },
        )
        manifest = Path(job.output_dir) / "manifest.json"
        record[f"{job.scope}_factorized_manifest"] = str(manifest)
        record[f"{job.scope}_factorized_manifest_sha256"] = _sha256(manifest)
    # Control held-out final logits remain bound to the parent artifact.  C has
    # a new held-out replay so base/residual counterfactuals can be evaluated.
    if len(by_checkpoint) != 12:
        raise RuntimeError("expected twelve unique frozen checkpoints")
    return {
        "schema_version": "rc-irstd-aaai27-tier2s-factorized-export-handoff-v1",
        "protocol_id": PROTOCOL_ID,
        "governance_binding": packet["governance_binding"],
        "tier2s_preregistration_binding": _tier2s_preregistration_binding(packet),
        "research_mode": "exploratory_source_only",
        "source_only": True,
        "outer_target_access_authorized": False,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "source_tier3_authorized": False,
        "paper_claim_authorized": False,
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "queue_manifest_sha256": _sha256(QUEUE_MANIFEST),
        "scheduler_event_log": event_log_binding["path"],
        "scheduler_event_log_sha256": event_log_binding["sha256"],
        "scheduler_event_log_sidecar": event_log_binding["sidecar_path"],
        "scheduler_event_log_sidecar_sha256": event_log_binding["sidecar_sha256"],
        "runs": by_checkpoint,
    }


def _verify_existing_handoff(packet: Mapping[str, Any]) -> str:
    _verify_frozen_registered_json(HANDOFF_PATH)
    handoff = _load_json(HANDOFF_PATH)
    event_log_binding = _verify_frozen_event_log()
    if (
        handoff.get("schema_version")
        != "rc-irstd-aaai27-tier2s-factorized-export-handoff-v1"
        or handoff.get("protocol_id") != PROTOCOL_ID
        or handoff.get("research_mode") != "exploratory_source_only"
        or handoff.get("source_only") is not True
        or handoff.get("outer_target_access_authorized") is not False
        or handoff.get("outer_target_images_used") is not False
        or handoff.get("outer_target_labels_used") is not False
        or handoff.get("source_tier3_authorized") is not False
        or handoff.get("paper_claim_authorized") is not False
        or handoff.get("governance_binding") != packet.get("governance_binding")
        or handoff.get("tier2s_preregistration_binding")
        != _tier2s_preregistration_binding(packet)
        or handoff.get("preregistration_sha256") != _sha256(PREREGISTRATION)
        or handoff.get("queue_manifest_sha256") != _sha256(QUEUE_MANIFEST)
        or handoff.get("scheduler_event_log") != event_log_binding["path"]
        or handoff.get("scheduler_event_log_sha256") != event_log_binding["sha256"]
        or handoff.get("scheduler_event_log_sidecar")
        != event_log_binding["sidecar_path"]
        or handoff.get("scheduler_event_log_sidecar_sha256")
        != event_log_binding["sidecar_sha256"]
    ):
        raise RuntimeError("existing Tier2S export handoff identity drift")
    runs = handoff.get("runs")
    if not isinstance(runs, Mapping) or len(runs) != 12:
        raise RuntimeError("existing Tier2S export handoff run set drift")
    for value in packet["schedule"]:
        job = ExportJob(**value)
        run = runs.get(job.checkpoint_run_id)
        if not isinstance(run, Mapping):
            raise RuntimeError(f"existing handoff lacks {job.checkpoint_run_id}")
        for field, expected in (
            ("checkpoint_run_id", job.checkpoint_run_id),
            ("seed", job.seed),
            ("role", job.role),
            ("fold", job.fold),
            ("checkpoint", job.checkpoint),
            ("checkpoint_sha256", job.checkpoint_sha256),
            ("parent_heldout_score_manifest", job.parent_heldout_score_manifest),
            (
                "parent_heldout_score_manifest_sha256",
                job.parent_heldout_score_manifest_sha256,
            ),
        ):
            if run.get(field) != expected:
                raise RuntimeError(
                    f"existing handoff parent binding drift: {job.checkpoint_run_id}.{field}"
                )
        manifest = Path(job.output_dir) / "manifest.json"
        manifest_key = f"{job.scope}_factorized_manifest"
        if (
            run.get(manifest_key) != str(manifest)
            or not manifest.is_file()
            or manifest.is_symlink()
            or run.get(f"{manifest_key}_sha256") != _sha256(manifest)
        ):
            raise RuntimeError(f"existing handoff manifest drift: {job.run_id}")
    return _sha256(HANDOFF_PATH)


def _run_evaluator(
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    preregistration = registration.get("tier2s_preregistration_binding")
    if not isinstance(preregistration, Mapping):
        raise RuntimeError("Tier2S evaluator preregistration binding is absent")
    output = AUDIT_ROOT / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON_EXECUTABLE),
        str(EVALUATOR),
        "--protocol",
        str(PROTOCOL_PATH),
        "--governance-registration-sha256",
        str(preregistration.get("governance_registration_sha256", "")),
        "--handoff",
        str(HANDOFF_PATH),
        "--output-dir",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    evaluation_logs: dict[str, dict[str, str]] = {}
    for role, raw in (("stdout", completed.stdout), ("stderr", completed.stderr)):
        path = AUDIT_ROOT / f"evaluation.{role}.log"
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"Tier2S evaluator {role} log already exists or is aliased")
        _atomic_write(path, raw.encode("utf-8"))
        path.chmod(0o444)
        evaluation_logs[role] = {"path": str(path), "sha256": _sha256(path)}
    if completed.returncode != 0:
        raise RuntimeError(f"factorized evaluator failed with exit code {completed.returncode}")
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("factorized evaluator stdout was not one JSON object")
    if (
        value.get("outer_target_access_authorized") is not False
        or value.get("authorizes_outer_target_access") is not False
        or value.get("source_tier3_authorized") is not False
        or value.get("paper_claim_authorized") is not False
    ):
        raise RuntimeError("factorized evaluator attempted unauthorized escalation")
    value["coordinator_evaluation_logs"] = evaluation_logs
    return value


def _exclusive_lock() -> Any:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            LOCK_PATH,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as error:
        raise RuntimeError("Tier2S coordinator lock path is unsafe") from error
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise RuntimeError("Tier2S coordinator lock is not a regular file")
    handle = os.fdopen(descriptor, "r+b")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError("another Tier2S coordinator owns the lock") from error
    return handle


def execute(packet: Mapping[str, Any]) -> dict[str, Any]:
    registration = _verify_registration(packet)
    lock = _exclusive_lock()
    try:
        if HANDOFF_PATH.exists() or HANDOFF_PATH.is_symlink():
            handoff_sha = _verify_existing_handoff(packet)
        else:
            events = EventWriter(EVENT_LOG)
            _write_status(
                "exporting",
                total_jobs=18,
                physical_gpus=list(PHYSICAL_GPUS),
                lane_lengths=packet["lane_lengths"],
                **registration,
            )
            jobs = [ExportJob(**value) for value in packet["schedule"]]
            futures = {}
            all_results: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=len(PHYSICAL_GPUS), thread_name_prefix="tier2s-gpu") as pool:
                for gpu in PHYSICAL_GPUS:
                    lane = [job for job in jobs if job.physical_gpu == gpu]
                    futures[pool.submit(_run_lane, gpu, lane, events, registration)] = gpu
                for future in as_completed(futures):
                    all_results.extend(future.result())
            failures = [item for item in all_results if item.get("event") != "job_completed"]
            if failures:
                raise RuntimeError(
                    "Tier2S export failed closed: "
                    + ",".join(str(item["run_id"]) for item in failures)
                )
            events.append({"event": "all_exports_completed", "completed_jobs": 18})
            event_log_binding = _freeze_event_log()
            handoff_sha = _write_once_json(
                HANDOFF_PATH, _handoff(packet, all_results, event_log_binding)
            )
        _write_status("evaluating", handoff_sha256=handoff_sha, completed_jobs=18)
        result = _run_evaluator(registration)
        final = {
            "schema_version": SCHEMA,
            "status": "completed_exploratory_audit",
            "protocol_id": PROTOCOL_ID,
            "research_mode": "exploratory_source_only",
            "source_only": True,
            "outer_target_access_authorized": False,
            "outer_target_images_used": False,
            "outer_target_labels_used": False,
            "source_tier3_authorized": False,
            "paper_claim_authorized": False,
            "handoff": str(HANDOFF_PATH),
            "handoff_sha256": handoff_sha,
            "evaluation": result,
        }
        _write_status(final["status"], handoff_sha256=handoff_sha, evaluation=result)
        return final
    except BaseException as error:
        try:
            _write_status("failed_closed", error_type=type(error).__name__, error=str(error))
        except Exception:
            pass
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--register-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    packet = verify_prerequisites()
    if args.verify_only:
        result = _public_packet(packet)
    elif args.register_only:
        result = {**_public_packet(packet), **register(packet)}
    else:
        result = execute(packet)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
