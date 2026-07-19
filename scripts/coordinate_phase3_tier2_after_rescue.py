#!/usr/bin/env python3
"""Continue the frozen Phase-3 source LODO protocol after raw-logit rescue.

This coordinator is intentionally separate from the immutable historical
probability-grid coordinator.  It accepts only the canonical, hash-frozen
``RESCUE_GO_TIER2`` authorization, runs the already-preregistered Tier-2
source-only jobs on GPUs 2/3, exports their raw logits, and hands those four
exports to the dedicated raw-logit Tier-2 gate.  It never calls the old
probability threshold sweep and never authorizes or opens NUAA data.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Iterator


# ``--verify-only`` is intended to be observational, including on a freshly
# checked-out host where imported-module bytecode caches do not yet exist.
sys.dont_write_bytecode = True


CANONICAL_PROJECT_ROOT = Path("/home/ly/RC-IRSTD-v2")
PROJECT_ROOT = CANONICAL_PROJECT_ROOT
EXPECTED_PYTHON = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.raw_logit_rescue_gate import (  # noqa: E402
    DEFAULT_NUMERIC_ATOL,
    RESCUE_GO_TIER2,
    evaluate_rescue_decision,
)
from scripts import coordinate_phase3_source_lodo_gate as historical  # noqa: E402


SCHEMA = "rc-irstd-aaai27-phase3-tier2-after-rescue-coordinator-v1"
HANDOFF_SCHEMA = "rc-irstd-aaai27-phase3-tier2-raw-logit-handoff-v1"
INTENT_SCHEMA = "rc-irstd-aaai27-phase3-tier2-after-rescue-intent-v1"
COMPLETION_SCHEMA = "rc-irstd-aaai27-phase3-tier2-raw-logit-gate-completion-v1"
STATUS_SCHEMA = "rc-irstd-aaai27-phase3-tier2-after-rescue-status-v1"
FAILURE_SCHEMA = "rc-irstd-aaai27-phase3-tier2-after-rescue-failure-v1"

RESCUE_REQUIRED_JSON = frozenset(
    {
        "protocol_amendment.json",
        "input_manifest.json",
        "input_hashes.json",
        "exact_source_curves/control.json",
        "exact_source_curves/full.json",
        "operating_points/strict.json",
        "operating_points/medium.json",
        "operating_points/loose.json",
        "dense_tail_grid_gap.json",
        "cross_domain_calibration_gap.json",
        "false_alarm_concentration.json",
        "evidence_manifest.json",
        "rescue_decision.json",
        "effective_authorization.json",
    }
)
EVIDENCE_KEYS = frozenset(
    {
        "exact_source_curves/control",
        "exact_source_curves/full",
        "operating_points/strict",
        "operating_points/medium",
        "operating_points/loose",
        "dense_tail_grid_gap",
        "cross_domain_calibration_gap",
        "false_alarm_concentration",
    }
)
RESCUE_RUN_IDS = frozenset(
    {
        "control_heldout_nudt",
        "control_heldout_irstd",
        "full_heldout_nudt",
        "full_heldout_irstd",
    }
)
TIER2_RUN_IDS = frozenset(
    spec.run_id
    for round_specs in historical.TIER2_PREREGISTERED_ROUNDS
    for spec in round_specs
)
EXPECTED_TIER2_POLICY = {
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
            "budgets": ["strict", "medium", "loose"],
            "operator": "greater_than_or_equal",
            "value": 0.0,
        },
        {
            "metric": "worst_pd_delta",
            "budgets": ["strict", "medium", "loose"],
            "operator": "greater_than_or_equal",
            "value": 0.0,
        },
    ],
    "numeric_tolerance": 1.0e-12,
    "missing_or_infeasible_point": "HOLD_TIER2",
}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
    if path.suffix != ".json":
        raise ValueError(f"Expected a JSON artifact: {path}")
    return path.with_suffix(".sha256")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return payload


def _assert_regular_unsymlinked(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Artifact must be a regular non-symlink file: {path}")


def _verify_frozen_pair(path: Path, *, require_canonical: bool = True) -> str:
    sidecar = _sidecar(path)
    for candidate in (path, sidecar):
        _assert_regular_unsymlinked(candidate)
        if stat.S_IMODE(candidate.stat().st_mode) != 0o444:
            raise RuntimeError(f"Frozen artifact mode is not 0444: {candidate}")
    payload = _load_json(path)
    if require_canonical and path.read_bytes() != _canonical_json_bytes(payload):
        raise RuntimeError(f"Frozen JSON is not canonical: {path}")
    digest = _sha256(path)
    expected = f"{digest}  {path.name}\n"
    if sidecar.read_text(encoding="ascii") != expected:
        raise RuntimeError(f"SHA-256 sidecar mismatch: {sidecar}")
    return digest


def _write_once_json(path: Path, payload: Mapping[str, Any]) -> str:
    content = _canonical_json_bytes(dict(payload))
    digest = hashlib.sha256(content).hexdigest()
    digest_content = f"{digest}  {path.name}\n".encode("ascii")
    sidecar = _sidecar(path)
    if path.exists() or sidecar.exists():
        if path.read_bytes() != content or sidecar.read_bytes() != digest_content:
            raise RuntimeError(f"Immutable artifact drift: {path}")
        _verify_frozen_pair(path)
        return digest
    _atomic_write(path, content)
    _atomic_write(sidecar, digest_content)
    path.chmod(0o444)
    sidecar.chmod(0o444)
    return digest


def _write_status(path: Path, status: str, **fields: Any) -> None:
    _atomic_write(
        path,
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


def _must_be_false(payload: Mapping[str, Any], fields: Sequence[str], label: str) -> None:
    for field in fields:
        if payload.get(field) is not False:
            raise RuntimeError(f"{label}.{field} must be exactly false")


def _verify_outer_false_recursively(value: Any, label: str = "root") -> None:
    """Reject any affirmative NUAA/outer-target access or use claim."""

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).lower()
            access_key = ("outer_target" in key or "nuaa" in key) and any(
                token in key
                for token in ("used", "loaded", "authorized", "authorizes", "access")
            )
            if access_key and isinstance(item, bool) and item is not False:
                raise RuntimeError(f"Forbidden outer-target boolean at {label}.{raw_key}")
            _verify_outer_false_recursively(item, f"{label}.{raw_key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _verify_outer_false_recursively(item, f"{label}[{index}]")


def _canonical_under(path: Path, root: Path) -> Path:
    if path.is_symlink():
        raise RuntimeError(f"Symlink is forbidden in the formal chain: {path}")
    resolved = path.resolve(strict=True)
    expected_root = root.resolve(strict=True)
    if not resolved.is_relative_to(expected_root):
        raise RuntimeError(f"Artifact escaped canonical root: {path}")
    return resolved


def _verify_rescue_directory(rescue_root: Path) -> dict[str, str]:
    expected_root = (
        PROJECT_ROOT
        / "artifacts/aaai27/audit/phase3_source_lodo_gate/raw_logit_rescue_v1"
    ).absolute()
    if rescue_root.absolute() != expected_root or rescue_root.is_symlink():
        raise RuntimeError("Raw-logit rescue root is not the fixed canonical directory")
    rescue_root.resolve(strict=True)
    json_files = {str(path.relative_to(rescue_root)) for path in rescue_root.rglob("*.json")}
    sha_files = {
        str(path.relative_to(rescue_root).with_suffix(".json"))
        for path in rescue_root.rglob("*.sha256")
    }
    if json_files != sha_files:
        raise RuntimeError("Rescue JSON/sidecar set is incomplete")
    missing = sorted(RESCUE_REQUIRED_JSON - json_files)
    if missing:
        raise RuntimeError(f"Rescue artifact set is incomplete: {missing}")
    digests: dict[str, str] = {}
    for relative in sorted(json_files):
        path = rescue_root / relative
        _canonical_under(path, rescue_root)
        _canonical_under(_sidecar(path), rescue_root)
        digests[relative] = _verify_frozen_pair(path)
    return digests


def _verify_live_binding(record: Mapping[str, Any], path_field: str, sha_field: str) -> Path:
    path = Path(str(record.get(path_field, ""))).resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Bound file is missing or a symlink: {path}")
    if record.get(sha_field) != _sha256(path):
        raise RuntimeError(f"Bound file hash drifted: {path}")
    return path


def _verify_historical_pair(path: Path) -> tuple[dict[str, Any], str]:
    # The historical coordinator used pretty-printed JSON before the rescue
    # introduced canonical compact JSON.  Its immutable bytes and sidecar are
    # authoritative; canonical reserialization is therefore intentionally not
    # imposed retroactively.
    digest = _verify_frozen_pair(path, require_canonical=False)
    return _load_json(path), digest


def _verify_tier2_schedule(prereg: Mapping[str, Any], layout: historical.Layout) -> list[dict[str, Any]]:
    if prereg.get("tier2_policy") != EXPECTED_TIER2_POLICY:
        raise RuntimeError("Historical Tier2 policy drifted")
    future = prereg.get("future_schedule")
    if not isinstance(future, list):
        raise RuntimeError("Historical future_schedule is absent")
    indexed = {
        item.get("run_id"): item
        for item in future
        if isinstance(item, Mapping) and isinstance(item.get("run_id"), str)
    }
    if not TIER2_RUN_IDS.issubset(indexed):
        raise RuntimeError("Historical future_schedule lacks a Tier2 run")
    schedule: list[dict[str, Any]] = []
    for round_index, round_specs in enumerate(historical.TIER2_PREREGISTERED_ROUNDS, 1):
        for spec in round_specs:
            item = indexed[spec.run_id]
            config_name = historical.BASE_CONFIGS[spec.role]
            initializer_name = spec.fold.initializer_name
            config = layout.configs / config_name
            initializer = layout.initializers / initializer_name
            expected = {
                "base_config": config_name,
                "base_config_sha256": _sha256(config),
                "initializer": initializer_name,
                "initializer_sha256": _sha256(initializer),
            }
            for field, value in expected.items():
                if item.get(field) != value:
                    raise RuntimeError(f"Tier2 future binding drifted: {spec.run_id}.{field}")
            if spec.physical_gpu not in (2, 3):
                raise RuntimeError(f"Tier2 GPU binding drifted: {spec.run_id}")
            schedule.append(
                {
                    "round": round_index,
                    "run_id": spec.run_id,
                    "role": spec.role,
                    "fold": spec.fold_key,
                    "held_out_source": spec.fold.held_out_name,
                    "training_source": spec.fold.train_name,
                    "physical_gpu": spec.physical_gpu,
                    "logical_device": "cuda:0",
                    "base_config": str(config.resolve()),
                    "base_config_sha256": expected["base_config_sha256"],
                    "initializer": str(initializer.resolve()),
                    "initializer_sha256": expected["initializer_sha256"],
                }
            )
    return schedule


def verify_authorization() -> dict[str, Any]:
    """Verify the complete frozen rescue chain without creating any artifact."""

    root = PROJECT_ROOT.resolve()
    historical_root = root / "artifacts/aaai27/audit/phase3_source_lodo_gate"
    rescue_root = historical_root / "raw_logit_rescue_v1"
    digests = _verify_rescue_directory(rescue_root)

    prereg_path = historical_root / "PHASE3_SOURCE_LODO_PREREGISTRATION.json"
    tier1_path = historical_root / "tier1_decision.json"
    prereg, prereg_sha = _verify_historical_pair(prereg_path)
    tier1, tier1_sha = _verify_historical_pair(tier1_path)
    coordinator_path = root / "scripts/coordinate_phase3_source_lodo_gate.py"
    coordinator_sha = _sha256(coordinator_path)
    tier1_bindings = tier1.get("input_bindings")
    if not isinstance(tier1_bindings, Mapping):
        raise RuntimeError("Historical Tier1 input bindings are absent")
    if (
        prereg.get("schema_version") != historical.PREREG_SCHEMA
        or prereg.get("coordinator_sha256") != coordinator_sha
        or tier1_bindings.get("coordinator_sha256") != coordinator_sha
        or tier1_bindings.get("preregistration_sha256") != prereg_sha
    ):
        raise RuntimeError("Historical preregistration/coordinator hash chain drifted")
    if (
        tier1.get("schema_version") != historical.DECISION_SCHEMA
        or tier1.get("decision") != "HOLD"
        or tier1.get("authorizes_tier2") is not False
    ):
        raise RuntimeError("Historical probability Tier1 HOLD is not immutable")
    _must_be_false(
        tier1,
        (
            "authorizes_outer_target_label_access",
            "outer_target_images_used",
            "outer_target_labels_used",
        ),
        "tier1_decision",
    )
    _must_be_false(
        prereg,
        ("outer_target_images_loaded", "outer_target_masks_loaded"),
        "preregistration",
    )
    if (
        prereg.get("outer_target") != "NUAA-SIRST"
        or prereg.get("source_split") != "train"
        or prereg.get("source_pseudo_targets") != ["NUDT-SIRST", "IRSTD-1K"]
    ):
        raise RuntimeError("Historical source-only LODO scope drifted")

    target_hold = root / "outputs/phase_state/HOLD_PHASE3_TARGET_LABEL_ACCESS"
    tier1_hold = root / "outputs/phase_state/PHASE3_SOURCE_TIER1_HOLD"
    tier1_go = root / "outputs/phase_state/PHASE3_SOURCE_TIER1_GO"
    if target_hold.read_bytes() != b"HOLD\n":
        raise RuntimeError("Target-label HOLD sentinel drifted")
    if target_hold.is_symlink() or tier1_hold.is_symlink():
        raise RuntimeError("Phase3 HOLD sentinels must not be symlinks")
    if tier1_hold.read_bytes() != f"HOLD {tier1_sha}\n".encode("ascii"):
        raise RuntimeError("Historical Tier1 HOLD sentinel drifted")
    if tier1_go.exists():
        raise RuntimeError("Historical Tier1 GO sentinel must remain absent")

    amendment = _load_json(rescue_root / "protocol_amendment.json")
    input_manifest = _load_json(rescue_root / "input_manifest.json")
    input_hashes = _load_json(rescue_root / "input_hashes.json")
    evidence_manifest = _load_json(rescue_root / "evidence_manifest.json")
    decision = _load_json(rescue_root / "rescue_decision.json")
    authorization = _load_json(rescue_root / "effective_authorization.json")
    for label, payload in (
        ("preregistration", prereg),
        ("tier1_decision", tier1),
        ("protocol_amendment", amendment),
        ("input_manifest", input_manifest),
        ("rescue_decision", decision),
        ("effective_authorization", authorization),
    ):
        _verify_outer_false_recursively(payload, label)

    if (
        amendment.get("schema_version")
        != "rc-irstd-aaai27-phase3-raw-logit-rescue-amendment-v1"
        or amendment.get("post_hoc_amendment") is not True
        or amendment.get("artifact_type")
        != "post_hoc_source_only_raw_logit_protocol_amendment"
    ):
        raise RuntimeError("Raw-logit protocol amendment identity drifted")
    data_policy = amendment.get("data_access_policy")
    invariants = amendment.get("frozen_invariants")
    historical_bindings = amendment.get("historical_bindings")
    if not all(isinstance(value, Mapping) for value in (data_policy, invariants, historical_bindings)):
        raise RuntimeError("Raw-logit amendment contract is incomplete")
    assert isinstance(data_policy, Mapping)
    assert isinstance(invariants, Mapping)
    assert isinstance(historical_bindings, Mapping)
    _must_be_false(
        data_policy,
        (
            "outer_target_images_authorized",
            "outer_target_images_used",
            "outer_target_labels_authorized",
            "outer_target_labels_used",
        ),
        "protocol_amendment.data_access_policy",
    )
    if (
        data_policy.get("allowed_source_pseudo_targets") != ["NUDT-SIRST", "IRSTD-1K"]
        or data_policy.get("allowed_split") != "train"
        or data_policy.get("outer_target") != "NUAA-SIRST"
        or invariants.get("exact_state_enumeration_is_primary") is not True
        or invariants.get("dense_grid_is_diagnostic_only") is not True
        or invariants.get("new_training_allowed") is not False
        or invariants.get("prediction_rule")
        != "float32_raw_logit >= shared_threshold_logit"
        or invariants.get("shared_threshold_across_nudt_and_irstd") is not True
    ):
        raise RuntimeError("Raw-logit amendment invariants drifted")
    if (
        historical_bindings.get("probability_preregistration_sha256") != prereg_sha
        or historical_bindings.get("probability_tier1_decision_sha256") != tier1_sha
    ):
        raise RuntimeError("Raw-logit amendment historical bindings drifted")
    phase2 = Path(str(historical_bindings.get("phase2_status", ""))).resolve()
    expected_phase2 = root / "artifacts/aaai27/audit/phase2_status.json"
    if (
        phase2 != expected_phase2.resolve()
        or not phase2.is_file()
        or phase2.is_symlink()
        or historical_bindings.get("probability_preregistration")
        != str(prereg_path.resolve())
        or historical_bindings.get("probability_tier1_decision")
        != str(tier1_path.resolve())
        or historical_bindings.get("phase2_status_sha256") != _sha256(phase2)
    ):
        raise RuntimeError("Raw-logit amendment Phase2 binding drifted")
    code_bindings = amendment.get("code_bindings")
    if not isinstance(code_bindings, Mapping) or not code_bindings:
        raise RuntimeError("Raw-logit amendment code bindings are absent")
    for relative, expected_sha in code_bindings.items():
        path = (root / str(relative)).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or path.is_symlink()
            or _sha256(path) != expected_sha
        ):
            raise RuntimeError(f"Raw-logit implementation binding drifted: {relative}")

    if (
        input_manifest.get("schema_version")
        != "rc-irstd-aaai27-phase3-raw-logit-rescue-input-v1"
        or input_manifest.get("source_only") is not True
        or input_manifest.get("expected_split_role") != "train"
        or Path(str(input_manifest.get("protocol_amendment", ""))).resolve()
        != (rescue_root / "protocol_amendment.json").resolve()
        or input_manifest.get("protocol_amendment_sha256")
        != digests["protocol_amendment.json"]
        or input_manifest.get("historical_probability_preregistration_sha256") != prereg_sha
        or input_manifest.get("historical_probability_tier1_decision_sha256") != tier1_sha
    ):
        raise RuntimeError("Raw-logit rescue input manifest drifted")
    _must_be_false(
        input_manifest,
        ("outer_target_images_used", "outer_target_labels_used"),
        "input_manifest",
    )
    rescue_runs = input_manifest.get("runs")
    hash_runs = input_hashes.get("runs")
    if (
        not isinstance(rescue_runs, Mapping)
        or set(rescue_runs) != RESCUE_RUN_IDS
        or not isinstance(hash_runs, Mapping)
        or set(hash_runs) != RESCUE_RUN_IDS
    ):
        raise RuntimeError("Raw-logit rescue run set drifted")
    expected_lodo = {
        "heldout_nudt": ("NUDT-SIRST", "IRSTD-1K"),
        "heldout_irstd": ("IRSTD-1K", "NUDT-SIRST"),
    }
    historical_runs = tier1_bindings.get("runs")
    if not isinstance(historical_runs, Mapping) or set(historical_runs) != RESCUE_RUN_IDS:
        raise RuntimeError("Historical Tier1 run bindings drifted")
    for run_id in sorted(RESCUE_RUN_IDS):
        run = rescue_runs[run_id]
        hashes = hash_runs[run_id]
        if not isinstance(run, Mapping) or not isinstance(hashes, Mapping):
            raise RuntimeError(f"Malformed rescue input run: {run_id}")
        role, fold = run_id.split("_heldout_", 1)
        fold = "heldout_" + fold
        held_out, training_source = expected_lodo[fold]
        if (
            run.get("role") != role
            or run.get("fold") != fold
            or run.get("held_out_source_pseudo_target") != held_out
            or run.get("training_source") != training_source
            or run.get("split_role") != "train"
            or run.get("labels_loaded") is not True
            or run.get("logit_dtype") != "float32"
            or run.get("inference_autocast_enabled") is not False
        ):
            raise RuntimeError(f"Rescue source-only run contract drifted: {run_id}")
        expected_run_dir = (
            root
            / "outputs/aaai27/detectors/source_lodo_gate/seed42"
            / role
            / fold
        ).resolve()
        expected_paths = {
            "checkpoint": expected_run_dir / "last.pt",
            "phase3_identity": expected_run_dir / "PHASE3_IDENTITY.json",
            "export_identity": expected_run_dir / "EXPORT_IDENTITY.json",
            "score_manifest": expected_run_dir / "scores_heldout_train/manifest.json",
        }
        if Path(str(run.get("score_dir", ""))).resolve() != (
            expected_run_dir / "scores_heldout_train"
        ):
            raise RuntimeError(f"Rescue score directory escaped canonical run: {run_id}")
        for path_field, sha_field in (
            ("checkpoint", "checkpoint_sha256"),
            ("phase3_identity", "phase3_identity_sha256"),
            ("export_identity", "export_identity_sha256"),
            ("score_manifest", "score_manifest_sha256"),
        ):
            bound_path = _verify_live_binding(run, path_field, sha_field)
            if bound_path != expected_paths[path_field]:
                raise RuntimeError(f"Rescue run path drifted: {run_id}.{path_field}")
            if hashes.get(sha_field) != run.get(sha_field):
                raise RuntimeError(f"Input-hash link drifted: {run_id}.{sha_field}")
        historical_run = historical_runs[run_id]
        if not isinstance(historical_run, Mapping):
            raise RuntimeError(f"Malformed historical Tier1 run binding: {run_id}")
        for historical_field, rescue_field in (
            ("checkpoint_sha256", "checkpoint_sha256"),
            ("identity_sha256", "phase3_identity_sha256"),
            ("export_identity_sha256", "export_identity_sha256"),
        ):
            if historical_run.get(historical_field) != run.get(rescue_field):
                raise RuntimeError(
                    f"Rescue input differs from historical Tier1: {run_id}.{rescue_field}"
                )
        for field in (
            "raw_logit_stream_sha256",
            "score_records_sha256",
            "score_ordered_image_ids_sha256",
            "split_file_sha256",
            "split_ordered_ids_sha256",
        ):
            if hashes.get(field) != run.get(field):
                raise RuntimeError(f"Input-hash link drifted: {run_id}.{field}")
        score_manifest = _load_json(Path(str(run["score_manifest"])))
        if (
            score_manifest.get("records_sha256") != run.get("score_records_sha256")
            or score_manifest.get("ordered_image_ids_sha256")
            != run.get("score_ordered_image_ids_sha256")
        ):
            raise RuntimeError(f"Score manifest binding drifted: {run_id}")

    if (
        input_hashes.get("schema_version")
        != "rc-irstd-aaai27-phase3-raw-logit-rescue-input-hashes-v1"
        or input_hashes.get("protocol_amendment_sha256")
        != digests["protocol_amendment.json"]
        or input_hashes.get("input_manifest_sha256") != digests["input_manifest.json"]
    ):
        raise RuntimeError("Raw-logit input-hash manifest drifted")
    probability_points = input_manifest.get("historical_probability_grid_operating_points")
    hash_probability_points = input_hashes.get("historical_probability_grid_operating_points")
    if probability_points != hash_probability_points or not isinstance(probability_points, Mapping):
        raise RuntimeError("Historical probability diagnostic bindings drifted")
    for role in ("control", "full"):
        for budget in ("strict", "medium", "loose"):
            record = probability_points.get(role, {}).get(budget)
            if not isinstance(record, Mapping):
                raise RuntimeError("Historical probability diagnostic binding is missing")
            point_path = _verify_live_binding(record, "path", "sha256")
            expected_point = (
                historical_root / "source_operating_points" / role / f"{budget}.json"
            ).resolve()
            if point_path != expected_point:
                raise RuntimeError("Historical probability point path drifted")

    if (
        evidence_manifest.get("schema_version")
        != "rc-irstd-aaai27-phase3-raw-logit-rescue-evidence-v1"
        or evidence_manifest.get("exact_raw_logit_states_are_primary") is not True
        or evidence_manifest.get("dense_grid_is_diagnostic_only") is not True
        or evidence_manifest.get("input_manifest_sha256") != digests["input_manifest.json"]
        or evidence_manifest.get("input_hashes_sha256") != digests["input_hashes.json"]
    ):
        raise RuntimeError("Raw-logit evidence manifest drifted")
    evidence = evidence_manifest.get("artifacts")
    if not isinstance(evidence, Mapping) or set(evidence) != EVIDENCE_KEYS:
        raise RuntimeError("Raw-logit evidence set drifted")
    for key in sorted(EVIDENCE_KEYS):
        record = evidence[key]
        if not isinstance(record, Mapping):
            raise RuntimeError(f"Malformed evidence binding: {key}")
        expected_path = rescue_root / f"{key}.json"
        expected_sidecar = _sidecar(expected_path)
        if (
            Path(str(record.get("path", ""))).resolve() != expected_path.resolve()
            or Path(str(record.get("sha256_sidecar", ""))).resolve()
            != expected_sidecar.resolve()
            or record.get("sha256") != digests[f"{key}.json"]
        ):
            raise RuntimeError(f"Evidence hash/path binding drifted: {key}")

    control_points: dict[str, Any] = {}
    full_points: dict[str, Any] = {}
    for budget in ("strict", "medium", "loose"):
        point = _load_json(rescue_root / f"operating_points/{budget}.json")
        control_points[budget] = point.get("control", {}).get("gate_point")
        full_points[budget] = point.get("full", {}).get("gate_point")
    rederived = evaluate_rescue_decision(
        control_points, full_points, numeric_atol=DEFAULT_NUMERIC_ATOL
    )
    for field, expected in rederived.items():
        if decision.get(field) != expected:
            raise RuntimeError(f"Frozen rescue decision disagrees with evidence: {field}")
    if (
        decision.get("decision") != RESCUE_GO_TIER2
        or decision.get("gate_valid") is not True
        or decision.get("authorizes_tier2") is not True
        or decision.get("failed_conditions") != []
        or decision.get("protocol_errors") != []
        or len(decision.get("conditions", [])) != 9
        or not all(item.get("passed") is True for item in decision["conditions"])
        or decision.get("protocol_amendment_sha256") != digests["protocol_amendment.json"]
        or decision.get("input_manifest_sha256") != digests["input_manifest.json"]
        or decision.get("input_hashes_sha256") != digests["input_hashes.json"]
        or decision.get("evidence_manifest_sha256") != digests["evidence_manifest.json"]
    ):
        raise RuntimeError("Raw-logit rescue GO authorization is invalid")
    _must_be_false(
        decision,
        (
            "authorizes_outer_target_image_access",
            "authorizes_outer_target_label_access",
            "outer_target_images_used",
            "outer_target_labels_used",
        ),
        "rescue_decision",
    )
    if (
        authorization.get("schema_version")
        != "rc-irstd-aaai27-phase3-raw-logit-rescue-authorization-v1"
        or authorization.get("decision") != RESCUE_GO_TIER2
        or authorization.get("derived_from_rescue_decision_sha256")
        != digests["rescue_decision.json"]
        or Path(str(authorization.get("derived_from_rescue_decision", ""))).resolve()
        != (rescue_root / "rescue_decision.json").resolve()
        or authorization.get("tier2_authorized") is not True
        or authorization.get("tier2_source_lodo_authorized") is not True
        or authorization.get("historical_probability_tier1_hold_remains_immutable")
        is not True
    ):
        raise RuntimeError("Effective rescue authorization drifted")
    _must_be_false(
        authorization,
        (
            "outer_target_access_authorized",
            "outer_target_image_access_authorized",
            "outer_target_label_access_authorized",
            "outer_target_images_used",
            "outer_target_labels_used",
        ),
        "effective_authorization",
    )

    layout = historical.Layout(root=root)
    schedule = _verify_tier2_schedule(prereg, layout)
    return {
        "schema_version": SCHEMA,
        "verified": True,
        "source_only": True,
        "tier2_source_lodo_authorized": True,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "preregistration": {"path": str(prereg_path), "sha256": prereg_sha},
        "historical_tier1_decision": {"path": str(tier1_path), "sha256": tier1_sha},
        "historical_coordinator": {
            "path": str(coordinator_path),
            "sha256": coordinator_sha,
        },
        "rescue_protocol_amendment": {
            "path": str(rescue_root / "protocol_amendment.json"),
            "sha256": digests["protocol_amendment.json"],
        },
        "rescue_input_manifest": {
            "path": str(rescue_root / "input_manifest.json"),
            "sha256": digests["input_manifest.json"],
        },
        "rescue_input_hashes": {
            "path": str(rescue_root / "input_hashes.json"),
            "sha256": digests["input_hashes.json"],
        },
        "rescue_evidence_manifest": {
            "path": str(rescue_root / "evidence_manifest.json"),
            "sha256": digests["evidence_manifest.json"],
        },
        "rescue_decision": {
            "path": str(rescue_root / "rescue_decision.json"),
            "sha256": digests["rescue_decision.json"],
        },
        "effective_authorization": {
            "path": str(rescue_root / "effective_authorization.json"),
            "sha256": digests["effective_authorization.json"],
        },
        "tier2_policy": prereg["tier2_policy"],
        "tier2_schedule": schedule,
    }


@contextmanager
def _exclusive_existing_lock(path: Path) -> Iterator[BinaryIO]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Required pre-existing lock file is absent: {path}")
    with path.open("r+b", buffering=0) as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Coordinator lock is already held: {path}") from error
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def _phase3_locks() -> Iterator[None]:
    state = PROJECT_ROOT / "outputs/phase_state"
    historical_root = PROJECT_ROOT / "artifacts/aaai27/audit/phase3_source_lodo_gate"
    with ExitStack() as stack:
        stack.enter_context(
            _exclusive_existing_lock(state / "phase3_source_lodo_coordinator.lock")
        )
        stack.enter_context(
            _exclusive_existing_lock(historical_root / ".raw_logit_rescue_v1.lock")
        )
        yield


def _registered_at(path: Path) -> str:
    if path.is_file():
        value = _load_json(path).get("registered_at")
        if isinstance(value, str) and value:
            return value
        raise RuntimeError(f"Immutable artifact lacks registered_at: {path}")
    return _now()


def _intent_payload(
    packet: Mapping[str, Any], path: Path, gate_runner: Path
) -> dict[str, Any]:
    continuation = Path(__file__).resolve()
    return {
        "schema_version": INTENT_SCHEMA,
        "registered_at": _registered_at(path),
        "source_only": True,
        "tier2_source_lodo_authorized": True,
        "activation_basis": "RESCUE_GO_TIER2_semantic_successor_to_historical_Tier1_HOLD",
        "historical_probability_tier1_hold_remains_immutable": True,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "outer_target_image_access_authorized": False,
        "outer_target_label_access_authorized": False,
        "preregistration": packet["preregistration"],
        "historical_tier1_decision": packet["historical_tier1_decision"],
        "rescue_decision": packet["rescue_decision"],
        "effective_authorization": packet["effective_authorization"],
        "rescue_evidence_manifest": packet["rescue_evidence_manifest"],
        "continuation": {"path": str(continuation), "sha256": _sha256(continuation)},
        "required_gate_runner": {
            "path": str(gate_runner.resolve()),
            "sha256": _sha256(gate_runner),
        },
        "tier2_policy": packet["tier2_policy"],
        "schedule": packet["tier2_schedule"],
        "evaluation_continuation": {
            "primary_score_domain": "float32_raw_logit",
            "prediction_rule": "float32_raw_logit >= shared_threshold_logit",
            "exact_state_enumeration_is_primary": True,
            "shared_threshold_across_nudt_and_irstd": True,
            "dense_grid_is_diagnostic_only": True,
            "probability_threshold_sweep_allowed": False,
        },
        "restart_contract": {
            "training_and_export_reuse_historical_recovery_functions": True,
            "gate_is_idempotently_reinvoked_only_without_a_valid_completion_record": True,
        },
    }


def _tier2_product(layout: historical.Layout, spec: historical.RunSpec) -> dict[str, Any]:
    run_dir = layout.output / spec.role / spec.fold_key
    checkpoint = run_dir / "last.pt"
    phase3_identity = run_dir / "PHASE3_IDENTITY.json"
    export_identity = run_dir / "EXPORT_IDENTITY.json"
    score_dir = run_dir / "scores_heldout_train"
    score_manifest = score_dir / "manifest.json"
    for path in (checkpoint, phase3_identity, export_identity, score_manifest):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Tier2 export product is absent: {path}")
    phase3_payload = _load_json(phase3_identity)
    export_payload = _load_json(export_identity)
    manifest = _load_json(score_manifest)
    checkpoint_sha = _sha256(checkpoint)
    if (
        phase3_payload.get("checkpoint_sha256") != checkpoint_sha
        or export_payload.get("checkpoint_sha256") != checkpoint_sha
        or manifest.get("labels_loaded") is not True
        or manifest.get("split_role") != "train"
        or manifest.get("logit_dtype") != "float32"
        or manifest.get("inference_autocast_enabled") is not False
    ):
        raise RuntimeError(f"Tier2 export identity chain drifted: {spec.run_id}")
    _verify_outer_false_recursively(phase3_payload, f"{spec.run_id}.phase3_identity")
    _verify_outer_false_recursively(export_payload, f"{spec.run_id}.export_identity")
    _verify_outer_false_recursively(manifest, f"{spec.run_id}.score_manifest")
    return {
        "run_id": spec.run_id,
        "role": spec.role,
        "fold": spec.fold_key,
        "held_out_source": spec.fold.held_out_name,
        "training_source": spec.fold.train_name,
        "source_only": True,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "score_dir": str(score_dir.resolve()),
        "score_manifest": str(score_manifest.resolve()),
        "score_manifest_sha256": _sha256(score_manifest),
        "score_records_sha256": manifest.get("records_sha256"),
        "raw_logit_stream_sha256": manifest.get("raw_logit_stream_sha256"),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "phase3_identity": str(phase3_identity.resolve()),
        "phase3_identity_sha256": _sha256(phase3_identity),
        "export_identity": str(export_identity.resolve()),
        "export_identity_sha256": _sha256(export_identity),
    }


def _handoff_payload(
    packet: Mapping[str, Any],
    layout: historical.Layout,
    path: Path,
    gate_runner: Path,
) -> dict[str, Any]:
    continuation = Path(__file__).resolve()
    runs = {
        spec.run_id: _tier2_product(layout, spec)
        for round_specs in historical.TIER2_PREREGISTERED_ROUNDS
        for spec in round_specs
    }
    if set(runs) != TIER2_RUN_IDS:
        raise RuntimeError("Tier2 handoff run set is incomplete")
    return {
        "schema_version": HANDOFF_SCHEMA,
        "registered_at": _registered_at(path),
        "source_only": True,
        "tier2_source_lodo_authorized": True,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "outer_target_image_access_authorized": False,
        "outer_target_label_access_authorized": False,
        "preregistration": packet["preregistration"],
        "historical_tier1_decision": packet["historical_tier1_decision"],
        "rescue_decision": packet["rescue_decision"],
        "effective_authorization": packet["effective_authorization"],
        "rescue_evidence_manifest": packet["rescue_evidence_manifest"],
        "continuation": {"path": str(continuation), "sha256": _sha256(continuation)},
        "runner": {"path": str(gate_runner), "sha256": _sha256(gate_runner)},
        "tier2_policy": packet["tier2_policy"],
        "exact_raw_logit_continuation": {
            "primary_score_domain": "float32_raw_logit",
            "prediction_rule": "float32_raw_logit >= shared_threshold_logit",
            "exact_state_enumeration_is_primary": True,
            "shared_threshold_across_nudt_and_irstd": True,
            "dense_grid_is_diagnostic_only": True,
            "probability_threshold_sweep_used": False,
        },
        "runs": runs,
    }


def _gate_verify_command(
    output_root: Path, handoff: Path, gate_runner: Path
) -> list[str]:
    return [
        str(EXPECTED_PYTHON.resolve()),
        str(gate_runner.resolve()),
        "--verify-only",
        "--handoff",
        str(handoff.resolve()),
        "--output-root",
        str(output_root.resolve()),
    ]


def _valid_completion(path: Path, handoff: Path, gate_runner: Path) -> bool:
    if not path.exists() and not _sidecar(path).exists():
        return False
    _verify_frozen_pair(path)
    payload = _load_json(path)
    log_path = Path(str(payload.get("log", ""))).resolve()
    verification = payload.get("frozen_product_verification")
    if (
        payload.get("schema_version") != COMPLETION_SCHEMA
        or payload.get("status") != "completed"
        or payload.get("handoff_sha256") != _sha256(handoff)
        or payload.get("gate_runner_sha256") != _sha256(gate_runner)
        or payload.get("returncode") != 0
        or not log_path.is_file()
        or payload.get("log_sha256") != _sha256(log_path)
        or not isinstance(verification, Mapping)
        or verification.get("command")
        != _gate_verify_command(path.parent, handoff, gate_runner)
        or verification.get("returncode") != 0
        or not all(
            isinstance(verification.get(field), str)
            and len(verification[field]) == 64
            for field in ("stdout_sha256", "stderr_sha256")
        )
    ):
        raise RuntimeError("Tier2 gate completion record drifted")
    return True


def _verify_gate_products(
    output_root: Path, handoff: Path, gate_runner: Path
) -> dict[str, Any]:
    """Run the dedicated runner's own frozen-product verifier.

    The verifier is deliberately captured through pipes: verification on a
    restart must not append to the main gate log whose hash is frozen in the
    completion record.
    """

    command = _gate_verify_command(output_root, handoff, gate_runner)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout = completed.stdout if isinstance(completed.stdout, bytes) else b""
    stderr = completed.stderr if isinstance(completed.stderr, bytes) else b""
    if completed.returncode != 0:
        diagnostic = stderr.decode("utf-8", errors="replace")[-2000:].strip()
        suffix = f": {diagnostic}" if diagnostic else ""
        raise RuntimeError(
            "Tier2 raw-logit gate frozen-product verification failed "
            f"with exit code {completed.returncode}{suffix}"
        )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
    }


def _run_gate(output_root: Path, handoff: Path, gate_runner: Path) -> Path:
    completion = output_root / "TIER2_GATE_COMPLETED.json"
    if _valid_completion(completion, handoff, gate_runner):
        _verify_gate_products(output_root, handoff, gate_runner)
        return completion
    log_path = output_root / "tier2_raw_logit_gate.log"
    command = [
        str(EXPECTED_PYTHON.resolve()),
        str(gate_runner.resolve()),
        "--handoff",
        str(handoff.resolve()),
        "--output-root",
        str(output_root.resolve()),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
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
        raise RuntimeError(f"Tier2 raw-logit gate failed with exit code {completed.returncode}")
    verification = _verify_gate_products(output_root, handoff, gate_runner)
    _write_once_json(
        completion,
        {
            "schema_version": COMPLETION_SCHEMA,
            "registered_at": _registered_at(completion),
            "status": "completed",
            "command": command,
            "returncode": completed.returncode,
            "handoff": str(handoff.resolve()),
            "handoff_sha256": _sha256(handoff),
            "gate_runner": str(gate_runner.resolve()),
            "gate_runner_sha256": _sha256(gate_runner),
            "log": str(log_path.resolve()),
            "log_sha256": _sha256(log_path),
            "frozen_product_verification": verification,
            "source_only": True,
            "outer_target_images_used": False,
            "outer_target_labels_used": False,
        },
    )
    return completion


def _record_failure(output_root: Path, error: BaseException) -> None:
    failures = output_root / "failures"
    path = failures / f"failure_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex}.json"
    try:
        _write_once_json(
            path,
            {
                "schema_version": FAILURE_SCHEMA,
                "registered_at": _now(),
                "status": "failed_closed",
                "error_type": type(error).__name__,
                "error": str(error),
                "source_only": True,
                "outer_target_images_used": False,
                "outer_target_labels_used": False,
                "outer_target_access_authorized": False,
            },
        )
    except Exception:
        pass


def run(*, verify_only: bool = False) -> dict[str, Any]:
    """Verify authorization, or execute the recoverable Tier2 continuation."""

    root = PROJECT_ROOT.resolve()
    if root != CANONICAL_PROJECT_ROOT.resolve():
        raise RuntimeError("Phase3 Tier2 continuation requires the fixed project root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError(
            "Phase3 Tier2 continuation is running under the wrong Python executable"
        )
    output_root = (
        root
        / "artifacts/aaai27/audit/phase3_source_lodo_gate/tier2_raw_logit_gate_v1"
    )
    status_path = output_root / "tier2_status.json"
    phase3_status_path = (
        root
        / "artifacts/aaai27/audit/phase3_source_lodo_gate/phase3_status.json"
    )
    pid_path = root / "outputs/phase_state/phase3_tier2_after_rescue.pid"
    with _phase3_locks():
        if verify_only:
            return verify_authorization()
        try:
            packet = verify_authorization()
            output_root.mkdir(parents=True, exist_ok=True)
            gate_runner = root / "scripts/run_phase3_tier2_raw_logit_gate.py"
            if not gate_runner.is_file() or gate_runner.is_symlink():
                raise FileNotFoundError(
                    f"Required Tier2 raw-logit gate is missing: {gate_runner}"
                )
            intent_path = output_root / "TIER2_EXECUTION_INTENT.json"
            intent_sha = _write_once_json(
                intent_path, _intent_payload(packet, intent_path, gate_runner)
            )
            _atomic_write(pid_path, f"{os.getpid()}\n".encode("ascii"))
            _write_status(
                status_path,
                "tier2_after_rescue_running",
                pid=os.getpid(),
                execution_intent_sha256=intent_sha,
            )
            _write_status(
                phase3_status_path,
                "tier2_after_rescue_running",
                pid=os.getpid(),
                tier2_status=str(status_path.resolve()),
                execution_intent_sha256=intent_sha,
            )
            layout = historical.Layout(root=root)
            for round_index, round_specs in enumerate(
                historical.TIER2_PREREGISTERED_ROUNDS, 1
            ):
                round_ids = [spec.run_id for spec in round_specs]
                _write_status(
                    status_path,
                    "tier2_after_rescue_running",
                    pid=os.getpid(),
                    execution_intent_sha256=intent_sha,
                    current_round=round_index,
                    run_ids=round_ids,
                    stage="training",
                )
                verify_authorization()
                historical.run_training_round(
                    layout, f"tier2_after_rescue_round{round_index}", round_specs
                )
                _write_status(
                    status_path,
                    "tier2_after_rescue_running",
                    pid=os.getpid(),
                    execution_intent_sha256=intent_sha,
                    current_round=round_index,
                    run_ids=round_ids,
                    stage="training_completed",
                )
                for spec in round_specs:
                    historical.ensure_export(layout, spec)
                    _write_status(
                        status_path,
                        "tier2_after_rescue_running",
                        pid=os.getpid(),
                        execution_intent_sha256=intent_sha,
                        current_round=round_index,
                        run_ids=round_ids,
                        current_run_id=spec.run_id,
                        stage="export_completed",
                    )
                verify_authorization()

            handoff = output_root / "TIER2_HANDOFF.json"
            handoff_sha = _write_once_json(
                handoff, _handoff_payload(packet, layout, handoff, gate_runner.resolve())
            )
            verify_authorization()
            _write_status(
                status_path,
                "tier2_raw_logit_gate_running",
                pid=os.getpid(),
                execution_intent_sha256=intent_sha,
                handoff_sha256=handoff_sha,
            )
            completion = _run_gate(output_root, handoff, gate_runner)
            verify_authorization()
            result = {
                "schema_version": SCHEMA,
                "status": "tier2_raw_logit_gate_completed",
                "source_only": True,
                "outer_target_images_used": False,
                "outer_target_labels_used": False,
                "execution_intent": str(intent_path),
                "execution_intent_sha256": intent_sha,
                "handoff": str(handoff),
                "handoff_sha256": handoff_sha,
                "completion": str(completion),
                "completion_sha256": _sha256(completion),
            }
            _write_status(
                status_path,
                result["status"],
                **{
                    key: value
                    for key, value in result.items()
                    if key not in {"schema_version", "status"}
                },
            )
            _write_status(
                phase3_status_path,
                "tier2_after_rescue_completed",
                tier2_status=str(status_path.resolve()),
                handoff_sha256=handoff_sha,
                completion_sha256=result["completion_sha256"],
            )
            return result
        except BaseException as error:
            _record_failure(output_root, error)
            try:
                _write_status(
                    status_path,
                    "failed_closed",
                    error_type=type(error).__name__,
                    error=str(error),
                )
                _write_status(
                    phase3_status_path,
                    "tier2_after_rescue_failed_closed",
                    tier2_status=str(status_path.resolve()),
                    error_type=type(error).__name__,
                    error=str(error),
                )
            except Exception:
                pass
            raise
        finally:
            pid_path.unlink(missing_ok=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify locks and the complete authorization chain without writing products",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run(verify_only=args.verify_only)
    except BaseException as error:
        print(f"FAILED_CLOSED {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
