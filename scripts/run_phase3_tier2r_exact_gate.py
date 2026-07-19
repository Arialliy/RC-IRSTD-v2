#!/usr/bin/env python3
"""Run the preregistered source-only Tier2R exact raw-logit gate.

The formal decision has two ordered levels: ``C - matched control`` and then
``CV-v1 - C``.  Each arm is trained on seeds 43/44/45 and evaluated on the two
source LODO folds with one shared FP32 raw-logit threshold.  A dense threshold
grid may be produced by a separate diagnostic, but this runner never consumes
it for a decision.

This module deliberately writes into a new component-rescue namespace.  It
only reads and hash-verifies the historical ``TIER2_HOLD`` chain and can never
modify or supersede those files.  A source-Tier3 design authorization and an
outer-target access authorization are separate artifacts; the latter is
always false here, including after a Tier2R GO.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from evaluation.artifact_integrity import file_sha256
from evaluation.raw_logit_oracle import (
    RawLogitSample,
    load_formal_raw_logit_directory,
    raw_logit_stream_sha256,
)
from evaluation.raw_logit_source_operating_point import (
    enumerate_exact_shared_states,
    evaluate_domains_at_threshold,
    select_exact_shared_source_operating_points,
)
from evaluation.threshold_sweep import domain_key
from scripts.run_phase3_raw_logit_rescue_v1 import _extract_gate_point
from scripts.run_phase3_tier2_raw_logit_gate import verify_selected_operating_points


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_RELATIVE = Path("configs/tier2r_component_rescue_protocol.json")
AUDIT_RELATIVE = Path("artifacts/aaai27/audit/component_rescue/tier2r_c_v1")
GATE_RELATIVE = AUDIT_RELATIVE / "exact_gate"
HANDOFF_NAME = "TIER2R_HANDOFF.json"
PREREGISTRATION_NAME = "COMPONENT_RESCUE_PREREGISTRATION.json"
INITIAL_TARGET_AUTHORIZATION_NAME = "NUAA_ACCESS_AUTHORIZATION.json"
COMPLETION_NAME = "TIER2R_GATE_COMPLETED.json"
PREREGISTRATION_SCHEMA = "rc-irstd-aaai27-tier2r-component-rescue-preregistration-v1"
INITIAL_TARGET_AUTHORIZATION_SCHEMA = "rc-irstd-aaai27-tier2r-initial-outer-target-authorization-v1"

PROTOCOL_SCHEMA = "rc-irstd-aaai27-tier2r-component-rescue-preregistered-protocol-v1"
HANDOFF_SCHEMA = "rc-irstd-aaai27-tier2r-exact-gate-handoff-v1"
DECISION_SCHEMA = "rc-irstd-aaai27-tier2r-exact-decision-v1"
SOURCE_AUTHORIZATION_SCHEMA = "rc-irstd-aaai27-tier2r-source-tier3-authorization-v1"
TARGET_AUTHORIZATION_SCHEMA = "rc-irstd-aaai27-tier2r-outer-target-authorization-v1"

TIER2R_GO = "TIER2R_GO_SOURCE_TIER3_DESIGN"
TIER2R_HOLD = "TIER2R_HOLD"
SEEDS: tuple[int, ...] = (43, 44, 45)
ROLES: tuple[str, ...] = ("control", "c", "cv")
FOLDS: tuple[str, ...] = ("heldout_nudt", "heldout_irstd")
BUDGETS: tuple[tuple[str, float, float], ...] = (
    ("strict", 1.0e-6, 1.0),
    ("medium", 5.0e-6, 5.0),
    ("loose", 1.0e-5, 10.0),
)
MATCHING_PROTOCOL = {
    "matching_rule": "overlap",
    "centroid_distance": 3.0,
    "connectivity": 2,
    "min_component_area": 1,
}
STRICT_MACRO_MEAN_MIN_DELTA = 0.005
OTHER_MEAN_MIN_DELTA = 0.0
WORST_SEED_MIN_DELTA = -0.005
STRICT_MACRO_POSITIVE_SEED_MIN = 2
NUMERIC_ATOL = 1.0e-12


@dataclass(frozen=True)
class RunSpec:
    seed: int
    role: str
    fold: str

    @property
    def run_id(self) -> str:
        return f"seed{self.seed}_{self.role}_{self.fold}"


RUN_SPECS: tuple[RunSpec, ...] = tuple(
    RunSpec(seed, role, fold)
    for seed in SEEDS
    for role in ROLES
    for fold in FOLDS
)


def expected_physical_gpu(seed: int, role: str, fold: str) -> int:
    """Return the frozen seed-local, role-counterbalanced GPU assignment."""

    if seed not in SEEDS or role not in ROLES or fold not in FOLDS:
        raise ValueError("unregistered Tier2R job identity")
    base = {
        ("control", "heldout_nudt"): 2,
        ("c", "heldout_nudt"): 3,
        ("cv", "heldout_nudt"): 2,
        ("control", "heldout_irstd"): 3,
        ("c", "heldout_irstd"): 2,
        ("cv", "heldout_irstd"): 3,
    }[(role, fold)]
    if SEEDS.index(seed) % 2:
        base = 5 - base
    return base


@dataclass(frozen=True)
class ExactAPI:
    enumerate_states: Callable[..., Any] = enumerate_exact_shared_states
    select_points: Callable[..., Any] = select_exact_shared_source_operating_points
    evaluate_threshold: Callable[..., Any] = evaluate_domains_at_threshold


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


def _write_once_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Create an immutable canonical JSON+digest pair, never overwrite bytes."""

    content = _canonical_json_bytes(dict(payload))
    digest = hashlib.sha256(content).hexdigest()
    digest_content = f"{digest}  {path.name}\n".encode("ascii")
    sidecar = _sidecar(path)
    if path.exists() or sidecar.exists():
        if (
            path.is_symlink()
            or sidecar.is_symlink()
            or not path.is_file()
            or not sidecar.is_file()
            or path.read_bytes() != content
            or sidecar.read_bytes() != digest_content
        ):
            raise RuntimeError(f"immutable Tier2R artifact drift: {path}")
        return digest
    _atomic_write(path, content)
    _atomic_write(sidecar, digest_content)
    path.chmod(0o444)
    sidecar.chmod(0o444)
    return digest


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return payload


def _verify_frozen_json(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    sidecar = _sidecar(path)
    for item in (path, sidecar):
        if item.is_symlink() or not item.is_file():
            raise RuntimeError(f"frozen artifact is absent or a symlink: {item}")
        if stat.S_IMODE(item.stat().st_mode) & 0o222:
            raise RuntimeError(f"frozen artifact remains writable: {item}")
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"frozen artifact SHA-256 drift: {path}")
    if sidecar.read_text(encoding="ascii") != f"{digest}  {path.name}\n":
        raise RuntimeError(f"frozen artifact sidecar drift: {sidecar}")
    return _load_object(path)


def _lexical_absolute(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if ".." in candidate.parts:
        raise RuntimeError(f"parent traversal is forbidden: {path}")
    return Path(os.path.abspath(candidate))


def _assert_no_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise RuntimeError(f"symlink path component is forbidden: {current}")


def validate_source_path(
    path: str | Path,
    *,
    allowed_roots: Sequence[str | Path],
    forbidden_root: str | Path | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve one source path through a strict non-symlink root allowlist.

    Lexical traversal and any symlink component are rejected before resolution,
    so a link from an allowed source tree into NUAA cannot pass this check.
    """

    candidate = _lexical_absolute(path)
    _assert_no_symlink_components(candidate)
    if must_exist and not candidate.exists():
        raise FileNotFoundError(candidate)
    resolved = candidate.resolve(strict=must_exist)
    roots: list[Path] = []
    for raw_root in allowed_roots:
        root = _lexical_absolute(raw_root)
        _assert_no_symlink_components(root)
        if must_exist and not root.is_dir():
            raise FileNotFoundError(root)
        roots.append(root.resolve(strict=must_exist))
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise RuntimeError(f"path is outside the source-root allowlist: {candidate}")
    if forbidden_root is not None:
        forbidden = _lexical_absolute(forbidden_root)
        if (
            candidate == forbidden
            or candidate.is_relative_to(forbidden)
            or forbidden.is_relative_to(candidate)
        ):
            raise RuntimeError(f"outer-target path is forbidden: {candidate}")
    return resolved

def _metric_definitions() -> tuple[dict[str, str], ...]:
    metrics: list[dict[str, str]] = [
        {"budget": "strict", "metric": "macro_pd"},
        {"budget": "strict", "metric": "domain_pd/irstd1k"},
        {"budget": "strict", "metric": "domain_pd/nudt"},
    ]
    metrics.extend(
        {"budget": budget, "metric": "pooled_pd"}
        for budget in ("strict", "medium", "loose")
    )
    metrics.extend(
        {"budget": budget, "metric": "worst_pd"}
        for budget in ("strict", "medium", "loose")
    )
    return tuple(metrics)


NINE_CRITERIA = _metric_definitions()


def _seed_mapping(value: Mapping[Any, Any], role: str) -> dict[int, Any]:
    result: dict[int, Any] = {}
    for seed in SEEDS:
        if seed in value:
            result[seed] = value[seed]
        elif str(seed) in value:
            result[seed] = value[str(seed)]
        else:
            raise ValueError(f"{role} is missing preregistered seed {seed}")
    extra = {int(key) for key in value if str(key).isdigit()} - set(SEEDS)
    if extra:
        raise ValueError(f"{role} contains unregistered seeds: {sorted(extra)}")
    return result


def _point_metric(point: Mapping[str, Any], metric: str) -> float:
    if point.get("found") is not True:
        raise ValueError("missing or infeasible operating point")
    if metric.startswith("domain_pd/"):
        domain = metric.split("/", 1)[1]
        domain_pd = point.get("domain_pd")
        if not isinstance(domain_pd, Mapping) or set(domain_pd) != {"nudt", "irstd1k"}:
            raise ValueError("operating point has an incomplete source-domain set")
        raw = domain_pd.get(domain)
    else:
        raw = point.get(metric)
    if isinstance(raw, bool):
        raise ValueError(f"metric {metric} must be numeric")
    number = float(raw)
    if not math.isfinite(number):
        raise ValueError(f"metric {metric} must be finite")
    return number


def evaluate_gate_level(
    points_by_role: Mapping[str, Mapping[Any, Mapping[str, Mapping[str, Any]]]],
    *,
    candidate: str,
    baseline: str,
    level_name: str,
    strict_macro_mean_min_delta: float = STRICT_MACRO_MEAN_MIN_DELTA,
    other_mean_min_delta: float = OTHER_MEAN_MIN_DELTA,
    worst_seed_min_delta: float = WORST_SEED_MIN_DELTA,
    numeric_atol: float = NUMERIC_ATOL,
) -> dict[str, Any]:
    """Evaluate one nine-criterion paired multi-seed gate."""

    for number, label in (
        (strict_macro_mean_min_delta, "strict macro margin"),
        (other_mean_min_delta, "other mean margin"),
        (worst_seed_min_delta, "worst-seed floor"),
        (numeric_atol, "numeric tolerance"),
    ):
        if not math.isfinite(float(number)):
            raise ValueError(f"{label} must be finite")
    if numeric_atol < 0:
        raise ValueError("numeric tolerance must be non-negative")
    role_error: str | None = None
    try:
        if candidate not in points_by_role or baseline not in points_by_role:
            raise ValueError(f"gate {level_name} lacks candidate or baseline")
        candidate_seeds = _seed_mapping(points_by_role[candidate], candidate)
        baseline_seeds = _seed_mapping(points_by_role[baseline], baseline)
    except (TypeError, ValueError) as error:
        role_error = str(error)
        candidate_seeds = {}
        baseline_seeds = {}
    criteria: list[dict[str, Any]] = []
    failures: list[str] = []
    for definition in NINE_CRITERIA:
        budget = definition["budget"]
        metric = definition["metric"]
        deltas: dict[str, float] = {}
        protocol_error: str | None = role_error
        for seed in SEEDS:
            if protocol_error is not None:
                break
            try:
                candidate_value = _point_metric(candidate_seeds[seed][budget], metric)
                baseline_value = _point_metric(baseline_seeds[seed][budget], metric)
                deltas[str(seed)] = candidate_value - baseline_value
            except (KeyError, TypeError, ValueError) as error:
                protocol_error = f"seed{seed}: {error}"
                break
        required_mean = (
            strict_macro_mean_min_delta
            if budget == "strict" and metric == "macro_pd"
            else other_mean_min_delta
        )
        if protocol_error is None:
            mean_delta: float | None = sum(deltas.values()) / len(SEEDS)
            worst_seed_delta: float | None = min(deltas.values())
            mean_passed = mean_delta >= required_mean - numeric_atol
            worst_seed_passed = worst_seed_delta >= worst_seed_min_delta - numeric_atol
            positive_seed_count: int | None = sum(
                delta > numeric_atol for delta in deltas.values()
            )
            consensus_required = budget == "strict" and metric == "macro_pd"
            consensus_passed = (
                not consensus_required
                or positive_seed_count >= STRICT_MACRO_POSITIVE_SEED_MIN
            )
        else:
            mean_delta = None
            worst_seed_delta = None
            positive_seed_count = None
            mean_passed = False
            worst_seed_passed = False
            consensus_required = budget == "strict" and metric == "macro_pd"
            consensus_passed = False if consensus_required else True
        passed = mean_passed and worst_seed_passed and consensus_passed
        criterion_id = f"{level_name}:{budget}:{metric}"
        record = {
            "criterion_id": criterion_id,
            "budget": budget,
            "metric": metric,
            "comparison": f"{candidate}_minus_{baseline}",
            "paired_seed_deltas": deltas,
            "paired_mean_delta": mean_delta,
            "required_mean_delta": required_mean,
            "worst_seed_delta": worst_seed_delta,
            "required_worst_seed_delta": worst_seed_min_delta,
            "numeric_atol": numeric_atol,
            "mean_passed": mean_passed,
            "worst_seed_passed": worst_seed_passed,
            "positive_seed_count": positive_seed_count,
            "positive_seed_min_required": (
                STRICT_MACRO_POSITIVE_SEED_MIN if consensus_required else None
            ),
            "positive_seed_consensus_passed": consensus_passed,
            "protocol_error": protocol_error,
            "passed": passed,
        }
        criteria.append(record)
        if not passed:
            failures.append(criterion_id)
    passed = not failures
    return {
        "name": level_name,
        "candidate": candidate,
        "baseline": baseline,
        "comparison_direction": "candidate_minus_baseline",
        "num_criteria": len(criteria),
        "criteria": criteria,
        "failed_criteria": failures,
        "passed": passed,
    }


def evaluate_two_level_decision(
    points_by_role: Mapping[str, Mapping[Any, Mapping[str, Mapping[str, Any]]]],
    *,
    strict_macro_mean_min_delta: float = STRICT_MACRO_MEAN_MIN_DELTA,
    other_mean_min_delta: float = OTHER_MEAN_MIN_DELTA,
    worst_seed_min_delta: float = WORST_SEED_MIN_DELTA,
    numeric_atol: float = NUMERIC_ATOL,
) -> dict[str, Any]:
    """Apply the preregistered fallback table after both evidence levels."""

    if set(points_by_role) != set(ROLES):
        raise ValueError(f"Tier2R requires exactly the three roles {ROLES}")
    common = {
        "strict_macro_mean_min_delta": strict_macro_mean_min_delta,
        "other_mean_min_delta": other_mean_min_delta,
        "worst_seed_min_delta": worst_seed_min_delta,
        "numeric_atol": numeric_atol,
    }
    contrast = evaluate_gate_level(
        points_by_role,
        candidate="c",
        baseline="control",
        level_name="contrast_vs_control",
        **common,
    )
    component = evaluate_gate_level(
        points_by_role,
        candidate="cv",
        baseline="c",
        level_name="component_context_vs_contrast",
        **common,
    )
    contrast_passed = contrast["passed"] is True
    component_passed = component["passed"] is True
    if contrast_passed and component_passed:
        selected_candidate: str | None = "cv"
        component_claim_retained = True
    elif contrast_passed:
        selected_candidate = "c"
        component_claim_retained = False
    else:
        selected_candidate = None
        component_claim_retained = False
    source_go = contrast_passed
    return {
        "schema_version": DECISION_SCHEMA,
        "decision": TIER2R_GO if source_go else TIER2R_HOLD,
        "gate_valid": True,
        "selected_candidate": selected_candidate,
        "component_claim_retained": component_claim_retained,
        "sequential_gate_order": [
            "contrast_vs_control",
            "component_context_vs_contrast",
        ],
        "levels": {
            "contrast_vs_control": contrast,
            "component_context_vs_contrast": {
                **component,
                "authorization_eligible_only_if_prior_level_passed": True,
                "prior_level_passed": contrast["passed"],
            },
        },
        "fallback_policy": "A_GO_B_HOLD_selects_C_and_drops_component_claim",
        "authorizes_source_tier3_design": source_go,
        "authorizes_outer_target_access": False,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
    }


def _load_protocol(root: Path) -> tuple[dict[str, Any], Path]:
    path = root / PROTOCOL_RELATIVE
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Tier2R protocol is absent or a symlink: {path}")
    payload = _load_object(path)
    if payload.get("schema_version") != PROTOCOL_SCHEMA:
        raise RuntimeError("Tier2R protocol schema drift")
    if payload.get("source_only") is not True:
        raise RuntimeError("Tier2R protocol is not source-only")
    score = payload.get("score_protocol")
    decision = payload.get("decision_protocol")
    training = payload.get("training_protocol")
    access = payload.get("data_access")
    if not all(isinstance(value, Mapping) for value in (score, decision, training, access)):
        raise RuntimeError("Tier2R protocol sections are incomplete")
    assert isinstance(score, Mapping)
    assert isinstance(decision, Mapping)
    assert isinstance(training, Mapping)
    expected_budgets = [
        {"name": name, "pixel": pixel, "component": component}
        for name, pixel, component in BUDGETS
    ]
    expected_gates = [
        {"name": "contrast_vs_control", "candidate": "c", "baseline": "control"},
        {
            "name": "component_context_vs_contrast",
            "candidate": "cv",
            "baseline": "c",
        },
    ]
    expected_selection = {
        "A_GO_B_GO": "select_cv_and_retain_component_claim",
        "A_GO_B_HOLD": "select_c_and_drop_component_claim",
        "A_HOLD": "overall_hold",
    }
    assert isinstance(access, Mapping)
    if (
        payload.get("seeds") != list(SEEDS)
        or set(payload.get("roles", {})) != set(ROLES)
        or set(payload.get("folds", {})) != set(FOLDS)
        or score.get("primary_representation") != "float32_raw_logit"
        or score.get("exact_state_enumeration_is_primary") is not True
        or score.get("prediction_rule")
        != "float32_raw_logit >= shared_threshold_logit"
        or score.get("inference_autocast_enabled") is not False
        or score.get("budgets") != expected_budgets
        or score.get("shared_threshold_across_source_domains") is not True
        or score.get("dense_grid_is_diagnostic_only") is not True
        or score.get("probability_threshold_sweep_allowed") is not False
        or score.get("matching") != MATCHING_PROTOCOL
        or decision.get("strict_macro_mean_min_delta") != STRICT_MACRO_MEAN_MIN_DELTA
        or decision.get("strict_macro_positive_seed_consensus_min")
        != STRICT_MACRO_POSITIVE_SEED_MIN
        or decision.get("sequential_gates") != expected_gates
        or decision.get("selection_table") != expected_selection
        or decision.get("other_mean_min_delta") != OTHER_MEAN_MIN_DELTA
        or decision.get("worst_seed_min_delta") != WORST_SEED_MIN_DELTA
        or decision.get("numeric_atol") != NUMERIC_ATOL
        or training.get("physical_gpus") != [2, 3]
        or training.get("physical_gpu_assignment")
        != "preregistered_exact_per_job"
        or training.get("schedule_order")
        != "seed_local_interleaved_and_role_gpu_counterbalanced"
        or decision.get("missing_or_infeasible_point") != TIER2R_HOLD
        or decision.get("non_finite_metric") != TIER2R_HOLD
        or decision.get("all_nine_criteria_and_worst_seed_checks_must_pass") is not True
        or training.get("matched_initial_extension_state_within_seed_fold") is not True
        or training.get("checkpoint_selection") != "fixed_last"
        or training.get("checkpoint_epoch_zero_based") != 79
        or training.get("matched_initializer_within_seed_fold") is not True
        or training.get("wait_for_idle_gpu") is not False
        or training.get("allow_gpu_fallback") is not False
        or training.get("total_detector_jobs") != len(RUN_SPECS)
        or access.get("outer_target_images_authorized") is not False
        or access.get("outer_target_images_used") is not False
        or access.get("outer_target_labels_used") is not False
        or access.get("allowed_source_roots") != ["datasets/NUDT-SIRST", "datasets/IRSTD-1K"]
        or access.get("outer_target_labels_authorized") is not False
    ):
        raise RuntimeError("Tier2R preregistered protocol drift")
    return payload, path


def _binding(payload: Mapping[str, Any], name: str, expected: Path) -> Path:
    raw = payload.get(name)
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"handoff lacks {name} binding")
    path = Path(str(raw.get("path", ""))).resolve()
    if path != expected.resolve() or raw.get("sha256") != file_sha256(path):
        raise RuntimeError(f"handoff {name} binding drift")
    return path


def verify_historical_hold(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    frozen = protocol.get("historical_tier2_hold")
    if not isinstance(frozen, Mapping):
        raise RuntimeError("protocol lacks historical Tier2 HOLD binding")
    decision_path = root / str(frozen.get("decision_path", ""))
    authorization_path = root / str(frozen.get("authorization_path", ""))
    decision = _verify_frozen_json(decision_path, str(frozen.get("decision_sha256")))
    authorization = _verify_frozen_json(
        authorization_path, str(frozen.get("authorization_sha256"))
    )
    if (
        decision.get("decision") != "TIER2_HOLD"
        or decision.get("authorizes_tier3") is not False
        or authorization.get("decision") != "TIER2_HOLD"
        or authorization.get("tier3_authorized") is not False
        or authorization.get("tier3_source_lodo_authorized") is not False
        or authorization.get("outer_target_access_authorized") is not False
        or authorization.get("outer_target_image_access_authorized") is not False
        or authorization.get("outer_target_label_access_authorized") is not False
    ):
        raise RuntimeError("historical Tier2 HOLD semantics drift")
    return {
        "decision": {"path": str(decision_path.resolve()), "sha256": file_sha256(decision_path)},
        "authorization": {
            "path": str(authorization_path.resolve()),
            "sha256": file_sha256(authorization_path),
        },
    }


def _expected_run_ids() -> set[str]:
    return {spec.run_id for spec in RUN_SPECS}


REQUIRED_CODE_BINDINGS = {
    "train_detector.py",
    "export_scores.py",
    "rc_irstd/models/rc_mshnet.py",
    "rc_irstd/models/__init__.py",
    "rc_irstd/cli/train_detector.py",
    "rc_irstd/cli/export_scores.py",
    "rc_irstd/training/detector_trainer.py",
    "rc_irstd/losses/detector.py",
    "rc_irstd/losses/sls.py",
    "evaluation/artifact_integrity.py",
    "evaluation/export_score_maps.py",
    "evaluation/raw_logit_oracle.py",
    "evaluation/raw_logit_source_operating_point.py",
    "evaluation/threshold_sweep.py",
    "evaluation/component_matching.py",
    "scripts/run_phase3_raw_logit_rescue_v1.py",
    "scripts/run_phase3_tier2_raw_logit_gate.py",
    "scripts/run_phase3_tier2r_exact_gate.py",
    "scripts/coordinate_phase3_tier2r_component_rescue.py",
}
REQUIRED_CODE_BINDINGS.update(
    str(path.relative_to(PROJECT_ROOT))
    for relative in ("model", "rc_irstd/models", "rc_irstd/data", "data_ext")
    for path in sorted((PROJECT_ROOT / relative).glob("*.py"))
)

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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_bound_regular_file(
    record: Any,
    *,
    expected: Path | None = None,
    label: str,
) -> Path:
    if not isinstance(record, Mapping):
        raise RuntimeError(f"{label} binding is absent")
    candidate = _lexical_absolute(str(record.get("path", "")))
    _assert_no_symlink_components(candidate)
    if expected is not None and candidate != _lexical_absolute(expected):
        raise RuntimeError(f"{label} canonical path drift")
    if not candidate.is_file() or candidate.is_symlink():
        raise RuntimeError(f"{label} is absent or a symlink")
    if not _is_sha256(record.get("sha256")):
        raise RuntimeError(f"{label} SHA-256 binding is malformed")
    if record["sha256"] != file_sha256(candidate):
        raise RuntimeError(f"{label} SHA-256 binding drift")
    return candidate.resolve(strict=True)


def _verify_preregistration(
    root: Path,
    protocol: Mapping[str, Any],
    protocol_path: Path,
) -> dict[str, Any]:
    prereg_path = root / AUDIT_RELATIVE / PREREGISTRATION_NAME
    prereg = _verify_frozen_json(prereg_path)
    if (
        prereg.get("schema_version") != PREREGISTRATION_SCHEMA
        or prereg.get("protocol_id") != protocol.get("protocol_id")
        or prereg.get("candidate_frozen_name") != protocol.get("candidate_frozen_name")
        or prereg.get("source_only") is not True
        or prereg.get("outer_target_images_used") is not False
        or prereg.get("outer_target_labels_used") is not False
        or prereg.get("outer_target_access_authorized") is not False
        or prereg.get("seeds") != list(SEEDS)
        or prereg.get("roles") != list(ROLES)
        or prereg.get("folds") != list(FOLDS)
        or prereg.get("score_protocol") != protocol.get("score_protocol")
        or prereg.get("decision_protocol") != protocol.get("decision_protocol")
        or prereg.get("training_protocol") != protocol.get("training_protocol")
        or prereg.get("failure_policy")
        != "missing_or_drifted_artifact_fails_closed_to_TIER2R_HOLD"
    ):
        raise RuntimeError("Tier2R preregistration protocol drift")
    _binding(prereg, "protocol", protocol_path)
    historical = verify_historical_hold(root, protocol)
    if prereg.get("historical_tier2_hold") != historical:
        raise RuntimeError("Tier2R preregistration historical HOLD binding drift")

    initial_path = _verify_bound_regular_file(
        prereg.get("initial_outer_target_authorization"),
        expected=root / AUDIT_RELATIVE / INITIAL_TARGET_AUTHORIZATION_NAME,
        label="initial outer-target authorization",
    )
    initial = _verify_frozen_json(
        initial_path,
        prereg["initial_outer_target_authorization"]["sha256"],
    )
    if (
        initial.get("schema_version") != INITIAL_TARGET_AUTHORIZATION_SCHEMA
        or initial.get("decision") != "OUTER_TARGET_ACCESS_LOCKED_BEFORE_TIER2R"
        or initial.get("protocol_sha256") != file_sha256(protocol_path)
        or initial.get("historical_tier2_hold_decision_sha256")
        != historical["decision"]["sha256"]
        or initial.get("outer_target_access_authorized") is not False
        or initial.get("outer_target_image_access_authorized") is not False
        or initial.get("outer_target_label_access_authorized") is not False
        or initial.get("outer_target_images_used") is not False
        or initial.get("outer_target_labels_used") is not False
    ):
        raise RuntimeError("Tier2R initial outer-target lock drift")

    base_configs = prereg.get("base_configs")
    if not isinstance(base_configs, Mapping) or set(base_configs) != set(ROLES):
        raise RuntimeError("Tier2R preregistration base-config set drift")
    protocol_roles = protocol.get("roles")
    if not isinstance(protocol_roles, Mapping):
        raise RuntimeError("Tier2R protocol role map is absent")
    for role in ROLES:
        expected = root / str(protocol_roles[role]["config"])
        _verify_bound_regular_file(
            base_configs[role], expected=expected, label=f"{role} base config"
        )

    initializers = prereg.get("initializers")
    protocol_folds = protocol.get("folds")
    if (
        not isinstance(initializers, Mapping)
        or set(initializers) != set(FOLDS)
        or not isinstance(protocol_folds, Mapping)
    ):
        raise RuntimeError("Tier2R preregistration initializer set drift")
    for fold in FOLDS:
        expected = root / str(protocol_folds[fold]["initializer"])
        _verify_bound_regular_file(
            initializers[fold], expected=expected, label=f"{fold} initializer"
        )

    expected_models = prereg.get("expected_model_configs")
    if not isinstance(expected_models, Mapping) or set(expected_models) != set(ROLES):
        raise RuntimeError("Tier2R expected-model-config set drift")
    for role, flags in EXPECTED_MODEL_FLAGS.items():
        model = expected_models[role]
        if (
            not isinstance(model, Mapping)
            or model.get("architecture_version")
            != "rc-mshnet-v2-component-role-split"
            or model.get("backend") != "rc_mshnet"
            or model.get("baseline_identity") != "canonical_mshnet"
            or model.get("initialization_contract") != "zero_residual_exact_mshnet"
        ):
            raise RuntimeError(f"Tier2R expected model identity drift: {role}")
        for field, expected in flags.items():
            if type(model.get(field)) is not bool or model[field] is not expected:
                raise RuntimeError(f"Tier2R expected model flag drift: {role}.{field}")

    source_splits = prereg.get("source_splits")
    allowed_roots = [root / value for value in protocol["data_access"]["allowed_source_roots"]]
    if not isinstance(source_splits, Mapping) or len(source_splits) != len(allowed_roots):
        raise RuntimeError("Tier2R source-split binding set drift")
    split_by_root: dict[Path, Mapping[str, Any]] = {}
    for raw in source_splits.values():
        if not isinstance(raw, Mapping):
            raise RuntimeError("Tier2R source-split binding must be an object")
        bound_root = _lexical_absolute(str(raw.get("root", "")))
        split_by_root[bound_root] = raw
    if set(split_by_root) != {_lexical_absolute(path) for path in allowed_roots}:
        raise RuntimeError("Tier2R source-split root set drift")
    for allowed_root in allowed_roots:
        record = split_by_root[_lexical_absolute(allowed_root)]
        split_path = _lexical_absolute(str(record.get("split_file", "")))
        _assert_no_symlink_components(split_path)
        if (
            record.get("split") != "train"
            or not split_path.is_file()
            or split_path.is_symlink()
            or not split_path.is_relative_to(_lexical_absolute(allowed_root))
            or record.get("split_file_sha256") != file_sha256(split_path)
            or not _is_sha256(record.get("ordered_ids_sha256"))
            or type(record.get("num_samples")) is not int
            or record["num_samples"] <= 0
        ):
            raise RuntimeError(f"Tier2R source-split binding drift: {allowed_root}")

    code_bindings = prereg.get("code_bindings")
    if not isinstance(code_bindings, Mapping) or set(code_bindings) != REQUIRED_CODE_BINDINGS:
        raise RuntimeError("Tier2R code-closure binding set drift")
    for relative, digest in code_bindings.items():
        candidate = root / relative
        _assert_no_symlink_components(_lexical_absolute(candidate))
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or not _is_sha256(digest)
            or digest != file_sha256(candidate)
        ):
            raise RuntimeError(f"Tier2R code-closure drift: {relative}")

    schedule = prereg.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != len(RUN_SPECS):
        raise RuntimeError("Tier2R preregistered schedule size drift")
    scheduled: dict[str, Mapping[str, Any]] = {}
    for item in schedule:
        if not isinstance(item, Mapping) or item.get("run_id") in scheduled:
            raise RuntimeError("Tier2R preregistered schedule contains an invalid duplicate")
        scheduled[str(item.get("run_id"))] = item
    if set(scheduled) != _expected_run_ids():
        raise RuntimeError("Tier2R preregistered schedule run set drift")
    for spec in RUN_SPECS:
        item = scheduled[spec.run_id]
        if (
            item.get("seed") != spec.seed
            or item.get("role") != spec.role
            or item.get("fold") != spec.fold
            or item.get("physical_gpu")
            != expected_physical_gpu(spec.seed, spec.role, spec.fold)
            or item.get("logical_device") != "cuda:0"
            or type(item.get("round")) is not int
            or not 1 <= item["round"] <= 9
        ):
            raise RuntimeError(f"Tier2R preregistered schedule drift: {spec.run_id}")
    return prereg


def validate_handoff(
    handoff_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]], dict[str, Any]]:
    root = Path(project_root).resolve()
    protocol, protocol_path = _load_protocol(root)
    historical = verify_historical_hold(root, protocol)
    preregistration = _verify_preregistration(root, protocol, protocol_path)
    path = Path(handoff_path).resolve()
    expected = root / AUDIT_RELATIVE / HANDOFF_NAME
    if path != expected.resolve():
        raise RuntimeError(f"Tier2R handoff must use the canonical path: {expected}")
    handoff = _verify_frozen_json(path)
    if (
        handoff.get("schema_version") != HANDOFF_SCHEMA
        or handoff.get("source_only") is not True
        or handoff.get("outer_target_images_used") is not False
        or handoff.get("outer_target_labels_used") is not False
        or handoff.get("outer_target_access_authorized") is not False
    ):
        raise RuntimeError("Tier2R handoff violates source-only target lock")
    _binding(handoff, "protocol", protocol_path)
    _binding(
        handoff,
        "preregistration",
        root / AUDIT_RELATIVE / PREREGISTRATION_NAME,
    )
    _binding(
        handoff,
        "initial_outer_target_authorization",
        root / AUDIT_RELATIVE / INITIAL_TARGET_AUTHORIZATION_NAME,
    )
    if (
        handoff.get("historical_tier2_hold") != historical
        or handoff.get("score_protocol") != protocol.get("score_protocol")
        or handoff.get("decision_protocol") != protocol.get("decision_protocol")
        or handoff["preregistration"]["sha256"]
        != file_sha256(root / AUDIT_RELATIVE / PREREGISTRATION_NAME)
        or preregistration.get("historical_tier2_hold") != historical
    ):
        raise RuntimeError("Tier2R handoff/preregistration cross-binding drift")
    runs = handoff.get("runs")
    if not isinstance(runs, Mapping) or set(runs) != _expected_run_ids():
        raise RuntimeError("Tier2R handoff run set is incomplete or contains extras")
    if any(not isinstance(record, Mapping) for record in runs.values()):
        raise RuntimeError("Tier2R handoff run records must be objects")
    return handoff, dict(runs), protocol


def _load_inputs(
    root: Path,
    runs: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    loader: Callable[..., Any] = load_formal_raw_logit_directory,
) -> tuple[dict[str, dict[int, dict[str, Sequence[RawLogitSample]]]], dict[str, Any]]:
    access = protocol["data_access"]
    allowed = [root / value for value in access["allowed_source_roots"]]
    forbidden = root / access["forbidden_outer_target_root"]
    fold_protocol = protocol["folds"]
    loaded: dict[str, dict[int, dict[str, Sequence[RawLogitSample]]]] = {
        role: {seed: {} for seed in SEEDS} for role in ROLES
    }
    records: dict[str, Any] = {}
    for spec in RUN_SPECS:
        record = runs[spec.run_id]
        fold = fold_protocol[spec.fold]
        expected_scalars = {
            "run_id": spec.run_id,
            "seed": spec.seed,
            "role": spec.role,
            "fold": spec.fold,
            "training_source": fold["training_source"],
            "held_out_source": fold["held_out_source"],
            "checkpoint_selection": "fixed_last",
            "checkpoint_epoch": 79,
            "physical_gpu": expected_physical_gpu(spec.seed, spec.role, spec.fold),
            "logical_device": "cuda:0",
            "source_only": True,
            "outer_target_images_used": False,
            "outer_target_labels_used": False,
        }
        for field, expected in expected_scalars.items():
            if record.get(field) != expected:
                raise RuntimeError(f"Tier2R run identity drift: {spec.run_id}.{field}")
        extension_sha = record.get("initial_extension_state_sha256")
        if (
            not isinstance(extension_sha, str)
            or len(extension_sha) != 64
            or any(character not in "0123456789abcdef" for character in extension_sha)
        ):
            raise RuntimeError(f"Tier2R extension-state identity drift: {spec.run_id}")
        training_root = validate_source_path(
            record.get("training_root", ""),
            allowed_roots=allowed,
            forbidden_root=forbidden,
        )
        heldout_root = validate_source_path(
            record.get("held_out_root", ""),
            allowed_roots=allowed,
            forbidden_root=forbidden,
        )
        if training_root != (root / fold["training_root"]).resolve():
            raise RuntimeError(f"Tier2R training root drift: {spec.run_id}")
        if heldout_root != (root / fold["held_out_root"]).resolve():
            raise RuntimeError(f"Tier2R held-out root drift: {spec.run_id}")
        checkpoint = Path(str(record.get("checkpoint", ""))).resolve()
        manifest_path = Path(str(record.get("score_manifest", ""))).resolve()
        score_dir = Path(str(record.get("score_dir", ""))).resolve()
        run_identity_path = Path(str(record.get("run_identity", ""))).resolve()
        export_identity_path = Path(str(record.get("export_identity", ""))).resolve()
        expected_run_dir = (
            root
            / "outputs/aaai27/detectors/component_rescue/tier2r_c_v1"
            / f"seed{spec.seed}"
            / spec.role
            / spec.fold
        ).resolve()
        expected_paths = {
            "checkpoint": expected_run_dir / "last.pt",
            "score_dir": expected_run_dir / "scores_heldout_train",
            "score_manifest": expected_run_dir / "scores_heldout_train/manifest.json",
            "run_identity": expected_run_dir / "TIER2R_RUN_IDENTITY.json",
            "export_identity": expected_run_dir / "TIER2R_EXPORT_IDENTITY.json",
        }
        actual_paths = {
            "checkpoint": checkpoint,
            "score_dir": score_dir,
            "score_manifest": manifest_path,
            "run_identity": run_identity_path,
            "export_identity": export_identity_path,
        }
        if actual_paths != expected_paths:
            raise RuntimeError(f"Tier2R run product canonical path drift: {spec.run_id}")
        for product in actual_paths.values():
            _assert_no_symlink_components(product)
        if checkpoint.is_symlink() or manifest_path.is_symlink() or score_dir.is_symlink():
            raise RuntimeError(f"Tier2R run product contains a symlink: {spec.run_id}")
        run_identity = _verify_frozen_json(
            run_identity_path, str(record.get("run_identity_sha256", ""))
        )
        export_identity = _verify_frozen_json(
            export_identity_path, str(record.get("export_identity_sha256", ""))
        )
        if (
            run_identity.get("run_id") != spec.run_id
            or run_identity.get("source_only") is not True
            or run_identity.get("outer_target_images_used") is not False
            or run_identity.get("outer_target_labels_used") is not False
            or run_identity.get("checkpoint_sha256") != record.get("checkpoint_sha256")
            or run_identity.get("initializer_sha256") != record.get("initializer_sha256")
            or run_identity.get("formal_config_sha256")
            != record.get("formal_config_sha256")
            or run_identity.get("initial_extension_state_sha256") != extension_sha
            or export_identity.get("run_id") != spec.run_id
            or export_identity.get("source_only") is not True
            or export_identity.get("outer_target_images_used") is not False
            or export_identity.get("outer_target_labels_used") is not False
            or export_identity.get("checkpoint_sha256")
            != record.get("checkpoint_sha256")
            or export_identity.get("score_manifest_sha256")
            != record.get("score_manifest_sha256")
            or export_identity.get("score_representation")
            != "raw_logit_float32+sigmoid_probability_float32"
            or export_identity.get("logit_dtype") != "float32"
            or export_identity.get("inference_autocast_enabled") is not False
        ):
            raise RuntimeError(f"Tier2R run/export identity drift: {spec.run_id}")
        if (
            not checkpoint.is_file()
            or record.get("checkpoint_sha256") != file_sha256(checkpoint)
            or manifest_path != score_dir / "manifest.json"
            or not manifest_path.is_file()
            or record.get("score_manifest_sha256") != file_sha256(manifest_path)
        ):
            raise RuntimeError(f"Tier2R run product binding drift: {spec.run_id}")
        samples, manifest, integrity, contract = loader(
            score_dir, expected_split_role="train"
        )
        if (
            domain_key(str(contract.get("target_dataset")))
            != domain_key(str(fold["held_out_source"]))
            or [domain_key(str(value)) for value in contract.get("source_datasets", [])]
            != [domain_key(str(fold["training_source"]))]
            or contract.get("split_role") != "train"
            or manifest.get("labels_loaded") is not True
            or manifest.get("logit_dtype") != "float32"
            or manifest.get("inference_autocast_enabled") is not False
            or contract.get("detector_weight_sha256") != file_sha256(checkpoint)
        ):
            raise RuntimeError(f"Tier2R raw-logit export contract drift: {spec.run_id}")
        domain = domain_key(str(fold["held_out_source"]))
        loaded[spec.role][spec.seed][domain] = samples
        records[spec.run_id] = {
            **expected_scalars,
            "training_root": str(training_root),
            "held_out_root": str(heldout_root),
            "physical_gpu": record.get("physical_gpu"),
            "initializer_sha256": record.get("initializer_sha256"),
            "initial_extension_state_sha256": extension_sha,
            "formal_config_sha256": record.get("formal_config_sha256"),
            "checkpoint_sha256": file_sha256(checkpoint),
            "score_manifest_sha256": integrity["manifest_sha256"],
            "score_records_sha256": integrity["records_sha256"],
            "score_ordered_image_ids_sha256": integrity["ordered_image_ids_sha256"],
            "score_num_records": integrity["num_records"],
            "raw_logit_stream_sha256": raw_logit_stream_sha256(samples),
            "split_file_sha256": contract["split_file_sha256"],
            "split_ordered_ids_sha256": contract["split_ordered_ids_sha256"],
        }
    for role in ROLES:
        for seed in SEEDS:
            if set(loaded[role][seed]) != {"nudt", "irstd1k"}:
                raise RuntimeError(f"Tier2R source coverage is incomplete: {role}/seed{seed}")
    for seed in SEEDS:
        for fold in FOLDS:
            group = [records[f"seed{seed}_{role}_{fold}"] for role in ROLES]
            for field in (
                "initializer_sha256",
                "initial_extension_state_sha256",
                "score_ordered_image_ids_sha256",
                "score_num_records",
                "split_file_sha256",
                "split_ordered_ids_sha256",
            ):
                if len({item[field] for item in group}) != 1:
                    raise RuntimeError(
                        f"Tier2R matched-arm binding differs: seed{seed}/{fold}/{field}"
                    )
    return loaded, records


def _compute_exact_points(
    samples_by_domain: Mapping[str, Sequence[RawLogitSample]],
    *,
    api: ExactAPI,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    enumeration = api.enumerate_states(
        samples_by_domain,
        loose_pixel_budget=1.0e-5,
        **MATCHING_PROTOCOL,
    )
    if (
        enumeration.get("exact_state_enumeration") is not True
        or enumeration.get("shared_threshold_across_domains") is not True
    ):
        raise RuntimeError("exact shared-state enumerator violated its contract")
    selections: dict[str, Any] = {}
    points: dict[str, Any] = {}
    for name, pixel, component in BUDGETS:
        selection = api.select_points(
            enumeration, pixel_budget=pixel, component_budget=component
        )
        selections[name] = selection
        points[name] = _extract_gate_point(selection)
    verification = verify_selected_operating_points(
        samples_by_domain,
        selections,
        evaluator=api.evaluate_threshold,
    )
    if verification.get("all_selected_points_verified") is not True:
        raise RuntimeError("Tier2R legacy selected-point verification failed")
    return enumeration, selections, {"points": points, "verification": verification}


def _gate_layout(
    project_root: str | Path, output_root: str | Path, handoff_path: str | Path
) -> tuple[Path, Path, Path]:
    root = Path(project_root).resolve()
    output = _lexical_absolute(output_root)
    handoff = _lexical_absolute(handoff_path)
    expected_output = root / GATE_RELATIVE
    expected_handoff = root / AUDIT_RELATIVE / HANDOFF_NAME
    if output != expected_output or handoff != expected_handoff:
        raise RuntimeError("Tier2R gate requires canonical, non-historical paths")
    _assert_no_symlink_components(output)
    _assert_no_symlink_components(handoff)
    return root, output, handoff


def _selected_checkpoint_set(
    decision: Mapping[str, Any],
    runs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    selected = decision.get("selected_candidate")
    retained = decision.get("component_claim_retained")
    go = decision.get("decision") == TIER2R_GO
    if not go:
        if selected is not None or retained is not False:
            raise RuntimeError("Tier2R HOLD must not select a checkpoint set")
        return None
    if selected not in {"c", "cv"}:
        raise RuntimeError("Tier2R GO lacks a registered selected candidate")
    if retained is not (selected == "cv"):
        raise RuntimeError("Tier2R component-claim selection semantics drift")
    expected_ids = {
        spec.run_id for spec in RUN_SPECS if spec.role == selected
    }
    selected_runs = {
        run_id: {
            "checkpoint": record["checkpoint"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "run_identity": record["run_identity"],
            "run_identity_sha256": record["run_identity_sha256"],
            "export_identity": record["export_identity"],
            "export_identity_sha256": record["export_identity_sha256"],
            "score_manifest": record["score_manifest"],
            "score_manifest_sha256": record["score_manifest_sha256"],
        }
        for run_id, record in runs.items()
        if record.get("role") == selected
    }
    if set(selected_runs) != expected_ids:
        raise RuntimeError("Tier2R selected checkpoint set is incomplete")
    return {
        "selected_role": selected,
        "num_checkpoints": len(selected_runs),
        "runs": dict(sorted(selected_runs.items())),
    }


def _source_authorization_payload(
    decision: Mapping[str, Any],
    decision_path: Path,
    handoff_path: Path,
    runs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    go = decision.get("decision") == TIER2R_GO
    return {
        "schema_version": SOURCE_AUTHORIZATION_SCHEMA,
        "decision": decision.get("decision"),
        "derived_from_decision": str(decision_path.resolve()),
        "derived_from_decision_sha256": file_sha256(decision_path),
        "handoff": {
            "path": str(handoff_path.resolve()),
            "sha256": file_sha256(handoff_path),
        },
        "selected_candidate": decision.get("selected_candidate"),
        "component_claim_retained": decision.get("component_claim_retained"),
        "selected_checkpoint_set": _selected_checkpoint_set(decision, runs),
        "source_tier3_design_authorized": go,
        "outer_target_access_authorized": False,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
    }


def _target_authorization_payload(
    decision_path: Path,
    source_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": TARGET_AUTHORIZATION_SCHEMA,
        "decision": "OUTER_TARGET_ACCESS_REMAINS_LOCKED",
        "derived_from_tier2r_decision_sha256": file_sha256(decision_path),
        "source_authorization_sha256": file_sha256(source_path),
        "outer_target_access_authorized": False,
        "outer_target_image_access_authorized": False,
        "outer_target_label_access_authorized": False,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "tier2r_go_does_not_authorize_outer_target_access": True,
    }


def _finalize_gate_chain(root: Path, output: Path, handoff_path: Path) -> None:
    _, runs, _ = validate_handoff(handoff_path, project_root=root)
    decision_path = output / "COMPONENT_RESCUE_DECISION.json"
    evidence_path = output / "evidence_manifest.json"
    decision = _verify_frozen_json(decision_path)
    _verify_frozen_json(evidence_path)
    if (
        decision.get("decision") not in {TIER2R_GO, TIER2R_HOLD}
        or decision.get("handoff_sha256") != file_sha256(handoff_path)
        or decision.get("evidence_manifest_sha256") != file_sha256(evidence_path)
    ):
        raise RuntimeError("Tier2R decision cannot be finalized")
    source_path = output / "SOURCE_TIER3_DESIGN_AUTHORIZATION.json"
    _write_once_json(
        source_path,
        _source_authorization_payload(decision, decision_path, handoff_path, runs),
    )
    target_path = output / "OUTER_TARGET_ACCESS_AUTHORIZATION.json"
    _write_once_json(
        target_path,
        _target_authorization_payload(decision_path, source_path),
    )
    completion_path = output / COMPLETION_NAME
    _write_once_json(
        completion_path,
        {
            "schema_version": "rc-irstd-aaai27-tier2r-exact-gate-completion-v1",
            "completed": True,
            "handoff_sha256": file_sha256(handoff_path),
            "decision_sha256": file_sha256(decision_path),
            "evidence_manifest_sha256": file_sha256(evidence_path),
            "source_authorization_sha256": file_sha256(source_path),
            "outer_target_authorization_sha256": file_sha256(target_path),
        },
    )


def run_gate(
    *,
    handoff_path: str | Path,
    output_root: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    loader: Callable[..., Any] = load_formal_raw_logit_directory,
    exact_api: ExactAPI | None = None,
) -> dict[str, Any]:
    root, output, handoff_file = _gate_layout(project_root, output_root, handoff_path)
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".tier2r_exact_gate.lock"
    if lock_path.is_symlink():
        raise RuntimeError("Tier2R gate lock must not be a symlink")
    lock = lock_path.open("a+b")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError("another Tier2R exact gate owns the lock") from error
    try:
        if (output / "COMPONENT_RESCUE_DECISION.json").exists():
            _finalize_gate_chain(root, output, handoff_file)
            return verify_frozen_gate(
                handoff_path=handoff_file,
                output_root=output,
                project_root=root,
            )
        handoff, run_bindings, protocol = validate_handoff(
            handoff_file, project_root=root
        )
        historical = verify_historical_hold(root, protocol)
        initial_target_auth = _verify_frozen_json(
            root / AUDIT_RELATIVE / INITIAL_TARGET_AUTHORIZATION_NAME
        )
        if initial_target_auth.get("outer_target_access_authorized") is not False:
            raise RuntimeError("initial outer-target authorization is not false")
        loaded, records = _load_inputs(
            root, run_bindings, protocol, loader=loader
        )
        api = exact_api or ExactAPI()
        protocol_path = root / PROTOCOL_RELATIVE
        prereg_path = root / AUDIT_RELATIVE / PREREGISTRATION_NAME
        input_manifest_path = output / "input_manifest.json"
        _write_once_json(
            input_manifest_path,
            {
                "schema_version": "rc-irstd-aaai27-tier2r-exact-input-v1",
                "source_only": True,
                "outer_target_images_used": False,
                "outer_target_labels_used": False,
                "protocol": {"path": str(protocol_path), "sha256": file_sha256(protocol_path)},
                "preregistration": {
                    "path": str(prereg_path),
                    "sha256": file_sha256(prereg_path),
                },
                "historical_tier2_hold": historical,
                "handoff": {"path": str(handoff_file), "sha256": file_sha256(handoff_file)},
                "runs": records,
                "score_representation": "float32_raw_logit",
                "exact_state_enumeration_is_primary": True,
                "dense_grid_is_diagnostic_only": True,
                "matching_protocol": MATCHING_PROTOCOL,
                "budgets": [
                    {"name": name, "pixel": pixel, "component": component}
                    for name, pixel, component in BUDGETS
                ],
            },
        )
        points_by_role: dict[str, dict[int, Any]] = {role: {} for role in ROLES}
        artifact_bindings: dict[str, Any] = {}
        for role in ROLES:
            for seed in SEEDS:
                enumeration, selections, result = _compute_exact_points(
                    loaded[role][seed], api=api
                )
                points_by_role[role][seed] = result["points"]
                for label, payload in (
                    (f"exact_curves/{role}_seed{seed}", enumeration),
                    (
                        f"operating_points/{role}_seed{seed}",
                        {
                            "schema_version": "rc-irstd-aaai27-tier2r-operating-points-v1",
                            "role": role,
                            "seed": seed,
                            "selections": selections,
                            "gate_points": result["points"],
                            "legacy_verification": result["verification"],
                        },
                    ),
                ):
                    path = output / f"{label}.json"
                    _write_once_json(path, payload)
                    artifact_bindings[label] = {
                        "path": str(path.resolve()),
                        "sha256": file_sha256(path),
                    }
        evidence_manifest_path = output / "evidence_manifest.json"
        _write_once_json(
            evidence_manifest_path,
            {
                "schema_version": "rc-irstd-aaai27-tier2r-exact-evidence-v1",
                "input_manifest_sha256": file_sha256(input_manifest_path),
                "exact_raw_logit_states_are_primary": True,
                "shared_threshold_across_source_domains": True,
                "dense_grid_is_diagnostic_only": True,
                "dense_grid_used_for_decision": False,
                "all_selected_points_legacy_verified": True,
                "artifacts": artifact_bindings,
            },
        )
        gate = evaluate_two_level_decision(points_by_role)
        decision_path = output / "COMPONENT_RESCUE_DECISION.json"
        _write_once_json(
            decision_path,
            {
                **gate,
                "scope": "source_only_tier2r_multi_seed_lodo_exact_raw_logit",
                "protocol_sha256": file_sha256(protocol_path),
                "preregistration_sha256": file_sha256(prereg_path),
                "handoff_sha256": file_sha256(handoff_file),
                "input_manifest_sha256": file_sha256(input_manifest_path),
                "evidence_manifest_sha256": file_sha256(evidence_manifest_path),
                "historical_tier2_hold_decision_sha256": historical["decision"]["sha256"],
                "historical_tier2_hold_remains_immutable": True,
            },
        )
        _finalize_gate_chain(root, output, handoff_file)
        result = verify_frozen_gate(
            handoff_path=handoff_file, output_root=output, project_root=root
        )
        result["verified_only"] = False
        return result
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def verify_frozen_gate(
    *,
    handoff_path: str | Path,
    output_root: str | Path,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root, output, handoff = _gate_layout(project_root, output_root, handoff_path)
    _, runs, _ = validate_handoff(handoff, project_root=root)
    decision_path = output / "COMPONENT_RESCUE_DECISION.json"
    source_path = output / "SOURCE_TIER3_DESIGN_AUTHORIZATION.json"
    target_path = output / "OUTER_TARGET_ACCESS_AUTHORIZATION.json"
    evidence_path = output / "evidence_manifest.json"
    completion_path = output / COMPLETION_NAME
    decision = _verify_frozen_json(decision_path)
    source = _verify_frozen_json(source_path)
    target = _verify_frozen_json(target_path)
    evidence = _verify_frozen_json(evidence_path)
    completion = _verify_frozen_json(completion_path)
    expected_go = decision.get("decision") == TIER2R_GO
    expected_selection = _selected_checkpoint_set(decision, runs)
    expected_handoff = {
        "path": str(handoff.resolve()),
        "sha256": file_sha256(handoff),
    }
    if (
        decision.get("decision") not in {TIER2R_GO, TIER2R_HOLD}
        or decision.get("handoff_sha256") != file_sha256(handoff)
        or source.get("schema_version") != SOURCE_AUTHORIZATION_SCHEMA
        or source.get("derived_from_decision_sha256") != file_sha256(decision_path)
        or source.get("handoff") != expected_handoff
        or source.get("selected_candidate") != decision.get("selected_candidate")
        or source.get("component_claim_retained")
        != decision.get("component_claim_retained")
        or source.get("selected_checkpoint_set") != expected_selection
        or source.get("source_tier3_design_authorized") is not expected_go
        or source.get("outer_target_access_authorized") is not False
        or target.get("schema_version") != TARGET_AUTHORIZATION_SCHEMA
        or target.get("derived_from_tier2r_decision_sha256")
        != file_sha256(decision_path)
        or target.get("source_authorization_sha256") != file_sha256(source_path)
        or target.get("outer_target_access_authorized") is not False
        or target.get("outer_target_image_access_authorized") is not False
        or target.get("outer_target_label_access_authorized") is not False
        or evidence.get("exact_raw_logit_states_are_primary") is not True
        or evidence.get("dense_grid_used_for_decision") is not False
        or completion.get("schema_version")
        != "rc-irstd-aaai27-tier2r-exact-gate-completion-v1"
        or completion.get("completed") is not True
        or completion.get("handoff_sha256") != file_sha256(handoff)
        or completion.get("decision_sha256") != file_sha256(decision_path)
        or completion.get("evidence_manifest_sha256") != file_sha256(evidence_path)
        or completion.get("source_authorization_sha256") != file_sha256(source_path)
        or completion.get("outer_target_authorization_sha256")
        != file_sha256(target_path)
    ):
        raise RuntimeError("frozen Tier2R gate chain drift")
    return {
        "decision": decision["decision"],
        "selected_candidate": decision.get("selected_candidate"),
        "component_claim_retained": decision.get("component_claim_retained"),
        "decision_path": str(decision_path),
        "source_authorization_path": str(source_path),
        "outer_target_authorization_path": str(target_path),
        "completion_path": str(completion_path),
        "verified_only": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--handoff", default=str(PROJECT_ROOT / AUDIT_RELATIVE / HANDOFF_NAME)
    )
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / GATE_RELATIVE))
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_only:
        result = verify_frozen_gate(
            handoff_path=args.handoff, output_root=args.output_root
        )
    else:
        result = run_gate(handoff_path=args.handoff, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BUDGETS",
    "ExactAPI",
    "FOLDS",
    "MATCHING_PROTOCOL",
    "NINE_CRITERIA",
    "ROLES",
    "RUN_SPECS",
    "SEEDS",
    "TIER2R_GO",
    "TIER2R_HOLD",
    "evaluate_gate_level",
    "evaluate_two_level_decision",
    "run_gate",
    "validate_handoff",
    "validate_source_path",
    "verify_frozen_gate",
    "verify_historical_hold",
]
