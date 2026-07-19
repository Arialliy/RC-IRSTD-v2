"""Strict two-fold Gate C for AnchorWarp versus train-only RC-Direct.

This is deliberately separate from :mod:`risk_curve.aggregate_gate_c_v4`.
The legacy five-method Gate C replay contract is therefore unchanged.  The
present module consumes exactly the two sealed AnchorWarp comparison JSON
files, revalidates every referenced archive/checkpoint byte binding, replays
the two-phase evaluator, reconstructs all metrics from per-episode sufficient
counts, and applies the registered Gate C without weakening any threshold.

In particular, the literal Pd comparison is stricter here: AnchorWarp is not
allowed to lose any Pd against the paired train-only RC-Direct model (up to the
global numerical tolerance).  A decision produced here is source-only and is
never an outer-domain claim.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .bind_anchor_warp_validation_v4 import validate_anchor_warp_bound_package
from .evaluate_anchor_warp_source_pseudo_target_v4 import (
    ANCHOR_WARP_SOURCE_COMPARISON_SCHEMA_VERSION,
    evaluate_anchor_warp_source_comparison,
)
from .evaluate_source_pseudo_target_v4 import (
    FORMAL_OUTER_DOMAIN_KEY,
    FORMAL_SOURCE_DOMAIN_KEYS,
)
from .representation import LOGIT_REPRESENTATION
from .train_anchor_warp_predictor_v4 import (
    validate_anchor_warp_train_only_checkpoint,
)
from .train_direct_calibrator_train_only_v4 import (
    validate_train_only_direct_checkpoint,
)


AGGREGATE_ANCHOR_WARP_GATE_C_SCHEMA_VERSION = (
    "rc-v4-anchor-warp-gate-c-aggregate-decision-v1"
)
AGGREGATE_ANCHOR_WARP_GATE_C_POLICY_VERSION = (
    "rc-v4-anchor-warp-gate-c-strict-cross-fold-v1"
)
RUNTIME_CODE_TREE_SCHEMA_VERSION = "rc-v4-anchor-warp-gate-c-runtime-code-tree-v1"
RUNTIME_CODE_TREE_FILES = (
    "risk_curve/evaluate_anchor_warp_source_pseudo_target_v4.py",
    "risk_curve/anchor_warp_inference.py",
    "risk_curve/anchor_warp_predictor.py",
    "risk_curve/anchor_warp_training.py",
    "risk_curve/train_anchor_warp_predictor_v4.py",
    "risk_curve/bind_anchor_warp_validation_v4.py",
    "risk_curve/train_direct_calibrator_train_only_v4.py",
    "risk_curve/aggregate_anchor_warp_gate_c_v4.py",
)
COMPARISON_PROTOCOL = "source_only_two_phase_action_then_future_e_evaluation"
METHODS = ("risk_curve", "rc_direct")
REQUIRED_FOLD_COUNT = 2
FORMAL_BUDGETS = ((1.0e-5, 5.0), (1.0e-6, 1.0))
RELATIVE_BENEFIT_THRESHOLD = 0.20
STRICT_PD_GAIN_THRESHOLD = 0.03
# This must not be relaxed to the legacy single-fold -2 percentage-point
# diagnostic allowance.  Gate C here requires literal non-decrease.
PD_NON_DEGRADATION_FLOOR = 0.0
REJECT_RATE_LIMIT = 0.20
NUMERIC_ATOL = 1.0e-12
FORMAL_BUDGET_REL_TOL = 1.0e-7


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_runtime_code_bytes(path: Path) -> bytes:
    """Read a runtime-code file through one patchable audit boundary."""

    return path.read_bytes()


def _snapshot_runtime_code_tree() -> dict[str, Any]:
    """Capture the exact minimal runtime code tree at aggregate entry."""

    repository_root = Path(__file__).resolve().parents[1]
    files: dict[str, dict[str, Any]] = {}
    for relative_path in RUNTIME_CODE_TREE_FILES:
        path = (repository_root / relative_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Runtime code dependency does not exist: {path}")
        raw = _read_runtime_code_bytes(path)
        files[relative_path] = {
            "path": str(path),
            "sha256": _sha256_bytes(raw),
            "size_bytes": len(raw),
        }
    if len(files) != len(RUNTIME_CODE_TREE_FILES):
        raise ValueError("Runtime code tree contains duplicate file entries")
    return {
        "schema_version": RUNTIME_CODE_TREE_SCHEMA_VERSION,
        "entry_snapshot_before_input_load": True,
        "required_file_count": len(RUNTIME_CODE_TREE_FILES),
        "required_files": list(RUNTIME_CODE_TREE_FILES),
        "files": files,
    }


def _verify_runtime_code_tree(snapshot: Mapping[str, Any]) -> None:
    """Fail closed if any snapshotted runtime-code path, size, or hash drifts."""

    if snapshot.get("schema_version") != RUNTIME_CODE_TREE_SCHEMA_VERSION:
        raise ValueError("Runtime code tree snapshot schema mismatch")
    if snapshot.get("required_files") != list(RUNTIME_CODE_TREE_FILES):
        raise ValueError("Runtime code tree required-file order mismatch")
    files = _mapping(snapshot.get("files"), "runtime_code_tree.files")
    if set(files) != set(RUNTIME_CODE_TREE_FILES):
        raise ValueError("Runtime code tree file matrix mismatch")
    for relative_path in RUNTIME_CODE_TREE_FILES:
        record = _mapping(
            files[relative_path], f"runtime_code_tree.files.{relative_path}"
        )
        path = Path(
            _text(record.get("path"), f"runtime_code_tree.files.{relative_path}.path")
        ).resolve()
        if not path.is_file():
            raise ValueError(f"Runtime code dependency disappeared: {relative_path}")
        raw = _read_runtime_code_bytes(path)
        expected_size = _integer(
            record.get("size_bytes"),
            f"runtime_code_tree.files.{relative_path}.size_bytes",
            positive=True,
        )
        expected_sha = _sha256(
            record.get("sha256"),
            f"runtime_code_tree.files.{relative_path}.sha256",
        )
        if len(raw) != expected_size or _sha256_bytes(raw) != expected_sha:
            raise ValueError(f"Runtime code dependency drifted: {relative_path}")


def _load_json_snapshot(path: Path, *, role: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"{role} does not exist: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} is invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{role} must contain a JSON object")
    return payload, _sha256_bytes(raw)


def _load_torch_snapshot(
    path: Path, *, role: str
) -> tuple[Mapping[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"{role} does not exist: {path}")
    raw = path.read_bytes()
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{role} must contain a mapping")
    return payload, _sha256_bytes(raw)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _sha256(value: Any, field: str) -> str:
    digest = _text(value, field).lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _boolean(value: Any, field: str, expected: bool | None = None) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    if expected is not None and value is not expected:
        raise ValueError(f"{field} must be {expected}")
    return value


def _integer(value: Any, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return int(value)


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _assert_close(recorded: Any, expected: float, field: str) -> None:
    value = _finite(recorded, field)
    if not math.isclose(value, expected, rel_tol=1.0e-12, abs_tol=NUMERIC_ATOL):
        raise ValueError(f"{field} does not match reconstructed evidence")


def _domain_key(value: Any, field: str) -> str:
    text = _text(value, field)
    key = "".join(character for character in text.casefold() if character.isalnum())
    if key.endswith("sirst"):
        key = key[: -len("sirst")]
    if not key:
        raise ValueError(f"{field} normalises to an empty domain key")
    return key


def _resolve_bound_file(
    path_value: Any,
    digest_value: Any,
    field: str,
    tracked: dict[Path, str],
) -> Path:
    path = Path(_text(path_value, f"{field}.path")).expanduser().resolve()
    expected = _sha256(digest_value, f"{field}.sha256")
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(f"{field} SHA-256 mismatch")
    prior = tracked.get(path)
    if prior is not None and prior != expected:
        raise ValueError(f"{field} path is bound to two different byte hashes")
    tracked[path] = expected
    return path


def _same_resolved_path(left: Any, right: Path, field: str) -> None:
    observed = Path(_text(left, field)).expanduser().resolve()
    if observed != right:
        raise ValueError(f"{field} does not resolve to the comparison-bound path")


def _budget_pairs(payload: Mapping[str, Any], owner: str) -> list[tuple[float, float]]:
    rows = _list(payload.get("budgets"), f"{owner}.budgets")
    if len(rows) != len(FORMAL_BUDGETS):
        raise ValueError(f"{owner} must contain exactly two formal budgets")
    result: list[tuple[float, float]] = []
    for position, (raw, formal) in enumerate(zip(rows, FORMAL_BUDGETS)):
        row = _mapping(raw, f"{owner}.budgets[{position}]")
        if _integer(
            row.get("budget_position"),
            f"{owner}.budgets[{position}].budget_position",
        ) != position:
            raise ValueError(f"{owner} budget positions are not ordered")
        pair = (
            _finite(row.get("pixel_budget"), f"{owner}.budgets[{position}].pixel_budget"),
            _finite(
                row.get("component_budget"),
                f"{owner}.budgets[{position}].component_budget",
            ),
        )
        if not all(
            math.isclose(value, expected, rel_tol=FORMAL_BUDGET_REL_TOL, abs_tol=1e-15)
            for value, expected in zip(pair, formal)
        ):
            raise ValueError(
                f"{owner}.budgets[{position}] is not formal Gate C budget {formal!r}"
            )
        result.append(pair)
    return result


def _validate_action(
    raw: Any,
    *,
    method: str,
    thresholds: Sequence[float],
    pixel_budget: float,
    component_budget: float,
    budget_position: int,
    field: str,
) -> dict[str, Any]:
    action = _mapping(raw, field)
    reject = _boolean(action.get("reject"), f"{field}.reject")
    index_raw = action.get("threshold_index")
    selected = action.get("selected_logit_threshold")
    if reject:
        if index_raw is not None or selected != "+inf":
            raise ValueError(f"{field} rejected action must use null index and '+inf'")
        index: int | None = None
    else:
        index = _integer(index_raw, f"{field}.threshold_index")
        if index >= len(thresholds):
            raise ValueError(f"{field}.threshold_index is outside the finite grid")
        _assert_close(selected, float(thresholds[index]), f"{field}.selected_logit_threshold")

    if method == "risk_curve":
        if action.get("model_class") != "CountAllAnchorWarpRiskCurve":
            raise ValueError(f"{field}.model_class is not AnchorWarp")
        if action.get("selection_rule") != "earliest_jointly_feasible_grid_index":
            raise ValueError(f"{field}.selection_rule mismatch")
        for name in (
            "predicted_pixel_log_risk_at_action",
            "predicted_component_log_risk_at_action",
        ):
            value = action.get(name)
            if reject:
                if value is not None:
                    raise ValueError(f"{field}.{name} must be null for rejection")
            else:
                _finite(value, f"{field}.{name}")
    elif method == "rc_direct":
        if action.get("selection_rule") != "conservative_left_grid_quantization":
            raise ValueError(f"{field}.selection_rule mismatch")
        if _integer(
            action.get("registered_budget_index"),
            f"{field}.registered_budget_index",
        ) != budget_position:
            raise ValueError(f"{field} uses the wrong registered budget")
        _finite(
            action.get("predicted_logit_before_quantization"),
            f"{field}.predicted_logit_before_quantization",
        )
    else:  # pragma: no cover - guarded by the exact method matrix
        raise ValueError(f"Unknown method {method!r}")

    pixel_fp = _integer(action.get("pixel_fp_count"), f"{field}.pixel_fp_count")
    component_fp = _integer(
        action.get("component_fp_count"), f"{field}.component_fp_count"
    )
    tp = _integer(action.get("tp_object_count"), f"{field}.tp_object_count")
    gt = _integer(action.get("gt_object_count"), f"{field}.gt_object_count")
    total_pixels = _integer(
        action.get("total_pixels"), f"{field}.total_pixels", positive=True
    )
    if tp > gt:
        raise ValueError(f"{field}.tp_object_count exceeds gt_object_count")
    if reject and any(value != 0 for value in (pixel_fp, component_fp, tp)):
        raise ValueError(f"{field} rejected action contributes non-zero FP/TP")
    pixel_risk = pixel_fp / float(total_pixels)
    component_risk = component_fp / (total_pixels / 1_000_000.0)
    pd = tp / float(max(gt, 1))
    pixel_excess = max(pixel_risk / pixel_budget - 1.0, 0.0)
    component_excess = max(component_risk / component_budget - 1.0, 0.0)
    rebuilt: dict[str, Any] = {
        "threshold_index": index,
        "selected_logit_threshold": "+inf" if reject else float(thresholds[index]),
        "reject": reject,
        "pixel_fp_count": pixel_fp,
        "component_fp_count": component_fp,
        "tp_object_count": tp,
        "gt_object_count": gt,
        "total_pixels": total_pixels,
        "pd": pd,
        "pixel_risk": pixel_risk,
        "component_risk": component_risk,
        "pixel_budget_violated": pixel_risk > pixel_budget,
        "component_budget_violated": component_risk > component_budget,
        "joint_budget_violated": (
            pixel_risk > pixel_budget or component_risk > component_budget
        ),
        "pixel_relative_excess": pixel_excess,
        "component_relative_excess": component_excess,
        "joint_relative_excess": max(pixel_excess, component_excess),
    }
    for name in (
        "pd",
        "pixel_risk",
        "component_risk",
        "pixel_relative_excess",
        "component_relative_excess",
        "joint_relative_excess",
    ):
        _assert_close(action.get(name), float(rebuilt[name]), f"{field}.{name}")
    for name in (
        "pixel_budget_violated",
        "component_budget_violated",
        "joint_budget_violated",
    ):
        if _boolean(action.get(name), f"{field}.{name}") is not rebuilt[name]:
            raise ValueError(f"{field}.{name} does not match sufficient counts")
    return rebuilt


def _aggregate_actions(actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not actions:
        raise ValueError("Cannot aggregate an empty action sequence")
    episodes = len(actions)
    total_pixels = sum(int(action["total_pixels"]) for action in actions)
    gt = sum(int(action["gt_object_count"]) for action in actions)
    tp = sum(int(action["tp_object_count"]) for action in actions)
    pixel_fp = sum(int(action["pixel_fp_count"]) for action in actions)
    component_fp = sum(int(action["component_fp_count"]) for action in actions)
    violations = sum(bool(action["joint_budget_violated"]) for action in actions)
    rejects = sum(bool(action["reject"]) for action in actions)
    excess_sum = sum(float(action["joint_relative_excess"]) for action in actions)
    indices = sorted(
        {
            int(action["threshold_index"])
            for action in actions
            if not bool(action["reject"])
        }
    )
    return {
        "aggregate_counts": {
            "num_episodes": episodes,
            "total_pixels": total_pixels,
            "gt_object_count": gt,
            "tp_object_count": tp,
            "pixel_fp_count": pixel_fp,
            "component_fp_count": component_fp,
            "joint_violation_count": int(violations),
            "reject_count": int(rejects),
            "sum_joint_relative_excess": excess_sum,
        },
        "num_episodes": episodes,
        "pd": tp / float(max(gt, 1)),
        "pixel_risk": pixel_fp / float(total_pixels),
        "component_risk": component_fp / (total_pixels / 1_000_000.0),
        "mean_episode_pd": sum(float(action["pd"]) for action in actions) / episodes,
        "mean_episode_pixel_risk": (
            sum(float(action["pixel_risk"]) for action in actions) / episodes
        ),
        "mean_episode_component_risk": (
            sum(float(action["component_risk"]) for action in actions) / episodes
        ),
        "joint_violation_rate": violations / float(episodes),
        "mean_relative_excess": excess_sum / float(episodes),
        "max_relative_excess": max(
            float(action["joint_relative_excess"]) for action in actions
        ),
        "reject_rate": rejects / float(episodes),
        "unique_finite_indices": indices,
        "num_unique_finite_indices": len(indices),
    }


def _validate_recorded_aggregate(raw: Any, rebuilt: Mapping[str, Any], field: str) -> None:
    recorded = _mapping(raw, field)
    for name in ("num_episodes", "num_unique_finite_indices"):
        if _integer(recorded.get(name), f"{field}.{name}") != rebuilt[name]:
            raise ValueError(f"{field}.{name} does not match per-episode evidence")
    for name in (
        "pd",
        "pixel_risk",
        "component_risk",
        "mean_episode_pd",
        "mean_episode_pixel_risk",
        "mean_episode_component_risk",
        "joint_violation_rate",
        "mean_relative_excess",
        "max_relative_excess",
        "reject_rate",
    ):
        _assert_close(recorded.get(name), float(rebuilt[name]), f"{field}.{name}")
    if _list(
        recorded.get("unique_finite_indices"), f"{field}.unique_finite_indices"
    ) != rebuilt["unique_finite_indices"]:
        raise ValueError(f"{field}.unique_finite_indices mismatch")


def _action_digest(records: Sequence[Mapping[str, Any]]) -> str:
    raw = json.dumps(
        list(records), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha256_bytes(raw)


def _mvr(actions: Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]], method: str, grid_size: int) -> dict[str, Any]:
    rows = len(actions[0][method])
    violations = 0
    for row in range(rows):
        codes = [
            grid_size
            if bool(actions[position][method][row]["reject"])
            else int(actions[position][method][row]["threshold_index"])
            for position in sorted(actions)
        ]
        violations += int(any(right < left for left, right in zip(codes, codes[1:])))
    return {
        "num_episodes": rows,
        "violating_episode_count": violations,
        "rate": violations / float(rows),
    }


def _macro_aggregate(fold_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not fold_metrics:
        raise ValueError("Cannot macro-average an empty fold set")
    fields = (
        "pd",
        "pixel_risk",
        "component_risk",
        "mean_episode_pd",
        "mean_episode_pixel_risk",
        "mean_episode_component_risk",
        "joint_violation_rate",
        "mean_relative_excess",
        "reject_rate",
    )
    count = float(len(fold_metrics))
    result = {
        field: sum(float(row[field]) for row in fold_metrics) / count
        for field in fields
    }
    result.update(
        {
            "num_folds": len(fold_metrics),
            "fold_episode_counts": [int(row["num_episodes"]) for row in fold_metrics],
            "mean_of_fold_max_relative_excess": sum(
                float(row["max_relative_excess"]) for row in fold_metrics
            )
            / count,
            "global_max_relative_excess": max(
                float(row["max_relative_excess"]) for row in fold_metrics
            ),
        }
    )
    return result


def _relation(proposed: float, baseline: float) -> dict[str, Any]:
    delta = proposed - baseline
    if baseline <= NUMERIC_ATOL:
        if proposed <= NUMERIC_ATOL:
            status = "tie_at_zero"
            reduction: float | None = 0.0
        else:
            status = "worse_from_zero"
            reduction = None
    else:
        status = "finite_relative_reduction"
        reduction = (baseline - proposed) / baseline
    return {
        "risk_curve": proposed,
        "rc_direct": baseline,
        "delta": delta,
        "relative_reduction": reduction,
        "status": status,
        "non_adverse": proposed <= baseline + NUMERIC_ATOL,
    }


def _anchor_parent_fields_equal(
    embedded: Mapping[str, Any], parent: Mapping[str, Any]
) -> None:
    fields = (
        "checkpoint_schema_version",
        "artifact_stage",
        "method_name",
        "model_class",
        "model_architecture_version",
        "parameter_count",
        "representation",
        "state_dict_semantic_sha256",
        "policy_semantic_sha256",
        "threshold_grid_sha256",
        "threshold_grid_manifest_sha256",
        "feature_schema_sha256",
        "train_archive",
        "train_archive_sha256",
        "train_pseudo_targets",
        "seed",
        "selected_epoch",
        "final_refit_epochs",
        "train_episode_keys_sha256",
        "fold_assignment_sha256",
        "cv_history_sha256",
        "oof_prediction_sha256",
    )
    for field in fields:
        if embedded.get(field) != parent.get(field):
            raise ValueError(
                f"AnchorWarp bound package embedded checkpoint differs from parent: {field}"
            )


def _parse_fold(
    *,
    fold_id: str,
    file_path: Path,
    payload: Mapping[str, Any],
    file_sha256: str,
    tracked: dict[Path, str],
) -> dict[str, Any]:
    owner = f"folds.{fold_id}"
    if payload.get("schema_version") != ANCHOR_WARP_SOURCE_COMPARISON_SCHEMA_VERSION:
        raise ValueError(f"{owner} comparison schema mismatch")
    if payload.get("protocol") != COMPARISON_PROTOCOL:
        raise ValueError(f"{owner} comparison protocol mismatch")
    if payload.get("representation") != LOGIT_REPRESENTATION:
        raise ValueError(f"{owner} must use {LOGIT_REPRESENTATION}")
    if payload.get("device") != "cpu":
        raise ValueError(f"{owner} must be the deterministic CPU comparison")
    _boolean(
        payload.get("labels_used_for_action_selection"),
        f"{owner}.labels_used_for_action_selection",
        False,
    )
    _boolean(
        payload.get("future_e_arrays_loaded_before_action_digest"),
        f"{owner}.future_e_arrays_loaded_before_action_digest",
        False,
    )
    _boolean(
        payload.get("source_pseudo_target_labels_used_for_post_selection_evaluation"),
        f"{owner}.source_pseudo_target_labels_used_for_post_selection_evaluation",
        True,
    )
    _boolean(
        payload.get("outer_target_labels_used"),
        f"{owner}.outer_target_labels_used",
        False,
    )
    sources = _list(payload.get("formal_source_domains"), f"{owner}.formal_source_domains")
    if {_domain_key(item, f"{owner}.formal_source_domains") for item in sources} != FORMAL_SOURCE_DOMAIN_KEYS:
        raise ValueError(f"{owner} formal source domains are not canonical")
    if _domain_key(payload.get("excluded_outer_target"), f"{owner}.excluded_outer_target") != FORMAL_OUTER_DOMAIN_KEY:
        raise ValueError(f"{owner} excluded outer target is not NUAA-SIRST")
    validation_target = _text(
        payload.get("validation_pseudo_target"), f"{owner}.validation_pseudo_target"
    )
    validation_key = _domain_key(validation_target, f"{owner}.validation_pseudo_target")
    if validation_key not in FORMAL_SOURCE_DOMAIN_KEYS:
        raise ValueError(f"{owner} validation pseudo-target is not canonical")
    pseudo_targets = _list(payload.get("pseudo_targets"), f"{owner}.pseudo_targets")
    if len(pseudo_targets) != 1 or _domain_key(
        pseudo_targets[0], f"{owner}.pseudo_targets[0]"
    ) != validation_key:
        raise ValueError(f"{owner} pseudo-target list does not match the held fold")
    budgets = _budget_pairs(payload, owner)
    num_episodes = _integer(
        payload.get("num_episodes"), f"{owner}.num_episodes", positive=True
    )
    grid_size = _integer(
        payload.get("threshold_grid_size"), f"{owner}.threshold_grid_size", positive=True
    )
    raw_seed = payload.get("seed")
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
        raise ValueError(f"{owner}.seed must be an integer")
    seed = int(raw_seed)

    comparison_path = file_path.resolve()
    prior = tracked.get(comparison_path)
    if prior is not None and prior != file_sha256:
        raise ValueError(f"{owner} comparison path has conflicting hashes")
    tracked[comparison_path] = file_sha256
    episode_path = _resolve_bound_file(
        payload.get("episode_archive"),
        payload.get("episode_archive_sha256"),
        f"{owner}.episode_archive",
        tracked,
    )
    anchor_path = _resolve_bound_file(
        payload.get("anchor_warp_package"),
        payload.get("anchor_warp_package_sha256"),
        f"{owner}.anchor_warp_package",
        tracked,
    )
    direct_path = _resolve_bound_file(
        payload.get("rc_direct_checkpoint"),
        payload.get("rc_direct_checkpoint_sha256"),
        f"{owner}.rc_direct_checkpoint",
        tracked,
    )
    anchor_package, anchor_sha = _load_torch_snapshot(
        anchor_path, role=f"{owner} AnchorWarp package"
    )
    direct_checkpoint, direct_sha = _load_torch_snapshot(
        direct_path, role=f"{owner} RC-Direct checkpoint"
    )
    if anchor_sha != payload["anchor_warp_package_sha256"]:
        raise ValueError(f"{owner} AnchorWarp package changed during snapshot")
    if direct_sha != payload["rc_direct_checkpoint_sha256"]:
        raise ValueError(f"{owner} RC-Direct checkpoint changed during snapshot")
    anchor_contract = validate_anchor_warp_bound_package(anchor_package)
    direct_contract = validate_train_only_direct_checkpoint(direct_checkpoint)
    frozen = _mapping(anchor_package.get("frozen_checkpoint"), f"{owner}.frozen_checkpoint")
    binding = _mapping(anchor_package.get("validation_binding"), f"{owner}.validation_binding")
    parent_path = _resolve_bound_file(
        anchor_package.get("parent_frozen_checkpoint_path"),
        anchor_package.get("parent_frozen_checkpoint_sha256"),
        f"{owner}.anchor_parent_checkpoint",
        tracked,
    )
    parent_checkpoint, parent_sha = _load_torch_snapshot(
        parent_path, role=f"{owner} AnchorWarp train-only parent"
    )
    if parent_sha != anchor_package["parent_frozen_checkpoint_sha256"]:
        raise ValueError(f"{owner} AnchorWarp parent changed during snapshot")
    parent_contract = validate_anchor_warp_train_only_checkpoint(parent_checkpoint)
    _anchor_parent_fields_equal(frozen, parent_checkpoint)
    if anchor_contract["state_dict_semantic_sha256"] != parent_contract["state_dict_semantic_sha256"]:
        raise ValueError(f"{owner} AnchorWarp parent/embedded state hash mismatch")
    if anchor_contract["policy_semantic_sha256"] != parent_contract["policy_semantic_sha256"]:
        raise ValueError(f"{owner} AnchorWarp parent/embedded policy hash mismatch")
    if _sha256(
        payload.get("anchor_warp_policy_semantic_sha256"),
        f"{owner}.anchor_warp_policy_semantic_sha256",
    ) != anchor_contract["policy_semantic_sha256"]:
        raise ValueError(f"{owner} comparison AnchorWarp policy hash mismatch")
    if _sha256(
        payload.get("rc_direct_canonical_frozen_model_sha256"),
        f"{owner}.rc_direct_canonical_frozen_model_sha256",
    ) != direct_contract["canonical_frozen_model_sha256"]:
        raise ValueError(f"{owner} comparison RC-Direct frozen hash mismatch")

    episode_sha = _sha256(
        payload.get("episode_archive_sha256"), f"{owner}.episode_archive_sha256"
    )
    if binding.get("validation_archive_sha256") != episode_sha:
        raise ValueError(f"{owner} AnchorWarp validation binding hash mismatch")
    if direct_checkpoint.get("validation_archive_sha256") != episode_sha:
        raise ValueError(f"{owner} RC-Direct validation binding hash mismatch")
    _same_resolved_path(
        binding.get("validation_archive"), episode_path, f"{owner}.validation_binding.archive"
    )
    _same_resolved_path(
        direct_checkpoint.get("validation_archive"), episode_path, f"{owner}.rc_direct.validation_archive"
    )
    if binding.get("validation_num_episodes") != num_episodes:
        raise ValueError(f"{owner} validation episode-count binding mismatch")
    if _domain_key(binding.get("validation_domain"), f"{owner}.binding.validation_domain") != validation_key:
        raise ValueError(f"{owner} validation-domain binding mismatch")

    train_sha = _sha256(
        payload.get("train_archive_sha256"), f"{owner}.train_archive_sha256"
    )
    if frozen.get("train_archive_sha256") != train_sha:
        raise ValueError(f"{owner} AnchorWarp train archive hash mismatch")
    if direct_checkpoint.get("train_archive_sha256") != train_sha:
        raise ValueError(f"{owner} RC-Direct train archive hash mismatch")
    train_path = _resolve_bound_file(
        frozen.get("train_archive"), train_sha, f"{owner}.train_archive", tracked
    )
    _same_resolved_path(
        direct_checkpoint.get("train_archive"), train_path, f"{owner}.rc_direct.train_archive"
    )
    train_targets = _list(frozen.get("train_pseudo_targets"), f"{owner}.train_pseudo_targets")
    if len(train_targets) != 1:
        raise ValueError(f"{owner} AnchorWarp must train on one source domain")
    train_key = _domain_key(train_targets[0], f"{owner}.train_pseudo_targets[0]")
    if {train_key, validation_key} != FORMAL_SOURCE_DOMAIN_KEYS:
        raise ValueError(f"{owner} train/held domains are not complementary")
    if int(frozen.get("seed", -1)) != seed or int(direct_checkpoint.get("seed", -1)) != seed:
        raise ValueError(f"{owner} AnchorWarp/RC-Direct seed binding mismatch")

    shared_fields = (
        "threshold_grid_sha256",
        "threshold_grid_manifest_sha256",
        "feature_schema_sha256",
        "threshold_grid_detector_protocol",
        "threshold_grid_detector_checkpoint_sha256s",
        "threshold_grid_outer_detector_checkpoint_sha256",
        "threshold_grid_episode_detector_checkpoint_sha256s",
    )
    shared_contract: dict[str, Any] = {
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_size": grid_size,
    }
    for field in shared_fields:
        top = payload.get(field)
        anchor_value = frozen.get(field)
        direct_value = direct_checkpoint.get(field)
        if top != anchor_value or top != direct_value:
            raise ValueError(f"{owner} shared contract mismatch: {field}")
        shared_contract[field] = top
    thresholds = [float(value) for value in frozen.get("thresholds", [])]
    if len(thresholds) != grid_size or any(not math.isfinite(value) for value in thresholds):
        raise ValueError(f"{owner} AnchorWarp threshold grid is malformed")
    if any(right <= left for left, right in zip(thresholds, thresholds[1:])):
        raise ValueError(f"{owner} AnchorWarp threshold grid is not strictly increasing")

    raw_episodes = _list(payload.get("per_episode"), f"{owner}.per_episode")
    if len(raw_episodes) != num_episodes:
        raise ValueError(f"{owner} per-episode matrix is incomplete")
    actions: dict[int, dict[str, list[dict[str, Any]]]] = {
        position: {method: [] for method in METHODS}
        for position in range(len(budgets))
    }
    identities: list[dict[str, Any]] = []
    digest_records: list[dict[str, Any]] = []
    parsed_episode_actions: list[list[Mapping[str, Any]]] = []
    for row, raw_episode in enumerate(raw_episodes):
        episode = _mapping(raw_episode, f"{owner}.per_episode[{row}]")
        if _integer(
            episode.get("episode_index"), f"{owner}.per_episode[{row}].episode_index"
        ) != row:
            raise ValueError(f"{owner} episode indices are not ordered")
        if _domain_key(
            episode.get("pseudo_target"), f"{owner}.per_episode[{row}].pseudo_target"
        ) != validation_key:
            raise ValueError(f"{owner} episode pseudo-target mismatch")
        adaptation = [
            _text(value, f"{owner}.per_episode[{row}].adaptation_ids")
            for value in _list(
                episode.get("adaptation_ids"),
                f"{owner}.per_episode[{row}].adaptation_ids",
            )
        ]
        evaluation = [
            _text(value, f"{owner}.per_episode[{row}].evaluation_ids")
            for value in _list(
                episode.get("evaluation_ids"),
                f"{owner}.per_episode[{row}].evaluation_ids",
            )
        ]
        if not adaptation or not evaluation:
            raise ValueError(f"{owner} episode windows must not be empty")
        if len(set(adaptation)) != len(adaptation) or len(set(evaluation)) != len(evaluation):
            raise ValueError(f"{owner} episode windows contain duplicate image IDs")
        if set(adaptation).intersection(evaluation):
            raise ValueError(f"{owner} adaptation and future-E windows overlap")
        identities.append(
            {
                "episode_index": row,
                "pseudo_target": validation_target,
                "adaptation_ids": adaptation,
                "evaluation_ids": evaluation,
            }
        )
        budget_actions = _list(
            episode.get("actions"), f"{owner}.per_episode[{row}].actions"
        )
        if len(budget_actions) != len(budgets):
            raise ValueError(f"{owner} episode action matrix is incomplete")
        parsed_episode_actions.append(
            [_mapping(item, f"{owner}.per_episode[{row}].actions") for item in budget_actions]
        )

    # The digest writer used budget-major, episode-minor order.  Reconstruct in
    # that exact order before any aggregate statistic is trusted.
    for position, (pixel_budget, component_budget) in enumerate(budgets):
        for row in range(num_episodes):
            record = parsed_episode_actions[row][position]
            if _integer(
                record.get("budget_position"),
                f"{owner}.per_episode[{row}].actions[{position}].budget_position",
            ) != position:
                raise ValueError(f"{owner} episode budget positions are not ordered")
            _assert_close(
                record.get("pixel_budget"), pixel_budget,
                f"{owner}.per_episode[{row}].actions[{position}].pixel_budget",
            )
            _assert_close(
                record.get("component_budget"), component_budget,
                f"{owner}.per_episode[{row}].actions[{position}].component_budget",
            )
            methods = _mapping(
                record.get("methods"),
                f"{owner}.per_episode[{row}].actions[{position}].methods",
            )
            if set(methods) != set(METHODS):
                raise ValueError(f"{owner} paired method matrix is incomplete")
            parsed_methods: dict[str, dict[str, Any]] = {}
            for method in METHODS:
                parsed = _validate_action(
                    methods[method],
                    method=method,
                    thresholds=thresholds,
                    pixel_budget=pixel_budget,
                    component_budget=component_budget,
                    budget_position=position,
                    field=(
                        f"{owner}.per_episode[{row}].actions[{position}].methods.{method}"
                    ),
                )
                actions[position][method].append(parsed)
                parsed_methods[method] = parsed
            digest_records.append(
                {
                    "budget_position": position,
                    "episode_index": row,
                    "anchor_warp": {
                        "threshold_index": parsed_methods["risk_curve"]["threshold_index"],
                        "threshold": parsed_methods["risk_curve"]["selected_logit_threshold"],
                        "reject": parsed_methods["risk_curve"]["reject"],
                    },
                    "rc_direct": {
                        "threshold_index": parsed_methods["rc_direct"]["threshold_index"],
                        "threshold": parsed_methods["rc_direct"]["selected_logit_threshold"],
                        "reject": parsed_methods["rc_direct"]["reject"],
                    },
                }
            )

    freeze = _mapping(payload.get("action_freeze"), f"{owner}.action_freeze")
    _boolean(
        freeze.get("frozen_before_future_e_load"),
        f"{owner}.action_freeze.frozen_before_future_e_load",
        True,
    )
    _boolean(
        freeze.get("future_e_reselection_performed"),
        f"{owner}.action_freeze.future_e_reselection_performed",
        False,
    )
    expected_num_actions = num_episodes * len(budgets)
    if _integer(
        freeze.get("num_actions"), f"{owner}.action_freeze.num_actions"
    ) != expected_num_actions:
        raise ValueError(f"{owner} action-freeze matrix size mismatch")
    reconstructed_digest = _action_digest(digest_records)
    if _sha256(
        freeze.get("action_digest_sha256"),
        f"{owner}.action_freeze.action_digest_sha256",
    ) != reconstructed_digest:
        raise ValueError(f"{owner} frozen action digest mismatch")

    aggregates: dict[int, dict[str, dict[str, Any]]] = {}
    raw_budgets = _list(payload.get("budgets"), f"{owner}.budgets")
    for position in range(len(budgets)):
        aggregate_methods = _mapping(
            _mapping(raw_budgets[position], f"{owner}.budgets[{position}]").get("methods"),
            f"{owner}.budgets[{position}].methods",
        )
        if set(aggregate_methods) != set(METHODS):
            raise ValueError(f"{owner} recorded aggregate method matrix is incomplete")
        aggregates[position] = {}
        for method in METHODS:
            rebuilt = _aggregate_actions(actions[position][method])
            _validate_recorded_aggregate(
                aggregate_methods[method], rebuilt,
                f"{owner}.budgets[{position}].methods.{method}",
            )
            aggregates[position][method] = rebuilt

    monotonicity = {
        method: _mvr(actions, method, grid_size) for method in METHODS
    }
    recorded_mvr = _mapping(
        payload.get("monotonic_violation_rates"), f"{owner}.monotonic_violation_rates"
    )
    if set(recorded_mvr) != set(METHODS):
        raise ValueError(f"{owner} monotonicity method matrix is incomplete")
    for method in METHODS:
        _assert_close(
            recorded_mvr[method], monotonicity[method]["rate"],
            f"{owner}.monotonic_violation_rates.{method}",
        )

    # The direct checkpoint validator above verifies the immutable six-event
    # train-read/CV/freeze/held-read ordering.  Preserve the exact event proof
    # in the aggregate artifact rather than reducing it to a boolean.
    direct_protocol = _mapping(
        direct_checkpoint.get("train_only_selection_protocol"),
        f"{owner}.rc_direct.train_only_selection_protocol",
    )
    return {
        "file": str(comparison_path),
        "file_sha256": file_sha256,
        "validation_target": validation_target,
        "validation_key": validation_key,
        "train_target": train_targets[0],
        "train_key": train_key,
        "num_episodes": num_episodes,
        "seed": seed,
        "budgets": budgets,
        "shared_contract": shared_contract,
        "actions": actions,
        "aggregates": aggregates,
        "monotonicity": monotonicity,
        "identities": identities,
        "artifacts": {
            "validation_episode_archive": {
                "path": str(episode_path), "sha256": episode_sha
            },
            "train_episode_archive": {"path": str(train_path), "sha256": train_sha},
            "anchor_warp_bound_package": {
                "path": str(anchor_path), "sha256": anchor_sha
            },
            "anchor_warp_train_only_parent": {
                "path": str(parent_path), "sha256": parent_sha
            },
            "rc_direct_train_only_checkpoint": {
                "path": str(direct_path), "sha256": direct_sha
            },
        },
        "checkpoint_binding": {
            "anchor_train_only_checkpoint_validated": True,
            "anchor_bound_package_validated": True,
            "anchor_parent_bytes_and_embedded_semantics_match": True,
            "anchor_state_dict_semantic_sha256": anchor_contract[
                "state_dict_semantic_sha256"
            ],
            "anchor_policy_semantic_sha256": anchor_contract[
                "policy_semantic_sha256"
            ],
            "anchor_state_frozen_before_validation_binding": True,
            "anchor_validation_labels_read_by_binder": False,
            "rc_direct_train_only_checkpoint_validated": True,
            "rc_direct_canonical_frozen_model_sha256": direct_contract[
                "canonical_frozen_model_sha256"
            ],
            "identical_train_archive_bytes": True,
            "identical_seed": True,
            "validation_archive_bound_post_freeze": True,
            "direct_train_only_read_event_sequence": direct_protocol[
                "read_event_sequence"
            ],
        },
        "action_order_proof": {
            "action_digest_sha256": reconstructed_digest,
            "num_actions": expected_num_actions,
            "digest_order": "budget_major_then_episode_minor",
            "phase_a_label_free_action_selection": True,
            "digest_frozen_before_future_e_load": True,
            "future_e_reselection_performed": False,
        },
        "episode_archive_sha256": episode_sha,
        "anchor_path": anchor_path,
        "direct_path": direct_path,
        "episode_path": episode_path,
    }


def _replay_fold_exact(parsed: Mapping[str, Any]) -> dict[str, Any]:
    """Rerun the registered two-phase evaluator and require byte identity."""

    with tempfile.TemporaryDirectory(prefix="rc-v4-anchor-gate-c-replay-") as root:
        replay_path = Path(root) / "comparison.json"
        evaluate_anchor_warp_source_comparison(
            episode_file=parsed["episode_path"],
            anchor_warp_package=parsed["anchor_path"],
            rc_direct_checkpoint=parsed["direct_path"],
            output=replay_path,
            pixel_budgets=[pair[0] for pair in parsed["budgets"]],
            component_budgets=[pair[1] for pair in parsed["budgets"]],
            device="cpu",
            batch_size=64,
        )
        replay_sha = _sha256_file(replay_path)
        if replay_sha != parsed["file_sha256"]:
            raise ValueError(
                f"Two-phase evaluator replay is not byte-exact for {parsed['validation_target']}"
            )
    evaluator_path = Path(
        evaluate_anchor_warp_source_comparison.__code__.co_filename
    ).resolve()
    return {
        "performed": True,
        "device": "cpu",
        "batch_size": 64,
        "comparison_replay_byte_exact": True,
        "replayed_comparison_sha256": replay_sha,
        "evaluator_path": str(evaluator_path),
        "evaluator_sha256": _sha256_file(evaluator_path),
        "phase_order": [
            "A_load_explicit_label_free_allowlist",
            "A_predict_and_freeze_complete_action_matrix",
            "A_compute_action_digest",
            "E_load_future_e_sufficient_counts",
            "E_evaluate_without_reselection",
        ],
    }


def _absolute_feasible(
    metrics: Mapping[str, Any], *, pixel_budget: float, component_budget: float
) -> bool:
    return bool(
        float(metrics["pixel_risk"]) <= pixel_budget + NUMERIC_ATOL
        and float(metrics["component_risk"]) <= component_budget + NUMERIC_ATOL
    )


def _build_decision(
    *,
    ordered_ids: Sequence[str],
    budgets: Sequence[tuple[float, float]],
    fold_aggregates: Mapping[str, Mapping[int, Mapping[str, Mapping[str, Any]]]],
    macro: Mapping[int, Mapping[str, Mapping[str, Any]]],
    micro: Mapping[int, Mapping[str, Mapping[str, Any]]],
    fold_mvr: Mapping[str, Mapping[str, Mapping[str, Any]]],
    macro_mvr: Mapping[str, float],
    micro_mvr: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    relations: dict[int, dict[str, Any]] = {}
    for position in range(len(budgets)):
        per_fold: dict[str, Any] = {}
        for fold_id in ordered_ids:
            risk = fold_aggregates[fold_id][position]["risk_curve"]
            direct = fold_aggregates[fold_id][position]["rc_direct"]
            per_fold[fold_id] = {
                "pd_delta": float(risk["pd"]) - float(direct["pd"]),
                "joint_violation_rate": _relation(
                    float(risk["joint_violation_rate"]),
                    float(direct["joint_violation_rate"]),
                ),
                "mean_relative_excess": _relation(
                    float(risk["mean_relative_excess"]),
                    float(direct["mean_relative_excess"]),
                ),
                "max_relative_excess": _relation(
                    float(risk["max_relative_excess"]),
                    float(direct["max_relative_excess"]),
                ),
            }
        risk_macro = macro[position]["risk_curve"]
        direct_macro = macro[position]["rc_direct"]
        risk_micro = micro[position]["risk_curve"]
        direct_micro = micro[position]["rc_direct"]
        relations[position] = {
            "budget_position": position,
            "pixel_budget": budgets[position][0],
            "component_budget": budgets[position][1],
            "per_fold": per_fold,
            "macro": {
                "pd_delta": float(risk_macro["pd"]) - float(direct_macro["pd"]),
                "joint_violation_rate": _relation(
                    float(risk_macro["joint_violation_rate"]),
                    float(direct_macro["joint_violation_rate"]),
                ),
                "mean_relative_excess": _relation(
                    float(risk_macro["mean_relative_excess"]),
                    float(direct_macro["mean_relative_excess"]),
                ),
                "max_relative_excess": _relation(
                    float(risk_macro["global_max_relative_excess"]),
                    float(direct_macro["global_max_relative_excess"]),
                ),
            },
            "micro": {
                "pd_delta": float(risk_micro["pd"]) - float(direct_micro["pd"]),
                "joint_violation_rate": _relation(
                    float(risk_micro["joint_violation_rate"]),
                    float(direct_micro["joint_violation_rate"]),
                ),
                "mean_relative_excess": _relation(
                    float(risk_micro["mean_relative_excess"]),
                    float(direct_micro["mean_relative_excess"]),
                ),
                "max_relative_excess": _relation(
                    float(risk_micro["max_relative_excess"]),
                    float(direct_micro["max_relative_excess"]),
                ),
            },
        }

    absolute_checks: list[dict[str, Any]] = []
    non_adverse_checks: list[dict[str, Any]] = []
    pd_checks: list[dict[str, Any]] = []
    reject_checks: list[dict[str, Any]] = []
    for position, (pixel_budget, component_budget) in enumerate(budgets):
        metric_scopes = [
            *[
                (f"fold:{fold_id}", fold_aggregates[fold_id][position])
                for fold_id in ordered_ids
            ],
            ("macro", macro[position]),
            ("micro", micro[position]),
        ]
        relation_scopes = [
            *[
                (f"fold:{fold_id}", relations[position]["per_fold"][fold_id])
                for fold_id in ordered_ids
            ],
            ("macro", relations[position]["macro"]),
            ("micro", relations[position]["micro"]),
        ]
        for scope, methods in metric_scopes:
            risk = methods["risk_curve"]
            direct = methods["rc_direct"]
            pixel_ok = float(risk["pixel_risk"]) <= pixel_budget + NUMERIC_ATOL
            component_ok = (
                float(risk["component_risk"]) <= component_budget + NUMERIC_ATOL
            )
            absolute_checks.append(
                {
                    "budget_position": position,
                    "scope": scope,
                    "pixel_risk": risk["pixel_risk"],
                    "component_risk": risk["component_risk"],
                    "pixel_budget_satisfied": pixel_ok,
                    "component_budget_satisfied": component_ok,
                    "joint_budget_satisfied": pixel_ok and component_ok,
                }
            )
            delta = float(risk["pd"]) - float(direct["pd"])
            pd_checks.append(
                {
                    "budget_position": position,
                    "scope": scope,
                    "comparator_method": "rc_direct",
                    "risk_curve_pd": risk["pd"],
                    "rc_direct_pd": direct["pd"],
                    "pd_delta": delta,
                    "non_degraded": (
                        delta >= PD_NON_DEGRADATION_FLOOR - NUMERIC_ATOL
                    ),
                }
            )
            reject_checks.append(
                {
                    "budget_position": position,
                    "scope": scope,
                    "reject_rate": risk["reject_rate"],
                    "acceptable": float(risk["reject_rate"]) < REJECT_RATE_LIMIT,
                }
            )
        for scope, relation in relation_scopes:
            checks = {
                metric: bool(relation[metric]["non_adverse"])
                for metric in (
                    "joint_violation_rate",
                    "mean_relative_excess",
                    "max_relative_excess",
                )
            }
            non_adverse_checks.append(
                {
                    "budget_position": position,
                    "scope": scope,
                    **checks,
                    "all_risk_metrics_non_adverse": all(checks.values()),
                }
            )

    def stable_benefit(metric: str, position: int) -> bool:
        relation = relations[position]
        fold_reductions = [
            relation["per_fold"][fold_id][metric]["relative_reduction"]
            for fold_id in ordered_ids
        ]
        macro_reduction = relation["macro"][metric]["relative_reduction"]
        micro_reduction = relation["micro"][metric]["relative_reduction"]
        return bool(
            all(value is not None and value >= -NUMERIC_ATOL for value in fold_reductions)
            and any(
                value is not None and value >= RELATIVE_BENEFIT_THRESHOLD
                for value in fold_reductions
            )
            and macro_reduction is not None
            and macro_reduction >= RELATIVE_BENEFIT_THRESHOLD
            and micro_reduction is not None
            and micro_reduction >= RELATIVE_BENEFIT_THRESHOLD
        )

    violation_by_budget = [
        stable_benefit("joint_violation_rate", position)
        for position in range(len(budgets))
    ]
    excess_by_budget = [
        stable_benefit("mean_relative_excess", position)
        for position in range(len(budgets))
    ]
    violation_benefit = any(violation_by_budget)
    excess_benefit = any(excess_by_budget)
    strict_position = len(budgets) - 1
    strict_rows = [row for row in pd_checks if row["budget_position"] == strict_position]
    strict_folds = [row for row in strict_rows if str(row["scope"]).startswith("fold:")]
    strict_macro = next(row for row in strict_rows if row["scope"] == "macro")
    strict_micro = next(row for row in strict_rows if row["scope"] == "micro")
    strict_pd_gain = bool(
        strict_macro["pd_delta"] >= STRICT_PD_GAIN_THRESHOLD
        and strict_micro["pd_delta"] >= STRICT_PD_GAIN_THRESHOLD
        and all(row["non_degraded"] for row in strict_folds)
        and any(row["pd_delta"] >= STRICT_PD_GAIN_THRESHOLD for row in strict_folds)
    )
    success_count = sum((violation_benefit, excess_benefit, strict_pd_gain))
    c4 = bool((violation_benefit or excess_benefit) and success_count >= 2)

    criteria = {
        "c0_integrity_binding_and_two_phase_replay": True,
        "c1_complete_two_fold_paired_method_matrix": True,
        "c2_absolute_pixel_component_risk_control": all(
            row["joint_budget_satisfied"] for row in absolute_checks
        ),
        "c3_jvr_mre_max_excess_non_adverse": all(
            row["all_risk_metrics_non_adverse"] for row in non_adverse_checks
        ),
        "c4_stable_benefit_not_pd_only": c4,
        "c5_literal_pd_not_lower_than_rc_direct": all(
            row["non_degraded"] for row in pd_checks
        ),
        "c6_reject_rate_below_20_percent": all(
            row["acceptable"] for row in reject_checks
        ),
        "c7_risk_curve_mvr_zero": bool(
            all(fold_mvr[fold_id]["risk_curve"]["rate"] == 0.0 for fold_id in ordered_ids)
            and macro_mvr["risk_curve"] == 0.0
            and micro_mvr["risk_curve"]["rate"] == 0.0
        ),
    }
    return {
        "decision": "GO" if all(criteria.values()) else "HOLD",
        "criteria": criteria,
        "failure_reasons": [name for name, passed in criteria.items() if not passed],
        "relations_by_budget": {
            str(position): relations[position] for position in range(len(budgets))
        },
        "absolute_risk_budget_checks": absolute_checks,
        "risk_non_adverse_checks": non_adverse_checks,
        "literal_pd_vs_rc_direct_checks": pd_checks,
        "reject_rate_checks": reject_checks,
        "benefit_evidence": {
            "violation_stable_by_budget": violation_by_budget,
            "mean_relative_excess_stable_by_budget": excess_by_budget,
            "stable_joint_violation_benefit": violation_benefit,
            "stable_mean_relative_excess_benefit": excess_benefit,
            "literal_strict_budget_pd_gain": strict_pd_gain,
            "authoritative_success_count": success_count,
            "at_least_two_success_items_required": True,
            "at_least_one_stable_risk_benefit_required": True,
            "pd_only_can_never_pass": True,
        },
    }


def aggregate_anchor_warp_gate_c(
    *, folds: Sequence[tuple[str, str | Path]], output: str | Path
) -> Path:
    """Validate and aggregate exactly two AnchorWarp source-only folds."""

    # This must remain the first filesystem evidence capture in the aggregate
    # call.  The final verification is performed immediately before the atomic
    # publication step below.
    runtime_code_tree = _snapshot_runtime_code_tree()
    if len(folds) != REQUIRED_FOLD_COUNT:
        raise ValueError("AnchorWarp Gate C requires exactly two folds")
    fold_ids = [str(fold_id) for fold_id, _path in folds]
    if any(not fold_id for fold_id in fold_ids) or len(set(fold_ids)) != len(fold_ids):
        raise ValueError("AnchorWarp Gate C fold IDs must be non-empty and unique")
    file_paths = {
        str(fold_id): Path(path).expanduser().resolve() for fold_id, path in folds
    }
    if len(set(file_paths.values())) != REQUIRED_FOLD_COUNT:
        raise ValueError("AnchorWarp Gate C cannot count one comparison file twice")
    tracked: dict[Path, str] = {}
    snapshots: dict[str, tuple[dict[str, Any], str]] = {}
    for fold_id in sorted(file_paths):
        snapshots[fold_id] = _load_json_snapshot(
            file_paths[fold_id], role=f"{fold_id} AnchorWarp comparison"
        )
    parsed: dict[str, dict[str, Any]] = {}
    for fold_id in sorted(file_paths):
        payload, digest = snapshots[fold_id]
        parsed[fold_id] = _parse_fold(
            fold_id=fold_id,
            file_path=file_paths[fold_id],
            payload=payload,
            file_sha256=digest,
            tracked=tracked,
        )

    ordered_ids = sorted(parsed)
    if {parsed[fold_id]["validation_key"] for fold_id in ordered_ids} != FORMAL_SOURCE_DOMAIN_KEYS:
        raise ValueError(
            "AnchorWarp Gate C held pseudo-targets must be exactly IRSTD-1K and NUDT-SIRST"
        )
    if {parsed[fold_id]["train_key"] for fold_id in ordered_ids} != FORMAL_SOURCE_DOMAIN_KEYS:
        raise ValueError("AnchorWarp Gate C train-domain coverage is incomplete")
    if len({parsed[fold_id]["episode_archive_sha256"] for fold_id in ordered_ids}) != REQUIRED_FOLD_COUNT:
        raise ValueError("AnchorWarp Gate C cannot count one validation archive twice")
    reference = parsed[ordered_ids[0]]
    for fold_id in ordered_ids[1:]:
        if parsed[fold_id]["budgets"] != reference["budgets"]:
            raise ValueError("AnchorWarp Gate C fold budgets differ")
        if parsed[fold_id]["shared_contract"] != reference["shared_contract"]:
            raise ValueError("AnchorWarp Gate C shared threshold/feature contract differs")
        if parsed[fold_id]["seed"] != reference["seed"]:
            raise ValueError("AnchorWarp Gate C fold seeds differ")

    # Runtime replay is part of C0, not a caller assertion.  It re-executes the
    # implementation that freezes the digest before loading future-E arrays.
    replay = {
        fold_id: _replay_fold_exact(parsed[fold_id]) for fold_id in ordered_ids
    }
    evaluator_relative_path = (
        "risk_curve/evaluate_anchor_warp_source_pseudo_target_v4.py"
    )
    snapshotted_evaluator_sha = runtime_code_tree["files"][
        evaluator_relative_path
    ]["sha256"]
    if any(
        replay[fold_id]["evaluator_sha256"] != snapshotted_evaluator_sha
        for fold_id in ordered_ids
    ):
        raise ValueError(
            "Two-phase replay evaluator source differs from the entry code-tree snapshot"
        )
    fold_aggregates = {
        fold_id: parsed[fold_id]["aggregates"] for fold_id in ordered_ids
    }
    budgets = reference["budgets"]
    macro: dict[int, dict[str, dict[str, Any]]] = {}
    micro: dict[int, dict[str, dict[str, Any]]] = {}
    for position in range(len(budgets)):
        macro[position] = {
            method: _macro_aggregate(
                [fold_aggregates[fold_id][position][method] for fold_id in ordered_ids]
            )
            for method in METHODS
        }
        micro[position] = {
            method: _aggregate_actions(
                [
                    action
                    for fold_id in ordered_ids
                    for action in parsed[fold_id]["actions"][position][method]
                ]
            )
            for method in METHODS
        }
    fold_mvr = {
        fold_id: parsed[fold_id]["monotonicity"] for fold_id in ordered_ids
    }
    macro_mvr = {
        method: sum(fold_mvr[fold_id][method]["rate"] for fold_id in ordered_ids)
        / float(len(ordered_ids))
        for method in METHODS
    }
    micro_mvr: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        episodes = sum(
            fold_mvr[fold_id][method]["num_episodes"] for fold_id in ordered_ids
        )
        violations = sum(
            fold_mvr[fold_id][method]["violating_episode_count"]
            for fold_id in ordered_ids
        )
        micro_mvr[method] = {
            "num_episodes": episodes,
            "violating_episode_count": violations,
            "rate": violations / float(episodes),
        }
    decision = _build_decision(
        ordered_ids=ordered_ids,
        budgets=budgets,
        fold_aggregates=fold_aggregates,
        macro=macro,
        micro=micro,
        fold_mvr=fold_mvr,
        macro_mvr=macro_mvr,
        micro_mvr=micro_mvr,
    )

    # Rehash every comparison, NPZ, bound package, parent train-only checkpoint,
    # direct checkpoint, and train archive immediately before publication.
    for path, expected in tracked.items():
        if _sha256_file(path) != expected:
            raise ValueError(f"Referenced artifact changed during aggregation: {path}")

    per_fold: dict[str, Any] = {}
    for fold_id in ordered_ids:
        fold = parsed[fold_id]
        per_fold[fold_id] = {
            "validation_pseudo_target": fold["validation_target"],
            "train_pseudo_target": fold["train_target"],
            "num_episodes": fold["num_episodes"],
            "comparison_file": fold["file"],
            "comparison_sha256": fold["file_sha256"],
            "artifacts": fold["artifacts"],
            "checkpoint_binding": fold["checkpoint_binding"],
            "two_phase_action_evidence": fold["action_order_proof"],
            "runtime_two_phase_replay": replay[fold_id],
            "metrics_by_budget": {
                str(position): fold["aggregates"][position]
                for position in range(len(budgets))
            },
            "monotonicity": fold["monotonicity"],
        }
    aggregator_relative_path = "risk_curve/aggregate_anchor_warp_gate_c_v4.py"
    aggregator_record = runtime_code_tree["files"][aggregator_relative_path]
    aggregator_path = Path(aggregator_record["path"])
    aggregator_sha256 = aggregator_record["sha256"]
    published_runtime_code_tree = {
        **runtime_code_tree,
        "verified_unchanged_immediately_before_atomic_publish": True,
    }
    payload = {
        "schema_version": AGGREGATE_ANCHOR_WARP_GATE_C_SCHEMA_VERSION,
        "policy_version": AGGREGATE_ANCHOR_WARP_GATE_C_POLICY_VERSION,
        "decision": decision["decision"],
        "scope": "source_only_anchor_warp_vs_train_only_rc_direct_cross_fold_gate_c",
        "not_an_outer_target_claim": True,
        "outer_target_labels_used": False,
        "required_fold_count": REQUIRED_FOLD_COUNT,
        "required_methods": list(METHODS),
        "required_source_pseudo_targets": ["IRSTD-1K", "NUDT-SIRST"],
        "required_excluded_outer_target": "NUAA-SIRST",
        "seed": reference["seed"],
        "representation": LOGIT_REPRESENTATION,
        "shared_contract": reference["shared_contract"],
        "required_budgets": [
            {
                "budget_position": position,
                "pixel_budget": pair[0],
                "component_budget": pair[1],
            }
            for position, pair in enumerate(budgets)
        ],
        "folds": per_fold,
        "macro_by_budget": {
            str(position): macro[position] for position in range(len(budgets))
        },
        "micro_by_budget": {
            str(position): micro[position] for position in range(len(budgets))
        },
        "monotonicity": {
            "macro_rate_by_method": macro_mvr,
            "micro_by_method": micro_mvr,
        },
        "risk_curve_vs_rc_direct": {
            "by_budget": decision["relations_by_budget"]
        },
        "absolute_risk_budget_checks": decision["absolute_risk_budget_checks"],
        "risk_non_adverse_checks": decision["risk_non_adverse_checks"],
        "benefit_evidence": decision["benefit_evidence"],
        "literal_pd_vs_rc_direct_checks": decision[
            "literal_pd_vs_rc_direct_checks"
        ],
        "reject_rate_checks": decision["reject_rate_checks"],
        "criteria": decision["criteria"],
        "failure_reasons": decision["failure_reasons"],
        "runtime_code_tree": published_runtime_code_tree,
        "aggregation_contract": {
            "macro": "equal_weight_mean_over_two_held_source_folds",
            "micro": "recomputed_from_concatenated_per_episode_sufficient_counts",
            "absolute_risk": "pixel_and_component_feasible_at_every_fold_macro_micro_cell",
            "risk_non_adverse": "JVR_MRE_and_max_excess_not_worse_at_every_fold_macro_micro_cell",
            "stable_benefit": (
                "at_least_two_of_stable_JVR_stable_MRE_strict_Pd_gain_with_"
                "at_least_one_stable_risk_benefit"
            ),
            "literal_pd_non_degradation": "risk_curve_pd_minus_rc_direct_pd_must_be_at_least_zero",
            "relative_benefit_threshold": RELATIVE_BENEFIT_THRESHOLD,
            "strict_pd_gain_threshold": STRICT_PD_GAIN_THRESHOLD,
            "pd_non_degradation_floor": PD_NON_DEGRADATION_FLOOR,
            "reject_rate_limit_strict": REJECT_RATE_LIMIT,
            "risk_curve_mvr_required": 0.0,
            "numeric_atol": NUMERIC_ATOL,
            "criteria_not_lowered": True,
        },
        "provenance_validation": {
            "aggregator_path": str(aggregator_path),
            "aggregator_sha256": aggregator_sha256,
            "comparison_json_immutable_snapshots": True,
            "all_referenced_artifact_hashes_verified": True,
            "anchor_train_only_parent_hash_and_semantics_verified": True,
            "anchor_validation_binding_verified": True,
            "rc_direct_train_only_freeze_order_verified": True,
            "two_phase_evaluator_byte_exact_replay_verified": True,
            "future_e_used_only_after_action_digest": True,
            "future_e_reselection_performed": False,
        },
    }
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    try:
        os.chmod(temporary, 0o644)
        _verify_runtime_code_tree(runtime_code_tree)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def _parse_fold_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--fold must use ID=COMPARISON_JSON")
    fold_id, path = value.split("=", 1)
    if not fold_id or not path:
        raise argparse.ArgumentTypeError("--fold must use non-empty ID=COMPARISON_JSON")
    return fold_id, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold",
        action="append",
        type=_parse_fold_arg,
        required=True,
        help="Exactly two ID=COMPARISON_JSON entries",
    )
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(aggregate_anchor_warp_gate_c(folds=args.fold, output=args.output))


if __name__ == "__main__":
    main()


__all__ = [
    "AGGREGATE_ANCHOR_WARP_GATE_C_POLICY_VERSION",
    "AGGREGATE_ANCHOR_WARP_GATE_C_SCHEMA_VERSION",
    "PD_NON_DEGRADATION_FLOOR",
    "RUNTIME_CODE_TREE_FILES",
    "RUNTIME_CODE_TREE_SCHEMA_VERSION",
    "aggregate_anchor_warp_gate_c",
]
