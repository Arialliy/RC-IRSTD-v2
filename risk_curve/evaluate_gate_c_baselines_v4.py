"""Evaluate the frozen source baselines required by the v4 Gate C protocol.

``Source-static`` and ``Source-worst`` select one finite raw-logit grid index
from the *training* source pseudo-target episodes.  The selected action is then
frozen and audited on the held-out validation pseudo-target with the stored
future-E sufficient counts.  Validation labels can therefore never influence
selection.

``Count-all`` selects an episode-specific action from complete per-threshold
prediction counts over label-free adaptation window A.  It treats every
retained A prediction as a false alarm, then audits the frozen action with the
stored future-E sufficient counts.  The historical probability-grid baseline
and any reconstruction from compressed statistics are forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .curve_dataset import (
    COUNT_ALL_ADAPTATION_ARCHIVE_FIELDS,
    COUNT_ALL_ADAPTATION_SCHEMA_VERSION,
    validate_count_all_adaptation_contract,
)
from .direct_calibrator import (
    ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL,
    DirectTrainingPair,
    load_direct_training_pair,
    validate_joint_budget_pairs,
)
from .representation import (
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_REPRESENTATION,
    logit_threshold_grid_sha256,
    validate_logit_threshold_grid,
)


GATE_C_BASELINES_SCHEMA_VERSION = "rc-v4-gate-c-source-baselines-v1"


def _scalar_text(value: Any, field: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{field} must be scalar")
    return str(array.item())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _domain_key(value: Any) -> str:
    key = "".join(
        character for character in str(value).casefold() if character.isalnum()
    ).removesuffix("sirst")
    if not key:
        raise ValueError("Pseudo-target domain names must be non-empty")
    return key


def _load_provenance(
    archive: Mapping[str, np.ndarray], *, split: str
) -> dict[str, Any]:
    try:
        payload = json.loads(
            _scalar_text(archive["provenance_json"], f"{split}.provenance_json")
        )
    except KeyError as error:
        raise ValueError(f"{split} archive lacks provenance_json") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{split}.provenance_json is invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{split}.provenance_json must decode to an object")
    return payload


def _integer_array(
    archive: Mapping[str, np.ndarray],
    field: str,
    *,
    split: str,
    shape: tuple[int, ...],
    minimum: int = 0,
) -> np.ndarray:
    if field not in archive:
        raise ValueError(f"{split} archive lacks stored count field {field}")
    raw = np.asarray(archive[field])
    if raw.shape != shape:
        raise ValueError(f"{split}.{field} must have shape {shape}")
    if raw.dtype.kind not in "iu":
        if not np.isfinite(raw).all() or not np.all(np.equal(raw, np.floor(raw))):
            raise ValueError(f"{split}.{field} must contain finite integer counts")
    values = raw.astype(np.int64)
    if np.any(values < minimum):
        raise ValueError(f"{split}.{field} contains values below {minimum}")
    return values


def _validate_count_archive(
    archive: Mapping[str, np.ndarray], *, split: str
) -> dict[str, np.ndarray]:
    """Validate stored E counts and bind them to all persisted risk curves."""

    thresholds = validate_logit_threshold_grid(
        np.asarray(archive["thresholds"], dtype=np.float32)
    )
    rows = int(np.asarray(archive["statistics"]).shape[0])
    grid_size = int(thresholds.size)
    pixel_fp = _integer_array(
        archive,
        "pixel_fp_counts",
        split=split,
        shape=(rows, grid_size),
    )
    component_fp = _integer_array(
        archive,
        "component_fp_counts",
        split=split,
        shape=(rows, grid_size),
    )
    tp = _integer_array(
        archive,
        "tp_object_counts",
        split=split,
        shape=(rows, grid_size),
    )
    gt = _integer_array(
        archive, "gt_object_counts", split=split, shape=(rows,)
    )
    total_pixels = _integer_array(
        archive,
        "total_pixels",
        split=split,
        shape=(rows,),
        minimum=1,
    )
    if np.any(pixel_fp > total_pixels[:, None]):
        raise ValueError(f"{split}.pixel_fp_counts exceed their pixel exposure")
    if np.any(tp > gt[:, None]):
        raise ValueError(f"{split}.tp_object_counts exceed gt_object_counts")

    expected_pixel_log = np.log10(
        pixel_fp.astype(np.float64) / total_pixels[:, None] + 1e-12
    )
    expected_component_raw = np.log10(
        component_fp.astype(np.float64)
        / (total_pixels[:, None] / 1_000_000.0)
        + 1e-6
    )
    expected_component_upper = np.maximum.accumulate(
        expected_component_raw[:, ::-1], axis=1
    )[:, ::-1]
    expected_pd = tp.astype(np.float64) / np.maximum(gt[:, None], 1)
    comparisons = (
        ("pixel_log_risk", expected_pixel_log),
        ("component_log_risk_raw", expected_component_raw),
        ("component_log_risk_upper", expected_component_upper),
        ("pd_curve", expected_pd),
    )
    for field, expected in comparisons:
        if field not in archive:
            raise ValueError(f"{split} archive lacks count-binding field {field}")
        observed = np.asarray(archive[field], dtype=np.float64)
        if observed.shape != expected.shape or not np.allclose(
            observed, expected, rtol=2e-6, atol=2e-6
        ):
            raise ValueError(f"{split}.{field} disagrees with stored sufficient counts")

    targets = np.asarray(archive.get("pseudo_targets"))
    if targets.shape != (rows,):
        raise ValueError(f"{split}.pseudo_targets must have shape ({rows},)")
    target_names = np.asarray([str(item) for item in targets.tolist()], dtype=str)
    if any(not item.strip() for item in target_names.tolist()):
        raise ValueError(f"{split}.pseudo_targets contains an empty name")
    return {
        "thresholds": thresholds,
        "pixel_fp_counts": pixel_fp,
        "component_fp_counts": component_fp,
        "tp_object_counts": tp,
        "gt_object_counts": gt,
        "total_pixels": total_pixels,
        "pseudo_targets": target_names,
    }


def _validate_pair_scope(
    pair: DirectTrainingPair,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Add Gate-C split and raw-count checks to the shared v4 pair validator."""

    train = _validate_count_archive(pair.train_archive, split="train")
    validation = _validate_count_archive(
        pair.validation_archive, split="validation"
    )
    if not np.array_equal(train["thresholds"], validation["thresholds"]):
        raise ValueError("Train/validation raw-logit grids differ")
    provenance_train = _load_provenance(pair.train_archive, split="train")
    provenance_validation = _load_provenance(
        pair.validation_archive, split="validation"
    )
    for field in (
        "protocol",
        "representation",
        "pseudo_targets",
        "validation_domain",
        "expected_split_role",
        "pseudo_target_split",
        "adaptation_window",
        "evaluation_window",
        "stride",
        "connectivity",
        "min_component_area",
        "threshold_grid_sha256",
        "threshold_grid_manifest_sha256",
        "threshold_grid_detector_protocol",
        "threshold_grid_detector_checkpoint_sha256s",
        "threshold_grid_outer_detector_checkpoint_sha256",
        "threshold_grid_episode_detector_checkpoint_sha256s",
        "feature_schema_sha256",
        "count_all_adaptation_schema_version",
        "count_all_adaptation_sample_role",
        "count_all_adaptation_masks_read",
        "count_all_adaptation_prediction_rule",
        "count_all_adaptation_pixel_count_semantics",
        "count_all_adaptation_component_count_semantics",
        "count_all_adaptation_component_envelope",
    ):
        if provenance_train.get(field) != provenance_validation.get(field):
            raise ValueError(
                f"Train/validation semantic provenance differs in {field}"
            )
    provenance = provenance_train
    validation_domain = str(provenance.get("validation_domain", "")).strip()
    if not validation_domain:
        raise ValueError("Gate C baselines require a held-out validation_domain")
    validation_key = _domain_key(validation_domain)
    train_keys = {_domain_key(item) for item in train["pseudo_targets"]}
    validation_keys = {
        _domain_key(item) for item in validation["pseudo_targets"]
    }
    if validation_keys != {validation_key}:
        raise ValueError(
            "Validation rows must contain exactly the declared held-out pseudo-target"
        )
    if validation_key in train_keys:
        raise ValueError("Training rows include the held-out validation pseudo-target")
    if train_keys.intersection(validation_keys):
        raise ValueError("Train/validation pseudo-target domains overlap")
    if provenance.get("threshold_grid_detector_protocol") != (
        ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL
    ):
        raise ValueError("Gate C baselines require the all-source detector-fold grid")
    return train, validation, provenance


def _aggregate_count_rows(
    counts: Mapping[str, np.ndarray], rows: np.ndarray, index: int
) -> dict[str, Any]:
    if rows.ndim != 1 or rows.size == 0:
        raise ValueError("Cannot aggregate an empty row selection")
    pixel_fp = int(counts["pixel_fp_counts"][rows, index].sum())
    component_fp = int(counts["component_fp_counts"][rows, index].sum())
    tp = int(counts["tp_object_counts"][rows, index].sum())
    gt = int(counts["gt_object_counts"][rows].sum())
    total_pixels = int(counts["total_pixels"][rows].sum())
    return {
        "threshold_index": int(index),
        "selected_logit_threshold": float(counts["thresholds"][index]),
        "pixel_fp_count": pixel_fp,
        "component_fp_count": component_fp,
        "tp_object_count": tp,
        "gt_object_count": gt,
        "total_pixels": total_pixels,
        "pd": tp / float(max(gt, 1)),
        "pixel_risk": pixel_fp / float(total_pixels),
        "component_risk": component_fp / (total_pixels / 1_000_000.0),
    }


def _is_feasible(
    row: Mapping[str, Any], *, pixel_budget: float, component_budget: float
) -> bool:
    return bool(
        float(row["pixel_risk"]) <= pixel_budget
        and float(row["component_risk"]) <= component_budget
    )


def _reject_selection(*, strategy: str, candidates: int) -> dict[str, Any]:
    return {
        "found": False,
        "reject": True,
        "reason": "no_finite_raw_logit_grid_index_satisfies_source_budgets",
        "strategy": strategy,
        "threshold_index": None,
        "selected_logit_threshold": "+inf",
        "finite_feasible_candidates": int(candidates),
        "external_reject_action": True,
    }


def select_source_baseline_actions(
    counts: Mapping[str, np.ndarray],
    *,
    pixel_budget: float,
    component_budget: float,
) -> dict[str, dict[str, Any]]:
    """Select Source-static and Source-worst actions from training counts only."""

    if not np.isfinite(pixel_budget) or pixel_budget <= 0.0:
        raise ValueError("pixel_budget must be finite and positive")
    if not np.isfinite(component_budget) or component_budget <= 0.0:
        raise ValueError("component_budget must be finite and positive")
    thresholds = validate_logit_threshold_grid(np.asarray(counts["thresholds"]))
    rows = int(np.asarray(counts["pixel_fp_counts"]).shape[0])
    all_rows = np.arange(rows, dtype=np.int64)
    raw_targets = np.asarray(counts["pseudo_targets"])
    if raw_targets.shape != (rows,):
        raise ValueError("pseudo_targets must contain one name per count row")
    display_names: dict[str, str] = {}
    domain_rows: dict[str, list[int]] = {}
    for row, raw_name in enumerate(raw_targets.tolist()):
        name = str(raw_name)
        key = _domain_key(name)
        display_names.setdefault(key, name)
        domain_rows.setdefault(key, []).append(row)

    static_candidates: list[dict[str, Any]] = []
    worst_candidates: list[dict[str, Any]] = []
    for index in range(thresholds.size):
        pooled = _aggregate_count_rows(counts, all_rows, index)
        if _is_feasible(
            pooled,
            pixel_budget=float(pixel_budget),
            component_budget=float(component_budget),
        ):
            static_candidates.append(pooled)
        domain_evidence = {
            display_names[key]: _aggregate_count_rows(
                counts, np.asarray(indices, dtype=np.int64), index
            )
            for key, indices in sorted(domain_rows.items())
        }
        if all(
            _is_feasible(
                evidence,
                pixel_budget=float(pixel_budget),
                component_budget=float(component_budget),
            )
            for evidence in domain_evidence.values()
        ):
            worst_candidates.append(
                {
                    "pooled": pooled,
                    "per_domain": domain_evidence,
                    "worst_domain_pd": min(
                        float(item["pd"]) for item in domain_evidence.values()
                    ),
                }
            )

    static_strategy = "aggregate_train_counts_then_maximize_pooled_pd"
    if static_candidates:
        selected_static = min(
            static_candidates,
            key=lambda item: (-float(item["pd"]), int(item["threshold_index"])),
        )
        static = {
            "found": True,
            "reject": False,
            "reason": "source_pooled_budget_feasible_maximum_pd",
            "strategy": static_strategy,
            "tie_break": "lowest_raw_logit_grid_index",
            "finite_feasible_candidates": len(static_candidates),
            "external_reject_action": False,
            "train_evidence_at_action": selected_static,
            "threshold_index": int(selected_static["threshold_index"]),
            "selected_logit_threshold": float(
                selected_static["selected_logit_threshold"]
            ),
        }
    else:
        static = _reject_selection(strategy=static_strategy, candidates=0)

    worst_strategy = (
        "require_every_train_pseudo_target_budget_then_maximize_min_domain_pd"
    )
    if worst_candidates:
        selected_worst = min(
            worst_candidates,
            key=lambda item: (
                -float(item["worst_domain_pd"]),
                -float(item["pooled"]["pd"]),
                int(item["pooled"]["threshold_index"]),
            ),
        )
        worst = {
            "found": True,
            "reject": False,
            "reason": "all_source_domains_budget_feasible_maximin_pd",
            "strategy": worst_strategy,
            "tie_break": ["maximize_pooled_pd", "lowest_raw_logit_grid_index"],
            "finite_feasible_candidates": len(worst_candidates),
            "external_reject_action": False,
            "num_train_pseudo_target_domains": len(domain_rows),
            "degenerate_k1": len(domain_rows) == 1,
            "worst_domain_pd": float(selected_worst["worst_domain_pd"]),
            "train_pooled_evidence_at_action": selected_worst["pooled"],
            "train_per_domain_evidence_at_action": selected_worst["per_domain"],
            "threshold_index": int(
                selected_worst["pooled"]["threshold_index"]
            ),
            "selected_logit_threshold": float(
                selected_worst["pooled"]["selected_logit_threshold"]
            ),
        }
    else:
        worst = _reject_selection(strategy=worst_strategy, candidates=0)
        worst.update(
            {
                "num_train_pseudo_target_domains": len(domain_rows),
                "degenerate_k1": len(domain_rows) == 1,
                "worst_domain_pd": None,
                "train_pooled_evidence_at_action": None,
                "train_per_domain_evidence_at_action": None,
            }
        )
    if worst["degenerate_k1"]:
        worst["degeneracy_note"] = (
            "Only one pseudo-target domain is present in train.npz; Source-worst "
            "is algebraically the K=1 special case, not a multi-domain stress test."
        )
    return {"source_static": static, "source_worst": worst}


def _episode_ids(archive: Mapping[str, np.ndarray], row: int) -> list[str]:
    try:
        decoded = json.loads(str(np.asarray(archive["evaluation_ids"])[row]))
    except (KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"validation.evaluation_ids[{row}] is invalid") from error
    if not isinstance(decoded, list) or not decoded or any(
        not isinstance(item, str) or not item for item in decoded
    ):
        raise ValueError(
            f"validation.evaluation_ids[{row}] must be a non-empty string list"
        )
    return decoded


def _validation_action(
    counts: Mapping[str, np.ndarray],
    *,
    row: int,
    selection: Mapping[str, Any],
    pixel_budget: float,
    component_budget: float,
) -> dict[str, Any]:
    index = selection["threshold_index"]
    reject = index is None
    total_pixels = int(counts["total_pixels"][row])
    gt = int(counts["gt_object_counts"][row])
    if reject:
        pixel_fp = component_fp = tp = 0
    else:
        index = int(index)
        pixel_fp = int(counts["pixel_fp_counts"][row, index])
        component_fp = int(counts["component_fp_counts"][row, index])
        tp = int(counts["tp_object_counts"][row, index])
    pixel_risk = pixel_fp / float(total_pixels)
    component_risk = component_fp / (total_pixels / 1_000_000.0)
    pixel_excess = max(pixel_risk / pixel_budget - 1.0, 0.0)
    component_excess = max(component_risk / component_budget - 1.0, 0.0)
    return {
        "threshold_index": None if reject else int(index),
        "selected_logit_threshold": (
            "+inf" if reject else float(counts["thresholds"][int(index)])
        ),
        "reject": reject,
        "pixel_fp_count": pixel_fp,
        "component_fp_count": component_fp,
        "tp_object_count": tp,
        "gt_object_count": gt,
        "total_pixels": total_pixels,
        "pd": tp / float(max(gt, 1)),
        "pixel_risk": pixel_risk,
        "component_risk": component_risk,
        "pixel_budget_violated": bool(pixel_risk > pixel_budget),
        "component_budget_violated": bool(component_risk > component_budget),
        "joint_budget_violated": bool(
            pixel_risk > pixel_budget or component_risk > component_budget
        ),
        "pixel_relative_excess": pixel_excess,
        "component_relative_excess": component_excess,
        "joint_relative_excess": max(pixel_excess, component_excess),
    }


def _aggregate_actions(actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not actions:
        raise ValueError("Cannot aggregate an empty validation action list")
    total_pixels = sum(int(item["total_pixels"]) for item in actions)
    gt = sum(int(item["gt_object_count"]) for item in actions)
    tp = sum(int(item["tp_object_count"]) for item in actions)
    pixel_fp = sum(int(item["pixel_fp_count"]) for item in actions)
    component_fp = sum(int(item["component_fp_count"]) for item in actions)
    finite_indices = sorted(
        {
            int(item["threshold_index"])
            for item in actions
            if item["threshold_index"] is not None
        }
    )
    return {
        "num_episodes": len(actions),
        "pd": tp / float(max(gt, 1)),
        "pixel_risk": pixel_fp / float(total_pixels),
        "component_risk": component_fp / (total_pixels / 1_000_000.0),
        "joint_violation_rate": float(
            np.mean([item["joint_budget_violated"] for item in actions])
        ),
        "mean_relative_excess": float(
            np.mean([item["joint_relative_excess"] for item in actions])
        ),
        "max_relative_excess": float(
            np.max([item["joint_relative_excess"] for item in actions])
        ),
        "reject_rate": float(np.mean([item["reject"] for item in actions])),
        "unique_finite_indices": finite_indices,
        "num_unique_finite_indices": len(finite_indices),
        "aggregate_counts": {
            "pixel_fp_count": pixel_fp,
            "component_fp_count": component_fp,
            "tp_object_count": tp,
            "gt_object_count": gt,
            "total_pixels": total_pixels,
        },
    }


def _evaluate_frozen_selection(
    archive: Mapping[str, np.ndarray],
    counts: Mapping[str, np.ndarray],
    selection: Mapping[str, Any],
    *,
    pixel_budget: float,
    component_budget: float,
) -> dict[str, Any]:
    per_episode: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row, target in enumerate(counts["pseudo_targets"].tolist()):
        action = _validation_action(
            counts,
            row=row,
            selection=selection,
            pixel_budget=pixel_budget,
            component_budget=component_budget,
        )
        record = {
            "episode_index": row,
            "pseudo_target": str(target),
            "evaluation_ids": _episode_ids(archive, row),
            "action": action,
        }
        per_episode.append(record)
        grouped.setdefault(str(target), []).append(action)
    return {
        "selection_function_received_training_counts_only": True,
        "validation_counts_used_for_selection": False,
        "validation_labels_used_for_selection": False,
        "aggregate": _aggregate_actions([item["action"] for item in per_episode]),
        "per_pseudo_target": {
            name: _aggregate_actions(actions) for name, actions in sorted(grouped.items())
        },
        "per_episode": per_episode,
    }


def _adaptation_ids(archive: Mapping[str, np.ndarray], row: int) -> list[str]:
    try:
        decoded = json.loads(str(np.asarray(archive["adaptation_ids"])[row]))
    except (KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"validation.adaptation_ids[{row}] is invalid") from error
    if not isinstance(decoded, list) or not decoded or any(
        not isinstance(item, str) or not item for item in decoded
    ):
        raise ValueError(
            f"validation.adaptation_ids[{row}] must be a non-empty string list"
        )
    return decoded


def _select_count_all_action(
    archive: Mapping[str, np.ndarray],
    *,
    row: int,
    pixel_budget: float,
    component_budget: float,
) -> dict[str, Any]:
    """Select only from one episode's label-free A-window count curves."""

    thresholds = validate_logit_threshold_grid(
        np.asarray(archive["thresholds"], dtype=np.float32)
    )
    pixel_counts = np.asarray(
        archive["adaptation_predicted_pixel_counts"], dtype=np.int64
    )[row]
    component_raw = np.asarray(
        archive["adaptation_predicted_component_counts_raw"], dtype=np.int64
    )[row]
    component_upper = np.asarray(
        archive["adaptation_predicted_component_counts_upper"], dtype=np.int64
    )[row]
    total_pixels = int(np.asarray(archive["adaptation_total_pixels"])[row])
    pixel_upper_risk = pixel_counts.astype(np.float64) / float(total_pixels)
    component_upper_risk = component_upper.astype(np.float64) / (
        total_pixels / 1_000_000.0
    )
    feasible = np.flatnonzero(
        (pixel_upper_risk <= float(pixel_budget))
        & (component_upper_risk <= float(component_budget))
    )
    if feasible.size == 0:
        return {
            "found": False,
            "reject": True,
            "reason": "no_finite_grid_index_satisfies_count_all_A_upper_bounds",
            "selection_rule": "earliest_jointly_feasible_raw_logit_grid_index",
            "threshold_index": None,
            "selected_logit_threshold": "+inf",
            "external_reject_action": True,
            "adaptation_total_pixels": total_pixels,
            "adaptation_masks_read": False,
            "future_e_counts_used_for_selection": False,
        }
    index = int(feasible[0])
    return {
        "found": True,
        "reject": False,
        "reason": "first_count_all_A_upper_bound_feasible_grid_index",
        "selection_rule": "earliest_jointly_feasible_raw_logit_grid_index",
        "threshold_index": index,
        "selected_logit_threshold": float(thresholds[index]),
        "external_reject_action": False,
        "adaptation_total_pixels": total_pixels,
        "adaptation_predicted_pixel_count_at_action": int(pixel_counts[index]),
        "adaptation_predicted_component_count_raw_at_action": int(
            component_raw[index]
        ),
        "adaptation_predicted_component_count_upper_at_action": int(
            component_upper[index]
        ),
        "adaptation_pixel_upper_risk_at_action": float(pixel_upper_risk[index]),
        "adaptation_component_upper_risk_at_action": float(
            component_upper_risk[index]
        ),
        "adaptation_masks_read": False,
        "future_e_counts_used_for_selection": False,
    }


def _evaluate_count_all(
    archive: Mapping[str, np.ndarray],
    counts: Mapping[str, np.ndarray],
    *,
    pixel_budget: float,
    component_budget: float,
) -> dict[str, Any]:
    per_episode: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    selections: list[dict[str, Any]] = []
    for row, target in enumerate(counts["pseudo_targets"].tolist()):
        selection = _select_count_all_action(
            archive,
            row=row,
            pixel_budget=pixel_budget,
            component_budget=component_budget,
        )
        action = _validation_action(
            counts,
            row=row,
            selection=selection,
            pixel_budget=pixel_budget,
            component_budget=component_budget,
        )
        selections.append(selection)
        per_episode.append(
            {
                "episode_index": row,
                "pseudo_target": str(target),
                "adaptation_ids": _adaptation_ids(archive, row),
                "evaluation_ids": _episode_ids(archive, row),
                "selection": selection,
                "action": action,
            }
        )
        grouped.setdefault(str(target), []).append(action)
    finite_indices = sorted(
        {
            int(item["threshold_index"])
            for item in selections
            if item["threshold_index"] is not None
        }
    )
    return {
        "selection_source": "validation_episode_adaptation_window_A_counts_only",
        "adaptation_masks_read": False,
        "future_e_counts_used_for_selection": False,
        "validation_labels_used_for_selection": False,
        "selection_summary": {
            "num_episodes": len(selections),
            "reject_rate": float(np.mean([item["reject"] for item in selections])),
            "unique_finite_indices": finite_indices,
            "num_unique_finite_indices": len(finite_indices),
        },
        "aggregate": _aggregate_actions([item["action"] for item in per_episode]),
        "per_pseudo_target": {
            name: _aggregate_actions(actions) for name, actions in sorted(grouped.items())
        },
        "per_episode": per_episode,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_immutable_training_pair(
    train_path: Path,
    validation_path: Path,
) -> tuple[DirectTrainingPair, str, str]:
    """Parse and hash each NPZ from the same entry-time byte snapshot."""

    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("Gate C baseline train/validation archive is absent")
    train_raw = train_path.read_bytes()
    validation_raw = validation_path.read_bytes()
    train_sha256 = hashlib.sha256(train_raw).hexdigest()
    validation_sha256 = hashlib.sha256(validation_raw).hexdigest()
    with tempfile.TemporaryDirectory(prefix="rc-v4-baseline-snapshot-") as root:
        snapshot_root = Path(root)
        os.chmod(snapshot_root, 0o700)
        train_snapshot = snapshot_root / "train.npz"
        validation_snapshot = snapshot_root / "validation.npz"
        train_snapshot.write_bytes(train_raw)
        validation_snapshot.write_bytes(validation_raw)
        pair = load_direct_training_pair(train_snapshot, validation_snapshot)
    return pair, train_sha256, validation_sha256


def evaluate_gate_c_baselines(
    *,
    train_file: str | Path,
    validation_file: str | Path,
    output: str | Path,
    pixel_budgets: Sequence[float],
    component_budgets: Sequence[float],
) -> Path:
    """Run all three source-only Gate C baselines on one held-out fold."""

    train_path = Path(train_file).expanduser().resolve()
    validation_path = Path(validation_file).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    pair, train_archive_sha256, validation_archive_sha256 = (
        _load_immutable_training_pair(train_path, validation_path)
    )
    train, validation, provenance = _validate_pair_scope(pair)
    train_count_all_contract = validate_count_all_adaptation_contract(
        pair.train_archive, required=True
    )
    validation_count_all_contract = validate_count_all_adaptation_contract(
        pair.validation_archive, required=True
    )
    # Episode counts legitimately differ.  Every semantic field must not.
    for field in (
        "present",
        "verified",
        "schema_version",
        "representation",
        "threshold_grid_sha256",
        "num_thresholds",
        "connectivity",
        "min_component_area",
        "adaptation_masks_read",
        "prediction_rule",
    ):
        if train_count_all_contract[field] != validation_count_all_contract[field]:
            raise ValueError(
                f"Train/validation Count-all adaptation {field} mismatch"
            )
    pixels, components = validate_joint_budget_pairs(
        pixel_budgets, component_budgets
    )
    thresholds = train["thresholds"]
    recorded_grid_hash = _scalar_text(
        pair.train_archive["threshold_grid_sha256"], "threshold_grid_sha256"
    )
    if recorded_grid_hash != logit_threshold_grid_sha256(thresholds):
        raise ValueError("Raw-logit threshold-grid hash mismatch")

    budget_results: list[dict[str, Any]] = []
    for position, (pixel_budget, component_budget) in enumerate(
        zip(pixels.tolist(), components.tolist())
    ):
        selections = select_source_baseline_actions(
            train,
            pixel_budget=float(pixel_budget),
            component_budget=float(component_budget),
        )
        methods: dict[str, Any] = {}
        for method, selection in selections.items():
            methods[method] = {
                "selection": selection,
                "validation_evaluation": _evaluate_frozen_selection(
                    pair.validation_archive,
                    validation,
                    selection,
                    pixel_budget=float(pixel_budget),
                    component_budget=float(component_budget),
                ),
            }
        methods["count_all"] = {
            "selection": "episode_specific_from_label_free_A_count_curves",
            "validation_evaluation": _evaluate_count_all(
                pair.validation_archive,
                validation,
                pixel_budget=float(pixel_budget),
                component_budget=float(component_budget),
            ),
        }
        budget_results.append(
            {
                "budget_position": position,
                "pixel_budget": float(pixel_budget),
                "component_budget": float(component_budget),
                "methods": methods,
            }
        )

    episode_contract = pair.episode_contract
    payload = {
        "schema_version": GATE_C_BASELINES_SCHEMA_VERSION,
        "protocol": "source_only_pseudo_target_gate_c_frozen_baselines",
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_size": int(thresholds.size),
        "threshold_grid_sha256": recorded_grid_hash,
        "threshold_grid_manifest_sha256": _scalar_text(
            pair.train_archive["threshold_grid_manifest_sha256"],
            "threshold_grid_manifest_sha256",
        ),
        "feature_schema_sha256": _scalar_text(
            pair.train_archive["feature_schema_sha256"], "feature_schema_sha256"
        ),
        "threshold_grid_detector_protocol": _scalar_text(
            pair.train_archive["threshold_grid_detector_protocol"],
            "threshold_grid_detector_protocol",
        ),
        "threshold_grid_detector_checkpoint_sha256s": list(
            episode_contract["threshold_grid_detector_checkpoint_sha256s"]
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": episode_contract[
            "threshold_grid_outer_detector_checkpoint_sha256"
        ],
        "threshold_grid_episode_detector_checkpoint_sha256s": list(
            episode_contract[
                "threshold_grid_episode_detector_checkpoint_sha256s"
            ]
        ),
        "train_archive": str(train_path),
        "train_archive_sha256": train_archive_sha256,
        "validation_archive": str(validation_path),
        "validation_archive_sha256": validation_archive_sha256,
        "train_pseudo_targets": sorted(
            set(str(item) for item in train["pseudo_targets"].tolist())
        ),
        "validation_pseudo_targets": sorted(
            set(str(item) for item in validation["pseudo_targets"].tolist())
        ),
        "excluded_outer_target": provenance["threshold_grid_outer_target_key"],
        "labels_policy": {
            "train_source_future_e_labels_used_for_static_selection": True,
            "validation_source_future_e_labels_used_only_after_selection": True,
            "validation_labels_used_for_selection": False,
            "count_all_validation_A_masks_read_for_selection": False,
            "count_all_validation_A_raw_logits_used_for_selection": True,
            "outer_target_labels_used": False,
        },
        "external_reject_action": {
            "threshold": "+inf",
            "threshold_index": None,
            "inside_finite_model_grid": False,
        },
        "budgets": budget_results,
        "count_all": {
            "status": "AVAILABLE_AND_EVALUATED",
            "available": True,
            "evaluated": True,
            "formal_protocol_eligible": True,
            "schema_version": COUNT_ALL_ADAPTATION_SCHEMA_VERSION,
            "archive_fields": list(COUNT_ALL_ADAPTATION_ARCHIVE_FIELDS),
            "train_contract": train_count_all_contract,
            "validation_contract": validation_count_all_contract,
            "selection_source": "adaptation_window_A_label_free_counts_only",
            "adaptation_masks_read": False,
            "future_e_counts_used_for_selection": False,
            "forbidden_fallbacks": [
                "historical probability-grid Count-all",
                "sigmoid conversion of raw logits",
                "future-E supervised counts for selection",
                "reconstruction from compressed adaptation statistics",
            ],
        },
        "complete_required_baseline_matrix_ready": True,
        "status": "COMPLETE",
    }
    if _sha256_file(train_path) != train_archive_sha256:
        raise ValueError("Training archive changed after its immutable byte snapshot")
    if _sha256_file(validation_path) != validation_archive_sha256:
        raise ValueError("Validation archive changed after its immutable byte snapshot")
    _write_json_atomic(output_path, payload)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pixel-budgets", nargs="+", required=True, type=float)
    parser.add_argument("--component-budgets", nargs="+", required=True, type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = evaluate_gate_c_baselines(
        train_file=args.train_file,
        validation_file=args.val_file,
        output=args.output,
        pixel_budgets=args.pixel_budgets,
        component_budgets=args.component_budgets,
    )
    print(json.dumps({"output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "COUNT_ALL_ADAPTATION_SCHEMA_VERSION",
    "GATE_C_BASELINES_SCHEMA_VERSION",
    "build_parser",
    "evaluate_gate_c_baselines",
    "select_source_baseline_actions",
]
