"""Exact source pseudo-target dynamic-budget oracle for v4 diagnostics.

This module answers one deliberately narrow question: if a different finite
raw-logit grid action (or the external reject action) could be chosen for every
source validation episode *with access to its future-E labels*, what is the
largest aggregate true-positive count attainable under the fold-wide integer
pixel-FP and component-FP capacities?

The answer is an oracle upper bound, not a deployable selector.  Source
pseudo-target future-E sufficient counts are used inside the action
optimisation.  Outer-target data are excluded by the formal archive contract
and must never be supplied to this evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .curve_dataset import load_curve_archive
from .evaluate_source_pseudo_target_v4 import (
    _episode_identifier,
    _validate_formal_archive,
)
from .representation import (
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_REPRESENTATION,
    empty_action_contract,
)


DYNAMIC_BUDGET_ORACLE_SCHEMA_VERSION = (
    "rc-v4-source-dynamic-budget-oracle-diagnostic-v1"
)
REGISTERED_BUDGETS = (
    ("loose", 1e-5, 5.0),
    ("strict", 1e-6, 1.0),
)
FORMAL_SOURCE_DOMAIN_KEYS = frozenset(("irstd1k", "nudt"))
FORMAL_OUTER_DOMAIN_KEY = "nuaa"


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


def _domain_key(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty domain name")
    text = value.strip()
    key = "".join(character for character in text.casefold() if character.isalnum())
    if key.endswith("sirst"):
        key = key[: -len("sirst")]
    if not key:
        raise ValueError(f"{field} normalises to an empty domain key")
    return key


def _integer_array(
    archive: Mapping[str, np.ndarray],
    field: str,
    *,
    shape: tuple[int, ...],
    minimum: int = 0,
) -> np.ndarray:
    if field not in archive:
        raise ValueError(f"Formal oracle archive is missing {field}")
    raw = np.asarray(archive[field])
    if raw.shape != shape:
        raise ValueError(f"{field} must have shape {shape}")
    if raw.dtype.kind not in "iu":
        if not np.isfinite(raw).all() or not np.all(np.equal(raw, np.floor(raw))):
            raise ValueError(f"{field} must contain finite integer counts")
    values = raw.astype(np.int64)
    if np.any(values < minimum):
        raise ValueError(f"{field} contains values below {minimum}")
    return values


@dataclass(frozen=True)
class _OracleArchive:
    thresholds: np.ndarray
    pixel_fp: np.ndarray
    component_fp: np.ndarray
    true_positive: np.ndarray
    gt_objects: np.ndarray
    total_pixels: np.ndarray
    pseudo_targets: tuple[str, ...]
    evaluation_ids: tuple[tuple[str, ...], ...]
    provenance: dict[str, Any]

    @property
    def num_episodes(self) -> int:
        return int(self.pixel_fp.shape[0])


def _validate_oracle_archive(
    archive: dict[str, np.ndarray],
) -> _OracleArchive:
    """Fail closed on representation, source scope, counts, and curve binding."""

    thresholds, statistics, _names, provenance = _validate_formal_archive(archive)
    rows, grid_size = int(statistics.shape[0]), int(thresholds.size)
    if "adaptation_ids" not in archive:
        raise ValueError("Formal oracle archive is missing adaptation_ids")
    adaptation_ids = np.asarray(archive["adaptation_ids"])
    if adaptation_ids.shape != (rows,):
        raise ValueError(f"adaptation_ids must have shape ({rows},)")
    parsed_adaptation = [
        tuple(_episode_identifier(value, index))
        for index, value in enumerate(adaptation_ids.tolist())
    ]
    evaluation_raw = np.asarray(archive["evaluation_ids"])
    evaluation_ids = tuple(
        tuple(_episode_identifier(value, index))
        for index, value in enumerate(evaluation_raw.tolist())
    )
    adaptation_flat = [item for episode in parsed_adaptation for item in episode]
    evaluation_flat = [item for episode in evaluation_ids for item in episode]
    if len(adaptation_flat) != len(set(adaptation_flat)):
        raise ValueError("Formal oracle archive repeats an adaptation sample")
    if len(evaluation_flat) != len(set(evaluation_flat)):
        raise ValueError("Formal oracle archive repeats an evaluation sample")
    if set(adaptation_flat).intersection(evaluation_flat):
        raise ValueError("Formal oracle archive reuses a sample across A and E")

    pixel_fp = _integer_array(
        archive, "pixel_fp_counts", shape=(rows, grid_size)
    )
    component_fp = _integer_array(
        archive, "component_fp_counts", shape=(rows, grid_size)
    )
    true_positive = _integer_array(
        archive, "tp_object_counts", shape=(rows, grid_size)
    )
    gt_objects = _integer_array(
        archive, "gt_object_counts", shape=(rows,)
    )
    total_pixels = _integer_array(
        archive, "total_pixels", shape=(rows,), minimum=1
    )
    if np.any(pixel_fp > total_pixels[:, None]):
        raise ValueError("pixel_fp_counts exceed their episode pixel exposure")
    if np.any(component_fp > total_pixels[:, None]):
        raise ValueError("component_fp_counts exceed their episode pixel exposure")
    if np.any(true_positive > gt_objects[:, None]):
        raise ValueError("tp_object_counts exceed gt_object_counts")

    # Bind the integer sufficient counts to every persisted supervision curve;
    # otherwise a tampered label/count archive could still look formally valid.
    expected_pixel = np.log10(
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
    expected_pd = true_positive.astype(np.float64) / np.maximum(
        gt_objects[:, None], 1
    )
    for field, expected in (
        ("pixel_log_risk", expected_pixel),
        ("component_log_risk_raw", expected_component_raw),
        ("component_log_risk_upper", expected_component_upper),
        ("pd_curve", expected_pd),
    ):
        if field not in archive:
            raise ValueError(f"Formal oracle archive is missing {field}")
        observed = np.asarray(archive[field], dtype=np.float64)
        if observed.shape != expected.shape or not np.allclose(
            observed, expected, rtol=2e-6, atol=2e-6
        ):
            raise ValueError(f"{field} disagrees with stored sufficient counts")

    targets_raw = np.asarray(archive["pseudo_targets"])
    pseudo_targets = tuple(str(item) for item in targets_raw.tolist())
    if len(pseudo_targets) != rows:
        raise ValueError("pseudo_targets must contain one domain per episode")
    if provenance.get("archive_split") != "validation":
        raise ValueError("Dynamic oracle requires a formal validation archive")
    declared_targets = provenance.get("pseudo_targets")
    if not isinstance(declared_targets, list):
        raise ValueError("Dynamic oracle requires declared formal source domains")
    declared_source_keys = {
        _domain_key(value, "provenance.pseudo_targets")
        for value in declared_targets
    }
    if declared_source_keys != FORMAL_SOURCE_DOMAIN_KEYS:
        raise ValueError(
            "Dynamic oracle source domains must be exactly IRSTD-1K and NUDT-SIRST"
        )
    outer_key = _domain_key(
        provenance.get("threshold_grid_outer_target_key"),
        "threshold_grid_outer_target_key",
    )
    if outer_key != FORMAL_OUTER_DOMAIN_KEY:
        raise ValueError("Dynamic oracle excluded outer target must be NUAA-SIRST")
    row_target_keys = {
        _domain_key(value, "pseudo_targets row") for value in pseudo_targets
    }
    if not row_target_keys or not row_target_keys.issubset(FORMAL_SOURCE_DOMAIN_KEYS):
        raise ValueError("Dynamic oracle rows must contain formal source domains only")
    validation_key = _domain_key(
        provenance.get("validation_domain"), "validation_domain"
    )
    if validation_key not in FORMAL_SOURCE_DOMAIN_KEYS:
        raise ValueError("Dynamic oracle validation domain must be a formal source")
    if row_target_keys != {validation_key}:
        raise ValueError("Dynamic oracle rows do not match the validation domain")
    if _scalar_text(archive["representation"], "representation") != (
        LOGIT_REPRESENTATION
    ):
        raise ValueError("Dynamic oracle requires raw-logit v4 representation")
    if _scalar_text(
        archive["threshold_grid_schema_version"],
        "threshold_grid_schema_version",
    ) != LOGIT_GRID_SCHEMA_VERSION:
        raise ValueError("Dynamic oracle requires the formal raw-logit grid")
    return _OracleArchive(
        thresholds=thresholds,
        pixel_fp=pixel_fp,
        component_fp=component_fp,
        true_positive=true_positive,
        gt_objects=gt_objects,
        total_pixels=total_pixels,
        pseudo_targets=pseudo_targets,
        evaluation_ids=evaluation_ids,
        provenance=provenance,
    )


@dataclass(frozen=True)
class _Option:
    pixel_fp: int
    component_fp: int
    true_positive: int
    action_rank: int


def _episode_options(
    pixel_fp: np.ndarray,
    component_fp: np.ndarray,
    true_positive: np.ndarray,
    *,
    pixel_capacity: int,
    component_capacity: int,
) -> tuple[_Option, ...]:
    """Return the exact nondominated option set, including external reject."""

    grid_size = int(pixel_fp.size)
    best_exact: dict[tuple[int, int], _Option] = {}
    for index in range(grid_size + 1):
        if index == grid_size:
            option = _Option(0, 0, 0, grid_size)
        else:
            option = _Option(
                int(pixel_fp[index]),
                int(component_fp[index]),
                int(true_positive[index]),
                index,
            )
        if (
            option.pixel_fp > pixel_capacity
            or option.component_fp > component_capacity
        ):
            continue
        key = (option.pixel_fp, option.component_fp)
        old = best_exact.get(key)
        if old is None or (
            option.true_positive > old.true_positive
            or (
                option.true_positive == old.true_positive
                and option.action_rank < old.action_rank
            )
        ):
            best_exact[key] = option

    # A 2-D prefix maximum identifies every cost/reward dominated option.
    # Keeping only strict improvements is exact because any future episode adds
    # the same non-negative cost/reward to both the option and its dominator.
    reward = np.full(
        (pixel_capacity + 1, component_capacity + 1), -1, dtype=np.int64
    )
    for (pixel_cost, component_cost), option in best_exact.items():
        reward[pixel_cost, component_cost] = option.true_positive
    prefix = np.full_like(reward, -1)
    retained: list[_Option] = []
    for pixel_cost in range(pixel_capacity + 1):
        for component_cost in range(component_capacity + 1):
            prior = -1
            if pixel_cost > 0:
                prior = max(prior, int(prefix[pixel_cost - 1, component_cost]))
            if component_cost > 0:
                prior = max(prior, int(prefix[pixel_cost, component_cost - 1]))
            value = int(reward[pixel_cost, component_cost])
            if value >= 0 and value > prior:
                retained.append(best_exact[(pixel_cost, component_cost)])
            prefix[pixel_cost, component_cost] = max(prior, value)
    if not retained:
        raise AssertionError("External reject must make every episode feasible")
    return tuple(retained)


@dataclass(frozen=True)
class KnapsackSolution:
    action_ranks: tuple[int, ...]
    total_true_positive: int
    used_pixel_fp: int
    used_component_fp: int
    pixel_capacity: int
    component_capacity: int


def _prune_states(
    states: Mapping[tuple[int, int], tuple[int, tuple[int, ...]]],
    *,
    pixel_capacity: int,
    component_capacity: int,
) -> dict[tuple[int, int], tuple[int, tuple[int, ...]]]:
    reward = np.full(
        (pixel_capacity + 1, component_capacity + 1), -1, dtype=np.int64
    )
    for (pixel_cost, component_cost), (value, _path) in states.items():
        reward[pixel_cost, component_cost] = value
    prefix = np.full_like(reward, -1)
    retained: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {}
    for pixel_cost in range(pixel_capacity + 1):
        for component_cost in range(component_capacity + 1):
            prior = -1
            if pixel_cost > 0:
                prior = max(prior, int(prefix[pixel_cost - 1, component_cost]))
            if component_cost > 0:
                prior = max(prior, int(prefix[pixel_cost, component_cost - 1]))
            value = int(reward[pixel_cost, component_cost])
            if value >= 0 and value > prior:
                retained[(pixel_cost, component_cost)] = states[
                    (pixel_cost, component_cost)
                ]
            prefix[pixel_cost, component_cost] = max(prior, value)
    return retained


def solve_multiple_choice_2d_knapsack(
    pixel_fp_counts: np.ndarray,
    component_fp_counts: np.ndarray,
    tp_object_counts: np.ndarray,
    *,
    pixel_capacity: int,
    component_capacity: int,
) -> KnapsackSolution:
    """Solve the exact per-episode multiple-choice 2-D integer knapsack.

    The deterministic ordering is: maximum total TP, minimum used pixel FP,
    minimum used component FP, then lexicographically smallest action ranks.
    Finite grid indices have ranks ``0..G-1`` and external reject has rank
    ``G``.  Exactly one rank is returned for every episode.
    """

    if isinstance(pixel_capacity, bool) or not isinstance(pixel_capacity, int):
        raise TypeError("pixel_capacity must be an integer")
    if isinstance(component_capacity, bool) or not isinstance(
        component_capacity, int
    ):
        raise TypeError("component_capacity must be an integer")
    if pixel_capacity < 0 or component_capacity < 0:
        raise ValueError("Knapsack capacities must be non-negative")
    pixel = np.asarray(pixel_fp_counts)
    component = np.asarray(component_fp_counts)
    true_positive = np.asarray(tp_object_counts)
    if (
        pixel.ndim != 2
        or pixel.shape != component.shape
        or pixel.shape != true_positive.shape
        or pixel.shape[0] == 0
        or pixel.shape[1] < 2
    ):
        raise ValueError("Count arrays must share non-empty shape [N,G], G >= 2")
    for name, values in (
        ("pixel_fp_counts", pixel),
        ("component_fp_counts", component),
        ("tp_object_counts", true_positive),
    ):
        if values.dtype.kind not in "iu":
            if not np.isfinite(values).all() or not np.all(
                np.equal(values, np.floor(values))
            ):
                raise ValueError(f"{name} must contain finite integer counts")
        if np.any(values < 0):
            raise ValueError(f"{name} must contain non-negative counts")
    pixel = pixel.astype(np.int64)
    component = component.astype(np.int64)
    true_positive = true_positive.astype(np.int64)

    states: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {
        (0, 0): (0, ())
    }
    for row in range(pixel.shape[0]):
        options = _episode_options(
            pixel[row],
            component[row],
            true_positive[row],
            pixel_capacity=pixel_capacity,
            component_capacity=component_capacity,
        )
        candidates: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {}
        for (used_pixel, used_component), (reward, path) in states.items():
            for option in options:
                new_pixel = used_pixel + option.pixel_fp
                new_component = used_component + option.component_fp
                if (
                    new_pixel > pixel_capacity
                    or new_component > component_capacity
                ):
                    continue
                key = (new_pixel, new_component)
                candidate = (
                    reward + option.true_positive,
                    path + (option.action_rank,),
                )
                old = candidates.get(key)
                if old is None or candidate[0] > old[0] or (
                    candidate[0] == old[0] and candidate[1] < old[1]
                ):
                    candidates[key] = candidate
        states = _prune_states(
            candidates,
            pixel_capacity=pixel_capacity,
            component_capacity=component_capacity,
        )
        if not states:
            raise AssertionError("External reject must keep DP feasible")

    (used_pixel, used_component), (reward, path) = min(
        states.items(),
        key=lambda item: (
            -item[1][0],
            item[0][0],
            item[0][1],
            item[1][1],
        ),
    )
    return KnapsackSolution(
        action_ranks=path,
        total_true_positive=int(reward),
        used_pixel_fp=int(used_pixel),
        used_component_fp=int(used_component),
        pixel_capacity=pixel_capacity,
        component_capacity=component_capacity,
    )


def _registered_budgets(
    budgets: Sequence[tuple[str, float, float]] | None,
) -> tuple[tuple[str, float, float], ...]:
    values = REGISTERED_BUDGETS if budgets is None else tuple(budgets)
    if not values:
        raise ValueError("At least one oracle budget is required")
    names: set[str] = set()
    parsed: list[tuple[str, float, float]] = []
    for item in values:
        if len(item) != 3:
            raise ValueError("Every budget must be (name,pixel,component)")
        name, pixel, component = item
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Budget names must be non-empty and unique")
        pixel_value, component_value = float(pixel), float(component)
        if (
            not math.isfinite(pixel_value)
            or not math.isfinite(component_value)
            or pixel_value <= 0.0
            or component_value <= 0.0
        ):
            raise ValueError("Oracle budgets must be finite and positive")
        names.add(name)
        parsed.append((name, pixel_value, component_value))
    return tuple(parsed)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def evaluate_dynamic_budget_oracle(
    *,
    episode_file: str | Path,
    output: str | Path,
    budgets: Sequence[tuple[str, float, float]] | None = None,
) -> Path:
    """Evaluate and persist the exact diagnostic-only source oracle bound."""

    episode_path = Path(episode_file).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not episode_path.is_file():
        raise FileNotFoundError(f"Oracle episode archive does not exist: {episode_path}")
    episode_sha256 = _sha256_file(episode_path)
    archive = load_curve_archive(episode_path)
    validated = _validate_oracle_archive(archive)
    if _sha256_file(episode_path) != episode_sha256:
        raise ValueError("Oracle episode archive changed during validation")

    total_pixels = int(validated.total_pixels.sum())
    total_gt = int(validated.gt_objects.sum())
    results: list[dict[str, Any]] = []
    for budget_name, pixel_budget, component_budget in _registered_budgets(budgets):
        pixel_capacity = int(math.floor(pixel_budget * total_pixels))
        component_capacity = int(
            math.floor(component_budget * (total_pixels / 1_000_000.0))
        )
        solution = solve_multiple_choice_2d_knapsack(
            validated.pixel_fp,
            validated.component_fp,
            validated.true_positive,
            pixel_capacity=pixel_capacity,
            component_capacity=component_capacity,
        )
        grid_size = int(validated.thresholds.size)
        actions: list[dict[str, Any]] = []
        for row, rank in enumerate(solution.action_ranks):
            reject = rank == grid_size
            if reject:
                pixel_fp = component_fp = true_positive = 0
                threshold_index: int | None = None
                selected_threshold: float | str = "+inf"
            else:
                pixel_fp = int(validated.pixel_fp[row, rank])
                component_fp = int(validated.component_fp[row, rank])
                true_positive = int(validated.true_positive[row, rank])
                threshold_index = int(rank)
                selected_threshold = float(validated.thresholds[rank])
            actions.append(
                {
                    "episode_index": row,
                    "pseudo_target": validated.pseudo_targets[row],
                    "evaluation_ids": list(validated.evaluation_ids[row]),
                    "action_rank": int(rank),
                    "threshold_index": threshold_index,
                    "selected_logit_threshold": selected_threshold,
                    "reject": reject,
                    "pixel_fp_count": pixel_fp,
                    "component_fp_count": component_fp,
                    "tp_object_count": true_positive,
                    "gt_object_count": int(validated.gt_objects[row]),
                    "total_pixels": int(validated.total_pixels[row]),
                }
            )
        if sum(item["tp_object_count"] for item in actions) != (
            solution.total_true_positive
        ):
            raise AssertionError("Oracle action reconstruction changed objective")
        if sum(item["pixel_fp_count"] for item in actions) != solution.used_pixel_fp:
            raise AssertionError("Oracle action reconstruction changed pixel cost")
        if sum(item["component_fp_count"] for item in actions) != (
            solution.used_component_fp
        ):
            raise AssertionError("Oracle action reconstruction changed component cost")

        pixel_risk = solution.used_pixel_fp / float(total_pixels)
        component_risk = solution.used_component_fp / (
            total_pixels / 1_000_000.0
        )
        results.append(
            {
                "budget_name": budget_name,
                "pixel_budget": pixel_budget,
                "component_budget": component_budget,
                "capacities": {
                    "pixel_fp_capacity": pixel_capacity,
                    "component_fp_capacity": component_capacity,
                    "pixel_capacity_formula": "floor(pixel_budget * total_pixels)",
                    "component_capacity_formula": (
                        "floor(component_budget * (total_pixels / 1e6))"
                    ),
                },
                "used_capacities": {
                    "pixel_fp_used": solution.used_pixel_fp,
                    "component_fp_used": solution.used_component_fp,
                    "pixel_fp_slack": pixel_capacity - solution.used_pixel_fp,
                    "component_fp_slack": (
                        component_capacity - solution.used_component_fp
                    ),
                },
                "aggregate": {
                    "tp_object_count": solution.total_true_positive,
                    "gt_object_count": total_gt,
                    "total_pixels": total_pixels,
                    "pd": solution.total_true_positive / float(max(total_gt, 1)),
                    "pixel_risk": pixel_risk,
                    "component_risk": component_risk,
                    "pixel_budget_satisfied": bool(pixel_risk <= pixel_budget),
                    "component_budget_satisfied": bool(
                        component_risk <= component_budget
                    ),
                    "joint_budget_satisfied": bool(
                        solution.used_pixel_fp <= pixel_capacity
                        and solution.used_component_fp <= component_capacity
                    ),
                    "reject_rate": sum(item["reject"] for item in actions)
                    / float(validated.num_episodes),
                },
                "optimality": {
                    "exact": True,
                    "algorithm": (
                        "multiple_choice_2d_integer_knapsack_sparse_pareto_dp"
                    ),
                    "objective": "maximize aggregate source future-E TP count",
                    "one_action_per_episode": True,
                    "external_reject_included": True,
                    "deterministic_tie_break": [
                        "maximum total TP",
                        "minimum used pixel FP",
                        "minimum used component FP",
                        (
                            "lexicographically smallest action ranks; finite grid "
                            "indices 0..G-1, external reject rank G"
                        ),
                    ],
                    "proof_scope": (
                        "all registered finite grid actions plus external reject "
                        "under the two integer capacities"
                    ),
                },
                "actions": actions,
            }
        )

    payload = {
        "schema_version": DYNAMIC_BUDGET_ORACLE_SCHEMA_VERSION,
        "method_name": "source_dynamic_budget_oracle",
        "role": "diagnostic_upper_bound_only",
        "status": "COMPLETE",
        "diagnostic_only": True,
        "deployment_eligible": False,
        "selector_eligible": False,
        "must_not_be_used_by_selector": True,
        "not_a_proposed_method_result": True,
        "not_an_outer_target_claim": True,
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": _scalar_text(
            archive["threshold_grid_sha256"], "threshold_grid_sha256"
        ),
        "threshold_grid_size": int(validated.thresholds.size),
        "external_reject_action": empty_action_contract(),
        "episode_archive": str(episode_path),
        "episode_archive_sha256": episode_sha256,
        "num_episodes": validated.num_episodes,
        "pseudo_targets": sorted(set(validated.pseudo_targets)),
        "excluded_outer_target": validated.provenance.get(
            "threshold_grid_outer_target_key"
        ),
        "labels_policy": {
            "source_pseudo_target_future_E_labels_used_for_optimization": True,
            "action_selection_is_label_free": False,
            "labels_used_after_action_selection_only": False,
            "labels_used_after": (
                "not_applicable: source future-E labels are used inside the offline "
                "oracle optimization, not merely after action selection"
            ),
            "outer_target_labels_used": False,
            "outer_target_data_used": False,
            "formal_source_domain_scope_verified": True,
            "formal_outer_target_exclusion_verified": True,
        },
        "interpretation": (
            "Upper bound on aggregate source pseudo-target Pd when thresholds may "
            "vary by episode and only fold-wide FP capacities are enforced. It "
            "cannot train, tune, validate, or deploy a selector."
        ),
        "budgets": results,
    }
    _write_json_atomic(output_path, payload)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-file", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = evaluate_dynamic_budget_oracle(
        episode_file=args.episode_file,
        output=args.output,
    )
    print(json.dumps({"output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "DYNAMIC_BUDGET_ORACLE_SCHEMA_VERSION",
    "KnapsackSolution",
    "REGISTERED_BUDGETS",
    "build_parser",
    "evaluate_dynamic_budget_oracle",
    "solve_multiple_choice_2d_knapsack",
]
