"""Exact raw-logit Oracle diagnostics matched to deployment decision units.

This module is intentionally diagnostic-only.  For every decision unit it
selects a threshold from that unit's *query/evaluation* labels and never passes
adaptation samples to the exact selector.  Consequently the result is an
upper-bound diagnostic, not a deployable threshold or a formal test result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .policy_matched_oracle import (
    DEFAULT_ADAPTATION_WINDOW,
    DEFAULT_CAUSAL_STRIDE,
    DEFAULT_EVALUATION_WINDOW,
    DEFAULT_SEED,
    DEFAULT_STATIC_FOLDS,
    DecisionUnit,
    build_decision_units,
)
from .raw_logit_oracle import (
    RawLogitSample,
    load_formal_raw_logit_directory,
    raw_logit_stream_sha256,
    select_exact_global_oracle,
)
from .threshold_sweep import write_json_atomic


RAW_LOGIT_POLICY_ORACLE_SCHEMA_VERSION = (
    "rc-v2-exact-raw-logit-policy-oracle-v1"
)
RAW_LOGIT_POLICY_ORACLE_ARTIFACT_TYPE = (
    "rc-irstd-exact-raw-logit-policy-matched-oracle"
)


def _canonical_sha256(value: Any, *, schema: str) -> str:
    payload = json.dumps(
        {"schema": schema, "value": value},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_budget(value: float, *, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return number


def _aggregate_operating_points(
    operating_points: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not operating_points:
        raise ValueError("At least one selected operating point is required")
    fields = (
        "tp_objects",
        "gt_objects",
        "fp_components",
        "fp_pixels",
        "total_pixels",
    )
    totals = {
        field: int(sum(int(point[field]) for point in operating_points))
        for field in fields
    }
    if totals["total_pixels"] <= 0:
        raise ValueError("Selected operating points contain no evaluated pixels")
    totals.update(
        {
            "pd": (
                float(totals["tp_objects"] / totals["gt_objects"])
                if totals["gt_objects"] > 0
                else 0.0
            ),
            "fa_pixel": float(totals["fp_pixels"] / totals["total_pixels"]),
            "fa_component_mp": float(
                totals["fp_components"]
                / (totals["total_pixels"] / 1_000_000.0)
            ),
        }
    )
    return totals


def _policy_interpretation(policy: str) -> dict[str, str]:
    return {
        "global": {
            "evidence_role": "globally_pooled_exact_raw_logit_oracle_upper_bound",
            "budget_enforcement_unit": "single_global_evaluation_pool",
            "aggregate_metric_scope": "all_score_images",
            "interpretation": (
                "One threshold is selected from all target labels. This is an "
                "exact pooled Oracle upper bound, not a deployable threshold."
            ),
        },
        "static": {
            "evidence_role": "primary_exact_raw_logit_policy_matched_oracle_evidence",
            "budget_enforcement_unit": "each_seeded_static_query_fold",
            "aggregate_metric_scope": (
                "raw_count_micro_aggregate_across_exactly_once_query_folds"
            ),
            "interpretation": (
                "The seeded query folds cover the artifact exactly once. Each "
                "fold threshold is selected only from that fold's query labels; "
                "adaptation identities define policy support but their labels are "
                "not passed to threshold selection."
            ),
        },
        "causal": {
            "evidence_role": "exact_raw_logit_causal_sensitivity_evidence",
            "budget_enforcement_unit": "each_causal_evaluation_window",
            "aggregate_metric_scope": (
                "raw_count_micro_aggregate_over_complete_causal_evaluation_windows_only"
            ),
            "interpretation": (
                "Only E frames in complete disjoint A-to-E blocks are evaluated. "
                "For 214 images under A32/E1/stride33 this is exactly 6/214, so "
                "the aggregate must not be presented as full-test-set Pd."
            ),
        },
        "image": {
            "evidence_role": "extremely_permissive_per_image_exact_oracle_upper_bound",
            "budget_enforcement_unit": "each_individual_image",
            "aggregate_metric_scope": (
                "raw_count_micro_aggregate_after_per_image_oracle_selection"
            ),
            "interpretation": (
                "Every image uses its own label-selected exact threshold. This is "
                "an extremely permissive upper bound and is not a realizable policy."
            ),
        },
    }[policy]


def _partition_payload(units: Sequence[DecisionUnit]) -> list[dict[str, Any]]:
    return [
        {
            "unit_id": unit.unit_id,
            "unit_index": int(unit.unit_index),
            "adaptation_ids": list(unit.adaptation_ids),
            "evaluation_ids": list(unit.evaluation_ids),
        }
        for unit in units
    ]


def evaluate_exact_raw_logit_policy(
    samples: Sequence[RawLogitSample],
    *,
    policy: str = "static",
    pixel_budget: float,
    component_budget: float,
    folds: int = DEFAULT_STATIC_FOLDS,
    seed: int = DEFAULT_SEED,
    adaptation_window: int = DEFAULT_ADAPTATION_WINDOW,
    evaluation_window: int = DEFAULT_EVALUATION_WINDOW,
    stride: int = DEFAULT_CAUSAL_STRIDE,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
) -> dict[str, Any]:
    """Run one exact query-label Oracle search per deterministic decision unit."""

    if policy not in {"global", "static", "causal", "image"}:
        raise ValueError("policy must be global, static, causal, or image")
    pixel_budget = _validate_budget(pixel_budget, name="pixel_budget")
    component_budget = _validate_budget(
        component_budget, name="component_budget"
    )
    if not samples:
        raise ValueError("at least one raw-logit sample is required")
    image_ids = [sample.image_id for sample in samples]
    if any(not isinstance(image_id, str) or not image_id for image_id in image_ids):
        raise ValueError("every raw-logit sample requires a non-empty image_id")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("raw-logit sample image_ids must be globally unique")

    units = build_decision_units(
        image_ids,
        policy,
        folds=folds,
        seed=seed,
        adaptation_window=adaptation_window,
        evaluation_window=evaluation_window,
        stride=stride,
    )
    unit_results: list[dict[str, Any]] = []
    selected_points: list[dict[str, Any]] = []
    for unit in units:
        # Leakage boundary: only query/evaluation samples are materialized and
        # passed to the label-derived selector.  Adaptation indices are used
        # solely for partition identity and audit evidence.
        query_samples = [samples[index] for index in unit.evaluation_indices]
        selection = select_exact_global_oracle(
            query_samples,
            pixel_budget=pixel_budget,
            component_budget=component_budget,
            matching_rule=matching_rule,
            centroid_distance=centroid_distance,
            connectivity=connectivity,
            min_component_area=min_component_area,
        )
        point = dict(selection["operating_point"])
        pixel_ok = float(point["fa_pixel"]) <= pixel_budget
        component_ok = float(point["fa_component_mp"]) <= component_budget
        if not (pixel_ok and component_ok):
            raise RuntimeError("exact selector returned a budget-infeasible point")
        selected_points.append(point)
        overlap = sorted(set(unit.adaptation_ids).intersection(unit.evaluation_ids))
        unit_results.append(
            {
                "unit_id": unit.unit_id,
                "unit_index": int(unit.unit_index),
                "adaptation_indices": list(unit.adaptation_indices),
                "evaluation_indices": list(unit.evaluation_indices),
                "adaptation_ids": list(unit.adaptation_ids),
                "evaluation_ids": list(unit.evaluation_ids),
                "num_adaptation_images": len(unit.adaptation_indices),
                "num_evaluation_images": len(unit.evaluation_indices),
                "adaptation_complement_size": unit.adaptation_complement_size,
                "adaptation_sampling_rule": unit.adaptation_sampling_rule,
                "adaptation_sampling_seed_components": (
                    list(unit.adaptation_sampling_seed_components)
                    if unit.adaptation_sampling_seed_components is not None
                    else None
                ),
                "adaptation_evaluation_role_overlap_ids": overlap,
                "labels_used_for_selection_ids": list(unit.evaluation_ids),
                "selection_input_image_ids": [
                    sample.image_id for sample in query_samples
                ],
                "adaptation_labels_used_for_selection": False,
                "query_raw_logit_stream_sha256": raw_logit_stream_sha256(
                    query_samples
                ),
                "threshold_domain": "raw_logit_float32",
                "threshold_logit": float(point["threshold_logit"]),
                "threshold_probability_float64": float(
                    point["threshold_probability_float64"]
                ),
                "threshold_probability_float32": float(
                    point["threshold_probability_float32"]
                ),
                "pd": float(point["pd"]),
                "fa_pixel": float(point["fa_pixel"]),
                "fa_component_mp": float(point["fa_component_mp"]),
                "tp_objects": int(point["tp_objects"]),
                "gt_objects": int(point["gt_objects"]),
                "fp_components": int(point["fp_components"]),
                "fp_pixels": int(point["fp_pixels"]),
                "total_pixels": int(point["total_pixels"]),
                "budget_feasibility": {
                    "pixel_budget_satisfied": pixel_ok,
                    "component_budget_satisfied": component_ok,
                    "joint_budget_satisfied": pixel_ok and component_ok,
                },
                "exact_search": selection["search"],
            }
        )

    aggregate = _aggregate_operating_points(selected_points)
    aggregate["global_aggregate_budget_satisfied"] = bool(
        aggregate["fa_pixel"] <= pixel_budget
        and aggregate["fa_component_mp"] <= component_budget
    )
    evaluation_ids = [
        image_id for unit in units for image_id in unit.evaluation_ids
    ]
    adaptation_ids = [
        image_id for unit in units for image_id in unit.adaptation_ids
    ]
    evaluated_set = set(evaluation_ids)
    full_coverage = (
        len(evaluation_ids) == len(image_ids) and evaluated_set == set(image_ids)
    )
    interpretation = _policy_interpretation(policy)
    policy_contract: dict[str, Any] = {
        "name": policy,
        "num_decision_units": len(units),
        "query_labels_only": True,
        "adaptation_labels_used_for_selection": False,
        "evaluation_ids_globally_unique": len(evaluation_ids)
        == len(evaluated_set),
        "full_evaluation_coverage": full_coverage,
        **interpretation,
    }
    if policy == "static":
        policy_contract.update(
            {
                "folds": int(folds),
                "seed": int(seed),
                "adaptation_window": int(adaptation_window),
                "adaptation_sampling_rule": (
                    "seedsequence(seed,fold_index)_without_replacement_from_complement"
                ),
                "query_fold_sizes": [
                    len(unit.evaluation_indices) for unit in units
                ],
                "selected_adaptation_sizes": [
                    len(unit.adaptation_indices) for unit in units
                ],
                "query_folds_pairwise_disjoint": len(evaluation_ids)
                == len(evaluated_set),
                "exactly_once_query_coverage": full_coverage,
            }
        )
    elif policy == "causal":
        policy_contract.update(
            {
                "adaptation_window": int(adaptation_window),
                "evaluation_window": int(evaluation_window),
                "stride": int(stride),
                "num_complete_causal_blocks": len(units),
                "num_adaptation_assignments": sum(
                    len(unit.adaptation_indices) for unit in units
                ),
                "num_evaluation_assignments": len(evaluation_ids),
                "global_adaptation_evaluation_role_overlap": sorted(
                    set(adaptation_ids).intersection(evaluated_set)
                ),
            }
        )

    partition = _partition_payload(units)
    configuration = {
        "policy": policy,
        "folds": int(folds),
        "seed": int(seed),
        "adaptation_window": int(adaptation_window),
        "evaluation_window": int(evaluation_window),
        "stride": int(stride),
        "pixel_budget": pixel_budget,
        "component_budget": component_budget,
        "matching_rule": matching_rule,
        "centroid_distance": float(centroid_distance),
        "connectivity": int(connectivity),
        "min_component_area": int(min_component_area),
    }
    return {
        "policy": policy_contract,
        "budgets": {
            "pixel": pixel_budget,
            "component_per_megapixel": component_budget,
        },
        "budget_semantics": {
            "selection_enforcement": (
                "each_decision_unit_independently_satisfies_both_budgets"
            ),
            "enforcement_unit": interpretation["budget_enforcement_unit"],
            "global_aggregate_role": (
                "post_selection_audit_not_a_joint_selection_constraint"
            ),
            "cross_unit_label_pooling_for_selection": policy == "global",
        },
        "coverage": {
            "num_score_images": len(image_ids),
            "num_evaluated_images": len(evaluation_ids),
            "num_unique_evaluated_images": len(evaluated_set),
            "num_unevaluated_images": len(image_ids) - len(evaluated_set),
            "evaluated_fraction": float(len(evaluation_ids) / len(image_ids)),
            "evaluated_over_total": f"{len(evaluation_ids)}/{len(image_ids)}",
            "full_evaluation_coverage": full_coverage,
            "aggregate_metrics_cover_evaluated_images_only": True,
            "aggregate_pd_is_full_score_artifact_pd": full_coverage,
            "aggregate_metric_scope": interpretation["aggregate_metric_scope"],
            "unevaluated_image_ids": [
                image_id for image_id in image_ids if image_id not in evaluated_set
            ],
        },
        "partition_audit": {
            "decision_partition_sha256": _canonical_sha256(
                partition, schema="rc-v2-raw-logit-policy-partition-v1"
            ),
            "diagnostic_configuration_sha256": _canonical_sha256(
                configuration, schema="rc-v2-raw-logit-policy-configuration-v1"
            ),
            "evaluation_ids_globally_unique": len(evaluation_ids)
            == len(evaluated_set),
            "within_unit_role_disjoint": all(
                not set(unit.adaptation_ids).intersection(unit.evaluation_ids)
                for unit in units
            ),
            "num_evaluation_assignments": len(evaluation_ids),
            "num_unique_evaluation_ids": len(evaluated_set),
        },
        "units": unit_results,
        "aggregate": aggregate,
        "all_units_individually_budget_satisfied": all(
            bool(unit["budget_feasibility"]["joint_budget_satisfied"])
            for unit in unit_results
        ),
        "matching": {
            "rule": matching_rule,
            "centroid_distance": float(centroid_distance),
            "connectivity": int(connectivity),
            "min_component_area": int(min_component_area),
        },
    }


def build_raw_logit_policy_oracle_payload(
    score_dir: str | Path,
    *,
    policy: str = "static",
    pixel_budget: float,
    component_budget: float,
    folds: int = DEFAULT_STATIC_FOLDS,
    seed: int = DEFAULT_SEED,
    adaptation_window: int = DEFAULT_ADAPTATION_WINDOW,
    evaluation_window: int = DEFAULT_EVALUATION_WINDOW,
    stride: int = DEFAULT_CAUSAL_STRIDE,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
) -> dict[str, Any]:
    """Load a strict raw-logit v3 artifact and build a hash-bound payload."""

    samples, manifest, _integrity, contract = load_formal_raw_logit_directory(
        score_dir
    )
    evaluated = evaluate_exact_raw_logit_policy(
        samples,
        policy=policy,
        pixel_budget=pixel_budget,
        component_budget=component_budget,
        folds=folds,
        seed=seed,
        adaptation_window=adaptation_window,
        evaluation_window=evaluation_window,
        stride=stride,
        matching_rule=matching_rule,
        centroid_distance=centroid_distance,
        connectivity=connectivity,
        min_component_area=min_component_area,
    )
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != len(samples):
        raise ValueError("Raw-logit manifest records do not align with samples")
    record_hashes = [
        {
            "image_id": str(record.get("image_id", "")),
            "file": str(record.get("file", "")),
            "file_sha256": str(record.get("sha256", "")),
        }
        for record in records
    ]
    hashes = {
        "score_manifest_sha256": contract["score_manifest_sha256"],
        "score_records_sha256": contract["score_records_sha256"],
        "score_ordered_image_ids_sha256": contract[
            "score_ordered_image_ids_sha256"
        ],
        "raw_logit_stream_sha256": raw_logit_stream_sha256(samples),
        "detector_weight_sha256": contract["detector_weight_sha256"],
        "split_file_sha256": contract["split_file_sha256"],
        "split_ordered_ids_sha256": contract["split_ordered_ids_sha256"],
        "decision_partition_sha256": evaluated["partition_audit"][
            "decision_partition_sha256"
        ],
        "diagnostic_configuration_sha256": evaluated["partition_audit"][
            "diagnostic_configuration_sha256"
        ],
        "record_file_sha256_by_image_id": record_hashes,
    }
    return {
        "schema_version": RAW_LOGIT_POLICY_ORACLE_SCHEMA_VERSION,
        "artifact_type": RAW_LOGIT_POLICY_ORACLE_ARTIFACT_TYPE,
        "mode": "exact_raw_logit_policy_matched_target_oracle_diagnostic",
        "diagnostic_only": True,
        "oracle_only": True,
        "test_labels_used_for_threshold_selection": True,
        "adaptation_labels_used_for_threshold_selection": False,
        "formal_protocol_eligible": False,
        "deployment_threshold_eligible": False,
        "guarantee": (
            "none; each action is selected from its query/evaluation labels"
        ),
        **evaluated,
        "provenance": {
            "score_dir": str(Path(score_dir).expanduser().resolve()),
            "score_manifest_schema_version": int(manifest["schema_version"]),
            "checkpoint_selection_rule": contract["checkpoint_selection_rule"],
            "model_backend": contract["model_backend"],
            "target_dataset": contract["target_dataset"],
            "source_datasets": contract["source_datasets"],
            "requested_split": contract["requested_split"],
            "split_role": contract["split_role"],
            "score_representation": contract["score_representation"],
            "probability_dtype": contract["probability_dtype"],
            "logit_dtype": contract["logit_dtype"],
            "probability_transform": contract["probability_transform"],
            "probability_clipping": contract["probability_clipping"],
            "inference_autocast_enabled": contract[
                "inference_autocast_enabled"
            ],
            "input_artifact_formal_contract_verified": True,
            "hashes": hashes,
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--policy", choices=("global", "static", "causal", "image"), default="static"
    )
    parser.add_argument("--pixel-budget", type=float, required=True)
    parser.add_argument("--component-budget", type=float, required=True)
    parser.add_argument("--folds", type=int, default=DEFAULT_STATIC_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--adaptation-window", type=int, default=DEFAULT_ADAPTATION_WINDOW)
    parser.add_argument("--evaluation-window", type=int, default=DEFAULT_EVALUATION_WINDOW)
    parser.add_argument("--stride", type=int, default=DEFAULT_CAUSAL_STRIDE)
    parser.add_argument("--matching-rule", choices=("overlap", "centroid"), default="overlap")
    parser.add_argument("--centroid-distance", type=float, default=3.0)
    parser.add_argument("--connectivity", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument(
        "--oracle-diagnostic",
        action="store_true",
        help=(
            "Required acknowledgement: target query labels select each exact action"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if not args.oracle_diagnostic:
        raise ValueError(
            "raw_logit_policy_oracle reads target query labels; pass "
            "--oracle-diagnostic"
        )
    payload = build_raw_logit_policy_oracle_payload(
        args.score_dir,
        policy=args.policy,
        pixel_budget=args.pixel_budget,
        component_budget=args.component_budget,
        folds=args.folds,
        seed=args.seed,
        adaptation_window=args.adaptation_window,
        evaluation_window=args.evaluation_window,
        stride=args.stride,
        matching_rule=args.matching_rule,
        centroid_distance=args.centroid_distance,
        connectivity=args.connectivity,
        min_component_area=args.min_component_area,
    )
    write_json_atomic(args.output, payload)
    print(Path(args.output))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "RAW_LOGIT_POLICY_ORACLE_ARTIFACT_TYPE",
    "RAW_LOGIT_POLICY_ORACLE_SCHEMA_VERSION",
    "build_argument_parser",
    "build_raw_logit_policy_oracle_payload",
    "evaluate_exact_raw_logit_policy",
    "main",
]
