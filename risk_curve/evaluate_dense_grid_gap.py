"""Source-only exact-vs-dense raw-logit Oracle discretisation diagnostic.

This command is label-derived and diagnostic only.  It may be run on official
source-train pseudo-target artifacts to decide whether the frozen source-only
grid is dense enough; its output must never select a deployment threshold and
must never consume the outer target.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.artifact_integrity import (
    PROBABILITY_DTYPE,
    RAW_LOGIT_DTYPE,
    RAW_LOGIT_SCORE_REPRESENTATION,
    verify_score_map_directory,
)
from evaluation.budget_metrics import is_budget_satisfied
from evaluation.component_matching import match_components
from evaluation.raw_logit_oracle import RawLogitSample, select_exact_global_oracle
from evaluation.threshold_sweep import (
    domain_key,
    validate_formal_score_manifest,
    write_json_atomic,
)

from .representation import (
    LOGIT_REPRESENTATION,
    LogitGridArtifact,
    load_logit_grid_artifact,
    validate_logit_threshold_grid,
)


DENSE_GRID_GAP_SCHEMA_VERSION = "rc-v4-source-dense-grid-gap-v1"
DEFAULT_BUDGETS = (
    ("loose", 1e-5, 5.0),
    ("strict", 1e-6, 1.0),
)


def _validate_raw_train_contract(
    manifest: Mapping[str, Any] | None,
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    contract = validate_formal_score_manifest(
        manifest,
        integrity,
        expected_split_role="train",
    )
    assert manifest is not None
    expected = {
        "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
        "probability_dtype": PROBABILITY_DTYPE,
        "logit_dtype": RAW_LOGIT_DTYPE,
        "probability_transform": "sigmoid",
        "probability_clipping": "none",
        "inference_autocast_enabled": False,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"Dense-grid gap requires manifest {field}={value!r}"
            )
    return {**contract, **expected}


def load_source_pseudo_target(
    score_dir: str | Path,
) -> tuple[list[RawLogitSample], dict[str, Any]]:
    """Load one held-out source-train pseudo-target with strict raw provenance."""

    manifest, paths, integrity = verify_score_map_directory(
        score_dir,
        require_integrity=True,
        require_masks=True,
    )
    contract = _validate_raw_train_contract(manifest, integrity)
    assert manifest is not None
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != len(paths):
        raise ValueError("Pseudo-target manifest/path records differ")
    samples: list[RawLogitSample] = []
    for index, (record, path) in enumerate(zip(records, paths)):
        with np.load(path, allow_pickle=False) as payload:
            required = {"image_id", "logit", "prob", "mask"}
            missing = sorted(required.difference(payload.files))
            if missing:
                raise ValueError(
                    f"Pseudo-target raw record {index} is missing: "
                    + ", ".join(missing)
                )
            image_id = str(np.asarray(payload["image_id"]).item())
            logits = np.asarray(payload["logit"])
            probability = np.asarray(payload["prob"])
            mask = np.asarray(payload["mask"])
        if not isinstance(record, dict) or image_id != record.get("image_id"):
            raise ValueError(f"Pseudo-target raw record {index} identity mismatch")
        if logits.dtype != np.float32 or probability.dtype != np.float32:
            raise ValueError("Pseudo-target raw scores must use float32")
        if (
            logits.ndim != 2
            or probability.shape != logits.shape
            or mask.shape != logits.shape
            or not np.isfinite(logits).all()
            or not np.isfinite(probability).all()
        ):
            raise ValueError(f"Pseudo-target raw record {index} has invalid arrays")
        samples.append(
            RawLogitSample(
                image_id=image_id,
                logits=logits,
                probability=probability,
                mask=(mask > 0),
            )
        )
    return samples, {
        **contract,
        "score_dir": str(Path(score_dir).expanduser().resolve()),
        "representation": LOGIT_REPRESENTATION,
    }


def _dense_metrics_row(
    samples: Sequence[RawLogitSample],
    threshold: float,
    *,
    threshold_index: int,
    matching_rule: str,
    centroid_distance: float,
    connectivity: int,
    min_component_area: int,
) -> dict[str, Any]:
    totals = {
        "tp_objects": 0,
        "gt_objects": 0,
        "fp_components": 0,
        "fp_pixels": 0,
        "total_pixels": 0,
    }
    for sample in samples:
        result = match_components(
            sample.logits >= np.float32(threshold),
            sample.mask,
            rule=matching_rule,
            centroid_distance=centroid_distance,
            connectivity=connectivity,
            min_component_area=min_component_area,
        )
        totals["tp_objects"] += int(result.num_tp_objects)
        totals["gt_objects"] += int(result.num_gt)
        totals["fp_components"] += int(result.num_fp_components)
        totals["fp_pixels"] += int(result.num_fp_pixels)
        totals["total_pixels"] += int(sample.mask.size)
    total_pixels = totals["total_pixels"]
    return {
        "threshold_index": int(threshold_index),
        "threshold_logit": float(np.float32(threshold)),
        "empty_action": False,
        "pd": (
            float(totals["tp_objects"] / totals["gt_objects"])
            if totals["gt_objects"]
            else 0.0
        ),
        "fa_pixel": float(totals["fp_pixels"] / total_pixels),
        "fa_component_mp": float(
            totals["fp_components"] / (total_pixels / 1_000_000.0)
        ),
        **totals,
    }


def _empty_metrics_row(
    samples: Sequence[RawLogitSample],
    *,
    matching_rule: str,
    centroid_distance: float,
    connectivity: int,
    min_component_area: int,
) -> dict[str, Any]:
    gt_objects = 0
    total_pixels = 0
    for sample in samples:
        result = match_components(
            np.zeros(sample.mask.shape, dtype=bool),
            sample.mask,
            rule=matching_rule,
            centroid_distance=centroid_distance,
            connectivity=connectivity,
            min_component_area=min_component_area,
        )
        gt_objects += int(result.num_gt)
        total_pixels += int(sample.mask.size)
    return {
        "threshold_index": None,
        "threshold_logit": "+inf",
        "empty_action": True,
        "pd": 0.0,
        "fa_pixel": 0.0,
        "fa_component_mp": 0.0,
        "tp_objects": 0,
        "gt_objects": gt_objects,
        "fp_components": 0,
        "fp_pixels": 0,
        "total_pixels": total_pixels,
    }


def select_dense_grid_oracle(
    samples: Sequence[RawLogitSample],
    thresholds: np.ndarray,
    *,
    pixel_budget: float,
    component_budget: float,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
    provably_infeasible_at_or_below_logit: float | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Select the best label-derived operating point on one fixed finite grid."""

    if not samples:
        raise ValueError("at least one pseudo-target sample is required")
    grid = validate_logit_threshold_grid(np.asarray(thresholds))
    if pixel_budget <= 0.0 or component_budget <= 0.0:
        raise ValueError("risk budgets must be positive")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")

    selected = _empty_metrics_row(
        samples,
        matching_rule=matching_rule,
        centroid_distance=centroid_distance,
        connectivity=connectivity,
        min_component_area=min_component_area,
    )
    feasible_finite = 0
    pruned_finite = 0
    candidates: list[tuple[int, float]] = []
    for index, threshold in enumerate(grid.tolist()):
        if (
            provably_infeasible_at_or_below_logit is not None
            and float(threshold) <= provably_infeasible_at_or_below_logit
        ):
            pruned_finite += 1
            continue
        candidates.append((index, float(threshold)))

    def evaluate_candidate(candidate: tuple[int, float]) -> dict[str, Any]:
        index, threshold = candidate
        return _dense_metrics_row(
            samples,
            threshold,
            threshold_index=index,
            matching_rule=matching_rule,
            centroid_distance=centroid_distance,
            connectivity=connectivity,
            min_component_area=min_component_area,
        )

    if workers == 1 or len(candidates) <= 1:
        rows = map(evaluate_candidate, candidates)
    else:
        # Threads share the multi-gigabyte immutable score maps.  executor.map
        # preserves candidate order, so integer aggregation and tie-breaking
        # are bitwise identical to the single-worker path.
        executor = ThreadPoolExecutor(max_workers=min(workers, len(candidates)))
        rows = executor.map(evaluate_candidate, candidates)
    try:
        for row in rows:
            if not is_budget_satisfied(
                row,
                pixel_budget=pixel_budget,
                component_budget=component_budget,
            ):
                continue
            feasible_finite += 1
            if (
                int(row["tp_objects"]) > int(selected["tp_objects"])
                or (
                    int(row["tp_objects"]) == int(selected["tp_objects"])
                    and selected["threshold_index"] is None
                )
                or (
                    int(row["tp_objects"]) == int(selected["tp_objects"])
                    and selected["threshold_index"] is not None
                    and int(row["threshold_index"])
                    < int(selected["threshold_index"])
                )
            ):
                selected = row
    finally:
        if workers > 1 and len(candidates) > 1:
            executor.shutdown(wait=True)
    return {
        "found": True,
        "strategy": "maximize_global_pd_then_lowest_dense_logit_threshold",
        "operating_point": selected,
        "search": {
            "finite_grid_points_total": int(grid.size),
            "finite_grid_points_evaluated": len(candidates),
            "finite_grid_points_pixel_budget_proven_infeasible": (
                pruned_finite
            ),
            "provably_infeasible_at_or_below_logit": (
                provably_infeasible_at_or_below_logit
            ),
            "external_empty_action_evaluated": True,
            "feasible_finite_grid_points": feasible_finite,
            "workers": workers,
        },
    }


def compare_exact_and_dense(
    samples: Sequence[RawLogitSample],
    thresholds: np.ndarray,
    *,
    pixel_budget: float,
    component_budget: float,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
    workers: int = 1,
) -> dict[str, Any]:
    exact = select_exact_global_oracle(
        samples,
        pixel_budget=pixel_budget,
        component_budget=component_budget,
        matching_rule=matching_rule,
        centroid_distance=centroid_distance,
        connectivity=connectivity,
        min_component_area=min_component_area,
    )
    pruning = exact["search"]["lossless_pruning"]
    cutoff_value = pruning.get("provably_infeasible_at_or_below_logit")
    cutoff = float(cutoff_value) if cutoff_value is not None else None
    dense = select_dense_grid_oracle(
        samples,
        thresholds,
        pixel_budget=pixel_budget,
        component_budget=component_budget,
        matching_rule=matching_rule,
        centroid_distance=centroid_distance,
        connectivity=connectivity,
        min_component_area=min_component_area,
        provably_infeasible_at_or_below_logit=cutoff,
        workers=workers,
    )
    exact_pd = float(exact["operating_point"]["pd"])
    dense_pd = float(dense["operating_point"]["pd"])
    if dense_pd > exact_pd + 1e-12:
        raise AssertionError("Dense-grid Oracle cannot exceed the exact raw Oracle")
    return {
        "pixel_budget": float(pixel_budget),
        "component_budget": float(component_budget),
        "exact": exact,
        "dense": dense,
        "exact_pd": exact_pd,
        "dense_pd": dense_pd,
        "grid_gap": float(exact_pd - dense_pd),
    }


def _parse_budget(value: str) -> tuple[str, float, float]:
    parts = value.split(":")
    if len(parts) != 3 or not parts[0]:
        raise argparse.ArgumentTypeError("budget must be NAME:PIXEL:COMPONENT")
    try:
        pixel = float(parts[1])
        component = float(parts[2])
    except ValueError as error:
        raise argparse.ArgumentTypeError("budget values must be numeric") from error
    if not np.isfinite(pixel) or not np.isfinite(component) or min(pixel, component) <= 0:
        raise argparse.ArgumentTypeError("budget values must be finite and positive")
    return parts[0], pixel, component


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-map-dir", action="append", required=True)
    parser.add_argument("--threshold-grid-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--budget",
        action="append",
        type=_parse_budget,
        help="Repeat NAME:PIXEL:COMPONENT; defaults to loose and strict",
    )
    parser.add_argument("--matching-rule", choices=("overlap", "centroid"), default="overlap")
    parser.add_argument("--centroid-distance", type=float, default=3.0)
    parser.add_argument("--connectivity", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of deterministic dense-threshold worker threads "
            "(default: 1; score maps are shared read-only)"
        ),
    )
    return parser


def _validate_global_provenance(
    artifact: LogitGridArtifact,
    contracts: Sequence[Mapping[str, Any]],
) -> None:
    target_keys = {str(contract["target_domain_key"]) for contract in contracts}
    if target_keys != set(artifact.manifest["source_domain_keys"]):
        raise ValueError(
            "Pseudo-target domains do not exactly match global grid source domains"
        )
    checkpoint_hashes = {
        str(contract["detector_weight_sha256"]) for contract in contracts
    }
    if checkpoint_hashes != set(
        artifact.manifest["episode_detector_checkpoint_sha256s"]
    ):
        raise ValueError(
            "Pseudo-target detector checkpoints do not match the inner folds "
            "bound by the global grid"
        )
    checkpoint_to_source_set = {
        str(item["detector_checkpoint_sha256"]): set(
            map(str, item["source_domain_keys"])
        )
        for item in artifact.manifest["detector_folds"]
        if item["role"] == "inner_pseudo_target_detector"
    }
    for contract in contracts:
        checkpoint_hash = str(contract["detector_weight_sha256"])
        expected_sources = checkpoint_to_source_set.get(checkpoint_hash)
        actual_sources = set(map(str, contract["source_domain_keys"]))
        expected_target = set(artifact.manifest["source_domain_keys"]).difference(
            expected_sources or set()
        )
        if (
            expected_sources is None
            or actual_sources != expected_sources
            or expected_target != {str(contract["target_domain_key"])}
        ):
            raise ValueError(
                "Pseudo-target detector source does not match its self-score "
                "fold in the global grid"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be a positive integer")
    artifact = load_logit_grid_artifact(args.threshold_grid_manifest)
    loaded = [load_source_pseudo_target(path) for path in args.score_map_dir]
    samples_by_domain = [item[0] for item in loaded]
    contracts = [item[1] for item in loaded]
    _validate_global_provenance(artifact, contracts)
    budgets = tuple(args.budget) if args.budget else DEFAULT_BUDGETS
    if len({name for name, _, _ in budgets}) != len(budgets):
        raise ValueError("budget names must be unique")

    domain_results: list[dict[str, Any]] = []
    for samples, contract in zip(samples_by_domain, contracts):
        comparisons = {
            name: compare_exact_and_dense(
                samples,
                artifact.thresholds,
                pixel_budget=pixel,
                component_budget=component,
                matching_rule=args.matching_rule,
                centroid_distance=args.centroid_distance,
                connectivity=args.connectivity,
                min_component_area=args.min_component_area,
                workers=args.workers,
            )
            for name, pixel, component in budgets
        }
        loose = comparisons.get("loose")
        strict = comparisons.get("strict")
        gate_reasons: list[str] = []
        if loose is not None and float(loose["grid_gap"]) > 0.02:
            gate_reasons.append("loose_grid_gap_gt_0.02")
        if strict is not None:
            strict_limit = max(0.01, 0.10 * float(strict["exact_pd"]))
            strict["gate_limit"] = strict_limit
            if float(strict["grid_gap"]) > strict_limit:
                gate_reasons.append("strict_grid_gap_exceeds_limit")
        domain_results.append(
            {
                "pseudo_target": contract["target_dataset"],
                "contract": contract,
                "comparisons": comparisons,
                "grid_gap_gate_pass": not gate_reasons,
                "grid_gap_gate_reasons": gate_reasons,
            }
        )

    payload = {
        "schema_version": DENSE_GRID_GAP_SCHEMA_VERSION,
        "artifact_type": "rc-irstd-source-only-dense-grid-gap",
        "diagnostic_only": True,
        "labels_used": "source_official_train_pseudo_target_only",
        "outer_target_labels_used": False,
        "may_select_deployment_threshold": False,
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_sha256": artifact.semantic_sha256,
        "threshold_grid_manifest": str(artifact.manifest_path),
        "outer_target": artifact.manifest["outer_target"],
        "grid_gap_gate_pass": all(
            bool(row["grid_gap_gate_pass"]) for row in domain_results
        ),
        "domains": domain_results,
    }
    write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "grid_gap_gate_pass": payload["grid_gap_gate_pass"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BUDGETS",
    "DENSE_GRID_GAP_SCHEMA_VERSION",
    "compare_exact_and_dense",
    "load_source_pseudo_target",
    "main",
    "select_dense_grid_oracle",
]
