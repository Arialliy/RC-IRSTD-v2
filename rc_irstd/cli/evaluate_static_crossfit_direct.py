"""Evaluate RC-Direct with a label-blind static five-fold protocol.

This CLI is intentionally separate from the proposed risk-curve pipeline.  It
uses a formal ``MonotoneBudgetCalibrator`` trained with ``A=32, E=1`` and
adapts it to an unordered target collection as an explicitly transductive,
non-causal empirical baseline.  Thresholds are frozen from score/gray support
statistics before any query mask is parsed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation.artifact_integrity import (
    file_sha256,
    ordered_ids_sha256,
    verify_score_map_directory,
)
from rc_irstd.evaluation import (
    ScoreItem,
    audit_unseen_target_contract,
    evaluate_threshold,
    load_score_item,
)
from rc_irstd.features import FeatureSpec, extract_window_features
from rc_irstd.features.domain_statistics import feature_names
from rc_irstd.models import MonotoneBudgetCalibrator
from rc_irstd.training import resolve_device
from rc_irstd.utils.checkpoint import load_checkpoint
from rc_irstd.utils.io import atomic_write_json
from risk_curve.build_curve_episodes import load_score_sample
from risk_curve.build_deployment_statistics import stable_cross_fit_fold_id


SCHEMA_VERSION = "rc-v2-direct-static-crossfit-evaluation-v1"
STATIC_FOLDS = 5
FORMAL_SUPPORT_SIZE = 32
FORMAL_QUERY_SIZE = 1
_BUDGET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class BudgetPair:
    name: str
    pixel: float
    component: float


@dataclass(frozen=True)
class StaticFold:
    fold_index: int
    support_indices: tuple[int, ...]
    query_indices: tuple[int, ...]
    support_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    block_id: str


def parse_budget_pair(value: str) -> BudgetPair:
    """Parse ``name:pixel:component`` without accepting ambiguous names."""

    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "budget pairs must use name:pixel:component"
        )
    name = parts[0].strip()
    if not _BUDGET_NAME.fullmatch(name):
        raise argparse.ArgumentTypeError(
            "budget-pair name must use letters, digits, '.', '_' or '-'"
        )
    try:
        pixel = float(parts[1])
        component = float(parts[2])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "pixel and component budgets must be numbers"
        ) from error
    if not math.isfinite(pixel) or pixel <= 0.0:
        raise argparse.ArgumentTypeError("pixel budget must be finite and positive")
    if not math.isfinite(component) or component <= 0.0:
        raise argparse.ArgumentTypeError(
            "component budget must be finite and positive"
        )
    return BudgetPair(name=name, pixel=pixel, component=component)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_checkpoint_contract(
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], FeatureSpec, tuple[str, ...]]:
    if int(checkpoint.get("format_version", 0)) < 2:
        raise ValueError("RC-Direct static evaluation requires checkpoint format_version >= 2")
    if checkpoint.get("kind") != "calibrator":
        raise ValueError("--calibrator must contain a calibrator checkpoint")
    if checkpoint.get("method_name") != "direct_threshold":
        raise ValueError("Checkpoint method_name must be 'direct_threshold'")
    if checkpoint.get("model_class") != "MonotoneBudgetCalibrator":
        raise ValueError("Checkpoint model_class must be MonotoneBudgetCalibrator")
    if checkpoint.get("role") != "baseline":
        raise ValueError("Checkpoint role must be 'baseline'")
    if bool(checkpoint.get("diagnostic_only", True)) or not bool(
        checkpoint.get("formal_causal_contract", False)
    ):
        raise ValueError(
            "RC-Direct static evaluation requires a non-diagnostic formal calibrator"
        )
    if checkpoint.get("formal_paper_checkpoint") is not True:
        raise ValueError(
            "RC-Direct static evaluation requires formal_paper_checkpoint=true"
        )

    episode = checkpoint.get("episode_metadata")
    if not isinstance(episode, Mapping):
        raise ValueError("Calibrator checkpoint lacks episode_metadata")
    support_size = int(episode.get("support_size", 0))
    query_size = int(episode.get("query_size", 0))
    if support_size != FORMAL_SUPPORT_SIZE or query_size != FORMAL_QUERY_SIZE:
        raise ValueError(
            "RC-Direct static evaluation requires checkpoint A=32/E=1; "
            f"received A={support_size}/E={query_size}"
        )
    if episode.get("mode") != "causal":
        raise ValueError("The RC-Direct checkpoint must declare causal training episodes")
    stride = int(episode.get("stride", 0))
    if stride != support_size + query_size:
        raise ValueError(
            "RC-Direct static evaluation requires checkpoint stride=33 with "
            "disjoint A/E blocks"
        )
    if episode.get("formal_causal_contract") is not True or bool(
        episode.get("diagnostic_only", True)
    ):
        raise ValueError("Checkpoint episode_metadata is not formal and non-diagnostic")

    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Calibrator checkpoint lacks model_config")
    feature_payload = checkpoint.get("feature_spec")
    if not isinstance(feature_payload, dict):
        raise ValueError("Calibrator checkpoint lacks feature_spec")
    spec = FeatureSpec.from_dict(feature_payload)
    expected_names = tuple(feature_names(spec))
    recorded_names = checkpoint.get("feature_names")
    if not isinstance(recorded_names, (list, tuple)) or tuple(
        str(name) for name in recorded_names
    ) != expected_names:
        raise ValueError("Checkpoint feature_names do not match feature_spec")
    if int(model_config.get("feature_dim", 0)) != len(expected_names):
        raise ValueError("Checkpoint model feature_dim does not match feature_spec")

    checkpoint_budgets = np.asarray(checkpoint.get("budgets", []), dtype=np.float64)
    model_budgets = np.asarray(model_config.get("budget_grid", []), dtype=np.float64)
    if (
        checkpoint_budgets.ndim != 1
        or checkpoint_budgets.size < 2
        or model_budgets.shape != checkpoint_budgets.shape
        or not np.allclose(model_budgets, checkpoint_budgets, rtol=1e-6, atol=0.0)
    ):
        raise ValueError("Checkpoint budgets and model_config budget_grid differ")
    return dict(model_config), spec, expected_names


def _validate_target_manifest(manifest: Mapping[str, Any]) -> str:
    target = manifest.get("target_dataset")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("Score manifest target_dataset must be a non-empty string")
    if manifest.get("score_type") != "sigmoid_probability":
        raise ValueError("RC-Direct evaluation requires sigmoid_probability scores")
    if manifest.get("labels_loaded") is not True:
        raise ValueError("RC-Direct labelled audit requires labels_loaded=true")
    if manifest.get("split_role") != "test":
        raise ValueError("RC-Direct static evaluation requires official target test")
    if manifest.get("split_authority_verified") is not True:
        raise ValueError("Target test split authority is not verified")
    required_split_fields = {
        "split_file",
        "split_file_sha256",
        "split_ordered_ids_sha256",
    }
    if not required_split_fields.issubset(manifest):
        raise ValueError("Formal target score manifest lacks complete split provenance")
    return target.strip()


def _build_static_folds(
    image_ids: Sequence[str],
    *,
    folds: int,
    seed: int,
) -> list[StaticFold]:
    if folds != STATIC_FOLDS:
        raise ValueError(f"RC-Direct static baseline requires exactly {STATIC_FOLDS} folds")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    ids = [str(image_id) for image_id in image_ids]
    if not ids or any(not image_id for image_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Score manifest image IDs must be non-empty and unique")
    if folds > len(ids):
        raise ValueError("Static cross-fit requires at least one query image per fold")

    permutation = np.random.default_rng(int(seed)).permutation(len(ids))
    query_folds = [
        np.asarray(row, dtype=np.int64)
        for row in np.array_split(permutation, folds)
    ]
    all_indices = np.arange(len(ids), dtype=np.int64)
    result: list[StaticFold] = []
    for fold_index, query_indices in enumerate(query_folds):
        query_set = set(int(index) for index in query_indices.tolist())
        complement = np.asarray(
            [index for index in all_indices.tolist() if index not in query_set],
            dtype=np.int64,
        )
        if complement.size < FORMAL_SUPPORT_SIZE:
            raise ValueError(
                "Static cross-fit complement is smaller than fixed A=32 in fold "
                f"{fold_index}: complement_size={int(complement.size)}"
            )
        fold_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(fold_index)])
        )
        selected = fold_rng.choice(
            int(complement.size),
            size=FORMAL_SUPPORT_SIZE,
            replace=False,
        )
        support_indices = complement[selected]
        support_ids = tuple(ids[int(index)] for index in support_indices)
        query_ids = tuple(ids[int(index)] for index in query_indices)
        if len(set(support_ids)) != FORMAL_SUPPORT_SIZE:
            raise RuntimeError("Static support sampling was not without replacement")
        if set(support_ids).intersection(query_ids):
            raise RuntimeError("Static support/query roles overlap within a fold")
        result.append(
            StaticFold(
                fold_index=fold_index,
                support_indices=tuple(int(index) for index in support_indices),
                query_indices=tuple(int(index) for index in query_indices),
                support_ids=support_ids,
                query_ids=query_ids,
                block_id=stable_cross_fit_fold_id(
                    fold_index, support_ids, query_ids
                ),
            )
        )

    flattened = [image_id for fold in result for image_id in fold.query_ids]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(ids):
        raise RuntimeError("Static folds do not provide exact full query coverage")
    return result


def _unlabelled_support_item(path: Path, expected_id: str, dataset: str) -> ScoreItem:
    """Read only probability/gray fields; the mask array is never indexed."""

    sample = load_score_sample(path, require_mask=False)
    if sample.image_id != expected_id:
        raise ValueError("Support score ID differs from the verified manifest")
    if sample.gray is None:
        raise ValueError(f"Support score lacks gray array: {path}")
    return ScoreItem(
        probability=sample.probability,
        mask=np.zeros((0, 0), dtype=np.uint8),
        gray=sample.gray,
        image_id=sample.image_id,
        dataset_name=dataset,
        sequence_id=dataset,
        original_hw=tuple(int(value) for value in sample.probability.shape),
        has_mask=False,
        path=path,
    )


def _freeze_thresholds(
    *,
    paths: Sequence[Path],
    folds: Sequence[StaticFold],
    target_dataset: str,
    model: MonotoneBudgetCalibrator,
    spec: FeatureSpec,
    budgets: Sequence[BudgetPair],
    device: torch.device,
) -> tuple[list[list[float]], str]:
    features: list[torch.Tensor] = []
    for fold in folds:
        support = [
            _unlabelled_support_item(
                paths[index], fold.support_ids[position], target_dataset
            )
            for position, index in enumerate(fold.support_indices)
        ]
        features.append(torch.from_numpy(extract_window_features(support, spec)))
    matrix = torch.stack(features).to(device)
    # A two-dimensional request avoids the calibrator's intentional
    # one-budget-per-batch interpretation when K happens to equal five folds.
    requests = torch.tensor(
        [[pair.pixel for pair in budgets]], dtype=torch.float32, device=device
    )
    with torch.no_grad():
        output = model(matrix, requests)
    if output.requested_thresholds is None:
        raise RuntimeError("RC-Direct calibrator returned no requested thresholds")
    thresholds = output.requested_thresholds.detach().cpu().numpy()
    if thresholds.shape != (len(folds), len(budgets)):
        raise RuntimeError("RC-Direct threshold output has an unexpected shape")
    if not np.isfinite(thresholds).all():
        raise FloatingPointError("RC-Direct produced a non-finite threshold")
    frozen = [[float(value) for value in row] for row in thresholds.tolist()]
    freeze_evidence = [
        {
            "block_id": fold.block_id,
            "support_ids": list(fold.support_ids),
            "query_ids": list(fold.query_ids),
            "thresholds": {
                pair.name: frozen[fold_index][budget_index]
                for budget_index, pair in enumerate(budgets)
            },
        }
        for fold_index, fold in enumerate(folds)
    ]
    return frozen, _canonical_sha256(freeze_evidence)


def _metric_record(item: ScoreItem, threshold: float) -> dict[str, float | int]:
    metrics = evaluate_threshold([item], threshold)
    record = metrics.to_dict()
    exposure = max(int(metrics.total_pixels), 1)
    record["pixel_risk"] = float(metrics.false_positive_pixels / exposure)
    record["component_risk_raw"] = float(
        metrics.false_positive_components / max(exposure / 1_000_000.0, 1e-12)
    )
    return record


def _summarise_budget(
    pair: BudgetPair,
    records: Sequence[Mapping[str, float | int]],
    thresholds: Sequence[float],
) -> dict[str, Any]:
    if not records:
        raise ValueError("A budget audit requires at least one query record")
    total_tp = sum(int(record["detected_objects"]) for record in records)
    total_gt = sum(int(record["total_objects"]) for record in records)
    total_fp_pixels = sum(int(record["false_positive_pixels"]) for record in records)
    total_fp_components = sum(
        int(record["false_positive_components"]) for record in records
    )
    total_pixels = sum(int(record["total_pixels"]) for record in records)
    pixel = np.asarray([float(record["pixel_risk"]) for record in records])
    component = np.asarray(
        [float(record["component_risk_raw"]) for record in records]
    )
    satisfied = (pixel <= pair.pixel) & (component <= pair.component)
    relative_excess = np.maximum(
        np.maximum(pixel / pair.pixel - 1.0, component / pair.component - 1.0),
        0.0,
    )
    return {
        "name": pair.name,
        "pixel_budget": pair.pixel,
        "component_budget": pair.component,
        "threshold_request_uses": "pixel_budget",
        "threshold_mean": float(np.mean(thresholds)),
        "threshold_min": float(np.min(thresholds)),
        "threshold_max": float(np.max(thresholds)),
        "pd": float(total_tp / max(total_gt, 1)),
        "detected_objects": total_tp,
        "total_objects": total_gt,
        "pixel_risk": float(total_fp_pixels / max(total_pixels, 1)),
        "component_risk": float(
            total_fp_components / max(total_pixels / 1_000_000.0, 1e-12)
        ),
        "component_risk_raw": float(
            total_fp_components / max(total_pixels / 1_000_000.0, 1e-12)
        ),
        "component_risk_variant": (
            "realised_raw_connected_components_at_frozen_scalar_threshold"
        ),
        "false_positive_pixels": total_fp_pixels,
        "false_positive_components": total_fp_components,
        "total_pixels": total_pixels,
        "pixel_budget_satisfaction_rate": float(np.mean(pixel <= pair.pixel)),
        "component_budget_satisfaction_rate": float(
            np.mean(component <= pair.component)
        ),
        "joint_bsr": float(np.mean(satisfied)),
        "joint_bsr_definition": (
            "fraction_of_query_images_satisfying_both_empirical_budgets"
        ),
        "mean_relative_excess": float(np.mean(relative_excess)),
        "max_relative_excess": float(np.max(relative_excess)),
        "full_query_coverage": True,
    }


def evaluate_static_crossfit_direct(
    *,
    score_dir: str | Path,
    calibrator_path: str | Path,
    output_path: str | Path,
    budget_pairs: Sequence[BudgetPair],
    seed: int = 42,
    folds: int = STATIC_FOLDS,
    device_name: str = "auto",
) -> dict[str, Any]:
    if not budget_pairs:
        raise ValueError("At least one --budget-pair is required")
    if any(not isinstance(pair, BudgetPair) for pair in budget_pairs):
        raise TypeError("budget_pairs must contain BudgetPair values")
    for pair in budget_pairs:
        if not _BUDGET_NAME.fullmatch(pair.name):
            raise ValueError("Budget-pair names have an invalid format")
        if not math.isfinite(pair.pixel) or pair.pixel <= 0.0:
            raise ValueError("Pixel budgets must be finite and positive")
        if not math.isfinite(pair.component) or pair.component <= 0.0:
            raise ValueError("Component budgets must be finite and positive")
    names = [pair.name for pair in budget_pairs]
    if len(set(names)) != len(names):
        raise ValueError("Budget-pair names must be unique")

    score_root = Path(score_dir).expanduser().resolve()
    manifest, paths, integrity = verify_score_map_directory(
        score_root,
        require_integrity=True,
        require_masks=True,
    )
    if manifest is None or not integrity.get("verified", False):
        raise ValueError("RC-Direct static evaluation requires verified v3 scores")
    target_dataset = _validate_target_manifest(manifest)
    image_ids = [str(record["image_id"]) for record in manifest["records"]]
    static_folds = _build_static_folds(image_ids, folds=folds, seed=seed)

    device = resolve_device(device_name)
    checkpoint = load_checkpoint(calibrator_path, device)
    model_config, spec, recorded_feature_names = _strict_checkpoint_contract(
        checkpoint
    )
    diagnostic_reasons = audit_unseen_target_contract(
        target_dataset=target_dataset,
        score_manifest=manifest,
        calibrator_checkpoint=checkpoint,
        allow_diagnostic=False,
    )
    if diagnostic_reasons:
        raise RuntimeError("Unseen-target audit unexpectedly returned diagnostics")

    model = MonotoneBudgetCalibrator(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if not bool(model.normalizer.is_fitted.item()):
        raise ValueError("Calibrator checkpoint contains an unfitted feature normalizer")
    model.eval()
    lower_budget = float(model.budget_grid[-1].detach().cpu())
    upper_budget = float(model.budget_grid[0].detach().cpu())
    for pair in budget_pairs:
        if pair.pixel < lower_budget * (1.0 - 1e-6) or pair.pixel > upper_budget * (
            1.0 + 1e-6
        ):
            raise ValueError(
                f"Budget pair {pair.name!r} pixel budget is outside the trained "
                f"grid [{lower_budget:.6g}, {upper_budget:.6g}]"
            )

    # Phase 1: only score/gray support arrays are parsed.  The complete action
    # table and its digest are fixed before phase 2 invokes ``load_score_item``.
    thresholds_by_fold, frozen_actions_sha256 = _freeze_thresholds(
        paths=paths,
        folds=static_folds,
        target_dataset=target_dataset,
        model=model,
        spec=spec,
        budgets=budget_pairs,
        device=device,
    )

    # Phase 2: masks are now loaded solely for the labelled query audit.
    records_by_budget: dict[str, list[dict[str, float | int]]] = {
        pair.name: [] for pair in budget_pairs
    }
    fold_records: list[dict[str, Any]] = []
    evaluated_ids: list[str] = []
    for fold_index, fold in enumerate(static_folds):
        labelled_queries: list[ScoreItem] = []
        for position, index in enumerate(fold.query_indices):
            item = load_score_item(paths[index])
            expected_id = fold.query_ids[position]
            if item.image_id != expected_id:
                raise ValueError("Query score ID differs from the verified manifest")
            if not item.has_mask:
                raise ValueError(f"Query audit requires a mask: {item.image_id}")
            labelled_queries.append(item)
            evaluated_ids.append(item.image_id)

        per_budget_fold: dict[str, Any] = {}
        for budget_index, pair in enumerate(budget_pairs):
            threshold = thresholds_by_fold[fold_index][budget_index]
            query_records = [
                _metric_record(item, threshold) for item in labelled_queries
            ]
            records_by_budget[pair.name].extend(query_records)
            per_budget_fold[pair.name] = {
                "threshold": threshold,
                "detected_objects": sum(
                    int(record["detected_objects"]) for record in query_records
                ),
                "total_objects": sum(
                    int(record["total_objects"]) for record in query_records
                ),
                "false_positive_pixels": sum(
                    int(record["false_positive_pixels"]) for record in query_records
                ),
                "false_positive_components": sum(
                    int(record["false_positive_components"])
                    for record in query_records
                ),
                "total_pixels": sum(
                    int(record["total_pixels"]) for record in query_records
                ),
            }
        fold_records.append(
            {
                "fold_index": fold.fold_index,
                "block_id": fold.block_id,
                "support_ids": list(fold.support_ids),
                "query_ids": list(fold.query_ids),
                "support_size": len(fold.support_ids),
                "query_size": len(fold.query_ids),
                "support_query_disjoint": True,
                "adaptation_sampling_rule": (
                    "seedsequence(seed,fold_index)_without_replacement_from_"
                    "four_fold_complement"
                ),
                "budgets": per_budget_fold,
            }
        )

    if (
        len(evaluated_ids) != len(image_ids)
        or len(set(evaluated_ids)) != len(image_ids)
        or set(evaluated_ids) != set(image_ids)
    ):
        raise RuntimeError("Labelled audit did not cover every target image exactly once")

    results = [
        _summarise_budget(
            pair,
            records_by_budget[pair.name],
            [row[budget_index] for row in thresholds_by_fold],
        )
        for budget_index, pair in enumerate(budget_pairs)
    ]
    manifest_path = score_root / "manifest.json"
    checkpoint_path = Path(calibrator_path).expanduser().resolve()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method_name": "direct_threshold",
        "model_class": "MonotoneBudgetCalibrator",
        "role": "baseline",
        "adaptation_protocol": "static_5fold_fixed_A_cross_fit",
        "transductive": True,
        "causal_claim": False,
        "formal_certification": False,
        "formal_crc_eligible": False,
        "guarantee": "none; empirical static transductive audit only",
        "target_dataset": target_dataset,
        "num_images": len(image_ids),
        "num_folds": folds,
        "support_size": FORMAL_SUPPORT_SIZE,
        "checkpoint_evaluation_size": FORMAL_QUERY_SIZE,
        "num_evaluated_query_images": len(evaluated_ids),
        "full_test_coverage": True,
        "every_query_evaluated_exactly_once": True,
        "support_masks_parsed": False,
        "query_masks_loaded_after_threshold_freeze": True,
        "test_labels_used_for_threshold_selection": False,
        "test_labels_used_for_labelled_audit": True,
        "unseen_target_contract_verified": True,
        "results": results,
        "folds": fold_records,
        "provenance": {
            "score_dir": str(score_root),
            "score_manifest": str(manifest_path),
            "score_manifest_sha256": file_sha256(manifest_path),
            "score_records_sha256": integrity.get("records_sha256"),
            "score_ordered_image_ids_sha256": integrity.get(
                "ordered_image_ids_sha256"
            ),
            "score_integrity_verified": True,
            "calibrator": str(checkpoint_path),
            "calibrator_sha256": file_sha256(checkpoint_path),
            "checkpoint_contract": {
                "support_size": FORMAL_SUPPORT_SIZE,
                "evaluation_size": FORMAL_QUERY_SIZE,
                "stride": int(checkpoint["episode_metadata"]["stride"]),
                "mode": checkpoint["episode_metadata"]["mode"],
                "formal_causal_contract": True,
                "meta_domains": list(
                    checkpoint["episode_metadata"]["domain_names"]
                ),
                "feature_names_sha256": ordered_ids_sha256(
                    recorded_feature_names
                ),
            },
            "fold_seed": int(seed),
            "fold_assignment_rule": "default_rng(seed).permutation_then_array_split_5",
            "support_sampling_rule": (
                "SeedSequence([seed,fold_index])_without_replacement_from_"
                "four_fold_complement"
            ),
            "query_ids_sha256": ordered_ids_sha256(evaluated_ids),
            "frozen_actions_sha256": frozen_actions_sha256,
            "thresholds_frozen_before_query_mask_loading": True,
        },
    }
    atomic_write_json(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--calibrator", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--budget-pair",
        action="append",
        type=parse_budget_pair,
        required=True,
        help="Repeat name:pixel:component for each operating budget",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=STATIC_FOLDS)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = evaluate_static_crossfit_direct(
        score_dir=args.score_dir,
        calibrator_path=args.calibrator,
        output_path=args.output,
        budget_pairs=args.budget_pair,
        seed=args.seed,
        folds=args.folds,
        device_name=args.device,
    )
    print(Path(args.output).resolve())
    if not payload["full_test_coverage"]:
        raise RuntimeError("Internal error: RC-Direct audit lacks full coverage")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
