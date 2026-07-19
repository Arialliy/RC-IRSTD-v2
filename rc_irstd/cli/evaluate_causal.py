from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from typing import Sequence

import torch

from evaluation.artifact_integrity import verify_score_map_directory
from rc_irstd.evaluation import (
    ScoreStore,
    audit_unseen_target_contract,
    evaluate_threshold,
)
from rc_irstd.features import FeatureSpec, extract_window_features
from rc_irstd.models import MonotoneBudgetCalibrator
from rc_irstd.training import resolve_device
from rc_irstd.utils.checkpoint import load_checkpoint
from rc_irstd.utils.io import atomic_write_json, ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Causal prefix-to-future RC-IRSTD evaluation")
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--calibrator", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--support-size", type=int, default=None)
    parser.add_argument("--query-size", type=int, default=None)
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Window stride; defaults to the calibrator training contract",
    )
    parser.add_argument("--budgets", nargs="+", type=float, default=None)
    parser.add_argument(
        "--all-windows",
        action="store_true",
        help="Evaluate every complete window under the trained stride contract",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-diagnostic-artifacts",
        action="store_true",
        help="Allow seen-domain or diagnostic artifacts and label the result diagnostic",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, _, integrity = verify_score_map_directory(
        args.score_dir,
        require_integrity=True,
        require_masks=True,
    )
    if manifest is None or not integrity.get("verified", False):
        raise ValueError("Causal evaluation requires a complete v3 score manifest")
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.calibrator, device)
    if checkpoint.get("kind") != "calibrator":
        raise ValueError("--calibrator must be a calibrator checkpoint")
    model_config = checkpoint["model_config"]
    model = MonotoneBudgetCalibrator(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    store = ScoreStore(args.score_dir)
    diagnostic_reasons = audit_unseen_target_contract(
        target_dataset=store.dataset_name,
        score_manifest=manifest,
        calibrator_checkpoint=checkpoint,
        allow_diagnostic=args.allow_diagnostic_artifacts,
    )
    episode_metadata = checkpoint.get("episode_metadata", {})
    default_support = int(episode_metadata.get("support_size", 32))
    default_query = int(episode_metadata.get("query_size", 1))
    default_stride = int(
        episode_metadata.get("stride", default_support + default_query)
    )
    support_size = default_support if args.support_size is None else int(args.support_size)
    if support_size <= 0:
        raise ValueError("support_size must be positive")
    query_size = default_query if args.query_size is None else int(args.query_size)
    if query_size <= 0:
        raise ValueError("query_size must be positive")
    if support_size != default_support:
        if not args.allow_diagnostic_artifacts:
            raise ValueError("support_size differs from the calibrator training contract")
        diagnostic_reasons.append("support_size_contract_override")
    if query_size != default_query:
        if not args.allow_diagnostic_artifacts:
            raise ValueError("query_size differs from the calibrator training contract")
        diagnostic_reasons.append("query_size_contract_override")
    window_span = support_size + query_size
    stride = default_stride if args.stride is None else int(args.stride)
    if stride <= 0:
        raise ValueError("stride must be positive")
    if stride != default_stride:
        if not args.allow_diagnostic_artifacts:
            raise ValueError("stride differs from the calibrator training contract")
        diagnostic_reasons.append("stride_contract_override")
    if stride < window_span:
        if not args.allow_diagnostic_artifacts:
            raise ValueError(
                "Formal causal evaluation requires stride >= support_size + query_size"
            )
        diagnostic_reasons.append("cross_window_role_reuse")
    if len(store) < window_span:
        raise ValueError(
            f"Score store has {len(store)} images but the causal contract requires "
            f"{window_span}"
        )
    starts = (
        list(range(0, len(store) - window_span + 1, stride))
        if args.all_windows
        else [0]
    )
    windows = []
    for start in starts:
        support = [store[index] for index in range(start, start + support_size)]
        query = [
            store[index]
            for index in range(start + support_size, start + window_span)
        ]
        windows.append((support, query))
    support_ids = [item.image_id for support, _ in windows for item in support]
    query_ids = [item.image_id for _, query in windows for item in query]
    overlap = set(support_ids).intersection(query_ids)
    if overlap:
        raise ValueError(
            "Support/query image IDs must be disjoint: "
            + ", ".join(sorted(overlap)[:5])
        )
    spec = FeatureSpec.from_dict(checkpoint["feature_spec"])
    features = torch.stack(
        [
            torch.from_numpy(extract_window_features(support, spec))
            for support, _ in windows
        ]
    ).to(device)
    budgets = list(args.budgets or checkpoint["budgets"])
    if not budgets or any(
        not math.isfinite(float(value)) or not float(value) > 0.0
        for value in budgets
    ):
        raise ValueError("All requested budgets must be finite and positive")
    budget_tensor = torch.tensor(budgets, dtype=torch.float32, device=device)
    with torch.no_grad():
        output = model(features, budget_tensor)
    thresholds_by_window = output.requested_thresholds.cpu().tolist()
    if any(
        not math.isfinite(float(value))
        for row in thresholds_by_window
        for value in row
    ):
        raise FloatingPointError("Calibrator produced a non-finite threshold")

    results: list[dict[str, object]] = []
    window_records: list[dict[str, object]] = []
    for window_index, ((support, query), thresholds) in enumerate(
        zip(windows, thresholds_by_window)
    ):
        per_budget: list[dict[str, object]] = []
        for budget, threshold in zip(budgets, thresholds):
            metrics = evaluate_threshold(query, float(threshold))
            record = metrics.to_dict()
            record["budget"] = float(budget)
            per_budget.append(record)
        window_records.append(
            {
                "window_index": window_index,
                "start_index": starts[window_index],
                "support_ids": [item.image_id for item in support],
                "query_ids": [item.image_id for item in query],
                "results": per_budget,
            }
        )
    for budget_index, budget in enumerate(budgets):
        records = [record["results"][budget_index] for record in window_records]
        detected_objects = sum(int(record["detected_objects"]) for record in records)
        total_objects = sum(int(record["total_objects"]) for record in records)
        false_positive_pixels = sum(
            int(record["false_positive_pixels"]) for record in records
        )
        false_positive_components = sum(
            int(record["false_positive_components"]) for record in records
        )
        total_pixels = sum(int(record["total_pixels"]) for record in records)
        threshold_values = [float(record["threshold"]) for record in records]
        pd = detected_objects / max(total_objects, 1)
        fa_pixel = false_positive_pixels / max(total_pixels, 1)
        fa_component_mp = false_positive_components / max(
            total_pixels / 1_000_000.0,
            1e-12,
        )
        results.append(
            {
                "threshold": sum(threshold_values) / len(threshold_values),
                "threshold_min": min(threshold_values),
                "threshold_max": max(threshold_values),
                "pd": float(pd),
                "fa_pixel": float(fa_pixel),
                "fa_component_mp": float(fa_component_mp),
                "detected_objects": detected_objects,
                "total_objects": total_objects,
                "false_positive_pixels": false_positive_pixels,
                "false_positive_components": false_positive_components,
                "total_pixels": total_pixels,
                "budget": float(budget),
                "budget_satisfied": bool(fa_pixel <= float(budget)),
                "relative_excess": max(fa_pixel - float(budget), 0.0)
                / max(float(budget), 1e-12),
            }
        )

    output_dir = ensure_dir(args.output_dir)
    covered_indices = {
        index
        for start in starts
        for index in range(start, start + window_span)
    }
    unused_indices = sorted(set(range(len(store))).difference(covered_indices))
    last_covered_exclusive = starts[-1] + window_span
    unused_tail = max(len(store) - last_covered_exclusive, 0)
    unused_gaps = len(unused_indices) - unused_tail
    payload = {
        "schema_version": "rc-v2-causal-evaluation-v2-trained-window-contract",
        "dataset_name": store.dataset_name,
        "score_dir": str(Path(args.score_dir).resolve()),
        "score_manifest": str((Path(args.score_dir) / "manifest.json").resolve()),
        "score_manifest_sha256": hashlib.sha256(
            (Path(args.score_dir) / "manifest.json").read_bytes()
        ).hexdigest(),
        "score_integrity_verified": True,
        "calibrator": str(Path(args.calibrator).resolve()),
        "calibrator_sha256": hashlib.sha256(Path(args.calibrator).read_bytes()).hexdigest(),
        "support_size": support_size,
        "query_size_per_window": query_size,
        "stride": stride,
        "trained_stride": default_stride,
        "num_windows": len(windows),
        "all_trained_windows": bool(args.all_windows),
        "all_non_overlapping_windows": bool(args.all_windows and stride >= window_span),
        "num_evaluated_query_images": len(query_ids),
        "num_unused_images": len(unused_indices),
        "num_unused_gap_images": unused_gaps,
        "num_unused_tail_images": unused_tail,
        "unused_indices": unused_indices,
        "support_ids": support_ids,
        "query_ids": query_ids,
        "support_query_disjoint": True,
        "test_labels_used": True,
        "threshold_selected_from_query_labels": False,
        "formal_certification": False,
        "formal_unseen_target_contract": not diagnostic_reasons,
        "diagnostic_only": bool(diagnostic_reasons),
        "diagnostic_reasons": diagnostic_reasons,
        "guarantee_scope": "empirical causal audit only",
        "results": results,
        "windows": window_records,
    }
    atomic_write_json(output_dir / "causal_metrics.json", payload)
    with (output_dir / "causal_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(output_dir / "causal_metrics.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
