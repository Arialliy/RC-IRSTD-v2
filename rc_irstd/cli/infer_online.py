from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from evaluation.artifact_integrity import verify_score_map_directory
from rc_irstd.evaluation import ScoreStore, audit_unseen_target_contract
from rc_irstd.features import FeatureSpec, extract_window_features
from rc_irstd.models import MonotoneBudgetCalibrator
from rc_irstd.training import resolve_device
from rc_irstd.utils.checkpoint import load_checkpoint
from rc_irstd.utils.io import atomic_write_json, ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply a calibrated threshold to an unlabeled target stream")
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--calibrator", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--support-size", type=int, default=None)
    parser.add_argument("--query-size", type=int, default=None)
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Window stride; defaults to the calibrator training contract",
    )
    parser.add_argument(
        "--all-windows",
        action="store_true",
        help="Predict every complete query window under the trained stride contract",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-diagnostic-artifacts",
        action="store_true",
        help="Allow seen-domain or diagnostic artifacts and label the result diagnostic",
    )
    return parser


def _safe_mask_name(image_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", image_id).strip("._") or "sample"
    if value != image_id:
        value += "-" + hashlib.sha1(image_id.encode("utf-8")).hexdigest()[:8]
    return value + ".png"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, _, integrity = verify_score_map_directory(
        args.score_dir,
        require_integrity=True,
        require_masks=False,
    )
    if manifest is None or not integrity.get("verified", False):
        raise ValueError("Online inference requires a complete mask-free v3 score manifest")
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.calibrator, device)
    if checkpoint.get("kind") != "calibrator":
        raise ValueError("--calibrator must be a calibrator checkpoint")
    model = MonotoneBudgetCalibrator(**checkpoint["model_config"]).to(device)
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
    query_size = default_query if args.query_size is None else int(args.query_size)
    if support_size <= 0:
        raise ValueError("support_size must be positive")
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
                "Formal online inference requires stride >= support_size + query_size"
            )
        diagnostic_reasons.append("cross_window_role_reuse")
    if not math.isfinite(args.budget) or args.budget <= 0:
        raise ValueError("budget must be finite and positive")
    if len(store) < window_span:
        raise ValueError("The stream is shorter than the trained support/query contract")
    image_ids = [store.manifest["records"][index]["image_id"] for index in range(len(store))]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Online score stream contains duplicate image IDs")
    starts = (
        list(range(0, len(store) - window_span + 1, stride))
        if args.all_windows
        else [0]
    )
    windows = [
        (
            [store[index] for index in range(start, start + support_size)],
            [
                store[index]
                for index in range(start + support_size, start + window_span)
            ],
        )
        for start in starts
    ]
    support_ids = [item.image_id for support, _ in windows for item in support]
    query_ids = [item.image_id for _, query in windows for item in query]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Online windows contain duplicate query image IDs")
    overlap = set(support_ids).intersection(query_ids)
    if overlap:
        raise ValueError(
            "Support/query image IDs must be disjoint across online windows: "
            + ", ".join(sorted(overlap)[:5])
        )
    spec = FeatureSpec.from_dict(checkpoint["feature_spec"])
    features = torch.stack(
        [
            torch.from_numpy(extract_window_features(support, spec))
            for support, _ in windows
        ]
    ).to(device)
    with torch.no_grad():
        thresholds = (
            model(features, torch.tensor([args.budget], device=device))
            .requested_thresholds[:, 0]
            .cpu()
            .tolist()
        )
    if any(not math.isfinite(float(threshold)) for threshold in thresholds):
        raise FloatingPointError("Calibrator produced a non-finite threshold")
    output_dir = ensure_dir(args.output_dir)
    mask_dir = ensure_dir(output_dir / "masks")
    entries: list[dict[str, object]] = []
    window_records: list[dict[str, object]] = []
    for window_index, ((support, query), threshold) in enumerate(
        zip(windows, thresholds)
    ):
        for query_offset, item in enumerate(query):
            # Keep deployment identical to training/oracle/evaluation semantics:
            # pixels exactly on the selected threshold are positive.
            prediction = (item.probability >= float(threshold)).astype(np.uint8) * 255
            path = mask_dir / _safe_mask_name(item.image_id)
            temporary = path.with_name(path.name + ".tmp")
            Image.fromarray(prediction, mode="L").save(temporary, format="PNG")
            temporary.replace(path)
            entry = {
                "image_id": item.image_id,
                "stream_index": starts[window_index] + support_size + query_offset,
                "window_index": window_index,
                "threshold": float(threshold),
                "mask": str(Path("masks") / path.name),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            entries.append(entry)
        window_records.append(
            {
                "window_index": window_index,
                "start_index": starts[window_index],
                "support_ids": [item.image_id for item in support],
                "query_ids": [item.image_id for item in query],
                "threshold": float(threshold),
            }
        )
    covered_indices = {
        index
        for start in starts
        for index in range(start, start + window_span)
    }
    unused_indices = sorted(set(range(len(store))).difference(covered_indices))
    last_covered_exclusive = starts[-1] + window_span
    unused_tail = max(len(store) - last_covered_exclusive, 0)
    unused_gaps = len(unused_indices) - unused_tail
    threshold_values = [float(value) for value in thresholds]
    atomic_write_json(
        output_dir / "inference.json",
        {
            "schema_version": "rc-v2-online-inference-v2-trained-window-contract",
            "dataset_name": store.dataset_name,
            "score_dir": str(Path(args.score_dir).resolve()),
            "score_manifest_sha256": integrity["manifest_sha256"],
            "score_integrity_verified": True,
            "labels_loaded": False,
            "formal_unseen_target_contract": not diagnostic_reasons,
            "diagnostic_only": bool(diagnostic_reasons),
            "diagnostic_reasons": diagnostic_reasons,
            "calibrator": str(Path(args.calibrator).resolve()),
            "calibrator_sha256": hashlib.sha256(
                Path(args.calibrator).read_bytes()
            ).hexdigest(),
            "budget": args.budget,
            "threshold": threshold_values[0] if len(threshold_values) == 1 else None,
            "threshold_mean": sum(threshold_values) / len(threshold_values),
            "threshold_min": min(threshold_values),
            "threshold_max": max(threshold_values),
            "support_size": support_size,
            "query_size_per_window": query_size,
            "stride": stride,
            "trained_stride": default_stride,
            "all_trained_windows": bool(args.all_windows),
            "num_windows": len(windows),
            "num_stream_images": len(store),
            "num_support_observations": len(support_ids),
            "num_unused_images": len(unused_indices),
            "num_unused_gap_images": unused_gaps,
            "num_unused_tail_images": unused_tail,
            "unused_indices": unused_indices,
            "support_ids": support_ids,
            "query_ids": query_ids,
            "support_query_disjoint": True,
            "num_predictions": len(entries),
            "entries": entries,
            "windows": window_records,
        },
    )
    print(output_dir / "inference.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
