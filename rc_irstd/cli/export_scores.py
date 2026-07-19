"""Export score maps through the repository's integrity-checked v3 path.

The command keeps the complete-package interface (``--checkpoint`` and
``--resize-eval``) while deliberately delegating storage, split resolution,
mask alignment, and provenance hashing to :mod:`evaluation.export_score_maps`.
It must not create the legacy unhashed ``entries`` manifest.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from data_ext.eval_dataset import IRSTDEvalDataset
from evaluation.export_score_maps import (
    _load_checkpoint_safely,
    export_dataset_score_maps,
    load_mshnet,
    resolve_checkpoint_warm_flag,
)
from model.MSHNet import MSHNet
from rc_irstd.models import build_mshnet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--checkpoint", dest="weight_path")
    checkpoint.add_argument("--weight-path", dest="weight_path")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--split-file",
        help="Explicit frozen split manifest; otherwise strict local discovery is used",
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument(
        "--source-dataset",
        action="append",
        help="Detector training domain; repeat to bind LODO provenance",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--pad-multiple", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--resize-eval",
        action="store_true",
        help="Diagnostic fixed-size evaluation; native resolution is the default",
    )
    parser.add_argument(
        "--allow-missing-masks",
        action="store_true",
        help=(
            "Compatibility alias for a genuinely label-free export. The masks "
            "directory is not opened and no masks are embedded."
        ),
    )
    parser.add_argument(
        "--labels-loaded",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicitly control whether ground-truth masks are loaded",
    )
    parser.add_argument(
        "--warm-flag",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Select the detector head; defaults to checkpoint metadata",
    )
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument(
        "--export-raw-logits",
        "--raw-logits",
        dest="export_raw_logits",
        action="store_true",
        help=(
            "Also export FP32 raw logits alongside FP32 sigmoid probabilities; "
            "autocast is explicitly disabled for this precision-audit mode"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def _checkpoint_manifest_metadata(
    checkpoint: str | Path,
    checkpoint_metadata: Mapping[str, Any],
    source_datasets: Sequence[str] | None,
) -> dict[str, Any]:
    path = Path(checkpoint).expanduser().resolve()
    metadata: dict[str, Any] = {
        "weight_path": str(path),
        "weight_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checkpoint_epoch": checkpoint_metadata.get("epoch"),
        "checkpoint_warm_flag": checkpoint_metadata.get("warm_flag"),
        "checkpoint_inference_head": checkpoint_metadata.get("inference_head"),
        "checkpoint_selection_rule": checkpoint_metadata.get("selection_rule"),
        "checkpoint_diagnostic_only": bool(
            checkpoint_metadata.get("diagnostic_only", False)
        ),
    }
    model_config = checkpoint_metadata.get("model_config")
    if isinstance(model_config, Mapping):
        metadata["model_backend"] = str(model_config.get("backend", "canonical"))
    saved_sources: list[str] | None = None
    config = checkpoint_metadata.get("config")
    if isinstance(config, Mapping):
        values = config.get("domain_names")
        if isinstance(values, (list, tuple)) and all(
            isinstance(value, str) and value for value in values
        ):
            saved_sources = [str(value) for value in values]
    requested_sources = [str(value) for value in source_datasets or []]
    if requested_sources and len(set(requested_sources)) != len(requested_sources):
        raise ValueError("--source-dataset values must be unique")
    if requested_sources and saved_sources is not None and requested_sources != saved_sources:
        raise ValueError(
            "--source-dataset does not match checkpoint config: "
            f"requested={requested_sources}, checkpoint={saved_sources}"
        )
    resolved_sources = requested_sources or saved_sources
    if resolved_sources:
        metadata["source_datasets"] = resolved_sources
        if len(resolved_sources) == 1:
            metadata["source_dataset"] = resolved_sources[0]
    return metadata


def _load_detector(
    path: str | Path,
    *,
    device: str,
    strict: bool,
) -> torch.nn.Module:
    """Load both canonical raw weights and fixed-last trainer checkpoints."""

    payload = _load_checkpoint_safely(Path(path).expanduser())
    if isinstance(payload, Mapping) and isinstance(payload.get("model_state"), Mapping):
        raw_state = payload["model_state"]
        if not raw_state or not all(isinstance(value, torch.Tensor) for value in raw_state.values()):
            raise ValueError("checkpoint model_state contains non-tensor values")
        state = {
            (str(key)[7:] if str(key).startswith("module.") else str(key)): value
            for key, value in raw_state.items()
        }
        model_config = payload.get("model_config")
        model = (
            build_mshnet(dict(model_config))
            if isinstance(model_config, Mapping)
            else MSHNet(3)
        )
        model.load_state_dict(state, strict=strict)
        config = payload.get("config")
        source_names = payload.get("source_names")
        metadata: dict[str, Any] = {
            "epoch": payload.get("epoch"),
            "warm_flag": payload.get("warm_flag"),
            "inference_head": payload.get("inference_head"),
            "selection_rule": payload.get(
                "selection_rule", payload.get("checkpoint_selection")
            ),
            "model_config": dict(model_config)
            if isinstance(model_config, Mapping)
            else {"backend": "canonical", "input_channels": 3},
            "diagnostic_only": bool(payload.get("diagnostic_only", False)),
            "config": dict(config) if isinstance(config, Mapping) else {},
        }
        if isinstance(source_names, (list, tuple)) and all(
            isinstance(value, str) and value for value in source_names
        ):
            metadata["config"]["domain_names"] = list(source_names)
        model._rc_irstd_checkpoint_metadata = metadata
        return model.to(torch.device(device))
    # Raw/canonical upstream weight files have no complete-package
    # ``model_config``. Keep the original strict canonical loader for them.
    return load_mshnet(path, device=device, strict=strict)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.image_size < 16 or args.image_size % 16 != 0:
        raise ValueError("--image-size must be a positive multiple of 16")
    if args.pad_multiple <= 0 or args.pad_multiple % 16 != 0:
        raise ValueError("--pad-multiple must be a positive multiple of 16")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers non-negative")
    if args.labels_loaded is None:
        labels_loaded = not args.allow_missing_masks
    else:
        labels_loaded = bool(args.labels_loaded)
        if args.allow_missing_masks and labels_loaded:
            raise ValueError(
                "--allow-missing-masks conflicts with --labels-loaded; use "
                "--no-labels-loaded or omit the explicit option"
            )
    device = _resolve_device(args.device)
    dataset = IRSTDEvalDataset(
        args.dataset_dir,
        split_file=args.split_file,
        split=args.split,
        spatial_mode="resize" if args.resize_eval else "native",
        base_size=args.image_size,
        pad_multiple=args.pad_multiple,
        dataset_name=args.dataset_name,
        load_masks=labels_loaded,
    )
    model = _load_detector(
        args.weight_path,
        device=device,
        strict=not args.non_strict,
    )
    checkpoint_metadata = getattr(model, "_rc_irstd_checkpoint_metadata", {})
    warm_flag = resolve_checkpoint_warm_flag(args.warm_flag, checkpoint_metadata)
    metadata = _checkpoint_manifest_metadata(
        args.weight_path,
        checkpoint_metadata,
        args.source_dataset,
    )
    if args.non_strict:
        metadata["checkpoint_diagnostic_only"] = True
        metadata["non_strict_state_loading"] = True
    export_dataset_score_maps(
        model,
        dataset,
        args.output_dir,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        warm_flag=warm_flag,
        labels_loaded=labels_loaded,
        manifest_metadata=metadata,
        export_raw_logits=args.export_raw_logits,
    )
    print(Path(args.output_dir) / "manifest.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
