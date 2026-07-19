"""Export cropped, continuous sigmoid score maps and a provenance manifest."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from data_ext.dataset_meta import crop_to_valid, meta_to_jsonable
from data_ext.eval_dataset import IRSTDEvalDataset
from data_ext.mask_alignment import (
    MASK_ALIGNMENT_NOT_LOADED_POLICY,
    MASK_ALIGNMENT_POLICY,
    validate_mask_alignment_evidence,
)
from evaluation.artifact_integrity import (
    PROBABILITY_DTYPE,
    RAW_LOGIT_DTYPE,
    RAW_LOGIT_SCORE_REPRESENTATION,
    SCORE_MANIFEST_SCHEMA_VERSION,
    SCORE_MASK_ALIGNMENT_SCHEMA,
    SCORE_RECORD_INTEGRITY_SCHEMA,
    file_sha256,
    ordered_ids_sha256,
    score_records_sha256,
)


def _extract_logits(output: Any) -> torch.Tensor:
    """Extract the last image-like tensor from common segmentation outputs."""

    if isinstance(output, torch.Tensor):
        if output.ndim < 3:
            raise ValueError(f"Model tensor output must have ndim >= 3, got {output.shape}")
        return output
    # The complete-package facade exposes a typed dataclass instead of a raw
    # tuple.  Read only its public tensor field so the canonical exporter can
    # serve both APIs without depending on rc_irstd internals.
    structured_logits = getattr(output, "logits", None)
    if isinstance(structured_logits, torch.Tensor):
        return _extract_logits(structured_logits)
    if isinstance(output, Mapping):
        for preferred_key in ("logits", "out", "output", "prediction"):
            if preferred_key in output:
                return _extract_logits(output[preferred_key])
        candidates = list(output.values())
    elif isinstance(output, (tuple, list)):
        candidates = list(output)
    else:
        raise TypeError(f"Unsupported model output type: {type(output).__name__}")
    for candidate in reversed(candidates):
        try:
            return _extract_logits(candidate)
        except (TypeError, ValueError):
            continue
    raise ValueError("Model output did not contain an image-like tensor")


def _forward_model(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    warm_flag: bool = True,
) -> torch.Tensor:
    """Call MSHNet's warm_flag API while remaining usable with ordinary models."""

    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "warm_flag" in parameters:
        output = model(images, warm_flag=warm_flag)
    else:
        output = model(images)
    logits = _extract_logits(output)
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"Expected Bx1xHxW logits, got {tuple(logits.shape)}")
    if logits.shape[0] != images.shape[0]:
        raise ValueError("Model output batch size differs from input batch size")
    if logits.shape[-2:] != images.shape[-2:]:
        logits = functional.interpolate(
            logits,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return logits


def _unbatch_meta(
    batched_meta: Mapping[str, Any],
    index: int,
    batch_size: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batched_meta.items():
        if isinstance(value, torch.Tensor):
            if value.ndim == 0 or value.shape[0] != batch_size:
                raise ValueError(f"Collated metadata tensor {key!r} has invalid shape")
            result[key] = value[index]
        elif isinstance(value, np.ndarray):
            if value.ndim == 0 or value.shape[0] != batch_size:
                raise ValueError(f"Collated metadata array {key!r} has invalid shape")
            result[key] = value[index]
        elif isinstance(value, (list, tuple)):
            if len(value) != batch_size:
                raise ValueError(f"Collated metadata sequence {key!r} has invalid length")
            result[key] = value[index]
        else:
            if batch_size != 1:
                raise ValueError(f"Cannot unbatch metadata field {key!r}")
            result[key] = value
    return result


def _safe_record_name(image_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", image_id).strip("._")
    if not safe:
        safe = "sample"
    if safe != image_id:
        digest = hashlib.sha1(image_id.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe}-{digest}"
    return f"{safe}.npz"


def save_score_map(
    output_path: str | Path,
    *,
    probability: np.ndarray,
    gray: np.ndarray,
    mask: np.ndarray | None,
    meta: Mapping[str, Any],
    labels_loaded: bool | None = None,
    raw_logits: np.ndarray | None = None,
) -> Path:
    """Validate and save one compressed score-map record."""

    path = Path(output_path).expanduser()
    probability = np.asarray(probability, dtype=np.float32)
    gray = np.asarray(gray, dtype=np.float32)
    logit_array = (
        np.asarray(raw_logits, dtype=np.float32) if raw_logits is not None else None
    )
    if labels_loaded is None:
        labels_loaded = mask is not None
    if not isinstance(labels_loaded, bool):
        raise TypeError("labels_loaded must be boolean")
    if labels_loaded != (mask is not None):
        raise ValueError("labels_loaded must exactly describe whether a mask is supplied")
    mask_array = np.asarray(mask) if mask is not None else None
    if probability.ndim != 2 or gray.ndim != 2:
        raise ValueError("probability and gray must be 2-D after cropping")
    if mask_array is not None and mask_array.ndim != 2:
        raise ValueError("mask must be 2-D after cropping")
    if logit_array is not None and logit_array.ndim != 2:
        raise ValueError("raw_logits must be 2-D after cropping")
    if probability.shape != gray.shape or (
        mask_array is not None and probability.shape != mask_array.shape
    ) or (
        logit_array is not None and probability.shape != logit_array.shape
    ):
        raise ValueError("probability, raw_logits, gray, and mask shapes differ")
    if not np.isfinite(probability).all() or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError("probability must be finite and lie in [0, 1]")
    if not np.isfinite(gray).all() or np.any((gray < 0.0) | (gray > 1.0)):
        raise ValueError("gray must be finite and lie in [0, 1]")
    if mask_array is not None and not np.isfinite(mask_array).all():
        raise ValueError("mask contains NaN or infinity")
    if logit_array is not None and not np.isfinite(logit_array).all():
        raise ValueError("raw_logits contains NaN or infinity")
    json_meta = meta_to_jsonable(dict(meta))
    required = {
        "image_id",
        "dataset_name",
        "original_hw",
        "input_hw",
        "valid_hw",
        "padding_ltrb",
        "spatial_mode",
        "mask_alignment_applied",
        "mask_original_hw",
        "mask_aspect_relative_error",
        "mask_alignment_policy",
    }
    missing = required.difference(json_meta)
    if missing:
        raise ValueError(f"metadata is missing fields: {', '.join(sorted(missing))}")
    alignment_applied = json_meta["mask_alignment_applied"]
    if not isinstance(alignment_applied, bool):
        raise ValueError("metadata mask_alignment_applied must be boolean")
    original_hw = tuple(int(value) for value in json_meta["original_hw"])
    mask_original_hw = tuple(int(value) for value in json_meta["mask_original_hw"])
    if len(original_hw) != 2 or len(mask_original_hw) != 2:
        raise ValueError("metadata source dimensions must each contain two integers")
    alignment_error = float(json_meta["mask_aspect_relative_error"])
    alignment_policy = str(json_meta["mask_alignment_policy"])
    validate_mask_alignment_evidence(
        labels_loaded=labels_loaded,
        image_hw=original_hw,
        original_mask_hw=mask_original_hw,
        applied=alignment_applied,
        relative_error=alignment_error,
        policy=alignment_policy,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "prob": probability,
        "gray": gray,
        "image_id": np.asarray(str(json_meta["image_id"])),
        "dataset_name": np.asarray(str(json_meta["dataset_name"])),
        "original_hw": np.asarray(json_meta["original_hw"], dtype=np.int32),
        "input_hw": np.asarray(json_meta["input_hw"], dtype=np.int32),
        "valid_hw": np.asarray(json_meta["valid_hw"], dtype=np.int32),
        "padding_ltrb": np.asarray(json_meta["padding_ltrb"], dtype=np.int32),
        "spatial_mode": np.asarray(str(json_meta["spatial_mode"])),
        "labels_loaded": np.asarray(labels_loaded),
        "mask_alignment_applied": np.asarray(alignment_applied),
        "mask_original_hw": np.asarray(mask_original_hw, dtype=np.int32),
        "mask_aspect_relative_error": np.asarray(
            alignment_error,
            dtype=np.float64,
        ),
        "mask_alignment_policy": np.asarray(alignment_policy),
    }
    if mask_array is not None:
        arrays["mask"] = (mask_array > 0).astype(np.uint8)
    if logit_array is not None:
        arrays.update(
            {
                "logit": logit_array,
                "score_representation": np.asarray(
                    RAW_LOGIT_SCORE_REPRESENTATION
                ),
                "probability_dtype": np.asarray(PROBABILITY_DTYPE),
                "logit_dtype": np.asarray(RAW_LOGIT_DTYPE),
                "probability_transform": np.asarray("sigmoid"),
                "probability_clipping": np.asarray("none"),
                "inference_autocast_enabled": np.asarray(False),
            }
        )
    np.savez_compressed(path, **arrays)
    return path


@torch.inference_mode()
def export_dataset_score_maps(
    model: torch.nn.Module,
    dataset: Dataset,
    output_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 1,
    num_workers: int = 0,
    overwrite: bool = False,
    warm_flag: bool = True,
    labels_loaded: bool | None = None,
    manifest_metadata: Mapping[str, Any] | None = None,
    export_raw_logits: bool = False,
) -> dict[str, Any]:
    """Run inference, crop native padding, and save NPZ records plus manifest."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    if not isinstance(export_raw_logits, bool):
        raise TypeError("export_raw_logits must be boolean")
    if len(dataset) == 0:
        raise ValueError("dataset is empty")
    if labels_loaded is None:
        labels_loaded = bool(getattr(dataset, "load_masks", True))
    if not isinstance(labels_loaded, bool):
        raise TypeError("labels_loaded must be boolean")
    dataset_load_masks = getattr(dataset, "load_masks", None)
    if dataset_load_masks is not None and bool(dataset_load_masks) != labels_loaded:
        raise ValueError("labels_loaded contradicts dataset.load_masks")
    spatial_mode = getattr(dataset, "spatial_mode", None)
    if spatial_mode == "native" and batch_size != 1:
        raise ValueError("native spatial mode requires batch_size=1 for variable image sizes")

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    existing = list(destination.glob("*.npz"))
    manifest_path = destination / "manifest.json"
    if (existing or manifest_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Output directory already contains score records: {destination}; use overwrite=True"
        )
    if overwrite:
        for path in existing:
            path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()
    base_manifest_fields = {
        "schema_version",
        "record_integrity_schema",
        "score_type",
        "warm_flag",
        "labels_loaded",
        "num_images",
        "records",
        "records_sha256",
        "ordered_image_ids_sha256",
        "target_dataset",
        "split_file",
        "split_file_sha256",
        "split_ordered_ids_sha256",
        "requested_split",
        "split_role",
        "split_authority_verified",
        "spatial_mode",
        "base_hw",
        "pad_multiple",
        "mask_alignment_policy",
        "mask_alignment_schema",
        "mask_alignment_count",
        "mask_aligned_sample_ids",
        "score_representation",
        "probability_dtype",
        "logit_dtype",
        "probability_transform",
        "probability_clipping",
        "inference_autocast_enabled",
    }
    if manifest_metadata:
        reserved = base_manifest_fields.intersection(manifest_metadata)
        if reserved:
            raise ValueError(
                f"manifest_metadata cannot override reserved fields: {', '.join(sorted(reserved))}"
            )

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model = model.to(torch_device)
    if export_raw_logits:
        # Formal raw-logit evidence must be generated in one unambiguous FP32
        # domain.  ``enabled=False`` also defeats an ambient autocast context
        # held by a library caller; simply casting a half-precision result after
        # the forward pass would not recover the lost ranking information.
        model = model.float()
    was_training = model.training
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
    )
    records: list[dict[str, Any]] = []
    used_filenames: set[str] = set()
    for batch in loader:
        if not isinstance(batch, Mapping) or not {
            "image",
            "gray",
            "meta",
        }.issubset(batch):
            raise ValueError(
                "evaluation dataset batches must contain image, gray, and meta"
            )
        if labels_loaded and "mask" not in batch:
            raise ValueError("labels_loaded=True requires a mask in every batch")
        if not labels_loaded and "mask" in batch:
            raise ValueError(
                "labels_loaded=False requires a dataset that does not load masks"
            )
        images = batch["image"].to(torch_device, non_blocking=True)
        grays = batch["gray"]
        masks = batch.get("mask")
        logits_cpu: torch.Tensor | None = None
        if export_raw_logits:
            images = images.float()
            with torch.autocast(device_type=torch_device.type, enabled=False):
                logits = _forward_model(model, images, warm_flag=warm_flag).float()
                probabilities = torch.sigmoid(logits).float()
            logits_cpu = logits.cpu()
            probabilities = probabilities.cpu()
        else:
            probabilities = torch.sigmoid(
                _forward_model(model, images, warm_flag=warm_flag)
            ).cpu()
        batch_count = int(images.shape[0])
        for index in range(batch_count):
            meta = _unbatch_meta(batch["meta"], index, batch_count)
            probability = crop_to_valid(probabilities[index, 0], meta).numpy()
            raw_logits = (
                crop_to_valid(logits_cpu[index, 0], meta).numpy()
                if logits_cpu is not None
                else None
            )
            gray = crop_to_valid(grays[index, 0], meta).numpy()
            mask = (
                crop_to_valid(masks[index, 0], meta).numpy()
                if masks is not None
                else None
            )
            image_id = str(meta["image_id"])
            filename = _safe_record_name(image_id)
            if filename in used_filenames:
                raise ValueError(f"Duplicate output filename generated for image id {image_id!r}")
            used_filenames.add(filename)
            record_path = save_score_map(
                destination / filename,
                probability=probability,
                gray=gray,
                mask=mask,
                meta=meta,
                labels_loaded=labels_loaded,
                raw_logits=raw_logits,
            )
            json_meta = meta_to_jsonable(meta)
            record: dict[str, Any] = {
                "image_id": image_id,
                "file": filename,
                "shape": [int(probability.shape[0]), int(probability.shape[1])],
                "sha256": file_sha256(record_path),
                "mask_alignment_applied": bool(
                    json_meta["mask_alignment_applied"]
                ),
                "mask_original_hw": [
                    int(value) for value in json_meta["mask_original_hw"]
                ],
                "mask_aspect_relative_error": float(
                    json_meta["mask_aspect_relative_error"]
                ),
            }
            if export_raw_logits:
                record.update(
                    {
                        "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
                        "probability_dtype": PROBABILITY_DTYPE,
                        "logit_dtype": RAW_LOGIT_DTYPE,
                        "probability_transform": "sigmoid",
                        "probability_clipping": "none",
                        "inference_autocast_enabled": False,
                    }
                )
            records.append(record)
    if was_training:
        model.train()

    if not records:
        raise RuntimeError("score-map export produced no records")
    ordered_ids = [str(record["image_id"]) for record in records]
    dataset_ids = getattr(dataset, "image_ids", None)
    if dataset_ids is not None and ordered_ids != [str(value) for value in dataset_ids]:
        raise ValueError("Export record order differs from the dataset's declared image order")

    manifest: dict[str, Any] = {
        "schema_version": SCORE_MANIFEST_SCHEMA_VERSION,
        "record_integrity_schema": SCORE_RECORD_INTEGRITY_SCHEMA,
        "score_type": "sigmoid_probability",
        "warm_flag": bool(warm_flag),
        "labels_loaded": labels_loaded,
        "num_images": len(records),
        "records": records,
        "records_sha256": score_records_sha256(records),
        "ordered_image_ids_sha256": ordered_ids_sha256(ordered_ids),
        "mask_alignment_schema": SCORE_MASK_ALIGNMENT_SCHEMA,
        "mask_alignment_policy": (
            MASK_ALIGNMENT_POLICY
            if labels_loaded
            else MASK_ALIGNMENT_NOT_LOADED_POLICY
        ),
        "mask_alignment_count": int(
            sum(bool(record["mask_alignment_applied"]) for record in records)
        ),
        "mask_aligned_sample_ids": [
            str(record["image_id"])
            for record in records
            if bool(record["mask_alignment_applied"])
        ],
    }
    if export_raw_logits:
        manifest.update(
            {
                "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
                "probability_dtype": PROBABILITY_DTYPE,
                "logit_dtype": RAW_LOGIT_DTYPE,
                "probability_transform": "sigmoid",
                "probability_clipping": "none",
                "inference_autocast_enabled": False,
            }
        )
    if hasattr(dataset, "dataset_name"):
        manifest["target_dataset"] = str(getattr(dataset, "dataset_name"))
    if hasattr(dataset, "split_file"):
        split_path = Path(getattr(dataset, "split_file")).expanduser().resolve()
        if not split_path.is_file():
            raise FileNotFoundError(f"Dataset split file disappeared during export: {split_path}")
        manifest["split_file"] = str(split_path)
        manifest["split_file_sha256"] = file_sha256(split_path)
        manifest["split_ordered_ids_sha256"] = ordered_ids_sha256(ordered_ids)
    if hasattr(dataset, "requested_split"):
        manifest["requested_split"] = str(getattr(dataset, "requested_split"))
    if hasattr(dataset, "split_role"):
        manifest["split_role"] = str(getattr(dataset, "split_role"))
    if hasattr(dataset, "split_authority_verified"):
        manifest["split_authority_verified"] = bool(
            getattr(dataset, "split_authority_verified")
        )
    if hasattr(dataset, "spatial_mode"):
        manifest["spatial_mode"] = str(getattr(dataset, "spatial_mode"))
    if hasattr(dataset, "base_hw"):
        manifest["base_hw"] = [int(value) for value in getattr(dataset, "base_hw")]
    if hasattr(dataset, "pad_multiple"):
        manifest["pad_multiple"] = int(getattr(dataset, "pad_multiple"))
    if manifest_metadata:
        manifest.update(dict(manifest_metadata))
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _checkpoint_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("net", "state_dict", "model_state_dict"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                payload = candidate
                break
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("checkpoint does not contain a state dictionary")
    if not all(isinstance(value, torch.Tensor) for value in payload.values()):
        raise ValueError("resolved checkpoint state dictionary contains non-tensor values")
    return {
        (key[len("module.") :] if str(key).startswith("module.") else str(key)): value
        for key, value in payload.items()
    }


def _checkpoint_metadata(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    metadata: dict[str, Any] = {}
    for key in (
        "epoch",
        "warm_flag",
        "inference_head",
        "selection_rule",
        "checkpoint_selection",
        "diagnostic_only",
        "formal_paper_checkpoint",
    ):
        if key in payload:
            metadata[key] = payload[key]
    config = payload.get("config")
    if isinstance(config, Mapping):
        metadata["config"] = dict(config)
    return metadata


def resolve_checkpoint_warm_flag(
    requested: bool | None,
    checkpoint_metadata: Mapping[str, Any],
) -> bool:
    """Resolve the MSHNet head without silently contradicting its checkpoint."""

    saved = checkpoint_metadata.get("warm_flag")
    if requested is None:
        if saved is None:
            raise ValueError(
                "checkpoint has no warm_flag metadata; pass --warm-flag for a "
                "trained multi-scale head or --no-warm-flag for a warm-stage head"
            )
        return bool(saved)
    resolved = bool(requested)
    if saved is not None and bool(saved) != resolved:
        raise ValueError(
            "requested inference head contradicts checkpoint metadata: "
            f"requested warm_flag={resolved}, checkpoint warm_flag={bool(saved)}"
        )
    return resolved


def _load_checkpoint_safely(path: Path) -> Any:
    """Load legacy MSHNet checkpoints without enabling arbitrary pickle globals."""

    safe_globals: list[Any] = []
    for namespace_name in ("core", "_core"):
        numpy_core = getattr(np, namespace_name, None)
        numpy_multiarray = getattr(numpy_core, "multiarray", None)
        numpy_scalar = getattr(numpy_multiarray, "scalar", None)
        if numpy_scalar is not None:
            safe_globals.append(
                (numpy_scalar, f"numpy.{namespace_name}.multiarray.scalar")
            )
    safe_globals.append((np.dtype, "numpy.dtype"))
    numpy_dtypes = getattr(np, "dtypes", None)
    if numpy_dtypes is not None and hasattr(numpy_dtypes, "Float64DType"):
        safe_globals.append(numpy_dtypes.Float64DType)
    with torch.serialization.safe_globals(safe_globals):
        return torch.load(path, map_location="cpu", weights_only=True)


def load_mshnet(
    weight_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> torch.nn.Module:
    """Instantiate the repository MSHNet and load a state/checkpoint file."""

    path = Path(weight_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Weight file does not exist: {path}")
    from model.MSHNet import MSHNet

    model = MSHNet(3)
    payload = _load_checkpoint_safely(path)
    model.load_state_dict(_checkpoint_state_dict(payload), strict=strict)
    model._rc_irstd_checkpoint_metadata = _checkpoint_metadata(payload)
    return model.to(torch.device(device))


def _parse_base_size(value: str) -> int | tuple[int, int]:
    normalized = value.lower().replace("×", "x")
    try:
        if "x" in normalized:
            height, width = normalized.split("x", maxsplit=1)
            return int(height), int(width)
        return int(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError("base size must be INT or HEIGHTxWIDTH") from error


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split-file")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--weight-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source-dataset",
        action="append",
        help="Detector training domain; repeat for multi-source/LODO provenance",
    )
    parser.add_argument(
        "--spatial-mode",
        choices=("resize", "native"),
        default="native",
        help="Native resolution is the paper/deployment default; resize is diagnostic only",
    )
    parser.add_argument("--base-size", type=_parse_base_size, default=256)
    parser.add_argument("--pad-multiple", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--labels-loaded",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Load and embed masks (default). Use --no-labels-loaded for a "
            "genuinely mask-free zero-label export; masks/ is then neither "
            "required nor opened."
        ),
    )
    parser.add_argument(
        "--warm-flag",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Select MSHNet head; default auto-reads checkpoint metadata and "
            "requires an explicit flag for legacy raw weights"
        ),
    )
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument(
        "--export-raw-logits",
        "--raw-logits",
        dest="export_raw_logits",
        action="store_true",
        help=(
            "Also export native raw logits in float32. This precision-audit "
            "mode explicitly disables autocast and records the representation "
            "and dtypes in both manifest and NPZ files."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    device = _resolve_device(args.device)
    dataset = IRSTDEvalDataset(
        args.dataset_dir,
        split_file=args.split_file,
        split=args.split,
        spatial_mode=args.spatial_mode,
        base_size=args.base_size,
        pad_multiple=args.pad_multiple,
        load_masks=args.labels_loaded,
    )
    model = load_mshnet(args.weight_path, device=device, strict=not args.non_strict)
    checkpoint_metadata = getattr(model, "_rc_irstd_checkpoint_metadata", {})
    warm_flag = resolve_checkpoint_warm_flag(args.warm_flag, checkpoint_metadata)
    metadata: dict[str, Any] = {
        "weight_path": str(Path(args.weight_path).expanduser().resolve()),
        "weight_sha256": hashlib.sha256(
            Path(args.weight_path).expanduser().read_bytes()
        ).hexdigest(),
        "checkpoint_epoch": checkpoint_metadata.get("epoch"),
        "checkpoint_warm_flag": checkpoint_metadata.get("warm_flag"),
        "checkpoint_inference_head": checkpoint_metadata.get("inference_head"),
        "checkpoint_selection_rule": checkpoint_metadata.get("selection_rule"),
    }
    checkpoint_config = checkpoint_metadata.get("config")
    checkpoint_sources = None
    if isinstance(checkpoint_config, Mapping):
        saved_names = checkpoint_config.get("domain_names")
        if isinstance(saved_names, (list, tuple)) and all(
            isinstance(value, str) and value for value in saved_names
        ):
            checkpoint_sources = [str(value) for value in saved_names]
    if args.source_dataset:
        source_datasets = [str(value) for value in args.source_dataset]
        if len(set(source_datasets)) != len(source_datasets):
            raise ValueError("--source-dataset values must be unique")
        if checkpoint_sources is not None and source_datasets != checkpoint_sources:
            raise ValueError(
                "--source-dataset does not match checkpoint config: "
                f"requested={source_datasets}, checkpoint={checkpoint_sources}"
            )
        metadata["source_datasets"] = source_datasets
        if len(source_datasets) == 1:
            metadata["source_dataset"] = source_datasets[0]
    elif checkpoint_sources is not None:
        metadata["source_datasets"] = checkpoint_sources
        if len(checkpoint_sources) == 1:
            metadata["source_dataset"] = checkpoint_sources[0]
    export_dataset_score_maps(
        model,
        dataset,
        args.output_dir,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        warm_flag=warm_flag,
        labels_loaded=args.labels_loaded,
        manifest_metadata=metadata,
        export_raw_logits=args.export_raw_logits,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "export_dataset_score_maps",
    "load_mshnet",
    "resolve_checkpoint_warm_flag",
    "save_score_map",
]
