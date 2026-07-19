"""Frozen source-only RC-MSHNet component-fusion counterfactual diagnosis.

The two fixed-last full source-LODO checkpoints are replayed in FP32 eval
inference mode. The baseline logits are captured directly from the
fusion_head.forward keyword arguments, never inferred by subtraction. This
diagnostic cannot authorize an outer-target run.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_ext.dataset_meta import crop_to_valid, meta_to_jsonable
from data_ext.eval_dataset import IRSTDEvalDataset
from evaluation.artifact_integrity import (
    ordered_ids_sha256,
    score_records_sha256,
    verify_score_map_directory,
)
from evaluation.export_score_maps import _load_checkpoint_safely
from evaluation.raw_logit_oracle import (
    RawLogitSample,
    raw_logit_stream_sha256,
    validate_formal_raw_logit_manifest,
)
from rc_irstd.models import build_mshnet


SOURCE_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts/aaai27/audit/phase3_source_lodo_gate"
    / "tier2r_component_counterfactual_v1"
)
ARCHITECTURE_VERSION = "rc-mshnet-v1"
SCHEMA_VERSION = "rc-mshnet-frozen-component-counterfactual-v1"
SCORE_REPRESENTATION = "fp32_raw_logits_no_autocast"

FROZEN_FULL_FOLDS: dict[str, dict[str, str]] = {
    "heldout_nudt": {
        "held_out_source": "NUDT-SIRST",
        "training_source": "IRSTD-1K",
    },
    "heldout_irstd": {
        "held_out_source": "IRSTD-1K",
        "training_source": "NUDT-SIRST",
    },
}

COUNTERFACTUAL_VARIANTS = (
    "drop_component_action_fixed_full_gate",
    "drop_component_action_conditional_full_gate",
    "context_preserved_component_expert_off_reforward",
    "context_zero_component_expert_off_reforward",
    "independently_trained_v1_no_component",
)
BATCH_RAW_LOGIT_KEYS = (
    "frozen_full",
    "replayed_full",
    *COUNTERFACTUAL_VARIANTS[:-1],
)


@dataclass(frozen=True)
class FrozenCheckpointBinding:
    fold: str
    checkpoint_path: Path
    checkpoint_sha256: str
    identity_path: Path
    identity_sha256: str
    held_out_source: str
    training_source: str

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "identity_path": str(self.identity_path),
            "identity_sha256": self.identity_sha256,
            "held_out_source": self.held_out_source,
            "training_source": self.training_source,
        }


@dataclass(frozen=True)
class ComponentCounterfactualBatch:
    raw_logits: Mapping[str, torch.Tensor]
    base_raw_logits: torch.Tensor
    full_gate_logits: torch.Tensor
    full_gates: torch.Tensor
    conditional_full_gates: torch.Tensor
    context_preserved_expert_off_gates: torch.Tensor
    context_zero_expert_off_gates: torch.Tensor
    contrast_delta: torch.Tensor
    component_delta: torch.Tensor
    component_contribution: torch.Tensor
    residual_gain: torch.Tensor
    replay_consistency: Mapping[str, float]
    base_logits_capture_source: str = "fusion_head.forward_kwargs.base_logits"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _lexical_absolute(path: str | Path, *, name: str) -> Path:
    raw = Path(path).expanduser()
    if ".." in raw.parts:
        raise RuntimeError(f"{name} contains forbidden parent traversal")
    return Path(os.path.abspath(raw))


def _contains_outer_target(path: str | Path) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", str(path).lower())
    return "nuaa" in normalized


def _assert_no_symlink_components(
    path: Path,
    *,
    anchor: Path,
    name: str,
) -> None:
    path = _lexical_absolute(path, name=name)
    anchor = _lexical_absolute(anchor, name=f"{name}_anchor")
    if path != anchor and anchor not in path.parents:
        raise RuntimeError(f"{name} escapes its allowlisted root: {path}")
    candidates = [anchor]
    current = anchor
    if path != anchor:
        for part in path.relative_to(anchor).parts:
            current = current / part
            candidates.append(current)
    for candidate in candidates:
        if candidate.is_symlink():
            raise RuntimeError(f"{name} contains a symlink: {candidate}")


def validate_source_dataset_path(
    path: str | Path,
    *,
    expected_dataset: str,
    source_root: str | Path = SOURCE_ROOT,
) -> Path:
    allowed = {value["held_out_source"] for value in FROZEN_FULL_FOLDS.values()}
    if expected_dataset not in allowed:
        raise ValueError(f"Unsupported source dataset: {expected_dataset}")
    root = _lexical_absolute(source_root, name="source_root")
    candidate = _lexical_absolute(path, name="source_dataset")
    expected = root / expected_dataset
    if candidate != expected:
        raise RuntimeError(
            f"Source dataset must use the allowlisted path: {expected}"
        )
    if _contains_outer_target(candidate):
        raise RuntimeError("Outer-target data is forbidden")
    _assert_no_symlink_components(
        candidate,
        anchor=root,
        name="source_dataset",
    )
    if not candidate.is_dir() or candidate.resolve() != candidate:
        raise RuntimeError(
            f"Source dataset is missing or resolves through an alias: {candidate}"
        )
    return candidate


def _validate_source_record_path(
    path: str | Path,
    *,
    dataset_root: Path,
) -> Path:
    candidate = _lexical_absolute(path, name="source_record")
    if _contains_outer_target(candidate):
        raise RuntimeError("Outer-target record path is forbidden")
    _assert_no_symlink_components(
        candidate,
        anchor=dataset_root,
        name="source_record",
    )
    if not candidate.is_file() or candidate.resolve() != candidate:
        raise RuntimeError(
            f"Source record is missing or resolves through an alias: {candidate}"
        )
    return candidate


def _require_frozen_file(path: Path, *, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{name} must be a regular file: {path}")
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise RuntimeError(f"{name} is not frozen read-only: {path}")


def validate_two_fold_checkpoint_contract(
    checkpoints: Mapping[str, str | Path],
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, FrozenCheckpointBinding]:
    if set(checkpoints) != set(FROZEN_FULL_FOLDS):
        raise RuntimeError(
            "Exactly heldout_nudt and heldout_irstd checkpoints are required"
        )
    root = _lexical_absolute(project_root, name="project_root")
    if root.is_symlink() or root.resolve() != root:
        raise RuntimeError("project_root must be a canonical non-symlink path")
    bindings: dict[str, FrozenCheckpointBinding] = {}
    for fold, contract in FROZEN_FULL_FOLDS.items():
        checkpoint = _lexical_absolute(
            checkpoints[fold],
            name=f"{fold}_checkpoint",
        )
        expected = (
            root
            / "outputs/aaai27/detectors/source_lodo_gate/seed42/full"
            / fold
            / "last.pt"
        )
        if checkpoint != expected:
            raise RuntimeError(
                f"{fold} must use its canonical frozen checkpoint: {expected}"
            )
        if _contains_outer_target(checkpoint):
            raise RuntimeError("Outer-target checkpoint path is forbidden")
        _assert_no_symlink_components(
            checkpoint,
            anchor=root,
            name=f"{fold}_checkpoint",
        )
        identity_path = checkpoint.parent / "PHASE3_IDENTITY.json"
        sidecar_path = checkpoint.parent / "checkpoint.sha256"
        for item, name in (
            (checkpoint, f"{fold} checkpoint"),
            (identity_path, f"{fold} identity"),
            (sidecar_path, f"{fold} digest sidecar"),
        ):
            _require_frozen_file(item, name=name)
        checkpoint_sha = _sha256(checkpoint)
        fields = sidecar_path.read_text(encoding="ascii").strip().split()
        if fields != [checkpoint_sha, "last.pt"]:
            raise RuntimeError(f"{fold} checkpoint digest sidecar drifted")
        identity = _load_json(identity_path)
        expected_identity = {
            "run_id": f"full_{fold}",
            "role": "full",
            "held_out_pseudo_target": contract["held_out_source"],
            "training_source": contract["training_source"],
            "outer_target_labels_used": False,
            "checkpoint_selection": "fixed_last",
            "checkpoint_sha256": checkpoint_sha,
        }
        for key, expected_value in expected_identity.items():
            if identity.get(key) != expected_value:
                raise RuntimeError(f"{fold} identity mismatch at {key}")
        bindings[fold] = FrozenCheckpointBinding(
            fold=fold,
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            identity_path=identity_path,
            identity_sha256=_sha256(identity_path),
            held_out_source=contract["held_out_source"],
            training_source=contract["training_source"],
        )
    if len({item.checkpoint_sha256 for item in bindings.values()}) != 2:
        raise RuntimeError("The two folds bind identical checkpoint bytes")
    return bindings


def _normalize_state_dict(value: Any) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("checkpoint model_state must be a non-empty mapping")
    state: dict[str, torch.Tensor] = {}
    for raw_key, tensor in value.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("checkpoint model_state contains non-tensor values")
        key = str(raw_key)
        if key.startswith("module."):
            key = key[7:]
        if key in state:
            raise ValueError("checkpoint key normalization produced duplicates")
        state[key] = tensor
    return state


def load_frozen_full_model(
    binding: FrozenCheckpointBinding,
    *,
    device: str | torch.device,
) -> nn.Module:
    payload = _load_checkpoint_safely(binding.checkpoint_path)
    if not isinstance(payload, Mapping):
        raise ValueError("trainer checkpoint must be a mapping")
    expected_metadata = {
        "kind": "detector",
        "checkpoint_selection": "fixed_last",
        "selection_rule": "fixed_last",
        "test_labels_used_for_selection": False,
        "diagnostic_test_eval": False,
        "diagnostic_only": False,
        "formal_paper_checkpoint": True,
        "warm_flag": True,
        "inference_head": "multi_scale_fused",
    }
    for key, expected in expected_metadata.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"Frozen checkpoint metadata mismatch at {key}")
    if payload.get("source_names") != [binding.training_source]:
        raise RuntimeError("Checkpoint training source conflicts with identity")
    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise RuntimeError("Frozen checkpoint lacks embedded model_config")
    full_contract = {
        "architecture_version": ARCHITECTURE_VERSION,
        "backend": "rc_mshnet",
        "use_contrast": True,
        "use_component_context": True,
        "use_risk_gate": True,
    }
    for key, expected in full_contract.items():
        actual = model_config.get(key)
        if actual != expected or (
            isinstance(expected, bool) and type(actual) is not bool
        ):
            raise RuntimeError(
                f"Checkpoint is not v1 full at model_config.{key}"
            )
    if "use_component_expert" in model_config:
        raise RuntimeError(
            "Legacy v1 model_config must not contain use_component_expert"
        )
    model = build_mshnet(dict(model_config))
    model.load_state_dict(
        _normalize_state_dict(payload.get("model_state")),
        strict=True,
    )
    model.to(torch.device(device), dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    return model


@dataclass(frozen=True)
class FrozenRawLogitRecord:
    image_id: str
    path: Path


@dataclass(frozen=True)
class FrozenFoldComparators:
    full_records: tuple[FrozenRawLogitRecord, ...]
    independently_trained_no_component_records: tuple[FrozenRawLogitRecord, ...]
    evidence: Mapping[str, Any]


def _validate_legacy_no_component_checkpoint(
    binding: FrozenCheckpointBinding,
) -> tuple[Path, str, str]:
    seed_root = binding.checkpoint_path.parents[2]
    run_root = seed_root / "no_component" / binding.fold
    checkpoint = run_root / "last.pt"
    identity_path = run_root / "PHASE3_IDENTITY.json"
    sidecar_path = run_root / "checkpoint.sha256"
    for path, name in (
        (checkpoint, "no-component checkpoint"),
        (identity_path, "no-component identity"),
        (sidecar_path, "no-component digest sidecar"),
    ):
        _assert_no_symlink_components(
            path,
            anchor=PROJECT_ROOT,
            name=name.replace(" ", "_"),
        )
        _require_frozen_file(path, name=name)
    checkpoint_sha = _sha256(checkpoint)
    if sidecar_path.read_text(encoding="ascii").strip().split() != [
        checkpoint_sha,
        "last.pt",
    ]:
        raise RuntimeError("No-component checkpoint digest sidecar drifted")
    identity = _load_json(identity_path)
    expected_identity = {
        "run_id": f"no_component_{binding.fold}",
        "role": "no_component",
        "held_out_pseudo_target": binding.held_out_source,
        "training_source": binding.training_source,
        "outer_target_labels_used": False,
        "checkpoint_selection": "fixed_last",
        "checkpoint_sha256": checkpoint_sha,
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise RuntimeError(f"No-component identity mismatch at {key}")

    payload = _load_checkpoint_safely(checkpoint)
    if not isinstance(payload, Mapping):
        raise RuntimeError("No-component checkpoint must be a mapping")
    expected_metadata = {
        "kind": "detector",
        "checkpoint_selection": "fixed_last",
        "selection_rule": "fixed_last",
        "test_labels_used_for_selection": False,
        "diagnostic_test_eval": False,
        "diagnostic_only": False,
        "formal_paper_checkpoint": True,
        "warm_flag": True,
        "inference_head": "multi_scale_fused",
    }
    for key, expected in expected_metadata.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"No-component checkpoint metadata mismatch at {key}"
            )
    if payload.get("source_names") != [binding.training_source]:
        raise RuntimeError("No-component checkpoint source mismatch")
    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise RuntimeError("No-component checkpoint lacks model_config")
    expected_model = {
        "architecture_version": ARCHITECTURE_VERSION,
        "backend": "rc_mshnet",
        "use_contrast": True,
        "use_component_context": False,
        "use_risk_gate": True,
    }
    for key, expected in expected_model.items():
        actual = model_config.get(key)
        if actual != expected or (
            isinstance(expected, bool) and type(actual) is not bool
        ):
            raise RuntimeError(
                f"No-component v1 model identity mismatch at {key}"
            )
    if "use_component_expert" in model_config:
        raise RuntimeError(
            "Legacy no-component v1 config must not contain expert key"
        )
    return checkpoint, checkpoint_sha, _sha256(identity_path)


def _load_frozen_raw_logit_artifact(
    score_dir: Path,
    *,
    binding: FrozenCheckpointBinding,
    checkpoint_sha256: str,
) -> tuple[
    tuple[FrozenRawLogitRecord, ...],
    dict[str, Any],
]:
    _assert_no_symlink_components(
        score_dir,
        anchor=PROJECT_ROOT,
        name="frozen_score_directory",
    )
    if _contains_outer_target(score_dir):
        raise RuntimeError("Outer-target score artifact is forbidden")
    manifest, paths, integrity = verify_score_map_directory(
        score_dir,
        require_integrity=True,
        require_masks=True,
    )
    contract = validate_formal_raw_logit_manifest(
        manifest,
        integrity,
        expected_split_role="train",
    )
    assert manifest is not None
    expected_contract = {
        "target_dataset": binding.held_out_source,
        "source_datasets": [binding.training_source],
        "detector_weight_sha256": checkpoint_sha256,
        "requested_split": "train",
        "split_role": "train",
        "score_representation": (
            "raw_logit_float32+sigmoid_probability_float32"
        ),
        "logit_dtype": "float32",
        "inference_autocast_enabled": False,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"Frozen score contract mismatch at {key}")
    records: list[FrozenRawLogitRecord] = []
    stream_hasher = _RawLogitStreamHasher()
    for index, (record, path) in enumerate(
        zip(manifest["records"], paths)
    ):
        image_id = str(record["image_id"])
        with np.load(path, allow_pickle=False) as payload:
            embedded_image_id = str(
                np.asarray(payload["image_id"]).item()
            )
            if embedded_image_id != image_id:
                raise RuntimeError(
                    "Frozen raw-logit image_id mismatch at "
                    f"record {index}"
                )
            logits = np.asarray(payload["logit"])
            stream_hasher.update(image_id, logits)
        records.append(
            FrozenRawLogitRecord(image_id=image_id, path=path)
        )
    if len(records) != len(paths):
        raise AssertionError("Frozen raw-logit record binding lost records")
    evidence = {
        "score_directory": str(score_dir),
        "manifest_sha256": integrity["manifest_sha256"],
        "records_sha256": integrity["records_sha256"],
        "ordered_image_ids_sha256": integrity[
            "ordered_image_ids_sha256"
        ],
        "num_records": integrity["num_records"],
        "raw_logit_stream_sha256": stream_hasher.hexdigest(),
        "checkpoint_sha256": checkpoint_sha256,
    }
    return tuple(records), evidence


def load_frozen_fold_comparators(
    binding: FrozenCheckpointBinding,
) -> FrozenFoldComparators:
    full_score_dir = binding.checkpoint_path.parent / "scores_heldout_train"
    no_component_checkpoint, no_component_sha, no_component_identity_sha = (
        _validate_legacy_no_component_checkpoint(binding)
    )
    no_component_score_dir = (
        no_component_checkpoint.parent / "scores_heldout_train"
    )
    full_records, full_evidence = _load_frozen_raw_logit_artifact(
        full_score_dir,
        binding=binding,
        checkpoint_sha256=binding.checkpoint_sha256,
    )
    no_component_records, no_component_evidence = (
        _load_frozen_raw_logit_artifact(
            no_component_score_dir,
            binding=binding,
            checkpoint_sha256=no_component_sha,
        )
    )
    full_ids = [record.image_id for record in full_records]
    no_component_ids = [
        record.image_id for record in no_component_records
    ]
    if full_ids != no_component_ids:
        raise RuntimeError(
            "Full and no-component comparator stream orders differ"
        )
    return FrozenFoldComparators(
        full_records=full_records,
        independently_trained_no_component_records=no_component_records,
        evidence={
            "frozen_full": full_evidence,
            "independently_trained_v1_no_component": {
                **no_component_evidence,
                "identity_sha256": no_component_identity_sha,
            },
            "ordered_streams_aligned": True,
        },
    )


def _autocast_enabled(device_type: str) -> bool:
    try:
        return bool(torch.is_autocast_enabled(device_type))
    except TypeError:  # pragma: no cover
        return bool(torch.is_autocast_enabled())


def _require_fp32(
    value: Any,
    *,
    name: str,
    ndim: int | None = 4,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} has invalid shape: {tuple(value.shape)}")
    if value.dtype != torch.float32:
        raise RuntimeError(f"{name} must be FP32, got {value.dtype}")
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} contains NaN or infinity")
    return value


def _capture(value: Any, *, name: str) -> torch.Tensor:
    return _require_fp32(value, name=name).detach().clone()


def _validate_gate(
    gates: torch.Tensor,
    *,
    reference: torch.Tensor,
    name: str,
    component_off: bool,
    atol: float,
) -> None:
    _require_fp32(gates, name=name)
    expected_shape = (reference.shape[0], 3, *reference.shape[-2:])
    if tuple(gates.shape) != expected_shape:
        raise RuntimeError(f"{name} shape mismatch")
    if bool(torch.any(gates < 0.0)) or bool(torch.any(gates > 1.0)):
        raise RuntimeError(f"{name} lies outside [0, 1]")
    error = float((gates.sum(dim=1, keepdim=True) - 1.0).abs().max())
    if error > atol:
        raise RuntimeError(f"{name} does not sum to one: {error}")
    if component_off and float(gates[:, 1:2].abs().max()) > atol:
        raise RuntimeError(f"{name} failed to mask component expert")


def assert_full_raw_logit_replay_consistency(
    frozen_full: torch.Tensor,
    replayed_full: torch.Tensor,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-7,
) -> dict[str, float]:
    _require_fp32(frozen_full, name="frozen_full")
    _require_fp32(replayed_full, name="replayed_full")
    if frozen_full.shape != replayed_full.shape:
        raise RuntimeError("Full raw-logit replay shape mismatch")
    difference = (frozen_full - replayed_full).abs()
    maximum = float(difference.max())
    mean = float(difference.mean())
    if not torch.allclose(frozen_full, replayed_full, rtol=rtol, atol=atol):
        raise RuntimeError(
            "Frozen full raw-logit replay is inconsistent: "
            f"max_abs_error={maximum}, mean_abs_error={mean}"
        )
    return {"max_abs_error": maximum, "mean_abs_error": mean}


def assert_historical_full_raw_logit_consistency(
    replayed_full: np.ndarray,
    historical_full: np.ndarray,
    *,
    image_id: str,
) -> dict[str, float | bool]:
    """Require bitwise equality to the frozen pre-diagnosis full export."""

    replayed = np.asarray(replayed_full, dtype=np.float32)
    historical = np.asarray(historical_full)
    if historical.dtype != np.float32:
        raise RuntimeError(
            f"Historical full logit is not FP32 for {image_id}"
        )
    if replayed.shape != historical.shape:
        raise RuntimeError(
            f"Historical full logit shape mismatch for {image_id}"
        )
    difference = np.abs(replayed - historical)
    maximum = float(difference.max(initial=0.0))
    mean = float(difference.mean()) if difference.size else 0.0
    if not np.array_equal(replayed, historical):
        differing = int(np.count_nonzero(replayed != historical))
        raise RuntimeError(
            "Re-forward full raw logits differ from frozen artifact for "
            f"{image_id}: differing_pixels={differing}, "
            f"max_abs_error={maximum}"
        )
    return {
        "bitwise_equal": True,
        "max_abs_error": maximum,
        "mean_abs_error": mean,
    }


def _model_logits(output: Any) -> torch.Tensor:
    return _require_fp32(
        getattr(output, "logits", output),
        name="model_output_logits",
    )


def _validate_full_model(model: nn.Module) -> None:
    for name in ("use_contrast", "use_component_context", "use_risk_gate"):
        if getattr(model, name, None) is not True:
            raise RuntimeError(f"Frozen full diagnosis requires {name}=True")
    if (
        hasattr(model, "use_component_expert")
        and model.use_component_expert is not True
    ):
        raise RuntimeError("Frozen full diagnosis requires component expert")
    if getattr(model, "architecture_version", None) != ARCHITECTURE_VERSION:
        raise RuntimeError("Frozen diagnosis requires rc-mshnet-v1")
    if not isinstance(getattr(model, "fusion_head", None), nn.Module):
        raise TypeError("model.fusion_head must be a module")
    if not isinstance(getattr(model.fusion_head, "gate", None), nn.Module):
        raise TypeError("model.fusion_head.gate must be a module")


def diagnose_frozen_full_batch(
    model: nn.Module,
    images: torch.Tensor,
    *,
    warm_flag: bool = True,
    rtol: float = 1e-6,
    atol: float = 1e-7,
    gate_atol: float = 2e-6,
) -> ComponentCounterfactualBatch:
    """Build corrected counterfactual logits for one frozen full batch."""

    if not isinstance(warm_flag, bool):
        raise TypeError("warm_flag must be boolean")
    if rtol < 0.0 or atol < 0.0 or gate_atol <= 0.0:
        raise ValueError("comparison tolerances are invalid")
    _validate_full_model(model)
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise TypeError("images must be BxCxHxW")
    model.eval()
    model.float()
    model.requires_grad_(False)
    images = images.to(dtype=torch.float32)
    inputs: dict[str, torch.Tensor | bool] = {}
    outputs: dict[str, torch.Tensor] = {}

    def pre_hook(
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if inputs:
            raise RuntimeError("fusion_head was invoked more than once")
        if not torch.is_inference_mode_enabled():
            raise RuntimeError("Capture requires inference_mode")
        if _autocast_enabled(images.device.type):
            raise RuntimeError("Capture forbids autocast")
        if len(args) != 3:
            raise RuntimeError("fusion_head positional contract drifted")
        required = {
            "base_logits",
            "noise_proxy",
            "component_proxy",
            "contrast_enabled",
            "component_enabled",
        }
        if not required.issubset(kwargs):
            raise RuntimeError(
                f"fusion_head missing kwargs: {sorted(required - set(kwargs))}"
            )
        if (
            kwargs["contrast_enabled"] is not True
            or kwargs["component_enabled"] is not True
        ):
            raise RuntimeError("Captured action set is not frozen full")
        for name, value in zip(
            ("decoder_feature", "contrast_feature", "component_feature"),
            args,
        ):
            inputs[name] = _capture(value, name=name)
        for name in ("base_logits", "noise_proxy", "component_proxy"):
            inputs[name] = _capture(kwargs[name], name=name)
        inputs["contrast_enabled"] = True
        inputs["component_enabled"] = True

    def gate_hook(
        _module: nn.Module,
        _args: tuple[Any, ...],
        output: Any,
    ) -> None:
        if "full_gate_logits" in outputs:
            raise RuntimeError("gate was invoked more than once")
        outputs["full_gate_logits"] = _capture(
            output,
            name="full_gate_logits",
        )

    def output_hook(
        _module: nn.Module,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        if not isinstance(output, (tuple, list)) or len(output) != 4:
            raise RuntimeError("fusion_head output contract drifted")
        names = (
            "frozen_full",
            "contrast_delta",
            "component_delta",
            "full_gates",
        )
        for name, value in zip(names, output):
            outputs[name] = _capture(value, name=name)

    pre_handle = model.fusion_head.register_forward_pre_hook(
        pre_hook,
        with_kwargs=True,
    )
    gate_handle = model.fusion_head.gate.register_forward_hook(gate_hook)
    output_handle = model.fusion_head.register_forward_hook(
        output_hook,
        with_kwargs=True,
    )
    try:
        with torch.inference_mode(), torch.autocast(
            device_type=images.device.type,
            enabled=False,
        ):
            try:
                parameters = inspect.signature(model.forward).parameters
            except (TypeError, ValueError):
                parameters = {}
            output = (
                model(images, warm_flag=warm_flag)
                if "warm_flag" in parameters
                else model(images)
            )
            model_raw = _model_logits(output).detach().clone()
    finally:
        output_handle.remove()
        gate_handle.remove()
        pre_handle.remove()

    expected_inputs = {
        "decoder_feature",
        "contrast_feature",
        "component_feature",
        "base_logits",
        "noise_proxy",
        "component_proxy",
        "contrast_enabled",
        "component_enabled",
    }
    expected_outputs = {
        "frozen_full",
        "contrast_delta",
        "component_delta",
        "full_gates",
        "full_gate_logits",
    }
    if set(inputs) != expected_inputs or set(outputs) != expected_outputs:
        raise RuntimeError("Incomplete full fusion capture")

    frozen_full = outputs["frozen_full"]
    if not torch.equal(model_raw, frozen_full):
        raise RuntimeError("Captured fusion output differs from model logits")
    base = inputs["base_logits"]
    assert isinstance(base, torch.Tensor)
    contrast_delta = outputs["contrast_delta"]
    component_delta = outputs["component_delta"]
    full_gates = outputs["full_gates"]
    full_gate_logits = outputs["full_gate_logits"]
    if contrast_delta.shape != base.shape or component_delta.shape != base.shape:
        raise RuntimeError("Correction shape differs from direct base_logits")
    _validate_gate(
        full_gates,
        reference=base,
        name="full_gates",
        component_off=False,
        atol=gate_atol,
    )
    if full_gate_logits.shape != full_gates.shape:
        raise RuntimeError("Full gate-logit shape mismatch")
    if not torch.allclose(
        torch.softmax(full_gate_logits, dim=1),
        full_gates,
        rtol=1e-6,
        atol=gate_atol,
    ):
        raise RuntimeError("Full gates do not match captured gate logits")

    gain_parameter = getattr(model.fusion_head, "raw_residual_gain", None)
    if not isinstance(gain_parameter, torch.Tensor):
        raise RuntimeError("Missing raw_residual_gain")
    if gain_parameter.numel() != 1:
        raise RuntimeError("raw_residual_gain must be scalar")
    gain = (2.0 * torch.sigmoid(gain_parameter.detach().float())).reshape(())
    gc = full_gates[:, 0:1]
    gk = full_gates[:, 1:2]
    formula_replay = base + gain * (
        gc * contrast_delta + gk * component_delta
    )
    formula_check = assert_full_raw_logit_replay_consistency(
        frozen_full,
        formula_replay,
        rtol=rtol,
        atol=atol,
    )
    fixed_action_removal = base + gain * gc * contrast_delta

    masked_logits = full_gate_logits.clone()
    masked_logits[:, 1:2] = -torch.inf
    conditional_gates = torch.softmax(masked_logits, dim=1)
    _validate_gate(
        conditional_gates,
        reference=base,
        name="conditional_full_gates",
        component_off=True,
        atol=gate_atol,
    )
    conditional_renorm = (
        base + gain * conditional_gates[:, 0:1] * contrast_delta
    )

    decoder = inputs["decoder_feature"]
    contrast_feature = inputs["contrast_feature"]
    component_feature = inputs["component_feature"]
    noise_proxy = inputs["noise_proxy"]
    component_proxy = inputs["component_proxy"]
    assert isinstance(decoder, torch.Tensor)
    assert isinstance(contrast_feature, torch.Tensor)
    assert isinstance(component_feature, torch.Tensor)
    assert isinstance(noise_proxy, torch.Tensor)
    assert isinstance(component_proxy, torch.Tensor)

    def refusion(
        component_feature_value: torch.Tensor,
        component_proxy_value: torch.Tensor,
        *,
        component_enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        value = model.fusion_head(
            decoder,
            contrast_feature,
            component_feature_value,
            base_logits=base,
            noise_proxy=noise_proxy,
            component_proxy=component_proxy_value,
            contrast_enabled=True,
            component_enabled=component_enabled,
        )
        if not isinstance(value, (tuple, list)) or len(value) != 4:
            raise RuntimeError("Fusion re-forward contract drifted")
        tensors = tuple(
            _require_fp32(item, name=f"refusion[{index}]")
            for index, item in enumerate(value)
        )
        return tensors  # type: ignore[return-value]

    with torch.inference_mode(), torch.autocast(
        device_type=images.device.type,
        enabled=False,
    ):
        replayed = refusion(
            component_feature,
            component_proxy,
            component_enabled=True,
        )
        context_preserved = refusion(
            component_feature,
            component_proxy,
            component_enabled=False,
        )
        context_zero = refusion(
            torch.zeros_like(component_feature),
            torch.zeros_like(component_proxy),
            component_enabled=False,
        )

    replay_check = assert_full_raw_logit_replay_consistency(
        frozen_full,
        replayed[0],
        rtol=rtol,
        atol=atol,
    )
    preserved_gates = context_preserved[3].detach().clone()
    zero_gates = context_zero[3].detach().clone()
    _validate_gate(
        preserved_gates,
        reference=base,
        name="context_preserved_expert_off_gates",
        component_off=True,
        atol=gate_atol,
    )
    _validate_gate(
        zero_gates,
        reference=base,
        name="context_zero_expert_off_gates",
        component_off=True,
        atol=gate_atol,
    )
    if torch.count_nonzero(context_preserved[2]).item() != 0:
        raise RuntimeError("Preserved-context expert-off emitted component delta")
    if torch.count_nonzero(context_zero[2]).item() != 0:
        raise RuntimeError("Zero-context expert-off emitted component delta")

    raw_logits = {
        "frozen_full": frozen_full,
        "replayed_full": replayed[0].detach().clone(),
        "drop_component_action_fixed_full_gate": (
            fixed_action_removal.detach().clone()
        ),
        "drop_component_action_conditional_full_gate": (
            conditional_renorm.detach().clone()
        ),
        "context_preserved_component_expert_off_reforward": (
            context_preserved[0].detach().clone()
        ),
        "context_zero_component_expert_off_reforward": (
            context_zero[0].detach().clone()
        ),
    }
    if tuple(raw_logits) != BATCH_RAW_LOGIT_KEYS:
        raise AssertionError("Counterfactual naming contract drifted")
    for name, tensor in raw_logits.items():
        _require_fp32(tensor, name=f"raw_logits.{name}")
        if tensor.shape != base.shape:
            raise RuntimeError(f"raw_logits.{name} shape mismatch")
    return ComponentCounterfactualBatch(
        raw_logits=raw_logits,
        base_raw_logits=base,
        full_gate_logits=full_gate_logits,
        full_gates=full_gates,
        conditional_full_gates=conditional_gates.detach().clone(),
        context_preserved_expert_off_gates=preserved_gates,
        context_zero_expert_off_gates=zero_gates,
        contrast_delta=contrast_delta,
        component_delta=component_delta,
        component_contribution=(gain * gk * component_delta).detach().clone(),
        residual_gain=gain.detach().clone(),
        replay_consistency={
            **replay_check,
            "formula_max_abs_error": formula_check["max_abs_error"],
            "formula_mean_abs_error": formula_check["mean_abs_error"],
        },
    )


def _safe_record_name(image_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", image_id).strip("._")
    safe = safe or "sample"
    if safe != image_id:
        digest = hashlib.sha1(image_id.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe}-{digest}"
    return f"{safe}.npz"


class _DiagnosticAccumulator:
    def __init__(self) -> None:
        self.images = 0
        self.gt_pixels = 0
        self.background_pixels = 0
        self.component_gate_gt_sum = 0.0
        self.component_gate_background_sum = 0.0
        self.component_contribution_gt_sum = 0.0
        self.component_contribution_background_sum = 0.0
        self.gt_negative_count = 0
        self.background_positive_count = 0
        self.contrast_gate_before_sum = 0.0
        self.contrast_gate_after_sum = 0.0
        self.all_pixels = 0
        self.maximum_replay_error = 0.0

    def update(
        self,
        *,
        component_gate: np.ndarray,
        component_contribution: np.ndarray,
        contrast_gate_before: np.ndarray,
        contrast_gate_after: np.ndarray,
        mask: np.ndarray,
        replay_error: float,
    ) -> None:
        gt = np.asarray(mask, dtype=bool)
        background = ~gt
        arrays = (
            component_gate,
            component_contribution,
            contrast_gate_before,
            contrast_gate_after,
        )
        if any(np.asarray(value).shape != gt.shape for value in arrays):
            raise RuntimeError("Cropped diagnostic maps and mask differ")
        self.images += 1
        self.gt_pixels += int(gt.sum())
        self.background_pixels += int(background.sum())
        self.component_gate_gt_sum += float(component_gate[gt].sum())
        self.component_gate_background_sum += float(
            component_gate[background].sum()
        )
        self.component_contribution_gt_sum += float(
            component_contribution[gt].sum()
        )
        self.component_contribution_background_sum += float(
            component_contribution[background].sum()
        )
        self.gt_negative_count += int(
            (component_contribution[gt] < 0.0).sum()
        )
        self.background_positive_count += int(
            (component_contribution[background] > 0.0).sum()
        )
        self.contrast_gate_before_sum += float(contrast_gate_before.sum())
        self.contrast_gate_after_sum += float(contrast_gate_after.sum())
        self.all_pixels += int(gt.size)
        self.maximum_replay_error = max(
            self.maximum_replay_error,
            float(replay_error),
        )

    @staticmethod
    def _mean(total: float, count: int) -> float | None:
        return float(total / count) if count else None

    def summary(self) -> dict[str, Any]:
        return {
            "num_images": self.images,
            "gt_pixels": self.gt_pixels,
            "background_pixels": self.background_pixels,
            "component_gate_mean_gt": self._mean(
                self.component_gate_gt_sum,
                self.gt_pixels,
            ),
            "component_gate_mean_background": self._mean(
                self.component_gate_background_sum,
                self.background_pixels,
            ),
            "component_contribution_mean_gt": self._mean(
                self.component_contribution_gt_sum,
                self.gt_pixels,
            ),
            "component_contribution_mean_background": self._mean(
                self.component_contribution_background_sum,
                self.background_pixels,
            ),
            "gt_negative_contribution_fraction": self._mean(
                float(self.gt_negative_count),
                self.gt_pixels,
            ),
            "background_positive_contribution_fraction": self._mean(
                float(self.background_positive_count),
                self.background_pixels,
            ),
            "contrast_gate_before_renorm": self._mean(
                self.contrast_gate_before_sum,
                self.all_pixels,
            ),
            "contrast_gate_after_renorm": self._mean(
                self.contrast_gate_after_sum,
                self.all_pixels,
            ),
            "maximum_full_replay_abs_error": self.maximum_replay_error,
        }


def _crop_fp32(tensor: torch.Tensor, meta: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        crop_to_valid(tensor.detach().cpu(), meta),
        dtype=np.float32,
    )


class _RawLogitStreamHasher:
    def __init__(self) -> None:
        self.digest = hashlib.sha256()
        self.digest.update(b"rc-irstd-ordered-raw-logit-f32-v1\0")

    def update(self, image_id: str, logits: np.ndarray) -> None:
        identifier = str(image_id).encode("utf-8")
        values = np.asarray(logits)
        if values.dtype != np.float32 or values.ndim != 2:
            raise RuntimeError("Raw-logit stream requires 2-D FP32 arrays")
        if not np.isfinite(values).all():
            raise RuntimeError("Raw-logit stream contains NaN or infinity")
        self.digest.update(
            len(identifier).to_bytes(8, "little", signed=False)
        )
        self.digest.update(identifier)
        self.digest.update(
            np.asarray(values.shape, dtype="<i8").tobytes()
        )
        self.digest.update(
            np.ascontiguousarray(values, dtype="<f4").tobytes(order="C")
        )

    def hexdigest(self) -> str:
        return self.digest.hexdigest()


def _load_frozen_raw_logit_record(
    record: FrozenRawLogitRecord,
) -> tuple[np.ndarray, np.ndarray]:
    if not record.path.is_file():
        raise FileNotFoundError(
            f"Frozen raw-logit record is missing: {record.path}"
        )
    with np.load(record.path, allow_pickle=False) as payload:
        required = {"image_id", "logit", "mask"}
        missing = required.difference(payload.files)
        if missing:
            raise RuntimeError(
                "Frozen raw-logit record lacks fields: "
                + ", ".join(sorted(missing))
            )
        embedded_id = str(np.asarray(payload["image_id"]).item())
        if embedded_id != record.image_id:
            raise RuntimeError(
                "Frozen raw-logit record image_id changed after binding"
            )
        logits = np.asarray(payload["logit"])
        mask = np.asarray(payload["mask"])
    if logits.dtype != np.float32 or logits.ndim != 2:
        raise RuntimeError("Frozen comparator logits must be 2-D FP32")
    if not np.isfinite(logits).all():
        raise RuntimeError("Frozen comparator logits are non-finite")
    if mask.dtype not in {np.dtype(np.uint8), np.dtype(np.bool_)}:
        raise RuntimeError("Frozen comparator mask dtype is invalid")
    if mask.ndim != 2 or mask.shape != logits.shape:
        raise RuntimeError("Frozen comparator logit/mask shapes differ")
    if not np.isin(np.unique(mask), (0, 1, False, True)).all():
        raise RuntimeError("Frozen comparator mask is not binary")
    return (
        np.ascontiguousarray(logits),
        np.ascontiguousarray(mask.astype(bool, copy=False)),
    )


def _run_fold(
    binding: FrozenCheckpointBinding,
    *,
    source_root: Path,
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    dataset_root = validate_source_dataset_path(
        source_root / binding.held_out_source,
        expected_dataset=binding.held_out_source,
        source_root=source_root,
    )
    dataset = IRSTDEvalDataset(
        dataset_root,
        split="train",
        spatial_mode="native",
        pad_multiple=16,
        dataset_name=binding.held_out_source,
        load_masks=True,
    )
    comparators = load_frozen_fold_comparators(binding)
    if (
        len(dataset) != len(comparators.full_records)
        or len(dataset)
        != len(comparators.independently_trained_no_component_records)
    ):
        raise RuntimeError("Dataset and frozen comparator lengths differ")
    model = load_frozen_full_model(binding, device=device)
    records_root = output_root / binding.fold / "records"
    records_root.mkdir(parents=True, exist_ok=False)
    accumulator = _DiagnosticAccumulator()
    records: list[dict[str, Any]] = []
    stream_hashers = {
        name: _RawLogitStreamHasher()
        for name in BATCH_RAW_LOGIT_KEYS + (
            "independently_trained_v1_no_component",
        )
    }
    for index in range(len(dataset)):
        sample = dataset[index]
        meta = sample["meta"]
        if not isinstance(meta, dict):
            raise TypeError("Dataset metadata must be a mapping")
        image_id = str(meta["image_id"])
        historical_full_record = comparators.full_records[index]
        historical_no_component_record = (
            comparators.independently_trained_no_component_records[index]
        )
        if (
            image_id != historical_full_record.image_id
            or image_id != historical_no_component_record.image_id
        ):
            raise RuntimeError(
                f"Dataset/comparator order differs at record {index}"
            )
        _validate_source_record_path(
            str(meta["image_path"]),
            dataset_root=dataset_root,
        )
        _validate_source_record_path(
            str(meta["mask_path"]),
            dataset_root=dataset_root,
        )
        image = sample["image"]
        mask = sample.get("mask")
        if not isinstance(image, torch.Tensor) or not isinstance(
            mask,
            torch.Tensor,
        ):
            raise RuntimeError("Source image and source mask are required")
        result = diagnose_frozen_full_batch(
            model,
            image.unsqueeze(0).to(
                torch.device(device),
                dtype=torch.float32,
            ),
        )
        arrays: dict[str, np.ndarray] = {}
        for name, tensor in result.raw_logits.items():
            arrays[f"raw_logits__{name}"] = _crop_fp32(
                tensor[0, 0],
                meta,
            )
        historical_full_logits, historical_full_mask = (
            _load_frozen_raw_logit_record(historical_full_record)
        )
        historical_no_component_logits, historical_no_component_mask = (
            _load_frozen_raw_logit_record(
                historical_no_component_record
            )
        )
        arrays[
            "raw_logits__independently_trained_v1_no_component"
        ] = historical_no_component_logits
        historical_replay = (
            assert_historical_full_raw_logit_consistency(
                arrays["raw_logits__replayed_full"],
                historical_full_logits,
                image_id=image_id,
            )
        )
        component_gate = _crop_fp32(result.full_gates[0, 1], meta)
        component_contribution = _crop_fp32(
            result.component_contribution[0, 0],
            meta,
        )
        contrast_before = _crop_fp32(result.full_gates[0, 0], meta)
        contrast_after = _crop_fp32(
            result.conditional_full_gates[0, 0],
            meta,
        )
        diagnostic_maps = {
            "base_raw_logits": _crop_fp32(
                result.base_raw_logits[0, 0],
                meta,
            ),
            "component_contribution": component_contribution,
            "full_component_gate": component_gate,
            "full_contrast_gate": contrast_before,
            "conditional_contrast_gate": contrast_after,
            "context_preserved_expert_off_contrast_gate": _crop_fp32(
                result.context_preserved_expert_off_gates[0, 0],
                meta,
            ),
            "context_zero_expert_off_contrast_gate": _crop_fp32(
                result.context_zero_expert_off_gates[0, 0],
                meta,
            ),
        }
        arrays.update(diagnostic_maps)
        cropped_mask = np.asarray(
            crop_to_valid(mask[0].detach().cpu(), meta),
            dtype=np.uint8,
        )
        if not np.array_equal(
            cropped_mask.astype(bool),
            historical_full_mask,
        ):
            raise RuntimeError(
                f"Dataset/frozen-full mask differs for {image_id}"
            )
        if not np.array_equal(
            cropped_mask.astype(bool),
            historical_no_component_mask,
        ):
            raise RuntimeError(
                f"Dataset/no-component mask differs for {image_id}"
            )
        for variant in stream_hashers:
            stream_hashers[variant].update(
                image_id,
                arrays[f"raw_logits__{variant}"],
            )
        arrays.update(
            {
                "mask": cropped_mask,
                "image_id": np.asarray(image_id),
                "dataset_name": np.asarray(binding.held_out_source),
                "score_representation": np.asarray(SCORE_REPRESENTATION),
                "inference_autocast_enabled": np.asarray(False),
            }
        )
        record_path = records_root / _safe_record_name(
            image_id
        )
        if record_path.exists():
            raise RuntimeError(f"Duplicate output record: {record_path}")
        np.savez_compressed(record_path, **arrays)
        record_path.chmod(0o444)
        replay_error = float(result.replay_consistency["max_abs_error"])
        accumulator.update(
            component_gate=component_gate,
            component_contribution=component_contribution,
            contrast_gate_before=contrast_before,
            contrast_gate_after=contrast_after,
            mask=cropped_mask,
            replay_error=replay_error,
        )
        records.append(
            {
                "image_id": image_id,
                "file": str(record_path.relative_to(output_root)),
                "sha256": _sha256(record_path),
                "meta": meta_to_jsonable(meta),
                "full_replay_max_abs_error": replay_error,
                "historical_full_bitwise_equal": historical_replay[
                    "bitwise_equal"
                ],
            }
        )
    stream_hashes = {
        name: hasher.hexdigest()
        for name, hasher in stream_hashers.items()
    }
    frozen_full_stream = comparators.evidence["frozen_full"][
        "raw_logit_stream_sha256"
    ]
    no_component_stream = comparators.evidence[
        "independently_trained_v1_no_component"
    ]["raw_logit_stream_sha256"]
    if (
        stream_hashes["frozen_full"] != frozen_full_stream
        or stream_hashes["replayed_full"] != frozen_full_stream
    ):
        raise RuntimeError("Full output streams do not match frozen artifact")
    if (
        stream_hashes["independently_trained_v1_no_component"]
        != no_component_stream
    ):
        raise RuntimeError(
            "No-component output stream does not match frozen artifact"
        )
    ordered_ids = [str(record["image_id"]) for record in records]
    return {
        "fold": binding.fold,
        "held_out_source": binding.held_out_source,
        "training_source": binding.training_source,
        "device": device,
        "checkpoint": binding.to_jsonable(),
        "dataset_split": "train",
        "dataset_spatial_mode": "native",
        "num_records": len(records),
        "ordered_image_ids_sha256": ordered_ids_sha256(ordered_ids),
        "records_sha256": score_records_sha256(records),
        "raw_logit_stream_sha256_by_variant": stream_hashes,
        "frozen_comparator_evidence": dict(comparators.evidence),
        "all_historical_full_bitwise_equal": True,
        "exact_state_evaluator_input_ready": True,
        "dense_grid_is_conclusive": False,
        "records": records,
        "statistics": accumulator.summary(),
    }


def load_variant_raw_logit_samples(
    output_root: str | Path,
    variant: str,
) -> dict[str, tuple[RawLogitSample, ...]]:
    """Strict adapter from combined diagnostic records to exact-state inputs."""

    if variant not in COUNTERFACTUAL_VARIANTS:
        raise ValueError(f"Unsupported counterfactual variant: {variant}")
    root = _lexical_absolute(output_root, name="diagnostic_output_root")
    if _contains_outer_target(root):
        raise RuntimeError("Outer-target diagnostic output is forbidden")
    _assert_no_symlink_components(
        root,
        anchor=PROJECT_ROOT,
        name="diagnostic_output_root",
    )
    manifest_path = root / "manifest.json"
    sidecar_path = root / "manifest.sha256"
    _require_frozen_file(manifest_path, name="diagnostic manifest")
    _require_frozen_file(sidecar_path, name="diagnostic manifest sidecar")
    if sidecar_path.read_text(encoding="ascii").strip().split() != [
        _sha256(manifest_path),
        "manifest.json",
    ]:
        raise RuntimeError("Diagnostic manifest SHA256 sidecar drifted")
    manifest = _load_json(manifest_path)
    expected_manifest = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_only": True,
        "authorizes_go": False,
        "source_only": True,
        "outer_target_images_loaded": False,
        "outer_target_masks_loaded": False,
        "outer_target_access_authorized": False,
        "inference_dtype": "float32",
        "inference_autocast_enabled": False,
        "two_source_domains_closed": True,
        "exact_state_evaluator_input_ready": True,
        "dense_grid_is_conclusive": False,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"Diagnostic manifest mismatch at {key}")
    if manifest.get("counterfactual_variants") != list(
        COUNTERFACTUAL_VARIANTS
    ):
        raise RuntimeError("Diagnostic variant registration drifted")
    folds = manifest.get("folds")
    if not isinstance(folds, Mapping) or set(folds) != set(
        FROZEN_FULL_FOLDS
    ):
        raise RuntimeError("Diagnostic manifest does not close both folds")

    samples_by_domain: dict[str, tuple[RawLogitSample, ...]] = {}
    all_paths: set[Path] = set()
    for fold, expected_fold in FROZEN_FULL_FOLDS.items():
        fold_payload = folds[fold]
        if not isinstance(fold_payload, Mapping):
            raise RuntimeError(f"Invalid fold payload: {fold}")
        if (
            fold_payload.get("held_out_source")
            != expected_fold["held_out_source"]
            or fold_payload.get("training_source")
            != expected_fold["training_source"]
            or fold_payload.get("exact_state_evaluator_input_ready") is not True
            or fold_payload.get("all_historical_full_bitwise_equal") is not True
        ):
            raise RuntimeError(f"Fold identity/evidence mismatch: {fold}")
        records = fold_payload.get("records")
        if not isinstance(records, list) or not records:
            raise RuntimeError(f"Fold has no diagnostic records: {fold}")
        if fold_payload.get("num_records") != len(records):
            raise RuntimeError(f"Fold record count mismatch: {fold}")
        if fold_payload.get("records_sha256") != score_records_sha256(
            records
        ):
            raise RuntimeError(f"Fold records hash mismatch: {fold}")
        ids = [str(record.get("image_id", "")) for record in records]
        if (
            not all(ids)
            or len(set(ids)) != len(ids)
            or fold_payload.get("ordered_image_ids_sha256")
            != ordered_ids_sha256(ids)
        ):
            raise RuntimeError(f"Fold ordered image IDs drifted: {fold}")
        stream_hashes = fold_payload.get(
            "raw_logit_stream_sha256_by_variant"
        )
        if not isinstance(stream_hashes, Mapping) or variant not in stream_hashes:
            raise RuntimeError(f"Fold lacks variant stream hash: {fold}")

        loaded: list[RawLogitSample] = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise RuntimeError(f"Invalid record {index} in {fold}")
            raw_relative = record.get("file")
            if not isinstance(raw_relative, str) or not raw_relative:
                raise RuntimeError(f"Record path missing at {fold}:{index}")
            record_path = _lexical_absolute(
                root / raw_relative,
                name="diagnostic_record",
            )
            _assert_no_symlink_components(
                record_path,
                anchor=root,
                name="diagnostic_record",
            )
            _require_frozen_file(
                record_path,
                name="diagnostic record",
            )
            if record_path in all_paths:
                raise RuntimeError("Diagnostic folds reuse a record path")
            all_paths.add(record_path)
            if record.get("sha256") != _sha256(record_path):
                raise RuntimeError(
                    f"Diagnostic record hash drift at {fold}:{index}"
                )
            with np.load(record_path, allow_pickle=False) as payload:
                logit_key = f"raw_logits__{variant}"
                required = {
                    logit_key,
                    "mask",
                    "image_id",
                    "dataset_name",
                    "score_representation",
                    "inference_autocast_enabled",
                }
                if not required.issubset(payload.files):
                    raise RuntimeError(
                        f"Diagnostic record lacks adapter fields: {fold}:{index}"
                    )
                image_id = str(payload["image_id"].item())
                dataset_name = str(payload["dataset_name"].item())
                logits = np.asarray(payload[logit_key])
                mask = np.asarray(payload["mask"])
                if (
                    image_id != ids[index]
                    or dataset_name != expected_fold["held_out_source"]
                ):
                    raise RuntimeError(
                        f"Embedded diagnostic identity drift: {fold}:{index}"
                    )
                if (
                    logits.dtype != np.float32
                    or logits.ndim != 2
                    or not np.isfinite(logits).all()
                ):
                    raise RuntimeError(
                        f"Variant logits are not finite FP32: {fold}:{index}"
                    )
                if (
                    mask.shape != logits.shape
                    or not np.isin(mask, (0, 1, False, True)).all()
                ):
                    raise RuntimeError(
                        f"Variant mask contract drift: {fold}:{index}"
                    )
                if (
                    str(payload["score_representation"].item())
                    != SCORE_REPRESENTATION
                    or bool(
                        payload["inference_autocast_enabled"].item()
                    )
                ):
                    raise RuntimeError(
                        f"Embedded precision contract drift: {fold}:{index}"
                    )
                probability = torch.sigmoid(
                    torch.from_numpy(logits.copy())
                ).numpy()
                loaded.append(
                    RawLogitSample(
                        image_id=image_id,
                        logits=np.ascontiguousarray(logits),
                        probability=np.ascontiguousarray(
                            probability,
                            dtype=np.float32,
                        ),
                        mask=np.ascontiguousarray(mask.astype(bool)),
                    )
                )
        if raw_logit_stream_sha256(loaded) != stream_hashes[variant]:
            raise RuntimeError(f"Variant stream hash mismatch: {fold}")
        samples_by_domain[expected_fold["held_out_source"]] = tuple(loaded)

    on_disk = {
        path.resolve()
        for path in root.glob("*/records/*.npz")
    }
    if on_disk != {path.resolve() for path in all_paths}:
        raise RuntimeError("Diagnostic directory contains unlisted NPZ records")
    if set(samples_by_domain) != {"NUDT-SIRST", "IRSTD-1K"}:
        raise RuntimeError("Adapter did not close both source domains")
    return samples_by_domain


def validate_physical_gpu_binding(
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fail closed unless logical CUDA 0/1 are physical GPUs 2/3."""

    values = os.environ if environment is None else environment
    raw = values.get("CUDA_VISIBLE_DEVICES")
    visible = (
        [item.strip() for item in raw.split(",")]
        if isinstance(raw, str)
        else []
    )
    if visible != ["2", "3"]:
        raise RuntimeError(
            "Formal diagnosis requires CUDA_VISIBLE_DEVICES=2,3"
        )
    return {
        "cuda_visible_devices": "2,3",
        "physical_to_logical": {
            "2": 0,
            "3": 1,
        },
        "fold_to_physical": {
            "heldout_nudt": 2,
            "heldout_irstd": 3,
        },
        "fold_to_logical": {
            "heldout_nudt": 0,
            "heldout_irstd": 1,
        },
    }


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = 0 if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device is not visible: {requested}")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA devices are supported")
    return str(device)


def _prepare_output(path: str | Path, *, project_root: Path) -> Path:
    output = _lexical_absolute(path, name="output_root")
    expected = project_root / DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)
    if output != expected:
        raise RuntimeError(
            f"Diagnosis output must use the canonical path: {expected}"
        )
    if _contains_outer_target(output):
        raise RuntimeError("Outer-target output path is forbidden")
    _assert_no_symlink_components(
        output,
        anchor=project_root,
        name="output_root",
    )
    if output.exists():
        raise FileExistsError(f"Diagnosis output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    base = (
        PROJECT_ROOT
        / "outputs/aaai27/detectors/source_lodo_gate/seed42/full"
    )
    parser.add_argument(
        "--heldout-nudt-checkpoint",
        default=str(base / "heldout_nudt/last.pt"),
    )
    parser.add_argument(
        "--heldout-irstd-checkpoint",
        default=str(base / "heldout_irstd/last.pt"),
    )
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--heldout-nudt-device", default="cuda:0")
    parser.add_argument("--heldout-irstd-device", default="cuda:1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = _lexical_absolute(PROJECT_ROOT, name="project_root")
    source_root = _lexical_absolute(args.source_root, name="source_root")
    if source_root != project_root / "datasets":
        raise RuntimeError("--source-root must be the canonical datasets root")
    if _contains_outer_target(source_root):
        raise RuntimeError("Outer-target source root is forbidden")
    _assert_no_symlink_components(
        source_root,
        anchor=project_root,
        name="source_root",
    )
    bindings = validate_two_fold_checkpoint_contract(
        {
            "heldout_nudt": args.heldout_nudt_checkpoint,
            "heldout_irstd": args.heldout_irstd_checkpoint,
        },
        project_root=project_root,
    )
    devices = {
        "heldout_nudt": _resolve_device(args.heldout_nudt_device),
        "heldout_irstd": _resolve_device(args.heldout_irstd_device),
    }
    gpu_binding = validate_physical_gpu_binding()
    if devices != {
        "heldout_nudt": "cuda:0",
        "heldout_irstd": "cuda:1",
    }:
        raise RuntimeError("Fold devices must remain logical cuda:0/cuda:1")
    output_root = _prepare_output(
        args.output_dir,
        project_root=project_root,
    )

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            fold: executor.submit(
                _run_fold,
                binding,
                source_root=source_root,
                output_root=output_root,
                device=devices[fold],
            )
            for fold, binding in bindings.items()
        }
        folds = {fold: future.result() for fold, future in futures.items()}

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_only": True,
        "authorizes_go": False,
        "source_only": True,
        "outer_target_images_loaded": False,
        "outer_target_masks_loaded": False,
        "outer_target_access_authorized": False,
        "score_representation": SCORE_REPRESENTATION,
        "inference_dtype": "float32",
        "inference_autocast_enabled": False,
        "model_mode": "eval",
        "gradient_mode": "inference_mode",
        "base_logits_capture_source": (
            "fusion_head.forward_kwargs.base_logits"
        ),
        "counterfactual_variants": list(COUNTERFACTUAL_VARIANTS),
        "closed_source_domains": [
            "NUDT-SIRST",
            "IRSTD-1K",
        ],
        "two_source_domains_closed": set(folds)
        == set(FROZEN_FULL_FOLDS),
        "exact_state_evaluator_input_ready": all(
            value["exact_state_evaluator_input_ready"]
            for value in folds.values()
        ),
        "dense_grid_is_conclusive": False,
        "checkpoint_contract": (
            "two_fold_seed42_frozen_fixed_last_full_v1"
        ),
        "devices": devices,
        "physical_gpu_binding": gpu_binding,
        "folds": folds,
        "diagnostic_script_sha256": _sha256(Path(__file__).resolve()),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path.chmod(0o444)
    sidecar_path = output_root / "manifest.sha256"
    sidecar_path.write_text(
        f"{_sha256(manifest_path)}  {manifest_path.name}\n",
        encoding="ascii",
    )
    sidecar_path.chmod(0o444)
    print(manifest_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARCHITECTURE_VERSION",
    "BATCH_RAW_LOGIT_KEYS",
    "COUNTERFACTUAL_VARIANTS",
    "ComponentCounterfactualBatch",
    "FROZEN_FULL_FOLDS",
    "FrozenRawLogitRecord",
    "FrozenCheckpointBinding",
    "FrozenFoldComparators",
    "assert_full_raw_logit_replay_consistency",
    "assert_historical_full_raw_logit_consistency",
    "diagnose_frozen_full_batch",
    "load_frozen_fold_comparators",
    "load_frozen_full_model",
    "load_variant_raw_logit_samples",
    "validate_physical_gpu_binding",
    "validate_source_dataset_path",
    "validate_two_fold_checkpoint_contract",
]
