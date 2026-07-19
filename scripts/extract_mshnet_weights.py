#!/usr/bin/env python3
"""Extract a tensor-only canonical MSHNet initialization checkpoint.

Restricted loading is attempted first. ``--trust-checkpoint`` enables ordinary
pickle loading only for a checkpoint produced locally by the user; never use it
for an untrusted download.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


STABLE_SLS_V1 = {
    "loss_id": "stable_sls_v1",
    "loss_implementation": "rc_irstd.losses.sls.StableSLSLoss",
    "bce_weight": 0.5,
    "scale_iou_weight": 1.0,
    "location_weight": 0.25,
    "max_positive_weight": 50.0,
    "lambda_tail": 0.0,
    "lambda_miss": 0.0,
    "lambda_margin": 0.0,
}


def _state_dict(payload: Any) -> dict[str, torch.Tensor]:
    value = payload
    if isinstance(value, Mapping):
        for key in ("model_state", "net", "state_dict", "model_state_dict", "model"):
            candidate = value.get(key)
            if isinstance(candidate, Mapping):
                value = candidate
                break
    if not isinstance(value, Mapping) or not value:
        raise ValueError("checkpoint does not contain a non-empty model state")
    result: dict[str, torch.Tensor] = {}
    for raw_key, tensor in value.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("resolved model state contains non-tensor values")
        key = str(raw_key)
        if key.startswith("module."):
            key = key[7:]
        if key in result:
            raise ValueError(f"duplicate normalized key: {key}")
        result[key] = tensor.detach().cpu()
    return result


def _source_training_identity(payload: Any) -> dict[str, Any] | None:
    """Extract a conservative, serialization-safe identity from a trainer payload."""

    if not isinstance(payload, Mapping):
        return None
    config = payload.get("config")
    if not isinstance(config, Mapping):
        return None
    model = config.get("model")
    loss = config.get("loss")
    data = config.get("data")
    training = config.get("training")
    if not all(isinstance(value, Mapping) for value in (model, loss, data, training)):
        return None
    sls = loss.get("sls_kwargs")
    sources = data.get("sources")
    if not isinstance(sls, Mapping) or not isinstance(sources, list):
        return None
    source_names: list[str] = []
    for source in sources:
        if not isinstance(source, Mapping) or "name" not in source:
            return None
        source_names.append(str(source["name"]))
    return {
        "schema_version": "rc-irstd-source-training-identity-v1",
        "architecture_id": "canonical_mshnet",
        "model_backend": str(model.get("backend", "")),
        "loss_id": "stable_sls_v1",
        "loss_implementation": "rc_irstd.losses.sls.StableSLSLoss",
        "bce_weight": float(sls.get("bce_weight", float("nan"))),
        "scale_iou_weight": float(sls.get("iou_weight", float("nan"))),
        "location_weight": float(sls.get("location_weight", float("nan"))),
        "max_positive_weight": float(
            sls.get("max_positive_weight", float("nan"))
        ),
        "auxiliary_weight": float(loss.get("auxiliary_weight", float("nan"))),
        "lambda_tail": float(loss.get("lambda_tail", float("nan"))),
        "lambda_miss": float(loss.get("lambda_miss", float("nan"))),
        "lambda_margin": float(loss.get("lambda_margin", float("nan"))),
        "source_names": source_names,
        "train_split": str(data.get("train_split", "")),
        "diagnostic_test_eval": bool(data.get("diagnostic_test_eval", False)),
        "checkpoint_policy": str(training.get("checkpoint_selection", "")),
        "warmup_epochs": int(training.get("warmup_epochs", -1)),
        "configured_epochs": int(training.get("epochs", -1)),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "inference_head": str(payload.get("inference_head", "")),
        "target_test_labels_used_for_checkpoint_selection": bool(
            payload.get("test_labels_used_for_selection", False)
        ),
    }


def _require_stable_sls_v1(identity: Mapping[str, Any] | None) -> None:
    if identity is None:
        raise ValueError("source checkpoint has no embedded trainer identity")
    expected = {
        **STABLE_SLS_V1,
        "architecture_id": "canonical_mshnet",
        "model_backend": "canonical",
        "train_split": "train",
        "diagnostic_test_eval": False,
        "checkpoint_policy": "fixed_last",
        "auxiliary_weight": 0.25,
        "warmup_epochs": 5,
        "configured_epochs": 400,
        "checkpoint_epoch": 399,
        "inference_head": "multi_scale_fused",
        "target_test_labels_used_for_checkpoint_selection": False,
    }
    mismatches = {
        key: {"expected": value, "actual": identity.get(key)}
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if mismatches:
        raise ValueError(f"source checkpoint is not StableSLS-v1: {mismatches}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Allow weights_only=False for a locally produced trusted checkpoint",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--require-stable-sls-v1",
        action="store_true",
        help="Fail unless the source checkpoint embeds the frozen StableSLS-v1 identity",
    )
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    target = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists() and not args.force:
        raise FileExistsError(f"output exists: {target}; pass --force")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
        load_mode = "restricted_weights_only"
    except Exception as error:
        if not args.trust_checkpoint:
            raise RuntimeError(
                "Restricted checkpoint loading failed. For a checkpoint you created "
                "locally, rerun with --trust-checkpoint; never trust a downloaded file."
            ) from error
        payload = torch.load(source, map_location="cpu", weights_only=False)
        load_mode = "trusted_legacy_pickle"
    state = _state_dict(payload)
    source_training_identity = _source_training_identity(payload)
    if args.require_stable_sls_v1:
        _require_stable_sls_v1(source_training_identity)
        extension_prefixes = ("contrast_pyramid.", "component_context.", "fusion_head.")
        extension_keys = sorted(
            key for key in state if key.startswith(extension_prefixes)
        )
        if extension_keys:
            raise ValueError(
                "StableSLS-v1 initializer must be canonical MSHNet only; "
                f"found RC extension keys: {extension_keys[:5]}"
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "mshnet_tensor_only_initialization",
            "format_version": 1,
            "model_config": {
                "backend": "canonical",
                "input_channels": 3,
                "channels": [16, 32, 64, 128, 256],
                "block_counts": [2, 2, 2, 2],
            },
            "model_state": state,
            "source_checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_load_mode": load_mode,
            "source_training_identity": source_training_identity,
        },
        target,
    )
    print(target)
    print(f"tensor_keys={len(state)}")
    print(f"source_load_mode={load_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
