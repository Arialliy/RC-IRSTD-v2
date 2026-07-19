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
        },
        target,
    )
    print(target)
    print(f"tensor_keys={len(state)}")
    print(f"source_load_mode={load_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
