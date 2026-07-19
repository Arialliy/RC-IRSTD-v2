"""Normalize a public MSHNet checkpoint without renaming canonical keys."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from model.MSHNet import MSHNet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--warm-flag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record which canonical MSHNet inference head the checkpoint uses",
    )
    return parser


def _extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("net", "state_dict", "model_state_dict", "model_state", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                payload = candidate
                break
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("Unable to locate a model state dictionary")
    if not all(isinstance(value, torch.Tensor) for value in payload.values()):
        raise ValueError("Resolved state dictionary contains non-tensor values")
    state = {
        (str(key)[len("module.") :] if str(key).startswith("module.") else str(key)): value
        for key, value in payload.items()
    }
    if len(state) != len(payload):
        raise ValueError("Removing the DataParallel prefix produced duplicate keys")
    return state


def _safe_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(
            "Checkpoint could not be loaded in weights-only mode; convert it in a "
            "trusted environment instead of enabling arbitrary pickle execution"
        ) from error


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = Path(args.output).expanduser()
    if source == destination.resolve():
        raise ValueError("--input and --output must be different files")
    raw_payload = _safe_load(source)
    raw_state: Any = raw_payload
    if isinstance(raw_payload, Mapping):
        for key in ("net", "state_dict", "model_state_dict", "model_state", "model"):
            candidate = raw_payload.get(key)
            if isinstance(candidate, Mapping):
                raw_state = candidate
                break
    module_prefix_removed = bool(
        isinstance(raw_state, Mapping)
        and any(str(key).startswith("module.") for key in raw_state)
    )
    state = _extract_state_dict(raw_payload)
    model = MSHNet(3)
    incompatible = model.load_state_dict(state, strict=False)
    if (incompatible.missing_keys or incompatible.unexpected_keys) and not args.allow_partial:
        raise RuntimeError(
            f"Checkpoint is not fully compatible. Missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}. Use --allow-partial only "
            "after inspecting the conversion report."
        )
    partial = bool(incompatible.missing_keys or incompatible.unexpected_keys)
    payload = {
        "format_version": "rc-v2-canonical-mshnet-v1",
        "kind": "detector",
        "epoch": -1,
        "net": model.state_dict(),
        "warm_flag": bool(args.warm_flag),
        "inference_head": "multi_scale" if args.warm_flag else "warm_stage",
        "selection_rule": "checkpoint_conversion_only",
        "diagnostic_only": partial,
        "formal_paper_checkpoint": not partial,
        "conversion": {
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "module_prefix_removed": module_prefix_removed,
            "randomly_initialized_missing_parameters": list(
                incompatible.missing_keys
            ),
        },
    }
    _atomic_torch_save(destination, payload)
    print(destination)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
