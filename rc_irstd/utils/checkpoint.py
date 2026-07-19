from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import torch

from .io import ensure_dir


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    *,
    allow_unsafe_legacy: bool = False,
) -> dict[str, Any]:
    """Load a checkpoint with PyTorch's restricted unpickler by default.

    ``allow_unsafe_legacy`` is only for locally trusted historical resume files
    that contain unsupported pickle objects.  It must never be enabled for a
    downloaded deployment/inference checkpoint.
    """

    source = Path(path)
    try:
        checkpoint = torch.load(source, map_location=device, weights_only=True)
    except Exception:
        if not allow_unsafe_legacy:
            raise
        warnings.warn(
            "Loading a trusted legacy checkpoint with weights_only=False; "
            "do not use this option for untrusted files.",
            RuntimeWarning,
            stacklevel=2,
        )
        checkpoint = torch.load(source, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {path}")
    return checkpoint
