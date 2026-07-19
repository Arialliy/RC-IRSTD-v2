"""Collatable metadata helpers for evaluation samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch

from .mask_alignment import MASK_ALIGNMENT_NOT_LOADED_POLICY


class SampleMeta(TypedDict):
    """Metadata represented only by values supported by default_collate."""

    image_id: str
    dataset_name: str
    original_hw: torch.Tensor
    input_hw: torch.Tensor
    valid_hw: torch.Tensor
    padding_ltrb: torch.Tensor
    spatial_mode: str
    image_path: str
    mask_path: str
    mask_alignment_applied: bool
    mask_original_hw: torch.Tensor
    mask_aspect_relative_error: float
    mask_alignment_policy: str


def _pair(value: tuple[int, int], name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two integers")
    first, second = int(value[0]), int(value[1])
    if first <= 0 or second <= 0:
        raise ValueError(f"{name} values must be positive")
    return first, second


def make_sample_meta(
    *,
    image_id: str,
    dataset_name: str,
    original_hw: tuple[int, int],
    input_hw: tuple[int, int],
    valid_hw: tuple[int, int],
    padding_ltrb: tuple[int, int, int, int],
    spatial_mode: str,
    image_path: str | Path,
    mask_path: str | Path | None,
    mask_alignment_applied: bool = False,
    mask_original_hw: tuple[int, int] | None = None,
    mask_aspect_relative_error: float = -1.0,
    mask_alignment_policy: str = MASK_ALIGNMENT_NOT_LOADED_POLICY,
) -> SampleMeta:
    """Build a metadata dictionary that PyTorch can collate without hooks."""

    if not image_id or not dataset_name:
        raise ValueError("image_id and dataset_name must be non-empty")
    original_hw = _pair(original_hw, "original_hw")
    input_hw = _pair(input_hw, "input_hw")
    valid_hw = _pair(valid_hw, "valid_hw")
    if len(padding_ltrb) != 4 or any(int(value) < 0 for value in padding_ltrb):
        raise ValueError("padding_ltrb must contain four non-negative integers")
    if spatial_mode not in {"resize", "native"}:
        raise ValueError("spatial_mode must be 'resize' or 'native'")
    return {
        "image_id": str(image_id),
        "dataset_name": str(dataset_name),
        "original_hw": torch.tensor(original_hw, dtype=torch.int64),
        "input_hw": torch.tensor(input_hw, dtype=torch.int64),
        "valid_hw": torch.tensor(valid_hw, dtype=torch.int64),
        "padding_ltrb": torch.tensor(padding_ltrb, dtype=torch.int64),
        "spatial_mode": spatial_mode,
        "image_path": str(Path(image_path)),
        "mask_path": str(Path(mask_path)) if mask_path is not None else "",
        "mask_alignment_applied": bool(mask_alignment_applied),
        "mask_original_hw": torch.tensor(
            mask_original_hw if mask_original_hw is not None else (0, 0),
            dtype=torch.int64,
        ),
        "mask_aspect_relative_error": float(mask_aspect_relative_error),
        "mask_alignment_policy": str(mask_alignment_policy),
    }


def _integer_sequence(value: Any, *, length: int, name: str) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} integers")
    return tuple(int(item) for item in value)


def crop_to_valid(array: np.ndarray | torch.Tensor, meta: dict[str, Any]):
    """Remove spatial padding from an array/tensor using a single sample's meta.

    The last two dimensions are interpreted as height and width.  This works for
    ``H x W``, ``C x H x W``, and other arrays with leading dimensions.
    """

    if not isinstance(array, (np.ndarray, torch.Tensor)) or array.ndim < 2:
        raise TypeError("array must be a NumPy array or torch tensor with ndim >= 2")
    valid_h, valid_w = _integer_sequence(
        meta.get("valid_hw"), length=2, name="valid_hw"
    )
    left, top, right, bottom = _integer_sequence(
        meta.get("padding_ltrb"), length=4, name="padding_ltrb"
    )
    height, width = int(array.shape[-2]), int(array.shape[-1])
    if min(left, top, right, bottom) < 0:
        raise ValueError("padding values must be non-negative")
    if top + valid_h > height or left + valid_w > width:
        raise ValueError(
            f"valid region {(valid_h, valid_w)} with padding {(left, top, right, bottom)} "
            f"does not fit array shape {(height, width)}"
        )
    return array[..., top : top + valid_h, left : left + valid_w]


def meta_to_jsonable(meta: dict[str, Any]) -> dict[str, Any]:
    """Convert one unbatched metadata dictionary to JSON-compatible values."""

    result: dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, torch.Tensor):
            flattened = value.detach().cpu().reshape(-1).tolist()
            result[key] = flattened[0] if len(flattened) == 1 else flattened
        elif isinstance(value, np.ndarray):
            flattened = value.reshape(-1).tolist()
            result[key] = flattened[0] if len(flattened) == 1 else flattened
        elif isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result
