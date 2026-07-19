"""Evaluation dataset with explicit split and spatial-protocol metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torch.utils.data import Dataset

from .dataset_meta import SampleMeta, make_sample_meta
from .mask_alignment import (
    MASK_ALIGNMENT_NOT_LOADED_POLICY,
    MASK_ALIGNMENT_POLICY,
    align_mask_to_image,
)
from .split_utils import (
    ensure_unique_sample_ids,
    read_split_file,
    resolve_split_file,
)


_RASTER_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
_IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]


def _normalize_size(base_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(base_size, bool):
        raise TypeError("base_size must be an int or (height, width) tuple")
    if isinstance(base_size, int):
        size = (base_size, base_size)
    elif isinstance(base_size, tuple) and len(base_size) == 2:
        size = (int(base_size[0]), int(base_size[1]))
    else:
        raise TypeError("base_size must be an int or (height, width) tuple")
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("base_size values must be positive")
    return size


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected an RGB image, got array shape {array.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
    return (tensor - _IMAGENET_MEAN) / _IMAGENET_STD


def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    array = np.asarray(mask, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError(f"Expected a single-channel mask, got array shape {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array > 0)).unsqueeze(0).float()


def _gray_to_tensor(image: Image.Image) -> torch.Tensor:
    """Return label-free grayscale intensity in [0, 1]."""

    array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if array.ndim != 2:
        raise ValueError(f"Expected grayscale image, got array shape {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)


class IRSTDEvalDataset(Dataset):
    """Read evaluation samples without changing the legacy ``IRSTD_Dataset``.

    ``spatial_mode='resize'`` creates the historical fixed-size protocol.
    ``spatial_mode='native'`` keeps the original pixels and pads the bottom/right
    edges to ``pad_multiple``.  Exporters should call ``crop_to_valid`` before
    computing metrics so padded pixels never enter a false-alarm denominator.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        split_file: str | Path | None = None,
        split: Literal["train", "val", "test"] = "test",
        spatial_mode: Literal["resize", "native"] = "resize",
        base_size: int | tuple[int, int] = 256,
        pad_multiple: int = 16,
        dataset_name: str | None = None,
        load_masks: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(dataset_dir).expanduser().resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset directory does not exist: {self.root}")
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Expected images/ directory under {self.root}")
        if not isinstance(load_masks, bool):
            raise TypeError("load_masks must be boolean")
        self.load_masks = load_masks
        if self.load_masks and not self.masks_dir.is_dir():
            raise FileNotFoundError(f"Expected masks/ directory under {self.root}")
        if spatial_mode not in {"resize", "native"}:
            raise ValueError("spatial_mode must be 'resize' or 'native'")
        if isinstance(pad_multiple, bool) or not isinstance(pad_multiple, int):
            raise TypeError("pad_multiple must be an integer")
        if pad_multiple <= 0:
            raise ValueError("pad_multiple must be positive")
        self.spatial_mode = spatial_mode
        self.base_hw = _normalize_size(base_size)
        self.pad_multiple = pad_multiple
        self.dataset_name = dataset_name or self.root.name
        if not self.dataset_name:
            raise ValueError("dataset_name must be non-empty")

        self.requested_split = str(split)
        self.split_role = "test" if split == "val" else str(split)
        self.split_file = resolve_split_file(
            self.root,
            split_file,
            split=self.split_role,
        )
        if split_file is None:
            self.split_authority_verified = True
        else:
            try:
                automatic = resolve_split_file(
                    self.root,
                    None,
                    split=self.split_role,
                )
            except (FileNotFoundError, ValueError):
                self.split_authority_verified = False
            else:
                self.split_authority_verified = automatic == self.split_file
        self.entries = read_split_file(self.split_file)
        self.image_ids = ensure_unique_sample_ids(self.entries)

    @staticmethod
    def _resolve_raster(directory: Path, image_id: str, *, is_mask: bool) -> Path:
        stems = [image_id]
        if is_mask:
            stems.append(f"{image_id}_pixels0")
        matches: list[Path] = []
        for stem in stems:
            for suffix in _RASTER_SUFFIXES:
                path = directory / f"{stem}{suffix}"
                if path.is_file():
                    matches.append(path)
            if matches:
                break
        if not matches:
            expected = ", ".join(f"{stem}<image-ext>" for stem in stems)
            raise FileNotFoundError(
                f"Missing {'mask' if is_mask else 'image'} for {image_id!r} under "
                f"{directory}; expected {expected}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple raster files found for {image_id!r}: "
                f"{', '.join(path.name for path in matches)}"
            )
        return matches[0]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | SampleMeta]:
        if not isinstance(index, int):
            raise TypeError("dataset index must be an integer")
        image_id = self.image_ids[index]
        image_path = self._resolve_raster(self.images_dir, image_id, is_mask=False)
        mask_path = (
            self._resolve_raster(self.masks_dir, image_id, is_mask=True)
            if self.load_masks
            else None
        )

        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
        mask = None
        alignment = None
        if mask_path is not None:
            with Image.open(mask_path) as opened_mask:
                mask = opened_mask.convert("L")
            mask, alignment = align_mask_to_image(mask, image, image_id)
        original_hw = (image.height, image.width)

        if self.spatial_mode == "resize":
            base_h, base_w = self.base_hw
            image = image.resize((base_w, base_h), Image.Resampling.BILINEAR)
            if mask is not None:
                mask = mask.resize((base_w, base_h), Image.Resampling.NEAREST)
            gray_tensor = _gray_to_tensor(image)
            image_tensor = _image_to_tensor(image)
            mask_tensor = _mask_to_tensor(mask) if mask is not None else None
            valid_hw = (base_h, base_w)
            input_hw = valid_hw
            padding = (0, 0, 0, 0)
        else:
            gray_tensor = _gray_to_tensor(image)
            image_tensor = _image_to_tensor(image)
            mask_tensor = _mask_to_tensor(mask) if mask is not None else None
            valid_h, valid_w = original_hw
            pad_h = (-valid_h) % self.pad_multiple
            pad_w = (-valid_w) % self.pad_multiple
            # Zero in normalized space corresponds to an ImageNet-mean border.
            image_tensor = functional.pad(image_tensor, (0, pad_w, 0, pad_h), value=0.0)
            gray_tensor = functional.pad(gray_tensor, (0, pad_w, 0, pad_h), value=0.0)
            if mask_tensor is not None:
                mask_tensor = functional.pad(
                    mask_tensor, (0, pad_w, 0, pad_h), value=0.0
                )
            valid_hw = original_hw
            input_hw = (valid_h + pad_h, valid_w + pad_w)
            padding = (0, 0, pad_w, pad_h)

        meta = make_sample_meta(
            image_id=image_id,
            dataset_name=self.dataset_name,
            original_hw=original_hw,
            input_hw=input_hw,
            valid_hw=valid_hw,
            padding_ltrb=padding,
            spatial_mode=self.spatial_mode,
            image_path=image_path,
            mask_path=mask_path,
            mask_alignment_applied=(alignment.applied if alignment is not None else False),
            mask_original_hw=(
                (
                    alignment.original_mask_size_wh[1],
                    alignment.original_mask_size_wh[0],
                )
                if alignment is not None
                else None
            ),
            mask_aspect_relative_error=(
                alignment.relative_aspect_error if alignment is not None else -1.0
            ),
            mask_alignment_policy=(
                MASK_ALIGNMENT_POLICY
                if alignment is not None
                else MASK_ALIGNMENT_NOT_LOADED_POLICY
            ),
        )
        sample: dict[str, torch.Tensor | SampleMeta] = {
            "image": image_tensor,
            "gray": gray_tensor,
            "meta": meta,
        }
        if mask_tensor is not None:
            sample["mask"] = mask_tensor
        return sample
