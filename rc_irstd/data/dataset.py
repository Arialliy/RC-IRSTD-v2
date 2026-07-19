"""Canonical dataset exports plus the complete-solution mapping API.

The facade deliberately delegates to the existing loaders.  It therefore
cannot drift to a second mask convention, augmentation policy, raster resolver,
or split-discovery rule.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as functional
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from data_ext.eval_dataset import IRSTDEvalDataset
from data_ext.mask_alignment import (
    MASK_ALIGNMENT_ASPECT_TOLERANCE,
    MASK_ALIGNMENT_NOT_LOADED_POLICY,
    MASK_ALIGNMENT_POLICY,
    MaskAlignment,
    align_mask_to_image,
    aspect_error_within_tolerance,
    mask_alignment_policy,
    relative_aspect_error,
    validate_mask_alignment_evidence,
)
from data_ext.split_utils import (
    ensure_unique_sample_ids,
    read_split_file,
    resolve_split_file,
    sample_id_from_entry,
)
from utils.data import IRSTD_Dataset


_IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)
_IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)


@dataclass(frozen=True)
class SampleMeta:
    """Complete-solution metadata with explicit alignment provenance."""

    image_id: str
    dataset_name: str
    domain_id: int
    original_hw: tuple[int, int]
    valid_hw: tuple[int, int]
    input_hw: tuple[int, int]
    sequence_id: str
    image_path: str
    mask_path: str
    has_mask: bool
    mask_alignment_applied: bool
    mask_original_hw: tuple[int, int]
    mask_aspect_relative_error: float
    mask_alignment_policy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rgb_to_tensor(image: Image.Image) -> torch.Tensor:
    tensor = pil_to_tensor(image.convert("RGB")).float().div_(255.0)
    return (tensor - _IMAGENET_MEAN[:, None, None]) / _IMAGENET_STD[:, None, None]


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    tensor = pil_to_tensor(mask.convert("L"))
    return (tensor > 0).to(dtype=torch.float32)


def unnormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    mean = _IMAGENET_MEAN.to(device=tensor.device, dtype=tensor.dtype)[:, None, None]
    std = _IMAGENET_STD.to(device=tensor.device, dtype=tensor.dtype)[:, None, None]
    return (tensor * std + mean).clamp(0.0, 1.0)


def _augment_pair(
    image: Image.Image,
    mask: Image.Image,
) -> tuple[Image.Image, Image.Image]:
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    rotation = random.choice((0, 90, 180, 270))
    if rotation:
        transpose = {
            90: Image.Transpose.ROTATE_90,
            180: Image.Transpose.ROTATE_180,
            270: Image.Transpose.ROTATE_270,
        }[rotation]
        # PIL.Image.rotate defaults to expand=False and therefore crops most
        # rectangular infrared frames (including small edge targets). The
        # transpose rotations preserve every pixel and swap H/W when needed.
        image = image.transpose(transpose)
        mask = mask.transpose(transpose)
    if random.random() < 0.35:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.35:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
    return image, mask


def _pad_pair(
    image: torch.Tensor,
    mask: torch.Tensor,
    multiple: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    height, width = (int(value) for value in image.shape[-2:])
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    # Match the repository's formal native evaluator: zero in normalized space
    # is an ImageNet-mean border, and padded pixels are cropped before metrics.
    image = functional.pad(image, (0, pad_w, 0, pad_h), value=0.0)
    mask = functional.pad(mask, (0, pad_w, 0, pad_h), value=0.0)
    return image, mask, (height, width)


class IRSTDDataset(Dataset[dict[str, Any]]):
    """Dictionary-returning compatibility loader backed by frozen local splits.

    This preserves the complete-solution constructor and collate API while
    delegating split discovery, raster resolution, mask binarization, and the
    NUAA guarded alignment rule to the target repository's canonical policy.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        domain_id: int = 0,
        dataset_name: str | None = None,
        training: bool = False,
        image_size: int | tuple[int, int] = 256,
        preserve_aspect_eval: bool = True,
        pad_multiple: int = 16,
        augment: bool = True,
        allow_missing_masks: bool = False,
        split_file: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset directory does not exist: {self.root}")
        self.requested_split = str(split).lower()
        split_key = self.requested_split
        if split_key in {"val", "validation"}:
            split_key = "test"
        if split_key not in {"train", "test"}:
            raise ValueError("Only frozen train/test splits are supported")
        if isinstance(domain_id, bool) or not isinstance(domain_id, int) or domain_id < 0:
            raise ValueError("domain_id must be a non-negative integer")
        self.split = split_key
        self.split_role = split_key
        self.domain_id = int(domain_id)
        self.dataset_name = str(dataset_name or self.root.name)
        if not self.dataset_name:
            raise ValueError("dataset_name must be non-empty")
        self.training = bool(training)
        if self.training and self.split != "train":
            raise ValueError(
                "training=True is permitted only for the frozen train split; "
                "test/val aliases are evaluation-only"
            )
        if isinstance(image_size, bool):
            raise TypeError("image_size must be an int or (height, width)")
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        elif isinstance(image_size, tuple) and len(image_size) == 2:
            self.image_size = (int(image_size[0]), int(image_size[1]))
        else:
            raise TypeError("image_size must be an int or (height, width)")
        if min(self.image_size) <= 0:
            raise ValueError("image_size values must be positive")
        if isinstance(pad_multiple, bool) or not isinstance(pad_multiple, int):
            raise TypeError("pad_multiple must be an integer")
        if pad_multiple <= 0:
            raise ValueError("pad_multiple must be positive")
        self.preserve_aspect_eval = bool(preserve_aspect_eval)
        self.pad_multiple = int(pad_multiple)
        self.augment = bool(augment)
        self.allow_missing_masks = bool(allow_missing_masks)
        if self.training and self.allow_missing_masks:
            raise ValueError("Training samples require ground-truth masks")
        self.split_file = resolve_split_file(
            self.root, split_file, split=self.split
        )
        if split_file is None:
            self.split_authority_verified = True
        else:
            try:
                automatic = resolve_split_file(self.root, None, split=self.split)
            except (FileNotFoundError, ValueError):
                self.split_authority_verified = False
            else:
                self.split_authority_verified = automatic == self.split_file
        self.image_ids = ensure_unique_sample_ids(read_split_file(self.split_file))
        self.items = list(self.image_ids)
        if not self.items:
            raise ValueError(f"Empty split file: {self.split_file}")
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        if not self.images_dir.is_dir():
            raise FileNotFoundError(self.images_dir)
        if not self.masks_dir.is_dir() and not self.allow_missing_masks:
            raise FileNotFoundError(self.masks_dir)

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _resolve_raster(folder: Path, image_id: str, *, is_mask: bool) -> Path:
        return IRSTDEvalDataset._resolve_raster(folder, image_id, is_mask=is_mask)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not isinstance(index, int):
            raise TypeError("dataset index must be an integer")
        image_id = self.items[index]
        image_path = self._resolve_raster(self.images_dir, image_id, is_mask=False)
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")

        has_mask = True
        try:
            mask_path = self._resolve_raster(self.masks_dir, image_id, is_mask=True)
        except (FileNotFoundError, NotADirectoryError):
            if not self.allow_missing_masks:
                raise
            mask_path = None
            has_mask = False
        if has_mask and mask_path is not None:
            with Image.open(mask_path) as opened:
                mask = opened.convert("L")
            mask, alignment = align_mask_to_image(mask, image, image_id)
            alignment_applied = alignment.applied
            mask_original_hw = (
                alignment.original_mask_size_wh[1],
                alignment.original_mask_size_wh[0],
            )
            alignment_error = alignment.relative_aspect_error
            alignment_policy_value = alignment.policy
        else:
            mask = Image.new("L", image.size, 0)
            alignment_applied = False
            mask_original_hw = (0, 0)
            alignment_error = -1.0
            alignment_policy_value = MASK_ALIGNMENT_NOT_LOADED_POLICY
        original_hw = (image.height, image.width)

        if self.training:
            if self.augment:
                image, mask = _augment_pair(image, mask)
            target_h, target_w = self.image_size
            image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
            mask = mask.resize((target_w, target_h), Image.Resampling.NEAREST)
            image_tensor = rgb_to_tensor(image)
            mask_tensor = mask_to_tensor(mask)
            valid_hw = (target_h, target_w)
        elif self.preserve_aspect_eval:
            image_tensor = rgb_to_tensor(image)
            mask_tensor = mask_to_tensor(mask)
            image_tensor, mask_tensor, valid_hw = _pad_pair(
                image_tensor, mask_tensor, self.pad_multiple
            )
        else:
            target_h, target_w = self.image_size
            image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
            mask = mask.resize((target_w, target_h), Image.Resampling.NEAREST)
            image_tensor = rgb_to_tensor(image)
            mask_tensor = mask_to_tensor(mask)
            valid_hw = (target_h, target_w)

        sequence_id = (
            image_id.rsplit("_", 1)[0]
            if "_" in image_id
            else self.dataset_name
        )
        meta = SampleMeta(
            image_id=image_id,
            dataset_name=self.dataset_name,
            domain_id=self.domain_id,
            original_hw=original_hw,
            valid_hw=valid_hw,
            input_hw=tuple(int(value) for value in image_tensor.shape[-2:]),
            sequence_id=sequence_id,
            image_path=str(image_path),
            mask_path=str(mask_path) if mask_path is not None else "",
            has_mask=has_mask,
            mask_alignment_applied=alignment_applied,
            mask_original_hw=mask_original_hw,
            mask_aspect_relative_error=float(alignment_error),
            mask_alignment_policy=alignment_policy_value,
        )
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "domain_name": self.dataset_name,
            "meta": meta,
        }


def collate_fixed(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("batch cannot be empty")
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "domain_id": torch.stack([item["domain_id"] for item in batch]),
        "domain_name": [str(item["domain_name"]) for item in batch],
        "meta": [item["meta"] for item in batch],
    }


def collate_eval(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    shapes = [tuple(item["image"].shape) for item in batch]
    if len(set(shapes)) != 1:
        raise ValueError(
            "Variable-resolution evaluation requires batch_size=1 or equal padded "
            f"shapes; received {shapes}"
        )
    return collate_fixed(batch)


def crop_to_valid(
    tensor: torch.Tensor,
    meta: SampleMeta | Mapping[str, Any],
) -> torch.Tensor:
    valid_hw = meta.valid_hw if isinstance(meta, SampleMeta) else meta["valid_hw"]
    if isinstance(valid_hw, torch.Tensor):
        valid_hw = valid_hw.detach().cpu().reshape(-1).tolist()
    height, width = (int(value) for value in valid_hw)
    return tensor[..., :height, :width]

__all__ = [
    "IRSTD_Dataset",
    "IRSTDDataset",
    "IRSTDEvalDataset",
    "MASK_ALIGNMENT_ASPECT_TOLERANCE",
    "MASK_ALIGNMENT_NOT_LOADED_POLICY",
    "MASK_ALIGNMENT_POLICY",
    "MaskAlignment",
    "SampleMeta",
    "align_mask_to_image",
    "aspect_error_within_tolerance",
    "mask_alignment_policy",
    "ensure_unique_sample_ids",
    "read_split_file",
    "rgb_to_tensor",
    "resolve_split_file",
    "relative_aspect_error",
    "validate_mask_alignment_evidence",
    "sample_id_from_entry",
    "collate_eval",
    "collate_fixed",
    "crop_to_valid",
    "mask_to_tensor",
    "unnormalize_image",
]
