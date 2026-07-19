"""Extended data loading utilities used by risk-aware evaluation."""

from .dataset_meta import SampleMeta, crop_to_valid, make_sample_meta, meta_to_jsonable
from .eval_dataset import IRSTDEvalDataset
from .mask_alignment import (
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
from .split_utils import read_split_file, resolve_split_file, sample_id_from_entry

__all__ = [
    "IRSTDEvalDataset",
    "MASK_ALIGNMENT_ASPECT_TOLERANCE",
    "MASK_ALIGNMENT_NOT_LOADED_POLICY",
    "MASK_ALIGNMENT_POLICY",
    "MaskAlignment",
    "SampleMeta",
    "align_mask_to_image",
    "aspect_error_within_tolerance",
    "mask_alignment_policy",
    "crop_to_valid",
    "make_sample_meta",
    "meta_to_jsonable",
    "read_split_file",
    "relative_aspect_error",
    "validate_mask_alignment_evidence",
    "resolve_split_file",
    "sample_id_from_entry",
]
