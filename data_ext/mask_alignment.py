"""Auditable alignment for the known same-scene mask resolution quirk."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from PIL import Image


MASK_ALIGNMENT_ASPECT_TOLERANCE = 0.01
MASK_ALIGNMENT_POLICY = (
    "nearest_neighbor_to_image_if_relative_aspect_error_le_0.01_v1"
)
MASK_ALIGNMENT_NOT_LOADED_POLICY = "labels_not_loaded"


def mask_alignment_policy(aspect_tolerance: float) -> str:
    """Return an explicit policy identifier for a guarded tolerance."""

    tolerance = float(aspect_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0 or tolerance >= 1.0:
        raise ValueError("aspect_tolerance must be finite and lie in [0, 1)")
    if tolerance == MASK_ALIGNMENT_ASPECT_TOLERANCE:
        return MASK_ALIGNMENT_POLICY
    return (
        "nearest_neighbor_to_image_if_relative_aspect_error_le_"
        f"{tolerance:.17g}_v1"
    )


@dataclass(frozen=True)
class MaskAlignment:
    applied: bool
    image_size_wh: tuple[int, int]
    original_mask_size_wh: tuple[int, int]
    relative_aspect_error: float
    policy: str = MASK_ALIGNMENT_POLICY

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def relative_aspect_error(
    image_size_wh: tuple[int, int],
    mask_size_wh: tuple[int, int],
) -> float:
    """Return the relative width/height-ratio discrepancy used by the guard."""

    if len(image_size_wh) != 2 or len(mask_size_wh) != 2:
        raise ValueError("image_size_wh and mask_size_wh must each contain two values")
    image_size = (int(image_size_wh[0]), int(image_size_wh[1]))
    mask_size = (int(mask_size_wh[0]), int(mask_size_wh[1]))
    if min(*image_size, *mask_size) <= 0:
        raise ValueError("image and mask dimensions must be positive")
    image_ratio = image_size[0] / image_size[1]
    mask_ratio = mask_size[0] / mask_size[1]
    return float(abs(image_ratio - mask_ratio) / max(abs(image_ratio), 1e-12))


def aspect_error_within_tolerance(
    relative_error: float,
    aspect_tolerance: float = MASK_ALIGNMENT_ASPECT_TOLERANCE,
) -> bool:
    """Apply the corrected compatibility rule's inclusive tolerance gate."""

    error = float(relative_error)
    tolerance = float(aspect_tolerance)
    mask_alignment_policy(tolerance)  # validates the tolerance
    if not math.isfinite(error) or error < 0.0:
        raise ValueError("relative_error must be finite and non-negative")
    # Preserve the explicit inclusive ``error <= tolerance`` contract. In
    # particular, a representable value even one ULP above 0.01 must not be
    # accepted.
    return error <= tolerance


def validate_mask_alignment_evidence(
    *,
    labels_loaded: bool,
    image_hw: tuple[int, int],
    original_mask_hw: tuple[int, int],
    applied: bool,
    relative_error: float,
    policy: str,
    aspect_tolerance: float = MASK_ALIGNMENT_ASPECT_TOLERANCE,
) -> None:
    """Validate the alignment provenance stored with an exported sample."""

    if not isinstance(labels_loaded, bool) or not isinstance(applied, bool):
        raise TypeError("labels_loaded and applied must be boolean")
    if len(image_hw) != 2 or len(original_mask_hw) != 2:
        raise ValueError("image_hw and original_mask_hw must each contain two values")
    image_hw = (int(image_hw[0]), int(image_hw[1]))
    original_mask_hw = (int(original_mask_hw[0]), int(original_mask_hw[1]))
    relative_error = float(relative_error)
    if not math.isfinite(relative_error):
        raise ValueError("mask alignment relative_error must be finite")
    if not labels_loaded:
        if applied:
            raise ValueError("mask alignment cannot be applied when labels were not loaded")
        if original_mask_hw != (0, 0) or relative_error != -1.0:
            raise ValueError("mask-free alignment evidence must use (0, 0) and -1.0 sentinels")
        if policy != MASK_ALIGNMENT_NOT_LOADED_POLICY:
            raise ValueError("mask-free alignment evidence has an invalid policy")
        return

    if min(*image_hw, *original_mask_hw) <= 0:
        raise ValueError("labeled alignment evidence must contain positive source dimensions")
    tolerance = float(aspect_tolerance)
    if policy != mask_alignment_policy(tolerance):
        raise ValueError("labeled alignment evidence has an invalid policy")
    expected_applied = original_mask_hw != image_hw
    if applied != expected_applied:
        raise ValueError("mask alignment applied flag disagrees with the source dimensions")
    # relative_aspect_error accepts width/height, while artifact metadata uses
    # the conventional height/width order.
    expected_error = relative_aspect_error(
        (image_hw[1], image_hw[0]),
        (original_mask_hw[1], original_mask_hw[0]),
    )
    if not math.isclose(relative_error, expected_error, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("recorded mask aspect error disagrees with the source dimensions")
    if not aspect_error_within_tolerance(expected_error, tolerance):
        raise ValueError("recorded mask alignment exceeds the guarded aspect-ratio tolerance")


def align_mask_to_image(
    mask: Image.Image,
    image: Image.Image,
    image_id: str,
    *,
    aspect_tolerance: float = MASK_ALIGNMENT_ASPECT_TOLERANCE,
) -> tuple[Image.Image, MaskAlignment]:
    """Align a same-aspect mask to its image, otherwise fail closed.

    The local NUAA ``Misc_111`` image is a resized version of the coordinate
    system declared by its mask/XML.  Nearest-neighbour resampling preserves
    categorical labels.  The aspect-ratio gate prevents this compatibility
    rule from hiding unrelated or geometrically corrupted pairs.
    """

    if not isinstance(mask, Image.Image) or not isinstance(image, Image.Image):
        raise TypeError("mask and image must be PIL images")
    if not isinstance(image_id, str) or not image_id.strip():
        raise ValueError("image_id must be a non-empty string")
    tolerance = float(aspect_tolerance)
    policy = mask_alignment_policy(tolerance)
    image_size = (int(image.size[0]), int(image.size[1]))
    mask_size = (int(mask.size[0]), int(mask.size[1]))
    if min(*image_size, *mask_size) <= 0:
        raise ValueError(f"Invalid image/mask size for {image_id!r}")
    relative_error = relative_aspect_error(image_size, mask_size)
    if mask_size == image_size:
        return mask, MaskAlignment(
            applied=False,
            image_size_wh=image_size,
            original_mask_size_wh=mask_size,
            relative_aspect_error=float(relative_error),
            policy=policy,
        )
    if not aspect_error_within_tolerance(relative_error, tolerance):
        raise ValueError(
            "Image/mask aspect-ratio mismatch for {!r}: image={} mask={}, "
            "relative_error={:.6f} > tolerance={:.6f}".format(
                image_id,
                image_size,
                mask_size,
                relative_error,
                tolerance,
            )
        )
    resampling = getattr(Image, "Resampling", Image).NEAREST
    aligned = mask.resize(image_size, resampling)
    return aligned, MaskAlignment(
        applied=True,
        image_size_wh=image_size,
        original_mask_size_wh=mask_size,
        relative_aspect_error=float(relative_error),
        policy=policy,
    )


__all__ = [
    "MASK_ALIGNMENT_ASPECT_TOLERANCE",
    "MASK_ALIGNMENT_NOT_LOADED_POLICY",
    "MASK_ALIGNMENT_POLICY",
    "MaskAlignment",
    "align_mask_to_image",
    "aspect_error_within_tolerance",
    "mask_alignment_policy",
    "relative_aspect_error",
    "validate_mask_alignment_evidence",
]
