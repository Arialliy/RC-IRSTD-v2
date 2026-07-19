from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch


PREFIX_REPLACEMENTS = (
    ("conv_init.", "input_projection."),
    ("encoder_0.", "encoder0."),
    ("encoder_1.", "encoder1."),
    ("encoder_2.", "encoder2."),
    ("encoder_3.", "encoder3."),
    ("middle_layer.", "middle."),
    ("decoder_3.", "decoder3."),
    ("decoder_2.", "decoder2."),
    ("decoder_1.", "decoder1."),
    ("decoder_0.", "decoder0."),
    ("output_0.", "scale_heads.0."),
    ("output_1.", "scale_heads.1."),
    ("output_2.", "scale_heads.2."),
    ("output_3.", "scale_heads.3."),
    ("final.", "fusion."),
)

INNER_REPLACEMENTS = (
    (".ca.fc1.", ".channel_attention.shared.0."),
    (".ca.fc2.", ".channel_attention.shared.2."),
    (".sa.conv1.", ".spatial_attention.conv."),
)


def extract_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state", "state_dict", "model", "net"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                return candidate  # type: ignore[return-value]
        if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint  # type: ignore[return-value]
    raise ValueError("Unable to locate a model state dictionary in the upstream checkpoint")


def convert_upstream_mshnet_state(
    state_dict: Mapping[str, torch.Tensor],
    *,
    backend: str = "canonical",
) -> OrderedDict[str, torch.Tensor]:
    """Normalize DataParallel keys for the canonical target MSHNet.

    The target repository already uses the public ``conv_init``/``encoder_0``
    namespace.  The old delivery-package renaming remains available only via
    ``backend='compact'`` for explicit legacy diagnostics.
    """

    if backend not in {"canonical", "compact"}:
        raise ValueError("backend must be 'canonical' or 'compact'")
    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    for raw_key, value in state_dict.items():
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if backend == "compact":
            for source, destination in PREFIX_REPLACEMENTS:
                if key.startswith(source):
                    key = destination + key[len(source) :]
                    break
            for source, destination in INNER_REPLACEMENTS:
                key = key.replace(source, destination)
        if key in converted:
            raise ValueError(f"Checkpoint key normalization produced duplicate key: {key}")
        converted[key] = value
    return converted


__all__ = ["convert_upstream_mshnet_state", "extract_state_dict"]
