"""Non-wrapping facade for the repository's canonical MSHNet.

``build_mshnet`` returns the original class itself.  Structured output is an
opt-in view over a raw forward result, so model parameter names, state-dict
keys, and forward tensors stay unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypeAlias

import torch
from torch import nn

from model.MSHNet import MSHNet, ResNet


RawMSHNetOutput: TypeAlias = tuple[list[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class MSHNetOutput:
    """Typed view of the legacy ``(auxiliary_logits, logits)`` result."""

    auxiliary_logits: tuple[torch.Tensor, ...]
    logits: torch.Tensor


class StructuredCanonicalMSHNet(MSHNet):
    """Canonical detector with the complete-solution structured forward API.

    No child module is added, so state-dict keys remain byte-for-byte compatible
    with :class:`model.MSHNet.MSHNet` and upstream checkpoints.
    """

    backend = "canonical"

    def __init__(self, input_channels: int = 3, *, block: type[nn.Module] = ResNet):
        super().__init__(input_channels=input_channels, block=block)
        self.input_channels = int(input_channels)

    def forward(
        self,
        inputs: torch.Tensor,
        multi_scale: bool = True,
        *,
        warm_flag: bool | None = None,
    ) -> MSHNetOutput:
        if warm_flag is not None:
            if not isinstance(warm_flag, bool):
                raise TypeError("warm_flag must be boolean")
            if not isinstance(multi_scale, bool):
                raise TypeError("multi_scale must be boolean")
            multi_scale = warm_flag
        if not isinstance(multi_scale, bool):
            raise TypeError("multi_scale must be boolean")
        auxiliary, logits = super().forward(inputs, multi_scale)
        return MSHNetOutput(tuple(auxiliary), logits)

    def export_config(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "input_channels": self.input_channels,
            "channels": [16, 32, 64, 128, 256],
            "block_counts": [2, 2, 2, 2],
        }


def build_mshnet(
    input_channels: int | Mapping[str, object] = 3,
    *,
    block: type[nn.Module] = ResNet,
) -> nn.Module:
    """Construct the canonical detector or an explicit smoke-only backend.

    Integer input preserves the original non-wrapping façade.  A mapping uses
    the complete-solution structured API.  ``backend=complete_compat`` must be
    explicitly requested for the configurable compact architecture; paper
    configs default to the checkpoint-compatible canonical detector.
    """

    if isinstance(input_channels, Mapping):
        config = dict(input_channels)
        backend = str(config.get("backend", "canonical")).lower()
        channels = tuple(
            int(value)
            for value in config.get("channels", (16, 32, 64, 128, 256))
        )
        block_counts = tuple(
            int(value) for value in config.get("block_counts", (2, 2, 2, 2))
        )
        channel_count = int(config.get("input_channels", 3))
        # RC-MSHNET-PATCH: model builder
        if backend in {"rc_mshnet", "rc-mshnet", "proposed"}:
            from .rc_mshnet import build_rc_mshnet

            return build_rc_mshnet(config)
        if backend in {"complete_compat", "compact", "smoke"}:
            from .compact_mshnet import CompactMSHNet

            return CompactMSHNet(
                input_channels=channel_count,
                channels=channels,
                block_counts=block_counts,
            )
        if backend not in {"canonical", "upstream", "mshnet"}:
            raise ValueError(f"Unsupported MSHNet backend: {backend}")
        if channels != (16, 32, 64, 128, 256) or block_counts != (2, 2, 2, 2):
            raise ValueError(
                "Canonical MSHNet has fixed channels/block_counts. Set "
                "model.backend=complete_compat only for diagnostic smoke runs."
            )
        return StructuredCanonicalMSHNet(channel_count, block=block)

    if isinstance(input_channels, bool) or not isinstance(input_channels, int):
        raise TypeError("input_channels must be an integer")
    if input_channels <= 0:
        raise ValueError("input_channels must be positive")
    if not isinstance(block, type) or not issubclass(block, nn.Module):
        raise TypeError("block must be a torch.nn.Module class")
    return MSHNet(input_channels=input_channels, block=block)


def structure_mshnet_output(output: object) -> MSHNetOutput:
    """Validate and expose a raw MSHNet result without copying its tensors."""

    if isinstance(output, MSHNetOutput):
        return output

    if not isinstance(output, (tuple, list)) or len(output) != 2:
        raise TypeError(
            "MSHNet output must be a two-item (auxiliary_logits, logits) sequence"
        )
    auxiliary_logits, logits = output
    if not isinstance(auxiliary_logits, (tuple, list)):
        raise TypeError("auxiliary_logits must be a list or tuple of tensors")
    if not all(isinstance(value, torch.Tensor) for value in auxiliary_logits):
        raise TypeError("every auxiliary logit must be a torch tensor")
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch tensor")
    return MSHNetOutput(tuple(auxiliary_logits), logits)


def forward_mshnet(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    warm_flag: bool,
) -> MSHNetOutput:
    """Run the canonical forward method and return its opt-in typed view."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(warm_flag, bool):
        raise TypeError("warm_flag must be a bool")
    if isinstance(model, StructuredCanonicalMSHNet) or getattr(
        model, "backend", None
    ) == "complete_compat":
        return structure_mshnet_output(model(inputs, multi_scale=warm_flag))
    if isinstance(model, MSHNet):
        return structure_mshnet_output(model(inputs, warm_flag))
    raise TypeError("model is not a supported RC-IRSTD MSHNet backend")


__all__ = [
    "MSHNet",
    "MSHNetOutput",
    "RawMSHNetOutput",
    "ResNet",
    "StructuredCanonicalMSHNet",
    "build_mshnet",
    "forward_mshnet",
    "structure_mshnet_output",
]
