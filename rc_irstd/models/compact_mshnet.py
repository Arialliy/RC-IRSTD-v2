"""Configurable compact MSHNet used only by complete-solution smoke configs.

The publication/default backend remains :mod:`model.MSHNet`.  This separate
implementation preserves the reference package's tiny CPU-smoke capability
without changing canonical checkpoint keys or silently substituting a model in
paper runs.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as functional

from .mshnet import MSHNetOutput


class _ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(
            self.shared(inputs.mean(dim=(-2, -1), keepdim=True))
            + self.shared(inputs.amax(dim=(-2, -1), keepdim=True))
        )


class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in {3, 7}:
            raise ValueError("kernel_size must be 3 or 7")
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean_map = inputs.mean(dim=1, keepdim=True)
        max_map = inputs.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat((mean_map, max_map), dim=1)))


class _AttentionResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.channel_attention = _ChannelAttention(out_channels)
        self.spatial_attention = _SpatialAttention()
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        output = self.activation(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        output = output * self.channel_attention(output)
        output = output * self.spatial_attention(output)
        return self.activation(output + residual)


def _make_stage(
    in_channels: int,
    out_channels: int,
    blocks: int,
) -> nn.Sequential:
    if blocks < 1:
        raise ValueError("each stage must contain at least one block")
    layers: list[nn.Module] = [_AttentionResidualBlock(in_channels, out_channels)]
    layers.extend(
        _AttentionResidualBlock(out_channels, out_channels)
        for _ in range(blocks - 1)
    )
    return nn.Sequential(*layers)


class CompactMSHNet(nn.Module):
    """Explicit non-canonical backend for fast functional smoke tests."""

    backend = "complete_compat"

    def __init__(
        self,
        input_channels: int = 3,
        channels: Sequence[int] = (2, 4, 8, 16, 32),
        block_counts: Sequence[int] = (1, 1, 1, 1),
    ) -> None:
        super().__init__()
        if len(channels) != 5 or len(block_counts) != 4:
            raise ValueError("channels must have 5 values and block_counts must have 4")
        values = tuple(int(value) for value in channels)
        blocks = tuple(int(value) for value in block_counts)
        if min(values) <= 0 or min(blocks) <= 0:
            raise ValueError("channels and block_counts must be positive")
        self.input_channels = int(input_channels)
        self.channels = values
        self.block_counts = blocks
        c0, c1, c2, c3, c4 = values
        b1, b2, b3, b4 = blocks
        self.pool = nn.MaxPool2d(2, 2)
        self.input_projection = nn.Conv2d(self.input_channels, c0, 1)
        self.encoder0 = _make_stage(c0, c0, 1)
        self.encoder1 = _make_stage(c0, c1, b1)
        self.encoder2 = _make_stage(c1, c2, b2)
        self.encoder3 = _make_stage(c2, c3, b3)
        self.middle = _make_stage(c3, c4, b4)
        self.decoder3 = _make_stage(c3 + c4, c3, b3)
        self.decoder2 = _make_stage(c2 + c3, c2, b2)
        self.decoder1 = _make_stage(c1 + c2, c1, b1)
        self.decoder0 = _make_stage(c0 + c1, c0, 1)
        self.scale_heads = nn.ModuleList(
            (
                nn.Conv2d(c0, 1, 1),
                nn.Conv2d(c1, 1, 1),
                nn.Conv2d(c2, 1, 1),
                nn.Conv2d(c3, 1, 1),
            )
        )
        self.fusion = nn.Conv2d(4, 1, 3, padding=1)

    @staticmethod
    def _resize_like(inputs: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return functional.interpolate(
            inputs,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

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
            multi_scale = warm_flag
        e0 = self.encoder0(self.input_projection(inputs))
        e1 = self.encoder1(self.pool(e0))
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        middle = self.middle(self.pool(e3))
        d3 = self.decoder3(torch.cat((e3, self._resize_like(middle, e3)), dim=1))
        d2 = self.decoder2(torch.cat((e2, self._resize_like(d3, e2)), dim=1))
        d1 = self.decoder1(torch.cat((e1, self._resize_like(d2, e1)), dim=1))
        d0 = self.decoder0(torch.cat((e0, self._resize_like(d1, e0)), dim=1))
        head0 = self.scale_heads[0](d0)
        if not multi_scale:
            return MSHNetOutput(auxiliary_logits=(), logits=head0)
        heads = (
            head0,
            self.scale_heads[1](d1),
            self.scale_heads[2](d2),
            self.scale_heads[3](d3),
        )
        resized = (heads[0],) + tuple(
            self._resize_like(head, heads[0]) for head in heads[1:]
        )
        logits = self.fusion(torch.cat(resized, dim=1))
        return MSHNetOutput(auxiliary_logits=heads, logits=logits)

    def export_config(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "input_channels": self.input_channels,
            "channels": list(self.channels),
            "block_counts": list(self.block_counts),
        }


__all__ = ["CompactMSHNet"]
