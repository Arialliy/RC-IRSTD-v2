"""Risk-conditioned MSHNet detector proposed for the RC-IRSTD sprint.

RC-MSHNet preserves the canonical MSHNet encoder/decoder and multi-scale head,
then adds three lightweight, independently ablatable modules:

1. a scale-normalized local-contrast pyramid (SN-LCP),
2. cross-scale soft component-context fusion (CS-CCF), and
3. a risk-proxy gated residual fusion head (RGF).

The two residual prediction heads are zero-initialized.  Consequently, after
loading a canonical MSHNet state dict, RC-MSHNet initially produces exactly the
same logits as MSHNet.  This property enables deadline-friendly fine-tuning
without discarding an already trained MSHNet checkpoint.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from model.MSHNet import ResNet
from .mshnet import MSHNetOutput, StructuredCanonicalMSHNet

RC_MSHNET_ARCHITECTURE_VERSION_V1 = "rc-mshnet-v1"
RC_MSHNET_ARCHITECTURE_VERSION_V2 = "rc-mshnet-v2-component-role-split"
# Backwards-compatible public name. Existing callers and v1 checkpoints use
# this symbol, so it must continue to identify the frozen v1 architecture.
RC_MSHNET_ARCHITECTURE_VERSION = RC_MSHNET_ARCHITECTURE_VERSION_V1
RC_MSHNET_BACKEND = "rc_mshnet"
_RC_MSHNET_ARCHITECTURE_VERSIONS = frozenset(
    (RC_MSHNET_ARCHITECTURE_VERSION_V1, RC_MSHNET_ARCHITECTURE_VERSION_V2)
)


def _group_count(channels: int) -> int:
    """Choose the largest small GroupNorm divisor supported by ``channels``."""

    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _normalization(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_group_count(channels), channels)


def _validate_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_bool(value: object, *, name: str) -> bool:
    """Reject truthy configuration values instead of silently coercing them."""

    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _validate_architecture_version(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("architecture_version must be a string")
    if value not in _RC_MSHNET_ARCHITECTURE_VERSIONS:
        supported = ", ".join(sorted(_RC_MSHNET_ARCHITECTURE_VERSIONS))
        raise ValueError(
            f"Unsupported RC-MSHNet architecture_version {value!r}; "
            f"expected one of: {supported}"
        )
    return value


def _validate_odd_sequence(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value <= 0 or value % 2 == 0 for value in result):
        raise ValueError(f"{name} must contain positive odd integers")
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{name} must be strictly increasing without duplicates")
    return result


def _validate_dilations(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value <= 0 for value in result):
        raise ValueError("context_dilations must contain positive integers")
    if tuple(sorted(set(result))) != result:
        raise ValueError(
            "context_dilations must be strictly increasing without duplicates"
        )
    return result


class ConvNormAct(nn.Sequential):
    """Small convolution block using batch-size-independent normalization."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            _normalization(output_channels),
            nn.SiLU(inplace=True),
        )


class ScaleNormalizedContrastPyramid(nn.Module):
    """Extract source-agnostic local contrast and local noise evidence.

    For each window, the module computes a center-surround residual and divides
    it by the local standard deviation.  Both the signed and positive contrast
    are retained because background suppression and hot-target enhancement are
    complementary.  A log-standard-deviation channel exposes a differentiable
    local clutter/noise proxy to the final gate.
    """

    def __init__(
        self,
        output_channels: int,
        *,
        windows: Sequence[int] = (3, 7, 15),
        epsilon: float = 1e-4,
        z_clip: float = 8.0,
    ) -> None:
        super().__init__()
        self.windows = _validate_odd_sequence(windows, name="contrast_windows")
        if epsilon <= 0.0:
            raise ValueError("contrast_epsilon must be positive")
        if z_clip <= 0.0:
            raise ValueError("contrast_z_clip must be positive")
        self.epsilon = float(epsilon)
        self.z_clip = float(z_clip)
        input_channels = 3 * len(self.windows)
        self.project = nn.Sequential(
            ConvNormAct(input_channels, output_channels, kernel_size=1),
            ConvNormAct(
                output_channels,
                output_channels,
                kernel_size=3,
                groups=output_channels,
            ),
            nn.Conv2d(output_channels, output_channels, 1, bias=False),
            _normalization(output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 4:
            raise ValueError("inputs must have shape BxCxHxW")
        gray = inputs.mean(dim=1, keepdim=True)
        channels: list[torch.Tensor] = []
        noise_maps: list[torch.Tensor] = []
        for window in self.windows:
            mean = F.avg_pool2d(
                gray,
                kernel_size=window,
                stride=1,
                padding=window // 2,
                count_include_pad=False,
            )
            mean_square = F.avg_pool2d(
                gray.square(),
                kernel_size=window,
                stride=1,
                padding=window // 2,
                count_include_pad=False,
            )
            variance = (mean_square - mean.square()).clamp_min(0.0)
            standard_deviation = torch.sqrt(variance + self.epsilon)
            normalized = ((gray - mean) / standard_deviation).clamp(
                min=-self.z_clip,
                max=self.z_clip,
            )
            channels.extend(
                (
                    normalized,
                    F.relu(normalized),
                    torch.log1p(standard_deviation),
                )
            )
            noise_maps.append(standard_deviation)
        features = self.project(torch.cat(channels, dim=1))
        # A single spatial risk proxy is sufficient for the gate and keeps the
        # method inexpensive at native resolution.
        noise_proxy = torch.stack(noise_maps, dim=0).mean(dim=0)
        return features, noise_proxy


class DilatedDepthwiseContext(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(
                channels,
                channels,
                kernel_size=3,
                dilation=dilation,
                groups=channels,
            ),
            nn.Conv2d(channels, channels, 1, bias=False),
            _normalization(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class CrossScaleComponentContext(nn.Module):
    """Fuse decoder scales and learn a soft compact-component context.

    Connected components themselves are discrete and unsuitable inside an
    end-to-end detector.  This module uses a differentiable proxy: a learned
    seed map, local support, and a support-minus-ring fragmentation map.  The
    proxy is not presented as exact topology; it supplies compactness/clutter
    evidence that is later audited with the repository's exact component-risk
    evaluator.
    """

    def __init__(
        self,
        output_channels: int,
        *,
        decoder_channels: Sequence[int] = (16, 32, 64, 128),
        dilations: Sequence[int] = (1, 2, 4),
        support_window: int = 3,
        ring_window: int = 9,
    ) -> None:
        super().__init__()
        self.decoder_channels = tuple(int(value) for value in decoder_channels)
        if len(self.decoder_channels) != 4 or any(
            value <= 0 for value in self.decoder_channels
        ):
            raise ValueError("decoder_channels must contain four positive values")
        self.dilations = _validate_dilations(dilations)
        self.support_window = _validate_odd_sequence(
            (support_window,), name="component_support_window"
        )[0]
        self.ring_window = _validate_odd_sequence(
            (ring_window,), name="component_ring_window"
        )[0]
        if self.ring_window <= self.support_window:
            raise ValueError("component_ring_window must exceed support_window")

        self.projections = nn.ModuleList(
            ConvNormAct(channels, output_channels, kernel_size=1)
            for channels in self.decoder_channels
        )
        self.scale_fusion = ConvNormAct(
            4 * output_channels,
            output_channels,
            kernel_size=1,
        )
        self.context_branches = nn.ModuleList(
            DilatedDepthwiseContext(output_channels, dilation)
            for dilation in self.dilations
        )
        self.context_fusion = ConvNormAct(
            (1 + len(self.dilations)) * output_channels,
            output_channels,
            kernel_size=1,
        )
        self.seed_head = nn.Conv2d(output_channels, 1, 1)
        self.compactness_fusion = nn.Sequential(
            ConvNormAct(output_channels + 3, output_channels, kernel_size=3),
            nn.Conv2d(output_channels, output_channels, 1, bias=False),
            _normalization(output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        decoder_features: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(decoder_features) != 4:
            raise ValueError("decoder_features must contain x_d0, x_d1, x_d2, x_d3")
        target_size = decoder_features[0].shape[-2:]
        projected: list[torch.Tensor] = []
        for feature, projection in zip(decoder_features, self.projections):
            value = projection(feature)
            if value.shape[-2:] != target_size:
                value = F.interpolate(
                    value,
                    size=target_size,
                    mode="bilinear",
                    align_corners=True,
                )
            projected.append(value)
        fused = self.scale_fusion(torch.cat(projected, dim=1))
        contexts = [fused]
        contexts.extend(branch(fused) for branch in self.context_branches)
        context = self.context_fusion(torch.cat(contexts, dim=1))

        seed = torch.sigmoid(self.seed_head(context))
        support = F.max_pool2d(
            seed,
            kernel_size=self.support_window,
            stride=1,
            padding=self.support_window // 2,
        )
        ring_average = F.avg_pool2d(
            seed,
            kernel_size=self.ring_window,
            stride=1,
            padding=self.ring_window // 2,
            count_include_pad=False,
        )
        fragmentation = F.relu(support - ring_average)
        refined = self.compactness_fusion(
            torch.cat((context, seed, support, fragmentation), dim=1)
        )
        return refined, fragmentation


class RiskGatedResidualFusion(nn.Module):
    """Conservatively fuse contrast and component corrections into MSHNet.

    The third softmax expert is a no-op expert.  Its positive initial bias makes
    early fine-tuning conservative, while zero-initialized correction heads
    make the exact initial detector output equal to the MSHNet baseline.
    """

    def __init__(
        self,
        feature_channels: int,
        *,
        decoder_channels: int = 16,
        use_risk_gate: bool = True,
        initial_noop_bias: float = 2.0,
    ) -> None:
        super().__init__()
        self.use_risk_gate = bool(use_risk_gate)
        self.decoder_projection = ConvNormAct(
            decoder_channels,
            feature_channels,
            kernel_size=1,
        )
        branch_channels = 2 * feature_channels
        self.contrast_delta = nn.Sequential(
            ConvNormAct(branch_channels, feature_channels, kernel_size=3),
            nn.Conv2d(feature_channels, 1, 1),
        )
        self.component_delta = nn.Sequential(
            ConvNormAct(branch_channels, feature_channels, kernel_size=3),
            nn.Conv2d(feature_channels, 1, 1),
        )
        gate_channels = 3 * feature_channels + 4
        self.gate = nn.Sequential(
            ConvNormAct(gate_channels, feature_channels, kernel_size=3),
            nn.Conv2d(feature_channels, 3, 1),
        )
        self.raw_residual_gain = nn.Parameter(torch.tensor(0.0))
        self._zero_initialize_prediction_heads(initial_noop_bias)

    def _zero_initialize_prediction_heads(self, initial_noop_bias: float) -> None:
        for head in (self.contrast_delta[-1], self.component_delta[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        gate_output = self.gate[-1]
        nn.init.zeros_(gate_output.weight)
        nn.init.zeros_(gate_output.bias)
        with torch.no_grad():
            gate_output.bias[2] = float(initial_noop_bias)

    def forward(
        self,
        decoder_feature: torch.Tensor,
        contrast_feature: torch.Tensor,
        component_feature: torch.Tensor,
        *,
        base_logits: torch.Tensor,
        noise_proxy: torch.Tensor,
        component_proxy: torch.Tensor,
        contrast_enabled: bool,
        component_enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not contrast_enabled and not component_enabled:
            zeros = torch.zeros_like(base_logits)
            no_op = torch.ones_like(base_logits)
            gates = torch.cat((zeros, zeros, no_op), dim=1)
            return base_logits, zeros, zeros, gates

        decoder = self.decoder_projection(decoder_feature)
        contrast_delta = (
            self.contrast_delta(torch.cat((decoder, contrast_feature), dim=1))
            if contrast_enabled
            else torch.zeros_like(base_logits)
        )
        component_delta = (
            self.component_delta(torch.cat((decoder, component_feature), dim=1))
            if component_enabled
            else torch.zeros_like(base_logits)
        )

        if self.use_risk_gate:
            disagreement = (contrast_delta - component_delta).abs()
            gate_inputs = torch.cat(
                (
                    decoder,
                    contrast_feature,
                    component_feature,
                    noise_proxy,
                    component_proxy,
                    disagreement,
                    base_logits.abs(),
                ),
                dim=1,
            )
            gate_logits = self.gate(gate_inputs)
            # Disabled experts receive an effectively impossible logit.  The
            # explicit no-op expert always remains available.
            masks: list[torch.Tensor] = []
            for enabled in (contrast_enabled, component_enabled, True):
                fill = 0.0 if enabled else -1.0e4
                masks.append(torch.full_like(gate_logits[:, :1], fill))
            gate_logits = gate_logits + torch.cat(masks, dim=1)
            gates = torch.softmax(gate_logits, dim=1)
        else:
            active = float(int(contrast_enabled) + int(component_enabled))
            contrast_gate = torch.full_like(
                base_logits, 1.0 / active if contrast_enabled else 0.0
            )
            component_gate = torch.full_like(
                base_logits, 1.0 / active if component_enabled else 0.0
            )
            no_op_gate = torch.zeros_like(base_logits)
            gates = torch.cat((contrast_gate, component_gate, no_op_gate), dim=1)

        # 2*sigmoid(0)=1 and bounds the learned correction multiplier in (0, 2).
        gain = 2.0 * torch.sigmoid(self.raw_residual_gain)
        fused = base_logits + gain * (
            gates[:, 0:1] * contrast_delta + gates[:, 1:2] * component_delta
        )
        return fused, contrast_delta, component_delta, gates


class RCMSHNet(StructuredCanonicalMSHNet):
    """Canonical MSHNet plus contrast, component-context, and risk gating."""

    backend = RC_MSHNET_BACKEND
    architecture_version = RC_MSHNET_ARCHITECTURE_VERSION
    extension_prefixes = (
        "contrast_pyramid.",
        "component_context.",
        "fusion_head.",
    )

    def __init__(
        self,
        input_channels: int = 3,
        *,
        block: type[nn.Module] = ResNet,
        fusion_channels: int = 16,
        contrast_windows: Sequence[int] = (3, 7, 15),
        context_dilations: Sequence[int] = (1, 2, 4),
        component_support_window: int = 3,
        component_ring_window: int = 9,
        use_contrast: bool = True,
        use_component_context: bool = True,
        use_component_expert: bool | None = None,
        use_risk_gate: bool = True,
        expose_branch_auxiliary: bool = True,
        architecture_version: str = RC_MSHNET_ARCHITECTURE_VERSION_V1,
    ) -> None:
        super().__init__(input_channels=input_channels, block=block)
        self.architecture_version = _validate_architecture_version(
            architecture_version
        )
        self.fusion_channels = _validate_positive_int(
            fusion_channels, name="fusion_channels"
        )
        self.contrast_windows = _validate_odd_sequence(
            contrast_windows, name="contrast_windows"
        )
        self.context_dilations = _validate_dilations(context_dilations)
        self.component_support_window = int(component_support_window)
        self.component_ring_window = int(component_ring_window)
        self.use_contrast = _validate_bool(use_contrast, name="use_contrast")
        self.use_component_context = _validate_bool(
            use_component_context, name="use_component_context"
        )
        self.use_risk_gate = _validate_bool(use_risk_gate, name="use_risk_gate")
        self.expose_branch_auxiliary = _validate_bool(
            expose_branch_auxiliary, name="expose_branch_auxiliary"
        )

        if self.architecture_version == RC_MSHNET_ARCHITECTURE_VERSION_V1:
            if use_component_expert is not None:
                raise ValueError(
                    "rc-mshnet-v1 does not define use_component_expert; omit "
                    "the field to preserve the frozen v1 component semantics"
                )
            # Deliberately derived rather than exported: this preserves both
            # the old v1 behavior and its byte-for-byte config contract.
            self.use_component_expert = self.use_component_context
        else:
            if use_component_expert is None:
                raise ValueError(
                    "rc-mshnet-v2-component-role-split requires an explicit "
                    "boolean use_component_expert"
                )
            self.use_component_expert = _validate_bool(
                use_component_expert, name="use_component_expert"
            )

        if self.use_component_expert and not self.use_component_context:
            raise ValueError(
                "use_component_expert=true requires use_component_context=true"
            )
        if self.use_component_context and not self.use_component_expert:
            if not self.use_contrast:
                raise ValueError(
                    "component context without a component expert requires "
                    "use_contrast=true; otherwise the context is a dead branch"
                )
            if not self.use_risk_gate:
                raise ValueError(
                    "component context without a component expert requires "
                    "use_risk_gate=true; otherwise the context is a dead branch"
                )

        self.contrast_pyramid = ScaleNormalizedContrastPyramid(
            self.fusion_channels,
            windows=self.contrast_windows,
        )
        self.component_context = CrossScaleComponentContext(
            self.fusion_channels,
            dilations=self.context_dilations,
            support_window=self.component_support_window,
            ring_window=self.component_ring_window,
        )
        self.fusion_head = RiskGatedResidualFusion(
            self.fusion_channels,
            use_risk_gate=self.use_risk_gate,
        )

    def _forward_backbone(
        self,
        inputs: torch.Tensor,
        *,
        multi_scale: bool,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        # This is intentionally the same operation order as model.MSHNet.MSHNet
        # so a loaded baseline state produces byte-for-byte-compatible logits.
        x_e0 = self.encoder_0(self.conv_init(inputs))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))
        x_m = self.middle_layer(self.pool(x_e3))
        x_d3 = self.decoder_3(torch.cat((x_e3, self.up(x_m)), dim=1))
        x_d2 = self.decoder_2(torch.cat((x_e2, self.up(x_d3)), dim=1))
        x_d1 = self.decoder_1(torch.cat((x_e1, self.up(x_d2)), dim=1))
        x_d0 = self.decoder_0(torch.cat((x_e0, self.up(x_d1)), dim=1))

        if multi_scale:
            mask0 = self.output_0(x_d0)
            mask1 = self.output_1(x_d1)
            mask2 = self.output_2(x_d2)
            mask3 = self.output_3(x_d3)
            base_logits = self.final(
                torch.cat(
                    (
                        mask0,
                        self.up(mask1),
                        self.up_4(mask2),
                        self.up_8(mask3),
                    ),
                    dim=1,
                )
            )
            auxiliary = (mask0, mask1, mask2, mask3)
        else:
            base_logits = self.output_0(x_d0)
            auxiliary = ()
        return auxiliary, base_logits, (x_d0, x_d1, x_d2, x_d3)

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
        if not isinstance(multi_scale, bool):
            raise TypeError("multi_scale must be boolean")
        if inputs.ndim != 4 or inputs.shape[1] != self.input_channels:
            raise ValueError(
                f"inputs must have shape Bx{self.input_channels}xHxW"
            )
        if any(int(value) % 16 != 0 for value in inputs.shape[-2:]):
            raise ValueError("RC-MSHNet input height and width must be multiples of 16")

        auxiliary, base_logits, decoder_features = self._forward_backbone(
            inputs,
            multi_scale=multi_scale,
        )
        x_d0 = decoder_features[0]
        zeros = x_d0.new_zeros(
            x_d0.shape[0], self.fusion_channels, *x_d0.shape[-2:]
        )
        scalar_zeros = base_logits.new_zeros(base_logits.shape)

        if self.use_contrast:
            contrast_feature, noise_proxy = self.contrast_pyramid(inputs)
        else:
            contrast_feature, noise_proxy = zeros, scalar_zeros
        if self.use_component_context:
            component_feature, component_proxy = self.component_context(
                decoder_features
            )
        else:
            component_feature, component_proxy = zeros, scalar_zeros

        logits, contrast_delta, component_delta, _ = self.fusion_head(
            x_d0,
            contrast_feature,
            component_feature,
            base_logits=base_logits,
            noise_proxy=noise_proxy,
            component_proxy=component_proxy,
            contrast_enabled=self.use_contrast,
            component_enabled=self.use_component_expert,
        )

        output_auxiliary = list(auxiliary)
        if self.expose_branch_auxiliary:
            # Detaching the baseline makes these terms train each correction as
            # a residual target without multiplying gradients into MSHNet.
            if self.use_contrast:
                output_auxiliary.append(base_logits.detach() + contrast_delta)
            if self.use_component_expert:
                output_auxiliary.append(base_logits.detach() + component_delta)
        return MSHNetOutput(tuple(output_auxiliary), logits)

    def export_config(self) -> dict[str, object]:
        config: dict[str, object] = {
            "architecture_version": self.architecture_version,
            "backend": self.backend,
            "input_channels": self.input_channels,
            "channels": [16, 32, 64, 128, 256],
            "block_counts": [2, 2, 2, 2],
            "fusion_channels": self.fusion_channels,
            "contrast_windows": list(self.contrast_windows),
            "context_dilations": list(self.context_dilations),
            "component_support_window": self.component_support_window,
            "component_ring_window": self.component_ring_window,
            "use_contrast": self.use_contrast,
            "use_component_context": self.use_component_context,
            "use_risk_gate": self.use_risk_gate,
            "expose_branch_auxiliary": self.expose_branch_auxiliary,
            "baseline_identity": "canonical_mshnet",
            "initialization_contract": "zero_residual_exact_mshnet",
        }
        # Adding this key to v1 would change its frozen export/checkpoint
        # contract. Only v2 makes the separated role explicit.
        if self.architecture_version == RC_MSHNET_ARCHITECTURE_VERSION_V2:
            config["use_component_expert"] = self.use_component_expert
        return config


def build_rc_mshnet(config: Mapping[str, object] | None = None) -> RCMSHNet:
    """Build RC-MSHNet from the repository's ``model`` config mapping."""

    values = dict(config or {})
    architecture_version = _validate_architecture_version(
        values.get(
            "architecture_version", RC_MSHNET_ARCHITECTURE_VERSION_V1
        )
    )
    has_component_expert = "use_component_expert" in values
    if architecture_version == RC_MSHNET_ARCHITECTURE_VERSION_V1:
        if has_component_expert:
            raise ValueError(
                "use_component_expert is not valid for rc-mshnet-v1; declare "
                "architecture_version=rc-mshnet-v2-component-role-split"
            )
        component_expert: bool | None = None
    else:
        if not has_component_expert:
            raise ValueError(
                "rc-mshnet-v2-component-role-split requires the explicit "
                "use_component_expert field"
            )
        component_expert = _validate_bool(
            values["use_component_expert"], name="use_component_expert"
        )

    channels = tuple(int(value) for value in values.get(
        "channels", (16, 32, 64, 128, 256)
    ))
    block_counts = tuple(int(value) for value in values.get(
        "block_counts", (2, 2, 2, 2)
    ))
    if channels != (16, 32, 64, 128, 256) or block_counts != (2, 2, 2, 2):
        raise ValueError(
            "RC-MSHNet preserves canonical MSHNet channels/block_counts; "
            "architecture ablations must modify only the RC-MSHNet modules"
        )
    return RCMSHNet(
        input_channels=int(values.get("input_channels", 3)),
        fusion_channels=int(values.get("fusion_channels", 16)),
        contrast_windows=tuple(
            int(value) for value in values.get("contrast_windows", (3, 7, 15))
        ),
        context_dilations=tuple(
            int(value) for value in values.get("context_dilations", (1, 2, 4))
        ),
        component_support_window=int(values.get("component_support_window", 3)),
        component_ring_window=int(values.get("component_ring_window", 9)),
        use_contrast=_validate_bool(
            values.get("use_contrast", True), name="use_contrast"
        ),
        use_component_context=_validate_bool(
            values.get("use_component_context", True),
            name="use_component_context",
        ),
        use_component_expert=component_expert,
        use_risk_gate=_validate_bool(
            values.get("use_risk_gate", True), name="use_risk_gate"
        ),
        expose_branch_auxiliary=_validate_bool(
            values.get("expose_branch_auxiliary", True),
            name="expose_branch_auxiliary",
        ),
        architecture_version=architecture_version,
    )


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    value = payload
    if isinstance(value, Mapping):
        for key in (
            "model_state",
            "net",
            "state_dict",
            "model_state_dict",
            "model",
        ):
            candidate = value.get(key)
            if isinstance(candidate, Mapping):
                value = candidate
                break
    if not isinstance(value, Mapping) or not value:
        raise ValueError("checkpoint does not contain a model state dictionary")
    if not all(isinstance(tensor, torch.Tensor) for tensor in value.values()):
        raise ValueError("resolved checkpoint state contains non-tensor values")
    state: dict[str, torch.Tensor] = {}
    for raw_key, tensor in value.items():
        key = str(raw_key)
        if key.startswith("module."):
            key = key[len("module.") :]
        if key in state:
            raise ValueError("checkpoint key normalization produced duplicates")
        state[key] = tensor
    return state


def _checkpoint_model_config(payload: Any) -> dict[str, object] | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("model_config")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint model_config must be a mapping when present")
    return dict(value)


def _validate_canonical_checkpoint_identity(payload: Any) -> dict[str, object]:
    """Fail closed when metadata identifies an RC-MSHNet checkpoint."""

    config = _checkpoint_model_config(payload)
    if config is None:
        return {
            "model_config_present": False,
            "resolved_identity": "canonical_mshnet_raw_state",
        }
    backend = config.get("backend")
    if not isinstance(backend, str):
        raise RuntimeError(
            "Checkpoint model_config must explicitly identify a canonical backend"
        )
    normalized_backend = backend.lower()
    if normalized_backend not in {"canonical", "upstream", "mshnet"}:
        raise RuntimeError(
            "Canonical MSHNet initialization rejects checkpoint model_config "
            f"backend={backend!r}"
        )
    if config.get("architecture_version") is not None:
        raise RuntimeError(
            "Canonical MSHNet initialization rejects model_config containing "
            "an architecture_version"
        )
    if isinstance(payload, Mapping):
        for key in ("model_architecture_version", "architecture_version"):
            if payload.get(key) is not None:
                raise RuntimeError(
                    "Canonical MSHNet initialization rejects checkpoint "
                    f"metadata field {key}"
                )
    return {
        "model_config_present": True,
        "resolved_identity": "canonical_mshnet",
        "backend": normalized_backend,
    }


def rc_mshnet_extension_state_sha256(model: nn.Module) -> str:
    """Hash every RC extension parameter/buffer in stable key order."""

    if not isinstance(model, RCMSHNet):
        raise TypeError("extension state hashing is supported only for RCMSHNet")
    state = model.state_dict()
    extension_keys = sorted(
        key
        for key in state
        if any(key.startswith(prefix) for prefix in model.extension_prefixes)
    )
    if not extension_keys:
        raise RuntimeError("RC-MSHNet has no extension state to hash")
    digest = hashlib.sha256()
    for key in extension_keys:
        tensor = state[key].detach().cpu().contiguous()
        fields = (
            key.encode("utf-8"),
            str(tensor.dtype).encode("ascii"),
            repr(tuple(tensor.shape)).encode("ascii"),
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes(),
        )
        for field in fields:
            digest.update(len(field).to_bytes(8, byteorder="big"))
            digest.update(field)
    return digest.hexdigest()


def initialize_rc_mshnet_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load canonical MSHNet weights while requiring full backbone coverage."""

    if not isinstance(model, RCMSHNet):
        raise TypeError("MSHNet initialization is supported only for RCMSHNet")
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception as error:
        raise RuntimeError(
            "Restricted initialization loading failed. Convert a locally trusted "
            "trainer checkpoint first with scripts/extract_mshnet_weights.py; "
            "never enable unsafe pickle loading for an untrusted checkpoint."
        ) from error
    source_identity = _validate_canonical_checkpoint_identity(payload)
    initial_extension_state_sha256 = rc_mshnet_extension_state_sha256(model)
    expected_extension_keys = sorted(
        key
        for key in model.state_dict()
        if any(key.startswith(prefix) for prefix in model.extension_prefixes)
    )
    state = _extract_state_dict(payload)
    extension_source_keys = sorted(
        key
        for key in state
        if any(key.startswith(prefix) for prefix in model.extension_prefixes)
    )
    if extension_source_keys:
        raise RuntimeError(
            "Canonical MSHNet initialization must not contain RC-MSHNet extension "
            f"weights: {extension_source_keys[:5]}"
        )
    expected_backbone_keys = {
        key
        for key in model.state_dict()
        if not any(key.startswith(prefix) for prefix in model.extension_prefixes)
    }
    state_keys = set(state)
    missing_source_keys = sorted(expected_backbone_keys - state_keys)
    unexpected_source_keys = sorted(state_keys - expected_backbone_keys)
    if missing_source_keys or unexpected_source_keys:
        raise RuntimeError(
            "Checkpoint is not an exact canonical MSHNet state: "
            f"missing={missing_source_keys[:5]}, unexpected={unexpected_source_keys[:5]}"
        )
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    missing_backbone = [
        key
        for key in missing
        if not any(key.startswith(prefix) for prefix in model.extension_prefixes)
    ]
    if missing_backbone or unexpected:
        raise RuntimeError(
            "Checkpoint is not a complete canonical MSHNet initialization: "
            f"missing_backbone={missing_backbone}, unexpected={unexpected}"
        )
    if sorted(missing) != expected_extension_keys:
        raise RuntimeError(
            "Canonical initialization did not preserve the complete RC-MSHNet "
            "extension state"
        )
    current_extension_state_sha256 = rc_mshnet_extension_state_sha256(model)
    if current_extension_state_sha256 != initial_extension_state_sha256:
        raise RuntimeError(
            "Canonical initialization unexpectedly modified RC-MSHNet "
            "extension weights"
        )
    loaded_keys = sorted(set(model.state_dict()).intersection(state))
    return {
        "schema_version": 1,
        "initialization_type": "canonical_mshnet_to_rc_mshnet",
        "source_path": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_checkpoint_identity": source_identity,
        "target_architecture_version": model.architecture_version,
        "initial_extension_key_count": len(expected_extension_keys),
        "initial_extension_state_sha256": initial_extension_state_sha256,
        "extension_state_preserved": True,
        "loaded_key_count": len(loaded_keys),
        "missing_extension_keys": missing,
        "unexpected_keys": unexpected,
        "backbone_fully_loaded": True,
        "zero_residual_identity_preserved": True,
    }


__all__ = [
    "RCMSHNet",
    "RC_MSHNET_ARCHITECTURE_VERSION",
    "RC_MSHNET_ARCHITECTURE_VERSION_V1",
    "RC_MSHNET_ARCHITECTURE_VERSION_V2",
    "RC_MSHNET_BACKEND",
    "ScaleNormalizedContrastPyramid",
    "CrossScaleComponentContext",
    "RiskGatedResidualFusion",
    "build_rc_mshnet",
    "initialize_rc_mshnet_from_checkpoint",
    "rc_mshnet_extension_state_sha256",
]
