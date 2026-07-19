"""Architecture-only smoke test; no dataset is opened."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from model.MSHNet import MSHNet
from rc_irstd.models.rc_mshnet import (
    RCMSHNet,
    initialize_rc_mshnet_from_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--fusion-channels", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.size < 16 or args.size % 16 != 0:
        raise ValueError("--size must be a positive multiple of 16")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    torch.manual_seed(23)
    baseline = MSHNet(3).to(device).eval()
    model = RCMSHNet(fusion_channels=args.fusion_channels).to(device).eval()
    incompatible = model.load_state_dict(baseline.state_dict(), strict=False)
    inputs = torch.randn(args.batch_size, 3, args.size, args.size, device=device)
    with torch.no_grad():
        _, baseline_logits = baseline(inputs, True)
        output = model(inputs, multi_scale=True)
    maximum_delta = float((baseline_logits - output.logits).abs().max().cpu())
    if maximum_delta != 0.0:
        raise RuntimeError(f"zero-residual identity failed: max delta={maximum_delta}")

    model.train()
    loss = model(inputs, multi_scale=True).logits.square().mean()
    loss.backward()
    contrast_gradient = float(
        model.fusion_head.contrast_delta[-1].weight.grad.abs().sum().cpu()
    )
    component_gradient = float(
        model.fusion_head.component_delta[-1].weight.grad.abs().sum().cpu()
    )
    if contrast_gradient <= 0.0 or component_gradient <= 0.0:
        raise RuntimeError("one or more correction heads received no gradient")

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "mshnet.pt"
        torch.save({"model_state": baseline.state_dict()}, checkpoint)
        load_report = initialize_rc_mshnet_from_checkpoint(
            RCMSHNet(fusion_channels=args.fusion_channels), checkpoint
        )

    baseline_parameters = sum(value.numel() for value in baseline.parameters())
    proposed_parameters = sum(value.numel() for value in model.parameters())
    report = {
        "status": "passed",
        "device": str(device),
        "input_shape": list(inputs.shape),
        "output_shape": list(output.logits.shape),
        "auxiliary_count": len(output.auxiliary_logits),
        "zero_residual_max_abs_delta": maximum_delta,
        "contrast_head_gradient_l1": contrast_gradient,
        "component_head_gradient_l1": component_gradient,
        "baseline_parameters": baseline_parameters,
        "proposed_parameters": proposed_parameters,
        "parameter_overhead_percent": 100.0
        * (proposed_parameters - baseline_parameters)
        / baseline_parameters,
        "missing_extension_key_count": len(incompatible.missing_keys),
        "initializer_backbone_fully_loaded": load_report[
            "backbone_fully_loaded"
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
