"""Architecture-only smoke test; no dataset is opened."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch

# Keep the documented ``python scripts/smoke_rc_mshnet.py`` entry point
# self-contained instead of requiring callers to inject PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional tensor-only MSHNet initializer to verify byte-exact identity",
    )
    return parser


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    value: object = payload
    if isinstance(value, Mapping):
        for key in ("model_state", "net", "state_dict", "model_state_dict", "model"):
            candidate = value.get(key)
            if isinstance(candidate, Mapping):
                value = candidate
                break
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"checkpoint does not contain a model state: {path}")
    if not all(isinstance(tensor, torch.Tensor) for tensor in value.values()):
        raise ValueError(f"checkpoint state contains non-tensor values: {path}")
    return {str(key): tensor for key, tensor in value.items()}


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
    model = RCMSHNet(
        fusion_channels=args.fusion_channels,
        expose_branch_auxiliary=False,
    ).to(device).eval()
    checkpoint_sha256: str | None = None
    if args.checkpoint is None:
        incompatible = model.load_state_dict(baseline.state_dict(), strict=False)
        load_report: dict[str, object] | None = None
    else:
        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        state = _checkpoint_state(checkpoint)
        baseline.load_state_dict(state, strict=True)
        load_report = initialize_rc_mshnet_from_checkpoint(model, checkpoint, device=device)
        incompatible = model.load_state_dict(state, strict=False)
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    inputs = torch.randn(args.batch_size, 3, args.size, args.size, device=device)
    with torch.no_grad():
        _, baseline_logits = baseline(inputs, True)
        output = model(inputs, multi_scale=True)
    maximum_delta = float((baseline_logits - output.logits).abs().max().cpu())
    if maximum_delta != 0.0:
        raise RuntimeError(f"zero-residual identity failed: max delta={maximum_delta}")
    if len(output.auxiliary_logits) != 4:
        raise RuntimeError(
            "formal RC-MSHNet smoke requires the four canonical auxiliary logits; "
            f"received {len(output.auxiliary_logits)}"
        )

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

    if load_report is None:
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
        "checkpoint_sha256": checkpoint_sha256,
        "initializer_backbone_fully_loaded": load_report[
            "backbone_fully_loaded"
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
