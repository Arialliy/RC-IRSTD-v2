#!/usr/bin/env python3
"""Fail-closed preflight for the outer=NUAA RC-MSHNet Phase-2 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PHASE2_CASES = {
    "mshnet_ft_matched_control": (
        "phase2_mshnet_ft_outer_nuaa.yaml",
        (False, False, False, False),
    ),
    "rc_mshnet_full": (
        "phase2_rc_mshnet_full_outer_nuaa.yaml",
        (True, True, True, False),
    ),
    "rc_mshnet_no_contrast": (
        "phase2_rc_mshnet_no_contrast_outer_nuaa.yaml",
        (False, True, True, False),
    ),
    "rc_mshnet_no_component": (
        "phase2_rc_mshnet_no_component_outer_nuaa.yaml",
        (True, False, True, False),
    ),
    "rc_mshnet_no_gate_fixed_average": (
        "phase2_rc_mshnet_no_gate_outer_nuaa.yaml",
        (True, True, False, False),
    ),
    "rc_mshnet_branch_aux": (
        "phase2_rc_mshnet_branch_aux_outer_nuaa.yaml",
        (True, True, True, True),
    ),
}
MODEL_FLAGS = (
    "use_contrast",
    "use_component_context",
    "use_risk_gate",
    "expose_branch_auxiliary",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return payload


def validate_phase2_configs(config_dir: Path | None = None) -> dict[str, str]:
    directory = (config_dir or PROJECT_ROOT / "configs").resolve()
    loaded: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for role, (filename, expected_flags) in PHASE2_CASES.items():
        path = directory / filename
        config = _load_yaml(path)
        loaded[role] = config
        hashes[role] = sha256_file(path)
        identity = config.get("experiment_identity")
        if not isinstance(identity, Mapping):
            raise ValueError(f"{filename} has no experiment_identity")
        if identity.get("run_role") != role:
            raise ValueError(f"{filename} run_role does not match {role}")
        if identity.get("base_loss_id") != "stable_sls_v1":
            raise ValueError(f"{filename} is not explicitly StableSLS-v1")
        if bool(identity.get("target_labels_used_for_training", True)):
            raise ValueError(f"{filename} permits target labels during training")
        model = config.get("model")
        if not isinstance(model, Mapping) or model.get("backend") != "rc_mshnet":
            raise ValueError(f"{filename} must use backend=rc_mshnet")
        actual_flags = tuple(bool(model.get(flag)) for flag in MODEL_FLAGS)
        if actual_flags != expected_flags:
            raise ValueError(
                f"{filename} flags {actual_flags} do not match {expected_flags}"
            )
        loss = config.get("loss")
        if not isinstance(loss, Mapping):
            raise ValueError(f"{filename} has no loss mapping")
        if any(float(loss.get(key, float("nan"))) != 0.0 for key in (
            "lambda_tail",
            "lambda_miss",
            "lambda_margin",
        )):
            raise ValueError(f"{filename} enables Tail/Miss/Margin")
        sls = loss.get("sls_kwargs")
        if not isinstance(sls, Mapping) or dict(sls) != {
            "bce_weight": 0.5,
            "iou_weight": 1.0,
            "location_weight": 0.25,
            "max_positive_weight": 50.0,
        }:
            raise ValueError(f"{filename} does not use StableSLS-v1 weights")

    reference = loaded["rc_mshnet_full"]
    for role, config in loaded.items():
        for section in ("seed", "deterministic", "device", "data", "loss", "optimizer", "training"):
            if config.get(section) != reference.get(section):
                raise ValueError(f"{role} differs from full in shared section {section}")
        reference_model = dict(reference["model"])
        actual_model = dict(config["model"])
        for flag in MODEL_FLAGS:
            reference_model.pop(flag)
            actual_model.pop(flag)
        if actual_model != reference_model:
            raise ValueError(f"{role} changes non-ablation model fields")
    outputs = [str(config["output_dir"]) for config in loaded.values()]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Phase-2 output directories are not unique")
    if loaded["rc_mshnet_full"]["model"]["expose_branch_auxiliary"] is not False:
        raise ValueError("formal full configuration enables BranchAux")
    if loaded["rc_mshnet_no_gate_fixed_average"]["experiment_identity"]["run_role"] != (
        "rc_mshnet_no_gate_fixed_average"
    ):
        raise ValueError("use_risk_gate=false must be named fixed-average")
    return hashes


def assert_sentinel_allows(state_dir: Path) -> None:
    hold = state_dir / "HOLD_RC_MSHNET_GATE"
    allow = state_dir / "ALLOW_RC_MSHNET_GATE"
    if hold.exists() or not allow.exists():
        raise RuntimeError(
            "RC-MSHNet gate is blocked: HOLD exists or ALLOW is absent"
        )


def validate_initializer(
    initializer: Path,
    *,
    source_manifest: Path,
) -> dict[str, str]:
    initializer = initializer.resolve()
    manifest = _load_json(source_manifest.resolve())
    if manifest.get("artifact_state") != "finalized":
        raise ValueError("source baseline identity is not finalized")
    source_sha256 = manifest.get("checkpoint_sha256")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise ValueError("source baseline has no final checkpoint SHA256")
    payload = torch.load(initializer, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("initializer root is not a mapping")
    if payload.get("kind") != "mshnet_tensor_only_initialization":
        raise ValueError("initializer kind is not tensor-only MSHNet")
    if payload.get("source_checkpoint_sha256") != source_sha256:
        raise ValueError("initializer does not bind the finalized source checkpoint")
    identity = payload.get("source_training_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("initializer has no source training identity")
    expected_identity = {
        "architecture_id": "canonical_mshnet",
        "model_backend": "canonical",
        "loss_id": "stable_sls_v1",
        "loss_implementation": "rc_irstd.losses.sls.StableSLSLoss",
        "auxiliary_weight": 0.25,
        "lambda_tail": 0.0,
        "lambda_miss": 0.0,
        "lambda_margin": 0.0,
        "source_names": ["NUDT-SIRST", "IRSTD-1K"],
        "train_split": "train",
        "diagnostic_test_eval": False,
        "checkpoint_policy": "fixed_last",
        "warmup_epochs": 5,
        "configured_epochs": 400,
        "checkpoint_epoch": 399,
        "inference_head": "multi_scale_fused",
        "target_test_labels_used_for_checkpoint_selection": False,
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise ValueError(
                f"initializer identity {key}: expected {expected!r}, "
                f"received {identity.get(key)!r}"
            )
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("initializer has no model state")
    extension_prefixes = ("contrast_pyramid.", "component_context.", "fusion_head.")
    if any(str(key).startswith(extension_prefixes) for key in state):
        raise ValueError("initializer contains RC-MSHNet extension weights")
    return {
        "initializer_sha256": sha256_file(initializer),
        "source_checkpoint_sha256": source_sha256,
    }


def validate_gate_ready(
    *,
    state_dir: Path,
    initializer: Path,
    source_manifest: Path,
) -> dict[str, Any]:
    assert_sentinel_allows(state_dir.resolve())
    config_hashes = validate_phase2_configs()
    initializer_report = validate_initializer(
        initializer,
        source_manifest=source_manifest,
    )
    return {
        "status": "passed",
        "config_sha256": config_hashes,
        **initializer_report,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_gate_ready(
        state_dir=args.state_dir,
        initializer=args.initializer,
        source_manifest=args.source_manifest,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
