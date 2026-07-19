"""Regression tests for the AAAI-27 RC-MSHNet fairness contract."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from model.MSHNet import MSHNet
from rc_irstd.models import build_mshnet, forward_mshnet
from rc_irstd.models.rc_mshnet import (
    RCMSHNet,
    initialize_rc_mshnet_from_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    (
        "nuaa",
        "NUAA-SIRST",
        ("NUDT-SIRST", "IRSTD-1K"),
    ),
    (
        "nudt",
        "NUDT-SIRST",
        ("NUAA-SIRST", "IRSTD-1K"),
    ),
    (
        "irstd",
        "IRSTD-1K",
        ("NUAA-SIRST", "NUDT-SIRST"),
    ),
)
EXTENSION_FLAGS = {
    "use_contrast",
    "use_component_context",
    "use_risk_gate",
}
TARGET_STAGE_DEFAULT = {
    "selection_scores": "scores_unlabeled",
    "selection_labels_loaded": False,
    "selection_freeze_required": True,
    "labeled_audit_scores": "scores_labeled_audit",
    "labeled_audit_after_freeze": True,
}
PIPELINE_CASES = (
    (
        "nuaa",
        "NUAA-SIRST",
        ("NUDT-SIRST", "IRSTD-1K"),
        "../artifacts/aaai27/initializers/"
        "mshnet_seed42_train_nudt_irstd_tensor_only.pt",
        {
            "NUDT-SIRST": "../artifacts/aaai27/initializers/"
            "mshnet_seed42_train_irstd_tensor_only.pt",
            "IRSTD-1K": "../artifacts/aaai27/initializers/"
            "mshnet_seed42_train_nudt_tensor_only.pt",
        },
    ),
    (
        "nudt",
        "NUDT-SIRST",
        ("NUAA-SIRST", "IRSTD-1K"),
        "../artifacts/aaai27/initializers/"
        "mshnet_seed42_train_nuaa_irstd_tensor_only.pt",
        {
            "NUAA-SIRST": "../artifacts/aaai27/initializers/"
            "mshnet_seed42_train_irstd_tensor_only.pt",
            "IRSTD-1K": "../artifacts/aaai27/initializers/"
            "mshnet_seed42_train_nuaa_tensor_only.pt",
        },
    ),
    (
        "irstd",
        "IRSTD-1K",
        ("NUAA-SIRST", "NUDT-SIRST"),
        "../artifacts/aaai27/initializers/"
        "mshnet_seed42_train_nuaa_nudt_tensor_only.pt",
        {
            "NUAA-SIRST": "../artifacts/aaai27/initializers/"
            "mshnet_seed42_train_nudt_tensor_only.pt",
            "NUDT-SIRST": "../artifacts/aaai27/initializers/"
            "mshnet_seed42_train_nuaa_tensor_only.pt",
        },
    ),
)
SOURCE_INITIALIZER_CASES = (
    (
        "nuaa",
        "NUAA-SIRST",
        "../outputs/aaai27_detectors/source_initializers/"
        "train_nuaa_mshnet_sls_seed42",
    ),
    (
        "nudt",
        "NUDT-SIRST",
        "../outputs/aaai27_detectors/source_initializers/"
        "train_nudt_mshnet_sls_seed42",
    ),
    (
        "irstd",
        "IRSTD-1K",
        "../outputs/aaai27_detectors/source_initializers/"
        "train_irstd_mshnet_sls_seed42",
    ),
)


def _load(filename: str) -> dict[str, object]:
    payload = yaml.safe_load(
        (ROOT / "configs" / filename).read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    return payload


def _paired_configs(target_key: str) -> tuple[dict[str, object], dict[str, object]]:
    full = _load(f"detector_rc_mshnet_outer_{target_key}_fast.yaml")
    control = _load(f"detector_mshnet_ft_outer_{target_key}.yaml")
    return full, control


@pytest.mark.parametrize("target_key,target_name,source_names", CASES)
def test_full_and_matched_control_configs_are_source_correct_and_budget_matched(
    target_key: str,
    target_name: str,
    source_names: tuple[str, str],
) -> None:
    full, control = _paired_configs(target_key)

    expected_sources = [
        {"name": name, "path": f"../datasets/{name}"} for name in source_names
    ]
    assert full["data"]["sources"] == expected_sources
    assert control["data"]["sources"] == expected_sources
    assert target_name not in {entry["name"] for entry in expected_sources}
    assert full["data"] == control["data"]

    assert full["seed"] == control["seed"] == 42
    assert full["deterministic"] is control["deterministic"] is True
    assert full["device"] == control["device"] == "auto"
    assert full["loss"] == control["loss"]
    assert full["optimizer"] == control["optimizer"]
    assert full["training"] == control["training"]
    assert full["training"]["epochs"] == 80
    assert full["training"]["initialize_from"] is None
    assert full["optimizer"] == {
        "name": "adamw",
        "lr": 0.0002,
        "backbone_lr_scale": 0.25,
        "weight_decay": 0.0001,
    }

    full_model = dict(full["model"])
    control_model = dict(control["model"])
    assert full_model["backend"] == control_model["backend"] == "rc_mshnet"
    assert full_model["expose_branch_auxiliary"] is False
    assert control_model["expose_branch_auxiliary"] is False
    assert all(full_model[name] is True for name in EXTENSION_FLAGS)
    assert all(control_model[name] is False for name in EXTENSION_FLAGS)
    for name in EXTENSION_FLAGS:
        full_model.pop(name)
        control_model.pop(name)
    assert full_model == control_model
    assert full["output_dir"] != control["output_dir"]


def test_main_rc_mshnet_preserves_the_four_baseline_auxiliary_logits() -> None:
    full, control = _paired_configs("nuaa")
    baseline = build_mshnet({"backend": "canonical"}).eval()
    proposed = build_mshnet(full["model"]).eval()
    matched_control = build_mshnet(control["model"]).eval()
    inputs = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        baseline_output = forward_mshnet(baseline, inputs, warm_flag=True)
        proposed_output = forward_mshnet(proposed, inputs, warm_flag=True)
        control_output = forward_mshnet(matched_control, inputs, warm_flag=True)

    assert len(baseline_output.auxiliary_logits) == 4
    assert len(proposed_output.auxiliary_logits) == 4
    assert len(control_output.auxiliary_logits) == 4


def test_only_branch_aux_ablation_exposes_two_extra_auxiliary_logits() -> None:
    full, _ = _paired_configs("nuaa")
    main_model = build_mshnet(full["model"]).eval()
    branch_aux_config = deepcopy(full["model"])
    branch_aux_config["expose_branch_auxiliary"] = True
    branch_aux_model = build_mshnet(branch_aux_config).eval()
    branch_aux_model.load_state_dict(main_model.state_dict(), strict=True)
    inputs = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        main_output = forward_mshnet(main_model, inputs, warm_flag=True)
        branch_aux_output = forward_mshnet(
            branch_aux_model, inputs, warm_flag=True
        )

    torch.testing.assert_close(
        branch_aux_output.logits, main_output.logits, rtol=0.0, atol=0.0
    )
    assert len(main_output.auxiliary_logits) == 4
    assert len(branch_aux_output.auxiliary_logits) == 6


def test_main_rc_mshnet_zero_residual_initialization_is_exact() -> None:
    torch.manual_seed(23)
    full, _ = _paired_configs("nuaa")
    baseline = MSHNet(3).eval()
    proposed = build_mshnet(full["model"]).eval()
    assert isinstance(proposed, RCMSHNet)
    proposed.load_state_dict(baseline.state_dict(), strict=False)
    inputs = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        _, baseline_logits = baseline(inputs, True)
        proposed_logits = proposed(inputs, multi_scale=True).logits

    max_abs_delta = float((proposed_logits - baseline_logits).abs().max())
    assert max_abs_delta == 0.0


def test_full_and_matched_control_share_initializer_hash_and_training_budget(
    tmp_path: Path,
) -> None:
    baseline = MSHNet(3)
    checkpoint = tmp_path / "shared_mshnet_initializer.pt"
    torch.save({"model_state": baseline.state_dict()}, checkpoint)
    full, control = _paired_configs("nuaa")
    full_model = build_mshnet(full["model"])
    control_model = build_mshnet(control["model"])

    full_report = initialize_rc_mshnet_from_checkpoint(full_model, checkpoint)
    control_report = initialize_rc_mshnet_from_checkpoint(
        control_model, checkpoint
    )

    assert full_report["source_sha256"] == control_report["source_sha256"]
    assert len(full_report["source_sha256"]) == 64
    assert full_report["backbone_fully_loaded"] is True
    assert control_report["backbone_fully_loaded"] is True
    assert full["training"]["epochs"] == control["training"]["epochs"] == 80
    assert full["optimizer"] == control["optimizer"]


@pytest.mark.parametrize(
    "target_key,target_name,source_names,outer_initializer,inner_initializers",
    PIPELINE_CASES,
)
def test_gate_f_pipeline_config_is_formal_raw_logit_and_domain_correct(
    target_key: str,
    target_name: str,
    source_names: tuple[str, str],
    outer_initializer: str,
    inner_initializers: dict[str, str],
) -> None:
    config = _load(f"pipeline_outer_{target_key}_rc_mshnet_v4.yaml")

    assert config["seed"] == 42
    assert config["device"] == "cuda:0"
    assert config["devices"] == ["cuda:0"]
    assert config["diagnostic_only"] is False
    assert config["method_contract"] == "rc_v2_aaai27_main.yaml"
    assert config["require_method_contract"] is True
    assert config["detector_template"] == (
        f"detector_rc_mshnet_outer_{target_key}_fast.yaml"
    )
    detector = _load(config["detector_template"])
    assert detector["model"]["backend"] == "rc_mshnet"
    assert detector["model"]["expose_branch_auxiliary"] is False

    expected_sources = [
        {"name": name, "path": f"../datasets/{name}"} for name in source_names
    ]
    assert config["meta_sources"] == expected_sources
    assert target_name not in {item["name"] for item in expected_sources}
    assert config["final_targets"] == [
        {
            "name": target_name,
            "path": f"../datasets/{target_name}",
            "split": "test",
            "evaluate": True,
            "risk_curve_mode": "static-cross-fit",
            "cross_fit_folds": 5,
            "cross_fit_seed": 42,
            "num_workers": 2,
        }
    ]

    initializers = config["detector_initializers"]
    assert initializers["outer_final"] == outer_initializer
    assert initializers["inner_by_held_out"] == inner_initializers
    assert set(inner_initializers) == set(source_names)
    for placeholder in (outer_initializer, *inner_initializers.values()):
        assert placeholder.startswith("../artifacts/aaai27/initializers/")
        assert placeholder.endswith("_tensor_only.pt")

    assert config["method"]["name"] == "risk_curve"
    assert config["baseline"]["direct_threshold"]["enabled"] is True
    assert config["meta"]["split"] == "train"
    assert (
        config["meta"]["adaptation_window"],
        config["meta"]["evaluation_window"],
        config["meta"]["stride"],
    ) == (32, 1, 33)

    risk_curve = config["risk_curve"]
    assert risk_curve["representation"] == "raw_logit_float32"
    assert risk_curve["outer_target"] == target_name
    assert risk_curve["max_grid_points"] == 1024
    assert risk_curve["budget_pairs"] == [
        {"name": "strict", "pixel": 1.0e-6, "component": 1.0},
        {"name": "medium", "pixel": 5.0e-6, "component": 5.0},
        {"name": "loose", "pixel": 1.0e-5, "component": 10.0},
    ]
    assert config["target_stage_separation"] == TARGET_STAGE_DEFAULT


@pytest.mark.parametrize(
    "source_key,source_name,output_dir", SOURCE_INITIALIZER_CASES
)
def test_single_source_initializer_config_reuses_outer_baseline_contract(
    source_key: str,
    source_name: str,
    output_dir: str,
) -> None:
    config = _load(f"detector_mshnet_source_{source_key}_sls.yaml")
    outer_baseline = _load("detector_mshnet_outer_nuaa_sls.yaml")

    assert config["data"]["sources"] == [
        {"name": source_name, "path": f"../datasets/{source_name}"}
    ]
    assert config["output_dir"] == output_dir
    assert config["model"]["backend"] == "canonical"
    assert config["loss"]["lambda_tail"] == 0.0
    assert config["loss"]["lambda_miss"] == 0.0
    assert config["loss"]["lambda_margin"] == 0.0
    assert config["optimizer"]["name"] == "sgd"
    assert config["training"]["epochs"] == 400
    assert config["training"]["checkpoint_selection"] == "fixed_last"
    assert config["training"]["validation_interval"] == 0
    assert config["data"]["val_split"] is None

    # Ignore only the fields that intentionally identify this single-source
    # run; every architecture, objective, optimizer and data-policy field must
    # remain byte-for-byte equivalent after YAML parsing.
    source_contract = deepcopy(config)
    baseline_contract = deepcopy(outer_baseline)
    source_contract.pop("output_dir")
    baseline_contract.pop("output_dir")
    source_contract["data"].pop("sources")
    baseline_contract["data"].pop("sources")
    assert source_contract == baseline_contract


def test_single_source_configs_cover_every_gate_f_inner_training_domain() -> None:
    available_train_sets = {
        frozenset(
            item["name"] for item in _load(
                f"detector_mshnet_source_{source_key}_sls.yaml"
            )["data"]["sources"]
        )
        for source_key, _, _ in SOURCE_INITIALIZER_CASES
    }
    required_train_sets: set[frozenset[str]] = set()
    for target_key, _, source_names, _, _ in PIPELINE_CASES:
        pipeline = _load(f"pipeline_outer_{target_key}_rc_mshnet_v4.yaml")
        held_out_mapping = pipeline["detector_initializers"][
            "inner_by_held_out"
        ]
        for held_out in held_out_mapping:
            required_train_sets.add(frozenset(set(source_names) - {held_out}))

    assert available_train_sets == required_train_sets == {
        frozenset({"NUAA-SIRST"}),
        frozenset({"NUDT-SIRST"}),
        frozenset({"IRSTD-1K"}),
    }
