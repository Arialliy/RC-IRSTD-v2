"""Static contracts for the AAAI-27 method configuration and smoke launcher."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "rc_v2_aaai27_main.yaml"
LAUNCHER = ROOT / "scripts" / "launch_risk_curve_smoke.sh"


def _load_config() -> dict[str, object]:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def test_aaai27_method_roles_are_unambiguous() -> None:
    config = _load_config()

    method = config["method"]
    baseline = config["baseline"]
    assert method["name"] == "risk_curve"
    assert method["role"] == "proposed_method"
    assert method["model"].endswith(".RiskCurvePredictor")
    assert str(method["output"]).endswith(".pt")

    assert baseline["name"] == "direct_threshold"
    assert baseline["enabled"] is True
    assert baseline["role"] == "strong_baseline"
    assert baseline["model"].endswith(".MonotoneBudgetCalibrator")
    assert baseline["static_evaluator"].endswith(
        ".evaluate_static_crossfit_direct"
    )
    assert str(baseline["output"]).endswith(".pt")


def test_aaai27_protocol_separates_standard_static_and_temporal_results() -> None:
    config = _load_config()
    protocol = config["protocol"]

    standard = protocol["standard_detection"]
    assert standard == {
        "train_split": "train",
        "evaluation_split": "test",
        "evaluate_all_images": True,
    }

    static = protocol["static_rc"]
    assert static["mode"] == "cross_fit"
    assert static["evaluation_split"] == "test"
    assert static["folds"] == 5
    assert static["adaptation_size"] == 32
    assert static["adaptation_sampling"] == (
        "deterministic_without_replacement_from_complement"
    )
    assert static["evaluate_all_images"] is True
    assert static["causal_claim"] is False
    assert static["formal_crc_eligible"] is False

    temporal = protocol["temporal_rc"]
    assert temporal["mode"] == "causal"
    assert temporal["enabled_when_sequence_metadata_available"] is True
    assert temporal["stride"] >= (
        temporal["support_size"] + temporal["evaluation_size"]
    )
    assert temporal["causal_claim"] is True


def test_aaai27_threshold_grid_uses_the_raw_logit_v4_contract() -> None:
    config = _load_config()
    representation = config["representation"]
    grid = config["threshold_grid"]
    risk_curve = config["risk_curve"]

    assert representation["name"] == "raw_logit_float32"
    assert representation["score_dtype"] == "float32"
    assert representation["prediction_rule"] == (
        "raw_logits_greater_equal_threshold"
    )
    assert grid["builder"] == "risk_curve.build_logit_threshold_grid"
    assert grid["source"] == "source_official_train_only"
    assert grid["finite_grid_points"] == 1024
    assert grid["max_grid_points"] == 2048
    assert grid["empty_action"] == "external_plus_inf"
    assert grid["require_outer_target_excluded"] is True
    assert risk_curve["representation"] == "raw_logit_float32"
    assert risk_curve["training"]["val_file"].endswith("/val.npz")


def test_smoke_launcher_only_passes_supported_training_arguments() -> None:
    launcher_text = LAUNCHER.read_text(encoding="utf-8")
    launcher_arguments = set(re.findall(r"--[a-z][a-z0-9-]*", launcher_text))

    help_result = subprocess.run(
        [sys.executable, "-m", "risk_curve.train_curve_predictor", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    supported_arguments = set(
        re.findall(r"--[a-z][a-z0-9-]*", help_result.stdout)
    )

    assert {"--train-file", "--val-file", "--output"} <= launcher_arguments
    assert launcher_arguments <= supported_arguments
    assert "--threshold-grid" not in launcher_arguments
    assert "--component-target" not in launcher_arguments
    assert '${PYTHON_BIN:-python3}' in launcher_text
    assert re.search(r'OUT="\$\{OUT:-[^"\n]*\.pt\}"', launcher_text)
    assert "outputs/curve_episodes/val.npz" in launcher_text
    assert "heldout_pseudo_domain.npz" not in launcher_text


def test_smoke_launcher_is_valid_bash() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
