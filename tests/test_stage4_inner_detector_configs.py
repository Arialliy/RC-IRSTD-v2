"""Static leakage and training-contract checks for Stage 4 inner detectors."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_CASES = (
    (
        "stage4_inner_nudt_from_irstd_sls_20ep.yaml",
        "cuda:0",
        "../outputs/stage4_inner_nudt_from_irstd_sls_20ep",
        "IRSTD-1K",
        "../datasets/IRSTD-1K",
        "NUDT-SIRST",
        0.0,
        0.0,
    ),
    (
        "stage4_inner_irstd_from_nudt_sls_20ep.yaml",
        "cuda:1",
        "../outputs/stage4_inner_irstd_from_nudt_sls_20ep",
        "NUDT-SIRST",
        "../datasets/NUDT-SIRST",
        "IRSTD-1K",
        0.0,
        0.0,
    ),
    (
        "stage4_inner_nudt_from_irstd_tailmiss_20ep.yaml",
        "cuda:0",
        "../outputs/stage4_inner_nudt_from_irstd_tailmiss_20ep",
        "IRSTD-1K",
        "../datasets/IRSTD-1K",
        "NUDT-SIRST",
        0.10,
        0.10,
    ),
    (
        "stage4_inner_irstd_from_nudt_tailmiss_20ep.yaml",
        "cuda:1",
        "../outputs/stage4_inner_irstd_from_nudt_tailmiss_20ep",
        "NUDT-SIRST",
        "../datasets/NUDT-SIRST",
        "IRSTD-1K",
        0.10,
        0.10,
    ),
)


def _load_config(filename: str) -> dict[str, object]:
    path = ROOT / "configs" / filename
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


@pytest.mark.parametrize(
    (
        "filename,device,output_dir,source_name,source_path,held_out_name,"
        "lambda_tail,lambda_miss"
    ),
    CONFIG_CASES,
)
def test_stage4_inner_detector_is_single_source_and_test_blind(
    filename: str,
    device: str,
    output_dir: str,
    source_name: str,
    source_path: str,
    held_out_name: str,
    lambda_tail: float,
    lambda_miss: float,
) -> None:
    config = _load_config(filename)

    assert config["seed"] == 42
    assert config["deterministic"] is True
    assert config["device"] == device
    assert config["output_dir"] == output_dir

    data = config["data"]
    assert data["sources"] == [{"name": source_name, "path": source_path}]
    assert held_out_name not in str(data)
    assert data["train_split"] == "train"
    assert data["val_split"] is None
    assert data["diagnostic_test_eval"] is False

    assert config["model"]["backend"] == "canonical"
    assert config["loss"]["lambda_tail"] == lambda_tail
    assert config["loss"]["lambda_miss"] == lambda_miss
    assert config["loss"]["lambda_margin"] == 0.0

    training = config["training"]
    assert training["checkpoint_selection"] == "fixed_last"
    assert training["epochs"] == 20
    assert training["resume"] is None
