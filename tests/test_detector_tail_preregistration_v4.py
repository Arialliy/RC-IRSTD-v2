from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from rc_irstd.losses.tail_rank import (
    RAW_LOGIT_TAILRANK_CONNECTIVITY,
    RAW_LOGIT_TAILRANK_LAMBDA_MARGIN,
    RAW_LOGIT_TAILRANK_LAMBDA_MISS,
    RAW_LOGIT_TAILRANK_LAMBDA_TAIL,
    RAW_LOGIT_TAILRANK_MARGIN,
    RAW_LOGIT_TAILRANK_MODE,
)


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT
    / "outputs"
    / "v4_source_only"
    / "preregistration"
    / "detector_tail_branches_seed42.json"
)
# Do not recreate or reseal a deleted historical preregistration just to make
# its replay tests green.  The active AAAI27 contract has separate tests.
pytestmark = pytest.mark.skipif(
    not PREREGISTRATION.is_file(),
    reason="historical detector-tail experiment products were intentionally removed",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load() -> dict[str, object]:
    payload = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_detector_tail_preregistration_binds_every_declared_input() -> None:
    payload = _load()
    assert payload["schema_version"] == (
        "rc-v4-source-development-detector-tail-preregistration-v1"
    )
    assert payload["project_root"] == str(ROOT)
    assert payload["claim_scope"] == (
        "source_only_development_gate_c_not_outer_or_blind"
    )
    development = payload["development_evidence_status"]
    assert isinstance(development, dict)
    assert development["source_validation_folds_previously_opened_for_diagnosis"] is True
    assert development["source_validation_folds_must_not_be_described_as_historically_blind"] is True
    assert development["outer_target_labels_authorized_for_training_or_selection"] is False

    splits = payload["official_train_splits"]
    assert isinstance(splits, dict)
    for record in splits.values():
        assert isinstance(record, dict)
        path = Path(str(record["path"]))
        assert path.is_relative_to(ROOT)
        assert _sha256(path) == record["sha256"]

    branches = payload["branch_sequence"]
    assert isinstance(branches, list) and len(branches) == 2
    assert [branch["branch_id"] for branch in branches] == [
        "tailmiss_probability_v1",
        "raw_logit_tailrank_margin_a_v1",
    ]
    for branch in branches:
        items = list(branch.get("inner_configs", [])) + list(
            branch.get("configs", [])
        )
        for item in items:
            path = Path(str(item["path"]))
            assert path.is_relative_to(ROOT)
            assert _sha256(path) == item["sha256"]
        checkpoint = branch.get("full_source_checkpoint")
        if checkpoint is not None:
            path = Path(str(checkpoint["path"]))
            assert path.is_relative_to(ROOT)
            assert _sha256(path) == checkpoint["sha256"]
            assert checkpoint["preexisting_before_this_preregistration"] is True


def test_candidate_a_preregistered_loss_and_gate_c_cannot_drift() -> None:
    payload = _load()
    candidate = payload["branch_sequence"][1]
    loss = candidate["loss_contract"]
    assert loss == {
        "tail_mode": RAW_LOGIT_TAILRANK_MODE,
        "lambda_tail": RAW_LOGIT_TAILRANK_LAMBDA_TAIL,
        "lambda_miss": RAW_LOGIT_TAILRANK_LAMBDA_MISS,
        "lambda_margin": RAW_LOGIT_TAILRANK_LAMBDA_MARGIN,
        "margin": RAW_LOGIT_TAILRANK_MARGIN,
        "target_connectivity": RAW_LOGIT_TAILRANK_CONNECTIVITY,
        "canonical_loss_mapping_sha256": (
            "3f0976641ff045c1341e2bd72150f8e198cdd13acd9bd8e05fca93e446463497"
        ),
        "full_source_order_sha256": (
            "d7da61f2ad9b6ead6a7149470d99a614a2ddb4f3113e19700d392a42a8f373c4"
        ),
        "tail_rank_module_sha256": _sha256(ROOT / "rc_irstd/losses/tail_rank.py"),
        "detector_objective_module_sha256": _sha256(
            ROOT / "rc_irstd/losses/detector.py"
        ),
    }
    config_losses = []
    for item in candidate["configs"]:
        config = yaml.safe_load(Path(item["path"]).read_text(encoding="utf-8"))
        config_losses.append(config["loss"])
        assert "nuaa" not in json.dumps(config, ensure_ascii=False).casefold()
    assert config_losses[0] == config_losses[1] == config_losses[2]

    shared = payload["shared_protocol"]
    assert shared["pixel_budgets"] == [1.0e-5, 1.0e-6]
    assert shared["component_budgets"] == [5.0, 1.0]
    assert shared["episode_adaptation_window"] == 32
    assert shared["episode_evaluation_window"] == 1
    assert shared["episode_stride"] == 33
    gate = payload["gate_c_success_contract"]
    assert gate["criteria_must_not_be_lowered"] is True
    assert gate["pd_non_degradation_floor"] == 0.0
    assert set(key for key in gate if len(key) == 2 and key.startswith("c")) == {
        f"c{index}" for index in range(8)
    }
    assert gate["gate_decision"] == "GO_only_if_all_c0_through_c7_are_true"
    assert payload[
        "new_tailmiss_inner_or_candidate_a_gpu_training_started_before_registration"
    ] is False
    assert payload["results_present_in_this_preregistration"] is False
