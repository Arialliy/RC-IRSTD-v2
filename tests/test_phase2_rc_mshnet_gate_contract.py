from __future__ import annotations

import json
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
from scripts.extract_mshnet_weights import (
    _require_stable_sls_v1,
    _source_training_identity,
)
from scripts.phase2_gatekeeper import (
    PHASE2_CASES,
    assert_sentinel_allows,
    validate_phase2_configs,
)
from scripts.phase2_identity import write_identity


ROOT = Path(__file__).resolve().parents[1]


def _yaml(filename: str) -> dict[str, object]:
    payload = yaml.safe_load((ROOT / "configs" / filename).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_six_phase2_configs_exist_and_match_the_frozen_matrix() -> None:
    hashes = validate_phase2_configs()
    assert set(hashes) == set(PHASE2_CASES)
    assert all(len(value) == 64 for value in hashes.values())


def test_formal_full_keeps_four_auxiliaries_and_branchaux_has_six() -> None:
    full = build_mshnet(_yaml("phase2_rc_mshnet_full_outer_nuaa.yaml")["model"])
    branch_aux = build_mshnet(
        _yaml("phase2_rc_mshnet_branch_aux_outer_nuaa.yaml")["model"]
    )
    branch_aux.load_state_dict(full.state_dict(), strict=True)
    inputs = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        full_output = forward_mshnet(full.eval(), inputs, warm_flag=True)
        branch_output = forward_mshnet(branch_aux.eval(), inputs, warm_flag=True)
    torch.testing.assert_close(full_output.logits, branch_output.logits, rtol=0.0, atol=0.0)
    assert len(full_output.auxiliary_logits) == 4
    assert len(branch_output.auxiliary_logits) == 6


def test_sentinel_is_fail_closed(tmp_path: Path) -> None:
    state = tmp_path / "phase_state"
    state.mkdir()
    with pytest.raises(RuntimeError, match="blocked"):
        assert_sentinel_allows(state)
    (state / "HOLD_RC_MSHNET_GATE").touch()
    (state / "ALLOW_RC_MSHNET_GATE").touch()
    with pytest.raises(RuntimeError, match="blocked"):
        assert_sentinel_allows(state)
    (state / "HOLD_RC_MSHNET_GATE").unlink()
    assert_sentinel_allows(state)


def test_real_initializer_helper_is_numerically_exact(tmp_path: Path) -> None:
    torch.manual_seed(91)
    baseline = MSHNet(3).eval()
    checkpoint = tmp_path / "canonical.pt"
    torch.save({"model_state": baseline.state_dict()}, checkpoint)
    proposed = RCMSHNet(expose_branch_auxiliary=False).eval()
    report = initialize_rc_mshnet_from_checkpoint(proposed, checkpoint)
    inputs = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        _, baseline_logits = baseline(inputs, True)
        proposed_logits = proposed(inputs, multi_scale=True).logits
    torch.testing.assert_close(proposed_logits, baseline_logits, rtol=0.0, atol=0.0)
    assert report["backbone_fully_loaded"] is True


def test_initializer_helper_rejects_rc_extension_weights(tmp_path: Path) -> None:
    contaminated = RCMSHNet(expose_branch_auxiliary=False)
    checkpoint = tmp_path / "contaminated.pt"
    torch.save({"model_state": contaminated.state_dict()}, checkpoint)
    with pytest.raises(RuntimeError, match="extension weights"):
        initialize_rc_mshnet_from_checkpoint(
            RCMSHNet(expose_branch_auxiliary=False),
            checkpoint,
        )


def test_stable_sls_source_identity_requires_a_final_checkpoint() -> None:
    config = _yaml("detector_mshnet_outer_nuaa_sls.yaml")
    payload = {
        "config": config,
        "epoch": 399,
        "inference_head": "multi_scale_fused",
        "test_labels_used_for_selection": False,
    }
    identity = _source_training_identity(payload)
    _require_stable_sls_v1(identity)
    assert identity is not None
    assert identity["auxiliary_weight"] == 0.25
    payload["epoch"] = 398
    with pytest.raises(ValueError, match="not StableSLS-v1"):
        _require_stable_sls_v1(_source_training_identity(payload))


def test_running_identity_sidecars_are_explicitly_unfrozen(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _yaml("detector_mshnet_outer_nuaa_sls.yaml")
    (run_dir / "config.json").write_text(
        json.dumps(config, sort_keys=True),
        encoding="utf-8",
    )
    manifest = write_identity(run_dir, finalize=False)
    assert manifest["artifact_state"] == "running_unfrozen"
    assert manifest["checkpoint_sha256"] is None
    for filename in (
        "LOSS_IDENTITY.txt",
        "MODEL_IDENTITY.txt",
        "DATA_IDENTITY.txt",
        "IDENTITY_MANIFEST.json",
    ):
        assert (run_dir / filename).is_file()
    assert "checkpoint_sha256=PENDING_FINAL_EPOCH_400" in (
        run_dir / "LOSS_IDENTITY.txt"
    ).read_text(encoding="utf-8")
