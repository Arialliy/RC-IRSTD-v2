from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import risk_curve.aggregate_anchor_warp_gate_c_v4 as gate_module
from risk_curve.aggregate_anchor_warp_gate_c_v4 import (
    AGGREGATE_ANCHOR_WARP_GATE_C_SCHEMA_VERSION,
    PD_NON_DEGRADATION_FLOOR,
    RUNTIME_CODE_TREE_FILES,
    RUNTIME_CODE_TREE_SCHEMA_VERSION,
    aggregate_anchor_warp_gate_c,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IRSTD_COMPARISON = (
    REPOSITORY_ROOT
    / "outputs/v4_anchor_warp_source_only/val_irstd/comparison_seed42.json"
)
NUDT_COMPARISON = (
    REPOSITORY_ROOT
    / "outputs/v4_anchor_warp_source_only/val_nudt/comparison_seed42.json"
)
# The user deliberately removed all pre-AAAI27 experiment products.  These
# tests replay that historical evidence only when both immutable inputs are
# present; the current pipeline is covered by independent contract tests.
pytestmark = pytest.mark.skipif(
    not IRSTD_COMPARISON.is_file() or not NUDT_COMPARISON.is_file(),
    reason="historical v4 experiment products were intentionally removed",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mutated_copy(source: Path, destination: Path, mutate) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    mutate(payload)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


@pytest.fixture(scope="module")
def current_gate_result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    assert IRSTD_COMPARISON.is_file()
    assert NUDT_COMPARISON.is_file()
    before = (_sha256(IRSTD_COMPARISON), _sha256(NUDT_COMPARISON))
    output = tmp_path_factory.mktemp("anchor-gate-c") / "decision.json"
    aggregate_anchor_warp_gate_c(
        folds=[("val_irstd", IRSTD_COMPARISON), ("val_nudt", NUDT_COMPARISON)],
        output=output,
    )
    after = (_sha256(IRSTD_COMPARISON), _sha256(NUDT_COMPARISON))
    assert after == before, "aggregation must not modify either comparison input"
    return json.loads(output.read_text(encoding="utf-8"))


def test_current_two_fold_gate_is_machine_readable_and_fails_closed(
    current_gate_result: dict,
) -> None:
    result = current_gate_result
    assert result["schema_version"] == AGGREGATE_ANCHOR_WARP_GATE_C_SCHEMA_VERSION
    assert result["decision"] == "HOLD"
    assert set(result["folds"]) == {"val_irstd", "val_nudt"}
    assert result["criteria"] == {
        "c0_integrity_binding_and_two_phase_replay": True,
        "c1_complete_two_fold_paired_method_matrix": True,
        "c2_absolute_pixel_component_risk_control": False,
        "c3_jvr_mre_max_excess_non_adverse": False,
        "c4_stable_benefit_not_pd_only": True,
        "c5_literal_pd_not_lower_than_rc_direct": False,
        "c6_reject_rate_below_20_percent": True,
        "c7_risk_curve_mvr_zero": True,
    }
    assert result["failure_reasons"] == [
        "c2_absolute_pixel_component_risk_control",
        "c3_jvr_mre_max_excess_non_adverse",
        "c5_literal_pd_not_lower_than_rc_direct",
    ]
    assert set(result["macro_by_budget"]) == {"0", "1"}
    assert set(result["micro_by_budget"]) == {"0", "1"}
    # Re-encoding with allow_nan=False proves the artifact is strict JSON data.
    json.dumps(result, sort_keys=True, allow_nan=False)


def test_checkpoint_hashes_and_two_phase_order_are_explicit(
    current_gate_result: dict,
) -> None:
    result = current_gate_result
    for fold in result["folds"].values():
        assert set(fold["artifacts"]) == {
            "validation_episode_archive",
            "train_episode_archive",
            "anchor_warp_bound_package",
            "anchor_warp_train_only_parent",
            "rc_direct_train_only_checkpoint",
        }
        for artifact in fold["artifacts"].values():
            assert len(artifact["sha256"]) == 64
            assert _sha256(Path(artifact["path"])) == artifact["sha256"]
        binding = fold["checkpoint_binding"]
        assert binding["anchor_train_only_checkpoint_validated"] is True
        assert binding["anchor_parent_bytes_and_embedded_semantics_match"] is True
        assert binding["identical_train_archive_bytes"] is True
        assert binding["rc_direct_train_only_checkpoint_validated"] is True
        assert [
            event["event"] for event in binding["direct_train_only_read_event_sequence"]
        ] == [
            "train_bytes_captured",
            "train_only_cross_validation_started",
            "fixed_epoch_selected_from_train_only_cv",
            "all_train_model_frozen",
            "validation_bytes_captured",
            "post_freeze_episode_contract_bound",
        ]
        proof = fold["two_phase_action_evidence"]
        assert proof["digest_order"] == "budget_major_then_episode_minor"
        assert proof["digest_frozen_before_future_e_load"] is True
        assert proof["future_e_reselection_performed"] is False
        replay = fold["runtime_two_phase_replay"]
        assert replay["comparison_replay_byte_exact"] is True
        assert replay["phase_order"].index("A_compute_action_digest") < replay[
            "phase_order"
        ].index("E_load_future_e_sufficient_counts")


def test_runtime_replay_code_tree_binds_all_eight_minimal_modules(
    current_gate_result: dict,
) -> None:
    tree = current_gate_result["runtime_code_tree"]
    assert tree["schema_version"] == RUNTIME_CODE_TREE_SCHEMA_VERSION
    assert tree["entry_snapshot_before_input_load"] is True
    assert tree["verified_unchanged_immediately_before_atomic_publish"] is True
    assert tree["required_file_count"] == 8
    assert tree["required_files"] == list(RUNTIME_CODE_TREE_FILES)
    assert set(tree["files"]) == set(RUNTIME_CODE_TREE_FILES)
    for relative_path in RUNTIME_CODE_TREE_FILES:
        record = tree["files"][relative_path]
        path = Path(record["path"])
        raw = path.read_bytes()
        assert record["size_bytes"] == len(raw)
        assert record["sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("relative_path", RUNTIME_CODE_TREE_FILES)
def test_each_runtime_code_dependency_drift_is_rejected(
    relative_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = gate_module._snapshot_runtime_code_tree()
    target = Path(snapshot["files"][relative_path]["path"]).resolve()
    original_read = gate_module._read_runtime_code_bytes

    def read_with_drift(path: Path) -> bytes:
        raw = original_read(path)
        return raw + b"\n# simulated runtime-code drift\n" if path.resolve() == target else raw

    monkeypatch.setattr(gate_module, "_read_runtime_code_bytes", read_with_drift)
    with pytest.raises(ValueError, match=f"Runtime code dependency drifted: {relative_path}"):
        gate_module._verify_runtime_code_tree(snapshot)


def test_aggregate_fails_closed_if_dependency_drifts_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_suffix = "risk_curve/anchor_warp_inference.py"
    original_read = gate_module._read_runtime_code_bytes
    target_reads = 0

    def read_with_post_entry_drift(path: Path) -> bytes:
        nonlocal target_reads
        raw = original_read(path)
        if str(path).endswith(target_suffix):
            target_reads += 1
            if target_reads >= 2:
                return raw + b"\n# simulated post-entry drift\n"
        return raw

    monkeypatch.setattr(
        gate_module, "_read_runtime_code_bytes", read_with_post_entry_drift
    )
    output = tmp_path / "must_not_publish.json"
    with pytest.raises(
        ValueError,
        match="Runtime code dependency drifted: risk_curve/anchor_warp_inference.py",
    ):
        aggregate_anchor_warp_gate_c(
            folds=[("val_irstd", IRSTD_COMPARISON), ("val_nudt", NUDT_COMPARISON)],
            output=output,
        )
    assert target_reads >= 2
    assert not output.exists()


def test_literal_pd_criterion_cannot_use_the_legacy_two_point_allowance(
    current_gate_result: dict,
) -> None:
    assert PD_NON_DEGRADATION_FLOOR == 0.0
    checks = current_gate_result["literal_pd_vs_rc_direct_checks"]
    assert any(row["pd_delta"] < 0.0 for row in checks)
    assert all(
        row["non_degraded"] == (row["pd_delta"] >= -1.0e-12)
        for row in checks
    )
    assert current_gate_result["aggregation_contract"][
        "criteria_not_lowered"
    ] is True


def test_tampered_action_digest_is_rejected_before_decision(tmp_path: Path) -> None:
    changed = _mutated_copy(
        IRSTD_COMPARISON,
        tmp_path / "changed_digest.json",
        lambda payload: payload["action_freeze"].__setitem__(
            "action_digest_sha256", "0" * 64
        ),
    )
    with pytest.raises(ValueError, match="frozen action digest mismatch"):
        aggregate_anchor_warp_gate_c(
            folds=[("val_irstd", changed), ("val_nudt", NUDT_COMPARISON)],
            output=tmp_path / "decision.json",
        )


def test_future_e_before_digest_attestation_is_rejected(tmp_path: Path) -> None:
    changed = _mutated_copy(
        IRSTD_COMPARISON,
        tmp_path / "changed_phase.json",
        lambda payload: payload.__setitem__(
            "future_e_arrays_loaded_before_action_digest", True
        ),
    )
    with pytest.raises(
        ValueError, match="future_e_arrays_loaded_before_action_digest must be False"
    ):
        aggregate_anchor_warp_gate_c(
            folds=[("val_irstd", changed), ("val_nudt", NUDT_COMPARISON)],
            output=tmp_path / "decision.json",
        )


def test_tampered_checkpoint_hash_is_rejected(tmp_path: Path) -> None:
    changed = _mutated_copy(
        IRSTD_COMPARISON,
        tmp_path / "changed_checkpoint_hash.json",
        lambda payload: payload.__setitem__(
            "anchor_warp_package_sha256", "0" * 64
        ),
    )
    with pytest.raises(ValueError, match="anchor_warp_package SHA-256 mismatch"):
        aggregate_anchor_warp_gate_c(
            folds=[("val_irstd", changed), ("val_nudt", NUDT_COMPARISON)],
            output=tmp_path / "decision.json",
        )


def test_tampered_recorded_metric_is_rejected(tmp_path: Path) -> None:
    def mutate(payload: dict) -> None:
        payload["budgets"][0]["methods"]["risk_curve"]["pd"] += 0.01

    changed = _mutated_copy(
        IRSTD_COMPARISON, tmp_path / "changed_metric.json", mutate
    )
    with pytest.raises(ValueError, match="does not match reconstructed evidence"):
        aggregate_anchor_warp_gate_c(
            folds=[("val_irstd", changed), ("val_nudt", NUDT_COMPARISON)],
            output=tmp_path / "decision.json",
        )


def test_two_distinct_held_source_folds_are_mandatory(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate_irstd.json"
    duplicate.write_bytes(IRSTD_COMPARISON.read_bytes())
    with pytest.raises(ValueError, match="held pseudo-targets must be exactly"):
        aggregate_anchor_warp_gate_c(
            folds=[("first", IRSTD_COMPARISON), ("second", duplicate)],
            output=tmp_path / "decision.json",
        )
