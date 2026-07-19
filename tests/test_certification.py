import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from certification.build_calibration_losses import (
    COMPONENT_BUDGET_UNIT,
    PIXEL_BUDGET_UNIT,
    build_calibration_losses,
    conservative_suffix_max,
    load_count_curve_archive,
    save_calibration_losses,
)
from certification.calibrate_target_offset import (
    assert_disjoint_image_ids,
    assert_three_way_disjoint_image_ids,
    calibrate_target_offset,
)
from certification.conformal_offset import (
    build_grid_rank_candidates,
    finite_sample_feasibility,
    select_conformal_offset,
)
from evaluation.artifact_integrity import (
    SCORE_MANIFEST_SCHEMA_VERSION,
    SCORE_MASK_ALIGNMENT_SCHEMA,
    SCORE_RECORD_INTEGRITY_SCHEMA,
    file_sha256,
    ordered_ids_sha256,
    score_records_sha256,
    verify_score_map_directory,
)
from data_ext.mask_alignment import MASK_ALIGNMENT_POLICY
from certification.evaluate_certified_mode import evaluate_selected_operating_point


def _count_losses(prefix="cal", num_images=20, num_thresholds=4):
    thresholds = np.linspace(0.1, 0.9, num_thresholds)
    fp_pixels = np.zeros((num_images, num_thresholds), dtype=np.float64)
    fp_components = np.zeros_like(fp_pixels)
    if num_thresholds >= 2:
        fp_pixels[:, 0] = 2
        fp_pixels[:, 1] = 1
        fp_components[:, 0] = 2
        fp_components[:, 1] = 1
    return build_calibration_losses(
        image_ids=[f"{prefix}-{index}" for index in range(num_images)],
        thresholds=thresholds,
        false_positive_pixels=fp_pixels,
        false_positive_components=fp_components,
        total_pixels=np.full(num_images, 1_000_000),
        pixel_budget=1e-6,
        component_budget=1.0,
    )


def _write_count_archive(path, ids, thresholds, transition, protocol_fingerprint=None):
    num_images = len(ids)
    num_thresholds = len(thresholds)
    fp_pixels = np.zeros((num_images, num_thresholds), dtype=np.int64)
    fp_components = np.zeros_like(fp_pixels)
    fp_pixels[:, :transition] = 1
    fp_components[:, :transition] = 1
    arrays = {
        "image_ids": np.asarray(ids),
        "thresholds": np.asarray(thresholds),
        "false_positive_pixels": fp_pixels,
        "false_positive_components": fp_components,
        "total_pixels": np.full(num_images, 1_000_000, dtype=np.int64),
    }
    if protocol_fingerprint is not None:
        arrays["provenance_json"] = np.asarray(
            json.dumps({"protocol_fingerprint": protocol_fingerprint})
        )
    np.savez_compressed(path, **arrays)


def _write_score_map_directory(path: Path) -> tuple[np.ndarray, list[str]]:
    path.mkdir()
    thresholds = np.asarray([0.5, 0.85, 0.95], dtype=np.float64)
    records = []
    # Deliberately make manifest order differ from filename sort order.
    for image_id, filename, target_score, fp_col in (
        ("image-b", "b.npz", 0.95, 4),
        ("image-a", "a.npz", 0.90, 0),
    ):
        probability = np.zeros((5, 5), dtype=np.float32)
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        probability[2, 2] = target_score
        probability[0, fp_col] = 0.8 if image_id == "image-a" else 0.7
        np.savez_compressed(
            path / filename,
            prob=probability,
            gray=np.zeros_like(probability, dtype=np.float32),
            mask=mask,
            image_id=np.asarray(image_id),
            labels_loaded=np.asarray(True),
            original_hw=np.asarray([5, 5], dtype=np.int32),
            mask_alignment_applied=np.asarray(False),
            mask_original_hw=np.asarray([5, 5], dtype=np.int32),
            mask_aspect_relative_error=np.asarray(0.0),
            mask_alignment_policy=np.asarray(MASK_ALIGNMENT_POLICY),
        )
        records.append(
            {
                "image_id": image_id,
                "file": filename,
                "shape": [5, 5],
                "sha256": file_sha256(path / filename),
                "mask_alignment_applied": False,
                "mask_original_hw": [5, 5],
                "mask_aspect_relative_error": 0.0,
            }
        )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCORE_MANIFEST_SCHEMA_VERSION,
                "record_integrity_schema": SCORE_RECORD_INTEGRITY_SCHEMA,
                "score_type": "sigmoid_probability",
                "warm_flag": True,
                "labels_loaded": True,
                "spatial_mode": "native",
                "target_dataset": "synthetic",
                "pad_multiple": 16,
                "weight_sha256": "a" * 64,
                "num_images": 2,
                "records": records,
                "records_sha256": score_records_sha256(records),
                "ordered_image_ids_sha256": ordered_ids_sha256(
                    [record["image_id"] for record in records]
                ),
                "mask_alignment_schema": SCORE_MASK_ALIGNMENT_SCHEMA,
                "mask_alignment_policy": MASK_ALIGNMENT_POLICY,
                "mask_alignment_count": 0,
                "mask_aligned_sample_ids": [],
            }
        ),
        encoding="utf-8",
    )
    return thresholds, [record["image_id"] for record in records]


def test_component_suffix_max_is_conservative_and_monotone():
    raw = np.asarray([[5.0, 1.0, 3.0, 0.0]])
    envelope = conservative_suffix_max(raw)
    np.testing.assert_array_equal(envelope, [[5.0, 3.0, 3.0, 0.0]])
    assert np.all(envelope >= raw)
    assert np.all(np.diff(envelope, axis=1) <= 0.0)


def test_default_loss_is_binary_joint_budget_violation():
    losses = build_calibration_losses(
        image_ids=["image-a"],
        thresholds=np.asarray([0.1, 0.2, 0.3, 0.4]),
        false_positive_pixels=np.asarray([[2, 1, 0, 0]]),
        false_positive_components=np.asarray([[5, 1, 3, 0]]),
        total_pixels=np.asarray([1_000_000]),
        pixel_budget=1e-6,
        component_budget=2.0,
    )
    np.testing.assert_allclose(losses.pixel_risk, [[2e-6, 1e-6, 0.0, 0.0]])
    np.testing.assert_allclose(losses.component_risk_envelope, [[5.0, 3.0, 3.0, 0.0]])
    np.testing.assert_allclose(losses.pixel_loss, [[1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(losses.component_loss, [[1.0, 1.0, 1.0, 0.0]])
    np.testing.assert_allclose(losses.joint_loss, [[1.0, 1.0, 1.0, 0.0]])
    assert losses.metadata()["pixel_budget"]["unit"] == PIXEL_BUDGET_UNIT
    assert losses.metadata()["component_budget"]["unit"] == COMPONENT_BUDGET_UNIT
    assert losses.metadata()["joint_bsr_interpretation"] is True
    assert np.all((0.0 <= losses.joint_loss) & (losses.joint_loss <= 1.0))


def test_score_map_cli_builds_two_per_image_count_curves_in_manifest_order(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    score_dir = tmp_path / "scores"
    thresholds, expected_ids = _write_score_map_directory(score_dir)
    grid_path = tmp_path / "thresholds.npy"
    np.save(grid_path, thresholds)
    output = tmp_path / "calibration-losses.npz"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "certification.build_calibration_losses",
            "--score-dir",
            str(score_dir),
            "--threshold-grid",
            str(grid_path),
            "--pixel-budget",
            "0.039",
            "--component-budget",
            "39999",
            "--output",
            str(output),
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    with np.load(output, allow_pickle=False) as archive:
        assert archive["image_ids"].tolist() == expected_ids
        np.testing.assert_array_equal(
            archive["false_positive_pixels"], [[1, 0, 0], [1, 0, 0]]
        )
        np.testing.assert_array_equal(
            archive["false_positive_components"], [[1, 0, 0], [1, 0, 0]]
        )
        np.testing.assert_array_equal(archive["total_pixels"], [25, 25])
        np.testing.assert_array_equal(archive["tp_object_counts"], [[1, 1, 0], [1, 1, 0]])
        np.testing.assert_array_equal(archive["gt_object_counts"], [1, 1])
        np.testing.assert_allclose(archive["joint_loss"], [[1, 0, 0], [1, 0, 0]])
        provenance = json.loads(str(archive["provenance_json"].item()))
    assert provenance["source_type"] == "exported_score_map_directory"
    assert len(provenance["threshold_grid_sha256"]) == 64
    assert len(provenance["score_records_sha256"]) == 64
    assert len(provenance["score_ordered_image_ids_sha256"]) == 64
    with np.load(output, allow_pickle=False) as archive:
        assert len(str(archive["count_archive_payload_sha256"].item())) == 64


def test_score_manifest_reordering_and_count_archive_tampering_are_rejected(tmp_path):
    score_dir = tmp_path / "scores"
    _write_score_map_directory(score_dir)
    manifest_path = score_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"] = list(reversed(manifest["records"]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="records_sha256 mismatch"):
        verify_score_map_directory(score_dir, require_integrity=True)

    archive_path = tmp_path / "count.npz"
    losses = _count_losses(num_images=2)
    save_calibration_losses(
        archive_path,
        losses,
        provenance={"source_type": "diagnostic-fixture"},
    )
    load_count_curve_archive(archive_path, require_integrity=True)
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["provenance_json"] = np.asarray('{"source_type":"forged"}')
    np.savez_compressed(archive_path, **arrays)
    with pytest.raises(ValueError, match="payload sha256 mismatch"):
        load_count_curve_archive(archive_path, require_integrity=True)


def test_nonmonotone_pixel_counts_are_rejected_not_silently_enveloped():
    with pytest.raises(ValueError, match="pixel counts must be non-increasing"):
        build_calibration_losses(
            image_ids=["a"],
            thresholds=np.asarray([0.1, 0.2, 0.3]),
            false_positive_pixels=np.asarray([[1, 2, 0]]),
            false_positive_components=np.asarray([[0, 0, 0]]),
            total_pixels=np.asarray([10]),
            pixel_budget=0.1,
            component_budget=1.0,
        )


def test_default_grid_rank_candidates_cover_full_suffix_and_terminal_reject():
    candidates = build_grid_rank_candidates(zero_index=10, num_thresholds=70)
    assert candidates[0] == 0
    assert 32 in candidates
    assert candidates[-2] == 59
    assert candidates[-1] == 60  # terminal reject rank, not a threshold index


def test_selector_can_choose_offset_larger_than_32():
    curves = np.ones((20, 70), dtype=np.float64)
    curves[:, 40:] = 0.0
    result = select_conformal_offset(curves, zero_index=0, alpha=0.1)
    assert result.success is True
    assert result.reject is False
    assert result.offset_rank == 40
    assert result.selected_threshold_index == 40
    assert result.corrected_loss_bound == pytest.approx(1.0 / 21.0)


def test_shared_offset_cannot_certify_via_partial_calibration_rejects():
    curves = np.ones((20, 6), dtype=np.float64)
    curves[:, 4:] = 0.0
    zero_indices = np.asarray([0] * 10 + [3] * 10)
    result = select_conformal_offset(
        curves, zero_indices=zero_indices, alpha=0.1
    )
    assert result.success is False
    assert result.reject is True
    assert result.zero_index is None
    assert result.offset_rank == curves.shape[1]
    assert result.selected_threshold_index is None
    assert result.selected_threshold_indices == (None,) * 20
    partial = next(item for item in result.candidate_trace if item["offset_rank"] == 4)
    assert partial["reject_rate"] == pytest.approx(0.5)
    assert partial["corrected_bound_satisfies_alpha"] is True
    assert partial["eligible_for_formal_success"] is False


def test_five_shot_alpha_point_one_is_explicitly_infeasible():
    feasible, minimum = finite_sample_feasibility(5, 0.1)
    assert feasible is False
    assert minimum == pytest.approx(1.0 / 6.0)
    result = select_conformal_offset(np.zeros((5, 4)), zero_index=0, alpha=0.1)
    assert result.success is False
    assert result.reject is True
    assert "alpha is below 1/(n+1)" in result.reason
    assert "certified" not in json.dumps(result.to_dict()).lower()


def test_only_terminal_feasible_returns_reject_not_success():
    result = select_conformal_offset(np.ones((20, 8)), zero_index=2, alpha=0.1)
    assert result.success is False
    assert result.reject is True
    assert result.reason == "only_terminal_reject_is_feasible"
    assert result.selected_threshold_index is None
    assert result.offset_rank == 6


def test_split_id_duplicates_and_leakage_are_rejected():
    with pytest.raises(ValueError, match="Duplicate IDs within calibration"):
        assert_disjoint_image_ids(["a", "a"], ["b"])
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint_image_ids(["a", "b"], ["c", "b"])
    with pytest.raises(ValueError, match="adaptation/calibration"):
        assert_three_way_disjoint_image_ids(["warm", "a"], ["a"], ["test"])


def test_calibration_record_and_independent_evaluation():
    calibration = _count_losses("cal")
    test = _count_losses("test")
    result = calibrate_target_offset(
        calibration,
        zero_index=0,
        alpha=0.1,
        test_image_ids=test.image_ids,
    )
    assert result["success"] is True
    assert result["reject"] is False
    assert result["selected_threshold_index"] == 1
    assert result["split_audit"]["overlap_count"] == 0
    assert result["budgets"]["pixel"]["unit"] == PIXEL_BUDGET_UNIT
    audit = evaluate_selected_operating_point(result, test)
    assert audit["success"] is True
    assert audit["test_labels_used_for_selection"] is False
    assert audit["selected_threshold_index"] == result["selected_threshold_index"]
    assert audit["metrics"]["mean_joint_bounded_loss"] == 0.0


def test_sample_adaptive_calibration_applies_shared_rank_to_test_actions():
    calibration = _count_losses("cal")
    test = _count_losses("test")
    calibration_bases = np.asarray([0] * 10 + [1] * 10)
    test_bases = np.asarray([0, 1] * 10)
    result = calibrate_target_offset(
        calibration,
        calibration_zero_indices=calibration_bases,
        test_zero_indices=test_bases,
        alpha=0.1,
        test_image_ids=test.image_ids,
        adaptation_image_ids=["warm-a", "warm-b"],
    )
    assert result["success"] is True
    assert result["adaptation_mode"] == "sample_adaptive_zero_plus_shared_offset"
    assert result["offset_rank"] == 1
    assert result["selected_threshold_index"] is None
    assert result["selected_test_threshold_indices"] == [1, 2] * 10
    audit = evaluate_selected_operating_point(result, test)
    assert audit["success"] is True
    assert audit["selected_threshold_index"] is None
    assert audit["metrics"]["reject_rate"] == 0.0
    assert audit["metrics"]["joint_budget_satisfaction_rate_per_image_suffix_max"] == 1.0


def test_versioned_evaluation_recomputes_actions_from_base_and_selection():
    calibration = _count_losses("cal")
    test = _count_losses("test")
    selection = calibrate_target_offset(
        calibration,
        zero_index=0,
        alpha=0.1,
        test_image_ids=test.image_ids,
    )

    audit = evaluate_selected_operating_point(selection, test)

    assert audit["selected_test_threshold_indices"] == [1] * test.num_images
    assert audit["test_action_contract_audit"]["verified"] is True
    assert (
        audit["test_action_contract_audit"]["mode"]
        == "base_plus_calibration_offset"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("test_action", "test action differs"),
        ("test_base", "Global zero threshold"),
        ("top_offset", "Top-level offset_rank differs"),
        ("selection_offset", "selected actions do not equal base"),
        ("reject", "success/reject flags differ"),
        ("threshold_value", "selected-threshold value"),
        ("result_schema", "Unsupported calibration result schema"),
        ("selection_schema", "selection schema"),
        ("loss_schema", "loss schema"),
        ("formal_upgrade", "conflicts with artifact audits"),
    ],
)
def test_versioned_evaluation_rejects_calibration_action_tampering(
    mutation: str, message: str
):
    calibration = _count_losses("cal")
    test = _count_losses("test")
    original = calibrate_target_offset(
        calibration,
        zero_index=0,
        alpha=0.1,
        test_image_ids=test.image_ids,
    )
    selection = copy.deepcopy(original)
    if mutation == "test_action":
        selection["selected_test_threshold_indices"][0] = 2
    elif mutation == "test_base":
        selection["test_zero_threshold_indices"][0] = 1
    elif mutation == "top_offset":
        selection["offset_rank"] = 2
    elif mutation == "selection_offset":
        selection["offset_rank"] = 2
        selection["selection"]["offset_rank"] = 2
    elif mutation == "reject":
        selection["reject"] = True
    elif mutation == "threshold_value":
        selection["selected_threshold"] = 0.9
    elif mutation == "result_schema":
        selection["schema_version"] = "forged-result-schema"
    elif mutation == "selection_schema":
        selection["selection"]["schema_version"] = "forged-selection-schema"
    elif mutation == "loss_schema":
        selection["loss"]["schema_version"] = "forged-loss-schema"
    elif mutation == "formal_upgrade":
        selection["formal_artifact_chain_verified"] = True
    else:  # pragma: no cover - guards the table above
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=message):
        evaluate_selected_operating_point(selection, test)


def test_formal_flag_cannot_downgrade_recorded_formal_evidence():
    calibration = _count_losses("cal")
    test = _count_losses("test")
    selection = calibrate_target_offset(
        calibration,
        zero_index=0,
        alpha=0.1,
        test_image_ids=test.image_ids,
    )
    selection["protocol_audit"] = {"verified": True}
    selection["zero_artifact_audit"] = {"verified": True}
    selection["provenance"] = {"formal_artifact_chain_verified": True}

    with pytest.raises(ValueError, match="recorded formal evidence"):
        evaluate_selected_operating_point(selection, test)


def test_pre_rejected_calibration_samples_prevent_formal_crc_success():
    calibration = _count_losses("cal")
    test = _count_losses("test")
    calibration_bases = [0] * 10 + [None] * 10
    test_bases = [0, None] * 10
    result = calibrate_target_offset(
        calibration,
        calibration_zero_indices=calibration_bases,
        test_zero_indices=test_bases,
        alpha=0.1,
        test_image_ids=test.image_ids,
        adaptation_image_ids=["warm-a"],
    )
    assert result["success"] is False
    assert result["reject"] is True
    assert result["offset_rank"] == calibration.num_thresholds
    assert result["test_reject_rate"] == 1.0
    assert result["selected_test_threshold_indices"] == []
    audit = evaluate_selected_operating_point(result, test)
    assert audit["success"] is False
    assert audit["metrics"] is None


def test_rejected_calibration_stays_rejected_during_evaluation():
    calibration = _count_losses("cal", num_images=5)
    test = _count_losses("test", num_images=5)
    result = calibrate_target_offset(
        calibration,
        zero_index=0,
        alpha=0.1,
        test_image_ids=test.image_ids,
    )
    audit = evaluate_selected_operating_point(result, test)
    assert audit["success"] is False
    assert audit["reject"] is True
    assert audit["metrics"] is None
    assert "certified" not in audit["mode"].lower()


def test_calibration_and_evaluation_clis_write_provenance_json(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    thresholds = np.linspace(0.0, 0.99, 50)
    calibration_path = tmp_path / "calibration.npz"
    test_path = tmp_path / "test.npz"
    fingerprint = "f" * 64
    _write_count_archive(
        calibration_path,
        [f"cal-{index}" for index in range(20)],
        thresholds,
        transition=35,
        protocol_fingerprint=fingerprint,
    )
    _write_count_archive(
        test_path,
        [f"test-{index}" for index in range(20)],
        thresholds,
        transition=35,
        protocol_fingerprint=fingerprint,
    )
    zero_path = tmp_path / "zero.json"
    zero_path.write_text(
        json.dumps(
            {
                "threshold_index": 0,
                "threshold": float(thresholds[0]),
                "reject": False,
                "pixel_budget": 5e-7,
                "component_budget": 0.5,
                "thresholds": thresholds.tolist(),
                "window_ids": ["warm-0", "warm-1"],
                "adaptation_protocol": "causal_warmup",
                "protocol_fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )
    selection_path = tmp_path / "selection.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "certification.calibrate_target_offset",
            "--calibration-curves",
            str(calibration_path),
            "--test-curves",
            str(test_path),
            "--zero-result",
            str(zero_path),
            "--alpha",
            "0.1",
            "--allow-unverified-protocol",
            "--output",
            str(selection_path),
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["success"] is True
    assert selection["offset_rank"] == 35
    assert selection["provenance"]["test_labels_used_for_selection"] is False
    assert len(selection["provenance"]["calibration_curves_sha256"]) == 64
    assert selection["assumptions"]
    assert selection["split_audit"]["three_way_disjoint_verified"] is True
    assert selection["protocol_audit"]["verified"] is False
    assert selection["formal_artifact_chain_verified"] is False
    assert selection["guarantee_scope"].startswith("none:")

    audit_path = tmp_path / "audit.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "certification.evaluate_certified_mode",
            "--selection-result",
            str(selection_path),
            "--test-curves",
            str(test_path),
            "--output",
            str(audit_path),
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["success"] is True
    assert audit["provenance"]["test_labels_used_for_selection"] is False
    assert audit["metrics"]["mean_joint_bounded_loss"] == 0.0
    assert len(audit["provenance"]["selection_result_sha256"]) == 64
