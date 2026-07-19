from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from certification.build_calibration_losses import (
    build_calibration_losses,
    build_count_curves_from_score_maps,
    count_archive_payload_sha256,
    save_calibration_losses,
    score_map_protocol,
)
from data_ext.mask_alignment import (
    MASK_ALIGNMENT_NOT_LOADED_POLICY,
    MASK_ALIGNMENT_POLICY,
)
from evaluation.artifact_integrity import (
    RAW_LOGIT_SCORE_REPRESENTATION,
    SCORE_MANIFEST_SCHEMA_VERSION,
    SCORE_MASK_ALIGNMENT_SCHEMA,
    SCORE_RECORD_INTEGRITY_SCHEMA,
    file_sha256,
    ordered_ids_sha256,
    score_records_sha256,
    verify_score_map_directory,
)
from evaluation.evaluate_selected_actions import evaluate_selected_actions
from evaluation.standard_metrics import evaluate_standard_score_directory
from evaluation.target_stage_separation import (
    audit_target_score_stage_pair,
    freeze_zero_label_actions,
)
from risk_curve.select_zero_label_threshold import (
    SELECTION_DATA_CONTRACT_SCHEMA_VERSION,
    ZERO_RESULT_SCHEMA_VERSION,
)
from risk_curve.representation import (
    GRID_DETECTOR_PROTOCOL,
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_PREDICTION_RULE,
    LOGIT_REPRESENTATION,
    empty_action_contract,
    logit_threshold_grid_sha256,
)


DETECTOR_SHA = "a" * 64
PROTOCOL_FINGERPRINT = "b" * 64
INNER_A_SHA = "c" * 64
INNER_B_SHA = "d" * 64
GRID_MANIFEST_SHA = "e" * 64


def _raw_score_directory(
    root: Path,
    logits: list[np.ndarray],
    masks: list[np.ndarray],
    *,
    model_backend: str = "rc_mshnet",
    labels_loaded: bool = True,
) -> Path:
    root.mkdir(parents=True)
    image_ids = [f"image-{index}" for index in range(len(logits))]
    records: list[dict[str, object]] = []
    for image_id, raw_logit, mask in zip(image_ids, logits, masks):
        raw_logit = np.asarray(raw_logit, dtype=np.float32)
        probability = (1.0 / (1.0 + np.exp(-raw_logit))).astype(np.float32)
        height, width = raw_logit.shape
        filename = f"{image_id}.npz"
        record_arrays = {
            "prob": probability,
            "logit": raw_logit,
            "gray": np.zeros_like(raw_logit, dtype=np.float32),
            "image_id": np.asarray(image_id),
            "labels_loaded": np.asarray(labels_loaded),
            "spatial_mode": np.asarray("native"),
            "original_hw": np.asarray([height, width], dtype=np.int32),
            "input_hw": np.asarray([height, width], dtype=np.int32),
            "valid_hw": np.asarray([height, width], dtype=np.int32),
            "padding_ltrb": np.asarray([0, 0, 0, 0], dtype=np.int32),
            "mask_alignment_applied": np.asarray(False),
            "mask_original_hw": np.asarray(
                [height, width] if labels_loaded else [0, 0],
                dtype=np.int32,
            ),
            "mask_aspect_relative_error": np.asarray(
                0.0 if labels_loaded else -1.0,
                dtype=np.float64,
            ),
            "mask_alignment_policy": np.asarray(
                MASK_ALIGNMENT_POLICY
                if labels_loaded
                else MASK_ALIGNMENT_NOT_LOADED_POLICY
            ),
            "score_representation": np.asarray(RAW_LOGIT_SCORE_REPRESENTATION),
            "probability_dtype": np.asarray("float32"),
            "logit_dtype": np.asarray("float32"),
            "probability_transform": np.asarray("sigmoid"),
            "probability_clipping": np.asarray("none"),
            "inference_autocast_enabled": np.asarray(False),
        }
        if labels_loaded:
            record_arrays["mask"] = np.asarray(mask, dtype=np.uint8)
        np.savez_compressed(root / filename, **record_arrays)
        records.append(
            {
                "image_id": image_id,
                "file": filename,
                "shape": [height, width],
                "sha256": file_sha256(root / filename),
                "mask_alignment_applied": False,
                "mask_original_hw": (
                    [height, width] if labels_loaded else [0, 0]
                ),
                "mask_aspect_relative_error": 0.0 if labels_loaded else -1.0,
                "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
                "probability_dtype": "float32",
                "logit_dtype": "float32",
                "probability_transform": "sigmoid",
                "probability_clipping": "none",
                "inference_autocast_enabled": False,
            }
        )
    split_file = root / "official-test.txt"
    split_file.write_text("\n".join(image_ids) + "\n", encoding="utf-8")
    ids_sha = ordered_ids_sha256(image_ids)
    manifest = {
        "schema_version": SCORE_MANIFEST_SCHEMA_VERSION,
        "record_integrity_schema": SCORE_RECORD_INTEGRITY_SCHEMA,
        "score_type": "sigmoid_probability",
        "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
        "probability_dtype": "float32",
        "logit_dtype": "float32",
        "probability_transform": "sigmoid",
        "probability_clipping": "none",
        "inference_autocast_enabled": False,
        "warm_flag": True,
        "labels_loaded": labels_loaded,
        "num_images": len(records),
        "records": records,
        "records_sha256": score_records_sha256(records),
        "ordered_image_ids_sha256": ids_sha,
        "target_dataset": "held-out-target",
        "source_datasets": ["source-a", "source-b"],
        "split_file": str(split_file.resolve()),
        "split_file_sha256": file_sha256(split_file),
        "split_ordered_ids_sha256": ids_sha,
        "requested_split": "test",
        "split_role": "test",
        "split_authority_verified": True,
        "spatial_mode": "native",
        "base_hw": None,
        "pad_multiple": 16,
        "mask_alignment_schema": SCORE_MASK_ALIGNMENT_SCHEMA,
        "mask_alignment_policy": (
            MASK_ALIGNMENT_POLICY
            if labels_loaded
            else MASK_ALIGNMENT_NOT_LOADED_POLICY
        ),
        "mask_alignment_count": 0,
        "mask_aligned_sample_ids": [],
        "weight_sha256": DETECTOR_SHA,
        "checkpoint_selection_rule": "fixed_last",
        "checkpoint_diagnostic_only": False,
        "non_strict_state_loading": False,
        "model_backend": model_backend,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return root


def _selection(
    score_dir: Path,
    actions: dict[str, int | None],
    thresholds: list[float],
) -> dict[str, object]:
    _, _, integrity = verify_score_map_directory(
        score_dir,
        require_integrity=True,
        require_masks=True,
    )
    return {
        "adaptation_protocol": "static_cross_fit",
        "threshold_indices_by_image": actions,
        "thresholds": thresholds,
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "protocol": {"detector_weight_sha256": DETECTOR_SHA},
        "statistics_artifact": {
            "provenance": {
                "score_map_dir": str(score_dir.resolve()),
                "score_manifest_sha256": integrity["manifest_sha256"],
                "score_records_sha256": integrity["records_sha256"],
                "score_ordered_image_ids_sha256": integrity[
                    "ordered_image_ids_sha256"
                ],
                "score_num_records": integrity["num_records"],
            }
        },
    }


def _example(tmp_path: Path, *, backend: str = "rc_mshnet") -> tuple[Path, dict[str, object]]:
    logits = [
        np.asarray([[0.0, -2.0], [-2.0, 2.0]], dtype=np.float32),
        np.full((2, 2), -2.0, dtype=np.float32),
    ]
    masks = [
        np.asarray([[1, 1], [0, 0]], dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint8),
    ]
    score_dir = _raw_score_directory(
        tmp_path / f"scores-{backend}", logits, masks, model_backend=backend
    )
    selection = _selection(score_dir, {"image-0": 0, "image-1": 0}, [0.0, 1.0])
    return score_dir, selection


def _count_archive_for_score_dir(
    output_dir: Path,
    score_dir: Path,
) -> tuple[Path, dict[str, object], str, dict[str, object]]:
    grid = np.asarray([0.0, 1.0], dtype=np.float32)
    detector_hashes = [DETECTOR_SHA, INNER_A_SHA, INNER_B_SHA]
    episode_hashes = [INNER_A_SHA, INNER_B_SHA]
    counts = build_count_curves_from_score_maps(
        score_dir,
        grid,
        matching_rule="overlap",
        centroid_distance=3.0,
        connectivity=2,
        min_component_area=1,
        require_integrity=True,
        representation=LOGIT_REPRESENTATION,
    )
    protocol, fingerprint = score_map_protocol(
        score_dir,
        grid,
        matching_rule="overlap",
        centroid_distance=3.0,
        connectivity=2,
        min_component_area=1,
        representation=LOGIT_REPRESENTATION,
        threshold_grid_detector_protocol=GRID_DETECTOR_PROTOCOL,
        threshold_grid_detector_checkpoint_sha256s=detector_hashes,
        threshold_grid_outer_detector_checkpoint_sha256=DETECTOR_SHA,
        threshold_grid_episode_detector_checkpoint_sha256s=episode_hashes,
    )
    grid_sha = logit_threshold_grid_sha256(grid)
    losses = build_calibration_losses(
        **counts,
        pixel_budget=1.0e-6,
        component_budget=1.0,
        representation=LOGIT_REPRESENTATION,
        threshold_grid_schema_version=LOGIT_GRID_SCHEMA_VERSION,
        recorded_threshold_grid_sha256=grid_sha,
        threshold_grid_manifest_sha256=GRID_MANIFEST_SHA,
        threshold_grid_detector_protocol=GRID_DETECTOR_PROTOCOL,
        threshold_grid_detector_checkpoint_sha256s=detector_hashes,
        threshold_grid_outer_detector_checkpoint_sha256=DETECTOR_SHA,
        threshold_grid_episode_detector_checkpoint_sha256s=episode_hashes,
    )
    _, _, integrity = verify_score_map_directory(
        score_dir,
        require_integrity=True,
        require_masks=True,
    )
    provenance = {
        "source_type": "exported_score_map_directory",
        "score_dir": str(score_dir.resolve()),
        "manifest_sha256": integrity["manifest_sha256"],
        "score_records_sha256": integrity["records_sha256"],
        "score_ordered_image_ids_sha256": integrity["ordered_image_ids_sha256"],
        "score_num_records": integrity["num_records"],
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": grid_sha,
        "threshold_grid_manifest_sha256": GRID_MANIFEST_SHA,
        "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": detector_hashes,
        "threshold_grid_outer_detector_checkpoint_sha256": DETECTOR_SHA,
        "threshold_grid_episode_detector_checkpoint_sha256s": episode_hashes,
        "protocol": protocol,
        "protocol_fingerprint": fingerprint,
    }
    count_path = save_calibration_losses(
        output_dir / "calibration_losses.npz",
        losses,
        provenance=provenance,
    )
    shared_contract = {
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": grid_sha,
        "threshold_grid_manifest_sha256": GRID_MANIFEST_SHA,
        "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": detector_hashes,
        "threshold_grid_outer_detector_checkpoint_sha256": DETECTOR_SHA,
        "threshold_grid_episode_detector_checkpoint_sha256s": episode_hashes,
        "prediction_rule": LOGIT_PREDICTION_RULE,
        "empty_action": empty_action_contract(),
    }
    return count_path, protocol, fingerprint, shared_contract


def _bound_count_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, object], Path]:
    score_dir, selection = _example(tmp_path)
    count_path, protocol, fingerprint, shared_contract = (
        _count_archive_for_score_dir(tmp_path, score_dir)
    )
    selection.update(
        {
            **shared_contract,
            "pixel_budget": 1.0e-6,
            "component_budget": 1.0,
            "protocol": protocol,
            "protocol_fingerprint": fingerprint,
            "selection_data_contract": dict(shared_contract),
        }
    )
    return score_dir, selection, count_path


def _formal_pair_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, object], Path, dict[str, object], str]:
    logits = [
        np.asarray([[0.0, -2.0], [-2.0, 2.0]], dtype=np.float32),
        np.full((2, 2), -2.0, dtype=np.float32),
    ]
    masks = [
        np.asarray([[1, 1], [0, 0]], dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint8),
    ]
    unlabeled = _raw_score_directory(
        tmp_path / "scores-unlabeled",
        logits,
        masks,
        labels_loaded=False,
    )
    _, _, unlabeled_integrity = verify_score_map_directory(
        unlabeled,
        require_integrity=True,
        require_masks=False,
    )
    grid = np.asarray([0.0, 1.0], dtype=np.float32)
    detector_hashes = [DETECTOR_SHA, INNER_A_SHA, INNER_B_SHA]
    episode_hashes = [INNER_A_SHA, INNER_B_SHA]
    protocol, fingerprint = score_map_protocol(
        unlabeled,
        grid,
        matching_rule="overlap",
        centroid_distance=3.0,
        connectivity=2,
        min_component_area=1,
        representation=LOGIT_REPRESENTATION,
        threshold_grid_detector_protocol=GRID_DETECTOR_PROTOCOL,
        threshold_grid_detector_checkpoint_sha256s=detector_hashes,
        threshold_grid_outer_detector_checkpoint_sha256=DETECTOR_SHA,
        threshold_grid_episode_detector_checkpoint_sha256s=episode_hashes,
    )
    shared_contract = {
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": logit_threshold_grid_sha256(grid),
        "threshold_grid_manifest_sha256": GRID_MANIFEST_SHA,
        "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": detector_hashes,
        "threshold_grid_outer_detector_checkpoint_sha256": DETECTOR_SHA,
        "threshold_grid_episode_detector_checkpoint_sha256s": episode_hashes,
        "prediction_rule": LOGIT_PREDICTION_RULE,
        "empty_action": empty_action_contract(),
    }
    image_ids = ["image-0", "image-1"]
    statistics_provenance = {
        "score_map_dir": str(unlabeled.resolve()),
        "score_manifest_sha256": unlabeled_integrity["manifest_sha256"],
        "score_records_sha256": unlabeled_integrity["records_sha256"],
        "score_ordered_image_ids_sha256": unlabeled_integrity[
            "ordered_image_ids_sha256"
        ],
        "score_num_records": unlabeled_integrity["num_records"],
        "masks_read": False,
        "full_test_coverage": True,
        "formal_crc_eligible": False,
        "score_integrity_verified": True,
    }
    selection: dict[str, object] = {
        "schema_version": ZERO_RESULT_SCHEMA_VERSION,
        "mode": "zero_label_empirical_adaptation",
        "masks_read": False,
        "adaptation_protocol": "static_cross_fit",
        "num_windows": 2,
        "evaluation_ids": [[image_ids[0]], [image_ids[1]]],
        "threshold_indices": [0, 0],
        "threshold_indices_by_image": {image_id: 0 for image_id in image_ids},
        "thresholds": grid.tolist(),
        "pixel_budget": 1.0e-6,
        "component_budget": 1.0,
        "selection_data_contract": {
            "schema_version": SELECTION_DATA_CONTRACT_SCHEMA_VERSION,
            "masks_read": False,
            "statistics_computed_from": "static_cross_fit_complement_folds",
            "evaluation_labels_or_masks_used": False,
            "threshold_mapping_rule": (
                "one_complement_fold_prediction_to_held_out_fold_ids"
            ),
            "deployment_identity_contract_verified": True,
            "formal_crc_eligible": False,
            "static_checkpoint_compatibility_verified": True,
            **shared_contract,
        },
        "statistics_artifact": {
            "identity_contract": {"verified": True},
            "provenance": statistics_provenance,
        },
        "static_checkpoint_compatibility_audit": {"verified": True},
        "protocol": protocol,
        "protocol_fingerprint": fingerprint,
        **shared_contract,
    }
    selection_path = tmp_path / "zero_selection.json"
    selection_path.write_text(
        json.dumps(selection, sort_keys=True),
        encoding="utf-8",
    )
    freeze_record = freeze_zero_label_actions(
        [selection_path],
        bound_artifacts=[unlabeled / "manifest.json"],
        output_dir=tmp_path / "freeze",
    )

    labeled = _raw_score_directory(
        tmp_path / "scores-labeled-audit",
        logits,
        masks,
        labels_loaded=True,
    )
    pair_audit = audit_target_score_stage_pair(
        unlabeled,
        labeled,
        freeze_record=freeze_record,
        output=tmp_path / "target-stage-pair-audit.json",
    )
    count_path, labeled_protocol, labeled_fingerprint, labeled_contract = (
        _count_archive_for_score_dir(tmp_path, labeled)
    )
    assert labeled_protocol == protocol
    assert labeled_fingerprint == fingerprint
    assert labeled_contract == shared_contract
    return labeled, selection, count_path, pair_audit, file_sha256(selection_path)


def test_selected_actions_all_scalar_equals_standard_metrics(tmp_path: Path) -> None:
    score_dir, selection = _example(tmp_path)
    selected = evaluate_selected_actions(
        score_dir,
        selection,
        allow_unverified_diagnostic=True,
    )
    standard = evaluate_standard_score_directory(score_dir, 0.5)
    assert selected["metrics_all"] == standard["metrics"]
    assert selected["counts_all"] == standard["counts"]


def test_selected_actions_reject_is_empty_prediction_and_reports_active(tmp_path: Path) -> None:
    score_dir, selection = _example(tmp_path)
    selection["threshold_indices_by_image"] = {"image-0": None, "image-1": 0}
    result = evaluate_selected_actions(
        score_dir,
        selection,
        allow_unverified_diagnostic=True,
    )
    assert result["coverage_rate"] == 0.5
    assert result["reject_rate"] == 0.5
    assert result["counts_all"]["pixel_tp"] == 0
    assert result["counts_all"]["pixel_fn"] == 2
    assert result["counts_active_only"]["num_samples"] == 1
    assert result["selected_actions"][0]["action"] == "no_detection_reject"


def test_selected_actions_raw_logit_threshold_rule_is_greater_equal(tmp_path: Path) -> None:
    score_dir, selection = _example(tmp_path)
    result = evaluate_selected_actions(
        score_dir,
        selection,
        allow_unverified_diagnostic=True,
    )
    assert result["counts_all"]["pixel_tp"] == 1
    assert result["protocol"]["prediction_rule"] == (
        "prediction = (raw_logits >= threshold)"
    )


def test_selected_actions_image_id_mapping_must_be_exact(tmp_path: Path) -> None:
    score_dir, selection = _example(tmp_path)
    selection["threshold_indices_by_image"] = {"image-0": 0}
    with pytest.raises(ValueError, match="cover every evaluated image ID"):
        evaluate_selected_actions(
            score_dir,
            selection,
            allow_unverified_diagnostic=True,
        )
    selection["threshold_indices_by_image"] = {
        "image-0": 0,
        "image-1": 0,
        "extra": 0,
    }
    with pytest.raises(ValueError, match="absent from the score manifest"):
        evaluate_selected_actions(
            score_dir,
            selection,
            allow_unverified_diagnostic=True,
        )


def test_selected_actions_rejects_manifest_selection_mismatch(tmp_path: Path) -> None:
    score_dir, selection = _example(tmp_path)
    selection["statistics_artifact"]["provenance"]["score_records_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differ in score_records_sha256"):
        evaluate_selected_actions(
            score_dir,
            selection,
            allow_unverified_diagnostic=True,
        )


@pytest.mark.parametrize("backend", ["canonical", "rc_mshnet"])
def test_selected_actions_accepts_canonical_and_rc_mshnet(
    tmp_path: Path,
    backend: str,
) -> None:
    score_dir, selection = _example(tmp_path, backend=backend)
    result = evaluate_selected_actions(
        score_dir,
        selection,
        allow_unverified_diagnostic=True,
    )
    assert result["protocol"]["model_backend"] == backend


def test_selected_actions_real_count_archive_is_recomputed_per_image(
    tmp_path: Path,
) -> None:
    score_dir, selection, count_path = _bound_count_fixture(tmp_path)
    result = evaluate_selected_actions(
        score_dir,
        selection,
        count_curves=count_path,
        allow_unverified_diagnostic=True,
    )
    assert result["selected_count_consistency_verified"] is True
    assert len(result["per_image_count_audits"]) == 2
    assert result["count_archive_payload_sha256"]
    assert result["formal_protocol_eligible"] is False


def test_selected_actions_formal_paired_count_happy_path(tmp_path: Path) -> None:
    score_dir, selection, count_path, pair_audit, selection_sha = (
        _formal_pair_fixture(tmp_path)
    )
    result = evaluate_selected_actions(
        score_dir,
        selection,
        count_curves=count_path,
        selection_sha256=selection_sha,
        target_stage_pair_audit=pair_audit,
    )
    assert result["formal_protocol_eligible"] is True
    assert result["formal_empirical_evaluation_eligible"] is True
    assert result["formal_crc_eligible"] is False
    assert result["selected_count_consistency_verified"] is True
    assert result["score_binding_audit"]["binding_mode"] == (
        "paired_unlabeled_selection_and_labeled_audit"
    )


def test_selected_actions_rejects_integrity_valid_but_wrong_selected_count(
    tmp_path: Path,
) -> None:
    score_dir, selection, count_path = _bound_count_fixture(tmp_path)
    with np.load(count_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["false_positive_pixels"] = arrays["false_positive_pixels"].copy()
    arrays["false_positive_pixels"][0, 0] += 1
    arrays["count_archive_payload_sha256"] = np.asarray(
        count_archive_payload_sha256(arrays)
    )
    np.savez_compressed(count_path, **arrays)
    with pytest.raises(ValueError, match="count consistency mismatch"):
        evaluate_selected_actions(
            score_dir,
            selection,
            count_curves=count_path,
            allow_unverified_diagnostic=True,
        )


def test_selected_actions_rejects_grid_and_empty_action_drift(tmp_path: Path) -> None:
    score_dir, selection, count_path = _bound_count_fixture(tmp_path)
    selection["empty_action"] = {"threshold": "+inf"}
    with pytest.raises(ValueError, match="empty_action differs"):
        evaluate_selected_actions(
            score_dir,
            selection,
            count_curves=count_path,
            allow_unverified_diagnostic=True,
        )

    score_dir, selection, count_path = _bound_count_fixture(tmp_path / "second")
    selection["thresholds"] = [-1.0, 1.0]
    with pytest.raises(ValueError, match="threshold grids differ"):
        evaluate_selected_actions(
            score_dir,
            selection,
            count_curves=count_path,
            allow_unverified_diagnostic=True,
        )


def test_selected_actions_formal_requires_count_and_paired_stage_audit(
    tmp_path: Path,
) -> None:
    score_dir, selection, count_path = _bound_count_fixture(tmp_path)
    with pytest.raises(ValueError, match="requires count_curves"):
        evaluate_selected_actions(score_dir, selection)
    with pytest.raises(ValueError, match="requires target_stage_pair_audit"):
        evaluate_selected_actions(
            score_dir,
            selection,
            count_curves=count_path,
        )


def test_selected_actions_all_reject_reports_null_active_metrics(tmp_path: Path) -> None:
    score_dir, selection, count_path = _bound_count_fixture(tmp_path)
    selection["threshold_indices_by_image"] = {"image-0": None, "image-1": None}
    result = evaluate_selected_actions(
        score_dir,
        selection,
        count_curves=count_path,
        allow_unverified_diagnostic=True,
    )
    assert result["coverage_rate"] == 0.0
    assert result["reject_rate"] == 1.0
    assert all(value is None for value in result["metrics_active_only"].values())
    assert result["counts_all"]["pixel_fn"] == 2
    assert result["counts_all"]["num_tp_objects"] == 0
