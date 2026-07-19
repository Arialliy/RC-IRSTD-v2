from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from risk_curve.build_curve_episodes import COMPONENT_RISK_SCHEMA_VERSION
from risk_curve.curve_dataset import LOGIT_EPISODE_SCHEMA_VERSION
from risk_curve.direct_calibrator import ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL
from risk_curve.domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    feature_schema_sha256,
    statistics_names_sha256,
)
from risk_curve.evaluate_gate_c_baselines_v4 import (
    COUNT_ALL_ADAPTATION_SCHEMA_VERSION,
    GATE_C_BASELINES_SCHEMA_VERSION,
    evaluate_gate_c_baselines,
    select_source_baseline_actions,
)
from risk_curve.representation import (
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_REPRESENTATION,
    logit_threshold_grid_sha256,
)


GRID = np.asarray([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
GRID_HASH = logit_threshold_grid_sha256(GRID)
MANIFEST_HASH = "a" * 64
OUTER_HASH = "1" * 64
INNER_HASHES = ("2" * 64, "3" * 64)
ALL_HASHES = (OUTER_HASH, *INNER_HASHES)
NAMES = ("logit_feature_a", "logit_feature_b")
FEATURE_HASH = feature_schema_sha256(
    LOGIT_STATISTICS_SCHEMA_VERSION, statistics_names=NAMES
)
PIXEL_BUDGETS = [1e-5, 1e-6]
COMPONENT_BUDGETS = [5.0, 1.0]


def _provenance() -> dict[str, object]:
    return {
        "protocol": "causal_adaptation_then_future_evaluation",
        "representation": LOGIT_REPRESENTATION,
        "prediction_rule": "prediction = (raw_logits >= threshold)",
        "adaptation_window": 1,
        "evaluation_window": 1,
        "stride": 2,
        "matching_rule": "overlap",
        "centroid_distance": 3.0,
        "connectivity": 2,
        "min_component_area": 1,
        "source_reference": None,
        "source_reference_sha256": None,
        "source_reference_domain_names": [],
        "source_reference_statistics_names_sha256": None,
        "pseudo_targets": ["source-a", "source-b"],
        "validation_domain": "source-b",
        "pseudo_target_split": "train",
        "expected_split_role": "train",
        "threshold_grid_outer_target_key": "outer-c",
        "threshold_grid_outer_target_excluded": True,
        "threshold_grid_source_domains": ["sourcea", "sourceb"],
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": GRID_HASH,
        "threshold_grid_manifest_sha256": MANIFEST_HASH,
        "threshold_grid_detector_protocol": ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": list(ALL_HASHES),
        "threshold_grid_outer_detector_checkpoint_sha256": OUTER_HASH,
        "threshold_grid_episode_detector_checkpoint_sha256s": list(INNER_HASHES),
        "feature_schema_sha256": FEATURE_HASH,
        "fold_provenance_audits": [
            {
                "verified": True,
                "pseudo_target": "source-a",
                "detector_weight_sha256": INNER_HASHES[0],
            },
            {
                "verified": True,
                "pseudo_target": "source-b",
                "detector_weight_sha256": INNER_HASHES[1],
            },
        ],
        "fold_provenance_verified": True,
        "allow_unverified_fold_provenance": False,
        "allow_cross_episode_role_reuse": False,
        "cross_episode_role_reuse_detected": False,
        "formal_causal_contract_verified": True,
        "protocol_scope": "formal_causal",
        "statistics_sample_role": "adaptation_window_A_label_free",
        "risk_label_sample_role": "immediately_following_evaluation_window_E",
        "count_all_adaptation_schema_version": (
            COUNT_ALL_ADAPTATION_SCHEMA_VERSION
        ),
        "count_all_adaptation_sample_role": "adaptation_window_A_label_free",
        "count_all_adaptation_masks_read": False,
        "count_all_adaptation_prediction_rule": (
            "prediction = (raw_logits >= threshold)"
        ),
        "count_all_adaptation_pixel_count_semantics": (
            "pixels retained after connectivity/min_component_area filtering"
        ),
        "count_all_adaptation_component_count_semantics": (
            "connected components retained after min_component_area filtering"
        ),
        "count_all_adaptation_component_envelope": (
            "suffix_max_of_window_aggregate_raw_component_counts"
        ),
    }


def _write_archive(
    path: Path,
    *,
    targets: list[str],
    pixel_fp: np.ndarray,
    component_fp: np.ndarray,
    tp: np.ndarray,
    gt: np.ndarray,
    id_prefix: str,
) -> None:
    rows = len(targets)
    total_pixels = np.full(rows, 1_000_000, dtype=np.int64)
    pixel_fp = np.asarray(pixel_fp, dtype=np.int64)
    component_fp = np.asarray(component_fp, dtype=np.int64)
    tp = np.asarray(tp, dtype=np.int64)
    gt = np.asarray(gt, dtype=np.int64)
    pixel_log = np.log10(
        pixel_fp / total_pixels[:, None] + 1e-12
    ).astype(np.float32)
    component_raw = np.log10(
        component_fp / (total_pixels[:, None] / 1_000_000.0) + 1e-6
    ).astype(np.float32)
    component_upper = np.maximum.accumulate(
        component_raw[:, ::-1], axis=1
    )[:, ::-1]
    pd_curve = (tp / np.maximum(gt[:, None], 1)).astype(np.float32)
    statistics = np.stack(
        [np.arange(rows, dtype=np.float32), np.arange(rows, dtype=np.float32) + 0.5],
        axis=1,
    )
    adaptation_pixel = np.tile(
        np.asarray([100, 8, 0, 0], dtype=np.int64), (rows, 1)
    )
    adaptation_component_raw = np.tile(
        np.asarray([10, 4, 0, 0], dtype=np.int64), (rows, 1)
    )
    adaptation_component_upper = np.maximum.accumulate(
        adaptation_component_raw[:, ::-1], axis=1
    )[:, ::-1]
    np.savez_compressed(
        path,
        statistics=statistics,
        statistics_names=np.asarray(NAMES),
        statistics_names_sha256=np.asarray(statistics_names_sha256(NAMES)),
        statistics_schema_version=np.asarray(LOGIT_STATISTICS_SCHEMA_VERSION),
        feature_schema_sha256=np.asarray(FEATURE_HASH),
        pixel_log_risk=pixel_log,
        component_log_risk=component_upper,
        component_log_risk_raw=component_raw,
        component_log_risk_upper=component_upper,
        component_risk_schema_version=np.asarray(COMPONENT_RISK_SCHEMA_VERSION),
        component_log_risk_alias=np.asarray("component_log_risk_upper"),
        pd_curve=pd_curve,
        thresholds=GRID,
        pixel_fp_counts=pixel_fp,
        component_fp_counts=component_fp,
        tp_object_counts=tp,
        gt_object_counts=gt,
        total_pixels=total_pixels,
        adaptation_predicted_pixel_counts=adaptation_pixel,
        adaptation_predicted_component_counts_raw=adaptation_component_raw,
        adaptation_predicted_component_counts_upper=adaptation_component_upper,
        adaptation_total_pixels=total_pixels.copy(),
        count_all_adaptation_schema_version=np.asarray(
            COUNT_ALL_ADAPTATION_SCHEMA_VERSION
        ),
        pseudo_targets=np.asarray(targets),
        adaptation_ids=np.asarray(
            [json.dumps([f"{id_prefix}-adapt-{index}"]) for index in range(rows)]
        ),
        evaluation_ids=np.asarray(
            [json.dumps([f"{id_prefix}-eval-{index}"]) for index in range(rows)]
        ),
        adaptation_sizes=np.ones(rows, dtype=np.int64),
        evaluation_sizes=np.ones(rows, dtype=np.int64),
        representation=np.asarray(LOGIT_REPRESENTATION),
        threshold_grid_schema_version=np.asarray(LOGIT_GRID_SCHEMA_VERSION),
        threshold_grid_sha256=np.asarray(GRID_HASH),
        threshold_grid_manifest_sha256=np.asarray(MANIFEST_HASH),
        threshold_grid_detector_protocol=np.asarray(
            ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL
        ),
        threshold_grid_detector_checkpoint_sha256s=np.asarray(ALL_HASHES),
        threshold_grid_outer_detector_checkpoint_sha256=np.asarray(OUTER_HASH),
        threshold_grid_episode_detector_checkpoint_sha256s=np.asarray(INNER_HASHES),
        episode_schema_version=np.asarray(LOGIT_EPISODE_SCHEMA_VERSION),
        provenance_json=np.asarray(json.dumps(_provenance(), sort_keys=True)),
    )


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    train = tmp_path / "train.npz"
    validation = tmp_path / "val.npz"
    _write_archive(
        train,
        targets=["source-a", "source-a"],
        # Keep the intended feasible points strictly inside the float32 budget
        # boundary so this fixture tests policy rather than decimal rounding.
        pixel_fp=np.asarray([[100, 9, 1, 1], [100, 9, 0, 0]]),
        component_fp=np.asarray([[10, 4, 1, 1], [10, 4, 0, 0]]),
        tp=np.asarray([[10, 9, 5, 1], [10, 9, 5, 1]]),
        gt=np.asarray([10, 10]),
        id_prefix="train",
    )
    _write_archive(
        validation,
        targets=["source-b"],
        pixel_fp=np.asarray([[50, 20, 0, 0]]),
        component_fp=np.asarray([[5, 2, 0, 0]]),
        tp=np.asarray([[8, 7, 3, 0]]),
        gt=np.asarray([8]),
        id_prefix="validation",
    )
    return train, validation


def test_gate_c_static_worst_use_train_and_audit_validation_counts(
    tmp_path: Path,
) -> None:
    train, validation = _pair(tmp_path)
    output = tmp_path / "baselines.json"
    evaluate_gate_c_baselines(
        train_file=train,
        validation_file=validation,
        output=output,
        pixel_budgets=PIXEL_BUDGETS,
        component_budgets=COMPONENT_BUDGETS,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == GATE_C_BASELINES_SCHEMA_VERSION
    assert payload["representation"] == LOGIT_REPRESENTATION
    assert payload["labels_policy"]["validation_labels_used_for_selection"] is False
    assert payload["labels_policy"]["outer_target_labels_used"] is False
    assert payload["external_reject_action"]["threshold"] == "+inf"
    assert payload["status"] == "COMPLETE"
    assert payload["complete_required_baseline_matrix_ready"] is True

    loose, strict = payload["budgets"]
    for method in ("source_static", "source_worst"):
        assert loose["methods"][method]["selection"]["threshold_index"] == 1
        assert strict["methods"][method]["selection"]["threshold_index"] == 2
        assert loose["methods"][method]["validation_evaluation"][
            "selection_function_received_training_counts_only"
        ] is True
    assert loose["methods"]["count_all"]["validation_evaluation"][
        "per_episode"
    ][0]["selection"]["threshold_index"] == 1
    assert strict["methods"]["count_all"]["validation_evaluation"][
        "per_episode"
    ][0]["selection"]["threshold_index"] == 2
    assert loose["methods"]["count_all"]["validation_evaluation"][
        "adaptation_masks_read"
    ] is False
    loose_metrics = loose["methods"]["source_static"]["validation_evaluation"][
        "aggregate"
    ]
    assert loose_metrics["aggregate_counts"]["pixel_fp_count"] == 20
    assert loose_metrics["aggregate_counts"]["tp_object_count"] == 7
    assert loose_metrics["pd"] == pytest.approx(7 / 8)
    assert loose_metrics["joint_violation_rate"] == 1.0
    worst = loose["methods"]["source_worst"]["selection"]
    assert worst["degenerate_k1"] is True
    assert worst["num_train_pseudo_target_domains"] == 1
    assert "K=1" in worst["degeneracy_note"]

    count_all = payload["count_all"]
    assert count_all["status"] == "AVAILABLE_AND_EVALUATED"
    assert count_all["formal_protocol_eligible"] is True
    assert count_all["evaluated"] is True
    assert count_all["schema_version"] == COUNT_ALL_ADAPTATION_SCHEMA_VERSION
    assert count_all["adaptation_masks_read"] is False
    assert "historical probability-grid Count-all" in count_all[
        "forbidden_fallbacks"
    ]


def test_source_worst_maximizes_min_domain_pd_not_pooled_pd() -> None:
    counts = {
        "thresholds": np.asarray([-2.0, 0.0, 2.0], dtype=np.float32),
        "pixel_fp_counts": np.asarray([[10, 5, 1], [20, 5, 1]]),
        "component_fp_counts": np.asarray([[5, 1, 1], [5, 1, 1]]),
        "tp_object_counts": np.asarray([[10, 9, 6], [10, 4, 6]]),
        "gt_object_counts": np.asarray([10, 10]),
        "total_pixels": np.asarray([1_000_000, 1_000_000]),
        "pseudo_targets": np.asarray(["source-a", "source-b"]),
    }
    selected = select_source_baseline_actions(
        counts, pixel_budget=1e-5, component_budget=5.0
    )
    assert selected["source_static"]["threshold_index"] == 1
    assert selected["source_worst"]["threshold_index"] == 2
    assert selected["source_worst"]["worst_domain_pd"] == pytest.approx(0.6)
    assert selected["source_worst"]["degenerate_k1"] is False


def test_no_finite_source_action_uses_external_positive_infinity_reject() -> None:
    counts = {
        "thresholds": np.asarray([-1.0, 1.0], dtype=np.float32),
        "pixel_fp_counts": np.ones((1, 2), dtype=np.int64),
        "component_fp_counts": np.ones((1, 2), dtype=np.int64),
        "tp_object_counts": np.ones((1, 2), dtype=np.int64),
        "gt_object_counts": np.ones(1, dtype=np.int64),
        "total_pixels": np.asarray([1_000_000]),
        "pseudo_targets": np.asarray(["source-a"]),
    }
    selected = select_source_baseline_actions(
        counts, pixel_budget=1e-9, component_budget=0.1
    )
    for method in selected.values():
        assert method["reject"] is True
        assert method["threshold_index"] is None
        assert method["selected_logit_threshold"] == "+inf"
        assert method["external_reject_action"] is True


def test_gate_c_fails_closed_when_risk_curve_disagrees_with_stored_counts(
    tmp_path: Path,
) -> None:
    train, validation = _pair(tmp_path)
    with np.load(train, allow_pickle=False) as archive:
        tampered = {key: archive[key] for key in archive.files}
    tampered["pixel_log_risk"] = tampered["pixel_log_risk"].copy()
    tampered["pixel_log_risk"][0, 1] += 0.25
    np.savez_compressed(train, **tampered)
    with pytest.raises(ValueError, match="pixel_log_risk.*stored sufficient counts"):
        evaluate_gate_c_baselines(
            train_file=train,
            validation_file=validation,
            output=tmp_path / "must-not-exist.json",
            pixel_budgets=PIXEL_BUDGETS,
            component_budgets=COMPONENT_BUDGETS,
        )


def test_gate_c_rejects_validation_domain_leakage_into_train(tmp_path: Path) -> None:
    train, validation = _pair(tmp_path)
    with np.load(train, allow_pickle=False) as archive:
        leaked = {key: archive[key] for key in archive.files}
    leaked["pseudo_targets"] = np.asarray(["source-b", "source-b"])
    np.savez_compressed(train, **leaked)
    with pytest.raises(ValueError, match="held-out validation pseudo-target"):
        evaluate_gate_c_baselines(
            train_file=train,
            validation_file=validation,
            output=tmp_path / "must-not-exist.json",
            pixel_budgets=PIXEL_BUDGETS,
            component_budgets=COMPONENT_BUDGETS,
        )


def test_count_all_rejects_when_no_finite_A_action_is_feasible(
    tmp_path: Path,
) -> None:
    train, validation = _pair(tmp_path)
    with np.load(validation, allow_pickle=False) as archive:
        changed = {key: archive[key] for key in archive.files}
    changed["adaptation_predicted_pixel_counts"] = np.ones(
        (1, GRID.size), dtype=np.int64
    )
    changed["adaptation_predicted_component_counts_raw"] = np.ones(
        (1, GRID.size), dtype=np.int64
    )
    changed["adaptation_predicted_component_counts_upper"] = np.ones(
        (1, GRID.size), dtype=np.int64
    )
    np.savez_compressed(validation, **changed)
    output = tmp_path / "reject.json"
    evaluate_gate_c_baselines(
        train_file=train,
        validation_file=validation,
        output=output,
        pixel_budgets=[1e-9, 5e-10],
        component_budgets=[0.1, 0.05],
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    for budget in payload["budgets"]:
        episode = budget["methods"]["count_all"]["validation_evaluation"][
            "per_episode"
        ][0]
        assert episode["selection"]["reject"] is True
        assert episode["selection"]["threshold_index"] is None
        assert episode["selection"]["selected_logit_threshold"] == "+inf"
        assert episode["action"]["reject"] is True
        assert episode["action"]["tp_object_count"] == 0
        assert episode["action"]["pixel_fp_count"] == 0


def test_count_all_tampered_upper_curve_fails_closed(tmp_path: Path) -> None:
    train, validation = _pair(tmp_path)
    with np.load(validation, allow_pickle=False) as archive:
        tampered = {key: archive[key] for key in archive.files}
    tampered["adaptation_predicted_component_counts_upper"] = tampered[
        "adaptation_predicted_component_counts_upper"
    ].copy()
    tampered["adaptation_predicted_component_counts_upper"][0, 2] = 1
    np.savez_compressed(validation, **tampered)
    with pytest.raises(ValueError, match="must equal suffix_max"):
        evaluate_gate_c_baselines(
            train_file=train,
            validation_file=validation,
            output=tmp_path / "must-not-exist.json",
            pixel_budgets=PIXEL_BUDGETS,
            component_budgets=COMPONENT_BUDGETS,
        )


def test_count_all_missing_A_contract_fails_closed_without_probability_fallback(
    tmp_path: Path,
) -> None:
    train, validation = _pair(tmp_path)
    for path in (train, validation):
        with np.load(path, allow_pickle=False) as archive:
            legacy = {
                key: archive[key]
                for key in archive.files
                if key
                not in {
                    "adaptation_predicted_pixel_counts",
                    "adaptation_predicted_component_counts_raw",
                    "adaptation_predicted_component_counts_upper",
                    "adaptation_total_pixels",
                    "count_all_adaptation_schema_version",
                }
            }
        np.savez_compressed(path, **legacy)
    with pytest.raises(ValueError, match="predates that sub-contract"):
        evaluate_gate_c_baselines(
            train_file=train,
            validation_file=validation,
            output=tmp_path / "must-not-exist.json",
            pixel_budgets=PIXEL_BUDGETS,
            component_budgets=COMPONENT_BUDGETS,
        )
