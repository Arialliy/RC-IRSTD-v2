from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from risk_curve.anchor_warp_inference import (
    load_anchor_warp_policy,
    predict_anchor_warp_curves,
    prepare_anchor_warp_inputs,
)
from risk_curve.bind_anchor_warp_validation_v4 import (
    FORBIDDEN_BINDER_FIELDS,
    LABEL_FREE_BINDING_FIELDS,
    _load_label_free_fields,
    bind_anchor_warp_validation,
    validate_anchor_warp_bound_package,
)
from risk_curve.count_all_anchor import (
    derive_anchor_log_curves,
    validate_count_all_anchor_archive,
)
from risk_curve.curve_dataset import COUNT_ALL_ADAPTATION_SCHEMA_VERSION
from risk_curve.domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    feature_schema_sha256,
    statistics_names_sha256,
)
from risk_curve.evaluate_anchor_warp_source_pseudo_target_v4 import (
    evaluate_anchor_warp_source_comparison,
)
from risk_curve.train_anchor_warp_predictor_v4 import (
    train_anchor_warp_checkpoint,
)
from risk_curve.train_direct_calibrator_train_only_v4 import (
    train_direct_calibrator_train_only,
)
from tests.test_rc_direct_v4_contract import _write_archive


def _expand_anchor_archive(
    path: Path, *, split: str, rows: int, validation_domain: str
) -> None:
    with np.load(path, allow_pickle=False) as source:
        payload = {field: source[field] for field in source.files}
    thresholds = np.asarray(payload["thresholds"], dtype=np.float32)
    names = tuple(f"feature_{index:03d}" for index in range(119))
    statistics = np.stack(
        [
            np.linspace(-0.5 + row * 0.02, 0.5 + row * 0.02, 119)
            for row in range(rows)
        ]
    ).astype(np.float32)
    pixel_counts = np.tile(
        np.asarray([100, 20, 1, 0], dtype=np.int64), (rows, 1)
    )
    component_raw = np.tile(
        np.asarray([10, 4, 1, 0], dtype=np.int64), (rows, 1)
    )
    component_upper = np.maximum.accumulate(
        component_raw[:, ::-1], axis=1
    )[:, ::-1]
    exposure = np.full(rows, 1_000_000, dtype=np.int64)
    pseudo_target = "NUDT-SIRST" if split == "train" else validation_domain
    adaptation_ids = [[f"{split}-a-{row}"] for row in range(rows)]
    evaluation_ids = [[f"{split}-e-{row}"] for row in range(rows)]
    feature_hash = feature_schema_sha256(
        LOGIT_STATISTICS_SCHEMA_VERSION, statistics_names=names
    )
    provenance = json.loads(str(np.asarray(payload["provenance_json"]).item()))
    provenance.update(
        {
            "validation_domain": validation_domain,
            "archive_split": "train" if split == "train" else "validation",
            "paired_lodo_validation_domains": ["IRSTD-1K", "NUDT-SIRST"],
            "pseudo_target_splits": {
                "IRSTD-1K": "train",
                "NUDT-SIRST": "train",
            },
            "cross_episode_role_reuse_ids": [],
            "allow_cross_episode_role_reuse": False,
            "count_all_adaptation_schema_version": (
                COUNT_ALL_ADAPTATION_SCHEMA_VERSION
            ),
            "count_all_adaptation_sample_role": (
                "adaptation_window_A_label_free"
            ),
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
            "feature_schema_sha256": feature_hash,
        }
    )
    payload.update(
        {
            "statistics": statistics,
            "statistics_names": np.asarray(names),
            "statistics_names_sha256": np.asarray(statistics_names_sha256(names)),
            "feature_schema_sha256": np.asarray(feature_hash),
            "adaptation_sizes": np.ones(rows, dtype=np.int64),
            "evaluation_sizes": np.ones(rows, dtype=np.int64),
            "adaptation_ids": np.asarray(
                [json.dumps(value) for value in adaptation_ids]
            ),
            "evaluation_ids": np.asarray(
                [json.dumps(value) for value in evaluation_ids]
            ),
            "pseudo_targets": np.asarray([pseudo_target] * rows),
            "adaptation_predicted_pixel_counts": pixel_counts,
            "adaptation_predicted_component_counts_raw": component_raw,
            "adaptation_predicted_component_counts_upper": component_upper,
            "adaptation_total_pixels": exposure,
            "count_all_adaptation_schema_version": np.asarray(
                COUNT_ALL_ADAPTATION_SCHEMA_VERSION
            ),
            "provenance_json": np.asarray(json.dumps(provenance, sort_keys=True)),
            "total_pixels": np.full(rows, 1_000_000, dtype=np.int64),
            "gt_object_counts": np.ones(rows, dtype=np.int64),
            "pixel_fp_counts": np.tile(
                np.asarray([40, 8, 1, 0], dtype=np.int64), (rows, 1)
            ),
            "component_fp_counts": np.tile(
                np.asarray([8, 3, 1, 0], dtype=np.int64), (rows, 1)
            ),
            "component_fp_counts_raw": np.tile(
                np.asarray([8, 3, 1, 0], dtype=np.int64), (rows, 1)
            ),
            "component_fp_counts_upper": np.tile(
                np.asarray([8, 3, 1, 0], dtype=np.int64), (rows, 1)
            ),
            "tp_object_counts": np.tile(
                np.asarray([1, 1, 1, 0], dtype=np.int64), (rows, 1)
            ),
        }
    )
    anchors = validate_count_all_anchor_archive(payload)
    pixel_anchor, component_anchor = derive_anchor_log_curves(anchors)
    shift = np.linspace(0.02, 0.08, rows, dtype=np.float32)[:, None]
    payload["pixel_log_risk"] = pixel_anchor + shift
    payload["component_log_risk"] = component_anchor + shift
    payload["component_log_risk_raw"] = component_anchor + shift
    payload["component_log_risk_upper"] = component_anchor + shift
    payload["pd_curve"] = np.ones((rows, thresholds.size), dtype=np.float32)
    np.savez_compressed(path, **payload)


def _pair(root: Path) -> tuple[Path, Path]:
    train = root / "train.npz"
    validation = root / "validation.npz"
    _write_archive(train, split="train")
    _write_archive(validation, split="validation")
    _expand_anchor_archive(
        train, split="train", rows=10, validation_domain="IRSTD-1K"
    )
    _expand_anchor_archive(
        validation, split="validation", rows=6, validation_domain="IRSTD-1K"
    )
    return train, validation


def test_post_freeze_binder_never_loads_future_e_labels_and_inference_is_shared(
    tmp_path: Path,
) -> None:
    train, validation = _pair(tmp_path)
    frozen_path = tmp_path / "frozen.pt"
    bound_path = tmp_path / "bound.pt"
    train_anchor_warp_checkpoint(
        train_file=train,
        output=frozen_path,
        max_epochs=1,
        patience=1,
        seed=3407,
        left_radius=2,
        right_radius=1,
        device="cpu",
    )
    bind_anchor_warp_validation(
        frozen_checkpoint=frozen_path,
        validation_file=validation,
        output=bound_path,
    )
    package = torch.load(bound_path, map_location="cpu", weights_only=True)
    contract = validate_anchor_warp_bound_package(package)
    binding = package["validation_binding"]
    assert contract["formal_source_evaluation_bound"] is True
    assert binding["validation_labels_read_by_binder"] is False
    assert binding["accessed_npz_fields"] == list(LABEL_FREE_BINDING_FIELDS)
    assert binding["forbidden_npz_fields_accessed"] == []
    assert set(binding["accessed_npz_fields"]).isdisjoint(FORBIDDEN_BINDER_FIELDS)

    raw = validation.read_bytes()
    label_free = _load_label_free_fields(raw, role="validation")
    policy = load_anchor_warp_policy(package)
    inputs = prepare_anchor_warp_inputs(label_free, policy)
    prediction = predict_anchor_warp_curves(policy, inputs, batch_size=3)
    assert inputs.preprocessing_audit["hard_reject_applied"] is False
    for curve in prediction.values():
        assert curve.shape == (6, 4)
        assert np.isfinite(curve).all()
        assert np.all(np.diff(curve, axis=1) <= 1.0e-6)


def test_changing_only_future_e_targets_changes_binding_file_sha_not_policy(
    tmp_path: Path,
) -> None:
    train, validation = _pair(tmp_path)
    changed = tmp_path / "validation-changed.npz"
    with np.load(validation, allow_pickle=False) as source:
        payload = {field: source[field] for field in source.files}
    for field in (
        "pixel_log_risk",
        "component_log_risk",
        "component_log_risk_raw",
        "component_log_risk_upper",
    ):
        payload[field] = np.asarray(payload[field], dtype=np.float32) + 0.75
    np.savez_compressed(changed, **payload)
    frozen = tmp_path / "frozen.pt"
    train_anchor_warp_checkpoint(
        train_file=train,
        output=frozen,
        max_epochs=1,
        patience=1,
        seed=17,
        left_radius=2,
        right_radius=1,
    )
    first_path = bind_anchor_warp_validation(
        frozen_checkpoint=frozen,
        validation_file=validation,
        output=tmp_path / "bound-first.pt",
    )
    changed_path = bind_anchor_warp_validation(
        frozen_checkpoint=frozen,
        validation_file=changed,
        output=tmp_path / "bound-changed.pt",
    )
    first = torch.load(first_path, map_location="cpu", weights_only=True)
    second = torch.load(changed_path, map_location="cpu", weights_only=True)
    assert first["validation_binding"]["validation_archive_sha256"] != second[
        "validation_binding"
    ]["validation_archive_sha256"]
    assert first["validation_binding"][
        "validation_label_free_input_semantic_sha256"
    ] == second["validation_binding"][
        "validation_label_free_input_semantic_sha256"
    ]
    for field in (
        "state_dict_semantic_sha256",
        "policy_semantic_sha256",
        "selected_epoch",
        "pixel_oof_inflation",
        "component_oof_inflation",
    ):
        assert first["frozen_checkpoint"][field] == second["frozen_checkpoint"][field]

    tampered = copy.deepcopy(first)
    tampered["frozen_checkpoint"]["state_dict"]["controller.bias"][0] += 1.0
    with pytest.raises(ValueError, match="state hash"):
        validate_anchor_warp_bound_package(tampered)


def test_two_phase_evaluator_freezes_actions_before_future_e_counts(
    tmp_path: Path,
) -> None:
    train, validation = _pair(tmp_path)
    frozen = tmp_path / "anchor-frozen.pt"
    bound = tmp_path / "anchor-bound.pt"
    direct = tmp_path / "direct.pt"
    comparison = tmp_path / "comparison.json"
    train_anchor_warp_checkpoint(
        train_file=train,
        output=frozen,
        max_epochs=1,
        patience=1,
        seed=42,
        left_radius=2,
        right_radius=1,
    )
    bind_anchor_warp_validation(
        frozen_checkpoint=frozen,
        validation_file=validation,
        output=bound,
    )
    train_direct_calibrator_train_only(
        train_file=train,
        validation_file=validation,
        output=direct,
        pixel_budgets=[1.0e-5, 1.0e-6],
        component_budgets=[5.0, 1.0],
        hidden_dims=(4,),
        dropout=0.0,
        max_epochs=1,
        batch_size=5,
        learning_rate=2.0e-3,
        weight_decay=0.0,
        under_weight=2.0,
        seed=42,
        device="cpu",
    )
    evaluate_anchor_warp_source_comparison(
        episode_file=validation,
        anchor_warp_package=bound,
        rc_direct_checkpoint=direct,
        output=comparison,
        device="cpu",
    )
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    assert payload["labels_used_for_action_selection"] is False
    assert payload["future_e_arrays_loaded_before_action_digest"] is False
    assert payload["action_freeze"]["frozen_before_future_e_load"] is True
    assert payload["action_freeze"]["future_e_reselection_performed"] is False
    assert payload["action_freeze"]["num_actions"] == 12
    assert len(payload["action_freeze"]["action_digest_sha256"]) == 64
