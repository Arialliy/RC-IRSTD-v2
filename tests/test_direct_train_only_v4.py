from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from rc_irstd.models.calibrator import MonotoneBudgetCalibrator
from risk_curve.train_direct_calibrator_train_only_v4 import (
    TRAIN_ONLY_CV_FOLDS,
    canonical_state_dict_sha256,
    deterministic_five_fold_indices,
    train_direct_calibrator_train_only,
    validate_train_only_direct_checkpoint,
)
from tests.test_rc_direct_v4_contract import _write_archive


PIXEL_BUDGETS = [1e-1, 1e-3]
COMPONENT_BUDGETS = [1.0, 1e-2]


def _expand_archive(
    path: Path,
    *,
    split: str,
    num_rows: int,
    label_shift: float = 0.0,
) -> None:
    with np.load(path, allow_pickle=False) as source:
        payload = {name: source[name] for name in source.files}
    statistics = np.stack(
        [
            np.asarray(
                [float(index) / max(num_rows - 1, 1), float((index * 3) % 7) / 7.0],
                dtype=np.float32,
            )
            for index in range(num_rows)
        ]
    )
    pixel_rows: list[np.ndarray] = []
    component_rows: list[np.ndarray] = []
    for index in range(num_rows):
        pixel_rows.append(
            np.asarray([-0.5, -1.4, -2.5, -3.4], dtype=np.float32)
            - 0.25 * float(index % 3)
            - float(label_shift)
        )
        component_rows.append(
            np.asarray([0.5, 0.0, -1.0, -2.0], dtype=np.float32)
            - 0.2 * float(index % 2)
            - float(label_shift)
        )
    pixel = np.stack(pixel_rows).astype(np.float32)
    component = np.stack(component_rows).astype(np.float32)
    payload.update(
        {
            "statistics": statistics,
            "pixel_log_risk": pixel,
            "component_log_risk": component,
            "component_log_risk_raw": component.copy(),
            "component_log_risk_upper": component.copy(),
            "pd_curve": np.ones_like(pixel),
            "adaptation_sizes": np.ones(num_rows, dtype=np.int64),
            "evaluation_sizes": np.ones(num_rows, dtype=np.int64),
            "adaptation_ids": np.asarray(
                [json.dumps([f"{split}-a-{index}"]) for index in range(num_rows)]
            ),
            "evaluation_ids": np.asarray(
                [json.dumps([f"{split}-e-{index}"]) for index in range(num_rows)]
            ),
            "pseudo_targets": np.asarray(
                ["NUDT-SIRST" if split == "train" else "IRSTD-1K"] * num_rows
            ),
        }
    )
    np.savez_compressed(path, **payload)


def _archives(
    root: Path,
    *,
    validation_label_shift: float = 0.0,
) -> tuple[Path, Path]:
    train = root / "train.npz"
    validation = root / "validation.npz"
    _write_archive(train, split="train", outer="held-outer")
    _write_archive(validation, split="validation", outer="held-outer")
    _expand_archive(train, split="train", num_rows=10)
    _expand_archive(
        validation,
        split="validation",
        num_rows=6,
        label_shift=validation_label_shift,
    )
    return train, validation


def _train(
    train: Path,
    validation: Path,
    output: Path,
    *,
    observer=None,
) -> dict[str, object]:
    train_direct_calibrator_train_only(
        train_file=train,
        validation_file=validation,
        output=output,
        pixel_budgets=PIXEL_BUDGETS,
        component_budgets=COMPONENT_BUDGETS,
        hidden_dims=(4,),
        dropout=0.0,
        max_epochs=3,
        batch_size=4,
        learning_rate=2e-3,
        weight_decay=0.0,
        under_weight=2.0,
        seed=3407,
        device="cpu",
        _event_observer=observer,
    )
    return torch.load(output, map_location="cpu", weights_only=True)


def test_deterministic_five_fold_partition_is_exhaustive_without_overlap() -> None:
    first = deterministic_five_fold_indices(13, seed=3407)
    second = deterministic_five_fold_indices(13, seed=3407)
    assert len(first) == TRAIN_ONLY_CV_FOLDS == 5
    holdouts: list[int] = []
    for (train, validation), (train_again, validation_again) in zip(first, second):
        assert np.array_equal(train, train_again)
        assert np.array_equal(validation, validation_again)
        assert not np.intersect1d(train, validation).size
        assert np.array_equal(
            np.sort(np.concatenate([train, validation])), np.arange(13)
        )
        holdouts.extend(validation.tolist())
    assert sorted(holdouts) == list(range(13))
    assert len(set(holdouts)) == 13


def test_training_is_deterministic_and_frozen_checkpoint_restores_predictions(
    tmp_path: Path,
) -> None:
    train, validation = _archives(tmp_path)
    first = _train(train, validation, tmp_path / "first.pt")
    second = _train(train, validation, tmp_path / "second.pt")
    first_audit = validate_train_only_direct_checkpoint(first)
    second_audit = validate_train_only_direct_checkpoint(second)
    assert first_audit["fixed_epoch"] == second_audit["fixed_epoch"]
    assert first_audit["canonical_state_dict_sha256"] == second_audit[
        "canonical_state_dict_sha256"
    ]
    assert first["canonical_frozen_model_sha256"] == second[
        "canonical_frozen_model_sha256"
    ]
    for name in first["state_dict"]:
        assert torch.equal(first["state_dict"][name], second["state_dict"][name])

    model = MonotoneBudgetCalibrator(**first["model_config"])
    model.load_state_dict(first["state_dict"], strict=True)
    model.eval()
    statistics = torch.asarray([[0.25, 0.5], [0.75, 0.125]])
    with torch.no_grad():
        prediction = model(statistics).grid_logits
    reloaded = torch.load(
        tmp_path / "first.pt", map_location="cpu", weights_only=True
    )
    restored = MonotoneBudgetCalibrator(**reloaded["model_config"])
    restored.load_state_dict(reloaded["state_dict"], strict=True)
    restored.eval()
    with torch.no_grad():
        restored_prediction = restored(statistics).grid_logits
    assert torch.equal(prediction, restored_prediction)
    assert canonical_state_dict_sha256(restored.state_dict()) == first[
        "canonical_state_dict_sha256"
    ]

    read_order_tamper = copy.deepcopy(first)
    events = read_order_tamper["train_only_selection_protocol"][
        "read_event_sequence"
    ]
    events[3], events[4] = events[4], events[3]
    with pytest.raises(ValueError, match="event order"):
        validate_train_only_direct_checkpoint(read_order_tamper)


def test_validation_label_changes_cannot_change_fixed_epoch_or_frozen_state(
    tmp_path: Path,
) -> None:
    base_root = tmp_path / "base"
    changed_root = tmp_path / "changed"
    base_root.mkdir()
    changed_root.mkdir()
    train, validation = _archives(base_root, validation_label_shift=0.0)
    changed_train, changed_validation = _archives(
        changed_root, validation_label_shift=0.85
    )
    # The two train byte snapshots are intentionally identical semantically;
    # file-container metadata is irrelevant because selection consumes arrays.
    base = _train(train, validation, base_root / "model.pt")
    changed = _train(
        changed_train, changed_validation, changed_root / "model.pt"
    )
    base_protocol = base["train_only_selection_protocol"]
    changed_protocol = changed["train_only_selection_protocol"]
    assert base_protocol["fixed_epoch"] == changed_protocol["fixed_epoch"]
    assert base["canonical_model_config_sha256"] == changed[
        "canonical_model_config_sha256"
    ]
    assert base["canonical_state_dict_sha256"] == changed[
        "canonical_state_dict_sha256"
    ]
    assert base["canonical_frozen_model_sha256"] == changed[
        "canonical_frozen_model_sha256"
    ]
    assert base["validation_archive_sha256"] != changed[
        "validation_archive_sha256"
    ]
    assert base["held_validation_labels_used_for_checkpoint_selection"] is False
    for name in base["state_dict"]:
        assert torch.equal(base["state_dict"][name], changed["state_dict"][name])


def test_validation_path_can_first_appear_only_after_model_freeze(
    tmp_path: Path,
) -> None:
    train, validation = _archives(tmp_path)
    validation_raw = validation.read_bytes()
    validation.unlink()
    observed: list[str] = []

    def observer(event: dict[str, object]) -> None:
        observed.append(str(event["event"]))
        if event["event"] == "all_train_model_frozen":
            assert not validation.exists()
            validation.write_bytes(validation_raw)

    checkpoint = _train(
        train, validation, tmp_path / "late-validation.pt", observer=observer
    )
    assert observed.index("all_train_model_frozen") < observed.index(
        "validation_bytes_captured"
    )
    protocol = checkpoint["train_only_selection_protocol"]
    assert protocol["validation_first_read_phase"] == (
        "post_freeze_episode_contract_binding"
    )
    assert protocol["held_validation_labels_used_for_checkpoint_selection"] is False
    assert validate_train_only_direct_checkpoint(checkpoint)["train_only"] is True


def test_archive_path_replacement_after_capture_fails_closed(
    tmp_path: Path,
) -> None:
    train, validation = _archives(tmp_path)
    captured_train = train.read_bytes()
    replaced = False

    def observer(event: dict[str, object]) -> None:
        nonlocal replaced
        if event["event"] == "all_train_model_frozen" and not replaced:
            replacement = tmp_path / "replacement.npz"
            replacement.write_bytes(captured_train)
            os.replace(replacement, train)
            replaced = True

    output = tmp_path / "must-not-exist.pt"
    with pytest.raises(RuntimeError, match="changed after first read"):
        _train(train, validation, output, observer=observer)
    assert replaced is True
    assert not output.exists()
