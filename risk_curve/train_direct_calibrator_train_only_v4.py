"""Train RC-Direct with train-only epoch selection and post-freeze validation binding.

This module is intentionally separate from the historical validation-selected
trainer.  The source-train archive is captured once as an immutable byte
snapshot.  A deterministic five-fold split of that snapshot selects one fixed
epoch; each fold fits its normalizer on fold-train rows only.  The model is then
refit on all train rows and cryptographically frozen *before* the held
validation archive is read for the first time.  Validation is used only to bind
the existing v4 archive/episode contract and never enters a gradient, epoch, or
hyperparameter decision.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from rc_irstd.models.calibrator import (
    RC_DIRECT_ARCHITECTURE_VERSION,
    MonotoneBudgetCalibrator,
)

from .curve_dataset import load_curve_archive
from .direct_calibrator import (
    ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL,
    RC_DIRECT_BUDGET_SCHEMA_VERSION,
    RC_DIRECT_CHECKPOINT_SCHEMA_VERSION,
    _audit_source_only_archive,
    _load_provenance,
    derive_direct_threshold_targets,
    load_direct_training_pair,
    normalise_detector_checkpoint_sha256s,
    validate_detector_role_contract,
    validate_direct_checkpoint_contract,
    validate_joint_budget_pairs,
)
from .domain_statistics import statistics_names_sha256
from .representation import LOGIT_REPRESENTATION


TRAIN_ONLY_SELECTION_SCHEMA_VERSION = "rc-direct-v4-train-only-selection-v1"
TRAIN_ONLY_CV_FOLDS = 5
_EventObserver = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class _PathSnapshot:
    requested_path: Path
    resolved_path: Path
    raw: bytes
    sha256: str
    stat_signature: tuple[int, int, int, int, int, int]
    first_read_phase: str


def _stat_signature(stat: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_mode),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _capture_path_once(path: str | Path, *, phase: str) -> _PathSnapshot:
    """Read one stable path snapshot through one file descriptor exactly once."""

    requested = Path(path).expanduser().absolute()
    resolved = requested.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"Archive is not a regular file: {resolved}")
    with resolved.open("rb") as handle:
        before = os.fstat(handle.fileno())
        raw = handle.read()
        after = os.fstat(handle.fileno())
    if _stat_signature(before) != _stat_signature(after):
        raise RuntimeError(f"Archive changed while it was captured: {resolved}")
    if not raw:
        raise ValueError(f"Archive is empty: {resolved}")
    return _PathSnapshot(
        requested_path=requested,
        resolved_path=resolved,
        raw=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
        stat_signature=_stat_signature(after),
        first_read_phase=str(phase),
    )


def _assert_path_unchanged(snapshot: _PathSnapshot) -> None:
    """Fail closed if name resolution or inode metadata drifted after capture."""

    try:
        current_resolved = snapshot.requested_path.resolve(strict=True)
        current_stat = current_resolved.stat()
    except (FileNotFoundError, OSError) as error:
        raise RuntimeError(
            f"Captured archive path disappeared after first read: "
            f"{snapshot.requested_path}"
        ) from error
    if current_resolved != snapshot.resolved_path:
        raise RuntimeError(
            f"Captured archive path changed resolution: {snapshot.requested_path}"
        )
    if _stat_signature(current_stat) != snapshot.stat_signature:
        raise RuntimeError(
            f"Captured archive path changed after first read: {snapshot.resolved_path}"
        )


def _emit(
    events: list[dict[str, Any]],
    event: str,
    observer: _EventObserver | None,
    **fields: Any,
) -> None:
    record = {"ordinal": len(events) + 1, "event": str(event), **fields}
    events.append(record)
    if observer is not None:
        observer(copy.deepcopy(record))


def _scalar(value: Any, field: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{field} must be a scalar")
    return str(array.item())


def _validate_train_archive_before_optimisation(
    archive: dict[str, np.ndarray],
) -> tuple[tuple[str, ...], str, tuple[str, ...]]:
    """Apply every source-only invariant that does not require held validation."""

    if _scalar(archive["representation"], "representation") != LOGIT_REPRESENTATION:
        raise ValueError("Train-only RC-Direct requires raw_logit_float32")
    detector_protocol = _scalar(
        archive.get("threshold_grid_detector_protocol"),
        "threshold_grid_detector_protocol",
    )
    if detector_protocol != ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL:
        raise ValueError("Train-only RC-Direct requires all source detector folds")
    role_contract = validate_detector_role_contract(
        archive.get("threshold_grid_detector_checkpoint_sha256s"),
        archive.get("threshold_grid_outer_detector_checkpoint_sha256"),
        archive.get("threshold_grid_episode_detector_checkpoint_sha256s"),
    )
    detector_digests, outer_digest, episode_digests = role_contract
    provenance = _load_provenance(archive, "train")
    if provenance.get("threshold_grid_detector_protocol") != detector_protocol:
        raise ValueError("train detector-grid protocol provenance mismatch")
    if tuple(provenance.get("threshold_grid_detector_checkpoint_sha256s", [])) != (
        detector_digests
    ):
        raise ValueError("train detector-grid checkpoint provenance mismatch")
    if provenance.get("threshold_grid_outer_detector_checkpoint_sha256") != (
        outer_digest
    ):
        raise ValueError("train outer-detector provenance mismatch")
    if tuple(
        provenance.get("threshold_grid_episode_detector_checkpoint_sha256s", [])
    ) != episode_digests:
        raise ValueError("train episode-detector provenance mismatch")
    audits = provenance.get("fold_provenance_audits")
    if not isinstance(audits, list) or not audits:
        raise ValueError("train lacks detector-fold episode provenance")
    observed = {
        str(item.get("detector_weight_sha256"))
        for item in audits
        if isinstance(item, dict) and item.get("verified") is True
    }
    if observed != set(episode_digests):
        raise ValueError(
            "train episode artifacts do not use exactly the inner detectors"
        )
    _audit_source_only_archive(archive, split="train")
    return role_contract


def deterministic_five_fold_indices(
    num_rows: int, *, seed: int
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return five deterministic, exhaustive, pairwise-disjoint holdouts."""

    if isinstance(num_rows, bool) or int(num_rows) < TRAIN_ONLY_CV_FOLDS:
        raise ValueError(
            f"Train-only CV requires at least {TRAIN_ONLY_CV_FOLDS} rows"
        )
    generator = np.random.default_rng(int(seed))
    permutation = generator.permutation(int(num_rows))
    validation_folds = tuple(
        np.sort(values.astype(np.int64, copy=False))
        for values in np.array_split(permutation, TRAIN_ONLY_CV_FOLDS)
    )
    all_indices = np.arange(int(num_rows), dtype=np.int64)
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for validation_indices in validation_folds:
        train_indices = np.setdiff1d(
            all_indices, validation_indices, assume_unique=True
        )
        if np.intersect1d(train_indices, validation_indices).size:
            raise RuntimeError("Train-only CV fold train/validation overlap")
        result.append((train_indices, validation_indices))
    concatenated = np.concatenate(validation_folds)
    if not np.array_equal(np.sort(concatenated), all_indices):
        raise RuntimeError("Train-only CV holdouts do not cover every train row")
    if np.unique(concatenated).size != int(num_rows):
        raise RuntimeError("Train-only CV holdouts are not pairwise disjoint")
    return tuple(result)


def _direct_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    under_weight: float,
) -> torch.Tensor:
    per_item = nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    weights = torch.where(
        prediction < target,
        torch.full_like(prediction, float(under_weight)),
        torch.ones_like(prediction),
    )
    return (per_item * weights).mean()


def _make_model(
    *,
    feature_dim: int,
    thresholds: np.ndarray,
    pixel_budgets: np.ndarray,
    hidden_dims: tuple[int, ...],
    dropout: float,
) -> MonotoneBudgetCalibrator:
    return MonotoneBudgetCalibrator(
        feature_dim=int(feature_dim),
        budget_grid=pixel_budgets.tolist(),
        hidden_dims=hidden_dims,
        dropout=float(dropout),
        representation=LOGIT_REPRESENTATION,
        threshold_grid=thresholds.tolist(),
        architecture_version=RC_DIRECT_ARCHITECTURE_VERSION,
    )


def _train_for_epochs(
    *,
    statistics: torch.Tensor,
    targets: torch.Tensor,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray | None,
    thresholds: np.ndarray,
    pixel_budgets: np.ndarray,
    hidden_dims: tuple[int, ...],
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    under_weight: float,
    seed: int,
) -> tuple[MonotoneBudgetCalibrator, list[float], list[float]]:
    """Fit one model; the normalizer sees exactly ``train_indices``."""

    torch.manual_seed(int(seed))
    model = _make_model(
        feature_dim=int(statistics.shape[1]),
        thresholds=thresholds,
        pixel_budgets=pixel_budgets,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    train_index_tensor = torch.from_numpy(train_indices.astype(np.int64, copy=False))
    model.normalizer.fit(statistics[train_index_tensor])
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    permutation_generator = torch.Generator().manual_seed(int(seed) + 1_000_003)
    train_losses: list[float] = []
    evaluation_losses: list[float] = []
    for _epoch in range(int(epochs)):
        model.train()
        permutation = torch.randperm(
            int(train_index_tensor.numel()), generator=permutation_generator
        )
        loss_sum = 0.0
        element_count = 0
        for start in range(0, int(permutation.numel()), int(batch_size)):
            positions = permutation[start : start + int(batch_size)]
            indices = train_index_tensor[positions]
            prediction = model(statistics[indices]).grid_logits
            target = targets[indices]
            loss = _direct_loss(prediction, target, under_weight=under_weight)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Train-only RC-Direct loss is non-finite")
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            elements = int(target.numel())
            loss_sum += float(loss.detach()) * elements
            element_count += elements
        train_losses.append(loss_sum / max(element_count, 1))

        if evaluation_indices is not None:
            model.eval()
            indices = torch.from_numpy(
                evaluation_indices.astype(np.int64, copy=False)
            )
            with torch.no_grad():
                prediction = model(statistics[indices]).grid_logits
                loss = _direct_loss(
                    prediction, targets[indices], under_weight=under_weight
                )
            evaluation_losses.append(float(loss))
    return model, train_losses, evaluation_losses


def canonical_model_config_sha256(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash state semantically, independently of torch serialization metadata."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"state_dict[{name!r}] is not a tensor")
        value = tensor.detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        payload = value.numpy().tobytes(order="C")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _frozen_model_sha256(config_sha256: str, state_sha256: str) -> str:
    return hashlib.sha256(
        f"{config_sha256}:{state_sha256}".encode("ascii")
    ).hexdigest()


def _bound_pair_from_snapshots(
    train: _PathSnapshot,
    validation: _PathSnapshot,
) -> Any:
    """Run the existing complete pair contract on captured, not live, bytes."""

    with tempfile.TemporaryDirectory(prefix="rc-direct-train-only-contract-") as root:
        root_path = Path(root)
        train_path = root_path / "train.npz"
        validation_path = root_path / "validation.npz"
        train_path.write_bytes(train.raw)
        validation_path.write_bytes(validation.raw)
        pair = load_direct_training_pair(train_path, validation_path)
    episode_contract = copy.deepcopy(pair.episode_contract)
    for split, snapshot in (("train", train), ("validation", validation)):
        bound = episode_contract.get(split)
        if isinstance(bound, dict):
            bound["archive"] = str(snapshot.resolved_path)
            bound["archive_sha256"] = snapshot.sha256
    return pair, episode_contract


def validate_train_only_direct_checkpoint(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate both the existing RC-Direct contract and train-only evidence."""

    base = validate_direct_checkpoint_contract(checkpoint)
    protocol = checkpoint.get("train_only_selection_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("RC-Direct checkpoint lacks train_only_selection_protocol")
    if protocol.get("schema_version") != TRAIN_ONLY_SELECTION_SCHEMA_VERSION:
        raise ValueError("RC-Direct train-only selection schema mismatch")
    if protocol.get("cv_folds") != TRAIN_ONLY_CV_FOLDS:
        raise ValueError("RC-Direct train-only protocol must use deterministic 5-fold CV")
    if protocol.get("held_validation_labels_used_for_checkpoint_selection") is not False:
        raise ValueError("Held validation labels were not excluded from selection")
    if protocol.get("held_validation_labels_used_for_gradient") is not False:
        raise ValueError("Held validation labels were not excluded from gradients")
    if checkpoint.get("held_validation_labels_used_for_checkpoint_selection") is not False:
        raise ValueError("Checkpoint does not exclude held validation selection")
    if protocol.get("validation_first_read_phase") != (
        "post_freeze_episode_contract_binding"
    ):
        raise ValueError("Held validation was not first read post-freeze")
    events = protocol.get("read_event_sequence")
    expected_events = [
        "train_bytes_captured",
        "train_only_cross_validation_started",
        "fixed_epoch_selected_from_train_only_cv",
        "all_train_model_frozen",
        "validation_bytes_captured",
        "post_freeze_episode_contract_bound",
    ]
    if not isinstance(events, list) or [
        event.get("event") if isinstance(event, Mapping) else None for event in events
    ] != expected_events:
        raise ValueError("Train-only archive read/freeze event order is invalid")
    if [event.get("ordinal") for event in events] != list(
        range(1, len(expected_events) + 1)
    ):
        raise ValueError("Train-only archive read event ordinals are invalid")
    if events[0].get("phase") != protocol.get("train_archive_first_read_phase"):
        raise ValueError("Train archive first-read phase evidence is inconsistent")
    if events[4].get("phase") != protocol.get("validation_first_read_phase"):
        raise ValueError("Validation first-read phase evidence is inconsistent")
    for archive, event_index in (("train", 0), ("validation", 4)):
        captured = protocol.get(f"{archive}_archive_captured_sha256")
        if (
            not isinstance(captured, str)
            or len(captured) != 64
            or events[event_index].get("archive_sha256") != captured
            or checkpoint.get(f"{archive}_archive_sha256") != captured
        ):
            raise ValueError(f"Train-only {archive} archive hash evidence mismatch")
    fixed_epoch = protocol.get("fixed_epoch")
    if isinstance(fixed_epoch, bool) or not isinstance(fixed_epoch, int) or fixed_epoch < 1:
        raise ValueError("Train-only fixed_epoch must be a positive integer")
    folds = protocol.get("folds")
    if not isinstance(folds, list) or len(folds) != TRAIN_ONLY_CV_FOLDS:
        raise ValueError("Train-only checkpoint lacks five CV fold records")
    holdouts: list[int] = []
    parsed_folds: list[tuple[list[int], list[int]]] = []
    for expected_index, fold in enumerate(folds):
        if not isinstance(fold, Mapping) or fold.get("fold_index") != expected_index:
            raise ValueError("Train-only CV fold ordering is invalid")
        train_indices = fold.get("train_indices")
        validation_indices = fold.get("validation_indices")
        normalizer_indices = fold.get("normalizer_fit_indices")
        if not all(
            isinstance(value, list)
            for value in (train_indices, validation_indices, normalizer_indices)
        ):
            raise ValueError("Train-only CV indices are invalid")
        if train_indices != normalizer_indices:
            raise ValueError("Fold normalizer was not fitted only on fold-train")
        parsed_train = [int(value) for value in train_indices]
        parsed_validation = [int(value) for value in validation_indices]
        if len(set(parsed_train)) != len(parsed_train) or len(
            set(parsed_validation)
        ) != len(parsed_validation):
            raise ValueError("Train-only CV fold indices contain duplicates")
        if set(parsed_train).intersection(parsed_validation):
            raise ValueError("Train-only CV fold train/validation overlap")
        parsed_folds.append((parsed_train, parsed_validation))
        holdouts.extend(parsed_validation)
    if sorted(holdouts) != list(range(len(holdouts))) or len(set(holdouts)) != len(
        holdouts
    ):
        raise ValueError("Train-only CV holdouts are not an exhaustive partition")
    all_rows = set(range(len(holdouts)))
    for train_indices, validation_indices in parsed_folds:
        if set(train_indices) != all_rows.difference(validation_indices):
            raise ValueError("Train-only fold-train is not the holdout complement")
    if protocol.get("all_train_normalizer_fit_indices") != list(range(len(holdouts))):
        raise ValueError("All-train normalizer was not fitted on every train row")
    max_epoch = protocol.get("max_candidate_epoch")
    if isinstance(max_epoch, bool) or not isinstance(max_epoch, int) or max_epoch < 1:
        raise ValueError("Train-only max_candidate_epoch must be positive")
    if fixed_epoch > max_epoch:
        raise ValueError("Train-only fixed_epoch exceeds its candidate range")
    mean_losses = protocol.get("mean_train_internal_fold_holdout_loss_by_epoch")
    if not isinstance(mean_losses, list) or len(mean_losses) != max_epoch:
        raise ValueError("Train-only mean fold-holdout loss history is incomplete")
    for fold in folds:
        losses = fold.get("train_internal_fold_holdout_loss_by_epoch")
        if not isinstance(losses, list) or len(losses) != max_epoch:
            raise ValueError("Train-only fold-holdout loss history is incomplete")

    config = checkpoint.get("model_config")
    state = checkpoint.get("state_dict")
    if not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("Train-only checkpoint lacks frozen model payload")
    config_hash = canonical_model_config_sha256(config)
    state_hash = canonical_state_dict_sha256(state)
    frozen_hash = _frozen_model_sha256(config_hash, state_hash)
    expected = {
        "canonical_model_config_sha256": config_hash,
        "canonical_state_dict_sha256": state_hash,
        "canonical_frozen_model_sha256": frozen_hash,
    }
    for field, value in expected.items():
        if protocol.get(field) != value or checkpoint.get(field) != value:
            raise ValueError(f"Train-only checkpoint {field} mismatch")
    if int(checkpoint.get("epoch", -1)) + 1 != fixed_epoch:
        raise ValueError("Checkpoint epoch and train-only fixed_epoch disagree")
    return {**base, **expected, "fixed_epoch": fixed_epoch, "train_only": True}


def train_direct_calibrator_train_only(
    *,
    train_file: str | Path,
    validation_file: str | Path,
    output: str | Path,
    pixel_budgets: Sequence[float],
    component_budgets: Sequence[float],
    hidden_dims: Sequence[int] = (256, 128),
    dropout: float = 0.1,
    max_epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    under_weight: float = 4.0,
    seed: int = 42,
    device: str = "cpu",
    _event_observer: _EventObserver | None = None,
) -> Path:
    """Train and save the deterministic train-only fair RC-Direct baseline."""

    if device != "cpu":
        raise ValueError("The deterministic train-only protocol currently requires CPU")
    if int(max_epochs) < 1 or int(batch_size) < 1:
        raise ValueError("max_epochs and batch_size must be positive")
    if float(learning_rate) <= 0.0 or float(weight_decay) < 0.0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if not np.isfinite(under_weight) or float(under_weight) < 1.0:
        raise ValueError("under_weight must be finite and at least one")
    widths = tuple(int(value) for value in hidden_dims)
    if not widths or any(value < 1 for value in widths):
        raise ValueError("hidden_dims must contain positive widths")
    pixel_budget, component_budget = validate_joint_budget_pairs(
        pixel_budgets, component_budgets
    )
    output_path = Path(output).expanduser().absolute()
    if output_path == Path(train_file).expanduser().absolute() or output_path == Path(
        validation_file
    ).expanduser().absolute():
        raise ValueError("output must not overwrite an input archive")

    events: list[dict[str, Any]] = []
    train_snapshot = _capture_path_once(
        train_file, phase="train_snapshot_before_cross_validation"
    )
    _emit(
        events,
        "train_bytes_captured",
        _event_observer,
        phase=train_snapshot.first_read_phase,
        archive_sha256=train_snapshot.sha256,
    )
    train_archive = load_curve_archive(io.BytesIO(train_snapshot.raw))
    _validate_train_archive_before_optimisation(train_archive)
    thresholds = np.asarray(train_archive["thresholds"], dtype=np.float32)
    statistics_array = np.asarray(train_archive["statistics"], dtype=np.float32)
    targets_array = derive_direct_threshold_targets(
        train_archive["pixel_log_risk"],
        train_archive["component_log_risk_upper"],
        thresholds,
        pixel_budget,
        component_budget,
    ).logits
    if statistics_array.shape[0] < TRAIN_ONLY_CV_FOLDS:
        raise ValueError("Train archive has too few rows for deterministic 5-fold CV")
    statistics = torch.from_numpy(statistics_array.copy())
    targets = torch.from_numpy(np.asarray(targets_array, dtype=np.float32).copy())
    folds = deterministic_five_fold_indices(
        int(statistics.shape[0]), seed=int(seed)
    )

    prior_python_state = random.getstate()
    prior_numpy_state = np.random.get_state()
    prior_torch_state = torch.random.get_rng_state()
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        torch.use_deterministic_algorithms(True)
        _emit(events, "train_only_cross_validation_started", _event_observer)
        fold_records: list[dict[str, Any]] = []
        fold_losses: list[list[float]] = []
        for fold_index, (train_indices, validation_indices) in enumerate(folds):
            _model, _train_losses, validation_losses = _train_for_epochs(
                statistics=statistics,
                targets=targets,
                train_indices=train_indices,
                evaluation_indices=validation_indices,
                thresholds=thresholds,
                pixel_budgets=pixel_budget,
                hidden_dims=widths,
                dropout=float(dropout),
                epochs=int(max_epochs),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                weight_decay=float(weight_decay),
                under_weight=float(under_weight),
                seed=int(seed) + fold_index,
            )
            fold_losses.append(validation_losses)
            fold_records.append(
                {
                    "fold_index": fold_index,
                    "train_indices": train_indices.tolist(),
                    "validation_indices": validation_indices.tolist(),
                    "normalizer_fit_indices": train_indices.tolist(),
                    "train_internal_fold_holdout_loss_by_epoch": validation_losses,
                }
            )
        mean_loss = np.asarray(fold_losses, dtype=np.float64).mean(axis=0)
        fixed_epoch = int(np.argmin(mean_loss)) + 1
        _emit(
            events,
            "fixed_epoch_selected_from_train_only_cv",
            _event_observer,
            fixed_epoch=fixed_epoch,
        )

        all_indices = np.arange(int(statistics.shape[0]), dtype=np.int64)
        model, refit_train_losses, _unused = _train_for_epochs(
            statistics=statistics,
            targets=targets,
            train_indices=all_indices,
            evaluation_indices=None,
            thresholds=thresholds,
            pixel_budgets=pixel_budget,
            hidden_dims=widths,
            dropout=float(dropout),
            epochs=fixed_epoch,
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            under_weight=float(under_weight),
            seed=int(seed) + 100_000,
        )
        model.eval()
        frozen_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        model_config = model.export_config()
        config_sha256 = canonical_model_config_sha256(model_config)
        state_sha256 = canonical_state_dict_sha256(frozen_state)
        frozen_sha256 = _frozen_model_sha256(config_sha256, state_sha256)
        _emit(
            events,
            "all_train_model_frozen",
            _event_observer,
            canonical_frozen_model_sha256=frozen_sha256,
        )

        # The held archive is physically unopened until all train-only choices
        # and frozen model bytes above are final.
        validation_snapshot = _capture_path_once(
            validation_file, phase="post_freeze_episode_contract_binding"
        )
        _emit(
            events,
            "validation_bytes_captured",
            _event_observer,
            phase=validation_snapshot.first_read_phase,
            archive_sha256=validation_snapshot.sha256,
        )
        pair, episode_contract = _bound_pair_from_snapshots(
            train_snapshot, validation_snapshot
        )
        _emit(events, "post_freeze_episode_contract_bound", _event_observer)

        manifest_hash = _scalar(
            train_archive["threshold_grid_manifest_sha256"],
            "threshold_grid_manifest_sha256",
        )
        feature_hash = _scalar(
            train_archive["feature_schema_sha256"], "feature_schema_sha256"
        )
        grid_hash = _scalar(
            train_archive["threshold_grid_sha256"], "threshold_grid_sha256"
        )
        detector_checkpoint_sha256s = list(
            normalise_detector_checkpoint_sha256s(
                train_archive["threshold_grid_detector_checkpoint_sha256s"]
            )
        )
        protocol = {
            "schema_version": TRAIN_ONLY_SELECTION_SCHEMA_VERSION,
            "selection_source": "source_train_rows_only",
            "cv_folds": TRAIN_ONLY_CV_FOLDS,
            "cv_split_rule": "seeded_permutation_then_numpy_array_split",
            "cv_seed": int(seed),
            "max_candidate_epoch": int(max_epochs),
            "mean_train_internal_fold_holdout_loss_by_epoch": mean_loss.tolist(),
            "fixed_epoch": fixed_epoch,
            "folds": fold_records,
            "all_train_refit_epochs": fixed_epoch,
            "all_train_refit_loss_by_epoch": refit_train_losses,
            "all_train_normalizer_fit_indices": all_indices.tolist(),
            "held_validation_labels_used_for_checkpoint_selection": False,
            "held_validation_labels_used_for_gradient": False,
            "validation_first_read_phase": (
                "post_freeze_episode_contract_binding"
            ),
            "train_archive_first_read_phase": train_snapshot.first_read_phase,
            "train_archive_captured_sha256": train_snapshot.sha256,
            "validation_archive_captured_sha256": validation_snapshot.sha256,
            "canonical_model_config_sha256": config_sha256,
            "canonical_state_dict_sha256": state_sha256,
            "canonical_frozen_model_sha256": frozen_sha256,
            "read_event_sequence": copy.deepcopy(events),
        }
        payload: dict[str, Any] = {
            "checkpoint_schema_version": RC_DIRECT_CHECKPOINT_SCHEMA_VERSION,
            "format_version": 4,
            "kind": "calibrator",
            "method_name": "direct_threshold",
            "model_class": "MonotoneBudgetCalibrator",
            "role": "baseline",
            "representation": LOGIT_REPRESENTATION,
            "thresholds": torch.from_numpy(thresholds.copy()),
            "threshold_grid_schema_version": _scalar(
                train_archive["threshold_grid_schema_version"],
                "threshold_grid_schema_version",
            ),
            "threshold_grid_sha256": grid_hash,
            "threshold_grid_manifest_sha256": manifest_hash,
            "threshold_grid_detector_protocol": _scalar(
                train_archive["threshold_grid_detector_protocol"],
                "threshold_grid_detector_protocol",
            ),
            "threshold_grid_detector_checkpoint_sha256s": (
                detector_checkpoint_sha256s
            ),
            "threshold_grid_outer_detector_checkpoint_sha256": (
                pair.outer_detector_checkpoint_sha256
            ),
            "threshold_grid_episode_detector_checkpoint_sha256s": list(
                pair.episode_detector_checkpoint_sha256s
            ),
            "statistics_schema_version": _scalar(
                train_archive["statistics_schema_version"],
                "statistics_schema_version",
            ),
            "statistics_names": list(pair.statistics_names),
            "statistics_names_sha256": statistics_names_sha256(
                pair.statistics_names
            ),
            "feature_schema_sha256": feature_hash,
            "statistics_mean": torch.from_numpy(pair.statistics_mean.copy()),
            "statistics_std": torch.from_numpy(pair.statistics_std.copy()),
            "budget_schema_version": RC_DIRECT_BUDGET_SCHEMA_VERSION,
            "pixel_budgets": pixel_budget.tolist(),
            "component_budgets": component_budget.tolist(),
            "model_architecture_version": RC_DIRECT_ARCHITECTURE_VERSION,
            "model_config": model_config,
            "state_dict": frozen_state,
            "episode_contract": episode_contract,
            "target_label_policy": {
                "model_inputs": "adaptation_window_A_label_free_statistics_only",
                "supervision": "source_official_train_future_E_risk_only",
                "outer_target_labels_used_for_features": False,
                "outer_target_labels_used_for_checkpoint_selection": False,
            },
            "checkpoint_selection": (
                "source_train_only_deterministic_5fold_cv_fixed_epoch"
            ),
            "held_validation_labels_used_for_checkpoint_selection": False,
            "train_only_selection_protocol": protocol,
            "canonical_model_config_sha256": config_sha256,
            "canonical_state_dict_sha256": state_sha256,
            "canonical_frozen_model_sha256": frozen_sha256,
            "train_archive": str(train_snapshot.resolved_path),
            "train_archive_sha256": train_snapshot.sha256,
            "validation_archive": str(validation_snapshot.resolved_path),
            "validation_archive_sha256": validation_snapshot.sha256,
            "epoch": fixed_epoch - 1,
            "seed": int(seed),
        }
        validate_train_only_direct_checkpoint(payload)
        # Re-check frozen bytes and both live path bindings immediately before
        # publication.  No archive is re-read; inode metadata makes drift fail
        # closed without opening train/validation a second time.
        if canonical_model_config_sha256(payload["model_config"]) != config_sha256:
            raise RuntimeError("Frozen RC-Direct config changed before publication")
        if canonical_state_dict_sha256(payload["state_dict"]) != state_sha256:
            raise RuntimeError("Frozen RC-Direct state changed before publication")
        _assert_path_unchanged(train_snapshot)
        _assert_path_unchanged(validation_snapshot)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        temporary = output_path.with_name(output_path.name + ".tmp")
        try:
            temporary.write_bytes(buffer.getvalue())
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        torch.use_deterministic_algorithms(prior_deterministic)
        random.setstate(prior_python_state)
        np.random.set_state(prior_numpy_state)
        torch.random.set_rng_state(prior_torch_state)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pixel-budgets", nargs="+", type=float, required=True)
    parser.add_argument("--component-budgets", nargs="+", type=float, required=True)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 128])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--under-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_direct_calibrator_train_only(
        train_file=args.train_file,
        validation_file=args.val_file,
        output=args.output,
        pixel_budgets=args.pixel_budgets,
        component_budgets=args.component_budgets,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        under_weight=args.under_weight,
        seed=args.seed,
        device=args.device,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TRAIN_ONLY_CV_FOLDS",
    "TRAIN_ONLY_SELECTION_SCHEMA_VERSION",
    "canonical_model_config_sha256",
    "canonical_state_dict_sha256",
    "deterministic_five_fold_indices",
    "train_direct_calibrator_train_only",
    "validate_train_only_direct_checkpoint",
]
