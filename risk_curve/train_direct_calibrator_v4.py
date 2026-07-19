"""Train the fair raw-logit RC-Direct baseline on v4 curve episodes."""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rc_irstd.models.calibrator import (
    RC_DIRECT_ARCHITECTURE_VERSION,
    MonotoneBudgetCalibrator,
)

from .direct_calibrator import (
    ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL,
    RC_DIRECT_BUDGET_SCHEMA_VERSION,
    RC_DIRECT_CHECKPOINT_SCHEMA_VERSION,
    derive_direct_threshold_targets,
    load_direct_training_pair,
    normalise_detector_checkpoint_sha256s,
    quantize_direct_logit_threshold,
    validate_direct_checkpoint_contract,
    validate_joint_budget_pairs,
)
from .domain_statistics import statistics_names_sha256
from .representation import LOGIT_REPRESENTATION


class _DirectDataset(Dataset[dict[str, torch.Tensor]]):
    """Expose only label-free A statistics and derived direct supervision."""

    def __init__(self, statistics: np.ndarray, target_logits: np.ndarray) -> None:
        values = np.asarray(statistics, dtype=np.float32)
        targets = np.asarray(target_logits, dtype=np.float32)
        if values.ndim != 2 or targets.ndim != 2 or values.shape[0] != targets.shape[0]:
            raise ValueError("RC-Direct dataset arrays have incompatible shapes")
        if not np.isfinite(values).all() or not np.isfinite(targets).all():
            raise ValueError("RC-Direct dataset arrays must be finite")
        self.statistics = torch.from_numpy(values)
        self.target_logits = torch.from_numpy(targets)

    def __len__(self) -> int:
        return int(self.statistics.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "statistics": self.statistics[index],
            "target_logits": self.target_logits[index],
        }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _evaluate(
    model: MonotoneBudgetCalibrator,
    loader: DataLoader,
    *,
    thresholds: np.ndarray,
    device: torch.device,
    under_weight: float,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    absolute_sum = 0.0
    count = 0
    exact = 0
    rejects = 0
    with torch.no_grad():
        for batch in loader:
            statistics = batch["statistics"].to(device)
            target = batch["target_logits"].to(device)
            prediction = model(statistics).grid_logits
            loss = _direct_loss(prediction, target, under_weight=under_weight)
            elements = int(target.numel())
            loss_sum += float(loss.cpu()) * elements
            absolute_sum += float((prediction - target).abs().sum().cpu())
            count += elements
            predicted_values = prediction.cpu().numpy().reshape(-1)
            target_values = target.cpu().numpy().reshape(-1)
            for predicted, expected in zip(predicted_values, target_values):
                predicted_action = quantize_direct_logit_threshold(predicted, thresholds)
                expected_action = quantize_direct_logit_threshold(expected, thresholds)
                exact += int(
                    predicted_action.threshold_index == expected_action.threshold_index
                )
                rejects += int(predicted_action.reject)
    if count == 0:
        raise ValueError("RC-Direct validation loader is empty")
    return {
        "loss": loss_sum / count,
        "threshold_logit_mae": absolute_sum / count,
        "grid_action_accuracy": exact / count,
        "reject_rate": rejects / count,
    }


def train_direct_calibrator(
    *,
    train_file: str | Path,
    validation_file: str | Path,
    output: str | Path,
    pixel_budgets: Sequence[float],
    component_budgets: Sequence[float],
    hidden_dims: Sequence[int] = (256, 128),
    dropout: float = 0.1,
    epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 30,
    under_weight: float = 4.0,
    num_workers: int = 0,
    seed: int = 42,
    device: str = "auto",
) -> Path:
    """Train RC-Direct without creating an alternative data/feature archive."""

    pixel_budget, component_budget = validate_joint_budget_pairs(
        pixel_budgets, component_budgets
    )
    if epochs <= 0 or batch_size <= 0 or patience <= 0:
        raise ValueError("epochs, batch_size, and patience must be positive")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if under_weight < 1.0 or not np.isfinite(under_weight):
        raise ValueError("under_weight must be finite and at least one")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    _seed_everything(int(seed))
    device_name = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if device_name == "auto":
        device_name = "cpu"
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch_device = torch.device(device_name)

    pair = load_direct_training_pair(train_file, validation_file)
    train_archive = pair.train_archive
    validation_archive = pair.validation_archive
    thresholds = np.asarray(train_archive["thresholds"], dtype=np.float32)
    train_targets = derive_direct_threshold_targets(
        train_archive["pixel_log_risk"],
        train_archive["component_log_risk_upper"],
        thresholds,
        pixel_budget,
        component_budget,
    )
    validation_targets = derive_direct_threshold_targets(
        validation_archive["pixel_log_risk"],
        validation_archive["component_log_risk_upper"],
        thresholds,
        pixel_budget,
        component_budget,
    )
    train_set = _DirectDataset(train_archive["statistics"], train_targets.logits)
    validation_set = _DirectDataset(
        validation_archive["statistics"], validation_targets.logits
    )
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(
        train_set,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
    )

    model = MonotoneBudgetCalibrator(
        feature_dim=int(np.asarray(train_archive["statistics"]).shape[1]),
        budget_grid=pixel_budget.tolist(),
        hidden_dims=tuple(int(value) for value in hidden_dims),
        dropout=float(dropout),
        representation=LOGIT_REPRESENTATION,
        threshold_grid=thresholds.tolist(),
        architecture_version=RC_DIRECT_ARCHITECTURE_VERSION,
    ).to(torch_device)
    model.normalizer.fit(
        torch.from_numpy(np.asarray(train_archive["statistics"], dtype=np.float32)).to(
            torch_device
        )
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    def checkpoint_payload(epoch: int) -> dict[str, object]:
        manifest_hash = _scalar(train_archive["threshold_grid_manifest_sha256"])
        feature_hash = _scalar(train_archive["feature_schema_sha256"])
        grid_hash = _scalar(train_archive["threshold_grid_sha256"])
        statistics_schema = _scalar(train_archive["statistics_schema_version"])
        detector_protocol = _scalar(
            train_archive["threshold_grid_detector_protocol"]
        )
        if detector_protocol != ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL:
            raise ValueError("Unexpected detector-grid protocol after archive validation")
        detector_checkpoint_sha256s = list(
            normalise_detector_checkpoint_sha256s(
                train_archive["threshold_grid_detector_checkpoint_sha256s"]
            )
        )
        payload: dict[str, object] = {
            "checkpoint_schema_version": RC_DIRECT_CHECKPOINT_SCHEMA_VERSION,
            "format_version": 4,
            "kind": "calibrator",
            "method_name": "direct_threshold",
            "model_class": "MonotoneBudgetCalibrator",
            "role": "baseline",
            "representation": LOGIT_REPRESENTATION,
            "thresholds": torch.from_numpy(thresholds.copy()),
            "threshold_grid_schema_version": _scalar(
                train_archive["threshold_grid_schema_version"]
            ),
            "threshold_grid_sha256": grid_hash,
            "threshold_grid_manifest_sha256": manifest_hash,
            "threshold_grid_detector_protocol": detector_protocol,
            "threshold_grid_detector_checkpoint_sha256s": (
                detector_checkpoint_sha256s
            ),
            "threshold_grid_outer_detector_checkpoint_sha256": (
                pair.outer_detector_checkpoint_sha256
            ),
            "threshold_grid_episode_detector_checkpoint_sha256s": list(
                pair.episode_detector_checkpoint_sha256s
            ),
            "statistics_schema_version": statistics_schema,
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
            "model_config": model.export_config(),
            "state_dict": model.state_dict(),
            "episode_contract": pair.episode_contract,
            "target_label_policy": {
                "model_inputs": "adaptation_window_A_label_free_statistics_only",
                "supervision": "source_official_train_future_E_risk_only",
                "outer_target_labels_used_for_features": False,
                "outer_target_labels_used_for_checkpoint_selection": False,
            },
            "checkpoint_selection": "source_only_pseudo_target_validation_loss",
            "train_archive": str(Path(train_file).expanduser().resolve()),
            "train_archive_sha256": hashlib.sha256(Path(train_file).read_bytes()).hexdigest(),
            "validation_archive": str(Path(validation_file).expanduser().resolve()),
            "validation_archive_sha256": hashlib.sha256(
                Path(validation_file).read_bytes()
            ).hexdigest(),
            "epoch": int(epoch),
            "best_validation_loss": float(best_loss),
            "history": history,
            "seed": int(seed),
        }
        validate_direct_checkpoint_contract(payload)
        return payload

    for epoch in range(int(epochs)):
        model.train()
        train_loss_sum = 0.0
        train_elements = 0
        for batch in train_loader:
            statistics = batch["statistics"].to(torch_device)
            targets = batch["target_logits"].to(torch_device)
            predictions = model(statistics).grid_logits
            loss = _direct_loss(predictions, targets, under_weight=under_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("RC-Direct training loss is non-finite")
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            elements = int(targets.numel())
            train_loss_sum += float(loss.detach().cpu()) * elements
            train_elements += elements
        validation_metrics = _evaluate(
            model,
            validation_loader,
            thresholds=thresholds,
            device=torch_device,
            under_weight=under_weight,
        )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_elements, 1),
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(record)
        improved = validation_metrics["loss"] < best_loss
        if improved:
            best_loss = validation_metrics["loss"]
            stale_epochs = 0
            torch.save(checkpoint_payload(epoch), output_path)
        else:
            stale_epochs += 1
        if stale_epochs >= int(patience):
            break
    if not output_path.is_file():
        raise RuntimeError("RC-Direct training did not produce a checkpoint")
    return output_path


def _scalar(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError("Expected a scalar archive contract field")
    return str(array.item())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pixel-budgets", nargs="+", type=float, required=True)
    parser.add_argument("--component-budgets", nargs="+", type=float, required=True)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 128])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--under-weight", type=float, default=4.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_direct_calibrator(
        train_file=args.train_file,
        validation_file=args.val_file,
        output=args.output,
        pixel_budgets=args.pixel_budgets,
        component_budgets=args.component_budgets,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        under_weight=args.under_weight,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
