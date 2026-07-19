from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Sampler, Subset

from evaluation.artifact_integrity import file_sha256, ordered_ids_sha256
from rc_irstd.config import public_config, resolve_config_path
from rc_irstd.losses import calibrator_objective
from rc_irstd.meta import EpisodeDataset
from rc_irstd.models import MonotoneBudgetCalibrator
from rc_irstd.training.detector_trainer import resolve_device
from rc_irstd.utils.checkpoint import load_checkpoint, save_checkpoint
from rc_irstd.utils.io import atomic_write_json, ensure_dir
from rc_irstd.utils.logger import append_csv, build_logger
from rc_irstd.utils.seed import capture_rng_state, restore_rng_state, seed_everything


class BalancedEpisodeSampler(Sampler[int]):
    """Yield exactly the same number of episodes from every train domain."""

    def __init__(self, domain_ids: list[int], generator: torch.Generator) -> None:
        if not domain_ids:
            raise ValueError("BalancedEpisodeSampler requires at least one episode")
        self.generator = generator
        self.groups = [
            torch.tensor(
                [index for index, value in enumerate(domain_ids) if value == domain],
                dtype=torch.long,
            )
            for domain in sorted(set(domain_ids))
        ]
        if len(self.groups) < 2 or any(group.numel() == 0 for group in self.groups):
            raise ValueError("Calibrator training requires at least two non-empty domains")
        self.samples_per_domain = max(int(group.numel()) for group in self.groups)

    def __len__(self) -> int:
        return self.samples_per_domain * len(self.groups)

    def __iter__(self):
        sampled: list[torch.Tensor] = []
        for group in self.groups:
            chunks: list[torch.Tensor] = []
            remaining = self.samples_per_domain
            while remaining > 0:
                permutation = group[
                    torch.randperm(group.numel(), generator=self.generator)
                ]
                chunks.append(permutation[:remaining])
                remaining -= min(remaining, int(permutation.numel()))
            sampled.append(torch.cat(chunks))
        # Round-robin ordering keeps ordinary mini-batches balanced whenever
        # batch_size is a multiple of the number of training domains.
        return iter(torch.stack(sampled, dim=1).reshape(-1).tolist())


class CalibratorTrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.seed = int(config.get("seed", 42))
        seed_everything(self.seed, deterministic=bool(config.get("deterministic", True)))
        self.device = resolve_device(config.get("device"))
        self.output_dir = ensure_dir(
            resolve_config_path(config, config.get("output_dir", "outputs/calibrator"))
        )
        self.logger = build_logger("calibrator_train", self.output_dir)
        atomic_write_json(self.output_dir / "config.json", public_config(config))

        episodes_dir = resolve_config_path(config, config["episodes_dir"])
        self.dataset = EpisodeDataset(episodes_dir)
        self.diagnostic_only = bool(config.get("diagnostic_only", False))
        metadata = self.dataset.metadata
        support_size = int(metadata.get("support_size", 0))
        query_size = int(metadata.get("query_size", 0))
        stride = int(metadata.get("stride", 0))
        archive_formal = bool(metadata.get("formal_causal_contract", False))
        archive_diagnostic = bool(metadata.get("diagnostic_only", True))
        if not self.diagnostic_only:
            if int(metadata.get("format_version", 0)) < 2:
                raise ValueError("Formal calibrator training requires episode format_version >= 2")
            if not isinstance(metadata.get("archive_sha256"), str):
                raise ValueError("Formal calibrator training requires archive_sha256")
            if not archive_formal or archive_diagnostic:
                raise ValueError(
                    "Formal calibrator training requires a strict, non-diagnostic "
                    "episode archive"
                )
            if metadata.get("mode") != "causal" or stride < support_size + query_size:
                raise ValueError(
                    "Formal calibrator episodes must be causal with disjoint "
                    "cross-episode support/query roles"
                )
            score_stores = metadata.get("score_stores")
            if not isinstance(score_stores, list) or not score_stores or any(
                not isinstance(record, dict)
                or not bool(record.get("integrity_verified", False))
                for record in score_stores
            ):
                raise ValueError(
                    "Formal calibrator episodes require integrity-verified score stores"
                )
        self.formal_causal_contract = bool(
            archive_formal and not archive_diagnostic and not self.diagnostic_only
        )
        self.episode_archive_provenance = {
            "episodes_npz": str((episodes_dir / "episodes.npz").resolve()),
            "episodes_npz_sha256": file_sha256(episodes_dir / "episodes.npz"),
            "metadata_json": str((episodes_dir / "metadata.json").resolve()),
            "metadata_json_sha256": file_sha256(episodes_dir / "metadata.json"),
        }
        self.domain_names = list(self.dataset.metadata["domain_names"])
        self.fixed_last_selection = False
        train_indices, val_indices = self._split_indices(config.get("validation", {}))
        self.train_indices = [int(value) for value in train_indices]
        self.val_indices = [int(value) for value in val_indices]
        self.validation_contract = {
            "train_indices_sha256": ordered_ids_sha256(
                [str(value) for value in self.train_indices]
            ),
            "val_indices_sha256": ordered_ids_sha256(
                [str(value) for value in self.val_indices]
            ),
            "num_train": len(self.train_indices),
            "num_val": len(self.val_indices),
        }
        self.train_dataset = Subset(self.dataset, train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)

        model_config = config.get("model", {})
        self.model = MonotoneBudgetCalibrator(
            feature_dim=int(self.dataset.features.shape[1]),
            budget_grid=self.dataset.budgets.tolist(),
            hidden_dims=tuple(model_config.get("hidden_dims", (256, 128))),
            dropout=float(model_config.get("dropout", 0.1)),
            min_logit=float(model_config.get("min_logit", -10.0)),
            max_logit=float(model_config.get("max_logit", 18.0)),
        ).to(self.device)
        self.model.normalizer.fit(self.dataset.features[train_indices].to(self.device))

        training = config.get("training", {})
        self.epochs = int(training.get("epochs", 200))
        batch_size = int(training.get("batch_size", 64))
        workers = int(training.get("num_workers", 0))
        if self.epochs <= 0:
            raise ValueError("training.epochs must be positive")
        if batch_size <= 0:
            raise ValueError("training.batch_size must be positive")
        if workers < 0:
            raise ValueError("training.num_workers cannot be negative")
        self.train_generator = torch.Generator().manual_seed(self.seed)
        train_domain_ids = [
            int(self.dataset.domain_indices[index].item()) for index in train_indices
        ]
        self.train_sampler = BalancedEpisodeSampler(
            train_domain_ids,
            self.train_generator,
        )
        self.sampling_contract = {
            "strategy": "exact_balanced_domain_round_robin",
            "domain_ids": sorted(set(train_domain_ids)),
            "samples_per_domain": self.train_sampler.samples_per_domain,
            "samples_per_epoch": len(self.train_sampler),
        }
        if not self.diagnostic_only and batch_size % len(self.train_sampler.groups) != 0:
            raise ValueError(
                "Formal calibrator batch_size must be divisible by the number "
                "of training domains for balanced mini-batches"
            )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=self.train_sampler,
            num_workers=workers,
            generator=self.train_generator,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
        )
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(training.get("lr", 1e-3)),
            weight_decay=float(training.get("weight_decay", 1e-4)),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(self.epochs, 1),
            eta_min=float(training.get("min_lr", 1e-6)),
        )
        self.grad_clip = float(training.get("grad_clip", 5.0))
        self.patience = int(training.get("early_stopping_patience", 30))
        if self.fixed_last_selection and self.patience != 0:
            raise ValueError(
                "Fixed-last calibrator training requires early_stopping_patience=0"
            )
        self.loss_config = config.get("loss", {})
        self.resume_contract = {
            "schema_version": 1,
            "seed": self.seed,
            "deterministic": bool(config.get("deterministic", True)),
            "determinism_contract": {
                "cublas_workspace_config": ":4096:8",
                "torch_deterministic_algorithms": True,
                "warn_only_for_unsupported_kernels": True,
            },
            "diagnostic_only": self.diagnostic_only,
            "checkpoint_selection": (
                "fixed_last" if self.fixed_last_selection else "best_domain_validation"
            ),
            "model": self.model.export_config(),
            "loss": dict(self.loss_config),
            "training": {
                "epochs": self.epochs,
                "batch_size": batch_size,
                "num_workers": workers,
                "lr": float(training.get("lr", 1e-3)),
                "weight_decay": float(training.get("weight_decay", 1e-4)),
                "min_lr": float(training.get("min_lr", 1e-6)),
                "grad_clip": self.grad_clip,
                "early_stopping_patience": self.patience,
            },
        }
        self.best_loss = math.inf
        self.best_epoch = -1
        self.start_epoch = 0
        self.epochs_without_improvement = 0
        self.allow_legacy_resume = bool(training.get("allow_legacy_resume", False))
        resume = training.get("resume")
        if resume:
            self._resume(resolve_config_path(config, resume))

    def _resume(self, path: Path) -> None:
        checkpoint = load_checkpoint(
            path,
            self.device,
            allow_unsafe_legacy=self.allow_legacy_resume,
        )
        if checkpoint.get("kind") != "calibrator":
            raise ValueError(f"Not a calibrator checkpoint: {path}")
        checkpoint_budgets = torch.as_tensor(checkpoint.get("budgets", []), dtype=torch.float32)
        if checkpoint_budgets.shape != self.dataset.budgets.shape or not torch.allclose(
            checkpoint_budgets, self.dataset.budgets.cpu(), rtol=1e-6, atol=0.0
        ):
            raise ValueError("Resume checkpoint budget grid does not match the episode archive")
        if checkpoint.get("episode_archive_provenance") != self.episode_archive_provenance:
            raise ValueError("Resume checkpoint episode archive provenance has changed")
        if checkpoint.get("validation_contract") != self.validation_contract:
            raise ValueError("Resume checkpoint validation split contract has changed")
        saved_sampling_contract = checkpoint.get("sampling_contract")
        if saved_sampling_contract != self.sampling_contract:
            if not (self.allow_legacy_resume and saved_sampling_contract is None):
                raise ValueError(
                    "Resume checkpoint domain-balanced sampling contract has changed"
                )
            self.logger.warning("Legacy resume has no balanced sampling contract")
        saved_resume_contract = checkpoint.get("resume_contract")
        if saved_resume_contract != self.resume_contract:
            if not (self.allow_legacy_resume and saved_resume_contract is None):
                raise ValueError("Resume checkpoint training contract has changed")
            self.logger.warning("Legacy resume has no complete training contract")
        self.model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        self.best_loss = float(checkpoint.get("best_loss", math.inf))
        self.best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", -1)))
        self.epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        rng_state = checkpoint.get("rng_state")
        generator_state = checkpoint.get("train_generator_state")
        if rng_state is None or generator_state is None:
            if not self.allow_legacy_resume:
                raise ValueError(
                    "Resume checkpoint lacks RNG/DataLoader state; set "
                    "training.allow_legacy_resume=true only for diagnostics"
                )
            self.logger.warning("Resuming legacy calibrator without exact RNG restoration")
        else:
            if not isinstance(generator_state, torch.Tensor):
                raise TypeError("train_generator_state must be a torch tensor")
            self.train_generator.set_state(generator_state.detach().to(device="cpu"))
            restore_rng_state(rng_state)
        self.logger.info("Resumed calibrator from %s at epoch %d", path, self.start_epoch)

    def _split_indices(self, validation: dict[str, Any]) -> tuple[list[int], list[int]]:
        """Split meta episodes without silently leaking overlapping windows.

        The default strategy holds out an entire pseudo-target domain. Random
        episode splitting is available only when explicitly requested and is
        intended for smoke tests or diagnostics, not final cross-domain runs.
        """
        strategy = str(validation.get("strategy", "domain")).lower()
        val_domains = list(validation.get("domains", []))
        domain_indices = self.dataset.domain_indices.numpy()
        unique_domain_ids = sorted(int(value) for value in np.unique(domain_indices))

        if strategy in {"fixed_last", "none_fixed_last", "none"}:
            if val_domains:
                raise ValueError("Fixed-last validation cannot also specify validation domains")
            self.fixed_last_selection = True
            return list(range(len(self.dataset))), []
        if val_domains:
            unknown = sorted(set(val_domains) - set(self.domain_names))
            if unknown:
                raise ValueError(f"Unknown validation domains: {unknown}")
            selected_ids = {self.domain_names.index(name) for name in val_domains}
            val_mask = np.asarray([int(value) in selected_ids for value in domain_indices])
        elif strategy in {"domain", "domain_holdout", "grouped"}:
            if len(unique_domain_ids) < 2:
                raise ValueError(
                    "Domain-level validation requires episodes from at least two "
                    "pseudo-target domains. Set validation.strategy=random only for "
                    "a diagnostic single-domain run."
                )
            requested_name = validation.get("domain")
            if requested_name is not None:
                if requested_name not in self.domain_names:
                    raise ValueError(f"Unknown validation domain: {requested_name}")
                selected_id = self.domain_names.index(str(requested_name))
            else:
                # Deterministic default: reserve the final domain listed in metadata.
                selected_id = unique_domain_ids[-1]
            val_mask = domain_indices == selected_id
            self.logger.info(
                "Using domain-level calibrator validation: %s",
                self.domain_names[selected_id],
            )
        elif strategy == "random":
            if not self.diagnostic_only:
                raise ValueError(
                    "validation.strategy=random requires diagnostic_only=true; "
                    "formal calibration must hold out complete domains"
                )
            fraction = float(validation.get("fraction", 0.2))
            if not 0.0 < fraction < 1.0:
                raise ValueError("validation.fraction must be in (0,1)")
            rng = np.random.default_rng(self.seed)
            indices = rng.permutation(len(self.dataset))
            count = max(1, int(round(len(indices) * fraction)))
            val_indices = indices[:count].tolist()
            train_indices = indices[count:].tolist()
            if not train_indices or not val_indices:
                raise ValueError("Training and validation splits must both be non-empty")
            self.logger.warning(
                "Using random episode validation. Overlapping windows can make this "
                "optimistic; do not use it for final cross-domain experiments."
            )
            return train_indices, val_indices
        else:
            raise ValueError(
                "validation.strategy must be one of: domain, domain_holdout, grouped, random"
            )

        val_indices = np.nonzero(val_mask)[0].tolist()
        train_indices = np.nonzero(~val_mask)[0].tolist()
        if not train_indices or not val_indices:
            raise ValueError("Training and validation splits must both be non-empty")
        return train_indices, val_indices

    def _move(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    def _compute(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        output = self.model(batch["features"])
        loss = calibrator_objective(
            output.grid_logits,
            batch["threshold_logits"],
            self.dataset.budgets.to(self.device),
            batch["background_histogram"],
            batch["object_histogram"],
            self.dataset.bin_centers.to(self.device),
            total_pixels=batch["total_pixels"],
            **self.loss_config,
        )
        mae = (output.grid_thresholds - batch["thresholds"]).abs().mean()
        bsr = (
            loss.soft_fa
            <= self.dataset.budgets.to(self.device, dtype=loss.soft_fa.dtype)[None, :]
        ).float().mean()
        metrics = {
            "loss": float(loss.total.detach().cpu()),
            "oracle_loss": float(loss.oracle.detach().cpu()),
            "violation_loss": float(loss.violation.detach().cpu()),
            "utility_loss": float(loss.utility.detach().cpu()),
            "smoothness_loss": float(loss.smoothness.detach().cpu()),
            "threshold_mae": float(mae.detach().cpu()),
            "soft_bsr": float(bsr.detach().cpu()),
        }
        return loss.total, metrics

    def _epoch(self, loader: DataLoader[Any], training: bool) -> dict[str, float]:
        self.model.train(training)
        sums: dict[str, float] = defaultdict(float)
        steps = 0
        for raw_batch in loader:
            batch = self._move(raw_batch)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
                total, metrics = self._compute(batch)
                if not torch.isfinite(total):
                    raise FloatingPointError("Non-finite calibrator loss")
                total.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            else:
                with torch.no_grad():
                    total, metrics = self._compute(batch)
                if not torch.isfinite(total):
                    raise FloatingPointError("Non-finite calibrator validation loss")
            for key, value in metrics.items():
                sums[key] += value
            steps += 1
        return {key: value / max(steps, 1) for key, value in sums.items()}

    def _payload(self, epoch: int) -> dict[str, Any]:
        return {
            "format_version": 2,
            "kind": "calibrator",
            "method_name": "direct_threshold",
            "model_class": "MonotoneBudgetCalibrator",
            "role": "baseline",
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "checkpoint_selection": (
                "fixed_last" if self.fixed_last_selection else "best_domain_validation"
            ),
            "selection_rule": (
                "fixed_last" if self.fixed_last_selection else "best_domain_validation_loss"
            ),
            "diagnostic_only": self.diagnostic_only,
            "formal_causal_contract": self.formal_causal_contract,
            "formal_paper_checkpoint": self.formal_causal_contract,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "model_config": self.model.export_config(),
            "budgets": self.dataset.budgets.tolist(),
            "bin_centers": self.dataset.bin_centers.tolist(),
            "feature_spec": self.dataset.metadata["feature_spec"],
            "feature_names": self.dataset.metadata["feature_names"],
            "episode_metadata": {
                key: self.dataset.metadata[key]
                for key in (
                    "support_size",
                    "query_size",
                    "stride",
                    "mode",
                    "domain_names",
                    "formal_causal_contract",
                    "diagnostic_only",
                )
            },
            "episode_archive_provenance": self.episode_archive_provenance,
            "validation_contract": self.validation_contract,
            "sampling_contract": self.sampling_contract,
            "resume_contract": self.resume_contract,
            "determinism_contract": self.resume_contract["determinism_contract"],
            "epochs_without_improvement": self.epochs_without_improvement,
            "train_generator_state": self.train_generator.get_state().clone(),
            "rng_state": capture_rng_state(),
            "config": public_config(self.config),
        }

    def run(self) -> Path:
        history_path = self.output_dir / "history.csv"
        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self._epoch(self.train_loader, training=True)
            val_metrics = (
                {} if self.fixed_last_selection else self._epoch(self.val_loader, training=False)
            )
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            append_csv(
                history_path,
                {
                    "epoch": epoch,
                    "lr": learning_rate,
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{f"val_{key}": value for key, value in val_metrics.items()},
                },
            )
            self.logger.info(
                "epoch=%d/%d train=%.6f validation=%s",
                epoch + 1,
                self.epochs,
                train_metrics["loss"],
                (
                    "fixed_last"
                    if self.fixed_last_selection
                    else f"loss={val_metrics['loss']:.6f},bsr={val_metrics['soft_bsr']:.3f}"
                ),
            )
            improved = (
                False
                if self.fixed_last_selection
                else val_metrics["loss"] < self.best_loss
            )
            if improved:
                self.best_loss = val_metrics["loss"]
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            # Save the post-step scheduler state so continuation does not repeat
            # the learning-rate value from the completed epoch.
            self.scheduler.step()
            if improved:
                save_checkpoint(self.output_dir / "best.pt", self._payload(epoch))
            save_checkpoint(self.output_dir / "last.pt", self._payload(epoch))
            if (
                self.patience > 0
                and self.epochs_without_improvement >= self.patience
            ):
                self.logger.info("Early stopping after %d epochs without improvement", self.patience)
                break
        return self.output_dir / ("last.pt" if self.fixed_last_selection else "best.pt")
