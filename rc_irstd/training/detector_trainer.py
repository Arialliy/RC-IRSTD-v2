from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from evaluation.artifact_integrity import file_sha256, ordered_ids_sha256
from rc_irstd.config import public_config, resolve_config_path
from rc_irstd.data import (
    BalancedDomainBatcher,
    IRSTDDataset,
    collate_fixed,
    ensure_unique_sample_ids,
    read_split_file,
    resolve_split_file,
)
from rc_irstd.losses import DetectorObjective
from rc_irstd.models import build_mshnet, forward_mshnet
from rc_irstd.utils.checkpoint import load_checkpoint, save_checkpoint
from rc_irstd.utils.io import atomic_write_json, ensure_dir
from rc_irstd.utils.logger import append_csv, build_logger
from rc_irstd.utils.seed import capture_rng_state, restore_rng_state, seed_everything


def resolve_device(value: str | None) -> torch.device:
    requested = (value or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _build_optimizer(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    # RC-MSHNET-PATCH: differential learning rates
    name = str(config.get("name", "adamw")).lower()
    learning_rate = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))
    backbone_lr_scale = float(config.get("backbone_lr_scale", 1.0))
    if learning_rate <= 0.0:
        raise ValueError("optimizer.lr must be positive")
    if not 0.0 < backbone_lr_scale <= 1.0:
        raise ValueError("optimizer.backbone_lr_scale must be in (0, 1]")

    parameters: object = model.parameters()
    extension_prefixes = tuple(getattr(model, "extension_prefixes", ()))
    if backbone_lr_scale != 1.0:
        if not extension_prefixes:
            raise ValueError(
                "optimizer.backbone_lr_scale is supported only by a model that "
                "declares extension_prefixes"
            )
        extension_parameters: list[torch.nn.Parameter] = []
        backbone_parameters: list[torch.nn.Parameter] = []
        for parameter_name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            destination = (
                extension_parameters
                if any(parameter_name.startswith(prefix) for prefix in extension_prefixes)
                else backbone_parameters
            )
            destination.append(parameter)
        if not extension_parameters or not backbone_parameters:
            raise ValueError("Could not partition RC-MSHNet backbone and extensions")
        # The existing logger reports group 0 as the headline LR.
        parameters = [
            {
                "params": extension_parameters,
                "lr": learning_rate,
                "group_name": "rc_mshnet_extensions",
            },
            {
                "params": backbone_parameters,
                "lr": learning_rate * backbone_lr_scale,
                "group_name": "mshnet_backbone",
            },
        ]

    if name == "adamw":
        return AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(
            parameters,
            lr=learning_rate,
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(config.get("nesterov", True)),
        )
    raise ValueError(f"Unsupported optimizer: {name}")

def _average_metrics(sums: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in sums.items()}


class DetectorTrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.seed = int(config.get("seed", 42))
        seed_everything(self.seed, deterministic=bool(config.get("deterministic", True)))
        self.device = resolve_device(config.get("device"))
        self.output_dir = ensure_dir(
            resolve_config_path(config, config.get("output_dir", "outputs/detector"))
        )
        self.logger = build_logger("detector_train", self.output_dir)
        atomic_write_json(self.output_dir / "config.json", public_config(config))

        data_config = config.get("data", {})
        sources = data_config.get("sources", [])
        if len(sources) < 1:
            raise ValueError("data.sources must list at least one source domain")
        image_size = data_config.get("image_size", 256)
        image_hw = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        if len(image_hw) != 2 or min(int(value) for value in image_hw) < 16:
            raise ValueError("data.image_size must contain two dimensions of at least 16 pixels")
        if any(int(value) % 16 != 0 for value in image_hw):
            raise ValueError(
                "data.image_size dimensions must be divisible by 16 for MSHNet's "
                "four encoder/decoder scales"
            )
        batch_per_domain = int(data_config.get("batch_per_domain", 2))
        num_workers = int(data_config.get("num_workers", 4))
        if batch_per_domain <= 0:
            raise ValueError("data.batch_per_domain must be positive")
        if num_workers < 0:
            raise ValueError("data.num_workers cannot be negative")
        train_split = str(data_config.get("train_split", "train"))
        if train_split.lower() != "train":
            raise ValueError(
                "Formal detector optimization is restricted to train_split=train"
            )
        val_split = data_config.get("val_split")
        self.diagnostic_test_eval = bool(data_config.get("diagnostic_test_eval", False))
        configured_diagnostic_only = bool(config.get("diagnostic_only", False))
        if self.diagnostic_test_eval and not configured_diagnostic_only:
            raise ValueError(
                "data.diagnostic_test_eval=true requires diagnostic_only=true; "
                "formal training must not inspect test labels"
            )
        if val_split is not None and not self.diagnostic_test_eval:
            raise ValueError(
                "data.val_split is a test alias and may be set only together with "
                "diagnostic_test_eval=true"
            )
        if self.diagnostic_test_eval and val_split is None:
            val_split = "test"
        augment = bool(data_config.get("augment", True))
        self.source_names: list[str] = []
        self.source_split_records: list[dict[str, Any]] = []
        source_paths: list[Path] = []
        train_loaders: list[DataLoader[Any]] = []
        self.val_loaders: list[DataLoader[Any]] = []
        for domain_id, source in enumerate(sources):
            if not isinstance(source, dict) or "path" not in source:
                raise ValueError("Each source must be a mapping with path and optional name")
            path = resolve_config_path(config, source["path"])
            name = str(source.get("name", path.name))
            if name in self.source_names:
                raise ValueError(f"Duplicate source domain name: {name}")
            if path in source_paths:
                raise ValueError(f"The same source path was listed more than once: {path}")
            self.source_names.append(name)
            source_paths.append(path)
            train_dataset = IRSTDDataset(
                path,
                train_split,
                domain_id=domain_id,
                dataset_name=name,
                training=True,
                image_size=image_size,
                augment=augment,
                split_file=source.get("train_split_file"),
            )
            if not train_dataset.split_authority_verified and not configured_diagnostic_only:
                raise ValueError(
                    f"Source '{name}' uses an explicit train manifest that does not "
                    "match the unambiguous frozen split authority"
                )
            test_split_file = resolve_split_file(
                path,
                source.get("test_split_file"),
                split="test",
            )
            if source.get("test_split_file") is not None and not configured_diagnostic_only:
                try:
                    automatic_test_split = resolve_split_file(path, None, split="test")
                except (FileNotFoundError, ValueError) as error:
                    raise ValueError(
                        f"Source '{name}' has no unambiguous frozen test authority"
                    ) from error
                if automatic_test_split != test_split_file:
                    raise ValueError(
                        f"Source '{name}' explicit test manifest does not match the "
                        "frozen split authority"
                    )
            test_ids = ensure_unique_sample_ids(read_split_file(test_split_file))
            overlap = sorted(set(train_dataset.image_ids).intersection(test_ids))
            if overlap:
                raise ValueError(
                    f"Source '{name}' frozen train/test manifests overlap: "
                    + ", ".join(overlap[:5])
                )
            self.source_split_records.append(
                {
                    "name": name,
                    "path": str(path),
                    "train_split_file": str(train_dataset.split_file),
                    "train_split_file_sha256": file_sha256(train_dataset.split_file),
                    "train_ordered_ids_sha256": ordered_ids_sha256(
                        train_dataset.image_ids
                    ),
                    "num_train_samples": len(train_dataset),
                    "test_split_file": str(test_split_file),
                    "test_split_file_sha256": file_sha256(test_split_file),
                    "test_ordered_ids_sha256": ordered_ids_sha256(test_ids),
                    "num_test_samples": len(test_ids),
                    "train_test_id_overlap": False,
                }
            )
            if len(train_dataset) < batch_per_domain:
                raise ValueError(
                    f"Source '{name}' has {len(train_dataset)} training samples, fewer than "
                    f"batch_per_domain={batch_per_domain}. Reduce the per-domain batch size."
                )
            generator = torch.Generator().manual_seed(self.seed + domain_id)
            train_loaders.append(
                DataLoader(
                    train_dataset,
                    batch_size=batch_per_domain,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=self.device.type == "cuda",
                    drop_last=True,
                    collate_fn=collate_fixed,
                    generator=generator,
                    # Worker RNG state cannot be serialized reliably; recreating
                    # workers preserves epoch-boundary resume reproducibility.
                    persistent_workers=False,
                )
            )
            if self.diagnostic_test_eval and val_split:
                val_dataset = IRSTDDataset(
                    path,
                    str(val_split),
                    domain_id=domain_id,
                    dataset_name=name,
                    training=False,
                    image_size=image_size,
                    preserve_aspect_eval=False,
                    augment=False,
                    split_file=source.get("diagnostic_split_file"),
                )
                self.val_loaders.append(
                    DataLoader(
                        val_dataset,
                        batch_size=batch_per_domain,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=self.device.type == "cuda",
                        collate_fn=collate_fixed,
                        persistent_workers=False,
                    )
                )
        self.train_batches = BalancedDomainBatcher(train_loaders)

        requested_model = dict(config.get("model", {}))
        self.model = build_mshnet(requested_model).to(self.device)
        if not hasattr(self.model, "export_config"):
            raise TypeError("Configured detector backend does not export its architecture")
        self.model_config = dict(self.model.export_config())
        self.diagnostic_only = configured_diagnostic_only
        if (
            str(self.model_config.get("backend", "canonical")) not in {"canonical", "rc_mshnet"}
            and not self.diagnostic_only
        ):
            raise ValueError(
                "Unregistered detector backends require diagnostic_only=true; "
                "formal runs permit canonical or rc_mshnet"
            )
        self.objective = DetectorObjective(**config.get("loss", {})).to(self.device)
        self.optimizer = _build_optimizer(self.model, config.get("optimizer", {}))
        training_config = config.get("training", {})
        # RC-MSHNET-PATCH: pretrained initialization
        self.initialization_report: dict[str, Any] | None = None
        initialize_from = training_config.get("initialize_from")
        if initialize_from and training_config.get("resume"):
            raise ValueError(
                "training.initialize_from and training.resume are mutually "
                "exclusive; resume already contains initialized weights"
            )
        if initialize_from:
            from rc_irstd.models.rc_mshnet import (
                initialize_rc_mshnet_from_checkpoint,
            )

            initialization_path = resolve_config_path(config, initialize_from)
            self.initialization_report = (
                initialize_rc_mshnet_from_checkpoint(
                    self.model, initialization_path, device=self.device
                )
            )
            atomic_write_json(
                self.output_dir / "initialization_report.json",
                self.initialization_report,
            )
            self.logger.info(
                "Initialized RC-MSHNet from canonical MSHNet %s",
                initialization_path,
            )
        self.epochs = int(training_config.get("epochs", 100))
        self.warmup_epochs = int(training_config.get("warmup_epochs", 5))
        self.grad_clip = float(training_config.get("grad_clip", 5.0))
        self.validation_interval = int(training_config.get("validation_interval", 1))
        if self.epochs <= 0:
            raise ValueError("training.epochs must be positive")
        if self.warmup_epochs < 0:
            raise ValueError("training.warmup_epochs cannot be negative")
        if self.validation_interval < 0:
            raise ValueError("training.validation_interval cannot be negative")
        if not self.diagnostic_only and self.epochs <= self.warmup_epochs:
            raise ValueError(
                "Formal training must include at least one multi-scale epoch: "
                "training.epochs must exceed training.warmup_epochs"
            )
        self.use_amp = bool(training_config.get("amp", True)) and self.device.type == "cuda"
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(self.epochs, 1),
            eta_min=float(training_config.get("min_lr", 1e-6)),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
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
            "data": {
                "image_size": [int(value) for value in image_hw],
                "batch_per_domain": batch_per_domain,
                "num_workers": num_workers,
                "augment": augment,
                "train_split": train_split,
            },
            "model": self.model_config,
            "loss": dict(config.get("loss", {})),
            "optimizer": dict(config.get("optimizer", {})),
            "training": {
                "epochs": self.epochs,
                "warmup_epochs": self.warmup_epochs,
                "grad_clip": self.grad_clip,
                "amp": bool(training_config.get("amp", True)),
                "min_lr": float(training_config.get("min_lr", 1e-6)),
            },
        }
        self.start_epoch = 0
        self.allow_legacy_resume = bool(training_config.get("allow_legacy_resume", False))
        resume = training_config.get("resume")
        if resume:
            self._resume(resolve_config_path(config, resume))

    def _resume(self, path: Path) -> None:
        checkpoint = load_checkpoint(
            path,
            self.device,
            allow_unsafe_legacy=self.allow_legacy_resume,
        )
        if checkpoint.get("kind") != "detector":
            raise ValueError(f"Not a detector checkpoint: {path}")
        saved_model_config = checkpoint.get("model_config")
        if saved_model_config != self.model_config:
            raise ValueError(
                "Resume checkpoint model contract differs from the current canonical MSHNet"
            )
        if checkpoint.get("source_split_records") != self.source_split_records:
            raise ValueError("Resume checkpoint frozen source-split contract has changed")
        saved_resume_contract = checkpoint.get("resume_contract")
        if saved_resume_contract != self.resume_contract:
            if not (self.allow_legacy_resume and saved_resume_contract is None):
                raise ValueError("Resume checkpoint training contract has changed")
            self.logger.warning("Legacy resume has no complete training contract")
        self.model.load_state_dict(checkpoint["model_state"])
        # RC-MSHNET-PATCH: preserve initialization on resume
        self.initialization_report = checkpoint.get("initialization")
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler_state" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        rng_state = checkpoint.get("rng_state")
        loader_state = checkpoint.get("balanced_batcher_state")
        if rng_state is None or loader_state is None:
            if not self.allow_legacy_resume:
                raise ValueError(
                    "Resume checkpoint lacks RNG/loader state; set "
                    "training.allow_legacy_resume=true only for a diagnostic continuation"
                )
            self.logger.warning("Resuming legacy checkpoint without exact RNG restoration")
        else:
            self.train_batches.load_state_dict(loader_state)
            restore_rng_state(rng_state)
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        self.logger.info("Resumed from %s at epoch %d", path, self.start_epoch)

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        sums: dict[str, float] = defaultdict(float)
        steps = 0
        multi_scale = epoch >= self.warmup_epochs
        for batch in self.train_batches:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            domain_ids = batch["domain_id"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                output = forward_mshnet(self.model, images, warm_flag=multi_scale)
                loss_output = self.objective(output, masks, domain_ids)
            if not torch.isfinite(loss_output.total):
                raise FloatingPointError(f"Non-finite detector loss at epoch {epoch}")
            self.scaler.scale(loss_output.total).backward()
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            for key, value in loss_output.metrics.items():
                sums[key] += value
            steps += 1
        return _average_metrics(sums, steps)

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        if not self.val_loaders:
            return {}
        self.model.eval()
        sums: dict[str, float] = defaultdict(float)
        steps = 0
        for loader in self.val_loaders:
            for batch in loader:
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)
                domain_ids = batch["domain_id"].to(self.device, non_blocking=True)
                output = forward_mshnet(self.model, images, warm_flag=True)
                loss_output = self.objective(output, masks, domain_ids)
                for key, value in loss_output.metrics.items():
                    sums[key] += value
                steps += 1
        return {
            f"diagnostic_test_{key}": value
            for key, value in _average_metrics(sums, steps).items()
        }

    def _checkpoint_payload(self, epoch: int) -> dict[str, Any]:
        model_state = self.model.state_dict()
        multi_scale_trained = epoch >= self.warmup_epochs
        return {
            "format_version": 2,
            "kind": "detector",
            "epoch": epoch,
            "checkpoint_selection": "fixed_last",
            "selection_rule": "fixed_last",
            "test_labels_used_for_selection": False,
            "diagnostic_test_eval": self.diagnostic_test_eval,
            "diagnostic_only": self.diagnostic_only,
            "formal_paper_checkpoint": not self.diagnostic_only,
            "warm_flag": multi_scale_trained,
            "inference_head": (
                "multi_scale_fused" if multi_scale_trained else "warm_stage_single"
            ),
            "model_state": model_state,
            "net": model_state,
            "model_config": self.model_config,
            # RC-MSHNET-PATCH: initialization provenance
            "initialization": getattr(self, "initialization_report", None),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "source_names": self.source_names,
            "source_split_records": self.source_split_records,
            "balanced_batcher_state": self.train_batches.state_dict(),
            "rng_state": capture_rng_state(),
            "resume_contract": self.resume_contract,
            "determinism_contract": self.resume_contract["determinism_contract"],
            "config": public_config(self.config),
        }

    def run(self) -> Path:
        history_path = self.output_dir / "history.csv"
        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self._train_epoch(epoch)
            val_metrics = (
                self._validate()
                if self.validation_interval > 0 and (epoch + 1) % self.validation_interval == 0
                else {}
            )
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            row = {"epoch": epoch, "lr": learning_rate, **train_metrics, **val_metrics}
            append_csv(history_path, row)
            self.logger.info(
                "epoch=%d/%d lr=%.3e train=%.6f diagnostic_test=%s",
                epoch + 1,
                self.epochs,
                learning_rate,
                train_metrics["loss_total"],
                (
                    f"{val_metrics['diagnostic_test_loss_total']:.6f}"
                    if val_metrics
                    else "disabled"
                ),
            )
            # Advance the scheduler before serializing so resume starts with the
            # learning rate intended for the next epoch rather than repeating one step.
            self.scheduler.step()
            save_checkpoint(self.output_dir / "last.pt", self._checkpoint_payload(epoch))
        return self.output_dir / "last.pt"
