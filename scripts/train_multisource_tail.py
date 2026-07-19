"""Train MSHNet with balanced multi-source Tail-CVaR and Miss-CVaR."""

from __future__ import annotations

import json
import hashlib
import math
import os
import random
import sys
import time
import warnings
from argparse import ArgumentParser, Namespace
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adagrad
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_ext.balanced_domain_loader import BalancedDomainLoader
from data_ext.multi_source_dataset import wrap_domain_datasets
from data_ext.split_utils import ensure_unique_sample_ids
from losses.hard_target_loss import (
    hard_target_miss_from_scores,
    target_object_scores,
)
from losses.local_peak_cvar import (
    domain_local_peak_tail_risks,
    stack_domain_risks,
)
from losses.smooth_worst_domain import smooth_worst_domain
from main import (
    canonical_model_state_dict,
    load_model_state_dict,
    seed_everything,
    select_device,
    torch_load_compat,
    validate_input_size,
)
from model.MSHNet import MSHNet
from model.loss import SLSIoULoss
from utils.data import IRSTD_Dataset


RESUME_CRITICAL_CONFIG_KEYS = (
    "source_dirs",
    "source_train_splits",
    "domain_names",
    "base_size",
    "crop_size",
    "batch_per_domain",
    "steps_per_epoch",
    "lr",
    "warm_epoch",
    "num_workers",
    "multi_gpus",
    "seed",
    "lambda_tail",
    "lambda_miss",
    "tail_q",
    "miss_q",
    "object_response_q",
    "tail_gamma",
    "peak_kernel",
    "peak_min_score",
    "checkpoint_selection",
)


class ScalarAverages:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, name: str, value: float, count: int = 1) -> None:
        self.sums[name] = self.sums.get(name, 0.0) + float(value) * int(count)
        self.counts[name] = self.counts.get(name, 0) + int(count)

    def averages(self) -> dict[str, float]:
        return {
            name: self.sums[name] / self.counts[name]
            for name in sorted(self.sums)
            if self.counts[name] > 0
        }


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(
        description="Balanced multi-source MSHNet Tail-CVaR training"
    )
    parser.add_argument("--source-dirs", nargs="+", required=True)
    parser.add_argument(
        "--source-train-split-files",
        nargs="+",
        default=None,
        help="Optional explicit train manifest per source directory, in source order",
    )
    parser.add_argument(
        "--source-test-split-files",
        nargs="+",
        default=None,
        help="Optional explicit test manifest per source directory, used only for leakage audit",
    )
    parser.add_argument("--domain-names", nargs="+", default=None)
    parser.add_argument("--batch-per-domain", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--warm-epoch", type=int, default=5)
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="workers per source-domain DataLoader",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument("--multi-gpus", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="repro_runs/rc_tail")
    parser.add_argument("--resume-path", default="")
    parser.add_argument(
        "--allow-legacy-resume",
        action="store_true",
        help=(
            "allow a diagnostic continuation from a checkpoint without the "
            "complete optimizer/RNG/loader contract"
        ),
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=("fixed_last", "train_loss"),
        default="fixed_last",
        help=(
            "weight.pkl selection rule; fixed_last is the leakage-safe paper "
            "default, while train_loss is diagnostic only"
        ),
    )

    parser.add_argument("--lambda-tail", type=float, default=0.1)
    parser.add_argument("--lambda-miss", type=float, default=0.1)
    parser.add_argument("--tail-q", type=float, default=0.01)
    parser.add_argument("--miss-q", type=float, default=0.2)
    parser.add_argument("--object-response-q", type=float, default=0.25)
    parser.add_argument("--tail-gamma", type=float, default=10.0)
    parser.add_argument("--peak-kernel", type=int, default=3)
    parser.add_argument("--peak-min-score", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args(argv)


def validate_training_args(args: Namespace) -> None:
    if len(args.source_dirs) < 2:
        raise ValueError("at least two --source-dirs are required")
    resolved_sources = [str(Path(path).expanduser().resolve()) for path in args.source_dirs]
    if len(set(resolved_sources)) != len(resolved_sources):
        raise ValueError("--source-dirs must be unique")
    missing = [path for path in resolved_sources if not Path(path).is_dir()]
    if missing:
        raise FileNotFoundError("missing source directories: " + ", ".join(missing))
    if args.domain_names is not None:
        if len(args.domain_names) != len(args.source_dirs):
            raise ValueError("--domain-names must match --source-dirs")
        if len(set(args.domain_names)) != len(args.domain_names):
            raise ValueError("--domain-names must be unique")
        if any(not name.strip() for name in args.domain_names):
            raise ValueError("--domain-names cannot contain empty names")
    for attribute in ("source_train_split_files", "source_test_split_files"):
        values = getattr(args, attribute, None)
        if values is not None and len(values) != len(args.source_dirs):
            raise ValueError(
                f"--{attribute.replace('_', '-')} must provide one path per source"
            )

    validate_input_size("base_size", args.base_size)
    validate_input_size("crop_size", args.crop_size)
    integer_positive = {
        "batch_per_domain": args.batch_per_domain,
        "epochs": args.epochs,
        "log_every": args.log_every,
    }
    for name, value in integer_positive.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if args.steps_per_epoch is not None and args.steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.multi_gpus:
        raise ValueError(
            "--multi-gpus is disabled for the balanced-domain trainer: "
            "DataParallel splits domain-contiguous batches across replicas and "
            "MSHNet BatchNorm would then use domain-biased local statistics. "
            "Run independent single-GPU jobs until a validated DDP/SyncBN path exists."
        )
    if args.warm_epoch < -1:
        raise ValueError("warm_epoch must be at least -1")
    if args.lr <= 0.0:
        raise ValueError("lr must be positive")
    if args.lambda_tail < 0.0 or args.lambda_miss < 0.0:
        raise ValueError("loss weights must be non-negative")
    for name in ("tail_q", "miss_q", "object_response_q"):
        value = float(getattr(args, name))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
    if args.tail_gamma <= 0.0:
        raise ValueError("tail_gamma must be positive")
    if args.peak_kernel <= 0 or args.peak_kernel % 2 == 0:
        raise ValueError("peak_kernel must be a positive odd integer")
    if not 0.0 <= args.peak_min_score <= 1.0:
        raise ValueError("peak_min_score must be in [0, 1]")


def _domain_names(args: Namespace) -> list[str]:
    if args.domain_names is not None:
        return [name.strip() for name in args.domain_names]
    names = [Path(path).expanduser().resolve().name for path in args.source_dirs]
    if len(set(names)) != len(names):
        raise ValueError(
            "source directory basenames collide; pass explicit --domain-names"
        )
    return names


def _serializable_config(
    args: Namespace,
    domain_names: list[str],
) -> dict[str, object]:
    config = dict(vars(args))
    config["source_dirs"] = [
        str(Path(path).expanduser().resolve()) for path in args.source_dirs
    ]
    config["domain_names"] = list(domain_names)
    split_records: list[dict[str, object]] = []
    explicit_train = getattr(args, "source_train_split_files", None)
    explicit_test = getattr(args, "source_test_split_files", None)
    for source_index, (source_dir, domain_name) in enumerate(
        zip(config["source_dirs"], domain_names)
    ):
        split_path = Path(
            IRSTD_Dataset._find_split_file(
                str(source_dir),
                "train",
                split_file=(explicit_train[source_index] if explicit_train else None),
            )
        ).resolve()
        split_entries = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        if not split_entries:
            raise ValueError(f"source train split is empty: {split_path}")
        if len(set(split_entries)) != len(split_entries):
            raise ValueError(f"source train split contains duplicate entries: {split_path}")
        train_ids = set(ensure_unique_sample_ids(split_entries))
        record: dict[str, object] = {
            "domain_name": domain_name,
            "path": str(split_path),
            "sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "num_entries": len(split_entries),
        }
        try:
            evaluation_path = Path(
                IRSTD_Dataset._find_split_file(
                    str(source_dir),
                    "test",
                    split_file=(explicit_test[source_index] if explicit_test else None),
                )
            ).resolve()
        except FileNotFoundError:
            evaluation_path = None
        if evaluation_path is not None:
            evaluation_entries = [
                line.strip()
                for line in evaluation_path.read_text(
                    encoding="utf-8-sig"
                ).splitlines()
                if line.strip()
            ]
            evaluation_ids = set(ensure_unique_sample_ids(evaluation_entries))
            overlap = sorted(train_ids.intersection(evaluation_ids))
            if overlap:
                raise ValueError(
                    f"source train/evaluation split leakage for {domain_name}: "
                    + ", ".join(overlap[:10])
                )
            record["evaluation_path"] = str(evaluation_path)
            record["evaluation_sha256"] = hashlib.sha256(
                evaluation_path.read_bytes()
            ).hexdigest()
            record["evaluation_num_entries"] = len(evaluation_entries)
        split_records.append(record)
    config["source_train_splits"] = split_records
    return config


def _new_run_directory(save_dir: str) -> Path:
    root = Path(save_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    candidate = root / f"MSHNet-tail-{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"MSHNet-tail-{stamp}-{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _append_json_line(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def capture_rng_state() -> dict[str, object]:
    """Capture parent-process RNG state at a completed epoch boundary."""

    state: dict[str, object] = {
        "schema_version": 1,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "torch_cuda": [],
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: Mapping[str, object]) -> None:
    """Restore RNG state before constructing the next epoch iterator."""

    if not isinstance(state, Mapping):
        raise TypeError("rng_state must be a mapping")
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required.difference(state)
    if missing:
        raise ValueError("rng_state is missing fields: " + ", ".join(sorted(missing)))
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_cpu = state["torch_cpu"]
    if not isinstance(torch_cpu, torch.Tensor):
        raise TypeError("rng_state.torch_cpu must be a tensor")
    torch.set_rng_state(torch_cpu.detach().to(device="cpu"))

    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, (list, tuple)):
        raise TypeError("rng_state.torch_cuda must be a sequence")
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "CUDA topology changed across resume: "
                f"saved={len(cuda_states)}, current={torch.cuda.device_count()}"
            )
        if not all(isinstance(item, torch.Tensor) for item in cuda_states):
            raise TypeError("every CUDA RNG state must be a tensor")
        torch.cuda.set_rng_state_all(
            [item.detach().to(device="cpu") for item in cuda_states]
        )


def _build_balanced_loader(
    args: Namespace,
    domain_names: list[str],
    device: torch.device,
) -> BalancedDomainLoader:
    legacy_datasets = []
    explicit_train = getattr(args, "source_train_split_files", None)
    for source_index, source_dir in enumerate(args.source_dirs):
        dataset_args = SimpleNamespace(
            dataset_dir=str(Path(source_dir).expanduser().resolve()),
            crop_size=args.crop_size,
            base_size=args.base_size,
            train_split_file=(
                explicit_train[source_index] if explicit_train else ""
            ),
            test_split_file="",
        )
        legacy_datasets.append(IRSTD_Dataset(dataset_args, mode="train"))
    datasets = wrap_domain_datasets(legacy_datasets, domain_names)
    return BalancedDomainLoader(
        datasets,
        batch_size_per_domain=args.batch_per_domain,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        seed=args.seed,
        steps_per_epoch=args.steps_per_epoch,
    )


def _multi_scale_sls(
    loss_function: SLSIoULoss,
    auxiliary_logits: list[torch.Tensor],
    final_logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    warm_epoch: int,
    epoch: int,
    downsample: nn.Module,
) -> torch.Tensor:
    loss = loss_function(final_logits, masks, warm_epoch, epoch)
    scaled_masks = masks
    for scale_index, scale_logits in enumerate(auxiliary_logits):
        if scale_index > 0:
            scaled_masks = downsample(scaled_masks)
        loss = loss + loss_function(
            scale_logits,
            scaled_masks,
            warm_epoch,
            epoch,
        )
    return loss / (len(auxiliary_logits) + 1)


def validate_resume_config(
    current_config: Mapping[str, object],
    checkpoint: Mapping[str, object],
) -> None:
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, Mapping):
        raise ValueError(
            "Tail-CVaR resume checkpoint has no config mapping; refusing an "
            "unverifiable continuation"
        )

    mismatches: list[str] = []
    for key in RESUME_CRITICAL_CONFIG_KEYS:
        if key not in saved_config or key not in current_config:
            mismatches.append(
                f"{key}: saved={saved_config.get(key, '<missing>')!r}, "
                f"current={current_config.get(key, '<missing>')!r}"
            )
            continue
        saved_value = saved_config[key]
        current_value = current_config[key]
        if isinstance(saved_value, float) or isinstance(current_value, float):
            try:
                equal = math.isclose(
                    float(saved_value),
                    float(current_value),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            except (TypeError, ValueError):
                equal = False
        else:
            equal = saved_value == current_value
        if not equal:
            mismatches.append(
                f"{key}: saved={saved_value!r}, current={current_value!r}"
            )
    if mismatches:
        raise ValueError(
            "Tail-CVaR resume config mismatch; start a new run instead:\n- "
            + "\n- ".join(mismatches)
        )


def _load_resume(
    path: str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    current_config: Mapping[str, object],
) -> tuple[int, int, float, Mapping[str, object]]:
    checkpoint = torch_load_compat(
        path,
        map_location=device,
        full_checkpoint=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("resume checkpoint must be a mapping")
    validate_resume_config(current_config, checkpoint)
    load_model_state_dict(model, checkpoint)
    if "optimizer" not in checkpoint:
        raise ValueError("resume checkpoint has no optimizer state")
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_train_loss = float(checkpoint.get("best_train_loss", float("inf")))
    return start_epoch, global_step, best_train_loss, checkpoint


def train(args: Namespace) -> Path:
    validate_training_args(args)
    seed_everything(args.seed)
    device = select_device(args.device)
    domain_names = _domain_names(args)
    config = _serializable_config(args, domain_names)
    balanced_loader = _build_balanced_loader(args, domain_names, device)

    model: nn.Module = MSHNet(3)
    model.to(device)
    optimizer = Adagrad(model.parameters(), lr=args.lr)
    sls_loss = SLSIoULoss()
    downsample = nn.MaxPool2d(2, 2)

    start_epoch = 0
    global_step = 0
    best_train_loss = float("inf")
    resume_reproducibility = "fresh_run"
    if args.resume_path:
        resume_path = Path(args.resume_path).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        run_directory = resume_path.parent
        start_epoch, global_step, best_train_loss, resume_checkpoint = _load_resume(
            str(resume_path),
            model=model,
            optimizer=optimizer,
            device=device,
            current_config=config,
        )
        loader_state = resume_checkpoint.get("balanced_loader_state")
        rng_state = resume_checkpoint.get("rng_state")
        if isinstance(loader_state, Mapping) and isinstance(rng_state, Mapping):
            balanced_loader.load_state_dict(loader_state)
            restore_rng_state(rng_state)
            resume_reproducibility = "epoch_boundary_rng_restored"
        else:
            if not args.allow_legacy_resume:
                raise ValueError(
                    "resume checkpoint lacks RNG and/or balanced-loader state; "
                    "pass --allow-legacy-resume only for a diagnostic continuation"
                )
            resume_reproducibility = "legacy_rng_state_missing"
            warnings.warn(
                "resume checkpoint lacks RNG and/or balanced-loader state; "
                "continuation is diagnostic and not epoch-boundary reproducible",
                RuntimeWarning,
            )
        _append_json_line(
            run_directory / "events.jsonl",
            {
                "event": "resume",
                "checkpoint": str(resume_path),
                "start_epoch": start_epoch,
                "resume_reproducibility": resume_reproducibility,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
    else:
        run_directory = _new_run_directory(args.save_dir)
        _atomic_json(run_directory / "config.json", config)

    metrics_path = run_directory / "metrics.jsonl"
    expected_domain_ids = list(range(len(domain_names)))
    print(f"device={device}; domains={domain_names}; steps/epoch={len(balanced_loader)}")
    print(f"run_directory={run_directory}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        warm_flag = bool(epoch > args.warm_epoch)
        epoch_start = time.time()
        meters = ScalarAverages()
        progress = tqdm(
            balanced_loader,
            total=len(balanced_loader),
            desc=f"Epoch {epoch}",
            disable=not sys.stderr.isatty(),
        )
        for step_index, batch in enumerate(progress):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            domain_ids = batch["domain_id"].to(device, non_blocking=True)

            auxiliary_logits, final_logits = model(images, warm_flag)
            loss_sls = _multi_scale_sls(
                sls_loss,
                auxiliary_logits,
                final_logits,
                masks,
                warm_epoch=args.warm_epoch,
                epoch=epoch,
                downsample=downsample,
            )
            domain_risk_map = domain_local_peak_tail_risks(
                final_logits,
                masks,
                domain_ids,
                tail_fraction=args.tail_q,
                kernel_size=args.peak_kernel,
                min_score=args.peak_min_score,
            )
            sorted_ids, domain_risk_tensor = stack_domain_risks(domain_risk_map)
            if sorted_ids != expected_domain_ids:
                raise RuntimeError(
                    f"domain ids changed inside a balanced batch: {sorted_ids}"
                )
            loss_tail = smooth_worst_domain(
                domain_risk_tensor,
                gamma=args.tail_gamma,
            )
            domain_miss_risks: dict[int, torch.Tensor] = {}
            active_domain_miss_risks: dict[int, torch.Tensor] = {}
            domain_object_counts: dict[int, int] = {}
            target_object_count = 0
            for domain_id in expected_domain_ids:
                selected = domain_ids == domain_id
                object_scores = target_object_scores(
                    final_logits[selected],
                    masks[selected],
                    response_fraction=args.object_response_q,
                )
                object_count = int(object_scores.numel())
                domain_object_counts[domain_id] = object_count
                target_object_count += object_count
                domain_miss_risks[domain_id] = hard_target_miss_from_scores(
                    object_scores,
                    miss_fraction=args.miss_q,
                )
                if object_count:
                    active_domain_miss_risks[domain_id] = domain_miss_risks[domain_id]
            if active_domain_miss_risks:
                _, domain_miss_tensor = stack_domain_risks(
                    active_domain_miss_risks
                )
                loss_miss = smooth_worst_domain(
                    domain_miss_tensor,
                    gamma=args.tail_gamma,
                )
            else:
                # No target exists in this augmented batch; the miss objective
                # is undefined and contributes a differentiable zero.  The SLS
                # empty-target branch still penalises foreground false alarms.
                loss_miss = final_logits.sum() * 0.0
            total_loss = (
                loss_sls
                + args.lambda_tail * loss_tail
                + args.lambda_miss * loss_miss
            )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"non-finite loss at epoch={epoch}, step={step_index}"
                )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float("inf"),
                error_if_nonfinite=True,
            )
            optimizer.step()
            global_step += 1

            batch_size = int(images.shape[0])
            meters.update("loss_total", total_loss.detach().item(), batch_size)
            meters.update("loss_sls", loss_sls.detach().item(), batch_size)
            meters.update("loss_tail", loss_tail.detach().item(), batch_size)
            meters.update("loss_miss", loss_miss.detach().item(), batch_size)
            meters.update("target_objects", target_object_count, 1)
            for domain_id, domain_name in enumerate(domain_names):
                meters.update(
                    f"tail_risk/{domain_name}",
                    domain_risk_map[domain_id].detach().item(),
                    1,
                )
                meters.update(
                    f"miss_risk/{domain_name}",
                    domain_miss_risks[domain_id].detach().item(),
                    1,
                )
                meters.update(
                    f"target_objects/{domain_name}",
                    domain_object_counts[domain_id],
                    1,
                )
            meters.update(
                "empty_crop_fraction",
                float((masks.flatten(1).sum(dim=1) == 0).float().mean().item()),
                1,
            )

            current = meters.averages()
            progress.set_postfix(
                total=f"{current['loss_total']:.4f}",
                sls=f"{current['loss_sls']:.4f}",
                tail=f"{current['loss_tail']:.4f}",
                miss=f"{current['loss_miss']:.4f}",
            )
            if (step_index + 1) % args.log_every == 0:
                print(
                    f"epoch={epoch} step={step_index + 1}/{len(balanced_loader)} "
                    f"global_step={global_step} loss={current['loss_total']:.6f}"
                )

        epoch_metrics: dict[str, object] = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "warm_flag": warm_flag,
            "inference_head": (
                "multiscale_fusion" if warm_flag else "single_output_0"
            ),
            "duration_seconds": float(time.time() - epoch_start),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "steps": int(len(balanced_loader)),
            "batch_per_domain": int(args.batch_per_domain),
            "samples_per_domain": int(balanced_loader.samples_per_domain),
            "checkpoint_selection": str(args.checkpoint_selection),
            "resume_reproducibility": resume_reproducibility,
        }
        epoch_metrics.update(meters.averages())

        mean_train_loss = float(epoch_metrics["loss_total"])
        state_dict = canonical_model_state_dict(model)
        improved_train_loss = mean_train_loss < best_train_loss
        if improved_train_loss:
            best_train_loss = mean_train_loss
        inference_payload = {
            "state_dict": state_dict,
            "epoch": int(epoch),
            "warm_flag": warm_flag,
            "inference_head": epoch_metrics["inference_head"],
            "train_loss": mean_train_loss,
            "best_train_loss": float(best_train_loss),
            "config": config,
        }
        _atomic_torch_save(
            run_directory / "last.pkl",
            {
                **inference_payload,
                "selection_rule": "fixed_last_complete_epoch",
            },
        )
        if improved_train_loss:
            _atomic_torch_save(
                run_directory / "best_train_loss.pkl",
                {
                    **inference_payload,
                    "selection_rule": "minimum_source_training_loss_diagnostic_only",
                },
            )
        if args.checkpoint_selection == "fixed_last" or improved_train_loss:
            selected_rule = (
                "fixed_last_complete_epoch"
                if args.checkpoint_selection == "fixed_last"
                else "minimum_source_training_loss_diagnostic_only"
            )
            _atomic_torch_save(
                run_directory / "weight.pkl",
                {
                    **inference_payload,
                    "selection_rule": selected_rule,
                },
            )
        checkpoint = {
            "net": state_dict,
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "warm_flag": warm_flag,
            "inference_head": epoch_metrics["inference_head"],
            "best_train_loss": float(best_train_loss),
            "last_metrics": epoch_metrics,
            "config": config,
            "checkpoint_selection": str(args.checkpoint_selection),
            "balanced_loader_state": balanced_loader.state_dict(),
            "rng_state": capture_rng_state(),
            "resume_reproducibility": resume_reproducibility,
        }
        _atomic_torch_save(run_directory / "checkpoint.pkl", checkpoint)
        # Append the human-readable record only after the corresponding atomic
        # checkpoint is durable.  A crash can then at worst omit a log line; it
        # cannot advertise an epoch that has no resumable state.
        _append_json_line(metrics_path, epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False, sort_keys=True))

    if start_epoch >= args.epochs:
        print(
            f"checkpoint epoch {start_epoch - 1} already reaches "
            f"requested epochs={args.epochs}"
        )
    return run_directory


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
