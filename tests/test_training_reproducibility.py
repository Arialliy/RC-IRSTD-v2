from __future__ import annotations

import random
from argparse import Namespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from data_ext.balanced_domain_loader import BalancedDomainLoader
from data_ext.multi_source_dataset import DomainDataset
from evaluation.export_score_maps import resolve_checkpoint_warm_flag
from main import Trainer, parse_args as parse_main_args
from scripts.train_multisource_tail import (
    _serializable_config,
    capture_rng_state,
    parse_args,
    restore_rng_state,
)


class _RandomAugmentationDataset(Dataset):
    def __init__(self, offset: int) -> None:
        self.offset = int(offset)

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int):
        random_part = random.random()
        numpy_part = float(np.random.random())
        torch_part = float(torch.rand(()))
        value = self.offset + index + random_part + numpy_part + torch_part
        return torch.tensor([value]), torch.tensor([index])


def _loader(seed: int) -> BalancedDomainLoader:
    return BalancedDomainLoader(
        [
            DomainDataset(_RandomAugmentationDataset(0), 0, "first"),
            DomainDataset(_RandomAugmentationDataset(100), 1, "second"),
        ],
        batch_size_per_domain=2,
        num_workers=1,
        seed=seed,
        steps_per_epoch=4,
    )


def _collect_epoch(loader: BalancedDomainLoader) -> list[torch.Tensor]:
    return [batch["image"].clone() for batch in loader]


def test_balanced_loader_epoch_boundary_resume_replays_workers_and_sampler() -> None:
    uninterrupted = _loader(seed=17)
    _collect_epoch(uninterrupted)
    state = uninterrupted.state_dict()
    expected = _collect_epoch(uninterrupted)

    resumed = _loader(seed=17)
    resumed.load_state_dict(state)
    actual = _collect_epoch(resumed)
    assert len(actual) == len(expected)
    for actual_batch, expected_batch in zip(actual, expected):
        torch.testing.assert_close(actual_batch, expected_batch, rtol=0.0, atol=0.0)


def test_balanced_loader_rejects_changed_resume_contract() -> None:
    loader = _loader(seed=3)
    state = loader.state_dict()
    state["steps_per_epoch"] = 99
    with pytest.raises(ValueError, match="steps_per_epoch"):
        loader.load_state_dict(state)


def test_parent_rng_capture_and_restore_is_exact() -> None:
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    restore_rng_state(state)
    actual = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert actual == expected


def test_export_head_auto_resolution_and_mismatch_rejection() -> None:
    assert resolve_checkpoint_warm_flag(None, {"warm_flag": True}) is True
    assert resolve_checkpoint_warm_flag(None, {"warm_flag": False}) is False
    with pytest.raises(ValueError, match="no warm_flag"):
        resolve_checkpoint_warm_flag(None, {})
    with pytest.raises(ValueError, match="contradicts"):
        resolve_checkpoint_warm_flag(True, {"warm_flag": False})
    assert resolve_checkpoint_warm_flag(True, {}) is True


def test_multisource_checkpoint_selection_defaults_to_fixed_last() -> None:
    args = parse_args(["--source-dirs", "/source/a", "/source/b"])
    assert args.checkpoint_selection == "fixed_last"


def test_source_split_overlap_is_rejected_before_training(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "trainval.txt").write_text("a\nb\n", encoding="utf-8")
    (source / "test.txt").write_text("b\nc\n", encoding="utf-8")
    args = Namespace(source_dirs=[str(source)])
    with pytest.raises(ValueError, match="split leakage"):
        _serializable_config(args, ["source"])


class _TinyDetector(nn.Module):
    def __init__(self, _channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def forward(self, images: torch.Tensor, _warm: bool):
        logits = images[:, :1] * self.weight
        return [logits], logits


def _split_only_dataset(root) -> None:
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    index = root / "img_idx"
    index.mkdir()
    (index / "train_local.txt").write_text("train-a\ntrain-b\n", encoding="utf-8")
    (index / "test_local.txt").write_text("test-a\n", encoding="utf-8")


def test_single_domain_fixed_last_does_not_construct_a_test_loader(
    tmp_path, monkeypatch
) -> None:
    dataset = tmp_path / "dataset"
    _split_only_dataset(dataset)
    monkeypatch.setattr("main.MSHNet", _TinyDetector)
    args = parse_main_args(
        argv=[
            "--dataset-dir",
            str(dataset),
            "--save-dir",
            str(tmp_path / "runs"),
            "--base-size",
            "16",
            "--crop-size",
            "16",
            "--batch-size",
            "1",
            "--epochs",
            "1",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--mode",
            "train",
        ]
    )
    trainer = Trainer(args)
    assert trainer.test_loader is None
    assert trainer.train_split_contract["role"] == "train"
    assert trainer.test_split_contract["role"] == "test"

    trainer.save_epoch(0, {"loss_total": 1.25})
    weight = torch.load(
        tmp_path / "runs" / trainer.save_folder.split("/")[-1] / "weight.pkl",
        map_location="cpu",
        weights_only=True,
    )
    assert weight["selection_rule"] == "fixed_last_complete_epoch"
    assert weight["selection_uses_test_labels"] is False
    assert weight["selection_test_metrics"] is None


def test_val_is_only_a_legacy_alias_for_the_local_test_manifest(tmp_path) -> None:
    from utils.data import IRSTD_Dataset

    dataset = tmp_path / "dataset"
    _split_only_dataset(dataset)
    args = Namespace(
        dataset_dir=str(dataset),
        base_size=16,
        crop_size=16,
        train_split_file="",
        test_split_file="",
    )
    val = IRSTD_Dataset(args, mode="val")
    test = IRSTD_Dataset(args, mode="test")
    assert val.list_dir == test.list_dir
    assert val.image_ids == test.image_ids == ["test-a"]
    assert val.split_role == test.split_role == "test"
