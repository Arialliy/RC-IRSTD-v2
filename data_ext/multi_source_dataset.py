"""Thin domain annotations for legacy IRSTD training datasets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import Dataset


class DomainDataset(Dataset):
    """Attach stable domain identity without changing the wrapped dataset."""

    def __init__(
        self,
        dataset: Dataset,
        domain_id: int,
        domain_name: str,
    ) -> None:
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a torch Dataset")
        if isinstance(domain_id, bool) or not isinstance(domain_id, int):
            raise TypeError("domain_id must be an integer")
        if domain_id < 0:
            raise ValueError("domain_id must be non-negative")
        if not isinstance(domain_name, str) or not domain_name.strip():
            raise ValueError("domain_name must be a non-empty string")
        if len(dataset) == 0:
            raise ValueError(f"domain dataset {domain_name!r} is empty")
        self.dataset = dataset
        self.domain_id = domain_id
        self.domain_name = domain_name.strip()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        if isinstance(sample, Mapping):
            if "image" not in sample or "mask" not in sample:
                raise KeyError("mapping samples must contain image and mask")
            image, mask = sample["image"], sample["mask"]
        elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
            image, mask = sample[0], sample[1]
        else:
            raise TypeError(
                "wrapped dataset samples must be (image, mask) or a mapping"
            )
        if not isinstance(image, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise TypeError("wrapped image and mask must be torch tensors")
        return {
            "image": image,
            "mask": mask,
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "domain_name": self.domain_name,
        }


def wrap_domain_datasets(
    datasets: list[Dataset],
    domain_names: list[str],
) -> list[DomainDataset]:
    if len(datasets) != len(domain_names):
        raise ValueError("datasets and domain_names must have the same length")
    if len(set(domain_names)) != len(domain_names):
        raise ValueError("domain_names must be unique")
    return [
        DomainDataset(dataset, domain_id, domain_name)
        for domain_id, (dataset, domain_name) in enumerate(
            zip(datasets, domain_names)
        )
    ]
