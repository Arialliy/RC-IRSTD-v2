"""Canonical balanced-domain loader exports and DataLoader compatibility.

The primary names remain aliases, so fixes and resume contracts in
:mod:`data_ext.balanced_domain_loader` remain authoritative.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import DataLoader

from data_ext.balanced_domain_loader import (
    BalancedDomainLoader,
    merge_domain_batches,
)


class BalancedDomainBatcher:
    """Cycle pre-built per-domain loaders into equal composite batches.

    The complete-solution trainers construct the individual DataLoaders.  This
    compatibility adapter retains their metadata while enforcing the same
    equal-domain-count invariant as :class:`BalancedDomainLoader`.
    """

    def __init__(self, loaders: Sequence[DataLoader[Any]]) -> None:
        if not loaders:
            raise ValueError("At least one domain DataLoader is required")
        if len({id(loader) for loader in loaders}) != len(loaders):
            raise ValueError("Each source domain must use a distinct DataLoader")
        if any(len(loader) == 0 for loader in loaders):
            raise ValueError("Every source-domain DataLoader must yield at least one batch")
        self.loaders = list(loaders)
        self.steps_per_epoch = max(len(loader) for loader in self.loaders)
        self.dataset_lengths = [int(len(loader.dataset)) for loader in self.loaders]

    def __len__(self) -> int:
        return self.steps_per_epoch

    def state_dict(self) -> dict[str, Any]:
        states: list[torch.Tensor] = []
        for loader in self.loaders:
            if loader.generator is None:
                raise RuntimeError("Every training DataLoader must have an explicit generator")
            states.append(loader.generator.get_state().clone())
        return {
            "schema_version": 1,
            "dataset_lengths": list(self.dataset_lengths),
            "steps_per_epoch": int(self.steps_per_epoch),
            "batch_sizes": [loader.batch_size for loader in self.loaders],
            "num_workers": [loader.num_workers for loader in self.loaders],
            "generator_states": states,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("balanced batcher state must be a mapping")
        expected = {
            "dataset_lengths": list(self.dataset_lengths),
            "steps_per_epoch": int(self.steps_per_epoch),
            "batch_sizes": [loader.batch_size for loader in self.loaders],
            "num_workers": [loader.num_workers for loader in self.loaders],
        }
        mismatches = [
            f"{key}: saved={state.get(key)!r}, current={value!r}"
            for key, value in expected.items()
            if state.get(key) != value
        ]
        if mismatches:
            raise ValueError("balanced batcher resume mismatch:\n- " + "\n- ".join(mismatches))
        generator_states = state.get("generator_states")
        if not isinstance(generator_states, (list, tuple)) or len(generator_states) != len(
            self.loaders
        ):
            raise ValueError("balanced batcher has invalid generator_states")
        for loader, generator_state in zip(self.loaders, generator_states):
            if loader.generator is None or not isinstance(generator_state, torch.Tensor):
                raise TypeError("invalid DataLoader generator state")
            loader.generator.set_state(generator_state.detach().to(device="cpu"))

    def __iter__(self) -> Iterator[dict[str, Any]]:
        iterators = [iter(loader) for loader in self.loaders]
        for _ in range(self.steps_per_epoch):
            batches: list[dict[str, Any]] = []
            for index, loader in enumerate(self.loaders):
                try:
                    batch = next(iterators[index])
                except StopIteration:
                    iterators[index] = iter(loader)
                    batch = next(iterators[index])
                if not isinstance(batch, dict):
                    raise TypeError("BalancedDomainBatcher requires mapping batches")
                batches.append(batch)
            merged = merge_domain_batches(batches)
            merged["meta"] = [
                meta
                for batch in batches
                for meta in list(batch.get("meta", []))
            ]
            counts = torch.bincount(
                merged["domain_id"].detach().to(device="cpu", dtype=torch.long)
            )
            nonzero = counts[counts > 0]
            if nonzero.numel() != len(self.loaders) or not torch.equal(
                nonzero, torch.full_like(nonzero, nonzero[0])
            ):
                raise RuntimeError(
                    f"Unbalanced source-domain batch counts: {counts.tolist()}"
                )
            yield merged


__all__ = [
    "BalancedDomainBatcher",
    "BalancedDomainLoader",
    "merge_domain_batches",
]
