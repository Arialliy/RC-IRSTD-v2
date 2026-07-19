"""Per-domain DataLoaders merged into exactly balanced training batches."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


def merge_domain_batches(
    batches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batches:
        raise ValueError("batches cannot be empty")
    tensor_keys = ("image", "mask", "domain_id")
    merged: dict[str, Any] = {}
    for key in tensor_keys:
        values = [batch.get(key) for batch in batches]
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError(f"every domain batch must contain tensor key {key!r}")
        merged[key] = torch.cat(values, dim=0)

    names: list[str] = []
    for batch in batches:
        value = batch.get("domain_name")
        if isinstance(value, str):
            names.append(value)
        elif isinstance(value, (list, tuple)) and all(
            isinstance(item, str) for item in value
        ):
            names.extend(value)
        else:
            raise TypeError("every domain batch must contain domain_name strings")
    merged["domain_name"] = names
    return merged


class BalancedDomainLoader:
    """Cycle per-domain loaders so every optimization step is balanced.

    The longest domain defines the default epoch length.  Shorter domains are
    reshuffled and restarted as needed, while drop-last guarantees exactly
    batch_size_per_domain samples from every source on every step.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        *,
        batch_size_per_domain: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
        steps_per_epoch: int | None = None,
    ) -> None:
        if not datasets:
            raise ValueError("at least one domain dataset is required")
        if (
            isinstance(batch_size_per_domain, bool)
            or not isinstance(batch_size_per_domain, int)
            or batch_size_per_domain <= 0
        ):
            raise ValueError("batch_size_per_domain must be a positive integer")
        if (
            isinstance(num_workers, bool)
            or not isinstance(num_workers, int)
            or num_workers < 0
        ):
            raise ValueError("num_workers must be a non-negative integer")
        if len({id(dataset) for dataset in datasets}) != len(datasets):
            raise ValueError("each source domain must use a distinct dataset object")
        for dataset in datasets:
            if len(dataset) < batch_size_per_domain:
                raise ValueError(
                    "every domain must contain at least batch_size_per_domain samples"
                )

        self.batch_size_per_domain = int(batch_size_per_domain)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.domain_count = len(datasets)
        self.dataset_lengths = [int(len(dataset)) for dataset in datasets]
        self.loaders: list[DataLoader] = []
        self.generators: list[torch.Generator] = []
        for domain_index, dataset in enumerate(datasets):
            generator = torch.Generator()
            generator.manual_seed(int(seed) + domain_index)
            self.generators.append(generator)
            self.loaders.append(
                DataLoader(
                    dataset,
                    batch_size=self.batch_size_per_domain,
                    shuffle=True,
                    drop_last=True,
                    num_workers=self.num_workers,
                    pin_memory=bool(pin_memory),
                    # Recreate workers at every epoch boundary.  Their Python,
                    # NumPy and Torch RNGs are then derived from the saved
                    # per-domain generator state, making an epoch-boundary
                    # resume reproducible.  Persistent worker RNG state is not
                    # observable from the parent process and cannot be saved in
                    # a checkpoint.
                    persistent_workers=False,
                    generator=generator,
                )
            )

        default_steps = max(len(loader) for loader in self.loaders)
        if steps_per_epoch is None:
            self.steps_per_epoch = default_steps
        else:
            if (
                isinstance(steps_per_epoch, bool)
                or not isinstance(steps_per_epoch, int)
                or steps_per_epoch <= 0
            ):
                raise ValueError("steps_per_epoch must be a positive integer")
            self.steps_per_epoch = int(steps_per_epoch)

    @property
    def batch_size(self) -> int:
        return self.domain_count * self.batch_size_per_domain

    @property
    def samples_per_domain(self) -> int:
        return self.steps_per_epoch * self.batch_size_per_domain

    def __len__(self) -> int:
        return self.steps_per_epoch

    def state_dict(self) -> dict[str, Any]:
        """Return the sampler state needed for an epoch-boundary resume."""

        return {
            "schema_version": 1,
            "domain_count": int(self.domain_count),
            "dataset_lengths": list(self.dataset_lengths),
            "batch_size_per_domain": int(self.batch_size_per_domain),
            "num_workers": int(self.num_workers),
            "seed": int(self.seed),
            "steps_per_epoch": int(self.steps_per_epoch),
            "generator_states": [
                generator.get_state().clone() for generator in self.generators
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore sampler generators after validating the loader contract."""

        if not isinstance(state, Mapping):
            raise TypeError("balanced loader state must be a mapping")
        expected = {
            "domain_count": int(self.domain_count),
            "dataset_lengths": list(self.dataset_lengths),
            "batch_size_per_domain": int(self.batch_size_per_domain),
            "num_workers": int(self.num_workers),
            "seed": int(self.seed),
            "steps_per_epoch": int(self.steps_per_epoch),
        }
        mismatches = [
            f"{key}: saved={state.get(key)!r}, current={value!r}"
            for key, value in expected.items()
            if state.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                "balanced loader resume contract mismatch:\n- "
                + "\n- ".join(mismatches)
            )
        generator_states = state.get("generator_states")
        if not isinstance(generator_states, (list, tuple)) or len(
            generator_states
        ) != len(self.generators):
            raise ValueError("balanced loader state has invalid generator_states")
        for generator, generator_state in zip(self.generators, generator_states):
            if not isinstance(generator_state, torch.Tensor):
                raise TypeError("each saved generator state must be a torch tensor")
            generator.set_state(generator_state.detach().to(device="cpu"))

    def __iter__(self) -> Iterator[dict[str, Any]]:
        iterators = [iter(loader) for loader in self.loaders]
        for _ in range(self.steps_per_epoch):
            batches: list[Mapping[str, Any]] = []
            for domain_index, loader in enumerate(self.loaders):
                try:
                    batch = next(iterators[domain_index])
                except StopIteration:
                    iterators[domain_index] = iter(loader)
                    batch = next(iterators[domain_index])
                batches.append(batch)
            merged = merge_domain_batches(batches)
            counts = torch.bincount(
                merged["domain_id"].to(dtype=torch.long, device="cpu"),
                minlength=self.domain_count,
            )
            expected = torch.full_like(counts, self.batch_size_per_domain)
            if counts.shape[0] != self.domain_count or not torch.equal(counts, expected):
                raise RuntimeError(
                    f"unbalanced domain batch: got {counts.tolist()}, "
                    f"expected {expected.tolist()}"
                )
            yield merged
