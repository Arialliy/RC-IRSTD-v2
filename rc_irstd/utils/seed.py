from __future__ import annotations

import os
import random
from collections.abc import Mapping

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def capture_rng_state() -> dict[str, object]:
    """Capture process RNGs at an epoch boundary for exact, safe resume.

    NumPy's native RNG tuple contains an ``ndarray`` whose pickle encoding
    needs globals rejected by ``torch.load(weights_only=True)``.  Store that
    array as a tensor instead so checkpoints produced by this repository stay
    loadable through PyTorch's restricted unpickler.
    """

    numpy_state = np.random.get_state()

    state: dict[str, object] = {
        "schema_version": 2,
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [],
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [value.clone() for value in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: Mapping[str, object]) -> None:
    """Restore process RNGs before constructing the next epoch iterator."""

    if not isinstance(state, Mapping):
        raise TypeError("rng_state must be a mapping")
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required.difference(state)
    if missing:
        raise ValueError("rng_state is missing: " + ", ".join(sorted(missing)))
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if isinstance(numpy_state, Mapping):
        numpy_tensor = numpy_state.get("state")
        if not isinstance(numpy_tensor, torch.Tensor):
            raise TypeError("rng_state.numpy.state must be a tensor")
        np.random.set_state(
            (
                str(numpy_state.get("bit_generator", "MT19937")),
                numpy_tensor.detach().to(device="cpu", dtype=torch.uint32).numpy(),
                int(numpy_state.get("position", 0)),
                int(numpy_state.get("has_gauss", 0)),
                float(numpy_state.get("cached_gaussian", 0.0)),
            )
        )
    else:
        # Compatibility for an explicitly trusted legacy checkpoint that has
        # already been loaded with the unrestricted unpickler.
        np.random.set_state(numpy_state)
    cpu_state = state["torch_cpu"]
    if not isinstance(cpu_state, torch.Tensor):
        raise TypeError("rng_state.torch_cpu must be a tensor")
    torch.set_rng_state(cpu_state.detach().to(device="cpu"))
    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, (list, tuple)):
        raise TypeError("rng_state.torch_cuda must be a sequence")
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("CUDA topology changed across resume")
        if not all(isinstance(value, torch.Tensor) for value in cuda_states):
            raise TypeError("every CUDA RNG state must be a tensor")
        torch.cuda.set_rng_state_all(
            [value.detach().to(device="cpu") for value in cuda_states]
        )


__all__ = ["capture_rng_state", "restore_rng_state", "seed_everything"]
