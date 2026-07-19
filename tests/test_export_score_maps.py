from __future__ import annotations

import numpy as np
import torch

from evaluation.export_score_maps import _load_checkpoint_safely


def test_safe_checkpoint_loader_accepts_numpy_float64(tmp_path) -> None:
    checkpoint = tmp_path / "numpy_float64.pt"
    expected = np.float64(0.125)
    torch.save({"metric": expected}, checkpoint)

    payload = _load_checkpoint_safely(checkpoint)

    assert isinstance(payload["metric"], np.float64)
    assert payload["metric"] == expected
