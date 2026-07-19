from __future__ import annotations

import numpy as np
from PIL import Image

from rc_irstd.data import dataset as dataset_module


def test_rectangular_pair_rotation_preserves_every_mask_pixel(monkeypatch) -> None:
    """A 90-degree augmentation must swap H/W without cropping edge targets."""

    image_values = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    mask_values = np.zeros((3, 5), dtype=np.uint8)
    mask_values[0, 4] = 255
    image = Image.fromarray(image_values, mode="RGB")
    mask = Image.fromarray(mask_values, mode="L")

    monkeypatch.setattr(dataset_module.random, "random", lambda: 1.0)
    monkeypatch.setattr(dataset_module.random, "choice", lambda _values: 90)

    rotated_image, rotated_mask = dataset_module._augment_pair(image, mask)

    assert rotated_image.size == (3, 5)
    assert rotated_mask.size == (3, 5)
    assert np.count_nonzero(np.asarray(rotated_mask)) == 1
    assert sorted(np.asarray(rotated_image).reshape(-1).tolist()) == sorted(
        image_values.reshape(-1).tolist()
    )
