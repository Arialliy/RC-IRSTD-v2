from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy import ndimage

from rc_irstd.utils.io import ensure_dir


def _smooth_noise(rng: np.random.Generator, height: int, width: int, scale: int) -> np.ndarray:
    values = rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    values = ndimage.gaussian_filter(values, sigma=max(float(scale) / 2.0, 0.5))
    values -= values.mean()
    values /= values.std() + 1e-6
    return values.astype(np.float32)


def _add_gaussian_target(
    canvas: np.ndarray,
    mask: np.ndarray,
    center_y: float,
    center_x: float,
    sigma: float,
    amplitude: float,
) -> None:
    height, width = canvas.shape
    radius = max(2, int(math.ceil(3.0 * sigma)))
    y0 = max(0, int(center_y) - radius)
    y1 = min(height, int(center_y) + radius + 1)
    x0 = max(0, int(center_x) - radius)
    x1 = min(width, int(center_x) + radius + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    blob = amplitude * np.exp(
        -((yy - center_y) ** 2 + (xx - center_x) ** 2) / (2.0 * sigma**2)
    )
    canvas[y0:y1, x0:x1] += blob.astype(np.float32)
    mask[y0:y1, x0:x1] |= blob >= amplitude * 0.30


def make_synthetic_domain(
    root: str | Path,
    *,
    name: str,
    num_train: int,
    num_test: int,
    image_size: int = 64,
    seed: int = 0,
    background_scale: float = 1.0,
    stripe_strength: float = 0.0,
    target_amplitude: float = 2.8,
    noise_std: float = 0.28,
    max_targets: int = 3,
) -> Path:
    domain_root = ensure_dir(Path(root) / name)
    images_dir = ensure_dir(domain_root / "images")
    masks_dir = ensure_dir(domain_root / "masks")
    split_dir = ensure_dir(domain_root / "img_idx")
    rng = np.random.default_rng(seed)
    total = num_train + num_test
    ids: list[str] = []

    yy, xx = np.mgrid[0:image_size, 0:image_size]
    for index in range(total):
        image_id = f"{name}_{index:04d}"
        ids.append(image_id)
        low = _smooth_noise(rng, image_size, image_size, scale=8)
        medium = _smooth_noise(rng, image_size, image_size, scale=4)
        background = 0.45 + 0.11 * background_scale * low + 0.05 * medium
        if stripe_strength > 0:
            angle = rng.uniform(0, 2 * math.pi)
            phase = rng.uniform(0, 2 * math.pi)
            frequency = rng.uniform(0.08, 0.16)
            stripe = np.sin((math.cos(angle) * xx + math.sin(angle) * yy) * frequency + phase)
            background += stripe_strength * stripe.astype(np.float32)
        background += rng.normal(0.0, noise_std, size=background.shape).astype(np.float32)
        mask = np.zeros((image_size, image_size), dtype=bool)
        num_targets = int(rng.integers(0, max_targets + 1))
        for _ in range(num_targets):
            center_y = rng.uniform(5, image_size - 5)
            center_x = rng.uniform(5, image_size - 5)
            sigma = rng.uniform(0.8, 1.8)
            amplitude = target_amplitude * rng.uniform(0.75, 1.15)
            _add_gaussian_target(background, mask, center_y, center_x, sigma, amplitude)
        normalized = background - np.percentile(background, 1)
        normalized /= np.percentile(normalized, 99.8) + 1e-6
        normalized = np.clip(normalized, 0.0, 1.0)
        rgb = np.stack(
            [normalized, np.clip(normalized * 0.96 + 0.01, 0, 1), normalized], axis=-1
        )
        Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB").save(
            images_dir / f"{image_id}.png"
        )
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(
            masks_dir / f"{image_id}.png"
        )

    train_ids = ids[:num_train]
    test_ids = ids[num_train:]
    (split_dir / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (split_dir / "test.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")
    return domain_root


def make_synthetic_collection(
    output_root: str | Path,
    specs: Iterable[dict[str, object]],
) -> list[Path]:
    roots: list[Path] = []
    for spec in specs:
        roots.append(make_synthetic_domain(output_root, **spec))
    return roots
