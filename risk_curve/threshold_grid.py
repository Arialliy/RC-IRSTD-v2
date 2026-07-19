"""Versioned threshold grid shared by curve labels, models, and deployment."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from .representation import (
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_REPRESENTATION,
    MAX_MODEL_GRID_POINTS,
    load_logit_grid_artifact,
    logit_threshold_grid_sha256,
    validate_logit_threshold_grid,
)


GRID_VERSION = "rc-v2-grid-v1"
CUSTOM_GRID_VERSION = "custom-grid"


def build_threshold_grid() -> np.ndarray:
    """Return the fixed, dense high-score grid used by the v2 predictor.

    Segment end points intentionally overlap.  ``np.unique`` therefore returns
    253 (not 256) strictly increasing thresholds.
    """

    return np.unique(
        np.concatenate(
            [
                np.linspace(0.00, 0.90, 64),
                np.linspace(0.90, 0.99, 64),
                np.linspace(0.99, 0.999, 64),
                np.linspace(0.999, 0.99999, 64),
            ]
        )
    ).astype(np.float32)


def validate_threshold_grid(thresholds: np.ndarray) -> np.ndarray:
    values = np.asarray(thresholds, dtype=np.float32).reshape(-1)
    if values.size < 2:
        raise ValueError("A threshold grid must contain at least two values")
    if not np.isfinite(values).all():
        raise ValueError("Threshold grid contains NaN or infinite values")
    if values[0] < 0.0 or values[-1] > 1.0:
        raise ValueError("Thresholds must lie in [0, 1]")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("Thresholds must be strictly increasing")
    return values


def threshold_grid_sha256(thresholds: np.ndarray) -> str:
    values = validate_threshold_grid(thresholds)
    canonical_bytes = np.ascontiguousarray(values, dtype="<f4").tobytes()
    return hashlib.sha256(canonical_bytes).hexdigest()


def threshold_grid_version(thresholds: np.ndarray) -> str:
    """Return the canonical version only for the exact canonical float32 grid."""

    values = validate_threshold_grid(thresholds)
    return GRID_VERSION if np.array_equal(values, build_threshold_grid()) else CUSTOM_GRID_VERSION


def load_threshold_grid(path: str | Path | None = None) -> np.ndarray:
    if path is None:
        return build_threshold_grid()
    return validate_threshold_grid(np.load(Path(path), allow_pickle=False))


def save_threshold_grid(path: str | Path, overwrite: bool = False) -> Path:
    output = Path(path)
    if output.exists() and not overwrite:
        existing = load_threshold_grid(output)
        expected = build_threshold_grid()
        if not np.array_equal(existing, expected):
            raise FileExistsError(
                f"{output} already exists with a different grid; use --force to replace it"
            )
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, build_threshold_grid())
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/threshold_grid.npy")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = save_threshold_grid(args.output, overwrite=args.force)
    grid = load_threshold_grid(path)
    print(f"saved {grid.size} thresholds ({GRID_VERSION}) to {path}")


if __name__ == "__main__":
    main()
