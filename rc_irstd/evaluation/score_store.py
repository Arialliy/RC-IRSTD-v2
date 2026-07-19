from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from evaluation.artifact_integrity import (
    SCORE_MANIFEST_SCHEMA_VERSION,
    verify_score_map_directory,
)
from rc_irstd.utils.io import atomic_write_json, read_json, save_npz_atomic


@dataclass
class ScoreItem:
    probability: np.ndarray
    mask: np.ndarray
    gray: np.ndarray
    image_id: str
    dataset_name: str
    sequence_id: str
    original_hw: tuple[int, int]
    has_mask: bool
    path: Path | None = None


def _legacy_record_path(
    root: Path,
    filename: object,
    *,
    require_file: bool,
) -> Path:
    """Resolve one legacy NPZ name without allowing directory traversal."""

    if not isinstance(filename, str) or not filename:
        raise ValueError("Legacy score-map filename must be a non-empty string")
    relative = Path(filename)
    if (
        relative.is_absolute()
        or len(relative.parts) != 1
        or relative.name != filename
        or relative.suffix.lower() != ".npz"
    ):
        raise ValueError(f"Unsafe legacy score-map filename: {filename!r}")
    path = root / relative
    if require_file:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.resolve().parent != root:
            raise ValueError(f"Legacy score-map file escapes its artifact directory: {path}")
    return path


def save_score_item(directory: str | Path, item: ScoreItem) -> Path:
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not isinstance(item.image_id, str) or not item.image_id:
        raise ValueError("ScoreItem.image_id must be a non-empty string")
    path = _legacy_record_path(
        root,
        f"{item.image_id}.npz",
        require_file=False,
    )
    save_npz_atomic(
        path,
        probability=item.probability.astype(np.float32),
        mask=item.mask.astype(np.uint8),
        gray=item.gray.astype(np.float32),
        image_id=np.asarray(item.image_id),
        dataset_name=np.asarray(item.dataset_name),
        sequence_id=np.asarray(item.sequence_id),
        original_hw=np.asarray(item.original_hw, dtype=np.int32),
        has_mask=np.asarray(int(item.has_mask), dtype=np.uint8),
    )
    return path


def load_score_item(path: str | Path) -> ScoreItem:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        probability_key = "prob" if "prob" in payload else "probability"
        if probability_key not in payload:
            raise ValueError(f"Score map lacks prob/probability array: {source}")
        probability = np.asarray(payload[probability_key], dtype=np.float32)
        mask = (
            np.asarray(payload["mask"], dtype=np.uint8)
            if "mask" in payload
            else np.zeros(probability.shape, dtype=np.uint8)
        )
        gray = np.asarray(payload["gray"], dtype=np.float32)
        image_id = str(payload["image_id"].item())
        dataset_name = (
            str(payload["dataset_name"].item())
            if "dataset_name" in payload
            else source.parent.name
        )
        sequence_id = (
            str(payload["sequence_id"].item())
            if "sequence_id" in payload
            else (image_id.rsplit("_", 1)[0] if "_" in image_id else dataset_name)
        )
        original_hw_array = np.asarray(payload["original_hw"], dtype=np.int32).tolist()
        if "labels_loaded" in payload:
            has_mask = bool(np.asarray(payload["labels_loaded"]).item())
        elif "has_mask" in payload:
            has_mask = bool(int(np.asarray(payload["has_mask"]).item()))
        else:
            has_mask = "mask" in payload
    if probability.shape != mask.shape or probability.shape != gray.shape:
        raise ValueError(f"Shape mismatch in {source}")
    if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise ValueError(f"Invalid probability values in {source}")
    return ScoreItem(
        probability=probability,
        mask=mask,
        gray=gray,
        image_id=image_id,
        dataset_name=dataset_name,
        sequence_id=sequence_id,
        original_hw=(int(original_hw_array[0]), int(original_hw_array[1])),
        has_mask=has_mask,
        path=source,
    )


class ScoreStore(Sequence[ScoreItem]):
    def __init__(self, directory: str | Path) -> None:
        self.root = Path(directory).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        self.manifest: dict[str, Any] = read_json(manifest_path)
        records = self.manifest.get("records")
        if (
            self.manifest.get("schema_version") == SCORE_MANIFEST_SCHEMA_VERSION
            or isinstance(records, list)
        ):
            verified_manifest, paths, integrity = verify_score_map_directory(
                self.root,
                require_integrity=True,
            )
            if verified_manifest is None or not integrity.get("verified"):
                raise ValueError(f"Version-3 score store failed verification: {self.root}")
            self.manifest = verified_manifest
            self.paths = paths
            self._entry_metadata = list(self.manifest["records"])
            self.integrity = integrity
        else:
            entries = self.manifest.get("entries")
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"Manifest has no entries: {manifest_path}")
            if any(not isinstance(entry, dict) or "file" not in entry for entry in entries):
                raise ValueError("Every legacy score-store entry must contain a file")
            self.paths = [
                _legacy_record_path(
                    self.root,
                    entry["file"],
                    require_file=True,
                )
                for entry in entries
            ]
            if len({path.name for path in self.paths}) != len(self.paths):
                raise ValueError("Legacy score-store manifest contains duplicate filenames")
            self._entry_metadata = entries
            self.integrity = {
                "verified": False,
                "diagnostic_reason": "legacy_v1_score_store",
            }

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int | slice) -> ScoreItem | list[ScoreItem]:
        if isinstance(index, slice):
            return [load_score_item(path) for path in self.paths[index]]
        return load_score_item(self.paths[index])

    def __iter__(self) -> Iterator[ScoreItem]:
        for path in self.paths:
            yield load_score_item(path)

    @property
    def dataset_name(self) -> str:
        return str(
            self.manifest.get(
                "target_dataset",
                self.manifest.get("dataset_name", self.root.name),
            )
        )


def write_manifest(
    directory: str | Path,
    *,
    dataset_name: str,
    checkpoint: str,
    split: str,
    entries: list[dict[str, Any]],
    model_config: dict[str, Any],
    preserve_aspect: bool,
) -> None:
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(not isinstance(entry, dict) or "file" not in entry for entry in entries):
        raise ValueError("Every legacy score-store entry must contain a file")
    paths = [
        _legacy_record_path(root, entry["file"], require_file=True)
        for entry in entries
    ]
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("Legacy score-store manifest contains duplicate filenames")
    atomic_write_json(
        root / "manifest.json",
        {
            "format_version": 1,
            "dataset_name": dataset_name,
            "checkpoint": checkpoint,
            "split": split,
            "score_type": "sigmoid_probability",
            "preserve_aspect": preserve_aspect,
            "model_config": model_config,
            "num_images": len(entries),
            "entries": entries,
        },
    )
