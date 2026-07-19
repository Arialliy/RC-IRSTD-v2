"""Utilities for resolving this repository's frozen train/test split files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _validate_dataset_root(dataset_dir: str | Path) -> Path:
    root = Path(dataset_dir).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {root}")
    return root.resolve()


def resolve_split_file(
    dataset_dir: str | Path,
    split_file: str | Path | None = None,
    *,
    split: str = "test",
) -> Path:
    """Resolve an explicit split file or discover an unambiguous default.

    Relative explicit paths are first interpreted relative to ``dataset_dir`` and
    then relative to ``dataset_dir/img_idx``.  Automatic discovery deliberately
    raises when more than one glob match exists so callers cannot silently use the
    wrong protocol.
    """

    root = _validate_dataset_root(dataset_dir)
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")
    # The local benchmark distribution has only frozen train/test manifests.
    # ``val`` is a compatibility spelling for the exact test manifest; callers
    # must keep it evaluation-only and may not use it for model selection.
    if split == "val":
        split = "test"

    if split_file is not None:
        requested = Path(split_file).expanduser()
        candidates = [requested] if requested.is_absolute() else [
            root / requested,
            root / "img_idx" / requested,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        rendered = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Explicit split file was not found. Checked: {rendered}"
        )

    if split == "train":
        preferred = [root / "trainval.txt", root / "train.txt"]
        pattern = "train*.txt"
    else:
        preferred = [root / f"{split}.txt"]
        pattern = f"{split}*.txt"

    # Automatic discovery is permitted only when the repository declares one
    # unambiguous protocol.  Silently preferring ``trainval.txt`` over
    # ``train.txt`` (or either root file over an ``img_idx`` manifest) makes a
    # run depend on incidental filenames and can change the paper split without
    # changing the command line.
    discovered = [candidate.resolve() for candidate in preferred if candidate.is_file()]
    discovered.extend(
        candidate.resolve() for candidate in sorted((root / "img_idx").glob(pattern))
    )
    discovered = list(dict.fromkeys(discovered))
    if len(discovered) == 1:
        return discovered[0]
    if not discovered:
        raise FileNotFoundError(
            f"No {split!r} split found under {root}; pass split_file explicitly"
        )
    raise ValueError(
        f"Multiple {split!r} splits found under {root}: "
        f"{', '.join(str(path.relative_to(root)) for path in discovered)}. "
        "Pass split_file explicitly."
    )


def read_split_file(
    split_file: str | Path,
    *,
    reject_duplicates: bool = True,
) -> list[str]:
    """Read non-empty sample entries while preserving their declared order."""

    path = Path(split_file).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Split file does not exist: {path}")
    entries = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not entries:
        raise ValueError(f"Split file is empty: {path}")
    if reject_duplicates:
        seen: set[str] = set()
        duplicates: list[str] = []
        for entry in entries:
            if entry in seen and entry not in duplicates:
                duplicates.append(entry)
            seen.add(entry)
        if duplicates:
            raise ValueError(
                f"Split file contains duplicate entries: {', '.join(duplicates[:5])}"
            )
    return entries


def sample_id_from_entry(entry: str) -> str:
    """Return a safe logical image id from one split-file entry."""

    if not isinstance(entry, str) or not entry.strip():
        raise ValueError("split entry must be a non-empty string")
    normalized = entry.strip().replace("\\", "/")
    path = Path(normalized)
    if any(part == ".." for part in path.parts):
        raise ValueError(f"Parent traversal is not allowed in split entries: {entry}")
    image_id = path.stem
    if image_id.endswith("_pixels0"):
        image_id = image_id[: -len("_pixels0")]
    if not image_id:
        raise ValueError(f"Could not derive an image id from split entry: {entry}")
    return image_id


def ensure_unique_sample_ids(entries: Iterable[str]) -> list[str]:
    """Convert entries to ids and reject filename collisions."""

    image_ids = [sample_id_from_entry(entry) for entry in entries]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Split entries map to duplicate image ids")
    return image_ids
