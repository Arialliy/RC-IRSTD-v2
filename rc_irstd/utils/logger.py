from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from .io import ensure_dir


def build_logger(name: str, output_dir: str | Path) -> logging.Logger:
    directory = ensure_dir(output_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(directory / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def append_csv(path: str | Path, row: dict[str, Any]) -> None:
    """Append a row while preserving a valid union schema across epochs.

    Detector validation columns can appear only on validation epochs. A naïve
    append would then produce rows wider than the original header. When new
    fields appear, this function atomically rewrites the small history file with
    the union of old and new columns; otherwise it performs a normal append.
    """

    target = Path(path)
    ensure_dir(target.parent)
    if not target.exists():
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with target.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        existing_fields = list(reader.fieldnames or [])
        missing_fields = [key for key in row if key not in existing_fields]
        if not missing_fields:
            existing_rows: list[dict[str, Any]] | None = None
        else:
            existing_rows = list(reader)

    if existing_rows is None:
        with target.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=existing_fields)
            writer.writerow(row)
        return

    union_fields = existing_fields + missing_fields
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=union_fields)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow(row)
    temporary.replace(target)
