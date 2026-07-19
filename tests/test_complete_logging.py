from __future__ import annotations

import csv
from pathlib import Path

from rc_irstd.utils.logger import append_csv


def test_append_csv_expands_schema_without_corrupting_history(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    append_csv(path, {"epoch": 0, "loss": 1.0})
    append_csv(path, {"epoch": 1, "loss": 0.8, "val_loss": 0.9})
    append_csv(path, {"epoch": 2, "loss": 0.7})

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == ["epoch", "loss", "val_loss"]
    assert len(rows) == 3
    assert rows[0]["val_loss"] == ""
    assert rows[1]["val_loss"] == "0.9"
    assert rows[2]["val_loss"] == ""
