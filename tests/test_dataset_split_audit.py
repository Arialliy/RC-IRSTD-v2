from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from scripts.audit_dataset_splits import main


def _write_raster(path: Path, *, color: int, size: tuple[int, int] = (3, 2)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "L" if path.parent.name == "masks" else "RGB"
    value = color if mode == "L" else (color, color, color)
    Image.new(mode, size, value).save(path)


def _make_dataset(
    root: Path,
    *,
    train_ids: list[str],
    test_ids: list[str],
    colors: dict[str, tuple[int, int]],
) -> Path:
    index = root / "img_idx"
    index.mkdir(parents=True)
    (index / "train_local.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in train_ids), encoding="utf-8"
    )
    (index / "test_local.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in test_ids), encoding="utf-8"
    )
    for sample_id, (image_color, mask_color) in colors.items():
        _write_raster(root / "images" / f"{sample_id}.png", color=image_color)
        _write_raster(root / "masks" / f"{sample_id}.png", color=mask_color)
    return root


def test_clean_multi_dataset_audit_writes_complete_atomic_json(tmp_path) -> None:
    first = _make_dataset(
        tmp_path / "first",
        train_ids=["train-a"],
        test_ids=["test-a"],
        colors={"train-a": (10, 0), "test-a": (20, 255)},
    )
    second = _make_dataset(
        tmp_path / "second",
        train_ids=["train-b"],
        test_ids=["test-b"],
        colors={"train-b": (30, 64), "test-b": (40, 128)},
    )
    output = tmp_path / "reports" / "audit.json"

    return_code = main(
        [
            "--dataset-dir",
            str(first),
            "--dataset-dir",
            str(second),
            "--output",
            str(output),
        ]
    )

    assert return_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["audit_mode"] == "read_only"
    assert report["dataset_count"] == 2
    assert report["strict_pass"] is True
    assert report["issue_count"] == 0
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))

    audited = report["datasets"][0]
    expected_order_hash = hashlib.sha256(b"train-a\n").hexdigest()
    assert audited["splits"]["train"]["ordered_ids_sha256"] == expected_order_hash
    assert audited["splits"]["train"]["count"] == 1
    assert len(audited["splits"]["train"]["file_sha256"]) == 64
    sample = audited["samples"]["train"][0]
    assert sample["image"]["status"] == "unique"
    assert sample["mask"]["status"] == "unique"
    assert sample["shape_match"] is True
    assert len(sample["image"]["content_sha256"]) == 64
    assert len(sample["mask"]["content_sha256"]) == 64
    assert len(sample["pair_content_sha256"]) == 64


def test_issues_fail_strict_mode_and_allow_flag_is_explicitly_marked(tmp_path) -> None:
    dataset = _make_dataset(
        tmp_path / "dataset",
        train_ids=["train-copy"],
        test_ids=["test-copy", "wrong-shape"],
        colors={
            "train-copy": (50, 255),
            "test-copy": (50, 255),
            "wrong-shape": (90, 0),
        },
    )
    _write_raster(
        dataset / "masks" / "wrong-shape.png", color=0, size=(2, 2)
    )
    strict_output = tmp_path / "strict.json"

    assert main(
        ["--dataset-dir", str(dataset), "--output", str(strict_output)]
    ) == 1
    strict = json.loads(strict_output.read_text(encoding="utf-8"))
    issue_codes = {issue["code"] for issue in strict["datasets"][0]["issues"]}
    assert strict["strict_pass"] is False
    assert strict["status"] == "fail"
    assert "image_mask_shape_mismatch" in issue_codes
    assert "train_test_exact_image_content_duplicate" in issue_codes
    assert "train_test_exact_mask_content_duplicate" not in issue_codes
    assert len(
        strict["datasets"][0]["train_test_exact_content_duplicates"]["mask"]
    ) == 1
    assert "train_test_exact_image_mask_pair_content_duplicate" in issue_codes

    allowed_output = tmp_path / "allowed.json"
    assert main(
        [
            "--dataset-dir",
            str(dataset),
            "--output",
            str(allowed_output),
            "--allow-known-issues",
        ]
    ) == 0
    allowed = json.loads(allowed_output.read_text(encoding="utf-8"))
    assert allowed["allow_known_issues"] is True
    assert allowed["strict_pass"] is False
    assert allowed["status"] == "issues_allowed_for_diagnostics"
    assert allowed["issue_count"] == strict["issue_count"]


def test_ambiguous_raster_resolution_is_reported_without_guessing(tmp_path) -> None:
    dataset = _make_dataset(
        tmp_path / "dataset",
        train_ids=["ambiguous"],
        test_ids=["clean"],
        colors={"ambiguous": (25, 0), "clean": (75, 255)},
    )
    _write_raster(dataset / "images" / "ambiguous.jpg", color=25)
    output = tmp_path / "audit.json"

    assert main(["--dataset-dir", str(dataset), "--output", str(output)]) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    sample = report["datasets"][0]["samples"]["train"][0]
    assert sample["image"]["status"] == "ambiguous"
    assert sample["image"]["candidate_count"] == 2
    assert "content_sha256" not in sample["image"]
    assert any(
        issue["code"] == "image_raster_ambiguous"
        for issue in report["datasets"][0]["issues"]
    )


def test_same_aspect_resolution_mismatch_is_audited_as_guarded_alignment(
    tmp_path,
) -> None:
    dataset = _make_dataset(
        tmp_path / "dataset",
        train_ids=["train"],
        test_ids=["Misc_111"],
        colors={"train": (10, 0), "Misc_111": (20, 255)},
    )
    # The image is 3x2; a 6x4 mask differs in resolution but has the same
    # coordinate aspect ratio and is therefore eligible for guarded NN resize.
    _write_raster(dataset / "masks" / "Misc_111.png", color=255, size=(6, 4))
    output = tmp_path / "audit.json"

    assert main(["--dataset-dir", str(dataset), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    audited = report["datasets"][0]
    sample = audited["samples"]["test"][0]
    assert report["schema_version"] == 2
    assert report["strict_pass"] is True
    assert report["guarded_alignment_count"] == 1
    assert audited["issue_count"] == 0
    assert audited["diagnostic_count"] == 1
    assert sample["shape_match"] is False
    assert sample["mask_alignment"]["required"] is True
    assert sample["mask_alignment"]["eligible"] is True
    assert sample["mask_alignment"]["relative_aspect_error"] == 0.0
    assert audited["diagnostics"][0]["code"] == (
        "image_mask_guarded_alignment_required"
    )
