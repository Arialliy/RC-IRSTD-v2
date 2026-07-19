from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data_ext.dataset_meta import crop_to_valid
from data_ext.eval_dataset import IRSTDEvalDataset
from data_ext.mask_alignment import (
    MASK_ALIGNMENT_POLICY,
    align_mask_to_image,
)
from data_ext.split_utils import resolve_split_file
from evaluation.budget_metrics import (
    compute_budget_excess,
    summarize_budget_results,
)
from evaluation.component_matching import connected_components, match_components
from evaluation.artifact_integrity import verify_score_map_directory
from evaluation.export_score_maps import export_dataset_score_maps
from evaluation.operating_point import main as operating_point_main
from evaluation.operating_point import select_operating_point
from evaluation.threshold_sweep import (
    EMPTY_SET_THRESHOLD,
    build_default_thresholds,
    load_score_map_directory,
    pixel_fa_is_monotone,
    read_curve_csv,
    sweep_thresholds,
    write_curve_csv,
)
from utils.metric import FixedThresholdMetrics


def _create_dataset(root: Path) -> tuple[Path, Path]:
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    (root / "protocols").mkdir()
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    image[..., 0] = np.arange(7, dtype=np.uint8)[None, :] * 20
    image[..., 1] = 80
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[2, 3] = 255
    Image.fromarray(image).save(root / "images" / "Misc_1.png")
    # Exercise the supported optional NUAA-style mask suffix.
    Image.fromarray(mask).save(root / "masks" / "Misc_1_pixels0.png")
    split_file = root / "protocols" / "held_out.txt"
    split_file.write_text("Misc_1\n", encoding="utf-8")
    return root, split_file


def test_explicit_split_resize_native_padding_and_collatable_meta(tmp_path: Path) -> None:
    root, split_file = _create_dataset(tmp_path / "NUAA-SIRST")
    assert resolve_split_file(root, "protocols/held_out.txt") == split_file.resolve()

    resized = IRSTDEvalDataset(
        root,
        split_file=split_file,
        spatial_mode="resize",
        base_size=(6, 10),
    )
    sample = resized[0]
    assert sample["image"].shape == (3, 6, 10)
    assert sample["gray"].shape == (1, 6, 10)
    assert 0.0 <= float(sample["gray"].min()) <= float(sample["gray"].max()) <= 1.0
    assert sample["mask"].shape == (1, 6, 10)
    assert set(torch.unique(sample["mask"]).tolist()).issubset({0.0, 1.0})
    batch = next(iter(DataLoader(resized, batch_size=1)))
    assert isinstance(batch["meta"], dict)
    assert batch["meta"]["original_hw"].shape == (1, 2)
    assert batch["meta"]["image_id"] == ["Misc_1"]

    native = IRSTDEvalDataset(
        root,
        split_file="protocols/held_out.txt",
        spatial_mode="native",
        pad_multiple=4,
    )
    native_sample = native[0]
    assert native_sample["image"].shape == (3, 8, 8)
    assert native_sample["gray"].shape == (1, 8, 8)
    assert native_sample["mask"].shape == (1, 8, 8)
    assert native_sample["meta"]["padding_ltrb"].tolist() == [0, 0, 1, 3]
    cropped = crop_to_valid(native_sample["mask"], native_sample["meta"])
    cropped_gray = crop_to_valid(native_sample["gray"], native_sample["meta"])
    assert cropped.shape == (1, 5, 7)
    assert cropped_gray.shape == (1, 5, 7)
    assert int(cropped.sum()) == 1


def test_misc_111_guarded_nearest_neighbor_alignment_is_audited(
    tmp_path: Path,
) -> None:
    root = tmp_path / "NUAA-SIRST"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    (root / "img_idx").mkdir()
    image = np.zeros((220, 325, 3), dtype=np.uint8)
    mask = np.zeros((400, 592), dtype=np.uint8)
    mask[206:217, 260:274] = 255
    Image.fromarray(image).save(root / "images" / "Misc_111.png")
    Image.fromarray(mask).save(root / "masks" / "Misc_111.png")
    split_file = root / "img_idx" / "test_NUAA-SIRST.txt"
    split_file.write_text("Misc_111\n", encoding="utf-8")

    dataset = IRSTDEvalDataset(
        root,
        split_file=split_file,
        spatial_mode="native",
        pad_multiple=16,
    )
    sample = dataset[0]
    meta = sample["meta"]
    assert meta["original_hw"].tolist() == [220, 325]
    assert meta["mask_original_hw"].tolist() == [400, 592]
    assert meta["mask_alignment_applied"] is True
    assert meta["mask_alignment_policy"] == MASK_ALIGNMENT_POLICY
    assert meta["mask_aspect_relative_error"] == pytest.approx(
        0.0018461538461538327
    )
    cropped = crop_to_valid(sample["mask"], meta)
    assert cropped.shape == (1, 220, 325)
    assert set(torch.unique(cropped).tolist()).issubset({0.0, 1.0})
    assert int(cropped.sum()) > 0

    output_dir = tmp_path / "aligned-scores"
    manifest = export_dataset_score_maps(
        _ZeroLogitModel(), dataset, output_dir, labels_loaded=True
    )
    assert manifest["mask_alignment_count"] == 1
    assert manifest["mask_aligned_sample_ids"] == ["Misc_111"]
    assert manifest["records"][0]["mask_original_hw"] == [400, 592]
    _, _, audit = verify_score_map_directory(
        output_dir, require_integrity=True, require_masks=True
    )
    assert audit["mask_alignment_verified"] is True

    boundary_mask, boundary_meta = align_mask_to_image(
        Image.new("L", (1009, 1000)),
        Image.new("RGB", (1000, 1000)),
        "below-one-percent-boundary",
    )
    assert boundary_mask.size == (1000, 1000)
    assert boundary_meta.applied is True
    assert boundary_meta.relative_aspect_error == pytest.approx(0.009)
    unchanged, unchanged_meta = align_mask_to_image(
        Image.new("L", (100, 100)),
        Image.new("RGB", (100, 100)),
        "same-size",
    )
    assert unchanged.size == (100, 100)
    assert unchanged_meta.applied is False

    with pytest.raises(ValueError, match="aspect-ratio mismatch"):
        align_mask_to_image(
            Image.new("L", (102, 100)),
            Image.new("RGB", (100, 100)),
            "bad-pair",
        )
    with pytest.raises(ValueError, match="finite"):
        align_mask_to_image(
            Image.new("L", (10, 5)),
            Image.new("RGB", (10, 5)),
            "bad-tolerance",
            aspect_tolerance=float("nan"),
        )


def test_ambiguous_auto_split_requires_explicit_protocol(tmp_path: Path) -> None:
    root, _ = _create_dataset(tmp_path / "ambiguous")
    (root / "img_idx").mkdir()
    (root / "img_idx" / "test_a.txt").write_text("Misc_1\n", encoding="utf-8")
    (root / "img_idx" / "test_b.txt").write_text("Misc_1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Multiple"):
        resolve_split_file(root, split="test")


def test_fixed_threshold_metrics_counts_equal_area_false_component_by_identity():
    probabilities = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
    labels = torch.zeros_like(probabilities)
    probabilities[0, 0, 1, 1] = 0.9
    probabilities[0, 0, 4, 4] = 0.9
    labels[0, 0, 1, 1] = 1.0
    evaluator = FixedThresholdMetrics(
        threshold=0.5,
        matching_rule="overlap",
        connectivity=2,
    )
    evaluator.update(probabilities, labels)
    metrics = evaluator.get()
    assert metrics["tp_objects"] == 1
    assert metrics["gt_objects"] == 1
    assert metrics["fp_components"] == 1
    assert metrics["fp_pixels"] == 1
    assert metrics["Pd"] == 1.0


def test_operating_point_cli_requires_oracle_acknowledgement(tmp_path: Path) -> None:
    curve = tmp_path / "curve.csv"
    write_curve_csv(
        [
            {
                "threshold": 0.5,
                "pd": 1.0,
                "fa_pixel": 0.0,
                "fa_component_mp": 0.0,
                "tp_objects": 1,
                "gt_objects": 1,
                "fp_components": 0,
                "fp_pixels": 0,
                "total_pixels": 16,
            }
        ],
        curve,
    )
    output = tmp_path / "selection.json"
    with pytest.raises(ValueError, match="oracle-diagnostic"):
        operating_point_main(["--curve", str(curve), "--output", str(output)])
    assert operating_point_main(
        [
            "--curve",
            str(curve),
            "--output",
            str(output),
            "--oracle-diagnostic",
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["oracle_only"] is True
    assert payload["formal_protocol_eligible"] is False


def test_overlap_centroid_matching_and_connectivity() -> None:
    ground_truth = np.zeros((9, 9), dtype=np.uint8)
    ground_truth[1:3, 1:3] = 1
    ground_truth[6:8, 6:8] = 1
    prediction = np.zeros_like(ground_truth)
    prediction[1:3, 1:3] = 1
    prediction[0, 8] = 1
    result = match_components(prediction, ground_truth, rule="overlap")
    assert result.num_gt == 2
    assert result.num_tp_objects == 1
    assert result.num_fp_components == 1
    assert result.num_fp_pixels == 1
    assert len(result.matched_pairs) == 1

    centroid_gt = np.zeros((7, 7), dtype=np.uint8)
    centroid_prediction = np.zeros_like(centroid_gt)
    centroid_gt[2, 2] = 1
    centroid_prediction[2, 4] = 1
    assert match_components(
        centroid_prediction,
        centroid_gt,
        rule="centroid",
        centroid_distance=2.0,
    ).num_tp_objects == 1
    assert match_components(
        centroid_prediction,
        centroid_gt,
        rule="overlap",
    ).num_tp_objects == 0

    diagonal = np.eye(2, dtype=np.uint8)
    assert connected_components(diagonal, connectivity=1)[1] == 2
    assert connected_components(diagonal, connectivity=2)[1] == 1
    with pytest.raises(ValueError, match="binary"):
        match_components(np.full((2, 2), 0.5), np.zeros((2, 2)))


def test_threshold_sweep_csv_operating_point_and_budget_metrics(tmp_path: Path) -> None:
    default_thresholds = build_default_thresholds()
    assert default_thresholds[-2] == 1.0
    assert default_thresholds[-1] == EMPTY_SET_THRESHOLD
    assert EMPTY_SET_THRESHOLD > 1.0
    probability = np.zeros((3, 3), dtype=np.float32)
    probability[1, 1] = 0.9
    probability[0, 2] = 0.8
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 1
    rows = sweep_thresholds(
        [probability],
        [mask],
        [0.0, 0.5, 0.85, 1.0],
    )
    assert [row["threshold"] for row in rows] == [0.0, 0.5, 0.85, 1.0]
    assert pixel_fa_is_monotone(rows)
    assert rows[0]["fp_pixels"] == 8
    assert rows[2]["pd"] == 1.0
    assert rows[2]["fp_pixels"] == 0
    assert rows[-1]["tp_objects"] == 0

    saturated = np.ones((1, 1), dtype=np.float32)
    empty_row = sweep_thresholds(
        [saturated], [np.zeros_like(saturated)], [EMPTY_SET_THRESHOLD]
    )[0]
    assert empty_row["fp_pixels"] == 0
    assert empty_row["fp_components"] == 0

    # The operating-point CLI must directly consume the default sweep's
    # evaluation-only endpoint.  With a saturated score, only that endpoint is
    # feasible under a near-zero false-alarm budget.
    saturated_rows = sweep_thresholds(
        [saturated],
        [np.zeros_like(saturated)],
        [1.0, EMPTY_SET_THRESHOLD],
    )
    saturated_curve = write_curve_csv(saturated_rows, tmp_path / "saturated.csv")
    selected_empty = select_operating_point(
        read_curve_csv(saturated_curve),
        pixel_budget=1e-12,
        strategy="max_pd",
    )
    assert selected_empty is not None
    assert selected_empty["threshold"] == EMPTY_SET_THRESHOLD

    with pytest.raises(ValueError, match="empty-set sentinel"):
        select_operating_point(
            [{"threshold": 1.01, "pd": 0.0}],
            pixel_budget=1e-12,
        )

    curve_path = write_curve_csv(rows, tmp_path / "curve.csv")
    loaded = read_curve_csv(curve_path)
    assert loaded == rows
    selected = select_operating_point(
        rows,
        pixel_budget=0.01,
        strategy="max_pd",
    )
    assert selected is not None
    assert selected["threshold"] == pytest.approx(0.85)
    assert select_operating_point(
        [
            {
                "threshold": 0.5,
                "pd": 1.0,
                "fa_pixel": 0.1,
                "fa_component_mp": 2.0,
            }
        ],
        pixel_budget=0.01,
        component_budget=1.0,
        strategy="max_pd",
    ) is None

    deployment_rows = [
        {"fa_pixel": 1e-6, "fa_component_mp": 1.0},
        {"fa_pixel": 3e-6, "fa_component_mp": 0.5},
    ]
    summary = summarize_budget_results(
        deployment_rows,
        pixel_budget=1e-6,
        component_budget=1.0,
    )
    assert summary["bsr"] == pytest.approx(0.5)
    assert summary["max_relative_excess"] == pytest.approx(2.0)
    excess = compute_budget_excess(deployment_rows[1], pixel_budget=1e-6)
    assert excess["pixel_excess"] == pytest.approx(2e-6)


class _ZeroLogitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_warm_flag: bool | None = None

    def forward(self, images: torch.Tensor, warm_flag: bool) -> torch.Tensor:
        self.seen_warm_flag = warm_flag
        return torch.zeros(
            (images.shape[0], 1, images.shape[-2], images.shape[-1]),
            dtype=images.dtype,
            device=images.device,
        )


def test_score_export_is_sigmoid_continuous_cropped_and_manifested(tmp_path: Path) -> None:
    root, split_file = _create_dataset(tmp_path / "NUAA-SIRST")
    dataset = IRSTDEvalDataset(
        root,
        split_file=split_file,
        spatial_mode="native",
        pad_multiple=4,
    )
    model = _ZeroLogitModel()
    output_dir = tmp_path / "scores"
    manifest = export_dataset_score_maps(
        model,
        dataset,
        output_dir,
        manifest_metadata={"source_dataset": "synthetic-source", "weight_path": "toy"},
    )
    assert manifest["score_type"] == "sigmoid_probability"
    assert manifest["warm_flag"] is True
    assert model.seen_warm_flag is True
    assert manifest["num_images"] == 1
    assert manifest["target_dataset"] == "NUAA-SIRST"
    assert manifest["source_dataset"] == "synthetic-source"
    assert manifest["labels_loaded"] is True
    assert manifest["mask_alignment_policy"] == MASK_ALIGNMENT_POLICY
    assert manifest["mask_alignment_count"] == 0
    assert manifest["mask_aligned_sample_ids"] == []
    assert len(manifest["records"][0]["sha256"]) == 64
    assert len(manifest["records_sha256"]) == 64
    assert len(manifest["ordered_image_ids_sha256"]) == 64
    assert len(manifest["split_file_sha256"]) == 64
    disk_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    record_path = output_dir / disk_manifest["records"][0]["file"]
    with np.load(record_path, allow_pickle=False) as payload:
        assert payload["prob"].shape == (5, 7)
        assert payload["mask"].shape == (5, 7)
        assert payload["gray"].shape == (5, 7)
        assert payload["prob"].dtype == np.float32
        assert np.all(payload["prob"] == np.float32(0.5))
        assert set(np.unique(payload["mask"])).issubset({0, 1})
        assert np.isfinite(payload["gray"]).all()
        assert 0.0 <= float(payload["gray"].min()) <= float(payload["gray"].max()) <= 1.0
        assert payload["original_hw"].tolist() == [5, 7]
        assert payload["input_hw"].tolist() == [8, 8]
        assert bool(payload["mask_alignment_applied"].item()) is False
        assert payload["mask_original_hw"].tolist() == [5, 7]
        assert float(payload["mask_aspect_relative_error"].item()) == 0.0
        assert str(payload["mask_alignment_policy"].item()) == MASK_ALIGNMENT_POLICY
    _, _, integrity = verify_score_map_directory(
        output_dir, require_integrity=True, require_masks=True
    )
    assert integrity["mask_alignment_verified"] is True
    probabilities, masks = load_score_map_directory(output_dir)
    assert probabilities[0].shape == masks[0].shape == (5, 7)
    with pytest.raises(FileExistsError):
        export_dataset_score_maps(model, dataset, output_dir)
    tampered_manifest = dict(disk_manifest)
    tampered_manifest["mask_alignment_count"] = 1
    (output_dir / "manifest.json").write_text(
        json.dumps(tampered_manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="mask_alignment_count mismatch"):
        verify_score_map_directory(output_dir, require_integrity=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(disk_manifest), encoding="utf-8"
    )
    record_path.write_bytes(record_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_score_map_directory(output_dir)


def test_zero_label_export_never_requires_or_embeds_masks(tmp_path: Path) -> None:
    root, split_file = _create_dataset(tmp_path / "unlabelled-target")
    for mask_path in (root / "masks").iterdir():
        mask_path.unlink()
    (root / "masks").rmdir()
    dataset = IRSTDEvalDataset(
        root,
        split_file=split_file,
        spatial_mode="native",
        pad_multiple=4,
        load_masks=False,
    )
    sample = dataset[0]
    assert "mask" not in sample
    assert sample["meta"]["mask_path"] == ""

    output_dir = tmp_path / "mask-free-scores"
    manifest = export_dataset_score_maps(
        _ZeroLogitModel(),
        dataset,
        output_dir,
        labels_loaded=False,
        manifest_metadata={
            "source_datasets": ["source-a", "source-b"],
            "weight_sha256": "a" * 64,
        },
    )
    assert manifest["labels_loaded"] is False
    assert manifest["mask_alignment_policy"] == "labels_not_loaded"
    assert manifest["mask_alignment_count"] == 0
    record_path = output_dir / manifest["records"][0]["file"]
    with np.load(record_path, allow_pickle=False) as payload:
        assert "mask" not in payload
        assert bool(payload["labels_loaded"].item()) is False
        assert bool(payload["mask_alignment_applied"].item()) is False
        assert payload["mask_original_hw"].tolist() == [0, 0]
        assert float(payload["mask_aspect_relative_error"].item()) == -1.0
    _, _, audit = verify_score_map_directory(
        output_dir, require_integrity=True, require_masks=False
    )
    assert audit["verified"] is True
    with pytest.raises(ValueError, match="label mode"):
        load_score_map_directory(output_dir, require_integrity=True)
