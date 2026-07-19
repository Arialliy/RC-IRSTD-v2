from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

import data_ext.eval_dataset as eval_dataset_module
import data_ext.mask_alignment as alignment_module
import evaluation.export_score_maps as export_module
import rc_irstd.cli.export_scores as export_cli_module
import rc_irstd.data.dataset as complete_dataset_module
import utils.data as legacy_dataset_module
from data_ext.dataset_meta import crop_to_valid
from data_ext.eval_dataset import IRSTDEvalDataset
from data_ext.mask_alignment import (
    MASK_ALIGNMENT_ASPECT_TOLERANCE,
    align_mask_to_image,
    aspect_error_within_tolerance,
)
from rc_irstd.data.dataset import IRSTDDataset
from utils.data import IRSTD_Dataset


def _write_misc_111_fixture(root) -> tuple[Image.Image, Image.Image]:
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    (root / "img_idx").mkdir()

    image_array = np.zeros((220, 325, 3), dtype=np.uint8)
    image_array[..., 0] = 31
    image_array[..., 1] = 63
    image_array[..., 2] = 127
    mask_array = np.zeros((400, 592), dtype=np.uint8)
    mask_array[101:127, 251:279] = 255
    image = Image.fromarray(image_array, mode="RGB")
    mask = Image.fromarray(mask_array, mode="L")
    image.save(root / "images" / "Misc_111.png")
    mask.save(root / "masks" / "Misc_111.png")
    (root / "img_idx" / "train_NUAA-SIRST.txt").write_text(
        "Misc_111\n", encoding="utf-8"
    )
    (root / "img_idx" / "test_NUAA-SIRST.txt").write_text(
        "Misc_111\n", encoding="utf-8"
    )
    return image, mask


def test_corrected_alignment_uses_nearest_and_strict_one_percent_gate(
    tmp_path,
) -> None:
    image, mask = _write_misc_111_fixture(tmp_path / "NUAA-SIRST")
    aligned, evidence = align_mask_to_image(mask, image, "Misc_111")
    resampling = getattr(Image, "Resampling", Image).NEAREST
    expected = mask.resize(image.size, resampling)

    image_ratio = float(image.size[0]) / float(image.size[1])
    mask_ratio = float(mask.size[0]) / float(mask.size[1])
    expected_relative_error = abs(image_ratio - mask_ratio) / max(
        abs(image_ratio), 1e-12
    )
    assert evidence.relative_aspect_error == expected_relative_error
    assert evidence.relative_aspect_error <= MASK_ALIGNMENT_ASPECT_TOLERANCE
    assert np.array_equal(np.asarray(aligned), np.asarray(expected))

    assert aspect_error_within_tolerance(0.01)
    one_ulp_over = float(np.nextafter(np.float64(0.01), np.float64(np.inf)))
    assert one_ulp_over > 0.01
    assert not aspect_error_within_tolerance(one_ulp_over)


def test_training_evaluation_and_export_share_the_one_alignment_implementation(
    tmp_path,
) -> None:
    root = tmp_path / "NUAA-SIRST"
    image, source_mask = _write_misc_111_fixture(root)
    resampling = getattr(Image, "Resampling", Image).NEAREST
    expected = torch.from_numpy(
        np.ascontiguousarray(
            np.asarray(source_mask.resize(image.size, resampling), dtype=np.uint8) > 0
        )
    ).unsqueeze(0)

    # Both detector-training loaders, the evaluator, and both export entry
    # points must reference the same central function rather than local copies.
    assert legacy_dataset_module.align_mask_to_image is alignment_module.align_mask_to_image
    assert complete_dataset_module.align_mask_to_image is alignment_module.align_mask_to_image
    assert eval_dataset_module.align_mask_to_image is alignment_module.align_mask_to_image
    assert export_module.IRSTDEvalDataset is eval_dataset_module.IRSTDEvalDataset
    assert export_cli_module.IRSTDEvalDataset is eval_dataset_module.IRSTDEvalDataset

    legacy_args = SimpleNamespace(
        dataset_dir=str(root),
        crop_size=256,
        base_size=256,
        train_split_file="",
        test_split_file="",
    )
    legacy_train = IRSTD_Dataset(legacy_args, mode="train")
    # Isolate the pre-augmentation alignment contract.  Production training
    # applies this same alignment immediately before its synchronized transform.
    legacy_train._sync_transform = lambda img, mask: (img, mask)
    _, legacy_mask = legacy_train[0]
    assert torch.equal(legacy_mask.bool(), expected)

    complete_train = IRSTDDataset(
        root,
        "train",
        dataset_name="NUAA-SIRST",
        training=True,
        image_size=(image.height, image.width),
        augment=False,
    )
    complete_mask = complete_train[0]["mask"]
    assert torch.equal(complete_mask.bool(), expected)

    evaluation = IRSTDEvalDataset(
        root,
        split="test",
        spatial_mode="native",
        pad_multiple=16,
        dataset_name="NUAA-SIRST",
        load_masks=True,
    )
    evaluation_sample = evaluation[0]
    evaluation_mask = crop_to_valid(
        evaluation_sample["mask"], evaluation_sample["meta"]
    )
    assert torch.equal(evaluation_mask.bool(), expected)
    assert evaluation_sample["meta"]["mask_alignment_applied"] is True
    assert evaluation_sample["meta"]["mask_original_hw"].tolist() == [400, 592]
