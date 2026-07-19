from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import evaluation.raw_logit_oracle as raw_logit_oracle_module
from data_ext.eval_dataset import IRSTDEvalDataset
from evaluation.artifact_integrity import (
    RAW_LOGIT_SCORE_REPRESENTATION,
    SCORE_MANIFEST_SCHEMA_VERSION,
    SCORE_RECORD_INTEGRITY_SCHEMA,
    file_sha256,
    score_records_sha256,
    verify_score_map_directory,
)
from evaluation.export_score_maps import export_dataset_score_maps
from evaluation.raw_logit_oracle import (
    RawLogitSample,
    build_raw_logit_oracle_payload,
    compare_probability_with_reference,
    load_formal_raw_logit_directory,
    select_exact_global_oracle,
    select_exact_probability_global_oracle,
    validate_formal_raw_logit_manifest,
)


class _PatternLogitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.probe = torch.nn.Conv2d(3, 1, kernel_size=1, bias=False)
        torch.nn.init.zeros_(self.probe.weight)
        self.seen_probe_dtype: torch.dtype | None = None

    def forward(self, images: torch.Tensor, warm_flag: bool) -> torch.Tensor:
        del warm_flag
        probe = self.probe(images)
        self.seen_probe_dtype = probe.dtype
        logits = torch.full_like(probe, -20.0)
        logits[:, :, 0, 0] = 100.0
        logits[:, :, 0, 2] = 90.0
        logits[:, :, 2, 2] = 80.0
        return logits


def _dataset(tmp_path: Path) -> IRSTDEvalDataset:
    root = tmp_path / "target-domain"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[0, 0] = 255
    mask[2, 2] = 255
    Image.fromarray(image).save(root / "images" / "sample.png")
    Image.fromarray(mask).save(root / "masks" / "sample.png")
    (root / "test.txt").write_text("sample\n", encoding="utf-8")
    return IRSTDEvalDataset(
        root,
        split="test",
        spatial_mode="native",
        pad_multiple=4,
    )


def _formal_metadata() -> dict[str, object]:
    return {
        "weight_sha256": "a" * 64,
        "checkpoint_selection_rule": "fixed_last",
        "checkpoint_diagnostic_only": False,
        "diagnostic_only": False,
        "non_strict_state_loading": False,
        "formal_protocol_eligible": True,
        "model_backend": "canonical",
        "source_datasets": ["source-domain"],
    }


def _export_raw(tmp_path: Path, name: str = "raw") -> tuple[Path, dict, _PatternLogitModel]:
    dataset = _dataset(tmp_path / name)
    model = _PatternLogitModel()
    output = tmp_path / f"{name}-scores"
    manifest = export_dataset_score_maps(
        model,
        dataset,
        output,
        labels_loaded=True,
        manifest_metadata=_formal_metadata(),
        export_raw_logits=True,
    )
    return output, manifest, model


def test_raw_export_is_additive_fp32_no_clip_and_matches_legacy_probability(
    tmp_path: Path,
) -> None:
    raw_dir, raw_manifest, raw_model = _export_raw(tmp_path, "raw")
    legacy_dataset = _dataset(tmp_path / "legacy")
    legacy_dir = tmp_path / "legacy-scores"
    legacy_manifest = export_dataset_score_maps(
        _PatternLogitModel(),
        legacy_dataset,
        legacy_dir,
        labels_loaded=True,
        manifest_metadata=_formal_metadata(),
    )

    assert raw_manifest["score_type"] == legacy_manifest["score_type"]
    assert "score_representation" not in legacy_manifest
    assert raw_manifest["score_representation"] == RAW_LOGIT_SCORE_REPRESENTATION
    assert raw_manifest["probability_dtype"] == "float32"
    assert raw_manifest["logit_dtype"] == "float32"
    assert raw_manifest["probability_transform"] == "sigmoid"
    assert raw_manifest["probability_clipping"] == "none"
    assert raw_manifest["inference_autocast_enabled"] is False
    assert raw_model.seen_probe_dtype == torch.float32

    raw_path = raw_dir / raw_manifest["records"][0]["file"]
    legacy_path = legacy_dir / legacy_manifest["records"][0]["file"]
    with np.load(raw_path, allow_pickle=False) as raw, np.load(
        legacy_path, allow_pickle=False
    ) as legacy:
        assert raw["logit"].dtype == np.float32
        assert raw["prob"].dtype == np.float32
        assert np.array_equal(raw["prob"], legacy["prob"])
        assert int(np.count_nonzero(raw["prob"] == np.float32(1.0))) == 3
        assert str(raw["score_representation"].item()) == (
            RAW_LOGIT_SCORE_REPRESENTATION
        )
        assert str(raw["probability_clipping"].item()) == "none"
        assert bool(raw["inference_autocast_enabled"].item()) is False
    _, _, integrity = verify_score_map_directory(
        raw_dir, require_integrity=True, require_masks=True
    )
    assert integrity["verified"] is True
    comparison = compare_probability_with_reference(raw_dir, legacy_dir)
    assert comparison["bitwise_equal"] is True
    assert comparison["num_pixels"] == 9


def test_raw_export_disables_ambient_cpu_autocast(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "ambient")
    model = _PatternLogitModel()
    output = tmp_path / "ambient-scores"
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        manifest = export_dataset_score_maps(
            model,
            dataset,
            output,
            labels_loaded=True,
            manifest_metadata=_formal_metadata(),
            export_raw_logits=True,
        )
    assert model.seen_probe_dtype == torch.float32
    with np.load(output / manifest["records"][0]["file"], allow_pickle=False) as data:
        assert data["logit"].dtype == np.float32
        assert data["prob"].dtype == np.float32


def test_probability_reference_comparison_rejects_one_pixel_difference(
    tmp_path: Path,
) -> None:
    raw_dir, _, _ = _export_raw(tmp_path, "raw-reference")
    dataset = _dataset(tmp_path / "changed-reference")
    reference_dir = tmp_path / "changed-reference-scores"
    manifest = export_dataset_score_maps(
        _PatternLogitModel(),
        dataset,
        reference_dir,
        labels_loaded=True,
        manifest_metadata=_formal_metadata(),
    )
    record_path = reference_dir / manifest["records"][0]["file"]
    with np.load(record_path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays["prob"] = arrays["prob"].copy()
    arrays["prob"][1, 1] = np.nextafter(
        arrays["prob"][1, 1], np.float32(1.0), dtype=np.float32
    )
    np.savez_compressed(record_path, **arrays)
    manifest["records"][0]["sha256"] = file_sha256(record_path)
    manifest["records_sha256"] = score_records_sha256(manifest["records"])
    (reference_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not bitwise equal"):
        compare_probability_with_reference(raw_dir, reference_dir)


def test_exact_raw_logit_oracle_recovers_order_hidden_by_probability_saturation(
    tmp_path: Path,
) -> None:
    score_dir, manifest, _ = _export_raw(tmp_path, "oracle")
    payload = build_raw_logit_oracle_payload(
        score_dir,
        pixel_budget=1e-6,
        component_budget=1.0,
    )
    point = payload["operating_point"]
    assert point["threshold_domain"] == "raw_logit"
    assert point["threshold_logit"] == 100.0
    assert point["threshold_probability_float32"] == 1.0
    assert point["pd"] == pytest.approx(0.5)
    assert point["fp_pixels"] == 0
    assert point["fp_components"] == 0
    assert payload["checkpoint_sha256"] == "a" * 64
    assert payload["score_records_sha256"] == manifest["records_sha256"]
    assert len(payload["raw_logit_stream_sha256"]) == 64
    assert payload["oracle_only"] is True
    assert payload["formal_protocol_eligible"] is False

    search = payload["search"]
    assert search["exact"] is True
    assert search["num_unique_logits_total"] == 4
    assert search["num_unique_logits_evaluated"] == 1
    assert search["num_unique_logits_proven_infeasible"] == 3
    assert search["num_prediction_states_evaluated"] == 2
    assert search["reject_all_threshold_logit"] > 100.0

    saturation = payload["exact_one_saturation_audit"]
    assert saturation["exact_one_pixels"] == 3
    assert saturation["exact_one_target_pixels"] == 2
    assert saturation["exact_one_background_pixels"] == 1
    assert saturation["candidate_components"] == 3
    assert saturation["false_candidate_components"] == 1
    assert saturation["per_image"][0]["exact_one_pixels"] == 3

    ranking = payload["target_background_ranking_audit"]
    assert ranking["gt_objects"] == 2
    assert ranking["background_logit_quantiles"]["q99.99"] <= 90.0
    assert ranking["highest_background_false_peak_logit"] == 90.0
    assert ranking["gt_max_below_highest_background_false_peak"] == 1
    assert ranking["gt_objects_saturated_at_probability_one"] == 2
    assert ranking["background_false_peaks_saturated_at_probability_one"] >= 1
    assert ranking["target_vs_false_peak_pairwise_ranking"]["num_pairs"] > 0

    probability_oracle = payload["exact_oracles"][
        "sigmoid_probability_float32"
    ]
    assert probability_oracle["operating_point"]["pd"] == 0.0
    assert probability_oracle["operating_point"][
        "threshold_probability_float32"
    ] > 1.0
    comparison = payload["exact_domain_comparison"]
    assert comparison["pd_difference_raw_logit_minus_probability"] == 0.5
    assert comparison["selected_prediction_states_equal"] is False
    image_diagnostics = payload["selected_operating_point_image_diagnostics"]
    assert image_diagnostics["raw_logit_float32"]["aggregate_raw_counts"] == {
        "tp_objects": 1,
        "gt_objects": 2,
        "fp_pixels": 0,
        "fp_components": 0,
        "total_pixels": 9,
    }
    assert (
        image_diagnostics["raw_logit_float32"]["false_alarm_concentration"]
        ["fp_pixels"]["top_1"]["fraction"]
        == 0.0
    )


def _stored_sigmoid(logits: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    return torch.sigmoid(tensor).numpy().astype(np.float32, copy=False)


def test_exact_probability_and_logit_oracles_select_same_state_without_saturation() -> None:
    logits = np.asarray([[2.0, 1.0], [-1.0, -2.0]], dtype=np.float32)
    probability = _stored_sigmoid(logits)
    mask = np.asarray([[1, 0], [0, 0]], dtype=np.uint8)
    sample = RawLogitSample("unsaturated", logits, probability, mask)

    raw = select_exact_global_oracle(
        [sample], pixel_budget=0.3, component_budget=1e9
    )
    prob = select_exact_probability_global_oracle(
        [sample], pixel_budget=0.3, component_budget=1e9
    )

    raw_point = raw["operating_point"]
    probability_point = prob["operating_point"]
    raw_prediction = logits >= raw_point["threshold_logit"]
    probability_prediction = (
        probability >= probability_point["threshold_probability_float32"]
    )
    assert np.array_equal(raw_prediction, probability_prediction)
    assert raw_point["pd"] == probability_point["pd"] == 1.0
    assert raw_point["tp_objects"] == probability_point["tp_objects"] == 1
    assert raw_point["fp_pixels"] == probability_point["fp_pixels"] == 1
    assert prob["search"]["exact"] is True
    assert prob["search"]["num_unique_probabilities_total"] == 4
    assert prob["search"]["tie_audit"]["unique_state_collapse_count"] == 0
    assert prob["search"]["saturation_audit"]["exact_one_pixels"] == 0
    assert prob["search"]["reject_all_threshold_probability_float32"] > 1.0


def test_exact_logit_oracle_can_strictly_outperform_saturated_probability_oracle() -> None:
    logits = np.asarray([[100.0, 90.0]], dtype=np.float32)
    probability = _stored_sigmoid(logits)
    assert np.array_equal(probability, np.ones_like(probability))
    mask = np.asarray([[1, 0]], dtype=np.uint8)
    sample = RawLogitSample("saturated", logits, probability, mask)

    raw = select_exact_global_oracle(
        [sample], pixel_budget=0.1, component_budget=1e9
    )
    prob = select_exact_probability_global_oracle(
        [sample], pixel_budget=0.1, component_budget=1e9
    )

    assert raw["operating_point"]["threshold_logit"] == 100.0
    assert raw["operating_point"]["pd"] == 1.0
    assert raw["operating_point"]["fp_pixels"] == 0
    assert prob["operating_point"]["pd"] == 0.0
    assert prob["operating_point"]["threshold_probability_float32"] > 1.0
    tie_audit = prob["search"]["tie_audit"]
    assert tie_audit["num_probability_tie_groups"] == 1
    assert tie_audit["largest_probability_tie_group_pixels"] == 2
    assert tie_audit["unique_state_collapse_count"] == 1
    saturation = prob["search"]["saturation_audit"]
    assert saturation["exact_one_pixels"] == 2
    assert saturation["exact_one_unique_logits"] == 2


def test_exact_search_has_reject_all_and_never_splits_equal_logit_ties() -> None:
    logits = np.asarray([[5.0, 5.0]], dtype=np.float32)
    probability = np.asarray([[0.99, 0.99]], dtype=np.float32)
    mask = np.asarray([[1, 0]], dtype=np.uint8)
    sample = RawLogitSample("tie", logits, probability, mask)

    strict = select_exact_global_oracle(
        [sample], pixel_budget=0.1, component_budget=1e9
    )
    assert strict["operating_point"]["pd"] == 0.0
    assert strict["operating_point"]["threshold_logit"] > 5.0
    assert strict["search"]["num_unique_logits_evaluated"] == 0
    assert strict["search"]["num_prediction_states_evaluated"] == 1

    permissive = select_exact_global_oracle(
        [sample], pixel_budget=1.0, component_budget=1e9
    )
    assert permissive["operating_point"]["threshold_logit"] == 5.0
    assert permissive["operating_point"]["pd"] == 1.0
    assert permissive["operating_point"]["fp_pixels"] == 1
    assert permissive["search"]["num_unique_logits_evaluated"] == 1


def _valid_contract() -> tuple[dict[str, object], dict[str, object]]:
    digest = "b" * 64
    manifest: dict[str, object] = {
        "schema_version": SCORE_MANIFEST_SCHEMA_VERSION,
        "record_integrity_schema": SCORE_RECORD_INTEGRITY_SCHEMA,
        "labels_loaded": True,
        "spatial_mode": "native",
        "split_authority_verified": True,
        "split_role": "test",
        "requested_split": "test",
        "checkpoint_diagnostic_only": False,
        "diagnostic_only": False,
        "non_strict_state_loading": False,
        "formal_protocol_eligible": True,
        "model_backend": "canonical",
        "checkpoint_selection_rule": "fixed_last",
        "score_type": "sigmoid_probability",
        "target_dataset": "target-domain",
        "source_datasets": ["source-domain"],
        "weight_sha256": "a" * 64,
        "split_file_sha256": digest,
        "split_ordered_ids_sha256": digest,
        "score_representation": RAW_LOGIT_SCORE_REPRESENTATION,
        "probability_dtype": "float32",
        "logit_dtype": "float32",
        "probability_transform": "sigmoid",
        "probability_clipping": "none",
        "inference_autocast_enabled": False,
    }
    integrity: dict[str, object] = {
        "verified": True,
        "mask_alignment_verified": True,
        "manifest_sha256": digest,
        "records_sha256": digest,
        "ordered_image_ids_sha256": digest,
        "num_records": 1,
    }
    return manifest, integrity


@pytest.mark.parametrize(
    ("call_kwargs", "expected_split_role"),
    [({}, "test"), ({"expected_split_role": "train"}, "train")],
)
def test_raw_manifest_validator_forwards_default_and_explicit_split_role(
    monkeypatch: pytest.MonkeyPatch,
    call_kwargs: dict[str, str],
    expected_split_role: str,
) -> None:
    manifest, integrity = _valid_contract()
    observed: list[str] = []

    def fake_formal_validator(
        observed_manifest: object,
        observed_integrity: object,
        *,
        expected_split_role: str,
    ) -> dict[str, object]:
        assert observed_manifest is manifest
        assert observed_integrity is integrity
        observed.append(expected_split_role)
        return {"delegated_expected_split_role": expected_split_role}

    monkeypatch.setattr(
        raw_logit_oracle_module,
        "validate_formal_score_manifest",
        fake_formal_validator,
    )
    contract = validate_formal_raw_logit_manifest(
        manifest,
        integrity,
        **call_kwargs,
    )

    assert observed == [expected_split_role]
    assert contract["delegated_expected_split_role"] == expected_split_role


def test_raw_loader_forwards_explicit_train_split_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score_dir, _, _ = _export_raw(tmp_path, "loader-split-role")
    original = raw_logit_oracle_module.validate_formal_raw_logit_manifest
    observed: list[str] = []

    def spy_raw_manifest_validator(
        manifest: object,
        integrity: object,
        *,
        expected_split_role: str = "test",
    ) -> dict[str, object]:
        observed.append(expected_split_role)
        # The fixture is a legacy test-split export.  Delegate its substantive
        # contract audit with the historical role while isolating this test to
        # the loader's new keyword-forwarding responsibility.
        return original(manifest, integrity, expected_split_role="test")

    monkeypatch.setattr(
        raw_logit_oracle_module,
        "validate_formal_raw_logit_manifest",
        spy_raw_manifest_validator,
    )
    samples, _, _, _ = load_formal_raw_logit_directory(
        score_dir,
        expected_split_role="train",
    )

    assert observed == ["train"]
    assert [sample.image_id for sample in samples] == ["sample"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema version 3"),
        ("labels_loaded", False, "ground-truth masks"),
        ("spatial_mode", "resize", "native spatial_mode"),
        ("checkpoint_selection_rule", "best", "fixed_last"),
        ("source_datasets", ["target-domain"], "held out"),
    ],
)
def test_raw_oracle_contract_fails_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    manifest, integrity = _valid_contract()
    manifest[field] = value
    with pytest.raises(ValueError, match=message):
        validate_formal_raw_logit_manifest(manifest, integrity)


def test_raw_loader_rejects_probability_only_v3_and_tampered_record(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "probability-only")
    probability_dir = tmp_path / "probability-only-scores"
    export_dataset_score_maps(
        _PatternLogitModel(),
        dataset,
        probability_dir,
        labels_loaded=True,
        manifest_metadata=_formal_metadata(),
    )
    with pytest.raises(ValueError, match="score_representation"):
        load_formal_raw_logit_directory(probability_dir)

    raw_dir, manifest, _ = _export_raw(tmp_path, "tampered")
    record = raw_dir / manifest["records"][0]["file"]
    record.write_bytes(record.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_formal_raw_logit_directory(raw_dir)


def test_raw_manifest_precision_contract_cannot_be_partially_declared(
    tmp_path: Path,
) -> None:
    raw_dir, manifest, _ = _export_raw(tmp_path, "partial")
    manifest = copy.deepcopy(manifest)
    del manifest["logit_dtype"]
    (raw_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="precision contract is incomplete"):
        verify_score_map_directory(raw_dir, require_integrity=True, require_masks=True)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("prob_float16", "probability map must be float32"),
        ("prob_3d", "probability map must be a non-empty 2-D"),
        ("prob_nan", "probability map must be finite"),
        ("prob_out_of_range", "probability map must be finite"),
        ("gray_float16", "gray map must be float32"),
        ("gray_shape", "gray/probability shape mismatch"),
        ("gray_inf", "gray map must be finite"),
        ("mask_float32", "mask must use uint8 or bool"),
        ("mask_shape", "mask/probability shape mismatch"),
        ("mask_nonbinary", "mask must be binary"),
    ],
)
def test_formal_v3_numeric_semantics_fail_closed_but_diagnostic_stays_compatible(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    dataset = _dataset(tmp_path / corruption)
    score_dir = tmp_path / f"{corruption}-scores"
    manifest = export_dataset_score_maps(
        _PatternLogitModel(),
        dataset,
        score_dir,
        labels_loaded=True,
        manifest_metadata=_formal_metadata(),
    )
    record_path = score_dir / manifest["records"][0]["file"]
    with np.load(record_path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}

    if corruption == "prob_float16":
        arrays["prob"] = arrays["prob"].astype(np.float16)
    elif corruption == "prob_3d":
        arrays["prob"] = arrays["prob"][None, ...]
        manifest["records"][0]["shape"] = list(arrays["prob"].shape)
    elif corruption == "prob_nan":
        arrays["prob"] = arrays["prob"].copy()
        arrays["prob"][0, 0] = np.nan
    elif corruption == "prob_out_of_range":
        arrays["prob"] = arrays["prob"].copy()
        arrays["prob"][0, 0] = 1.5
    elif corruption == "gray_float16":
        arrays["gray"] = arrays["gray"].astype(np.float16)
    elif corruption == "gray_shape":
        arrays["gray"] = arrays["gray"][:2, :]
    elif corruption == "gray_inf":
        arrays["gray"] = arrays["gray"].copy()
        arrays["gray"][0, 0] = np.inf
    elif corruption == "mask_float32":
        arrays["mask"] = arrays["mask"].astype(np.float32)
    elif corruption == "mask_shape":
        arrays["mask"] = arrays["mask"][:2, :]
    elif corruption == "mask_nonbinary":
        arrays["mask"] = arrays["mask"].copy()
        arrays["mask"][0, 1] = 2
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(corruption)

    np.savez_compressed(record_path, **arrays)
    manifest["records"][0]["sha256"] = file_sha256(record_path)
    manifest["records_sha256"] = score_records_sha256(manifest["records"])
    (score_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    # Diagnostic/legacy callers retain the historical permissive behavior.
    verify_score_map_directory(score_dir, require_integrity=False, require_masks=True)
    with pytest.raises(ValueError, match=message):
        verify_score_map_directory(
            score_dir,
            require_integrity=True,
            require_masks=True,
        )
