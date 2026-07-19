import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from risk_curve.curve_dataset import (
    CurveDataset,
    load_curve_archive,
    validate_archive_compatibility,
)
from risk_curve.curve_metrics import curve_regression_metrics
from risk_curve.build_deployment_statistics import build_deployment_statistics
from risk_curve.build_curve_episodes import (
    ScoreSample,
    _pack_episodes,
    _resolve_window_config,
    assert_source_reference_excludes_pseudo_targets,
    audit_fold_score_manifest,
    build_causal_windows,
    build_episode,
    load_score_sample,
    main as build_curve_episodes_main,
    monotone_upper_envelope,
)
from risk_curve.domain_statistics import (
    STATISTICS_SCHEMA_VERSION,
    append_source_distances,
    extract_window_statistics,
    fit_source_reference,
    load_source_reference,
    save_source_reference,
    statistics_names_sha256,
)
from risk_curve.domain_statistics import _peak_values
from risk_curve.monotone_curve_predictor import (
    COMPONENT_LOG_RISK_FLOOR,
    PIXEL_LOG_RISK_FLOOR,
    RiskCurvePredictor,
)
from risk_curve.quantile_loss import pinball_loss
from risk_curve.select_zero_label_threshold import (
    _score_map_protocol_provenance,
    _statistics_from_archive,
    _threshold_indices_by_image,
    assess_ood_statistics,
    select_dual_budget_threshold,
)
from risk_curve.threshold_grid import (
    CUSTOM_GRID_VERSION,
    GRID_VERSION,
    build_threshold_grid,
    threshold_grid_sha256,
    threshold_grid_version,
)
from risk_curve.train_curve_predictor import _evaluate


def test_threshold_grid_is_dense_and_strictly_increasing():
    grid = build_threshold_grid()
    assert grid.size == 253
    assert grid[0] == 0.0
    assert grid[-1] < 1.0
    assert np.all(np.diff(grid) > 0.0)


def test_predictor_is_structurally_monotone():
    model = RiskCurvePredictor(input_dim=8, num_thresholds=17, hidden_dim=16, dropout=0.0)
    curves = model(torch.randn(5, 8))
    for curve in curves.values():
        assert tuple(curve.shape) == (5, 17)
        assert torch.all(torch.diff(curve, dim=1) <= 0.0)


def test_predictor_cannot_fall_below_physical_log_risk_floors():
    model = RiskCurvePredictor(input_dim=2, num_thresholds=17, hidden_dim=4, dropout=0.0)
    with torch.no_grad():
        for head in (model.pixel_head, model.component_head):
            head.start.weight.zero_()
            head.start.bias.zero_()
            head.decrements.weight.zero_()
            head.decrements.bias.fill_(10.0)
    curves = model(torch.zeros(1, 2))
    assert curves["pixel_log_risk"].min().detach().item() >= PIXEL_LOG_RISK_FLOOR
    assert (
        curves["component_log_risk"].min().detach().item()
        >= COMPONENT_LOG_RISK_FLOOR
    )
    assert torch.all(torch.diff(curves["pixel_log_risk"], dim=1) <= 0.0)
    assert torch.all(torch.diff(curves["component_log_risk"], dim=1) <= 0.0)


def test_pinball_loss_prefers_exact_prediction():
    target = torch.tensor([0.0, 1.0, 2.0])
    assert pinball_loss(target, target).item() == 0.0
    assert pinball_loss(target + 1.0, target).item() > 0.0


def test_zero_label_selector_and_reject():
    thresholds = np.asarray([0.1, 0.5, 0.9])
    pixel = np.log10([1e-3, 1e-6, 1e-8])
    component = np.log10([10.0, 0.5, 0.1])
    threshold, reject, index = select_dual_budget_threshold(
        thresholds, pixel, component, 1e-6, 1.0
    )
    assert (threshold, reject, index) == (0.5, False, 1)
    threshold, reject, index = select_dual_budget_threshold(
        thresholds, pixel, component, 1e-12, 1e-6
    )
    assert (threshold, reject, index) == (1.0, True, None)


def test_zero_selector_rejects_invalid_grid_curve_and_budget_values():
    valid_grid = np.asarray([0.1, 0.5, 0.9])
    with pytest.raises(ValueError, match="strictly increasing"):
        select_dual_budget_threshold(
            valid_grid[::-1], [-1, -2, -3], [1, 0, -1], 1e-2, 1.0
        )
    with pytest.raises(ValueError, match="non-increasing"):
        select_dual_budget_threshold(
            valid_grid, [-1, -2, -1.5], [1, 0, -1], 1e-2, 1.0
        )
    with pytest.raises(ValueError, match="physical epsilon floor"):
        select_dual_budget_threshold(
            valid_grid, [-1, -2, -13], [1, 0, -1], 1e-2, 1.0
        )
    with pytest.raises(ValueError, match="finite and positive"):
        select_dual_budget_threshold(
            valid_grid, [-1, -2, -3], [1, 0, -1], np.nan, 1.0
        )


def test_ood_audit_is_conservative_and_names_are_ordered():
    safe = assess_ood_statistics(np.asarray([0.0, 7.9]), ["a", "b"], 8.0)
    assert safe["is_ood"] is False
    unsafe = assess_ood_statistics(np.asarray([0.0, -9.0]), ["a", "b"], 8.0)
    assert unsafe["is_ood"] is True
    assert unsafe["top_exceeding_features"][0]["name"] == "b"
    with pytest.raises(ValueError, match="unique"):
        assess_ood_statistics(np.asarray([0.0, 1.0]), ["a", "a"], 8.0)


def test_score_map_protocol_provenance_matches_certification_schema(tmp_path: Path):
    from certification.build_calibration_losses import score_map_protocol

    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    manifest = {
        "score_type": "sigmoid_probability",
        "warm_flag": False,
        "spatial_mode": "native",
        "target_dataset": "NUAA-SIRST",
        "source_dataset": "IRSTD-1k",
        "weight_sha256": "d" * 64,
    }
    manifest_path = score_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    thresholds = np.asarray([0.0, 0.5, 0.9], dtype=np.float32)

    protocol, fingerprint, provenance = _score_map_protocol_provenance(
        score_dir, thresholds
    )
    expected_protocol, expected_fingerprint = score_map_protocol(
        score_dir,
        thresholds,
        matching_rule="overlap",
        centroid_distance=3.0,
        connectivity=2,
        min_component_area=1,
    )
    assert protocol == expected_protocol
    assert fingerprint == expected_fingerprint
    assert provenance["detector_weight_sha256"] == "d" * 64
    assert provenance["warm_flag"] is False
    assert provenance["spatial_mode"] == "native"
    assert provenance["target_dataset"] == "NUAA-SIRST"
    assert len(provenance["manifest_sha256"]) == 64


def test_statistics_archive_batch_ids_and_duplicate_mapping_conflicts(tmp_path: Path):
    path = tmp_path / "statistics.npz"
    np.savez_compressed(
        path,
        statistics=np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        statistics_names=np.asarray(["a", "b"]),
        statistics_names_sha256=np.asarray(statistics_names_sha256(["a", "b"])),
        statistics_schema_version=np.asarray(STATISTICS_SCHEMA_VERSION),
        evaluation_ids=np.asarray(
            [json.dumps(["cal-a", "cal-b"]), json.dumps(["test-a"])]
        ),
        adaptation_ids=np.asarray(
            [json.dumps(["warm-a"]), json.dumps(["warm-b"])]
        ),
    )
    (
        statistics,
        names,
        schema,
        evaluation_ids,
        adaptation_ids,
        *_audit,
    ) = _statistics_from_archive(path)
    assert statistics.shape == (2, 2)
    assert names == ("a", "b")
    assert schema == STATISTICS_SCHEMA_VERSION
    assert evaluation_ids == [["cal-a", "cal-b"], ["test-a"]]
    assert adaptation_ids == [["warm-a"], ["warm-b"]]
    assert _threshold_indices_by_image(evaluation_ids, [1, 2]) == {
        "cal-a": 1,
        "cal-b": 1,
        "test-a": 2,
    }
    assert _threshold_indices_by_image([["same"], ["same"]], [1, 1]) == {
        "same": 1
    }
    with pytest.raises(ValueError, match="conflicting threshold indices"):
        _threshold_indices_by_image([["same"], ["same"]], [1, 2])

    image_ids_path = tmp_path / "image-ids.npz"
    np.savez_compressed(
        image_ids_path,
        statistics=np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        statistics_names=np.asarray(["a", "b"]),
        statistics_schema_version=np.asarray(STATISTICS_SCHEMA_VERSION),
        image_ids=np.asarray(["image-a", "image-b"]),
    )
    image_archive = _statistics_from_archive(image_ids_path)
    image_id_rows, no_adaptation_ids = image_archive[3], image_archive[4]
    assert image_id_rows == [["image-a"], ["image-b"]]
    assert no_adaptation_ids == []


def test_curve_metrics_report_zero_violation_for_decreasing_curve():
    true = np.asarray([[0.0, -1.0, -2.0]])
    metrics = curve_regression_metrics(true + 0.1, true)
    assert metrics["upper_bound_coverage"] == 1.0
    assert metrics["upper_bound_coverage_by_threshold"] == [1.0, 1.0, 1.0]
    assert metrics["minimum_threshold_coverage"] == 1.0
    assert metrics["worst_coverage_threshold_index"] == 0
    assert metrics["monotonic_violation_rate"] == 0.0


def test_component_upper_envelope_is_monotone_and_conservative():
    original = np.asarray([3.0, 2.0, 2.5, 0.0])
    envelope = monotone_upper_envelope(original)
    assert np.all(envelope >= original)
    assert np.all(np.diff(envelope) <= 0.0)


def test_label_free_statistics_and_optional_source_distances_are_finite():
    probability = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    base = extract_window_statistics([probability], [probability])
    reference = fit_source_reference(
        {
            "source_a": np.stack([base.values, base.values + 0.01]),
            "source_b": np.stack([base.values + 0.1, base.values + 0.2]),
        }
    )
    augmented = append_source_distances(base, reference)
    assert base.values.size == 105
    assert augmented.values.size == base.values.size + 4
    assert np.isfinite(augmented.values).all()


def test_flat_peak_plateau_contributes_one_candidate_not_every_pixel():
    flat = np.full((64, 64), 0.5, dtype=np.float32)
    peaks = _peak_values(flat)
    np.testing.assert_array_equal(peaks, [0.5])


def _score_sample(
    image_id: str,
    probability: np.ndarray,
    mask: np.ndarray,
) -> ScoreSample:
    return ScoreSample(
        image_id=image_id,
        probability=np.asarray(probability, dtype=np.float32),
        mask=np.asarray(mask, dtype=np.uint8),
        gray=np.asarray(probability, dtype=np.float32),
        source_path=f"/{image_id}.npz",
    )


def test_causal_episode_statistics_and_labels_use_disjoint_roles(tmp_path: Path):
    shape = (8, 8)
    adaptation_probability = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(shape)
    evaluation_probability = np.zeros(shape, dtype=np.float32)
    evaluation_probability[3, 4] = 0.9
    zeros = np.zeros(shape, dtype=np.uint8)
    ones_in_a = np.ones(shape, dtype=np.uint8)
    target_in_e = np.zeros(shape, dtype=np.uint8)
    target_in_e[3, 4] = 1
    thresholds = np.asarray([0.5, 0.95], dtype=np.float32)

    base = build_episode(
        [_score_sample("a", adaptation_probability, zeros)],
        thresholds,
        "domain",
        evaluation_samples=[_score_sample("e", evaluation_probability, zeros)],
    )
    changed_a_mask = build_episode(
        [_score_sample("a", adaptation_probability, ones_in_a)],
        thresholds,
        "domain",
        evaluation_samples=[_score_sample("e", evaluation_probability, zeros)],
    )
    changed_e_mask = build_episode(
        [_score_sample("a", adaptation_probability, zeros)],
        thresholds,
        "domain",
        evaluation_samples=[_score_sample("e", evaluation_probability, target_in_e)],
    )

    # A masks are neither features nor curve labels.
    np.testing.assert_array_equal(base.statistics.values, changed_a_mask.statistics.values)
    np.testing.assert_array_equal(base.pixel_fp_counts, changed_a_mask.pixel_fp_counts)
    np.testing.assert_array_equal(base.tp_object_counts, changed_a_mask.tp_object_counts)
    # E masks change supervised labels but cannot affect A-only statistics.
    np.testing.assert_array_equal(base.statistics.values, changed_e_mask.statistics.values)
    assert base.pixel_fp_counts[0] == 1
    assert changed_e_mask.pixel_fp_counts[0] == 0
    assert base.tp_object_counts[0] == 0
    assert changed_e_mask.tp_object_counts[0] == 1
    assert changed_e_mask.pd_curve[0] == 1.0

    archive_path = tmp_path / "episodes.npz"
    _pack_episodes([changed_e_mask], archive_path, {"protocol": "causal"})
    with np.load(archive_path, allow_pickle=False) as archive:
        assert json.loads(str(archive["adaptation_ids"][0])) == ["a"]
        assert json.loads(str(archive["evaluation_ids"][0])) == ["e"]
        assert str(archive["window_ids_alias"].item()) == "evaluation_ids"
        assert json.loads(str(archive["window_ids"][0])) == ["e"]


def test_causal_episode_rejects_overlapping_image_ids():
    probability = np.zeros((8, 8), dtype=np.float32)
    mask = np.zeros_like(probability, dtype=np.uint8)
    repeated = _score_sample("same", probability, mask)
    with pytest.raises(ValueError, match="must be disjoint"):
        build_episode(
            [repeated],
            np.asarray([0.5], dtype=np.float32),
            "domain",
            evaluation_samples=[repeated],
        )


def test_adaptation_score_sample_does_not_require_a_mask(tmp_path: Path):
    path = tmp_path / "unlabeled.npz"
    probability = np.zeros((8, 8), dtype=np.float32)
    np.savez_compressed(
        path,
        prob=probability,
        gray=probability,
        image_id=np.asarray("warmup"),
    )
    sample = load_score_sample(path, require_mask=False)
    assert sample.mask is None
    with pytest.raises(ValueError, match="required array 'mask'"):
        load_score_sample(path, require_mask=True)


def test_fold_manifest_rejects_detector_target_leakage(tmp_path: Path):
    root = tmp_path / "scores"
    root.mkdir()
    manifest = {
        "target_dataset": "NUAA-SIRST",
        "source_datasets": ["NUDT-SIRST", "NUAA-SIRST"],
        "weight_sha256": "d" * 64,
        "warm_flag": True,
        "spatial_mode": "native",
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Pseudo-target leakage"):
        audit_fold_score_manifest(root, "nuaa")
    manifest["source_datasets"] = ["NUDT-SIRST", "IRSTD-1K"]
    manifest["weight_sha256"] = "not-a-digest"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    invalid = audit_fold_score_manifest(root, "nuaa")
    assert invalid["verified"] is False
    assert invalid["reason"] == "invalid_detector_weight_sha256"
    manifest["weight_sha256"] = "d" * 64
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    audit = audit_fold_score_manifest(root, "nuaa")
    assert audit["verified"] is True


def test_fold_manifest_rejects_cli_target_mismatch(tmp_path: Path):
    root = tmp_path / "scores"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "target_dataset": "NUDT-SIRST",
                "source_datasets": ["NUAA-SIRST", "IRSTD-1K"],
                "weight_sha256": "d" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Pseudo-target mismatch"):
        audit_fold_score_manifest(root, "NUAA-SIRST")


def test_deployment_statistics_are_causal_mask_free_and_protocol_bound(tmp_path: Path):
    root = tmp_path / "deployment-scores"
    root.mkdir()
    records = []
    for index in range(4):
        filename = f"{index}.npz"
        probability = np.full((8, 8), index / 10.0, dtype=np.float32)
        # Loading this object array with allow_pickle=False would fail if the
        # mask were touched.  The deployment-statistics path must ignore it.
        np.savez_compressed(
            root / filename,
            prob=probability,
            gray=probability,
            mask=np.asarray([{"private": "label"}], dtype=object),
            image_id=np.asarray(f"image-{index}"),
        )
        records.append({"file": filename, "image_id": f"image-{index}"})
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "score_type": "sigmoid_probability",
                "warm_flag": True,
                "spatial_mode": "native",
                "pad_multiple": 16,
                "target_dataset": "target",
                "source_datasets": ["source-a"],
                "weight_sha256": "a" * 64,
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    arrays = build_deployment_statistics(
        root,
        np.asarray([0.0, 0.5, 0.9], dtype=np.float32),
        adaptation_window=1,
        evaluation_window=1,
        stride=2,
    )
    assert np.asarray(arrays["statistics"]).shape[0] == 2
    assert [json.loads(value) for value in arrays["adaptation_ids"]] == [
        ["image-0"],
        ["image-2"],
    ]
    assert [json.loads(value) for value in arrays["evaluation_ids"]] == [
        ["image-1"],
        ["image-3"],
    ]
    assert len(str(np.asarray(arrays["protocol_fingerprint"]).item())) == 64
    with pytest.raises(ValueError, match="change roles"):
        build_deployment_statistics(
            root,
            np.asarray([0.0, 0.5, 0.9], dtype=np.float32),
            adaptation_window=1,
            evaluation_window=1,
            stride=1,
        )


def test_causal_windows_are_ordered_disjoint_and_default_stride_is_full_span():
    files = [Path(f"{index:02d}.npz") for index in range(10)]
    windows = build_causal_windows(files, adaptation_window=3, evaluation_window=2, stride=5)
    assert windows[0].adaptation_files == tuple(files[0:3])
    assert windows[0].evaluation_files == tuple(files[3:5])
    assert windows[1].adaptation_files == tuple(files[5:8])
    assert windows[1].evaluation_files == tuple(files[8:10])
    for window in windows:
        assert set(window.adaptation_files).isdisjoint(window.evaluation_files)

    args = argparse.Namespace(
        adaptation_window=None,
        evaluation_window=None,
        window_size=None,
        stride=None,
    )
    _resolve_window_config(args)
    assert (args.adaptation_window, args.evaluation_window, args.stride) == (32, 1, 33)


def test_deprecated_window_size_maps_to_two_disjoint_windows():
    args = argparse.Namespace(
        adaptation_window=None,
        evaluation_window=None,
        window_size=4,
        stride=None,
    )
    with pytest.warns(FutureWarning, match="deprecated"):
        _resolve_window_config(args)
    assert (args.adaptation_window, args.evaluation_window, args.stride) == (4, 4, 8)


def test_insufficient_causal_windows_are_recorded_in_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    probability = np.zeros((8, 8), dtype=np.float32)
    mask = np.zeros_like(probability, dtype=np.uint8)
    for index in range(3):
        np.savez_compressed(
            score_dir / f"{index:02d}.npz",
            probability=probability,
            prob=probability,
            mask=mask,
            image_id=np.asarray(f"image-{index}"),
        )
    output_dir = tmp_path / "episodes"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_curve_episodes",
            "--score-map-dir",
            str(score_dir),
            "--output-dir",
            str(output_dir),
            "--adaptation-window",
            "2",
            "--evaluation-window",
            "2",
            "--allow-unverified-fold-provenance",
        ],
    )
    with pytest.raises(ValueError, match="No complete causal episodes"):
        build_curve_episodes_main()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "insufficient_complete_windows"
    assert manifest["formal_protocol_eligible"] is False
    assert manifest["score_artifact_integrity_verified"] is False
    assert manifest["protocol_scope"] == "diagnostic_only"
    target_summary = manifest["split_summary"][score_dir.name]
    assert target_summary["train_windowing"]["skipped_reason"] is not None
    assert target_summary["val_windowing"]["skipped_reason"] is not None


def _write_curve_archive(
    path: Path,
    names: tuple[str, ...] = ("feature_a", "feature_b"),
    *,
    num_windows: int = 4,
) -> None:
    thresholds = np.asarray([0.0, 0.5, 0.9], dtype=np.float32)
    statistics = np.arange(num_windows * 2, dtype=np.float32).reshape(num_windows, 2)
    pixel = np.tile(np.asarray([-1.0, -2.0, -3.0], dtype=np.float32), (num_windows, 1))
    component = np.tile(
        np.asarray([1.0, 0.0, -1.0], dtype=np.float32), (num_windows, 1)
    )
    try:
        names_hash = statistics_names_sha256(names)
    except ValueError:
        names_hash = "invalid-schema"
    np.savez_compressed(
        path,
        statistics=statistics,
        statistics_names=np.asarray(names),
        statistics_names_sha256=np.asarray(names_hash),
        statistics_schema_version=np.asarray(STATISTICS_SCHEMA_VERSION),
        pixel_log_risk=pixel,
        component_log_risk=component,
        pd_curve=np.ones_like(pixel),
        thresholds=thresholds,
    )


def test_curve_archives_enforce_unique_exact_ordered_statistics_names(tmp_path: Path):
    train_path = tmp_path / "train.npz"
    validation_path = tmp_path / "validation.npz"
    _write_curve_archive(train_path)
    _write_curve_archive(validation_path, names=("feature_b", "feature_a"))
    train = load_curve_archive(train_path)
    validation = load_curve_archive(validation_path)
    with pytest.raises(ValueError, match="exactly in order"):
        validate_archive_compatibility(train, validation)

    duplicate_path = tmp_path / "duplicate.npz"
    _write_curve_archive(duplicate_path, names=("same", "same"))
    with pytest.raises(ValueError, match="unique"):
        load_curve_archive(duplicate_path)

    with pytest.raises(ValueError, match="non-negative std"):
        CurveDataset(train_path, statistics_std=np.asarray([1.0, -1.0]))


def test_source_reference_roundtrip_validates_feature_schema(tmp_path: Path):
    probability = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    base = extract_window_statistics([probability], [probability])
    reference = fit_source_reference(
        {"source_a": np.stack([base.values, base.values + 0.01])},
        statistics_names=base.names,
    )
    path = save_source_reference(reference, tmp_path / "source-reference.npz")
    loaded = load_source_reference(path)
    assert loaded.statistics_names == base.names
    with pytest.raises(ValueError, match="do not exactly match"):
        loaded.validate(base.values.size, tuple(reversed(base.names)))
    assert_source_reference_excludes_pseudo_targets(loaded, ["external_target"])
    with pytest.raises(ValueError, match="contains pseudo-target domains"):
        assert_source_reference_excludes_pseudo_targets(loaded, ["SOURCE_A"])


def test_grid_version_and_hash_distinguish_canonical_from_custom():
    canonical = build_threshold_grid()
    custom = np.asarray([0.0, 0.5, 0.9], dtype=np.float32)
    assert threshold_grid_version(canonical) == GRID_VERSION
    assert threshold_grid_version(custom) == CUSTOM_GRID_VERSION
    noncontiguous = np.stack([custom, custom], axis=1)[:, 0]
    assert threshold_grid_sha256(noncontiguous) == threshold_grid_sha256(custom)


class _ZeroCurveModel(torch.nn.Module):
    def forward(self, statistics):
        shape = (statistics.shape[0], 3)
        return {
            "pixel_log_risk": torch.zeros(shape, device=statistics.device),
            "component_log_risk": torch.zeros(shape, device=statistics.device),
        }


def test_validation_objective_is_quantile_pinball_not_mae():
    batch = {
        "statistics": torch.zeros(2, 2),
        "pixel_log_risk": torch.ones(2, 3),
        "component_log_risk": torch.full((2, 3), 2.0),
    }
    loader = DataLoader([batch], batch_size=None)
    objective, metrics = _evaluate(
        _ZeroCurveModel(),
        loader,
        torch.device("cpu"),
        quantile=0.9,
        lambda_component=2.0,
    )
    assert objective == pytest.approx(0.9 * 1.0 + 2.0 * 0.9 * 2.0)
    assert objective == pytest.approx(metrics["quantile_pinball_objective"])
    assert metrics["pixel_log_risk_mae"] == pytest.approx(1.0)


def test_train_checkpoint_schema_and_deployment_ood_reject(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    train_path = tmp_path / "train.npz"
    validation_path = tmp_path / "validation.npz"
    _write_curve_archive(train_path)
    _write_curve_archive(validation_path)
    checkpoint_path = tmp_path / "curve.pt"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "risk_curve.train_curve_predictor",
            "--train-file",
            str(train_path),
            "--val-file",
            str(validation_path),
            "--output",
            str(checkpoint_path),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--hidden-dim",
            "4",
            "--device",
            "cpu",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["statistics_schema_version"] == STATISTICS_SCHEMA_VERSION
    assert checkpoint["statistics_names"] == ["feature_a", "feature_b"]
    assert checkpoint["statistics_names_sha256"] == statistics_names_sha256(
        checkpoint["statistics_names"]
    )
    assert checkpoint["threshold_grid_version"] == CUSTOM_GRID_VERSION
    assert "quantile_pinball_objective" in checkpoint["validation_metrics"]

    deployment_path = tmp_path / "deployment.npz"
    np.savez_compressed(
        deployment_path,
        statistics=np.asarray([[1_000.0, 1_000.0]], dtype=np.float32),
        statistics_names=np.asarray(["feature_a", "feature_b"]),
        statistics_schema_version=np.asarray(STATISTICS_SCHEMA_VERSION),
    )
    output_path = tmp_path / "zero.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "risk_curve.select_zero_label_threshold",
            "--statistics-file",
            str(deployment_path),
            "--curve-checkpoint",
            str(checkpoint_path),
            "--pixel-budget",
            "1e-6",
            "--component-budget",
            "1.0",
            "--output",
            str(output_path),
            "--device",
            "cpu",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["reject"] is True
    assert result["reject_reason"] == "ood_statistics"
    assert result["threshold"] is None
    assert result["selected_thresholds"] == [None]
    assert result["row_results"][0]["threshold"] is None
    assert result["ood_audit"]["is_ood"] is True
    assert result["predicted_pixel_log_risk"] is None
    assert result["threshold_indices_by_image"] is None
    assert result["protocol"] is None
    assert result["protocol_fingerprint"] is None
    assert result["score_map_provenance"] is None

    batch_path = tmp_path / "deployment-batch.npz"
    np.savez_compressed(
        batch_path,
        statistics=np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        statistics_names=np.asarray(["feature_a", "feature_b"]),
        statistics_schema_version=np.asarray(STATISTICS_SCHEMA_VERSION),
        evaluation_ids=np.asarray(
            [json.dumps(["cal-a", "cal-b"]), json.dumps(["test-a"])]
        ),
        adaptation_ids=np.asarray(
            [json.dumps(["warm-a"]), json.dumps(["warm-b"])]
        ),
    )
    batch_output_path = tmp_path / "zero-batch.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "risk_curve.select_zero_label_threshold",
            "--statistics-file",
            str(batch_path),
            "--curve-checkpoint",
            str(checkpoint_path),
            "--pixel-budget",
            "1e12",
            "--component-budget",
            "1e12",
            "--output",
            str(batch_output_path),
            "--device",
            "cpu",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    batch_result = json.loads(batch_output_path.read_text(encoding="utf-8"))
    assert batch_result["num_windows"] == 2
    assert batch_result["threshold"] is None
    assert batch_result["threshold_index"] is None
    assert batch_result["threshold_indices"] == [0, 0]
    assert batch_result["threshold_indices_by_image"] == {
        "cal-a": 0,
        "cal-b": 0,
        "test-a": 0,
    }
    assert batch_result["rejects"] == [False, False]
    assert batch_result["reject"] is False
    assert batch_result["reject_rate"] == 0.0
    assert len(batch_result["ood_audits"]) == 2
    assert batch_result["window_ids"] == ["warm-a", "warm-b"]
    assert batch_result["evaluation_ids"] == [
        ["cal-a", "cal-b"],
        ["test-a"],
    ]
    assert len(batch_result["predicted_pixel_log_risk"]) == 2
