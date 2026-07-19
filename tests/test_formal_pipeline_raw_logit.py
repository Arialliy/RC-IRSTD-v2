"""Regression tests for the formal raw-logit v4 pipeline wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from rc_irstd.cli.run_pipeline import (
    FORMAL_DETECTOR_BACKENDS,
    PROBABILITY_REPRESENTATION,
    RAW_LOGIT_REPRESENTATION,
    RISK_CURVE_METHOD,
    _append_risk_export_contract,
    _detector_initializer_contract,
    _formal_outer_target_name,
    _load_and_validate_method_contract,
    _risk_representation,
)


ROOT = Path(__file__).resolve().parents[1]


def test_formal_backends_and_representation_fail_closed() -> None:
    assert FORMAL_DETECTOR_BACKENDS == {"canonical", "rc_mshnet"}
    assert _risk_representation({}, diagnostic_only=False) == (
        RAW_LOGIT_REPRESENTATION
    )
    with pytest.raises(ValueError, match="raw_logit_float32"):
        _risk_representation(
            {"representation": PROBABILITY_REPRESENTATION},
            diagnostic_only=False,
        )
    assert _risk_representation(
        {"representation": PROBABILITY_REPRESENTATION}, diagnostic_only=True
    ) == PROBABILITY_REPRESENTATION


def test_risk_exports_have_explicit_raw_logit_and_label_modes() -> None:
    unlabeled = ["python", "export_scores.py"]
    _append_risk_export_contract(
        unlabeled,
        representation=RAW_LOGIT_REPRESENTATION,
        labels_loaded=False,
    )
    assert unlabeled[-2:] == ["--export-raw-logits", "--no-labels-loaded"]

    labeled = ["python", "export_scores.py"]
    _append_risk_export_contract(
        labeled,
        representation=RAW_LOGIT_REPRESENTATION,
        labels_loaded=True,
    )
    assert labeled[-2:] == ["--export-raw-logits", "--labels-loaded"]


def test_formal_method_contract_is_executable_and_detects_dead_field_drift() -> None:
    pipeline = {
        "_config_dir": str(ROOT / "configs"),
        "method_contract": "rc_v2_aaai27_main.yaml",
        "target_stage_separation": {
            "selection_scores": "scores_unlabeled",
            "selection_labels_loaded": False,
            "selection_freeze_required": True,
            "labeled_audit_scores": "scores_labeled_audit",
            "labeled_audit_after_freeze": True,
        },
    }
    result = _load_and_validate_method_contract(
        pipeline,
        method_name=RISK_CURVE_METHOD,
        detector_backend="rc_mshnet",
        representation=RAW_LOGIT_REPRESENTATION,
        diagnostic_only=False,
    )
    assert result is not None
    assert result["validated_for_formal"] is True
    assert len(result["sha256"]) == 64

    drifted = dict(pipeline)
    drifted["target_stage_separation"] = {
        **pipeline["target_stage_separation"],
        "selection_labels_loaded": True,
    }
    with pytest.raises(ValueError, match="differs from method_contract"):
        _load_and_validate_method_contract(
            drifted,
            method_name=RISK_CURVE_METHOD,
            detector_backend="rc_mshnet",
            representation=RAW_LOGIT_REPRESENTATION,
            diagnostic_only=False,
        )


def test_formal_rc_mshnet_requires_complete_fold_initializers(tmp_path: Path) -> None:
    sources = [
        {"name": "source-a", "path": "/data/a"},
        {"name": "source-b", "path": "/data/b"},
    ]
    with pytest.raises(ValueError, match="detector_initializers"):
        _detector_initializer_contract(
            {"_config_dir": str(tmp_path)},
            sources,
            detector_backend="rc_mshnet",
            diagnostic_only=False,
        )

    for name in ("outer.pt", "inner-a.pt", "inner-b.pt"):
        (tmp_path / name).write_bytes(name.encode("utf-8"))
    pipeline = {
        "_config_dir": str(tmp_path),
        "detector_initializers": {
            "outer_final": "outer.pt",
            "inner_by_held_out": {
                "source-a": "inner-a.pt",
                "source-b": "inner-b.pt",
            },
        },
    }
    resolved = _detector_initializer_contract(
        pipeline,
        sources,
        detector_backend="rc_mshnet",
        diagnostic_only=False,
    )
    assert resolved is not None
    assert len(resolved["outer_final"]["sha256"]) == 64
    assert set(resolved["inner_by_held_out"]) == {"source-a", "source-b"}


def test_raw_logit_grid_uses_self_scores_and_precedes_curve_training() -> None:
    runner = (ROOT / "rc_irstd" / "cli" / "run_pipeline.py").read_text(
        encoding="utf-8"
    )
    final_detector = runner.index("final_detector = _require_fixed_last_checkpoint")
    grid_builder = runner.index(
        '"risk_curve.build_logit_threshold_grid"', final_detector
    )
    curve_builder = runner.index('"risk_curve.build_curve_episodes"', grid_builder)
    curve_training = runner.index(
        '"risk_curve.train_curve_predictor"', curve_builder
    )
    assert final_detector < grid_builder < curve_builder < curve_training
    assert '/ "grid_source_scores"\n                    / "inner"' in runner
    assert '/ "grid_source_scores"\n                / "outer_final"' in runner
    assert 'for score_dir in grid_source_score_dirs:' in runner
    assert 'for source, score_dir in zip(sources, pseudo_score_dirs):' in runner
    assert '"--threshold-grid-manifest"' in runner


def test_grid_outer_target_is_unambiguous() -> None:
    target = [{"name": "NUAA-SIRST", "path": "/data/nuaa"}]
    assert _formal_outer_target_name({}, target) == "NUAA-SIRST"
    with pytest.raises(ValueError, match="exactly one final target"):
        _formal_outer_target_name({}, [])
