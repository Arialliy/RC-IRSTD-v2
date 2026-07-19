from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from risk_curve.build_curve_episodes import COMPONENT_RISK_SCHEMA_VERSION
from risk_curve.curve_dataset import LOGIT_EPISODE_SCHEMA_VERSION
from risk_curve.direct_calibrator import ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL
from risk_curve.domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    feature_schema_sha256,
    statistics_names_sha256,
)
from risk_curve.evaluate_dynamic_budget_oracle_v4 import (
    DYNAMIC_BUDGET_ORACLE_SCHEMA_VERSION,
    evaluate_dynamic_budget_oracle,
    solve_multiple_choice_2d_knapsack,
)
from risk_curve.representation import (
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_REPRESENTATION,
    logit_threshold_grid_sha256,
)


GRID = np.asarray([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
GRID_HASH = logit_threshold_grid_sha256(GRID)
NAMES = ("logit_feature_a", "logit_feature_b")
FEATURE_HASH = feature_schema_sha256(
    LOGIT_STATISTICS_SCHEMA_VERSION, statistics_names=NAMES
)
MANIFEST_HASH = "a" * 64
OUTER_HASH = "1" * 64
INNER_HASHES = ("2" * 64, "3" * 64)
ALL_HASHES = (OUTER_HASH, *INNER_HASHES)


def _provenance() -> dict[str, object]:
    return {
        "archive_split": "validation",
        "protocol": "causal_adaptation_then_future_evaluation",
        "representation": LOGIT_REPRESENTATION,
        "adaptation_window": 1,
        "evaluation_window": 1,
        "stride": 2,
        "pseudo_targets": ["NUDT-SIRST", "IRSTD-1K"],
        "validation_domain": "IRSTD-1K",
        "threshold_grid_source_domains": ["irstd1k", "nudt"],
        "paired_lodo_validation_domains": ["IRSTD-1K", "NUDT-SIRST"],
        "pseudo_target_split": "train",
        "expected_split_role": "train",
        "threshold_grid_outer_target_key": "NUAA-SIRST",
        "threshold_grid_outer_target_excluded": True,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": GRID_HASH,
        "threshold_grid_manifest_sha256": MANIFEST_HASH,
        "threshold_grid_detector_protocol": ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": list(ALL_HASHES),
        "threshold_grid_outer_detector_checkpoint_sha256": OUTER_HASH,
        "threshold_grid_episode_detector_checkpoint_sha256s": list(INNER_HASHES),
        "fold_provenance_audits": [
            {
                "verified": True,
                "pseudo_target": "NUDT-SIRST",
                "detector_weight_sha256": INNER_HASHES[0],
            },
            {
                "verified": True,
                "pseudo_target": "IRSTD-1K",
                "detector_weight_sha256": INNER_HASHES[1],
            },
        ],
        "feature_schema_sha256": FEATURE_HASH,
        "fold_provenance_verified": True,
        "allow_unverified_fold_provenance": False,
        "allow_cross_episode_role_reuse": False,
        "cross_episode_role_reuse_detected": False,
        "cross_episode_role_reuse_ids": [],
        "formal_causal_contract_verified": True,
        "protocol_scope": "formal_causal",
        "statistics_sample_role": "adaptation_window_A_label_free",
        "risk_label_sample_role": "immediately_following_evaluation_window_E",
    }


def _write_archive(path: Path) -> None:
    rows = 3
    pixel_fp = np.asarray(
        [[20, 5, 1, 0], [24, 6, 2, 0], [22, 4, 1, 0]], dtype=np.int64
    )
    component_fp = np.asarray(
        [[3, 2, 1, 0], [4, 2, 1, 0], [3, 2, 1, 0]], dtype=np.int64
    )
    tp = np.asarray(
        [[5, 4, 2, 0], [4, 3, 2, 0], [6, 5, 2, 0]], dtype=np.int64
    )
    gt = np.asarray([5, 4, 6], dtype=np.int64)
    total_pixels = np.full(rows, 1_000_000, dtype=np.int64)
    pixel_log = np.log10(
        pixel_fp / total_pixels[:, None] + 1e-12
    ).astype(np.float32)
    component_raw = np.log10(
        component_fp / (total_pixels[:, None] / 1_000_000.0) + 1e-6
    ).astype(np.float32)
    component_upper = np.maximum.accumulate(
        component_raw[:, ::-1], axis=1
    )[:, ::-1]
    pd_curve = (tp / np.maximum(gt[:, None], 1)).astype(np.float32)
    np.savez_compressed(
        path,
        statistics=np.asarray(
            [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]], dtype=np.float32
        ),
        statistics_names=np.asarray(NAMES),
        statistics_names_sha256=np.asarray(statistics_names_sha256(NAMES)),
        statistics_schema_version=np.asarray(LOGIT_STATISTICS_SCHEMA_VERSION),
        feature_schema_sha256=np.asarray(FEATURE_HASH),
        pixel_log_risk=pixel_log,
        component_log_risk=component_upper,
        component_log_risk_raw=component_raw,
        component_log_risk_upper=component_upper,
        component_risk_schema_version=np.asarray(COMPONENT_RISK_SCHEMA_VERSION),
        component_log_risk_alias=np.asarray("component_log_risk_upper"),
        pd_curve=pd_curve,
        thresholds=GRID,
        pixel_fp_counts=pixel_fp,
        component_fp_counts=component_fp,
        tp_object_counts=tp,
        gt_object_counts=gt,
        total_pixels=total_pixels,
        representation=np.asarray(LOGIT_REPRESENTATION),
        threshold_grid_schema_version=np.asarray(LOGIT_GRID_SCHEMA_VERSION),
        threshold_grid_sha256=np.asarray(GRID_HASH),
        threshold_grid_manifest_sha256=np.asarray(MANIFEST_HASH),
        threshold_grid_detector_protocol=np.asarray(
            ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL
        ),
        threshold_grid_detector_checkpoint_sha256s=np.asarray(ALL_HASHES),
        threshold_grid_outer_detector_checkpoint_sha256=np.asarray(OUTER_HASH),
        threshold_grid_episode_detector_checkpoint_sha256s=np.asarray(INNER_HASHES),
        episode_schema_version=np.asarray(LOGIT_EPISODE_SCHEMA_VERSION),
        adaptation_sizes=np.ones(rows, dtype=np.int64),
        evaluation_sizes=np.ones(rows, dtype=np.int64),
        adaptation_ids=np.asarray(
            [json.dumps([f"adapt-{index}"]) for index in range(rows)]
        ),
        evaluation_ids=np.asarray(
            [json.dumps([f"eval-{index}"]) for index in range(rows)]
        ),
        pseudo_targets=np.asarray(["IRSTD-1K"] * rows),
        provenance_json=np.asarray(json.dumps(_provenance(), sort_keys=True)),
    )


def _rewrite(path: Path, **changes: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as archive:
        payload = {field: archive[field] for field in archive.files}
    payload.update(changes)
    np.savez_compressed(path, **payload)


def _brute_force(
    pixel: np.ndarray,
    component: np.ndarray,
    tp: np.ndarray,
    pixel_capacity: int,
    component_capacity: int,
) -> tuple[int, int, int, tuple[int, ...]]:
    rows, grid_size = pixel.shape
    candidates: list[tuple[int, int, int, tuple[int, ...]]] = []
    for path in itertools.product(range(grid_size + 1), repeat=rows):
        pixel_used = sum(
            0 if index == grid_size else int(pixel[row, index])
            for row, index in enumerate(path)
        )
        component_used = sum(
            0 if index == grid_size else int(component[row, index])
            for row, index in enumerate(path)
        )
        reward = sum(
            0 if index == grid_size else int(tp[row, index])
            for row, index in enumerate(path)
        )
        if pixel_used <= pixel_capacity and component_used <= component_capacity:
            candidates.append((-reward, pixel_used, component_used, path))
    best = min(candidates)
    return -best[0], best[1], best[2], best[3]


def test_exact_dp_matches_complete_brute_force_and_tie_break() -> None:
    pixel = np.asarray([[3, 1, 0], [4, 2, 0], [3, 1, 0]], dtype=np.int64)
    component = np.asarray([[2, 1, 0], [1, 1, 0], [2, 1, 0]], dtype=np.int64)
    tp = np.asarray([[5, 3, 0], [6, 4, 0], [5, 3, 0]], dtype=np.int64)
    expected = _brute_force(pixel, component, tp, 5, 3)
    solution = solve_multiple_choice_2d_knapsack(
        pixel,
        component,
        tp,
        pixel_capacity=5,
        component_capacity=3,
    )
    assert (
        solution.total_true_positive,
        solution.used_pixel_fp,
        solution.used_component_fp,
        solution.action_ranks,
    ) == expected


def test_exact_dp_matches_randomised_complete_enumeration() -> None:
    """Exercise dominance pruning and all deterministic tie-break levels."""

    generator = np.random.default_rng(20260715)
    for _case in range(96):
        rows = int(generator.integers(1, 5))
        grid_size = int(generator.integers(2, 6))
        pixel = generator.integers(0, 7, size=(rows, grid_size), dtype=np.int64)
        component = generator.integers(
            0, 5, size=(rows, grid_size), dtype=np.int64
        )
        tp = generator.integers(0, 9, size=(rows, grid_size), dtype=np.int64)
        pixel_capacity = int(generator.integers(0, 11))
        component_capacity = int(generator.integers(0, 8))
        expected = _brute_force(
            pixel,
            component,
            tp,
            pixel_capacity,
            component_capacity,
        )
        solution = solve_multiple_choice_2d_knapsack(
            pixel,
            component,
            tp,
            pixel_capacity=pixel_capacity,
            component_capacity=component_capacity,
        )
        assert (
            solution.total_true_positive,
            solution.used_pixel_fp,
            solution.used_component_fp,
            solution.action_ranks,
        ) == expected


def test_evaluator_emits_exact_diagnostic_bound_and_capacity_evidence(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "val.npz"
    output = tmp_path / "oracle.json"
    _write_archive(archive)
    evaluate_dynamic_budget_oracle(episode_file=archive, output=output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == DYNAMIC_BUDGET_ORACLE_SCHEMA_VERSION
    assert payload["diagnostic_only"] is True
    assert payload["deployment_eligible"] is False
    assert payload["selector_eligible"] is False
    assert payload["must_not_be_used_by_selector"] is True
    assert payload["labels_policy"][
        "source_pseudo_target_future_E_labels_used_for_optimization"
    ] is True
    assert payload["labels_policy"]["action_selection_is_label_free"] is False
    assert payload["labels_policy"]["outer_target_labels_used"] is False
    assert payload["episode_archive_sha256"] == hashlib.sha256(
        archive.read_bytes()
    ).hexdigest()
    assert [item["budget_name"] for item in payload["budgets"]] == [
        "loose",
        "strict",
    ]
    for item in payload["budgets"]:
        assert item["optimality"]["exact"] is True
        assert item["aggregate"]["joint_budget_satisfied"] is True
        assert len(item["actions"]) == 3
        assert all(
            action["reject"] or action["threshold_index"] is not None
            for action in item["actions"]
        )
        used = item["used_capacities"]
        assert used["pixel_fp_used"] == sum(
            action["pixel_fp_count"] for action in item["actions"]
        )
        assert used["component_fp_used"] == sum(
            action["component_fp_count"] for action in item["actions"]
        )


def test_evaluator_fails_closed_on_semantic_grid_hash_tamper(tmp_path: Path) -> None:
    archive = tmp_path / "val.npz"
    _write_archive(archive)
    _rewrite(archive, threshold_grid_sha256=np.asarray("f" * 64))
    with pytest.raises(ValueError, match="threshold_grid_sha256"):
        evaluate_dynamic_budget_oracle(
            episode_file=archive, output=tmp_path / "oracle.json"
        )


def test_evaluator_fails_closed_when_outer_enters_source_scope(tmp_path: Path) -> None:
    archive = tmp_path / "val.npz"
    _write_archive(archive)
    provenance = _provenance()
    provenance["threshold_grid_outer_target_key"] = "IRSTD-1K"
    _rewrite(
        archive,
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    with pytest.raises(ValueError, match="excluded outer target"):
        evaluate_dynamic_budget_oracle(
            episode_file=archive, output=tmp_path / "oracle.json"
        )


def test_evaluator_rejects_outer_alias_swap_and_nuaa_rows(tmp_path: Path) -> None:
    archive = tmp_path / "val.npz"
    _write_archive(archive)
    provenance = _provenance()
    provenance["pseudo_targets"] = ["NUDT-SIRST", "NUAA-SIRST"]
    provenance["validation_domain"] = "NUAA-SIRST"
    provenance["threshold_grid_outer_target_key"] = "Other-SIRST"
    _rewrite(
        archive,
        pseudo_targets=np.asarray(["NUAA-SIRST"] * 3),
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    with pytest.raises(ValueError, match="source domains must be exactly"):
        evaluate_dynamic_budget_oracle(
            episode_file=archive, output=tmp_path / "oracle.json"
        )


@pytest.mark.parametrize("archive_split", [None, "train"])
def test_evaluator_requires_explicit_validation_archive_split(
    tmp_path: Path, archive_split: str | None
) -> None:
    archive = tmp_path / "val.npz"
    _write_archive(archive)
    provenance = _provenance()
    if archive_split is None:
        provenance.pop("archive_split")
    else:
        provenance["archive_split"] = archive_split
    _rewrite(
        archive,
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    with pytest.raises(ValueError, match="formal validation archive"):
        evaluate_dynamic_budget_oracle(
            episode_file=archive, output=tmp_path / "oracle.json"
        )


def test_evaluator_fails_closed_on_fractional_count_tamper(tmp_path: Path) -> None:
    archive = tmp_path / "val.npz"
    _write_archive(archive)
    with np.load(archive, allow_pickle=False) as source:
        pixel = source["pixel_fp_counts"].astype(np.float64)
    pixel[0, 0] += 0.5
    _rewrite(archive, pixel_fp_counts=pixel)
    with pytest.raises(ValueError, match="integer counts"):
        evaluate_dynamic_budget_oracle(
            episode_file=archive, output=tmp_path / "oracle.json"
        )
