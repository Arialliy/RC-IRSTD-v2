from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evaluation import raw_logit_policy_oracle as policy_oracle
from evaluation.raw_logit_oracle import (
    RawLogitSample,
    select_exact_global_oracle,
)


def _sample(
    image_id: str,
    *,
    target_logit: float = 5.0,
    false_logit: float = -5.0,
) -> RawLogitSample:
    logits = np.full((4, 4), -10.0, dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[2, 2] = 1
    logits[2, 2] = np.float32(target_logit)
    logits[0, 0] = np.float32(false_logit)
    probability = np.asarray(1.0 / (1.0 + np.exp(-logits)), dtype=np.float32)
    return RawLogitSample(image_id, logits, probability, mask)


def test_global_policy_is_identical_to_exact_global_selector() -> None:
    samples = [
        _sample("a", target_logit=8.0, false_logit=6.0),
        _sample("b", target_logit=7.0, false_logit=9.0),
        _sample("c", target_logit=5.0, false_logit=-5.0),
    ]
    expected = select_exact_global_oracle(
        samples,
        pixel_budget=0.03,
        component_budget=30_000.0,
    )["operating_point"]
    result = policy_oracle.evaluate_exact_raw_logit_policy(
        samples,
        policy="global",
        pixel_budget=0.03,
        component_budget=30_000.0,
    )
    unit = result["units"][0]
    for field in (
        "threshold_logit",
        "threshold_probability_float64",
        "threshold_probability_float32",
        "pd",
        "fa_pixel",
        "fa_component_mp",
        "tp_objects",
        "gt_objects",
        "fp_components",
        "fp_pixels",
        "total_pixels",
    ):
        assert unit[field] == expected[field]
    assert result["aggregate"]["tp_objects"] == expected["tp_objects"]
    assert result["coverage"]["evaluated_over_total"] == "3/3"


def test_static_exact_selector_receives_query_samples_and_never_adaptation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = [_sample(f"sample-{index:02d}") for index in range(10)]
    calls: list[list[str]] = []
    original = policy_oracle.select_exact_global_oracle

    def _record(query_samples: list[RawLogitSample], **kwargs: object) -> dict:
        calls.append([sample.image_id for sample in query_samples])
        return original(query_samples, **kwargs)

    monkeypatch.setattr(policy_oracle, "select_exact_global_oracle", _record)
    result = policy_oracle.evaluate_exact_raw_logit_policy(
        samples,
        policy="static",
        pixel_budget=0.01,
        component_budget=10_000.0,
        folds=5,
        seed=42,
        adaptation_window=4,
    )

    assert calls == [unit["evaluation_ids"] for unit in result["units"]]
    assert result["policy"]["evidence_role"] == (
        "primary_exact_raw_logit_policy_matched_oracle_evidence"
    )
    assert result["policy"]["exactly_once_query_coverage"] is True
    assert result["policy"]["query_folds_pairwise_disjoint"] is True
    assert result["coverage"]["evaluated_over_total"] == "10/10"
    assert sorted(image_id for call in calls for image_id in call) == sorted(
        sample.image_id for sample in samples
    )
    assert all(
        unit["selection_input_image_ids"] == unit["evaluation_ids"]
        and unit["adaptation_labels_used_for_selection"] is False
        and not unit["adaptation_evaluation_role_overlap_ids"]
        for unit in result["units"]
    )


def test_default_static_contract_is_five_fold_seed42_a32_primary_evidence() -> None:
    samples = [_sample(f"sample-{index:02d}") for index in range(40)]
    result = policy_oracle.evaluate_exact_raw_logit_policy(
        samples,
        pixel_budget=0.01,
        component_budget=10_000.0,
    )

    assert result["policy"]["name"] == "static"
    assert result["policy"]["folds"] == 5
    assert result["policy"]["seed"] == 42
    assert result["policy"]["adaptation_window"] == 32
    assert result["policy"]["selected_adaptation_sizes"] == [32] * 5
    assert result["policy"]["evidence_role"] == (
        "primary_exact_raw_logit_policy_matched_oracle_evidence"
    )
    assert result["coverage"]["evaluated_over_total"] == "40/40"
    assert result["coverage"]["aggregate_pd_is_full_score_artifact_pd"] is True


def test_each_unit_satisfies_both_budgets_and_aggregate_uses_raw_counts() -> None:
    samples = [
        _sample("recoverable", target_logit=8.0, false_logit=5.0),
        _sample("inseparable", target_logit=5.0, false_logit=8.0),
    ]
    result = policy_oracle.evaluate_exact_raw_logit_policy(
        samples,
        policy="image",
        pixel_budget=0.01,
        component_budget=10_000.0,
    )

    assert [unit["tp_objects"] for unit in result["units"]] == [1, 0]
    assert [unit["fp_pixels"] for unit in result["units"]] == [0, 0]
    assert result["all_units_individually_budget_satisfied"] is True
    assert all(
        unit["budget_feasibility"]["pixel_budget_satisfied"]
        and unit["budget_feasibility"]["component_budget_satisfied"]
        and unit["budget_feasibility"]["joint_budget_satisfied"]
        for unit in result["units"]
    )
    assert result["aggregate"]["tp_objects"] == 1
    assert result["aggregate"]["gt_objects"] == 2
    assert result["aggregate"]["pd"] == pytest.approx(0.5)
    assert result["aggregate"]["fp_pixels"] == 0
    assert result["aggregate"]["fp_components"] == 0


def test_default_causal_contract_reports_exact_six_of_214_coverage() -> None:
    samples = [_sample(f"frame-{index:03d}") for index in range(214)]
    result = policy_oracle.evaluate_exact_raw_logit_policy(
        samples,
        policy="causal",
        pixel_budget=0.01,
        component_budget=10_000.0,
    )

    assert result["policy"]["adaptation_window"] == 32
    assert result["policy"]["evaluation_window"] == 1
    assert result["policy"]["stride"] == 33
    assert result["policy"]["num_complete_causal_blocks"] == 6
    assert result["policy"]["global_adaptation_evaluation_role_overlap"] == []
    assert result["coverage"]["num_score_images"] == 214
    assert result["coverage"]["num_evaluated_images"] == 6
    assert result["coverage"]["num_unevaluated_images"] == 208
    assert result["coverage"]["evaluated_over_total"] == "6/214"
    assert result["coverage"]["evaluated_fraction"] == pytest.approx(6 / 214)
    assert result["coverage"]["aggregate_pd_is_full_score_artifact_pd"] is False
    assert [unit["evaluation_indices"] for unit in result["units"]] == [
        [32],
        [65],
        [98],
        [131],
        [164],
        [197],
    ]


def test_v3_payload_is_diagnostic_only_and_exposes_complete_hash_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    samples = [_sample(f"target-{index}") for index in range(5)]
    manifest = {
        "schema_version": 3,
        "records": [
            {
                "image_id": sample.image_id,
                "file": f"sample-{index}.npz",
                "sha256": f"{index + 1:x}" * 64,
            }
            for index, sample in enumerate(samples)
        ],
    }
    contract = {
        "score_manifest_sha256": "a" * 64,
        "score_records_sha256": "b" * 64,
        "score_ordered_image_ids_sha256": "c" * 64,
        "detector_weight_sha256": "d" * 64,
        "split_file_sha256": "e" * 64,
        "split_ordered_ids_sha256": "f" * 64,
        "checkpoint_selection_rule": "fixed_last",
        "model_backend": "canonical",
        "target_dataset": "target-domain",
        "source_datasets": ["source-domain"],
        "requested_split": "test",
        "split_role": "test",
        "score_representation": (
            "raw_logit_float32+sigmoid_probability_float32"
        ),
        "probability_dtype": "float32",
        "logit_dtype": "float32",
        "probability_transform": "sigmoid",
        "probability_clipping": "none",
        "inference_autocast_enabled": False,
    }
    monkeypatch.setattr(
        policy_oracle,
        "load_formal_raw_logit_directory",
        lambda _score_dir: (samples, manifest, {"verified": True}, contract),
    )
    payload = policy_oracle.build_raw_logit_policy_oracle_payload(
        tmp_path / "scores",
        policy="static",
        pixel_budget=0.01,
        component_budget=10_000.0,
        adaptation_window=3,
    )

    assert payload["diagnostic_only"] is True
    assert payload["oracle_only"] is True
    assert payload["test_labels_used_for_threshold_selection"] is True
    assert payload["adaptation_labels_used_for_threshold_selection"] is False
    assert payload["formal_protocol_eligible"] is False
    assert payload["deployment_threshold_eligible"] is False
    hashes = payload["provenance"]["hashes"]
    assert hashes["score_manifest_sha256"] == "a" * 64
    assert hashes["score_records_sha256"] == "b" * 64
    assert hashes["score_ordered_image_ids_sha256"] == "c" * 64
    assert hashes["detector_weight_sha256"] == "d" * 64
    assert hashes["split_file_sha256"] == "e" * 64
    assert hashes["split_ordered_ids_sha256"] == "f" * 64
    assert len(hashes["raw_logit_stream_sha256"]) == 64
    assert len(hashes["decision_partition_sha256"]) == 64
    assert len(hashes["diagnostic_configuration_sha256"]) == 64
    assert len(hashes["record_file_sha256_by_image_id"]) == 5


def test_cli_requires_explicit_oracle_acknowledgement() -> None:
    with pytest.raises(ValueError, match="--oracle-diagnostic"):
        policy_oracle.main(
            [
                "--score-dir",
                "unused",
                "--output",
                "unused.json",
                "--pixel-budget",
                "1e-5",
                "--component-budget",
                "5",
            ]
        )
