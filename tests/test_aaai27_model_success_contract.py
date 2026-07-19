from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "configs" / "aaai27_model_success_contract_v1.json"


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


@pytest.fixture(scope="module")
def contract() -> dict:
    payload = json.loads(
        CONTRACT_PATH.read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_constant,
    )
    assert isinstance(payload, dict)
    return payload


def test_contract_is_versioned_frozen_success_criteria_that_cannot_authorize_training(
    contract: dict,
) -> None:
    assert contract["schema_version"] == "rc-irstd-aaai27-model-success-contract-v1"
    assert contract["contract_id"] == "aaai27_model_success_contract_v1"
    assert contract["contract_version"] == 1

    lifecycle = contract["lifecycle"]
    assert lifecycle["state"] == "reviewed_frozen_success_criteria"
    assert lifecycle["formal_training_authorized_by_this_file"] is False
    assert lifecycle["registration_envelope_and_sha256_required"] is True
    assert lifecycle["seed_freshness_ledger_must_be_frozen_at_registration"] is True
    assert lifecycle["success_criteria_freeze_does_not_authorize_model_training"] is True
    assert (
        lifecycle["formal_model_training_requires_separate_versioned_v3_preregistration"]
        is True
    )
    assert lifecycle["threshold_changes_require_a_new_version_before_training"] is True


def test_success_requires_publication_targets_but_stretch_is_optional(
    contract: dict,
) -> None:
    success = contract["success_semantics"]
    assert success["source_gate_go"] == (
        "all_hard_and_all_applicable_conditional_hard_requirements_pass"
    )
    assert success["aaai27_submission_performance_success"] == (
        "source_gate_go_and_all_publication_targets_pass"
    )
    assert success["stretch_required_for_submission_performance_success"] is False
    assert success["performance_success_is_necessary_but_not_sufficient_for_submission_readiness"] is True
    assert success["acceptance_guaranteed"] is False


def test_source_only_raw_logit_baseline_and_historical_hold_are_preserved(
    contract: dict,
) -> None:
    scope = contract["scope"]
    assert scope["design_and_source_confirmation_domains"] == [
        "NUDT-SIRST",
        "IRSTD-1K",
    ]
    assert scope["forbidden_outer_domain_before_separate_authorization"] == "NUAA-SIRST"
    assert scope["source_only"] is True
    assert scope["baseline_identity"] == "canonical_MSHNet_plus_StableSLS"
    assert scope["primary_score_representation"] == "float32_raw_logit"
    assert scope["inference_autocast"] is False
    assert "never_overwrite" in scope["historical_hold_policy"]
    assert contract["protocol_invariants"]["historical_hold_mutation_allowed"] is False


def test_frozen_tier2r_hold_is_anchored_without_being_relabelled(contract: dict) -> None:
    anchor = contract["historical_evidence_anchor"]
    assert anchor["status"] == "frozen_failure_evidence_not_a_success_precedent"
    assert anchor["protocol_sha256"] == (
        "72e799849f3552c871156514e35e0af0458ed76b6b8b8186f3f3ea8a280c32ba"
    )
    assert anchor["decision_sha256"] == (
        "a1c8193ebbf3d2b9cc9cf35f7921516fb2e8df44c9ff75596e5b4df50928be6b"
    )
    assert anchor["decision"] == "TIER2R_HOLD"
    assert (anchor["gate_a_pass_count"], anchor["gate_a_num_criteria"]) == (6, 9)
    assert (anchor["gate_b_pass_count"], anchor["gate_b_num_criteria"]) == (4, 9)
    assert anchor["selected_candidate"] is None
    assert anchor["component_claim_retained"] is False
    assert anchor["source_tier3_design_authorized"] is False
    assert anchor["outer_target_access_authorized"] is False
    assert anchor["new_contract_may_not_overwrite_relabel_or_supersede_this_evidence"] is True

    protocol_path = PROJECT_ROOT / anchor["protocol_path"]
    decision_path = PROJECT_ROOT / anchor["decision_path"]
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == anchor["protocol_sha256"]
    assert hashlib.sha256(decision_path.read_bytes()).hexdigest() == anchor["decision_sha256"]
    frozen = json.loads(
        decision_path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_constant,
    )
    gate_a = frozen["levels"]["contrast_vs_control"]
    gate_b = frozen["levels"]["component_context_vs_contrast"]
    assert sum(row["passed"] is True for row in gate_a["criteria"]) == 6
    assert sum(row["passed"] is True for row in gate_b["criteria"]) == 4
    assert frozen["decision"] == "TIER2R_HOLD"
    assert frozen["selected_candidate"] is None
    assert frozen["authorizes_outer_target_access"] is False


def test_historical_target_exposure_is_bound_and_never_called_untouched(
    contract: dict,
) -> None:
    exposure = contract["historical_target_exposure"]
    registry_path = PROJECT_ROOT / exposure["registry_path"]
    assert hashlib.sha256(registry_path.read_bytes()).hexdigest() == (
        exposure["registry_sha256"]
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["datasets"]["NUAA-SIRST"]["official_test_labels_previously_viewed"] is True
    assert exposure["nuaa_official_test_labels_previously_viewed"] is True
    assert exposure["nuaa_paper_role"] == "development_domain_not_untouched"
    assert exposure["untouched_claim_allowed"] is False
    assert (
        exposure["nudt_and_irstd_prior_test_label_exposure_requires_author_confirmation"]
        is True
    )
    assert exposure["future_access_freeze_semantics"] == (
        "no_additional_NUAA_feedback_after_governance_freeze_not_never_seen"
    )
    assert exposure["source_gate_go_alone_does_not_authorize_new_NUAA_access"] is True
    assert exposure["strong_blind_generalization_claim_requires_a_new_unexposed_external_target"] is True


def test_novelty_boundary_does_not_promote_posthoc_calibration_to_core(
    contract: dict,
) -> None:
    novelty = contract["novelty_boundary"]
    assert novelty["nearest_neighbor_literature_audit_required_before_formal_preregistration"] is True
    assert novelty["core_mechanism_must_change_raw_score_ordering"] is True
    assert (
        novelty[
            "posthoc_threshold_calibration_simple_scaling_loss_reweighting_or_routine_module_stacking_is_not_a_core_mechanism"
        ]
        is True
    )
    assert novelty["mechanism_must_be_trainable_interpretable_and_falsifiable"] is True
    assert novelty["performance_gate_alone_does_not_establish_novelty"] is True


def test_fresh5_is_exact_paired_and_fail_closed(contract: dict) -> None:
    fresh = contract["fresh5_confirmation"]
    formal = fresh["formal_seed_ids"]
    excluded = fresh["known_excluded_seed_ids"]
    assert fresh["rule_name"] == "fresh5"
    assert fresh["freshness_ledger_path"].endswith("FRESH_SEED_LEDGER.json")
    assert fresh["formal_seeds_reserved_for_future_v3_gate_a_only"] is True
    assert fresh["current_contract_model_scope"] == (
        "one_unique_v3_core_mechanism_with_no_component_extension_arm"
    )
    assert fresh["gate_b_applicable_under_current_contract"] is False
    assert (
        fresh[
            "complete_gate_a_formal_arm_matrix_must_be_frozen_before_any_reserved_seed_is_consumed"
        ]
        is True
    )
    assert fresh["future_component_extension_requires_new_contract_new_ledger_and_new_unseen_seeds"] is True
    assert (
        fresh["future_component_extension_may_not_reuse_46_47_48_49_50_after_any_gate_a_result_is_observed"]
        is True
    )
    assert fresh["tier2s_reuses_only_frozen_seeds_43_44_45_without_training"] is True
    assert formal == [46, 47, 48, 49, 50]
    assert len(formal) == fresh["num_formal_seeds"] == 5
    assert len(set(formal)) == 5
    assert excluded == [42, 43, 44, 45]
    assert set(formal).isdisjoint(excluded)
    assert fresh["exclude_every_seed_in_frozen_diagnostic_or_development_seed_ledger"] is True
    assert fresh["registration_requires_proof_each_formal_seed_is_fresh"] is True
    assert fresh["freshness_conflict_action"] == (
        "fail_closed_and_issue_new_contract_version_before_training"
    )
    assert fresh["single_seed_replacement_after_registration_allowed"] is False
    assert fresh["seed_screening_or_selective_reporting_allowed"] is False
    assert fresh["all_registered_seeds_must_be_run_and_reported"] is True
    assert fresh["paired_within_seed_across_compared_arms"] is True
    assert fresh["source_lodo_folds_per_seed"] == 2
    assert set(fresh["required_fold_coverage"]) == {
        "heldout_NUDT-SIRST",
        "heldout_IRSTD-1K",
    }


def test_freshness_local_scan_and_external_attestation_remain_pending_fail_closed_prerequisites(
    contract: dict,
) -> None:
    fresh = contract["fresh5_confirmation"]
    assert (
        fresh[
            "local_repository_freshness_scan_must_be_frozen_at_governance_registration"
        ]
        is True
    )
    assert (
        fresh[
            "external_or_deleted_artifact_freshness_requires_dated_author_attestation_before_v3_training"
        ]
        is True
    )

    ledger_path = PROJECT_ROOT / fresh["freshness_ledger_path"]
    ledger = json.loads(
        ledger_path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_constant,
    )
    assert ledger["lifecycle"]["formal_model_training_authorized"] is False
    status = ledger["reserved_seed_status"]
    assert status["classification"] == (
        "locally_unconsumed_reserved_pending_external_author_attestation"
    )
    assert status["local_machine_scan_required"] is True
    assert status["local_machine_scan_report_revision"] == 2
    assert status["local_machine_scan_algorithm_version"] == (
        "fresh-seed-local-filesystem-scan-v2"
    )
    scan_path = PROJECT_ROOT / status["local_machine_scan_report_path"]
    scan_sidecar = PROJECT_ROOT / status["local_machine_scan_report_sidecar_path"]
    scan_sha256 = hashlib.sha256(scan_path.read_bytes()).hexdigest()
    assert status["local_machine_scan_report_sha256"] == scan_sha256
    assert hashlib.sha256(scan_sidecar.read_bytes()).hexdigest() == status[
        "local_machine_scan_report_sidecar_sha256"
    ]
    assert scan_sidecar.read_text(encoding="ascii") == (
        f"{scan_sha256}  {scan_path.name}\n"
    )
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    assert scan_path.name == "FRESH_SEED_LOCAL_SCAN_V2.json"
    assert scan["algorithm_version"] == "fresh-seed-local-filesystem-scan-v2"
    assert status["local_machine_scan_decision"] == (
        "PASS_NO_LOCAL_CONSUMPTION_EVIDENCE_FOUND"
    )
    assert scan["decision"] == status["local_machine_scan_decision"]
    assert scan["counts"]["consumption_like_hits"] == 0
    assert scan["freshness_certified"] is False
    assert scan["author_attestation_required"] is True
    assert status["registration_time_current_workspace_rescan_required"] is True
    assert status["repository_external_deleted_artifact_coverage"] is False
    assert status["dated_author_attestation_required_before_v3_training"] is True
    assert status["dated_author_attestation_present"] is False
    assert status["formal_v3_training_eligible"] is False

    revision = ledger["pre_registration_local_scan_revision"]
    assert revision["revision"] == 2
    assert revision["replacement_algorithm_version"] == (
        "fresh-seed-local-filesystem-scan-v2"
    )
    superseded = revision["superseded_candidate"]
    assert superseded["status"] == (
        "REJECTED_PRE_REGISTRATION_NEVER_AUTHORIZED_TIER2S_OR_V3"
    )
    superseded_path = PROJECT_ROOT / superseded["path"]
    assert hashlib.sha256(superseded_path.read_bytes()).hexdigest() == (
        superseded["sha256"]
    )

    conditions = set(contract["fail_closed_conditions"])
    assert {
        "fresh_seed_local_scan_not_frozen_or_any_consumption_like_hit_unresolved",
        "fresh_seed_external_deleted_or_off_repository_author_attestation_missing_before_v3_training",
    }.issubset(conditions)


def test_data_and_metric_integrity_is_group_aware_and_unambiguous(contract: dict) -> None:
    integrity = contract["data_and_metric_integrity"]
    assert integrity["source_split_manifests_and_ordered_id_sha256_must_be_frozen"] is True
    assert integrity["perceptual_near_duplicate_and_sequence_scene_group_audit_required_before_v3_training"] is True
    assert integrity["group_leakage_across_fit_selection_and_evaluation_partitions_allowed"] is False
    assert "group_not_individual_image" in integrity["cross_fit_partition_unit"]
    assert integrity["primary_object_matching_rule"] == "one_to_one_any_overlap"
    assert integrity["required_matching_sensitivity"] == "one_to_one_centroid_distance_at_most_3_pixels"
    assert integrity["component_connectivity"] == 2
    assert integrity["formal_min_component_area"] == 1
    assert integrity["component_area_filter_may_not_change_pixel_false_alarm_counts"] is True
    assert "dataset_global_binary" in integrity["mIoU_definition"]
    assert "per_image_binary" in integrity["nIoU_definition"]


def test_dual_false_alarm_budgets_are_exact(contract: dict) -> None:
    actual = {
        row["name"]: (
            row["pixel_fa_per_pixel_max"],
            row["component_fa_per_million_pixels_max"],
        )
        for row in contract["false_alarm_budgets"]
    }
    assert actual == {
        "strict": (1.0e-6, 1.0),
        "medium": (5.0e-6, 5.0),
        "loose": (1.0e-5, 10.0),
    }


def _evaluate_fresh5_criterion(
    template: dict,
    criterion: dict,
    seed_deltas: list[float] | None,
) -> bool | str:
    if seed_deltas is None or len(seed_deltas) != 5:
        return "HOLD"
    try:
        deltas = [float(value) for value in seed_deltas]
    except (TypeError, ValueError):
        return "HOLD"
    if not all(math.isfinite(value) for value in deltas):
        return "HOLD"

    atol = float(template["numeric_atol"])
    mean_pass = sum(deltas) / 5 >= float(criterion["paired_mean_delta_min"]) - atol
    worst_seed_pass = min(deltas) >= (
        float(template["every_criterion_worst_seed_delta_min"]) - atol
    )
    consensus_pass = True
    if criterion["id"] == "strict_macro_pd":
        consensus_pass = sum(value > atol for value in deltas) >= int(
            template["strict_macro_positive_seed_min"]
        )
    return mean_pass and worst_seed_pass and consensus_pass


def test_gate_a_keeps_all_nine_frozen_criteria(contract: dict) -> None:
    hard = contract["tiers"]["hard"]
    template = hard["fresh5_nine_criterion_gate_template"]
    assert template["normative_status"] == (
        "newly_preregistered_fresh5_gate_not_a_historical_result"
    )
    criteria = template["criteria"]
    assert template["required_pass_count"] == template["num_criteria"] == 9
    assert [row["id"] for row in criteria] == [
        "strict_macro_pd",
        "strict_domain_pd_nudt",
        "strict_domain_pd_irstd1k",
        "pooled_pd_strict",
        "pooled_pd_medium",
        "pooled_pd_loose",
        "worst_domain_pd_strict",
        "worst_domain_pd_medium",
        "worst_domain_pd_loose",
    ]
    assert criteria[0]["paired_mean_delta_min"] == pytest.approx(0.005)
    assert all(
        row["paired_mean_delta_min"] == pytest.approx(0.0)
        for row in criteria[1:]
    )
    assert all(
        row["operating_point_view"] == "frozen_gate_source_pooled"
        for row in criteria[:6]
    )
    assert all(
        row["operating_point_view"] == "frozen_gate_source_worst"
        for row in criteria[6:]
    )
    assert template["numeric_atol"] == pytest.approx(1.0e-12)
    assert template["paired_seed_aggregation"] == (
        "arithmetic_mean_of_candidate_minus_baseline_per_seed"
    )
    assert template["seed_delta_definition"] == (
        "one_candidate_minus_baseline_delta_per_registered_seed_after_the_criterion_specific_two_fold_source_reduction_no_image_domain_or_threshold_pseudoreplication"
    )
    assert template["paired_mean_delta_definition"] == (
        "arithmetic_mean_of_exactly_five_finite_registered_seed_deltas"
    )
    assert template["mean_pass_operator"] == (
        "paired_mean_delta_greater_than_or_equal_to_criterion_floor_minus_numeric_atol"
    )
    assert template["worst_seed_pass_operator"] == (
        "minimum_of_the_five_seed_deltas_greater_than_or_equal_to_every_criterion_worst_seed_delta_min_minus_numeric_atol"
    )
    assert template["consensus_applicability"] == "strict_macro_pd_only"
    assert template["strict_macro_consensus_pass_operator"] == (
        "count_of_seed_deltas_strictly_greater_than_numeric_atol_greater_than_or_equal_to_strict_macro_positive_seed_min"
    )
    assert template["non_strict_macro_consensus_pass_value"] is True
    assert template["criterion_pass_formula"] == (
        "mean_pass_and_worst_seed_pass_and_applicable_consensus_pass"
    )
    assert template["gate_pass_formula"] == "all_nine_criterion_pass_values_true"
    assert template["missing_fold_seed_arm_metric_or_non_finite_reduction"] == (
        "HOLD_without_partial_aggregation"
    )
    assert template["every_criterion_worst_seed_delta_min"] == pytest.approx(-0.005)
    assert template["strict_macro_positive_seed_min"] == 4
    assert template["strict_macro_positive_consensus_status"] == (
        "new_fresh5_ceil_two_thirds_rule_4_of_5"
    )
    assert template["strict_macro_positive_definition"] == (
        "paired_seed_delta_strictly_greater_than_numeric_atol"
    )
    assert template["missing_infeasible_or_non_finite"] == "HOLD"

    gate_a = hard["gate_a"]
    assert gate_a["name"] == "core_mechanism_vs_canonical_baseline"
    assert gate_a["candidate"] == "new_core_model"
    assert gate_a["required"] is True
    assert gate_a["baseline"] == "canonical_MSHNet_plus_StableSLS"
    assert gate_a["template"] == "fresh5_nine_criterion_gate_template"
    assert gate_a["required_result"] == "9_of_9_GO"
    assert gate_a["evaluated_on_core_model_float32_raw_logits_before_riskcurve"] is True
    assert gate_a["riskcurve_calibration_or_system_gain_may_compensate_gate_a_failure"] is False


def test_fresh5_gate_formula_is_five_seed_four_of_five_and_fail_closed(
    contract: dict,
) -> None:
    hard = contract["tiers"]["hard"]
    template = hard["fresh5_nine_criterion_gate_template"]
    criteria = template["criteria"]
    strict_macro = criteria[0]

    passing_strict = [0.008, 0.008, 0.008, 0.008, -0.005]
    assert _evaluate_fresh5_criterion(
        template, strict_macro, passing_strict
    ) is True
    assert _evaluate_fresh5_criterion(
        template, strict_macro, [0.004] * 5
    ) is False
    assert _evaluate_fresh5_criterion(
        template, strict_macro, [0.012, 0.012, 0.012, 0.0, -0.005]
    ) is False
    assert _evaluate_fresh5_criterion(
        template, strict_macro, [0.01, 0.01, 0.01, 0.01, -0.006]
    ) is False

    passing_by_id = {strict_macro["id"]: passing_strict}
    passing_by_id.update(
        {row["id"]: [0.0] * 5 for row in criteria[1:]}
    )
    criterion_passes = [
        _evaluate_fresh5_criterion(template, row, passing_by_id[row["id"]])
        for row in criteria
    ]
    assert criterion_passes == [True] * 9
    assert all(criterion_passes) is True
    criterion_passes[-1] = False
    assert all(criterion_passes) is False

    assert _evaluate_fresh5_criterion(template, strict_macro, None) == "HOLD"
    assert _evaluate_fresh5_criterion(
        template, strict_macro, passing_strict[:4]
    ) == "HOLD"
    assert _evaluate_fresh5_criterion(
        template, strict_macro, [*passing_strict[:4], math.nan]
    ) == "HOLD"


def test_gate_b_is_currently_inapplicable_and_future_activation_requires_new_unseen_seeds(
    contract: dict,
) -> None:
    gate_b = contract["tiers"]["hard"]["gate_b"]
    fresh = contract["fresh5_confirmation"]
    assert gate_b["name"] == "component_extension_vs_gate_a_core_model"
    assert gate_b["candidate"] == "component_extended_model"
    assert gate_b["baseline"] == "gate_a_core_model"
    assert gate_b["conditional_hard"] is True
    assert gate_b["applicable_if"] == (
        "false_under_contract_v1_unique_core_scope"
    )
    assert gate_b["current_contract_applicable"] is False
    assert gate_b["current_contract_component_extension_arm_allowed"] is False
    assert gate_b["activation_requires"] == (
        "new_versioned_contract_and_fresh_seed_ledger_frozen_before_any_component_extension_training"
    )
    assert gate_b["seeds_46_47_48_49_50_may_be_used_for_gate_b"] is False
    assert (
        gate_b[
            "future_gate_b_must_use_new_unseen_seeds_if_any_gate_a_result_has_been_observed"
        ]
        is True
    )
    assert fresh["gate_b_applicable_under_current_contract"] is False
    assert (
        fresh[
            "future_component_extension_requires_new_contract_new_ledger_and_new_unseen_seeds"
        ]
        is True
    )
    assert (
        fresh[
            "future_component_extension_may_not_reuse_46_47_48_49_50_after_any_gate_a_result_is_observed"
        ]
        is True
    )
    assert gate_b["template"] == "fresh5_nine_criterion_gate_template"
    assert gate_b["required_result_when_applicable"] == "9_of_9_GO"
    assert gate_b["failure_action"] == (
        "select_gate_a_core_model_and_delete_component_contribution_claims"
    )


def test_legacy_multiview_gate_cannot_replace_single_deployable_action(
    contract: dict,
) -> None:
    hard = contract["tiers"]["hard"]
    limitation = hard["legacy_gate_limit"]
    assert limitation["source_pooled_and_source_worst_may_select_different_thresholds"] is True
    assert limitation["nine_criterion_gate_is_necessary_but_not_sufficient"] is True
    assert limitation["legacy_views_cannot_satisfy_the_single_deployable_action_guard"] is True
    assert limitation["source_worst_uses_heldout_labels_to_maximize_worst_domain_pd"] is True
    assert limitation["source_worst_is_a_labelled_audit_oracle_not_a_deployable_selector"] is True

    guard = hard["single_deployable_action_guard"]
    assert guard["selector_view"] == (
        "registered_training_source_only_shared_lodo_bundle_selector"
    )
    assert guard["selector_must_be_fully_specified_and_hash_frozen_before_training"] is True
    assert "never_heldout_evaluation_labels" in guard["selector_inputs"]
    assert guard["action_unit"] == (
        "one_two_fold_checkpoint_bundle_plus_one_shared_score_transform_threshold_or_policy_per_arm_seed_budget"
    )
    assert guard["two_fold_checkpoint_bundle"] == {
        "heldout_NUDT-SIRST_member": "checkpoint_trained_only_on_IRSTD-1K",
        "heldout_IRSTD-1K_member": "checkpoint_trained_only_on_NUDT-SIRST",
    }
    assert guard["one_bundle_action_per_model_seed_and_budget"] is True
    assert guard["one_shared_threshold_or_policy_across_the_two_fold_members"] is True
    assert guard["bundle_dispatches_only_the_preregistered_fold_member_for_each_domain"] is True
    assert guard["same_non_checkpoint_action_fields_applied_unchanged_to_both_source_domains"] is True
    assert guard["domain_identity_or_heldout_labels_may_not_select_or_modify_action"] is True
    assert set(guard["action_includes"]) == {
        "two_fold_checkpoint_bundle",
        "score_transform",
        "threshold_or_policy",
        "reject_rule",
    }
    assert guard["all_deployable_guards_and_system_metrics_must_be_computed_from_this_action"] is True
    assert guard["each_domain_must_simultaneously_satisfy_pixel_and_component_fa_budget"] is True
    assert guard["domain_row_must_bind_bundle_action_sha256_and_dispatched_member_id"] is True
    assert guard["pixel_and_component_fa_must_come_from_the_same_prediction_mask_and_raw_count_row"] is True
    assert set(guard["required_raw_count_fields"]) == {
        "fp_pixels",
        "fp_components",
        "total_valid_pixels",
    }
    assert guard["exact_budget_checks"]["arithmetic"] == (
        "integer_cross_multiplication_before_display_rounding"
    )
    assert guard["exact_budget_checks"]["domain_reduction"] == (
        "all_source_domains_must_pass_both_checks"
    )
    assert guard["budget_names"] == ["strict", "medium", "loose"]
    assert set(guard["domain_names"]) == {"NUDT-SIRST", "IRSTD-1K"}
    assert guard["pooled_budget_feasibility_is_not_a_substitute"] is True


def _raw_count_row_passes_both_budgets(row: dict, budget: dict) -> bool:
    total = int(row["total_valid_pixels"])
    assert total > 0
    pixel_rate = Fraction(int(row["fp_pixels"]), total)
    component_rate = Fraction(int(row["fp_components"]) * 1_000_000, total)
    return pixel_rate <= Fraction(str(budget["pixel_fa_per_pixel_max"])) and (
        component_rate
        <= Fraction(str(budget["component_fa_per_million_pixels_max"]))
    )


def test_pooled_dual_budget_pass_cannot_hide_a_failing_domain(contract: dict) -> None:
    strict = next(
        row for row in contract["false_alarm_budgets"] if row["name"] == "strict"
    )
    rows = [
        {
            "bundle_action_sha256": "a" * 64,
            "dispatched_member_id": "heldout_NUDT-SIRST_member",
            "fp_pixels": 2,
            "fp_components": 2,
            "total_valid_pixels": 1_000_000,
        },
        {
            "bundle_action_sha256": "a" * 64,
            "dispatched_member_id": "heldout_IRSTD-1K_member",
            "fp_pixels": 7,
            "fp_components": 7,
            "total_valid_pixels": 9_000_000,
        },
    ]
    pooled = {
        key: sum(int(row[key]) for row in rows)
        for key in ("fp_pixels", "fp_components", "total_valid_pixels")
    }
    assert _raw_count_row_passes_both_budgets(pooled, strict) is True
    assert [_raw_count_row_passes_both_budgets(row, strict) for row in rows] == [
        False,
        True,
    ]
    assert all(row["bundle_action_sha256"] == "a" * 64 for row in rows)
    assert contract["tiers"]["hard"]["single_deployable_action_guard"][
        "pooled_budget_feasibility_is_not_a_substitute"
    ] is True


def test_single_action_adds_medium_and_loose_per_domain_noninferiority(
    contract: dict,
) -> None:
    section = contract["tiers"]["hard"]["deployable_action_per_domain_pd_guards"]
    guards = section["guards"]
    keyed = {(row["budget"], row["domain"]): row for row in guards}
    assert set(keyed) == {
        (budget, domain)
        for budget in ("strict", "medium", "loose")
        for domain in ("NUDT-SIRST", "IRSTD-1K")
    }
    assert section["aggregation"] == "paired_seed_mean_delta"
    assert section["every_guard_worst_seed_delta_min"] == pytest.approx(-0.005)
    assert all(row["pd_delta_min"] == pytest.approx(0.0) for row in guards)
    assert all(
        keyed[(budget, domain)]["status"] == "new_guard"
        for budget in ("medium", "loose")
        for domain in ("NUDT-SIRST", "IRSTD-1K")
    )


def test_hard_system_floors_prevent_reject_and_metric_collapse(contract: dict) -> None:
    hard = contract["tiers"]["hard"]
    finite = hard["finite_nonreject_tp_requirements"]
    assert finite["all_primary_metrics_and_all_active_thresholds_must_be_finite"] is True
    assert finite["fixed_global_threshold_mode_requires_finite_threshold_and_all_reject_sentinel_false"] is True
    assert finite["fixed_global_threshold_mode_coverage_required"] == pytest.approx(1.0)
    assert finite["adaptive_policy_mode_may_reject_individual_images"] is True
    assert finite["adaptive_policy_active_thresholds_must_be_finite"] is True
    assert finite["selected_bundle_action_must_not_be_reject_all"] is True
    assert finite["every_seed_domain_budget_cell_must_have_tp_objects_greater_than"] == 0
    assert finite["gt_denominator_must_include_rejected_images"] is True
    assert finite["reject_semantics"] == (
        "empty_prediction_with_zero_tp_zero_fp_pixels_zero_fp_components"
    )
    assert finite["rejects_may_not_be_removed_from_all_action_metrics"] is True

    risk = hard["risk_and_coverage_floor"]
    assert risk["required_for_each_budget"] is True
    assert risk["paper_display_name"] == "JointBSR-upper"
    assert risk["bound_metric_field"] == (
        "joint_budget_satisfaction_rate_per_image_suffix_max_including_rejects_as_no_detection"
    )
    assert risk["component_envelope"] == (
        "per_image_suffix_max_conservative_majorant_of_raw_component_risk"
    )
    assert risk["budget_boundary_operator"] == "less_than_or_equal"
    assert risk["denominator"] == "all_images_including_rejects_as_empty_predictions"
    assert risk["joint_budget_satisfaction_rate_suffix_max_min_over_source_domains"] == pytest.approx(0.90)
    assert risk["coverage_min_over_source_domains"] == pytest.approx(0.90)
    assert risk["seed_domain_budget_reduction"] == (
        "both_floors_must_pass_in_every_registered_seed_source_domain_budget_cell"
    )
    assert risk["averaging_across_a_failed_seed_domain_or_budget_cell_allowed"] is False
    assert risk["missing_rejected_or_non_finite_cell"] == "HOLD"
    assert risk["active_only_metrics_must_also_be_reported_but_cannot_replace_all_action_metrics"] is True

    noncollapse = hard["segmentation_noncollapse"]
    assert set(noncollapse["metrics"]) == {"mIoU", "nIoU"}
    assert noncollapse["canonical_raw_logit_threshold"] == pytest.approx(0.0)
    assert noncollapse["same_action_for_candidate_and_baseline"] is True
    assert noncollapse["source_macro_paired_mean_delta_min"] == pytest.approx(-0.005)
    assert noncollapse["each_source_domain_paired_mean_delta_min"] == pytest.approx(-0.01)
    assert noncollapse["rejected_images_are_empty_predictions_and_remain_in_denominators"] is True


def _evaluate_risk_coverage_cells(
    rows: dict,
    expected_cells: set,
    risk: dict,
) -> bool | str:
    if set(rows) != expected_cells:
        return "HOLD"

    joint_floor = float(
        risk[
            "joint_budget_satisfaction_rate_suffix_max_min_over_source_domains"
        ]
    )
    coverage_floor = float(risk["coverage_min_over_source_domains"])
    for cell in expected_cells:
        row = rows[cell]
        try:
            joint = float(row["joint_budget_satisfaction_rate"])
            coverage = float(row["coverage"])
        except (KeyError, TypeError, ValueError):
            return "HOLD"
        if not all(math.isfinite(value) for value in (joint, coverage)):
            return "HOLD"
        if joint < joint_floor or coverage < coverage_floor:
            return False
    return True


def test_risk_and_coverage_floors_are_hard_in_all_30_seed_domain_budget_cells(
    contract: dict,
) -> None:
    risk = contract["tiers"]["hard"]["risk_and_coverage_floor"]
    seeds = contract["fresh5_confirmation"]["formal_seed_ids"]
    domains = contract["tiers"]["hard"]["single_deployable_action_guard"]["domain_names"]
    budgets = [row["name"] for row in contract["false_alarm_budgets"]]
    expected_cells = {
        (seed, domain, budget)
        for seed in seeds
        for domain in domains
        for budget in budgets
    }
    assert len(expected_cells) == 5 * 2 * 3 == 30

    passing_rows = {
        cell: {
            "joint_budget_satisfaction_rate": 0.90,
            "coverage": 0.90,
        }
        for cell in expected_cells
    }
    assert _evaluate_risk_coverage_cells(
        passing_rows, expected_cells, risk
    ) is True

    first_cell = next(iter(expected_cells))
    one_failed = {cell: dict(row) for cell, row in passing_rows.items()}
    one_failed[first_cell]["coverage"] = 0.899
    assert _evaluate_risk_coverage_cells(
        one_failed, expected_cells, risk
    ) is False

    one_missing = dict(passing_rows)
    one_missing.pop(first_cell)
    assert _evaluate_risk_coverage_cells(
        one_missing, expected_cells, risk
    ) == "HOLD"

    one_non_finite = {cell: dict(row) for cell, row in passing_rows.items()}
    one_non_finite[first_cell]["joint_budget_satisfaction_rate"] = math.nan
    assert _evaluate_risk_coverage_cells(
        one_non_finite, expected_cells, risk
    ) == "HOLD"


def test_ci_uses_five_paired_seeds_without_pseudoreplication(contract: dict) -> None:
    stats = contract["statistics"]
    assert stats["repeat_unit"] == "seed"
    assert stats["num_independent_repeats"] == 5
    assert stats["paired_difference_direction"] == (
        "candidate_minus_baseline_within_same_seed"
    )
    assert stats["domains_folds_and_images_are_not_independent_repeats"] is True
    assert stats["fold_aggregation_occurs_within_seed_before_between_seed_inference"] is True
    assert stats["inference_scope"] == (
        "training_randomness_conditional_on_the_two_fixed_source_domains_and_frozen_splits"
    )
    assert stats["no_domain_image_component_or_threshold_pseudoreplication"] is True
    ci = stats["confidence_interval"]
    assert ci["method"] == "two_sided_paired_student_t_interval_for_mean_seed_delta"
    assert ci["confidence_level"] == pytest.approx(0.95)
    assert ci["degrees_of_freedom"] == 4
    assert ci["t_critical"] == pytest.approx(2.7764451051977987)
    assert ci["non_finite_or_missing_input"] == "fail_closed"


def test_publication_targets_are_not_hard_source_authorization_thresholds(
    contract: dict,
) -> None:
    tiers = contract["tiers"]
    publication = tiers["publication_target"]
    assert publication["normative_status"] == (
        "newly_preregistered_judgment_targets_not_historical_evidence"
    )
    assert publication["hard_source_gate_status"] == (
        "unchanged_not_a_hard_authorization_gate"
    )
    assert publication["required_for_aaai27_submission_performance_success"] is True
    assert "point_estimate_targets" not in tiers["hard"]

    point = publication["point_estimate_targets"]
    assert point["same_deployable_action_source_macro_pd_delta"] == {
        "strict": pytest.approx(0.02),
        "medium": pytest.approx(0.01),
        "loose": pytest.approx(0.01),
    }
    assert point["canonical_segmentation_source_macro_delta"] == {
        "mIoU": pytest.approx(0.01),
        "nIoU": pytest.approx(0.01),
    }
    assert point["end_to_end_latency_ratio_candidate_over_baseline_max"] == pytest.approx(1.25)
    assert set(
        publication[
            "primary_effect_metrics_requiring_ci_lower_strictly_greater_than_zero"
        ]
    ) == {
        "source_macro_pd_delta_strict",
        "source_macro_pd_delta_medium",
        "source_macro_pd_delta_loose",
        "source_macro_mIoU_delta",
        "source_macro_nIoU_delta",
    }
    assert publication["effect_seed_reduction"] == (
        "compute_two_domain_source_macro_within_each_seed_then_apply_the_registered_five_seed_mean_and_paired_t_interval"
    )
    assert publication["domain_macro_reduction"] == (
        "unweighted_arithmetic_mean_of_NUDT-SIRST_and_IRSTD-1K_within_seed"
    )
    assert publication["latency_gate_statistic"] == (
        "median_end_to_end_latency_ratio_candidate_over_baseline"
    )
    assert publication["latency_p95_role"] == (
        "required_reported_sensitivity_not_the_1.25_ratio_gate_statistic"
    )
    assert publication["latency_measurement"] == {
        "same_hardware_software_precision_input_size_batch_size_and_timing_harness": True,
        "includes_all_deployable_model_and_policy_compute": True,
        "reports_warmup_iterations_timed_iterations_median_and_p95": True,
        "excludes_offline_training_and_audit_only_label_oracle": True,
    }

    domain_effects = {
        seed: {
            "NUDT-SIRST": float(index),
            "IRSTD-1K": float(index + 2),
        }
        for index, seed in enumerate(
            contract["fresh5_confirmation"]["formal_seed_ids"]
        )
    }
    within_seed_source_macro = [
        (row["NUDT-SIRST"] + row["IRSTD-1K"]) / 2
        for row in domain_effects.values()
    ]
    assert within_seed_source_macro == pytest.approx([1, 2, 3, 4, 5])
    assert sum(within_seed_source_macro) / 5 == pytest.approx(3.0)

    latency_ratios = [1.0] * 19 + [2.0] * 2
    ordered_ratios = sorted(latency_ratios)
    median_ratio = ordered_ratios[len(ordered_ratios) // 2]
    p95_nearest_rank = ordered_ratios[
        math.ceil(0.95 * len(ordered_ratios)) - 1
    ]
    latency_gate_max = point[
        "end_to_end_latency_ratio_candidate_over_baseline_max"
    ]
    assert median_ratio <= latency_gate_max
    assert p95_nearest_rank > latency_gate_max
    assert publication["all_point_targets_and_all_primary_ci_targets_must_pass"] is True


def test_stretch_targets_are_stronger_but_optional(contract: dict) -> None:
    stretch = contract["tiers"]["stretch"]
    assert stretch["normative_status"] == (
        "newly_preregistered_optional_stretch_targets_not_historical_evidence"
    )
    assert stretch["required_for_aaai27_submission_performance_success"] is False
    assert stretch["same_deployable_action_worst_source_domain_pd_min"] == {
        "strict": pytest.approx(0.10),
        "medium": pytest.approx(0.25),
        "loose": pytest.approx(0.40),
    }
    assert stretch["primary_effect_ci_lower_must_reach_point_effect_target"] == {
        "source_macro_pd_delta_strict": pytest.approx(0.02),
        "source_macro_pd_delta_medium": pytest.approx(0.01),
        "source_macro_pd_delta_loose": pytest.approx(0.01),
        "source_macro_mIoU_delta": pytest.approx(0.01),
        "source_macro_nIoU_delta": pytest.approx(0.01),
    }
    assert stretch["end_to_end_latency_ratio_candidate_over_baseline_max"] == pytest.approx(1.10)
    assert stretch[
        "joint_budget_satisfaction_rate_suffix_max_min_over_source_domains"
    ] == pytest.approx(0.95)
    assert stretch["coverage_min_over_source_domains"] == pytest.approx(0.95)


def test_label_oracle_and_dense_grid_are_diagnostic_only(contract: dict) -> None:
    diagnostic = contract["diagnostic_only_evidence"]
    oracle = diagnostic["exact_per_domain_label_oracle"]
    assert oracle["allowed"] is True
    assert oracle["purpose"] == "frontier_and_raw_ranking_diagnostic_only"
    assert oracle["run_only_after_model_checkpoint_and_deployable_action_are_frozen"] is True
    assert oracle["may_select_model_checkpoint_seed_threshold_or_deployable_action"] is False
    assert oracle["may_count_toward_hard_or_publication_target"] is False
    assert oracle["must_be_reported_separately_from_deployable_action"] is True
    assert oracle[
        "required_as_frozen_post_training_falsification_if_raw_ordering_or_mechanism_claim_is_retained"
    ] is True
    assert oracle["falsification_direction_must_be_preregistered_before_training"] is True
    assert oracle["failure_action"] == (
        "mechanism_claim_fails_without_retuning_or_reselecting_the_model"
    )
    dense = diagnostic["dense_threshold_grid"]
    assert dense["allowed"] is True
    assert dense["may_count_toward_primary_decision"] is False


def test_outer_target_requires_one_complete_pre_access_sealed_batch(
    contract: dict,
) -> None:
    outer = contract["outer_target_sealed_batch"]
    assert outer["outer_domain"] == "NUAA-SIRST"
    assert outer["historical_official_test_label_exposure"] is True
    assert outer["paper_role"] == (
        "precommitted_no_feedback_development_domain_evaluation_not_untouched_blind_test"
    )
    assert outer["future_authorization_controls_only_additional_access_and_does_not_reset_history"] is True
    assert outer["access_before_separate_source_gate_authorization"] is False
    assert outer["source_gate_go_alone_authorizes_access"] is False
    assert outer["authorization_derivation"] == (
        "source_gate_go_and_complete_pre_access_batch_seal_and_independent_outer_access_authorization_artifact"
    )
    assert outer["independent_outer_access_authorization_path_and_sha256_required"] is True
    assert "complete_sealed_batch_manifest_and_sha256" in outer["before_first_access_must_freeze"]
    assert outer["all_NUAA_models_seeds_metrics_and_comparisons_in_one_complete_batch"] is True
    assert outer["post_freeze_batch_id_required"] is True
    assert outer["historical_access_does_not_start_or_satisfy_a_future_post_freeze_batch"] is True
    assert outer["batch_start_definition"] == (
        "first_additional_NUAA_read_or_execution_under_the_registered_post_freeze_batch_id_after_complete_batch_seal_and_independent_authorization"
    )
    assert outer["batch_may_start_without_complete_seal_and_independent_authorization"] is False
    assert outer["future_batch_semantics_do_not_reset_or_erase_historical_exposure"] is True
    assert outer["add_or_remove_tasks_after_batch_start_allowed"] is False
    assert outer["target_result_feedback_to_model_checkpoint_threshold_or_claim_selection_allowed"] is False
    assert outer["one_shot_outer_evaluation_and_three_domain_confirmation_must_be_distinguished"] is True
    assert outer["confirmation_results_do_not_participate_in_model_selection"] is True
    assert outer["all_preregistered_results_including_failures_must_be_reported"] is True


def test_fail_closed_list_covers_the_known_shortcuts(contract: dict) -> None:
    conditions = set(contract["fail_closed_conditions"])
    assert {
        "contract_or_preregistration_not_frozen_before_training",
        "fresh_seed_ledger_or_registered_code_sha256_not_frozen_before_formal_v3_training",
        "fresh_seed_local_scan_not_frozen_or_any_consumption_like_hit_unresolved",
        "fresh_seed_external_deleted_or_off_repository_author_attestation_missing_before_v3_training",
        "fresh5_seed_conflict_or_incomplete_seed_fold_matrix",
        "any_failed_hard_or_applicable_conditional_hard_requirement",
        "missing_infeasible_non_finite_or_reject_all_primary_action",
        "posthoc_seed_checkpoint_action_metric_or_threshold_selection",
        "component_claim_retained_after_gate_b_failure",
        "pooled_only_false_alarm_compliance_without_per_domain_dual_budget_compliance",
        "label_oracle_or_dense_grid_used_for_primary_selection",
        "riskcurve_calibration_or_system_metrics_used_to_compensate_gate_a_failure",
        "known_near_duplicate_or_sequence_scene_group_split_across_fit_and_evaluation",
        "heldout_source_labels_used_to_select_the_deployable_action",
        "retained_mechanism_claim_after_failed_frozen_oracle_or_ranking_falsification",
        "source_go_treated_as_outer_access_without_complete_seal_and_independent_authorization",
        "unsealed_or_premature_outer_target_access",
        "historical_hold_mutation_or_suppression_of_failed_results",
        "NUAA_described_as_untouched_despite_frozen_historical_exposure_registry",
    }.issubset(conditions)
