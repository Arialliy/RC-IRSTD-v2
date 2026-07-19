"""Evaluate the fixed Tail-Guard policy with selection-before-E auditing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .count_all_anchor import validate_count_all_anchor_archive
from .curve_dataset import load_curve_archive
from .evaluate_gate_c_baselines_v4 import (
    _adaptation_ids,
    _aggregate_actions,
    _episode_ids,
    _sha256_file,
    _validate_count_archive,
    _validation_action,
    _write_json_atomic,
)
from .representation import LOGIT_GRID_SCHEMA_VERSION, LOGIT_REPRESENTATION
from .tail_guard_policy_v4 import (
    CANONICAL_TAIL_GUARD_CONTRACT,
    LOOSE_COMPONENT_BUDGET,
    LOOSE_PIXEL_BUDGET,
    STRICT_COMPONENT_BUDGET,
    STRICT_PIXEL_BUDGET,
    TailGuardContract,
    policy_contract_record,
    select_tail_guard_action,
)


TAIL_GUARD_EVALUATION_SCHEMA_VERSION = "rc-v4-tail-guard-evaluation-v1"
_SELECTION_ARCHIVE_FIELDS = (
    "statistics",
    "thresholds",
    "representation",
    "threshold_grid_sha256",
    "adaptation_predicted_pixel_counts",
    "adaptation_predicted_component_counts_raw",
    "adaptation_predicted_component_counts_upper",
    "adaptation_total_pixels",
    "count_all_adaptation_schema_version",
    "provenance_json",
)


def _scalar_text(value: Any, field: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{field} must be scalar")
    return str(array.item())


def _provenance(archive: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if "provenance_json" not in archive:
        raise ValueError("Tail-Guard archive is missing provenance_json")
    try:
        payload = json.loads(_scalar_text(archive["provenance_json"], "provenance_json"))
    except json.JSONDecodeError as error:
        raise ValueError("Tail-Guard provenance_json is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError("Tail-Guard provenance_json must decode to an object")
    return payload


def _load_selection_archive(path: Path) -> dict[str, np.ndarray]:
    """Read only label-free A fields; future-E arrays remain unopened."""

    if not path.is_file():
        raise FileNotFoundError(f"Tail-Guard episode archive does not exist: {path}")
    with np.load(path, allow_pickle=False) as source:
        missing = [field for field in _SELECTION_ARCHIVE_FIELDS if field not in source]
        if missing:
            raise ValueError(
                "Tail-Guard selection archive is missing: " + ", ".join(missing)
            )
        return {field: source[field] for field in _SELECTION_ARCHIVE_FIELDS}


def evaluate_tail_guard(
    *,
    episode_file: str | Path,
    output: str | Path,
    contract: TailGuardContract = CANONICAL_TAIL_GUARD_CONTRACT,
) -> Path:
    """Freeze all A-only actions, then and only then open future-E evidence."""

    episode_path = Path(episode_file).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    episode_archive_sha256 = _sha256_file(episode_path)
    selection_archive = _load_selection_archive(episode_path)
    provenance = _provenance(selection_archive)
    anchors = validate_count_all_anchor_archive(
        selection_archive, expected_grid_sha256=contract.grid_sha256
    )
    budgets = (
        (LOOSE_PIXEL_BUDGET, LOOSE_COMPONENT_BUDGET),
        (STRICT_PIXEL_BUDGET, STRICT_COMPONENT_BUDGET),
    )

    # Critical ordering invariant: policy functions receive immutable A-count
    # anchors only.  Future-E sufficient counts are not validated/read until
    # every episode action for every registered budget has been frozen.
    frozen_selections: list[list[dict[str, Any]]] = []
    for pixel_budget, component_budget in budgets:
        frozen_selections.append(
            [
                select_tail_guard_action(
                    anchors.episode(row),
                    pixel_budget=pixel_budget,
                    component_budget=component_budget,
                    contract=contract,
                )
                for row in range(anchors.num_episodes)
            ]
        )

    if _sha256_file(episode_path) != episode_archive_sha256:
        raise ValueError("Tail-Guard episode archive changed during A-only selection")
    # Post-selection audit only: this is the first full-archive load and the
    # first access to future-E sufficient counts in this evaluator.
    archive = load_curve_archive(episode_path)
    audit_anchors = validate_count_all_anchor_archive(
        archive, expected_grid_sha256=contract.grid_sha256
    )
    if audit_anchors.semantic_sha256 != anchors.semantic_sha256:
        raise ValueError("Tail-Guard A anchor changed before future-E audit")
    future_e = _validate_count_archive(archive, split="validation")
    if int(future_e["thresholds"].size) != int(anchors.thresholds.size):
        raise ValueError("Tail-Guard A anchor and E audit grid sizes differ")

    per_episode: list[dict[str, Any]] = [
        {
            "episode_index": row,
            "pseudo_target": str(future_e["pseudo_targets"][row]),
            "adaptation_ids": _adaptation_ids(archive, row),
            "evaluation_ids": _episode_ids(archive, row),
            "actions": [],
        }
        for row in range(anchors.num_episodes)
    ]
    budget_results: list[dict[str, Any]] = []
    action_codes: list[list[int]] = [
        [] for _ in range(anchors.num_episodes)
    ]
    for budget_position, ((pixel_budget, component_budget), selections) in enumerate(
        zip(budgets, frozen_selections)
    ):
        actions: list[dict[str, Any]] = []
        for row, selection in enumerate(selections):
            action = _validation_action(
                future_e,
                row=row,
                selection=selection,
                pixel_budget=pixel_budget,
                component_budget=component_budget,
            )
            action.update(
                {
                    "selection_rule": selection["selection_rule"],
                    "selection_sha256": selection["selection_sha256"],
                    "factor": selection["factor"],
                    "factor_reason": selection["factor_reason"],
                    "tail_both_counts_zero": selection["tail_both_counts_zero"],
                    "future_e_counts_used_for_selection": False,
                }
            )
            actions.append(action)
            action_codes[row].append(
                int(anchors.thresholds.size)
                if bool(selection["reject"])
                else int(selection["threshold_index"])
            )
            per_episode[row]["actions"].append(
                {
                    "budget_position": budget_position,
                    "pixel_budget": pixel_budget,
                    "component_budget": component_budget,
                    "methods": {"tail_guard": action},
                    "selection": selection,
                }
            )
        aggregate = _aggregate_actions(actions)
        factor_values, factor_counts = np.unique(
            np.asarray([item["factor"] for item in selections], dtype=np.float64),
            return_counts=True,
        )
        aggregate.update(
            {
                "monotonic_violation_rate": 0.0,
                "factor_histogram": {
                    str(float(value)): int(count)
                    for value, count in zip(factor_values.tolist(), factor_counts.tolist())
                },
            }
        )
        budget_results.append(
            {
                "budget_position": budget_position,
                "pixel_budget": pixel_budget,
                "component_budget": component_budget,
                "methods": {"tail_guard": aggregate},
            }
        )

    monotonic_violations = sum(
        int(any(right < left for left, right in zip(codes, codes[1:])))
        for codes in action_codes
    )
    monotonic_violation_rate = monotonic_violations / float(anchors.num_episodes)
    if monotonic_violations:
        raise ValueError(
            "Tail-Guard selected a less conservative action for a stricter budget"
        )

    payload = {
        "schema_version": TAIL_GUARD_EVALUATION_SCHEMA_VERSION,
        "method_name": "tail_guard",
        "role": "post_hoc_source_pseudo_target_development_candidate",
        "protocol": "label_free_A_selection_then_future_E_audit",
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_size": int(anchors.thresholds.size),
        "threshold_grid_sha256": anchors.threshold_grid_sha256,
        "threshold_grid_manifest_sha256": _scalar_text(
            archive["threshold_grid_manifest_sha256"],
            "threshold_grid_manifest_sha256",
        ),
        "feature_schema_sha256": _scalar_text(
            archive["feature_schema_sha256"], "feature_schema_sha256"
        ),
        "threshold_grid_detector_protocol": _scalar_text(
            archive["threshold_grid_detector_protocol"],
            "threshold_grid_detector_protocol",
        ),
        "threshold_grid_detector_checkpoint_sha256s": [
            str(item)
            for item in np.asarray(
                archive["threshold_grid_detector_checkpoint_sha256s"]
            ).tolist()
        ],
        "threshold_grid_outer_detector_checkpoint_sha256": _scalar_text(
            archive["threshold_grid_outer_detector_checkpoint_sha256"],
            "threshold_grid_outer_detector_checkpoint_sha256",
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": [
            str(item)
            for item in np.asarray(
                archive["threshold_grid_episode_detector_checkpoint_sha256s"]
            ).tolist()
        ],
        "episode_archive": str(episode_path),
        "episode_archive_sha256": episode_archive_sha256,
        "anchor_semantic_sha256": anchors.semantic_sha256,
        "num_episodes": anchors.num_episodes,
        "pseudo_targets": sorted(
            set(str(item) for item in future_e["pseudo_targets"].tolist())
        ),
        "excluded_outer_target": provenance.get("threshold_grid_outer_target_key"),
        "outer_target_labels_used": False,
        "labels_used_for_action_selection": False,
        "source_pseudo_target_labels_used_for_post_selection_evaluation": True,
        "selection_finalized_before_future_e_audit": True,
        "adaptation_masks_read": False,
        "future_e_counts_used_for_selection": False,
        "post_hoc_source_pseudo_target_development_selection": True,
        "confirmatory_gate_c_eligible": False,
        "policy_contract": policy_contract_record(contract),
        "monotonic_violation_rates": {
            "tail_guard": monotonic_violation_rate
        },
        "monotonic_violation_counts": {
            "tail_guard": monotonic_violations
        },
        "budgets": budget_results,
        "per_episode": per_episode,
        "status": "DEVELOPMENT_COMPLETE_NOT_CONFIRMATORY",
    }
    _write_json_atomic(output_path, payload)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-file", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = evaluate_tail_guard(
        episode_file=args.episode_file,
        output=args.output,
    )
    print(json.dumps({"output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "TAIL_GUARD_EVALUATION_SCHEMA_VERSION",
    "build_parser",
    "evaluate_tail_guard",
]
