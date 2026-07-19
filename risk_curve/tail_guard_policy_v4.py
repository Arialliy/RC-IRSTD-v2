"""Fixed source-only Tail-Guard policy over canonical Count-all A anchors."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np

from .count_all_anchor import CountAllAnchor, derive_anchor_rates
from .representation import logit_threshold_grid_sha256, validate_logit_threshold_grid


TAIL_GUARD_POLICY_SCHEMA_VERSION = "rc-v4-tail-guard-policy-v1"
CANONICAL_GRID_SHA256 = (
    "1675cae81dace9ee92f1cf23fe5742f2d5ba0ce7ecea6e00c9f9648fa1825ced"
)
CANONICAL_TAIL_INDEX = 1750
CANONICAL_TAIL_LOGIT = float(np.float32(17.8231925964))
LOOSE_PIXEL_BUDGET = 1e-5
LOOSE_COMPONENT_BUDGET = 5.0
STRICT_PIXEL_BUDGET = 1e-6
STRICT_COMPONENT_BUDGET = 1.0
LOOSE_ZERO_TAIL_FACTOR = 0.35
LOOSE_NONZERO_TAIL_FACTOR = 1.0
STRICT_FACTOR = 3.0


@dataclass(frozen=True)
class TailGuardContract:
    grid_sha256: str = CANONICAL_GRID_SHA256
    tail_index: int = CANONICAL_TAIL_INDEX
    tail_logit: float = CANONICAL_TAIL_LOGIT


CANONICAL_TAIL_GUARD_CONTRACT = TailGuardContract()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _same_budget(observed: float, expected: float) -> bool:
    return math.isclose(float(observed), float(expected), rel_tol=1e-7, abs_tol=0.0)


def _validate_anchor_for_policy(
    anchor: CountAllAnchor,
    contract: TailGuardContract,
) -> np.ndarray:
    if not isinstance(anchor, CountAllAnchor):
        raise TypeError("anchor must be CountAllAnchor")
    if not _valid_sha256(anchor.archive_semantic_sha256):
        raise ValueError("Count-all anchor semantic hash is invalid")
    if anchor.adaptation_masks_read is not False:
        raise ValueError("Tail-Guard forbids adaptation masks")
    thresholds = validate_logit_threshold_grid(
        np.asarray(anchor.thresholds, dtype=np.float32)
    )
    semantic_hash = logit_threshold_grid_sha256(thresholds)
    if anchor.threshold_grid_sha256 != semantic_hash:
        raise ValueError("Tail-Guard anchor grid/hash mismatch")
    if semantic_hash != contract.grid_sha256:
        raise ValueError("Tail-Guard requires its fixed canonical grid SHA")
    if isinstance(contract.tail_index, bool) or contract.tail_index < 0:
        raise ValueError("Tail-Guard tail index is invalid")
    if contract.tail_index >= thresholds.size:
        raise ValueError("Tail-Guard tail index is outside the finite grid")
    if np.float32(thresholds[contract.tail_index]).tobytes() != np.float32(
        contract.tail_logit
    ).tobytes():
        raise ValueError("Tail-Guard canonical tail logit/index binding failed")

    pixel = np.asarray(anchor.pixel_counts)
    component_raw = np.asarray(anchor.component_counts_raw)
    component_upper = np.asarray(anchor.component_counts_upper)
    if pixel.dtype.kind not in "iu" or component_raw.dtype.kind not in "iu" or component_upper.dtype.kind not in "iu":
        raise ValueError("Tail-Guard requires integer A-window counts")
    if pixel.shape != thresholds.shape or component_raw.shape != thresholds.shape or component_upper.shape != thresholds.shape:
        raise ValueError("Tail-Guard A-count curves do not match the fixed grid")
    if np.any(pixel < 0) or np.any(component_raw < 0) or np.any(component_upper < 0):
        raise ValueError("Tail-Guard A counts must be non-negative")
    if np.any(component_raw > pixel):
        raise ValueError("Tail-Guard component counts exceed retained pixels")
    if np.any(pixel > int(anchor.total_pixels)):
        raise ValueError("Tail-Guard pixel counts exceed integer exposure")
    if np.any(np.diff(pixel.astype(np.int64)) > 0):
        raise ValueError("Tail-Guard pixel count curve is not monotone")
    expected_upper = np.maximum.accumulate(component_raw[::-1])[::-1]
    if not np.array_equal(component_upper, expected_upper):
        raise ValueError("Tail-Guard component envelope was tampered")
    return thresholds


def _registered_budget_position(pixel_budget: float, component_budget: float) -> int:
    if _same_budget(pixel_budget, LOOSE_PIXEL_BUDGET) and _same_budget(
        component_budget, LOOSE_COMPONENT_BUDGET
    ):
        return 0
    if _same_budget(pixel_budget, STRICT_PIXEL_BUDGET) and _same_budget(
        component_budget, STRICT_COMPONENT_BUDGET
    ):
        return 1
    raise ValueError("Tail-Guard only supports its two registered budget pairs")


def _selection_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def select_tail_guard_action(
    anchor: CountAllAnchor,
    *,
    pixel_budget: float,
    component_budget: float,
    contract: TailGuardContract = CANONICAL_TAIL_GUARD_CONTRACT,
) -> dict[str, Any]:
    """Select before any future-E counts are made available to the policy."""

    thresholds = _validate_anchor_for_policy(anchor, contract)
    budget_position = _registered_budget_position(pixel_budget, component_budget)
    pixel_rates, component_rates = derive_anchor_rates(anchor)
    tail_pixel_count = int(anchor.pixel_counts[contract.tail_index])
    tail_component_count = int(anchor.component_counts_upper[contract.tail_index])
    tail_is_zero = tail_pixel_count == 0 and tail_component_count == 0
    if budget_position == 0:
        factor = (
            LOOSE_ZERO_TAIL_FACTOR if tail_is_zero else LOOSE_NONZERO_TAIL_FACTOR
        )
        factor_reason = (
            "loose_tail_both_counts_zero"
            if tail_is_zero
            else "loose_tail_has_retained_count"
        )
    else:
        factor = STRICT_FACTOR
        factor_reason = "strict_fixed_factor"
    guarded_pixel = factor * pixel_rates
    guarded_component = factor * component_rates
    if np.any(np.diff(guarded_pixel) > 0.0) or np.any(
        np.diff(guarded_component) > 0.0
    ):
        raise ValueError("Tail-Guard scaled anchor risks are not monotone")
    feasible = np.flatnonzero(
        (guarded_pixel <= float(pixel_budget))
        & (guarded_component <= float(component_budget))
    )
    reject = feasible.size == 0
    index = None if reject else int(feasible[0])
    core = {
        "policy_schema_version": TAIL_GUARD_POLICY_SCHEMA_VERSION,
        "threshold_grid_sha256": contract.grid_sha256,
        "anchor_semantic_sha256": anchor.archive_semantic_sha256,
        "episode_index": int(anchor.row),
        "budget_position": budget_position,
        "pixel_budget": float(pixel_budget),
        "component_budget": float(component_budget),
        "tail_index": int(contract.tail_index),
        "tail_logit": float(contract.tail_logit),
        "tail_pixel_count": tail_pixel_count,
        "tail_component_upper_count": tail_component_count,
        "tail_both_counts_zero": tail_is_zero,
        "factor": float(factor),
        "factor_reason": factor_reason,
        "threshold_index": index,
        "selected_logit_threshold": "+inf" if reject else float(thresholds[index]),
        "reject": bool(reject),
        "external_reject_action": bool(reject),
        "adaptation_total_pixels": int(anchor.total_pixels),
        "adaptation_masks_read": False,
        "future_e_counts_used_for_selection": False,
        "selection_rule": (
            "earliest_jointly_feasible_factor_scaled_count_all_A_anchor"
        ),
    }
    if index is not None:
        core.update(
            {
                "adaptation_pixel_count_at_action": int(anchor.pixel_counts[index]),
                "adaptation_component_raw_count_at_action": int(
                    anchor.component_counts_raw[index]
                ),
                "adaptation_component_upper_count_at_action": int(
                    anchor.component_counts_upper[index]
                ),
                "guarded_pixel_risk_at_action": float(guarded_pixel[index]),
                "guarded_component_risk_at_action": float(guarded_component[index]),
            }
        )
    core["selection_sha256"] = _selection_sha256(core)
    return core


def policy_contract_record(
    contract: TailGuardContract = CANONICAL_TAIL_GUARD_CONTRACT,
) -> dict[str, Any]:
    return {
        "schema_version": TAIL_GUARD_POLICY_SCHEMA_VERSION,
        "threshold_grid_sha256": contract.grid_sha256,
        "tail_index": int(contract.tail_index),
        "tail_logit": float(contract.tail_logit),
        "loose_budget": [LOOSE_PIXEL_BUDGET, LOOSE_COMPONENT_BUDGET],
        "loose_zero_tail_factor": LOOSE_ZERO_TAIL_FACTOR,
        "loose_nonzero_tail_factor": LOOSE_NONZERO_TAIL_FACTOR,
        "strict_budget": [STRICT_PIXEL_BUDGET, STRICT_COMPONENT_BUDGET],
        "strict_factor": STRICT_FACTOR,
        "factor_applies_to": "integer_A_count_derived_pixel_and_component_upper_rates",
        "adaptation_masks_read": False,
        "future_e_counts_used_for_selection": False,
        # These constants were frozen only after development diagnostics on the
        # current two source pseudo-target folds.  Keeping that fact in the
        # machine-readable contract prevents this candidate from being
        # misrepresented as an independently pre-registered Gate-C result.
        "post_hoc_source_pseudo_target_development_selection": True,
        "confirmatory_gate_c_eligible": False,
    }


__all__ = [
    "CANONICAL_GRID_SHA256",
    "CANONICAL_TAIL_GUARD_CONTRACT",
    "CANONICAL_TAIL_INDEX",
    "CANONICAL_TAIL_LOGIT",
    "LOOSE_COMPONENT_BUDGET",
    "LOOSE_NONZERO_TAIL_FACTOR",
    "LOOSE_PIXEL_BUDGET",
    "LOOSE_ZERO_TAIL_FACTOR",
    "STRICT_COMPONENT_BUDGET",
    "STRICT_FACTOR",
    "STRICT_PIXEL_BUDGET",
    "TAIL_GUARD_POLICY_SCHEMA_VERSION",
    "TailGuardContract",
    "policy_contract_record",
    "select_tail_guard_action",
]
