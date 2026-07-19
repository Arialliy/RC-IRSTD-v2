"""Exact shared-threshold source operating points in the FP32 logit domain.

This evaluator is the formal, source-only counterpart of the diagnostic
single-domain raw-logit Oracle.  It preserves one numeric threshold across all
held-out source domains, aggregates raw counts before computing pooled risk,
and enumerates every state that can possibly satisfy the frozen loose pixel
budget.  Lower states are omitted only after an integer-count proof that they
are infeasible for the pooled curve and for every per-domain diagnostic.

The hot path uses a sparse incremental 8-connected union-find per image.  With
``min_component_area == 1`` and overlap matching, this is exactly equivalent
to repeatedly thresholding the full image and calling ``match_components``;
selected formal points are independently rechecked by the rescue runner.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

import numpy as np

from .component_matching import connected_components
from .raw_logit_oracle import RawLogitSample
from .raw_logit_rescue_diagnostics import select_dense_operating_points


SOURCE_RAW_LOGIT_SCHEMA = "rc-irstd-aaai27-source-raw-logit-exact-states-v1"
DEFAULT_LOOSE_PIXEL_BUDGET = 1.0e-5

_COUNT_FIELDS = (
    "tp_objects",
    "gt_objects",
    "fp_components",
    "fp_pixels",
    "total_pixels",
)


def _validated_samples_by_domain(
    samples_by_domain: Mapping[str, Sequence[RawLogitSample]],
) -> dict[str, tuple[RawLogitSample, ...]]:
    if not isinstance(samples_by_domain, Mapping) or len(samples_by_domain) < 2:
        raise ValueError("at least two named source domains are required")
    output: dict[str, tuple[RawLogitSample, ...]] = {}
    for raw_name in sorted(samples_by_domain):
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("domain names must be non-empty strings")
        raw_samples = samples_by_domain[raw_name]
        if not isinstance(raw_samples, Sequence) or isinstance(
            raw_samples, (str, bytes)
        ) or not raw_samples:
            raise ValueError(f"domain {raw_name!r} has no samples")
        seen: set[str] = set()
        samples: list[RawLogitSample] = []
        for index, sample in enumerate(raw_samples):
            if not isinstance(sample, RawLogitSample):
                raise TypeError(
                    f"domain {raw_name!r} sample {index} is not RawLogitSample"
                )
            if not isinstance(sample.image_id, str) or not sample.image_id:
                raise ValueError(f"domain {raw_name!r} sample {index} has no image_id")
            if sample.image_id in seen:
                raise ValueError(
                    f"domain {raw_name!r} contains duplicate image_id {sample.image_id!r}"
                )
            seen.add(sample.image_id)
            logits = np.asarray(sample.logits)
            probability = np.asarray(sample.probability)
            mask = np.asarray(sample.mask)
            if logits.dtype != np.float32 or probability.dtype != np.float32:
                raise ValueError("formal raw logits and probabilities must be float32")
            if logits.ndim != 2 or logits.size == 0:
                raise ValueError("formal raw logits must be non-empty 2-D arrays")
            if probability.shape != logits.shape or mask.shape != logits.shape:
                raise ValueError("raw-logit sample arrays have inconsistent shapes")
            if not np.isfinite(logits).all() or not np.isfinite(probability).all():
                raise ValueError("raw-logit sample contains non-finite scores")
            if np.any((probability < 0.0) | (probability > 1.0)):
                raise ValueError("raw-logit sample probability lies outside [0, 1]")
            if mask.dtype != np.bool_:
                if not np.issubdtype(mask.dtype, np.number) or not np.isfinite(mask).all():
                    raise ValueError("raw-logit sample mask must be finite and binary")
                if not np.isin(np.unique(mask), (0, 1)).all():
                    raise ValueError("raw-logit sample mask must be binary")
            samples.append(
                RawLogitSample(
                    image_id=sample.image_id,
                    logits=np.ascontiguousarray(logits),
                    probability=np.ascontiguousarray(probability),
                    mask=np.ascontiguousarray(mask.astype(bool, copy=False)),
                )
            )
        output[raw_name] = tuple(samples)
    return output


def _positive_fraction(value: Any, *, name: str) -> Fraction:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    try:
        number = float(value)
        fraction = Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(number) or fraction <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return fraction


def _top_background_cutoff(
    samples: Sequence[RawLogitSample],
    *,
    pixel_budget: Fraction,
) -> tuple[float | None, dict[str, Any]]:
    """Find the first infeasible background tie without concatenating all pixels."""

    total_pixels = int(sum(sample.mask.size for sample in samples))
    background_pixels = int(sum(np.count_nonzero(~sample.mask) for sample in samples))
    maximum_feasible = (
        pixel_budget.numerator * total_pixels // pixel_budget.denominator
    )
    if background_pixels <= maximum_feasible:
        return None, {
            "cutoff_logit_float32": None,
            "maximum_feasible_fp_pixels": int(maximum_feasible),
            "background_pixels": background_pixels,
            "reason": "all_background_pixels_fit_pixel_budget",
        }
    required_rank = int(maximum_feasible + 1)
    local_tails: list[np.ndarray] = []
    for sample in samples:
        background = sample.logits[~sample.mask].reshape(-1)
        if background.size == 0:
            continue
        count = min(required_rank, int(background.size))
        local_tails.append(
            np.partition(background, int(background.size) - count)[-count:]
        )
    candidates = np.concatenate(local_tails)
    cutoff = np.partition(candidates, int(candidates.size) - required_rank)[
        -required_rank
    ]
    cutoff32 = float(np.float32(cutoff))
    strictly_above = int(
        sum(np.count_nonzero(sample.logits[~sample.mask] > cutoff) for sample in samples)
    )
    at_or_above = int(
        sum(np.count_nonzero(sample.logits[~sample.mask] >= cutoff) for sample in samples)
    )
    if strictly_above > maximum_feasible or at_or_above <= maximum_feasible:
        raise AssertionError("background order-statistic cutoff proof failed")
    return cutoff32, {
        "cutoff_logit_float32": cutoff32,
        "maximum_feasible_fp_pixels": int(maximum_feasible),
        "background_pixels_strictly_above_cutoff": strictly_above,
        "background_pixels_at_or_above_cutoff": at_or_above,
        "background_pixels": background_pixels,
        "reason": "threshold_at_or_below_cutoff_is_pixel_budget_infeasible",
    }


def _maximum_cardinality_overlap(adjacency: Mapping[int, set[int]]) -> int:
    ground_truth_to_prediction: dict[int, int] = {}

    def augment(prediction: int, seen: set[int]) -> bool:
        for ground_truth in sorted(adjacency[prediction]):
            if ground_truth in seen:
                continue
            seen.add(ground_truth)
            previous = ground_truth_to_prediction.get(ground_truth)
            if previous is None or augment(previous, seen):
                ground_truth_to_prediction[ground_truth] = prediction
                return True
        return False

    for prediction in sorted(adjacency):
        augment(prediction, set())
    return len(ground_truth_to_prediction)


def _sample_sparse_local_curve(
    sample: RawLogitSample,
    *,
    cutoff: float | None,
    connectivity: int,
) -> tuple[dict[str, int], list[tuple[float, dict[str, int]]], int]:
    """Build the exact local staircase with sparse incremental union-find."""

    gt_labels, num_gt = connected_components(
        sample.mask, connectivity=connectivity, min_component_area=1
    )
    flat_logits = sample.logits.reshape(-1)
    if cutoff is None:
        retained = np.arange(flat_logits.size, dtype=np.int64)
    else:
        retained = np.flatnonzero(flat_logits > np.float32(cutoff))
    retained_values = flat_logits[retained]
    if retained.size:
        order = np.argsort(retained_values, kind="stable")[::-1]
        retained = retained[order]
        retained_values = retained_values[order]

    height, width = sample.logits.shape
    parent = np.full(flat_logits.size, -1, dtype=np.int32)
    area = np.zeros(flat_logits.size, dtype=np.int32)
    overlaps: dict[int, set[int]] = {}
    roots: set[int] = set()
    flat_gt = gt_labels.reshape(-1)
    flat_mask = sample.mask.reshape(-1)
    fp_pixels = 0

    def find(value: int) -> int:
        current = value
        while int(parent[current]) != current:
            parent[current] = parent[int(parent[current])]
            current = int(parent[current])
        return current

    def union(first: int, second: int) -> int:
        root_a = find(first)
        root_b = find(second)
        if root_a == root_b:
            return root_a
        if int(area[root_a]) < int(area[root_b]) or (
            int(area[root_a]) == int(area[root_b]) and root_a > root_b
        ):
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        area[root_a] += area[root_b]
        overlaps[root_a].update(overlaps.pop(root_b))
        roots.remove(root_b)
        return root_a

    events: list[tuple[float, dict[str, int]]] = []
    cursor = 0
    while cursor < retained.size:
        threshold = retained_values[cursor]
        stop = cursor + 1
        while stop < retained.size and retained_values[stop] == threshold:
            stop += 1

        activated = retained[cursor:stop]
        for raw_position in activated:
            position = int(raw_position)
            parent[position] = position
            area[position] = 1
            gt_label = int(flat_gt[position])
            overlaps[position] = {gt_label} if gt_label else set()
            roots.add(position)
            fp_pixels += int(not bool(flat_mask[position]))

        for raw_position in activated:
            position = int(raw_position)
            row, column = divmod(position, width)
            for delta_row in (-1, 0, 1):
                neighbor_row = row + delta_row
                if neighbor_row < 0 or neighbor_row >= height:
                    continue
                for delta_column in (-1, 0, 1):
                    if delta_row == 0 and delta_column == 0:
                        continue
                    neighbor_column = column + delta_column
                    if neighbor_column < 0 or neighbor_column >= width:
                        continue
                    neighbor = neighbor_row * width + neighbor_column
                    if parent[neighbor] >= 0:
                        union(position, neighbor)

        adjacency = {root: overlaps[root] for root in roots}
        num_tp = _maximum_cardinality_overlap(adjacency)
        events.append(
            (
                float(np.float32(threshold)),
                {
                    "tp_objects": int(num_tp),
                    "gt_objects": int(num_gt),
                    "fp_components": int(len(roots) - num_tp),
                    "fp_pixels": int(fp_pixels),
                    "total_pixels": int(sample.mask.size),
                },
            )
        )
        cursor = stop

    empty = {
        "tp_objects": 0,
        "gt_objects": int(num_gt),
        "fp_components": 0,
        "fp_pixels": 0,
        "total_pixels": int(sample.mask.size),
    }
    return empty, events, int(retained.size)


def _metrics(row: Mapping[str, int]) -> dict[str, int | float]:
    counts = {field: int(row[field]) for field in _COUNT_FIELDS}
    return {
        **counts,
        "pd": float(counts["tp_objects"] / counts["gt_objects"])
        if counts["gt_objects"]
        else 0.0,
        "fa_pixel": float(counts["fp_pixels"] / counts["total_pixels"]),
        "fa_component_mp": float(
            counts["fp_components"] / (counts["total_pixels"] / 1_000_000.0)
        ),
    }


def _aggregate(per_domain: Mapping[str, Mapping[str, int]]) -> dict[str, int | float]:
    return _metrics(
        {
            field: sum(int(row[field]) for row in per_domain.values())
            for field in _COUNT_FIELDS
        }
    )


def enumerate_exact_shared_states(
    samples_by_domain: Mapping[str, Sequence[RawLogitSample]],
    *,
    loose_pixel_budget: float = DEFAULT_LOOSE_PIXEL_BUDGET,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
) -> dict[str, Any]:
    """Enumerate every potentially feasible shared FP32 raw-logit state."""

    if matching_rule != "overlap":
        raise ValueError("rescue v1 is frozen to overlap matching")
    if not math.isfinite(float(centroid_distance)) or float(centroid_distance) != 3.0:
        raise ValueError("rescue v1 is frozen to centroid_distance=3.0")
    if connectivity not in {2, 8}:
        raise ValueError("rescue v1 is frozen to 8-neighbor connectivity (2/8)")
    if min_component_area != 1:
        raise ValueError("lossless rescue v1 pruning requires min_component_area=1")
    domains = _validated_samples_by_domain(samples_by_domain)
    loose = _positive_fraction(loose_pixel_budget, name="loose_pixel_budget")

    pooled_samples = tuple(
        sample for name in sorted(domains) for sample in domains[name]
    )
    pooled_cutoff, pooled_proof = _top_background_cutoff(
        pooled_samples, pixel_budget=loose
    )
    domain_proofs: dict[str, Any] = {}
    finite_cutoffs: list[float] = []
    if pooled_cutoff is not None:
        finite_cutoffs.append(pooled_cutoff)
    for name, samples in domains.items():
        cutoff, proof = _top_background_cutoff(samples, pixel_budget=loose)
        domain_proofs[name] = proof
        if cutoff is not None:
            finite_cutoffs.append(cutoff)
    enumeration_cutoff = min(finite_cutoffs) if len(finite_cutoffs) == len(domains) + 1 else None

    global_events: dict[float, list[tuple[str, int, dict[str, int]]]] = defaultdict(list)
    empty_rows: dict[tuple[str, int], dict[str, int]] = {}
    retained_pixels = 0
    for domain, samples in domains.items():
        for sample_index, sample in enumerate(samples):
            empty, local_events, local_retained = _sample_sparse_local_curve(
                sample,
                cutoff=enumeration_cutoff,
                connectivity=connectivity,
            )
            empty_rows[(domain, sample_index)] = empty
            retained_pixels += local_retained
            for threshold, row in local_events:
                global_events[threshold].append((domain, sample_index, row))

    cached = {key: dict(row) for key, row in empty_rows.items()}
    totals: dict[str, dict[str, int]] = {}
    for domain, samples in domains.items():
        totals[domain] = {
            field: sum(empty_rows[(domain, index)][field] for index in range(len(samples)))
            for field in _COUNT_FIELDS
        }

    states: list[dict[str, Any]] = []

    def append_state(threshold: float | None, *, reject: bool) -> None:
        per_domain = {name: _metrics(totals[name]) for name in sorted(totals)}
        states.append(
            {
                "threshold_domain": "raw_logit",
                "threshold_logit_float32": threshold,
                "threshold_state_rank": len(states),
                "all_reject_sentinel": reject,
                "pooled": _aggregate(totals),
                "per_domain": per_domain,
            }
        )

    append_state(None, reject=True)
    for threshold in sorted(global_events, reverse=True):
        for domain, sample_index, row in global_events[threshold]:
            key = (domain, sample_index)
            previous = cached[key]
            cached[key] = row
            for field in ("tp_objects", "fp_components", "fp_pixels"):
                totals[domain][field] += int(row[field]) - int(previous[field])
        append_state(float(np.float32(threshold)), reject=False)

    num_states = len(states)
    for state in states:
        state["num_distinct_logit_states"] = num_states

    return {
        "schema_version": SOURCE_RAW_LOGIT_SCHEMA,
        "artifact_type": "source-only-exact-shared-fp32-raw-logit-states",
        "formal_protocol_eligible": True,
        "diagnostic_only": False,
        "source_only": True,
        "threshold_domain": "raw_logit",
        "prediction_rule": "float32 raw_logit >= shared threshold_logit",
        "shared_threshold_across_domains": True,
        "exact_state_enumeration": True,
        "matching_protocol": {
            "matching_rule": matching_rule,
            "centroid_distance": float(centroid_distance),
            "connectivity": int(connectivity),
            "min_component_area": int(min_component_area),
        },
        "domain_names": list(sorted(domains)),
        "states": states,
        "search": {
            "num_prediction_states_evaluated": num_states,
            "num_finite_threshold_states_evaluated": len(global_events),
            "num_retained_tail_pixels": retained_pixels,
            "all_reject_sentinel_evaluated": True,
            "lossless_pruning": {
                "loose_pixel_budget": float(loose),
                "enumeration_cutoff_logit_float32": enumeration_cutoff,
                "pooled": pooled_proof,
                "per_domain": domain_proofs,
                "proof": (
                    "every omitted threshold is at or below every pooled/domain "
                    "first-infeasible background order statistic"
                ),
            },
        },
    }


def _states(enumeration: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    if not isinstance(enumeration, Mapping):
        raise TypeError("exact enumeration must be a mapping")
    states = enumeration.get("states")
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)) or not states:
        raise ValueError("exact enumeration has no states")
    if enumeration.get("exact_state_enumeration") is not True:
        raise ValueError("formal source selection requires exact state enumeration")
    return states


def select_exact_shared_source_operating_points(
    enumeration: Mapping[str, Any],
    *,
    pixel_budget: float | None = None,
    component_budget: float | None = None,
    budgets: Mapping[str, Any] | Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Select exact pooled/worst points with the original shared-threshold rules."""

    if budgets is not None:
        if pixel_budget is not None or component_budget is not None:
            raise ValueError("pass either budgets or one pixel/component pair")
        return select_dense_operating_points(_states(enumeration), budgets)
    if pixel_budget is None or component_budget is None:
        raise ValueError("pixel_budget and component_budget are required")
    selected = select_dense_operating_points(
        _states(enumeration),
        [("requested", pixel_budget, component_budget)],
    )["requested"]
    original_states = _states(enumeration)
    for mode in ("source_pooled", "source_worst"):
        point = selected[mode]
        if point.get("found") is not True:
            continue
        state_index = int(point["state_index"])
        state = original_states[state_index]
        state_rank = int(state["threshold_state_rank"])
        num_states = int(state["num_distinct_logit_states"])
        point["threshold_state_rank"] = state_rank
        point["num_distinct_logit_states"] = num_states
        point["operating_point"]["threshold_state_rank"] = state_rank
        point["operating_point"]["num_distinct_logit_states"] = num_states
        for row in point["source_rows"].values():
            row["threshold_state_rank"] = state_rank
            row["num_distinct_logit_states"] = num_states
    return {
        "schema_version": "rc-irstd-aaai27-source-raw-logit-operating-point-v1",
        "threshold_domain": "raw_logit",
        "exact_state_enumeration_is_primary": True,
        "pixel_budget": float(pixel_budget),
        "component_budget": float(component_budget),
        "source_pooled": selected["source_pooled"],
        "source_worst": selected["source_worst"],
    }


def evaluate_domains_at_threshold(
    samples_by_domain: Mapping[str, Sequence[RawLogitSample]],
    threshold_logit_float32: float | None,
    *,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
) -> dict[str, Any]:
    """Legacy full-image evaluation used to verify selected exact states."""

    from .component_matching import match_components

    domains = _validated_samples_by_domain(samples_by_domain)
    rows: dict[str, Any] = {}
    for domain, samples in domains.items():
        totals = {field: 0 for field in _COUNT_FIELDS}
        for sample in samples:
            prediction = (
                np.zeros(sample.mask.shape, dtype=bool)
                if threshold_logit_float32 is None
                else sample.logits >= np.float32(threshold_logit_float32)
            )
            result = match_components(
                prediction,
                sample.mask,
                rule=matching_rule,
                centroid_distance=centroid_distance,
                connectivity=connectivity,
                min_component_area=min_component_area,
            )
            totals["tp_objects"] += int(result.num_tp_objects)
            totals["gt_objects"] += int(result.num_gt)
            totals["fp_components"] += int(result.num_fp_components)
            totals["fp_pixels"] += int(result.num_fp_pixels)
            totals["total_pixels"] += int(sample.mask.size)
        rows[domain] = _metrics(totals)
    return {
        "threshold_logit_float32": threshold_logit_float32,
        "all_reject_sentinel": threshold_logit_float32 is None,
        "per_domain": rows,
        "pooled": _aggregate(rows),
    }


def select_domain_oracles(
    enumeration: Mapping[str, Any],
    *,
    pixel_budget: float,
    component_budget: float,
) -> dict[str, Any]:
    """Select per-domain diagnostic Oracles from the losslessly complete states."""

    pixel = _positive_fraction(pixel_budget, name="pixel_budget")
    component = _positive_fraction(component_budget, name="component_budget")
    states = _states(enumeration)
    domains = tuple(sorted(states[0]["per_domain"]))
    output: dict[str, Any] = {}
    for domain in domains:
        candidates: list[Mapping[str, Any]] = []
        for state in states:
            row = state["per_domain"][domain]
            total = int(row["total_pixels"])
            if (
                int(row["fp_pixels"]) * pixel.denominator
                <= pixel.numerator * total
                and int(row["fp_components"]) * 1_000_000 * component.denominator
                <= component.numerator * total
            ):
                candidates.append(state)
        if not candidates:
            output[domain] = {"found": False}
            continue
        chosen = min(
            candidates,
            key=lambda state: (
                -Fraction(
                    int(state["per_domain"][domain]["tp_objects"]),
                    max(1, int(state["per_domain"][domain]["gt_objects"])),
                ),
                math.inf
                if state["all_reject_sentinel"]
                else float(state["threshold_logit_float32"]),
            ),
        )
        output[domain] = {
            "found": True,
            "threshold_logit_float32": chosen["threshold_logit_float32"],
            "all_reject_sentinel": bool(chosen["all_reject_sentinel"]),
            "threshold_state_rank": int(chosen["threshold_state_rank"]),
            "operating_point": dict(chosen["per_domain"][domain]),
        }
    return {
        "diagnostic_only": True,
        "per_domain_oracles_are_not_formal_shared_points": True,
        "pixel_budget": float(pixel_budget),
        "component_budget": float(component_budget),
        "domains": output,
    }


def _state_at_threshold(
    states: Sequence[Mapping[str, Any]], threshold: float | None
) -> Mapping[str, Any]:
    for state in states:
        if threshold is None and state["all_reject_sentinel"]:
            return state
        if (
            threshold is not None
            and not state["all_reject_sentinel"]
            and float(np.float32(state["threshold_logit_float32"]))
            == float(np.float32(threshold))
        ):
            return state
    raise ValueError(f"threshold is absent from exact state enumeration: {threshold}")


def build_cross_domain_calibration_gap(
    enumeration: Mapping[str, Any],
    shared_selection: Mapping[str, Any],
    domain_oracles: Mapping[str, Any],
    *,
    samples_by_domain: Mapping[str, Sequence[RawLogitSample]] | None = None,
) -> dict[str, Any]:
    """Build numeric/rank/cross-application calibration diagnostics."""

    states = _states(enumeration)
    oracle_payload = domain_oracles.get("domains", domain_oracles)
    if not isinstance(oracle_payload, Mapping) or len(oracle_payload) < 2:
        raise ValueError("domain_oracles are missing")
    formal = shared_selection.get("source_pooled", shared_selection)
    if not isinstance(formal, Mapping) or formal.get("found") is not True:
        raise ValueError("shared pooled operating point is missing")
    shared_threshold = formal.get("threshold_logit_float32")

    ranks: dict[str, Any] = {}
    validated_samples = (
        _validated_samples_by_domain(samples_by_domain)
        if samples_by_domain is not None
        else None
    )
    for domain, oracle in oracle_payload.items():
        if not isinstance(oracle, Mapping) or oracle.get("found") is not True:
            raise ValueError(f"domain Oracle is missing for {domain}")
        threshold = oracle.get("threshold_logit_float32")
        if validated_samples is not None and threshold is not None:
            samples = validated_samples[domain]
            total = int(sum(sample.logits.size for sample in samples))
            background = int(sum(np.count_nonzero(~sample.mask) for sample in samples))
            total_exceed = int(
                sum(np.count_nonzero(sample.logits >= np.float32(threshold)) for sample in samples)
            )
            background_exceed = int(
                sum(
                    np.count_nonzero(
                        sample.logits[~sample.mask] >= np.float32(threshold)
                    )
                    for sample in samples
                )
            )
            ranks[domain] = {
                "total_exceedance_rank": total_exceed,
                "total_tail_fraction": float(total_exceed / total),
                "empirical_total_quantile": float(1.0 - total_exceed / total),
                "background_exceedance_rank": background_exceed,
                "background_tail_fraction": float(background_exceed / background)
                if background
                else 0.0,
            }

    cross: dict[str, Any] = {}
    names = sorted(oracle_payload)
    for source in names:
        threshold = oracle_payload[source]["threshold_logit_float32"]
        state = _state_at_threshold(states, threshold)
        source_pd = float(oracle_payload[source]["operating_point"]["pd"])
        for target in names:
            if target == source:
                continue
            row = state["per_domain"][target]
            target_oracle_pd = float(oracle_payload[target]["operating_point"]["pd"])
            cross[f"{source}_threshold_on_{target}"] = {
                "threshold_logit_float32": threshold,
                "target_operating_point": dict(row),
                "target_oracle_pd": target_oracle_pd,
                "target_pd_loss_vs_oracle": float(target_oracle_pd - float(row["pd"])),
                "source_oracle_pd": source_pd,
            }

    shared_state = _state_at_threshold(states, shared_threshold)
    shared_regret = {
        domain: float(
            float(oracle_payload[domain]["operating_point"]["pd"])
            - float(shared_state["per_domain"][domain]["pd"])
        )
        for domain in names
    }
    finite_oracle_thresholds = [
        float(oracle_payload[name]["threshold_logit_float32"])
        for name in names
        if oracle_payload[name]["threshold_logit_float32"] is not None
    ]
    return {
        "diagnostic_only": True,
        "per_domain_oracles_not_used_for_formal_gate": True,
        "formal_shared_threshold_logit_float32": shared_threshold,
        "domain_oracles": dict(oracle_payload),
        "oracle_threshold_numeric_range": (
            float(max(finite_oracle_thresholds) - min(finite_oracle_thresholds))
            if len(finite_oracle_thresholds) >= 2
            else None
        ),
        "tail_rank_and_quantile": ranks,
        "cross_application": cross,
        "shared_threshold_pd_regret_vs_domain_oracle": shared_regret,
    }


__all__ = [
    "DEFAULT_LOOSE_PIXEL_BUDGET",
    "SOURCE_RAW_LOGIT_SCHEMA",
    "build_cross_domain_calibration_gap",
    "enumerate_exact_shared_states",
    "evaluate_domains_at_threshold",
    "select_domain_oracles",
    "select_exact_shared_source_operating_points",
]
