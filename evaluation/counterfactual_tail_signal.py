"""Pure primitives for the unregistered zero-training counterfactual signal.

The module performs no file or dataset I/O and cannot authorize training.  It
defines only deterministic ranking arithmetic that can be tested synthetically
before a later source-only preregistration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatchedTailMetrics:
    num_pairs: int
    raw_tail_pair_accuracy: float
    evidence_tail_pair_accuracy: float
    tail_pair_delta: float
    baseline_inversion_repair_rate: float
    harmful_inversion_rate: float
    net_repair: float
    matched_pair_order_inversion_fraction: float
    factual_counterfactual_margin_median: float


def _finite_vector(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def robust_cross_scale_evidence(drops: object) -> tuple[np.ndarray, np.ndarray]:
    """Return median-minus-MAD evidence and positive-sign consensus per row."""

    array = np.asarray(drops, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError("drops must be a non-empty candidate-by-scale matrix")
    if not np.isfinite(array).all():
        raise ValueError("drops must contain only finite values")
    median = np.median(array, axis=1)
    mad = np.median(np.abs(array - median[:, None]), axis=1)
    evidence = median - mad
    sign_consensus = np.mean(array > 0.0, axis=1)
    if not np.isfinite(evidence).all() or not np.isfinite(sign_consensus).all():
        raise RuntimeError("cross-scale evidence computation became non-finite")
    return evidence, sign_consensus


def matched_tail_pair_metrics(
    target_raw: object,
    clutter_raw: object,
    target_evidence: object,
    clutter_evidence: object,
) -> MatchedTailMetrics:
    """Evaluate strict matched-pair ordering without thresholds or calibration."""

    target_raw_array = _finite_vector(target_raw, name="target_raw")
    clutter_raw_array = _finite_vector(clutter_raw, name="clutter_raw")
    target_evidence_array = _finite_vector(
        target_evidence, name="target_evidence"
    )
    clutter_evidence_array = _finite_vector(
        clutter_evidence, name="clutter_evidence"
    )
    sizes = {
        target_raw_array.size,
        clutter_raw_array.size,
        target_evidence_array.size,
        clutter_evidence_array.size,
    }
    if len(sizes) != 1:
        raise ValueError("all matched-pair arrays must have identical length")
    raw_margin = target_raw_array - clutter_raw_array
    evidence_margin = target_evidence_array - clutter_evidence_array
    raw_correct = raw_margin > 0.0
    evidence_correct = evidence_margin > 0.0
    repaired = (~raw_correct) & evidence_correct
    harmed = raw_correct & (~evidence_correct)
    changed = raw_correct != evidence_correct
    raw_accuracy = float(np.mean(raw_correct))
    evidence_accuracy = float(np.mean(evidence_correct))
    tail_pair_delta = evidence_accuracy - raw_accuracy
    repair_rate = float(np.mean(repaired))
    harm_rate = float(np.mean(harmed))
    net_repair = repair_rate - harm_rate
    inversion_fraction = float(np.mean(changed))
    margin_median = float(np.median(evidence_margin))
    if not np.isclose(
        tail_pair_delta,
        net_repair,
        rtol=0.0,
        atol=8.0 * np.finfo(np.float64).eps,
    ):
        raise RuntimeError("matched-pair delta and net repair are inconsistent")
    return MatchedTailMetrics(
        num_pairs=int(target_raw_array.size),
        raw_tail_pair_accuracy=raw_accuracy,
        evidence_tail_pair_accuracy=evidence_accuracy,
        tail_pair_delta=tail_pair_delta,
        baseline_inversion_repair_rate=repair_rate,
        harmful_inversion_rate=harm_rate,
        net_repair=net_repair,
        matched_pair_order_inversion_fraction=inversion_fraction,
        factual_counterfactual_margin_median=margin_median,
    )


__all__ = [
    "MatchedTailMetrics",
    "matched_tail_pair_metrics",
    "robust_cross_scale_evidence",
]

def _bilinear_sample_last2(
    array: np.ndarray,
    *,
    y: float,
    x: float,
    forbidden_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Sample all leading channels at one in-bounds spatial coordinate."""

    height, width = array.shape[-2:]
    if not (0.0 <= y <= height - 1 and 0.0 <= x <= width - 1):
        raise ValueError("annular sample coordinate is outside the image")
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    y1 = min(y0 + 1, height - 1)
    x1 = min(x0 + 1, width - 1)
    if forbidden_mask is not None:
        if forbidden_mask.shape != (height, width):
            raise ValueError("forbidden_mask shape differs from spatial axes")
        if any(
            forbidden_mask[yy, xx]
            for yy, xx in ((y0, x0), (y0, x1), (y1, x0), (y1, x1))
        ):
            raise ValueError(
                "annular bilinear support intersects the factual core"
            )
    wy = float(y - y0)
    wx = float(x - x0)
    return (
        (1.0 - wy) * (1.0 - wx) * array[..., y0, x0]
        + (1.0 - wy) * wx * array[..., y0, x1]
        + wy * (1.0 - wx) * array[..., y1, x0]
        + wy * wx * array[..., y1, x1]
    )


def opposite_ray_annular_core_fill(
    image: object,
    *,
    center_yx: tuple[float, float],
    core_mask: object,
    annulus_inner_radius: float,
    annulus_outer_radius: float,
    radial_samples: int = 5,
    center_angular_samples: int = 32,
) -> np.ndarray:
    """Fill a core from its own annulus by deterministic opposite-ray chords.

    Spatial axes are the final two axes, so both H x W and C x H x W float32
    arrays are supported. Every non-central core pixel is filled by
    interpolating between robust values sampled on opposite annular rays. The
    exact center uses the median over a fixed angular/radial annular grid.

    The function performs no model or dataset I/O and does not estimate a
    causal effect. It fails closed when the full annulus is not observable.
    """

    source = np.asarray(image)
    if source.ndim < 2 or 0 in source.shape:
        raise ValueError("image must have non-empty spatial axes")
    if source.dtype != np.dtype(np.float32):
        raise ValueError("image must have dtype float32")
    if not np.isfinite(source).all():
        raise ValueError("image must contain only finite values")
    height, width = source.shape[-2:]
    mask = np.asarray(core_mask)
    if mask.dtype != np.dtype(np.bool_) or mask.shape != (height, width):
        raise ValueError("core_mask must be a boolean image-sized array")
    core_y, core_x = np.nonzero(mask)
    if core_y.size == 0:
        raise ValueError("core_mask must contain at least one pixel")

    center = np.asarray(center_yx, dtype=np.float64)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center_yx must contain two finite coordinates")
    center_y = float(center[0])
    center_x = float(center[1])
    inner = float(annulus_inner_radius)
    outer = float(annulus_outer_radius)
    if not np.isfinite([inner, outer]).all() or not (0.0 < inner < outer):
        raise ValueError("annulus radii must satisfy 0 < inner < outer")
    if (
        center_y - outer < 0.0
        or center_y + outer > height - 1
        or center_x - outer < 0.0
        or center_x + outer > width - 1
    ):
        raise ValueError("the full annulus must lie inside the image")
    if (
        isinstance(radial_samples, bool)
        or not isinstance(radial_samples, int)
        or radial_samples < 2
    ):
        raise ValueError("radial_samples must be an integer of at least two")
    if (
        isinstance(center_angular_samples, bool)
        or not isinstance(center_angular_samples, int)
        or center_angular_samples < 8
    ):
        raise ValueError(
            "center_angular_samples must be an integer of at least eight"
        )

    core_radius = np.hypot(core_y - center_y, core_x - center_x)
    if np.any(core_radius >= inner):
        raise ValueError("every core pixel must lie strictly inside the annulus")
    work = source.astype(np.float64, copy=True)
    flat = work.reshape((-1, height, width))
    radii = np.linspace(inner, outer, radial_samples, dtype=np.float64)
    reference_radius = float(np.median(radii))
    epsilon = 8.0 * np.finfo(np.float64).eps
    center_angles = np.linspace(
        0.0,
        2.0 * np.pi,
        center_angular_samples,
        endpoint=False,
        dtype=np.float64,
    )

    for y_index, x_index, distance in zip(
        core_y.tolist(),
        core_x.tolist(),
        core_radius.tolist(),
        strict=True,
    ):
        if distance <= epsilon:
            center_values = [
                _bilinear_sample_last2(
                    flat,
                    y=center_y + radius * float(np.sin(angle)),
                    x=center_x + radius * float(np.cos(angle)),
                    forbidden_mask=mask,
                )
                for radius in radii
                for angle in center_angles
            ]
            fill_value = np.median(np.stack(center_values, axis=0), axis=0)
        else:
            unit_y = (float(y_index) - center_y) / distance
            unit_x = (float(x_index) - center_x) / distance
            positive_values = np.stack(
                [
                    _bilinear_sample_last2(
                        flat,
                        y=center_y + radius * unit_y,
                        x=center_x + radius * unit_x,
                        forbidden_mask=mask,
                    )
                    for radius in radii
                ],
                axis=0,
            )
            negative_values = np.stack(
                [
                    _bilinear_sample_last2(
                        flat,
                        y=center_y - radius * unit_y,
                        x=center_x - radius * unit_x,
                        forbidden_mask=mask,
                    )
                    for radius in radii
                ],
                axis=0,
            )
            positive = np.median(positive_values, axis=0)
            negative = np.median(negative_values, axis=0)
            interpolation = 0.5 * (1.0 + distance / reference_radius)
            fill_value = (
                (1.0 - interpolation) * negative
                + interpolation * positive
            )
        flat[:, y_index, x_index] = fill_value

    result = work.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise RuntimeError("annular fill became non-finite")
    return result


__all__ = [*__all__, "opposite_ray_annular_core_fill"]


@dataclass(frozen=True)
class StrictRankInversionMetrics:
    num_candidates: int
    num_comparable_pairs: int
    num_strict_inversions: int
    strict_inversion_fraction: float


def strict_pairwise_rank_inversion_metrics(
    raw_score: object,
    evidence_score: object,
) -> StrictRankInversionMetrics:
    """Measure strict order reversals, conservatively excluding every tie."""

    raw = _finite_vector(raw_score, name="raw_score")
    evidence = _finite_vector(evidence_score, name="evidence_score")
    if raw.size != evidence.size:
        raise ValueError("raw_score and evidence_score must have identical length")
    if raw.size < 2:
        raise ValueError("at least two candidates are required")
    left, right = np.triu_indices(raw.size, k=1)
    raw_difference = raw[left] - raw[right]
    evidence_difference = evidence[left] - evidence[right]
    comparable = (raw_difference != 0.0) & (evidence_difference != 0.0)
    comparable_count = int(np.sum(comparable))
    if comparable_count == 0:
        raise ValueError("no strict non-tied candidate pairs are comparable")
    inversions = (
        raw_difference[comparable] * evidence_difference[comparable]
    ) < 0.0
    inversion_count = int(np.sum(inversions))
    return StrictRankInversionMetrics(
        num_candidates=int(raw.size),
        num_comparable_pairs=comparable_count,
        num_strict_inversions=inversion_count,
        strict_inversion_fraction=float(inversion_count / comparable_count),
    )


__all__ = [
    *__all__,
    "StrictRankInversionMetrics",
    "strict_pairwise_rank_inversion_metrics",
]


@dataclass(frozen=True)
class CandidateMatch:
    target_id: str
    clutter_id: str
    stratum: str
    target_index: int
    clutter_index: int
    match_slot: int
    l1_distance: float


def _strict_string_vector(values: object, *, name: str) -> tuple[str, ...]:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    result = tuple(array.tolist())
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{name} must contain only non-empty strings")
    return result


def deterministic_stratified_greedy_match(
    *,
    target_ids: object,
    clutter_ids: object,
    target_strata: object,
    clutter_strata: object,
    target_features: object,
    clutter_features: object,
    clutter_per_target: int = 1,
) -> tuple[CandidateMatch, ...]:
    """Match in frozen strata with deterministic ID and distance tie breaks.

    Feature columns must already use preregistered scaling. Matching is
    one-to-many without clutter reuse. Targets are processed by target ID; each
    eligible clutter is ranked by L1 feature distance and then clutter ID.
    """

    targets = _strict_string_vector(target_ids, name="target_ids")
    clutters = _strict_string_vector(clutter_ids, name="clutter_ids")
    target_group = _strict_string_vector(target_strata, name="target_strata")
    clutter_group = _strict_string_vector(clutter_strata, name="clutter_strata")
    if len(set(targets)) != len(targets):
        raise ValueError("target_ids must be unique")
    if len(set(clutters)) != len(clutters):
        raise ValueError("clutter_ids must be unique")
    if len(target_group) != len(targets):
        raise ValueError("target_strata length must match target_ids")
    if len(clutter_group) != len(clutters):
        raise ValueError("clutter_strata length must match clutter_ids")
    if (
        isinstance(clutter_per_target, bool)
        or not isinstance(clutter_per_target, int)
        or clutter_per_target < 1
    ):
        raise ValueError("clutter_per_target must be a positive integer")

    target_matrix = np.asarray(target_features, dtype=np.float64)
    clutter_matrix = np.asarray(clutter_features, dtype=np.float64)
    if (
        target_matrix.ndim != 2
        or clutter_matrix.ndim != 2
        or target_matrix.shape[0] != len(targets)
        or clutter_matrix.shape[0] != len(clutters)
        or target_matrix.shape[1] == 0
        or target_matrix.shape[1] != clutter_matrix.shape[1]
    ):
        raise ValueError(
            "feature matrices must be candidate-by-common-nonempty-feature"
        )
    if not np.isfinite(target_matrix).all() or not np.isfinite(
        clutter_matrix
    ).all():
        raise ValueError("feature matrices must contain only finite values")

    required_by_stratum: dict[str, int] = {}
    available_by_stratum: dict[str, int] = {}
    for stratum in target_group:
        required_by_stratum[stratum] = (
            required_by_stratum.get(stratum, 0) + clutter_per_target
        )
    for stratum in clutter_group:
        available_by_stratum[stratum] = available_by_stratum.get(stratum, 0) + 1
    shortages = {
        stratum: (required, available_by_stratum.get(stratum, 0))
        for stratum, required in required_by_stratum.items()
        if available_by_stratum.get(stratum, 0) < required
    }
    if shortages:
        raise ValueError(f"insufficient clutter coverage by stratum: {shortages}")

    used_clutter: set[int] = set()
    matches: list[CandidateMatch] = []
    target_order = sorted(range(len(targets)), key=lambda index: targets[index])
    for target_index in target_order:
        stratum = target_group[target_index]
        eligible = [
            clutter_index
            for clutter_index in range(len(clutters))
            if clutter_index not in used_clutter
            and clutter_group[clutter_index] == stratum
        ]
        ranked = sorted(
            eligible,
            key=lambda clutter_index: (
                float(
                    np.sum(
                        np.abs(
                            target_matrix[target_index]
                            - clutter_matrix[clutter_index]
                        )
                    )
                ),
                clutters[clutter_index],
            ),
        )
        chosen = ranked[:clutter_per_target]
        if len(chosen) != clutter_per_target:
            raise RuntimeError("prechecked clutter coverage became inconsistent")
        for match_slot, clutter_index in enumerate(chosen):
            used_clutter.add(clutter_index)
            matches.append(
                CandidateMatch(
                    target_id=targets[target_index],
                    clutter_id=clutters[clutter_index],
                    stratum=stratum,
                    target_index=target_index,
                    clutter_index=clutter_index,
                    match_slot=match_slot,
                    l1_distance=float(
                        np.sum(
                            np.abs(
                                target_matrix[target_index]
                                - clutter_matrix[clutter_index]
                            )
                        )
                    ),
                )
            )
    if len(matches) != len(targets) * clutter_per_target:
        raise RuntimeError("deterministic matching returned an incomplete matrix")
    return tuple(matches)


__all__ = [
    *__all__,
    "CandidateMatch",
    "deterministic_stratified_greedy_match",
]



def _rectangular_minimum_cost_assignment(cost: object) -> np.ndarray:
    """Solve a finite rectangular linear assignment deterministically."""

    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("cost must be a non-empty two-dimensional matrix")
    if matrix.shape[0] > matrix.shape[1]:
        raise ValueError("assignment requires at least as many columns as rows")
    if not np.isfinite(matrix).all():
        raise ValueError("cost must contain only finite values")

    row_count, column_count = matrix.shape
    row_potential = np.zeros(row_count + 1, dtype=np.float64)
    column_potential = np.zeros(column_count + 1, dtype=np.float64)
    column_row = np.zeros(column_count + 1, dtype=np.int64)
    predecessor = np.zeros(column_count + 1, dtype=np.int64)

    for row in range(1, row_count + 1):
        column_row[0] = row
        minimum = np.full(column_count + 1, np.inf, dtype=np.float64)
        used = np.zeros(column_count + 1, dtype=np.bool_)
        column = 0
        while True:
            used[column] = True
            active_row = int(column_row[column])
            delta = np.inf
            next_column = -1
            for candidate_column in range(1, column_count + 1):
                if used[candidate_column]:
                    continue
                reduced = (
                    matrix[active_row - 1, candidate_column - 1]
                    - row_potential[active_row]
                    - column_potential[candidate_column]
                )
                if reduced < minimum[candidate_column]:
                    minimum[candidate_column] = reduced
                    predecessor[candidate_column] = column
                if (
                    minimum[candidate_column] < delta
                    or (
                        minimum[candidate_column] == delta
                        and (
                            next_column < 0
                            or candidate_column < next_column
                        )
                    )
                ):
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            if next_column < 0 or not np.isfinite(delta):
                raise RuntimeError("minimum-cost assignment became infeasible")
            for candidate_column in range(column_count + 1):
                if used[candidate_column]:
                    row_potential[column_row[candidate_column]] += delta
                    column_potential[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if column_row[column] == 0:
                break

        while True:
            previous = int(predecessor[column])
            column_row[column] = column_row[previous]
            column = previous
            if column == 0:
                break

    assignment = np.full(row_count, -1, dtype=np.int64)
    for column in range(1, column_count + 1):
        assigned_row = int(column_row[column])
        if assigned_row != 0:
            assignment[assigned_row - 1] = column - 1
    if np.any(assignment < 0) or len(set(assignment.tolist())) != row_count:
        raise RuntimeError("minimum-cost assignment is incomplete")
    return assignment


def deterministic_stratified_min_cost_match(
    *,
    target_ids: object,
    clutter_ids: object,
    target_strata: object,
    clutter_strata: object,
    target_features: object,
    clutter_features: object,
    clutter_per_target: int = 1,
) -> tuple[CandidateMatch, ...]:
    """Globally minimize within-stratum L1 cost without clutter reuse.

    Target slots and clutter columns are sorted by stable IDs before solving.
    Equal reduced costs select the lowest pending clutter-ID column, making the
    exact implementation deterministic and invariant to input row order.
    Feature columns must already use preregistered scaling.
    """

    targets = _strict_string_vector(target_ids, name="target_ids")
    clutters = _strict_string_vector(clutter_ids, name="clutter_ids")
    target_group = _strict_string_vector(target_strata, name="target_strata")
    clutter_group = _strict_string_vector(clutter_strata, name="clutter_strata")
    if len(set(targets)) != len(targets):
        raise ValueError("target_ids must be unique")
    if len(set(clutters)) != len(clutters):
        raise ValueError("clutter_ids must be unique")
    if len(target_group) != len(targets):
        raise ValueError("target_strata length must match target_ids")
    if len(clutter_group) != len(clutters):
        raise ValueError("clutter_strata length must match clutter_ids")
    if (
        isinstance(clutter_per_target, bool)
        or not isinstance(clutter_per_target, int)
        or clutter_per_target < 1
    ):
        raise ValueError("clutter_per_target must be a positive integer")

    target_matrix = np.asarray(target_features, dtype=np.float64)
    clutter_matrix = np.asarray(clutter_features, dtype=np.float64)
    if (
        target_matrix.ndim != 2
        or clutter_matrix.ndim != 2
        or target_matrix.shape[0] != len(targets)
        or clutter_matrix.shape[0] != len(clutters)
        or target_matrix.shape[1] == 0
        or target_matrix.shape[1] != clutter_matrix.shape[1]
    ):
        raise ValueError(
            "feature matrices must be candidate-by-common-nonempty-feature"
        )
    if not np.isfinite(target_matrix).all() or not np.isfinite(
        clutter_matrix
    ).all():
        raise ValueError("feature matrices must contain only finite values")

    matches: list[CandidateMatch] = []
    for stratum in sorted(set(target_group)):
        target_indices = sorted(
            (
                index
                for index, value in enumerate(target_group)
                if value == stratum
            ),
            key=lambda index: targets[index],
        )
        clutter_indices = sorted(
            (
                index
                for index, value in enumerate(clutter_group)
                if value == stratum
            ),
            key=lambda index: clutters[index],
        )
        required = len(target_indices) * clutter_per_target
        if len(clutter_indices) < required:
            raise ValueError(
                "insufficient clutter coverage by stratum: "
                f"{{'{stratum}': ({required}, {len(clutter_indices)})}}"
            )
        slot_targets = [
            target_index
            for target_index in target_indices
            for _ in range(clutter_per_target)
        ]
        cost = np.sum(
            np.abs(
                target_matrix[np.asarray(slot_targets), None, :]
                - clutter_matrix[np.asarray(clutter_indices)][None, :, :]
            ),
            axis=2,
        )
        local_assignment = _rectangular_minimum_cost_assignment(cost)
        assigned_by_target: dict[int, list[int]] = {
            target_index: [] for target_index in target_indices
        }
        for slot_index, local_clutter_index in enumerate(
            local_assignment.tolist()
        ):
            assigned_by_target[slot_targets[slot_index]].append(
                clutter_indices[local_clutter_index]
            )

        for target_index in target_indices:
            assigned = sorted(
                assigned_by_target[target_index],
                key=lambda clutter_index: clutters[clutter_index],
            )
            if len(assigned) != clutter_per_target:
                raise RuntimeError("minimum-cost target slots are incomplete")
            for match_slot, clutter_index in enumerate(assigned):
                distance = float(
                    np.sum(
                        np.abs(
                            target_matrix[target_index]
                            - clutter_matrix[clutter_index]
                        )
                    )
                )
                matches.append(
                    CandidateMatch(
                        target_id=targets[target_index],
                        clutter_id=clutters[clutter_index],
                        stratum=stratum,
                        target_index=target_index,
                        clutter_index=clutter_index,
                        match_slot=match_slot,
                        l1_distance=distance,
                    )
                )

    matches.sort(
        key=lambda match: (
            match.target_id,
            match.match_slot,
            match.clutter_id,
        )
    )
    if len(matches) != len(targets) * clutter_per_target:
        raise RuntimeError("minimum-cost matching returned an incomplete matrix")
    if len({match.clutter_id for match in matches}) != len(matches):
        raise RuntimeError("minimum-cost matching reused a clutter candidate")
    return tuple(matches)


__all__ = [
    *__all__,
    "deterministic_stratified_min_cost_match",
]
