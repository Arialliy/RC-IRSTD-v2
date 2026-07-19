"""Finite-sample grid-rank offset selection for bounded monotone losses.

Offsets are ranks on the shared threshold grid, never probability-distance
increments.  By default every threshold from the zero-label index through the
last grid point is considered, followed by a terminal reject action with zero
loss.  The terminal action is an abstention, not a certified threshold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


SELECTION_SCHEMA_VERSION = "rc-v2-grid-rank-crc-v3-full-coverage"


@dataclass(frozen=True)
class OffsetSelection:
    """Result of finite-sample corrected grid-rank selection."""

    success: bool
    reject: bool
    reason: str
    zero_index: int | None
    zero_threshold_indices: tuple[int, ...]
    selected_threshold_index: int | None
    selected_threshold_indices: tuple[int | None, ...]
    offset_rank: int | None
    alpha: float
    num_calibration_images: int
    minimum_attainable_bound: float
    empirical_loss: float | None
    corrected_loss_bound: float | None
    candidate_offset_ranks: tuple[int, ...]
    candidate_trace: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["schema_version"] = SELECTION_SCHEMA_VERSION
        result["status"] = "selected_operating_point" if self.success else "reject"
        return result


def finite_sample_feasibility(
    num_calibration_images: int, alpha: float
) -> tuple[bool, float]:
    """Check whether the CRC correction can ever be at most ``alpha``.

    For a bounded loss in ``[0, 1]``, the correction used here is
    ``(n * empirical_loss + 1) / (n + 1)``.  Its minimum is ``1/(n+1)``.
    """

    if num_calibration_images <= 0:
        raise ValueError("num_calibration_images must be positive")
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be finite and lie in (0, 1]")
    minimum = 1.0 / (num_calibration_images + 1.0)
    return bool(alpha + 1e-15 >= minimum), float(minimum)


def corrected_empirical_loss(empirical_loss: float, num_images: int) -> float:
    if num_images <= 0:
        raise ValueError("num_images must be positive")
    if not np.isfinite(empirical_loss) or not 0.0 <= empirical_loss <= 1.0:
        raise ValueError("empirical_loss must lie in [0, 1]")
    return float((num_images * empirical_loss + 1.0) / (num_images + 1.0))


def build_grid_rank_candidates(
    *,
    zero_index: int,
    num_thresholds: int,
    candidate_offset_ranks: Sequence[int] | None = None,
    include_terminal_reject: bool = True,
) -> tuple[int, ...]:
    """Build sorted rank offsets, optionally ending in terminal rejection.

    The terminal rank is ``num_thresholds - zero_index``.  Ordinary ranks are
    smaller and map to ``zero_index + rank``.
    """

    if num_thresholds <= 0:
        raise ValueError("num_thresholds must be positive")
    if not 0 <= zero_index < num_thresholds:
        raise ValueError("zero_index lies outside the threshold grid")
    terminal_rank = num_thresholds - zero_index
    if candidate_offset_ranks is None:
        values = list(range(terminal_rank + int(include_terminal_reject)))
    else:
        values = []
        for value in candidate_offset_ranks:
            if isinstance(value, bool) or int(value) != value:
                raise ValueError("candidate offset ranks must be integers")
            rank = int(value)
            maximum = terminal_rank if include_terminal_reject else terminal_rank - 1
            if not 0 <= rank <= maximum:
                raise ValueError(
                    f"offset rank {rank} is outside the allowed range [0, {maximum}]"
                )
            values.append(rank)
        if include_terminal_reject:
            values.append(terminal_rank)
    if not values:
        raise ValueError("At least one candidate offset rank is required")
    return tuple(sorted(set(values)))


def build_adaptive_grid_rank_candidates(
    *,
    zero_indices: Sequence[int] | np.ndarray,
    num_thresholds: int,
    candidate_offset_ranks: Sequence[int] | None = None,
    include_terminal_reject: bool = True,
) -> tuple[int, ...]:
    """Build shared rank offsets for sample-adaptive base thresholds.

    A shared rank ``k`` maps image ``i`` to ``zero_indices[i] + k``.  If that
    index lies beyond the fixed grid, the image abstains.  Such partial-reject
    candidates are retained for diagnostics but are not eligible for formal
    selection: a successful calibration action must cover the entire
    calibration cohort.  The final candidate rejects every image.
    """

    if num_thresholds <= 0:
        raise ValueError("num_thresholds must be positive")
    raw = np.asarray(zero_indices)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("zero_indices must be a non-empty one-dimensional array")
    if raw.dtype.kind not in "iu" and not np.all(np.equal(raw, np.floor(raw))):
        raise ValueError("zero_indices must contain integers")
    values = raw.astype(np.int64)
    if np.any((values < 0) | (values > num_thresholds)):
        raise ValueError(
            "zero_indices must be grid indices or num_thresholds for a pre-rejected sample"
        )
    terminal_rank = int(np.max(num_thresholds - values))
    if candidate_offset_ranks is None:
        ranks = list(range(terminal_rank + int(include_terminal_reject)))
    else:
        ranks = []
        maximum = terminal_rank if include_terminal_reject else terminal_rank - 1
        for value in candidate_offset_ranks:
            if isinstance(value, bool) or int(value) != value:
                raise ValueError("candidate offset ranks must be integers")
            rank = int(value)
            if not 0 <= rank <= maximum:
                raise ValueError(
                    f"offset rank {rank} is outside the allowed range [0, {maximum}]"
                )
            ranks.append(rank)
        if include_terminal_reject:
            ranks.append(terminal_rank)
    if not ranks:
        raise ValueError("At least one candidate offset rank is required")
    return tuple(sorted(set(ranks)))


def _validate_loss_curves(losses: np.ndarray) -> np.ndarray:
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError("losses must have shape [N, T] with N,T > 0")
    if not np.isfinite(values).all():
        raise ValueError("losses contain NaN or infinite values")
    if np.any(values < -1e-12) or np.any(values > 1.0 + 1e-12):
        raise ValueError("losses must lie in [0, 1]")
    values = np.clip(values, 0.0, 1.0)
    if np.any(np.diff(values, axis=1) > 1e-9):
        raise ValueError("loss curves must be non-increasing over threshold rank")
    return values


def select_conformal_offset(
    losses: np.ndarray,
    *,
    zero_index: int | None = None,
    zero_indices: Sequence[int] | np.ndarray | None = None,
    alpha: float,
    candidate_offset_ranks: Sequence[int] | None = None,
    include_terminal_reject: bool = True,
) -> OffsetSelection:
    """Select the smallest feasible non-negative grid-rank offset.

    The returned ``success`` is true only when every calibration image receives
    an actual threshold-grid action and the corrected bound is satisfied.
    Partial-reject candidates remain in ``candidate_trace`` but cannot obtain a
    formal success by replacing hard calibration examples with zero-loss
    abstentions.  If only rejecting actions are feasible, or finite-sample
    feasibility fails, this returns ``success=False`` and ``reject=True``.
    """

    curves = _validate_loss_curves(losses)
    num_images, num_thresholds = curves.shape
    if (zero_index is None) == (zero_indices is None):
        raise ValueError("Provide exactly one of zero_index or zero_indices")
    if zero_indices is None:
        if not 0 <= int(zero_index) < num_thresholds:
            raise ValueError("zero_index lies outside the threshold grid")
        bases = np.full(num_images, int(zero_index), dtype=np.int64)
        scalar_zero_index: int | None = int(zero_index)
        candidates = build_grid_rank_candidates(
            zero_index=int(zero_index),
            num_thresholds=num_thresholds,
            candidate_offset_ranks=candidate_offset_ranks,
            include_terminal_reject=include_terminal_reject,
        )
    else:
        bases = np.asarray(zero_indices)
        if bases.ndim != 1 or bases.shape[0] != num_images:
            raise ValueError(f"zero_indices must have shape [{num_images}]")
        if bases.dtype.kind not in "iu" and not np.all(np.equal(bases, np.floor(bases))):
            raise ValueError("zero_indices must contain integers")
        bases = bases.astype(np.int64)
        if np.any((bases < 0) | (bases > num_thresholds)):
            raise ValueError(
                "zero_indices must be grid indices or num_thresholds for a pre-rejected sample"
            )
        scalar_zero_index = int(bases[0]) if np.all(bases == bases[0]) else None
        candidates = build_adaptive_grid_rank_candidates(
            zero_indices=bases,
            num_thresholds=num_thresholds,
            candidate_offset_ranks=candidate_offset_ranks,
            include_terminal_reject=include_terminal_reject,
        )
    feasible, minimum = finite_sample_feasibility(num_images, alpha)
    if not feasible:
        return OffsetSelection(
            success=False,
            reject=True,
            reason=(
                "finite_sample_infeasible: alpha is below 1/(n+1) "
                f"({alpha:.12g} < {minimum:.12g})"
            ),
            zero_index=scalar_zero_index,
            zero_threshold_indices=tuple(int(value) for value in bases),
            selected_threshold_index=None,
            selected_threshold_indices=(),
            offset_rank=None,
            alpha=float(alpha),
            num_calibration_images=int(num_images),
            minimum_attainable_bound=minimum,
            empirical_loss=None,
            corrected_loss_bound=None,
            candidate_offset_ranks=candidates,
            candidate_trace=(),
        )

    trace: list[dict[str, Any]] = []
    for rank in candidates:
        indices = bases + int(rank)
        active = indices < num_thresholds
        reject_count = int(np.sum(~active))
        per_image_loss = np.zeros(num_images, dtype=np.float64)
        rows = np.flatnonzero(active)
        if rows.size:
            per_image_loss[rows] = curves[rows, indices[rows]]
        empirical = float(np.mean(per_image_loss))
        selected_indices = tuple(
            int(index) if is_active else None
            for index, is_active in zip(indices.tolist(), active.tolist())
        )
        terminal = not bool(np.any(active))
        partial_reject = bool(reject_count and not terminal)
        full_calibration_coverage = bool(np.all(active))
        threshold_index = (
            int(indices[0])
            if full_calibration_coverage and np.all(indices == indices[0])
            else None
        )
        corrected = corrected_empirical_loss(empirical, num_images)
        corrected_bound_satisfies_alpha = bool(corrected <= alpha + 1e-15)
        eligible_for_formal_success = bool(full_calibration_coverage and not terminal)
        trace.append(
            {
                "offset_rank": int(rank),
                "threshold_index": threshold_index,
                "terminal_reject": terminal,
                "partial_reject": partial_reject,
                "reject_count": reject_count,
                "reject_rate": float(np.mean(~active)),
                "calibration_coverage_rate": float(np.mean(active)),
                "empirical_loss": empirical,
                "corrected_loss_bound": corrected,
                "corrected_bound_satisfies_alpha": corrected_bound_satisfies_alpha,
                "eligible_for_formal_success": eligible_for_formal_success,
                "feasible": bool(
                    corrected_bound_satisfies_alpha and eligible_for_formal_success
                ),
            }
        )
        if not corrected_bound_satisfies_alpha:
            continue
        if terminal:
            return OffsetSelection(
                success=False,
                reject=True,
                reason="only_terminal_reject_is_feasible",
                zero_index=scalar_zero_index,
                zero_threshold_indices=tuple(int(value) for value in bases),
                selected_threshold_index=None,
                selected_threshold_indices=selected_indices,
                offset_rank=int(rank),
                alpha=float(alpha),
                num_calibration_images=int(num_images),
                minimum_attainable_bound=minimum,
                empirical_loss=empirical,
                corrected_loss_bound=corrected,
                candidate_offset_ranks=candidates,
                candidate_trace=tuple(trace),
            )
        if not eligible_for_formal_success:
            # A partial reject can make the empirical average arbitrarily small.
            # It is an auditable diagnostic action, never a formal certificate.
            continue
        return OffsetSelection(
            success=True,
            reject=False,
            reason="corrected_bound_satisfies_alpha",
            zero_index=scalar_zero_index,
            zero_threshold_indices=tuple(int(value) for value in bases),
            selected_threshold_index=threshold_index,
            selected_threshold_indices=selected_indices,
            offset_rank=int(rank),
            alpha=float(alpha),
            num_calibration_images=int(num_images),
            minimum_attainable_bound=minimum,
            empirical_loss=empirical,
            corrected_loss_bound=corrected,
            candidate_offset_ranks=candidates,
            candidate_trace=tuple(trace),
        )

    partial_only = any(
        bool(item["corrected_bound_satisfies_alpha"])
        and bool(item["partial_reject"])
        for item in trace
    )
    return OffsetSelection(
        success=False,
        reject=True,
        reason=(
            "only_partial_reject_candidates_satisfy_corrected_bound"
            if partial_only
            else "no_candidate_satisfies_corrected_bound"
        ),
        zero_index=scalar_zero_index,
        zero_threshold_indices=tuple(int(value) for value in bases),
        selected_threshold_index=None,
        selected_threshold_indices=(),
        offset_rank=None,
        alpha=float(alpha),
        num_calibration_images=int(num_images),
        minimum_attainable_bound=minimum,
        empirical_loss=None,
        corrected_loss_bound=None,
        candidate_offset_ranks=candidates,
        candidate_trace=tuple(trace),
    )
