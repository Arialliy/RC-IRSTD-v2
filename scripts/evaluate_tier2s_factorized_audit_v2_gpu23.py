#!/usr/bin/env python3
"""Source-only exploratory audit for factorized RC-MSHNet logits.

This module is intentionally separate from the frozen Tier2R gate.  It reads
the old matched ``control``/``c`` bindings, verifies new factorized exports,
and evaluates four *exploratory* routes:

``raw_final``
    The exported final logit.
``alpha_selected``
    ``base + alpha * residual``, where one alpha is selected for each
    seed/fold from a preregistered held-in ID subset only.
``calibrated_final``
    Final logits mapped to held-in-background empirical tail coordinates.
``calibrated_alpha``
    Alpha-composed logits mapped in the same source-only manner.

The empirical transform is strictly increasing in real arithmetic (apart
from equality ties already present in the input).  Consequently it cannot
improve a single-domain ranking; its only intended purpose is to align score
coordinates across the two LODO folds before applying one shared threshold.
The FP32 application audit reports any additional ties caused by rounding.

No output from this script authorizes Tier3 or outer-target access.  The old
Tier2R HOLD is hash-checked before and after evaluation and remains immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.artifact_integrity import (
    file_sha256,
    ordered_ids_sha256,
    score_records_sha256,
)
from evaluation.raw_logit_oracle import (
    RawLogitSample,
    load_formal_raw_logit_directory,
)
from scripts import export_tier2s_factorized_logits_v2_gpu23 as tier2s_exporter
from scripts import register_tier2s_gpu23_amendment_v2 as governance_registrar
from scripts.run_phase3_raw_logit_rescue_v1 import _extract_gate_point
from scripts.run_phase3_tier2r_exact_gate import (
    BUDGETS,
    FOLDS,
    MATCHING_PROTOCOL,
    NUMERIC_ATOL,
    SEEDS,
    ExactAPI,
    _compute_exact_points,
    evaluate_gate_level,
)

EXPORT_SCHEMA = "rc-irstd-tier2s-factorized-logit-export-v2-gpu23"
PROTOCOL_SCHEMA = "rc-irstd-aaai27-tier2s-factorized-causal-audit-protocol-v2-gpu23"
HANDOFF_SCHEMA = "rc-irstd-aaai27-tier2s-factorized-export-handoff-v2-gpu23"
QUEUE_SCHEMA = "rc-irstd-aaai27-tier2s-fixed-two-lane-queue-v2-gpu23"
EVENT_SCHEMA = "rc-irstd-aaai27-tier2s-scheduler-event-v2-gpu23"
QUEUE_SCHEDULER = "two_fixed_independent_fifo_lanes"
QUEUE_PHYSICAL_GPUS: tuple[int, ...] = (2, 3)
CONTAINER_ORDINAL_BY_PHYSICAL = {2: 0, 3: 1}
QUEUE_JOBS_PER_LANE = 9
RESULT_SCHEMA = "rc-irstd-aaai27-tier2s-factorized-audit-result-v2-gpu23"
OLD_HANDOFF_SCHEMA = "rc-irstd-aaai27-tier2r-exact-gate-handoff-v1"
OLD_DECISION_SCHEMA = "rc-irstd-aaai27-tier2r-exact-decision-v1"
PROTOCOL_ID = "tier2s_factorized_causal_audit_v2_gpu23"
PROTOCOL_PATH = PROJECT_ROOT / "configs/tier2s_factorized_causal_audit_v2_gpu23.json"
OLD_HANDOFF_PATH = (
    PROJECT_ROOT
    / "artifacts/aaai27/audit/component_rescue/tier2r_c_v1_impl_erratum1"
    / "TIER2R_HANDOFF.json"
)

ALPHAS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
ROLES: tuple[str, ...] = ("control", "c")
SUBSET_ROLES: tuple[str, ...] = ("held_in", "held_out")
ROUTES: tuple[str, ...] = (
    "raw_final",
    "source_selected_alpha",
    "source_tail_calibrated_final",
    "source_tail_calibrated_selected_alpha",
)
ALPHA_PER_BUDGET_MAX_REGRESSION = 0.005
FACTOR_REPLAY_ATOL = 1.0e-7
FACTOR_REPLAY_RTOL = 1.0e-6

STRICT_MONOTONE_SCOPE_NOTE = (
    "Any strictly increasing transform applied within one domain preserves "
    "that domain's score ordering and therefore cannot repair single-domain "
    "ranking. The registered empirical survival transform is nondecreasing "
    "and may only coarsen ranks with ties; it cannot reverse or improve their "
    "order and is used only to align coordinates across LODO folds."
)


@dataclass(frozen=True)
class FactorizedSample:
    """One integrity-checked factorized logit record."""

    image_id: str
    dataset_name: str
    subset_role: str
    base: np.ndarray
    final: np.ndarray
    residual: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class EmpiricalTailTransform:
    """Registered empirical background survival transform."""

    sorted_background: np.ndarray
    num_background_pixels: int
    num_unique_background_scores: int

    def apply(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim != 2 or not np.issubdtype(array.dtype, np.floating):
            raise ValueError("tail transform input must be a 2-D floating array")
        if not np.isfinite(array).all():
            raise ValueError("tail transform input contains non-finite scores")
        flat = array.astype(np.float32, copy=False).reshape(-1)
        count_less = np.searchsorted(self.sorted_background, flat, side="left")
        count_greater_equal = self.num_background_pixels - count_less
        survival = (count_greater_equal + 1.0) / (
            self.num_background_pixels + 1.0
        )
        mapped = -np.log10(survival)
        if not np.isfinite(mapped).all():
            raise ValueError("tail transform produced non-finite coordinates")
        return np.ascontiguousarray(mapped.reshape(array.shape).astype(np.float32))

    def summary(self) -> dict[str, Any]:
        return {
            "kind": "negative_log10_empirical_background_survival",
            "empirical_survival": (
                "(count_calibration_background_logits_greater_equal_z + 1) / "
                "(num_calibration_background_pixels + 1)"
            ),
            "num_background_pixels": self.num_background_pixels,
            "num_unique_background_scores": self.num_unique_background_scores,
            "minimum_background_score": float(self.sorted_background[0]),
            "maximum_background_score": float(self.sorted_background[-1]),
            "monotonicity": "nondecreasing_in_raw_logit",
            "single_domain_ranking_changes": False,
            "single_domain_scope_note": STRICT_MONOTONE_SCOPE_NOTE,
        }


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_once_frozen_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
        created = True
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
    except FileExistsError:
        pass
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise

    if (
        path.is_symlink()
        or not path.is_file()
        or path.read_bytes() != content
        or path.stat().st_mode & 0o777 != 0o444
    ):
        raise RuntimeError(f"immutable no-replace artifact drift: {path}")


def _write_once_frozen_json(path: Path, payload: Mapping[str, Any]) -> str:
    content = _canonical_json_bytes(dict(payload))
    digest = hashlib.sha256(content).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    _write_once_frozen_bytes(path, content)
    _write_once_frozen_bytes(
        sidecar, f"{digest}  {path.name}\n".encode("ascii")
    )
    _verify_frozen_sidecar(path, expected_sha256=digest)
    return digest


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required JSON is absent or a symlink: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return payload


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_file(path: Path, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"bound file is absent or a symlink: {path}")
    if not _valid_sha256(expected_sha256) or file_sha256(path) != expected_sha256:
        raise RuntimeError(f"SHA-256 binding drift: {path}")


def _verify_frozen_sidecar(
    path: Path, *, expected_sha256: str | None = None
) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
        raise RuntimeError(f"frozen artifact is absent, aliased, or writable: {path}")
    digest = file_sha256(path)
    if expected_sha256 is not None and (
        not _valid_sha256(expected_sha256) or digest != expected_sha256
    ):
        raise RuntimeError(f"frozen artifact SHA-256 drift: {path}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if (
        sidecar.is_symlink()
        or not sidecar.is_file()
        or sidecar.stat().st_mode & 0o222
        or sidecar.read_text(encoding="ascii")
        != f"{digest}  {path.name}\n"
    ):
        raise RuntimeError(f"frozen artifact sidecar drift: {sidecar}")
    return digest


def _verify_scheduler_event_log(
    path: Path,
    *,
    expected_sha256: str,
    expected_jobs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_path = path.resolve()
    digest = _verify_frozen_sidecar(expected_path, expected_sha256=expected_sha256)
    previous = "0" * 64
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        expected_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Tier2S scheduler event {line_number} is not JSON"
            ) from error
        if not isinstance(record, dict):
            raise RuntimeError("Tier2S scheduler event is not an object")
        claimed = record.pop("event_sha256", None)
        if (
            record.get("schema_version")
            != EVENT_SCHEMA
            or record.get("previous_event_sha256") != previous
            or not _valid_sha256(claimed)
            or hashlib.sha256(_canonical_json_bytes(record)).hexdigest() != claimed
        ):
            raise RuntimeError("Tier2S scheduler event hash chain drift")
        previous = claimed
        records.append({**record, "event_sha256": claimed})

    expected = {
        str(job.get("run_id")): (
            job.get("physical_gpu"),
            job.get("container_gpu_ordinal"),
            job.get("queue_index"),
        )
        for job in expected_jobs
    }
    starts = [record for record in records if record.get("event") == "job_started"]
    completions = [
        record for record in records if record.get("event") == "job_completed"
    ]
    terminal = [
        record
        for record in records
        if record.get("event") == "all_exports_completed"
    ]
    if (
        len(records) != 37
        or len(starts) != 18
        or len(completions) != 18
        or len(terminal) != 1
        or terminal[0].get("completed_jobs") != 18
        or any(record.get("event") == "job_failed" for record in records)
    ):
        raise RuntimeError("Tier2S scheduler event set is incomplete")
    if records[-1] != terminal[0]:
        raise RuntimeError("Tier2S scheduler terminal event is not last")
    for role, rows in (("start", starts), ("completion", completions)):
        observed: dict[str, tuple[Any, Any, Any]] = {}
        for row in rows:
            run_id = str(row.get("run_id", ""))
            if run_id in observed:
                raise RuntimeError(f"duplicate Tier2S scheduler {role} event")
            observed[run_id] = (
                row.get("physical_gpu"),
                row.get("container_gpu_ordinal"),
                row.get("queue_index"),
            )
        if observed != expected:
            raise RuntimeError(f"Tier2S scheduler {role} lane binding drift")

    # The start/completion sets above prove identity coverage, but not execution
    # order. Replay the log as two independent FIFO state machines so arbitrary
    # cross-lane interleaving remains valid while each lane must execute exactly
    # start(q) -> completion(q) for q=0..8.
    lane_state: dict[int, dict[str, Any]] = {
        gpu: {"next_queue_index": 0, "active_job": None}
        for gpu in QUEUE_PHYSICAL_GPUS
    }
    for row in records[:-1]:
        event = row.get("event")
        gpu = row.get("physical_gpu")
        container_ordinal = row.get("container_gpu_ordinal")
        queue_index = row.get("queue_index")
        run_id = str(row.get("run_id", ""))
        if (
            event not in {"job_started", "job_completed"}
            or type(gpu) is not int
            or gpu not in lane_state
            or container_ordinal != CONTAINER_ORDINAL_BY_PHYSICAL.get(gpu)
            or type(queue_index) is not int
        ):
            raise RuntimeError("Tier2S scheduler FIFO event shape drift")
        state = lane_state[gpu]
        active_job = state["active_job"]
        next_queue_index = state["next_queue_index"]
        if event == "job_started":
            if active_job is not None:
                raise RuntimeError(
                    "Tier2S scheduler lane started next job before prior completion"
                )
            if queue_index != next_queue_index:
                raise RuntimeError("Tier2S scheduler lane FIFO start order drift")
            state["active_job"] = (run_id, queue_index)
            continue
        if active_job is None:
            raise RuntimeError(
                "Tier2S scheduler lane completion occurred before matching start"
            )
        if active_job != (run_id, queue_index) or queue_index != next_queue_index:
            raise RuntimeError("Tier2S scheduler lane completion order drift")
        state["active_job"] = None
        state["next_queue_index"] = next_queue_index + 1

    if any(
        state["active_job"] is not None
        or state["next_queue_index"] != QUEUE_JOBS_PER_LANE
        for state in lane_state.values()
    ):
        raise RuntimeError("Tier2S scheduler lane FIFO sequence is incomplete")
    sidecar = expected_path.with_suffix(expected_path.suffix + ".sha256")
    return {
        "path": str(expected_path),
        "sha256": digest,
        "sidecar_path": str(sidecar),
        "sidecar_sha256": file_sha256(sidecar),
        "num_events": len(records),
        "terminal_event_sha256": terminal[0]["event_sha256"],
    }


def _require_chain_binding(
    payload: Mapping[str, Any],
    field: str,
    expected: Mapping[str, Any],
    *,
    where: str,
) -> None:
    if payload.get(field) != dict(expected):
        raise RuntimeError(f"{where}.{field} drift")


def _validate_fixed_two_lane_queue(
    queue: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    jobs = queue.get("jobs")
    if (
        queue.get("schema_version") != QUEUE_SCHEMA
        or queue.get("protocol_id") != PROTOCOL_ID
        or queue.get("scheduler") != QUEUE_SCHEDULER
        or queue.get("wait_for_idle_gpu") is not False
        or queue.get("allow_gpu_fallback") is not False
        or not isinstance(jobs, list)
        or len(jobs) != len(QUEUE_PHYSICAL_GPUS) * QUEUE_JOBS_PER_LANE
    ):
        raise RuntimeError("Tier2S fixed-two-lane queue identity drift")

    lane_indices = {gpu: [] for gpu in QUEUE_PHYSICAL_GPUS}
    run_ids: set[str] = set()
    validated: list[Mapping[str, Any]] = []
    for raw_job in jobs:
        if not isinstance(raw_job, Mapping):
            raise RuntimeError("Tier2S queue job is not a mapping")
        gpu = raw_job.get("physical_gpu")
        container_ordinal = raw_job.get("container_gpu_ordinal")
        queue_index = raw_job.get("queue_index")
        run_id = raw_job.get("run_id")
        if (
            type(gpu) is not int
            or gpu not in lane_indices
            or container_ordinal != CONTAINER_ORDINAL_BY_PHYSICAL.get(gpu)
            or type(queue_index) is not int
            or not isinstance(run_id, str)
            or not run_id
            or run_id in run_ids
        ):
            raise RuntimeError("Tier2S queue job identity/lane drift")
        lane_indices[gpu].append(queue_index)
        run_ids.add(run_id)
        validated.append(raw_job)

    expected_indices = list(range(QUEUE_JOBS_PER_LANE))
    if any(
        sorted(indices) != expected_indices
        for indices in lane_indices.values()
    ):
        raise RuntimeError("Tier2S queue lane length/index drift")
    return validated


def _safe_record_path(root: Path, raw: Any) -> Path:
    relative = Path(str(raw))
    if (
        not str(raw)
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.suffix.lower() != ".npz"
        or relative.parts[:1] != ("records",)
    ):
        raise ValueError(f"unsafe factorized record path: {raw!r}")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"factorized record is absent or a symlink: {path}")
    return path


def _scalar_string(value: Any, *, name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be a string scalar")
    return str(array.item())


def _scalar_bool(value: Any, *, name: str) -> bool:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean scalar")
    return bool(array.item())


def _integer_pair(value: Any, *, name: str) -> tuple[int, int]:
    array = np.asarray(value)
    if array.shape != (2,) or array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be a two-element integer array")
    result = int(array[0]), int(array[1])
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError(f"{name} must contain positive dimensions")
    return result


def _stable_sigmoid_float32(logits: np.ndarray) -> np.ndarray:
    source = np.asarray(logits, dtype=np.float32)
    output = np.empty(source.shape, dtype=np.float32)
    positive = source >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-source[positive]))
    exponent = np.exp(source[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return np.ascontiguousarray(output)


def compose_alpha(sample: FactorizedSample, alpha: float) -> np.ndarray:
    """Compose one registered residual attenuation in FP32."""

    number = float(alpha)
    if number not in ALPHAS:
        raise ValueError(f"alpha is not in the frozen grid {ALPHAS}")
    return np.ascontiguousarray(
        sample.base + np.float32(number) * sample.residual, dtype=np.float32
    )


def to_raw_sample(sample: FactorizedSample, logits: np.ndarray) -> RawLogitSample:
    array = np.asarray(logits)
    if array.dtype != np.float32 or array.shape != sample.mask.shape:
        raise ValueError("composed logit shape/dtype mismatch")
    return RawLogitSample(
        image_id=sample.image_id,
        logits=np.ascontiguousarray(array),
        probability=_stable_sigmoid_float32(array),
        mask=np.ascontiguousarray(sample.mask),
    )


def fit_empirical_background_tail(
    samples: Sequence[FactorizedSample],
    score_getter: Callable[[FactorizedSample], np.ndarray],
) -> EmpiricalTailTransform:
    """Fit an exact empirical background-tail coordinate on held-in data."""

    if not samples:
        raise ValueError("tail fit requires at least one held-in sample")
    parts: list[np.ndarray] = []
    for sample in samples:
        values = np.asarray(score_getter(sample))
        if values.dtype != np.float32 or values.shape != sample.mask.shape:
            raise ValueError("tail-fit score shape/dtype mismatch")
        background = values[~sample.mask]
        if background.size:
            parts.append(background)
    if not parts:
        raise ValueError("tail fit requires at least one background pixel")
    background = np.concatenate(parts).astype(np.float32, copy=False)
    if not np.isfinite(background).all():
        raise ValueError("tail-fit background contains non-finite values")
    background.sort(kind="stable")
    unique_count = 1 + int(
        np.count_nonzero(background[1:] != background[:-1])
    )
    return EmpiricalTailTransform(
        sorted_background=np.ascontiguousarray(background),
        num_background_pixels=int(background.size),
        num_unique_background_scores=unique_count,
    )


def audit_order_preservation(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    """Audit the FP32 realization of a mathematically monotone transform."""

    source = np.asarray(before).reshape(-1)
    mapped = np.asarray(after).reshape(-1)
    if source.shape != mapped.shape or not np.isfinite(source).all() or not np.isfinite(mapped).all():
        raise ValueError("order audit requires same-shape finite arrays")
    order = np.argsort(source, kind="stable")
    sorted_source = source[order]
    sorted_mapped = mapped[order]
    distinct = np.diff(sorted_source) > 0
    mapped_differences = np.diff(sorted_mapped)
    inversions = int(np.count_nonzero(distinct & (mapped_differences < 0)))
    introduced_ties = int(np.count_nonzero(distinct & (mapped_differences == 0)))
    return {
        "registered_transform_monotonicity": "nondecreasing_in_raw_logit",
        "fp32_order_inversions": inversions,
        "fp32_introduced_adjacent_ties": introduced_ties,
        "fp32_order_preserved": inversions == 0,
        "single_domain_scope_note": STRICT_MONOTONE_SCOPE_NOTE,
    }


def select_alpha_from_held_in_scores(
    scores_by_alpha: Mapping[float, Mapping[str, float]],
) -> dict[str, Any]:
    """Select one alpha with a frozen, conservative held-in-only rule.

    Feasibility is relative to the alpha=0 base: Pd at every budget may not
    decrease by more than 0.005.  The registered objective maximizes the
    arithmetic mean Pd across strict/medium/loose; candidates tied within
    1e-12 are resolved toward the smaller alpha.  Alpha zero is therefore an
    always-defined fail-closed fallback.
    """

    if set(float(value) for value in scores_by_alpha) != set(ALPHAS):
        raise ValueError(f"alpha evidence must cover exactly {ALPHAS}")
    normalized: dict[float, dict[str, float]] = {}
    for alpha in ALPHAS:
        raw = scores_by_alpha[alpha]
        if set(raw) != {"strict", "medium", "loose"}:
            raise ValueError("alpha evidence must cover exactly three budgets")
        row: dict[str, float] = {}
        for budget in ("strict", "medium", "loose"):
            value = float(raw[budget])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("held-in Pd must be finite and in [0, 1]")
            row[budget] = value
        normalized[alpha] = row
    base = normalized[0.0]
    candidates: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        row = normalized[alpha]
        deltas = {budget: row[budget] - base[budget] for budget in row}
        feasible = all(
            delta >= -ALPHA_PER_BUDGET_MAX_REGRESSION - NUMERIC_ATOL
            for delta in deltas.values()
        )
        objective = sum(row.values()) / 3.0
        candidates.append(
            {
                "alpha": alpha,
                "pd": row,
                "delta_vs_alpha0": deltas,
                "feasible": feasible,
                "objective_mean_pd": objective,
            }
        )
    feasible_rows = [row for row in candidates if row["feasible"]]
    if not feasible_rows:  # alpha=0 must make this unreachable.
        raise AssertionError("alpha=0 fail-closed fallback became infeasible")
    best_mean = max(float(row["objective_mean_pd"]) for row in feasible_rows)
    tied = [
        row
        for row in feasible_rows
        if float(row["objective_mean_pd"]) >= best_mean - NUMERIC_ATOL
    ]
    selected = min(tied, key=lambda row: float(row["alpha"]))
    return {
        "selected_alpha": float(selected["alpha"]),
        "selection_scope": "one_seed_one_fold",
        "selection_data": "preregistered_held_in_ids_only",
        "held_out_metrics_used_for_selection": False,
        "alpha_grid": list(ALPHAS),
        "constraints": {
            "per_budget_pd_delta_vs_alpha0_min": -ALPHA_PER_BUDGET_MAX_REGRESSION,
            "applies_to": ["strict", "medium", "loose"],
        },
        "objective": "maximize_arithmetic_mean_pd_across_strict_medium_loose",
        "numeric_atol": NUMERIC_ATOL,
        "tie_break": "smaller_alpha",
        "candidates": candidates,
    }


def _record_to_factorized_sample(
    path: Path,
    record: Mapping[str, Any],
    *,
    expected_dataset: str,
    expected_subset_role: str,
    expected_role: str,
) -> tuple[FactorizedSample, dict[str, Any]]:
    required = {
        "base_raw_logit_float32",
        "final_raw_logit_float32",
        "residual_raw_logit_float32",
        "mask",
        "image_id",
        "dataset_name",
        "subset_role",
        "split_role",
        "original_hw",
        "input_hw",
        "valid_hw",
        "padding_ltrb",
        "spatial_mode",
        "labels_loaded",
        "inference_autocast_enabled",
        "model_output_bitwise_equal",
    }
    with np.load(path, allow_pickle=False) as payload:
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"factorized record lacks fields: {sorted(missing)}")
        image_id = _scalar_string(payload["image_id"], name="image_id")
        dataset_name = _scalar_string(payload["dataset_name"], name="dataset_name")
        subset_role = _scalar_string(payload["subset_role"], name="subset_role")
        if image_id != str(record.get("image_id")):
            raise ValueError("factorized image_id differs from manifest")
        if dataset_name != expected_dataset or subset_role != expected_subset_role:
            raise ValueError("factorized dataset/subset identity drift")
        if _scalar_string(payload["spatial_mode"], name="spatial_mode") != "native":
            raise ValueError("factorized record is not native resolution")
        if not _scalar_bool(payload["labels_loaded"], name="labels_loaded"):
            raise ValueError("factorized audit requires source labels")
        if _scalar_bool(
            payload["inference_autocast_enabled"],
            name="inference_autocast_enabled",
        ):
            raise ValueError("factorized audit requires autocast disabled")
        if not _scalar_bool(
            payload["model_output_bitwise_equal"],
            name="model_output_bitwise_equal",
        ):
            raise ValueError("factorized record lacks model-output equality")
        base = np.asarray(payload["base_raw_logit_float32"])
        final = np.asarray(payload["final_raw_logit_float32"])
        residual = np.asarray(payload["residual_raw_logit_float32"])
        mask = np.asarray(payload["mask"])
        if any(array.dtype != np.float32 for array in (base, final, residual)):
            raise ValueError("factorized logits must all be float32")
        if base.ndim != 2 or base.size == 0 or final.shape != base.shape or residual.shape != base.shape:
            raise ValueError("factorized logits must be same-shape non-empty 2-D arrays")
        if not all(np.isfinite(array).all() for array in (base, final, residual)):
            raise ValueError("factorized logits contain non-finite values")
        if mask.shape != base.shape or (
            not np.issubdtype(mask.dtype, np.number) and mask.dtype != np.bool_
        ):
            raise ValueError("factorized mask shape/dtype mismatch")
        if not np.isfinite(mask).all() or not np.isin(np.unique(mask), (0, 1, False, True)).all():
            raise ValueError("factorized mask must be finite and binary")
        if _integer_pair(payload["original_hw"], name="original_hw") != base.shape:
            raise ValueError("factorized original_hw differs from native output shape")
        if _integer_pair(payload["valid_hw"], name="valid_hw") != base.shape:
            raise ValueError("factorized valid_hw differs from native output shape")
        recomputed_residual = np.subtract(final, base, dtype=np.float32)
        if not np.array_equal(residual, recomputed_residual):
            raise ValueError("factorized residual is not exact FP32 final-base")
        replay = np.add(base, residual, dtype=np.float32)
        max_error = float(np.max(np.abs(replay.astype(np.float64) - final.astype(np.float64))))
        if not np.allclose(
            replay,
            final,
            rtol=FACTOR_REPLAY_RTOL,
            atol=FACTOR_REPLAY_ATOL,
            equal_nan=False,
        ):
            raise ValueError("factorized base+residual replay exceeds frozen tolerance")
        if expected_role == "control" and not np.all(residual == np.float32(0.0)):
            raise ValueError("control factorized export has non-zero residual")
    return (
        FactorizedSample(
            image_id=image_id,
            dataset_name=dataset_name,
            subset_role=subset_role,
            base=np.ascontiguousarray(base),
            final=np.ascontiguousarray(final),
            residual=np.ascontiguousarray(residual),
            mask=np.ascontiguousarray(mask.astype(bool, copy=False)),
        ),
        {"replay_max_abs_error": max_error},
    )


def load_factorized_directory(
    directory: str | Path,
    *,
    expected_manifest_sha256: str,
    old_run: Mapping[str, Any],
    expected_subset_role: str,
    expected_governance_binding: Mapping[str, Any] | None = None,
    expected_tier2s_preregistration_binding: Mapping[str, Any] | None = None,
) -> tuple[list[FactorizedSample], dict[str, Any]]:
    """Load one new export only after complete manifest/content validation."""

    governance_binding, preregistration_binding = (
        tier2s_exporter._manifest_consumer_bindings(
            expected_governance_binding,
            expected_tier2s_preregistration_binding,
        )
    )
    root = Path(directory).resolve()
    manifest_path = root / "manifest.json"
    _verify_file(manifest_path, expected_manifest_sha256)
    sidecar = root / "manifest.sha256"
    if sidecar.is_symlink() or not sidecar.is_file():
        raise RuntimeError(f"factorized manifest sidecar is absent: {sidecar}")
    expected_sidecar = f"{expected_manifest_sha256}  manifest.json\n"
    if sidecar.read_text(encoding="ascii") != expected_sidecar:
        raise RuntimeError("factorized manifest sidecar drift")
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != EXPORT_SCHEMA:
        raise RuntimeError("factorized export schema drift")
    frozen_false = (
        manifest.get("source_only") is True
        and manifest.get("outer_target_access_authorized") is False
        and manifest.get("outer_target_images_loaded") is False
        and manifest.get("outer_target_masks_loaded") is False
        and manifest.get("outer_target_images_used") is False
        and manifest.get("outer_target_labels_used") is False
        and manifest.get("diagnostic_only") is True
        and manifest.get("authorizes_go") is False
        and manifest.get("authorizes_source_tier3") is False
    )
    if not frozen_false:
        raise RuntimeError("factorized export is not source-only/outer-locked")
    if manifest.get("architecture_version") != "rc-mshnet-v2-component-role-split":
        raise RuntimeError("factorized architecture identity drift")
    protocol_binding = manifest.get("protocol_binding")
    if not isinstance(protocol_binding, Mapping):
        raise RuntimeError("factorized protocol binding is absent")
    if (
        Path(str(protocol_binding.get("path", ""))).resolve() != PROTOCOL_PATH
        or protocol_binding.get("protocol_id") != PROTOCOL_ID
        or protocol_binding.get("sha256") != file_sha256(PROTOCOL_PATH)
    ):
        raise RuntimeError("factorized protocol binding drift")
    if manifest.get("governance_binding") != governance_binding:
        raise RuntimeError("factorized governance binding drift")
    if (
        manifest.get("tier2s_preregistration_binding")
        != preregistration_binding
    ):
        raise RuntimeError("factorized Tier2S preregistration binding drift")
    checkpoint = manifest.get("checkpoint_binding")
    dataset = manifest.get("dataset_binding")
    records = manifest.get("records")
    if not isinstance(checkpoint, Mapping) or not isinstance(dataset, Mapping) or not isinstance(records, list):
        raise RuntimeError("factorized manifest sections are incomplete")
    expected_checkpoint = {
        "checkpoint_path": old_run.get("checkpoint"),
        "checkpoint_sha256": old_run.get("checkpoint_sha256"),
        "formal_config_sha256": old_run.get("formal_config_sha256"),
        "seed": old_run.get("seed"),
        "role": old_run.get("role"),
        "fold": old_run.get("fold"),
        "training_source": old_run.get("training_source"),
        "held_out_source": old_run.get("held_out_source"),
        "checkpoint_selection": "fixed_last",
        "epoch": 79,
        "architecture_version": "rc-mshnet-v2-component-role-split",
    }
    for field, expected in expected_checkpoint.items():
        if checkpoint.get(field) != expected:
            raise RuntimeError(f"factorized checkpoint binding drift: {field}")
    formal_config_path = Path(str(checkpoint.get("formal_config_path", ""))).resolve()
    _verify_file(formal_config_path, str(old_run.get("formal_config_sha256", "")))
    if dataset.get("subset_role") != expected_subset_role or dataset.get("spatial_mode") != "native":
        raise RuntimeError("factorized dataset subset/spatial binding drift")
    expected_dataset = str(
        old_run["training_source"]
        if expected_subset_role == "held_in"
        else old_run["held_out_source"]
    )
    if dataset.get("dataset_name") != expected_dataset:
        raise RuntimeError("factorized dataset name differs from frozen LODO role")
    if manifest.get("role") != old_run.get("role") or manifest.get("fold") != old_run.get("fold"):
        raise RuntimeError("factorized top-level role/fold identity drift")
    if manifest.get("subset_role") != expected_subset_role:
        raise RuntimeError("factorized top-level subset identity drift")
    if manifest.get("all_model_outputs_bitwise_equal") is not True:
        raise RuntimeError("factorized export lacks model-output equality")
    if old_run.get("role") == "control" and manifest.get("all_residual_exact_zero") is not True:
        raise RuntimeError("control manifest does not certify zero residual")
    if not records:
        raise RuntimeError("factorized export contains no records")
    if score_records_sha256(records) != manifest.get("records_sha256"):
        raise RuntimeError("factorized records_sha256 drift")
    image_ids = [str(record.get("image_id", "")) for record in records]
    if len(set(image_ids)) != len(image_ids) or any(not value for value in image_ids):
        raise RuntimeError("factorized record IDs are empty or duplicated")
    if ordered_ids_sha256(image_ids) != manifest.get("ordered_image_ids_sha256"):
        raise RuntimeError("factorized ordered_image_ids_sha256 drift")
    samples: list[FactorizedSample] = []
    replay_max = 0.0
    listed: set[Path] = set()
    for record in records:
        if not isinstance(record, Mapping) or not _valid_sha256(record.get("sha256")):
            raise RuntimeError("factorized record binding is malformed")
        path = _safe_record_path(root, record.get("file"))
        if path in listed:
            raise RuntimeError("factorized manifest lists one file more than once")
        listed.add(path)
        _verify_file(path, str(record["sha256"]))
        sample, audit = _record_to_factorized_sample(
            path,
            record,
            expected_dataset=expected_dataset,
            expected_subset_role=expected_subset_role,
            expected_role=str(old_run["role"]),
        )
        if record.get("shape") != list(sample.base.shape):
            raise RuntimeError("factorized record shape summary drift")
        if not math.isclose(
            float(record.get("replay_max_abs_error", math.inf)),
            float(audit["replay_max_abs_error"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RuntimeError("factorized record replay summary drift")
        exact_zero = bool(np.count_nonzero(sample.residual) == 0)
        if record.get("residual_exact_zero") is not exact_zero:
            raise RuntimeError("factorized record residual-zero summary drift")
        replay_max = max(replay_max, float(audit["replay_max_abs_error"]))
        samples.append(sample)
    unlisted = {path.resolve() for path in (root / "records").glob("*.npz")} - listed
    if unlisted:
        raise RuntimeError("factorized directory contains unlisted NPZ records")
    if not math.isclose(
        float(manifest.get("replay_max_abs_error", math.inf)),
        replay_max,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError("factorized manifest replay summary drift")
    streams = manifest.get("raw_logit_stream_sha256")
    if not isinstance(streams, Mapping):
        raise RuntimeError("factorized stream hashes are absent")
    for factor in ("base", "final", "residual"):
        digest = hashlib.sha256()
        digest.update(b"rc-irstd-tier2s-factorized-stream-sha256-v1\0")
        digest.update(factor.encode("ascii") + b"\0")
        for sample in samples:
            identity = sample.image_id.encode("utf-8")
            value = getattr(sample, factor)
            digest.update(len(identity).to_bytes(8, "big"))
            digest.update(identity)
            digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
            digest.update(value.astype("<f4", copy=False).tobytes(order="C"))
        if streams.get(factor) != digest.hexdigest():
            raise RuntimeError(f"factorized {factor} stream hash drift")
    return samples, {
        "manifest": str(manifest_path),
        "manifest_sha256": expected_manifest_sha256,
        "num_records": len(samples),
        "ordered_image_ids_sha256": ordered_ids_sha256(image_ids),
        "replay_max_abs_error": replay_max,
        "checkpoint_binding_verified": True,
        "source_only_verified": True,
        "governance_binding_verified": True,
        "tier2s_preregistration_binding_verified": True,
    }


def _verify_new_against_old_heldout(
    new_samples: Sequence[FactorizedSample], old_run: Mapping[str, Any]
) -> dict[str, Any]:
    old_samples, _, integrity, _ = load_formal_raw_logit_directory(
        old_run["score_dir"], expected_split_role="train"
    )
    if len(new_samples) != len(old_samples):
        raise RuntimeError("factorized held-out record count differs from frozen export")
    for index, (new, old) in enumerate(zip(new_samples, old_samples)):
        if (
            new.image_id != old.image_id
            or not np.array_equal(new.final, old.logits)
            or not np.array_equal(new.mask, old.mask)
        ):
            raise RuntimeError(
                f"factorized final/mask differs from frozen export at record {index}"
            )
    return {
        "bitwise_final_matches_frozen_raw_logit": True,
        "bitwise_mask_matches_frozen_mask": True,
        "num_records": len(new_samples),
        "frozen_score_manifest_sha256": integrity["manifest_sha256"],
    }


def _subset_by_ids(
    samples: Sequence[FactorizedSample], ids: Sequence[str]
) -> list[FactorizedSample]:
    if not ids or len(set(ids)) != len(ids):
        raise RuntimeError("preregistered held-in IDs are empty or duplicated")
    mapping = {sample.image_id: sample for sample in samples}
    missing = [value for value in ids if value not in mapping]
    if missing:
        raise RuntimeError(f"held-in export lacks preregistered IDs: {missing[:3]}")
    return [mapping[value] for value in ids]


def _held_in_alpha_scores(
    samples: Sequence[FactorizedSample], *, exact_api: ExactAPI | None = None
) -> dict[float, dict[str, float]]:
    """Compute all three held-in budget points from one exact enumeration/alpha.

    The registered selection bucket is deterministically split into two
    disjoint evaluator shards solely because the reusable source enumerator
    requires at least two names.  Only pooled raw counts are used, so this is
    algebraically identical to treating the held-in bucket as one domain.
    """

    if len(samples) < 2:
        raise RuntimeError("alpha-selection bucket needs at least two samples")
    api = exact_api or ExactAPI()
    result: dict[float, dict[str, float]] = {}
    for alpha in ALPHAS:
        raw = [to_raw_sample(sample, compose_alpha(sample, alpha)) for sample in samples]
        shards = {
            "held_in_selection_shard0": raw[::2],
            "held_in_selection_shard1": raw[1::2],
        }
        enumeration = api.enumerate_states(
            shards,
            loose_pixel_budget=1.0e-5,
            **MATCHING_PROTOCOL,
        )
        row: dict[str, float] = {}
        for name, pixel, component in BUDGETS:
            selection = api.select_points(
                enumeration, pixel_budget=pixel, component_budget=component
            )
            pooled = selection.get("source_pooled")
            point = pooled.get("operating_point") if isinstance(pooled, Mapping) else None
            if not isinstance(point, Mapping) or not math.isfinite(float(point.get("pd", math.nan))):
                raise RuntimeError("held-in exact Oracle did not produce finite Pd")
            threshold = point.get("threshold_logit_float32")
            verification = api.evaluate_threshold(
                shards, threshold, **MATCHING_PROTOCOL
            )
            verified = verification.get("pooled")
            for field in ("tp_objects", "gt_objects", "fp_components", "fp_pixels", "total_pixels"):
                if not isinstance(verified, Mapping) or int(verified[field]) != int(point[field]):
                    raise RuntimeError("held-in selected point exact verification failed")
            row[name] = float(point["pd"])
        result[alpha] = row
    return result


def _summarize_enumeration(enumeration: Mapping[str, Any]) -> dict[str, Any]:
    states = enumeration.get("states")
    return {
        "schema_version": enumeration.get("schema_version"),
        "exact_state_enumeration": enumeration.get("exact_state_enumeration"),
        "shared_threshold_across_domains": enumeration.get("shared_threshold_across_domains"),
        "num_states": len(states) if isinstance(states, list) else enumeration.get("num_states"),
        "pruning": enumeration.get("pruning"),
    }


def _compute_route_points(
    samples: Mapping[str, Mapping[int, Mapping[str, Sequence[RawLogitSample]]]],
    *,
    exact_api: ExactAPI,
) -> tuple[dict[str, Any], dict[str, Any]]:
    points: dict[str, dict[int, Any]] = {role: {} for role in ROLES}
    evidence: dict[str, Any] = {}
    for role in ROLES:
        for seed in SEEDS:
            enumeration, selections, output = _compute_exact_points(
                samples[role][seed], api=exact_api
            )
            points[role][seed] = output["points"]
            evidence[f"{role}_seed{seed}"] = {
                "enumeration": _summarize_enumeration(enumeration),
                "selections": selections,
                "verification": output["verification"],
            }
    gate = evaluate_gate_level(
        points,
        candidate="c",
        baseline="control",
        level_name="exploratory_factorized_c_vs_frozen_matched_control",
    )
    return {"points_by_role": points, "nine_criterion_replay": gate}, evidence


def _validate_protocol(
    protocol_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    protocol = _load_json(protocol_path)
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("research_mode") != "exploratory_source_only"
    ):
        raise RuntimeError("factorized audit protocol schema drift")
    limits = protocol.get("scientific_limits")
    if (
        not isinstance(limits, Mapping)
        or limits.get("paper_claim_authorized") is not False
        or limits.get("source_tier3_authorized") is not False
        or limits.get("outer_target_access_authorized") is not False
        or limits.get("outer_target_images_used") is not False
        or limits.get("outer_target_labels_used") is not False
    ):
        raise RuntimeError("factorized audit protocol violates exploratory lock")
    intervention = protocol.get("factorized_intervention")
    if not isinstance(intervention, Mapping):
        raise RuntimeError("factorized intervention protocol is absent")
    if tuple(float(value) for value in intervention.get("alpha_candidates", [])) != ALPHAS:
        raise RuntimeError("factorized alpha grid drift")
    partition = intervention.get("held_in_id_partition")
    selection = intervention.get("alpha_selection")
    if (
        not isinstance(partition, Mapping)
        or partition.get("hash") != "sha256(protocol_id + NUL + dataset_name + NUL + image_id)"
        or partition.get("modulus") != 4
        or partition.get("tail_fit_buckets") != [0, 1]
        or partition.get("alpha_selection_bucket") != 2
        or partition.get("unused_audit_bucket") != 3
        or not isinstance(selection, Mapping)
        or selection.get("held_out_labels_used") is not False
        or float(selection.get("per_budget_pd_noninferiority_vs_alpha0", math.nan)) != -0.005
        or selection.get("objective") != "maximize_arithmetic_mean_pd_across_strict_medium_loose"
        or float(selection.get("numeric_atol", math.nan)) != NUMERIC_ATOL
        or selection.get("tie_break") != "smaller_alpha"
    ):
        raise RuntimeError("factorized held-in partition/alpha rule drift")
    tail = protocol.get("source_tail_coordinate")
    if (
        not isinstance(tail, Mapping)
        or tail.get("fit_pixels") != "background_pixels_from_held_in_tail_fit_buckets_only"
        or tail.get("tail_score") != "negative_log10_empirical_survival"
        or tail.get("target_labels_used_for_fit") is not False
    ):
        raise RuntimeError("factorized source-tail protocol drift")
    if tuple(protocol.get("audit_branches", ())) != ROUTES:
        raise RuntimeError("factorized audit branch names/order drift")
    execution = protocol.get("execution")
    if (
        not isinstance(execution, Mapping)
        or execution.get("physical_gpus") != [2, 3]
        or execution.get("container_logical_ordinals") != {"2": 0, "3": 1}
        or execution.get("container_must_expose_only_physical_gpus") != [2, 3]
        or execution.get("allow_gpu_fallback") is not False
        or execution.get("export_jobs") != 18
    ):
        raise RuntimeError("factorized GPU2/3 execution contract drift")
    parent = protocol.get("immutable_parent_evidence")
    if not isinstance(parent, Mapping):
        raise RuntimeError("immutable parent evidence is absent")
    decision_binding = parent.get("tier2r_decision")
    source_binding = parent.get("source_authorization")
    target_binding = parent.get("outer_target_authorization")
    if not all(isinstance(value, Mapping) for value in (decision_binding, source_binding, target_binding)):
        raise RuntimeError("immutable parent evidence bindings are incomplete")

    def bound_parent(binding: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
        relative = Path(str(binding.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("parent evidence path must be project-relative")
        path = (PROJECT_ROOT / relative).resolve()
        _verify_file(path, str(binding.get("sha256", "")))
        return path, _load_json(path)

    old_decision_path, decision = bound_parent(decision_binding)
    _, source = bound_parent(source_binding)
    _, target = bound_parent(target_binding)
    if (
        decision.get("decision") != "TIER2R_HOLD"
        or decision.get("selected_candidate") is not None
        or source.get("source_tier3_design_authorized") is not False
        or target.get("outer_target_access_authorized") is not False
    ):
        raise RuntimeError("immutable parent HOLD/authorization semantics drift")
    if OLD_HANDOFF_PATH.is_symlink() or not OLD_HANDOFF_PATH.is_file():
        raise RuntimeError("old Tier2R handoff is absent or aliased")
    return protocol, dict(partition), OLD_HANDOFF_PATH, old_decision_path


def held_in_partition_bucket(dataset_name: str, image_id: str) -> int:
    if dataset_name not in {"NUDT-SIRST", "IRSTD-1K"} or not image_id:
        raise ValueError("invalid source dataset/image identity for partition")
    payload = f"{PROTOCOL_ID}\0{dataset_name}\0{image_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big") % 4


def _partition_held_in(
    samples: Sequence[FactorizedSample], *, buckets: set[int]
) -> list[FactorizedSample]:
    if not buckets or not buckets.issubset({0, 1, 2, 3}):
        raise ValueError("invalid registered held-in buckets")
    selected = [
        sample
        for sample in samples
        if held_in_partition_bucket(sample.dataset_name, sample.image_id) in buckets
    ]
    if not selected:
        raise RuntimeError("registered held-in partition is empty")
    return selected


def _validate_old_hold(
    old_handoff_path: Path, old_decision_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    old_handoff = _load_json(old_handoff_path)
    old_decision = _load_json(old_decision_path)
    if old_handoff.get("schema_version") != OLD_HANDOFF_SCHEMA or old_handoff.get("source_only") is not True:
        raise RuntimeError("old Tier2R handoff identity drift")
    if (
        old_decision.get("schema_version") != OLD_DECISION_SCHEMA
        or old_decision.get("decision") != "TIER2R_HOLD"
        or old_decision.get("selected_candidate") is not None
        or old_decision.get("authorizes_source_tier3_design") is not False
        or old_decision.get("authorizes_outer_target_access") is not False
    ):
        raise RuntimeError("old Tier2R HOLD/authorization identity drift")
    runs = old_handoff.get("runs")
    expected = {
        f"seed{seed}_{role}_{fold}"
        for seed in SEEDS
        for role in ROLES
        for fold in FOLDS
    }
    if not isinstance(runs, Mapping) or not expected.issubset(runs):
        raise RuntimeError("old Tier2R handoff lacks matched control/C runs")
    return old_handoff, old_decision


def _load_export_handoff(
    handoff_path: Path,
    old_runs: Mapping[str, Any],
    *,
    expected_governance_binding: Mapping[str, Any],
    expected_governance_registration_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, list[FactorizedSample]]], dict[str, Any]]:
    if not _valid_sha256(expected_governance_registration_sha256):
        raise RuntimeError("expected governance registration SHA-256 is invalid")
    _verify_frozen_sidecar(handoff_path)
    handoff = _load_json(handoff_path)
    if (
        handoff.get("schema_version") != HANDOFF_SCHEMA
        or handoff.get("source_only") is not True
        or handoff.get("research_mode") != "exploratory_source_only"
        or handoff.get("outer_target_access_authorized") is not False
        or handoff.get("outer_target_images_used") is not False
        or handoff.get("outer_target_labels_used") is not False
        or handoff.get("source_tier3_authorized") is not False
        or handoff.get("paper_claim_authorized") is not False
    ):
        raise RuntimeError("factorized exporter handoff violates source-only lock")
    _require_chain_binding(
        handoff,
        "governance_binding",
        expected_governance_binding,
        where="export_handoff",
    )
    preregistration_path = handoff_path.parent / "PREREGISTRATION.json"
    if preregistration_path != tier2s_exporter.DEFAULT_PREREGISTRATION:
        raise RuntimeError("factorized handoff is outside the canonical Tier2S audit root")
    governance_binding, preregistration_binding = (
        tier2s_exporter.require_frozen_tier2s_consumer_bindings(
            expected_governance_registration_sha256=(
                expected_governance_registration_sha256
            ),
            tier2s_preregistration_path=preregistration_path,
            expected_tier2s_preregistration_sha256=str(
                handoff.get("preregistration_sha256", "")
            ),
            project_root=PROJECT_ROOT,
        )
    )
    if governance_binding != dict(expected_governance_binding):
        raise RuntimeError("governance binding changed before handoff consumption")
    _require_chain_binding(
        handoff,
        "tier2s_preregistration_binding",
        preregistration_binding,
        where="export_handoff",
    )

    queue_path = handoff_path.parent / "QUEUE_MANIFEST.json"
    _verify_frozen_sidecar(
        queue_path,
        expected_sha256=str(handoff.get("queue_manifest_sha256", "")),
    )
    preregistration = _load_json(preregistration_path)
    queue = _load_json(queue_path)
    _require_chain_binding(
        preregistration,
        "governance_binding",
        governance_binding,
        where="preregistration",
    )
    _require_chain_binding(
        queue, "governance_binding", governance_binding, where="queue_manifest"
    )
    _require_chain_binding(
        queue,
        "tier2s_preregistration_binding",
        preregistration_binding,
        where="queue_manifest",
    )
    parent_binding = preregistration.get("parent_handoff")
    protocol_binding = preregistration.get("protocol")
    queue_jobs = _validate_fixed_two_lane_queue(queue)
    if (
        not isinstance(parent_binding, Mapping)
        or Path(str(parent_binding.get("path", ""))).resolve() != OLD_HANDOFF_PATH
        or parent_binding.get("sha256") != file_sha256(OLD_HANDOFF_PATH)
        or not isinstance(protocol_binding, Mapping)
        or Path(str(protocol_binding.get("path", ""))).resolve() != PROTOCOL_PATH
        or protocol_binding.get("sha256") != file_sha256(PROTOCOL_PATH)
    ):
        raise RuntimeError("factorized preregistration/queue/parent chain drift")
    scheduler_log = Path(str(handoff.get("scheduler_event_log", ""))).resolve()
    expected_scheduler_log = handoff_path.parent / "scheduler_events.jsonl"
    if scheduler_log != expected_scheduler_log:
        raise RuntimeError("Tier2S scheduler event log path drift")
    scheduler_binding = _verify_scheduler_event_log(
        scheduler_log,
        expected_sha256=str(handoff.get("scheduler_event_log_sha256", "")),
        expected_jobs=queue_jobs,
    )
    if (
        Path(str(handoff.get("scheduler_event_log_sidecar", ""))).resolve()
        != Path(scheduler_binding["sidecar_path"])
        or handoff.get("scheduler_event_log_sidecar_sha256")
        != scheduler_binding["sidecar_sha256"]
    ):
        raise RuntimeError("Tier2S scheduler event sidecar binding drift")
    exports = handoff.get("runs")
    expected_runs = {
        f"seed{seed}_{role}_{fold}"
        for seed in SEEDS
        for role in ROLES
        for fold in FOLDS
    }
    if not isinstance(exports, Mapping) or set(exports) != expected_runs:
        raise RuntimeError("factorized exporter handoff must bind exactly 12 runs")
    loaded: dict[str, dict[str, list[FactorizedSample]]] = {}
    audits: dict[str, Any] = {
        "handoff_chain": {
            "preregistration_sha256": file_sha256(preregistration_path),
            "queue_manifest_sha256": file_sha256(queue_path),
            "parent_handoff_sha256": file_sha256(OLD_HANDOFF_PATH),
            "protocol_sha256": file_sha256(PROTOCOL_PATH),
            "scheduler_event_log": scheduler_binding,
            "governance_registration_sha256": (
                expected_governance_registration_sha256
            ),
            "governance_binding": governance_binding,
            "tier2s_preregistration_binding": preregistration_binding,
            "queue_contract": {
                "schema_version": QUEUE_SCHEMA,
                "scheduler": QUEUE_SCHEDULER,
                "num_jobs": len(queue_jobs),
                "physical_gpus": list(QUEUE_PHYSICAL_GPUS),
                "physical_to_container_ordinal": {
                    str(gpu): CONTAINER_ORDINAL_BY_PHYSICAL[gpu]
                    for gpu in QUEUE_PHYSICAL_GPUS
                },
                "jobs_per_lane": QUEUE_JOBS_PER_LANE,
            },
            "verified": True,
        }
    }
    for run_id in sorted(expected_runs):
        binding = exports[run_id]
        if not isinstance(binding, Mapping):
            raise RuntimeError(f"factorized handoff run binding drift: {run_id}")
        old_run = old_runs[run_id]
        for field, expected in (
            ("checkpoint_run_id", run_id),
            ("seed", old_run.get("seed")),
            ("role", old_run.get("role")),
            ("fold", old_run.get("fold")),
            ("checkpoint", old_run.get("checkpoint")),
            ("checkpoint_sha256", old_run.get("checkpoint_sha256")),
            ("parent_heldout_score_manifest", old_run.get("score_manifest")),
            ("parent_heldout_score_manifest_sha256", old_run.get("score_manifest_sha256")),
        ):
            if binding.get(field) != expected:
                raise RuntimeError(f"factorized handoff parent binding drift: {run_id}.{field}")
        loaded[run_id] = {}
        audits[run_id] = {}
        scopes = ("held_in", "held_out") if old_run.get("role") == "c" else ("held_in",)
        for subset_role in scopes:
            manifest_key = f"{subset_role}_factorized_manifest"
            digest_key = f"{manifest_key}_sha256"
            manifest_path = Path(str(binding.get(manifest_key, ""))).resolve()
            samples, audit = load_factorized_directory(
                manifest_path.parent,
                expected_manifest_sha256=str(binding.get(digest_key, "")),
                old_run=old_run,
                expected_subset_role=subset_role,
                expected_governance_binding=governance_binding,
                expected_tier2s_preregistration_binding=preregistration_binding,
            )
            loaded[run_id][subset_role] = samples
            audits[run_id][subset_role] = audit
        if old_run.get("role") == "c":
            audits[run_id]["frozen_replay"] = _verify_new_against_old_heldout(
                loaded[run_id]["held_out"], old_run
            )
        else:
            old_samples, _, integrity, _ = load_formal_raw_logit_directory(
                old_run["score_dir"], expected_split_role="train"
            )
            dataset_name = str(old_run["held_out_source"])
            loaded[run_id]["held_out"] = [
                FactorizedSample(
                    image_id=sample.image_id,
                    dataset_name=dataset_name,
                    subset_role="held_out",
                    base=np.ascontiguousarray(sample.logits),
                    final=np.ascontiguousarray(sample.logits),
                    residual=np.zeros_like(sample.logits, dtype=np.float32),
                    mask=np.ascontiguousarray(sample.mask),
                )
                for sample in old_samples
            ]
            audits[run_id]["held_out"] = {
                "source": "immutable_parent_raw_logit_export",
                "manifest": old_run["score_manifest"],
                "manifest_sha256": integrity["manifest_sha256"],
                "num_records": len(old_samples),
                "control_residual_exact_zero_by_construction": True,
            }
    return handoff, loaded, audits


def verify_raw_final_parent_replay(
    raw_result: Mapping[str, Any],
    *,
    old_decision: Mapping[str, Any],
    old_decision_path: Path,
) -> dict[str, Any]:
    """Require exact point and criterion replay of the immutable parent HOLD."""

    evidence_path = old_decision_path.parent / "evidence_manifest.json"
    _verify_file(evidence_path, str(old_decision.get("evidence_manifest_sha256", "")))
    evidence = _load_json(evidence_path)
    artifacts = evidence.get("artifacts")
    observed_points = raw_result.get("points_by_role")
    if not isinstance(artifacts, Mapping) or not isinstance(observed_points, Mapping):
        raise RuntimeError("parent/raw exact-point evidence is incomplete")
    point_bindings: dict[str, Any] = {}
    for role in ROLES:
        role_points = observed_points.get(role)
        if not isinstance(role_points, Mapping):
            raise RuntimeError("raw-final role points are absent")
        for seed in SEEDS:
            key = f"operating_points/{role}_seed{seed}"
            binding = artifacts.get(key)
            if not isinstance(binding, Mapping):
                raise RuntimeError(f"parent evidence lacks {key}")
            path = Path(str(binding.get("path", ""))).resolve()
            _verify_file(path, str(binding.get("sha256", "")))
            parent_points = _load_json(path).get("gate_points")
            current_points = role_points.get(seed, role_points.get(str(seed)))
            if _canonical_json_bytes(parent_points) != _canonical_json_bytes(current_points):
                raise RuntimeError(f"raw-final gate points do not reproduce parent: {role}/seed{seed}")
            point_bindings[f"{role}_seed{seed}"] = {
                "path": str(path),
                "sha256": str(binding["sha256"]),
                "exactly_reproduced": True,
            }

    parent_level = old_decision.get("levels", {}).get("contrast_vs_control")
    current_level = raw_result.get("nine_criterion_replay")
    if not isinstance(parent_level, Mapping) or not isinstance(current_level, Mapping):
        raise RuntimeError("parent/raw nine-criterion evidence is incomplete")
    parent_criteria = parent_level.get("criteria")
    current_criteria = current_level.get("criteria")
    if not isinstance(parent_criteria, list) or not isinstance(current_criteria, list):
        raise RuntimeError("parent/raw criterion rows are absent")

    def normalized(rows: Sequence[Any]) -> dict[str, dict[str, Any]]:
        output_rows: dict[str, dict[str, Any]] = {}
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise RuntimeError("criterion row is malformed")
            row = dict(raw)
            row.pop("criterion_id", None)
            key = f"{row.get('budget')}\0{row.get('metric')}"
            output_rows[key] = row
        return output_rows

    if _canonical_json_bytes(normalized(parent_criteria)) != _canonical_json_bytes(
        normalized(current_criteria)
    ):
        raise RuntimeError("raw-final nine criteria do not reproduce parent paired evidence")
    if current_level.get("passed") is not False or parent_level.get("passed") is not False:
        raise RuntimeError("raw-final replay did not preserve the parent contrast HOLD")
    return {
        "parent_hold_reproduced": True,
        "gate_points_exactly_reproduced": True,
        "paired_deltas_and_pass_flags_exactly_reproduced": True,
        "num_criteria": len(current_criteria),
        "point_bindings": point_bindings,
        "evidence_manifest_sha256": file_sha256(evidence_path),
    }


def classify_factor_diagnosis(route_results: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the preregistered non-selective exploratory classification."""

    factor_routes = {
        "source_selected_alpha": "residual_amplitude_dominant",
        "source_tail_calibrated_final": "cross_fold_scale_dominant",
        "source_tail_calibrated_selected_alpha": "joint_residual_and_scale",
    }
    passed: list[str] = []
    for route in factor_routes:
        result = route_results.get(route)
        gate = result.get("nine_criterion_replay") if isinstance(result, Mapping) else None
        if not isinstance(gate, Mapping) or not isinstance(gate.get("passed"), bool):
            raise RuntimeError(f"factor route lacks valid nine-criterion result: {route}")
        if gate["passed"]:
            passed.append(route)
    if not passed:
        classification = "contrast_route_unsupported"
    elif len(passed) > 1:
        classification = "report_all_supported_factors_without_posthoc_selection"
    else:
        classification = factor_routes[passed[0]]
    return {
        "classification": classification,
        "passing_factor_routes": passed,
        "num_passing_factor_routes": len(passed),
        "posthoc_route_selection_performed": False,
        "exploratory_only": True,
        "authorizes_source_tier3_design": False,
        "authorizes_outer_target_access": False,
        "outer_target_access_authorized": False,
    }


def run_audit(
    *,
    protocol_path: str | Path,
    handoff_path: str | Path,
    output_dir: str | Path,
    governance_registration_sha256: str,
    exact_api: ExactAPI | None = None,
) -> dict[str, Any]:
    if not _valid_sha256(governance_registration_sha256):
        raise RuntimeError("governance registration SHA-256 is invalid")
    governance_binding = governance_registrar.require_frozen_tier2s_governance(
        expected_registration_sha256=governance_registration_sha256
    )
    protocol_file = Path(protocol_path).resolve()
    export_handoff_file = Path(handoff_path).resolve()
    output = Path(output_dir).resolve()
    protocol, partition, old_handoff_path, old_decision_path = _validate_protocol(protocol_file)
    old_handoff_sha_before = file_sha256(old_handoff_path)
    old_decision_sha_before = file_sha256(old_decision_path)
    old_handoff, old_decision = _validate_old_hold(old_handoff_path, old_decision_path)
    export_handoff, loaded, input_audit = _load_export_handoff(
        export_handoff_file,
        old_handoff["runs"],
        expected_governance_binding=governance_binding,
        expected_governance_registration_sha256=(
            governance_registration_sha256
        ),
    )

    selected: dict[str, Any] = {}
    transforms: dict[str, dict[str, EmpiricalTailTransform]] = {}
    transform_evidence: dict[str, Any] = {}
    for seed in SEEDS:
        for fold in FOLDS:
            c_run = f"seed{seed}_c_{fold}"
            control_run = f"seed{seed}_control_{fold}"
            held_in_dataset = str(old_handoff["runs"][c_run]["training_source"])
            c_selection = _partition_held_in(
                loaded[c_run]["held_in"],
                buckets={int(partition["alpha_selection_bucket"])},
            )
            c_tail_fit = _partition_held_in(
                loaded[c_run]["held_in"],
                buckets={int(value) for value in partition["tail_fit_buckets"]},
            )
            control_tail_fit = _partition_held_in(
                loaded[control_run]["held_in"],
                buckets={int(value) for value in partition["tail_fit_buckets"]},
            )
            alpha_evidence = select_alpha_from_held_in_scores(
                _held_in_alpha_scores(c_selection)
            )
            alpha = float(alpha_evidence["selected_alpha"])
            selected[c_run] = alpha_evidence
            transforms[c_run] = {
                "final": fit_empirical_background_tail(c_tail_fit, lambda sample: sample.final),
                "alpha": fit_empirical_background_tail(c_tail_fit, lambda sample, value=alpha: compose_alpha(sample, value)),
            }
            control_transform = fit_empirical_background_tail(
                control_tail_fit, lambda sample: sample.final
            )
            # Control has an exactly zero residual by frozen contract, so the
            # selected-alpha and final coordinates are identical.
            transforms[control_run] = {
                "final": control_transform,
                "alpha": control_transform,
            }
            transform_evidence[c_run] = {
                "selected_alpha": alpha,
                "held_in_dataset": held_in_dataset,
                "alpha_selection_bucket": int(partition["alpha_selection_bucket"]),
                "alpha_selection_ids_sha256": ordered_ids_sha256(
                    [sample.image_id for sample in c_selection]
                ),
                "tail_fit_buckets": list(partition["tail_fit_buckets"]),
                "candidate_tail_fit_ids_sha256": ordered_ids_sha256(
                    [sample.image_id for sample in c_tail_fit]
                ),
                "control_tail_fit_ids_sha256": ordered_ids_sha256(
                    [sample.image_id for sample in control_tail_fit]
                ),
                "candidate": {name: transform.summary() for name, transform in transforms[c_run].items()},
                "control": {name: transform.summary() for name, transform in transforms[control_run].items()},
            }

    route_results: dict[str, Any] = {}
    exact_evidence: dict[str, Any] = {}
    fp32_order_audit: dict[str, Any] = {}
    api = exact_api or ExactAPI()
    for route in ROUTES:
        route_samples: dict[
            str, dict[int, dict[str, list[RawLogitSample]]]
        ] = {role: {seed: {} for seed in SEEDS} for role in ROLES}
        for seed in SEEDS:
            for fold in FOLDS:
                c_run = f"seed{seed}_c_{fold}"
                alpha = float(selected[c_run]["selected_alpha"])
                domain = "nudt" if fold == "heldout_nudt" else "irstd1k"
                for role in ROLES:
                    run_id = f"seed{seed}_{role}_{fold}"
                    converted: list[RawLogitSample] = []
                    for sample in loaded[run_id]["held_out"]:
                        alpha_logits = compose_alpha(sample, alpha)
                        if route == "raw_final":
                            logits = sample.final
                        elif route == "source_selected_alpha":
                            logits = alpha_logits
                        elif route == "source_tail_calibrated_final":
                            logits = transforms[run_id]["final"].apply(sample.final)
                        elif route == "source_tail_calibrated_selected_alpha":
                            logits = transforms[run_id]["alpha"].apply(alpha_logits)
                        else:  # guarded by the frozen ROUTES tuple
                            raise AssertionError(f"unknown audit route: {route}")
                        converted.append(to_raw_sample(sample, logits))
                    route_samples[role][seed][domain] = converted
                    if route.startswith("source_tail_calibrated"):
                        fp32_order_audit[f"{route}:{run_id}"] = {
                            "num_images": len(converted),
                            "fp32_order_inversions": 0,
                            "fp32_introduced_adjacent_ties": "not_enumerated",
                            "fp32_order_preserved": True,
                            "proof": (
                                "searchsorted count_ge is nonincreasing in raw score; "
                                "-log10 survival and FP32 rounding are nondecreasing"
                            ),
                            "single_domain_scope_note": STRICT_MONOTONE_SCOPE_NOTE,
                        }
        route_results[route], exact_evidence[route] = _compute_route_points(
            route_samples, exact_api=api
        )

    parent_replay = verify_raw_final_parent_replay(
        route_results["raw_final"],
        old_decision=old_decision,
        old_decision_path=old_decision_path,
    )
    factor_diagnosis = classify_factor_diagnosis(route_results)

    if file_sha256(old_handoff_path) != old_handoff_sha_before or file_sha256(old_decision_path) != old_decision_sha_before:
        raise RuntimeError("old Tier2R HOLD chain changed during exploratory evaluation")
    export_handoff_sidecar = export_handoff_file.with_suffix(
        export_handoff_file.suffix + ".sha256"
    )
    bindings = {
        "protocol": {
            "path": str(protocol_file),
            "sha256": file_sha256(protocol_file),
        },
        "export_handoff": {
            "path": str(export_handoff_file),
            "sha256": file_sha256(export_handoff_file),
            "sidecar_path": str(export_handoff_sidecar),
            "sidecar_sha256": file_sha256(export_handoff_sidecar),
        },
        "governance_binding": governance_binding,
        "tier2s_preregistration_binding": input_audit["handoff_chain"][
            "tier2s_preregistration_binding"
        ],
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "source_only": True,
        "exploratory_only": True,
        "formal_gate": False,
        "old_tier2r_hold_immutable": True,
        "old_tier2r_handoff_sha256": old_handoff_sha_before,
        "old_tier2r_decision_sha256": old_decision_sha_before,
        "raw_final_reproduces_parent_hold": True,
        "parent_replay": parent_replay,
        "alpha_grid": list(ALPHAS),
        "alpha_selection": selected,
        "calibration": {
            "fit_data": "background_pixels_from_held_in_tail_fit_buckets_only",
            "held_out_labels_or_metrics_used": False,
            "single_domain_scope_note": STRICT_MONOTONE_SCOPE_NOTE,
            "transforms": transform_evidence,
            "fp32_order_audit": fp32_order_audit,
        },
        "routes": route_results,
        "factor_diagnosis": factor_diagnosis,
        "input_audit": input_audit,
        "bindings": bindings,
        "decision": "EXPLORATORY_FACTOR_DISSECTION_ONLY",
        "selected_candidate": None,
        "authorizes_source_tier3_design": False,
        "source_tier3_authorized": False,
        "paper_claim_authorized": False,
        "authorizes_outer_target_access": False,
        "outer_target_access_authorized": False,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "cannot_supersede_old_tier2r_hold": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    exact_evidence_path = output / "exact_evidence.json"
    exact_evidence_payload = {
        "schema_version": "rc-irstd-aaai27-tier2s-exact-evidence-v1",
        "bindings": bindings,
        "routes": exact_evidence,
    }
    exact_evidence_sha = _write_once_frozen_json(
        exact_evidence_path, exact_evidence_payload
    )
    exact_evidence_sidecar = exact_evidence_path.with_suffix(
        exact_evidence_path.suffix + ".sha256"
    )
    result["exact_evidence"] = {
        "path": str(exact_evidence_path),
        "sha256": exact_evidence_sha,
        "sidecar_path": str(exact_evidence_sidecar),
        "sidecar_sha256": file_sha256(exact_evidence_sidecar),
    }
    _write_once_frozen_json(output / "factorized_audit_result.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--handoff", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--governance-registration-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_audit(
            protocol_path=args.protocol,
            handoff_path=args.handoff,
            output_dir=args.output_dir,
            governance_registration_sha256=args.governance_registration_sha256,
        )
    except Exception as error:
        print(f"factorized audit failed closed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(_canonical_json_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
