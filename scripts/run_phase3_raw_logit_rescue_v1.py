#!/usr/bin/env python3
"""Run the immutable source-only Phase-3 raw-logit rescue gate.

The runner is intentionally pinned to the four frozen inner-LODO score-map
artifacts produced by Phase 3 (control/full x held-out NUDT/IRSTD).  It has no
CLI option for selecting another dataset or score directory, performs no
training, and never grants access to NUAA images or labels.

All claim-bearing JSON is written canonically, atomically, once, with a
read-only SHA-256 sidecar.  The historical probability preregistration and
Tier-1 HOLD decision are read and hash-bound but never rewritten or chmod'ed.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import inspect
import json
import math
import os
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path("/home/ly/RC-IRSTD-v2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.artifact_integrity import file_sha256  # noqa: E402
from evaluation.component_matching import (  # noqa: E402
    connected_components,
    match_components,
)
from evaluation.raw_logit_oracle import (  # noqa: E402
    load_formal_raw_logit_directory,
    raw_logit_stream_sha256,
)
from evaluation.raw_logit_rescue_gate import (  # noqa: E402
    DEFAULT_NUMERIC_ATOL,
    RESCUE_GO_TIER2,
    evaluate_rescue_decision,
)
from evaluation.threshold_sweep import domain_key  # noqa: E402


RUNNER_SCHEMA = "rc-irstd-aaai27-phase3-raw-logit-rescue-runner-v1"
AMENDMENT_SCHEMA = "rc-irstd-aaai27-phase3-raw-logit-rescue-amendment-v1"
INPUT_MANIFEST_SCHEMA = "rc-irstd-aaai27-phase3-raw-logit-rescue-input-v1"
INPUT_HASHES_SCHEMA = "rc-irstd-aaai27-phase3-raw-logit-rescue-input-hashes-v1"
EVIDENCE_MANIFEST_SCHEMA = "rc-irstd-aaai27-phase3-raw-logit-rescue-evidence-v1"
AUTHORIZATION_SCHEMA = "rc-irstd-aaai27-phase3-raw-logit-rescue-authorization-v1"

BUDGETS: tuple[tuple[str, float, float], ...] = (
    ("strict", 1.0e-6, 1.0),
    ("medium", 5.0e-6, 5.0),
    ("loose", 1.0e-5, 10.0),
)
MATCHING_PROTOCOL = {
    "matching_rule": "overlap",
    "centroid_distance": 3.0,
    "connectivity": 2,
    "min_component_area": 1,
}
STRICT_MACRO_PD_GAIN_MINIMUM = 0.01


@dataclass(frozen=True)
class ScoreSpec:
    role: str
    fold: str
    target_dataset: str
    source_dataset: str

    @property
    def run_id(self) -> str:
        return f"{self.role}_{self.fold}"

    def score_dir(self, project_root: Path) -> Path:
        return (
            project_root
            / "outputs/aaai27/detectors/source_lodo_gate/seed42"
            / self.role
            / self.fold
            / "scores_heldout_train"
        )


SCORE_SPECS: tuple[ScoreSpec, ...] = (
    ScoreSpec("control", "heldout_nudt", "NUDT-SIRST", "IRSTD-1K"),
    ScoreSpec("control", "heldout_irstd", "IRSTD-1K", "NUDT-SIRST"),
    ScoreSpec("full", "heldout_nudt", "NUDT-SIRST", "IRSTD-1K"),
    ScoreSpec("full", "heldout_irstd", "IRSTD-1K", "NUDT-SIRST"),
)


@dataclass(frozen=True)
class Layout:
    project_root: Path
    output_root: Path

    @property
    def historical_root(self) -> Path:
        return self.project_root / "artifacts/aaai27/audit/phase3_source_lodo_gate"

    @property
    def preregistration(self) -> Path:
        return self.historical_root / "PHASE3_SOURCE_LODO_PREREGISTRATION.json"

    @property
    def tier1_decision(self) -> Path:
        return self.historical_root / "tier1_decision.json"

    @property
    def rescue_plan(self) -> Path:
        return self.project_root / "RC-MSHNet_raw-logit_rescue_gate.md"

    @property
    def target_label_hold(self) -> Path:
        return self.project_root / "outputs/phase_state/HOLD_PHASE3_TARGET_LABEL_ACCESS"


@dataclass(frozen=True)
class AlgorithmAPI:
    enumerate_exact_shared_states: Callable[..., Any]
    select_exact_shared_source_operating_points: Callable[..., Any]
    evaluate_domains_at_threshold: Callable[..., Any]
    select_domain_oracles: Callable[..., Any]
    build_cross_domain_calibration_gap: Callable[..., Any]
    deterministic_dense_state_indices: Callable[..., Any]
    select_dense_operating_points: Callable[..., Any]
    build_realized_fa_sensitivity: Callable[..., Any]
    summarize_false_alarm_concentration: Callable[..., Any]


@dataclass(frozen=True)
class OriginalSnapshot:
    path: Path
    content: bytes
    mode: int
    mtime_ns: int


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Formal rescue JSON cannot contain NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported formal JSON value: {type(value).__name__}")


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=path.name + ".tmp.", delete=False
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _sha_sidecar(path: Path) -> Path:
    if path.suffix != ".json":
        raise ValueError(f"Claim-bearing artifact is not JSON: {path}")
    return path.with_suffix(".sha256")


def _write_once_json(path: Path, payload: Any) -> Path:
    """Atomically freeze canonical JSON and its digest, or fail on drift."""

    content = _canonical_json_bytes(payload)
    digest = hashlib.sha256(content).hexdigest()
    sidecar = _sha_sidecar(path)
    digest_content = f"{digest}  {path.name}\n".encode("ascii")
    if path.exists() or sidecar.exists():
        if not path.is_file() or not sidecar.is_file():
            raise RuntimeError(f"Incomplete immutable artifact pair: {path}")
        if path.read_bytes() != content:
            raise RuntimeError(f"Immutable JSON content drift: {path}")
        if sidecar.read_bytes() != digest_content:
            raise RuntimeError(f"Immutable JSON digest drift: {sidecar}")
        for frozen in (path, sidecar):
            if frozen.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                raise RuntimeError(f"Immutable artifact became writable: {frozen}")
        return path
    _atomic_write_bytes(path, content)
    try:
        _atomic_write_bytes(sidecar, digest_content)
        path.chmod(0o444)
        sidecar.chmod(0o444)
    except BaseException:
        # Never silently treat a half-frozen pair as complete on the next run.
        raise
    return path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _snapshot(path: Path) -> OriginalSnapshot:
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata = path.stat()
    return OriginalSnapshot(
        path=path,
        content=path.read_bytes(),
        mode=stat.S_IMODE(metadata.st_mode),
        mtime_ns=metadata.st_mtime_ns,
    )


def _verify_snapshot(snapshot: OriginalSnapshot) -> None:
    metadata = snapshot.path.stat()
    if (
        snapshot.path.read_bytes() != snapshot.content
        or stat.S_IMODE(metadata.st_mode) != snapshot.mode
        or metadata.st_mtime_ns != snapshot.mtime_ns
    ):
        raise RuntimeError(f"Historical immutable artifact changed: {snapshot.path}")


def _verify_digest_sidecar(path: Path) -> None:
    sidecar = _sha_sidecar(path)
    expected = f"{file_sha256(path)}  {path.name}\n"
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii") != expected:
        raise RuntimeError(f"Historical digest sidecar mismatch: {path}")


def _parse_registered_at(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("registered_at must be a non-empty timezone-aware ISO timestamp")
    raw = value.strip()
    parsed = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
    if parsed.tzinfo is None:
        raise ValueError("registered_at must include a timezone")
    return raw


def _resolve_registered_at(output_root: Path, requested: str | None) -> str:
    amendment = output_root / "protocol_amendment.json"
    if requested is not None:
        return _parse_registered_at(requested)
    if amendment.is_file():
        existing = _load_json(amendment).get("registered_at")
        return _parse_registered_at(str(existing))
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _contains_nuaa(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_nuaa(key) or _contains_nuaa(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_nuaa(item) for item in value)
    if isinstance(value, (str, Path)):
        text = str(value).casefold()
        return "nuaa" in "".join(character for character in text if character.isalnum())
    return False


def _required_callable(module: Any, names: Sequence[str]) -> Callable[..., Any]:
    for name in names:
        value = getattr(module, name, None)
        if callable(value):
            return value
    raise ImportError(
        f"{module.__name__} lacks required callable; accepted names={list(names)}"
    )


def _load_algorithm_api() -> AlgorithmAPI:
    source = importlib.import_module("evaluation.raw_logit_source_operating_point")
    diagnostics = importlib.import_module("evaluation.raw_logit_rescue_diagnostics")
    return AlgorithmAPI(
        enumerate_exact_shared_states=_required_callable(
            source, ("enumerate_exact_shared_states",)
        ),
        select_exact_shared_source_operating_points=_required_callable(
            source, ("select_exact_shared_source_operating_points",)
        ),
        evaluate_domains_at_threshold=_required_callable(
            source, ("evaluate_domains_at_threshold",)
        ),
        select_domain_oracles=_required_callable(source, ("select_domain_oracles",)),
        build_cross_domain_calibration_gap=_required_callable(
            source, ("build_cross_domain_calibration_gap",)
        ),
        deterministic_dense_state_indices=_required_callable(
            diagnostics, ("deterministic_dense_state_indices",)
        ),
        select_dense_operating_points=_required_callable(
            diagnostics, ("select_dense_operating_points",)
        ),
        build_realized_fa_sensitivity=_required_callable(
            diagnostics, ("build_realized_fa_sensitivity",)
        ),
        summarize_false_alarm_concentration=_required_callable(
            diagnostics, ("summarize_false_alarm_concentration",)
        ),
    )


def _callable_source_bindings(api: AlgorithmAPI, layout: Layout) -> dict[str, str]:
    paths = {
        Path(__file__).resolve(),
        (layout.project_root / "evaluation/raw_logit_oracle.py").resolve(),
        (layout.project_root / "evaluation/raw_logit_rescue_gate.py").resolve(),
    }
    for function in (
        api.enumerate_exact_shared_states,
        api.select_exact_shared_source_operating_points,
        api.evaluate_domains_at_threshold,
        api.select_domain_oracles,
        api.build_cross_domain_calibration_gap,
        api.deterministic_dense_state_indices,
        api.select_dense_operating_points,
        api.build_realized_fa_sensitivity,
        api.summarize_false_alarm_concentration,
    ):
        source = inspect.getsourcefile(function)
        if source is not None:
            paths.add(Path(source).resolve())
    bindings: dict[str, str] = {}
    for path in sorted(paths, key=str):
        if not path.is_file():
            raise FileNotFoundError(f"Bound rescue code is missing: {path}")
        try:
            name = str(path.relative_to(layout.project_root.resolve()))
        except ValueError:
            name = str(path)
        bindings[name] = file_sha256(path)
    return bindings


def _historical_snapshots(layout: Layout) -> tuple[OriginalSnapshot, ...]:
    paths = (
        layout.preregistration,
        _sha_sidecar(layout.preregistration),
        layout.tier1_decision,
        _sha_sidecar(layout.tier1_decision),
    )
    snapshots = tuple(_snapshot(path) for path in paths)
    _verify_digest_sidecar(layout.preregistration)
    _verify_digest_sidecar(layout.tier1_decision)
    for snapshot in snapshots:
        if snapshot.mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeError(f"Historical artifact is writable: {snapshot.path}")
    return snapshots


def _validate_historical_contract(layout: Layout) -> tuple[dict[str, Any], dict[str, Any]]:
    prereg = _load_json(layout.preregistration)
    decision = _load_json(layout.tier1_decision)
    expected_budgets = [
        {"name": name, "pixel": pixel, "component": component}
        for name, pixel, component in BUDGETS
    ]
    if prereg.get("budgets") != expected_budgets:
        raise RuntimeError("Historical preregistration FA budgets drifted")
    threshold = prereg.get("threshold_protocol")
    if not isinstance(threshold, Mapping) or any(
        threshold.get(name) != value for name, value in MATCHING_PROTOCOL.items()
    ):
        raise RuntimeError("Historical preregistration matching protocol drifted")
    policy = prereg.get("tier1_policy")
    if not isinstance(policy, Mapping) or (
        float(policy.get("strict_macro_pd_gain_minimum", float("nan")))
        != STRICT_MACRO_PD_GAIN_MINIMUM
        or policy.get("strict_each_source_pd_non_degraded") is not True
        or policy.get("all_budget_pooled_pd_non_degraded") is not True
        or policy.get("all_budget_worst_pd_non_degraded") is not True
    ):
        raise RuntimeError("Historical Tier1 decision criteria drifted")
    if (
        prereg.get("outer_target") != "NUAA-SIRST"
        or prereg.get("outer_target_images_loaded") is not False
        or prereg.get("outer_target_masks_loaded") is not False
    ):
        raise RuntimeError("Historical outer-target isolation contract drifted")
    if (
        decision.get("decision") != "HOLD"
        or decision.get("authorizes_tier2") is not False
        or decision.get("authorizes_outer_target_label_access") is not False
        or decision.get("outer_target_images_used") is not False
        or decision.get("outer_target_labels_used") is not False
    ):
        raise RuntimeError("Raw-logit rescue requires the frozen probability Tier1 HOLD")
    if not layout.target_label_hold.is_file() or layout.target_label_hold.read_text(
        encoding="utf-8"
    ) != "HOLD\n":
        raise RuntimeError("Outer-target label HOLD sentinel is absent or drifted")
    return prereg, decision


def _load_and_validate_inputs(layout: Layout) -> tuple[dict[str, Any], dict[str, Any]]:
    loaded: dict[str, Any] = {"control": {}, "full": {}}
    records: dict[str, Any] = {}
    decision = _load_json(layout.tier1_decision)
    bound_runs = decision.get("input_bindings", {}).get("runs", {})
    if not isinstance(bound_runs, Mapping):
        raise RuntimeError("Historical Tier1 decision lacks frozen run bindings")

    for spec in SCORE_SPECS:
        score_dir = spec.score_dir(layout.project_root).resolve()
        samples, manifest, integrity, contract = load_formal_raw_logit_directory(
            score_dir,
            expected_split_role="train",
        )
        if _contains_nuaa(manifest) or _contains_nuaa(contract):
            raise RuntimeError(f"NUAA reference detected in rescue input: {spec.run_id}")
        if (
            domain_key(str(contract.get("target_dataset")))
            != domain_key(spec.target_dataset)
            or [domain_key(str(value)) for value in contract.get("source_datasets", [])]
            != [domain_key(spec.source_dataset)]
            or contract.get("split_role") != "train"
            or contract.get("requested_split") != "train"
        ):
            raise RuntimeError(f"Inner-LODO closure mismatch: {spec.run_id}")
        if manifest.get("labels_loaded") is not True:
            raise RuntimeError(f"Source pseudo-target labels are absent: {spec.run_id}")

        run_dir = score_dir.parent
        manifest_path = score_dir / "manifest.json"
        checkpoint = run_dir / "last.pt"
        phase3_identity_path = run_dir / "PHASE3_IDENTITY.json"
        export_identity_path = run_dir / "EXPORT_IDENTITY.json"
        for required in (
            manifest_path,
            checkpoint,
            phase3_identity_path,
            export_identity_path,
        ):
            if not required.is_file():
                raise FileNotFoundError(required)
        phase3_identity = _load_json(phase3_identity_path)
        export_identity = _load_json(export_identity_path)
        checkpoint_sha = file_sha256(checkpoint)
        if (
            checkpoint_sha != contract.get("detector_weight_sha256")
            or phase3_identity.get("checkpoint_sha256") != checkpoint_sha
            or export_identity.get("checkpoint_sha256") != checkpoint_sha
        ):
            raise RuntimeError(f"Checkpoint/export binding mismatch: {spec.run_id}")
        historical = bound_runs.get(spec.run_id)
        if not isinstance(historical, Mapping):
            raise RuntimeError(f"Tier1 decision does not bind run: {spec.run_id}")
        expected_historical = {
            "identity_sha256": file_sha256(phase3_identity_path),
            "checkpoint_sha256": checkpoint_sha,
            "export_identity_sha256": file_sha256(export_identity_path),
        }
        for field, expected in expected_historical.items():
            if historical.get(field) != expected:
                raise RuntimeError(
                    f"Historical Tier1 run binding drifted at {spec.run_id}.{field}"
                )

        loaded[spec.role][domain_key(spec.target_dataset)] = samples
        records[spec.run_id] = {
            "role": spec.role,
            "fold": spec.fold,
            "held_out_source_pseudo_target": spec.target_dataset,
            "training_source": spec.source_dataset,
            "score_dir": str(score_dir),
            "score_manifest": str(manifest_path.resolve()),
            "score_manifest_sha256": integrity["manifest_sha256"],
            "score_records_sha256": integrity["records_sha256"],
            "score_ordered_image_ids_sha256": integrity[
                "ordered_image_ids_sha256"
            ],
            "score_num_records": integrity["num_records"],
            "raw_logit_stream_sha256": raw_logit_stream_sha256(samples),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "phase3_identity": str(phase3_identity_path.resolve()),
            "phase3_identity_sha256": expected_historical["identity_sha256"],
            "export_identity": str(export_identity_path.resolve()),
            "export_identity_sha256": expected_historical[
                "export_identity_sha256"
            ],
            "split_file_sha256": contract["split_file_sha256"],
            "split_ordered_ids_sha256": contract["split_ordered_ids_sha256"],
            "split_role": "train",
            "labels_loaded": True,
            "score_representation": contract["score_representation"],
            "probability_dtype": contract["probability_dtype"],
            "logit_dtype": contract["logit_dtype"],
            "probability_transform": contract["probability_transform"],
            "probability_clipping": contract["probability_clipping"],
            "inference_autocast_enabled": contract[
                "inference_autocast_enabled"
            ],
        }

    for role in ("control", "full"):
        if set(loaded[role]) != {"nudt", "irstd1k"}:
            raise RuntimeError(f"Incomplete source LODO coverage for role={role}")
    for fold in ("heldout_nudt", "heldout_irstd"):
        first = records[f"control_{fold}"]
        second = records[f"full_{fold}"]
        for field in (
            "score_ordered_image_ids_sha256",
            "score_num_records",
            "split_file_sha256",
            "split_ordered_ids_sha256",
        ):
            if first[field] != second[field]:
                raise RuntimeError(f"Matched roles differ in {fold}.{field}")
    return loaded, records


def _amendment_payload(
    layout: Layout,
    *,
    registered_at: str,
    code_bindings: Mapping[str, str],
    dense_grid_size: int,
) -> dict[str, Any]:
    return {
        "schema_version": AMENDMENT_SCHEMA,
        "artifact_type": "post_hoc_source_only_raw_logit_protocol_amendment",
        "amendment_id": "phase3_source_raw_logit_rescue_v1",
        "formal_name": "PHASE3_SOURCE_RAW_LOGIT_RESCUE_PROTOCOL_AMENDMENT_V1",
        "registered_at": registered_at,
        "post_hoc_amendment": True,
        "trigger": {
            "historical_probability_gate_decision": "HOLD",
            "defect": "float32_sigmoid_saturation_collapsed_extreme_tail_states",
            "change": "replace_probability_grid_primary_gate_with_exact_raw_logit_states",
        },
        "observed_before_registration": {
            "diagnostic_only": True,
            "sigmoid_probability_one_saturation_observed": True,
            "preliminary_raw_logit_pooled_pd_gain": {
                "strict": 0.01606805293005673,
                "medium": 0.021266540642722213,
                "loose": 0.06238185255198491,
            },
            "preliminary_raw_logit_macro_pd_gain": {
                "strict": 0.017372818168261566,
                "medium": 0.018654002131366373,
                "loose": 0.05432801946599445,
            },
            "preliminary_values_are_non_authorizing_and_require_exact_recheck": True,
            "values_not_authorizing": True,
            "disclosure_source": str(layout.rescue_plan.resolve()),
            "disclosure_source_sha256": file_sha256(layout.rescue_plan),
        },
        "historical_bindings": {
            "probability_preregistration": str(layout.preregistration.resolve()),
            "probability_preregistration_sha256": file_sha256(
                layout.preregistration
            ),
            "probability_tier1_decision": str(layout.tier1_decision.resolve()),
            "probability_tier1_decision_sha256": file_sha256(
                layout.tier1_decision
            ),
            "phase2_status": str(
                (layout.project_root / "artifacts/aaai27/audit/phase2_status.json").resolve()
            ),
            "phase2_status_sha256": file_sha256(
                layout.project_root / "artifacts/aaai27/audit/phase2_status.json"
            ),
        },
        "frozen_invariants": {
            "budgets": [
                {"name": name, "pixel": pixel, "component": component}
                for name, pixel, component in BUDGETS
            ],
            "matching_protocol": MATCHING_PROTOCOL,
            "strict_macro_pd_gain_minimum": STRICT_MACRO_PD_GAIN_MINIMUM,
            "strict_each_source_pd_non_degraded": True,
            "all_budget_pooled_pd_non_degraded": True,
            "all_budget_worst_pd_non_degraded": True,
            "pooled_tie_break": ["maximize_pooled_pd", "lowest_threshold"],
            "worst_tie_break": [
                "maximize_worst_domain_pd",
                "maximize_pooled_pd",
                "lowest_threshold",
            ],
            "prediction_rule": "float32_raw_logit >= shared_threshold_logit",
            "shared_threshold_across_nudt_and_irstd": True,
            "exact_state_enumeration_is_primary": True,
            "dense_grid_is_diagnostic_only": True,
            "dense_grid_size": dense_grid_size,
            "dense_grid_rule": (
                "integer_rank_spaced_subset_of_exact_retained_fp32_breakpoints_"
                "including_reject_and_lowest_tail_state"
            ),
            "new_training_allowed": False,
            "checkpoint_selection_allowed": False,
            "budget_or_success_rule_changes_allowed": False,
        },
        "data_access_policy": {
            "allowed_source_pseudo_targets": ["NUDT-SIRST", "IRSTD-1K"],
            "allowed_split": "train",
            "outer_target": "NUAA-SIRST",
            "outer_target_images_authorized": False,
            "outer_target_labels_authorized": False,
            "outer_target_images_used": False,
            "outer_target_labels_used": False,
        },
        "code_bindings": dict(code_bindings),
    }


def _input_manifest_payload(
    layout: Layout,
    *,
    registered_at: str,
    amendment_path: Path,
    records: Mapping[str, Any],
    probability_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": INPUT_MANIFEST_SCHEMA,
        "registered_at": registered_at,
        "source_only": True,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "protocol_amendment": str(amendment_path.resolve()),
        "protocol_amendment_sha256": file_sha256(amendment_path),
        "historical_probability_preregistration_sha256": file_sha256(
            layout.preregistration
        ),
        "historical_probability_tier1_decision_sha256": file_sha256(
            layout.tier1_decision
        ),
        "expected_split_role": "train",
        "lodo_closure": {
            "heldout_nudt": {"target": "NUDT-SIRST", "source": "IRSTD-1K"},
            "heldout_irstd": {"target": "IRSTD-1K", "source": "NUDT-SIRST"},
        },
        "runs": dict(records),
        "historical_probability_grid_operating_points": dict(
            probability_bindings or {}
        ),
    }


def _input_hashes_payload(
    *,
    registered_at: str,
    amendment_path: Path,
    input_manifest_path: Path,
    records: Mapping[str, Any],
    probability_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": INPUT_HASHES_SCHEMA,
        "registered_at": registered_at,
        "protocol_amendment_sha256": file_sha256(amendment_path),
        "input_manifest_sha256": file_sha256(input_manifest_path),
        "runs": {
            run_id: {
                key: record[key]
                for key in (
                    "score_manifest_sha256",
                    "score_records_sha256",
                    "score_ordered_image_ids_sha256",
                    "raw_logit_stream_sha256",
                    "checkpoint_sha256",
                    "phase3_identity_sha256",
                    "export_identity_sha256",
                    "split_file_sha256",
                    "split_ordered_ids_sha256",
                )
            }
            for run_id, record in records.items()
        },
        "historical_probability_grid_operating_points": dict(
            probability_bindings or {}
        ),
    }


def _load_historical_probability_points(
    layout: Layout,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Reload and hash-check the original 653-grid source operating points."""

    decision = _load_json(layout.tier1_decision)
    bound = decision.get("input_bindings", {}).get("source_operating_points")
    if not isinstance(bound, Mapping):
        raise RuntimeError("Tier1 decision lacks probability operating-point bindings")
    points: dict[str, dict[str, Any]] = {"control": {}, "full": {}}
    bindings: dict[str, Any] = {"control": {}, "full": {}}
    for role in ("control", "full"):
        for name, pixel, component in BUDGETS:
            path = (
                layout.historical_root
                / "source_operating_points"
                / role
                / f"{name}.json"
            ).resolve()
            record = bound.get(role, {}).get(name) if isinstance(bound.get(role), Mapping) else None
            if not isinstance(record, Mapping):
                raise RuntimeError(
                    f"Tier1 decision lacks probability point binding: {role}/{name}"
                )
            if Path(str(record.get("path", ""))).resolve() != path:
                raise RuntimeError(f"Probability point path drifted: {role}/{name}")
            digest = file_sha256(path)
            if record.get("sha256") != digest:
                raise RuntimeError(f"Probability point hash drifted: {role}/{name}")
            payload = _load_json(path)
            if (
                float(payload.get("pixel_budget", float("nan"))) != pixel
                or float(payload.get("component_budget", float("nan"))) != component
                or payload.get("matching_protocol") != MATCHING_PROTOCOL
                or payload.get("formal_protocol_eligible") is not True
            ):
                raise RuntimeError(f"Probability point protocol drifted: {role}/{name}")
            results = payload.get("results")
            if not isinstance(results, Mapping):
                raise RuntimeError(f"Probability point lacks results: {role}/{name}")
            points[role][name] = dict(results)
            bindings[role][name] = {"path": str(path), "sha256": digest}
    return points, bindings


def _extract_gate_point(selection: Mapping[str, Any]) -> dict[str, Any]:
    if all(
        field in selection
        for field in ("found", "pooled_pd", "worst_pd", "macro_pd", "domain_pd")
    ):
        return {
            field: selection[field]
            for field in ("found", "pooled_pd", "worst_pd", "macro_pd", "domain_pd")
        }
    payload = selection.get("results", selection)
    if not isinstance(payload, Mapping):
        return {"found": False}
    pooled = payload.get("source_pooled")
    worst = payload.get("source_worst")
    if (
        not isinstance(pooled, Mapping)
        or not isinstance(worst, Mapping)
        or pooled.get("found") is not True
        or worst.get("found") is not True
    ):
        return {"found": False}
    point = pooled.get("operating_point")
    rows = pooled.get("source_rows")
    if not isinstance(point, Mapping) or not isinstance(rows, Mapping):
        return {"found": False}
    domain_pd = {
        domain_key(str(name)): float(row["pd"])
        for name, row in rows.items()
        if isinstance(row, Mapping) and "pd" in row
    }
    if set(domain_pd) != {"nudt", "irstd1k"}:
        return {"found": False}
    return {
        "found": True,
        "pooled_pd": float(point["pd"]),
        "worst_pd": float(worst["worst_domain_pd"]),
        "macro_pd": sum(domain_pd.values()) / len(domain_pd),
        "domain_pd": domain_pd,
    }


def _count_metrics(row: Mapping[str, int]) -> dict[str, int | float]:
    counts = {
        field: int(row[field])
        for field in (
            "tp_objects",
            "gt_objects",
            "fp_components",
            "fp_pixels",
            "total_pixels",
        )
    }
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


def _per_image_selected_threshold_audit(
    samples_by_domain: Mapping[str, Sequence[Any]],
    threshold: float | None,
) -> dict[str, Any]:
    """Legacy full-image CC audit plus per-image false-alarm anatomy."""

    rows: list[dict[str, Any]] = []
    per_domain: dict[str, dict[str, int | float]] = {}
    for domain in sorted(samples_by_domain):
        totals = {
            "tp_objects": 0,
            "gt_objects": 0,
            "fp_components": 0,
            "fp_pixels": 0,
            "total_pixels": 0,
        }
        for sample in samples_by_domain[domain]:
            prediction = (
                np.zeros(sample.mask.shape, dtype=bool)
                if threshold is None
                else sample.logits >= np.float32(threshold)
            )
            matched = match_components(
                prediction,
                sample.mask,
                rule=MATCHING_PROTOCOL["matching_rule"],
                centroid_distance=MATCHING_PROTOCOL["centroid_distance"],
                connectivity=MATCHING_PROTOCOL["connectivity"],
                min_component_area=MATCHING_PROTOCOL["min_component_area"],
            )
            labels, num_predictions = connected_components(
                prediction,
                connectivity=MATCHING_PROTOCOL["connectivity"],
                min_component_area=MATCHING_PROTOCOL["min_component_area"],
            )
            matched_labels = {int(pair[0]) for pair in matched.matched_pairs}
            false_labels = [
                label
                for label in range(1, int(num_predictions) + 1)
                if label not in matched_labels
            ]
            areas = np.bincount(
                labels.reshape(-1), minlength=int(num_predictions) + 1
            )
            false_component_areas = [int(areas[label]) for label in false_labels]
            matched_lookup = np.zeros(int(num_predictions) + 1, dtype=bool)
            false_lookup = np.zeros(int(num_predictions) + 1, dtype=bool)
            if matched_labels:
                matched_lookup[np.asarray(sorted(matched_labels), dtype=np.int64)] = True
            if false_labels:
                false_lookup[np.asarray(false_labels, dtype=np.int64)] = True
            background = ~sample.mask
            unmatched_background = int(
                np.count_nonzero(false_lookup[labels] & background)
            )
            matched_spillover = int(
                np.count_nonzero(matched_lookup[labels] & background)
            )
            if unmatched_background + matched_spillover != int(
                matched.num_fp_pixels
            ):
                raise RuntimeError("selected-point FP-pixel attribution mismatch")
            if len(false_component_areas) != int(matched.num_fp_components):
                raise RuntimeError("selected-point false-component anatomy mismatch")
            row = {
                "image_id": f"{domain}/{sample.image_id}",
                "domain": domain,
                "tp_objects": int(matched.num_tp_objects),
                "gt_objects": int(matched.num_gt),
                "fp_components": int(matched.num_fp_components),
                "fp_pixels": int(matched.num_fp_pixels),
                "total_pixels": int(sample.mask.size),
                "false_component_areas": false_component_areas,
                "unmatched_background_fp_pixels": unmatched_background,
                "matched_spillover_fp_pixels": matched_spillover,
            }
            rows.append(row)
            for field in totals:
                totals[field] += int(row[field])
        per_domain[domain] = _count_metrics(totals)
    pooled = _count_metrics(
        {
            field: sum(int(row[field]) for row in per_domain.values())
            for field in (
                "tp_objects",
                "gt_objects",
                "fp_components",
                "fp_pixels",
                "total_pixels",
            )
        }
    )
    return {"per_image_rows": rows, "per_domain": per_domain, "pooled": pooled}


def _verify_selected_counts(
    selection: Mapping[str, Any],
    mode: str,
    audited: Mapping[str, Any],
) -> None:
    selected = selection[mode]
    if selected.get("found") is not True:
        raise RuntimeError(f"required exact operating point is missing: {mode}")
    expected_rows = selected.get("source_rows")
    expected_pooled = selected.get("operating_point")
    if not isinstance(expected_rows, Mapping) or not isinstance(
        expected_pooled, Mapping
    ):
        raise RuntimeError(f"selected exact point lacks raw counts: {mode}")
    fields = (
        "tp_objects",
        "gt_objects",
        "fp_components",
        "fp_pixels",
        "total_pixels",
    )
    if set(expected_rows) != set(audited["per_domain"]):
        raise RuntimeError("selected exact point domain set drifted")
    for domain in expected_rows:
        for field in fields:
            if int(expected_rows[domain][field]) != int(
                audited["per_domain"][domain][field]
            ):
                raise RuntimeError(
                    f"DSU/legacy selected-point mismatch: {mode}.{domain}.{field}"
                )
    for field in fields:
        if int(expected_pooled[field]) != int(audited["pooled"][field]):
            raise RuntimeError(f"DSU/legacy selected-point mismatch: {mode}.{field}")


def _saturation_audit(
    samples_by_domain: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    per_domain: dict[str, Any] = {}
    pooled_logits: list[np.ndarray] = []
    for domain in sorted(samples_by_domain):
        saturated: list[np.ndarray] = []
        exact_one_pixels = 0
        exact_one_background = 0
        exact_one_target = 0
        total_pixels = 0
        for sample in samples_by_domain[domain]:
            exact_one = sample.probability == np.float32(1.0)
            values = sample.logits[exact_one]
            if values.size:
                saturated.append(values)
                pooled_logits.append(values)
            exact_one_pixels += int(values.size)
            exact_one_background += int(np.count_nonzero(exact_one & ~sample.mask))
            exact_one_target += int(np.count_nonzero(exact_one & sample.mask))
            total_pixels += int(sample.mask.size)
        values = np.concatenate(saturated) if saturated else np.asarray([], np.float32)
        per_domain[domain] = {
            "total_pixels": total_pixels,
            "exact_probability_one_pixels": exact_one_pixels,
            "exact_probability_one_background_pixels": exact_one_background,
            "exact_probability_one_target_pixels": exact_one_target,
            "unique_raw_logits_inside_probability_one": int(np.unique(values).size),
            "raw_logit_min_inside_probability_one": float(values.min())
            if values.size
            else None,
            "raw_logit_max_inside_probability_one": float(values.max())
            if values.size
            else None,
            "distinct_decision_states_inside_saturated_region": int(
                np.unique(values).size
            ),
        }
    pooled = (
        np.concatenate(pooled_logits) if pooled_logits else np.asarray([], np.float32)
    )
    return {
        "per_domain": per_domain,
        "pooled": {
            "exact_probability_one_pixels": int(
                sum(row["exact_probability_one_pixels"] for row in per_domain.values())
            ),
            "exact_probability_one_background_pixels": int(
                sum(
                    row["exact_probability_one_background_pixels"]
                    for row in per_domain.values()
                )
            ),
            "unique_raw_logits_inside_probability_one": int(np.unique(pooled).size),
            "raw_logit_min_inside_probability_one": float(pooled.min())
            if pooled.size
            else None,
            "raw_logit_max_inside_probability_one": float(pooled.max())
            if pooled.size
            else None,
            "distinct_decision_states_inside_saturated_region": int(
                np.unique(pooled).size
            ),
        },
    }


def _budget_excess(
    row: Mapping[str, Any], *, pixel_budget: float, component_budget: float
) -> dict[str, Any]:
    pixel = float(row["fa_pixel"])
    component = float(row["fa_component_mp"])
    return {
        "joint_budget_satisfied": pixel <= pixel_budget
        and component <= component_budget,
        "pixel_absolute_excess": max(pixel - pixel_budget, 0.0),
        "pixel_budget_ratio": float(pixel / pixel_budget),
        "component_absolute_excess": max(component - component_budget, 0.0),
        "component_budget_ratio": float(component / component_budget),
    }


def _enrich_calibration_budget_excess(
    payload: dict[str, Any], *, pixel_budget: float, component_budget: float
) -> None:
    for result in payload.get("cross_application", {}).values():
        result["target_budget_excess"] = _budget_excess(
            result["target_operating_point"],
            pixel_budget=pixel_budget,
            component_budget=component_budget,
        )
    ranks = payload.get("tail_rank_and_quantile", {})
    if len(ranks) == 2:
        names = sorted(ranks)
        payload["pairwise_tail_rank_gap"] = {
            "domains": names,
            "total_tail_fraction_absolute_gap": abs(
                float(ranks[names[0]]["total_tail_fraction"])
                - float(ranks[names[1]]["total_tail_fraction"])
            ),
            "background_tail_fraction_absolute_gap": abs(
                float(ranks[names[0]]["background_tail_fraction"])
                - float(ranks[names[1]]["background_tail_fraction"])
            ),
        }


def _compute_rescue_artifacts(
    loaded: Mapping[str, Mapping[str, Any]],
    *,
    dense_grid_size: int,
    api: AlgorithmAPI,
    probability_points: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Compute exact formal evidence plus every required diagnostic audit."""

    exact_curves: dict[str, Any] = {}
    operating_points: dict[str, dict[str, Any]] = {
        name: {} for name, _, _ in BUDGETS
    }
    domain_oracles: dict[str, dict[str, Any]] = {
        name: {} for name, _, _ in BUDGETS
    }
    dense_selections: dict[str, Any] = {}
    dense_indices: dict[str, list[int]] = {}
    saturation: dict[str, Any] = {}

    for role in ("control", "full"):
        samples_by_domain = loaded[role]
        exact = api.enumerate_exact_shared_states(
            samples_by_domain,
            loose_pixel_budget=1.0e-5,
            **MATCHING_PROTOCOL,
        )
        exact_curves[role] = exact
        states = exact["states"]
        dense_indices[role] = api.deterministic_dense_state_indices(
            len(states), dense_grid_size
        )
        dense_states = [states[index] for index in dense_indices[role]]
        dense_selections[role] = api.select_dense_operating_points(
            dense_states, BUDGETS
        )
        saturation[role] = _saturation_audit(samples_by_domain)
        for name, pixel, component in BUDGETS:
            selection = api.select_exact_shared_source_operating_points(
                exact,
                pixel_budget=pixel,
                component_budget=component,
            )
            operating_points[name][role] = {
                "selection": selection,
                "gate_point": _extract_gate_point(selection),
            }
            domain_oracles[name][role] = api.select_domain_oracles(
                exact,
                pixel_budget=pixel,
                component_budget=component,
            )

    realized = api.build_realized_fa_sensitivity(
        {
            name: operating_points[name]["control"]["selection"]
            for name, _, _ in BUDGETS
        },
        exact_curves["full"]["states"],
    )
    for name, _, _ in BUDGETS:
        control = operating_points[name]["control"]["gate_point"]
        full = operating_points[name]["full"]["gate_point"]
        operating_points[name]["deltas_full_minus_control"] = {
            "pooled_pd": float(full["pooled_pd"] - control["pooled_pd"]),
            "worst_pd": float(full["worst_pd"] - control["worst_pd"]),
            "macro_pd": float(full["macro_pd"] - control["macro_pd"]),
            "domain_pd": {
                domain: float(full["domain_pd"][domain] - control["domain_pd"][domain])
                for domain in sorted(control["domain_pd"])
            },
        }
        operating_points[name]["realized_fa_sensitivity"] = realized[name]

    calibration: dict[str, Any] = {
        "schema_version": "rc-irstd-aaai27-cross-domain-calibration-gap-v1",
        "diagnostic_only": True,
        "per_domain_oracles_not_used_for_formal_gate": True,
        "roles": {},
    }
    for role in ("control", "full"):
        calibration["roles"][role] = {}
        for name, pixel, component in BUDGETS:
            selection = operating_points[name][role]["selection"]
            oracles = domain_oracles[name][role]
            pooled_gap = api.build_cross_domain_calibration_gap(
                exact_curves[role],
                selection,
                oracles,
                samples_by_domain=loaded[role],
            )
            worst_gap = api.build_cross_domain_calibration_gap(
                exact_curves[role],
                {"source_pooled": selection["source_worst"]},
                oracles,
                samples_by_domain=loaded[role],
            )
            _enrich_calibration_budget_excess(
                pooled_gap, pixel_budget=pixel, component_budget=component
            )
            _enrich_calibration_budget_excess(
                worst_gap, pixel_budget=pixel, component_budget=component
            )
            calibration["roles"][role][name] = {
                "pixel_budget": pixel,
                "component_budget": component,
                "domain_oracles": oracles,
                "formal_shared_pooled": pooled_gap,
                "formal_shared_worst": worst_gap,
            }

    dense_gap: dict[str, Any] = {
        "schema_version": "rc-irstd-aaai27-dense-tail-grid-gap-v1",
        "diagnostic_only": True,
        "exact_raw_logit_states_are_primary": True,
        "dense_grid_definition": (
            "1024_or_requested_integer_rank_spaced_subset_of_exact_retained_"
            "fp32_breakpoints_including_reject_and_lowest_tail_state"
        ),
        "requested_dense_grid_size": dense_grid_size,
        "roles": {},
    }
    for role in ("control", "full"):
        role_payload: dict[str, Any] = {
            "num_exact_states": len(exact_curves[role]["states"]),
            "num_dense_states": len(dense_indices[role]),
            "dense_exact_state_indices": dense_indices[role],
            "saturation_audit": saturation[role],
            "budgets": {},
        }
        for name, _, _ in BUDGETS:
            exact_selection = operating_points[name][role]["selection"]
            dense_selection = dense_selections[role][name]
            probability = (
                probability_points.get(role, {}).get(name)
                if probability_points is not None
                else None
            )
            budget_payload: dict[str, Any] = {}
            for mode in ("source_pooled", "source_worst"):
                exact_pd = (
                    float(exact_selection[mode]["operating_point"]["pd"])
                    if mode == "source_pooled"
                    else float(exact_selection[mode]["worst_domain_pd"])
                )
                dense_pd = (
                    float(dense_selection[mode]["operating_point"]["pd"])
                    if mode == "source_pooled"
                    else float(dense_selection[mode]["worst_domain_pd"])
                )
                probability_pd = None
                probability_all_reject = None
                if probability is not None:
                    probability_pd = (
                        float(probability[mode]["operating_point"]["pd"])
                        if mode == "source_pooled"
                        else float(probability[mode]["worst_domain_pd"])
                    )
                    probability_threshold = float(
                        probability[mode]["operating_point"]["threshold"]
                    )
                    probability_all_reject = probability_threshold > 1.0
                budget_payload[mode] = {
                    "exact_pd": exact_pd,
                    "probability_grid_pd": probability_pd,
                    "dense_logit_grid_pd": dense_pd,
                    "exact_minus_probability_pd": float(exact_pd - probability_pd)
                    if probability_pd is not None
                    else None,
                    "exact_minus_dense_pd": float(exact_pd - dense_pd),
                    "probability_selected_all_reject": probability_all_reject,
                }
            role_payload["budgets"][name] = budget_payload
        dense_gap["roles"][role] = role_payload

    false_alarm: dict[str, Any] = {
        "schema_version": "rc-irstd-aaai27-false-alarm-concentration-v1",
        "diagnostic_only": True,
        "not_a_go_hold_condition": True,
        "roles": {},
    }
    for role in ("control", "full"):
        cache: dict[float | None, dict[str, Any]] = {}
        role_output: dict[str, Any] = {}
        for name, _, _ in BUDGETS:
            role_output[name] = {}
            selection = operating_points[name][role]["selection"]
            for mode in ("source_pooled", "source_worst"):
                selected = selection[mode]
                threshold_value = selected["threshold_logit_float32"]
                threshold = (
                    None
                    if bool(selected["all_reject_sentinel"])
                    else float(threshold_value)
                )
                if threshold not in cache:
                    cache[threshold] = _per_image_selected_threshold_audit(
                        loaded[role], threshold
                    )
                audited = cache[threshold]
                _verify_selected_counts(selection, mode, audited)
                rows = audited["per_image_rows"]
                role_output[name][mode] = {
                    "threshold_logit_float32": threshold,
                    "all_reject_sentinel": threshold is None,
                    "legacy_full_image_counts_verified": True,
                    "pooled": api.summarize_false_alarm_concentration(rows),
                    "per_domain": {
                        domain: api.summarize_false_alarm_concentration(
                            [row for row in rows if row["domain"] == domain]
                        )
                        for domain in sorted(loaded[role])
                    },
                }
        false_alarm["roles"][role] = role_output

    return {
        "exact_source_curves": exact_curves,
        "operating_points": operating_points,
        "dense_tail_grid_gap": dense_gap,
        "cross_domain_calibration_gap": calibration,
        "false_alarm_concentration": false_alarm,
    }


def _freeze_evidence(output_root: Path, artifacts: Mapping[str, Any]) -> dict[str, Any]:
    paths: dict[str, Path] = {}
    for role in ("control", "full"):
        path = output_root / "exact_source_curves" / f"{role}.json"
        _write_once_json(path, artifacts["exact_source_curves"][role])
        paths[f"exact_source_curves/{role}"] = path
    for name, _, _ in BUDGETS:
        path = output_root / "operating_points" / f"{name}.json"
        _write_once_json(path, artifacts["operating_points"][name])
        paths[f"operating_points/{name}"] = path
    for name in (
        "dense_tail_grid_gap",
        "cross_domain_calibration_gap",
        "false_alarm_concentration",
    ):
        path = output_root / f"{name}.json"
        _write_once_json(path, artifacts[name])
        paths[name] = path
    return {
        name: {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "sha256_sidecar": str(_sha_sidecar(path).resolve()),
        }
        for name, path in paths.items()
    }


def run_rescue(
    *,
    output_root: str | Path,
    registered_at: str | None = None,
    dense_grid_size: int = 1024,
) -> dict[str, Any]:
    if isinstance(dense_grid_size, bool) or not isinstance(dense_grid_size, int):
        raise ValueError("dense_grid_size must be an integer")
    if dense_grid_size < 2:
        raise ValueError("dense_grid_size must be at least 2")
    layout = Layout(
        project_root=PROJECT_ROOT.resolve(),
        output_root=Path(output_root).expanduser().resolve(),
    )
    forbidden_target = (layout.project_root / "datasets/NUAA-SIRST").resolve()
    if layout.output_root == forbidden_target or forbidden_target in layout.output_root.parents:
        raise RuntimeError("Rescue output must not be written under the NUAA dataset")
    layout.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = layout.output_root.parent / ".raw_logit_rescue_v1.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        snapshots = _historical_snapshots(layout)
        try:
            _validate_historical_contract(layout)
            resolved_registered_at = _resolve_registered_at(
                layout.output_root, registered_at
            )
            api = _load_algorithm_api()
            loaded, input_records = _load_and_validate_inputs(layout)
            probability_points, probability_bindings = (
                _load_historical_probability_points(layout)
            )
            code_bindings = _callable_source_bindings(api, layout)

            amendment_path = layout.output_root / "protocol_amendment.json"
            _write_once_json(
                amendment_path,
                _amendment_payload(
                    layout,
                    registered_at=resolved_registered_at,
                    code_bindings=code_bindings,
                    dense_grid_size=dense_grid_size,
                ),
            )
            input_manifest_path = layout.output_root / "input_manifest.json"
            _write_once_json(
                input_manifest_path,
                _input_manifest_payload(
                    layout,
                    registered_at=resolved_registered_at,
                    amendment_path=amendment_path,
                    records=input_records,
                    probability_bindings=probability_bindings,
                ),
            )
            input_hashes_path = layout.output_root / "input_hashes.json"
            _write_once_json(
                input_hashes_path,
                _input_hashes_payload(
                    registered_at=resolved_registered_at,
                    amendment_path=amendment_path,
                    input_manifest_path=input_manifest_path,
                    records=input_records,
                    probability_bindings=probability_bindings,
                ),
            )

            # Compute every evidence artifact before writing any of them, so an
            # algorithm failure cannot leave an apparently terminal decision.
            artifacts = _compute_rescue_artifacts(
                loaded,
                dense_grid_size=dense_grid_size,
                api=api,
                probability_points=probability_points,
            )
            evidence_bindings = _freeze_evidence(layout.output_root, artifacts)
            evidence_manifest_path = layout.output_root / "evidence_manifest.json"
            _write_once_json(
                evidence_manifest_path,
                {
                    "schema_version": EVIDENCE_MANIFEST_SCHEMA,
                    "registered_at": resolved_registered_at,
                    "input_manifest_sha256": file_sha256(input_manifest_path),
                    "input_hashes_sha256": file_sha256(input_hashes_path),
                    "exact_raw_logit_states_are_primary": True,
                    "dense_grid_is_diagnostic_only": True,
                    "artifacts": evidence_bindings,
                },
            )

            control_points = {
                name: artifacts["operating_points"][name]["control"]["gate_point"]
                for name, _, _ in BUDGETS
            }
            full_points = {
                name: artifacts["operating_points"][name]["full"]["gate_point"]
                for name, _, _ in BUDGETS
            }
            gate = evaluate_rescue_decision(
                control_points,
                full_points,
                numeric_atol=DEFAULT_NUMERIC_ATOL,
            )
            decision = {
                **gate,
                "runner_schema_version": RUNNER_SCHEMA,
                "scope": "source_only_inner_lodo_raw_logit_rescue_v1",
                "registered_at": resolved_registered_at,
                "protocol_amendment_sha256": file_sha256(amendment_path),
                "input_manifest_sha256": file_sha256(input_manifest_path),
                "input_hashes_sha256": file_sha256(input_hashes_path),
                "evidence_manifest_sha256": file_sha256(evidence_manifest_path),
                "outer_target_images_used": False,
                "outer_target_labels_used": False,
            }
            decision_path = layout.output_root / "rescue_decision.json"
            _write_once_json(decision_path, decision)
            authorization = {
                "schema_version": AUTHORIZATION_SCHEMA,
                "registered_at": resolved_registered_at,
                "decision": gate["decision"],
                "derived_from_rescue_decision": str(decision_path.resolve()),
                "derived_from_rescue_decision_sha256": file_sha256(decision_path),
                "tier2_source_lodo_authorized": (
                    gate["decision"] == RESCUE_GO_TIER2
                ),
                "tier2_authorized": gate["decision"] == RESCUE_GO_TIER2,
                "outer_target_image_access_authorized": False,
                "outer_target_label_access_authorized": False,
                "outer_target_access_authorized": False,
                "outer_target_images_used": False,
                "outer_target_labels_used": False,
                "historical_probability_tier1_hold_remains_immutable": True,
            }
            authorization_path = layout.output_root / "effective_authorization.json"
            _write_once_json(authorization_path, authorization)
            return {
                "decision": gate["decision"],
                "decision_path": str(decision_path),
                "authorization_path": str(authorization_path),
            }
        finally:
            for snapshot in snapshots:
                _verify_snapshot(snapshot)
            if layout.target_label_hold.read_text(encoding="utf-8") != "HOLD\n":
                raise RuntimeError("Outer-target label HOLD sentinel drifted")
            fcntl.flock(lock, fcntl.LOCK_UN)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=str(
            PROJECT_ROOT
            / "artifacts/aaai27/audit/phase3_source_lodo_gate/raw_logit_rescue_v1"
        ),
    )
    parser.add_argument(
        "--registered-at",
        help="Timezone-aware ISO timestamp; existing amendment value is reused by default",
    )
    parser.add_argument("--dense-grid-size", type=int, default=1024)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = run_rescue(
        output_root=args.output_root,
        registered_at=args.registered_at,
        dense_grid_size=args.dense_grid_size,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
