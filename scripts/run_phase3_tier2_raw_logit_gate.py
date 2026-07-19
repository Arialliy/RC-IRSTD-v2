#!/usr/bin/env python3
"""Run the immutable source-only Tier2 raw-logit ablation gate.

This runner never trains a model and never opens the NUAA outer target.  It
accepts a frozen coordinator handoff after all four Tier2 exports exist,
enumerates exact shared raw-logit states for ``no_contrast`` and
``no_component``, independently rechecks every selected point with the legacy
full-image evaluator, and compares each ablation with the already-frozen full
baseline from raw-logit rescue v1.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib
import inspect
import json
import math
import os
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from evaluation.artifact_integrity import file_sha256
from evaluation.raw_logit_oracle import (
    RawLogitSample,
    load_formal_raw_logit_directory,
    raw_logit_stream_sha256,
)
from evaluation.threshold_sweep import domain_key
from scripts.run_phase3_raw_logit_rescue_v1 import (
    _atomic_write_bytes,
    _canonical_json_bytes,
    _contains_nuaa,
    _extract_gate_point,
    _load_json,
    _verify_digest_sidecar,
    _write_once_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_ROOT = Path("artifacts/aaai27/audit/phase3_source_lodo_gate")
RESCUE_RELATIVE = HISTORICAL_ROOT / "raw_logit_rescue_v1"
TIER2_RELATIVE = HISTORICAL_ROOT / "tier2_raw_logit_gate_v1"
DEFAULT_HANDOFF_NAME = "TIER2_HANDOFF.json"

HANDOFF_SCHEMA = "rc-irstd-aaai27-phase3-tier2-raw-logit-handoff-v1"
RUNNER_SCHEMA = "rc-irstd-aaai27-phase3-tier2-raw-logit-runner-v1"
INPUT_MANIFEST_SCHEMA = "rc-irstd-aaai27-phase3-tier2-input-manifest-v1"
EVIDENCE_MANIFEST_SCHEMA = "rc-irstd-aaai27-phase3-tier2-evidence-manifest-v1"
DECISION_SCHEMA = "rc-irstd-aaai27-phase3-tier2-decision-v1"
AUTHORIZATION_SCHEMA = "rc-irstd-aaai27-phase3-tier2-authorization-v1"

TIER2_GO_TIER3 = "TIER2_GO_TIER3"
TIER2_HOLD = "TIER2_HOLD"
NUMERIC_ATOL = 1.0e-12
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
EXPECTED_TIER2_POLICY = {
    "enabled_only_after_tier1_go": True,
    "roles": ["no_contrast", "no_component"],
    "comparison_pairs": [
        ["full", "no_contrast"],
        ["full", "no_component"],
    ],
    "pass_conditions": [
        {
            "metric": "macro_pd_delta",
            "budgets": ["strict"],
            "operator": "greater_than",
            "value": 0.0,
        },
        {
            "metric": "each_domain_pd_delta",
            "budgets": ["strict"],
            "operator": "greater_than_or_equal",
            "value": 0.0,
        },
        {
            "metric": "pooled_pd_delta",
            "budgets": ["strict", "medium", "loose"],
            "operator": "greater_than_or_equal",
            "value": 0.0,
        },
        {
            "metric": "worst_pd_delta",
            "budgets": ["strict", "medium", "loose"],
            "operator": "greater_than_or_equal",
            "value": 0.0,
        },
    ],
    "numeric_tolerance": NUMERIC_ATOL,
    "missing_or_infeasible_point": "HOLD_TIER2",
}


@dataclass(frozen=True)
class RunSpec:
    role: str
    fold: str
    held_out_source: str
    training_source: str
    physical_gpu: int

    @property
    def run_id(self) -> str:
        return f"{self.role}_{self.fold}"

    def run_dir(self, root: Path) -> Path:
        return (
            root
            / "outputs/aaai27/detectors/source_lodo_gate/seed42"
            / self.role
            / self.fold
        )


RUN_SPECS: tuple[RunSpec, ...] = (
    RunSpec("no_contrast", "heldout_nudt", "NUDT-SIRST", "IRSTD-1K", 2),
    RunSpec("no_component", "heldout_nudt", "NUDT-SIRST", "IRSTD-1K", 3),
    RunSpec("no_contrast", "heldout_irstd", "IRSTD-1K", "NUDT-SIRST", 2),
    RunSpec("no_component", "heldout_irstd", "IRSTD-1K", "NUDT-SIRST", 3),
)


@dataclass(frozen=True)
class AlgorithmAPI:
    enumerate_exact_shared_states: Callable[..., Any]
    select_exact_shared_source_operating_points: Callable[..., Any]
    evaluate_domains_at_threshold: Callable[..., Any]


def _load_algorithm_api() -> AlgorithmAPI:
    module = importlib.import_module("evaluation.raw_logit_source_operating_point")
    return AlgorithmAPI(
        enumerate_exact_shared_states=module.enumerate_exact_shared_states,
        select_exact_shared_source_operating_points=(
            module.select_exact_shared_source_operating_points
        ),
        evaluate_domains_at_threshold=module.evaluate_domains_at_threshold,
    )


def _parse_time(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("handoff registered_at must be a timezone-aware ISO string")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("handoff registered_at must include a UTC offset")
    return value


def _sidecar(path: Path) -> Path:
    return path.with_suffix(".sha256")


def _require_frozen_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or _sidecar(path).is_symlink():
        raise RuntimeError(f"Frozen artifact must not be a symlink: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    _verify_digest_sidecar(path)
    for item in (path, _sidecar(path)):
        if stat.S_IMODE(item.stat().st_mode) & 0o222:
            raise RuntimeError(f"Frozen artifact remains writable: {item}")
    return _load_json(path)


def _lexical_absolute(path: str | Path, *, name: str) -> Path:
    raw = Path(path).expanduser()
    if ".." in raw.parts:
        raise RuntimeError(f"{name} contains a forbidden parent traversal")
    return Path(os.path.abspath(raw))


def _assert_no_symlink_components(path: Path, *, anchor: Path, name: str) -> None:
    if path != anchor and anchor not in path.parents:
        raise RuntimeError(f"{name} escapes the project root: {path}")
    current = anchor
    candidates = [anchor]
    if path != anchor:
        for part in path.relative_to(anchor).parts:
            current = current / part
            candidates.append(current)
    for candidate in candidates:
        if candidate.is_symlink():
            raise RuntimeError(f"{name} contains a symlink component: {candidate}")


def _validate_gate_layout(
    *,
    project_root: str | Path,
    output_root: str | Path,
    handoff_path: str | Path,
) -> tuple[Path, Path, Path]:
    """Require the canonical project-relative Tier2 write and handoff layout."""

    root_input = _lexical_absolute(project_root, name="project_root")
    if root_input.is_symlink():
        raise RuntimeError(f"project_root must not be a symlink: {root_input}")
    root = root_input.resolve()
    output = _lexical_absolute(output_root, name="output_root")
    handoff = _lexical_absolute(handoff_path, name="handoff_path")
    expected_output = root / TIER2_RELATIVE
    expected_handoff = expected_output / DEFAULT_HANDOFF_NAME
    if output != expected_output:
        raise RuntimeError(
            f"Tier2 output_root must be the canonical audit directory: {expected_output}"
        )
    if handoff != expected_handoff:
        raise RuntimeError(f"Tier2 handoff must be canonical: {expected_handoff}")
    if _contains_nuaa(output):
        raise RuntimeError("Tier2 source-only gate refuses any NUAA write path")
    _assert_no_symlink_components(output, anchor=root, name="output_root")
    _assert_no_symlink_components(handoff, anchor=root, name="handoff_path")
    return root, output, handoff


def _write_gate_json(output_root: Path, path: Path, payload: Any) -> Path:
    _assert_no_symlink_components(path, anchor=output_root, name="Tier2 artifact")
    _assert_no_symlink_components(
        _sidecar(path), anchor=output_root, name="Tier2 digest sidecar"
    )
    # Recover only the runner's own known write order.  A byte-identical JSON
    # may exist without its sidecar after SIGKILL between the two atomic
    # writes; mismatched bytes, a sidecar-only state, or symlinks still fail
    # closed.  This never touches the frozen rescue tree.
    sidecar = _sidecar(path)
    if path.exists() or sidecar.exists():
        expected = _canonical_json_bytes(payload)
        if (
            path.is_symlink()
            or sidecar.is_symlink()
            or not path.is_file()
            or path.read_bytes() != expected
        ):
            raise RuntimeError(f"Immutable Tier2 JSON content drift: {path}")
        digest_content = f"{file_sha256(path)}  {path.name}\n".encode("ascii")
        if sidecar.exists():
            if not sidecar.is_file() or sidecar.read_bytes() != digest_content:
                raise RuntimeError(f"Immutable Tier2 JSON digest drift: {sidecar}")
        else:
            _atomic_write_bytes(sidecar, digest_content)
        path.chmod(0o444)
        sidecar.chmod(0o444)
    return _write_once_json(path, payload)


def _binding_path(
    container: Mapping[str, Any],
    keys: Sequence[str],
    *,
    expected: Path,
    name: str,
) -> Path:
    raw: Any = None
    selected_key: str | None = None
    for key in keys:
        if key in container:
            raw = container[key]
            selected_key = key
            break
    if selected_key is None:
        raise RuntimeError(f"handoff lacks {name} binding")
    if isinstance(raw, Mapping):
        raw_path = raw.get("path")
        digest = raw.get("sha256")
    else:
        raw_path = raw
        digest = container.get(f"{selected_key}_sha256")
    raw_path_object = Path(str(raw_path)).expanduser()
    if raw_path_object.is_symlink():
        raise RuntimeError(f"handoff {name} path is a symlink")
    path = raw_path_object.resolve()
    if path != expected.resolve():
        raise RuntimeError(f"handoff {name} path drifted: {path}")
    if not isinstance(digest, str) or digest != file_sha256(path):
        raise RuntimeError(f"handoff {name} SHA-256 mismatch")
    return path


def _runs_mapping(raw: Any) -> dict[str, Mapping[str, Any]]:
    if isinstance(raw, Mapping):
        result = dict(raw)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        result = {}
        for entry in raw:
            if not isinstance(entry, Mapping) or not isinstance(
                entry.get("run_id"), str
            ):
                raise RuntimeError("handoff run list contains an invalid entry")
            if entry["run_id"] in result:
                raise RuntimeError(f"duplicate handoff run: {entry['run_id']}")
            result[entry["run_id"]] = entry
    else:
        raise RuntimeError("handoff runs must be a mapping or list")
    expected = {spec.run_id for spec in RUN_SPECS}
    if set(result) != expected:
        raise RuntimeError(
            f"handoff run set mismatch: expected={sorted(expected)}, got={sorted(result)}"
        )
    if any(not isinstance(value, Mapping) for value in result.values()):
        raise RuntimeError("handoff run records must be mappings")
    return result


def validate_handoff(
    handoff_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    """Validate the frozen coordinator handoff without opening score maps."""

    root = Path(project_root).expanduser().resolve()
    expected_output = root / TIER2_RELATIVE
    _, _, path = _validate_gate_layout(
        project_root=root,
        output_root=expected_output,
        handoff_path=handoff_path,
    )
    handoff = _require_frozen_json(path)
    if handoff.get("schema_version") != HANDOFF_SCHEMA:
        raise RuntimeError("Tier2 handoff schema mismatch")
    _parse_time(handoff.get("registered_at"))
    if (
        handoff.get("source_only") is not True
        or handoff.get("tier2_source_lodo_authorized") is not True
        or handoff.get("outer_target_image_access_authorized") is not False
        or handoff.get("outer_target_label_access_authorized") is not False
        or handoff.get("outer_target_images_used") is not False
        or handoff.get("outer_target_labels_used") is not False
    ):
        raise RuntimeError("Tier2 handoff violates source-only/outer-target HOLD")

    historical = root / HISTORICAL_ROOT
    rescue = root / RESCUE_RELATIVE
    _binding_path(
        handoff,
        ("preregistration", "phase3_preregistration"),
        expected=historical / "PHASE3_SOURCE_LODO_PREREGISTRATION.json",
        name="Phase3 preregistration",
    )
    _binding_path(
        handoff,
        ("rescue_decision",),
        expected=rescue / "rescue_decision.json",
        name="raw-logit rescue decision",
    )
    _binding_path(
        handoff,
        ("rescue_effective_authorization", "effective_authorization"),
        expected=rescue / "effective_authorization.json",
        name="raw-logit rescue authorization",
    )
    formal_root = PROJECT_ROOT.resolve()
    expected_runner = Path(__file__).resolve()
    if root != formal_root:
        # Test/fixture roots preserve the same project-relative layout.
        expected_runner = root / "scripts/run_phase3_tier2_raw_logit_gate.py"
    _binding_path(
        handoff,
        ("runner",),
        expected=expected_runner,
        name="Tier2 gate runner",
    )
    if "continuation" in handoff:
        _binding_path(
            handoff,
            ("continuation",),
            expected=Path(__file__).resolve().with_name(
                "coordinate_phase3_tier2_after_rescue.py"
            ),
            name="Tier2 continuation coordinator",
        )
    if handoff.get("tier2_policy") != EXPECTED_TIER2_POLICY:
        raise RuntimeError("Tier2 handoff/preregistration policy drifted")
    return handoff, _runs_mapping(handoff.get("runs"))


def _verify_rescue_chain(root: Path) -> dict[str, Any]:
    rescue = root / RESCUE_RELATIVE
    names = (
        "protocol_amendment.json",
        "input_manifest.json",
        "input_hashes.json",
        "evidence_manifest.json",
        "rescue_decision.json",
        "effective_authorization.json",
    )
    payloads = {name: _require_frozen_json(rescue / name) for name in names}
    amendment = payloads["protocol_amendment.json"]
    input_manifest = payloads["input_manifest.json"]
    input_hashes = payloads["input_hashes.json"]
    evidence = payloads["evidence_manifest.json"]
    decision = payloads["rescue_decision.json"]
    authorization = payloads["effective_authorization.json"]
    expected_links = (
        (decision, "protocol_amendment_sha256", rescue / "protocol_amendment.json"),
        (decision, "input_manifest_sha256", rescue / "input_manifest.json"),
        (decision, "input_hashes_sha256", rescue / "input_hashes.json"),
        (decision, "evidence_manifest_sha256", rescue / "evidence_manifest.json"),
        (evidence, "input_manifest_sha256", rescue / "input_manifest.json"),
        (evidence, "input_hashes_sha256", rescue / "input_hashes.json"),
        (input_hashes, "protocol_amendment_sha256", rescue / "protocol_amendment.json"),
        (input_hashes, "input_manifest_sha256", rescue / "input_manifest.json"),
        (
            authorization,
            "derived_from_rescue_decision_sha256",
            rescue / "rescue_decision.json",
        ),
    )
    for container, field, path in expected_links:
        if container.get(field) != file_sha256(path):
            raise RuntimeError(f"raw-logit rescue hash chain drifted at {field}")
    if (
        decision.get("decision") != "RESCUE_GO_TIER2"
        or decision.get("gate_valid") is not True
        or decision.get("authorizes_tier2") is not True
        or decision.get("outer_target_images_used") is not False
        or decision.get("outer_target_labels_used") is not False
        or authorization.get("tier2_source_lodo_authorized") is not True
        or authorization.get("outer_target_image_access_authorized") is not False
        or authorization.get("outer_target_label_access_authorized") is not False
    ):
        raise RuntimeError("raw-logit rescue does not authorize source-only Tier2")
    invariants = amendment.get("frozen_invariants")
    if (
        not isinstance(invariants, Mapping)
        or invariants.get("exact_state_enumeration_is_primary") is not True
        or invariants.get("shared_threshold_across_nudt_and_irstd") is not True
        or invariants.get("matching_protocol") != MATCHING_PROTOCOL
    ):
        raise RuntimeError("raw-logit rescue protocol invariants drifted")

    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("rescue evidence manifest lacks artifact bindings")
    for artifact_name, raw_binding in artifacts.items():
        if not isinstance(raw_binding, Mapping):
            raise RuntimeError(f"invalid rescue artifact binding: {artifact_name}")
        path = Path(str(raw_binding.get("path"))).resolve()
        if rescue.resolve() not in path.parents:
            raise RuntimeError(f"rescue artifact escapes rescue root: {path}")
        if raw_binding.get("sha256") != file_sha256(path):
            raise RuntimeError(f"rescue artifact hash drifted: {artifact_name}")
        _verify_digest_sidecar(path)

    full_curve_binding = artifacts.get("exact_source_curves/full")
    if not isinstance(full_curve_binding, Mapping):
        raise RuntimeError("rescue evidence lacks frozen full exact curve")
    full_curve = _load_json(Path(str(full_curve_binding["path"])))
    if (
        full_curve.get("exact_state_enumeration") is not True
        or full_curve.get("shared_threshold_across_domains") is not True
        or set(full_curve.get("domain_names", [])) != {"nudt", "irstd1k"}
    ):
        raise RuntimeError("frozen full exact curve violates rescue contract")

    full_points: dict[str, Any] = {}
    point_bindings: dict[str, Any] = {}
    for budget_name, pixel, component in BUDGETS:
        key = f"operating_points/{budget_name}"
        binding = artifacts.get(key)
        if not isinstance(binding, Mapping):
            raise RuntimeError(f"rescue evidence lacks {key}")
        path = Path(str(binding["path"])).resolve()
        payload = _load_json(path)
        full = payload.get("full")
        if not isinstance(full, Mapping):
            raise RuntimeError(f"rescue {key} lacks full baseline")
        selection = full.get("selection")
        if (
            not isinstance(selection, Mapping)
            or selection.get("exact_state_enumeration_is_primary") is not True
            or float(selection.get("pixel_budget", math.nan)) != pixel
            or float(selection.get("component_budget", math.nan)) != component
        ):
            raise RuntimeError(f"rescue full selection protocol drifted: {budget_name}")
        point = _extract_gate_point(full.get("gate_point", full))
        frozen_evidence = decision.get("evidence", {}).get(budget_name, {}).get("full")
        if point.get("found") is not True or not isinstance(frozen_evidence, Mapping):
            raise RuntimeError(f"rescue full gate point missing: {budget_name}")
        for field in ("pooled_pd", "worst_pd", "macro_pd", "domain_pd"):
            if point[field] != frozen_evidence.get(field):
                raise RuntimeError(
                    f"rescue decision/full point mismatch: {budget_name}.{field}"
                )
        full_points[budget_name] = point
        point_bindings[budget_name] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    return {
        "full_points": full_points,
        "full_point_bindings": point_bindings,
        "full_curve_binding": dict(full_curve_binding),
        "rescue_decision_sha256": file_sha256(rescue / "rescue_decision.json"),
        "rescue_authorization_sha256": file_sha256(
            rescue / "effective_authorization.json"
        ),
        "rescue_input_manifest": input_manifest,
        "rescue_input_hashes": input_hashes,
        "rescue_evidence_manifest_sha256": file_sha256(
            rescue / "evidence_manifest.json"
        ),
    }


def _record_bound_path(
    record: Mapping[str, Any],
    key: str,
    expected: Path,
) -> Path:
    raw = record.get(key)
    if isinstance(raw, Mapping):
        path = Path(str(raw.get("path"))).resolve()
        digest = raw.get("sha256")
    else:
        path = Path(str(raw)).resolve()
        digest = record.get(f"{key}_sha256")
    if path != expected.resolve():
        raise RuntimeError(f"handoff run {key} path mismatch: {path}")
    if digest != file_sha256(path):
        raise RuntimeError(f"handoff run {key} SHA-256 mismatch: {path}")
    return path


def _load_tier2_inputs(
    root: Path,
    run_bindings: Mapping[str, Mapping[str, Any]],
    rescue_chain: Mapping[str, Any],
    *,
    loader: Callable[..., Any] = load_formal_raw_logit_directory,
) -> tuple[dict[str, dict[str, Sequence[RawLogitSample]]], dict[str, Any]]:
    loaded: dict[str, dict[str, Sequence[RawLogitSample]]] = {
        "no_contrast": {},
        "no_component": {},
    }
    records: dict[str, Any] = {}
    rescue_runs = rescue_chain["rescue_input_manifest"].get("runs")
    if not isinstance(rescue_runs, Mapping):
        raise RuntimeError("rescue input manifest lacks full run bindings")

    for spec in RUN_SPECS:
        binding = run_bindings[spec.run_id]
        expected_scalar = {
            "role": spec.role,
            "fold": spec.fold,
            "held_out_source": spec.held_out_source,
            "training_source": spec.training_source,
        }
        for field, expected in expected_scalar.items():
            observed = binding.get(field)
            if field == "held_out_source" and observed is None:
                observed = binding.get("held_out_source_pseudo_target")
            if observed != expected:
                raise RuntimeError(f"handoff {spec.run_id}.{field} mismatch")

        run_dir = spec.run_dir(root).resolve()
        checkpoint = _record_bound_path(binding, "checkpoint", run_dir / "last.pt")
        identity_path = _record_bound_path(
            binding, "phase3_identity", run_dir / "PHASE3_IDENTITY.json"
        )
        export_identity_path = _record_bound_path(
            binding, "export_identity", run_dir / "EXPORT_IDENTITY.json"
        )
        score_dir = Path(str(binding.get("score_dir"))).resolve()
        expected_score_dir = run_dir / "scores_heldout_train"
        if score_dir != expected_score_dir:
            raise RuntimeError(f"handoff score_dir mismatch: {spec.run_id}")
        manifest_path = score_dir / "manifest.json"
        manifest_digest = binding.get("score_manifest_sha256")
        if manifest_digest != file_sha256(manifest_path):
            raise RuntimeError(f"handoff score manifest hash mismatch: {spec.run_id}")

        samples, manifest, integrity, contract = loader(
            score_dir, expected_split_role="train"
        )
        if _contains_nuaa(manifest) or _contains_nuaa(contract):
            raise RuntimeError(f"NUAA reference detected in Tier2 input: {spec.run_id}")
        if (
            domain_key(str(contract.get("target_dataset")))
            != domain_key(spec.held_out_source)
            or [domain_key(str(value)) for value in contract.get("source_datasets", [])]
            != [domain_key(spec.training_source)]
            or contract.get("split_role") != "train"
            or contract.get("requested_split") != "train"
            or manifest.get("labels_loaded") is not True
        ):
            raise RuntimeError(f"Tier2 inner-LODO closure mismatch: {spec.run_id}")

        checkpoint_sha = file_sha256(checkpoint)
        identity = _load_json(identity_path)
        export_identity = _load_json(export_identity_path)
        if (
            contract.get("detector_weight_sha256") != checkpoint_sha
            or identity.get("run_id") != spec.run_id
            or identity.get("role") != spec.role
            or identity.get("held_out_pseudo_target") != spec.held_out_source
            or identity.get("training_source") != spec.training_source
            or identity.get("checkpoint_sha256") != checkpoint_sha
            or identity.get("checkpoint_selection") != "fixed_last"
            or identity.get("checkpoint_epoch") != 79
            or identity.get("physical_gpu") != spec.physical_gpu
            or identity.get("outer_target_labels_used") is not False
            or export_identity.get("run_id") != spec.run_id
            or export_identity.get("source_only") is not True
            or export_identity.get("checkpoint_sha256") != checkpoint_sha
        ):
            raise RuntimeError(f"Tier2 checkpoint/export identity mismatch: {spec.run_id}")
        output = export_identity.get("output")
        if (
            not isinstance(output, Mapping)
            or output.get("manifest_sha256") != integrity["manifest_sha256"]
            or output.get("records_sha256") != integrity["records_sha256"]
            or output.get("ordered_image_ids_sha256")
            != integrity["ordered_image_ids_sha256"]
        ):
            raise RuntimeError(f"Tier2 export output binding mismatch: {spec.run_id}")

        full_run_id = f"full_{spec.fold}"
        frozen_full = rescue_runs.get(full_run_id)
        if not isinstance(frozen_full, Mapping):
            raise RuntimeError(f"rescue input lacks matched full fold: {full_run_id}")
        matched_values = {
            "score_ordered_image_ids_sha256": integrity[
                "ordered_image_ids_sha256"
            ],
            "score_num_records": integrity["num_records"],
            "split_file_sha256": contract["split_file_sha256"],
            "split_ordered_ids_sha256": contract["split_ordered_ids_sha256"],
        }
        for field, value in matched_values.items():
            if frozen_full.get(field) != value:
                raise RuntimeError(
                    f"Tier2/full matched-fold binding differs: {spec.run_id}.{field}"
                )

        domain = domain_key(spec.held_out_source)
        loaded[spec.role][domain] = samples
        records[spec.run_id] = {
            **expected_scalar,
            "score_dir": str(score_dir),
            "score_manifest_sha256": integrity["manifest_sha256"],
            "score_records_sha256": integrity["records_sha256"],
            "score_ordered_image_ids_sha256": integrity[
                "ordered_image_ids_sha256"
            ],
            "score_num_records": integrity["num_records"],
            "raw_logit_stream_sha256": raw_logit_stream_sha256(samples),
            "checkpoint_sha256": checkpoint_sha,
            "phase3_identity_sha256": file_sha256(identity_path),
            "export_identity_sha256": file_sha256(export_identity_path),
            "split_file_sha256": contract["split_file_sha256"],
            "split_ordered_ids_sha256": contract["split_ordered_ids_sha256"],
        }

    for role in loaded:
        if set(loaded[role]) != {"nudt", "irstd1k"}:
            raise RuntimeError(f"incomplete Tier2 source coverage for {role}")
    for fold in ("heldout_nudt", "heldout_irstd"):
        first = records[f"no_contrast_{fold}"]
        second = records[f"no_component_{fold}"]
        for field in (
            "score_ordered_image_ids_sha256",
            "score_num_records",
            "split_file_sha256",
            "split_ordered_ids_sha256",
        ):
            if first[field] != second[field]:
                raise RuntimeError(f"Tier2 matched ablations differ: {fold}.{field}")
    return loaded, records


def _counts_equal(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    fields = ("tp_objects", "gt_objects", "fp_components", "fp_pixels", "total_pixels")
    return all(int(first[field]) == int(second[field]) for field in fields)


def verify_selected_operating_points(
    samples_by_domain: Mapping[str, Sequence[RawLogitSample]],
    selections: Mapping[str, Mapping[str, Any]],
    *,
    evaluator: Callable[..., Any],
) -> dict[str, Any]:
    """Legacy-recheck every unique pooled/worst threshold and raw count."""

    cached: dict[str, Mapping[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    for budget_name, selection in selections.items():
        for mode in ("source_pooled", "source_worst"):
            point = selection.get(mode)
            if not isinstance(point, Mapping) or point.get("found") is not True:
                entries.append(
                    {"budget": budget_name, "mode": mode, "found": False}
                )
                continue
            threshold = point.get("threshold_logit_float32")
            cache_key = "reject" if threshold is None else repr(float(threshold))
            if cache_key not in cached:
                cached[cache_key] = evaluator(
                    samples_by_domain,
                    threshold,
                    **MATCHING_PROTOCOL,
                )
            observed = cached[cache_key]
            expected_rows = point.get("source_rows")
            if not isinstance(expected_rows, Mapping):
                raise RuntimeError(f"selected {budget_name}/{mode} lacks source rows")
            if set(expected_rows) != set(observed.get("per_domain", {})):
                raise RuntimeError(f"legacy domain set mismatch: {budget_name}/{mode}")
            for domain, row in expected_rows.items():
                if not _counts_equal(row, observed["per_domain"][domain]):
                    raise RuntimeError(
                        f"legacy full-image count mismatch: {budget_name}/{mode}/{domain}"
                    )
            if not _counts_equal(point["operating_point"], observed["pooled"]):
                raise RuntimeError(
                    f"legacy pooled count mismatch: {budget_name}/{mode}"
                )
            entries.append(
                {
                    "budget": budget_name,
                    "mode": mode,
                    "found": True,
                    "threshold_logit_float32": threshold,
                    "state_index": point.get("state_index"),
                    "legacy_full_image_raw_counts_match": True,
                }
            )
    return {
        "all_selected_points_verified": all(
            not entry["found"] or entry.get("legacy_full_image_raw_counts_match")
            for entry in entries
        ),
        "num_unique_thresholds_rechecked": len(cached),
        "entries": entries,
    }


def _compute_role_evidence(
    samples_by_domain: Mapping[str, Sequence[RawLogitSample]],
    *,
    api: AlgorithmAPI,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    enumeration = api.enumerate_exact_shared_states(
        samples_by_domain,
        loose_pixel_budget=1.0e-5,
        **MATCHING_PROTOCOL,
    )
    selections: dict[str, Any] = {}
    gate_points: dict[str, Any] = {}
    for name, pixel, component in BUDGETS:
        selection = api.select_exact_shared_source_operating_points(
            enumeration,
            pixel_budget=pixel,
            component_budget=component,
        )
        selections[name] = selection
        gate_points[name] = _extract_gate_point(selection)
    verification = verify_selected_operating_points(
        samples_by_domain,
        selections,
        evaluator=api.evaluate_domains_at_threshold,
    )
    if verification["all_selected_points_verified"] is not True:
        raise RuntimeError("legacy selected-point verification is incomplete")
    return enumeration, selections, {"gate_points": gate_points, **verification}


def evaluate_tier2_decision(
    full_points: Mapping[str, Mapping[str, Any]],
    ablation_points: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    numeric_atol: float = NUMERIC_ATOL,
) -> dict[str, Any]:
    """Pure preregistered Tier2 decision, always using full minus ablation."""

    if not math.isfinite(float(numeric_atol)) or numeric_atol < 0:
        raise ValueError("numeric_atol must be finite and non-negative")
    expected_roles = {"no_contrast", "no_component"}
    if set(ablation_points) != expected_roles:
        raise ValueError("Tier2 requires no_contrast and no_component points")
    conditions: list[dict[str, Any]] = []
    failures: list[str] = []
    evidence: dict[str, Any] = {}

    def add_condition(
        *, role: str, budget: str, metric: str, delta: float | None, strict: bool
    ) -> None:
        passed = (
            delta is not None
            and (delta > numeric_atol if strict else delta >= -numeric_atol)
        )
        conditions.append(
            {
                "comparison": f"full_minus_{role}",
                "role": role,
                "budget": budget,
                "metric": metric,
                "delta_full_minus_ablation": delta,
                "operator": "greater_than" if strict else "greater_than_or_equal",
                "value": 0.0,
                "numeric_atol": numeric_atol,
                "passed": passed,
            }
        )
        if not passed:
            failures.append(f"full_minus_{role}:{budget}:{metric}")

    for role in sorted(expected_roles):
        evidence[role] = {}
        for budget, _, _ in BUDGETS:
            full = full_points.get(budget)
            ablation = ablation_points[role].get(budget)
            if (
                not isinstance(full, Mapping)
                or not isinstance(ablation, Mapping)
                or full.get("found") is not True
                or ablation.get("found") is not True
            ):
                evidence[role][budget] = {"found": False}
                for metric in ("pooled_pd", "worst_pd"):
                    add_condition(
                        role=role,
                        budget=budget,
                        metric=metric,
                        delta=None,
                        strict=False,
                    )
                if budget == "strict":
                    add_condition(
                        role=role,
                        budget=budget,
                        metric="macro_pd",
                        delta=None,
                        strict=True,
                    )
                    for domain in ("irstd1k", "nudt"):
                        add_condition(
                            role=role,
                            budget=budget,
                            metric=f"domain_pd/{domain}",
                            delta=None,
                            strict=False,
                        )
                continue
            domain_names = set(full.get("domain_pd", {}))
            if domain_names != {"nudt", "irstd1k"} or set(
                ablation.get("domain_pd", {})
            ) != domain_names:
                raise ValueError("Tier2 gate point domain set is incomplete")
            deltas = {
                "pooled_pd": float(full["pooled_pd"])
                - float(ablation["pooled_pd"]),
                "worst_pd": float(full["worst_pd"])
                - float(ablation["worst_pd"]),
                "macro_pd": float(full["macro_pd"])
                - float(ablation["macro_pd"]),
                "domain_pd": {
                    domain: float(full["domain_pd"][domain])
                    - float(ablation["domain_pd"][domain])
                    for domain in sorted(domain_names)
                },
            }
            evidence[role][budget] = {
                "found": True,
                "full": dict(full),
                "ablation": dict(ablation),
                "deltas_full_minus_ablation": deltas,
            }
            add_condition(
                role=role,
                budget=budget,
                metric="pooled_pd",
                delta=deltas["pooled_pd"],
                strict=False,
            )
            add_condition(
                role=role,
                budget=budget,
                metric="worst_pd",
                delta=deltas["worst_pd"],
                strict=False,
            )
            if budget == "strict":
                add_condition(
                    role=role,
                    budget=budget,
                    metric="macro_pd",
                    delta=deltas["macro_pd"],
                    strict=True,
                )
                for domain, delta in deltas["domain_pd"].items():
                    add_condition(
                        role=role,
                        budget=budget,
                        metric=f"domain_pd/{domain}",
                        delta=delta,
                        strict=False,
                    )

    go = not failures
    return {
        "schema_version": DECISION_SCHEMA,
        "decision": TIER2_GO_TIER3 if go else TIER2_HOLD,
        "gate_valid": True,
        "comparison_direction": "full_minus_ablation",
        "comparison_pairs": [
            ["full", "no_contrast"],
            ["full", "no_component"],
        ],
        "criteria": {
            "strict_macro_pd_delta": "> 0",
            "strict_each_domain_pd_delta": ">= 0",
            "all_budget_pooled_pd_delta": ">= 0",
            "all_budget_worst_pd_delta": ">= 0",
            "missing_or_infeasible_point": TIER2_HOLD,
            "numeric_atol": numeric_atol,
        },
        "conditions": conditions,
        "failed_conditions": failures,
        "evidence": evidence,
        "authorizes_tier3": go,
        "authorizes_outer_target_image_access": False,
        "authorizes_outer_target_label_access": False,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
    }


def _code_bindings(api: AlgorithmAPI, root: Path) -> dict[str, str]:
    paths = {Path(__file__).resolve()}
    for function in (
        _write_once_json,
        _extract_gate_point,
        api.enumerate_exact_shared_states,
        api.select_exact_shared_source_operating_points,
        api.evaluate_domains_at_threshold,
    ):
        source = inspect.getsourcefile(function)
        if source:
            paths.add(Path(source).resolve())
    return {
        str(path.relative_to(root) if root in path.parents else path): file_sha256(path)
        for path in sorted(paths, key=str)
    }


def _freeze_evidence(output_root: Path, artifacts: Mapping[str, Any]) -> dict[str, Any]:
    bindings: dict[str, Any] = {}
    for name, payload in artifacts.items():
        path = output_root / f"{name}.json"
        _write_gate_json(output_root, path, payload)
        bindings[name] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    return bindings


def verify_frozen_tier2_gate(
    *,
    output_root: str | Path,
    handoff_path: str | Path,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root, output, handoff = _validate_gate_layout(
        project_root=project_root,
        output_root=output_root,
        handoff_path=handoff_path,
    )
    validate_handoff(handoff, project_root=root)
    _verify_rescue_chain(root)
    hold = root / "outputs/phase_state/HOLD_PHASE3_TARGET_LABEL_ACCESS"
    if not hold.is_file() or hold.is_symlink() or hold.read_text(encoding="utf-8") != "HOLD\n":
        raise RuntimeError("outer-target HOLD sentinel is absent or drifted")
    for name in (
        "input_manifest.json",
        "input_hashes.json",
        "evidence_manifest.json",
        "tier2_decision.json",
        "effective_authorization.json",
    ):
        _require_frozen_json(output / name)
    input_manifest = _load_json(output / "input_manifest.json")
    input_hashes = _load_json(output / "input_hashes.json")
    evidence = _load_json(output / "evidence_manifest.json")
    decision = _load_json(output / "tier2_decision.json")
    authorization = _load_json(output / "effective_authorization.json")
    links = (
        (input_hashes, "input_manifest_sha256", output / "input_manifest.json"),
        (evidence, "input_manifest_sha256", output / "input_manifest.json"),
        (evidence, "input_hashes_sha256", output / "input_hashes.json"),
        (decision, "input_manifest_sha256", output / "input_manifest.json"),
        (decision, "input_hashes_sha256", output / "input_hashes.json"),
        (decision, "evidence_manifest_sha256", output / "evidence_manifest.json"),
        (
            authorization,
            "derived_from_tier2_decision_sha256",
            output / "tier2_decision.json",
        ),
    )
    for container, field, path in links:
        if container.get(field) != file_sha256(path):
            raise RuntimeError(f"Tier2 frozen hash chain drifted at {field}")
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("Tier2 evidence manifest lacks artifacts")
    for name, binding in artifacts.items():
        path = Path(str(binding.get("path"))).resolve()
        if output not in path.parents or binding.get("sha256") != file_sha256(path):
            raise RuntimeError(f"Tier2 evidence binding drifted: {name}")
        _require_frozen_json(path)
    expected_go = decision.get("decision") == TIER2_GO_TIER3
    if (
        decision.get("decision") not in {TIER2_GO_TIER3, TIER2_HOLD}
        or authorization.get("tier3_source_lodo_authorized") is not expected_go
        or authorization.get("outer_target_image_access_authorized") is not False
        or authorization.get("outer_target_label_access_authorized") is not False
        or authorization.get("outer_target_images_used") is not False
        or authorization.get("outer_target_labels_used") is not False
        or input_manifest.get("outer_target_images_used") is not False
        or input_manifest.get("outer_target_labels_used") is not False
    ):
        raise RuntimeError("Tier2 frozen authorization contract drifted")
    return {
        "decision": decision["decision"],
        "decision_path": str((output / "tier2_decision.json").resolve()),
        "authorization_path": str(
            (output / "effective_authorization.json").resolve()
        ),
        "verified_only": True,
    }


def run_tier2_gate(
    *,
    handoff_path: str | Path,
    output_root: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    loader: Callable[..., Any] = load_formal_raw_logit_directory,
    algorithm_api: AlgorithmAPI | None = None,
) -> dict[str, Any]:
    root, output, handoff_file = _validate_gate_layout(
        project_root=project_root,
        output_root=output_root,
        handoff_path=handoff_path,
    )
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".tier2_raw_logit_gate.lock"
    if lock_path.is_symlink():
        raise RuntimeError(f"Tier2 gate lock must not be a symlink: {lock_path}")
    lock = lock_path.open("a+b")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError("another Tier2 raw-logit gate runner owns the lock") from error
    try:
        handoff, run_bindings = validate_handoff(handoff_file, project_root=root)
        rescue = _verify_rescue_chain(root)
        hold = root / "outputs/phase_state/HOLD_PHASE3_TARGET_LABEL_ACCESS"
        if not hold.is_file() or hold.read_text(encoding="utf-8") != "HOLD\n":
            raise RuntimeError("outer-target HOLD sentinel is absent or drifted")
        decision_exists = (output / "tier2_decision.json").exists()
        authorization_exists = (output / "effective_authorization.json").exists()
        if decision_exists and authorization_exists:
            return verify_frozen_tier2_gate(
                output_root=output,
                handoff_path=handoff_file,
                project_root=root,
            )
        if authorization_exists and not decision_exists:
            raise RuntimeError(
                "Tier2 authorization exists without its prerequisite decision"
            )

        loaded, records = _load_tier2_inputs(
            root, run_bindings, rescue, loader=loader
        )
        api = algorithm_api or _load_algorithm_api()
        registered_at = _parse_time(handoff["registered_at"])
        input_manifest = {
            "schema_version": INPUT_MANIFEST_SCHEMA,
            "registered_at": registered_at,
            "source_only": True,
            "outer_target_images_used": False,
            "outer_target_labels_used": False,
            "handoff": str(handoff_file),
            "handoff_sha256": file_sha256(handoff_file),
            "rescue_decision_sha256": rescue["rescue_decision_sha256"],
            "rescue_authorization_sha256": rescue[
                "rescue_authorization_sha256"
            ],
            "rescue_evidence_manifest_sha256": rescue[
                "rescue_evidence_manifest_sha256"
            ],
            "frozen_full_exact_curve": rescue["full_curve_binding"],
            "frozen_full_operating_points": rescue["full_point_bindings"],
            "runs": records,
            "matching_protocol": MATCHING_PROTOCOL,
            "budgets": [
                {"name": name, "pixel": pixel, "component": component}
                for name, pixel, component in BUDGETS
            ],
            "code_bindings": _code_bindings(api, root),
        }
        input_manifest_path = output / "input_manifest.json"
        _write_gate_json(output, input_manifest_path, input_manifest)
        input_hashes = {
            "schema_version": "rc-irstd-aaai27-phase3-tier2-input-hashes-v1",
            "registered_at": registered_at,
            "input_manifest_sha256": file_sha256(input_manifest_path),
            "handoff_sha256": file_sha256(handoff_file),
            "rescue_decision_sha256": rescue["rescue_decision_sha256"],
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
        }
        input_hashes_path = output / "input_hashes.json"
        _write_gate_json(output, input_hashes_path, input_hashes)

        exact_curves: dict[str, Any] = {}
        selections: dict[str, Any] = {}
        verifications: dict[str, Any] = {}
        gate_points: dict[str, Any] = {}
        for role in ("no_contrast", "no_component"):
            enumeration, role_selections, verification = _compute_role_evidence(
                loaded[role], api=api
            )
            exact_curves[role] = enumeration
            selections[role] = role_selections
            verifications[role] = verification
            gate_points[role] = verification["gate_points"]

        artifacts: dict[str, Any] = {
            "exact_source_curves/no_contrast": exact_curves["no_contrast"],
            "exact_source_curves/no_component": exact_curves["no_component"],
            "legacy_selected_point_verification": {
                "schema_version": "rc-irstd-aaai27-phase3-tier2-legacy-recheck-v1",
                "all_selected_points_verified": all(
                    value["all_selected_points_verified"]
                    for value in verifications.values()
                ),
                "roles": verifications,
            },
        }
        for budget_name, _, _ in BUDGETS:
            artifacts[f"operating_points/{budget_name}"] = {
                "schema_version": "rc-irstd-aaai27-phase3-tier2-operating-points-v1",
                "budget": budget_name,
                "full": {
                    "gate_point": rescue["full_points"][budget_name],
                    "frozen_rescue_binding": rescue["full_point_bindings"][
                        budget_name
                    ],
                },
                "no_contrast": {
                    "selection": selections["no_contrast"][budget_name],
                    "gate_point": gate_points["no_contrast"][budget_name],
                },
                "no_component": {
                    "selection": selections["no_component"][budget_name],
                    "gate_point": gate_points["no_component"][budget_name],
                },
            }
        evidence_bindings = _freeze_evidence(output, artifacts)
        evidence_manifest = {
            "schema_version": EVIDENCE_MANIFEST_SCHEMA,
            "registered_at": registered_at,
            "input_manifest_sha256": file_sha256(input_manifest_path),
            "input_hashes_sha256": file_sha256(input_hashes_path),
            "exact_raw_logit_states_are_primary": True,
            "full_baseline_reused_read_only": True,
            "all_ablation_selected_points_legacy_verified": True,
            "artifacts": evidence_bindings,
        }
        evidence_manifest_path = output / "evidence_manifest.json"
        _write_gate_json(output, evidence_manifest_path, evidence_manifest)
        gate = evaluate_tier2_decision(rescue["full_points"], gate_points)
        decision = {
            **gate,
            "runner_schema_version": RUNNER_SCHEMA,
            "scope": "source_only_inner_lodo_tier2_raw_logit_gate",
            "registered_at": registered_at,
            "handoff_sha256": file_sha256(handoff_file),
            "input_manifest_sha256": file_sha256(input_manifest_path),
            "input_hashes_sha256": file_sha256(input_hashes_path),
            "evidence_manifest_sha256": file_sha256(evidence_manifest_path),
        }
        decision_path = output / "tier2_decision.json"
        _write_gate_json(output, decision_path, decision)
        go = gate["decision"] == TIER2_GO_TIER3
        authorization = {
            "schema_version": AUTHORIZATION_SCHEMA,
            "registered_at": registered_at,
            "decision": gate["decision"],
            "derived_from_tier2_decision": str(decision_path.resolve()),
            "derived_from_tier2_decision_sha256": file_sha256(decision_path),
            "tier3_source_lodo_authorized": go,
            "tier3_authorized": go,
            "outer_target_image_access_authorized": False,
            "outer_target_label_access_authorized": False,
            "outer_target_access_authorized": False,
            "outer_target_images_used": False,
            "outer_target_labels_used": False,
        }
        authorization_path = output / "effective_authorization.json"
        _write_gate_json(output, authorization_path, authorization)
        return {
            "decision": gate["decision"],
            "decision_path": str(decision_path),
            "authorization_path": str(authorization_path),
            "verified_only": False,
        }
    finally:
        try:
            hold = root / "outputs/phase_state/HOLD_PHASE3_TARGET_LABEL_ACCESS"
            if hold.is_file() and hold.read_text(encoding="utf-8") != "HOLD\n":
                raise RuntimeError("outer-target HOLD sentinel drifted")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


def build_argument_parser() -> argparse.ArgumentParser:
    default_output = PROJECT_ROOT / TIER2_RELATIVE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--handoff",
        default=str(default_output / DEFAULT_HANDOFF_NAME),
    )
    parser.add_argument("--output-root", default=str(default_output))
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.verify_only:
        result = verify_frozen_tier2_gate(
            output_root=args.output_root,
            handoff_path=args.handoff,
        )
    else:
        result = run_tier2_gate(
            output_root=args.output_root,
            handoff_path=args.handoff,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AlgorithmAPI",
    "HANDOFF_SCHEMA",
    "RUN_SPECS",
    "TIER2_GO_TIER3",
    "TIER2_HOLD",
    "evaluate_tier2_decision",
    "run_tier2_gate",
    "validate_handoff",
    "verify_frozen_tier2_gate",
    "verify_selected_operating_points",
]
