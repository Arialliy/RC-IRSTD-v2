"""Strict two-fold source-only Gate C aggregation for RC-IRSTD v4.

The per-fold comparison writer intentionally retains a local diagnostic gate.
That diagnostic is *not* authorised to open the outer-domain pilot.  This
module is the sole cross-fold Gate C decision: it joins the two held-out
pseudo-target folds, reconstructs every metric from per-episode sufficient
counts, and fails closed on provenance, label-boundary, matrix, or identity
inconsistencies.

No outer-target data are loaded here. Referenced source-only archives and
checkpoints are replayed on CPU inside the aggregate call, then all decision
metrics are reconstructed from immutable source pseudo-target JSON snapshots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .direct_calibrator import (
    ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL,
    validate_detector_role_contract,
)
from .evaluate_gate_c_baselines_v4 import GATE_C_BASELINES_SCHEMA_VERSION
from .evaluate_source_pseudo_target_v4 import (
    FORMAL_OUTER_DOMAIN_KEY,
    FORMAL_SOURCE_DOMAIN_KEYS,
    SOURCE_PSEUDO_TARGET_COMPARISON_SCHEMA_VERSION,
)
from .representation import LOGIT_GRID_SCHEMA_VERSION, LOGIT_REPRESENTATION


AGGREGATE_GATE_C_SCHEMA_VERSION = "rc-v4-gate-c-aggregate-decision-v2"
AGGREGATE_GATE_C_POLICY_VERSION = "rc-v4-gate-c-strict-cross-fold-v2"
GATE_C_INPUT_SEAL_SCHEMA_VERSION = "rc-v4-gate-c-input-seal-v2"
GATE_C_SEMANTIC_PREFLIGHT_SCHEMA_VERSION = (
    "rc-v4-gate-c-semantic-preflight-report-v1"
)
_VALIDATOR_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_PREFLIGHT_VALIDATOR_FILES = tuple(
    sorted(
        str(path.relative_to(_VALIDATOR_REPOSITORY_ROOT))
        for package in ("risk_curve", "rc_irstd", "evaluation", "certification")
        for path in (_VALIDATOR_REPOSITORY_ROOT / package).rglob("*.py")
        if path.is_file()
    )
)
COMPARISON_METHODS = ("risk_curve", "rc_direct")
BASELINE_METHODS = ("source_static", "source_worst", "count_all")
REQUIRED_METHODS = COMPARISON_METHODS + BASELINE_METHODS
REQUIRED_FOLD_COUNT = 2
FORMAL_BUDGETS = ((1e-5, 5.0), (1e-6, 1.0))
SEMANTIC_PREFLIGHT_TOOL_VERSION = "rc-v4-gate-c-semantic-preflight-tool-v2-runtime-replay"
RELATIVE_BENEFIT_THRESHOLD = 0.20
STRICT_PD_GAIN_THRESHOLD = 0.03
PD_NON_DEGRADATION_FLOOR = -0.02
REJECT_RATE_LIMIT = 0.20
NUMERIC_ATOL = 1e-12
FORMAL_BUDGET_REL_TOL = 1e-7

# A GO requires a byte-bound report produced by the semantic preflight tool.
# The report reruns both evaluators from the referenced NPZ/PT artifacts and
# compares their normalised decision evidence with the submitted JSON.  This
# aggregator then verifies the report, its validator-code hashes, and all input
# byte hashes.  A missing report/seal still produces a diagnostic HOLD artifact.
DEEP_ARCHIVE_CHECKPOINT_REVALIDATION_IMPLEMENTED = True


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _load_json_object_with_sha256(
    path: Path, *, kind: str
) -> tuple[dict[str, Any], str]:
    """Load and hash one immutable byte snapshot of a JSON input.

    The digest is computed from the exact bytes passed to ``json.loads``.  This
    prevents a preflight hash from being checked against one file generation
    while the decision logic parses a later replacement at the same path.
    """

    if not path.is_file():
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        payload = json.loads(
            text,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{kind} is invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{kind} must contain a JSON object: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _load_json_object(path: Path, *, kind: str) -> dict[str, Any]:
    payload, _digest = _load_json_object_with_sha256(path, kind=kind)
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _nonnegative_integer(value: Any, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return int(value)


def _require_bool(value: Any, field: str, expected: bool | None = None) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    if expected is not None and value is not expected:
        raise ValueError(f"{field} must be {expected}")
    return value


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_sha256(value: Any, field: str) -> str:
    text = _require_text(value, field).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _normalise_json_value(value: Any, field: str) -> Any:
    """Return a canonical finite JSON value for semantic replay comparison."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _finite_float(value, field)
    if isinstance(value, list):
        return [
            _normalise_json_value(item, f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        return {
            _require_text(key, f"{field}.key"): _normalise_json_value(
                item, f"{field}.{key}"
            )
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    raise ValueError(f"{field} contains a non-JSON value")


def _normalise_selection(
    raw: Any,
    *,
    field: str,
    grid_size: int,
    require_degenerate_k1: bool = False,
) -> dict[str, Any]:
    selection = _require_mapping(raw, field)
    found = _require_bool(selection.get("found"), f"{field}.found")
    reject = _require_bool(selection.get("reject"), f"{field}.reject")
    index_raw = selection.get("threshold_index")
    if reject:
        if found or index_raw is not None:
            raise ValueError(f"{field} reject/found/index contract is inconsistent")
        index: int | None = None
    else:
        if not found:
            raise ValueError(f"{field} finite selection must be found")
        index = _nonnegative_integer(index_raw, f"{field}.threshold_index")
        if index >= grid_size:
            raise ValueError(f"{field}.threshold_index is outside the finite grid")
    if require_degenerate_k1:
        _require_bool(selection.get("degenerate_k1"), f"{field}.degenerate_k1", True)
    normalised = _normalise_json_value(selection, field)
    normalised["found"] = found
    normalised["reject"] = reject
    normalised["threshold_index"] = index
    return normalised


def _assert_close(recorded: Any, expected: float, field: str) -> None:
    value = _finite_float(recorded, field)
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=NUMERIC_ATOL):
        raise ValueError(f"{field} does not match reconstructed sufficient counts")


def _domain_key(value: Any) -> str:
    text = _require_text(value, "pseudo-target domain")
    key = "".join(character for character in text.casefold() if character.isalnum())
    if key.endswith("sirst"):
        key = key[: -len("sirst")]
    if not key:
        raise ValueError("Pseudo-target domain normalises to an empty key")
    return key


def _verify_referenced_file(path_value: Any, digest_value: Any, field: str) -> Path:
    path = Path(_require_text(path_value, f"{field}.path")).expanduser().resolve()
    expected = _require_sha256(digest_value, f"{field}.sha256")
    if not path.is_file():
        raise FileNotFoundError(f"{field} referenced file does not exist: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(f"{field} referenced file SHA-256 mismatch")
    return path


def _budget_pairs(payload: Mapping[str, Any], *, owner: str) -> list[tuple[float, float]]:
    rows = _require_list(payload.get("budgets"), f"{owner}.budgets")
    if len(rows) != 2:
        raise ValueError(f"{owner} must contain exactly two registered budget pairs")
    pairs: list[tuple[float, float]] = []
    for position, raw in enumerate(rows):
        row = _require_mapping(raw, f"{owner}.budgets[{position}]")
        if _nonnegative_integer(
            row.get("budget_position"), f"{owner}.budgets[{position}].budget_position"
        ) != position:
            raise ValueError(f"{owner} budget positions must be contiguous and ordered")
        pixel = _finite_float(
            row.get("pixel_budget"), f"{owner}.budgets[{position}].pixel_budget"
        )
        component = _finite_float(
            row.get("component_budget"),
            f"{owner}.budgets[{position}].component_budget",
        )
        if pixel <= 0.0 or component <= 0.0:
            raise ValueError(f"{owner} budgets must be positive")
        pairs.append((pixel, component))
    if not (
        pairs[1][0] <= pairs[0][0]
        and pairs[1][1] <= pairs[0][1]
        and pairs[1] != pairs[0]
    ):
        raise ValueError(f"{owner} budget pairs must be ordered loose-to-strict")
    for position, (observed, registered) in enumerate(zip(pairs, FORMAL_BUDGETS)):
        if not all(
            math.isclose(
                value,
                expected,
                rel_tol=FORMAL_BUDGET_REL_TOL,
                abs_tol=1e-15,
            )
            for value, expected in zip(observed, registered)
        ):
            raise ValueError(
                f"{owner}.budgets[{position}] is not the registered formal "
                f"Gate C budget pair {registered!r}"
            )
    return pairs


def _validate_input_seal(
    raw: Mapping[str, Any] | None,
    *,
    fold_hashes: Mapping[str, tuple[str, str]],
) -> tuple[bool, dict[str, Any] | None]:
    """Verify caller-presealed JSON hashes required for an authoritative GO.

    The v2 seal must bind a machine-generated semantic preflight report.  That
    report reruns the comparison and baseline evaluators from their NPZ/PT
    inputs; a caller-supplied boolean alone can no longer authorise GO.
    """

    if raw is None:
        return False, None
    seal = _require_mapping(raw, "input_seal")
    if seal.get("schema_version") != GATE_C_INPUT_SEAL_SCHEMA_VERSION:
        raise ValueError("input_seal schema mismatch")
    _require_bool(
        seal.get("upstream_semantic_validation_complete"),
        "input_seal.upstream_semantic_validation_complete",
        True,
    )
    _require_bool(
        seal.get("outer_target_labels_used"),
        "input_seal.outer_target_labels_used",
        False,
    )
    report_path = Path(
        _require_text(
            seal.get("semantic_preflight_report"),
            "input_seal.semantic_preflight_report",
        )
    ).expanduser().resolve()
    expected_report_sha256 = _require_sha256(
        seal.get("semantic_preflight_report_sha256"),
        "input_seal.semantic_preflight_report_sha256",
    )
    report, actual_report_sha256 = _load_json_object_with_sha256(
        report_path, kind="Gate C semantic preflight report"
    )
    if actual_report_sha256 != expected_report_sha256:
        raise ValueError("input_seal semantic preflight report SHA-256 mismatch")
    if report.get("schema_version") != GATE_C_SEMANTIC_PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("Gate C semantic preflight report schema mismatch")
    if report.get("tool_version") != SEMANTIC_PREFLIGHT_TOOL_VERSION:
        raise ValueError("Gate C semantic preflight tool version mismatch")
    if report.get("status") != "PASS":
        raise ValueError("Gate C semantic preflight report did not pass")
    _require_bool(
        report.get("deep_archive_checkpoint_revalidation_complete"),
        "semantic_preflight.deep_archive_checkpoint_revalidation_complete",
        True,
    )
    _require_bool(
        report.get("submitted_decision_evidence_exactly_reproduced"),
        "semantic_preflight.submitted_decision_evidence_exactly_reproduced",
        True,
    )
    _require_bool(
        report.get("outer_target_labels_used"),
        "semantic_preflight.outer_target_labels_used",
        False,
    )
    report_sources = _require_list(
        report.get("formal_source_domains"),
        "semantic_preflight.formal_source_domains",
    )
    if {_domain_key(value) for value in report_sources} != FORMAL_SOURCE_DOMAIN_KEYS:
        raise ValueError("semantic preflight source-domain scope mismatch")
    if _domain_key(report.get("excluded_outer_target")) != FORMAL_OUTER_DOMAIN_KEY:
        raise ValueError("semantic preflight outer-domain scope mismatch")
    code_hashes = _require_mapping(
        report.get("validator_code_sha256s"),
        "semantic_preflight.validator_code_sha256s",
    )
    if set(code_hashes) != set(SEMANTIC_PREFLIGHT_VALIDATOR_FILES):
        raise ValueError("semantic preflight validator-code file set mismatch")
    repository_root = Path(__file__).resolve().parents[1]
    normalised_code_hashes: dict[str, str] = {}
    for relative_path in SEMANTIC_PREFLIGHT_VALIDATOR_FILES:
        recorded = _require_sha256(
            code_hashes[relative_path],
            f"semantic_preflight.validator_code_sha256s.{relative_path}",
        )
        actual = _sha256_file(repository_root / relative_path)
        if actual != recorded:
            raise ValueError(
                f"semantic preflight validator code changed: {relative_path}"
            )
        normalised_code_hashes[relative_path] = recorded
    sealed_folds = _require_mapping(seal.get("folds"), "input_seal.folds")
    if set(sealed_folds) != set(fold_hashes):
        raise ValueError("input_seal fold matrix does not match requested folds")
    report_folds = _require_mapping(
        report.get("folds"), "semantic_preflight.folds"
    )
    if set(report_folds) != set(fold_hashes):
        raise ValueError("semantic preflight fold matrix mismatch")
    normalised_folds: dict[str, Any] = {}
    for fold_id in sorted(fold_hashes):
        actual_comparison, actual_baselines = fold_hashes[fold_id]
        row = _require_mapping(
            sealed_folds[fold_id], f"input_seal.folds.{fold_id}"
        )
        expected_comparison = _require_sha256(
            row.get("comparison_sha256"),
            f"input_seal.folds.{fold_id}.comparison_sha256",
        )
        expected_baselines = _require_sha256(
            row.get("baselines_sha256"),
            f"input_seal.folds.{fold_id}.baselines_sha256",
        )
        if actual_comparison != expected_comparison:
            raise ValueError(f"input_seal comparison SHA-256 mismatch for {fold_id}")
        if actual_baselines != expected_baselines:
            raise ValueError(f"input_seal baselines SHA-256 mismatch for {fold_id}")
        report_row = _require_mapping(
            report_folds[fold_id], f"semantic_preflight.folds.{fold_id}"
        )
        _require_bool(
            report_row.get("comparison_replay_exact"),
            f"semantic_preflight.folds.{fold_id}.comparison_replay_exact",
            True,
        )
        _require_bool(
            report_row.get("baselines_replay_exact"),
            f"semantic_preflight.folds.{fold_id}.baselines_replay_exact",
            True,
        )
        if _require_sha256(
            report_row.get("comparison_sha256"),
            f"semantic_preflight.folds.{fold_id}.comparison_sha256",
        ) != expected_comparison:
            raise ValueError(f"semantic preflight comparison binding mismatch for {fold_id}")
        if _require_sha256(
            report_row.get("baselines_sha256"),
            f"semantic_preflight.folds.{fold_id}.baselines_sha256",
        ) != expected_baselines:
            raise ValueError(f"semantic preflight baseline binding mismatch for {fold_id}")
        comparison_semantic_sha256 = _require_sha256(
            report_row.get("reproduced_comparison_semantic_sha256"),
            f"semantic_preflight.folds.{fold_id}.reproduced_comparison_semantic_sha256",
        )
        baselines_semantic_sha256 = _require_sha256(
            report_row.get("reproduced_baselines_semantic_sha256"),
            f"semantic_preflight.folds.{fold_id}.reproduced_baselines_semantic_sha256",
        )
        referenced = _require_mapping(
            report_row.get("referenced_artifacts"),
            f"semantic_preflight.folds.{fold_id}.referenced_artifacts",
        )
        required_references = {
            "validation_episode_archive",
            "risk_curve_checkpoint",
            "rc_direct_checkpoint",
            "train_episode_archive",
        }
        if set(referenced) != required_references:
            raise ValueError(
                f"semantic preflight referenced-artifact matrix mismatch for {fold_id}"
            )
        normalised_references: dict[str, dict[str, str]] = {}
        for reference_name in sorted(required_references):
            reference = _require_mapping(
                referenced[reference_name],
                f"semantic_preflight.folds.{fold_id}.referenced_artifacts.{reference_name}",
            )
            reference_path = _verify_referenced_file(
                reference.get("path"),
                reference.get("sha256"),
                (
                    f"semantic_preflight.folds.{fold_id}.referenced_artifacts."
                    f"{reference_name}"
                ),
            )
            normalised_references[reference_name] = {
                "path": str(reference_path),
                "sha256": _require_sha256(
                    reference.get("sha256"),
                    (
                        f"semantic_preflight.folds.{fold_id}.referenced_artifacts."
                        f"{reference_name}.sha256"
                    ),
                ),
            }
        normalised_folds[fold_id] = {
            "comparison_sha256": expected_comparison,
            "baselines_sha256": expected_baselines,
            "reproduced_comparison_semantic_sha256": comparison_semantic_sha256,
            "reproduced_baselines_semantic_sha256": baselines_semantic_sha256,
            "referenced_artifacts": normalised_references,
        }
    return True, {
        "schema_version": GATE_C_INPUT_SEAL_SCHEMA_VERSION,
        "upstream_semantic_validation_complete": True,
        "outer_target_labels_used": False,
        "semantic_preflight_report": str(report_path),
        "semantic_preflight_report_sha256": expected_report_sha256,
        "validator_code_sha256s": normalised_code_hashes,
        "folds": normalised_folds,
    }


def _runtime_replay_sealed_inputs(
    *,
    fold_files: Mapping[str, tuple[Path, Path]],
    normalised_input_seal: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-execute both evaluators inside the aggregate decision process.

    A JSON report is evidence, not an authority boundary: without this replay a
    caller could manufacture a syntactically valid PASS report and hashes.  The
    local import avoids a module cycle because the preflight helper itself uses
    this module's strict parsers.
    """

    from .semantic_preflight_gate_c_v4 import _replay_fold

    sealed_folds = _require_mapping(
        normalised_input_seal.get("folds"), "input_seal.folds"
    )
    runtime: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="rc-v4-gate-c-runtime-replay-") as root:
        temporary_root = Path(root)
        for fold_id in sorted(fold_files):
            comparison_path, baseline_path = fold_files[fold_id]
            replay = _replay_fold(
                fold_id=fold_id,
                comparison_path=comparison_path,
                baseline_path=baseline_path,
                temporary_root=temporary_root,
                device="cpu",
                batch_size=64,
            )
            sealed = _require_mapping(
                sealed_folds[fold_id], f"input_seal.folds.{fold_id}"
            )
            for field in (
                "comparison_sha256",
                "baselines_sha256",
                "reproduced_comparison_semantic_sha256",
                "reproduced_baselines_semantic_sha256",
            ):
                if replay[field] != sealed[field]:
                    raise ValueError(
                        f"Runtime semantic replay disagrees with the sealed {field} "
                        f"for {fold_id}"
                    )
            if replay.get("referenced_artifacts") != sealed.get(
                "referenced_artifacts"
            ):
                raise ValueError(
                    f"Runtime semantic replay referenced-artifact binding "
                    f"disagrees for {fold_id}"
                )
            if not replay.get("comparison_replay_exact") or not replay.get(
                "baselines_replay_exact"
            ):
                raise ValueError(f"Runtime evaluator replay was not exact for {fold_id}")
            if _domain_key(replay["validation_pseudo_target"]) not in (
                FORMAL_SOURCE_DOMAIN_KEYS
            ):
                raise ValueError(f"Runtime replay source-domain scope mismatch for {fold_id}")
            if _domain_key(replay["excluded_outer_target"]) != FORMAL_OUTER_DOMAIN_KEY:
                raise ValueError(f"Runtime replay outer-domain scope mismatch for {fold_id}")
            runtime[fold_id] = {
                "comparison_sha256": replay["comparison_sha256"],
                "baselines_sha256": replay["baselines_sha256"],
                "reproduced_comparison_semantic_sha256": replay[
                    "reproduced_comparison_semantic_sha256"
                ],
                "reproduced_baselines_semantic_sha256": replay[
                    "reproduced_baselines_semantic_sha256"
                ],
                "comparison_replay_exact": True,
                "baselines_replay_exact": True,
            }
    return {
        "performed_inside_aggregate": True,
        "device": "cpu",
        "batch_size": 64,
        "folds": runtime,
    }


def _episode_identity(
    record: Mapping[str, Any],
    field: str,
    *,
    require_adaptation: bool,
) -> tuple[str, tuple[str, ...] | None, tuple[str, ...]]:
    pseudo_target = _require_text(record.get("pseudo_target"), f"{field}.pseudo_target")
    raw_adaptation = record.get("adaptation_ids")
    adaptation: tuple[str, ...] | None = None
    if raw_adaptation is not None:
        adaptation_values = _require_list(
            raw_adaptation, f"{field}.adaptation_ids"
        )
        if not adaptation_values:
            raise ValueError(f"{field}.adaptation_ids must not be empty")
        adaptation = tuple(
            _require_text(value, f"{field}.adaptation_ids")
            for value in adaptation_values
        )
        if len(set(adaptation)) != len(adaptation):
            raise ValueError(f"{field}.adaptation_ids contains duplicates")
    elif require_adaptation:
        raise ValueError(f"{field}.adaptation_ids is required")
    raw_ids = _require_list(record.get("evaluation_ids"), f"{field}.evaluation_ids")
    if not raw_ids:
        raise ValueError(f"{field}.evaluation_ids must not be empty")
    ids = tuple(_require_text(value, f"{field}.evaluation_ids") for value in raw_ids)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{field}.evaluation_ids contains duplicates")
    if adaptation is not None and set(adaptation) & set(ids):
        raise ValueError(f"{field} reuses identifiers between adaptation and evaluation")
    return pseudo_target, adaptation, ids


def _validate_action(
    raw: Any,
    *,
    pixel_budget: float,
    component_budget: float,
    grid_size: int,
    field: str,
) -> dict[str, Any]:
    action = _require_mapping(raw, field)
    reject = _require_bool(action.get("reject"), f"{field}.reject")
    index_raw = action.get("threshold_index")
    if reject:
        if index_raw is not None:
            raise ValueError(f"{field} rejected action must use threshold_index=null")
        index: int | None = None
    else:
        index = _nonnegative_integer(index_raw, f"{field}.threshold_index")
        if index >= grid_size:
            raise ValueError(f"{field}.threshold_index is outside the finite grid")
    pixel_fp = _nonnegative_integer(action.get("pixel_fp_count"), f"{field}.pixel_fp_count")
    component_fp = _nonnegative_integer(
        action.get("component_fp_count"), f"{field}.component_fp_count"
    )
    tp = _nonnegative_integer(action.get("tp_object_count"), f"{field}.tp_object_count")
    gt = _nonnegative_integer(action.get("gt_object_count"), f"{field}.gt_object_count")
    total_pixels = _nonnegative_integer(
        action.get("total_pixels"), f"{field}.total_pixels", positive=True
    )
    if tp > gt:
        raise ValueError(f"{field}.tp_object_count cannot exceed gt_object_count")
    if reject and any(value != 0 for value in (pixel_fp, component_fp, tp)):
        raise ValueError(f"{field} rejected action must contribute zero FP and TP counts")
    pixel_risk = pixel_fp / float(total_pixels)
    component_risk = component_fp / (total_pixels / 1_000_000.0)
    pd = tp / float(max(gt, 1))
    pixel_excess = max(pixel_risk / pixel_budget - 1.0, 0.0)
    component_excess = max(component_risk / component_budget - 1.0, 0.0)
    joint_excess = max(pixel_excess, component_excess)
    pixel_violated = pixel_risk > pixel_budget
    component_violated = component_risk > component_budget
    joint_violated = pixel_violated or component_violated
    reconstructed = {
        "threshold_index": index,
        "reject": reject,
        "pixel_fp_count": pixel_fp,
        "component_fp_count": component_fp,
        "tp_object_count": tp,
        "gt_object_count": gt,
        "total_pixels": total_pixels,
        "pd": pd,
        "pixel_risk": pixel_risk,
        "component_risk": component_risk,
        "pixel_budget_violated": pixel_violated,
        "component_budget_violated": component_violated,
        "joint_budget_violated": joint_violated,
        "pixel_relative_excess": pixel_excess,
        "component_relative_excess": component_excess,
        "joint_relative_excess": joint_excess,
    }
    for name in (
        "pd",
        "pixel_risk",
        "component_risk",
        "pixel_relative_excess",
        "component_relative_excess",
        "joint_relative_excess",
    ):
        _assert_close(action.get(name), float(reconstructed[name]), f"{field}.{name}")
    for name in (
        "pixel_budget_violated",
        "component_budget_violated",
        "joint_budget_violated",
    ):
        if _require_bool(action.get(name), f"{field}.{name}") is not reconstructed[name]:
            raise ValueError(f"{field}.{name} does not match reconstructed counts")
    return reconstructed


def _aggregate_actions(actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not actions:
        raise ValueError("Cannot aggregate an empty action sequence")
    episodes = len(actions)
    total_pixels = sum(int(action["total_pixels"]) for action in actions)
    gt = sum(int(action["gt_object_count"]) for action in actions)
    tp = sum(int(action["tp_object_count"]) for action in actions)
    pixel_fp = sum(int(action["pixel_fp_count"]) for action in actions)
    component_fp = sum(int(action["component_fp_count"]) for action in actions)
    violation_count = sum(bool(action["joint_budget_violated"]) for action in actions)
    reject_count = sum(bool(action["reject"]) for action in actions)
    excess_sum = sum(float(action["joint_relative_excess"]) for action in actions)
    finite_indices = sorted(
        {
            int(action["threshold_index"])
            for action in actions
            if not bool(action["reject"])
        }
    )
    return {
        "aggregate_counts": {
            "num_episodes": episodes,
            "total_pixels": total_pixels,
            "gt_object_count": gt,
            "tp_object_count": tp,
            "pixel_fp_count": pixel_fp,
            "component_fp_count": component_fp,
            "joint_violation_count": int(violation_count),
            "reject_count": int(reject_count),
            "sum_joint_relative_excess": excess_sum,
        },
        "num_episodes": episodes,
        "pd": tp / float(max(gt, 1)),
        "pixel_risk": pixel_fp / float(total_pixels),
        "component_risk": component_fp / (total_pixels / 1_000_000.0),
        "mean_episode_pd": sum(float(action["pd"]) for action in actions) / episodes,
        "mean_episode_pixel_risk": (
            sum(float(action["pixel_risk"]) for action in actions) / episodes
        ),
        "mean_episode_component_risk": (
            sum(float(action["component_risk"]) for action in actions) / episodes
        ),
        "joint_violation_rate": violation_count / float(episodes),
        "mean_relative_excess": excess_sum / float(episodes),
        "max_relative_excess": max(
            float(action["joint_relative_excess"]) for action in actions
        ),
        "reject_rate": reject_count / float(episodes),
        "unique_finite_indices": finite_indices,
        "num_unique_finite_indices": len(finite_indices),
    }


def _validate_recorded_aggregate(raw: Any, rebuilt: Mapping[str, Any], field: str) -> None:
    recorded = _require_mapping(raw, field)
    for name in (
        "num_episodes",
        "num_unique_finite_indices",
    ):
        if _nonnegative_integer(recorded.get(name), f"{field}.{name}") != rebuilt[name]:
            raise ValueError(f"{field}.{name} does not match per-episode records")
    for name in (
        "pd",
        "pixel_risk",
        "component_risk",
        "joint_violation_rate",
        "mean_relative_excess",
        "max_relative_excess",
        "reject_rate",
    ):
        _assert_close(recorded.get(name), float(rebuilt[name]), f"{field}.{name}")
    indices = _require_list(recorded.get("unique_finite_indices"), f"{field}.unique_finite_indices")
    if indices != rebuilt["unique_finite_indices"]:
        raise ValueError(f"{field}.unique_finite_indices does not match actions")


def _shared_contract(payload: Mapping[str, Any], *, owner: str) -> dict[str, Any]:
    list_fields = (
        "threshold_grid_detector_checkpoint_sha256s",
        "threshold_grid_episode_detector_checkpoint_sha256s",
    )
    result: dict[str, Any] = {
        "representation": _require_text(payload.get("representation"), f"{owner}.representation"),
        "threshold_grid_schema_version": _require_text(
            payload.get("threshold_grid_schema_version"),
            f"{owner}.threshold_grid_schema_version",
        ),
        "threshold_grid_size": _nonnegative_integer(
            payload.get("threshold_grid_size"), f"{owner}.threshold_grid_size", positive=True
        ),
        "threshold_grid_sha256": _require_sha256(
            payload.get("threshold_grid_sha256"), f"{owner}.threshold_grid_sha256"
        ),
        "threshold_grid_manifest_sha256": _require_sha256(
            payload.get("threshold_grid_manifest_sha256"),
            f"{owner}.threshold_grid_manifest_sha256",
        ),
        "feature_schema_sha256": _require_sha256(
            payload.get("feature_schema_sha256"), f"{owner}.feature_schema_sha256"
        ),
        "threshold_grid_detector_protocol": _require_text(
            payload.get("threshold_grid_detector_protocol"),
            f"{owner}.threshold_grid_detector_protocol",
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": _require_sha256(
            payload.get("threshold_grid_outer_detector_checkpoint_sha256"),
            f"{owner}.threshold_grid_outer_detector_checkpoint_sha256",
        ),
    }
    for field in list_fields:
        values = _require_list(payload.get(field), f"{owner}.{field}")
        if not values:
            raise ValueError(f"{owner}.{field} must not be empty")
        result[field] = [_require_sha256(value, f"{owner}.{field}") for value in values]
    if result["representation"] != LOGIT_REPRESENTATION:
        raise ValueError(f"{owner} must use {LOGIT_REPRESENTATION}")
    if result["threshold_grid_schema_version"] != LOGIT_GRID_SCHEMA_VERSION:
        raise ValueError(
            f"{owner} must use threshold grid schema {LOGIT_GRID_SCHEMA_VERSION}"
        )
    if result["threshold_grid_detector_protocol"] != (
        ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL
    ):
        raise ValueError(
            f"{owner} must use detector protocol "
            f"{ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL!r}"
        )
    all_hashes, outer_hash, episode_hashes = validate_detector_role_contract(
        result["threshold_grid_detector_checkpoint_sha256s"],
        result["threshold_grid_outer_detector_checkpoint_sha256"],
        result["threshold_grid_episode_detector_checkpoint_sha256s"],
    )
    result["threshold_grid_detector_checkpoint_sha256s"] = list(all_hashes)
    result["threshold_grid_outer_detector_checkpoint_sha256"] = outer_hash
    result["threshold_grid_episode_detector_checkpoint_sha256s"] = list(
        episode_hashes
    )
    return result


def _parse_comparison(
    payload: Mapping[str, Any],
    *,
    owner: str,
    file_path: Path,
    file_sha256: str,
) -> dict[str, Any]:
    if payload.get("schema_version") != SOURCE_PSEUDO_TARGET_COMPARISON_SCHEMA_VERSION:
        raise ValueError(f"{owner} comparison schema mismatch")
    if payload.get("protocol") != "source_only_pseudo_target_fair_comparison":
        raise ValueError(f"{owner} comparison protocol mismatch")
    _require_bool(payload.get("labels_used_for_action_selection"), f"{owner}.labels_used_for_action_selection", False)
    _require_bool(payload.get("outer_target_labels_used"), f"{owner}.outer_target_labels_used", False)
    _require_bool(
        payload.get("source_pseudo_target_labels_used_for_post_selection_evaluation"),
        f"{owner}.source_pseudo_target_labels_used_for_post_selection_evaluation",
        True,
    )
    if payload.get("archive_split") != "validation":
        raise ValueError(f"{owner} comparison archive_split must be validation")
    formal_sources = _require_list(
        payload.get("formal_source_domains"), f"{owner}.formal_source_domains"
    )
    if {_domain_key(value) for value in formal_sources} != FORMAL_SOURCE_DOMAIN_KEYS:
        raise ValueError(f"{owner} comparison source-domain scope is not canonical")
    excluded_outer_target = _require_text(
        payload.get("excluded_outer_target"), f"{owner}.excluded_outer_target"
    )
    if _domain_key(excluded_outer_target) != FORMAL_OUTER_DOMAIN_KEY:
        raise ValueError(f"{owner} comparison excluded outer target is not NUAA-SIRST")
    contract = _shared_contract(payload, owner=owner)
    budgets = _budget_pairs(payload, owner=owner)
    train_hash = _require_sha256(payload.get("train_archive_sha256"), f"{owner}.train_archive_sha256")
    validation_hash = _require_sha256(
        payload.get("validation_archive_sha256"), f"{owner}.validation_archive_sha256"
    )
    if _require_sha256(payload.get("episode_archive_sha256"), f"{owner}.episode_archive_sha256") != validation_hash:
        raise ValueError(f"{owner} episode archive is not the bound validation archive")
    episode_path = _verify_referenced_file(
        payload.get("episode_archive"), validation_hash, f"{owner}.episode_archive"
    )
    risk_checkpoint = _verify_referenced_file(
        payload.get("risk_curve_checkpoint"),
        payload.get("risk_curve_checkpoint_sha256"),
        f"{owner}.risk_curve_checkpoint",
    )
    direct_checkpoint = _verify_referenced_file(
        payload.get("rc_direct_checkpoint"),
        payload.get("rc_direct_checkpoint_sha256"),
        f"{owner}.rc_direct_checkpoint",
    )
    pseudo_targets = _require_list(payload.get("pseudo_targets"), f"{owner}.pseudo_targets")
    if len(pseudo_targets) != 1:
        raise ValueError(f"{owner} must hold out exactly one pseudo-target")
    validation_target = _require_text(pseudo_targets[0], f"{owner}.pseudo_targets[0]")
    declared_validation_target = _require_text(
        payload.get("validation_pseudo_target"),
        f"{owner}.validation_pseudo_target",
    )
    if _domain_key(declared_validation_target) != _domain_key(validation_target):
        raise ValueError(f"{owner} comparison validation-domain fields disagree")
    episodes = _require_list(payload.get("per_episode"), f"{owner}.per_episode")
    declared_episodes = _nonnegative_integer(payload.get("num_episodes"), f"{owner}.num_episodes", positive=True)
    if len(episodes) != declared_episodes:
        raise ValueError(f"{owner}.num_episodes does not match per_episode")
    identities: list[tuple[str, tuple[str, ...] | None, tuple[str, ...]]] = []
    used_role_ids: set[str] = set()
    actions: dict[int, dict[str, list[dict[str, Any]]]] = {
        position: {method: [] for method in COMPARISON_METHODS}
        for position in range(len(budgets))
    }
    for row_index, raw_episode in enumerate(episodes):
        episode = _require_mapping(raw_episode, f"{owner}.per_episode[{row_index}]")
        if _nonnegative_integer(
            episode.get("episode_index"), f"{owner}.per_episode[{row_index}].episode_index"
        ) != row_index:
            raise ValueError(f"{owner} episode indices must be contiguous and ordered")
        identity = _episode_identity(
            episode,
            f"{owner}.per_episode[{row_index}]",
            require_adaptation=True,
        )
        if _domain_key(identity[0]) != _domain_key(validation_target):
            raise ValueError(f"{owner} episode pseudo-target differs from held-out target")
        role_ids = set(identity[1] or ()) | set(identity[2])
        if used_role_ids & role_ids:
            raise ValueError(f"{owner} reuses A/E identifiers across episodes")
        used_role_ids.update(role_ids)
        identities.append(identity)
        raw_budget_actions = _require_list(
            episode.get("actions"), f"{owner}.per_episode[{row_index}].actions"
        )
        if len(raw_budget_actions) != len(budgets):
            raise ValueError(f"{owner} episode action budget count mismatch")
        for position, raw_budget_action in enumerate(raw_budget_actions):
            budget_action = _require_mapping(
                raw_budget_action,
                f"{owner}.per_episode[{row_index}].actions[{position}]",
            )
            if _nonnegative_integer(
                budget_action.get("budget_position"),
                f"{owner}.per_episode[{row_index}].actions[{position}].budget_position",
            ) != position:
                raise ValueError(f"{owner} episode budget positions are not ordered")
            _assert_close(
                budget_action.get("pixel_budget"), budgets[position][0],
                f"{owner}.per_episode[{row_index}].actions[{position}].pixel_budget",
            )
            _assert_close(
                budget_action.get("component_budget"), budgets[position][1],
                f"{owner}.per_episode[{row_index}].actions[{position}].component_budget",
            )
            raw_methods = _require_mapping(
                budget_action.get("methods"),
                f"{owner}.per_episode[{row_index}].actions[{position}].methods",
            )
            if set(raw_methods) != set(COMPARISON_METHODS):
                raise ValueError(f"{owner} comparison method matrix is incomplete")
            for method in COMPARISON_METHODS:
                actions[position][method].append(
                    _validate_action(
                        raw_methods[method],
                        pixel_budget=budgets[position][0],
                        component_budget=budgets[position][1],
                        grid_size=contract["threshold_grid_size"],
                        field=(
                            f"{owner}.per_episode[{row_index}].actions[{position}]"
                            f".methods.{method}"
                        ),
                    )
                )
    if len(set(identities)) != len(identities):
        raise ValueError(f"{owner} contains duplicate evaluation episode identities")
    aggregates: dict[int, dict[str, dict[str, Any]]] = {}
    raw_budgets = _require_list(payload.get("budgets"), f"{owner}.budgets")
    for position, raw_budget in enumerate(raw_budgets):
        methods = _require_mapping(
            _require_mapping(raw_budget, f"{owner}.budgets[{position}]").get("methods"),
            f"{owner}.budgets[{position}].methods",
        )
        if set(methods) != set(COMPARISON_METHODS):
            raise ValueError(f"{owner} comparison aggregate matrix is incomplete")
        aggregates[position] = {}
        for method in COMPARISON_METHODS:
            rebuilt = _aggregate_actions(actions[position][method])
            _validate_recorded_aggregate(
                methods[method], rebuilt, f"{owner}.budgets[{position}].methods.{method}"
            )
            aggregates[position][method] = rebuilt
    return {
        "file": str(file_path.resolve()),
        "file_sha256": _require_sha256(file_sha256, f"{owner}.file_sha256"),
        "contract": contract,
        "budgets": budgets,
        "train_archive_sha256": train_hash,
        "validation_archive_sha256": validation_hash,
        "episode_archive": str(episode_path),
        "risk_curve_checkpoint": str(risk_checkpoint),
        "rc_direct_checkpoint": str(direct_checkpoint),
        "seed": _nonnegative_integer(payload.get("seed"), f"{owner}.seed"),
        "validation_target": validation_target,
        "formal_source_domains": tuple(sorted(_domain_key(value) for value in formal_sources)),
        "excluded_outer_target": excluded_outer_target,
        "identities": identities,
        "actions": actions,
        "aggregates": aggregates,
        "single_fold_decision": _require_text(
            _require_mapping(payload.get("gate"), f"{owner}.gate").get("decision"),
            f"{owner}.gate.decision",
        ),
    }


def _parse_baselines(
    payload: Mapping[str, Any],
    *,
    owner: str,
    file_path: Path,
    file_sha256: str,
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    if payload.get("schema_version") != GATE_C_BASELINES_SCHEMA_VERSION:
        raise ValueError(f"{owner} baseline schema mismatch")
    if payload.get("protocol") != "source_only_pseudo_target_gate_c_frozen_baselines":
        raise ValueError(f"{owner} baseline protocol mismatch")
    if payload.get("status") != "COMPLETE" or payload.get("complete_required_baseline_matrix_ready") is not True:
        raise ValueError(f"{owner} baseline matrix is not complete")
    contract = _shared_contract(payload, owner=owner)
    if contract != comparison["contract"]:
        raise ValueError(f"{owner} baseline/comparison shared contract mismatch")
    budgets = _budget_pairs(payload, owner=owner)
    if budgets != comparison["budgets"]:
        raise ValueError(f"{owner} baseline/comparison budget mismatch")
    train_hash = _require_sha256(payload.get("train_archive_sha256"), f"{owner}.train_archive_sha256")
    validation_hash = _require_sha256(
        payload.get("validation_archive_sha256"), f"{owner}.validation_archive_sha256"
    )
    if train_hash != comparison["train_archive_sha256"] or validation_hash != comparison["validation_archive_sha256"]:
        raise ValueError(f"{owner} baseline/comparison archive SHA-256 binding mismatch")
    train_path = _verify_referenced_file(payload.get("train_archive"), train_hash, f"{owner}.train_archive")
    validation_path = _verify_referenced_file(
        payload.get("validation_archive"), validation_hash, f"{owner}.validation_archive"
    )
    labels = _require_mapping(payload.get("labels_policy"), f"{owner}.labels_policy")
    for field, expected in {
        "validation_labels_used_for_selection": False,
        "count_all_validation_A_masks_read_for_selection": False,
        "count_all_validation_A_raw_logits_used_for_selection": True,
        "outer_target_labels_used": False,
    }.items():
        _require_bool(labels.get(field), f"{owner}.labels_policy.{field}", expected)
    count_all = _require_mapping(payload.get("count_all"), f"{owner}.count_all")
    if count_all.get("status") != "AVAILABLE_AND_EVALUATED":
        raise ValueError(f"{owner}.count_all is not available and evaluated")
    for field, expected in {
        "available": True,
        "evaluated": True,
        "formal_protocol_eligible": True,
        "adaptation_masks_read": False,
        "future_e_counts_used_for_selection": False,
    }.items():
        _require_bool(count_all.get(field), f"{owner}.count_all.{field}", expected)
    if count_all.get("selection_source") != "adaptation_window_A_label_free_counts_only":
        raise ValueError(f"{owner}.count_all selection source is not label-free A counts")
    external = _require_mapping(payload.get("external_reject_action"), f"{owner}.external_reject_action")
    if external.get("threshold") != "+inf" or external.get("threshold_index") is not None:
        raise ValueError(f"{owner} external reject action contract mismatch")
    _require_bool(external.get("inside_finite_model_grid"), f"{owner}.external_reject_action.inside_finite_model_grid", False)
    train_targets = _require_list(payload.get("train_pseudo_targets"), f"{owner}.train_pseudo_targets")
    validation_targets = _require_list(
        payload.get("validation_pseudo_targets"), f"{owner}.validation_pseudo_targets"
    )
    if len(train_targets) != 1 or len(validation_targets) != 1:
        raise ValueError(f"{owner} must expose one train and one validation pseudo-target")
    train_target = _require_text(train_targets[0], f"{owner}.train_pseudo_targets[0]")
    validation_target = _require_text(
        validation_targets[0], f"{owner}.validation_pseudo_targets[0]"
    )
    if _domain_key(validation_target) != _domain_key(comparison["validation_target"]):
        raise ValueError(f"{owner} baseline validation target differs from comparison")
    outer_target = _require_text(payload.get("excluded_outer_target"), f"{owner}.excluded_outer_target")
    if _domain_key(outer_target) in {
        _domain_key(train_target),
        _domain_key(validation_target),
    }:
        raise ValueError(f"{owner} excluded outer target overlaps a source pseudo-target")
    raw_budgets = _require_list(payload.get("budgets"), f"{owner}.budgets")
    actions: dict[int, dict[str, list[dict[str, Any]]]] = {
        position: {method: [] for method in BASELINE_METHODS}
        for position in range(len(budgets))
    }
    aggregates: dict[int, dict[str, dict[str, Any]]] = {}
    selections: dict[int, dict[str, Any]] = {}
    source_worst_k1: list[bool] = []
    for position, raw_budget in enumerate(raw_budgets):
        methods = _require_mapping(
            _require_mapping(raw_budget, f"{owner}.budgets[{position}]").get("methods"),
            f"{owner}.budgets[{position}].methods",
        )
        if set(methods) != set(BASELINE_METHODS):
            raise ValueError(f"{owner} baseline method matrix is incomplete")
        aggregates[position] = {}
        selections[position] = {}
        for method in BASELINE_METHODS:
            method_record = _require_mapping(
                methods[method], f"{owner}.budgets[{position}].methods.{method}"
            )
            evaluation = _require_mapping(
                method_record.get("validation_evaluation"),
                f"{owner}.budgets[{position}].methods.{method}.validation_evaluation",
            )
            if method == "count_all":
                if method_record.get("selection") != (
                    "episode_specific_from_label_free_A_count_curves"
                ):
                    raise ValueError(
                        f"{owner} Count-all top-level selection contract mismatch"
                    )
                selections[position][method] = []
                _require_bool(
                    evaluation.get("adaptation_masks_read"),
                    f"{owner}.budgets[{position}].methods.count_all.adaptation_masks_read",
                    False,
                )
                _require_bool(
                    evaluation.get("future_e_counts_used_for_selection"),
                    f"{owner}.budgets[{position}].methods.count_all.future_e_counts_used_for_selection",
                    False,
                )
                _require_bool(
                    evaluation.get("validation_labels_used_for_selection"),
                    f"{owner}.budgets[{position}].methods.count_all.validation_labels_used_for_selection",
                    False,
                )
            else:
                fixed_selection = _normalise_selection(
                    method_record.get("selection"),
                    field=f"{owner}.budgets[{position}].methods.{method}.selection",
                    grid_size=contract["threshold_grid_size"],
                    require_degenerate_k1=method == "source_worst",
                )
                selections[position][method] = fixed_selection
                _require_bool(
                    evaluation.get("selection_function_received_training_counts_only"),
                    f"{owner}.budgets[{position}].methods.{method}.selection_function_received_training_counts_only",
                    True,
                )
                _require_bool(
                    evaluation.get("validation_counts_used_for_selection"),
                    f"{owner}.budgets[{position}].methods.{method}.validation_counts_used_for_selection",
                    False,
                )
                _require_bool(
                    evaluation.get("validation_labels_used_for_selection"),
                    f"{owner}.budgets[{position}].methods.{method}.validation_labels_used_for_selection",
                    False,
                )
            raw_episodes = _require_list(
                evaluation.get("per_episode"),
                f"{owner}.budgets[{position}].methods.{method}.per_episode",
            )
            if len(raw_episodes) != len(comparison["identities"]):
                raise ValueError(f"{owner} baseline/comparison episode count mismatch")
            identities: list[
                tuple[str, tuple[str, ...] | None, tuple[str, ...]]
            ] = []
            for row_index, raw_episode in enumerate(raw_episodes):
                episode = _require_mapping(
                    raw_episode,
                    f"{owner}.budgets[{position}].methods.{method}.per_episode[{row_index}]",
                )
                if _nonnegative_integer(
                    episode.get("episode_index"),
                    f"{owner}.budgets[{position}].methods.{method}.per_episode[{row_index}].episode_index",
                ) != row_index:
                    raise ValueError(f"{owner} baseline episode indices are not ordered")
                identities.append(
                    _episode_identity(
                        episode,
                        f"{owner}.budgets[{position}].methods.{method}.per_episode[{row_index}]",
                        require_adaptation=method == "count_all",
                    )
                )
                action = _validate_action(
                        episode.get("action"),
                        pixel_budget=budgets[position][0],
                        component_budget=budgets[position][1],
                        grid_size=contract["threshold_grid_size"],
                        field=(
                            f"{owner}.budgets[{position}].methods.{method}"
                            f".per_episode[{row_index}].action"
                        ),
                    )
                if method == "count_all":
                    episode_selection = _normalise_selection(
                        episode.get("selection"),
                        field=(
                            f"{owner}.budgets[{position}].methods.{method}."
                            f"per_episode[{row_index}].selection"
                        ),
                        grid_size=contract["threshold_grid_size"],
                    )
                    if (
                        episode_selection["reject"] != action["reject"]
                        or episode_selection["threshold_index"]
                        != action["threshold_index"]
                    ):
                        raise ValueError(
                            f"{owner} Count-all selection/action mismatch"
                        )
                    selections[position][method].append(episode_selection)
                else:
                    fixed_selection = selections[position][method]
                    if (
                        fixed_selection["reject"] != action["reject"]
                        or fixed_selection["threshold_index"]
                        != action["threshold_index"]
                    ):
                        raise ValueError(
                            f"{owner} fixed baseline selection/action mismatch"
                        )
                actions[position][method].append(action)
            if method == "count_all":
                identities_match = identities == comparison["identities"]
            else:
                identities_match = [
                    (target, evaluation)
                    for target, _adaptation, evaluation in identities
                ] == [
                    (target, evaluation)
                    for target, _adaptation, evaluation in comparison["identities"]
                ]
            if not identities_match:
                raise ValueError(f"{owner} baseline/comparison episode identity mismatch")
            rebuilt = _aggregate_actions(actions[position][method])
            _validate_recorded_aggregate(
                evaluation.get("aggregate"),
                rebuilt,
                f"{owner}.budgets[{position}].methods.{method}.aggregate",
            )
            aggregates[position][method] = rebuilt
            if method == "source_worst":
                selection = _require_mapping(
                    method_record.get("selection"),
                    f"{owner}.budgets[{position}].methods.source_worst.selection",
                )
                source_worst_k1.append(selection.get("degenerate_k1") is True)
    return {
        "file": str(file_path.resolve()),
        "file_sha256": _require_sha256(file_sha256, f"{owner}.file_sha256"),
        "train_archive": str(train_path),
        "validation_archive": str(validation_path),
        "train_target": train_target,
        "validation_target": validation_target,
        "outer_target": outer_target,
        "actions": actions,
        "aggregates": aggregates,
        "selections": selections,
        "source_worst_k1_all_budgets": all(source_worst_k1),
    }


def _macro_aggregate(fold_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not fold_metrics:
        raise ValueError("Cannot macro-average an empty fold set")
    scalar_fields = (
        "pd",
        "pixel_risk",
        "component_risk",
        "mean_episode_pd",
        "mean_episode_pixel_risk",
        "mean_episode_component_risk",
        "joint_violation_rate",
        "mean_relative_excess",
        "reject_rate",
    )
    count = float(len(fold_metrics))
    result = {field: sum(float(row[field]) for row in fold_metrics) / count for field in scalar_fields}
    result.update(
        {
            "num_folds": len(fold_metrics),
            "fold_episode_counts": [int(row["num_episodes"]) for row in fold_metrics],
            "mean_of_fold_max_relative_excess": sum(
                float(row["max_relative_excess"]) for row in fold_metrics
            ) / count,
            "global_max_relative_excess": max(
                float(row["max_relative_excess"]) for row in fold_metrics
            ),
        }
    )
    return result


def _relation(proposed: float, baseline: float) -> dict[str, Any]:
    delta = proposed - baseline
    non_adverse = proposed <= baseline + NUMERIC_ATOL
    if baseline <= NUMERIC_ATOL:
        if proposed <= NUMERIC_ATOL:
            status = "tie_at_zero"
            reduction: float | None = 0.0
        else:
            status = "worse_from_zero"
            reduction = None
    else:
        status = "finite_relative_reduction"
        reduction = (baseline - proposed) / baseline
    return {
        "risk_curve": proposed,
        "rc_direct": baseline,
        "delta": delta,
        "relative_reduction": reduction,
        "status": status,
        "non_adverse": non_adverse,
    }


def _absolute_budget_feasible(
    metrics: Mapping[str, Any], *, pixel_budget: float, component_budget: float
) -> bool:
    return bool(
        float(metrics["pixel_risk"]) <= pixel_budget + NUMERIC_ATOL
        and float(metrics["component_risk"]) <= component_budget + NUMERIC_ATOL
    )


def _mvr(actions: Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]], method: str, grid_size: int) -> dict[str, Any]:
    rows = len(actions[0][method])
    violations = 0
    for row in range(rows):
        codes = [
            grid_size if actions[position][method][row]["reject"] else int(actions[position][method][row]["threshold_index"])
            for position in sorted(actions)
        ]
        violations += int(any(right < left for left, right in zip(codes, codes[1:])))
    return {
        "num_episodes": rows,
        "violating_episode_count": violations,
        "rate": violations / float(rows),
    }


def aggregate_gate_c(
    *,
    folds: Sequence[tuple[str, str | Path, str | Path]],
    output: str | Path,
    input_seal: Mapping[str, Any] | None = None,
) -> Path:
    """Validate and aggregate exactly two source-only pseudo-target folds."""

    if len(folds) != REQUIRED_FOLD_COUNT:
        raise ValueError("Aggregate Gate C requires exactly two folds")
    fold_ids = [str(item[0]) for item in folds]
    if any(not value for value in fold_ids) or len(set(fold_ids)) != len(fold_ids):
        raise ValueError("Aggregate Gate C fold IDs must be non-empty and unique")
    fold_files = {
        str(fold_id): (
            Path(comparison_file).expanduser().resolve(),
            Path(baseline_file).expanduser().resolve(),
        )
        for fold_id, comparison_file, baseline_file in folds
    }
    loaded_inputs: dict[str, dict[str, Any]] = {}
    for fold_id in sorted(fold_files):
        comparison_path, baseline_path = fold_files[fold_id]
        comparison_payload, comparison_sha256 = _load_json_object_with_sha256(
            comparison_path, kind=f"{fold_id} comparison"
        )
        baseline_payload, baseline_sha256 = _load_json_object_with_sha256(
            baseline_path, kind=f"{fold_id} baselines"
        )
        loaded_inputs[fold_id] = {
            "comparison_payload": comparison_payload,
            "comparison_sha256": comparison_sha256,
            "baseline_payload": baseline_payload,
            "baseline_sha256": baseline_sha256,
        }
    input_seal_binding_verified, normalised_input_seal = _validate_input_seal(
        input_seal,
        fold_hashes={
            fold_id: (
                str(values["comparison_sha256"]),
                str(values["baseline_sha256"]),
            )
            for fold_id, values in loaded_inputs.items()
        },
    )
    for fold_id, (comparison_path, baseline_path) in fold_files.items():
        snapshot = loaded_inputs[fold_id]
        if _sha256_file(comparison_path) != snapshot["comparison_sha256"]:
            raise ValueError(
                f"{fold_id} comparison changed after its immutable byte snapshot"
            )
        if _sha256_file(baseline_path) != snapshot["baseline_sha256"]:
            raise ValueError(
                f"{fold_id} baselines changed after its immutable byte snapshot"
            )
    parsed: dict[str, dict[str, Any]] = {}
    for fold_id, comparison_file, baseline_file in sorted(folds, key=lambda item: str(item[0])):
        comparison_path, baseline_path = fold_files[str(fold_id)]
        snapshot = loaded_inputs[str(fold_id)]
        comparison = _parse_comparison(
            snapshot["comparison_payload"],
            owner=f"folds.{fold_id}.comparison",
            file_path=comparison_path,
            file_sha256=str(snapshot["comparison_sha256"]),
        )
        baselines = _parse_baselines(
            snapshot["baseline_payload"],
            owner=f"folds.{fold_id}.baselines",
            file_path=baseline_path,
            file_sha256=str(snapshot["baseline_sha256"]),
            comparison=comparison,
        )
        parsed[str(fold_id)] = {"comparison": comparison, "baselines": baselines}

    ordered_ids = sorted(parsed)
    reference = parsed[ordered_ids[0]]["comparison"]
    source_targets = {
        _domain_key(parsed[fold_id]["comparison"]["validation_target"])
        for fold_id in ordered_ids
    }
    if len(source_targets) != REQUIRED_FOLD_COUNT:
        raise ValueError("Aggregate Gate C folds must hold out two distinct pseudo-targets")
    if source_targets != FORMAL_SOURCE_DOMAIN_KEYS:
        raise ValueError(
            "Aggregate Gate C source pseudo-targets must be exactly "
            "IRSTD-1K and NUDT-SIRST"
        )
    outer_targets = {
        _domain_key(parsed[fold_id]["baselines"]["outer_target"])
        for fold_id in ordered_ids
    }
    if len(outer_targets) != 1 or source_targets & outer_targets:
        raise ValueError("Aggregate Gate C outer exclusion is inconsistent with source folds")
    if outer_targets != {FORMAL_OUTER_DOMAIN_KEY}:
        raise ValueError("Aggregate Gate C excluded outer target must be NUAA-SIRST")
    comparison_outer_targets = {
        _domain_key(parsed[fold_id]["comparison"]["excluded_outer_target"])
        for fold_id in ordered_ids
    }
    if comparison_outer_targets != outer_targets:
        raise ValueError("Comparison/baseline excluded-outer scopes disagree")
    if any(
        set(parsed[fold_id]["comparison"]["formal_source_domains"])
        != FORMAL_SOURCE_DOMAIN_KEYS
        for fold_id in ordered_ids
    ):
        raise ValueError("Comparison canonical source-domain bindings disagree")
    if len({parsed[fold_id]["comparison"]["validation_archive_sha256"] for fold_id in ordered_ids}) != REQUIRED_FOLD_COUNT:
        raise ValueError("Aggregate Gate C cannot count the same validation archive twice")
    for fold_id in ordered_ids:
        comparison = parsed[fold_id]["comparison"]
        baselines = parsed[fold_id]["baselines"]
        if comparison["contract"] != reference["contract"]:
            raise ValueError("Aggregate Gate C fold shared contracts differ")
        if comparison["budgets"] != reference["budgets"]:
            raise ValueError("Aggregate Gate C fold budgets differ")
        if comparison["seed"] != reference["seed"]:
            raise ValueError("Aggregate Gate C fold seeds differ")
        if _domain_key(baselines["train_target"]) != (
            source_targets - {_domain_key(comparison["validation_target"])}
        ).pop():
            raise ValueError("Aggregate Gate C train/validation pseudo-target folds are not complementary")

    runtime_semantic_replay: dict[str, Any] | None = None
    if input_seal_binding_verified:
        if normalised_input_seal is None:
            raise AssertionError("Verified input seal was not normalised")
        runtime_semantic_replay = _runtime_replay_sealed_inputs(
            fold_files=fold_files,
            normalised_input_seal=normalised_input_seal,
        )
        normalised_input_seal["runtime_semantic_replay"] = runtime_semantic_replay
    input_seal_verified = bool(
        input_seal_binding_verified and runtime_semantic_replay is not None
    )

    budgets = reference["budgets"]
    grid_size = int(reference["contract"]["threshold_grid_size"])
    fold_actions: dict[str, dict[int, dict[str, list[dict[str, Any]]]]] = {}
    fold_aggregates: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    fold_mvr: dict[str, dict[str, dict[str, Any]]] = {}
    for fold_id in ordered_ids:
        comparison = parsed[fold_id]["comparison"]
        baselines = parsed[fold_id]["baselines"]
        fold_actions[fold_id] = {}
        fold_aggregates[fold_id] = {}
        for position in range(len(budgets)):
            fold_actions[fold_id][position] = {
                **comparison["actions"][position],
                **baselines["actions"][position],
            }
            if set(fold_actions[fold_id][position]) != set(REQUIRED_METHODS):
                raise ValueError("Aggregate Gate C five-method action matrix is incomplete")
            fold_aggregates[fold_id][position] = {
                method: _aggregate_actions(fold_actions[fold_id][position][method])
                for method in REQUIRED_METHODS
            }
        fold_mvr[fold_id] = {
            method: _mvr(fold_actions[fold_id], method, grid_size)
            for method in REQUIRED_METHODS
        }

    macro: dict[int, dict[str, dict[str, Any]]] = {}
    micro: dict[int, dict[str, dict[str, Any]]] = {}
    for position in range(len(budgets)):
        macro[position] = {
            method: _macro_aggregate(
                [fold_aggregates[fold_id][position][method] for fold_id in ordered_ids]
            )
            for method in REQUIRED_METHODS
        }
        micro[position] = {
            method: _aggregate_actions(
                [
                    action
                    for fold_id in ordered_ids
                    for action in fold_actions[fold_id][position][method]
                ]
            )
            for method in REQUIRED_METHODS
        }

    mvr_micro: dict[str, dict[str, Any]] = {}
    mvr_macro: dict[str, float] = {}
    for method in REQUIRED_METHODS:
        violations = sum(fold_mvr[fold_id][method]["violating_episode_count"] for fold_id in ordered_ids)
        episodes = sum(fold_mvr[fold_id][method]["num_episodes"] for fold_id in ordered_ids)
        mvr_micro[method] = {
            "num_episodes": episodes,
            "violating_episode_count": violations,
            "rate": violations / float(episodes),
        }
        mvr_macro[method] = sum(fold_mvr[fold_id][method]["rate"] for fold_id in ordered_ids) / len(ordered_ids)

    relations: dict[int, dict[str, Any]] = {}
    for position in range(len(budgets)):
        per_fold: dict[str, Any] = {}
        for fold_id in ordered_ids:
            risk = fold_aggregates[fold_id][position]["risk_curve"]
            direct = fold_aggregates[fold_id][position]["rc_direct"]
            per_fold[fold_id] = {
                "pd_delta": risk["pd"] - direct["pd"],
                "reject_rate_delta": risk["reject_rate"] - direct["reject_rate"],
                "joint_violation_rate": _relation(
                    risk["joint_violation_rate"], direct["joint_violation_rate"]
                ),
                "mean_relative_excess": _relation(
                    risk["mean_relative_excess"], direct["mean_relative_excess"]
                ),
                "max_relative_excess": _relation(
                    risk["max_relative_excess"], direct["max_relative_excess"]
                ),
            }
        risk_macro = macro[position]["risk_curve"]
        direct_macro = macro[position]["rc_direct"]
        risk_micro = micro[position]["risk_curve"]
        direct_micro = micro[position]["rc_direct"]
        relations[position] = {
            "budget_position": position,
            "pixel_budget": budgets[position][0],
            "component_budget": budgets[position][1],
            "per_fold": per_fold,
            "macro": {
                "pd_delta": risk_macro["pd"] - direct_macro["pd"],
                "joint_violation_rate": _relation(
                    risk_macro["joint_violation_rate"], direct_macro["joint_violation_rate"]
                ),
                "mean_relative_excess": _relation(
                    risk_macro["mean_relative_excess"], direct_macro["mean_relative_excess"]
                ),
                "max_relative_excess": _relation(
                    risk_macro["global_max_relative_excess"], direct_macro["global_max_relative_excess"]
                ),
            },
            "micro": {
                "pd_delta": risk_micro["pd"] - direct_micro["pd"],
                "joint_violation_rate": _relation(
                    risk_micro["joint_violation_rate"], direct_micro["joint_violation_rate"]
                ),
                "mean_relative_excess": _relation(
                    risk_micro["mean_relative_excess"], direct_micro["mean_relative_excess"]
                ),
                "max_relative_excess": _relation(
                    risk_micro["max_relative_excess"], direct_micro["max_relative_excess"]
                ),
            },
        }

    absolute_checks: list[dict[str, Any]] = []
    for position, (pixel_budget, component_budget) in enumerate(budgets):
        for scope, metrics in [
            *[(f"fold:{fold_id}", fold_aggregates[fold_id][position]["risk_curve"]) for fold_id in ordered_ids],
            ("macro", macro[position]["risk_curve"]),
            ("micro", micro[position]["risk_curve"]),
        ]:
            pixel_ok = float(metrics["pixel_risk"]) <= pixel_budget + NUMERIC_ATOL
            component_ok = float(metrics["component_risk"]) <= component_budget + NUMERIC_ATOL
            absolute_checks.append(
                {
                    "budget_position": position,
                    "scope": scope,
                    "pixel_risk": metrics["pixel_risk"],
                    "component_risk": metrics["component_risk"],
                    "pixel_budget_satisfied": pixel_ok,
                    "component_budget_satisfied": component_ok,
                    "joint_budget_satisfied": pixel_ok and component_ok,
                }
            )
    absolute_risk_control = all(row["joint_budget_satisfied"] for row in absolute_checks)

    non_adverse_checks: list[dict[str, Any]] = []
    for position, relation in relations.items():
        for scope_name, scope in [
            *[(f"fold:{fold_id}", relation["per_fold"][fold_id]) for fold_id in ordered_ids],
            ("macro", relation["macro"]),
            ("micro", relation["micro"]),
        ]:
            values = {
                metric: bool(scope[metric]["non_adverse"])
                for metric in (
                    "joint_violation_rate",
                    "mean_relative_excess",
                    "max_relative_excess",
                )
            }
            non_adverse_checks.append(
                {
                    "budget_position": position,
                    "scope": scope_name,
                    **values,
                    "all_risk_metrics_non_adverse": all(values.values()),
                }
            )
    risk_non_adverse = all(row["all_risk_metrics_non_adverse"] for row in non_adverse_checks)

    def stable_benefit(metric: str, position: int) -> bool:
        relation = relations[position]
        fold_reductions = [
            relation["per_fold"][fold_id][metric]["relative_reduction"]
            for fold_id in ordered_ids
        ]
        macro_reduction = relation["macro"][metric]["relative_reduction"]
        micro_reduction = relation["micro"][metric]["relative_reduction"]
        return bool(
            all(value is not None and value >= -NUMERIC_ATOL for value in fold_reductions)
            and any(value is not None and value >= RELATIVE_BENEFIT_THRESHOLD for value in fold_reductions)
            and macro_reduction is not None
            and macro_reduction >= RELATIVE_BENEFIT_THRESHOLD
            and micro_reduction is not None
            and micro_reduction >= RELATIVE_BENEFIT_THRESHOLD
        )

    violation_by_budget = [stable_benefit("joint_violation_rate", position) for position in range(len(budgets))]
    excess_by_budget = [stable_benefit("mean_relative_excess", position) for position in range(len(budgets))]
    violation_benefit = any(violation_by_budget)
    excess_benefit = any(excess_by_budget)
    literal_pd_checks: list[dict[str, Any]] = []
    constrained_pd_checks: list[dict[str, Any]] = []
    reject_checks: list[dict[str, Any]] = []
    for position, (pixel_budget, component_budget) in enumerate(budgets):
        scopes: list[tuple[str, Mapping[str, Mapping[str, Any]]]] = [
            *[
                (f"fold:{fold_id}", fold_aggregates[fold_id][position])
                for fold_id in ordered_ids
            ],
            ("macro", macro[position]),
            ("micro", micro[position]),
        ]
        for scope, method_metrics in scopes:
            risk_metrics = method_metrics["risk_curve"]
            direct_metrics = method_metrics["rc_direct"]
            risk_feasible = _absolute_budget_feasible(
                risk_metrics,
                pixel_budget=pixel_budget,
                component_budget=component_budget,
            )
            direct_feasible = _absolute_budget_feasible(
                direct_metrics,
                pixel_budget=pixel_budget,
                component_budget=component_budget,
            )
            literal_pd_delta = float(risk_metrics["pd"]) - float(
                direct_metrics["pd"]
            )
            literal_non_degraded = bool(
                literal_pd_delta >= PD_NON_DEGRADATION_FLOOR - NUMERIC_ATOL
            )
            literal_pd_checks.append(
                {
                    "budget_position": position,
                    "scope": scope,
                    "comparator_method": "rc_direct",
                    "comparator_pd": direct_metrics["pd"],
                    "risk_curve_pd": risk_metrics["pd"],
                    "pd_delta": literal_pd_delta,
                    "non_degraded": literal_non_degraded,
                    "authoritative_for_c4_strict_pd_gain": True,
                    "authoritative_for_c5": True,
                }
            )
            feasible_source_candidates = [
                method
                for method in BASELINE_METHODS
                if _absolute_budget_feasible(
                    method_metrics[method],
                    pixel_budget=pixel_budget,
                    component_budget=component_budget,
                )
            ]
            if direct_feasible:
                comparator_method: str | None = "rc_direct"
                comparator_reason = "rc_direct_is_absolutely_budget_feasible"
            elif feasible_source_candidates:
                comparator_method = max(
                    feasible_source_candidates,
                    key=lambda method: float(method_metrics[method]["pd"]),
                )
                comparator_reason = (
                    "rc_direct_is_unsafe_use_highest_pd_absolutely_feasible_source_baseline"
                )
            else:
                comparator_method = None
                comparator_reason = "rc_direct_is_unsafe_and_no_source_baseline_is_feasible"
            comparator_pd = (
                None
                if comparator_method is None
                else float(method_metrics[comparator_method]["pd"])
            )
            constrained_pd_delta = (
                None
                if comparator_pd is None
                else float(risk_metrics["pd"]) - comparator_pd
            )
            constrained_non_degraded = bool(
                constrained_pd_delta is not None
                and constrained_pd_delta
                >= PD_NON_DEGRADATION_FLOOR - NUMERIC_ATOL
            )
            constrained_pd_checks.append(
                {
                    "budget_position": position,
                    "scope": scope,
                    "risk_curve_absolute_feasible": risk_feasible,
                    "rc_direct_absolute_feasible": direct_feasible,
                    "feasible_source_baseline_candidates": [
                        {
                            "method": method,
                            "pd": method_metrics[method]["pd"],
                            "pixel_risk": method_metrics[method]["pixel_risk"],
                            "component_risk": method_metrics[method]["component_risk"],
                        }
                        for method in feasible_source_candidates
                    ],
                    "comparator_method": comparator_method,
                    "comparator_reason": comparator_reason,
                    "comparator_pd": comparator_pd,
                    "risk_curve_pd": risk_metrics["pd"],
                    "pd_delta": constrained_pd_delta,
                    "non_degraded": constrained_non_degraded,
                    "diagnostic_only": True,
                    "authoritative_for_c4_strict_pd_gain": False,
                    "authoritative_for_c5": False,
                }
            )
            reject_checks.append(
                {
                    "budget_position": position,
                    "scope": scope,
                    "reject_rate": risk_metrics["reject_rate"],
                    "acceptable": risk_metrics["reject_rate"] < REJECT_RATE_LIMIT,
                }
            )

    strict_position = len(budgets) - 1

    def strict_pd_gain(checks: Sequence[Mapping[str, Any]]) -> bool:
        strict_checks = [
            row for row in checks if row["budget_position"] == strict_position
        ]
        strict_fold_checks = [
            row for row in strict_checks if str(row["scope"]).startswith("fold:")
        ]
        strict_macro = next(row for row in strict_checks if row["scope"] == "macro")
        strict_micro = next(row for row in strict_checks if row["scope"] == "micro")
        return bool(
            strict_macro["pd_delta"] is not None
            and strict_macro["pd_delta"] >= STRICT_PD_GAIN_THRESHOLD
            and strict_micro["pd_delta"] is not None
            and strict_micro["pd_delta"] >= STRICT_PD_GAIN_THRESHOLD
            and all(row["non_degraded"] for row in strict_fold_checks)
            and any(
                row["pd_delta"] is not None
                and row["pd_delta"] >= STRICT_PD_GAIN_THRESHOLD
                for row in strict_fold_checks
            )
        )

    literal_strict_pd_gain = strict_pd_gain(literal_pd_checks)
    constrained_strict_pd_gain = strict_pd_gain(constrained_pd_checks)
    feasibility_improved_cells = [
        {
            "budget_position": row["budget_position"],
            "scope": row["scope"],
        }
        for row in constrained_pd_checks
        if str(row["scope"]).startswith("fold:")
        and row["risk_curve_absolute_feasible"]
        and not row["rc_direct_absolute_feasible"]
    ]
    feasibility_benefit = bool(
        absolute_risk_control and feasibility_improved_cells
    )
    # Conservative authoritative C4: retain the pre-existing two-success-item
    # policy, but only the two stable risk benefits and the *literal* strict-Pd
    # gain versus RC-Direct may count.  At least one stable risk benefit is
    # mandatory, so Pd-only can never pass.  Feasibility-aware comparisons are
    # useful diagnostics but are deliberately excluded from the success count.
    authoritative_success_count = sum(
        (violation_benefit, excess_benefit, literal_strict_pd_gain)
    )
    stable_risk_benefit = bool(violation_benefit or excess_benefit)
    benefit_rule = stable_risk_benefit and authoritative_success_count >= 2

    literal_pd_non_degraded = all(
        row["non_degraded"] for row in literal_pd_checks
    )
    constrained_pd_non_degraded = all(
        row["non_degraded"] for row in constrained_pd_checks
    )
    reject_acceptable = all(row["acceptable"] for row in reject_checks)
    risk_mvr_zero = bool(
        all(fold_mvr[fold_id]["risk_curve"]["rate"] == 0.0 for fold_id in ordered_ids)
        and mvr_micro["risk_curve"]["rate"] == 0.0
        and mvr_macro["risk_curve"] == 0.0
    )

    criteria = {
        "c0_integrity_and_source_boundary": input_seal_verified,
        "c1_complete_five_method_matrix": True,
        "c2_absolute_risk_budget_control": absolute_risk_control,
        "c3_risk_non_adverse_to_rc_direct": risk_non_adverse,
        "c4_stable_benefit_not_pd_only": benefit_rule,
        "c5_literal_pd_non_degraded_vs_rc_direct": literal_pd_non_degraded,
        "c6_reject_rate_below_20_percent": reject_acceptable,
        "c7_risk_curve_mvr_zero": risk_mvr_zero,
    }
    go = all(criteria.values())
    failure_reasons = [name for name, passed in criteria.items() if not passed]

    # The decision above was reconstructed from the immutable byte snapshots
    # loaded at function entry.  Also require the caller-visible paths to still
    # contain those exact bytes before publishing an artifact; this makes a
    # concurrent replacement explicit instead of leaving a stale path/hash
    # pair for downstream reproducibility checks.
    for fold_id, (comparison_path, baseline_path) in fold_files.items():
        snapshot = loaded_inputs[fold_id]
        if _sha256_file(comparison_path) != snapshot["comparison_sha256"]:
            raise ValueError(
                f"{fold_id} comparison changed after its immutable byte snapshot"
            )
        if _sha256_file(baseline_path) != snapshot["baseline_sha256"]:
            raise ValueError(
                f"{fold_id} baselines changed after its immutable byte snapshot"
            )
    if normalised_input_seal is not None:
        report_path = Path(
            str(normalised_input_seal["semantic_preflight_report"])
        )
        if _sha256_file(report_path) != normalised_input_seal[
            "semantic_preflight_report_sha256"
        ]:
            raise ValueError(
                "Gate C semantic preflight report changed during aggregation"
            )

    per_fold_output: dict[str, Any] = {}
    for fold_id in ordered_ids:
        per_fold_output[fold_id] = {
            "validation_pseudo_target": parsed[fold_id]["comparison"]["validation_target"],
            "train_pseudo_target": parsed[fold_id]["baselines"]["train_target"],
            "num_episodes": len(parsed[fold_id]["comparison"]["identities"]),
            "comparison_file": parsed[fold_id]["comparison"]["file"],
            "comparison_sha256": parsed[fold_id]["comparison"]["file_sha256"],
            "baselines_file": parsed[fold_id]["baselines"]["file"],
            "baselines_sha256": parsed[fold_id]["baselines"]["file_sha256"],
            "train_archive_sha256": parsed[fold_id]["comparison"]["train_archive_sha256"],
            "validation_archive_sha256": parsed[fold_id]["comparison"]["validation_archive_sha256"],
            "single_fold_decision": parsed[fold_id]["comparison"]["single_fold_decision"],
            "single_fold_decision_authoritative": False,
            "source_worst_k1_all_budgets": parsed[fold_id]["baselines"]["source_worst_k1_all_budgets"],
            "metrics_by_budget": {
                str(position): fold_aggregates[fold_id][position]
                for position in range(len(budgets))
            },
            "monotonicity": fold_mvr[fold_id],
        }

    payload = {
        "schema_version": AGGREGATE_GATE_C_SCHEMA_VERSION,
        "policy_version": AGGREGATE_GATE_C_POLICY_VERSION,
        "decision": "GO" if go else "HOLD",
        "scope": "source_only_pseudo_target_cross_fold_gate_c",
        "decision_authority": "aggregate_only",
        "not_an_outer_target_claim": True,
        "outer_target_labels_used": False,
        "single_fold_decisions_non_authoritative": True,
        "seed": reference["seed"],
        "representation": reference["contract"]["representation"],
        "shared_contract": reference["contract"],
        "required_methods": list(REQUIRED_METHODS),
        "required_fold_count": REQUIRED_FOLD_COUNT,
        "required_source_pseudo_targets": ["IRSTD-1K", "NUDT-SIRST"],
        "required_excluded_outer_target": "NUAA-SIRST",
        "registered_formal_budgets": [
            {
                "budget_position": position,
                "pixel_budget": pair[0],
                "component_budget": pair[1],
            }
            for position, pair in enumerate(FORMAL_BUDGETS)
        ],
        "required_budgets": [
            {
                "budget_position": position,
                "pixel_budget": pair[0],
                "component_budget": pair[1],
            }
            for position, pair in enumerate(budgets)
        ],
        "folds": per_fold_output,
        "macro_by_budget": {
            str(position): macro[position] for position in range(len(budgets))
        },
        "micro_by_budget": {
            str(position): micro[position] for position in range(len(budgets))
        },
        "monotonicity": {
            "macro_rate_by_method": mvr_macro,
            "micro_by_method": mvr_micro,
        },
        "risk_curve_vs_rc_direct": {
            "by_budget": {str(position): relations[position] for position in range(len(budgets))}
        },
        "absolute_risk_budget_checks": absolute_checks,
        "risk_non_adverse_checks": non_adverse_checks,
        "benefit_evidence": {
            "violation_stable_by_budget": violation_by_budget,
            "excess_stable_by_budget": excess_by_budget,
            "violation_benefit": violation_benefit,
            "excess_benefit": excess_benefit,
            "feasibility_benefit": feasibility_benefit,
            "feasibility_improved_fold_budget_cells": feasibility_improved_cells,
            "feasibility_benefit_diagnostic_only": True,
            "literal_strict_pd_gain_vs_rc_direct": literal_strict_pd_gain,
            "constrained_strict_pd_gain_diagnostic": constrained_strict_pd_gain,
            "authoritative_success_criterion_count": authoritative_success_count,
            "authoritative_success_items": [
                "stable_joint_violation_reduction",
                "stable_mean_relative_excess_reduction",
                "literal_strict_pd_gain_vs_rc_direct",
            ],
            "stable_violation_or_excess_benefit_mandatory": True,
            "at_least_two_success_criteria_required": True,
            "pd_only_can_never_pass": True,
        },
        "literal_pd_vs_rc_direct_checks": literal_pd_checks,
        "constrained_pd_vs_best_feasible_baseline_diagnostic": (
            constrained_pd_checks
        ),
        "authoritative_c5_comparator": "literal_rc_direct",
        "literal_pd_non_degraded_vs_rc_direct": literal_pd_non_degraded,
        "constrained_pd_non_degraded_diagnostic": constrained_pd_non_degraded,
        "constrained_pd_comparator_authoritative": False,
        "reject_rate_checks": reject_checks,
        "criteria": criteria,
        "failure_reasons": failure_reasons,
        "source_worst_evidence": {
            "k1_degenerate_in_all_folds": all(
                parsed[fold_id]["baselines"]["source_worst_k1_all_budgets"]
                for fold_id in ordered_ids
            ),
            "multi_domain_worst_case_evidence": False,
        },
        "aggregation_contract": {
            "macro": "equal_weight_mean_over_held_out_pseudo_target_folds",
            "micro": "recomputed_from_concatenated_per_episode_sufficient_counts",
            "relative_zero_baseline": "tie_at_zero_or_worse_from_zero_never_ambiguous_null",
            "authoritative_pd_non_degradation_comparator": "literal_rc_direct",
            "diagnostic_constrained_pd_comparator": (
                "rc_direct_when_absolutely_feasible_else_highest_pd_among_"
                "absolutely_feasible_source_static_source_worst_count_all"
            ),
            "c4_authoritative_rule": (
                "at_least_two_of_stable_violation_stable_excess_literal_strict_"
                "pd_gain_and_at_least_one_stable_risk_benefit_mandatory"
            ),
            "feasibility_benefit_counts_toward_c4": False,
            "numeric_atol": NUMERIC_ATOL,
        },
        "input_seal": normalised_input_seal,
        "provenance_validation": {
            "input_seal_required_for_go": True,
            "input_seal_verified": input_seal_verified,
            "upstream_semantic_validation_attested": input_seal_binding_verified,
            "machine_generated_semantic_preflight_required_for_go": True,
            "machine_generated_semantic_preflight_verified": input_seal_verified,
            "deep_archive_checkpoint_semantic_revalidation_performed": (
                input_seal_verified
                and DEEP_ARCHIVE_CHECKPOINT_REVALIDATION_IMPLEMENTED
            ),
            "limitation": (
                "The preflight report is byte-bound evidence, while authority is "
                "established by an independent CPU replay of both evaluators inside "
                "this aggregate call. GO is disabled unless the report, immutable "
                "input snapshots, and runtime replay all agree exactly."
            ),
        },
    }
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold",
        nargs=3,
        action="append",
        required=True,
        metavar=("FOLD_ID", "COMPARISON_JSON", "BASELINES_JSON"),
        help="Repeat exactly twice; fold order is canonicalised by FOLD_ID.",
    )
    parser.add_argument(
        "--input-seal",
        help=(
            "Presealed Gate C input manifest. It is mandatory for an "
            "authoritative GO; omission produces a diagnostic HOLD."
        ),
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_seal = (
        None
        if args.input_seal is None
        else _load_json_object(
            Path(args.input_seal).expanduser().resolve(),
            kind="Gate C input seal",
        )
    )
    path = aggregate_gate_c(
        folds=[(item[0], item[1], item[2]) for item in args.fold],
        output=args.output,
        input_seal=input_seal,
    )
    payload = _load_json_object(path, kind="aggregate Gate C output")
    print(json.dumps({"output": str(path), "decision": payload["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AGGREGATE_GATE_C_POLICY_VERSION",
    "AGGREGATE_GATE_C_SCHEMA_VERSION",
    "GATE_C_INPUT_SEAL_SCHEMA_VERSION",
    "REQUIRED_METHODS",
    "aggregate_gate_c",
    "build_parser",
    "main",
]
