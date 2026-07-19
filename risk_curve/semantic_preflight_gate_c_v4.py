"""Machine-generate a deep semantic preflight report and Gate-C input seal.

For every source pseudo-target fold this tool replays the RiskCurve/RC-Direct
comparison and all three frozen baselines directly from the referenced NPZ/PT
artifacts.  It then compares the normalised, decision-relevant actions and
sufficient counts with the submitted JSON.  Only an exact replay produces a
v2 seal that can authorise the aggregate Gate C decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .aggregate_gate_c_v4 import (
    FORMAL_OUTER_DOMAIN_KEY,
    FORMAL_SOURCE_DOMAIN_KEYS,
    GATE_C_INPUT_SEAL_SCHEMA_VERSION,
    GATE_C_SEMANTIC_PREFLIGHT_SCHEMA_VERSION,
    SEMANTIC_PREFLIGHT_TOOL_VERSION,
    SEMANTIC_PREFLIGHT_VALIDATOR_FILES,
    _domain_key,
    _load_json_object_with_sha256,
    _parse_baselines,
    _parse_comparison,
    _sha256_file,
)
from .evaluate_gate_c_baselines_v4 import evaluate_gate_c_baselines
from .evaluate_source_pseudo_target_v4 import (
    evaluate_source_pseudo_target_comparison,
)


_SAFE_FOLD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _semantic_projection(
    value: Mapping[str, Any], *, excluded: frozenset[str]
) -> dict[str, Any]:
    return {
        str(key): _jsonable(item)
        for key, item in value.items()
        if key not in excluded
    }


def _semantic_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _action_surface(value: Any, *, path: str = "$") -> list[dict[str, Any]]:
    """Extract every decision action, including its persisted threshold value."""

    records: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        required = {
            "threshold_index",
            "selected_logit_threshold",
            "reject",
            "pixel_fp_count",
            "component_fp_count",
            "tp_object_count",
            "gt_object_count",
            "total_pixels",
        }
        if required.issubset(value):
            records.append(
                {
                    "path": path,
                    **{field: _jsonable(value[field]) for field in sorted(required)},
                }
            )
        for key, item in value.items():
            records.extend(_action_surface(item, path=f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            records.extend(_action_surface(item, path=f"{path}[{index}]"))
    return records


def _write_json_atomic(
    path: Path, payload: Mapping[str, Any], *, overwrite: bool
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing preflight output: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _assert_declared_file(
    payload: Mapping[str, Any], *, path_field: str, digest_field: str
) -> dict[str, str]:
    path = Path(str(payload[path_field])).expanduser().resolve()
    digest = str(payload[digest_field])
    if not path.is_file():
        raise FileNotFoundError(f"Referenced semantic-preflight input is absent: {path}")
    if _sha256_file(path) != digest:
        raise ValueError(f"Referenced {path_field} SHA-256 mismatch")
    return {"path": str(path), "sha256": digest}


def _replay_fold(
    *,
    fold_id: str,
    comparison_path: Path,
    baseline_path: Path,
    temporary_root: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    original_comparison_payload, comparison_sha256 = (
        _load_json_object_with_sha256(
            comparison_path, kind=f"{fold_id} submitted comparison"
        )
    )
    original_baseline_payload, baseline_sha256 = _load_json_object_with_sha256(
        baseline_path, kind=f"{fold_id} submitted baselines"
    )
    original_comparison = _parse_comparison(
        original_comparison_payload,
        owner=f"preflight.{fold_id}.submitted_comparison",
        file_path=comparison_path,
        file_sha256=comparison_sha256,
    )
    original_baselines = _parse_baselines(
        original_baseline_payload,
        owner=f"preflight.{fold_id}.submitted_baselines",
        file_path=baseline_path,
        file_sha256=baseline_sha256,
        comparison=original_comparison,
    )

    if _SAFE_FOLD_ID.fullmatch(fold_id) is None:
        raise ValueError(f"Unsafe semantic-preflight fold ID: {fold_id!r}")
    resolved_root = temporary_root.resolve()
    fold_temp = (resolved_root / fold_id).resolve()
    if fold_temp.parent != resolved_root:
        raise ValueError(f"Semantic-preflight fold path escapes its temporary root")
    fold_temp.mkdir(parents=True, exist_ok=False)
    replay_comparison_path = fold_temp / "comparison.json"
    replay_baseline_path = fold_temp / "baselines.json"
    pixel_budgets = [pair[0] for pair in original_comparison["budgets"]]
    component_budgets = [pair[1] for pair in original_comparison["budgets"]]
    evaluate_source_pseudo_target_comparison(
        episode_file=original_comparison["episode_archive"],
        risk_curve_checkpoint=original_comparison["risk_curve_checkpoint"],
        rc_direct_checkpoint=original_comparison["rc_direct_checkpoint"],
        output=replay_comparison_path,
        pixel_budgets=pixel_budgets,
        component_budgets=component_budgets,
        device=device,
        batch_size=batch_size,
    )
    evaluate_gate_c_baselines(
        train_file=original_baselines["train_archive"],
        validation_file=original_baselines["validation_archive"],
        output=replay_baseline_path,
        pixel_budgets=pixel_budgets,
        component_budgets=component_budgets,
    )

    replay_comparison_payload, replay_comparison_sha256 = (
        _load_json_object_with_sha256(
            replay_comparison_path, kind=f"{fold_id} replayed comparison"
        )
    )
    replay_baseline_payload, replay_baseline_sha256 = _load_json_object_with_sha256(
        replay_baseline_path, kind=f"{fold_id} replayed baselines"
    )
    replay_comparison = _parse_comparison(
        replay_comparison_payload,
        owner=f"preflight.{fold_id}.replayed_comparison",
        file_path=replay_comparison_path,
        file_sha256=replay_comparison_sha256,
    )
    replay_baselines = _parse_baselines(
        replay_baseline_payload,
        owner=f"preflight.{fold_id}.replayed_baselines",
        file_path=replay_baseline_path,
        file_sha256=replay_baseline_sha256,
        comparison=replay_comparison,
    )

    comparison_projection = _semantic_projection(
        original_comparison,
        excluded=frozenset(("file", "file_sha256", "single_fold_decision")),
    )
    replay_comparison_projection = _semantic_projection(
        replay_comparison,
        excluded=frozenset(("file", "file_sha256", "single_fold_decision")),
    )
    comparison_projection["submitted_action_surface"] = _action_surface(
        original_comparison_payload
    )
    replay_comparison_projection["submitted_action_surface"] = _action_surface(
        replay_comparison_payload
    )
    if comparison_projection != replay_comparison_projection:
        raise ValueError(
            f"{fold_id} submitted comparison decision evidence was not reproduced"
        )
    baseline_projection = _semantic_projection(
        original_baselines, excluded=frozenset(("file", "file_sha256"))
    )
    replay_baseline_projection = _semantic_projection(
        replay_baselines, excluded=frozenset(("file", "file_sha256"))
    )
    baseline_projection["submitted_action_surface"] = _action_surface(
        original_baseline_payload
    )
    replay_baseline_projection["submitted_action_surface"] = _action_surface(
        replay_baseline_payload
    )
    if baseline_projection != replay_baseline_projection:
        raise ValueError(
            f"{fold_id} submitted baseline decision evidence was not reproduced"
        )
    if _sha256_file(comparison_path) != comparison_sha256:
        raise ValueError(f"{fold_id} comparison changed during semantic replay")
    if _sha256_file(baseline_path) != baseline_sha256:
        raise ValueError(f"{fold_id} baselines changed during semantic replay")

    referenced_artifacts = {
        "validation_episode_archive": _assert_declared_file(
            original_comparison_payload,
            path_field="episode_archive",
            digest_field="episode_archive_sha256",
        ),
        "risk_curve_checkpoint": _assert_declared_file(
            original_comparison_payload,
            path_field="risk_curve_checkpoint",
            digest_field="risk_curve_checkpoint_sha256",
        ),
        "rc_direct_checkpoint": _assert_declared_file(
            original_comparison_payload,
            path_field="rc_direct_checkpoint",
            digest_field="rc_direct_checkpoint_sha256",
        ),
        "train_episode_archive": _assert_declared_file(
            original_baseline_payload,
            path_field="train_archive",
            digest_field="train_archive_sha256",
        ),
    }
    return {
        "comparison": str(comparison_path),
        "comparison_sha256": comparison_sha256,
        "baselines": str(baseline_path),
        "baselines_sha256": baseline_sha256,
        "validation_pseudo_target": original_comparison["validation_target"],
        "excluded_outer_target": original_baselines["outer_target"],
        "num_episodes": len(original_comparison["identities"]),
        "reproduced_comparison_semantic_sha256": _semantic_sha256(
            comparison_projection
        ),
        "reproduced_baselines_semantic_sha256": _semantic_sha256(
            baseline_projection
        ),
        "referenced_artifacts": referenced_artifacts,
        "comparison_replay_exact": True,
        "baselines_replay_exact": True,
    }


def build_gate_c_semantic_preflight(
    *,
    folds: Sequence[tuple[str, str | Path, str | Path]],
    report_output: str | Path,
    seal_output: str | Path,
    device: str = "cpu",
    batch_size: int = 64,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    if len(folds) != 2:
        raise ValueError("Gate C semantic preflight requires exactly two folds")
    fold_ids = [str(item[0]) for item in folds]
    if any(not value for value in fold_ids) or len(set(fold_ids)) != len(fold_ids):
        raise ValueError("Semantic-preflight fold IDs must be non-empty and unique")
    if any(_SAFE_FOLD_ID.fullmatch(value) is None for value in fold_ids):
        raise ValueError("Semantic-preflight fold IDs contain unsafe path characters")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    report_path = Path(report_output).expanduser().resolve()
    seal_path = Path(seal_output).expanduser().resolve()
    if report_path == seal_path:
        raise ValueError("report_output and seal_output must be different files")
    if not overwrite and (report_path.exists() or seal_path.exists()):
        raise FileExistsError("Refusing to overwrite an existing report or seal")

    repository_root = Path(__file__).resolve().parents[1]
    code_hashes = {
        relative_path: _sha256_file(repository_root / relative_path)
        for relative_path in SEMANTIC_PREFLIGHT_VALIDATOR_FILES
    }
    fold_results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="rc-v4-gate-c-preflight-") as temporary:
        temporary_root = Path(temporary)
        for fold_id, comparison, baselines in sorted(
            folds, key=lambda item: str(item[0])
        ):
            fold_results[str(fold_id)] = _replay_fold(
                fold_id=str(fold_id),
                comparison_path=Path(comparison).expanduser().resolve(),
                baseline_path=Path(baselines).expanduser().resolve(),
                temporary_root=temporary_root,
                device=device,
                batch_size=batch_size,
            )

    source_keys = {
        _domain_key(row["validation_pseudo_target"])
        for row in fold_results.values()
    }
    if source_keys != FORMAL_SOURCE_DOMAIN_KEYS:
        raise ValueError(
            "Semantic preflight source folds must be exactly IRSTD-1K and NUDT-SIRST"
        )
    outer_keys = {
        _domain_key(row["excluded_outer_target"]) for row in fold_results.values()
    }
    if outer_keys != {FORMAL_OUTER_DOMAIN_KEY}:
        raise ValueError("Semantic preflight excluded outer must be NUAA-SIRST")

    report = {
        "schema_version": GATE_C_SEMANTIC_PREFLIGHT_SCHEMA_VERSION,
        "tool_version": SEMANTIC_PREFLIGHT_TOOL_VERSION,
        "status": "PASS",
        "deep_archive_checkpoint_revalidation_complete": True,
        "submitted_decision_evidence_exactly_reproduced": True,
        "outer_target_labels_used": False,
        "formal_source_domains": ["IRSTD-1K", "NUDT-SIRST"],
        "excluded_outer_target": "NUAA-SIRST",
        "replay_device": str(device),
        "replay_batch_size": int(batch_size),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "validator_code_sha256s": code_hashes,
        "folds": fold_results,
    }
    _write_json_atomic(report_path, report, overwrite=overwrite)
    report_sha256 = _sha256_file(report_path)
    seal = {
        "schema_version": GATE_C_INPUT_SEAL_SCHEMA_VERSION,
        "upstream_semantic_validation_complete": True,
        "outer_target_labels_used": False,
        "semantic_preflight_report": str(report_path),
        "semantic_preflight_report_sha256": report_sha256,
        "folds": {
            fold_id: {
                "comparison_sha256": row["comparison_sha256"],
                "baselines_sha256": row["baselines_sha256"],
            }
            for fold_id, row in fold_results.items()
        },
    }
    _write_json_atomic(seal_path, seal, overwrite=overwrite)
    return report_path, seal_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold",
        nargs=3,
        action="append",
        required=True,
        metavar=("FOLD_ID", "COMPARISON_JSON", "BASELINES_JSON"),
    )
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--seal-output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report, seal = build_gate_c_semantic_preflight(
        folds=args.fold,
        report_output=args.report_output,
        seal_output=args.seal_output,
        device=args.device,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "report": str(report),
                "report_sha256": _sha256_file(report),
                "seal": str(seal),
                "seal_sha256": _sha256_file(seal),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "SEMANTIC_PREFLIGHT_TOOL_VERSION",
    "build_gate_c_semantic_preflight",
    "build_parser",
]
