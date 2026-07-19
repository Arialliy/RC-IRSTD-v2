"""Select source-only pooled and worst-domain operating points.

Only formal threshold curves with verified sidecars are accepted.  Optional
target labels are read after source selection and can only report performance
at the already-frozen source thresholds.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifact_integrity import file_sha256, verify_score_map_directory
from .budget_metrics import is_budget_satisfied
from .threshold_sweep import (
    CURVE_COLUMNS,
    CURVE_METADATA_ARTIFACT_TYPE,
    CURVE_METADATA_SCHEMA_VERSION,
    curve_metadata_path,
    domain_key,
    evaluation_threshold_grid_sha256,
    read_curve_csv,
    validate_formal_score_manifest,
    validate_thresholds,
    write_json_atomic,
)


SOURCE_SELECTION_SCHEMA_VERSION = 1
SOURCE_SELECTION_ARTIFACT_TYPE = "rc-irstd-source-only-operating-point"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FormalCurve:
    path: Path
    metadata_path: Path
    rows: tuple[dict[str, float | int], ...]
    metadata: dict[str, Any]
    metadata_sha256: str


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_budget(value: float, *, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return number


def _validate_curve_rows(
    rows: Sequence[Mapping[str, object]],
) -> np.ndarray:
    if not rows:
        raise ValueError("Formal threshold curve must contain at least one row")
    thresholds = validate_thresholds([float(row["threshold"]) for row in rows])
    ordered = np.asarray([float(row["threshold"]) for row in rows], dtype=np.float64)
    if not np.array_equal(ordered, thresholds):
        raise ValueError("Formal threshold curve must use a unique increasing grid")

    fixed_gt: int | None = None
    fixed_pixels: int | None = None
    for index, row in enumerate(rows):
        for column in CURVE_COLUMNS:
            if column not in row:
                raise ValueError(f"Curve row {index} is missing {column!r}")
        numeric = {
            key: float(row[key])
            for key in ("threshold", "pd", "fa_pixel", "fa_component_mp")
        }
        if not all(np.isfinite(value) for value in numeric.values()):
            raise ValueError(f"Curve row {index} contains a non-finite metric")
        if not 0.0 <= numeric["pd"] <= 1.0:
            raise ValueError(f"Curve row {index} has pd outside [0, 1]")
        if numeric["fa_pixel"] < 0.0 or numeric["fa_component_mp"] < 0.0:
            raise ValueError(f"Curve row {index} has negative false-alarm risk")

        counts: dict[str, int] = {}
        for key in (
            "tp_objects",
            "gt_objects",
            "fp_components",
            "fp_pixels",
            "total_pixels",
        ):
            value = row[key]
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ValueError(f"Curve row {index} has invalid count {key!r}")
            counts[key] = int(value)
        if counts["total_pixels"] <= 0:
            raise ValueError("Curve total_pixels must be strictly positive")
        if counts["tp_objects"] > counts["gt_objects"]:
            raise ValueError("Curve tp_objects cannot exceed gt_objects")
        if counts["fp_pixels"] > counts["total_pixels"]:
            raise ValueError("Curve fp_pixels cannot exceed total_pixels")
        if fixed_gt is None:
            fixed_gt = counts["gt_objects"]
            fixed_pixels = counts["total_pixels"]
        elif counts["gt_objects"] != fixed_gt or counts["total_pixels"] != fixed_pixels:
            raise ValueError("Curve denominators must be constant across thresholds")

        expected_pd = (
            counts["tp_objects"] / counts["gt_objects"]
            if counts["gt_objects"]
            else 0.0
        )
        expected_pixel = counts["fp_pixels"] / counts["total_pixels"]
        expected_component = counts["fp_components"] / (
            counts["total_pixels"] / 1_000_000.0
        )
        for key, expected in (
            ("pd", expected_pd),
            ("fa_pixel", expected_pixel),
            ("fa_component_mp", expected_component),
        ):
            if not np.isclose(numeric[key], expected, rtol=1e-12, atol=1e-15):
                raise ValueError(
                    f"Curve row {index} {key} disagrees with its raw counts"
                )
    return thresholds


def load_formal_curve(
    curve_path: str | Path,
    *,
    expected_split_role: str,
) -> FormalCurve:
    """Load a curve and revalidate its CSV, sidecar, and live score artifact."""

    curve = Path(curve_path).expanduser().resolve()
    if not curve.is_file():
        raise FileNotFoundError(f"Curve CSV does not exist: {curve}")
    sidecar = curve_metadata_path(curve)
    if not sidecar.is_file():
        raise FileNotFoundError(f"Formal curve metadata sidecar does not exist: {sidecar}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Formal curve metadata must decode to a JSON object")
    if metadata.get("schema_version") != CURVE_METADATA_SCHEMA_VERSION:
        raise ValueError("Unsupported formal curve metadata schema_version")
    if metadata.get("artifact_type") != CURVE_METADATA_ARTIFACT_TYPE:
        raise ValueError("Unsupported formal curve artifact_type")
    if metadata.get("formal_protocol_eligible") is not True:
        raise ValueError("Curve sidecar is not formal_protocol_eligible")
    if metadata.get("diagnostic_only") is not False:
        raise ValueError("Source selection rejects diagnostic curve sidecars")
    if metadata.get("labels_used_for_curve") is not True:
        raise ValueError("Formal threshold curves must be label-derived")
    if metadata.get("curve_file") != curve.name:
        raise ValueError("Curve sidecar curve_file does not match the CSV")
    if _require_sha256(metadata.get("curve_sha256"), name="curve_sha256") != file_sha256(
        curve
    ):
        raise ValueError("Curve CSV sha256 mismatch")

    rows = read_curve_csv(curve)
    thresholds = _validate_curve_rows(rows)
    recorded_thresholds = np.asarray(metadata.get("thresholds"), dtype=np.float64)
    if recorded_thresholds.ndim != 1 or not np.array_equal(recorded_thresholds, thresholds):
        raise ValueError("Curve sidecar threshold grid differs from the CSV")
    if metadata.get("threshold_grid_size") != int(thresholds.size):
        raise ValueError("Curve sidecar threshold_grid_size mismatch")
    grid_sha = evaluation_threshold_grid_sha256(thresholds)
    if _require_sha256(
        metadata.get("threshold_grid_sha256"), name="threshold_grid_sha256"
    ) != grid_sha:
        raise ValueError("Curve sidecar threshold_grid_sha256 mismatch")
    if metadata.get("curve_num_rows") != len(rows):
        raise ValueError("Curve sidecar curve_num_rows mismatch")
    if metadata.get("curve_columns") != list(CURVE_COLUMNS):
        raise ValueError("Curve sidecar curve_columns mismatch")
    if metadata.get("expected_split_role") != expected_split_role:
        raise ValueError("Curve sidecar expected_split_role mismatch")

    score_dir_value = metadata.get("score_dir")
    if not isinstance(score_dir_value, str) or not score_dir_value:
        raise ValueError("Curve sidecar requires score_dir")
    score_dir = Path(score_dir_value).expanduser().resolve()
    manifest, _, integrity = verify_score_map_directory(
        score_dir,
        require_integrity=True,
        require_masks=True,
    )
    contract = validate_formal_score_manifest(
        manifest,
        integrity,
        expected_split_role=expected_split_role,
    )
    if manifest is None:  # pragma: no cover - require_integrity already fails closed
        raise AssertionError("verified formal score maps must have a manifest")
    expected_fields = {
        "score_manifest_schema_version": int(manifest["schema_version"]),
        "score_manifest_sha256": contract["score_manifest_sha256"],
        "score_records_sha256": contract["score_records_sha256"],
        "score_ordered_image_ids_sha256": contract[
            "score_ordered_image_ids_sha256"
        ],
        "score_num_records": contract["score_num_records"],
        "detector_weight_sha256": contract["detector_weight_sha256"],
        "checkpoint_selection_rule": contract["checkpoint_selection_rule"],
        "model_backend": contract["model_backend"],
        "target_dataset": contract["target_dataset"],
        "target_domain_key": contract["target_domain_key"],
        "source_datasets": contract["source_datasets"],
        "source_domain_keys": contract["source_domain_keys"],
        "requested_split": contract["requested_split"],
        "split_role": contract["split_role"],
        "split_file_sha256": contract["split_file_sha256"],
        "split_ordered_ids_sha256": contract["split_ordered_ids_sha256"],
        "labels_loaded": True,
        "spatial_mode": "native",
        "checkpoint_diagnostic_only": False,
        "non_strict_state_loading": False,
    }
    for field, expected in expected_fields.items():
        if metadata.get(field) != expected:
            raise ValueError(f"Curve sidecar {field} does not match score provenance")
    if metadata.get("matching_rule") not in {"overlap", "centroid"}:
        raise ValueError("Curve sidecar matching_rule is invalid")
    centroid_distance = float(metadata.get("centroid_distance", float("nan")))
    if not np.isfinite(centroid_distance) or centroid_distance < 0.0:
        raise ValueError("Curve sidecar centroid_distance is invalid")
    if metadata.get("connectivity") not in {1, 2, 4, 8}:
        raise ValueError("Curve sidecar connectivity is invalid")
    min_area = metadata.get("min_component_area")
    if isinstance(min_area, bool) or not isinstance(min_area, int) or min_area <= 0:
        raise ValueError("Curve sidecar min_component_area is invalid")

    return FormalCurve(
        path=curve,
        metadata_path=sidecar,
        rows=tuple(dict(row) for row in rows),
        metadata=metadata,
        metadata_sha256=file_sha256(sidecar),
    )


def _aggregate_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("Cannot aggregate an empty source-row set")
    threshold = float(rows[0]["threshold"])
    if any(float(row["threshold"]) != threshold for row in rows):
        raise ValueError("Source rows at one grid index have different thresholds")
    totals = {
        key: int(sum(int(row[key]) for row in rows))
        for key in (
            "tp_objects",
            "gt_objects",
            "fp_components",
            "fp_pixels",
            "total_pixels",
        )
    }
    return {
        "threshold": threshold,
        "pd": (
            float(totals["tp_objects"] / totals["gt_objects"])
            if totals["gt_objects"]
            else 0.0
        ),
        "fa_pixel": float(totals["fp_pixels"] / totals["total_pixels"]),
        "fa_component_mp": float(
            totals["fp_components"] / (totals["total_pixels"] / 1_000_000.0)
        ),
        **totals,
    }


def select_source_operating_points(
    source_curves: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    pixel_budget: float,
    component_budget: float,
) -> dict[str, dict[str, Any]]:
    """Select pooled and all-source-feasible worst-domain operating points.

    Pooled ties use the lowest threshold.  Worst-domain ties first maximize the
    pooled Pd and then use the lowest threshold.
    """

    pixel_budget = _validate_budget(pixel_budget, name="pixel_budget")
    component_budget = _validate_budget(component_budget, name="component_budget")
    if len(source_curves) < 2:
        raise ValueError("At least two named source curves are required")
    if any(not isinstance(name, str) or not name for name in source_curves):
        raise ValueError("Source curve names must be non-empty strings")
    names = sorted(source_curves)
    validated_grids = {
        name: _validate_curve_rows(source_curves[name]) for name in names
    }
    reference_grid = validated_grids[names[0]]
    for name in names[1:]:
        if not np.array_equal(validated_grids[name], reference_grid):
            raise ValueError("Source threshold grids are inconsistent")

    pooled_candidates: list[tuple[dict[str, float | int], int]] = []
    worst_candidates: list[tuple[float, float, float, int]] = []
    for index, threshold in enumerate(reference_grid):
        rows_at_threshold = [source_curves[name][index] for name in names]
        pooled = _aggregate_rows(rows_at_threshold)
        if is_budget_satisfied(
            pooled,
            pixel_budget=pixel_budget,
            component_budget=component_budget,
        ):
            pooled_candidates.append((pooled, index))
        if all(
            is_budget_satisfied(
                row,
                pixel_budget=pixel_budget,
                component_budget=component_budget,
            )
            for row in rows_at_threshold
        ):
            worst_pd = min(float(row["pd"]) for row in rows_at_threshold)
            worst_candidates.append((worst_pd, float(pooled["pd"]), float(threshold), index))

    pooled_result: dict[str, Any] = {
        "found": bool(pooled_candidates),
        "strategy": "aggregate_raw_counts_then_maximize_pooled_pd",
        "tie_break": ["lowest_threshold"],
        "operating_point": None,
        "source_rows": None,
    }
    if pooled_candidates:
        pooled_row, pooled_index = min(
            pooled_candidates,
            key=lambda item: (-float(item[0]["pd"]), float(item[0]["threshold"])),
        )
        pooled_result["operating_point"] = pooled_row
        pooled_result["source_rows"] = {
            name: dict(source_curves[name][pooled_index]) for name in names
        }

    worst_result: dict[str, Any] = {
        "found": bool(worst_candidates),
        "strategy": "require_every_source_budget_then_maximize_worst_domain_pd",
        "tie_break": ["maximize_pooled_pd", "lowest_threshold"],
        "operating_point": None,
        "worst_domain_pd": None,
        "worst_domain_names": [],
        "source_rows": None,
    }
    if worst_candidates:
        worst_pd, _, _, worst_index = min(
            worst_candidates,
            key=lambda item: (-item[0], -item[1], item[2]),
        )
        selected_rows = {
            name: dict(source_curves[name][worst_index]) for name in names
        }
        worst_result["operating_point"] = _aggregate_rows(list(selected_rows.values()))
        worst_result["worst_domain_pd"] = float(worst_pd)
        worst_result["worst_domain_names"] = [
            name
            for name, row in selected_rows.items()
            if np.isclose(float(row["pd"]), worst_pd, rtol=0.0, atol=1e-15)
        ]
        worst_result["source_rows"] = selected_rows

    return {"source_pooled": pooled_result, "source_worst": worst_result}


def _protocol_signature(curve: FormalCurve) -> tuple[Any, ...]:
    metadata = curve.metadata
    return (
        metadata["threshold_grid_sha256"],
        tuple(float(value) for value in metadata["thresholds"]),
        metadata["matching_rule"],
        float(metadata["centroid_distance"]),
        int(metadata["connectivity"]),
        int(metadata["min_component_area"]),
    )


def _parse_named_source(values: Sequence[Sequence[str]]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    parsed_keys: set[str] = set()
    for value in values:
        if len(value) == 1 and "=" in value[0]:
            name, raw_path = value[0].split("=", maxsplit=1)
        elif len(value) == 2:
            name, raw_path = value
        else:
            raise ValueError(
                "--source-curve expects NAME=PATH or two values: NAME PATH"
            )
        name = name.strip()
        raw_path = raw_path.strip()
        if not name or not raw_path:
            raise ValueError("Source curve name and path must be non-empty")
        if name in parsed:
            raise ValueError(f"Duplicate source curve name: {name}")
        key = domain_key(name, field="source curve name")
        if key in parsed_keys:
            raise ValueError(f"Duplicate source curve domain alias: {name}")
        parsed_keys.add(key)
        parsed[name] = Path(raw_path).expanduser()
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-curve",
        action="append",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help="Repeat for each source; accepts NAME=PATH or NAME PATH",
    )
    parser.add_argument(
        "--target-curve",
        help=(
            "Optional formal target-test curve. It is evaluated only after "
            "source thresholds are frozen and never participates in selection."
        ),
    )
    parser.add_argument("--pixel-budget", type=float, required=True)
    parser.add_argument("--component-budget", type=float, required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    pixel_budget = _validate_budget(args.pixel_budget, name="pixel_budget")
    component_budget = _validate_budget(
        args.component_budget, name="component_budget"
    )
    source_paths = _parse_named_source(args.source_curve)
    source_artifacts = {
        name: load_formal_curve(path, expected_split_role="train")
        for name, path in sorted(source_paths.items())
    }
    source_datasets = [
        str(artifact.metadata["target_dataset"])
        for artifact in source_artifacts.values()
    ]
    source_dataset_keys = [
        domain_key(value, field="source curve target_dataset")
        for value in source_datasets
    ]
    if len(set(source_dataset_keys)) != len(source_dataset_keys):
        raise ValueError("Named source curves must represent distinct source datasets")
    for name, artifact in source_artifacts.items():
        if domain_key(name, field="source curve name") != domain_key(
            artifact.metadata["target_dataset"], field="source curve target_dataset"
        ):
            raise ValueError(
                f"Source curve name {name!r} does not identify its target_dataset"
            )
    meta_domain_keys = set(source_dataset_keys)
    for name, artifact in source_artifacts.items():
        held_out_key = domain_key(
            artifact.metadata["target_dataset"],
            field="source curve target_dataset",
        )
        observed_training_keys = set(artifact.metadata["source_domain_keys"])
        expected_training_keys = meta_domain_keys.difference({held_out_key})
        if observed_training_keys != expected_training_keys:
            raise ValueError(
                "Source curves do not form a complete pseudo-target LODO closure: "
                f"{name!r} source_domain_keys={sorted(observed_training_keys)}, "
                f"expected={sorted(expected_training_keys)}"
            )
    signatures = {_protocol_signature(curve) for curve in source_artifacts.values()}
    if len(signatures) != 1:
        raise ValueError(
            "Source curves must use an identical threshold grid and matching protocol"
        )

    results = select_source_operating_points(
        {name: artifact.rows for name, artifact in source_artifacts.items()},
        pixel_budget=pixel_budget,
        component_budget=component_budget,
    )

    target_evaluation: dict[str, Any] = {
        "provided": False,
        "target_labels_used": False,
        "target_labels_used_for_selection": False,
        "selection_frozen_before_target_load": True,
        "dataset": None,
        "curve_provenance": None,
        "rows_at_source_selected_thresholds": {},
    }
    target_artifact: FormalCurve | None = None
    if args.target_curve:
        # This load deliberately occurs only after ``results`` has been fixed.
        target_artifact = load_formal_curve(
            args.target_curve, expected_split_role="test"
        )
        target_dataset = str(target_artifact.metadata["target_dataset"])
        if domain_key(target_dataset, field="optional target dataset") in set(
            source_dataset_keys
        ):
            raise ValueError(
                "Optional target dataset must not be included among source curves"
            )
        if set(target_artifact.metadata["source_domain_keys"]) != meta_domain_keys:
            raise ValueError(
                "Optional final-target curve must be produced by a detector trained "
                "on exactly the complete meta-domain set"
            )
        if _protocol_signature(target_artifact) not in signatures:
            raise ValueError(
                "Target curve must use the same threshold grid and matching protocol"
            )
        target_grid = np.asarray(
            [float(row["threshold"]) for row in target_artifact.rows]
        )
        target_rows: dict[str, Any] = {}
        for mode, result in results.items():
            operating_point = result.get("operating_point")
            if operating_point is None:
                target_rows[mode] = None
                continue
            threshold = float(operating_point["threshold"])
            matches = np.flatnonzero(target_grid == threshold)
            if matches.size != 1:  # pragma: no cover - signature check protects this
                raise ValueError("Selected source threshold is absent from target grid")
            target_rows[mode] = dict(target_artifact.rows[int(matches[0])])
        target_evaluation = {
            "provided": True,
            "target_labels_used": True,
            "target_labels_used_for_selection": False,
            "selection_frozen_before_target_load": True,
            "dataset": target_dataset,
            "curve_provenance": {
                "curve": str(target_artifact.path),
                "curve_sha256": target_artifact.metadata["curve_sha256"],
                "metadata": str(target_artifact.metadata_path),
                "metadata_sha256": target_artifact.metadata_sha256,
                "score_manifest_sha256": target_artifact.metadata[
                    "score_manifest_sha256"
                ],
                "score_records_sha256": target_artifact.metadata[
                    "score_records_sha256"
                ],
            },
            "rows_at_source_selected_thresholds": target_rows,
        }

    first = next(iter(source_artifacts.values()))
    payload: dict[str, Any] = {
        "schema_version": SOURCE_SELECTION_SCHEMA_VERSION,
        "artifact_type": SOURCE_SELECTION_ARTIFACT_TYPE,
        "formal_protocol_eligible": True,
        "diagnostic_only": False,
        "selection_is_source_only": True,
        "pseudo_target_lodo_closure_verified": True,
        "meta_domain_keys": sorted(meta_domain_keys),
        "target_curve_used_for_selection": False,
        "pixel_budget": pixel_budget,
        "component_budget": component_budget,
        "threshold_grid_sha256": first.metadata["threshold_grid_sha256"],
        "matching_protocol": {
            "matching_rule": first.metadata["matching_rule"],
            "centroid_distance": first.metadata["centroid_distance"],
            "connectivity": first.metadata["connectivity"],
            "min_component_area": first.metadata["min_component_area"],
        },
        "source_curves": {
            name: {
                "dataset": artifact.metadata["target_dataset"],
                "curve": str(artifact.path),
                "curve_sha256": artifact.metadata["curve_sha256"],
                "metadata": str(artifact.metadata_path),
                "metadata_sha256": artifact.metadata_sha256,
                "score_manifest_sha256": artifact.metadata[
                    "score_manifest_sha256"
                ],
                "score_records_sha256": artifact.metadata[
                    "score_records_sha256"
                ],
                "detector_weight_sha256": artifact.metadata[
                    "detector_weight_sha256"
                ],
                "source_datasets": artifact.metadata["source_datasets"],
                "split_role": artifact.metadata["split_role"],
            }
            for name, artifact in source_artifacts.items()
        },
        "results": results,
        "target_evaluation": target_evaluation,
    }
    write_json_atomic(args.output, payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FormalCurve",
    "SOURCE_SELECTION_ARTIFACT_TYPE",
    "SOURCE_SELECTION_SCHEMA_VERSION",
    "load_formal_curve",
    "select_source_operating_points",
]
