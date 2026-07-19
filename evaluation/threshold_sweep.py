"""Threshold sweeps over exported continuous score maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .component_matching import match_components
from .artifact_integrity import file_sha256, verify_score_map_directory


CURVE_COLUMNS = (
    "threshold",
    "pd",
    "fa_pixel",
    "fa_component_mp",
    "tp_objects",
    "gt_objects",
    "fp_components",
    "fp_pixels",
    "total_pixels",
)

CURVE_METADATA_SCHEMA_VERSION = 1
CURVE_METADATA_ARTIFACT_TYPE = "rc-irstd-formal-threshold-curve"
FORMAL_MODEL_BACKENDS = frozenset({"canonical", "rc_mshnet"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# ``sigmoid`` values stored as float32 can equal exactly 1.0.  With the shared
# ``score >= threshold`` rule, 1.0 is therefore not an empty prediction.  The
# next representable float64 value is a deterministic evaluation-only reject
# sentinel; it is not part of the predictor/calibration grid.
EMPTY_SET_THRESHOLD = float(np.nextafter(np.float64(1.0), np.float64(np.inf)))


def build_default_thresholds() -> np.ndarray:
    """Build the dense evaluation grid plus an explicit empty-set endpoint.

    This is the *evaluation* grid.  The fixed predictor grid is versioned
    separately and intentionally need not include 1.0.
    """

    return np.unique(
        np.concatenate(
            [
                np.linspace(0.00, 0.90, 91),
                np.linspace(0.90, 0.99, 181),
                np.linspace(0.99, 0.999, 181),
                np.linspace(0.999, 0.99999, 201),
                # Float32 sigmoid maps can contain exact 1.0 values.  Threshold
                # 1.0 is therefore a valid non-empty operating point and must
                # be evaluated before the explicit reject-all sentinel.
                np.asarray([1.0, EMPTY_SET_THRESHOLD]),
            ]
        )
    ).astype(np.float64)


def validate_thresholds(thresholds: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(thresholds, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all():
        raise ValueError("thresholds contain NaN or infinity")
    if np.any((values < 0.0) | (values > EMPTY_SET_THRESHOLD)):
        raise ValueError(
            "thresholds must lie in [0, 1] or equal the evaluation-only "
            "empty-set sentinel nextafter(1,+inf)"
        )
    return np.unique(values)


def evaluation_threshold_grid_sha256(
    thresholds: Sequence[float] | np.ndarray,
) -> str:
    """Hash the exact float64 evaluation grid, including its reject sentinel."""

    values = validate_thresholds(thresholds)
    canonical = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def curve_metadata_path(curve_path: str | Path) -> Path:
    """Return the deterministic JSON sidecar path for a threshold CSV."""

    path = Path(curve_path).expanduser()
    return path.with_name(path.name + ".metadata.json")


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a JSON object through an atomic same-directory replacement."""

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            delete=False,
            suffix=".tmp",
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, target)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return target


def _validate_score_pair(
    probability: np.ndarray,
    mask: np.ndarray,
    index: int,
) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(probability)
    target = np.asarray(mask)
    if score.ndim == 3 and score.shape[0] == 1:
        score = score[0]
    if target.ndim == 3 and target.shape[0] == 1:
        target = target[0]
    if score.ndim != 2 or target.ndim != 2:
        raise ValueError(f"score/mask pair {index} must contain 2-D arrays")
    if score.shape != target.shape:
        raise ValueError(
            f"score/mask pair {index} has mismatched shapes: {score.shape} vs {target.shape}"
        )
    if not np.issubdtype(score.dtype, np.number) or not np.isfinite(score).all():
        raise ValueError(f"probability map {index} must be finite and numeric")
    if np.any((score < 0.0) | (score > 1.0)):
        raise ValueError(f"probability map {index} contains values outside [0, 1]")
    if not np.issubdtype(target.dtype, np.number) and target.dtype != np.bool_:
        raise TypeError(f"mask {index} must be numeric or boolean")
    if not np.isfinite(target).all():
        raise ValueError(f"mask {index} contains NaN or infinity")
    return np.ascontiguousarray(score, dtype=np.float64), np.ascontiguousarray(target > 0)


def sweep_thresholds(
    probabilities: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    thresholds: Sequence[float] | np.ndarray,
    *,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
) -> list[dict[str, float | int]]:
    """Aggregate object and false-alarm metrics at every threshold."""

    if len(probabilities) != len(masks):
        raise ValueError("probabilities and masks must have the same length")
    if len(probabilities) == 0:
        raise ValueError("at least one score/mask pair is required")
    threshold_array = validate_thresholds(thresholds)
    pairs = [
        _validate_score_pair(probability, mask, index)
        for index, (probability, mask) in enumerate(zip(probabilities, masks))
    ]
    total_pixels = int(sum(mask.size for _, mask in pairs))
    rows: list[dict[str, float | int]] = []
    for threshold in threshold_array:
        total_tp = 0
        total_gt = 0
        total_fp_components = 0
        total_fp_pixels = 0
        for probability, mask in pairs:
            result = match_components(
                probability >= threshold,
                mask,
                rule=matching_rule,
                centroid_distance=centroid_distance,
                connectivity=connectivity,
                min_component_area=min_component_area,
            )
            total_tp += result.num_tp_objects
            total_gt += result.num_gt
            total_fp_components += result.num_fp_components
            total_fp_pixels += result.num_fp_pixels
        pd = float(total_tp / total_gt) if total_gt else 0.0
        fa_pixel = float(total_fp_pixels / total_pixels)
        fa_component_mp = float(total_fp_components / (total_pixels / 1_000_000.0))
        rows.append(
            {
                "threshold": float(threshold),
                "pd": pd,
                "fa_pixel": fa_pixel,
                "fa_component_mp": fa_component_mp,
                "tp_objects": int(total_tp),
                "gt_objects": int(total_gt),
                "fp_components": int(total_fp_components),
                "fp_pixels": int(total_fp_pixels),
                "total_pixels": int(total_pixels),
            }
        )
    return rows


def pixel_fa_is_monotone(
    rows: Sequence[Mapping[str, float | int]],
    *,
    atol: float = 0.0,
) -> bool:
    """Check non-increasing pixel false alarm after sorting by threshold."""

    if atol < 0 or not np.isfinite(atol):
        raise ValueError("atol must be finite and non-negative")
    ordered = sorted(rows, key=lambda row: float(row["threshold"]))
    values = np.asarray([float(row["fa_pixel"]) for row in ordered])
    return bool(np.all(np.diff(values) <= atol))


def write_curve_csv(
    rows: Sequence[Mapping[str, float | int]],
    output_path: str | Path,
) -> Path:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            delete=False,
            suffix=".tmp",
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=CURVE_COLUMNS, extrasaction="ignore"
            )
            writer.writeheader()
            for row in rows:
                missing = [column for column in CURVE_COLUMNS if column not in row]
                if missing:
                    raise ValueError(
                        f"curve row is missing columns: {', '.join(missing)}"
                    )
                writer.writerow({column: row[column] for column in CURVE_COLUMNS})
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return path


def read_curve_csv(path: str | Path) -> list[dict[str, float | int]]:
    input_path = Path(path).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Curve CSV does not exist: {input_path}")
    rows: list[dict[str, float | int]] = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(
            column not in reader.fieldnames for column in CURVE_COLUMNS
        ):
            raise ValueError(f"Curve CSV does not contain the required columns: {input_path}")
        for raw in reader:
            row: dict[str, float | int] = {}
            for column in CURVE_COLUMNS:
                if column in {"threshold", "pd", "fa_pixel", "fa_component_mp"}:
                    row[column] = float(raw[column])
                else:
                    row[column] = int(raw[column])
            rows.append(row)
    if not rows:
        raise ValueError(f"Curve CSV has no data rows: {input_path}")
    return rows


def _normalise_source_datasets(manifest: Mapping[str, Any]) -> list[str]:
    raw = manifest.get("source_datasets")
    if raw is None and manifest.get("source_dataset") is not None:
        raw = [manifest.get("source_dataset")]
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(value, str) or not value for value in raw)
    ):
        raise ValueError(
            "Formal score-map manifest requires unique, non-empty source_datasets"
        )
    values = [str(value) for value in raw]
    keys = [domain_key(value, field="source_datasets entry") for value in values]
    if len(set(keys)) != len(keys):
        raise ValueError("Formal source_datasets contain duplicate domain aliases")
    return values


def domain_key(value: Any, *, field: str = "domain") -> str:
    """Canonicalise common aliases such as ``NUAA`` and ``NUAA-SIRST``."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty domain name")
    key = "".join(character for character in value.casefold() if character.isalnum())
    if key.endswith("sirst") and len(key) > len("sirst"):
        key = key[: -len("sirst")]
    if not key:
        raise ValueError(f"{field} must contain an alphanumeric domain name")
    return key


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def validate_formal_score_manifest(
    manifest: Mapping[str, Any] | None,
    integrity: Mapping[str, Any],
    *,
    expected_split_role: str,
) -> dict[str, Any]:
    """Fail closed unless a score artifact is eligible for formal evaluation."""

    if expected_split_role not in {"train", "test"}:
        raise ValueError("expected_split_role must be 'train' or 'test'")
    if manifest is None or integrity.get("verified") is not True:
        raise ValueError("Formal threshold sweep requires verified v3 score maps")
    if manifest.get("labels_loaded") is not True:
        raise ValueError("Formal threshold sweep requires embedded ground-truth masks")
    if manifest.get("spatial_mode") != "native":
        raise ValueError("Formal threshold sweep requires native spatial_mode")
    if manifest.get("split_authority_verified") is not True:
        raise ValueError("Formal threshold sweep requires verified split authority")
    if manifest.get("split_role") != expected_split_role:
        raise ValueError(
            "Score-map split_role does not match --expected-split-role: "
            f"{manifest.get('split_role')!r} vs {expected_split_role!r}"
        )
    requested_split = manifest.get("requested_split")
    allowed_requested = {"train"} if expected_split_role == "train" else {"test", "val"}
    if requested_split not in allowed_requested:
        raise ValueError(
            "Score-map requested_split is inconsistent with its formal split role"
        )
    if manifest.get("checkpoint_diagnostic_only") is not False:
        raise ValueError(
            "Formal threshold sweep requires checkpoint_diagnostic_only=false"
        )
    for field in ("diagnostic_only", "non_strict_state_loading"):
        value = manifest.get(field, False)
        if not isinstance(value, bool):
            raise ValueError(f"Formal score-map manifest {field} must be boolean")
        if value:
            raise ValueError(
                "Formal threshold sweep rejects diagnostic or non-strict detector artifacts"
            )
    if manifest.get("formal_protocol_eligible") is False:
        raise ValueError("Score-map manifest explicitly marks itself non-formal")
    model_backend = manifest.get("model_backend")
    if model_backend not in FORMAL_MODEL_BACKENDS:
        allowed = ", ".join(sorted(FORMAL_MODEL_BACKENDS))
        raise ValueError(
            f"Formal threshold sweep requires model_backend in {{{allowed}}}"
        )
    if manifest.get("checkpoint_selection_rule") != "fixed_last":
        raise ValueError(
            "Formal threshold sweep requires checkpoint_selection_rule=fixed_last"
        )
    if manifest.get("score_type") != "sigmoid_probability":
        raise ValueError("Formal threshold sweep requires sigmoid_probability scores")

    target_dataset = manifest.get("target_dataset")
    if not isinstance(target_dataset, str) or not target_dataset:
        raise ValueError("Formal score-map manifest requires target_dataset")
    source_datasets = _normalise_source_datasets(manifest)
    target_domain_key = domain_key(target_dataset, field="target_dataset")
    source_domain_keys = [
        domain_key(value, field="source_datasets entry") for value in source_datasets
    ]
    if target_domain_key in set(source_domain_keys):
        raise ValueError(
            "Formal score-map target_dataset must be held out from source_datasets"
        )
    detector_weight_sha256 = _require_sha256(
        manifest.get("weight_sha256"), name="score manifest weight_sha256"
    )
    manifest_sha256 = _require_sha256(
        integrity.get("manifest_sha256"), name="score manifest sha256"
    )
    records_sha256 = _require_sha256(
        integrity.get("records_sha256"), name="score records_sha256"
    )
    ordered_ids_sha256 = _require_sha256(
        integrity.get("ordered_image_ids_sha256"),
        name="score ordered_image_ids_sha256",
    )
    split_file_sha256 = _require_sha256(
        manifest.get("split_file_sha256"), name="split_file_sha256"
    )
    split_ids_sha256 = _require_sha256(
        manifest.get("split_ordered_ids_sha256"),
        name="split_ordered_ids_sha256",
    )
    if split_ids_sha256 != ordered_ids_sha256:
        raise ValueError("Formal score-map split/image ordering hashes differ")
    return {
        "target_dataset": target_dataset,
        "target_domain_key": target_domain_key,
        "source_datasets": source_datasets,
        "source_domain_keys": source_domain_keys,
        "detector_weight_sha256": detector_weight_sha256,
        "checkpoint_selection_rule": "fixed_last",
        "model_backend": model_backend,
        "requested_split": str(requested_split),
        "split_role": expected_split_role,
        "split_file_sha256": split_file_sha256,
        "split_ordered_ids_sha256": split_ids_sha256,
        "score_manifest_sha256": manifest_sha256,
        "score_records_sha256": records_sha256,
        "score_ordered_image_ids_sha256": ordered_ids_sha256,
        "score_num_records": int(integrity.get("num_records", 0)),
    }


def _load_score_map_artifact(
    score_dir: str | Path,
    *,
    require_integrity: bool = False,
    expected_split_role: str | None = None,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    dict[str, Any] | None,
    dict[str, Any],
]:
    """Load score maps in manifest order and verify their artifact chain.

    Legacy unhashed inputs remain available for diagnostic use.  Formal callers
    must set ``require_integrity=True``.
    """

    root = Path(score_dir).expanduser()
    manifest, paths, integrity = verify_score_map_directory(
        root,
        require_integrity=require_integrity,
        require_masks=True,
    )
    if expected_split_role is not None:
        if not require_integrity:
            raise ValueError("expected_split_role requires integrity verification")
        validate_formal_score_manifest(
            manifest, integrity, expected_split_role=expected_split_role
        )

    probabilities: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            if "prob" not in payload or "mask" not in payload:
                raise ValueError(f"Score map lacks prob/mask arrays: {path}")
            if expected_split_role is not None:
                embedded_mode = str(np.asarray(payload["spatial_mode"]).item())
                if embedded_mode != "native":
                    raise ValueError(
                        f"Formal score map has non-native embedded spatial_mode: {path}"
                    )
            probabilities.append(np.asarray(payload["prob"]))
            masks.append(np.asarray(payload["mask"]))
    return probabilities, masks, manifest, integrity


def load_score_map_directory(
    score_dir: str | Path,
    *,
    require_integrity: bool = False,
    expected_split_role: str | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load score/mask arrays while preserving the legacy two-value API."""

    probabilities, masks, _, _ = _load_score_map_artifact(
        score_dir,
        require_integrity=require_integrity,
        expected_split_role=expected_split_role,
    )
    return probabilities, masks


def write_formal_curve_metadata(
    curve_path: str | Path,
    *,
    score_dir: str | Path,
    manifest: Mapping[str, Any],
    integrity: Mapping[str, Any],
    expected_split_role: str,
    thresholds: Sequence[float] | np.ndarray,
    matching_rule: str,
    centroid_distance: float,
    connectivity: int,
    min_component_area: int,
) -> Path:
    """Atomically bind a formal curve to its score and evaluation protocol."""

    curve = Path(curve_path).expanduser().resolve()
    if not curve.is_file():
        raise FileNotFoundError(f"Curve CSV does not exist: {curve}")
    contract = validate_formal_score_manifest(
        manifest, integrity, expected_split_role=expected_split_role
    )
    grid = validate_thresholds(thresholds)
    rows = read_curve_csv(curve)
    row_grid = np.asarray([float(row["threshold"]) for row in rows], dtype=np.float64)
    if not np.array_equal(row_grid, grid):
        raise ValueError("Curve rows do not exactly match the supplied threshold grid")
    payload: dict[str, Any] = {
        "schema_version": CURVE_METADATA_SCHEMA_VERSION,
        "artifact_type": CURVE_METADATA_ARTIFACT_TYPE,
        "formal_protocol_eligible": True,
        "diagnostic_only": False,
        "labels_used_for_curve": True,
        "curve_file": curve.name,
        "curve_sha256": file_sha256(curve),
        "curve_columns": list(CURVE_COLUMNS),
        "curve_num_rows": len(rows),
        "score_dir": str(Path(score_dir).expanduser().resolve()),
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
        "expected_split_role": expected_split_role,
        "split_file_sha256": contract["split_file_sha256"],
        "split_ordered_ids_sha256": contract["split_ordered_ids_sha256"],
        "labels_loaded": True,
        "spatial_mode": "native",
        "checkpoint_diagnostic_only": False,
        "non_strict_state_loading": False,
        "matching_rule": str(matching_rule),
        "centroid_distance": float(centroid_distance),
        "connectivity": int(connectivity),
        "min_component_area": int(min_component_area),
        "thresholds": [float(value) for value in grid],
        "threshold_grid_size": int(grid.size),
        "threshold_grid_sha256": evaluation_threshold_grid_sha256(grid),
    }
    return write_json_atomic(curve_metadata_path(curve), payload)


def _parse_threshold_argument(value: str) -> np.ndarray:
    try:
        return validate_thresholds([float(item.strip()) for item in value.split(",")])
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--output", required=True)
    threshold_group = parser.add_mutually_exclusive_group()
    threshold_group.add_argument("--thresholds", type=_parse_threshold_argument)
    threshold_group.add_argument("--threshold-grid")
    parser.add_argument("--matching-rule", choices=("overlap", "centroid"), default="overlap")
    parser.add_argument("--centroid-distance", type=float, default=3.0)
    parser.add_argument("--connectivity", type=int, default=2)
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument(
        "--formal",
        action="store_true",
        help=(
            "Fail closed on v3 integrity, labels, native resolution, verified "
            "split provenance, and non-diagnostic strict checkpoint loading"
        ),
    )
    parser.add_argument(
        "--expected-split-role",
        choices=("train", "test"),
        help="Required with --formal; binds the curve to its intended split role",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.formal and args.expected_split_role is None:
        raise ValueError("--formal requires --expected-split-role")
    if not args.formal and args.expected_split_role is not None:
        raise ValueError("--expected-split-role is meaningful only with --formal")
    if args.threshold_grid:
        thresholds = np.load(args.threshold_grid, allow_pickle=False)
    elif args.thresholds is not None:
        thresholds = args.thresholds
    else:
        thresholds = build_default_thresholds()
    probabilities, masks, manifest, integrity = _load_score_map_artifact(
        args.score_dir,
        require_integrity=bool(args.formal),
        expected_split_role=args.expected_split_role,
    )
    rows = sweep_thresholds(
        probabilities,
        masks,
        thresholds,
        matching_rule=args.matching_rule,
        centroid_distance=args.centroid_distance,
        connectivity=args.connectivity,
        min_component_area=args.min_component_area,
    )
    curve = write_curve_csv(rows, args.output)
    if args.formal:
        assert manifest is not None
        write_formal_curve_metadata(
            curve,
            score_dir=args.score_dir,
            manifest=manifest,
            integrity=integrity,
            expected_split_role=args.expected_split_role,
            thresholds=thresholds,
            matching_rule=args.matching_rule,
            centroid_distance=args.centroid_distance,
            connectivity=args.connectivity,
            min_component_area=args.min_component_area,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())


__all__ = [
    "CURVE_COLUMNS",
    "CURVE_METADATA_ARTIFACT_TYPE",
    "CURVE_METADATA_SCHEMA_VERSION",
    "FORMAL_MODEL_BACKENDS",
    "build_default_thresholds",
    "curve_metadata_path",
    "domain_key",
    "evaluation_threshold_grid_sha256",
    "load_score_map_directory",
    "pixel_fa_is_monotone",
    "read_curve_csv",
    "sweep_thresholds",
    "validate_thresholds",
    "validate_formal_score_manifest",
    "write_curve_csv",
    "write_formal_curve_metadata",
    "write_json_atomic",
]
