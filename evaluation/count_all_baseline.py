"""Conservative unlabeled ``count-all`` threshold baseline.

The warm-up selector reads only exported probability maps.  Every retained
predicted pixel and connected component is treated as a false alarm, giving an
observable upper bound that does not depend on target labels.  An optional
future split is audited only after the warm-up action has been frozen; its
masks are never used to select or revise the threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifact_integrity import verify_score_map_directory
from .component_matching import connected_components, match_components
from risk_curve.threshold_grid import threshold_grid_sha256


RESULT_SCHEMA_VERSION = "rc-v2-count-all-baseline-v1"
PIXEL_BUDGET_UNIT = "false_positive_pixels_per_evaluated_pixel"
COMPONENT_BUDGET_UNIT = "false_positive_components_per_megapixel"


@dataclass(frozen=True)
class ProbabilityRecord:
    """One exported score map loaded without touching its stored mask."""

    image_id: str
    probability: np.ndarray
    path: Path


@dataclass(frozen=True)
class FutureRecord:
    """One future score map and label, loaded only after action selection."""

    image_id: str
    probability: np.ndarray
    mask: np.ndarray
    path: Path


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ids_sha256(image_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(image_ids).encode("utf-8")).hexdigest()


def _scalar_string(value: np.ndarray | str) -> str:
    array = np.asarray(value)
    return str(array.item() if array.ndim == 0 else array.reshape(-1)[0])


def _validate_probability(value: np.ndarray, source: Path) -> np.ndarray:
    probability = np.asarray(value, dtype=np.float32)
    if probability.ndim == 3 and probability.shape[0] == 1:
        probability = probability[0]
    if probability.ndim != 2:
        raise ValueError(f"Probability map must be 2-D in {source}, got {probability.shape}")
    if not np.isfinite(probability).all() or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError(f"Probability map must be finite and lie in [0, 1]: {source}")
    return np.ascontiguousarray(probability)


def _validate_mask(value: np.ndarray, probability: np.ndarray, source: Path) -> np.ndarray:
    mask = np.asarray(value)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2 or mask.shape != probability.shape:
        raise ValueError(
            f"Future mask/probability shapes differ in {source}: "
            f"{mask.shape} vs {probability.shape}"
        )
    if not np.issubdtype(mask.dtype, np.number) and mask.dtype != np.bool_:
        raise TypeError(f"Future mask must be numeric or boolean: {source}")
    if not np.isfinite(mask).all():
        raise ValueError(f"Future mask contains NaN or infinity: {source}")
    return np.ascontiguousarray(mask > 0)


def validate_threshold_grid(thresholds: Sequence[float] | np.ndarray) -> np.ndarray:
    """Validate a frozen, strictly increasing probability threshold grid."""

    grid = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if grid.size == 0:
        raise ValueError("threshold grid must not be empty")
    if not np.isfinite(grid).all() or np.any((grid < 0.0) | (grid > 1.0)):
        raise ValueError("threshold grid must be finite and lie in [0, 1]")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("threshold grid must be strictly increasing")
    return grid


def _manifest_records(root: Path) -> tuple[list[tuple[Path, str | None]], dict[str, Any]]:
    if not root.is_dir():
        raise NotADirectoryError(f"Score-map directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    manifest: dict[str, Any] = {}
    entries: list[tuple[Path, str | None]] = []
    if manifest_path.is_file():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("Score-map manifest must contain a JSON object")
        manifest = loaded
        records = manifest.get("records")
        if records is not None:
            if not isinstance(records, list):
                raise ValueError("Score-map manifest records must be a list")
            for record in records:
                if not isinstance(record, Mapping) or "file" not in record:
                    raise ValueError("Every score-map manifest record must contain a file")
                entries.append(
                    (
                        root / str(record["file"]),
                        (
                            None
                            if record.get("image_id") is None
                            else str(record["image_id"])
                        ),
                    )
                )
        elif manifest.get("files"):
            entries = [(root / str(filename), None) for filename in manifest["files"]]
    if not entries:
        entries = [(path, None) for path in sorted(root.glob("*.npz"))]
    if not entries:
        raise FileNotFoundError(f"No .npz score maps found under {root}")
    manifest_count = manifest.get("num_images")
    if manifest_count is not None and int(manifest_count) != len(entries):
        raise ValueError(
            f"Manifest declares {manifest_count} images but lists {len(entries)} records"
        )
    missing = [str(path) for path, _ in entries if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Manifest references missing score maps: {missing[:3]}")
    return entries, manifest


def _manifest_provenance(
    root: Path,
    manifest: Mapping[str, Any],
    image_ids: Sequence[str],
    *,
    integrity_audit: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    metadata = {
        str(key): value
        for key, value in manifest.items()
        if key not in {"records", "files"}
    }
    return {
        "score_dir": str(root.resolve()),
        "manifest_path": str(manifest_path.resolve()) if manifest_path.is_file() else None,
        "manifest_sha256": _file_sha256(manifest_path) if manifest_path.is_file() else None,
        "manifest_metadata": metadata,
        "num_records": len(image_ids),
        "image_ids_sha256_in_record_order": _ids_sha256(image_ids),
        "integrity_audit": dict(integrity_audit),
    }


def load_probability_records(
    score_dir: str | Path,
    *,
    formal: bool = False,
    require_masks: bool | None = None,
) -> tuple[list[ProbabilityRecord], dict[str, Any]]:
    """Load IDs and probabilities only; this function never reads ``mask`` arrays.

    Formal mode first verifies the complete version-3 artifact and then uses
    the verifier's authoritative manifest and ordered paths.  The verifier may
    check that a mask key is present/absent according to the manifest, but the
    count-all selector itself loads only ``prob`` and ``image_id``.
    """

    root = Path(score_dir).expanduser()
    if formal:
        manifest, verified_paths, integrity = verify_score_map_directory(
            root,
            require_integrity=True,
            require_masks=require_masks,
        )
        if manifest is None or not integrity.get("verified", False):
            raise ValueError(f"Formal score artifact failed verification: {root}")
        manifest_records = manifest.get("records")
        if not isinstance(manifest_records, list) or len(manifest_records) != len(
            verified_paths
        ):
            raise ValueError("Verified score manifest record order is unavailable")
        entries = [
            (path, str(record["image_id"]))
            for path, record in zip(verified_paths, manifest_records)
        ]
        integrity_audit: dict[str, Any] = {
            "formal_requested": True,
            "required_mask_mode": require_masks,
            "labels_loaded": manifest.get("labels_loaded"),
            **integrity,
        }
    else:
        entries, manifest = _manifest_records(root)
        integrity_audit = {
            "formal_requested": False,
            "required_mask_mode": None,
            "labels_loaded": manifest.get("labels_loaded"),
            "verified": False,
            "mask_alignment_verified": False,
            "diagnostic_reason": "formal_integrity_not_requested",
            "num_records": len(entries),
        }
    records: list[ProbabilityRecord] = []
    for source, manifest_id in entries:
        with np.load(source, allow_pickle=False) as archive:
            if "prob" not in archive:
                raise ValueError(f"Score map is missing prob: {source}")
            probability = _validate_probability(archive["prob"], source)
            archive_id = (
                _scalar_string(archive["image_id"]) if "image_id" in archive else None
            )
        if manifest_id is not None and archive_id is not None and manifest_id != archive_id:
            raise ValueError(
                f"Manifest/archive image_id mismatch for {source}: "
                f"{manifest_id!r} vs {archive_id!r}"
            )
        image_id = manifest_id or archive_id or source.stem
        records.append(ProbabilityRecord(str(image_id), probability, source))
    image_ids = [record.image_id for record in records]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Score-map image IDs must be unique")
    return records, _manifest_provenance(
        root,
        manifest,
        image_ids,
        integrity_audit=integrity_audit,
    )


def load_future_records(
    score_dir: str | Path,
    *,
    formal: bool = False,
) -> tuple[list[FutureRecord], dict[str, Any]]:
    """Load future labels for audit after the warm-up action has been frozen."""

    probability_records, provenance = load_probability_records(
        score_dir,
        formal=formal,
        require_masks=True if formal else None,
    )
    records: list[FutureRecord] = []
    for record in probability_records:
        with np.load(record.path, allow_pickle=False) as archive:
            if "mask" not in archive:
                raise ValueError(f"Future score map is missing mask: {record.path}")
            mask = _validate_mask(archive["mask"], record.probability, record.path)
        records.append(
            FutureRecord(record.image_id, record.probability, mask, record.path)
        )
    return records, provenance


def conservative_suffix_max(curves: np.ndarray) -> np.ndarray:
    """Return a non-increasing component-count upper envelope."""

    values = np.asarray(curves, dtype=np.float64)
    if values.ndim not in (1, 2) or values.shape[-1] == 0:
        raise ValueError("component curves must have shape [T] or [N, T]")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("component curves must contain finite non-negative counts")
    return np.maximum.accumulate(values[..., ::-1], axis=-1)[..., ::-1]


def build_count_all_curves(
    records: Sequence[ProbabilityRecord],
    thresholds: Sequence[float] | np.ndarray,
    *,
    connectivity: int = 2,
    min_component_area: int = 1,
) -> dict[str, np.ndarray | int]:
    """Count every retained warm-up prediction as a false-alarm upper bound."""

    if not records:
        raise ValueError("at least one warm-up probability map is required")
    grid = validate_threshold_grid(thresholds)
    num_images = len(records)
    pixel_counts = np.zeros((num_images, grid.size), dtype=np.int64)
    component_counts = np.zeros_like(pixel_counts)
    total_pixels = np.zeros(num_images, dtype=np.int64)
    for image_index, record in enumerate(records):
        probability = _validate_probability(record.probability, record.path)
        total_pixels[image_index] = probability.size
        for threshold_index, threshold in enumerate(grid):
            labels, num_components = connected_components(
                probability >= float(threshold),
                connectivity=connectivity,
                min_component_area=min_component_area,
            )
            pixel_counts[image_index, threshold_index] = int(
                np.count_nonzero(labels)
            )
            component_counts[image_index, threshold_index] = int(num_components)
    component_envelope = conservative_suffix_max(component_counts).astype(np.int64)
    total_exposure = int(np.sum(total_pixels))
    aggregate_pixels = np.sum(pixel_counts, axis=0)
    aggregate_components_raw = np.sum(component_counts, axis=0)
    aggregate_components_envelope = np.sum(component_envelope, axis=0)
    pixel_risk = aggregate_pixels / float(total_exposure)
    component_risk = aggregate_components_envelope / (total_exposure / 1_000_000.0)
    if np.any(np.diff(pixel_risk) > 0.0):
        raise AssertionError("Internal error: count-all pixel curve is not monotone")
    if np.any(np.diff(component_risk) > 0.0):
        raise AssertionError("Internal error: component envelope is not monotone")
    return {
        "thresholds": grid,
        "per_image_predicted_pixel_counts": pixel_counts,
        "per_image_component_counts_raw": component_counts,
        "per_image_component_counts_suffix_max": component_envelope,
        "total_pixels_per_image": total_pixels,
        "total_pixels": total_exposure,
        "predicted_pixel_counts": aggregate_pixels,
        "component_counts_raw": aggregate_components_raw,
        "component_counts_suffix_max": aggregate_components_envelope,
        "pixel_upper_bound_risk": pixel_risk,
        "component_upper_bound_risk_per_mp": component_risk,
    }


def select_count_all_threshold(
    curves: Mapping[str, np.ndarray | int],
    *,
    pixel_budget: float,
    component_budget: float,
) -> dict[str, Any]:
    """Select the first dual-budget-feasible grid point or explicitly reject."""

    if not np.isfinite(pixel_budget) or pixel_budget <= 0.0:
        raise ValueError("pixel_budget must be finite and positive")
    if not np.isfinite(component_budget) or component_budget <= 0.0:
        raise ValueError("component_budget must be finite and positive")
    thresholds = validate_threshold_grid(np.asarray(curves["thresholds"]))
    pixel = np.asarray(curves["pixel_upper_bound_risk"], dtype=np.float64)
    component = np.asarray(
        curves["component_upper_bound_risk_per_mp"], dtype=np.float64
    )
    if pixel.shape != thresholds.shape or component.shape != thresholds.shape:
        raise ValueError("count-all risks and threshold grid must have identical shapes")
    feasible = np.flatnonzero(
        (pixel <= float(pixel_budget)) & (component <= float(component_budget))
    )
    if feasible.size == 0:
        return {
            "success": False,
            "reject": True,
            "reason": "no_threshold_satisfies_count_all_upper_bounds",
            "threshold_index": None,
            "threshold": None,
        }
    index = int(feasible[0])
    return {
        "success": True,
        "reject": False,
        "reason": "first_dual_budget_feasible_grid_point",
        "threshold_index": index,
        "threshold": float(thresholds[index]),
    }


def audit_frozen_action(
    records: Sequence[FutureRecord],
    *,
    threshold: float | None,
    reject: bool,
    pixel_budget: float,
    component_budget: float,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
) -> dict[str, Any]:
    """Audit one already frozen threshold/reject action without reselection."""

    if not records:
        raise ValueError("future audit requires at least one record")
    if reject:
        if threshold is not None:
            raise ValueError("a rejected action must not carry a threshold")
    elif threshold is None or not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("a non-rejected action requires a threshold in [0, 1]")

    per_image: list[dict[str, Any]] = []
    total_pixels = 0
    total_fp_pixels = 0
    total_fp_components = 0
    total_tp_objects = 0
    total_gt_objects = 0
    for record in records:
        prediction = (
            np.zeros_like(record.mask, dtype=bool)
            if reject
            else record.probability >= float(threshold)
        )
        matched = match_components(
            prediction,
            record.mask,
            rule=matching_rule,
            centroid_distance=centroid_distance,
            connectivity=connectivity,
            min_component_area=min_component_area,
        )
        exposure = int(record.mask.size)
        pixel_risk = matched.num_fp_pixels / float(exposure)
        component_risk = matched.num_fp_components / (exposure / 1_000_000.0)
        satisfied = pixel_risk <= pixel_budget and component_risk <= component_budget
        per_image.append(
            {
                "image_id": record.image_id,
                "total_pixels": exposure,
                "fp_pixels": int(matched.num_fp_pixels),
                "fp_components": int(matched.num_fp_components),
                "tp_objects": int(matched.num_tp_objects),
                "gt_objects": int(matched.num_gt),
                "fa_pixel": float(pixel_risk),
                "fa_component_mp": float(component_risk),
                "budget_satisfied": bool(satisfied),
            }
        )
        total_pixels += exposure
        total_fp_pixels += int(matched.num_fp_pixels)
        total_fp_components += int(matched.num_fp_components)
        total_tp_objects += int(matched.num_tp_objects)
        total_gt_objects += int(matched.num_gt)
    return {
        "action": "terminal_reject_empty_prediction" if reject else "frozen_threshold",
        "threshold": None if reject else float(threshold),
        "threshold_reselected_on_future": False,
        "num_images": len(records),
        "image_ids": [record.image_id for record in records],
        "bsr": float(np.mean([row["budget_satisfied"] for row in per_image])),
        "pd": (
            float(total_tp_objects / total_gt_objects)
            if total_gt_objects > 0
            else None
        ),
        "fa_pixel": float(total_fp_pixels / total_pixels),
        "fa_component_mp": float(
            total_fp_components / (total_pixels / 1_000_000.0)
        ),
        "tp_objects": total_tp_objects,
        "gt_objects": total_gt_objects,
        "fp_pixels": total_fp_pixels,
        "fp_components": total_fp_components,
        "total_pixels": total_pixels,
        "per_image": per_image,
    }


def _curve_json(curves: Mapping[str, np.ndarray | int]) -> dict[str, Any]:
    return {
        "thresholds": np.asarray(curves["thresholds"]).tolist(),
        "predicted_pixel_counts": np.asarray(curves["predicted_pixel_counts"]).tolist(),
        "component_counts_raw": np.asarray(curves["component_counts_raw"]).tolist(),
        "component_counts_suffix_max": np.asarray(
            curves["component_counts_suffix_max"]
        ).tolist(),
        "pixel_upper_bound_risk": np.asarray(
            curves["pixel_upper_bound_risk"]
        ).tolist(),
        "component_upper_bound_risk_per_mp": np.asarray(
            curves["component_upper_bound_risk_per_mp"]
        ).tolist(),
        "total_pixels": int(curves["total_pixels"]),
    }


def run_count_all_baseline(
    warmup_score_dir: str | Path,
    thresholds: Sequence[float] | np.ndarray,
    *,
    pixel_budget: float,
    component_budget: float,
    future_score_dir: str | Path | None = None,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    connectivity: int = 2,
    min_component_area: int = 1,
    formal: bool = False,
) -> dict[str, Any]:
    """Run warm-up-only selection and an optional post-freeze future audit."""

    warmup_records, warmup_provenance = load_probability_records(
        warmup_score_dir,
        formal=formal,
        require_masks=None,
    )
    grid = validate_threshold_grid(thresholds)
    curves = build_count_all_curves(
        warmup_records,
        grid,
        connectivity=connectivity,
        min_component_area=min_component_area,
    )
    selection = select_count_all_threshold(
        curves,
        pixel_budget=pixel_budget,
        component_budget=component_budget,
    )

    # The action is fully frozen before this branch can open a future mask.
    future_audit: dict[str, Any] | None = None
    future_provenance: dict[str, Any] | None = None
    if future_score_dir is not None:
        future_records, future_provenance = load_future_records(
            future_score_dir,
            formal=formal,
        )
        overlap = sorted(
            {record.image_id for record in warmup_records}.intersection(
                record.image_id for record in future_records
            )
        )
        if overlap:
            raise ValueError(
                "Warm-up/future image ID leakage detected: " + ", ".join(overlap[:5])
            )
        future_audit = audit_frozen_action(
            future_records,
            threshold=selection["threshold"],
            reject=selection["reject"],
            pixel_budget=pixel_budget,
            component_budget=component_budget,
            matching_rule=matching_rule,
            centroid_distance=centroid_distance,
            connectivity=connectivity,
            min_component_area=min_component_area,
        )

    warmup_integrity = dict(warmup_provenance["integrity_audit"])
    future_integrity = (
        dict(future_provenance["integrity_audit"])
        if future_provenance is not None
        else None
    )
    integrity_verified = bool(
        formal
        and warmup_integrity.get("verified", False)
        and (
            future_integrity is None
            or future_integrity.get("verified", False)
        )
    )

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "mode": "unlabeled_count_all_upper_bound",
        "formal_requested": bool(formal),
        "formal_protocol_eligible": integrity_verified,
        **selection,
        "pixel_budget": float(pixel_budget),
        "component_budget": float(component_budget),
        "budgets": {
            "pixel": {"value": float(pixel_budget), "unit": PIXEL_BUDGET_UNIT},
            "component": {
                "value": float(component_budget),
                "unit": COMPONENT_BUDGET_UNIT,
            },
        },
        "thresholds": grid.tolist(),
        "warmup_image_ids": [record.image_id for record in warmup_records],
        "future_image_ids": (
            future_audit["image_ids"] if future_audit is not None else []
        ),
        "bsr": future_audit["bsr"] if future_audit is not None else None,
        "pd": future_audit["pd"] if future_audit is not None else None,
        "warmup_curve": _curve_json(curves),
        "future_audit": future_audit,
        "split_audit": {
            "warmup_count": len(warmup_records),
            "future_count": (
                int(future_audit["num_images"]) if future_audit is not None else 0
            ),
            "overlap_count": 0,
            "warmup_future_disjoint": True,
        },
        "protocol": {
            "selection_source": "warmup_score_maps_only",
            "warmup_arrays_read_for_selection": ["prob", "image_id_if_present"],
            "warmup_masks_read_for_selection": False,
            "warmup_integrity_verification_before_selection": bool(formal),
            "all_retained_warmup_predictions_counted_as_false_alarms": True,
            "component_monotonicity": "per-image suffix-maximum count envelope",
            "threshold_comparison": "probability >= threshold",
            "future_labels_used_for_selection": False,
            "future_threshold_reselection": False,
            "reject_action": "empty prediction / abstention",
            "spatial_handling": "use exported valid score-map pixels as-is; variable native shapes supported",
            "matching_rule": matching_rule,
            "centroid_distance": float(centroid_distance),
            "connectivity": int(connectivity),
            "min_component_area": int(min_component_area),
        },
        "provenance": {
            "warmup": warmup_provenance,
            "future": future_provenance,
        },
        "integrity_audit": {
            "verified": integrity_verified,
            "formal_requested": bool(formal),
            "warmup": warmup_integrity,
            "future": future_integrity,
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warmup-score-dir",
        "--score-dir",
        dest="warmup_score_dir",
        required=True,
    )
    parser.add_argument("--future-score-dir")
    parser.add_argument(
        "--formal",
        action="store_true",
        help=(
            "Require complete version-3 score integrity; future inputs must be "
            "labeled and mask-alignment verified"
        ),
    )
    parser.add_argument("--threshold-grid", required=True)
    parser.add_argument("--pixel-budget", required=True, type=float)
    parser.add_argument("--component-budget", required=True, type=float)
    parser.add_argument(
        "--matching-rule", choices=("overlap", "centroid"), default="overlap"
    )
    parser.add_argument("--centroid-distance", type=float, default=3.0)
    parser.add_argument("--connectivity", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    grid_path = Path(args.threshold_grid).expanduser()
    thresholds = np.load(grid_path, allow_pickle=False)
    result = run_count_all_baseline(
        args.warmup_score_dir,
        thresholds,
        pixel_budget=args.pixel_budget,
        component_budget=args.component_budget,
        future_score_dir=args.future_score_dir,
        matching_rule=args.matching_rule,
        centroid_distance=args.centroid_distance,
        connectivity=args.connectivity,
        min_component_area=args.min_component_area,
        formal=args.formal,
    )
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["provenance"]["command"] = shlex.join(
        sys.argv if argv is None else [sys.argv[0], *argv]
    )
    result["provenance"]["threshold_grid"] = {
        "path": str(grid_path.resolve()),
        "file_sha256": _file_sha256(grid_path),
        "threshold_grid_sha256": threshold_grid_sha256(
            validate_threshold_grid(thresholds)
        ),
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "success": result["success"],
                "reject": result["reject"],
                "threshold_index": result["threshold_index"],
                "threshold": result["threshold"],
                "bsr": result["bsr"],
                "pd": result["pd"],
                "formal_protocol_eligible": result["formal_protocol_eligible"],
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FutureRecord",
    "ProbabilityRecord",
    "audit_frozen_action",
    "build_count_all_curves",
    "conservative_suffix_max",
    "load_probability_records",
    "run_count_all_baseline",
    "select_count_all_threshold",
    "validate_threshold_grid",
]
