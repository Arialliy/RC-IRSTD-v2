"""Select native raw-logit actions from a formal RC-Direct v4 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from rc_irstd.models.calibrator import MonotoneBudgetCalibrator

from .build_deployment_statistics import (
    LOGIT_DEPLOYMENT_STATISTICS_SCHEMA_VERSION,
    LOGIT_STATIC_CROSS_FIT_STATISTICS_SCHEMA_VERSION,
)
from .direct_calibrator import (
    ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL,
    RC_DIRECT_SELECTION_SCHEMA_VERSION,
    normalise_detector_checkpoint_sha256s,
    quantize_direct_logit_threshold,
    validate_direct_checkpoint_contract,
    validate_detector_role_contract,
)
from .domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    statistics_names_sha256,
    validate_statistics_names,
)
from .representation import (
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_PREDICTION_RULE,
    LOGIT_REPRESENTATION,
    empty_action_contract,
    logit_threshold_grid_sha256,
    validate_logit_threshold_grid,
)


def _scalar(value: Any, field: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{field} must be a scalar")
    return str(array.item())


def _sigmoid_display(value: float) -> float:
    if value >= 0.0:
        return float(1.0 / (1.0 + math.exp(-value)))
    exponential = math.exp(value)
    return float(exponential / (1.0 + exponential))


def _find_budget_index(
    pixel_budgets: np.ndarray,
    component_budgets: np.ndarray,
    pixel_budget: float,
    component_budget: float,
) -> int:
    if (
        not math.isfinite(pixel_budget)
        or not math.isfinite(component_budget)
        or pixel_budget <= 0.0
        or component_budget <= 0.0
    ):
        raise ValueError("Requested RC-Direct budgets must be finite and positive")
    match = np.flatnonzero(
        np.isclose(pixel_budgets, pixel_budget, rtol=1e-6, atol=0.0)
        & np.isclose(component_budgets, component_budget, rtol=1e-6, atol=0.0)
    )
    if match.size != 1:
        raise ValueError(
            "RC-Direct v4 selects only registered joint budget pairs; "
            "no unique checkpoint pair matches the request"
        )
    return int(match[0])


def _load_deployment_statistics(
    path: Path,
    *,
    checkpoint: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], list[str], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Deployment statistics do not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "statistics",
            "statistics_names",
            "statistics_names_sha256",
            "statistics_schema_version",
            "feature_schema_sha256",
            "representation",
            "thresholds",
            "threshold_grid_schema_version",
            "threshold_grid_sha256",
            "threshold_grid_manifest_sha256",
            "threshold_grid_detector_protocol",
            "threshold_grid_detector_checkpoint_sha256s",
            "threshold_grid_outer_detector_checkpoint_sha256",
            "threshold_grid_episode_detector_checkpoint_sha256s",
            "provenance_json",
            "protocol_json",
            "protocol_fingerprint",
            "deployment_statistics_schema_version",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(
                "RC-Direct deployment statistics are missing: " + ", ".join(missing)
            )
        forbidden = {
            "pixel_log_risk",
            "component_log_risk",
            "component_log_risk_upper",
            "pd_curve",
            "mask",
        }.intersection(archive.files)
        if forbidden:
            raise ValueError(
                "RC-Direct selection refuses labelled/risk-target archives: "
                + ", ".join(sorted(forbidden))
            )
        statistics = np.asarray(archive["statistics"], dtype=np.float32)
        names = validate_statistics_names(
            archive["statistics_names"], expected_dim=statistics.shape[-1]
        )
        recorded_names_hash = _scalar(
            archive["statistics_names_sha256"], "statistics_names_sha256"
        )
        representation = _scalar(archive["representation"], "representation")
        deployment_schema = _scalar(
            archive["deployment_statistics_schema_version"],
            "deployment_statistics_schema_version",
        )
        statistics_schema = _scalar(
            archive["statistics_schema_version"], "statistics_schema_version"
        )
        feature_hash = _scalar(archive["feature_schema_sha256"], "feature_schema_sha256")
        grid_schema = _scalar(
            archive["threshold_grid_schema_version"],
            "threshold_grid_schema_version",
        )
        grid_hash = _scalar(archive["threshold_grid_sha256"], "threshold_grid_sha256")
        grid_manifest_hash = _scalar(
            archive["threshold_grid_manifest_sha256"],
            "threshold_grid_manifest_sha256",
        )
        detector_protocol = _scalar(
            archive["threshold_grid_detector_protocol"],
            "threshold_grid_detector_protocol",
        )
        (
            detector_checkpoint_sha256s,
            outer_detector_checkpoint_sha256,
            episode_detector_checkpoint_sha256s,
        ) = validate_detector_role_contract(
            archive["threshold_grid_detector_checkpoint_sha256s"],
            archive["threshold_grid_outer_detector_checkpoint_sha256"],
            archive["threshold_grid_episode_detector_checkpoint_sha256s"],
        )
        thresholds = validate_logit_threshold_grid(
            np.asarray(archive["thresholds"], dtype=np.float32)
        )
        try:
            provenance = json.loads(
                _scalar(archive["provenance_json"], "provenance_json")
            )
        except json.JSONDecodeError as error:
            raise ValueError("Deployment provenance_json is invalid") from error
        try:
            protocol_text = _scalar(archive["protocol_json"], "protocol_json")
            protocol = json.loads(protocol_text)
        except json.JSONDecodeError as error:
            raise ValueError("Deployment protocol_json is invalid") from error
        recorded_protocol_fingerprint = _scalar(
            archive["protocol_fingerprint"], "protocol_fingerprint"
        )
        evaluation_ids = (
            [str(value) for value in np.asarray(archive["block_ids"]).tolist()]
            if "block_ids" in archive
            else [str(index) for index in range(int(statistics.shape[0]))]
        )
    if statistics.ndim == 1:
        statistics = statistics[None, :]
    if statistics.ndim != 2 or min(statistics.shape) <= 0:
        raise ValueError("Deployment statistics must have shape [N,D]")
    if not np.isfinite(statistics).all():
        raise ValueError("Deployment statistics contain NaN or infinity")
    if representation != LOGIT_REPRESENTATION:
        raise ValueError("RC-Direct v4 deployment requires raw_logit_float32")
    if deployment_schema not in {
        LOGIT_DEPLOYMENT_STATISTICS_SCHEMA_VERSION,
        LOGIT_STATIC_CROSS_FIT_STATISTICS_SCHEMA_VERSION,
    }:
        raise ValueError("RC-Direct deployment statistics artifact schema mismatch")
    if statistics_schema != LOGIT_STATISTICS_SCHEMA_VERSION:
        raise ValueError("RC-Direct deployment statistics schema mismatch")
    if grid_schema != LOGIT_GRID_SCHEMA_VERSION:
        raise ValueError("RC-Direct deployment grid schema mismatch")
    if grid_hash != logit_threshold_grid_sha256(thresholds):
        raise ValueError("Deployment statistics semantic threshold-grid hash mismatch")
    if grid_hash != checkpoint["threshold_grid_sha256"]:
        raise ValueError("Deployment/checkpoint threshold grids differ")
    if grid_manifest_hash != checkpoint["threshold_grid_manifest_sha256"]:
        raise ValueError("Deployment/checkpoint grid-manifest hashes differ")
    if detector_protocol != ALL_SOURCE_DETECTOR_FOLDS_PROTOCOL:
        raise ValueError("Deployment grid omits one or more source-only detector folds")
    if detector_protocol != checkpoint["threshold_grid_detector_protocol"]:
        raise ValueError("Deployment/checkpoint detector-grid protocols differ")
    if detector_checkpoint_sha256s != normalise_detector_checkpoint_sha256s(
        checkpoint["threshold_grid_detector_checkpoint_sha256s"]
    ):
        raise ValueError("Deployment/checkpoint detector-grid checkpoint sets differ")
    if outer_detector_checkpoint_sha256 != checkpoint[
        "threshold_grid_outer_detector_checkpoint_sha256"
    ]:
        raise ValueError("Deployment/checkpoint outer detector roles differ")
    if episode_detector_checkpoint_sha256s != normalise_detector_checkpoint_sha256s(
        checkpoint["threshold_grid_episode_detector_checkpoint_sha256s"],
        field="threshold_grid_episode_detector_checkpoint_sha256s",
    ):
        raise ValueError("Deployment/checkpoint episode detector roles differ")
    checkpoint_names = tuple(str(value) for value in checkpoint["statistics_names"])
    if names != checkpoint_names:
        raise ValueError("Deployment/checkpoint ordered feature schemas differ")
    if recorded_names_hash != statistics_names_sha256(names):
        raise ValueError("Deployment statistics_names_sha256 is invalid")
    if recorded_names_hash != checkpoint["statistics_names_sha256"]:
        raise ValueError("Deployment statistic names do not match checkpoint hash")
    if feature_hash != checkpoint["feature_schema_sha256"]:
        raise ValueError("Deployment/checkpoint feature-schema hashes differ")
    if not isinstance(provenance, dict) or provenance.get("masks_read") is not False:
        raise ValueError("RC-Direct deployment must prove masks_read=false")
    if not isinstance(protocol, dict):
        raise ValueError("Deployment protocol_json must decode to an object")
    expected_protocol_fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if recorded_protocol_fingerprint != expected_protocol_fingerprint:
        raise ValueError("Deployment score protocol fingerprint mismatch")
    score_detector_checkpoint_sha256 = str(
        protocol.get("detector_weight_sha256", "")
    )
    if score_detector_checkpoint_sha256 != outer_detector_checkpoint_sha256:
        raise ValueError(
            "Deployment score detector must equal the frozen outer detector checkpoint"
        )
    protocol_bindings = {
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": grid_hash,
        "threshold_grid_detector_protocol": detector_protocol,
        "threshold_grid_detector_checkpoint_sha256s": list(
            detector_checkpoint_sha256s
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": (
            outer_detector_checkpoint_sha256
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": list(
            episode_detector_checkpoint_sha256s
        ),
    }
    for field, expected in protocol_bindings.items():
        if protocol.get(field) != expected:
            raise ValueError(f"Deployment score protocol {field} mismatch")
    provenance_bindings = {
        **protocol_bindings,
        "threshold_grid_manifest_sha256": grid_manifest_hash,
        "feature_schema_sha256": feature_hash,
    }
    for field, expected in provenance_bindings.items():
        if provenance.get(field) != expected:
            raise ValueError(f"Deployment statistics provenance {field} mismatch")
    episode_contract = checkpoint["episode_contract"]
    for field in ("adaptation_window", "evaluation_window", "stride"):
        if int(provenance.get(field, -1)) != int(episode_contract.get(field, -2)):
            raise ValueError(
                f"Deployment {field} differs from the RC-Direct training contract"
            )
    protocol_fields = episode_contract.get("protocol_fields", {})
    for field in (
        "source_reference_sha256",
        "source_reference_statistics_names_sha256",
    ):
        if provenance.get(field) != protocol_fields.get(field):
            raise ValueError(f"Deployment/training {field} contracts differ")
    return statistics, provenance, evaluation_ids, score_detector_checkpoint_sha256


def build_direct_selection_payload(
    *,
    checkpoint_path: str | Path,
    statistics_file: str | Path,
    pixel_budget: float,
    component_budget: float,
    device: str = "auto",
    ood_z_threshold: float = 8.0,
) -> dict[str, Any]:
    """Return a fully validated RC-Direct selection without writing an artifact."""

    if not math.isfinite(ood_z_threshold) or ood_z_threshold <= 0.0:
        raise ValueError("ood_z_threshold must be finite and positive")
    device_name = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if device_name == "auto":
        device_name = "cpu"
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch_device = torch.device(device_name)
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"RC-Direct checkpoint does not exist: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    contract = validate_direct_checkpoint_contract(checkpoint)
    statistics_path = Path(statistics_file).expanduser().resolve()
    statistics, provenance, block_ids, score_detector_checkpoint_sha256 = (
        _load_deployment_statistics(
            statistics_path, checkpoint=checkpoint
        )
    )
    budget_index = _find_budget_index(
        contract["pixel_budgets"],
        contract["component_budgets"],
        float(pixel_budget),
        float(component_budget),
    )
    model = MonotoneBudgetCalibrator(**checkpoint["model_config"]).to(torch_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    mean = np.asarray(checkpoint["statistics_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["statistics_std"], dtype=np.float32)
    normalised = (statistics - mean[None, :]) / np.maximum(std[None, :], 1e-6)
    is_ood = np.max(np.abs(normalised), axis=1) > float(ood_z_threshold)
    with torch.no_grad():
        prediction = model(
            torch.from_numpy(statistics).to(torch_device)
        ).grid_logits[:, budget_index].cpu().numpy()
    thresholds = validate_logit_threshold_grid(
        np.asarray(checkpoint["thresholds"], dtype=np.float32)
    )
    records: list[dict[str, object]] = []
    for row, predicted_logit in enumerate(prediction.tolist()):
        decoded = quantize_direct_logit_threshold(predicted_logit, thresholds)
        reject_reason: str | None = "model_above_finite_grid" if decoded.reject else None
        if bool(is_ood[row]):
            decoded = quantize_direct_logit_threshold(float(thresholds[-1]) + 1.0, thresholds)
            reject_reason = "feature_ood"
        records.append(
            {
                "block_id": block_ids[row],
                "budget_index": budget_index,
                "predicted_logit_threshold": float(predicted_logit),
                "selected_logit_threshold": (
                    "+inf" if decoded.reject else decoded.selected_logit_threshold
                ),
                "threshold_index": decoded.threshold_index,
                "probability_threshold": (
                    None
                    if decoded.reject
                    else _sigmoid_display(decoded.selected_logit_threshold)
                ),
                "reject": decoded.reject,
                "reject_reason": reject_reason,
                "max_abs_feature_z": float(np.max(np.abs(normalised[row]))),
            }
        )
    payload = {
        "schema_version": RC_DIRECT_SELECTION_SCHEMA_VERSION,
        "method_name": "direct_threshold",
        "display_name": "RC-Direct",
        "role": "baseline",
        "representation": LOGIT_REPRESENTATION,
        "prediction_rule": LOGIT_PREDICTION_RULE,
        "empty_action": empty_action_contract(),
        "thresholds": thresholds.tolist(),
        "pixel_budget": float(pixel_budget),
        "component_budget": float(component_budget),
        "budget_index": budget_index,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": contract["threshold_grid_sha256"],
        "threshold_grid_manifest_sha256": contract[
            "threshold_grid_manifest_sha256"
        ],
        "threshold_grid_detector_protocol": contract[
            "threshold_grid_detector_protocol"
        ],
        "threshold_grid_detector_checkpoint_sha256s": list(
            contract["threshold_grid_detector_checkpoint_sha256s"]
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": contract[
            "threshold_grid_outer_detector_checkpoint_sha256"
        ],
        "threshold_grid_episode_detector_checkpoint_sha256s": list(
            contract["threshold_grid_episode_detector_checkpoint_sha256s"]
        ),
        "deployment_score_detector_checkpoint_sha256": (
            score_detector_checkpoint_sha256
        ),
        "feature_schema_sha256": contract["feature_schema_sha256"],
        "model_architecture_version": contract["model_architecture_version"],
        "checkpoint": str(checkpoint_file),
        "checkpoint_sha256": hashlib.sha256(checkpoint_file.read_bytes()).hexdigest(),
        "statistics_file": str(statistics_path),
        "statistics_file_sha256": hashlib.sha256(statistics_path.read_bytes()).hexdigest(),
        "target_labels_used": False,
        "deployment_masks_read": provenance["masks_read"],
        "ood_z_threshold": float(ood_z_threshold),
        "num_actions": len(records),
        "num_rejects": sum(int(record["reject"]) for record in records),
        "records": records,
    }
    return payload


def select_direct_thresholds(
    *,
    checkpoint_path: str | Path,
    statistics_file: str | Path,
    output: str | Path,
    pixel_budget: float,
    component_budget: float,
    device: str = "auto",
    ood_z_threshold: float = 8.0,
) -> Path:
    payload = build_direct_selection_payload(
        checkpoint_path=checkpoint_path,
        statistics_file=statistics_file,
        pixel_budget=pixel_budget,
        component_budget=component_budget,
        device=device,
        ood_z_threshold=ood_z_threshold,
    )
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--statistics-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pixel-budget", type=float, required=True)
    parser.add_argument("--component-budget", type=float, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ood-z-threshold", type=float, default=8.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    select_direct_thresholds(
        checkpoint_path=args.checkpoint,
        statistics_file=args.statistics_file,
        output=args.output,
        pixel_budget=args.pixel_budget,
        component_budget=args.component_budget,
        device=args.device,
        ood_z_threshold=args.ood_z_threshold,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
