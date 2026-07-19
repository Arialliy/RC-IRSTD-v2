"""Build a formal CRC zero-result from an RC-Direct v4 deployment action.

The adapter reruns RC-Direct from its frozen checkpoint and the label-free
causal deployment-statistics archive.  It does not consume target masks or
manufacture count curves.  Its output is the same canonical zero-result
contract consumed by :mod:`certification.calibrate_target_offset`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from certification.build_calibration_losses import validate_formal_protocol

from .build_deployment_statistics import LOGIT_DEPLOYMENT_STATISTICS_SCHEMA_VERSION
from .deployment_contract import audit_checkpoint_deployment_contract
from .direct_calibrator import RC_DIRECT_SELECTION_SCHEMA_VERSION
from .domain_statistics import LOGIT_STATISTICS_SCHEMA_VERSION
from .representation import (
    GRID_DETECTOR_PROTOCOL,
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_PREDICTION_RULE,
    LOGIT_REPRESENTATION,
    empty_action_contract,
)
from .select_direct_threshold_v4 import build_direct_selection_payload
from .select_zero_label_threshold import (
    SELECTION_DATA_CONTRACT_SCHEMA_VERSION,
    ZERO_RESULT_SCHEMA_VERSION,
    _statistics_from_archive,
)


DIRECT_CRC_ADAPTER_SCHEMA_VERSION = "rc-direct-v4-crc-adapter-v1"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sigmoid_display(value: float | None) -> float | None:
    if value is None:
        return None
    if value >= 0.0:
        return float(1.0 / (1.0 + math.exp(-value)))
    exponential = math.exp(value)
    return float(exponential / (1.0 + exponential))


def _block_ids(path: Path, expected_rows: int) -> list[str]:
    with np.load(path, allow_pickle=False) as archive:
        if "block_ids" not in archive:
            raise ValueError("Formal RC-Direct CRC requires deployment block_ids")
        values = np.asarray(archive["block_ids"])
    if values.ndim != 1 or values.shape[0] != expected_rows:
        raise ValueError("Deployment block_ids do not match statistics rows")
    block_ids = [str(value) for value in values.tolist()]
    if any(not value for value in block_ids) or len(set(block_ids)) != len(block_ids):
        raise ValueError("Deployment block_ids must be unique non-empty strings")
    return block_ids


def _bound_fields(selection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "representation": LOGIT_REPRESENTATION,
        "threshold_grid_schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "threshold_grid_sha256": selection["threshold_grid_sha256"],
        "threshold_grid_manifest_sha256": selection[
            "threshold_grid_manifest_sha256"
        ],
        "threshold_grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "threshold_grid_detector_checkpoint_sha256s": list(
            selection["threshold_grid_detector_checkpoint_sha256s"]
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": selection[
            "threshold_grid_outer_detector_checkpoint_sha256"
        ],
        "threshold_grid_episode_detector_checkpoint_sha256s": list(
            selection["threshold_grid_episode_detector_checkpoint_sha256s"]
        ),
    }


def build_direct_crc_zero_result(
    *,
    checkpoint_path: str | Path,
    statistics_file: str | Path,
    output: str | Path,
    pixel_budget: float,
    component_budget: float,
    device: str = "auto",
    ood_z_threshold: float = 8.0,
) -> Path:
    """Persist a formal, label-free RC-Direct base action for CRC calibration."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    statistics_path = Path(statistics_file).expanduser().resolve()
    direct_selection = build_direct_selection_payload(
        checkpoint_path=checkpoint,
        statistics_file=statistics_path,
        pixel_budget=float(pixel_budget),
        component_budget=float(component_budget),
        device=device,
        ood_z_threshold=float(ood_z_threshold),
    )
    if direct_selection.get("schema_version") != RC_DIRECT_SELECTION_SCHEMA_VERSION:
        raise ValueError("RC-Direct selection schema is unsupported")
    (
        statistics,
        _statistics_names,
        statistics_schema,
        evaluation_rows,
        adaptation_rows,
        protocol,
        protocol_fingerprint,
        archive_evidence,
    ) = _statistics_from_archive(statistics_path)
    if archive_evidence.get("deployment_statistics_schema_version") != (
        LOGIT_DEPLOYMENT_STATISTICS_SCHEMA_VERSION
    ):
        raise ValueError(
            "Formal RC-Direct CRC requires causal raw-logit deployment statistics"
        )
    if statistics_schema != LOGIT_STATISTICS_SCHEMA_VERSION:
        raise ValueError("RC-Direct CRC statistics schema is unsupported")
    identity_contract = archive_evidence.get("identity_contract")
    if not isinstance(identity_contract, Mapping) or (
        identity_contract.get("verified") is not True
        or identity_contract.get("scope") != "causal_blocks"
        or identity_contract.get("computed_one_to_one_evaluation") is not True
    ):
        raise ValueError("RC-Direct CRC requires verified one-to-one causal block identity")
    provenance = archive_evidence.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("masks_read") is not False:
        raise ValueError("RC-Direct CRC requires label-free deployment provenance")
    if not isinstance(protocol, Mapping) or not isinstance(protocol_fingerprint, str):
        raise ValueError("RC-Direct CRC requires a complete deployment protocol")
    canonical_protocol, canonical_fingerprint = validate_formal_protocol(
        protocol, protocol_fingerprint
    )
    if canonical_protocol != dict(protocol) or canonical_fingerprint != protocol_fingerprint:
        raise ValueError("RC-Direct deployment protocol is not canonical")

    binding = _bound_fields(direct_selection)
    for owner, record in (
        ("deployment archive", archive_evidence),
        ("deployment provenance", provenance),
    ):
        for field, expected in binding.items():
            if record.get(field) != expected:
                raise ValueError(f"{owner} differs from RC-Direct in {field}")
    protocol_binding = {
        key: value
        for key, value in binding.items()
        if key != "threshold_grid_manifest_sha256"
    }
    for field, expected in protocol_binding.items():
        if protocol.get(field) != expected:
            raise ValueError(f"deployment protocol differs from RC-Direct in {field}")
    if protocol.get("detector_weight_sha256") != binding[
        "threshold_grid_outer_detector_checkpoint_sha256"
    ]:
        raise ValueError("Deployment scores were not generated by the outer-final detector")

    records = direct_selection.get("records")
    if not isinstance(records, list) or len(records) != int(statistics.shape[0]):
        raise ValueError("RC-Direct action rows do not match deployment statistics")
    block_ids = _block_ids(statistics_path, len(records))
    if [record.get("block_id") for record in records if isinstance(record, Mapping)] != block_ids:
        raise ValueError("RC-Direct actions are not aligned to deployment block IDs")
    if len(adaptation_rows) != len(records) or len(evaluation_rows) != len(records):
        raise ValueError("RC-Direct causal identity rows do not match action rows")
    if any(len(row) != 1 for row in evaluation_rows):
        raise ValueError("Formal RC-Direct CRC requires one future image per action")

    thresholds = np.asarray(direct_selection["thresholds"], dtype=np.float32)
    threshold_indices: list[int | None] = []
    selected_logit_thresholds: list[float | None] = []
    selected_probability_thresholds: list[float | None] = []
    rejects: list[bool] = []
    reject_reasons: list[str | None] = []
    for row_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"RC-Direct action row {row_index} is not an object")
        reject = record.get("reject")
        index = record.get("threshold_index")
        if not isinstance(reject, bool):
            raise ValueError(f"RC-Direct action row {row_index} has invalid reject state")
        if reject:
            if index is not None or record.get("selected_logit_threshold") != "+inf":
                raise ValueError("Rejected RC-Direct action lacks the external +inf action")
            selected_logit = None
        else:
            if isinstance(index, bool) or not isinstance(index, int) or not (
                0 <= index < thresholds.size
            ):
                raise ValueError(f"RC-Direct action row {row_index} has invalid grid index")
            selected_logit = float(thresholds[index])
            if not np.isclose(
                float(record.get("selected_logit_threshold")),
                selected_logit,
                rtol=0.0,
                atol=0.0,
            ):
                raise ValueError("RC-Direct logit action does not match its grid index")
        threshold_indices.append(index)
        selected_logit_thresholds.append(selected_logit)
        selected_probability_thresholds.append(_sigmoid_display(selected_logit))
        rejects.append(reject)
        reject_reasons.append(
            str(record["reject_reason"])
            if record.get("reject_reason") is not None
            else None
        )

    checkpoint_sha = _file_sha256(checkpoint)
    statistics_sha = _file_sha256(statistics_path)
    if direct_selection.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("RC-Direct selection/checkpoint SHA-256 mismatch")
    if direct_selection.get("statistics_file_sha256") != statistics_sha:
        raise ValueError("RC-Direct selection/statistics SHA-256 mismatch")
    checkpoint_audit = audit_checkpoint_deployment_contract(
        checkpoint,
        deployment_provenance=provenance,
        target_dataset=str(protocol.get("target_dataset", "")),
        expected_threshold_grid_sha256=str(binding["threshold_grid_sha256"]),
        expected_representation=LOGIT_REPRESENTATION,
        expected_threshold_grid_schema_version=LOGIT_GRID_SCHEMA_VERSION,
        expected_threshold_grid_manifest_sha256=str(
            binding["threshold_grid_manifest_sha256"]
        ),
        expected_threshold_grid_detector_protocol=GRID_DETECTOR_PROTOCOL,
        expected_threshold_grid_detector_checkpoint_sha256s=list(
            binding["threshold_grid_detector_checkpoint_sha256s"]
        ),
        expected_threshold_grid_outer_detector_checkpoint_sha256=str(
            binding["threshold_grid_outer_detector_checkpoint_sha256"]
        ),
        expected_threshold_grid_episode_detector_checkpoint_sha256s=list(
            binding["threshold_grid_episode_detector_checkpoint_sha256s"]
        ),
        expected_curve_checkpoint_sha256=checkpoint_sha,
    )
    if checkpoint_audit.get("verified") is not True:
        raise ValueError(
            "RC-Direct checkpoint/deployment contract failed: "
            + json.dumps(checkpoint_audit.get("errors", {}), sort_keys=True)
        )

    mapping = {
        evaluation_row[0]: threshold_index
        for evaluation_row, threshold_index in zip(evaluation_rows, threshold_indices)
    }
    flattened_adaptation = [item for row in adaptation_rows for item in row]
    scalar_index = threshold_indices[0] if len(threshold_indices) == 1 else None
    scalar_logit = (
        selected_logit_thresholds[0] if len(selected_logit_thresholds) == 1 else None
    )
    scalar_probability = (
        selected_probability_thresholds[0]
        if len(selected_probability_thresholds) == 1
        else None
    )
    artifact = {
        "source_type": "deployment_statistics_archive",
        "path": str(statistics_path),
        "sha256": statistics_sha,
        "deployment_statistics_schema_version": (
            LOGIT_DEPLOYMENT_STATISTICS_SCHEMA_VERSION
        ),
        "statistics_schema_version": statistics_schema,
        "feature_schema_sha256": archive_evidence.get("feature_schema_sha256"),
        **binding,
        "provenance": dict(provenance),
        "identity_contract": dict(identity_contract),
    }
    selection_data_contract = {
        "schema_version": SELECTION_DATA_CONTRACT_SCHEMA_VERSION,
        "masks_read": False,
        "statistics_computed_from": "causal_adaptation_blocks_A",
        "evaluation_labels_or_masks_used": False,
        "threshold_mapping_rule": "one_A_block_prediction_to_its_future_E_identity",
        "checkpoint_training_contract_verified": True,
        "formal_crc_eligible": True,
        "deployment_identity_contract_verified": True,
        "static_checkpoint_compatibility_verified": False,
        "feature_schema_sha256": archive_evidence.get("feature_schema_sha256"),
        "prediction_rule": LOGIT_PREDICTION_RULE,
        "empty_action": empty_action_contract(),
        **binding,
    }
    payload: dict[str, Any] = {
        "schema_version": ZERO_RESULT_SCHEMA_VERSION,
        "mode": "zero_label_empirical_adaptation",
        "method_name": "direct_threshold",
        "display_name": "RC-Direct + CRC base",
        "role": "baseline",
        "adapter_schema_version": DIRECT_CRC_ADAPTER_SCHEMA_VERSION,
        "threshold": scalar_logit,
        "selected_logit_threshold": scalar_logit,
        "selected_probability_threshold": scalar_probability,
        "threshold_index": scalar_index,
        "reject": any(rejects),
        "reject_reason": (
            reject_reasons[0]
            if len(reject_reasons) == 1
            else ("one_or_more_windows_rejected" if any(rejects) else None)
        ),
        "num_windows": len(records),
        "selected_thresholds": selected_logit_thresholds,
        "selected_logit_thresholds": selected_logit_thresholds,
        "selected_probability_thresholds": selected_probability_thresholds,
        "threshold_indices": threshold_indices,
        "threshold_indices_by_image": mapping,
        "rejects": rejects,
        "reject_reasons": reject_reasons,
        "reject_rate": float(np.mean(rejects)),
        "pixel_budget": float(pixel_budget),
        "component_budget": float(component_budget),
        # The shared CRC audit field keeps its historical name, while the
        # explicit direct fields below prevent any model-family ambiguity.
        "curve_checkpoint": str(checkpoint),
        "curve_checkpoint_sha256": checkpoint_sha,
        "direct_checkpoint": str(checkpoint),
        "direct_checkpoint_sha256": checkpoint_sha,
        "curve_checkpoint_deployment_audit": checkpoint_audit,
        "selection_data_contract": selection_data_contract,
        "masks_read": False,
        "statistics_file": str(statistics_path),
        "statistics_file_sha256": statistics_sha,
        "deployment_statistics_schema_version": (
            LOGIT_DEPLOYMENT_STATISTICS_SCHEMA_VERSION
        ),
        "adaptation_window": provenance.get("adaptation_window"),
        "evaluation_window": provenance.get("evaluation_window"),
        "stride": provenance.get("stride"),
        "statistics_artifact": artifact,
        "protocol": dict(protocol),
        "protocol_fingerprint": protocol_fingerprint,
        "score_map_provenance": dict(provenance),
        "statistics_schema_version": statistics_schema,
        "feature_schema_sha256": archive_evidence.get("feature_schema_sha256"),
        **binding,
        "prediction_rule": LOGIT_PREDICTION_RULE,
        "empty_action": empty_action_contract(),
        "adaptation_protocol": "external_causal_statistics",
        "warmup_window": len(flattened_adaptation),
        "window_ids": flattened_adaptation,
        "adaptation_image_ids": flattened_adaptation,
        "adaptation_ids": adaptation_rows,
        "evaluation_ids": evaluation_rows,
        "thresholds": thresholds.tolist(),
        "target_labels_used": False,
        "deployment_masks_read": False,
        "guarantee": "none before few-shot CRC calibration",
        "direct_selection": direct_selection,
    }
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
    parser.add_argument("--pixel-budget", type=float, required=True)
    parser.add_argument("--component-budget", type=float, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ood-z-threshold", type=float, default=8.0)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_direct_crc_zero_result(
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
