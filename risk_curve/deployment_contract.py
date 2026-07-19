"""Independent audits linking a trained risk curve to formal deployment blocks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from .representation import (
    GRID_DETECTOR_PROTOCOL,
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_REPRESENTATION,
    PROBABILITY_REPRESENTATION,
)
from .train_curve_predictor import (
    TRAINING_EPISODE_CONTRACT_SCHEMA_VERSION,
    validate_curve_checkpoint_contract,
)


CHECKPOINT_DEPLOYMENT_AUDIT_SCHEMA_VERSION = (
    "rc-v2-checkpoint-deployment-contract-audit-v1"
)
RAW_LOGIT_CHECKPOINT_DEPLOYMENT_AUDIT_SCHEMA_VERSION = (
    "rc-v4-checkpoint-deployment-contract-audit-v1-raw-logit"
)


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be 64 lowercase hex digits")
    return value


def _distinct_hashes(value: Any, field: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} must be a non-empty sequence")
    hashes = [_sha256(item, field) for item in value]
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"{field} must contain distinct checkpoint hashes")
    return hashes


def _detector_roles(
    all_hashes: Any,
    outer_hash: Any,
    episode_hashes: Any,
) -> tuple[list[str], str, list[str]]:
    hashes = _distinct_hashes(
        all_hashes, "threshold_grid_detector_checkpoint_sha256s"
    )
    outer = _sha256(
        outer_hash, "threshold_grid_outer_detector_checkpoint_sha256"
    )
    episodes = _distinct_hashes(
        episode_hashes,
        "threshold_grid_episode_detector_checkpoint_sha256s",
    )
    if outer not in hashes or outer in episodes:
        raise ValueError("outer detector checkpoint role is invalid")
    if set(hashes) != set(episodes).union({outer}):
        raise ValueError(
            "all detector hashes must equal outer plus episode detector hashes"
        )
    return hashes, outer, episodes


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a positive integer") from error
    if integer <= 0 or integer != value:
        raise ValueError(f"{field} must be a positive integer")
    return integer


def _domain_key(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty domain name")
    key = "".join(character for character in value.casefold() if character.isalnum())
    if not key:
        raise ValueError(f"{field} must contain an alphanumeric domain name")
    return key


def validate_checkpoint_deployment_contract(
    checkpoint: Mapping[str, Any],
    *,
    deployment_provenance: Mapping[str, Any] | None,
    target_dataset: str,
    expected_threshold_grid_sha256: str,
    expected_representation: str | None = None,
    expected_threshold_grid_schema_version: str | None = None,
    expected_threshold_grid_manifest_sha256: str | None = None,
    expected_threshold_grid_detector_protocol: str | None = None,
    expected_threshold_grid_detector_checkpoint_sha256s: (
        list[str] | tuple[str, ...] | None
    ) = None,
    expected_threshold_grid_outer_detector_checkpoint_sha256: str | None = None,
    expected_threshold_grid_episode_detector_checkpoint_sha256s: (
        list[str] | tuple[str, ...] | None
    ) = None,
) -> dict[str, Any]:
    """Validate the semantic unit and domain independence of a curve checkpoint."""

    if not isinstance(checkpoint, Mapping):
        raise ValueError("curve checkpoint must decode to a mapping")
    contract = checkpoint.get("episode_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("curve checkpoint lacks episode_contract")
    if contract.get("schema_version") != TRAINING_EPISODE_CONTRACT_SCHEMA_VERSION:
        raise ValueError("curve checkpoint training-contract schema is unsupported")
    if contract.get("verified") is not True:
        raise ValueError("curve checkpoint training contract is not verified")
    if contract.get("formal_protocol_eligible") is not True:
        reasons = contract.get("ineligibility_reasons")
        raise ValueError(
            "curve checkpoint is not formal-protocol eligible"
            + (f": {reasons}" if reasons else "")
        )
    if contract.get("ineligibility_reasons") not in ([], ()):  # refuse stale flags
        raise ValueError("curve checkpoint records formal ineligibility reasons")

    adaptation_window = _positive_int(
        contract.get("adaptation_window"), "checkpoint adaptation_window"
    )
    evaluation_window = _positive_int(
        contract.get("evaluation_window"), "checkpoint evaluation_window"
    )
    stride = _positive_int(contract.get("stride"), "checkpoint stride")
    if evaluation_window != 1:
        raise ValueError(
            "formal image-level deployment requires a checkpoint trained with "
            "evaluation_window=1"
        )
    if stride != adaptation_window + evaluation_window:
        raise ValueError("checkpoint stride must equal adaptation_window+evaluation_window")
    if contract.get("one_to_one_future_target") is not True:
        raise ValueError("checkpoint does not record one-to-one future risk targets")
    if contract.get("risk_target_unit") != "aggregate_risk_over_1_future_images":
        raise ValueError("checkpoint risk-target unit is not one future image")

    protocol_fields = contract.get("protocol_fields")
    if not isinstance(protocol_fields, Mapping):
        raise ValueError("checkpoint lacks training protocol_fields")
    if protocol_fields.get("protocol") != "causal_adaptation_then_future_evaluation":
        raise ValueError("checkpoint was not trained with the causal A->E protocol")
    for field, expected in (
        ("adaptation_window", adaptation_window),
        ("evaluation_window", evaluation_window),
        ("stride", stride),
    ):
        if protocol_fields.get(field) != expected:
            raise ValueError(f"checkpoint protocol_fields disagree in {field}")

    representation = str(
        checkpoint.get("representation", PROBABILITY_REPRESENTATION)
    )
    if representation not in {
        PROBABILITY_REPRESENTATION,
        LOGIT_REPRESENTATION,
    }:
        raise ValueError("curve checkpoint uses an unsupported representation")
    if expected_representation is not None and representation != expected_representation:
        raise ValueError("checkpoint and deployment representations differ")
    checkpoint_grid_sha = checkpoint.get("threshold_grid_sha256")
    if (
        not isinstance(expected_threshold_grid_sha256, str)
        or len(expected_threshold_grid_sha256) != 64
    ):
        raise ValueError("expected threshold-grid sha256 is invalid")
    if checkpoint_grid_sha != expected_threshold_grid_sha256:
        raise ValueError("checkpoint and deployment threshold grids differ")
    if protocol_fields.get("threshold_grid_sha256") != expected_threshold_grid_sha256:
        raise ValueError("checkpoint training protocol threshold grid differs")

    raw_logit_contract: dict[str, Any] | None = None
    checkpoint_method_name = str(checkpoint.get("method_name") or "risk_curve")
    if representation == LOGIT_REPRESENTATION:
        if checkpoint_method_name == "direct_threshold":
            # Import lazily: the direct contract already depends on the common
            # curve-episode validator, while this module is also used by the
            # RiskCurve selector.
            from .direct_calibrator import validate_direct_checkpoint_contract

            representation_contract = validate_direct_checkpoint_contract(checkpoint)
        else:
            representation_contract = validate_curve_checkpoint_contract(checkpoint)
        if representation_contract.get("formal_v4_eligible") is not True:
            raise ValueError("raw-logit selector checkpoint is not formal-v4 eligible")
        grid_schema = str(checkpoint.get("threshold_grid_schema_version"))
        if grid_schema != LOGIT_GRID_SCHEMA_VERSION:
            raise ValueError("raw-logit checkpoint grid schema is unsupported")
        if (
            expected_threshold_grid_schema_version is not None
            and grid_schema != expected_threshold_grid_schema_version
        ):
            raise ValueError("checkpoint and deployment grid schemas differ")
        grid_manifest_sha = _sha256(
            checkpoint.get("threshold_grid_manifest_sha256"),
            "checkpoint threshold_grid_manifest_sha256",
        )
        if (
            expected_threshold_grid_manifest_sha256 is not None
            and grid_manifest_sha != expected_threshold_grid_manifest_sha256
        ):
            raise ValueError("checkpoint and deployment grid manifests differ")
        detector_protocol = str(
            checkpoint.get("threshold_grid_detector_protocol")
        )
        if detector_protocol != GRID_DETECTOR_PROTOCOL:
            raise ValueError("raw-logit checkpoint grid detector protocol is invalid")
        if (
            expected_threshold_grid_detector_protocol is not None
            and detector_protocol != expected_threshold_grid_detector_protocol
        ):
            raise ValueError("checkpoint and deployment grid detector protocols differ")
        detector_hashes, outer_detector_hash, episode_detector_hashes = _detector_roles(
            checkpoint.get("threshold_grid_detector_checkpoint_sha256s"),
            checkpoint.get("threshold_grid_outer_detector_checkpoint_sha256"),
            checkpoint.get("threshold_grid_episode_detector_checkpoint_sha256s"),
        )
        if expected_threshold_grid_detector_checkpoint_sha256s is not None:
            expected_detector_hashes = _distinct_hashes(
                expected_threshold_grid_detector_checkpoint_sha256s,
                "expected threshold_grid_detector_checkpoint_sha256s",
            )
            if detector_hashes != expected_detector_hashes:
                raise ValueError("checkpoint and deployment grid detector hashes differ")
        if expected_threshold_grid_outer_detector_checkpoint_sha256 is not None and (
            outer_detector_hash
            != _sha256(
                expected_threshold_grid_outer_detector_checkpoint_sha256,
                "expected threshold_grid_outer_detector_checkpoint_sha256",
            )
        ):
            raise ValueError("checkpoint and deployment outer detector hashes differ")
        if expected_threshold_grid_episode_detector_checkpoint_sha256s is not None:
            expected_episode_hashes = _distinct_hashes(
                expected_threshold_grid_episode_detector_checkpoint_sha256s,
                "expected threshold_grid_episode_detector_checkpoint_sha256s",
            )
            if episode_detector_hashes != expected_episode_hashes:
                raise ValueError(
                    "checkpoint and deployment episode detector hashes differ"
                )
        required_bindings = {
            "representation": LOGIT_REPRESENTATION,
            "threshold_grid_schema_version": grid_schema,
            "threshold_grid_sha256": expected_threshold_grid_sha256,
            "threshold_grid_manifest_sha256": grid_manifest_sha,
            "threshold_grid_detector_protocol": detector_protocol,
            "threshold_grid_detector_checkpoint_sha256s": detector_hashes,
            "threshold_grid_outer_detector_checkpoint_sha256": (
                outer_detector_hash
            ),
            "threshold_grid_episode_detector_checkpoint_sha256s": (
                episode_detector_hashes
            ),
        }
        for owner, record in (
            ("checkpoint training protocol", protocol_fields),
            ("deployment provenance", deployment_provenance),
        ):
            if not isinstance(record, Mapping):
                raise ValueError(f"{owner} is required for raw-logit deployment")
            for field, expected in required_bindings.items():
                observed = record.get(field)
                if field == "threshold_grid_detector_checkpoint_sha256s":
                    observed = list(observed or [])
                if observed != expected:
                    raise ValueError(f"{owner} differs in {field}")
        raw_logit_contract = required_bindings
    elif expected_representation == LOGIT_REPRESENTATION:
        raise ValueError("legacy probability checkpoint cannot enter v4 CRC")

    pseudo_targets = protocol_fields.get("pseudo_targets")
    if not isinstance(pseudo_targets, (list, tuple)) or not pseudo_targets:
        raise ValueError("checkpoint does not identify its pseudo-target training domains")
    pseudo_target_names = [str(value) for value in pseudo_targets]
    pseudo_target_keys = {
        _domain_key(value, "checkpoint pseudo-target") for value in pseudo_target_names
    }
    target_key = _domain_key(target_dataset, "deployment target_dataset")
    if target_key in pseudo_target_keys:
        raise ValueError(
            "deployment target_dataset appears in risk-predictor pseudo-target training"
        )

    for split in ("train", "validation"):
        split_contract = contract.get(split)
        if not isinstance(split_contract, Mapping):
            raise ValueError(f"checkpoint lacks nested {split} episode contract")
        if split_contract.get("verified") is not True:
            raise ValueError(f"checkpoint nested {split} contract is unverified")
        if split_contract.get("formal_protocol_eligible") is not True:
            raise ValueError(f"checkpoint nested {split} contract is ineligible")
        for field, expected in (
            ("adaptation_window", adaptation_window),
            ("evaluation_window", evaluation_window),
            ("stride", stride),
        ):
            if split_contract.get(field) != expected:
                raise ValueError(f"checkpoint nested {split} contract disagrees in {field}")

    if deployment_provenance is None or not isinstance(deployment_provenance, Mapping):
        raise ValueError("formal deployment provenance is required")
    for field, expected in (
        ("adaptation_window", adaptation_window),
        ("evaluation_window", evaluation_window),
        ("stride", stride),
    ):
        if deployment_provenance.get(field) != expected:
            raise ValueError(
                f"checkpoint and deployment provenance differ in {field}"
            )
    if deployment_provenance.get("threshold_grid_sha256") != expected_threshold_grid_sha256:
        raise ValueError("deployment provenance threshold grid differs from checkpoint")

    training_reference = {
        "sha256": protocol_fields.get("source_reference_sha256"),
        "domain_names": protocol_fields.get("source_reference_domain_names") or [],
        "statistics_names_sha256": protocol_fields.get(
            "source_reference_statistics_names_sha256"
        ),
    }
    deployment_reference = {
        "sha256": deployment_provenance.get("source_reference_sha256"),
        "domain_names": deployment_provenance.get("source_reference_domain_names") or [],
        "statistics_names_sha256": deployment_provenance.get(
            "source_reference_statistics_names_sha256"
        ),
    }
    if training_reference != deployment_reference:
        raise ValueError(
            "checkpoint and deployment source-reference contracts differ"
        )

    return {
        "schema_version": (
            RAW_LOGIT_CHECKPOINT_DEPLOYMENT_AUDIT_SCHEMA_VERSION
            if representation == LOGIT_REPRESENTATION
            else CHECKPOINT_DEPLOYMENT_AUDIT_SCHEMA_VERSION
        ),
        "verified": True,
        "errors": {},
        "adaptation_window": adaptation_window,
        "evaluation_window": evaluation_window,
        "stride": stride,
        "risk_target_unit": contract.get("risk_target_unit"),
        "checkpoint_deployment_windows_match": True,
        "source_reference_contract_match": True,
        "source_reference": training_reference,
        "target_dataset": target_dataset,
        "pseudo_target_training_domains": pseudo_target_names,
        "target_domain_excluded_from_pseudo_targets": True,
        "checkpoint_method_name": checkpoint_method_name,
        "threshold_grid_sha256": expected_threshold_grid_sha256,
        "representation": representation,
        "threshold_grid_schema_version": (
            raw_logit_contract.get("threshold_grid_schema_version")
            if raw_logit_contract is not None
            else None
        ),
        "threshold_grid_manifest_sha256": (
            raw_logit_contract.get("threshold_grid_manifest_sha256")
            if raw_logit_contract is not None
            else None
        ),
        "threshold_grid_detector_protocol": (
            raw_logit_contract.get("threshold_grid_detector_protocol")
            if raw_logit_contract is not None
            else None
        ),
        "threshold_grid_detector_checkpoint_sha256s": (
            raw_logit_contract.get(
                "threshold_grid_detector_checkpoint_sha256s"
            )
            if raw_logit_contract is not None
            else []
        ),
        "threshold_grid_outer_detector_checkpoint_sha256": (
            raw_logit_contract.get(
                "threshold_grid_outer_detector_checkpoint_sha256"
            )
            if raw_logit_contract is not None
            else None
        ),
        "threshold_grid_episode_detector_checkpoint_sha256s": (
            raw_logit_contract.get(
                "threshold_grid_episode_detector_checkpoint_sha256s"
            )
            if raw_logit_contract is not None
            else []
        ),
        "raw_logit_crc_contract_verified": raw_logit_contract is not None,
    }


def audit_checkpoint_deployment_contract(
    checkpoint_path: str | Path,
    *,
    deployment_provenance: Mapping[str, Any] | None,
    target_dataset: str,
    expected_threshold_grid_sha256: str,
    expected_representation: str | None = None,
    expected_threshold_grid_schema_version: str | None = None,
    expected_threshold_grid_manifest_sha256: str | None = None,
    expected_threshold_grid_detector_protocol: str | None = None,
    expected_threshold_grid_detector_checkpoint_sha256s: (
        list[str] | tuple[str, ...] | None
    ) = None,
    expected_threshold_grid_outer_detector_checkpoint_sha256: str | None = None,
    expected_threshold_grid_episode_detector_checkpoint_sha256s: (
        list[str] | tuple[str, ...] | None
    ) = None,
    expected_curve_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Safely reload and independently audit a persisted curve checkpoint."""

    try:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise ValueError("curve checkpoint artifact is not readable")
        checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_curve_checkpoint_sha256 is not None and (
            checkpoint_sha256
            != _sha256(
                expected_curve_checkpoint_sha256,
                "expected curve checkpoint sha256",
            )
        ):
            raise ValueError("curve checkpoint SHA-256 mismatch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        result = validate_checkpoint_deployment_contract(
            checkpoint,
            deployment_provenance=deployment_provenance,
            target_dataset=target_dataset,
            expected_threshold_grid_sha256=expected_threshold_grid_sha256,
            expected_representation=expected_representation,
            expected_threshold_grid_schema_version=(
                expected_threshold_grid_schema_version
            ),
            expected_threshold_grid_manifest_sha256=(
                expected_threshold_grid_manifest_sha256
            ),
            expected_threshold_grid_detector_protocol=(
                expected_threshold_grid_detector_protocol
            ),
            expected_threshold_grid_detector_checkpoint_sha256s=(
                expected_threshold_grid_detector_checkpoint_sha256s
            ),
            expected_threshold_grid_outer_detector_checkpoint_sha256=(
                expected_threshold_grid_outer_detector_checkpoint_sha256
            ),
            expected_threshold_grid_episode_detector_checkpoint_sha256s=(
                expected_threshold_grid_episode_detector_checkpoint_sha256s
            ),
        )
        result["curve_checkpoint_sha256"] = checkpoint_sha256
        result["curve_checkpoint_sha256_verified"] = (
            expected_curve_checkpoint_sha256 is not None
        )
        return result
    except Exception as error:  # audit boundary: malformed safe-load artifacts reject
        return {
            "schema_version": CHECKPOINT_DEPLOYMENT_AUDIT_SCHEMA_VERSION,
            "verified": False,
            "errors": {"curve_checkpoint_contract": str(error)},
        }
