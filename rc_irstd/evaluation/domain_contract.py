"""Cross-domain provenance checks shared by evaluation and deployment CLIs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def domain_key(value: object) -> str:
    key = "".join(character for character in str(value).casefold() if character.isalnum())
    if key.endswith("sirst") and len(key) > len("sirst"):
        key = key[: -len("sirst")]
    return key


def audit_unseen_target_contract(
    *,
    target_dataset: str,
    score_manifest: Mapping[str, Any],
    calibrator_checkpoint: Mapping[str, Any],
    allow_diagnostic: bool,
) -> list[str]:
    """Return diagnostic reasons or fail closed for a formal unseen target."""

    reasons: list[str] = []
    target = domain_key(target_dataset)
    sources = score_manifest.get("source_datasets")
    if not isinstance(sources, list) or not sources or any(
        not isinstance(value, str) or not value for value in sources
    ):
        reasons.append("detector_source_domains_missing")
    elif target in {domain_key(value) for value in sources}:
        reasons.append("target_seen_by_detector")

    episode_metadata = calibrator_checkpoint.get("episode_metadata")
    calibrator_domains = (
        episode_metadata.get("domain_names")
        if isinstance(episode_metadata, Mapping)
        else None
    )
    if not isinstance(calibrator_domains, (list, tuple)) or not calibrator_domains:
        reasons.append("calibrator_meta_domains_missing")
    elif target in {domain_key(value) for value in calibrator_domains}:
        reasons.append("target_seen_by_calibrator")

    if bool(score_manifest.get("checkpoint_diagnostic_only", False)):
        reasons.append("diagnostic_detector_checkpoint")
    if bool(score_manifest.get("non_strict_state_loading", False)):
        reasons.append("non_strict_detector_state_loading")
    if score_manifest.get("spatial_mode") != "native":
        reasons.append("diagnostic_resized_score_protocol")
    if score_manifest.get("split_role") != "test":
        reasons.append("non_test_target_split")
    if score_manifest.get("split_authority_verified") is not True:
        reasons.append("unverified_split_authority")
    if bool(calibrator_checkpoint.get("diagnostic_only", False)) or not bool(
        calibrator_checkpoint.get("formal_causal_contract", False)
    ):
        reasons.append("diagnostic_calibrator_checkpoint")

    reasons = list(dict.fromkeys(reasons))
    if reasons and not allow_diagnostic:
        raise ValueError(
            "Formal unseen-target contract failed: " + ", ".join(reasons)
        )
    return reasons


__all__ = ["audit_unseen_target_contract", "domain_key"]
