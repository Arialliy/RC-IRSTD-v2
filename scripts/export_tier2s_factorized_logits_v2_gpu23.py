"""Export source-only Tier2S base/final/residual raw-logit evidence.

This exporter is deliberately independent from the frozen Tier2R execution
chain.  It accepts only the fixed-last Tier2R ``control`` and ``c`` v2
checkpoints, and only the two registered source datasets.  The canonical
MSHNet base logits and fusion result are captured directly from one
``fusion_head.forward`` hook with keyword arguments; the final model output is
then required to be bitwise identical to the captured fusion output.

No outer-target path is accepted and this diagnostic cannot authorize a new
training stage or an outer-target evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_ext.dataset_meta import crop_to_valid, meta_to_jsonable
from data_ext.eval_dataset import IRSTDEvalDataset
from data_ext.split_utils import ensure_unique_sample_ids, read_split_file
from evaluation.artifact_integrity import (
    SCORE_RECORD_INTEGRITY_SCHEMA,
    file_sha256,
    ordered_ids_sha256,
    score_records_sha256,
)
from evaluation.export_score_maps import (
    _extract_logits,
    _load_checkpoint_safely,
    _safe_record_name,
)
from rc_irstd.models import (
    RC_MSHNET_ARCHITECTURE_VERSION_V2,
    build_mshnet,
)
from scripts.diagnose_component_fusion import validate_source_dataset_path
from scripts import register_tier2s_gpu23_amendment_v2 as governance_registrar


ARCHITECTURE_VERSION = RC_MSHNET_ARCHITECTURE_VERSION_V2
SCHEMA_VERSION = "rc-irstd-tier2s-factorized-logit-export-v2-gpu23"
PROTOCOL_ID = "tier2s_factorized_causal_audit_v2_gpu23"
PROTOCOL_SCHEMA = (
    "rc-irstd-aaai27-tier2s-factorized-causal-audit-protocol-v2-gpu23"
)
STREAM_HASH_SCHEMA = "rc-irstd-tier2s-factorized-stream-sha256-v2-gpu23"
SCORE_REPRESENTATION = "base+final+residual_raw_logit_float32"
ALLOWED_SOURCES = frozenset(("NUDT-SIRST", "IRSTD-1K"))
ALLOWED_ROLES = frozenset(("control", "c"))
ALLOWED_SEEDS = frozenset((43, 44, 45))
FOLD_CONTRACTS: dict[str, dict[str, str]] = {
    "heldout_nudt": {
        "training_source": "IRSTD-1K",
        "held_out_source": "NUDT-SIRST",
    },
    "heldout_irstd": {
        "training_source": "NUDT-SIRST",
        "held_out_source": "IRSTD-1K",
    },
}
ROLE_CONTRACTS: dict[str, dict[str, bool]] = {
    "control": {
        "use_contrast": False,
        "use_component_context": False,
        "use_component_expert": False,
        "use_risk_gate": False,
    },
    "c": {
        "use_contrast": True,
        "use_component_context": False,
        "use_component_expert": False,
        "use_risk_gate": True,
    },
}
TIER2R_ROOT_RELATIVE = Path(
    "outputs/aaai27/detectors/component_rescue/tier2r_c_v1"
)
TIER2S_OUTPUT_RELATIVE = Path(
    "outputs/aaai27/source_rescue/tier2s_factorized_causal_audit_v2_gpu23"
)
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/tier2s_factorized_causal_audit_v2_gpu23.json"
TIER2S_AUDIT_RELATIVE = Path(
    "artifacts/aaai27/audit/source_rescue/tier2s_factorized_causal_audit_v2_gpu23"
)
DEFAULT_PREREGISTRATION = PROJECT_ROOT / TIER2S_AUDIT_RELATIVE / "PREREGISTRATION.json"
PREREGISTRATION_SCHEMA = (
    "rc-irstd-aaai27-tier2s-factorized-preregistration-v2-gpu23"
)
PREREGISTRATION_BINDING_SCHEMA = (
    "rc-irstd-aaai27-tier2s-preregistration-binding-v2-gpu23"
)
UNIT_TEST_UNBOUND_GOVERNANCE_BINDING: dict[str, Any] = {
    "schema_version": "rc-irstd-tier2s-unit-test-unbound-governance-v1",
    "bound": False,
}
UNIT_TEST_UNBOUND_PREREGISTRATION_BINDING: dict[str, Any] = {
    "schema_version": "rc-irstd-tier2s-unit-test-unbound-preregistration-v1",
    "bound": False,
}


@dataclass(frozen=True)
class Tier2SProtocolBinding:
    path: Path
    sha256: str
    protocol_id: str = PROTOCOL_ID

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "protocol_id": self.protocol_id,
        }


@dataclass(frozen=True)
class Tier2SCheckpointBinding:
    checkpoint_path: Path
    checkpoint_sha256: str
    formal_config_path: Path
    formal_config_sha256: str
    seed: int
    role: str
    fold: str
    training_source: str
    held_out_source: str
    training_split_file: Path
    training_split_file_sha256: str
    training_ordered_ids_sha256: str
    num_training_samples: int
    epoch: int

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "formal_config_path": str(self.formal_config_path),
            "formal_config_sha256": self.formal_config_sha256,
            "seed": self.seed,
            "role": self.role,
            "fold": self.fold,
            "training_source": self.training_source,
            "held_out_source": self.held_out_source,
            "training_split_file": str(self.training_split_file),
            "training_split_file_sha256": self.training_split_file_sha256,
            "training_ordered_ids_sha256": self.training_ordered_ids_sha256,
            "num_training_samples": self.num_training_samples,
            "checkpoint_selection": "fixed_last",
            "epoch": self.epoch,
            "architecture_version": ARCHITECTURE_VERSION,
        }


@dataclass(frozen=True)
class FactorizedLogitBatch:
    base_logits: torch.Tensor
    final_logits: torch.Tensor
    residual_logits: torch.Tensor
    replay_max_abs_error: float
    replay_mean_abs_error: float
    model_output_bitwise_equal: bool
    capture_source: str = "fusion_head.forward_kwargs.base_logits+output[0]"


def _lexical_absolute(path: str | Path, *, name: str) -> Path:
    raw = Path(path).expanduser()
    if ".." in raw.parts:
        raise RuntimeError(f"{name} contains forbidden parent traversal")
    return Path(os.path.abspath(raw))


def _contains_outer_target(value: str | Path) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    return "nuaa" in normalized


def _assert_path_under_anchor(
    path: str | Path,
    *,
    anchor: str | Path,
    name: str,
    require_exists: bool,
) -> Path:
    candidate = _lexical_absolute(path, name=name)
    root = _lexical_absolute(anchor, name=f"{name}_anchor")
    if candidate != root and root not in candidate.parents:
        raise RuntimeError(f"{name} escapes its allowlisted root: {candidate}")
    current = root
    parts = () if candidate == root else candidate.relative_to(root).parts
    for component in (root, *(root.joinpath(*parts[: index + 1]) for index in range(len(parts)))):
        current = component
        if current.is_symlink():
            raise RuntimeError(f"{name} contains a symlink: {current}")
        if not current.exists():
            break
    if require_exists and not candidate.exists():
        raise FileNotFoundError(f"{name} does not exist: {candidate}")
    return candidate


def _require_regular_read_only(path: Path, *, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{name} must be a regular non-symlink file: {path}")
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise RuntimeError(f"{name} must be frozen read-only: {path}")


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must decode to a JSON object")
    return value


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return value


def _strict_equal(
    mapping: Mapping[str, Any],
    key: str,
    expected: Any,
    *,
    where: str,
) -> None:
    actual = mapping.get(key)
    if actual != expected or (
        isinstance(expected, bool) and type(actual) is not bool
    ):
        raise RuntimeError(
            f"{where}.{key} mismatch: expected {expected!r}, got {actual!r}"
        )


def _validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _load_strict_json_object(path: Path, *, name: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"{name} contains non-standard JSON constant {token!r}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{name} must decode to a JSON object")
    return value


def _binding_copy(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        raw = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        result = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{name} is not a canonical JSON mapping") from error
    if not isinstance(result, dict):
        raise RuntimeError(f"{name} must be a mapping")
    return result


def _manifest_consumer_bindings(
    governance_binding: Mapping[str, Any] | None,
    tier2s_preregistration_binding: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if governance_binding is None and tier2s_preregistration_binding is None:
        return (
            dict(UNIT_TEST_UNBOUND_GOVERNANCE_BINDING),
            dict(UNIT_TEST_UNBOUND_PREREGISTRATION_BINDING),
        )
    if governance_binding is None or tier2s_preregistration_binding is None:
        raise RuntimeError(
            "governance and Tier2S preregistration bindings must be supplied together"
        )
    return (
        _binding_copy(governance_binding, name="governance_binding"),
        _binding_copy(
            tier2s_preregistration_binding,
            name="tier2s_preregistration_binding",
        ),
    )


def require_frozen_tier2s_consumer_bindings(
    *,
    expected_governance_registration_sha256: str,
    tier2s_preregistration_path: str | Path,
    expected_tier2s_preregistration_sha256: str,
    project_root: str | Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return exact frozen bindings required by every production export.

    The governance registrar remains the authority for the registration,
    success contract, fresh-seed ledger, code closure, and authorization
    semantics. This consumer additionally binds the canonical Tier2S
    preregistration and both SHA-256 sidecars before model or data access.
    """

    registration_sha = _validate_sha256(
        expected_governance_registration_sha256,
        name="expected_governance_registration_sha256",
    )
    preregistration_sha = _validate_sha256(
        expected_tier2s_preregistration_sha256,
        name="expected_tier2s_preregistration_sha256",
    )
    governance_binding = governance_registrar.require_frozen_tier2s_governance(
        expected_registration_sha256=registration_sha
    )
    governance = _binding_copy(governance_binding, name="governance_binding")
    registration = _mapping(
        governance.get("registration"), name="governance_binding.registration"
    )
    if (
        governance.get("schema_version")
        != "rc-irstd-aaai27-tier2s-governance-binding-v2-gpu23"
        or registration.get("sha256") != registration_sha
        or governance.get("physical_gpus") != [2, 3]
        or governance.get("container_logical_ordinals") != [0, 1]
        or governance.get("physical_to_container_ordinal") != {"2": 0, "3": 1}
        or governance.get("gpu_fallback_allowed") is not False
        or governance.get("tier2s_source_only_diagnostic_authorized") is not True
        or governance.get("formal_v3_model_training_authorized") is not False
        or governance.get("source_gate_a_authorized") is not False
        or governance.get("riskcurve_authorized") is not False
        or governance.get("outer_target_access_authorized") is not False
    ):
        raise RuntimeError("frozen governance binding semantics drifted")

    root = _lexical_absolute(project_root, name="project_root")
    expected_path = root / TIER2S_AUDIT_RELATIVE / "PREREGISTRATION.json"
    preregistration = _assert_path_under_anchor(
        tier2s_preregistration_path,
        anchor=root,
        name="tier2s_preregistration",
        require_exists=True,
    )
    if preregistration != expected_path:
        raise RuntimeError(
            f"Tier2S preregistration must use the canonical path: {expected_path}"
        )
    sidecar = preregistration.with_suffix(preregistration.suffix + ".sha256")
    _require_regular_read_only(preregistration, name="Tier2S preregistration")
    _require_regular_read_only(sidecar, name="Tier2S preregistration sidecar")
    if file_sha256(preregistration) != preregistration_sha:
        raise RuntimeError("Tier2S preregistration SHA-256 drifted")
    expected_sidecar = f"{preregistration_sha}  {preregistration.name}\n"
    if sidecar.read_text(encoding="ascii") != expected_sidecar:
        raise RuntimeError("Tier2S preregistration SHA-256 sidecar drifted")
    payload = _load_strict_json_object(
        preregistration, name="Tier2S preregistration"
    )
    if (
        payload.get("schema_version") != PREREGISTRATION_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("research_mode") != "exploratory_source_only"
        or payload.get("source_only") is not True
        or payload.get("outer_target_access_authorized") is not False
        or payload.get("outer_target_images_used") is not False
        or payload.get("outer_target_labels_used") is not False
        or payload.get("source_tier3_authorized") is not False
        or payload.get("paper_claim_authorized") is not False
        or payload.get("governance_binding") != governance
    ):
        raise RuntimeError("Tier2S preregistration governance/source-only drift")

    binding = {
        "schema_version": PREREGISTRATION_BINDING_SCHEMA,
        "path": str(preregistration.relative_to(root)),
        "sha256": preregistration_sha,
        "sidecar_path": str(sidecar.relative_to(root)),
        "sidecar_sha256": file_sha256(sidecar),
        "protocol_id": PROTOCOL_ID,
        "governance_registration_sha256": registration_sha,
    }
    if file_sha256(preregistration) != preregistration_sha:
        raise RuntimeError("Tier2S preregistration changed during verification")
    return governance, binding


def validate_protocol(
    protocol_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> Tier2SProtocolBinding:
    root = _lexical_absolute(project_root, name="project_root")
    expected = root / "configs/tier2s_factorized_causal_audit_v2_gpu23.json"
    protocol = _assert_path_under_anchor(
        protocol_path,
        anchor=root,
        name="protocol",
        require_exists=True,
    )
    if protocol != expected:
        raise RuntimeError(f"Tier2S protocol must use the canonical path: {expected}")
    if protocol.is_symlink() or not protocol.is_file():
        raise RuntimeError("Tier2S protocol must be a regular non-symlink file")
    payload = _load_json_object(protocol, name="Tier2S protocol")
    _strict_equal(payload, "schema_version", PROTOCOL_SCHEMA, where="protocol")
    _strict_equal(payload, "protocol_id", PROTOCOL_ID, where="protocol")
    _strict_equal(payload, "research_mode", "exploratory_source_only", where="protocol")
    limits = _mapping(payload.get("scientific_limits"), name="scientific_limits")
    for key in (
        "paper_claim_authorized",
        "source_tier3_authorized",
        "outer_target_access_authorized",
        "outer_target_images_used",
        "outer_target_labels_used",
    ):
        _strict_equal(limits, key, False, where="scientific_limits")
    if payload.get("roles") != ["control", "c"]:
        raise RuntimeError("Tier2S protocol role set drifted")
    if payload.get("seeds") != [43, 44, 45]:
        raise RuntimeError("Tier2S protocol seed set drifted")
    folds = _mapping(payload.get("folds"), name="protocol.folds")
    if set(folds) != set(FOLD_CONTRACTS):
        raise RuntimeError("Tier2S protocol fold set drifted")
    for fold, expected_fold in FOLD_CONTRACTS.items():
        value = _mapping(folds[fold], name=f"protocol.folds.{fold}")
        for key, expected_value in expected_fold.items():
            _strict_equal(value, key, expected_value, where=f"protocol.folds.{fold}")
    checkpoint = _mapping(
        payload.get("checkpoint_binding"), name="protocol.checkpoint_binding"
    )
    _strict_equal(
        checkpoint,
        "root",
        str(TIER2R_ROOT_RELATIVE),
        where="protocol.checkpoint_binding",
    )
    _strict_equal(
        checkpoint,
        "selection",
        "fixed_last_epoch_79",
        where="protocol.checkpoint_binding",
    )
    _strict_equal(
        checkpoint,
        "required_architecture_version",
        ARCHITECTURE_VERSION,
        where="protocol.checkpoint_binding",
    )
    required_roles = _mapping(
        checkpoint.get("required_roles"),
        name="protocol.checkpoint_binding.required_roles",
    )
    if set(required_roles) != set(ROLE_CONTRACTS):
        raise RuntimeError("Tier2S protocol checkpoint roles drifted")
    for role, flags in ROLE_CONTRACTS.items():
        configured = _mapping(
            required_roles[role], name=f"protocol.required_roles.{role}"
        )
        for key, expected_value in flags.items():
            _strict_equal(
                configured,
                key,
                expected_value,
                where=f"protocol.required_roles.{role}",
            )
    data_access = _mapping(payload.get("data_access"), name="protocol.data_access")
    if data_access.get("allowed_source_roots") != [
        "datasets/NUDT-SIRST",
        "datasets/IRSTD-1K",
    ]:
        raise RuntimeError("Tier2S source-root allowlist drifted")
    _strict_equal(data_access, "official_split", "train", where="protocol.data_access")
    execution = _mapping(payload.get("execution"), name="protocol.execution")
    if (
        execution.get("physical_gpus") != [2, 3]
        or execution.get("container_logical_ordinals") != {"2": 0, "3": 1}
        or execution.get("container_must_expose_only_physical_gpus") != [2, 3]
        or execution.get("allow_gpu_fallback") is not False
        or execution.get("export_jobs") != 18
    ):
        raise RuntimeError("Tier2S GPU2/3 execution contract drifted")

    parent = _mapping(
        payload.get("immutable_parent_evidence"),
        name="protocol.immutable_parent_evidence",
    )
    for evidence_name, evidence in parent.items():
        item = _mapping(evidence, name=f"immutable_parent_evidence.{evidence_name}")
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise RuntimeError(f"Unsafe parent evidence path: {evidence_name}")
        evidence_path = _assert_path_under_anchor(
            root / relative,
            anchor=root,
            name=f"parent_evidence_{evidence_name}",
            require_exists=True,
        )
        if evidence_path.is_symlink() or not evidence_path.is_file():
            raise RuntimeError(f"Invalid parent evidence file: {evidence_path}")
        expected_sha = _validate_sha256(
            item.get("sha256"), name=f"{evidence_name}.sha256"
        )
        if file_sha256(evidence_path) != expected_sha:
            raise RuntimeError(f"Parent evidence SHA-256 drifted: {evidence_name}")
    return Tier2SProtocolBinding(
        path=protocol,
        sha256=file_sha256(protocol),
    )


def validate_source_dataset_root(
    path: str | Path,
    *,
    dataset_name: str,
    source_root: str | Path,
) -> Path:
    if dataset_name not in ALLOWED_SOURCES:
        raise ValueError(f"Unsupported source dataset: {dataset_name}")
    if _contains_outer_target(path) or _contains_outer_target(dataset_name):
        raise RuntimeError("NUAA outer-target data is forbidden")
    # Reuse the already audited source-lock implementation from the frozen
    # counterfactual diagnostic without modifying that implementation.
    return validate_source_dataset_path(
        path,
        expected_dataset=dataset_name,
        source_root=source_root,
    )


def _validate_source_record_path(path: str | Path, *, dataset_root: Path) -> Path:
    raw = Path(path).expanduser()
    if ".." in raw.parts:
        raise RuntimeError("source record contains forbidden parent traversal")
    if _contains_outer_target(raw):
        raise RuntimeError("NUAA outer-target record is forbidden")
    candidate = _assert_path_under_anchor(
        raw,
        anchor=dataset_root,
        name="source_record",
        require_exists=True,
    )
    if candidate.is_symlink() or not candidate.is_file() or candidate.resolve() != candidate:
        raise RuntimeError(f"Source record must be a canonical regular file: {candidate}")
    return candidate


def _checkpoint_path_identity(
    checkpoint_path: str | Path,
    *,
    project_root: Path,
) -> tuple[Path, int, str, str]:
    checkpoint = _assert_path_under_anchor(
        checkpoint_path,
        anchor=project_root,
        name="checkpoint",
        require_exists=True,
    )
    run_root = project_root / TIER2R_ROOT_RELATIVE
    try:
        relative = checkpoint.relative_to(run_root)
    except ValueError as error:
        raise RuntimeError("Checkpoint is outside the frozen Tier2R root") from error
    if len(relative.parts) != 4 or relative.name != "last.pt":
        raise RuntimeError("Checkpoint path does not match seed/role/fold/last.pt")
    seed_name, role, fold, _ = relative.parts
    if re.fullmatch(r"seed(?:43|44|45)", seed_name) is None:
        raise RuntimeError("Checkpoint seed is outside the frozen Tier2R set")
    seed = int(seed_name[4:])
    if role not in ALLOWED_ROLES:
        raise RuntimeError("Checkpoint role must be control or c")
    if fold not in FOLD_CONTRACTS:
        raise RuntimeError("Checkpoint fold is not registered")
    return checkpoint, seed, role, fold


def _assert_source_metadata(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    project_root: Path,
    training_source: str,
) -> dict[str, Any]:
    if payload.get("source_names") != [training_source]:
        raise RuntimeError("Checkpoint source_names conflicts with the fold")
    records = payload.get("source_split_records")
    if not isinstance(records, list) or len(records) != 1:
        raise RuntimeError("Checkpoint must bind exactly one source split record")
    record = _mapping(records[0], name="checkpoint.source_split_records[0]")
    _strict_equal(record, "name", training_source, where="source_split_records[0]")
    _strict_equal(record, "train_test_id_overlap", False, where="source_split_records[0]")
    expected_root = project_root / "datasets" / training_source
    validate_source_dataset_root(
        str(record.get("path", "")),
        dataset_name=training_source,
        source_root=project_root / "datasets",
    )
    if _lexical_absolute(record["path"], name="source_split_record.path") != expected_root:
        raise RuntimeError("Checkpoint source split path is non-canonical")
    training_split_file = _validate_source_record_path(
        record.get("train_split_file", ""),
        dataset_root=expected_root,
    )
    training_ids = ensure_unique_sample_ids(read_split_file(training_split_file))
    training_split_file_sha256 = file_sha256(training_split_file)
    training_ordered_ids_sha256 = ordered_ids_sha256(training_ids)
    for key, expected in (
        ("train_split_file_sha256", training_split_file_sha256),
        ("train_ordered_ids_sha256", training_ordered_ids_sha256),
        ("num_train_samples", len(training_ids)),
    ):
        _strict_equal(record, key, expected, where="source_split_records[0]")
    data = _mapping(config.get("data"), name="checkpoint.config.data")
    sources = data.get("sources")
    if not isinstance(sources, list) or len(sources) != 1:
        raise RuntimeError("Checkpoint config must contain exactly one source")
    source = _mapping(sources[0], name="checkpoint.config.data.sources[0]")
    _strict_equal(source, "name", training_source, where="config.data.sources[0]")
    if _lexical_absolute(
        str(source.get("path", "")), name="config source path"
    ) != expected_root:
        raise RuntimeError("Checkpoint config source path is non-canonical")
    _strict_equal(data, "train_split", "train", where="config.data")
    _strict_equal(data, "diagnostic_test_eval", False, where="config.data")
    return {
        "training_split_file": training_split_file,
        "training_split_file_sha256": training_split_file_sha256,
        "training_ordered_ids_sha256": training_ordered_ids_sha256,
        "num_training_samples": len(training_ids),
    }


def load_and_validate_tier2r_checkpoint(
    checkpoint_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> tuple[Tier2SCheckpointBinding, Mapping[str, Any]]:
    root = _lexical_absolute(project_root, name="project_root")
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise RuntimeError("project_root must be a canonical non-symlink directory")
    checkpoint, seed, role, fold = _checkpoint_path_identity(
        checkpoint_path,
        project_root=root,
    )
    _require_regular_read_only(checkpoint, name="Tier2R checkpoint")
    formal_config_path = checkpoint.parent / "formal_config.yaml"
    _require_regular_read_only(formal_config_path, name="Tier2R formal config")
    payload = _load_checkpoint_safely(checkpoint)
    if not isinstance(payload, Mapping):
        raise RuntimeError("Tier2R checkpoint must decode to a mapping")
    config = _mapping(payload.get("config"), name="checkpoint.config")
    formal_config = yaml.safe_load(formal_config_path.read_text(encoding="utf-8"))
    if not isinstance(formal_config, Mapping) or dict(formal_config) != dict(config):
        raise RuntimeError("Frozen formal role config differs from checkpoint config")

    expected_metadata = {
        "format_version": 2,
        "kind": "detector",
        "epoch": 79,
        "checkpoint_selection": "fixed_last",
        "selection_rule": "fixed_last",
        "test_labels_used_for_selection": False,
        "diagnostic_test_eval": False,
        "diagnostic_only": False,
        "formal_paper_checkpoint": True,
        "warm_flag": True,
        "inference_head": "multi_scale_fused",
    }
    for key, expected in expected_metadata.items():
        _strict_equal(payload, key, expected, where="checkpoint")

    fold_contract = FOLD_CONTRACTS[fold]
    training_source = fold_contract["training_source"]
    held_out_source = fold_contract["held_out_source"]
    runtime = _mapping(
        config.get("tier2r_runtime_contract"),
        name="checkpoint.config.tier2r_runtime_contract",
    )
    runtime_expected = {
        "protocol_id": "tier2r_c_v1",
        "seed": seed,
        "role": role,
        "fold": fold,
        "training_source": training_source,
        "held_out_source_pseudo_target": held_out_source,
        "outer_target_dataset_loaded": False,
    }
    for key, expected in runtime_expected.items():
        _strict_equal(runtime, key, expected, where="tier2r_runtime_contract")
    _strict_equal(config, "seed", seed, where="checkpoint.config")
    _strict_equal(config, "deterministic", True, where="checkpoint.config")
    if _lexical_absolute(
        str(config.get("output_dir", "")), name="checkpoint output_dir"
    ) != checkpoint.parent:
        raise RuntimeError("Checkpoint output_dir conflicts with its frozen path")

    identity = _mapping(
        config.get("experiment_identity"),
        name="checkpoint.config.experiment_identity",
    )
    identity_expected = {
        "schema_version": "rc-irstd-aaai27-tier2r-component-rescue-v1",
        "stage": "tier2r_c_source_only_confirmation",
        "run_role": f"seed{seed}_{role}_{fold}",
        "architecture_id": "rc_mshnet_v2_component_role_split",
        "outer_target": held_out_source,
        "target_labels_used_for_training": False,
    }
    for key, expected in identity_expected.items():
        _strict_equal(identity, key, expected, where="experiment_identity")

    model_config = _mapping(payload.get("model_config"), name="checkpoint.model_config")
    role_config = _mapping(config.get("model"), name="checkpoint.config.model")
    for where, model in (("model_config", model_config), ("config.model", role_config)):
        _strict_equal(model, "architecture_version", ARCHITECTURE_VERSION, where=where)
        _strict_equal(model, "backend", "rc_mshnet", where=where)
        for key, expected in ROLE_CONTRACTS[role].items():
            _strict_equal(model, key, expected, where=where)
        _strict_equal(model, "expose_branch_auxiliary", False, where=where)
    _strict_equal(
        model_config,
        "baseline_identity",
        "canonical_mshnet",
        where="model_config",
    )
    _strict_equal(
        model_config,
        "initialization_contract",
        "zero_residual_exact_mshnet",
        where="model_config",
    )
    training = _mapping(config.get("training"), name="checkpoint.config.training")
    _strict_equal(training, "checkpoint_selection", "fixed_last", where="config.training")
    _strict_equal(training, "epochs", 80, where="config.training")
    source_binding = _assert_source_metadata(
        payload,
        config,
        project_root=root,
        training_source=training_source,
    )
    return (
        Tier2SCheckpointBinding(
            checkpoint_path=checkpoint,
            checkpoint_sha256=file_sha256(checkpoint),
            formal_config_path=formal_config_path,
            formal_config_sha256=file_sha256(formal_config_path),
            seed=seed,
            role=role,
            fold=fold,
            training_source=training_source,
            held_out_source=held_out_source,
            **source_binding,
            epoch=79,
        ),
        payload,
    )


def _normalize_state_dict(value: Any) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError("Checkpoint model_state must be a non-empty mapping")
    result: dict[str, torch.Tensor] = {}
    for raw_key, tensor in value.items():
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError("Checkpoint model_state contains non-tensor values")
        key = str(raw_key)
        if key.startswith("module."):
            key = key[7:]
        if key in result:
            raise RuntimeError("Checkpoint state key normalization produced duplicates")
        result[key] = tensor
    return result


def _validate_model_role(model: nn.Module, *, expected_role: str) -> None:
    if expected_role not in ALLOWED_ROLES:
        raise ValueError("expected_role must be control or c")
    if getattr(model, "architecture_version", None) != ARCHITECTURE_VERSION:
        raise RuntimeError("Factorized export requires RC-MSHNet architecture v2")
    for key, expected in ROLE_CONTRACTS[expected_role].items():
        actual = getattr(model, key, None)
        if type(actual) is not bool or actual is not expected:
            raise RuntimeError(
                f"Model role mismatch at {key}: expected {expected!r}, got {actual!r}"
            )
    if not isinstance(getattr(model, "fusion_head", None), nn.Module):
        raise RuntimeError("Model lacks the required fusion_head module")


def load_tier2r_model(
    binding: Tier2SCheckpointBinding,
    payload: Mapping[str, Any],
    *,
    device: str | torch.device,
) -> nn.Module:
    model_config = _mapping(payload.get("model_config"), name="checkpoint.model_config")
    model = build_mshnet(dict(model_config))
    model.load_state_dict(
        _normalize_state_dict(payload.get("model_state")),
        strict=True,
    )
    model.to(torch.device(device), dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    _validate_model_role(model, expected_role=binding.role)
    return model


def _autocast_enabled(device_type: str) -> bool:
    try:
        return bool(torch.is_autocast_enabled(device_type))
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return bool(torch.is_autocast_enabled())


def _require_fp32_tensor(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 4:
        raise RuntimeError(f"{name} must be a BxCxHxW tensor")
    if value.dtype != torch.float32:
        raise RuntimeError(f"{name} must be FP32, got {value.dtype}")
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{name} contains NaN or infinity")
    return value


def capture_factorized_logits(
    model: nn.Module,
    images: torch.Tensor,
    *,
    expected_role: str,
    warm_flag: bool = True,
    rtol: float = 1e-6,
    atol: float = 1e-7,
) -> FactorizedLogitBatch:
    """Capture direct base/final tensors and verify ``base+residual=final``."""

    if not isinstance(warm_flag, bool):
        raise TypeError("warm_flag must be boolean")
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("Replay tolerances must be non-negative")
    _validate_model_role(model, expected_role=expected_role)
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise TypeError("images must be a BxCxHxW tensor")
    model.float()
    model.eval()
    model.requires_grad_(False)
    model_device = next(model.parameters(), images).device
    images = images.to(model_device, dtype=torch.float32)
    captured: dict[str, torch.Tensor] = {}

    def fusion_hook(
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        if captured:
            raise RuntimeError("fusion_head was invoked more than once")
        if not torch.is_inference_mode_enabled():
            raise RuntimeError("Fusion capture requires torch.inference_mode")
        if _autocast_enabled(images.device.type):
            raise RuntimeError("Fusion capture forbids autocast")
        if "base_logits" not in kwargs:
            raise RuntimeError("fusion_head.forward kwargs lack base_logits")
        if not isinstance(output, (tuple, list)) or not output:
            raise RuntimeError("fusion_head output contract drifted")
        base = _require_fp32_tensor(kwargs["base_logits"], name="base_logits")
        final = _require_fp32_tensor(output[0], name="fusion_final_logits")
        if base.shape != final.shape:
            raise RuntimeError("Direct base/final logit shapes differ")
        captured["base"] = base.detach().clone()
        captured["final"] = final.detach().clone()

    handle = model.fusion_head.register_forward_hook(
        fusion_hook,
        with_kwargs=True,
    )
    try:
        with torch.inference_mode(), torch.autocast(
            device_type=images.device.type,
            enabled=False,
        ):
            try:
                parameters = inspect.signature(model.forward).parameters
            except (TypeError, ValueError):
                parameters = {}
            output = (
                model(images, warm_flag=warm_flag)
                if "warm_flag" in parameters
                else model(images)
            )
            model_logits = _require_fp32_tensor(
                _extract_logits(output), name="model_output_logits"
            ).detach().clone()
    finally:
        handle.remove()
    if set(captured) != {"base", "final"}:
        raise RuntimeError("fusion_head direct capture was incomplete")
    base = captured["base"]
    final = captured["final"]
    if not torch.equal(final, model_logits):
        difference = float((final - model_logits).abs().max())
        raise RuntimeError(
            "Captured fusion final differs from model output: "
            f"max_abs_error={difference}"
        )
    residual = final - base
    replay = base + residual
    difference = (replay - final).abs()
    maximum = float(difference.max())
    mean = float(difference.mean())
    if not torch.allclose(replay, final, rtol=rtol, atol=atol):
        raise RuntimeError(
            "Factorized final replay is inconsistent: "
            f"max_abs_error={maximum}, mean_abs_error={mean}"
        )
    return FactorizedLogitBatch(
        base_logits=base,
        final_logits=final,
        residual_logits=residual,
        replay_max_abs_error=maximum,
        replay_mean_abs_error=mean,
        model_output_bitwise_equal=True,
    )


class _FactorizedStreamHasher:
    def __init__(self, factor: str) -> None:
        self._digest = hashlib.sha256()
        self._digest.update(STREAM_HASH_SCHEMA.encode("ascii") + b"\0")
        self._digest.update(factor.encode("ascii") + b"\0")

    def update(self, image_id: str, array: np.ndarray) -> None:
        value = np.ascontiguousarray(array)
        if value.dtype != np.float32 or value.ndim != 2:
            raise RuntimeError("Factorized stream arrays must be 2-D FP32")
        identity = image_id.encode("utf-8")
        self._digest.update(len(identity).to_bytes(8, "big"))
        self._digest.update(identity)
        self._digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        self._digest.update(value.astype("<f4", copy=False).tobytes(order="C"))

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _crop_fp32(tensor: torch.Tensor, meta: Mapping[str, Any]) -> np.ndarray:
    value = crop_to_valid(tensor.detach().cpu(), dict(meta)).numpy()
    value = np.ascontiguousarray(value, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise RuntimeError("Cropped raw logits must be finite 2-D FP32")
    return value


def _dataset_binding(
    dataset: Any,
    *,
    dataset_root: Path,
    dataset_name: str,
    subset_role: str,
) -> dict[str, Any]:
    if getattr(dataset, "dataset_name", None) != dataset_name:
        raise RuntimeError("Dataset object name conflicts with the requested source")
    if getattr(dataset, "load_masks", None) is not True:
        raise RuntimeError("Tier2S source audit requires source masks")
    if getattr(dataset, "spatial_mode", None) != "native":
        raise RuntimeError("Tier2S factorized export requires native spatial mode")
    if getattr(dataset, "split_role", None) != "train":
        raise RuntimeError("Tier2S factorized export requires the source train split")
    split_path = _validate_source_record_path(
        getattr(dataset, "split_file", ""),
        dataset_root=dataset_root,
    )
    ids = [str(value) for value in getattr(dataset, "image_ids", ())]
    if not ids or len(set(ids)) != len(ids):
        raise RuntimeError("Dataset must declare non-empty unique ordered image IDs")
    return {
        "dataset_root": str(dataset_root),
        "dataset_name": dataset_name,
        "subset_role": subset_role,
        "requested_split": str(getattr(dataset, "requested_split", "")),
        "split_role": "train",
        "split_file": str(split_path),
        "split_file_sha256": file_sha256(split_path),
        "split_ordered_ids_sha256": ordered_ids_sha256(ids),
        "split_authority_verified": bool(
            getattr(dataset, "split_authority_verified", False)
        ),
        "spatial_mode": "native",
        "pad_multiple": int(getattr(dataset, "pad_multiple", 16)),
        "num_images": len(ids),
    }


def _expected_dataset_for_scope(
    binding: Tier2SCheckpointBinding,
    subset_role: str,
) -> str:
    if subset_role == "held_in":
        return binding.training_source
    if subset_role == "held_out":
        return binding.held_out_source
    raise ValueError("subset_role must be held_in or held_out")


def _prepare_output_path(
    output_dir: str | Path,
    *,
    output_namespace_root: str | Path,
) -> Path:
    namespace = _lexical_absolute(output_namespace_root, name="output_namespace_root")
    if namespace.is_symlink() or not namespace.is_dir() or namespace.resolve() != namespace:
        raise RuntimeError("Output namespace must be an existing canonical directory")
    output = _assert_path_under_anchor(
        output_dir,
        anchor=namespace,
        name="output_dir",
        require_exists=False,
    )
    if output == namespace:
        raise RuntimeError("Output directory must be a child of the Tier2S namespace")
    if _contains_outer_target(output):
        raise RuntimeError("Outer-target output naming is forbidden")
    return output


def _record_path_from_manifest(output: Path, value: Any) -> Path:
    relative = Path(str(value))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 2
        or relative.parts[0] != "records"
        or relative.suffix.lower() != ".npz"
    ):
        raise RuntimeError(f"Unsafe factorized record path: {value!r}")
    path = output / relative
    if path.is_symlink() or not path.is_file() or path.resolve().parent != (output / "records"):
        raise RuntimeError(f"Invalid factorized record path: {path}")
    return path


def _validate_record_npz(
    path: Path,
    *,
    image_id: str,
    dataset_name: str,
    subset_role: str,
) -> tuple[float, bool]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "base_raw_logit_float32",
            "final_raw_logit_float32",
            "residual_raw_logit_float32",
            "mask",
            "image_id",
            "dataset_name",
            "subset_role",
            "split_role",
            "original_hw",
            "input_hw",
            "valid_hw",
            "padding_ltrb",
            "spatial_mode",
            "labels_loaded",
            "inference_autocast_enabled",
            "model_output_bitwise_equal",
        }
        missing = required.difference(payload.files)
        if missing:
            raise RuntimeError(f"Factorized NPZ is incomplete: {sorted(missing)}")
        base = np.asarray(payload["base_raw_logit_float32"])
        final = np.asarray(payload["final_raw_logit_float32"])
        residual = np.asarray(payload["residual_raw_logit_float32"])
        for name, value in (("base", base), ("final", final), ("residual", residual)):
            if value.dtype != np.float32 or value.ndim != 2 or not np.isfinite(value).all():
                raise RuntimeError(f"Factorized {name} array is not finite 2-D FP32")
        if base.shape != final.shape or base.shape != residual.shape:
            raise RuntimeError("Factorized logit shapes differ")
        mask = np.asarray(payload["mask"])
        if (
            mask.dtype not in {np.dtype(np.uint8), np.dtype(np.bool_)}
            or mask.ndim != 2
            or mask.shape != base.shape
            or not np.isin(np.unique(mask), (0, 1, False, True)).all()
        ):
            raise RuntimeError("Factorized mask contract is invalid")
        scalar_strings = {
            "image_id": image_id,
            "dataset_name": dataset_name,
            "subset_role": subset_role,
            "split_role": "train",
            "spatial_mode": "native",
        }
        for key, expected in scalar_strings.items():
            if np.asarray(payload[key]).ndim != 0 or str(np.asarray(payload[key]).item()) != expected:
                raise RuntimeError(f"Factorized NPZ metadata mismatch at {key}")
        for key in ("labels_loaded", "inference_autocast_enabled", "model_output_bitwise_equal"):
            value = np.asarray(payload[key])
            if value.ndim != 0 or value.dtype.kind != "b":
                raise RuntimeError(f"Factorized NPZ {key} must be boolean")
        if not bool(np.asarray(payload["labels_loaded"]).item()):
            raise RuntimeError("Factorized NPZ must contain source labels")
        if bool(np.asarray(payload["inference_autocast_enabled"]).item()):
            raise RuntimeError("Factorized NPZ cannot enable autocast")
        if not bool(np.asarray(payload["model_output_bitwise_equal"]).item()):
            raise RuntimeError("Factorized NPZ lacks model-output equality")
        replay = base + residual
        difference = np.abs(replay - final)
        maximum = float(difference.max(initial=0.0))
        if not np.allclose(replay, final, rtol=1e-6, atol=1e-7):
            raise RuntimeError("Factorized NPZ replay is inconsistent")
        return maximum, bool(np.count_nonzero(residual) == 0)


def load_valid_existing_export(
    output_dir: str | Path,
    *,
    binding: Tier2SCheckpointBinding,
    protocol: Tier2SProtocolBinding,
    dataset_binding: Mapping[str, Any],
    subset_role: str,
    governance_binding: Mapping[str, Any] | None = None,
    tier2s_preregistration_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_governance, expected_preregistration = _manifest_consumer_bindings(
        governance_binding, tier2s_preregistration_binding
    )
    output = Path(output_dir)
    if output.is_symlink() or not output.is_dir():
        raise RuntimeError("Existing Tier2S output is not a regular directory")
    manifest_path = output / "manifest.json"
    sidecar_path = output / "manifest.sha256"
    _require_regular_read_only(manifest_path, name="Tier2S manifest")
    _require_regular_read_only(sidecar_path, name="Tier2S manifest sidecar")
    if sidecar_path.read_text(encoding="ascii").strip().split() != [
        file_sha256(manifest_path),
        "manifest.json",
    ]:
        raise RuntimeError("Tier2S manifest SHA-256 sidecar drifted")
    manifest = _load_json_object(manifest_path, name="Tier2S manifest")
    expected_top = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "source_only": True,
        "outer_target_images_loaded": False,
        "outer_target_masks_loaded": False,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "outer_target_access_authorized": False,
        "inference_dtype": "float32",
        "inference_autocast_enabled": False,
        "architecture_version": ARCHITECTURE_VERSION,
        "role": binding.role,
        "fold": binding.fold,
        "subset_role": subset_role,
    }
    for key, expected in expected_top.items():
        _strict_equal(manifest, key, expected, where="existing_manifest")
    if manifest.get("checkpoint_binding") != binding.to_jsonable():
        raise RuntimeError("Existing Tier2S checkpoint binding differs")
    if manifest.get("protocol_binding") != protocol.to_jsonable():
        raise RuntimeError("Existing Tier2S protocol binding differs")
    if manifest.get("governance_binding") != expected_governance:
        raise RuntimeError("Existing Tier2S governance binding differs")
    if (
        manifest.get("tier2s_preregistration_binding")
        != expected_preregistration
    ):
        raise RuntimeError("Existing Tier2S preregistration binding differs")
    if manifest.get("dataset_binding") != dict(dataset_binding):
        raise RuntimeError("Existing Tier2S dataset binding differs")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Existing Tier2S manifest has no records")
    if manifest.get("num_records") != len(records):
        raise RuntimeError("Existing Tier2S record count drifted")
    if manifest.get("records_sha256") != score_records_sha256(records):
        raise RuntimeError("Existing Tier2S records hash drifted")
    ids: list[str] = []
    filenames: set[str] = set()
    replay_errors: list[float] = []
    zero_flags: list[bool] = []
    for raw_record in records:
        record = _mapping(raw_record, name="existing_manifest.record")
        image_id = str(record.get("image_id", ""))
        if not image_id or image_id in ids:
            raise RuntimeError("Existing Tier2S record IDs are invalid")
        path = _record_path_from_manifest(output, record.get("file"))
        if str(record.get("file")) in filenames:
            raise RuntimeError("Existing Tier2S record filenames are duplicated")
        filenames.add(str(record.get("file")))
        if file_sha256(path) != _validate_sha256(
            record.get("sha256"), name="record.sha256"
        ):
            raise RuntimeError(f"Existing Tier2S record SHA-256 drifted: {image_id}")
        replay_error, exact_zero = _validate_record_npz(
            path,
            image_id=image_id,
            dataset_name=str(dataset_binding["dataset_name"]),
            subset_role=subset_role,
        )
        replay_errors.append(replay_error)
        zero_flags.append(exact_zero)
        ids.append(image_id)
    on_disk = {
        str(path.relative_to(output))
        for path in (output / "records").glob("*.npz")
    }
    if on_disk != filenames:
        raise RuntimeError("Existing Tier2S record inventory drifted")
    if manifest.get("ordered_image_ids_sha256") != ordered_ids_sha256(ids):
        raise RuntimeError("Existing Tier2S ordered IDs hash drifted")
    if ids != [str(value) for value in dataset_binding.get("ordered_image_ids", ids)]:
        # ``ordered_image_ids`` is intentionally optional in the public
        # dataset binding; when present it becomes an additional exact check.
        raise RuntimeError("Existing Tier2S ordered IDs differ from dataset")
    if not np.isclose(
        float(manifest.get("replay_max_abs_error", np.inf)),
        max(replay_errors, default=0.0),
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("Existing Tier2S replay summary drifted")
    if binding.role == "control" and not all(zero_flags):
        raise RuntimeError("Existing control export has nonzero residual logits")
    return manifest


def export_factorized_dataset(
    model: nn.Module,
    dataset: Any,
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    output_namespace_root: str | Path,
    binding: Tier2SCheckpointBinding,
    protocol: Tier2SProtocolBinding | None = None,
    subset_role: str,
    device: str | torch.device,
    governance_binding: Mapping[str, Any] | None = None,
    tier2s_preregistration_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_governance, manifest_preregistration = _manifest_consumer_bindings(
        governance_binding, tier2s_preregistration_binding
    )
    expected_dataset = _expected_dataset_for_scope(binding, subset_role)
    root = validate_source_dataset_root(
        dataset_root,
        dataset_name=expected_dataset,
        source_root=Path(dataset_root).parent,
    )
    dataset_info = _dataset_binding(
        dataset,
        dataset_root=root,
        dataset_name=expected_dataset,
        subset_role=subset_role,
    )
    if subset_role == "held_in":
        expected_split = {
            "split_file": str(binding.training_split_file),
            "split_file_sha256": binding.training_split_file_sha256,
            "split_ordered_ids_sha256": binding.training_ordered_ids_sha256,
            "num_images": binding.num_training_samples,
        }
        for key, expected in expected_split.items():
            if dataset_info.get(key) != expected:
                raise RuntimeError(
                    f"Held-in dataset differs from frozen checkpoint split: {key}"
                )
    output = _prepare_output_path(
        output_dir,
        output_namespace_root=output_namespace_root,
    )
    if protocol is None:
        protocol = Tier2SProtocolBinding(
            path=Path("<unit-test-unbound-protocol>"),
            sha256="0" * 64,
        )
    if output.exists():
        return load_valid_existing_export(
            output,
            binding=binding,
            protocol=protocol,
            dataset_binding=dataset_info,
            subset_role=subset_role,
            governance_binding=manifest_governance,
            tier2s_preregistration_binding=manifest_preregistration,
        )
    output.mkdir(parents=False, exist_ok=False)
    records_root = output / "records"
    records_root.mkdir(exist_ok=False)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model.to(torch_device, dtype=torch.float32)
    _validate_model_role(model, expected_role=binding.role)
    if len(dataset) == 0:
        raise RuntimeError("Tier2S source dataset is empty")
    records: list[dict[str, Any]] = []
    used_filenames: set[str] = set()
    replay_errors: list[float] = []
    residual_zero_flags: list[bool] = []
    stream_hashers = {
        name: _FactorizedStreamHasher(name)
        for name in ("base", "final", "residual")
    }
    declared_ids = [str(value) for value in getattr(dataset, "image_ids", ())]
    for index in range(len(dataset)):
        sample = dataset[index]
        if not isinstance(sample, Mapping):
            raise RuntimeError("Tier2S dataset sample must be a mapping")
        image = sample.get("image")
        mask = sample.get("mask")
        meta = sample.get("meta")
        if (
            not isinstance(image, torch.Tensor)
            or image.ndim != 3
            or not isinstance(mask, torch.Tensor)
            or mask.ndim != 3
            or not isinstance(meta, Mapping)
        ):
            raise RuntimeError("Tier2S sample requires image, mask, and metadata")
        json_meta = meta_to_jsonable(dict(meta))
        image_id = str(json_meta.get("image_id", ""))
        if not image_id or image_id != declared_ids[index]:
            raise RuntimeError("Tier2S sample order differs from dataset image IDs")
        if json_meta.get("dataset_name") != expected_dataset:
            raise RuntimeError("Tier2S sample dataset identity drifted")
        _validate_source_record_path(
            str(json_meta.get("image_path", "")), dataset_root=root
        )
        _validate_source_record_path(
            str(json_meta.get("mask_path", "")), dataset_root=root
        )
        result = capture_factorized_logits(
            model,
            image.unsqueeze(0).to(torch_device, dtype=torch.float32),
            expected_role=binding.role,
        )
        base = _crop_fp32(result.base_logits[0, 0], meta)
        final = _crop_fp32(result.final_logits[0, 0], meta)
        residual = _crop_fp32(result.residual_logits[0, 0], meta)
        cropped_mask = np.ascontiguousarray(
            crop_to_valid(mask[0].detach().cpu(), dict(meta)).numpy() > 0,
            dtype=np.uint8,
        )
        if cropped_mask.ndim != 2 or cropped_mask.shape != base.shape:
            raise RuntimeError("Tier2S cropped mask/logit shapes differ")
        replay = base + residual
        cropped_error = float(np.abs(replay - final).max(initial=0.0))
        if not np.allclose(replay, final, rtol=1e-6, atol=1e-7):
            raise RuntimeError(f"Cropped factorized replay failed for {image_id}")
        residual_exact_zero = bool(np.count_nonzero(residual) == 0)
        if binding.role == "control" and not residual_exact_zero:
            raise RuntimeError(
                f"Matched control emitted nonzero residual logits for {image_id}"
            )
        filename = _safe_record_name(image_id)
        if filename in used_filenames:
            raise RuntimeError(f"Duplicate Tier2S record filename: {filename}")
        used_filenames.add(filename)
        arrays: dict[str, np.ndarray] = {
            "base_raw_logit_float32": base,
            "final_raw_logit_float32": final,
            "residual_raw_logit_float32": residual,
            "mask": cropped_mask,
            "image_id": np.asarray(image_id),
            "dataset_name": np.asarray(expected_dataset),
            "subset_role": np.asarray(subset_role),
            "split_role": np.asarray("train"),
            "original_hw": np.asarray(json_meta["original_hw"], dtype=np.int32),
            "input_hw": np.asarray(json_meta["input_hw"], dtype=np.int32),
            "valid_hw": np.asarray(json_meta["valid_hw"], dtype=np.int32),
            "padding_ltrb": np.asarray(json_meta["padding_ltrb"], dtype=np.int32),
            "spatial_mode": np.asarray("native"),
            "labels_loaded": np.asarray(True),
            "score_representation": np.asarray(SCORE_REPRESENTATION),
            "raw_logit_dtype": np.asarray("float32"),
            "inference_autocast_enabled": np.asarray(False),
            "model_output_bitwise_equal": np.asarray(True),
            "replay_max_abs_error": np.asarray(cropped_error, dtype=np.float64),
            "residual_exact_zero": np.asarray(residual_exact_zero),
            "checkpoint_sha256": np.asarray(binding.checkpoint_sha256),
            "protocol_sha256": np.asarray(protocol.sha256),
            "seed": np.asarray(binding.seed, dtype=np.int32),
            "role": np.asarray(binding.role),
            "fold": np.asarray(binding.fold),
            "mask_alignment_applied": np.asarray(
                bool(json_meta.get("mask_alignment_applied", False))
            ),
            "mask_original_hw": np.asarray(
                json_meta.get("mask_original_hw", (0, 0)), dtype=np.int32
            ),
            "mask_aspect_relative_error": np.asarray(
                float(json_meta.get("mask_aspect_relative_error", -1.0)),
                dtype=np.float64,
            ),
            "mask_alignment_policy": np.asarray(
                str(json_meta.get("mask_alignment_policy", ""))
            ),
        }
        record_path = records_root / filename
        np.savez_compressed(record_path, **arrays)
        record_path.chmod(0o444)
        for factor, value in (("base", base), ("final", final), ("residual", residual)):
            stream_hashers[factor].update(image_id, value)
        relative = str(record_path.relative_to(output))
        records.append(
            {
                "image_id": image_id,
                "file": relative,
                "sha256": file_sha256(record_path),
                "shape": [int(base.shape[0]), int(base.shape[1])],
                "replay_max_abs_error": cropped_error,
                "model_output_bitwise_equal": True,
                "residual_exact_zero": residual_exact_zero,
            }
        )
        replay_errors.append(cropped_error)
        residual_zero_flags.append(residual_exact_zero)
    ordered_ids = [str(record["image_id"]) for record in records]
    if ordered_ids != declared_ids:
        raise RuntimeError("Tier2S export order differs from dataset order")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_integrity_schema": SCORE_RECORD_INTEGRITY_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_binding": protocol.to_jsonable(),
        "governance_binding": manifest_governance,
        "tier2s_preregistration_binding": manifest_preregistration,
        "diagnostic_only": True,
        "authorizes_go": False,
        "authorizes_source_tier3": False,
        "source_only": True,
        "outer_target_images_loaded": False,
        "outer_target_masks_loaded": False,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "outer_target_access_authorized": False,
        "architecture_version": ARCHITECTURE_VERSION,
        "role": binding.role,
        "fold": binding.fold,
        "subset_role": subset_role,
        "checkpoint_binding": binding.to_jsonable(),
        "dataset_binding": dataset_info,
        "score_representation": SCORE_REPRESENTATION,
        "inference_dtype": "float32",
        "inference_autocast_enabled": False,
        "model_mode": "eval",
        "gradient_mode": "inference_mode",
        "warm_flag": True,
        "base_capture_source": "fusion_head.forward_kwargs.base_logits",
        "final_capture_source": "fusion_head.forward_output[0]",
        "model_output_verification_source": "model_output.logits",
        "all_model_outputs_bitwise_equal": True,
        "residual_definition": "final_raw_logit_float32_minus_base_raw_logit_float32",
        "replay_formula": "base_raw_logit_float32+residual_raw_logit_float32",
        "replay_rtol": 1e-6,
        "replay_atol": 1e-7,
        "replay_max_abs_error": max(replay_errors, default=0.0),
        "all_residual_exact_zero": all(residual_zero_flags),
        "control_residual_exact_zero_required": binding.role == "control",
        "stream_hash_schema": STREAM_HASH_SCHEMA,
        "raw_logit_stream_sha256": {
            name: hasher.hexdigest() for name, hasher in stream_hashers.items()
        },
        "num_records": len(records),
        "records": records,
        "records_sha256": score_records_sha256(records),
        "ordered_image_ids_sha256": ordered_ids_sha256(ordered_ids),
        "exporter_sha256": file_sha256(Path(__file__).resolve()),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest_path.chmod(0o444)
    sidecar = output / "manifest.sha256"
    sidecar.write_text(
        f"{file_sha256(manifest_path)}  manifest.json\n",
        encoding="ascii",
    )
    sidecar.chmod(0o444)
    return manifest


def _quarantine_uncommitted_directory(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"Uncommitted Tier2S output is not a regular directory: {path}")
    for index in range(100_000):
        destination = path.parent / f".{path.name}.quarantine.{index:05d}"
        if destination.exists() or destination.is_symlink():
            continue
        os.replace(path, destination)
        return destination
    raise RuntimeError(f"Tier2S quarantine namespace is exhausted: {path}")


def export_factorized_dataset_atomically(
    model: nn.Module,
    dataset: Any,
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    output_namespace_root: str | Path,
    binding: Tier2SCheckpointBinding,
    protocol: Tier2SProtocolBinding,
    subset_role: str,
    device: str | torch.device,
    governance_binding: Mapping[str, Any] | None = None,
    tier2s_preregistration_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export through a sibling staging directory and atomically commit it.

    A final directory containing a manifest is immutable and must validate.
    A final directory without a manifest can only be an unregistered partial
    artifact, so it is preserved under a quarantine name before retrying.
    A complete staging directory is committed without recomputation; an
    incomplete or invalid staging directory is likewise preserved and rebuilt.
    """

    manifest_governance, manifest_preregistration = _manifest_consumer_bindings(
        governance_binding, tier2s_preregistration_binding
    )
    expected_dataset = _expected_dataset_for_scope(binding, subset_role)
    root = validate_source_dataset_root(
        dataset_root,
        dataset_name=expected_dataset,
        source_root=Path(dataset_root).parent,
    )
    dataset_info = _dataset_binding(
        dataset,
        dataset_root=root,
        dataset_name=expected_dataset,
        subset_role=subset_role,
    )
    final = _prepare_output_path(
        output_dir,
        output_namespace_root=output_namespace_root,
    )
    staging = _prepare_output_path(
        final.parent / f".{final.name}.staging",
        output_namespace_root=output_namespace_root,
    )

    if final.exists() or final.is_symlink():
        if final.is_symlink() or not final.is_dir():
            raise RuntimeError("Existing Tier2S final output is not a regular directory")
        if (final / "manifest.json").exists() or (final / "manifest.json").is_symlink():
            return load_valid_existing_export(
                final,
                binding=binding,
                protocol=protocol,
                dataset_binding=dataset_info,
                subset_role=subset_role,
                governance_binding=manifest_governance,
                tier2s_preregistration_binding=manifest_preregistration,
            )
        _quarantine_uncommitted_directory(final)

    if staging.exists() or staging.is_symlink():
        try:
            staged_manifest = load_valid_existing_export(
                staging,
                binding=binding,
                protocol=protocol,
                dataset_binding=dataset_info,
                subset_role=subset_role,
                governance_binding=manifest_governance,
                tier2s_preregistration_binding=manifest_preregistration,
            )
        except Exception:
            _quarantine_uncommitted_directory(staging)
        else:
            if final.exists() or final.is_symlink():
                raise RuntimeError("Tier2S final output appeared during staging recovery")
            os.replace(staging, final)
            committed = load_valid_existing_export(
                final,
                binding=binding,
                protocol=protocol,
                dataset_binding=dataset_info,
                subset_role=subset_role,
                governance_binding=manifest_governance,
                tier2s_preregistration_binding=manifest_preregistration,
            )
            if committed != staged_manifest:
                raise RuntimeError("Tier2S manifest changed during atomic staging recovery")
            return committed

    staged_manifest = export_factorized_dataset(
        model,
        dataset,
        dataset_root=root,
        output_dir=staging,
        output_namespace_root=output_namespace_root,
        binding=binding,
        protocol=protocol,
        subset_role=subset_role,
        device=device,
        governance_binding=manifest_governance,
        tier2s_preregistration_binding=manifest_preregistration,
    )
    verified_staging = load_valid_existing_export(
        staging,
        binding=binding,
        protocol=protocol,
        dataset_binding=dataset_info,
        subset_role=subset_role,
        governance_binding=manifest_governance,
        tier2s_preregistration_binding=manifest_preregistration,
    )
    if verified_staging != staged_manifest:
        raise RuntimeError("Tier2S staging manifest changed before atomic commit")
    if final.exists() or final.is_symlink():
        raise RuntimeError("Tier2S final output appeared before atomic commit")
    os.replace(staging, final)
    return load_valid_existing_export(
        final,
        binding=binding,
        protocol=protocol,
        dataset_binding=dataset_info,
        subset_role=subset_role,
        governance_binding=manifest_governance,
        tier2s_preregistration_binding=manifest_preregistration,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-name", choices=sorted(ALLOWED_SOURCES), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-role", choices=sorted(ALLOWED_ROLES), required=True)
    parser.add_argument("--fold", choices=tuple(FOLD_CONTRACTS), required=True)
    parser.add_argument("--scope", choices=("held_in", "held_out"), required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--governance-registration-sha256", required=True)
    parser.add_argument("--tier2s-preregistration", required=True, type=Path)
    parser.add_argument("--tier2s-preregistration-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    governance_binding, preregistration_binding = (
        require_frozen_tier2s_consumer_bindings(
            expected_governance_registration_sha256=(
                args.governance_registration_sha256
            ),
            tier2s_preregistration_path=args.tier2s_preregistration,
            expected_tier2s_preregistration_sha256=(
                args.tier2s_preregistration_sha256
            ),
            project_root=PROJECT_ROOT,
        )
    )
    if args.split != "train":
        raise RuntimeError("Tier2S protocol permits only the official source train split")
    protocol = validate_protocol(args.protocol, project_root=PROJECT_ROOT)
    binding, payload = load_and_validate_tier2r_checkpoint(
        args.checkpoint,
        project_root=PROJECT_ROOT,
    )
    if binding.role != args.expected_role:
        raise RuntimeError("--expected-role conflicts with the checkpoint binding")
    if binding.fold != args.fold:
        raise RuntimeError("--fold conflicts with the checkpoint binding")
    expected_dataset = _expected_dataset_for_scope(binding, args.scope)
    if args.dataset_name != expected_dataset:
        raise RuntimeError("--dataset-name conflicts with checkpoint fold/scope")
    dataset_root = validate_source_dataset_root(
        args.dataset_dir,
        dataset_name=args.dataset_name,
        source_root=PROJECT_ROOT / "datasets",
    )
    namespace = PROJECT_ROOT / TIER2S_OUTPUT_RELATIVE
    namespace.mkdir(parents=True, exist_ok=True)
    if namespace.is_symlink() or namespace.resolve() != namespace:
        raise RuntimeError("Tier2S output namespace is not canonical")
    dataset = IRSTDEvalDataset(
        dataset_root,
        split=args.split,
        spatial_mode="native",
        pad_multiple=16,
        dataset_name=args.dataset_name,
        load_masks=True,
    )
    model = load_tier2r_model(binding, payload, device=args.device)
    manifest = export_factorized_dataset_atomically(
        model,
        dataset,
        dataset_root=dataset_root,
        output_dir=args.output_dir,
        output_namespace_root=namespace,
        binding=binding,
        protocol=protocol,
        subset_role=args.scope,
        device=args.device,
        governance_binding=governance_binding,
        tier2s_preregistration_binding=preregistration_binding,
    )
    summary = {
        "status": "valid",
        "manifest": str(Path(args.output_dir).resolve() / "manifest.json"),
        "manifest_sha256": file_sha256(Path(args.output_dir).resolve() / "manifest.json"),
        "checkpoint_sha256": binding.checkpoint_sha256,
        "seed": binding.seed,
        "role": binding.role,
        "fold": binding.fold,
        "scope": args.scope,
        "dataset_name": args.dataset_name,
        "num_records": int(manifest["num_records"]),
        "governance_registration_sha256": governance_binding["registration"][
            "sha256"
        ],
        "tier2s_preregistration_sha256": preregistration_binding["sha256"],
    }
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALLOWED_ROLES",
    "ALLOWED_SEEDS",
    "ALLOWED_SOURCES",
    "ARCHITECTURE_VERSION",
    "FOLD_CONTRACTS",
    "FactorizedLogitBatch",
    "PROTOCOL_ID",
    "ROLE_CONTRACTS",
    "SCHEMA_VERSION",
    "Tier2SCheckpointBinding",
    "Tier2SProtocolBinding",
    "export_factorized_dataset_atomically",
    "capture_factorized_logits",
    "export_factorized_dataset",
    "load_and_validate_tier2r_checkpoint",
    "load_tier2r_model",
    "load_valid_existing_export",
    "validate_protocol",
    "validate_source_dataset_root",
]
