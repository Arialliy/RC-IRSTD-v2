"""Freeze zero-label actions and audit a later labelled score export.

The formal target protocol deliberately uses two independent score-map
artifacts.  Selection reads ``scores_unlabeled`` only; after its actions are
hashed and frozen, a second export may load masks for a labelled audit.  This
module binds the two exports by detector, split identity, spatial protocol and
bit-exact FP32 raw logits without ever reading a mask from the selection
artifact.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.artifact_integrity import verify_score_map_directory
from rc_irstd.utils.io import atomic_write_json


FREEZE_SCHEMA_VERSION = "rc-v4-zero-label-selection-freeze-v1"
PAIR_AUDIT_SCHEMA_VERSION = "rc-v4-target-stage-pair-audit-v1"
_SHA256_FIELDS = (
    "weight_sha256",
    "ordered_image_ids_sha256",
    "split_file_sha256",
    "split_ordered_ids_sha256",
)
_SPATIAL_FIELDS = (
    "target_dataset",
    "requested_split",
    "split_role",
    "split_authority_verified",
    "spatial_mode",
    "base_hw",
    "pad_multiple",
    "score_representation",
    "probability_dtype",
    "logit_dtype",
    "probability_transform",
    "probability_clipping",
    "inference_autocast_enabled",
    "warm_flag",
    "model_backend",
    "source_datasets",
)


def _file_sha256(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {resolved}")
    return payload


def freeze_zero_label_actions(
    selection_files: Sequence[str | Path],
    *,
    bound_artifacts: Sequence[str | Path],
    output_dir: str | Path,
) -> Path:
    """Hash and make zero-label selections read-only before labels are loaded."""

    selections = [Path(value).expanduser().resolve() for value in selection_files]
    if not selections or len(set(selections)) != len(selections):
        raise ValueError("selection_files must contain unique paths")
    bindings = [Path(value).expanduser().resolve() for value in bound_artifacts]
    if not bindings or len(set(bindings)) != len(bindings):
        raise ValueError("bound_artifacts must contain unique paths")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    frozen_at_unix_ns = time.time_ns()
    records = [
        {
            "role": "zero_label_action",
            "path": str(path),
            "sha256": _file_sha256(path),
        }
        for path in selections
    ]
    records.extend(
        {
            "role": "bound_selection_input",
            "path": str(path),
            "sha256": _file_sha256(path),
        }
        for path in bindings
    )
    checksum_path = root / "FROZEN_SELECTION_SHA256SUMS"
    checksum_path.write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in records),
        encoding="utf-8",
    )
    frozen_at_path = root / "FROZEN_AT.txt"
    frozen_at_path.write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )
    freeze_record = root / "freeze_record.json"
    atomic_write_json(
        freeze_record,
        {
            "schema_version": FREEZE_SCHEMA_VERSION,
            "selection_frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_at_unix_ns": frozen_at_unix_ns,
            "target_labels_loaded_during_selection": False,
            "post_selection_hyperparameter_changes_allowed": False,
            "selection_freeze_required": True,
            "records": records,
            "checksum_file": str(checksum_path),
            "checksum_file_sha256": _file_sha256(checksum_path),
        },
    )
    digest_path = root / "freeze_record.sha256"
    digest_path.write_text(f"{_file_sha256(freeze_record)}\n", encoding="ascii")
    for path in (*selections, checksum_path, frozen_at_path, freeze_record, digest_path):
        path.chmod(0o444)
    return freeze_record


def verify_selection_freeze(path: str | Path) -> dict[str, Any]:
    """Fail closed if any frozen action or bound input has changed."""

    freeze_path = Path(path).expanduser().resolve()
    payload = _load_json(freeze_path)
    if payload.get("schema_version") != FREEZE_SCHEMA_VERSION:
        raise ValueError("Unsupported zero-label selection freeze schema")
    if payload.get("target_labels_loaded_during_selection") is not False:
        raise ValueError("Selection freeze does not prove labels_loaded=false")
    if payload.get("post_selection_hyperparameter_changes_allowed") is not False:
        raise ValueError("Selection freeze permits post-selection changes")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Selection freeze contains no bound records")
    for item in records:
        if not isinstance(item, Mapping):
            raise ValueError("Selection freeze record is not an object")
        expected = str(item.get("sha256", ""))
        if _file_sha256(str(item.get("path", ""))) != expected:
            raise ValueError("A frozen selection artifact changed after freeze")
    checksum_path = Path(str(payload.get("checksum_file", ""))).expanduser().resolve()
    if _file_sha256(checksum_path) != payload.get("checksum_file_sha256"):
        raise ValueError("Frozen selection checksum file changed after freeze")
    digest_path = freeze_path.with_suffix(".sha256")
    expected_record_digest = digest_path.read_text(encoding="ascii").strip()
    if _file_sha256(freeze_path) != expected_record_digest:
        raise ValueError("Selection freeze record digest mismatch")
    return payload


def _manifest_records(
    root: Path, manifest: Mapping[str, Any]
) -> list[tuple[str, Path]]:
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Score manifest contains no records: {root}")
    resolved: list[tuple[str, Path]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Score manifest record is not an object")
        image_id = str(record.get("image_id", ""))
        filename = str(record.get("file", ""))
        if not image_id or not filename:
            raise ValueError("Score manifest record lacks image_id/file")
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved.append((image_id, path))
    if len({image_id for image_id, _ in resolved}) != len(resolved):
        raise ValueError("Score manifest image IDs are not unique")
    return resolved


def audit_target_score_stage_pair(
    unlabeled_score_dir: str | Path,
    labeled_score_dir: str | Path,
    *,
    freeze_record: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Prove that the post-freeze labelled export replays identical logits."""

    freeze = verify_selection_freeze(freeze_record)
    unlabeled_root = Path(unlabeled_score_dir).expanduser().resolve()
    labeled_root = Path(labeled_score_dir).expanduser().resolve()
    if unlabeled_root == labeled_root:
        raise ValueError("Unlabeled selection and labeled audit must use distinct dirs")
    unlabeled_manifest_path = unlabeled_root / "manifest.json"
    labeled_manifest_path = labeled_root / "manifest.json"
    unlabeled, _, unlabeled_integrity = verify_score_map_directory(
        unlabeled_root, require_integrity=True, require_masks=False
    )
    labeled, _, labeled_integrity = verify_score_map_directory(
        labeled_root, require_integrity=True, require_masks=True
    )
    if (
        unlabeled is None
        or labeled is None
        or unlabeled_integrity.get("verified") is not True
        or labeled_integrity.get("verified") is not True
    ):
        raise ValueError("Target score stages require complete verified integrity")
    if unlabeled.get("labels_loaded") is not False:
        raise ValueError("Selection score artifact must record labels_loaded=false")
    if labeled.get("labels_loaded") is not True:
        raise ValueError("Labeled audit score artifact must record labels_loaded=true")
    frozen_at_ns = int(freeze.get("frozen_at_unix_ns", 0))
    if frozen_at_ns <= 0 or labeled_manifest_path.stat().st_mtime_ns <= frozen_at_ns:
        raise ValueError("Labeled audit artifact was not created after action freeze")
    for field in (*_SHA256_FIELDS, *_SPATIAL_FIELDS):
        if unlabeled.get(field) != labeled.get(field):
            raise ValueError(f"Target score stages differ in {field}")
    unlabeled_records = _manifest_records(unlabeled_root, unlabeled)
    labeled_records = _manifest_records(labeled_root, labeled)
    if [value[0] for value in unlabeled_records] != [
        value[0] for value in labeled_records
    ]:
        raise ValueError("Target score stages have different ordered image IDs")
    raw_logit_payload = hashlib.sha256()
    for (image_id, first), (_, second) in zip(unlabeled_records, labeled_records):
        with np.load(first, allow_pickle=False) as first_data, np.load(
            second, allow_pickle=False
        ) as second_data:
            if "mask" in first_data:
                raise ValueError("Selection score records must not contain masks")
            if "mask" not in second_data:
                raise ValueError("Labeled audit score records must contain masks")
            if "logit" not in first_data or "logit" not in second_data:
                raise ValueError("Both target score stages must contain raw logits")
            first_logit = np.asarray(first_data["logit"])
            second_logit = np.asarray(second_data["logit"])
        if first_logit.dtype != np.float32 or second_logit.dtype != np.float32:
            raise ValueError("Target stage raw logits must be float32")
        if not np.array_equal(first_logit, second_logit):
            raise ValueError(f"Target stage raw logits differ for {image_id}")
        raw_logit_payload.update(image_id.encode("utf-8"))
        raw_logit_payload.update(b"\0")
        raw_logit_payload.update(first_logit.tobytes(order="C"))
    result = {
        "schema_version": PAIR_AUDIT_SCHEMA_VERSION,
        "verified": True,
        "unlabeled_score_dir": str(unlabeled_root),
        "labeled_score_dir": str(labeled_root),
        "selection_freeze_record": str(Path(freeze_record).expanduser().resolve()),
        "selection_freeze_record_sha256": _file_sha256(freeze_record),
        "frozen_zero_results": [
            {"path": str(item["path"]), "sha256": str(item["sha256"])}
            for item in freeze["records"]
            if item.get("role") == "zero_label_action"
        ],
        "labels_loaded_during_selection": False,
        "labels_loaded_during_audit": True,
        "labeled_audit_created_after_freeze": True,
        "ordered_image_ids_sha256": unlabeled["ordered_image_ids_sha256"],
        "detector_weight_sha256": unlabeled["weight_sha256"],
        "unlabeled_manifest_sha256": unlabeled_integrity["manifest_sha256"],
        "unlabeled_records_sha256": unlabeled_integrity["records_sha256"],
        "labeled_manifest_sha256": labeled_integrity["manifest_sha256"],
        "labeled_records_sha256": labeled_integrity["records_sha256"],
        "raw_logit_stream_sha256": raw_logit_payload.hexdigest(),
        "num_records": len(unlabeled_records),
        "spatial_protocol_fields_verified": list(_SPATIAL_FIELDS),
    }
    atomic_write_json(output, result)
    return result


__all__ = [
    "FREEZE_SCHEMA_VERSION",
    "PAIR_AUDIT_SCHEMA_VERSION",
    "audit_target_score_stage_pair",
    "freeze_zero_label_actions",
    "verify_selection_freeze",
]
