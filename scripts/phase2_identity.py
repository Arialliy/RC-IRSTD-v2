#!/usr/bin/env python3
"""Write and verify Phase-2 MSHNet + StableSLS identity artifacts.

Running jobs receive identity sidecars marked ``running_unfrozen``.  Only a
400-epoch fixed-last checkpoint can be finalized and bound to a SHA256.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.artifact_integrity import ordered_ids_sha256
from rc_irstd.data import ensure_unique_sample_ids, read_split_file
from rc_irstd.models import build_mshnet


STABLE_SLS_V1 = {
    "loss_id": "stable_sls_v1",
    "implementation": "rc_irstd.losses.sls.StableSLSLoss",
    "bce_weight": 0.5,
    "scale_iou_weight": 1.0,
    "location_weight": 0.25,
    "max_positive_weight": 50.0,
    "lambda_tail": 0.0,
    "lambda_miss": 0.0,
    "lambda_margin": 0.0,
    "auxiliary_weight": 0.25,
}
CANONICAL_MODEL = {
    "architecture_id": "local_canonical_mshnet",
    "runtime_class": "rc_irstd.models.mshnet.StructuredCanonicalMSHNet",
    "base_class": "model.MSHNet.MSHNet",
    "backend": "canonical",
    "input_channels": 3,
    "channels": [16, 32, 64, 128, 256],
    "block_counts": [2, 2, 2, 2],
}
LAUNCH_CODE_SHA256 = {
    "rc_irstd/training/detector_trainer.py": (
        "6a6767acdd036615908f8a07c37c39f39a27ed8c4380bf935e4b2bba698bc74c"
    ),
    "rc_irstd/losses/sls.py": (
        "6407c245b1723c43c03a140b291eb3417d7a0e5ae70add47098024412791eaf8"
    ),
    "rc_irstd/losses/detector.py": (
        "59a0278ca29f67779589e061b9d203c9b47b70d4f1e5023139ce4d2bf07f363f"
    ),
    "rc_irstd/models/mshnet.py": (
        "bbcf0b7e1cfebe9a13e748a523cd941ccd1fd194cd1b87ac363de3688fb9a221"
    ),
    "model/MSHNet.py": (
        "6e88d132c09cbdcc149598b577f67f005c16baaa7e76884763bbeca9eee502df"
    ),
}
EXPECTED_HISTORY_FIELDS = [
    "epoch",
    "lr",
    "loss_total",
    "loss_sls",
    "loss_bce",
    "loss_scale_iou",
    "loss_location",
    "loss_auxiliary",
    "loss_tail",
    "loss_miss",
    "loss_margin",
    "num_background_peaks",
    "num_target_objects",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_launch_code_unchanged() -> dict[str, str]:
    actual = {
        relative: sha256_file(PROJECT_ROOT / relative)
        for relative in LAUNCH_CODE_SHA256
    }
    _assert_equal(actual, LAUNCH_CODE_SHA256, name="running baseline source identity")
    return actual


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _assert_equal(actual: Any, expected: Any, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name}: expected {expected!r}, received {actual!r}")


def _validate_config(config: Mapping[str, Any]) -> None:
    model = _mapping(config.get("model"), name="model")
    loss = _mapping(config.get("loss"), name="loss")
    sls = _mapping(loss.get("sls_kwargs"), name="loss.sls_kwargs")
    data = _mapping(config.get("data"), name="data")
    optimizer = _mapping(config.get("optimizer"), name="optimizer")
    training = _mapping(config.get("training"), name="training")

    expected_model = {
        "backend": "canonical",
        "input_channels": 3,
        "channels": [16, 32, 64, 128, 256],
        "block_counts": [2, 2, 2, 2],
    }
    _assert_equal(dict(model), expected_model, name="canonical MSHNet config")
    for key, expected in {
        "lambda_tail": 0.0,
        "lambda_miss": 0.0,
        "lambda_margin": 0.0,
        "auxiliary_weight": 0.25,
    }.items():
        _assert_equal(float(loss.get(key, float("nan"))), expected, name=f"loss.{key}")
    for key, expected in {
        "bce_weight": 0.5,
        "iou_weight": 1.0,
        "location_weight": 0.25,
        "max_positive_weight": 50.0,
    }.items():
        _assert_equal(float(sls.get(key, float("nan"))), expected, name=f"sls.{key}")
    _assert_equal(data.get("train_split"), "train", name="data.train_split")
    _assert_equal(data.get("val_split"), None, name="data.val_split")
    _assert_equal(
        bool(data.get("diagnostic_test_eval", False)),
        False,
        name="data.diagnostic_test_eval",
    )
    sources = data.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("data.sources must be a non-empty list")
    for index, source in enumerate(sources):
        item = _mapping(source, name=f"data.sources[{index}]")
        if not item.get("name") or not item.get("path"):
            raise ValueError(f"data.sources[{index}] must contain name and path")
    _assert_equal(optimizer.get("name"), "sgd", name="optimizer.name")
    _assert_equal(int(training.get("epochs", -1)), 400, name="training.epochs")
    _assert_equal(
        training.get("checkpoint_selection"),
        "fixed_last",
        name="training.checkpoint_selection",
    )
    _assert_equal(training.get("resume"), None, name="training.resume")
    _assert_equal(int(training.get("warmup_epochs", -1)), 5, name="warmup_epochs")


def _source_paths(config: Mapping[str, Any]) -> list[dict[str, str]]:
    data = _mapping(config["data"], name="data")
    result: list[dict[str, str]] = []
    for source in data["sources"]:
        item = _mapping(source, name="data source")
        configured = Path(str(item["path"])).expanduser()
        resolved = (
            configured.resolve()
            if configured.is_absolute()
            else (PROJECT_ROOT / "configs" / configured).resolve()
        )
        if not resolved.is_dir():
            raise FileNotFoundError(resolved)
        result.append(
            {
                "name": str(item["name"]),
                "configured_path": str(item["path"]),
                "resolved_path": str(resolved),
            }
        )
    return result


def _read_history(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "history.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _assert_equal(reader.fieldnames, EXPECTED_HISTORY_FIELDS, name="history header")
        rows = list(reader)
    for index, row in enumerate(rows):
        if int(row.get("epoch", -1)) != index:
            raise ValueError(f"{path} has non-contiguous epoch at row {index}")
        for field in EXPECTED_HISTORY_FIELDS[1:]:
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"{path} row {index} field {field} is not finite")
        for field in (
            "loss_tail",
            "loss_miss",
            "loss_margin",
            "num_background_peaks",
            "num_target_objects",
        ):
            if float(row[field]) != 0.0:
                raise ValueError(f"{path} row {index} field {field} is not zero")
    return rows


def _load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint root must be a mapping: {path}")
    return payload


def _verify_split_records(records: list[dict[str, Any]]) -> None:
    for record in records:
        train_path = Path(str(record["train_split_file"]))
        test_path = Path(str(record["test_split_file"]))
        for path in (train_path, test_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        train_ids = ensure_unique_sample_ids(read_split_file(train_path))
        test_ids = ensure_unique_sample_ids(read_split_file(test_path))
        checks = {
            "train_split_file_sha256": sha256_file(train_path),
            "test_split_file_sha256": sha256_file(test_path),
            "train_ordered_ids_sha256": ordered_ids_sha256(train_ids),
            "test_ordered_ids_sha256": ordered_ids_sha256(test_ids),
            "num_train_samples": len(train_ids),
            "num_test_samples": len(test_ids),
            "train_test_id_overlap": bool(set(train_ids).intersection(test_ids)),
        }
        for key, expected in checks.items():
            _assert_equal(record.get(key), expected, name=f"split record {record.get('name')}.{key}")


def _verify_canonical_state(payload: Mapping[str, Any]) -> dict[str, int]:
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint has no model_state")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("checkpoint model_state contains non-tensor values")
    model = build_mshnet({"backend": "canonical"})
    model.load_state_dict(dict(state), strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "trainable_parameters": parameter_count,
        "state_tensor_count": len(state),
        "state_numel": sum(value.numel() for value in state.values()),
    }
    _assert_equal(report["trainable_parameters"], 4_065_513, name="parameter count")
    _assert_equal(report["state_tensor_count"], 340, name="state tensor count")
    _assert_equal(report["state_numel"], 4_072_753, name="state numel")
    return report


def _text_payload(values: Mapping[str, Any]) -> str:
    lines: list[str] = []
    for key, value in values.items():
        rendered = (
            json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            if isinstance(value, (dict, list))
            else str(value).lower()
            if isinstance(value, bool)
            else "null"
            if value is None
            else str(value)
        )
        lines.append(f"{key}={rendered}")
    return "\n".join(lines) + "\n"


def write_identity(run_dir: Path, *, finalize: bool) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    previous_manifest = run_dir / "IDENTITY_MANIFEST.json"
    if previous_manifest.is_file() and not finalize:
        previous = _load_json(previous_manifest)
        if previous.get("artifact_state") == "finalized":
            raise RuntimeError(f"refusing to downgrade finalized identity: {run_dir}")

    runtime_code_sha256 = _assert_launch_code_unchanged()
    config = _load_json(config_path)
    _validate_config(config)
    data = _mapping(config["data"], name="data")
    training = _mapping(config["training"], name="training")
    checkpoint_sha256: str | None = None
    source_split_records: list[dict[str, Any]] | None = None
    final_epoch: int | None = None
    model_state_report: dict[str, int] | None = None

    if finalize:
        rows = _read_history(run_dir)
        expected_epochs = int(training["epochs"])
        if len(rows) != expected_epochs:
            raise ValueError(
                f"{run_dir} has {len(rows)} history rows; expected {expected_epochs}"
            )
        checkpoint_path = run_dir / "last.pt"
        payload = _load_checkpoint(checkpoint_path)
        _assert_equal(payload.get("kind"), "detector", name="checkpoint kind")
        _assert_equal(payload.get("format_version"), 2, name="checkpoint format")
        final_epoch = int(payload.get("epoch", -1))
        _assert_equal(final_epoch, expected_epochs - 1, name="checkpoint epoch")
        _assert_equal(payload.get("checkpoint_selection"), "fixed_last", name="checkpoint policy")
        _assert_equal(
            bool(payload.get("test_labels_used_for_selection", True)),
            False,
            name="checkpoint target-label selection flag",
        )
        _assert_equal(
            bool(payload.get("diagnostic_test_eval", True)),
            False,
            name="checkpoint diagnostic_test_eval",
        )
        _assert_equal(bool(payload.get("warm_flag", False)), True, name="warm flag")
        _assert_equal(
            payload.get("inference_head"),
            "multi_scale_fused",
            name="inference head",
        )
        _assert_equal(payload.get("initialization"), None, name="baseline initialization")
        _assert_equal(payload.get("config"), config, name="checkpoint embedded config")
        _assert_equal(payload.get("model_config"), config["model"], name="model config")
        resume_contract = _mapping(payload.get("resume_contract"), name="resume contract")
        for key in ("model", "loss", "optimizer"):
            _assert_equal(resume_contract.get(key), config[key], name=f"resume contract {key}")
        resume_training = _mapping(resume_contract.get("training"), name="resume training")
        for key in ("epochs", "warmup_epochs", "grad_clip", "amp", "min_lr"):
            _assert_equal(
                resume_training.get(key),
                config["training"][key],
                name=f"resume training {key}",
            )
        expected_names = [str(item["name"]) for item in data["sources"]]
        _assert_equal(payload.get("source_names"), expected_names, name="source names")
        raw_records = payload.get("source_split_records")
        if not isinstance(raw_records, list) or len(raw_records) != len(expected_names):
            raise ValueError("checkpoint source_split_records are missing or incomplete")
        source_split_records = [dict(_mapping(item, name="split record")) for item in raw_records]
        if any(bool(item.get("train_test_id_overlap", True)) for item in source_split_records):
            raise ValueError("checkpoint reports train/test ID overlap")
        _verify_split_records(source_split_records)
        model_state_report = _verify_canonical_state(payload)
        temporary_checkpoint = run_dir / "last.pt.tmp"
        if temporary_checkpoint.exists():
            raise RuntimeError(f"checkpoint temporary file still exists: {temporary_checkpoint}")
        checkpoint_sha256 = sha256_file(checkpoint_path)
        _atomic_write_text(
            run_dir / "checkpoint.sha256",
            f"{checkpoint_sha256}  last.pt\n",
        )

    artifact_state = "finalized" if finalize else "running_unfrozen"
    checkpoint_binding = checkpoint_sha256 or "PENDING_FINAL_EPOCH_400"
    config_sha256 = sha256_file(config_path)
    loss_identity = {
        "schema_version": "rc-irstd-loss-identity-v1",
        "artifact_state": artifact_state,
        **STABLE_SLS_V1,
        "main_objective": "stable_sls_v1",
        "auxiliary_objective": "mean_stable_sls_v1_over_canonical_mshnet_logits",
        "warmup_epochs": int(training["warmup_epochs"]),
        "canonical_auxiliary_count_after_warmup": 4,
        "total_objective": "main_stable_sls+0.25*mean(auxiliary_stable_sls)",
        "warmup_epoch_indices": "0..4",
        "multi_scale_epoch_indices": "5..399",
        "checkpoint_policy": "fixed_last",
        "target_test_labels_used_for_checkpoint_selection": False,
        "identity_derivation": "checkpoint_config_plus_launch_source_sha256",
        "checkpoint_embeds_loss_hyperparameters": bool(finalize),
        "upstream_sls_iou_loss_used": False,
        "exact_official_reproduction": False,
        "runtime_code_sha256": runtime_code_sha256,
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_binding,
    }
    model_identity = {
        "schema_version": "rc-irstd-model-identity-v1",
        "artifact_state": artifact_state,
        **CANONICAL_MODEL,
        **(model_state_report or {
            "trainable_parameters": 4_065_513,
            "state_tensor_count": 340,
            "state_numel": 4_072_753,
        }),
        "rc_mshnet_extensions_present": False,
        "pretrained_initializer": "none",
        "exact_upstream_source_identity": "not_proven",
        "official_reproduction": False,
        "runtime_code_sha256": runtime_code_sha256,
        "checkpoint_sha256": checkpoint_binding,
    }
    data_identity = {
        "schema_version": "rc-irstd-data-identity-v1",
        "artifact_state": artifact_state,
        "sources": _source_paths(config),
        "train_split": str(data["train_split"]),
        "val_split": data.get("val_split"),
        "diagnostic_test_eval": bool(data["diagnostic_test_eval"]),
        "image_size": data.get("image_size"),
        "batch_per_domain": data.get("batch_per_domain"),
        "augment": bool(data.get("augment", True)),
        "source_split_records": source_split_records or "PENDING_FINAL_CHECKPOINT",
        "test_manifest_ids_read_for_overlap_audit": True,
        "test_images_loaded": False,
        "test_masks_loaded": False,
        "dataset_raster_content_sha256": "not_recorded",
        "checkpoint_sha256": checkpoint_binding,
    }
    manifest = {
        "schema_version": "rc-irstd-baseline-identity-manifest-v1",
        "artifact_state": artifact_state,
        "run_dir": str(run_dir),
        "final_epoch": final_epoch,
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_embeds_resolved_config": bool(finalize),
        "runtime_code_sha256": runtime_code_sha256,
        "loss": loss_identity,
        "model": model_identity,
        "data": data_identity,
    }
    _atomic_write_text(run_dir / "LOSS_IDENTITY.txt", _text_payload(loss_identity))
    _atomic_write_text(run_dir / "MODEL_IDENTITY.txt", _text_payload(model_identity))
    _atomic_write_text(run_dir / "DATA_IDENTITY.txt", _text_payload(data_identity))
    _atomic_write_json(run_dir / "IDENTITY_MANIFEST.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Require the complete 400-epoch checkpoint and bind all identities to its SHA256",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifests = [write_identity(run_dir, finalize=args.finalize) for run_dir in args.run_dir]
    print(
        json.dumps(
            {
                "status": "passed",
                "artifact_state": "finalized" if args.finalize else "running_unfrozen",
                "runs": [manifest["run_dir"] for manifest in manifests],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
