from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation.artifact_integrity import (
    SCORE_MANIFEST_SCHEMA_VERSION,
    SCORE_RECORD_INTEGRITY_SCHEMA,
    ordered_ids_sha256,
    score_records_sha256,
)
from risk_curve.build_curve_episodes import LOGIT_EPISODE_SCHEMA_VERSION
from risk_curve.domain_statistics import (
    LOGIT_STATISTICS_SCHEMA_VERSION,
    feature_schema_sha256,
    statistics_names_sha256,
)
from risk_curve.representation import (
    GRID_DETECTOR_PROTOCOL,
    LOGIT_GRID_ARTIFACT_TYPE,
    LOGIT_GRID_SCHEMA_VERSION,
    LOGIT_PREDICTION_RULE,
    LOGIT_REPRESENTATION,
    MAX_MODEL_GRID_POINTS,
    canonical_json_sha256,
    empty_action_contract,
    logit_threshold_grid_sha256,
)
from risk_curve.source_provenance_v4 import (
    SOURCE_PROVENANCE_SCHEMA_VERSION,
    SOURCE_PROVENANCE_RUN_SCHEMA_VERSION,
    _SnapshotRegistry,
    load_source_provenance_run_evidence,
    main as source_provenance_main,
    validate_source_provenance_run_evidence,
    verify_source_only_provenance_v4,
)


SOURCES = ("IRSTD-1K", "NUDT-SIRST")
SOURCE_KEYS = {"IRSTD-1K": "irstd1k", "NUDT-SIRST": "nudt"}
GRID = np.asarray([-2.0, 0.0, 2.0], dtype=np.float32)
NAMES = ("logit_feature_a", "logit_feature_b")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class _Bundle:
    root: Path
    grid_manifest: Path
    splits: dict[str, Path]
    checkpoints: list[Path]
    episodes: dict[str, Path]
    score_dirs: list[Path]

    def kwargs(self) -> dict[str, object]:
        return {
            "project_root": self.root,
            "threshold_grid_manifest": self.grid_manifest,
            "official_train_split_manifests": self.splits,
            "detector_checkpoints": self.checkpoints,
            "episode_archives": list(self.episodes.values()),
        }


def _write_checkpoint(
    path: Path,
    marker: str,
    *,
    sources: tuple[str, ...],
    splits: dict[str, Path],
    split_ids: dict[str, tuple[str, ...]],
    changes: dict[str, object] | None = None,
) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    split_records = [
        {
            "name": source,
            "path": str(splits[source].parent.parent.resolve()),
            "train_split_file": str(splits[source].resolve()),
            "train_split_file_sha256": _sha256(splits[source]),
            "train_ordered_ids_sha256": ordered_ids_sha256(split_ids[source]),
            "num_train_samples": len(split_ids[source]),
            "test_split_file": str(
                splits[source].with_name(f"test_{source}.txt").resolve()
            ),
            "train_test_id_overlap": False,
        }
        for source in sources
    ]
    resolved_changes = dict(changes or {})
    if resolved_changes.pop("__tamper_train_split_sha", False):
        split_records[0]["train_split_file_sha256"] = "0" * 64
    checkpoint: dict[str, object] = {
        "kind": "detector",
        "checkpoint_selection": "fixed_last",
        "selection_rule": "fixed_last",
        "test_labels_used_for_selection": False,
        "diagnostic_test_eval": False,
        "diagnostic_only": False,
        "formal_paper_checkpoint": True,
        "format_version": 2,
        "epoch": 19,
        "warm_flag": True,
        "inference_head": "multi_scale_fused",
        "source_names": list(sources),
        "source_split_records": split_records,
        "config": {
            "marker": marker,
            "data": {
                "sources": [
                    {"name": source, "path": f"../datasets/{source}"}
                    for source in sources
                ],
                "train_split": "train",
                "val_split": None,
                "diagnostic_test_eval": False,
            },
            "training": {"checkpoint_selection": "fixed_last"},
        },
        "model_state": {"weight": torch.asarray([1.0])},
    }
    checkpoint.update(resolved_changes)
    torch.save(checkpoint, path)
    return path, _sha256(path)


def _write_score_dir(
    root: Path,
    *,
    target: str,
    sources: tuple[str, ...],
    checkpoint: Path,
    checkpoint_sha: str,
    split: Path,
    ids: tuple[str, ...],
) -> tuple[Path, dict[str, object]]:
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index, image_id in enumerate(ids):
        record_path = root / f"{index:03d}.npz"
        np.savez_compressed(
            record_path,
            image_id=np.asarray(image_id),
            labels_loaded=np.asarray(True),
        )
        records.append(
            {
                "image_id": image_id,
                "file": record_path.name,
                "sha256": _sha256(record_path),
            }
        )
    manifest = {
        "schema_version": SCORE_MANIFEST_SCHEMA_VERSION,
        "record_integrity_schema": SCORE_RECORD_INTEGRITY_SCHEMA,
        "labels_loaded": True,
        "requested_split": "train",
        "split_role": "train",
        "split_authority_verified": True,
        "spatial_mode": "native",
        "checkpoint_selection_rule": "fixed_last",
        "checkpoint_diagnostic_only": False,
        "checkpoint_epoch": 19,
        "warm_flag": True,
        "checkpoint_inference_head": "multi_scale_fused",
        "non_strict_state_loading": False,
        "target_dataset": target,
        "source_datasets": list(sources),
        "weight_path": str(checkpoint.resolve()),
        "weight_sha256": checkpoint_sha,
        "split_file": str(split.resolve()),
        "split_file_sha256": _sha256(split),
        "split_ordered_ids_sha256": ordered_ids_sha256(ids),
        "num_images": len(records),
        "records": records,
        "records_sha256": score_records_sha256(records),
        "ordered_image_ids_sha256": ordered_ids_sha256(ids),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    reference: dict[str, object] = {
        "target_dataset": target,
        "target_domain_key": SOURCE_KEYS[target],
        "score_dir": str(root.resolve()),
        "score_manifest": str(manifest_path.resolve()),
        "score_manifest_sha256": _sha256(manifest_path),
        "score_records_sha256": manifest["records_sha256"],
        "score_ordered_image_ids_sha256": manifest[
            "ordered_image_ids_sha256"
        ],
        "num_records": len(records),
        "split_file_sha256": manifest["split_file_sha256"],
        "split_ordered_ids_sha256": manifest["split_ordered_ids_sha256"],
        "detector_source_datasets": list(sources),
        "detector_source_domain_keys": [SOURCE_KEYS[value] for value in sources],
        "detector_weight_sha256": checkpoint_sha,
    }
    return root, reference


def _write_grid(
    root: Path,
    *,
    inputs: list[dict[str, object]],
    hashes: dict[str, str],
) -> Path:
    root.mkdir(parents=True)
    grid_path = root / "threshold_grid.npy"
    np.save(grid_path, GRID, allow_pickle=False)
    semantic_sha = logit_threshold_grid_sha256(GRID)
    checkpoint_hashes = sorted(hashes.values())
    episode_hashes = sorted((hashes["irstd"], hashes["nudt"]))
    folds = [
        {
            "detector_checkpoint_sha256": hashes["irstd"],
            "source_domain_keys": ["irstd1k"],
            "scored_official_train_domain_keys": ["irstd1k"],
            "held_out_pseudo_target_keys": ["nudt"],
            "role": "inner_pseudo_target_detector",
        },
        {
            "detector_checkpoint_sha256": hashes["nudt"],
            "source_domain_keys": ["nudt"],
            "scored_official_train_domain_keys": ["nudt"],
            "held_out_pseudo_target_keys": ["irstd1k"],
            "role": "inner_pseudo_target_detector",
        },
        {
            "detector_checkpoint_sha256": hashes["full"],
            "source_domain_keys": ["irstd1k", "nudt"],
            "scored_official_train_domain_keys": ["irstd1k", "nudt"],
            "held_out_pseudo_target_keys": [],
            "role": "outer_final_detector",
        },
    ]
    manifest = {
        "schema_version": LOGIT_GRID_SCHEMA_VERSION,
        "artifact_type": LOGIT_GRID_ARTIFACT_TYPE,
        "representation": LOGIT_REPRESENTATION,
        "dtype": "float32",
        "prediction_rule": LOGIT_PREDICTION_RULE,
        "grid_source": "source_official_train_only",
        "grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "grid_file": "threshold_grid.npy",
        "digest_file": "threshold_grid.sha256",
        "grid_points": int(GRID.size),
        "finite_grid_points": int(GRID.size),
        "max_model_grid_points": MAX_MODEL_GRID_POINTS,
        "grid_sha256": semantic_sha,
        "grid_file_sha256": _sha256(grid_path),
        "empty_action": empty_action_contract(),
        "source_domains": ["IRSTD-1K", "NUDT-SIRST"],
        "source_domain_keys": ["irstd1k", "nudt"],
        "expected_source_domains": ["IRSTD-1K", "NUDT-SIRST"],
        "expected_source_domain_keys": ["irstd1k", "nudt"],
        "outer_target": "NUAA-SIRST",
        "outer_target_key": "nuaa",
        "outer_target_excluded": True,
        "outer_target_labels_used": False,
        "grid_detector_protocol": GRID_DETECTOR_PROTOCOL,
        "detector_checkpoint_count": 3,
        "detector_checkpoint_sha256s": checkpoint_hashes,
        "outer_detector_checkpoint_sha256": hashes["full"],
        "episode_detector_checkpoint_sha256s": episode_hashes,
        "detector_folds": folds,
        "input_score_artifacts": inputs,
        "source_provenance_sha256": canonical_json_sha256(inputs),
        "formal_protocol_eligible": True,
    }
    manifest_path = root / "threshold_grid.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "threshold_grid.sha256").write_text(
        semantic_sha + "\n", encoding="ascii"
    )
    return manifest_path


def _episode_integrity_reference(
    score_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    manifest_path = score_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    integrity = {
        "verified": True,
        "mask_alignment_verified": True,
        "labels_loaded": True,
        "pseudo_target": manifest["target_dataset"],
        "score_dir": str(score_dir.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "records_sha256": manifest["records_sha256"],
        "ordered_image_ids_sha256": manifest["ordered_image_ids_sha256"],
        "num_records": manifest["num_images"],
    }
    fold = {
        "verified": True,
        "pseudo_target": manifest["target_dataset"],
        "target_dataset": manifest["target_dataset"],
        "source_datasets": manifest["source_datasets"],
        "detector_weight_sha256": manifest["weight_sha256"],
        "manifest_sha256": _sha256(manifest_path),
    }
    return integrity, fold


def _write_episode(
    path: Path,
    *,
    validation_domain: str,
    ids: tuple[str, ...],
    grid_manifest_sha: str,
    checkpoint_hashes: dict[str, str],
    episode_score_dirs: tuple[Path, Path],
) -> None:
    rows = 2
    statistics = np.asarray([[0.0, 0.5], [1.0, 1.5]], dtype=np.float32)
    pixel = np.asarray([[-2.0, -3.0, -4.0]] * rows, dtype=np.float32)
    component = np.asarray([[1.0, 0.0, -1.0]] * rows, dtype=np.float32)
    pd = np.asarray([[1.0, 0.5, 0.0]] * rows, dtype=np.float32)
    integrity_and_folds = [
        _episode_integrity_reference(score_dir)
        for score_dir in episode_score_dirs
    ]
    integrity = [item[0] for item in integrity_and_folds]
    folds = [item[1] for item in integrity_and_folds]
    provenance = {
        "protocol": "causal_adaptation_then_future_evaluation",
        "archive_split": "validation",
        "representation": LOGIT_REPRESENTATION,
        "pseudo_target_split": "train",
        "expected_split_role": "train",
        "fold_provenance_verified": True,
        "score_artifact_integrity_verified": True,
        "formal_causal_contract_verified": True,
        "protocol_scope": "formal_causal",
        "threshold_grid_outer_target_excluded": True,
        "threshold_grid_outer_target_key": "nuaa",
        "threshold_grid_source_domains": ["irstd1k", "nudt"],
        "threshold_grid_manifest_sha256": grid_manifest_sha,
        "pseudo_targets": ["IRSTD-1K", "NUDT-SIRST"],
        "paired_lodo_validation_domains": ["IRSTD-1K", "NUDT-SIRST"],
        "validation_domain": validation_domain,
        "allow_unverified_fold_provenance": False,
        "allow_cross_episode_role_reuse": False,
        "cross_episode_role_reuse_detected": False,
        "cross_episode_role_reuse_ids": [],
        "score_artifact_integrity_audits": integrity,
        "fold_provenance_audits": folds,
        "score_map_dirs": [str(value.resolve()) for value in episode_score_dirs],
    }
    feature_hash = feature_schema_sha256(
        LOGIT_STATISTICS_SCHEMA_VERSION, statistics_names=NAMES
    )
    all_hashes = sorted(checkpoint_hashes.values())
    episode_hashes = sorted(
        (checkpoint_hashes["irstd"], checkpoint_hashes["nudt"])
    )
    np.savez_compressed(
        path,
        statistics=statistics,
        statistics_names=np.asarray(NAMES),
        statistics_names_sha256=np.asarray(statistics_names_sha256(NAMES)),
        statistics_schema_version=np.asarray(LOGIT_STATISTICS_SCHEMA_VERSION),
        feature_schema_sha256=np.asarray(feature_hash),
        pixel_log_risk=pixel,
        component_log_risk=component,
        pd_curve=pd,
        thresholds=GRID,
        representation=np.asarray(LOGIT_REPRESENTATION),
        threshold_grid_schema_version=np.asarray(LOGIT_GRID_SCHEMA_VERSION),
        threshold_grid_sha256=np.asarray(logit_threshold_grid_sha256(GRID)),
        threshold_grid_manifest_sha256=np.asarray(grid_manifest_sha),
        threshold_grid_detector_protocol=np.asarray(GRID_DETECTOR_PROTOCOL),
        threshold_grid_detector_checkpoint_sha256s=np.asarray(all_hashes),
        threshold_grid_outer_detector_checkpoint_sha256=np.asarray(
            checkpoint_hashes["full"]
        ),
        threshold_grid_episode_detector_checkpoint_sha256s=np.asarray(
            episode_hashes
        ),
        episode_schema_version=np.asarray(LOGIT_EPISODE_SCHEMA_VERSION),
        pseudo_targets=np.asarray([validation_domain] * rows),
        adaptation_ids=np.asarray(
            [json.dumps([ids[0]]), json.dumps([ids[2]])]
        ),
        evaluation_ids=np.asarray(
            [json.dumps([ids[1]]), json.dumps([ids[3]])]
        ),
        adaptation_sizes=np.ones(rows, dtype=np.int64),
        evaluation_sizes=np.ones(rows, dtype=np.int64),
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )


def _build_bundle(
    tmp_path: Path,
    *,
    checkpoint_changes: dict[str, dict[str, object]] | None = None,
) -> _Bundle:
    root = tmp_path / "project"
    root.mkdir(parents=True)
    split_root = root / "datasets" / "splits"
    split_root.mkdir(parents=True)
    split_ids = {
        "IRSTD-1K": ("i-0", "i-1", "i-2", "i-3"),
        "NUDT-SIRST": ("n-0", "n-1", "n-2", "n-3"),
    }
    splits: dict[str, Path] = {}
    for domain, ids in split_ids.items():
        path = split_root / f"train_{domain}.txt"
        path.write_text("\n".join(ids) + "\n", encoding="utf-8")
        splits[domain] = path

    checkpoint_root = root / "checkpoints"
    checkpoint_changes = checkpoint_changes or {}
    checkpoint_data = {
        "irstd": _write_checkpoint(
            checkpoint_root / "inner-irstd.pt",
            "irstd",
            sources=("IRSTD-1K",),
            splits=splits,
            split_ids=split_ids,
            changes=checkpoint_changes.get("irstd"),
        ),
        "nudt": _write_checkpoint(
            checkpoint_root / "inner-nudt.pt",
            "nudt",
            sources=("NUDT-SIRST",),
            splits=splits,
            split_ids=split_ids,
            changes=checkpoint_changes.get("nudt"),
        ),
        "full": _write_checkpoint(
            checkpoint_root / "full.pt",
            "full",
            sources=SOURCES,
            splits=splits,
            split_ids=split_ids,
            changes=checkpoint_changes.get("full"),
        ),
    }
    checkpoint_paths = {key: value[0] for key, value in checkpoint_data.items()}
    checkpoint_hashes = {key: value[1] for key, value in checkpoint_data.items()}

    score_root = root / "scores"
    grid_score_specs = (
        ("grid-inner-irstd", "IRSTD-1K", ("IRSTD-1K",), "irstd"),
        ("grid-inner-nudt", "NUDT-SIRST", ("NUDT-SIRST",), "nudt"),
        ("grid-full-irstd", "IRSTD-1K", SOURCES, "full"),
        ("grid-full-nudt", "NUDT-SIRST", SOURCES, "full"),
    )
    grid_inputs: list[dict[str, object]] = []
    score_dirs: list[Path] = []
    for name, target, sources, checkpoint_key in grid_score_specs:
        score_dir, reference = _write_score_dir(
            score_root / name,
            target=target,
            sources=sources,
            checkpoint=checkpoint_paths[checkpoint_key],
            checkpoint_sha=checkpoint_hashes[checkpoint_key],
            split=splits[target],
            ids=split_ids[target],
        )
        score_dirs.append(score_dir)
        grid_inputs.append(reference)

    episode_irstd, _ = _write_score_dir(
        score_root / "episode-irstd-from-nudt",
        target="IRSTD-1K",
        sources=("NUDT-SIRST",),
        checkpoint=checkpoint_paths["nudt"],
        checkpoint_sha=checkpoint_hashes["nudt"],
        split=splits["IRSTD-1K"],
        ids=split_ids["IRSTD-1K"],
    )
    episode_nudt, _ = _write_score_dir(
        score_root / "episode-nudt-from-irstd",
        target="NUDT-SIRST",
        sources=("IRSTD-1K",),
        checkpoint=checkpoint_paths["irstd"],
        checkpoint_sha=checkpoint_hashes["irstd"],
        split=splits["NUDT-SIRST"],
        ids=split_ids["NUDT-SIRST"],
    )
    score_dirs.extend((episode_irstd, episode_nudt))

    grid_manifest = _write_grid(
        root / "grid", inputs=grid_inputs, hashes=checkpoint_hashes
    )
    grid_manifest_sha = _sha256(grid_manifest)
    episode_root = root / "episodes"
    episode_root.mkdir()
    episodes = {
        "IRSTD-1K": episode_root / "val-irstd.npz",
        "NUDT-SIRST": episode_root / "val-nudt.npz",
    }
    for domain, path in episodes.items():
        _write_episode(
            path,
            validation_domain=domain,
            ids=split_ids[domain],
            grid_manifest_sha=grid_manifest_sha,
            checkpoint_hashes=checkpoint_hashes,
            episode_score_dirs=(episode_irstd, episode_nudt),
        )
    return _Bundle(
        root=root,
        grid_manifest=grid_manifest,
        splits=splits,
        checkpoints=list(checkpoint_paths.values()),
        episodes=episodes,
        score_dirs=score_dirs,
    )


def _rewrite_npz(path: Path, **changes: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload.update(changes)
    np.savez_compressed(path, **payload)


def test_transitive_source_provenance_accepts_complete_canonical_chain(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(tmp_path)
    audit = verify_source_only_provenance_v4(**bundle.kwargs())

    assert audit["schema_version"] == SOURCE_PROVENANCE_SCHEMA_VERSION
    assert audit["verified"] is True
    assert audit["formal_source_domains"] == ["IRSTD-1K", "NUDT-SIRST"]
    assert audit["excluded_outer_target"] == "NUAA-SIRST"
    assert audit["outer_target_labels_read"] is False
    assert audit["immutable_byte_snapshot_verified"] is True
    assert audit["all_paths_unchanged_after_snapshot"] is True
    assert len(audit["detector_checkpoints"]) == 3
    assert audit["checkpoint_metadata_verified_from_bytes"] is True
    assert audit["checkpoint_safe_load_weights_only"] is True
    assert audit["checkpoint_test_split_artifacts_read"] is False
    assert {
        frozenset(item["source_names"])
        for item in audit["detector_checkpoints"]
    } == {
        frozenset(("IRSTD-1K",)),
        frozenset(("NUDT-SIRST",)),
        frozenset(SOURCES),
    }
    for checkpoint in audit["detector_checkpoints"]:
        assert checkpoint["checkpoint_selection"] == "fixed_last"
        assert checkpoint["test_labels_used_for_selection"] is False
        assert checkpoint["diagnostic_test_eval"] is False
        assert checkpoint["diagnostic_only"] is False
        assert checkpoint["formal_paper_checkpoint"] is True
        assert checkpoint["format_version"] == 2
        assert checkpoint["epoch"] == 19
        assert checkpoint["warm_flag"] is True
        assert checkpoint["inference_head"] == "multi_scale_fused"
        assert checkpoint["safe_load"] == {
            "weights_only": True,
            "map_location": "cpu",
            "unsafe_pickle_fallback_used": False,
        }
        assert checkpoint["test_split_artifacts_read"] is False
    assert len(audit["grid_score_artifacts"]) == 4
    assert len(audit["episode_score_artifacts"]) == 2
    assert len(audit["validation_archives"]) == 2
    assert audit["global_unique_episode_a_e_ids"] == 8
    assert audit["threshold_grid"]["manifest_path"] == str(
        bundle.grid_manifest.resolve()
    )


def test_transitive_source_provenance_rejects_checkpoint_byte_tamper(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(tmp_path)
    bundle.checkpoints[0].write_bytes(b"tampered checkpoint")
    with pytest.raises(ValueError, match="safely loaded with weights_only=True"):
        verify_source_only_provenance_v4(**bundle.kwargs())


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"checkpoint_selection": "best"}, "checkpoint_selection"),
        ({"selection_rule": "best"}, "selection_rule"),
        ({"test_labels_used_for_selection": True}, "test_labels"),
        ({"diagnostic_test_eval": True}, "diagnostic_test_eval"),
        ({"diagnostic_only": True}, "diagnostic_only"),
        ({"formal_paper_checkpoint": False}, "formal_paper_checkpoint"),
        ({"format_version": 1}, "format_version"),
        ({"epoch": 18}, "epoch"),
        ({"warm_flag": False}, "warm_flag"),
        ({"inference_head": "single"}, "inference_head"),
    ],
)
def test_transitive_source_provenance_rejects_nonformal_checkpoint_metadata(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    bundle = _build_bundle(
        tmp_path, checkpoint_changes={"irstd": changes}
    )
    with pytest.raises(ValueError, match=message):
        verify_source_only_provenance_v4(**bundle.kwargs())


def test_transitive_source_provenance_rejects_checkpoint_split_hash_tamper(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(
        tmp_path,
        checkpoint_changes={"nudt": {"__tamper_train_split_sha": True}},
    )
    with pytest.raises(ValueError, match="train split byte SHA mismatch"):
        verify_source_only_provenance_v4(**bundle.kwargs())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_epoch", 18),
        ("warm_flag", False),
        ("checkpoint_inference_head", "single"),
    ],
)
def test_transitive_source_provenance_cross_checks_score_checkpoint_metadata(
    tmp_path: Path, field: str, value: object
) -> None:
    bundle = _build_bundle(tmp_path)
    manifest_path = bundle.score_dirs[0] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=field):
        verify_source_only_provenance_v4(**bundle.kwargs())


def test_transitive_source_provenance_does_not_trust_exported_source_names(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(tmp_path)
    manifest_path = bundle.score_dirs[0] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_datasets"] = ["NUDT-SIRST"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(
        ValueError, match="source_datasets disagree.*checkpoint bytes"
    ):
        verify_source_only_provenance_v4(**bundle.kwargs())


def test_transitive_source_provenance_rejects_score_record_byte_tamper(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(tmp_path)
    record = bundle.score_dirs[0] / "000.npz"
    record.write_bytes(record.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="score-record SHA mismatch"):
        verify_source_only_provenance_v4(**bundle.kwargs())


def test_transitive_source_provenance_rejects_outer_id_and_global_reuse(
    tmp_path: Path,
) -> None:
    outer_bundle = _build_bundle(tmp_path / "outer")
    episode = outer_bundle.episodes["IRSTD-1K"]
    _rewrite_npz(
        episode,
        adaptation_ids=np.asarray(
            [json.dumps(["NUAA-leak"]), json.dumps(["i-2"])]
        ),
    )
    with pytest.raises(ValueError, match="Outer target appears"):
        verify_source_only_provenance_v4(**outer_bundle.kwargs())

    reuse_bundle = _build_bundle(tmp_path / "reuse")
    episode = reuse_bundle.episodes["NUDT-SIRST"]
    _rewrite_npz(
        episode,
        adaptation_ids=np.asarray(
            [json.dumps(["n-0"]), json.dumps(["n-0"])]
        ),
    )
    with pytest.raises(ValueError, match="globally unique"):
        verify_source_only_provenance_v4(**reuse_bundle.kwargs())


def test_transitive_source_provenance_rejects_outer_and_escaping_paths(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(tmp_path)
    outer_checkpoint = bundle.root / "checkpoints" / "nuaa-leak.pt"
    outer_checkpoint.write_bytes(b"no outer source is permitted")
    values = bundle.kwargs()
    values["detector_checkpoints"] = [
        outer_checkpoint,
        bundle.checkpoints[1],
        bundle.checkpoints[2],
    ]
    with pytest.raises(ValueError, match="Outer target appears"):
        verify_source_only_provenance_v4(**values)

    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    values["detector_checkpoints"] = [
        outside,
        bundle.checkpoints[1],
        bundle.checkpoints[2],
    ]
    with pytest.raises(ValueError, match="escapes project_root"):
        verify_source_only_provenance_v4(**values)


def test_snapshot_registry_detects_content_and_logical_path_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    path = root / "artifact.bin"
    path.write_bytes(b"sealed")
    registry = _SnapshotRegistry(root)
    payload, _stamp = registry.capture(path)
    assert payload == b"sealed"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after its byte snapshot"):
        registry.assert_unchanged()

    target_a = root / "a.bin"
    target_b = root / "b.bin"
    target_a.write_bytes(b"a")
    target_b.write_bytes(b"b")
    link = root / "logical.bin"
    link.symlink_to(target_a.name)
    registry = _SnapshotRegistry(root)
    registry.capture(link)
    link.unlink()
    link.symlink_to(target_b.name)
    with pytest.raises(ValueError, match="path drifted"):
        registry.assert_unchanged()


def test_replay_cli_atomically_publishes_self_checked_complete_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _build_bundle(tmp_path)
    output = bundle.root / "outputs" / "provenance" / "source-chain.json"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    arguments = [
        "--project-root",
        str(bundle.root),
        "--threshold-grid-manifest",
        str(bundle.grid_manifest),
        "--official-train-split",
        f"IRSTD-1K={bundle.splits['IRSTD-1K']}",
        "--official-train-split",
        f"NUDT-SIRST={bundle.splits['NUDT-SIRST']}",
    ]
    for checkpoint in bundle.checkpoints:
        arguments.extend(("--detector-checkpoint", str(checkpoint)))
    for archive in bundle.episodes.values():
        arguments.extend(("--episode-archive", str(archive)))
    arguments.extend(("--output", str(output)))

    assert source_provenance_main(arguments) == 0
    printed = json.loads(capsys.readouterr().out)
    evidence, file_sha = load_source_provenance_run_evidence(output)

    assert printed["output_sha256"] == file_sha == _sha256(output)
    assert printed["source_chain_sha256"] == evidence["verification"][
        "source_chain_sha256"
    ]
    assert evidence["schema_version"] == SOURCE_PROVENANCE_RUN_SCHEMA_VERSION
    assert evidence["verification"]["verified"] is True
    assert evidence["execution"]["cpu_only_requested"] is True
    assert evidence["execution"]["elapsed_seconds"] >= 0.0
    assert evidence["execution"]["peak_rss_kib"] > 0
    assert evidence["verifier_module"]["sha256"] == _sha256(
        Path(evidence["verifier_module"]["path"])
    )
    assert evidence["outer_target_access_declaration"][
        "outer_target_labels_read"
    ] is False
    assert not list(output.parent.glob(f".{output.name}.*"))

    tampered = json.loads(json.dumps(evidence))
    tampered["outer_target_access_declaration"][
        "outer_target_labels_read"
    ] = True
    with pytest.raises(ValueError, match="non-access declaration"):
        validate_source_provenance_run_evidence(tampered)

    tampered_checkpoint = json.loads(json.dumps(evidence))
    tampered_checkpoint["verification"]["detector_checkpoints"][0][
        "epoch"
    ] = 18
    with pytest.raises(ValueError, match="checkpoint evidence.*epoch"):
        validate_source_provenance_run_evidence(tampered_checkpoint)

    tampered_chain = json.loads(json.dumps(evidence))
    tampered_chain["verification"]["source_chain_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match its evidence"):
        validate_source_provenance_run_evidence(tampered_chain)
