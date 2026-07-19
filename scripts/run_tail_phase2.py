"""Run the preregistered source-only detector-tail phase-2 evidence chain.

The script is intentionally narrower than a general experiment launcher.  It
accepts only the two branches frozen in
``detector_tail_branches_seed42.json`` and constructs exactly this DAG::

    six source-train score exports -> 2048-point raw-logit grid
      -> paired A32/E1/stride33 episodes -> fresh provenance replay
      -> AnchorWarp/RC-Direct train-only fits -> held-source binding/evaluation
      -> strict two-fold Gate C aggregation

No NUAA path is accepted or opened.  The string ``NUAA-SIRST`` is passed only
as the excluded outer-domain name recorded by the source-only grid contract.
Running without ``--execute`` is a read-only dry run.  Execution is fail-closed
and resumable: already-complete artifacts are revalidated and skipped, while a
partial or invalid artifact is never overwritten automatically.
"""

from __future__ import annotations

import argparse
import fcntl
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
# ``python scripts/run_tail_phase2.py`` sets sys.path[0] to ``scripts/``
# rather than the repository root.  Validators are intentionally imported
# lazily after each subprocess, so without this explicit root entry a valid
# score artifact can be rejected only because ``evaluation`` is not importable.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PREREGISTRATION = (
    ROOT
    / "outputs"
    / "v4_source_only"
    / "preregistration"
    / "detector_tail_branches_seed42.json"
)
PREREGISTRATION_SHA256 = (
    "ac65d2d60b90a308c687aacb23d8612a57406134b1782e15d5dcc910798756be"
)
DEFAULT_PYTHON = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
SOURCE_DOMAINS = ("NUDT-SIRST", "IRSTD-1K")
OUTER_DOMAIN_NAME_ONLY = "NUAA-SIRST"
PIXEL_BUDGETS = (1.0e-5, 1.0e-6)
COMPONENT_BUDGETS = (5.0, 1.0)

DATASETS = {
    "NUDT-SIRST": ROOT / "datasets/NUDT-SIRST",
    "IRSTD-1K": ROOT / "datasets/IRSTD-1K",
}
SPLITS = {
    "NUDT-SIRST": DATASETS["NUDT-SIRST"]
    / "img_idx/train_NUDT-SIRST.txt",
    "IRSTD-1K": DATASETS["IRSTD-1K"]
    / "img_idx/train_IRSTD-1K.txt",
}


@dataclass(frozen=True)
class CheckpointSpec:
    role: str
    path: Path
    sources: tuple[str, ...]
    config_path: Path | None
    registered_sha256: str | None = None


@dataclass(frozen=True)
class BranchSpec:
    cli_name: str
    branch_id: str
    namespace: Path
    prereg_position: int
    checkpoints: tuple[CheckpointSpec, ...]
    loss_contract: Mapping[str, Any]


@dataclass(frozen=True)
class Step:
    step_id: str
    kind: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...]
    details: Mapping[str, Any]

    def to_json(self, *, state: str, reason: str | None = None) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "kind": self.kind,
            "state": state,
            "reason": reason,
            "command": list(self.command),
            "outputs": [str(path) for path in self.outputs],
            "details": dict(self.details),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _registered_branch(prereg: Mapping[str, Any], position: int) -> Mapping[str, Any]:
    matches = [
        item
        for item in prereg.get("branch_sequence", [])
        if isinstance(item, Mapping) and item.get("position") == position
    ]
    if len(matches) != 1:
        raise ValueError(f"Preregistration has no unique branch position {position}")
    return matches[0]


def load_branch_spec(name: str) -> tuple[BranchSpec, dict[str, Any]]:
    if ROOT != Path("/home/ly/RC-IRSTD-v2"):
        raise RuntimeError(f"Unexpected project root: {ROOT}")
    if _sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("Detector-tail preregistration bytes have drifted")
    prereg = _load_json(PREREGISTRATION)
    if prereg.get("excluded_outer_target") != OUTER_DOMAIN_NAME_ONLY:
        raise ValueError("Preregistration outer target declaration drifted")
    if prereg.get("canonical_sources") != list(SOURCE_DOMAINS):
        raise ValueError("Preregistration canonical source order drifted")
    shared = prereg.get("shared_protocol")
    required_shared = {
        "seed": 42,
        "detector_checkpoint_selection": "fixed_last_epoch_19_of_20",
        "detector_validation_or_test_labels_used_for_selection": False,
        "representation": "raw_logit_float32",
        "grid_points": 2048,
        "grid_detector_protocol": "all_source_only_detector_folds",
        "episode_adaptation_window": 32,
        "episode_evaluation_window": 1,
        "episode_stride": 33,
        "episode_a_e_global_id_reuse_allowed": False,
        "component_connectivity_argument": 2,
        "min_component_area": 1,
        "held_validation_binding_after_state_freeze_required": True,
        "two_phase_action_digest_before_future_e_load_required": True,
        "same_detector_grid_episode_seed_and_budgets_for_both_methods": True,
    }
    if not isinstance(shared, Mapping):
        raise ValueError("Preregistration shared_protocol is missing")
    for field, expected in required_shared.items():
        if shared.get(field) != expected:
            raise ValueError(f"Preregistered shared_protocol.{field} drifted")
    if shared.get("pixel_budgets") != list(PIXEL_BUDGETS):
        raise ValueError("Preregistered pixel budgets drifted")
    if shared.get("component_budgets") != list(COMPONENT_BUDGETS):
        raise ValueError("Preregistered component budgets drifted")

    if name == "tailmiss":
        registered = _registered_branch(prereg, 1)
        checkpoints = (
            CheckpointSpec(
                "inner_from_irstd",
                ROOT / "outputs/stage4_inner_nudt_from_irstd_tailmiss_20ep/last.pt",
                ("IRSTD-1K",),
                ROOT / "configs/stage4_inner_nudt_from_irstd_tailmiss_20ep.yaml",
            ),
            CheckpointSpec(
                "inner_from_nudt",
                ROOT / "outputs/stage4_inner_irstd_from_nudt_tailmiss_20ep/last.pt",
                ("NUDT-SIRST",),
                ROOT / "configs/stage4_inner_irstd_from_nudt_tailmiss_20ep.yaml",
            ),
            CheckpointSpec(
                "full_sources",
                ROOT
                / "outputs/v4_tailmiss_source_only/checkpoints/"
                "full_sources_tail_miss_seed42_epoch19.pt",
                SOURCE_DOMAINS,
                None,
                str(registered["full_source_checkpoint"]["sha256"]),
            ),
        )
    elif name == "candidate_a":
        registered = _registered_branch(prereg, 2)
        checkpoints = (
            CheckpointSpec(
                "inner_from_irstd",
                ROOT
                / "outputs/stage4_inner_nudt_from_irstd_tailrank_margin_a_20ep/last.pt",
                ("IRSTD-1K",),
                ROOT
                / "configs/stage4_inner_nudt_from_irstd_tailrank_margin_a_20ep.yaml",
            ),
            CheckpointSpec(
                "inner_from_nudt",
                ROOT
                / "outputs/stage4_inner_irstd_from_nudt_tailrank_margin_a_20ep/last.pt",
                ("NUDT-SIRST",),
                ROOT
                / "configs/stage4_inner_irstd_from_nudt_tailrank_margin_a_20ep.yaml",
            ),
            CheckpointSpec(
                "full_sources",
                ROOT / "outputs/stage4_full_sources_tailrank_margin_a_20ep/last.pt",
                SOURCE_DOMAINS,
                ROOT / "configs/stage4_full_sources_tailrank_margin_a_20ep.yaml",
            ),
        )
    else:  # pragma: no cover - argparse prevents this
        raise ValueError(f"Unknown branch: {name}")

    expected_namespace = Path(str(registered["output_namespace"])).resolve()
    spec = BranchSpec(
        cli_name=name,
        branch_id=str(registered["branch_id"]),
        namespace=expected_namespace,
        prereg_position=int(registered["position"]),
        checkpoints=checkpoints,
        loss_contract=dict(registered["loss_contract"]),
    )
    if spec.namespace.parent != ROOT / "outputs":
        raise ValueError("Preregistered output namespace escaped the project outputs")
    return spec, prereg


def _checkpoint_by_role(spec: BranchSpec, role: str) -> CheckpointSpec:
    matches = [item for item in spec.checkpoints if item.role == role]
    if len(matches) != 1:
        raise RuntimeError(f"No unique checkpoint role {role!r}")
    return matches[0]


def _export_command(
    python_bin: Path,
    *,
    checkpoint: CheckpointSpec,
    target: str,
    output: Path,
    device: str,
    num_workers: int,
) -> tuple[str, ...]:
    command = [
        str(python_bin),
        "-m",
        "rc_irstd.cli.export_scores",
        "--checkpoint",
        str(checkpoint.path),
        "--dataset-dir",
        str(DATASETS[target]),
        "--output-dir",
        str(output),
        "--split",
        "train",
        "--split-file",
        str(SPLITS[target]),
        "--dataset-name",
        target,
    ]
    for source in checkpoint.sources:
        command.extend(("--source-dataset", source))
    command.extend(
        (
            "--device",
            device,
            "--num-workers",
            str(num_workers),
            "--batch-size",
            "1",
            "--export-raw-logits",
        )
    )
    return tuple(command)


def build_plan(
    spec: BranchSpec,
    *,
    python_bin: Path,
    count_all_workers: int = 16,
    export_workers: int = 4,
) -> tuple[Step, ...]:
    if count_all_workers <= 0 or export_workers < 0:
        raise ValueError("worker counts must be positive/non-negative")
    namespace = spec.namespace
    score_root = namespace / "scores"
    grid_root = namespace / "grid/source_global_raw_logit_2048"
    episodes_root = namespace / "episodes"
    val_irstd_root = episodes_root / "val_irstd"
    val_nudt_root = episodes_root / "val_nudt"
    gate_root = namespace / "gate_c"

    inner_irstd = _checkpoint_by_role(spec, "inner_from_irstd")
    inner_nudt = _checkpoint_by_role(spec, "inner_from_nudt")
    full = _checkpoint_by_role(spec, "full_sources")
    exports = (
        (
            "export_inner_irstd_self",
            inner_irstd,
            "IRSTD-1K",
            score_root / "source_grid/inner_from_irstd/irstd_train",
            "cuda:0",
            "grid_self_score",
        ),
        (
            "export_inner_nudt_self",
            inner_nudt,
            "NUDT-SIRST",
            score_root / "source_grid/inner_from_nudt/nudt_train",
            "cuda:1",
            "grid_self_score",
        ),
        (
            "export_full_irstd_self",
            full,
            "IRSTD-1K",
            score_root / "source_grid/full_sources/irstd_train",
            "cuda:2",
            "grid_self_score",
        ),
        (
            "export_full_nudt_self",
            full,
            "NUDT-SIRST",
            score_root / "source_grid/full_sources/nudt_train",
            "cuda:2",
            "grid_self_score",
        ),
        (
            "export_nudt_pseudo_target",
            inner_irstd,
            "NUDT-SIRST",
            score_root / "pseudo_targets/nudt_from_irstd_train",
            "cuda:0",
            "held_source_pseudo_target",
        ),
        (
            "export_irstd_pseudo_target",
            inner_nudt,
            "IRSTD-1K",
            score_root / "pseudo_targets/irstd_from_nudt_train",
            "cuda:1",
            "held_source_pseudo_target",
        ),
    )
    steps: list[Step] = []
    for step_id, checkpoint, target, output, device, role in exports:
        steps.append(
            Step(
                step_id,
                "score_export",
                _export_command(
                    python_bin,
                    checkpoint=checkpoint,
                    target=target,
                    output=output,
                    device=device,
                    num_workers=export_workers,
                ),
                (output,),
                {
                    "checkpoint_role": checkpoint.role,
                    "checkpoint": str(checkpoint.path),
                    "checkpoint_sources": list(checkpoint.sources),
                    "target_source_domain": target,
                    "artifact_role": role,
                    "labels_loaded": True,
                    "outer_labels_loaded": False,
                },
            )
        )

    grid_inputs = tuple(item[3] for item in exports[:4])
    grid_command: list[str] = [
        str(python_bin),
        "-m",
        "risk_curve.build_logit_threshold_grid",
    ]
    for path in grid_inputs:
        grid_command.extend(("--source-score-dir", str(path)))
    for source in ("IRSTD-1K", "NUDT-SIRST"):
        grid_command.extend(("--expected-source-domain", source))
    grid_command.extend(
        (
            "--outer-target",
            OUTER_DOMAIN_NAME_ONLY,
            "--output-dir",
            str(grid_root),
            "--max-grid-points",
            "2048",
        )
    )
    steps.append(
        Step(
            "build_raw_logit_grid_2048",
            "grid",
            tuple(grid_command),
            (
                grid_root / "threshold_grid.npy",
                grid_root / "threshold_grid.json",
                grid_root / "threshold_grid.sha256",
            ),
            {
                "outer_target_name_only": OUTER_DOMAIN_NAME_ONLY,
                "outer_target_path_opened": False,
                "grid_points": 2048,
                "grid_detector_protocol": "all_source_only_detector_folds",
                "source_score_dirs": [str(value) for value in grid_inputs],
            },
        )
    )

    pseudo_nudt = exports[4][3]
    pseudo_irstd = exports[5][3]
    episodes_command = (
        str(python_bin),
        "-m",
        "risk_curve.build_curve_episodes",
        "--score-map-dir",
        str(pseudo_nudt),
        "--pseudo-target",
        "NUDT-SIRST",
        "--score-map-dir",
        str(pseudo_irstd),
        "--pseudo-target",
        "IRSTD-1K",
        "--output-dir",
        str(val_irstd_root),
        "--paired-output-dir",
        str(val_nudt_root),
        "--threshold-grid-manifest",
        str(grid_root / "threshold_grid.json"),
        "--representation",
        "raw_logit_float32",
        "--expected-split-role",
        "train",
        "--adaptation-window",
        "32",
        "--evaluation-window",
        "1",
        "--stride",
        "33",
        "--validation-domain",
        "IRSTD-1K",
        "--connectivity",
        "2",
        "--min-component-area",
        "1",
        "--count-all-workers",
        str(count_all_workers),
    )
    steps.append(
        Step(
            "build_paired_source_episodes",
            "episodes",
            episodes_command,
            (
                val_irstd_root / "train.npz",
                val_irstd_root / "val.npz",
                val_irstd_root / "manifest.json",
                val_nudt_root / "train.npz",
                val_nudt_root / "val.npz",
                val_nudt_root / "manifest.json",
            ),
            {
                "adaptation_window": 32,
                "evaluation_window": 1,
                "stride": 33,
                "cross_episode_role_reuse_allowed": False,
                "paired_validation_domains": ["IRSTD-1K", "NUDT-SIRST"],
                "source_score_dirs": [str(pseudo_nudt), str(pseudo_irstd)],
                "threshold_grid_manifest": str(
                    grid_root / "threshold_grid.json"
                ),
            },
        )
    )

    provenance_output = namespace / "provenance/source_chain_verification.json"
    provenance_command: list[str] = [
        str(python_bin),
        "-m",
        "risk_curve.source_provenance_v4",
        "--project-root",
        str(ROOT),
        "--threshold-grid-manifest",
        str(grid_root / "threshold_grid.json"),
        "--official-train-split",
        f"IRSTD-1K={SPLITS['IRSTD-1K']}",
        "--official-train-split",
        f"NUDT-SIRST={SPLITS['NUDT-SIRST']}",
    ]
    for checkpoint in spec.checkpoints:
        provenance_command.extend(("--detector-checkpoint", str(checkpoint.path)))
    provenance_command.extend(
        (
            "--episode-archive",
            str(val_irstd_root / "val.npz"),
            "--episode-archive",
            str(val_nudt_root / "val.npz"),
            "--output",
            str(provenance_output),
        )
    )
    steps.append(
        Step(
            "replay_source_only_provenance",
            "source_provenance",
            tuple(provenance_command),
            (provenance_output,),
            {
                "accepted_source_paths": [str(SPLITS[value]) for value in SOURCE_DOMAINS],
                "outer_split_labels_masks_read": False,
                "checkpoint_count": 3,
                "validation_archive_count": 2,
                "threshold_grid_manifest": str(
                    grid_root / "threshold_grid.json"
                ),
                "detector_checkpoints": [
                    str(value.path) for value in spec.checkpoints
                ],
                "episode_archives": [
                    str(val_irstd_root / "val.npz"),
                    str(val_nudt_root / "val.npz"),
                ],
            },
        )
    )

    comparisons: list[Path] = []
    for fold_id, fold_root in (
        ("val_irstd", val_irstd_root),
        ("val_nudt", val_nudt_root),
    ):
        fold_gate = gate_root / fold_id
        train_archive = fold_root / "train.npz"
        val_archive = fold_root / "val.npz"
        anchor_frozen = fold_gate / "anchor_warp_train_only_seed42.pt"
        anchor_bound = fold_gate / "anchor_warp_bound_seed42.pt"
        direct = fold_gate / "rc_direct_train_only_seed42.pt"
        comparison = fold_gate / "comparison_seed42.json"
        comparisons.append(comparison)
        steps.extend(
            (
                Step(
                    f"{fold_id}_train_anchor_warp",
                    "anchor_train",
                    (
                        str(python_bin),
                        "-m",
                        "risk_curve.train_anchor_warp_predictor_v4",
                        "--train-file",
                        str(train_archive),
                        "--output",
                        str(anchor_frozen),
                        "--seed",
                        "42",
                        "--num-folds",
                        "5",
                        "--max-epochs",
                        "400",
                        "--patience",
                        "40",
                        "--learning-rate",
                        "0.003",
                        "--weight-decay",
                        "0.001",
                        "--quantile",
                        "0.90",
                        "--device",
                        "cpu",
                    ),
                    (anchor_frozen, anchor_frozen.with_suffix(".pt.metrics.json")),
                    {
                        "training_labels": "source_train_future_E_only",
                        "cv_folds": 5,
                        "train_archive": str(train_archive),
                    },
                ),
                Step(
                    f"{fold_id}_train_rc_direct",
                    "direct_train",
                    (
                        str(python_bin),
                        "-m",
                        "risk_curve.train_direct_calibrator_train_only_v4",
                        "--train-file",
                        str(train_archive),
                        "--val-file",
                        str(val_archive),
                        "--output",
                        str(direct),
                        "--pixel-budgets",
                        "1e-5",
                        "1e-6",
                        "--component-budgets",
                        "5",
                        "1",
                        "--hidden-dims",
                        "256",
                        "128",
                        "--dropout",
                        "0.1",
                        "--max-epochs",
                        "200",
                        "--batch-size",
                        "32",
                        "--lr",
                        "0.001",
                        "--weight-decay",
                        "0.0001",
                        "--under-weight",
                        "4.0",
                        "--seed",
                        "42",
                        "--device",
                        "cpu",
                    ),
                    (direct,),
                    {
                        "selection": "deterministic_train_only_5fold_fixed_epoch",
                        "held_validation_labels_used_for_selection": False,
                        "train_archive": str(train_archive),
                        "validation_archive": str(val_archive),
                    },
                ),
                Step(
                    f"{fold_id}_bind_anchor_validation",
                    "anchor_bind",
                    (
                        str(python_bin),
                        "-m",
                        "risk_curve.bind_anchor_warp_validation_v4",
                        "--frozen-checkpoint",
                        str(anchor_frozen),
                        "--validation-file",
                        str(val_archive),
                        "--output",
                        str(anchor_bound),
                    ),
                    (anchor_bound,),
                    {
                        "state_frozen_before_validation_binding": True,
                        "binding_fields": "label_free_allowlist_only",
                        "frozen_checkpoint": str(anchor_frozen),
                        "validation_archive": str(val_archive),
                    },
                ),
                Step(
                    f"{fold_id}_evaluate_two_phase",
                    "comparison",
                    (
                        str(python_bin),
                        "-m",
                        "risk_curve.evaluate_anchor_warp_source_pseudo_target_v4",
                        "--episode-file",
                        str(val_archive),
                        "--anchor-warp-package",
                        str(anchor_bound),
                        "--rc-direct-checkpoint",
                        str(direct),
                        "--output",
                        str(comparison),
                        "--pixel-budgets",
                        "1e-5",
                        "1e-6",
                        "--component-budgets",
                        "5",
                        "1",
                        "--device",
                        "cpu",
                        "--batch-size",
                        "64",
                    ),
                    (comparison,),
                    {
                        "future_e_loaded_after_action_digest": True,
                        "future_e_reselection": False,
                        "episode_archive": str(val_archive),
                        "anchor_warp_package": str(anchor_bound),
                        "rc_direct_checkpoint": str(direct),
                    },
                ),
            )
        )

    aggregate_output = namespace / "gate_c_anchor_warp_vs_direct_seed42.json"
    steps.append(
        Step(
            "aggregate_strict_gate_c",
            "aggregate_gate_c",
            (
                str(python_bin),
                "-m",
                "risk_curve.aggregate_anchor_warp_gate_c_v4",
                "--fold",
                f"val_irstd={comparisons[0]}",
                "--fold",
                f"val_nudt={comparisons[1]}",
                "--output",
                str(aggregate_output),
            ),
            (aggregate_output,),
            {
                "criteria": [f"c{index}" for index in range(8)],
                "criteria_must_not_be_lowered": True,
                "pd_non_degradation_floor": 0.0,
                "decision": "GO_only_if_all_c0_through_c7_true",
                "fold_comparisons": [str(value) for value in comparisons],
            },
        )
    )
    _assert_source_only_commands(steps)
    return tuple(steps)


def _assert_source_only_commands(steps: Sequence[Step]) -> None:
    forbidden = ("/NUAA", "/nuaa", "datasets/NUAA", "datasets/nuaa")
    for step in steps:
        for token in step.command:
            if any(fragment in token for fragment in forbidden):
                raise ValueError(f"Outer-domain path leaked into step {step.step_id}")
        if "--allow-unverified-fold-provenance" in step.command:
            raise ValueError("Formal phase 2 cannot allow unverified fold provenance")
        if "--allow-cross-episode-role-reuse" in step.command:
            raise ValueError("Formal phase 2 cannot allow A/E role reuse")
        if "--force" in step.command or "--overwrite" in step.command:
            raise ValueError("Phase 2 plan must never overwrite existing evidence")


def _config_hash_requirements(spec: BranchSpec, prereg: Mapping[str, Any]) -> list[str]:
    registered = _registered_branch(prereg, spec.prereg_position)
    entries = registered.get("inner_configs", registered.get("configs", []))
    expected_by_path = {
        str(Path(str(item["path"])).resolve()): str(item["sha256"])
        for item in entries
        if isinstance(item, Mapping)
    }
    problems: list[str] = []
    for checkpoint in spec.checkpoints:
        if checkpoint.config_path is None:
            continue
        expected = expected_by_path.get(str(checkpoint.config_path.resolve()))
        if expected is None:
            problems.append(f"config_not_registered:{checkpoint.config_path}")
        elif not checkpoint.config_path.is_file():
            problems.append(f"config_missing:{checkpoint.config_path}")
        elif _sha256(checkpoint.config_path) != expected:
            problems.append(f"config_sha256_mismatch:{checkpoint.config_path}")
    contract = spec.loss_contract
    if spec.cli_name == "tailmiss":
        module_bindings = {ROOT / "rc_irstd/losses/tail.py": contract["loss_module_sha256"]}
    else:
        module_bindings = {
            ROOT / "rc_irstd/losses/tail_rank.py": contract["tail_rank_module_sha256"],
            ROOT / "rc_irstd/losses/detector.py": contract["detector_objective_module_sha256"],
        }
    for path, expected in module_bindings.items():
        if _sha256(path) != expected:
            problems.append(f"loss_module_sha256_mismatch:{path}")
    return problems


def _audit_checkpoint(spec: BranchSpec, checkpoint: CheckpointSpec) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": checkpoint.role,
        "path": str(checkpoint.path),
        "sources": list(checkpoint.sources),
        "exists": checkpoint.path.is_file(),
        "valid": False,
        "problems": [],
    }
    problems: list[str] = result["problems"]
    if not checkpoint.path.is_file():
        problems.append("missing")
        return result
    checkpoint_sha = _sha256(checkpoint.path)
    result["sha256"] = checkpoint_sha
    result["size_bytes"] = checkpoint.path.stat().st_size
    if checkpoint.registered_sha256 and checkpoint_sha != checkpoint.registered_sha256:
        problems.append("registered_sha256_mismatch")
    try:
        import torch

        payload = torch.load(checkpoint.path, map_location="cpu", weights_only=True)
    except Exception as error:  # pragma: no cover - error text is diagnostic
        problems.append(f"safe_load_failed:{type(error).__name__}:{error}")
        return result
    if not isinstance(payload, Mapping):
        problems.append("payload_not_mapping")
        return result
    exact = {
        "kind": "detector",
        "format_version": 2,
        "epoch": 19,
        "checkpoint_selection": "fixed_last",
        "selection_rule": "fixed_last",
        "test_labels_used_for_selection": False,
        "diagnostic_test_eval": False,
        "diagnostic_only": False,
        "formal_paper_checkpoint": True,
        "warm_flag": True,
        "inference_head": "multi_scale_fused",
        "source_names": list(checkpoint.sources),
    }
    for field, expected in exact.items():
        if type(payload.get(field)) is not type(expected) or payload.get(field) != expected:
            problems.append(f"metadata_mismatch:{field}")
    source_records = payload.get("source_split_records")
    if not isinstance(source_records, list) or len(source_records) != len(checkpoint.sources):
        problems.append("invalid_source_split_records")
    else:
        by_name = {
            str(record.get("name")): record
            for record in source_records
            if isinstance(record, Mapping)
        }
        for source in checkpoint.sources:
            record = by_name.get(source)
            if record is None:
                problems.append(f"missing_source_split_record:{source}")
                continue
            if Path(str(record.get("train_split_file", ""))).resolve() != SPLITS[source]:
                problems.append(f"train_split_path_mismatch:{source}")
            if record.get("train_split_file_sha256") != _sha256(SPLITS[source]):
                problems.append(f"train_split_sha256_mismatch:{source}")
            if record.get("train_test_id_overlap") is not False:
                problems.append(f"train_test_overlap_not_false:{source}")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        problems.append("checkpoint_config_missing")
    else:
        data = config.get("data")
        training = config.get("training")
        if not isinstance(data, Mapping) or data.get("val_split") is not None:
            problems.append("checkpoint_val_split_not_null")
        if not isinstance(data, Mapping) or data.get("diagnostic_test_eval") is not False:
            problems.append("checkpoint_diagnostic_test_eval_not_false")
        if not isinstance(training, Mapping) or training.get("checkpoint_selection") != "fixed_last":
            problems.append("checkpoint_selection_config_mismatch")
        loss = config.get("loss")
        if not isinstance(loss, Mapping):
            problems.append("checkpoint_loss_missing")
        else:
            expected_loss = spec.loss_contract
            resolved_mode = loss.get("tail_mode", "probability_tail_miss_v1")
            for field in ("lambda_tail", "lambda_miss", "lambda_margin", "margin"):
                if float(loss.get(field, float("nan"))) != float(expected_loss[field]):
                    problems.append(f"loss_contract_mismatch:{field}")
            if resolved_mode != expected_loss["tail_mode"]:
                problems.append("loss_contract_mismatch:tail_mode")
            resolved_connectivity = loss.get("target_connectivity", 4)
            if int(resolved_connectivity) != int(expected_loss["target_connectivity"]):
                problems.append("loss_contract_mismatch:target_connectivity")
    result["valid"] = not problems
    return result


def _candidate_order_ready() -> tuple[bool, str | None]:
    path = (
        ROOT
        / "outputs/v4_tailmiss_source_only/"
        "gate_c_anchor_warp_vs_direct_seed42.json"
    )
    if not path.is_file():
        return False, f"tailmiss_gate_missing:{path}"
    comparison_root = ROOT / "outputs/v4_tailmiss_source_only/gate_c"
    gate_step = Step(
        "tailmiss_preregistered_gate_order_check",
        "aggregate_gate_c",
        (),
        (path,),
        {
            "fold_comparisons": [
                str(comparison_root / "val_irstd/comparison_seed42.json"),
                str(comparison_root / "val_nudt/comparison_seed42.json"),
            ]
        },
    )
    valid, reason = _json_artifact_complete(gate_step)
    if not valid:
        return False, f"tailmiss_gate_invalid:{reason}"
    return True, None


def audit_prerequisites(
    spec: BranchSpec, prereg: Mapping[str, Any]
) -> dict[str, Any]:
    problems = _config_hash_requirements(spec, prereg)
    official = prereg.get("official_train_splits", {})
    for source, path in SPLITS.items():
        if not path.is_file():
            problems.append(f"official_split_missing:{path}")
            continue
        record = official.get(source) if isinstance(official, Mapping) else None
        if not isinstance(record, Mapping):
            problems.append(f"official_split_not_registered:{source}")
            continue
        if Path(str(record.get("path", ""))).resolve() != path:
            problems.append(f"official_split_path_mismatch:{source}")
        if record.get("sha256") != _sha256(path):
            problems.append(f"official_split_sha256_mismatch:{source}")
    checkpoint_audits = [_audit_checkpoint(spec, value) for value in spec.checkpoints]
    for audit in checkpoint_audits:
        for problem in audit["problems"]:
            problems.append(f"checkpoint:{audit['role']}:{problem}")
    branch_order_ready = True
    branch_order_reason = None
    if spec.cli_name == "candidate_a":
        branch_order_ready, branch_order_reason = _candidate_order_ready()
        if not branch_order_ready:
            problems.append(str(branch_order_reason))
    return {
        "ready": not problems,
        "problems": problems,
        "branch_order_ready": branch_order_ready,
        "branch_order_reason": branch_order_reason,
        "checkpoints": checkpoint_audits,
    }


def _score_complete(step: Step) -> tuple[bool, str | None]:
    output = step.outputs[0]
    if not output.exists():
        return False, "absent"
    try:
        from evaluation.artifact_integrity import verify_score_map_directory

        manifest, _, integrity = verify_score_map_directory(
            output, require_integrity=True, require_masks=True
        )
        if manifest is None or integrity.get("verified") is not True:
            raise ValueError("score integrity did not verify")
        checkpoint = Path(str(step.details["checkpoint"]))
        expected = {
            "target_dataset": step.details["target_source_domain"],
            "source_datasets": step.details["checkpoint_sources"],
            "labels_loaded": True,
            "split_role": "train",
            "requested_split": "train",
            "split_authority_verified": True,
            "spatial_mode": "native",
            "checkpoint_epoch": 19,
            "checkpoint_selection_rule": "fixed_last",
            "checkpoint_diagnostic_only": False,
            "model_backend": "canonical",
            "score_representation": (
                "raw_logit_float32+sigmoid_probability_float32"
            ),
            "logit_dtype": "float32",
            "inference_autocast_enabled": False,
            "weight_sha256": _sha256(checkpoint),
            "split_file_sha256": _sha256(SPLITS[str(step.details["target_source_domain"])]),
        }
        for field, required in expected.items():
            if manifest.get(field) != required:
                raise ValueError(f"score manifest {field} mismatch")
        return True, None
    except Exception as error:
        return False, f"invalid:{type(error).__name__}:{error}"


def _resolved_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path string")
    return Path(value).expanduser().resolve()


def _require_bound_file(
    payload: Mapping[str, Any],
    *,
    path_field: str,
    sha_field: str,
    expected_path: str | Path,
) -> None:
    expected = Path(expected_path).expanduser().resolve(strict=True)
    observed = _resolved_path(payload.get(path_field), field=path_field)
    if observed != expected:
        raise ValueError(f"{path_field} does not bind the planned input")
    if payload.get(sha_field) != _sha256(expected):
        raise ValueError(f"{sha_field} does not bind current input bytes")


def _grid_complete(step: Step, spec: BranchSpec) -> tuple[bool, str | None]:
    if not any(path.exists() for path in step.outputs):
        return False, "absent"
    if not all(path.is_file() for path in step.outputs):
        return False, "invalid:partial_grid_artifact"
    try:
        from risk_curve.representation import load_logit_grid_artifact

        artifact = load_logit_grid_artifact(step.outputs[1])
        manifest = artifact.manifest
        expected_hashes = sorted(_sha256(value.path) for value in spec.checkpoints)
        if int(manifest.get("grid_points", -1)) != 2048:
            raise ValueError("grid_points is not 2048")
        if manifest.get("outer_target") != OUTER_DOMAIN_NAME_ONLY:
            raise ValueError("outer-target name declaration mismatch")
        if manifest.get("outer_target_labels_used") is not False:
            raise ValueError("outer-target labels were not excluded")
        if sorted(manifest.get("detector_checkpoint_sha256s", [])) != expected_hashes:
            raise ValueError("grid checkpoint set mismatch")
        references = manifest.get("input_score_artifacts")
        if not isinstance(references, list) or len(references) != 4:
            raise ValueError("grid input score artifact set is incomplete")
        observed_dirs = {
            _resolved_path(value.get("score_dir"), field="grid score_dir")
            for value in references
            if isinstance(value, Mapping)
        }
        expected_dirs = {
            Path(value).expanduser().resolve()
            for value in step.details["source_score_dirs"]
        }
        if observed_dirs != expected_dirs or len(observed_dirs) != 4:
            raise ValueError("grid input directories do not match the frozen plan")
        for reference in references:
            assert isinstance(reference, Mapping)
            score_dir = _resolved_path(reference.get("score_dir"), field="score_dir")
            score_manifest = _resolved_path(
                reference.get("score_manifest"), field="score_manifest"
            )
            if score_manifest != score_dir / "manifest.json":
                raise ValueError("grid score manifest path is not score_dir/manifest.json")
            if reference.get("score_manifest_sha256") != _sha256(score_manifest):
                raise ValueError("grid score manifest live-byte binding mismatch")
        return True, None
    except Exception as error:
        return False, f"invalid:{type(error).__name__}:{error}"


def _episodes_complete(step: Step) -> tuple[bool, str | None]:
    if not any(path.exists() for path in step.outputs):
        return False, "absent"
    if not all(path.is_file() for path in step.outputs):
        return False, "invalid:partial_episode_artifact"
    try:
        from risk_curve.curve_dataset import load_curve_archive

        manifests = [_load_json(step.outputs[2]), _load_json(step.outputs[5])]
        expected_validation_domains = ("IRSTD-1K", "NUDT-SIRST")
        expected_score_dirs = {
            Path(value).expanduser().resolve()
            for value in step.details["source_score_dirs"]
        }
        grid_manifest = Path(
            str(step.details["threshold_grid_manifest"])
        ).expanduser().resolve(strict=True)
        for manifest, expected_domain in zip(
            manifests, expected_validation_domains, strict=True
        ):
            exact = {
                "representation": "raw_logit_float32",
                "adaptation_window": 32,
                "evaluation_window": 1,
                "stride": 33,
                "connectivity": 2,
                "min_component_area": 1,
                "cross_episode_role_reuse_detected": False,
                "formal_causal_contract_verified": True,
                "formal_protocol_eligible": True,
                "status": "complete",
                "threshold_grid_outer_target_excluded": True,
            }
            for field, expected in exact.items():
                if manifest.get(field) != expected:
                    raise ValueError(f"episode manifest {field} mismatch")
            if manifest.get("validation_domain") != expected_domain:
                raise ValueError("paired episode validation domain/order mismatch")
            observed_score_dirs = {
                _resolved_path(value, field="episode score_map_dirs")
                for value in manifest.get("score_map_dirs", [])
            }
            if observed_score_dirs != expected_score_dirs:
                raise ValueError("episode inputs do not match planned pseudo-target scores")
            if _resolved_path(
                manifest.get("threshold_grid_manifest"),
                field="threshold_grid_manifest",
            ) != grid_manifest:
                raise ValueError("episode threshold-grid path mismatch")
            if manifest.get("threshold_grid_manifest_sha256") != _sha256(grid_manifest):
                raise ValueError("episode threshold-grid live-byte binding mismatch")
        for path in (step.outputs[0], step.outputs[1], step.outputs[3], step.outputs[4]):
            load_curve_archive(path)
        return True, None
    except Exception as error:
        return False, f"invalid:{type(error).__name__}:{error}"


def _torch_artifact_complete(step: Step) -> tuple[bool, str | None]:
    output = step.outputs[0]
    if not output.exists():
        return False, "absent"
    try:
        import torch

        payload = torch.load(output, map_location="cpu", weights_only=True)
        if step.kind == "anchor_train":
            from risk_curve.train_anchor_warp_predictor_v4 import (
                validate_anchor_warp_train_only_checkpoint,
            )

            validate_anchor_warp_train_only_checkpoint(payload)
            if not step.outputs[1].is_file():
                raise ValueError("AnchorWarp metrics sidecar is missing")
            _require_bound_file(
                payload,
                path_field="train_archive",
                sha_field="train_archive_sha256",
                expected_path=step.details["train_archive"],
            )
            if payload.get("seed") != 42:
                raise ValueError("AnchorWarp seed mismatch")
            metrics = _load_json(step.outputs[1])
            if _resolved_path(
                metrics.get("checkpoint_path"), field="metrics checkpoint_path"
            ) != output.resolve():
                raise ValueError("AnchorWarp metrics checkpoint path mismatch")
            if metrics.get("checkpoint_sha256") != _sha256(output):
                raise ValueError("AnchorWarp metrics checkpoint SHA mismatch")
        elif step.kind == "direct_train":
            from risk_curve.train_direct_calibrator_train_only_v4 import (
                validate_train_only_direct_checkpoint,
            )

            validate_train_only_direct_checkpoint(payload)
            _require_bound_file(
                payload,
                path_field="train_archive",
                sha_field="train_archive_sha256",
                expected_path=step.details["train_archive"],
            )
            _require_bound_file(
                payload,
                path_field="validation_archive",
                sha_field="validation_archive_sha256",
                expected_path=step.details["validation_archive"],
            )
            if payload.get("seed") != 42:
                raise ValueError("RC-Direct seed mismatch")
        elif step.kind == "anchor_bind":
            from risk_curve.bind_anchor_warp_validation_v4 import (
                validate_anchor_warp_bound_package,
            )

            validate_anchor_warp_bound_package(payload)
            parent = Path(
                str(step.details["frozen_checkpoint"])
            ).expanduser().resolve(strict=True)
            if _resolved_path(
                payload.get("parent_frozen_checkpoint_path"),
                field="parent_frozen_checkpoint_path",
            ) != parent:
                raise ValueError("bound package parent path mismatch")
            if payload.get("parent_frozen_checkpoint_sha256") != _sha256(parent):
                raise ValueError("bound package parent SHA mismatch")
            binding = payload.get("validation_binding")
            if not isinstance(binding, Mapping):
                raise ValueError("bound package validation_binding is missing")
            _require_bound_file(
                binding,
                path_field="validation_archive",
                sha_field="validation_archive_sha256",
                expected_path=step.details["validation_archive"],
            )
        else:  # pragma: no cover
            raise RuntimeError(step.kind)
        return True, None
    except Exception as error:
        return False, f"invalid:{type(error).__name__}:{error}"


def _json_artifact_complete(step: Step) -> tuple[bool, str | None]:
    output = step.outputs[0]
    if not output.exists():
        return False, "absent"
    try:
        payload = _load_json(output)
        if step.kind == "source_provenance":
            from risk_curve.source_provenance_v4 import (
                load_source_provenance_run_evidence,
            )

            evidence, _ = load_source_provenance_run_evidence(output)
            if evidence["outer_target_access_declaration"]["outer_target_labels_read"] is not False:
                raise ValueError("provenance reports outer labels read")
            verifier = evidence.get("verifier_module")
            execution = evidence.get("execution")
            if not isinstance(verifier, Mapping) or not isinstance(execution, Mapping):
                raise ValueError("provenance verifier/execution binding is missing")
            verifier_path = ROOT / "risk_curve/source_provenance_v4.py"
            if _resolved_path(
                verifier.get("path"), field="verifier_module.path"
            ) != verifier_path.resolve():
                raise ValueError("provenance verifier path mismatch")
            if verifier.get("sha256") != _sha256(verifier_path):
                raise ValueError("provenance verifier live-byte SHA mismatch")
            parameters = execution.get("parameters")
            if not isinstance(parameters, Mapping):
                raise ValueError("provenance parameters are missing")
            expected_grid = Path(
                str(step.details["threshold_grid_manifest"])
            ).expanduser().resolve(strict=True)
            if _resolved_path(
                parameters.get("threshold_grid_manifest"),
                field="parameters.threshold_grid_manifest",
            ) != expected_grid:
                raise ValueError("provenance grid parameter mismatch")
            for field in ("detector_checkpoints", "episode_archives"):
                values = parameters.get(field)
                if not isinstance(values, list):
                    raise ValueError(f"provenance {field} parameter is missing")
                observed = {
                    _resolved_path(value, field=f"parameters.{field}")
                    for value in values
                }
                expected = {
                    Path(value).expanduser().resolve(strict=True)
                    for value in step.details[field]
                }
                if observed != expected or len(values) != len(expected):
                    raise ValueError(f"provenance {field} plan binding mismatch")
            split_parameters = parameters.get("official_train_split_manifests")
            if not isinstance(split_parameters, Mapping):
                raise ValueError("provenance official split parameters are missing")
            if {
                _resolved_path(value, field="official split")
                for value in split_parameters.values()
            } != {Path(value).resolve() for value in step.details["accepted_source_paths"]}:
                raise ValueError("provenance official split plan binding mismatch")
            verification = evidence.get("verification")
            if not isinstance(verification, Mapping):
                raise ValueError("provenance verification payload is missing")
            threshold_grid = verification.get("threshold_grid")
            if (
                not isinstance(threshold_grid, Mapping)
                or threshold_grid.get("manifest_sha256") != _sha256(expected_grid)
            ):
                raise ValueError("provenance live grid SHA binding mismatch")
            checkpoint_rows = verification.get("detector_checkpoints")
            if not isinstance(checkpoint_rows, list) or {
                str(value.get("sha256"))
                for value in checkpoint_rows
                if isinstance(value, Mapping)
            } != {
                _sha256(Path(value)) for value in step.details["detector_checkpoints"]
            }:
                raise ValueError("provenance live checkpoint SHA set mismatch")
        elif step.kind == "comparison":
            from risk_curve.evaluate_anchor_warp_source_pseudo_target_v4 import (
                ANCHOR_WARP_SOURCE_COMPARISON_SCHEMA_VERSION,
            )

            if payload.get("schema_version") != ANCHOR_WARP_SOURCE_COMPARISON_SCHEMA_VERSION:
                raise ValueError("comparison schema mismatch")
            if payload.get("outer_target_labels_used") is not False:
                raise ValueError("comparison reports outer labels used")
            audit = payload.get("action_freeze")
            if (
                not isinstance(audit, Mapping)
                or audit.get("frozen_before_future_e_load") is not True
                or audit.get("future_e_reselection_performed") is not False
                or payload.get("future_e_arrays_loaded_before_action_digest") is not False
                or payload.get("labels_used_for_action_selection") is not False
            ):
                raise ValueError("comparison lacks two-phase action freeze")
            _require_bound_file(
                payload,
                path_field="episode_archive",
                sha_field="episode_archive_sha256",
                expected_path=step.details["episode_archive"],
            )
            _require_bound_file(
                payload,
                path_field="anchor_warp_package",
                sha_field="anchor_warp_package_sha256",
                expected_path=step.details["anchor_warp_package"],
            )
            _require_bound_file(
                payload,
                path_field="rc_direct_checkpoint",
                sha_field="rc_direct_checkpoint_sha256",
                expected_path=step.details["rc_direct_checkpoint"],
            )
            if payload.get("seed") != 42:
                raise ValueError("comparison seed mismatch")
        elif step.kind == "aggregate_gate_c":
            if payload.get("decision") not in {"GO", "HOLD"}:
                raise ValueError("Gate C has no final decision")
            criteria = payload.get("criteria")
            if not isinstance(criteria, Mapping) or len(criteria) != 8:
                raise ValueError("Gate C criteria are incomplete")
            if any(type(value) is not bool for value in criteria.values()):
                raise ValueError("Gate C criteria must be boolean")
            contract = payload.get("aggregation_contract")
            if not isinstance(contract, Mapping):
                raise ValueError("Gate C aggregation contract is missing")
            if float(contract.get("pd_non_degradation_floor", float("nan"))) != 0.0:
                raise ValueError("Gate C Pd floor was lowered")
            if payload.get("outer_target_labels_used") is not False:
                raise ValueError("Gate C reports outer labels used")
            folds = payload.get("folds")
            if not isinstance(folds, Mapping) or set(folds) != {
                "val_irstd",
                "val_nudt",
            }:
                raise ValueError("Gate C fold set mismatch")
            expected_comparisons = {
                "val_irstd": Path(step.details["fold_comparisons"][0])
                .expanduser()
                .resolve(strict=True),
                "val_nudt": Path(step.details["fold_comparisons"][1])
                .expanduser()
                .resolve(strict=True),
            }
            for fold_name, expected in expected_comparisons.items():
                fold = folds[fold_name]
                if not isinstance(fold, Mapping):
                    raise ValueError("Gate C fold payload is invalid")
                if _resolved_path(
                    fold.get("comparison_file"), field="comparison_file"
                ) != expected:
                    raise ValueError("Gate C comparison path mismatch")
                if fold.get("comparison_sha256") != _sha256(expected):
                    raise ValueError("Gate C comparison live-byte SHA mismatch")
            runtime_tree = payload.get("runtime_code_tree")
            if (
                not isinstance(runtime_tree, Mapping)
                or runtime_tree.get(
                    "verified_unchanged_immediately_before_atomic_publish"
                )
                is not True
            ):
                raise ValueError("Gate C runtime code tree was not sealed")
        return True, None
    except Exception as error:
        return False, f"invalid:{type(error).__name__}:{error}"


def step_complete(step: Step, spec: BranchSpec) -> tuple[bool, str | None]:
    if step.kind == "score_export":
        return _score_complete(step)
    if step.kind == "grid":
        return _grid_complete(step, spec)
    if step.kind == "episodes":
        return _episodes_complete(step)
    if step.kind in {"anchor_train", "direct_train", "anchor_bind"}:
        return _torch_artifact_complete(step)
    if step.kind in {"source_provenance", "comparison", "aggregate_gate_c"}:
        return _json_artifact_complete(step)
    raise RuntimeError(f"No validator for step kind {step.kind!r}")


def _immutable_code_paths(spec: BranchSpec) -> tuple[Path, ...]:
    roots = ("rc_irstd", "risk_curve", "evaluation", "data_ext", "model", "losses", "utils")
    paths: set[Path] = {Path(__file__).resolve(), PREREGISTRATION.resolve()}
    for relative in roots:
        paths.update(path.resolve() for path in (ROOT / relative).rglob("*.py"))
    paths.update(path.resolve() for path in SPLITS.values())
    paths.update(
        value.config_path.resolve()
        for value in spec.checkpoints
        if value.config_path is not None
    )
    # Detector bytes are upstream inputs to every score/grid/episode artifact.
    # Freezing only Python/config bytes would permit a checkpoint replacement
    # between resumed exports; include all three exact checkpoint files in the
    # run snapshot so that such drift fails closed before another step starts.
    paths.update(value.path.resolve() for value in spec.checkpoints)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("A phase-2 code/config/split input is missing")
    return tuple(sorted(paths))


def _snapshot(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    return {
        str(path): {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in paths
    }


def _verify_snapshot(snapshot: Mapping[str, Mapping[str, Any]]) -> None:
    for value, expected in snapshot.items():
        path = Path(value)
        if not path.is_file():
            raise RuntimeError(f"Frozen phase-2 input disappeared: {path}")
        if path.stat().st_size != expected["size_bytes"] or _sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Frozen phase-2 input drifted: {path}")


def _gpu_compute_pids() -> tuple[int, ...]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    pids: list[int] = []
    for line in result.stdout.splitlines():
        value = line.strip()
        if not value or value.casefold().startswith("no running"):
            continue
        pids.append(int(value))
    return tuple(sorted(set(pids)))


def _run_execute_unlocked(
    spec: BranchSpec,
    prereg: Mapping[str, Any],
    steps: Sequence[Step],
    *,
    max_steps: int | None,
) -> int:
    preflight = audit_prerequisites(spec, prereg)
    if not preflight["ready"]:
        raise RuntimeError(
            "Phase-2 prerequisites are not ready: " + "; ".join(preflight["problems"])
        )
    snapshot = _snapshot(_immutable_code_paths(spec))
    status_path = spec.namespace / "phase2/run_status.json"
    status: dict[str, Any] = {
        "schema_version": "rc-v4-detector-tail-phase2-run-v1",
        "branch": spec.branch_id,
        "branch_position": spec.prereg_position,
        "scope": "source_only_development_gate_c",
        "outer_target_labels_used": False,
        "preregistration": str(PREREGISTRATION),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "orchestrator": str(Path(__file__).resolve()),
        "orchestrator_sha256": _sha256(Path(__file__).resolve()),
        "frozen_input_tree_sha256": _canonical_sha256(snapshot),
        "started_utc": _utc_now(),
        "state": "running",
        "steps": [],
    }
    _atomic_json(status_path, status)
    executed = 0
    try:
        for step in steps:
            _verify_snapshot(snapshot)
            complete, reason = step_complete(step, spec)
            if complete:
                status["steps"].append(step.to_json(state="revalidated_and_skipped"))
                status["updated_utc"] = _utc_now()
                _atomic_json(status_path, status)
                continue
            if reason != "absent":
                raise RuntimeError(
                    f"Step {step.step_id} has a partial/invalid artifact and will not "
                    f"be overwritten automatically: {reason}"
                )
            if max_steps is not None and executed >= max_steps:
                status["state"] = "paused_at_max_steps"
                status["updated_utc"] = _utc_now()
                _atomic_json(status_path, status)
                return 0
            if step.kind == "score_export":
                occupied = _gpu_compute_pids()
                if occupied:
                    raise RuntimeError(
                        "Refusing phase-2 score export while GPU compute "
                        "processes exist: " + ", ".join(map(str, occupied))
                    )
            status["active_step"] = step.step_id
            status["updated_utc"] = _utc_now()
            _atomic_json(status_path, status)
            result = subprocess.run(step.command, cwd=ROOT, check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Step {step.step_id} exited with code {result.returncode}"
                )
            complete, reason = step_complete(step, spec)
            if not complete:
                raise RuntimeError(
                    f"Step {step.step_id} did not publish a valid artifact: {reason}"
                )
            status["steps"].append(step.to_json(state="completed"))
            status.pop("active_step", None)
            status["updated_utc"] = _utc_now()
            _atomic_json(status_path, status)
            executed += 1
        _verify_snapshot(snapshot)
        status["state"] = "complete"
        status["completed_utc"] = _utc_now()
        status.pop("active_step", None)
        _atomic_json(status_path, status)
        print(status_path)
        return 0
    except Exception as error:
        status["state"] = "failed_closed"
        status["failure"] = f"{type(error).__name__}: {error}"
        status["updated_utc"] = _utc_now()
        _atomic_json(status_path, status)
        raise


def _run_execute(
    spec: BranchSpec,
    prereg: Mapping[str, Any],
    steps: Sequence[Step],
    *,
    max_steps: int | None,
) -> int:
    """Serialize one branch run without relying on a PID/stale-lock protocol."""

    phase2_root = spec.namespace / "phase2"
    phase2_root.mkdir(parents=True, exist_ok=True)
    lock_path = phase2_root / "run.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another phase-2 orchestrator holds the branch lock: {lock_path}"
            ) from error
        lock_handle.seek(0)
        lock_handle.truncate()
        lock_handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "branch": spec.branch_id,
                    "acquired_utc": _utc_now(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        lock_handle.flush()
        os.fsync(lock_handle.fileno())
        return _run_execute_unlocked(
            spec,
            prereg,
            steps,
            max_steps=max_steps,
        )
    finally:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=("tailmiss", "candidate_a"), required=True)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--count-all-workers", type=int, default=16)
    parser.add_argument("--export-workers", type=int, default=4)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the plan; omission is a read-only dry run",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Execute at most this many currently-incomplete steps, then pause cleanly",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    # Keep a virtual-environment interpreter symlink intact.  Resolving the
    # final symlink can silently replace ``venv/bin/python`` with the host
    # interpreter and lose the environment's torch/CUDA dependencies.
    python_bin = Path(os.path.abspath(Path(args.python).expanduser()))
    if not python_bin.is_file():
        raise FileNotFoundError(python_bin)
    spec, prereg = load_branch_spec(args.branch)
    steps = build_plan(
        spec,
        python_bin=python_bin,
        count_all_workers=args.count_all_workers,
        export_workers=args.export_workers,
    )
    if args.execute:
        return _run_execute(
            spec,
            prereg,
            steps,
            max_steps=args.max_steps,
        )
    preflight = audit_prerequisites(spec, prereg)
    rendered_steps: list[dict[str, Any]] = []
    for step in steps:
        complete, reason = step_complete(step, spec)
        rendered_steps.append(
            step.to_json(
                state="complete" if complete else "pending",
                reason=reason,
            )
        )
    payload = {
        "schema_version": "rc-v4-detector-tail-phase2-dry-run-v1",
        "dry_run": True,
        "gpu_processes_queried": False,
        "gpu_work_started": False,
        "branch": spec.branch_id,
        "branch_position": spec.prereg_position,
        "namespace": str(spec.namespace),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "source_domains": list(SOURCE_DOMAINS),
        "excluded_outer_target_name_only": OUTER_DOMAIN_NAME_ONLY,
        "outer_target_path_argument_present": False,
        "outer_target_labels_used": False,
        "preflight": preflight,
        "plan_ready_for_execute": bool(preflight["ready"]),
        "steps": rendered_steps,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BranchSpec",
    "CheckpointSpec",
    "Step",
    "audit_prerequisites",
    "build_plan",
    "load_branch_spec",
    "main",
    "step_complete",
]
