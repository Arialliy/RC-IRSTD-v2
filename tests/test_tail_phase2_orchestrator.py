from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_tail_phase2 as phase2


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_PREREGISTRATION = (
    ROOT
    / "outputs/v4_source_only/preregistration/detector_tail_branches_seed42.json"
)
# This orchestrator replays the removed Tail Phase-2 study.  It is collected
# only when that study's preregistration and products are present.
pytestmark = pytest.mark.skipif(
    not HISTORICAL_PREREGISTRATION.is_file(),
    reason="historical Tail Phase-2 experiment products were intentionally removed",
)


def _step(steps: tuple[phase2.Step, ...], step_id: str) -> phase2.Step:
    matches = [value for value in steps if value.step_id == step_id]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("branch", ["tailmiss", "candidate_a"])
def test_phase2_plan_is_exact_source_only_preregistered_dag(branch: str) -> None:
    spec, prereg = phase2.load_branch_spec(branch)
    steps = phase2.build_plan(spec, python_bin=Path("/venv/python"))

    assert prereg["excluded_outer_target"] == "NUAA-SIRST"
    assert len(steps) == 18
    assert [value.kind for value in steps[:9]] == [
        "score_export",
        "score_export",
        "score_export",
        "score_export",
        "score_export",
        "score_export",
        "grid",
        "episodes",
        "source_provenance",
    ]
    assert [value.kind for value in steps[-9:]] == [
        "anchor_train",
        "direct_train",
        "anchor_bind",
        "comparison",
        "anchor_train",
        "direct_train",
        "anchor_bind",
        "comparison",
        "aggregate_gate_c",
    ]
    assert all(value.outputs for value in steps)
    assert all(path.is_relative_to(spec.namespace) for value in steps for path in value.outputs)

    serialized_commands = json.dumps(
        [value.command for value in steps], ensure_ascii=False
    )
    assert "datasets/NUAA" not in serialized_commands
    assert "datasets/nuaa" not in serialized_commands
    assert "/NUAA" not in serialized_commands
    assert "NUAA-SIRST" in _step(steps, "build_raw_logit_grid_2048").command
    assert all("--force" not in value.command for value in steps)
    assert all("--overwrite" not in value.command for value in steps)

    exports = steps[:6]
    assert all("--split" in value.command for value in exports)
    assert all("train" in value.command for value in exports)
    assert all("--export-raw-logits" in value.command for value in exports)
    assert {value.details["target_source_domain"] for value in exports} == {
        "NUDT-SIRST",
        "IRSTD-1K",
    }
    grid = _step(steps, "build_raw_logit_grid_2048")
    assert grid.command.count("--source-score-dir") == 4
    assert grid.command[grid.command.index("--max-grid-points") + 1] == "2048"
    episodes = _step(steps, "build_paired_source_episodes")
    assert episodes.command[episodes.command.index("--adaptation-window") + 1] == "32"
    assert episodes.command[episodes.command.index("--evaluation-window") + 1] == "1"
    assert episodes.command[episodes.command.index("--stride") + 1] == "33"
    aggregate = _step(steps, "aggregate_strict_gate_c")
    assert aggregate.details["pd_non_degradation_floor"] == 0.0
    assert aggregate.details["criteria"] == [f"c{index}" for index in range(8)]


def test_dry_run_never_queries_or_starts_gpu(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden_gpu_query() -> tuple[int, ...]:
        raise AssertionError("dry-run queried the GPU")

    def forbidden_subprocess(*args: object, **kwargs: object) -> object:
        raise AssertionError("dry-run started a subprocess")

    monkeypatch.setattr(phase2, "_gpu_compute_pids", forbidden_gpu_query)
    monkeypatch.setattr(phase2.subprocess, "run", forbidden_subprocess)

    assert phase2.main(["--branch", "tailmiss"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["gpu_processes_queried"] is False
    assert payload["gpu_work_started"] is False
    assert payload["outer_target_path_argument_present"] is False
    assert payload["outer_target_labels_used"] is False


def test_main_preserves_virtual_environment_python_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_python = tmp_path / "host-python"
    real_python.write_bytes(b"binary")
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(real_python)
    observed: list[Path] = []

    original_build_plan = phase2.build_plan

    def capture_python(
        spec: phase2.BranchSpec,
        *,
        python_bin: Path,
        count_all_workers: int,
        export_workers: int,
    ) -> tuple[phase2.Step, ...]:
        observed.append(python_bin)
        return original_build_plan(
            spec,
            python_bin=python_bin,
            count_all_workers=count_all_workers,
            export_workers=export_workers,
        )

    monkeypatch.setattr(phase2, "build_plan", capture_python)
    assert phase2.main(
        ["--branch", "tailmiss", "--python", str(venv_python)]
    ) == 0
    capsys.readouterr()
    assert observed == [venv_python.absolute()]
    assert observed[0] != real_python


def test_script_path_execution_exposes_repository_modules(tmp_path: Path) -> None:
    script = ROOT / "scripts/run_tail_phase2.py"
    probe = (
        "import runpy; "
        f"ns=runpy.run_path({str(script)!r}, run_name='phase2_probe'); "
        "import evaluation; "
        "print(ns['ROOT'])"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(ROOT)


def test_existing_formal_artifact_validators_require_live_input_hashes() -> None:
    # These previously sealed artifacts exercise the same validators without
    # training or exporting anything in this test.
    fold = ROOT / "outputs/v4_anchor_warp_source_only/val_irstd"
    episodes = ROOT / "outputs/v4_source_only/episodes/val_irstd"
    anchor = phase2.Step(
        "anchor",
        "anchor_train",
        (),
        (
            fold / "anchor_warp_train_only_seed42.pt",
            fold / "anchor_warp_train_only_seed42.pt.metrics.json",
        ),
        {"train_archive": str(episodes / "train.npz")},
    )
    direct = phase2.Step(
        "direct",
        "direct_train",
        (),
        (fold / "rc_direct_train_only_seed42.pt",),
        {
            "train_archive": str(episodes / "train.npz"),
            "validation_archive": str(episodes / "val.npz"),
        },
    )
    bound = phase2.Step(
        "bound",
        "anchor_bind",
        (),
        (fold / "anchor_warp_bound_seed42.pt",),
        {
            "frozen_checkpoint": str(fold / "anchor_warp_train_only_seed42.pt"),
            "validation_archive": str(episodes / "val.npz"),
        },
    )
    comparison = phase2.Step(
        "comparison",
        "comparison",
        (),
        (fold / "comparison_seed42.json",),
        {
            "episode_archive": str(episodes / "val.npz"),
            "anchor_warp_package": str(fold / "anchor_warp_bound_seed42.pt"),
            "rc_direct_checkpoint": str(fold / "rc_direct_train_only_seed42.pt"),
        },
    )

    assert phase2._torch_artifact_complete(anchor) == (True, None)
    assert phase2._torch_artifact_complete(direct) == (True, None)
    assert phase2._torch_artifact_complete(bound) == (True, None)
    assert phase2._json_artifact_complete(comparison) == (True, None)

    stale = phase2.Step(
        **{
            **comparison.__dict__,
            "details": {
                **comparison.details,
                "episode_archive": str(episodes / "train.npz"),
            },
        }
    )
    complete, reason = phase2._json_artifact_complete(stale)
    assert complete is False
    assert reason is not None and "planned input" in reason
