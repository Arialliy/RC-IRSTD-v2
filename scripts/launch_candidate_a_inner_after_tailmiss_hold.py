"""Launch Candidate-A inner folds only after a sealed Tail/Miss HOLD.

This contingency launcher is intentionally fail-closed.  It never opens an
outer-domain dataset path, never signals an external process, and launches
exactly the two preregistered source-only Candidate-A inner detectors.  GPU0/1
must be compute-idle for consecutive observations, all inputs are byte-frozen,
the Tail/Miss strict Gate C must validate as ``HOLD``, and the already-trained
Candidate-A full-source checkpoint must be a valid fixed-last epoch-19 artifact.

The default invocation is an audit-only dry run.  ``--execute`` is required to
acquire the launcher lock, wait for GPUs, or start detector subprocesses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_tail_phase2 as phase2
from scripts import wait_and_launch_tail_phase1 as safety


DEFAULT_PYTHON = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
PREREGISTRATION = phase2.PREREGISTRATION
PREREGISTRATION_SHA256 = phase2.PREREGISTRATION_SHA256
TARGET_GPUS = (0, 1)
VISIBLE_GPUS = "0,1"

GATE_PATH = (
    ROOT
    / "outputs/v4_tailmiss_source_only/"
    "gate_c_anchor_warp_vs_direct_seed42.json"
)
GATE_COMPARISON_DIR = ROOT / "outputs/v4_tailmiss_source_only/gate_c"
FULL_CHECKPOINT = ROOT / "outputs/stage4_full_sources_tailrank_margin_a_20ep/last.pt"

STATUS_PATH = PREREGISTRATION.with_name("candidate_a_inner_contingency_status.json")
LOCK_PATH = PREREGISTRATION.with_name("candidate_a_inner_contingency.lock")
LOCK_GUARD_PATH = PREREGISTRATION.with_name("candidate_a_inner_contingency.lock.guard")
LOG_DIR = PREREGISTRATION.with_name("candidate_a_inner_contingency_logs")

RUNS = (
    {
        "run_id": "candidate_a_irstd_only_inner",
        "role": "inner_from_irstd",
        "config": ROOT
        / "configs/stage4_inner_nudt_from_irstd_tailrank_margin_a_20ep.yaml",
        "output_dir": ROOT
        / "outputs/stage4_inner_nudt_from_irstd_tailrank_margin_a_20ep",
    },
    {
        "run_id": "candidate_a_nudt_only_inner",
        "role": "inner_from_nudt",
        "config": ROOT
        / "configs/stage4_inner_irstd_from_nudt_tailrank_margin_a_20ep.yaml",
        "output_dir": ROOT
        / "outputs/stage4_inner_irstd_from_nudt_tailrank_margin_a_20ep",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates
    )
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


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
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _tailmiss_hold_evidence() -> dict[str, Any]:
    comparisons = (
        GATE_COMPARISON_DIR / "val_irstd/comparison_seed42.json",
        GATE_COMPARISON_DIR / "val_nudt/comparison_seed42.json",
    )
    step = phase2.Step(
        "tailmiss_contingency_gate",
        "aggregate_gate_c",
        (),
        (GATE_PATH,),
        {"fold_comparisons": [str(path) for path in comparisons]},
    )
    valid, reason = phase2._json_artifact_complete(step)
    if not valid:
        raise RuntimeError(f"Tail/Miss strict Gate C is not valid: {reason}")
    payload = _load_json(GATE_PATH)
    if payload.get("decision") != "HOLD":
        raise RuntimeError("Candidate-A contingency requires Tail/Miss decision=HOLD")
    criteria = payload.get("criteria")
    if (
        not isinstance(criteria, Mapping)
        or len(criteria) != 8
        or any(type(value) is not bool for value in criteria.values())
        or all(criteria.values())
    ):
        raise RuntimeError("Tail/Miss HOLD criteria are incomplete or inconsistent")
    contract = payload.get("aggregation_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("criteria_not_lowered") is not True
        or float(contract.get("pd_non_degradation_floor", float("nan"))) != 0.0
        or payload.get("outer_target_labels_used") is not False
    ):
        raise RuntimeError("Tail/Miss Gate C strict/no-outer contract is invalid")
    return {
        "path": str(GATE_PATH),
        "sha256": _sha256(GATE_PATH),
        "decision": "HOLD",
        "comparison_paths": [str(path) for path in comparisons],
        "comparison_sha256": {str(path): _sha256(path) for path in comparisons},
    }


def _candidate_full_checkpoint_evidence(
    spec: phase2.BranchSpec,
) -> dict[str, Any]:
    full = phase2._checkpoint_by_role(spec, "full_sources")
    if full.path.resolve() != FULL_CHECKPOINT.resolve():
        raise RuntimeError("Candidate-A full checkpoint path drifted")
    audit = phase2._audit_checkpoint(spec, full)
    if audit.get("valid") is not True:
        raise RuntimeError(
            "Candidate-A full checkpoint is not a valid fixed-last epoch-19 "
            f"artifact: {audit.get('problems')}"
        )
    return audit


def _assert_source_only_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        if "nuaa" in str(path).casefold():
            raise RuntimeError(f"Outer-domain path is forbidden: {path}")


def _frozen_paths(
    gate_evidence: Mapping[str, Any], spec: phase2.BranchSpec
) -> tuple[Path, ...]:
    roots = ("rc_irstd", "data_ext")
    paths: set[Path] = {
        Path(__file__).resolve(),
        Path(safety.__file__).resolve(),
        Path(phase2.__file__).resolve(),
        PREREGISTRATION.resolve(),
        FULL_CHECKPOINT.resolve(),
        GATE_PATH.resolve(),
        *(path.resolve() for path in phase2.SPLITS.values()),
        *(Path(run["config"]).resolve() for run in RUNS),
        *(Path(value).resolve() for value in gate_evidence["comparison_paths"]),
    }
    for relative in roots:
        paths.update(path.resolve() for path in (ROOT / relative).rglob("*.py"))
    for relative in (
        "evaluation/artifact_integrity.py",
        "losses/local_peak_cvar.py",
        "model/MSHNet.py",
        "utils/data.py",
    ):
        paths.add((ROOT / relative).resolve())
    paths.update(
        checkpoint.config_path.resolve()
        for checkpoint in spec.checkpoints
        if checkpoint.config_path is not None
    )
    result = tuple(sorted(paths))
    _assert_source_only_paths(result)
    if any(not path.is_file() for path in result):
        raise FileNotFoundError("A frozen Candidate-A contingency input is missing")
    return result


def _snapshot(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    return {
        str(path): {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in paths
    }


def _verify_snapshot(snapshot: Mapping[str, Mapping[str, Any]]) -> None:
    for raw_path, expected in snapshot.items():
        path = Path(raw_path)
        if not path.is_file():
            raise RuntimeError(f"Frozen contingency input disappeared: {path}")
        if (
            path.stat().st_size != expected["size_bytes"]
            or _sha256(path) != expected["sha256"]
        ):
            raise RuntimeError(f"Frozen contingency input drifted: {path}")


def _assert_outputs_absent() -> None:
    for run in RUNS:
        if Path(run["output_dir"]).exists():
            raise FileExistsError(
                f"Refusing to overwrite Candidate-A inner output: {run['output_dir']}"
            )
    if LOG_DIR.exists():
        raise FileExistsError(f"Refusing to reuse contingency log directory: {LOG_DIR}")


def _gpu_compute_processes() -> tuple[dict[str, Any], ...]:
    """Observe GPU0/1 without signalling or inspecting any dataset path."""

    return safety._gpu_compute_processes(TARGET_GPUS)


def audit_readiness() -> dict[str, Any]:
    if ROOT != Path("/home/ly/RC-IRSTD-v2"):
        raise RuntimeError(f"Unexpected project root: {ROOT}")
    if _sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("Detector-tail preregistration bytes have drifted")
    spec, prereg = phase2.load_branch_spec("candidate_a")
    config_problems = phase2._config_hash_requirements(spec, prereg)
    if config_problems:
        raise RuntimeError(f"Candidate-A registered inputs drifted: {config_problems}")
    gate = _tailmiss_hold_evidence()
    full = _candidate_full_checkpoint_evidence(spec)
    _assert_outputs_absent()
    paths = _frozen_paths(gate, spec)
    snapshot = _snapshot(paths)
    return {
        "ready": True,
        "tailmiss_gate": gate,
        "candidate_full_checkpoint": full,
        "frozen_snapshot": snapshot,
        "frozen_file_count": len(snapshot),
        "outer_target_label_paths_opened": False,
        "run_ids": [str(run["run_id"]) for run in RUNS],
    }


def _execute(args: argparse.Namespace, readiness: Mapping[str, Any]) -> int:
    python_bin = Path(args.python).expanduser().absolute()
    if not python_bin.is_file():
        raise FileNotFoundError(python_bin)
    frozen_snapshot = readiness["frozen_snapshot"]
    lock_record, stale_owner = safety._acquire_launcher_lock(
        LOCK_PATH, LOCK_GUARD_PATH
    )
    status: dict[str, Any] = {
        "schema_version": "rc-v4-candidate-a-inner-contingency-v1",
        "state": "waiting_for_gpu_compute_quiescence",
        "launcher_pid": os.getpid(),
        "updated_utc": safety._utc_now(),
        "lock_record": lock_record,
        "stale_lock_owner": stale_owner,
        "target_physical_gpus": list(TARGET_GPUS),
        "required_clear_observations": args.clear_observations,
        "poll_seconds": args.poll_seconds,
        "tailmiss_gate": readiness["tailmiss_gate"],
        "candidate_full_checkpoint": readiness["candidate_full_checkpoint"],
        "frozen_snapshot": frozen_snapshot,
        "commands": {},
        "external_process_signals_sent": False,
        "outer_target_label_paths_opened": False,
    }
    children: dict[str, subprocess.Popen[bytes]] = {}
    handles: list[Any] = []
    try:
        _atomic_json(STATUS_PATH, status)

        def verify_ready() -> None:
            _verify_snapshot(frozen_snapshot)
            _assert_outputs_absent()

        def observe(
            processes: Sequence[Mapping[str, Any]], count: int, phase: str
        ) -> None:
            status.update(
                {
                    "state": "waiting_for_gpu_compute_quiescence",
                    "updated_utc": safety._utc_now(),
                    "observation_phase": phase,
                    "consecutive_clear_observations": count,
                    "observed_gpu_compute_processes": list(processes),
                    "observed_gpu_compute_pids": sorted(
                        {int(value["pid"]) for value in processes}
                    ),
                }
            )
            _atomic_json(STATUS_PATH, status)

        safety._wait_for_gpu_clear(
            required_clear_observations=args.clear_observations,
            poll_seconds=args.poll_seconds,
            observe=observe,
            verify_ready=verify_ready,
            query=_gpu_compute_processes,
        )
        verify_ready()
        immediate = tuple(_gpu_compute_processes())
        observe(
            immediate,
            args.clear_observations + 1 if not immediate else 0,
            "immediate_prelaunch_confirmation",
        )
        if immediate:
            raise RuntimeError("GPU activity appeared at the launch boundary")

        LOG_DIR.mkdir(parents=True, exist_ok=False)
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": VISIBLE_GPUS,
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTHONUNBUFFERED": "1",
            }
        )
        for run in RUNS:
            run_id = str(run["run_id"])
            command = [
                str(python_bin),
                "-m",
                "rc_irstd.cli.train_detector",
                "--config",
                str(run["config"]),
            ]
            log_path = LOG_DIR / f"{run_id}.stdout_stderr.log"
            handle = log_path.open("wb")
            handles.append(handle)
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children[run_id] = process
            status["commands"][run_id] = {
                "argv": command,
                "pid": process.pid,
                "config_sha256": _sha256(Path(run["config"])),
                "log_path": str(log_path),
            }
        status.update({"state": "running", "updated_utc": safety._utc_now()})
        _atomic_json(STATUS_PATH, status)

        while True:
            returncodes = {name: child.poll() for name, child in children.items()}
            failures = {
                name: code
                for name, code in returncodes.items()
                if code is not None and code != 0
            }
            status.update(
                {
                    "state": "running",
                    "updated_utc": safety._utc_now(),
                    "returncodes": returncodes,
                }
            )
            _atomic_json(STATUS_PATH, status)
            if failures:
                raise RuntimeError(f"Candidate-A inner subprocess failed: {failures}")
            if all(code == 0 for code in returncodes.values()):
                break
            time.sleep(args.poll_seconds)

        spec, _ = phase2.load_branch_spec("candidate_a")
        completed: dict[str, Any] = {}
        for run in RUNS:
            checkpoint = phase2._checkpoint_by_role(spec, str(run["role"]))
            audit = phase2._audit_checkpoint(spec, checkpoint)
            if audit.get("valid") is not True:
                raise RuntimeError(
                    f"Completed Candidate-A inner checkpoint invalid: {audit}"
                )
            completed[str(run["run_id"])] = audit
        _verify_snapshot(frozen_snapshot)
        status.update(
            {
                "state": "completed",
                "updated_utc": safety._utc_now(),
                "returncodes": {name: 0 for name in children},
                "completed_checkpoints": completed,
                "external_process_signals_sent": False,
                "outer_target_label_paths_opened": False,
            }
        )
        _atomic_json(STATUS_PATH, status)
        return 0
    except BaseException as error:
        if children:
            safety._terminate_owned(children)
        status.update(
            {
                "state": "failed",
                "updated_utc": safety._utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "owned_children_termination_attempted": bool(children),
                "external_process_signals_sent": False,
                "outer_target_label_paths_opened": False,
            }
        )
        _atomic_json(STATUS_PATH, status)
        raise
    finally:
        for handle in handles:
            handle.close()
        safety._release_launcher_lock(lock_record, LOCK_PATH, LOCK_GUARD_PATH)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--clear-observations", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 5.0 <= args.poll_seconds <= 300.0:
        raise ValueError("--poll-seconds must lie in [5, 300]")
    if not 2 <= args.clear_observations <= 20:
        raise ValueError("--clear-observations must lie in [2, 20]")
    try:
        readiness = audit_readiness()
    except Exception as error:
        if args.execute:
            raise
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "ready": False,
                    "reason": f"{type(error).__name__}: {error}",
                    "gpu_processes_queried": False,
                    "gpu_work_started": False,
                    "outer_target_label_paths_opened": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.execute:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "ready": True,
                    "frozen_file_count": readiness["frozen_file_count"],
                    "tailmiss_decision": readiness["tailmiss_gate"]["decision"],
                    "candidate_full_checkpoint_valid": readiness[
                        "candidate_full_checkpoint"
                    ]["valid"],
                    "gpu_processes_queried": False,
                    "gpu_work_started": False,
                    "outer_target_label_paths_opened": False,
                },
                sort_keys=True,
            )
        )
        return 0
    return _execute(args, readiness)


if __name__ == "__main__":
    raise SystemExit(main())
