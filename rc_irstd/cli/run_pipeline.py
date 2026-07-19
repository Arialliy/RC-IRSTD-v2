from __future__ import annotations

import argparse
import copy
import hashlib
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from evaluation.target_stage_separation import (
    audit_target_score_stage_pair,
    freeze_zero_label_actions,
)
from rc_irstd.config import apply_overrides, load_yaml, public_config, resolve_config_path
from rc_irstd.training import CalibratorTrainer, DetectorTrainer
from rc_irstd.utils.io import atomic_write_json, ensure_dir


RISK_CURVE_METHOD = "risk_curve"
DIRECT_THRESHOLD_METHOD = "direct_threshold"
RAW_LOGIT_REPRESENTATION = "raw_logit_float32"
PROBABILITY_REPRESENTATION = "sigmoid_probability_float32"
FORMAL_DETECTOR_BACKENDS = frozenset(("canonical", "rc_mshnet"))
FORMAL_METHOD_CONTRACT_SCHEMA = "rc-v2-aaai27-method-contract-v2-raw-logit"
FORMAL_RISK_ADAPTATION_WINDOW = 32
FORMAL_RISK_EVALUATION_WINDOW = 1
FORMAL_RISK_STRIDE = 33


def _absolute_sources(config: dict[str, Any], sources: list[dict[str, Any]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for source in sources:
        path = resolve_config_path(config, source["path"])
        result.append({"name": str(source.get("name", path.name)), "path": str(path)})
    return result


def _save_yaml(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(public_config(config), handle, sort_keys=False, allow_unicode=True)


def _run_command(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)




def _append_export_options(command: list[str], options: dict[str, Any]) -> None:
    command.append("--warm-flag")
    if options.get("num_workers") is not None:
        command.extend(["--num-workers", str(options["num_workers"])])
    if options.get("image_size") is not None:
        command.extend(["--image-size", str(options["image_size"])])
    if bool(options.get("resize_eval", False)):
        command.append("--resize-eval")


def _append_risk_export_contract(
    command: list[str], *, representation: str, labels_loaded: bool
) -> None:
    """Append the explicit score/label contract for one RiskCurve export."""

    if representation == RAW_LOGIT_REPRESENTATION:
        command.append("--export-raw-logits")
    command.append("--labels-loaded" if labels_loaded else "--no-labels-loaded")


def _risk_representation(
    risk_config: dict[str, Any], *, diagnostic_only: bool
) -> str:
    representation = str(
        risk_config.get("representation", RAW_LOGIT_REPRESENTATION)
    ).strip()
    if not diagnostic_only and representation != RAW_LOGIT_REPRESENTATION:
        raise ValueError(
            "Formal RiskCurve requires representation=raw_logit_float32"
        )
    if representation not in {
        RAW_LOGIT_REPRESENTATION,
        PROBABILITY_REPRESENTATION,
    }:
        raise ValueError(f"Unsupported RiskCurve representation: {representation!r}")
    return representation


def _formal_outer_target_name(
    risk_config: dict[str, Any], final_targets: list[dict[str, Any]]
) -> str:
    explicit = str(risk_config.get("outer_target", "")).strip()
    target_names = [
        str(target.get("name", Path(str(target["path"])).name)).strip()
        for target in final_targets
    ]
    if explicit:
        if target_names and explicit not in target_names:
            raise ValueError(
                "risk_curve.outer_target must name one configured final target"
            )
        return explicit
    if len(target_names) != 1:
        raise ValueError(
            "Formal raw-logit grid construction requires exactly one final target "
            "or explicit risk_curve.outer_target"
        )
    return target_names[0]


def _load_and_validate_method_contract(
    pipeline: dict[str, Any],
    *,
    method_name: str,
    detector_backend: str,
    representation: str | None,
    diagnostic_only: bool,
) -> dict[str, Any] | None:
    """Load an optional frozen AAAI method contract and fail closed on drift."""

    configured = pipeline.get("method_contract")
    if configured is None:
        if bool(pipeline.get("require_method_contract", False)):
            raise ValueError("This pipeline requires method_contract")
        return None
    path = resolve_config_path(pipeline, configured)
    contract = load_yaml(path)
    if diagnostic_only:
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "schema_version": contract.get("schema_version"),
            "validated_for_formal": False,
        }
    if contract.get("schema_version") != FORMAL_METHOD_CONTRACT_SCHEMA:
        raise ValueError("Formal method_contract has an unsupported schema_version")
    contract_method = contract.get("method")
    if not isinstance(contract_method, dict) or contract_method.get("name") != method_name:
        raise ValueError("Formal method_contract method.name differs from pipeline")
    contract_representation = contract.get("representation")
    if (
        not isinstance(contract_representation, dict)
        or contract_representation.get("name") != RAW_LOGIT_REPRESENTATION
        or representation != RAW_LOGIT_REPRESENTATION
    ):
        raise ValueError("Formal method_contract requires raw_logit_float32")
    threshold_grid = contract.get("threshold_grid")
    if (
        not isinstance(threshold_grid, dict)
        or threshold_grid.get("builder")
        != "risk_curve.build_logit_threshold_grid"
        or threshold_grid.get("require_outer_target_excluded") is not True
        or threshold_grid.get("require_outer_final_detector") is not True
        or threshold_grid.get("require_inner_pseudo_target_detectors") is not True
    ):
        raise ValueError("Formal method_contract raw-logit grid contract is invalid")
    detector = contract.get("detector")
    proposed = detector.get("proposed") if isinstance(detector, dict) else None
    if not isinstance(proposed, dict) or proposed.get("backend") != detector_backend:
        raise ValueError(
            "Formal method_contract proposed detector backend differs from template"
        )
    protocol = contract.get("protocol")
    separation = (
        protocol.get("target_stage_separation")
        if isinstance(protocol, dict)
        else None
    )
    required_separation = {
        "selection_scores": "scores_unlabeled",
        "selection_labels_loaded": False,
        "selection_freeze_required": True,
        "labeled_audit_scores": "scores_labeled_audit",
        "labeled_audit_after_freeze": True,
    }
    if not isinstance(separation, dict) or any(
        separation.get(key) != value for key, value in required_separation.items()
    ):
        raise ValueError("Formal method_contract target-stage separation is invalid")
    pipeline_separation = pipeline.get("target_stage_separation")
    if pipeline_separation is not None and (
        not isinstance(pipeline_separation, dict)
        or any(
            pipeline_separation.get(key) != value
            for key, value in required_separation.items()
        )
    ):
        raise ValueError(
            "pipeline target_stage_separation differs from method_contract"
        )
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "schema_version": contract["schema_version"],
        "validated_for_formal": True,
        "target_stage_separation": required_separation,
    }


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detector_initializer_contract(
    pipeline: dict[str, Any],
    sources: list[dict[str, str]],
    *,
    detector_backend: str,
    diagnostic_only: bool,
) -> dict[str, Any] | None:
    """Resolve fold-specific canonical MSHNet initializers for RC-MSHNet."""

    configured = pipeline.get("detector_initializers")
    required = detector_backend == "rc_mshnet" and not diagnostic_only
    if configured is None:
        if required:
            raise ValueError(
                "Formal RC-MSHNet requires detector_initializers.outer_final and "
                "detector_initializers.inner_by_held_out"
            )
        return None
    if not isinstance(configured, dict):
        raise TypeError("detector_initializers must be a mapping")
    outer_value = configured.get("outer_final")
    inner_values = configured.get("inner_by_held_out")
    if outer_value is None or not isinstance(inner_values, dict):
        raise ValueError(
            "detector_initializers requires outer_final and inner_by_held_out"
        )
    source_names = [source["name"] for source in sources]
    if set(inner_values) != set(source_names):
        raise ValueError(
            "detector_initializers.inner_by_held_out must cover every source exactly"
        )

    def resolve(value: Any, role: str) -> dict[str, str]:
        path = resolve_config_path(pipeline, value)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {role} detector initializer: {path}")
        return {"path": str(path), "sha256": _sha256_file(path)}

    return {
        "outer_final": resolve(outer_value, "outer-final"),
        "inner_by_held_out": {
            name: resolve(inner_values[name], f"inner {name}")
            for name in source_names
        },
    }


def _force_fixed_last_detector_config(config: dict[str, Any]) -> None:
    """Prevent the two-way local protocol from selecting on test labels."""

    config.setdefault("data", {})["val_split"] = None
    training = config.setdefault("training", {})
    training["validation_interval"] = 0
    training["checkpoint_selection"] = "fixed_last"


def _require_fixed_last_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path)
    if checkpoint.name != "last.pt":
        raise RuntimeError(
            "DetectorTrainer must return last.pt under the fixed-last protocol; "
            f"received {checkpoint}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _append_source_provenance(command: list[str], sources: list[dict[str, str]]) -> None:
    for source in sources:
        command.extend(["--source-dataset", source["name"]])


def _validate_domain_sets(
    sources: list[dict[str, str]],
    targets: list[dict[str, Any]],
    pipeline: dict[str, Any],
    *,
    diagnostic_only: bool,
) -> None:
    source_names = [source["name"] for source in sources]
    if len(set(source_names)) != len(source_names):
        raise ValueError("meta_sources must use unique domain names")
    source_paths = {str(Path(source["path"]).resolve()) for source in sources}
    target_names: list[str] = []
    for target in targets:
        target_path = resolve_config_path(pipeline, target["path"])
        target_name = str(target.get("name", target_path.name))
        target_names.append(target_name)
        overlap_allowed = bool(target.get("allow_source_overlap_for_debug", False))
        if overlap_allowed and not diagnostic_only:
            raise ValueError(
                "allow_source_overlap_for_debug requires pipeline diagnostic_only=true"
            )
        if bool(target.get("resize_eval", False)) and not diagnostic_only:
            raise ValueError(
                "Final-target resize_eval requires pipeline diagnostic_only=true; "
                "formal metrics use native resolution"
            )
        overlaps_source_name = target_name in source_names
        overlaps_source_path = str(target_path.resolve()) in source_paths
        if (overlaps_source_name or overlaps_source_path) and not overlap_allowed:
            reasons: list[str] = []
            if overlaps_source_name:
                reasons.append("name")
            if overlaps_source_path:
                reasons.append("path")
            raise ValueError(
                f"Final target '{target_name}' overlaps a meta-source by "
                f"{' and '.join(reasons)}. Use a genuinely unseen target, or set "
                "allow_source_overlap_for_debug=true only for a smoke test."
            )
    if len(set(target_names)) != len(target_names):
        raise ValueError("final_targets must use unique domain names")


def _method_name(pipeline: dict[str, Any]) -> str:
    method = pipeline.get("method", {})
    if method is None:
        method = {}
    if not isinstance(method, dict):
        raise TypeError("pipeline method must be a mapping")
    name = str(method.get("name", RISK_CURVE_METHOD)).strip().lower()
    if name not in {RISK_CURVE_METHOD, DIRECT_THRESHOLD_METHOD}:
        raise ValueError(
            "method.name must be 'risk_curve' or 'direct_threshold'; "
            f"received {name!r}"
        )
    return name


def _direct_baseline_enabled(pipeline: dict[str, Any], method_name: str) -> bool:
    if method_name == DIRECT_THRESHOLD_METHOD:
        return True
    baseline = pipeline.get("baseline", {})
    if not isinstance(baseline, dict):
        raise TypeError("pipeline baseline must be a mapping")
    direct = baseline.get("direct_threshold", {})
    if not isinstance(direct, dict):
        raise TypeError("baseline.direct_threshold must be a mapping")
    return bool(direct.get("enabled", False))


def _validated_meta_split(
    meta_config: dict[str, Any],
    *,
    method_name: str,
    diagnostic_only: bool,
) -> str:
    """Return the pseudo-target split without permitting formal test leakage."""

    split = str(meta_config.get("split", "train")).strip().lower()
    if method_name == RISK_CURVE_METHOD and split != "train":
        raise ValueError(
            "risk_curve pseudo-target score maps and curve labels must use each "
            "source domain's official train split; source test is forbidden"
        )
    if method_name == DIRECT_THRESHOLD_METHOD and not diagnostic_only and split != "train":
        raise ValueError(
            "Formal direct_threshold pseudo-target episodes must use each source "
            "domain's official train split"
        )
    return split


def _risk_window_contract(
    meta_config: dict[str, Any], *, diagnostic_only: bool
) -> tuple[int, int, int]:
    adaptation = int(
        meta_config.get(
            "adaptation_window",
            meta_config.get("support_size", FORMAL_RISK_ADAPTATION_WINDOW),
        )
    )
    evaluation = int(
        meta_config.get(
            "evaluation_window",
            meta_config.get("query_size", FORMAL_RISK_EVALUATION_WINDOW),
        )
    )
    stride = int(meta_config.get("stride", adaptation + evaluation))
    if min(adaptation, evaluation, stride) <= 0:
        raise ValueError("risk_curve A, E, and stride must be positive")
    if str(meta_config.get("mode", "causal")).strip().lower() != "causal":
        raise ValueError("risk_curve meta.mode must be causal")
    if stride < adaptation + evaluation:
        raise ValueError("risk_curve stride must be at least A+E")
    if not diagnostic_only and (
        adaptation,
        evaluation,
        stride,
    ) != (
        FORMAL_RISK_ADAPTATION_WINDOW,
        FORMAL_RISK_EVALUATION_WINDOW,
        FORMAL_RISK_STRIDE,
    ):
        raise ValueError("Formal risk_curve episodes require A=32, E=1, stride=33")
    return adaptation, evaluation, stride


def _direct_window_contract(meta_config: dict[str, Any]) -> tuple[int, int, int]:
    support = int(meta_config.get("support_size", 32))
    query = int(meta_config.get("query_size", 64))
    stride = int(meta_config.get("stride", support + query))
    if min(support, query, stride) <= 0:
        raise ValueError("direct_threshold support, query, and stride must be positive")
    if str(meta_config.get("mode", "causal")).strip().lower() != "causal":
        raise ValueError("Formal pipeline meta.mode must be causal")
    if stride < support + query:
        raise ValueError(
            "Formal direct_threshold episodes require meta.stride >= "
            f"support_size + query_size ({support + query}); received {stride}"
        )
    return support, query, stride


def _risk_budget_pairs(risk_config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_pairs = risk_config.get(
        "budget_pairs",
        [{"name": "pixel_1e-6_component_1", "pixel": 1e-6, "component": 1.0}],
    )
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("risk_curve.budget_pairs must be a non-empty list")
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(raw_pairs):
        if not isinstance(item, dict):
            raise TypeError("Every risk_curve budget pair must be a mapping")
        pixel = float(item["pixel"])
        component = float(item["component"])
        if pixel <= 0.0 or component <= 0.0:
            raise ValueError("Pixel and component budgets must be positive")
        raw_name = str(item.get("name", f"budget_{index}"))
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("._")
        if not name or name in names:
            raise ValueError("risk_curve budget-pair names must be non-empty and unique")
        names.add(name)
        result.append({"name": name, "pixel": pixel, "component": component})
    return result


def _target_risk_mode(target: dict[str, Any]) -> str:
    raw = str(target.get("risk_curve_mode", "static-cross-fit")).strip().lower()
    if raw in {"static", "cross_fit", "cross-fit", "static_cross_fit"}:
        return "static-cross-fit"
    if raw in {"temporal", "causal"}:
        return "causal"
    if raw != "static-cross-fit":
        raise ValueError(
            "final target risk_curve_mode must be static-cross-fit or causal"
        )
    return raw


def _static_cross_fit_statistics_arguments(
    target: dict[str, Any],
    *,
    seed: int,
    adaptation_window: int,
) -> list[str]:
    """Return the explicit fixed-A contract for static target statistics."""

    if adaptation_window <= 0:
        raise ValueError("static cross-fit adaptation_window must be positive")
    return [
        "--folds",
        str(int(target.get("cross_fit_folds", 5))),
        "--seed",
        str(int(target.get("cross_fit_seed", seed))),
        "--adaptation-window",
        str(int(adaptation_window)),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the complete strict LODO RC-IRSTD pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a nested key, e.g. --set devices='[cuda:0,cuda:1,cuda:2]'",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = apply_overrides(load_yaml(args.config), args.overrides)
    pipeline_diagnostic = bool(pipeline.get("diagnostic_only", False))
    method_name = _method_name(pipeline)
    direct_baseline_enabled = _direct_baseline_enabled(pipeline, method_name)
    project_root = Path(__file__).resolve().parents[2]
    output_root = ensure_dir(resolve_config_path(pipeline, pipeline.get("output_dir", "outputs/pipeline")))
    detector_template = load_yaml(resolve_config_path(pipeline, pipeline["detector_template"]))
    calibrator_template: dict[str, Any] | None = None
    if direct_baseline_enabled:
        calibrator_template = load_yaml(
            resolve_config_path(pipeline, pipeline["calibrator_template"])
        )
    sources = _absolute_sources(pipeline, list(pipeline["meta_sources"]))
    detector_backend = str(
        detector_template.get("model", {}).get("backend", "canonical")
    ).lower()
    if detector_backend not in FORMAL_DETECTOR_BACKENDS and not pipeline_diagnostic:
        raise ValueError(
            "Formal pipeline permits only canonical MSHNet or RC-MSHNet"
        )
    if pipeline_diagnostic:
        detector_template["diagnostic_only"] = True
    minimum_sources = 2
    if calibrator_template is not None:
        template_validation_strategy = str(
            calibrator_template.get("validation", {}).get("strategy", "domain")
        ).lower()
        fixed_last_calibrator = template_validation_strategy in {
            "fixed_last",
            "none_fixed_last",
            "none",
        }
        minimum_sources = max(minimum_sources, 2 if fixed_last_calibrator else 3)
    if len(sources) < minimum_sources:
        raise ValueError(
            f"This method/validation protocol requires at least "
            f"{minimum_sources} meta-source domains"
        )
    final_targets = list(pipeline.get("final_targets", []))
    _validate_domain_sets(
        sources,
        final_targets,
        pipeline,
        diagnostic_only=pipeline_diagnostic,
    )
    meta_config = pipeline.get("meta", {})
    if not isinstance(meta_config, dict):
        raise TypeError("pipeline meta must be a mapping")
    meta_split = _validated_meta_split(
        meta_config,
        method_name=method_name,
        diagnostic_only=pipeline_diagnostic,
    )
    if bool(meta_config.get("resize_eval", False)) and not pipeline_diagnostic:
        raise ValueError(
            "Formal pipeline requires native-resolution pseudo-target scores; "
            "meta.resize_eval is diagnostic only"
        )
    risk_config: dict[str, Any] = {}
    risk_representation: str | None = None
    if method_name == RISK_CURVE_METHOD:
        risk_config = pipeline.get("risk_curve", {})
        if not isinstance(risk_config, dict):
            raise TypeError("pipeline risk_curve must be a mapping")
        risk_representation = _risk_representation(
            risk_config, diagnostic_only=pipeline_diagnostic
        )
        support_size, query_size, episode_stride = _risk_window_contract(
            meta_config, diagnostic_only=pipeline_diagnostic
        )
    else:
        support_size, query_size, episode_stride = _direct_window_contract(meta_config)
    method_contract_record = _load_and_validate_method_contract(
        pipeline,
        method_name=method_name,
        detector_backend=detector_backend,
        representation=risk_representation,
        diagnostic_only=pipeline_diagnostic,
    )
    detector_initializers = _detector_initializer_contract(
        pipeline,
        sources,
        detector_backend=detector_backend,
        diagnostic_only=pipeline_diagnostic,
    )
    pseudo_score_dirs: list[Path] = []
    grid_source_score_dirs: list[Path] = []
    fold_records: list[dict[str, Any]] = []
    raw_devices = pipeline.get("devices")
    devices = (
        [str(value) for value in raw_devices]
        if isinstance(raw_devices, list)
        else [str(pipeline.get("device", detector_template.get("device", "auto")))]
    )
    if not devices or any(not value for value in devices) or len(set(devices)) != len(devices):
        raise ValueError("pipeline devices must be a non-empty list of unique devices")
    if any(value.startswith("cuda:") and not value[5:].isdigit() for value in devices):
        raise ValueError("CUDA devices must use the cuda:N form")
    parallel_lodo = bool(pipeline.get("parallel_lodo", len(devices) > 1))
    method_output_root = output_root / (
        "risk_curve_main"
        if method_name == RISK_CURVE_METHOD
        else "direct_threshold_baseline"
    )

    def run_fold(
        index: int,
        held_out: dict[str, str],
        fold_device: str,
    ) -> tuple[int, Path, list[Path], dict[str, Any]]:
        fold_root = ensure_dir(output_root / "lodo" / held_out["name"])
        fold_config = copy.deepcopy(detector_template)
        fold_config["seed"] = int(pipeline.get("seed", fold_config.get("seed", 42)))
        fold_config["device"] = fold_device
        fold_config.setdefault("data", {})["sources"] = [
            source for source in sources if source["name"] != held_out["name"]
        ]
        initializer_record: dict[str, str] | None = None
        if detector_initializers is not None:
            initializer_record = detector_initializers["inner_by_held_out"][
                held_out["name"]
            ]
            fold_config.setdefault("training", {})["initialize_from"] = (
                initializer_record["path"]
            )
        _force_fixed_last_detector_config(fold_config)
        fold_config["output_dir"] = str(fold_root / "detector")
        fold_config["_config_dir"] = str(Path(args.config).resolve().parent)
        fold_config_path = fold_root / "detector_config.yaml"
        _save_yaml(fold_config_path, fold_config)
        _run_command(
            [
                sys.executable,
                "train_detector.py",
                "--config",
                str(fold_config_path),
            ],
            project_root,
        )
        detector_checkpoint = _require_fixed_last_checkpoint(
            fold_root / "detector" / "last.pt"
        )
        score_dir = method_output_root / "pseudo_target_scores" / held_out["name"]
        command = [
            sys.executable,
            "export_scores.py",
            "--checkpoint",
            str(detector_checkpoint),
            "--dataset-dir",
            held_out["path"],
            "--dataset-name",
            held_out["name"],
            "--split",
            meta_split,
            "--output-dir",
            str(score_dir),
            "--device",
            fold_device,
        ]
        _append_source_provenance(command, fold_config["data"]["sources"])
        if method_name == RISK_CURVE_METHOD:
            assert risk_representation is not None
            _append_risk_export_contract(
                command,
                representation=risk_representation,
                labels_loaded=True,
            )
        _append_export_options(command, meta_config)
        _run_command(command, project_root)
        fold_grid_dirs: list[Path] = []
        if (
            method_name == RISK_CURVE_METHOD
            and risk_representation == RAW_LOGIT_REPRESENTATION
        ):
            for trained_source in fold_config["data"]["sources"]:
                grid_score_dir = (
                    method_output_root
                    / "grid_source_scores"
                    / "inner"
                    / held_out["name"]
                    / trained_source["name"]
                )
                grid_command = [
                    sys.executable,
                    "export_scores.py",
                    "--checkpoint",
                    str(detector_checkpoint),
                    "--dataset-dir",
                    trained_source["path"],
                    "--dataset-name",
                    trained_source["name"],
                    "--split",
                    "train",
                    "--output-dir",
                    str(grid_score_dir),
                    "--device",
                    fold_device,
                ]
                _append_source_provenance(
                    grid_command, fold_config["data"]["sources"]
                )
                _append_risk_export_contract(
                    grid_command,
                    representation=RAW_LOGIT_REPRESENTATION,
                    labels_loaded=True,
                )
                _append_export_options(grid_command, meta_config)
                _run_command(grid_command, project_root)
                fold_grid_dirs.append(grid_score_dir)
        return (
            index,
            score_dir,
            fold_grid_dirs,
            {
                "held_out": held_out["name"],
                "device": fold_device,
                "detector": str(detector_checkpoint),
                "scores": str(score_dir),
                "grid_self_scores": [str(value) for value in fold_grid_dirs],
                "initializer": initializer_record,
            },
        )

    assignments = [
        (index, held_out, devices[index % len(devices)])
        for index, held_out in enumerate(sources)
    ]
    if parallel_lodo and len(devices) > 1:
        buckets: list[list[tuple[int, dict[str, str], str]]] = [
            [] for _ in devices
        ]
        for assignment in assignments:
            buckets[assignment[0] % len(devices)].append(assignment)

        def run_bucket(bucket: list[tuple[int, dict[str, str], str]]):
            return [run_fold(*assignment) for assignment in bucket]

        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            nested = list(executor.map(run_bucket, buckets))
        fold_results = [result for bucket in nested for result in bucket]
    else:
        fold_results = [run_fold(*assignment) for assignment in assignments]
    for _, score_dir, fold_grid_dirs, record in sorted(
        fold_results, key=lambda value: value[0]
    ):
        pseudo_score_dirs.append(score_dir)
        grid_source_score_dirs.extend(fold_grid_dirs)
        fold_records.append(record)

    primary_device = str(pipeline.get("device", devices[0]))
    seed = int(pipeline.get("seed", 42))
    curve_checkpoint: Path | None = None
    threshold_grid: Path | None = None
    threshold_grid_manifest: Path | None = None
    threshold_grid_digest: Path | None = None
    curve_episodes_dir: Path | None = None
    calibrator_checkpoint: Path | None = None
    direct_baseline_record: dict[str, Any] = {
        "enabled": direct_baseline_enabled,
        "method_name": DIRECT_THRESHOLD_METHOD,
        "role": "strong_baseline",
        "status": "disabled" if not direct_baseline_enabled else "pending",
    }

    final_config = copy.deepcopy(detector_template)
    final_config["seed"] = seed
    final_config["device"] = primary_device
    final_config.setdefault("data", {})["sources"] = sources
    outer_initializer_record: dict[str, str] | None = None
    if detector_initializers is not None:
        outer_initializer_record = detector_initializers["outer_final"]
        final_config.setdefault("training", {})["initialize_from"] = (
            outer_initializer_record["path"]
        )
    _force_fixed_last_detector_config(final_config)
    final_config["output_dir"] = str(output_root / "final_detector")
    final_config["_config_dir"] = str(Path(args.config).resolve().parent)
    _save_yaml(output_root / "final_detector_config.yaml", final_config)
    final_detector = _require_fixed_last_checkpoint(DetectorTrainer(final_config).run())

    if (
        method_name == RISK_CURVE_METHOD
        and risk_representation == RAW_LOGIT_REPRESENTATION
    ):
        for source in sources:
            grid_score_dir = (
                method_output_root
                / "grid_source_scores"
                / "outer_final"
                / source["name"]
            )
            grid_command = [
                sys.executable,
                "export_scores.py",
                "--checkpoint",
                str(final_detector),
                "--dataset-dir",
                source["path"],
                "--dataset-name",
                source["name"],
                "--split",
                "train",
                "--output-dir",
                str(grid_score_dir),
                "--device",
                primary_device,
            ]
            _append_source_provenance(grid_command, sources)
            _append_risk_export_contract(
                grid_command,
                representation=RAW_LOGIT_REPRESENTATION,
                labels_loaded=True,
            )
            _append_export_options(grid_command, meta_config)
            _run_command(grid_command, project_root)
            grid_source_score_dirs.append(grid_score_dir)

    def train_direct_threshold(*, diagnostic_only: bool) -> tuple[Path, Path]:
        if calibrator_template is None:
            raise RuntimeError("calibrator_template is required for direct_threshold")
        direct_root = ensure_dir(output_root / "direct_threshold_baseline")
        direct_episodes = direct_root / "episodes"
        direct_command = [
            sys.executable,
            "build_episodes.py",
            "--output-dir",
            str(direct_episodes),
            "--expected-split-role",
            meta_split,
            "--support-size",
            str(support_size),
            "--query-size",
            str(query_size),
            "--stride",
            str(episode_stride),
            "--seed",
            str(seed),
            "--risk-bins",
            str(int(meta_config.get("risk_bins", 256))),
            "--budgets",
            *[
                str(value)
                for value in meta_config.get("budgets", [1e-4, 1e-5, 1e-6])
            ],
        ]
        for score_dir in pseudo_score_dirs:
            direct_command.extend(["--score-dir", str(score_dir)])
        if diagnostic_only:
            direct_command.append("--allow-diagnostic-detector")
        max_episodes = meta_config.get("max_episodes_per_domain")
        if max_episodes is not None:
            direct_command.extend(["--max-episodes-per-domain", str(max_episodes)])
        _run_command(direct_command, project_root)
        calibrator_config = copy.deepcopy(calibrator_template)
        calibrator_config["seed"] = seed
        calibrator_config["device"] = primary_device
        calibrator_config["episodes_dir"] = str(direct_episodes)
        calibrator_config["output_dir"] = str(direct_root / "checkpoints")
        calibrator_config["diagnostic_only"] = bool(diagnostic_only)
        calibrator_config["_config_dir"] = str(Path(args.config).resolve().parent)
        validation_strategy = str(
            calibrator_config.get("validation", {}).get("strategy", "domain")
        ).lower()
        if validation_strategy not in {
            "domain",
            "domain_holdout",
            "grouped",
            "fixed_last",
            "none_fixed_last",
            "none",
        }:
            raise ValueError(
                "Calibrator validation must hold out complete pseudo-target domains "
                "or use fixed-last"
            )
        _save_yaml(direct_root / "training_config.yaml", calibrator_config)
        return CalibratorTrainer(calibrator_config).run(), direct_episodes

    if method_name == RISK_CURVE_METHOD:
        assert risk_representation is not None
        quantile = float(risk_config.get("quantile", 0.90))
        if not pipeline_diagnostic and quantile != 0.90:
            raise ValueError("Formal main pipeline requires risk_curve.quantile=0.90")
        risk_root = ensure_dir(output_root / "risk_curve_main")
        if risk_representation == RAW_LOGIT_REPRESENTATION:
            grid_root = risk_root / "threshold_grid"
            threshold_grid = grid_root / "threshold_grid.npy"
            threshold_grid_manifest = grid_root / "threshold_grid.json"
            threshold_grid_digest = grid_root / "threshold_grid.sha256"
            grid_command = [
                sys.executable,
                "-m",
                "risk_curve.build_logit_threshold_grid",
                "--outer-target",
                _formal_outer_target_name(risk_config, final_targets),
                "--output-dir",
                str(grid_root),
                "--max-grid-points",
                str(int(risk_config.get("max_grid_points", 1024))),
            ]
            for score_dir in grid_source_score_dirs:
                grid_command.extend(["--source-score-dir", str(score_dir)])
            for source in sources:
                grid_command.extend(["--expected-source-domain", source["name"]])
            _run_command(grid_command, project_root)
            for artifact in (
                threshold_grid,
                threshold_grid_manifest,
                threshold_grid_digest,
            ):
                if not artifact.is_file():
                    raise FileNotFoundError(artifact)
        else:
            # Explicit diagnostic compatibility path. Formal RiskCurve never
            # reaches the legacy probability grid.
            threshold_grid = risk_root / "threshold_grid.npy"
            _run_command(
                [
                    sys.executable,
                    "-m",
                    "risk_curve.threshold_grid",
                    "--output",
                    str(threshold_grid),
                ],
                project_root,
            )
        curve_episodes_dir = risk_root / "curve_episodes"
        validation_domain = str(
            risk_config.get("validation_domain", sources[-1]["name"])
        )
        if validation_domain not in {source["name"] for source in sources}:
            raise ValueError("risk_curve.validation_domain must be a meta-source")
        curve_episode_command = [
            sys.executable,
            "-m",
            "risk_curve.build_curve_episodes",
            "--output-dir",
            str(curve_episodes_dir),
            "--expected-split-role",
            meta_split,
            "--adaptation-window",
            str(support_size),
            "--evaluation-window",
            str(query_size),
            "--stride",
            str(episode_stride),
            "--validation-domain",
            validation_domain,
        ]
        if risk_representation == RAW_LOGIT_REPRESENTATION:
            assert threshold_grid_manifest is not None
            curve_episode_command.extend(
                [
                    "--threshold-grid-manifest",
                    str(threshold_grid_manifest),
                    "--representation",
                    RAW_LOGIT_REPRESENTATION,
                    "--count-all-workers",
                    str(int(risk_config.get("count_all_workers", 1))),
                ]
            )
        else:
            curve_episode_command.extend(["--threshold-grid", str(threshold_grid)])
        for source, score_dir in zip(sources, pseudo_score_dirs):
            curve_episode_command.extend(["--score-map-dir", str(score_dir)])
            curve_episode_command.extend(["--pseudo-target", source["name"]])
        if pipeline_diagnostic:
            curve_episode_command.append("--allow-unverified-fold-provenance")
        _run_command(curve_episode_command, project_root)
        curve_checkpoint = risk_root / "best.pt"
        curve_training_command = [
            sys.executable,
            "-m",
            "risk_curve.train_curve_predictor",
            "--train-file",
            str(curve_episodes_dir / "train.npz"),
            "--val-file",
            str(curve_episodes_dir / "val.npz"),
            "--output",
            str(curve_checkpoint),
            "--quantile",
            str(quantile),
            "--lambda-component",
            str(float(risk_config.get("lambda_component", 1.0))),
            "--hidden-dim",
            str(int(risk_config.get("hidden_dim", 256))),
            "--dropout",
            str(float(risk_config.get("dropout", 0.1))),
            "--epochs",
            str(int(risk_config.get("epochs", 200))),
            "--batch-size",
            str(int(risk_config.get("batch_size", 32))),
            "--lr",
            str(float(risk_config.get("lr", 1e-3))),
            "--weight-decay",
            str(float(risk_config.get("weight_decay", 1e-4))),
            "--patience",
            str(int(risk_config.get("patience", 30))),
            "--num-workers",
            str(int(risk_config.get("num_workers", 0))),
            "--seed",
            str(seed),
            "--device",
            primary_device,
        ]
        _save_yaml(
            risk_root / "training_config.yaml",
            {
                "method": {"name": RISK_CURVE_METHOD, "role": "proposed_method"},
                "risk_curve": risk_config,
                "meta_split": meta_split,
                "adaptation_window": support_size,
                "evaluation_window": query_size,
                "stride": episode_stride,
                "validation_domain": validation_domain,
                "device": primary_device,
            },
        )
        _run_command(curve_training_command, project_root)
        if not curve_checkpoint.is_file():
            raise FileNotFoundError(curve_checkpoint)

        if direct_baseline_enabled:
            calibrator_checkpoint, direct_episodes = train_direct_threshold(
                diagnostic_only=pipeline_diagnostic
            )
            direct_baseline_record.update(
                {
                    "status": (
                        "complete_diagnostic_only"
                        if pipeline_diagnostic
                        else "complete"
                    ),
                    "checkpoint": str(calibrator_checkpoint),
                    "episodes": str(direct_episodes),
                    "formal_paper_artifact": bool(not pipeline_diagnostic),
                    "output_dir": str(
                        output_root / "direct_threshold_baseline"
                    ),
                }
            )
    else:
        calibrator_checkpoint, direct_episodes = train_direct_threshold(
            diagnostic_only=pipeline_diagnostic
        )
        direct_baseline_record.update(
            {
                "status": "complete_diagnostic_only"
                if pipeline_diagnostic
                else "complete",
                "checkpoint": str(calibrator_checkpoint),
                "episodes": str(direct_episodes),
                "formal_paper_artifact": bool(not pipeline_diagnostic),
            }
        )

    def run_standard_detection(
        score_dir: Path, target_root: Path
    ) -> dict[str, str]:
        standard_root = ensure_dir(target_root / "standard_detection")
        standard_curve = standard_root / "threshold_sweep.csv"
        sweep_command = [
            sys.executable,
            "-m",
            "evaluation.threshold_sweep",
            "--score-dir",
            str(score_dir),
            "--output",
            str(standard_curve),
        ]
        if not pipeline_diagnostic:
            sweep_command.extend(["--formal", "--expected-split-role", "test"])
        _run_command(sweep_command, project_root)
        result = {"standard_threshold_sweep": str(standard_curve)}
        if not pipeline_diagnostic:
            standard_metrics = standard_root / "fixed_0_5_metrics.json"
            _run_command(
                [
                    sys.executable,
                    "-m",
                    "evaluation.standard_metrics",
                    "--score-dir",
                    str(score_dir),
                    "--threshold",
                    "0.5",
                    "--output",
                    str(standard_metrics),
                ],
                project_root,
            )
            result.update(
                {
                    "standard_threshold_sweep_metadata": str(
                        standard_curve.with_name(standard_curve.name + ".metadata.json")
                    ),
                    "standard_fixed_0_5_metrics": str(standard_metrics),
                }
            )
        return result

    target_records: list[dict[str, Any]] = []
    for target in final_targets:
        target_path = resolve_config_path(pipeline, target["path"])
        target_name = str(target.get("name", target_path.name))
        target_root = ensure_dir(output_root / "targets" / target_name)
        target_split = str(target.get("split", "test")).lower()
        if not pipeline_diagnostic and target_split != "test":
            raise ValueError("Formal final-target evaluation must use official test")
        evaluate_target = bool(target.get("evaluate", True))
        raw_logit_target = (
            method_name == RISK_CURVE_METHOD
            and risk_representation == RAW_LOGIT_REPRESENTATION
        )
        selection_score_dir = (
            target_root / "scores_unlabeled"
            if raw_logit_target
            else target_root / "scores"
        )
        export_command = [
            sys.executable,
            "export_scores.py",
            "--checkpoint",
            str(final_detector),
            "--dataset-dir",
            str(target_path),
            "--dataset-name",
            target_name,
            "--split",
            target_split,
            "--output-dir",
            str(selection_score_dir),
            "--device",
            primary_device,
        ]
        _append_source_provenance(export_command, sources)
        if method_name == RISK_CURVE_METHOD:
            assert risk_representation is not None
            _append_risk_export_contract(
                export_command,
                representation=risk_representation,
                labels_loaded=False if raw_logit_target else evaluate_target,
            )
        elif not evaluate_target:
            export_command.append("--allow-missing-masks")
        if raw_logit_target:
            export_command.append("--overwrite")
        _append_export_options(export_command, target)
        _run_command(export_command, project_root)
        audit_score_dir: Path | None = (
            selection_score_dir if evaluate_target and not raw_logit_target else None
        )
        record: dict[str, Any] = {
            "target": target_name,
            "split": target_split,
            "scores_unlabeled": (
                str(selection_score_dir) if raw_logit_target else None
            ),
            "scores": str(selection_score_dir),
        }
        if audit_score_dir is not None:
            record.update(run_standard_detection(audit_score_dir, target_root))

        if method_name == RISK_CURVE_METHOD:
            assert curve_checkpoint is not None
            assert threshold_grid is not None
            assert risk_representation is not None
            risk_target_root = ensure_dir(target_root / "risk_curve_main")
            deployment_statistics = risk_target_root / "deployment_statistics.npz"
            deployment_mode = _target_risk_mode(target)
            statistics_grid = (
                threshold_grid_manifest
                if risk_representation == RAW_LOGIT_REPRESENTATION
                else threshold_grid
            )
            assert statistics_grid is not None
            statistics_command = [
                sys.executable,
                "-m",
                "risk_curve.build_deployment_statistics",
                "--score-map-dir",
                str(selection_score_dir),
                "--threshold-grid",
                str(statistics_grid),
                "--representation",
                risk_representation,
                "--output",
                str(deployment_statistics),
                "--mode",
                deployment_mode,
            ]
            if deployment_mode == "static-cross-fit":
                statistics_command.extend(
                    _static_cross_fit_statistics_arguments(
                        target, seed=seed, adaptation_window=support_size
                    )
                )
            else:
                if not pipeline_diagnostic and not bool(
                    target.get("sequence_metadata_verified", False)
                ):
                    raise ValueError(
                        f"Causal target {target_name} requires "
                        "sequence_metadata_verified=true"
                    )
                temporal_meta = {
                    "mode": "causal",
                    "adaptation_window": target.get(
                        "adaptation_window", FORMAL_RISK_ADAPTATION_WINDOW
                    ),
                    "evaluation_window": target.get(
                        "evaluation_window", FORMAL_RISK_EVALUATION_WINDOW
                    ),
                    "stride": target.get("stride", FORMAL_RISK_STRIDE),
                }
                target_a, target_e, target_stride = _risk_window_contract(
                    temporal_meta, diagnostic_only=pipeline_diagnostic
                )
                statistics_command.extend(
                    [
                        "--adaptation-window",
                        str(target_a),
                        "--evaluation-window",
                        str(target_e),
                        "--stride",
                        str(target_stride),
                    ]
                )
            _run_command(statistics_command, project_root)

            budget_records: list[dict[str, Any]] = []
            zero_results: list[Path] = []
            for budget in _risk_budget_pairs(risk_config):
                budget_root = ensure_dir(risk_target_root / "budgets" / budget["name"])
                zero_result = budget_root / "zero_selection.json"
                _run_command(
                    [
                        sys.executable,
                        "-m",
                        "risk_curve.select_zero_label_threshold",
                        "--statistics-file",
                        str(deployment_statistics),
                        "--curve-checkpoint",
                        str(curve_checkpoint),
                        "--pixel-budget",
                        str(budget["pixel"]),
                        "--component-budget",
                        str(budget["component"]),
                        "--output",
                        str(zero_result),
                        "--device",
                        primary_device,
                    ],
                    project_root,
                )
                zero_results.append(zero_result)
                budget_records.append({**budget, "zero_selection": str(zero_result)})

            freeze_record: Path | None = None
            pair_audit_path: Path | None = None
            if raw_logit_target:
                assert threshold_grid_manifest is not None
                assert threshold_grid_digest is not None
                freeze_record = freeze_zero_label_actions(
                    zero_results,
                    bound_artifacts=(
                        deployment_statistics,
                        curve_checkpoint,
                        threshold_grid,
                        threshold_grid_manifest,
                        threshold_grid_digest,
                    ),
                    output_dir=risk_target_root / "zero_label_freeze",
                )
                record["zero_label_selection_freeze"] = str(freeze_record)

            if evaluate_target:
                if raw_logit_target:
                    assert freeze_record is not None
                    audit_score_dir = target_root / "scores_labeled_audit"
                    labeled_command = [
                        sys.executable,
                        "export_scores.py",
                        "--checkpoint",
                        str(final_detector),
                        "--dataset-dir",
                        str(target_path),
                        "--dataset-name",
                        target_name,
                        "--split",
                        target_split,
                        "--output-dir",
                        str(audit_score_dir),
                        "--device",
                        primary_device,
                    ]
                    _append_source_provenance(labeled_command, sources)
                    _append_risk_export_contract(
                        labeled_command,
                        representation=RAW_LOGIT_REPRESENTATION,
                        labels_loaded=True,
                    )
                    labeled_command.append("--overwrite")
                    _append_export_options(labeled_command, target)
                    _run_command(labeled_command, project_root)
                    pair_audit_path = risk_target_root / "target_stage_pair_audit.json"
                    audit_target_score_stage_pair(
                        selection_score_dir,
                        audit_score_dir,
                        freeze_record=freeze_record,
                        output=pair_audit_path,
                    )
                    record.update(
                        {
                            "scores": str(audit_score_dir),
                            "scores_labeled_audit": str(audit_score_dir),
                            "target_stage_pair_audit": str(pair_audit_path),
                        }
                    )
                    record.update(run_standard_detection(audit_score_dir, target_root))
                assert audit_score_dir is not None
                for budget, budget_record, zero_result in zip(
                    _risk_budget_pairs(risk_config), budget_records, zero_results
                ):
                    budget_root = ensure_dir(
                        risk_target_root / "budgets" / budget["name"]
                    )
                    count_curves = budget_root / "calibration_losses.npz"
                    count_command = [
                        sys.executable,
                        "-m",
                        "certification.build_calibration_losses",
                        "--score-dir",
                        str(audit_score_dir),
                        "--threshold-grid",
                        str(threshold_grid),
                        "--representation",
                        risk_representation,
                        "--pixel-budget",
                        str(budget["pixel"]),
                        "--component-budget",
                        str(budget["component"]),
                        "--output",
                        str(count_curves),
                    ]
                    if risk_representation == RAW_LOGIT_REPRESENTATION:
                        assert threshold_grid_manifest is not None
                        count_command.extend(
                            [
                                "--threshold-grid-manifest",
                                str(threshold_grid_manifest),
                            ]
                        )
                    _run_command(count_command, project_root)
                    zero_evaluation = budget_root / "zero_label_evaluation.json"
                    evaluate_zero_command = [
                        sys.executable,
                        "-m",
                        "risk_curve.evaluate_zero_label",
                        "--zero-result",
                        str(zero_result),
                        "--count-curves",
                        str(count_curves),
                        "--output",
                        str(zero_evaluation),
                    ]
                    if pair_audit_path is not None:
                        evaluate_zero_command.extend(
                            ["--target-stage-pair-audit", str(pair_audit_path)]
                        )
                    if deployment_mode == "causal":
                        evaluate_zero_command.append("--mapped-actions-only")
                    _run_command(evaluate_zero_command, project_root)
                    selected_action_metrics: Path | None = None
                    if (
                        raw_logit_target
                        and deployment_mode == "static-cross-fit"
                        and not pipeline_diagnostic
                    ):
                        assert pair_audit_path is not None
                        selected_action_metrics = (
                            budget_root / "selected_action_metrics.json"
                        )
                        _run_command(
                            [
                                sys.executable,
                                "-m",
                                "evaluation.evaluate_selected_actions",
                                "--score-dir",
                                str(audit_score_dir),
                                "--selection",
                                str(zero_result),
                                "--count-curves",
                                str(count_curves),
                                "--target-stage-pair-audit",
                                str(pair_audit_path),
                                "--representation",
                                risk_representation,
                                "--matching-rule",
                                "overlap",
                                "--connectivity",
                                "2",
                                "--min-component-area",
                                "1",
                                "--reject-as-empty",
                                "--output",
                                str(selected_action_metrics),
                            ],
                            project_root,
                        )
                    budget_record.update(
                        {
                            "calibration_losses": str(count_curves),
                            "zero_label_evaluation": str(zero_evaluation),
                            "selected_action_metrics": (
                                str(selected_action_metrics)
                                if selected_action_metrics is not None
                                else None
                            ),
                        }
                    )

            record.update(
                {
                    "risk_curve_mode": deployment_mode,
                    "deployment_statistics": str(deployment_statistics),
                    "risk_curve_budgets": budget_records,
                }
            )
            if direct_baseline_enabled:
                assert calibrator_checkpoint is not None
                direct_target_root = ensure_dir(target_root / "direct_threshold_baseline")
                if deployment_mode != "static-cross-fit":
                    record["direct_threshold_baseline"] = {
                        "status": "not_run",
                        "reason": "RC-Direct target evaluation requires static cross-fit",
                    }
                elif not evaluate_target:
                    record["direct_threshold_baseline"] = {
                        "status": "not_run",
                        "reason": "Labelled target evaluation was disabled",
                    }
                elif pipeline_diagnostic:
                    record["direct_threshold_baseline"] = {
                        "status": "not_run",
                        "reason": "Strict RC-Direct rejects diagnostic artifacts",
                    }
                else:
                    assert audit_score_dir is not None
                    direct_evaluation = direct_target_root / "static_cross_fit.json"
                    direct_command = [
                        sys.executable,
                        "-m",
                        "rc_irstd.cli.evaluate_static_crossfit_direct",
                        "--score-dir",
                        str(audit_score_dir),
                        "--calibrator",
                        str(calibrator_checkpoint),
                        "--output",
                        str(direct_evaluation),
                        "--seed",
                        str(int(target.get("cross_fit_seed", seed))),
                        "--folds",
                        str(int(target.get("cross_fit_folds", 5))),
                        "--device",
                        primary_device,
                    ]
                    for budget in _risk_budget_pairs(risk_config):
                        direct_command.extend(
                            [
                                "--budget-pair",
                                f"{budget['name']}:{budget['pixel']}:{budget['component']}",
                            ]
                        )
                    _run_command(direct_command, project_root)
                    record["direct_threshold_baseline"] = {
                        "status": "complete",
                        "evaluation": str(direct_evaluation),
                        "protocol": "static_5fold_fixed_A_cross_fit",
                        "full_test_coverage": True,
                    }
            crc_config = pipeline.get("crc", {})
            if isinstance(crc_config, dict) and bool(crc_config.get("enabled", False)):
                explicit_calibration = target.get("crc_calibration_split")
                explicit_test = target.get("crc_test_split")
                record["crc"] = {
                    "status": "not_run",
                    "reason": (
                        "explicit CRC is a separate few-shot stage"
                        if explicit_calibration and explicit_test
                        else "disjoint crc_calibration_split and crc_test_split missing"
                    ),
                    "calibration_split": (
                        str(explicit_calibration) if explicit_calibration else None
                    ),
                    "test_split": str(explicit_test) if explicit_test else None,
                }
        elif evaluate_target:
            assert calibrator_checkpoint is not None
            assert audit_score_dir is not None
            eval_dir = target_root / "causal_evaluation"
            evaluate_command = [
                sys.executable,
                "evaluate_causal.py",
                "--score-dir",
                str(audit_score_dir),
                "--calibrator",
                str(calibrator_checkpoint),
                "--output-dir",
                str(eval_dir),
                "--support-size",
                str(target.get("support_size", meta_config.get("support_size", 32))),
                "--device",
                primary_device,
                "--all-windows",
                "--query-size",
                str(int(target.get("query_size", query_size))),
            ]
            requested_budgets = target.get("budgets")
            if requested_budgets:
                evaluate_command.append("--budgets")
                evaluate_command.extend(str(value) for value in requested_budgets)
            if pipeline_diagnostic:
                evaluate_command.append("--allow-diagnostic-artifacts")
            _run_command(evaluate_command, project_root)
            record["evaluation"] = str(eval_dir / "causal_metrics.json")
        elif method_name == DIRECT_THRESHOLD_METHOD:
            assert calibrator_checkpoint is not None
            inference_dir = target_root / "online_predictions"
            default_budget = list(meta_config.get("budgets", [1e-4, 1e-5, 1e-6]))[-1]
            inference_command = [
                sys.executable,
                "infer_online.py",
                "--score-dir",
                str(selection_score_dir),
                "--calibrator",
                str(calibrator_checkpoint),
                "--output-dir",
                str(inference_dir),
                "--budget",
                str(target.get("budget", default_budget)),
                "--support-size",
                str(target.get("support_size", meta_config.get("support_size", 32))),
                "--query-size",
                str(target.get("query_size", meta_config.get("query_size", 64))),
                "--device",
                primary_device,
                "--all-windows",
            ]
            if pipeline_diagnostic:
                inference_command.append("--allow-diagnostic-artifacts")
            _run_command(inference_command, project_root)
            record["inference"] = str(inference_dir / "inference.json")
        target_records.append(record)

    episode_manifest: Path | None = None
    if curve_episodes_dir is not None:
        episode_manifest = curve_episodes_dir / "manifest.json"
    summary = {
        "schema_version": "rc-v2-complete-pipeline-v4-raw-logit",
        "config": str(Path(args.config).resolve()),
        "config_sha256": _sha256_file(args.config),
        "method_contract": method_contract_record,
        "method_name": method_name,
        "method_role": (
            "proposed_method"
            if method_name == RISK_CURVE_METHOD
            else "strong_baseline"
        ),
        "formal_protocol_contract": not pipeline_diagnostic,
        "diagnostic_only": pipeline_diagnostic,
        "detector_backend": detector_backend,
        "detector_initializers": detector_initializers,
        "outer_final_initializer": outer_initializer_record,
        "lodo_devices": devices,
        "parallel_lodo": parallel_lodo,
        "detector_checkpoint_selection": "fixed_last",
        "test_labels_used_for_detector_selection": False,
        "pseudo_target_split": meta_split,
        "source_test_used_for_meta_training": False if meta_split == "train" else True,
        "risk_curve_representation": risk_representation,
        "threshold_grid_contract": (
            {
                "grid": str(threshold_grid),
                "grid_file_sha256": _sha256_file(threshold_grid),
                "manifest": str(threshold_grid_manifest),
                "manifest_sha256": _sha256_file(threshold_grid_manifest),
                "digest": str(threshold_grid_digest),
                "grid_source_scores": [
                    str(value) for value in grid_source_score_dirs
                ],
            }
            if threshold_grid is not None
            and threshold_grid.is_file()
            and threshold_grid_manifest is not None
            and threshold_grid_manifest.is_file()
            and threshold_grid_digest is not None
            and threshold_grid_digest.is_file()
            else None
        ),
        "risk_curve_episode_contract": (
            {
                "adaptation_window": support_size,
                "evaluation_window": query_size,
                "stride": episode_stride,
                "manifest": str(episode_manifest),
                "manifest_sha256": _sha256_file(episode_manifest),
            }
            if episode_manifest is not None and episode_manifest.is_file()
            else None
        ),
        "lodo_folds": fold_records,
        "risk_curve_checkpoint": str(curve_checkpoint) if curve_checkpoint else None,
        "direct_threshold_baseline": direct_baseline_record,
        "final_detector": str(final_detector),
        "final_detector_sha256": _sha256_file(final_detector),
        "targets": target_records,
    }
    atomic_write_json(output_root / "pipeline_summary.json", summary)
    print(output_root / "pipeline_summary.json")


if __name__ == "__main__":
    main()
