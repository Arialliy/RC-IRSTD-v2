"""Two-phase source evaluation for AnchorWarp versus train-only RC-Direct."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from rc_irstd.models.calibrator import MonotoneBudgetCalibrator

from .anchor_warp_inference import (
    load_anchor_warp_policy,
    predict_anchor_warp_curves,
    prepare_anchor_warp_inputs,
)
from .bind_anchor_warp_validation_v4 import (
    _load_label_free_fields,
    validate_anchor_warp_bound_package,
)
from .curve_dataset import load_curve_archive
from .direct_calibrator import quantize_direct_logit_threshold
from .evaluate_source_pseudo_target_v4 import (
    FORMAL_OUTER_DOMAIN_NAME,
    FORMAL_SOURCE_DOMAIN_NAMES,
    _action_evidence,
    _aggregate_actions,
    _episode_identifier,
    _gate_decision,
    _validate_formal_archive,
    _validate_requested_budget_pairs,
)
from .representation import LOGIT_REPRESENTATION
from .select_zero_label_threshold import (
    EMPTY_ACTION_THRESHOLD,
    select_dual_budget_threshold,
)
from .train_direct_calibrator_train_only_v4 import (
    validate_train_only_direct_checkpoint,
)


ANCHOR_WARP_SOURCE_COMPARISON_SCHEMA_VERSION = (
    "rc-v4-anchor-warp-source-pseudo-target-comparison-v1-two-phase"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_torch_mapping(path: Path, *, role: str) -> tuple[Mapping[str, Any], bytes, str]:
    if not path.is_file():
        raise FileNotFoundError(f"{role} artifact does not exist: {path}")
    raw = path.read_bytes()
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{role} artifact must contain a mapping")
    return payload, raw, _sha256_bytes(raw)


def _resolve_device(value: str) -> torch.device:
    name = "cuda" if value == "auto" and torch.cuda.is_available() else value
    if name == "auto":
        name = "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _direct_budget_indices(
    checkpoint: Mapping[str, Any],
    pixel: np.ndarray,
    component: np.ndarray,
) -> list[int]:
    registered_pixel = np.asarray(checkpoint["pixel_budgets"], dtype=np.float64)
    registered_component = np.asarray(
        checkpoint["component_budgets"], dtype=np.float64
    )
    indices: list[int] = []
    for requested_pixel, requested_component in zip(pixel, component):
        matches = np.flatnonzero(
            np.isclose(registered_pixel, requested_pixel, rtol=1.0e-6, atol=0.0)
            & np.isclose(
                registered_component, requested_component, rtol=1.0e-6, atol=0.0
            )
        )
        if matches.size != 1:
            raise ValueError("requested budget is not uniquely registered by RC-Direct")
        indices.append(int(matches[0]))
    return indices


def _action_digest(records: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        list(records), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_anchor_warp_source_comparison(
    *,
    episode_file: str | Path,
    anchor_warp_package: str | Path,
    rc_direct_checkpoint: str | Path,
    output: str | Path,
    pixel_budgets: Sequence[float] = (1.0e-5, 1.0e-6),
    component_budgets: Sequence[float] = (5.0, 1.0),
    device: str = "cpu",
    batch_size: int = 64,
) -> Path:
    episode_path = Path(episode_file).expanduser().resolve()
    anchor_path = Path(anchor_warp_package).expanduser().resolve()
    direct_path = Path(rc_direct_checkpoint).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not episode_path.is_file():
        raise FileNotFoundError(f"episode archive does not exist: {episode_path}")
    torch_device = _resolve_device(device)
    requested_pixel, requested_component = _validate_requested_budget_pairs(
        pixel_budgets, component_budgets
    )
    episode_raw = episode_path.read_bytes()
    episode_sha = _sha256_bytes(episode_raw)
    anchor_package, anchor_raw, anchor_sha = _load_torch_mapping(
        anchor_path, role="AnchorWarp package"
    )
    direct_checkpoint, direct_raw, direct_sha = _load_torch_mapping(
        direct_path, role="RC-Direct checkpoint"
    )
    anchor_contract = validate_anchor_warp_bound_package(anchor_package)
    direct_contract = validate_train_only_direct_checkpoint(direct_checkpoint)
    binding = anchor_package["validation_binding"]
    frozen = anchor_package["frozen_checkpoint"]
    if binding["validation_archive_sha256"] != episode_sha:
        raise ValueError("AnchorWarp package is not bound to the supplied episode archive")
    if direct_checkpoint["validation_archive_sha256"] != episode_sha:
        raise ValueError("RC-Direct checkpoint is not bound to the supplied episode archive")
    if frozen["train_archive_sha256"] != direct_checkpoint["train_archive_sha256"]:
        raise ValueError("AnchorWarp/RC-Direct were not trained on identical train bytes")
    if int(frozen["seed"]) != int(direct_checkpoint["seed"]):
        raise ValueError("AnchorWarp/RC-Direct seed mismatch")
    for field in (
        "threshold_grid_sha256",
        "threshold_grid_manifest_sha256",
        "feature_schema_sha256",
        "threshold_grid_detector_protocol",
        "threshold_grid_detector_checkpoint_sha256s",
        "threshold_grid_outer_detector_checkpoint_sha256",
        "threshold_grid_episode_detector_checkpoint_sha256s",
    ):
        left = frozen[field]
        right = direct_checkpoint[field]
        if tuple(left) != tuple(right) if isinstance(left, list) else left != right:
            raise ValueError(f"AnchorWarp/RC-Direct {field} mismatch")
    direct_budget_indices = _direct_budget_indices(
        direct_checkpoint, requested_pixel, requested_component
    )

    # Phase A: load only the explicit label-free allowlist, predict both
    # methods, and freeze an action digest before future-E sufficient counts
    # are decompressed.
    label_free_archive = _load_label_free_fields(episode_raw, role="evaluation")
    policy = load_anchor_warp_policy(anchor_package, device=torch_device)
    anchor_inputs = prepare_anchor_warp_inputs(label_free_archive, policy)
    anchor_predictions = predict_anchor_warp_curves(
        policy, anchor_inputs, batch_size=batch_size
    )
    thresholds = np.asarray(label_free_archive["thresholds"], dtype=np.float32)
    statistics = np.asarray(label_free_archive["statistics"], dtype=np.float32)
    direct_model = MonotoneBudgetCalibrator(**direct_checkpoint["model_config"])
    direct_model.load_state_dict(direct_checkpoint["state_dict"], strict=True)
    direct_model.to(torch_device).eval()
    direct_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, statistics.shape[0], batch_size):
            stop = min(statistics.shape[0], start + batch_size)
            direct_batches.append(
                direct_model(
                    torch.from_numpy(
                        np.array(
                            statistics[start:stop],
                            dtype=np.float32,
                            copy=True,
                            order="C",
                        )
                    ).to(torch_device)
                ).grid_logits.cpu().numpy()
            )
    direct_logits = np.concatenate(direct_batches, axis=0)
    frozen_actions: list[dict[str, Any]] = []
    for budget_position, (pixel_budget, component_budget, direct_budget_index) in enumerate(
        zip(requested_pixel, requested_component, direct_budget_indices)
    ):
        for row in range(statistics.shape[0]):
            threshold, reject, index = select_dual_budget_threshold(
                thresholds,
                anchor_predictions["pixel_log_risk"][row],
                anchor_predictions["component_log_risk"][row],
                float(pixel_budget),
                float(component_budget),
                representation=LOGIT_REPRESENTATION,
            )
            direct_selection = quantize_direct_logit_threshold(
                float(direct_logits[row, direct_budget_index]), thresholds
            )
            frozen_actions.append(
                {
                    "budget_position": budget_position,
                    "episode_index": row,
                    "anchor_warp": {
                        "threshold_index": index,
                        "threshold": (
                            "+inf" if reject else float(threshold)
                        ),
                        "reject": bool(reject),
                    },
                    "rc_direct": {
                        "threshold_index": direct_selection.threshold_index,
                        "threshold": (
                            "+inf"
                            if direct_selection.threshold_index is None
                            else float(direct_selection.selected_logit_threshold)
                        ),
                        "reject": direct_selection.threshold_index is None,
                    },
                }
            )
    action_digest = _action_digest(frozen_actions)
    if len(frozen_actions) != statistics.shape[0] * requested_pixel.size:
        raise RuntimeError("two-phase action matrix is incomplete")

    # Phase E: only now parse the full future-E evidence archive.  No model,
    # scaler, budget, or action is recomputed after this line.
    full_archive = load_curve_archive(io.BytesIO(episode_raw))
    full_thresholds, full_statistics, names, provenance = _validate_formal_archive(
        full_archive
    )
    if not np.array_equal(full_thresholds, thresholds) or not np.array_equal(
        full_statistics, statistics
    ):
        raise ValueError("label-free snapshot and full evidence archive disagree")
    if tuple(names) != tuple(frozen["statistics_names"]):
        raise ValueError("evaluation feature order changed between phases")
    adaptation_ids = [
        _episode_identifier(value, row, field="adaptation_ids")
        for row, value in enumerate(np.asarray(full_archive["adaptation_ids"]).tolist())
    ]
    evaluation_ids = [
        _episode_identifier(value, row)
        for row, value in enumerate(np.asarray(full_archive["evaluation_ids"]).tolist())
    ]
    row_targets = [str(value) for value in np.asarray(full_archive["pseudo_targets"]).tolist()]
    per_episode = [
        {
            "episode_index": row,
            "pseudo_target": row_targets[row],
            "adaptation_ids": adaptation_ids[row],
            "evaluation_ids": evaluation_ids[row],
            "actions": [],
        }
        for row in range(statistics.shape[0])
    ]
    budget_results: list[dict[str, Any]] = []
    action_codes = {
        "risk_curve": [[] for _ in range(statistics.shape[0])],
        "rc_direct": [[] for _ in range(statistics.shape[0])],
    }
    cursor = 0
    for budget_position, (pixel_budget, component_budget, direct_budget_index) in enumerate(
        zip(requested_pixel, requested_component, direct_budget_indices)
    ):
        method_actions: dict[str, list[dict[str, Any]]] = {
            "risk_curve": [],
            "rc_direct": [],
        }
        for row in range(statistics.shape[0]):
            frozen_action = frozen_actions[cursor]
            cursor += 1
            episode_methods: dict[str, Any] = {}
            for method, frozen_name in (
                ("risk_curve", "anchor_warp"),
                ("rc_direct", "rc_direct"),
            ):
                selection = frozen_action[frozen_name]
                index = selection["threshold_index"]
                threshold = (
                    EMPTY_ACTION_THRESHOLD
                    if index is None
                    else float(selection["threshold"])
                )
                action = _action_evidence(
                    full_archive,
                    row=row,
                    index=index,
                    threshold=threshold,
                    pixel_budget=float(pixel_budget),
                    component_budget=float(component_budget),
                )
                if method == "risk_curve":
                    action.update(
                        {
                            "model_class": "CountAllAnchorWarpRiskCurve",
                            "selection_rule": "earliest_jointly_feasible_grid_index",
                            "predicted_pixel_log_risk_at_action": (
                                None
                                if index is None
                                else float(
                                    anchor_predictions["pixel_log_risk"][row, index]
                                )
                            ),
                            "predicted_component_log_risk_at_action": (
                                None
                                if index is None
                                else float(
                                    anchor_predictions["component_log_risk"][row, index]
                                )
                            ),
                        }
                    )
                else:
                    action.update(
                        {
                            "selection_rule": "conservative_left_grid_quantization",
                            "predicted_logit_before_quantization": float(
                                direct_logits[row, direct_budget_index]
                            ),
                            "registered_budget_index": direct_budget_index,
                        }
                    )
                method_actions[method].append(action)
                episode_methods[method] = action
                action_codes[method][row].append(
                    thresholds.size if index is None else int(index)
                )
            per_episode[row]["actions"].append(
                {
                    "budget_position": budget_position,
                    "pixel_budget": float(pixel_budget),
                    "component_budget": float(component_budget),
                    "methods": episode_methods,
                }
            )
        budget_results.append(
            {
                "budget_position": budget_position,
                "pixel_budget": float(pixel_budget),
                "component_budget": float(component_budget),
                "methods": {
                    method: _aggregate_actions(actions)
                    for method, actions in method_actions.items()
                },
            }
        )
    monotonic_violation_rates: dict[str, float] = {}
    for method, rows in action_codes.items():
        violations = sum(
            int(any(right < left for left, right in zip(row, row[1:])))
            for row in rows
        )
        monotonic_violation_rates[method] = violations / float(len(rows))
    gate = _gate_decision(
        budget_results,
        risk_monotonic_violation_rate=monotonic_violation_rates["risk_curve"],
    )
    payload = {
        "schema_version": ANCHOR_WARP_SOURCE_COMPARISON_SCHEMA_VERSION,
        "protocol": "source_only_two_phase_action_then_future_e_evaluation",
        "representation": LOGIT_REPRESENTATION,
        "labels_used_for_action_selection": False,
        "future_e_arrays_loaded_before_action_digest": False,
        "source_pseudo_target_labels_used_for_post_selection_evaluation": True,
        "outer_target_labels_used": False,
        "formal_source_domains": list(FORMAL_SOURCE_DOMAIN_NAMES),
        "excluded_outer_target": FORMAL_OUTER_DOMAIN_NAME,
        "validation_pseudo_target": str(provenance["validation_domain"]),
        "episode_archive": str(episode_path),
        "episode_archive_sha256": episode_sha,
        "anchor_warp_package": str(anchor_path),
        "anchor_warp_package_sha256": anchor_sha,
        "anchor_warp_policy_semantic_sha256": anchor_contract[
            "policy_semantic_sha256"
        ],
        "rc_direct_checkpoint": str(direct_path),
        "rc_direct_checkpoint_sha256": direct_sha,
        "rc_direct_canonical_frozen_model_sha256": direct_checkpoint[
            "canonical_frozen_model_sha256"
        ],
        "train_archive_sha256": frozen["train_archive_sha256"],
        "seed": int(frozen["seed"]),
        "device": str(torch_device),
        "num_episodes": int(statistics.shape[0]),
        "pseudo_targets": sorted(set(row_targets)),
        "threshold_grid_size": int(thresholds.size),
        "threshold_grid_sha256": frozen["threshold_grid_sha256"],
        "threshold_grid_manifest_sha256": frozen[
            "threshold_grid_manifest_sha256"
        ],
        "feature_schema_sha256": frozen["feature_schema_sha256"],
        "threshold_grid_detector_protocol": frozen[
            "threshold_grid_detector_protocol"
        ],
        "threshold_grid_detector_checkpoint_sha256s": frozen[
            "threshold_grid_detector_checkpoint_sha256s"
        ],
        "threshold_grid_outer_detector_checkpoint_sha256": frozen[
            "threshold_grid_outer_detector_checkpoint_sha256"
        ],
        "threshold_grid_episode_detector_checkpoint_sha256s": frozen[
            "threshold_grid_episode_detector_checkpoint_sha256s"
        ],
        "preprocessing_audit": dict(anchor_inputs.preprocessing_audit),
        "action_freeze": {
            "action_digest_sha256": action_digest,
            "num_actions": len(frozen_actions),
            "frozen_before_future_e_load": True,
            "future_e_reselection_performed": False,
        },
        "monotonic_violation_rates": monotonic_violation_rates,
        "budgets": budget_results,
        "per_episode": per_episode,
        "gate": gate,
    }
    if _sha256_bytes(episode_path.read_bytes()) != episode_sha:
        raise RuntimeError("episode archive changed during two-phase evaluation")
    if _sha256_bytes(anchor_path.read_bytes()) != anchor_sha:
        raise RuntimeError("AnchorWarp package changed during evaluation")
    if _sha256_bytes(direct_path.read_bytes()) != direct_sha:
        raise RuntimeError("RC-Direct checkpoint changed during evaluation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-file", required=True)
    parser.add_argument("--anchor-warp-package", required=True)
    parser.add_argument("--rc-direct-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pixel-budgets", nargs="+", type=float, default=[1e-5, 1e-6])
    parser.add_argument("--component-budgets", nargs="+", type=float, default=[5.0, 1.0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        evaluate_anchor_warp_source_comparison(
            episode_file=args.episode_file,
            anchor_warp_package=args.anchor_warp_package,
            rc_direct_checkpoint=args.rc_direct_checkpoint,
            output=args.output,
            pixel_budgets=args.pixel_budgets,
            component_budgets=args.component_budgets,
            device=args.device,
            batch_size=args.batch_size,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ANCHOR_WARP_SOURCE_COMPARISON_SCHEMA_VERSION",
    "evaluate_anchor_warp_source_comparison",
]
