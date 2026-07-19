from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import scripts.diagnose_component_fusion as diagnosis


class SyntheticGate(nn.Module):
    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        contrast = evidence[:, 0:1] + 0.5 * evidence[:, 1:2]
        component = evidence[:, 1:2] - 0.25 * evidence[:, 2:3]
        noop = -evidence[:, 2:3] + 0.2 * evidence[:, 1:2]
        return torch.cat((contrast, component, noop), dim=1)


class SyntheticFusion(nn.Module):
    def __init__(self, *, corrupt_gate: bool = False) -> None:
        super().__init__()
        self.gate = SyntheticGate()
        self.raw_residual_gain = nn.Parameter(torch.tensor(0.0))
        self.corrupt_gate = corrupt_gate
        self.runtime: list[dict[str, Any]] = []

    def forward(
        self,
        decoder_feature: torch.Tensor,
        contrast_feature: torch.Tensor,
        component_feature: torch.Tensor,
        *,
        base_logits: torch.Tensor,
        noise_proxy: torch.Tensor,
        component_proxy: torch.Tensor,
        contrast_enabled: bool,
        component_enabled: bool,
    ):
        self.runtime.append(
            {
                "training": self.training,
                "inference": torch.is_inference_mode_enabled(),
                "autocast": torch.is_autocast_enabled(
                    decoder_feature.device.type
                ),
                "dtype": decoder_feature.dtype,
                "component_enabled": component_enabled,
                "component_feature": component_feature.detach().clone(),
                "component_proxy": component_proxy.detach().clone(),
            }
        )
        contrast_delta = (
            0.5 * contrast_feature[:, :1] + 0.25
            if contrast_enabled
            else torch.zeros_like(base_logits)
        )
        component_delta = (
            -0.75 * component_feature[:, :1] + 0.1
            if component_enabled
            else torch.zeros_like(base_logits)
        )
        disagreement = (contrast_delta - component_delta).abs()
        evidence = torch.cat(
            (
                contrast_feature[:, :1],
                component_feature[:, :1] + component_proxy,
                disagreement + noise_proxy,
            ),
            dim=1,
        )
        gate_logits = self.gate(evidence)
        masks = []
        for enabled in (contrast_enabled, component_enabled, True):
            masks.append(
                torch.full_like(
                    gate_logits[:, :1],
                    0.0 if enabled else -1.0e4,
                )
            )
        gates = torch.softmax(gate_logits + torch.cat(masks, dim=1), dim=1)
        if self.corrupt_gate:
            gates = gates * 0.5
        gain = 2.0 * torch.sigmoid(self.raw_residual_gain)
        fused = base_logits + gain * (
            gates[:, 0:1] * contrast_delta
            + gates[:, 1:2] * component_delta
        )
        return fused, contrast_delta, component_delta, gates


class SyntheticFullModel(nn.Module):
    architecture_version = diagnosis.ARCHITECTURE_VERSION
    use_contrast = True
    use_component_context = True
    use_component_expert = True
    use_risk_gate = True

    def __init__(
        self,
        *,
        corrupt_gate: bool = False,
        raise_before_fusion: bool = False,
    ) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.75, dtype=torch.float64))
        self.fusion_head = SyntheticFusion(corrupt_gate=corrupt_gate)
        self.raise_before_fusion = raise_before_fusion
        self.forward_runtime: dict[str, Any] = {}

    def forward(
        self,
        images: torch.Tensor,
        *,
        warm_flag: bool = True,
    ):
        self.forward_runtime = {
            "training": self.training,
            "inference": torch.is_inference_mode_enabled(),
            "autocast": torch.is_autocast_enabled(images.device.type),
            "dtype": images.dtype,
            "warm_flag": warm_flag,
        }
        if self.raise_before_fusion:
            raise RuntimeError("synthetic forward failure")
        base = images[:, 0:1] * self.scale
        decoder = images[:, 0:1]
        contrast = images[:, 1:2]
        component = images[:, 2:3] + 0.4
        noise = torch.full_like(base, 0.05)
        component_proxy = component * 0.2
        fused, contrast_delta, component_delta, gates = self.fusion_head(
            decoder,
            contrast,
            component,
            base_logits=base,
            noise_proxy=noise,
            component_proxy=component_proxy,
            contrast_enabled=True,
            component_enabled=True,
        )
        return SimpleNamespace(logits=fused)


def _images() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [[1.0, -0.5], [0.25, 2.0]],
                [[0.2, 0.8], [-0.4, 1.5]],
                [[0.7, -0.2], [1.0, 0.1]],
            ]
        ],
        dtype=torch.float64,
    )


def test_corrected_counterfactual_math_and_names() -> None:
    model = SyntheticFullModel().train()
    result = diagnosis.diagnose_frozen_full_batch(model, _images())

    assert tuple(result.raw_logits) == diagnosis.BATCH_RAW_LOGIT_KEYS
    assert (
        result.base_logits_capture_source
        == "fusion_head.forward_kwargs.base_logits"
    )
    expected_base = _images().float()[:, 0:1] * 0.75
    torch.testing.assert_close(
        result.base_raw_logits,
        expected_base,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.raw_logits["frozen_full"],
        result.raw_logits["replayed_full"],
        rtol=0.0,
        atol=0.0,
    )

    gain = result.residual_gain
    expected_fixed = result.base_raw_logits + gain * (
        result.full_gates[:, 0:1] * result.contrast_delta
    )
    torch.testing.assert_close(
        result.raw_logits["drop_component_action_fixed_full_gate"],
        expected_fixed,
    )
    masked = result.full_gate_logits.clone()
    masked[:, 1:2] = -torch.inf
    expected_conditional_gates = torch.softmax(masked, dim=1)
    expected_conditional = result.base_raw_logits + gain * (
        expected_conditional_gates[:, 0:1] * result.contrast_delta
    )
    torch.testing.assert_close(
        result.conditional_full_gates,
        expected_conditional_gates,
    )
    torch.testing.assert_close(
        result.raw_logits["drop_component_action_conditional_full_gate"],
        expected_conditional,
    )
    assert torch.count_nonzero(
        result.context_preserved_expert_off_gates[:, 1:2]
    ) == 0
    assert torch.count_nonzero(
        result.context_zero_expert_off_gates[:, 1:2]
    ) == 0
    assert not torch.equal(
        result.raw_logits[
            "context_preserved_component_expert_off_reforward"
        ],
        result.raw_logits["context_zero_component_expert_off_reforward"],
    )


def test_eval_fp32_inference_no_autocast_and_model_is_frozen() -> None:
    model = SyntheticFullModel().train()
    assert any(parameter.requires_grad for parameter in model.parameters())

    diagnosis.diagnose_frozen_full_batch(model, _images())

    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(
        not parameter.dtype.is_floating_point
        or parameter.dtype == torch.float32
        for parameter in model.parameters()
    )
    assert model.forward_runtime == {
        "training": False,
        "inference": True,
        "autocast": False,
        "dtype": torch.float32,
        "warm_flag": True,
    }
    assert len(model.fusion_head.runtime) == 4
    assert all(record["training"] is False for record in model.fusion_head.runtime)
    assert all(record["inference"] is True for record in model.fusion_head.runtime)
    assert all(record["autocast"] is False for record in model.fusion_head.runtime)
    assert all(
        record["dtype"] == torch.float32
        for record in model.fusion_head.runtime
    )
    assert [
        record["component_enabled"]
        for record in model.fusion_head.runtime
    ] == [True, True, False, False]
    assert torch.count_nonzero(
        model.fusion_head.runtime[-2]["component_feature"]
    ) > 0
    assert torch.count_nonzero(
        model.fusion_head.runtime[-1]["component_feature"]
    ) == 0
    assert torch.count_nonzero(
        model.fusion_head.runtime[-1]["component_proxy"]
    ) == 0


def test_hooks_are_removed_when_model_forward_raises() -> None:
    model = SyntheticFullModel(raise_before_fusion=True)
    pre_count = len(model.fusion_head._forward_pre_hooks)
    fusion_count = len(model.fusion_head._forward_hooks)
    gate_count = len(model.fusion_head.gate._forward_hooks)

    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        diagnosis.diagnose_frozen_full_batch(model, _images())

    assert len(model.fusion_head._forward_pre_hooks) == pre_count
    assert len(model.fusion_head._forward_hooks) == fusion_count
    assert len(model.fusion_head.gate._forward_hooks) == gate_count


def test_invalid_gate_fails_closed() -> None:
    model = SyntheticFullModel(corrupt_gate=True)
    with pytest.raises(RuntimeError, match="does not sum to one"):
        diagnosis.diagnose_frozen_full_batch(model, _images())


def test_replay_consistency_interface_rejects_drift() -> None:
    frozen = torch.zeros(1, 1, 2, 2)
    replayed = frozen.clone()
    replayed[0, 0, 0, 0] = 0.1
    with pytest.raises(RuntimeError, match="replay is inconsistent"):
        diagnosis.assert_full_raw_logit_replay_consistency(
            frozen,
            replayed,
            rtol=0.0,
            atol=0.0,
        )


def test_source_lock_rejects_outer_target_and_symlink_alias(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    root.mkdir()
    source = root / "NUDT-SIRST"
    source.mkdir()
    assert diagnosis.validate_source_dataset_path(
        source,
        expected_dataset="NUDT-SIRST",
        source_root=root,
    ) == source

    with pytest.raises(RuntimeError):
        diagnosis.validate_source_dataset_path(
            root / "NUAA-SIRST",
            expected_dataset="NUDT-SIRST",
            source_root=root,
        )

    alias_root = tmp_path / "alias-datasets"
    alias_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (alias_root / "NUDT-SIRST").symlink_to(
        outside,
        target_is_directory=True,
    )
    with pytest.raises(RuntimeError, match="symlink"):
        diagnosis.validate_source_dataset_path(
            alias_root / "NUDT-SIRST",
            expected_dataset="NUDT-SIRST",
            source_root=alias_root,
        )


def _freeze(path: Path) -> None:
    path.chmod(0o444)


def _make_fold(project: Path, fold: str, checkpoint_bytes: bytes) -> Path:
    contract = diagnosis.FROZEN_FULL_FOLDS[fold]
    run = (
        project
        / "outputs/aaai27/detectors/source_lodo_gate/seed42/full"
        / fold
    )
    run.mkdir(parents=True)
    checkpoint = run / "last.pt"
    checkpoint.write_bytes(checkpoint_bytes)
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    sidecar = run / "checkpoint.sha256"
    sidecar.write_text(f"{digest}  last.pt\n", encoding="ascii")
    identity = run / "PHASE3_IDENTITY.json"
    identity.write_text(
        json.dumps(
            {
                "run_id": f"full_{fold}",
                "role": "full",
                "held_out_pseudo_target": contract["held_out_source"],
                "training_source": contract["training_source"],
                "outer_target_labels_used": False,
                "checkpoint_selection": "fixed_last",
                "checkpoint_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    for path in (checkpoint, sidecar, identity):
        _freeze(path)
    return checkpoint


def test_two_fold_frozen_checkpoint_contract(tmp_path: Path) -> None:
    checkpoints = {
        "heldout_nudt": _make_fold(tmp_path, "heldout_nudt", b"nudt"),
        "heldout_irstd": _make_fold(tmp_path, "heldout_irstd", b"irstd"),
    }
    bindings = diagnosis.validate_two_fold_checkpoint_contract(
        checkpoints,
        project_root=tmp_path,
    )
    assert set(bindings) == set(diagnosis.FROZEN_FULL_FOLDS)
    assert len({binding.checkpoint_sha256 for binding in bindings.values()}) == 2

    with pytest.raises(RuntimeError, match="Exactly"):
        diagnosis.validate_two_fold_checkpoint_contract(
            {"heldout_nudt": checkpoints["heldout_nudt"]},
            project_root=tmp_path,
        )


def test_embedded_model_config_boolean_identity_is_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = diagnosis.FrozenCheckpointBinding(
        fold="heldout_nudt",
        checkpoint_path=tmp_path / "last.pt",
        checkpoint_sha256="a" * 64,
        identity_path=tmp_path / "identity.json",
        identity_sha256="b" * 64,
        held_out_source="NUDT-SIRST",
        training_source="IRSTD-1K",
    )
    payload = {
        "kind": "detector",
        "checkpoint_selection": "fixed_last",
        "selection_rule": "fixed_last",
        "test_labels_used_for_selection": False,
        "diagnostic_test_eval": False,
        "diagnostic_only": False,
        "formal_paper_checkpoint": True,
        "warm_flag": True,
        "inference_head": "multi_scale_fused",
        "source_names": ["IRSTD-1K"],
        "model_config": {
            "architecture_version": diagnosis.ARCHITECTURE_VERSION,
            "backend": "rc_mshnet",
            "use_contrast": True,
            "use_component_context": 1,
            "use_risk_gate": True,
        },
        "model_state": {"weight": torch.ones(1)},
    }
    monkeypatch.setattr(
        diagnosis,
        "_load_checkpoint_safely",
        lambda _path: payload,
    )
    with pytest.raises(RuntimeError, match="use_component_context"):
        diagnosis.load_frozen_full_model(binding, device="cpu")

    payload["model_config"]["use_component_context"] = True
    payload["model_config"]["use_component_expert"] = True
    with pytest.raises(RuntimeError, match="must not contain"):
        diagnosis.load_frozen_full_model(binding, device="cpu")


def test_historical_full_replay_is_bitwise_strict() -> None:
    historical = np.asarray([[1.0, -2.0]], dtype=np.float32)
    assert diagnosis.assert_historical_full_raw_logit_consistency(
        historical.copy(),
        historical,
        image_id="sample",
    )["bitwise_equal"] is True
    changed = historical.copy()
    changed[0, 0] = np.nextafter(
        changed[0, 0],
        np.float32(2.0),
    )
    with pytest.raises(RuntimeError, match="frozen artifact"):
        diagnosis.assert_historical_full_raw_logit_consistency(
            changed,
            historical,
            image_id="sample",
        )


def test_fold_result_closes_comparator_and_variant_stream_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = _images()[0].float()
    mask = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float32)
    reference_model = SyntheticFullModel()
    reference = diagnosis.diagnose_frozen_full_batch(
        reference_model,
        image.unsqueeze(0),
    )
    full_logits = (
        reference.raw_logits["replayed_full"][0, 0].cpu().numpy().copy()
    )
    no_component_logits = np.asarray(
        [[-0.1, 0.2], [0.3, -0.4]],
        dtype=np.float32,
    )
    mask_array = mask[0].numpy().astype(bool)
    full_sample = diagnosis.RawLogitSample(
        image_id="sample",
        logits=full_logits,
        probability=torch.sigmoid(
            torch.from_numpy(full_logits)
        ).numpy(),
        mask=mask_array,
    )
    no_component_sample = diagnosis.RawLogitSample(
        image_id="sample",
        logits=no_component_logits,
        probability=torch.sigmoid(
            torch.from_numpy(no_component_logits)
        ).numpy(),
        mask=mask_array,
    )
    full_record_path = tmp_path / "full.npz"
    no_component_record_path = tmp_path / "no_component.npz"
    np.savez_compressed(
        full_record_path,
        image_id=np.asarray("sample"),
        logit=full_logits,
        mask=mask_array,
    )
    np.savez_compressed(
        no_component_record_path,
        image_id=np.asarray("sample"),
        logit=no_component_logits,
        mask=mask_array,
    )
    comparators = diagnosis.FrozenFoldComparators(
        full_records=(
            diagnosis.FrozenRawLogitRecord(
                image_id="sample",
                path=full_record_path,
            ),
        ),
        independently_trained_no_component_records=(
            diagnosis.FrozenRawLogitRecord(
                image_id="sample",
                path=no_component_record_path,
            ),
        ),
        evidence={
            "frozen_full": {
                "raw_logit_stream_sha256": (
                    diagnosis.raw_logit_stream_sha256([full_sample])
                )
            },
            "independently_trained_v1_no_component": {
                "raw_logit_stream_sha256": (
                    diagnosis.raw_logit_stream_sha256(
                        [no_component_sample]
                    )
                )
            },
            "ordered_streams_aligned": True,
        },
    )
    meta = {
        "image_id": "sample",
        "dataset_name": "NUDT-SIRST",
        "original_hw": torch.tensor([2, 2]),
        "input_hw": torch.tensor([2, 2]),
        "valid_hw": torch.tensor([2, 2]),
        "padding_ltrb": torch.tensor([0, 0, 0, 0]),
        "spatial_mode": "native",
        "image_path": str(tmp_path / "image.png"),
        "mask_path": str(tmp_path / "mask.png"),
        "mask_alignment_applied": False,
        "mask_original_hw": torch.tensor([2, 2]),
        "mask_aspect_relative_error": 0.0,
        "mask_alignment_policy": "synthetic",
    }

    class OneSampleDataset:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            assert index == 0
            return {
                "image": image,
                "mask": mask,
                "meta": meta,
            }

    binding = diagnosis.FrozenCheckpointBinding(
        fold="heldout_nudt",
        checkpoint_path=tmp_path / "last.pt",
        checkpoint_sha256="a" * 64,
        identity_path=tmp_path / "identity.json",
        identity_sha256="b" * 64,
        held_out_source="NUDT-SIRST",
        training_source="IRSTD-1K",
    )
    monkeypatch.setattr(
        diagnosis,
        "validate_source_dataset_path",
        lambda *_args, **_kwargs: tmp_path,
    )
    monkeypatch.setattr(
        diagnosis,
        "_validate_source_record_path",
        lambda path, **_kwargs: Path(path),
    )
    monkeypatch.setattr(
        diagnosis,
        "IRSTDEvalDataset",
        lambda *_args, **_kwargs: OneSampleDataset(),
    )
    monkeypatch.setattr(
        diagnosis,
        "load_frozen_fold_comparators",
        lambda _binding: comparators,
    )
    monkeypatch.setattr(
        diagnosis,
        "load_frozen_full_model",
        lambda _binding, *, device: SyntheticFullModel(),
    )
    output = tmp_path / "output"
    output.mkdir()
    result = diagnosis._run_fold(
        binding,
        source_root=tmp_path,
        output_root=output,
        device="cpu",
    )

    assert result["all_historical_full_bitwise_equal"] is True
    assert result["exact_state_evaluator_input_ready"] is True
    assert result["frozen_comparator_evidence"] == comparators.evidence
    assert set(result["raw_logit_stream_sha256_by_variant"]) == set(
        diagnosis.BATCH_RAW_LOGIT_KEYS
    ) | {"independently_trained_v1_no_component"}
    assert len(result["ordered_image_ids_sha256"]) == 64
    assert len(result["records_sha256"]) == 64
    record = output / result["records"][0]["file"]
    with np.load(record, allow_pickle=False) as payload:
        np.testing.assert_array_equal(
            payload[
                "raw_logits__independently_trained_v1_no_component"
            ],
            no_component_logits,
        )


def test_strict_variant_adapter_feeds_exact_state_enumerator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation.raw_logit_source_operating_point import (
        enumerate_exact_shared_states,
    )

    monkeypatch.setattr(diagnosis, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "diagnostic"
    folds: dict[str, dict[str, Any]] = {}
    record_paths: list[Path] = []
    for fold_index, (fold, contract) in enumerate(
        diagnosis.FROZEN_FULL_FOLDS.items()
    ):
        image_id = f"sample-{fold_index}"
        mask = np.asarray([[0, 1], [0, 0]], dtype=np.uint8)
        base = np.asarray(
            [[-2.0, 3.0], [-1.0, 0.5]],
            dtype=np.float32,
        ) + np.float32(fold_index)
        arrays: dict[str, np.ndarray] = {
            "mask": mask,
            "image_id": np.asarray(image_id),
            "dataset_name": np.asarray(contract["held_out_source"]),
            "score_representation": np.asarray(
                diagnosis.SCORE_REPRESENTATION
            ),
            "inference_autocast_enabled": np.asarray(False),
        }
        stream_hashes: dict[str, str] = {}
        for variant_index, variant in enumerate(
            diagnosis.COUNTERFACTUAL_VARIANTS
        ):
            logits = base + np.float32(variant_index * 0.01)
            arrays[f"raw_logits__{variant}"] = logits
            raw_sample = diagnosis.RawLogitSample(
                image_id=image_id,
                logits=logits,
                probability=torch.sigmoid(
                    torch.from_numpy(logits)
                ).numpy(),
                mask=mask.astype(bool),
            )
            stream_hashes[variant] = (
                diagnosis.raw_logit_stream_sha256([raw_sample])
            )
        record = output / fold / "records" / f"{image_id}.npz"
        record.parent.mkdir(parents=True)
        np.savez_compressed(record, **arrays)
        record.chmod(0o444)
        record_paths.append(record)
        records = [
            {
                "image_id": image_id,
                "file": str(record.relative_to(output)),
                "sha256": diagnosis._sha256(record),
            }
        ]
        folds[fold] = {
            "fold": fold,
            "held_out_source": contract["held_out_source"],
            "training_source": contract["training_source"],
            "num_records": 1,
            "ordered_image_ids_sha256": diagnosis.ordered_ids_sha256(
                [image_id]
            ),
            "records_sha256": diagnosis.score_records_sha256(records),
            "raw_logit_stream_sha256_by_variant": stream_hashes,
            "all_historical_full_bitwise_equal": True,
            "exact_state_evaluator_input_ready": True,
            "records": records,
        }
    manifest = {
        "schema_version": diagnosis.SCHEMA_VERSION,
        "diagnostic_only": True,
        "authorizes_go": False,
        "source_only": True,
        "outer_target_images_loaded": False,
        "outer_target_masks_loaded": False,
        "outer_target_access_authorized": False,
        "inference_dtype": "float32",
        "inference_autocast_enabled": False,
        "two_source_domains_closed": True,
        "exact_state_evaluator_input_ready": True,
        "dense_grid_is_conclusive": False,
        "counterfactual_variants": list(
            diagnosis.COUNTERFACTUAL_VARIANTS
        ),
        "folds": folds,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    manifest_path.chmod(0o444)
    sidecar = output / "manifest.sha256"
    sidecar.write_text(
        f"{diagnosis._sha256(manifest_path)}  manifest.json\n",
        encoding="ascii",
    )
    sidecar.chmod(0o444)

    selected_variant = diagnosis.COUNTERFACTUAL_VARIANTS[0]
    samples = diagnosis.load_variant_raw_logit_samples(
        output,
        selected_variant,
    )
    assert set(samples) == {"NUDT-SIRST", "IRSTD-1K"}
    exact = enumerate_exact_shared_states(
        samples,
        loose_pixel_budget=0.5,
    )
    assert exact["exact_state_enumeration"] is True
    assert exact["states"][0]["all_reject_sentinel"] is True

    record_paths[0].chmod(0o644)
    with pytest.raises(RuntimeError, match="not frozen read-only"):
        diagnosis.load_variant_raw_logit_samples(
            output,
            selected_variant,
        )


def test_physical_gpu_binding_is_fail_closed() -> None:
    binding = diagnosis.validate_physical_gpu_binding(
        {"CUDA_VISIBLE_DEVICES": "2,3"}
    )
    assert binding["physical_to_logical"] == {"2": 0, "3": 1}
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES=2,3"):
        diagnosis.validate_physical_gpu_binding(
            {"CUDA_VISIBLE_DEVICES": "0,1"}
        )
