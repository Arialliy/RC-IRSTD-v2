from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from data_ext.dataset_meta import make_sample_meta
from evaluation.artifact_integrity import (
    file_sha256,
    ordered_ids_sha256,
    score_records_sha256,
)
from scripts import export_tier2s_factorized_logits as exporter


class SyntheticFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
                "inference": torch.is_inference_mode_enabled(),
                "autocast": torch.is_autocast_enabled(
                    decoder_feature.device.type
                ),
                "dtype": decoder_feature.dtype,
                "base_object_id": id(base_logits),
            }
        )
        residual = (
            contrast_feature[:, :1] * 0.25
            if contrast_enabled
            else torch.zeros_like(base_logits)
        )
        final = base_logits + residual
        gates = torch.zeros(
            base_logits.shape[0],
            3,
            *base_logits.shape[-2:],
            dtype=base_logits.dtype,
            device=base_logits.device,
        )
        return final, residual, torch.zeros_like(base_logits), gates


class SyntheticV2Model(nn.Module):
    architecture_version = exporter.ARCHITECTURE_VERSION
    use_component_context = False
    use_component_expert = False
    use_risk_gate = True

    def __init__(self, *, use_contrast: bool = True) -> None:
        super().__init__()
        self.use_contrast = use_contrast
        self.scale = nn.Parameter(torch.tensor(0.75, dtype=torch.float64))
        self.fusion_head = SyntheticFusion()
        self.forward_runtime: dict[str, Any] = {}

    def forward(self, images: torch.Tensor, *, warm_flag: bool = True):
        self.forward_runtime = {
            "training": self.training,
            "inference": torch.is_inference_mode_enabled(),
            "autocast": torch.is_autocast_enabled(images.device.type),
            "dtype": images.dtype,
            "warm_flag": warm_flag,
        }
        base = images[:, :1] * self.scale
        final, *_ = self.fusion_head(
            images[:, :1],
            images[:, 1:2],
            torch.zeros_like(images[:, :1]),
            base_logits=base,
            noise_proxy=torch.zeros_like(base),
            component_proxy=torch.zeros_like(base),
            contrast_enabled=self.use_contrast,
            component_enabled=False,
        )
        return SimpleNamespace(logits=final)


def _model_config(role: str) -> dict[str, object]:
    contrast = role == "c"
    return {
        "architecture_version": exporter.ARCHITECTURE_VERSION,
        "backend": "rc_mshnet",
        "input_channels": 3,
        "channels": [16, 32, 64, 128, 256],
        "block_counts": [2, 2, 2, 2],
        "fusion_channels": 16,
        "contrast_windows": [3, 7, 15],
        "context_dilations": [1, 2, 4],
        "component_support_window": 3,
        "component_ring_window": 9,
        "use_contrast": contrast,
        "use_component_context": False,
        "use_component_expert": False,
        "use_risk_gate": contrast,
        "expose_branch_auxiliary": False,
    }


def _checkpoint_payload(
    project: Path,
    *,
    seed: int = 43,
    role: str = "c",
    fold: str = "heldout_nudt",
) -> dict[str, Any]:
    held_out = "NUDT-SIRST" if fold == "heldout_nudt" else "IRSTD-1K"
    training = "IRSTD-1K" if fold == "heldout_nudt" else "NUDT-SIRST"
    training_root = project / "datasets" / training
    training_root.mkdir(parents=True, exist_ok=True)
    training_split = training_root / "train.txt"
    if not training_split.exists():
        training_split.write_text("sample\n", encoding="utf-8")
    training_ids = ["sample"]
    output = (
        project
        / "outputs/aaai27/detectors/component_rescue/tier2r_c_v1"
        / f"seed{seed}"
        / role
        / fold
    )
    config = {
        "seed": seed,
        "deterministic": True,
        "device": "cuda:0",
        "output_dir": str(output),
        "experiment_identity": {
            "schema_version": "rc-irstd-aaai27-tier2r-component-rescue-v1",
            "stage": "tier2r_c_source_only_confirmation",
            "run_role": f"seed{seed}_{role}_{fold}",
            "architecture_id": "rc_mshnet_v2_component_role_split",
            "outer_target": held_out,
            "target_labels_used_for_training": False,
        },
        "data": {
            "sources": [
                {
                    "name": training,
                    "path": str(project / "datasets" / training),
                }
            ],
            "train_split": "train",
            "val_split": None,
            "diagnostic_test_eval": False,
        },
        "model": _model_config(role),
        "training": {"checkpoint_selection": "fixed_last", "epochs": 80},
        "tier2r_runtime_contract": {
            "protocol_id": "tier2r_c_v1",
            "seed": seed,
            "role": role,
            "fold": fold,
            "training_source": training,
            "held_out_source_pseudo_target": held_out,
            "outer_target_dataset_loaded": False,
        },
    }
    return {
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
        "model_config": {
            **_model_config(role),
            "baseline_identity": "canonical_mshnet",
            "initialization_contract": "zero_residual_exact_mshnet",
        },
        "model_state": {"weight": torch.ones(1)},
        "source_names": [training],
        "source_split_records": [
            {
                "name": training,
                "path": str(training_root),
                "train_split_file": str(training_split),
                "train_split_file_sha256": file_sha256(training_split),
                "train_ordered_ids_sha256": ordered_ids_sha256(training_ids),
                "num_train_samples": len(training_ids),
                "train_test_id_overlap": False,
            }
        ],
        "config": config,
    }


def _write_checkpoint(
    project: Path,
    *,
    seed: int = 43,
    role: str = "c",
    fold: str = "heldout_nudt",
) -> tuple[Path, dict[str, Any]]:
    payload = _checkpoint_payload(project, seed=seed, role=role, fold=fold)
    run = Path(payload["config"]["output_dir"])
    run.mkdir(parents=True)
    checkpoint = run / "last.pt"
    torch.save(payload, checkpoint)
    formal = run / "formal_config.yaml"
    formal.write_text(
        yaml.safe_dump(payload["config"], sort_keys=False),
        encoding="utf-8",
    )
    checkpoint.chmod(0o444)
    formal.chmod(0o444)
    return checkpoint, payload


def _synthetic_governance_binding(registration_sha256: str) -> dict[str, Any]:
    artifact = {
        "path": "frozen.json",
        "sha256": "a" * 64,
        "sidecar_path": "frozen.json.sha256",
        "sidecar_sha256": "b" * 64,
    }
    return {
        "schema_version": "rc-irstd-aaai27-tier2s-governance-binding-v1",
        "registration": {
            **artifact,
            "path": "governance/GOVERNANCE_REGISTRATION.json",
            "sha256": registration_sha256,
        },
        "contract": artifact,
        "fresh_seed_ledger": artifact,
        "fresh_seed_local_scan": artifact,
        "code_sha256_canonical_sha256": "c" * 64,
        "tier2s_source_only_diagnostic_authorized": True,
        "formal_v3_model_training_authorized": False,
        "source_gate_a_authorized": False,
        "riskcurve_authorized": False,
        "outer_target_access_authorized": False,
    }


def _write_frozen_preregistration(
    project: Path, governance_binding: dict[str, Any]
) -> tuple[Path, str]:
    path = (
        project
        / exporter.TIER2S_AUDIT_RELATIVE
        / "PREREGISTRATION.json"
    )
    path.parent.mkdir(parents=True)
    payload = {
        "schema_version": exporter.PREREGISTRATION_SCHEMA,
        "protocol_id": exporter.PROTOCOL_ID,
        "research_mode": "exploratory_source_only",
        "source_only": True,
        "outer_target_access_authorized": False,
        "outer_target_images_used": False,
        "outer_target_labels_used": False,
        "source_tier3_authorized": False,
        "paper_claim_authorized": False,
        "governance_binding": governance_binding,
    }
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    digest = file_sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    path.chmod(0o444)
    sidecar.chmod(0o444)
    return path, digest


def test_production_binding_verifier_accepts_exact_frozen_chain_and_rejects_sidecar_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registration_sha = "9" * 64
    governance = _synthetic_governance_binding(registration_sha)
    preregistration, preregistration_sha = _write_frozen_preregistration(
        tmp_path, governance
    )
    calls: list[str | None] = []

    def verify(*, expected_registration_sha256: str | None = None):
        calls.append(expected_registration_sha256)
        return governance

    monkeypatch.setattr(
        exporter.governance_registrar,
        "require_frozen_tier2s_governance",
        verify,
    )
    observed_governance, observed_preregistration = (
        exporter.require_frozen_tier2s_consumer_bindings(
            expected_governance_registration_sha256=registration_sha,
            tier2s_preregistration_path=preregistration,
            expected_tier2s_preregistration_sha256=preregistration_sha,
            project_root=tmp_path,
        )
    )
    assert observed_governance == governance
    assert observed_preregistration["sha256"] == preregistration_sha
    assert observed_preregistration["path"] == str(
        exporter.TIER2S_AUDIT_RELATIVE / "PREREGISTRATION.json"
    )
    assert calls == [registration_sha]

    sidecar = preregistration.with_suffix(preregistration.suffix + ".sha256")
    sidecar.chmod(0o644)
    sidecar.write_text(f"{'0' * 64}  {preregistration.name}\n", encoding="ascii")
    sidecar.chmod(0o444)
    with pytest.raises(RuntimeError, match="sidecar drifted"):
        exporter.require_frozen_tier2s_consumer_bindings(
            expected_governance_registration_sha256=registration_sha,
            tier2s_preregistration_path=preregistration,
            expected_tier2s_preregistration_sha256=preregistration_sha,
            project_root=tmp_path,
        )


def test_production_cli_requires_governance_and_preregistration_bindings() -> None:
    required = {
        action.dest
        for action in exporter.build_parser()._actions
        if getattr(action, "required", False)
    }
    assert {
        "governance_registration_sha256",
        "tier2s_preregistration",
        "tier2s_preregistration_sha256",
    }.issubset(required)


def test_direct_fusion_hook_captures_base_and_final_in_fp32() -> None:
    model = SyntheticV2Model().train()
    images = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]], [[0.4, -0.8], [1.2, 0.0]], [[0, 0], [0, 0]]]],
        dtype=torch.float64,
    )

    result = exporter.capture_factorized_logits(model, images, expected_role="c")

    expected_base = images.float()[:, :1] * 0.75
    expected_final = expected_base + images.float()[:, 1:2] * 0.25
    expected_residual = expected_final - expected_base
    torch.testing.assert_close(result.base_logits, expected_base, rtol=0, atol=0)
    torch.testing.assert_close(result.residual_logits, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(
        result.final_logits,
        expected_final,
        rtol=0,
        atol=0,
    )
    assert result.capture_source == "fusion_head.forward_kwargs.base_logits+output[0]"
    assert result.model_output_bitwise_equal is True
    assert result.replay_max_abs_error <= 1e-7
    assert model.forward_runtime == {
        "training": False,
        "inference": True,
        "autocast": False,
        "dtype": torch.float32,
        "warm_flag": True,
    }
    assert model.fusion_head.runtime[0]["autocast"] is False
    assert model.fusion_head.runtime[0]["dtype"] == torch.float32


@pytest.mark.parametrize("role", ["control", "c"])
def test_strict_tier2r_checkpoint_binding_accepts_only_registered_roles(
    tmp_path: Path,
    role: str,
) -> None:
    (tmp_path / "datasets/NUDT-SIRST").mkdir(parents=True)
    (tmp_path / "datasets/IRSTD-1K").mkdir(parents=True)
    checkpoint, _ = _write_checkpoint(tmp_path, role=role)

    binding, loaded = exporter.load_and_validate_tier2r_checkpoint(
        checkpoint,
        project_root=tmp_path,
    )

    assert binding.role == role
    assert binding.seed == 43
    assert binding.fold == "heldout_nudt"
    assert binding.training_source == "IRSTD-1K"
    assert binding.held_out_source == "NUDT-SIRST"
    assert binding.checkpoint_sha256 == file_sha256(checkpoint)
    assert loaded["epoch"] == 79


def test_checkpoint_validation_rejects_role_architecture_and_source_drift(
    tmp_path: Path,
) -> None:
    (tmp_path / "datasets/NUDT-SIRST").mkdir(parents=True)
    (tmp_path / "datasets/IRSTD-1K").mkdir(parents=True)
    checkpoint, payload = _write_checkpoint(tmp_path)
    checkpoint.chmod(0o644)
    formal = checkpoint.parent / "formal_config.yaml"
    formal.chmod(0o644)

    payload["model_config"]["architecture_version"] = "rc-mshnet-v1"
    torch.save(payload, checkpoint)
    formal.write_text(
        yaml.safe_dump(payload["config"], sort_keys=False), encoding="utf-8"
    )
    checkpoint.chmod(0o444)
    formal.chmod(0o444)
    with pytest.raises(RuntimeError, match="architecture_version"):
        exporter.load_and_validate_tier2r_checkpoint(
            checkpoint, project_root=tmp_path
        )

    checkpoint.chmod(0o644)
    formal.chmod(0o644)
    payload = _checkpoint_payload(tmp_path)
    payload["config"]["tier2r_runtime_contract"]["role"] = "cv"
    torch.save(payload, checkpoint)
    formal.write_text(
        yaml.safe_dump(payload["config"], sort_keys=False), encoding="utf-8"
    )
    checkpoint.chmod(0o444)
    formal.chmod(0o444)
    with pytest.raises(RuntimeError, match="role"):
        exporter.load_and_validate_tier2r_checkpoint(
            checkpoint, project_root=tmp_path
        )

    checkpoint.chmod(0o644)
    formal.chmod(0o644)
    payload = _checkpoint_payload(tmp_path)
    payload["source_names"] = ["NUAA-SIRST"]
    torch.save(payload, checkpoint)
    formal.write_text(
        yaml.safe_dump(payload["config"], sort_keys=False), encoding="utf-8"
    )
    checkpoint.chmod(0o444)
    formal.chmod(0o444)
    with pytest.raises(RuntimeError, match="source"):
        exporter.load_and_validate_tier2r_checkpoint(
            checkpoint, project_root=tmp_path
        )


def test_source_path_lock_rejects_nuaa_traversal_and_symlink(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    source = root / "NUDT-SIRST"
    source.mkdir(parents=True)
    assert exporter.validate_source_dataset_root(
        source,
        dataset_name="NUDT-SIRST",
        source_root=root,
    ) == source

    with pytest.raises((ValueError, RuntimeError), match="NUAA|Unsupported"):
        exporter.validate_source_dataset_root(
            root / "NUAA-SIRST",
            dataset_name="NUAA-SIRST",
            source_root=root,
        )
    with pytest.raises(RuntimeError, match="parent traversal"):
        exporter.validate_source_dataset_root(
            root / "nested" / ".." / "NUDT-SIRST",
            dataset_name="NUDT-SIRST",
            source_root=root,
        )

    alias_root = tmp_path / "aliases"
    alias_root.mkdir()
    (alias_root / "NUDT-SIRST").symlink_to(source, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        exporter.validate_source_dataset_root(
            alias_root / "NUDT-SIRST",
            dataset_name="NUDT-SIRST",
            source_root=alias_root,
        )


def test_export_writes_factorized_integrity_chain_and_never_overwrites(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "datasets/NUDT-SIRST"
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir()
    image_path = images_dir / "sample.png"
    mask_path = masks_dir / "sample.png"
    image_path.write_bytes(b"image-path-binding")
    mask_path.write_bytes(b"mask-path-binding")
    meta = make_sample_meta(
        image_id="sample",
        dataset_name="NUDT-SIRST",
        original_hw=(2, 2),
        input_hw=(2, 2),
        valid_hw=(2, 2),
        padding_ltrb=(0, 0, 0, 0),
        spatial_mode="native",
        image_path=image_path,
        mask_path=mask_path,
        mask_alignment_applied=False,
        mask_original_hw=(2, 2),
        mask_aspect_relative_error=0.0,
        mask_alignment_policy="synthetic",
    )

    class OneSampleDataset:
        load_masks = True
        spatial_mode = "native"
        split_role = "train"
        requested_split = "train"
        split_authority_verified = True
        dataset_name = "NUDT-SIRST"
        image_ids = ["sample"]
        split_file = dataset_root / "train.txt"
        pad_multiple = 16

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            assert index == 0
            return {
                "image": torch.tensor(
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[0.4, -0.8], [1.2, 0.0]],
                        [[0.0, 0.0], [0.0, 0.0]],
                    ]
                ),
                "mask": torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]),
                "meta": meta,
            }

    OneSampleDataset.split_file.write_text("sample\n", encoding="utf-8")
    binding = exporter.Tier2SCheckpointBinding(
        checkpoint_path=tmp_path / "last.pt",
        checkpoint_sha256="a" * 64,
        formal_config_path=tmp_path / "formal_config.yaml",
        formal_config_sha256="b" * 64,
        seed=43,
        role="c",
        fold="heldout_nudt",
        training_source="IRSTD-1K",
        held_out_source="NUDT-SIRST",
        training_split_file=tmp_path / "datasets/IRSTD-1K/train.txt",
        training_split_file_sha256="c" * 64,
        training_ordered_ids_sha256="d" * 64,
        num_training_samples=1,
        epoch=79,
    )
    namespace = tmp_path / "tier2s"
    namespace.mkdir()
    output = namespace / "seed43_c_heldout_nudt_held_out"
    governance_binding = {
        "schema_version": "synthetic-governance-binding-v1",
        "registration": {"sha256": "1" * 64},
    }
    preregistration_binding = {
        "schema_version": "synthetic-preregistration-binding-v1",
        "sha256": "2" * 64,
        "governance_registration_sha256": "1" * 64,
    }

    manifest = exporter.export_factorized_dataset(
        SyntheticV2Model(),
        OneSampleDataset(),
        dataset_root=dataset_root,
        output_dir=output,
        output_namespace_root=namespace,
        binding=binding,
        subset_role="held_out",
        device="cpu",
        governance_binding=governance_binding,
        tier2s_preregistration_binding=preregistration_binding,
    )

    assert manifest["schema_version"] == exporter.SCHEMA_VERSION
    assert manifest["source_only"] is True
    assert manifest["outer_target_images_loaded"] is False
    assert manifest["outer_target_masks_loaded"] is False
    assert manifest["subset_role"] == "held_out"
    assert manifest["governance_binding"] == governance_binding
    assert (
        manifest["tier2s_preregistration_binding"]
        == preregistration_binding
    )
    assert manifest["records_sha256"] == score_records_sha256(
        manifest["records"]
    )
    assert manifest["ordered_image_ids_sha256"] == ordered_ids_sha256(
        ["sample"]
    )
    record = output / manifest["records"][0]["file"]
    assert file_sha256(record) == manifest["records"][0]["sha256"]
    with np.load(record, allow_pickle=False) as payload:
        assert {
            "base_raw_logit_float32",
            "final_raw_logit_float32",
            "residual_raw_logit_float32",
            "mask",
            "image_id",
            "original_hw",
            "valid_hw",
        }.issubset(payload.files)
        assert payload["base_raw_logit_float32"].dtype == np.float32
        assert payload["final_raw_logit_float32"].dtype == np.float32
        assert payload["residual_raw_logit_float32"].dtype == np.float32
        np.testing.assert_allclose(
            payload["base_raw_logit_float32"]
            + payload["residual_raw_logit_float32"],
            payload["final_raw_logit_float32"],
            rtol=1e-6,
            atol=1e-7,
        )
    sidecar = output / "manifest.sha256"
    assert sidecar.read_text(encoding="ascii").split() == [
        file_sha256(output / "manifest.json"),
        "manifest.json",
    ]
    record_sha = file_sha256(record)
    repeated = exporter.export_factorized_dataset(
        SyntheticV2Model(),
        OneSampleDataset(),
        dataset_root=dataset_root,
        output_dir=output,
        output_namespace_root=namespace,
        binding=binding,
        subset_role="held_out",
        device="cpu",
        governance_binding=governance_binding,
        tier2s_preregistration_binding=preregistration_binding,
    )
    assert repeated == manifest
    assert file_sha256(record) == record_sha

    manifest_path = output / "manifest.json"
    manifest_path.chmod(0o644)
    stale = json.loads(manifest_path.read_text(encoding="utf-8"))
    stale.pop("governance_binding")
    manifest_path.write_text(
        json.dumps(stale, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path.chmod(0o444)
    sidecar.chmod(0o644)
    sidecar.write_text(
        f"{file_sha256(manifest_path)}  manifest.json\n",
        encoding="ascii",
    )
    sidecar.chmod(0o444)
    with pytest.raises(RuntimeError, match="governance binding differs"):
        exporter.export_factorized_dataset(
            SyntheticV2Model(),
            OneSampleDataset(),
            dataset_root=dataset_root,
            output_dir=output,
            output_namespace_root=namespace,
            binding=binding,
            subset_role="held_out",
            device="cpu",
            governance_binding=governance_binding,
            tier2s_preregistration_binding=preregistration_binding,
        )
