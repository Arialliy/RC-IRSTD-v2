#!/usr/bin/env python3
"""Apply the RC-MSHNet overlay to a checkout of Arialliy/RC-IRSTD-v2.

The patcher is idempotent, creates one-time ``.rc_mshnet.bak`` backups for
modified tracked files, and refuses an unknown source layout instead of
silently producing a partial integration.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Callable

PATCH_ID = "rc-mshnet-aaai-sprint-v1"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_ROOT = PACKAGE_ROOT / "overlay"


class PatchError(RuntimeError):
    """Raised when the target checkout does not match the audited layout."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_text(path: Path, text: str, *, dry_run: bool) -> None:
    if dry_run:
        return
    backup = path.with_name(path.name + ".rc_mshnet.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text, encoding="utf-8")


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise PatchError(f"{label}: expected one source match, found {count}")
    return text.replace(old, new, 1)


def patch_model_builder(text: str) -> str:
    sentinel = "RC-MSHNET-PATCH: model builder"
    if sentinel in text:
        return text
    match = re.search(
        r'^(?P<indent>\s*)if backend in \{"complete_compat", "compact", "smoke"\}:',
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise PatchError("model builder: compact-backend marker was not found")
    indent = match.group("indent")
    insertion = (
        f"{indent}# {sentinel}\n"
        f"{indent}if backend in {{\"rc_mshnet\", \"rc-mshnet\", \"proposed\"}}:\n"
        f"{indent}    from .rc_mshnet import build_rc_mshnet\n\n"
        f"{indent}    return build_rc_mshnet(config)\n"
    )
    return text[: match.start()] + insertion + text[match.start() :]


def patch_models_facade(text: str) -> str:
    sentinel = "RC-MSHNET-PATCH: facade exports"
    if sentinel not in text:
        marker = "from .system import CalibratedPrediction, RCIRSTDSystem"
        if marker not in text:
            raise PatchError("models facade: system import marker was not found")
        addition = (
            f"# {sentinel}\n"
            "from .rc_mshnet import (\n"
            "    RCMSHNet,\n"
            "    RC_MSHNET_ARCHITECTURE_VERSION,\n"
            "    build_rc_mshnet,\n"
            "    initialize_rc_mshnet_from_checkpoint,\n"
            ")\n"
        )
        text = text.replace(marker, addition + marker, 1)
    names = (
        "RCMSHNet",
        "RC_MSHNET_ARCHITECTURE_VERSION",
        "build_rc_mshnet",
        "initialize_rc_mshnet_from_checkpoint",
    )
    list_match = re.search(r"__all__\s*=\s*\[", text)
    if list_match is None:
        raise PatchError("models facade: __all__ list was not found")
    insertion_point = list_match.end()
    additions = ""
    for name in names:
        if f'"{name}"' not in text:
            additions += f'\n    "{name}",'
    if additions:
        text = text[:insertion_point] + additions + text[insertion_point:]
    return text


def patch_optimizer(text: str) -> str:
    sentinel = "RC-MSHNET-PATCH: differential learning rates"
    if sentinel in text:
        return text
    start = text.find("def _build_optimizer(")
    end = text.find("\ndef _average_metrics", start)
    if start < 0 or end < 0:
        raise PatchError("trainer: optimizer function was not found")
    replacement = '''def _build_optimizer(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    # RC-MSHNET-PATCH: differential learning rates
    name = str(config.get("name", "adamw")).lower()
    learning_rate = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))
    backbone_lr_scale = float(config.get("backbone_lr_scale", 1.0))
    if learning_rate <= 0.0:
        raise ValueError("optimizer.lr must be positive")
    if not 0.0 < backbone_lr_scale <= 1.0:
        raise ValueError("optimizer.backbone_lr_scale must be in (0, 1]")

    parameters: object = model.parameters()
    extension_prefixes = tuple(getattr(model, "extension_prefixes", ()))
    if backbone_lr_scale != 1.0:
        if not extension_prefixes:
            raise ValueError(
                "optimizer.backbone_lr_scale is supported only by a model that "
                "declares extension_prefixes"
            )
        extension_parameters: list[torch.nn.Parameter] = []
        backbone_parameters: list[torch.nn.Parameter] = []
        for parameter_name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            destination = (
                extension_parameters
                if any(parameter_name.startswith(prefix) for prefix in extension_prefixes)
                else backbone_parameters
            )
            destination.append(parameter)
        if not extension_parameters or not backbone_parameters:
            raise ValueError("Could not partition RC-MSHNet backbone and extensions")
        # The existing logger reports group 0 as the headline LR.
        parameters = [
            {
                "params": extension_parameters,
                "lr": learning_rate,
                "group_name": "rc_mshnet_extensions",
            },
            {
                "params": backbone_parameters,
                "lr": learning_rate * backbone_lr_scale,
                "group_name": "mshnet_backbone",
            },
        ]

    if name == "adamw":
        return AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(
            parameters,
            lr=learning_rate,
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(config.get("nesterov", True)),
        )
    raise ValueError(f"Unsupported optimizer: {name}")
'''
    return text[:start] + replacement + text[end:]


def patch_detector_trainer(text: str) -> str:
    text = patch_optimizer(text)

    if 'not in {"canonical", "rc_mshnet"}' not in text:
        old = 'str(self.model_config.get("backend", "canonical")) != "canonical"'
        new = (
            'str(self.model_config.get("backend", "canonical")) '
            'not in {"canonical", "rc_mshnet"}'
        )
        text = _replace_once(text, old, new, label="trainer formal backend")
        text = text.replace(
            '"Non-canonical detector backends require diagnostic_only=true; "\n'
            '                "paper runs must use model.backend=canonical"',
            '"Unregistered detector backends require diagnostic_only=true; "\n'
            '                "formal runs permit canonical or rc_mshnet"',
            1,
        )

    init_sentinel = "RC-MSHNET-PATCH: pretrained initialization"
    if init_sentinel not in text:
        marker = '        training_config = config.get("training", {})\n'
        if marker not in text:
            raise PatchError("trainer: training_config marker was not found")
        block = marker + (
            f"        # {init_sentinel}\n"
            "        self.initialization_report: dict[str, Any] | None = None\n"
            "        initialize_from = training_config.get(\"initialize_from\")\n"
            "        if initialize_from and training_config.get(\"resume\"):\n"
            "            raise ValueError(\n"
            "                \"training.initialize_from and training.resume are mutually \"\n"
            "                \"exclusive; resume already contains initialized weights\"\n"
            "            )\n"
            "        if initialize_from:\n"
            "            from rc_irstd.models.rc_mshnet import (\n"
            "                initialize_rc_mshnet_from_checkpoint,\n"
            "            )\n\n"
            "            initialization_path = resolve_config_path(config, initialize_from)\n"
            "            self.initialization_report = (\n"
            "                initialize_rc_mshnet_from_checkpoint(\n"
            "                    self.model, initialization_path, device=self.device\n"
            "                )\n"
            "            )\n"
            "            atomic_write_json(\n"
            "                self.output_dir / \"initialization_report.json\",\n"
            "                self.initialization_report,\n"
            "            )\n"
            "            self.logger.info(\n"
            "                \"Initialized RC-MSHNet from canonical MSHNet %s\",\n"
            "                initialization_path,\n"
            "            )\n"
        )
        text = text.replace(marker, block, 1)

    payload_sentinel = "RC-MSHNET-PATCH: initialization provenance"
    if payload_sentinel not in text:
        marker = '            "model_config": self.model_config,\n'
        if marker not in text:
            raise PatchError("trainer: checkpoint model_config marker was not found")
        text = text.replace(
            marker,
            marker
            + f"            # {payload_sentinel}\n"
            + '            "initialization": self.initialization_report,\n',
            1,
        )

    resume_sentinel = "RC-MSHNET-PATCH: preserve initialization on resume"
    if resume_sentinel not in text:
        marker = '        self.model.load_state_dict(checkpoint["model_state"])\n'
        if marker not in text:
            raise PatchError("trainer: resume model-state marker was not found")
        text = text.replace(
            marker,
            marker
            + f"        # {resume_sentinel}\n"
            + '        self.initialization_report = checkpoint.get("initialization")\n',
            1,
        )
    return text


def patch_detector_objective(text: str) -> str:
    sentinel = "RC-MSHNET-PATCH: SLS-only fast path"
    if sentinel in text:
        return text
    marker = "        if self.tail_mode == RAW_LOGIT_TAILRANK_MODE:\n"
    if marker not in text:
        raise PatchError("detector objective: tail-mode marker was not found")
    block = (
        f"        # {sentinel}\n"
        "        if (\n"
        "            self.lambda_tail == 0.0\n"
        "            and self.lambda_miss == 0.0\n"
        "            and self.lambda_margin == 0.0\n"
        "        ):\n"
        "            total = base.total + self.auxiliary_weight * auxiliary_total\n"
        "            metrics = {\n"
        "                \"loss_total\": float(total.detach().cpu()),\n"
        "                \"loss_sls\": float(base.total.detach().cpu()),\n"
        "                \"loss_bce\": float(base.bce.detach().cpu()),\n"
        "                \"loss_scale_iou\": float(base.scale_iou.detach().cpu()),\n"
        "                \"loss_location\": float(base.location.detach().cpu()),\n"
        "                \"loss_auxiliary\": float(auxiliary_total.detach().cpu()),\n"
        "                \"loss_tail\": 0.0,\n"
        "                \"loss_miss\": 0.0,\n"
        "                \"loss_margin\": 0.0,\n"
        "                \"num_background_peaks\": 0.0,\n"
        "                \"num_target_objects\": 0.0,\n"
        "            }\n"
        "            return DetectorLossOutput(total=total, metrics=metrics)\n\n"
    )
    return text.replace(marker, block + marker, 1)


def patch_logit_grid(text: str) -> str:
    sentinel = "RC-MSHNET-PATCH: formal detector backends"
    if sentinel not in text:
        exact_line = '        "model_backend": "canonical",\n'
        if exact_line not in text:
            raise PatchError("logit grid: canonical model_backend field was not found")
        text = text.replace(exact_line, "", 1)
        marker = '    for field in ("diagnostic_only", "non_strict_state_loading"):\n'
        if marker not in text:
            raise PatchError("logit grid: boolean-field marker was not found")
        block = (
            f"    # {sentinel}\n"
            "    model_backend = manifest.get(\"model_backend\")\n"
            "    if model_backend not in {\"canonical\", \"rc_mshnet\"}:\n"
            "        raise ValueError(\n"
            "            \"Grid source manifest model_backend must be canonical or \"\n"
            "            \"rc_mshnet\"\n"
            "        )\n"
        )
        text = text.replace(marker, block + marker, 1)

    consistency_sentinel = "RC-MSHNET-PATCH: backend consistency"
    if consistency_sentinel not in text:
        start = text.find("    loaded.sort(\n")
        next_marker = "    input_pairs = [\n"
        next_index = text.find(next_marker, start)
        if start < 0 or next_index < 0:
            raise PatchError("logit grid: source-input ordering markers were not found")
        block = (
            f"    # {consistency_sentinel}\n"
            "    model_backends = {\n"
            "        str(item.manifest.get(\"model_backend\")) for item in loaded\n"
            "    }\n"
            "    if len(model_backends) != 1:\n"
            "        raise ValueError(\n"
            "            \"All source-only grid artifacts must use one detector backend\"\n"
            "        )\n"
            "    model_backend = next(iter(model_backends))\n"
        )
        text = text[:next_index] + block + text[next_index:]

    manifest_sentinel = "RC-MSHNET-PATCH: record grid backend"
    if manifest_sentinel not in text:
        marker = '        "grid_detector_protocol": GRID_DETECTOR_PROTOCOL,\n'
        if marker not in text:
            raise PatchError("logit grid: manifest protocol marker was not found")
        text = text.replace(
            marker,
            marker
            + f"        # {manifest_sentinel}\n"
            + '        "model_backend": model_backend,\n',
            1,
        )
    return text


PATCHERS: dict[str, Callable[[str], str]] = {
    "rc_irstd/models/mshnet.py": patch_model_builder,
    "rc_irstd/models/__init__.py": patch_models_facade,
    "rc_irstd/training/detector_trainer.py": patch_detector_trainer,
    "rc_irstd/losses/detector.py": patch_detector_objective,
    "risk_curve/build_logit_threshold_grid.py": patch_logit_grid,
}


def _copy_overlay(repo: Path, *, dry_run: bool, force: bool) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    sources = (
        path
        for path in OVERLAY_ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    for source in sorted(sources):
        relative = source.relative_to(OVERLAY_ROOT)
        destination = repo / relative
        same = destination.exists() and _sha256(destination) == _sha256(source)
        if destination.exists() and not same and not force:
            raise PatchError(
                f"overlay target exists with different content: {relative}; use --force"
            )
        records.append(
            {
                "path": str(relative),
                "source_sha256": _sha256(source),
                "action": "unchanged" if same else "copy",
            }
        )
        if not dry_run and not same:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                backup = destination.with_name(destination.name + ".rc_mshnet.bak")
                if not backup.exists():
                    shutil.copy2(destination, backup)
            shutil.copy2(source, destination)
    return records


def apply(repo: Path, *, dry_run: bool, force: bool) -> dict[str, object]:
    repo = repo.expanduser().resolve()
    required = [repo / "model/MSHNet.py", repo / "rc_irstd/models/mshnet.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise PatchError("not an RC-IRSTD-v2 checkout; missing: " + ", ".join(missing))

    changes: list[dict[str, object]] = []
    for relative, patcher in PATCHERS.items():
        path = repo / relative
        if not path.is_file():
            raise PatchError(f"required patch target is missing: {relative}")
        before = path.read_text(encoding="utf-8")
        after = patcher(before)
        if before != after:
            preview = "".join(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    fromfile=f"a/{relative}",
                    tofile=f"b/{relative}",
                    n=2,
                )
            )
            changes.append(
                {
                    "path": relative,
                    "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
                    "after_sha256": hashlib.sha256(after.encode()).hexdigest(),
                    "diff_preview": preview[:16000],
                }
            )
            _write_text(path, after, dry_run=dry_run)
        else:
            changes.append({"path": relative, "action": "already_patched"})

    overlay = _copy_overlay(repo, dry_run=dry_run, force=force)
    report: dict[str, object] = {
        "patch_id": PATCH_ID,
        "repo": str(repo),
        "dry_run": dry_run,
        "modified_files": changes,
        "overlay_files": overlay,
    }
    if not dry_run:
        report_path = repo / "artifacts/rc_mshnet_patch_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = apply(args.repo, dry_run=args.dry_run, force=args.force)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
