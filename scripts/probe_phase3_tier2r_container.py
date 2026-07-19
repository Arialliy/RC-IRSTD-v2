#!/usr/bin/env python3
"""Emit source-only runtime, GPU, and mount evidence from inside Docker."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess

import numpy
import pandas
import PIL
import scipy
import skimage
import torch
import torchvision
import tqdm
import yaml


PROJECT_ROOT = Path("/home/ly/RC-IRSTD-v2")
OUTPUT_ROOT = PROJECT_ROOT / "outputs/aaai27/detectors/component_rescue/tier2r_c_v1"
AUDIT_ROOT = PROJECT_ROOT / "artifacts/aaai27/audit/component_rescue/tier2r_c_v1"
TARGET_ROOT = PROJECT_ROOT / "datasets/NUAA-SIRST"
MOUNT_POINTS = (PROJECT_ROOT, OUTPUT_ROOT, AUDIT_ROOT, TARGET_ROOT)


def _unescape_mount_path(value: str) -> str:
    for encoded, decoded in (
        ("\\040", " "),
        ("\\011", "\t"),
        ("\\012", "\n"),
        ("\\134", "\\"),
    ):
        value = value.replace(encoded, decoded)
    return value


def _mount_evidence() -> dict[str, dict[str, object]]:
    expected = {str(path) for path in MOUNT_POINTS}
    evidence: dict[str, dict[str, object]] = {}
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        separator = fields.index("-")
        mount_point = _unescape_mount_path(fields[4])
        if mount_point in expected:
            evidence[mount_point] = {
                "mount_options": fields[5].split(","),
                "filesystem_type": fields[separator + 1],
                "mount_source": fields[separator + 2],
                "super_options": fields[separator + 3].split(","),
            }
    return evidence


def _nvidia_inventory() -> list[dict[str, object]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    inventory = []
    for line in completed.stdout.splitlines():
        index, uuid, name = (value.strip() for value in line.split(",", 2))
        inventory.append({"index": int(index), "uuid": uuid, "name": name})
    return inventory


def _torch_inventory() -> list[dict[str, object]]:
    result = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        result.append(
            {
                "ordinal": index,
                "uuid": "GPU-" + str(properties.uuid),
                "name": properties.name,
            }
        )
    return result


def probe() -> dict[str, object]:
    target_error: str | None = None
    try:
        os.listdir(TARGET_ROOT)
    except OSError as error:
        target_error = type(error).__name__
    home_entries = sorted(path.name for path in Path("/home/ly").iterdir())
    return {
        "schema_version": "rc-irstd-aaai27-tier2r-container-attestation-v1",
        "project_root": str(PROJECT_ROOT),
        "home_ly_entries": home_entries,
        "target": {
            "path": str(TARGET_ROOT),
            "mode": stat.S_IMODE(TARGET_ROOT.stat().st_mode),
            "list_error": target_error,
        },
        "mounts": _mount_evidence(),
        "nvidia_inventory": _nvidia_inventory(),
        "torch_inventory": _torch_inventory(),
        "environment": {
            "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        },
        "runtime": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torchvision": torchvision.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "PIL": PIL.__version__,
            "skimage": skimage.__version__,
            "yaml": yaml.__version__,
            "pandas": pandas.__version__,
            "tqdm": tqdm.__version__,
        },
    }


def main() -> int:
    print(json.dumps(probe(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
