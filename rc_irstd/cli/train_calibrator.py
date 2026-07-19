from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rc_irstd.config import apply_overrides, load_yaml, resolve_config_path
from rc_irstd.training import CalibratorTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the monotone budget calibrator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument(
        "--allow-diagnostic-episodes",
        action="store_true",
        help="Allow an explicitly diagnostic/non-causal episode archive",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = apply_overrides(load_yaml(args.config), args.overrides)
    episodes_value = config.get("episodes_dir")
    if not isinstance(episodes_value, (str, Path)):
        raise ValueError("Calibrator config must define episodes_dir")
    metadata_path = resolve_config_path(config, episodes_value) / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    formal = metadata.get("formal_causal_contract") is True
    if not formal and not args.allow_diagnostic_episodes:
        raise ValueError(
            "Calibrator training requires an episode archive with "
            "formal_causal_contract=true; use --allow-diagnostic-episodes only "
            "for a non-paper diagnostic run"
        )
    if not formal:
        config["diagnostic_only"] = True
    config["episode_contract"] = {
        "metadata": str(metadata_path.resolve()),
        "formal_causal_contract": formal,
        "diagnostic_only": not formal,
    }
    best = CalibratorTrainer(config).run()
    print(best)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
