"""Generate explicit nested leave-one-domain-out fold manifests."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


LODO_SCHEMA_VERSION = "rc-v2-nested-lodo-v1"


@dataclass(frozen=True)
class LodoFold:
    fold_id: str
    detector_sources: tuple[str, ...]
    pseudo_target: str
    outer_target: str | None


def build_lodo_folds(
    domains: Sequence[str], outer_target: str | None = None
) -> list[LodoFold]:
    names = tuple(str(name).strip() for name in domains)
    if len(names) < 2 or any(not name for name in names):
        raise ValueError("At least two non-empty domain names are required")
    if len(set(names)) != len(names):
        raise ValueError("Domain names must be unique")
    if outer_target is not None:
        outer_target = str(outer_target).strip()
        if outer_target not in names:
            raise ValueError("outer_target must be one of domains")
        inner_domains = tuple(name for name in names if name != outer_target)
        if len(inner_domains) < 2:
            raise ValueError("Nested LODO needs at least three total domains")
    else:
        inner_domains = names
    folds: list[LodoFold] = []
    for pseudo_target in inner_domains:
        detector_sources = tuple(name for name in inner_domains if name != pseudo_target)
        folds.append(
            LodoFold(
                fold_id=(
                    f"outer-{outer_target}__pseudo-{pseudo_target}"
                    if outer_target is not None
                    else f"pseudo-{pseudo_target}"
                ),
                detector_sources=detector_sources,
                pseudo_target=pseudo_target,
                outer_target=outer_target,
            )
        )
    return folds


def write_lodo_manifest(
    path: str | Path, domains: Sequence[str], outer_target: str | None = None
) -> Path:
    folds = build_lodo_folds(domains, outer_target)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": LODO_SCHEMA_VERSION,
        "domains": list(domains),
        "outer_target": outer_target,
        "folds": [asdict(fold) for fold in folds],
        "leakage_rule": (
            "outer_target is excluded from every detector and pseudo-target episode; "
            "each pseudo_target is excluded from its detector_sources"
        ),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--outer-target")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = write_lodo_manifest(args.output, args.domains, args.outer_target)
    print(f"wrote LODO manifest to {output}")


if __name__ == "__main__":
    main()
