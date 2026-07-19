"""Fit source-domain statistic centres from label-free episode features."""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from .domain_statistics import (
    STATISTICS_SCHEMA_VERSION,
    fit_source_reference,
    save_source_reference,
    validate_statistics_names,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-file", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument(
        "--exclude-domain",
        action="append",
        default=[],
        help="Fold-held-out domain to exclude; repeat as needed",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    reference_names: tuple[str, ...] | None = None
    for filename in args.episode_file:
        with np.load(filename, allow_pickle=False) as archive:
            required = {
                "statistics",
                "statistics_names",
                "statistics_schema_version",
                "pseudo_targets",
            }
            missing = sorted(required.difference(archive.files))
            if missing:
                raise ValueError(
                    f"{filename} is missing required arrays: {', '.join(missing)}"
                )
            statistics = np.asarray(archive["statistics"], dtype=np.float32)
            targets = np.asarray(archive["pseudo_targets"], dtype=str)
            names = validate_statistics_names(
                archive["statistics_names"],
                expected_dim=statistics.shape[1] if statistics.ndim == 2 else None,
            )
            schema_version = str(
                np.asarray(archive["statistics_schema_version"]).item()
            )
        if schema_version != STATISTICS_SCHEMA_VERSION:
            raise ValueError(
                f"{filename} uses incompatible statistics schema {schema_version!r}"
            )
        if reference_names is None:
            reference_names = names
        elif names != reference_names:
            raise ValueError(
                f"statistics_names order differs across source archives: {filename}"
            )
        if statistics.shape[0] != targets.size:
            raise ValueError(f"statistics/target count mismatch in {filename}")
        excluded = set(args.exclude_domain)
        for target in np.unique(targets):
            if str(target) in excluded:
                continue
            grouped[str(target)].append(statistics[targets == target])
    domain_statistics = {name: np.concatenate(parts, axis=0) for name, parts in grouped.items()}
    if reference_names is None:
        raise RuntimeError("No statistics schema was loaded")
    if not domain_statistics:
        raise ValueError("All source domains were excluded from the reference")
    reference = fit_source_reference(
        domain_statistics,
        regularization=args.regularization,
        statistics_names=reference_names,
        statistics_schema_version=STATISTICS_SCHEMA_VERSION,
    )
    path = save_source_reference(reference, args.output)
    print(f"saved {len(reference.domain_names)} source-domain references to {path}")


if __name__ == "__main__":
    main()
