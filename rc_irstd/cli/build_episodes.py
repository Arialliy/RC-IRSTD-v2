"""Build integrity-verified, disjoint support/query meta episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from evaluation.artifact_integrity import verify_score_map_directory
from rc_irstd.features import FeatureSpec
from rc_irstd.meta import build_episode_archive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", "--score-map-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expected-split-role",
        choices=("train", "test"),
        default="test",
        help=(
            "Official pseudo-target split expected in every score manifest. "
            "The default preserves the legacy test-split contract; formal "
            "source-only meta training must pass 'train' explicitly."
        ),
    )
    parser.add_argument("--budgets", nargs="+", type=float, default=[1e-4, 1e-5, 1e-6])
    parser.add_argument("--support-size", type=int, default=32)
    parser.add_argument("--query-size", type=int, default=64)
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Defaults to support_size + query_size to prevent cross-role reuse",
    )
    parser.add_argument("--max-episodes-per-domain", type=int, default=None)
    parser.add_argument("--mode", choices=["causal", "random"], default="causal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probability-bins", type=int, default=32)
    parser.add_argument("--logit-bins", type=int, default=32)
    parser.add_argument("--peak-bins", type=int, default=32)
    parser.add_argument("--risk-bins", type=int, default=256)
    parser.add_argument(
        "--allow-cross-episode-role-reuse",
        action="store_true",
        help="Diagnostic only: allow stride smaller than support + query",
    )
    parser.add_argument(
        "--allow-diagnostic-random",
        action="store_true",
        help="Diagnostic only: allow non-causal random support/query episodes",
    )
    parser.add_argument(
        "--allow-diagnostic-detector",
        action="store_true",
        help="Diagnostic only: accept diagnostic detector or resized score artifacts",
    )
    return parser


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _domain_key(value: object) -> str:
    key = "".join(character for character in str(value).casefold() if character.isalnum())
    if key.endswith("sirst") and len(key) > len("sirst"):
        key = key[: -len("sirst")]
    return key


def _audit_lodo_manifest(
    manifest: dict[str, object],
    score_dir: str,
    *,
    allow_diagnostic_detector: bool,
    expected_split_role: str = "test",
) -> tuple[str, bool]:
    if expected_split_role not in {"train", "test"}:
        raise ValueError("expected_split_role must be 'train' or 'test'")
    target = manifest.get("target_dataset")
    sources = manifest.get("source_datasets")
    if not isinstance(target, str) or not target:
        raise ValueError(f"Score manifest lacks target_dataset: {score_dir}")
    if not isinstance(sources, list) or not sources or any(
        not isinstance(value, str) or not value for value in sources
    ):
        raise ValueError(
            f"Score manifest lacks detector source_datasets provenance: {score_dir}"
        )
    target_key = _domain_key(target)
    if target_key in {_domain_key(value) for value in sources}:
        raise ValueError(
            f"Pseudo-target {target!r} appears in detector source domains for {score_dir}"
        )
    artifact_is_diagnostic = bool(
        manifest.get("checkpoint_diagnostic_only", False)
        or manifest.get("non_strict_state_loading", False)
        or manifest.get("spatial_mode") != "native"
        or manifest.get("split_role") != expected_split_role
        or manifest.get("split_authority_verified") is not True
    )
    if artifact_is_diagnostic and not allow_diagnostic_detector:
        raise ValueError(
            f"Score maps use a diagnostic detector/spatial protocol: {score_dir}. Pass "
            "--allow-diagnostic-detector only for smoke/diagnostic episodes."
        )
    return target, artifact_is_diagnostic


def _annotate_metadata(
    output_dir: Path,
    *,
    score_dirs: Sequence[str],
    stride: int,
    support_size: int,
    query_size: int,
    mode: str,
    diagnostic: bool,
    pseudo_target_split: str = "test",
    expected_split_role: str = "test",
) -> None:
    path = output_dir / "metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(
        {
            "formal_causal_contract": bool(not diagnostic),
            "diagnostic_only": bool(diagnostic),
            "causal_contract": "disjoint_support_then_future_query_no_cross_role_reuse",
            "support_size": int(support_size),
            "query_size": int(query_size),
            "stride": int(stride),
            "mode": mode,
            "pseudo_target_split": pseudo_target_split,
            "expected_split_role": expected_split_role,
            "diagnostic_reason": (
                "diagnostic detector and/or non-formal episode window contract"
                if diagnostic
                else None
            ),
            "score_manifests": [
                {
                    "path": str((Path(root) / "manifest.json").resolve()),
                    "sha256": _file_sha256(Path(root) / "manifest.json"),
                }
                for root in score_dirs
            ],
        }
    )
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.support_size <= 0 or args.query_size <= 0:
        raise ValueError("support-size and query-size must be positive")
    span = args.support_size + args.query_size
    stride = span if args.stride is None else args.stride
    if stride <= 0:
        raise ValueError("stride must be positive")
    if args.mode == "random" and not args.allow_diagnostic_random:
        raise ValueError(
            "Random episodes are not eligible for the causal protocol; pass "
            "--allow-diagnostic-random only for an explicitly diagnostic artifact"
        )
    if stride < span and not args.allow_cross_episode_role_reuse:
        raise ValueError(
            f"Formal episodes require stride >= support + query ({span}); pass "
            "--allow-cross-episode-role-reuse only for diagnostic output"
        )
    pseudo_targets: list[str] = []
    pseudo_target_splits: list[str] = []
    diagnostic_detector_seen = False
    for score_dir in args.score_dir:
        manifest, _, _ = verify_score_map_directory(
            score_dir,
            require_integrity=True,
            require_masks=True,
        )
        if manifest is None:
            raise ValueError(f"Score directory has no integrity manifest: {score_dir}")
        target_name, detector_is_diagnostic = _audit_lodo_manifest(
            manifest,
            score_dir,
            expected_split_role=args.expected_split_role,
            allow_diagnostic_detector=args.allow_diagnostic_detector,
        )
        pseudo_targets.append(target_name)
        pseudo_target_splits.append(str(manifest.get("split_role")))
        diagnostic_detector_seen = diagnostic_detector_seen or detector_is_diagnostic
    if len({_domain_key(value) for value in pseudo_targets}) != len(pseudo_targets):
        raise ValueError("Every score directory must represent a distinct pseudo-target")
    if len(set(pseudo_target_splits)) != 1:
        raise ValueError("Every score directory must use the same pseudo-target split role")
    spec = FeatureSpec(
        probability_bins=args.probability_bins,
        logit_bins=args.logit_bins,
        peak_bins=args.peak_bins,
    )
    archive = build_episode_archive(
        args.score_dir,
        args.output_dir,
        budgets=args.budgets,
        support_size=args.support_size,
        query_size=args.query_size,
        stride=stride,
        max_episodes_per_domain=args.max_episodes_per_domain,
        mode=args.mode,
        seed=args.seed,
        feature_spec=spec,
        risk_bins=args.risk_bins,
    )
    diagnostic = bool(
        args.mode != "causal"
        or stride < span
        or args.allow_cross_episode_role_reuse
        or args.allow_diagnostic_random
        or diagnostic_detector_seen
    )
    _annotate_metadata(
        Path(args.output_dir),
        score_dirs=args.score_dir,
        stride=stride,
        support_size=args.support_size,
        query_size=args.query_size,
        mode=args.mode,
        diagnostic=diagnostic,
        pseudo_target_split=pseudo_target_splits[0],
        expected_split_role=args.expected_split_role,
    )
    print(archive)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
