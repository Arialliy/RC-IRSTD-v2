from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from rc_irstd.data.synthetic import make_synthetic_domain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create small RC-IRSTD smoke-test domains")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-train", type=int, default=24)
    parser.add_argument("--num-test", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_train <= 0 or args.num_test <= 0:
        raise ValueError("num-train and num-test must be positive")
    if args.image_size < 16:
        raise ValueError("image-size must be at least 16")
    specifications = [
        dict(name="domain_a", seed=11, background_scale=0.8, stripe_strength=0.00, noise_std=0.22),
        dict(name="domain_b", seed=22, background_scale=1.1, stripe_strength=0.04, noise_std=0.27),
        dict(name="domain_c", seed=33, background_scale=1.4, stripe_strength=0.08, noise_std=0.32),
        dict(name="domain_d", seed=44, background_scale=1.7, stripe_strength=0.12, noise_std=0.36),
    ]
    for specification in specifications:
        path = make_synthetic_domain(
            args.output_root,
            num_train=args.num_train,
            num_test=args.num_test,
            image_size=args.image_size,
            **specification,
        )
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
