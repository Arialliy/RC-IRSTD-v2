"""YAML multi-domain detector-training entry point.

The historical :mod:`train` command remains untouched for the upstream
BasicIRSTD-compatible workflow.
"""

from rc_irstd.cli.train_detector import main


if __name__ == "__main__":
    raise SystemExit(main())
