"""Stable import facade for the RC-IRSTD-v2 repository.

The package intentionally re-exports the repository's existing implementations;
it does not define a second training, data, loss, or evaluation protocol.
"""

from .models import (
    MSHNet,
    MSHNetOutput,
    MonotoneBudgetCalibrator,
    build_mshnet,
    forward_mshnet,
    structure_mshnet_output,
)

__all__ = [
    "MSHNet",
    "MSHNetOutput",
    "MonotoneBudgetCalibrator",
    "build_mshnet",
    "forward_mshnet",
    "structure_mshnet_output",
]

__version__ = "0.1.0"
