"""Model facade backed by :mod:`model.MSHNet`."""

from .calibrator import (
    RC_DIRECT_ARCHITECTURE_VERSION,
    CalibratorOutput,
    FeatureNormalizer,
    MonotoneBudgetCalibrator,
)
from .mshnet import (
    MSHNet,
    MSHNetOutput,
    ResNet,
    build_mshnet,
    forward_mshnet,
    structure_mshnet_output,
)
# RC-MSHNET-PATCH: facade exports
from .rc_mshnet import (
    RCMSHNet,
    RC_MSHNET_ARCHITECTURE_VERSION,
    RC_MSHNET_ARCHITECTURE_VERSION_V1,
    RC_MSHNET_ARCHITECTURE_VERSION_V2,
    build_rc_mshnet,
    initialize_rc_mshnet_from_checkpoint,
    rc_mshnet_extension_state_sha256,
)
from .system import CalibratedPrediction, RCIRSTDSystem

__all__ = [
    "RCMSHNet",
    "RC_MSHNET_ARCHITECTURE_VERSION",
    "RC_MSHNET_ARCHITECTURE_VERSION_V1",
    "RC_MSHNET_ARCHITECTURE_VERSION_V2",
    "build_rc_mshnet",
    "initialize_rc_mshnet_from_checkpoint",
    "rc_mshnet_extension_state_sha256",
    "CalibratedPrediction",
    "CalibratorOutput",
    "FeatureNormalizer",
    "MSHNet",
    "MSHNetOutput",
    "MonotoneBudgetCalibrator",
    "RC_DIRECT_ARCHITECTURE_VERSION",
    "RCIRSTDSystem",
    "ResNet",
    "build_mshnet",
    "forward_mshnet",
    "structure_mshnet_output",
]
