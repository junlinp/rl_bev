"""BEV perception pipeline for CARLA using stereo cameras."""

from .calibration import intrinsics_from_carla, print_intrinsics
from .depth import decode_carla_depth, depth_to_pointcloud
from .segmentation import remap_segmentation, BEV_CLASSES, NUM_BEV_CLASSES, colorize_bev
from .bev_grid import BEVGrid
from .query_heads import (
    GeometricSegHead,
    GeometricOccHead,
)
from .camera_rig import CameraRig

try:
    from .query_heads import (
        StereoBEVModel,
        SegQueryHead,
        OccQueryHead,
        stereo_bev_loss,
        prepare_stereo_input,
    )
except ImportError:
    pass

__all__ = [
    "intrinsics_from_carla",
    "print_intrinsics",
    "decode_carla_depth",
    "depth_to_pointcloud",
    "remap_segmentation",
    "BEV_CLASSES",
    "NUM_BEV_CLASSES",
    "colorize_bev",
    "BEVGrid",
    "GeometricSegHead",
    "GeometricOccHead",
    "CameraRig",
]
