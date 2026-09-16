"""BEV perception pipeline for CARLA using stereo cameras."""

from .calibration import (
    intrinsics_from_carla,
    print_intrinsics,
    ego_from_camera,
)
from .depth import decode_carla_depth, depth_to_pointcloud
from .segmentation import remap_segmentation, BEV_CLASSES, NUM_BEV_CLASSES, colorize_bev
from .bev_grid import BEVGrid, occupancy_to_bev
from .query_heads import (
    GeometricSegHead,
    GeometricOccHead,
)

try:
    from .camera_rig import CameraRig
except ImportError:
    CameraRig = None  # CARLA Python API not installed

try:
    from .query_heads import (
        StereoBEVModel,
        OccQueryHead,
        stereo_bev_loss,
        occupancy_iou_counts,
        occupancy_iou_from_counts,
    )
except ImportError:
    pass

__all__ = [
    "intrinsics_from_carla",
    "print_intrinsics",
    "ego_from_camera",
    "decode_carla_depth",
    "depth_to_pointcloud",
    "remap_segmentation",
    "BEV_CLASSES",
    "NUM_BEV_CLASSES",
    "colorize_bev",
    "BEVGrid",
    "occupancy_to_bev",
    "GeometricSegHead",
    "GeometricOccHead",
    "CameraRig",
]
