"""Camera intrinsics for CARLA (ideal pinhole, no distortion)."""

import math
import numpy as np


def intrinsics_from_carla(image_w: int, image_h: int, fov_deg: float) -> tuple[np.ndarray, float]:
    """
    Compute intrinsic matrix K from CARLA camera parameters.

    Returns:
        K: (3, 3) intrinsic matrix
        focal_px: focal length in pixels
    """
    focal = (image_w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    K = np.array([
        [focal, 0.0, image_w / 2.0],
        [0.0, focal, image_h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return K, focal


def print_intrinsics(K: np.ndarray, image_w: int, image_h: int, fov_deg: float):
    """Pretty-print camera intrinsics."""
    print(f"=== Camera Intrinsics ({image_w}x{image_h}, FOV={fov_deg}°) ===")
    print(f"focal_px = {K[0,0]:.2f}")
    print(f"cx = {K[0,2]:.2f}")
    print(f"cy = {K[1,2]:.2f}")
    print(f"K =\n{K}")
    print(f"distortion = [0, 0, 0, 0, 0]  (CARLA is ideal pinhole)")
