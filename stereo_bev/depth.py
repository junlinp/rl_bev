"""Depth decoding from CARLA's depth sensor + point cloud unprojection."""

import numpy as np


def decode_carla_depth(image: np.ndarray) -> np.ndarray:
    """
    Decode CARLA's `sensor.camera.depth` raw image to metric depth (meters).

    CARLA encodes depth in RGB channels:
        normalized = (R + G*256 + B*256*256) / (256^3 - 1)
        depth_m    = normalized * 1000.0

    Args:
        image: (H, W, 3) uint8 BGR image from depth sensor

    Returns:
        (H, W) float32 depth in meters
    """
    r = image[:, :, 2].astype(np.float64)
    g = image[:, :, 1].astype(np.float64)
    b = image[:, :, 0].astype(np.float64)

    normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 ** 3 - 1.0)
    return (normalized * 1000.0).astype(np.float32)


def depth_to_pointcloud(
    depth: np.ndarray,
    K: np.ndarray,
    max_depth: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unproject a depth map to a 3D point cloud in camera frame (X-right, Y-down, Z-forward).

    Args:
        depth: (H, W) float depth in meters
        K: (3, 3) intrinsic matrix
        max_depth: far clip

    Returns:
        points: (N, 3) XYZ
        pixel_coords: (N, 2) (u, v) back-projection indices
    """
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid = (depth > 0.1) & (depth < max_depth)

    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)
    pixels = np.stack([u[valid], v[valid]], axis=-1)

    return points, pixels
