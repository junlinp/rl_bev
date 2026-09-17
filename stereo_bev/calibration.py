"""Camera intrinsics and ego-from-camera extrinsics for CARLA (ideal pinhole)."""

import math
import numpy as np

# CARLA vehicle frame: X forward, Y right, Z up
# Ego / BEV frame:     X forward, Y left,  Z up
DEFAULT_MOUNT = (1.5, 0.0, 1.6)  # meters in vehicle frame
DEFAULT_PITCH_DEG = -25.0        # negative = look down


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


def ego_from_camera(
    mount: tuple[float, float, float] = DEFAULT_MOUNT,
    pitch_deg: float = 0.0,
) -> np.ndarray:
    """
    4×4 ego-from-camera transform.

    Camera (OpenCV): X right, Y down, Z forward.
    Ego:             X forward, Y left, Z up.

    `pitch_deg` uses CARLA's convention: negative looks down.
    `mount` is the camera location in the CARLA vehicle frame (X fwd, Y right, Z up).
    """
    # untilted camera axes expressed in ego
    R0 = np.array([
        [0.0,  0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ], dtype=np.float64)

    # pitch down α > 0 when pitch_deg < 0; rotation about camera X (right)
    alpha = -math.radians(pitch_deg)
    ca, sa = math.cos(alpha), math.sin(alpha)
    r_pitch = np.array([
        [1.0, 0.0, 0.0],
        [0.0, ca,  sa],
        [0.0, -sa, ca],
    ], dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R0 @ r_pitch
    T[0, 3] = mount[0]
    T[1, 3] = -mount[1]  # vehicle Y-right → ego Y-left
    T[2, 3] = mount[2]
    return T


def camera_origin_xy(cam_extrinsic: np.ndarray) -> tuple[float, float]:
    """Left-camera XY in vehicle-center FLU (occupancy origin is the car)."""
    t = np.asarray(cam_extrinsic, dtype=np.float64)
    return float(t[0, 3]), float(t[1, 3])


def vehicle_to_occupancy(points: np.ndarray, origin_xy: tuple[float, float]) -> np.ndarray:
    """Shift vehicle-center FLU into a frame whose XY origin is ``origin_xy``."""
    out = np.asarray(points, dtype=np.float64).copy()
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    if out.ndim == 1:
        out[0] -= ox
        if out.shape[0] > 1:
            out[1] -= oy
        return out
    out[..., 0] -= ox
    out[..., 1] -= oy
    return out
