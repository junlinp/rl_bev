"""Tesla Model 3 3D body sample points in the vehicle / ego frame."""

from __future__ import annotations

import numpy as np

# Approximate Model 3 outer dimensions (meters). Origin: vehicle center, z=ground.
MODEL3_LENGTH = 4.694
MODEL3_WIDTH = 1.849
MODEL3_HEIGHT = 1.443
MODEL3_WHEELBASE = 2.875


def model3_body_samples(
    nx: int = 3,
    ny: int = 3,
    nz: int = 3,
    length: float = MODEL3_LENGTH,
    width: float = MODEL3_WIDTH,
    height: float = MODEL3_HEIGHT,
    z_min: float = 0.50,
    z_max: float | None = None,
) -> np.ndarray:
    """
    Coarse 3D lattice covering the vehicle bounding box.

    Ego / vehicle frame: X forward, Y left, Z up. XY origin is the vehicle
    center (CARLA actor origin); Z=0 is ground.

    Returns:
        (N, 3) float64 sample points (x, y, z)
    """
    if z_max is None:
        z_max = max(z_min + 0.05, height - 0.05)
    xs = np.linspace(-0.5 * length, 0.5 * length, nx)
    ys = np.linspace(-0.5 * width, 0.5 * width, ny)
    zs = np.linspace(z_min, z_max, nz)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float64)


def transform_body(
    x: np.ndarray | float,
    y: np.ndarray | float,
    psi: np.ndarray | float,
    body: np.ndarray,
) -> np.ndarray:
    """
    Transform body samples by planar pose(s).

    If x, y, psi are scalars, returns (N, 3).
    If they are length-K arrays, returns (K, N, 3).
    """
    body = np.asarray(body, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    scalar = x.ndim == 0
    x, y, psi = np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(psi)
    c = np.cos(psi)
    s = np.sin(psi)
    # (K, N)
    px = x[:, None] + c[:, None] * body[None, :, 0] - s[:, None] * body[None, :, 1]
    py = y[:, None] + s[:, None] * body[None, :, 0] + c[:, None] * body[None, :, 1]
    pz = np.broadcast_to(body[None, :, 2], px.shape)
    out = np.stack([px, py, pz], axis=-1)
    if scalar:
        return out[0]
    return out


def body_sweep_voxels(
    traj_x: np.ndarray,
    traj_y: np.ndarray,
    traj_psi: np.ndarray,
    body: np.ndarray,
    grid,
) -> np.ndarray:
    """Boolean (Z, Y, X) mask of voxels the 3D body occupies along a trajectory."""
    from .occ_field import points_to_voxels

    pts = transform_body(traj_x, traj_y, traj_psi, body).reshape(-1, 3)
    zi, yi, xi, valid = points_to_voxels(pts, grid)
    mask = np.zeros((grid.grid_z, grid.grid_h, grid.grid_w), dtype=np.uint8)
    if valid.any():
        mask[zi[valid], yi[valid], xi[valid]] = 1
    return mask
