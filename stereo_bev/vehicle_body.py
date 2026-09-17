"""Tesla Model 3 3D body: lattice samples and overlapping collision balls."""

from __future__ import annotations

import math
import numpy as np

# Approximate Model 3 outer dimensions (meters). Origin: vehicle center, z=ground.
MODEL3_LENGTH = 4.694
MODEL3_WIDTH = 1.849
MODEL3_HEIGHT = 1.443
MODEL3_WHEELBASE = 2.875
MODEL3_MAX_STEER_DEG = 70.0


def model3_collision_balls(
    n_long: int = 5,
    n_lat: int = 2,
    n_vert: int = 2,
    length: float = MODEL3_LENGTH,
    width: float = MODEL3_WIDTH,
    height: float = MODEL3_HEIGHT,
    margin: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Overlapping spheres that cover the Model 3 axis-aligned bounding box.

    Each cell of an ``n_long × n_lat × n_vert`` grid gets a sphere whose
    radius is the cell half-diagonal plus ``margin``, so the eight cell
    corners (and therefore the whole box) are inside some ball.

    Ego frame: X forward, Y left, Z up. XY origin is the vehicle center;
    Z=0 is ground.

    Returns:
        centers: (N, 3) float64
        radii:   (N,) float64
    """
    hx = 0.5 * length / n_long
    hy = 0.5 * width / n_lat
    hz = 0.5 * height / n_vert
    radius = math.sqrt(hx * hx + hy * hy + hz * hz) + float(margin)
    xs = np.linspace(-0.5 * length + hx, 0.5 * length - hx, n_long)
    ys = np.linspace(-0.5 * width + hy, 0.5 * width - hy, n_lat)
    zs = np.linspace(hz, height - hz, n_vert)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    centers = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float64)
    radii = np.full(centers.shape[0], radius, dtype=np.float64)
    return centers, radii


def model3_body_samples(
    nx: int = 7,
    ny: int = 5,
    nz: int = 3,
    length: float = MODEL3_LENGTH,
    width: float = MODEL3_WIDTH,
    height: float = MODEL3_HEIGHT,
    z_min: float = 0.40,
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


def collision_inflation_radius(
    body: np.ndarray,
    radii: np.ndarray,
    extra: float = 0.10,
) -> float:
    """A* inflation matching the lateral collision-ball footprint."""
    body = np.asarray(body, dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    return float(np.max(np.abs(body[:, 1]) + radii) + extra)


def body_xy_support(
    nx: np.ndarray,
    ny: np.ndarray,
    yaw: float,
    body: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """Minkowski offset of XY halfspaces by the yawed collision balls."""
    nx = np.asarray(nx, dtype=np.float64).reshape(-1)
    ny = np.asarray(ny, dtype=np.float64).reshape(-1)
    body = np.asarray(body, dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    bx = c * body[:, 0] - s * body[:, 1]
    by = s * body[:, 0] + c * body[:, 1]
    proj = nx[:, None] * bx[None, :] + ny[:, None] * by[None, :]
    return np.max(proj + radii[None, :], axis=1)


def transform_body_se3(
    t: np.ndarray,
    R: np.ndarray,
    body: np.ndarray,
) -> np.ndarray:
    """
    Transform body samples by SE(3) pose(s): p = t + R @ body.

    ``t`` is (3,) or (K, 3); ``R`` is (3, 3) or (K, 3, 3).
    Returns (N, 3) or (K, N, 3).
    """
    body = np.asarray(body, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    if t.ndim == 1:
        return body @ np.asarray(R, dtype=np.float64).reshape(3, 3).T + t.reshape(3)
    t = t.reshape(-1, 3)
    R = R.reshape(-1, 3, 3)
    return np.einsum("kij,nj->kni", R, body) + t[:, None, :]


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
