"""3D occupancy Euclidean signed-distance field (no XY collapse)."""

from __future__ import annotations

import numpy as np

from .bev_grid import BEVGrid
from .vehicle_body import MODEL3_LENGTH, MODEL3_WIDTH

# BEV classes treated as planning obstacles (not road / sidewalk / terrain / vegetation).
# "other" keeps CARLA Static / Fence / GuardRail from being wiped with the road.
OBSTACLE_BEV_CLASSES = (3, 4, 5, 8, 9)  # vehicle, ped, building, pole_sign, other


def esdf_axis_grids(grid: BEVGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxel-center coordinates (x, y, z) matching occupancy layout (Z, Y, X)."""
    vs = grid.voxel_size
    x = grid.x_range[0] + (np.arange(grid.grid_w, dtype=np.float64) + 0.5) * vs
    y = grid.y_range[0] + (np.arange(grid.grid_h, dtype=np.float64) + 0.5) * vs
    z = grid.z_range[0] + (np.arange(grid.grid_z, dtype=np.float64) + 0.5) * vs
    return x, y, z


def obstacle_volume(
    occ: np.ndarray,
    grid: BEVGrid,
    z_ground: float = 0.5,
    voxel_class: np.ndarray | None = None,
    bev_classes: np.ndarray | None = None,
    class_histogram: np.ndarray | None = None,
    clear_ego: bool = True,
    inflate_m: float = 0.4,
    ego_x_max: float | None = None,
    ego_half_width: float | None = None,
) -> np.ndarray:
    """
    3D boolean obstacle mask for planning.

    Clears ground (z < z_ground), the ego footprint, and non-obstacle
    semantics; optional binary inflation. No XY max-collapse.

    ``voxel_class`` is (Z, Y, X): each voxel keeps its own class. Road at
    the ground does not wipe a pole voxel above it. 2D ``bev_classes`` /
    Z-collapsed histograms are only a fallback.
    """
    if occ.ndim != 3:
        raise ValueError(f"expected 3D occupancy (Z,Y,X), got shape {occ.shape}")
    occupied = np.asarray(occ).astype(bool)
    if occupied.shape != (grid.grid_z, grid.grid_h, grid.grid_w):
        raise ValueError(
            f"occupancy shape {occupied.shape} != "
            f"({grid.grid_z}, {grid.grid_h}, {grid.grid_w})"
        )

    occupied = occupied.copy()
    _, _, z_coords = esdf_axis_grids(grid)
    occupied[z_coords < z_ground, :, :] = False

    cls3 = None
    if voxel_class is not None:
        vc = np.asarray(voxel_class)
        if vc.shape == occupied.shape:
            cls3 = vc
    if cls3 is None and class_histogram is not None:
        hist = np.asarray(class_histogram)
        if hist.ndim == 4 and hist.shape[1:] == occupied.shape:
            cls3 = hist.argmax(axis=0).astype(np.uint8)
            cls3[hist.sum(axis=0) <= 0] = 0

    if cls3 is not None:
        is_obs = np.isin(cls3, OBSTACLE_BEV_CLASSES)
        is_free_sem = (cls3 != 0) & ~is_obs
        occupied |= is_obs
        occupied &= ~is_free_sem
        occupied[z_coords < z_ground, :, :] = False
    else:
        obs_xy = None
        if class_histogram is not None:
            hist = np.asarray(class_histogram)
            if hist.ndim == 3 and hist.shape[1:] == occupied.shape[1:]:
                idx = np.array(OBSTACLE_BEV_CLASSES, dtype=np.int32)
                idx = idx[idx < hist.shape[0]]
                if idx.size:
                    obs_xy = hist[idx].sum(axis=0) > 0
        if obs_xy is None and bev_classes is not None:
            cls = np.asarray(bev_classes)
            if cls.shape == occupied.shape[1:]:
                obs_xy = np.isin(cls, OBSTACLE_BEV_CLASSES)
        if obs_xy is not None:
            occupied &= obs_xy[None, :, :]

    if inflate_m > 0 and occupied.any():
        from scipy.ndimage import binary_dilation
        n = max(1, int(round(inflate_m / grid.voxel_size)))
        # Dilate in XY only so overhangs stay overhangs.
        struct = np.zeros((1, 2 * n + 1, 2 * n + 1), dtype=bool)
        struct[0, :, :] = True
        occupied = binary_dilation(occupied, structure=struct)

    if clear_ego:
        if ego_x_max is None:
            ego_x_max = 0.5 * MODEL3_LENGTH + 0.4
        if ego_half_width is None:
            ego_half_width = 0.5 * MODEL3_WIDTH + 0.3
        vs = grid.voxel_size
        xi_max = int(np.clip(np.ceil((ego_x_max - grid.x_range[0]) / vs), 0, grid.grid_w))
        yi0 = int(np.clip(np.floor((-ego_half_width - grid.y_range[0]) / vs), 0, grid.grid_h))
        yi1 = int(np.clip(np.ceil((ego_half_width - grid.y_range[0]) / vs), 0, grid.grid_h))
        occupied[:, yi0:yi1, :xi_max] = False

    return occupied


def occupancy_to_esdf_3d(
    occ: np.ndarray,
    grid: BEVGrid,
    z_ground: float = 0.5,
    max_dist: float | None = None,
    voxel_class: np.ndarray | None = None,
    bev_classes: np.ndarray | None = None,
    class_histogram: np.ndarray | None = None,
    clear_ego: bool = True,
    inflate_m: float = 0.4,
) -> np.ndarray:
    """
    Convert a 3D occupancy volume to a 3D signed distance field.

    Distances are in meters: positive outside obstacles, negative inside.
    """
    occupied = obstacle_volume(
        occ, grid, z_ground=z_ground, voxel_class=voxel_class,
        bev_classes=bev_classes, class_histogram=class_histogram,
        clear_ego=clear_ego, inflate_m=inflate_m,
    )

    if max_dist is None:
        max_dist = float(np.linalg.norm([
            grid.x_range[1] - grid.x_range[0],
            grid.y_range[1] - grid.y_range[0],
            grid.z_range[1] - grid.z_range[0],
        ]))

    if not occupied.any():
        return np.full(occupied.shape, max_dist, dtype=np.float32)
    if occupied.all():
        return np.full(occupied.shape, -max_dist, dtype=np.float32)

    from scipy.ndimage import distance_transform_edt

    occupied_u8 = occupied.astype(np.uint8)
    free_u8 = (1 - occupied_u8).astype(np.uint8)
    dist_out = distance_transform_edt(free_u8, sampling=grid.voxel_size)
    dist_in = distance_transform_edt(occupied_u8, sampling=grid.voxel_size)
    esdf = dist_out - dist_in
    return np.clip(esdf, -max_dist, max_dist).astype(np.float32)


def flatten_esdf_casadi(esdf: np.ndarray) -> np.ndarray:
    """Flatten (Z, Y, X) with X fastest, matching CasADi grid [x, y, z]."""
    return np.ascontiguousarray(esdf, dtype=np.float64).ravel()


def query_esdf(esdf: np.ndarray, grid: BEVGrid, points: np.ndarray) -> np.ndarray:
    """
    Trilinear sample of a 3D ESDF at (N, 3) ego-frame points.

    Points outside the grid are clamped to the border.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    xg, yg, zg = esdf_axis_grids(grid)
    return _trilinear(esdf.astype(np.float64), xg, yg, zg, pts)


def _trilinear(
    vol: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    zg: np.ndarray,
    pts: np.ndarray,
) -> np.ndarray:
    """vol is (Z, Y, X); pts are (N, 3) as (x, y, z)."""
    nx, ny, nz = xg.size, yg.size, zg.size

    def _axis(p, g):
        t = (p - g[0]) / (g[1] - g[0]) if g.size > 1 else np.zeros_like(p)
        i0 = np.floor(t).astype(np.int32)
        i0 = np.clip(i0, 0, g.size - 2) if g.size >= 2 else np.zeros_like(i0)
        a = t - i0
        return i0, np.clip(a, 0.0, 1.0)

    ix, ax = _axis(pts[:, 0], xg)
    iy, ay = _axis(pts[:, 1], yg)
    iz, az = _axis(pts[:, 2], zg)
    ix1 = np.minimum(ix + 1, nx - 1)
    iy1 = np.minimum(iy + 1, ny - 1)
    iz1 = np.minimum(iz + 1, nz - 1)

    c000 = vol[iz, iy, ix]
    c100 = vol[iz, iy, ix1]
    c010 = vol[iz, iy1, ix]
    c110 = vol[iz, iy1, ix1]
    c001 = vol[iz1, iy, ix]
    c101 = vol[iz1, iy, ix1]
    c011 = vol[iz1, iy1, ix]
    c111 = vol[iz1, iy1, ix1]

    c00 = c000 * (1 - ax) + c100 * ax
    c10 = c010 * (1 - ax) + c110 * ax
    c01 = c001 * (1 - ax) + c101 * ax
    c11 = c011 * (1 - ax) + c111 * ax
    c0 = c00 * (1 - ay) + c10 * ay
    c1 = c01 * (1 - ay) + c11 * ay
    return (c0 * (1 - az) + c1 * az).astype(np.float64)


def points_to_voxels(points: np.ndarray, grid: BEVGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map (N, 3) ego points → voxel indices (zi, yi, xi) plus in-bounds mask."""
    pts = np.asarray(points, dtype=np.float64)
    vs = grid.voxel_size
    xi = np.floor((pts[:, 0] - grid.x_range[0]) / vs).astype(np.int32)
    yi = np.floor((pts[:, 1] - grid.y_range[0]) / vs).astype(np.int32)
    zi = np.floor((pts[:, 2] - grid.z_range[0]) / vs).astype(np.int32)
    valid = (
        (xi >= 0) & (xi < grid.grid_w) &
        (yi >= 0) & (yi < grid.grid_h) &
        (zi >= 0) & (zi < grid.grid_z)
    )
    return zi, yi, xi, valid
