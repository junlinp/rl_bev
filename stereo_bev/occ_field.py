"""3D occupancy Euclidean signed-distance field (no XY collapse)."""

from __future__ import annotations

import numpy as np

from .bev_grid import BEVGrid
from .global_target import rotz
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
    ego_center_xy: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    3D boolean obstacle mask for planning.

    Clears ground (z < z_ground) and the ego footprint. Obstacle-class
    voxels are added even with few hits. Vegetation/terrain are dropped.
    Elevated occupancy is kept even when the class vote is road — that is
    how a car in the depth image stays in the 3D volume.

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

    # Ground (z < z_ground) is already cleared. Do not wipe elevated
    # occupancy just because the class vote is road/sidewalk — that is how
    # a car in the depth image disappears from the 3D volume. Vegetation
    # and terrain canopies are still dropped.
    _WIPE_ABOVE_GROUND = (6, 7)

    if cls3 is not None:
        occupied |= np.isin(cls3, OBSTACLE_BEV_CLASSES)
        occupied &= ~np.isin(cls3, _WIPE_ABOVE_GROUND)
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
            occupied |= obs_xy[None, :, :]
            occupied[z_coords < z_ground, :, :] = False

    if inflate_m > 0 and occupied.any():
        from scipy.ndimage import binary_dilation
        n = max(1, int(round(inflate_m / grid.voxel_size)))
        # Dilate in XY only so overhangs stay overhangs.
        struct = np.zeros((1, 2 * n + 1, 2 * n + 1), dtype=bool)
        struct[0, :, :] = True
        occupied = binary_dilation(occupied, structure=struct)

    if clear_ego:
        # Occupancy XY origin is the vehicle center unless ``ego_center_xy``
        # is set (legacy camera-origin volumes).
        cx, cy = (0.0, 0.0) if ego_center_xy is None else (
            float(ego_center_xy[0]), float(ego_center_xy[1]),
        )
        hx = 0.5 * MODEL3_LENGTH - 0.05 if ego_x_max is None else float(ego_x_max) - cx
        hy = 0.5 * MODEL3_WIDTH - 0.05 if ego_half_width is None else float(ego_half_width)
        vs = grid.voxel_size
        xi0 = int(np.clip(np.floor((cx - hx - grid.x_range[0]) / vs), 0, grid.grid_w))
        xi1 = int(np.clip(np.ceil((cx + hx - grid.x_range[0]) / vs), 0, grid.grid_w))
        yi0 = int(np.clip(np.floor((cy - hy - grid.y_range[0]) / vs), 0, grid.grid_h))
        yi1 = int(np.clip(np.ceil((cy + hy - grid.y_range[0]) / vs), 0, grid.grid_h))
        occupied[:, yi0:yi1, xi0:xi1] = False

    return occupied


def stamp_oriented_boxes(
    occupied: np.ndarray,
    grid: BEVGrid,
    centers: np.ndarray,
    halves: np.ndarray,
    yaws: np.ndarray | None = None,
    inflate_m: float = 0.08,
) -> np.ndarray:
    """OR actor bounding boxes into a planning volume (ego frame, Y left)."""
    out = np.asarray(occupied).astype(bool).copy()
    if centers is None or len(centers) == 0:
        return out
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    halves = np.asarray(halves, dtype=np.float64).reshape(-1, 3)
    if yaws is None:
        yaws = np.zeros(len(centers), dtype=np.float64)
    else:
        yaws = np.asarray(yaws, dtype=np.float64).reshape(-1)
    xg, yg, zg = esdf_axis_grids(grid)
    vs = grid.voxel_size
    pad = float(inflate_m)
    for c, h, yaw in zip(centers, halves, yaws):
        if not np.isfinite(c).all() or c[0] < grid.x_range[0] or c[0] > grid.x_range[1]:
            continue
        hx, hy, hz = float(h[0]) + pad, float(h[1]) + pad, float(h[2]) + pad
        cy, sy = abs(np.cos(yaw)), abs(np.sin(yaw))
        ext_x = hx * cy + hy * sy
        ext_y = hx * sy + hy * cy
        xi0 = int(np.clip(np.floor((c[0] - ext_x - grid.x_range[0]) / vs), 0, grid.grid_w))
        xi1 = int(np.clip(np.ceil((c[0] + ext_x - grid.x_range[0]) / vs), 0, grid.grid_w))
        yi0 = int(np.clip(np.floor((c[1] - ext_y - grid.y_range[0]) / vs), 0, grid.grid_h))
        yi1 = int(np.clip(np.ceil((c[1] + ext_y - grid.y_range[0]) / vs), 0, grid.grid_h))
        zi0 = int(np.clip(np.floor((c[2] - hz - grid.z_range[0]) / vs), 0, grid.grid_z))
        zi1 = int(np.clip(np.ceil((c[2] + hz - grid.z_range[0]) / vs), 0, grid.grid_z))
        if xi1 <= xi0 or yi1 <= yi0 or zi1 <= zi0:
            continue
        xs, ys, zs = xg[xi0:xi1], yg[yi0:yi1], zg[zi0:zi1]
        zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
        dx, dy = xx - c[0], yy - c[1]
        ca, sa = np.cos(yaw), np.sin(yaw)
        bx = ca * dx + sa * dy
        by = -sa * dx + ca * dy
        bz = zz - c[2]
        out[zi0:zi1, yi0:yi1, xi0:xi1] |= (
            (np.abs(bx) <= hx) & (np.abs(by) <= hy) & (np.abs(bz) <= hz)
        )
    return out


def shift_ref_se3(
    ref_t: np.ndarray,
    ref_R: np.ndarray,
    esdf: np.ndarray,
    grid: BEVGrid,
    r_need: float,
    z_body: float = 0.7,
    max_shift: float = 3.5,
    n_samples: int = 21,
    origin_xy: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide each SE(3) knot along its reference +Y (left) onto the highest 3D ESDF.

    Translation moves in the pose frame; rotation is unchanged. ``origin_xy``
    is unused when occupancy is already vehicle-center FLU.
    """
    t = np.asarray(ref_t, dtype=np.float64).reshape(-1, 3).copy()
    R = np.asarray(ref_R, dtype=np.float64).reshape(-1, 3, 3).copy()
    if len(t) == 0:
        return t, R
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    ys = np.linspace(-max_shift, max_shift, n_samples)
    offsets = np.stack(
        [np.zeros(n_samples), ys, np.full(n_samples, z_body)], axis=1,
    )
    for k in range(len(t)):
        if float(t[k, 0]) < 2.5:
            continue
        pts = t[k] + offsets @ R[k].T
        pts[:, 0] -= ox
        pts[:, 1] -= oy
        d = query_esdf(esdf, grid, pts)
        d_here = float(d[n_samples // 2])
        i = int(np.argmax(d))
        if float(d[i]) >= r_need and (d_here < r_need or float(d[i]) > d_here + 0.15):
            t[k] = t[k] + R[k] @ np.array([0.0, float(ys[i]), 0.0])
    return t, R


def shift_ref_from_esdf(
    p_ref: np.ndarray,
    esdf: np.ndarray,
    grid: BEVGrid,
    r_need: float,
    z: float = 0.7,
    max_shift: float = 3.5,
    n_samples: int = 21,
    origin_xy: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Planar wrapper: shift XY knots using identity-Z poses along the path heading."""
    out = np.asarray(p_ref, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[1] != 2 or len(out) == 0:
        return out
    n = len(out)
    t = np.zeros((n, 3), dtype=np.float64)
    t[:, :2] = out
    if n >= 2:
        dxy = np.diff(out, axis=0)
        yaw = np.arctan2(dxy[:, 1], dxy[:, 0])
        yaw = np.concatenate([yaw[:1], yaw])
    else:
        yaw = np.zeros(n)
    R = np.stack([rotz(float(y)) for y in yaw], axis=0)
    t2, _ = shift_ref_se3(
        t, R, esdf, grid, r_need, z_body=z, max_shift=max_shift,
        n_samples=n_samples, origin_xy=origin_xy,
    )
    return t2[:, :2]


def occupied_to_esdf(
    occupied: np.ndarray,
    grid: BEVGrid,
    max_dist: float | None = None,
) -> np.ndarray:
    """ESDF in meters from an already-built 3D obstacle mask."""
    occupied = np.asarray(occupied).astype(bool)
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
    return occupied_to_esdf(occupied, grid, max_dist=max_dist)


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


def query_esdf_z_slack(
    esdf: np.ndarray,
    grid: BEVGrid,
    points: np.ndarray,
    z_slack: float = 0.30,
    n_z: int = 5,
) -> np.ndarray:
    """
    ESDF at each point, taking the max over occupancy-Z in ±z_slack.

    A ball may sit slightly higher or lower so a sloped road does not count
    as collision; XY obstacles still block.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    slack = float(z_slack)
    if slack <= 0.0 or n_z <= 1:
        return query_esdf(esdf, grid, pts)
    d = None
    for dz in np.linspace(-slack, slack, int(n_z)):
        q = pts.copy()
        q[:, 2] = q[:, 2] + dz
        di = query_esdf(esdf, grid, q)
        d = di if d is None else np.maximum(d, di)
    return d


def shift_path_for_balls(
    path: np.ndarray,
    esdf: np.ndarray,
    grid: BEVGrid,
    body: np.ndarray,
    radii: np.ndarray,
    r_need: float = 0.10,
    max_shift: float = 3.5,
    z_slack: float = 0.30,
    n_samples: int = 29,
) -> np.ndarray:
    """Slide each skeleton knot in its left/right axis until the balls are clear."""
    from .vehicle_body import transform_body_se3

    path = np.asarray(path, dtype=np.float64).reshape(-1, 3).copy()
    body = np.asarray(body, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    if len(path) < 2:
        return path
    dxy = np.diff(path[:, :2], axis=0)
    yaw = np.arctan2(dxy[:, 1], dxy[:, 0])
    yaw = np.concatenate([yaw[:1], yaw])
    ys = np.linspace(-max_shift, max_shift, int(n_samples))

    def _clearance(t, R):
        balls = transform_body_se3(t, R, body)
        d = query_esdf_z_slack(esdf, grid, balls, z_slack=z_slack) - radii
        return float(np.min(d))

    for k in range(1, len(path)):
        R = rotz(float(yaw[k]))
        t0 = path[k]
        best_t = t0
        best_d = _clearance(t0, R)
        if best_d >= float(r_need):
            continue
        chosen = None
        for yoff in sorted(ys, key=lambda y: abs(float(y))):
            t = t0 + R @ np.array([0.0, float(yoff), 0.0])
            if (
                t[0] < grid.x_range[0] + 0.2
                or t[0] > grid.x_range[1] - 0.2
                or t[1] < grid.y_range[0] + 0.2
                or t[1] > grid.y_range[1] - 0.2
            ):
                continue
            d = _clearance(t, R)
            if d >= float(r_need):
                chosen = t
                break
            if d > best_d:
                best_d = d
                best_t = t
        path[k] = chosen if chosen is not None else best_t
    return path


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
