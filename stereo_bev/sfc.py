"""Safe Flight Corridor: 3D A* skeleton + overlapping convex polyhedra.

Liu et al., IEEE RA-L 2017 / FASTER (Tordesillas & How, arXiv:2001.04420):
a piecewise-linear path in free voxels, then one H-rep polyhedron per
segment from an inscribed ellipsoid and obstacle tangent planes.

NMPC stays inside the union of polyhedra; collision is linear halfspaces
instead of a nonconvex ESDF homotopy search.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from .bev_grid import BEVGrid
from .global_target import rotz, wrap_angle
from .occ_field import esdf_axis_grids, occupied_to_esdf, shift_path_for_balls

# CasADi packs a fixed number of faces per knot.
SFC_N_FACES = 16
_SFC_INACTIVE_B = 1.0e6


def _ijk_to_xyz(iz: int, iy: int, ix: int, grid: BEVGrid) -> np.ndarray:
    vs = grid.voxel_size
    return np.array([
        grid.x_range[0] + (ix + 0.5) * vs,
        grid.y_range[0] + (iy + 0.5) * vs,
        grid.z_range[0] + (iz + 0.5) * vs,
    ], dtype=np.float64)


def _xyz_to_ijk(p: np.ndarray, grid: BEVGrid) -> tuple[int, int, int]:
    vs = grid.voxel_size
    ix = int(np.floor((float(p[0]) - grid.x_range[0]) / vs))
    iy = int(np.floor((float(p[1]) - grid.y_range[0]) / vs))
    iz = int(np.floor((float(p[2]) - grid.z_range[0]) / vs))
    return iz, iy, ix


def _in_grid(iz: int, iy: int, ix: int, grid: BEVGrid) -> bool:
    return (
        0 <= ix < grid.grid_w
        and 0 <= iy < grid.grid_h
        and 0 <= iz < grid.grid_z
    )


def _snap_free(
    p: np.ndarray,
    blocked: np.ndarray,
    grid: BEVGrid,
    max_r: int = 12,
) -> np.ndarray | None:
    """Nearest free voxel center to ``p`` (search expanding cube)."""
    iz0, iy0, ix0 = _xyz_to_ijk(p, grid)
    if _in_grid(iz0, iy0, ix0, grid) and not blocked[iz0, iy0, ix0]:
        return _ijk_to_xyz(iz0, iy0, ix0, grid)
    best = None
    best_d = 1e9
    for r in range(1, max_r + 1):
        for dz in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy), abs(dz)) != r:
                        continue
                    iz, iy, ix = iz0 + dz, iy0 + dy, ix0 + dx
                    if not _in_grid(iz, iy, ix, grid) or blocked[iz, iy, ix]:
                        continue
                    q = _ijk_to_xyz(iz, iy, ix, grid)
                    d = float(np.linalg.norm(q - p))
                    if d < best_d:
                        best_d = d
                        best = q
        if best is not None:
            return best
    return None


def _occupied_points(occupied: np.ndarray, grid: BEVGrid) -> np.ndarray:
    zi, yi, xi = np.nonzero(occupied)
    if xi.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    xg, yg, zg = esdf_axis_grids(grid)
    return np.stack([xg[xi], yg[yi], zg[zi]], axis=1)


def astar_3d(
    blocked: np.ndarray,
    grid: BEVGrid,
    start: np.ndarray,
    goal: np.ndarray,
    z_lo: float = 0.35,
    z_hi: float = 1.55,
    bias: np.ndarray | None = None,
    bias_w: float = 2.5,
) -> np.ndarray | None:
    """26-connected A* on the occupancy grid. Returns voxel-center XYZ or None.

    ``z_lo``/``z_hi`` keep a ground vehicle in a driving slab so the search
    cannot tunnel under pavement or climb over a car roof.
    """
    p0 = _snap_free(np.asarray(start, dtype=np.float64).reshape(3), blocked, grid)
    p1 = _snap_free(np.asarray(goal, dtype=np.float64).reshape(3), blocked, grid)
    if p0 is None or p1 is None:
        return None
    s = _xyz_to_ijk(p0, grid)
    g = _xyz_to_ijk(p1, grid)
    if s == g:
        return np.stack([p0, p1], axis=0)

    nx, ny, nz = grid.grid_w, grid.grid_h, grid.grid_z
    nflat = nx * ny * nz

    def pack(iz, iy, ix):
        return ix + nx * (iy + ny * iz)

    def unpack(u):
        ix = u % nx
        t = u // nx
        iy = t % ny
        iz = t // ny
        return iz, iy, ix

    vs = float(grid.voxel_size)
    nbrs = [
        (dz, dy, dx, vs * math.sqrt(dx * dx + dy * dy + dz * dz))
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if dx or dy or dz
    ]
    g_goal = pack(*g)
    s_id = pack(*s)

    def h_of(u):
        iz, iy, ix = unpack(u)
        return vs * math.sqrt(
            (ix - g[2]) ** 2 + (iy - g[1]) ** 2 + (iz - g[0]) ** 2
        )

    inf = 1e18
    gscore = np.full(nflat, inf, dtype=np.float64)
    parent = np.full(nflat, -1, dtype=np.int32)
    gscore[s_id] = 0.0
    open_heap = [(h_of(s_id), 0.0, s_id)]
    closed = np.zeros(nflat, dtype=bool)
    best_u, best_h = s_id, h_of(s_id)

    while open_heap:
        _, gc, u = heapq.heappop(open_heap)
        if closed[u]:
            continue
        if gc > gscore[u] + 1e-9:
            continue
        closed[u] = True
        hu = h_of(u)
        if hu < best_h:
            best_h = hu
            best_u = u
        if u == g_goal:
            best_u = u
            break
        iz, iy, ix = unpack(u)
        for dz, dy, dx, step in nbrs:
            jz, jy, jx = iz + dz, iy + dy, ix + dx
            if not _in_grid(jz, jy, jx, grid) or blocked[jz, jy, jx]:
                continue
            zc = grid.z_range[0] + (jz + 0.5) * vs
            if zc < z_lo or zc > z_hi:
                continue
            v = pack(jz, jy, jx)
            if closed[v]:
                continue
            ng = gscore[u] + step
            if bias is not None:
                ng = ng + float(bias_w) * float(bias[jz, jy, jx])
            if ng + 1e-9 < gscore[v]:
                gscore[v] = ng
                parent[v] = u
                heapq.heappush(open_heap, (ng + h_of(v), ng, v))

    if parent[best_u] < 0 and best_u != s_id:
        return None
    if best_u != g_goal and best_h > 3.0:
        return None

    chain = []
    u = int(best_u)
    seen = 0
    while u >= 0 and seen < nflat:
        chain.append(unpack(u))
        if u == s_id:
            break
        u = int(parent[u])
        seen += 1
    chain.reverse()
    pts = np.stack([_ijk_to_xyz(iz, iy, ix, grid) for iz, iy, ix in chain], axis=0)
    pts[0] = p0
    pts[-1] = _ijk_to_xyz(*unpack(int(best_u)), grid) if best_u != g_goal else p1
    if best_u == g_goal:
        pts[-1] = p1
    return pts


def _line_free(
    p0: np.ndarray,
    p1: np.ndarray,
    blocked: np.ndarray,
    grid: BEVGrid,
) -> bool:
    vs = float(grid.voxel_size)
    dist = float(np.linalg.norm(p1 - p0))
    n = max(2, int(math.ceil(dist / (0.45 * vs))) + 1)
    for a in np.linspace(0.0, 1.0, n):
        p = p0 + a * (p1 - p0)
        iz, iy, ix = _xyz_to_ijk(p, grid)
        if not _in_grid(iz, iy, ix, grid) or blocked[iz, iy, ix]:
            return False
    return True


def shortcut_path(
    path: np.ndarray,
    blocked: np.ndarray,
    grid: BEVGrid,
) -> np.ndarray:
    """Greedy line-of-sight shortcut on a voxel path."""
    path = np.asarray(path, dtype=np.float64).reshape(-1, 3)
    if len(path) < 3:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _line_free(path[i], path[j], blocked, grid):
            j -= 1
        out.append(path[j])
        i = j
    return np.stack(out, axis=0)


def _resample_segments(path: np.ndarray, min_len: float = 1.2, max_len: float = 3.5) -> np.ndarray:
    """Keep corners and split long edges so each segment is a polyhedron seed."""
    path = np.asarray(path, dtype=np.float64).reshape(-1, 3)
    if len(path) < 2:
        return path
    out = [path[0]]
    for i in range(1, len(path)):
        p0 = out[-1]
        p1 = path[i]
        d = float(np.linalg.norm(p1 - p0))
        if d < 1e-6:
            continue
        n = max(1, int(math.ceil(d / max_len)))
        for k in range(1, n + 1):
            q = p0 + (k / n) * (p1 - p0)
            if float(np.linalg.norm(q - out[-1])) >= min_len or k == n:
                out.append(q)
    return np.stack(out, axis=0)


def _pad_path_for_body(
    path: np.ndarray,
    grid: BEVGrid,
    half_length: float = 2.4,
) -> np.ndarray:
    """Extend the skeleton so the first/last polyhedra cover the vehicle box."""
    path = np.asarray(path, dtype=np.float64).reshape(-1, 3)
    if len(path) < 2:
        return path
    out = path
    d0 = path[1] - path[0]
    n0 = float(np.linalg.norm(d0))
    if n0 > 1e-6:
        behind = path[0] - half_length * (d0 / n0)
        behind[0] = float(np.clip(behind[0], grid.x_range[0] + 0.15, grid.x_range[1] - 0.15))
        behind[1] = float(np.clip(behind[1], grid.y_range[0] + 0.15, grid.y_range[1] - 0.15))
        behind[2] = float(np.clip(behind[2], grid.z_range[0] + 0.15, grid.z_range[1] - 0.15))
        out = np.vstack([behind, out])
    d1 = path[-1] - path[-2]
    n1 = float(np.linalg.norm(d1))
    if n1 > 1e-6:
        ahead = path[-1] + half_length * (d1 / n1)
        ahead[0] = float(np.clip(ahead[0], grid.x_range[0] + 0.15, grid.x_range[1] - 0.15))
        ahead[1] = float(np.clip(ahead[1], grid.y_range[0] + 0.15, grid.y_range[1] - 0.15))
        ahead[2] = float(np.clip(ahead[2], grid.z_range[0] + 0.15, grid.z_range[1] - 0.15))
        out = np.vstack([out, ahead])
    return out


def _orthonormal_frame(axis: np.ndarray) -> np.ndarray:
    x = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(x))
    if n < 1e-9:
        return np.eye(3, dtype=np.float64)
    x = x / n
    helper = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    y = np.cross(helper, x)
    y = y / (np.linalg.norm(y) + 1e-12)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def _volume_planes(grid: BEVGrid, margin: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax = grid.x_range
    ymin, ymax = grid.y_range
    zmin, zmax = grid.z_range
    m = float(margin)
    A = np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)
    b = np.array([
        xmax - m, -(xmin + m),
        ymax - m, -(ymin + m),
        zmax - m, -(zmin + m),
    ], dtype=np.float64)
    return A, b


def _append_plane(
    planes: list[tuple[np.ndarray, float]],
    n: np.ndarray,
    offset: float,
    p0: np.ndarray,
    p1: np.ndarray,
    eps: float = 1e-4,
) -> None:
    n = np.asarray(n, dtype=np.float64).reshape(3)
    nrm = float(np.linalg.norm(n))
    if nrm < 1e-9:
        return
    n = n / nrm
    off = float(offset) / nrm if nrm != 1.0 else float(offset)
    # Keep the segment strictly inside; relax if a vertex sits on the wrong side.
    for p in (p0, p1):
        viol = float(n @ p - off)
        if viol > 0.0:
            off += viol + eps
    planes.append((n, off))


def ellipsoid_polyhedron(
    p0: np.ndarray,
    p1: np.ndarray,
    obs: np.ndarray,
    grid: BEVGrid,
    n_faces: int = SFC_N_FACES,
    max_radius: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One convex polyhedron around segment p0→p1 (Liu / DecompUtil).

    Grow the largest obstacle-free ellipsoid aligned with the segment, take
    supporting planes of that ellipsoid plus polar/tangent planes at nearby
    occupied voxels, clip to the occupancy AABB.
    """
    p0 = np.asarray(p0, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    d = 0.5 * (p0 + p1)
    axis = p1 - p0
    half = 0.5 * float(np.linalg.norm(axis))
    a_axis = max(half + 0.15, 0.35)
    R = _orthonormal_frame(axis if half > 1e-6 else np.array([1.0, 0.0, 0.0]))

    Av, bv = _volume_planes(grid)
    planes: list[tuple[np.ndarray, float]] = []
    for n, off in zip(Av, bv):
        _append_plane(planes, n, off, p0, p1)

    if obs is None or len(obs) == 0:
        b_lat = min(max_radius, 3.5)
    else:
        rel = obs - d
        lo, hi = 0.20, float(max_radius)
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            Cinv = R @ np.diag([1.0 / (a_axis * a_axis), 1.0 / (mid * mid), 1.0 / (mid * mid)]) @ R.T
            q = np.einsum("ni,ij,nj->n", rel, Cinv, rel)
            if np.any(q <= 1.0):
                hi = mid
            else:
                lo = mid
        b_lat = lo

    # Supporting planes of the ellipsoid (oriented bounding slab).
    for i, s in enumerate((a_axis, b_lat, b_lat)):
        n = R[:, i]
        _append_plane(planes, n, n @ d + s, p0, p1)
        _append_plane(planes, -n, (-n) @ d + s, p0, p1)

    vs = float(grid.voxel_size)
    if obs is not None and len(obs) > 0:
        Cinv = R @ np.diag(
            [1.0 / (a_axis * a_axis), 1.0 / (b_lat * b_lat), 1.0 / (b_lat * b_lat)]
        ) @ R.T
        rel = obs - d
        q = np.einsum("ni,ij,nj->n", rel, Cinv, rel)
        near = np.flatnonzero((q > 1.0) & (q < 9.0))
        if near.size:
            order = near[np.argsort(q[near])]
            added = 0
            for idx in order:
                if added >= max(4, n_faces // 2):
                    break
                p = obs[idx]
                n = Cinv @ (p - d)
                nrm = float(np.linalg.norm(n))
                if nrm < 1e-9:
                    continue
                n = n / nrm
                off = float(n @ p) - 0.5 * vs
                _append_plane(planes, n, off, p0, p1)
                added += 1

    A = np.stack([n for n, _ in planes], axis=0)
    b = np.array([off for _, off in planes], dtype=np.float64)
    # Tiny inward shrink, then put the seed segment back inside so the
    # skeleton remains a feasible point for the hard QP.
    b = b - 0.02
    for i in range(len(b)):
        for p in (p0, p1):
            viol = float(A[i] @ p - b[i])
            if viol > 0.0:
                b[i] += viol + 1e-4
    A, b = _prune_planes(A, b, d, n_faces)
    return A, b


def _prune_planes(
    A: np.ndarray,
    b: np.ndarray,
    center: np.ndarray,
    n_faces: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the tightest unique halfspaces, always including AABB if present."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if len(b) <= n_faces:
        return A, b
    slack = b - A @ center
    keep = []
    used = np.zeros(len(b), dtype=bool)
    # First 6 rows are volume planes in ellipsoid_polyhedron.
    n_keep_first = min(6, len(b))
    for i in range(n_keep_first):
        keep.append(i)
        used[i] = True
    rest = np.argsort(slack)
    for i in rest:
        if used[i]:
            continue
        n = A[i]
        dup = False
        for j in keep:
            if float(np.dot(n, A[j])) > 0.97 and abs(slack[i] - slack[j]) < 0.15:
                dup = True
                break
        if dup:
            continue
        keep.append(int(i))
        used[i] = True
        if len(keep) >= n_faces:
            break
    keep = np.array(keep[:n_faces], dtype=np.int32)
    return A[keep], b[keep]


def polyhedra_from_path(
    path: np.ndarray,
    occupied: np.ndarray,
    grid: BEVGrid,
    n_faces: int = SFC_N_FACES,
    overlap: float = 0.8,
) -> list[tuple[np.ndarray, np.ndarray]]:
    path = np.asarray(path, dtype=np.float64).reshape(-1, 3)
    if len(path) < 2:
        A, b = _volume_planes(grid)
        return [(A, b)]
    obs = _occupied_points(occupied, grid)
    out = []
    for i in range(len(path) - 1):
        p0, p1 = path[i], path[i + 1]
        axis = p1 - p0
        nrm = float(np.linalg.norm(axis))
        if nrm > 1e-6 and overlap > 0.0:
            u = axis / nrm
            p0e = p0 - float(overlap) * u
            p1e = p1 + float(overlap) * u
        else:
            p0e, p1e = p0, p1
        A, b = ellipsoid_polyhedron(p0e, p1e, obs, grid, n_faces=n_faces)
        out.append((A, b))
    return out


def point_in_polyhedron(p: np.ndarray, A: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> bool:
    return bool(np.all(A @ np.asarray(p, dtype=np.float64).reshape(3) <= b + tol))


def _project_segment(p: np.ndarray, path: np.ndarray) -> int:
    """Index of the polyline segment that contains the closest point to ``p``.

    At shared vertices the forward segment is preferred so early knots are not
    locked into the next cell's rear face (which makes the QP empty).
    """
    p = np.asarray(p, dtype=np.float64).reshape(3)
    path = np.asarray(path, dtype=np.float64).reshape(-1, 3)
    n_seg = len(path) - 1
    best_i, best_d, best_s = 0, 1e18, 1.0
    for i in range(n_seg):
        a, b = path[i], path[i + 1]
        ab = b - a
        l2 = float(ab @ ab)
        if l2 < 1e-12:
            s, q = 0.0, a
        else:
            s = float(np.clip(((p - a) @ ab) / l2, 0.0, 1.0))
            q = a + s * ab
        d = float(np.linalg.norm(p - q))
        take = d < best_d - 1e-9
        if not take and abs(d - best_d) <= 1e-9:
            # Prefer an interior / forward segment over one that ends at p.
            take = (s < 0.999 and best_s >= 0.999) or (s < 0.999 and i > best_i)
        if take:
            best_d, best_i, best_s = d, i, s
    return best_i


def assign_polyhedra(
    ref_t: np.ndarray,
    path: np.ndarray,
    polyhedra: list[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Assign each knot the path-segment polyhedron that contains it."""
    ref_t = np.asarray(ref_t, dtype=np.float64).reshape(-1, 3)
    path = np.asarray(path, dtype=np.float64).reshape(-1, 3)
    n_seg = max(len(polyhedra), 1)
    if len(path) < 2:
        return [polyhedra[0]] * len(ref_t)
    out = []
    for t in ref_t:
        seg = min(_project_segment(t, path), n_seg - 1)
        best_i, best_slack = seg, -1e18
        for j in range(max(0, seg - 1), min(n_seg, seg + 2)):
            A, b = polyhedra[j]
            slack = float(np.min(b - A @ t))
            if slack > best_slack:
                best_slack, best_i = slack, j
        if best_slack < -1e-3:
            for j, (A, b) in enumerate(polyhedra):
                slack = float(np.min(b - A @ t))
                if slack > best_slack:
                    best_slack, best_i = slack, j
        out.append(polyhedra[best_i])
    return out


def pack_sfc_parameters(
    knot_polyhedra: list[tuple[np.ndarray, np.ndarray]] | None,
    n_knots: int,
    n_faces: int = SFC_N_FACES,
) -> tuple[np.ndarray, np.ndarray]:
    """CasADi parameters: poly_A (n_faces*3, K) C-order of (F,3), poly_b (F, K)."""
    poly_A = np.zeros((n_faces * 3, n_knots), dtype=np.float64)
    poly_b = np.full((n_faces, n_knots), _SFC_INACTIVE_B, dtype=np.float64)
    if not knot_polyhedra:
        return poly_A, poly_b
    for k in range(min(n_knots, len(knot_polyhedra))):
        A, b = knot_polyhedra[k]
        A = np.asarray(A, dtype=np.float64).reshape(-1, 3)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        F = min(n_faces, len(b))
        padded = np.zeros((n_faces, 3), dtype=np.float64)
        padded[:F] = A[:F]
        poly_A[:, k] = padded.ravel()
        poly_b[:F, k] = b[:F]
    return poly_A, poly_b


def refs_along_path(
    path: np.ndarray,
    n_knots: int,
    v: float,
    horizon_s: float,
    d_min: float = 5.0,
    d_max: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """SE(3) knot references sampled along the SFC skeleton."""
    path = np.asarray(path, dtype=np.float64).reshape(-1, 3)
    n_knots = int(n_knots)
    if len(path) == 0:
        t = np.zeros((n_knots, 3), dtype=np.float64)
        t[:, 0] = np.linspace(0.0, d_min, n_knots)
        R = np.repeat(np.eye(3)[None, ...], n_knots, axis=0)
        return t, R
    if len(path) == 1:
        t = np.repeat(path[0][None, :], n_knots, axis=0)
        R = np.repeat(np.eye(3)[None, ...], n_knots, axis=0)
        return t, R
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    v_plan = max(float(v), 0.0)
    lookahead = float(np.clip(v_plan * horizon_s, d_min, d_max))
    lookahead = float(min(lookahead, s[-1]))
    s_knots = np.linspace(0.0, lookahead, n_knots)
    t = np.stack([
        np.interp(s_knots, s, path[:, 0]),
        np.interp(s_knots, s, path[:, 1]),
        np.interp(s_knots, s, path[:, 2]),
    ], axis=1)
    yaw_v = np.arctan2(np.diff(path[:, 1], prepend=path[0, 1]), np.diff(path[:, 0], prepend=path[0, 0]))
    yaw_v[0] = yaw_v[1] if len(yaw_v) > 1 else 0.0
    yaw_v = np.unwrap(yaw_v)
    yaw = np.interp(s_knots, s, yaw_v)
    R = np.stack([rotz(float(wrap_angle(y))) for y in yaw], axis=0)
    t[0] = 0.0
    t[0, 2] = path[0, 2]
    return t, R


def polyhedron_wireframe(
    A: np.ndarray,
    b: np.ndarray,
    grid: BEVGrid | None = None,
) -> np.ndarray:
    """Vertex-vertex edges of an H-rep polyhedron, shape (E, 2, 3)."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    F = len(b)
    verts = []
    keys = []
    for i in range(F):
        for j in range(i + 1, F):
            for k in range(j + 1, F):
                M = A[[i, j, k]]
                det = float(np.linalg.det(M))
                if abs(det) < 1e-8:
                    continue
                try:
                    x = np.linalg.solve(M, b[[i, j, k]])
                except np.linalg.LinAlgError:
                    continue
                if not np.all(np.isfinite(x)):
                    continue
                if grid is not None:
                    if (
                        x[0] < grid.x_range[0] - 0.5 or x[0] > grid.x_range[1] + 0.5
                        or x[1] < grid.y_range[0] - 0.5 or x[1] > grid.y_range[1] + 0.5
                        or x[2] < grid.z_range[0] - 0.5 or x[2] > grid.z_range[1] + 0.5
                    ):
                        continue
                if np.all(A @ x <= b + 1e-4):
                    verts.append(x)
                    keys.append(tuple(sorted((i, j, k))))
    if len(verts) < 2:
        return np.zeros((0, 2, 3), dtype=np.float64)
    verts = np.asarray(verts, dtype=np.float64)
    # Dedup
    uniq = []
    for v in verts:
        if not uniq or min(float(np.linalg.norm(v - u)) for u in uniq) > 1e-3:
            uniq.append(v)
    verts = np.stack(uniq, axis=0)
    edges = []
    seen = set()
    for a in range(len(verts)):
        d = np.linalg.norm(verts - verts[a], axis=1)
        nn = np.argsort(d)[1:min(7, len(verts))]
        for bidx in nn:
            key = (min(a, int(bidx)), max(a, int(bidx)))
            if key in seen:
                continue
            seen.add(key)
            edges.append(np.stack([verts[a], verts[int(bidx)]], axis=0))
    if not edges:
        return np.zeros((0, 2, 3), dtype=np.float64)
    return np.stack(edges, axis=0)


def _path_bias_field(path: np.ndarray, grid: BEVGrid) -> np.ndarray | None:
    """ESDF of the previous skeleton, used as an A* extra cost."""
    path = np.asarray(path, dtype=np.float64).reshape(-1, 3)
    if len(path) == 0:
        return None
    mask = np.zeros((grid.grid_z, grid.grid_h, grid.grid_w), dtype=bool)
    for p in path:
        iz, iy, ix = _xyz_to_ijk(p, grid)
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    jz, jy, jx = iz + dz, iy + dy, ix + dx
                    if _in_grid(jz, jy, jx, grid):
                        mask[jz, jy, jx] = True
    if not mask.any():
        return None
    return occupied_to_esdf(mask, grid)


def _adopt_preferred_path(
    prefer: np.ndarray,
    blocked: np.ndarray,
    grid: BEVGrid,
    start: np.ndarray,
    goal: np.ndarray,
    z_lo: float,
    z_hi: float,
) -> np.ndarray | None:
    """Reuse last frame's skeleton if it is still free in the inflated map."""
    p = np.asarray(prefer, dtype=np.float64).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 2:
        return None
    kept = []
    for q in p:
        if float(q[0]) < -0.4:
            continue
        iz, iy, ix = _xyz_to_ijk(q, grid)
        if not _in_grid(iz, iy, ix, grid) or blocked[iz, iy, ix]:
            if len(kept) < 2:
                continue
            break
        kept.append(q)
    if len(kept) < 2:
        return None
    path = np.vstack([np.asarray(start, dtype=np.float64).reshape(1, 3), np.stack(kept, axis=0)])
    uniq = [path[0]]
    for q in path[1:]:
        if float(np.linalg.norm(q - uniq[-1])) > 0.08:
            uniq.append(q)
    path = np.stack(uniq, axis=0)
    if len(path) < 2:
        return None
    for i in range(len(path) - 1):
        if not _line_free(path[i], path[i + 1], blocked, grid):
            return None
    if float(np.linalg.norm(path[-1, :2] - np.asarray(goal[:2], dtype=np.float64))) > 2.0:
        tail = astar_3d(blocked, grid, path[-1], goal, z_lo=z_lo, z_hi=z_hi)
        if tail is not None and len(tail) >= 2:
            join = np.vstack([path, tail[1:]])
            if _line_free(path[-1], tail[0], blocked, grid) or _line_free(path[-1], tail[1], blocked, grid):
                path = join
    return path


def build_safe_flight_corridor(
    occupied: np.ndarray,
    grid: BEVGrid,
    start: np.ndarray,
    goal: np.ndarray,
    r_inflate: float = 0.94,
    esdf: np.ndarray | None = None,
    n_faces: int = SFC_N_FACES,
    z_body: float = 0.70,
    z_lo: float = 0.35,
    z_hi: float = 1.55,
    prefer_path: np.ndarray | None = None,
    body: np.ndarray | None = None,
    radii: np.ndarray | None = None,
    r_clear: float = 0.10,
) -> dict:
    """
    Front-end path + overlapping 3D polyhedra from start to goal.

    ``occupied`` is the planning mask (ground already cleared). A* runs on
    a spherical inflation of radius ``r_inflate`` (COM as a point robot).
    Polyhedra are grown in the uninflated occupancy so the overlapping
    H-rep cells stay fat enough for a hard linearized-bicycle QP.
    ``prefer_path`` is the previous-frame skeleton in the current ego frame;
    it is reused when still free, otherwise A* is biased onto that homotopy.
    """
    occupied = np.asarray(occupied).astype(bool)
    start = np.asarray(start, dtype=np.float64).reshape(3).copy()
    goal = np.asarray(goal, dtype=np.float64).reshape(3).copy()
    if start[2] < grid.z_range[0] + 0.2:
        start[2] = float(z_body)
    if goal[2] < grid.z_range[0] + 0.2:
        goal[2] = float(z_body)

    if esdf is None:
        esdf = occupied_to_esdf(occupied, grid)
    blocked = esdf < float(r_inflate)
    xg, yg, zg = esdf_axis_grids(grid)
    blocked = blocked.copy()
    blocked[(zg < float(z_lo)) | (zg > float(z_hi)), :, :] = True

    empty = {
        "ok": False,
        "path": np.zeros((0, 3), dtype=np.float64),
        "polyhedra": [],
        "n_poly": 0,
    }
    raw = None
    if prefer_path is not None and len(np.asarray(prefer_path).reshape(-1, 3)) >= 2:
        raw = _adopt_preferred_path(
            prefer_path, blocked, grid, start, goal, z_lo=z_lo, z_hi=z_hi,
        )
    if raw is None:
        bias = None
        if prefer_path is not None:
            bias = _path_bias_field(prefer_path, grid)
        raw = astar_3d(
            blocked, grid, start, goal, z_lo=z_lo, z_hi=z_hi, bias=bias,
        )
    if raw is None or len(raw) < 2:
        return empty
    path = shortcut_path(raw, blocked, grid)
    path = _resample_segments(path)
    if len(path) < 2:
        return empty
    # Keep the geometric start at the vehicle (identity knot).
    path[0] = start
    if body is not None and radii is not None:
        path = shift_path_for_balls(
            path, esdf, grid, body, radii, r_need=float(r_clear),
        )
        path[0] = start
    polys = polyhedra_from_path(path, occupied, grid, n_faces=n_faces)
    return {
        "ok": True,
        "path": path,
        "path_poly": path,
        "polyhedra": polys,
        "n_poly": len(polys),
        "raw_path": raw,
    }
