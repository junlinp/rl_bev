"""5 s target pose and knot reference from CARLA GlobalRoutePlanner.

CARLA vehicle frame is X forward, Y right, Z up.
Ego / BEV / NMPC frame is X forward, Y left, Z up.

World waypoints are mapped through the vehicle SE(3) pose, then Y is
flipped into the ego BEV frame.
"""

from __future__ import annotations

import math
import numpy as np


def wrap_angle(yaw: np.ndarray | float) -> np.ndarray | float:
    """Wrap to (-pi, pi]."""
    return (np.asarray(yaw, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def carla_rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """CARLA/Unreal vehicle-local → world rotation. Angles in radians."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp, -cp * sr, cp * cr],
    ], dtype=np.float64)


def _as_se3_pose(pose) -> tuple[np.ndarray, np.ndarray]:
    """
    World-from-vehicle SE(3) as (t, R).

    pose may be:
      (x, y, yaw)                              — planar, radians
      (x, y, z, pitch, yaw, roll)              — radians
      (4, 4) matrix
      CARLA Transform (uses get_matrix when present)
    """
    if hasattr(pose, "get_matrix"):
        T = np.array(pose.get_matrix(), dtype=np.float64)
        return T[:3, 3].copy(), T[:3, :3].copy()
    if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
        return pose[:3, 3].astype(np.float64).copy(), pose[:3, :3].astype(np.float64).copy()
    if hasattr(pose, "location") and hasattr(pose, "rotation"):
        loc, rot = pose.location, pose.rotation
        t = np.array([loc.x, loc.y, loc.z], dtype=np.float64)
        R = carla_rotation_matrix(
            math.radians(rot.pitch), math.radians(rot.yaw), math.radians(rot.roll),
        )
        return t, R
    arr = tuple(float(v) for v in pose)
    if len(arr) == 3:
        x, y, yaw = arr
        return np.array([x, y, 0.0], dtype=np.float64), carla_rotation_matrix(0.0, yaw, 0.0)
    if len(arr) == 6:
        x, y, z, pitch, yaw, roll = arr
        return np.array([x, y, z], dtype=np.float64), carla_rotation_matrix(pitch, yaw, roll)
    raise TypeError(f"unsupported SE(3) pose: {type(pose)!r}")


_YFLIP = np.diag([1.0, -1.0, 1.0]).astype(np.float64)


def rotz(yaw: float) -> np.ndarray:
    """Ego-FLU yaw: +ψ toward +Y (left)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def yaw_from_R(R: np.ndarray) -> float:
    """Heading about +Z from an ego-FLU rotation."""
    return float(wrap_angle(math.atan2(float(R[1, 0]), float(R[0, 0]))))


def se3_from_planar(x: float, y: float, yaw: float, z: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    return np.array([x, y, z], dtype=np.float64), rotz(float(yaw))


def world_se3_to_ego(t_w: np.ndarray, R_w: np.ndarray, pose) -> tuple[np.ndarray, np.ndarray]:
    """
    World waypoint SE(3) → ego FLU (X forward, Y left, Z up).

    CARLA vehicle Y is right; conjugate by the Y-flip so the result matches
    the occupancy / NMPC frame.
    """
    T_w = np.eye(4, dtype=np.float64)
    T_w[:3, :3] = np.asarray(R_w, dtype=np.float64)
    T_w[:3, 3] = np.asarray(t_w, dtype=np.float64).reshape(3)
    if hasattr(pose, "get_inverse_matrix"):
        T_veh = np.array(pose.get_inverse_matrix(), dtype=np.float64) @ T_w
    else:
        t_v, R_v = _as_se3_pose(pose)
        T_inv = np.eye(4, dtype=np.float64)
        T_inv[:3, :3] = R_v.T
        T_inv[:3, 3] = -R_v.T @ t_v
        T_veh = T_inv @ T_w
    F = np.eye(4, dtype=np.float64)
    F[:3, :3] = _YFLIP
    T_ego = F @ T_veh @ F
    return T_ego[:3, 3].copy(), T_ego[:3, :3].copy()


def log_so3(R: np.ndarray) -> np.ndarray:
    """Axis-angle (rotation vector) of R ∈ SO(3)."""
    tr = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(tr)
    if theta < 1e-8:
        return 0.5 * np.array([
            R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1],
        ], dtype=np.float64)
    w = np.array([
        R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1],
    ], dtype=np.float64) / (2.0 * math.sin(theta))
    return theta * w


def exp_so3(rv: np.ndarray) -> np.ndarray:
    """Rodrigues: rotation vector → SO(3)."""
    rv = np.asarray(rv, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    k = rv / theta
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ], dtype=np.float64)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def interpolate_se3(
    xyz: np.ndarray,
    Rs: np.ndarray,
    s: np.ndarray,
    s_query: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Lerp translation / slerp rotation along arclength."""
    s = np.asarray(s, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    Rs = np.asarray(Rs, dtype=np.float64)
    sq = float(np.clip(s_query, s[0], s[-1]))
    if s[-1] - s[0] < 1e-9:
        return xyz[0].copy(), Rs[0].copy()
    i1 = int(np.clip(np.searchsorted(s, sq, side="right"), 1, len(s) - 1))
    i0 = i1 - 1
    span = s[i1] - s[i0]
    a = 0.0 if span < 1e-9 else (sq - s[i0]) / span
    t = (1.0 - a) * xyz[i0] + a * xyz[i1]
    Rrel = Rs[i0].T @ Rs[i1]
    R = Rs[i0] @ exp_so3(a * log_so3(Rrel))
    return t, R


def interpolate_se3_path(
    xyz: np.ndarray,
    Rs: np.ndarray,
    s: np.ndarray,
    s_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``interpolate_se3`` at every arclength in ``s_query``."""
    s_query = np.atleast_1d(np.asarray(s_query, dtype=np.float64))
    ts = np.empty((s_query.size, 3), dtype=np.float64)
    Rs_out = np.empty((s_query.size, 3, 3), dtype=np.float64)
    for i, sq in enumerate(s_query):
        ts[i], Rs_out[i] = interpolate_se3(xyz, Rs, s, float(sq))
    return ts, Rs_out


def pose_from_carla_transform(tf) -> tuple[float, float, float, float, float, float]:
    """CARLA Transform → (x, y, z, pitch, yaw, roll) with angles in radians."""
    loc, rot = tf.location, tf.rotation
    return (
        float(loc.x), float(loc.y), float(loc.z),
        math.radians(rot.pitch), math.radians(rot.yaw), math.radians(rot.roll),
    )


def world_to_ego_bev(
    wx: np.ndarray,
    wy: np.ndarray,
    pose,
    wz: np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    CARLA world points → ego BEV XY (Y left) via the vehicle SE(3) pose.

    If `wz` is omitted, points are taken at the vehicle origin height so a
    yaw-only pose matches the old planar transform.
    """
    wx = np.asarray(wx, dtype=np.float64)
    wy = np.asarray(wy, dtype=np.float64)
    shape = wx.shape
    if hasattr(pose, "get_inverse_matrix"):
        if wz is None:
            loc = pose.location
            wz = np.full(wx.shape, float(loc.z), dtype=np.float64)
        else:
            wz = np.broadcast_to(np.asarray(wz, dtype=np.float64), wx.shape)
        hom = np.stack([
            wx.reshape(-1), wy.reshape(-1), np.asarray(wz, dtype=np.float64).reshape(-1),
            np.ones(wx.size, dtype=np.float64),
        ], axis=0)
        veh = np.array(pose.get_inverse_matrix(), dtype=np.float64) @ hom
        return veh[0].reshape(shape), (-veh[1]).reshape(shape)

    t, R = _as_se3_pose(pose)
    if wz is None:
        wz = np.full_like(wx, t[2])
    else:
        wz = np.broadcast_to(np.asarray(wz, dtype=np.float64), np.shape(wx))
    pts = np.stack([wx, wy, np.asarray(wz, dtype=np.float64)], axis=0).reshape(3, -1)
    veh = R.T @ (pts - t.reshape(3, 1))
    return veh[0].reshape(shape), (-veh[1]).reshape(shape)


def yaw_world_to_ego(yaw_world: np.ndarray | float, ego_yaw_world: float) -> np.ndarray | float:
    """CARLA world yaw (rad) → relative ego-frame yaw (Y-left, right-handed)."""
    return wrap_angle(-(np.asarray(yaw_world, dtype=np.float64) - ego_yaw_world))


def world_heading_to_ego(
    wyaw: np.ndarray,
    pose,
    wpitch: np.ndarray | float | None = None,
    wroll: np.ndarray | float | None = None,
) -> np.ndarray:
    """Waypoint heading → ego yaw by rotating the forward axis through SE(3)."""
    wyaw = np.asarray(wyaw, dtype=np.float64).reshape(-1)
    n = int(wyaw.size)
    if n == 0:
        return wyaw
    if wpitch is None:
        wpitch = np.zeros(n, dtype=np.float64)
    if wroll is None:
        wroll = np.zeros(n, dtype=np.float64)
    wpitch = np.broadcast_to(np.asarray(wpitch, dtype=np.float64).reshape(-1), (n,))
    wroll = np.broadcast_to(np.asarray(wroll, dtype=np.float64).reshape(-1), (n,))
    if hasattr(pose, "get_inverse_matrix"):
        R_vw = np.array(pose.get_inverse_matrix(), dtype=np.float64)[:3, :3]
    else:
        _t, R_v = _as_se3_pose(pose)
        R_vw = R_v.T
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        fwd_w = carla_rotation_matrix(float(wpitch[i]), float(wyaw[i]), float(wroll[i]))[:, 0]
        fwd_v = R_vw @ fwd_w
        out[i] = math.atan2(-fwd_v[1], fwd_v[0])
    return wrap_angle(out)


def ego_bev_to_world(
    x: np.ndarray | float,
    y: np.ndarray | float,
    pose,
    z: np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ego BEV XY (Y left) → CARLA world XY via the vehicle SE(3) pose."""
    t, R = _as_se3_pose(pose)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if z is None:
        z = np.zeros_like(x, dtype=np.float64)
    else:
        z = np.broadcast_to(np.asarray(z, dtype=np.float64), np.shape(x))
    veh = np.stack([x, -np.asarray(y, dtype=np.float64), np.asarray(z, dtype=np.float64)], axis=0).reshape(3, -1)
    world = (R @ veh) + t.reshape(3, 1)
    return world[0].reshape(x.shape), world[1].reshape(x.shape)


def ego_yaw_to_world(psi_ego: np.ndarray | float, ego_yaw_world: float) -> np.ndarray | float:
    """Ego-frame yaw (Y-left) → CARLA world yaw (rad)."""
    return wrap_angle(ego_yaw_world - np.asarray(psi_ego, dtype=np.float64))


def transform_points_between_ego(
    pts: np.ndarray,
    last_ego_xy_yaw: tuple[float, float, float],
    cur_ego_xy_yaw: tuple[float, float, float],
) -> np.ndarray:
    """Transform (N, 3) ego-FLU points from last ego frame into the current one."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return pts
    wx, wy = ego_bev_to_world(pts[:, 0], pts[:, 1], last_ego_xy_yaw, z=pts[:, 2])
    x, y = world_to_ego_bev(wx, wy, cur_ego_xy_yaw, wz=pts[:, 2])
    out = pts.copy()
    out[:, 0] = np.asarray(x, dtype=np.float64).reshape(-1)
    out[:, 1] = np.asarray(y, dtype=np.float64).reshape(-1)
    return out


def transform_states_between_ego(
    states: np.ndarray,
    last_ego_xy_yaw: tuple[float, float, float],
    cur_ego_xy_yaw: tuple[float, float, float],
) -> np.ndarray:
    """
    Transform (4, K) or (K, 4) states [x, y, psi, v] from last ego frame
    into the current ego frame. v is unchanged.
    """
    arr = np.asarray(states, dtype=np.float64)
    transpose = arr.ndim == 2 and arr.shape[0] != 4 and arr.shape[1] == 4
    if transpose:
        arr = arr.T
    wx, wy = ego_bev_to_world(arr[0], arr[1], last_ego_xy_yaw)
    yaw_w = ego_yaw_to_world(arr[2], last_ego_xy_yaw[2])
    x, y = world_to_ego_bev(wx, wy, cur_ego_xy_yaw)
    psi = yaw_world_to_ego(yaw_w, cur_ego_xy_yaw[2])
    out = arr.copy()
    out[0], out[1], out[2] = np.asarray(x), np.asarray(y), np.asarray(psi)
    return out.T if transpose else out


def remaining_route_index(
    wx: np.ndarray,
    wy: np.ndarray,
    pose,
    behind_slack_m: float = 0.0,
    wz: np.ndarray | float | None = None,
    along_window_m: float = 15.0,
) -> int:
    """
    Start index of the still-active route.

    Search only the next ``along_window_m`` of the polyline after dropping
    clearly-behind points. A later loop that comes back beside the car must
    not win a global nearest-waypoint contest — that collapses the 5 s
    target onto the hood.
    """
    wx = np.asarray(wx, dtype=np.float64).reshape(-1)
    wy = np.asarray(wy, dtype=np.float64).reshape(-1)
    if wx.size == 0:
        return 0
    x, y = world_to_ego_bev(wx, wy, pose, wz=wz)
    n = int(x.size)
    i_lo = 0
    while i_lo < n - 1 and float(x[i_lo]) < -2.0:
        i_lo += 1
    pts = np.stack([wx, wy], axis=1)
    s = _cum_arclength(pts)
    s_lo = float(s[i_lo])
    d2 = x * x + y * y
    win = (s >= s_lo - 1e-6) & (s <= s_lo + float(along_window_m))
    if not np.any(win):
        win[i_lo] = True
    i0 = int(np.argmin(np.where(win, d2, np.inf)))
    while i0 < n - 1 and float(x[i0]) < -behind_slack_m:
        i0 += 1
    return i0


def _closest_on_polyline(
    xy: np.ndarray, along_window_m: float = 8.0,
) -> tuple[int, float, np.ndarray]:
    """Closest point on the next ``along_window_m`` of the polyline to the origin."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    s = _cum_arclength(xy)
    best_i, best_t, best_d = 0, 0.0, float("inf")
    best_q = xy[0].copy()
    for i in range(len(xy) - 1):
        if float(s[i]) >= float(along_window_m):
            break
        a = xy[i]
        ab = xy[i + 1] - a
        l2 = float(np.dot(ab, ab))
        if l2 < 1e-12:
            t = 0.0
            q = a
        else:
            t = float(np.clip(np.dot(-a, ab) / l2, 0.0, 1.0))
            q = a + t * ab
        d = float(np.hypot(q[0], q[1]))
        if d < best_d:
            best_d, best_i, best_t, best_q = d, i, t, q
    return best_i, best_t, best_q


def _cut_route_at_closest(
    xy: np.ndarray,
    yaw: np.ndarray,
    xyz: np.ndarray | None = None,
    Rs: np.ndarray | None = None,
) -> tuple:
    """
    Start the route at the closest point on the lane, not at the vehicle.

    Prepending the ego origin makes a chord to the nearest waypoint. If that
    waypoint is beside the car, NMPC has to yaw ~90° even when the 5 s target
    is already ahead on the lane.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    yaw = np.asarray(yaw, dtype=np.float64).reshape(-1)
    if xyz is not None:
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if Rs is not None:
        Rs = np.asarray(Rs, dtype=np.float64).reshape(-1, 3, 3)
    if len(xy) == 0:
        return xy, yaw, xyz, Rs
    if len(xy) == 1:
        return xy, yaw, xyz, Rs

    i, t, q = _closest_on_polyline(xy)
    if float(q[0]) < -0.25:
        ahead = np.where(xy[:, 0] >= 0.0)[0]
        i0 = int(ahead[0]) if ahead.size else max(len(xy) - 1, 0)
        xy, yaw = xy[i0:], yaw[i0:]
        if xyz is not None:
            xyz = xyz[i0:]
        if Rs is not None:
            Rs = Rs[i0:]
        return xy, yaw, xyz, Rs

    if t >= 1.0 - 1e-8:
        i0 = min(i + 1, len(xy) - 1)
        xy, yaw = xy[i0:], yaw[i0:]
        if xyz is not None:
            xyz = xyz[i0:]
        if Rs is not None:
            Rs = Rs[i0:]
        return xy, yaw, xyz, Rs

    yaw_unw = np.unwrap([float(yaw[i]), float(yaw[i + 1])])
    yaw_q = float(wrap_angle((1.0 - t) * yaw_unw[0] + t * yaw_unw[1]))
    xy = np.vstack([q.reshape(1, 2), xy[i + 1:]])
    yaw = np.concatenate([np.array([yaw_q]), yaw[i + 1:]])
    if xyz is not None:
        t_q = (1.0 - t) * xyz[i] + t * xyz[i + 1]
        t_q[:2] = q
        xyz = np.vstack([t_q.reshape(1, 3), xyz[i + 1:]])
    if Rs is not None:
        Rrel = Rs[i].T @ Rs[i + 1]
        R_q = Rs[i] @ exp_so3(t * log_so3(Rrel))
        Rs = np.concatenate([R_q[None, ...], Rs[i + 1:]], axis=0)
    return xy, yaw, xyz, Rs


def route_world_to_ego(
    wx: np.ndarray,
    wy: np.ndarray,
    wyaw: np.ndarray,
    pose,
    wz: np.ndarray | float | None = None,
    wpitch: np.ndarray | float | None = None,
    wroll: np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Remaining global route in ego BEV XY / yaw, starting at the vehicle."""
    wx = np.asarray(wx, dtype=np.float64).reshape(-1)
    wy = np.asarray(wy, dtype=np.float64).reshape(-1)
    wyaw = np.asarray(wyaw, dtype=np.float64).reshape(-1)
    if wx.size == 0:
        return np.zeros((1, 2)), np.zeros(1)
    if wz is not None:
        wz = np.asarray(wz, dtype=np.float64).reshape(-1)
    i0 = remaining_route_index(wx, wy, pose, wz=wz)
    wx, wy, wyaw = wx[i0:], wy[i0:], wyaw[i0:]
    if wz is not None:
        wz = wz[i0:]
    if wpitch is not None:
        wpitch = np.asarray(wpitch, dtype=np.float64).reshape(-1)[i0:]
    if wroll is not None:
        wroll = np.asarray(wroll, dtype=np.float64).reshape(-1)[i0:]
    x, y = world_to_ego_bev(wx, wy, pose, wz=wz)
    xy = np.stack([x, y], axis=1)
    yaw = world_heading_to_ego(wyaw, pose, wpitch=wpitch, wroll=wroll)
    xy, yaw, _, _ = _cut_route_at_closest(xy, yaw)
    return xy, yaw


def route_world_to_ego_se3(
    wx, wy, wz, wpitch, wyaw, wroll, pose,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remaining route as ego XY, yaw, XYZ, and SO(3) in FLU."""
    wx = np.asarray(wx, dtype=np.float64).reshape(-1)
    wy = np.asarray(wy, dtype=np.float64).reshape(-1)
    wz = np.asarray(wz, dtype=np.float64).reshape(-1)
    wpitch = np.asarray(wpitch, dtype=np.float64).reshape(-1)
    wyaw = np.asarray(wyaw, dtype=np.float64).reshape(-1)
    wroll = np.asarray(wroll, dtype=np.float64).reshape(-1)
    n = int(wx.size)
    if n == 0:
        return (
            np.zeros((1, 2)), np.zeros(1),
            np.zeros((1, 3)), np.eye(3, dtype=np.float64)[None, ...],
        )
    i0 = remaining_route_index(wx, wy, pose, wz=wz)
    ts, Rs = [], []
    for i in range(i0, n):
        t_w = np.array([wx[i], wy[i], wz[i]], dtype=np.float64)
        R_w = carla_rotation_matrix(float(wpitch[i]), float(wyaw[i]), float(wroll[i]))
        t_e, R_e = world_se3_to_ego(t_w, R_w, pose)
        ts.append(t_e)
        Rs.append(R_e)
    xyz = np.stack(ts, axis=0)
    Rs = np.stack(Rs, axis=0)
    xy = xyz[:, :2].copy()
    yaw = np.array([yaw_from_R(R) for R in Rs], dtype=np.float64)
    xy, yaw, xyz, Rs = _cut_route_at_closest(xy, yaw, xyz, Rs)
    return xy, yaw, xyz, Rs


def _cum_arclength(xy: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s


def interpolate_polyline(xy: np.ndarray, s: np.ndarray, s_query: np.ndarray) -> np.ndarray:
    """Linear interpolate a polyline at arclengths s_query."""
    s_query = np.clip(np.asarray(s_query, dtype=np.float64), s[0], s[-1])
    x = np.interp(s_query, s, xy[:, 0])
    y = np.interp(s_query, s, xy[:, 1])
    return np.stack([x, y], axis=1)


def interpolate_yaw(yaw: np.ndarray, s: np.ndarray, s_query: np.ndarray) -> np.ndarray:
    """Interpolate yaw via unwrapped angles."""
    unwrapped = np.unwrap(np.asarray(yaw, dtype=np.float64))
    s_query = np.clip(np.asarray(s_query, dtype=np.float64), s[0], s[-1])
    return wrap_angle(np.interp(s_query, s, unwrapped))


def sample_target_and_reference(
    route_ego_xy: np.ndarray,
    route_ego_yaw: np.ndarray,
    v: float,
    horizon_s: float = 5.0,
    n_knots: int = 26,
    d_min: float = 8.0,
    d_max: float = 25.0,
    route_xyz: np.ndarray | None = None,
    route_R: np.ndarray | None = None,
) -> dict:
    """
    From an ego-frame route, pick the 5 s SE(3) target and per-knot SE(3) reference.

    ``v`` is the planning speed used for lookahead (typically ``v_max``),
    not the vehicle's current speed: target arclength = clip(v * horizon_s, d_min, d_max).
    """
    xy = np.asarray(route_ego_xy, dtype=np.float64)
    yaw = np.asarray(route_ego_yaw, dtype=np.float64)
    if (
        xy.shape[0] >= 2
        and float(np.hypot(xy[0, 0], xy[0, 1])) < 1e-6
        and abs(float(xy[1, 1])) > abs(float(xy[1, 0])) + 0.05
        and float(np.hypot(xy[1, 0], xy[1, 1])) > 0.15
    ):
        xy, yaw = xy[1:], yaw[1:]
        if route_xyz is not None and route_R is not None and len(route_xyz) == len(yaw) + 1:
            route_xyz = np.asarray(route_xyz, dtype=np.float64)[1:]
            route_R = np.asarray(route_R, dtype=np.float64)[1:]
    if xy.shape[0] < 2:
        target = np.array([d_min, 0.0, 0.0], dtype=np.float64)
        t, R = se3_from_planar(d_min, 0.0, 0.0)
        p_ref = np.stack([
            np.linspace(0.0, target[0], n_knots),
            np.zeros(n_knots),
        ], axis=1)
        yaw_ref = np.zeros(n_knots)
        ref_t = np.column_stack([p_ref, np.zeros(n_knots)])
        ref_R = np.repeat(np.eye(3, dtype=np.float64)[None, ...], n_knots, axis=0)
        ref_t[-1] = t
        ref_R[-1] = R
        return {
            "target": target, "target_t": t, "target_R": R,
            "p_ref": p_ref, "yaw_ref": yaw_ref,
            "ref_t": ref_t, "ref_R": ref_R,
        }

    s = _cum_arclength(xy)
    v_plan = max(float(v), 0.0)
    lookahead = float(np.clip(v_plan * horizon_s, d_min, d_max))
    # If the remaining polyline collapsed (loop snap, 1–2 stacked WPs), do
    # not interpolate at s=0 — that puts the 5 s pose on the vehicle.
    if float(s[-1]) < float(d_min) - 1e-6:
        t, R = se3_from_planar(d_min, 0.0, 0.0)
        p_ref = np.stack([
            np.linspace(0.0, d_min, n_knots),
            np.zeros(n_knots),
        ], axis=1)
        yaw_ref = np.zeros(n_knots)
        ref_t = np.column_stack([p_ref, np.zeros(n_knots)])
        ref_R = np.repeat(np.eye(3, dtype=np.float64)[None, ...], n_knots, axis=0)
        ref_t[-1], ref_R[-1] = t, R
        target = np.array([d_min, 0.0, 0.0], dtype=np.float64)
        return {
            "target": target, "target_t": t, "target_R": R,
            "p_ref": p_ref, "yaw_ref": yaw_ref,
            "ref_t": ref_t, "ref_R": ref_R,
        }
    lookahead = float(min(lookahead, s[-1]))

    dt = horizon_s / max(n_knots - 1, 1)
    s_knots = np.linspace(0.0, lookahead, n_knots)

    if route_xyz is not None and route_R is not None and len(route_xyz) == len(xy):
        ref_t, ref_R = interpolate_se3_path(route_xyz, route_R, s, s_knots)
        t_tgt, R_tgt = ref_t[-1].copy(), ref_R[-1].copy()
    else:
        p_ref = interpolate_polyline(xy, s, s_knots)
        yaw_ref = interpolate_yaw(yaw, s, s_knots)
        t_tgt, R_tgt = se3_from_planar(
            float(p_ref[-1, 0]), float(p_ref[-1, 1]), float(yaw_ref[-1]),
        )
        ref_t = np.column_stack([p_ref, np.zeros(n_knots)])
        ref_R = np.stack([rotz(float(y)) for y in yaw_ref], axis=0)

    ref_t[-1], ref_R[-1] = t_tgt, R_tgt
    p_ref = ref_t[:, :2].copy()
    yaw_ref = np.array([yaw_from_R(Rk) for Rk in ref_R], dtype=np.float64)
    target = np.array([t_tgt[0], t_tgt[1], yaw_from_R(R_tgt)], dtype=np.float64)
    return {
        "target": target, "target_t": t_tgt, "target_R": R_tgt,
        "p_ref": p_ref, "yaw_ref": yaw_ref,
        "ref_t": ref_t, "ref_R": ref_R,
    }


def _import_global_route_planner():
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        return GlobalRoutePlanner
    except ImportError:
        pass
    import os
    import sys
    extra = os.environ.get("CARLA_PYTHONAPI")
    if extra and extra not in sys.path:
        sys.path.append(extra)
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        return GlobalRoutePlanner
    raise ImportError(
        "CARLA GlobalRoutePlanner not found. Install the carla Python API "
        "or set CARLA_PYTHONAPI to the PythonAPI/carla folder."
    )


class GlobalTarget:
    """Tracks a CARLA global route and samples the 5 s ego-frame target pose."""

    def __init__(
        self,
        world,
        vehicle,
        horizon_s: float = 5.0,
        sampling_resolution: float = 1.0,
        d_min: float = 5.0,
        d_max: float = 25.0,
        v_max: float = 2.0,
        replan_remaining_m: float = 15.0,
    ):
        self.world = world
        self.vehicle = vehicle
        self.horizon_s = horizon_s
        self.d_min = d_min
        self.d_max = d_max
        self.v_max = float(v_max)
        self.replan_remaining_m = replan_remaining_m
        self._map = world.get_map()
        GRP = _import_global_route_planner()
        self._grp = GRP(self._map, sampling_resolution)
        self._route = []  # list of (waypoint, RoadOption)
        self._destination = None
        self.pick_destination()

    def pick_destination(self) -> None:
        loc = self.vehicle.get_location()
        spawns = self._map.get_spawn_points()
        if not spawns:
            raise RuntimeError("map has no spawn points")
        dest = max(spawns, key=lambda t: loc.distance(t.location))
        if loc.distance(dest.location) < 20.0:
            dest = spawns[len(spawns) // 2]
        self._destination = dest.location
        self._route = self._grp.trace_route(loc, self._destination)

    def _remaining_arclength(self) -> float:
        if len(self._route) < 2:
            return 0.0
        tf = self.vehicle.get_transform()
        wx = np.array([wp.transform.location.x for wp, _ in self._route], dtype=np.float64)
        wy = np.array([wp.transform.location.y for wp, _ in self._route], dtype=np.float64)
        wz = np.array([wp.transform.location.z for wp, _ in self._route], dtype=np.float64)
        pts = np.stack([wx, wy], axis=1)
        i0 = remaining_route_index(wx, wy, tf, wz=wz)
        s = _cum_arclength(pts[i0:])
        return float(s[-1]) if s.size else 0.0

    def route_in_ego(self):
        """Current remaining global route as ego XY, yaw, XYZ, and SO(3)."""
        tf = self.vehicle.get_transform()
        if not self._route:
            return (
                np.zeros((1, 2)), np.zeros(1),
                np.zeros((1, 3)), np.eye(3, dtype=np.float64)[None, ...],
            )
        wx = np.array([wp.transform.location.x for wp, _ in self._route], dtype=np.float64)
        wy = np.array([wp.transform.location.y for wp, _ in self._route], dtype=np.float64)
        wz = np.array([wp.transform.location.z for wp, _ in self._route], dtype=np.float64)
        wyaw = np.radians([wp.transform.rotation.yaw for wp, _ in self._route])
        wpitch = np.radians([wp.transform.rotation.pitch for wp, _ in self._route])
        wroll = np.radians([wp.transform.rotation.roll for wp, _ in self._route])
        i0 = remaining_route_index(wx, wy, tf, wz=wz)
        if i0:
            self._route = self._route[i0:]
            wx, wy, wz = wx[i0:], wy[i0:], wz[i0:]
            wyaw, wpitch, wroll = wyaw[i0:], wpitch[i0:], wroll[i0:]
        return route_world_to_ego_se3(
            wx, wy, wz, wpitch, wyaw, wroll, tf,
        )

    def update(self, speed: float, n_knots: int) -> dict:
        if self._remaining_arclength() < self.replan_remaining_m:
            self.pick_destination()
        xy, yaw, xyz, Rs = self.route_in_ego()
        out = sample_target_and_reference(
            xy, yaw, v=self.v_max, horizon_s=self.horizon_s,
            n_knots=n_knots, d_min=self.d_min, d_max=self.d_max,
            route_xyz=xyz, route_R=Rs,
        )
        out["route_ego"] = xy
        out["route_xyz"] = xyz
        return out
