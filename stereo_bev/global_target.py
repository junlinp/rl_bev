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
) -> int:
    """
    Start index of the still-active route.

    Closest waypoint, then skip any that sit behind the vehicle. A heading
    change must not revive already-passed waypoints that have merely rotated
    into x > 0.
    """
    wx = np.asarray(wx, dtype=np.float64).reshape(-1)
    wy = np.asarray(wy, dtype=np.float64).reshape(-1)
    if wx.size == 0:
        return 0
    x, y = world_to_ego_bev(wx, wy, pose, wz=wz)
    i0 = int(np.argmin(x * x + y * y))
    n = int(x.size)
    while i0 < n - 1 and float(x[i0]) < -behind_slack_m:
        i0 += 1
    return i0


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
    if xy.shape[0] == 0 or float(np.hypot(xy[0, 0], xy[0, 1])) > 0.15:
        xy = np.vstack([np.zeros((1, 2)), xy])
        yaw = np.concatenate([np.zeros(1), yaw])
    else:
        xy[0] = (0.0, 0.0)
        yaw[0] = 0.0
    return xy, yaw


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
) -> dict:
    """
    From an ego-frame route polyline, pick the 5 s target and per-knot reference.

    Args:
        route_ego_xy: (M, 2) waypoints in ego BEV frame
        route_ego_yaw: (M,) yaw at each waypoint (ego frame)
        v: current speed (m/s)
        n_knots: N+1 including the current pose (index 0)

    Returns dict with target (x, y, yaw), p_ref (n_knots, 2), yaw_ref (n_knots,)
    """
    xy = np.asarray(route_ego_xy, dtype=np.float64)
    yaw = np.asarray(route_ego_yaw, dtype=np.float64)
    if xy.shape[0] < 2:
        target = np.array([d_min, 0.0, 0.0], dtype=np.float64)
        p_ref = np.stack([
            np.linspace(0.0, target[0], n_knots),
            np.zeros(n_knots),
        ], axis=1)
        return {"target": target, "p_ref": p_ref, "yaw_ref": np.zeros(n_knots)}

    s = _cum_arclength(xy)
    v_plan = max(float(v), 1.0)
    lookahead = float(np.clip(v_plan * horizon_s, d_min, d_max))
    lookahead = float(min(lookahead, s[-1])) if s[-1] > 1e-3 else lookahead

    target_xy = interpolate_polyline(xy, s, np.array([lookahead]))[0]
    target_yaw = float(interpolate_yaw(yaw, s, np.array([lookahead]))[0])

    dt = horizon_s / max(n_knots - 1, 1)
    s_knots = np.clip(np.arange(n_knots, dtype=np.float64) * v_plan * dt, 0.0, lookahead)
    # Always stretch the last knot onto the 5 s target so terminal cost matches
    s_knots[-1] = lookahead
    p_ref = interpolate_polyline(xy, s, s_knots)
    yaw_ref = interpolate_yaw(yaw, s, s_knots)
    target = np.array([target_xy[0], target_xy[1], target_yaw], dtype=np.float64)
    return {"target": target, "p_ref": p_ref, "yaw_ref": yaw_ref}


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
        replan_remaining_m: float = 15.0,
    ):
        self.world = world
        self.vehicle = vehicle
        self.horizon_s = horizon_s
        self.d_min = d_min
        self.d_max = d_max
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
        loc = self.vehicle.get_location()
        pts = np.array(
            [[wp.transform.location.x, wp.transform.location.y] for wp, _ in self._route],
            dtype=np.float64,
        )
        d = np.linalg.norm(pts - np.array([loc.x, loc.y]), axis=1)
        i0 = int(np.argmin(d))
        s = _cum_arclength(pts[i0:])
        return float(s[-1]) if s.size else 0.0

    def route_in_ego(self) -> tuple[np.ndarray, np.ndarray]:
        """Current remaining global route as ego-frame XY and yaw."""
        tf = self.vehicle.get_transform()
        if not self._route:
            return np.zeros((1, 2)), np.zeros(1)
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
        return route_world_to_ego(
            wx, wy, wyaw, tf, wz=wz, wpitch=wpitch, wroll=wroll,
        )

    def update(self, speed: float, n_knots: int) -> dict:
        if self._remaining_arclength() < self.replan_remaining_m:
            self.pick_destination()
        xy, yaw = self.route_in_ego()
        out = sample_target_and_reference(
            xy, yaw, v=speed, horizon_s=self.horizon_s,
            n_knots=n_knots, d_min=self.d_min, d_max=self.d_max,
        )
        out["route_ego"] = xy
        return out
