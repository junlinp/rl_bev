"""5 s target pose and knot reference from CARLA GlobalRoutePlanner.

CARLA vehicle frame is X forward, Y right, Z up (left-handed yaw).
Ego / BEV / NMPC frame is X forward, Y left, Z up (right-handed yaw).
World→ego conversion flips Y and negates relative yaw.
"""

from __future__ import annotations

import math
import numpy as np


def wrap_angle(yaw: np.ndarray | float) -> np.ndarray | float:
    """Wrap to (-pi, pi]."""
    return (np.asarray(yaw, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def world_to_ego_bev(
    wx: np.ndarray,
    wy: np.ndarray,
    ego_xy_yaw: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    CARLA world XY → ego BEV XY (Y left).

    ego_xy_yaw: (x, y, yaw_rad) of the vehicle in CARLA world (yaw is CARLA's).
    """
    ex, ey, eyaw = ego_xy_yaw
    dx = np.asarray(wx, dtype=np.float64) - ex
    dy = np.asarray(wy, dtype=np.float64) - ey
    c, s = math.cos(eyaw), math.sin(eyaw)
    x_veh = c * dx + s * dy
    y_veh = -s * dx + c * dy  # CARLA Y-right
    return x_veh, -y_veh


def yaw_world_to_ego(yaw_world: np.ndarray | float, ego_yaw_world: float) -> np.ndarray | float:
    """CARLA world yaw (rad) → relative ego-frame yaw (Y-left, right-handed)."""
    return wrap_angle(-(np.asarray(yaw_world, dtype=np.float64) - ego_yaw_world))


def ego_bev_to_world(
    x: np.ndarray | float,
    y: np.ndarray | float,
    ego_xy_yaw: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Ego BEV XY (Y left) → CARLA world XY."""
    ex, ey, eyaw = ego_xy_yaw
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_veh, y_veh = x, -y
    c, s = math.cos(eyaw), math.sin(eyaw)
    wx = ex + c * x_veh - s * y_veh
    wy = ey + s * x_veh + c * y_veh
    return wx, wy


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
        """Current global route as ego-frame XY and yaw."""
        tf = self.vehicle.get_transform()
        ego = (tf.location.x, tf.location.y, math.radians(tf.rotation.yaw))
        if not self._route:
            return np.zeros((1, 2)), np.zeros(1)
        wx = np.array([wp.transform.location.x for wp, _ in self._route], dtype=np.float64)
        wy = np.array([wp.transform.location.y for wp, _ in self._route], dtype=np.float64)
        wyaw = np.radians([wp.transform.rotation.yaw for wp, _ in self._route])
        ex, ey = world_to_ego_bev(wx, wy, ego)
        xy = np.stack([ex, ey], axis=1)
        yaw = np.array([yaw_world_to_ego(w, ego[2]) for w in wyaw], dtype=np.float64)
        # drop points well behind the vehicle
        ahead = xy[:, 0] > -2.0
        if ahead.sum() >= 2:
            xy, yaw = xy[ahead], yaw[ahead]
        return xy, yaw

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
