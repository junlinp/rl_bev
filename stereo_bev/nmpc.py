"""CasADi kinematic-bicycle NMPC with a 3D occupancy ESDF cost."""

from __future__ import annotations

import time
import numpy as np

from .bev_grid import BEVGrid
from .global_target import transform_states_between_ego, wrap_angle
from .occ_field import esdf_axis_grids, flatten_esdf_casadi, query_esdf
from .vehicle_body import MODEL3_WHEELBASE, transform_body

try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    ca = None
    HAS_CASADI = False


class OccupancyNMPC:
    """
    Receding-horizon kinematic bicycle in the ego frame.

    State z = [x, y, psi, v], control u = [a, delta], T = 5 s.
    Occupancy cost is a 3D ESDF queried at vehicle body sample points.
    """

    def __init__(
        self,
        grid: BEVGrid,
        body: np.ndarray,
        horizon_s: float = 5.0,
        dt: float = 0.2,
        wheelbase: float = MODEL3_WHEELBASE,
        r_safe: float = 0.30,
        d0: float = 0.40,
        a_max: float = 3.0,
        delta_max: float = 0.50,
        v_max: float = 1.0,
        w_pos: float = 2.0,
        w_yaw: float = 0.4,
        w_term_pos: float = 5.0,
        w_term_yaw: float = 1.5,
        w_u: float = 0.08,
        w_du: float = 1.5,
        w_v: float = 0.6,
        w_occ: float = 160.0,
        ipopt_max_iter: int = 80,
        stitch_s: float = 0.2,
    ):
        if not HAS_CASADI:
            raise ImportError("casadi is required for OccupancyNMPC (pip install casadi)")

        self.grid = grid
        self.body = np.asarray(body, dtype=np.float64)
        self.horizon_s = horizon_s
        self.dt = dt
        self.N = int(round(horizon_s / dt))
        self.L = wheelbase
        self.r_safe = r_safe
        self.d0 = d0
        self.a_max = a_max
        self.delta_max = delta_max
        self.v_max = v_max
        self.stitch_s = stitch_s
        self.n_body = self.body.shape[0]
        self.n_vox = grid.grid_z * grid.grid_h * grid.grid_w
        self.nx, self.ny, self.nz = grid.grid_w, grid.grid_h, grid.grid_z

        xg, yg, zg = esdf_axis_grids(grid)
        self._xg, self._yg, self._zg = xg, yg, zg
        self._x_min, self._x_max = float(xg[0]), float(xg[-1])
        self._y_min, self._y_max = float(yg[0]), float(yg[-1])
        self._z_min, self._z_max = float(zg[0]), float(zg[-1])
        self._dx = float(grid.voxel_size)

        opti = ca.Opti()
        X = opti.variable(4, self.N + 1)
        U = opti.variable(2, self.N)
        x0 = opti.parameter(4)
        target = opti.parameter(3)
        p_ref = opti.parameter(2, self.N + 1)
        yaw_ref = opti.parameter(1, self.N + 1)
        v_ref = opti.parameter()
        esdf_p = opti.parameter(self.n_vox)

        x, y, psi, v = X[0, :], X[1, :], X[2, :], X[3, :]
        a, delta = U[0, :], U[1, :]

        opti.subject_to(X[:, 0] == x0)
        for k in range(self.N):
            zk = X[:, k]
            uk = U[:, k]
            k1 = self._f(zk, uk)
            k2 = self._f(zk + 0.5 * dt * k1, uk)
            k3 = self._f(zk + 0.5 * dt * k2, uk)
            k4 = self._f(zk + dt * k3, uk)
            zk1 = zk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            opti.subject_to(X[:, k + 1] == zk1)

        opti.subject_to(opti.bounded(0.0, v, v_max))
        opti.subject_to(opti.bounded(-a_max, a, a_max))
        opti.subject_to(opti.bounded(-delta_max, delta, delta_max))

        cost = 0
        for k in range(self.N + 1):
            cost = cost + w_pos * (
                (x[k] - p_ref[0, k]) ** 2 + (y[k] - p_ref[1, k]) ** 2
            )
            cost = cost + w_yaw * (psi[k] - yaw_ref[0, k]) ** 2
            cost = cost + w_v * (v[k] - v_ref) ** 2
            cost = cost + w_occ * self._occupancy_cost(x[k], y[k], psi[k], esdf_p)

        cost = cost + w_term_pos * (
            (x[self.N] - target[0]) ** 2 + (y[self.N] - target[1]) ** 2
        )
        cost = cost + w_term_yaw * (psi[self.N] - target[2]) ** 2

        for k in range(self.N):
            cost = cost + w_u * (a[k] ** 2 + delta[k] ** 2)
        for k in range(self.N - 1):
            cost = cost + w_du * (
                (a[k + 1] - a[k]) ** 2 + (delta[k + 1] - delta[k]) ** 2
            )

        opti.minimize(cost)
        opti.solver(
            "ipopt",
            {
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.max_iter": ipopt_max_iter,
                "ipopt.tol": 1e-3,
                "ipopt.acceptable_tol": 1e-2,
                "ipopt.acceptable_iter": 8,
                "ipopt.warm_start_init_point": "yes",
                "ipopt.sb": "yes",
                "ipopt.hessian_approximation": "limited-memory",
            },
        )

        self.opti = opti
        self.X, self.U = X, U
        self.x0, self.target = x0, target
        self.p_ref, self.yaw_ref = p_ref, yaw_ref
        self.v_ref = v_ref
        self.esdf_p = esdf_p
        self._last_X = None
        self._last_U = None
        self._last_u0 = np.zeros(2)
        self._last_ego_xy_yaw = None
        self._last_solve_s = None

    def _f(self, z, u):
        psi = z[2]
        v = z[3]
        a = u[0]
        delta = u[1]
        return ca.vertcat(
            v * ca.cos(psi),
            v * ca.sin(psi),
            v * ca.tan(delta) / self.L,
            a,
        )

    def _sample_esdf(self, esdf_p, px, py, pz):
        """Trilinear sample of flattened (Z,Y,X) ESDF at body points."""
        nx, ny, nz = self.nx, self.ny, self.nz
        tx = (px - self._x_min) / self._dx
        ty = (py - self._y_min) / self._dx
        tz = (pz - self._z_min) / self._dx
        ix0 = ca.fmin(ca.fmax(ca.floor(tx), 0), nx - 2)
        iy0 = ca.fmin(ca.fmax(ca.floor(ty), 0), ny - 2)
        iz0 = ca.fmin(ca.fmax(ca.floor(tz), 0), nz - 2)
        ax = ca.fmin(ca.fmax(tx - ix0, 0), 1)
        ay = ca.fmin(ca.fmax(ty - iy0, 0), 1)
        az = ca.fmin(ca.fmax(tz - iz0, 0), 1)
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        iz1 = iz0 + 1

        def corner(iz, iy, ix):
            return esdf_p[ix + nx * (iy + ny * iz)]

        c000 = corner(iz0, iy0, ix0)
        c100 = corner(iz0, iy0, ix1)
        c010 = corner(iz0, iy1, ix0)
        c110 = corner(iz0, iy1, ix1)
        c001 = corner(iz1, iy0, ix0)
        c101 = corner(iz1, iy0, ix1)
        c011 = corner(iz1, iy1, ix0)
        c111 = corner(iz1, iy1, ix1)
        c00 = c000 * (1 - ax) + c100 * ax
        c10 = c010 * (1 - ax) + c110 * ax
        c01 = c001 * (1 - ax) + c101 * ax
        c11 = c011 * (1 - ax) + c111 * ax
        c0 = c00 * (1 - ay) + c10 * ay
        c1 = c01 * (1 - ay) + c11 * ay
        return c0 * (1 - az) + c1 * az

    def _occupancy_cost(self, x, y, psi, esdf_p):
        c = ca.cos(psi)
        s = ca.sin(psi)
        bx = self.body[:, 0]
        by = self.body[:, 1]
        bz = self.body[:, 2]
        px = x + c * bx - s * by
        py = y + s * bx + c * by
        pz = bz
        px_c = ca.fmin(ca.fmax(px, self._x_min), self._x_max)
        py_c = ca.fmin(ca.fmax(py, self._y_min), self._y_max)
        pz_c = ca.fmin(ca.fmax(pz, self._z_min), self._z_max)
        d = ca.vec(self._sample_esdf(esdf_p, px_c, py_c, pz_c))
        in_vol = ca.vec(0.5 * (1 + ca.tanh((px - self._x_min) / 0.08)))
        gap = self.r_safe - d
        hinge = (ca.fmax(gap, 0) / self.d0) ** 2
        penetrate = (ca.fmax(-d, 0.0) / self.d0) ** 2
        return ca.sum1(in_vol * (hinge + 8.0 * penetrate))

    def _straight_guess(self, target: np.ndarray, v: float) -> tuple[np.ndarray, np.ndarray]:
        X = np.zeros((4, self.N + 1))
        U = np.zeros((2, self.N))
        for k in range(self.N + 1):
            t = k / self.N
            X[0, k] = target[0] * t
            X[1, k] = target[1] * t + 0.35 * np.sin(np.pi * t)
            X[2, k] = math_atan2(target[1], target[0]) if (target[0] ** 2 + target[1] ** 2) > 0.05 else 0.0
            X[3, k] = max(v, 0.5)
        return X, U

    def _interp_traj(self, X: np.ndarray, t_query: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        """Interpolate (4, K) trajectory at times t_query. Returns (4, len(t_query))."""
        if t is None:
            t = np.arange(X.shape[1], dtype=np.float64) * self.dt
        tq = np.clip(np.atleast_1d(np.asarray(t_query, dtype=np.float64)), t[0], t[-1])
        yaw = np.unwrap(X[2])
        return np.vstack([
            np.interp(tq, t, X[0]),
            np.interp(tq, t, X[1]),
            wrap_angle(np.interp(tq, t, yaw)),
            np.interp(tq, t, X[3]),
        ])

    def _interp_u(self, U: np.ndarray, t_query: np.ndarray) -> np.ndarray:
        """Zero-order hold of (2, N) controls at interval start times t_query."""
        tq = np.atleast_1d(np.asarray(t_query, dtype=np.float64))
        idx = np.clip(np.floor(tq / self.dt).astype(np.int32), 0, U.shape[1] - 1)
        return U[:, idx]

    def _to_current_ego(self, states: np.ndarray, ego_xy_yaw) -> np.ndarray:
        if ego_xy_yaw is None or self._last_ego_xy_yaw is None:
            return states
        return transform_states_between_ego(states, self._last_ego_xy_yaw, ego_xy_yaw)

    def _stitch_x0(self, v: float, ego_xy_yaw) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Start the new plan at the previous trajectory's ~200 ms knot,
        sampled from the last vehicle origin (not from the last virtual x0).
        """
        tau = self.stitch_s
        pred = self._to_current_ego(self._interp_traj(self._last_X, tau), ego_xy_yaw)[:, 0]
        if v < 0.4:
            x0 = np.array([0.0, 0.0, 0.0, v], dtype=np.float64)
        else:
            x0 = pred
            x0[3] = v
        t_x = tau + np.arange(self.N + 1, dtype=np.float64) * self.dt
        t_u = tau + np.arange(self.N, dtype=np.float64) * self.dt
        Xg = self._to_current_ego(self._interp_traj(self._last_X, t_x), ego_xy_yaw)
        Ug = self._interp_u(self._last_U, t_u)
        Xg[:, 0] = x0
        stub = self._to_current_ego(
            self._interp_traj(self._last_X, np.linspace(0.0, tau, num=5)),
            ego_xy_yaw,
        )
        return x0, Xg, Ug, stub

    def solve(
        self,
        v: float,
        esdf: np.ndarray,
        target: np.ndarray,
        p_ref: np.ndarray,
        yaw_ref: np.ndarray | None = None,
        ego_xy_yaw: tuple[float, float, float] | None = None,
    ) -> dict:
        """
        Solve one NMPC step.

        After the first call, X[:,0] is the previous plan evaluated at ~200 ms
        (solve delay), expressed in the current ego frame, so consecutive
        trajectories stay C0-continuous.
        """
        v = float(np.clip(v, 0.0, self.v_max))
        target = np.asarray(target, dtype=np.float64).reshape(3)
        p_ref = np.asarray(p_ref, dtype=np.float64)
        if p_ref.shape[0] == 2:
            p_ref = p_ref.T
        if p_ref.shape[0] != self.N + 1:
            raise ValueError(f"p_ref must be ({self.N + 1}, 2), got {p_ref.shape}")
        if yaw_ref is None:
            yaw_ref = np.zeros(self.N + 1)
        yaw_ref = np.asarray(yaw_ref, dtype=np.float64).reshape(self.N + 1)

        stub = np.zeros((4, 1))
        if self._last_X is not None:
            x0, Xg, Ug, stub = self._stitch_x0(v, ego_xy_yaw)
        else:
            x0 = np.array([0.0, 0.0, 0.0, v], dtype=np.float64)
            Xg, Ug = self._straight_guess(target, v)
            Xg[:, 0] = x0

        esdf_flat = flatten_esdf_casadi(esdf)
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.target, target)
        self.opti.set_value(self.p_ref, p_ref.T)
        self.opti.set_value(self.yaw_ref, yaw_ref.reshape(1, -1))
        v_ref = float(np.clip(np.linalg.norm(target[:2]) / max(self.horizon_s, 1e-6), 0.4, self.v_max))
        self.opti.set_value(self.v_ref, v_ref)
        self.opti.set_value(self.esdf_p, esdf_flat)
        self.opti.set_initial(self.X, Xg)
        self.opti.set_initial(self.U, Ug)

        t0 = time.perf_counter()
        status = "ok"
        used_fallback = False
        _usable = {
            "Solve_Succeeded", "Solved_To_Acceptable_Level",
            "Maximum_Iterations_Exceeded",
        }
        try:
            sol = self.opti.solve()
            X = np.array(sol.value(self.X)).reshape(4, self.N + 1)
            U = np.array(sol.value(self.U)).reshape(2, self.N)
            try:
                status = self.opti.stats().get("return_status", "ok")
            except Exception:
                status = "ok"
        except Exception:
            try:
                status = self.opti.stats().get("return_status", "fail")
            except Exception:
                status = "fail"
            try:
                X = np.array(self.opti.debug.value(self.X)).reshape(4, self.N + 1)
                U = np.array(self.opti.debug.value(self.U)).reshape(2, self.N)
                used_fallback = status not in _usable or not np.isfinite(X).all()
            except Exception:
                used_fallback = True
                if self._last_X is not None:
                    X, U = Xg, Ug
                else:
                    X, U = Xg, Ug
            if used_fallback:
                status = f"fail:{status}"
        solve_ms = (time.perf_counter() - t0) * 1000.0

        if not used_fallback:
            veh0 = np.array([0.0, 0.0, 0.0, v], dtype=np.float64)
            if float(np.hypot(X[0, 0], X[1, 0])) < 0.05:
                stored = X.copy()
                stored[:, 0] = veh0
            else:
                tpts = np.concatenate([
                    [0.0],
                    self.stitch_s + np.arange(self.N + 1, dtype=np.float64) * self.dt,
                ])
                Tpts = np.hstack([veh0.reshape(4, 1), X])
                stored = self._interp_traj(Tpts, np.arange(self.N + 1) * self.dt, t=tpts)
                stored[:, 0] = veh0
            self._last_X, self._last_U = stored, U
            self._last_ego_xy_yaw = ego_xy_yaw
            self._last_solve_s = solve_ms / 1000.0
        u0 = U[:, 0].copy() if np.isfinite(U[:, 0]).all() else self._last_u0
        self._last_u0 = u0

        pts = transform_body(X[0], X[1], X[2], self.body).reshape(-1, 3)
        clearance = query_esdf(esdf, self.grid, pts)
        min_clearance = float(np.min(clearance)) if clearance.size else float("inf")
        terminal_err = float(np.linalg.norm(X[:2, -1] - target[:2]))
        vis = np.hstack([stub[:, :-1], X]) if stub.shape[1] > 1 else X

        return {
            "traj": X.T.copy(),
            "traj_vis": vis.T.copy(),
            "x0": x0.copy(),
            "u": U.T.copy(),
            "u0": u0,
            "status": status,
            "solve_ms": solve_ms,
            "used_fallback": used_fallback,
            "min_clearance": min_clearance,
            "terminal_err": terminal_err,
        }


def math_atan2(y, x):
    return float(np.arctan2(y, x))


def nmpc_to_carla_control(
    a: float,
    delta: float,
    a_max: float = 3.0,
    delta_max: float = 0.50,
    speed: float | None = None,
    v_max: float | None = None,
):
    """
    Map NMPC (a, delta) to carla.VehicleControl.

    NMPC delta is right-handed (positive = left / +Y). CARLA steer positive is right.
    If speed exceeds v_max, throttle is cut and brake is applied.
    """
    import carla

    if v_max is not None and speed is not None:
        err = v_max - float(speed)
        if a < -2.0:
            throttle = 0.0
            brake = float(np.clip(-a / max(a_max, 1e-3), 0.0, 1.0))
        elif err < -0.05:
            throttle = 0.0
            brake = float(np.clip(-err / 0.25, 0.15, 1.0))
        elif err < 0.08:
            throttle = 0.24
            brake = 0.0
        else:
            throttle = float(np.clip(0.24 + 0.40 * err / max(v_max, 0.3), 0.24, 0.55))
            brake = 0.0
    elif a >= 0.0:
        throttle = float(np.clip(0.12 + 0.35 * a / max(a_max, 1e-3), 0.0, 0.40))
        brake = 0.0
    else:
        throttle = 0.0
        brake = float(np.clip(-a / max(a_max, 1e-3), 0.0, 1.0))
    steer = float(np.clip(-delta / max(delta_max, 1e-3), -1.0, 1.0))
    return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
