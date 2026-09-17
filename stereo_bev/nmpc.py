"""CasADi kinematic-bicycle NMPC with a 3D occupancy ESDF cost."""

from __future__ import annotations

import math
import time
import numpy as np

from .bev_grid import BEVGrid
from .global_target import (
    wrap_angle, se3_from_planar, rotz, yaw_from_R, log_so3, exp_so3,
)
from .occ_field import esdf_axis_grids, flatten_esdf_casadi, query_esdf, query_esdf_z_slack
from .sfc import SFC_N_FACES, pack_sfc_parameters
from .vehicle_body import (
    MODEL3_WHEELBASE, MODEL3_MAX_STEER_DEG, transform_body_se3,
)

try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    ca = None
    HAS_CASADI = False


class OccupancyNMPC:
    """
    Receding-horizon NMPC whose knot state lives on extended SE(3) = SE_2(3).

    Knot state z = [t (3), φ (3), v (3)] with R = exp(φ^) ∈ SO(3) and spatial
    velocity v ∈ ℝ³. The group element is
        X = (R, v, t) ∈ SE_2(3)
    so velocity transforms with pose (left action: v ↦ R_Δ v). The vehicle
    starts at the identity of the current ego frame, v = (s, 0, 0). Integration
    is left-invariant: X_{k+1} = X_k exp(Δt ξ^) with bicycle body twist
    ξ = (s, 0, 0, 0, 0, s tan(δ)/L) and ṡ = a. Cost is the left-invariant
    error X_ref^{-1} X, and collision balls are p = t + R @ body in the ESDF.
    Each ball may shift ±z_slack in occupancy Z (smooth max ESDF) so a sloped
    road surface is not treated as a hit; XY obstacles still block.

    Optional Safe Flight Corridor: each knot is assigned a 3D H-rep polyhedron
    {x | A x ≤ b}. Every collision ball must stay inside that polyhedron
    (Liu / FASTER). Inactive planes use b = 1e6.
    """

    def __init__(
        self,
        grid: BEVGrid,
        body: np.ndarray,
        radii: np.ndarray | None = None,
        horizon_s: float = 5.0,
        dt: float = 0.2,
        wheelbase: float = MODEL3_WHEELBASE,
        r_safe: float = 0.15,
        r_hard: float = 0.10,
        z_slack: float = 0.30,
        d0: float = 0.40,
        a_max: float = 3.0,
        delta_max: float = 0.50,
        v_max: float = 2.0,
        w_pos: float = 2.0,
        w_yaw: float = 0.4,
        w_term_pos: float = 5.0,
        w_term_yaw: float = 1.5,
        w_u: float = 0.08,
        w_du: float = 1.5,
        w_v: float = 0.6,
        w_occ: float = 400.0,
        w_slack: float = 8.0e3,
        w_sfc_slack: float = 8.0e3,
        n_sfc_faces: int = SFC_N_FACES,
        ipopt_max_iter: int = 180,
        stitch_s: float = 0.2,
        esdf_offset_xy: tuple[float, float] = (0.0, 0.0),
    ):
        if not HAS_CASADI:
            raise ImportError("casadi is required for OccupancyNMPC (pip install casadi)")

        self.grid = grid
        self.body = np.asarray(body, dtype=np.float64)
        if radii is None:
            self.radii = np.zeros(self.body.shape[0], dtype=np.float64)
        else:
            self.radii = np.asarray(radii, dtype=np.float64).reshape(-1)
            if self.radii.shape[0] != self.body.shape[0]:
                raise ValueError(
                    f"radii length {self.radii.shape[0]} != body {self.body.shape[0]}"
                )
        self.horizon_s = horizon_s
        self.dt = dt
        self.N = int(round(horizon_s / dt))
        self.L = wheelbase
        self.r_safe = r_safe
        self.r_hard = r_hard
        self.z_slack = float(max(z_slack, 0.0))
        n_z = 5 if self.z_slack > 0 else 1
        self._z_slack_off = np.linspace(-self.z_slack, self.z_slack, n_z)
        self.d0 = d0
        self.w_slack = w_slack
        self.w_sfc_slack = w_sfc_slack
        self.n_sfc_faces = int(n_sfc_faces)
        self.a_max = a_max
        self.delta_max = delta_max
        self.v_max = v_max
        self.stitch_s = stitch_s
        self._esdf_ox = float(esdf_offset_xy[0])
        self._esdf_oy = float(esdf_offset_xy[1])
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
        self.n_state = 9
        X = opti.variable(self.n_state, self.N + 1)
        U = opti.variable(2, self.N)
        x0 = opti.parameter(self.n_state)
        target_t = opti.parameter(3)
        target_R = opti.parameter(3, 3)
        ref_t = opti.parameter(3, self.N + 1)
        ref_R = opti.parameter(9, self.N + 1)
        v_ref = opti.parameter()
        esdf_p = opti.parameter(self.n_vox)
        slack = opti.variable(self.N + 1)
        poly_A = opti.parameter(self.n_sfc_faces * 3, self.N + 1)
        poly_b = opti.parameter(self.n_sfc_faces, self.N + 1)
        sfc_slack = opti.variable(self.N + 1)

        t_z = X[2, :]
        phi_x, phi_y = X[3, :], X[4, :]
        a, delta = U[0, :], U[1, :]
        e_x = ca.vertcat(1.0, 0.0, 0.0)

        opti.subject_to(X[:, 0] == x0)
        opti.subject_to(slack >= 0)
        opti.subject_to(sfc_slack >= 0)
        for k in range(self.N):
            opti.subject_to(X[:, k + 1] == self._integrate_se3(X[:, k], U[:, k], dt))

        opti.subject_to(opti.bounded(-a_max, a, a_max))
        opti.subject_to(opti.bounded(-delta_max, delta, delta_max))
        opti.subject_to(opti.bounded(-0.5, t_z, 1.0))
        opti.subject_to(opti.bounded(-0.30, phi_x, 0.30))
        opti.subject_to(opti.bounded(-0.30, phi_y, 0.30))

        cost = 0
        rad = ca.vec(self.radii)
        for k in range(self.N + 1):
            tk = X[0:3, k]
            Rk_pred = self._exp_so3(X[3:6, k])
            vk = X[6:9, k]
            R_ref_k = ca.reshape(ref_R[:, k], 3, 3)
            v_ref_k = v_ref * (R_ref_k @ e_x)
            t_err, rv, v_err = self._se2_3_error(
                tk, Rk_pred, vk, ref_t[:, k], R_ref_k, v_ref_k,
            )
            wt = w_pos if k < self.N else (w_pos + w_term_pos)
            wr = w_yaw if k < self.N else (w_yaw + w_term_yaw)
            cost = cost + wt * ca.dot(t_err, t_err) + wr * ca.dot(rv, rv)
            cost = cost + w_v * ca.dot(v_err, v_err)
            s_k = ca.dot(Rk_pred @ e_x, vk)
            opti.subject_to(opti.bounded(0.0, s_k, v_max))
            d = self._ball_esdf(tk, Rk_pred, esdf_p)
            cost = cost + w_occ * self._occupancy_cost(d, rad, tk, Rk_pred)
            Ak = ca.reshape(poly_A[:, k], 3, self.n_sfc_faces).T
            pts_b = ca.repmat(tk, 1, self.n_body) + Rk_pred @ self.body.T
            val = Ak @ pts_b + ca.repmat(rad.T, self.n_sfc_faces, 1)
            if k >= 1:
                opti.subject_to(d >= rad + self.r_hard - slack[k])
                cost = cost + w_slack * slack[k] ** 2
                for f in range(self.n_sfc_faces):
                    opti.subject_to(
                        ca.mmax(val[f, :]) <= poly_b[f, k] + sfc_slack[k]
                    )
                cost = cost + w_sfc_slack * sfc_slack[k] ** 2

        RN = self._exp_so3(X[3:6, self.N])
        vN = X[6:9, self.N]
        v_tgt = v_ref * (target_R @ e_x)
        t_err, rv, v_err = self._se2_3_error(
            X[0:3, self.N], RN, vN, target_t, target_R, v_tgt,
        )
        cost = cost + w_term_pos * ca.dot(t_err, t_err)
        cost = cost + w_term_yaw * ca.dot(rv, rv)
        cost = cost + w_v * ca.dot(v_err, v_err)

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
        self.x0 = x0
        self.target_t, self.target_R = target_t, target_R
        self.ref_t, self.ref_R = ref_t, ref_R
        self.v_ref = v_ref
        self.esdf_p = esdf_p
        self.slack = slack
        self.poly_A = poly_A
        self.poly_b = poly_b
        self.sfc_slack = sfc_slack
        self._last_X = None
        self._last_U = None
        self._last_u0 = np.zeros(2)
        self._last_ego_xy_yaw = None
        self._last_solve_s = None

    @staticmethod
    def _skew(w):
        return ca.vertcat(
            ca.horzcat(0, -w[2], w[1]),
            ca.horzcat(w[2], 0, -w[0]),
            ca.horzcat(-w[1], w[0], 0),
        )

    def _exp_so3(self, w):
        """Rodrigues: so(3) vector → SO(3)."""
        th2 = ca.dot(w, w) + 1e-12
        th = ca.sqrt(th2)
        K = self._skew(w)
        I = ca.MX.eye(3)
        A = ca.sin(th) / th
        B = (1.0 - ca.cos(th)) / th2
        return I + A * K + B * (K @ K)

    def _log_so3(self, R):
        """SO(3) → so(3) rotation vector."""
        vee = 0.5 * ca.vertcat(
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        )
        c = ca.fmin(ca.fmax(0.5 * (ca.trace(R) - 1.0), -1.0 + 1e-9), 1.0 - 1e-9)
        th = ca.fmax(ca.acos(c), 1e-6)
        return (th / ca.sin(th)) * vee

    def _integrate_se3(self, z, u, dt):
        """Left-invariant SE_2(3) step X+ = X exp(Δt ξ^), ξ bicycle body twist."""
        t = z[0:3]
        phi = z[3:6]
        vel = z[6:9]
        a = u[0]
        delta = u[1]
        R = self._exp_so3(phi)
        s = ca.dot(R @ ca.vertcat(1.0, 0.0, 0.0), vel)
        nu = ca.vertcat(s, 0, 0)
        omega = ca.vertcat(0, 0, s * ca.tan(delta) / self.L)
        w = omega * dt
        Rinc = self._exp_so3(w)
        K = self._skew(w)
        I = ca.MX.eye(3)
        th2 = ca.dot(w, w) + 1e-12
        th = ca.sqrt(th2)
        B = (1.0 - ca.cos(th)) / th2
        C = (th - ca.sin(th)) / (th2 * th)
        V = I + B * K + C * (K @ K)
        R_new = R @ Rinc
        t_new = t + R @ (V @ (nu * dt))
        phi_new = self._log_so3(R_new)
        s_new = s + a * dt
        vel_new = R_new @ ca.vertcat(s_new, 0, 0)
        return ca.vertcat(t_new, phi_new, vel_new)

    @staticmethod
    def _se2_3_error(t_pred, R_pred, v_pred, t_ref, R_ref, v_ref):
        """Left-invariant SE_2(3) error X_ref^{-1} X_pred: (t, rotvec, v)."""
        R_err = R_ref.T @ R_pred
        t_err = R_ref.T @ (t_pred - t_ref)
        v_err = R_ref.T @ (v_pred - v_ref)
        rv = 0.5 * ca.vertcat(
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        )
        return t_err, rv, v_err

    @staticmethod
    def _se3_error(t_pred, R_pred, t_ref, R_ref):
        """Left-invariant (t, rotation-vector) error T_ref^{-1} T_pred."""
        R_err = R_ref.T @ R_pred
        t_err = R_ref.T @ (t_pred - t_ref)
        rv = 0.5 * ca.vertcat(
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        )
        return t_err, rv

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

    def _body_world(self, t, R):
        """Collision-ball centers: t + R @ body. t is vehicle-center occupancy FLU."""
        p = ca.repmat(t, 1, self.n_body) + R @ self.body.T
        px = ca.vec(p[0, :]) - self._esdf_ox
        py = ca.vec(p[1, :]) - self._esdf_oy
        pz = ca.vec(p[2, :])
        return px, py, pz

    def _ball_esdf(self, t, R, esdf_p):
        """ESDF at each ball, using the best occupancy-Z in ±z_slack."""
        px, py, pz = self._body_world(t, R)
        px_c = ca.fmin(ca.fmax(px, self._x_min), self._x_max)
        py_c = ca.fmin(ca.fmax(py, self._y_min), self._y_max)
        d = None
        for dz in self._z_slack_off:
            pz_c = ca.fmin(ca.fmax(pz + float(dz), self._z_min), self._z_max)
            di = ca.vec(self._sample_esdf(esdf_p, px_c, py_c, pz_c))
            d = di if d is None else ca.fmax(d, di)
        return d

    def _occupancy_cost(self, d, rad, t, R):
        """Soft hinge: penalize ESDF < radius + r_safe, harder if the ball is inside."""
        px, _, _ = self._body_world(t, R)
        in_vol = ca.vec(0.5 * (1 + ca.tanh((px + rad - self._x_min) / 0.08)))
        gap = (rad + self.r_safe) - d
        hinge = (ca.fmax(gap, 0) / self.d0) ** 2
        penetrate = (ca.fmax(rad - d, 0.0) / self.d0) ** 2
        return ca.sum1(in_vol * (hinge + 8.0 * penetrate))

    def _straight_guess(
        self,
        target: np.ndarray,
        v: float,
        p_ref: np.ndarray | None = None,
        yaw_ref: np.ndarray | None = None,
        ref_t: np.ndarray | None = None,
        ref_R: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        X = np.zeros((self.n_state, self.N + 1))
        U = np.zeros((2, self.N))
        if ref_t is not None:
            rt = np.asarray(ref_t, dtype=np.float64).reshape(self.N + 1, 3)
            X[0:3, :] = rt.T
            if ref_R is not None:
                for k, Rk in enumerate(np.asarray(ref_R).reshape(self.N + 1, 3, 3)):
                    X[3:6, k] = log_so3(Rk)
        elif p_ref is not None and np.asarray(p_ref).shape[0] == self.N + 1:
            pr = np.asarray(p_ref, dtype=np.float64)
            X[0, :] = pr[:, 0]
            X[1, :] = pr[:, 1]
            if yaw_ref is not None:
                yaws = np.asarray(yaw_ref, dtype=np.float64).reshape(-1)
            else:
                d = np.diff(pr, axis=0, prepend=pr[:1])
                yaws = np.arctan2(d[:, 1], d[:, 0])
            X[5, :] = yaws
        else:
            for k in range(self.N + 1):
                alpha = k / self.N
                X[0, k] = target[0] * alpha
                X[1, k] = target[1] * alpha
                X[5, k] = math_atan2(target[1], target[0]) if (target[0] ** 2 + target[1] ** 2) > 0.05 else 0.0
        spd = max(float(v), 0.5)
        for k in range(self.N + 1):
            Rk = exp_so3(X[3:6, k])
            X[6:9, k] = Rk @ np.array([spd, 0.0, 0.0])
        X[:, 0] = 0.0
        X[6, 0] = spd
        return X, U

    def _interp_traj(self, X: np.ndarray, t_query: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        """Interpolate (9, K) SE_2(3) trajectory at times t_query."""
        if t is None:
            t = np.arange(X.shape[1], dtype=np.float64) * self.dt
        tq = np.clip(np.atleast_1d(np.asarray(t_query, dtype=np.float64)), t[0], t[-1])
        yaw = np.unwrap(X[5])
        rows = [
            np.interp(tq, t, X[0]),
            np.interp(tq, t, X[1]),
            np.interp(tq, t, X[2]),
            np.interp(tq, t, X[3]),
            np.interp(tq, t, X[4]),
            wrap_angle(np.interp(tq, t, yaw)),
            np.interp(tq, t, X[6]),
            np.interp(tq, t, X[7]),
            np.interp(tq, t, X[8]),
        ]
        return np.vstack(rows)

    def _interp_u(self, U: np.ndarray, t_query: np.ndarray) -> np.ndarray:
        """Zero-order hold of (2, N) controls at interval start times t_query."""
        tq = np.atleast_1d(np.asarray(t_query, dtype=np.float64))
        idx = np.clip(np.floor(tq / self.dt).astype(np.int32), 0, U.shape[1] - 1)
        return U[:, idx]

    def _to_current_ego(self, states: np.ndarray, ego_xy_yaw) -> np.ndarray:
        if ego_xy_yaw is None or self._last_ego_xy_yaw is None:
            return states
        t0, R0 = se3_from_planar(*self._last_ego_xy_yaw)
        t1, R1 = se3_from_planar(*ego_xy_yaw)
        R_rel = R1.T @ R0
        t_rel = R1.T @ (t0 - t1)
        out = np.asarray(states, dtype=np.float64).copy()
        for k in range(out.shape[1]):
            t = out[0:3, k]
            R = exp_so3(out[3:6, k])
            vel = out[6:9, k]
            out[0:3, k] = t_rel + R_rel @ t
            out[3:6, k] = log_so3(R_rel @ R)
            out[6:9, k] = R_rel @ vel
        return out

    def _identity_state(self, v: float) -> np.ndarray:
        z = np.zeros(self.n_state, dtype=np.float64)
        z[6] = float(v)
        return z

    def _stitch_x0(self, v: float, ego_xy_yaw) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        New plan starts at the current vehicle (ego identity). Warm-start from
        the previous SE_2(3) trajectory expressed in this frame.
        """
        x0 = self._identity_state(v)
        t_x = np.arange(self.N + 1, dtype=np.float64) * self.dt
        t_u = np.arange(self.N, dtype=np.float64) * self.dt
        Xg = self._to_current_ego(self._interp_traj(self._last_X, t_x), ego_xy_yaw)
        Ug = self._interp_u(self._last_U, t_u)
        Xg[:, 0] = x0
        stub = x0.reshape(self.n_state, 1)
        return x0, Xg, Ug, stub

    @staticmethod
    def _coerce_target_se3(target, target_R=None) -> tuple[np.ndarray, np.ndarray]:
        """Accept (x,y,yaw), (t, R), or a 4×4 matrix."""
        if target_R is not None:
            t = np.asarray(target, dtype=np.float64).reshape(3)
            R = np.asarray(target_R, dtype=np.float64).reshape(3, 3)
            return t, R
        arr = np.asarray(target, dtype=np.float64)
        if arr.shape == (4, 4):
            return arr[:3, 3].copy(), arr[:3, :3].copy()
        if arr.shape == (3,):
            return se3_from_planar(float(arr[0]), float(arr[1]), float(arr[2]))
        if arr.shape == (6,):
            from .global_target import carla_rotation_matrix
            t = arr[:3].copy()
            # ego FLU R_z(yaw) R_y(pitch) R_x(roll) ≈ CARLA with Y already left
            R = rotz(float(arr[4]))
            if abs(float(arr[3])) + abs(float(arr[5])) > 1e-8:
                R = carla_rotation_matrix(float(arr[3]), float(arr[4]), float(arr[5]))
                R = np.diag([1.0, -1.0, 1.0]) @ R @ np.diag([1.0, -1.0, 1.0])
            return t, R
        raise ValueError(f"unsupported target pose shape {arr.shape}")

    @staticmethod
    def _pack_ref_R(ref_R: np.ndarray) -> np.ndarray:
        """(K, 3, 3) → (9, K) column-major, matching CasADi reshape."""
        R = np.asarray(ref_R, dtype=np.float64).reshape(-1, 3, 3)
        return np.stack([Rk.reshape(9, order="F") for Rk in R], axis=1)

    def _poses_from_state(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Unpack knot states to T = (t, R) ∈ SE(3)."""
        K = X.shape[1]
        t = np.asarray(X[0:3, :], dtype=np.float64).T.copy()
        R_out = np.stack([exp_so3(X[3:6, k]) for k in range(K)], axis=0)
        return t, R_out

    def _traj4(self, X: np.ndarray) -> np.ndarray:
        t, R = self._poses_from_state(X)
        yaw = np.array([yaw_from_R(Rk) for Rk in R], dtype=np.float64)
        speed = np.linalg.norm(np.asarray(X[6:9, :], dtype=np.float64), axis=0)
        return np.column_stack([t[:, 0], t[:, 1], yaw, speed])

    def solve(
        self,
        v: float,
        esdf: np.ndarray,
        target: np.ndarray,
        p_ref: np.ndarray | None = None,
        yaw_ref: np.ndarray | None = None,
        ego_xy_yaw: tuple[float, float, float] | None = None,
        target_R: np.ndarray | None = None,
        ref_t: np.ndarray | None = None,
        ref_R: np.ndarray | None = None,
        poly_A: np.ndarray | None = None,
        poly_b: np.ndarray | None = None,
    ) -> dict:
        """
        Solve one NMPC step.

        After the first call, the previous plan is transformed into the
        current ego frame as a warm start. X[:,0] is always the live
        vehicle identity with spatial velocity (v, 0, 0).

        ``target`` is an SE(3) pose in ego FLU: (x, y, yaw), (4, 4), or
        translation plus ``target_R``. Horizon references are ``ref_t``
        (N+1, 3) and ``ref_R`` (N+1, 3, 3); planar ``p_ref`` / ``yaw_ref``
        are accepted and lifted with z=0, R_z(yaw).

        ``poly_A`` (n_faces*3, N+1) and ``poly_b`` (n_faces, N+1) are the
        Safe Flight Corridor H-rep per knot. Omit them to disable SFC.
        """
        v = float(np.clip(v, 0.0, self.v_max))
        t_tgt, R_tgt = self._coerce_target_se3(target, target_R)
        target_xyyaw = np.array(
            [t_tgt[0], t_tgt[1], yaw_from_R(R_tgt)], dtype=np.float64,
        )
        n = self.N + 1
        if ref_t is None:
            if p_ref is None:
                raise ValueError("solve requires ref_t or p_ref")
            p_ref = np.asarray(p_ref, dtype=np.float64)
            if p_ref.shape[0] == 2:
                p_ref = p_ref.T
            if p_ref.shape[0] != n:
                raise ValueError(f"p_ref must be ({n}, 2), got {p_ref.shape}")
            if yaw_ref is None:
                yaw_ref = np.zeros(n)
            yaw_ref = np.asarray(yaw_ref, dtype=np.float64).reshape(n)
            ref_t = np.column_stack([p_ref, np.zeros(n)])
            ref_R = np.stack([rotz(float(y)) for y in yaw_ref], axis=0)
        else:
            ref_t = np.asarray(ref_t, dtype=np.float64).reshape(n, 3)
            ref_R = np.asarray(ref_R, dtype=np.float64).reshape(n, 3, 3)
            p_ref = ref_t[:, :2].copy()
            yaw_ref = np.array([yaw_from_R(Rk) for Rk in ref_R], dtype=np.float64)
        ref_t[-1] = t_tgt
        ref_R[-1] = R_tgt
        p_ref[-1] = t_tgt[:2]
        yaw_ref[-1] = yaw_from_R(R_tgt)

        stub = np.zeros((self.n_state, 1))
        if self._last_X is not None:
            x0, Xg, Ug, stub = self._stitch_x0(v, ego_xy_yaw)
        else:
            x0 = self._identity_state(v)
            Xg, Ug = self._straight_guess(
                target_xyyaw, v, p_ref=p_ref, yaw_ref=yaw_ref,
                ref_t=ref_t, ref_R=ref_R,
            )
            Xg[:, 0] = x0

        esdf_flat = flatten_esdf_casadi(esdf)
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.target_t, t_tgt)
        self.opti.set_value(self.target_R, R_tgt)
        self.opti.set_value(self.ref_t, ref_t.T)
        self.opti.set_value(self.ref_R, self._pack_ref_R(ref_R))
        v_ref = float(np.clip(np.linalg.norm(t_tgt[:2]) / max(self.horizon_s, 1e-6), 0.4, self.v_max))
        self.opti.set_value(self.v_ref, v_ref)
        self.opti.set_value(self.esdf_p, esdf_flat)
        if poly_A is None or poly_b is None:
            poly_A, poly_b = pack_sfc_parameters(None, n, self.n_sfc_faces)
        else:
            poly_A = np.asarray(poly_A, dtype=np.float64).reshape(
                self.n_sfc_faces * 3, n,
            )
            poly_b = np.asarray(poly_b, dtype=np.float64).reshape(
                self.n_sfc_faces, n,
            )
        self.opti.set_value(self.poly_A, poly_A)
        self.opti.set_value(self.poly_b, poly_b)
        self.opti.set_initial(self.X, Xg)
        self.opti.set_initial(self.U, Ug)
        self.opti.set_initial(self.slack, np.zeros(self.N + 1))
        self.opti.set_initial(self.sfc_slack, np.zeros(self.N + 1))

        t0 = time.perf_counter()
        status = "ok"
        used_fallback = False
        _usable = {
            "Solve_Succeeded", "Solved_To_Acceptable_Level",
            "Maximum_Iterations_Exceeded",
        }
        try:
            sol = self.opti.solve()
            X = np.array(sol.value(self.X)).reshape(self.n_state, self.N + 1)
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
                X = np.array(self.opti.debug.value(self.X)).reshape(self.n_state, self.N + 1)
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
            stored = X.copy()
            stored[:, 0] = self._identity_state(v)
            self._last_X, self._last_U = stored, U
            self._last_ego_xy_yaw = ego_xy_yaw
            self._last_solve_s = solve_ms / 1000.0
        u0 = U[:, 0].copy() if np.isfinite(U[:, 0]).all() else self._last_u0
        self._last_u0 = u0

        pts_t, pts_R = self._poses_from_state(X)
        pts = transform_body_se3(pts_t, pts_R, self.body).reshape(-1, 3)
        pts[:, 0] -= self._esdf_ox
        pts[:, 1] -= self._esdf_oy
        clearance = query_esdf_z_slack(
            esdf, self.grid, pts, z_slack=self.z_slack,
        ) - np.tile(self.radii, X.shape[1])
        min_clearance = float(np.min(clearance)) if clearance.size else float("inf")
        t_err = R_tgt.T @ (pts_t[-1] - t_tgt)
        terminal_err = float(np.linalg.norm(t_err))
        traj4 = self._traj4(X)
        vis4 = traj4 if stub.shape[1] <= 1 else traj4

        return {
            "traj": traj4,
            "traj_vis": vis4,
            "traj_t": pts_t.copy(),
            "traj_R": pts_R.copy(),
            "state": X.T.copy(),
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
    steer_max_rad: float | None = None,
):
    """
    Map NMPC (a, delta) to carla.VehicleControl.

    NMPC delta is the bicycle wheel angle in radians (positive = left / +Y).
    CARLA ``steer`` is a fraction of the vehicle's max steer angle, and
    positive steer is right, so we scale by ``steer_max_rad`` and negate.
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
    if steer_max_rad is None:
        steer_max_rad = math.radians(MODEL3_MAX_STEER_DEG)
    steer = float(np.clip(-delta / max(float(steer_max_rad), 1e-3), -1.0, 1.0))
    return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
