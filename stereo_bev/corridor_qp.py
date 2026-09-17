"""Linearized bicycle MPC as a QP with hard 3D Safe Flight Corridor constraints.

Liu / FASTER back-end: the geometric skeleton is a reference; OSQP fine-tunes
a kinematic-bicycle trajectory whose collision balls stay inside the assigned
H-rep polyhedra. No slack. Dynamics are affine (linearized around the
reference each receding-horizon step).
"""

from __future__ import annotations

import math
import time
import numpy as np

from .bev_grid import BEVGrid
from .global_target import rotz, yaw_from_R, wrap_angle, transform_states_between_ego
from .occ_field import query_esdf, query_esdf_z_slack
from .sfc import SFC_N_FACES
from .vehicle_body import MODEL3_WHEELBASE, transform_body_se3, body_xy_support

try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    ca = None
    HAS_CASADI = False


def _bicycle_step(z: np.ndarray, u: np.ndarray, L: float, dt: float) -> np.ndarray:
    x, y, psi, v = [float(v_) for v_ in z]
    a, delta = float(u[0]), float(np.clip(u[1], -1.2, 1.2))
    c, s = math.cos(psi), math.sin(psi)
    td = math.tan(delta)
    return np.array([
        x + dt * v * c,
        y + dt * v * s,
        psi + dt * v * td / L,
        v + dt * a,
    ], dtype=np.float64)


def _linearize_bicycle(z: np.ndarray, u: np.ndarray, L: float, dt: float):
    """z+ = A z + B u + r, exact at (z, u) to first order nearby."""
    x, y, psi, v = [float(v_) for v_ in z]
    a, delta = float(u[0]), float(np.clip(u[1], -1.2, 1.2))
    c, s = math.cos(psi), math.sin(psi)
    cd = math.cos(delta)
    td = math.tan(delta)
    A = np.array([
        [1.0, 0.0, -dt * v * s, dt * c],
        [0.0, 1.0, dt * v * c, dt * s],
        [0.0, 0.0, 1.0, dt * td / L],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)
    sec2 = 1.0 / max(cd * cd, 1e-6)
    B = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, dt * v * sec2 / L],
        [dt, 0.0],
    ], dtype=np.float64)
    z1 = _bicycle_step(z, u, L, dt)
    r = z1 - A @ z - B @ np.array([a, delta], dtype=np.float64)
    return A, B, r


def _path_delta(yaw: np.ndarray, v: np.ndarray, dt: float, L: float, delta_max: float) -> np.ndarray:
    dyaw = np.diff(np.unwrap(np.asarray(yaw, dtype=np.float64)), append=yaw[-1])
    speed = np.maximum(np.asarray(v, dtype=np.float64), 0.4)
    kappa = dyaw / np.maximum(speed * dt, 1e-3)
    return np.clip(np.arctan(L * kappa), -delta_max, delta_max)


def shrink_halfspaces(
    A: np.ndarray,
    b: np.ndarray,
    R: np.ndarray,
    body: np.ndarray,
    radii: np.ndarray,
    z_ref: float,
    margin: float = 0.10,
    keep_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    COM inequalities Hx x + Hy y <= h at height z_ref.

    Occupancy AABB planes are left un-eroded so the ego origin stays feasible.
    Obstacle / ellipsoid planes are Minkowski-shrunk by the *lateral*
    collision balls plus ``margin`` (``r_hard``). Vehicle length is left
    to the along-track corridor; shrinking by the full nose empties the QP.
    """
    A = np.asarray(A, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    b2 = b - A[:, 2] * float(z_ref)
    an = np.abs(A)
    aabb = (
        ((an[:, 0] > 0.95) & (an[:, 1] < 0.08) & (an[:, 2] < 0.08))
        | ((an[:, 1] > 0.95) & (an[:, 0] < 0.08) & (an[:, 2] < 0.08))
        | ((an[:, 2] > 0.95) & (an[:, 0] < 0.08) & (an[:, 1] < 0.08))
    )
    b2 = b2.copy()
    hx, hy = A[:, 0].copy(), A[:, 1].copy()
    yaw = 0.0
    if R is not None:
        Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)
        yaw = math.atan2(float(Rm[1, 0]), float(Rm[0, 0]))
    body_lat = np.asarray(body, dtype=np.float64).reshape(-1, 3).copy()
    body_lat[:, 0] = 0.0
    support = body_xy_support(hx, hy, yaw, body_lat, radii)
    b2[~aabb] = b2[~aabb] - support[~aabb] - float(margin)
    if keep_xy is not None:
        xy = np.asarray(keep_xy, dtype=np.float64).reshape(2)
        viol = hx * xy[0] + hy * xy[1] - b2
        tiny = (viol > 0.0) & (viol < 0.08)
        b2 = np.where(tiny, b2 + viol + 1e-4, b2)
    return hx, hy, b2


class CorridorQP:
    """Receding-horizon QP: linearized bicycle + hard SFC polyhedra."""

    def __init__(
        self,
        grid: BEVGrid,
        body: np.ndarray,
        radii: np.ndarray | None = None,
        horizon_s: float = 5.0,
        dt: float = 0.2,
        wheelbase: float = MODEL3_WHEELBASE,
        a_max: float = 3.0,
        delta_max: float = 0.50,
        v_max: float = 2.0,
        r_hard: float = 0.10,
        r_safe: float = 0.15,
        z_slack: float = 0.30,
        n_sfc_faces: int = SFC_N_FACES,
        w_pos: float = 4.0,
        w_yaw: float = 1.2,
        w_v: float = 0.4,
        w_term_pos: float = 12.0,
        w_term_yaw: float = 3.0,
        w_u: float = 0.08,
        w_du: float = 1.5,
        w_hold: float = 6.0,
    ):
        if not HAS_CASADI:
            raise ImportError("casadi is required for CorridorQP")
        self.grid = grid
        self.body = np.asarray(body, dtype=np.float64)
        self.radii = (
            np.zeros(self.body.shape[0], dtype=np.float64)
            if radii is None
            else np.asarray(radii, dtype=np.float64).reshape(-1)
        )
        self.horizon_s = horizon_s
        self.dt = dt
        self.N = int(round(horizon_s / dt))
        self.L = wheelbase
        self.a_max = a_max
        self.delta_max = delta_max
        self.v_max = v_max
        self.r_hard = r_hard
        self.r_safe = r_safe
        self.z_slack = z_slack
        self.n_sfc_faces = int(n_sfc_faces)
        self.n_state = 4
        n = self.N + 1
        F = self.n_sfc_faces

        opti = ca.Opti("conic")
        Z = opti.variable(4, n)
        U = opti.variable(2, self.N)
        z0 = opti.parameter(4)
        z_ref = opti.parameter(4, n)
        z_tgt = opti.parameter(4)
        z_hold = opti.parameter(4, n)
        Ad = opti.parameter(4 * 4, self.N)
        Bd = opti.parameter(4 * 2, self.N)
        rd = opti.parameter(4, self.N)
        Hx = opti.parameter(F, n)
        Hy = opti.parameter(F, n)
        Hh = opti.parameter(F, n)
        n_ball_cuts = 4
        Gx = opti.parameter(n_ball_cuts, n)
        Gy = opti.parameter(n_ball_cuts, n)
        Gh_b = opti.parameter(n_ball_cuts, n)

        opti.subject_to(Z[:, 0] == z0)
        a, delta = U[0, :], U[1, :]
        opti.subject_to(opti.bounded(-a_max, a, a_max))
        opti.subject_to(opti.bounded(-delta_max, delta, delta_max))
        opti.subject_to(opti.bounded(0.0, Z[3, :], v_max))

        for k in range(self.N):
            Ak = ca.reshape(Ad[:, k], 4, 4)
            Bk = ca.reshape(Bd[:, k], 4, 2)
            opti.subject_to(Z[:, k + 1] == Ak @ Z[:, k] + Bk @ U[:, k] + rd[:, k])

        for k in range(1, n):
            for f in range(F):
                opti.subject_to(
                    Hx[f, k] * Z[0, k] + Hy[f, k] * Z[1, k] <= Hh[f, k]
                )
        for k in range(1, n):
            for j in range(n_ball_cuts):
                opti.subject_to(
                    Gx[j, k] * Z[0, k] + Gy[j, k] * Z[1, k] <= Gh_b[j, k]
                )

        cost = 0
        e_x, e_y = Z[0, :] - z_ref[0, :], Z[1, :] - z_ref[1, :]
        e_psi, e_v = Z[2, :] - z_ref[2, :], Z[3, :] - z_ref[3, :]
        for k in range(n):
            wp = w_pos if k < self.N else (w_pos + w_term_pos)
            wy = w_yaw if k < self.N else (w_yaw + w_term_yaw)
            cost = cost + wp * (e_x[k] ** 2 + e_y[k] ** 2)
            cost = cost + wy * (e_psi[k] ** 2)
            cost = cost + w_v * (e_v[k] ** 2)
        cost = cost + w_term_pos * (
            (Z[0, self.N] - z_tgt[0]) ** 2 + (Z[1, self.N] - z_tgt[1]) ** 2
        )
        cost = cost + w_term_yaw * (Z[2, self.N] - z_tgt[2]) ** 2
        for k in range(self.N):
            cost = cost + w_u * (a[k] ** 2 + delta[k] ** 2)
        for k in range(self.N - 1):
            cost = cost + w_du * (
                (a[k + 1] - a[k]) ** 2 + (delta[k + 1] - delta[k]) ** 2
            )
        e_h_x, e_h_y = Z[0, :] - z_hold[0, :], Z[1, :] - z_hold[1, :]
        cost = cost + w_hold * ca.sumsqr(e_h_x) + w_hold * ca.sumsqr(e_h_y)
        opti.minimize(cost)
        opti.solver(
            "osqp",
            {
                "print_time": 0,
                "error_on_fail": False,
                "warm_start_primal": True,
                "warm_start_dual": True,
            },
            {
                "verbose": False,
                "polish": True,
                "eps_abs": 1e-4,
                "eps_rel": 1e-4,
                "max_iter": 8000,
            },
        )

        self.opti = opti
        self.Z, self.U = Z, U
        self.z0, self.z_ref, self.z_tgt = z0, z_ref, z_tgt
        self.z_hold = z_hold
        self.Ad, self.Bd, self.rd = Ad, Bd, rd
        self.Hx, self.Hy, self.Hh = Hx, Hy, Hh
        self.Gx, self.Gy, self.Gh_b = Gx, Gy, Gh_b
        self.n_ball_cuts = n_ball_cuts
        self._last_Z = None
        self._last_U = None
        self._last_u0 = np.zeros(2)
        self._last_ego_xy_yaw = None

    def _pack_ref(self, ref_t, ref_R, v: float) -> np.ndarray:
        n = self.N + 1
        t = np.asarray(ref_t, dtype=np.float64).reshape(n, 3)
        R = np.asarray(ref_R, dtype=np.float64).reshape(n, 3, 3)
        yaw = np.unwrap([yaw_from_R(Rk) for Rk in R])
        z = np.zeros((4, n), dtype=np.float64)
        z[0, :] = t[:, 0]
        z[1, :] = t[:, 1]
        z[2, :] = yaw
        z[3, :] = float(np.clip(v, 0.4, self.v_max))
        z[0, 0] = 0.0
        z[1, 0] = 0.0
        z[2, 0] = 0.0
        return z, t[:, 2].copy()

    def _halfspaces(
        self, poly_A, poly_b, ref_R, z_height, keep_xy=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, F = self.N + 1, self.n_sfc_faces
        Hx = np.zeros((F, n), dtype=np.float64)
        Hy = np.zeros((F, n), dtype=np.float64)
        Hh = np.full((F, n), 1.0e6, dtype=np.float64)
        if poly_A is None or poly_b is None:
            return Hx, Hy, Hh
        PA = np.asarray(poly_A, dtype=np.float64).reshape(F * 3, n)
        Pb = np.asarray(poly_b, dtype=np.float64).reshape(F, n)
        Rall = np.asarray(ref_R, dtype=np.float64).reshape(n, 3, 3)
        if keep_xy is not None:
            keep_xy = np.asarray(keep_xy, dtype=np.float64).reshape(2, n)
        for k in range(n):
            A = PA[:, k].reshape(F, 3)
            b = Pb[:, k]
            if float(np.max(np.abs(A))) < 1e-12:
                continue
            xy = None if keep_xy is None else keep_xy[:, k]
            hx, hy, h = shrink_halfspaces(
                A, b, Rall[k], self.body, self.radii, float(z_height[k]),
                margin=self.r_hard,
                keep_xy=xy,
            )
            Hx[:, k] = hx
            Hy[:, k] = hy
            Hh[:, k] = h
        return Hx, Hy, Hh

    def _ball_cuts(
        self, esdf, z_ref, z_height, ref_R,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, nb = self.N + 1, self.n_ball_cuts
        Gx = np.zeros((nb, n), dtype=np.float64)
        Gy = np.zeros((nb, n), dtype=np.float64)
        Gh = np.full((nb, n), 1.0e6, dtype=np.float64)
        Rall = np.asarray(ref_R, dtype=np.float64).reshape(n, 3, 3)
        h = 0.25
        for k in range(1, n):
            t = np.array(
                [float(z_ref[0, k]), float(z_ref[1, k]), float(z_height[k])],
                dtype=np.float64,
            )
            balls = transform_body_se3(t, Rall[k], self.body)
            d = query_esdf_z_slack(
                esdf, self.grid, balls, z_slack=self.z_slack,
            ) - self.radii
            order = np.argsort(d)[:nb]
            for j, i in enumerate(order):
                if float(d[i]) > 1.2:
                    continue
                p = balls[int(i)]
                probe = np.vstack([
                    p,
                    p + np.array([h, 0.0, 0.0]),
                    p + np.array([-h, 0.0, 0.0]),
                    p + np.array([0.0, h, 0.0]),
                    p + np.array([0.0, -h, 0.0]),
                ])
                e = query_esdf(esdf, self.grid, probe)
                nxy = np.array([
                    float(e[1] - e[2]) / (2.0 * h),
                    float(e[3] - e[4]) / (2.0 * h),
                ])
                nrm = float(np.linalg.norm(nxy))
                if nrm < 0.05:
                    nxy = p[:2] - t[:2]
                    nrm = float(np.linalg.norm(nxy))
                if nrm < 1e-6:
                    continue
                nxy = nxy / nrm
                bvec = p[:2] - t[:2]
                rhs = float(self.radii[int(i)] + self.r_hard) - float(nxy @ bvec)
                # Do not cut through the reference COM (keeps the QP feasible).
                n_dot = float(nxy[0] * z_ref[0, k] + nxy[1] * z_ref[1, k])
                if n_dot < rhs:
                    rhs = n_dot - 0.05
                Gx[j, k] = -nxy[0]
                Gy[j, k] = -nxy[1]
                Gh[j, k] = -rhs
        return Gx, Gy, Gh

    def _linearize_along(self, z_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        yaw = z_ref[2, :]
        v = z_ref[3, :]
        delta = _path_delta(yaw, v, self.dt, self.L, self.delta_max)
        Ad = np.zeros((16, self.N), dtype=np.float64)
        Bd = np.zeros((8, self.N), dtype=np.float64)
        rd = np.zeros((4, self.N), dtype=np.float64)
        for k in range(self.N):
            u = np.array([0.0, float(delta[k])])
            A, B, r = _linearize_bicycle(z_ref[:, k], u, self.L, self.dt)
            Ad[:, k] = A.reshape(16, order="F")
            Bd[:, k] = B.reshape(8, order="F")
            rd[:, k] = r
        return Ad, Bd, rd

    def solve(
        self,
        v: float,
        esdf: np.ndarray,
        target: np.ndarray,
        target_R: np.ndarray,
        ref_t: np.ndarray,
        ref_R: np.ndarray,
        poly_A: np.ndarray | None = None,
        poly_b: np.ndarray | None = None,
        ego_xy_yaw=None,
    ) -> dict:
        v = float(np.clip(v, 0.0, self.v_max))
        n = self.N + 1
        t_tgt = np.asarray(target, dtype=np.float64).reshape(3)
        R_tgt = np.asarray(target_R, dtype=np.float64).reshape(3, 3)
        z_ref, z_height = self._pack_ref(ref_t, ref_R, self.v_max)
        z_tgt = np.array([
            float(t_tgt[0]), float(t_tgt[1]),
            yaw_from_R(R_tgt), float(self.v_max),
        ], dtype=np.float64)
        tgt_balls = transform_body_se3(t_tgt, R_tgt, self.body)
        d_tgt = query_esdf_z_slack(
            esdf, self.grid, tgt_balls, z_slack=self.z_slack,
        ) - self.radii
        if float(np.min(d_tgt)) < self.r_hard:
            z_tgt = z_ref[:, -1].copy()
        z0 = np.array([0.0, 0.0, 0.0, v], dtype=np.float64)
        Ad, Bd, rd = self._linearize_along(z_ref)
        Hx, Hy, Hh = self._halfspaces(
            poly_A, poly_b, ref_R, z_height, keep_xy=z_ref[:2, :],
        )
        Gx, Gy, Gh_b = self._ball_cuts(esdf, z_ref, z_height, ref_R)

        z_hold = z_ref.copy()
        if self._last_Z is not None:
            Zg, Ug = self._last_Z.copy(), self._last_U.copy()
            if (
                ego_xy_yaw is not None
                and self._last_ego_xy_yaw is not None
                and ego_xy_yaw != self._last_ego_xy_yaw
            ):
                Zg = transform_states_between_ego(
                    Zg, self._last_ego_xy_yaw, ego_xy_yaw,
                )
            Zg[:, 0] = z0
            z_hold = Zg.copy()
            z_hold[:, 0] = z0
        else:
            Zg, Ug = z_ref.copy(), np.zeros((2, self.N))
            Ug[1, :] = _path_delta(z_ref[2, :], z_ref[3, :], self.dt, self.L, self.delta_max)[: self.N]
            Zg[:, 0] = z0

        self.opti.set_value(self.z0, z0)
        self.opti.set_value(self.z_ref, z_ref)
        self.opti.set_value(self.z_tgt, z_tgt)
        self.opti.set_value(self.z_hold, z_hold)
        self.opti.set_value(self.Ad, Ad)
        self.opti.set_value(self.Bd, Bd)
        self.opti.set_value(self.rd, rd)
        self.opti.set_value(self.Hx, Hx)
        self.opti.set_value(self.Hy, Hy)
        self.opti.set_value(self.Hh, Hh)
        self.opti.set_value(self.Gx, Gx)
        self.opti.set_value(self.Gy, Gy)
        self.opti.set_value(self.Gh_b, Gh_b)
        self.opti.set_initial(self.Z, Zg)
        self.opti.set_initial(self.U, Ug)

        t0 = time.perf_counter()
        used_fallback = False
        status = "ok"
        Z, U = Zg, Ug
        try:
            sol = self.opti.solve()
            status = str(self.opti.stats().get("return_status", "solved"))
            Zv = np.array(sol.value(self.Z)).reshape(4, n)
            Uv = np.array(sol.value(self.U)).reshape(2, self.N)
            ok = status.lower() in {"solved", "solved_accurate", "optimal"} and np.isfinite(Zv).all()
            if ok:
                Z, U = Zv, Uv
            else:
                used_fallback = True
                if np.isfinite(Zv).all():
                    Z, U = Zv, Uv
        except Exception:
            used_fallback = True
            try:
                status = str(self.opti.stats().get("return_status", "fail"))
            except Exception:
                status = "fail"
            status = f"fail:{status}"
        solve_ms = (time.perf_counter() - t0) * 1000.0

        if not used_fallback and np.isfinite(Z).all():
            self._last_Z, self._last_U = Z.copy(), U.copy()
            self._last_ego_xy_yaw = ego_xy_yaw
        u0 = U[:, 0].copy() if np.isfinite(U[:, 0]).all() else self._last_u0
        self._last_u0 = u0

        pts_t = np.column_stack([Z[0, :], Z[1, :], z_height])
        pts_R = np.stack([rotz(float(wrap_angle(p))) for p in Z[2, :]], axis=0)
        speed = Z[3, :]
        traj4 = np.column_stack([Z[0, :], Z[1, :], np.array([wrap_angle(p) for p in Z[2, :]]), speed])
        balls = transform_body_se3(pts_t, pts_R, self.body).reshape(-1, 3)
        clearance = query_esdf_z_slack(
            esdf, self.grid, balls, z_slack=self.z_slack,
        ) - np.tile(self.radii, n)
        min_clearance = float(np.min(clearance)) if clearance.size else float("inf")
        terminal_err = float(np.linalg.norm(R_tgt.T @ (pts_t[-1] - t_tgt)))
        return {
            "traj": traj4,
            "traj_vis": traj4,
            "traj_t": pts_t.copy(),
            "traj_R": pts_R.copy(),
            "state": Z.T.copy(),
            "x0": z0.copy(),
            "u": U.T.copy(),
            "u0": u0,
            "status": status,
            "solve_ms": solve_ms,
            "used_fallback": used_fallback,
            "min_clearance": min_clearance,
            "terminal_err": terminal_err,
        }
