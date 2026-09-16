"""Offline tests for 3D occupancy ESDF + CasADi NMPC (no CARLA)."""

from __future__ import annotations

import time
import unittest

import numpy as np

from stereo_bev.bev_grid import BEVGrid
from stereo_bev.global_target import sample_target_and_reference, transform_states_between_ego
from stereo_bev.occ_field import occupancy_to_esdf_3d, query_esdf
from stereo_bev.vehicle_body import model3_body_samples, transform_body


def _grid() -> BEVGrid:
    return BEVGrid()


def _empty_occ(grid: BEVGrid) -> np.ndarray:
    return np.zeros((grid.grid_z, grid.grid_h, grid.grid_w), dtype=np.uint8)


def _fill_box(occ, grid, x0, x1, y0, y1, z0, z1):
    vs = grid.voxel_size
    xi0 = int(np.clip((x0 - grid.x_range[0]) / vs, 0, grid.grid_w - 1))
    xi1 = int(np.clip((x1 - grid.x_range[0]) / vs, 0, grid.grid_w))
    yi0 = int(np.clip((y0 - grid.y_range[0]) / vs, 0, grid.grid_h - 1))
    yi1 = int(np.clip((y1 - grid.y_range[0]) / vs, 0, grid.grid_h))
    zi0 = int(np.clip((z0 - grid.z_range[0]) / vs, 0, grid.grid_z - 1))
    zi1 = int(np.clip((z1 - grid.z_range[0]) / vs, 0, grid.grid_z))
    occ[zi0:zi1, yi0:yi1, xi0:xi1] = 1


class TestOccField(unittest.TestCase):
    def test_empty_volume_large_distance(self):
        grid = _grid()
        esdf = occupancy_to_esdf_3d(_empty_occ(grid), grid, z_ground=0.3, inflate_m=0)
        self.assertGreater(float(esdf.min()), 5.0)

    def test_ground_only_ignored(self):
        grid = _grid()
        occ = _empty_occ(grid)
        _fill_box(occ, grid, 0.0, 20.0, -10.0, 10.0, -1.0, 0.25)
        esdf = occupancy_to_esdf_3d(occ, grid, z_ground=0.3, inflate_m=0)
        self.assertGreater(float(esdf.min()), 5.0)

    def test_obstacle_has_zero_distance(self):
        grid = _grid()
        occ = _empty_occ(grid)
        _fill_box(occ, grid, 8.0, 9.0, -0.5, 0.5, 0.4, 1.8)
        esdf = occupancy_to_esdf_3d(occ, grid, z_ground=0.3, inflate_m=0)
        self.assertLess(float(esdf.min()), 1e-6)
        d = query_esdf(esdf, grid, np.array([[8.5, 0.0, 1.0]]))
        self.assertLess(float(d[0]), 0.3)

    def test_ego_footprint_cleared(self):
        grid = _grid()
        occ = _empty_occ(grid)
        _fill_box(occ, grid, 0.0, 3.0, -1.5, 1.5, 0.4, 1.5)
        esdf = occupancy_to_esdf_3d(occ, grid, z_ground=0.3, inflate_m=0)
        d = query_esdf(esdf, grid, np.array([[1.0, 0.0, 0.8]]))
        self.assertGreater(float(d[0]), 0.5)

    def test_road_semantics_ignored(self):
        grid = _grid()
        occ = np.ones((grid.grid_z, grid.grid_h, grid.grid_w), dtype=np.uint8)
        classes = np.ones((grid.grid_h, grid.grid_w), dtype=np.uint8)
        esdf = occupancy_to_esdf_3d(
            occ, grid, z_ground=0.3, inflate_m=0, bev_classes=classes,
        )
        self.assertGreater(float(esdf.min()), 5.0)


class TestVehicleBody(unittest.TestCase):
    def test_sample_count_and_bounds(self):
        body = model3_body_samples(3, 3, 3)
        self.assertEqual(body.shape, (27, 3))
        self.assertLess(body[:, 2].max(), 1.5)
        self.assertGreater(body[:, 2].min(), 0.1)

    def test_transform_identity(self):
        body = model3_body_samples(2, 2, 2)
        out = transform_body(0.0, 0.0, 0.0, body)
        np.testing.assert_allclose(out, body, atol=1e-9)


class TestGlobalTargetSampling(unittest.TestCase):
    def test_five_second_lookahead(self):
        x = np.linspace(0, 40, 41)
        xy = np.stack([x, np.zeros_like(x)], axis=1)
        yaw = np.zeros(41)
        out = sample_target_and_reference(xy, yaw, v=4.0, horizon_s=5.0, n_knots=26, d_min=8.0, d_max=25.0)
        # 4 m/s * 5 s = 20 m
        self.assertAlmostEqual(out["target"][0], 20.0, delta=0.6)
        self.assertEqual(out["p_ref"].shape, (26, 2))
        self.assertAlmostEqual(out["p_ref"][-1, 0], out["target"][0], delta=1e-6)

    def test_stopped_spreads_reference(self):
        x = np.linspace(0, 40, 41)
        xy = np.stack([x, np.zeros_like(x)], axis=1)
        yaw = np.zeros(41)
        out = sample_target_and_reference(xy, yaw, v=0.0, horizon_s=5.0, n_knots=26, d_min=8.0, d_max=25.0)
        self.assertAlmostEqual(out["target"][0], 8.0, delta=0.6)
        self.assertGreater(float(out["p_ref"][12, 0]), 2.0)
        self.assertGreater(float(out["p_ref"][-2, 0]), 4.0)

    def test_ego_frame_shift(self):
        # 1 m ahead in last ego becomes the origin after driving 1 m forward
        states = np.array([[1.0], [0.0], [0.0], [5.0]])
        out = transform_states_between_ego(states, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        np.testing.assert_allclose(out[:2, 0], [0.0, 0.0], atol=1e-9)


def _try_nmpc():
    from stereo_bev.nmpc import HAS_CASADI, OccupancyNMPC
    if not HAS_CASADI:
        return None, None, None
    grid = _grid()
    body = model3_body_samples()
    nmpc = OccupancyNMPC(grid=grid, body=body, horizon_s=5.0, dt=0.2, ipopt_max_iter=80, v_max=12.0)
    return nmpc, grid, body


class TestNMPCOffline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nmpc, cls.grid, cls.body = _try_nmpc()

    def _solve(self, occ, target, v=4.0):
        self.assertIsNotNone(self.nmpc, "casadi is required for NMPC tests")
        self.nmpc._last_X = None
        self.nmpc._last_U = None
        self.nmpc._last_ego_xy_yaw = None
        self.nmpc._last_solve_s = None
        esdf = occupancy_to_esdf_3d(occ, self.grid, z_ground=0.3)
        n = self.nmpc.N + 1
        p_ref = np.stack([
            np.linspace(0.0, target[0], n),
            np.linspace(0.0, target[1], n),
        ], axis=1)
        yaw_ref = np.full(n, np.arctan2(target[1], target[0]) if abs(target[0]) + abs(target[1]) > 0.1 else 0.0)
        t0 = time.perf_counter()
        sol = self.nmpc.solve(v=v, esdf=esdf, target=np.array(target), p_ref=p_ref, yaw_ref=yaw_ref)
        sol["build_note"] = ""
        print(
            f"    NMPC {sol['status']}  {sol['solve_ms']:.0f}ms  "
            f"terr={sol['terminal_err']:.2f}  dmin={sol['min_clearance']:.2f}  "
            f"wall={time.perf_counter() - t0:.2f}s",
            flush=True,
        )
        return sol

    def test_empty_reaches_target(self):
        if self.nmpc is None:
            self.skipTest("casadi not installed")
        occ = _empty_occ(self.grid)
        target = (12.0, 0.0, 0.0)
        sol = self._solve(occ, target)
        self.assertLess(sol["terminal_err"], 3.0)
        self.assertGreater(sol["traj"][-1, 0], 8.0)

    def test_tall_wall_bends(self):
        if self.nmpc is None:
            self.skipTest("casadi not installed")
        occ = _empty_occ(self.grid)
        _fill_box(occ, self.grid, 7.5, 8.5, -1.6, 1.6, 0.35, 2.4)
        target = (16.0, 0.0, 0.0)
        sol = self._solve(occ, target)
        y_abs = np.abs(sol["traj"][:, 1])
        self.assertGreater(float(y_abs.max()), 1.0)
        self.assertGreater(sol["min_clearance"], -0.05)

    def test_overhang_allows_straight(self):
        if self.nmpc is None:
            self.skipTest("casadi not installed")
        occ = _empty_occ(self.grid)
        # occupied only above the roof — 2D max-over-Z would block this
        _fill_box(occ, self.grid, 6.0, 14.0, -4.0, 4.0, 1.65, 2.8)
        target = (16.0, 0.0, 0.0)
        sol = self._solve(occ, target)
        y_abs = np.abs(sol["traj"][:, 1])
        self.assertLess(float(y_abs.max()), 1.5)
        self.assertFalse(sol["used_fallback"], sol["status"])
        self.assertLess(sol["terminal_err"], 4.0)

    def test_ground_only_ignored_by_planner(self):
        if self.nmpc is None:
            self.skipTest("casadi not installed")
        occ = _empty_occ(self.grid)
        _fill_box(occ, self.grid, 0.0, 20.0, -10.0, 10.0, -1.0, 0.25)
        target = (12.0, 1.0, 0.0)
        sol = self._solve(occ, target)
        self.assertLess(sol["terminal_err"], 4.0)

    def test_replan_starts_at_200ms_knot(self):
        if self.nmpc is None:
            self.skipTest("casadi not installed")
        occ = _empty_occ(self.grid)
        target = (12.0, 0.0, 0.0)
        sol1 = self._solve(occ, target)
        # second solve must NOT reset last trajectory
        esdf = occupancy_to_esdf_3d(occ, self.grid, z_ground=0.3)
        n = self.nmpc.N + 1
        p_ref = np.stack([np.linspace(0.0, target[0], n), np.zeros(n)], axis=1)
        sol2 = self.nmpc.solve(
            v=4.0, esdf=esdf, target=np.array(target), p_ref=p_ref,
            yaw_ref=np.zeros(n),
        )
        expected = sol1["traj"][1]  # knot at dt = 0.2 s
        np.testing.assert_allclose(sol2["x0"][:3], expected[:3], atol=1e-3)
        np.testing.assert_allclose(sol2["traj"][0, :3], expected[:3], atol=1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
