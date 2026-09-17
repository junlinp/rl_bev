"""Offline tests for 3D occupancy ESDF + CasADi NMPC (no CARLA)."""

from __future__ import annotations

import math
import time
import unittest

import numpy as np

from stereo_bev.bev_grid import BEVGrid
from stereo_bev.global_target import (
    route_world_to_ego,
    sample_target_and_reference,
    transform_states_between_ego,
    world_heading_to_ego,
    world_to_ego_bev,
)
from stereo_bev.occ_field import (
    occupancy_to_esdf_3d, occupied_to_esdf, obstacle_volume, query_esdf,
    stamp_oriented_boxes, shift_ref_from_esdf, shift_ref_se3,
)
from stereo_bev.sfc import (
    assign_polyhedra, astar_3d, build_safe_flight_corridor, ellipsoid_polyhedron,
    pack_sfc_parameters, point_in_polyhedron, refs_along_path,
)
from stereo_bev.vehicle_body import (
    MODEL3_HEIGHT, MODEL3_LENGTH, MODEL3_WIDTH,
    model3_body_samples, model3_collision_balls, transform_body, transform_body_se3,
    collision_inflation_radius,
)
from run_nmpc import _lane_offset_xy


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

    def test_esdf_is_3d_not_xy_collapse(self):
        grid = _grid()
        occ = _empty_occ(grid)
        _fill_box(occ, grid, 8.0, 10.0, -1.0, 1.0, 1.6, 2.6)
        esdf = occupied_to_esdf(occ.astype(bool), grid)
        self.assertEqual(esdf.ndim, 3)
        self.assertEqual(esdf.shape, occ.shape)
        d_high = query_esdf(esdf, grid, np.array([[9.0, 0.0, 2.1]]))
        d_low = query_esdf(esdf, grid, np.array([[9.0, 0.0, 0.6]]))
        self.assertLess(float(d_high[0]), 0.5)
        self.assertGreater(float(d_low[0]), 0.7)

    def test_z_slack_ignores_low_road_keeps_tall_wall(self):
        from stereo_bev.occ_field import query_esdf_z_slack
        grid = _grid()
        occ = _empty_occ(grid)
        _fill_box(occ, grid, 6.0, 14.0, -3.0, 3.0, 0.35, 0.65)
        esdf = occupied_to_esdf(occ.astype(bool), grid)
        p = np.array([[10.0, 0.0, 0.8]])
        d0 = float(query_esdf(esdf, grid, p)[0])
        d_s = float(query_esdf_z_slack(esdf, grid, p, z_slack=0.30)[0])
        self.assertLess(d0, 0.4)
        self.assertGreater(d_s, 0.4)
        occ2 = _empty_occ(grid)
        _fill_box(occ2, grid, 6.0, 14.0, -1.2, 1.2, 0.3, 2.2)
        esdf2 = occupied_to_esdf(occ2.astype(bool), grid)
        d_wall = float(query_esdf_z_slack(esdf2, grid, p, z_slack=0.30)[0])
        self.assertLess(d_wall, 0.3)

    def test_vehicle_to_occupancy_shifts_xy_only(self):
        from stereo_bev.calibration import vehicle_to_occupancy
        p = np.array([[1.5, 0.06, 0.8], [8.0, 1.0, 0.5]])
        out = vehicle_to_occupancy(p, (1.5, 0.06))
        np.testing.assert_allclose(out[0], [0.0, 0.0, 0.8])
        np.testing.assert_allclose(out[1], [6.5, 0.94, 0.5])

    def test_ego_footprint_cleared(self):
        grid = _grid()
        occ = _empty_occ(grid)
        _fill_box(occ, grid, 0.0, 3.0, -1.5, 1.5, 0.4, 1.5)
        esdf = occupancy_to_esdf_3d(occ, grid, z_ground=0.3, inflate_m=0)
        d = query_esdf(esdf, grid, np.array([[1.0, 0.0, 0.8]]))
        self.assertGreater(float(d[0]), 0.5)

    def test_ahead_of_bumper_not_cleared(self):
        grid = _grid()
        occ = _empty_occ(grid)
        _fill_box(occ, grid, 2.5, 3.4, -0.4, 0.4, 0.5, 1.4)
        kept = obstacle_volume(occ, grid, z_ground=0.3, inflate_m=0)
        self.assertGreater(int(kept.sum()), 5)
        d = query_esdf(
            occupancy_to_esdf_3d(occ, grid, z_ground=0.3, inflate_m=0),
            grid, np.array([[2.8, 0.0, 0.9]]),
        )
        self.assertLess(float(d[0]), 0.4)

    def test_stamp_box_marks_voxels(self):
        grid = _grid()
        occ = _empty_occ(grid)
        kept = stamp_oriented_boxes(
            occ.astype(bool), grid,
            centers=np.array([[10.0, 0.0, 0.8]]),
            halves=np.array([[1.5, 0.8, 0.7]]),
            yaws=np.array([0.0]),
            inflate_m=0.0,
        )
        self.assertGreater(int(kept.sum()), 20)
        d = query_esdf(
            occupied_to_esdf(kept, grid),
            grid, np.array([[10.0, 0.0, 0.8]]),
        )
        self.assertLess(float(d[0]), 0.3)

    def test_shift_ref_goes_around_blocker(self):
        grid = _grid()
        occ = _empty_occ(grid)
        _fill_box(occ, grid, 8.0, 10.0, -1.2, 1.2, 0.4, 1.8)
        esdf = occupancy_to_esdf_3d(occ, grid, z_ground=0.3, inflate_m=0)
        p_ref = np.stack([np.linspace(0.0, 16.0, 26), np.zeros(26)], axis=1)
        shifted = shift_ref_from_esdf(p_ref, esdf, grid, r_need=1.0)
        mid = shifted[(shifted[:, 0] > 8.0) & (shifted[:, 0] < 10.0)]
        self.assertGreater(float(np.max(np.abs(mid[:, 1]))), 1.2)
        t0 = np.column_stack([p_ref, np.zeros(len(p_ref))])
        R0 = np.repeat(np.eye(3)[None, ...], len(p_ref), axis=0)
        t1, _ = shift_ref_se3(t0, R0, esdf, grid, r_need=1.0)
        mid3 = t1[(t1[:, 0] > 8.0) & (t1[:, 0] < 10.0)]
        self.assertGreater(float(np.max(np.abs(mid3[:, 1]))), 1.2)

    def test_road_semantics_ignored(self):
        grid = _grid()
        occ = _empty_occ(grid)
        _fill_box(occ, grid, 0.0, 20.0, -10.0, 10.0, -1.0, 0.25)
        voxel_class = np.ones((grid.grid_z, grid.grid_h, grid.grid_w), dtype=np.uint8)
        esdf = occupancy_to_esdf_3d(
            occ, grid, z_ground=0.3, inflate_m=0, voxel_class=voxel_class,
        )
        self.assertGreater(float(esdf.min()), 5.0)

    def test_car_kept_when_labeled_road(self):
        grid = _grid()
        occ = _empty_occ(grid)
        _fill_box(occ, grid, 8.0, 12.0, -1.0, 1.0, 0.6, 1.6)
        voxel_class = np.ones((grid.grid_z, grid.grid_h, grid.grid_w), dtype=np.uint8)
        kept = obstacle_volume(
            occ, grid, z_ground=0.5, inflate_m=0, voxel_class=voxel_class,
        )
        self.assertGreater(int(kept.sum()), 50)
        d = query_esdf(
            occupancy_to_esdf_3d(occ, grid, z_ground=0.5, inflate_m=0, voxel_class=voxel_class),
            grid, np.array([[10.0, 0.0, 1.0]]),
        )
        self.assertLess(float(d[0]), 0.5)

    def test_thin_pole_kept_when_road_is_argmax(self):
        grid = _grid()
        occ = _empty_occ(grid)
        _fill_box(occ, grid, 8.0, 8.4, -0.2, 0.2, 0.4, 2.5)
        zi, yi, xi = np.argwhere(occ > 0)[0]
        classes = np.ones((grid.grid_h, grid.grid_w), dtype=np.uint8)
        kept_even_if_road = obstacle_volume(
            occ, grid, z_ground=0.3, inflate_m=0, bev_classes=classes,
        )
        voxel_class = np.ones((grid.grid_z, grid.grid_h, grid.grid_w), dtype=np.uint8)
        voxel_class[occ > 0] = 8
        kept = obstacle_volume(
            occ, grid, z_ground=0.3, inflate_m=0, voxel_class=voxel_class,
        )
        self.assertTrue(bool(kept_even_if_road[zi, yi, xi]))
        self.assertTrue(bool(kept[zi, yi, xi]))

    def test_pole_voxel_not_extruded_through_z(self):
        grid = _grid()
        occ = _empty_occ(grid)
        voxel_class = np.zeros((grid.grid_z, grid.grid_h, grid.grid_w), dtype=np.uint8)
        vs = grid.voxel_size
        zi = int((1.2 - grid.z_range[0]) / vs)
        yi, xi = grid.grid_h // 2, int((8.2 - grid.x_range[0]) / vs)
        voxel_class[zi, yi, xi] = 8
        kept = obstacle_volume(
            occ, grid, z_ground=0.3, inflate_m=0, voxel_class=voxel_class,
        )
        self.assertTrue(bool(kept[zi, yi, xi]))
        self.assertFalse(bool(kept[:, yi, xi].all()))


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

    def test_transform_se3_matches_planar_yaw(self):
        from stereo_bev.global_target import rotz
        body = model3_body_samples(2, 2, 2)
        planar = transform_body(3.0, -1.0, 0.4, body)
        se3 = transform_body_se3(np.array([3.0, -1.0, 0.0]), rotz(0.4), body)
        np.testing.assert_allclose(se3, planar, atol=1e-9)

    def test_transform_se3_pitch_lifts_forward(self):
        pitch = 0.2
        c, s = math.cos(pitch), math.sin(pitch)
        R = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=np.float64)
        body = np.array([[2.0, 0.0, 0.5]])
        out = transform_body_se3(np.zeros(3), R, body)[0]
        self.assertGreater(float(out[2]), 0.5)
        self.assertLess(float(out[0]), 2.0)

    def test_collision_balls_cover_bbox_corners(self):
        centers, radii = model3_collision_balls()
        self.assertEqual(centers.shape[0], radii.shape[0])
        self.assertGreaterEqual(centers.shape[0], 8)
        L, W, H = MODEL3_LENGTH, MODEL3_WIDTH, MODEL3_HEIGHT
        corners = np.array([
            [sx * 0.5 * L, sy * 0.5 * W, sz * H]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (0.0, 1.0)
        ])
        for p in corners:
            gap = np.min(np.linalg.norm(centers - p, axis=1) - radii)
            self.assertLessEqual(float(gap), 1e-6, msg=f"corner {p} not covered ({gap:.3f})")


class TestGlobalTargetSampling(unittest.TestCase):
    def test_five_second_lookahead(self):
        x = np.linspace(0, 40, 41)
        xy = np.stack([x, np.zeros_like(x)], axis=1)
        yaw = np.zeros(41)
        out = sample_target_and_reference(xy, yaw, v=4.0, horizon_s=5.0, n_knots=26, d_min=8.0, d_max=25.0)
        # 4 m/s * 5 s = 20 m
        self.assertAlmostEqual(out["target"][0], 20.0, delta=0.6)
        self.assertEqual(out["p_ref"].shape, (26, 2))
        self.assertEqual(out["ref_t"].shape, (26, 3))
        self.assertEqual(out["ref_R"].shape, (26, 3, 3))
        self.assertAlmostEqual(out["p_ref"][-1, 0], out["target"][0], delta=1e-6)
        np.testing.assert_allclose(out["ref_t"][-1], out["target_t"], atol=1e-6)

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

    def test_ego_frame_yaw_90(self):
        # 1 m ahead at yaw=0 is 1 m to the left after a +90° CARLA yaw
        states = np.array([[1.0], [0.0], [0.0], [5.0]])
        out = transform_states_between_ego(
            states, (0.0, 0.0, 0.0), (0.0, 0.0, 0.5 * np.pi),
        )
        np.testing.assert_allclose(out[:2, 0], [0.0, 1.0], atol=1e-9)

    def test_route_drops_passed_after_heading_change(self):
        # World route along +X, then +Y. Vehicle at the corner facing +Y.
        wx = np.array([0.0, 5.0, 10.0, 10.0, 10.0])
        wy = np.array([0.0, 0.0, 0.0, 5.0, 10.0])
        wyaw = np.array([0.0, 0.0, 0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi])
        xy, yaw = route_world_to_ego(wx, wy, wyaw, (10.0, 0.0, 0.5 * np.pi))
        # Passed +X segment would sit at y≈-10 after the turn; it must be gone.
        self.assertFalse(np.any(np.abs(xy[:, 1] + 10.0) < 1.0))
        self.assertGreater(float(xy[-1, 0]), 8.0)
        self.assertLess(float(np.max(np.abs(xy[:, 1]))), 1.0)
        np.testing.assert_allclose(xy[0], [0.0, 0.0], atol=1e-6)
        out = sample_target_and_reference(
            xy, yaw, v=2.0, horizon_s=5.0, n_knots=26, d_min=5.0, d_max=25.0,
        )
        self.assertGreater(float(out["target"][0]), 4.0)
        self.assertLess(abs(float(out["target"][1])), 1.0)
        self.assertEqual(out["target_t"].shape, (3,))
        self.assertEqual(out["target_R"].shape, (3, 3))

    def test_offset_lane_keeps_forward_heading(self):
        # Lane 1.5 m left, heading +X. Must not build a 90° chord from the car.
        x = np.linspace(0.3, 20.0, 20)
        xy = np.stack([x, np.full_like(x, 1.5)], axis=1)
        yaw = np.zeros(20)
        out = sample_target_and_reference(
            xy, yaw, v=2.0, horizon_s=5.0, n_knots=26, d_min=5.0, d_max=25.0,
        )
        self.assertGreater(float(out["target"][0]), 4.0)
        self.assertLess(abs(float(out["yaw_ref"][1])), 0.25)
        self.assertLess(abs(float(out["yaw_ref"][8])), 0.25)
        self.assertGreater(float(out["p_ref"][4, 0]), abs(float(out["p_ref"][4, 1] - 1.5)) + 1.0)

    def test_loop_does_not_snap_target_onto_car(self):
        # Outbound +X, return 2 m left. Vehicle sits between the two legs.
        wx = np.array([0.0, 10.0, 20.0, 20.0, 10.0, 0.0])
        wy = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0])
        wyaw = np.array([0.0, 0.0, 0.0, np.pi, np.pi, np.pi])
        pose = (10.0, 1.2, 0.0)
        xy, yaw = route_world_to_ego(wx, wy, wyaw, pose)
        out = sample_target_and_reference(
            xy, yaw, v=2.0, horizon_s=5.0, n_knots=26, d_min=5.0, d_max=25.0,
        )
        self.assertGreater(float(out["target"][0]), 4.0)
        self.assertLess(float(out["target"][1]), 1.5)

    def test_short_remaining_keeps_target_ahead(self):
        xy = np.array([[0.0, 0.0], [0.4, 0.0]])
        yaw = np.zeros(2)
        out = sample_target_and_reference(
            xy, yaw, v=2.0, horizon_s=5.0, n_knots=26, d_min=5.0, d_max=25.0,
        )
        self.assertGreater(float(out["target"][0]), 4.5)

    def test_world_se3_to_ego_yflip(self):
        from stereo_bev.global_target import world_se3_to_ego
        pose = (0.0, 0.0, 0.0)
        t, R = world_se3_to_ego(
            np.array([10.0, 0.0, 1.0]), np.eye(3), pose,
        )
        np.testing.assert_allclose(t, [10.0, 0.0, 1.0], atol=1e-9)
        t_r, _ = world_se3_to_ego(
            np.array([0.0, 4.0, 0.0]), np.eye(3), pose,
        )
        np.testing.assert_allclose(t_r, [0.0, -4.0, 0.0], atol=1e-9)

    def test_se3_matches_yaw_only_pose(self):
        wx = np.array([5.0, 8.0])
        wy = np.array([1.0, -2.0])
        pose3 = (1.0, 2.0, 0.4)
        pose6 = (1.0, 2.0, 0.0, 0.0, 0.4, 0.0)
        a = world_to_ego_bev(wx, wy, pose3)
        b = world_to_ego_bev(wx, wy, pose6)
        np.testing.assert_allclose(a, b, atol=1e-9)
        np.testing.assert_allclose(
            world_heading_to_ego(np.array([0.4, 0.9]), pose6),
            np.array([0.0, -0.5]),
            atol=1e-9,
        )

    def test_se3_pitch_projects_forward_xy(self):
        pitch = np.deg2rad(10.0)
        pose = (0.0, 0.0, 0.0, pitch, 0.0, 0.0)
        x, y = world_to_ego_bev(np.array([10.0]), np.array([0.0]), pose, wz=0.0)
        self.assertAlmostEqual(float(x[0]), 10.0 * math.cos(pitch), places=6)
        self.assertAlmostEqual(float(y[0]), 0.0, places=6)


def _try_nmpc():
    from stereo_bev.nmpc import HAS_CASADI, OccupancyNMPC
    if not HAS_CASADI:
        return None, None, None
    grid = _grid()
    body, radii = model3_collision_balls()
    nmpc = OccupancyNMPC(
        grid=grid, body=body, radii=radii, horizon_s=5.0, dt=0.2, ipopt_max_iter=100, v_max=12.0,
    )
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
        r_need = float(self.nmpc.radii.max()) + self.nmpc.r_safe
        p_ref = shift_ref_from_esdf(p_ref, esdf, self.grid, r_need=r_need)
        yaw_ref = np.full(n, np.arctan2(target[1], target[0]) if abs(target[0]) + abs(target[1]) > 0.1 else 0.0)
        dxy = np.diff(p_ref, axis=0)
        if len(dxy):
            yaw_ref = np.concatenate([
                [np.arctan2(dxy[0, 1], dxy[0, 0])],
                np.arctan2(dxy[:, 1], dxy[:, 0]),
            ])
        target = (float(p_ref[-1, 0]), float(p_ref[-1, 1]), float(yaw_ref[-1]))
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

    def test_obstacle_on_global_path_swerves(self):
        """Parked-car box on the lane-center global path; 5 s target stays on it."""
        if self.nmpc is None:
            self.skipTest("casadi not installed")
        occ = _empty_occ(self.grid)
        # Vehicle-sized blocker sitting on the global path at ~10 m.
        _fill_box(occ, self.grid, 9.0, 12.0, -1.15, 1.15, 0.4, 1.7)
        esdf = occupancy_to_esdf_3d(occ, self.grid, z_ground=0.3)
        n = self.nmpc.N + 1
        p_path = np.stack([np.linspace(0.0, 16.0, n), np.zeros(n)], axis=1)
        yaw_path = np.zeros(n)
        r_need = float(self.nmpc.radii.max()) + self.nmpc.r_safe
        p_ref = shift_ref_from_esdf(p_path, esdf, self.grid, r_need=r_need)
        dxy = np.diff(p_ref, axis=0)
        yaw_ref = np.concatenate([
            [0.0],
            np.arctan2(dxy[:, 1], dxy[:, 0]),
        ]) if len(dxy) else yaw_path
        # Keep the terminal knot on the global path (same as run_nmpc).
        p_ref[-1] = p_path[-1]
        yaw_ref[-1] = 0.0
        target = np.array([16.0, 0.0, 0.0], dtype=np.float64)
        self.nmpc._last_X = None
        self.nmpc._last_U = None
        self.nmpc._last_ego_xy_yaw = None
        sol = self.nmpc.solve(
            v=4.0, esdf=esdf, target=target, p_ref=p_ref, yaw_ref=yaw_ref,
        )
        print(
            f"    NMPC {sol['status']}  {sol['solve_ms']:.0f}ms  "
            f"terr={sol['terminal_err']:.2f}  dmin={sol['min_clearance']:.2f}  "
            f"ymax={float(np.max(np.abs(sol['traj'][:, 1]))):.2f}",
            flush=True,
        )
        mid = sol["traj"][(sol["traj"][:, 0] > 8.0) & (sol["traj"][:, 0] < 13.0)]
        self.assertGreater(len(mid), 2)
        self.assertGreater(float(np.max(np.abs(mid[:, 1]))), 1.3)
        self.assertGreater(sol["min_clearance"], -0.15)
        self.assertGreater(sol["traj"][-1, 0], 10.0)
        self.assertLess(abs(float(sol["traj"][-1, 1])), 3.5)

    def test_sfc_corridor_swerves_around_blocker(self):
        """3D SFC finds a homotopy; NMPC tracks it inside the polyhedra."""
        if self.nmpc is None:
            self.skipTest("casadi not installed")
        occ = _empty_occ(self.grid)
        _fill_box(occ, self.grid, 8.0, 10.5, -1.2, 1.2, 0.4, 1.8)
        esdf = occupancy_to_esdf_3d(occ, self.grid, z_ground=0.3)
        r_need = float(self.nmpc.radii.max()) + self.nmpc.r_safe
        start = np.array([0.0, 0.0, 0.7])
        goal = np.array([16.0, 0.0, 0.7])
        sfc = build_safe_flight_corridor(
            occ.astype(bool), self.grid, start, goal,
            r_inflate=r_need, esdf=esdf, n_faces=self.nmpc.n_sfc_faces,
        )
        self.assertTrue(sfc["ok"], "SFC A* should go around the box")
        self.assertGreater(float(np.max(np.abs(sfc["path"][:, 1]))), 1.2)
        n = self.nmpc.N + 1
        ref_t, ref_R = refs_along_path(
            sfc["path"], n, v=4.0, horizon_s=5.0, d_min=5.0, d_max=16.0,
        )
        assigned = assign_polyhedra(
            ref_t, sfc.get("path_poly", sfc["path"]), sfc["polyhedra"],
        )
        for t, (A, b) in zip(ref_t, assigned):
            self.assertTrue(point_in_polyhedron(t, A, b, tol=0.15), t)
        poly_A, poly_b = pack_sfc_parameters(assigned, n, self.nmpc.n_sfc_faces)
        t_tgt = np.array([16.0, 0.0, 0.7], dtype=np.float64)
        R_tgt = np.eye(3, dtype=np.float64)
        ref_t[-1] = t_tgt
        ref_R[-1] = R_tgt
        self.nmpc._last_X = None
        self.nmpc._last_U = None
        self.nmpc._last_ego_xy_yaw = None
        sol = self.nmpc.solve(
            v=4.0, esdf=esdf, target=t_tgt, target_R=R_tgt,
            ref_t=ref_t, ref_R=ref_R, poly_A=poly_A, poly_b=poly_b,
        )
        print(
            f"    SFC-NMPC {sol['status']}  {sol['solve_ms']:.0f}ms  "
            f"terr={sol['terminal_err']:.2f}  dmin={sol['min_clearance']:.2f}  "
            f"ymax={float(np.max(np.abs(sol['traj'][:, 1]))):.2f}  "
            f"n_poly={sfc['n_poly']}",
            flush=True,
        )
        self.assertFalse(sol["used_fallback"], sol["status"])
        mid = sol["traj"][(sol["traj"][:, 0] > 7.0) & (sol["traj"][:, 0] < 12.0)]
        self.assertGreater(len(mid), 1)
        self.assertGreater(float(np.max(np.abs(mid[:, 1]))), 1.2)
        self.assertGreater(sol["min_clearance"], -0.2)

    def test_overhang_allows_straight(self):
        if self.nmpc is None:
            self.skipTest("casadi not installed")
        occ = _empty_occ(self.grid)
        # occupied well above the roof — balls reach ~roof+r, so keep this high
        _fill_box(occ, self.grid, 6.0, 14.0, -4.0, 4.0, 2.05, 2.8)
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

    def test_replan_starts_at_current_ego(self):
        if self.nmpc is None:
            self.skipTest("casadi not installed")
        occ = _empty_occ(self.grid)
        target = (12.0, 0.0, 0.0)
        self._solve(occ, target)
        esdf = occupancy_to_esdf_3d(occ, self.grid, z_ground=0.3)
        n = self.nmpc.N + 1
        p_ref = np.stack([np.linspace(0.0, target[0], n), np.zeros(n)], axis=1)
        sol2 = self.nmpc.solve(
            v=4.0, esdf=esdf, target=np.array(target), p_ref=p_ref,
            yaw_ref=np.zeros(n),
        )
        np.testing.assert_allclose(sol2["x0"][:3], [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(sol2["traj"][0, :3], [0.0, 0.0, 0.0], atol=1e-5)

    def test_state_lives_on_se3(self):
        if self.nmpc is None:
            self.skipTest("casadi not installed")
        from stereo_bev.global_target import exp_so3
        occ = _empty_occ(self.grid)
        sol = self._solve(occ, (12.0, 0.0, 0.0))
        st = sol["state"]
        self.assertEqual(st.shape[1], 9)
        R = exp_so3(st[-1, 3:6])
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-5)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=4)
        self.assertLess(abs(float(st[-1, 2])), 0.6)
        np.testing.assert_allclose(sol["traj_t"][0], [0.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(sol["x0"][6:9], [4.0, 0.0, 0.0], atol=1e-6)
        vel = st[-1, 6:9]
        body_v = R.T @ vel
        self.assertGreater(float(body_v[0]), 0.3)
        self.assertLess(abs(float(body_v[1])), 0.35)
        self.assertLess(abs(float(body_v[2])), 0.35)

    def test_spatial_velocity_rotates_with_heading(self):
        if self.nmpc is None:
            self.skipTest("casadi not installed")
        import casadi as ca
        from stereo_bev.global_target import exp_so3
        nmpc = self.nmpc
        z_s = ca.MX.sym("z", nmpc.n_state)
        u_s = ca.MX.sym("u", 2)
        step = ca.Function("se23_step", [z_s, u_s], [nmpc._integrate_se3(z_s, u_s, nmpc.dt)])
        z0 = np.zeros(nmpc.n_state)
        z0[6] = 2.0
        z1 = np.array(step(z0, np.array([0.0, 0.35]))).reshape(-1)
        R = exp_so3(z1[3:6])
        vel = z1[6:9]
        s = float(R[:, 0] @ vel)
        np.testing.assert_allclose(vel, R @ np.array([s, 0.0, 0.0]), atol=1e-6)
        self.assertGreater(abs(float(vel[1])), 0.01)
        self.assertGreater(s, 1.5)


class TestRerunVis(unittest.TestCase):
    def test_voxel_log_without_viewer(self):
        try:
            from stereo_bev.rerun_vis import RerunOccViewer, is_available
        except ImportError:
            self.skipTest("stereo_bev.rerun_vis missing")
        if not is_available():
            self.skipTest("rerun-sdk not installed")
        grid = _grid()
        viewer = RerunOccViewer(grid, application_id="test_nmpc_rerun", spawn=False)
        occ = _empty_occ(grid)
        _fill_box(occ, grid, 8.0, 8.4, -0.2, 0.2, 0.4, 2.2)
        classes = np.ones((grid.grid_h, grid.grid_w), dtype=np.uint8)
        classes[:] = 8  # pole_sign
        viewer.log_frame(
            0,
            occ=occ,
            sweep=np.zeros_like(occ),
            bev_classes=classes,
            traj_xy=np.array([[0.0, 0.0], [10.0, 0.0]]),
            route_xy=np.array([[0.0, 0.0], [12.0, 1.0]]),
            target_xy=np.array([12.0, 1.0]),
            rgb_bgr=np.zeros((32, 48, 3), dtype=np.uint8),
            depth=np.ones((32, 48), dtype=np.float32),
            bev_bgr=np.zeros((40, 40, 3), dtype=np.uint8),
            speed=1.5, accel=0.2, steer=0.05, clearance=3.0,
        )
        viewer.log_frame(1, occ=_empty_occ(grid))
        viewer.close()

    def test_voxel_class_keeps_pole_above_road(self):
        grid = _grid()
        pts = np.array([[8.1, 0.0, 0.0], [8.1, 0.0, 1.5]], dtype=np.float64)
        labels = np.array([1, 8], dtype=np.uint8)
        out = grid.voxelize(pts, class_labels=labels)
        zi_r = int((0.0 - grid.z_range[0]) / grid.voxel_size)
        zi_p = int((1.5 - grid.z_range[0]) / grid.voxel_size)
        xi, yi = grid.world_to_grid(np.array([8.1]), np.array([0.0]))
        xi, yi = int(xi[0]), int(yi[0])
        self.assertEqual(int(out["voxel_class"][zi_r, yi, xi]), 1)
        self.assertEqual(int(out["voxel_class"][zi_p, yi, xi]), 8)
        self.assertEqual(int(out["class_histogram"].argmax(axis=0)[yi, xi]), 1)


class TestSFC(unittest.TestCase):
    def test_astar_goes_around_wall(self):
        grid = _grid()
        occ = _empty_occ(grid)
        _fill_box(occ, grid, 8.0, 10.0, -4.0, 4.0, 0.3, 2.2)
        esdf = occupied_to_esdf(occ.astype(bool), grid)
        blocked = esdf < 0.9
        path = astar_3d(
            blocked, grid,
            np.array([1.0, 0.0, 0.7]), np.array([16.0, 0.0, 0.7]),
        )
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 3)
        self.assertGreater(float(np.max(np.abs(path[:, 1]))), 3.5)
        self.assertGreater(float(path[-1, 0]), 14.0)

    def test_polyhedron_contains_segment_excludes_obstacle(self):
        grid = _grid()
        occ = _empty_occ(grid)
        _fill_box(occ, grid, 9.0, 11.0, -0.6, 0.6, 0.4, 1.8)
        from stereo_bev.occ_field import esdf_axis_grids
        zi, yi, xi = np.nonzero(occ)
        xg, yg, zg = esdf_axis_grids(grid)
        obs = np.stack([xg[xi], yg[yi], zg[zi]], axis=1)
        p0 = np.array([2.0, 2.5, 0.7])
        p1 = np.array([6.0, 2.5, 0.7])
        A, b = ellipsoid_polyhedron(p0, p1, obs, grid)
        self.assertTrue(point_in_polyhedron(p0, A, b, tol=0.08))
        self.assertTrue(point_in_polyhedron(p1, A, b, tol=0.08))
        self.assertTrue(point_in_polyhedron(0.5 * (p0 + p1), A, b, tol=0.08))
        # Obstacle center is not inside the corridor grown around y=+2.5.
        self.assertFalse(point_in_polyhedron(np.array([10.0, 0.0, 1.0]), A, b))

    def test_empty_sfc_reaches_goal(self):
        grid = _grid()
        occ = _empty_occ(grid)
        sfc = build_safe_flight_corridor(
            occ.astype(bool), grid,
            np.array([0.0, 0.0, 0.7]), np.array([12.0, 0.0, 0.7]),
            r_inflate=0.94,
        )
        self.assertTrue(sfc["ok"])
        self.assertGreater(sfc["n_poly"], 0)
        self.assertGreater(float(sfc["path"][-1, 0]), 10.0)
        self.assertLess(float(np.max(np.abs(sfc["path"][:, 1]))), 1.5)

    def test_prefer_path_keeps_homotopy(self):
        grid = _grid()
        occ = _empty_occ(grid)
        _fill_box(occ, grid, 8.0, 10.5, -1.2, 1.2, 0.4, 1.8)
        esdf = occupied_to_esdf(occ.astype(bool), grid)
        start = np.array([0.0, 0.0, 0.7])
        goal = np.array([16.0, 0.0, 0.7])
        left = np.array([
            [0.0, 0.0, 0.7],
            [4.0, 2.4, 0.7],
            [8.0, 2.4, 0.7],
            [12.0, 2.4, 0.7],
            [16.0, 0.0, 0.7],
        ])
        sfc_l = build_safe_flight_corridor(
            occ.astype(bool), grid, start, goal,
            r_inflate=0.94, esdf=esdf, prefer_path=left,
        )
        self.assertTrue(sfc_l["ok"])
        mid_l = sfc_l["path"][(sfc_l["path"][:, 0] > 7.0) & (sfc_l["path"][:, 0] < 11.0)]
        self.assertGreater(len(mid_l), 0)
        self.assertGreater(float(np.min(mid_l[:, 1])), 1.0)
        right = left.copy()
        right[:, 1] *= -1.0
        sfc_r = build_safe_flight_corridor(
            occ.astype(bool), grid, start, goal,
            r_inflate=0.94, esdf=esdf, prefer_path=right,
        )
        self.assertTrue(sfc_r["ok"])
        mid_r = sfc_r["path"][(sfc_r["path"][:, 0] > 7.0) & (sfc_r["path"][:, 0] < 11.0)]
        self.assertGreater(len(mid_r), 0)
        self.assertLess(float(np.max(mid_r[:, 1])), -1.0)


class TestCorridorQP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from stereo_bev.corridor_qp import HAS_CASADI, CorridorQP
        cls.grid = _grid()
        body, radii = model3_collision_balls()
        cls.body = body
        if not HAS_CASADI:
            cls.qp = None
            return
        cls.qp = CorridorQP(
            grid=cls.grid, body=body, radii=radii,
            horizon_s=5.0, dt=0.2, v_max=2.0,
        )

    def test_empty_tracks_vmax_horizon(self):
        if self.qp is None:
            self.skipTest("casadi not installed")
        occ = _empty_occ(self.grid)
        esdf = occupancy_to_esdf_3d(occ, self.grid, z_ground=0.3)
        n = self.qp.N + 1
        t_tgt = np.array([10.0, 0.0, 0.7])
        R_tgt = np.eye(3)
        ref_t = np.zeros((n, 3))
        ref_t[:, 0] = np.linspace(0.0, 10.0, n)
        ref_t[:, 2] = 0.7
        ref_R = np.repeat(np.eye(3)[None, ...], n, axis=0)
        self.qp._last_Z = None
        self.qp._last_U = None
        sol = self.qp.solve(
            v=2.0, esdf=esdf, target=t_tgt, target_R=R_tgt,
            ref_t=ref_t, ref_R=ref_R,
        )
        print(
            f"    QP empty {sol['status']}  {sol['solve_ms']:.0f}ms  "
            f"terr={sol['terminal_err']:.2f}  xmax={sol['traj'][-1, 0]:.2f}",
            flush=True,
        )
        self.assertFalse(sol["used_fallback"], sol["status"])
        self.assertGreater(float(sol["traj"][-1, 0]), 6.0)

    def test_hard_polyhedra_swerve(self):
        if self.qp is None:
            self.skipTest("casadi not installed")
        occ = _empty_occ(self.grid)
        _fill_box(occ, self.grid, 8.0, 10.5, -1.2, 1.2, 0.4, 1.8)
        esdf = occupancy_to_esdf_3d(occ, self.grid, z_ground=0.3)
        r_need = collision_inflation_radius(
            self.qp.body, self.qp.radii, extra=self.qp.r_hard,
        )
        sfc = build_safe_flight_corridor(
            occ.astype(bool), self.grid,
            np.array([0.0, 0.0, 0.7]), np.array([10.0, 0.0, 0.7]),
            r_inflate=r_need, esdf=esdf, n_faces=self.qp.n_sfc_faces,
            body=self.qp.body, radii=self.qp.radii, r_clear=self.qp.r_hard + 0.15,
        )
        self.assertTrue(sfc["ok"])
        n = self.qp.N + 1
        ref_t, ref_R = refs_along_path(
            sfc["path"], n, v=self.qp.v_max, horizon_s=5.0, d_min=5.0, d_max=10.0,
        )
        assigned = assign_polyhedra(
            ref_t, sfc.get("path_poly", sfc["path"]), sfc["polyhedra"],
        )
        for t, (A, b) in zip(ref_t, assigned):
            self.assertTrue(point_in_polyhedron(t, A, b, tol=0.15), t)
        poly_A, poly_b = pack_sfc_parameters(assigned, n, self.qp.n_sfc_faces)
        t_tgt = np.array([10.0, 0.0, 0.7])
        R_tgt = np.eye(3)
        self.qp._last_Z = None
        self.qp._last_U = None
        sol = self.qp.solve(
            v=2.0, esdf=esdf, target=t_tgt, target_R=R_tgt,
            ref_t=ref_t, ref_R=ref_R, poly_A=poly_A, poly_b=poly_b,
        )
        print(
            f"    QP-SFC {sol['status']}  {sol['solve_ms']:.0f}ms  "
            f"terr={sol['terminal_err']:.2f}  dmin={sol['min_clearance']:.2f}  "
            f"ymax={float(np.max(np.abs(sol['traj'][:, 1]))):.2f}",
            flush=True,
        )
        self.assertFalse(sol["used_fallback"], sol["status"])
        self.assertGreater(sol["min_clearance"], -0.15)
        mid = sol["traj"][(sol["traj"][:, 0] > 6.5) & (sol["traj"][:, 0] < 11.0)]
        self.assertGreater(len(mid), 1)
        self.assertGreater(float(np.max(np.abs(mid[:, 1]))), 1.0)
        z_ref, z_h = self.qp._pack_ref(ref_t, ref_R, self.qp.v_max)
        Hx, Hy, Hh = self.qp._halfspaces(
            poly_A, poly_b, ref_R, z_h, keep_xy=z_ref[:2, :],
        )
        xy = sol["traj"][:, :2]
        max_viol = 0.0
        for k in range(1, n):
            viol = Hx[:, k] * xy[k, 0] + Hy[:, k] * xy[k, 1] - Hh[:, k]
            max_viol = max(max_viol, float(np.max(viol)))
        self.assertLess(max_viol, 1e-3, f"hard SFC violated by {max_viol:.4f}m")


class TestRoadObstacles(unittest.TestCase):
    def test_heading_right_is_carla_plus_y(self):
        # yaw 0: +X forward, +Y right. lat=+1.5 → world (x, y+1.5)
        x, y = _lane_offset_xy(10.0, 5.0, 0.0, 1.5)
        np.testing.assert_allclose([x, y], [10.0, 6.5])

    def test_lane_offset_right_at_yaw90(self):
        x, y = _lane_offset_xy(0.0, 0.0, 0.5 * np.pi, 2.0)
        np.testing.assert_allclose([x, y], [-2.0, 0.0], atol=1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
