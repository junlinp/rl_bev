"""Offline tests for occupancy field, body, global target, and VA spawn helpers (no CARLA)."""

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
from stereo_bev.vehicle_body import (
    MODEL3_HEIGHT, MODEL3_LENGTH, MODEL3_WIDTH,
    model3_body_samples, model3_collision_balls, transform_body, transform_body_se3,
    collision_inflation_radius,
)


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
    def test_vehicle_origin_is_grid_center(self):
        grid = _grid()
        self.assertEqual(grid.grid_w, 100)
        self.assertEqual(grid.grid_h, 100)
        self.assertEqual(grid.origin_index(), (50, 50, 5))

    def test_empty_volume_large_distance(self):
        grid = _grid()
        esdf = occupancy_to_esdf_3d(_empty_occ(grid), grid, z_ground=0.3, inflate_m=0)
        self.assertGreater(float(esdf.min()), 5.0)

    def test_ground_only_ignored(self):
        grid = _grid()
        occ = _empty_occ(grid)
        _fill_box(occ, grid, grid.x_range[0], grid.x_range[1], grid.y_range[0], grid.y_range[1], -1.0, 0.25)
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

    def test_temporal_fusion_keeps_obstacle_after_ego_moves(self):
        from stereo_bev.occ_fusion import TemporalOccFusion

        grid = _grid()
        fusion = TemporalOccFusion(grid, decay=0.95, occ_thresh=0.25)
        occ = _empty_occ(grid)
        _fill_box(occ, grid, 4.0, 5.0, -0.4, 0.4, 0.5, 1.2)
        cls = np.zeros_like(occ)
        cls[occ > 0] = 3
        fused, fused_cls = fusion.update(occ, cls, (0.0, 0.0, 0.0))
        self.assertGreater(int(fused.sum()), 5)
        empty = _empty_occ(grid)
        fused, fused_cls = fusion.update(empty, empty, (2.0, 0.0, 0.0))
        xi, yi = grid.world_to_grid(np.array([2.5]), np.array([0.0]))
        zk = int((0.8 - grid.z_range[0]) / grid.voxel_size)
        self.assertEqual(int(fused[zk, int(yi[0]), int(xi[0])]), 1)
        self.assertEqual(int(fused_cls[zk, int(yi[0]), int(xi[0])]), 3)
        # Original 4 m cell is now behind the moved ego.
        xi4, yi4 = grid.world_to_grid(np.array([4.5]), np.array([0.0]))
        self.assertEqual(int(fused[zk, int(yi4[0]), int(xi4[0])]), 0)

    def test_temporal_fusion_decays_without_hits(self):
        from stereo_bev.occ_fusion import TemporalOccFusion

        grid = _grid()
        fusion = TemporalOccFusion(grid, decay=0.5, occ_thresh=0.25)
        occ = _empty_occ(grid)
        _fill_box(occ, grid, 3.0, 3.6, -0.3, 0.3, 0.5, 1.0)
        pose = (0.0, 0.0, 0.0)
        fusion.update(occ, None, pose)
        empty = _empty_occ(grid)
        last = None
        for _ in range(8):
            last, _ = fusion.update(empty, empty, pose)
        self.assertEqual(int(last.sum()), 0)

    def test_temporal_fusion_accumulates_counts_across_frames(self):
        from stereo_bev.occ_fusion import TemporalOccFusion

        grid = _grid()
        fusion = TemporalOccFusion(grid, decay=1.0, occ_thresh=2.0, hit=1.0)
        pose = (0.0, 0.0, 0.0)
        first = _empty_occ(grid)
        _fill_box(first, grid, 4.0, 5.0, -0.4, 0.4, 0.5, 1.2)
        fused, _ = fusion.update(first, None, pose)
        self.assertEqual(int(fused.sum()), 0)
        second = _empty_occ(grid)
        _fill_box(second, grid, 4.0, 5.0, -0.4, 0.4, 0.5, 1.2)
        fused, fused_cls = fusion.update(second, None, pose)
        self.assertGreater(int(fused.sum()), 5)
        xi, yi = grid.world_to_grid(np.array([4.5]), np.array([0.0]))
        zk = int((0.8 - grid.z_range[0]) / grid.voxel_size)
        self.assertEqual(int(fused[zk, int(yi[0]), int(xi[0])]), 1)

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
        p_ref = np.stack([np.linspace(0.0, 9.5, 26), np.zeros(26)], axis=1)
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
        _fill_box(occ, grid, grid.x_range[0], grid.x_range[1], grid.y_range[0], grid.y_range[1], -1.0, 0.25)
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



class TestRoadObstacles(unittest.TestCase):
    def test_heading_right_is_carla_plus_y(self):
        # yaw 0: +X forward, +Y right. lat=+1.5 → world (x, y+1.5)
        from run_planner import _lane_offset_xy
        x, y = _lane_offset_xy(10.0, 5.0, 0.0, 1.5)
        np.testing.assert_allclose([x, y], [10.0, 6.5])

    def test_lane_offset_right_at_yaw90(self):
        from run_planner import _lane_offset_xy
        x, y = _lane_offset_xy(0.0, 0.0, 0.5 * np.pi, 2.0)
        np.testing.assert_allclose([x, y], [-2.0, 0.0], atol=1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
