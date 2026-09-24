"""Unit tests for the stereo VA control head (no CARLA)."""

from __future__ import annotations

import unittest

import numpy as np

from stereo_bev.query_heads import HAS_TORCH

if HAS_TORCH:
    import torch
    from stereo_bev.bev_grid import DEFAULT_X_RANGE, DEFAULT_Y_RANGE
    from stereo_bev.query_heads import ControlQueryHead, stereo_bev_loss


@unittest.skipUnless(HAS_TORCH, "PyTorch required")
class TestControlQueryHead(unittest.TestCase):
    def test_output_ranges_and_shape(self):
        head = ControlQueryHead(
            8, grid_z=4, x_range=DEFAULT_X_RANGE, y_range=DEFAULT_Y_RANGE,
        )
        bev = torch.randn(2, 8, 10, 10)
        occ = torch.randn(2, 4, 10, 10)
        target = torch.tensor([
            [5.0, 0.0, 0.1, 2.0],
            [3.0, -1.0, -0.2, 0.0],
        ])
        control = head(bev, occ, target)
        self.assertEqual(tuple(control.shape), (2, 3))
        self.assertTrue(torch.all(control[:, :2] >= 0.0))
        self.assertTrue(torch.all(control[:, :2] <= 1.0))
        self.assertTrue(torch.all(control[:, 2] >= -1.0))
        self.assertTrue(torch.all(control[:, 2] <= 1.0))
        both = (control[:, 0] > 1e-6) & (control[:, 1] > 1e-6)
        self.assertFalse(bool(both.any()))

    def test_squash_keeps_stronger_pedal(self):
        from stereo_bev.query_heads import squash_control
        out = squash_control(torch.tensor([[0.0, 2.0, 0.0], [2.0, 0.0, 0.0]]))
        self.assertAlmostEqual(float(out[0, 0]), 0.0, places=5)
        self.assertGreater(float(out[0, 1]), 0.5)
        self.assertGreater(float(out[1, 0]), 0.5)
        self.assertAlmostEqual(float(out[1, 1]), 0.0, places=5)

    def test_masked_control_loss_does_not_train_on_dummy(self):
        B, C, H, W = 2, 3, 4, 4
        seg_logits = torch.zeros(B, C, H, W)
        seg_gt = torch.zeros(B, H, W, dtype=torch.long)
        occ_logits = torch.zeros(B, 2, 3, 3)
        occ_gt = torch.zeros(B, 2, 3, 3)
        bev_seg_logits = torch.zeros(B, C, 2, 3, 3)
        occupancy_seg_gt = torch.zeros(B, C, 2, 3, 3)
        pred = torch.ones(B, 3, requires_grad=True)
        gt = torch.zeros(B, 3)
        has = torch.zeros(B)
        out = stereo_bev_loss(
            seg_logits, occ_logits, bev_seg_logits, seg_gt, occ_gt, occupancy_seg_gt,
            control_pred=pred, control_gt=gt, has_control=has,
        )
        self.assertEqual(float(out["control_loss"].detach()), 0.0)
        self.assertFalse(out["loss"].requires_grad)

    def test_control_loss_trains_when_labeled(self):
        B, C, H, W = 1, 3, 4, 4
        seg_logits = torch.zeros(B, C, H, W)
        seg_gt = torch.zeros(B, H, W, dtype=torch.long)
        occ_logits = torch.zeros(B, 2, 3, 3)
        occ_gt = torch.zeros(B, 2, 3, 3)
        bev_seg_logits = torch.zeros(B, C, 2, 3, 3)
        occupancy_seg_gt = torch.zeros(B, C, 2, 3, 3)
        pred = torch.ones(B, 3, requires_grad=True)
        gt = torch.zeros(B, 3)
        has = torch.ones(B)
        out = stereo_bev_loss(
            seg_logits, occ_logits, bev_seg_logits, seg_gt, occ_gt, occupancy_seg_gt,
            control_pred=pred, control_gt=gt, has_control=has,
        )
        self.assertGreater(float(out["control_loss"].detach()), 0.0)
        out["loss"].backward()
        self.assertIsNotNone(pred.grad)
        self.assertGreater(float(pred.grad.abs().sum()), 0.0)

    def test_act_from_feat_samples_in_range(self):
        head = ControlQueryHead(
            8, grid_z=4, x_range=DEFAULT_X_RANGE, y_range=DEFAULT_Y_RANGE,
        )
        bev = torch.randn(4, 8, 10, 10)
        occ = torch.randn(4, 4, 10, 10)
        target = torch.zeros(4, 4)
        target[:, 0] = 4.0
        feat = head.encode(bev, occ, target)
        control, logp, value, z, ent = head.act_from_feat(feat, deterministic=False)
        self.assertEqual(tuple(control.shape), (4, 3))
        self.assertEqual(tuple(logp.shape), (4,))
        self.assertEqual(tuple(value.shape), (4,))
        self.assertTrue(torch.all(control[:, :2] >= 0.0) and torch.all(control[:, :2] <= 1.0))
        lp2, v2, _ = head.evaluate_z(feat, z)
        self.assertTrue(torch.allclose(logp, lp2, atol=1e-5))
        self.assertTrue(torch.allclose(value, v2, atol=1e-5))


class TestControlRL(unittest.TestCase):
    def test_gae_terminal_zero_bootstrap(self):
        from stereo_bev.control_rl import gae, nearest_forward_m, step_reward
        rewards = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        values = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        dones = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        adv, ret = gae(rewards, values, dones, gamma=0.99, lam=1.0, last_value=99.0)
        # last done=1 so last_value must not leak into returns
        self.assertAlmostEqual(float(ret[-1]), 1.0, places=5)
        self.assertLess(float(ret[0]), 99.0)
        r = step_reward(1.0, 0.0, 2.0, collided=True)
        self.assertEqual(r, -100.0)
        alive = step_reward(0.0, 8.0, 0.0, collided=False)
        crash = step_reward(10.0, 0.0, 8.0, collided=True)
        self.assertGreater(alive, crash)
        idle = step_reward(0.0, 0.0, 0.0, collided=False)
        moving = step_reward(0.0, 0.0, 4.0, collided=False)
        self.assertGreater(moving, idle)
        close = step_reward(0.0, 0.0, 2.0, collided=False,
                            target_xy=np.array([4.0, 0.0]), heading_rad=0.0)
        far = step_reward(0.0, 0.0, 2.0, collided=False,
                          target_xy=np.array([40.0, 40.0]), heading_rad=1.6)
        self.assertGreater(close, far)
        still_crash = step_reward(0.0, 0.0, 2.0, collided=True,
                                  target_xy=np.array([1.0, 0.0]))
        self.assertEqual(still_crash, -100.0)
        from stereo_bev.control_rl import dodge_target, pedal_safety
        dodged = dodge_target(np.array([8.0, 0.0, 0.0, 3.0]), np.array([[8.0, 0.0]]))
        self.assertLess(float(dodged[1]), 0.0)
        safe = pedal_safety(np.array([0.8, 0.0, 0.0]), np.array([[3.0, 0.0]]))
        self.assertGreater(float(safe[1]), 0.4)
        self.assertEqual(float(safe[0]), 0.0)

    def test_occ_safety_brakes_for_blocked_front(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch required")
        head = ControlQueryHead(
            8, grid_z=4, x_range=DEFAULT_X_RANGE, y_range=DEFAULT_Y_RANGE,
        )
        ctrl = torch.tensor([[0.8, 0.0, 0.0]])
        blocked = torch.full((1, 4, 10, 10), 4.0)
        out = head.apply_occ_safety(ctrl, blocked)
        self.assertGreater(float(out[0, 1]), 0.4)
        self.assertLess(float(out[0, 0]), 0.3)

    @unittest.skipUnless(HAS_TORCH, "PyTorch required")
    def test_ppo_update_changes_policy(self):
        from stereo_bev.control_rl import gae, ppo_update
        head = ControlQueryHead(
            4, grid_z=2, x_range=DEFAULT_X_RANGE, y_range=DEFAULT_Y_RANGE,
        )
        feat = torch.randn(32, head.feat_dim)
        with torch.no_grad():
            control, logp, value, z, _ = head.act_from_feat(feat)
        rewards = np.ones(32, dtype=np.float64)
        dones = np.zeros(32, dtype=np.float64)
        dones[-1] = 1.0
        adv, ret = gae(
            rewards, value.detach().cpu().numpy(), dones, last_value=0.0,
        )
        adv_t = torch.from_numpy(adv)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        before = [p.detach().clone() for p in head.mlp.parameters()]
        ppo_update(
            head, feat, z, logp.detach(), adv_t, torch.from_numpy(ret),
            epochs=2, minibatch=16, lr=1e-3,
        )
        changed = any(
            not torch.allclose(a, b) for a, b in zip(before, head.mlp.parameters())
        )
        self.assertTrue(changed)
        self.assertIsNotNone(control)


if __name__ == "__main__":
    unittest.main()
