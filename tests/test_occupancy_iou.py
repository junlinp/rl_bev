"""Occupancy voxel / BEV IoU used as the occupancy evaluation metric."""

import unittest

import torch

from stereo_bev.query_heads import occupancy_iou_counts, occupancy_iou_from_counts


def _logits_from_mask(mask):
    return torch.where(mask, torch.tensor(10.0), torch.tensor(-10.0))


class OccupancyIoUTests(unittest.TestCase):
    def test_perfect_3d_and_bev(self):
        gt = torch.zeros(2, 3, 4, 4)
        gt[0, 1, 2, 2] = 1
        gt[1, 0, 0, 1] = 1
        inter, union, inter_bev, union_bev = occupancy_iou_counts(_logits_from_mask(gt.bool()), gt)
        self.assertEqual(occupancy_iou_from_counts(inter, union), 1.0)
        self.assertEqual(occupancy_iou_from_counts(inter_bev, union_bev), 1.0)

    def test_partial_overlap(self):
        gt = torch.zeros(1, 2, 2, 2)
        pred = torch.zeros(1, 2, 2, 2)
        gt[0, 0, 0, 0] = 1
        gt[0, 0, 0, 1] = 1
        pred[0, 0, 0, 0] = 1
        pred[0, 1, 1, 1] = 1
        inter, union, _, _ = occupancy_iou_counts(_logits_from_mask(pred.bool()), gt)
        # 1 intersection, 3 union
        self.assertAlmostEqual(occupancy_iou_from_counts(inter, union), 1.0 / 3.0)

    def test_bev_collapses_z(self):
        gt = torch.zeros(1, 2, 2, 2)
        pred = torch.zeros(1, 2, 2, 2)
        gt[0, 0, 0, 0] = 1
        pred[0, 1, 0, 0] = 1  # same XY, different Z
        inter, union, inter_bev, union_bev = occupancy_iou_counts(
            _logits_from_mask(pred.bool()), gt,
        )
        self.assertEqual(occupancy_iou_from_counts(inter, union), 0.0)
        self.assertEqual(occupancy_iou_from_counts(inter_bev, union_bev), 1.0)

    def test_empty_is_one(self):
        gt = torch.zeros(1, 2, 2, 2)
        inter, union, inter_bev, union_bev = occupancy_iou_counts(
            _logits_from_mask(gt.bool()), gt,
        )
        self.assertEqual(occupancy_iou_from_counts(inter, union), 1.0)
        self.assertEqual(occupancy_iou_from_counts(inter_bev, union_bev), 1.0)

    def test_dataset_micro_average(self):
        a_gt = torch.zeros(1, 1, 2, 2)
        a_pred = torch.zeros(1, 1, 2, 2)
        a_gt[0, 0, 0, 0] = 1
        a_pred[0, 0, 0, 0] = 1
        b_gt = torch.zeros(1, 1, 2, 2)
        b_pred = torch.zeros(1, 1, 2, 2)
        b_gt[0, 0, 0, 0] = 1
        b_pred[0, 0, 1, 1] = 1

        inter = union = 0
        for logits, gt in (
            (_logits_from_mask(a_pred.bool()), a_gt),
            (_logits_from_mask(b_pred.bool()), b_gt),
        ):
            i, u, _, _ = occupancy_iou_counts(logits, gt)
            inter += i
            union += u
        # batch A: 1/1, batch B: 0/2 → micro 1/3
        self.assertAlmostEqual(occupancy_iou_from_counts(inter, union), 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
