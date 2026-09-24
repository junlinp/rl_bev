"""Temporal fusion of ego-centric 3D occupancy.

Each new lift is added to the previous volume after a rigid XY/yaw warp
into the current vehicle frame. Hit counts decay so stale evidence fades;
voxels the camera cannot see this frame (behind / sides) stay until they
age out. The current grid is the thresholded sum of recent frames.
"""

from __future__ import annotations

import numpy as np

from .bev_grid import BEVGrid
from .global_target import transform_points_between_ego
from .occ_field import esdf_axis_grids, points_to_voxels


class TemporalOccFusion:
    """Receding occupancy memory in the current ego FLU grid."""

    def __init__(
        self,
        grid: BEVGrid,
        decay: float = 0.94,
        hit: float = 1.0,
        occ_thresh: float = 0.25,
    ):
        self.grid = grid
        self.decay = float(np.clip(decay, 0.0, 1.0))
        self.hit = float(hit)
        self.occ_thresh = float(occ_thresh)
        z, y, x = grid.grid_z, grid.grid_h, grid.grid_w
        self.evidence = np.zeros((z, y, x), dtype=np.float32)
        self.voxel_class = np.zeros((z, y, x), dtype=np.uint8)
        self._last_ego: tuple[float, float, float] | None = None

    def reset(self) -> None:
        self.evidence.fill(0.0)
        self.voxel_class.fill(0)
        self._last_ego = None

    def update(
        self,
        occ: np.ndarray,
        voxel_class: np.ndarray | None,
        ego_xy_yaw: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Warp memory into ``ego_xy_yaw``, decay it, and add this frame's hits.

        ``occ`` may be a hit-count volume or a 0/1 mask. Binary stamps add
        ``hit``; counts are accumulated as-is. Occupied = evidence >= occ_thresh.

        Returns fused binary occupancy and per-voxel class, both (Z, Y, X).
        """
        add = np.asarray(occ, dtype=np.float32)
        if add.shape != self.evidence.shape:
            raise ValueError(
                f"occupancy shape {add.shape} != fusion grid {self.evidence.shape}"
            )
        peak = float(np.max(add)) if add.size else 0.0
        if peak <= 1.0 + 1e-6:
            add = (add > 0).astype(np.float32) * np.float32(self.hit)
        observed = add > 0
        if self._last_ego is not None and ego_xy_yaw != self._last_ego:
            warped_e, warped_c = self._warp(self._last_ego, ego_xy_yaw)
        else:
            warped_e, warped_c = self.evidence, self.voxel_class

        evidence = (self.decay * warped_e).astype(np.float32, copy=True)
        evidence += add
        cls = warped_c.copy()
        if voxel_class is not None:
            vc = np.asarray(voxel_class)
            if vc.shape == evidence.shape:
                cls[observed] = vc[observed]
        else:
            cls[observed] = np.maximum(cls[observed], 1)

        self.evidence = evidence
        self.voxel_class = cls
        self._last_ego = (
            float(ego_xy_yaw[0]), float(ego_xy_yaw[1]), float(ego_xy_yaw[2]),
        )
        fused = (evidence >= self.occ_thresh).astype(np.uint8)
        fused_cls = cls.copy()
        fused_cls[fused == 0] = 0
        return fused, fused_cls

    def _warp(
        self,
        last_ego: tuple[float, float, float],
        cur_ego: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = self.evidence > 1e-4
        if not np.any(mask):
            z, y, x = self.evidence.shape
            return (
                np.zeros((z, y, x), dtype=np.float32),
                np.zeros((z, y, x), dtype=np.uint8),
            )
        zi, yi, xi = np.nonzero(mask)
        xg, yg, zg = esdf_axis_grids(self.grid)
        pts = np.column_stack([xg[xi], yg[yi], zg[zi]])
        pts_cur = transform_points_between_ego(pts, last_ego, cur_ego)
        z2, y2, x2, valid = points_to_voxels(pts_cur, self.grid)
        out_e = np.zeros_like(self.evidence)
        out_c = np.zeros_like(self.voxel_class)
        if not np.any(valid):
            return out_e, out_c
        z2, y2, x2 = z2[valid], y2[valid], x2[valid]
        src_e = self.evidence[zi[valid], yi[valid], xi[valid]]
        src_c = self.voxel_class[zi[valid], yi[valid], xi[valid]]
        np.maximum.at(out_e, (z2, y2, x2), src_e)
        out_c[z2, y2, x2] = src_c
        return out_e, out_c
