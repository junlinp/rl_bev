"""BEV grid: lift depth + segmentation into a voxel grid, then collapse to BEV."""

import numpy as np


class BEVGrid:
    """
    Axis-aligned BEV grid in the ego (vehicle) frame.

    Coordinate convention (right-hand, Z-up):
        X  → forward
        Y  → left
        Z  → up

    BEV cell (i, j) corresponds to:
        x = x_range[0] + (i + 0.5) * voxel_size
        y = y_range[0] + (j + 0.5) * voxel_size
    """

    def __init__(
        self,
        x_range: tuple[float, float] = (-50.0, 50.0),
        y_range: tuple[float, float] = (-50.0, 50.0),
        z_range: tuple[float, float] = (-3.0, 5.0),
        voxel_size: float = 0.5,
        num_classes: int = 10,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.voxel_size = voxel_size
        self.num_classes = num_classes

        self.grid_w = int((x_range[1] - x_range[0]) / voxel_size)
        self.grid_h = int((y_range[1] - y_range[0]) / voxel_size)
        self.grid_z = int((z_range[1] - z_range[0]) / voxel_size)

    # ── coordinate helpers ──

    def world_to_grid(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map world XY → grid cell indices (i, j)."""
        i = ((x - self.x_range[0]) / self.voxel_size).astype(np.int32)
        j = ((y - self.y_range[0]) / self.voxel_size).astype(np.int32)
        return i, j

    def grid_to_world(self, i: np.ndarray, j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = (i + 0.5) * self.voxel_size + self.x_range[0]
        y = (j + 0.5) * self.voxel_size + self.y_range[0]
        return x, y

    # ── projection: camera frame → ego frame → grid ──

    def camera_to_ego(
        self,
        points_cam: np.ndarray,
        cam_extrinsic: np.ndarray,
    ) -> np.ndarray:
        """
        Transform points from camera frame to ego frame.

        cam_extrinsic: (4, 4) ego-from-camera transform.
            Rotation + translation that maps camera-frame XYZ to ego-frame XYZ.
        """
        R = cam_extrinsic[:3, :3]
        t = cam_extrinsic[:3, 3]
        return (R @ points_cam.T).T + t

    # ── core lifting ──

    def voxelize(
        self,
        points_ego: np.ndarray,
        class_labels: np.ndarray | None = None,
        confidence: np.ndarray | None = None,
    ) -> dict:
        """
        Scatter points into the BEV voxel grid.

        Args:
            points_ego: (N, 3) in ego frame
            class_labels: (N,) int class indices, or None
            confidence: (N,) float weights, or None

        Returns dict with:
            occupancy_count: (grid_z, grid_h, grid_w) hit count
            class_histogram: (num_classes, grid_h, grid_w) per-class weighted count
        """
        N = points_ego.shape[0]
        xi, yi = self.world_to_grid(points_ego[:, 0], points_ego[:, 1])
        zi = ((points_ego[:, 2] - self.z_range[0]) / self.voxel_size).astype(np.int32)

        # mask out-of-bounds
        valid = (
            (xi >= 0) & (xi < self.grid_w) &
            (yi >= 0) & (yi < self.grid_h) &
            (zi >= 0) & (zi < self.grid_z)
        )
        xi, yi, zi = xi[valid], yi[valid], zi[valid]
        if class_labels is not None:
            class_labels = class_labels[valid]
        if confidence is not None:
            weights = confidence[valid]
        else:
            weights = np.ones(xi.shape[0], dtype=np.float32)

        # 3D occupancy histogram
        occ = np.zeros((self.grid_z, self.grid_h, self.grid_w), dtype=np.float32)
        np.add.at(occ, (zi, yi, xi), weights)

        # per-class histogram collapsed along Z
        cls_hist = np.zeros((self.num_classes, self.grid_h, self.grid_w), dtype=np.float32)
        if class_labels is not None:
            # flatten (z, y, x) → one index for add.at
            for c in range(self.num_classes):
                mask = class_labels == c
                if mask.any():
                    np.add.at(cls_hist[c], (yi[mask], xi[mask]), weights[mask])

        return {"occupancy_count": occ, "class_histogram": cls_hist}

    def bev_from_frame(
        self,
        depth_map: np.ndarray,
        seg_map: np.ndarray,
        K: np.ndarray,
        cam_extrinsic: np.ndarray,
        max_depth: float = 120.0,
    ) -> dict:
        """
        Single-frame BEV generation: depth + seg → point cloud → voxelize.

        Args:
            depth_map: (H, W) float depth in meters
            seg_map: (H, W) uint8 BEV class indices
            K: (3, 3) intrinsic matrix
            cam_extrinsic: (4, 4) world-from-camera or ego-from-camera
            max_depth: max depth to include

        Returns dict from voxelize()
        """
        from .depth import depth_to_pointcloud

        points_cam, _ = depth_to_pointcloud(depth_map, K, max_depth=max_depth)
        if points_cam.shape[0] == 0:
            return {
                "occupancy_count": np.zeros((self.grid_z, self.grid_h, self.grid_w), dtype=np.float32),
                "class_histogram": np.zeros((self.num_classes, self.grid_h, self.grid_w), dtype=np.float32),
            }

        points_ego = self.camera_to_ego(points_cam, cam_extrinsic)

        # extract class labels at valid pixels
        h, w = depth_map.shape
        valid_mask = (depth_map > 0) & (depth_map < max_depth)
        labels = seg_map[valid_mask]

        return self.voxelize(points_ego, class_labels=labels)

    # ── BEV map queries ──

    @staticmethod
    def get_occupancy_map(occ_count: np.ndarray, threshold: float = 1.0) -> np.ndarray:
        """
        Collapse 3D occupancy → 2D BEV occupancy.

        Returns (grid_h, grid_w) binary: 1 if any voxel above threshold is hit.
        """
        # sum along Z axis
        projected = occ_count.sum(axis=0)
        return (projected >= threshold).astype(np.uint8)

    @staticmethod
    def get_bev_semantic(class_histogram: np.ndarray) -> np.ndarray:
        """
        Collapse class histogram → (grid_h, grid_w) argmax class.

        Returns class index per cell. 0 (empty) for cells with no hits.
        """
        total = class_histogram.sum(axis=0)
        # argmax over class axis; cells with 0 hits stay 0 (empty)
        bev_class = class_histogram.argmax(axis=0)
        bev_class[total == 0] = 0
        return bev_class.astype(np.uint8)
