"""BEV grid: lift depth + segmentation into a 3D voxel occupancy volume."""

import numpy as np

# Local 3D occupancy volume (ego frame: X forward, Y left, Z up)
DEFAULT_X_RANGE = (0.0, 20.0)
DEFAULT_Y_RANGE = (-10.0, 10.0)
DEFAULT_Z_RANGE = (-1.0, 3.0)
DEFAULT_VOXEL = 0.2


def occupancy_to_bev(occ: np.ndarray) -> np.ndarray:
    """(Z, H, W) or (H, W) occupancy → (H, W) binary bird's-eye mask."""
    if occ.ndim == 3:
        return (occ.max(axis=0) > 0).astype(np.uint8)
    return (occ > 0).astype(np.uint8)


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
        x_range: tuple[float, float] = DEFAULT_X_RANGE,
        y_range: tuple[float, float] = DEFAULT_Y_RANGE,
        z_range: tuple[float, float] = DEFAULT_Z_RANGE,
        voxel_size: float = DEFAULT_VOXEL,
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
            class_histogram: (num_classes, grid_h, grid_w) Z-sum, for the 2D BEV map
            voxel_class:     (grid_z, grid_h, grid_w) per-voxel class (no Z collapse)
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

        cls_3d = np.zeros(
            (self.num_classes, self.grid_z, self.grid_h, self.grid_w), dtype=np.float32,
        )
        if class_labels is not None:
            for c in range(self.num_classes):
                mask = class_labels == c
                if mask.any():
                    np.add.at(cls_3d[c], (zi[mask], yi[mask], xi[mask]), weights[mask])
        cls_hist = cls_3d.sum(axis=1)
        voxel_class = cls_3d.argmax(axis=0).astype(np.uint8)
        voxel_class[cls_3d.sum(axis=0) <= 0] = 0

        return {
            "occupancy_count": occ,
            "class_histogram": cls_hist,
            "voxel_class": voxel_class,
        }

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

        points_cam, pixels = depth_to_pointcloud(depth_map, K, max_depth=max_depth)
        if points_cam.shape[0] == 0:
            return {
                "occupancy_count": np.zeros((self.grid_z, self.grid_h, self.grid_w), dtype=np.float32),
                "class_histogram": np.zeros((self.num_classes, self.grid_h, self.grid_w), dtype=np.float32),
                "voxel_class": np.zeros((self.grid_z, self.grid_h, self.grid_w), dtype=np.uint8),
            }

        points_ego = self.camera_to_ego(points_cam, cam_extrinsic)
        v = pixels[:, 1].astype(np.int32)
        u = pixels[:, 0].astype(np.int32)
        labels = seg_map[v, u]

        return self.voxelize(points_ego, class_labels=labels)

    # ── BEV map queries ──

    @staticmethod
    def get_occupancy_3d(occ_count: np.ndarray, threshold: float = 2.0) -> np.ndarray:
        """
        Threshold the 3D hit-count volume.

        Returns (grid_z, grid_h, grid_w) uint8 binary occupancy.
        """
        return (occ_count >= threshold).astype(np.uint8)

    @staticmethod
    def get_occupancy_map(occ_count: np.ndarray, threshold: float = 1.0) -> np.ndarray:
        """
        Collapse 3D occupancy → 2D BEV occupancy.

        Returns (grid_h, grid_w) binary: 1 if any voxel above threshold is hit.
        """
        return occupancy_to_bev((occ_count >= threshold).astype(np.uint8))

    @staticmethod
    def get_bev_semantic(class_histogram: np.ndarray) -> np.ndarray:
        """
        Collapse a 2D class histogram → (H, W) argmax class.

        This is only for the bird's-eye map. 3D occupancy uses per-voxel
        ``voxel_class``, not this XY collapse.
        """
        total = class_histogram.sum(axis=0)
        # argmax over class axis; cells with 0 hits stay 0 (empty)
        bev_class = class_histogram.argmax(axis=0)
        bev_class[total == 0] = 0
        return bev_class.astype(np.uint8)
