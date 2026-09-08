"""Stereo camera rig: left/right RGB + depth + segmentation, co-located on left."""

import threading
import numpy as np
import carla

from .calibration import intrinsics_from_carla


class CameraRig:
    """
    Stereo camera rig on a vehicle.

    Left camera:   RGB + depth + segmentation (all co-located)
    Right camera:  RGB only (offset by baseline on Y-axis)

    Depth sensor provides ground-truth depth for the left camera view.
    """

    def __init__(
        self,
        vehicle: carla.Vehicle,
        world: carla.World,
        image_w: int = 960,
        image_h: int = 540,
        fov: float = 90.0,
        fps: float = 15.0,
        baseline: float = 0.12,
        mount_offset: tuple[float, float, float] = (1.5, 0.0, 1.6),
    ):
        self.w = image_w
        self.h = image_h
        self.baseline = baseline
        self.lock = threading.Lock()
        self._frames: dict[str, tuple[int, np.ndarray]] = {}

        # intrinsics (same for both cameras)
        self.K, self.focal_px = intrinsics_from_carla(image_w, image_h, fov)

        # mount transforms
        x0, y0, z0 = mount_offset
        half_b = baseline / 2.0

        left_mount = carla.Transform(carla.Location(x=x0, y=y0 - half_b, z=z0))
        right_mount = carla.Transform(carla.Location(x=x0, y=y0 + half_b, z=z0))

        # blueprints — all sensors share same resolution/fov
        bp_lib = world.get_blueprint_library()

        def _make_bp(sensor_type: str):
            bp = bp_lib.find(sensor_type)
            bp.set_attribute("image_size_x", str(image_w))
            bp.set_attribute("image_size_y", str(image_h))
            bp.set_attribute("fov", str(fov))
            bp.set_attribute("sensor_tick", str(1.0 / fps))
            return bp

        rgb_bp = _make_bp("sensor.camera.rgb")
        depth_bp = _make_bp("sensor.camera.depth")
        seg_bp = _make_bp("sensor.camera.semantic_segmentation")

        # spawn actors
        self.left_rgb = world.spawn_actor(
            rgb_bp, left_mount, attach_to=vehicle,
            attachment_type=carla.AttachmentType.Rigid,
        )
        self.right_rgb = world.spawn_actor(
            rgb_bp, right_mount, attach_to=vehicle,
            attachment_type=carla.AttachmentType.Rigid,
        )
        self.depth_cam = world.spawn_actor(
            depth_bp, left_mount, attach_to=vehicle,
            attachment_type=carla.AttachmentType.Rigid,
        )
        self.seg_cam = world.spawn_actor(
            seg_bp, left_mount, attach_to=vehicle,
            attachment_type=carla.AttachmentType.Rigid,
        )

        # listeners
        self.left_rgb.listen(self._on_left_rgb)
        self.right_rgb.listen(self._on_right_rgb)
        self.depth_cam.listen(self._on_depth)
        self.seg_cam.listen(self._on_seg)

    def _on_left_rgb(self, image: carla.Image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(self.h, self.w, 4)[:, :, :3].copy()
        with self.lock:
            self._frames["left_rgb"] = (image.frame, arr)

    def _on_right_rgb(self, image: carla.Image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(self.h, self.w, 4)[:, :, :3].copy()
        with self.lock:
            self._frames["right_rgb"] = (image.frame, arr)

    def _on_depth(self, image: carla.Image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(self.h, self.w, 4)[:, :, :3].copy()
        with self.lock:
            self._frames["depth"] = (image.frame, arr)

    def _on_seg(self, image: carla.Image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(self.h, self.w, 4)[:, :, 2].copy()
        with self.lock:
            self._frames["seg"] = (image.frame, arr)

    def grab(self) -> dict | None:
        """
        Return latest synchronized frame or None.

        Keys:
            left_rgb:   (H, W, 3) uint8 BGR
            right_rgb:  (H, W, 3) uint8 BGR
            depth_raw:  (H, W, 3) uint8 (CARLA depth encoding)
            seg_raw:    (H, W)    uint8 (tag indices)
            frame:      int frame number
        """
        with self.lock:
            if len(self._frames) < 4:
                return None

            entry_l = self._frames.get("left_rgb")
            entry_r = self._frames.get("right_rgb")
            entry_d = self._frames.get("depth")
            entry_s = self._frames.get("seg")

            if entry_l is None or entry_r is None or entry_d is None or entry_s is None:
                return None

            f_l, left_rgb = entry_l
            _, right_rgb = entry_r
            _, depth_raw = entry_d
            _, seg_raw = entry_s

            # ±1 frame tolerance across all 4 sensors
            for key in ("right_rgb", "depth", "seg"):
                f_k, _ = self._frames[key]
                if abs(f_l - f_k) > 1:
                    return None

        return {
            "left_rgb": left_rgb,
            "right_rgb": right_rgb,
            "depth_raw": depth_raw,
            "seg_raw": seg_raw,
            "frame": f_l,
        }

    def destroy(self):
        for actor in (self.left_rgb, self.right_rgb, self.depth_cam, self.seg_cam):
            if actor is not None:
                actor.destroy()
