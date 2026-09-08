"""Main pipeline: stereo RGB → model → BEV seg + occ."""

import carla
import cv2
import numpy as np
import time

from .camera_rig import CameraRig
from .depth import decode_carla_depth
from .segmentation import remap_segmentation
from .bev_grid import BEVGrid
from .query_heads import GeometricSegHead, GeometricOccHead, HAS_TORCH
from .visualize import draw_bev_map, draw_legend, draw_depth_heatmap


def run(
    host: str = "localhost",
    port: int = 2000,
    duration_sec: float = 60.0,
    image_w: int = 960,
    image_h: int = 540,
    fov: float = 90.0,
    baseline: float = 0.12,
    fps: float = 15.0,
    bev_range_xy: float = 5.0,
    bev_z_range: float = 5.0,
    bev_voxel: float = 0.1,
    max_depth: float = 80.0,
    mode: str = "geometric",     # "geometric" or "model"
    model_checkpoint: str | None = None,
):
    """
    Full BEV perception loop.

    Two modes:
      - "geometric": uses CARLA depth + seg sensors to build BEV (ground truth)
      - "model":     uses trained StereoBEVModel on stereo RGB only

    Args:
        mode: "geometric" or "model"
    """
    # ── CARLA setup ──
    client = carla.Client(host, port)
    client.set_timeout(15.0)
    world = client.get_world()
    tm = client.get_trafficmanager()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / fps
    world.apply_settings(settings)
    tm.set_synchronous_mode(True)

    # ── spawn ego vehicle ──
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True, tm.get_port())

    # ── camera rig ──
    rig = CameraRig(vehicle, world, image_w, image_h, fov, fps, baseline)

    # ── BEV grid (used for geometric mode + GT generation) ──
    from .segmentation import NUM_BEV_CLASSES
    bev = BEVGrid(
        x_range=(-bev_range_xy, bev_range_xy),
        y_range=(-bev_range_xy, bev_range_xy),
        z_range=(0.0, bev_z_range),
        voxel_size=bev_voxel,
        num_classes=NUM_BEV_CLASSES,
    )

    # ── load model or geometric heads ──
    model = None
    device = "cpu"

    if mode == "model":
        assert HAS_TORCH, "PyTorch required for model mode"
        assert model_checkpoint, "model_checkpoint required for model mode"
        import torch
        from .query_heads import StereoBEVModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = StereoBEVModel(
            num_classes=NUM_BEV_CLASSES,
            image_h=image_h,
            image_w=image_w,
            bev_x_range=(-bev_range_xy, bev_range_xy),
            bev_y_range=(-bev_range_xy, bev_range_xy),
            bev_z_range=(0.0, bev_z_range),
            bev_voxel=bev_voxel,
            max_depth=max_depth,
            pretrained_backbone=False,
        )
        model.load_state_dict(torch.load(model_checkpoint, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"[BEV] Loaded StereoBEVModel on {device}")
    else:
        seg_head_geo = GeometricSegHead()
        occ_head_geo = GeometricOccHead(min_hits=2.0)

    # Camera frame: X=right, Y=down, Z=forward
    # Ego frame:    X=forward, Y=left, Z=up
    # Rotation: ego_x=cam_z, ego_y=-cam_x, ego_z=-cam_y
    # Translation: camera mount at (1.5, 0, 1.6) in ego frame
    cam_extrinsic = np.array([
        [ 0,  0,  1,  1.5],
        [-1,  0,  0,  0.0],
        [ 0, -1,  0,  1.6],
        [ 0,  0,  0,  1.0],
    ], dtype=np.float64)

    print(f"[BEV] Mode: {mode}")
    print(f"[BEV] Grid: {bev.grid_w}x{bev.grid_h}x{bev.grid_z} voxels, voxel={bev_voxel}m")
    print(f"[BEV] Camera: {image_w}x{image_h}, fov={fov}°, baseline={baseline}m")
    print(f"[BEV] Press 'q' to quit")

    # ── main loop ──
    num_frames = int(duration_sec * fps)
    try:
        for i in range(num_frames):
            world.tick()
            time.sleep(0.01)

            data = rig.grab()
            if data is None:
                continue

            left_rgb = data["left_rgb"]
            right_rgb = data["right_rgb"]

            # ── BEV prediction ──
            if model is not None:
                # model mode: stereo RGB → BEV
                bev_classes, occ_map = model.infer(
                    left_rgb, right_rgb, rig.K, device=device,
                )
                # decode depth for visualization only
                depth = decode_carla_depth(data["depth_raw"])
            else:
                # geometric mode: depth + seg sensors → BEV
                depth = decode_carla_depth(data["depth_raw"])
                seg_bev = remap_segmentation(data["seg_raw"])

                bev_result = bev.bev_from_frame(
                    depth_map=depth,
                    seg_map=seg_bev,
                    K=rig.K,
                    cam_extrinsic=cam_extrinsic,
                    max_depth=max_depth,
                )
                bev_classes = seg_head_geo(bev_result["class_histogram"])
                occ_map = occ_head_geo(bev_result["occupancy_count"])

            # ── visualization ──
            bev_img = draw_bev_map(bev_classes, occ_map, scale=4)
            bev_img = draw_legend(bev_img, top_left=(bev_img.shape[1] - 130, 10))

            depth_vis = draw_depth_heatmap(depth, max_depth=max_depth)

            bev_h = bev_img.shape[0]
            cam_w = int(bev_h * image_w / image_h)
            left_small = cv2.resize(left_rgb, (cam_w, bev_h))
            depth_small = cv2.resize(depth_vis, (cam_w, bev_h))

            cam_total = left_small.shape[1] + depth_small.shape[1]
            bev_w = bev_img.shape[1]
            if cam_total < bev_w:
                pad = np.zeros((bev_h, bev_w - cam_total, 3), dtype=np.uint8)
                top_row = np.concatenate([left_small, depth_small, pad], axis=1)
            elif cam_total > bev_w:
                crop_w = bev_w // 2
                left_small = cv2.resize(left_small, (crop_w, bev_h))
                depth_small = cv2.resize(depth_small, (bev_w - crop_w, bev_h))
                top_row = np.concatenate([left_small, depth_small], axis=1)
            else:
                top_row = np.concatenate([left_small, depth_small], axis=1)

            canvas = np.concatenate([top_row, bev_img], axis=0)

            cv2.imshow("Stereo BEV Perception", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if i % 30 == 0:
                print(f"  frame {i}/{num_frames}  "
                      f"depth pts: {(depth > 0.1).sum():,}  "
                      f"BEV occupied: {occ_map.sum()}/{occ_map.size}")

    finally:
        rig.destroy()
        vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)
        cv2.destroyAllWindows()
        print("[BEV] Done.")
