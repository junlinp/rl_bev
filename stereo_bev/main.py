"""Main pipeline: stereo RGB → model → BEV seg + occ."""

import carla
import cv2
import numpy as np
import time
import math

from .camera_rig import CameraRig
from .depth import decode_carla_depth
from .segmentation import remap_segmentation
from .bev_grid import BEVGrid, occupancy_to_bev, DEFAULT_X_RANGE, DEFAULT_Y_RANGE, DEFAULT_Z_RANGE, DEFAULT_VOXEL
from .query_heads import GeometricOccHead, HAS_TORCH
from .visualize import draw_bev_map, draw_legend, draw_depth_heatmap, draw_occ_3d_projections
from .calibration import DEFAULT_PITCH_DEG
from .occ_fusion import TemporalOccFusion


def run(
    host: str = "localhost",
    port: int = 2000,
    duration_sec: float = 60.0,
    image_w: int = 960,
    image_h: int = 540,
    fov: float = 90.0,
    baseline: float = 0.12,
    fps: float = 15.0,
    bev_x_range: tuple[float, float] = DEFAULT_X_RANGE,
    bev_y_range: tuple[float, float] = DEFAULT_Y_RANGE,
    bev_z_range: tuple[float, float] = DEFAULT_Z_RANGE,
    bev_voxel: float = DEFAULT_VOXEL,
    max_depth: float = 80.0,
    pitch_deg: float = DEFAULT_PITCH_DEG,
    mode: str = "geometric",     # "geometric" or "model"
    model_checkpoint: str | None = None,
    use_occ_fusion: bool = True,
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
    rig = CameraRig(vehicle, world, image_w, image_h, fov, fps, baseline, pitch_deg=pitch_deg)

    from .segmentation import NUM_BEV_CLASSES
    bev = BEVGrid(
        x_range=bev_x_range,
        y_range=bev_y_range,
        z_range=bev_z_range,
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
            bev_x_range=bev_x_range,
            bev_y_range=bev_y_range,
            bev_z_range=bev_z_range,
            bev_voxel=bev_voxel,
            max_depth=max_depth,
            pretrained_backbone=False,
            cam_extrinsic=rig.cam_extrinsic,
            pitch_deg=pitch_deg,
        )
        model.load_state_dict(torch.load(model_checkpoint, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"[BEV] Loaded StereoBEVModel on {device}")
    else:
        occ_head_geo = GeometricOccHead(min_hits=2.0)

    occ_fusion = None
    if use_occ_fusion:
        occ_thresh = 2.0 if mode != "model" else 0.25
        occ_fusion = TemporalOccFusion(bev, occ_thresh=occ_thresh)

    cam_extrinsic = rig.cam_extrinsic

    print(f"[BEV] Mode: {mode}")
    print(f"[BEV] Occupancy: {bev.grid_z}x{bev.grid_h}x{bev.grid_w} voxels, voxel={bev_voxel}m, pitch={pitch_deg}°")
    print(f"[BEV] Camera: {image_w}x{image_h}, fov={fov}°, baseline={baseline}m")
    if occ_fusion is not None:
        print(
            f"[BEV] temporal occupancy fusion  decay={occ_fusion.decay:.2f}  "
            f"thresh={occ_fusion.occ_thresh:.2f}"
        )
    else:
        print("[BEV] temporal occupancy fusion  off")
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
            tf = vehicle.get_transform()
            ego_xy_yaw = (
                float(tf.location.x), float(tf.location.y),
                math.radians(tf.rotation.yaw),
            )

            # ── BEV prediction ──
            voxel_class = None
            if model is not None:
                # model mode: stereo RGB → BEV
                _, occ_map, bev_classes, _, _ = model.infer(
                    left_rgb, right_rgb, rig.K, device=device, cam_ext=cam_extrinsic,
                )
                # decode depth for visualization only
                depth = decode_carla_depth(data["depth_raw"])
                occ_map = np.asarray(occ_map).astype(np.uint8)
                if occ_fusion is not None:
                    occ_map, voxel_class = occ_fusion.update(
                        occ_map, voxel_class, ego_xy_yaw,
                    )
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
                bev_classes = bev.get_bev_semantic(bev_result["class_histogram"])
                if occ_fusion is not None:
                    occ_map, voxel_class = occ_fusion.update(
                        bev_result["occupancy_count"],
                        bev_result["voxel_class"],
                        ego_xy_yaw,
                    )
                else:
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

            occ_panel = draw_occ_3d_projections(
                occ_map, scale=3,
                x_range=bev.x_range, y_range=bev.y_range, z_range=bev.z_range,
            )

            canvas = np.concatenate([top_row, bev_img], axis=0)
            # match occupancy panel width
            if occ_panel.shape[1] < canvas.shape[1]:
                pad = np.zeros((occ_panel.shape[0], canvas.shape[1] - occ_panel.shape[1], 3), dtype=np.uint8)
                occ_panel = np.concatenate([occ_panel, pad], axis=1)
            elif occ_panel.shape[1] > canvas.shape[1]:
                occ_panel = occ_panel[:, :canvas.shape[1]]
            canvas = np.concatenate([canvas, occ_panel], axis=0)

            cv2.imshow("Stereo BEV Perception", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if i % 30 == 0:
                occ_bev = occupancy_to_bev(occ_map)
                z_hit = int((occ_map.reshape(occ_map.shape[0], -1).sum(1) > 0).sum()) if occ_map.ndim == 3 else 1
                print(f"  frame {i}/{num_frames}  "
                      f"depth pts: {(depth > 0.1).sum():,}  "
                      f"occ voxels: {int(occ_map.sum())}/{occ_map.size}  "
                      f"z_bins={z_hit}  bev cells: {occ_bev.sum()}")

    finally:
        rig.destroy()
        vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)
        cv2.destroyAllWindows()
        print("[BEV] Done.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Stereo BEV perception in CARLA")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--mode", default="geometric", choices=["geometric", "model"])
    p.add_argument("--model-checkpoint", default=None)
    p.add_argument(
        "--no-occ-fusion", action="store_true",
        help="Disable multi-frame occupancy merge (single-frame lift only)",
    )
    args = p.parse_args()
    run(
        host=args.host, port=args.port, duration_sec=args.duration,
        mode=args.mode, model_checkpoint=args.model_checkpoint,
        use_occ_fusion=not args.no_occ_fusion,
    )
