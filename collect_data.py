"""
Collect stereo RGB + BEV ground truth from CARLA.

Saves train/ and val/ splits with:
  left_rgb, right_rgb, seg_gt, occ_gt, K

Usage:
  python collect_data.py --samples 500 --val-ratio 0.2
"""

import os
import argparse
import numpy as np
import carla
import time
import sys

sys.path.insert(0, os.path.dirname(__file__))

from stereo_bev.camera_rig import CameraRig
from stereo_bev.depth import decode_carla_depth
from stereo_bev.segmentation import remap_segmentation, NUM_BEV_CLASSES, BEV_CLASSES
from stereo_bev.bev_grid import BEVGrid
from stereo_bev.query_heads import GeometricSegHead, GeometricOccHead


def collect(
    host: str = "localhost",
    port: int = 2000,
    num_samples: int = 500,
    output_dir: str = "bev_data",
    val_ratio: float = 0.2,
    image_w: int = 960,
    image_h: int = 540,
    fov: float = 90.0,
    baseline: float = 0.12,
    fps: float = 10.0,
    bev_range_xy: float = 5.0,
    bev_z_range: float = 5.0,
    bev_voxel: float = 0.1,
    max_depth: float = 80.0,
    steps_per_sample: int = 3,  # ticks between samples (variety)
):
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    num_val = int(num_samples * val_ratio)
    num_train = num_samples - num_val

    print(f"[Collect] Target: {num_train} train + {num_val} val = {num_samples} total")
    print(f"[Collect] Output: {output_dir}/")

    # connect
    client = carla.Client(host, port)
    client.set_timeout(15.0)
    world = client.get_world()
    tm = client.get_trafficmanager()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / fps
    world.apply_settings(settings)
    tm.set_synchronous_mode(True)

    # spawn vehicle (try multiple points)
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    spawn_points = world.get_map().get_spawn_points()

    vehicle = None
    for sp in spawn_points:
        try:
            vehicle = world.spawn_actor(vehicle_bp, sp)
            break
        except RuntimeError:
            continue
    if vehicle is None:
        print("[ERROR] Could not spawn vehicle")
        return

    vehicle.set_autopilot(True, tm.get_port())
    loc = vehicle.get_location()
    print(f"[Collect] Spawned at ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")

    # rig + BEV
    rig = CameraRig(vehicle, world, image_w, image_h, fov, fps, baseline)
    bev = BEVGrid(
        x_range=(-bev_range_xy, bev_range_xy),
        y_range=(-bev_range_xy, bev_range_xy),
        z_range=(0.0, bev_z_range),
        voxel_size=bev_voxel,
        num_classes=NUM_BEV_CLASSES,
    )
    seg_head = GeometricSegHead()
    occ_head = GeometricOccHead(min_hits=2.0)
    cam_ext = np.array([
        [ 0,  0,  1,  1.5],
        [-1,  0,  0,  0.0],
        [ 0, -1,  0,  1.6],
        [ 0,  0,  0,  1.0],
    ], dtype=np.float64)

    # warm up sensors
    print("[Collect] Warming up sensors...", flush=True)
    for _ in range(10):
        world.tick()
        time.sleep(0.1)

    # collect
    collected = 0
    empty_streak = 0
    class_stats = np.zeros(NUM_BEV_CLASSES, dtype=np.int64)

    print(f"[Collect] Collecting...", flush=True)
    try:
        while collected < num_samples:
            # advance simulation for variety
            for _ in range(steps_per_sample):
                world.tick()
                time.sleep(0.01)

            data = rig.grab()
            if data is None:
                empty_streak += 1
                if empty_streak > 50:
                    print(f"  [WARN] 50 consecutive no-data frames, skipping...", flush=True)
                    empty_streak = 0
                continue
            empty_streak = 0

            depth = decode_carla_depth(data["depth_raw"])
            seg_bev = remap_segmentation(data["seg_raw"])
            result = bev.bev_from_frame(depth, seg_bev, rig.K, cam_ext, max_depth=max_depth)

            if result["occupancy_count"].sum() < 10:
                continue

            seg_gt = seg_head(result["class_histogram"])
            occ_gt = occ_head(result["occupancy_count"])

            # track class distribution
            for c in range(NUM_BEV_CLASSES):
                class_stats[c] += (seg_gt == c).sum()

            # save
            if collected < num_train:
                out_path = os.path.join(train_dir, f"sample_{collected:06d}.npz")
            else:
                val_idx = collected - num_train
                out_path = os.path.join(val_dir, f"sample_{val_idx:06d}.npz")

            np.savez_compressed(
                out_path,
                left_rgb=data["left_rgb"],
                right_rgb=data["right_rgb"],
                depth_gt=depth,
                seg_gt=seg_gt,
                occ_gt=occ_gt,
                K=rig.K,
            )

            collected += 1
            split = "train" if collected <= num_train else "val"
            if collected % 25 == 0 or collected == num_samples:
                loc = vehicle.get_location()
                print(
                    f"  [{split}] {collected}/{num_samples}  "
                    f"pos=({loc.x:.0f},{loc.y:.0f})  "
                    f"occ={occ_gt.sum()}/{occ_gt.size}  "
                    f"depth_pts={int((depth>0.1).sum()):,}",
                    flush=True,
                )

    finally:
        rig.destroy()
        vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)

    # summary
    print(f"\n[Collect] Done. {collected} samples saved.")
    print(f"  Train: {min(collected, num_train)} → {train_dir}/")
    print(f"  Val:   {max(0, collected - num_train)} → {val_dir}/")
    print(f"\n  BEV class distribution (pixel counts):")
    total_px = class_stats.sum()
    for c in range(NUM_BEV_CLASSES):
        pct = 100.0 * class_stats[c] / max(total_px, 1)
        print(f"    {c:2d} {BEV_CLASSES[c]:12s}: {class_stats[c]:>12,}  ({pct:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--output", default="bev_data")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--steps-per-sample", type=int, default=3)
    args = parser.parse_args()

    collect(
        host=args.host,
        port=args.port,
        num_samples=args.samples,
        output_dir=args.output,
        val_ratio=args.val_ratio,
        steps_per_sample=args.steps_per_sample,
    )
