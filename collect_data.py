"""
Collect stereo RGB + BEV ground truth from CARLA with nearby obstacles.

Spawns NPC traffic near the ego vehicle to ensure objects appear
within the 10m×10m×5m BEV grid.

Samples are stored only in minikeyvalue (default http://localhost:3000).

Usage:
  python collect_data.py --samples 500 --val-ratio 0.2
"""

import os
import io
import argparse
import random
import numpy as np
import carla
import time
import sys

sys.path.insert(0, os.path.dirname(__file__))

from stereo_bev.camera_rig import CameraRig
from stereo_bev.depth import decode_carla_depth
from stereo_bev.segmentation import remap_segmentation, NUM_BEV_CLASSES, BEV_CLASSES
from stereo_bev.bev_grid import (
    BEVGrid, occupancy_to_bev,
    DEFAULT_X_RANGE, DEFAULT_Y_RANGE, DEFAULT_Z_RANGE, DEFAULT_VOXEL,
)
from stereo_bev.query_heads import GeometricSegHead, GeometricOccHead
from stereo_bev.calibration import DEFAULT_PITCH_DEG
from minikeyvalue_client import MiniKV

DEFAULT_KV_URL = "http://localhost:3000"


def spawn_npc_traffic(world, tm, ego_vehicle, num_vehicles=15, num_walkers=10):
    """Spawn NPC vehicles and walkers near the ego vehicle."""
    bp_lib = world.get_blueprint_library()
    ego_loc = ego_vehicle.get_location()
    spawned = []

    # ── NPC vehicles ──
    vehicle_bps = bp_lib.filter("vehicle.*")
    vehicle_bps = [b for b in vehicle_bps if int(b.get_attribute("number_of_wheels")) == 4]
    spawn_points = world.get_map().get_spawn_points()

    # pick spawn points within 30m of ego
    nearby_points = []
    for sp in spawn_points:
        dist = sp.location.distance(ego_loc)
        if 3.0 < dist < 30.0:
            nearby_points.append(sp)

    random.shuffle(nearby_points)

    count = 0
    for sp in nearby_points[:num_vehicles]:
        bp = random.choice(vehicle_bps)
        try:
            npc = world.spawn_actor(bp, sp)
            npc.set_autopilot(True, tm.get_port())
            spawned.append(npc)
            count += 1
        except RuntimeError:
            continue

    print(f"  Spawned {count} NPC vehicles", flush=True)

    # ── NPC walkers ──
    walker_bp = random.choice(bp_lib.filter("walker.pedestrian.*"))
    walker_controller_bp = bp_lib.find("controller.ai.walker")

    walker_count = 0
    for _ in range(num_walkers):
        # random spawn near ego
        loc = carla.Location(
            x=ego_loc.x + random.uniform(-15, 15),
            y=ego_loc.y + random.uniform(-15, 15),
            z=ego_loc.z,
        )
        # snap to sidewalk
        wp = world.get_map().get_waypoint(loc, project_to_road=False)
        if wp is None:
            wp = world.get_map().get_waypoint(loc, project_to_road=True)
        if wp:
            spawn_loc = wp.transform.location
            spawn_loc.z += 1.0  # lift above ground
            spawn_tf = carla.Transform(spawn_loc)
            try:
                walker = world.spawn_actor(walker_bp, spawn_tf)
                controller = world.spawn_actor(
                    walker_controller_bp, carla.Transform(), attach_to=walker,
                )
                controller.start()
                controller.go_to_location(world.get_random_location_from_navigation())
                controller.set_max_speed(1.0 + random.random())
                spawned.append(walker)
                spawned.append(controller)
                walker_count += 1
            except RuntimeError:
                continue

    print(f"  Spawned {walker_count} walkers", flush=True)
    return spawned


def collect(
    host: str = "localhost",
    port: int = 2000,
    num_samples: int = 500,
    val_ratio: float = 0.2,
    image_w: int = 960,
    image_h: int = 540,
    fov: float = 90.0,
    baseline: float = 0.12,
    fps: float = 10.0,
    bev_x_range: tuple[float, float] = DEFAULT_X_RANGE,
    bev_y_range: tuple[float, float] = DEFAULT_Y_RANGE,
    bev_z_range: tuple[float, float] = DEFAULT_Z_RANGE,
    bev_voxel: float = DEFAULT_VOXEL,
    max_depth: float = 80.0,
    pitch_deg: float = DEFAULT_PITCH_DEG,
    steps_per_sample: int = 3,
    num_vehicles: int = 15,
    num_walkers: int = 10,
    min_occupied: int = 50,  # min occupied cells to keep a sample
    min_classes: int = 2,    # min distinct BEV classes to keep a sample
    kv_url: str = DEFAULT_KV_URL,
):
    if not kv_url:
        raise ValueError("kv_url is required; collect_data writes only to minikeyvalue")
    kv = MiniKV(kv_url)

    # resume from existing counts in minikeyvalue
    existing_train = kv.count("/train/")
    existing_val = kv.count("/val/")
    train_count = existing_train
    val_count = existing_val
    print(f"[Collect] minikeyvalue: {kv_url}")
    print(f"[Collect] Existing: {existing_train} train + {existing_val} val")

    num_val = int(num_samples * val_ratio)
    num_train = num_samples - num_val

    print(f"[Collect] Target: {num_train} train + {num_val} val = {num_samples} total")
    print(f"[Collect] Filtering: min_occupied={min_occupied}, min_classes={min_classes}")

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
    print(f"[Collect] Ego at ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")

    # ── spawn nearby NPC traffic ──
    print("[Collect] Spawning NPC traffic...", flush=True)
    npcs = spawn_npc_traffic(world, tm, vehicle, num_vehicles, num_walkers)

    # ── rig + BEV ──
    rig = CameraRig(vehicle, world, image_w, image_h, fov, fps, baseline, pitch_deg=pitch_deg)
    bev = BEVGrid(
        x_range=bev_x_range,
        y_range=bev_y_range,
        z_range=bev_z_range,
        voxel_size=bev_voxel,
        num_classes=NUM_BEV_CLASSES,
    )
    seg_head = GeometricSegHead()
    occ_head = GeometricOccHead(min_hits=2.0)
    cam_ext = rig.cam_extrinsic
    print(f"[Collect] Occupancy volume: {bev.grid_z}x{bev.grid_h}x{bev.grid_w}  "
          f"voxel={bev_voxel}m  pitch={pitch_deg}°")

    # warm up
    print("[Collect] Warming up...", flush=True)
    for _ in range(10):
        world.tick()
        time.sleep(0.1)

    # ── collect loop ──
    collected = 0
    skipped = 0
    class_stats = np.zeros(NUM_BEV_CLASSES, dtype=np.int64)

    print(f"[Collect] Collecting...", flush=True)
    try:
        while collected < num_samples:
            for _ in range(steps_per_sample):
                world.tick()
                time.sleep(0.01)

            data = rig.grab()
            if data is None:
                continue

            depth = decode_carla_depth(data["depth_raw"])
            seg_image = remap_segmentation(data["seg_raw"])  # image-space segmentation (H, W)
            result = bev.bev_from_frame(depth, seg_image, rig.K, cam_ext, max_depth=max_depth)

            seg_gt = seg_image   # IMAGE-SPACE segmentation (camera view)
            occ_gt = occ_head(result["occupancy_count"])  # (Z, H, W) 3D occupancy
            bev_seg_gt = bev.get_bev_semantic(result["class_histogram"])
            occ_bev = occupancy_to_bev(occ_gt)

            n_occupied = int(occ_gt.sum())
            n_classes = len(np.unique(seg_gt))

            if n_occupied < min_occupied or n_classes < min_classes:
                skipped += 1
                continue

            # track stats (image-space)
            for c in range(NUM_BEV_CLASSES):
                class_stats[c] += (seg_gt == c).sum()

            if collected < num_train:
                key = f"/train/sample_{train_count:06d}"
                train_count += 1
            else:
                key = f"/val/sample_{val_count:06d}"
                val_count += 1
            buf = io.BytesIO()
            np.savez_compressed(
                buf,
                left_rgb=data["left_rgb"],
                right_rgb=data["right_rgb"],
                depth_gt=depth,
                seg_gt=seg_gt,
                occ_gt=occ_gt,
                bev_seg_gt=bev_seg_gt,
                K=rig.K,
                cam_ext=cam_ext,
                x_range=np.array(bev_x_range, dtype=np.float32),
                y_range=np.array(bev_y_range, dtype=np.float32),
                z_range=np.array(bev_z_range, dtype=np.float32),
                voxel_size=np.float32(bev_voxel),
            )
            if not kv.put(key, buf.getvalue()):
                raise OSError(f"PUT {key} failed on {kv_url}")

            collected += 1
            split = "train" if collected <= num_train else "val"
            if collected % 25 == 0 or collected == num_samples:
                loc = vehicle.get_location()
                occupied_classes = np.unique(bev_seg_gt[occ_bev > 0]) if n_occupied else []
                class_names = [BEV_CLASSES.get(int(c), str(c)) for c in occupied_classes]
                z_hit = int((occ_gt.reshape(occ_gt.shape[0], -1).sum(axis=1) > 0).sum())
                print(
                    f"  [{split}] {collected}/{num_samples}  "
                    f"pos=({loc.x:.0f},{loc.y:.0f})  "
                    f"occ={n_occupied}/{occ_gt.size}  z_bins={z_hit}/{occ_gt.shape[0]}  "
                    f"classes={class_names}  "
                    f"skipped={skipped}",
                    flush=True,
                )

    finally:
        for actor in reversed(npcs):
            try:
                if actor.is_alive:
                    stop = getattr(actor, "stop", None)
                    if callable(stop):
                        stop()
                    actor.destroy()
            except Exception:
                pass
        try:
            rig.destroy()
        except Exception:
            pass
        try:
            if vehicle is not None and vehicle.is_alive:
                vehicle.destroy()
        except Exception:
            pass
        try:
            settings.synchronous_mode = False
            world.apply_settings(settings)
            tm.set_synchronous_mode(False)
        except Exception:
            pass

    # summary
    print(f"\n[Collect] Done. {collected} saved, {skipped} skipped.")
    print(f"  Train: {kv.count('/train/')} → {kv_url}/train/")
    print(f"  Val:   {kv.count('/val/')} → {kv_url}/val/")
    print(f"\n  BEV class distribution:")
    total_px = class_stats.sum()
    for c in range(NUM_BEV_CLASSES):
        if class_stats[c] > 0:
            pct = 100.0 * class_stats[c] / max(total_px, 1)
            print(f"    {c:2d} {BEV_CLASSES[c]:12s}: {class_stats[c]:>12,}  ({pct:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--steps-per-sample", type=int, default=3)
    parser.add_argument("--num-vehicles", type=int, default=15)
    parser.add_argument("--num-walkers", type=int, default=10)
    parser.add_argument("--min-occupied", type=int, default=50)
    parser.add_argument("--min-classes", type=int, default=2)
    parser.add_argument("--kv-url", default=DEFAULT_KV_URL, help="minikeyvalue URL")
    args = parser.parse_args()

    collect(
        host=args.host, port=args.port,
        num_samples=args.samples,
        val_ratio=args.val_ratio, steps_per_sample=args.steps_per_sample,
        num_vehicles=args.num_vehicles, num_walkers=args.num_walkers,
        min_occupied=args.min_occupied, min_classes=args.min_classes,
        kv_url=args.kv_url,
    )
