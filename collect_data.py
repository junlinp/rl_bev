"""
Collect stereo RGB + BEV ground truth + expert control from CARLA.

Saves stereo RGB, K, occupancy/seg GT, plus the 5 s target pose and
autopilot (throttle, brake, steer) for the vision-action control head.

Spawns NPC traffic near the ego vehicle to ensure objects appear
within the 10m×10m×5m BEV grid.

Samples are stored only in minikeyvalue (default http://localhost:3000).

Usage:
  python collect_data.py --samples 500 --val-ratio 0.2
  python collect_data.py --trajectories --samples 2000 --episode-len 200
"""

import os
import io
import argparse
import random
import math
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
from stereo_bev.occ_fusion import TemporalOccFusion
from stereo_bev.global_target import GlobalTarget, yaw_from_R
from minikeyvalue_client import MiniKV

DEFAULT_KV_URL = "http://localhost:3000"

def _speed(vehicle) -> float:
    v = vehicle.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def _expert_control(vehicle) -> np.ndarray:
    c = vehicle.get_control()
    return np.array([c.throttle, c.brake, c.steer], dtype=np.float32)


def _target_from_plan(plan, speed: float) -> np.ndarray:
    t = np.asarray(plan["target_t"], dtype=np.float64).reshape(3)
    R = np.asarray(plan["target_R"], dtype=np.float64).reshape(3, 3)
    return np.array([t[0], t[1], yaw_from_R(R), speed], dtype=np.float32)


def _cross_track(route_xy: np.ndarray, x: float = 0.0, y: float = 0.0) -> float:
    if route_xy is None or len(route_xy) < 2:
        return 0.0
    d = np.linalg.norm(route_xy - np.array([x, y]), axis=1)
    i = int(np.clip(np.argmin(d), 0, len(route_xy) - 2))
    a, b = route_xy[i], route_xy[i + 1]
    ab = b - a
    denom = np.dot(ab, ab) + 1e-9
    t = np.clip(np.dot(np.array([x, y]) - a, ab) / denom, 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(np.array([x, y]) - proj))


def _world_xy_yaw(vehicle) -> np.ndarray:
    tf = vehicle.get_transform()
    return np.array(
        [tf.location.x, tf.location.y, math.radians(tf.rotation.yaw)],
        dtype=np.float32,
    )


def _attach_collision(world, vehicle):
    hit = {"flag": False}
    bp = world.get_blueprint_library().find("sensor.other.collision")
    sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)

    def _on(_event):
        hit["flag"] = True

    sensor.listen(_on)
    return sensor, hit


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
    use_occ_fusion: bool = True,
    trajectories: bool = False,
    episode_len: int = 200,
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

    if trajectories:
        steps_per_sample = 1
        min_occupied = 0
        min_classes = 0
        print(
            f"[Collect] trajectory mode  expert=CARLA Traffic Manager autopilot  "
            f"episode_len={episode_len}",
            flush=True,
        )
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
    world.tick()
    collision_sensor, collision_hit = _attach_collision(world, vehicle)
    global_tgt = GlobalTarget(
        world, vehicle, horizon_s=5.0, d_min=5.0, d_max=bev_x_range[1], v_max=8.0,
    )

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
    occ_fusion = TemporalOccFusion(bev, occ_thresh=2.0) if use_occ_fusion else None
    cam_ext = rig.cam_extrinsic
    print(f"[Collect] Occupancy volume: {bev.grid_z}x{bev.grid_h}x{bev.grid_w}  "
          f"voxel={bev_voxel}m  pitch={pitch_deg}°")
    if occ_fusion is not None:
        print(
            f"[Collect] temporal occupancy fusion  decay={occ_fusion.decay:.2f}  "
            f"thresh={occ_fusion.occ_thresh:.2f}",
            flush=True,
        )
    else:
        print("[Collect] temporal occupancy fusion  off", flush=True)

    # warm up
    print("[Collect] Warming up...", flush=True)
    for _ in range(10):
        world.tick()
        time.sleep(0.1)

    # ── collect loop ──
    collected = 0
    skipped = 0
    class_stats = np.zeros(NUM_BEV_CLASSES, dtype=np.int64)
    episode_id = 0
    step = 0
    prev_remain = None

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

            tf = vehicle.get_transform()
            ego_xy_yaw = (
                float(tf.location.x), float(tf.location.y),
                math.radians(tf.rotation.yaw),
            )
            seg_gt = seg_image   # IMAGE-SPACE segmentation (camera view)
            if occ_fusion is not None:
                occ_gt, _ = occ_fusion.update(
                    result["occupancy_count"], result["voxel_class"], ego_xy_yaw,
                )
            else:
                occ_gt = occ_head(result["occupancy_count"])  # (Z, H, W)
            occupancy_seg_gt = result["class_volume"]  # (C, Z, H, W)
            occ_bev = occupancy_to_bev(occ_gt)

            n_occupied = int(occ_gt.sum())
            n_classes = len(np.unique(seg_gt))

            if n_occupied < min_occupied or n_classes < min_classes:
                skipped += 1
                continue

            speed = _speed(vehicle)
            plan = global_tgt.update(speed=speed, n_knots=8)
            target = _target_from_plan(plan, speed)
            control_gt = _expert_control(vehicle)
            remain = float(global_tgt._remaining_arclength())
            progress_m = 0.0 if prev_remain is None else float(prev_remain - remain)
            prev_remain = remain
            cte = _cross_track(plan.get("route_ego"))
            collided = bool(collision_hit["flag"])
            collision_hit["flag"] = False
            done = bool(trajectories and (collided or (step + 1) >= episode_len))

            # track stats (image-space)
            for c in range(NUM_BEV_CLASSES):
                class_stats[c] += (seg_gt == c).sum()

            sample_kw = dict(
                left_rgb=data["left_rgb"],
                right_rgb=data["right_rgb"],
                depth_gt=depth,
                seg_gt=seg_gt,
                occ_gt=occ_gt,
                occupancy_seg_gt=occupancy_seg_gt,
                K=rig.K,
                cam_ext=cam_ext,
                x_range=np.array(bev_x_range, dtype=np.float32),
                y_range=np.array(bev_y_range, dtype=np.float32),
                z_range=np.array(bev_z_range, dtype=np.float32),
                voxel_size=np.float32(bev_voxel),
                control_gt=control_gt,
                target=target,
                episode_id=np.int32(episode_id),
                step=np.int32(step),
                done=np.uint8(done),
                collided=np.uint8(collided),
                speed=np.float32(speed),
                progress_m=np.float32(progress_m),
                cte=np.float32(cte),
                world_xy_yaw=_world_xy_yaw(vehicle),
            )

            if collected < num_train:
                key = f"/train/sample_{train_count:06d}"
                train_count += 1
            else:
                key = f"/val/sample_{val_count:06d}"
                val_count += 1
            buf = io.BytesIO()
            np.savez_compressed(buf, **sample_kw)
            if not kv.put(key, buf.getvalue()):
                raise OSError(f"PUT {key} failed on {kv_url}")

            collected += 1
            split = "train" if collected <= num_train else "val"
            if collected % 25 == 0 or collected == num_samples:
                loc = vehicle.get_location()
                seg_bev = occupancy_seg_gt.sum(axis=1).argmax(axis=0)
                occupied_classes = np.unique(seg_bev[occ_bev > 0]) if n_occupied else []
                class_names = [BEV_CLASSES.get(int(c), str(c)) for c in occupied_classes]
                z_hit = int((occ_gt.reshape(occ_gt.shape[0], -1).sum(axis=1) > 0).sum())
                print(
                    f"  [{split}] {collected}/{num_samples}  "
                    f"ep={episode_id} step={step}  "
                    f"pos=({loc.x:.0f},{loc.y:.0f})  "
                    f"thr={control_gt[0]:.2f} str={control_gt[2]:+.2f}  "
                    f"occ={n_occupied}/{occ_gt.size}  z_bins={z_hit}/{occ_gt.shape[0]}  "
                    f"classes={class_names}  "
                    f"skipped={skipped}",
                    flush=True,
                )

            if trajectories:
                step += 1
                if done:
                    episode_id += 1
                    step = 0
                    prev_remain = None
                    if occ_fusion is not None:
                        occ_fusion.reset()
                    vehicle.set_autopilot(False)
                    sp = random.choice(spawn_points)
                    vehicle.set_transform(sp)
                    world.tick()
                    vehicle.set_autopilot(True, tm.get_port())
                    global_tgt.pick_destination()
                    collision_hit["flag"] = False

    finally:
        try:
            collision_sensor.stop()
            collision_sensor.destroy()
        except Exception:
            pass
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
    parser.add_argument(
        "--no-occ-fusion", action="store_true",
        help="Disable multi-frame occupancy merge (single-frame lift only)",
    )
    parser.add_argument(
        "--trajectories", action="store_true",
        help="Save consecutive expert episodes (Traffic Manager autopilot); "
             "disables occupancy skip so the sequence stays intact",
    )
    parser.add_argument(
        "--episode-len", type=int, default=200,
        help="Ticks per expert episode when --trajectories is set",
    )
    args = parser.parse_args()

    collect(
        host=args.host, port=args.port,
        num_samples=args.samples,
        val_ratio=args.val_ratio, steps_per_sample=args.steps_per_sample,
        num_vehicles=args.num_vehicles, num_walkers=args.num_walkers,
        min_occupied=args.min_occupied, min_classes=args.min_classes,
        kv_url=args.kv_url,
        use_occ_fusion=not args.no_occ_fusion,
        trajectories=args.trajectories,
        episode_len=args.episode_len,
    )
