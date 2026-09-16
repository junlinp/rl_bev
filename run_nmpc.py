"""Run 3D-occupancy NMPC against a live CARLA server.

Open-loop (default): autopilot drives; overlay the 5 s plan on BEV/occ.
Closed-loop: ``python run_nmpc.py --closed-loop`` applies receding-horizon controls.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
_API = os.path.join(_ROOT, "PythonAPI", "carla")
if os.path.isdir(_API) and _API not in sys.path:
    sys.path.insert(0, _API)

from stereo_bev.bev_grid import (
    BEVGrid, occupancy_to_bev,
    DEFAULT_X_RANGE, DEFAULT_Y_RANGE, DEFAULT_Z_RANGE, DEFAULT_VOXEL,
)
from stereo_bev.calibration import DEFAULT_PITCH_DEG
from stereo_bev.camera_rig import CameraRig
from stereo_bev.depth import decode_carla_depth
from stereo_bev.global_target import GlobalTarget
from stereo_bev.nmpc import OccupancyNMPC, nmpc_to_carla_control
from stereo_bev.occ_field import occupancy_to_esdf_3d, obstacle_volume, query_esdf
from stereo_bev.query_heads import GeometricOccHead, HAS_TORCH
from stereo_bev.segmentation import remap_segmentation, NUM_BEV_CLASSES
from stereo_bev.vehicle_body import model3_body_samples, body_sweep_voxels, transform_body
from stereo_bev.rerun_vis import RerunOccViewer, is_available as rerun_available
from stereo_bev.visualize import (
    draw_bev_map, draw_legend, draw_depth_heatmap,
    draw_planning_on_bev, draw_control_curves,
)


def _banner(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.zeros((22, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def _hpad(img: np.ndarray, width: int) -> np.ndarray:
    if img.shape[1] >= width:
        return img
    pad = np.zeros((img.shape[0], width - img.shape[1], 3), dtype=np.uint8)
    return np.concatenate([img, pad], axis=1)


def _vpad(img: np.ndarray, height: int) -> np.ndarray:
    if img.shape[0] >= height:
        return img
    pad = np.zeros((height - img.shape[0], img.shape[1], 3), dtype=np.uint8)
    return np.concatenate([img, pad], axis=0)


def _hstack(imgs: list[np.ndarray]) -> np.ndarray:
    h = max(im.shape[0] for im in imgs)
    return np.concatenate([_vpad(im, h) for im in imgs], axis=1)


def _vstack(imgs: list[np.ndarray]) -> np.ndarray:
    w = max(im.shape[1] for im in imgs)
    return np.concatenate([_hpad(im, w) for im in imgs], axis=0)


def _side_legend(height: int) -> np.ndarray:
    panel = np.full((height, 150, 3), 24, dtype=np.uint8)
    panel = draw_legend(panel, top_left=(8, 8))
    y = 8 + 16 * NUM_BEV_CLASSES + 14
    keys = (
        ((255, 220, 0), "route"),
        ((0, 255, 255), "nmpc"),
        ((0, 0, 255), "target"),
        ((255, 255, 255), "ego"),
    )
    for color, name in keys:
        cv2.rectangle(panel, (8, y), (24, y + 12), color, -1)
        cv2.putText(panel, name, (30, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        y += 16
    return panel


def _steer_max_rad(vehicle) -> float:
    try:
        ang = float(vehicle.get_physics_control().wheels[0].max_steer_angle)
    except Exception:
        ang = 70.0
    if ang > 2.0:
        return math.radians(ang)
    return max(ang, 1e-3)


def _speed(vehicle) -> float:
    v = vehicle.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


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


def run(
    host: str = "localhost",
    port: int = 2000,
    duration_sec: float = 60.0,
    image_w: int = 960,
    image_h: int = 540,
    fov: float = 90.0,
    baseline: float = 0.12,
    fps: float = 15.0,
    bev_x_range=DEFAULT_X_RANGE,
    bev_y_range=DEFAULT_Y_RANGE,
    bev_z_range=DEFAULT_Z_RANGE,
    bev_voxel: float = DEFAULT_VOXEL,
    max_depth: float = 80.0,
    pitch_deg: float = DEFAULT_PITCH_DEG,
    mode: str = "geometric",
    model_checkpoint: str | None = None,
    closed_loop: bool = False,
    metrics_csv: str | None = None,
    z_ground: float = 0.5,
    horizon_s: float = 5.0,
    dt: float = 0.2,
    v_max: float = 2.0,
    use_rerun: bool = True,
    use_opencv: bool = True,
):
    import carla

    client = carla.Client(host, port)
    client.set_timeout(15.0)
    world = client.get_world()
    tm = client.get_trafficmanager()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / fps
    world.apply_settings(settings)
    tm.set_synchronous_mode(True)
    world.tick()
    print("[NMPC] connected, synchronous mode on", flush=True)

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    try:
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    except RuntimeError:
        for actor in world.get_actors().filter("vehicle.*"):
            actor.destroy()
        world.tick()
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    world.tick()
    steer_max_rad = _steer_max_rad(vehicle)

    collision_bp = bp_lib.find("sensor.other.collision")
    collision_sensor = world.spawn_actor(
        collision_bp, carla.Transform(), attach_to=vehicle,
    )
    collisions = {"n": 0}

    def _on_collision(_event):
        collisions["n"] += 1

    collision_sensor.listen(_on_collision)

    npcs = []
    npc_bps = list(bp_lib.filter("vehicle.*"))
    ego_loc = spawn_point.location
    for i, sp in enumerate(world.get_map().get_spawn_points()[1:]):
        if ego_loc.distance(sp.location) < 18.0:
            continue
        try:
            actor = world.spawn_actor(npc_bps[i % len(npc_bps)], sp)
            actor.set_autopilot(True, tm.get_port())
            npcs.append(actor)
            world.tick()
        except RuntimeError:
            pass
        if len(npcs) >= 8:
            break
    print(f"[NMPC] spawned {len(npcs)} traffic vehicles")

    if not closed_loop:
        vehicle.set_autopilot(True, tm.get_port())

    rig = CameraRig(vehicle, world, image_w, image_h, fov, fps, baseline, pitch_deg=pitch_deg)
    bev = BEVGrid(
        x_range=bev_x_range, y_range=bev_y_range, z_range=bev_z_range,
        voxel_size=bev_voxel, num_classes=NUM_BEV_CLASSES,
    )

    model = None
    device = "cpu"
    if mode == "model":
        assert HAS_TORCH, "PyTorch required for model mode"
        assert model_checkpoint, "model_checkpoint required for model mode"
        import torch
        from stereo_bev.query_heads import StereoBEVModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = StereoBEVModel(
            num_classes=NUM_BEV_CLASSES,
            image_h=image_h, image_w=image_w,
            bev_x_range=bev_x_range, bev_y_range=bev_y_range,
            bev_z_range=bev_z_range, bev_voxel=bev_voxel,
            max_depth=max_depth, pretrained_backbone=False,
            cam_extrinsic=rig.cam_extrinsic, pitch_deg=pitch_deg,
        )
        model.load_state_dict(torch.load(model_checkpoint, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"[NMPC] Loaded StereoBEVModel on {device}")
    else:
        occ_head_geo = GeometricOccHead(min_hits=2.0)

    body = model3_body_samples()
    print("[NMPC] Building CasADi problem (once)...")
    nmpc = OccupancyNMPC(grid=bev, body=body, horizon_s=horizon_s, dt=dt, v_max=v_max)
    global_tgt = GlobalTarget(world, vehicle, horizon_s=horizon_s, d_min=5.0, d_max=bev_x_range[1])

    writer = None
    metrics_fp = None
    if metrics_csv:
        metrics_dir = os.path.dirname(os.path.abspath(metrics_csv))
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        metrics_fp = open(metrics_csv, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(metrics_fp, fieldnames=[
            "frame", "t", "speed", "solve_ms", "status", "terminal_err",
            "min_clearance", "cte", "a", "delta", "collisions", "fallback",
        ])
        writer.writeheader()

    cam_extrinsic = rig.cam_extrinsic
    rerun_viewer = None
    if use_rerun:
        if not rerun_available():
            print("[NMPC] rerun-sdk not installed; 3D view disabled (pip install rerun-sdk)")
        else:
            try:
                rerun_viewer = RerunOccViewer(bev, spawn=True)
                print("[NMPC] 3D occupancy is in the Rerun viewer (FLU: +X forward, +Y left, +Z up)")
            except Exception as exc:
                print(f"[NMPC] rerun viewer failed ({exc}); continuing without it", flush=True)
                rerun_viewer = None

    print(f"[NMPC] mode={mode}  closed_loop={closed_loop}  T={horizon_s}s dt={dt}  v_max={nmpc.v_max:.1f}m/s")
    print(f"[NMPC] occupancy {bev.grid_z}x{bev.grid_h}x{bev.grid_w}  voxel={bev_voxel}m")
    print("[NMPC] Press 'q' in the OpenCV window to quit")

    num_frames = int(duration_sec * fps)
    last_u = np.zeros(2)
    speed = 0.0
    try:
        for i in range(num_frames):
            if closed_loop:
                ctrl = nmpc_to_carla_control(
                    float(last_u[0]), float(last_u[1]),
                    speed=speed, v_max=nmpc.v_max,
                    steer_max_rad=steer_max_rad,
                )
                vehicle.apply_control(ctrl)
            world.tick()
            speed = _speed(vehicle)

            data = rig.grab()
            if data is None:
                continue

            left_rgb = data["left_rgb"]
            cls_hist = None
            voxel_class = None
            if model is not None:
                _, occ_map, bev_classes, _ = model.infer(
                    left_rgb, data["right_rgb"], rig.K, device=device, cam_ext=cam_extrinsic,
                )
                depth = decode_carla_depth(data["depth_raw"])
            else:
                depth = decode_carla_depth(data["depth_raw"])
                seg_bev = remap_segmentation(data["seg_raw"])
                bev_result = bev.bev_from_frame(
                    depth_map=depth, seg_map=seg_bev, K=rig.K,
                    cam_extrinsic=cam_extrinsic, max_depth=max_depth,
                )
                cls_hist = bev_result["class_histogram"]
                voxel_class = bev_result["voxel_class"]
                bev_classes = bev.get_bev_semantic(cls_hist)
                occ_map = occ_head_geo(bev_result["occupancy_count"])

            occ_obs = obstacle_volume(
                occ_map, bev, z_ground=z_ground, voxel_class=voxel_class,
                bev_classes=bev_classes,
            )
            esdf = occupancy_to_esdf_3d(
                occ_map, bev, z_ground=z_ground, voxel_class=voxel_class,
                bev_classes=bev_classes,
            )
            plan = global_tgt.update(speed=speed, n_knots=nmpc.N + 1)
            tf = vehicle.get_transform()
            sol = nmpc.solve(
                v=speed, esdf=esdf, target=plan["target"],
                p_ref=plan["p_ref"], yaw_ref=plan["yaw_ref"],
                ego_xy_yaw=(tf.location.x, tf.location.y, math.radians(tf.rotation.yaw)),
            )
            if not sol["used_fallback"]:
                last_u = sol["u0"]

            traj = sol["traj"]
            traj_vis = sol.get("traj_vis", traj)
            sweep = body_sweep_voxels(traj[:, 0], traj[:, 1], traj[:, 2], body, bev)
            cte = _cross_track(plan["route_ego"])

            bev_scale = 4
            bev_img = draw_bev_map(bev_classes, occ_obs, scale=bev_scale, mark_center=False)
            bev_img = draw_planning_on_bev(
                bev_img, bev, bev_scale,
                traj_xy=traj_vis[:, :2],
                route_xy=plan["route_ego"],
                target_xy=plan["target"][:2],
            )
            bev_img = np.ascontiguousarray(np.fliplr(np.rot90(bev_img, k=1)))
            bev_img = np.concatenate([bev_img, _side_legend(bev_img.shape[0])], axis=1)
            bev_img = _banner(bev_img, "BEV  +X up  cyan=route  yellow=nmpc")

            cam_h = 240
            cam_w = int(cam_h * image_w / image_h)
            left_small = cv2.resize(left_rgb, (cam_w, cam_h))
            depth_vis = draw_depth_heatmap(depth, max_depth=max_depth)
            depth_small = cv2.resize(depth_vis, (cam_w, cam_h))
            hud = (
                f"{'CLOSED' if closed_loop else 'OPEN'}  "
                f"v={speed:4.1f}m/s  solve={sol['solve_ms']:.0f}ms  "
                f"{sol['status']}  terr={sol['terminal_err']:.1f}m  "
                f"dmin={sol['min_clearance']:.2f}m  cte={cte:.2f}  col={collisions['n']}"
            )
            cv2.putText(left_small, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(left_small, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
            cam_col = _vstack([
                _banner(left_small, "left RGB"),
                _banner(depth_small, "depth"),
            ])

            if rerun_viewer is not None and i % 2 == 0:
                try:
                    log_cams = (i % 10 == 0)
                    rerun_viewer.log_frame(
                        i,
                        occ=occ_obs.astype(np.uint8),
                        sweep=sweep,
                        bev_classes=bev_classes,
                        voxel_class=voxel_class,
                        traj_xy=traj_vis[:, :2],
                        route_xy=plan["route_ego"],
                        target_xy=plan["target"][:2],
                        rgb_bgr=left_rgb if log_cams else None,
                        depth=depth if log_cams else None,
                        bev_bgr=bev_img if log_cams else None,
                        speed=speed,
                        accel=float(last_u[0]),
                        steer=float(last_u[1]),
                        clearance=float(sol["min_clearance"]),
                        max_depth=max_depth,
                    )
                except Exception as exc:
                    print(f"[NMPC] rerun log failed ({exc}); disabling 3D viewer", flush=True)
                    try:
                        rerun_viewer.close()
                    except Exception:
                        pass
                    rerun_viewer = None

            if use_opencv:
                ctrl_panel = draw_control_curves(
                    sol["u"], dt=dt, a_max=nmpc.a_max, delta_max=nmpc.delta_max,
                    width=bev_img.shape[1], height=140,
                )
                mid_col = _vstack([
                    bev_img,
                    _banner(ctrl_panel, "u(t)  a green  delta orange"),
                ])
                canvas = _hstack([cam_col, mid_col])
                try:
                    if i == 0:
                        cv2.namedWindow("NMPC 3D occupancy", cv2.WINDOW_NORMAL)
                        max_w, max_h = 1600, 900
                        scale = min(max_w / canvas.shape[1], max_h / canvas.shape[0], 1.0)
                        cv2.resizeWindow(
                            "NMPC 3D occupancy",
                            max(int(canvas.shape[1] * scale), 640),
                            max(int(canvas.shape[0] * scale), 360),
                        )
                        cv2.moveWindow("NMPC 3D occupancy", 30, 40)
                    cv2.imshow("NMPC 3D occupancy", canvas)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error as exc:
                    if i == 0:
                        print(f"[NMPC] opencv viz unavailable ({exc}); continuing", flush=True)

            if writer is not None:
                writer.writerow({
                    "frame": i, "t": i / fps, "speed": f"{speed:.3f}",
                    "solve_ms": f"{sol['solve_ms']:.2f}", "status": sol["status"],
                    "terminal_err": f"{sol['terminal_err']:.3f}",
                    "min_clearance": f"{sol['min_clearance']:.3f}",
                    "cte": f"{cte:.3f}",
                    "a": f"{float(last_u[0]):.3f}", "delta": f"{float(last_u[1]):.3f}",
                    "collisions": collisions["n"], "fallback": int(sol["used_fallback"]),
                })
                metrics_fp.flush()

            if i % 15 == 0:
                occ_bev = occupancy_to_bev(occ_obs)
                body0 = transform_body(0.0, 0.0, 0.0, body)
                d0 = query_esdf(esdf, bev, body0)
                print(
                    f"  frame {i}/{num_frames}  solve={sol['solve_ms']:.0f}ms  "
                    f"{sol['status']}  terr={sol['terminal_err']:.2f}  "
                    f"dmin={sol['min_clearance']:.2f}  ego_d={float(np.min(d0)):.2f}  "
                    f"occ={int(occ_obs.sum())}  raw={int(occ_map.sum())}  bev={int(occ_bev.sum())}",
                    flush=True,
                )
    finally:
        if metrics_fp is not None:
            metrics_fp.close()
        collision_sensor.stop()
        collision_sensor.destroy()
        for actor in npcs:
            try:
                actor.destroy()
            except Exception:
                pass
        rig.destroy()
        vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)
        if rerun_viewer is not None:
            rerun_viewer.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print("[NMPC] Done.")


def main():
    p = argparse.ArgumentParser(description="3D occupancy NMPC in CARLA")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--closed-loop", action="store_true")
    p.add_argument("--mode", default="geometric", choices=["geometric", "model"])
    p.add_argument("--model-checkpoint", default=None)
    p.add_argument("--metrics-csv", default="nmpc_metrics.csv")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--v-max", type=float, default=2.0, help="NMPC speed limit (m/s)")
    p.add_argument("--no-rerun", action="store_true", help="Disable the Rerun 3D occupancy viewer")
    p.add_argument("--no-opencv", action="store_true", help="Disable the OpenCV HUD window")
    args = p.parse_args()
    run(
        host=args.host, port=args.port, duration_sec=args.duration,
        fps=args.fps, mode=args.mode, model_checkpoint=args.model_checkpoint,
        closed_loop=args.closed_loop, metrics_csv=args.metrics_csv,
        v_max=args.v_max, use_rerun=not args.no_rerun, use_opencv=not args.no_opencv,
    )


if __name__ == "__main__":
    main()
