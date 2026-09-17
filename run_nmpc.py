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
from stereo_bev.calibration import DEFAULT_PITCH_DEG, camera_origin_xy
from stereo_bev.camera_rig import CameraRig
from stereo_bev.depth import decode_carla_depth
from stereo_bev.global_target import (
    GlobalTarget, interpolate_polyline, interpolate_yaw, _cum_arclength,
    transform_points_between_ego,
)
from stereo_bev.nmpc import nmpc_to_carla_control
from stereo_bev.corridor_qp import CorridorQP
from stereo_bev.occ_field import (
    obstacle_volume, query_esdf, query_esdf_z_slack,
    occupied_to_esdf, stamp_oriented_boxes, shift_ref_se3,
)
from stereo_bev.sfc import (
    assign_polyhedra, build_safe_flight_corridor, pack_sfc_parameters,
    polyhedron_wireframe, refs_along_path,
)
from stereo_bev.query_heads import GeometricOccHead, HAS_TORCH
from stereo_bev.segmentation import remap_segmentation, NUM_BEV_CLASSES
from stereo_bev.vehicle_body import (
    MODEL3_HEIGHT, MODEL3_LENGTH, MODEL3_WIDTH,
    model3_collision_balls, transform_body, transform_body_se3,
    collision_inflation_radius,
)
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


def _lane_offset_xy(x: float, y: float, yaw_rad: float, lat_m: float) -> tuple[float, float]:
    """World XY of a point ``lat_m`` to the right of a CARLA heading."""
    return x - lat_m * math.sin(yaw_rad), y + lat_m * math.cos(yaw_rad)


def _yaw_along(xy: np.ndarray) -> np.ndarray:
    if len(xy) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(xy) == 1:
        return np.zeros(1, dtype=np.float64)
    d = np.diff(xy, axis=0)
    yaw = np.arctan2(d[:, 1], d[:, 0])
    return np.concatenate([yaw[:1], yaw])


def _find_blueprint(bp_lib, names: tuple[str, ...]):
    for name in names:
        try:
            return bp_lib.find(name)
        except IndexError:
            continue
    return None


def spawn_road_obstacles(world, vehicle, count: int = 8, route=None) -> list:
    """Park static props and cars on the global route, in camera range."""
    import carla

    if count <= 0:
        return []
    bp_lib = world.get_blueprint_library()
    tf = vehicle.get_transform()
    yaw0 = math.radians(tf.rotation.yaw)
    fx0, fy0 = math.cos(yaw0), math.sin(yaw0)
    rx0, ry0 = -fy0, fx0
    mmap = world.get_map()

    route_xy = route_s = route_z = route_yaw = None
    if route:
        wx = np.array([wp.transform.location.x for wp, _ in route], dtype=np.float64)
        wy = np.array([wp.transform.location.y for wp, _ in route], dtype=np.float64)
        wz = np.array([wp.transform.location.z for wp, _ in route], dtype=np.float64)
        wyaw = np.radians([wp.transform.rotation.yaw for wp, _ in route])
        if len(wx) >= 2:
            route_xy = np.stack([wx, wy], axis=1)
            route_s = _cum_arclength(route_xy)
            route_z = wz
            route_yaw = wyaw
            print(
                f"[NMPC] placing obstacles on global route  "
                f"({float(route_s[-1]):.0f} m, {len(wx)} waypoints)",
                flush=True,
            )

    cone_bp = _find_blueprint(bp_lib, (
        "static.prop.constructioncone",
        "static.prop.trafficcone01",
        "static.prop.trafficcone02",
        "static.prop.trafficwarning",
    ))
    barrier_bp = _find_blueprint(bp_lib, (
        "static.prop.streetbarrier",
        "static.prop.warningconstruction",
        "static.prop.container",
    ))
    box_bp = _find_blueprint(bp_lib, (
        "static.prop.container",
        "static.prop.box01",
        "static.prop.bin",
        "static.prop.barrel",
        "static.prop.trashcan01",
    ))
    car_bps = [
        bp for bp in bp_lib.filter("vehicle.*")
        if bp.has_attribute("number_of_wheels")
        and int(bp.get_attribute("number_of_wheels")) == 4
        and "tesla.model3" not in bp.id
    ]

    # Along-route meters, lateral m CARLA-right, kind, extra yaw deg.
    # Centerline entries sit on the global path so NMPC must leave the lane.
    layout = (
        (8.0, 0.0, "cone", 0.0),
        (8.6, 0.55, "cone", 0.0),
        (8.6, -0.55, "cone", 0.0),
        (12.0, 0.0, "car", 0.0),
        (16.0, 1.3, "barrier", 90.0),
        (19.0, -1.1, "box", 0.0),
        (22.0, 0.4, "cone", 0.0),
        (26.0, -0.8, "barrier", 90.0),
        (33.0, 0.0, "car", 0.0),
        (40.0, 1.2, "box", 0.0),
        (46.0, -1.0, "cone", 0.0),
    )

    spawned = []
    car_i = 0
    for dist, lat, kind, yaw_off in layout:
        if len(spawned) >= count:
            break
        if route_s is not None and float(route_s[-1]) >= dist:
            xy = interpolate_polyline(route_xy, route_s, np.array([dist]))[0]
            yaw = float(interpolate_yaw(route_yaw, route_s, np.array([dist]))[0])
            wz = float(np.interp(dist, route_s, route_z))
            fx, fy = math.cos(yaw), math.sin(yaw)
            rx, ry = -fy, fx
            wx = float(xy[0]) + lat * rx
            wy = float(xy[1]) + lat * ry
            yaw_deg = math.degrees(yaw) + yaw_off
        else:
            wx = tf.location.x + dist * fx0 + lat * rx0
            wy = tf.location.y + dist * fy0 + lat * ry0
            wz = tf.location.z
            yaw_deg = tf.rotation.yaw + yaw_off
        wz = wz + 0.15
        probe = carla.Location(x=wx, y=wy, z=wz + 2.0)
        wp = mmap.get_waypoint(probe, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is not None and wp.transform.location.distance(carla.Location(x=wx, y=wy, z=wz)) < 6.0:
            wz = wp.transform.location.z + 0.12
        spawn_tf = carla.Transform(
            carla.Location(x=wx, y=wy, z=wz),
            carla.Rotation(pitch=tf.rotation.pitch, yaw=yaw_deg, roll=0.0),
        )
        if kind == "cone":
            bp = cone_bp or box_bp or barrier_bp
        elif kind == "barrier":
            bp = barrier_bp or box_bp or cone_bp
        elif kind == "box":
            bp = box_bp or barrier_bp or cone_bp
        else:
            bp = car_bps[car_i % len(car_bps)] if car_bps else (box_bp or barrier_bp)
            car_i += 1
        if bp is None:
            continue
        actor = world.try_spawn_actor(bp, spawn_tf)
        if actor is None:
            spawn_tf.location.z += 0.5
            actor = world.try_spawn_actor(bp, spawn_tf)
        if actor is None:
            continue
        try:
            actor.set_simulate_physics(False)
        except Exception:
            pass
        if actor.type_id.startswith("vehicle."):
            actor.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
        spawned.append(actor)
        world.tick()
        loc = actor.get_transform().transform(actor.bounding_box.location)
        now = np.array(vehicle.get_transform().get_inverse_matrix(), dtype=np.float64)
        veh = now @ np.array([loc.x, loc.y, loc.z, 1.0])
        print(
            f"[NMPC]   {actor.type_id.split('.')[-1]:16s}  "
            f"ego x={veh[0]:5.1f} y={-veh[1]:5.1f} (left+)",
            flush=True,
        )
    names = ", ".join(a.type_id.split(".")[-1] for a in spawned) or "none"
    print(f"[NMPC] spawned {len(spawned)} road obstacles on global path: {names}", flush=True)
    return spawned


def actor_boxes_in_ego(actors, ego_tf, x_max: float = 22.0, y_max: float = 10.0):
    """CARLA actor bounding boxes in ego BEV (Y left), for Rerun."""
    from stereo_bev.global_target import yaw_world_to_ego

    inv = np.array(ego_tf.get_inverse_matrix(), dtype=np.float64)
    ego_yaw = math.radians(ego_tf.rotation.yaw)
    centers, halves, quats = [], [], []
    for actor in actors:
        try:
            bb = actor.bounding_box
            wt = actor.get_transform()
            loc = wt.transform(bb.location)
        except Exception:
            continue
        veh = inv @ np.array([loc.x, loc.y, loc.z, 1.0], dtype=np.float64)
        x, y, z = float(veh[0]), float(-veh[1]), float(veh[2])
        if x < -2.0 or x > x_max or abs(y) > y_max:
            continue
        ext = bb.extent
        yaw_e = float(yaw_world_to_ego(math.radians(wt.rotation.yaw), ego_yaw))
        centers.append([x, y, z])
        halves.append([float(ext.x), float(ext.y), float(ext.z)])
        quats.append([0.0, 0.0, math.sin(0.5 * yaw_e), math.cos(0.5 * yaw_e)])
    if not centers:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
        )
    return (
        np.asarray(centers, dtype=np.float32),
        np.asarray(halves, dtype=np.float32),
        np.asarray(quats, dtype=np.float32),
    )


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
    road_obstacles: int = 8,
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
    for leftover in list(world.get_actors().filter("vehicle.*")):
        leftover.destroy()
    for leftover in list(world.get_actors().filter("static.prop.*")):
        leftover.destroy()
    world.tick()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    spawn_point = world.get_map().get_spawn_points()[0]
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

    body, radii = model3_collision_balls()
    cam_origin = camera_origin_xy(rig.cam_extrinsic)
    print(
        f"[NMPC] Model 3 volume: {len(radii)} balls  r={float(radii[0]):.2f}m  "
        f"(cover {MODEL3_LENGTH:.2f}x{MODEL3_WIDTH:.2f}x{MODEL3_HEIGHT:.2f} m box)",
        flush=True,
    )
    print(
        f"[NMPC] occupancy origin = vehicle center  "
        f"(left camera XY {cam_origin[0]:.2f}, {cam_origin[1]:.2f})",
        flush=True,
    )
    print("[NMPC] Building corridor QP (hard 3D SFC polyhedra, OSQP)...")
    nmpc = CorridorQP(
        grid=bev, body=body, radii=radii, horizon_s=horizon_s, dt=dt, v_max=v_max,
    )
    global_tgt = GlobalTarget(
        world, vehicle, horizon_s=horizon_s, d_min=5.0, d_max=bev_x_range[1],
        v_max=v_max,
    )
    obstacles = spawn_road_obstacles(
        world, vehicle, count=road_obstacles, route=global_tgt._route,
    )

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
                rerun_viewer = RerunOccViewer(bev, spawn=True, origin_xy=cam_origin)
                print("[NMPC] 3D occupancy is in the Rerun viewer (vehicle-center FLU: +X forward, +Y left, +Z up)")
            except Exception as exc:
                print(f"[NMPC] rerun viewer failed ({exc}); continuing without it", flush=True)
                rerun_viewer = None

    print(f"[NMPC] mode={mode}  closed_loop={closed_loop}  T={horizon_s}s dt={dt}  v_max={nmpc.v_max:.1f}m/s")
    print(f"[NMPC] occupancy {bev.grid_z}x{bev.grid_h}x{bev.grid_w}  voxel={bev_voxel}m")
    print("[NMPC] Press 'q' in the OpenCV window to quit")

    num_frames = int(duration_sec * fps)
    last_u = np.zeros(2)
    last_sfc_path = None
    last_sfc_ego = None
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
                occ_vis = np.asarray(occ_map).astype(np.uint8)
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
                occ_count = bev_result["occupancy_count"]
                occ_map = occ_head_geo(occ_count)
                occ_vis = (occ_count > 0).astype(np.uint8)

            occ_obs = obstacle_volume(
                occ_map, bev, z_ground=z_ground, voxel_class=voxel_class,
                bev_classes=bev_classes,
            )
            box_c, box_h, box_q = actor_boxes_in_ego(obstacles, vehicle.get_transform())
            if len(box_c):
                yaws = 2.0 * np.arctan2(box_q[:, 2], box_q[:, 3])
                occ_obs = stamp_oriented_boxes(occ_obs, bev, box_c, box_h, yaws)
            esdf = occupied_to_esdf(occ_obs, bev)
            plan = global_tgt.update(speed=speed, n_knots=nmpc.N + 1)
            r_need = collision_inflation_radius(body, radii, extra=nmpc.r_hard)
            ref_t = np.asarray(plan["ref_t"], dtype=np.float64)
            ref_R = np.asarray(plan["ref_R"], dtype=np.float64)
            t_tgt = np.asarray(plan["target_t"], dtype=np.float64).reshape(3)
            R_tgt = np.asarray(plan["target_R"], dtype=np.float64).reshape(3, 3)
            start = np.array([0.0, 0.0, 0.70], dtype=np.float64)
            goal = t_tgt.copy()
            if float(goal[2]) < 0.3:
                goal[2] = 0.70
            tf = vehicle.get_transform()
            ego_xy_yaw = (
                float(tf.location.x), float(tf.location.y),
                math.radians(tf.rotation.yaw),
            )
            prefer = None
            if last_sfc_path is not None and last_sfc_ego is not None:
                prefer = transform_points_between_ego(
                    last_sfc_path, last_sfc_ego, ego_xy_yaw,
                )
            sfc = build_safe_flight_corridor(
                occ_obs, bev, start, goal, r_inflate=r_need, esdf=esdf,
                n_faces=nmpc.n_sfc_faces, prefer_path=prefer,
                body=body, radii=radii, r_clear=nmpc.r_hard + 0.15,
            )
            poly_A = poly_b = None
            sfc_path = None
            sfc_n = 0
            sfc_edges = None
            if sfc["ok"]:
                ref_t, ref_R = refs_along_path(
                    sfc["path"], nmpc.N + 1, v=nmpc.v_max,
                    horizon_s=nmpc.horizon_s,
                    d_min=global_tgt.d_min, d_max=global_tgt.d_max,
                )
                assigned = assign_polyhedra(
                    ref_t, sfc.get("path_poly", sfc["path"]), sfc["polyhedra"],
                )
                poly_A, poly_b = pack_sfc_parameters(
                    assigned, nmpc.N + 1, nmpc.n_sfc_faces,
                )
                sfc_path = sfc["path"]
                sfc_n = int(sfc["n_poly"])
                last_sfc_path = sfc["path"]
                last_sfc_ego = ego_xy_yaw
                edges = []
                for A, b in sfc["polyhedra"]:
                    e = polyhedron_wireframe(A, b, bev)
                    if len(e):
                        edges.append(e)
                if edges:
                    sfc_edges = np.concatenate(edges, axis=0)
            else:
                ref_t, ref_R = shift_ref_se3(ref_t, ref_R, esdf, bev, r_need=r_need)
            # Cost target stays the unmodified GRP 5 s pose. Do not copy it
            # onto the last knot — an occupied target would drag the body
            # through the obstacle.
            sol = nmpc.solve(
                v=speed, esdf=esdf, target=t_tgt, target_R=R_tgt,
                ref_t=ref_t, ref_R=ref_R,
                ego_xy_yaw=ego_xy_yaw,
                poly_A=poly_A, poly_b=poly_b,
            )
            body_now = transform_body(0.0, 0.0, 0.0, body)
            d_now = query_esdf_z_slack(
                esdf, bev, body_now, z_slack=nmpc.z_slack,
            ) - radii
            ego_clear = float(np.min(d_now)) if d_now.size else 0.0
            if (not sol["used_fallback"]) or (sol["min_clearance"] >= 0.0):
                last_u = sol["u0"].copy()
            # Last-resort stop only if a ball is already inside occupancy.
            # Do not brake at r_hard or the car freezes on the constraint floor.
            if ego_clear < 0.0:
                last_u = np.array([min(float(last_u[0]), -nmpc.a_max), float(last_u[1])])

            traj = sol["traj"]
            traj_vis = sol.get("traj_vis", traj)
            step = 2
            traj_R = sol.get("traj_R")
            traj_xyz = sol.get("traj_t")
            if traj_R is not None and traj_xyz is not None:
                balls_plan = transform_body_se3(traj_xyz[::step], traj_R[::step], body)
            else:
                balls_plan = transform_body(
                    traj[::step, 0], traj[::step, 1], traj[::step, 2], body,
                )
            sweep_centers = balls_plan.reshape(-1, 3)
            sweep_radii = np.tile(radii, balls_plan.shape[0])
            cte = _cross_track(plan["route_ego"])
            traj_xyz = sol.get("traj_t")
            if traj_xyz is not None and len(traj_xyz) == len(traj_vis):
                traj_occ = traj_xyz
            else:
                traj_occ = traj_vis[:, :2]
            route_xyz = plan.get("route_xyz")
            if route_xyz is not None and len(route_xyz):
                route_occ = route_xyz
            else:
                route_occ = plan["route_ego"]
            target_occ = t_tgt
            body_occ = body_now

            bev_scale = 4
            bev_img = draw_bev_map(bev_classes, occ_vis, scale=bev_scale, mark_center=False)
            bev_img = draw_planning_on_bev(
                bev_img, bev, bev_scale,
                traj_xy=traj_occ,
                route_xy=route_occ,
                target_xy=target_occ,
                ref_xy=sfc_path,
            )
            bev_img = np.ascontiguousarray(np.fliplr(np.rot90(bev_img, k=1)))
            bev_img = np.concatenate([bev_img, _side_legend(bev_img.shape[0])], axis=1)
            bev_img = _banner(bev_img, "BEV  +X up  cyan=route  yellow=nmpc  green=sfc")

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
                _banner(depth_small, "depth  0-80m  occ is 0-20m"),
            ])

            if rerun_viewer is not None and i % 2 == 0:
                try:
                    log_cams = (i % 10 == 0)
                    v_boxes = actor_boxes_in_ego(npcs + obstacles, tf)
                    rerun_viewer.log_frame(
                        i,
                        occ=occ_vis.astype(np.uint8),
                        bev_classes=bev_classes,
                        voxel_class=voxel_class,
                        traj_xy=traj_occ,
                        route_xy=route_occ,
                        target_xy=target_occ,
                        target_R=plan.get("target_R"),
                        rgb_bgr=left_rgb if log_cams else None,
                        depth=depth if log_cams else None,
                        bev_bgr=bev_img if log_cams else None,
                        vehicle_boxes=v_boxes,
                        body_balls=(body_occ, radii),
                        sweep_balls=(sweep_centers, sweep_radii),
                        sfc_path=sfc_path,
                        sfc_edges=sfc_edges,
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
                nearest = "none"
                if len(box_c):
                    j = int(np.argmin(box_c[:, 0] ** 2 + box_c[:, 1] ** 2))
                    nearest = f"({box_c[j, 0]:.1f},{box_c[j, 1]:.1f})"
                print(
                    f"  frame {i}/{num_frames}  solve={sol['solve_ms']:.0f}ms  "
                    f"{sol['status']}  terr={sol['terminal_err']:.2f}  "
                    f"dmin={sol['min_clearance']:.2f}  ego_d={ego_clear:.2f}  "
                    f"plan_y={float(np.max(np.abs(traj[:, 1]))):.2f}  "
                    f"occ={int(occ_vis.sum())}  plan={int(occ_obs.sum())}  "
                    f"raw={int(occ_map.sum())}  bev={int(occ_bev.sum())}  "
                    f"veh={int(((voxel_class == 3) & (occ_vis > 0)).sum()) if voxel_class is not None else 0}  "
                    f"obs={nearest}  col={collisions['n']}  sfc={sfc_n}",
                    flush=True,
                )
    finally:
        if metrics_fp is not None:
            metrics_fp.close()
        collision_sensor.stop()
        collision_sensor.destroy()
        for actor in npcs + obstacles:
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
    p.add_argument(
        "--road-obstacles", type=int, default=8,
        help="Parked cars / cones / barriers on the global route (0 disables)",
    )
    args = p.parse_args()
    run(
        host=args.host, port=args.port, duration_sec=args.duration,
        fps=args.fps, mode=args.mode, model_checkpoint=args.model_checkpoint,
        closed_loop=args.closed_loop, metrics_csv=args.metrics_csv,
        v_max=args.v_max, use_rerun=not args.no_rerun, use_opencv=not args.no_opencv,
        road_obstacles=args.road_obstacles,
    )


if __name__ == "__main__":
    main()
