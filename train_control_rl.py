"""PPO for the stereo VA control head in live CARLA.

Freezes depth / seg / occupancy (the 3D BEV encoder) and updates only the
control policy + critic. Collect expert episodes first if you want a BC
warm-start:

    python collect_data.py --trajectories --samples 2000 --episode-len 200
    python train_bev.py --data bev_data --epochs 20
    python train_control_rl.py --model-checkpoint stereo_bev_model.pth --updates 50

The expert is CARLA Traffic Manager autopilot (not a human). RL does not
need those trajectories at update time; it rolls out the policy online.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
_API = os.path.join(_ROOT, "PythonAPI", "carla")
if os.path.isdir(_API) and _API not in sys.path:
    sys.path.insert(0, _API)

from stereo_bev.bev_grid import (
    BEVGrid, DEFAULT_VOXEL, DEFAULT_X_RANGE, DEFAULT_Y_RANGE, DEFAULT_Z_RANGE,
)
from stereo_bev.calibration import DEFAULT_PITCH_DEG
from stereo_bev.camera_rig import CameraRig
from stereo_bev.control_rl import dodge_target, gae, pedal_safety, ppo_update, step_reward
from stereo_bev.depth import decode_carla_depth
from stereo_bev.global_target import GlobalTarget, yaw_from_R
from stereo_bev.query_heads import GeometricOccHead, StereoBEVModel
from stereo_bev.segmentation import NUM_BEV_CLASSES, remap_segmentation
from run_planner import actor_boxes_in_ego, _cross_track, _speed, _va_to_carla, spawn_road_obstacles


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _nchw(bgr, device, mean, std):
    rgb = np.ascontiguousarray(bgr[:, :, ::-1].transpose(2, 0, 1))
    t = torch.from_numpy(rgb).float().unsqueeze(0).to(device) / 255.0
    return (t - mean) / std


def _target_from_plan(plan, speed: float) -> np.ndarray:
    t = np.asarray(plan["target_t"], dtype=np.float64).reshape(3)
    R = np.asarray(plan["target_R"], dtype=np.float64).reshape(3, 3)
    return np.array([t[0], t[1], yaw_from_R(R), speed], dtype=np.float32)


def _load_model(path, image_h, image_w, cam_ext, pitch_deg, device):
    model = StereoBEVModel(
        num_classes=NUM_BEV_CLASSES,
        image_h=image_h, image_w=image_w,
        bev_x_range=DEFAULT_X_RANGE, bev_y_range=DEFAULT_Y_RANGE,
        bev_z_range=DEFAULT_Z_RANGE, bev_voxel=DEFAULT_VOXEL,
        pretrained_backbone=False, cam_extrinsic=cam_ext, pitch_deg=pitch_deg,
    )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, _ = model.load_state_dict(state, strict=False)
    ctrl_missing = [k for k in missing if k.startswith("control_head")]
    if ctrl_missing:
        print("[rl] control head partially random (missing "
              f"{len(ctrl_missing)} keys)", flush=True)
    model.freeze_perception()
    model.to(device)
    model.eval()
    nudge = "control_rl" not in os.path.basename(path).lower()
    if nudge:
        with torch.no_grad():
            # BC v1 mean is brake-on; shift logits so exclusive pedals start rolling.
            last = model.control_head.mlp[-1]
            last.bias.data[0] += 2.0
            last.bias.data[1] -= 2.5
            model.control_head.log_std.fill_(-0.3)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[rl] trainable control params: {n_train:,}", flush=True)
    return model


def train(
    host: str = "localhost",
    port: int = 2000,
    model_checkpoint: str = "stereo_bev_model.pth",
    output: str = "stereo_bev_control_rl.pth",
    updates: int = 50,
    rollout: int = 256,
    fps: float = 10.0,
    v_max: float = 8.0,
    episode_len: int = 200,
    lr: float = 3e-4,
    road_obstacles: int = 8,
    image_w: int = 960,
    image_h: int = 540,
    fov: float = 90.0,
    baseline: float = 0.12,
    pitch_deg: float = DEFAULT_PITCH_DEG,
    collect_dir: str | None = "bev_data_rl",
    collect_stride: int = 4,
    collect_max: int = 1000,
):
    import carla

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    bp_lib = world.get_blueprint_library()
    for leftover in list(world.get_actors().filter("sensor.*")):
        leftover.destroy()
    for leftover in list(world.get_actors().filter("vehicle.*")):
        leftover.destroy()
    world.tick()
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(bp_lib.filter("vehicle.tesla.model3")[0], spawn_points[0])
    world.tick()

    hit = {"flag": False}
    col = world.spawn_actor(
        bp_lib.find("sensor.other.collision"), carla.Transform(), attach_to=vehicle,
    )
    col.listen(lambda _e: hit.__setitem__("flag", True))

    rig = CameraRig(vehicle, world, image_w, image_h, fov, fps, baseline, pitch_deg=pitch_deg)
    model = _load_model(
        model_checkpoint, image_h, image_w, rig.cam_extrinsic, pitch_deg, device,
    )
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    opt = torch.optim.Adam(
        [p for p in model.control_head.parameters() if p.requires_grad], lr=lr,
    )

    global_tgt = GlobalTarget(
        world, vehicle, horizon_s=5.0, d_min=5.0, d_max=DEFAULT_X_RANGE[1], v_max=v_max,
    )
    obstacles = spawn_road_obstacles(
        world, vehicle, count=road_obstacles, route=global_tgt._route,
    )

    bev = BEVGrid(
        x_range=DEFAULT_X_RANGE, y_range=DEFAULT_Y_RANGE,
        z_range=DEFAULT_Z_RANGE, voxel_size=DEFAULT_VOXEL,
        num_classes=NUM_BEV_CLASSES,
    )
    occ_head = GeometricOccHead(min_hits=2.0)
    collect_n = {"n": 0, "ep": 0}
    last_ctrl = {"u": np.zeros(3, dtype=np.float32)}
    train_dir = val_dir = None
    if collect_dir:
        train_dir = os.path.join(collect_dir, "train")
        val_dir = os.path.join(collect_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        n_val = max(1, int(0.2 * collect_max))
        n_train = collect_max - n_val
        print(
            f"[rl] collecting depth/seg/occ → {collect_dir}/  "
            f"stride={collect_stride}  cap={collect_max}",
            flush=True,
        )

    def _reset():
        nonlocal obstacles
        hit["flag"] = False
        vehicle.set_autopilot(False)
        vehicle.set_transform(random.choice(spawn_points))
        world.tick()
        global_tgt.pick_destination()
        for actor in obstacles:
            try:
                actor.destroy()
            except Exception:
                pass
        obstacles = spawn_road_obstacles(
            world, vehicle, count=road_obstacles, route=global_tgt._route, quiet=True,
        )
        world.tick()
        collect_n["ep"] += 1

    def _obs_xy():
        box_c, _, _ = actor_boxes_in_ego(obstacles, vehicle.get_transform())
        return box_c[:, :2] if len(box_c) else None

    def _save_gt(data, target, speed, cte):
        if not collect_dir or collect_n["n"] >= collect_max:
            return
        try:
            depth = decode_carla_depth(data["depth_raw"])
            seg_gt = remap_segmentation(data["seg_raw"])
            result = bev.bev_from_frame(
                depth, seg_gt, rig.K, rig.cam_extrinsic, max_depth=80.0,
            )
            occ_gt = occ_head(result["occupancy_count"])
            occupancy_seg_gt = result["class_volume"]
        except Exception:
            return
        i = collect_n["n"]
        if i < n_train:
            path = os.path.join(train_dir, f"sample_{i:06d}.npz")
        else:
            path = os.path.join(val_dir, f"sample_{i - n_train:06d}.npz")
        np.savez_compressed(
            path,
            left_rgb=data["left_rgb"],
            right_rgb=data["right_rgb"],
            depth_gt=depth,
            seg_gt=seg_gt,
            occ_gt=occ_gt,
            occupancy_seg_gt=occupancy_seg_gt,
            K=rig.K,
            cam_ext=np.asarray(rig.cam_extrinsic, dtype=np.float32),
            x_range=np.array(DEFAULT_X_RANGE, dtype=np.float32),
            y_range=np.array(DEFAULT_Y_RANGE, dtype=np.float32),
            z_range=np.array(DEFAULT_Z_RANGE, dtype=np.float32),
            voxel_size=np.float32(DEFAULT_VOXEL),
            control_gt=np.asarray(last_ctrl["u"], dtype=np.float32),
            target=np.asarray(target, dtype=np.float32),
            episode_id=np.int32(collect_n["ep"]),
            step=np.int32(ep_step),
            speed=np.float32(speed),
            cte=np.float32(cte),
        )
        collect_n["n"] = i + 1
        if collect_n["n"] % 50 == 0 or collect_n["n"] == collect_max:
            print(f"[rl] saved {collect_n['n']}/{collect_max} depth/seg/occ samples", flush=True)

    def _observe():
        data = rig.grab()
        if data is None:
            return None
        speed = _speed(vehicle)
        plan = global_tgt.update(speed=speed, n_knots=8)
        target = dodge_target(_target_from_plan(plan, speed), _obs_xy())
        left = _nchw(data["left_rgb"], device, mean, std)
        right = _nchw(data["right_rgb"], device, mean, std)
        k = torch.from_numpy(np.asarray(rig.K)).float().unsqueeze(0).to(device)
        tgt = torch.from_numpy(target).unsqueeze(0).to(device)
        with torch.no_grad():
            bev_feat, occ_logits, *_ = model.encode_bev(
                left, right, k, cam_ext=rig.cam_extrinsic,
            )
            feat = model.control_head.encode(bev_feat, occ_logits, tgt)
        cte = _cross_track(plan.get("route_ego"))
        remain = float(global_tgt._remaining_arclength())
        if collect_dir and ep_step % max(int(collect_stride), 1) == 0:
            _save_gt(data, target, speed, cte)
        return feat, target, speed, cte, remain, plan

    def _wait_obs(max_ticks: int = 40):
        for _ in range(max_ticks):
            world.tick()
            obs = _observe()
            if obs is not None:
                return obs
        raise RuntimeError("camera rig produced no frames")

    print(f"[rl] PPO  device={device}  rollout={rollout}  updates={updates}", flush=True)
    ep_step = 0
    try:
        for upd in range(updates):
            feats, zs, logps, values, rewards, dones = [], [], [], [], [], []
            feat, target, speed, cte, remain, plan = _wait_obs()
            ret_sum = 0.0
            cols = 0
            speed_sum = 0.0
            thr_sum = 0.0
            brk_sum = 0.0
            cte_sum = 0.0
            tgt_sum = 0.0
            for _t in range(rollout):
                control, logp, value, z, _ent = model.control_head.act_from_feat(feat)
                remain0 = remain
                ctrl_np = control.squeeze(0).detach().cpu().numpy()
                last_ctrl["u"] = np.asarray(ctrl_np, dtype=np.float32).reshape(3)
                thr_sum += float(ctrl_np[0])
                brk_sum += float(ctrl_np[1])
                vehicle.apply_control(_va_to_carla(pedal_safety(ctrl_np, _obs_xy())))
                world.tick()
                collided = bool(hit["flag"])
                hit["flag"] = False
                ep_step += 1
                nxt = _observe()
                if nxt is None:
                    progress = 0.0
                    speed_n = speed
                    cte_n = cte
                    remain_n = remain
                    feat_n = feat
                    target_n = target
                else:
                    feat_n, target_n, speed_n, cte_n, remain_n, plan = nxt
                    progress = remain0 - remain_n
                if collided:
                    progress = 0.0
                done = (
                    collided
                    or ep_step >= episode_len
                    or abs(float(cte_n)) > 3.5
                )
                r = step_reward(
                    progress, cte_n, speed_n, collided, v_max=v_max,
                    obs_xy=_obs_xy(), heading_rad=float(target_n[2]),
                    target_xy=target_n[:2],
                )
                feats.append(feat.detach())
                zs.append(z.detach())
                logps.append(logp.detach())
                values.append(value.detach())
                rewards.append(r)
                dones.append(1.0 if done else 0.0)
                ret_sum += r
                cols += int(collided)
                speed_sum += float(speed_n)
                cte_sum += abs(float(cte_n))
                tgt_sum += float(math.hypot(float(target_n[0]), float(target_n[1])))
                feat, target, speed, cte, remain = feat_n, target_n, speed_n, cte_n, remain_n
                if done:
                    _reset()
                    ep_step = 0
                    feat, target, speed, cte, remain, plan = _wait_obs()

            last_v = 0.0
            if dones[-1] < 0.5:
                with torch.no_grad():
                    last_v = float(model.control_head.critic(feat).squeeze().cpu())
            adv, ret = gae(
                np.asarray(rewards),
                torch.stack(values).squeeze(-1).cpu().numpy(),
                np.asarray(dones),
                last_value=last_v,
            )
            adv_t = torch.from_numpy(adv).to(device)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            stats = ppo_update(
                model.control_head,
                torch.cat(feats, dim=0),
                torch.cat(zs, dim=0),
                torch.cat(logps, dim=0),
                adv_t,
                torch.from_numpy(ret).to(device),
                optimizer=opt, lr=lr,
            )
            torch.save({"model": model.state_dict(), "update": upd}, output)
            print(
                f"  update {upd+1}/{updates}  ret={ret_sum:.1f}  col={cols}  "
                f"v={speed_sum / rollout:.2f}  cte={cte_sum / rollout:.2f}  "
                f"tgt={tgt_sum / rollout:.1f}  "
                f"thr={thr_sum / rollout:.2f}  "
                f"brk={brk_sum / rollout:.2f}  "
                f"pi={stats['pi_loss']:.3f} vloss={stats['v_loss']:.3f} "
                f"ent={stats['entropy']:.3f}",
                flush=True,
            )
    finally:
        col.stop()
        col.destroy()
        for actor in obstacles:
            try:
                actor.destroy()
            except Exception:
                pass
        rig.destroy()
        vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)
        if collect_dir:
            print(f"[rl] wrote {collect_n['n']} gt samples under {collect_dir}", flush=True)
        print(f"[rl] checkpoint → {output}", flush=True)


def main():
    p = argparse.ArgumentParser(description="PPO for the stereo VA control head")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--model-checkpoint", default="stereo_bev_model.pth")
    p.add_argument("--output", default="stereo_bev_control_rl.pth")
    p.add_argument("--updates", type=int, default=50)
    p.add_argument("--rollout", type=int, default=256)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--v-max", type=float, default=8.0)
    p.add_argument("--episode-len", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--road-obstacles", type=int, default=8)
    p.add_argument("--collect-dir", default="bev_data_rl")
    p.add_argument("--collect-stride", type=int, default=4)
    p.add_argument("--collect-max", type=int, default=1000)
    p.add_argument("--no-collect", action="store_true")
    args = p.parse_args()
    train(
        host=args.host, port=args.port,
        model_checkpoint=args.model_checkpoint, output=args.output,
        updates=args.updates, rollout=args.rollout, fps=args.fps,
        v_max=args.v_max, episode_len=args.episode_len, lr=args.lr,
        road_obstacles=args.road_obstacles,
        collect_dir=None if args.no_collect else args.collect_dir,
        collect_stride=args.collect_stride,
        collect_max=args.collect_max,
    )


if __name__ == "__main__":
    main()
