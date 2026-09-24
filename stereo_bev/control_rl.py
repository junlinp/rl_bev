"""PPO helpers for the stereo VA control head (no CARLA)."""

from __future__ import annotations

import math
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def nearest_forward_m(xy) -> float:
    """Nearest obstacle distance (m) among points with x>0.3 in ego FLU."""
    if xy is None:
        return 99.0
    pts = np.asarray(xy, dtype=np.float64)
    if pts.size == 0:
        return 99.0
    pts = pts.reshape(-1, pts.shape[-1])[:, :2]
    fwd = pts[pts[:, 0] > 0.3]
    if fwd.shape[0] == 0:
        return 99.0
    return float(np.min(np.hypot(fwd[:, 0], fwd[:, 1])))


def front_corridor_cost(xy, half_width: float = 1.8, x_max: float = 14.0) -> float:
    """Cost for obstacles in a  ~3.6 m wide box ahead of the hood.

    Steering so |y| > half_width clears the term; braking increases x and
    also shrinks it. Isotropic distance does not, because route props are
    always a few meters ahead.
    """
    if xy is None:
        return 0.0
    pts = np.asarray(xy, dtype=np.float64)
    if pts.size == 0:
        return 0.0
    pts = pts.reshape(-1, pts.shape[-1])[:, :2]
    cost = 0.0
    for x, y in pts:
        if x <= 0.5 or x > x_max:
            continue
        lat = abs(float(y))
        if lat >= half_width:
            continue
        cost += (half_width - lat) / (float(x) + 0.3)
    return float(cost)


def pedal_safety(control, obs_xy, x_max: float = 6.5, half_width: float = 1.4):
    """Close-range brake if an actor is in a tight box ahead of the hood."""
    arr = np.asarray(control, dtype=np.float64).reshape(-1).copy()
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - int(arr.size)))
    cost = front_corridor_cost(obs_xy, half_width=half_width, x_max=x_max)
    if cost >= 0.25:
        arr[1] = max(float(arr[1]), min(1.0, 0.40 + 0.80 * cost))
        arr[0] = 0.0
    return arr[:3]


def dodge_target(
    target,
    obs_xy,
    lat: float = 2.4,
    y_clip: float = 3.2,
    half_width: float = 1.8,
    x_max: float = 12.0,
) -> np.ndarray:
    """Shift the 5 s target off occupied centerline, clipped so it stays near-lane."""
    t = np.asarray(target, dtype=np.float32).reshape(-1).copy()
    if t.size < 4:
        t = np.pad(t, (0, 4 - int(t.size)))
    t = t[:4]
    if obs_xy is None:
        return t
    pts = np.asarray(obs_xy, dtype=np.float64)
    if pts.size == 0:
        return t
    pts = pts.reshape(-1, pts.shape[-1])[:, :2]
    hit = pts[
        (pts[:, 0] > 0.8)
        & (pts[:, 0] < x_max)
        & (np.abs(pts[:, 1]) < half_width)
    ]
    if hit.shape[0] == 0:
        return t
    left = float(np.sum(hit[:, 1] >= 0.0))
    right = float(np.sum(hit[:, 1] < 0.0))
    side = -1.0 if left >= right else 1.0
    t[1] = float(np.clip(float(t[1]) + side * lat, -y_clip, y_clip))
    return t


def step_reward(
    progress_m: float,
    cte: float,
    speed: float,
    collided: bool,
    v_max: float = 8.0,
    idle_speed: float = 0.4,
    obs_dist: float = 99.0,
    obs_xy=None,
    heading_rad: float = 0.0,
    target_xy=None,
) -> float:
    """Survive without collision, and stay close to the 5 s target pose.

    +1 for a collision-free step, −100 on a hit (episode ends). A small
    speed term keeps the policy from parking. Distance / heading to the
    ego-FLU target add a closeness bonus so off-route targets (large xy)
    are not rewarded.
    """
    del progress_m, cte, obs_dist, obs_xy
    if collided:
        return -100.0
    r = 1.0
    r += 0.15 * float(np.clip(speed, 0.0, v_max))
    if float(speed) < idle_speed:
        r -= 0.5
    if target_xy is not None:
        xy = np.asarray(target_xy, dtype=np.float64).reshape(-1)
        dist = math.hypot(float(xy[0]), float(xy[1]) if xy.size > 1 else 0.0)
        r += 3.0 * math.exp(-dist / 6.0)
    yaw = abs((float(heading_rad) + math.pi) % (2.0 * math.pi) - math.pi)
    r += 0.8 * math.exp(-yaw / 0.5)
    return float(r)


def gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
    last_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation. ``dones[t]`` is 1 if step t ended the episode."""
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    t_len = int(rewards.shape[0])
    adv = np.zeros(t_len, dtype=np.float64)
    last_gae = 0.0
    for t in range(t_len - 1, -1, -1):
        next_v = last_value if t == t_len - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    ret = adv + values
    return adv.astype(np.float32), ret.astype(np.float32)


def ppo_update(
    control_head,
    feats,
    zs,
    old_logp,
    advantages,
    returns,
    clip: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    epochs: int = 4,
    minibatch: int = 64,
    lr: float = 3e-4,
    optimizer=None,
) -> dict[str, float]:
    """On-policy PPO on cached control features. Perception is not in the graph."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for PPO")
    if optimizer is None:
        optimizer = torch.optim.Adam(
            [p for p in control_head.parameters() if p.requires_grad], lr=lr,
        )
    n = int(feats.shape[0])
    total = {"pi": 0.0, "v": 0.0, "ent": 0.0, "n": 0}
    for _ in range(epochs):
        idx = torch.randperm(n, device=feats.device)
        for start in range(0, n, minibatch):
            b = idx[start:start + minibatch]
            logp, value, ent = control_head.evaluate_z(feats[b], zs[b])
            ratio = torch.exp(logp - old_logp[b])
            adv = advantages[b]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
            pi_loss = -torch.min(surr1, surr2).mean()
            v_loss = F.mse_loss(value, returns[b])
            ent_loss = -ent.mean()
            loss = pi_loss + value_coef * v_loss + entropy_coef * ent_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in control_head.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()
            total["pi"] += float(pi_loss.detach())
            total["v"] += float(v_loss.detach())
            total["ent"] += float(ent.mean().detach())
            total["n"] += 1
    k = max(total["n"], 1)
    return {
        "pi_loss": total["pi"] / k,
        "v_loss": total["v"] / k,
        "entropy": total["ent"] / k,
    }
