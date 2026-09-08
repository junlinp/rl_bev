"""
RL-based BEV planner.

Takes BEV segmentation + occupancy as observation, outputs trajectory
waypoints for vehicle autopilot.

Architecture:
  BEV (seg+occ) → CNN encoder ─┐
                                 ├→ policy head → waypoints (x, y, heading)
  ego state (vel, heading) → MLP┘

Training: PPO in CARLA environment
Inference: greedy action from policy
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ════════════════════════════════════════════════════════════════
#  Trajectory representation
# ════════════════════════════════════════════════════════════════

WAYPOINTS_AHEAD = 10      # predict 10 future waypoints
WAYPOINT_DT = 0.5         # seconds between waypoints


def waypoints_to_control(
    waypoints: np.ndarray,
    current_speed: float,
    dt: float = 0.1,
) -> tuple[float, float]:
    """
    Convert predicted waypoints to throttle/steer control.

    Args:
        waypoints: (N, 2) array of (x, y) in ego frame, meters
        current_speed: m/s
        dt: control timestep

    Returns:
        throttle: [-1, 1] (negative = brake)
        steer:    [-1, 1] (left/right)
    """
    if len(waypoints) < 2:
        return 0.0, 0.0

    # steering: angle to first waypoint
    target = waypoints[min(2, len(waypoints) - 1)]
    angle = np.arctan2(target[1], target[0])  # atan2(y, x)
    steer = np.clip(angle / (np.pi / 4), -1.0, 1.0)  # normalize to ±45°

    # speed control: target speed from waypoint spacing
    if len(waypoints) >= 2:
        dist = np.linalg.norm(waypoints[1] - waypoints[0])
        target_speed = dist / WAYPOINT_DT
    else:
        target_speed = 3.0  # default 3 m/s

    speed_error = target_speed - current_speed
    throttle = np.clip(speed_error * 0.5, -1.0, 1.0)

    return throttle, steer


# ════════════════════════════════════════════════════════════════
#  RL Policy Network
# ════════════════════════════════════════════════════════════════

if HAS_TORCH:

    class BEVEncoder(nn.Module):
        """CNN encoder for BEV observation."""
        def __init__(self, in_channels: int, out_features: int = 256):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
            )
            self.fc = nn.Linear(128 * 4 * 4, out_features)

        def forward(self, bev: torch.Tensor) -> torch.Tensor:
            return F.relu(self.fc(self.conv(bev)))

    class EgoStateEncoder(nn.Module):
        """MLP encoder for ego vehicle state."""
        def __init__(self, state_dim: int = 4, out_features: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, out_features),
                nn.ReLU(inplace=True),
            )

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return self.net(state)

    class WaypointPolicy(nn.Module):
        """
        Policy network that outputs trajectory waypoints.

        Observation: BEV grid (seg + occ) + ego state (vx, vy, speed, heading)
        Action: sequence of (dx, dy) deltas → waypoints in ego frame

        Supports both deterministic (inference) and stochastic (training) modes.
        """
        def __init__(
            self,
            bev_channels: int = 11,   # num_classes + 1 (occ)
            bev_size: int = 100,      # grid H/W
            num_waypoints: int = WAYPOINTS_AHEAD,
            bev_feat_dim: int = 256,
            ego_feat_dim: int = 64,
        ):
            super().__init__()
            self.num_waypoints = num_waypoints

            self.bev_encoder = BEVEncoder(bev_channels, bev_feat_dim)
            self.ego_encoder = EgoStateEncoder(4, ego_feat_dim)

            combined_dim = bev_feat_dim + ego_feat_dim

            # shared trunk
            self.trunk = nn.Sequential(
                nn.Linear(combined_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
            )

            # policy head: outputs mean + log_std for each waypoint delta
            # each waypoint is (dx, dy) → 2 values per waypoint
            self.action_dim = num_waypoints * 2
            self.policy_mean = nn.Linear(128, self.action_dim)
            self.policy_log_std = nn.Linear(128, self.action_dim)

            # value head (for PPO critic)
            self.value_head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )

        def forward(
            self,
            bev_obs: torch.Tensor,
            ego_state: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Args:
                bev_obs:   (B, C, H, W) BEV observation
                ego_state: (B, 4) [vx, vy, speed, heading]

            Returns:
                action_mean: (B, num_waypoints*2)
                action_log_std: (B, num_waypoints*2)
                value: (B, 1)
            """
            bev_feat = self.bev_encoder(bev_obs)
            ego_feat = self.ego_encoder(ego_state)
            combined = torch.cat([bev_feat, ego_feat], dim=1)
            trunk_out = self.trunk(combined)

            action_mean = self.policy_mean(trunk_out)
            action_log_std = self.policy_log_std(trunk_out).clamp(-5, 2)
            value = self.value_head(trunk_out)

            return action_mean, action_log_std, value

        def get_action(
            self,
            bev_obs: torch.Tensor,
            ego_state: torch.Tensor,
            deterministic: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Sample action from policy.

            Returns:
                action: (B, num_waypoints*2) waypoint deltas
                log_prob: (B,) log probability
                value: (B, 1)
            """
            mean, log_std, value = self.forward(bev_obs, ego_state)
            std = log_std.exp()

            if deterministic:
                action = mean
                log_prob = torch.zeros(mean.shape[0], device=mean.device)
            else:
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

            return action, log_prob, value

        def action_to_waypoints(
            self,
            action: torch.Tensor,
        ) -> torch.Tensor:
            """
            Convert action deltas to absolute waypoints in ego frame.

            Args:
                action: (B, num_waypoints*2) as (dx, dy) deltas

            Returns:
                waypoints: (B, num_waypoints, 2) as (x, y) in ego frame
            """
            B = action.shape[0]
            deltas = action.view(B, self.num_waypoints, 2)  # (B, N, 2)
            # cumulative sum to get absolute positions
            waypoints = torch.cumsum(deltas, dim=1)
            return waypoints

        def infer(
            self,
            bev_seg: np.ndarray,
            bev_occ: np.ndarray,
            ego_state: np.ndarray,
            device: str = "cpu",
        ) -> np.ndarray:
            """
            Inference: BEV → waypoints.

            Args:
                bev_seg: (H, W) uint8 class indices
                bev_occ: (H, W) uint8 binary
                ego_state: (4,) [vx, vy, speed, heading]

            Returns:
                waypoints: (num_waypoints, 2) (x, y) in ego frame, meters
            """
            # build BEV observation: one-hot seg + occ channel
            num_classes = 10
            H, W = bev_seg.shape
            bev_onehot = np.zeros((num_classes, H, W), dtype=np.float32)
            for c in range(num_classes):
                bev_onehot[c] = (bev_seg == c).astype(np.float32)
            bev_obs = np.concatenate([bev_onehot, bev_occ[np.newaxis].astype(np.float32)], axis=0)

            bev_t = torch.from_numpy(bev_obs).float().unsqueeze(0).to(device)
            ego_t = torch.from_numpy(ego_state).float().unsqueeze(0).to(device)

            self.eval()
            with torch.no_grad():
                action, _, _ = self.get_action(bev_t, ego_t, deterministic=True)
                waypoints = self.action_to_waypoints(action)

            return waypoints.squeeze(0).cpu().numpy()


# ════════════════════════════════════════════════════════════════
#  PPO Trainer
# ════════════════════════════════════════════════════════════════

if HAS_TORCH:

    class PPOTrainer:
        """Proximal Policy Optimization for the waypoint policy."""

        def __init__(
            self,
            policy: WaypointPolicy,
            lr: float = 3e-4,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_epsilon: float = 0.2,
            entropy_coeff: float = 0.01,
            value_coeff: float = 0.5,
            max_grad_norm: float = 0.5,
        ):
            self.policy = policy
            self.gamma = gamma
            self.gae_lambda = gae_lambda
            self.clip_epsilon = clip_epsilon
            self.entropy_coeff = entropy_coeff
            self.value_coeff = value_coeff
            self.max_grad_norm = max_grad_norm

            self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

        def compute_gae(
            self,
            rewards: torch.Tensor,
            values: torch.Tensor,
            dones: torch.Tensor,
        ) -> torch.Tensor:
            """Generalized Advantage Estimation."""
            T = len(rewards)
            advantages = torch.zeros_like(rewards)
            gae = 0.0

            for t in reversed(range(T)):
                if t == T - 1:
                    next_value = 0.0
                else:
                    next_value = values[t + 1]

                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae

            return advantages

        def update(
            self,
            bev_obs: torch.Tensor,
            ego_states: torch.Tensor,
            actions: torch.Tensor,
            old_log_probs: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            epochs: int = 4,
            batch_size: int = 64,
        ) -> dict[str, float]:
            """
            PPO update step.

            Args:
                bev_obs:      (T, C, H, W)
                ego_states:   (T, 4)
                actions:      (T, action_dim)
                old_log_probs: (T,)
                rewards:      (T,)
                dones:        (T,)
                values:       (T,)

            Returns:
                dict with loss metrics
            """
            T = bev_obs.shape[0]
            advantages = self.compute_gae(rewards, values, dones)
            returns = advantages + values

            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            n_updates = 0

            for _ in range(epochs):
                indices = torch.randperm(T)
                for start in range(0, T, batch_size):
                    end = min(start + batch_size, T)
                    idx = indices[start:end]

                    b_bev = bev_obs[idx]
                    b_ego = ego_states[idx]
                    b_act = actions[idx]
                    b_old_lp = old_log_probs[idx]
                    b_adv = advantages[idx]
                    b_ret = returns[idx]

                    # forward
                    mean, log_std, new_values = self.policy(b_bev, b_ego)
                    std = log_std.exp()
                    dist = Normal(mean, std)
                    new_log_probs = dist.log_prob(b_act).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()

                    # policy loss (clipped surrogate)
                    ratio = (new_log_probs - b_old_lp).exp()
                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # value loss
                    value_loss = F.mse_loss(new_values.squeeze(-1), b_ret)

                    # total
                    loss = (
                        policy_loss
                        + self.value_coeff * value_loss
                        - self.entropy_coeff * entropy
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.item()
                    n_updates += 1

            return {
                "policy_loss": total_policy_loss / n_updates,
                "value_loss": total_value_loss / n_updates,
                "entropy": total_entropy / n_updates,
            }
