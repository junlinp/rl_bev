"""
Stereo vision-action model.

Architecture:
  1. Shared backbone extracts features from left + right RGB
  2. Depth predictor: stereo features → per-pixel depth distribution
  3. Seg head: left features → per-pixel semantic segmentation (IMAGE SPACE)
  4. LSS lifter: lift features using predicted depth + K + extrinsics → 3D BEV
  5. Occ / BEV-seg heads query the BEV volume
  6. Control head: BEV queries at ego + target pose → CARLA (throttle, brake, steer)

Key: segmentation is in image space (camera view), NOT BEV projection.
Ground truth seg comes from CARLA's semantic segmentation camera directly.
Occupancy ground truth is a 3D voxel volume lifted from depth+seg.
Expert control labels come from CARLA autopilot at collect time.
"""

import numpy as np

from .bev_grid import DEFAULT_X_RANGE, DEFAULT_Y_RANGE, DEFAULT_Z_RANGE, DEFAULT_VOXEL

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import resnet18, ResNet18_Weights
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ════════════════════════════════════════════════════════════════
#  Geometric baselines (no model needed)
# ════════════════════════════════════════════════════════════════

class GeometricSegHead:
    """Direct CARLA segmentation camera output (already in image space)."""

    def __call__(self, seg_image: np.ndarray) -> np.ndarray:
        """seg_image: (H, W) uint8 BEV class indices from CARLA seg camera."""
        return seg_image.astype(np.uint8)


class GeometricOccHead:
    """Threshold-based 3D occupancy from the lifted hit-count volume."""

    def __init__(self, min_hits: float = 2.0):
        self.min_hits = min_hits

    def __call__(self, occ_count: np.ndarray) -> np.ndarray:
        """
        occ_count: (grid_z, grid_h, grid_w) hit counts
        Returns:   (grid_z, grid_h, grid_w) uint8 binary occupancy
        """
        return (occ_count >= self.min_hits).astype(np.uint8)


# ════════════════════════════════════════════════════════════════
#  Backbone
# ════════════════════════════════════════════════════════════════

if HAS_TORCH:

    class StereoBackbone(nn.Module):
        """Shared ResNet-18 backbone for left and right images."""
        def __init__(self, pretrained: bool = True):
            super().__init__()
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = resnet18(weights=weights)
            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1  # 64ch  /4
            self.layer2 = base.layer2  # 128ch /8
            self.layer3 = base.layer3  # 256ch /16
            self.out_ch = 256
            # save skip connections for FPN decoder
            self._skip_ch = [64, 128]

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list]:
            """
            Returns:
                feat: (B, 256, H/16, W/16) main features
                skips: [(B, 64, H/4, W/4), (B, 128, H/8, W/8)] for decoder
            """
            x = self.stem(x)       # /4, 64ch
            s1 = self.layer1(x)    # /4, 64ch
            s2 = self.layer2(s1)   # /8, 128ch
            feat = self.layer3(s2) # /16, 256ch
            return feat, [s1, s2]

    # ── Depth predictor ──

    class DepthPredictor(nn.Module):
        """Stereo features → depth distribution (per-pixel, feature resolution)."""
        def __init__(self, in_channels: int, depth_bins: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, depth_bins, 1),
            )

        def forward(self, feat: torch.Tensor) -> torch.Tensor:
            return self.net(feat)  # (B, D, Hf, Wf)

    # ── Image-space segmentation decoder (FPN-style) ──

    class SegDecoder(nn.Module):
        """
        FPN-style decoder: backbone features → per-pixel segmentation.

        Takes /16 features + skip connections, upsamples to full resolution.
        Output: (B, num_classes, H, W) logits at image resolution.
        """
        def __init__(self, backbone_ch: int = 256, skip_chs: list = [64, 128], num_classes: int = 10):
            super().__init__()
            # lateral convs to match channels
            self.lateral2 = nn.Conv2d(skip_chs[1], 128, 1)
            self.lateral1 = nn.Conv2d(skip_chs[0], 64, 1)

            # top-down convs
            self.td_conv2 = nn.Sequential(
                nn.Conv2d(128 + backbone_ch, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            )
            self.td_conv1 = nn.Sequential(
                nn.Conv2d(128 + 64, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            )

            # final classifier (upsample to /4 then to full res)
            self.classifier = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, num_classes, 1),
            )

        def forward(
            self,
            feat: torch.Tensor,
            skips: list,
            target_h: int,
            target_w: int,
        ) -> torch.Tensor:
            """
            Args:
                feat: (B, 256, H/16, W/16)
                skips: [(B, 64, H/4, W/4), (B, 128, H/8, W/8)]
                target_h, target_w: full image resolution

            Returns:
                seg_logits: (B, num_classes, H, W)
            """
            s1, s2 = skips

            # top-down: /16 → /8
            up2 = F.interpolate(feat, size=s2.shape[2:], mode='bilinear', align_corners=False)
            lat2 = self.lateral2(s2)
            x = self.td_conv2(torch.cat([up2, lat2], dim=1))

            # /8 → /4
            up1 = F.interpolate(x, size=s1.shape[2:], mode='bilinear', align_corners=False)
            lat1 = self.lateral1(s1)
            x = self.td_conv1(torch.cat([up1, lat1], dim=1))

            # /4 → full resolution
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            return self.classifier(x)

    # ── Stereo correlation ──

    def stereo_correlation(feat_left, feat_right, max_disp=48):
        B, C, H, W = feat_left.shape
        costs = []
        for d in range(max_disp):
            if d == 0:
                cost = (feat_left * feat_right).sum(dim=1, keepdim=True)
            else:
                shifted = F.pad(feat_right[:, :, :, d:], (d, 0, 0, 0))
                cost = (feat_left * shifted).sum(dim=1, keepdim=True)
            costs.append(cost)
        return torch.cat(costs, dim=1) / C

    # ── LSS Lift-Splat-Shoot ──

    class LSSLifter(nn.Module):
        """Lift 2D features to BEV using predicted depth distribution."""
        def __init__(self, feat_channels, depth_bins=64, image_h=540, image_w=960,
                     bev_x_range=DEFAULT_X_RANGE, bev_y_range=DEFAULT_Y_RANGE,
                     bev_z_range=DEFAULT_Z_RANGE, bev_voxel=DEFAULT_VOXEL, max_depth=80.0):
            super().__init__()
            self.feat_channels = feat_channels
            self.depth_bins = depth_bins
            self.bev_x_range = bev_x_range
            self.bev_y_range = bev_y_range
            self.bev_z_range = bev_z_range
            self.bev_voxel = bev_voxel
            self.bev_w = int((bev_x_range[1] - bev_x_range[0]) / bev_voxel)
            self.bev_h = int((bev_y_range[1] - bev_y_range[0]) / bev_voxel)
            self.bev_z = int((bev_z_range[1] - bev_z_range[0]) / bev_voxel)
            self._img_h = image_h
            self._img_w = image_w
            bins = torch.linspace(1.0, min(max_depth, 80.0), depth_bins)
            self.register_buffer("depth_bins_center", bins)

        def forward(self, img_feat, depth_logits, K, cam_ext):
            B, C, Hf, Wf = img_feat.shape
            D = self.depth_bins
            device = img_feat.device
            out_dtype = img_feat.dtype
            # Geometry + scatter in fp32 even under AMP (stable voxel pooling).
            img_feat = img_feat.float()
            depth_logits = depth_logits.float()
            K = K.float()
            cam_ext = cam_ext.float()

            depth_prob = F.softmax(depth_logits, dim=1)
            img_feat_exp = img_feat.unsqueeze(2)
            depth_prob_exp = depth_prob.unsqueeze(1)
            frustum_feat = img_feat_exp * depth_prob_exp  # (B, C, D, Hf, Wf)

            fu = (torch.arange(Wf, device=device, dtype=torch.float32) + 0.5) * (self._img_w / Wf)
            fv = (torch.arange(Hf, device=device, dtype=torch.float32) + 0.5) * (self._img_h / Hf)
            grid_v, grid_u = torch.meshgrid(fv, fu, indexing='ij')

            depth_centers = self.depth_bins_center.view(D, 1, 1)
            fx = K[:, 0, 0].view(B, 1, 1, 1)
            fy = K[:, 1, 1].view(B, 1, 1, 1)
            cx = K[:, 0, 2].view(B, 1, 1, 1)
            cy = K[:, 1, 2].view(B, 1, 1, 1)

            Z = depth_centers.unsqueeze(0)
            grid_u4 = grid_u.unsqueeze(0).unsqueeze(0)
            grid_v4 = grid_v.unsqueeze(0).unsqueeze(0)
            X = (grid_u4 - cx) * Z / fx
            Y = (grid_v4 - cy) * Z / fy
            Z = Z.expand_as(X)

            # cam → ego via 4×4 extrinsic (includes camera pitch)
            pts_cam = torch.stack([X, Y, Z], dim=1).reshape(B, 3, -1)
            ego = torch.bmm(cam_ext[:, :3, :3], pts_cam) + cam_ext[:, :3, 3].unsqueeze(-1)
            ego = ego.reshape(B, 3, D, Hf, Wf)
            # Occupancy origin = vehicle center: default voxel (50, 50, 5).
            # Camera is at ~+1.5 m X in this same grid (xi ≈ 57).
            ego_x, ego_y, ego_z = ego[:, 0], ego[:, 1], ego[:, 2]

            gi = ((ego_x - self.bev_x_range[0]) / self.bev_voxel).long()
            gj = ((ego_y - self.bev_y_range[0]) / self.bev_voxel).long()
            gk = ((ego_z - self.bev_z_range[0]) / self.bev_voxel).long()

            valid = (
                (gi >= 0) & (gi < self.bev_w) &
                (gj >= 0) & (gj < self.bev_h) &
                (gk >= 0) & (gk < self.bev_z)
            )

            # Vectorized splat: avoid per-sample boolean indexing and
            # `if vb.sum() == 0`, both of which sync the GPU back to the CPU.
            n = D * Hf * Wf
            bev_n = self.bev_h * self.bev_w
            feat_flat = frustum_feat.reshape(B, C, n) * valid.reshape(B, 1, n).to(frustum_feat.dtype)
            lin_idx = (gj.reshape(B, n) * self.bev_w + gi.reshape(B, n)).clamp(0, bev_n - 1)
            bev_flat = frustum_feat.new_zeros(B, C, bev_n)
            bev_flat.scatter_add_(2, lin_idx.unsqueeze(1).expand(B, C, n), feat_flat)
            return bev_flat.view(B, C, self.bev_h, self.bev_w).to(out_dtype)

    # ── BEV Encoder ──

    class BEVEncoder(nn.Module):
        def __init__(self, in_ch, out_ch=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.net(x)

    # ── Query Heads ──

    class OccQueryHead(nn.Module):
        """BEV features → per-voxel occupancy logits (B, Z, H, W)."""
        def __init__(self, in_ch, grid_z: int):
            super().__init__()
            self.head = nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, grid_z, 1),
            )
        def forward(self, x): return self.head(x)

    class BevSegQueryHead(nn.Module):
        """BEV features → per-voxel class logits (B, num_classes, Z, H, W)."""
        def __init__(self, in_ch, num_classes, grid_z: int):
            super().__init__()
            self.num_classes = num_classes
            self.grid_z = grid_z
            self.head = nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes * grid_z, 1),
            )
        def forward(self, x):
            y = self.head(x)
            b, _, h, w = y.shape
            return y.view(b, self.num_classes, self.grid_z, h, w)

    class ControlQueryHead(nn.Module):
        """Query 3D BEV at ego and the target pose, then emit CARLA control.

        BEV tensor layout is (B, C, H=y, W=x). Occupancy is (B, Z, H, W).
        ``target`` is (B, 4): ego-FLU x, y, yaw (rad), speed (m/s).
        Output is (B, 3): throttle, brake in [0, 1], steer in [-1, 1].
        """

        def __init__(self, in_ch, grid_z: int, x_range, y_range, v_scale: float = 15.0):
            super().__init__()
            self.v_scale = float(v_scale)
            self.register_buffer(
                "x_range", torch.tensor([float(x_range[0]), float(x_range[1])]),
            )
            self.register_buffer(
                "y_range", torch.tensor([float(y_range[0]), float(y_range[1])]),
            )
            occ_dim = 16
            self.occ_proj = nn.Sequential(
                nn.Linear(grid_z, occ_dim),
                nn.ReLU(inplace=True),
            )
            pose_dim = 5
            hid = 256
            mlp_in = in_ch * 3 + occ_dim * 2 + pose_dim
            self.feat_dim = mlp_in
            self.mlp = nn.Sequential(
                nn.Linear(mlp_in, hid),
                nn.ReLU(inplace=True),
                nn.Linear(hid, hid),
                nn.ReLU(inplace=True),
                nn.Linear(hid, 3),
            )
            self.critic = nn.Sequential(
                nn.Linear(mlp_in, hid),
                nn.ReLU(inplace=True),
                nn.Linear(hid, 1),
            )
            self.log_std = nn.Parameter(torch.zeros(3))

        def _xy_grid(self, xy: torch.Tensor) -> torch.Tensor:
            x0, x1 = self.x_range[0], self.x_range[1]
            y0, y1 = self.y_range[0], self.y_range[1]
            gx = 2.0 * (xy[:, 0] - x0) / (x1 - x0).clamp_min(1e-6) - 1.0
            gy = 2.0 * (xy[:, 1] - y0) / (y1 - y0).clamp_min(1e-6) - 1.0
            return torch.stack([gx, gy], dim=-1).view(-1, 1, 1, 2)

        def _sample(self, feat: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
            grid = self._xy_grid(xy).to(dtype=feat.dtype)
            return F.grid_sample(
                feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False,
            ).squeeze(-1).squeeze(-1)

        def _sample_many(self, feat: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
            """xy (B, P, 2) → (B, P, C)."""
            x0, x1 = self.x_range[0], self.x_range[1]
            y0, y1 = self.y_range[0], self.y_range[1]
            gx = 2.0 * (xy[..., 0] - x0) / (x1 - x0).clamp_min(1e-6) - 1.0
            gy = 2.0 * (xy[..., 1] - y0) / (y1 - y0).clamp_min(1e-6) - 1.0
            grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)
            out = F.grid_sample(
                feat, grid.to(dtype=feat.dtype),
                mode="bilinear", padding_mode="zeros", align_corners=False,
            )
            return out.squeeze(-1).permute(0, 2, 1)

        def encode(self, bev_feat, occ_logits, target):
            """BEV queries + target pose → control feature (B, feat_dim)."""
            self._occ_logits = occ_logits
            B = bev_feat.shape[0]
            xy_ego = bev_feat.new_zeros(B, 2)
            xy_tgt = target[:, :2]
            gap = F.adaptive_avg_pool2d(bev_feat, 1).flatten(1)
            ego_f = self._sample(bev_feat, xy_ego)
            tgt_f = self._sample(bev_feat, xy_tgt)
            ego_o = self.occ_proj(self._sample(occ_logits, xy_ego))
            tgt_o = self.occ_proj(self._sample(occ_logits, xy_tgt))
            xs = target.new_tensor([2.0, 4.0, 6.0, 8.0])
            ys = target.new_tensor([-1.5, 0.0, 1.5])
            xx, yy = torch.meshgrid(xs, ys, indexing="ij")
            pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
            xy_front = pts.unsqueeze(0).expand(B, -1, -1)
            front_o = self.occ_proj(self._sample_many(occ_logits, xy_front)).max(dim=1).values
            ego_o = 0.5 * ego_o + 0.5 * front_o
            x0, x1 = self.x_range[0], self.x_range[1]
            y0, y1 = self.y_range[0], self.y_range[1]
            xn = 2.0 * (target[:, 0] - x0) / (x1 - x0).clamp_min(1e-6) - 1.0
            yn = 2.0 * (target[:, 1] - y0) / (y1 - y0).clamp_min(1e-6) - 1.0
            yaw = target[:, 2]
            vn = target[:, 3] / max(self.v_scale, 1e-3)
            pose = torch.stack([xn, yn, torch.cos(yaw), torch.sin(yaw), vn], dim=-1)
            return torch.cat([gap, ego_f, tgt_f, ego_o, tgt_o, pose], dim=-1)

        def _front_lanes(self, occ_logits):
            """Occupancy in a 2–6 m, 3-lane strip, ignoring near-ground voxels."""
            B, Z = occ_logits.shape[0], occ_logits.shape[1]
            occ_hi = occ_logits[:, max(Z // 2, 1):]
            xs = occ_hi.new_tensor([2.0, 4.0, 6.0])
            ys = occ_hi.new_tensor([-1.6, 0.0, 1.6])
            xx, yy = torch.meshgrid(xs, ys, indexing="ij")
            pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
            xy = pts.unsqueeze(0).expand(B, -1, -1)
            p = torch.sigmoid(self._sample_many(occ_hi, xy)).amax(dim=-1).view(B, 3, 3)
            center = p[:, :, 1].amax(dim=1)
            left = p[:, :, 2].mean(dim=1)
            right = p[:, :, 0].mean(dim=1)
            hazard = ((center - 0.75) / 0.25).clamp(0.0, 1.0)
            steer_bias = torch.tanh(left - right)
            return hazard, steer_bias

        def apply_occ_safety(self, control, occ_logits):
            """Brake / ease throttle if the front strip is occupied; nudge steer to free side."""
            if occ_logits is None:
                return control
            hazard, _steer_bias = self._front_lanes(occ_logits)
            h = hazard.unsqueeze(-1)
            thr = control[:, 0:1] * (1.0 - 0.85 * h)
            brk = torch.maximum(control[:, 1:2], 0.75 * h)
            st = control[:, 2:3]
            go = thr >= brk
            thr = torch.where(go, thr, torch.zeros_like(thr))
            brk = torch.where(go, torch.zeros_like(brk), brk)
            return torch.cat([thr, brk, st], dim=-1)

        def forward(self, bev_feat, occ_logits, target):
            feat = self.encode(bev_feat, occ_logits, target)
            return self.apply_occ_safety(squash_control(self.mlp(feat)), occ_logits)

        def act_from_feat(self, feat, deterministic: bool = False, occ_logits=None):
            """Sample CARLA control from a cached encode() vector."""
            from torch.distributions import Normal
            mu = self.mlp(feat)
            value = self.critic(feat).squeeze(-1)
            std = self.log_std.exp().clamp(1e-4, 2.0)
            dist = Normal(mu, std)
            z = mu if deterministic else dist.sample()
            control = squash_control(z)
            occ = occ_logits if occ_logits is not None else getattr(self, "_occ_logits", None)
            control = self.apply_occ_safety(control, occ)
            log_prob = dist.log_prob(z).sum(dim=-1) - squash_log_absdet(z)
            entropy = dist.entropy().sum(dim=-1)
            return control, log_prob, value, z, entropy

        def evaluate_z(self, feat, z):
            from torch.distributions import Normal
            mu = self.mlp(feat)
            value = self.critic(feat).squeeze(-1)
            std = self.log_std.exp().clamp(1e-4, 2.0)
            dist = Normal(mu, std)
            log_prob = dist.log_prob(z).sum(dim=-1) - squash_log_absdet(z)
            entropy = dist.entropy().sum(dim=-1)
            return log_prob, value, entropy

    def squash_control(raw: torch.Tensor) -> torch.Tensor:
        throttle = torch.sigmoid(raw[:, 0:1])
        brake = torch.sigmoid(raw[:, 1:2])
        steer = torch.tanh(raw[:, 2:3])
        # CARLA brake overrides throttle; keep only the stronger pedal.
        go = throttle >= brake
        throttle = torch.where(go, throttle, torch.zeros_like(throttle))
        brake = torch.where(go, torch.zeros_like(brake), brake)
        return torch.cat([throttle, brake, steer], dim=-1)

    def squash_log_absdet(z: torch.Tensor) -> torch.Tensor:
        s0 = torch.sigmoid(z[:, 0])
        s1 = torch.sigmoid(z[:, 1])
        t2 = torch.tanh(z[:, 2])
        return (
            torch.log(s0 * (1.0 - s0) + 1e-6)
            + torch.log(s1 * (1.0 - s1) + 1e-6)
            + torch.log(1.0 - t2 * t2 + 1e-6)
        )


    # ── Full Model ──

    class StereoBEVModel(nn.Module):
        """
        Stereo vision-action model: RGB + K → depth / seg / 3D BEV, plus control.

        Input:  left_rgb (B, 3, H, W), right_rgb (B, 3, H, W), K (B, 3, 3),
                optional target (B, 4) = (x, y, yaw, speed) in ego FLU
        Output:
            seg_logits:     (B, num_classes, H, W) — image-space segmentation
            occ_logits:     (B, grid_z, bev_h, bev_w) — 3D occupancy
            bev_seg_logits: (B, num_classes, Z, bev_h, bev_w) — per-voxel class
            depth_logits:   (B, D, Hf, Wf) — depth distribution
            control:        (B, 3) throttle, brake, steer
        """
        def __init__(self, num_classes=10, feat_channels=64, depth_bins=64,
                     image_h=540, image_w=960,
                     bev_x_range=DEFAULT_X_RANGE, bev_y_range=DEFAULT_Y_RANGE,
                     bev_z_range=DEFAULT_Z_RANGE, bev_voxel=DEFAULT_VOXEL, max_depth=80.0,
                     pretrained_backbone=True, cam_extrinsic=None, pitch_deg=None):
            super().__init__()
            from .calibration import ego_from_camera, DEFAULT_PITCH_DEG

            if pitch_deg is None:
                pitch_deg = DEFAULT_PITCH_DEG
            if cam_extrinsic is None:
                cam_extrinsic = ego_from_camera(pitch_deg=pitch_deg)

            self.backbone = StereoBackbone(pretrained=pretrained_backbone)
            backbone_ch = self.backbone.out_ch  # 256

            self.depth_predictor = DepthPredictor(backbone_ch * 2, depth_bins)

            self.seg_decoder = SegDecoder(backbone_ch, self.backbone._skip_ch, num_classes)

            self.lifter = LSSLifter(
                feat_channels=backbone_ch, depth_bins=depth_bins,
                image_h=image_h, image_w=image_w,
                bev_x_range=bev_x_range, bev_y_range=bev_y_range,
                bev_z_range=bev_z_range, bev_voxel=bev_voxel, max_depth=max_depth,
            )
            self.bev_encoder = BEVEncoder(backbone_ch, feat_channels)
            self.grid_z = self.lifter.bev_z
            self.occ_head = OccQueryHead(feat_channels, self.grid_z)
            self.bev_seg_head = BevSegQueryHead(feat_channels, num_classes, self.grid_z)
            self.control_head = ControlQueryHead(
                feat_channels, self.grid_z,
                x_range=bev_x_range, y_range=bev_y_range,
            )

            self._image_h = image_h
            self._image_w = image_w
            self._num_classes = num_classes
            self.register_buffer("cam_extrinsic", torch.from_numpy(np.asarray(cam_extrinsic, dtype=np.float32)))

        def _batch_cam_ext(self, cam_ext, batch, device):
            if cam_ext is None:
                cam_ext = self.cam_extrinsic
            if not torch.is_tensor(cam_ext):
                cam_ext = torch.from_numpy(np.asarray(cam_ext, dtype=np.float32))
            cam_ext = cam_ext.to(device=device, dtype=torch.float32)
            if cam_ext.dim() == 2:
                cam_ext = cam_ext.unsqueeze(0).expand(batch, -1, -1)
            return cam_ext

        def _batch_target(self, target, batch, device, dtype):
            if target is None:
                return None
            if not torch.is_tensor(target):
                target = torch.as_tensor(target, dtype=torch.float32)
            target = target.to(device=device, dtype=dtype)
            if target.dim() == 1:
                target = target.unsqueeze(0)
            if target.shape[0] == 1 and batch > 1:
                target = target.expand(batch, -1)
            if target.shape[-1] < 4:
                pad = target.new_zeros(target.shape[0], 4 - target.shape[-1])
                target = torch.cat([target, pad], dim=-1)
            return target[:, :4]

        def encode_bev(self, left_rgb, right_rgb, K, cam_ext=None):
            """Shared stereo → BEV encode used by perception heads and RL."""
            B = left_rgb.shape[0]
            feat_l, skips = self.backbone(left_rgb)
            feat_r, _ = self.backbone(right_rgb)
            stereo_feat = torch.cat([feat_l, feat_r], dim=1)
            depth_logits = self.depth_predictor(stereo_feat)
            seg_logits = self.seg_decoder(feat_l, skips, self._image_h, self._image_w)
            cam_ext = self._batch_cam_ext(cam_ext, B, left_rgb.device)
            bev_feat = self.lifter(feat_l, depth_logits, K, cam_ext)
            bev_feat = self.bev_encoder(bev_feat)
            occ_logits = self.occ_head(bev_feat)
            bev_seg_logits = self.bev_seg_head(bev_feat)
            return bev_feat, occ_logits, bev_seg_logits, depth_logits, seg_logits

        def freeze_perception(self):
            """Train only the control policy / critic / occ query (PPO)."""
            for name, p in self.named_parameters():
                p.requires_grad = (
                    name.startswith("control_head.mlp")
                    or name.startswith("control_head.critic")
                    or name.startswith("control_head.log_std")
                    or name.startswith("control_head.occ_proj")
                )

        def forward(self, left_rgb, right_rgb, K, cam_ext=None, target=None):
            """
            Returns:
                seg_logits:     (B, num_classes, H, W) image-space
                occ_logits:     (B, grid_z, bev_h, bev_w) 3D occupancy
                bev_seg_logits: (B, num_classes, Z, bev_h, bev_w) per-voxel class
                depth_logits:   (B, D, Hf, Wf)
                control:        (B, 3) throttle, brake, steer
            """
            bev_feat, occ_logits, bev_seg_logits, depth_logits, seg_logits = self.encode_bev(
                left_rgb, right_rgb, K, cam_ext=cam_ext,
            )
            B = left_rgb.shape[0]
            tgt = self._batch_target(target, B, left_rgb.device, bev_feat.dtype)
            if tgt is None:
                control = bev_feat.new_zeros(B, 3)
            else:
                control = self.control_head(bev_feat, occ_logits, tgt)
            return seg_logits, occ_logits, bev_seg_logits, depth_logits, control

        def infer(self, left_rgb, right_rgb, K, device=None, cam_ext=None,
                  target=None, speed=0.0):
            """
            Inference from numpy arrays.

            ``target`` is ego-FLU (x, y, yaw[, speed]). ``speed`` fills the
            last slot when ``target`` is length 3.

            Returns:
                seg_classes:     (H, W) uint8 image-space segmentation
                occ_map:         (grid_z, bev_h, bev_w) uint8 binary 3D occupancy
                bev_seg_classes: (Z, bev_h, bev_w) uint8 per-voxel class
                depth_map:       (H, W) float32 meters
                control:         (3,) float32 throttle, brake, steer
            """
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

            def to_tensor(img):
                rgb = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
                t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                return (t.to(device) - mean) / std

            left_t = to_tensor(left_rgb)
            right_t = to_tensor(right_rgb)
            K_t = torch.from_numpy(np.asarray(K)).float().unsqueeze(0).to(device)
            tgt = None
            if target is not None:
                t = np.asarray(target, dtype=np.float32).reshape(-1)
                sp = float(t[3]) if t.size >= 4 else float(speed)
                tgt = torch.tensor([[float(t[0]), float(t[1]), float(t[2]), sp]],
                                   device=device, dtype=torch.float32)

            self.eval()
            with torch.no_grad():
                seg_logits, occ_logits, bev_seg_logits, depth_logits, control = self(
                    left_t, right_t, K_t, cam_ext=cam_ext, target=tgt,
                )

            seg_classes = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            occ_map = (torch.sigmoid(occ_logits) > 0.5).squeeze(0).cpu().numpy().astype(np.uint8)
            bev_seg_classes = bev_seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            control_np = control.squeeze(0).cpu().numpy().astype(np.float32)

            D = depth_logits.shape[1]
            depth_bins = torch.linspace(1.0, 80.0, D).to(device)
            depth_prob = torch.softmax(depth_logits, dim=1)
            expected = (depth_prob * depth_bins.view(1, -1, 1, 1)).sum(dim=1)
            depth_map = F.interpolate(
                expected.unsqueeze(1), size=(self._image_h, self._image_w),
                mode="bilinear", align_corners=False,
            ).squeeze().cpu().numpy()

            return seg_classes, occ_map, bev_seg_classes, depth_map, control_np


# ════════════════════════════════════════════════════════════════
#  Loss
# ════════════════════════════════════════════════════════════════

if HAS_TORCH:

    def stereo_bev_loss(seg_logits, occ_logits, bev_seg_logits, seg_gt, occ_gt, occupancy_seg_gt,
                        seg_weight=1.0, occ_weight=1.0, bev_seg_weight=1.0, occ_pos_weight=5.0,
                        control_pred=None, control_gt=None, has_control=None,
                        control_weight=1.0):
        """
        Combined loss for image-space segmentation + 3D occupancy + BEV semantics
        + optional imitation control (throttle, brake, steer).

        Args:
            seg_logits:     (B, C, H, W) image-space logits
            occ_logits:     (B, Z, bev_h, bev_w) 3D occupancy logits
            bev_seg_logits: (B, C, Z, bev_h, bev_w) per-voxel class logits
            seg_gt:         (B, H, W) long — image-space class indices
            occ_gt:         (B, Z, bev_h, bev_w) float — 3D occupancy
            occupancy_seg_gt: (B, C, Z, bev_h, bev_w) class hits, or (B, Z, H, W) indices
            control_pred:   (B, 3) optional predicted CARLA control
            control_gt:     (B, 3) optional expert CARLA control
            has_control:    (B,) mask; 0 skips imitation on that sample
        """
        seg_loss = F.cross_entropy(seg_logits, seg_gt)
        pos_w = occ_logits.new_tensor(occ_pos_weight)
        occ_loss = F.binary_cross_entropy_with_logits(
            occ_logits, occ_gt.float(), pos_weight=pos_w,
        )
        if occupancy_seg_gt.dim() == bev_seg_logits.dim():
            seg_idx = occupancy_seg_gt.argmax(dim=1)
        else:
            seg_idx = occupancy_seg_gt
        bev_seg_loss = F.cross_entropy(bev_seg_logits, seg_idx.long())
        total = seg_weight * seg_loss + occ_weight * occ_loss + bev_seg_weight * bev_seg_loss
        ctrl_loss = occ_logits.new_zeros(())
        if control_pred is not None and control_gt is not None:
            per = F.smooth_l1_loss(control_pred, control_gt.float(), reduction="none").mean(dim=1)
            if has_control is None:
                ctrl_loss = per.mean()
                total = total + control_weight * ctrl_loss
            else:
                w = has_control.reshape(-1).to(dtype=per.dtype)
                wsum = w.sum()
                if float(wsum.detach()) > 0.0:
                    ctrl_loss = (per * w).sum() / wsum
                    total = total + control_weight * ctrl_loss
        return {
            "loss": total,
            "seg_loss": seg_loss,
            "occ_loss": occ_loss,
            "bev_seg_loss": bev_seg_loss,
            "control_loss": ctrl_loss,
        }


    def occupancy_iou_counts(occ_logits, occ_gt, threshold: float = 0.5):
        """
        Occupancy IoU counts for a batch.

        occ_logits: (B, Z, H, W) or (B, H, W)
        occ_gt:     same shape, {0,1}

        Returns (inter_3d, union_3d, inter_bev, union_bev) as scalar long tensors.
        BEV IoU collapses occupied voxels over Z (max-over-height).
        """
        pred = occ_logits.detach().float().sigmoid() >= threshold
        gt = occ_gt.detach() > 0.5
        if pred.shape != gt.shape:
            raise ValueError(f"occ pred/gt shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")

        inter = (pred & gt).sum()
        union = (pred | gt).sum()
        if pred.dim() == 4:
            pred_bev = pred.any(dim=1)
            gt_bev = gt.any(dim=1)
            inter_bev = (pred_bev & gt_bev).sum()
            union_bev = (pred_bev | gt_bev).sum()
        else:
            inter_bev, union_bev = inter, union
        return inter, union, inter_bev, union_bev


    def occupancy_iou_from_counts(inter, union) -> float:
        """IoU from accumulated intersection/union. Both-empty → 1.0."""
        inter = float(inter)
        union = float(union)
        if union <= 0.0:
            return 1.0
        return inter / union
