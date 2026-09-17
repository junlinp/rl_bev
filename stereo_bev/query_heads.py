"""
Stereo-to-BEV perception model.

Architecture:
  1. Shared backbone extracts features from left + right RGB
  2. Depth predictor: stereo features → per-pixel depth distribution
  3. Seg head: left features → per-pixel semantic segmentation (IMAGE SPACE)
  4. LSS lifter: lift features using predicted depth + seg → BEV grid
  5. Occ head: BEV features → per-cell occupancy

Key: segmentation is in image space (camera view), NOT BEV projection.
Ground truth seg comes from CARLA's semantic segmentation camera directly.
Occupancy ground truth is a 3D voxel volume lifted from depth+seg.
"""

import numpy as np

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
                     bev_x_range=(0.0, 20.0), bev_y_range=(-10.0, 10.0),
                     bev_z_range=(-1.0, 3.0), bev_voxel=0.2, max_depth=80.0):
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
            # Occupancy origin = vehicle center (same as geometric lift).
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
        """BEV semantics: (B, C, H, W) → (B, num_classes, H, W) logits (per-cell class)."""
        def __init__(self, in_ch, num_classes):
            super().__init__()
            self.head = nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes, 1),
            )
        def forward(self, x): return self.head(x)

    # ── Full Model ──

    class StereoBEVModel(nn.Module):
        """
        Stereo perception model with image-space segmentation.

        Input:  left_rgb (B, 3, H, W), right_rgb (B, 3, H, W), K (B, 3, 3)
        Output:
            seg_logits:     (B, num_classes, H, W) — image-space segmentation
            occ_logits:     (B, grid_z, bev_h, bev_w) — 3D occupancy
            bev_seg_logits: (B, num_classes, bev_h, bev_w) — per-cell BEV semantic class
            depth_logits:   (B, D, Hf, Wf) — depth distribution
        """
        def __init__(self, num_classes=10, feat_channels=64, depth_bins=64,
                     image_h=540, image_w=960,
                     bev_x_range=(0.0, 20.0), bev_y_range=(-10.0, 10.0),
                     bev_z_range=(-1.0, 3.0), bev_voxel=0.2, max_depth=80.0,
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
            self.bev_seg_head = BevSegQueryHead(feat_channels, num_classes)

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

        def forward(self, left_rgb, right_rgb, K, cam_ext=None):
            """
            Returns:
                seg_logits:     (B, num_classes, H, W) image-space
                occ_logits:     (B, grid_z, bev_h, bev_w) 3D occupancy
                bev_seg_logits: (B, num_classes, bev_h, bev_w) BEV per-cell class
                depth_logits:   (B, D, Hf, Wf)
            """
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

            return seg_logits, occ_logits, bev_seg_logits, depth_logits

        def infer(self, left_rgb, right_rgb, K, device=None, cam_ext=None):
            """
            Inference from numpy arrays.

            Returns:
                seg_classes:     (H, W) uint8 image-space segmentation
                occ_map:         (grid_z, bev_h, bev_w) uint8 binary 3D occupancy
                bev_seg_classes: (bev_h, bev_w) uint8 per-cell BEV class
                depth_map:       (H, W) float32 meters
            """
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

            def to_tensor(img):
                t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                return ((t - mean) / std).to(device)

            left_t = to_tensor(left_rgb)
            right_t = to_tensor(right_rgb)
            K_t = torch.from_numpy(K).float().unsqueeze(0).to(device)

            self.eval()
            with torch.no_grad():
                seg_logits, occ_logits, bev_seg_logits, depth_logits = self(
                    left_t, right_t, K_t, cam_ext=cam_ext,
                )

            seg_classes = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            occ_map = (torch.sigmoid(occ_logits) > 0.5).squeeze(0).cpu().numpy().astype(np.uint8)
            bev_seg_classes = bev_seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            D = depth_logits.shape[1]
            depth_bins = torch.linspace(1.0, 80.0, D).to(device)
            depth_prob = torch.softmax(depth_logits, dim=1)
            expected = (depth_prob * depth_bins.view(1, -1, 1, 1)).sum(dim=1)
            depth_map = F.interpolate(
                expected.unsqueeze(1), size=(self._image_h, self._image_w),
                mode="bilinear", align_corners=False,
            ).squeeze().cpu().numpy()

            return seg_classes, occ_map, bev_seg_classes, depth_map


# ════════════════════════════════════════════════════════════════
#  Loss
# ════════════════════════════════════════════════════════════════

if HAS_TORCH:

    def stereo_bev_loss(seg_logits, occ_logits, bev_seg_logits, seg_gt, occ_gt, bev_seg_gt,
                        seg_weight=1.0, occ_weight=1.0, bev_seg_weight=1.0, occ_pos_weight=5.0):
        """
        Combined loss for image-space segmentation + 3D occupancy + BEV semantics.

        Args:
            seg_logits:     (B, C, H, W) image-space logits
            occ_logits:     (B, Z, bev_h, bev_w) 3D occupancy logits
            bev_seg_logits: (B, C, bev_h, bev_w) BEV per-cell class logits
            seg_gt:         (B, H, W) long — image-space class indices
            occ_gt:         (B, Z, bev_h, bev_w) float — 3D occupancy
            bev_seg_gt:     (B, bev_h, bev_w) long — BEV per-cell class indices
        """
        seg_loss = F.cross_entropy(seg_logits, seg_gt)
        pos_w = occ_logits.new_tensor(occ_pos_weight)
        occ_loss = F.binary_cross_entropy_with_logits(
            occ_logits, occ_gt.float(), pos_weight=pos_w,
        )
        bev_seg_loss = F.cross_entropy(bev_seg_logits, bev_seg_gt)
        total = seg_weight * seg_loss + occ_weight * occ_loss + bev_seg_weight * bev_seg_loss
        return {
            "loss": total,
            "seg_loss": seg_loss,
            "occ_loss": occ_loss,
            "bev_seg_loss": bev_seg_loss,
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
