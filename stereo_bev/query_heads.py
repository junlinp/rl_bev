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
Occupancy ground truth comes from lifting depth+seg into the BEV grid.
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
    """Threshold-based occupancy from depth+seg lifting."""

    def __init__(self, min_hits: float = 2.0):
        self.min_hits = min_hits

    def __call__(self, occ_count: np.ndarray) -> np.ndarray:
        projected = occ_count.sum(axis=0)
        return (projected >= self.min_hits).astype(np.uint8)


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
                     bev_x_range=(-5,5), bev_y_range=(-5,5), bev_z_range=(0,5),
                     bev_voxel=0.1, max_depth=80.0):
            super().__init__()
            self.feat_channels = feat_channels
            self.depth_bins = depth_bins
            self.bev_x_range = bev_x_range
            self.bev_y_range = bev_y_range
            self.bev_z_range = bev_z_range
            self.bev_voxel = bev_voxel
            self.bev_w = int((bev_x_range[1] - bev_x_range[0]) / bev_voxel)
            self.bev_h = int((bev_y_range[1] - bev_y_range[0]) / bev_voxel)
            self._img_h = image_h
            self._img_w = image_w
            bins = torch.linspace(1.0, min(max_depth, 80.0), depth_bins)
            self.register_buffer("depth_bins_center", bins)

        def forward(self, img_feat, depth_logits, K):
            B, C, Hf, Wf = img_feat.shape
            D = self.depth_bins
            device = img_feat.device

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

            # cam→ego: X=fwd, Y=left, Z=up
            ego_x, ego_y, ego_z = Z, -X, -Y

            gi = ((ego_x - self.bev_x_range[0]) / self.bev_voxel).long()
            gj = ((ego_y - self.bev_y_range[0]) / self.bev_voxel).long()
            gk = ((ego_z - self.bev_z_range[0]) / self.bev_voxel).long()
            z_cells = int((self.bev_z_range[1] - self.bev_z_range[0]) / self.bev_voxel)

            valid = (gi >= 0) & (gi < self.bev_w) & (gj >= 0) & (gj < self.bev_h) & (gk >= 0) & (gk < z_cells)

            bev_feat = torch.zeros(B, C, self.bev_h, self.bev_w, device=device)
            for b in range(B):
                vb = valid[b]
                if vb.sum() == 0: continue
                feat_vals = frustum_feat[b][:, vb]
                lin_idx = (gj[b][vb] * self.bev_w + gi[b][vb])
                bev_flat = torch.zeros(C, self.bev_h * self.bev_w, device=device)
                bev_flat.scatter_add_(1, lin_idx.unsqueeze(0).expand(C, -1), feat_vals)
                bev_feat[b] = bev_flat.view(C, self.bev_h, self.bev_w)
            return bev_feat

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
        """BEV occupancy: (B, C, H, W) → (B, 1, H, W) logits."""
        def __init__(self, in_ch):
            super().__init__()
            self.head = nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
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
            occ_logits:     (B, 1, bev_h, bev_w) — BEV occupancy
            bev_seg_logits: (B, num_classes, bev_h, bev_w) — per-cell BEV semantic class
            depth_logits:   (B, D, Hf, Wf) — depth distribution
        """
        def __init__(self, num_classes=10, feat_channels=64, depth_bins=64,
                     image_h=540, image_w=960,
                     bev_x_range=(-5,5), bev_y_range=(-5,5), bev_z_range=(0,5),
                     bev_voxel=0.1, max_depth=80.0, pretrained_backbone=True):
            super().__init__()
            self.backbone = StereoBackbone(pretrained=pretrained_backbone)
            backbone_ch = self.backbone.out_ch  # 256

            self.depth_predictor = DepthPredictor(backbone_ch * 2, depth_bins)

            # IMAGE-SPACE segmentation decoder (FPN)
            self.seg_decoder = SegDecoder(backbone_ch, self.backbone._skip_ch, num_classes)

            self.lifter = LSSLifter(
                feat_channels=backbone_ch, depth_bins=depth_bins,
                image_h=image_h, image_w=image_w,
                bev_x_range=bev_x_range, bev_y_range=bev_y_range,
                bev_z_range=bev_z_range, bev_voxel=bev_voxel, max_depth=max_depth,
            )
            self.bev_encoder = BEVEncoder(backbone_ch, feat_channels)
            self.occ_head = OccQueryHead(feat_channels)
            self.bev_seg_head = BevSegQueryHead(feat_channels, num_classes)

            self._image_h = image_h
            self._image_w = image_w
            self._num_classes = num_classes

        def forward(self, left_rgb, right_rgb, K):
            """
            Returns:
                seg_logits:     (B, num_classes, H, W) image-space
                occ_logits:     (B, 1, bev_h, bev_w) BEV
                bev_seg_logits: (B, num_classes, bev_h, bev_w) BEV per-cell class
                depth_logits:   (B, D, Hf, Wf)
            """
            B = left_rgb.shape[0]
            feat_l, skips = self.backbone(left_rgb)    # (B, 256, H/16, W/16)
            feat_r, _ = self.backbone(right_rgb)

            # depth from stereo
            stereo_feat = torch.cat([feat_l, feat_r], dim=1)
            depth_logits = self.depth_predictor(stereo_feat)

            # IMAGE-SPACE segmentation from left features
            seg_logits = self.seg_decoder(feat_l, skips, self._image_h, self._image_w)

            # lift to BEV using predicted depth
            bev_feat = self.lifter(feat_l, depth_logits, K)
            bev_feat = self.bev_encoder(bev_feat)
            occ_logits = self.occ_head(bev_feat)
            bev_seg_logits = self.bev_seg_head(bev_feat)

            return seg_logits, occ_logits, bev_seg_logits, depth_logits

        def infer(self, left_rgb, right_rgb, K, device="cpu"):
            """
            Inference from numpy arrays.

            Returns:
                seg_classes:     (H, W) uint8 image-space segmentation
                occ_map:         (bev_h, bev_w) uint8 binary
                bev_seg_classes: (bev_h, bev_w) uint8 per-cell BEV class
                depth_map:       (H, W) float32 meters
            """
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
                seg_logits, occ_logits, bev_seg_logits, depth_logits = self(left_t, right_t, K_t)

            seg_classes = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            occ_map = (torch.sigmoid(occ_logits).squeeze() > 0.5).cpu().numpy().astype(np.uint8)
            bev_seg_classes = bev_seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # depth: expected value from distribution
            D = depth_logits.shape[1]
            depth_bins = torch.linspace(1.0, 80.0, D).to(device)
            depth_prob = torch.softmax(depth_logits, dim=1)
            expected = (depth_prob * depth_bins.view(1, -1, 1, 1)).sum(dim=1)
            depth_map = F.interpolate(
                expected.unsqueeze(1), size=(self._image_h, self._image_w),
                mode='bilinear', align_corners=False,
            ).squeeze().cpu().numpy()

            return seg_classes, occ_map, bev_seg_classes, depth_map


# ════════════════════════════════════════════════════════════════
#  Loss
# ════════════════════════════════════════════════════════════════

if HAS_TORCH:

    def stereo_bev_loss(seg_logits, occ_logits, bev_seg_logits, seg_gt, occ_gt, bev_seg_gt,
                        seg_weight=1.0, occ_weight=1.0, bev_seg_weight=1.0):
        """
        Combined loss for image-space segmentation + BEV occupancy + BEV semantics.

        Args:
            seg_logits:     (B, C, H, W) image-space logits
            occ_logits:     (B, 1, bev_h, bev_w) BEV occupancy logits
            bev_seg_logits: (B, C, bev_h, bev_w) BEV per-cell class logits
            seg_gt:         (B, H, W) long — image-space class indices
            occ_gt:         (B, bev_h, bev_w) float — BEV occupancy
            bev_seg_gt:     (B, bev_h, bev_w) long — BEV per-cell class indices
        """
        seg_loss = F.cross_entropy(seg_logits, seg_gt)
        occ_loss = F.binary_cross_entropy_with_logits(occ_logits.squeeze(1), occ_gt)
        bev_seg_loss = F.cross_entropy(bev_seg_logits, bev_seg_gt)
        total = seg_weight * seg_loss + occ_weight * occ_loss + bev_seg_weight * bev_seg_loss
        return {
            "loss": total,
            "seg_loss": seg_loss,
            "occ_loss": occ_loss,
            "bev_seg_loss": bev_seg_loss,
        }
