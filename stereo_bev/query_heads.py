"""
Stereo-to-BEV perception model.

Input:  left RGB (B, 3, H, W) + right RGB (B, 3, H, W)
Output: seg_logits (B, num_classes, bev_h, bev_w)
        occ_logits (B, 1, bev_h, bev_w)

Architecture (Lift-Splat-Shoot style):
  1. Shared backbone extracts features from left + right
  2. Stereo correlation builds a cost volume → depth distribution
  3. Features are lifted to 3D using predicted depth
  4. 3D points are splatted into BEV grid
  5. BEV encoder refines features
  6. Seg + Occ query heads produce predictions

Depth and segmentation sensors are only used to generate BEV ground truth
for training — not used at inference time.
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
#  Geometric heads (zero-shot baseline, no model needed)
# ════════════════════════════════════════════════════════════════

class GeometricSegHead:
    """Argmax segmentation from the class histogram (GT depth + seg)."""

    def __call__(self, class_histogram: np.ndarray) -> np.ndarray:
        total = class_histogram.sum(axis=0)
        pred = class_histogram.argmax(axis=0)
        pred[total == 0] = 0
        return pred.astype(np.uint8)


class GeometricOccHead:
    """Threshold-based occupancy from hit count (GT depth + seg)."""

    def __init__(self, min_hits: float = 2.0):
        self.min_hits = min_hits

    def __call__(self, occ_count: np.ndarray) -> np.ndarray:
        projected = occ_count.sum(axis=0)
        return (projected >= self.min_hits).astype(np.uint8)


# ════════════════════════════════════════════════════════════════
#  Stereo-to-BEV Model (requires PyTorch)
# ════════════════════════════════════════════════════════════════

if HAS_TORCH:

    # ── Backbone ──

    class StereoBackbone(nn.Module):
        """
        Shared ResNet-18 backbone for left and right images.
        Outputs multi-scale features + skips.
        """
        def __init__(self, pretrained: bool = True):
            super().__init__()
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = resnet18(weights=weights)
            # stem: conv1 + bn + relu + maxpool → /4
            self.stem = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
            )
            # stages → /8, /16, /32
            self.layer1 = base.layer1  # 64ch  → /4  (after maxpool /4)
            self.layer2 = base.layer2  # 128ch → /8
            self.layer3 = base.layer3  # 256ch → /16
            self.out_ch = 256

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:  x (B, 3, H, W)
            Returns: features (B, 256, H/16, W/16)
            """
            x = self.stem(x)       # /4, 64ch
            x = self.layer1(x)     # /4, 64ch
            x = self.layer2(x)     # /8, 128ch
            x = self.layer3(x)     # /16, 256ch
            return x

    # ── Depth prediction ──

    class DepthPredictor(nn.Module):
        """
        Predicts depth distribution from stereo features.

        Takes concatenated left+right features, produces D-bin
        depth probability distribution per pixel.
        """
        def __init__(self, in_channels: int, depth_bins: int = 64):
            super().__init__()
            self.depth_bins = depth_bins
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, depth_bins, 1),
            )

        def forward(self, feat: torch.Tensor) -> torch.Tensor:
            """
            Args:  feat (B, C, Hf, Wf)
            Returns: depth_logits (B, D, Hf, Wf)
            """
            return self.net(feat)

    # ── Stereo correlation ──

    def stereo_correlation(feat_left: torch.Tensor, feat_right: torch.Tensor,
                           max_disp: int = 48) -> torch.Tensor:
        """
        Build a stereo cost volume using correlation.

        Args:
            feat_left:  (B, C, Hf, Wf)
            feat_right: (B, C, Hf, Wf)
            max_disp: maximum disparity in feature pixels

        Returns:
            cost_volume (B, max_disp, Hf, Wf)
        """
        B, C, H, W = feat_left.shape
        costs = []
        for d in range(max_disp):
            if d == 0:
                cost = (feat_left * feat_right).sum(dim=1, keepdim=True)  # (B,1,H,W)
            else:
                shifted = F.pad(feat_right[:, :, :, d:], (d, 0, 0, 0))
                cost = (feat_left * shifted).sum(dim=1, keepdim=True)
            costs.append(cost)
        return torch.cat(costs, dim=1) / C  # (B, max_disp, H, W)

    # ── Lift-Splat-Shoot ──

    class LSSLifter(nn.Module):
        """
        Lift 2D image features to 3D using depth distribution,
        then splat into a BEV grid.

        This is the core of the LSS (Lift-Splat-Shoot) approach:
        1. Outer product: image_feat ⊗ depth_dist → frustum features
        2. Create 3D points in camera frustum
        3. Transform to ego frame
        4. Splat into BEV voxels via scatter
        """
        def __init__(
            self,
            feat_channels: int,
            depth_bins: int = 64,
            image_h: int = 540,
            image_w: int = 960,
            bev_x_range: tuple[float, float] = (-5.0, 5.0),
            bev_y_range: tuple[float, float] = (-5.0, 5.0),
            bev_z_range: tuple[float, float] = (0.0, 5.0),
            bev_voxel: float = 0.1,
            max_depth: float = 80.0,
        ):
            super().__init__()
            self.feat_channels = feat_channels
            self.depth_bins = depth_bins
            self.max_depth = max_depth

            self.bev_x_range = bev_x_range
            self.bev_y_range = bev_y_range
            self.bev_z_range = bev_z_range
            self.bev_voxel = bev_voxel

            self.bev_w = int((bev_x_range[1] - bev_x_range[0]) / bev_voxel)
            self.bev_h = int((bev_y_range[1] - bev_y_range[0]) / bev_voxel)

            # compress lifted features to fewer channels for BEV
            self.compress = nn.Sequential(
                nn.Conv2d(feat_channels * depth_bins, feat_channels, 1, bias=False),
                nn.BatchNorm2d(feat_channels),
                nn.ReLU(inplace=True),
            ) if feat_channels * depth_bins > 512 else None

            # depth bin centers (linear spacing)
            depth_max = min(max_depth, 80.0)
            bins = torch.linspace(1.0, depth_max, depth_bins)
            self.register_buffer("depth_bins_center", bins)

        def forward(
            self,
            img_feat: torch.Tensor,
            depth_logits: torch.Tensor,
            K: torch.Tensor,
        ) -> torch.Tensor:
            """
            Args:
                img_feat: (B, C, Hf, Wf) image features
                depth_logits: (B, D, Hf, Wf) depth distribution
                K: (B, 3, 3) intrinsics (per batch)

            Returns:
                bev_feat: (B, C, bev_h, bev_w) BEV features
            """
            B, C, Hf, Wf = img_feat.shape
            D = self.depth_bins

            # depth probabilities
            depth_prob = F.softmax(depth_logits, dim=1)  # (B, D, Hf, Wf)

            # outer product: lift features into (B, C*D, Hf, Wf)
            # then reduce to (B, C, D, Hf, Wf)
            img_feat_exp = img_feat.unsqueeze(2)          # (B, C, 1, Hf, Wf)
            depth_prob_exp = depth_prob.unsqueeze(1)      # (B, 1, D, Hf, Wf)
            frustum_feat = img_feat_exp * depth_prob_exp   # (B, C, D, Hf, Wf)

            # create 3D points in camera frustum
            device = img_feat.device
            # feature grid pixel coords (center of each feature cell)
            img_h = Hf * 16  # backbone stride
            img_w = Wf * 16
            fu = (torch.arange(Wf, device=device, dtype=torch.float32) + 0.5) * (img_w / Wf)
            fv = (torch.arange(Hf, device=device, dtype=torch.float32) + 0.5) * (img_h / Wf)
            grid_v, grid_u = torch.meshgrid(fv, fu, indexing='ij')  # (Hf, Wf)

            # project each depth bin to 3D in camera frame
            # Z = depth, X = (u - cx)*Z/fx, Y = (v - cy)*Z/fy
            depth_centers = self.depth_bins_center.view(D, 1, 1)  # (D,1,1)

            fx = K[:, 0, 0].view(B, 1, 1, 1)   # (B,1,1,1)
            fy = K[:, 1, 1].view(B, 1, 1, 1)
            cx = K[:, 0, 2].view(B, 1, 1, 1)
            cy = K[:, 1, 2].view(B, 1, 1, 1)

            # 3D points: (B, D, Hf, Wf)
            Z = depth_centers.unsqueeze(0)  # (1, D, 1, 1)
            grid_u4 = grid_u.unsqueeze(0).unsqueeze(0)  # (1, 1, Hf, Wf)
            grid_v4 = grid_v.unsqueeze(0).unsqueeze(0)  # (1, 1, Hf, Wf)
            X = (grid_u4 - cx) * Z / fx
            Y = (grid_v4 - cy) * Z / fy
            Z = Z.expand_as(X)  # (B, D, Hf, Wf) — explicit broadcast

            # camera frame → ego frame (X-right,Y-down,Z-fwd → X-fwd,Y-left,Z-up)
            # ego X = Z_cam, ego Y = -X_cam, ego Z = -Y_cam
            ego_x = Z                    # forward
            ego_y = -X                   # left
            ego_z = -Y                   # up

            # map to BEV grid indices
            gi = ((ego_x - self.bev_x_range[0]) / self.bev_voxel).long()
            gj = ((ego_y - self.bev_y_range[0]) / self.bev_voxel).long()
            gk = ((ego_z - self.bev_z_range[0]) / self.bev_voxel).long()

            # mask in-bounds
            valid = (
                (gi >= 0) & (gi < self.bev_w) &
                (gj >= 0) & (gj < self.bev_h) &
                (gk >= 0) & (gk < int((self.bev_z_range[1]-self.bev_z_range[0])/self.bev_voxel))
            )

            # splat features into BEV grid (collapse Z axis)
            bev_feat = torch.zeros(B, C, self.bev_h, self.bev_w, device=device)

            for b in range(B):
                vb = valid[b]  # (D, Hf, Wf) bool
                if vb.sum() == 0:
                    continue
                feat_vals = frustum_feat[b][:, vb]  # (C, N)
                idx_i = gi[b][vb]  # (N,)
                idx_j = gj[b][vb]

                # flatten to linear index
                lin_idx = idx_j * self.bev_w + idx_i  # (N,)
                # scatter add
                bev_flat = torch.zeros(C, self.bev_h * self.bev_w, device=device)
                bev_flat.scatter_add_(1, lin_idx.unsqueeze(0).expand(C, -1), feat_vals)
                bev_feat[b] = bev_flat.view(C, self.bev_h, self.bev_w)

            return bev_feat

    # ── BEV Encoder ──

    class BEVEncoder(nn.Module):
        """Refine BEV features with conv layers."""
        def __init__(self, in_channels: int, out_channels: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    # ── Query Heads ──

    class SegQueryHead(nn.Module):
        """BEV segmentation: (B, C, H, W) → (B, num_classes, H, W) logits."""
        def __init__(self, in_ch: int, num_classes: int):
            super().__init__()
            self.head = nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(x)

    class OccQueryHead(nn.Module):
        """BEV occupancy: (B, C, H, W) → (B, 1, H, W) logits."""
        def __init__(self, in_ch: int):
            super().__init__()
            self.head = nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(x)

    # ── Full Model ──

    class StereoBEVModel(nn.Module):
        """
        Full stereo-to-BEV perception model.

        Input:  left_rgb (B, 3, H, W), right_rgb (B, 3, H, W)
        Output: seg_logits (B, num_classes, bev_h, bev_w)
                occ_logits (B, 1, bev_h, bev_w)
        """
        def __init__(
            self,
            num_classes: int = 10,
            feat_channels: int = 64,
            depth_bins: int = 64,
            image_h: int = 540,
            image_w: int = 960,
            bev_x_range: tuple = (-5.0, 5.0),
            bev_y_range: tuple = (-5.0, 5.0),
            bev_z_range: tuple = (0.0, 5.0),
            bev_voxel: float = 0.1,
            max_depth: float = 80.0,
            pretrained_backbone: bool = True,
        ):
            super().__init__()
            self.backbone = StereoBackbone(pretrained=pretrained_backbone)
            backbone_ch = self.backbone.out_ch  # 256

            # depth predictor takes concatenated left+right features
            self.depth_predictor = DepthPredictor(backbone_ch * 2, depth_bins)

            # LSS lifter
            self.lifter = LSSLifter(
                feat_channels=backbone_ch,
                depth_bins=depth_bins,
                image_h=image_h,
                image_w=image_w,
                bev_x_range=bev_x_range,
                bev_y_range=bev_y_range,
                bev_z_range=bev_z_range,
                bev_voxel=bev_voxel,
                max_depth=max_depth,
            )

            # BEV encoder
            self.bev_encoder = BEVEncoder(backbone_ch, feat_channels)

            # query heads
            self.seg_head = SegQueryHead(feat_channels, num_classes)
            self.occ_head = OccQueryHead(feat_channels)

            # store intrinsics template
            self._image_h = image_h
            self._image_w = image_w

        def forward(
            self,
            left_rgb: torch.Tensor,
            right_rgb: torch.Tensor,
            K: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Args:
                left_rgb:  (B, 3, H, W)
                right_rgb: (B, 3, H, W)
                K:         (B, 3, 3) camera intrinsics

            Returns:
                seg_logits:   (B, num_classes, bev_h, bev_w)
                occ_logits:   (B, 1, bev_h, bev_w)
                depth_logits: (B, D, Hf, Wf) stereo depth distribution
            """
            # extract features (shared backbone)
            feat_l = self.backbone(left_rgb)    # (B, 256, Hf, Wf)
            feat_r = self.backbone(right_rgb)   # (B, 256, Hf, Wf)

            # predict depth from stereo features
            stereo_feat = torch.cat([feat_l, feat_r], dim=1)  # (B, 512, Hf, Wf)
            depth_logits = self.depth_predictor(stereo_feat)   # (B, D, Hf, Wf)

            # lift left features to BEV using predicted depth
            bev_feat = self.lifter(feat_l, depth_logits, K)    # (B, 256, bev_h, bev_w)

            # refine BEV features
            bev_feat = self.bev_encoder(bev_feat)              # (B, 64, bev_h, bev_w)

            # query heads
            seg_logits = self.seg_head(bev_feat)
            occ_logits = self.occ_head(bev_feat)

            return seg_logits, occ_logits, depth_logits

        def infer(
            self,
            left_rgb: np.ndarray,
            right_rgb: np.ndarray,
            K: np.ndarray,
            device: str = "cpu",
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            Inference from numpy arrays.

            Args:
                left_rgb:  (H, W, 3) uint8 BGR
                right_rgb: (H, W, 3) uint8 BGR
                K:         (3, 3) float64 intrinsics

            Returns:
                seg_classes: (bev_h, bev_w) uint8 class indices
                occ_map:     (bev_h, bev_w) uint8 binary
            """
            # normalize to ImageNet stats
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
                seg_logits, occ_logits, _ = self(left_t, right_t, K_t)

            seg_classes = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            occ_map = (torch.sigmoid(occ_logits).squeeze() > 0.5).cpu().numpy().astype(np.uint8)
            return seg_classes, occ_map


# ════════════════════════════════════════════════════════════════
#  Training utilities
# ════════════════════════════════════════════════════════════════

if HAS_TORCH:

    def stereo_bev_loss(
        seg_logits: torch.Tensor,
        occ_logits: torch.Tensor,
        seg_gt: torch.Tensor,
        occ_gt: torch.Tensor,
        seg_weight: float = 1.0,
        occ_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        Combined loss for BEV segmentation + occupancy.

        Args:
            seg_logits: (B, C, H, W) class logits
            occ_logits: (B, 1, H, W) occupancy logits
            seg_gt:     (B, H, W) long class indices
            occ_gt:     (B, H, W) float binary occupancy

        Returns:
            dict with 'loss', 'seg_loss', 'occ_loss'
        """
        seg_loss = F.cross_entropy(seg_logits, seg_gt)
        occ_loss = F.binary_cross_entropy_with_logits(occ_logits.squeeze(1), occ_gt)
        total = seg_weight * seg_loss + occ_weight * occ_loss
        return {"loss": total, "seg_loss": seg_loss, "occ_loss": occ_loss}

    def prepare_stereo_input(
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        K: np.ndarray,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert numpy stereo pair to normalized tensors.

        Args:
            left_rgb:  (H, W, 3) uint8 BGR
            right_rgb: (H, W, 3) uint8 BGR
            K:         (3, 3) intrinsics

        Returns:
            left_t, right_t: (1, 3, H, W) normalized float
            K_t: (1, 3, 3) float
        """
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        def norm(img):
            rgb = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB
            rgb = (rgb - mean) / std
            return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()

        left_t = norm(left_rgb).to(device)
        right_t = norm(right_rgb).to(device)
        K_t = torch.from_numpy(K).float().unsqueeze(0).to(device)
        return left_t, right_t, K_t
