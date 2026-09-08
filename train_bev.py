"""
Training script for StereoBEVModel with train/val split + TensorBoard visualization.

Usage:
  python train_bev.py --data bev_data --epochs 50 --batch-size 2

TensorBoard:
  tensorboard --logdir runs/bev_train
"""

import os
import argparse
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, os.path.dirname(__file__))

from stereo_bev.segmentation import NUM_BEV_CLASSES, BEV_CLASSES, BEV_COLORS
from stereo_bev.query_heads import StereoBEVModel, stereo_bev_loss


# ════════════════════════════════════════════════════════════════
#  Dataset
# ════════════════════════════════════════════════════════════════

class StereoBEVDataset(Dataset):
    """
    Each .npz: left_rgb, right_rgb, depth_gt, seg_gt, occ_gt, K
    """

    def __init__(self, data_dir: str):
        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".npz")
        ])
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz files in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])

        left = d["left_rgb"]
        right = d["right_rgb"]

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        def norm(img):
            rgb = img[:, :, ::-1].astype(np.float32) / 255.0
            return torch.from_numpy((rgb - mean) / std).permute(2, 0, 1)

        left_t = norm(left)
        right_t = norm(right)
        depth_gt = torch.from_numpy(d["depth_gt"]).float()
        seg_gt = torch.from_numpy(d["seg_gt"]).long()
        occ_gt = torch.from_numpy(d["occ_gt"]).float()
        K = torch.from_numpy(d["K"]).float()

        return left_t, right_t, K, depth_gt, seg_gt, occ_gt


# ════════════════════════════════════════════════════════════════
#  Visualization helpers
# ════════════════════════════════════════════════════════════════

def colorize_bev_seg(seg_classes: np.ndarray) -> np.ndarray:
    """(H, W) class indices → (H, W, 3) RGB image."""
    return BEV_COLORS[seg_classes.clip(0, NUM_BEV_CLASSES - 1)]


def depth_to_heatmap(depth: np.ndarray, max_depth: float = 80.0) -> np.ndarray:
    """(H, W) depth in meters → (H, W, 3) BGR heatmap."""
    norm = np.clip(depth / max_depth, 0, 1)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[depth < 0.1] = 0
    return colored


def occ_to_heatmap(occ: np.ndarray) -> np.ndarray:
    """(H, W) binary occupancy → (H, W, 3) RGB heatmap."""
    img = np.zeros((*occ.shape, 3), dtype=np.uint8)
    img[occ > 0] = [200, 100, 50]  # blue-ish for occupied
    return img


def render_occupancy_3d(
    occ_gt: np.ndarray,
    occ_pred: np.ndarray,
    voxel_size: float = 0.1,
    scale: int = 4,
) -> np.ndarray:
    """
    Render top-down + isometric view of occupancy grids side by side.

    Args:
        occ_gt:   (H, W) binary
        occ_pred: (H, W) binary
        voxel_size: meters per cell
        scale: pixel scale

    Returns:
        (H_out, W_out, 3) RGB image
    """
    H, W = occ_gt.shape
    cell = scale

    # top-down view: GT on left, pred on right
    def draw_grid(occ, color_occupied, color_empty=(30, 30, 30)):
        img = np.full((H * cell, W * cell, 3), color_empty, dtype=np.uint8)
        for y in range(H):
            for x in range(W):
                if occ[y, x] > 0:
                    y0, y1 = y * cell, (y + 1) * cell
                    x0, x1 = x * cell, (x + 1) * cell
                    img[y0:y1, x0:x1] = color_occupied
        return img

    gt_img = draw_grid(occ_gt, (50, 200, 50))     # green = GT occupied
    pred_img = draw_grid(occ_pred, (50, 50, 200))  # red = predicted occupied

    # add labels
    cv2.putText(gt_img, "GT Occ", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(pred_img, "Pred Occ", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # isometric: overlay both with transparency
    overlay = gt_img.copy()
    mask = pred_img[:, :, 2] > 100  # red channel of pred
    overlay[mask] = [50, 200, 200]  # yellow = overlap
    cv2.putText(overlay, "Overlay (green=GT, red=pred, yellow=both)", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return np.concatenate([gt_img, pred_img, overlay], axis=1)


def make_comparison_panel(
    left_rgb: np.ndarray,
    depth_gt: np.ndarray,
    depth_pred_map: np.ndarray,
    seg_gt: np.ndarray,
    seg_pred: np.ndarray,
    occ_gt: np.ndarray,
    occ_pred: np.ndarray,
) -> np.ndarray:
    """
    Build a single comparison image for TensorBoard.

    Layout:
      row 1: left_rgb | depth_gt | depth_pred
      row 2: seg_gt   | seg_pred | occ_3d

    Returns (H, W, 3) uint8 RGB.
    """
    # resize everything to same height
    target_h = 200

    def resize(img, w=None):
        if w is None:
            w = int(target_h * img.shape[1] / max(img.shape[0], 1))
        return cv2.resize(img, (w, target_h), interpolation=cv2.INTER_NEAREST)

    # row 1: RGB + depth
    left_vis = resize(cv2.cvtColor(left_rgb, cv2.COLOR_BGR2RGB))
    depth_gt_vis = resize(depth_to_heatmap(depth_gt))
    depth_gt_vis = cv2.cvtColor(depth_gt_vis, cv2.COLOR_BGR2RGB)
    depth_pred_vis = resize(depth_to_heatmap(depth_pred_map))
    depth_pred_vis = cv2.cvtColor(depth_pred_vis, cv2.COLOR_BGR2RGB)

    # pad to same width
    max_w1 = max(left_vis.shape[1], depth_gt_vis.shape[1], depth_pred_vis.shape[1])
    def pad_to(img, w):
        if img.shape[1] < w:
            pad = np.zeros((img.shape[0], w - img.shape[1], 3), dtype=np.uint8)
            return np.concatenate([img, pad], axis=1)
        return img[:, :w]

    row1 = np.concatenate([
        pad_to(left_vis, max_w1),
        pad_to(depth_gt_vis, max_w1),
        pad_to(depth_pred_vis, max_w1),
    ], axis=1)

    # row 2: segmentation + occupancy
    seg_gt_vis = resize(colorize_bev_seg(seg_gt))
    seg_pred_vis = resize(colorize_bev_seg(seg_pred))
    occ_3d = resize(render_occupancy_3d(occ_gt, occ_pred), w=seg_gt_vis.shape[1] * 2)

    max_w2 = max(seg_gt_vis.shape[1] + seg_pred_vis.shape[1], occ_3d.shape[1])
    row2_left = np.concatenate([
        pad_to(seg_gt_vis, max_w2 // 2),
        pad_to(seg_pred_vis, max_w2 // 2),
    ], axis=1)
    row2 = np.concatenate([
        pad_to(row2_left, max_w2),
        pad_to(occ_3d, max_w2),
    ], axis=1)

    # match widths between rows
    max_w = max(row1.shape[1], row2.shape[1])
    row1 = pad_to(row1, max_w)
    row2 = pad_to(row2, max_w)

    # add labels
    cv2.putText(row1, "Input", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(row1, "GT Depth", (max_w1 + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(row1, "Pred Depth", (max_w1 * 2 + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return np.concatenate([row1, row2], axis=0)


# ════════════════════════════════════════════════════════════════
#  Depth visualization from model
# ════════════════════════════════════════════════════════════════

def depth_logits_to_map(
    depth_logits: torch.Tensor,
    depth_bins: torch.Tensor,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """
    Convert model's depth distribution to a coarse depth map.

    Args:
        depth_logits: (1, D, Hf, Wf)
        depth_bins: (D,) bin centers in meters
        target_h, target_w: output size

    Returns:
        (H, W) float32 depth in meters
    """
    depth_prob = torch.softmax(depth_logits, dim=1)  # (1, D, Hf, Wf)
    # expected depth = sum(prob * bin_center)
    expected = (depth_prob * depth_bins.view(1, -1, 1, 1)).sum(dim=1)  # (1, Hf, Wf)
    # upsample to image resolution
    depth_map = torch.nn.functional.interpolate(
        expected.unsqueeze(1), size=(target_h, target_w), mode="bilinear", align_corners=False,
    ).squeeze().cpu().numpy()
    return depth_map


# ════════════════════════════════════════════════════════════════
#  Training loop
# ════════════════════════════════════════════════════════════════

def evaluate(model, loader, device):
    """Run validation, return avg losses."""
    model.eval()
    total_seg = 0.0
    total_occ = 0.0
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for left_t, right_t, K_t, depth_gt, seg_gt, occ_gt in loader:
            left_t = left_t.to(device)
            right_t = right_t.to(device)
            K_t = K_t.to(device)
            seg_gt = seg_gt.to(device)
            occ_gt = occ_gt.to(device)

            seg_logits, occ_logits, _ = model(left_t, right_t, K_t)
            losses = stereo_bev_loss(seg_logits, occ_logits, seg_gt, occ_gt)

            bs = left_t.size(0)
            total_loss += losses["loss"].item() * bs
            total_seg += losses["seg_loss"].item() * bs
            total_occ += losses["occ_loss"].item() * bs
            n += bs

    return total_loss / n, total_seg / n, total_occ / n


def occ_to_pointcloud(
    occ: np.ndarray,
    seg: np.ndarray,
    voxel_size: float = 0.1,
    x_range: tuple = (-5.0, 5.0),
    y_range: tuple = (-5.0, 5.0),
    z_range: tuple = (0.0, 5.0),
    color: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert BEV occupancy + segmentation to a colored 3D point cloud.

    Args:
        occ: (H, W) binary occupancy
        seg: (H, W) class indices
        voxel_size: meters per cell
        color: (3,) RGB override, or None to use class colors

    Returns:
        vertices: (N, 3) XYZ in meters
        colors: (N, 3) RGB [0,1]
    """
    ys, xs = np.where(occ > 0)
    if len(ys) == 0:
        return np.zeros((1, 3)), np.zeros((1, 3))

    # grid indices → world coords (ego frame)
    x = xs * voxel_size + x_range[0] + voxel_size / 2
    y = ys * voxel_size + y_range[0] + voxel_size / 2

    # stack multiple Z layers to give height to the flat BEV
    z_layers = np.linspace(z_range[0] + 0.3, z_range[1] - 0.3, 5)
    all_pts = []
    all_colors = []
    for z in z_layers:
        pts = np.stack([x, y, np.full_like(x, z)], axis=-1)
        all_pts.append(pts)

        if color is not None:
            all_colors.append(np.tile(color, (len(x), 1)))
        else:
            cls_colors = BEV_COLORS[seg[ys, xs].clip(0, NUM_BEV_CLASSES - 1)].astype(np.float32) / 255.0
            all_colors.append(cls_colors)

    vertices = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    return vertices, colors


def log_3d_occupancy(
    writer: SummaryWriter,
    gt_occ: np.ndarray,
    pred_occ: np.ndarray,
    gt_seg: np.ndarray,
    pred_seg: np.ndarray,
    epoch: int,
    name: str = "occupancy3d",
    voxel_size: float = 0.1,
):
    """
    Log GT and predicted occupancy as 3D point clouds to TensorBoard.

    GT = class-colored, Pred = red-tinted
    """
    gt_verts, gt_colors = occ_to_pointcloud(gt_occ, gt_seg, voxel_size)
    pred_verts, pred_colors = occ_to_pointcloud(
        pred_occ, pred_seg, voxel_size,
        color=np.array([0.9, 0.2, 0.2]),  # red for predicted
    )

    # offset predicted along X so they sit side-by-side
    offset_x = 12.0  # meters
    pred_verts[:, 0] += offset_x

    # combine
    all_verts = np.concatenate([gt_verts, pred_verts], axis=0)
    all_colors = np.concatenate([gt_colors, pred_colors], axis=0)

    # TensorBoard add_mesh expects (1, N, 3) tensors
    vertices_t = torch.from_numpy(all_verts).float().unsqueeze(0)
    colors_t = torch.from_numpy(all_colors).float().unsqueeze(0)

    writer.add_mesh(
        name,
        vertices=vertices_t,
        colors=colors_t,
        global_step=epoch,
    )


def log_visualizations(
    model: StereoBEVModel,
    loader: DataLoader,
    writer: SummaryWriter,
    device: str,
    epoch: int,
    max_samples: int = 4,
    depth_bins: int = 64,
    max_depth: float = 80.0,
    bev_voxel: float = 0.1,
):
    """Log comparison images + 3D occupancy to TensorBoard."""
    model.eval()

    depth_bin_centers = torch.linspace(1.0, min(max_depth, 80.0), depth_bins).to(device)
    count = 0

    with torch.no_grad():
        for left_t, right_t, K_t, depth_gt_b, seg_gt_b, occ_gt_b in loader:
            left_t = left_t.to(device)
            right_t = right_t.to(device)
            K_t = K_t.to(device)

            seg_logits, occ_logits, depth_logits = model(left_t, right_t, K_t)

            B = left_t.size(0)
            for b in range(B):
                if count >= max_samples:
                    return

                # predicted depth map
                pred_depth = depth_logits_to_map(
                    depth_logits[b:b+1], depth_bin_centers,
                    depth_gt_b.shape[1], depth_gt_b.shape[2],
                )

                # predicted seg + occ
                pred_seg = seg_logits[b].argmax(dim=0).cpu().numpy().astype(np.uint8)
                pred_occ = (torch.sigmoid(occ_logits[b]).squeeze() > 0.5).cpu().numpy().astype(np.uint8)

                # GT
                gt_depth = depth_gt_b[b].numpy()
                gt_seg = seg_gt_b[b].numpy().astype(np.uint8)
                gt_occ = occ_gt_b[b].numpy().astype(np.uint8)

                # left RGB (denormalize)
                left_np = left_t[b].cpu().permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                left_np = ((left_np * std + mean) * 255).clip(0, 255).astype(np.uint8)
                left_np = cv2.cvtColor(left_np, cv2.COLOR_RGB2BGR)

                # 2D comparison panel
                panel = make_comparison_panel(
                    left_np, gt_depth, pred_depth,
                    gt_seg, pred_seg, gt_occ, pred_occ,
                )

                writer.add_image(
                    f"val/sample_{count}",
                    panel.transpose(2, 0, 1),
                    global_step=epoch,
                )

                # 3D occupancy visualization
                log_3d_occupancy(
                    writer, gt_occ, pred_occ, gt_seg, pred_seg,
                    epoch, name=f"occupancy3d/sample_{count}",
                    voxel_size=bev_voxel,
                )

                count += 1


def train(
    data_dir: str = "bev_data",
    epochs: int = 50,
    batch_size: int = 2,
    lr: float = 1e-4,
    checkpoint_path: str = "stereo_bev_model.pth",
    log_dir: str = "runs/bev_train",
    image_w: int = 960,
    image_h: int = 540,
    bev_range_xy: float = 5.0,
    bev_z_range: float = 5.0,
    bev_voxel: float = 0.1,
    max_depth: float = 80.0,
    viz_every: int = 5,
):
    """Train StereoBEVModel with train/val split + TensorBoard."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Device: {device}")

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = StereoBEVDataset(train_dir)
    val_dataset = StereoBEVDataset(val_dir)
    print(f"[Train] Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda"),
    )

    model = StereoBEVModel(
        num_classes=NUM_BEV_CLASSES,
        image_h=image_h,
        image_w=image_w,
        bev_x_range=(-bev_range_xy, bev_range_xy),
        bev_y_range=(-bev_range_xy, bev_range_xy),
        bev_z_range=(0.0, bev_z_range),
        bev_voxel=bev_voxel,
        max_depth=max_depth,
        pretrained_backbone=True,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model params: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    writer = SummaryWriter(log_dir)
    print(f"[Train] TensorBoard: tensorboard --logdir {log_dir}")

    print(f"[Train] Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        # ── train ──
        model.train()
        total_loss = 0.0
        total_seg = 0.0
        total_occ = 0.0
        n = 0

        for left_t, right_t, K_t, depth_gt, seg_gt, occ_gt in train_loader:
            left_t = left_t.to(device)
            right_t = right_t.to(device)
            K_t = K_t.to(device)
            seg_gt = seg_gt.to(device)
            occ_gt = occ_gt.to(device)

            seg_logits, occ_logits, _ = model(left_t, right_t, K_t)
            losses = stereo_bev_loss(seg_logits, occ_logits, seg_gt, occ_gt)

            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()

            bs = left_t.size(0)
            total_loss += losses["loss"].item() * bs
            total_seg += losses["seg_loss"].item() * bs
            total_occ += losses["occ_loss"].item() * bs
            n += bs

        scheduler.step()
        train_loss = total_loss / n
        train_seg = total_seg / n
        train_occ = total_occ / n

        # ── validate ──
        val_loss, val_seg, val_occ = evaluate(model, val_loader, device)

        # save latest every epoch
        torch.save(model.state_dict(), checkpoint_path)

        # tensorboard scalar logging
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("seg_loss", {"train": train_seg, "val": val_seg}, epoch)
        writer.add_scalars("occ_loss", {"train": train_occ, "val": val_occ}, epoch)
        writer.add_scalar("lr", current_lr, epoch)

        # tensorboard image logging every N epochs
        if epoch % viz_every == 0 or epoch == epochs - 1:
            log_visualizations(model, val_loader, writer, device, epoch, bev_voxel=bev_voxel)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  epoch {epoch+1:3d}/{epochs}  "
                f"train: loss={train_loss:.4f} seg={train_seg:.4f} occ={train_occ:.4f}  "
                f"val: loss={val_loss:.4f} seg={val_seg:.4f} occ={val_occ:.4f}  "
                f"lr={current_lr:.6f}",
                flush=True,
            )

    writer.close()
    print(f"[Train] Latest model saved → {checkpoint_path}")
    print(f"[Train] TensorBoard: tensorboard --logdir {log_dir}")


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo BEV training")
    parser.add_argument("--data", type=str, default="bev_data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="stereo_bev_model.pth")
    parser.add_argument("--logdir", type=str, default="runs/bev_train")
    parser.add_argument("--viz-every", type=int, default=5, help="Log images every N epochs")
    parser.add_argument("--image-w", type=int, default=960)
    parser.add_argument("--image-h", type=int, default=540)
    parser.add_argument("--bev-range", type=float, default=5.0)
    parser.add_argument("--bev-voxel", type=float, default=0.1)

    args = parser.parse_args()

    train(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_path=args.output,
        log_dir=args.logdir,
        viz_every=args.viz_every,
        image_w=args.image_w,
        image_h=args.image_h,
        bev_range_xy=args.bev_range,
        bev_voxel=args.bev_voxel,
    )
