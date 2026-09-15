"""
Training script for StereoBEVModel with image-space segmentation.

The model predicts:
  - Per-pixel semantic segmentation in IMAGE SPACE (camera view)
  - Per-pixel depth from stereo
  - BEV occupancy grid (lifted from depth + seg)

Usage:
    python train_bev.py --kv-url http://localhost:3000 --epochs 50 --batch-size 2
    python train_bev.py --data bev_data --epochs 50 --batch-size 2

TensorBoard:
  tensorboard --logdir runs/bev_train
"""

import os
import argparse
from contextlib import nullcontext
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, os.path.dirname(__file__))

from stereo_bev.segmentation import NUM_BEV_CLASSES, BEV_CLASSES, BEV_COLORS
from stereo_bev.query_heads import StereoBEVModel, stereo_bev_loss
from minikeyvalue_client import MiniKV


# ════════════════════════════════════════════════════════════════
#  Dataset
# ════════════════════════════════════════════════════════════════

_NPZ_FIELDS = (
    "left_rgb", "right_rgb", "K", "depth_gt", "seg_gt", "occ_gt", "bev_seg_gt",
)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _bgr_to_nchw_uint8(img: np.ndarray) -> torch.Tensor:
    """BGR uint8 (H, W, 3) → RGB uint8 (3, H, W). Normalization runs on GPU."""
    rgb = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
    return torch.from_numpy(rgb)


def _imagenet_norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    x = x.float().div_(255.0)
    x.sub_(mean).div_(std)
    return x


def _norm_bufs(device: torch.device):
    mean = torch.tensor(_IMAGENET_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    return mean, std


def _move_batch(left_t, right_t, K_t, seg_gt, occ_gt, bev_seg_gt, device, mean, std, non_blocking):
    left_t = left_t.to(device, non_blocking=non_blocking)
    right_t = right_t.to(device, non_blocking=non_blocking)
    K_t = K_t.to(device, non_blocking=non_blocking)
    seg_gt = seg_gt.to(device, non_blocking=non_blocking)
    occ_gt = occ_gt.to(device, non_blocking=non_blocking)
    bev_seg_gt = bev_seg_gt.to(device, non_blocking=non_blocking)
    if left_t.dtype == torch.uint8:
        left_t = _imagenet_norm(left_t, mean, std)
        right_t = _imagenet_norm(right_t, mean, std)
    return left_t, right_t, K_t, seg_gt, occ_gt, bev_seg_gt


def _load_npz_file(npz) -> tuple:
    missing = [k for k in _NPZ_FIELDS if k not in npz.files]
    if missing:
        raise KeyError(
            f"Sample missing {missing}. Re-collect with "
            "collect_data.py --kv-url http://localhost:3000 "
            "(run_collect_v2.py samples omit bev_seg_gt)."
        )
    return tuple(npz[k] for k in _NPZ_FIELDS)


class StereoBEVDataset(Dataset):
    """
    Each .npz contains:
      left_rgb:   (H, W, 3) uint8 BGR
      right_rgb:  (H, W, 3) uint8 BGR
      depth_gt:   (H, W) float32 meters
      seg_gt:     (H, W) uint8 — IMAGE-SPACE segmentation (camera view)
      occ_gt:     (bev_h, bev_w) uint8 — BEV occupancy
      bev_seg_gt: (bev_h, bev_w) uint8 — BEV per-cell semantic class
      K:          (3, 3) float64 intrinsics

    Loads from local .npz files or minikeyvalue (`--kv-url`).
    """

    def __init__(self, data_dir: str = None, kv_url: str = None, kv_prefix: str = "/train/"):
        self.kv_url = kv_url
        self._kv = None
        if kv_url:
            self.keys = MiniKV(kv_url).list_keys(kv_prefix)
            self.files = None
            if len(self.keys) == 0:
                raise FileNotFoundError(f"No keys in KV with prefix {kv_prefix}")
        else:
            self.keys = None
            self.files = sorted([
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".npz")
            ])
            if len(self.files) == 0:
                raise FileNotFoundError(f"No .npz files in {data_dir}")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_kv"] = None
        return state

    def _client(self) -> MiniKV:
        if self._kv is None:
            self._kv = MiniKV(self.kv_url)
        return self._kv

    def __len__(self):
        return len(self.keys) if self.kv_url else len(self.files)

    def __getitem__(self, idx):
        if self.kv_url:
            left, right, K, depth_gt, seg_gt, occ_gt, bev_seg_gt = _load_npz_file(
                self._client().get_npz(self.keys[idx])
            )
        else:
            left, right, K, depth_gt, seg_gt, occ_gt, bev_seg_gt = _load_npz_file(
                np.load(self.files[idx])
            )

        return (
            _bgr_to_nchw_uint8(left),            # (3, H, W) uint8 RGB
            _bgr_to_nchw_uint8(right),           # (3, H, W) uint8 RGB
            torch.from_numpy(np.ascontiguousarray(K)).float(),
            torch.from_numpy(np.ascontiguousarray(depth_gt)).float(),
            torch.from_numpy(np.ascontiguousarray(seg_gt)).long(),
            torch.from_numpy(np.ascontiguousarray(occ_gt)).float(),
            torch.from_numpy(np.ascontiguousarray(bev_seg_gt)).long(),
        )


# ════════════════════════════════════════════════════════════════
#  Visualization helpers
# ════════════════════════════════════════════════════════════════

def colorize_seg(seg_classes: np.ndarray) -> np.ndarray:
    """(H, W) class indices → (H, W, 3) RGB."""
    return BEV_COLORS[seg_classes.clip(0, NUM_BEV_CLASSES - 1)]


def depth_to_heatmap(depth: np.ndarray, max_depth: float = 80.0) -> np.ndarray:
    """(H, W) depth → (H, W, 3) BGR heatmap."""
    norm = np.clip(depth / max_depth, 0, 1)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[depth < 0.1] = 0
    return colored


def render_seg_comparison(
    left_rgb: np.ndarray,
    seg_gt: np.ndarray,
    seg_pred: np.ndarray,
) -> np.ndarray:
    """
    Side-by-side: left RGB overlay with GT seg | pred seg.
    Both in image space (camera view).
    """
    gt_color = colorize_seg(seg_gt)
    pred_color = colorize_seg(seg_pred)

    # resize all to same size
    H, W = seg_gt.shape
    left_vis = cv2.resize(left_rgb, (W, H))
    left_vis = cv2.cvtColor(left_vis, cv2.COLOR_BGR2RGB)

    # blend RGB with seg (60% seg, 40% RGB)
    blend_gt = (gt_color * 0.6 + left_vis * 0.4).astype(np.uint8)
    blend_pred = (pred_color * 0.6 + left_vis * 0.4).astype(np.uint8)

    combined = np.concatenate([blend_gt, blend_pred], axis=1)
    cv2.putText(combined, "GT Seg", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(combined, "Pred Seg", (W + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # legend
    legend_h = 16
    legend = np.zeros((legend_h * NUM_BEV_CLASSES, combined.shape[1], 3), dtype=np.uint8)
    for c in range(NUM_BEV_CLASSES):
        y0 = c * legend_h
        legend[y0:y0 + legend_h, :W] = BEV_COLORS[c]
        cv2.putText(legend[y0:y0 + legend_h], f"{c}: {BEV_CLASSES.get(c, '')}", (W + 5, 12),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    return np.concatenate([combined, legend], axis=0)


def render_occ_comparison(
    occ_gt: np.ndarray,
    occ_pred: np.ndarray,
    scale: int = 3,
) -> np.ndarray:
    """Side-by-side BEV occupancy GT vs predicted."""
    H, W = occ_gt.shape

    # upscale occ arrays to match the canvas
    occ_gt_up = cv2.resize(occ_gt.astype(np.uint8), (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    occ_pred_up = cv2.resize(occ_pred.astype(np.uint8), (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)

    def draw(occ, color, label):
        img = np.full((H * scale, W * scale, 3), (30, 30, 30), dtype=np.uint8)
        img[occ > 0] = color
        cv2.putText(img, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return img

    gt_img = draw(occ_gt_up, (50, 200, 50), "GT Occ")
    pred_img = draw(occ_pred_up, (50, 50, 200), "Pred Occ")

    # overlay
    overlay = draw(occ_gt_up, (50, 200, 50), "Overlay")
    overlay[occ_pred_up > 0] = (50, 50, 200)
    both = (occ_gt_up > 0) & (occ_pred_up > 0)
    overlay[both] = (50, 200, 200)

    return np.concatenate([gt_img, pred_img, overlay], axis=1)


def render_bev_seg_comparison(
    bev_seg_gt: np.ndarray,
    bev_seg_pred: np.ndarray,
    scale: int = 3,
) -> np.ndarray:
    """Side-by-side BEV per-cell semantic class: GT vs predicted."""
    H, W = bev_seg_gt.shape
    gt_up = cv2.resize(bev_seg_gt.astype(np.uint8), (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    pred_up = cv2.resize(bev_seg_pred.astype(np.uint8), (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)

    gt_img = colorize_seg(gt_up)
    pred_img = colorize_seg(pred_up)
    cv2.putText(gt_img, "GT BEV Seg", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(pred_img, "Pred BEV Seg", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return np.concatenate([gt_img, pred_img], axis=1)


def make_comparison_panel(
    left_rgb: np.ndarray,
    depth_gt: np.ndarray,
    depth_pred: np.ndarray,
    seg_gt: np.ndarray,
    seg_pred: np.ndarray,
    occ_gt: np.ndarray,
    occ_pred: np.ndarray,
    bev_seg_gt: np.ndarray,
    bev_seg_pred: np.ndarray,
) -> np.ndarray:
    """
    Full comparison panel for TensorBoard.

    Row 1: left_rgb | GT depth | pred depth
    Row 2: GT seg overlay | pred seg overlay (image space)
    Row 3: GT occ | pred occ | overlay (BEV)
    Row 4: GT BEV seg | pred BEV seg (BEV per-cell class)
    """
    target_h = 160

    def resize(img, w=None):
        if w is None:
            w = int(target_h * img.shape[1] / max(img.shape[0], 1))
        return cv2.resize(img, (w, target_h), interpolation=cv2.INTER_NEAREST)

    def pad_to(img, w):
        if img.shape[1] < w:
            return np.concatenate([img, np.zeros((img.shape[0], w - img.shape[1], 3), dtype=np.uint8)], axis=1)
        return img[:, :w]

    # row 1: RGB + depth
    left_vis = resize(cv2.cvtColor(left_rgb, cv2.COLOR_BGR2RGB))
    dgt = resize(cv2.cvtColor(depth_to_heatmap(depth_gt), cv2.COLOR_BGR2RGB))
    dpred = resize(cv2.cvtColor(depth_to_heatmap(depth_pred), cv2.COLOR_BGR2RGB))
    w1 = max(left_vis.shape[1], dgt.shape[1], dpred.shape[1])
    row1 = np.concatenate([pad_to(left_vis, w1), pad_to(dgt, w1), pad_to(dpred, w1)], axis=1)

    # row 2: image-space segmentation (resize to match)
    H, W = seg_gt.shape
    left_small = cv2.resize(cv2.cvtColor(left_rgb, cv2.COLOR_BGR2RGB), (W, H))
    gt_blend = (colorize_seg(seg_gt) * 0.6 + left_small * 0.4).astype(np.uint8)
    pred_blend = (colorize_seg(seg_pred) * 0.6 + left_small * 0.4).astype(np.uint8)
    gt_blend = resize(gt_blend)
    pred_blend = resize(pred_blend)
    w2 = max(gt_blend.shape[1], pred_blend.shape[1])
    row2 = np.concatenate([pad_to(gt_blend, w2), pad_to(pred_blend, w2)], axis=1)

    # row 3: BEV occupancy
    occ_vis = render_occ_comparison(occ_gt, occ_pred, scale=2)
    occ_vis = resize(occ_vis)
    row3 = occ_vis

    # row 4: BEV semantic class
    bev_seg_vis = render_bev_seg_comparison(bev_seg_gt, bev_seg_pred, scale=2)
    bev_seg_vis = resize(bev_seg_vis)
    row4 = bev_seg_vis

    # match widths
    w = max(row1.shape[1], row2.shape[1], row3.shape[1], row4.shape[1])
    row1 = pad_to(row1, w)
    row2 = pad_to(row2, w)
    row3 = pad_to(row3, w)
    row4 = pad_to(row4, w)

    # labels
    cv2.putText(row1, "Input", (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(row1, "GT Depth", (w1 + 5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(row1, "Pred Depth", (w1 * 2 + 5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(row2, "GT Seg (image)", (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(row2, "Pred Seg (image)", (w2 + 5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return np.concatenate([row1, row2, row3, row4], axis=0)


# ════════════════════════════════════════════════════════════════
#  3D voxel mesh for TensorBoard
# ════════════════════════════════════════════════════════════════

_CUBE_VERTS = np.array([
    [0,0,0],[1,0,0],[1,1,0],[0,1,0],
    [0,0,1],[1,0,1],[1,1,1],[0,1,1],
], dtype=np.float32)

_CUBE_FACES = np.array([
    [0,1,2],[0,2,3],[4,5,6],[4,6,7],
    [0,1,5],[0,5,4],[2,3,7],[2,7,6],
    [0,3,7],[0,7,4],[1,2,6],[1,6,5],
], dtype=np.int64)


def occ_to_voxel_mesh(occ, voxel_size=0.1, x_range=(-5,5), y_range=(-5,5),
                       z_min=0.0, color=None):
    """Occupancy grid → cube mesh for TensorBoard add_mesh."""
    ys, xs = np.where(occ > 0)
    N = len(ys)
    if N == 0:
        return np.zeros((8,3)), np.zeros((12,3),dtype=np.int64), np.zeros((8,3))

    base_x = xs * voxel_size + x_range[0]
    base_y = ys * voxel_size + y_range[0]
    scaled = _CUBE_VERTS * voxel_size
    offsets = np.stack([base_x, base_y, np.full(N, z_min)], axis=-1)[:, np.newaxis, :]
    verts = (scaled[np.newaxis] + offsets).reshape(-1, 3)

    face_off = (np.arange(N) * 8)[:, None, None]
    faces = (_CUBE_FACES[np.newaxis] + face_off).reshape(-1, 3)

    if color is not None:
        colors = np.tile(color, (N, 1))
    else:
        colors = np.full((N, 3), 0.5)
    colors = np.repeat(colors, 8, axis=0)

    return verts, faces, colors


def log_3d_occupancy(writer, gt_occ, pred_occ, epoch, name="occupancy3d",
                      voxel_size=0.1):
    """Log GT + predicted occupancy as 3D voxel cubes."""
    gt_v, gt_f, gt_c = occ_to_voxel_mesh(gt_occ, voxel_size, color=np.array([0.2, 0.8, 0.2]))
    pv, pf, pc = occ_to_voxel_mesh(pred_occ, voxel_size, color=np.array([0.9, 0.2, 0.2]))
    pv[:, 1] += 12.0  # offset pred on Y

    n_gt = gt_v.shape[0]
    all_v = np.concatenate([gt_v, pv])
    all_f = np.concatenate([gt_f, pf + n_gt])
    all_c = np.concatenate([gt_c, pc])

    writer.add_mesh(name,
        vertices=torch.from_numpy(all_v).float().unsqueeze(0),
        faces=torch.from_numpy(all_f).long().unsqueeze(0),
        colors=torch.from_numpy(all_c).float().unsqueeze(0),
        global_step=epoch)


# ════════════════════════════════════════════════════════════════
#  Training loop
# ════════════════════════════════════════════════════════════════

def evaluate(model, loader, device, mean, std, use_amp=False, non_blocking=False):
    model.eval()
    totals = {k: torch.zeros((), device=device) for k in ("loss", "seg", "occ", "bev_seg")}
    n = 0
    amp_cm = torch.cuda.amp.autocast if use_amp else nullcontext
    with torch.no_grad():
        for left_t, right_t, K_t, _, seg_gt, occ_gt, bev_seg_gt in loader:
            left_t, right_t, K_t, seg_gt, occ_gt, bev_seg_gt = _move_batch(
                left_t, right_t, K_t, seg_gt, occ_gt, bev_seg_gt,
                device, mean, std, non_blocking,
            )
            with amp_cm():
                seg_logits, occ_logits, bev_seg_logits, _ = model(left_t, right_t, K_t)
                losses = stereo_bev_loss(seg_logits, occ_logits, bev_seg_logits, seg_gt, occ_gt, bev_seg_gt)
            bs = left_t.size(0)
            totals["loss"] += losses["loss"].detach() * bs
            totals["seg"] += losses["seg_loss"].detach() * bs
            totals["occ"] += losses["occ_loss"].detach() * bs
            totals["bev_seg"] += losses["bev_seg_loss"].detach() * bs
            n += bs
    return tuple((totals[k] / n).item() for k in ("loss", "seg", "occ", "bev_seg"))


def log_visualizations(model, loader, writer, device, epoch, mean, std,
                       bev_voxel=0.1, max_samples=4, use_amp=False, non_blocking=False):
    """Log comparison images + 3D occupancy to TensorBoard."""
    model.eval()
    count = 0
    amp_cm = torch.cuda.amp.autocast if use_amp else nullcontext

    with torch.no_grad():
        for left_t, right_t, K_t, depth_gt_b, seg_gt_b, occ_gt_b, bev_seg_gt_b in loader:
            left_t, right_t, K_t, _, _, _ = _move_batch(
                left_t, right_t, K_t, seg_gt_b, occ_gt_b, bev_seg_gt_b,
                device, mean, std, non_blocking,
            )

            with amp_cm():
                seg_logits, occ_logits, bev_seg_logits, depth_logits = model(left_t, right_t, K_t)

            B = left_t.size(0)
            for b in range(B):
                if count >= max_samples:
                    return

                # predicted outputs
                pred_seg = seg_logits[b].argmax(dim=0).cpu().numpy().astype(np.uint8)
                pred_occ = (torch.sigmoid(occ_logits[b]).squeeze() > 0.5).cpu().numpy().astype(np.uint8)
                pred_bev_seg = bev_seg_logits[b].argmax(dim=0).cpu().numpy().astype(np.uint8)

                # predicted depth (from model)
                D = depth_logits.shape[1]
                depth_bins = torch.linspace(1.0, 80.0, D).to(device)
                depth_prob = torch.softmax(depth_logits[b:b+1].float(), dim=1)
                expected = (depth_prob * depth_bins.view(1, -1, 1, 1)).sum(dim=1)
                pred_depth = torch.nn.functional.interpolate(
                    expected.unsqueeze(1), size=depth_gt_b.shape[1:],
                    mode='bilinear', align_corners=False,
                ).squeeze().cpu().numpy()

                # GT
                gt_depth = depth_gt_b[b].numpy()
                gt_seg = seg_gt_b[b].numpy().astype(np.uint8)
                gt_occ = occ_gt_b[b].numpy().astype(np.uint8)
                gt_bev_seg = bev_seg_gt_b[b].numpy().astype(np.uint8)

                # left RGB (denormalize)
                left_np = left_t[b].cpu().permute(1, 2, 0).numpy()
                mean_np = np.array(_IMAGENET_MEAN)
                std_np = np.array(_IMAGENET_STD)
                left_np = ((left_np * std_np + mean_np) * 255).clip(0, 255).astype(np.uint8)
                left_np = cv2.cvtColor(left_np, cv2.COLOR_RGB2BGR)

                # 2D comparison panel
                panel = make_comparison_panel(
                    left_np, gt_depth, pred_depth,
                    gt_seg, pred_seg, gt_occ, pred_occ,
                    gt_bev_seg, pred_bev_seg,
                )
                writer.add_image(f"val/sample_{count}", panel.transpose(2, 0, 1), epoch)

                # dedicated seg comparison (image space)
                seg_cmp = render_seg_comparison(left_np, gt_seg, pred_seg)
                writer.add_image(f"seg/sample_{count}", seg_cmp.transpose(2, 0, 1), epoch)

                # dedicated occ comparison (BEV)
                occ_cmp = render_occ_comparison(gt_occ, pred_occ, scale=3)
                writer.add_image(f"occ/sample_{count}", occ_cmp.transpose(2, 0, 1), epoch)

                # dedicated BEV semantic comparison
                bev_seg_cmp = render_bev_seg_comparison(gt_bev_seg, pred_bev_seg, scale=3)
                writer.add_image(f"bevseg/sample_{count}", bev_seg_cmp.transpose(2, 0, 1), epoch)

                count += 1


def _dataloader_kwargs(batch_size, shuffle, workers, pin_memory):
    kw = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    if workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=2)
    return kw


def train(
    data_dir="bev_data", epochs=50, batch_size=2, lr=1e-4,
    checkpoint_path="stereo_bev_model.pth", log_dir="runs/bev_train",
    image_w=960, image_h=540,
    bev_range_xy=5.0, bev_z_range=5.0, bev_voxel=0.1, max_depth=80.0,
    viz_every=5, resume=False, kv_url=None,
    workers=4, amp=True, device=None,
):
    if device is None or device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    use_cuda = device.type == "cuda"
    use_amp = bool(amp) and use_cuda
    non_blocking = use_cuda
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Train] Device: cuda ({gpu_name})")
    else:
        print("[Train] Device: cpu  (CUDA not available)")

    if kv_url:
        train_dataset = StereoBEVDataset(kv_url=kv_url, kv_prefix="/train/")
        val_dataset = StereoBEVDataset(kv_url=kv_url, kv_prefix="/val/")
        print(f"[Train] KV mode: {kv_url}")
    else:
        train_dataset = StereoBEVDataset(data_dir=os.path.join(data_dir, "train"))
        val_dataset = StereoBEVDataset(data_dir=os.path.join(data_dir, "val"))
    print(f"[Train] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"[Train] DataLoader workers={workers}  amp={use_amp}  pin_memory={use_cuda}")

    train_loader = DataLoader(
        train_dataset, **_dataloader_kwargs(batch_size, True, workers, use_cuda),
    )
    val_loader = DataLoader(
        val_dataset, **_dataloader_kwargs(batch_size, False, workers, use_cuda),
    )

    model = StereoBEVModel(
        num_classes=NUM_BEV_CLASSES, image_h=image_h, image_w=image_w,
        bev_x_range=(-bev_range_xy, bev_range_xy),
        bev_y_range=(-bev_range_xy, bev_range_xy),
        bev_z_range=(0.0, bev_z_range), bev_voxel=bev_voxel,
        max_depth=max_depth, pretrained_backbone=True,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model params: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    mean, std = _norm_bufs(device)
    amp_cm = torch.cuda.amp.autocast if use_amp else nullcontext

    start_epoch = 0
    if resume and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if ckpt.get("scaler") is not None and use_amp:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"] + 1
            print(f"[Train] Resumed at epoch {start_epoch}")
        else:
            model.load_state_dict(ckpt)

    writer = SummaryWriter(log_dir)
    print(f"[Train] TensorBoard: tensorboard --logdir {log_dir}")

    print(f"[Train] Training for {epochs} epochs...")
    for epoch in range(start_epoch, epochs):
        model.train()
        totals = {k: torch.zeros((), device=device) for k in ("loss", "seg", "occ", "bev_seg")}
        n = 0

        for left_t, right_t, K_t, _, seg_gt, occ_gt, bev_seg_gt in train_loader:
            left_t, right_t, K_t, seg_gt, occ_gt, bev_seg_gt = _move_batch(
                left_t, right_t, K_t, seg_gt, occ_gt, bev_seg_gt,
                device, mean, std, non_blocking,
            )

            optimizer.zero_grad(set_to_none=True)
            with amp_cm():
                seg_logits, occ_logits, bev_seg_logits, _ = model(left_t, right_t, K_t)
                losses = stereo_bev_loss(seg_logits, occ_logits, bev_seg_logits, seg_gt, occ_gt, bev_seg_gt)
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = left_t.size(0)
            totals["loss"] += losses["loss"].detach() * bs
            totals["seg"] += losses["seg_loss"].detach() * bs
            totals["occ"] += losses["occ_loss"].detach() * bs
            totals["bev_seg"] += losses["bev_seg_loss"].detach() * bs
            n += bs

        scheduler.step()
        train_loss = (totals["loss"] / n).item()
        train_seg = (totals["seg"] / n).item()
        train_occ = (totals["occ"] / n).item()
        train_bev_seg = (totals["bev_seg"] / n).item()

        val_loss, val_seg, val_occ, val_bev_seg = evaluate(
            model, val_loader, device, mean, std, use_amp=use_amp, non_blocking=non_blocking,
        )

        # save checkpoint
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
        }, checkpoint_path)

        # tensorboard
        lr_now = scheduler.get_last_lr()[0]
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("seg_loss", {"train": train_seg, "val": val_seg}, epoch)
        writer.add_scalars("occ_loss", {"train": train_occ, "val": val_occ}, epoch)
        writer.add_scalars("bev_seg_loss", {"train": train_bev_seg, "val": val_bev_seg}, epoch)
        writer.add_scalar("lr", lr_now, epoch)

        if epoch % viz_every == 0 or epoch == epochs - 1:
            try:
                log_visualizations(
                    model, val_loader, writer, device, epoch, mean, std,
                    bev_voxel=bev_voxel, use_amp=use_amp, non_blocking=non_blocking,
                )
            except Exception as e:
                print(f"[Train] viz skipped: {e}", flush=True)

        if (epoch + 1) % 5 == 0 or epoch == start_epoch:
            print(f"  epoch {epoch+1:3d}/{epochs}  "
                  f"train: loss={train_loss:.4f} seg={train_seg:.4f} occ={train_occ:.4f} bev_seg={train_bev_seg:.4f}  "
                  f"val: loss={val_loss:.4f} seg={val_seg:.4f} occ={val_occ:.4f} bev_seg={val_bev_seg:.4f}  "
                  f"lr={lr_now:.6f}", flush=True)

    writer.close()
    print(f"[Train] Checkpoint → {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="bev_data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="stereo_bev_model.pth")
    parser.add_argument("--logdir", default="runs/bev_train")
    parser.add_argument("--viz-every", type=int, default=5)
    parser.add_argument("--image-w", type=int, default=960)
    parser.add_argument("--image-h", type=int, default=540)
    parser.add_argument("--bev-range", type=float, default=5.0)
    parser.add_argument("--bev-voxel", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--kv-url", default=None, help="minikeyvalue URL (e.g. http://localhost:3000)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers (0 = load on the training process)")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    train(data_dir=args.data, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, checkpoint_path=args.output, log_dir=args.logdir,
          viz_every=args.viz_every, image_w=args.image_w, image_h=args.image_h,
          bev_range_xy=args.bev_range, bev_voxel=args.bev_voxel, resume=args.resume,
          kv_url=args.kv_url, workers=args.workers, amp=not args.no_amp,
          device=args.device)
