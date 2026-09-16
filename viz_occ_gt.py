"""Visualize 3D ground-truth occupancy lifted from collected samples."""
import os
import io
import base64
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from stereo_bev.bev_grid import (
    BEVGrid,
    DEFAULT_X_RANGE, DEFAULT_Y_RANGE, DEFAULT_Z_RANGE, DEFAULT_VOXEL,
)
from stereo_bev.query_heads import GeometricOccHead
from stereo_bev.calibration import ego_from_camera
from stereo_bev.visualize import draw_occ_3d_projections, draw_depth_heatmap
from stereo_bev.segmentation import BEV_COLORS, NUM_BEV_CLASSES, BEV_CLASSES

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "occ_vis")
os.makedirs(OUT, exist_ok=True)

# Existing KV samples were captured with an untilted camera.
LEGACY_CAM_EXT = ego_from_camera((1.5, 0.0, 1.6), pitch_deg=0.0)


def list_samples():
    files = []
    for root, _, fnames in os.walk(os.path.join(BASE, "kv_data", "vol1")):
        for f in fnames:
            files.append(os.path.join(root, f))
    files.sort()
    return files


def load_npz(path):
    return np.load(io.BytesIO(open(path, "rb").read()))


def sample_name(path):
    try:
        return base64.b64decode(os.path.basename(path)).decode()
    except Exception:
        return os.path.basename(path)


def lift_3d(d, x_range=DEFAULT_X_RANGE, y_range=DEFAULT_Y_RANGE,
            z_range=DEFAULT_Z_RANGE, voxel=DEFAULT_VOXEL):
    bev = BEVGrid(
        x_range=x_range, y_range=y_range, z_range=z_range,
        voxel_size=voxel, num_classes=NUM_BEV_CLASSES,
    )
    cam_ext = d["cam_ext"] if "cam_ext" in d.files else LEGACY_CAM_EXT
    res = bev.bev_from_frame(d["depth_gt"], d["seg_gt"], d["K"], cam_ext, max_depth=80.0)
    occ = GeometricOccHead(2.0)(res["occupancy_count"])
    bev_seg = bev.get_bev_semantic(res["class_histogram"])
    return occ, bev_seg, bev


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def colorize_seg(seg):
    return BEV_COLORS[seg.clip(0, NUM_BEV_CLASSES - 1)]


DARK = "#1a1a1a"
AXBG = "#111111"
FG = "white"
MUTED = "0.65"


def _style_ax(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(AXBG)
    ax.set_title(title, color=FG, fontsize=11, pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.tick_params(colors="0.5", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("0.3")


def _class_rgb(cls_hw, mask):
    rgb = BEV_COLORS[np.clip(cls_hw, 0, NUM_BEV_CLASSES - 1)].astype(np.float32) / 255.0
    rgb = rgb.copy()
    rgb[~mask] = 0.09
    return rgb


def render_detail(path, tag):
    d = load_npz(path)
    name = sample_name(path)
    occ, bev_seg, bev = lift_3d(d)
    z_bins, ny, nx = occ.shape
    occupied = occ.astype(bool)
    n_vox = int(occupied.sum())
    z_hits = np.where(occupied.reshape(z_bins, -1).sum(1) > 0)[0]

    left = bgr_to_rgb(d["left_rgb"])
    dep = cv2.cvtColor(draw_depth_heatmap(d["depth_gt"], max_depth=40.0), cv2.COLOR_BGR2RGB)
    blend = (colorize_seg(d["seg_gt"]) * 0.65 + left * 0.35).astype(np.uint8)

    zs, ys, xs = np.where(occupied)
    x_m = bev.x_range[0] + (xs + 0.5) * bev.voxel_size
    y_m = bev.y_range[0] + (ys + 0.5) * bev.voxel_size
    z_m = bev.z_range[0] + (zs + 0.5) * bev.voxel_size
    cls = bev_seg[ys, xs]
    colors = BEV_COLORS[np.clip(cls, 0, NUM_BEV_CLASSES - 1)] / 255.0

    # keep structure; downsample the ground sheet so poles stay visible
    rng = np.random.default_rng(0)
    ground = z_m < 0.35
    keep = np.ones(len(xs), dtype=bool)
    gidx = np.where(ground)[0]
    if len(gidx) > 1200:
        drop = rng.choice(gidx, size=len(gidx) - 1200, replace=False)
        keep[drop] = False

    fig = plt.figure(figsize=(16, 11.5))
    fig.patch.set_facecolor(DARK)
    gs = fig.add_gridspec(
        3, 3, height_ratios=[1.0, 1.35, 1.05],
        wspace=0.22, hspace=0.32,
        left=0.05, right=0.98, top=0.91, bottom=0.06,
    )

    for ax, im, title in (
        (fig.add_subplot(gs[0, 0]), left, "Left RGB"),
        (fig.add_subplot(gs[0, 1]), dep, "Depth GT"),
        (fig.add_subplot(gs[0, 2]), blend, "Image-space seg"),
    ):
        ax.imshow(im)
        ax.set_title(title, color=FG, fontsize=11)
        ax.axis("off")

    ax3 = fig.add_subplot(gs[1, 0:2], projection="3d")
    ax3.set_facecolor(AXBG)
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False
    if keep.any():
        ax3.scatter(
            x_m[keep], y_m[keep], z_m[keep],
            c=colors[keep], s=np.where(ground[keep], 4, 14),
            linewidths=0, depthshade=False,
        )
    ax3.set_xlim(*bev.x_range)
    ax3.set_ylim(bev.y_range[1], bev.y_range[0])  # +left toward viewer-left
    ax3.set_zlim(*bev.z_range)
    ax3.set_xlabel("X forward (m)", color=MUTED, fontsize=8)
    ax3.set_ylabel("Y left (m)", color=MUTED, fontsize=8)
    ax3.set_zlabel("Z up (m)", color=MUTED, fontsize=8)
    ax3.tick_params(colors="0.5", labelsize=7)
    ax3.view_init(elev=18, azim=-115)
    try:
        ax3.set_box_aspect((
            bev.x_range[1] - bev.x_range[0],
            bev.y_range[1] - bev.y_range[0],
            (bev.z_range[1] - bev.z_range[0]) * 2.5,
        ))
    except Exception:
        pass
    ax3.set_title("3D occupancy  (color = BEV class, ground subsampled)", color=FG, fontsize=11)

    # BEV XY colored by class
    ax_xy = fig.add_subplot(gs[1, 2])
    xy_mask = occupied.any(axis=0)
    xy_rgb = _class_rgb(bev_seg, xy_mask)
    ax_xy.imshow(
        np.transpose(xy_rgb, (1, 0, 2)),
        origin="lower",
        extent=[bev.y_range[0], bev.y_range[1], bev.x_range[0], bev.x_range[1]],
        interpolation="nearest",
    )
    ax_xy.invert_xaxis()
    ax_xy.set_aspect("equal")
    _style_ax(ax_xy, "BEV  (top-down)", "Y left (m)", "X forward (m)")
    ax_xy.plot(0, 0, marker="^", color="white", markersize=7)

    # class along columns for side/front
    cls_vol = np.broadcast_to(bev_seg[None], occupied.shape)
    marked = np.where(occupied, cls_vol, 0)

    ax_xz = fig.add_subplot(gs[2, 0:2])
    xz_cls = marked.max(axis=1)  # (Z, X)
    xz_mask = occupied.any(axis=1)
    xz_rgb = _class_rgb(xz_cls, xz_mask)
    ax_xz.imshow(
        xz_rgb,
        origin="lower",
        extent=[bev.x_range[0], bev.x_range[1], bev.z_range[0], bev.z_range[1]],
        interpolation="nearest", aspect="auto",
    )
    _style_ax(ax_xz, "Side  XZ  (along-track)", "X forward (m)", "Z up (m)")

    ax_yz = fig.add_subplot(gs[2, 2])
    yz_cls = marked.max(axis=2)  # (Z, Y)
    yz_mask = occupied.any(axis=2)
    yz_rgb = _class_rgb(yz_cls, yz_mask)
    ax_yz.imshow(
        yz_rgb,
        origin="lower",
        extent=[bev.y_range[0], bev.y_range[1], bev.z_range[0], bev.z_range[1]],
        interpolation="nearest", aspect="auto",
    )
    ax_yz.invert_xaxis()
    _style_ax(ax_yz, "Front  YZ", "Y left (m)", "Z up (m)")

    present = sorted(set(np.unique(cls).tolist()))
    legend = "   ".join(
        f"{int(c)}:{BEV_CLASSES.get(int(c), str(c))}" for c in present if int(c) != 0
    )
    fig.suptitle(
        f"3D occupancy GT  {name}    {n_vox} voxels    "
        f"{len(z_hits)}/{z_bins} height bins occupied    grid {tuple(occ.shape)}\n{legend}",
        color=FG, fontsize=12, y=0.98,
    )
    out = os.path.join(OUT, f"occ3d_{tag}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, n_vox, len(z_hits), occ.shape


def render_zhist(files, n=12):
    bev = BEVGrid()
    z_counts = np.zeros(bev.grid_z, dtype=np.int64)
    totals = []
    for p in files[:n]:
        d = load_npz(p)
        occ, _, _ = lift_3d(d)
        z_counts += occ.reshape(occ.shape[0], -1).sum(1).astype(np.int64)
        totals.append(int(occ.sum()))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#111111")
    z_m = bev.z_range[0] + (np.arange(bev.grid_z) + 0.5) * bev.voxel_size
    ax.bar(z_m, z_counts / max(n, 1), width=bev.voxel_size * 0.85, color="#3ddc84", edgecolor="#1a1a1a")
    ax.set_title(f"Mean occupied voxels per height bin  ({n} samples)", color="white")
    ax.set_xlabel("Z up (m)", color="0.7")
    ax.set_ylabel("voxels / sample", color="0.7")
    ax.tick_params(colors="0.6")
    for spine in ax.spines.values():
        spine.set_color("0.3")
    fig.tight_layout()
    out = os.path.join(OUT, "occ3d_zhist.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out, totals


def main():
    files = list_samples()
    print(f"[viz] {len(files)} samples")
    picks = [
        (files[0], "first"),
        (files[len(files) // 3], "mid"),
        (files[-1], "last"),
    ]
    for p, tag in picks:
        out, n, zb, shape = render_detail(p, tag)
        print(f"[viz] {tag:8s} {sample_name(p):24s} voxels={n:6d} z_bins={zb:2d} shape={shape}")
        print(f"      {out}")
    hist, totals = render_zhist(files, n=min(16, len(files)))
    print(f"[viz] voxel counts over {len(totals)} samples: "
          f"min={min(totals)} median={int(np.median(totals))} max={max(totals)}")
    print(f"[viz] {hist}")


if __name__ == "__main__":
    main()
