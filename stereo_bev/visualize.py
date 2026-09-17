"""Visualization utilities for BEV perception outputs."""

import cv2
import numpy as np

from .segmentation import BEV_COLORS, NUM_BEV_CLASSES, BEV_CLASSES
from .bev_grid import occupancy_to_bev


def draw_bev_map(
    bev_classes: np.ndarray,
    occupancy: np.ndarray | None = None,
    scale: int = 4,
    mark_center: bool = True,
) -> np.ndarray:
    """
    Render BEV semantic map (with optional occupancy overlay) as an image.

    Args:
        bev_classes: (H, W) class indices
        occupancy:   (H, W) or (Z, H, W) binary, or None
        scale:       pixel scale-up factor

    Returns BGR image (H*scale, W*scale, 3).
    """
    h, w = bev_classes.shape
    img = BEV_COLORS[bev_classes.clip(0, NUM_BEV_CLASSES - 1)]  # (H, W, 3) RGB

    if occupancy is not None:
        occ_mask = occupancy_to_bev(occupancy).astype(bool)
        img[~occ_mask] = (img[~occ_mask] * 0.2).astype(np.uint8)

    img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    if mark_center:
        # Vehicle / occupancy origin (x=0, y=0): left edge, vertical mid.
        cx, cy = int(0.5 * scale), h * scale // 2
        cv2.drawMarker(img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def draw_legend(img: np.ndarray, top_left: tuple = (10, 10)) -> np.ndarray:
    """Overlay a small class legend on an image."""
    out = img.copy()
    x0, y0 = top_left
    for idx, name in BEV_CLASSES.items():
        color = BEV_COLORS[idx].tolist()
        cv2.rectangle(out, (x0, y0), (x0 + 16, y0 + 12), color, -1)
        cv2.putText(out, name, (x0 + 22, y0 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        y0 += 16
    return out


def draw_disparity_heatmap(disparity: np.ndarray, max_disp: float = 128.0) -> np.ndarray:
    """False-color disparity for debugging."""
    norm = np.clip(disparity / max_disp, 0, 1)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[disparity <= 0] = 0
    return colored


def draw_depth_heatmap(depth: np.ndarray, max_depth: float = 80.0) -> np.ndarray:
    """False-color depth for debugging."""
    norm = np.clip(depth / max_depth, 0, 1)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[depth <= 0] = 0
    return colored


def _scale_nearest(img: np.ndarray, scale: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def draw_occ_3d_projections(
    occ: np.ndarray,
    scale: int = 4,
    x_range: tuple[float, float] = (0.0, 20.0),
    y_range: tuple[float, float] = (-10.0, 10.0),
    z_range: tuple[float, float] = (-1.0, 3.0),
) -> np.ndarray:
    """
    Orthographic views of a 3D occupancy volume.

    occ: (Z, Y, X) uint8/bool
    Returns BGR panel: BEV (XY, height-colored) | side (XZ) | front (YZ).
    """
    if occ.ndim == 2:
        occ = occ[None, ...]
    z_bins, ny, nx = occ.shape
    occupied = occ.astype(bool)

    z_ids = np.where(occupied, np.arange(z_bins, dtype=np.int32)[:, None, None], -1)
    height_idx = z_ids.max(axis=0)
    t = np.clip(height_idx.astype(np.float32) / max(z_bins - 1, 1), 0, 1)
    xy = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    xy[height_idx < 0] = (24, 24, 24)
    # forward (+X) up, left (+Y) left
    xy = np.fliplr(np.rot90(xy, k=1))

    xz = np.full((z_bins, nx, 3), 24, dtype=np.uint8)
    xz[occupied.any(axis=1)] = (80, 220, 80)
    xz = np.flipud(xz)  # z up

    yz = np.full((z_bins, ny, 3), 24, dtype=np.uint8)
    yz[occupied.any(axis=2)] = (80, 220, 80)
    yz = np.flipud(yz)

    xy = _scale_nearest(xy, scale)
    z_scale = max(scale * 8, 1)
    xz = cv2.resize(xz, (nx * scale, z_bins * z_scale), interpolation=cv2.INTER_NEAREST)
    yz = cv2.resize(yz, (ny * scale, z_bins * z_scale), interpolation=cv2.INTER_NEAREST)

    def label(img, text):
        cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    xy = label(xy, f"BEV XY  z by height  {x_range[0]:.0f}..{x_range[1]:.0f}m fwd")
    xz = label(xz, f"side XZ  x {x_range[0]:.0f}..{x_range[1]:.0f}  z {z_range[0]:.0f}..{z_range[1]:.0f}")
    yz = label(yz, f"front YZ  y {y_range[0]:.0f}..{y_range[1]:.0f}  z {z_range[0]:.0f}..{z_range[1]:.0f}")

    # match heights for concatenation
    h = max(xy.shape[0], xz.shape[0], yz.shape[0])

    def pad_h(img):
        if img.shape[0] >= h:
            return img
        pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
        return np.concatenate([img, pad], axis=0)

    n_vox = int(occupied.sum())
    banner = np.zeros((22, pad_h(xy).shape[1] + pad_h(xz).shape[1] + pad_h(yz).shape[1], 3), dtype=np.uint8)
    cv2.putText(
        banner, f"3D occupancy  {n_vox} voxels  {z_bins}x{ny}x{nx}",
        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA,
    )
    row = np.concatenate([pad_h(xy), pad_h(xz), pad_h(yz)], axis=1)
    return np.concatenate([banner, row], axis=0)


def ego_to_bev_px(x, y, grid, scale: int) -> tuple[int, int]:
    """Map ego XY to pixel (col, row) on a scaled BEV image of shape (Y, X)."""
    col = int(((x - grid.x_range[0]) / grid.voxel_size) * scale + 0.5 * scale)
    row = int(((y - grid.y_range[0]) / grid.voxel_size) * scale + 0.5 * scale)
    return col, row


def ego_to_occ_xy_px(x, y, grid, scale: int) -> tuple[int, int]:
    """
    Map ego XY to pixels on the occupancy XY panel after rot90+fliplr
    (same transform as draw_occ_3d_projections: +X up, +Y left).
    """
    xi = (x - grid.x_range[0]) / grid.voxel_size
    yi = (y - grid.y_range[0]) / grid.voxel_size
    nx, ny = grid.grid_w, grid.grid_h
    row = int((nx - 1 - xi) * scale + 0.5 * scale)
    col = int((ny - 1 - yi) * scale + 0.5 * scale)
    return col, row


def _polyline(img, pts_px, color, thickness=2):
    if len(pts_px) < 2:
        return
    arr = np.array(pts_px, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [arr], False, color, thickness, cv2.LINE_AA)


def _draw_xy_path(img, xy, to_px, color, thickness=2):
    """Draw an ego-XY polyline, splitting wherever it leaves the image."""
    if xy is None or len(xy) < 2:
        return
    h, w = img.shape[:2]
    seg = []

    def flush():
        if len(seg) >= 2:
            _polyline(img, seg, color, thickness)
        seg.clear()

    pts = np.asarray(xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return
    for x, y in pts[:, :2]:
        c, r = to_px(float(x), float(y))
        if 0 <= c < w and 0 <= r < h:
            seg.append((c, r))
        else:
            flush()
    flush()


def draw_planning_on_bev(
    bev_img: np.ndarray,
    grid,
    scale: int,
    traj_xy: np.ndarray | None = None,
    route_xy: np.ndarray | None = None,
    target_xy: np.ndarray | None = None,
    ref_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Overlay route (cyan), NMPC traj (yellow), 5 s target (red), ego (white)."""
    out = bev_img.copy()
    h, w = out.shape[:2]

    def to_px(x, y):
        return ego_to_bev_px(x, y, grid, scale)

    _draw_xy_path(out, route_xy, to_px, (255, 220, 0), 1)
    _draw_xy_path(out, ref_xy, to_px, (180, 255, 180), 1)
    _draw_xy_path(out, traj_xy, to_px, (0, 255, 255), 2)
    if target_xy is not None:
        c, r = to_px(float(target_xy[0]), float(target_xy[1]))
        if 0 <= c < w and 0 <= r < h:
            cv2.drawMarker(out, (c, r), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
    ego = to_px(0.0, 0.0)
    cv2.drawMarker(out, (int(np.clip(ego[0], 0, w - 1)), int(np.clip(ego[1], 0, h - 1))),
                   (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
    return out


def draw_occ_3d_with_sweep(
    occ: np.ndarray,
    sweep: np.ndarray | None = None,
    traj_xy: np.ndarray | None = None,
    route_xy: np.ndarray | None = None,
    target_xy: np.ndarray | None = None,
    ref_xy: np.ndarray | None = None,
    grid=None,
    scale: int = 3,
    x_range: tuple[float, float] = (0.0, 20.0),
    y_range: tuple[float, float] = (-10.0, 10.0),
    z_range: tuple[float, float] = (-1.0, 3.0),
) -> np.ndarray:
    """3D occupancy projections with optional body-sweep tint and path overlay."""
    occ_u8 = (np.asarray(occ) > 0).astype(np.uint8)
    panel = draw_occ_3d_projections(
        occ_u8, scale=scale, x_range=x_range, y_range=y_range, z_range=z_range,
    )
    if grid is None:
        return panel

    banner_h = 22
    xy_w = grid.grid_h * scale  # after rot90+fliplr, width is ny * scale
    xy_h = grid.grid_w * scale
    roi = panel[banner_h:banner_h + xy_h, :xy_w]
    if roi.size == 0:
        return panel

    if sweep is not None:
        sw = np.asarray(sweep).astype(bool)
        if sw.any():
            # project sweep to XY then apply the same rot90+fliplr
            hit = sw.any(axis=0).astype(np.uint8)  # (Y, X)
            hit = np.fliplr(np.rot90(hit, k=1))
            hit = _scale_nearest(hit, scale)
            h = min(hit.shape[0], roi.shape[0])
            w = min(hit.shape[1], roi.shape[1])
            mask = hit[:h, :w] > 0
            roi[:h, :w][mask] = (0, 180, 255)

    _draw_xy_path(roi, route_xy, lambda x, y: ego_to_occ_xy_px(x, y, grid, scale), (255, 220, 0), 1)
    _draw_xy_path(roi, ref_xy, lambda x, y: ego_to_occ_xy_px(x, y, grid, scale), (180, 255, 180), 1)
    _draw_xy_path(roi, traj_xy, lambda x, y: ego_to_occ_xy_px(x, y, grid, scale), (0, 255, 255), 2)
    if target_xy is not None:
        c, r = ego_to_occ_xy_px(float(target_xy[0]), float(target_xy[1]), grid, scale)
        if 0 <= c < roi.shape[1] and 0 <= r < roi.shape[0]:
            cv2.drawMarker(roi, (c, r), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
    return panel


def draw_control_curves(
    u: np.ndarray,
    dt: float,
    a_max: float = 3.0,
    delta_max: float = 0.50,
    width: int = 640,
    height: int = 148,
) -> np.ndarray:
    """Plot NMPC control u(t)=[a, delta] over the horizon as a BGR panel."""
    img = np.full((height, width, 3), 18, dtype=np.uint8)
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2 or u.shape[0] < 1 or u.shape[1] < 2:
        cv2.putText(
            img, "u(t): no control", (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
        )
        return img

    n = int(u.shape[0])
    t = np.arange(n, dtype=np.float64) * float(dt)
    t_max = float(max(t[-1], dt, 1e-6))
    margin_l, margin_r = 52, 10
    margin_t, gap, margin_b = 22, 8, 8
    plot_w = max(width - margin_l - margin_r, 8)
    mid = margin_t + (height - margin_t - margin_b - gap) // 2
    a_color = (80, 220, 80)
    d_color = (0, 165, 255)

    def to_x(tk: float) -> int:
        return int(margin_l + (tk / t_max) * (plot_w - 1))

    def draw_series(y, y_lim, y0, y1, color, label, unit):
        h = max(y1 - y0, 2)
        ylim = max(float(y_lim), 1e-6)

        def to_y(val: float) -> int:
            nrm = 0.5 - 0.5 * (float(val) / ylim)
            return int(np.clip(y0 + nrm * (h - 1), y0, y1 - 1))

        cv2.rectangle(img, (margin_l, y0), (width - margin_r - 1, y1 - 1), (36, 36, 36), -1)
        zy = to_y(0.0)
        cv2.line(img, (margin_l, zy), (width - margin_r - 1, zy), (70, 70, 70), 1)
        for frac in (0.25, 0.5, 0.75, 1.0):
            xg = to_x(frac * t_max)
            cv2.line(img, (xg, y0), (xg, y1 - 1), (45, 45, 45), 1)
        pts = [(to_x(t[k]), to_y(y[k])) for k in range(n)]
        _polyline(img, pts, color, 2)
        cv2.circle(img, pts[0], 4, color, -1, cv2.LINE_AA)
        cv2.putText(
            img, f"{label} {y[0]:+.2f}{unit}",
            (margin_l + 4, y0 + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
        )
        cv2.putText(
            img, f"{ylim:g}",
            (4, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1, cv2.LINE_AA,
        )
        cv2.putText(
            img, f"-{ylim:g}",
            (4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1, cv2.LINE_AA,
        )

    draw_series(u[:, 0], a_max, margin_t, mid, a_color, "a", " m/s^2")
    draw_series(u[:, 1], delta_max, mid + gap, height - margin_b, d_color, "delta", " rad")
    cv2.putText(
        img, f"u(t)  0..{t_max:.1f}s  a green  delta orange",
        (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA,
    )
    cv2.putText(
        img, f"{t_max:.1f}s",
        (width - 44, height - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1, cv2.LINE_AA,
    )
    return img
