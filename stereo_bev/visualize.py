"""Visualization utilities for BEV perception outputs."""

import cv2
import numpy as np

from .segmentation import BEV_COLORS, NUM_BEV_CLASSES, BEV_CLASSES
from .bev_grid import occupancy_to_bev


def draw_bev_map(
    bev_classes: np.ndarray,
    occupancy: np.ndarray | None = None,
    scale: int = 4,
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

    cx, cy = w * scale // 2, h * scale // 2
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
    row = int((nx - 1 - xi) * scale)
    col = int((ny - 1 - yi) * scale)
    return col, row


def _polyline(img, pts_px, color, thickness=2):
    if len(pts_px) < 2:
        return
    arr = np.array(pts_px, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [arr], False, color, thickness, cv2.LINE_AA)


def draw_planning_on_bev(
    bev_img: np.ndarray,
    grid,
    scale: int,
    traj_xy: np.ndarray | None = None,
    route_xy: np.ndarray | None = None,
    target_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Overlay route (cyan), NMPC traj (yellow), 5 s target (red), ego (white)."""
    out = bev_img.copy()
    h, w = out.shape[:2]

    def clip_pts(xy):
        px = []
        for x, y in xy:
            c, r = ego_to_bev_px(float(x), float(y), grid, scale)
            if -8 <= c < w + 8 and -8 <= r < h + 8:
                px.append((int(np.clip(c, 0, w - 1)), int(np.clip(r, 0, h - 1))))
        return px

    if route_xy is not None and len(route_xy):
        _polyline(out, clip_pts(route_xy), (255, 220, 0), 1)
    if traj_xy is not None and len(traj_xy):
        _polyline(out, clip_pts(traj_xy), (0, 255, 255), 2)
    if target_xy is not None:
        c, r = ego_to_bev_px(float(target_xy[0]), float(target_xy[1]), grid, scale)
        if 0 <= c < w and 0 <= r < h:
            cv2.drawMarker(out, (c, r), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
    ego = ego_to_bev_px(0.0, 0.0, grid, scale)
    cv2.drawMarker(out, (int(np.clip(ego[0], 0, w - 1)), int(np.clip(ego[1], 0, h - 1))),
                   (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
    return out


def draw_occ_3d_with_sweep(
    occ: np.ndarray,
    sweep: np.ndarray | None = None,
    traj_xy: np.ndarray | None = None,
    route_xy: np.ndarray | None = None,
    target_xy: np.ndarray | None = None,
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

    def clip_occ(xy):
        pts = []
        for x, y in xy:
            c, r = ego_to_occ_xy_px(float(x), float(y), grid, scale)
            if 0 <= c < roi.shape[1] and 0 <= r < roi.shape[0]:
                pts.append((c, r))
        return pts

    if route_xy is not None and len(route_xy):
        _polyline(roi, clip_occ(route_xy), (255, 220, 0), 1)
    if traj_xy is not None and len(traj_xy):
        _polyline(roi, clip_occ(traj_xy), (0, 255, 255), 2)
    if target_xy is not None:
        c, r = ego_to_occ_xy_px(float(target_xy[0]), float(target_xy[1]), grid, scale)
        if 0 <= c < roi.shape[1] and 0 <= r < roi.shape[0]:
            cv2.drawMarker(roi, (c, r), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
    return panel
