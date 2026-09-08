"""Visualization utilities for BEV perception outputs."""

import cv2
import numpy as np

from .segmentation import BEV_COLORS, NUM_BEV_CLASSES, BEV_CLASSES


def draw_bev_map(
    bev_classes: np.ndarray,
    occupancy: np.ndarray | None = None,
    scale: int = 4,
) -> np.ndarray:
    """
    Render BEV semantic map (with optional occupancy overlay) as an image.

    Args:
        bev_classes: (H, W) class indices
        occupancy:   (H, W) binary or None
        scale:       pixel scale-up factor

    Returns BGR image (H*scale, W*scale, 3).
    """
    h, w = bev_classes.shape
    img = BEV_COLORS[bev_classes.clip(0, NUM_BEV_CLASSES - 1)]  # (H, W, 3) RGB

    if occupancy is not None:
        # darken unoccupied cells
        occ_mask = occupancy.astype(bool)
        img[~occ_mask] = (img[~occ_mask] * 0.2).astype(np.uint8)

    # scale up
    img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    # ego marker (center)
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
