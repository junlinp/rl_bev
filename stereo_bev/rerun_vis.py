"""Live 3D occupancy visualization via the rerun-sdk viewer."""

from __future__ import annotations

import os
import shutil
import socket
import sys
from pathlib import Path

import numpy as np

from .segmentation import BEV_CLASSES, BEV_COLORS
from .vehicle_body import MODEL3_HEIGHT, MODEL3_LENGTH, MODEL3_WIDTH

try:
    import rerun as rr
    import rerun.blueprint as rrb
except ImportError:  # pragma: no cover
    rr = None
    rrb = None


def is_available() -> bool:
    return rr is not None


def find_rerun_executable() -> str | None:
    """Locate the native viewer even when the venv Scripts dir is not on PATH."""
    name = "rerun.exe" if os.name == "nt" else "rerun"
    candidates: list[Path] = [
        Path(sys.executable).resolve().parent / name,
        Path(sys.executable).resolve().parent / "Scripts" / name,
    ]
    try:
        import rerun_cli  # type: ignore

        candidates.insert(0, Path(rerun_cli.__file__).resolve().parent / name)
    except ImportError:
        pass
    for path in candidates:
        if path.is_file():
            return str(path)
    return shutil.which("rerun")


def _viewer_listening(port: int = 9876) -> bool:
    sock = socket.socket()
    sock.settimeout(0.4)
    try:
        sock.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _xyz_from_xy(xy: np.ndarray | None, z: float = 0.7) -> np.ndarray | None:
    if xy is None:
        return None
    pts = np.asarray(xy, dtype=np.float32)
    if pts.size == 0:
        return None
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] < 2:
        return None
    if pts.shape[1] == 2:
        zcol = np.full((pts.shape[0], 1), z, dtype=np.float32)
        pts = np.concatenate([pts, zcol], axis=1)
    return np.ascontiguousarray(pts[:, :3])


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        return np.ascontiguousarray(img[:, :, ::-1])
    return img


POLE_BEV_CLASS = 8
POLE_COLOR = tuple(int(c) for c in BEV_COLORS[POLE_BEV_CLASS])


def default_blueprint():
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="world",
                name="3D occupancy",
                contents="world/**",
                line_grid=rrb.LineGrid3D(visible=True, spacing=1.0),
                eye_controls=rrb.EyeControls3D(
                    position=[-6.0, -14.0, 9.0],
                    look_target=[8.0, 0.0, 0.6],
                    eye_up=[0.0, 0.0, 1.0],
                ),
            ),
            rrb.Vertical(
                rrb.Spatial2DView(origin="camera/rgb", name="RGB", contents="camera/rgb"),
                rrb.Spatial2DView(origin="camera/depth", name="depth", contents="camera/depth"),
                rrb.Spatial2DView(origin="bev", name="BEV", contents="bev"),
                rrb.TimeSeriesView(origin="metrics", name="metrics", contents="metrics/**"),
                row_shares=[1.1, 1.0, 1.2, 0.9],
            ),
            column_shares=[2.4, 1.0],
        ),
        rrb.BlueprintPanel(state=rrb.PanelState.Collapsed),
        rrb.SelectionPanel(state=rrb.PanelState.Collapsed),
        rrb.TimePanel(state=rrb.PanelState.Collapsed),
        collapse_panels=True,
    )


class RerunOccViewer:
    """Stream occupancy voxels, plans, and cameras into a Rerun viewer."""

    def __init__(
        self,
        grid,
        application_id: str = "nmpc_occupancy",
        spawn: bool = True,
    ):
        if rr is None:
            raise ImportError("rerun-sdk is not installed; pip install rerun-sdk")
        self.grid = grid
        self._started = False
        blueprint = default_blueprint()
        rr.init(application_id, spawn=False, default_blueprint=blueprint)
        if spawn:
            exe = find_rerun_executable()
            if exe is None:
                raise FileNotFoundError(
                    "Rerun Viewer executable not found. "
                    "pip install rerun-sdk or add rerun.exe to PATH."
                )
            if _viewer_listening():
                rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy", default_blueprint=blueprint)
            else:
                rr.spawn(
                    executable_path=exe,
                    default_blueprint=blueprint,
                    memory_limit="2GiB",
                    server_memory_limit="512MB",
                )
        self._log_static()
        self._started = True

    def _log_static(self) -> None:
        x0, x1 = self.grid.x_range
        y0, y1 = self.grid.y_range
        z0, z1 = self.grid.z_range
        rr.log("world", rr.ViewCoordinates.FLU, static=True)
        rr.log(
            "world",
            rr.AnnotationContext([
                rr.AnnotationInfo(
                    id=int(i),
                    label=name,
                    color=tuple(int(c) for c in BEV_COLORS[i]),
                )
                for i, name in BEV_CLASSES.items()
            ]),
            static=True,
        )
        rr.log(
            "world/volume",
            rr.Boxes3D(
                centers=[[0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)]],
                half_sizes=[[0.5 * (x1 - x0), 0.5 * (y1 - y0), 0.5 * (z1 - z0)]],
                colors=[(70, 70, 80)],
                fill_mode=rr.components.FillMode.MajorWireframe,
            ),
            static=True,
        )
        rr.log(
            "world/ego",
            rr.Boxes3D(
                centers=[[0.0, 0.0, 0.5 * MODEL3_HEIGHT]],
                half_sizes=[[0.5 * MODEL3_LENGTH, 0.5 * MODEL3_WIDTH, 0.5 * MODEL3_HEIGHT]],
                colors=[(230, 230, 230)],
                fill_mode=rr.components.FillMode.MajorWireframe,
            ),
            static=True,
        )
        rr.log("world/ego_heading", rr.Arrows3D(
            origins=[[0.0, 0.0, 0.8]],
            vectors=[[2.0, 0.0, 0.0]],
            colors=[(255, 255, 255)],
        ), static=True)
        rr.log("metrics/speed", rr.SeriesLines(names=["speed m/s"], colors=[(80, 180, 255)]), static=True)
        rr.log("metrics/accel", rr.SeriesLines(names=["a m/s^2"], colors=[(80, 220, 80)]), static=True)
        rr.log("metrics/steer", rr.SeriesLines(names=["delta rad"], colors=[(255, 160, 40)]), static=True)
        rr.log("metrics/clearance", rr.SeriesLines(names=["dmin m"], colors=[(255, 80, 80)]), static=True)

    def _log_voxels(
        self,
        path: str,
        mask: np.ndarray | None,
        *,
        colors: np.ndarray | None = None,
        default_color: tuple[int, int, int] = (180, 180, 180),
        opacity: float = 0.9,
        max_voxels: int = 60000,
    ) -> None:
        if mask is None or not np.any(mask):
            rr.log(path, rr.Clear(recursive=False))
            return
        zi, yi, xi = np.nonzero(mask > 0)
        n = int(xi.size)
        if n > max_voxels:
            sel = np.linspace(0, n - 1, max_voxels).astype(np.int32)
            xi, yi, zi = xi[sel], yi[sel], zi[sel]
        indices = np.stack([xi, yi, zi], axis=1).astype(np.int32)
        if colors is None:
            rgb = np.broadcast_to(
                np.array(default_color, dtype=np.uint8), (indices.shape[0], 3),
            ).copy()
        else:
            rgb = np.asarray(colors, dtype=np.uint8)
            if rgb.ndim == 4:
                rgb = rgb[zi, yi, xi]
            elif rgb.ndim == 3 and rgb.shape[:3] == mask.shape:
                rgb = rgb[zi, yi, xi]
            elif rgb.ndim == 3:
                rgb = rgb[yi, xi]
            elif rgb.shape[0] != indices.shape[0]:
                rgb = np.broadcast_to(
                    np.array(default_color, dtype=np.uint8), (indices.shape[0], 3),
                ).copy()
        vs = float(self.grid.voxel_size)
        x = self.grid.x_range[0] + (xi.astype(np.float32) + 0.5) * vs
        y = self.grid.y_range[0] + (yi.astype(np.float32) + 0.5) * vs
        z = self.grid.z_range[0] + (zi.astype(np.float32) + 0.5) * vs
        radii = np.full(indices.shape[0], 0.55 * vs, dtype=np.float32)
        if rgb.ndim == 2 and rgb.shape[0] == indices.shape[0]:
            pole = np.all(rgb == np.array(POLE_COLOR, dtype=np.uint8), axis=1)
            radii[pole] = 0.16
        rr.log(
            path,
            rr.Points3D(
                np.stack([x, y, z], axis=1),
                colors=rgb,
                radii=radii,
            ),
        )

    def log_frame(
        self,
        frame: int,
        *,
        occ: np.ndarray,
        sweep: np.ndarray | None = None,
        bev_classes: np.ndarray | None = None,
        traj_xy: np.ndarray | None = None,
        route_xy: np.ndarray | None = None,
        target_xy: np.ndarray | None = None,
        class_histogram: np.ndarray | None = None,
        voxel_class: np.ndarray | None = None,
        rgb_bgr: np.ndarray | None = None,
        depth: np.ndarray | None = None,
        bev_bgr: np.ndarray | None = None,
        speed: float = 0.0,
        accel: float = 0.0,
        steer: float = 0.0,
        clearance: float = 0.0,
        max_depth: float = 40.0,
    ) -> None:
        if not self._started:
            return
        rr.set_time("frame", sequence=int(frame))

        colors = None
        if voxel_class is not None and np.asarray(voxel_class).shape == np.asarray(occ).shape:
            cls = np.clip(np.asarray(voxel_class), 0, len(BEV_COLORS) - 1).astype(np.int32)
            colors = BEV_COLORS[cls]
        elif bev_classes is not None:
            cls = np.clip(np.asarray(bev_classes), 0, len(BEV_COLORS) - 1).astype(np.int32)
            colors = BEV_COLORS[cls]
        self._log_voxels("world/occupancy", occ, colors=colors, opacity=0.92, max_voxels=25000)
        self._log_voxels(
            "world/sweep", sweep,
            default_color=(0, 220, 255), opacity=0.35, max_voxels=8000,
        )

        traj = _xyz_from_xy(traj_xy, z=0.8)
        if traj is not None and len(traj) >= 2:
            rr.log(
                "world/nmpc",
                rr.LineStrips3D([traj], colors=[(255, 220, 0)], radii=0.06),
            )
        else:
            rr.log("world/nmpc", rr.Clear(recursive=False))

        route = _xyz_from_xy(route_xy, z=0.35)
        if route is not None and len(route) >= 2:
            rr.log(
                "world/route",
                rr.LineStrips3D([route], colors=[(0, 220, 255)], radii=0.05),
            )
        else:
            rr.log("world/route", rr.Clear(recursive=False))

        target = _xyz_from_xy(target_xy, z=0.9)
        if target is not None:
            rr.log(
                "world/target",
                rr.Points3D(target, colors=[(255, 40, 40)], radii=0.25),
            )
        else:
            rr.log("world/target", rr.Clear(recursive=False))

        if rgb_bgr is not None:
            rr.log("camera/rgb", rr.Image(_bgr_to_rgb(rgb_bgr[::2, ::2]), color_model="RGB"))
        if depth is not None:
            depth_s = np.asarray(depth, dtype=np.float32)[::2, ::2]
            rr.log(
                "camera/depth",
                rr.DepthImage(depth_s, meter=1.0, depth_range=[0.0, float(max_depth)]),
            )
        if bev_bgr is not None:
            bev = bev_bgr[::2, ::2] if min(bev_bgr.shape[:2]) > 80 else bev_bgr
            rr.log("bev", rr.Image(_bgr_to_rgb(bev), color_model="RGB"))

        rr.log("metrics/speed", rr.Scalars(float(speed)))
        rr.log("metrics/accel", rr.Scalars(float(accel)))
        rr.log("metrics/steer", rr.Scalars(float(steer)))
        rr.log("metrics/clearance", rr.Scalars(float(clearance)))

    def close(self) -> None:
        if not self._started:
            return
        self._started = False
