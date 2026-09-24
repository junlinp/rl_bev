"""Entry point: run the BEV perception pipeline against a CARLA server."""

import argparse

from stereo_bev.main import run
from stereo_bev.bev_grid import DEFAULT_X_RANGE, DEFAULT_Y_RANGE, DEFAULT_Z_RANGE, DEFAULT_VOXEL
from stereo_bev.calibration import DEFAULT_PITCH_DEG

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stereo BEV perception in CARLA")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--mode", default="geometric", choices=["geometric", "model"])
    p.add_argument("--model-checkpoint", default=None)
    p.add_argument(
        "--no-occ-fusion", action="store_true",
        help="Disable multi-frame occupancy merge (single-frame lift only)",
    )
    args = p.parse_args()
    run(
        host=args.host,
        port=args.port,
        duration_sec=args.duration,
        image_w=960,
        image_h=540,
        fov=90.0,
        baseline=0.12,
        fps=15.0,
        bev_x_range=DEFAULT_X_RANGE,
        bev_y_range=DEFAULT_Y_RANGE,
        bev_z_range=DEFAULT_Z_RANGE,
        bev_voxel=DEFAULT_VOXEL,
        pitch_deg=DEFAULT_PITCH_DEG,
        max_depth=80.0,
        mode=args.mode,
        model_checkpoint=args.model_checkpoint,
        use_occ_fusion=not args.no_occ_fusion,
    )
