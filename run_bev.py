"""Entry point: run the BEV perception pipeline against a CARLA server."""

from stereo_bev.main import run
from stereo_bev.bev_grid import DEFAULT_X_RANGE, DEFAULT_Y_RANGE, DEFAULT_Z_RANGE, DEFAULT_VOXEL
from stereo_bev.calibration import DEFAULT_PITCH_DEG

if __name__ == "__main__":
    run(
        host="localhost",
        port=2000,
        duration_sec=120.0,
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
        mode="geometric",
        model_checkpoint=None,
    )
