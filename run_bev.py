"""Entry point: run the BEV perception pipeline against a CARLA server."""

from stereo_bev.main import run

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
        bev_range_xy=5.0,
        bev_z_range=5.0,
        bev_voxel=0.1,
        max_depth=80.0,
        mode="geometric",               # "geometric" or "model"
        model_checkpoint=None,          # e.g. "stereo_bev_model.pth"
    )
