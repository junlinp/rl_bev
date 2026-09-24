"""Compatibility alias for ``run_planner.py`` (stereo vision-action).

Prefer ``python run_planner.py --closed-loop --mode model --model-checkpoint stereo_bev_model.pth``.
"""

from run_planner import *  # noqa: F403
from run_planner import main, run, _lane_offset_xy

__all__ = ["main", "run", "_lane_offset_xy"]


if __name__ == "__main__":
    main()
