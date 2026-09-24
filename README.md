# rl_bev

Stereo bird's-eye-view (BEV) perception pipeline for [CARLA](https://carla.org/), with data collection, training, and a minikeyvalue-backed dataset store.

## Overview

This project builds a BEV occupancy/segmentation map of a CARLA scene from a stereo camera rig only (left + right RGB). Depth and segmentation sensors are used solely to generate ground truth during data collection and training — inference runs on stereo RGB alone.

See [stereo_bev/README.md](stereo_bev/README.md) for the model architecture, module reference, and BEV grid parameters.

## Components

| Path | Purpose |
|------|---------|
| [stereo_bev/](stereo_bev) | Core pipeline: camera rig, calibration, depth/segmentation decoding, BEV grid, LSS model, visualization |
| [run_bev.py](run_bev.py) | Entry point to run the perception pipeline (geometric baseline or trained model) against a live CARLA server |
| [run_planner.py](run_planner.py) | Live CARLA stereo vision-action: depth/seg/occupancy heads plus a control head on the 5 s target pose. `run_nmpc.py` is an alias. |
| [collect_data.py](collect_data.py) | Collects stereo RGB + BEV GT + expert autopilot control. `--trajectories` saves consecutive episodes. |
| [train_bev.py](train_bev.py) | Trains perception heads + optional imitation on expert control |
| [train_control_rl.py](train_control_rl.py) | PPO on the control head in live CARLA (perception frozen) |
| [minikeyvalue_client.py](minikeyvalue_client.py) | HTTP client for an external [minikeyvalue](https://github.com/geohot/minikeyvalue) dataset store |
| [migrate_to_kv.py](migrate_to_kv.py) | Migrates existing on-disk dataset samples into the minikeyvalue store |
| [test_kv.py](test_kv.py) | Client smoke test against a running minikeyvalue server |
| [docs/minikeyvalue_dataset_design.md](docs/minikeyvalue_dataset_design.md) | Design notes for the minikeyvalue-backed dataset storage |

## Requirements

- CARLA 0.9.14+ server running
- Python 3.10+
- `carla` Python API
- `opencv-python`, `numpy`
- `torch`, `torchvision`

## Quick Start

```bash
# 1. Collect expert trajectories (Traffic Manager autopilot) + perception GT
python collect_data.py --trajectories --samples 2000 --episode-len 200
# Or iid perception samples (occupancy filter on, sequence not preserved):
python collect_data.py --samples 2000

# 2. Imitation + perception
python train_bev.py --data bev_data --epochs 50

# 3. PPO on the control head only (live CARLA, perception frozen)
python train_control_rl.py --model-checkpoint stereo_bev_model.pth --updates 50

# 4. Closed-loop
python run_planner.py --closed-loop --mode model --model-checkpoint stereo_bev_control_rl.pth
```

This repo is a **client only**. Bring up minikeyvalue (master + volumes) separately, then pass `--kv-url`.
