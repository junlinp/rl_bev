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
| [collect_data.py](collect_data.py) | Collects stereo RGB + BEV ground truth samples from CARLA, with NPC traffic spawning and quality filtering |
| [train_bev.py](train_bev.py) | Trains the `StereoBEVModel` on collected data; logs to TensorBoard |
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
# 1. Collect training data (uses CARLA depth+seg sensors for ground truth)
python collect_data.py --samples 2000
# Or write samples to an already-running minikeyvalue server:
python collect_data.py --kv-url http://localhost:3000 --samples 2000

# 2. Train the stereo-to-BEV model
python train_bev.py --data bev_data --epochs 50
# Or read the dataset from minikeyvalue:
python train_bev.py --kv-url http://localhost:3000 --epochs 50

# 3. Run inference (edit run_bev.py to set mode="model" and model_checkpoint=...)
python run_bev.py
```

This repo is a **client only**. Bring up minikeyvalue (master + volumes) separately, then pass `--kv-url`.
