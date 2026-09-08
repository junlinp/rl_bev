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
| [minikeyvalue_client.py](minikeyvalue_client.py) | Client for storing/retrieving dataset samples in a [minikeyvalue](https://github.com/geohot/minikeyvalue) store |
| [kv_volume_server.py](kv_volume_server.py) | Pure-Python (Windows-compatible) minikeyvalue volume server |
| [start_kv.py](start_kv.py) / [start_kv.ps1](start_kv.ps1) / [stop_kv.py](stop_kv.py) | Start/stop the local minikeyvalue master + volume servers |
| [migrate_to_kv.py](migrate_to_kv.py) | Migrates existing on-disk dataset samples into the minikeyvalue store |
| [test_kv.py](test_kv.py) | Basic tests for the minikeyvalue client/server |
| [docs/minikeyvalue_dataset_design.md](docs/minikeyvalue_dataset_design.md) | Design notes for the minikeyvalue-backed dataset storage |

## Requirements

- CARLA 0.9.14+ server running
- Python 3.10+
- `carla` Python API
- `opencv-python`, `numpy`
- `torch`, `torchvision`

## Quick Start

```bash
# 1. (Optional) Start the local minikeyvalue store for dataset storage
python start_kv.py

# 2. Collect training data (uses CARLA depth+seg sensors for ground truth)
python collect_data.py --samples 2000

# 3. Train the stereo-to-BEV model
python train_bev.py --data bev_stereo_data --epochs 50

# 4. Run inference (edit run_bev.py to set mode="model" and model_checkpoint=...)
python run_bev.py
```
