# Stereo BEV Perception Pipeline for CARLA

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  CARLA Server                                                    │
│  ┌────────────┐  ┌────────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Left RGB   │  │ Right RGB  │  │ Depth    │  │ Seg      │    │
│  │ 960×540    │  │ 960×540    │  │ 960×540  │  │ 960×540  │    │
│  └─────┬──────┘  └─────┬──────┘  └────┬─────┘  └────┬─────┘    │
└────────┼───────────────┼──────────────┼─────────────┼───────────┘
         │               │              │             │
         │  Model Input  │              │  GT only    │  GT only
         ▼               ▼              │             │
  ┌─────────────────────────────┐       │             │
  │     StereoBEVModel          │       │             │
  │                             │       │             │
  │  ┌───────────────────────┐  │       │             │
  │  │  ResNet-18 Backbone   │  │       │             │
  │  │  (shared L + R)       │  │       │             │
  │  └───────────┬───────────┘  │       │             │
  │              │               │       │             │
  │  ┌───────────┴───────────┐  │       │             │
  │  │  Stereo Depth Pred    │  │       │             │
  │  │  (cost volume → D)    │  │       │             │
  │  └───────────┬───────────┘  │       │             │
  │              │               │       │             │
  │  ┌───────────┴───────────┐  │       │             │
  │  │  LSS Lift + Splat     │  │       │             │
  │  │  (features → BEV)     │  │       │             │
  │  └───────────┬───────────┘  │       │             │
  │              │               │       │             │
  │  ┌───────────┴───────────┐  │       │             │
  │  │  BEV Encoder (CNN)    │  │       │             │
  │  └───────────┬───────────┘  │       │             │
  │              │               │       │             │
  │     ┌────────┴────────┐     │       │             │
  │     ▼                 ▼     │       │             │
  │  Seg Head         Occ Head  │       │             │
  │  (H,W) class      (H,W) 0/1│       │             │
  └─────────────────────────────┘       │             │
                                        ▼             ▼
                               ┌──────────────────────────┐
                               │  BEV Ground Truth        │
                               │  (geometric projection)  │
                               └──────────────────────────┘
```

**Model input**: left RGB + right RGB only (no depth/seg at inference)

**Training**: depth + seg sensors provide BEV ground truth for supervision

## Quick Start

```bash
# 1. Collect training data (uses CARLA depth+seg for GT)
python train_bev.py collect --samples 2000

# 2. Train the stereo-to-BEV model
python train_bev.py train --data bev_stereo_data --epochs 50

# 3. Run inference with trained model (stereo RGB only)
# Edit run_bev.py: mode="model", model_checkpoint="stereo_bev_model.pth"
python run_bev.py

# Or run geometric baseline (uses depth+seg sensors directly)
python run_bev.py
```

## Module Reference

| Module | Purpose |
|--------|---------|
| `calibration.py` | Camera intrinsics (K matrix) from FOV + resolution |
| `depth.py` | CARLA depth sensor decoding, depth→point cloud |
| `segmentation.py` | CARLA tag→BEV class remapping (10 classes) |
| `bev_grid.py` | Axis-aligned voxel grid, lifting, projection |
| `query_heads.py` | StereoBEVModel (LSS) + geometric baseline heads |
| `camera_rig.py` | Stereo rig: left/right RGB + depth + seg on left |
| `visualize.py` | BEV map rendering, depth heatmaps, legend |
| `main.py` | Full perception loop (geometric or model mode) |

## Model Architecture (LSS)

| Stage | Description |
|-------|-------------|
| **Backbone** | Shared ResNet-18, outputs /16 features (256ch) |
| **Depth predictor** | Stereo cost volume → depth distribution (64 bins) |
| **Lift** | outer product: feat ⊗ depth_prob → frustum features |
| **Splat** | Project frustum → BEV grid via scatter-add |
| **BEV encoder** | 3-layer CNN refines BEV features (64ch) |
| **Seg head** | Conv → per-cell class logits |
| **Occ head** | Conv → per-cell occupancy logit |

## BEV Grid Parameters

| Param | Default | Effect |
|-------|---------|--------|
| `x/y_range` | ±5m | BEV extent (10m × 10m) |
| `z_range` | 0–5m | Height band for voxelization |
| `voxel_size` | 0.1m | Cell resolution (100×100×50 grid) |
| `max_depth` | 80m | Far clip for depth sensor |

## Classes

| ID | Class | Color |
|----|-------|-------|
| 0 | empty | black |
| 1 | road | grey |
| 2 | sidewalk | tan |
| 3 | vehicle | red |
| 4 | pedestrian | magenta |
| 5 | building | brown |
| 6 | vegetation | green |
| 7 | terrain | earth |
| 8 | pole/sign | yellow |
| 9 | other | dark grey |

## Camera Intrinsics (960×540 @ FOV=90°, baseline=0.12m)

```
focal_px = 480.0

K = [[ 480.0,    0.0,  480.0],
     [   0.0,  480.0,  270.0],
     [   0.0,    0.0,    1.0]]

distortion = [0, 0, 0, 0, 0]  (CARLA is ideal pinhole)
baseline   = 0.12 m
```

## Requirements

- CARLA 0.9.14+ server running
- Python 3.10+
- `carla` Python API
- `opencv-python`, `numpy`
- `torch`, `torchvision` (for StereoBEVModel)
