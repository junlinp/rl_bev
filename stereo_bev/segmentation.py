"""CARLA semantic segmentation sensor and label mapping."""

import numpy as np

# CARLA semantic segmentation tag indices (0.9.14+)
CARLA_TAGS = {
    0: "Unlabeled",
    1: "Building",
    2: "Fence",
    3: "Other",
    4: "Pedestrian",
    5: "Pole",
    6: "RoadLine",
    7: "Road",
    8: "SideWalk",
    9: "Vegetation",
    10: "Vehicle",
    11: "Wall",
    12: "TrafficSign",
    13: "Sky",
    14: "Ground",
    15: "Bridge",
    16: "RailTrack",
    17: "GuardRail",
    18: "TrafficLight",
    19: "Static",
    20: "Dynamic",
    21: "Water",
    22: "Terrain",
}

# Collapsed classes for BEV query (keeps the useful ones)
BEV_CLASSES = {
    0: "empty",
    1: "road",
    2: "sidewalk",
    3: "vehicle",
    4: "pedestrian",
    5: "building",
    6: "vegetation",
    7: "terrain",
    8: "pole_sign",
    9: "other",
}

NUM_BEV_CLASSES = len(BEV_CLASSES)

# Mapping from CARLA tag → BEV class
_TAG_TO_BEV = {
    0: 9,   # Unlabeled → other
    1: 5,   # Building
    2: 9,   # Fence → other
    3: 9,   # Other
    4: 4,   # Pedestrian
    5: 8,   # Pole
    6: 1,   # RoadLine → road
    7: 1,   # Road
    8: 2,   # SideWalk
    9: 6,   # Vegetation
    10: 3,  # Vehicle
    11: 5,  # Wall → building
    12: 8,  # TrafficSign
    13: 9,  # Sky → other (not in BEV)
    14: 7,  # Ground → terrain
    15: 5,  # Bridge → building
    16: 9,  # RailTrack → other
    17: 9,  # GuardRail → other
    18: 8,  # TrafficLight
    19: 9,  # Static → other
    20: 3,  # Dynamic → vehicle (approx)
    21: 9,  # Water → other
    22: 7,  # Terrain
}

_TAG_MAP = np.zeros(256, dtype=np.uint8)
for k, v in _TAG_TO_BEV.items():
    _TAG_MAP[k] = v


def remap_segmentation(seg_raw: np.ndarray) -> np.ndarray:
    """Remap CARLA tag image → collapsed BEV class indices (uint8)."""
    return _TAG_MAP[seg_raw.astype(np.uint8)]


# ── Color palette for visualization ──
BEV_COLORS = np.array([
    [0,   0,   0],     # 0 empty        — black
    [128, 128, 128],   # 1 road         — grey
    [200, 200, 100],   # 2 sidewalk     — tan
    [0,   0,   200],   # 3 vehicle      — red
    [200, 0,   200],   # 4 pedestrian   — magenta
    [100, 50,   50],   # 5 building     — brown
    [0,  128,   0],    # 6 vegetation   — green
    [80,  60,   40],   # 7 terrain      — earth
    [255, 200,   0],   # 8 pole/sign    — yellow
    [60,  60,   60],   # 9 other        — dark grey
], dtype=np.uint8)


def colorize_bev(bev_classes: np.ndarray) -> np.ndarray:
    """Map (H, W) class indices → (H, W, 3) BGR image."""
    return BEV_COLORS[bev_classes.clip(0, NUM_BEV_CLASSES - 1)]
