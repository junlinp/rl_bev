# minikeyvalue Dataset Backend Design

## Problem

Current dataset storage is flat `.npz` files on disk (`bev_data/train/*.npz`, `bev_data/val/*.npz`).
This works for small datasets but has limitations:
- No append — must know total count upfront for train/val split
- No distributed access — collector and trainer must share a filesystem
- No replication — single point of failure

## Solution

Use [minikeyvalue](https://github.com/geohot/minikeyvalue) as the dataset store.
It's a ~1000-line Go HTTP key-value store with LevelDB index + nginx volume servers.
Optimized for 1MB–1GB blobs (our `.npz` samples are ~2–10MB each, perfect fit).

## Architecture

```
┌──────────────┐     PUT /train/sample_000042      ┌──────────────────┐
│ collect_data │ ──────────────────────────────────>│                  │
│    .py       │                                    │  minikeyvalue    │
└──────────────┘                                    │  (Go server)     │
                                                    │                  │
┌──────────────┐     GET /train/sample_000042       │  master :3000    │
│  train_bev   │ <──────────────────────────────────│  vol1   :3001    │
│    .py       │     GET /train?list                │  vol2   :3002    │
└──────────────┘                                    └──────────────────┘
```

## Key Schema

```
/train/sample_{id:06d}    # training sample (npz bytes)
/val/sample_{id:06d}      # validation sample (npz bytes)
/meta/total_train          # metadata: number of training samples
/meta/total_val            # metadata: number of validation samples
```

- Keys are lexicographically sortable, so `?list` returns them in order
- Metadata keys track total counts so the trainer can pre-allocate

## Data Format

Each value is the raw `.npz` bytes (numpy compressed archive).
Fields inside: `left_rgb, right_rgb, depth_gt, seg_gt, occ_gt, bev_seg_gt, K` (unchanged from current).

## Components

### 1. `minikeyvalue_client.py` — Thin Python client

```python
class MiniKV:
    def __init__(self, url: str = "http://localhost:3000"):
        self.url = url

    def put(self, key: str, data: bytes) -> bool: ...
    def get(self, key: str) -> bytes: ...
    def delete(self, key: str) -> bool: ...
    def list(self, prefix: str) -> list[str]: ...
    def count(self, prefix: str) -> int: ...
```

Uses `urllib.request` (stdlib only, no deps).
- `put`: HTTP PUT, returns True on 201
- `get`: HTTP GET (follows 302 redirect to volume), returns bytes
- `list`: GET `{prefix}?list`, parses newline-separated keys
- `count`: len of list

### 2. `collect_data.py` — Modified to PUT to minikeyvalue

Changes:
- Add `--kv-url` arg (default `http://localhost:3000`)
- If `--kv-url` set: PUT each sample to minikeyvalue, increment meta counters
- If not set: fall back to current filesystem behavior (backward compatible)
- Replaces `np.savez_compressed(out_path, ...)` with `kv.put(key, npz_bytes)`
- npz bytes produced in-memory via `io.BytesIO` + `np.savez_compressed`

Key assignment:
```python
if split == "train":
    key = f"/train/sample_{train_count:06d}"
    train_count += 1
else:
    key = f"/val/sample_{val_count:06d}"
    val_count += 1
```

### 3. `train_bev.py` — Modified dataset to GET from minikeyvalue

Changes:
- Add `--kv-url` arg
- New `KVBEVDataset` class that implements `torch.utils.data.Dataset`:
  - `__init__`: calls `kv.list("/train")` or `kv.list("/val")` to get key list
  - `__len__`: `len(self.keys)`
  - `__getitem__`: `kv.get(key)` → `np.load(io.BytesIO(data))` → tensors
- If `--kv-url` set: use `KVBEVDataset`
- If not set: use current `StereoBEVDataset` (backward compatible)

### 4. Server setup script — `start_kv.sh`

```bash
#!/bin/bash
# Build minikeyvalue (one-time)
# cd minikeyvalue && go build -o mkv . && cd ..

VOLUME_DIR=${1:-/tmp/bev_kv}
mkdir -p $VOLUME_DIR/vol1

# Start volume server
PORT=3001 ./mkv/volume $VOLUME_DIR/vol1 &

# Start master
./mkv/mkv -volumes localhost:3001 -db $VOLUME_DIR/indexdb server &
```

## Append Workflow

The key advantage: **collect_data.py can append new data at any time.**

1. Start minikeyvalue server (one-time)
2. Run `collect_data.py --kv-url http://localhost:3000 --num-samples 100`
3. Run `collect_data.py --kv-url http://localhost:3000 --num-samples 100` again
4. Total dataset is now 200 samples — no conflict
5. Run `train_bev.py --kv-url http://localhost:3000 --epochs 50`
6. Trainer sees all 200 samples

## Migration from existing data

```python
# migrate_to_kv.py
import os, io, numpy as np
from minikeyvalue_client import MiniKV

kv = MiniKV("http://localhost:3000")

for split in ["train", "val"]:
    data_dir = f"bev_data/{split}"
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npz"))
    for i, f in enumerate(files):
        buf = io.BytesIO()
        np.savez_compressed(buf, **np.load(os.path.join(data_dir, f)))
        kv.put(f"/{split}/sample_{i:06d}", buf.getvalue())
    print(f"Migrated {len(files)} {split} samples")
```

## Dependencies

- **minikeyvalue**: Go binary, build once with `go build`
- **Python**: stdlib only (`urllib.request`, `io`, `numpy`)
- **nginx**: bundled in minikeyvalue's `volume` script

## Out of scope (future)

- Multi-machine replication (set `-replicas 3` and multiple volume servers)
- Streaming prefetch / caching (the tinygrad shared-memory pattern)
- S3-compatible client (minikeyvalue supports a subset of S3)
