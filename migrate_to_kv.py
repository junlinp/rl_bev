"""Migrate existing bev_data/ .npz files to minikeyvalue."""

import os
import io
import argparse
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(__file__))
from minikeyvalue_client import MiniKV


def migrate(kv_url: str, data_dir: str = "bev_data"):
    kv = MiniKV(kv_url)

    for split in ["train", "val"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"[Migrate] Skipping {split}/ (not found)")
            continue

        files = sorted(f for f in os.listdir(split_dir) if f.endswith(".npz"))
        print(f"[Migrate] {split}: {len(files)} files → {kv_url}/{split}/")

        for i, fname in enumerate(files):
            path = os.path.join(split_dir, fname)
            # load and re-save to get raw bytes
            arrays = np.load(path)
            buf = io.BytesIO()
            np.savez_compressed(buf, **{k: arrays[k] for k in arrays.files})
            key = f"/{split}/sample_{i:06d}"
            kv.put(key, buf.getvalue())
            if (i + 1) % 50 == 0:
                print(f"  {split}: {i + 1}/{len(files)}")

        print(f"[Migrate] {split}: done ({len(files)} samples)")

    print(f"[Migrate] Train count: {kv.count('/train/')}")
    print(f"[Migrate] Val count:   {kv.count('/val/')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate bev_data to minikeyvalue")
    parser.add_argument("--kv-url", required=True, help="minikeyvalue URL")
    parser.add_argument("--data", default="bev_data", help="Source data directory")
    args = parser.parse_args()
    migrate(args.kv_url, args.data)
