"""Export all samples from minikeyvalue to bev_data/ on disk."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from minikeyvalue_client import MiniKV

kv = MiniKV("http://localhost:3000")

for split in ["train", "val"]:
    out_dir = os.path.join(os.path.dirname(__file__), "bev_data", split)
    os.makedirs(out_dir, exist_ok=True)

    keys = kv.list_keys(f"/{split}/")
    print(f"[Export] {split}: {len(keys)} keys in KV", flush=True)

    for i, key in enumerate(keys):
        fname = key.lstrip("/").replace("/", "_") + ".npz"
        out_path = os.path.join(out_dir, fname)
        if os.path.exists(out_path):
            continue
        data = kv.get(key)
        with open(out_path, "wb") as f:
            f.write(data)
        if (i + 1) % 50 == 0:
            print(f"  [{split}] {i+1}/{len(keys)}", flush=True)

    print(f"[Export] {split} done: {len(os.listdir(out_dir))} files on disk", flush=True)

print("[Export] All done.", flush=True)
