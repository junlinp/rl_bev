"""Client smoke test against an already-running minikeyvalue server.

Uses /client_test/ keys only — does not touch /train/ or /val/.
Requires: minikeyvalue master at http://localhost:3000
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from minikeyvalue_client import MiniKV

PREFIX = "/client_test/"
kv = MiniKV("http://localhost:3000")

print("=== Cleanup ===")
for key in kv.list_keys(PREFIX):
    kv.delete(key)
print("Cleaned previous /client_test/ keys")

print("=== PUT test sample ===")
ok = kv.put_npz(
    f"{PREFIX}sample_000000",
    left_rgb=np.zeros((540, 960, 3), dtype=np.uint8),
    right_rgb=np.zeros((540, 960, 3), dtype=np.uint8),
    depth_gt=np.ones((100, 100), dtype=np.float32),
    seg_gt=np.zeros((100, 100), dtype=np.uint8),
    occ_gt=np.ones((100, 100), dtype=np.uint8),
    bev_seg_gt=np.zeros((100, 100), dtype=np.uint8),
    K=np.eye(3, dtype=np.float32),
)
assert ok, "PUT failed"
print("PUT: OK")

print("\n=== GET test sample ===")
d = kv.get_npz(f"{PREFIX}sample_000000")
print("GET: OK, keys:", d.files)
for k in d.files:
    print(f"  {k}: shape={d[k].shape}, dtype={d[k].dtype}")

print("\n=== LIST ===")
keys = kv.list_keys(PREFIX)
print(f"{PREFIX} keys:", keys)
print(f"{PREFIX} count:", kv.count(PREFIX))

print("\n=== PUT 5 more samples ===")
for i in range(1, 6):
    kv.put_npz(
        f"{PREFIX}sample_{i:06d}",
        left_rgb=np.random.randint(0, 255, (540, 960, 3), dtype=np.uint8),
        right_rgb=np.random.randint(0, 255, (540, 960, 3), dtype=np.uint8),
        depth_gt=np.random.rand(100, 100).astype(np.float32),
        seg_gt=np.random.randint(0, 10, (100, 100), dtype=np.uint8),
        occ_gt=np.random.randint(0, 2, (100, 100), dtype=np.uint8),
        bev_seg_gt=np.random.randint(0, 10, (100, 100), dtype=np.uint8),
        K=np.eye(3, dtype=np.float32),
    )
print(f"PUT 5 more: OK, total {PREFIX} count = {kv.count(PREFIX)}")

print("\n=== Verify all samples readable ===")
for key in kv.list_keys(PREFIX):
    d = kv.get_npz(key)
    assert "left_rgb" in d.files
    assert d["left_rgb"].shape == (540, 960, 3)
print(f"All {kv.count(PREFIX)} samples verified OK")

for key in kv.list_keys(PREFIX):
    kv.delete(key)

print("\n=== DONE - minikeyvalue client works ===")
