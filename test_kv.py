"""Quick smoke test for minikeyvalue dataset pipeline."""
import sys, os
sys.path.insert(0, r"D:\carla")
import numpy as np
from minikeyvalue_client import MiniKV

kv = MiniKV("http://localhost:3000")

# 0. Clean up any previous test data
print("=== Cleanup ===")
for prefix in ["/train/", "/val/"]:
    for key in kv.list_keys(prefix):
        kv.delete(key)
print("Cleaned previous data")

# 1. PUT a sample (simulates collect_data)
print("=== PUT test sample ===")
kv.put_npz("/train/sample_000000",
    left_rgb=np.zeros((540, 960, 3), dtype=np.uint8),
    right_rgb=np.zeros((540, 960, 3), dtype=np.uint8),
    depth_gt=np.ones((100, 100), dtype=np.float32),
    seg_gt=np.zeros((100, 100), dtype=np.uint8),
    occ_gt=np.ones((100, 100), dtype=np.uint8),
    K=np.eye(3, dtype=np.float32),
)
print("PUT: OK")

# 2. GET it back (simulates train_bev)
print("\n=== GET test sample ===")
d = kv.get_npz("/train/sample_000000")
print("GET: OK, keys:", d.files)
for k in d.files:
    print(f"  {k}: shape={d[k].shape}, dtype={d[k].dtype}")

# 3. LIST
print("\n=== LIST ===")
keys = kv.list_keys("/train/")
print("/train/ keys:", keys)
print("/train/ count:", kv.count("/train/"))
print("/val/ count:", kv.count("/val/"))

# 4. PUT a few more
print("\n=== PUT 5 more samples ===")
for i in range(1, 6):
    kv.put_npz(f"/train/sample_{i:06d}",
        left_rgb=np.random.randint(0, 255, (540, 960, 3), dtype=np.uint8),
        right_rgb=np.random.randint(0, 255, (540, 960, 3), dtype=np.uint8),
        depth_gt=np.random.rand(100, 100).astype(np.float32),
        seg_gt=np.random.randint(0, 10, (100, 100), dtype=np.uint8),
        occ_gt=np.random.randint(0, 2, (100, 100), dtype=np.uint8),
        K=np.eye(3, dtype=np.float32),
    )
print(f"PUT 5 more: OK, total /train/ count = {kv.count('/train/')}")

# 5. Verify all readable
print("\n=== Verify all samples readable ===")
for key in kv.list_keys("/train/"):
    d = kv.get_npz(key)
    assert "left_rgb" in d.files
    assert d["left_rgb"].shape == (540, 960, 3)
print(f"All {kv.count('/train/')} samples verified OK")

print("\n=== DONE - minikeyvalue dataset pipeline works ===")
