"""Thin Python client for minikeyvalue (geohot/minikeyvalue)."""

import io
import json
import urllib.request
import urllib.error
import numpy as np


class MiniKV:
    def __init__(self, url: str = "http://localhost:3000"):
        self.url = url.rstrip("/")

    def put(self, key: str, data: bytes) -> bool:
        """Store bytes at key. Returns True on success (201)."""
        req = urllib.request.Request(
            f"{self.url}{key}", data=data, method="PUT",
            headers={"Content-Type": "application/octet-stream"},
        )
        try:
            resp = urllib.request.urlopen(req)
            return resp.status == 201
        except urllib.error.HTTPError as e:
            if e.code == 403:
                return False  # already exists
            raise

    def get(self, key: str) -> bytes:
        """Retrieve bytes at key. Follows 302 redirects to volume."""
        req = urllib.request.Request(f"{self.url}{key}")
        resp = urllib.request.urlopen(req)
        return resp.read()

    def delete(self, key: str) -> bool:
        """Delete key. Returns True on success (204)."""
        req = urllib.request.Request(f"{self.url}{key}", method="DELETE")
        try:
            resp = urllib.request.urlopen(req)
            return resp.status == 204
        except urllib.error.HTTPError:
            return False

    def list_keys(self, prefix: str) -> list[str]:
        """List keys starting with prefix."""
        req = urllib.request.Request(f"{self.url}{prefix}?list")
        try:
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("keys", [])
        except urllib.error.HTTPError:
            return []

    def count(self, prefix: str) -> int:
        return len(self.list_keys(prefix))

    def put_npz(self, key: str, **arrays) -> bool:
        """Convenience: save numpy arrays as compressed npz at key."""
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        return self.put(key, buf.getvalue())

    def get_npz(self, key: str) -> dict:
        """Convenience: load numpy arrays from npz at key."""
        data = self.get(key)
        return np.load(io.BytesIO(data))

    def put_json(self, key: str, obj) -> bool:
        """Store a JSON-serializable object at key."""
        return self.put(key, json.dumps(obj).encode("utf-8"))

    def get_json(self, key: str):
        """Load a JSON object from key."""
        return json.loads(self.get(key).decode("utf-8"))
