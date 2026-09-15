"""Thin Python client for minikeyvalue (geohot/minikeyvalue)."""

import io
import json
import http.client
from urllib.parse import urlparse, urljoin
import numpy as np


class MiniKV:
    def __init__(self, url: str = "http://localhost:3000", timeout: float = 60.0):
        parsed = urlparse(url if "://" in url else f"http://{url}")
        self.host = parsed.hostname or "localhost"
        self.port = parsed.port or 80
        self.timeout = timeout
        self.url = f"http://{self.host}:{self.port}"
        self._conns: dict[tuple[str, int], http.client.HTTPConnection] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_conns"] = {}
        return state

    def _conn(self, host: str, port: int) -> http.client.HTTPConnection:
        key = (host, port)
        conn = self._conns.get(key)
        if conn is None:
            conn = http.client.HTTPConnection(host, port, timeout=self.timeout)
            self._conns[key] = conn
        return conn

    def _request(self, method: str, path: str, body=None, headers=None, host=None, port=None):
        host = host or self.host
        port = self.port if port is None else port
        hdrs = {"Connection": "keep-alive"}
        if body is not None:
            hdrs["Content-Length"] = str(len(body))
        if headers:
            hdrs.update(headers)
        last_err = None
        for _ in range(2):
            conn = self._conn(host, port)
            try:
                conn.request(method, path, body=body, headers=hdrs)
                resp = conn.getresponse()
                data = resp.read()
                return resp.status, resp.getheader("Location"), data
            except (http.client.HTTPException, ConnectionError, OSError, TimeoutError) as e:
                last_err = e
                try:
                    conn.close()
                except Exception:
                    pass
                self._conns.pop((host, port), None)
        raise last_err

    def _follow(self, method: str, key: str, body=None, headers=None):
        status, location, data = self._request(method, key, body=body, headers=headers)
        if status in (301, 302, 307, 308) and location:
            loc = urlparse(urljoin(self.url + "/", location))
            path = loc.path or "/"
            if loc.query:
                path = f"{path}?{loc.query}"
            status, _, data = self._request(
                method, path, body=body, headers=headers,
                host=loc.hostname or self.host,
                port=loc.port or 80,
            )
        return status, data

    def put(self, key: str, data: bytes) -> bool:
        """Store bytes at key. Returns True on success (201)."""
        status, _ = self._follow(
            "PUT", key, body=data,
            headers={"Content-Type": "application/octet-stream"},
        )
        if status == 403:
            return False
        return status == 201

    def get(self, key: str) -> bytes:
        """Retrieve bytes at key. Follows 302 redirects to volume."""
        status, data = self._follow("GET", key)
        if status != 200:
            raise OSError(f"GET {key} -> HTTP {status}")
        return data

    def delete(self, key: str) -> bool:
        """Delete key. Returns True on success (204)."""
        status, _ = self._follow("DELETE", key)
        return status == 204

    def list_keys(self, prefix: str) -> list[str]:
        """List keys starting with prefix."""
        path = prefix if "?" in prefix else f"{prefix}?list"
        try:
            status, data = self._follow("GET", path)
        except OSError:
            return []
        if status != 200:
            return []
        parsed = json.loads(data.decode("utf-8"))
        if isinstance(parsed, dict):
            return parsed.get("keys", [])
        if isinstance(parsed, list):
            return parsed
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
