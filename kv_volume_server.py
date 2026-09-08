"""
Simple Python volume server for minikeyvalue (Windows-compatible).

Replaces the nginx-based volume server for local testing.
Supports: PUT (create), GET (read), DELETE (remove), LIST (autoindex).
"""

import os
import sys
import json
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path


class VolumeHandler(BaseHTTPRequestHandler):
    root: str = ""

    def _data_path(self) -> Path:
        return Path(self.root) / self.path.lstrip("/")

    def do_PUT(self):
        path = self._data_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        path.write_bytes(body)
        self.send_response(201)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _serve_get_or_head(self, send_body: bool = True):
        path = self._data_path()
        if path.is_dir():
            entries = sorted(f.name for f in path.iterdir())
            body = json.dumps(entries).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if send_body:
                self.wfile.write(body)
        elif path.is_file():
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if send_body:
                self.wfile.write(data)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_GET(self):
        self._serve_get_or_head(send_body=True)

    def do_HEAD(self):
        self._serve_get_or_head(send_body=False)

    def do_DELETE(self):
        path = self._data_path()
        if path.exists():
            path.unlink()
            self.send_response(204)
        else:
            self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format, *args):
        # quieter logging
        pass


def main():
    parser = argparse.ArgumentParser(description="Python volume server for minikeyvalue")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--dir", default="./kv_volume")
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    VolumeHandler.root = os.path.abspath(args.dir)

    server = HTTPServer(("0.0.0.0", args.port), VolumeHandler)
    print(f"[Volume] Serving {args.dir} on :{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Volume] Stopped")


if __name__ == "__main__":
    main()
