"""Off-chain data for the devnet: job inputs (token batches) and outputs (embeddings).

The design leaves delivery off-chain; on the devnet one LAN-only HTTP service holds files:
  /inputs/<input_root>.json   {"batches": [[tok, ...], ...]}   (checked against input_root)
  /outputs/<job_id>.npy       float32 (batches, D) embeddings  (checked against output_root)
Run: python -m agent.data serve --root devnet/data --bind 192.168.29.71:18680
"""
from __future__ import annotations

import argparse
import io
import os
import re
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NAME = re.compile(r"^/(inputs|outputs)/[0-9a-f]{64}\.(json|npy)$")
MAX_BODY = 256 << 20


def handler(root: str):
    class H(BaseHTTPRequestHandler):
        def _path(self):
            if not NAME.match(self.path):
                self.send_error(404)
                return None
            return os.path.join(root, self.path.lstrip("/"))

        def do_GET(self):
            p = self._path()
            if p is None:
                return
            if not os.path.exists(p):
                self.send_error(404)
                return
            data = open(p, "rb").read()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_PUT(self):
            p = self._path()
            if p is None:
                return
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0 or n > MAX_BODY:
                self.send_error(400)
                return
            data = self.rfile.read(n)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            tmp = p + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, p)
            self.send_response(201)
            self.end_headers()

        def log_message(self, *a):
            pass
    return H


class Data:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def put(self, path: str, data: bytes) -> None:
        req = urllib.request.Request(self.url + path, data=data, method="PUT")
        urllib.request.urlopen(req, timeout=60).read()

    def get(self, path: str) -> bytes | None:
        try:
            return urllib.request.urlopen(self.url + path, timeout=60).read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["serve"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--bind", required=True)
    a = ap.parse_args()
    host, port = a.bind.rsplit(":", 1)
    os.makedirs(a.root, exist_ok=True)
    ThreadingHTTPServer((host, int(port)), handler(os.path.abspath(a.root))).serve_forever()


if __name__ == "__main__":
    main()
