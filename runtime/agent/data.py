"""Off-chain data for the devnet: job inputs (token batches) and outputs (embeddings).

The design leaves delivery off-chain; on the devnet one LAN-only HTTP service holds files:
  /inputs/<input_root>.json   {"batches": [[tok, ...], ...]}   (checked against input_root)
  /outputs/<job_id>.npy       float32 (batches, D) embeddings  (checked against output_root)
  /req/<job>/<stage>-<nonce>.json      a verifier asks for a stage's input; GET /req/<job>/ lists them
  /stagein/<job>/<stage>-<nonce>.json  the provider's answer: rows, Merkle proofs, previous stage opened
Run: python -m agent.data serve --root devnet/data --bind 192.168.29.71:18680
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NAME = re.compile(r"^/((inputs|outputs)/[0-9a-f]{64}\.(json|npy)"
                  r"|(req|stagein)/[0-9a-f]{64}/[0-9]{1,6}-[0-9a-f]{32}\.json)$")
LISTING = re.compile(r"^/req/([0-9a-f]{64})/$")
MAX_BODY = 256 << 20


METRICS_PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>UWC testnet metrics</title>
<style>
 :root{--bg:#f7f8f6;--fg:#1d2320;--mut:#5d6762;--line:#d9ded9;--bad:#b3261e}
 @media (prefers-color-scheme:dark){:root{--bg:#141816;--fg:#e4e9e5;--mut:#98a39d;--line:#2c332f;--bad:#f2b8b5}}
 body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:16px max(16px,4vw)}
 h1{font-weight:500;margin:0 0 4px}.mut{color:var(--mut)}
 .g{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(13rem,1fr));margin:16px 0}
 .t{border:1px solid var(--line);border-radius:8px;padding:12px}.v{font-size:1.6rem;font-variant-numeric:tabular-nums}
 table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line)}
</style>
<h1>UWC testnet metrics</h1><p class="mut" id="when">loading…</p>
<div class="g" id="tiles"></div><h2>Jobs</h2><div style="overflow-x:auto"><table id="jobs"></table></div>
<script>
const tiles=[["Settled jobs",d=>d.stats.SettledJobs],["False slashes",d=>d.false_slashes],
 ["Verification overhead",d=>d.verification_overhead_median==null?"–":d.verification_overhead_median.toFixed(2)+"×"],
 ["Live λ",d=>d.lambda_live_median==null?"–":d.lambda_live_median.toExponential(1)],
 ["Client price vs renting",d=>d.client_price_vs_rent_median==null?"–":d.client_price_vs_rent_median.toFixed(2)+"×"],
 ["Useful share",d=>d.useful_share.toExponential(1)],["Network TMAC/s",d=>d.network_TMACs],
 ["Seconds per block",d=>d.seconds_per_block]];
async function load(){try{const d=await (await fetch("/metrics.json",{cache:"no-store"})).json();
 document.getElementById("when").textContent="Block "+d.height+", refreshed every few minutes";
 const g=document.getElementById("tiles");g.replaceChildren(...tiles.map(([k,f])=>{const e=document.createElement("div");e.className="t";
  const a=document.createElement("div");a.className="mut";a.textContent=k;const b=document.createElement("div");b.className="v";b.textContent=f(d);e.append(a,b);return e}));
 const t=document.getElementById("jobs");const cols=["job","kind","status","stages","posted","fee"];
 t.replaceChildren(...[cols].concat(d.job_table.slice().reverse().map(r=>cols.map(c=>r[c])))).map((r,i)=>{const tr=document.createElement("tr");
  r.forEach(v=>{const c=document.createElement(i?"td":"th");c.textContent=v;tr.append(c)});return tr}));
}catch(e){document.getElementById("when").textContent="metrics not available yet"}}
load();setInterval(load,60000);
</script>"""


def handler(root: str):
    class H(BaseHTTPRequestHandler):
        def _path(self):
            if not NAME.match(self.path):
                self.send_error(404)
                return None
            return os.path.join(root, self.path.lstrip("/"))

        def do_GET(self):
            if self.path in ("/", "/metrics.json"):  # the live testnet metrics (agent.metrics --json)
                f = os.path.join(root, "metrics.json") if self.path != "/" else None
                if f is None:
                    data, ctype = METRICS_PAGE.encode(), "text/html; charset=utf-8"
                elif os.path.exists(f):
                    data, ctype = open(f, "rb").read(), "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            m = LISTING.match(self.path)
            if m:  # the stage-input requests open for one job
                d = os.path.join(root, "req", m.group(1))
                names = sorted(os.listdir(d)) if os.path.isdir(d) else []
                data = json.dumps(names).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
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
