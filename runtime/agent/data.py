"""Off-chain data for the devnet: job inputs (token batches) and outputs (embeddings).

The design leaves delivery off-chain; on the devnet one LAN-only HTTP service holds files:
  /inputs/<input_root>.json   {"batches": [[tok, ...], ...]}   (checked against input_root)
  /outputs/<job_id>.npy       float32 (batches, D) embeddings  (checked against output_root)
  /req/<job>/<stage>-<nonce>.json      a verifier asks for a stage's input; GET /req/<job>/ lists them
  /stagein/<job>/<stage>-<nonce>.json  the provider's answer: rows, Merkle proofs, previous stage opened
Run: python -m agent.data serve --root devnet/data --bind 192.168.29.71:18680

Public service (--auth --chain <network dir>): every PUT carries X-UWC-Sig, an account's
signature (uwcctl sign) over put_digest(path, body), and the chain decides who may write what:
  inputs   any account with a stake or --min-balance; write-once (readers check input_root)
  outputs  only the job's provider
  req      the job's committee, or an account with a stake or --min-balance; capped per job
  stagein  only the job's provider, and only for an open request
  stream   only the job's provider: a generation job's tokens so far, for streaming clients
GET /units/<work_root>.json serves pool work lists (placed by the operator; agents check them
against the category's work root), so a new pool category reaches every provider.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NAME = re.compile(r"^/((inputs|outputs)/[0-9a-f]{64}\.(json|npy)"
                  r"|(req|stagein)/[0-9a-f]{64}/[0-9]{1,6}-[0-9a-f]{32}\.json"
                  r"|units/[0-9a-f]{64}\.json"
                  r"|stream/[0-9a-f]{64}\.json)$")
LISTING = re.compile(r"^/req/([0-9a-f]{64})/$")
MAX_BODY = 256 << 20
SIG_HEADER = "X-UWC-Sig"
LIMITS = {"inputs": 16 << 20, "outputs": 64 << 20, "req": 1 << 10, "stagein": 32 << 20, "stream": 1 << 20}  # with --auth
REQ_PER_JOB, REQ_PER_SIGNER = 128, 32
MIN_FREE = 1 << 30  # refuse uploads below this much free disk


def put_digest(path: str, body: bytes) -> bytes:
    """What an upload's signature covers: the path and the body."""
    return hashlib.sha256(b"uwc-data\0PUT\0" + path.encode() + b"\0" + hashlib.sha256(body).digest()).digest()


class Policy:
    """Who may write what (--auth): signatures checked with uwcctl verify, rights from the chain."""

    def __init__(self, chain, min_balance: int):
        self.chain = chain
        self.min_balance = min_balance
        self.lock = threading.Lock()
        self.cache: dict[tuple, tuple[float, object]] = {}
        self.req_count: dict[tuple[str, str], int] = {}

    def _cached(self, key: tuple, fn, ttl: float = 10.0):
        now = time.time()
        with self.lock:
            hit = self.cache.get(key)
            if hit and now - hit[0] < ttl:
                return hit[1]
        val = fn()
        with self.lock:
            if len(self.cache) > 10_000:
                self.cache.clear()
            self.cache[key] = (now, val)
        return val

    def signer(self, sig: str | None, path: str, body: bytes) -> str | None:
        if not sig or len(sig) > 200:
            return None
        r = subprocess.run([self.chain.bin, "verify", sig, put_digest(path, body).hex()], capture_output=True,
                           text=True, env=self.chain.env, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None

    def job(self, jid: str) -> dict | None:
        return self._cached(("job", jid), lambda: self.chain.job(jid))

    def funded(self, acct: str) -> bool:
        a = self._cached(("acct", acct), lambda: self.chain.account(acct), ttl=60)
        return int(a.get("stake") or 0) > 0 or int(a.get("balance") or 0) >= self.min_balance

    def check(self, path: str, body: bytes, signer: str, root: str) -> tuple[int, str] | None:
        """None if signer may write body at path, else (HTTP status, reason)."""
        kind = path.split("/")[1]
        if len(body) > LIMITS[kind]:
            return 413, f"{kind} uploads are at most {LIMITS[kind]} bytes"
        if kind == "inputs":
            p = os.path.join(root, path.lstrip("/"))
            if os.path.exists(p) and open(p, "rb").read() != body:
                return 409, "inputs are write-once"
            return None if self.funded(signer) else (403, "inputs need an account with a stake or balance")
        jid = path.split("/")[2].split(".")[0]  # outputs/<job>.npy, req/<job>/..., stagein/<job>/...
        j = self.job(jid)
        if j is None:
            return 404, "unknown job"
        if kind in ("outputs", "stagein", "stream"):
            if signer != j.get("provider"):
                return 403, "only the job's provider writes its outputs, stage inputs and partial generations"
            if kind == "stagein" and not os.path.exists(os.path.join(root, "req", jid, path.split("/")[3])):
                return 404, "no such stage-input request"
            return None
        # req
        committee = {c.get("account") for c in (j.get("committee") or [])}
        if signer not in committee and not self.funded(signer):
            return 403, "stage-input requests need a committee seat, a stake or a balance"
        d = os.path.join(root, "req", jid)
        if os.path.isdir(d) and len(os.listdir(d)) >= REQ_PER_JOB:
            return 429, "too many requests for this job"
        with self.lock:
            n = self.req_count.get((jid, signer), 0)
            if n >= REQ_PER_SIGNER:
                return 429, "too many requests from this account for this job"
            self.req_count[(jid, signer)] = n + 1
        return None


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


def handler(root: str, policy: Policy | None = None, slots: threading.BoundedSemaphore | None = None):
    class H(BaseHTTPRequestHandler):
        timeout = 30  # seconds a client may take to send a request

        def handle_one_request(self):
            if slots is not None and not slots.acquire(blocking=False):
                try:
                    self.raw_requestline = self.rfile.readline(65537)
                    if self.parse_request():
                        self.send_error(503, "busy")
                except Exception:
                    pass
                self.close_connection = True
                return
            try:
                super().handle_one_request()
            finally:
                if slots is not None:
                    slots.release()

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
            if self.path.startswith("/units/"):  # pool work lists are published by the operator
                self.send_error(403, "work lists are read-only")
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
            except ValueError:
                n = -1
            if n <= 0 or n > MAX_BODY:
                self.send_error(400)
                return
            if policy is not None:
                kind = self.path.split("/")[1]
                if n > LIMITS[kind]:
                    self.send_error(413, f"{kind} uploads are at most {LIMITS[kind]} bytes")
                    return
                if shutil.disk_usage(root).free < MIN_FREE:
                    self.send_error(507, "the data service is full")
                    return
            data = self.rfile.read(n)
            if policy is not None:
                try:
                    who = policy.signer(self.headers.get(SIG_HEADER), self.path, data)
                    if who is None:
                        self.send_error(401, f"a valid {SIG_HEADER} signature is required")
                        return
                    bad = policy.check(self.path, data, who, root)
                except Exception as e:  # the node is unreachable: refuse rather than guess
                    self.send_error(503, f"cannot check the upload: {type(e).__name__}")
                    return
                if bad:
                    self.send_error(*bad)
                    return
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
    """Client. signer(digest) -> "<pubkey>:<sig>" signs uploads (needed by a --auth service)."""

    def __init__(self, url: str, signer=None):
        self.url = url.rstrip("/")
        self.signer = signer

    def put(self, path: str, data: bytes) -> None:
        req = urllib.request.Request(self.url + path, data=data, method="PUT")
        if self.signer is not None:
            try:
                req.add_header(SIG_HEADER, self.signer(put_digest(path, data)))
            except Exception:  # an older uwcctl without sign: fine for a LAN service without --auth
                pass
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
    ap.add_argument("--auth", action="store_true", help="require signed uploads, rights from the chain (public use)")
    ap.add_argument("--chain", default=None, help="--auth: network directory (env, bin/uwcctl) of a local node")
    ap.add_argument("--min-balance", type=int, default=100_000_000, help="--auth: balance that lets an account upload inputs and requests")
    ap.add_argument("--max-conn", type=int, default=64, help="requests handled at once")
    a = ap.parse_args()
    host, port = a.bind.rsplit(":", 1)
    os.makedirs(a.root, exist_ok=True)
    policy = None
    if a.auth:
        from .chain import Chain
        if not a.chain:
            ap.error("--auth needs --chain")
        policy = Policy(Chain(a.chain), a.min_balance)
    srv = ThreadingHTTPServer((host, int(port)), handler(os.path.abspath(a.root), policy, threading.BoundedSemaphore(a.max_conn)))
    srv.daemon_threads = True
    srv.serve_forever()


if __name__ == "__main__":
    main()
