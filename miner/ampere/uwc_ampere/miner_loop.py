"""Noise miner for the UWC devnet: the Ampere cuPOW kernels behind Pearl's gateway protocol.

Loop: getMiningInfo (header bytes, target, certificate version) -> commit a random A against a
fixed B^T under the block's key -> hashed noisy GEMM -> on a winning tile, build the plain proof
from the un-noised A and B^T and submitPlainProof. The gateway turns it into a zk certificate
and submits the block to pearld.

Each line of --stats is one interval: hashed MACs, attempts, wins, GPU-busy fraction. The
testnet's lambda metric divides a job's hashed MACs per second by this noise rate.

  python -m uwc_ampere.miner_loop --gateway /tmp/pearlgw.sock --device 0 --stats noise.jsonl
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pearl_mining as pm
import torch
from pearl_gateway.comm.dataclasses import MiningJob
from pearl_gateway.config import MinerRpcConfig
from miner_base.gateway_client import MiningClient

from . import ops
from . import pearl_glue as pg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="/tmp/pearlgw.sock", help="UDS path, or host:port for TCP")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--m", type=int, default=8192)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--refresh", type=float, default=1.0, help="seconds between getMiningInfo polls")
    ap.add_argument("--stats", default=None, help="append one JSON line per interval")
    ap.add_argument("--interval", type=float, default=30.0)
    a = ap.parse_args()

    if ":" in a.gateway:
        host, port = a.gateway.rsplit(":", 1)
        cfg_rpc = MinerRpcConfig(transport="tcp", host=host, port=int(port), socket_path=None)
    else:
        cfg_rpc = MinerRpcConfig(transport="uds", socket_path=a.gateway)
    client = MiningClient(cfg_rpc)

    def reconnect():
        nonlocal client
        try:
            client.close()
        except Exception:
            pass
        try:
            client = MiningClient(cfg_rpc)
        except Exception as e:  # gateway restarting
            print(f"reconnect: {e}", flush=True)

    dev = torch.device(f"cuda:{a.device}")
    torch.cuda.set_device(dev)
    gen = torch.Generator(device=dev).manual_seed(int.from_bytes(np.random.bytes(8), "little"))
    Bt = torch.randint(-64, 65, (a.n, a.k), dtype=torch.int8, device=dev, generator=gen)
    Bt_np = Bt.cpu().numpy()
    w = ops.MinedWeight(Bt=Bt, b_scale=torch.ones(a.n, device=dev), r=a.rank)
    cfg = pg.mining_config(a.k, a.rank)
    A = torch.empty((a.m, a.k), dtype=torch.int8, device=dev)

    job: MiningJob | None = None
    hdr = None
    last_poll = 0.0
    t0, macs, attempts, wins_n, submitted = time.time(), 0, 0, 0, 0
    print(f"mining on {torch.cuda.get_device_name(dev)}: m={a.m} n={a.n} k={a.k} r={a.rank}", flush=True)
    while True:
        now = time.time()
        if job is None or now - last_poll > a.refresh:
            last_poll = now
            try:
                nj = client.get_mining_info()
            except Exception as e:  # gateway not ready or restarted, mining paused, node syncing
                print(f"getMiningInfo: {e}", flush=True)
                time.sleep(2)
                reconnect()
                continue
            if job is None or nj.incomplete_header_bytes != job.incomplete_header_bytes or nj.target != job.target:
                job = nj
                hdr = pm.IncompleteBlockHeader.from_bytes(job.incomplete_header_bytes)
                key = pg.job_key(hdr, cfg)
                bound = pm.penalized_target_bound(job.target, cfg)
                w.setup_block(key, pg.bound_words(bound))
        torch.randint(-64, 65, A.shape, dtype=torch.int8, device=dev, generator=gen, out=A)
        w.noise_mine(A)
        wins = w.winners(a.m)  # syncs
        attempts += 1
        macs += a.m * a.n * a.k
        if wins:
            wins_n += len(wins)
            row_off, col_off, _ = wins[0]
            A_np = A.cpu().numpy()
            rows = [row_off + o for o in pg.AMPERE_ROWS]
            cols = [col_off + o for o in pg.AMPERE_COLS]
            proof = pg.build_plain_proof(A_np, Bt_np, key, a.rank, rows, cols)
            ok, msg = pm.verify_plain_proof_for_cert_version(int(job.cert_version), hdr, proof)
            if ok:
                try:
                    client.submit_plain_proof(proof, job)
                    submitted += 1
                    print(f"submitted a winning tile ({len(wins)} wins) for header {job.incomplete_header_bytes[:8].hex()}", flush=True)
                except Exception as e:
                    print(f"submitPlainProof: {e}", flush=True)
                    reconnect()
            else:
                print(f"local verification failed: {msg}", flush=True)
            job = None  # fetch the next template
        if now - t0 >= a.interval:
            rec = {"t": now, "device": torch.cuda.get_device_name(dev), "secs": now - t0, "macs": macs,
                   "macs_per_s": macs / (now - t0), "attempts": attempts, "wins": wins_n, "submitted": submitted}
            print(json.dumps(rec), flush=True)
            if a.stats:
                with open(a.stats, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            t0, macs, attempts, wins_n, submitted = now, 0, 0, 0, 0


if __name__ == "__main__":
    main()
