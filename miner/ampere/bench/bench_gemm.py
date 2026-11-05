"""Throughput of the Ampere kernels vs cuBLAS int8 (torch._int_mm), RTX 3080."""
import argparse
import json

import numpy as np
import torch

from uwc_ampere import ops
from uwc_ampere import pearl_glue as pg


def timed(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters / 1e3  # seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda"
    rows = []
    shapes = [(4096, 4096, 4096), (8192, 8192, 4096), (4096, 4096, 16384), (2048, 3072, 3072), (4096, 3072, 8192),
              (4096, 8192, 3072), (1024, 3072, 3072), (8192, 16384, 4096)]
    for m, n, k in shapes:
        A = torch.randint(-64, 65, (m, k), dtype=torch.int8, device=dev)
        Bt = torch.randint(-64, 65, (n, k), dtype=torch.int8, device=dev)
        macs = m * n * k
        t_cublas = timed(lambda: torch._int_mm(A, Bt.t()))
        out = torch.empty((m, n), dtype=torch.int32, device=dev)
        res = {"m": m, "n": n, "k": k, "cublas_tops": 2 * macs / t_cublas / 1e12}
        for cfg in ops.CONFIGS:
            t = timed(lambda: ops.gemm(A, Bt, out=out, out_mode=ops.OUT_I32, cfg=cfg))
            res[f"plain_{cfg}_tops"] = 2 * macs / t / 1e12
        # mining kernel alone: hash every tile, no denoise, no output (what a noise miner runs)
        tgt = torch.zeros(8, dtype=torch.int32, device=dev)
        key = torch.zeros(8, dtype=torch.int32, device=dev)
        wins = torch.zeros(1, dtype=torch.int32, device=dev)
        t_mine = timed(lambda: ops.gemm(A, Bt, flags=ops.FLAG_HASH, key=key, target=tgt, wins=wins, max_wins=0))
        res["mine_kernel_tops"] = 2 * macs / t_mine / 1e12
        # full useful forward: commit A, noise, projections, noisy GEMM + hash + exact denoise + bf16 out
        cfg = pg.mining_config(k, 128)
        w = ops.MinedWeight(Bt=Bt, b_scale=torch.ones(n, device=dev), r=128)
        w.setup_block(np.random.bytes(32), pg.bound_words(0))
        a_scale = torch.ones(m, device=dev)
        outb = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        t_fwd = timed(lambda: w.forward(A, a_scale, out=outb))
        res["useful_forward_tops"] = 2 * macs / t_fwd / 1e12
        res["t_cublas_ms"], res["t_mine_ms"], res["t_fwd_ms"] = t_cublas * 1e3, t_mine * 1e3, t_fwd * 1e3
        rows.append(res)
        print(json.dumps({kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in res.items()}), flush=True)
        del A, Bt, out, w, outb
        torch.cuda.empty_cache()
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
