"""Where a mined GEMM's time goes at a job's layer shapes, against noise mining on the same GPU.

For Qwen3-1.7B's four mined layers (qkv, o, gate/up, down) and several m (tokens per call):
CUPTI kernel time of MinedWeight.forward split by kernel, and its hashed MACs per second as a
share of noise mining's (MinedWeight.noise_mine on a large matrix).

  python bench/mined_shapes.py [--ms 1024,2048,4096,8192]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from uwc_ampere import ops  # noqa: E402

SHAPES = {"qkv": (4096, 2048), "o": (2048, 2048), "gu": (12288, 2048), "down": (2048, 6144)}  # (n, k)


def kernel_times(fn, reps=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    by = {}
    for e in p.events():
        if e.device_type.name == "CUDA":
            by[e.name] = by.get(e.name, 0.0) + e.device_time / reps / 1e6
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="1024,2048,4096,8192")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = "cuda"
    # noise mining reference on a large matrix
    Bt = torch.randint(-64, 65, (16384, 4096), dtype=torch.int8, device=dev)
    w = ops.MinedWeight(Bt=Bt, b_scale=torch.ones(16384, device=dev))
    w.setup_block(np.random.bytes(32), np.zeros(8, np.uint32))
    A = torch.randint(-64, 65, (8192, 4096), dtype=torch.int8, device=dev)
    for _ in range(10):
        w.noise_mine(A)
    by = kernel_times(lambda: w.noise_mine(A))
    noise = 8192 * 16384 * 4096 / sum(by.values())
    del w, Bt, A
    torch.cuda.empty_cache()
    res = {"gpu": torch.cuda.get_device_name(), "noise_tmac_s": noise / 1e12, "rows": []}
    print(f"{res['gpu']}: noise mining {noise / 1e12:.1f} TMAC/s (kernel time)")
    for name, (n, k) in SHAPES.items():
        Bt = torch.randint(-64, 65, (n, k), dtype=torch.int8, device=dev)
        w = ops.MinedWeight(Bt=Bt, b_scale=torch.ones(n, device=dev), r=128)
        w.setup_block(np.random.bytes(32), np.zeros(8, np.uint32))
        for m in [int(x) for x in a.ms.split(",")]:
            A = torch.randint(-64, 65, (m, k), dtype=torch.int8, device=dev)
            ones = torch.ones(m, dtype=torch.float32, device=dev)
            by = kernel_times(lambda: w.forward(A, ones, out_mode=ops.OUT_I32))
            tot = sum(by.values())
            share = {kk.split("(")[0][:28]: v / tot for kk, v in sorted(by.items(), key=lambda kv: -kv[1])[:5]}
            eff = m * n * k / tot / noise
            res["rows"].append({"layer": name, "n": n, "k": k, "m": m, "kernel_s": tot, "eff": eff, "split": share})
            print(f"  {name:4s} n={n:5d} k={k:4d} m={m:5d}: {tot * 1e3:6.2f} ms  {eff:5.2f} of noise rate  "
                  + "  ".join(f"{kk} {v:.0%}" for kk, v in share.items()))
        del w, Bt
        torch.cuda.empty_cache()
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
