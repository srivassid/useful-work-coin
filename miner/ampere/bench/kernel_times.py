"""True per-kernel GPU time of MinedWeight.forward (CUPTI via torch.profiler)."""
import sys

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from uwc_ampere import ops


def main(shapes):
    dev = "cuda"
    for m, n, k in shapes:
        A = torch.randint(-64, 65, (m, k), dtype=torch.int8, device=dev)
        Bt = torch.randint(-64, 65, (n, k), dtype=torch.int8, device=dev)
        w = ops.MinedWeight(Bt=Bt, b_scale=torch.ones(n, device=dev))
        w.setup_block(b"\x01" * 32, np.zeros(8, np.uint32))
        a = torch.ones(m, device=dev)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        for _ in range(3):
            w.forward(A, a, out=out)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(10):
                w.forward(A, a, out=out)
            torch.cuda.synchronize()
        tot = {}
        for e in p.events():
            if e.device_type.name == "CUDA":
                tot[e.name] = tot.get(e.name, 0) + e.device_time
        print(f"m={m} n={n} k={k}: GPU us per forward")
        s = 0.0
        for name, v in sorted(tot.items(), key=lambda x: -x[1]):
            print(f"   {name[:44]:44s} {v / 10:8.1f}")
            s += v / 10
        print(f"   {'SUM':44s} {s:8.1f}")


if __name__ == "__main__":
    args = list(map(int, sys.argv[1:]))
    main([tuple(args[i:i + 3]) for i in range(0, len(args), 3)] or [(4096, 4096, 4096)])
