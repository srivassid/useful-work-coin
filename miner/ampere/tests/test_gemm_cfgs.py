"""Every GEMM tile config must give the same mined result as the default: exact output and the
same winning tiles (hashes), so the config can be picked per shape and GPU for speed."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from uwc_ampere import ops, pearl_glue as pg  # noqa: E402

def test_main():
    dev = "cuda"
    torch.manual_seed(1)
    cfgs = os.environ.get("UWC_TEST_CFGS", "128x3,64x3,64x4,256x3").split(",")
    for (n, k) in ((2048, 2048), (2048, 6144), (4096, 2048)):
        for m in (128, 1024):
            Bt = torch.randint(-64, 65, (n, k), dtype=torch.int8, device=dev)
            A = torch.randint(-64, 65, (m, k), dtype=torch.int8, device=dev)
            w = ops.MinedWeight(Bt=Bt, b_scale=torch.ones(n, device=dev), r=128)
            w.setup_block(bytes(range(32)), pg.bound_words(((1 << 256) - 1) * 12 // ((m // 8) * (n // 16))))
            b = w._buf(m)
            ones = torch.ones(m, dtype=torch.float32, device=dev)
            ref = (A.cpu().long() @ Bt.cpu().long().t()).int()
            got = {}
            for cfg in cfgs:
                out = w._forward(A, ones, None, ops.OUT_I32, True, cfg, True)
                assert torch.equal(out.cpu(), ref), (cfg, n, k, m, "output")
                got[cfg] = sorted(w.winners(m))
            base = got[cfgs[0]]
            assert 0 < len(base) < w.max_wins
            for cfg in cfgs[1:]:
                assert got[cfg] == base, (cfg, n, k, m, len(got[cfg]), len(base))
            print(f"n={n} k={k} m={m}: {', '.join(cfgs)} identical ({len(base)} wins)")
    print("GEMM CONFIGS IDENTICAL")


if __name__ == "__main__":
    test_main()
