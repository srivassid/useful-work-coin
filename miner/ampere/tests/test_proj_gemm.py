"""The tensor-core projections (ops.PROJ_GEMM) must leave MinedWeight.forward bit-identical to the
shared-memory-atomic noise_rows path: output, both correction tensors, maxabs and winning tiles."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from uwc_ampere import ops  # noqa: E402

ops.AUTOTUNE = False  # compare the two projection paths directly

from uwc_ampere import pearl_glue as pg  # noqa: E402


def target_for(m, n):
    """A bound giving ~12 winning tiles (8 x 16 hash tiles), below the 64-record cap, so the
    recorded set does not depend on which CTAs ran first."""
    return pg.bound_words(((1 << 256) - 1) * 12 // ((m // 8) * (n // 16)))


def test_main():
    dev = "cuda"
    torch.manual_seed(0)
    for (n, k) in ((4096, 2048), (2048, 2048), (12288, 2048), (2048, 6144)):
        for m in (128, 1024, 2048):
            Bt = torch.randint(-64, 65, (n, k), dtype=torch.int8, device=dev)
            A = torch.randint(-64, 65, (m, k), dtype=torch.int8, device=dev)
            ones = torch.ones(m, dtype=torch.float32, device=dev)
            res = {}
            for mode in (False, True):
                w = ops.MinedWeight(Bt=Bt, b_scale=torch.ones(n, device=dev), r=128)
                w.setup_block(bytes(range(32)), target_for(m, n))
                out = w._forward(A, ones, None, ops.OUT_I32, True, ops.DEFAULT_CFG, mode)
                b = w._buf(m)
                res[mode] = (out.clone(), b["corrA"].clone(), w.corrB.clone(), b["maxabs"].clone(), w.winners(m))
                ref = (A.cpu().long() @ Bt.cpu().long().t()).int()
                assert torch.equal(out.cpu(), ref), "output is not the exact product"
            for name, x, y in zip(("out", "corrA", "corrB", "maxabs"), res[False][:4], res[True][:4]):
                assert torch.equal(x, y), (n, k, m, name)
            wa, wb = sorted(res[False][4]), sorted(res[True][4])   # win records are appended by atomics: order varies
            if wa != wb or not wb:
                print("wins", len(wa), len(wb), wa[:2], wb[:2])
            assert wa == wb and 0 < len(wb) < w.max_wins, (n, k, m, "wins", len(wa), len(wb))
            print(f"n={n:5d} k={k} m={m:4d}: identical (output exact, {len(res[True][4])} wins)")
    print("PROJ GEMM IDENTICAL")


if __name__ == "__main__":
    test_main()
