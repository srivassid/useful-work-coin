"""Per-stage timing of MinedWeight.forward."""
import sys

import numpy as np
import torch

from uwc_ampere import ops


def ev():
    return torch.cuda.Event(enable_timing=True)


def main(m, n, k):
    dev = "cuda"
    A = torch.randint(-64, 65, (m, k), dtype=torch.int8, device=dev)
    Bt = torch.randint(-64, 65, (n, k), dtype=torch.int8, device=dev)
    w = ops.MinedWeight(Bt=Bt, b_scale=torch.ones(n, device=dev), r=128)
    w.setup_block(np.random.bytes(32), np.zeros(8, np.uint32))
    a_scale = torch.ones(m, device=dev)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    w.forward(A, a_scale, out=out)
    b = w._buf(m)
    r = 128
    stages = {
        "commit_A": lambda: ops.b3_root_and_seed(A.view(-1), w.job_key, prefix=w.b_seed, salt=ops.SALT_A.to(dev), dim=m,
                                                 seed_out=b["a_seed"], scratch=b["scratch"]),
        "noise_EAL": lambda: ops.noise_uniform(b["EAL"], ops.LABEL_A.to(dev), b["a_seed"], zero_me=b["maxabs"], zero_n=2),
        "noise_PA": lambda: ops.noise_perm(b["PA"], k, r, ops.LABEL_A.to(dev), b["a_seed"]),
        "gemm_plain_i32": lambda: ops.gemm(A, Bt, out=torch.empty((m, n), dtype=torch.int32, device=dev), out_mode=ops.OUT_I32),
        "cublas_int8": lambda: torch._int_mm(A, Bt.t()),
        "rows_A(noise+X)": lambda: ops.noise_rows(A, m, k, r, E=b["EAL"], Pn=b["PA"], dst=b["ApEA"], Pp=w.PB, corr=b["corrA"],
                                                  corr_off=0, ecopy_off=r, maxabs=b["maxabs"][0:1]),
        "rows_B(Y)": lambda: ops.noise_rows(w.BpEBt, n, k, r, Pp=b["PA"], corr=w.corrB, corr_off=r, maxabs=b["maxabs"][1:2]),
        "gemm_hash": lambda: ops.gemm(b["ApEA"], w.BpEBt, r=r, flags=ops.FLAG_HASH, key=b["a_seed"], target=w.target,
                                      wins=b["wins"], max_wins=w.max_wins),
        "gemm_hash_denoise_bf16": lambda: ops.gemm(b["ApEA"], w.BpEBt, r=r, flags=3, corrA=b["corrA"], corrB=w.corrB,
                                                   maxabs=b["maxabs"], key=b["a_seed"], target=w.target, wins=b["wins"],
                                                   max_wins=w.max_wins, a_scale=a_scale, b_scale=w.b_scale, out=out,
                                                   out_mode=ops.OUT_BF16),
        "forward_total": lambda: w.forward(A, a_scale, out=out),
    }
    gf = ops.GraphedForward(w, m)
    gf.A.copy_(A)
    stages["forward_graphed"] = gf
    print(f"m={m} n={n} k={k}  maxabs={b['maxabs'].tolist()}")
    for name, fn in stages.items():
        for _ in range(3):
            fn()
        s, e = ev(), ev()
        torch.cuda.synchronize()
        s.record()
        for _ in range(20):
            fn()
        e.record()
        torch.cuda.synchronize()
        print(f"  {name:24s} {s.elapsed_time(e) / 20 * 1e3:8.1f} us")


if __name__ == "__main__":
    main(*map(int, sys.argv[1:4]))
