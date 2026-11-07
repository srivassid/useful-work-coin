"""Live lambda of the deterministic runtime, offline on one GPU (no gateway, no chain).

lambda = hashed MACs per second of wall time while computing a job's stage outputs
         / hashed MACs per second of noise mining on the same GPU.

The job is what a provider computes: B sequences of T tokens through every layer group
(detrt.fast.run_stages, keep_rows=False), with every int8 GEMM mined under a synthetic block
key. Also reports GPU-kernel time (CUPTI) split into mined GEMMs and everything else, and the
number of kernel launches, so launch overhead and float-op time can be told apart.

  python -m bench.lambda_det --device cuda:1 --batches 8 --seq 128
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "miner", "ampere"))

from detrt.fast import run_stages  # noqa: E402
from detrt.qwen3 import DetLinear  # noqa: E402
from uwc_ampere import pearl_glue as pg  # noqa: E402

GEMM_KERNELS = ("noisy_gemm", "noise_rows", "noise_uniform", "noise_perm", "b3_tree", "gemm")


def setup_mining(m):
    hdr = pg.header(0x1D00FFFF, 1)
    cfgs = {}

    def cfg(k):
        return cfgs.setdefault(k, pg.mining_config(k))

    for L in m.layers:
        for n in ("qkv", "o", "gu", "down"):
            L[n].setup_block(lambda k: pg.job_key(hdr, cfg(k)), lambda k: pg.bound_words(pg.jackpot_bound(hdr.nbits, cfg(k))))
    return sum(L[n].mw is not None for L in m.layers for n in ("qkv", "o", "gu", "down"))


def count_hashed():
    """Wrap DetLinear.int_gemm to count hashed MACs (m padded to the kernel's tile, as JobMiner does)."""
    from uwc_ampere import ops
    state = {"macs": 0}
    orig = DetLinear.int_gemm

    def int_gemm(lin, q):
        if lin.mine and lin.mw is not None:
            state["macs"] += -(-q.shape[0] // ops.BM) * ops.BM * lin.n * lin.k
        return orig(lin, q)
    DetLinear.int_gemm = int_gemm
    return state


def job(m, seqs, groups, run=run_stages):
    lpg = len(m.layers) // groups
    xs = None
    for g in range(groups):
        ys = run(m, seqs if g == 0 else None, xs, g * lpg, (g + 1) * lpg, keep_rows=False)
        xs = [torch.from_numpy(y) for y in ys]
    return xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-json", default=os.path.expanduser("~/Documents/useful-work-coin/testnet/models/qwen3-1.7b.json"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batches", type=int, default=8, help="sequences per job")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--profile", action="store_true", help="CUPTI kernel breakdown (one extra run)")
    ap.add_argument("--noise-rate", type=float, default=0.0, help="TMAC/s; measured on this GPU if 0")
    ap.add_argument("--engine", default="detrt", choices=["detrt", "detgpu"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from agent.model_info import load
    info = json.load(open(a.model_json))
    torch.cuda.set_device(torch.device(a.device))
    m = load(info["model_dir"], a.device, info["rotation_block"], info["cfg"]["MaxPos"], mine=True)
    G = info["layer_groups"]
    mined = setup_mining(m)
    state = count_hashed()
    g = torch.Generator().manual_seed(0)
    seqs = [torch.randint(0, 150000, (a.seq,), generator=g) for _ in range(a.batches)]
    run_job = job
    if a.engine == "detgpu":
        from detgpu.stages import job_outputs
        toks = torch.stack(seqs)

        def run_job(m, seqs, G):
            return job_outputs(m, toks, G)[-1].cpu()

    run_job(m, seqs, G)  # warm-up
    torch.cuda.synchronize()
    state["macs"] = 0
    t = time.perf_counter()
    for _ in range(a.reps):
        run_job(m, seqs, G)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t) / a.reps
    macs = state["macs"] / a.reps
    res = {"engine": a.engine, "device": torch.cuda.get_device_name(), "batches": a.batches, "seq": a.seq, "groups": G,
           "mined_linears": mined, "wall_s": wall, "hashed_macs": macs, "hashed_tmac_s": macs / wall / 1e12}

    if a.profile:
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            run_job(m, seqs, G)
            torch.cuda.synchronize()
        tot = gemm = 0.0
        n = ngemm = 0
        for e in p.events():
            if e.device_type.name == "CUDA":
                d = e.device_time / 1e6
                tot += d
                n += 1
                if any(s in e.name for s in GEMM_KERNELS):
                    gemm += d
                    ngemm += 1
        res.update(kernel_s=tot, gemm_kernel_s=gemm, kernels=n, gemm_kernels=ngemm)
        by = {}
        for e in p.events():
            if e.device_type.name == "CUDA":
                k = e.name[:90]
                c = by.setdefault(k, [0.0, 0])
                c[0] += e.device_time / 1e6
                c[1] += 1
        res["top_kernels"] = [[k, round(v[0], 4), v[1]] for k, v in sorted(by.items(), key=lambda kv: -kv[1][0])[:25]]

    if a.noise_rate:
        res["noise_tmac_s"] = a.noise_rate
    else:
        import importlib.util  # miner/ampere/bench/lambda_job.py (a different "bench" from this one)
        spec = importlib.util.spec_from_file_location("lambda_job", os.path.join(HERE, "..", "..", "miner", "ampere", "bench", "lambda_job.py"))
        lj = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lj)
        res["noise_tmac_s"] = lj.noise_rate(a.device)[1] / 1e12
    res["lambda_wall"] = res["hashed_tmac_s"] / res["noise_tmac_s"]
    if "kernel_s" in res:
        res["lambda_kernel"] = macs / res["kernel_s"] / 1e12 / res["noise_tmac_s"]
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
