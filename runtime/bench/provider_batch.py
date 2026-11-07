"""A provider's lambda end to end, offline: K jobs of B x T tokens computed one GPU pass per job
(as testnet run 2 did) or together in passes of up to --chunk sequences (Runtime.jobs_roots),
with every mined GEMM under a synthetic block key. Reports per mode the GPU time, the wall time
including the host side (roots, pooled outputs), and lambda over each. The old pipeline
(Runtime.job_outputs: every stage's rows copied to the host with Merkle proofs) is timed too;
it also saved those rows as JSON, which is not counted here.

  python -m bench.provider_batch --device cuda:0 --jobs 4 --batches 8 --chunk 32
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "miner", "ampere"))

from agent.compute import Runtime  # noqa: E402
from bench.lambda_det import count_hashed, setup_mining  # noqa: E402


def noise_rate(device: str) -> float:
    spec = importlib.util.spec_from_file_location("lambda_job", os.path.join(HERE, "..", "..", "miner", "ampere", "bench", "lambda_job.py"))
    lj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lj)
    return lj.noise_rate(device)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-json", default=os.path.expanduser("~/Documents/useful-work-coin/testnet/models/qwen3-1.7b.json"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--batches", type=int, default=8, help="sequences per job")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--chunks", default="16,32,64", help="sequences per pass to try in batched mode")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--noise-rate", type=float, default=0.0, help="TMAC/s; measured on this GPU if 0")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.cuda.set_device(torch.device(a.device))
    rt = Runtime(a.model_json, a.device, mine=True)
    setup_mining(rt.m)
    state = count_hashed()
    g = torch.Generator().manual_seed(0)
    jobs = [torch.randint(0, 150000, (a.batches, a.seq), generator=g).tolist() for _ in range(a.jobs)]

    def timed(fn):
        fn()  # warm-up (compiles kernels for this shape)
        best = None
        for _ in range(a.reps):
            state["macs"] = 0
            torch.cuda.synchronize()
            t = time.perf_counter()
            gpu = fn()
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
            if best is None or wall < best[0]:
                best = (wall, gpu, state["macs"])
        return best

    modes = {"old: job_outputs, one job per pass": lambda: sum(rt.job_outputs(b) and rt.last_gpu_s for b in jobs),
             "one job per pass": lambda: sum(rt.jobs_roots([b], tokens=a.batches * a.seq) and rt.last_gpu_s for b in jobs)}
    for c in [int(x) for x in a.chunks.split(",")]:
        modes[f"batched, {c} sequences per pass"] = lambda c=c: rt.jobs_roots(jobs, tokens=c * a.seq) and rt.last_gpu_s
    noise = (a.noise_rate * 1e12) if a.noise_rate else noise_rate(a.device)
    res = {"gpu": torch.cuda.get_device_name(), "jobs": a.jobs, "batches": a.batches, "seq": a.seq,
           "noise_tmac_s": noise / 1e12, "modes": {}}
    print(f"{res['gpu']}: {a.jobs} jobs of {a.batches} x {a.seq} tokens; noise {noise / 1e12:.1f} TMAC/s")
    for name, fn in modes.items():
        wall, gpu, macs = timed(fn)
        r = {"wall_s": wall, "gpu_s": gpu, "hashed_macs": macs,
             "lambda_gpu": macs / gpu / noise, "lambda_wall": macs / wall / noise}
        res["modes"][name] = r
        print(f"  {name:38s} GPU {gpu * 1e3:7.1f} ms  wall {wall * 1e3:7.1f} ms  "
              f"lambda {r['lambda_gpu']:.3f} (GPU)  {r['lambda_wall']:.3f} (wall)", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
