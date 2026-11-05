"""Measure lambda for real batch-inference jobs on one RTX 3080.

lambda = (hashed MACs per GPU-second while running the job)
         / (hashed MACs per GPU-second while noise mining)
Hashed MACs are m_pad * n * k of every mined GEMM; the cuPOW difficulty target scales
with tile size x k, so this is exactly proportional to tickets (expected block wins).

GPU time is the sum of kernel durations from CUPTI (launch gaps excluded, as a graphed
or fused production runtime would remove them); wall time is reported alongside.
"""
import argparse
import json
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from uwc_ampere import model as M
from uwc_ampere import ops
from uwc_ampere import pearl_glue as pg

MINING_KERNELS = ("noisy_gemm", "noise_rows", "noise_uniform", "noise_perm", "b3_tree")


def gpu_time(fn, reps=3):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / reps
    by = {}
    for e in p.events():
        if e.device_type.name == "CUDA":
            by[e.name] = by.get(e.name, 0.0) + e.device_time / reps / 1e6
    return sum(by.values()), wall, by


def noise_rate(dev, m=8192, n=16384, k=4096):
    Bt = torch.randint(-64, 65, (n, k), dtype=torch.int8, device=dev)
    w = ops.MinedWeight(Bt=Bt, b_scale=torch.ones(n, device=dev))
    w.setup_block(np.random.bytes(32), np.zeros(8, np.uint32))
    A = torch.randint(-64, 65, (m, k), dtype=torch.int8, device=dev)
    for _ in range(15):  # sustained load first, so clocks reach their power-limited steady state
        w.noise_mine(A)
    t, wall, by = gpu_time(lambda: w.noise_mine(A), reps=20)
    del w, Bt, A
    torch.cuda.empty_cache()
    return m * n * k / t, m * n * k / wall


def load_tokens(tok, path, n_tokens):
    import os
    if not os.path.exists(path):  # compute cost does not depend on the text
        g = torch.Generator().manual_seed(0)
        return torch.randint(0, tok.vocab_size, (n_tokens,), generator=g)
    from datasets import load_from_disk
    ds = load_from_disk(path)["test"]
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    reps = -(-n_tokens // ids.numel())
    return ids.repeat(reps)[:n_tokens]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-1.7B")
    ap.add_argument("--data", default="models/wikitext2")
    ap.add_argument("--mode", default="mined", choices=["mined", "plain", "int", "bf16"])
    ap.add_argument("--compile", action="store_true", help="torch.compile each decoder layer (mined GEMMs excluded)")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--tokens", type=int, default=8192, help="tokens per batch (m of each GEMM)")
    ap.add_argument("--jobs", default="embed,score")
    ap.add_argument("--score-chunk", type=int, default=2048)
    ap.add_argument("--hadamard", type=int, default=512)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cpu",
                                               attn_implementation="sdpa")
    mdl.eval()
    mls = M.convert_qwen3(mdl, dev, mode=args.mode, hadamard_block=args.hadamard)
    mdl.to(dev)
    torch.cuda.empty_cache()
    if args.compile:
        for layer in mdl.model.layers:
            layer.forward = torch.compile(layer.forward, dynamic=False)
        mdl.model.norm.forward = torch.compile(mdl.model.norm.forward, dynamic=False)

    if args.mode == "mined":
        hdr = pg.header(0x1D00FFFF, 1)
        cfgs = {}

        def key_for(k):
            cfgs.setdefault(k, pg.mining_config(k))
            return pg.job_key(hdr, cfgs[k])

        def bound_for(k):
            cfgs.setdefault(k, pg.mining_config(k))
            return pg.bound_words(pg.jackpot_bound(hdr.nbits, cfgs[k]))

        t0 = time.perf_counter()
        for ml in mls:
            ml.setup_block(key_for, bound_for)
        torch.cuda.synchronize()
        setup_s = time.perf_counter() - t0
    else:
        setup_s = 0.0

    ids = load_tokens(tok, args.data, args.tokens).view(-1, args.seq).to(dev)
    results = {"model": args.model, "mode": args.mode, "compile": args.compile, "tokens": args.tokens, "seq": args.seq,
               "hadamard": args.hadamard, "block_setup_s": setup_s,
               "mem_gb": torch.cuda.max_memory_allocated() / 1e9,
               "mineable_layers": sum(ml.mineable for ml in mls), "layers": len(mls)}
    noise_samples = []
    if args.mode == "mined":
        noise_samples.append(noise_rate(dev))

    @torch.no_grad()
    def embed():
        h = mdl.model(input_ids=ids).last_hidden_state
        return h.float().mean(1)

    @torch.no_grad()
    def score():
        h = mdl.model(input_ids=ids).last_hidden_state.reshape(-1, mdl.config.hidden_size)
        tgt = ids.reshape(-1)[1:]
        tot = torch.zeros((), device=dev)
        for s in range(0, h.shape[0] - 1, args.score_chunk):
            lg = mdl.lm_head(h[s:s + args.score_chunk])
            lp = torch.log_softmax(lg.float(), -1)
            t = tgt[s:s + args.score_chunk]
            tot += lp[: t.numel()].gather(1, t[:, None]).sum()
        return tot

    for job in args.jobs.split(","):
        fn = {"embed": embed, "score": score}[job]
        for _ in range(3 if args.compile else 2):
            fn()
        for ml in mls:
            ml.hashed_macs = ml.useful_macs = 0
        reps = 6
        t_gpu, t_wall, by = gpu_time(fn, reps)
        if args.mode == "mined":
            noise_samples.append(noise_rate(dev))
            results["noise_macs_per_gpu_s"] = float(np.mean([s[0] for s in noise_samples]))
            results["noise_macs_per_wall_s"] = float(np.mean([s[1] for s in noise_samples]))
            results["noise_gpu_spread"] = [float(min(s[0] for s in noise_samples)), float(max(s[0] for s in noise_samples))]
        hashed = sum(ml.hashed_macs for ml in mls) / reps
        useful = sum(ml.useful_macs for ml in mls) / reps
        mining_t = sum(v for kname, v in by.items() if kname.startswith(MINING_KERNELS))
        gemm_t = by.get("noisy_gemm", 0.0)
        r = {"gpu_s": t_gpu, "wall_s": t_wall, "hashed_macs": hashed, "linear_macs": useful,
             "tokens_per_gpu_s": args.tokens / t_gpu, "tokens_per_wall_s": args.tokens / t_wall,
             "noisy_gemm_s": gemm_t, "mining_aux_s": mining_t - gemm_t, "other_s": t_gpu - mining_t,
             "top_kernels": sorted(((kname[:60], v) for kname, v in by.items()), key=lambda x: -x[1])[:12]}
        if args.mode == "mined":
            r["lambda_gpu"] = hashed / t_gpu / results["noise_macs_per_gpu_s"]
            r["lambda_wall"] = hashed / t_wall / results["noise_macs_per_wall_s"]
        results[job] = r
        print(json.dumps({job: {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()
                                if kk != "top_kernels"}}), flush=True)
    print(json.dumps({kk: v for kk, v in results.items() if kk not in ("embed", "score")}), flush=True)
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
