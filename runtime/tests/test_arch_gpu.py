"""A real Llama / Mistral (or Qwen3) checkpoint on a GPU (Rules 4):

1. detgpu.job_outputs equals detrt.fast.run_stages bit for bit, every layer group, several
   batch shapes (the fused qkv_post without q/k norm, rows over 48 KB of shared memory);
2. detgen.fast generation and head outputs equal the reference (greedy and sampled);
3. W7A7 quality: perplexity on WikiText-2 test chunks, deterministic runtime vs bf16 transformers.

Output digests are printed and written to --out, so another GPU family can be compared later.

  python tests/test_arch_gpu.py MODEL_DIR --groups 7 --device cuda:0 [--hadamard 128] [--ppl-chunks 16]
"""
import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from detrt import core  # noqa: E402
from detrt.qwen3 import DetQwen3  # noqa: E402

DATASET = os.path.expanduser("~/Documents/useful-work-coin/models/wikitext2")


def reference_outputs(m, seqs, G):
    from detrt.fast import run_stages
    lpg = len(m.layers) // G
    outs, xs = [], None
    for g in range(G):
        ys = run_stages(m, [torch.tensor(s) for s in seqs] if g == 0 else None, xs, g * lpg, (g + 1) * lpg, keep_rows=False)
        outs.append(np.stack(ys))
        xs = [torch.from_numpy(y) for y in ys]
    return outs


def digest(arrays) -> str:
    return hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)).hexdigest()[:16]


@torch.no_grad()
def hf_ppl(hf, chunks, dev) -> float:
    nll, n = 0.0, 0
    for c in chunks:
        ids = torch.tensor([c], device=dev)
        lp = torch.log_softmax(hf(ids).logits.float()[0, :-1], -1)
        nll -= lp.gather(-1, ids[0, 1:, None]).sum().item()
        n += len(c) - 1
    return math.exp(nll / n)


@torch.no_grad()
def det_ppl(m, chunks) -> float:
    nll, n = 0.0, 0
    for c in chunks:
        ids = torch.tensor([c], device=m.device)
        lp = m.logprobs(m.forward(ids, m.new_cache(1, len(c))))[0, :-1]
        nll -= lp.gather(-1, ids[0, 1:, None]).double().sum().item()
        n += len(c) - 1
    return math.exp(nll / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--hadamard", type=int, default=128)
    ap.add_argument("--max-pos", type=int, default=1024)
    ap.add_argument("--ppl-chunks", type=int, default=16)
    ap.add_argument("--ppl-len", type=int, default=512)
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    from agent.inputs import wikitext_sequences
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    core.setup_determinism()
    res = {"model": a.model, "gpu": torch.cuda.get_device_name(dev), "hadamard": a.hadamard, "groups": a.groups}
    t = time.time()
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu").eval()
    print(f"{hf.config.model_type}: {sum(p.numel() for p in hf.parameters()) / 1e9:.2f}B params, loaded in {time.time() - t:.0f}s", flush=True)
    chunks = [] if a.skip_ppl else wikitext_sequences(DATASET, a.model, a.ppl_len, a.ppl_chunks)
    if chunks:
        hf.to(dev)
        res["ppl_bf16"] = hf_ppl(hf, chunks, dev)
        hf.to("cpu")
        torch.cuda.empty_cache()
        print(f"bf16 perplexity over {len(chunks)} x {a.ppl_len} tokens: {res['ppl_bf16']:.4f}", flush=True)
    t = time.time()
    m = DetQwen3(hf, dev, hadamard_block=a.hadamard, mine=False, max_pos=a.max_pos)
    del hf
    gc.collect()
    res["arch"], res["qk_norm"], res["rope_sha"] = m.arch, m.qk_norm, m.rope_sha
    print(f"deterministic runtime: arch {m.arch}, qk_norm {m.qk_norm}, D {m.D} I {m.layers[0]['gu'].n // 2} L {len(m.layers)} "
          f"V {m.embed.shape[0]}, built in {time.time() - t:.0f}s, GPU memory {torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB", flush=True)
    assert len(m.layers) % a.groups == 0, "layers do not split into groups"

    # 1. fast GPU engine == reference
    from detgpu.stages import job_outputs
    V = m.embed.shape[0]
    rng = np.random.default_rng(0)
    cases = [("random 3x128", rng.integers(0, V, (3, 128)).tolist()), ("random 1x512", rng.integers(0, V, (1, 512)).tolist()),
             ("random 2x300", rng.integers(0, V, (2, 300)).tolist())]
    if chunks:
        cases.insert(0, ("wiki 4x128", [c[:128] for c in chunks[:4]]))
    res["stages"] = {}
    for name, seqs in cases:
        t = time.time()
        ref = reference_outputs(m, seqs, a.groups)
        t_ref = time.time() - t
        torch.cuda.synchronize()
        t = time.time()
        got = [g.cpu().numpy() for g in job_outputs(m, torch.tensor(seqs), a.groups)]
        torch.cuda.synchronize()
        t_fast = time.time() - t
        for g, (x, y) in enumerate(zip(ref, got)):
            assert x.shape == y.shape, (name, g, x.shape, y.shape)
            nd = int((x.view(np.uint32) != y.view(np.uint32)).sum())
            assert nd == 0, f"{name}: group {g}: {nd} of {x.size} values differ"
        res["stages"][name] = digest(got)
        print(f"{name}: all {a.groups} groups bit-identical (reference {t_ref:.1f}s, detgpu {t_fast:.2f}s), sha256 {digest(got)}", flush=True)

    # 2. fast generation == reference
    from detgen import fast
    from detgen.generate import generate
    from detgen.head import Sampling, run_head_stage
    prompts = [rng.integers(0, V, n).tolist() for n in (9, 5, 13)]
    res["gen"] = {}
    for sname, s in (("greedy", Sampling(0)), ("t0.8", Sampling(800, bytes(range(32))))):
        t = time.time()
        got = fast.generate(m, prompts, 40, s)
        t_fast = time.time() - t
        for p, g in zip(prompts, got):
            assert g == generate(m, p, 40, s), f"{sname}: fast generation differs from the reference"
        r2 = job_outputs(m, torch.tensor(got, device=dev), a.groups)[-1]
        out = fast.head_outputs(m, r2, s)
        for b in range(len(got)):
            assert np.array_equal(out[b], run_head_stage(m, r2[b].cpu(), s, rows=False).output), f"{sname}: head outputs differ"
        res["gen"][sname] = hashlib.sha256(json.dumps(got).encode()).hexdigest()[:16]
        print(f"generation {sname}: 3 prompts x 40 tokens, fast == reference (fast {t_fast:.1f}s), head outputs equal, "
              f"tokens sha256 {res['gen'][sname]}", flush=True)

    # 3. quality
    if chunks:
        t = time.time()
        res["ppl_det"] = det_ppl(m, chunks)
        rel = res["ppl_det"] / res["ppl_bf16"] - 1
        print(f"deterministic W7A7 (Hadamard {a.hadamard}) perplexity {res['ppl_det']:.4f} vs bf16 {res['ppl_bf16']:.4f}: "
              f"{100 * rel:+.2f}% ({time.time() - t:.0f}s)", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print("ARCH GPU CHECKS PASS")


if __name__ == "__main__":
    main()
