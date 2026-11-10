"""Write cross-language test fixtures for node/jobstate (Go referee == Python runtime).

python -m bench.export_fixtures --tiny out_tiny.json.gz [--tiny-llama out_llama.json.gz] [--real MODEL_DIR out_real.json.gz]
"""
import argparse
import base64
import gzip
import json
import random

import numpy as np
import torch

from detrt import core
from detrt.qwen3 import DetQwen3
from detrt.stage import model_cfg, run_stage, weight_rows, weights_tree


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).view("<u4").tobytes()).decode()


def name_split(name: str):
    if "." in name and name.split(".")[0].isdigit():
        l, n = name.split(".", 1)
        return n, int(l)
    return name, 0


def export(m: DetQwen3, stages, keep_weight, path, full_weights: bool):
    cfg = model_cfg(m)
    out = {"cfg": dict(D=cfg["D"], H=cfg["H"], Hkv=cfg["Hkv"], Hd=cfg["hd"], I=cfg["I"], L=cfg["L"], V=cfg["V"],
                       EpsBits=int(np.float32(cfg["eps"]).view(np.uint32)), Bs=m.hadamard_block, MaxPos=int(m.cos.shape[0]),
                       Arch=cfg["arch"]),
           "stages": [], "weights": [], "full_weights": full_weights}
    for tr, T, l0, l1, toks, xin in stages:
        out["stages"].append(dict(T=T, L0=l0, L1=l1, tokens=None if toks is None else [int(t) for t in toks],
                                  input=None if xin is None else [b64(r) for r in xin],
                                  rows=[b64(r) for r in tr.rows], root=tr.root.hex()))
    counters = {}
    for name, row in weight_rows(m):
        n, l = name_split(name)
        r = counters.get(name, 0)
        counters[name] = r + 1
        if full_weights or keep_weight(n, l, r):
            out["weights"].append(dict(name=n, layer=l, row=r, w=b64(row)))
    if full_weights:
        out["weights_root"] = weights_tree(m)[0].hex()
    with gzip.open(path, "wt") as f:
        json.dump(out, f)
    print(path, "stages", len(stages), "weight rows", len(out["weights"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", required=True)
    ap.add_argument("--real", nargs=2, default=None)
    ap.add_argument("--tiny-llama", default=None, help="a tiny Llama (Arch 1, Llama 3 rotary scaling) with all weights")
    a = ap.parse_args()
    core.setup_determinism()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=64,
                      rms_norm_eps=1e-6, tie_word_embeddings=True)
    hf = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in hf.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    m = DetQwen3(hf, dev, hadamard_block=64, mine=False, max_pos=64)
    toks = torch.randint(0, 256, (12,), generator=torch.Generator().manual_seed(3))
    s0 = run_stage(m, toks, None, 0, 1)
    s1 = run_stage(m, None, torch.from_numpy(s0.output), 1, 2)
    s01 = run_stage(m, toks, None, 0, 2)
    export(m, [(s0, 12, 0, 1, toks.tolist(), None), (s1, 12, 1, 2, None, s0.output), (s01, 12, 0, 2, toks.tolist(), None)],
           None, a.tiny, full_weights=True)
    if a.tiny_llama:
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(0)
        cfg = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=64,
                          rms_norm_eps=1e-5, tie_word_embeddings=False, rope_theta=500000.0,
                          rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                        "high_freq_factor": 4.0, "original_max_position_embeddings": 16})
        hf = LlamaForCausalLM(cfg).eval()
        with torch.no_grad():
            for p in hf.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        m = DetQwen3(hf, dev, hadamard_block=64, mine=False, max_pos=64)
        assert m.arch == 1
        s0 = run_stage(m, toks, None, 0, 1)
        s1 = run_stage(m, None, torch.from_numpy(s0.output), 1, 2)
        s01 = run_stage(m, toks, None, 0, 2)
        export(m, [(s0, 12, 0, 1, toks.tolist(), None), (s1, 12, 1, 2, None, s0.output), (s01, 12, 0, 2, toks.tolist(), None)],
               None, a.tiny_llama, full_weights=True)
    if a.real:
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(a.real[0], dtype=torch.bfloat16, device_map="cpu").eval()
        m = DetQwen3(hf, dev, hadamard_block=512, mine=False)
        T = 8
        toks = torch.tensor([785, 3974, 13876, 38835, 34208, 916, 279, 15678])
        s0 = run_stage(m, toks, None, 0, 1)
        s1 = run_stage(m, None, torch.from_numpy(s0.output), 1, 2)
        rng = random.Random(7)
        sample = {}
        for l in (0, 1):
            for nm, n in (("qkv", m.layers[l]["qkv"].n), ("o", m.layers[l]["o"].n), ("gu", m.layers[l]["gu"].n),
                          ("down", m.layers[l]["down"].n)):
                sample[(nm, l)] = set(rng.sample(range(n), 24))
        tokset = set(toks.tolist())

        def keep(n, l, r):
            if n == "embed":
                return r in tokset
            if n in ("rope.cos", "rope.sin"):
                return r < T
            if l > 1:
                return False
            if n in ("in_norm", "post_norm", "q_norm", "k_norm"):
                return True
            return r in sample.get((n, l), ())
        export(m, [(s0, T, 0, 1, toks.tolist(), None), (s1, T, 1, 2, None, s0.output)], keep, a.real[1], full_weights=False)


if __name__ == "__main__":
    main()
