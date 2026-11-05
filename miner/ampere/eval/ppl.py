"""WikiText-2 perplexity: bf16 vs the integer-only {-64..64} runtime (W7A7 per-channel/per-token).

The "int" path is bit-identical to the mined path (tests/test_model.py), so this is the
quality the coin's useful work actually delivers.
"""
import argparse
import json
import math

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from uwc_ampere import model as M


@torch.no_grad()
def ppl(mdl, ids, seq, dev, max_windows):
    nll, cnt = 0.0, 0
    nw = min(max_windows, (ids.numel() - 1) // seq)
    for w in range(nw):
        x = ids[w * seq:(w + 1) * seq + 1].to(dev)
        h = mdl.model(input_ids=x[None, :-1]).last_hidden_state[0]
        for s in range(0, seq, 1024):
            lg = mdl.lm_head(h[s:s + 1024]).float()
            nll += torch.nn.functional.cross_entropy(lg, x[1 + s:1 + s + 1024], reduction="sum").item()
        cnt += seq
    return math.exp(nll / cnt), cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="models/wikitext2")
    ap.add_argument("--variants", default="bf16,int:0,int:128,int:512")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=1000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    text = "\n\n".join(load_from_disk(a.data)["test"]["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    res = {"model": a.model, "seq": a.seq}
    for v in a.variants.split(","):
        mode, _, had = v.partition(":")
        mdl = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu")
        mdl.eval()
        M.convert_qwen3(mdl, dev, mode=mode, hadamard_block=int(had or 0))
        mdl.to(dev)
        p, n = ppl(mdl, ids, a.seq, dev, a.max_windows)
        res[v] = p
        print(json.dumps({"variant": v, "ppl": round(p, 4), "tokens": n}), flush=True)
        del mdl
        torch.cuda.empty_cache()
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
