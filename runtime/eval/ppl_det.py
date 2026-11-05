"""WikiText-2 perplexity of the deterministic runtime (same protocol as miner/ampere/eval/ppl.py)."""
import argparse
import json
import math
import os

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from detrt import core
from detrt.qwen3 import DetQwen3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.expanduser("~/Documents/useful-work-coin/models/wikitext2"))
    ap.add_argument("--hadamard", type=int, default=512)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    core.setup_determinism()
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok("\n\n".join(load_from_disk(a.data)["test"]["text"]), return_tensors="pt").input_ids[0]
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu").eval()
    m = DetQwen3(hf, "cuda", hadamard_block=a.hadamard, mine=False)
    del hf
    nll, cnt = 0.0, 0
    for w in range((ids.numel() - 1) // a.seq):
        x = ids[w * a.seq:(w + 1) * a.seq + 1].cuda()
        h = m.forward(x[None, :-1], m.new_cache(1, a.seq), 0)
        lp = m.logprobs(h[0])
        nll -= lp.gather(-1, x[1:, None]).double().sum().item()
        cnt += a.seq
    res = {"model": a.model, "hadamard": a.hadamard, "ppl": math.exp(nll / cnt), "tokens": cnt}
    print(json.dumps(res))
    if a.out:
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
