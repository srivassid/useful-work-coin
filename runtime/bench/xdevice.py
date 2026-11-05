"""Build-step-2 exit check on a real model: run the same job on a device and dump hashes.

python -m bench.xdevice --device cuda|cpu --out results/xdev_<name>.json
Compare with: python -m bench.xdevice_compare results/xdev_*.json
"""
import argparse
import json
import os
import time

import torch

from detrt import core
from detrt.qwen3 import DetQwen3, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/Documents/useful-work-coin/models/Qwen3-1.7B"))
    ap.add_argument("--data", default=os.path.expanduser("~/Documents/useful-work-coin/models/wikitext2"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-mine", action="store_true")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--new", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    core.setup_determinism()
    torch.set_num_threads(os.cpu_count())
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    text = "\n\n".join(load_from_disk(a.data)["test"]["text"][:400])
    ids_all = tok(text, return_tensors="pt").input_ids[0]
    prompt = torch.stack([ids_all[:a.prompt_len], ids_all[1000:1000 + a.prompt_len]]).to(a.device)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu").eval()
    t0 = time.time()
    m = DetQwen3(hf, a.device, hadamard_block=512, mine=not a.no_mine)
    del hf
    mined = 0
    if a.device == "cuda" and not a.no_mine:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "miner", "ampere"))
        from uwc_ampere import pearl_glue as pg
        hdr = pg.header(0x1D00FFFF, 7)
        for lin in m.linears():
            lin.setup_block(lambda k: pg.job_key(hdr, pg.mining_config(k)),
                            lambda k: pg.bound_words(pg.jackpot_bound(hdr.nbits, pg.mining_config(k))))
        mined = sum(1 for lin in m.linears() if lin.mine)
    res = {"device": torch.cuda.get_device_name() if a.device == "cuda" else "cpu",
           "mined_linears": mined, "rope_table_sha": m.rope_sha, "build_s": time.time() - t0}

    t0 = time.time()
    lh = []
    h = m.forward(prompt, m.new_cache(2, a.prompt_len), 0, layer_hashes=lh)
    res["prefill_layer_sha"] = lh
    res["prefill_hidden_sha"] = sha(h)
    lp = m.logprobs(h[:, :-1])
    tgt = prompt[:, 1:]
    tok_lp = lp.gather(-1, tgt[..., None])[..., 0]
    res["score_token_logprob_sha"] = sha(tok_lp)
    res["score_mean_nll"] = float(-tok_lp.double().mean())
    res["prefill_s"] = time.time() - t0

    t0 = time.time()
    toks, step_h = m.generate_greedy(prompt, a.new)
    res["generated_tokens"] = toks.tolist()
    res["generated_text"] = [tok.decode(t) for t in toks.tolist()]
    res["decode_step_hidden_sha"] = [sha(step_h[:, i]) for i in range(a.new)]
    res["decode_s"] = time.time() - t0

    full = torch.cat([prompt, toks[:, :-1]], 1)
    h_tf = m.forward(full, m.new_cache(2, full.shape[1]), 0)
    tf = h_tf[:, a.prompt_len - 1:]
    res["teacher_forced_equals_decode"] = bool(torch.equal(tf, step_h))
    res["teacher_forced_hidden_sha"] = [sha(tf[:, i]) for i in range(a.new)]
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("_sha") or k == "prefill_hidden_sha"}, indent=1))


if __name__ == "__main__":
    main()
