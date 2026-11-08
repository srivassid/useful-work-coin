"""Write the generation fixture for node/jobstate (Go referee == Python head stage, Rules 3).

python -m bench.export_gen_fixture out_gen_tiny.json.gz
A tiny Qwen3 (vocabulary 1104: two chunks, the second 80 wide), a 7-token prompt, seq_len 16,
temperature 0.8: the generated sequence, both layer stages and the head stage (all rows), the
generation weights root, and vectors of the Gumbel noise and det_log.
"""
import argparse
import base64
import gzip
import hashlib
import json

import numpy as np
import torch

from bench.export_fixtures import b64, name_split
from detgen.generate import generate
from detgen.head import Sampling, gumbel, head_output_root, input_row, run_head_stage, uniform
from detgen.weights import gen_weight_rows, gen_weights_tree
from detrt import core
from detrt.qwen3 import DetQwen3
from detrt.stage import model_cfg, run_stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    a = ap.parse_args()
    core.setup_determinism()
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=1104, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=64,
                      rms_norm_eps=1e-6, tie_word_embeddings=True)
    hf = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in hf.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    m = DetQwen3(hf, "cpu", hadamard_block=64, mine=False, max_pos=64)
    seed = hashlib.sha256(b"gen-fixture-job").digest()
    s = Sampling(800, seed)
    prompt = [int(x) for x in torch.randint(0, 1104, (7,), generator=torch.Generator().manual_seed(5))]
    T = 16
    seq = generate(m, prompt, T, s)
    s0 = run_stage(m, torch.tensor(seq), None, 0, 1)
    s1 = run_stage(m, None, torch.from_numpy(s0.output), 1, 2)
    hd = run_head_stage(m, torch.from_numpy(s1.output), s)
    c = model_cfg(m)
    out = {"cfg": dict(D=c["D"], H=c["H"], Hkv=c["Hkv"], Hd=c["hd"], I=c["I"], L=c["L"], V=c["V"],
                       EpsBits=int(np.float32(c["eps"]).view(np.uint32)), Bs=m.hadamard_block, MaxPos=int(m.cos.shape[0])),
           "stages": [dict(T=T, L0=0, L1=1, tokens=seq, input=None, rows=[b64(r) for r in s0.rows], root=s0.root.hex()),
                      dict(T=T, L0=1, L1=2, tokens=None, input=[b64(r) for r in s0.output], rows=[b64(r) for r in s1.rows],
                           root=s1.root.hex())],
           "weights": [], "full_weights": True}
    counters = {}
    for name, row in gen_weight_rows(m):
        n, l = name_split(name)
        r = counters.get(name, 0)
        counters[name] = r + 1
        out["weights"].append(dict(name=n, layer=l, row=r, w=b64(row)))
    out["weights_root"] = gen_weights_tree(m)[0].hex()
    ts, vs = np.array([0, 3, 14, 1 << 20]), np.array([0, 1, 1103, 151935])
    vec = []
    for t in ts.tolist():
        for v in vs.tolist():
            u = uniform(s, np.array([[t]]), np.array([[v]]))[0, 0]
            gv = gumbel(s, torch.tensor([t]), v, v + 1, "cpu")[0, 0]
            vec.append(dict(t=t, v=v, u=int(np.float32(u).view(np.uint32)), g=int(np.float32(gv.item()).view(np.uint32))))
    logs = []
    for x in [1e-7, 0.001, 0.37, 1.0, 1.5, 2.0, 7.25, 1234.5]:
        xs = torch.tensor([x], dtype=torch.float32)
        logs.append(dict(x=int(xs.numpy().view(np.uint32)[0]), y=int(core.det_log(xs).numpy().view(np.uint32)[0])))
    out["gen"] = dict(temp_milli=s.temp_milli, seed=seed.hex(), prompt_len=len(prompt), input_row=input_row(prompt, T),
                      sequence=seq, layer_groups=2, head_rows=[b64(r) for r in hd.rows], head_root=hd.root.hex(),
                      head_output_root=head_output_root(hd.output).hex(), gumbel=vec, log=logs,
                      key=str(s.key()))
    with gzip.open(a.out, "wt") as f:
        json.dump(out, f)
    print(a.out, "sequence", seq, "weight rows", len(out["weights"]), "head rows", len(hd.rows))


if __name__ == "__main__":
    main()
