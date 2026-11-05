"""Qwen3-Embedding: bf16 vs the integer-only {-64..64} runtime (bit-identical to the mined path)."""
import argparse
import json

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer

from uwc_ampere import model as M


@torch.no_grad()
def embed(mdl, batches, dev):
    out = []
    for b in batches:
        h = mdl(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev)).last_hidden_state
        out.append(F.normalize(h[:, -1].float(), dim=-1).cpu())  # left padding -> last token
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--hadamard", default="128,512")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    texts = [t.strip() for t in load_from_disk(a.data)["test"]["text"] if len(t.strip()) > 300][: a.n]
    batches = [tok(texts[i:i + 16], padding=True, truncation=True, max_length=256, return_tensors="pt")
               for i in range(0, len(texts), 16)]
    ref_m = AutoModel.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto",
                                      max_memory={0: "4GiB", 1: "7GiB"}).eval()
    E_ref = embed(ref_m, batches, "cuda:0")
    print("ref done", torch.cuda.max_memory_allocated(0) / 1e9, flush=True)
    del ref_m
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("after free", torch.cuda.memory_allocated(0) / 1e9, flush=True)
    allres = {}
    for had in [int(h) for h in a.hadamard.split(",")]:
        q_m = AutoModel.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu").eval()
        M.convert_qwen3(q_m, "cuda:0", mode="int", hadamard_block=had, mine_lm_head=False)
        q_m.to("cuda:0")
        E_q = embed(q_m, batches, "cuda:0")
        del q_m
        torch.cuda.empty_cache()
        allres[had] = compare(E_ref, E_q, len(texts))
        print(json.dumps({"hadamard": had, **allres[had]}), flush=True)
    if a.out:
        json.dump({"model": a.model, "texts": len(texts), "results": allres}, open(a.out, "w"), indent=1)


def compare(E_ref, E_q, n):
    cos = (E_ref * E_q).sum(-1)
    S_ref, S_q = E_ref @ E_ref.T, E_q @ E_q.T
    eye = torch.eye(n, dtype=torch.bool)
    top_ref = S_ref.masked_fill(eye, -2).topk(10, dim=1).indices
    top_q = S_q.masked_fill(eye, -2).topk(10, dim=1).indices
    overlap = sum(len(set(a_.tolist()) & set(b_.tolist())) for a_, b_ in zip(top_ref, top_q)) / (10 * n)
    top1 = (top_ref[:, 0] == top_q[:, 0]).float().mean().item()
    off = ~eye
    x, y = S_ref[off], S_q[off]
    rx, ry = x.argsort().argsort().float(), y.argsort().argsort().float()
    spearman = torch.corrcoef(torch.stack([rx, ry]))[0, 1].item()
    return {"cos_mean": cos.mean().item(), "cos_min": cos.min().item(), "top1_neighbor_agreement": top1,
            "top10_overlap": overlap, "pairwise_sim_spearman": spearman}


if __name__ == "__main__":
    main()
