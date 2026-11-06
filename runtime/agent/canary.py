"""A pool category with hidden canary units (the curator knows their answers).

  python -m agent.canary --units 100 --offset 400 --canaries 5 --dummies 15 --cap-bp 3000 --device cuda:0
Builds the work list, computes the canaries' expected stage roots with the reference runtime,
hides their leaves among dummies, writes the curator's secrets to ~/.uwc/devnet/canary/ and has
the council open the category (council0 proposes, council1 approves).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import secrets

from detrt.stage import merkle_proof, merkle_root

from .chain import Chain
from .compute import Runtime
from .fe import wait_mempool
from .inputs import canary_leaf, save_units, wikitext_sequences

KEYS = os.path.join(os.path.expanduser(os.environ.get("UWC_HOME", "~/.uwc/devnet")), "keys")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devnet", default=os.path.expanduser("~/Documents/useful-work-coin/devnet"))
    ap.add_argument("--units", type=int, default=100)
    ap.add_argument("--offset", type=int, default=400)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--canaries", type=int, default=5)
    ap.add_argument("--dummies", type=int, default=15)
    ap.add_argument("--cap-bp", type=int, default=3000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--per-unit", type=int, default=1, help="sequences per unit")
    a = ap.parse_args()
    chain = Chain(a.devnet)
    model = os.path.join(a.devnet, "models", "qwen3-1.7b.json")
    info = json.load(open(model))
    model_id = json.load(open(model.replace(".json", ".reg.json")))["model"]
    seqs = wikitext_sequences(os.path.expanduser("~/Documents/useful-work-coin/models/wikitext2"), info["model_dir"],
                              a.seq_len, a.units * a.per_unit, a.offset)
    out = os.path.join(a.devnet, "pool", f"wiki{a.seq_len}-canary")
    doc = save_units(out + ".units.json", seqs, a.per_unit)
    rt = Runtime(model, a.device)
    picks = random.sample(range(len(doc["units"])), a.canaries)
    salt = secrets.token_bytes(32)
    entries = []
    for u in picks:
        unit = doc["units"][u]
        roots = [r for jb in rt.job_outputs(unit.get("batches") or [unit["tokens"]]) for r in jb[0]]
        entries.append((u, roots, canary_leaf(u, roots, salt)))
    leaves = [(e[0], e[1], e[2]) for e in entries] + [(None, None, secrets.token_bytes(32)) for _ in range(a.dummies)]
    random.shuffle(leaves)
    hashes = [x[2] for x in leaves]
    canaries = {str(u): {"leaf": i, "roots": [r.hex() for r in roots], "proof": [h.hex() for h in merkle_proof(hashes, i)]}
                for i, (u, roots, _) in enumerate(leaves) if u is not None}
    curator = chain.key_info(os.path.join(KEYS, "curator.json"))["account"]
    spec = {"Model": model_id, "WorkRoot": doc["work_root"], "Units": len(doc["units"]), "StagesPerUnit": info["layer_groups"] * a.per_unit,
            "SeqLen": a.seq_len, "K": 2, "N": 2, "UnitDeadline": 120, "CapBP": a.cap_bp, "InitPrice": 2_000_000_000,
            "Grant": 0, "Curator": curator, "CanaryRoot": merkle_root(hashes).hex(), "CanaryLeaves": len(hashes)}
    json.dump(spec, open(out + ".category.json", "w"), indent=1)
    r = chain.run("propose-category", os.path.join(KEYS, "council0.json"), out + ".category.json")
    wait_mempool(chain)
    chain.run("approve", os.path.join(KEYS, "council2.json"), r["proposal"])
    wait_mempool(chain)
    from .inputs import tag_hash
    cid = tag_hash("category", bytes.fromhex(r["proposal"])).hex()
    d = os.path.join(os.path.expanduser(os.environ.get("UWC_HOME", "~/.uwc/devnet")), "canary")
    os.makedirs(d, mode=0o700, exist_ok=True)
    p = os.path.join(d, f"{cid[:16]}.json")
    json.dump({"category": cid, "salt": salt.hex(), "canaries": canaries}, open(p, "w"))
    os.chmod(p, 0o600)
    print(json.dumps({"category": cid, "canary_units": sorted(int(u) for u in canaries), "work_root": doc["work_root"]}))


if __name__ == "__main__":
    main()
