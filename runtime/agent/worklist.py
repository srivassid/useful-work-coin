"""Build a pool category's work list and its POOL_CATEGORY spec.

  python -m agent.worklist --model-id <id> --units 200 --seq-len 128 --out devnet/pool/wiki128
writes <out>.units.json (public work list) and <out>.category.json (for uwcctl propose-category).
"""
from __future__ import annotations

import argparse
import json
import os

from .inputs import save_units, wikitext_sequences


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--units", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--offset", type=int, default=0, help="first chunk of the text")
    ap.add_argument("--groups", type=int, default=7)
    ap.add_argument("--curator", required=True, help="curator account (hex)")
    ap.add_argument("--cap-bp", type=int, default=6000)
    ap.add_argument("--init-price", type=int, default=2_000_000_000)
    ap.add_argument("--grant", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    seqs = wikitext_sequences(a.dataset, a.model_dir, a.seq_len, a.units, a.offset)
    doc = save_units(a.out + ".units.json", seqs)
    spec = {"Model": a.model_id, "WorkRoot": doc["work_root"], "Units": len(seqs), "StagesPerUnit": a.groups,
            "SeqLen": a.seq_len, "K": 2, "N": 2, "UnitDeadline": 120, "CapBP": a.cap_bp, "InitPrice": a.init_price,
            "Grant": a.grant, "Curator": a.curator, "CanaryRoot": "", "CanaryLeaves": 0}
    with open(a.out + ".category.json", "w") as f:
        json.dump(spec, f, indent=1)
    print(json.dumps(spec, indent=1))


if __name__ == "__main__":
    main()
