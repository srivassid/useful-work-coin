"""Export a model's canonical weight rows (the rows MODEL_REG's weights root commits to) for
the Go one-step builder (`uwcctl onestep`): a provider proving one element must supply the
weight rows the referee reads, each with a Merkle proof.

  <dir>/rows.bin      rows in weights-tree order; linear rows (qkv, o, gu, down) as int8 values
                      followed by the 4-byte scale, all others as 4-byte words
  <dir>/offsets.bin   uint64 start offset of each row, plus the end
  <dir>/leaves.bin    32-byte weights-tree leaves (for proofs)
  <dir>/meta.json     weights root and row count
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import time

import numpy as np

from detrt.stage import _leaf_hash, merkle_root, row_root, weight_rows

from .model_info import load

LINEAR = ("qkv", "o", "gu", "down", "lm_head")


def export(m, out: str, gen: bool = False) -> dict:
    os.makedirs(out, exist_ok=True)
    offs, leaves = [0], []
    with open(os.path.join(out, "rows.bin"), "wb") as f:
        rows = weight_rows(m)
        if gen:  # a generation model's tree continues with the final norm and lm_head
            from detgen.weights import gen_weight_rows
            rows = gen_weight_rows(m)
        for i, (name, row) in enumerate(rows):
            leaves.append(_leaf_hash(b"w" + struct.pack("<I", i) + row_root(row)))
            if name.split(".")[-1] in LINEAR:
                b = row[:-1].astype(np.int8).tobytes() + row[-1:].astype("<i4").tobytes()
            else:
                b = row.astype("<f4" if row.dtype == np.float32 else "<i4").tobytes()
            f.write(b)
            offs.append(offs[-1] + len(b))
    np.asarray(offs, dtype="<u8").tofile(os.path.join(out, "offsets.bin"))
    with open(os.path.join(out, "leaves.bin"), "wb") as f:
        f.write(b"".join(leaves))
    meta = {"weights_root": merkle_root(leaves).hex(), "rows": len(leaves)}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model json from agent.model_info")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", help="cuda:N: quantized weights on that GPU (a CPU runtime also keeps "
                    "float64 copies of every matrix: 8 bytes per weight). The rows are the same either way.")
    a = ap.parse_args()
    info = json.load(open(a.model))
    t = time.time()
    m = load(info["model_dir"], a.device, info["rotation_block"], info["cfg"]["MaxPos"])
    meta = export(m, a.out, gen=info.get("gen", False))
    ok = meta["weights_root"] == info["weights_root"]
    print(json.dumps(meta), "matches MODEL_REG" if ok else "DOES NOT MATCH MODEL_REG", f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
