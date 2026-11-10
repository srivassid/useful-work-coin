"""MODEL_REG data for a Qwen3 checkpoint under the deterministic runtime.

  python -m agent.model_info --model ~/models/Qwen3-1.7B --groups 7 --max-pos 512 > model.json
  uwcctl modelreg <key> model.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

from detrt import core
from detrt.qwen3 import DetQwen3
from detrt.stage import model_cfg, weights_tree

DETRT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "detrt")
DETGEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "detgen")
GEN_REFERENCE = ("__init__.py", "generate.py", "head.py", "weights.py")  # detgen's reference (not the GPU fast path)


def runtime_hash(gen: bool = False) -> bytes:
    """sha256 over the reference runtime's sources (sorted by name): the runtime_hash in MODEL_REG.
    A generation model's hash continues over detgen's reference files."""
    h = hashlib.sha256()
    for name in sorted(os.listdir(DETRT_DIR)):
        if name.endswith(".py"):
            h.update(name.encode() + b"\0")
            h.update(open(os.path.join(DETRT_DIR, name), "rb").read())
    if gen:
        for name in GEN_REFERENCE:
            h.update(b"detgen/" + name.encode() + b"\0")
            h.update(open(os.path.join(DETGEN_DIR, name), "rb").read())
    return h.digest()


def load(model_dir: str, device: str, hadamard_block: int, max_pos: int, mine: bool = False) -> DetQwen3:
    from transformers import AutoModelForCausalLM
    core.setup_determinism()
    hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cpu").eval()
    return DetQwen3(hf, device, hadamard_block=hadamard_block, mine=mine, max_pos=max_pos)


def go_cfg(m: DetQwen3, max_pos: int) -> dict:
    c = model_cfg(m)
    return {"D": c["D"], "H": c["H"], "Hkv": c["Hkv"], "Hd": c["hd"], "I": c["I"], "L": c["L"], "V": c["V"],
            "Eps": c["eps"], "Bs": c["bs"], "MaxPos": max_pos, "Arch": c["arch"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--groups", type=int, required=True, help="layer groups (stages per batch)")
    ap.add_argument("--hadamard", type=int, default=512)
    ap.add_argument("--max-pos", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gen", action="store_true", help="a generation model (Rules 3): commits the final norm and lm_head too")
    a = ap.parse_args()
    t = time.time()
    m = load(a.model, a.device, a.hadamard, a.max_pos)
    if len(m.layers) % a.groups:
        sys.exit(f"{len(m.layers)} layers do not split into {a.groups} groups")
    if a.gen:
        from detgen.weights import gen_weights_tree
        root, leaves = gen_weights_tree(m)
    else:
        root, leaves = weights_tree(m)
    out = {"weights_root": root.hex(), "runtime_hash": runtime_hash(a.gen).hex(), "layer_groups": a.groups,
           "rotation_block": m.layers[0]["qkv"].bs, "cfg": go_cfg(m, a.max_pos), "weight_rows": len(leaves),
           "model_dir": os.path.abspath(a.model), "gen": a.gen}
    print(json.dumps(out, indent=1))
    print(f"computed in {time.time() - t:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
