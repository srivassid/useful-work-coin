"""Job inputs and their commitments, bit-exact with node/jobstate (TokenLeaf, UnitLeaf)."""
from __future__ import annotations

import hashlib
import json
import struct

import numpy as np

from detrt.stage import _leaf_hash, merkle_root, merkle_proof, row_root


def token_leaf(batch: int, tokens: list[int]) -> bytes:
    # uint32 words (a generation row's sentinel is 0xFFFFFFFF), hashed as their int32 bit patterns
    return _leaf_hash(b"tok" + struct.pack("<I", batch) + row_root(np.asarray(tokens, dtype=np.uint32).view(np.int32)))


def input_root(batches: list[list[int]]) -> bytes:
    """JOB_POST input_root: one leaf per batch (one sequence per batch)."""
    return merkle_root([token_leaf(b, t) for b, t in enumerate(batches)])


def tag_hash(tag: str, *parts: bytes) -> bytes:
    h = hashlib.sha256(tag.encode())
    for p in parts:
        h.update(p)
    return h.digest()


def unit_leaf(i: int, root: bytes) -> bytes:
    return tag_hash("unit", struct.pack("<I", i), root)


def canary_leaf(unit: int, expected_roots: list[bytes], salt: bytes) -> bytes:
    return tag_hash("canary", struct.pack("<I", unit), merkle_root(expected_roots), salt)


def wikitext_sequences(dataset_dir: str, tokenizer_dir: str, seq_len: int, count: int, offset: int = 0,
                       split: str = "test") -> list[list[int]]:
    """Consecutive seq_len-token chunks of a WikiText-2 split, public text for the network's jobs."""
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    text = "\n\n".join(load_from_disk(dataset_dir)[split]["text"])
    ids = tok(text, return_tensors=None)["input_ids"]
    out = []
    for i in range(offset, offset + count):
        chunk = ids[i * seq_len:(i + 1) * seq_len]
        if len(chunk) < seq_len:
            break
        out.append(chunk)
    return out


def save_units(path: str, seqs: list[list[int]], per_unit: int = 1) -> dict:
    """A pool work list: unit i is per_unit consecutive sequences (one batch each). Writes the
    units and returns the commitment."""
    groups = [seqs[i:i + per_unit] for i in range(0, len(seqs) - per_unit + 1, per_unit)]
    roots = [input_root(g) for g in groups]
    leaves = [unit_leaf(i, r) for i, r in enumerate(roots)]
    units = [({"tokens": g[0]} if per_unit == 1 else {"batches": g}) | {"input_root": r.hex()} for g, r in zip(groups, roots)]
    doc = {"units": units, "work_root": merkle_root(leaves).hex()}
    with open(path, "w") as f:
        json.dump(doc, f)
    return doc


def check_units(doc: dict, work_root: str) -> bool:
    """A work list fetched from anywhere: every unit's input_root matches its tokens and the
    units commit to work_root (the category's on-chain root)."""
    try:
        roots = []
        for u in doc["units"]:
            b = u["batches"] if "batches" in u else [u["tokens"]]
            r = input_root(b)
            if r.hex() != u["input_root"]:
                return False
            roots.append(r)
        return merkle_root([unit_leaf(i, r) for i, r in enumerate(roots)]).hex() == work_root == doc.get("work_root")
    except (KeyError, TypeError, ValueError):
        return False


def unit_proof(doc: dict, i: int) -> list[str]:
    leaves = [unit_leaf(j, bytes.fromhex(u["input_root"])) for j, u in enumerate(doc["units"])]
    return [h.hex() for h in merkle_proof(leaves, i)]
