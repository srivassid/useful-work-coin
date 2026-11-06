"""Protocol helpers, bit-exact with node/jobstate: stage commitments, sampled positions,
verifier result hashes, pooled outputs. Checked by node/jobstate TestAgentVectors."""
from __future__ import annotations

import hashlib
import struct

import numpy as np

from detrt.stage import _leaf_hash, merkle_proof, merkle_root, row_root


def tag_hash(tag: str, *parts: bytes) -> bytes:
    h = hashlib.sha256(tag.encode())
    for p in parts:
        h.update(p)
    return h.digest()


def u32(v: int) -> bytes:
    return struct.pack("<I", v)


def stage_leaf(stage: int, salt: bytes, root: bytes) -> bytes:
    """Leaf of TRACE_COMMIT's stage_root: sha256(0x00 || "stage" || u32 stage || salt || root)."""
    return _leaf_hash(b"stage" + u32(stage) + salt + root)


def positions(salt: bytes, stages: int, n: int) -> list[int]:
    """The n distinct stages a verifier with this salt rechecks (jobstate.Positions)."""
    if n >= stages:
        return list(range(stages))
    seen, out = set(), []
    i = 0
    while len(out) < n and i < n * 64:
        x = tag_hash("pos", salt, u32(i))
        st = int.from_bytes(x[:8], "little") % stages
        if st not in seen:
            seen.add(st)
            out.append(st)
        i += 1
    st = 0
    while len(out) < n:
        if st not in seen:
            seen.add(st)
            out.append(st)
        st += 1
    return sorted(out)


def result_hash(roots: list[bytes], account: bytes, salt: bytes) -> bytes:
    return tag_hash("result", *roots, account, salt)


def job_id(ref: bytes) -> bytes:
    return tag_hash("job", ref)


def pooled(output: np.ndarray) -> np.ndarray:
    """A sequence's embedding for the devnet's embed job: the mean over tokens of the last
    layer's residual rows, accumulated in float64 in token order, rounded once to float32."""
    return (output.astype(np.float64).sum(axis=0) / output.shape[0]).astype(np.float32)


def output_root(embeddings: list[np.ndarray]) -> bytes:
    """TRACE_COMMIT output_root: Merkle root over the row roots of each batch's embedding."""
    return merkle_root([_leaf_hash(b"out" + u32(b) + row_root(e)) for b, e in enumerate(embeddings)])


class StageCommit:
    """A provider's salted commitment to its stage roots."""

    def __init__(self, roots: list[bytes], salts: list[bytes]):
        self.roots, self.salts = roots, salts
        self.leaves = [stage_leaf(i, s, r) for i, (s, r) in enumerate(zip(salts, roots))]
        self.root = merkle_root(self.leaves)

    def reveal(self, stage: int) -> dict:
        return {"stage": stage, "salt": self.salts[stage].hex(), "root": self.roots[stage].hex(),
                "proof": [h.hex() for h in merkle_proof(self.leaves, stage)]}
