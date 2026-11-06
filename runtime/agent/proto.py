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


def stage_out_leaf(t: int, rroot: bytes) -> bytes:
    """jobstate.StageOutLeaf: the output-tree leaf of token t."""
    return _leaf_hash(b"sout" + u32(t) + rroot)


def stage_out_leaves(rows) -> list[bytes]:
    return [stage_out_leaf(t, row_root(np.ascontiguousarray(r, dtype=np.float32))) for t, r in enumerate(rows)]


def stage_output_root(rows) -> bytes:
    """jobstate.StageOutputRoot: what a stage is committed by (its T output rows)."""
    return merkle_root(stage_out_leaves(rows))


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


def multi_proofs(leaves: list[bytes], indices) -> dict[int, list[bytes]]:
    """Merkle proofs (as detrt.stage.merkle_proof / jobstate.MerkleProof) for many leaves at
    once, computing each subtree root a single time: O(len(leaves)) hashes in total."""
    from detrt.stage import _node, _split
    want = sorted(set(indices))
    out: dict[int, list[bytes]] = {i: [] for i in want}

    def rec(lo: int, hi: int, idx: list[int]) -> bytes:
        n = hi - lo
        if n == 1:
            return leaves[lo]
        k = _split(n)
        left = [i for i in idx if i < lo + k]
        right = [i for i in idx if i >= lo + k]
        lroot = rec(lo, lo + k, left)
        rroot = rec(lo + k, hi, right)
        for i in left:
            out[i].append(rroot)
        for i in right:
            out[i].append(lroot)
        return _node(lroot, rroot)

    if leaves:
        rec(0, len(leaves), want)
    return out


def verify_merkle(root: bytes, leaf: bytes, i: int, n: int, proof: list[bytes]) -> bool:
    """jobstate.VerifyMerkleProof."""
    from detrt.stage import _node, _split

    def walk(leaf, i, n, proof):
        if n == 1:
            return leaf, len(proof) == 0
        if not proof:
            return b"", False
        k = _split(n)
        sib, rest = proof[-1], proof[:-1]
        if i < k:
            h, ok = walk(leaf, i, k, rest)
            return _node(h, sib), ok
        h, ok = walk(leaf, i - k, n - k, rest)
        return _node(sib, h), ok
    if not 0 <= i < n:
        return False
    h, ok = walk(leaf, i, n, proof)
    return ok and h == root


def check_stage_input(pkg: dict, stage: int, stage_root: bytes, stages: int):
    """A served stage input, checked against the provider's on-chain commitment: the previous
    stage's (salt, output root) must open stage_root, and every row must open that output root.
    Returns the (T, D) float32 rows, or raises ValueError. A provider cannot frame a verifier
    with a wrong input, because only the committed rows pass."""
    import base64
    rv = pkg["reveal"]
    if rv["stage"] != stage - 1:
        raise ValueError("reveal is not the previous stage")
    salt, root = bytes.fromhex(rv["salt"]), bytes.fromhex(rv["root"])
    if not verify_merkle(stage_root, stage_leaf(stage - 1, salt, root), stage - 1, stages,
                         [bytes.fromhex(h) for h in rv["proof"]]):
        raise ValueError("previous stage does not open the stage commitment")
    T, D = pkg["shape"]
    rows = np.frombuffer(base64.b64decode(pkg["rows"]), dtype="<f4").reshape(T, D).copy()
    for t in range(T):
        if not verify_merkle(root, stage_out_leaf(t, row_root(rows[t])), t, T,
                             [bytes.fromhex(h) for h in pkg["proofs"][t]]):
            raise ValueError(f"row {t} does not open the previous stage's output")
    return rows
