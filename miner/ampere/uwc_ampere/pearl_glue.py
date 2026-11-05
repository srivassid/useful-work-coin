"""Glue to Pearl's Rust bindings (pearl_mining, built from ext/pearl/py-pearl-mining).

Everything consensus-relevant (proof parsing, Merkle proofs, verification) stays in
Pearl's code. This module only builds the inputs it expects.
"""
from __future__ import annotations

import numpy as np
import pearl_mining as pm
from blake3 import blake3

# Hash-tile pattern for the Ampere kernel: one tile = the 8 x 16 accumulator values a
# thread holds for a 64 x 64 warp tile of mma.sync.m16n8k32 (rows g + 8i, cols 2t + {0,1} + 8j).
AMPERE_ROWS = [8 * i for i in range(8)]
AMPERE_COLS = sorted([8 * j + e for j in range(8) for e in (0, 1)])

# Pearl's Hopper default (16 x 256 warp slice per thread), kept for reference.
PEARL_ROWS = [0, 8]
PEARL_COLS = sorted([8 * j + e for j in range(32) for e in (0, 1)])


def header(nbits: int = 0x1D2FFFFF, seed: int = 0) -> pm.IncompleteBlockHeader:
    rng = np.random.default_rng(seed)
    return pm.IncompleteBlockHeader(
        version=0,
        prev_block=rng.bytes(32),
        merkle_root=rng.bytes(32),
        timestamp=0x66666666 + seed,
        nbits=nbits,
    )


def mining_config(k: int, rank: int = 128, rows=AMPERE_ROWS, cols=AMPERE_COLS) -> pm.MiningConfiguration:
    return pm.MiningConfiguration(
        common_dim=k,
        rank=rank,
        mma_type=pm.MMAType.Int7xInt7ToInt32,
        rows_pattern=pm.PeriodicPattern.from_list(rows),
        cols_pattern=pm.PeriodicPattern.from_list(cols),
        moe=None,
    )


def job_key(hdr: pm.IncompleteBlockHeader, cfg: pm.MiningConfiguration) -> bytes:
    return blake3(bytes(hdr.to_bytes()) + bytes(cfg.to_bytes())).digest()


def nbits_to_target(nbits: int) -> int:
    exp, mant = nbits >> 24, nbits & 0xFFFFFF
    if mant == 0 or exp == 0 or mant & 0x800000:
        return 0
    return mant >> (8 * (3 - exp)) if exp <= 3 else mant << (8 * (exp - 3))


def jackpot_bound(nbits: int, cfg: pm.MiningConfiguration) -> int:
    """min(extract_difficulty_bound, penalized bound); both must hold for a valid proof."""
    tile = cfg.rows_pattern.size * cfg.cols_pattern.size
    k, r = cfg.common_dim, cfg.rank
    t = nbits_to_target(nbits)
    plain_b = min(t * tile * k, 2**256 - 1)
    pen = min(t * tile * (k // r) * pm.PENALTY_BASE_RANK, 2**256 - 1)
    return min(plain_b, pen)


def bound_words(bound: int) -> np.ndarray:
    return np.array([(bound >> (32 * i)) & 0xFFFFFFFF for i in range(8)], dtype=np.uint32)


def build_plain_proof(A: np.ndarray, Bt: np.ndarray, key: bytes, rank: int,
                      a_rows: list[int], b_cols: list[int]) -> pm.PlainProof:
    """Mirror of miner_base.block_submission.create_proof, on un-noised A and B^T."""
    def tree(t: np.ndarray) -> pm.MerkleTree:
        return pm.MerkleTree(data=bytes(pm.pad_to_chunk_boundary(np.ascontiguousarray(t, np.int8).tobytes())), key=key)

    ta, tb = tree(A), tree(Bt)
    pa = pm.MatrixMerkleProof(
        proof=ta.get_multileaf_proof(pm.MerkleTree.compute_leaf_indices_from_rows(a_rows, tuple(A.shape))),
        row_indices=a_rows)
    pb = pm.MatrixMerkleProof(
        proof=tb.get_multileaf_proof(pm.MerkleTree.compute_leaf_indices_from_rows(b_cols, tuple(Bt.shape))),
        row_indices=b_cols)
    m, k = A.shape
    n = Bt.shape[0]
    return pm.PlainProof(m=m, n=n, k=k, noise_rank=rank, a_merkle_proof=pa, bt_merkle_proof=pb, moe=None)


def verify(hdr, proof, nbits_override=None) -> tuple[bool, str]:
    return pm.verify_plain_proof_v3(hdr, proof, nbits_override=nbits_override)
