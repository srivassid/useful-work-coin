"""Job input and output roots without the deterministic runtime (numpy and hashlib only), for
services that post jobs and check outputs but never compute a stage: the job gateway.

Bit-for-bit the same as agent.inputs.input_root and agent.proto.output_root, which build on
detrt.stage; tests/test_gateway.py checks both against each other. detrt itself stays untouched
because its sources are hashed into the registered runtime_hash.
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np

BLOCK = 1024  # detrt.stage.BLOCK


def _leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + data).digest()


def _node(l: bytes, r: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + l + r).digest()


def merkle_root(leaves: list[bytes]) -> bytes:
    if not leaves:
        return hashlib.sha256(b"").digest()
    if len(leaves) == 1:
        return leaves[0]
    k = 1
    while k << 1 < len(leaves):
        k <<= 1
    return _node(merkle_root(leaves[:k]), merkle_root(leaves[k:]))


def row_root(row: np.ndarray) -> bytes:
    assert row.dtype in (np.float32, np.int32) and row.ndim == 1
    b = row.astype("<f4" if row.dtype == np.float32 else "<i4").tobytes()
    return merkle_root([_leaf_hash(b[i:i + BLOCK]) for i in range(0, max(len(b), 1), BLOCK)])


def input_root(batches: list[list[int]]) -> bytes:
    """JOB_POST input_root: one leaf per batch (one sequence per batch)."""
    return merkle_root([_leaf_hash(b"tok" + struct.pack("<I", i) + row_root(np.asarray(t, dtype=np.uint32).view(np.int32)))
                        for i, t in enumerate(batches)])


def output_root(embeddings: list[np.ndarray]) -> bytes:
    """TRACE_COMMIT output_root: Merkle root over the row roots of each batch's embedding."""
    return merkle_root([_leaf_hash(b"out" + struct.pack("<I", b) + row_root(np.asarray(e, dtype=np.float32)))
                        for b, e in enumerate(embeddings)])


SENTINEL = 0xFFFFFFFF  # detgen.head.SENTINEL: a generation input row's generated positions


def head_output_root(out: np.ndarray) -> bytes:
    """detgen.head.head_output_root: a head stage's samples [token, score bits] as stage-output leaves."""
    return merkle_root([_leaf_hash(b"sout" + struct.pack("<I", t) + row_root(np.ascontiguousarray(r, dtype=np.int32)))
                        for t, r in enumerate(np.asarray(out).astype(np.int32))])


def gen_output_root(head_roots: list[bytes]) -> bytes:
    """detgen.head.gen_output_root: a generation job's output root over its head output roots."""
    return merkle_root([_leaf_hash(b"gout" + struct.pack("<I", b) + r) for b, r in enumerate(head_roots)])

