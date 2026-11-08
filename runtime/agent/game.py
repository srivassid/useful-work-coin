"""Dispute-game moves (node/jobstate dispute.go), played from a party's own stage trace.

Stages are committed by their output rows; a full trace is built only for a disputed stage and
committed in the top round as (body root, output root). Phases: 1 reveal (provider opens the
stage's output root), 7 top, 2 bisect the chosen tree (body or outputs), 3 the differing leaf's
row root, 4 bisect that row's block tree, 5 the differing 1 KiB block, 6 one-step proof
(provider only). Each move commits to the party's own trace, so an honest party never loses a
round; the last differing element is then recomputed by every full node.
"""
from __future__ import annotations

import struct

import numpy as np

from detrt.stage import R2, StageTrace, _leaf_hash, _split, leaf_hash, merkle_root, row_blocks, row_bytes, row_root

GREVEAL, GTREE, GLEAF, GBLOCKS, GBLOCK, GONESTEP, GTOP = 1, 2, 3, 4, 5, 6, 7
TREE_BODY, TREE_OUT = 0, 1


def split(tr: StageTrace):
    """(body program indices, output program indices by token, body leaves, output leaves) of a
    trace, as jobstate.TraceCommit; cached on the trace."""
    if getattr(tr, "_split", None) is None:
        from detgen.head import HS
        if any(lf.op == HS for lf in tr.leaves):  # a generation batch's head stage: the samples are its outputs
            is_out = lambda lf: lf.op == HS
        else:
            l1 = max(lf.layer for lf in tr.leaves) + 1
            is_out = lambda lf: lf.op == R2 and lf.layer == l1 - 1
        body_at = [i for i, lf in enumerate(tr.leaves) if not is_out(lf)]
        out_at = [i for _, i in sorted((lf.t, i) for i, lf in enumerate(tr.leaves) if is_out(lf))]
        body = [leaf_hash(bi, row_root(tr.rows[pi])) for bi, pi in enumerate(body_at)]
        out = [_leaf_hash(b"sout" + struct.pack("<I", t) + row_root(tr.rows[pi])) for t, pi in enumerate(out_at)]
        tr._split = (body_at, out_at, body, out)
    return tr._split


def move(job_id: str, g: dict, tr: StageTrace) -> dict | None:
    """This party's DISPUTE_MOVE for the current round of game g (from getjob), or None."""
    phase, lo, size = g["phase"], g["lo"], g["size"]
    body_at, out_at, body, out = split(tr)
    leaves, at = (out, out_at) if g.get("tree") == TREE_OUT else (body, body_at)
    op = {"kind": "DISPUTE_MOVE", "job": job_id}
    if phase == GTOP:
        op["h1"], op["h2"] = merkle_root(body).hex(), merkle_root(out).hex()
    elif phase == GTREE:
        k = _split(size)
        op["h1"] = merkle_root(leaves[lo:lo + k]).hex()
        op["h2"] = merkle_root(leaves[lo + k:lo + size]).hex()
    elif phase == GLEAF:
        op["h1"] = row_root(tr.rows[at[lo]]).hex()
    elif phase == GBLOCKS:
        bl = [_leaf_hash(b) for b in row_blocks(row_bytes(tr.rows[g["leaf"]]))]
        k = _split(size)
        op["h1"] = merkle_root(bl[lo:lo + k]).hex()
        op["h2"] = merkle_root(bl[lo + k:lo + size]).hex()
    elif phase == GBLOCK:
        op["data"] = row_blocks(row_bytes(tr.rows[g["leaf"]]))[g["block"]].hex()
    else:
        return None
    return op


def write_rows(path: str, rows) -> None:
    """Rows for uwcctl onestep: u32 count, then per row u32 words and the words."""
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(rows)))
        for r in rows:
            w = np.ascontiguousarray(r).view(np.uint32)
            f.write(struct.pack("<I", len(w)))
            f.write(w.astype("<u4").tobytes())
