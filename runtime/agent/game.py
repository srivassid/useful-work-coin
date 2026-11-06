"""Dispute-game moves (node/jobstate dispute.go), played from a party's own stage trace.

Phases: 1 reveal (provider opens the stage), 2 bisect the trace tree, 3 post the differing
leaf's row root, 4 bisect that row's block tree, 5 post the differing 1 KiB block, 6 one-step
proof (provider only). Each move commits to the party's own trace, so an honest party never
loses a round; the last differing element is then recomputed by every full node.
"""
from __future__ import annotations

from detrt.stage import StageTrace, _leaf_hash, _split, merkle_root, row_blocks, row_bytes, row_root

GREVEAL, GTREE, GLEAF, GBLOCKS, GBLOCK, GONESTEP = 1, 2, 3, 4, 5, 6


def move(job_id: str, g: dict, tr: StageTrace) -> dict | None:
    """This party's DISPUTE_MOVE for the current round of game g (from getjob), or None."""
    phase, lo, size = g["phase"], g["lo"], g["size"]
    op = {"kind": "DISPUTE_MOVE", "job": job_id}
    if phase == GTREE:
        k = _split(size)
        op["h1"] = merkle_root(tr.leaf_hashes[lo:lo + k]).hex()
        op["h2"] = merkle_root(tr.leaf_hashes[lo + k:lo + size]).hex()
    elif phase == GLEAF:
        op["h1"] = row_root(tr.rows[lo]).hex()
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
    """Trace rows for uwcctl onestep: u32 count, then per row u32 words and the words."""
    import struct
    import numpy as np
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(rows)))
        for r in rows:
            w = np.ascontiguousarray(r).view(np.uint32)
            f.write(struct.pack("<I", len(w)))
            f.write(w.astype("<u4").tobytes())
