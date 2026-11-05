"""GPU-mined winners must become proofs that Pearl's own Rust verifier accepts."""
import numpy as np
import torch

from uwc_ampere import ops
from uwc_ampere import pearl_glue as pg


def test_gpu_winners_verify_in_pearl():
    rng = np.random.default_rng(11)
    m, n, k, r = 512, 1024, 4096, 128
    nbits = 0x1E00FFFF
    A = rng.integers(-64, 65, size=(m, k), dtype=np.int8)
    Bt = rng.integers(-64, 65, size=(n, k), dtype=np.int8)
    hdr = pg.header(nbits, 11)
    cfg = pg.mining_config(k, r)
    key = pg.job_key(hdr, cfg)
    w = ops.MinedWeight(Bt=torch.from_numpy(Bt).cuda(), b_scale=torch.ones(n, device="cuda"), r=r)
    w.setup_block(key, pg.bound_words(pg.jackpot_bound(nbits, cfg)))
    w.forward(torch.from_numpy(A).cuda(), torch.ones(m, device="cuda"))
    wins = w.winners(m)
    assert wins
    for row_off, col_off, _ in wins[:3]:
        rows = [row_off + o for o in pg.AMPERE_ROWS]
        cols = [col_off + o for o in pg.AMPERE_COLS]
        ok, msg = pg.verify(hdr, pg.build_plain_proof(A, Bt, key, r, rows, cols))
        assert ok, msg
    # a tile next to a winner (shifted by one row offset) must be rejected unless it also won
    row_off, col_off, _ = wins[0]
    won = {(a, b) for a, b, _ in wins}
    alt = (row_off ^ 1, col_off)
    if alt not in won:
        rows = [alt[0] + o for o in pg.AMPERE_ROWS]
        cols = [alt[1] + o for o in pg.AMPERE_COLS]
        ok, _ = pg.verify(hdr, pg.build_plain_proof(A, Bt, key, r, rows, cols))
        assert not ok
