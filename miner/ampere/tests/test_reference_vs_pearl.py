"""The Python reference must produce proofs that Pearl's own Rust verifier accepts."""
import numpy as np
import torch

from uwc_ampere import pearl_glue as pg
from uwc_ampere import reference as ref

EASY_NBITS = 0x1E00FFFF  # about 1 in 64 tiles wins at k=2048, tile 128


def _setup(m=256, n=256, k=2048, r=128, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.integers(-64, 65, size=(m, k), dtype=np.int8)
    Bt = rng.integers(-64, 65, size=(n, k), dtype=np.int8)
    hdr = pg.header(EASY_NBITS, seed)
    cfg = pg.mining_config(k, r)
    key = pg.job_key(hdr, cfg)
    a_seed, b_seed, _, _ = ref.commitment(key, A, Bt, salted=True)
    nz = ref.noise(k, r, a_seed, b_seed, m, n)
    ApEA = (A.astype(np.int16) + nz["noise_a"]).astype(np.int8)
    BpEBt = (Bt.astype(np.int16) + nz["noise_bt"]).astype(np.int8)
    return A, Bt, hdr, cfg, key, a_seed, ApEA, BpEBt


def test_gpu_reference_matches_numpy_tile():
    A, Bt, hdr, cfg, key, a_seed, ApEA, BpEBt = _setup()
    msgs = ref.jackpot_all_tiles(torch.from_numpy(ApEA).cuda(), torch.from_numpy(BpEBt).cuda(), 128,
                                 None, None, 64, 64).cpu().numpy()
    for (rb, g, cb, t) in [(0, 0, 0, 0), (1, 5, 2, 3), (3, 7, 3, 1)]:
        rows = [rb * 64 + g + o for o in pg.AMPERE_ROWS]
        cols = [cb * 64 + 2 * t + o for o in pg.AMPERE_COLS]
        want = ref.jackpot_msg_tile(ApEA[rows].astype(np.int64), BpEBt[cols].astype(np.int64), 128)
        assert list(msgs[rb, g, cb, t]) == want


def test_reference_proof_accepted_by_pearl():
    A, Bt, hdr, cfg, key, a_seed, ApEA, BpEBt = _setup()
    msgs = ref.jackpot_all_tiles(torch.from_numpy(ApEA).cuda(), torch.from_numpy(BpEBt).cuda(), 128,
                                 None, None, 64, 64).cpu().numpy()
    bound = pg.jackpot_bound(EASY_NBITS, cfg)
    winners, losers = [], []
    for idx in np.ndindex(msgs.shape[:4]):
        hv = int.from_bytes(ref.jackpot_hash(list(msgs[idx]), a_seed), "little")
        (winners if hv <= bound else losers).append(idx)
    assert winners, "no winning tile at easy difficulty"

    def proof_for(idx):
        rb, g, cb, t = idx
        rows = [rb * 64 + g + o for o in pg.AMPERE_ROWS]
        cols = [cb * 64 + 2 * t + o for o in pg.AMPERE_COLS]
        return pg.build_plain_proof(A, Bt, key, 128, rows, cols)

    ok, msg = pg.verify(hdr, proof_for(winners[0]))
    assert ok, msg
    ok, msg = pg.verify(hdr, proof_for(losers[0]))
    assert not ok, "a losing tile was accepted"
