"""Each CUDA kernel against the Python reference (bit-exact)."""
import numpy as np
import pytest
import torch
from blake3 import blake3

from uwc_ampere import ops
from uwc_ampere import reference as ref

DEV = "cuda"


@pytest.mark.parametrize("nchunks", [1, 2, 3, 63, 64, 65, 130, 1000, 4097, 65536, 65537, 300_001])
def test_tree_hash_matches_blake3(nchunks):
    rng = np.random.default_rng(nchunks)
    data = rng.integers(0, 256, size=nchunks * 1024, dtype=np.uint8)
    key = rng.bytes(32)
    root = torch.zeros(8, dtype=torch.int32, device=DEV)
    ops.b3_root_and_seed(torch.from_numpy(data).to(DEV), ops.words(key), root_out=root)
    assert ops.words_to_bytes(root) == blake3(data.tobytes(), key=key).digest()


def test_seed_derivation_matches_reference():
    rng = np.random.default_rng(1)
    m, n, k = 256, 512, 2048
    A = rng.integers(-64, 65, size=(m, k), dtype=np.int8)
    Bt = rng.integers(-64, 65, size=(n, k), dtype=np.int8)
    job_key = rng.bytes(32)
    a_seed, b_seed, _, _ = ref.commitment(job_key, A, Bt, salted=True)
    jk = ops.words(job_key)
    bs = torch.zeros(8, dtype=torch.int32, device=DEV)
    ops.b3_root_and_seed(torch.from_numpy(Bt).to(DEV).view(-1), jk, prefix=jk, salt=ops.SALT_B.to(DEV), dim=n, seed_out=bs)
    assert ops.words_to_bytes(bs) == b_seed
    as_ = torch.zeros(8, dtype=torch.int32, device=DEV)
    ops.b3_root_and_seed(torch.from_numpy(A).to(DEV).view(-1), jk, prefix=bs, salt=ops.SALT_A.to(DEV), dim=m, seed_out=as_)
    assert ops.words_to_bytes(as_) == a_seed


def test_noise_generation_matches_reference():
    rng = np.random.default_rng(2)
    key = rng.bytes(32)
    rows, r, k = 256, 128, 3072
    E = torch.empty((rows, r), dtype=torch.int8, device=DEV)
    ops.noise_uniform(E, ops.LABEL_A.to(DEV), ops.words(key))
    assert np.array_equal(E.cpu().numpy(), ref.uniform_matrix(ref.SEED_LABEL_A, key, rows, r))
    P = torch.empty(k, dtype=torch.int32, device=DEV)
    ops.noise_perm(P, k, r, ops.LABEL_B.to(DEV), ops.words(key))
    p = P.cpu().numpy().view(np.uint32)
    want = ref.perm_matrix(ref.SEED_LABEL_B, key, k, r)
    assert np.array_equal(p & 0xFFFF, want[:, 0]) and np.array_equal(p >> 16, want[:, 1])


def _limbs_to_int(corr, off, r):
    c = corr[:, :, off:off + r].astype(np.int64)
    return sum(c[i] * (128 ** i) for i in range(c.shape[0]))


def test_noise_rows_noising_and_projection():
    rng = np.random.default_rng(3)
    rows, k, r = 128, 2048, 128
    A = rng.integers(-64, 65, size=(rows, k), dtype=np.int8)
    E = rng.integers(-32, 32, size=(rows, r), dtype=np.int8)
    Pn = ref.perm_matrix(ref.SEED_LABEL_A, b"\x01" * 32, k, r)
    Pp = ref.perm_matrix(ref.SEED_LABEL_B, b"\x02" * 32, k, r)
    pack = lambda P: torch.from_numpy((P[:, 0] | (P[:, 1] << 16)).astype(np.uint32).view(np.int32)).to(DEV)
    dst = torch.empty((rows, k), dtype=torch.int8, device=DEV)
    corr = torch.zeros((ops.limbs_for(k), rows, 2 * r), dtype=torch.int8, device=DEV)
    mx = torch.zeros(1, dtype=torch.int32, device=DEV)
    ops.noise_rows(torch.from_numpy(A).to(DEV), rows, k, r, E=torch.from_numpy(E).to(DEV), Pn=pack(Pn), dst=dst,
                   Pp=pack(Pp), corr=corr, corr_off=0, ecopy_off=r, maxabs=mx)
    noised = A.astype(np.int16) + E[:, Pn[:, 0]] - E[:, Pn[:, 1]]
    assert np.array_equal(dst.cpu().numpy(), noised.astype(np.int8))
    proj = np.zeros((rows, r), dtype=np.int64)
    for l in range(k):
        proj[:, Pp[l, 0]] += A[:, l]
        proj[:, Pp[l, 1]] -= A[:, l]
    c = corr.cpu().numpy()
    assert np.array_equal(_limbs_to_int(c, 0, r), proj)
    assert int(mx.item()) == int(np.abs(proj).max())
    for li in range(ops.limbs_for(k)):
        assert np.array_equal(c[li, :, r:], E)


@pytest.mark.parametrize("m,n,k", [(128, 256, 1024), (256, 512, 2048), (384, 768, 4096)])
def test_plain_gemm_matches_cublas(m, n, k):
    rng = np.random.default_rng(m + n + k)
    A = torch.from_numpy(rng.integers(-127, 128, size=(m, k), dtype=np.int8)).to(DEV)
    B = torch.from_numpy(rng.integers(-127, 128, size=(n, k), dtype=np.int8)).to(DEV)
    out = torch.empty((m, n), dtype=torch.int32, device=DEV)
    ops.gemm(A, B, out=out, out_mode=ops.OUT_I32)
    assert torch.equal(out, torch._int_mm(A, B.t()))


def test_mined_forward_exact_and_winners_match_reference():
    """Full per-call path: output == A @ Bt^T exactly; the winning tiles are exactly the reference's."""
    from uwc_ampere import pearl_glue as pg
    rng = np.random.default_rng(7)
    m, n, k, r = 256, 512, 2048, 128
    A = rng.integers(-64, 65, size=(m, k), dtype=np.int8)
    Bt = rng.integers(-64, 65, size=(n, k), dtype=np.int8)
    nbits = 0x1E00FFFF
    hdr = pg.header(nbits, 7)
    cfg = pg.mining_config(k, r)
    key = pg.job_key(hdr, cfg)
    bound = pg.jackpot_bound(nbits, cfg)

    w = ops.MinedWeight(Bt=torch.from_numpy(Bt).to(DEV), b_scale=torch.ones(n, device=DEV), r=r, max_wins=4096)
    w.setup_block(key, pg.bound_words(bound))
    out = w.forward(torch.from_numpy(A).to(DEV), torch.ones(m, device=DEV), out_mode=ops.OUT_I32)
    want = A.astype(np.int64) @ Bt.astype(np.int64).T
    assert np.array_equal(out.cpu().numpy().astype(np.int64), want)

    a_seed, b_seed, _, _ = ref.commitment(key, A, Bt, salted=True)
    nz = ref.noise(k, r, a_seed, b_seed, m, n)
    ApEA = (A.astype(np.int16) + nz["noise_a"]).astype(np.int8)
    BpEBt = (Bt.astype(np.int16) + nz["noise_bt"]).astype(np.int8)
    msgs = ref.jackpot_all_tiles(torch.from_numpy(ApEA).to(DEV), torch.from_numpy(BpEBt).to(DEV), r, None, None,
                                 64, 64).cpu().numpy()
    expect = set()
    for (rb, g, cb, t) in np.ndindex(msgs.shape[:4]):
        h = ref.jackpot_hash(list(msgs[rb, g, cb, t]), a_seed)
        if int.from_bytes(h, "little") <= bound:
            expect.add((rb * 64 + g, cb * 64 + 2 * t, h))
    got = set(w.winners(m))
    assert expect and got == expect


def test_graphed_forward_matches_eager():
    rng = np.random.default_rng(5)
    m, n, k = 256, 512, 2048
    A = torch.from_numpy(rng.integers(-64, 65, size=(m, k), dtype=np.int8)).to(DEV)
    Bt = torch.from_numpy(rng.integers(-64, 65, size=(n, k), dtype=np.int8)).to(DEV)
    w = ops.MinedWeight(Bt=Bt, b_scale=torch.ones(n, device=DEV))
    w.setup_block(rng.bytes(32), np.zeros(8, np.uint32))
    gf = ops.GraphedForward(w, m, out_mode=ops.OUT_I32)
    gf.A.copy_(A)
    got = gf().clone()
    assert torch.equal(got, torch._int_mm(A, Bt.t()))
