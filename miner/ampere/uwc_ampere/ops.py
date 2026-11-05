"""Python launchers for kernels.cu, on torch tensors and torch's current CUDA stream."""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field

import cupy as cp
import numpy as np
import torch

from . import reference as ref

_HERE = os.path.dirname(os.path.abspath(__file__))
BM, BN, BK = 128, 256, 64
LMAX = 4
WIN_WORDS = 10
CHUNKS_PER_CTA = 64
FLAG_HASH, FLAG_DENOISE = 1, 2
OUT_NONE, OUT_I32, OUT_BF16 = 0, 1, 2


# GEMM configurations: (BN, STAGES, MINB). The hash-tile pattern is the same for all.
CONFIGS = {"256x3": (256, 3, 1), "256x4": (256, 4, 1), "128x3": (128, 3, 2), "128x4": (128, 4, 2)}
DEFAULT_CFG = os.environ.get("UWC_GEMM_CFG", "128x3")


@functools.lru_cache(maxsize=8)
def _module(cfg: str = DEFAULT_CFG) -> cp.RawModule:
    bn, stages, minb = CONFIGS[cfg]
    with open(os.path.join(_HERE, "kernels.cu")) as f:
        src = f.read()
    mod = cp.RawModule(code=src, options=("-arch=sm_86", "-std=c++17", f"-DSTAGES={stages}", f"-DBN_={bn}",
                                          f"-DMINB={minb}"))
    k = mod.get_function("noisy_gemm")
    k.max_dynamic_shared_size_bytes = stages * (BM + bn) * BK
    return mod


def _fn(name: str, cfg: str = DEFAULT_CFG):
    return _module(cfg).get_function(name)


def _p(t: torch.Tensor | None) -> np.uint64:
    return np.uint64(0 if t is None else t.data_ptr())


_STREAMS: dict[int, cp.cuda.Stream] = {}


def _stream() -> cp.cuda.Stream:
    ts = torch.cuda.current_stream()
    s = _STREAMS.get(ts.cuda_stream)
    if s is None:
        s = _STREAMS[ts.cuda_stream] = cp.cuda.Stream.from_external(ts)
    return s


def words(b: bytes, device="cuda") -> torch.Tensor:
    return torch.from_numpy(np.frombuffer(b, dtype="<u4").view(np.int32).copy()).to(device)


def words_to_bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().numpy().view(np.uint32).astype("<u4").tobytes()


LABEL_A = words(ref.SEED_LABEL_A, "cpu")
LABEL_B = words(ref.SEED_LABEL_B, "cpu")
SALT_A = words(ref.SEED_SALT_A, "cpu")
SALT_B = words(ref.SEED_SALT_B, "cpu")
_CONSTS: dict = {}


def consts(device) -> dict:
    """Device copies of the label/salt constants (made once; forward() must not copy H2D)."""
    key = str(torch.device(device))
    c = _CONSTS.get(key)
    if c is None:
        c = _CONSTS[key] = {n: t.to(device) for n, t in
                            dict(LABEL_A=LABEL_A, LABEL_B=LABEL_B, SALT_A=SALT_A, SALT_B=SALT_B).items()}
    return c


# ------------------------------------------------------------------ BLAKE3
def b3_root_and_seed(data: torch.Tensor, key: torch.Tensor, *, prefix=None, salt=None, dim: int = 0,
                     root_out=None, seed_out=None, scratch=None):
    """Keyed BLAKE3 of `data` (uint8/int8, length a multiple of 1024) -> root, optional salted seed."""
    nbytes = data.numel()
    assert nbytes % 1024 == 0 and nbytes > 0
    nchunks = nbytes // 1024
    ncta = (nchunks + CHUNKS_PER_CTA - 1) // CHUNKS_PER_CTA
    assert ncta <= 1024, "input larger than 64 MiB"
    if scratch is None or scratch.numel() < ncta * 8:
        scratch = torch.empty(ncta * 8, dtype=torch.int32, device=data.device)
    st = _stream()
    _fn("b3_tree_stage1")((ncta,), (CHUNKS_PER_CTA,), (_p(data), np.uint64(nchunks), _p(key), _p(scratch)), stream=st)
    _fn("b3_tree_stage2")((1,), (512,), (_p(scratch), np.int32(ncta), _p(key), _p(prefix), _p(salt), np.uint32(dim),
                                         _p(root_out), _p(seed_out)), stream=st)
    return scratch


# ------------------------------------------------------------------ noise
def noise_uniform(out: torch.Tensor, label: torch.Tensor, key: torch.Tensor, zero_me=None, zero_n: int = 0):
    nblocks = out.numel() // 32
    _fn("noise_uniform")(((nblocks + 255) // 256,), (256,),
                         (_p(out), np.uint32(nblocks), _p(label), _p(key), _p(zero_me), np.int32(zero_n)), stream=_stream())


def noise_perm(out: torch.Tensor, k: int, r: int, label: torch.Tensor, key: torch.Tensor):
    nh = (k + 7) // 8
    _fn("noise_perm")(((nh + 127) // 128,), (128,), (_p(out), np.uint32(k), np.uint32(r), _p(label), _p(key)),
                      stream=_stream())


def noise_rows(src, rows, k, r, *, E=None, Pn=None, dst=None, Pp=None, corr=None, corr_off=-1, ecopy_off=-1,
               maxabs=None):
    assert rows % 8 == 0 and k % 4 == 0
    smem = 8 * r + 8 * r * 4
    _fn("noise_rows")((rows // 8,), (256,),
                      (_p(src), np.int32(rows), np.int32(k), np.int32(r), _p(E), _p(Pn), _p(dst), _p(Pp), _p(corr),
                       np.int32(corr_off), np.int32(ecopy_off), _p(maxabs)), stream=_stream(), shared_mem=smem)


# ------------------------------------------------------------------ GEMM
def gemm(A, B, *, r=128, flags=0, corrA=None, corrB=None, maxabs=None, key=None, target=None, wins=None,
         max_wins=0, a_scale=None, b_scale=None, out=None, out_mode=OUT_NONE, cfg=None):
    cfg = cfg or DEFAULT_CFG
    bn, stages, _ = CONFIGS[cfg]
    m, k = A.shape
    n = B.shape[0]
    assert m % BM == 0 and n % bn == 0 and k % BK == 0 and r % BK == 0 and k % r == 0, (m, n, k, r)
    assert A.dtype == torch.int8 and B.dtype == torch.int8 and A.is_contiguous() and B.is_contiguous()
    _fn("noisy_gemm", cfg)(((n // bn) * (m // BM),), (bn,),
                              (_p(A), _p(B), np.int32(m), np.int32(n), np.int32(k), np.int32(r), _p(corrA), _p(corrB),
                               _p(maxabs), _p(key), _p(target), _p(wins), np.int32(max_wins), _p(a_scale), _p(b_scale),
                               _p(out), np.int32(out_mode), np.int32(flags)),
                              stream=_stream(), shared_mem=stages * (BM + bn) * BK)


# ------------------------------------------------------------------ one mined linear layer
@dataclass
class MinedWeight:
    """A weight B^T (n x k int8, values in [-64, 64]) prepared for mining under one block template.

    setup_block() runs once per (block header, k); forward() runs per call and needs no host sync.
    """
    Bt: torch.Tensor            # n x k int8
    b_scale: torch.Tensor       # n float32
    r: int = 128
    max_wins: int = 64
    # per block
    job_key: torch.Tensor | None = None
    b_seed: torch.Tensor | None = None
    target: torch.Tensor | None = None
    BpEBt: torch.Tensor | None = None
    PB: torch.Tensor | None = None
    EBRt: torch.Tensor | None = None
    corrB: torch.Tensor | None = None
    _bufs: dict = field(default_factory=dict)

    @property
    def n(self):
        return self.Bt.shape[0]

    @property
    def k(self):
        return self.Bt.shape[1]

    def setup_block(self, job_key: bytes, target_words: np.ndarray):
        dev = self.Bt.device
        n, k, r = self.n, self.k, self.r
        self.job_key = words(job_key, dev)
        self.target = torch.from_numpy(target_words.astype(np.uint32).view(np.int32).copy()).to(dev)
        self.b_seed = torch.empty(8, dtype=torch.int32, device=dev)
        c = consts(dev)
        b3_root_and_seed(self.Bt.view(-1), self.job_key, prefix=self.job_key, salt=c["SALT_B"], dim=n,
                         seed_out=self.b_seed)
        self.EBRt = torch.empty((n, r), dtype=torch.int8, device=dev)
        noise_uniform(self.EBRt, c["LABEL_B"], self.b_seed)
        self.PB = torch.empty(k, dtype=torch.int32, device=dev)
        noise_perm(self.PB, k, r, c["LABEL_B"], self.b_seed)
        self.BpEBt = torch.empty_like(self.Bt)
        self.corrB = torch.zeros((LMAX, n, 2 * r), dtype=torch.int8, device=dev)
        noise_rows(self.Bt, n, k, r, E=self.EBRt, Pn=self.PB, dst=self.BpEBt, corr=self.corrB, ecopy_off=0)
        self._bufs.clear()

    def _buf(self, m: int):
        b = self._bufs.get(m)
        if b is None:
            dev, k, r, n = self.Bt.device, self.k, self.r, self.n
            b = dict(
                a_seed=torch.empty(8, dtype=torch.int32, device=dev),
                EAL=torch.empty((m, r), dtype=torch.int8, device=dev),
                PA=torch.empty(k, dtype=torch.int32, device=dev),
                ApEA=torch.empty((m, k), dtype=torch.int8, device=dev),
                corrA=torch.zeros((LMAX, m, 2 * r), dtype=torch.int8, device=dev),
                maxabs=torch.zeros(2, dtype=torch.int32, device=dev),
                wins=torch.zeros(1 + self.max_wins * WIN_WORDS, dtype=torch.int32, device=dev),
                scratch=torch.empty(8 * 1024, dtype=torch.int32, device=dev),
            )
            self._bufs[m] = b
        return b

    def forward(self, A: torch.Tensor, a_scale: torch.Tensor, out: torch.Tensor | None = None,
                out_mode: int = OUT_BF16, hash_tiles: bool = True) -> torch.Tensor:
        """A: m x k int8 (values in [-64, 64]). Returns A @ Bt^T scaled (bf16) or exact int32."""
        m, k = A.shape
        n, r = self.n, self.r
        dev = A.device
        b = self._buf(m)
        c = consts(dev)
        if out is None:
            out = torch.empty((m, n), dtype=torch.bfloat16 if out_mode == OUT_BF16 else torch.int32, device=dev)
        # 1. commit to A, derive a_seed = H(b_seed || bind_a(root_A, m))
        b3_root_and_seed(A.view(-1), self.job_key, prefix=self.b_seed, salt=c["SALT_A"], dim=m,
                         seed_out=b["a_seed"], scratch=b["scratch"])
        # 2. A-side noise (also zeroes maxabs and the win counter)
        noise_uniform(b["EAL"], c["LABEL_A"], b["a_seed"], zero_me=b["maxabs"], zero_n=2)
        b["wins"][:1].zero_()
        noise_perm(b["PA"], k, r, c["LABEL_A"], b["a_seed"])
        # 3. ApEA = A + EAL*P_A ; X = A * P_B^T (limbs) ; EAL copy
        noise_rows(A, m, k, r, E=b["EAL"], Pn=b["PA"], dst=b["ApEA"], Pp=self.PB, corr=b["corrA"], corr_off=0,
                   ecopy_off=r, maxabs=b["maxabs"][0:1])
        # 4. Y = BpEBt * P_A^T (limbs)
        noise_rows(self.BpEBt, n, k, r, Pp=b["PA"], corr=self.corrB, corr_off=r, maxabs=b["maxabs"][1:2])
        # 5. noisy GEMM: hash every tile, denoise exactly, scale
        gemm(b["ApEA"], self.BpEBt, r=r, flags=(FLAG_HASH if hash_tiles else 0) | FLAG_DENOISE,
             corrA=b["corrA"], corrB=self.corrB, maxabs=b["maxabs"], key=b["a_seed"], target=self.target,
             wins=b["wins"], max_wins=self.max_wins, a_scale=a_scale, b_scale=self.b_scale, out=out, out_mode=out_mode)
        return out

    def winners(self, m: int) -> list[tuple[int, int, bytes]]:
        """(row_offset, col_offset, hash) for winning tiles of the last forward with this m (syncs)."""
        w = self._buf(m)["wins"].cpu().numpy().view(np.uint32)
        cnt = min(int(w[0]), self.max_wins)
        recs = w[1:1 + cnt * WIN_WORDS].reshape(cnt, WIN_WORDS)
        return [(int(x[0]), int(x[1]), x[2:].astype("<u4").tobytes()) for x in recs]


class GraphedForward:
    """MinedWeight.forward captured as one CUDA graph for a fixed m (static input/output buffers)."""

    def __init__(self, w: MinedWeight, m: int, out_mode: int = OUT_BF16):
        dev = w.Bt.device
        self.w, self.m = w, m
        self.A = torch.zeros((m, w.k), dtype=torch.int8, device=dev)
        self.a_scale = torch.ones(m, dtype=torch.float32, device=dev)
        self.out = torch.empty((m, w.n), dtype=torch.bfloat16 if out_mode == OUT_BF16 else torch.int32, device=dev)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                w.forward(self.A, self.a_scale, out=self.out, out_mode=out_mode)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            w.forward(self.A, self.a_scale, out=self.out, out_mode=out_mode)

    def __call__(self):
        self.graph.replay()
        return self.out
