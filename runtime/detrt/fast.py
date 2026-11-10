"""Batched stage traces: the same rows as stage.run_stage, computed for many sequences at once.

A provider's job is many sequences through the same layers. The reference run_stage handles one
128-token sequence at a time and copies each of its ~76,000 trace rows to the CPU separately,
so its int8 GEMMs are tiny and most of its time is spent in transfers (step 7 profile: 61%
transfers, 28% fp ops, 11% hashing, ~0% GEMMs). Here:

- every row-wise op and every int8 GEMM runs on all sequences' tokens at once (B*T rows): each
  output row depends only on its own input row, and the integer GEMM is exact, so each row is
  bit-identical to the reference;
- attention runs per sequence, batched over sequences with the reference's exact ops;
- each op's output is copied to the CPU once, then cut into rows;
- rows are hashed by forked worker processes.

The leaf order, row contents and roots are those of stage.run_stage (checked by tests/test_fast.py).
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np
import torch

from . import core
from .core import F32, I64, c32, fadd, fdiv, fmul, fsub
from .qwen3 import KEY_CHUNK, MAX_KEYS, P_LEVELS, DetQwen3
from .stage import (EMBED, GU, KN, KQ, KR, N1, N2, OP, P, QA1, QA2, QA3, QA4, QKV, QN, QQ, QR, R1, R2, S, SM,
                    VQ, VS, VT, DN, O, Leaf, StageTrace, leaf_hash, merkle_root, model_cfg, program, row_root)


def _np(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu()
    return t.numpy().astype(np.int32) if t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64) else t.to(F32).numpy()


def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a).view(np.int32)


_ROWS: list | None = None


def _hash_range(span):
    lo, hi = span
    return [leaf_hash(i, row_root(_ROWS[i])) for i in range(lo, hi)]


def finalize_parallel(tr: StageTrace, workers: int) -> StageTrace:
    """StageTrace.finalize with the row hashing spread over forked processes."""
    global _ROWS
    n = len(tr.rows)
    if workers <= 1 or n < 4096:
        return tr.finalize()
    _ROWS = tr.rows
    step = -(-n // (workers * 4))
    spans = [(i, min(n, i + step)) for i in range(0, n, step)]
    with mp.get_context("fork").Pool(workers) as pool:
        parts = pool.map(_hash_range, spans)
    _ROWS = None
    tr.leaf_hashes = [h for p in parts for h in p]
    tr.root = merkle_root(tr.leaf_hashes)
    return tr


@torch.no_grad()
def run_stages(m: DetQwen3, tokens: list[torch.Tensor] | None, x_in: list[torch.Tensor] | None, l0: int, l1: int,
               workers: int | None = None, keep_rows: bool = True):
    """Stage traces for B sequences of equal length T: tokens (B x (T,)) when l0 == 0, else
    x_in (B x (T, D) float32, each the previous stage's output). Same rows as run_stage.
    With keep_rows=False, only the forward pass runs and the B stage outputs ((T, D) float32,
    the rows the stage is committed by) are returned: what providers and verifiers compute."""
    keep = keep_rows
    cfg = model_cfg(m)
    H, Hkv, hd, D = cfg["H"], cfg["Hkv"], cfg["hd"], cfg["D"]
    dev = m.device
    B = len(tokens) if l0 == 0 else len(x_in)
    T = tokens[0].shape[0] if l0 == 0 else x_in[0].shape[0]
    N = B * T
    rows = [dict() for _ in range(B)]

    def put_tok(op, l, arr):  # arr: (N, ...) one row per token
        for b in range(B):
            rb = rows[b]
            for t in range(T):
                rb[Leaf(op, l, t, -1)] = arr[b * T + t]

    def put_head(op, l, arr, nh):  # arr: (N, nh, w)
        for b in range(B):
            rb = rows[b]
            for t in range(T):
                a = arr[b * T + t]
                for hh in range(nh):
                    rb[Leaf(op, l, t, hh)] = a[hh]

    if l0 == 0:
        x = m.embed[torch.cat([t.to(dev) for t in tokens])]
        if keep:
            put_tok(EMBED, 0, _np(x))
    else:
        x = torch.cat([xi.to(dev, F32) for xi in x_in])
    pos = torch.arange(T, device=dev).repeat(B)
    rep = H // Hkv
    for l in range(l0, l1):
        L = m.layers[l]
        h1 = core.det_rmsnorm(x, L["in_norm"], m.eps)
        q1, s1 = core.quant_act(h1, L["qkv"].bs)
        qkv = fmul(fmul(L["qkv"].int_gemm(q1).to(F32), s1[:, None]), L["qkv"].sw[None, :])
        q, k, v = torch.split(qkv, [H * hd, Hkv * hd, Hkv * hd], dim=-1)
        qn = m.qkn(q.reshape(N, H, hd), L["q_norm"])
        kn = m.qkn(k.reshape(N, Hkv, hd), L["k_norm"])
        cos, sin = m.cos[pos][:, None, :], m.sin[pos][:, None, :]
        half = hd // 2
        rope = lambda z: fadd(fmul(z, cos), fmul(torch.cat([-z[..., half:], z[..., :half]], -1), sin))
        qr, kr = rope(qn), rope(kn)
        kq, ks = core.quant_sym(kr, 127)
        vq, vs = core.quant_sym(v.reshape(N, Hkv, hd), 127)
        qq, qs = core.quant_sym(qr, 127)

        if keep:
            put_tok(N1, l, _np(h1))
            put_tok(QA1, l, np.concatenate([_np(q1), _bits(_np(s1).reshape(N, 1))], axis=1))
            put_tok(QKV, l, _np(qkv))
            put_head(QN, l, _np(qn), H)
            put_head(QR, l, _np(qr), H)
            put_head(QQ, l, np.concatenate([_np(qq.to(torch.int32)), _bits(_np(qs).reshape(N, H, 1))], axis=2), H)
            put_head(KN, l, _np(kn), Hkv)
            put_head(KR, l, _np(kr), Hkv)
            put_head(KQ, l, np.concatenate([_np(kq.to(torch.int32)), _bits(_np(ks).reshape(N, Hkv, 1))], axis=2), Hkv)
            put_head(VQ, l, np.concatenate([_np(vq.to(torch.int32)), _bits(_np(vs).reshape(N, Hkv, 1))], axis=2), Hkv)
            vq_all, vs_all = _np(vq.to(torch.int32)), _np(vs)
        for b in range(B if keep else 0):
            vq_i = vq_all[b * T:(b + 1) * T]
            for hh in range(Hkv):
                for d in range(hd):
                    rows[b][Leaf(VT, l, d, hh)] = np.ascontiguousarray(vq_i[:, hh, d])
                rows[b][Leaf(VS, l, -1, hh)] = np.ascontiguousarray(vs_all[b * T:(b + 1) * T, hh, 0])

        # attention per sequence, batched over sequences: (B, H, T, hd)
        def heads(z, n):
            return z.reshape(B, T, n, -1).permute(0, 2, 1, 3)
        K = heads(kq, Hkv).repeat_interleave(rep, 1)
        SK = heads(ks, Hkv).repeat_interleave(rep, 1)
        V = heads(vq, Hkv).repeat_interleave(rep, 1)
        SV = heads(vs, Hkv).repeat_interleave(rep, 1)
        Q, QS = heads(qq, H), heads(qs, H)
        sc_int = core.exact_matmul_small(Q, K.transpose(-1, -2), 127 * 127 * hd)
        sc = fmul(fmul(fmul(sc_int, QS), SK.transpose(-1, -2)), c32(m.inv_sqrt_hd, sc_int))
        mask = torch.arange(T, device=dev)[None, :] <= torch.arange(T, device=dev)[:, None]
        sc = torch.where(mask[None, None], sc, torch.full_like(sc, float("-inf")))
        mx = sc.amax(-1, keepdim=True)
        e = core.det_exp(fsub(sc, mx))
        den = core.exact_rowsum(e, MAX_KEYS)
        Pr = fdiv(e, den.unsqueeze(-1))
        Pv = fmul(Pr, SV.transpose(-1, -2))
        pmax = Pv.amax(-1, keepdim=True)
        sp = fdiv(torch.maximum(pmax, c32(1e-30, pmax)), c32(float(P_LEVELS), pmax))
        pi = torch.round(fdiv(Pv, sp)).clamp_(0, P_LEVELS).to(torch.int32)
        hi, lo = (pi >> 7).to(F32), (pi & 127).to(F32)
        acc = torch.zeros(B, H, T, hd, dtype=I64, device=dev)
        Vf = V.to(F32)
        for c0 in range(0, T, KEY_CHUNK):
            bnd = 127 * 127 * KEY_CHUNK
            acc += (core.exact_matmul_small(hi[..., c0:c0 + KEY_CHUNK], Vf[..., c0:c0 + KEY_CHUNK, :], bnd).to(I64) << 7) \
                + core.exact_matmul_small(lo[..., c0:c0 + KEY_CHUNK], Vf[..., c0:c0 + KEY_CHUNK, :], bnd).to(I64)
        o = fmul(acc.to(F32), sp)                                          # (B, H, T, hd)
        if keep:
            sc_n, pi_n, sp_n, o_n = _np(sc), _np(pi), _bits(_np(sp)), _np(o)
        for b in range(B if keep else 0):
            rb = rows[b]
            for t in range(T):
                for hh in range(H):
                    rb[Leaf(S, l, t, hh)] = sc_n[b, hh, t, :t + 1].copy()
                    rb[Leaf(P, l, t, hh)] = np.concatenate([pi_n[b, hh, t, :t + 1], sp_n[b, hh, t]])
                    rb[Leaf(O, l, t, hh)] = o_n[b, hh, t]
        oc = o.permute(0, 2, 1, 3).reshape(N, H * hd)
        q2, s2 = core.quant_act(oc, L["o"].bs)
        op = fmul(fmul(L["o"].int_gemm(q2).to(F32), s2[:, None]), L["o"].sw[None, :])
        r1 = fadd(x, op)
        h2 = core.det_rmsnorm(r1, L["post_norm"], m.eps)
        q3, s3 = core.quant_act(h2, L["gu"].bs)
        gu = fmul(fmul(L["gu"].int_gemm(q3).to(F32), s3[:, None]), L["gu"].sw[None, :])
        I_ = gu.shape[1] // 2
        sm = fmul(core.det_silu(gu[:, :I_]), gu[:, I_:])
        q4, s4 = core.quant_act(sm, L["down"].bs)
        dn = fmul(fmul(L["down"].int_gemm(q4).to(F32), s4[:, None]), L["down"].sw[None, :])
        r2 = fadd(r1, dn)
        if keep:
            put_tok(QA2, l, np.concatenate([_np(q2), _bits(_np(s2).reshape(N, 1))], axis=1))
            put_tok(OP, l, _np(op))
            put_tok(R1, l, _np(r1))
            put_tok(N2, l, _np(h2))
            put_tok(QA3, l, np.concatenate([_np(q3), _bits(_np(s3).reshape(N, 1))], axis=1))
            put_tok(GU, l, _np(gu))
            put_tok(SM, l, _np(sm))
            put_tok(QA4, l, np.concatenate([_np(q4), _bits(_np(s4).reshape(N, 1))], axis=1))
            put_tok(DN, l, _np(dn))
            put_tok(R2, l, _np(r2))
        x = r2
    out = _np(x)
    if not keep:
        return [out[b * T:(b + 1) * T].copy() for b in range(B)]
    leaves = program(cfg, T, l0, l1)
    workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 2)
    traces = []
    for b in range(B):
        tr = StageTrace(leaves=leaves, rows=[rows[b][lf] for lf in leaves], output=out[b * T:(b + 1) * T].copy())
        traces.append(finalize_parallel(tr, workers))
    return traces
