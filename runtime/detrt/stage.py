"""Stage traces for verification (design doc step 4, spec/stage-program.md).

A stage is one sequence (positions 0..T-1) through layers [l0, l1) of the deterministic
Qwen3 runtime. It is written as a fixed list of row operations ("leaves"); each leaf's output
row is serialized as 4-byte little-endian words (float32 bits or int32), split into 1 KiB blocks,
and committed as  row_root = MerkleRoot(sha256(0x00 || block)),
                  leaf     = sha256(0x00 || "leaf" || u32 index || row_root),
                  trace_root = MerkleRoot(leaves)       (RFC 6962 trees, as node/jobstate).
The leaf order and every op's arithmetic are consensus: node/jobstate/referee recomputes any
single element bit-exactly from its inputs.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

import numpy as np
import torch

from . import core
from .core import F32, I64, c32, fadd, fdiv, fmul, fsub
from .qwen3 import KEY_CHUNK, MAX_KEYS, P_LEVELS, DetQwen3

BLOCK = 1024  # bytes per row block

# op codes (spec/stage-program.md); order inside a layer is the order below, op-major
EMBED, N1, QA1, QKV, QN, KN, QR, KR, KQ, VQ, QQ, VT, VS, S, P, O, QA2, OP, R1, N2, QA3, GU, SM, QA4, DN, R2 = range(26)
OP_NAMES = ["EMBED", "N1", "QA1", "QKV", "QN", "KN", "QR", "KR", "KQ", "VQ", "QQ", "VT", "VS", "S", "P", "O",
            "QA2", "OP", "R1", "N2", "QA3", "GU", "SM", "QA4", "DN", "R2"]


# ---------------------------------------------------------------- hashing (sha256, RFC 6962)
def _leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + data).digest()


def _node(l: bytes, r: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + l + r).digest()


def _split(n: int) -> int:
    k = 1
    while k << 1 < n:
        k <<= 1
    return k


def merkle_root(leaves: list[bytes]) -> bytes:
    if not leaves:
        return hashlib.sha256(b"").digest()
    if len(leaves) == 1:
        return leaves[0]
    k = _split(len(leaves))
    return _node(merkle_root(leaves[:k]), merkle_root(leaves[k:]))


def merkle_proof(leaves: list[bytes], i: int) -> list[bytes]:
    if len(leaves) <= 1:
        return []
    k = _split(len(leaves))
    if i < k:
        return merkle_proof(leaves[:k], i) + [merkle_root(leaves[k:])]
    return merkle_proof(leaves[k:], i - k) + [merkle_root(leaves[:k])]


def row_bytes(row: np.ndarray) -> bytes:
    assert row.dtype in (np.float32, np.int32) and row.ndim == 1
    return row.astype("<f4" if row.dtype == np.float32 else "<i4").tobytes()


def row_blocks(b: bytes) -> list[bytes]:
    return [b[i:i + BLOCK] for i in range(0, max(len(b), 1), BLOCK)]


def row_root(row: np.ndarray) -> bytes:
    return merkle_root([_leaf_hash(blk) for blk in row_blocks(row_bytes(row))])


def leaf_hash(index: int, rroot: bytes) -> bytes:
    return _leaf_hash(b"leaf" + struct.pack("<I", index) + rroot)


# ---------------------------------------------------------------- the program
@dataclass(frozen=True)
class Leaf:
    op: int
    layer: int
    t: int      # token (or channel d for VT, -1 for VS)
    h: int      # head (-1 if none)


def program(cfg: dict, T: int, l0: int, l1: int) -> list[Leaf]:
    """Canonical leaf list of a stage. Must equal jobstate/referee.Program."""
    H, Hkv, hd = cfg["H"], cfg["Hkv"], cfg["hd"]
    out: list[Leaf] = []
    if l0 == 0:
        out += [Leaf(EMBED, 0, t, -1) for t in range(T)]
    for l in range(l0, l1):
        per_t = lambda op: [Leaf(op, l, t, -1) for t in range(T)]
        per_th = lambda op, nh: [Leaf(op, l, t, h) for t in range(T) for h in range(nh)]
        out += per_t(N1) + per_t(QA1) + per_t(QKV)
        out += per_th(QN, H) + per_th(KN, Hkv) + per_th(QR, H) + per_th(KR, Hkv)
        out += per_th(KQ, Hkv) + per_th(VQ, Hkv) + per_th(QQ, H)
        out += [Leaf(VT, l, d, h) for h in range(Hkv) for d in range(hd)]
        out += [Leaf(VS, l, -1, h) for h in range(Hkv)]
        out += per_th(S, H) + per_th(P, H) + per_th(O, H)
        out += per_t(QA2) + per_t(OP) + per_t(R1) + per_t(N2) + per_t(QA3) + per_t(GU) + per_t(SM) + per_t(QA4)
        out += per_t(DN) + per_t(R2)
    return out


def model_cfg(m: DetQwen3) -> dict:
    return dict(D=m.D, H=m.H, Hkv=m.Hkv, hd=m.hd, I=m.layers[0]["gu"].n // 2, L=len(m.layers),
                V=m.embed.shape[0], eps=m.eps, bs=m.layers[0]["qkv"].bs, arch=m.arch)


@dataclass
class StageTrace:
    leaves: list[Leaf]
    rows: list[np.ndarray]           # output row of each leaf (float32 or int32)
    leaf_hashes: list[bytes] = field(default_factory=list)
    root: bytes = b""
    output: np.ndarray | None = None  # (T, D) float32: R2 rows of the last layer

    def finalize(self):
        self.leaf_hashes = [leaf_hash(i, row_root(r)) for i, r in enumerate(self.rows)]
        self.root = merkle_root(self.leaf_hashes)
        return self


def _np(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu()
    return t.numpy().astype(np.int32) if t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64) else t.to(F32).numpy()


@torch.no_grad()
def run_stage(m: DetQwen3, tokens: torch.Tensor | None, x_in: torch.Tensor | None, l0: int, l1: int) -> StageTrace:
    """tokens: (T,) ids when l0 == 0; else x_in: (T, D) float32 stage input (previous stage output)."""
    cfg = model_cfg(m)
    H, Hkv, hd, D = cfg["H"], cfg["Hkv"], cfg["hd"], cfg["D"]
    dev = m.device
    T = tokens.shape[0] if l0 == 0 else x_in.shape[0]
    rows: dict[Leaf, np.ndarray] = {}

    def put(op, l, t, h, row):
        rows[Leaf(op, l, t, h)] = row

    if l0 == 0:
        x = m.embed[tokens.to(dev)]
        for t in range(T):
            put(EMBED, 0, t, -1, _np(x[t]))
    else:
        x = x_in.to(dev, F32)
    pos = torch.arange(T, device=dev)
    rep = H // Hkv
    for l in range(l0, l1):
        L = m.layers[l]
        h1 = core.det_rmsnorm(x, L["in_norm"], m.eps)
        q1, s1 = core.quant_act(h1, L["qkv"].bs)
        qkv = fmul(fmul(L["qkv"].int_gemm(q1).to(F32), s1[:, None]), L["qkv"].sw[None, :])
        q, k, v = torch.split(qkv, [H * hd, Hkv * hd, Hkv * hd], dim=-1)
        qn = m.qkn(q.reshape(T, H, hd), L["q_norm"])
        kn = m.qkn(k.reshape(T, Hkv, hd), L["k_norm"])
        cos, sin = m.cos[pos][:, None, :], m.sin[pos][:, None, :]
        half = hd // 2
        rope = lambda z: fadd(fmul(z, cos), fmul(torch.cat([-z[..., half:], z[..., :half]], -1), sin))
        qr, kr = rope(qn), rope(kn)
        kq, ks = core.quant_sym(kr, 127)
        vq, vs = core.quant_sym(v.reshape(T, Hkv, hd), 127)
        qq, qs = core.quant_sym(qr, 127)
        for t in range(T):
            put(N1, l, t, -1, _np(h1[t]))
            put(QA1, l, t, -1, np.concatenate([_np(q1[t]), _np(s1[t:t + 1]).view(np.int32)]))
            put(QKV, l, t, -1, _np(qkv[t]))
            for hh in range(H):
                put(QN, l, t, hh, _np(qn[t, hh]))
                put(QR, l, t, hh, _np(qr[t, hh]))
                put(QQ, l, t, hh, np.concatenate([_np(qq[t, hh].to(torch.int32)), _np(qs[t, hh]).view(np.int32)]))
            for hh in range(Hkv):
                put(KN, l, t, hh, _np(kn[t, hh]))
                put(KR, l, t, hh, _np(kr[t, hh]))
                put(KQ, l, t, hh, np.concatenate([_np(kq[t, hh].to(torch.int32)), _np(ks[t, hh]).view(np.int32)]))
                put(VQ, l, t, hh, np.concatenate([_np(vq[t, hh].to(torch.int32)), _np(vs[t, hh]).view(np.int32)]))
        vq_i = _np(vq.to(torch.int32))  # (T, Hkv, hd) int32
        for hh in range(Hkv):
            for d in range(hd):
                put(VT, l, d, hh, np.ascontiguousarray(vq_i[:, hh, d]))
            put(VS, l, -1, hh, _np(vs[:, hh, 0]))
        # attention, batched exactly as DetQwen3._attention (masked full rows), rows cut to j <= t
        K = kq.permute(1, 0, 2).repeat_interleave(rep, 0)                 # (H, T, hd)
        SK = ks.permute(1, 0, 2).repeat_interleave(rep, 0)                # (H, T, 1)
        V = vq.permute(1, 0, 2).repeat_interleave(rep, 0)                 # (H, T, hd)
        SV = vs.permute(1, 0, 2).repeat_interleave(rep, 0)
        Q, QS = qq.permute(1, 0, 2), qs.permute(1, 0, 2)                  # (H, T, hd), (H, T, 1)
        sc_int = core.exact_matmul_small(Q, K.transpose(-1, -2), 127 * 127 * hd)
        sc = fmul(fmul(fmul(sc_int, QS), SK.transpose(-1, -2)), c32(m.inv_sqrt_hd, sc_int))
        mask = torch.arange(T, device=dev)[None, :] <= torch.arange(T, device=dev)[:, None]
        sc = torch.where(mask[None], sc, torch.full_like(sc, float("-inf")))
        mx = sc.amax(-1, keepdim=True)
        e = core.det_exp(fsub(sc, mx))
        den = core.exact_rowsum(e, MAX_KEYS)
        Pr = fdiv(e, den.unsqueeze(-1))
        Pv = fmul(Pr, SV.transpose(-1, -2))
        pmax = Pv.amax(-1, keepdim=True)
        sp = fdiv(torch.maximum(pmax, c32(1e-30, pmax)), c32(float(P_LEVELS), pmax))
        pi = torch.round(fdiv(Pv, sp)).clamp_(0, P_LEVELS).to(torch.int32)
        hi, lo = (pi >> 7).to(F32), (pi & 127).to(F32)
        acc = torch.zeros(H, T, hd, dtype=I64, device=dev)
        Vf = V.to(F32)
        for c0 in range(0, T, KEY_CHUNK):
            b = 127 * 127 * KEY_CHUNK
            acc += (core.exact_matmul_small(hi[..., c0:c0 + KEY_CHUNK], Vf[:, c0:c0 + KEY_CHUNK], b).to(I64) << 7) \
                + core.exact_matmul_small(lo[..., c0:c0 + KEY_CHUNK], Vf[:, c0:c0 + KEY_CHUNK], b).to(I64)
        o = fmul(acc.to(F32), sp)                                         # (H, T, hd)
        for t in range(T):
            for hh in range(H):
                put(S, l, t, hh, _np(sc[hh, t, :t + 1]))
                put(P, l, t, hh, np.concatenate([_np(pi[hh, t, :t + 1]), _np(sp[hh, t]).view(np.int32)]))
                put(O, l, t, hh, _np(o[hh, t]))
        oc = o.permute(1, 0, 2).reshape(T, H * hd)
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
        for t in range(T):
            put(QA2, l, t, -1, np.concatenate([_np(q2[t]), _np(s2[t:t + 1]).view(np.int32)]))
            put(OP, l, t, -1, _np(op[t]))
            put(R1, l, t, -1, _np(r1[t]))
            put(N2, l, t, -1, _np(h2[t]))
            put(QA3, l, t, -1, np.concatenate([_np(q3[t]), _np(s3[t:t + 1]).view(np.int32)]))
            put(GU, l, t, -1, _np(gu[t]))
            put(SM, l, t, -1, _np(sm[t]))
            put(QA4, l, t, -1, np.concatenate([_np(q4[t]), _np(s4[t:t + 1]).view(np.int32)]))
            put(DN, l, t, -1, _np(dn[t]))
            put(R2, l, t, -1, _np(r2[t]))
        x = r2
    leaves = program(cfg, T, l0, l1)
    tr = StageTrace(leaves=leaves, rows=[rows[lf] for lf in leaves], output=_np(x))
    return tr.finalize()


# ---------------------------------------------------------------- weights commitment
def weight_rows(m: DetQwen3):
    """Canonical (name, row) sequence committed by MODEL_REG's weights root (spec/stage-program.md).
    Linear rows are int8 weights widened to int32 followed by the row's fp32 scale bits."""
    def lin(name, dl):
        W = dl.Wq.cpu().numpy().astype(np.int32)
        sw = dl.sw.cpu().numpy().astype(np.float32).view(np.int32)
        for r in range(W.shape[0]):
            yield f"{name}", np.concatenate([W[r], sw[r:r + 1]])
    emb = m.embed.cpu().numpy()
    for r in range(emb.shape[0]):
        yield "embed", emb[r]
    for l, L in enumerate(m.layers):
        for nm in ("in_norm", "post_norm", "q_norm", "k_norm"):
            yield f"{l}.{nm}", L[nm].cpu().numpy().astype(np.float32)
        yield from lin(f"{l}.qkv", L["qkv"])
        yield from lin(f"{l}.o", L["o"])
        yield from lin(f"{l}.gu", L["gu"])
        yield from lin(f"{l}.down", L["down"])
    cos, sin = m.cos.cpu().numpy(), m.sin.cpu().numpy()
    for r in range(cos.shape[0]):
        yield "rope.cos", cos[r]
    for r in range(sin.shape[0]):
        yield "rope.sin", sin[r]


def weights_tree(m: DetQwen3) -> tuple[bytes, list[bytes]]:
    """Leaf i = sha256(0x00 || "w" || u32 i || row_root(row)); root = MerkleRoot(leaves)."""
    hs = [_leaf_hash(b"w" + struct.pack("<I", i) + row_root(row)) for i, (_, row) in enumerate(weight_rows(m))]
    return merkle_root(hs), hs
