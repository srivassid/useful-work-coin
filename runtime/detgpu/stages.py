"""Stage outputs (forward only) for B sequences, the same bits as detrt.fast.run_stages(keep_rows=False),
with native float32 arithmetic and GPU-resident activations. See the package docstring."""
from __future__ import annotations

import math

import torch

from detrt import core
from detrt.core import F32, F64, I64, c32, pow2_f32, pow2_f64
from detrt.qwen3 import KEY_CHUNK, MAX_KEYS, P_LEVELS, DetQwen3


# ---------------------------------------------------------------- core ops in native float32
def exact_rowsum(x: torch.Tensor, max_terms: int) -> torch.Tensor:
    """detrt.core.exact_rowsum: the float64 scaling there is exact, so it stays as it is."""
    return core.exact_rowsum(x, max_terms)


_EXP2_TOP = float(torch.tensor(core._EXP2[7]).to(F32))


def det_exp(z: torch.Tensor) -> torch.Tensor:
    t = z * c32(core._LOG2E, z)
    under = t < c32(-126.0, z)
    over = t >= c32(128.0, z)
    t = torch.where(under | over | torch.isnan(t), torch.zeros_like(t), t)
    n = torch.floor(t)
    f = t - n
    p = torch.full_like(f, _EXP2_TOP)
    for cf in reversed(core._EXP2[:7]):
        p = p * f + c32(cf, f)  # two separate kernels: no fused multiply-add
    r = p * pow2_f32(n)
    r = torch.where(under, torch.zeros_like(r), r)
    r = torch.where(over, torch.full_like(r, float("inf")), r)
    return torch.where(torch.isnan(z), z, r)


def det_silu(x: torch.Tensor) -> torch.Tensor:
    return x / (c32(1.0, x) + det_exp(-x))


def det_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    d = x.shape[-1]
    ss = exact_rowsum(x * x, d)
    mean = ss * c32(1.0 / d, ss)
    r = c32(1.0, mean) / torch.sqrt(mean + c32(eps, mean))
    return x * r.unsqueeze(-1) * w


def quant_sym(x: torch.Tensor, qmax: int) -> tuple[torch.Tensor, torch.Tensor]:
    amax = x.abs().amax(dim=-1, keepdim=True)
    s = torch.maximum(amax, c32(1e-8, amax)) / c32(float(qmax), amax)
    q = torch.round(x / s).clamp_(-qmax, qmax)
    return q, s


def fwht(x: torch.Tensor, bs: int) -> torch.Tensor:
    if bs <= 1:
        return x
    shp = x.shape
    k = shp[-1]
    v = x.reshape(-1, k)
    h = 1
    while h < bs:
        v = v.reshape(v.shape[0], k // (2 * h), 2, h)
        a, b = v[:, :, 0, :], v[:, :, 1, :]
        v = torch.stack([a + b, a - b], dim=2).reshape(-1, k)
        h *= 2
    return v.reshape(shp)


def quant_act(x: torch.Tensor, bs: int, qmax: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    y = fwht(x, bs)
    q, s = quant_sym(y, qmax)
    if bs > 1:
        s = s * (c32(1.0, s) / torch.sqrt(c32(float(bs), s)))
    return q.to(torch.int8), s.squeeze(-1)


USE_KERNELS = True
FUSED = True


def linear(L, name: str, x: torch.Tensor) -> torch.Tensor:
    lin = L[name]
    if USE_KERNELS and x.is_cuda:
        from . import kernels
        q, s = kernels.quant_act(x, lin.bs)
    else:
        q, s = quant_act(x, lin.bs)
    return lin.int_gemm(q).to(F32) * s[:, None] * lin.sw[None, :]


def norm_linear(L, norm: str, name: str, x: torch.Tensor, eps: float) -> torch.Tensor:
    """linear(L, name, det_rmsnorm(x, L[norm], eps)), with the norm fused into the quantizer."""
    lin = L[name]
    if USE_KERNELS and x.is_cuda:
        from . import kernels
        q, s = kernels.rmsnorm_quant(x, L[norm], eps, lin.bs)
        return lin.int_gemm(q).to(F32) * s[:, None] * lin.sw[None, :]
    return linear(L, name, det_rmsnorm(x, L[norm], eps))


def attention_torch(m, qq, qs, kq, ks, vq, vs, B, T):
    dev = m.device
    H, Hkv, hd = m.H, m.Hkv, m.hd
    N = B * T
    rep = H // Hkv
    mask = torch.arange(T, device=dev)[None, :] <= torch.arange(T, device=dev)[:, None]
    def heads(z, n):
        return z.reshape(B, T, n, -1).permute(0, 2, 1, 3)
    K = heads(kq, Hkv).repeat_interleave(rep, 1)
    SK = heads(ks, Hkv).repeat_interleave(rep, 1)
    V = heads(vq, Hkv).repeat_interleave(rep, 1)
    SV = heads(vs, Hkv).repeat_interleave(rep, 1)
    Q, QS = heads(qq, H), heads(qs, H)
    sc_int = core.exact_matmul_small(Q, K.transpose(-1, -2), 127 * 127 * hd)
    sc = sc_int * QS * SK.transpose(-1, -2) * c32(m.inv_sqrt_hd, sc_int)
    sc = torch.where(mask[None, None], sc, torch.full_like(sc, float("-inf")))
    mx = sc.amax(-1, keepdim=True)
    e = det_exp(sc - mx)
    den = exact_rowsum(e, MAX_KEYS)
    Pr = e / den.unsqueeze(-1)
    Pv = Pr * SV.transpose(-1, -2)
    pmax = Pv.amax(-1, keepdim=True)
    sp = torch.maximum(pmax, c32(1e-30, pmax)) / c32(float(P_LEVELS), pmax)
    pi = torch.round(Pv / sp).clamp_(0, P_LEVELS).to(torch.int32)
    hi, lo = (pi >> 7).to(F32), (pi & 127).to(F32)
    acc = torch.zeros(B, H, T, hd, dtype=I64, device=dev)
    Vf = V.to(F32)
    for c0 in range(0, T, KEY_CHUNK):
        bnd = 127 * 127 * KEY_CHUNK
        acc += (core.exact_matmul_small(hi[..., c0:c0 + KEY_CHUNK], Vf[..., c0:c0 + KEY_CHUNK, :], bnd).to(I64) << 7) \
            + core.exact_matmul_small(lo[..., c0:c0 + KEY_CHUNK], Vf[..., c0:c0 + KEY_CHUNK, :], bnd).to(I64)
    o = acc.to(F32) * sp
    oc = o.permute(0, 2, 1, 3).reshape(N, H * hd)
    return oc


def layer_fused(m: DetQwen3, L, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
    """One decoder layer with the fused kernels: x (N, D) float32 -> (N, D) float32, bit-identical to detrt."""
    from . import kernels
    H, Hkv = m.H, m.Hkv
    q1, s1 = kernels.rmsnorm_quant(x, L["in_norm"], m.eps, L["qkv"].bs)
    qq, qs, kq, ks, vq, vs = kernels.qkv_post(L["qkv"].int_gemm(q1), s1, L["qkv"].sw, L["q_norm"], L["k_norm"],
                                              m.cos[:T], m.sin[:T], T, H, Hkv, m.eps, m.qk_norm)
    oc = kernels.attention(qq, qs, kq, ks, vq, vs, B, T, H, Hkv, m.inv_sqrt_hd)
    q2, s2 = kernels.quant_act(oc, L["o"].bs)
    r1 = kernels.dequant_residual(L["o"].int_gemm(q2), s2, L["o"].sw, x)
    q3, s3 = kernels.rmsnorm_quant(r1, L["post_norm"], m.eps, L["gu"].bs)
    q4, s4 = kernels.swiglu_quant(L["gu"].int_gemm(q3), s3, L["gu"].sw, L["down"].bs)
    return kernels.dequant_residual(L["down"].int_gemm(q4), s4, L["down"].sw, r1)


# ---------------------------------------------------------------- stages
@torch.no_grad()
def stage_outputs(m: DetQwen3, tokens: torch.Tensor | None, x: torch.Tensor | None, l0: int, l1: int) -> torch.Tensor:
    """Layers l0..l1-1 for B sequences of T tokens: tokens (B, T) int64 when l0 == 0, else x
    (B, T, D) float32, the previous group's output. Returns (B, T, D) float32 on the GPU, the
    rows each stage is committed by (bit-identical to detrt.fast.run_stages)."""
    dev = m.device
    H, Hkv, hd, D = m.H, m.Hkv, m.hd, m.D
    if l0 == 0:
        B, T = tokens.shape
        x = m.embed[tokens.to(dev).reshape(-1)]
    else:
        B, T = x.shape[0], x.shape[1]
        x = x.to(dev, F32).reshape(B * T, D)
    N = B * T
    pos = torch.arange(T, device=dev).repeat(B)
    rep = H // Hkv
    half = hd // 2
    cos, sin = m.cos[pos][:, None, :], m.sin[pos][:, None, :]
    mask = torch.arange(T, device=dev)[None, :] <= torch.arange(T, device=dev)[:, None]
    for l in range(l0, l1):
        L = m.layers[l]
        if FUSED and x.is_cuda and hd == 128:
            x = layer_fused(m, L, x, B, T)
            continue
        qkv = norm_linear(L, "in_norm", "qkv", x, m.eps)
        q, k, v = torch.split(qkv, [H * hd, Hkv * hd, Hkv * hd], dim=-1)
        qn = det_rmsnorm(q.reshape(N, H, hd), L["q_norm"], m.eps) if m.qk_norm else q.reshape(N, H, hd)
        kn = det_rmsnorm(k.reshape(N, Hkv, hd), L["k_norm"], m.eps) if m.qk_norm else k.reshape(N, Hkv, hd)

        def rope(z):
            return z * cos + torch.cat([-z[..., half:], z[..., :half]], -1) * sin
        qr, kr = rope(qn), rope(kn)
        kq, ks = quant_sym(kr, 127)
        vq, vs = quant_sym(v.reshape(N, Hkv, hd), 127)
        qq, qs = quant_sym(qr, 127)

        if USE_KERNELS and x.is_cuda and hd == 128:
            from . import kernels
            oc = kernels.attention(qq, qs, kq, ks, vq, vs, B, T, H, Hkv, m.inv_sqrt_hd)
        else:
            oc = attention_torch(m, qq, qs, kq, ks, vq, vs, B, T)
        r1 = x + linear(L, "o", oc)
        gu = norm_linear(L, "post_norm", "gu", r1, m.eps)
        I_ = gu.shape[1] // 2
        sm = det_silu(gu[:, :I_]) * gu[:, I_:]
        x = r1 + linear(L, "down", sm)
    return x.reshape(B, T, D)


def job_outputs(m: DetQwen3, tokens: torch.Tensor, groups: int, upto: int | None = None) -> list[torch.Tensor]:
    """Every group's output for a batch of sequences, kept on the GPU: [(B, T, D)] * (upto + 1)."""
    lpg = len(m.layers) // groups
    upto = groups - 1 if upto is None else upto
    outs, x = [], None
    for g in range(upto + 1):
        x = stage_outputs(m, tokens if g == 0 else None, x, g * lpg, (g + 1) * lpg)
        outs.append(x)
    return outs
