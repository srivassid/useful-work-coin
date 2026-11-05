"""Deterministic math primitives for the reference runtime (design doc, build step 2).

Every function here gives bit-identical results on any CUDA GPU and on CPU, and does not
depend on how rows are batched, tiled, or split between prefill and decode, because:

  * sums are exact: integer accumulation, or fp32/fp64 GEMMs over small integers whose every
    partial sum is exactly representable, or int64 fixed point with a scale that depends
    only on the row's max (never on its length or order);
  * elementwise math is "emulated fp32": each + - * / sqrt is evaluated in float64 and rounded
    once to float32 (fadd, fmul, ...). For these ops that equals the correctly rounded fp32
    result (double rounding is innocuous since 53 >= 2*24 + 2), and float64 kernels are never
    affected by fast-math (PyTorch's CUDA fp32 sqrt is not correctly rounded, and fp32 fast-math
    can flush denormals); one op per kernel, so no fused multiply-add;
  * exp and log are polynomials built from those ops, with fixed coefficients;
  * max, frexp, floor, round (half-even), int<->float conversions are exact or
    correctly rounded everywhere.
"""
from __future__ import annotations

import math

import torch

F32, F64, I64 = torch.float32, torch.float64, torch.int64


def setup_determinism():
    """Disable the cuBLAS shortcuts that change float results (TF32, reduced-precision split-K)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_flush_denormal(False)


_CONST: dict = {}


def c32(v: float, like: torch.Tensor) -> torch.Tensor:
    """A 0-dim float32 constant on like's device (IEEE double -> float32 round-to-nearest), cached."""
    key = (float(v), str(like.device))
    t = _CONST.get(key)
    if t is None:
        t = _CONST[key] = torch.tensor(float(torch.tensor(v, dtype=F64).to(F32)), dtype=F32, device=like.device)
    return t


# ------------------------------------------------------------------ emulated fp32 arithmetic
def _d(x):
    return x.to(F64)


def fadd(a, b):
    return (_d(a) + _d(b)).to(F32)


def fsub(a, b):
    return (_d(a) - _d(b)).to(F32)


def fmul(a, b):
    return (_d(a) * _d(b)).to(F32)


def fdiv(a, b):
    return (_d(a) / _d(b)).to(F32)


def fsqrt(a):
    return torch.sqrt(_d(a)).to(F32)


# ------------------------------------------------------------------ powers of two
def pow2_f32(n: torch.Tensor) -> torch.Tensor:
    """Exactly 2**n as float32 for integer n in [-126, 127] (built from the bit pattern)."""
    return ((n.to(torch.int32) + 127) << 23).view(F32)


def pow2_f64(n: torch.Tensor) -> torch.Tensor:
    """Exactly 2**n as float64 for integer n in [-1022, 1023]."""
    return ((n.to(I64) + 1023) << 52).view(F64)


# ------------------------------------------------------------------ exact row sums
def exact_rowsum(x: torch.Tensor, max_terms: int) -> torch.Tensor:
    """Sum of float32 rows along the last dim, order-independent and hardware-independent.

    Each term is rounded once to int64 fixed point at scale 2**shift, where shift depends only
    on the row max and on `max_terms` (a fixed bound on the row length, so a softmax row of
    length t+1 in decode and length T with masked zeros in prefill use the same scale).
    The int64 sum is exact; the result is rounded once to float64, then to float32.
    """
    L = max(1, math.ceil(math.log2(max_terms)))
    amax = x.abs().amax(dim=-1, keepdim=True)
    _, e = torch.frexp(amax)  # amax < 2**e (e is 0-ish for amax == 0)
    shift = (62 - L) - e.to(I64)  # every |term| * 2**shift < 2**(62-L)
    shift = shift.clamp(-1000, 1000)
    q = torch.round(x.to(F64) * pow2_f64(shift)).to(I64)
    s = q.sum(dim=-1, keepdim=True)
    return (s.to(F64) * pow2_f64(-shift)).to(F32).squeeze(-1)


# ------------------------------------------------------------------ exp / log
# 2**f on [0, 1): degree-7 relative least squares, max rel err 7.7e-8 in fp32 Horner
_EXP2 = [1.0, 0.6931471824645996, 0.2402263879776001, 0.055505119264125824, 0.00961404386907816,
         0.001342222560197115, 0.0001435537706129253, 2.1489791834028438e-05]
_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453


def det_exp(z: torch.Tensor) -> torch.Tensor:
    """exp(z) in float32 from IEEE ops only. Underflow (z < ~-87.3) -> 0, overflow -> inf."""
    z = z.to(F32)
    t = fmul(z, c32(_LOG2E, z))
    under = t < c32(-126.0, z)
    over = t >= c32(128.0, z)
    t = torch.where(under | over | torch.isnan(t), torch.zeros_like(t), t)
    n = torch.floor(t)
    f = fsub(t, n)  # exact
    p = torch.full_like(f, float(torch.tensor(_EXP2[7]).to(F32)))
    for cf in reversed(_EXP2[:7]):
        p = fadd(fmul(p, f), c32(cf, f))
    r = fmul(p, pow2_f32(n))
    r = torch.where(under, torch.zeros_like(r), r)
    r = torch.where(over, torch.full_like(r, float("inf")), r)
    return torch.where(torch.isnan(z), z, r)


def det_log(x: torch.Tensor) -> torch.Tensor:
    """log(x) for x > 0 in float32: x = m 2^e, m in [1,2); log m = 2 atanh((m-1)/(m+1)) (odd series to s^15)."""
    x = x.to(F32)
    m, e = torch.frexp(x)  # m in [0.5, 1)
    m = fmul(m, c32(2.0, m))
    e = (e - 1).to(F32)
    one = c32(1.0, m)
    s = fdiv(fsub(m, one), fadd(m, one))
    s2 = fmul(s, s)
    terms = [1.0 / k for k in range(15, 0, -2)]
    acc = torch.full_like(s, float(torch.tensor(terms[0]).to(F32)))
    for t in terms[1:]:
        acc = fadd(fmul(acc, s2), c32(t, s))
    lm = fmul(fmul(acc, s), c32(2.0, s))
    return fadd(fmul(e, c32(_LN2, e)), lm)


def det_silu(x: torch.Tensor) -> torch.Tensor:
    x = x.to(F32)
    return fdiv(x, fadd(c32(1.0, x), det_exp(-x)))


def det_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """x * 1/sqrt(mean(x^2) + eps) * w along the last dim, with an exact sum of squares."""
    x = x.to(F32)
    d = x.shape[-1]
    ss = exact_rowsum(fmul(x, x), d)
    mean = fmul(ss, c32(1.0 / d, ss))
    r = fdiv(c32(1.0, mean), fsqrt(fadd(mean, c32(eps, mean))))
    return fmul(fmul(x, r.unsqueeze(-1)), w)


# ------------------------------------------------------------------ quantization
def quant_sym(x: torch.Tensor, qmax: int, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row quantization: s = max(max|x|, 1e-8) / qmax, q = round_half_even(x / s)."""
    x = x.to(F32)
    amax = x.abs().amax(dim=dim, keepdim=True)
    s = fdiv(torch.maximum(amax, c32(1e-8, amax)), c32(float(qmax), amax))
    q = torch.round(fdiv(x, s)).clamp_(-qmax, qmax)
    return q, s


def fwht(x: torch.Tensor, bs: int) -> torch.Tensor:
    """Unnormalized block Walsh-Hadamard transform along the last dim, in the exact add order of
    the GPU kernel quant_rows_had (stages h = 1, 2, 4, ...; pairs (i, i+h) -> (a+b, a-b))."""
    if bs <= 1:
        return x.to(F32)
    shp = x.shape
    k = shp[-1]
    v = x.to(F32).reshape(-1, k)
    h = 1
    while h < bs:
        v = v.reshape(v.shape[0], k // (2 * h), 2, h)
        a, b = v[:, :, 0, :], v[:, :, 1, :]
        v = torch.stack([fadd(a, b), fsub(a, b)], dim=2).reshape(-1, k)
        h *= 2
    return v.reshape(shp)


def quant_act(x: torch.Tensor, bs: int, qmax: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear-layer activation quantizer, identical to ops.quant_rows_had:
    q = round(y / s), s = max|y| / 64 with y = fwht(x); returned scale is s / sqrt(bs)."""
    y = fwht(x, bs)
    q, s = quant_sym(y, qmax)
    if bs > 1:
        s = fmul(s, fdiv(c32(1.0, s), fsqrt(c32(float(bs), s))))
    return q.to(torch.int8), s.squeeze(-1)


# ------------------------------------------------------------------ exact small-integer GEMM
def exact_matmul_small(a: torch.Tensor, b: torch.Tensor, bound: int) -> torch.Tensor:
    """a @ b for integer-valued float tensors where every partial sum is < `bound` in magnitude.
    Uses fp32 if bound <= 2**24 and fp64 if bound <= 2**53, so any summation order is exact."""
    if bound <= 2 ** 24:
        return a.to(F32) @ b.to(F32)
    assert bound <= 2 ** 53
    return (a.to(F64) @ b.to(F64))
