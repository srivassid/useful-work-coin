"""CUDA kernels for fast generation, bit-identical to the reference (detrt / detgen.head):

  attention_cache   one query per sequence against its KV cache: detgpu's attention_row with
                    the keys j <= pos[b] read from the cache. Keys past a query contribute exact
                    zeros in every reference path, so skipping them changes no bit.
  gumbel_scores     score = logit / temperature + gumbel(key, pos, v): splitmix64 in unsigned
                    64-bit arithmetic (wrapping is defined), u exact, det_log as detrt.core.
"""
from __future__ import annotations

import cupy as cp
import numpy as np
import torch

_SRC = r"""
__device__ __forceinline__ float det_log_f(float x) {
  int e; float m = frexpf(x, &e);
  m = __fmul_rn(m, 2.0f);
  const float ef = (float)(e - 1);
  const float s = __fdiv_rn(__fsub_rn(m, 1.0f), __fadd_rn(m, 1.0f));
  const float s2 = __fmul_rn(s, s);
  float acc = LOGT_15;
  acc = __fadd_rn(__fmul_rn(acc, s2), LOGT_13); acc = __fadd_rn(__fmul_rn(acc, s2), LOGT_11);
  acc = __fadd_rn(__fmul_rn(acc, s2), LOGT_9);  acc = __fadd_rn(__fmul_rn(acc, s2), LOGT_7);
  acc = __fadd_rn(__fmul_rn(acc, s2), LOGT_5);  acc = __fadd_rn(__fmul_rn(acc, s2), LOGT_3);
  acc = __fadd_rn(__fmul_rn(acc, s2), LOGT_1);
  const float lm = __fmul_rn(__fmul_rn(acc, s), 2.0f);
  return __fadd_rn(__fmul_rn(ef, LN2F), lm);
}

__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
  x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 27; x *= 0x94D049BB133111EBull; return x ^ (x >> 31);
}

// scores (B, V) from logits (B, V): logit / temp + gumbel(key, pos[b], v)
extern "C" __global__ void
gumbel_scores(const float* __restrict__ logits, const long long* __restrict__ pos, unsigned long long key,
              float temp, int V, long long n, float* __restrict__ out) {
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    const long long b = i / V; const int v = (int)(i - b * V);
    const unsigned long long x = mix64(key ^ mix64(((unsigned long long)pos[b] << 32) | (unsigned long long)(unsigned)v));
    const float u = (float)((double)((x >> 41) * 2ull + 1ull) * 0x1p-24);
    const float g = -det_log_f(-det_log_f(u));
    out[i] = __fadd_rn(__fdiv_rn(logits[i], temp), g);
  }
}

// One query per row r (sequence bidx[r] at position pos[r]) against keys 0..pos[r] of that
// sequence's cache. Cache layout: kc, vc (B, S, Hkv, 128) int8, ksc, vsc (B, S, Hkv) float32;
// q (R, H, 128) int8, qs (R, H); out (R, H*128).
extern "C" __global__ void __launch_bounds__(128)
attention_cache(const signed char* __restrict__ qq, const float* __restrict__ qs, const signed char* __restrict__ kc,
                const float* __restrict__ ksc, const signed char* __restrict__ vc, const float* __restrict__ vsc,
                const long long* __restrict__ pos, const long long* __restrict__ bidx, int S, int H, int Hkv,
                float INV_SQRT_HD, float* __restrict__ out) {
  constexpr int HD = 128;
  extern __shared__ float smem[];
  float* row = smem;                   // S floats: scores, then e, then Pv
  int* pi = (int*)(smem + S);          // S ints
  __shared__ float redf[4];
  __shared__ long long redl[4];
  __shared__ float qrow[HD];
  __shared__ float sh_mx, sh_den, sh_sp;
  __shared__ int sh_shift;
  const int h = blockIdx.x % H, r = blockIdx.x / H, tid = threadIdx.x;
  const int hk = h / (H / Hkv);
  const long long b = bidx[r];
  const int t = (int)pos[r], T = t + 1;
  qrow[tid] = (float)qq[((long long)r * H + h) * HD + tid];
  __syncthreads();
  const float qsc = qs[(long long)r * H + h];
  float mx = NEG_INF;
  for (int j = tid; j < T; j += 128) {
    const signed char* kr = kc + (((long long)b * S + j) * Hkv + hk) * HD;
    int acc = 0;
    for (int d = 0; d < HD; d++) acc += (int)qrow[d] * (int)kr[d];
    const float v = __fmul_rn(__fmul_rn(__fmul_rn((float)acc, qsc), ksc[((long long)b * S + j) * Hkv + hk]), INV_SQRT_HD);
    row[j] = v;
    mx = fmaxf(mx, v);
  }
  for (int s = 16; s > 0; s >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, s));
  if ((tid & 31) == 0) redf[tid >> 5] = mx;
  __syncthreads();
  if (tid == 0) sh_mx = fmaxf(fmaxf(redf[0], redf[1]), fmaxf(redf[2], redf[3]));
  __syncthreads();
  const float m = sh_mx;
  float emx = 0.f;
  for (int j = tid; j < T; j += 128) {
    float z = __fsub_rn(row[j], m);
    float tt = __fmul_rn(z, LOG2E);
    bool under = tt < -126.0f, over = tt >= 128.0f;
    if (under || over || isnan(tt)) tt = 0.0f;
    float n = floorf(tt);
    float f = __fsub_rn(tt, n);
    float p = EXP2_7;
    p = __fadd_rn(__fmul_rn(p, f), EXP2_6); p = __fadd_rn(__fmul_rn(p, f), EXP2_5);
    p = __fadd_rn(__fmul_rn(p, f), EXP2_4); p = __fadd_rn(__fmul_rn(p, f), EXP2_3);
    p = __fadd_rn(__fmul_rn(p, f), EXP2_2); p = __fadd_rn(__fmul_rn(p, f), EXP2_1);
    p = __fadd_rn(__fmul_rn(p, f), EXP2_0);
    float r = __fmul_rn(p, __int_as_float(((int)n + 127) << 23));
    if (under) r = 0.0f;
    if (over) r = POS_INF;
    if (isnan(z)) r = z;
    row[j] = r;
    emx = fmaxf(emx, fabsf(r));
  }
  for (int s = 16; s > 0; s >>= 1) emx = fmaxf(emx, __shfl_xor_sync(0xffffffffu, emx, s));
  __syncthreads();
  if ((tid & 31) == 0) redf[tid >> 5] = emx;
  __syncthreads();
  if (tid == 0) {
    float a = fmaxf(fmaxf(redf[0], redf[1]), fmaxf(redf[2], redf[3]));
    int e2; frexpf(a, &e2);
    int sh = (62 - 20) - e2;   // MAX_KEYS = 2**20
    sh_shift = sh < -1000 ? -1000 : (sh > 1000 ? 1000 : sh);
  }
  __syncthreads();
  const int sh = sh_shift;
  const double p2 = __longlong_as_double((long long)(sh + 1023) << 52);
  long long acc = 0;
  for (int j = tid; j < T; j += 128) acc += __double2ll_rn((double)row[j] * p2);
  for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
  if ((tid & 31) == 0) redl[tid >> 5] = acc;
  __syncthreads();
  if (tid == 0) {
    long long tot = redl[0] + redl[1] + redl[2] + redl[3];
    sh_den = __double2float_rn(__ll2double_rn(tot) * __longlong_as_double((long long)(-sh + 1023) << 52));
  }
  __syncthreads();
  const float den = sh_den;
  float pm = NEG_INF;
  for (int j = tid; j < T; j += 128) {
    float pv = __fmul_rn(__fdiv_rn(row[j], den), vsc[((long long)b * S + j) * Hkv + hk]);
    row[j] = pv;
    pm = fmaxf(pm, pv);
  }
  for (int s = 16; s > 0; s >>= 1) pm = fmaxf(pm, __shfl_xor_sync(0xffffffffu, pm, s));
  __syncthreads();
  if ((tid & 31) == 0) redf[tid >> 5] = pm;
  __syncthreads();
  if (tid == 0) sh_sp = __fdiv_rn(fmaxf(fmaxf(fmaxf(redf[0], redf[1]), fmaxf(redf[2], redf[3])), TINY), PLEVELS);
  __syncthreads();
  const float sp = sh_sp;
  for (int j = tid; j < T; j += 128) {
    float v = rintf(__fdiv_rn(row[j], sp));
    v = fminf(fmaxf(v, 0.0f), PLEVELS);
    pi[j] = (int)v;
  }
  __syncthreads();
  long long o = 0;
  for (int j = 0; j < T; j++) o += (long long)pi[j] * (long long)vc[(((long long)b * S + j) * Hkv + hk) * HD + tid];
  out[((long long)r * H + h) * HD + tid] = __fmul_rn(__ll2float_rn(o), sp);
}
"""

_MODS: dict = {}


def _constants() -> str:
    from detgpu.kernels import _constants as base

    def h(v: float) -> str:
        return float(torch.tensor(float(v), dtype=torch.float64).to(torch.float32)).hex() + "f"
    extra = {f"LOGT_{k}": h(1.0 / k) for k in range(1, 16, 2)}
    extra["LN2F"] = h(0.6931471805599453)
    return base() + "".join(f"#define {k} {v}\n" for k, v in extra.items())


def _fn(name: str):
    dev = cp.cuda.Device().id  # the device the wrapper selected from its tensors (layers may sit on several GPUs)
    mod = _MODS.get(dev)
    if mod is None:
        major, minor = torch.cuda.get_device_capability(dev)
        with cp.cuda.Device(dev):
            mod = _MODS[dev] = cp.RawModule(code=_constants() + _SRC, options=(f"-arch=sm_{major}{minor}", "-std=c++17"))
    return mod.get_function(name)


def _stream():
    from detgpu.kernels import _stream as s
    return s()


def _p(t):
    return np.uint64(t.data_ptr())


def gumbel_scores(logits: torch.Tensor, pos: torch.Tensor, key: int, temp: float) -> torch.Tensor:
    """logits (B, V) float32, pos (B,) int64 -> scores (B, V) float32; temp: the fp32 temperature."""
    logits = logits.contiguous()
    B, V = logits.shape
    out = torch.empty_like(logits)
    n = B * V
    with cp.cuda.Device(logits.device.index):
        _fn("gumbel_scores")((min(65535, -(-n // 256)),), (256,), (_p(logits), _p(pos.to(torch.int64).contiguous()),
                                                                 np.uint64(key), np.float32(temp), np.int32(V), np.int64(n),
                                                                 _p(out)), stream=_stream())
    return out


def attention_cache(qq, qs, kc, ksc, vc, vsc, pos, bidx, H: int, Hkv: int, inv_sqrt_hd: float) -> torch.Tensor:
    """qq (R, H, 128) int8, qs (R, H, 1): one query per row; caches (B, S, Hkv, 128) int8 and
    (B, S, Hkv) float32; pos (R,) the query positions (keys 0..pos attended), bidx (R,) each
    row's sequence. Returns (R, H*128) float32."""
    R, S = qq.shape[0], kc.shape[1]
    out = torch.empty((R, H * 128), dtype=torch.float32, device=qq.device)
    args = [qq.to(torch.int8).contiguous(), qs.contiguous().to(torch.float32).reshape(R, H), kc, ksc, vc, vsc,
            pos.to(torch.int64).contiguous(), bidx.to(torch.int64).contiguous()]
    with cp.cuda.Device(qq.device.index):
        f = _fn("attention_cache")
        smem = 8 * S
        f.max_dynamic_shared_size_bytes = max(smem, 48 * 1024)
        f((R * H,), (128,), tuple(_p(a) for a in args) + (np.int32(S), np.int32(H), np.int32(Hkv),
                                                         np.float32(inv_sqrt_hd), _p(out)), stream=_stream(), shared_mem=smem)
    return out
