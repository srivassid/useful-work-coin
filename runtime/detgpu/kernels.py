"""Fused CUDA kernels for detgpu, each bit-identical to the detrt.core ops it replaces.

Every float op is an explicitly rounded IEEE intrinsic (__fadd_rn, __fmul_rn, __fdiv_rn,
__fsqrt_rn), which the compiler never contracts into a fused multiply-add, in the same order
as detrt. Reductions are either exact (max) or done in detrt's integer fixed point, so their
order does not matter. Compiled with NVRTC through CuPy, as the miner's kernels are.
"""
from __future__ import annotations

import cupy as cp
import numpy as np
import torch

_SRC = r"""
typedef signed char i8;

__device__ __forceinline__ float det_exp_f(float z) {
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
  return r;
}

__device__ __forceinline__ float deq(int acc, float s, float sw) { return __fmul_rn(__fmul_rn(__int2float_rn(acc), s), sw); }

// one warp: 4 values per lane (index lane + 32 j) of a 128-wide head
__device__ __forceinline__ float warp_max(float v) {
  for (int s = 16; s > 0; s >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, s));
  return v;
}
__device__ __forceinline__ long long warp_sum_ll(long long v) {
  for (int s = 16; s > 0; s >>= 1) v += __shfl_xor_sync(0xffffffffu, v, s);
  return v;
}
// detrt rmsnorm's factor r = 1/sqrt(mean(x^2) + eps), exact fixed-point sum of squares, warp-wide
__device__ __forceinline__ float warp_rms_factor(const float* v, int nv, int L, float inv_d, float eps) {
  float mx = 0.f;
  for (int j = 0; j < nv; j++) mx = fmaxf(mx, __fmul_rn(v[j], v[j]));
  mx = warp_max(mx);
  int e; frexpf(mx, &e);
  int sh = (62 - L) - e;
  sh = sh < -1000 ? -1000 : (sh > 1000 ? 1000 : sh);
  const double p2 = __longlong_as_double((long long)(sh + 1023) << 52);
  long long acc = 0;
  for (int j = 0; j < nv; j++) acc += __double2ll_rn((double)__fmul_rn(v[j], v[j]) * p2);
  acc = warp_sum_ll(acc);
  float ss = __double2float_rn(__ll2double_rn(acc) * __longlong_as_double((long long)(-sh + 1023) << 52));
  return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fmul_rn(ss, inv_d), eps)));
}
// quant_sym(qmax=127) of a 128-wide head held as 4 values per lane
__device__ __forceinline__ void warp_quant127(const float* v, signed char* q, int lane, float* scale) {
  float mx = 0.f;
  for (int j = 0; j < 4; j++) mx = fmaxf(mx, fabsf(v[j]));
  mx = warp_max(mx);
  const float s = __fdiv_rn(fmaxf(mx, TINY8), 127.0f);
  for (int j = 0; j < 4; j++) {
    float r = fminf(fmaxf(rintf(__fdiv_rn(v[j], s)), -127.f), 127.f);
    q[lane + 32 * j] = (signed char)(int)r;
  }
  if (lane == 0) *scale = s;
}

// detrt.core.quant_act (fwht + quant_sym(qmax=64) + the 1/sqrt(bs) scale fold), one row per block.
extern "C" __global__ void __launch_bounds__(256)
quant_act_f32(const float* __restrict__ x, int k, int bs, i8* __restrict__ q, float* __restrict__ scale) {
  extern __shared__ float rowf[];
  __shared__ float red[8];
  const int r = blockIdx.x, tid = threadIdx.x;
  const float* xr = x + (long long)r * k;
  for (int i = tid; i < k; i += 256) rowf[i] = xr[i];
  __syncthreads();
  if (bs > 1) {
    for (int lh = 0; (1 << lh) < bs; lh++) {   // stages h = 1, 2, 4, ...: pairs (i, i + h) -> (a + b, a - b)
      const int h = 1 << lh;
      for (int p = tid; p < (k >> 1); p += 256) {
        int i = ((p >> lh) << (lh + 1)) | (p & (h - 1)), j = i + h;
        float a = rowf[i], b = rowf[j];
        rowf[i] = __fadd_rn(a, b);
        rowf[j] = __fsub_rn(a, b);
      }
      __syncthreads();
    }
  }
  float mx = 0.f;
  for (int i = tid; i < k; i += 256) mx = fmaxf(mx, fabsf(rowf[i]));
  for (int s = 16; s > 0; s >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, s));
  if ((tid & 31) == 0) red[tid >> 5] = mx;
  __syncthreads();
  if (tid < 32) {
    mx = tid < 8 ? red[tid] : 0.f;
    for (int s = 4; s > 0; s >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, s));
    if (tid == 0) red[0] = __fdiv_rn(fmaxf(mx, 1e-8f), 64.0f);
  }
  __syncthreads();
  const float s = red[0];
  i8* qr = q + (long long)r * k;
  for (int i = tid; i < k; i += 256) {
    float v = rintf(__fdiv_rn(rowf[i], s));
    v = fminf(fmaxf(v, -64.f), 64.f);
    qr[i] = (i8)(int)v;
  }
  if (tid == 0) scale[r] = bs > 1 ? __fmul_rn(s, __fdiv_rn(1.0f, __fsqrt_rn((float)bs))) : s;
}
// detrt rmsnorm (exact fixed-point sum of squares) then quant_act, one row per block: the
// normalized row is used only by the quantizer, so it never leaves shared memory.
extern "C" __global__ void __launch_bounds__(256)
rmsnorm_quant_f32(const float* __restrict__ x, const float* __restrict__ w, int k, int bs, float inv_d, float eps,
                  int L, i8* __restrict__ q, float* __restrict__ scale) {
  extern __shared__ float rowf[];
  __shared__ float red[8];
  __shared__ long long redl[8];
  __shared__ int sh_shift;
  __shared__ float sh_r;
  const int r = blockIdx.x, tid = threadIdx.x;
  const float* xr = x + (long long)r * k;
  float mx = 0.f;
  for (int i = tid; i < k; i += 256) { float v = xr[i]; rowf[i] = v; mx = fmaxf(mx, __fmul_rn(v, v)); }
  for (int s = 16; s > 0; s >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, s));
  if ((tid & 31) == 0) red[tid >> 5] = mx;
  __syncthreads();
  if (tid == 0) {
    float m = 0.f;
    for (int j = 0; j < 8; j++) m = fmaxf(m, red[j]);
    int e; frexpf(m, &e);
    int sh = (62 - L) - e;
    sh_shift = sh < -1000 ? -1000 : (sh > 1000 ? 1000 : sh);
  }
  __syncthreads();
  const int sh = sh_shift;
  const double p2 = __longlong_as_double((long long)(sh + 1023) << 52);
  long long acc = 0;
  for (int i = tid; i < k; i += 256) { float v = rowf[i]; acc += __double2ll_rn((double)__fmul_rn(v, v) * p2); }
  for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
  if ((tid & 31) == 0) redl[tid >> 5] = acc;
  __syncthreads();
  if (tid == 0) {
    long long tot = 0;
    for (int j = 0; j < 8; j++) tot += redl[j];
    const double m2 = __longlong_as_double((long long)(-sh + 1023) << 52);
    float ss = __double2float_rn(__ll2double_rn(tot) * m2);
    float mean = __fmul_rn(ss, inv_d);
    sh_r = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
  }
  __syncthreads();
  const float rr = sh_r;
  for (int i = tid; i < k; i += 256) rowf[i] = __fmul_rn(__fmul_rn(rowf[i], rr), w[i]);
  __syncthreads();
  if (bs > 1) {
    for (int lh = 0; (1 << lh) < bs; lh++) {
      const int h = 1 << lh;
      for (int p = tid; p < (k >> 1); p += 256) {
        int i = ((p >> lh) << (lh + 1)) | (p & (h - 1)), j = i + h;
        float a = rowf[i], b = rowf[j];
        rowf[i] = __fadd_rn(a, b);
        rowf[j] = __fsub_rn(a, b);
      }
      __syncthreads();
    }
  }
  mx = 0.f;
  for (int i = tid; i < k; i += 256) mx = fmaxf(mx, fabsf(rowf[i]));
  for (int s = 16; s > 0; s >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, s));
  __syncthreads();
  if ((tid & 31) == 0) red[tid >> 5] = mx;
  __syncthreads();
  if (tid < 32) {
    mx = tid < 8 ? red[tid] : 0.f;
    for (int s = 4; s > 0; s >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, s));
    if (tid == 0) red[0] = __fdiv_rn(fmaxf(mx, 1e-8f), 64.0f);
  }
  __syncthreads();
  const float s = red[0];
  i8* qr = q + (long long)r * k;
  for (int i = tid; i < k; i += 256) {
    float v = rintf(__fdiv_rn(rowf[i], s));
    v = fminf(fmaxf(v, -64.f), 64.f);
    qr[i] = (i8)(int)v;
  }
  if (tid == 0) scale[r] = bs > 1 ? __fmul_rn(s, __fdiv_rn(1.0f, __fsqrt_rn((float)bs))) : s;
}
// detrt attention for one query row (b, h, t), all of it: scores from int8 q.k (exact), scale,
// causal mask, max, det_exp, exact fixed-point row sum (MAX_KEYS), probabilities, 14-bit
// quantization, and the exact integer sum of P x V, written in the (N, H*hd) layout o_proj reads.
// Constants (EXP2_*, LOG2E, TINY, PLEVELS) are hex literals generated from detrt; inv_sqrt_hd is an argument.
extern "C" __global__ void __launch_bounds__(128)
attention_row(const signed char* __restrict__ qq, const float* __restrict__ qs, const signed char* __restrict__ kq,
              const float* __restrict__ ks, const signed char* __restrict__ vq, const float* __restrict__ vs,
              int B, int T, int H, int Hkv, float INV_SQRT_HD, float* __restrict__ out) {
  constexpr int HD = 128;
  extern __shared__ float smem[];
  float* row = smem;                   // T floats: scores, then e, then Pv
  int* pi = (int*)(smem + T);          // T ints
  __shared__ float redf[4];
  __shared__ long long redl[4];
  __shared__ float qrow[HD];
  __shared__ float sh_mx, sh_den, sh_sp;
  __shared__ int sh_shift;
  const int t = blockIdx.x % T, h = (blockIdx.x / T) % H, b = blockIdx.x / (T * H), tid = threadIdx.x;
  const int hk = h / (H / Hkv), rq = b * T + t;
  qrow[tid] = (float)qq[((long long)rq * H + h) * HD + tid];
  __syncthreads();
  const float qsc = qs[(long long)rq * H + h];
  // scores: sc = ((sc_int * qs) * ks_j) * inv_sqrt_hd, -inf where j > t
  float mx = NEG_INF;
  for (int j = tid; j < T; j += 128) {
    float v = NEG_INF;
    if (j <= t) {
      const signed char* kr = kq + ((long long)(b * T + j) * Hkv + hk) * HD;
      int acc = 0;
      for (int d = 0; d < HD; d++) acc += (int)qrow[d] * (int)kr[d];
      v = __fmul_rn(__fmul_rn(__fmul_rn((float)acc, qsc), ks[(long long)(b * T + j) * Hkv + hk]), INV_SQRT_HD);
    }
    row[j] = v;
    mx = fmaxf(mx, v);
  }
  for (int s = 16; s > 0; s >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, s));
  if ((tid & 31) == 0) redf[tid >> 5] = mx;
  __syncthreads();
  if (tid == 0) sh_mx = fmaxf(fmaxf(redf[0], redf[1]), fmaxf(redf[2], redf[3]));
  __syncthreads();
  const float m = sh_mx;
  // e = det_exp(sc - mx); row max of |e| for the fixed-point sum
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
  // Pv = (e / den) * vs_j, its row max, then 14-bit levels
  float pm = NEG_INF;
  for (int j = tid; j < T; j += 128) {
    float pv = __fmul_rn(__fdiv_rn(row[j], den), vs[(long long)(b * T + j) * Hkv + hk]);
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
  // o[d] = (sum_j pi_j * v_j[d]) * sp, exact in int64
  long long o = 0;
  for (int j = 0; j <= t && j < T; j++) o += (long long)pi[j] * (long long)vq[((long long)(b * T + j) * Hkv + hk) * HD + tid];
  out[(long long)rq * (H * HD) + h * HD + tid] = __fmul_rn(__ll2float_rn(o), sp);
}
// qkv GEMM output (int32, N x (H + 2 Hkv) * 128) -> quantized q, k, v exactly as detrt.fast:
// dequant, split, q/k per-head rmsnorm, rope, quant_sym(127) (v: quant_sym only). One warp per head.
extern "C" __global__ void __launch_bounds__(256)
qkv_post(const int* __restrict__ acc, const float* __restrict__ s1, const float* __restrict__ sw,
         const float* __restrict__ qnw, const float* __restrict__ knw, const float* __restrict__ cosT,
         const float* __restrict__ sinT, int T, int H, int Hkv, float inv_hd, float eps,
         signed char* __restrict__ qq, float* __restrict__ qs, signed char* __restrict__ kq, float* __restrict__ ks,
         signed char* __restrict__ vq, float* __restrict__ vs) {
  __shared__ float buf[8][128];
  const int row = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nh = H + 2 * Hkv, width = nh * 128, t = row % T;
  const float s = s1[row];
  for (int hh = warp; hh < nh; hh += 8) {
    float v[4];
    for (int j = 0; j < 4; j++) { int c = hh * 128 + lane + 32 * j; v[j] = deq(acc[(long long)row * width + c], s, sw[c]); }
    if (hh < H + Hkv) {                       // q or k head: rmsnorm, rope, quant
      const float* w = hh < H ? qnw : knw;
      const float r = warp_rms_factor(v, 4, 7, inv_hd, eps);
      for (int j = 0; j < 4; j++) { int i = lane + 32 * j; buf[warp][i] = __fmul_rn(__fmul_rn(v[j], r), w[i]); }
      __syncwarp();
      float o[4];
      for (int j = 0; j < 4; j++) {
        int i = lane + 32 * j;
        float z = buf[warp][i];
        float rot = i < 64 ? -buf[warp][i + 64] : buf[warp][i - 64];
        o[j] = __fadd_rn(__fmul_rn(z, cosT[t * 128 + i]), __fmul_rn(rot, sinT[t * 128 + i]));
      }
      __syncwarp();
      if (hh < H) warp_quant127(o, qq + ((long long)row * H + hh) * 128, lane, qs + (long long)row * H + hh);
      else warp_quant127(o, kq + ((long long)row * Hkv + hh - H) * 128, lane, ks + (long long)row * Hkv + hh - H);
    } else {
      const int hv = hh - H - Hkv;
      warp_quant127(v, vq + ((long long)row * Hkv + hv) * 128, lane, vs + (long long)row * Hkv + hv);
    }
  }
}

// out = x + deq(acc): the residual add after o_proj and down_proj
extern "C" __global__ void
dequant_residual(const int* __restrict__ acc, const float* __restrict__ s, const float* __restrict__ sw,
                 const float* __restrict__ x, int k, long long n, float* __restrict__ out) {
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    const long long r = i / k; const int c = (int)(i - r * k);
    out[i] = __fadd_rn(x[i], deq(acc[i], s[r], sw[c]));
  }
}

// gate/up GEMM output (int32, N x 2I) -> quant_act(silu(gate) * up, bs), one row per block
extern "C" __global__ void __launch_bounds__(256)
swiglu_quant(const int* __restrict__ acc, const float* __restrict__ s3, const float* __restrict__ sw, int I, int bs,
             signed char* __restrict__ q, float* __restrict__ scale) {
  extern __shared__ float rowf[];
  __shared__ float red[8];
  const int r = blockIdx.x, tid = threadIdx.x;
  const float s = s3[r];
  const int* ar = acc + (long long)r * 2 * I;
  for (int i = tid; i < I; i += 256) {
    float g = deq(ar[i], s, sw[i]), u = deq(ar[I + i], s, sw[I + i]);
    rowf[i] = __fmul_rn(__fdiv_rn(g, __fadd_rn(1.0f, det_exp_f(-g))), u);
  }
  __syncthreads();
  if (bs > 1) {
    for (int lh = 0; (1 << lh) < bs; lh++) {
      const int h = 1 << lh;
      for (int p = tid; p < (I >> 1); p += 256) {
        int i = ((p >> lh) << (lh + 1)) | (p & (h - 1)), j = i + h;
        float a = rowf[i], b = rowf[j];
        rowf[i] = __fadd_rn(a, b);
        rowf[j] = __fsub_rn(a, b);
      }
      __syncthreads();
    }
  }
  float mx = 0.f;
  for (int i = tid; i < I; i += 256) mx = fmaxf(mx, fabsf(rowf[i]));
  mx = warp_max(mx);
  if ((tid & 31) == 0) red[tid >> 5] = mx;
  __syncthreads();
  if (tid < 32) {
    mx = tid < 8 ? red[tid] : 0.f;
    for (int sft = 4; sft > 0; sft >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, sft));
    if (tid == 0) red[0] = __fdiv_rn(fmaxf(mx, TINY8), 64.0f);
  }
  __syncthreads();
  const float sc = red[0];
  signed char* qr = q + (long long)r * I;
  for (int i = tid; i < I; i += 256) qr[i] = (signed char)(int)fminf(fmaxf(rintf(__fdiv_rn(rowf[i], sc)), -64.f), 64.f);
  if (tid == 0) scale[r] = bs > 1 ? __fmul_rn(sc, __fdiv_rn(1.0f, __fsqrt_rn((float)bs))) : sc;
}
// Tiled version of attention_row: one block = QT query rows of one (b, h); K and V are staged in
// shared memory 64 keys at a time. Same per-element ops as attention_row; all sums are exact.
// smem: scores QT*T floats (reused for the 14-bit levels), q tile QT*128 B, K/V chunk 64*128 B.
extern "C" __global__ void __launch_bounds__(256)
attention_tile(const signed char* __restrict__ qq, const float* __restrict__ qs, const signed char* __restrict__ kq,
               const float* __restrict__ ks, const signed char* __restrict__ vq, const float* __restrict__ vs,
               int B, int T, int H, int Hkv, float INV_SQRT_HD, int QT, float* __restrict__ out) {
  constexpr int HD = 128, KC = 64;
  extern __shared__ float smem[];
  float* sc = smem;                                              // QT x T
  signed char* qt = (signed char*)(smem + QT * T);               // QT x 128
  signed char* kv = qt + QT * HD;                                // KC x 128
  __shared__ float rmx[32], rsp[32];
  const int bh = blockIdx.x, b = bh / H, h = bh % H, hk = h / (H / Hkv);
  const int t0 = blockIdx.y * QT, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nrows = min(QT, T - t0), kmax = t0 + nrows - 1;     // keys 0..kmax are needed
  for (int i = tid; i < nrows * HD; i += 256) {
    int r = i / HD, d = i % HD;
    qt[i] = qq[((long long)(b * T + t0 + r) * H + h) * HD + d];
  }
  // phase 1: scores
  for (int kc = 0; kc <= kmax; kc += KC) {
    __syncthreads();
    const int nk = min(KC, kmax + 1 - kc);
    for (int i = tid; i < nk * HD; i += 256) {
      int jj = i / HD, d = i % HD;
      kv[i] = kq[((long long)(b * T + kc + jj) * Hkv + hk) * HD + d];
    }
    __syncthreads();
    for (int p = tid; p < nrows * KC; p += 256) {
      const int r = p / KC, jj = p % KC, j = kc + jj, t = t0 + r;
      if (jj >= nk) continue;
      float v = NEG_INF;
      if (j <= t) {
        const int* qa = (const int*)(qt + r * HD); const int* kb = (const int*)(kv + jj * HD);
        int acc = 0;
        for (int d = 0; d < HD / 4; d++) acc = __dp4a(qa[d], kb[d], acc);   // exact int8 dot, 4 at a time
        v = __fmul_rn(__fmul_rn(__fmul_rn((float)acc, qs[(long long)(b * T + t) * H + h]),
                                ks[(long long)(b * T + j) * Hkv + hk]), INV_SQRT_HD);
      }
      sc[r * T + j] = v;
    }
  }
  __syncthreads();
  // phase 2: one warp per row; keys j > t are masked (their e and Pv are 0)
  for (int r = warp; r < nrows; r += 8) {
    const int t = t0 + r;
    float* row = sc + r * T;
    float mx = NEG_INF;
    for (int j = lane; j <= t; j += 32) mx = fmaxf(mx, row[j]);
    mx = warp_max(mx);
    float emx = 0.f;
    for (int j = lane; j <= t; j += 32) { float e = det_exp_f(__fsub_rn(row[j], mx)); row[j] = e; emx = fmaxf(emx, fabsf(e)); }
    emx = warp_max(emx);
    int e2; frexpf(emx, &e2);
    int sh = (62 - 20) - e2;
    sh = sh < -1000 ? -1000 : (sh > 1000 ? 1000 : sh);
    const double p2 = __longlong_as_double((long long)(sh + 1023) << 52);
    long long acc = 0;
    for (int j = lane; j <= t; j += 32) acc += __double2ll_rn((double)row[j] * p2);
    acc = warp_sum_ll(acc);
    const float den = __double2float_rn(__ll2double_rn(acc) * __longlong_as_double((long long)(-sh + 1023) << 52));
    float pm = t < T - 1 ? 0.0f : NEG_INF;   // masked keys contribute Pv = 0 to the row max
    for (int j = lane; j <= t; j += 32) {
      float pv = __fmul_rn(__fdiv_rn(row[j], den), vs[(long long)(b * T + j) * Hkv + hk]);
      row[j] = pv;
      pm = fmaxf(pm, pv);
    }
    pm = warp_max(pm);
    const float sp = __fdiv_rn(fmaxf(pm, TINY), PLEVELS);
    int* pi = (int*)row;
    for (int j = lane; j <= kmax; j += 32)
      pi[j] = j <= t ? (int)fminf(fmaxf(rintf(__fdiv_rn(row[j], sp)), 0.0f), PLEVELS) : 0;
    if (lane == 0) rsp[r] = sp;
  }
  // phase 3: o[r][d] = (sum_j pi[r][j] * v[j][d]) * sp[r], exact in int64
  const int d = tid % HD, r0 = tid / HD;        // rows r0, r0 + 2, ...
  long long acc[16];
  for (int i = 0; i < 16; i++) acc[i] = 0;
  for (int kc = 0; kc <= kmax; kc += KC) {
    __syncthreads();
    const int nk = min(KC, kmax + 1 - kc);
    for (int i = tid; i < nk * HD; i += 256) {
      int jj = i / HD, dd = i % HD;
      kv[i] = vq[((long long)(b * T + kc + jj) * Hkv + hk) * HD + dd];
    }
    __syncthreads();
    for (int i = 0; i < 16; i++) {
      const int r = r0 + 2 * i;
      if (r >= nrows) break;
      const int* pi = (const int*)(sc + r * T) + kc;
      int a = 0;   // 64 keys x (14-bit level x int8) < 2^27: exact in int32, then added to int64
      for (int jj = 0; jj < nk; jj++) a += pi[jj] * (int)kv[jj * HD + d];
      acc[i] += a;
    }
  }
  for (int i = 0; i < 16; i++) {
    const int r = r0 + 2 * i;
    if (r >= nrows) break;
    out[(long long)(b * T + t0 + r) * (H * HD) + h * HD + d] = __fmul_rn(__ll2float_rn(acc[i]), rsp[r]);
  }
}

// agent.proto.pooled: per (sequence, column) the mean over tokens, accumulated in float64 in
// token order (as numpy sums axis 0 of a (T, D) array), divided by T, rounded once to float32
extern "C" __global__ void pooled_f32(const float* __restrict__ x, int T, int D, long long BD, float* __restrict__ out) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= BD) return;
  const long long b = i / D; const int d = (int)(i % D);
  const float* p = x + b * T * D + d;
  double s = 0.0;
  for (int t = 0; t < T; t++) s = __dadd_rn(s, (double)p[(long long)t * D]);
  out[i] = __double2float_rn(__ddiv_rn(s, (double)T));
}
"""

_MODS: dict = {}
TILED = True


def _constants() -> str:
    """detrt's float32 constants as exact hex literals (no decimal re-rounding in the compiler)."""
    from detrt import core
    from detrt.qwen3 import P_LEVELS

    def h(v: float) -> str:
        return float(torch.tensor(float(v), dtype=torch.float64).to(torch.float32)).hex() + "f"
    defs = {f"EXP2_{i}": h(c) for i, c in enumerate(core._EXP2)}
    defs.update(LOG2E=h(core._LOG2E), TINY=h(1e-30), TINY8=h(1e-8), PLEVELS=h(float(P_LEVELS)),
                NEG_INF="__int_as_float(0xff800000)", POS_INF="__int_as_float(0x7f800000)")
    return "".join(f"#define {k} {v}\n" for k, v in defs.items())


def _fn(name: str):
    dev = torch.cuda.current_device()
    mod = _MODS.get(dev)
    if mod is None:
        major, minor = torch.cuda.get_device_capability()
        src = _constants() + _SRC
        with cp.cuda.Device(dev):
            mod = _MODS[dev] = cp.RawModule(code=src, options=(f"-arch=sm_{major}{minor}", "-std=c++17"))
    return mod.get_function(name)


_STREAMS: dict = {}


def _stream() -> cp.cuda.Stream:
    ts = torch.cuda.current_stream()
    s = _STREAMS.get(ts.cuda_stream)
    if s is None:
        s = _STREAMS[ts.cuda_stream] = cp.cuda.Stream.from_external(ts)
    return s


def quant_act(x: torch.Tensor, bs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """detrt.core.quant_act(x, bs) for float32 x (rows, k) on a GPU: (int8 q, float32 scale)."""
    x = x.contiguous()
    m, k = x.shape
    assert x.dtype == torch.float32 and (bs <= 1 or k % bs == 0) and 4 * k <= 48 * 1024
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty(m, dtype=torch.float32, device=x.device)
    with cp.cuda.Device(x.device.index):
        _fn("quant_act_f32")((m,), (256,), (np.uint64(x.data_ptr()), np.int32(k), np.int32(bs), np.uint64(q.data_ptr()),
                                          np.uint64(s.data_ptr())), stream=_stream(), shared_mem=4 * k)
    return q, s


def rmsnorm_quant(x: torch.Tensor, w: torch.Tensor, eps: float, bs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """detrt.core.quant_act(detrt.core.det_rmsnorm(x, w, eps), bs) for float32 x (rows, k)."""
    import math
    from detrt.core import F32, F64
    x = x.contiguous()
    m, k = x.shape
    assert x.dtype == torch.float32 and w.dtype == torch.float32 and (bs <= 1 or k % bs == 0) and 4 * k <= 48 * 1024
    inv_d = np.float32(float(torch.tensor(1.0 / k, dtype=F64).to(F32)))
    eps32 = np.float32(float(torch.tensor(eps, dtype=F64).to(F32)))
    L = max(1, math.ceil(math.log2(k)))
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty(m, dtype=torch.float32, device=x.device)
    with cp.cuda.Device(x.device.index):
        _fn("rmsnorm_quant_f32")((m,), (256,), (np.uint64(x.data_ptr()), np.uint64(w.contiguous().data_ptr()), np.int32(k),
                                               np.int32(bs), inv_d, eps32, np.int32(L), np.uint64(q.data_ptr()),
                                               np.uint64(s.data_ptr())), stream=_stream(), shared_mem=4 * k)
    return q, s


def attention(qq, qs, kq, ks, vq, vs, B: int, T: int, H: int, Hkv: int, inv_sqrt_hd_f32: float) -> torch.Tensor:
    """detrt's attention from quantized q, k, v: qq (N, H, hd), kq and vq (N, Hkv, hd) integer
    valued, qs (N, H, 1), ks and vs (N, Hkv, 1). Returns o as (N, H*hd) float32."""
    assert qq.shape[-1] == 128
    N = B * T
    i8 = lambda z: z.to(torch.int8).contiguous()
    f = lambda z: z.contiguous().to(torch.float32)
    out = torch.empty((N, H * 128), dtype=torch.float32, device=qq.device)
    args = [i8(qq), f(qs), i8(kq), f(ks), i8(vq), f(vs)]
    QT = 32
    while QT > 1 and QT * T * 4 > 32 * 1024:
        QT //= 2
    smem = QT * T * 4 + QT * 128 + 64 * 128
    with cp.cuda.Device(qq.device.index):
        if TILED:
            f = _fn("attention_tile")
            f.max_dynamic_shared_size_bytes = max(smem, 48 * 1024)
            f((B * H, -(-T // QT)), (256,), tuple(np.uint64(a.data_ptr()) for a in args) +
              (np.int32(B), np.int32(T), np.int32(H), np.int32(Hkv), np.float32(inv_sqrt_hd_f32), np.int32(QT),
               np.uint64(out.data_ptr())), stream=_stream(), shared_mem=smem)
        else:
            _fn("attention_row")((B * H * T,), (128,), tuple(np.uint64(a.data_ptr()) for a in args) +
                                 (np.int32(B), np.int32(T), np.int32(H), np.int32(Hkv), np.float32(inv_sqrt_hd_f32),
                                  np.uint64(out.data_ptr())),
                                 stream=_stream(), shared_mem=8 * T)
    return out


def _ptr(t):
    return np.uint64(0 if t is None else t.data_ptr())


def qkv_post(acc, s1, sw, qnw, knw, cos, sin, T: int, H: int, Hkv: int, eps: float):
    """Quantized q, k, v from the qkv GEMM's int32 output: (qq (N,H,128) int8, qs (N,H,1), kq, ks, vq, vs)."""
    from detrt.core import F32, F64
    N = acc.shape[0]
    dev = acc.device
    e = lambda *shape, dt=torch.int8: torch.empty(shape, dtype=dt, device=dev)
    qq, qs, kq, ks, vq, vs = e(N, H, 128), e(N, H, 1, dt=F32), e(N, Hkv, 128), e(N, Hkv, 1, dt=F32), e(N, Hkv, 128), e(N, Hkv, 1, dt=F32)
    inv_hd = np.float32(float(torch.tensor(1.0 / 128, dtype=F64).to(F32)))
    eps32 = np.float32(float(torch.tensor(eps, dtype=F64).to(F32)))
    args = [acc.contiguous(), s1.contiguous(), sw.contiguous(), qnw.contiguous(), knw.contiguous(), cos.contiguous(), sin.contiguous()]
    with cp.cuda.Device(dev.index):
        _fn("qkv_post")((N,), (256,), tuple(_ptr(a) for a in args) + (np.int32(T), np.int32(H), np.int32(Hkv), inv_hd, eps32) +
                        tuple(_ptr(a) for a in (qq, qs, kq, ks, vq, vs)), stream=_stream())
    return qq, qs, kq, ks, vq, vs


def dequant_residual(acc, s, sw, x):
    acc, x = acc.contiguous(), x.contiguous()
    N, k = acc.shape
    out = torch.empty((N, k), dtype=torch.float32, device=acc.device)
    n = N * k
    with cp.cuda.Device(acc.device.index):
        _fn("dequant_residual")((min(65535, -(-n // 256)),), (256,), (_ptr(acc), _ptr(s.contiguous()), _ptr(sw.contiguous()), _ptr(x),
                                                              np.int32(k), np.int64(n), _ptr(out)), stream=_stream())
    return out


def swiglu_quant(acc, s3, sw, bs: int):
    acc = acc.contiguous()
    N, two_i = acc.shape
    I = two_i // 2
    assert 4 * I <= 48 * 1024 and (bs <= 1 or I % bs == 0)
    q = torch.empty((N, I), dtype=torch.int8, device=acc.device)
    sc = torch.empty(N, dtype=torch.float32, device=acc.device)
    with cp.cuda.Device(acc.device.index):
        _fn("swiglu_quant")((N,), (256,), (_ptr(acc), _ptr(s3.contiguous()), _ptr(sw.contiguous()), np.int32(I), np.int32(bs),
                                           _ptr(q), _ptr(sc)), stream=_stream(), shared_mem=4 * I)
    return q, sc


def pooled(x: torch.Tensor) -> torch.Tensor:
    """agent.proto.pooled for each sequence of x (B, T, D) float32 on the GPU -> (B, D)."""
    x = x.contiguous()
    B, T, D = x.shape
    out = torch.empty((B, D), dtype=torch.float32, device=x.device)
    with cp.cuda.Device(x.device.index):
        _fn("pooled_f32")((-(-B * D // 256),), (256,), (_ptr(x), np.int32(T), np.int32(D), np.int64(B * D), _ptr(out)),
                          stream=_stream())
    return out
