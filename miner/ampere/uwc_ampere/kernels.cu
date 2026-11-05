// Useful-Work Coin: cuPOW (Pearl NoisyGEMM) kernels for NVIDIA Ampere (sm_86).
//
// Compiled at runtime with NVRTC (CuPy RawModule); no CUDA toolkit needed.
// Semantics follow Pearl's Rust reference (ext/pearl/zk-pow/src/ffi/mine.rs) exactly;
// tests/test_kernels.py checks every kernel against uwc_ampere/reference.py, and
// tests/test_e2e.py checks that winning tiles produce proofs Pearl's verifier accepts.
//
// Main kernel tiling: CTA 128 x 256 x 64, 8 warps (2 x 4), warp tile 64 x 64,
// mma.sync.m16n8k32 s8*s8->s32. A thread holds rows g+8i (i<8) and cols 2t+{0,1}+8j
// (j<8) of its warp tile, which is exactly one hash tile of the mining configuration
// rows=[0,8,..,56], cols=[0,1,8,9,..,56,57]. So the XOR fold is register-local.

typedef unsigned int u32;
typedef int i32;
typedef signed char i8;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned long long u64;

#ifndef STAGES
#define STAGES 3
#endif
#ifndef BN_
#define BN_ 256
#endif
#ifndef MINB
#define MINB 1
#endif
#ifndef GROUP_M
#define GROUP_M 8
#endif
#define BM 128
#define BN BN_
#define BK 64
#define WARPS_N (BN / 64)
#define NTHREADS (2 * WARPS_N * 32)
#define A_STAGE (BM * BK)
#define B_STAGE (BN * BK)
#define STAGE_BYTES (A_STAGE + B_STAGE)

// ------------------------------------------------------------------ BLAKE3
#define F_CHUNK_START 1u
#define F_CHUNK_END 2u
#define F_PARENT 4u
#define F_ROOT 8u
#define F_KEYED 16u

__device__ __constant__ u32 B3_IV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                                        0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};

__device__ __forceinline__ u32 rotr32(u32 x, int n) { return __funnelshift_r(x, x, n); }

#define B3G(a, b, c, d, x, y)              \
  a = a + b + (x); d = rotr32(d ^ a, 16);  \
  c = c + d;       b = rotr32(b ^ c, 12);  \
  a = a + b + (y); d = rotr32(d ^ a, 8);   \
  c = c + d;       b = rotr32(b ^ c, 7);

// out[0..7] = compress(cv, m, counter, block_len, flags) truncated to the chaining value
__device__ __forceinline__ void b3_compress(const u32 cv[8], const u32 min[16], u64 counter,
                                            u32 blen, u32 flags, u32 out[8]) {
  u32 m[16];
#pragma unroll
  for (int i = 0; i < 16; i++) m[i] = min[i];
  u32 v0 = cv[0], v1 = cv[1], v2 = cv[2], v3 = cv[3], v4 = cv[4], v5 = cv[5], v6 = cv[6], v7 = cv[7];
  u32 v8 = B3_IV[0], v9 = B3_IV[1], v10 = B3_IV[2], v11 = B3_IV[3];
  u32 v12 = (u32)counter, v13 = (u32)(counter >> 32), v14 = blen, v15 = flags;
#pragma unroll
  for (int rnd = 0; rnd < 7; rnd++) {
    B3G(v0, v4, v8, v12, m[0], m[1]);
    B3G(v1, v5, v9, v13, m[2], m[3]);
    B3G(v2, v6, v10, v14, m[4], m[5]);
    B3G(v3, v7, v11, v15, m[6], m[7]);
    B3G(v0, v5, v10, v15, m[8], m[9]);
    B3G(v1, v6, v11, v12, m[10], m[11]);
    B3G(v2, v7, v8, v13, m[12], m[13]);
    B3G(v3, v4, v9, v14, m[14], m[15]);
    // MSG_PERMUTATION = [2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]
    u32 t0 = m[2], t1 = m[6], t2 = m[3], t3 = m[10], t4 = m[7], t5 = m[0], t6 = m[4], t7 = m[13];
    u32 t8 = m[1], t9 = m[11], t10 = m[12], t11 = m[5], t12 = m[9], t13 = m[14], t14 = m[15], t15 = m[8];
    m[0] = t0; m[1] = t1; m[2] = t2; m[3] = t3; m[4] = t4; m[5] = t5; m[6] = t6; m[7] = t7;
    m[8] = t8; m[9] = t9; m[10] = t10; m[11] = t11; m[12] = t12; m[13] = t13; m[14] = t14; m[15] = t15;
  }
  out[0] = v0 ^ v8; out[1] = v1 ^ v9; out[2] = v2 ^ v10; out[3] = v3 ^ v11;
  out[4] = v4 ^ v12; out[5] = v5 ^ v13; out[6] = v6 ^ v14; out[7] = v7 ^ v15;
}

// Hash of a single 64-byte message (one chunk, one block). key==nullptr -> unkeyed.
__device__ __forceinline__ void b3_hash64(const u32* key, const u32 msg[16], u32 out[8]) {
  u32 cv[8];
  u32 flags = F_CHUNK_START | F_CHUNK_END | F_ROOT;
  if (key) {
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = key[i];
    flags |= F_KEYED;
  } else {
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = B3_IV[i];
  }
  b3_compress(cv, msg, 0, 64, flags, out);
}

// Level-wise pairwise reduction of `count` chaining values held in smem (8 words each).
// Matches BLAKE3's tree when the group starts at a multiple of a power of two >= blockDim.
__device__ void b3_reduce_smem(u32* cvs, int count, const u32 key[8], u32 kflag, bool root_at_end) {
  int tid = threadIdx.x;
  while (count > 1) {
    int pairs = count >> 1;
    u32 res[8];
    bool active = tid < pairs;
    bool last = (count == 2) && root_at_end;
    if (active) {
      u32 msg[16];
#pragma unroll
      for (int i = 0; i < 8; i++) { msg[i] = cvs[(2 * tid) * 8 + i]; msg[8 + i] = cvs[(2 * tid + 1) * 8 + i]; }
      b3_compress(key, msg, 0, 64, F_PARENT | kflag | (last ? F_ROOT : 0u), res);
    }
    u32 carry[8];
    bool has_carry = (count & 1) && tid == 0;
    if (has_carry) {
#pragma unroll
      for (int i = 0; i < 8; i++) carry[i] = cvs[(count - 1) * 8 + i];
    }
    __syncthreads();
    if (active) {
#pragma unroll
      for (int i = 0; i < 8; i++) cvs[tid * 8 + i] = res[i];
    }
    if (has_carry) {
#pragma unroll
      for (int i = 0; i < 8; i++) cvs[pairs * 8 + i] = carry[i];
    }
    __syncthreads();
    count = pairs + (count & 1);
  }
}

// Stage 1: one thread per 1 KiB chunk, CHUNKS_PER_CTA chunks per CTA reduced in smem.
// data length must be nchunks * 1024 (caller pads). key = 8 words (keyed hash).
#define CHUNKS_PER_CTA 64
extern "C" __global__ void __launch_bounds__(CHUNKS_PER_CTA)
b3_tree_stage1(const u8* __restrict__ data, u64 nchunks, const u32* __restrict__ key, u32* __restrict__ cta_cvs) {
  __shared__ u32 cvs[CHUNKS_PER_CTA * 8];
  u32 k8[8];
#pragma unroll
  for (int i = 0; i < 8; i++) k8[i] = key[i];
  u64 base = (u64)blockIdx.x * CHUNKS_PER_CTA;
  u64 chunk = base + threadIdx.x;
  int count = (int)min((u64)CHUNKS_PER_CTA, nchunks - base);
  // Blocks are staged through smem so each warp load covers whole 64-byte block rows
  // (the naive per-thread 1 KiB stride left most of every sector unused).
  __shared__ uint4 stage[2][CHUNKS_PER_CTA * 4];
  u32 cv[8];
#pragma unroll
  for (int i = 0; i < 8; i++) cv[i] = k8[i];
  const uint4* gbase = reinterpret_cast<const uint4*>(data + base * 1024ull);
  auto fetch = [&](int buf, int b) {
#pragma unroll
    for (int q = 0; q < 4; q++) {
      int idx = threadIdx.x + q * CHUNKS_PER_CTA, c = idx >> 2, part = idx & 3;
      if (c < count) stage[buf][idx] = __ldg(gbase + (u64)c * 64 + b * 4 + part);
    }
  };
  fetch(0, 0);
  __syncthreads();
  for (int b = 0; b < 16; b++) {
    if (b + 1 < 16) fetch((b + 1) & 1, b + 1);
    if (threadIdx.x < count) {
      u32 msg[16];
#pragma unroll
      for (int q = 0; q < 4; q++) {
        uint4 w = stage[b & 1][threadIdx.x * 4 + q];
        msg[4 * q] = w.x; msg[4 * q + 1] = w.y; msg[4 * q + 2] = w.z; msg[4 * q + 3] = w.w;
      }
      u32 flags = F_KEYED | (b == 0 ? F_CHUNK_START : 0u) | (b == 15 ? F_CHUNK_END : 0u) |
                  ((b == 15 && nchunks == 1) ? F_ROOT : 0u);
      u32 out[8];
      b3_compress(cv, msg, chunk, 64, flags, out);
#pragma unroll
      for (int i = 0; i < 8; i++) cv[i] = out[i];
    }
    __syncthreads();
  }
  if (threadIdx.x < count) {
#pragma unroll
    for (int i = 0; i < 8; i++) cvs[threadIdx.x * 8 + i] = cv[i];
  }
  __syncthreads();
  b3_reduce_smem(cvs, count, k8, F_KEYED, gridDim.x == 1);
  if (threadIdx.x < 8) cta_cvs[blockIdx.x * 8 + threadIdx.x] = cvs[threadIdx.x];
}

// Intermediate level for inputs above 64 MiB: each CTA reduces an aligned group of 1024
// chaining values (a perfect subtree, never the root) to one.
extern "C" __global__ void __launch_bounds__(512)
b3_tree_mid(const u32* __restrict__ in_cvs, int count, const u32* __restrict__ key, u32* __restrict__ out_cvs) {
  __shared__ u32 cvs[1024 * 8];
  const int base = blockIdx.x * 1024, cnt = min(1024, count - base);
  for (int i = threadIdx.x; i < cnt * 8; i += blockDim.x) cvs[i] = in_cvs[(u64)base * 8 + i];
  __syncthreads();
  u32 k8[8];
#pragma unroll
  for (int i = 0; i < 8; i++) k8[i] = key[i];
  b3_reduce_smem(cvs, cnt, k8, F_KEYED, false);
  if (threadIdx.x < 8) out_cvs[blockIdx.x * 8 + threadIdx.x] = cvs[threadIdx.x];
}

// Stage 2 (one CTA of 512 threads, up to 1024 stage-1 values = 64 MiB of input): reduce the stage-1 values to the root, then
//   seed_out = blake3( prefix(32B) || keyed_blake3(salt, root || dim_le32 || 0^28) )
// which is Pearl's v3 salted noise-seed derivation (b_seed: prefix=job_key; a_seed: prefix=b_seed).
// root_out (optional) receives the raw Merkle root.
extern "C" __global__ void __launch_bounds__(512)
b3_tree_stage2(const u32* __restrict__ in_cvs, int count, const u32* __restrict__ key,
               const u32* __restrict__ prefix, const u32* __restrict__ salt, u32 dim,
               u32* __restrict__ root_out, u32* __restrict__ seed_out) {
  __shared__ u32 cvs[1024 * 8];
  for (int i = threadIdx.x; i < count * 8; i += blockDim.x) cvs[i] = in_cvs[i];
  __syncthreads();
  u32 k8[8];
#pragma unroll
  for (int i = 0; i < 8; i++) k8[i] = key[i];
  if (count > 1) b3_reduce_smem(cvs, count, k8, F_KEYED, true);
  if (threadIdx.x == 0) {
    u32 root[8];
#pragma unroll
    for (int i = 0; i < 8; i++) root[i] = cvs[i];
    if (root_out) {
#pragma unroll
      for (int i = 0; i < 8; i++) root_out[i] = root[i];
    }
    if (seed_out) {
      u32 msg[16], bound[8], s8[8], seed[8];
#pragma unroll
      for (int i = 0; i < 8; i++) { msg[i] = root[i]; msg[8 + i] = 0; s8[i] = salt[i]; }
      msg[8] = dim;
      b3_hash64(s8, msg, bound);
#pragma unroll
      for (int i = 0; i < 8; i++) { msg[i] = prefix[i]; msg[8 + i] = bound[i]; }
      b3_hash64(nullptr, msg, seed);
#pragma unroll
      for (int i = 0; i < 8; i++) seed_out[i] = seed[i];
    }
  }
}

// ------------------------------------------------------------------ noise generation
// get_random_hash(index, label, key, prepend): msg[prepend] = 1 + index, msg[8..16] = label.
__device__ __forceinline__ void random_hash(u32 index, const u32* label, const u32* key, int prepend, u32 out[8]) {
  u32 msg[16];
#pragma unroll
  for (int i = 0; i < 8; i++) { msg[i] = 0; msg[8 + i] = label[i]; }
  msg[prepend] = 1u + index;
  u32 k8[8];
#pragma unroll
  for (int i = 0; i < 8; i++) k8[i] = key[i];
  b3_hash64(k8, msg, out);
}

// rows x r int8 in [-32, 31]; one thread per 32-byte hash block. Thread 0 of block 0 also
// zeroes `zero_me` (zero_n ints) so per-call counters start clean without a memset launch.
extern "C" __global__ void noise_uniform(i8* __restrict__ out, u32 nblocks, const u32* __restrict__ label,
                                         const u32* __restrict__ key, i32* __restrict__ zero_me, int zero_n) {
  u32 b = blockIdx.x * blockDim.x + threadIdx.x;
  if (zero_me && b < (u32)zero_n) zero_me[b] = 0;
  if (b >= nblocks) return;
  u32 h[8];
  random_hash(b, label, key, 0, h);
  u32 packed[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    u32 w = h[i], o = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      int v = (int)((w >> (8 * j)) & 63u) - 32;
      o |= ((u32)(u8)(i8)v) << (8 * j);
    }
    packed[i] = o;
  }
  uint4* dst = reinterpret_cast<uint4*>(out + (u64)b * 32);
  dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
  dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
}

// k pairs packed as (first | second << 16); one thread per hash (8 pairs).
extern "C" __global__ void noise_perm(u32* __restrict__ out, u32 k, u32 r, const u32* __restrict__ label,
                                      const u32* __restrict__ key) {
  u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i * 8 >= k) return;
  u32 h[8];
  random_hash(i, label, key, 1, h);
#pragma unroll
  for (int j = 0; j < 8; j++) {
    u32 l = i * 8 + j;
    if (l < k) {
      u32 u = h[j];
      u32 first = u & (r - 1);
      u32 second = first ^ (1u + __umulhi(r - 1, u));
      out[l] = first | (second << 16);
    }
  }
}

// ------------------------------------------------------------------ row kernel
// For each row of src (rows x k int8), optionally:
//   dst[l]  = src[l] + E[row][Pn.first(l)] - E[row][Pn.second(l)]            (noising)
//   proj[j] = sum_l x[l] * ([Pp.first(l)==j] - [Pp.second(l)==j]),  x = src  (rank-r projection)
// then writes proj in `lmax` balanced base-128 int8 limbs to corr[li][row][corr_off + j]
// (row stride 2r) and copies E rows to corr[li][row][ecopy_off + j]. maxabs = max |proj|.
#define NR_ROWS 8
extern "C" __global__ void __launch_bounds__(256)
noise_rows(const i8* __restrict__ src, int rows, int k, int r,
           const i8* __restrict__ E, const u32* __restrict__ Pn, i8* __restrict__ dst,
           const u32* __restrict__ Pp, i8* __restrict__ corr, int corr_off, int ecopy_off,
           i32* __restrict__ maxabs, int lmax) {
  extern __shared__ __align__(16) u8 smem_nr[];
  i8* Es = reinterpret_cast<i8*>(smem_nr);                         // NR_ROWS * r
  i32* proj = reinterpret_cast<i32*>(smem_nr + NR_ROWS * r);       // NR_ROWS * r
  int row0 = blockIdx.x * NR_ROWS;
  int tid = threadIdx.x;
  if (E) for (int i = tid; i < NR_ROWS * r; i += 256) Es[i] = E[(u64)row0 * r + i];
  if (Pp) for (int i = tid; i < NR_ROWS * r; i += 256) proj[i] = 0;
  __syncthreads();
  int quads = k >> 2;
  int total = NR_ROWS * quads;
  for (int it = tid; it < total; it += 256) {
    int rr = it / quads, q = it - rr * quads;
    u64 off = (u64)(row0 + rr) * k + 4 * q;
    u32 w = *reinterpret_cast<const u32*>(src + off);
    u32 o = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      int l = 4 * q + j;
      int x = (int)(i8)(u8)(w >> (8 * j));
      if (dst) {
        u32 pn = __ldg(Pn + l);
        int nv = x + (int)Es[rr * r + (pn & 0xFFFF)] - (int)Es[rr * r + (pn >> 16)];
        o |= ((u32)(u8)(i8)nv) << (8 * j);
      }
      if (Pp) {
        u32 pp = __ldg(Pp + l);
        atomicAdd(&proj[rr * r + (pp & 0xFFFF)], x);
        atomicSub(&proj[rr * r + (pp >> 16)], x);
      }
    }
    if (dst) *reinterpret_cast<u32*>(dst + off) = o;
  }
  __syncthreads();
  u64 limb_stride = (u64)rows * 2 * r;
  int local_max = 0;
  for (int i = tid; i < NR_ROWS * r; i += 256) {
    int rr = i / r, j = i - rr * r;
    u64 base = (u64)(row0 + rr) * 2 * r;
    if (Pp && corr_off >= 0) {
      int v = proj[i];
      local_max = max(local_max, abs(v));
      for (int li = 0; li < lmax; li++) {
        int d = ((v + 64) & 127) - 64;
        corr[li * limb_stride + base + corr_off + j] = (i8)d;
        v = (v - d) >> 7;
      }
    }
    if (E && ecopy_off >= 0) {
      i8 e = Es[i];
      for (int li = 0; li < lmax; li++) corr[li * limb_stride + base + ecopy_off + j] = e;
    }
  }
  if (Pp && maxabs) {
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) local_max = max(local_max, __shfl_xor_sync(0xffffffffu, local_max, s));
    if ((tid & 31) == 0) atomicMax(maxabs, local_max);
  }
}

// ------------------------------------------------------------------ main GEMM
__device__ __forceinline__ u32 smem_u32addr(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(u32 dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm_x4(u32& r0, u32& r1, u32& r2, u32& r3, u32 addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_s8(i32* d, const u32* a, const u32* b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// 64-byte rows, 16-byte chunks XOR-swizzled so ldmatrix over 8 consecutive rows is conflict-free.
__device__ __forceinline__ int swz(int row, int chunk) { return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4); }

__device__ __forceinline__ void load_stage(u8* sA, u8* sB, const i8* __restrict__ A, const i8* __restrict__ B,
                                           int bm0, int bn0, int k, int kt, int tid) {
#pragma unroll
  for (int i = 0; i < (BM * BK / 16) / NTHREADS; i++) {
    int idx = tid + i * NTHREADS, row = idx >> 2, c = idx & 3;
    cp_async16(smem_u32addr(sA + swz(row, c)), A + (u64)(bm0 + row) * k + kt * BK + c * 16);
  }
#pragma unroll
  for (int i = 0; i < (BN * BK / 16) / NTHREADS; i++) {
    int idx = tid + i * NTHREADS, row = idx >> 2, c = idx & 3;
    cp_async16(smem_u32addr(sB + swz(row, c)), B + (u64)(bn0 + row) * k + kt * BK + c * 16);
  }
}

__device__ __forceinline__ u32 bf16x2(float a, float b) {
  u32 ua = __float_as_uint(a), ub = __float_as_uint(b);
  ua = (ua + 0x7FFFu + ((ua >> 16) & 1u)) >> 16;
  ub = (ub + 0x7FFFu + ((ub >> 16) & 1u)) >> 16;
  return ua | (ub << 16);
}

#define CSTAGES 2
#define CBK 128
#define CORR_STAGE ((BM + WARPS_N * 16) * CBK)
// 128-byte rows, 16-byte chunks XOR-swizzled by row (conflict-free ldmatrix over 8 rows).
__device__ __forceinline__ int swz128(int row, int chunk) { return row * 128 + ((chunk ^ (row & 7)) << 4); }
// One pipeline step of the denoise phase: A-side [X_li | EAL] rows bm0..bm0+128 and the
// B-side [EBRt | Y_li] rows for this pass's column strip (16 cols per warp column), k-chunk kc.
__device__ __forceinline__ void load_corr(u8* st, const i8* __restrict__ corrA, const i8* __restrict__ corrB,
                                          u64 limbA, u64 limbB, int R2, int L, int KC, int gstep,
                                          int bm0, int bn0, int tid) {
  const int S = L * KC, p = gstep / S, step = gstep - p * S;
  int li = L - 1 - step / KC, kc = step - (step / KC) * KC;
  u8* sA = st;
  u8* sB = st + BM * CBK;
  const i8* pa = corrA + li * limbA + (u64)bm0 * R2 + kc * CBK;
#pragma unroll
  for (int i = 0; i < (BM * CBK / 16) / NTHREADS; i++) {
    int idx = tid + i * NTHREADS, row = idx >> 3, c = idx & 7;
    cp_async16(smem_u32addr(sA + swz128(row, c)), pa + (u64)row * R2 + c * 16);
  }
#pragma unroll
  for (int i = 0; i < (WARPS_N * 16 * CBK / 16) / NTHREADS; i++) {
    int idx = tid + i * NTHREADS, q = idx >> 3, c = idx & 7;
    int col = bn0 + (q >> 4) * 64 + p * 16 + (q & 15);
    cp_async16(smem_u32addr(sB + swz128(q, c)), corrB + li * limbB + (u64)col * R2 + kc * CBK + c * 16);
  }
}

#define FLAG_HASH 1
#define FLAG_DENOISE 2
#define OUT_NONE 0
#define OUT_I32 1
#define OUT_BF16 2
#define WIN_WORDS 10

// C = A * B^T with A: m x k, B: n x k (both int8, k contiguous). With FLAG_HASH, A and B are
// the noised operands and every thread hashes its 8 x 16 tile; with FLAG_DENOISE the
// epilogue subtracts sum_li 128^li [X_li | EAL] [EBRt | Y_li]^T exactly in int32.
extern "C" __global__ void __launch_bounds__(NTHREADS, MINB)
noisy_gemm(const i8* __restrict__ A, const i8* __restrict__ B, int m, int n, int k, int r,
           const i8* __restrict__ corrA, const i8* __restrict__ corrB, const i32* __restrict__ maxabs,
           const u32* __restrict__ key, const u32* __restrict__ target,
           u32* __restrict__ wins, int max_wins,
           const float* __restrict__ a_scale, const float* __restrict__ b_scale,
           void* __restrict__ out, int out_mode, int flags, int lmax) {
  extern __shared__ __align__(128) u8 smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp / WARPS_N, wn = warp % WARPS_N, g = lane >> 2, t4 = lane & 3;
  // grouped rasterization: GROUP_M row-tiles share B tiles in L2
  int bm0, bn0;
  {
    const int npm = m / BM, npn = n / BN, pid = blockIdx.x;
    const int grp = GROUP_M * npn, gid = pid / grp, first = gid * GROUP_M;
    const int gsz = min(npm - first, GROUP_M), in = pid - gid * grp;
    bm0 = (first + in % gsz) * BM;
    bn0 = (in / gsz) * BN;
  }
  const int KT = k / BK, tiles_per_chunk = r / BK;
  const bool do_hash = flags & FLAG_HASH;

  i32 acc[4][8][4];
#pragma unroll
  for (int a = 0; a < 4; a++)
#pragma unroll
    for (int b = 0; b < 8; b++)
#pragma unroll
      for (int c = 0; c < 4; c++) acc[a][b][c] = 0;
  u32 msg[16];
#pragma unroll
  for (int i = 0; i < 16; i++) msg[i] = 0;

#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < KT) load_stage(smem + s * STAGE_BYTES, smem + s * STAGE_BYTES + A_STAGE, A, B, bm0, bn0, k, s, tid);
    cp_async_commit();
  }

  for (int kt = 0; kt < KT; kt++) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    {
      int nk = kt + STAGES - 1;
      if (nk < KT) {
        int s = nk % STAGES;
        load_stage(smem + s * STAGE_BYTES, smem + s * STAGE_BYTES + A_STAGE, A, B, bm0, bn0, k, nk, tid);
      }
      cp_async_commit();
    }
    const u8* sA = smem + (kt % STAGES) * STAGE_BYTES;
    const u8* sB = sA + A_STAGE;
#pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      u32 af[4][4], bfr[8][2];
#pragma unroll
      for (int mi = 0; mi < 4; mi++) {
        int row = wm * 64 + mi * 16 + (lane & 15);
        int c = ks * 2 + (lane >> 4);
        ldsm_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3], smem_u32addr(sA + swz(row, c)));
      }
#pragma unroll
      for (int nj2 = 0; nj2 < 4; nj2++) {
        int row = wn * 64 + nj2 * 16 + ((lane >> 4) << 3) + (lane & 7);
        int c = ks * 2 + ((lane >> 3) & 1);
        ldsm_x4(bfr[2 * nj2][0], bfr[2 * nj2][1], bfr[2 * nj2 + 1][0], bfr[2 * nj2 + 1][1],
                smem_u32addr(sB + swz(row, c)));
      }
#pragma unroll
      for (int mi = 0; mi < 4; mi++)
#pragma unroll
        for (int nj = 0; nj < 8; nj++) mma_s8(acc[mi][nj], af[mi], bfr[nj]);
    }
    if (do_hash && ((kt + 1) % tiles_per_chunk) == 0) {
      u32 x = 0;
#pragma unroll
      for (int a = 0; a < 4; a++)
#pragma unroll
        for (int b = 0; b < 8; b++)
#pragma unroll
          for (int c = 0; c < 4; c++) x ^= (u32)acc[a][b][c];
      int slot = ((kt + 1) / tiles_per_chunk - 1) & 15;
#pragma unroll
      for (int w = 0; w < 16; w++)
        if (w == slot) msg[w] = __funnelshift_l(msg[w], msg[w], 13) ^ x;
    }
  }
  cp_async_wait<0>();

  if (do_hash) {
    u32 k8[8], h[8];
#pragma unroll
    for (int i = 0; i < 8; i++) k8[i] = key[i];
    b3_hash64(k8, msg, h);
    // little-endian U256 compare: hash <= target
    bool le = true, decided = false;
#pragma unroll
    for (int i = 7; i >= 0; i--) {
      u32 t = target[i];
      if (!decided && h[i] != t) { le = h[i] < t; decided = true; }
    }
    if (le) {
      u32 slot = atomicAdd(wins, 1u);
      if (slot < (u32)max_wins) {
        u32* rec = wins + 1 + slot * WIN_WORDS;
        rec[0] = bm0 + wm * 64 + g;
        rec[1] = bn0 + wn * 64 + 2 * t4;
#pragma unroll
        for (int i = 0; i < 8; i++) rec[2 + i] = h[i];
      }
    }
  }

  if (flags & FLAG_DENOISE) {
    // acc -= sum_li 128^li [X_li | EAL] [EBRt | Y_li]^T, exact in int32 (Horner over limbs).
    // Four passes over quarter-width column strips keep the temporary accumulator at 32
    // registers; operands stream through smem with the same cp.async + ldmatrix path.
    int L = 1;
    const int mx = max(maxabs[0], maxabs[1]);
    if (mx > 63) L = 2;
    if (mx > 8127) L = 3;
    if (mx > 1040319) L = 4;
    L = min(L, lmax);  // host sizes lmax for the worst case 127*k, so this never truncates
    const int R2 = 2 * r, KC = R2 / CBK, S = L * KC, TOT = 4 * S;
    const u64 limbA = (u64)m * R2, limbB = (u64)n * R2;
    __syncthreads();
#pragma unroll
    for (int s = 0; s < CSTAGES - 1; s++) {
      if (s < TOT) load_corr(smem + s * CORR_STAGE, corrA, corrB, limbA, limbB, R2, L, KC, s, bm0, bn0, tid);
      cp_async_commit();
    }
#pragma unroll
    for (int p = 0; p < 4; p++) {
      i32 tmp[4][2][4];
#pragma unroll
      for (int a = 0; a < 4; a++)
#pragma unroll
        for (int b = 0; b < 2; b++)
#pragma unroll
          for (int c = 0; c < 4; c++) tmp[a][b][c] = 0;
      for (int step = 0; step < S; step++) {
        const int gs = p * S + step;
        cp_async_wait<CSTAGES - 2>();
        __syncthreads();
        {
          int nx = gs + CSTAGES - 1;
          if (nx < TOT) load_corr(smem + (nx % CSTAGES) * CORR_STAGE, corrA, corrB, limbA, limbB, R2, L, KC, nx, bm0, bn0, tid);
          cp_async_commit();
        }
        if (step % KC == 0) {
#pragma unroll
          for (int a = 0; a < 4; a++)
#pragma unroll
            for (int b = 0; b < 2; b++)
#pragma unroll
              for (int c = 0; c < 4; c++) tmp[a][b][c] <<= 7;
        }
        const u8* sA = smem + (gs % CSTAGES) * CORR_STAGE;
        const u8* sB = sA + BM * CBK;
#pragma unroll
        for (int ks = 0; ks < CBK / 32; ks++) {
          u32 af[4][4], b0[2], b1[2];
#pragma unroll
          for (int mi = 0; mi < 4; mi++) {
            int row = wm * 64 + mi * 16 + (lane & 15);
            int c = ks * 2 + (lane >> 4);
            ldsm_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3], smem_u32addr(sA + swz128(row, c)));
          }
          {
            int row = wn * 16 + ((lane >> 4) << 3) + (lane & 7);
            int c = ks * 2 + ((lane >> 3) & 1);
            ldsm_x4(b0[0], b0[1], b1[0], b1[1], smem_u32addr(sB + swz128(row, c)));
          }
#pragma unroll
          for (int mi = 0; mi < 4; mi++) {
            mma_s8(tmp[mi][0], af[mi], b0);
            mma_s8(tmp[mi][1], af[mi], b1);
          }
        }
      }
#pragma unroll
      for (int a = 0; a < 4; a++)
#pragma unroll
        for (int b = 0; b < 2; b++)
#pragma unroll
          for (int c = 0; c < 4; c++) acc[a][2 * p + b][c] -= tmp[a][b][c];
    }
    cp_async_wait<0>();
  }

  if (out_mode == OUT_I32) {
#pragma unroll
    for (int mi = 0; mi < 4; mi++) {
      const int r0 = bm0 + wm * 64 + mi * 16 + g;
#pragma unroll
      for (int nj = 0; nj < 8; nj++) {
        const int c0 = bn0 + wn * 64 + nj * 8 + 2 * t4;
        i32* o = reinterpret_cast<i32*>(out);
        *reinterpret_cast<int2*>(o + (u64)r0 * n + c0) = make_int2(acc[mi][nj][0], acc[mi][nj][1]);
        *reinterpret_cast<int2*>(o + (u64)(r0 + 8) * n + c0) = make_int2(acc[mi][nj][2], acc[mi][nj][3]);
      }
    }
  } else if (out_mode == OUT_BF16) {
    // Stage the scaled bf16 tile in smem (padded rows), then write full 16-byte vectors.
    constexpr int PITCH = BN * 2 + 16;  // bytes
    __syncthreads();
#pragma unroll
    for (int mi = 0; mi < 4; mi++) {
      const int lr = wm * 64 + mi * 16 + g;
      const float s0 = a_scale[bm0 + lr], s1 = a_scale[bm0 + lr + 8];
#pragma unroll
      for (int nj = 0; nj < 8; nj++) {
        const int lc = wn * 64 + nj * 8 + 2 * t4;
        const float q0 = b_scale[bn0 + lc], q1 = b_scale[bn0 + lc + 1];
        *reinterpret_cast<u32*>(smem + lr * PITCH + lc * 2) =
            bf16x2((float)acc[mi][nj][0] * s0 * q0, (float)acc[mi][nj][1] * s0 * q1);
        *reinterpret_cast<u32*>(smem + (lr + 8) * PITCH + lc * 2) =
            bf16x2((float)acc[mi][nj][2] * s1 * q0, (float)acc[mi][nj][3] * s1 * q1);
      }
    }
    __syncthreads();
    constexpr int VPR = BN * 2 / 16;  // 16-byte vectors per row
    u8* o = reinterpret_cast<u8*>(out);
#pragma unroll 4
    for (int i = tid; i < BM * VPR; i += NTHREADS) {
      const int row = i / VPR, v = i - row * VPR;
      *reinterpret_cast<uint4*>(o + ((u64)(bm0 + row) * n + bn0) * 2 + v * 16) =
          *reinterpret_cast<const uint4*>(smem + row * PITCH + v * 16);
    }
  }
}

// ------------------------------------------------------------------ fused activation quantizer
// One CTA per row: bf16 -> float in smem -> optional block fast Walsh-Hadamard transform
// (block size bs, power of two dividing k, orthonormal) -> per-row absmax -> int8 in [-64, 64].
// q = rint(y / s), s = max|y| / 64, y = unnormalized transform (round-half-even). The 1/sqrt(bs)
// normalization does not change q, so it is folded into the returned (dequant) scale instead:
// every step that decides q is an IEEE add/max/divide in a fixed order -> identical on any GPU/CPU.
extern "C" __global__ void __launch_bounds__(256)
quant_rows_had(const unsigned short* __restrict__ x, int k, int bs, i8* __restrict__ q, float* __restrict__ scale) {
  extern __shared__ float rowf[];
  __shared__ float red[8];
  const int r = blockIdx.x, tid = threadIdx.x;
  const unsigned short* xr = x + (u64)r * k;
  for (int i = tid * 8; i < k; i += 256 * 8) {
    uint4 v = *reinterpret_cast<const uint4*>(xr + i);
    u32 w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < 4; j++) {
      rowf[i + 2 * j] = __uint_as_float(w[j] << 16);
      rowf[i + 2 * j + 1] = __uint_as_float(w[j] & 0xFFFF0000u);
    }
  }
  __syncthreads();
  float norm = 1.0f;
  if (bs > 1) {
    for (int lh = 0; (1 << lh) < bs; lh++) {
      const int h = 1 << lh;
      for (int p = tid; p < (k >> 1); p += 256) {
        int i = ((p >> lh) << (lh + 1)) | (p & (h - 1)), j = i + h;
        float a = rowf[i], b = rowf[j];
        rowf[i] = a + b;
        rowf[j] = a - b;
      }
      __syncthreads();
    }
    norm = 1.0f / sqrtf((float)bs);  // only used for the dequant scale
  }
  float mx = 0.f;
  for (int i = tid; i < k; i += 256) mx = fmaxf(mx, fabsf(rowf[i]));
#pragma unroll
  for (int s = 16; s > 0; s >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, s));
  if ((tid & 31) == 0) red[tid >> 5] = mx;
  __syncthreads();
  if (tid < 32) {
    mx = tid < 8 ? red[tid] : 0.f;
#pragma unroll
    for (int s = 4; s > 0; s >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, s));
    if (tid == 0) red[0] = fmaxf(mx, 1e-8f) / 64.0f;
  }
  __syncthreads();
  const float sc = red[0];
  if (tid == 0) scale[r] = sc * norm;
  i8* qr = q + (u64)r * k;
  for (int i = tid * 4; i < k; i += 256 * 4) {
    u32 o = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float v = fminf(fmaxf(rintf(__fdiv_rn(rowf[i + j], sc)), -64.f), 64.f);
      o |= ((u32)(u8)(i8)(int)v) << (8 * j);
    }
    *reinterpret_cast<u32*>(qr + i) = o;
  }
}
