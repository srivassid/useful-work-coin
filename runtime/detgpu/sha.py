"""Stage output commitments on the GPU, identical to agent.proto.stage_out_leaves / stage_output_root.

A stage is committed by the Merkle root (RFC 6962 style, detrt.stage.merkle_root) of one leaf
per token, sha256(0x00 || "sout" || u32 t || row_root), where row_root is the Merkle root of
sha256(0x00 || block) over the row's float32 bytes in 1024-byte blocks. Hashing on the GPU means
only 32 bytes per token cross the PCIe bus instead of the whole row (x1 risers move ~0.4 GB/s;
a provider's outputs are ~0.6 GB/s at detgpu's speed).

The per-row tree is built bottom-up, pairing neighbours and promoting an odd last node, which
equals merkle_root's recursive split (checked for 1..299 leaves).
"""
from __future__ import annotations

import cupy as cp
import numpy as np
import torch

from detrt.stage import BLOCK, merkle_root

_SRC = r"""
typedef unsigned int u32;
typedef unsigned char u8;
__constant__ u32 K[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

__device__ __forceinline__ u32 rotr(u32 x, int n) { return (x >> n) | (x << (32 - n)); }

__device__ void compress(u32* st, const u32* m) {
  u32 w[64];
  for (int i = 0; i < 16; i++) w[i] = m[i];
  for (int i = 16; i < 64; i++) {
    u32 s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
    u32 s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
    w[i] = w[i-16] + s0 + w[i-7] + s1;
  }
  u32 a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
  for (int i = 0; i < 64; i++) {
    u32 S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25), ch = (e & f) ^ (~e & g);
    u32 t1 = h + S1 + ch + K[i] + w[i];
    u32 S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22), mj = (a & b) ^ (a & c) ^ (b & c);
    u32 t2 = S0 + mj;
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

__device__ __forceinline__ void init(u32* st) {
  st[0] = 0x6a09e667; st[1] = 0xbb67ae85; st[2] = 0x3c6ef372; st[3] = 0xa54ff53a;
  st[4] = 0x510e527f; st[5] = 0x9b05688c; st[6] = 0x1f83d9ab; st[7] = 0x5be0cd19;
}

// sha256 of prefix byte || data[0..len) (len < 2^28), data read byte by byte from global memory
__device__ void sha_prefixed(u8 prefix, const u8* data, int len, u32* out) {
  u32 st[8]; init(st);
  const long long total = 1 + (long long)len;
  const int nblk = (int)((total + 8) / 64) + 1;
  u32 m[16];
  for (int bk = 0; bk < nblk; bk++) {
    for (int wi = 0; wi < 16; wi++) {
      u32 v = 0;
      for (int k = 0; k < 4; k++) {
        long long p = (long long)bk * 64 + wi * 4 + k;   // message byte index
        u32 byte;
        if (p == 0) byte = prefix;
        else if (p < total) byte = data[p - 1];
        else if (p == total) byte = 0x80;
        else byte = 0;
        v = (v << 8) | byte;
      }
      m[wi] = v;
    }
    if (bk == nblk - 1) { m[14] = (u32)((total * 8) >> 32); m[15] = (u32)(total * 8); }
    compress(st, m);
  }
  for (int i = 0; i < 8; i++) out[i] = st[i];
}

// sha256(0x01 || l || r) for two 32-byte digests held as 8 big-endian words each
__device__ void sha_node(const u32* l, const u32* r, u32* out) {
  u32 st[8]; init(st);
  u32 m[16];
  // bytes: 0x01, l[0..32), r[0..32), 0x80, zeros, length 65*8
  u8 buf[128];
  buf[0] = 1;
  for (int i = 0; i < 8; i++) for (int k = 0; k < 4; k++) { buf[1 + 4*i + k] = (u8)(l[i] >> (24 - 8*k)); buf[33 + 4*i + k] = (u8)(r[i] >> (24 - 8*k)); }
  buf[65] = 0x80;
  for (int i = 66; i < 120; i++) buf[i] = 0;
  const unsigned long long bits = 65ull * 8;
  for (int k = 0; k < 8; k++) buf[120 + k] = (u8)(bits >> (56 - 8*k));
  for (int bk = 0; bk < 2; bk++) {
    for (int wi = 0; wi < 16; wi++) { int p = bk * 64 + wi * 4; m[wi] = ((u32)buf[p] << 24) | ((u32)buf[p+1] << 16) | ((u32)buf[p+2] << 8) | buf[p+3]; }
    compress(st, m);
  }
  for (int i = 0; i < 8; i++) out[i] = st[i];
}

// one thread per (row, block): leaf hash sha256(0x00 || block bytes)
extern "C" __global__ void block_leaves(const u8* __restrict__ rows, int row_bytes, int nblk, long long nrows, u32* __restrict__ leaves) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nrows * nblk) return;
  const long long r = i / nblk; const int bk = (int)(i % nblk);
  const int off = bk * BLOCKB, len = min(BLOCKB, row_bytes - off);
  sha_prefixed(0, rows + r * row_bytes + off, len, leaves + i * 8);
}

// one thread per row: row_root over its nblk block leaves (bottom-up, odd node promoted), then
// the stage-output leaf sha256(0x00 || "sout" || u32le t || row_root), t = row % T
extern "C" __global__ void sout_leaves(const u32* __restrict__ leaves, int nblk, long long nrows, int T, u32* __restrict__ out) {
  long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= nrows) return;
  u32 lv[MAXBLK][8];
  for (int i = 0; i < nblk; i++) for (int k = 0; k < 8; k++) lv[i][k] = leaves[(r * nblk + i) * 8 + k];
  int n = nblk;
  while (n > 1) {
    int m = 0;
    for (int i = 0; i + 1 < n; i += 2) { u32 h[8]; sha_node(lv[i], lv[i + 1], h); for (int k = 0; k < 8; k++) lv[m][k] = h[k]; m++; }
    if (n & 1) { for (int k = 0; k < 8; k++) lv[m][k] = lv[n - 1][k]; m++; }
    n = m;
  }
  u8 msg[40];
  msg[0] = 's'; msg[1] = 'o'; msg[2] = 'u'; msg[3] = 't';
  const u32 t = (u32)(r % T);
  msg[4] = (u8)t; msg[5] = (u8)(t >> 8); msg[6] = (u8)(t >> 16); msg[7] = (u8)(t >> 24);
  for (int i = 0; i < 8; i++) for (int k = 0; k < 4; k++) msg[8 + 4*i + k] = (u8)(lv[0][i] >> (24 - 8*k));
  sha_prefixed(0, msg, 40, out + r * 8);
}
"""

MAXBLK = 32
_MODS: dict = {}


def _fn(name: str):
    dev = torch.cuda.current_device()
    mod = _MODS.get(dev)
    if mod is None:
        major, minor = torch.cuda.get_device_capability()
        src = f"#define BLOCKB {BLOCK}\n#define MAXBLK {MAXBLK}\n" + _SRC
        with cp.cuda.Device(dev):
            mod = _MODS[dev] = cp.RawModule(code=src, options=(f"-arch=sm_{major}{minor}", "-std=c++17"))
    return mod.get_function(name)


def _stream() -> cp.cuda.Stream:
    from .kernels import _stream as s
    return s()


def sout_leaves(out: torch.Tensor) -> np.ndarray:
    """Stage-output leaves of B stages at once: out (B, T, D) float32 on the GPU -> (B, T, 32) uint8
    on the CPU (the digests, big-endian as sha256 writes them)."""
    out = out.contiguous()
    B, T, D = out.shape
    nrows, row_bytes = B * T, D * 4
    nblk = max(1, -(-row_bytes // BLOCK))
    assert nblk <= MAXBLK
    leaves = torch.empty((nrows * nblk, 8), dtype=torch.int32, device=out.device)
    res = torch.empty((nrows, 8), dtype=torch.int32, device=out.device)
    with cp.cuda.Device(out.device.index):
        n = nrows * nblk
        _fn("block_leaves")((-(-n // 128),), (128,), (np.uint64(out.data_ptr()), np.int32(row_bytes), np.int32(nblk),
                                                      np.int64(nrows), np.uint64(leaves.data_ptr())), stream=_stream())
        _fn("sout_leaves")((-(-nrows // 64),), (64,), (np.uint64(leaves.data_ptr()), np.int32(nblk), np.int64(nrows),
                                                       np.int32(T), np.uint64(res.data_ptr())), stream=_stream())
    words = res.cpu().numpy().view(np.uint32)
    return words.byteswap().view(np.uint8).reshape(B, T, 32)  # big-endian words -> digest bytes


def stage_output_roots(out: torch.Tensor) -> list[bytes]:
    """agent.proto.stage_output_root for each of the B stages in out (B, T, D)."""
    lv = sout_leaves(out)
    return [merkle_root([bytes(lv[b, t]) for t in range(lv.shape[1])]) for b in range(lv.shape[0])]
