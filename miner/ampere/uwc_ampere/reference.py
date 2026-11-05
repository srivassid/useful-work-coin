"""Pure-Python/torch reference of Pearl's cuPOW mining rules.

Ported line by line from Pearl's Rust (ext/pearl/zk-pow/src):
  ffi/mine.rs            try_mine_one, compute_job_key, compute_commitment_hash_with_offsets
  circuit/pearl_noise.rs get_random_hash, generate_uniform_random_matrix,
                         generate_permutation_matrix, compute_noise_for_indices
  api/seed.rs            bind_roots (Salted derivation, cert v3)
  api/proof_utils.rs     compute_jackpot_hash

This file is the oracle for the GPU kernels. It favours clarity over speed, except
`jackpot_all_tiles`, which uses cuBLAS int8 matmuls so it can check full-size GEMMs.
"""
from __future__ import annotations

import numpy as np
import torch
from blake3 import blake3

JACKPOT_SIZE = 16
LROT_PER_TILE = 13
CHUNK_LEN = 1024

SEED_LABEL_A = b"A_tensor" + b"\x00" * 24
SEED_LABEL_B = b"B_tensor" + b"\x00" * 24
SEED_SALT_A = blake3(b"pearl/cert-v3/noise-seed/A").digest()
SEED_SALT_B = blake3(b"pearl/cert-v3/noise-seed/B").digest()

# NOISE_RANGE = 128, IDXS_PER_COL = 2 -> uniform values in [-32, 31]
_RANGE_MASK = 63
_ZERO_POINT = 32


def keyed(key: bytes, data: bytes) -> bytes:
    return blake3(data, key=key).digest()


def plain(data: bytes) -> bytes:
    return blake3(data).digest()


def pad_to_chunk(data: bytes) -> bytes:
    """pearl-blake3 padded_chunk_len: round up to a multiple of 1024."""
    return data + b"\x00" * (-len(data) % CHUNK_LEN)


def bind_root(root: bytes, dim: int, salt: bytes) -> bytes:
    return keyed(salt, root + dim.to_bytes(4, "little") + b"\x00" * 28)


def commitment(job_key: bytes, A: np.ndarray, Bt: np.ndarray, salted: bool = True):
    """Return (a_noise_seed, b_noise_seed, hash_a_root, hash_b_root).

    A is m x k int8 row-major; Bt is n x k int8 (B column-major), as in Pearl.
    """
    m, n = A.shape[0], Bt.shape[0]
    ha = keyed(job_key, pad_to_chunk(np.ascontiguousarray(A, dtype=np.int8).tobytes()))
    hb = keyed(job_key, pad_to_chunk(np.ascontiguousarray(Bt, dtype=np.int8).tobytes()))
    if salted:
        ha_b, hb_b = bind_root(ha, m, SEED_SALT_A), bind_root(hb, n, SEED_SALT_B)
    else:
        ha_b, hb_b = ha, hb
    b_seed = plain(job_key + hb_b)
    a_seed = plain(b_seed + ha_b)
    return a_seed, b_seed, ha, hb


def random_hash(index: int, label: bytes, key: bytes, prepend: int) -> bytes:
    msg = bytearray(64)
    msg[prepend * 4 : prepend * 4 + 4] = (1 + index).to_bytes(4, "little", signed=True)
    msg[32:64] = label
    return keyed(key, bytes(msg))


def uniform_matrix(label: bytes, key: bytes, rows: int, r: int) -> np.ndarray:
    """rows x r int8 in [-32, 31] (E_AL, or E_BR transposed)."""
    assert (rows * r) % 32 == 0
    raw = b"".join(random_hash(b, label, key, 0) for b in range((rows * r) // 32))
    vals = (np.frombuffer(raw, dtype=np.uint8) & _RANGE_MASK).astype(np.int16) - _ZERO_POINT
    return vals.astype(np.int8).reshape(rows, r)


def perm_matrix(label: bytes, key: bytes, k: int, r: int) -> np.ndarray:
    """k x 2 uint32: row l has +1 at [l,0] and -1 at [l,1] (E_AR^T, or E_BL)."""
    out = np.zeros((k, 2), dtype=np.uint32)
    for i in range((k + 7) // 8):
        u = np.frombuffer(random_hash(i, label, key, 1), dtype="<u4")
        first = u & np.uint32(r - 1)
        mulhi = ((np.uint64(r - 1) * u.astype(np.uint64)) >> np.uint64(32)).astype(np.uint32)
        second = first ^ (np.uint32(1) + mulhi)
        lo, hi = i * 8, min(k, i * 8 + 8)
        out[lo:hi, 0] = first[: hi - lo]
        out[lo:hi, 1] = second[: hi - lo]
    return out


def noise(k: int, r: int, a_seed: bytes, b_seed: bytes, m: int, n: int):
    """Full noise (Pearl computes only needed rows; same values). Returns dict of int8/uint32 arrays."""
    EAL = uniform_matrix(SEED_LABEL_A, a_seed, m, r)
    PA = perm_matrix(SEED_LABEL_A, a_seed, k, r)
    PB = perm_matrix(SEED_LABEL_B, b_seed, k, r)
    EBRt = uniform_matrix(SEED_LABEL_B, b_seed, n, r)
    noise_a = (EAL[:, PA[:, 0]].astype(np.int16) - EAL[:, PA[:, 1]]).astype(np.int8)  # m x k
    noise_bt = (EBRt[:, PB[:, 0]].astype(np.int16) - EBRt[:, PB[:, 1]]).astype(np.int8)  # n x k
    return dict(EAL=EAL, PA=PA, PB=PB, EBRt=EBRt, noise_a=noise_a, noise_bt=noise_bt)


def rotl(x: int, s: int) -> int:
    return ((x << s) | (x >> (32 - s))) & 0xFFFFFFFF


def jackpot_msg_tile(Ap: np.ndarray, Bp: np.ndarray, r: int) -> list[int]:
    """Ap: h x k int (noised A rows), Bp: w x k int (noised B^T rows). Mirrors try_mine_one."""
    k = Ap.shape[1]
    acc = np.zeros((Ap.shape[0], Bp.shape[0]), dtype=np.int64)
    msg = [0] * JACKPOT_SIZE
    for c in range(k // r):
        acc += Ap[:, c * r : (c + 1) * r].astype(np.int64) @ Bp[:, c * r : (c + 1) * r].astype(np.int64).T
        x = int(np.bitwise_xor.reduce((acc & 0xFFFFFFFF).astype(np.uint32).ravel()))
        t = c % JACKPOT_SIZE
        msg[t] = rotl(msg[t], LROT_PER_TILE) ^ x
    return msg


def jackpot_hash(msg: list[int], a_seed: bytes) -> bytes:
    return keyed(a_seed, b"".join(int(w).to_bytes(4, "little") for w in msg))


def pattern_list(shape) -> list[int]:
    """PeriodicPattern.shape [(stride, length)]*3 -> sorted offsets (Rust to_list)."""
    out = [0]
    for stride, length in shape:
        if length == 0:
            continue
        out = [o + i * stride for o in out for i in range(length)]
    return sorted(out)


def jackpot_all_tiles(ApEA: torch.Tensor, BpEBt: torch.Tensor, r: int, h_off: list[int], w_off: list[int],
                      tile_rows: int, tile_cols: int) -> torch.Tensor:
    """GPU reference for every hash tile at once, using cuBLAS int8 matmul per r-chunk.

    The hash tile pattern must be a regular grid inside each (tile_rows x tile_cols) block,
    where `h_off`/`w_off` are the pattern offsets and tiles are generated by offsets
    (o_r, o_c) with o_r % period in [0, rows_stride) etc. Here we support exactly the
    patterns used by the Ampere kernel: rows = g + 8*i (i<8) inside a 64-row block,
    cols = 2t + {0,1} + 8*j (j<8) inside a 64-col block.
    Returns msg words as int64 tensor [m/64*8, n/64*4, 16].
    """
    m, k = ApEA.shape
    n = BpEBt.shape[0]
    assert tile_rows == 64 and tile_cols == 64
    acc = torch.zeros((m, n), dtype=torch.int32, device=ApEA.device)
    msgs = torch.zeros((m // 64, 8, n // 64, 4, JACKPOT_SIZE), dtype=torch.int64, device=ApEA.device)
    for c in range(k // r):
        acc += torch._int_mm(ApEA[:, c * r : (c + 1) * r].contiguous(), BpEBt[:, c * r : (c + 1) * r].contiguous().t())
        # rows: block rb, g, i  -> row = rb*64 + 8*i + g ; cols: block cb, t, j, e -> col = cb*64 + 8*j + 2*t + e
        a = acc.view(m // 64, 8, 8, n // 64, 8, 4, 2)  # rb, i, g, cb, j, t, e
        a = a.permute(0, 2, 3, 5, 1, 4, 6).reshape(m // 64, 8, n // 64, 4, 128).to(torch.int64) & 0xFFFFFFFF
        x = a[..., 0].clone()
        for q in range(1, 128):
            x ^= a[..., q]
        t = c % JACKPOT_SIZE
        old = msgs[..., t]
        msgs[..., t] = (((old << LROT_PER_TILE) | (old >> (32 - LROT_PER_TILE))) & 0xFFFFFFFF) ^ x
    return msgs
