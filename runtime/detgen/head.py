"""The head stage of a generation job: from the last layer's residual rows to the next token.

A generation job's sequence b is its prompt (length P, from the job's inputs) followed by
generated tokens up to the job's seq_len T. The layer stages run over the whole sequence as in an
embedding job (teacher forcing: identical to token-by-token decoding, tests/test_detrt.py). The head
stage adds, for every position t in [P-1, T-2] (the positions whose output is a generated token):

  HN   final RMSNorm of the last layer's R2 row                         D float32
  HQ   lm_head activation quantization of HN (Hadamard, int8)          D int32 + scale bits
  HC   one leaf per vocabulary chunk c (VCHUNK tokens): the best token
       in the chunk and its score                                        [token, score bits]
  HS   the sample: the first best over the chunks                        [token, score bits]

score(v) = logit(v)                                   temperature 0 (greedy)
         = logit(v) / temperature + gumbel(seed, t, v) otherwise (Gumbel-max sampling)
logit(v) = fp32((sum_k q[k] * W[v,k]) * s_x) * s_w[v]  as every DetLinear; ties go to the lowest
token. The token at position t+1 must equal HS(t).token. Every leaf is recomputable from its
inputs, so a dispute narrows to one HC leaf (VCHUNK dot products) or one HS leaf (a max over the
chunk leaves).

Gumbel noise is counter-based integer hashing, so the referee can reproduce it exactly:
  x  = mix64(key ^ mix64(t << 32 | v)),  key = first 8 bytes (big-endian) of sha256("uwc-gumbel" || seed)
  u  = (2 * (x >> 41) + 1) * 2^-24                   an odd multiple of 2^-24: exact in fp32, in (0, 1)
  g  = -det_log(-det_log(u))
mix64 is the splitmix64 finalizer (unsigned 64-bit arithmetic, wrapping).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from detrt import core
from detrt.core import F32, c32, fadd, fdiv, fmul
from detrt.qwen3 import DetQwen3
from detrt.stage import Leaf, StageTrace, _np

VCHUNK = 1024
HN, HQ, HC, HS = 100, 101, 102, 103  # head op codes (layer field = -1); detrt.stage uses 0..25
M64 = (1 << 64) - 1


@dataclass(frozen=True)
class Sampling:
    temp_milli: int = 0          # temperature in thousandths; 0: greedy
    seed: bytes = b"\x00" * 32

    def key(self) -> int:
        return int.from_bytes(hashlib.sha256(b"uwc-gumbel" + self.seed).digest()[:8], "big")

    def temperature(self, like: torch.Tensor) -> torch.Tensor:
        return fdiv(c32(float(self.temp_milli), like), c32(1000.0, like))


def mix64(x: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer on uint64 arrays (numpy's unsigned arithmetic wraps)."""
    x = x ^ (x >> np.uint64(30))
    x = x * np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27))
    x = x * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def uniform(s: Sampling, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """u(t, v) in (0, 1) as float32 (exact): t (n, 1) positions, v (1, m) token ids."""
    with np.errstate(over="ignore"):
        x = mix64(np.uint64(s.key()) ^ mix64((t.astype(np.uint64) << np.uint64(32)) | v.astype(np.uint64)))
    num = (x >> np.uint64(41)) * np.uint64(2) + np.uint64(1)  # odd, < 2^24
    return (num.astype(np.float64) * 2.0 ** -24).astype(np.float32)


def gumbel(s: Sampling, t: torch.Tensor, v0: int, v1: int, device) -> torch.Tensor:
    """Gumbel noise (n, v1 - v0) for positions t (n,) and tokens v0..v1-1, float32, deterministic."""
    u = uniform(s, t.cpu().numpy()[:, None], np.arange(v0, v1)[None, :])
    u = torch.from_numpy(u).to(device)
    return -core.det_log(-core.det_log(u))


def logits(m: DetQwen3, hq: torch.Tensor, sx: torch.Tensor, v0: int, v1: int) -> torch.Tensor:
    """Logits of tokens [v0, v1) for quantized rows hq (n, D) int8 with scales sx (n,)."""
    lm = m.lm_head
    if hq.device.type == "cpu":
        acc = hq.to(torch.float64) @ lm.Wq64[v0:v1].t()
    else:  # torch._int_mm wants >16 rows and a multiple of 8 columns: pad both, slice the result
        n, w = hq.shape[0], v1 - v0
        mp, wp = max(32, -(-n // 8) * 8), -(-w // 8) * 8
        qp = torch.nn.functional.pad(hq, (0, 0, 0, mp - n)) if mp != n else hq
        Wc = lm.Wq[v0:v1]
        Wc = torch.nn.functional.pad(Wc, (0, 0, 0, wp - w)) if wp != w else Wc
        acc = torch._int_mm(qp.contiguous(), Wc.t().contiguous())[:n, :w]
    return fmul(fmul(acc.to(F32), sx[:, None]), lm.sw[None, v0:v1])


def scores(m: DetQwen3, hq, sx, t: torch.Tensor, s: Sampling, v0: int, v1: int) -> torch.Tensor:
    lg = logits(m, hq, sx, v0, v1)
    if s.temp_milli == 0:
        return lg
    return fadd(fdiv(lg, s.temperature(lg)), gumbel(s, t, v0, v1, lg.device))


def head_rows(m: DetQwen3, r2: torch.Tensor, t: torch.Tensor, s: Sampling, vchunk: int = VCHUNK):
    """For final-layer residual rows r2 (n, D) at positions t (n,): HN, HQ (q, sx), the chunk bests
    (n, chunks) tokens and scores, and the sample (n,) tokens and scores."""
    return sample_normed(m, core.det_rmsnorm(r2.to(F32), m.norm, m.eps), t, s, vchunk)


def sample_normed(m: DetQwen3, hn: torch.Tensor, t: torch.Tensor, s: Sampling, vchunk: int = VCHUNK):
    """head_rows from final-norm rows hn (DetQwen3.forward returns exactly these)."""
    hq, sx = core.quant_act(hn, m.lm_head.bs)
    V = m.lm_head.n
    best_tok, best_sc = [], []
    for v0 in range(0, V, vchunk):
        v1 = min(V, v0 + vchunk)
        sc = scores(m, hq, sx, t, s, v0, v1)
        i = sc.argmax(dim=-1)  # first maximum: the lowest token on ties
        best_tok.append(i + v0)
        best_sc.append(sc.gather(1, i[:, None])[:, 0])
    bt, bs = torch.stack(best_tok, 1), torch.stack(best_sc, 1)  # (n, chunks)
    c = bs.argmax(dim=-1)
    tok = bt.gather(1, c[:, None])[:, 0]
    sco = bs.gather(1, c[:, None])[:, 0]
    return hn, hq, sx, bt, bs, tok, sco


def head_program(P: int, T: int, chunks: int) -> list[Leaf]:
    out = []
    for t in range(P - 1, T - 1):
        out += [Leaf(HN, -1, t, -1), Leaf(HQ, -1, t, -1)] + [Leaf(HC, -1, t, c) for c in range(chunks)] + [Leaf(HS, -1, t, -1)]
    return out


@torch.no_grad()
def run_head_stage(m: DetQwen3, r2: torch.Tensor, P: int, s: Sampling, vchunk: int = VCHUNK,
                   tokens: torch.Tensor | None = None) -> StageTrace:
    """r2: (T, D) the last layer's R2 rows of the whole sequence. Returns the head trace; its
    `output` is the sampled token for each position P-1..T-2 (int32). If tokens (T,) are given,
    trace.mismatch is the first position t whose token t+1 differs from the sample, else -1."""
    T = r2.shape[0]
    dev = m.device
    t = torch.arange(P - 1, T - 1, device=dev)
    hn, hq, sx, bt, bs, tok, sco = head_rows(m, r2[P - 1:T - 1].to(dev), t, s, vchunk)
    chunks = bt.shape[1]
    rows = []
    as_i32 = lambda x: _np(x).view(np.int32)
    for i in range(t.shape[0]):
        rows.append(_np(hn[i]))
        rows.append(np.concatenate([_np(hq[i].to(torch.int32)), as_i32(sx[i:i + 1])]))
        for c in range(chunks):
            rows.append(np.array([int(bt[i, c]), int(as_i32(bs[i, c:c + 1])[0])], dtype=np.int32))
        rows.append(np.array([int(tok[i]), int(as_i32(sco[i:i + 1])[0])], dtype=np.int32))
    tr = StageTrace(leaves=head_program(P, T, chunks), rows=rows, output=_np(tok.to(torch.int32)))
    tr.finalize()
    tr.mismatch = -1
    if tokens is not None:
        want = tokens[P:T].to(dev).to(torch.int64)
        bad = (want != tok).nonzero()
        tr.mismatch = int(t[bad[0, 0]]) if bad.numel() else -1
    return tr


def head_weight_rows(m: DetQwen3):
    """Rows a generation model's weights root adds after detrt.stage.weight_rows: the final norm,
    then lm_head as linear rows (int8 weights widened to int32, then the row's fp32 scale bits)."""
    yield "norm", m.norm.cpu().numpy().astype(np.float32)
    W = m.lm_head.Wq.cpu().numpy().astype(np.int32)
    sw = m.lm_head.sw.cpu().numpy().astype(np.float32).view(np.int32)
    for r in range(W.shape[0]):
        yield "lm_head", np.concatenate([W[r], sw[r:r + 1]])
