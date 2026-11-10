"""The head stage of a generation job: from the last layer's residual rows to the next token.

A generation job's batch b is one sequence of the job's seq_len T. Its input row (the job's input
tree) holds the prompt tokens, then SENTINEL (0xFFFFFFFF) for every generated position: the
prompt length P is the index of the first sentinel. The provider fills the generated positions;
the layer stages run over the whole sequence as in an embedding job (teacher forcing, identical
to token-by-token decoding: tests/test_detrt.py), and each batch gets one more stage after its
layer groups, the head stage, with these leaves for every position t in [0, T-2]:

  HN  (t)      final RMSNorm of the last layer's R2 row (the stage input)        D float32
  HQ  (t)      lm_head activation quantization of HN (Hadamard, int8)            D int32 + scale bits
  HL  (t, c)   the scores of vocabulary chunk c (VCHUNK tokens; the last one shorter)   float32
  HC  (t, c)   the chunk's best: [token, score bits]
  HS  (t)      the sample, the first best over the chunks: [token, score bits]   (the stage outputs)

score(v) = logit(v)                                    temperature 0 (greedy)
         = logit(v) / temperature + gumbel(seed, t, v)  otherwise (Gumbel-max sampling; seed = job id)
logit(v) = fp32(fp32(sum_k q[k] W[v,k]) * s_x) * s_w[v], as every linear layer; ties go to the
lowest token. Token t+1 of the sequence, for t+1 >= P, is HS(t).token: layer stage 0 reads it
from the head stage's output rows. Every leaf is recomputable from small inputs: an HL element
from one lm_head row, an HC row from one HL row, an HS row from the chunk bests.

Gumbel noise is counter-based integer hashing, so the referee can reproduce it exactly:
  x  = mix64(key ^ mix64(t << 32 | v)),  key = first 8 bytes (big-endian) of sha256("uwc-gumbel" || seed)
  u  = (2 * (x >> 41) + 1) * 2^-24                   an odd multiple of 2^-24: exact in fp32, in (0, 1)
  g  = -det_log(-det_log(u))
mix64 is the splitmix64 finalizer (unsigned 64-bit arithmetic, wrapping).

A generation job's output root (TRACE_COMMIT) is MerkleRoot over batches of
  sha256(0x00 || "gout" || u32 b || head output root of batch b)
so a client checks the delivered samples against the commitment.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np
import torch

from detrt import core
from detrt.core import F32, c32, fadd, fdiv, fmul
from detrt.qwen3 import DetQwen3
from detrt.stage import Leaf, StageTrace, _leaf_hash, _np, merkle_root, row_root

VCHUNK = 1024
SENTINEL = 0xFFFFFFFF
HN, HQ, HL, HC, HS = 100, 101, 102, 103, 104  # head op codes; detrt.stage uses 0..25


@dataclass(frozen=True)
class Sampling:
    temp_milli: int = 0          # temperature in thousandths; 0: greedy
    seed: bytes = b"\x00" * 32   # the job id

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


def chunks(V: int, vchunk: int = VCHUNK) -> list[tuple[int, int]]:
    return [(v0, min(V, v0 + vchunk)) for v0 in range(0, V, vchunk)]


def logits(m: DetQwen3, hq: torch.Tensor, sx: torch.Tensor, v0: int, v1: int) -> torch.Tensor:
    """Logits of tokens [v0, v1) for quantized rows hq (n, D) int8 with scales sx (n,)."""
    lm = m.lm_head
    if hq.device.type == "cpu":
        acc = hq.to(torch.float64) @ lm.Wq64[v0:v1].t()
    else:  # torch._int_mm wants >16 rows and a multiple of 8 columns: pad both, slice the result.
        # The weights go in as a column-major view (as DetLinear does): on sm_86 cuBLASLt refuses a
        # row-major second operand unless the rows are a multiple of 32. Exact either way.
        n, w = hq.shape[0], v1 - v0
        mp, wp = max(32, -(-n // 8) * 8), -(-w // 8) * 8
        qp = torch.nn.functional.pad(hq, (0, 0, 0, mp - n)) if mp != n else hq
        Wc = lm.Wq[v0:v1]
        Wc = torch.nn.functional.pad(Wc, (0, 0, 0, wp - w)) if wp != w else Wc
        acc = torch._int_mm(qp.contiguous(), Wc.contiguous().t())[:n, :w]
    return fmul(fmul(acc.to(F32), sx[:, None]), lm.sw[None, v0:v1])


def scores(m: DetQwen3, hq, sx, t: torch.Tensor, s: Sampling, v0: int, v1: int) -> torch.Tensor:
    lg = logits(m, hq, sx, v0, v1)
    if s.temp_milli == 0:
        return lg
    return fadd(fdiv(lg, s.temperature(lg)), gumbel(s, t, v0, v1, lg.device))


def sample_normed(m: DetQwen3, hn: torch.Tensor, t: torch.Tensor, s: Sampling, vchunk: int = VCHUNK,
                  keep_scores: bool = False):
    """From final-norm rows hn (n, D) at positions t (n,): HQ (q, sx), the chunk scores (if
    keep_scores), the chunk bests (n, chunks) tokens and scores, and the samples (n,)."""
    hq, sx = core.quant_act(hn, m.lm_head.bs)
    best_tok, best_sc, kept = [], [], []
    for v0, v1 in chunks(m.lm_head.n, vchunk):
        sc = scores(m, hq, sx, t, s, v0, v1)
        if keep_scores:
            kept.append(sc)
        i = sc.argmax(dim=-1)  # the first maximum: the lowest token on ties
        best_tok.append(i + v0)
        best_sc.append(sc.gather(1, i[:, None])[:, 0])
    bt, bs = torch.stack(best_tok, 1), torch.stack(best_sc, 1)
    c = bs.argmax(dim=-1)
    tok = bt.gather(1, c[:, None])[:, 0]
    sco = bs.gather(1, c[:, None])[:, 0]
    return hq, sx, kept, bt, bs, tok, sco


def head_program(L: int, T: int, nchunks: int) -> list[Leaf]:
    out = []
    for t in range(T - 1):
        out += [Leaf(HN, L, t, -1), Leaf(HQ, L, t, -1)]
        for c in range(nchunks):
            out += [Leaf(HL, L, t, c), Leaf(HC, L, t, c)]
        out.append(Leaf(HS, L, t, -1))
    return out


def _w(x: torch.Tensor) -> int:
    """float32 scalar tensor -> its bits as a signed int32 (row words)."""
    return int(_np(x.reshape(1)).view(np.int32)[0])


@torch.no_grad()
def run_head_stage(m: DetQwen3, r2: torch.Tensor, s: Sampling, vchunk: int = VCHUNK, rows: bool = True) -> StageTrace:
    """r2: (T, D) the last layer's R2 rows of the whole sequence. The trace's output rows are the
    samples [token, score bits] for positions 0..T-2 (the stage's output root commits them);
    with rows=False only the outputs are kept (what providers and verifiers need)."""
    T = r2.shape[0]
    dev = m.device
    t = torch.arange(T - 1, device=dev)
    hn = core.det_rmsnorm(r2[:T - 1].to(dev, F32), m.norm, m.eps)
    hq, sx, kept, bt, bs, tok, sco = sample_normed(m, hn, t, s, vchunk, keep_scores=rows)
    out = np.stack([_np(tok.to(torch.int32)), _np(sco).view(np.int32)], axis=1).astype(np.int32)  # (T-1, 2)
    tr = StageTrace(leaves=head_program(len(m.layers), T, bt.shape[1]) if rows else [], rows=[], output=out)
    if rows:
        for i in range(T - 1):
            tr.rows.append(_np(hn[i]))
            tr.rows.append(np.concatenate([_np(hq[i].to(torch.int32)), _np(sx[i:i + 1]).view(np.int32)]))
            for c in range(bt.shape[1]):
                tr.rows.append(_np(kept[c][i]))
                tr.rows.append(np.array([int(bt[i, c]), _w(bs[i, c])], dtype=np.int32))
            tr.rows.append(out[i])
        tr.finalize()
    return tr


def head_output_root(out: np.ndarray) -> bytes:
    """The head stage's committed root: its output rows (samples) as stage-output leaves."""
    return merkle_root([_leaf_hash(b"sout" + struct.pack("<I", t) + row_root(np.ascontiguousarray(r)))
                        for t, r in enumerate(out)])


def gen_out_leaves(head_roots: list[bytes]) -> list[bytes]:
    return [_leaf_hash(b"gout" + struct.pack("<I", b) + r) for b, r in enumerate(head_roots)]


def gen_output_root(head_roots: list[bytes]) -> bytes:
    """A generation job's output root over its batches' head output roots."""
    return merkle_root(gen_out_leaves(head_roots))


def input_row(prompt: list[int], T: int) -> list[int]:
    """A generation batch's input row: the prompt, then SENTINEL up to seq_len T."""
    assert 0 < len(prompt) < T
    return list(prompt) + [SENTINEL] * (T - len(prompt))


def sequence(prompt: list[int], out: np.ndarray) -> list[int]:
    """The full sequence: the prompt, then the samples of positions P-1..T-2."""
    return list(prompt) + [int(x) for x in out[len(prompt) - 1:, 0]]


def mismatch(seq: list[int], out: np.ndarray, P: int) -> int:
    """The first position t >= P-1 whose sample differs from token t+1 of seq, or -1."""
    for t in range(P - 1, len(seq) - 1):
        if int(out[t, 0]) != seq[t + 1]:
            return t
    return -1


def head_weight_rows(m: DetQwen3):
    """Rows a generation model's weights tree adds after detrt.stage.weight_rows: the final norm,
    then lm_head as linear rows (int8 weights widened to int32, then the row's fp32 scale bits)."""
    yield "norm", m.norm.cpu().numpy().astype(np.float32)
    W = m.lm_head.Wq.cpu().numpy().astype(np.int32)
    sw = m.lm_head.sw.cpu().numpy().astype(np.float32).view(np.int32)
    for r in range(W.shape[0]):
        yield "lm_head", np.concatenate([W[r], sw[r:r + 1]])
