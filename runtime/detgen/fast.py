"""Fast generation on the GPU, bit-identical to detgen.generate.generate and detgen.head:

  prefill   detgpu's fused layer kernels over each prompt (all its positions at once), with the
            quantized K and V written to a per-sequence cache
  decode    one row per active sequence: the same per-row kernels (rope at each row's own
            position) and attention_cache against that sequence's cache
  sample    final norm, lm_head over the whole vocabulary, scores (greedy: the logits; else
            gumbel_scores), the first maximum: the same token and score as the head stage's
            chunk-by-chunk HC / HS leaves

Sequences of different prompt lengths decode in one batch: each row carries its position and
its sequence, and a sequence leaves the batch when it reaches seq_len. Needs head_dim 128 (the
fused kernels); other models use the reference generator.
"""
from __future__ import annotations

import numpy as np
import torch

from detrt import core
from detrt.core import F32
from detrt.qwen3 import DetQwen3

from . import kernels as GK
from .head import Sampling


class FastGen:
    def __init__(self, m: DetQwen3, B: int, S: int):
        assert m.hd == 128 and m.device.type == "cuda", "fast generation needs head_dim 128 on a GPU"
        self.m, self.B, self.S = m, B, S
        dev = m.device
        z = lambda *s, dt=torch.int8: torch.zeros(*s, dtype=dt, device=dev)
        self.cache = [dict(k=z(B, S, m.Hkv, 128), ks=z(B, S, m.Hkv, dt=F32), v=z(B, S, m.Hkv, 128), vs=z(B, S, m.Hkv, dt=F32))
                      for _ in m.layers]

    # ------------------------------------------------------------ layers
    def _qkv(self, L, c, x, bidx, pos):
        """Quantized q, k, v of rows x at positions pos (sequences bidx); k, v go to the cache."""
        from detgpu import kernels as K
        m = self.m
        q1, s1 = K.rmsnorm_quant(x, L["in_norm"], m.eps, L["qkv"].bs)
        N = x.shape[0]  # rope per row: T = N rows, cos/sin gathered at each row's position
        qq, qs, kq, ks, vq, vs = K.qkv_post(L["qkv"].int_gemm(q1), s1, L["qkv"].sw, L["q_norm"], L["k_norm"],
                                            m.cos[pos].contiguous(), m.sin[pos].contiguous(), N, m.H, m.Hkv, m.eps)
        c["k"][bidx, pos], c["ks"][bidx, pos] = kq, ks[..., 0]
        c["v"][bidx, pos], c["vs"][bidx, pos] = vq, vs[..., 0]
        return qq, qs, kq, ks, vq, vs

    def _rest(self, L, x, oc):
        from detgpu import kernels as K
        q2, s2 = K.quant_act(oc, L["o"].bs)
        r1 = K.dequant_residual(L["o"].int_gemm(q2), s2, L["o"].sw, x)
        q3, s3 = K.rmsnorm_quant(r1, L["post_norm"], self.m.eps, L["gu"].bs)
        q4, s4 = K.swiglu_quant(L["gu"].int_gemm(q3), s3, L["gu"].sw, L["down"].bs)
        return K.dequant_residual(L["down"].int_gemm(q4), s4, L["down"].sw, r1)

    @torch.no_grad()
    def prefill(self, b: int, prompt: list[int]) -> torch.Tensor:
        """Run sequence b's prompt; returns its final-norm row at position P-1 (1, D)."""
        from detgpu import kernels as K
        m, dev = self.m, self.m.device
        P = len(prompt)
        x = m.embed[torch.tensor(prompt, device=dev)]
        bidx = torch.full((P,), b, device=dev, dtype=torch.long)
        pos = torch.arange(P, device=dev)
        for L, c in zip(m.layers, self.cache):
            qq, qs, kq, ks, vq, vs = self._qkv(L, c, x, bidx, pos)
            # the prompt attends within itself: the full-sequence kernel (the cache holds the same k, v)
            oc = K.attention(qq, qs, kq, ks, vq, vs, 1, P, m.H, m.Hkv, m.inv_sqrt_hd)
            x = self._rest(L, x, oc)
        return core.det_rmsnorm(x[-1:], m.norm, m.eps)

    @torch.no_grad()
    def step(self, bidx: torch.Tensor, toks: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """One token per row (sequence bidx at position pos); returns final-norm rows (R, D)."""
        m = self.m
        x = m.embed[toks]
        for L, c in zip(m.layers, self.cache):
            qq, qs, *_ = self._qkv(L, c, x, bidx, pos)
            oc = GK.attention_cache(qq, qs, c["k"], c["ks"], c["v"], c["vs"], pos, bidx, m.H, m.Hkv, m.inv_sqrt_hd)
            x = self._rest(L, x, oc)
        return core.det_rmsnorm(x, m.norm, m.eps)


def sample(m: DetQwen3, hn: torch.Tensor, pos: torch.Tensor, s: Sampling) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokens and scores (float32) for final-norm rows hn (R, D) at positions pos (R,)."""
    from detgpu import kernels as K
    lm = m.lm_head
    q, sx = K.quant_act(hn.contiguous(), lm.bs)
    lg = lm.int_gemm(q).to(F32) * sx[:, None] * lm.sw[None, :]
    if s.temp_milli:
        temp = float(s.temperature(lg[:1, :1]).item())
        lg = GK.gumbel_scores(lg, pos, s.key(), temp)
    tok = lg.argmax(dim=-1)  # the first maximum, as the head stage
    return tok, lg.gather(1, tok[:, None])[:, 0]


@torch.no_grad()
def generate(m: DetQwen3, prompts: list[list[int]], T: int, s: Sampling) -> list[list[int]]:
    """Every sequence: its prompt followed by generated tokens up to length T."""
    dev = m.device
    g = FastGen(m, len(prompts), T)
    seqs = [list(p) for p in prompts]
    hn = torch.cat([g.prefill(b, p) for b, p in enumerate(prompts)])
    pos = torch.tensor([len(p) - 1 for p in prompts], device=dev)
    bidx = torch.arange(len(prompts), device=dev)
    while bidx.numel():
        tok, _ = sample(m, hn, pos, s)
        for b, t in zip(bidx.tolist(), tok.tolist()):
            seqs[b].append(t)
        pos = pos + 1
        keep = pos < T - 1  # a sequence whose last token is written needs no further step
        bidx, pos, tok, hn = bidx[keep], pos[keep], tok[keep], hn[keep]
        if bidx.numel():
            hn = g.step(bidx, tok, pos)
    return seqs


@torch.no_grad()
def head_outputs(m: DetQwen3, r2: torch.Tensor, s: Sampling, rows: int = 256) -> np.ndarray:
    """The head stage's output rows [token, score bits] (B, T-1, 2) int32 from the last layer's R2
    rows r2 (B, T, D) on the GPU (bit-identical to detgen.head.run_head_stage(rows=False))."""
    B, T, D = r2.shape
    flat = core.det_rmsnorm(r2[:, :T - 1].reshape(-1, D).to(F32), m.norm, m.eps)
    pos = torch.arange(T - 1, device=r2.device).repeat(B)
    toks, scs = [], []
    for i in range(0, flat.shape[0], rows):
        t, sc = sample(m, flat[i:i + rows], pos[i:i + rows], s)
        toks.append(t)
        scs.append(sc)
    tok = torch.cat(toks).to(torch.int32).cpu().numpy()
    sco = torch.cat(scs).cpu().numpy().astype(np.float32).view(np.int32)
    return np.stack([tok, sco], axis=1).reshape(B, T - 1, 2)
