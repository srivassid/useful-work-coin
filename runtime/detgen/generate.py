"""Reference generation: prefill the prompt, then decode one token at a time with the KV cache,
choosing each token with the head stage's own sampling (detgen.head.sample_normed). The result
equals the head stage recomputed over the whole sequence by teacher forcing (tests/test_detgen.py).
"""
from __future__ import annotations

import torch

from detrt.qwen3 import DetQwen3

from .head import VCHUNK, Sampling, sample_normed


@torch.no_grad()
def generate(m: DetQwen3, prompt: list[int], T: int, s: Sampling, vchunk: int = VCHUNK) -> list[int]:
    """The job sequence: prompt followed by generated tokens up to length T (no early stop: a
    client cuts at its stop token)."""
    P = len(prompt)
    assert 0 < P < T
    cache = m.new_cache(1, T)
    hn = m.forward(torch.tensor([prompt], device=m.device), cache, 0)[0, -1:]  # final-norm row at P-1
    seq = list(prompt)
    for pos in range(P - 1, T - 1):
        tok = int(sample_normed(m, hn, torch.tensor([pos], device=hn.device), s, vchunk)[5][0])
        seq.append(tok)
        if pos + 1 < T - 1:
            hn = m.forward(torch.tensor([[tok]], device=m.device), cache, pos + 1)[0]
    return seq
