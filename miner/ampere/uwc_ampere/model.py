"""Integer-only linear layers for Hugging Face Qwen3 models, mined with cuPOW on Ampere.

Quantization (the "{-64..64}" constraint of the design doc):
  weights      per-output-channel symmetric, W' = W H, s_w = max|W'_row| / 64
  activations  per-token symmetric (dynamic),  x' = x H, s_x = max|x'_row| / 64
  y = (q(x') q(W')^T) * s_x s_w  (+ bias)
H is an optional block-diagonal Walsh-Hadamard rotation (orthonormal), as in Pearl's
vLLM miner and QuaRot, which spreads activation outliers before rounding.

Modes per layer:
  "bf16"  original weights (baseline)
  "int"   the same integer math through cuBLAS int8 (no mining) -> quality eval
  "plain" the same integer math through the Ampere GEMM kernel with no mining at all
          (what a provider who does not mine would run)
  "mined" MinedWeight.forward: commit + noise + hashed GEMM + exact denoise.
          Output is bit-identical to "int" (tests/test_model.py checks this).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ops

QMAX = 64


def hadamard(n: int, device, dtype=torch.float32) -> torch.Tensor:
    assert n & (n - 1) == 0
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(n)).to(device=device, dtype=dtype)


def rotate(x: torch.Tensor, H: torch.Tensor | None) -> torch.Tensor:
    if H is None:
        return x
    bs = H.shape[0]
    shp = x.shape
    return (x.reshape(*shp[:-1], shp[-1] // bs, bs).to(H.dtype) @ H).reshape(shp)


def quant_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row quantization to int8 values in [-64, 64]."""
    xf = x.float()
    s = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / QMAX
    q = torch.round(xf / s).clamp_(-QMAX, QMAX).to(torch.int8)
    return q, s.squeeze(-1)


class MinedLinear(nn.Module):
    """Replacement for one or more nn.Linear layers that share an input (fused along n)."""

    def __init__(self, linears: list[nn.Linear], device, mode: str = "mined", hadamard_block: int = 512,
                 r: int = 128, cfg: str | None = None):
        super().__init__()
        self.mode = mode
        self.splits = [l.out_features for l in linears]
        W = torch.cat([l.weight.detach() for l in linears], 0)  # stays where it was loaded (CPU)
        bias = [l.bias for l in linears]
        self.bias = None if all(b is None for b in bias) else torch.cat(
            [b.detach() if b is not None else torch.zeros(l.out_features) for b, l in zip(bias, linears)]).to(device, torch.bfloat16)
        self.n, self.k = W.shape
        self.H = None
        if hadamard_block:
            bs = min(hadamard_block, self.k & -self.k)
            self.H = hadamard(bs, device)
        if mode == "bf16":
            self.W = W.to(device=device, dtype=torch.bfloat16)
        else:
            # quantize in row chunks on the GPU so huge layers (lm_head) never need a full fp32 copy
            qs, ss = [], []
            for s0 in range(0, self.n, 16384):
                q, sc = quant_rows(rotate(W[s0:s0 + 16384].to(device=device, dtype=torch.float32), self.H))
                qs.append(q.cpu() if mode == "mined" else q)
                ss.append(sc)
            self.Wq, self.sw = torch.cat(qs).contiguous(), torch.cat(ss).contiguous()
            if mode == "mined":  # only the noised copy will live on the GPU (made in setup_block)
                self.Wq = self.Wq.pin_memory()
        del W
        self.r = r
        self.cfg = cfg or ops.DEFAULT_CFG
        self.bn = ops.CONFIGS[self.cfg][0]
        if mode == "mined" and not self.mineable:  # falls back to cuBLAS int8, so weights must be on the GPU
            self.Wq = self.Wq.to(device)
        self.mw: ops.MinedWeight | None = None
        self.hashed_macs = 0   # m_pad * n * k summed over mined calls
        self.useful_macs = 0   # m * n * k summed over calls

    @property
    def mineable(self) -> bool:
        k, n, r = self.k, self.n, self.r
        return (k >= max(1024, 16 * r) and k % r == 0 and k % 64 == 0 and k <= 4 * r * r and n % self.bn == 0)

    def setup_block(self, job_key_for_k, bound_for_k):
        """job_key_for_k(k) -> 32-byte job key; bound_for_k(k) -> 8 target words."""
        if self.mode != "mined" or not self.mineable:
            return
        if self.mw is None:
            self.mw = ops.MinedWeight(Bt=self.Wq, b_scale=self.sw, r=self.r)
        self.mw.setup_block(job_key_for_k(self.k), bound_for_k(self.k))

    def _int_forward(self, xq, sx):
        y = torch._int_mm(xq, self.Wq.t()).float() * sx[:, None] * self.sw[None, :]
        return y.to(torch.bfloat16)

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x2 = x.reshape(-1, self.k)
        m = x2.shape[0]
        if self.mode == "bf16":
            y = F.linear(x2.to(torch.bfloat16), self.W)
        else:
            bs = self.H.shape[0] if self.H is not None else 0
            kernel_path = self.mode in ("mined", "plain") and self.mineable
            mp = -(-m // ops.BM) * ops.BM if kernel_path else m
            xb = x2.to(torch.bfloat16)
            if mp != m:
                xb = F.pad(xb, (0, 0, 0, mp - m))
            xq, sx = ops.quant_rows_had(xb.contiguous(), bs)  # fused rotate + quantize
            if self.mode == "mined" and self.mineable:
                y = self.mw.forward(xq.contiguous(), sx.contiguous())[:m]
                self.hashed_macs += mp * self.n * self.k
            elif self.mode == "plain" and self.mineable:
                y = torch.empty((mp, self.n), dtype=torch.bfloat16, device=xq.device)
                ops.gemm(xq, self.Wq, a_scale=sx, b_scale=self.sw, out=y, out_mode=ops.OUT_BF16, cfg=self.cfg)
                y = y[:m]
            else:
                if m < 17:  # cuBLAS int8 needs m > 16
                    y = (xq.float() @ self.Wq.float().t() * sx[:, None] * self.sw[None, :]).to(torch.bfloat16)
                else:
                    y = self._int_forward(xq.contiguous(), sx)
        self.useful_macs += m * self.n * self.k
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shp[:-1], self.n)


class _Slice(nn.Module):
    """k_proj / v_proj / up_proj stand-in: returns its slice of the fused output computed by the head layer."""

    def __init__(self, head: "_FusedHead", idx: int):
        super().__init__()
        self.head, self.idx = head, idx

    @torch.compiler.disable
    def forward(self, x):
        assert self.head.last_input is x, "fused layers must be called with the same input tensor, in order"
        y = self.head.outputs[self.idx]
        if self.idx == len(self.head.outputs) - 1:  # last consumer: drop the references
            self.head.outputs = self.head.last_input = None
        return y


class _FusedHead(nn.Module):
    def __init__(self, ml: MinedLinear):
        super().__init__()
        self.ml = ml
        self.last_input = None
        self.outputs = None

    @torch.compiler.disable
    def forward(self, x):
        y = self.ml(x)
        self.last_input = x
        self.outputs = torch.split(y, self.ml.splits, dim=-1)
        return self.outputs[0]


def convert_qwen3(model, device, mode="mined", hadamard_block=512, mine_lm_head=True, cfg=None) -> list[MinedLinear]:
    """Swap every linear in a HF Qwen3 model (q/k/v fused, gate/up fused). Returns the MinedLinears."""
    mls = []
    base = model.model if hasattr(model, "model") else model  # CausalLM or bare Qwen3Model (embeddings)
    for layer in base.layers:
        at, mlp = layer.self_attn, layer.mlp
        qkv = MinedLinear([at.q_proj, at.k_proj, at.v_proj], device, mode, hadamard_block, cfg=cfg)
        head = _FusedHead(qkv)
        at.q_proj, at.k_proj, at.v_proj = head, _Slice(head, 1), _Slice(head, 2)
        o = MinedLinear([at.o_proj], device, mode, hadamard_block, cfg=cfg)
        at.o_proj = o
        gu = MinedLinear([mlp.gate_proj, mlp.up_proj], device, mode, hadamard_block, cfg=cfg)
        head2 = _FusedHead(gu)
        mlp.gate_proj, mlp.up_proj = head2, _Slice(head2, 1)
        d = MinedLinear([mlp.down_proj], device, mode, hadamard_block, cfg=cfg)
        mlp.down_proj = d
        mls += [qkv, o, gu, d]
        torch.cuda.empty_cache()
    if mine_lm_head and hasattr(model, "lm_head"):
        lm = MinedLinear([model.lm_head], device, mode, hadamard_block, cfg=cfg)
        model.lm_head = lm
        mls.append(lm)
    return mls
