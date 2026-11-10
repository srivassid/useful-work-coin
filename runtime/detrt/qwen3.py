"""Deterministic Qwen3 reference runtime: bit-identical on every GPU and on CPU, and identical
between full prefill, token-by-token decode with a KV cache, and teacher forcing.

Architectures (ModelCfg.Arch): 0 Qwen3 (RMSNorm on each q and k head); 1 Llama / Mistral (no q/k
norm: the QN and KN rows are copies of q and k, and the weights tree holds all-ones q_norm and
k_norm rows that nothing reads). Llama 3.1+ rotary scaling ("llama3") is only a different pinned
cos/sin table.

Numerics (see core.py for why each step is exact or correctly rounded):
  linear      int8 W (W7, rotated) x int8 activations (A7, per token) -> exact int sum, then
              fp32 (acc * s_x) * s_w. GPU: Pearl-mined Ampere kernel (or cuBLAS int8);
              CPU: fp64 GEMM, exact for these integer ranges.
  norms       RMSNorm with an exact sum of squares.
  rotary      cos/sin from a pinned table artifact (no libm at run time).
  attention   Q, K, V quantized per (token, head) to int8; Q K^T as an exact small-integer
              GEMM; softmax with det_exp and an exact denominator; P V with P (pre-scaled by
              V's per-token scale) quantized per row to 14 bits and split into two 7-bit limbs,
              each an exact small-integer GEMM over key chunks of 1024, summed in int64.
  MLP         SiLU from det_exp.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import torch

from . import core
from .core import F32, I64, c32, fadd, fdiv, fmul, fsqrt, fsub

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MAX_KEYS = 1 << 20
P_LEVELS = 16383  # 14-bit attention probabilities = 128 * hi + lo, hi, lo in [0, 127]
KEY_CHUNK = 1024  # keeps every partial sum of a 7-bit x int8 GEMM below 2**24


def llama3_inv_freq(inv: np.ndarray, factor: float, low: float, high: float, orig: int) -> np.ndarray:
    """Llama 3.1+ frequency scaling (transformers' _compute_llama3_parameters), in float64."""
    low_wl, high_wl = orig / low, orig / high
    wl = 2 * np.pi / inv
    out = np.where(wl > low_wl, inv / factor, inv)
    smooth = (orig / wl - low) / (high - low)
    mid = (1 - smooth) * out / factor + smooth * out
    is_mid = ~(wl < high_wl) & ~(wl > low_wl)
    return np.where(is_mid, mid, out)


def rope_table(theta: float, head_dim: int, max_pos: int = 8192, scaling: dict | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """Load (or, once, build and pin) the fp32 cos/sin table. Returns cos, sin, sha256.
    scaling: None, or Llama 3's {"factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings"}."""
    name = f"rope_theta{theta:g}_hd{head_dim}_pos{max_pos}.npz"
    if scaling:
        sc = (float(scaling["factor"]), float(scaling["low_freq_factor"]), float(scaling["high_freq_factor"]),
              int(scaling["original_max_position_embeddings"]))
        name = f"rope_theta{theta:g}_hd{head_dim}_pos{max_pos}_llama3_{sc[0]:g}_{sc[1]:g}_{sc[2]:g}_{sc[3]}.npz"
    path = os.path.join(ART, name)
    if not os.path.exists(path):
        os.makedirs(ART, exist_ok=True)
        inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
        if scaling:
            inv = llama3_inv_freq(inv, *sc)
        ang = np.arange(max_pos, dtype=np.float64)[:, None] * inv[None, :]
        emb = np.concatenate([ang, ang], axis=1)
        np.savez(path, cos=np.cos(emb).astype(np.float32), sin=np.sin(emb).astype(np.float32))
    d = np.load(path)
    h = hashlib.sha256(d["cos"].tobytes() + d["sin"].tobytes()).hexdigest()
    return d["cos"], d["sin"], h


class DetLinear:
    def __init__(self, W: torch.Tensor, device, hadamard_block: int, mine: bool):
        """W: n x k (any float dtype, CPU). Quantized on CPU in fp32 with the same deterministic ops."""
        n, k = W.shape
        self.n, self.k, self.device = n, k, torch.device(device)
        self.bs = min(hadamard_block, k & -k) if hadamard_block else 0
        qs, ss = [], []
        for s0 in range(0, n, 8192):
            q, s = core.quant_sym(core.fwht(W[s0:s0 + 8192].to(F32), self.bs), 64)
            qs.append(q.to(torch.int8))
            ss.append(s.squeeze(-1))
        self.Wq = torch.cat(qs).contiguous()
        self.sw = torch.cat(ss).contiguous()
        if self.bs > 1:  # same normalization folding as quant_act
            self.sw = fmul(self.sw, fdiv(c32(1.0, self.sw), fsqrt(c32(float(self.bs), self.sw))))
        self.sw = self.sw.to(self.device)
        self.mw = None
        self.mine = mine and self.device.type == "cuda"
        if self.device.type == "cuda":
            self.Wq = self.Wq.to(self.device)
        else:
            self.Wq64 = self.Wq.to(torch.float64)

    def setup_block(self, job_key_for_k, bound_for_k):
        if not self.mine:
            return
        from uwc_ampere import ops
        ok = self.k >= 2048 and self.k % 128 == 0 and self.n % ops.CONFIGS[ops.DEFAULT_CFG][0] == 0
        if not ok:
            self.mine = False
            return
        if self.mw is None:
            self.mw = ops.MinedWeight(Bt=self.Wq, b_scale=self.sw, r=128)
        self.mw.setup_block(job_key_for_k(self.k), bound_for_k(self.k))

    def int_gemm(self, q: torch.Tensor) -> torch.Tensor:
        """Exact q @ Wq^T for int8 q (m x k); returned as an integer-valued tensor."""
        m = q.shape[0]
        if self.device.type == "cpu":
            return q.to(torch.float64) @ self.Wq64.t()
        if self.mine and self.mw is not None:
            from uwc_ampere import ops
            mp = -(-m // ops.BM) * ops.BM
            qp = torch.nn.functional.pad(q, (0, 0, 0, mp - m)) if mp != m else q
            ones = torch.ones(mp, dtype=F32, device=q.device)
            return self.mw.forward(qp.contiguous(), ones, out_mode=ops.OUT_I32)[:m]
        mp = max(32, -(-m // 8) * 8)
        qp = torch.nn.functional.pad(q, (0, 0, 0, mp - m)) if mp != m else q
        return torch._int_mm(qp.contiguous(), self.Wq.t())[:m]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        q, sx = core.quant_act(x.reshape(-1, self.k), self.bs)
        acc = self.int_gemm(q)
        y = fmul(fmul(acc.to(F32), sx[:, None]), self.sw[None, :])
        return y.reshape(*shp[:-1], self.n)


ARCH_QWEN3, ARCH_LLAMA = 0, 1  # jobstate.ModelCfg.Arch
ARCHS = {"qwen3": ARCH_QWEN3, "llama": ARCH_LLAMA, "mistral": ARCH_LLAMA}


def rope_config(cfg) -> tuple[float, dict | None]:
    """(theta, Llama 3 scaling or None) from a transformers config (v4 or v5 layout)."""
    rp = dict(getattr(cfg, "rope_parameters", None) or {})
    rs = dict(getattr(cfg, "rope_scaling", None) or {})
    theta = getattr(cfg, "rope_theta", None) or rp.get("rope_theta") or 1e6
    kind = rp.get("rope_type") or rs.get("rope_type") or rs.get("type") or "default"
    if kind == "default":
        return float(theta), None
    if kind != "llama3":
        raise ValueError(f"rope scaling {kind!r} is not supported by the deterministic runtime")
    src = rp if rp.get("rope_type") == "llama3" else rs
    return float(theta), {k: src[k] for k in ("factor", "low_freq_factor", "high_freq_factor",
                                              "original_max_position_embeddings")}


class DetQwen3:
    """Qwen3, Llama and Mistral checkpoints (see ARCHS).

    device may list several GPUs ("cuda:0,cuda:1"): the layers are split into contiguous blocks,
    one per device; the embedding sits on the first, the final norm and lm_head on the last.
    Rows cross devices as exact copies, so outputs are bit-identical to one device."""

    def __init__(self, hf_model, device, hadamard_block: int = 512, mine: bool = True, max_pos: int = 8192):
        cfg = hf_model.config
        self.devices = [torch.device(d) for d in str(device).split(",")]
        self.cfg, self.device, self.out_device = cfg, self.devices[0], self.devices[-1]
        self.multi = len(set(self.devices)) > 1
        if cfg.model_type not in ARCHS:
            raise ValueError(f"model type {cfg.model_type!r} is not supported (supported: {sorted(ARCHS)})")
        self.arch = ARCHS[cfg.model_type]
        if getattr(cfg, "attention_bias", False) or getattr(cfg, "mlp_bias", False):
            raise ValueError("biased attention or MLP projections are not supported")
        sw = getattr(cfg, "sliding_window", None)
        if self.arch == ARCH_LLAMA and sw is not None and sw < max_pos:
            raise ValueError(f"sliding-window attention ({sw} < MaxPos {max_pos}) is not supported")
        if getattr(cfg, "hidden_act", "silu") != "silu":
            raise ValueError(f"activation {cfg.hidden_act!r} is not supported")
        self.hadamard_block = hadamard_block
        self.D, self.H, self.Hkv = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or self.D // self.H
        self.eps = cfg.rms_norm_eps
        self.qk_norm = self.arch == ARCH_QWEN3
        theta, scaling = rope_config(cfg)
        cos, sin, self.rope_sha = rope_table(theta, self.hd, max_pos, scaling)
        tabs = {d: (torch.from_numpy(cos).to(d), torch.from_numpy(sin).to(d)) for d in self.devices}
        self.cos, self.sin = tabs[self.device]
        hb = hadamard_block
        base = hf_model.model
        self.embed = base.embed_tokens.weight.detach().to(F32).to(self.device)
        self.layers = []
        nl = len(base.layers)
        for i, lay in enumerate(base.layers):
            dev = self.devices[i * len(self.devices) // nl]
            at, mlp = lay.self_attn, lay.mlp
            W = lambda mod: mod.weight.detach()
            ones = torch.ones(self.hd, dtype=F32, device=dev)  # committed, never read (Arch 1)
            self.layers.append(dict(
                in_norm=W(lay.input_layernorm).to(F32).to(dev),
                post_norm=W(lay.post_attention_layernorm).to(F32).to(dev),
                q_norm=W(at.q_norm).to(F32).to(dev) if self.qk_norm else ones,
                k_norm=W(at.k_norm).to(F32).to(dev) if self.qk_norm else ones,
                qkv=DetLinear(torch.cat([W(at.q_proj), W(at.k_proj), W(at.v_proj)]), dev, hb, mine),
                o=DetLinear(W(at.o_proj), dev, hb, mine),
                gu=DetLinear(torch.cat([W(mlp.gate_proj), W(mlp.up_proj)]), dev, hb, mine),
                down=DetLinear(W(mlp.down_proj), dev, hb, mine),
                dev=dev, cos=tabs[dev][0], sin=tabs[dev][1],
            ))
        self.norm = base.norm.weight.detach().to(F32).to(self.out_device)
        self.lm_head = DetLinear(hf_model.lm_head.weight.detach(), self.out_device, hb, mine)
        self.inv_sqrt_hd = float(torch.tensor(1.0 / np.sqrt(self.hd)).to(F32))

    def enter(self, dev: torch.device) -> torch.device:
        """Make dev current for torch and CuPy (a layer's GPU) and return it; nothing on one device."""
        if self.multi and dev.type == "cuda":
            torch.cuda.set_device(dev)
            try:
                import cupy
                cupy.cuda.Device(dev.index).use()
            except ImportError:
                pass
        return dev

    def qkn(self, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """The QN / KN step: RMSNorm per head (Qwen3), or the head itself (Llama, Mistral)."""
        return core.det_rmsnorm(z, w, self.eps) if self.qk_norm else z

    def linears(self):
        for L in self.layers:
            yield from (L["qkv"], L["o"], L["gu"], L["down"])
        yield self.lm_head

    # ---------------------------------------------------------------- cache
    def new_cache(self, B: int, max_len: int):
        z = lambda d, *s, dt=torch.int8: torch.zeros(*s, dtype=dt, device=d)
        return [dict(k=z(L["dev"], B, self.Hkv, max_len, self.hd), sk=z(L["dev"], B, self.Hkv, max_len, 1, dt=F32),
                     v=z(L["dev"], B, self.Hkv, max_len, self.hd), sv=z(L["dev"], B, self.Hkv, max_len, 1, dt=F32))
                for L in self.layers]

    # ---------------------------------------------------------------- forward
    def _rope(self, x, pos, L):
        cos = L["cos"][pos][None, :, None, :]  # (1, T, 1, hd)
        sin = L["sin"][pos][None, :, None, :]
        h = self.hd // 2
        rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
        return fadd(fmul(x, cos), fmul(rot, sin))

    def _attention(self, L, h, cache, start):
        B, T, _ = h.shape
        H, Hkv, hd = self.H, self.Hkv, self.hd
        qkv = L["qkv"](h)
        q, k, v = torch.split(qkv, [H * hd, Hkv * hd, Hkv * hd], dim=-1)
        q = self.qkn(q.reshape(B, T, H, hd), L["q_norm"])
        k = self.qkn(k.reshape(B, T, Hkv, hd), L["k_norm"])
        v = v.reshape(B, T, Hkv, hd)
        pos = torch.arange(start, start + T, device=h.device)
        q, k = self._rope(q, pos, L), self._rope(k, pos, L)
        kq, ks = core.quant_sym(k, 127)
        vq, vs = core.quant_sym(v, 127)
        cache["k"][:, :, start:start + T] = kq.to(torch.int8).transpose(1, 2)
        cache["sk"][:, :, start:start + T] = ks.transpose(1, 2)
        cache["v"][:, :, start:start + T] = vq.to(torch.int8).transpose(1, 2)
        cache["sv"][:, :, start:start + T] = vs.transpose(1, 2)
        S = start + T
        rep = H // Hkv
        K = cache["k"][:, :, :S].repeat_interleave(rep, dim=1).to(F32)    # (B,H,S,hd)
        SK = cache["sk"][:, :, :S].repeat_interleave(rep, dim=1)         # (B,H,S,1)
        V = cache["v"][:, :, :S].repeat_interleave(rep, dim=1).to(F32)
        SV = cache["sv"][:, :, :S].repeat_interleave(rep, dim=1)
        qq, qs = core.quant_sym(q.transpose(1, 2), 127)                  # (B,H,T,hd), (B,H,T,1)
        sc_int = core.exact_matmul_small(qq, K.transpose(-1, -2), 127 * 127 * hd)  # (B,H,T,S)
        sc = fmul(fmul(fmul(sc_int, qs), SK.transpose(-1, -2)), c32(self.inv_sqrt_hd, sc_int))
        qpos = pos[None, None, :, None]
        kpos = torch.arange(S, device=h.device)[None, None, None, :]
        sc = torch.where(kpos <= qpos, sc, torch.full_like(sc, float("-inf")))
        mx = sc.amax(dim=-1, keepdim=True)
        e = core.det_exp(fsub(sc, mx))
        den = core.exact_rowsum(e, MAX_KEYS)
        P = fdiv(e, den.unsqueeze(-1))
        Pv = fmul(P, SV.transpose(-1, -2))
        pmax = Pv.amax(dim=-1, keepdim=True)
        sp = fdiv(torch.maximum(pmax, c32(1e-30, pmax)), c32(float(P_LEVELS), pmax))
        pi = torch.round(fdiv(Pv, sp)).clamp_(0, P_LEVELS).to(torch.int32)
        hi, lo = (pi >> 7).to(F32), (pi & 127).to(F32)
        acc = torch.zeros(B, H, T, hd, dtype=I64, device=h.device)
        for c0 in range(0, S, KEY_CHUNK):
            Vc = V[:, :, c0:c0 + KEY_CHUNK]
            bound = 127 * 127 * KEY_CHUNK
            ah = core.exact_matmul_small(hi[..., c0:c0 + KEY_CHUNK], Vc, bound).to(I64)
            al = core.exact_matmul_small(lo[..., c0:c0 + KEY_CHUNK], Vc, bound).to(I64)
            acc += (ah << 7) + al
        o = fmul(acc.to(F32), sp)                                       # (B,H,T,hd)
        return L["o"](o.transpose(1, 2).reshape(B, T, H * hd))

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, cache, start: int = 0, layer_hashes: list | None = None) -> torch.Tensor:
        """ids: (B, T) tokens at positions start..start+T-1. Returns final-norm hidden (B, T, D) fp32."""
        x = self.embed[ids]
        for L, c in zip(self.layers, cache):
            x = x.to(self.enter(L["dev"]))
            x = fadd(x, self._attention(L, core.det_rmsnorm(x, L["in_norm"], self.eps), c, start))
            h = core.det_rmsnorm(x, L["post_norm"], self.eps)
            g, u = torch.split(L["gu"](h), [L["gu"].n // 2, L["gu"].n // 2], dim=-1)
            x = fadd(x, L["down"](fmul(core.det_silu(g), u)))
            if layer_hashes is not None:
                layer_hashes.append(sha(x))
        x = x.to(self.enter(self.out_device))
        return core.det_rmsnorm(x, self.norm, self.eps)

    @torch.no_grad()
    def logprobs(self, hidden: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        """log_softmax(lm_head(hidden)) rows, deterministic (det_exp, exact sum, det_log)."""
        flat = hidden.reshape(-1, self.D).to(self.enter(self.out_device))
        out = []
        for s0 in range(0, flat.shape[0], chunk):
            lg = self.lm_head(flat[s0:s0 + chunk])
            mx = lg.amax(dim=-1, keepdim=True)
            z = fsub(lg, mx)
            den = core.exact_rowsum(core.det_exp(z), 1 << 18)
            out.append(fsub(z, core.det_log(den).unsqueeze(-1)))
        return torch.cat(out).reshape(*hidden.shape[:-1], -1)

    @torch.no_grad()
    def generate_greedy(self, prompt: torch.Tensor, n_new: int):
        """Prefill the prompt, then decode n_new tokens one at a time. Returns (tokens, step hiddens)."""
        B, T0 = prompt.shape
        cache = self.new_cache(B, T0 + n_new)
        h = self.forward(prompt, cache, 0)
        last = h[:, -1]
        toks, hs = [], []
        for i in range(n_new):
            lp = self.logprobs(last[:, None])[:, 0]
            nxt = lp.argmax(dim=-1)
            toks.append(nxt)
            hs.append(last)
            if i + 1 < n_new:
                last = self.forward(nxt[:, None], cache, T0 + i)[:, 0]
        return torch.stack(toks, 1), torch.stack(hs, 1)


def sha(t: torch.Tensor) -> str:
    t = t.detach().contiguous().cpu()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:20]
