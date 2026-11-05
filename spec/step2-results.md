# Step 2 results: deterministic reference runtime

Build-plan step 2 of *Useful-Work Coin: Design v3*: "Exit when outputs match bit for bit across
two GPU families and CPU, for prefill, decode, and teacher-forced runs." **Met on Qwen3-1.7B**,
2026-11-06, with mining switched on for every linear layer on both GPUs.

| Device | Role |
|---|---|
| RTX 3080 (Ampere, sm_86), box .71 | GPU family 1, all 113 linear layers mined |
| RTX 5090 (Blackwell, sm_120), box .38/.50 | GPU family 2, all 113 linear layers mined |
| CPU (32-core x86 on the .38 box) | CPU reference |

`runtime/bench/xdevice.py` (2 WikiText sequences x 128 tokens), compared with
`runtime/bench/xdevice_compare.py`; raw dumps in `results/step2/`:

| Output | 3080 vs 5090 vs CPU |
|---|---|
| RoPE table (pinned artifact) | identical |
| Every decoder layer's prefill output (28 sha256) | identical |
| Final hidden states | identical |
| Teacher-forced token log-probs (scoring job) | identical |
| 16 greedy-decoded tokens per sequence | identical |
| Hidden state at every decode step | identical |
| Teacher-forced re-run == decode-time hidden states | true on all three |

Unit tests (`runtime/tests`, 8 pass on both GPUs): prefill = decode = teacher forcing, batch
composition invariance (a sequence alone vs. inside a batch of 4), CPU = GPU, exp/log accuracy,
order- and length-independence of the exact row sum.

## How it is deterministic (`runtime/detrt/`)

A from-scratch Qwen3 forward in plain tensor ops, so GPU and CPU run the same program:

- **Linear layers:** W7A7 int8 (block-Hadamard rotation), exact integer GEMM: the Pearl-mined
  Ampere kernel on GPU, an fp64 GEMM on CPU (exact for these integer ranges).
- **Attention:** Q, K, V quantized per (token, head) to int8. QKᵀ is a GEMM over small
  integers whose partial sums are all exactly representable in fp32, so any GEMM order is exact.
  P is pre-scaled by V's per-token scale, quantized per row to 14 bits and split into two 7-bit
  limbs; each limb times V is an exact small-integer GEMM over key chunks of 1024, summed in int64.
- **Row sums** (RMSNorm, softmax, log-softmax): int64 fixed point with a scale set by the row max
  and a fixed length bound, so a prefill row with masked tail and a decode row agree.
- **Elementwise math** in "emulated fp32": each + − × ÷ √ is computed in float64 and rounded once to
  float32, which is exactly the correctly rounded fp32 result. The first attempt used native
  fp32 and failed: PyTorch's CUDA fp32 `sqrt` is not correctly rounded and differed from CPU on
  16% of inputs.
- **exp / log** from those ops (range reduction plus fixed polynomials, ≤ 1–4 ulp), and RoPE cos/sin from a
  pinned table file (no libm at run time).
- **Activation quantizer** for the mined GEMMs: bit-identical between the CUDA kernel, this
  runtime, and a NumPy emulation (the GPU's approximate `rsqrtf` was removed from the kernel).

## Quality (WikiText-2, context 2048)

| Model | bf16 | W7A7, float attention (step 1) | Deterministic runtime |
|---|---|---|---|
| Qwen3-1.7B | 16.711 | 16.706 | **16.755 (+0.26%)** |
| Qwen3-4B | 13.656 | 13.647 | QWEN4B_RESULT |

## Costs and limits

- **Speed:** this is a reference runtime, not a fast one. On the 5090 the 2 × 128-token prefill
  takes 0.32 s, and 16 decode steps take 2 s (each decode step mines a 128-row padded GEMM for one real row).
  On CPU (32 cores) it takes 66 s and 135 s. The fp64-emulated elementwise ops and unfused attention
  are what a production provider runtime would turn into fused CUDA kernels with the same op order.
- **Sampling:** greedy only so far. The doc's seeded sampling (H(job, sequence, position)) needs
  an exact inverse-CDF over the int64 fixed-point probabilities; not built yet.
- **Coverage:** one model family (Qwen3). Other architectures need their own ops in the same style.
- **Disputes:** bisection to one operation (doc step 4) can use this runtime's op list directly;
  every op is either integer or a single correctly rounded fp32 op.
