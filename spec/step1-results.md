# Step 1 results: λ and quantization quality (RTX 3080)

Build-plan step 1 of *Useful-Work Coin: Design v3* asks for two numbers before any chain code:
λ for real models in {−64..64}, and the quality loss of that quantization. Both are measured
here on one RTX 3080 10 GB (Ampere, sm_86, 225 W limit), box 192.168.29.71, 2026-11-05.

## What was built

Pearl's own NoisyGEMM kernels (`miner/pearl-gemm`) are Hopper-only (CUTLASS 3, TMA, wgmma,
`sm_90a`), so they were ported to Ampere (`miner/ampere/`). The port is a drop-in set of GPU
kernels under Pearl's unchanged consensus rules:

| Piece | Implementation |
|---|---|
| Commitment | BLAKE3 keyed tree hash of A and Bᵀ on GPU, v3 salted seed derivation |
| Noise | E_AL / E_BR uniform and E_AR / E_BL permutation matrices from BLAKE3, on GPU |
| Mining GEMM | int8 `mma.sync.m16n8k32`, 128×128 CTAs at 2 per SM; each thread XOR-folds its 8×16 accumulator tile every r = 128 along k and hashes it with BLAKE3 at the end |
| Hash-tile pattern | rows [0,8,…,56], cols [0,1,8,9,…,56,57] (h·w = 128); valid under Pearl's rules |
| Useful output | exact int32 denoise (balanced base-128 limbs), so the runtime is integer-only and bit-reproducible, as the design requires; Pearl itself denoises in fp16 (approximate) |
| Activations | fused block-Hadamard rotate + per-token quantize to [−64, 64] |
| Model | Qwen3 (HF), QKV and gate/up fused; every linear layer, including `lm_head`, is mined |

Correctness (25 tests, `miner/ampere/tests`):
- a pure-Python port of Pearl's Rust spec produces proofs that **Pearl's Rust verifier
  (`verify_plain_proof_v3`) accepts**, and rejects losing tiles;
- every GPU kernel is bit-exact against that reference; the set of winning tiles from a
  GPU forward equals the reference's exactly;
- **GPU-mined winning tiles produce proofs that Pearl's verifier accepts**;
- the mined layer's output is bit-identical to the same integer math through cuBLAS int8.

Kernel speed: the plain int8 GEMM runs at 85–110% of cuBLAS int8 (e.g. 179 vs 186 TOPS at
4096³); hashing every tile costs ~0–11% on top.

## λ

λ = (hashed MACs per GPU-second during the job) ÷ (hashed MACs per GPU-second during pure
noise mining), on the same GPU. Hashed MACs are exactly proportional to expected tickets,
because Pearl's target scales with tile size × k. GPU time is summed kernel time (CUPTI);
wall time is also reported. Noise mining: 8192×16384×4096 random GEMMs with commitment and
noise, sustained, **70–73 TMAC/s** (≈ 145 int8 TOPS, power-limited).

Jobs: batches of 8192 tokens (16 × 512) from WikiText-2. *Embed* = forward + pooling;
*score* = forward + mined `lm_head` + log-softmax (log-likelihood scoring).

| Model | Runtime | Job | GPU ms / batch | λ (GPU time) | λ (wall) |
|---|---|---|---|---|---|
| Qwen3-1.7B | HF eager | embed | 463 | 0.31 | 0.40 |
| Qwen3-1.7B | HF eager | score | 562 | 0.33 | 0.39 |
| Qwen3-1.7B | torch.compile (non-GEMM ops fused) | embed | 374 | 0.37 | 0.50 |
| Qwen3-1.7B | torch.compile | score | 468 | 0.37 | 0.45 |
| Qwen3-4B | HF eager | embed | 1014 | 0.41 | 0.44 |
| Qwen3-4B | HF eager | score | 1145 | 0.41 | 0.42 |
| **Qwen3-4B** | **torch.compile** | **embed** | **855** | **0.49** | 0.52 |
| **Qwen3-4B** | **torch.compile** | **score** | **986** | **0.47** | 0.49 |

Where the time goes (Qwen3-4B, compiled, embed, 855 ms): mined GEMMs 517 ms, mining
prep (commit, noise, projections) 82 ms, everything else (attention, norms, quantizer,
rotary) 256 ms. At the noise rate, the job's hashed work is worth 417 ms of noise mining.
So the ceilings are:

- GEMM-only λ ≈ 0.81 (the exact denoise costs 2·L·r/k extra work: +25% at k = 2048, less at larger k);
- λ with zero non-GEMM cost ≈ 0.70;
- λ grows with model size (1.7B → 4B: +0.1), because GEMM work grows faster than elementwise work.

Cost to the client: the same job without mining (int8, same kernels, no hashing or noise)
takes **21–28% less GPU time**; equivalently, mining adds 27–40% (4B compiled embed: 641 vs
855 ms). Mining-while-serving is still faster
than stock bf16 serving on this GPU (Qwen3-1.7B: 374 ms mined vs 504 ms bf16, compiled),
because int8 tensor cores run at 2× bf16. Qwen3-4B in bf16 does not fit a 10 GB 3080 at all;
the mined int8 version uses 6.6 GB.

## Quality of the {−64..64} runtime

W7A7: weights per output channel, activations per token, both symmetric into [−64, 64],
with an optional block Walsh–Hadamard rotation. The integer path is bit-identical to the
mined path, so these are the mined model's numbers.

WikiText-2 perplexity (context 2048, 299k tokens):

| Model | bf16 | W7A7, no rotation | + Hadamard 128 | + Hadamard 512 |
|---|---|---|---|---|
| Qwen3-1.7B | 16.711 | 17.529 (+4.9%) | 17.160 (+2.7%) | **16.706 (−0.0%)** |
| Qwen3-4B | 13.656 | 14.346 (+5.1%) | **13.647 (−0.1%)** | 13.895 (+1.7%) |

Qwen3-Embedding-4B, 256 WikiText paragraphs, bf16 vs W7A7:

| Rotation | cos(bf16, int) mean / min | top-1 neighbour agreement | top-10 overlap | pairwise-sim Spearman |
|---|---|---|---|---|
| Hadamard 128 | 0.990 / 0.986 | 91.0% | 96.0% | 0.9966 |
| Hadamard 512 | 0.993 / 0.990 | 93.8% | 96.6% | 0.9972 |

So the value-range constraint costs no measurable LLM quality, provided the rotation block
is calibrated per model (512 for 1.7B, 128 for 4B). Embeddings move slightly (cos ≈ 0.99).
The block size belongs in `MODEL_REG` with the reference runtime hash.

## What this means for the design

- **Go, but λ is medium, not high, on consumer GPUs.** Measured λ for the v1 job types
  (batched embeddings and scoring) is 0.41–0.49 for a 4B model with fused non-GEMM ops. The
  doc's table expected "high" λ for these; on a 3080 they are medium.
- **Client fee.** fee ≥ (1 − λ) × mining income ≈ 0.5 × mining income per GPU-second at λ ≈ 0.5.
- **Pool reach.** u = ρ / (1 − λ): at ρ = 10% and λ = 0.47–0.49, 19–20% of GPUs move to pool
  work (the doc's example used λ = 0.9 → 100%). Hashrate still falls by ρ.
- **New constraint from Pearl's rules:** only GEMMs with k ≥ 16·r = 2048 can mine (r ≥ 128).
  Models with hidden size < 2048 (most small embedding models, e.g. 1024) mine only in their
  FFN down-projection. The v1 job list should require hidden ≥ 2048. Pearl's miner also
  uses noisy GEMM only for m ≥ 1024 tokens, which matches "large batches only".
- **Bit-exact integer runtime is feasible** at a cost of 2·L·r/k extra GEMM work (L = 2 limbs
  in practice), which is exactly what makes sampled verification and one-operation dispute
  bisection possible.
- **Biggest remaining levers on λ:** fusing quantization into the preceding norm or activation, a
  faster prep path (projections via int8 GEMM instead of atomics), and larger models or GPUs
  where GEMMs dominate. None of these needs consensus changes.

## Caveats

- One GPU type (Ampere 3080). Hopper numbers would use Pearl's own kernel, not this port.
- λ is measured with my kernels for both the job and the noise miner; a better noise kernel
  would lower λ, and a better job runtime would raise it.
- GPU-time λ assumes launch gaps are removed (CUDA graphs or fusion); wall-time λ is with HF eager
  launch overheads.
- The quality evaluation covers perplexity and embedding similarity, not downstream task suites.
- Decode-heavy generation was not measured; per the doc's table it is expected to be lower.

## Reproduce

On the GPU box: `miner/ampere/bench/run_matrix.sh <model_dir> <tag>` (λ),
`python -m eval.ppl --model <dir>` (perplexity), `python -m eval.embed_quality --model <dir>`
(embeddings), `pytest tests/` (correctness). Raw outputs are in `results/`.
