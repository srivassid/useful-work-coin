# Raising λ: the detgpu engine (2026-11-07)

λ is the share of a GPU's noise-mining hash rate that it keeps while serving a job. A job is
cheaper for a client than renting the same GPU only above about λ ≈ 0.27 (step 6 formula with the
step-1 throughputs). Testnet run 1 measured live λ = **0.012**.

**Result:** live λ on the testnet, with real mining, jobs and agents, is now **0.254 on a 3080**
(behind a ×1 riser) and **0.227 on a 5090**. Outputs are bit-identical to the registered runtime
and identical across the two GPU families. Offline at 32 × 128 tokens it is 0.334 on the 3080.

## Where the time went (3080, 8 × 128 tokens, registered runtime)

| | Kernel time |
|---|---|
| fp32↔fp64 conversion kernels | 0.56 s (40%) |
| fp64 arithmetic (1/64 of fp32 rate on GeForce) | ~0.36 s |
| Stage outputs to the CPU and back between layer groups | 0.27 s |
| **Mined GEMMs** | **0.04 s (3%)** |
| Total (28,869 kernels; GPU busy 97% of wall time) | 1.38 s |

Launch overhead was not the problem, so CUDA graphs were dropped from the plan.

## What changed

- **Native fp32.** detrt emulates each fp32 op as an fp64 op rounded once. IEEE fp32 + − × ÷ √
  are correctly rounded, so they give exactly the same bits. `tests/test_fp32_native.py` found
  0 differences in 2²⁴ values per op, over all bit patterns and the model's range, on both the
  3080 and the 5090.
- **`runtime/detgpu`**, a separate package, so `detrt` and its registered `runtime_hash` are unchanged:
  - forward-only stage outputs, with activations kept on the GPU;
  - NVRTC kernels using explicitly rounded intrinsics in detrt's op order: RMSNorm + Hadamard +
    quantizer, qkv post-processing (per-head norm, RoPE, quant), tiled attention (exact int8
    scores with dp4a, det_exp, fixed-point softmax sum, 14-bit P, exact integer P·V),
    dequant + residual, and SwiGLU + quant;
  - about 45 kernels per layer instead of over 1,000.
- **Stage roots hashed on the GPU** (`detgpu/sha.py`): SHA-256 block leaves, row roots and
  `sout` leaves, so 32 B per token cross the bus. 8 MB takes 0.4 ms against 45 ms for copy plus
  CPU hashing. This is what makes ×1 risers fine: a provider's outputs would otherwise need about
  0.6 GB/s, and ×1 Gen 2 carries about 0.4 GB/s.
- **Agents** use detgpu by default (`--engine detgpu|detrt`). A job's `computed` time is its GPU
  part; `total` adds saving the stage inputs that verifiers may ask for. Dispute traces stay on
  detrt.

## Checks

| Test | Result |
|---|---|
| `tests/test_detgpu.py` | all 7 groups bit-identical to detrt.fast (wiki 8×128, random 3×128, 1×512, 2×300); output digests equal on 3080 and 5090 |
| `tests/test_sha_gpu.py` | leaves and roots equal to `agent.proto` for 6 shapes, including non-power-of-two trees |
| `tests/test_agent_engines.py` | identical stage roots, pooled outputs, served input rows, Merkle proofs and verifier recomputes under both engines |

## Measurements

Offline (`bench/lambda_det.py --engine detgpu-roots`; mined GEMMs under a synthetic key; every
stage's root hashed, as a provider commits):

| GPU | 8 × 128 | 32 × 128 |
|---|---|---|
| 3080 (×1 riser), noise 71.6 TMAC/s | 0.074 s/job, λ **0.273** | 0.242 s, λ **0.334** |
| 5090 (×16), noise 310 TMAC/s | 0.025 s/job, λ 0.187 | 0.066 s, λ 0.283 |
| Registered runtime, for comparison | 3080: 1.42 s, λ 0.014; 5090: 0.264 s, λ 0.018 | |

Live, on testnet run 2 (2026-11-07 17:00–17:45, blocks 1270–1304, nodes on box 71 and the 5090
box, 16 × 128-token client jobs; `results/testnet-run2-lambda.txt`):

| Provider | Jobs | GPU time per job (median) | Live λ (median; range) |
|---|---|---|---|
| miner71, RTX 3080 | 7 | 0.159 s | **0.254** (0.143–0.271) |
| provider245 on the RTX 5090 | 7 | 0.041 s | **0.227** (0.166–0.227) |

The lowest values are each provider's first job, which includes compiling the kernels.

## Round 2: batching, GPU-side commitments, tensor-core attention (2026-11-07 evening)

| Change | Effect |
|---|---|
| Mined GEMM at small m: projections as int8 GEMMs, 64-wide tiles, per-shape autotuner (`miner/ampere`, all opt-in) | ~1% at 1,024-token calls; off by default. A layer call is a chain of 6-7 small kernels whose fixed costs dominate, so single-kernel tuning barely moves it |
| **Jobs share GPU passes** (`Runtime.jobs_roots`, `--batch-tokens`, default 8,192): accepted jobs waiting for the GPU run in one forward pass | rows are independent, so each job commits to what it would alone |
| **Stage roots and pooled outputs on the GPU** (`detgpu.sha.stage_roots_gpu`, `detgpu.kernels.pooled`) | 32 B per stage leave the GPU instead of every stage's rows |
| **Stage inputs recomputed on demand** (`Runtime.stage_input`) instead of copied and saved as JSON for every job | removes 0.29 s (riser copy) + 0.55 s (JSON) per 16-sequence job on the 3080 box |
| **Attention on int8 tensor cores** (`attention_mma`): scores as int8 mma; the 14-bit levels split into 7-bit limbs (P_LEVELS = 16383 = 128·127 + 127) so P·V is two exact int8 mma's | 2.2-2.4x faster attention; dp4a kernel kept as fallback for T > ~900 |

All bit-identical: `tests/test_attention_mma.py` (12 shapes), `tests/test_provider_batch.py`, `tests/test_detgpu.py` (digests equal on 3080 and 5090), miner suite 28/28.

Provider end to end, 8 jobs of 8 × 128 (`bench/provider_batch.py`, `results/provider-batch-*.json`), wall-clock λ including the host side:

| GPU | Before (one job per pass, rows copied; JSON save not counted) | One job per pass now | Jobs batched (64 sequences per pass) |
|---|---|---|---|
| RTX 3080 (×1 riser) | 0.082 | 0.312 | **0.400** |
| RTX 5090 | 0.133 | 0.217 | **0.352** |

Single jobs (`bench/lambda_det.py --engine detgpu-roots`):

| | 16 × 128 | 64 × 128 | 16 × 512 |
|---|---|---|---|
| 3080 | 0.312 → **0.357** | 0.346 → **0.398** | 0.203 → **0.299** |
| 5090 | 0.249 → **0.293** | 0.305 → **0.351** | 0.187 → **0.265** |

Not yet confirmed on the live testnet (run 2 measured 0.254 and 0.227 before these changes).

## What's left

1. ~~The mined GEMM at small m~~: addressed by batching jobs (round 2).
2. ~~Attention at long sequences~~: tensor-core kernel (round 2); still the largest kernel at T = 512.
3. ~~Saving stage inputs~~: recomputed on demand (round 2).
4. **Confirm round 2 live** on the testnet (batched passes need providers with several queued jobs).
5. **The client price on the testnet** is set by the test client's fee, so it isn't a market
   measurement. With λ ≈ 0.25, a client pays about 0.75 of the GPU's mining income per GPU-second,
   against 0.99 at λ = 0.012.
