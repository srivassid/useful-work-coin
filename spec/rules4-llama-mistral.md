# Rules 4: Llama and Mistral models

Status (2026-11-10): runtime and consensus code done and tested on CPU; GPU paths written but
not yet run on a GPU; no network schedules Rules 4 yet.

## What changes

The deterministic runtime and the stage program were written for Qwen3. Llama (3.x) and
Mistral (7B v0.3) differ from it in two ways that matter:

1. **No q/k norm.** Qwen3 applies RMSNorm to every q and k head before rotary; Llama and
   Mistral do not. This is the one consensus change: `ModelCfg.Arch` (0 Qwen3, 1 Llama /
   Mistral). For Arch 1 the stage program is unchanged and the `QN`/`KN` rows are the q/k heads
   as they are (a copy of their slice of the `QKV` row). The weights tree is unchanged too: an
   Arch 1 registration commits all-ones `q_norm`/`k_norm` rows that the referee never reads.
   Keeping the program and tree shape means one branch in the referee and no new row layouts.
2. **Rotary scaling** (Llama 3.1 and later, `rope_type: llama3`). No rule change: the cos/sin
   values are a table in the weights tree, so scaling is only a different pinned table
   (`detrt.qwen3.llama3_inv_freq`, float64, equal to transformers' table to fp32 rounding).

Also: tied embeddings (Llama 3.2 1B/3B) need nothing (lm_head rows are the quantized embedding
matrix). The runtime refuses what it does not implement: attention/MLP biases (Qwen2),
sliding-window attention shorter than MaxPos (Mistral v0.1 with MaxPos > 4096), other rotary
scalings, activations other than SiLU.

## Wire format and activation (chain patch 0021)

MODEL_REG gains an optional suffix `'A', arch` after the generation marker `'G'`, written only
for Arch 1. Qwen3 registrations encode, decode and enter the state root exactly as before (an
explicit `'A', 0` is rejected, so each op has one encoding). Arch 1 registrations fail before
Rules 4. Nodes without patch 0021 skip such an op as undecodable; newer nodes reject it before
Rules 4 and restore the state snapshot, so both reach the same state root.

## Runtime hash

`runtime_hash` covers the `detrt` sources, which this change edits (`qwen3.py`, `stage.py`,
`fast.py`). Qwen3 numerics are unchanged: the tiny Qwen3 fixture regenerates byte-identical
(rows, stage roots, weights root). Models registered earlier keep the hash of the source they
were registered with (git `f4e7dc3`); new registrations get the new hash.

## Tests (CPU)

- Go: `TestRefereeMatchesPythonTinyLlama` (every row of three stages of a tiny Llama with Llama 3
  rotary scaling, bit-exact against the Python runtime; the same weights computed as Qwen3 give
  another root), `TestRules4ModelRegWireFormat`, `TestRules4ModelRegActivation`; full jobstate
  suite passes.
- Python: on tiny random Qwen3 / Llama 3 / Mistral models the deterministic runtime is as
  close to transformers' fp32 forward for the new family as for Qwen3 (KL 0.0018 vs 0.0023,
  top-1 agreement 0.91 vs 0.86, the W7A7 quantization noise); `detrt.fast` equals `run_stage`
  on the tiny Llama.

## Still to do

1. GPU (box 71 or the 5090, short session): `detgpu` fused `qkv_post` with `qknorm=0`, and rows
   over 48 KB of shared memory (I = 14336 for Mistral 7B and Llama 3.1 8B) in `quant_act`,
   `rmsnorm_quant`, `swiglu_quant`: run `tests/test_detgpu.py`, `tests/test_detgen_fast.py` and
   `tests/test_agent_engines.py` on a Llama checkpoint, and a 3080-vs-5090 bit-identity check.
2. Models: the fast GPU paths need head_dim 128, so Llama 3.2 3B, Llama 3.1 8B and Mistral 7B
   v0.3 (not Llama 3.2 1B, head_dim 64). Memory on a 10 GB 3080: the embedding is held in fp32
   (V x D x 4 bytes: 1.6 GB for Llama 3.2 3B, 2.1 GB for Llama 3.1 8B), so 3B and Mistral 7B
   fit, Llama 3.1 8B does not without moving the embedding table off the GPU. Llama and
   Mistral weights on Hugging Face are gated (licence acceptance and a token).
3. W7A7 quality per model (perplexity vs bf16, Hadamard block choice), as step 1 did for Qwen3.
4. Schedule Rules 4 on the testnet (`UWCTestNetRules4Height`), rebuild and deploy `pearld` /
   `uwcctl` on every node including the droplet, register the models, update the package.
