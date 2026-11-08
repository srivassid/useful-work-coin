# Generation jobs (Rules 3)

Verified text generation: a client posts prompts, a provider generates, and verifiers check the
result with the same stage machinery as embedding jobs. The reference is `runtime/detgen`; the
consensus side is `ext/pearl/node/jobstate/program_gen.go` (chain patch 0018).

## Why it fits the existing design

A finished generation can be re-run in one pass. Feeding the whole sequence (prompt plus
generated tokens) back through the model gives bit for bit the same hidden states as generating it
token by token with the KV cache. The deterministic runtime guarantees this (`tests/test_detrt.py`:
prefill == decode == teacher forcing). So the layer stages run over the final sequence exactly as
in an embedding job, and generation adds only one stage per sequence: the head.

## Layout

- **Batch:** a batch is one sequence of the job's `seq_len` T. Its input row (in the job's input
  tree) is the prompt, then `GenSentinel` (0xFFFFFFFF) for every position to be generated. The
  prompt length P is the index of the first sentinel.
- **Stages:** each batch has its G layer-group stages, then the head stage, so
  `stages = batches × (G + 1)`.
- **Head stage:** for every position t in [0, T-2]:

| leaf | row | recomputed from |
|---|---|---|
| HN(t) | final RMSNorm of the last layer's R2 row (D floats) | the stage input row t and the `norm` weight row |
| HQ(t) | lm_head activation quantization of HN (D int8 + scale) | HN(t) |
| HL(t, c) | the scores of vocabulary chunk c (1,024 tokens; the last chunk shorter) | per element: HQ(t) and one `lm_head` row |
| HC(t, c) | the chunk's best `[token, score]` (first maximum) | HL(t, c) |
| HS(t) | the sample `[token, score]`: the first best over the chunks | HC(t, ·) |

  The HS rows are the head stage's outputs: its committed root is their stage-output root.
- **Generated tokens:** token t+1 (for t+1 ≥ P) is HS(t).token. Layer stage 0 reads it from the head
  stage: in a one-step proof, an EMBED element at a generated position takes its token from a
  `RowGenToken` row proven against the batch's opened head stage.
- **Score:** greedy (temperature 0) uses the logit. Otherwise the score is
  `logit / temperature + gumbel(seed, t, v)` (Gumbel-max sampling), where the seed is the job id
  and `TempMilli` is at most 4,000. The noise is a counter-based hash:
  - `x = splitmix64(key ^ splitmix64(t << 32 | v))`, with key = the first 8 bytes of
    `sha256("uwc-gumbel" || seed)`;
  - `u = (2·(x >> 41) + 1)·2⁻²⁴`, exact in fp32;
  - `g = -det_log(-det_log(u))`.
- **Output root:** a generation job's output root (TRACE_COMMIT H2) is
  `MerkleRoot_b sha256(0x00 || "gout" || u32 b || head output root of b)`. A head stage opens
  (TRACE_REVEAL, ONE_STEP, DATA_SERVE) only with its root's path in the output root (`OutProof`). So
  the samples a client was delivered, which it checks against the output root, are the samples
  the verifiers check.
- **Weights:** a generation model's weights tree is the embedding layout followed by the `norm` row
  and the V `lm_head` rows (`detgen/weights.py`).

## Wire and state (only when set, so earlier encodings and state roots are unchanged)

- **MODEL_REG:** a trailing `G` marks a generation model.
- **JOB_POST:** after the optional 32-byte provider, `G` and a u32 temperature.
- **Reveals:** after an op's reveals (TRACE_REVEAL) or inputs (ONE_STEP, DATA_SERVE), the
  output-root paths of head stages, as `(reveal index, path)` entries.
- **Job state:** a generation job encodes `(3, TempMilli)` after the Rules 2 suffix. A generation
  model encodes a trailing `3`.

## Not yet

- **Early stop:** generation always runs to `seq_len`, and the client cuts at its stop token.
  Stopping early needs a consensus rule for the forced fill after a stop token.
- **Sampling options:** there is no top-k or top-p, only temperature.
- **Provider throughput:** the reference generator decodes one sequence at a time. A fast batched
  decode path (detgpu) and agent support come next.
- **Activation:** Rules 3 is not active on the testnet yet. It gets an activation height when the
  agents and a generation model registration are ready.
