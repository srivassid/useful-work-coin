# Step 4 results: verification

Build-plan step 4: "Build the committee draw, private-salt sampling, commit-reveal, serve-or-slash
and bisection to one operation. Exit when every fault injected in a test suite is caught and the
right party is slashed." **Met** (2026-11-06). Protocol: `spec/stage-program.md`.

## What was built

| Piece | Where |
|---|---|
| Committee draw, salted sampling, commit-reveal (from step 3) | `node/jobstate/apply.go` |
| Stage programs: every stage is a fixed list of 26 row-op types, committed as a trace tree of row roots | `runtime/detrt/stage.py`, `node/jobstate/program.go` |
| Weights and input commitments: MODEL_REG weights_root over canonical quantized rows, JOB_POST input_root over token lists; MODEL_REG now carries the architecture | same |
| Referee: recomputes any single element of any leaf, bit-identical to the Python runtime | `node/jobstate/referee.go`, `detmath.go` |
| Dispute game: trace bisection, then row-block bisection, then one element, then a one-step proof checked by every full node | `node/jobstate/dispute.go` |
| Serve-or-slash: on-chain request for a stage input, rows proven against the previous stage | same |

## Cross-language bit-exactness (the referee is the runtime)

| Fixture | Checked | Result |
|---|---|---|
| Tiny Qwen3 (all weights) | every row of three stages (layer group 0, group 1 on group 0's output, both layers in one stage), both trace roots, the weights root | identical |
| Qwen3-1.7B, two one-layer stages, 8 tokens | every non-linear row (~400k elements) and 2,400+ sampled linear outputs, all 26 op types; trace roots | identical |

## Fault-injection suite (`node/jobstate/dispute_test.go`)

Real commitments, real stage programs (tiny model, 2 sequences × 2 layer groups), each party
playing the game from its own trace, and the provider building one-step proofs from whatever the
referee reads.

| Fault | Caught by | Who pays |
|---|---|---|
| Provider fakes a whole stage | verifier flag → game → recomputation | provider bond (½ to the verifier) |
| Provider substitutes a weight row (model substitution) | one-step weight proof fails against weights_root | provider |
| Provider computes on wrong tokens | one-step token proof fails against input_root | provider |
| Provider computes a stage on a wrong input | recomputation from the proven previous-stage rows | provider |
| Provider silent in the game | round deadline | provider |
| Provider posts invalid moves | move rejected (children don't hash to its node) | provider |
| Lying verifier vs honest provider | recomputation | verifier stake penalty |
| Provider and verifier both wrong at the same element | recomputation disagrees with both | provider bond and verifier penalty |
| Lying bonded challenger | recomputation | challenger bond (½ provider) |
| Silent challenger | round deadline | challenger bond |
| Stage input not served | ServeWindow | provider (½ to the requester) |
| Corrupted stage input served | DATA_SERVE rejected | provider |
| Honest job, and honest job with data served on request | – | nobody: settles normally |
| **Property, 60 trials:** a consistent lie at a random element of a random leaf, cycling through all 26 op types, alternately by the provider and by a verifier | the verdict comes from exactly one accepted one-step recomputation every time | the liar |

All job-module tests pass (18, including the step-3 suite and the randomized conservation
property with dispute games running). Node integration tests (blockchain, mining, mempool) pass.

## Costs

- Rounds ≤ ⌈log₂ leaves⌉ + ⌈log₂ blocks⌉ + 3. A 7-layer, 512-token stage of Qwen3-1.7B has 513,080 leaves: **≤ 28 rounds**, so at `GameWindow` = 10 blocks, under 5 hours at one-minute blocks.
- The largest one-step proof is about 50 KB. Full nodes recompute one element (one dot product or one row op).
- Serve-or-slash on-chain fallback: one stage input is T × D fp32 rows (4 MB for 512 × 2048). It's only needed when off-chain serving fails.

## Limits and deviations

- **One active game per job:** disputes are queued and played in order.
- **Silence:** a silent provider loses even if the challenger is also silent.
- **Model coverage:** stages cover decoder layers only. Logit scoring (lm_head + log-softmax) isn't a verified stage yet.
- **Batching:** one sequence per batch, all with the same SeqLen.
- **Data requests:** one per verifier per job, with no bond against griefing.
- **Not built:** the provider/verifier daemons that watch a live chain and run the GPU runtime. The client strategies exist as test code (`party` in `dispute_test.go`) and Python stage tracing. A real-model end-to-end dispute (Python traces of Qwen3 played through the Go game) is the next integration test.
