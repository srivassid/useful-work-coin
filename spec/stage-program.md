# Stage programs, commitments and dispute games (consensus)

These rules fix how a job's work is committed and how a disagreement is reduced to one
recomputed element. Implementations: Python `runtime/detrt/stage.py` (providers, verifiers) and
Go `node/jobstate/{program,referee,detmath,dispute}.go` (full nodes). The two are checked
bit for bit against each other on every op (`node/jobstate/referee_test.go`).

## Stage

A job has `batches × G` stages (G = the model's layer groups). Stage `s = b·G + g` is sequence `b`
(SeqLen tokens, positions 0…T−1) through layers `[g·L/G, (g+1)·L/G)`. Its input is the
job's token list for g = 0, else the last layer's `R2` rows of stage `s − 1`.

## Program (leaf order)

Within a stage, layers in order. Inside a layer, op-major: all tokens of one op, then the next op.

| # | Op | Row (4-byte words) | Element / row rule (runtime ops) |
|---|---|---|---|
| 0 | EMBED[t] (first layer group only) | D | embed row of token t |
| 1 | N1[t] | D | RMSNorm(x_t, in_norm) |
| 2 | QA1[t] | D+1 | quant_act (Hadamard + per-token int, ±64); last word = scale |
| 3 | QKV[t] | (H+2Hkv)·hd | element j = (float(q·W_j) · s_x) · s_wj |
| 4–5 | QN[t,h], KN[t,h] | hd | RMSNorm of the head slice (q_norm / k_norm) |
| 6–7 | QR[t,h], KR[t,h] | hd | RoPE with the pinned cos/sin rows of position t |
| 8–10 | KQ, VQ, QQ [t,h] | hd+1 | quant_sym ±127 (+ scale) |
| 11 | VT[d,h] | T | column d of VQ over all tokens |
| 12 | VS[h] | T | V scales over all tokens |
| 13 | S[t,h] | t+1 | element j = ((float(q8·k8_j) · s_q) · s_kj) · 1/√hd |
| 14 | P[t,h] | t+2 | softmax with det_exp and exact sum, ×V scale, 14-bit quant (+ scale) |
| 15 | O[t,h] | hd | element d = float(Σ_j p_j · v_jd) · s_p |
| 16–17 | QA2[t], OP[t] | H·hd+1, D | quant_act of concatenated heads; o_proj |
| 18 | R1[t] | D | x + OP |
| 19–24 | N2, QA3, GU, SM, QA4, DN | D, D+1, 2I, I, I+1, D | RMSNorm, quant, gate_up, SiLU(g)·u, quant, down |
| 25 | R2[t] | D | R1 + DN (layer output) |

Floats are IEEE fp32 bit patterns; integers are int32; arithmetic is runtime/detrt/core.py
("emulated fp32": every op in float64 then rounded once to fp32; exact int and fixed-point sums).

## Commitments (sha256, RFC 6962 trees: leaf = H(0x00‖x), node = H(0x01‖l‖r), split at the largest power of two)

| Object | Leaf | Root |
|---|---|---|
| Row | 1 KiB blocks of the row's little-endian words | `row_root` |
| Stage trace | `H(0x00 ‖ "leaf" ‖ u32 index ‖ row_root)` in program order | stage root (committed salted in TRACE_COMMIT) |
| Weights (MODEL_REG) | `H(0x00 ‖ "w" ‖ u32 i ‖ row_root)`: embed rows; then per layer in_norm, post_norm, q_norm, k_norm, qkv rows, o rows, gu rows, down rows (int8 widened to int32 + fp32 row scale); then rope.cos, rope.sin rows | weights_root |
| Job input (JOB_POST) | `H(0x00 ‖ "tok" ‖ u32 batch ‖ row_root(token ids))` | input_root |

## Dispute game (`DISPUTE_MOVE`, `ONE_STEP`)

Between the provider and the first queued disputant (a flagged verifier, or a bonded
challenger), over one stage whose roots differ. Every round both parties move within `GameWindow`
blocks; a party that does not make a valid move in time loses (a silent provider loses first).

1. **Trace bisection**: from the root, each party posts the two children of its current node. Play
   continues left if the left children differ, else right. It ends at the first differing leaf;
   every earlier leaf is identical in both trees.
2. **Leaf**: both post the row root of that leaf (it must hash to their leaf).
3. **Block bisection**: the same over the row's block tree, down to one block; both post its bytes.
   The first differing 4-byte word is the disputed element.
4. **One step**: the provider posts the element's inputs (earlier leaves of the stage proven against
   its own stage root, weights rows against weights_root, stage-input rows against the previous
   stage's root, token ids against input_root). Every full node recomputes the element
   with `referee.go`. A party whose value differs from the recomputation loses.

Rounds ≤ ⌈log₂ leaves⌉ + ⌈log₂ blocks⌉ + 3. For Qwen3-1.7B (7-layer stages of 512 tokens): 513,080 leaves,
largest row 48 blocks, **≤ 28 rounds**. The largest one-step proof is about 50 KB (the SM element needs
the 12,288-word GU row).

Outcomes:
- **Provider wrong:** job failed. The client is refunded, and the provider's bond goes ½ to the disputant (if it was right), ½ to the treasury.
- **Disputant wrong:** a challenger's bond goes ½ to the provider, ½ to the treasury; a verifier loses `VerifierNoShowPenalty` of stake.
  Then the next queued dispute is played, or the job returns to its challenge window and settles.
- **Both wrong:** the provider fails and the disputant is penalized too.

## Serve-or-slash (`DATA_REQUEST`, `DATA_SERVE`)

A committee member may request the input of a stage with g > 0 while verification is open. The
provider must post every input row, each proven against the opened previous stage root, within
`ServeWindow`, or fail (bond ½ to the requester). Verification phases wait for pending requests
and then allow a full window.
