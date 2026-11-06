# Job module (build step 3)

The job layer of *Design v3* as consensus code in Pearl's node (`node/jobstate`, hooks in
`node/blockchain`). Block validity never depends on whether a job's work was honest; it
depends only on applying these deterministic rules.

## Money in, money out

| Flow | Encoding | Rule |
|---|---|---|
| Deposit | output `OP_RETURN "UWCD" <account:32>` with value v | v leaves the UTXO set (OP_RETURN outputs are unspendable) and is credited to `account` |
| Job operation | output `OP_RETURN "UWCJ" <op>` with value 0 | authorized by the transaction's first input: the actor is `sha256(prevout pkScript)` |
| Withdrawal | op `WITHDRAW(amount, pkScript)` | debited at once; at height h + `WithdrawDelay` the coinbase must contain an output paying exactly `amount` to `pkScript` |
| Pool issuance | coinbase output `OP_RETURN "UWCD" <POOL>` | value must be at least ⌊ρ · subsidy⌋ (ρ = `PoolShareBP` / 10000) |
| State commitment | coinbase output `OP_RETURN "UWCS" <root:32>` | must equal the job-state root after the block (like SegWit's witness commitment) |

Coinbase value rule becomes: outputs ≤ subsidy + fees + Σ withdrawals due at this height.
Conservation: Σ UTXO values + Σ module holdings = total issuance, at every height.

An operation that fails its checks (insufficient balance, wrong phase, bad proof) is recorded
as failed and changes nothing; the transaction stays valid and pays its fee. So the mempool
needs no job-module rules, and a block is never invalid because of a job operation.

## Lifecycle

| # | Op / event | By | Effect |
|---|---|---|---|
| 1 | `MODEL_REG(weights_root, runtime_hash, layer_groups, rotation_block)` | registrant | locks `ModelRegBond`; model id = hash of the record |
| 2 | `JOB_POST(model, input_root, stages S, fee F, deadline, k, n)` | client | escrows F; fixes verifier pay V = max(F·1%, k·n·`VerifierFloorPerSample`) and treasury T = F·1%; needs F ≥ V + T |
| 3 | `JOB_ACCEPT(job, bond B)` | provider | needs B ≥ F · `MinBondBP`; locks B; before the deadline |
| 4 | `TRACE_COMMIT(job, stage_root, output_root)` | provider | before the deadline. stage_root = Merkle root over H(stage ‖ salt_i ‖ root_i), so unrevealed roots stay hidden |
| 5 | committee draw (automatic) | – | at the end of block commit+`CommitteeDelay`: k verifiers by stake-weighted lottery on H(that block's hash ‖ job ‖ account), excluding client and provider |
| 6 | `VERIFY_COMMIT(job, H(salt), H(roots ‖ account ‖ salt))` | each drawn verifier | within `VerifyCommitWindow` |
| 7 | `VERIFY_REVEAL(job, salt, roots[n])` | verifier | in the following `VerifyRevealWindow`; both hashes must match; positions = n distinct H(salt ‖ i) mod S |
| 8 | `TRACE_REVEAL(job, [stage, salt_i, root_i, proof])` | provider | in the following `ProviderRevealWindow`; each proof must open stage_root |
| 9 | comparison (automatic, end of reveal window) | – | a requested stage left unrevealed → provider slashed. A verifier root ≠ provider root → dispute. Otherwise the challenge window opens |
| 10 | `CHALLENGE(job, stage, claimed_root, bond)` | anyone | during `ChallengeWindow` W; bond ≥ `ChallengeBondMin`; opens a dispute |
| 11 | dispute verdict | consensus (step 4: bisection to one op) | provider at fault → slashed; else the challenger (or flagging verifier) pays |
| 12 | settlement (automatic, end of W) | – | provider gets F − V − T plus B back; revealing verifiers share V; T and unpaid V go to the treasury |

Timeouts:
- Not accepted by the deadline: F refunded.
- Accepted but not committed by the deadline: B goes ½ to the client, ½ to the treasury, and F is refunded.
- A verifier that doesn't commit or reveal: `VerifierNoShowPenalty` taken from its stake, to the treasury.
- Dispute unresolved after `DisputeTimeout`: no-fault abort (F refunded, B and challenge bonds returned).

Slash split: the provider's B goes ½ to whoever caught it (the flagging verifiers or the challenger), ½ to the treasury, and the client gets F back. A failed challenge's bond goes ½ to the provider, ½ to the treasury.

Verifier staking: `STAKE(amount)`; `UNSTAKE(amount)` stays slashable and leaves the draw
immediately, and returns to balance after `UnbondDelay`.

## State root

Every record (accounts, stakes, models, jobs, pending withdrawals, counters) is serialized
canonically under a typed key. The root is an RFC 6962 Merkle tree over sha256(0x00 ‖ key ‖ value),
with keys sorted, so it doesn't depend on map order or history. v1 recomputes it per block; a
sparse Merkle tree can replace it later without changing the rules above.

## Deviations from the design doc

- **Reveal order:** verifiers reveal before the provider, not after. It's equally safe because the
  provider's roots are bound by stage_root before any position is known, and verifiers are bound by
  their commit hash. It removes an off-chain request step: the provider learns which stages to reveal from the chain.
- **Serve-or-slash for stage inputs** (doc step 7): the data itself is off-chain. On-chain
  request and serve evidence is left to step 4.
- **Pool spending** (doc step 5): this step only credits ρ · subsidy to the POOL account.
