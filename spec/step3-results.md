# Step 3 results: job module in consensus

Build-plan step 3: "Add the state tree, coinbase commitment, issuance split and the transaction
types to Pearl's btcd fork. Exit when property tests of escrow, bonds and settlement pass."
**Met** (2026-11-06). Design: `spec/job-module.md`; code: `chain/patches/` (Pearl `node/jobstate`,
`node/blockchain/jobs.go`, `chaincfg`, `mining`, `mempool`).

## What was built

- **`node/jobstate`** (pure, deterministic Go):
  - accounts, verifier stakes with unbonding, models, jobs, pending withdrawals;
  - 11 op types (`WITHDRAW`, `STAKE`, `UNSTAKE`, `MODEL_REG`, `JOB_POST`, `JOB_ACCEPT`, `TRACE_COMMIT`,
    `VERIFY_COMMIT`, `VERIFY_REVEAL`, `TRACE_REVEAL`, `CHALLENGE`) with a strict binary encoding;
  - phase timers, stake-weighted committee draws, salted Merkle stage commitments;
  - settlement, slashing, no-fault abort, and an RFC 6962 state root over canonical records.
- **Consensus** (when `chaincfg.Params.JobModule` is set; nil for every Pearl network):
  - deposits are `OP_RETURN "UWCD"` outputs;
  - ops are `OP_RETURN "UWCJ"` outputs, authorized by the transaction's first input;
  - the coinbase must deposit ⌊ρ·subsidy⌋ to the pool, commit the state root, and pay due withdrawals;
  - the coinbase cap is subsidy + fees + withdrawals;
  - state is cached by block hash and persisted in the DB (reorg- and restart-safe);
  - checkpointed blocks are replayed from the spend journal.
- **Mining:** block templates add the pool deposit, withdrawals and the commitment.
- **Policy:** job outputs are exempt from the one-OP_RETURN rule, and state commitments are allowed only in a coinbase.

## Tests

| Suite | What it checks | Result |
|---|---|---|
| `jobstate` scenarios (10) | Exact payouts: honest job, withholding provider slashed, cheating provider caught (verdict either way), dispute timeout returns all funds, no-show verifier penalized, expiry, accept-without-commit, no double spend of escrow, failed ops are no-ops, unbonding stays slashable, withdrawals due exactly once | pass |
| `jobstate` properties | 40 random seeds × 300 blocks of random honest/cheating/withholding jobs, lazy verifiers, griefing challengers, junk ops; after **every** block: conservation (holdings = deposits + issuance − paid), no negative amounts, replay determinism; at the end every job terminal with nothing escrowed | pass |
| `jobstate` primitives | Merkle proofs for sizes 1–70; op encoding round-trip and rejection of truncated or trailing bytes (2000 random ops); committee draw distinct, deterministic, excludes client and provider, exactly stake-proportional (3× stake wins 75% ± 1.5%) | pass |
| `blockchain` integration | Real Taproot-signed txs and `ProcessBlock` on UWC SimNet: pool credited each block; wrong root, short pool deposit, overpaying coinbase and an unpaid due withdrawal are all rejected; withdrawal paid and spendable; a full honest job settles through consensus with exact provider pay; reorg drops the orphaned branch's ops; state reloads from the DB after restart | pass |
| `mining` | 12 blocks mined from production templates (each passing `CheckConnectBlockTemplate`) | pass |
| `mempool` | deposit + op standard; commitment outside a coinbase rejected; 3 KB multi-push op allowed | pass |
| Pearl regression | blockchain, chaincfg, mining, mempool, txscript, indexers | unchanged (only the pre-existing `-tags zkpow` tests fail) |

`jobstate` statement coverage: 94.9%.

## Issues found while building

- A stake-weighted draw by "smallest hash ÷ stake" is not proportional to stake: a 3× staker won
  83% of draws, matching the analytic 5/6. Replaced with exact successive sampling over cumulative stake.
- Pearl's consensus allows only P2TR, P2MR and OP_RETURN outputs, so a withdrawal to any other script
  could never be paid, and a due withdrawal that can't be paid would make every later block
  invalid. `WITHDRAW` now accepts only P2TR/P2MR scripts.
- The committee seed must come from the *previous* block's hash: the block's own hash depends on
  the coinbase, which commits the state that the draw produces.

## Left for later steps

- **Dispute verdicts** (step 4): bisection to one operation of the step-2 runtime. Until then a
  dispute ends in a no-fault abort after `DisputeTimeout`.
- **Serve-or-slash evidence** for stage inputs (step 4).
- **Pool budget spending, canary jobs, forced-error decoys** (step 5).
- **RPCs and a client/verifier daemon** (steps 4 and 7).
- **Performance:** the state root is recomputed in full each block (a sparse Merkle tree later), and
  each op snapshots the state for all-or-nothing rollback.
