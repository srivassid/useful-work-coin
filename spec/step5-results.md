# Step 5 results: pool and watchers

Build-plan step 5: "Add pool budget accounting, unit pricing, forced errors with threshold keys,
and canary jobs." The plan gives no exit test for this step. This report uses the bar from steps
3 and 4: exact accounting under property tests, every injected fault caught with the right party
paying, and real games on the tiny Qwen3 for the new catch paths. **Met** (2026-11-06).

Design: `spec/pool-and-watchers.md`. Code: patch 0006 (`node/jobstate/pool.go` and hooks;
+2,045 lines with tests) and `verifier/forced_errors.py`.

## What was built

| Piece | Where |
|---|---|
| t-of-m council: `PROPOSE` / `APPROVE`; actions run as a system account | `pool.go` |
| Pool budget: per-block income split by category caps, reserve, grants, closure | `pool.go` (`endBlockPool`) |
| Unit pricing: open race for units, equal per-unit payout at epoch end, last payout = bond basis | `pool.go` |
| Pool units as ordinary verified jobs (client = pool); refunds go back to the category | `poolClaim`, `refundClient` |
| Canary units: hidden canary tree, full check, curator dispute through the step-4 game | `canaryReveal`, `canaryCheck` |
| Watcher checks: a correct `CHALLENGE` costs only `WatchFee` | `agreedCheck` (dispute.go) |
| Forced errors: commitment + treasury reserve, reveal after decoys end, jackpots, bond reimbursement, expiry | `feCommit`, `feReveal` |
| Threshold shares of the decoy record (Shamir over GF(256)); work-list and canary trees for curators | `verifier/forced_errors.py` |
| Testnet metrics in the state root: settled stages, pool units and payouts, checks, decoys caught and missed, jackpots | `Stats` |

## Tests

| Suite | What it checks | Result |
|---|---|---|
| `TestCouncilThreshold` | 1 of 2 approvals doesn't execute; double, outside and expired approvals rejected; council actions can't be sent directly; proposals can't carry ordinary ops; caps above 100% rejected | pass |
| `TestPoolBudgetAndUnitPricing` | exact per-block allocation (60% to the budget, 40% to the reserve); bad proof, low bond and out-of-range claims rejected; no double claim; abandoned unit released, with its escrow back in the budget; 3 deliveries split the epoch budget exactly (payout, treasury share, price, stats); 1 delivery the next epoch gets the whole budget; delivered units can't be reclaimed | pass |
| `TestPoolCategoryExhaustionAndRetirement` | grant from the reserve; an exhausted category closes, returns its budget and frees its cap; a retired one stops claims and closes | pass |
| `TestCanaryUnits` | wrong salt and non-curator reveals rejected; all stages requested; honest provider passes and the curator's bond returns; a substituted model is disputed by the curator and fails, and the unit is released | pass |
| `TestWatcherCheckAgreesOrDisputes` | an agreeing check costs the watcher exactly `WatchFee`, which goes to the provider; a check of an open stage resolves at once; a wrong claim is a dispute and loses the whole bond; the job still settles | pass |
| `TestForcedErrorJackpots` | the reserve leaves the treasury; reveal refused while a decoy runs and with a wrong salt; caught decoy pays the watcher `Jackpot`, the decoy bond is reimbursed and the rest returns; missed decoy counted; a decoy accepted before the commitment doesn't count; unrevealed commitments expire | pass |
| `TestCanaryAndDecoyGamesOnTinyModel` (real games) | **canary vs. a substituted weight row** (`o`, layer 1): disputed at stage 1, one game, the provider can't produce a one-step proof against weights_root and is slashed, the curator gets half the bond. **Honest provider on a canary:** settles and is paid. **Forced error (one wrong leaf in stage 3) vs. a watcher** checking stages 0 and 3: stage 0 is an agreeing check, stage 3 goes through one game and one accepted recomputation, and after `FE_REVEAL` the watcher holds ½ bond − `WatchFee` + `Jackpot` | pass |
| `TestPoolAndWatchersRandomized` | 6 seeds × 500 blocks (+ drain): 300-unit category with 30 hidden canaries among dummies; honest, cheating (substituted model) and abandoning pool providers; ~100 decoys per seed with one wrong stage; committees of 2 × 2 samples; a watcher checking half of all client-looking jobs with one stage; verdicts from the true roots. After every block: conservation, replay determinism, independent clones, and every 5th block the state continues from a gob round trip. At the end: every job ended, **no honest provider slashed**, every canary-checked cheater failed, category and stats balance | pass |
| decoy catch rate (from the run above) | 586 decoys, 267 caught = **0.456**; predicted 1 − (C(7,2)/C(8,2))² × (1 − 0.5/8) = **0.473 ± 0.021** | within 1σ |
| `TestStep5Vectors` + `verifier/tests` (5) | Python tool ≡ Go for the record encoding, commitment, work-list and canary trees and proofs; any t of m shares reconstruct the secret and t − 1 don't; one share is uniform; CLI round trip | pass |
| Step 3/4 suites, node integration (blockchain, mining, mempool) | unchanged | pass (only the pre-existing `-tags zkpow` test fails) |

`jobstate` statement coverage: 92.1%. The full suite takes ~100 s (the randomized runs and the
gob round trips). `go test -short` takes ~40 s.

Per-seed numbers from the randomized run: 110–127 units settled and paid per seed; 12–15
canaries opened; 7–22 cheating units failed; 17–32 agreeing checks; 195–230 M paid in jackpots.

## Step-4 bug found and fixed (patch 0005)

`Job.clone` copied the dispute `Game` pointer and `DataReqs`' `Served` slices. A snapshot
therefore shared mutable state with the live state. The node evaluates competing blocks on clones
of one cached parent state, so evaluating one block during a dispute could change the parent
that another block is evaluated on. That breaks consensus. The state root also left out the game,
data requests, `SeqLen`, `ChallengeEnd` and the model architecture, so the bug was invisible to
the root-based replay check. Both are fixed. The harness now asserts after every block that the
parent snapshot's root is unchanged. With either aliasing reintroduced, the step-4 fault suite
fails.

## Choices and limits

- **The council is the trust point the doc names** ("who picks categories"). It is fixed in
  chain parameters; rotating it is a parameter change.
- **Epoch payouts:** with `PoolSpendBP` = 100%, the block after a payout has no budget to escrow
  verifier pay. Claims succeed again one block later, once new income arrives (tested). A lower
  spend share would smooth payouts across epochs. That is left to the step-6 retune.
- **Unit payment comes at the epoch end**, after the challenge window, so pool income arrives a
  day or more after the work. The bond basis is the last payout, which lags when the budget grows.
- **Decoy funding:** the keyholders fund decoys from their own coins, so nothing links them to
  the treasury before the reveal. Only a caught decoy's bond is reimbursed. A missed decoy's fee
  circulates back to the keyholders minus verifier pay and the treasury share.
- **Jackpot** is a fixed parameter, so watchers know it. Total payouts are capped by each
  commitment's reserve.
- **`FE_REVEAL` scans all jobs** (O(jobs)). An index by provider can come with the sparse Merkle
  tree.
- **Check delays:** each queued check holds settlement until the provider opens the stage, at
  most one `GameWindow`. New checks stop at the end of the challenge window.
- **Not built:** the watcher and keyholder *daemons* on a live chain (the strategies exist as
  test code). The simulator retune of ρ, k, n and bonds with measured λ is step 6.
