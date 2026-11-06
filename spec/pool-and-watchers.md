# Pool, canaries, watchers and forced errors (build step 5)

Consensus code: `node/jobstate/pool.go` (patch 0006). Keyholder and curator tools:
`verifier/forced_errors.py`. Like the rest of the job module, none of this affects block
validity. A failed operation is a no-op.

## Council (the threshold keys)

`Params.Council` is a fixed list of m keyholder accounts with threshold t (`CouncilThreshold`).
A member sends `PROPOSE(action)`. Other members send `APPROVE(proposal)`. When t approvals have
arrived, the action runs as `CouncilAccount`, which no pkScript can sign for. If the action
fails, the approval that would have run it fails as well, so it can be retried. A proposal that
hasn't run expires after `ProposalWindow`. Actions:

| Action | Effect |
|---|---|
| `POOL_CATEGORY(spec)` | opens a category; takes `spec.Grant` from the pool reserve |
| `POOL_RETIRE(category)` | stops new claims and new income; the category closes once nothing is in flight |
| `FE_COMMIT(commitment, reserve)` | moves `reserve` from the treasury into a forced-error commitment |

With an empty council (the default, and UWC SimNet), there are no categories and no forced errors.

## Public-good pool

Every coinbase deposits ⌊ρ · subsidy⌋ to `PoolAccount` (step 3). At the end of each block, that
block's pool income is split between the open categories by `CapBP`. Caps of open categories
sum to at most 100%, and the remainder stays in the pool **reserve**, which funds the grants of
new categories.

A **category** is the design's "fixed job category, public job list":

| Field | Meaning |
|---|---|
| model, `WorkRoot`, `Units` | Merkle root over `UnitLeaf(i, input_root_i)`: the public work list |
| `StagesPerUnit`, `SeqLen`, k, n | every unit's shape and verification level |
| `UnitDeadline` | blocks from claim to `TRACE_COMMIT` |
| `CapBP`, `Grant`, `InitPrice` | income share, starting budget, bond basis until the first payout |
| `Curator`, `CanaryRoot`, `CanaryLeaves` | canary commitment (below) |

**Open race for units.** `POOL_CLAIM(category, i, input_root_i, proof, bond)` from any account:

- unit i must be free;
- the proof must open the work list;
- bond ≥ `MinBondBP` × the category's current unit price.

The claim escrows verifier pay V = k·n·`VerifierFloorPerSample` from the category budget and
creates an ordinary job in `accepted` state. Its client is the pool, and the fee is V. From
there it is verified exactly like a client job (committee, salts, commit-reveal, challenges,
games). Each unit can be delivered once. A unit that fails or is abandoned is released for
another claim, and whatever the job owed its client (V, plus half the bond of a provider that
never committed) goes back to the category budget.

**Unit pricing.** Units that settle during an epoch (`PoolEpoch` blocks) are paid at the
epoch's last block:

  per-unit payout = ⌊budget × `PoolSpendBP` ÷ units delivered⌋

The provider receives the payout minus the 1% treasury share. The category's price becomes that
payout, and the price sets the bond for new claims. With no deliveries, the budget carries over.
Fewer providers means more per unit, with no price to set (PoUI). Nobody can post a pool job, so
self-dealing has nothing to capture. A category that is retired or fully delivered, with nothing
in flight, closes at an epoch end and returns its budget to the reserve.

## Canary units

The curator knows the expected stage roots of a few units and commits, with the category:

  CanaryRoot = MerkleRoot(leaves)   leaf = H("canary" ‖ unit ‖ MerkleRoot(expected roots) ‖ salt)

The leaves are shuffled among random dummy leaves, so neither the canary units nor their number
can be read from the tree.

After the provider's `TRACE_COMMIT`, the curator sends `CANARY_REVEAL(job, leaf index, salt,
expected roots, proof)` and locks `ChallengeBondMin`. This is allowed from `committed` up to the
challenge window, but not during a dispute. A revealed canary is **checked in full**: every
stage becomes requested, and a canary revealed in the challenge window reopens the reveal phase.
Once all stages are open:

- every root matches: the curator's bond comes back;
- otherwise, the first differing stage is disputed by the curator, with the canary bond as the
  dispute bond, and the step-4 game settles it down to one operation.

A curator's wrong expected root therefore can't slash an honest provider. The curator loses the
game and its bond instead. A substituted model differs at every stage, so it is caught with
certainty.

## Watchers

Watchers use `CHALLENGE` as a sampled check. A watcher recomputes a stage (the runtime is
deterministic, and the job inputs are public) and posts its root with a bond. The claim is
binding before the provider opens the stage, so a watcher can't copy the provider's root.

| Provider's opened root | Outcome |
|---|---|
| equals the claim | **check**: `WatchFee` goes to the provider for opening the stage, the rest of the bond comes back |
| differs | dispute game; the loser pays as in step 4 |
| never opened | the provider loses (`GReveal` deadline) |

A check of an already-open stage resolves in the same block. The provider can answer many
queued checks with one `TRACE_REVEAL`.

## Forced errors (decoys)

1. **Commit.** The keyholders pick fresh decoy provider accounts. The record is those accounts;
   a 32-byte salt is drawn. They split salt ‖ record t-of-m (Shamir over GF(256),
   `forced_errors.py new`) and get `FE_COMMIT(H("fe-record" ‖ record ‖ salt), reserve)`
   approved. The reserve leaves the treasury.
2. **Run.** From fresh client accounts they post ordinary jobs. The decoy providers accept them
   *after* the commitment, commit a trace with one deliberately wrong stage root, and otherwise
   behave like honest providers. The chain can't tell decoys from client jobs, and neither can
   watchers.
3. **Catch.** A committee verifier that samples the wrong stage, or a watcher that checks it,
   wins the dispute game as usual and takes half the decoy's bond. The job records its catchers.
4. **Reveal.** After every decoy job has ended, any t keyholders combine their shares
   (`forced_errors.py reveal`), and anyone posts `FE_REVEAL(commitment, salt, record)`. For each
   job whose provider is listed and which was accepted after the commitment:
   - caught: `Jackpot` is split between the catchers, and the decoy's bond is reimbursed to the
     keyholders;
   - missed (settled): counted in `Stats.DecoysMissed`.

   Both are capped by the reserve, and the rest returns to the treasury. An unrevealed
   commitment expires after `FERevealWindow`, and its reserve returns.

What the threshold buys: a single keyholder can't fund jackpots, because commitments need t
approvals and each is capped by its reserve. Revealing needs t shares, so no single party has to
keep the secret. What remains (design doc, security table): the keyholders who run the decoys
know them and could watch only decoys, starving honest watchers. Truebit has the same weakness.

## New parameters (defaults)

| Parameter | Default | Note |
|---|---|---|
| `PoolEpoch` | 60 blocks | one hour at one-minute blocks |
| `PoolSpendBP` | 10000 | the whole category budget is paid out at each epoch with deliveries |
| `WatchFee` | 10,000 | 1% of `ChallengeBondMin` |
| `Jackpot` | 5,000,000 | 50 × `ChallengeBondMin` |
| `FERevealWindow` | 20,160 blocks | two weeks |
| `Council`, `CouncilThreshold` | empty, 0 | set per network |
| `ProposalWindow` | 1440 blocks | one day |

## Metrics in the state root

`Stats`: settled jobs and stages (the numerator of the doc's *useful share*), pool units and pool
payouts, agreeing checks, decoys caught and missed (the watch rate w), and jackpots paid.

## New operations

| Op | Fields |
|---|---|
| `PROPOSE` | encoded action (≤ 4 KB) |
| `APPROVE`, `POOL_RETIRE` | proposal or category id |
| `POOL_CATEGORY` | the category spec |
| `FE_COMMIT` | commitment, reserve |
| `POOL_CLAIM` | category, unit, input root, bond, Merkle proof |
| `CANARY_REVEAL` | job, leaf index, salt, expected roots, Merkle proof |
| `FE_REVEAL` | commitment id, salt, record |
