# Step 7 results: the devnet

Build-plan step 7: "Public testnet. Run real batch jobs from outside clients through the full
flow."

Agreed staging (2026-11-06): a **private devnet** first, with real cuPOW mining and synthetic
clients. Exposure to the internet and outside clients are a separate, later decision.

**Status:** the private devnet stage is done. Every mechanism of the design ran on a live chain,
and the run found eight problems that unit tests and the simulator did not. The public stage
(milestone 5) hasn't started. The devnet was stopped on 2026-11-06 at about block 355; its chain,
keys and agent state are kept on disk. Day-by-day notes: `spec/step7-progress.md`. Metrics page
(snapshot): https://claude.ai/artifact/764HEMAoNDH5DTwZdmyFgM

## What ran

| | |
|---|---|
| Network | `uwc-devnet`: own genesis and magic, full proof verification, one-minute target blocks, the job module with a 2-of-3 council. About 355 blocks. |
| Machines | Box 71 (3× RTX 3080); the 5090 box; .245 (2× RTX 3080 Ti, 3× RTX 3080). Three nodes, all on the LAN. |
| Mining | Our Ampere cuPOW kernels → Pearl's gateway (zk proof, about 22 s on 4 CPU cores) → `pearld`. Eight noise miners, up to about 340 TMAC/s. |
| Model | Qwen3-1.7B, registered with its weights root. 7 layer groups of 4 layers; 128-token sequences. |
| Agents (`runtime/agent`) | 2 providers (one mining while serving), 2 verifiers, 2 watchers (one lying on purpose), a client, a curator, a decoy runner, an off-chain data service |
| Tools | `uwcctl` (keys, Taproot-signed deposits and job ops, council proposals, one-step proofs); job-state RPCs; `agent/metrics.py` |

## Every mechanism, on the live chain

| Mechanism | What happened |
|---|---|
| Real blocks | Every block mined by the Ampere kernels and certified by Pearl's zk prover; every coinbase carried the pool deposit and the job-state commitment |
| Council (2-of-3) | Opened two pool categories and committed a forced-error record. A proposal waited at one approval until the second came. |
| Pool | Units claimed by Merkle proof from public work lists and computed by two providers racing for them. **12 units paid** at epoch ends. |
| Committees | Verifiers drawn by stake. They committed private salts and roots, then revealed, and recomputed sampled stages on a different GPU family (5090 vs 3080) **bit for bit**. |
| Client jobs | Posted with inputs off-chain. Outputs were delivered and checked against `output_root` every time. |
| Watcher checks | 9 agreeing checks cost only the watch fee |
| Dispute game | A lying watcher challenged a client job. Bisection over 76,320 leaves reached one element, and the provider's **one-step proof** was recomputed by every full node. The honest provider won and the liar lost its bond. |
| Canary | The curator revealed canary unit 7. All 7 stages were opened and matched, and the unit settled. |
| Forced errors | 5 decoys: 1 caught (by a watcher at the wrong stage) and 4 missed. After the last one ended, the record was revealed: a 5,000,000-grain jackpot paid, the decoy's bond reimbursed, the reserve returned. |
| Mining while serving | Every job GEMM hashed under the live block key. Roots stay **bit-identical** to unmined ones. |
| Totals | **27 jobs settled (217 stages)**; 5 failed: 1 caught decoy and 4 honest providers (below) |

## Metrics (design doc, "Testnet metrics")

| Metric | Devnet value | Target | Why |
|---|---|---|---|
| False-slash rate | **4 honest providers slashed**, all before the agent fixes | 0 | operational (problem 2) |
| Verification overhead | **1.8×** the provider's compute (29 jobs) | < 1% | 7-stage units; verifiers recompute every earlier stage (problem 5) |
| λ per job type, live | **1.6 × 10⁻⁴** (reference runtime, 3 jobs) | 0.37–0.49 measured in step 1 | the reference runtime's emulated float ops (problem 6) |
| Useful share | **9.5 × 10⁻⁷** | grows with demand and ρ | tiny jobs; only one provider mined while serving |
| Client price vs renting | **0.89×** (4 jobs) | < 1 | below 1, but the test client set the fees, so it says little about a market |

## What the devnet found

1. **Pool categories could freeze (consensus; fixed, patch 0011).** After a quiet period, the
   first delivered unit collected the category's whole accumulated budget (about 8,329 coins).
   Bonds were sized at half the last payout, so the next claim needed 4,165 coins and nobody
   could claim. Bonds are now sized against min(last payout, reference unit cost): a large
   withheld payout already deters cheating on its own. Regression test included.
2. **Honest providers slashed for operational reasons (fixed in the agent).** The rule that a
   silent provider loses turns any lost transaction into a slash.
   - One provider missed a 10-block reply window while computing another job single-threaded.
   - Three games were lost when both nodes were restarted at once. `pearld` doesn't keep its
     mempool across a restart, so the provider's pending moves vanished, and the agent never
     resent them.

   Fixes: computation on a worker thread, and a resend of any move, reveal, commit or verifier
   op the chain hasn't recorded after 3 blocks. Operators restart nodes one at a time.
3. **A fresh node could not sync the devnet.** I changed job-module rules on the running chain
   three times (the challenge window twice and the bond basis), without activation heights.
   Nodes A and B applied each change from their restart and stayed in agreement, but node C,
   validating from genesis, rejected block 99. Node C was started from a copy of node A's data.
   **Any change to job-module rules needs an activation height**, as Pearl already does for its
   own forks, and the testnet starts from a fresh genesis.
4. **The final game round was skipped by the agent (fixed).** Entering the one-step phase, the
   chain leaves the provider's "moved" flag set from the previous round, so the agent thought it
   had already moved. The first live one-step proof was built by hand, 5 blocks before its
   deadline.
5. **Verification costs 1.8× instead of under 1%.** Two causes:
   - with 7-stage units, k·n/S is already 4/7;
   - verifiers recompute every earlier layer group from the tokens instead of fetching the
     stage input, which the protocol supports (`DATA_REQUEST`, serve or be slashed).

   Real jobs need hundreds of stages, and verifiers should fetch stage inputs.
6. **The reference runtime cannot be a production runtime.** It hashed 180 GMAC in about
   24 s (0.007 TMAC/s) against 47–65 TMAC/s of noise mining: λ ≈ 1.6 × 10⁻⁴. Providers need a
   fast fused runtime with the same op order (the step-2 caveat, now measured on live jobs).
7. **Reply windows measured in blocks shrink when hashrate rises.** Blocks ran 35–85 s apart
   because hashrate changed as GPUs were shared with agents and retargeting lagged. Testnet
   reply windows should be at least 30 blocks.
8. **Pearl's gateway needed two changes** for this chain:
   - it builds its own coinbase from `coinbasevalue`, so it now appends the job module's
     required outputs, which `getblocktemplate` returns as `uwc_coinbase_outputs`;
   - it caches its template per height, so it must restart whenever its node restarts.

## Code

| Where | What |
|---|---|
| `chain/patches/0008`–`0012` | devnet params; job-state RPCs; job coinbase outputs; gateway support; `uwcclient` and `uwcctl`; `BuildOneStep` with a file-backed weights store; pool bond basis; devnet windows |
| `runtime/agent/` | roles (provider, verifier, watcher, client, decoy, curator); data service; game moves; mining while serving; council forced-error and canary tooling; metrics; tested against Go by `TestAgentVectors` |
| `miner/ampere/uwc_ampere/miner_loop.py` | the noise miner for Pearl's gateway |
| `devnet/` | per-box scripts: `up.sh`, `down.sh`, `agent.sh`, `data.sh`, `ctl.sh`, `uwc.sh`; a copy of the metrics page |

Tests added in this step (all pass): `TestBuildOneStepDefeatsALyingChallenger`,
`TestPoolBondAfterABigPayout`, `TestAgentVectors`, `TestLargeJobsNeedLargerCommittees`; node
tests pass under `-tags xmss,zkpow`.

## Before a public testnet

1. A fresh genesis with final rules, and activation heights for any later rule change.
2. Reply windows of 30 or more blocks, plus either mempool persistence in `pearld` or rebroadcast
   in every client.
3. Verifiers fetch stage inputs; jobs have hundreds of stages.
4. A fast provider runtime with the reference op order, so live λ approaches the step-1 figures.
5. A live metrics endpoint instead of a snapshot page.
6. **Your decision** on exposure (which ports, through what), hosting (home LAN boxes behind NAT
   or a VPS for seed nodes), and the outside clients and their job types.

## Restarting the devnet

On each box: `devnet/up.sh <gpus>`, then `devnet/agent.sh <role> <key> …` (`data.sh` on box 71
first). Start nodes one at a time. Node C syncs only from a copy of node A's data directory
(problem 3).
