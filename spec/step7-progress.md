# Step 7 progress: devnet (private), then public testnet

Build-plan step 7: "Public testnet. Run real batch jobs from outside clients through the full
flow." Agreed staging (2026-11-06): a **private devnet** on our two GPU boxes first, with **real
cuPOW mining** and synthetic client jobs. Opening anything to the internet and choosing outside
clients is a later, separate decision.

## Milestones

| # | Milestone | Status |
|---|---|---|
| 1 | Real blocks: devnet params, `pearld` with proof verification, Ampere miner → Pearl gateway (zk proof) → node | **done** |
| 2 | Job tooling: `uwcctl` deposits and job ops (Taproot-signed), council actions | **done** |
| 3 | Provider, verifier and watcher daemons on the GPUs (deterministic runtime), client CLI with off-chain output delivery | **running**; see below |
| 4 | Testnet metrics page: useful share, λ per job type, verification overhead, false-slash rate, client price | **done** (snapshot): `runtime/agent/metrics.py`, page https://claude.ai/artifact/764HEMAoNDH5DTwZdmyFgM |
| 5 | Public testnet: exposure, outside clients | needs a decision |

## Milestone 1: real blocks (2026-11-06)

- **Proof path.** Pearl's zk prover (`pearl_mining.generate_proof_for_cert_version`, v3) certifies a
  plain proof mined by our Ampere kernels: 21.9 s on box 71's 4-core CPU. Pearl's verifier
  accepts it. One-minute blocks leave room for the proof.
- **Node.** `pearld` built with `-tags xmss,zkpow` (Rust zk-pow builds with cargo 1.99). Under
  that build every node test passes, including `TestCheckBlockSanity` and the wire certificate
  tests, which fail without the tags.
- **Network** `uwc-devnet` (`--uwcdevnet`, patch 0008):
  - own magic and genesis; proof verification on (SimNet skips it by magic);
  - 1-minute blocks, WTEMA retargeting with a 1-hour half-life;
  - easiest target 2^212, genesis target 2^201;
  - coinbase maturity 10;
  - job module with a 2-of-3 council. Council keys and miner keys are in
    `~/.uwc/devnet/keys` on the laptop, not in git.
- **Coinbase.** Pearl's gateway builds the coinbase from `coinbasevalue`, so it left out the
  job-module outputs, and the first blocks were rejected ("coinbase deposits 0 to the pool").
  `getblocktemplate` now returns `uwc_coinbase_outputs` (pool deposit, due withdrawals,
  state commitment), and the gateway appends them.
- **Topology:**
  - Node A on box 71 (192.168.29.71) with its gateway, mining with 3× RTX 3080.
  - Node B on the 5090 box (192.168.29.50) with its gateway, mining with the RTX 5090.
  - P2P on LAN addresses only; RPC on 127.0.0.1. Scripts in `devnet/`.
- **Miner.** `miner/ampere/uwc_ampere/miner_loop.py` speaks the gateway protocol
  (`getMiningInfo`, `submitPlainProof`). It verifies each proof locally before submitting,
  reconnects when the gateway restarts, and logs noise rates for the λ metric
  (`devnet/logs/noise-gpu*.jsonl`). Measured rates: 66–68 TMAC/s per 3080, 218–278 TMAC/s on the
  5090.
- **Result:** both nodes at the same height with accepted blocks, each certificate verified by
  both nodes, and each coinbase passing the pool and state-commitment rules.

Lessons:
- The gateway caches its template per height, so after a node restart it must be restarted too.
- The genesis timestamp is the start of the day, so the first block's 9-hour gap eased the
  difficulty. It recovers with the 1-hour half-life.

## Milestone 2: job tooling on the real chain (2026-11-06)

- **`node/uwcclient`** (patch 0009), shared by `uwcctl` and the daemons to come:
  - Taproot key-path keys;
  - coins found through the node's address index, skipping immature coinbases and coins
    already spent in the mempool (`gettxout` doesn't see mempool spends; `gettxspendingprevout`
    does);
  - transaction signing, and several job ops in one transaction (all inputs from one key, so
    one actor);
  - op references computed as the node computes them: sha256(txid ‖ output index).
- **On the devnet:**
  - The miner deposited 200 coins into its job-module account.
  - The council members and the curator were funded with ordinary coins.
  - Qwen3-1.7B registered (model `12abc45b…`): weights root `89af7bfb…` over 726,512 canonical
    rows, computed in 4.3 min on CPU by `runtime/agent/model_info.py`; 7 layer groups of 4
    layers; Hadamard block 512; up to 512 positions.
  - The 2-of-3 council opened pool category `c951e9ad…`. Its work list
    (`runtime/agent/worklist.py`) is 200 units, each 128 tokens of WikiText-2 test text with 7
    stages, and it gets 60% of pool income. The proposal sat at one approval until the second
    member approved; then the category opened and started collecting about 6 coins per block.

## Milestone 3: the full job flow on the devnet (2026-11-06)

**Agent** (`runtime/agent/`, Python). It runs the reference runtime on the GPUs and sends
chain operations through `uwcctl`:

| Role | Where | What it does |
|---|---|---|
| provider (miner71) | box 71, GPU 1 | claims pool units, accepts client jobs, computes every stage, commits salted roots and the output root, uploads outputs, reveals, plays games, builds one-step proofs |
| verifier0, verifier1 | box 71 GPU 2; 5090 | stake, then per seat: private salt, recompute the sampled stages, commit, reveal; play games they open |
| watcher0 | 5090 | checks half of the jobs in their challenge window at one random stage |
| watcher1 (fault injection) | 5090 | the same, but every check claims a root with one element flipped |
| client0 | box 71 | posts embedding jobs over WikiText-2 and checks delivered outputs against `output_root` |
| decoy0 + decoyclient | box 71, GPU 2 | forced errors: posts and accepts its own jobs with one wrong stage root, never defends |
| curator | 5090 | reveals canary units of the canary category |

Data service: `agent/data.py` on 192.168.29.71:18680, LAN only, for job inputs and outputs.
The Python protocol helpers match Go (`TestAgentVectors`).

**What ran on the real chain (real cuPOW blocks):**
- **Pool and verification:**
  - Pool units claimed by Merkle proof against the work list and computed: 7 stages of 128
    tokens in about 44 s on a 3080.
  - Committees drawn from stake. The 5090 verifier recomputed sampled stages and matched the
    3080 provider bit for bit; cross-GPU determinism holds on live jobs.
  - Requested stages revealed. The first pool unit settled and was paid at its epoch end.
- **Client jobs:** outputs delivered off-chain; the client verified every one against its committed `output_root`.
- **Watchers:** a check of an opened stage matched, so nothing was posted. Other checks went on chain.
- **Games:** the lying watcher opened real bisection games over Qwen3-1.7B stage traces of about
  76,000 leaves, played move by move by both agents.
- **Forced errors:** the council's 2-of-3 commitment is on chain with a 0.6-coin reserve, and decoy jobs were posted and accepted.
- **Canaries:** category `29b34b31…` (100 units, canaries at units 7, 15, 55, 71 and 95 among 20 leaves) opened by council0 and council2.
- **Weights store:** `agent/weights_store.py` (2.6 GB) matches the registered weights root;
  `uwcctl onestep` builds one-step proofs from it (`TestBuildOneStepDefeatsALyingChallenger`).

**Problems the devnet found, and fixes:**

1. **Pool categories could freeze (consensus; fixed in patch 0011).** After a dry spell, the
   first delivered unit collected the whole accumulated budget (about 8,329 coins). Bonds were
   sized against the last payout, so the next claim needed a 4,165-coin bond, and nobody could
   claim. Bonds are now sized against min(last payout, the category's reference cost); the
   withheld payout already deters cheating. Regression test `TestPoolBondAfterABigPayout`.
2. **False slashes from agent liveness (operations).** All four honest-provider slashes so far
   are operational:
   - One provider missed a 10-block reply window while computing a 5-minute job
     single-threaded. Fixed: jobs compute on a worker thread; replies come first.
   - Three games were lost after both nodes were restarted at once. `pearld` doesn't persist
     its mempool, so the provider's pending moves vanished, and the agent had marked them as
     sent. Fixed: agents resend a move, reveal or commit the chain hasn't recorded after 3 blocks.

   Rule for operators: restart nodes one at a time.

   Protocol lesson: the silence rule ("a silent provider loses first") turns any lost
   transaction into a slash. Reply windows in blocks shrink in wall time when hashrate rises
   (38.5 s blocks at the moment). The testnet needs windows of 30+ blocks and agents that
   rebroadcast.
3. **Verification overhead is 1.9× the provider's compute, not the design's 0.4%.** Units have
   only 7 stages (k·n/S = 4/7 even ideally), and verifiers recompute every earlier layer group
   from the tokens instead of getting the stage input served (`DATA_REQUEST`). Testnet jobs need
   S in the hundreds, and verifiers should fetch stage inputs.

**More results (block 287):**
- 11 jobs settled (84 stages), including a client job.
- **A full dispute game on the live chain.** The lying watcher challenged client job `4d7d80b4`
  at stage 9. The game ran about 17 bisection rounds over 76,320 leaves to leaf 28,146, then the
  row's blocks, then element 10. The provider's one-step proof (`uwcctl onestep`, one proven
  input row, weights from the store) was recomputed by every full node. The honest provider won,
  the liar lost its bond, and the job settled. The proof was built by hand on the 5090 because
  the running agent had a bug: in the one-step phase the chain leaves "provider moved" set from
  the previous round. Fixed in the agent.
- Four pool units paid; one category is drained and refilling; the canary category is being claimed.
- Four decoys posted. One settled unnoticed (a missed decoy); three are in their windows. The
  decoy runner reveals the forced-error record itself once all its decoys have ended, so jackpots
  are paid without anyone stepping in.

**Testnet metrics (snapshot, block 284; `agent/metrics.py`, page above):**

| Metric | Value | Reading |
|---|---|---|
| False slashes | 4 of 4 failures | all operational (missed window, moves lost in a restart of both nodes); fixed in the agent |
| Verification overhead | 2.4× provider compute (19 jobs) | 7-stage units and recompute-from-tokens; needs large S and served stage inputs |
| Client price vs renting | 0.26× (1 job) | the test client's fee was below the provider's mining opportunity cost: a mispriced job, not a saving |
| Useful share | 3.8 × 10⁻⁷ | tiny jobs; providers don't mine during jobs yet |
| Network | 340 TMAC/s, 44.8 s blocks | 3080s at 61–66 TMAC/s each when not shared with agents |

## A third box and providers that mine (2026-11-06)

**Node C** on 192.168.29.245 (wired; the same host is 192.168.29.13 on Wi-Fi): 2× RTX 3080 Ti
12 GB and 3× RTX 3080 (10 and 20 GB). It runs a node, a gateway, noise miners on three 3080s and
one 3080 Ti, and a second provider (`provider245`) that mines while serving. The environment was
copied from box 71 (uv Python, venv, Qwen3-1.7B, weights store).

**A fresh node could not sync the devnet.** Node C, validating from genesis with the current
binary, rejected block 99: "coinbase must commit job-state root". It then banned node A. Job-module
rules were changed three times on the running chain (the challenge window 1440 → 60 → 20, and
the pool bond basis), without activation heights. Nodes A and B applied each change from the
block they restarted at, so they agree with each other, but the history isn't valid under any
single rule set.
- Node C was bootstrapped from node A's data directory (node A stopped briefly; node B kept the chain).
- Rule for the testnet: start from a fresh genesis with final rules. Any later change to job-module
  rules gets an activation height, as Pearl does for its own forks (`MoEForkHeight`,
  `SaltedSeedForkHeight`).

**Mining while serving** (`agent/mining.py`, provider `--mine`):
- Every int8 GEMM of a job runs Pearl's noisy GEMM under the gateway's current block key; layers
  are re-keyed when the template changes.
- Winning tiles are proven, verified locally and submitted like a miner's.
- Hashed MACs are counted per job, so λ is measured live.
- The reference runtime is not modified (its hash is registered on chain); the agent wraps it.
- **Bit-exact:** pool unit 0 computed with mining on, on a 3080 Ti, gives exactly the roots the
  3080 provider committed without mining.
- **Live λ of the reference runtime is about 1×10⁻⁴.** It hashed 180 GMAC in 25 s (0.007 TMAC/s)
  against about 65 TMAC/s of noise mining. Step 1's fast fused path reached 0.37–0.49. Emulated
  float ops and per-row hashing dominate, so production providers need a fast runtime with the
  same op order (the step-2 caveat, now measured on live jobs).

**State at block 324 (three boxes):**
- 24 jobs settled (196 stages); 12 pool units paid; 8 noise miners at 304 TMAC/s.
- **Two providers in the open race:** box 71's 3080, and the mining provider on .245's 3080 Ti
  (about 24 s per unit).
- **First canary:** the curator revealed canary unit 7; all 7 stages were opened and matched the
  expected roots, with no dispute.
- **First caught decoy:** the committee sampled stages 0, 2 and 4 of decoy `606731b3`. The honest
  watcher checked stage 3, the wrong one; the game opened and the decoy (which never defends)
  failed. Three other decoys went unsampled (missed). Jackpots are paid when the decoy runner
  reveals the record after its last decoy ends.

**Metrics (`agent/metrics.py --extra` with agent states from all boxes; page updated):**

| Metric | Value |
|---|---|
| live λ (reference runtime, 3 mining jobs) | 1.6 × 10⁻⁴ |
| verification overhead | 1.8× provider compute (29 jobs) |
| client price vs renting | 0.89× (4 client jobs) |
| false slashes | 4, all operational, the last before the agent fixes |
| useful share | 9.5 × 10⁻⁷ |

**Forced errors, end to end (block ~350):** the last decoy ended at 15:05 and the decoy runner
revealed the council's record in the next transaction. The chain counted 5 decoys: 1 caught
(by the watcher that checked stage 3 of `606731b3`) and 4 missed. It paid the 5,000,000-grain
jackpot to that watcher, reimbursed the decoy's bond, returned the rest of the reserve to the
treasury and removed the commitment. Canary unit 7 settled. 27 jobs settled in total
(217 stages).

The catch rate, 1 in 5, fits the odds for one wrong stage in 7 with two verifiers sampling two
stages each and a watcher checking half the jobs at one stage: about 0.49 + 0.51 × 0.5/7 ≈ 0.53
per decoy. With five decoys the sample is far too small to test that.

## Testnet fixes (2026-11-06, after the devnet was stopped)

| Fix | State |
|---|---|
| Rule changes with activation heights | **done** (patch 0013): `Params.Upgrades` and `Params.At(height)`; code-level changes versioned by `Params.Rules` (bond basis = Rules 1). Tests: `TestUpgradesActivateAtHeight`, `TestRulesVersionGatesBondBasis` |
| Fresh testnet with final rules | **done** (patch 0013): `--uwctestnet`, new genesis and magic, 30-block reply windows, 240-block challenge window, a new 2-of-3 council (keys in `~/.uwc/testnet/keys` on the laptop; should move to the user) |
| Pending transactions survive restarts | **done** (patch 0013): `pearld` writes `mempool.dat` on shutdown and re-admits it on startup. Round-trip test; a live check is still to do |
| Verifiers fetch stage inputs | **done** (agent): the provider keeps each later stage's input with Merkle proofs and opens the previous stage on request. Verifiers check it against the on-chain commitment and recompute one stage: 3.1 s instead of 15.8 s, same root; tampered rows are rejected |
| Live metrics | **done** (agent): the data service serves `/` and `/metrics.json`, refreshed by `devnet/metrics.sh` |
| Fast provider runtime | **partly done**: `detrt/fast.py` batches a job's sequences, copies each op's output once and hashes rows in parallel. Every row and root is identical to `run_stage` (`tests/test_fast.py`). The provider is 2.9× faster (6-sequence job: 134 s → 46 s) |

**Profile of the reference runtime** (one 4-layer stage, 128 tokens, 3080 Ti, 3.24 s): 61% copying
76,448 rows one at a time to the CPU, 28% emulated fp ops, 11% hashing, about 0% int8 GEMMs.

**Finding: full-trace commitments cap λ.** Each stage root commits to every op's output: about
76,000 rows and roughly 250 KB of trace per token-layer. Even hashed on a GPU at about 20 GB/s, that
is roughly 25× the time of the int8 GEMMs it describes, so λ stays near or below 0.04 however fast
the runtime is.

Proposed protocol change, not yet made:
- Providers commit only each stage's **output** rows. A full trace is computed and committed only
  when a stage is disputed, bound to the committed input and output.
- The bisection argument still holds: with input and output fixed, a wrong output forces a wrong
  op somewhere in the trace.
- Verifiers compare output roots, which is also cheaper.

This changes `TRACE_COMMIT`, the verifier result hashes and the start of a dispute game
(consensus), and the agents.

## Where to resume (2026-11-06, end of session)

The devnet keeps running on its own: nodes, gateways, miners (box 71 GPUs 0 and 2, the 5090),
and the agents. `devnet/up.sh` and `down.sh` start and stop a box; `agent.sh <role> <key>` runs a
role. Restart nodes one at a time. Still due, all automatic:
- canary units 7, 15, 55, 71 and 95 are claimed and revealed by the curator;
- the remaining decoys end and the forced-error record is revealed (jackpots for whoever caught them);
- the remaining client jobs settle.

Rerun `python -m agent.metrics --json` on box 71 and refresh the page's datasets to update it.

Next:
1. Providers mine while serving: run jobs with mining on, with the block key from the gateway,
   and submit winning tiles. This makes λ per job type a live metric.
2. Verifiers fetch stage inputs (`DATA_REQUEST` and serve) instead of recomputing prefixes, and
   jobs have hundreds of stages, so verification overhead approaches the design's k·n/S.
3. Testnet parameters: reply windows of 30+ blocks; mempool persistence or rebroadcast in `pearld`.
4. Milestone 5 (public) needs the user's decision on exposure, hosting and outside clients.
