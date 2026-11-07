# Step 7: first private testnet run

Run of `uwc-testnet` (fresh genesis, height-gated rules, patches through 0015) on the LAN, from
2026-11-06 16:20 to 2026-11-07 13:30, blocks 0 to about 1,225. Every agent role ran with real
cuPOW mining. Metrics file: `results/testnet-run1-metrics.json` (block 1224). Setup steps and
earlier notes: `spec/step7-progress.md`.

**Status:** every mechanism ran and settled on the live chain, and no honest provider was
slashed. The run found three design problems and three gaps in the node and agents (listed
below), and it confirms that λ (hashing kept while serving jobs) is the main gap to the design.
The chain is still running and the miners are still on; stop it with `down.sh` on each box.

## What ran

| | |
|---|---|
| Nodes | Box 71 (3× RTX 3080) and .245 (4× RTX 3080, 1× 3080 Ti). The 5090 box was stopped at the user's request and then dropped off the LAN, so no 5090 results this run. |
| Mining | 6 noise miners on 3080/3080 Ti (47.7–67.3 TMAC/s each) plus two providers that mine while serving. Network ≈ 316 TMAC/s; 57.6 s per block over the last 120 blocks (target 60 s). |
| Agents | providers `miner71 --mine` (71, GPU 1) and `provider245 --mine` (.245, GPU 0); verifiers `verifier0` (71) and `verifier1` (.245); watchers `watcher0` and `watcher1 --lie` (.245); `client0`, `curator` and `decoy0` (71). |
| Work | Pool category `wiki8x128` (100 units × 8 sequences × 7 stages) and the canary category (100 units, hidden canaries); 10 client jobs of 16 sequences (112 stages, fee 30 coins); 9 decoy posts. |

## Results

| Mechanism | Result |
|---|---|
| Settlement | **209 jobs settled, 11,746 stages.** First settlement at block 332, the first block after its 240-block challenge window. |
| Pool | Main category **100 of 100 units done**; canary category **97 of 100**. 84,639 coins paid out over the run. Treasury 848 coins at the end, filled only from pool payouts. |
| Canaries | At least 5 canary units revealed by the curator (counted in its current log), so every stage was checked. All passed: no canary job failed. |
| Disputes | `watcher1` lied 109 times and **lost every game**. Each game went from the top round down to a one-op proof, which the honest provider won. Its balance fell from 5.0 to 4.993 coins. |
| False slashes | **0.** The only 2 failed jobs were decoys. |
| Served inputs | Verifiers fetched stage inputs from providers and checked them against the commitment. Inputs that didn't match, from a provider working on a job it had lost, were rejected, and the verifier recomputed. |
| Forced errors | Commitment approved once the treasury held the 0.6-coin reserve. **2 decoys caught, 2 missed**, 0.1 coin of jackpots paid. Details below. |
| Node restarts | Each node was restarted with a deposit pending. Logs: `Saved 1 mempool transactions` / `Restored 1 of 1`. Node back in 4–5 s and peered; the gateway recovered in under 10 s; agents carried on; both deposits confirmed. |

### Forced errors, decoy by decoy

| Decoy | Wrong stage | Outcome |
|---|---|---|
| `3fef7b8f` | 6 | **False catch.** `watcher1` lied about stage 2, which was correct. The decoy didn't defend (as written then), and a silent provider loses any game, so the liar became the catcher and got a jackpot share. |
| `2b02aa19` | 5 | **Caught** by `verifier1`, whose committee sample included stage 5. |
| `60f21890` | 2 | Missed: the committee checked stages 1, 3 and 4. |
| `1ab9d01e` | 4 | Missed. It also defended a lying challenge on stage 0 and won it with a one-op proof, after the fix below. |
| `1c31184d`, `cec55784`, `c65c3039`, `97c944ca` | – | **Lost the accept race**: an honest provider's accept was mined first, so these ran as ordinary jobs. |
| `1598bd62` | – | Dropped before it was accepted, so the reveal could go ahead without waiting about 4.5 h. |

A first commitment was wasted: the decoy was started without `--post`, which defaults to 0, so it
revealed the record at once with no decoys. Its reserve went back to the treasury. The commitment
used was `daac3556`, opened at block 460 and revealed at block 1209.

## Metrics (block 1224)

| Metric | Value | Design | Note |
|---|---|---|---|
| λ live (median of 206 mining jobs) | **0.012** | 0.4–0.49 measured compiled | A job hashes 0.72 TMAC/s against 65 TMAC/s for noise mining. The deterministic runtime is bound by kernel launches (about 68k tiny kernels per 16 sequences). |
| Provider time per 56-stage unit | 2.0 s median | – | `detrt/fast.py`, mining on |
| Verification overhead | 6.1× median | k·n/S ≈ 0.07 | Verifier median 9.45 s for 2 stages. That is input transfer (JSON over HTTP) and recomputing earlier stages when inputs aren't served, not stage compute. |
| Client price vs renting | 23× | < 1 | Not a market price: the test client set a 30-coin fee for about 3 s of GPU time. |
| Useful share | 1.4e-5 | ρ/(1−λ) of capacity | Follows from λ and the job volume. |
| False-slash rate | 0 of 209 | 0 | |

## Design findings

1. **A silent provider loses any game, so a forced-error jackpot can pay a liar.** See decoy
   `3fef7b8f`: the liar was paid for an error it never found. The agent-side fix (decoys defend their
   honest stages) works: `1ab9d01e` and `60f21890` both beat lying challenges. But a real keyholder
   could make the same mistake, and the chain can't tell. **Proposed consensus fix:** the
   forced-error record also lists each decoy's wrong stage, and the jackpot goes only to a
   challenger whose dispute was on that stage.
2. **Honest providers race decoys for the accept.** 4 of the 8 decoys that were accepted went to
   an honest provider, because both accepts arrive within 1–2 s and senders can't set a fee to get
   ahead. Decoys must look like ordinary jobs, so this would happen on a public network too, and it
   cuts the decoy sample in half. **Proposed fix:** let any client post a job with a named provider
   (a "directed job"). Decoys then hide among ordinary directed jobs, and their accept can't be
   raced.
3. **A job operation that fails inside a mined transaction leaves no trace.** `ApplyBlock` keeps
   the error in `BlockResult.OpErrors`, but nothing logs it or serves it over RPC. Four client jobs
   (posted at blocks 228–318) expired although miner71's accept transactions were mined (now about
   1,000 confirmations). The most likely cause is that its module balance didn't cover the 15-coin
   bond while pool-claim bonds were locked, before the first payouts. This is an inference, since
   the error isn't visible. The agent read "tx left the mempool, job not mine" as a lost race and
   moved on. **Fix:** a per-transaction job-result RPC plus a debug log line; agents check their
   balance before bonding.
4. **The watcher bond is too small to deter lies.** The default bond is 0.001 coin. 109 lies
   cost `watcher1` about 0.007 coin in total, but each one made the provider play 15–17
   transactions over 35 minutes to 3.5 hours, ending in a one-op proof. Size the minimum challenge bond to the
   defender's cost of a full game, not to the watch fee.

## Node and agent gaps

- `uwcctl job` doesn't show `Catchers` (`rpcjobs.go`), so who caught a decoy is visible only in
  the totals.
- Decoys don't serve stage inputs, so verifiers wait 90 s and then recompute earlier stages.
- Verification overhead is dominated by input transfer: serve rows as binary, and batch a
  verifier's sampled stages.

## Agent fixes made during the run (`runtime/agent`, not yet committed)

- `compute.py`: `Runtime.bind()` sets the torch and CuPy device on the worker thread. A
  `--mine` provider on a GPU other than 0 crashed with CUDA_ERROR_ILLEGAL_ADDRESS.
- `roles.py`:
  - **Busy retry:** an error in a provider's previous job is raised once, and the next step retries, instead of the error being raised forever.
  - **Claim and accept guard:** released when the provider's transaction left the mempool and the job went to someone else (`chain.mempool()`). Providers had been sitting idle for 20 blocks after a lost race.
  - **Import race:** `transformers` is loaded before the worker thread starts. A first import racing the model load raised ImportError.
  - **Decoys:** they defend games on their honest stages and stay silent only on the wrong one. They also mark a job lost as soon as another provider wins its accept.

## Operating notes

- Agents need `--units testnet/pool/wiki8x128.units.json,…/wiki128-canary.units.json`. The default is the devnet list, which makes no pool claims.
- The decoy needs `--client-key …/decoyclient.json --post N --batches 1 --fee 20000000 --offset 1500`. WikiText-2 test has about 2,336 chunks of 128 tokens.
- Box 71's GPU 2 is full with `verifier0`, the curator and a noise miner. Run the decoy on GPU 0.
- Don't copy `testnet/agent/` between boxes whole: `onestep/` holds about 13 GB of proof rows. Copy `agent/*.json` only.

## Next

1. λ: fused deterministic kernels or CUDA graphs for `detrt`. This is the gap between a 23×
   client price and the design's 0.7–0.9×, and it needs no protocol change.
2. Consensus patch: jackpots tied to the forced-error stage, directed jobs, a job-result RPC,
   catchers in the job JSON, and a minimum challenge bond sized to a game.
3. Binary input serving for verifiers.
4. Cross-GPU checks on the 5090 when that box is back.
5. Milestone 5 (public testnet) still needs the user's decision on exposure, hosting and outside
   clients.
