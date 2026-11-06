# Step 6 results: the simulator, extended

Build-plan step 6: "Add a GPU count that responds to revenue, risk-averse providers, block
withholding and measured λ values to sim.py. Retune ρ, k, n and bonds." **Done** (2026-11-06).

`sim/sim.py` keeps parts A–D unchanged: their output is byte-identical, so the design doc's
numbers still reproduce. It adds parts E–I. It is still standard-library Python. A full run takes
about 40 s, and `--parts ABCD` takes 5 s. Output: `results/sim_step6.txt`. Checks:
`sim/test_sim.py`, 7 tests covering formulas against Monte Carlo and the measured table against
the raw results.

Results are only as good as the stated assumptions. The model doesn't test networking or real
markets. Its inputs are the measured λ, throughputs and noise rates. GPU prices, power and
electricity are assumptions and are labelled as such.

## E. Measured λ

The step-1 benchmarks (RTX 3080 and RTX 5090; compiled runtime; 8192-token batches) are in the
`MEASURED` table, which the script recomputes from `results/*.json` when present.

| GPU | Model | λ embed | λ score | Mining adds | Client ÷ rent (int8) | Client ÷ rent (bf16) | Hashes given up per pool token |
|---|---|---|---|---|---|---|---|
| 3080 | Qwen3-1.7B | 0.379 | 0.369 | 39% | 0.88 | 0.47 | 2.3 GMAC |
| 3080 | Qwen3-4B | 0.488 | 0.468 | 33% | 0.70 | – | **3.8 GMAC** |
| 5090 | Qwen3-1.7B | 0.357 | 0.334 | 42% | 0.93 | 0.50 | 2.5 GMAC |
| 5090 | Qwen3-4B | 0.407 | 0.394 | 35% | 0.82 | 0.43 | **5.3 GMAC** |
| 5090 | Qwen3-8B | 0.492 | 0.481 | 30% | 0.67 | 0.32 | 7.2 GMAC |

- **Client price.** Mining income per GPU-second equals the GPU's cost under free entry. A client
  therefore pays (1 − λ)/0.98 of that cost per mined GPU-second; the 0.98 is the verifier and
  treasury shares. Even though mining slows the job by 30–42%, buying a job costs 0.67–0.93× of
  renting the same GPU at cost to run it in int8 without mining, and 0.32–0.50× of stock bf16.
  This is the testnet metric "client price compared with renting the same GPU time".
- **Comparative advantage.** Per pool token, a 3080 gives up 3.8 GMAC of hashing and a 5090 gives
  up 5.3 GMAC on the same 4B job. The 5090 has 4.5× the noise rate but only 3.7× the throughput,
  and a lower λ. The pool's open race therefore goes to the 3080s first.

## F. A GPU count that responds to revenue

Model:
- **Supply:** 6,000 potential 3080s and 1,500 potential 5090s. Assumed capex is $500 and $2,500
  over 3 years, assumed power is 0.25 and 0.65 kW, and electricity is spread uniformly over
  $0.03–0.20/kWh.
- **Revenue:** block rewards worth Y = $300/h.
- **Choices:** each GPU repeatedly picks off, noise, pool (Qwen3-4B embeddings, budget split by
  tokens) or client work (demand in tokens/s at a price that clears).
- **Baseline:** at ρ = 0, 3,340 GPUs are on.

| ρ | GPUs on | Pool share of mining capacity | ρ/(1−λ) | GPUs on useful work (by count) | Hashrate kept | Fixed-count formula |
|---|---|---|---|---|---|---|
| 5% | 3,340 | 9.8% | 9.8% | 20.5% | 0.950 | 0.950 |
| 10% | 3,340 | 19.5% | 19.5% | 41.0% | 0.900 | 0.900 |
| 20% | 3,514 | 35.8% | 39.1% | 71.4% | 0.816 | 0.800 |
| 30% | 3,866 | 46.4% | 58.6% | 78.5% | 0.752 | 0.700 |

- **The doc's formula holds in hash terms.** The pool takes exactly ρ/(1 − λ) of mining capacity,
  and hashrate falls by exactly ρ. The pool picks the GPUs that give up the fewest hashes, so by
  count it reaches about twice that share: 41% of GPUs at ρ = 10%. No 5090 does pool work until
  ρ = 30%.
- **Free entry softens saturation.** Once every 3080 is on pool work (ρ ≥ 20%), pool money pulls
  more GPUs in, and hashrate falls less than the fixed-count formula says (0.816 vs 0.800; 0.752
  vs 0.700).
- **Client demand** of 1.5M tokens/s leaves hashrate unchanged (0.997–1.000 at the same ρ). Fees
  pay exactly for the hashes the client GPUs give up, and that money brings new GPUs in. With a
  fixed GPU count, demand would lower hashrate instead. Clients pay **0.68×** the cost of renting
  (rent priced at what the GPU would earn mining) at ρ ≤ 10%. This rises to 0.78–0.91 at ρ =
  20–30%, when the pool competes for the same 3080s.
- **Coin-price shock.** At ρ = 10%, block rewards halving to $150/h drops GPUs from 3,341 to
  1,934 and hashrate to 0.537 of baseline. The useful share stays at 41–42%. Recovery restores
  3,341 GPUs and 0.900.

## G. Risk-averse providers

- **G1, cheaters** (CARA utility). At fee = cost, no bond and one sample, a risk-neutral cheater
  breaks even. Any risk aversion makes every partial skip strictly worse, and a full skip is still
  exactly 0, because it is certain to be caught. With a 50% bond, the best cheat drops from −0.5 to
  −1.2 and −8.5 stage-costs as risk aversion rises (a·S·c = 0.5, 2). **Bonds sized for a
  risk-neutral cheater are therefore conservative.**
- **G2, honest providers' cost of a bond.** Capital is locked for about 1.5 days at 15%/year.
  Risk is a false slash with probability ε, priced with CARA, a·F = 1.

  | Bond ÷ fee | ε = 0 | ε = 10⁻⁴ | ε = 10⁻³ |
  |---|---|---|---|
  | 0.25 | 0.015% | 0.040% | 0.26% |
  | **0.5** | **0.031%** | 0.066% | 0.38% |
  | 1.0 | 0.062% | 0.13% | 0.70% |
  | 2.0 | 0.12% | 0.31% | 2.0% |

  With the bit-exact runtime (the target false-slash rate is zero), a bond costs only capital.
  A buggy reference runtime would cost far more than any bond size: that is where false slashing
  comes from.
- **G3, payout variance.** A small solo miner's daily income has CV 0.8–2.6, while pool units pay
  an equal share of each epoch's budget. Neither a 2% mining-pool fee, nor a 0.5% false-slash
  premium, moves the pool share meaningfully (19.5% → 19.8–19.9%). Discounting pool income by 10%
  for strong risk aversion lowers it to 17.7%, which equals ρ(1 − 0.1)/(1 − λ).

## H. Block withholding

Our draw (step 3) uses the hash of the block after `TRACE_COMMIT`. A provider that also mines (with
hashrate share α) can discard that block when the committee it draws contains an honest verifier,
losing the block reward R, and try again. With x = q^k(1 − w), the pass chance becomes
x/(1 − α(1 − x)). Re-rolling pays only when xV − (1 − x)R > x(xV − (1 − x)L); α cancels, so the
threshold depends only on the job's size. Analytic and Monte Carlo (200k trials per row) agree.

Here q = ⅓ colluding stake, w = 10% watchers, α = 0.3, and L = 1.5 job costs (fee plus 50% bond):

| k | Attacker's gain if the fake passes (V) | Job size | Pass chance | With re-rolls | Expected gain |
|---|---|---|---|---|---|
| 2 | 1 job cost (saved compute) | 100 R | 0.100 | 0.137 | −1.16 |
| 2 | 10 (tampering) | 1 R | 0.100 | 0.137 | −0.30 |
| 2 | 10 | 10 R | 0.100 | 0.137 | **+0.04** |
| 2 | 10 | 100 R | 0.100 | 0.137 | **+0.07** |
| 3 | 10 | 100 R | 0.033 | 0.047 | −0.96 |
| 5 | 10 | 100 R | 0.004 | 0.005 | −1.44 |

- Re-rolling never makes a fake that only saves compute (V = 1) pay. It does turn k = 2 profitable
  for an attacker who gains 10× the job, once jobs are worth more than about 5 block rewards.
- **A longer draw delay doesn't help.** Whoever mines the seed block can always discard it, so the
  doc's "draw from d ≥ 2 blocks later" mitigation is not effective.
- **Private sortition** (each staker learns privately whether it is drawn, via a VRF) is immune to
  re-rolls. The miner sees only its colluders' seats, and those are independent of the honest ones.
  But a random committee size needs 39–65% more expected seats for the same pass chance (e.g.
  4.9 seats to match k = 3 at q = ⅓). Pearl's signatures (XMSS) are not unique, so a VRF would be
  new cryptography. It is left for later.

## I. Retune

| Parameter | Doc v3 | Step 6 | Why |
|---|---|---|---|
| ρ | 10% | **10%** (unchanged) | At measured λ ≈ 0.45 it moves ρ/(1 − λ) ≈ 20% of mining capacity (≈ 40% of GPUs) onto pool work for a hashrate cost of ρ. Saturation is at ρ ≈ 0.5. Raising λ is still the cheapest lever. |
| k | ≥ 2 | **≥ 2; ≥ 3 when the fee is ≥ 5 block rewards** | Part H. At those fees, verifier pay is 1% of the fee whatever k is, so the rule costs clients nothing. Implemented as consensus rule `LargeJobRewards = 5`, `MinKLarge = 3` (patch 0007). Clients exposed to tampering worth more than about 10× the job should still buy more (k = 5). |
| n | ≥ 2 | **≥ 2** (unchanged) | Part C needs one sample once F + B ≥ S·c, and G1 shows risk aversion only strengthens that. |
| Bond | ≥ 50% of fee | **≥ 50%** (unchanged) | It costs an honest provider about 0.03% of the fee (G2). A larger bond buys margin only against risk-neutral cheaters, who are already deterred. |
| Watchers / jackpots | – | unchanged | w enters every pass chance as (1 − w). Part D and step 5's measured catch rate agree. |

**Patch 0007** (`node/jobstate`, `node/blockchain`): `BlockInput.Subsidy` is filled in by the node
at its three `ApplyBlock` call sites. A `JOB_POST` whose fee is ≥ `LargeJobRewards` × (subsidy −
pool share) needs k ≥ `MinKLarge`. The rule is off when the subsidy is 0, as in the pure unit
tests. Test: `TestLargeJobsNeedLargerCommittees`. All jobstate tests pass (coverage 92.2%), and
the node integration tests pass (only the pre-existing `-tags zkpow` test fails, as on unmodified
Pearl).

## Limits

- **Costs are assumptions.** The GPU cost curve (capex, power, electricity spread) is assumed. The
  qualitative results (the formula holds in hash terms, the cheaper-per-hash class goes to the pool,
  demand is hashrate-neutral) don't depend on the specific values; the counts do.
- **One pool category** (4B embeddings). Pool and client compete for one job type.
- **No coin-price feedback** from usefulness or from demand for fees.
- **Block withholding covers committee re-rolls only.** Classic withholding against Pearl's mining
  pools is Pearl's problem, unchanged by this design.
- **λ** comes from two consumer GPUs with my kernels. Datacenter GPUs, with Pearl's Hopper kernel,
  need measuring on the testnet (step 7).
