# Useful-Work Coin

Implementation of *Useful-Work Coin: Design v3* (`spec/design-v3.pdf`): Pearl's cuPOW lottery
plus a verified job market. Work follows the doc's build plan.

| Step | Status |
|---|---|
| 1. Go/no-go: λ and quantization quality | **done**: see `spec/step1-results.md` |
| 2. Reference runtime (bit-exact across GPU families) | **done**: see `spec/step2-results.md` |
| 3. Job module in Pearl's node | **done**: see `spec/step3-results.md`, code in `chain/patches/` |
| 4. Verification (bisection to one op, serve-or-slash) | **done**: see `spec/step4-results.md`, `spec/stage-program.md` |
| 5. Pool and watchers (budget, unit pricing, canaries, forced errors with threshold keys) | **done**: see `spec/step5-results.md`, `spec/pool-and-watchers.md` |
| 6. Simulator extended (free entry, risk aversion, block withholding, measured λ) and retune | **done**: see `spec/step6-results.md`, `sim/sim.py`, `results/sim_step6.txt` |
| 7. Public testnet | **private devnet done** (every mechanism live, 27 jobs settled): see `spec/step7-results.md`; public stage awaits a decision |

Layout (from the doc): `chain/`, `miner/` (`miner/ampere`: Ampere port of Pearl's NoisyGEMM
miner), `runtime/`, `verifier/` (`forced_errors.py`: keyholder shares and curator trees),
`sim/` (`sim.py`), `spec/`, `results/` (raw benchmark output).

Running `miner/ampere` needs an NVIDIA Ampere-or-newer GPU, Python 3.12, torch 2.11 (cu130),
cupy-cuda13x, and `pearl_mining` built from Pearl's `py-pearl-mining` with maturin. Tests:
`cd miner/ampere && pytest tests/`.
