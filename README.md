# Useful-Work Coin

Implementation of *Useful-Work Coin: Design v3* (`spec/design-v3.pdf`): Pearl's cuPOW lottery
plus a verified job market. Work follows the doc's build plan.

| Step | Status |
|---|---|
| 1. Go/no-go: λ and quantization quality | **done**: see `spec/step1-results.md` |
| 2. Reference runtime (bit-exact across GPU families) | **done**: see `spec/step2-results.md` |
| 3. Job module in Pearl's node | **done**: see `spec/step3-results.md`, code in `chain/patches/` |
| 4. Verification (bisection to one op, serve-or-slash) | **done**: see `spec/step4-results.md`, `spec/stage-program.md` |
| 5–7. Pool and watchers, simulator, testnet | not started |

Layout (from the doc): `chain/`, `miner/` (`miner/ampere`: Ampere port of Pearl's NoisyGEMM
miner), `runtime/`, `verifier/`, `sim/` (`sim.py`), `spec/`, `results/` (raw benchmark output).

Running `miner/ampere` needs an NVIDIA Ampere-or-newer GPU, Python 3.12, torch 2.11 (cu130),
cupy-cuda13x, and `pearl_mining` built from Pearl's `py-pearl-mining` with maturin. Tests:
`cd miner/ampere && pytest tests/`.
