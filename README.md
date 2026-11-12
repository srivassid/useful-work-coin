# Useful-Work Coin

Implementation of *Useful-Work Coin: Design v3* (`spec/design-v3.pdf`): Pearl's cuPOW lottery
plus a verified job market. Work follows the doc's build plan.

Public page for miners (install, design guide PDF): [srivassid/AICoin](https://github.com/srivassid/AICoin).

| Step | Status |
|---|---|
| 1. Go/no-go: λ and quantization quality | **done**: see `spec/step1-results.md` |
| 2. Reference runtime (bit-exact across GPU families) | **done**: see `spec/step2-results.md` |
| 3. Job module in Pearl's node | **done**: see `spec/step3-results.md`, code in `chain/patches/` |
| 4. Verification (bisection to one op, serve-or-slash) | **done**: see `spec/step4-results.md`, `spec/stage-program.md` |
| 5. Pool and watchers (budget, unit pricing, canaries, forced errors with threshold keys) | **done**: see `spec/step5-results.md`, `spec/pool-and-watchers.md` |
| 6. Simulator extended (free entry, risk aversion, block withholding, measured λ) and retune | **done**: see `spec/step6-results.md`, `sim/sim.py`, `results/sim_step6.txt` |
| 7. Public testnet | **done**: private devnet (`spec/step7-results.md`), then the public testnet from 2026-11-06 to 2026-11-12 (Rules 2 to 5, `spec/step7-progress.md`); retired when the mainnet launched |
| 8. Mainnet | **live since 2026-11-12**: see below |

## Mainnet

Launched 2026-11-12. Block 1 was mined at 06:26:29 UTC. Code: `chain/patches/0026` (`pearld --uwcmainnet`).

| | |
|---|---|
| Genesis | `450d4c952b1f9b434e0ae9a63811ef785cbb3f5c9a50f3d994a58e87c99aa4bc` (2026-11-12 06:00 UTC) |
| Block 1 | `67887d85068d5477af97fcabe9e2bfc9d8eba9d5da3ba1ff35aaebb058eb0bab` |
| Network | magic `UWCM`, P2P port 8690, RPC 8691 (local), data service 8692 |
| Seed node | `165.232.182.224:8690` |
| Addresses | `uwc1p…` (Taproot) |
| Rules | Rules 5 (the testnet's final rules) from block 0, no scheduled upgrades |
| Supply | 2.1 billion coins; block reward 999 coins at the start, about one block a minute; half the supply in the first 4 years |
| Reward split | 90% to the miner, 10% to the job pool |
| Coinbase maturity | 100 blocks |
| Dev fund | 21,000,000 coins (1% of supply), paid once by block 1 to `uwc1pxhsl08wxwn37dz3t3rgqwsg76azgwdl6duh4x9lg629umlrppnjsz6d7jz`; consensus rejects a block 1 that does not pay it, and no other block may |
| Council | 2 of 3 keys, held by the project at launch |

Status: mining and the job market are live; the mainnet `uwc-miner` package follows.

### Job market and API

Opened 2026-11-12 at block ~400. Four models are registered, each with a job pool and staked
providers and verifiers run by the project:

| Model | Chain id | Pool category | Price per 1,000 tokens |
|---|---|---|---|
| Qwen3 1.7B (chat, completion, embeddings) | `c84eea62…` | `62f57cc3…` | 1.5 coins |
| Llama 3.2 3B Instruct (chat, completion) | `08fde5c3…` | `7c995318…` | 2.5 coins |
| Mistral 7B Instruct v0.2 (chat, completion) | `0795f63d…` | `4eb0a8ae…` | 5 coins |
| Llama 3.1 8B, base (completion only) | `72de4c76…` | `f93a71b5…` | 6 coins |

The job gateway (`runtime/agent/gateway.py`) serves an OpenAI-compatible API and the jobs page at
<https://165-232-182-224.sslip.io/>. There is no faucet on mainnet:

1. `POST /v1/keys` (or **Create key** on the page) returns an API key and a `uwc1p…` address.
2. Send coins to that address. Every 5 minutes the gateway moves them into the key's jobs balance,
   keeping 1 coin in the wallet for transaction fees.
3. Call `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings` or `/v1/batches` with
   `Authorization: Bearer <key>`. `GET /v1/account` shows the balance; `GET /v1/jobs/<id>` a
   request's trust level.

Each request is posted on chain as a job (minimum fee 0.5 coins). A provider accepts it, computes
it and commits the result; a verifier recomputes sampled stages. Trust goes from *committed* (the
answer is back, about 5 minutes on a quiet network) to *verified* (about 10 minutes later) to
*final* when the challenge window closes, about 5 hours. End-to-end test on 2026-11-12: one chat
request on each instruct model and a completion on Llama 3.1 8B, 0.5 coins each.

Live mining stats: <https://aicoin-miners.vercel.app> (miners active in the last hour and day,
blocks, work rate), fed by `web/stats/exporter.py`, which publishes public chain data only.
Server setup: `deploy/mainnet/`.

Layout (from the doc): `chain/`, `miner/` (`miner/ampere`: Ampere port of Pearl's NoisyGEMM
miner), `runtime/`, `verifier/` (`forced_errors.py`: keyholder shares and curator trees),
`sim/` (`sim.py`), `spec/`, `results/` (raw benchmark output).

Running `miner/ampere` needs an NVIDIA Ampere-or-newer GPU, Python 3.12, torch 2.11 (cu130),
cupy-cuda13x, and `pearl_mining` built from Pearl's `py-pearl-mining` with maturin. Tests:
`cd miner/ampere && pytest tests/`.
