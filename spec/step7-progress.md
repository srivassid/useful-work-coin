# Step 7 progress: devnet (private), then public testnet

Build-plan step 7: "Public testnet. Run real batch jobs from outside clients through the full
flow." Agreed staging (2026-11-06): a **private devnet** on our two GPU boxes first, with **real
cuPOW mining** and synthetic client jobs. Opening anything to the internet and choosing outside
clients is a later, separate decision.

## Milestones

| # | Milestone | Status |
|---|---|---|
| 1 | Real blocks: devnet params, `pearld` with proof verification, Ampere miner → Pearl gateway (zk proof) → node | **done** |
| 2 | Job tooling: `uwcctl` deposits and job ops (Taproot-signed), council actions | next |
| 3 | Provider, verifier and watcher daemons on the GPUs (deterministic runtime), client CLI with off-chain output delivery | – |
| 4 | Testnet metrics page: useful share, λ per job type, verification overhead, false-slash rate, client price | – |
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
