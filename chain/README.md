# chain/: Pearl node fork with the Useful-Work Coin job module

The fork is kept as a patch series on top of Pearl commit `0dbdb2f`
(github.com/pearl-research-labs/pearl):

```
git clone https://github.com/pearl-research-labs/pearl.git ext/pearl
cd ext/pearl && git checkout -b uwc 0dbdb2f && git am ../../chain/patches/*.patch
cd node && go test ./jobstate/ ./blockchain/ ./mining/ ./mempool/
```

Go 1.26.10. Tests run on SimNet (dummy certificates), so the Rust zk-pow library is not needed;
`TestCheckBlockSanity` and the wire certificate tests need `-tags zkpow` and fail without it,
exactly as on unmodified Pearl.

| Patch | Content |
|---|---|
| 0001 | `node/jobstate`: the job module (pure Go) and its property tests |
| 0002 | consensus integration: `chaincfg.Params.JobModule`, coinbase rules, state persistence, mining templates, mempool policy |
| 0003 | stage programs and the bit-exact referee (with Python fixtures in `jobstate/testdata`) |
| 0004 | dispute games, one-step proofs, serve-or-slash, fault-injection suite |
| 0005 | fix: snapshots no longer share dispute-game state; the state root commits games and model architecture |
| 0006 | public-good pool, canary units, watcher checks, forced errors with a t-of-m council |
| 0007 | step-6 retune: jobs with fee ≥ 5 block rewards need k ≥ 3 (committee re-rolls by block withholding) |
| 0008 | UWC devnet (`--uwcdevnet`, real proof of work), job-state RPCs, job coinbase outputs in `getblocktemplate`, gateway support |
| 0009 | `uwcclient` and `uwcctl`: Taproot-signed deposits, job ops and council proposals |
| 0010 | `uwcctl ops` (any op as JSON); devnet windows |
| 0011 | one-step proofs for providers (`BuildOneStep`, `uwcctl onestep`); pool bonds capped at the reference unit cost |
| 0012 | devnet: 20-block challenge window |
| 0013 | scheduled job-module upgrades (`Params.Upgrades`, `Rules`), `--uwctestnet`, mempool persistence |
| 0014 | stages committed by their output roots; full traces only in disputes |
| 0015 | testnet genesis timestamp in the past |
| 0016 | **Rules 2** (testnet run 1 findings, active on the testnet from height 1400): forced-error jackpots only for a catch on the decoy's wrong stage (record v2 with a stage key; jobs record the stage they were caught on), directed jobs, challenge bonds of `ChallengeBondBP` of the provider's bond; node: `getjobopresult`, failed job ops logged, catchers / directed / caught stage in `getjob` |
| 0017 | `uwcctl sign` / `verify`: BIP-340 signatures by the P2TR output key over a 32-byte digest, verify prints the signer's account (the public data service's signed uploads) |
| 0018 | **Rules 3: generation jobs**: a head stage per batch (final norm, lm_head scores in 1024-token chunks, chunk bests, Gumbel-max or greedy sample), generated tokens read by layer stage 0 from the head stage, output root over head output roots (head stages open with an output-root path), generation models commit norm + lm_head; encodings and state gain suffixes only when set (spec/generation.md) |
| 0019 | uwcctl `jobgen`, `modelreg` with `"gen"`, `out_proof` in JSON reveals, `onestep` for generation stages; `getjob` reports `gen` and `tempmilli` |
| 0020 | Testnet: Rules 3 (generation jobs) from block 2120 (replayed the 2073-block testnet with this binary: same tip) |
| 0021 | Rules 4: Llama and Mistral models (`ModelCfg.Arch`, MODEL_REG suffix `A`+arch; QN/KN rows are copies for Arch 1) |
| 0022 | Testnet: Rules 4 (Llama and Mistral models) from block 2700 (replayed the 2554-block testnet with this binary: same tip and state root) |

Design: `spec/job-module.md`, `spec/stage-program.md`, `spec/pool-and-watchers.md`. Results:
`spec/step3-results.md`, `spec/step4-results.md`, `spec/step5-results.md`, `spec/step6-results.md`.
