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

Design: `spec/job-module.md`, `spec/stage-program.md`, `spec/pool-and-watchers.md`. Results:
`spec/step3-results.md`, `spec/step4-results.md`, `spec/step5-results.md`, `spec/step6-results.md`.
