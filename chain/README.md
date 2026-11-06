# chain/: Pearl node fork with the Useful-Work Coin job module

The fork is kept as a patch series on top of Pearl commit `0dbdb2f`
(github.com/pearl-research-labs/pearl):

```
git clone https://github.com/pearl-research-labs/pearl.git ext/pearl
cd ext/pearl && git checkout -b uwc 0dbdb2f && git am ../../chain/patches/*.patch
cd node && go test ./jobstate/ ./blockchain/ -run 'TestJobModule|.' ./mining/ ./mempool/
```

Go 1.26.10. Tests run on SimNet (dummy certificates), so the Rust zk-pow library is not needed;
`TestCheckBlockSanity` and the wire certificate tests need `-tags zkpow` and fail without it,
exactly as on unmodified Pearl.

| Patch | Content |
|---|---|
| 0001 | `node/jobstate`: the job module (pure Go) and its property tests |
| 0002 | consensus integration: `chaincfg.Params.JobModule`, coinbase rules, state persistence, mining templates, mempool policy |

Design: `spec/job-module.md`. Results: `spec/step3-results.md`.
