# Next chain reset: checklist

Changes held back for the next testnet reset (a new genesis wipes balances, so they wait for it).

## Decided

1. **Our own address prefix.** UWC networks still use Pearl's bech32m prefixes (the UWC devnet
   and testnet inherit regtest's `rprl`, so addresses read `rprl1p...`). Set
   `Bech32HRPSegwit` in `ext/pearl/node/chaincfg/uwc_devnet.go` and `uwc_testnet.go`:
   - mainnet `uwc` (addresses `uwc1p...`)
   - testnet `tuwc` (`tuwc1p...`)
   - devnet `ruwc` (`ruwc1p...`)

   Only the encoding changes. Keys, accounts (`sha256` of the P2TR script) and scripts stay the
   same; `uwcctl info <key>` prints the new address. What else to update:
   - `MINING_ADDR` in each box's `testnet/env` (regenerate it with `uwcctl info`);
   - addresses in docs and results.

   The gateway accepts any prefix (`get_script_pubkey_from_p2tr_address`). Old-prefix addresses
   will be refused by `pearld --miningaddr`, `uwcctl send` and `--address`.

2. **Supply stays 2.1 billion** (`totalSupply` in `blockchain/validate.go`, unchanged). A
   100 million supply was considered and dropped (2026-11-08).

## At the reset itself

- Fold the testnet upgrades into genesis rules: Rules 2 from block 1, with no activation height.
- Register the model again, then do the pool categories (work lists served from the droplet at
  `/units/<work_root>.json`), the canary category, the forced-error commit and the council keys.
- Publish a new package whose `testnet.json` has the new genesis. Miners keep their
  `wallet.json`; their coins start from zero.
