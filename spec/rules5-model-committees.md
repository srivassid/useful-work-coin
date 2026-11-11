# Rules 5: committees from verifiers that hold the job's model

Status: implemented and tested (chain patch 0024, agents); testnet from block 4280 (patch 0025).

## The problem (testnet, Rules 4, 2026-11-10)

With Qwen3, Llama 3.2 3B, Mistral 7B and Llama 3.1 8B pools open at once, `DrawCommittee` drew
seats from every staker, whatever models it ran. Box 71's verifiers held only Qwen3, so:

- they were seated on Llama and Mistral jobs they could not compute, did not show, and paid the
  no-show penalty each time;
- a job whose seats all went to such verifiers had no real check: 7 wrong Llama jobs (computed
  with another model by the claim bug) settled unchecked. Only the watcher caught the 8 wrong
  Mistral jobs.

So a pool could open, and be paid, before anyone could verify its model.

## The rule

1. **VERIFIER_MODELS** (new op): an account declares the models it verifies, as model ids in
   strictly increasing order, at most `MaxVerifierModels` (16), each a registered model. A new
   declaration replaces the old one; an empty one withdraws. Declaring needs no stake, but counts
   only while the account has at least `MinVerifierStake` staked.
2. **Committee draw**: a staker is eligible for a job's committee only if it declared the job's
   model (and, as before, has the minimum stake and is not the job's client or provider). The
   draw is otherwise unchanged: stake-weighted, deterministic from the block hash.
3. **No work nobody can check**: `JOB_ACCEPT` and `POOL_CLAIM` fail unless at least one eligible
   verifier exists for the job's model at that block.
4. **Verifiers leaving**: if no eligible verifier is left when the committee is drawn (all of
   them withdrew or unstaked after the claim), the job is **aborted**. The client's fee (for a
   pool unit, its verification pay, back to the category, freeing the unit) and the provider's
   bond are returned; nobody is paid or penalized. Before Rules 5 such a job went straight to the
   challenge window with no committee.

Before Rules 5 none of this applies: `VERIFIER_MODELS` fails, and draws, accepts and claims behave
exactly as in Rules 4.

## State and encoding

- `State.VerifierModels map[AccountID][]Hash`, committed in the state root under prefix `'v'`
  (account -> count, model ids). There are no entries before Rules 5, so earlier roots are
  unchanged. Old gob snapshots decode with the map nil.
- Op kind 24 (`VERIFIER_MODELS`): varint count, then 32-byte model ids.
- RPC: `getjobstate` adds `verifiers` (model -> number of staked verifiers that declared it);
  `getjobaccount` adds `models` (the account's declaration).
- `uwcctl verifiermodels <key> [model ...]` sends a declaration (sorted for you).

## Agents

- A **verifier** declares every model it loaded (`--model a,b,...`) once they are loaded, when
  the chain is on Rules 5, and again whenever the chain's list differs (checked each step; a
  declaration not yet mined is not resent for 10 blocks).
- A **provider** does not claim pool units or accept jobs of a model with no staked verifier on
  Rules 5 (the chain would refuse them); it logs this at most every 100 blocks.

## Activating on the testnet

Every node (box 71, .245, the seed droplet, the 5090's package) must run a Rules-5 binary
before the activation height, and the verifiers should run the new agent so they declare at that
height; until a verifier declares a model, providers of that model wait. Watchers are unchanged.

Tests: `ext/pearl/node/jobstate/rules5_test.go` (wire format, declarations, draw, refusals,
abort, snapshots), `runtime/tests/test_agent_models.py` (agent declarations, claim filter).
