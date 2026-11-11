# Public testnet services (DigitalOcean droplet 165.232.182.224)

Ubuntu 24.04, 1 vCPU, 458 MB RAM plus a 2 GB swap file. Services run as the system user `uwc` under
`/opt/uwc`; ufw allows only 22, 18690 (P2P) and 18692 (data service).

| unit | what | listens |
|---|---|---|
| `uwc-node` | `pearld --uwctestnet`, the seed node (no indexes, no mining), `--externalip` set | 0.0.0.0:18690 P2P; RPC 127.0.0.1:18691 |
| `uwc-data` | `python3 -m agent.data serve --auth` (stdlib only: `agent/{__init__,data,chain}.py`) | 0.0.0.0:18692 |

Layout:
- `/opt/uwc/bin/`: pearld, uwcctl, prlctl (stripped builds with `-tags xmss,zkpow`)
- `/opt/uwc/net/`: the node's network directory (`env`, `pearld.conf`, `rpcpass` mode 0600; `node/`,
  `logs/`; `bin` points at `../bin`). The data service reads jobs and accounts through it.
- `/opt/uwc/data/`: the job files (inputs, outputs, stage-input requests and answers), plus
  `units/<work_root>.json`, the pool work lists. These are read-only over HTTP; providers check
  each one against the category's on-chain work root before using it.
- `/opt/uwc/app/agent/`: the data service code

The data service with `--auth` accepts only uploads signed with `uwcctl sign` (header
`X-UWC-Sig`, over `agent.data.put_digest`). The chain decides who may write what:
- outputs and stage inputs: only the job's provider;
- stage-input requests: the job's committee, or an account with a stake or at least 1 coin,
  capped per job and per account;
- inputs: any such account, write-once.

Uploads are size-capped, and the service refuses them when less than 1 GB of disk is free. GETs
are public. See `runtime/tests/test_data_auth.py`.

Unit files are in this folder. Update the code with:

    rsync -a runtime/agent/{__init__,data,chain}.py root@165.232.182.224:/opt/uwc/app/agent/ && \
      ssh root@165.232.182.224 systemctl restart uwc-data

New pool category (box 71, council keys in ~/.uwc/testnet/keys):

    python -m agent.worklist --split train --offset <next free chunk> --units 500 --per-unit 8 ... --out testnet/pool/<name>
    cat testnet/pool/<name>.units.json | ssh root@165.232.182.224 "cat > /opt/uwc/data/units/<work_root>.json"
    ./uwc.sh propose-category council0.json testnet/pool/<name>.category.json   # then: approve council1.json <proposal>

The train split has about 19,600 chunks of 128 tokens; wiki-train8x128 (category 5e35742f, 2026-11-08)
uses chunks 0-3999.

## Job gateway and web page (`uwc-gateway`, https://165-232-182-224.sslip.io)

Caddy proxies the domain to the gateway on 127.0.0.1:18693. The gateway serves the web page at `/`
(sign up, chat, text completion, embeddings, batches, your jobs) and the OpenAI-compatible API
under `/v1`. Its files are in `/opt/uwc/gateway/`:
- `models.json`: the served models (see the doc at the top of `runtime/agent/gateway.py`): Qwen3
  1.7B (chat, embeddings), Llama 3.2 3B Instruct and Mistral 7B Instruct v0.2 (chat), Llama 3.1 8B
  (base: completions only), each with its model json and `.reg.json` (`models/`) and its
  `tokenizer.json` and `tokenizer_config.json` (`tokenizers/`, for the chat template);
- `env`: `UWC_GATEWAY_SIGNUP_CODE`; `faucet.json`: the key that funds new API keys; `state/`: the
  database, the API keys' chain keys, requests and results.

The venv needs `tokenizers`, `numpy` and `jinja2`. Update the code with:

    rsync -a runtime/agent/{gateway.py,gateway_page.html,roots.py} root@165.232.182.224:/opt/uwc/app/agent/ && \
      ssh root@165.232.182.224 systemctl restart uwc-gateway

A model is served only while a provider of it is running and a staked verifier has declared it
(Rules 5); otherwise its jobs expire and are posted again with a higher fee, then fail.
