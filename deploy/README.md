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
- `/opt/uwc/data/`: the job files (inputs, outputs, stage-input requests and answers)
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
