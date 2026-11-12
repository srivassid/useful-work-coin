# Mainnet services (droplet 165.232.182.224)

The testnet services in `../` are retired (disabled, data kept). The mainnet runs these, all as the
system user `uwc` except Caddy:

| unit | what | listens |
|---|---|---|
| `uwc-main-node` | `pearld --uwcmainnet`, the seed node | 0.0.0.0:8690 P2P; RPC 127.0.0.1:8691 |
| `uwc-main-data` | the job data service (`agent.data serve --auth`), work lists in `/opt/uwc/data-main/units` | 0.0.0.0:8692 |
| `uwc-main-gateway` | the job gateway: OpenAI-compatible API and the jobs page, no faucet | 127.0.0.1:8693 |
| `uwc-stats` | `web/stats/exporter.py`: `/var/www/uwc/stats.json` for the miners page | (file) |
| `caddy` | `Caddyfile`: https://165-232-182-224.sslip.io, `/stats.json` static, everything else to the gateway | 443 |

Layout: `/opt/uwc/main` (network dir: `env`, `rpcpass`, `node/`, `bin` -> `../bin-main`),
`/opt/uwc/gateway-main` (`models.json` = `gateway-models.json` here, `models/` = the mainnet model
jsons with their `.reg.json`, `state/` = API keys and their chain keys: back it up), tokenizers
shared with the testnet gateway in `/opt/uwc/gateway/`.

API keys are funded by sending coins to the key's address; the gateway moves them into the key's
jobs balance every 5 minutes (one coin stays in the wallet for transaction fees).
