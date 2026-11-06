#!/usr/bin/env bash
# Pearl's gateway: templates from pearld, zk proofs from miners' plain proofs, blocks to pearld.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
source "$D/env"
source "$VENV/bin/activate"
export PEARLD_RPC_URL=http://127.0.0.1:${RPC_PORT:-18671} PEARLD_RPC_USER="$RPC_USER" PEARLD_RPC_PASSWORD="$RPC_PASS"
export PEARLD_MINING_ADDRESS="$MINING_ADDR" MINER_RPC_TRANSPORT=uds MINER_RPC_SOCKET_PATH="$D/pearlgw.sock"
exec python -m pearl_gateway.cli start
