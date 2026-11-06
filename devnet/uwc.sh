#!/usr/bin/env bash
# uwcctl against the local devnet node: uwc.sh coins ~/.uwc/devnet/keys/miner71.json
D=$(cd "$(dirname "$0")" && pwd)
source "$D/env"
export UWC_RPC=127.0.0.1:18671 UWC_RPC_USER="$RPC_USER" UWC_RPC_PASS="$RPC_PASS"
exec "$D/bin/uwcctl" "$@"
