#!/usr/bin/env bash
# prlctl against the local devnet node: ctl.sh getjobstate
D=$(cd "$(dirname "$0")" && pwd)
source "$D/env"
exec "$D/bin/prlctl" --configfile=/dev/null --rpcserver=127.0.0.1:18671 --rpcuser="$RPC_USER" --rpcpass="$RPC_PASS" --notls "$@"
