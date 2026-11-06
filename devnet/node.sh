#!/usr/bin/env bash
# pearld on the UWC devnet, with the indexes uwcctl needs to find its coins.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
source "$D/env"
mkdir -p "$D/node" "$D/logs"
exec "$D/bin/pearld" --uwcdevnet --configfile=/dev/null --datadir="$D/node" --logdir="$D/logs/node" \
  --listen="$NODE_IP:18670" --addpeer="$PEER_IP:18670" \
  --rpclisten=127.0.0.1:18671 --rpcuser="$RPC_USER" --rpcpass="$RPC_PASS" --notls \
  --txindex --addrindex --miningaddr="$MINING_ADDR" --debuglevel=info
