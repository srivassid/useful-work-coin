#!/usr/bin/env bash
# pearld on the UWC devnet, with the indexes uwcctl needs to find its coins.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
source "$D/env"
mkdir -p "$D/node" "$D/logs"
NET=${NET:-uwcdevnet}; P2P_PORT=${P2P_PORT:-18670}; RPC_PORT=${RPC_PORT:-18671}
exec "$D/bin/pearld" --$NET --configfile=/dev/null --datadir="$D/node" --logdir="$D/logs/node" \
  --listen="$NODE_IP:$P2P_PORT" --addpeer="$PEER_IP:$P2P_PORT" \
  --rpclisten=127.0.0.1:$RPC_PORT --rpcuser="$RPC_USER" --rpcpass="$RPC_PASS" --notls \
  --txindex --addrindex --miningaddr="$MINING_ADDR" --debuglevel=info
