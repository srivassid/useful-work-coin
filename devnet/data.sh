#!/usr/bin/env bash
# The LAN-only data service for job inputs and outputs (box 71).
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
source "$D/env"
cd "$D/../repo/runtime"
setsid nohup "$VENV/bin/python" -m agent.data serve --root "$D/data" --bind "$NODE_IP:18680" > "$D/logs/data.out" 2>&1 < /dev/null &
echo $! > "$D/data.pid"
echo "data service on $NODE_IP:18680"
