#!/usr/bin/env bash
# Start this box's devnet processes: node, gateway, and noise miners on the given GPUs.
#   up.sh 0 1 2      (no GPUs: node and gateway only)
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd)
cd "$D"
mkdir -p logs
start() { # name command...
  local name=$1; shift
  if [ -f "$name.pid" ] && kill -0 "$(cat "$name.pid")" 2>/dev/null; then echo "$name already running"; return; fi
  setsid nohup "$@" > "logs/$name.out" 2>&1 < /dev/null &
  echo $! > "$name.pid"
  echo "$name started"
}
start node ./node.sh
sleep 5
start gateway ./gateway.sh
sleep 4
for g in "$@"; do start "miner$g" ./miner.sh "$g"; done
