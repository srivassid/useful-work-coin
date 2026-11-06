#!/usr/bin/env bash
# Run a devnet agent role in the background: agent.sh <role> <keyname> [extra args]
#   agent.sh provider miner71 --device cuda:1
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
source "$D/env"
role=$1 keyname=$2; shift 2
name="agent-$role-$keyname"
if [ -f "$D/$name.pid" ] && kill -0 "$(cat "$D/$name.pid")" 2>/dev/null; then echo "$name already running"; exit 0; fi
mkdir -p "$D/logs"
cd "$D/../repo/runtime"
setsid nohup "$VENV/bin/python" -m agent.roles "$role" --key "$HOME/.uwc/devnet/keys/$keyname.json" "$@" \
  > "$D/logs/$name.out" 2>&1 < /dev/null &
echo $! > "$D/$name.pid"
echo "$name started"
