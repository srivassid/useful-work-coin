#!/usr/bin/env bash
# Stop this box's devnet processes (miners, agents, gateway, then node).
D=$(cd "$(dirname "$0")" && pwd)
cd "$D"
for f in miner*.pid agent*.pid data.pid gateway.pid node.pid; do
  [ -f "$f" ] || continue
  kill "$(cat "$f")" 2>/dev/null && echo "stopped ${f%.pid}"
  rm -f "$f"
  [ "$f" = gateway.pid ] && sleep 2
done
