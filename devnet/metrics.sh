#!/usr/bin/env bash
# Refresh the live metrics (served by the data service at / and /metrics.json) every few minutes.
#   metrics.sh [extra devnet dirs with agent states]   (run in the background on the data box)
D=$(cd "$(dirname "$0")" && pwd)
source "$D/env"
cd "$D/../repo/runtime"
while true; do
  "$VENV/bin/python" -m agent.metrics --devnet "$D" --extra "$@" --json "$D/data/metrics.json.tmp" > /dev/null 2>&1 \
    && mv -f "$D/data/metrics.json.tmp" "$D/data/metrics.json"
  sleep 300
done
