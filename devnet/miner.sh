#!/usr/bin/env bash
# Ampere noise miner on one GPU: miner.sh <gpu index>
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
source "$D/env"
source "$VENV/bin/activate"
mkdir -p "$D/logs"
cd "$D/../repo/miner/ampere"
exec python -m uwc_ampere.miner_loop --gateway "$D/pearlgw.sock" --device "${1:-0}" \
  --stats "$D/logs/noise-gpu${1:-0}.jsonl"
