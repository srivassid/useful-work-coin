#!/usr/bin/env bash
# Build the portable uwc-miner package (one folder, one tarball; nothing installed on the system).
#
#   dist/build.sh <go-bin-dir> <testnet-dir> <out-dir>
#
#   go-bin-dir   pearld, uwcctl, prlctl (built with -tags xmss,zkpow)
#   testnet-dir  a testnet network directory: models/qwen3-1.7b{,.reg}.json and pool/*.json
#   out-dir      where uwc-miner-<version>/ and its .tar.gz go
#
# Needs uv and cargo (py-pearl-mining is built from source). PEARL (default <repo>/ext/pearl)
# points at the Pearl fork; VERSION overrides the version string.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
GOBIN=$(cd "$1" && pwd); NETDIR=$(cd "$2" && pwd); OUTDIR=$(mkdir -p "$3" && cd "$3" && pwd)
PEARL=${PEARL:-$R/ext/pearl}
V=${VERSION:-testnet-$(date +%Y%m%d)}
OUT=$OUTDIR/uwc-miner-$V
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
rm -rf "$OUT"; mkdir -p "$OUT"/{bin,app,share/models,share/pool}
echo "[$(date +%T)] python"
uv python install 3.12 >/dev/null
PYSRC=$(dirname "$(dirname "$(readlink -f "$(uv python find 3.12)")")")
cp -a "$PYSRC" "$OUT/python"
rm -f "$OUT"/python/lib/python3.12/EXTERNALLY-MANAGED
P=$OUT/python/bin/python3.12
SP=$OUT/python/lib/python3.12/site-packages
echo "[$(date +%T)] packages"
uv pip install --python "$P" --torch-backend cu130 -r "$R/dist/requirements.txt"
uv pip install --python "$P" --no-deps "$PEARL/miner/miner-utils" "$PEARL/miner/pearl-gateway" "$PEARL/miner/miner-base"
uv pip install --python "$P" --no-deps "$PEARL/py-pearl-mining"
echo "[$(date +%T)] app"
cp -a "$R/dist/uwc_miner" "$OUT/app/"
cp -a "$R/miner/ampere/uwc_ampere" "$OUT/app/"
mkdir -p "$OUT/app/runtime" "$OUT/app/verifier"
cp -a "$R/runtime/agent" "$R/runtime/detgpu" "$R/runtime/detrt" "$R/runtime/detgen" "$R/runtime/artifacts" "$OUT/app/runtime/"
cp -a "$R/verifier/forced_errors.py" "$OUT/app/verifier/"
find "$OUT/app" -name __pycache__ -prune -exec rm -rf {} +
printf '../../../../app\n../../../../app/runtime\n' > "$SP/uwc_app.pth"
"$P" -m compileall -q "$OUT/app" || true
echo "[$(date +%T)] binaries and network files"
for b in pearld uwcctl prlctl; do cp "$GOBIN/$b" "$OUT/bin/"; strip "$OUT/bin/$b" 2>/dev/null || true; done
cp "$R/dist/testnet.json" "$OUT/share/"
MODEL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["name"])' "$R/dist/testnet.json")
cp "$NETDIR/models/$MODEL.json" "$NETDIR/models/$MODEL.reg.json" "$OUT/share/models/"
cp "$NETDIR"/pool/*.json "$OUT/share/pool/"
cp "$R/dist/uwc-miner" "$OUT/"
cp "$R/dist/README.txt" "$OUT/"
echo "$V" > "$OUT/VERSION"
# the runtime hash the chain registered covers the deterministic runtime's sources: check the copy
"$P" -E -s - "$OUT/share/models/$MODEL.json" <<'PY'
import json, sys
from agent.model_info import runtime_hash
info = json.load(open(sys.argv[1]))
want = info["runtime_hash"]
got = runtime_hash(info.get("gen", False)).hex()
print("runtime hash", "ok" if got == want else f"MISMATCH {got} != {want}")
sys.exit(got != want)
PY
echo "[$(date +%T)] tarball"
du -sh "$OUT"
tar -C "$OUTDIR" -cf - "uwc-miner-$V" | pigz -6 > "$OUTDIR/uwc-miner-$V-linux-x64.tar.gz"
( cd "$OUTDIR" && sha256sum "uwc-miner-$V-linux-x64.tar.gz" > "uwc-miner-$V-linux-x64.tar.gz.sha256" )
ls -la "$OUTDIR"/uwc-miner-$V-linux-x64.tar.gz*
echo "[$(date +%T)] done: $OUT"
