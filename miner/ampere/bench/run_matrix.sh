#!/usr/bin/env bash
# usage: run_matrix.sh MODEL_DIR TAG [TOKENS]
set -u
PY=~/Documents/useful-work-coin/port/.venv/bin/python
MODEL=$1; TAG=$2; TOK=${3:-8192}
R=~/Documents/useful-work-coin/results
for comp in "" "--compile"; do
  for mode in ${MODES:-bf16 plain mined}; do
    name=${TAG}_${mode}${comp:+_compiled}_t${TOK}
    $PY -m bench.lambda_job --model "$MODEL" --data ~/Documents/useful-work-coin/models/wikitext2 --mode $mode $comp \
        --tokens $TOK ${EXTRA:-} --out $R/$name.json > $R/$name.log 2>&1 || { echo "$name FAILED"; tail -n 5 $R/$name.log; continue; }
    $PY - "$R/$name.json" "$name" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
for j in ("embed", "score"):
    v = d[j]
    lam = f" lambda_gpu={v['lambda_gpu']:.3f} lambda_wall={v['lambda_wall']:.3f}" if "lambda_gpu" in v else ""
    print(f"{sys.argv[2]:34s} {j:5s} gpu={v['gpu_s']*1e3:7.1f}ms wall={v['wall_s']*1e3:7.1f}ms tok/s(gpu)={v['tokens_per_gpu_s']:8.0f}{lam}")
if "noise_gpu_spread" in d:
    print(f"{'':34s} noise TMAC/s gpu mean={d['noise_macs_per_gpu_s']/1e12:.1f} spread={[round(x/1e12,1) for x in d['noise_gpu_spread']]}")
PYEOF
  done
done
