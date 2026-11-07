"""The agent's Runtime must commit to exactly the same things with engine detgpu as with detrt:
stage roots, pooled outputs, served stage-input rows and their Merkle proofs, and a verifier's
root recomputed from a served input (run on a GPU box)."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.compute import Runtime  # noqa: E402

T = os.path.expanduser("~/Documents/useful-work-coin/testnet")
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
rt = Runtime(T + "/models/qwen3-1.7b.json", dev)
units = json.load(open(T + "/pool/wiki8x128.units.json"))["units"]
batches = units[3]["batches"]
res = {}
for eng in ("detrt", "detgpu"):
    rt.gpu = eng == "detgpu"
    t = time.time()
    res[eng] = rt.job_outputs(batches)
    print(f"{eng}: {time.time() - t:.2f}s (GPU part {rt.last_gpu_s})")
a, b = res["detrt"], res["detgpu"]
assert len(a) == len(b)
for i, ((ra, ea, ia), (rb, eb, ib)) in enumerate(zip(a, b)):
    assert ra == rb, f"batch {i}: stage roots differ"
    assert np.array_equal(ea.view(np.uint32), eb.view(np.uint32)), f"batch {i}: pooled output differs"
    assert ia.keys() == ib.keys()
    for g in ia:
        assert np.array_equal(ia[g]["rows"].view(np.uint32), ib[g]["rows"].view(np.uint32)), (i, g)
        assert ia[g]["proofs"] == ib[g]["proofs"], (i, g)
print(f"{len(a)} batches x {rt.G} stages: roots, pooled outputs, input rows and proofs identical")
for g in (1, 4, 6):
    rows = a[2][2][g]["rows"]
    rt.gpu = False; r0 = rt.stage_root_from_input(rows, g)
    rt.gpu = True; r1 = rt.stage_root_from_input(rows, g)
    assert r0 == r1 == a[2][0][g], g
print("verifier recompute from a served input: identical, and equal to the provider's root")
print("AGENT ENGINES MATCH")
