"""Compare crosscheck outputs from several machines: python -m bench.compare_crosscheck a b [c ...]"""
import json
import sys

import numpy as np


def main(paths):
    runs = [(p, json.load(open(p + ".json")), np.load(p + ".npz")) for p in paths]
    names = [r[1]["device"] for r in runs]
    print("devices:", names)
    keys = [k for k in runs[0][1] if k.startswith("L") and k != "L3_layer_sha" and not k.startswith("L4")]
    for k in keys:
        vals = [r[1].get(k) for r in runs]
        present = [v for v in vals if v is not None]
        status = "IDENTICAL" if len(set(map(str, present))) == 1 else "DIFFERS"
        print(f"  {k:24s} {status:9s} {vals}")
    # element-level detail for arrays present in all runs
    common = set(runs[0][2].files)
    for r in runs[1:]:
        common &= set(r[2].files)
    for k in sorted(common):
        a0 = runs[0][2][k]
        for p, js, arr in runs[1:]:
            b = arr[k]
            ne = (a0 != b)
            msg = f"  {k:24s} {names[0]} vs {js['device']}: {ne.mean() * 100:.4f}% elements differ"
            if ne.any() and a0.dtype.kind in "if":
                msg += f", max |diff| = {np.abs(a0.astype(np.float64) - b.astype(np.float64)).max():.4g}"
            print(msg)
    gpu = [r for r in runs if "L3_layer_sha" in r[1]]
    if len(gpu) >= 2:
        a_, b_ = gpu[0][1]["L3_layer_sha"], gpu[1][1]["L3_layer_sha"]
        first = next((i for i, (x, y) in enumerate(zip(a_, b_)) if x != y), None)
        print(f"  L3 decoder layers identical: {sum(x == y for x, y in zip(a_, b_))}/{len(a_)}; first differing layer: {first}")
    for p, js, _ in runs:
        if "L4_prefill_vs_decode_identical" in js:
            print(f"  L4 [{js['device']}] prefill==decode: {js['L4_prefill_vs_decode_identical']}, "
                  f"{js['L4_frac_elements_differ'] * 100:.2f}% elements differ, max |diff| {js['L4_max_abs_diff']:.4g}")


if __name__ == "__main__":
    main(sys.argv[1:])
