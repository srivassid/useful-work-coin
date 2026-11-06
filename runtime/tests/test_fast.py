"""detrt.fast.run_stages must produce exactly stage.run_stage's rows and roots, for every
sequence of a batch, for first-group and later-group stages (run on a GPU box)."""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from detrt.fast import run_stages  # noqa: E402
from detrt.stage import run_stage  # noqa: E402


def check(m, seqs, verbose=True):
    B = len(seqs)
    ref0 = [run_stage(m, s, None, 0, 4) for s in seqs]
    t = time.time()
    f0 = run_stages(m, seqs, None, 0, 4)
    tf0 = time.time() - t
    xin = [torch.from_numpy(r.output) for r in ref0]
    ref1 = [run_stage(m, None, x, 4, 8) for x in xin]
    f1 = run_stages(m, None, xin, 4, 8)
    for ref, fast in ((ref0, f0), (ref1, f1)):
        for b in range(B):
            assert len(ref[b].rows) == len(fast[b].rows)
            for i, (a, c) in enumerate(zip(ref[b].rows, fast[b].rows)):
                assert a.dtype == c.dtype and a.shape == c.shape and np.array_equal(a.view(np.uint32), c.view(np.uint32)), \
                    (b, i, ref[b].leaves[i])
            assert ref[b].root == fast[b].root and np.array_equal(ref[b].output, fast[b].output)
    return tf0


if __name__ == "__main__":
    from agent.compute import Runtime
    D = os.path.expanduser("~/Documents/useful-work-coin/devnet")
    rt = Runtime(D + "/models/qwen3-1.7b.json", sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
    units = json.load(open(D + "/pool/wiki128.units.json"))["units"]
    seqs = [torch.tensor(u["tokens"]) for u in units[:3]]
    check(rt.m, seqs)
    print("3 sequences x 2 stages: every row and root identical to run_stage")
    for B in (1, 8, 32):
        s = [torch.tensor(u["tokens"]) for u in units[:B]]
        torch.cuda.synchronize(); t = time.time()
        run_stages(rt.m, s, None, 0, 4)
        torch.cuda.synchronize(); dt = time.time() - t
        print(f"B={B:3d}: {dt:6.2f}s per stage batch, {dt/B:5.2f}s per sequence-stage")
