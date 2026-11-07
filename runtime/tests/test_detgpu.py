"""detgpu.job_outputs must equal detrt.fast.run_stages(keep_rows=False) bit for bit, for every
group of a job, at several batch sizes and lengths (run on a GPU box)."""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from detgpu.stages import job_outputs  # noqa: E402
from detrt.fast import run_stages  # noqa: E402


def reference(m, seqs, G):
    lpg = len(m.layers) // G
    outs, xs = [], None
    for g in range(G):
        ys = run_stages(m, [torch.tensor(s) for s in seqs] if g == 0 else None, xs, g * lpg, (g + 1) * lpg, keep_rows=False)
        outs.append(np.stack(ys))
        xs = [torch.from_numpy(y) for y in ys]
    return outs


if __name__ == "__main__":
    from agent.model_info import load
    info = json.load(open(os.path.expanduser("~/Documents/useful-work-coin/testnet/models/qwen3-1.7b.json")))
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    torch.cuda.set_device(torch.device(dev))
    m = load(info["model_dir"], dev, info["rotation_block"], info["cfg"]["MaxPos"])
    G = info["layer_groups"]
    rng = np.random.default_rng(0)
    units = json.load(open(os.path.expanduser("~/Documents/useful-work-coin/testnet/pool/wiki8x128.units.json")))["units"]
    cases = [("wiki 8x128", units[0]["batches"]),
             ("random 3x128", rng.integers(0, 150000, (3, 128)).tolist()),
             ("random 1x512", rng.integers(0, 150000, (1, 512)).tolist()),
             ("random 2x300", rng.integers(0, 150000, (2, 300)).tolist())]
    for name, seqs in cases:
        ref = reference(m, seqs, G)
        torch.cuda.synchronize(); t = time.time()
        got = job_outputs(m, torch.tensor(seqs), G)
        torch.cuda.synchronize(); dt = time.time() - t
        for g, (a, b) in enumerate(zip(ref, got)):
            b = b.cpu().numpy()
            assert a.shape == b.shape, (name, g, a.shape, b.shape)
            nd = int((a.view(np.uint32) != b.view(np.uint32)).sum())
            assert nd == 0, f"{name}: group {g}: {nd} of {a.size} values differ"
        import hashlib
        dig = hashlib.sha256(b"".join(np.ascontiguousarray(g.cpu().numpy()).tobytes() for g in got)).hexdigest()[:16]
        print(f"{name}: all {G} groups bit-identical ({dt:.2f}s), outputs sha256 {dig}")
    print("DETGPU MATCHES DETRT")
