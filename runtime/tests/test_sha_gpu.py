"""detgpu.sha must reproduce agent.proto.stage_out_leaves / stage_output_root exactly (run on a GPU box)."""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.proto import stage_out_leaves, stage_output_root  # noqa: E402
from detgpu.sha import sout_leaves, stage_output_roots  # noqa: E402

dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
torch.cuda.set_device(torch.device(dev))
g = torch.Generator(device=dev).manual_seed(0)
for B, T, D in ((2, 128, 2048), (3, 5, 2048), (1, 300, 2048), (2, 7, 300), (1, 1, 256), (1, 3, 2560)):
    x = torch.randn(B, T, D, generator=g, device=dev) * 3
    got = sout_leaves(x)
    roots = stage_output_roots(x)
    xc = x.cpu().numpy()
    for b in range(B):
        ref = stage_out_leaves(list(xc[b]))
        assert [bytes(got[b, t]) for t in range(T)] == ref, (B, T, D, b)
        assert roots[b] == stage_output_root(list(xc[b])), (B, T, D, b)
    print(f"B={B} T={T} D={D}: leaves and roots identical")
x = torch.randn(8, 128, 2048, generator=g, device=dev)
for _ in range(2):
    torch.cuda.synchronize(); t = time.time()
    sout_leaves(x)
    torch.cuda.synchronize(); dt = time.time() - t
t = time.time(); [stage_output_root(list(r)) for r in x.cpu().numpy()]; dc = time.time() - t
print(f"8x128x2048 (8 MB): GPU {dt * 1000:.1f} ms, CPU reference (copy + hash) {dc * 1000:.0f} ms")
print("SHA GPU MATCHES")
