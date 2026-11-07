"""detgpu attention on tensor cores (attention_mma) must equal the dp4a kernel (attention_tile,
itself checked against detrt by tests/test_detgpu.py) bit for bit, for lengths that do and do
not fill a 16/32-row tile or a 64-key chunk. Run on a GPU box:  python tests/test_attention_mma.py cuda:0"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from detgpu import kernels  # noqa: E402


def case(B, T, H=16, Hkv=8, seed=0, dev="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    N = B * T
    qq = torch.randint(-127, 128, (N, H, 128), generator=g).to(dev)
    kq = torch.randint(-127, 128, (N, Hkv, 128), generator=g).to(dev)
    vq = torch.randint(-127, 128, (N, Hkv, 128), generator=g).to(dev)
    qs = (torch.rand((N, H, 1), generator=g) * 0.02 + 1e-4).to(dev)
    ks = (torch.rand((N, Hkv, 1), generator=g) * 0.02 + 1e-4).to(dev)
    vs = (torch.rand((N, Hkv, 1), generator=g) * 0.05 + 1e-4).to(dev)
    return qq, qs, kq, ks, vq, vs


def run(args, B, T, mma):
    kernels.MMA = mma
    torch.cuda.synchronize()
    t = time.time()
    o = kernels.attention(*args, B, T, 16, 8, float(np.float32(1 / np.sqrt(128))))
    torch.cuda.synchronize()
    return o, time.time() - t


def test_main():
    dev = os.environ.get("UWC_TEST_DEVICE", "cuda:0")
    torch.cuda.set_device(torch.device(dev))
    for B, T in [(1, 1), (2, 5), (1, 16), (3, 17), (1, 63), (2, 64), (1, 65), (4, 128), (2, 129), (1, 300), (2, 512), (1, 700)]:
        args = case(B, T, seed=B * 1000 + T, dev=dev)
        a, _ = run(args, B, T, False)
        b, _ = run(args, B, T, True)
        nd = int((a.view(torch.int32) != b.view(torch.int32)).sum())
        assert nd == 0, f"B={B} T={T}: {nd} of {a.numel()} values differ"
        print(f"B={B} T={T}: identical (tile {kernels._mma_qt(T, torch.cuda.current_device())} rows)")
    for B, T in [(16, 128), (16, 512)]:
        args = case(B, T, dev=dev)
        run(args, B, T, True)
        ta = min(run(args, B, T, False)[1] for _ in range(3))
        tb = min(run(args, B, T, True)[1] for _ in range(3))
        print(f"B={B} T={T}: dp4a {ta * 1e3:.2f} ms, tensor cores {tb * 1e3:.2f} ms per layer")
    kernels.MMA = True
    print("ATTENTION MMA IDENTICAL")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        os.environ["UWC_TEST_DEVICE"] = sys.argv[1]
    test_main()
