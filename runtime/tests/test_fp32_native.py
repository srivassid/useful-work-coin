"""Are torch's native CUDA fp32 + - * / sqrt bit-identical to detrt.core's emulated fp32
(float64 op, one rounding to float32)? Random values over the whole fp32 range, including
denormals, zeros, signs and values near overflow, plus values in the ranges the model uses."""
import sys

import torch

dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
torch.backends.cuda.matmul.allow_tf32 = False
F32, F64 = torch.float32, torch.float64
g = torch.Generator(device=dev).manual_seed(1)
N = 1 << 24


def rand_bits(n):
    """Uniform over all finite fp32 bit patterns (so every exponent, denormals included)."""
    b = torch.randint(-(2 ** 31), 2 ** 31 - 1, (n,), generator=g, device=dev, dtype=torch.int64).to(torch.int32)
    x = b.view(F32)
    return torch.where(torch.isfinite(x), x, torch.zeros_like(x))


def rand_model(n):
    """Values like activations: normal with random scale 2^-20..2^20."""
    e = torch.randint(-20, 21, (n,), generator=g, device=dev).to(F32)
    return torch.randn(n, generator=g, device=dev) * torch.pow(2.0, e)


bad = 0
for name, gen in (("all-bits", rand_bits), ("model-range", rand_model)):
    for rep in range(4):
        a, b = gen(N), gen(N)
        ad, bd = a.to(F64), b.to(F64)
        tests = {
            "add": (a + b, (ad + bd).to(F32)),
            "sub": (a - b, (ad - bd).to(F32)),
            "mul": (a * b, (ad * bd).to(F32)),
            "div": (a / b, (ad / bd).to(F32)),
            "sqrt": (torch.sqrt(a.abs()), torch.sqrt(ad.abs()).to(F32)),
        }
        for op, (x, y) in tests.items():
            same = (x.view(torch.int32) == y.view(torch.int32)) | (torch.isnan(x) & torch.isnan(y))
            nbad = int((~same).sum())
            if nbad or rep == 0:
                i = int((~same).nonzero()[0]) if nbad else 0
                ex = f" e.g. {a[i].item()!r} {op} {b[i].item()!r}: native {x[i].item()!r} vs emulated {y[i].item()!r}" if nbad else ""
                print(f"{name:11s} {op:4s} rep {rep}: {nbad} of {N} differ{ex}")
            bad += nbad
print("ALL IDENTICAL" if bad == 0 else f"{bad} DIFFERENCES")
