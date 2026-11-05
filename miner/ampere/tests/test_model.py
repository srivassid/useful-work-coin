"""MinedLinear in 'mined' mode is bit-identical to the same integer math through cuBLAS ('int')."""
import numpy as np
import torch
import torch.nn as nn

from uwc_ampere import model as M


def test_mined_equals_int_path():
    torch.manual_seed(0)
    lins = [nn.Linear(2560, 4096, bias=False), nn.Linear(2560, 1024, bias=False)]
    a = M.MinedLinear(lins, "cuda", mode="int")
    b = M.MinedLinear(lins, "cuda", mode="mined")
    assert b.mineable
    b.setup_block(lambda k: bytes(32), lambda k: np.zeros(8, np.uint32))
    x = torch.randn(3, 333, 2560, device="cuda", dtype=torch.bfloat16)  # m=999 -> padded to 1024
    ya, yb = a(x), b(x)
    assert ya.shape == yb.shape == (3, 333, 5120)
    assert torch.equal(ya, yb)
    ref = x.float() @ torch.cat([l.weight for l in lins]).cuda().t()
    rel = ((ya.float() - ref).norm() / ref.norm()).item()
    assert rel < 0.03, rel


def test_fused_quantizer_matches_torch():
    torch.manual_seed(1)
    x = (torch.randn(300, 2560, device="cuda") * torch.linspace(0.1, 8, 2560, device="cuda")).to(torch.bfloat16)
    for bs in (0, 128, 512):
        q, s = __import__("uwc_ampere.ops", fromlist=["ops"]).quant_rows_had(x, bs)
        H = M.hadamard(bs, "cuda") if bs else None
        qr, sr = M.quant_rows(M.rotate(x.float(), H))
        assert torch.allclose(s, sr, rtol=1e-5)
        assert (q.int() - qr.int()).abs().max().item() <= 1          # only float rounding ties differ
        assert (q != qr).float().mean().item() < 1e-3
