"""Find the first op where CPU and GPU diverge in DetQwen3 (tiny model)."""
import torch
from detrt import core
from detrt.qwen3 import DetQwen3
from tests.test_detrt import tiny_model

core.setup_determinism()
hf = tiny_model()
mc, mg = DetQwen3(hf, "cpu", 64, mine=False), DetQwen3(hf, "cuda", 64, mine=False)
ids = torch.randint(0, 1024, (2, 64), generator=torch.Generator().manual_seed(4))


def cmp(name, a, b):
    b = b.cpu()
    same = torch.equal(a, b)
    extra = "" if same else f"  {((a != b).float().mean() * 100):.3f}% differ, max {((a.double() - b.double()).abs().max()):.3g}"
    print(f"{name:22s} {'SAME' if same else 'DIFF'}{extra}")
    return same


L_c, L_g = mc.layers[0], mg.layers[0]
x_c, x_g = mc.embed[ids], mg.embed[ids.cuda()]
cmp("embed", x_c, x_g)
cmp("Wq", L_c["qkv"].Wq, L_g["qkv"].Wq); cmp("sw", L_c["qkv"].sw, L_g["qkv"].sw)
h_c, h_g = core.det_rmsnorm(x_c, L_c["in_norm"], mc.eps), core.det_rmsnorm(x_g, L_g["in_norm"], mg.eps)
cmp("x*x", x_c * x_c, x_g * x_g)
cmp("exact_rowsum", core.exact_rowsum(x_c * x_c, 256), core.exact_rowsum(x_g * x_g, 256))
cmp("rmsnorm", h_c, h_g)
cmp("fwht", core.fwht(h_c.reshape(-1, 256), 64), core.fwht(h_g.reshape(-1, 256), 64))
qc, sc = core.quant_act(h_c.reshape(-1, 256), 64); qg, sg = core.quant_act(h_g.reshape(-1, 256), 64)
cmp("quant q", qc, qg); cmp("quant s", sc, sg)
ac, ag = L_c["qkv"].int_gemm(qc), L_g["qkv"].int_gemm(qg)
cmp("int_gemm", ac.to(torch.int64), ag.to(torch.int64))
cmp("linear out", L_c["qkv"](h_c), L_g["qkv"](h_g))
z = torch.linspace(-30, 5, 100000)
cmp("det_exp", core.det_exp(z), core.det_exp(z.cuda()))
cmp("det_log", core.det_log(z.abs() + 1e-3), core.det_log(z.abs().cuda() + 1e-3))
cmp("det_silu", core.det_silu(z), core.det_silu(z.cuda()))
cmp("div", z / (z.abs() + 1), z.cuda() / (z.abs().cuda() + 1))
cmp("sqrt", torch.sqrt(z.abs()), torch.sqrt(z.abs().cuda()))
print("full attention layer:")
cmp("attn", mc._attention(L_c, h_c, mc.new_cache(2, 64)[0], 0), mg._attention(L_g, h_g, mg.new_cache(2, 64)[0], 0))
