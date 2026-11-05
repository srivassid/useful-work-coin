"""Determinism of the reference runtime: prefill == decode == teacher forcing, batch invariance,
CPU == GPU, on a tiny random Qwen3 (no download)."""
import pytest
import torch

from detrt import core
from detrt.qwen3 import DetQwen3

core.setup_determinism()
DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=1024, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=64, max_position_embeddings=4096,
                      rms_norm_eps=1e-6, tie_word_embeddings=True)
    m = Qwen3ForCausalLM(cfg).eval()
    # break the near-zero init so values exercise real ranges
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return m


@pytest.fixture(scope="module")
def hf():
    return tiny_model()


def test_det_exp_and_log_accuracy():
    z = torch.linspace(-87, 88, 200001)
    e = core.det_exp(z).double()
    rel = ((e - torch.exp(z.double())) / torch.exp(z.double())).abs().max().item()
    assert rel < 5e-6, rel  # = fp32 rounding of z*log2(e) at |z| ~ 88; < 2e-6 for |z| <= 30
    x = torch.logspace(-30, 30, 100001)
    err = (core.det_log(x).double() - torch.log(x.double())).abs().max().item()
    assert err < 1e-5, err


def test_exact_rowsum_order_and_length_independent():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(8, 3000, generator=g) * torch.logspace(-6, 4, 3000)
    s1 = core.exact_rowsum(x, 1 << 20)
    perm = torch.randperm(3000, generator=g)
    s2 = core.exact_rowsum(x[:, perm], 1 << 20)
    s3 = core.exact_rowsum(torch.cat([x, torch.zeros(8, 5000)], 1), 1 << 20)  # masked tail
    assert torch.equal(s1, s2) and torch.equal(s1, s3)
    assert torch.allclose(s1.double(), x.double().sum(1), rtol=1e-6)


@pytest.mark.parametrize("dev", DEVS)
def test_prefill_equals_decode_equals_teacher_forcing(hf, dev):
    m = DetQwen3(hf, dev, hadamard_block=64, mine=False)
    g = torch.Generator().manual_seed(2)
    prompt = torch.randint(0, 1024, (2, 37), generator=g).to(dev)
    toks, step_h = m.generate_greedy(prompt, 12)
    full = torch.cat([prompt, toks[:, :-1]], 1)
    h_tf = m.forward(full, m.new_cache(2, full.shape[1]), 0)
    assert torch.equal(h_tf[:, 36:], step_h), "teacher-forced hidden != decode hidden"


@pytest.mark.parametrize("dev", DEVS)
def test_batch_composition_invariance(hf, dev):
    m = DetQwen3(hf, dev, hadamard_block=64, mine=False)
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(0, 1024, (4, 50), generator=g).to(dev)
    h4 = m.forward(ids, m.new_cache(4, 50), 0)
    h1 = m.forward(ids[2:3], m.new_cache(1, 50), 0)
    assert torch.equal(h4[2:3], h1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_cpu_equals_gpu(hf):
    g = torch.Generator().manual_seed(4)
    ids = torch.randint(0, 1024, (2, 64), generator=g)
    mc, mg = DetQwen3(hf, "cpu", 64, mine=False), DetQwen3(hf, "cuda", 64, mine=False)
    hc = mc.forward(ids, mc.new_cache(2, 64), 0)
    hg = mg.forward(ids.cuda(), mg.new_cache(2, 64), 0).cpu()
    assert torch.equal(hc, hg)
    lc, lg = mc.logprobs(hc), mg.logprobs(hg.cuda()).cpu()
    assert torch.equal(lc, lg)


def test_close_to_hf_float(hf):
    """Sanity: the deterministic integer runtime tracks the float model (same argmax mostly)."""
    m = DetQwen3(hf, "cpu", hadamard_block=64, mine=False)
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(0, 1024, (1, 64), generator=g)
    with torch.no_grad():
        ref = hf(ids).logits.float()
    lp = m.logprobs(m.forward(ids, m.new_cache(1, 64), 0))
    agree = (lp.argmax(-1) == ref.argmax(-1)).float().mean().item()
    assert agree > 0.8, agree
