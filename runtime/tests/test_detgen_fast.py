"""Fast generation (detgen.fast) is bit-identical to the reference: same tokens as
detgen.generate.generate for prompts of different lengths in one batch, and the same head-stage
outputs as detgen.head.run_head_stage, greedy and with Gumbel-max sampling."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from detrt import core  # noqa: E402
from detrt.qwen3 import DetQwen3  # noqa: E402

core.setup_determinism()
V = 2504
PROMPTS = [[int(x) for x in torch.randint(0, V, (n,), generator=torch.Generator().manual_seed(n))] for n in (9, 5, 13)]


@pytest.fixture(scope="module")
def m():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=V, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=128, max_position_embeddings=256,
                      rms_norm_eps=1e-6, tie_word_embeddings=True)
    hf = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in hf.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return DetQwen3(hf, "cuda", hadamard_block=64, mine=False, max_pos=256)


SAMPLERS = ["greedy", "t0.8"]


def sampling(name):
    from detgen.head import Sampling
    return Sampling(0) if name == "greedy" else Sampling(800, bytes(range(32)))


@pytest.mark.parametrize("name", SAMPLERS)
def test_fast_generate_equals_reference(m, name):
    from detgen import fast
    from detgen.generate import generate
    s = sampling(name)
    got = fast.generate(m, PROMPTS, 24, s)
    for p, g in zip(PROMPTS, got):
        assert g == generate(m, p, 24, s)


@pytest.mark.parametrize("name", SAMPLERS)
def test_fast_head_outputs_equal_reference(m, name):
    from detgen import fast
    from detgen.head import run_head_stage
    from detgpu.stages import job_outputs
    s = sampling(name)
    seqs = fast.generate(m, PROMPTS, 24, s)
    r2 = job_outputs(m, torch.tensor(seqs, device="cuda"), 2)[-1]
    out = fast.head_outputs(m, r2, s, rows=7)  # small row chunks: exercises the chunking
    for b, seq in enumerate(seqs):
        ref = run_head_stage(m, r2[b].cpu(), s, rows=False).output
        assert np.array_equal(out[b], ref)
        assert [int(x) for x in out[b, len(PROMPTS[b]) - 1:, 0]] == seq[len(PROMPTS[b]):]


def test_gumbel_kernel_equals_reference(m):
    from detgen.head import Sampling, gumbel
    from detgen.kernels import gumbel_scores
    s = Sampling(1300, b"\x05" * 32)
    lg = torch.randn(4, 3000, device="cuda")
    pos = torch.tensor([0, 7, 511, 1 << 20], device="cuda")
    got = gumbel_scores(lg, pos, s.key(), float(s.temperature(lg[:1, :1]).item()))
    from detrt.core import fadd, fdiv
    want = fadd(fdiv(lg, s.temperature(lg)), gumbel(s, pos, 0, 3000, "cuda"))
    assert torch.equal(got, want)
