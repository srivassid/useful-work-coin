"""Generation: decoding with the KV cache gives exactly the tokens the head stage recomputes over
the teacher-forced sequence, on CPU and GPU alike, greedy and with seeded Gumbel-max sampling."""
import numpy as np
import pytest
import torch

from detgen.generate import generate
from detgen.head import (SENTINEL, Sampling, gen_output_root, head_output_root, head_program, head_weight_rows,
                         input_row, logits, mismatch, run_head_stage, sequence, uniform)
from detrt import core
from detrt.qwen3 import DetQwen3
from detrt.stage import run_stage

core.setup_determinism()
DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
V = 2504  # three vocabulary chunks of 1024, the last one partial (a multiple of 8: torch._int_mm)


def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=V, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=64, max_position_embeddings=4096,
                      rms_norm_eps=1e-6, tie_word_embeddings=True)
    m = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return m


@pytest.fixture(scope="module")
def hf():
    return tiny_model()


PROMPT = [int(x) for x in torch.randint(0, V, (9,), generator=torch.Generator().manual_seed(3))]
SAMPLERS = [Sampling(0), Sampling(800, bytes(range(32)))]


def recompute(m, seq, s, rows=True):
    r2 = run_stage(m, torch.tensor(seq), None, 0, len(m.layers)).output
    return run_head_stage(m, torch.from_numpy(r2), s, rows=rows)


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("s", SAMPLERS, ids=["greedy", "t0.8"])
def test_decode_equals_head_stage(hf, dev, s):
    m = DetQwen3(hf, dev, hadamard_block=64, mine=False)
    seq = generate(m, PROMPT, 24, s)
    assert seq[:9] == PROMPT and len(seq) == 24
    tr = recompute(m, seq, s)
    assert mismatch(seq, tr.output, 9) == -1 and sequence(PROMPT, tr.output) == seq
    assert len(tr.rows) == len(head_program(2, 24, 3)) == 23 * (2 + 2 * 3 + 1)
    assert np.array_equal(recompute(m, seq, s, rows=False).output, tr.output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("s", SAMPLERS, ids=["greedy", "t0.8"])
def test_cpu_equals_gpu(hf, s):
    seqs, roots = [], []
    for dev in ("cpu", "cuda"):
        m = DetQwen3(hf, dev, hadamard_block=64, mine=False)
        seqs.append(generate(m, PROMPT, 20, s))
        tr = recompute(m, seqs[-1], s)
        roots.append((tr.root, head_output_root(tr.output)))
    assert seqs[0] == seqs[1] and roots[0] == roots[1]


@pytest.mark.parametrize("dev", DEVS)
def test_logits_equal_the_runtimes_lm_head(hf, dev):
    m = DetQwen3(hf, dev, hadamard_block=64, mine=False)
    hn = m.forward(torch.tensor([PROMPT], device=m.device), m.new_cache(1, 9), 0)[0]
    q, sx = core.quant_act(hn, m.lm_head.bs)
    assert torch.equal(logits(m, q, sx, 0, V), m.lm_head(hn))


def test_a_wrong_token_is_found(hf):
    m = DetQwen3(hf, "cpu", hadamard_block=64, mine=False)
    s = SAMPLERS[1]
    seq = generate(m, PROMPT, 20, s)
    bad = list(seq)
    bad[14] = (bad[14] + 1) % V
    assert mismatch(bad, recompute(m, bad, s).output, 9) == 13  # position 13's sample is token 14


def test_sampling_follows_the_seed(hf):
    m = DetQwen3(hf, "cpu", hadamard_block=64, mine=False)
    a = generate(m, PROMPT, 30, Sampling(1000, b"\x01" * 32))
    b = generate(m, PROMPT, 30, Sampling(1000, b"\x01" * 32))
    c = generate(m, PROMPT, 30, Sampling(1000, b"\x02" * 32))
    assert a == b and a != c


def test_uniform_is_exact_and_uniform():
    s = Sampling(1000, b"\x07" * 32)
    u = uniform(s, np.arange(64)[:, None], np.arange(4096)[None, :])
    k = (u.astype(np.float64) * 2 ** 24)
    assert np.all(k == np.round(k)) and np.all(k % 2 == 1)  # odd multiples of 2^-24: exact
    assert 0 < u.min() and u.max() < 1 and abs(u.mean() - 0.5) < 0.01
    g = -np.log(-np.log(u.astype(np.float64)))
    assert abs(g.mean() - 0.5772) < 0.02  # Gumbel mean = Euler-Mascheroni constant


def test_head_weight_rows(hf):
    m = DetQwen3(hf, "cpu", hadamard_block=64, mine=False)
    rows = list(head_weight_rows(m))
    assert rows[0][0] == "norm" and len(rows) == 1 + V and rows[1][1].shape == (256 + 1,)


def test_input_row_and_output_root():
    row = input_row(PROMPT, 16)
    assert row[:9] == PROMPT and row[9:] == [SENTINEL] * 7
    out = np.arange(30, dtype=np.int32).reshape(15, 2)
    r = head_output_root(out)
    assert gen_output_root([r]) != gen_output_root([r, r]) and len(r) == 32

