"""Stage traces reproduce the runtime exactly, chain across layer groups, and are device-independent."""
import numpy as np
import pytest
import torch

from detrt import core
from detrt.qwen3 import DetQwen3
from detrt.stage import program, run_stage, weights_tree
from tests.test_detrt import DEVS, tiny_model

core.setup_determinism()


@pytest.fixture(scope="module")
def hf():
    return tiny_model()


@pytest.mark.parametrize("dev", DEVS)
def test_stage_matches_forward_and_chains(hf, dev):
    m = DetQwen3(hf, dev, hadamard_block=64, mine=False)
    toks = torch.randint(0, 1024, (23,), generator=torch.Generator().manual_seed(1))
    full = run_stage(m, toks, None, 0, 2)
    ref = m.forward(toks[None].to(dev), m.new_cache(1, 23), 0)[0]
    assert torch.equal(core.det_rmsnorm(torch.from_numpy(full.output).to(dev), m.norm, m.eps), ref)
    a = run_stage(m, toks, None, 0, 1)
    b = run_stage(m, None, torch.from_numpy(a.output), 1, 2)
    assert np.array_equal(b.output, full.output)
    assert len(full.rows) == len(program({"H": m.H, "Hkv": m.Hkv, "hd": m.hd}, 23, 0, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_stage_root_cpu_equals_gpu(hf):
    toks = torch.randint(0, 1024, (17,), generator=torch.Generator().manual_seed(2))
    rc = run_stage(DetQwen3(hf, "cpu", 64, mine=False), toks, None, 0, 2).root
    rg = run_stage(DetQwen3(hf, "cuda", 64, mine=False), toks, None, 0, 2).root
    assert rc == rg


def test_weights_root_device_independent(hf):
    r1, _ = weights_tree(DetQwen3(hf, "cpu", 64, mine=False))
    devs = ["cuda"] if torch.cuda.is_available() else ["cpu"]
    r2, _ = weights_tree(DetQwen3(hf, devs[0], 64, mine=False))
    assert r1 == r2
