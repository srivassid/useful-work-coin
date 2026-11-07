"""A provider's batched GPU pass (Runtime.jobs_roots: several jobs in one forward pass, only
output leaves leaving the GPU) must commit each job to exactly what it would alone, and a stage
input recomputed on demand (Runtime.stage_input) must equal the rows and proofs the provider
used to save (Runtime.job_outputs). Run on a GPU box:  python tests/test_provider_batch.py cuda:0"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.compute import Runtime  # noqa: E402
from agent.proto import check_stage_input, StageCommit  # noqa: E402

T = os.path.expanduser("~/Documents/useful-work-coin/testnet")


def test_main():
    dev = os.environ.get("UWC_TEST_DEVICE", "cuda:0")
    import torch
    from agent.proto import pooled, stage_output_root
    from detgpu.kernels import pooled as pooled_gpu
    from detgpu.sha import digest_bytes, stage_roots_gpu
    torch.cuda.set_device(torch.device(dev))
    g = torch.Generator().manual_seed(1)
    for n in (1, 2, 3, 5, 7, 64, 128, 129, 300):  # trees of every shape, rows of 2 blocks + a partial one
        x = (torch.randn((3, n, 600), generator=g) * 7).to(dev)
        got = digest_bytes(stage_roots_gpu(x))
        h = x.cpu().numpy()
        for b in range(3):
            assert bytes(got[b]) == stage_output_root(h[b]), (n, b)
            assert np.array_equal(pooled_gpu(x)[b].cpu().numpy().view(np.uint32), pooled(h[b]).view(np.uint32)), (n, b)
    print("stage_roots_gpu and pooled on the GPU: equal to agent.proto for T = 1..300")
    rt = Runtime(T + "/models/qwen3-1.7b.json", dev)
    units = json.load(open(T + "/pool/wiki8x128.units.json"))["units"]
    jobs = [units[3]["batches"], units[4]["batches"][:3], units[5]["batches"][:5]]
    alone = [rt.job_outputs(b) for b in jobs]
    for chunk in (4, 32):  # passes that split jobs, and one pass holding all of them
        got = rt.jobs_roots(jobs, tokens=chunk * 128)
        for i, (per, (roots, embs)) in enumerate(zip(alone, got)):
            assert roots == [r for rs, _, _ in per for r in rs], (chunk, i)
            assert all(np.array_equal(e.view(np.uint32), p[1].view(np.uint32)) for e, p in zip(embs, per)), (chunk, i)
    print(f"jobs_roots: {len(jobs)} jobs, chunks 4 and 32: roots and pooled outputs equal to one job at a time")
    i, per = 0, alone[0]
    roots = [r for rs, _, _ in per for r in rs]
    sc = StageCommit(roots, [os.urandom(32) for _ in roots])
    for stage in (1, 6, 2 * rt.G + 3, len(roots) - 1):
        b, g = divmod(stage, rt.G)
        pk = rt.stage_input(jobs[i], stage)
        old = per[b][2][g]
        assert np.array_equal(pk["rows"].view(np.uint32), old["rows"].view(np.uint32)), stage
        assert pk["proofs"] == old["proofs"], stage
        import base64
        doc = {"stage": stage, "shape": list(pk["rows"].shape), "proofs": pk["proofs"], "reveal": sc.reveal(stage - 1),
               "rows": base64.b64encode(pk["rows"].astype("<f4").tobytes()).decode()}
        check_stage_input(json.loads(json.dumps(doc)), stage, sc.root, len(roots))
    print("stage_input: rows and proofs equal to the saved ones; accepted by check_stage_input")
    print("PROVIDER BATCH OK")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        os.environ["UWC_TEST_DEVICE"] = sys.argv[1]
    test_main()
