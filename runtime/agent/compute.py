"""Stage roots and outputs with the deterministic runtime (detrt), as providers, verifiers and
watchers compute them. Stage s of a job is batch s // G through layer group s % G."""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from detrt.stage import merkle_root, run_stage

from .model_info import load
from .proto import pooled


class Runtime:
    def __init__(self, model_json: str, device: str, mine_socket: str | None = None, engine: str = "detgpu",
                 mine: bool | None = None):
        info = json.load(open(model_json))
        self.info = info
        self.device = torch.device(device)
        # detgpu: bit-identical to detrt (tests/test_detgpu.py), native fp32 and fused kernels, stage
        # roots hashed on the GPU. detrt.fast stays the reference and the CPU path.
        self.gpu = engine == "detgpu" and self.device.type == "cuda"
        self.last_gpu_s = None
        self.G = info["layer_groups"]
        t = time.time()
        self.m = load(info["model_dir"], device, info["rotation_block"], info["cfg"]["MaxPos"], mine=bool(mine_socket) if mine is None else mine)
        self.miner = None
        if mine_socket:
            from .mining import JobMiner
            self.miner = JobMiner(self.m, mine_socket)
        self.lpg = len(self.m.layers) // self.G
        self.cache: dict[tuple, tuple[list[bytes], np.ndarray]] = {}
        print(f"runtime: {info['model_dir']} on {device} in {time.time() - t:.0f}s", flush=True)

    def bind(self):
        """Make this runtime's GPU current in the calling thread. Both torch and CuPy (the mining
        kernels) keep the current device per thread, and a new thread starts on GPU 0, so jobs
        run on the agent's worker thread must bind before touching a GPU other than 0."""
        if self.device.type != "cuda":
            return
        torch.cuda.set_device(self.device)
        try:
            import cupy
            cupy.cuda.Device(self.device.index or 0).use()
        except ImportError:
            pass

    def outputs(self, seqs: list[list[int]], upto: int | None = None, chunk: int = 8):
        """Stage outputs (the rows each stage is committed by) for groups 0..upto of each
        sequence: forward pass only, batched over sequences. Returns per sequence a list of
        (T, D) float32 arrays, one per group."""
        from detrt.fast import run_stages
        self.bind()
        upto = self.G - 1 if upto is None else upto
        if self.miner is not None:
            self.miner.refresh()
        out = []
        if self.gpu:
            from detgpu.stages import job_outputs
            for c0 in range(0, len(seqs), chunk):
                outs = job_outputs(self.m, torch.tensor(seqs[c0:c0 + chunk], dtype=torch.long), self.G, upto)
                host = [o.cpu().numpy() for o in outs]
                out += [[h[b] for h in host] for b in range(len(seqs[c0:c0 + chunk]))]
            return out
        for c0 in range(0, len(seqs), chunk):
            toks = [torch.tensor(t, dtype=torch.long) for t in seqs[c0:c0 + chunk]]
            res = [[] for _ in toks]
            xs = None
            for g in range(upto + 1):
                ys = run_stages(self.m, toks if g == 0 else None, xs, g * self.lpg, (g + 1) * self.lpg, keep_rows=False)
                for r, y in zip(res, ys):
                    r.append(y)
                xs = [torch.from_numpy(y) for y in ys]
            out += res
        return out

    def batch(self, tokens: list[int], upto: int | None = None) -> tuple[list[bytes], np.ndarray | None]:
        """Output roots of groups 0..upto for one sequence, and its pooled output if all ran."""
        from .proto import stage_output_root
        outs = self.outputs([tokens], upto)[0]
        emb = pooled(outs[-1]) if len(outs) == self.G else None
        return [stage_output_root(o) for o in outs], emb

    def job_outputs(self, batches: list[list[int]]):
        """A provider's job: per batch (output roots of every stage, pooled output, and each later
        stage's input with Merkle proofs in the previous stage's output tree)."""
        from .proto import multi_proofs, stage_out_leaves
        if self.gpu:
            return self._job_outputs_gpu(batches)
        res = []
        for outs in self.outputs(batches):
            leaves = [stage_out_leaves(o) for o in outs]
            roots = [merkle_root(lv) for lv in leaves]
            inputs = {}
            for g in range(1, self.G):
                pr = multi_proofs(leaves[g - 1], range(len(leaves[g - 1])))
                inputs[g] = {"rows": outs[g - 1], "proofs": [[h.hex() for h in pr[t]] for t in range(len(pr))]}
            res.append((roots, pooled(outs[-1]), inputs))
        return res

    def _job_outputs_gpu(self, batches: list[list[int]], chunk: int = 16):
        """job_outputs with detgpu: every stage's output root from leaves hashed on the GPU. The
        GPU part (forward pass + hashing, where the job's GEMMs mine) is timed as last_gpu_s;
        copying the rows verifiers may ask for and the pooled outputs come after it."""
        from detgpu.sha import sout_leaves
        from detgpu.stages import job_outputs
        from .proto import multi_proofs
        self.bind()
        if self.miner is not None:
            self.miner.refresh()
        torch.cuda.synchronize()
        t = time.time()
        groups = []   # per chunk: (outs on GPU, leaves (G, B, T, 32))
        for c0 in range(0, len(batches), chunk):
            outs = job_outputs(self.m, torch.tensor(batches[c0:c0 + chunk], dtype=torch.long), self.G)
            groups.append((outs, [sout_leaves(o) for o in outs]))
        torch.cuda.synchronize()
        self.last_gpu_s = time.time() - t
        res = []
        for outs, leaves in groups:
            host = [o.cpu().numpy() for o in outs[:-1]] + [outs[-1].cpu().numpy()]
            for b in range(host[0].shape[0]):
                lv = [[bytes(leaves[g][b, tt]) for tt in range(leaves[g].shape[1])] for g in range(self.G)]
                roots = [merkle_root(x) for x in lv]
                inputs = {}
                for g in range(1, self.G):
                    pr = multi_proofs(lv[g - 1], range(len(lv[g - 1])))
                    inputs[g] = {"rows": host[g - 1][b], "proofs": [[h.hex() for h in pr[tt]] for tt in range(len(pr))]}
                res.append((roots, pooled(host[-1][b]), inputs))
        return res

    def jobs_roots(self, jobs: list[list[list[int]]], chunk: int = 32):
        """Several jobs through the GPU together: per job (stage output roots in stage order,
        pooled outputs). Sequences of all jobs share forward passes of up to `chunk` sequences
        (rows are independent, so each job's roots are what it would get alone; one pass per
        sequence length). Stage roots and pooled outputs are computed on the GPU, so only
        32 bytes per stage and one row per sequence leave it: a verifier's stage input is recomputed when asked for
        (stage_input). last_gpu_s is the forward pass + hashing, where the GEMMs mine."""
        if not self.gpu:
            res = []
            for bs in jobs:
                per = self.job_outputs(bs)
                res.append(([x for roots, _, _ in per for x in roots], [e for _, e, _ in per]))
            return res
        from detgpu.kernels import pooled as pooled_gpu
        from detgpu.sha import digest_bytes, stage_roots_gpu
        from detgpu.stages import job_outputs
        flat = [(i, b, seq) for i, bs in enumerate(jobs) for b, seq in enumerate(bs)]
        by_len: dict[int, list] = {}
        for item in flat:
            by_len.setdefault(len(item[2]), []).append(item)
        self.bind()
        if self.miner is not None:
            self.miner.refresh()
        torch.cuda.synchronize()
        t = time.time()
        done = []  # (items, stage roots (G, B, 8) words, pooled outputs (B, D)), all still on the GPU
        for items in by_len.values():
            for c0 in range(0, len(items), chunk):
                part = items[c0:c0 + chunk]
                outs = job_outputs(self.m, torch.tensor([x[2] for x in part], dtype=torch.long), self.G)
                done.append((part, torch.stack([stage_roots_gpu(o) for o in outs]), pooled_gpu(outs[-1])))
                del outs
        torch.cuda.synchronize()
        self.last_gpu_s = time.time() - t
        res = [[None] * len(bs) for bs in jobs]
        for part, roots, emb in done:
            roots, emb = digest_bytes(roots), emb.cpu().numpy()
            for r, (i, b, _) in enumerate(part):
                res[i][b] = ([bytes(roots[g, r]) for g in range(self.G)], emb[r])
        return [([x for roots, _ in per for x in roots], [e for _, e in per]) for per in res]

    def stage_input(self, batches: list[list[int]], stage: int) -> dict:
        """What a verifier asks a provider for: stage s's input rows (the previous stage's output,
        recomputed for that one sequence) and each row's Merkle proof in its output tree."""
        from .proto import multi_proofs, stage_out_leaves
        b, g = divmod(stage, self.G)
        assert g >= 1, "a stage of layer group 0 reads tokens"
        rows = np.ascontiguousarray(self.outputs([batches[b]], g - 1)[0][g - 1])
        leaves = stage_out_leaves(rows)
        pr = multi_proofs(leaves, range(len(leaves)))
        return {"rows": rows, "proofs": [[h.hex() for h in pr[t]] for t in range(len(pr))]}

    def stage_output(self, batches: list[list[int]], stage: int) -> np.ndarray:
        b, g = divmod(stage, self.G)
        return self.outputs([batches[b]], g)[0][g]

    def stage_root_from_input(self, rows: np.ndarray, g: int) -> bytes:
        """One stage (layer group g >= 1) from its served input rows: its output root."""
        from detrt.fast import run_stages
        from .proto import stage_output_root
        self.bind()
        if self.gpu:
            from detgpu.sha import stage_output_roots
            from detgpu.stages import stage_outputs
            y = stage_outputs(self.m, None, torch.from_numpy(np.ascontiguousarray(rows))[None], g * self.lpg, (g + 1) * self.lpg)
            return stage_output_roots(y)[0]
        y = run_stages(self.m, None, [torch.from_numpy(rows)], g * self.lpg, (g + 1) * self.lpg, keep_rows=False)[0]
        return stage_output_root(y)

    def trace(self, batches: list[list[int]], stage: int):
        """The full trace of one stage (for dispute games): recompute the batch up to it."""
        b, g = divmod(stage, self.G)
        key = ("trace", tuple(batches[b]), g)
        if key in getattr(self, "traces", {}):
            return self.traces[key]
        self.bind()
        toks = torch.tensor(batches[b], dtype=torch.long)
        x = torch.from_numpy(self.outputs([batches[b]], g - 1)[0][g - 1]) if g > 0 else None
        st = run_stage(self.m, toks if g == 0 else None, x, g * self.lpg, (g + 1) * self.lpg)
        traces = getattr(self, "traces", {})
        traces[key] = st
        while len(traces) > 3:  # traces are large: keep the last few (one per running game)
            traces.pop(next(iter(traces)))
        self.traces = traces
        return st

    def stage_root(self, batches: list[list[int]], stage: int) -> bytes:
        b, g = divmod(stage, self.G)
        return self.batch(batches[b], g)[0][g]
