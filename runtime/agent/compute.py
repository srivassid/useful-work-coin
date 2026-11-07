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
    def __init__(self, model_json: str, device: str, mine_socket: str | None = None, engine: str = "detgpu"):
        info = json.load(open(model_json))
        self.info = info
        self.device = torch.device(device)
        # detgpu: bit-identical to detrt (tests/test_detgpu.py), native fp32 and fused kernels, stage
        # roots hashed on the GPU. detrt.fast stays the reference and the CPU path.
        self.gpu = engine == "detgpu" and self.device.type == "cuda"
        self.last_gpu_s = None
        self.G = info["layer_groups"]
        t = time.time()
        self.m = load(info["model_dir"], device, info["rotation_block"], info["cfg"]["MaxPos"], mine=bool(mine_socket))
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
