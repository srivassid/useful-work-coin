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
    def __init__(self, model_json: str, device: str, mine_socket: str | None = None):
        info = json.load(open(model_json))
        self.info = info
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

    def outputs(self, seqs: list[list[int]], upto: int | None = None, chunk: int = 8):
        """Stage outputs (the rows each stage is committed by) for groups 0..upto of each
        sequence: forward pass only, batched over sequences. Returns per sequence a list of
        (T, D) float32 arrays, one per group."""
        from detrt.fast import run_stages
        upto = self.G - 1 if upto is None else upto
        if self.miner is not None:
            self.miner.refresh()
        out = []
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

    def stage_output(self, batches: list[list[int]], stage: int) -> np.ndarray:
        b, g = divmod(stage, self.G)
        return self.outputs([batches[b]], g)[0][g]

    def stage_root_from_input(self, rows: np.ndarray, g: int) -> bytes:
        """One stage (layer group g >= 1) from its served input rows: its output root."""
        from detrt.fast import run_stages
        from .proto import stage_output_root
        y = run_stages(self.m, None, [torch.from_numpy(rows)], g * self.lpg, (g + 1) * self.lpg, keep_rows=False)[0]
        return stage_output_root(y)

    def trace(self, batches: list[list[int]], stage: int):
        """The full trace of one stage (for dispute games): recompute the batch up to it."""
        b, g = divmod(stage, self.G)
        key = ("trace", tuple(batches[b]), g)
        if key in getattr(self, "traces", {}):
            return self.traces[key]
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
