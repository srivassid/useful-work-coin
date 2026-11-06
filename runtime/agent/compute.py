"""Stage roots and outputs with the deterministic runtime (detrt), as providers, verifiers and
watchers compute them. Stage s of a job is batch s // G through layer group s % G."""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from detrt.stage import run_stage

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

    def batch(self, tokens: list[int], upto: int | None = None) -> tuple[list[bytes], np.ndarray | None]:
        """Roots of groups 0..upto (default all) for one sequence, and the pooled output if all ran."""
        upto = self.G - 1 if upto is None else upto
        key = tuple(tokens)
        roots, out = self.cache.get(key, ([], None))
        if len(roots) > upto:
            return roots[:upto + 1], out
        x = None
        if roots:  # resume from the cached last output is not kept; recompute from the start
            roots = []
        toks = torch.tensor(tokens, dtype=torch.long)
        if self.miner is not None:
            self.miner.refresh()
        for g in range(upto + 1):
            st = run_stage(self.m, toks if g == 0 else None, x, g * self.lpg, (g + 1) * self.lpg)
            roots.append(st.root)
            x = torch.from_numpy(st.output)
            del st
        emb = pooled(x.numpy()) if upto == self.G - 1 else None
        if emb is not None:
            self.cache[key] = (roots, emb)
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
        return roots, emb

    def trace(self, batches: list[list[int]], stage: int):
        """The full trace of one stage (for dispute games): recompute the batch up to it."""
        b, g = divmod(stage, self.G)
        key = ("trace", tuple(batches[b]), g)
        if key in getattr(self, "traces", {}):
            return self.traces[key]
        toks = torch.tensor(batches[b], dtype=torch.long)
        x = None
        for gg in range(g + 1):
            st = run_stage(self.m, toks if gg == 0 else None, x, gg * self.lpg, (gg + 1) * self.lpg)
            x = torch.from_numpy(st.output)
        traces = getattr(self, "traces", {})
        traces[key] = st
        while len(traces) > 3:  # traces are large: keep the last few (one per running game)
            traces.pop(next(iter(traces)))
        self.traces = traces
        return st

    def batch_with_inputs(self, tokens: list[int]):
        """Like batch(), and also each later stage's input with proofs: for group g >= 1, the
        rows group g-1 produced and their Merkle proofs in group g-1's trace (for verifiers who
        recompute one stage instead of the whole prefix)."""
        from .proto import multi_proofs
        from detrt.stage import R2
        toks = torch.tensor(tokens, dtype=torch.long)
        if self.miner is not None:
            self.miner.refresh()
        roots, inputs, x = [], {}, None
        for g in range(self.G):
            st = run_stage(self.m, toks if g == 0 else None, x, g * self.lpg, (g + 1) * self.lpg)
            roots.append(st.root)
            if g + 1 < self.G:
                last = (g + 1) * self.lpg - 1
                idx = sorted((lf.t, i) for i, lf in enumerate(st.leaves) if lf.op == R2 and lf.layer == last)
                idx = [i for _, i in idx]
                pr = multi_proofs(st.leaf_hashes, idx)
                inputs[g + 1] = {"rows": st.output.copy(), "idx": idx, "proofs": [[h.hex() for h in pr[i]] for i in idx],
                                 "prev_leaves": len(st.leaves)}
            x = torch.from_numpy(st.output)
            del st
        return roots, pooled(x.numpy()), inputs

    def job_with_inputs(self, batches: list[list[int]], chunk: int = 8):
        """batch_with_inputs for a whole job, using the batched runtime (detrt.fast: same rows and
        roots, about 3x faster per sequence-stage). Returns per-batch (roots, embedding, inputs)."""
        from detrt.fast import run_stages
        from detrt.stage import R2
        from .proto import multi_proofs
        if self.miner is not None:
            self.miner.refresh()
        out = []
        for c0 in range(0, len(batches), chunk):
            seqs = [torch.tensor(t, dtype=torch.long) for t in batches[c0:c0 + chunk]]
            res = [([], {}) for _ in seqs]
            xs = None
            for g in range(self.G):
                trs = run_stages(self.m, seqs if g == 0 else None, xs, g * self.lpg, (g + 1) * self.lpg)
                for (roots, inputs), st in zip(res, trs):
                    roots.append(st.root)
                    if g + 1 < self.G:
                        last = (g + 1) * self.lpg - 1
                        idx = [i for _, i in sorted((lf.t, i) for i, lf in enumerate(st.leaves)
                                                    if lf.op == R2 and lf.layer == last)]
                        pr = multi_proofs(st.leaf_hashes, idx)
                        inputs[g + 1] = {"rows": st.output.copy(), "idx": idx,
                                         "proofs": [[h.hex() for h in pr[i]] for i in idx], "prev_leaves": len(st.leaves)}
                xs = [torch.from_numpy(st.output) for st in trs]
                del trs
            out += [(roots, pooled(x.numpy()), inputs) for (roots, inputs), x in zip(res, xs)]
        return out

    def stage_root_from_input(self, rows: np.ndarray, g: int) -> bytes:
        """One stage (layer group g >= 1) from its served input rows."""
        return run_stage(self.m, None, torch.from_numpy(rows), g * self.lpg, (g + 1) * self.lpg).root

    def stage_root(self, batches: list[list[int]], stage: int) -> bytes:
        b, g = divmod(stage, self.G)
        return self.batch(batches[b], g)[0][g]
