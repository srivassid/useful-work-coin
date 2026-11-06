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

    def stage_root(self, batches: list[list[int]], stage: int) -> bytes:
        b, g = divmod(stage, self.G)
        return self.batch(batches[b], g)[0][g]
