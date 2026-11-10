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


class GenJob:
    """A generation job's sampling: temperature in thousandths, seed = the job id."""

    def __init__(self, temp_milli: int, job_id: str):
        self.temp_milli, self.job_id = int(temp_milli), job_id

    def sampling(self):
        from detgen.head import Sampling
        return Sampling(self.temp_milli, bytes.fromhex(self.job_id))


def gen_of(j: dict) -> GenJob | None:
    return GenJob(j.get("tempmilli", 0), j["id"]) if j.get("gen") else None


def prompt_of(row: list[int]) -> list[int]:
    from detgen.head import SENTINEL
    return row[:row.index(SENTINEL)] if SENTINEL in row else list(row)


class Runtime:
    def __init__(self, model_json: str, device: str, mine_socket: str | None = None, engine: str = "detgpu",
                 mine: bool | None = None):
        info = json.load(open(model_json))
        self.info = info
        self.device = torch.device(str(device).split(",")[0])  # "cuda:0,cuda:1": layers split over GPUs
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

    def jobs_roots(self, jobs: list[list[list[int]]], tokens: int = 8192):
        """Several jobs through the GPU together: per job (stage output roots in stage order,
        pooled outputs). Sequences of all jobs share forward passes of up to `tokens` tokens
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
        for L, items in by_len.items():
            chunk = max(1, tokens // L)
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

    # ------------------------------------------------------------ generation jobs (Rules 3)
    def split(self, stage: int, gen: GenJob | None = None) -> tuple[int, int]:
        """(batch, group) of a stage; group G is a generation batch's head stage."""
        return divmod(stage, self.G + (1 if gen else 0))

    def head_out(self, r2: np.ndarray, gen: GenJob) -> np.ndarray:
        """Head-stage output rows (T-1, 2) int32 from the last layer's R2 rows (T, D)."""
        if self.gpu:
            from detgen.fast import head_outputs
            self.bind()
            return head_outputs(self.m, torch.from_numpy(np.ascontiguousarray(r2)).to(self.m.out_device)[None], gen.sampling())[0]
        from detgen.head import run_head_stage
        return run_head_stage(self.m, torch.from_numpy(r2), gen.sampling(), rows=False).output

    def generate(self, prompts: list[list[int]], T: int, gen: GenJob, on_step=None) -> list[list[int]]:
        """The full sequences of a generation job (deterministic: anyone gets the same)."""
        self.bind()
        if self.miner is not None:
            self.miner.refresh()
        if self.gpu and self.m.hd == 128:
            from detgen import fast
            return fast.generate(self.m, prompts, T, gen.sampling(), on_step=on_step)
        from detgen.generate import generate
        return [generate(self.m, p, T, gen.sampling()) for p in prompts]

    def gen_job(self, rows: list[list[int]], gen: GenJob, on_step=None):
        """A provider's generation job: (sequences, stage roots in stage order, head outputs per
        batch, the job's output root). last_gpu_s covers generation, layer stages and heads."""
        from detgen.head import gen_output_root, head_output_root
        from .proto import stage_output_root
        torch_sync = torch.cuda.synchronize if self.gpu else (lambda: None)
        torch_sync()
        t = time.time()
        T = len(rows[0])
        seqs = self.generate([prompt_of(r) for r in rows], T, gen, on_step=on_step)
        roots, heads = [], []
        if self.gpu:
            from detgpu.sha import digest_bytes, stage_roots_gpu
            from detgpu.stages import job_outputs
            from detgen.fast import head_outputs
            for c0 in range(0, len(seqs), 16):
                part = seqs[c0:c0 + 16]
                outs = job_outputs(self.m, torch.tensor(part, dtype=torch.long), self.G)
                rr = digest_bytes(torch.stack([stage_roots_gpu(o) for o in outs]))
                hs = head_outputs(self.m, outs[-1], gen.sampling())
                for b in range(len(part)):
                    heads.append(hs[b])
                    roots += [bytes(rr[g, b]) for g in range(self.G)] + [head_output_root(hs[b])]
                del outs
        else:
            for seq, outs in zip(seqs, self.outputs(seqs)):
                h = self.head_out(outs[-1], gen)
                heads.append(h)
                roots += [stage_output_root(o) for o in outs] + [head_output_root(h)]
        torch_sync()
        self.last_gpu_s = time.time() - t
        out_root = gen_output_root([roots[b * (self.G + 1) + self.G] for b in range(len(seqs))])
        return seqs, roots, heads, out_root

    def stage_input(self, batches: list[list[int]], stage: int, gen: GenJob | None = None) -> dict:
        """What a verifier asks a provider for: stage s's input rows (the previous stage's output,
        recomputed for that one sequence) and each row's Merkle proof in its output tree."""
        from .proto import multi_proofs, stage_out_leaves
        b, g = self.split(stage, gen)
        assert g >= 1, "a stage of layer group 0 reads tokens"
        rows = np.ascontiguousarray(self.outputs([batches[b]], g - 1)[0][g - 1])
        leaves = stage_out_leaves(rows)
        pr = multi_proofs(leaves, range(len(leaves)))
        return {"rows": rows, "proofs": [[h.hex() for h in pr[t]] for t in range(len(pr))]}

    def stage_output(self, batches: list[list[int]], stage: int, gen: GenJob | None = None) -> np.ndarray:
        b, g = self.split(stage, gen)
        if g == self.G:  # a head stage's outputs: the samples
            return self.head_out(self.outputs([batches[b]])[0][-1], gen)
        return self.outputs([batches[b]], g)[0][g]

    def stage_root_from_input(self, rows: np.ndarray, g: int, gen: GenJob | None = None) -> bytes:
        """One stage (layer group g >= 1, or the head stage g == G) from its served input rows: its output root."""
        from detrt.fast import run_stages
        from .proto import stage_output_root
        if gen is not None and g == self.G:
            from detgen.head import head_output_root
            return head_output_root(self.head_out(rows, gen))
        self.bind()
        if self.gpu:
            from detgpu.sha import stage_output_roots
            from detgpu.stages import stage_outputs
            y = stage_outputs(self.m, None, torch.from_numpy(np.ascontiguousarray(rows))[None], g * self.lpg, (g + 1) * self.lpg)
            return stage_output_roots(y)[0]
        y = run_stages(self.m, None, [torch.from_numpy(rows)], g * self.lpg, (g + 1) * self.lpg, keep_rows=False)[0]
        return stage_output_root(y)

    def trace(self, batches: list[list[int]], stage: int, gen: GenJob | None = None):
        """The full trace of one stage (for dispute games): recompute the batch up to it."""
        b, g = self.split(stage, gen)
        key = ("trace", tuple(batches[b]), g, gen.temp_milli if gen else None, gen.job_id if gen else None)
        if key in getattr(self, "traces", {}):
            return self.traces[key]
        self.bind()
        toks = torch.tensor(batches[b], dtype=torch.long)
        x = torch.from_numpy(self.outputs([batches[b]], g - 1)[0][g - 1]) if g > 0 else None
        if gen is not None and g == self.G:
            from detgen.head import run_head_stage
            st = run_head_stage(self.m, x, gen.sampling())
        else:
            st = run_stage(self.m, toks if g == 0 else None, x, g * self.lpg, (g + 1) * self.lpg)
        traces = getattr(self, "traces", {})
        traces[key] = st
        while len(traces) > 3:  # traces are large: keep the last few (one per running game)
            traces.pop(next(iter(traces)))
        self.traces = traces
        return st

    def stage_root(self, batches: list[list[int]], stage: int, gen: GenJob | None = None) -> bytes:
        b, g = self.split(stage, gen)
        if g == self.G:
            from detgen.head import head_output_root
            return head_output_root(self.stage_output(batches, stage, gen))
        return self.batch(batches[b], g)[0][g]
