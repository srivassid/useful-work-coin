"""Providers that mine while serving: every int8 GEMM of a job runs Pearl's noisy GEMM under the
current block's key (design doc, "What miners run"). Hashed MACs are counted so lambda becomes a
live metric (hashed MACs per GPU-second of job / the same GPU's noise rate), and winning tiles
are proven and submitted through the provider's gateway like any miner's.

The reference runtime's source (its hash is registered on chain) is not modified: this wraps
DetLinear.int_gemm from outside.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "miner", "ampere"))

import pearl_mining as pm  # noqa: E402
from detrt.qwen3 import DetLinear  # noqa: E402
from miner_base.gateway_client import MiningClient  # noqa: E402
from pearl_gateway.config import MinerRpcConfig  # noqa: E402
from uwc_ampere import ops, pearl_glue as pg  # noqa: E402

RANK = 128


class JobMiner:
    def __init__(self, model, socket_path: str, log=print):
        self.m, self.log = model, log
        self.client = MiningClient(MinerRpcConfig(transport="uds", socket_path=socket_path))
        self.socket_path = socket_path
        self.header = None
        self.job = None
        self.keys: dict[int, bytes] = {}
        self.hashed = 0
        self.wins = 0
        self.submitted = 0
        self.linears = [L[n] for L in model.layers for n in ("qkv", "o", "gu", "down")]
        miner = self

        def int_gemm(lin: DetLinear, q: torch.Tensor):
            m = q.shape[0]
            if not (lin.mine and lin.mw is not None):
                return DetLinear._orig_int_gemm(lin, q)
            mp = -(-m // ops.BM) * ops.BM
            qp = torch.nn.functional.pad(q, (0, 0, 0, mp - m)) if mp != m else q
            out = lin.mw.forward(qp.contiguous(), torch.ones(mp, dtype=torch.float32, device=q.device),
                                 out_mode=ops.OUT_I32)[:m]
            miner.hashed += mp * lin.n * lin.k
            wins = lin.mw.winners(mp)
            if wins:
                miner.submit(lin, qp, wins)
            return out

        if not hasattr(DetLinear, "_orig_int_gemm"):
            DetLinear._orig_int_gemm = DetLinear.int_gemm
        DetLinear.int_gemm = int_gemm

    def refresh(self) -> bool:
        """Re-key every mined layer when the block template changes. False if no template."""
        try:
            job = self.client.get_mining_info()
        except Exception as e:
            self.log(f"mining: no template ({e})")
            try:
                self.client.close()
            except Exception:
                pass
            self.client = MiningClient(MinerRpcConfig(transport="uds", socket_path=self.socket_path))
            for lin in self.linears:
                lin.mine = False
            return False
        if self.job is not None and job.incomplete_header_bytes == self.job.incomplete_header_bytes \
                and job.target == self.job.target:
            return True
        self.job = job
        self.header = pm.IncompleteBlockHeader.from_bytes(job.incomplete_header_bytes)
        cfgs = {}

        def key_for(k):
            cfgs.setdefault(k, pg.mining_config(k, RANK))
            return pg.job_key(self.header, cfgs[k])

        def bound_for(k):
            cfgs.setdefault(k, pg.mining_config(k, RANK))
            return pg.bound_words(pm.penalized_target_bound(job.target, cfgs[k]))
        for lin in self.linears:
            lin.mine = lin.device.type == "cuda"
            lin.setup_block(key_for, bound_for)
        self.keys = {k: key_for(k) for k in cfgs}
        return True

    def submit(self, lin: DetLinear, qp: torch.Tensor, wins):
        self.wins += len(wins)
        row_off, col_off, _ = wins[0]
        A = qp.cpu().numpy()
        Bt = lin.Wq.cpu().numpy()
        key = self.keys.get(lin.k) or pg.job_key(self.header, pg.mining_config(lin.k, RANK))
        proof = pg.build_plain_proof(A, Bt, key, RANK, [row_off + o for o in pg.AMPERE_ROWS],
                                     [col_off + o for o in pg.AMPERE_COLS])
        ok, msg = pm.verify_plain_proof_for_cert_version(int(self.job.cert_version), self.header, proof)
        if not ok:
            self.log(f"mining: a job tile won but failed local verification: {msg}")
            return
        try:
            self.client.submit_plain_proof(proof, self.job)
            self.submitted += 1
            self.log(f"mining: submitted a winning tile found while serving a job (layer k={lin.k})")
        except Exception as e:
            self.log(f"mining: submit failed: {e}")
