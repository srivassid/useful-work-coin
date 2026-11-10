"""Devnet roles: provider, verifier, watcher, client. Each polls the chain every few seconds,
acts through uwcctl, and keeps its secrets (salts) in a JSON state file so it survives restarts.

  python -m agent.roles provider --key K --model M --units U --device cuda:1
  python -m agent.roles verifier --key K --model M --units U --device cuda:0 --stake 100000000
  python -m agent.roles watcher  --key K --model M --units U --device cuda:0 --rate 0.5
  python -m agent.roles client   --key K --model M --post 3 --fee 3000000000
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import faulthandler
import hashlib
import io
import json
import os
import random
import secrets
import signal
import sys
import threading
import time
import traceback

import numpy as np

from .chain import Chain, ChainError
from .data import Data
from .inputs import check_units, input_root, unit_proof, wikitext_sequences
from .game import GONESTEP, GREVEAL, move, write_rows
from .proto import StageCommit, check_stage_input, job_id, output_root, positions, result_hash

DEVNET = os.path.expanduser("~/Documents/useful-work-coin/devnet")
DATA_URL = "http://192.168.29.71:18680"
GAME_REVEAL = 1  # jobstate.GReveal


def torch_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Agent:
    role = "agent"
    # True: every GPU use runs on the worker thread (roles whose worker runs passes while the
    # main loop answers deadlines). The runtime and its job miner are not thread-safe: on
    # 2026-11-09 a provider's main loop computing a game trace while the worker ran a job hung
    # for three hours in the miner's shared gateway socket and missed 28 reveal deadlines.
    gpu_async = False
    WATCH_EVERY = 30.0  # seconds between watchdog checks

    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.chain = Chain(a.devnet)
        self.key = os.path.expanduser(a.key)
        info = self.chain.key_info(self.key)
        self.account = info["account"]
        self.data = Data(a.data, signer=lambda d: self.chain.run("sign", self.key, d.hex()).strip())
        # --model a.json,b.json (and --weights in the same order): verifiers and watchers serve
        # committee seats and checks for every model listed; providers and clients use the first.
        # Committees are drawn from all stakers whatever the model, so a seat on a job of a model
        # an agent does not hold is skipped (a no-show penalty) rather than answered wrongly.
        self.init_models(a.model, a.weights)
        # public work lists by work root (comma-separated --units files)
        self.lists = {}
        for path in (a.units or "").split(","):
            if path and os.path.exists(path):
                doc = json.load(open(path))
                self.lists[doc["work_root"]] = doc
        cache = os.path.join(a.devnet, "agent", "units")  # work lists fetched from the data service
        if os.path.isdir(cache):
            for f in os.listdir(cache):
                with contextlib.suppress(Exception):
                    doc = json.load(open(os.path.join(cache, f)))
                    self.lists[doc["work_root"]] = doc
        self._list_tried: dict[str, float] = {}
        self.units = next(iter(self.lists.values()), None)
        self._cat_roots: dict[str, str] = {}
        self.path = os.path.join(a.devnet, "agent", f"{self.role}-{self.account[:8]}.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.st = json.load(open(self.path)) if os.path.exists(self.path) else {"jobs": {}}
        # long computations (whole jobs) run here, so the main loop can always answer reveals
        # and game rounds before their deadlines
        self.worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.worker_io = concurrent.futures.ThreadPoolExecutor(max_workers=1)  # uploads that must not hold up the GPU
        self.busy: concurrent.futures.Future | None = None
        self._gpu_tasks: dict[tuple, tuple[concurrent.futures.Future, float]] = {}
        self._task: tuple[str, float] | None = None  # the worker's running task, for the watchdog
        self._beat = time.time()  # the main loop's last finished step

    def submit(self, what: str, fn, *args) -> concurrent.futures.Future:
        """Queue fn on the GPU worker; the watchdog reports a task that runs too long."""
        def task():
            self._task = (what, time.time())
            try:
                return fn(*args)
            finally:
                self._task = None
        return self.worker.submit(task)

    def on_gpu(self, key: tuple, fn):
        """fn's result, computed on the GPU worker for roles with gpu_async: None until it is
        done (the caller retries on a later step, so the main loop never waits on the GPU).
        Other roles compute it here."""
        if not self.gpu_async:
            return fn()
        now = time.time()
        for k, (f, t) in list(self._gpu_tasks.items()):  # results nobody came back for (a game that ended)
            if f.done() and now - t > 3600:
                del self._gpu_tasks[k]
        f, _ = self._gpu_tasks.get(key, (None, 0))
        if f is None:
            self._gpu_tasks[key] = (self.submit(" ".join(str(k)[:16] for k in key), fn), now)
            return None
        if not f.done():
            return None
        del self._gpu_tasks[key]
        try:
            return f.result()
        except Exception:
            log(f"GPU task {key[0]} failed (retried next step):")
            traceback.print_exc()
            return None

    def watchdog(self, stop: threading.Event | None = None):
        """Log every thread's stack when the main loop stops finishing steps (deadlines would
        pass silently) or a GPU task runs far longer than any pass should."""
        logged = {"loop": 0.0, "task": 0.0}
        while not (stop and stop.wait(self.WATCH_EVERY)):
            if stop is None:
                time.sleep(self.WATCH_EVERY)
            now = time.time()
            stalled = now - self._beat
            if stalled >= self.a.stall and now - logged["loop"] >= self.a.stall:
                logged["loop"] = now
                log(f"WATCHDOG: the main loop has not finished a step for {stalled:.0f}s")
                self.dump_threads()
            t = self._task
            if t and now - t[1] >= self.a.task_stall and now - logged["task"] >= self.a.task_stall:
                logged["task"] = now
                log(f"WATCHDOG: the GPU worker has run '{t[0]}' for {now - t[1]:.0f}s")
                self.dump_threads()

    @staticmethod
    def dump_threads():
        names = {t.ident: t.name for t in threading.enumerate()}
        for ident, frame in sys._current_frames().items():
            if names.get(ident) == "watchdog":
                continue
            print(f"--- thread {names.get(ident, ident)}:\n" + "".join(traceback.format_stack(frame)), end="", flush=True)

    @contextlib.contextmanager
    def gpu_pass(self):
        """Pause a noise miner sharing this GPU (--pause-file) for the length of a GPU pass."""
        p = getattr(self.a, "pause_file", None)
        if p:
            open(p, "w").close()
        try:
            yield
        finally:
            if p:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(p)

    rt_lock = threading.Lock()

    def init_models(self, model_arg: str, weights_arg: str | None):
        paths = [p for p in str(model_arg).split(",") if p]
        weights = [w for w in str(weights_arg or "").split(",")]
        self.models: dict[str, dict] = {}
        self._runtimes: dict[tuple, object] = {}
        self._skipped_models: set[str] = set()
        for i, p in enumerate(paths):
            info = json.load(open(p))
            mid = json.load(open(p.replace(".json", ".reg.json")))["model"]
            # registrations of one checkpoint with the same settings share a runtime (the same
            # numerics: e.g. one model registered again under a newer runtime hash)
            rkey = (info["model_dir"], info["rotation_block"], info["cfg"]["MaxPos"])
            self.models[mid] = {"path": p, "info": info, "weights": weights[i] if i < len(weights) else "", "rkey": rkey}
        self.use(next(iter(self.models)))

    def use(self, model_id: str) -> bool:
        """Make model_id the current model (self.model, self.model_id, self.G, self.rt); False if
        this agent does not hold it (logged once per model)."""
        m = self.models.get(model_id)
        if m is None:
            if model_id not in self._skipped_models:
                self._skipped_models.add(model_id)
                log(f"model {model_id[:8]} is not one of this agent's models: its jobs are skipped")
            return False
        self.model, self.model_id, self.G = m["info"], model_id, m["info"]["layer_groups"]
        self.model_path, self.weights_path, self._rkey = m["path"], m["weights"], m["rkey"]
        return True

    def runtime_for(self, rkey: tuple):
        with self.rt_lock:
            if rkey not in self._runtimes:
                from .compute import Runtime
                path = next(m["path"] for m in self.models.values() if m["rkey"] == rkey)
                sock = os.path.join(self.a.devnet, "pearlgw.sock") if getattr(self.a, "mine", False) else None
                self._runtimes[rkey] = Runtime(path, self.a.device, mine_socket=sock, engine=self.a.engine)
            return self._runtimes[rkey]

    @property
    def rt(self):
        return self.runtime_for(self._rkey)

    def load_models(self):
        """Load every model's runtime (in the background at start: a seat must not wait for it)."""
        for rkey in dict.fromkeys(m["rkey"] for m in self.models.values()):
            self.runtime_for(rkey)

    lock = threading.Lock()

    def save(self):
        with self.lock:
            tmp = self.path + ".tmp"
            raw = json.dumps(self.st)
            with open(tmp, "w") as f:
                f.write(raw)
            os.replace(tmp, self.path)

    def send(self, ops: list[dict], what: str) -> dict | None:
        try:
            r = self.chain.ops(self.key, ops)
            log(f"{what}: tx {r['txid'][:12]}")
            return r
        except ChainError as e:
            log(f"{what}: {e}")
            return None

    def challenge_bond(self, j: dict) -> int:
        """The bond a CHALLENGE on j needs (jobstate.ChallengeBond, from the next block's rules)."""
        p = self.chain.state().get("params") or {}
        need = p.get("challengebondmin", 0)
        bp = p.get("challengebondbp", 0)
        if p.get("rules", 0) >= 2 and bp:
            b = j["bond"]
            need = max(need, b // 10_000 * bp + (b % 10_000) * bp // 10_000)
        return max(self.a.bond, need)

    @staticmethod
    def gen(j: dict):
        """A generation job's sampling (agent.compute.GenJob), or None."""
        if not j.get("gen"):
            return None
        from .compute import GenJob
        return GenJob(j.get("tempmilli", 0), j["id"])

    def per_batch(self, j: dict) -> int:
        """Stages per batch: the layer groups, plus a generation batch's head stage."""
        return self.G + (1 if j.get("gen") else 0)

    def batches(self, j: dict) -> list[list[int]] | None:
        """The token sequences the job's stages run over: the input batches, or for a generation
        job the full sequences (the provider's own, or regenerated here: generation is
        deterministic, so a verifier needs nothing from the provider)."""
        rows = self.input_batches(j)
        if rows is None or not j.get("gen"):
            return rows
        js = self.st["jobs"].get(j["id"], {})
        if js.get("seqs"):
            return js["seqs"]
        cache = self.__dict__.setdefault("_gen_seqs", {})
        if j["id"] not in cache:
            from .compute import prompt_of
            gen = self.gen(j)

            def regenerate():
                t = time.time()
                seqs = self.rt.generate([prompt_of(r) for r in rows], j["seqlen"], gen)
                log(f"job {j['id'][:8]}: regenerated {len(rows)} sequences in {time.time() - t:.1f}s")
                return seqs
            seqs = self.on_gpu(("regenerate", j["id"]), regenerate)
            if seqs is None:
                return None
            cache[j["id"]] = seqs
            while len(cache) > 16:
                cache.pop(next(iter(cache)))
        return cache[j["id"]]

    def input_batches(self, j: dict) -> list[list[int]] | None:
        """The job's input rows: a pool unit from the public work list, a client job from the
        data service (checked against input_root)."""
        if j.get("pool"):
            doc = self.work_list(self.work_root(j["pool"]))
            if not doc:
                return None
            u = doc["units"][j["unit"]]
            return u["batches"] if "batches" in u else [u["tokens"]]
        raw = self.data.get(f"/inputs/{j['inputroot']}.json")
        if raw is None:
            return None
        b = json.loads(raw)["batches"]
        if input_root(b).hex() != j["inputroot"]:
            log(f"job {j['id'][:8]}: inputs do not match input_root")
            return None
        return b

    def play(self, j: dict, as_provider: bool):
        """Make this party's move in the job's current game round, if one is due."""
        g = j.get("game")
        if not g or g["phase"] == GREVEAL:
            return
        # pmoved/cmoved are left over from the last block round when the game enters the
        # one-step phase; only the provider acts there
        if g["phase"] != GONESTEP and (g["pmoved"] if as_provider else g["cmoved"]):
            return
        js = self.st["jobs"].setdefault(j["id"], {})
        tag = f"{g['stage']}:{g['phase']}:{g['lo']}:{g['size']}:{g['leaf']}:{g['block']}"
        h = self.chain.height()
        if js.get("last_move") == tag and h - js.get("last_move_h", h) < 3:
            return  # sent recently; resend if it has not reached a block after 3 blocks (e.g. a node restart dropped it)
        b = self.batches(j)
        if b is None:
            return
        if g["phase"] == GONESTEP:
            if as_provider:
                self.one_step(j, b, js, tag)
            return
        tr = self.on_gpu(("trace", j["id"], g["stage"], as_provider), lambda: self.game_trace(j, b, g["stage"], as_provider))
        if tr is None:
            return
        op = move(j["id"], g, tr)
        if op and self.send([op], f"game {j['id'][:8]} stage {g['stage']} phase {g['phase']} lo {g['lo']} size {g['size']}"):
            js["last_move"], js["last_move_h"] = tag, h
            self.save()

    def game_trace(self, j: dict, b, stage: int, as_provider: bool):
        return self.rt.trace(b, stage, self.gen(j))

    def one_step(self, j: dict, b: list[list[int]], js: dict, tag: str):
        """Last round as provider: every row the referee reads, with proofs (uwcctl onestep)."""
        g = j["game"]
        d = os.path.join(self.a.devnet, "agent", "onestep", j["id"][:16])
        os.makedirs(d, exist_ok=True)
        gen = self.gen(j)
        per = self.per_batch(j)
        stage = g["stage"]
        hs = (stage // per) * per + self.G

        def rows():  # on the GPU worker: (previous stage's outputs, head outputs, trace rows)
            prev = list(self.rt.stage_output(b, stage - 1, gen)) if stage % per else None
            head = list(self.rt.stage_output(b, hs, gen)) if gen is not None else None
            return prev, head, self.rt.trace(b, stage, gen).rows
        got = self.on_gpu(("onestep", j["id"], tag), rows)
        if got is None:
            return
        prev, head, trace_rows = got
        req = {"job": j["id"], "model": self.model_path, "weights": self.weights_path, "rows": os.path.join(d, "rows.bin"),
               "prev_outputs": "", "batches": b, "salts": js["salts"], "roots": js["roots"]}
        if prev is not None:
            write_rows(os.path.join(d, "prev.bin"), prev)
            req["prev_outputs"] = os.path.join(d, "prev.bin")
        if gen is not None:
            req.update(gen=True, temp_milli=gen.temp_milli, input_rows=self.input_batches(j),
                       head_outputs=os.path.join(d, "head.bin"))
            write_rows(req["head_outputs"], head)
        write_rows(req["rows"], trace_rows)
        json.dump(req, open(os.path.join(d, "req.json"), "w"))
        try:
            r = self.chain.run("onestep", self.key, os.path.join(d, "req.json"))
            log(f"one-step {j['id'][:8]} stage {g['stage']} leaf {g['leaf']} elem {g['elem']}: tx {r['txid'][:12]}")
            js["last_move"], js["last_move_h"] = tag, self.chain.height()
            self.save()
        except ChainError as e:
            log(f"one-step {j['id'][:8]}: {e}")

    def challenger_games(self):
        """Play every game in which this account is the disputant (flagging verifier or watcher)."""
        for jid in self.chain.jobs("disputed"):
            j = self.chain.job(jid)
            if j and j.get("disputes") and j["disputes"][0]["challenger"] == self.account and self.use(j["model"]):
                self.play(j, as_provider=False)

    def work_list(self, root: str) -> dict | None:
        """A pool category's work list: from --units, the local cache, or the data service (checked
        against the on-chain work root before use; a failed fetch is retried after 10 minutes)."""
        if root in self.lists:
            return self.lists[root]
        if time.time() - self._list_tried.get(root, 0) < 600:
            return None
        self._list_tried[root] = time.time()
        try:
            raw = self.data.get(f"/units/{root}.json")
        except Exception as e:
            log(f"work list {root[:8]}: {e}")
            return None
        if raw is None:
            return None
        doc = json.loads(raw)
        if not check_units(doc, root):
            log(f"work list {root[:8]} from the data service does not match its work root")
            return None
        cache = os.path.join(self.a.devnet, "agent", "units")
        os.makedirs(cache, exist_ok=True)
        with open(os.path.join(cache, f"{root}.json"), "w") as f:
            f.write(raw.decode())
        self.lists[root] = doc
        self.units = self.units or doc
        log(f"work list {root[:8]}: {len(doc['units'])} units from the data service")
        return doc

    def work_root(self, cid: str) -> str:
        if cid not in self._cat_roots:
            self._cat_roots[cid] = self.chain.category(cid)["workroot"]
        return self._cat_roots[cid]

    def run(self):
        log(f"{self.role} {self.account[:12]} starting")
        faulthandler.register(signal.SIGUSR1, all_threads=True)  # kill -USR1 <pid>: every thread's stack
        threading.Thread(target=self.watchdog, name="watchdog", daemon=True).start()
        if self.role != "client":  # load the models in the background; reveals need no model
            self.busy = self.submit("load models", self.load_models)
        while True:
            try:
                self.step()
            except Exception:
                traceback.print_exc()
            self._beat = time.time()
            time.sleep(self.a.poll)

    def step(self):
        raise NotImplementedError


# ---------------------------------------------------------------- provider

class Provider(Agent):
    """Claims pool units and accepts client jobs, runs every stage, commits salted stage roots
    and the output root, delivers outputs off-chain, reveals what is requested."""
    role = "provider"
    gpu_async = True

    def step(self):
        s = self.chain.state()
        h = s["height"]
        # read the mempool before the jobs: a claim that leaves it in between is then already ours
        pending = self.chain.mempool() if any(js.get("tx") and 0 < h - js.get("claimed", js.get("accepted", h)) < 20
                                              for js in self.st["jobs"].values()) else set()
        mine = {jid: e for jid, e in self.chain.jobs().items() if e["provider"] == self.account}
        active = [jid for jid, e in mine.items() if e["status"] not in ("settled", "expired", "failed", "aborted")]
        # the GPU is busy only until a job is committed: limit jobs not yet committed, counting
        # claims and accepts sent but not yet in a block
        computing = [jid for jid in active if mine[jid]["status"] == "accepted"
                     and not self.st["jobs"].get(jid, {}).get("foreign_model")]  # left to expire: no GPU work
        self.pending_bonds = 0  # bonds of claims and accepts sent but not yet mined
        for jid, js in self.st["jobs"].items():
            sent = js.get("claimed", js.get("accepted"))
            if jid not in mine and sent is not None and h - sent < 20:
                if js.get("tx") and h > sent:
                    if js["tx"] not in pending:
                        continue  # left the mempool without making the job ours: another provider won it
                computing.append(jid)
                active.append(jid)
                self.pending_bonds += js.get("bond", 0)
        # new work: client jobs first, then pool units, up to max_jobs not yet committed (jobs
        # accepted while the GPU is busy run together in its next pass)
        if len(computing) < (self.a.max_jobs or 4):
            for jid, e in self.chain.jobs("posted").items():
                if e["client"] != self.account and jid not in self.st["jobs"]:
                    j = self.chain.job(jid)
                    if j and j["model"] == self.model_id and j["deadline"] > h + 5 and self.input_batches(j) is not None:
                        bond = (j["fee"] * 5000 + 9999) // 10000
                        if not self.can_bond(bond, h):
                            break
                        r = self.send([{"kind": "JOB_ACCEPT", "job": jid, "amount": bond}], f"accept {jid[:8]}")
                        if r:
                            self.st["jobs"][jid] = {"accepted": h, "tx": r["txid"], "bond": bond}
                            self.save()
                            return
            if self.a.pool:
                self.claim(s)
        urgent = {"disputed": 0, "provider-reveal": 1, "accepted": 3}
        self.ready, self.gen_ready = {}, {}
        for jid in sorted(active, key=lambda x: urgent.get(mine.get(x, {}).get("status"), 2)):
            self.advance(jid)
        self.launch()

    def can_bond(self, bond: int, h: int) -> bool:
        """A bond the free balance can cover after the bonds already in flight; a claim or accept
        without it would fail inside its block and leave no trace (testnet run 1: 100 such ops)."""
        try:
            free = self.chain.account(self.account).get("balance", 0) - getattr(self, "pending_bonds", 0)
        except Exception:
            return True
        if free >= bond:
            return True
        if self.st.get("low_balance_h") != h:
            self.st["low_balance_h"] = h
            log(f"free balance {free} cannot bond {bond}: waiting for bonds to return")
        return False

    def launch(self):
        """Start the GPU on every accepted job not yet computed (advance collects them in
        self.ready), oldest first, up to --batch-tokens tokens in one pass: small jobs alone
        leave the mined GEMMs short, so a pass of several jobs keeps more of the hash rate."""
        if not self.ready or (self.busy is not None and not self.busy.done()):
            return
        if self.busy is not None:
            busy, self.busy = self.busy, None
            busy.result()  # surface an error from the previous pass once; the next step retries
        order = sorted(self.ready, key=lambda x: self.st["jobs"].get(x, {}).get("accepted",
                                                                          self.st["jobs"].get(x, {}).get("claimed", 0)))
        if order and order[0] in self.gen_ready:  # a generation job runs alone in its pass
            jid = order[0]
            self.busy = self.submit(f"generation job {jid[:8]}", self.compute_gen, jid, self.ready[jid], self.gen_ready[jid])
            return
        order = [x for x in order if x not in self.gen_ready]
        items, n = [], 0
        for jid in order:
            b = self.ready[jid]
            tok = sum(len(x) for x in b)
            if items and n + tok > self.a.batch_tokens:
                break
            items.append((jid, b))
            n += tok
        self.busy = self.submit(f"pass {' '.join(x[:8] for x, _ in items)}", self.compute_jobs, items)

    def claim(self, s: dict):
        # only this provider's model: with Rules 4 several models' categories are open at once,
        # and a unit computed with another model's runtime commits wrong roots
        cats = [(cid, c) for cid, c in s["categories"].items()
                if not c["retired"] and not c.get("closed") and c.get("model") == self.model_id
                and self.work_list(c["workroot"]) is not None]
        random.shuffle(cats)
        cats.sort(key=lambda x: not (self.a.prefer and x[0].startswith(self.a.prefer)))
        for cid, c in cats:
            doc = self.lists[c["workroot"]]
            taken = set()
            for e in self.chain.jobs().values():
                if e.get("pool") == cid and e["status"] not in ("failed", "expired", "aborted"):
                    taken.add(e["unit"])
            tried = self.st.setdefault("tried", {}).setdefault(cid, {})
            free = [i for i in range(c["units"]) if i not in taken and str(i) not in tried]
            if not free:
                continue
            i = free[0]
            bond = (c.get("bondbasis", c["price"]) * 5000 + 9999) // 10000
            if not self.can_bond(bond, s["height"]):
                return
            op = {"kind": "POOL_CLAIM", "job": cid, "u1": i, "h1": doc["units"][i]["input_root"],
                  "proof": unit_proof(doc, i), "amount": bond}
            r = self.send([op], f"claim unit {i} of {cid[:8]}")
            tried[str(i)] = s["height"]
            if r:
                self.st["jobs"][r["job"]] = {"claimed": s["height"], "tx": r["txid"], "bond": bond}
            self.save()
            return

    def advance(self, jid: str):
        j = self.chain.job(jid)
        if j is None:
            return
        js = self.st["jobs"].setdefault(jid, {})
        if j["provider"] != self.account:
            return  # our accept lost the race (or is not mined yet): nothing to compute or commit
        if j["model"] not in self.models:
            # a job of a model this provider does not run (claimed by a version without the model
            # check): never compute it; its bond is lost when it expires, which is better than wrong roots
            if not js.get("foreign_model"):
                js["foreign_model"] = j["model"]
                log(f"job {jid[:8]} is for model {j['model'][:8]}, which this provider does not run: left to expire")
                self.save()
            return
        if j["status"] == "accepted" and "roots" not in js:
            b = self.input_batches(j) if j.get("gen") else self.job_batches(j)
            if b is not None:
                self.ready[jid] = b  # launch() starts the GPU on it, with any others waiting
                if j.get("gen"):
                    self.gen_ready[jid] = self.gen(j)
            return
        if j["status"] == "accepted" and js.get("committed") and self.chain.height() - js.get("committed_h", 0) >= 3:
            js["committed"] = False  # the commit never reached a block: send it again
        if j["status"] == "accepted" and "roots" in js and not js.get("committed"):
            sc = self.commit_of(js)
            if self.send([{"kind": "TRACE_COMMIT", "job": jid, "h1": sc.root.hex(), "h2": js["outputroot"]}],
                         f"commit {jid[:8]}"):
                js["committed"], js["committed_h"] = True, self.chain.height()
                self.save()
            return
        if "roots" not in js:
            return
        if j["status"] in ("committed", "verify-commit", "verify-reveal"):
            self.serve_stage_inputs(jid)
        sc = self.commit_of(js)
        open_ = {r["stage"] for r in (j.get("revealed") or [])}
        want = set()
        if j["status"] == "provider-reveal":
            for v in j.get("committee") or []:
                if v["revealed"]:
                    want |= set(v["positions"] or [])
            if j.get("canary"):
                want |= set(range(j["stages"]))
        g = j.get("game")
        if j["status"] == "disputed" and g and g["phase"] == GAME_REVEAL:
            want.add(g["stage"])
        if j["status"] == "disputed":
            self.play(j, as_provider=True)
        want -= open_
        key = f"{j['status']}:{sorted(want)}"
        h = self.chain.height()
        if want and (js.get("last_reveal") != key or h - js.get("last_reveal_h", h) >= 3):
            if self.send([{"kind": "TRACE_REVEAL", "job": jid, "reveals": [self.reveal_of(j, js, sc, x) for x in sorted(want)]}],
                         f"reveal {jid[:8]} stages {sorted(want)}"):
                js["last_reveal"], js["last_reveal_h"] = key, h
                self.save()

    def reveal_of(self, j: dict, js: dict, sc: StageCommit, stage: int) -> dict:
        """A stage reveal; a generation job's head stage also carries its path in the output root."""
        r = sc.reveal(stage)
        per = self.per_batch(j)
        if j.get("gen") and stage % per == self.G:
            from detgen.head import gen_out_leaves
            from detrt.stage import merkle_proof
            heads = [bytes.fromhex(js["roots"][b * per + self.G]) for b in range(len(js["roots"]) // per)]
            r["out_proof"] = [h.hex() for h in merkle_proof(gen_out_leaves(heads), stage // per)]
        return r

    def job_batches(self, j: dict) -> list[list[int]] | None:
        cache = self.__dict__.setdefault("_batches", {})
        if j["id"] not in cache:
            b = self.batches(j)
            if b is None:
                return None
            cache[j["id"]] = b
            while len(cache) > 64:
                cache.pop(next(iter(cache)))
        return cache[j["id"]]

    def compute_gen(self, jid: str, rows: list[list[int]], gen):
        """A generation job: generate, run every stage, commit salted stage roots and the output
        root; the head-stage outputs (samples) go to the data service for the client."""
        rt = self.rt
        t = time.time()
        mined0 = rt.miner.hashed if rt.miner else 0
        from .compute import prompt_of
        plen = [len(prompt_of(r)) for r in rows]
        last = [0.0]

        def stream(seqs):  # partial results for streaming clients, at most every --stream-every seconds
            now = time.time()
            if now - last[0] < self.a.stream_every:
                return
            last[0] = now
            try:
                body = json.dumps({"tokens": [sq[p:] for sq, p in zip(seqs, plen)]}).encode()
                self.worker_io.submit(self.data.put, f"/stream/{jid}.json", body)
            except Exception:
                pass
        with self.gpu_pass():
            seqs, roots, heads, out_root = rt.gen_job(rows, gen, on_step=stream if self.a.stream_every > 0 else None)
        hashed = (rt.miner.hashed - mined0) if rt.miner else 0
        buf = io.BytesIO()
        np.save(buf, np.stack(heads).astype(np.int32))
        self.data.put(f"/outputs/{jid}.npy", buf.getvalue())
        salts = [secrets.token_bytes(32) for _ in roots]
        from .compute import prompt_of
        new = sum(len(sq) - len(prompt_of(r)) for sq, r in zip(seqs, rows))
        with self.lock:
            self.st["jobs"].setdefault(jid, {}).update(
                roots=[r.hex() for r in roots], salts=[x.hex() for x in salts], outputroot=out_root.hex(), seqs=seqs,
                gen=True, computed=round(rt.last_gpu_s, 4), total=round(time.time() - t, 3), hashed_macs=hashed,
                generated=new)
        self.save()
        log(f"generation job {jid[:8]}: {len(seqs)} sequences, {new} tokens generated, {len(roots)} stages in "
            f"{rt.last_gpu_s:.2f}s on the GPU")

    def compute_jobs(self, items: list[tuple[str, list[list[int]]]]):
        """One GPU pass over several jobs; each job gets its own salts, commitment and outputs.
        Its GPU time and hashed MACs are the pass's, split by its share of the tokens."""
        rt = self.rt  # loads the model first, outside the pause
        t = time.time()
        mined0 = rt.miner.hashed if rt.miner else 0
        with self.gpu_pass():
            torch_sync()
            res = rt.jobs_roots([b for _, b in items], tokens=self.a.batch_tokens)
        gpu_s = self.rt.last_gpu_s
        hashed = (self.rt.miner.hashed - mined0) if self.rt.miner else 0
        tokens = [sum(len(x) for x in b) for _, b in items]
        extra = {}
        if self.rt.miner:
            import torch
            extra = {"device": self.a.device, "gpu": torch.cuda.get_device_name(torch.device(self.a.device))}
        for (jid, b), (roots, embs), tok in zip(items, res, tokens):
            share = tok / sum(tokens)
            salts = [secrets.token_bytes(32) for _ in roots]
            buf = io.BytesIO()
            np.save(buf, np.stack(embs))
            self.data.put(f"/outputs/{jid}.npy", buf.getvalue())
            if self.rt.miner:
                extra["hashed_macs"] = round(hashed * share)
            with self.lock:
                self.st["jobs"].setdefault(jid, {}).update(
                    roots=[r.hex() for r in roots], salts=[x.hex() for x in salts], outputroot=output_root(embs).hex(),
                    computed=round(gpu_s * share, 4), total=round((time.time() - t) * share, 3),
                    batch={"jobs": len(items), "tokens": sum(tokens), "gpu_s": round(gpu_s, 4)}, **extra)
        self.save()
        total = time.time() - t
        log(f"jobs {' '.join(jid[:8] for jid, _ in items)}: {sum(len(r) for r, _ in res)} stages, {sum(tokens)} tokens "
            f"in {gpu_s:.2f}s on the GPU, {total:.2f}s in all")

    def stagein_dir(self, jid: str) -> str:
        return os.path.join(self.a.devnet, "agent", "stagein", jid)

    def serve_stage_inputs(self, jid: str):
        """Answer verifiers' requests for stage inputs (off-chain; DATA_REQUEST stays the fallback).
        The input is recomputed for its one sequence on the GPU worker (queued behind at most one
        pass), so providers keep no rows; inputs saved by older versions are served as they are."""
        try:
            names = json.loads(self.data.get(f"/req/{jid}/") or b"[]")
        except Exception:
            return
        served = self.st["jobs"].setdefault(jid, {}).setdefault("served", [])
        queued = self.__dict__.setdefault("_serving", set())
        for name in names:
            if name in served or name in queued:
                continue
            stage = int(name.split("-")[0])
            path = os.path.join(self.stagein_dir(jid), f"{stage}.json")
            if os.path.exists(path):
                self.data.put(f"/stagein/{jid}/{name}", open(path, "rb").read())
                log(f"served stage {stage} input of {jid[:8]}")
                served.append(name)
                self.save()
                continue
            j = self.chain.job(jid)
            b = self.batches(j) if j else None
            per = self.per_batch(j) if j else self.G
            if b is None or stage % per == 0 or stage >= len(b) * per:
                served.append(name)  # nothing to serve: the verifier recomputes the prefix
                continue
            queued.add(name)
            self.submit(f"stage {stage} input of {jid[:8]}", self.serve_input, jid, name, stage, b, self.gen(j))

    def serve_input(self, jid: str, name: str, stage: int, b: list[list[int]], gen=None):
        import base64
        try:
            rt = self.rt
            t = time.time()
            with self.gpu_pass():
                pk = rt.stage_input(b, stage, gen)
            js = self.st["jobs"][jid]
            doc = {"stage": stage, "shape": list(pk["rows"].shape), "proofs": pk["proofs"],
                   "reveal": self.commit_of(js).reveal(stage - 1),
                   "rows": base64.b64encode(pk["rows"].astype("<f4").tobytes()).decode()}
            self.data.put(f"/stagein/{jid}/{name}", json.dumps(doc).encode())
            with self.lock:
                js.setdefault("served", []).append(name)
            self.save()
            log(f"served stage {stage} input of {jid[:8]} (recomputed in {time.time() - t:.2f}s)")
        except Exception:
            traceback.print_exc()
        finally:
            self._serving.discard(name)

    @staticmethod
    def commit_of(js: dict) -> StageCommit:
        return StageCommit([bytes.fromhex(r) for r in js["roots"]], [bytes.fromhex(s) for s in js["salts"]])


# ---------------------------------------------------------------- verifier

class Verifier(Agent):
    """Stakes, then for every committee seat: private salt, recompute the sampled stages,
    commit H(salt) and H(roots, account, salt), reveal in the next window."""
    role = "verifier"

    def step(self):
        acct = self.chain.account(self.account)
        if acct["stake"] < self.a.stake and acct["balance"] >= self.a.stake - acct["stake"] and not self.st.get("staking"):
            if self.send([{"kind": "STAKE", "amount": self.a.stake - acct["stake"]}], "stake"):
                self.st["staking"] = True
                self.save()
        elif acct["stake"] >= self.a.stake:
            self.st["staking"] = False
        self.challenger_games()
        for status in ("verify-commit", "verify-reveal"):
            for jid in self.chain.jobs(status):
                j = self.chain.job(jid)
                seat = next((v for v in j["committee"] or [] if v["account"] == self.account), None)
                if seat is None:
                    continue
                js = self.st["jobs"].setdefault(jid, {})
                h = self.chain.height()
                # resend a commit or reveal the chain has not recorded after 3 blocks
                if status == "verify-commit" and js.get("committed") and not seat["committed"] and h - js.get("sent_h", h) >= 3:
                    js["resend"] = True
                if status == "verify-reveal" and js.get("revealed") and not seat["revealed"] and h - js.get("sent_h", h) >= 3:
                    js["revealed"] = False
                if status == "verify-commit" and js.get("resend") and "salt" in js:
                    salt = bytes.fromhex(js["salt"])
                    rh = result_hash([bytes.fromhex(r) for r in js["roots"]], bytes.fromhex(self.account), salt)
                    if self.send([{"kind": "VERIFY_COMMIT", "job": jid, "h1": hashlib.sha256(salt).hexdigest(),
                                   "h2": rh.hex()}], f"verify-commit {jid[:8]} (resend)"):
                        js["resend"], js["sent_h"] = False, h
                        self.save()
                    continue
                if status == "verify-commit" and not seat["committed"] and not js.get("committed"):
                    if not self.use(j["model"]):
                        continue
                    b = self.batches(j)
                    if b is None:
                        continue
                    salt = secrets.token_bytes(32)
                    pos = positions(salt, j["stages"], j["n"])
                    t = time.time()
                    roots, served = [], 0
                    for p in pos:
                        r = self.served_stage_root(j, p)
                        served += r is not None
                        roots.append(r if r is not None else self.rt.stage_root(b, p, self.gen(j)))
                    js.update(salt=salt.hex(), positions=pos, roots=[r.hex() for r in roots],
                              computed=round(time.time() - t, 1), served_inputs=served)
                    rh = result_hash(roots, bytes.fromhex(self.account), salt)
                    if self.send([{"kind": "VERIFY_COMMIT", "job": jid, "h1": hashlib.sha256(salt).hexdigest(),
                                   "h2": rh.hex()}], f"verify-commit {jid[:8]} stages {pos}"):
                        js["committed"], js["sent_h"] = True, self.chain.height()
                    self.save()
                elif status == "verify-reveal" and seat["committed"] and not seat["revealed"] and js.get("committed") \
                        and not js.get("revealed"):
                    if self.send([{"kind": "VERIFY_REVEAL", "job": jid, "h1": js["salt"], "roots": js["roots"]}],
                                 f"verify-reveal {jid[:8]}"):
                        js["revealed"], js["sent_h"] = True, self.chain.height()
                        self.save()


    def served_stage_root(self, j: dict, stage: int) -> bytes | None:
        """Recompute one stage from its served input, checked against the commitment; None (so
        the caller recomputes the prefix) when the stage reads tokens or nothing valid arrives."""
        g = stage % self.per_batch(j)
        if g == 0 or not self.a.fetch_inputs:
            return None
        nonce = secrets.token_hex(16)
        name = f"{stage}-{nonce}.json"
        try:
            self.data.put(f"/req/{j['id']}/{name}", b"{}")
        except Exception as e:
            log(f"stage input request failed: {e}")
            return None
        deadline = time.time() + self.a.input_wait
        while time.time() < deadline:
            raw = self.data.get(f"/stagein/{j['id']}/{name}")
            if raw:
                try:
                    pkg = json.loads(raw)
                    rows = check_stage_input(pkg, stage, bytes.fromhex(j["stageroot"]), j["stages"])
                except Exception as e:
                    log(f"{j['id'][:8]} stage {stage}: served input rejected ({e}); recomputing the prefix")
                    return None
                return self.rt.stage_root_from_input(rows, g, self.gen(j))
            time.sleep(3)
        log(f"{j['id'][:8]} stage {stage}: no input served in {self.a.input_wait:.0f}s; recomputing the prefix")
        return None


# ---------------------------------------------------------------- watcher

class Watcher(Agent):
    """Checks a share of jobs in their challenge window: recompute one random stage and post it
    as a bonded CHALLENGE. A matching root costs only the watch fee; a wrong provider root
    becomes a dispute (and, on a decoy, a jackpot)."""
    role = "watcher"

    def step(self):
        self.challenger_games()
        rng = random.Random()
        for jid, e in self.chain.jobs("challenge").items():
            js = self.st["jobs"].get(jid)
            if js is not None or e["provider"] == self.account:
                continue
            js = self.st["jobs"][jid] = {"seen": True}
            if rng.random() >= self.a.rate:
                js["skipped"] = True
                self.save()
                continue
            j = self.chain.job(jid)
            if not self.use(j["model"]):
                js["skipped"] = "model"
                self.save()
                continue
            b = self.batches(j)
            if b is None:  # inputs not available yet (e.g. a work list fetch failed): try again later
                del self.st["jobs"][jid]
                continue
            stage = rng.randrange(j["stages"])
            t = time.time()
            root = self.rt.stage_root(b, stage, self.gen(j)).hex()
            if rng.random() < self.a.lie:  # fault injection: claim an output root with one element flipped
                from .game import split
                from .proto import stage_output_root
                tr = self.rt.trace(b, stage, self.gen(j))
                _, out_at, _, _ = split(tr)
                t = rng.randrange(len(out_at))
                leaf, elem = out_at[t], rng.randrange(len(tr.rows[out_at[t]]))
                js["lie"] = [leaf, elem]
                bad = self.corrupted(tr, leaf, elem)
                root = stage_output_root([bad.rows[i] for i in out_at]).hex()
                log(f"lying about {jid[:8]} stage {stage}: output row {t} elem {elem}")
            js.update(stage=stage, root=root, computed=round(time.time() - t, 1))
            opened = {r["stage"]: r["root"] for r in j.get("revealed") or []}
            if opened.get(stage) == root:
                js["agreed_open"] = True  # already open and correct: nothing to post
                log(f"check {jid[:8]} stage {stage}: matches the opened root")
            else:
                if self.send([{"kind": "CHALLENGE", "job": jid, "u1": stage, "h1": root, "amount": self.challenge_bond(j)}],
                             f"check {jid[:8]} stage {stage}"):
                    js["challenged"] = True
            self.save()


# ---------------------------------------------------------------- client

    def corrupted(self, tr, leaf: int, elem: int):
        import copy
        from detrt.stage import StageTrace
        rows = list(tr.rows)
        r = rows[leaf].copy()
        r.view(np.uint32)[elem] ^= 0x00400001
        rows[leaf] = r
        return StageTrace(leaves=tr.leaves, rows=rows, output=tr.output).finalize()

    def game_trace(self, j: dict, b, stage: int, as_provider: bool):
        tr = self.rt.trace(b, stage, self.gen(j))
        lie = self.st["jobs"].get(j["id"], {}).get("lie")
        return self.corrupted(tr, *lie) if lie and not as_provider else tr


class Client(Agent):
    """Posts embedding jobs over public text and checks the delivered outputs."""
    role = "client"

    def step(self):
        h = self.chain.height()
        for _ in range(self.a.burst if h - self.st.get("last_post", -10**9) >= self.a.every else 0):
            if not (self.a.post > 0 and len(self.st.get("posted", [])) < self.a.post):
                break
            n = len(self.st.get("posted", []))
            off = self.a.offset + n * self.a.batches
            if self.a.gen:  # prompts of --prompt-len tokens, generated up to --seq-len
                from detgen.head import input_row
                prompts = wikitext_sequences(self.a.dataset, self.model["model_dir"], self.a.prompt_len, self.a.batches, off)
                seqs = [input_row(p, self.a.seq_len) for p in prompts]
                stages = len(seqs) * (self.G + 1)
                cmd = ["jobgen", self.key, self.model_id, input_root(seqs).hex(), str(stages), str(self.a.fee),
                       str(self.a.deadline), str(self.a.k), str(self.a.n), str(self.a.seq_len), str(self.a.temp_milli)]
            else:
                seqs = wikitext_sequences(self.a.dataset, self.model["model_dir"], self.a.seq_len, self.a.batches, off)
                stages = len(seqs) * self.G
                cmd = ["jobpost", self.key, self.model_id, input_root(seqs).hex(), str(stages), str(self.a.fee),
                       str(self.a.deadline), str(self.a.k), str(self.a.n), str(self.a.seq_len)]
            root = input_root(seqs).hex()
            self.data.put(f"/inputs/{root}.json", json.dumps({"batches": seqs}).encode())
            try:
                r = self.chain.run(*cmd)
                log(f"posted {'generation ' if self.a.gen else ''}job {r['job'][:8]}: {len(seqs)} sequences, {stages} stages, fee {self.a.fee}")
                self.st.setdefault("posted", []).append({"job": r["job"], "height": h, "offset": off})
            except ChainError as e:
                log(f"post: {e}")
            self.st["last_post"] = h
            self.save()
        for p in self.st.get("posted", []):
            if p.get("final"):
                continue
            j = self.chain.job(p["job"])
            if j is None:
                continue
            if j["outputroot"] != "0" * 64 and "outputs_ok" not in p:
                raw = self.data.get(f"/outputs/{p['job']}.npy")
                if raw is not None:
                    outs = np.load(io.BytesIO(raw))
                    if j.get("gen"):  # head-stage samples per batch: their roots make the output root
                        from detgen.head import gen_output_root, head_output_root
                        p["outputs_ok"] = gen_output_root([head_output_root(o) for o in outs]).hex() == j["outputroot"]
                        if p["outputs_ok"]:
                            self.show_generation(j, outs)
                    else:
                        p["outputs_ok"] = output_root(list(outs)).hex() == j["outputroot"]
                    p["delivered"] = h
                    log(f"job {p['job'][:8]}: outputs {'match' if p['outputs_ok'] else 'DO NOT match'} output_root")
            if j["status"] in ("settled", "expired", "failed", "aborted"):
                p.update(final=j["status"], ended=h)
                log(f"job {p['job'][:8]}: {j['status']} at height {h}")
            self.save()


    def show_generation(self, j: dict, outs: np.ndarray):
        """Log the first generated continuation (the client's view of a generation job)."""
        try:
            from transformers import AutoTokenizer
            from .compute import prompt_of
            rows = self.input_batches(j)
            P = len(prompt_of(rows[0]))
            tok = AutoTokenizer.from_pretrained(self.model["model_dir"])
            log(f"job {j['id'][:8]} generated: {tok.decode([int(x) for x in outs[0][P - 1:, 0]])!r}"[:400])
        except Exception as e:
            log(f"job {j['id'][:8]}: could not decode the generation ({e})")


# ---------------------------------------------------------------- curator (canary units)

class Curator(Agent):
    """Reveals a canary unit's expected stage roots once a provider has committed it. Every stage
    is then opened, and the first stage that differs is disputed down to one operation."""
    role = "curator"

    def step(self):
        secrets_dir = os.path.join(os.path.expanduser(os.environ.get("UWC_HOME", "~/.uwc/devnet")), "canary")
        canaries = {}
        for f in os.listdir(secrets_dir) if os.path.isdir(secrets_dir) else []:
            c = json.load(open(os.path.join(secrets_dir, f)))
            canaries[c["category"]] = c
        for jid, e in self.chain.jobs().items():
            c = canaries.get(e.get("pool"))
            if not c or e["status"] not in ("committed", "verify-commit", "verify-reveal", "provider-reveal", "challenge"):
                continue
            entry = c["canaries"].get(str(e["unit"]))
            if entry is None or jid in self.st["jobs"]:
                continue
            j = self.chain.job(jid)
            if j.get("canary"):
                continue
            op = {"kind": "CANARY_REVEAL", "job": jid, "u1": entry["leaf"], "h1": c["salt"], "roots": entry["roots"],
                  "proof": entry["proof"]}
            if self.send([op], f"canary {jid[:8]} (unit {e['unit']})"):
                self.st["jobs"][jid] = {"unit": e["unit"]}
                self.save()
        # the curator also plays the canary disputes it opened
        self.challenger_games()


# ---------------------------------------------------------------- decoys (forced errors)

class Decoy(Provider):
    """The keyholders' decoy runner: posts a job from a fresh client account, accepts it at once
    with a decoy provider account listed in the council's forced-error record, commits honest
    roots except one deliberately wrong stage, reveals whatever is asked, and defends
    games over its honest stages but never the wrong one. Whoever catches it wins a jackpot once the record is revealed."""
    role = "decoy"

    def step(self):
        h = self.chain.height()
        for jid in list(self.st["jobs"]):
            js = self.st["jobs"][jid]
            if js.get("final"):
                continue
            j = self.chain.job(jid)
            if j is None:
                if h - js["posted"] > 20:
                    js["final"] = "lost"
                continue
            if j["provider"] not in ("0" * 64, self.account):
                # another provider's accept was mined first: an ordinary job now, not a decoy sample
                js["final"] = "lost"
                log(f"decoy {jid[:8]} (wrong stage {js['wrong']}): lost, accepted by {j['provider'][:12]}")
                self.save()
                continue
            if j["status"] in ("settled", "expired", "failed", "aborted"):
                js["final"] = j["status"]
                log(f"decoy {jid[:8]} (wrong stage {js['wrong']}): {j['status']}")
                self.save()
                continue
            if j["status"] == "posted" and j["provider"] == "0" * 64 and not js.get("accept_sent"):
                bond = (j["fee"] * 5000 + 9999) // 10000
                if self.send([{"kind": "JOB_ACCEPT", "job": jid, "amount": bond}], f"decoy accept {jid[:8]}"):
                    js["accept_sent"] = True
                    self.save()
            if j["provider"] != self.account:
                continue
            if j["status"] == "accepted" and not js.get("committed"):
                sc = self.commit_of(js)
                if self.send([{"kind": "TRACE_COMMIT", "job": jid, "h1": sc.root.hex(), "h2": js["outputroot"]}],
                             f"decoy commit {jid[:8]}"):
                    js["committed"] = True
                    self.save()
            elif j["status"] in ("provider-reveal", "disputed"):
                self.reveals_only(j, js)
                g = j.get("game")
                # Defend every honest stage: a silent provider loses any game, so a decoy that never
                # defends would make any challenger, even one lying about a correct stage, a catcher.
                if j["status"] == "disputed" and g and g["stage"] != js["wrong"]:
                    self.play(j, as_provider=True)
        active = [js for js in self.st["jobs"].values() if not js.get("final")]
        # every decoy has ended: the keyholders reveal the record so catchers get their jackpots
        if not active and len(self.st["jobs"]) >= self.a.post and not self.st.get("revealed"):
            import subprocess
            p = subprocess.run([sys.executable, "-m", "agent.fe", "reveal", "--devnet", self.a.devnet],
                               capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), ".."))
            log(f"forced-error reveal: {p.stdout.strip() or p.stderr.strip()[-300:]}")
            if p.returncode == 0:
                self.st["revealed"] = h
                self.save()
            return
        if len(active) < (self.a.max_jobs or 1) and h - self.st.get("last_post", -10**9) >= self.a.every and \
                self.a.post > len(self.st["jobs"]):
            self.post_decoy(h)

    def reveals_only(self, j: dict, js: dict):
        sc = self.commit_of(js)
        open_ = {r["stage"] for r in (j.get("revealed") or [])}
        want = set()
        if j["status"] == "provider-reveal":
            for v in j.get("committee") or []:
                if v["revealed"]:
                    want |= set(v["positions"] or [])
        g = j.get("game")
        if j["status"] == "disputed" and g and g["phase"] == GAME_REVEAL:
            want.add(g["stage"])
        want -= open_
        key = f"{j['status']}:{sorted(want)}"
        if want and js.get("last_reveal") != key:
            if self.send([{"kind": "TRACE_REVEAL", "job": j["id"], "reveals": [sc.reveal(x) for x in sorted(want)]}],
                         f"decoy reveal {j['id'][:8]} {sorted(want)}"):
                js["last_reveal"] = key
                self.save()

    def stage_key(self) -> bytes | None:
        """The stage key of the newest unrevealed forced-error commitment (agent.fe writes it for
        the keyholders' decoy runner): from Rules 2 it fixes each decoy's wrong stage."""
        from .fe import FE_DIR
        best = None
        for d in os.listdir(FE_DIR) if os.path.isdir(FE_DIR) else []:
            info_p, key_p = os.path.join(FE_DIR, d, "commit.json"), os.path.join(FE_DIR, d, "decoy.json")
            if os.path.exists(info_p) and os.path.exists(key_p) and not json.load(open(info_p)).get("revealed"):
                m = os.path.getmtime(key_p)
                if best is None or m > best[0]:
                    best = (m, bytes.fromhex(json.load(open(key_p))["stage_key"]))
        return best[1] if best else None

    def post_decoy(self, h: int):
        n = len(self.st["jobs"])
        seqs = wikitext_sequences(self.a.dataset, self.model["model_dir"], self.a.seq_len, self.a.batches,
                                  self.a.offset + n * self.a.batches)
        if not seqs:
            log("decoy: no text left at this offset")
            self.st["last_post"] = h
            return
        rules2 = (self.chain.state().get("params") or {}).get("rules", 0) >= 2
        key = self.stage_key() if rules2 else None
        if rules2 and key is None:
            log("decoy: Rules 2 needs the commitment's stage key (agent.fe commit writes decoy.json)")
            self.st["last_post"] = h
            return
        # honest roots first, on the GPU worker (they do not depend on the job id): the job is
        # posted on a later step, once they are ready
        done = self.on_gpu(("decoy roots", n), lambda: [self.rt.batch(toks) for toks in seqs])
        if done is None:
            return
        root = input_root(seqs).hex()
        self.data.put(f"/inputs/{root}.json", json.dumps({"batches": seqs}).encode())
        # Rules 2: the job is directed to the decoy account (no accept race), and its wrong stage
        # follows from the job id and the stage key, so a catch elsewhere earns no jackpot
        directed = [self.account] if rules2 else []
        try:
            r = self.chain.run("jobpost", os.path.expanduser(self.a.client_key), self.model_id, root,
                               str(len(seqs) * self.G), str(self.a.fee), str(self.a.deadline), str(self.a.k),
                               str(self.a.n), str(self.a.seq_len), *directed)
        except ChainError as e:
            log(f"decoy post: {e}")
            return
        jid = r["job"]
        roots, embs = [], []
        for rr, e in done:
            roots += rr
            embs.append(e)
        if key is not None:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "verifier"))
            import forced_errors as fe
            wrong = fe.fe_stage(key, bytes.fromhex(jid), len(roots))
        else:
            wrong = random.randrange(len(roots))
        roots[wrong] = secrets.token_bytes(32)  # the forced error
        salts = [secrets.token_bytes(32) for _ in roots]
        buf = io.BytesIO()
        np.save(buf, np.stack(embs))
        self.data.put(f"/outputs/{jid}.npy", buf.getvalue())
        self.st["jobs"][jid] = {"posted": h, "wrong": wrong, "roots": [x.hex() for x in roots],
                                "salts": [x.hex() for x in salts], "outputroot": output_root(embs).hex()}
        self.st["last_post"] = h
        self.save()
        log(f"decoy job {jid[:8]} posted: {len(roots)} stages, wrong root at stage {wrong}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["provider", "verifier", "watcher", "client", "decoy", "curator"])
    ap.add_argument("--client-key", default="~/.uwc/devnet/keys/decoyclient.json", help="decoy: the decoys' client")
    ap.add_argument("--key", required=True)
    ap.add_argument("--model", default=os.path.join(DEVNET, "models", "qwen3-1.7b.json"))
    ap.add_argument("--units", default=os.path.join(DEVNET, "pool", "wiki128.units.json"))
    ap.add_argument("--devnet", default=DEVNET)
    ap.add_argument("--data", default=DATA_URL)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--engine", default="detgpu", choices=["detgpu", "detrt"],
                    help="detgpu: fast engine, bit-identical to the registered runtime detrt")
    ap.add_argument("--weights", default=os.path.join(DEVNET, "weights", "qwen3-1.7b"))
    ap.add_argument("--stall", type=float, default=600.0,
                    help="watchdog: log every thread's stack when the main loop has not finished a step for this many seconds")
    ap.add_argument("--task-stall", type=float, default=1800.0,
                    help="watchdog: log every thread's stack when one GPU task has run this many seconds")
    # provider
    ap.add_argument("--max-jobs", type=int, default=None,
                    help="provider: jobs accepted or claimed but not yet committed (default 4); decoy: active decoys (default 1)")
    ap.add_argument("--batch-tokens", type=int, default=8192, help="provider: tokens per GPU pass (jobs share passes)")
    ap.add_argument("--pool", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--prefer", default="", help="provider: claim from categories with this id prefix first")
    ap.add_argument("--mine", action="store_true", help="provider: mine on every job GEMM (needs this box's gateway)")
    ap.add_argument("--pause-file", default=None, help="provider: exists during GPU passes (pauses a noise miner on the GPU)")
    # verifier
    ap.add_argument("--stake", type=int, default=100_000_000)
    ap.add_argument("--fetch-inputs", action=argparse.BooleanOptionalAction, default=True,
                    help="verifier: ask the provider for stage inputs instead of recomputing the prefix")
    ap.add_argument("--input-wait", type=float, default=90.0, help="verifier: seconds to wait for a served input")
    # watcher
    ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--bond", type=int, default=100_000)
    ap.add_argument("--lie", type=float, default=0.0, help="fault injection: share of checks posted with a wrong root")
    # client
    ap.add_argument("--post", type=int, default=0, help="jobs to post")
    ap.add_argument("--every", type=int, default=30, help="blocks between posts")
    ap.add_argument("--burst", type=int, default=1, help="client: jobs posted together each time")
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--gen", action="store_true", help="client: post generation jobs (Rules 3)")
    ap.add_argument("--stream-every", type=float, default=0.5, help="provider: seconds between partial generation uploads (0: off)")
    ap.add_argument("--prompt-len", type=int, default=32, help="client --gen: prompt tokens per sequence")
    ap.add_argument("--temp-milli", type=int, default=700, help="client --gen: temperature in thousandths (0: greedy)")
    ap.add_argument("--offset", type=int, default=1000, help="first WikiText chunk (the pool uses 0-199)")
    ap.add_argument("--fee", type=int, default=3_000_000_000)
    ap.add_argument("--deadline", type=int, default=60)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--dataset", default=os.path.expanduser("~/Documents/useful-work-coin/models/wikitext2"))
    a = ap.parse_args()
    # Resolve transformers' lazy attributes before the worker thread loads the model: a first
    # `from transformers import AutoTokenizer` racing that load fails with ImportError.
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    {"provider": Provider, "verifier": Verifier, "watcher": Watcher, "client": Client, "decoy": Decoy,
     "curator": Curator}[a.role](a).run()


if __name__ == "__main__":
    main()
