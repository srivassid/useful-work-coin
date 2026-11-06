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
import hashlib
import io
import json
import os
import random
import secrets
import threading
import time
import traceback

import numpy as np

from .chain import Chain, ChainError
from .data import Data
from .inputs import input_root, unit_proof, wikitext_sequences
from .game import GONESTEP, GREVEAL, move, write_rows
from .proto import StageCommit, job_id, output_root, positions, result_hash

DEVNET = os.path.expanduser("~/Documents/useful-work-coin/devnet")
DATA_URL = "http://192.168.29.71:18680"
GAME_REVEAL = 1  # jobstate.GReveal


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Agent:
    role = "agent"

    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.chain = Chain(a.devnet)
        self.key = os.path.expanduser(a.key)
        info = self.chain.key_info(self.key)
        self.account = info["account"]
        self.data = Data(a.data)
        self.model = json.load(open(a.model))
        self.model_id = json.load(open(a.model.replace(".json", ".reg.json")))["model"]
        self.G = self.model["layer_groups"]
        # public work lists by work root (comma-separated --units files)
        self.lists = {}
        for path in (a.units or "").split(","):
            if path and os.path.exists(path):
                doc = json.load(open(path))
                self.lists[doc["work_root"]] = doc
        self.units = next(iter(self.lists.values()), None)
        self._cat_roots: dict[str, str] = {}
        self.path = os.path.join(a.devnet, "agent", f"{self.role}-{self.account[:8]}.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.st = json.load(open(self.path)) if os.path.exists(self.path) else {"jobs": {}}
        self._rt = None
        # long computations (whole jobs) run here, so the main loop can always answer reveals
        # and game rounds before their deadlines
        self.worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.busy: concurrent.futures.Future | None = None

    rt_lock = threading.Lock()

    @property
    def rt(self):
        with self.rt_lock:
            if self._rt is None:
                from .compute import Runtime
                self._rt = Runtime(self.a.model, self.a.device)
            return self._rt

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

    def batches(self, j: dict) -> list[list[int]] | None:
        """The job's token batches: a pool unit from the public work list, a client job from the
        data service (checked against input_root)."""
        if j.get("pool"):
            doc = self.lists.get(self.work_root(j["pool"]))
            return [doc["units"][j["unit"]]["tokens"]] if doc else None
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
        if (g["pmoved"] if as_provider else g["cmoved"]):
            return
        js = self.st["jobs"].setdefault(j["id"], {})
        tag = f"{g['stage']}:{g['phase']}:{g['lo']}:{g['size']}:{g['leaf']}:{g['block']}"
        if js.get("last_move") == tag:
            return
        b = self.batches(j)
        if b is None:
            return
        if g["phase"] == GONESTEP:
            if as_provider:
                self.one_step(j, b, js, tag)
            return
        op = move(j["id"], g, self.game_trace(j, b, g["stage"], as_provider))
        if op and self.send([op], f"game {j['id'][:8]} stage {g['stage']} phase {g['phase']} lo {g['lo']} size {g['size']}"):
            js["last_move"] = tag
            self.save()

    def game_trace(self, j: dict, b, stage: int, as_provider: bool):
        return self.rt.trace(b, stage)

    def one_step(self, j: dict, b: list[list[int]], js: dict, tag: str):
        """Last round as provider: every row the referee reads, with proofs (uwcctl onestep)."""
        g = j["game"]
        d = os.path.join(self.a.devnet, "agent", "onestep", j["id"][:16])
        os.makedirs(d, exist_ok=True)
        req = {"job": j["id"], "model": self.a.model, "weights": self.a.weights, "rows": os.path.join(d, "rows.bin"),
               "prev_rows": "", "batches": b, "salts": js["salts"], "roots": js["roots"]}
        if g["stage"] % self.G:
            write_rows(os.path.join(d, "prev.bin"), self.rt.trace(b, g["stage"] - 1).rows)
            req["prev_rows"] = os.path.join(d, "prev.bin")
        write_rows(req["rows"], self.rt.trace(b, g["stage"]).rows)
        json.dump(req, open(os.path.join(d, "req.json"), "w"))
        try:
            r = self.chain.run("onestep", self.key, os.path.join(d, "req.json"))
            log(f"one-step {j['id'][:8]} stage {g['stage']} leaf {g['leaf']} elem {g['elem']}: tx {r['txid'][:12]}")
            js["last_move"] = tag
            self.save()
        except ChainError as e:
            log(f"one-step {j['id'][:8]}: {e}")

    def challenger_games(self):
        """Play every game in which this account is the disputant (flagging verifier or watcher)."""
        for jid in self.chain.jobs("disputed"):
            j = self.chain.job(jid)
            if j and j.get("disputes") and j["disputes"][0]["challenger"] == self.account:
                self.play(j, as_provider=False)

    def work_root(self, cid: str) -> str:
        if cid not in self._cat_roots:
            self._cat_roots[cid] = self.chain.category(cid)["workroot"]
        return self._cat_roots[cid]

    def run(self):
        log(f"{self.role} {self.account[:12]} starting")
        if self.role != "client":  # load the model in the background; reveals need no model
            self.busy = self.worker.submit(lambda: self.rt)
        while True:
            try:
                self.step()
            except Exception:
                traceback.print_exc()
            time.sleep(self.a.poll)

    def step(self):
        raise NotImplementedError


# ---------------------------------------------------------------- provider

class Provider(Agent):
    """Claims pool units and accepts client jobs, runs every stage, commits salted stage roots
    and the output root, delivers outputs off-chain, reveals what is requested."""
    role = "provider"

    def step(self):
        s = self.chain.state()
        h = s["height"]
        mine = {jid: e for jid, e in self.chain.jobs().items() if e["provider"] == self.account}
        active = [jid for jid, e in mine.items() if e["status"] not in ("settled", "expired", "failed", "aborted")]
        # the GPU is busy only until a job is committed: limit jobs not yet committed, counting
        # claims and accepts sent but not yet in a block
        computing = [jid for jid in active if mine[jid]["status"] == "accepted"]
        for jid, js in self.st["jobs"].items():
            sent = js.get("claimed", js.get("accepted"))
            if jid not in mine and sent is not None and h - sent < 20:
                computing.append(jid)
                active.append(jid)
        # new work: client jobs first, then a pool unit if nothing else is computing
        if len(computing) < self.a.max_jobs:
            for jid, e in self.chain.jobs("posted").items():
                if e["client"] != self.account and jid not in self.st["jobs"]:
                    j = self.chain.job(jid)
                    if j and j["model"] == self.model_id and j["deadline"] > h + 5 and self.batches(j) is not None:
                        bond = (j["fee"] * 5000 + 9999) // 10000
                        if self.send([{"kind": "JOB_ACCEPT", "job": jid, "amount": bond}], f"accept {jid[:8]}"):
                            self.st["jobs"][jid] = {"accepted": h}
                            self.save()
                            return
            if not computing and self.units and self.a.pool:
                self.claim(s)
        urgent = {"disputed": 0, "provider-reveal": 1, "accepted": 3}
        for jid in sorted(active, key=lambda x: urgent.get(mine.get(x, {}).get("status"), 2)):
            self.advance(jid)

    def claim(self, s: dict):
        cats = [(cid, c) for cid, c in s["categories"].items() if not c["retired"] and c["workroot"] in self.lists]
        random.shuffle(cats)
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
            op = {"kind": "POOL_CLAIM", "job": cid, "u1": i, "h1": doc["units"][i]["input_root"],
                  "proof": unit_proof(doc, i), "amount": bond}
            r = self.send([op], f"claim unit {i} of {cid[:8]}")
            tried[str(i)] = s["height"]
            if r:
                self.st["jobs"][r["job"]] = {"claimed": s["height"]}
            self.save()
            return

    def advance(self, jid: str):
        j = self.chain.job(jid)
        if j is None:
            return
        js = self.st["jobs"].setdefault(jid, {})
        if j["status"] == "accepted" and "roots" not in js:
            if self.busy is not None and not self.busy.done():
                return  # one job at a time on the GPU; the main loop keeps serving other jobs
            if self.busy is not None:
                self.busy.result()  # surface errors from the previous job
                self.busy = None
            b = self.batches(j)
            if b is None:
                return
            self.busy = self.worker.submit(self.compute_job, jid, b)
            return
        if j["status"] == "accepted" and "roots" in js and not js.get("committed"):
            sc = self.commit_of(js)
            if self.send([{"kind": "TRACE_COMMIT", "job": jid, "h1": sc.root.hex(), "h2": js["outputroot"]}],
                         f"commit {jid[:8]}"):
                js["committed"] = True
                self.save()
            return
        if "roots" not in js:
            return
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
        if want and js.get("last_reveal") != key:
            if self.send([{"kind": "TRACE_REVEAL", "job": jid, "reveals": [sc.reveal(x) for x in sorted(want)]}],
                         f"reveal {jid[:8]} stages {sorted(want)}"):
                js["last_reveal"] = key
                self.save()

    def compute_job(self, jid: str, b: list[list[int]]):
        t = time.time()
        roots, embs = [], []
        for toks in b:
            r, e = self.rt.batch(toks)
            roots += r
            embs.append(e)
        salts = [secrets.token_bytes(32) for _ in roots]
        out_root = output_root(embs).hex()
        buf = io.BytesIO()
        np.save(buf, np.stack(embs))
        self.data.put(f"/outputs/{jid}.npy", buf.getvalue())
        with self.lock:
            self.st["jobs"].setdefault(jid, {}).update(roots=[r.hex() for r in roots], salts=[x.hex() for x in salts],
                                                       outputroot=out_root, computed=round(time.time() - t, 1))
        self.save()
        log(f"job {jid[:8]}: {len(roots)} stages in {time.time() - t:.0f}s")

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
                if status == "verify-commit" and not seat["committed"] and not js.get("committed"):
                    b = self.batches(j)
                    if b is None:
                        continue
                    salt = secrets.token_bytes(32)
                    pos = positions(salt, j["stages"], j["n"])
                    t = time.time()
                    roots = [self.rt.stage_root(b, p) for p in pos]
                    js.update(salt=salt.hex(), positions=pos, roots=[r.hex() for r in roots],
                              computed=round(time.time() - t, 1))
                    rh = result_hash(roots, bytes.fromhex(self.account), salt)
                    if self.send([{"kind": "VERIFY_COMMIT", "job": jid, "h1": hashlib.sha256(salt).hexdigest(),
                                   "h2": rh.hex()}], f"verify-commit {jid[:8]} stages {pos}"):
                        js["committed"] = True
                    self.save()
                elif status == "verify-reveal" and seat["committed"] and not seat["revealed"] and js.get("committed") \
                        and not js.get("revealed"):
                    if self.send([{"kind": "VERIFY_REVEAL", "job": jid, "h1": js["salt"], "roots": js["roots"]}],
                                 f"verify-reveal {jid[:8]}"):
                        js["revealed"] = True
                        self.save()


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
            b = self.batches(j)
            if b is None:
                continue
            stage = rng.randrange(j["stages"])
            t = time.time()
            root = self.rt.stage_root(b, stage).hex()
            if rng.random() < self.a.lie:  # fault injection: claim a root with one element flipped
                tr = self.rt.trace(b, stage)
                leaf = rng.randrange(len(tr.rows))
                elem = rng.randrange(len(tr.rows[leaf]))
                js["lie"] = [leaf, elem]
                root = self.corrupted(tr, leaf, elem).root.hex()
                log(f"lying about {jid[:8]} stage {stage}: leaf {leaf} elem {elem}")
            js.update(stage=stage, root=root, computed=round(time.time() - t, 1))
            opened = {r["stage"]: r["root"] for r in j.get("revealed") or []}
            if opened.get(stage) == root:
                js["agreed_open"] = True  # already open and correct: nothing to post
                log(f"check {jid[:8]} stage {stage}: matches the opened root")
            else:
                if self.send([{"kind": "CHALLENGE", "job": jid, "u1": stage, "h1": root, "amount": self.a.bond}],
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
        tr = self.rt.trace(b, stage)
        lie = self.st["jobs"].get(j["id"], {}).get("lie")
        return self.corrupted(tr, *lie) if lie and not as_provider else tr


class Client(Agent):
    """Posts embedding jobs over public text and checks the delivered outputs."""
    role = "client"

    def step(self):
        h = self.chain.height()
        if self.a.post > 0 and len(self.st.get("posted", [])) < self.a.post and \
                h - self.st.get("last_post", -10**9) >= self.a.every:
            n = len(self.st.get("posted", []))
            off = self.a.offset + n * self.a.batches
            seqs = wikitext_sequences(self.a.dataset, self.model["model_dir"], self.a.seq_len, self.a.batches, off)
            root = input_root(seqs).hex()
            self.data.put(f"/inputs/{root}.json", json.dumps({"batches": seqs}).encode())
            try:
                r = self.chain.run("jobpost", self.key, self.model_id, root, str(len(seqs) * self.G), str(self.a.fee),
                                   str(self.a.deadline), str(self.a.k), str(self.a.n), str(self.a.seq_len))
                log(f"posted job {r['job'][:8]}: {len(seqs)} sequences, {len(seqs) * self.G} stages, fee {self.a.fee}")
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
                    embs = np.load(io.BytesIO(raw))
                    p["outputs_ok"] = output_root(list(embs)).hex() == j["outputroot"]
                    p["delivered"] = h
                    log(f"job {p['job'][:8]}: outputs {'match' if p['outputs_ok'] else 'DO NOT match'} output_root")
            if j["status"] in ("settled", "expired", "failed", "aborted"):
                p.update(final=j["status"], ended=h)
                log(f"job {p['job'][:8]}: {j['status']} at height {h}")
            self.save()


# ---------------------------------------------------------------- curator (canary units)

class Curator(Agent):
    """Reveals a canary unit's expected stage roots once a provider has committed it. Every stage
    is then opened, and the first stage that differs is disputed down to one operation."""
    role = "curator"

    def step(self):
        secrets_dir = os.path.expanduser("~/.uwc/devnet/canary")
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
    roots except one deliberately wrong stage, reveals whatever is asked, and never defends
    itself in a game. Whoever catches it wins a jackpot once the record is revealed."""
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
        active = [js for js in self.st["jobs"].values() if not js.get("final")]
        if len(active) < 1 and h - self.st.get("last_post", -10**9) >= self.a.every and self.a.post > len(self.st["jobs"]):
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

    def post_decoy(self, h: int):
        n = len(self.st["jobs"])
        seqs = wikitext_sequences(self.a.dataset, self.model["model_dir"], self.a.seq_len, self.a.batches,
                                  self.a.offset + n * self.a.batches)
        if not seqs:
            log("decoy: no text left at this offset")
            self.st["last_post"] = h
            return
        roots, embs = [], []
        for toks in seqs:
            r, e = self.rt.batch(toks)
            roots += r
            embs.append(e)
        wrong = random.randrange(len(roots))
        roots[wrong] = secrets.token_bytes(32)  # the forced error
        salts = [secrets.token_bytes(32) for _ in roots]
        root = input_root(seqs).hex()
        self.data.put(f"/inputs/{root}.json", json.dumps({"batches": seqs}).encode())
        try:
            r = self.chain.run("jobpost", os.path.expanduser(self.a.client_key), self.model_id, root,
                               str(len(seqs) * self.G), str(self.a.fee), str(self.a.deadline), str(self.a.k),
                               str(self.a.n), str(self.a.seq_len))
        except ChainError as e:
            log(f"decoy post: {e}")
            return
        jid = r["job"]
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
    ap.add_argument("--weights", default=os.path.join(DEVNET, "weights", "qwen3-1.7b"))
    # provider
    ap.add_argument("--max-jobs", type=int, default=1)
    ap.add_argument("--pool", action=argparse.BooleanOptionalAction, default=True)
    # verifier
    ap.add_argument("--stake", type=int, default=100_000_000)
    # watcher
    ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--bond", type=int, default=100_000)
    ap.add_argument("--lie", type=float, default=0.0, help="fault injection: share of checks posted with a wrong root")
    # client
    ap.add_argument("--post", type=int, default=0, help="jobs to post")
    ap.add_argument("--every", type=int, default=30, help="blocks between posts")
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--offset", type=int, default=1000, help="first WikiText chunk (the pool uses 0-199)")
    ap.add_argument("--fee", type=int, default=3_000_000_000)
    ap.add_argument("--deadline", type=int, default=60)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--dataset", default=os.path.expanduser("~/Documents/useful-work-coin/models/wikitext2"))
    a = ap.parse_args()
    {"provider": Provider, "verifier": Verifier, "watcher": Watcher, "client": Client, "decoy": Decoy,
     "curator": Curator}[a.role](a).run()


if __name__ == "__main__":
    main()
