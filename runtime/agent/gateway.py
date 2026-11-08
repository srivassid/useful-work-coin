"""Job gateway: an OpenAI-compatible embeddings API in front of the job market.

A request's texts are tokenized, padded to a length bucket (the dispute game assumes every
sequence of a job has exactly the job's seq_len) and posted as one job per bucket from the API
key's own chain account. The gateway returns the embeddings once the provider's outputs match
the output root it committed on chain, and keeps following each job to settlement:
  trust "committed"  the outputs match the provider's on-chain commitment (about 2 minutes)
  trust "verified"   the verifier committee has revealed its sampled stages
  trust "final"      the job settled after its challenge window
A job that expires without a provider is posted again with a higher fee; one that fails (its
provider lost a dispute) is posted again and its results are replaced.

  python -m agent.gateway --net /opt/uwc/net --data http://127.0.0.1:18692 --home /opt/uwc/gateway \\
      --model /opt/uwc/gateway/qwen3-1.7b.json --tokenizer /opt/uwc/gateway/tokenizer.json --bind 127.0.0.1:18693

Endpoints (Authorization: Bearer <api key>, except where noted):
  POST /v1/keys             create an API key and its chain account (no auth; --signup-code, faucet)
  GET  /v1/account          address, balances, spending
  GET  /v1/models           the model (no auth)
  POST /v1/embeddings       OpenAI embeddings request; waits for the outputs (--sync-timeout)
  POST /v1/chat/completions OpenAI chat request (a generation job, Rules 3); "stream": true sends
                            server-sent events from the provider's partial generations, then a
                            last chunk once the committed samples are checked (with their trust)
  POST /v1/jobs             the same request, answered at once with a request id
  GET  /v1/jobs, /v1/jobs/<id>   status, trust, jobs and (when done) the embeddings
  GET  /healthz             chain height (no auth)
Inputs are public on the data service: only send text that may be published.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import secrets
import sqlite3
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .chain import Chain, ChainError
from .data import Data
from .roots import SENTINEL, gen_output_root, head_output_root, input_root, output_root

COIN = 100_000_000
BUCKETS = (32, 64, 128, 256, 512)
MAX_INPUTS = 256
MAX_BODY = 4 << 20
STATUS_ORDER = ["posted", "accepted", "committed", "verify-commit", "verify-reveal", "provider-reveal",
                "challenge", "disputed", "settled"]
FINAL = ("settled", "expired", "failed", "aborted")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class ApiError(Exception):
    def __init__(self, status: int, message: str, code: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.message, self.code = status, message, code


def bucket(n: int, max_pos: int) -> int:
    for b in BUCKETS:
        if n <= b <= max_pos:
            return b
    raise ApiError(400, f"an input has {n} tokens; the most is {min(max_pos, BUCKETS[-1])}")


# ------------------------------------------------------------------ storage

SCHEMA = """
CREATE TABLE IF NOT EXISTS keys (id INTEGER PRIMARY KEY, key_hash TEXT UNIQUE, name TEXT, keyfile TEXT,
  account TEXT, address TEXT, created REAL, ip TEXT, spent INTEGER DEFAULT 0, cap INTEGER);
CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, key_id INTEGER, created REAL, status TEXT,
  n_inputs INTEGER, tokens INTEGER, fee INTEGER DEFAULT 0, error TEXT, done REAL);
CREATE TABLE IF NOT EXISTS jobs (job TEXT PRIMARY KEY, request TEXT, seq_len INTEGER, indexes TEXT,
  fee INTEGER, attempt INTEGER, posted_h INTEGER, chain_status TEXT, delivered INTEGER DEFAULT 0,
  replaced INTEGER DEFAULT 0, note TEXT);
"""


class Store:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(SCHEMA)
            cols = [r[1] for r in self.db.execute("PRAGMA table_info(requests)")]
            if "kind" not in cols:
                self.db.execute("ALTER TABLE requests ADD COLUMN kind TEXT DEFAULT 'embed'")

    def q(self, sql: str, *args) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def one(self, sql: str, *args) -> sqlite3.Row | None:
        r = self.q(sql, *args)
        return r[0] if r else None


# ------------------------------------------------------------------ the gateway


class Gateway:
    def __init__(self, a: argparse.Namespace):
        from tokenizers import Tokenizer
        self.a = a
        os.makedirs(a.home, mode=0o700, exist_ok=True)
        for sub in ("keys", "requests", "results"):
            os.makedirs(os.path.join(a.home, sub), mode=0o700, exist_ok=True)
        self.store = Store(os.path.join(a.home, "gateway.db"))
        self.chain = Chain(a.net)
        self.data_url = a.data
        info = json.load(open(a.model))
        self.model_id = json.load(open(a.model.replace(".json", ".reg.json")))["model"]
        self.model_name = a.model_name
        self.G = info["layer_groups"]
        self.max_pos = info["cfg"]["MaxPos"]
        self.dim = info["cfg"]["D"]
        self.tok = Tokenizer.from_file(a.tokenizer)
        self.pad = self.tok.token_to_id(a.pad_token)
        if self.pad is None:
            raise SystemExit(f"pad token {a.pad_token!r} is not in the tokenizer")
        self.gen_ok = bool(info.get("gen"))  # a generation model also serves chat completions
        self.stops = {i for i in (self.tok.token_to_id("<|im_end|>"), self.tok.token_to_id("<|endoftext|>")) if i is not None}
        self.faucet_lock = threading.Lock()
        self.wake = threading.Event()
        self.last_deposit = 0.0

    # -------------------------------------------------------------- keys and accounts

    def data_for(self, keyfile: str) -> Data:
        return Data(self.data_url, signer=lambda d: self.chain.run("sign", keyfile, d.hex()).strip())

    def create_key(self, name: str, ip: str, code: str | None) -> dict:
        if self.a.signup_code and code != self.a.signup_code:
            raise ApiError(403, "a valid signup_code is required", "permission_denied")
        day = time.time() - 86400
        if self.store.one("SELECT COUNT(*) n FROM keys WHERE ip = ? AND created > ?", ip, day)["n"] >= self.a.keys_per_ip:
            raise ApiError(429, f"at most {self.a.keys_per_ip} keys per address per day", "rate_limit_exceeded")
        api_key = "uwc_sk_" + secrets.token_hex(24)
        keyfile = os.path.join(self.a.home, "keys", f"{secrets.token_hex(8)}.json")
        self.chain.run("keygen", keyfile)
        os.chmod(keyfile, 0o600)
        info = self.chain.key_info(keyfile)
        self.store.q("INSERT INTO keys (key_hash, name, keyfile, account, address, created, ip, cap) VALUES (?,?,?,?,?,?,?,?)",
                     hashlib.sha256(api_key.encode()).hexdigest(), name[:80], keyfile, info["account"], info["address"],
                     time.time(), ip, int(self.a.spend_cap * COIN) if self.a.spend_cap else None)
        funded = 0
        if self.a.faucet_key and self.a.faucet_coins > 0:
            with self.faucet_lock:  # one faucet transaction at a time: they spend the same coins
                try:
                    # coins for transaction fees in the wallet, the rest straight into the jobs balance
                    self.chain.run("send", self.a.faucet_key, str(int(self.a.faucet_fee_coins * COIN)), info["address"])
                    self.chain.run("deposit", self.a.faucet_key, str(int(self.a.faucet_coins * COIN)), info["account"])
                    funded = self.a.faucet_coins
                except ChainError as e:
                    log(f"faucet: {e}")
        log(f"new key {name[:40]!r} -> {info['account'][:12]} (faucet {funded})")
        return {"api_key": api_key, "address": info["address"], "account": info["account"], "faucet_coins": funded,
                "note": "Keep the API key: it is shown once. Coins from the faucet arrive with the next block."}

    def auth(self, header: str | None) -> sqlite3.Row:
        if not header or not header.startswith("Bearer "):
            raise ApiError(401, "missing API key (Authorization: Bearer <key>)", "invalid_api_key")
        row = self.store.one("SELECT * FROM keys WHERE key_hash = ?", hashlib.sha256(header[7:].strip().encode()).hexdigest())
        if row is None:
            raise ApiError(401, "unknown API key", "invalid_api_key")
        return row

    def account(self, key: sqlite3.Row) -> dict:
        try:
            wallet = int(self.chain.run("coins", key["keyfile"])["total"])
            jobs = int(self.chain.account(key["account"]).get("balance") or 0)
        except ChainError as e:
            raise ApiError(503, f"the node is not answering: {e}", "service_unavailable")
        return {"address": key["address"], "account": key["account"], "wallet_coins": wallet / COIN,
                "jobs_balance_coins": jobs / COIN, "spent_coins": key["spent"] / COIN,
                "spend_cap_coins": key["cap"] / COIN if key["cap"] else None,
                "price_per_1k_tokens_coins": self.a.price_per_1k, "min_fee_coins": self.a.min_fee}

    # -------------------------------------------------------------- requests

    def tokenize(self, inp) -> list[list[int]]:
        if isinstance(inp, str):
            inp = [inp]
        if not isinstance(inp, list) or not inp:
            raise ApiError(400, "input must be a string, a list of strings or a list of token lists")
        if len(inp) > MAX_INPUTS:
            raise ApiError(400, f"at most {MAX_INPUTS} inputs per request")
        if all(isinstance(x, int) for x in inp):
            inp = [inp]  # one input given as tokens
        if all(isinstance(x, str) for x in inp):
            if any(not x for x in inp):
                raise ApiError(400, "inputs must not be empty")
            seqs = [e.ids for e in self.tok.encode_batch(inp, add_special_tokens=False)]
        elif all(isinstance(x, list) and x and all(isinstance(t, int) and 0 <= t < self.tok.get_vocab_size(True) for t in x)
                 for x in inp):
            seqs = inp
        else:
            raise ApiError(400, "input must be a string, a list of strings or a list of token lists")
        for s in seqs:
            if not s:
                raise ApiError(400, "an input has no tokens")
            bucket(len(s), self.max_pos)
        return seqs

    def submit(self, key: sqlite3.Row, body: dict) -> str:
        model = body.get("model", self.model_name)
        if model not in (self.model_name, self.model_id):
            raise ApiError(404, f"model {model!r} is not served here; use {self.model_name!r}", "model_not_found")
        seqs = self.tokenize(body.get("input"))
        padded = sum(bucket(len(s), self.max_pos) for s in seqs)
        fee = self.fee_for(padded)
        if key["cap"] is not None and key["spent"] + fee > key["cap"]:
            raise ApiError(402, "this key's spending cap is reached", "insufficient_quota")
        rid = secrets.token_hex(12)
        with open(os.path.join(self.a.home, "requests", f"{rid}.json"), "w") as f:
            json.dump({"seqs": seqs}, f)
        self.store.q("INSERT INTO requests (id, key_id, created, status, n_inputs, tokens) VALUES (?,?,?,?,?,?)",
                     rid, key["id"], time.time(), "pending", len(seqs), sum(len(s) for s in seqs))
        self.wake.set()
        return rid

    def chat_prompt(self, messages, think: bool) -> list[int]:
        """Qwen3's chat format; without thinking, the empty think block its template adds."""
        if not isinstance(messages, list) or not messages:
            raise ApiError(400, "messages must be a non-empty list")
        text = ""
        for m in messages:
            if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), str):
                raise ApiError(400, "each message needs a role (system, user, assistant) and string content")
            text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        text += "<|im_start|>assistant\n" + ("" if think else "<think>\n\n</think>\n\n")
        return self.tok.encode(text, add_special_tokens=False).ids

    def submit_chat(self, key: sqlite3.Row, body: dict) -> str:
        if not self.gen_ok:
            raise ApiError(404, "this gateway's model does not generate", "model_not_found")
        model = body.get("model", self.model_name)
        if model not in (self.model_name, self.model_id):
            raise ApiError(404, f"model {model!r} is not served here; use {self.model_name!r}", "model_not_found")
        prompt = self.chat_prompt(body.get("messages"), bool(body.get("uwc_think")))
        max_new = body.get("max_completion_tokens", body.get("max_tokens", self.a.max_new))
        if not isinstance(max_new, int) or max_new < 1:
            raise ApiError(400, "max_tokens must be a positive integer")
        T = len(prompt) + max_new
        if T > self.max_pos:
            raise ApiError(400, f"prompt ({len(prompt)} tokens) + max_tokens ({max_new}) exceed the model's {self.max_pos}")
        temp = body.get("temperature", self.a.temperature)
        if not isinstance(temp, (int, float)) or not 0 <= temp <= 4:
            raise ApiError(400, "temperature must be between 0 and 4")
        fee = self.fee_for(T)
        if key["cap"] is not None and key["spent"] + fee > key["cap"]:
            raise ApiError(402, "this key's spending cap is reached", "insufficient_quota")
        rid = secrets.token_hex(12)
        with open(os.path.join(self.a.home, "requests", f"{rid}.json"), "w") as f:
            json.dump({"kind": "chat", "prompt": prompt, "T": T, "temp_milli": int(round(temp * 1000))}, f)
        self.store.q("INSERT INTO requests (id, key_id, created, status, n_inputs, tokens, kind) VALUES (?,?,?,?,?,?,?)",
                     rid, key["id"], time.time(), "pending", 1, len(prompt), "chat")
        self.wake.set()
        return rid

    def chat_text(self, prompt_len: int, toks: list[int]) -> tuple[str, str, int]:
        """(text, finish reason, completion tokens) from the generated tokens, cut at a stop token."""
        n = len(toks)
        for i, t in enumerate(toks):
            if t in self.stops:
                return self.tok.decode(toks[:i], skip_special_tokens=True), "stop", i + 1
        return self.tok.decode(toks, skip_special_tokens=True), "length", n

    def chat_view(self, rid: str, r: sqlite3.Row) -> dict:
        req = json.load(open(os.path.join(self.a.home, "requests", f"{rid}.json")))
        out = {"id": f"chatcmpl-{rid}", "object": "chat.completion", "created": int(r["created"]), "model": self.model_name}
        if r["status"] == "done":
            toks = json.load(open(os.path.join(self.a.home, "results", f"{rid}.json")))["tokens"]
            text, finish, n = self.chat_text(len(req["prompt"]), toks)
            out["choices"] = [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}]
            out["usage"] = {"prompt_tokens": len(req["prompt"]), "completion_tokens": n, "total_tokens": len(req["prompt"]) + n}
        return out

    def fee_for(self, tokens: int) -> int:
        return max(int(self.a.min_fee * COIN), math.ceil(self.a.price_per_1k * COIN * tokens / 1000))

    def trust(self, rid: str) -> str:
        js = [j for j in self.store.q("SELECT * FROM jobs WHERE request = ? AND replaced = 0", rid)]
        if not js or not all(j["delivered"] for j in js):
            return "none"
        st = [j["chain_status"] or "posted" for j in js]
        if any(s in ("failed", "aborted") for s in st):
            return "failed"
        if any(s == "disputed" for s in st):
            return "disputed"
        if all(s == "settled" for s in st):
            return "final"
        if all(s in ("provider-reveal", "challenge", "settled") for s in st):
            return "verified"
        return "committed"

    def view(self, rid: str, key: sqlite3.Row, fmt: str = "float") -> dict:
        r = self.store.one("SELECT * FROM requests WHERE id = ? AND key_id = ?", rid, key["id"])
        if r is None:
            raise ApiError(404, "no such request", "not_found")
        jobs = [{"job": j["job"], "seq_len": j["seq_len"], "inputs": len(json.loads(j["indexes"])), "fee_coins": j["fee"] / COIN,
                 "attempt": j["attempt"], "status": j["chain_status"] or "posted", "delivered": bool(j["delivered"]),
                 **({"note": j["note"]} if j["note"] else {})}
                for j in self.store.q("SELECT * FROM jobs WHERE request = ? AND replaced = 0", rid)]
        out = {"id": rid, "object": "uwc.request", "status": r["status"], "trust": self.trust(rid), "model": self.model_name,
               "inputs": r["n_inputs"], "fee_coins": r["fee"] / COIN, "jobs": jobs, "created": int(r["created"])}
        if r["error"]:
            out["error"] = r["error"]
        if r["kind"] == "chat":
            out.update({k: v for k, v in self.chat_view(rid, r).items() if k in ("choices", "usage")})
        elif r["status"] == "done":
            out["data"] = self.embeddings(rid, fmt)
            out["usage"] = {"prompt_tokens": r["tokens"], "total_tokens": r["tokens"]}
        return out

    def embeddings(self, rid: str, fmt: str) -> list[dict]:
        embs = np.load(os.path.join(self.a.home, "results", f"{rid}.npy"))
        if fmt == "base64":
            return [{"object": "embedding", "index": i, "embedding": base64.b64encode(e.astype("<f4").tobytes()).decode()}
                    for i, e in enumerate(embs)]
        return [{"object": "embedding", "index": i, "embedding": [float(x) for x in e]} for i, e in enumerate(embs)]

    # -------------------------------------------------------------- the worker

    def run_worker(self):
        while True:
            try:
                self.tick()
            except Exception:
                traceback.print_exc()
            self.wake.wait(self.a.poll)
            self.wake.clear()

    def tick(self):
        h = self.chain.height()
        for r in self.store.q("SELECT * FROM requests WHERE status = 'pending'"):
            self.post_request(r, h)
        for j in self.store.q("SELECT * FROM jobs WHERE replaced = 0 AND (chain_status IS NULL OR chain_status NOT IN "
                              "('settled', 'expired', 'failed', 'aborted'))"):
            self.follow(j, h)
        if time.time() - self.last_deposit > 300:
            self.last_deposit = time.time()
            self.auto_deposit()

    def post_request(self, r: sqlite3.Row, h: int):
        """Post a pending request's jobs: one per length bucket, at most --max-batch sequences each."""
        key = self.store.one("SELECT * FROM keys WHERE id = ?", r["key_id"])
        req = json.load(open(os.path.join(self.a.home, "requests", f"{r['id']}.json")))
        if req.get("kind") == "chat":
            seqs = [req["prompt"]]
            parts = [(req["T"], [0])]
        else:
            seqs = req["seqs"]
            groups: dict[int, list[int]] = {}
            for i, s in enumerate(seqs):
                groups.setdefault(bucket(len(s), self.max_pos), []).append(i)
            parts = [(L, idx[k:k + self.a.max_batch]) for L, idx in sorted(groups.items())
                     for k in range(0, len(idx), self.a.max_batch)]
        fees = [self.fee_for(L * len(idx)) for L, idx in parts]
        try:
            balance = int(self.chain.account(key["account"]).get("balance") or 0)
        except ChainError:
            return
        pending = sum(x["fee"] for x in self.store.q(
            "SELECT j.fee FROM jobs j JOIN requests q ON q.id = j.request WHERE q.key_id = ? AND j.replaced = 0 "
            "AND (j.chain_status IS NULL OR j.chain_status = 'posted') AND j.posted_h >= ?", key["id"], h - 3))
        if balance - pending < sum(fees):
            if time.time() - r["created"] > self.a.give_up:
                self.store.q("UPDATE requests SET status = 'failed', error = ? WHERE id = ?",
                             f"the jobs balance ({balance / COIN} coins) cannot pay {sum(fees) / COIN} coins; send coins "
                             f"to {key['address']}", r["id"])
            else:
                self.store.q("UPDATE requests SET error = ? WHERE id = ?", "waiting for the jobs balance", r["id"])
            return
        data = self.data_for(key["keyfile"])
        for (L, idx), fee in zip(parts, fees):
            if self.store.one("SELECT 1 FROM jobs WHERE request = ? AND indexes = ? AND replaced = 0", r["id"], json.dumps(idx)):
                continue  # posted in an earlier tick
            if not self.post_job(r["id"], key, data, seqs, L, idx, fee, 1, h, req.get("temp_milli")):
                if time.time() - r["created"] > self.a.give_up:
                    self.store.q("UPDATE requests SET status = 'failed' WHERE id = ?", r["id"])
                return
        self.store.q("UPDATE requests SET status = 'posted', error = NULL, fee = ? WHERE id = ?", sum(fees), r["id"])
        self.store.q("UPDATE keys SET spent = spent + ? WHERE id = ?", sum(fees), key["id"])

    def post_job(self, rid: str, key, data: Data, seqs, L: int, idx: list[int], fee: int, attempt: int, h: int,
                 temp_milli: int | None = None) -> bool:
        """One job: embedding batches padded to L, or (temp_milli set) a generation job whose
        input rows are the prompts followed by the sentinel up to L."""
        if temp_milli is not None:
            batches = [seqs[i] + [SENTINEL] * (L - len(seqs[i])) for i in idx]
            cmd = ["jobgen", key["keyfile"], self.model_id, input_root(batches).hex(), str(len(batches) * (self.G + 1)),
                   str(fee), str(self.a.deadline), str(self.a.k), str(self.a.n), str(L), str(temp_milli)]
        else:
            batches = [seqs[i] + [self.pad] * (L - len(seqs[i])) for i in idx]
            cmd = ["jobpost", key["keyfile"], self.model_id, input_root(batches).hex(), str(len(batches) * self.G), str(fee),
                   str(self.a.deadline), str(self.a.k), str(self.a.n), str(L)]
        root = input_root(batches).hex()
        try:
            data.put(f"/inputs/{root}.json", json.dumps({"batches": batches}).encode())
            r = self.chain.run(*cmd)
        except Exception as e:
            self.store.q("UPDATE requests SET error = ? WHERE id = ?", f"posting: {e}"[:300], rid)
            log(f"request {rid[:8]}: posting failed: {e}")
            return False
        self.store.q("INSERT INTO jobs (job, request, seq_len, indexes, fee, attempt, posted_h) VALUES (?,?,?,?,?,?,?)",
                     r["job"], rid, L, json.dumps(idx), fee, attempt, h)
        log(f"request {rid[:8]}: job {r['job'][:8]} {len(batches)}x{L} tokens, fee {fee / COIN} (attempt {attempt})")
        return True

    def follow(self, j: sqlite3.Row, h: int):
        jid = j["job"]
        try:
            cj = self.chain.job(jid)
        except ChainError:
            return
        if cj is None:
            if h - j["posted_h"] > 10:  # the post never reached a block, or failed inside it
                self.store.q("UPDATE jobs SET chain_status = 'aborted', note = 'the post did not reach the chain' WHERE job = ?", jid)
                self.retry(j, h, same_fee=True)
            return
        st = cj["status"]
        if st != j["chain_status"]:
            self.store.q("UPDATE jobs SET chain_status = ? WHERE job = ?", st, jid)
        if not j["delivered"] and cj.get("outputroot", "0" * 64) != "0" * 64:
            raw = Data(self.data_url).get(f"/outputs/{jid}.npy")
            if raw is not None and cj.get("gen"):
                self.deliver_gen(j, cj, np.load(io.BytesIO(raw)))
            elif raw is not None:
                embs = np.load(io.BytesIO(raw))
                n = len(json.loads(j["indexes"]))
                if embs.shape == (n, self.dim) and output_root(list(embs)).hex() == cj["outputroot"]:
                    np.save(os.path.join(self.a.home, "results", f"{jid}.npy"), embs.astype(np.float32))
                    self.store.q("UPDATE jobs SET delivered = 1, note = NULL WHERE job = ?", jid)
                    log(f"job {jid[:8]}: outputs match the committed output root")
                    self.assemble(j["request"])
                else:
                    self.store.q("UPDATE jobs SET note = ? WHERE job = ?", "the delivered outputs do not match the commitment", jid)
        if st == "expired" or (st in ("failed", "aborted")):
            self.retry(j, h, same_fee=st != "expired")

    def deliver_gen(self, j: sqlite3.Row, cj: dict, heads: np.ndarray):
        """A generation job's samples: checked against the committed output root, then the
        generated tokens (positions P..T-1) become the request's result."""
        jid, rid = j["job"], j["request"]
        T = j["seq_len"]
        if heads.ndim != 3 or heads.shape[1:] != (T - 1, 2) or \
                gen_output_root([head_output_root(np.ascontiguousarray(h.astype(np.int32))) for h in heads]).hex() != cj["outputroot"]:
            self.store.q("UPDATE jobs SET note = ? WHERE job = ?", "the delivered samples do not match the commitment", jid)
            return
        req = json.load(open(os.path.join(self.a.home, "requests", f"{rid}.json")))
        P = len(req["prompt"])
        toks = [int(x) for x in heads[0, P - 1:, 0]]
        json.dump({"tokens": toks, "job": jid}, open(os.path.join(self.a.home, "results", f"{rid}.json"), "w"))
        self.store.q("UPDATE jobs SET delivered = 1, note = NULL WHERE job = ?", jid)
        self.store.q("UPDATE requests SET status = 'done', done = ? WHERE id = ?", time.time(), rid)
        log(f"request {rid[:8]}: generation job {jid[:8]} samples match the committed output root ({len(toks)} tokens)")

    def retry(self, j: sqlite3.Row, h: int, same_fee: bool):
        """Post a job again: expired (no provider took it) with a higher fee; failed (its provider
        lost a dispute) or lost as it was. Its earlier results stop counting."""
        r = self.store.one("SELECT * FROM requests WHERE id = ?", j["request"])
        key = self.store.one("SELECT * FROM keys WHERE id = ?", r["key_id"])
        self.store.q("UPDATE jobs SET replaced = 1 WHERE job = ?", j["job"])
        if j["attempt"] >= self.a.attempts:
            self.store.q("UPDATE requests SET status = 'failed', error = ? WHERE id = ?",
                         f"job {j['job'][:8]} ended {j['chain_status'] or 'lost'} after {j['attempt']} attempts", r["id"])
            return
        if r["status"] == "done":
            self.store.q("UPDATE requests SET status = 'posted' WHERE id = ?", r["id"])  # results are being redone
        req = json.load(open(os.path.join(self.a.home, "requests", f"{r['id']}.json")))
        seqs = [req["prompt"]] if req.get("kind") == "chat" else req["seqs"]
        fee = j["fee"] if same_fee else math.ceil(j["fee"] * self.a.fee_bump)
        if not self.post_job(r["id"], key, self.data_for(key["keyfile"]), seqs, j["seq_len"], json.loads(j["indexes"]),
                             fee, j["attempt"] + 1, h, req.get("temp_milli")):
            self.store.q("UPDATE jobs SET replaced = 0 WHERE job = ?", j["job"])  # try again next tick
            return
        # fees posted, gross: an expired or failed job's fee comes back to the jobs balance on chain
        self.store.q("UPDATE keys SET spent = spent + ? WHERE id = ?", fee, key["id"])
        self.store.q("UPDATE requests SET fee = fee + ? WHERE id = ?", fee, r["id"])

    def assemble(self, rid: str):
        js = self.store.q("SELECT * FROM jobs WHERE request = ? AND replaced = 0", rid)
        r = self.store.one("SELECT * FROM requests WHERE id = ?", rid)
        if not js or not all(j["delivered"] for j in js):
            return
        out = np.zeros((r["n_inputs"], self.dim), dtype=np.float32)
        for j in js:
            embs = np.load(os.path.join(self.a.home, "results", f"{j['job']}.npy"))
            for row, i in zip(embs, json.loads(j["indexes"])):
                out[i] = row
        np.save(os.path.join(self.a.home, "results", f"{rid}.npy"), out)
        self.store.q("UPDATE requests SET status = 'done', done = ? WHERE id = ?", time.time(), rid)
        log(f"request {rid[:8]}: done ({r['n_inputs']} embeddings)")

    def auto_deposit(self):
        """Coins sent to a key's address go into its jobs balance (a coin stays for transaction fees)."""
        for key in self.store.q("SELECT * FROM keys"):
            try:
                total = int(self.chain.run("coins", key["keyfile"])["total"])
                if total > 2 * COIN:
                    self.chain.run("deposit", key["keyfile"], str(total - COIN))
                    log(f"key {key['account'][:12]}: deposited {(total - COIN) / COIN} coins")
            except ChainError as e:
                log(f"deposit {key['account'][:12]}: {e}")


# ------------------------------------------------------------------ HTTP


def handler(gw: Gateway):
    class H(BaseHTTPRequestHandler):
        timeout = 60
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def send_json(self, status: int, obj):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.cors()
            self.end_headers()
            self.wfile.write(body)

        def cors(self):
            if gw.a.cors_origin:
                self.send_header("Access-Control-Allow-Origin", gw.a.cors_origin)
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ApiError(413, f"requests are at most {MAX_BODY} bytes")
            if n <= 0:
                return {}
            try:
                b = json.loads(self.rfile.read(n))
            except ValueError:
                raise ApiError(400, "the body must be JSON")
            if not isinstance(b, dict):
                raise ApiError(400, "the body must be a JSON object")
            return b

        def client_ip(self) -> str:
            return (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[0].strip()

        def do_OPTIONS(self):
            self.send_response(204)
            self.cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def handle_api(self, method: str):
            path = self.path.split("?")[0].rstrip("/")
            query = dict(p.split("=", 1) for p in self.path.split("?", 1)[1].split("&") if "=" in p) if "?" in self.path else {}
            if method == "GET" and path == "/healthz":
                return 200, {"ok": True, "height": gw.chain.height()}
            if method == "GET" and path == "/v1/models":
                return 200, {"object": "list", "data": [{"id": gw.model_name, "object": "model", "owned_by": "uwc-testnet",
                                                         "uwc_model": gw.model_id, "dimensions": gw.dim,
                                                         "max_tokens": min(gw.max_pos, BUCKETS[-1]),
                                                         "chat": gw.gen_ok, "context": gw.max_pos}]}
            if method == "POST" and path == "/v1/keys":
                b = self.body()
                return 201, gw.create_key(str(b.get("name") or "unnamed"), self.client_ip(), b.get("signup_code"))
            key = gw.auth(self.headers.get("Authorization"))
            if method == "GET" and path == "/v1/account":
                return 200, gw.account(key)
            if method == "POST" and path in ("/v1/embeddings", "/v1/jobs"):
                b = self.body()
                fmt = b.get("encoding_format") or "float"
                if fmt not in ("float", "base64"):
                    raise ApiError(400, "encoding_format must be float or base64")
                rid = gw.submit(key, b)
                if path == "/v1/jobs":
                    return 202, gw.view(rid, key)
                deadline = time.time() + gw.a.sync_timeout
                while time.time() < deadline:
                    r = gw.store.one("SELECT status FROM requests WHERE id = ?", rid)
                    if r["status"] in ("done", "failed"):
                        break
                    time.sleep(1)
                v = gw.view(rid, key, fmt)
                if v["status"] == "done":
                    return 200, {"object": "list", "data": v["data"], "model": gw.model_name, "usage": v["usage"],
                                 "uwc": {k: v[k] for k in ("id", "trust", "fee_coins", "jobs")}}
                if v["status"] == "failed":
                    raise ApiError(502, f"the request failed: {v.get('error')}", "job_failed")
                raise ApiError(504, f"still running; poll GET /v1/jobs/{rid}", "timeout")
            if method == "POST" and path == "/v1/chat/completions":
                b = self.body()
                rid = gw.submit_chat(key, b)
                if b.get("stream"):
                    self.stream_chat(rid, key)
                    return None, None
                deadline = time.time() + gw.a.sync_timeout
                while time.time() < deadline:
                    r = gw.store.one("SELECT status FROM requests WHERE id = ?", rid)
                    if r["status"] in ("done", "failed"):
                        break
                    time.sleep(1)
                r = gw.store.one("SELECT * FROM requests WHERE id = ?", rid)
                if r["status"] == "done":
                    v = gw.chat_view(rid, r)
                    v["uwc"] = {k: x for k, x in gw.view(rid, key).items() if k in ("id", "trust", "fee_coins", "jobs")}
                    return 200, v
                if r["status"] == "failed":
                    raise ApiError(502, f"the request failed: {r['error']}", "job_failed")
                raise ApiError(504, f"still running; poll GET /v1/jobs/{rid}", "timeout")
            if method == "GET" and path == "/v1/jobs":
                rows = gw.store.q("SELECT id FROM requests WHERE key_id = ? ORDER BY created DESC LIMIT 50", key["id"])
                return 200, {"object": "list", "data": [{k: v for k, v in gw.view(r["id"], key).items() if k != "data"}
                                                        for r in rows]}
            if method == "GET" and path.startswith("/v1/jobs/"):
                return 200, gw.view(path.rsplit("/", 1)[1], key, query.get("encoding_format", "float"))
            raise ApiError(404, f"no route {method} {path}", "not_found")

        def sse(self, obj) -> bool:
            try:
                self.wfile.write(b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n")
                self.wfile.flush()
                return True
            except OSError:
                return False

        def stream_chat(self, rid: str, key):
            """Server-sent events: deltas from the provider's partial generation (unverified while
            they stream), then the checked text's remainder, the finish chunk with its trust, [DONE]."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.cors()
            self.end_headers()
            self.close_connection = True
            req = json.load(open(os.path.join(gw.a.home, "requests", f"{rid}.json")))
            P = len(req["prompt"])
            base = {"id": f"chatcmpl-{rid}", "object": "chat.completion.chunk", "model": gw.model_name}
            chunk = lambda delta, finish=None, **kw: {**base, "created": int(time.time()),
                                                      "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **kw}
            if not self.sse(chunk({"role": "assistant", "content": ""})):
                return
            sent, job, deadline = "", None, time.time() + gw.a.sync_timeout
            while time.time() < deadline:
                r = gw.store.one("SELECT * FROM requests WHERE id = ?", rid)
                if r["status"] == "failed":
                    self.sse({"error": {"message": f"the request failed: {r['error']}", "type": "job_failed"}})
                    return
                if r["status"] == "done":
                    toks = json.load(open(os.path.join(gw.a.home, "results", f"{rid}.json")))["tokens"]
                    text, finish, n = gw.chat_text(P, toks)
                    if not text.startswith(sent):  # the streamed tokens were not the committed ones
                        self.sse({"error": {"message": "the provider's streamed text differs from its committed samples; "
                                                       "use the final message", "type": "stream_mismatch"}})
                        self.sse(chunk({"content": text}))
                    elif text[len(sent):]:
                        self.sse(chunk({"content": text[len(sent):]}))
                    v = gw.view(rid, key)
                    self.sse(chunk({}, finish, usage={"prompt_tokens": P, "completion_tokens": n, "total_tokens": P + n},
                                   uwc={"id": rid, "trust": v["trust"], "fee_coins": v["fee_coins"], "jobs": v["jobs"]}))
                    self.sse(b"[DONE]")
                    return
                j = gw.store.one("SELECT job FROM jobs WHERE request = ? AND replaced = 0", rid)
                if j and j["job"] != job:
                    job = j["job"]
                if job:
                    raw = Data(gw.data_url).get(f"/stream/{job}.json")
                    if raw:
                        try:
                            toks = json.loads(raw)["tokens"][0]
                            text, finish, _ = gw.chat_text(P, toks)
                            if text.startswith(sent) and len(text) > len(sent) and not text.endswith("\ufffd"):
                                if not self.sse(chunk({"content": text[len(sent):]})):
                                    return
                                sent = text
                        except (ValueError, KeyError, IndexError, TypeError):
                            pass
                time.sleep(0.4)
            self.sse({"error": {"message": f"still running; poll GET /v1/jobs/{rid}", "type": "timeout"}})

        def run(self, method: str):
            try:
                status, obj = self.handle_api(method)
                if status is None:
                    return  # streamed
                self.send_json(status, obj)
            except ApiError as e:
                self.send_json(e.status, {"error": {"message": e.message, "type": e.code, "code": e.code}})
            except ChainError as e:
                self.send_json(503, {"error": {"message": f"the node refused: {e}", "type": "service_unavailable"}})
            except Exception as e:
                traceback.print_exc()
                self.send_json(500, {"error": {"message": f"internal error: {type(e).__name__}", "type": "server_error"}})

        def do_GET(self):
            self.run("GET")

        def do_POST(self):
            self.run("POST")
    return H


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenAI-compatible embeddings gateway for the UWC job market")
    ap.add_argument("--net", required=True, help="network directory of a local node (env, bin/uwcctl)")
    ap.add_argument("--data", required=True, help="data service URL")
    ap.add_argument("--home", required=True, help="the gateway's own directory (database, keys, results)")
    ap.add_argument("--model", required=True, help="model json (with its .reg.json beside it)")
    ap.add_argument("--model-name", default="qwen3-1.7b")
    ap.add_argument("--tokenizer", required=True, help="the model's tokenizer.json")
    ap.add_argument("--pad-token", default="<|endoftext|>")
    ap.add_argument("--bind", default="127.0.0.1:18693")
    ap.add_argument("--price-per-1k", type=float, default=1.5, help="coins per 1,000 (padded) tokens")
    ap.add_argument("--min-fee", type=float, default=0.5, help="coins per job at least")
    ap.add_argument("--deadline", type=int, default=60, help="blocks for a provider to accept and commit")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-batch", type=int, default=64, help="sequences per job")
    ap.add_argument("--attempts", type=int, default=3, help="posts per job before the request fails")
    ap.add_argument("--fee-bump", type=float, default=1.5, help="fee multiplier when a job expires untaken")
    ap.add_argument("--give-up", type=float, default=1800, help="seconds a request may wait for balance or a post")
    ap.add_argument("--sync-timeout", type=float, default=600, help="seconds POST /v1/embeddings waits")
    ap.add_argument("--poll", type=float, default=5)
    ap.add_argument("--signup-code", default=os.environ.get("UWC_GATEWAY_SIGNUP_CODE"), help="required to create keys")
    ap.add_argument("--keys-per-ip", type=int, default=3)
    ap.add_argument("--spend-cap", type=float, default=0, help="coins a key may spend (0: no cap)")
    ap.add_argument("--faucet-key", default=None, help="key that funds new API keys (testnet)")
    ap.add_argument("--faucet-coins", type=float, default=100, help="coins into a new key's jobs balance")
    ap.add_argument("--faucet-fee-coins", type=float, default=2, help="coins into a new key's wallet for transaction fees")
    ap.add_argument("--cors-origin", default=None, help="Access-Control-Allow-Origin for a browser UI")
    ap.add_argument("--max-new", type=int, default=256, help="chat: default max_tokens")
    ap.add_argument("--temperature", type=float, default=0.7, help="chat: default temperature")
    a = ap.parse_args()
    gw = Gateway(a)
    threading.Thread(target=gw.run_worker, daemon=True).start()
    host, port = a.bind.rsplit(":", 1)
    srv = ThreadingHTTPServer((host, int(port)), handler(gw))
    srv.daemon_threads = True
    log(f"gateway for {a.model_name} ({gw.model_id[:12]}) on {a.bind}, data {a.data}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
