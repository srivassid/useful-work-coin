"""Job gateway: an OpenAI-compatible API (and a web page) in front of the job market.

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
      --models /opt/uwc/gateway/models.json --bind 127.0.0.1:18693
  (or one model: --model qwen3-1.7b-gen.json --tokenizer tokenizer.json --model-name qwen3-1.7b)

--models is a JSON list, one entry per served model:
  {"name": "llama-3.2-3b-instruct", "model": "<model json, .reg.json beside it>", "tokenizer": "<tokenizer.json>",
   "tokenizer_config": "<tokenizer_config.json: its chat_template>", "template": "qwen3" | "hf" | "none",
   "price_per_1k": 2.5, "embeddings": false, "pad_token": "<|endoftext|>", "label": "Llama 3.2 3B Instruct"}
"template" is how chat messages become a prompt: Qwen3's format, the model's own Jinja chat template, or
"none" for a base model (completions only).

Endpoints (Authorization: Bearer <api key>, except where noted):
  POST /v1/keys             create an API key and its chain account (no auth; --signup-code, faucet)
  GET  /v1/account          address, balances, spending
  GET  /v1/models           the served models (no auth)
  POST /v1/embeddings       OpenAI embeddings request; waits for the outputs (--sync-timeout)
  POST /v1/chat/completions OpenAI chat request (a generation job, Rules 3); "stream": true sends
                            server-sent events from the provider's partial generations, then a
                            last chunk once the committed samples are checked (with their trust)
  POST /v1/completions      OpenAI text completion (any generation model, base models included)
  POST /v1/jobs             an embeddings request, answered at once with a request id
  POST /v1/batches          many chat, completion or embedding inputs as one request (answered at once)
  GET  /v1/jobs, /v1/jobs/<id>   status, trust, jobs and (when done) the results
  GET  /v1/jobs/<id>/results     a finished request's results as JSON lines (a download)
  GET  /healthz             chain height (no auth)
  GET  /                    the web page: sign up, chat, embeddings, batches, your jobs
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
PAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gateway_page.html")
PAGE = open(PAGE_FILE).read() if os.path.exists(PAGE_FILE) else "<!doctype html><title>UWC gateway</title><p>No page installed."


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


# ------------------------------------------------------------------ served models

STOP_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "</s>")


class Served:
    """One model the gateway posts jobs for: its chain id, shape, tokenizer and chat format."""

    def __init__(self, spec: dict, defaults: argparse.Namespace):
        from tokenizers import Tokenizer
        self.name = spec["name"]
        self.label = spec.get("label") or self.name
        info = json.load(open(spec["model"]))
        self.model_id = json.load(open(spec["model"].replace(".json", ".reg.json")))["model"]
        self.G = info["layer_groups"]
        self.max_pos = info["cfg"]["MaxPos"]
        self.dim = info["cfg"]["D"]
        self.gen_ok = bool(info.get("gen"))  # a generation model also serves chat and completions
        self.tok = Tokenizer.from_file(spec["tokenizer"])
        self.price_per_1k = float(spec.get("price_per_1k", defaults.price_per_1k))
        self.template = spec.get("template", "qwen3")
        cfg = json.load(open(spec["tokenizer_config"])) if spec.get("tokenizer_config") else {}
        tokname = lambda t: t.get("content") if isinstance(t, dict) else t  # noqa: E731
        self.bos, self.eos = tokname(cfg.get("bos_token")) or "", tokname(cfg.get("eos_token")) or ""
        self.chat_template = cfg.get("chat_template") if self.template == "hf" else None
        if self.template == "hf" and not self.chat_template:
            raise SystemExit(f"{self.name}: template 'hf' needs a tokenizer_config with a chat_template")
        if self.template == "none":  # a base model ends its text with its own end token only
            stops = [self.eos] if self.eos else list(STOP_TOKENS)
        elif self.template == "qwen3":
            stops = ["<|im_end|>", "<|endoftext|>"]
        else:
            stops = list(STOP_TOKENS) + ([self.eos] if self.eos else [])
        self.stops = {i for i in (self.tok.token_to_id(t) for t in stops) if i is not None}
        pad = spec.get("pad_token", defaults.pad_token if self.template == "qwen3" else None)
        self.pad = self.tok.token_to_id(pad) if pad else None
        self.embeddings = bool(spec.get("embeddings", self.template == "qwen3")) and self.pad is not None
        if spec.get("embeddings") and self.pad is None:
            raise SystemExit(f"{self.name}: embeddings need a pad_token that is in the tokenizer")

    def describe(self) -> dict:
        return {"id": self.name, "object": "model", "owned_by": "uwc", "uwc_model": self.model_id,
                "label": self.label, "dimensions": self.dim, "max_tokens": min(self.max_pos, BUCKETS[-1]),
                "context": self.max_pos, "chat": self.gen_ok and self.template != "none",
                "completions": self.gen_ok, "embeddings": self.embeddings, "price_per_1k_tokens_coins": self.price_per_1k}

    def chat_prompt(self, messages, think: bool) -> list[int]:
        if not isinstance(messages, list) or not messages:
            raise ApiError(400, "messages must be a non-empty list")
        for m in messages:
            if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), str):
                raise ApiError(400, "each message needs a role (system, user, assistant) and string content")
        if self.template == "none":
            raise ApiError(400, f"{self.name} is a base model without a chat format: use /v1/completions", "model_not_chat")
        if self.template == "qwen3":
            # Qwen3's chat format; without thinking, the empty think block its template adds
            text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
            text += "<|im_start|>assistant\n" + ("" if think else "<think>\n\n</think>\n\n")
        else:
            text = render_chat_template(self.chat_template, [{"role": m["role"], "content": m["content"]} for m in messages],
                                        self.bos, self.eos)
        return self.tok.encode(text, add_special_tokens=False).ids

    def completion_prompt(self, prompt) -> list[int]:
        if not isinstance(prompt, str) or not prompt:
            raise ApiError(400, "prompt must be a non-empty string")
        return self.tok.encode(self.bos + prompt, add_special_tokens=False).ids

    def text(self, toks: list[int]) -> tuple[str, str, int]:
        """(text, finish reason, completion tokens) from the generated tokens, cut at a stop token."""
        for i, t in enumerate(toks):
            if t in self.stops:
                return self.tok.decode(toks[:i], skip_special_tokens=True), "stop", i + 1
        return self.tok.decode(toks, skip_special_tokens=True), "length", len(toks)


_templates: dict[str, object] = {}


def render_chat_template(template: str, messages: list[dict], bos: str, eos: str) -> str:
    """A Hugging Face chat template, rendered the way transformers' apply_chat_template does
    (sandboxed Jinja, trim_blocks and lstrip_blocks, raise_exception and strftime_now)."""
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    t = _templates.get(template)
    if t is None:
        def raise_exception(msg):
            raise ApiError(400, f"the model's chat template refused the messages: {msg}")
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
        env.filters["tojson"] = lambda x, indent=None, **kw: json.dumps(x, indent=indent, ensure_ascii=False)
        env.globals.update(raise_exception=raise_exception, strftime_now=lambda fmt: time.strftime(fmt))
        t = _templates[template] = env.from_string(template)
    return t.render(messages=messages, bos_token=bos, eos_token=eos, add_generation_prompt=True)


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
            if "model" not in cols:  # requests from before several models were served: the first one
                self.db.execute("ALTER TABLE requests ADD COLUMN model TEXT")

    def q(self, sql: str, *args) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def one(self, sql: str, *args) -> sqlite3.Row | None:
        r = self.q(sql, *args)
        return r[0] if r else None


# ------------------------------------------------------------------ the gateway


class Gateway:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        os.makedirs(a.home, mode=0o700, exist_ok=True)
        for sub in ("keys", "requests", "results"):
            os.makedirs(os.path.join(a.home, sub), mode=0o700, exist_ok=True)
        self.store = Store(os.path.join(a.home, "gateway.db"))
        self.chain = Chain(a.net)
        self.data_url = a.data
        if getattr(a, "models", None):
            specs = json.load(open(a.models))
        else:  # one model from the command line, as before --models
            specs = [{"name": a.model_name, "model": a.model, "tokenizer": a.tokenizer, "template": "qwen3",
                      "pad_token": a.pad_token, "embeddings": True}]
        self.models = {}
        for spec in specs:
            m = Served(spec, a)
            if m.name in self.models:
                raise SystemExit(f"model {m.name!r} is listed twice")
            self.models[m.name] = m
        self.default = next(iter(self.models.values()))
        if self.default.template == "qwen3" and self.default.pad is None:
            raise SystemExit(f"pad token {a.pad_token!r} is not in the tokenizer")
        # the first model under the names the gateway had when it served one
        d = self.default
        self.model_id, self.model_name, self.G, self.max_pos, self.dim = d.model_id, d.name, d.G, d.max_pos, d.dim
        self.tok, self.pad, self.gen_ok, self.stops = d.tok, d.pad, d.gen_ok, d.stops
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
                "price_per_1k_tokens_coins": self.default.price_per_1k, "min_fee_coins": self.a.min_fee,
                "prices_per_1k_tokens_coins": {m.name: m.price_per_1k for m in self.models.values()}}

    # -------------------------------------------------------------- requests

    def served(self, name) -> Served:
        """The model a request names (its name or chain id); the first model when it names none."""
        if name is None:
            return self.default
        for m in self.models.values():
            if name in (m.name, m.model_id):
                return m
        raise ApiError(404, f"model {name!r} is not served here; use one of {', '.join(self.models)}", "model_not_found")

    def model_of(self, r: sqlite3.Row) -> Served:
        try:
            return self.served(r["model"])
        except ApiError:  # a model no longer served: its requests still show
            return self.default

    def tokenize(self, inp, m: Served | None = None) -> list[list[int]]:
        m = m or self.default
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
            seqs = [e.ids for e in m.tok.encode_batch(inp, add_special_tokens=False)]
        elif all(isinstance(x, list) and x and all(isinstance(t, int) and 0 <= t < m.tok.get_vocab_size(True) for t in x)
                 for x in inp):
            seqs = inp
        else:
            raise ApiError(400, "input must be a string, a list of strings or a list of token lists")
        for s in seqs:
            if not s:
                raise ApiError(400, "an input has no tokens")
            bucket(len(s), m.max_pos)
        return seqs

    def check_cap(self, key: sqlite3.Row, fee: int):
        if key["cap"] is not None and key["spent"] + fee > key["cap"]:
            raise ApiError(402, "this key's spending cap is reached", "insufficient_quota")

    def new_request(self, key: sqlite3.Row, m: Served, kind: str, doc: dict, n: int, tokens: int) -> str:
        rid = secrets.token_hex(12)
        with open(os.path.join(self.a.home, "requests", f"{rid}.json"), "w") as f:
            json.dump(doc, f)
        self.store.q("INSERT INTO requests (id, key_id, created, status, n_inputs, tokens, kind, model) VALUES (?,?,?,?,?,?,?,?)",
                     rid, key["id"], time.time(), "pending", n, tokens, kind, m.name)
        self.wake.set()
        return rid

    def submit(self, key: sqlite3.Row, body: dict) -> str:
        """Embeddings."""
        m = self.served(body.get("model"))
        if not m.embeddings:
            raise ApiError(400, f"{m.name} does not serve embeddings here", "model_not_embeddings")
        seqs = self.tokenize(body.get("input"), m)
        self.check_cap(key, self.fee_for(sum(bucket(len(s), m.max_pos) for s in seqs), m))
        return self.new_request(key, m, "embed", {"seqs": seqs}, len(seqs), sum(len(s) for s in seqs))

    def gen_params(self, m: Served, body: dict, prompts: list[list[int]]) -> tuple[int, int]:
        """(max new tokens, temperature in thousandths) of a generation request."""
        if not m.gen_ok:
            raise ApiError(404, f"{m.name} does not generate", "model_not_found")
        max_new = body.get("max_completion_tokens", body.get("max_tokens", self.a.max_new))
        if not isinstance(max_new, int) or isinstance(max_new, bool) or max_new < 1:
            raise ApiError(400, "max_tokens must be a positive integer")
        longest = max(len(p) for p in prompts)
        if longest + max_new > m.max_pos:
            raise ApiError(400, f"prompt ({longest} tokens) + max_tokens ({max_new}) exceed the model's {m.max_pos}")
        temp = body.get("temperature", self.a.temperature)
        if not isinstance(temp, (int, float)) or isinstance(temp, bool) or not 0 <= temp <= 4:
            raise ApiError(400, "temperature must be between 0 and 4")
        return max_new, int(round(temp * 1000))

    def submit_gen(self, key: sqlite3.Row, m: Served, body: dict, prompts: list[list[int]], kind: str) -> str:
        max_new, temp_milli = self.gen_params(m, body, prompts)
        self.check_cap(key, sum(self.fee_for(len(p) + max_new, m) for p in prompts))
        doc = {"kind": "chat", "prompts": prompts, "max_new": max_new, "temp_milli": temp_milli}
        if len(prompts) == 1:  # the shape single requests always had (older code reads "prompt" and "T")
            doc.update(prompt=prompts[0], T=len(prompts[0]) + max_new)
        return self.new_request(key, m, kind, doc, len(prompts), sum(len(p) for p in prompts))

    def submit_chat(self, key: sqlite3.Row, body: dict) -> str:
        m = self.served(body.get("model"))
        if not m.gen_ok:
            raise ApiError(404, "this gateway's model does not generate", "model_not_found")
        return self.submit_gen(key, m, body, [m.chat_prompt(body.get("messages"), bool(body.get("uwc_think")))], "chat")

    def submit_completion(self, key: sqlite3.Row, body: dict) -> str:
        m = self.served(body.get("model"))
        prompt = body.get("prompt")
        if isinstance(prompt, list) and len(prompt) == 1:
            prompt = prompt[0]
        return self.submit_gen(key, m, body, [m.completion_prompt(prompt)], "completion")

    def submit_batch(self, key: sqlite3.Row, body: dict) -> str:
        """Many inputs as one request: "type" chat (each input a message list or a string, sent as
        one user message), completion (strings) or embeddings (strings)."""
        m = self.served(body.get("model"))
        kind, inputs = body.get("type", "chat"), body.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise ApiError(400, "inputs must be a non-empty list")
        if len(inputs) > MAX_INPUTS:
            raise ApiError(400, f"at most {MAX_INPUTS} inputs per batch")
        if kind == "embeddings":
            return self.submit(key, {"model": m.name, "input": inputs})
        if kind == "chat":
            prompts = [m.chat_prompt([{"role": "user", "content": x}] if isinstance(x, str) else x, bool(body.get("uwc_think")))
                       for x in inputs]
        elif kind == "completion":
            prompts = [m.completion_prompt(x) for x in inputs]
        else:
            raise ApiError(400, "type must be chat, completion or embeddings")
        return self.submit_gen(key, m, body, prompts, kind)

    def gen_doc(self, rid: str) -> dict:
        """A generation request's prompts and max new tokens (also for requests stored before batches)."""
        req = json.load(open(os.path.join(self.a.home, "requests", f"{rid}.json")))
        if "prompts" not in req:
            req["prompts"], req["max_new"] = [req["prompt"]], req["T"] - len(req["prompt"])
        return req

    def gen_results(self, rid: str) -> list[list[int]]:
        res = json.load(open(os.path.join(self.a.home, "results", f"{rid}.json")))
        return res.get("all") or [res["tokens"]]

    def chat_text(self, prompt_len: int, toks: list[int], m: Served | None = None) -> tuple[str, str, int]:
        return (m or self.default).text(toks)

    def chat_view(self, rid: str, r: sqlite3.Row) -> dict:
        m = self.model_of(r)
        req = self.gen_doc(rid)
        comp = r["kind"] == "completion"
        out = {"id": f"{'cmpl' if comp else 'chatcmpl'}-{rid}", "object": "text_completion" if comp else "chat.completion",
               "created": int(r["created"]), "model": m.name}
        if r["status"] == "done":
            choices, used = [], 0
            for i, toks in enumerate(self.gen_results(rid)):
                text, finish, n = m.text(toks[:req["max_new"]])
                used += n
                choices.append({"index": i, "text": text, "finish_reason": finish} if comp else
                               {"index": i, "message": {"role": "assistant", "content": text}, "finish_reason": finish})
            P = sum(len(p) for p in req["prompts"])
            out["choices"] = choices
            out["usage"] = {"prompt_tokens": P, "completion_tokens": used, "total_tokens": P + used}
        return out

    def fee_for(self, tokens: int, m: Served | None = None) -> int:
        return max(int(self.a.min_fee * COIN), math.ceil((m or self.default).price_per_1k * COIN * tokens / 1000))

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

    def view(self, rid: str, key: sqlite3.Row, fmt: str = "float", with_data: bool = True) -> dict:
        r = self.store.one("SELECT * FROM requests WHERE id = ? AND key_id = ?", rid, key["id"])
        if r is None:
            raise ApiError(404, "no such request", "not_found")
        m = self.model_of(r)
        jobs = [{"job": j["job"], "seq_len": j["seq_len"], "inputs": len(json.loads(j["indexes"])), "fee_coins": j["fee"] / COIN,
                 "attempt": j["attempt"], "status": j["chain_status"] or "posted", "delivered": bool(j["delivered"]),
                 **({"note": j["note"]} if j["note"] else {})}
                for j in self.store.q("SELECT * FROM jobs WHERE request = ? AND replaced = 0", rid)]
        out = {"id": rid, "object": "uwc.request", "kind": r["kind"] or "embed", "status": r["status"], "trust": self.trust(rid),
               "model": m.name, "inputs": r["n_inputs"], "fee_coins": r["fee"] / COIN, "jobs": jobs, "created": int(r["created"])}
        if r["error"]:
            out["error"] = r["error"]
        if r["kind"] in ("chat", "completion"):
            if with_data or r["n_inputs"] == 1:
                out.update({k: v for k, v in self.chat_view(rid, r).items() if k in ("choices", "usage")})
        elif r["status"] == "done" and with_data:
            out["data"] = self.embeddings(rid, fmt)
            out["usage"] = {"prompt_tokens": r["tokens"], "total_tokens": r["tokens"]}
        return out

    def results_jsonl(self, rid: str, key: sqlite3.Row) -> bytes:
        r = self.store.one("SELECT * FROM requests WHERE id = ? AND key_id = ?", rid, key["id"])
        if r is None:
            raise ApiError(404, "no such request", "not_found")
        if r["status"] != "done":
            raise ApiError(409, f"the request is {r['status']}, not done yet", "not_ready")
        if r["kind"] in ("chat", "completion"):
            rows = [{"index": c["index"], "text": c.get("text", (c.get("message") or {}).get("content")),
                     "finish_reason": c["finish_reason"]} for c in self.chat_view(rid, r)["choices"]]
        else:
            rows = [{"index": e["index"], "embedding": e["embedding"]} for e in self.embeddings(rid, "float")]
        return b"".join(json.dumps(x).encode() + b"\n" for x in rows)

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
        """Post a pending request's jobs: embeddings one per length bucket, generation prompts
        --max-gen-batch to a job; at most --max-batch sequences each."""
        key = self.store.one("SELECT * FROM keys WHERE id = ?", r["key_id"])
        m = self.model_of(r)
        if r["kind"] in ("chat", "completion"):
            req = self.gen_doc(r["id"])
            seqs, n = req["prompts"], max(1, min(self.a.max_batch, getattr(self.a, "max_gen_batch", 16)))
            parts = [(max(len(seqs[i]) for i in idx) + req["max_new"], idx)
                     for idx in (list(range(k, min(k + n, len(seqs)))) for k in range(0, len(seqs), n))]
            temp = req["temp_milli"]
        else:
            seqs = json.load(open(os.path.join(self.a.home, "requests", f"{r['id']}.json")))["seqs"]
            groups: dict[int, list[int]] = {}
            for i, s in enumerate(seqs):
                groups.setdefault(bucket(len(s), m.max_pos), []).append(i)
            parts = [(L, idx[k:k + self.a.max_batch]) for L, idx in sorted(groups.items())
                     for k in range(0, len(idx), self.a.max_batch)]
            temp = None
        fees = [self.fee_for(L * len(idx), m) for L, idx in parts]
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
            if not self.post_job(r["id"], key, data, seqs, L, idx, fee, 1, h, temp, m):
                if time.time() - r["created"] > self.a.give_up:
                    self.store.q("UPDATE requests SET status = 'failed' WHERE id = ?", r["id"])
                return
        self.store.q("UPDATE requests SET status = 'posted', error = NULL, fee = ? WHERE id = ?", sum(fees), r["id"])
        self.store.q("UPDATE keys SET spent = spent + ? WHERE id = ?", sum(fees), key["id"])

    def post_job(self, rid: str, key, data: Data, seqs, L: int, idx: list[int], fee: int, attempt: int, h: int,
                 temp_milli: int | None = None, m: Served | None = None) -> bool:
        """One job: embedding batches padded to L, or (temp_milli set) a generation job whose
        input rows are the prompts followed by the sentinel up to L."""
        m = m or self.default
        if temp_milli is not None:
            batches = [seqs[i] + [SENTINEL] * (L - len(seqs[i])) for i in idx]
            cmd = ["jobgen", key["keyfile"], m.model_id, input_root(batches).hex(), str(len(batches) * (m.G + 1)),
                   str(fee), str(self.a.deadline), str(self.a.k), str(self.a.n), str(L), str(temp_milli)]
        else:
            batches = [seqs[i] + [m.pad] * (L - len(seqs[i])) for i in idx]
            cmd = ["jobpost", key["keyfile"], m.model_id, input_root(batches).hex(), str(len(batches) * m.G), str(fee),
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
        log(f"request {rid[:8]}: {m.name} job {r['job'][:8]} {len(batches)}x{L} tokens, fee {fee / COIN} (attempt {attempt})")
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
                m = self.model_of(self.store.one("SELECT * FROM requests WHERE id = ?", j["request"]))
                embs = np.load(io.BytesIO(raw))
                n = len(json.loads(j["indexes"]))
                if embs.shape == (n, m.dim) and output_root(list(embs)).hex() == cj["outputroot"]:
                    np.save(os.path.join(self.a.home, "results", f"{jid}.npy"), embs.astype(np.float32))
                    self.store.q("UPDATE jobs SET delivered = 1, note = NULL WHERE job = ?", jid)
                    log(f"job {jid[:8]}: outputs match the committed output root")
                    self.assemble(j["request"])
                else:
                    self.store.q("UPDATE jobs SET note = ? WHERE job = ?", "the delivered outputs do not match the commitment", jid)
        if st == "expired" or (st in ("failed", "aborted")):
            self.retry(j, h, same_fee=st != "expired")

    def deliver_gen(self, j: sqlite3.Row, cj: dict, heads: np.ndarray):
        """A generation job's samples: checked against the committed output root, then each
        sequence's generated tokens (positions P..T-1) become its result."""
        jid, rid = j["job"], j["request"]
        T, idx = j["seq_len"], json.loads(j["indexes"])
        if heads.ndim != 3 or heads.shape != (len(idx), T - 1, 2) or \
                gen_output_root([head_output_root(np.ascontiguousarray(h.astype(np.int32))) for h in heads]).hex() != cj["outputroot"]:
            self.store.q("UPDATE jobs SET note = ? WHERE job = ?", "the delivered samples do not match the commitment", jid)
            return
        prompts = self.gen_doc(rid)["prompts"]
        toks = [[int(x) for x in heads[k, len(prompts[i]) - 1:, 0]] for k, i in enumerate(idx)]
        json.dump({"tokens": toks, "indexes": idx}, open(os.path.join(self.a.home, "results", f"{jid}.json"), "w"))
        self.store.q("UPDATE jobs SET delivered = 1, note = NULL WHERE job = ?", jid)
        log(f"request {rid[:8]}: generation job {jid[:8]} samples match the committed output root ({len(idx)} sequences)")
        js = self.store.q("SELECT * FROM jobs WHERE request = ? AND replaced = 0", rid)
        if not all(x["delivered"] for x in js):
            return
        out: list = [None] * len(prompts)
        for x in js:
            res = json.load(open(os.path.join(self.a.home, "results", f"{x['job']}.json")))
            for t, i in zip(res["tokens"], res["indexes"]):
                out[i] = t
        json.dump({"tokens": out[0], "all": out, "job": jid}, open(os.path.join(self.a.home, "results", f"{rid}.json"), "w"))
        self.store.q("UPDATE requests SET status = 'done', done = ? WHERE id = ?", time.time(), rid)

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
        if r["kind"] in ("chat", "completion"):
            req = self.gen_doc(r["id"])
            seqs, temp = req["prompts"], req["temp_milli"]
        else:
            seqs, temp = json.load(open(os.path.join(self.a.home, "requests", f"{r['id']}.json")))["seqs"], None
        fee = j["fee"] if same_fee else math.ceil(j["fee"] * self.a.fee_bump)
        if not self.post_job(r["id"], key, self.data_for(key["keyfile"]), seqs, j["seq_len"], json.loads(j["indexes"]),
                             fee, j["attempt"] + 1, h, temp, self.model_of(r)):
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
        out = np.zeros((r["n_inputs"], self.model_of(r).dim), dtype=np.float32)
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
            if method == "GET" and path in ("", "/index.html"):
                self.send_page()
                return None, None
            if method == "GET" and path == "/healthz":
                return 200, {"ok": True, "height": gw.chain.height()}
            if method == "GET" and path == "/v1/models":
                return 200, {"object": "list", "data": [m.describe() for m in gw.models.values()],
                             "signup_code_required": bool(gw.a.signup_code), "faucet_coins": gw.a.faucet_coins
                             if gw.a.faucet_key else 0, "min_fee_coins": gw.a.min_fee, "max_inputs": MAX_INPUTS}
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
                    return 200, {"object": "list", "data": v["data"], "model": v["model"], "usage": v["usage"],
                                 "uwc": {k: v[k] for k in ("id", "trust", "fee_coins", "jobs")}}
                if v["status"] == "failed":
                    raise ApiError(502, f"the request failed: {v.get('error')}", "job_failed")
                raise ApiError(504, f"still running; poll GET /v1/jobs/{rid}", "timeout")
            if method == "POST" and path == "/v1/batches":
                return 202, gw.view(gw.submit_batch(key, self.body()), key, with_data=False)
            if method == "POST" and path in ("/v1/chat/completions", "/v1/completions"):
                b = self.body()
                rid = gw.submit_chat(key, b) if path == "/v1/chat/completions" else gw.submit_completion(key, b)
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
                return 200, {"object": "list", "data": [gw.view(r["id"], key, with_data=False) for r in rows]}
            if method == "GET" and path.startswith("/v1/jobs/") and path.endswith("/results"):
                rid = path.split("/")[3]
                body = gw.results_jsonl(rid, key)
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Content-Disposition", f'attachment; filename="uwc-{rid}.jsonl"')
                self.send_header("Content-Length", str(len(body)))
                self.cors()
                self.end_headers()
                self.wfile.write(body)
                return None, None
            if method == "GET" and path.startswith("/v1/jobs/"):
                return 200, gw.view(path.rsplit("/", 1)[1], key, query.get("encoding_format", "float"))
            raise ApiError(404, f"no route {method} {path}", "not_found")

        def send_page(self):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                                                        "script-src 'self' 'unsafe-inline'; img-src 'self' data:")
            self.end_headers()
            self.wfile.write(body)

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
            r0 = gw.store.one("SELECT * FROM requests WHERE id = ?", rid)
            m, comp = gw.model_of(r0), r0["kind"] == "completion"
            req = gw.gen_doc(rid)
            P, max_new = len(req["prompts"][0]), req["max_new"]
            base = {"id": f"{'cmpl' if comp else 'chatcmpl'}-{rid}", "object": "text_completion" if comp else "chat.completion.chunk",
                    "model": m.name}

            def chunk(delta, finish=None, **kw):
                c = {"index": 0, "finish_reason": finish}
                c.update({"text": delta.get("content", "")} if comp else {"delta": delta})
                return {**base, "created": int(time.time()), "choices": [c], **kw}
            if not self.sse(chunk({"role": "assistant", "content": ""} if not comp else {"content": ""})):
                return
            sent, job, deadline = "", None, time.time() + gw.a.sync_timeout
            while time.time() < deadline:
                r = gw.store.one("SELECT * FROM requests WHERE id = ?", rid)
                if r["status"] == "failed":
                    self.sse({"error": {"message": f"the request failed: {r['error']}", "type": "job_failed"}})
                    return
                if r["status"] == "done":
                    text, finish, n = m.text(gw.gen_results(rid)[0][:max_new])
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
                            text, finish, _ = m.text(json.loads(raw)["tokens"][0][:max_new])
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
    ap.add_argument("--models", default=None, help="JSON list of served models (see the module doc); instead of --model")
    ap.add_argument("--model", default=None, help="one model json (with its .reg.json beside it)")
    ap.add_argument("--model-name", default="qwen3-1.7b")
    ap.add_argument("--tokenizer", default=None, help="the model's tokenizer.json (with --model)")
    ap.add_argument("--pad-token", default="<|endoftext|>")
    ap.add_argument("--bind", default="127.0.0.1:18693")
    ap.add_argument("--price-per-1k", type=float, default=1.5, help="coins per 1,000 (padded) tokens")
    ap.add_argument("--min-fee", type=float, default=0.5, help="coins per job at least")
    ap.add_argument("--deadline", type=int, default=60, help="blocks for a provider to accept and commit")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-batch", type=int, default=64, help="sequences per job")
    ap.add_argument("--max-gen-batch", type=int, default=16, help="generation prompts per job (a batch's chat inputs)")
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
    if not a.models and not (a.model and a.tokenizer):
        ap.error("give --models, or --model and --tokenizer")
    gw = Gateway(a)
    threading.Thread(target=gw.run_worker, daemon=True).start()
    host, port = a.bind.rsplit(":", 1)
    srv = ThreadingHTTPServer((host, int(port)), handler(gw))
    srv.daemon_threads = True
    log(f"gateway for {', '.join(f'{m.name} ({m.model_id[:8]})' for m in gw.models.values())} on {a.bind}, data {a.data}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
