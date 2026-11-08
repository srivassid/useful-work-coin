"""The job gateway against a fake chain and a simulated provider (real tokenizer, real data service)."""
import argparse
import base64
import io
import json
import os
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from agent import gateway as gwmod
from agent import roots
from agent.data import handler as data_handler

TOKENIZER = os.path.expanduser(os.environ.get("UWC_TOKENIZER", "~/Documents/useful-work-coin/models/Qwen3-1.7B/tokenizer.json"))
D = 2048


def test_roots_match_the_runtime():
    from agent.inputs import input_root
    from agent.proto import output_root
    from detgen import head
    out = np.random.default_rng(2).integers(0, 1 << 30, size=(15, 2)).astype(np.int32)
    assert roots.head_output_root(out) == head.head_output_root(out) and roots.SENTINEL == head.SENTINEL
    assert roots.gen_output_root([b"a" * 32, b"b" * 32]) == head.gen_output_root([b"a" * 32, b"b" * 32])
    assert roots.input_root([[1, 2, roots.SENTINEL]]) == input_root([[1, 2, roots.SENTINEL]])
    rng = np.random.default_rng(1)
    for n in (1, 3, 7):
        batches = [list(map(int, rng.integers(0, 151000, size=int(rng.integers(1, 600))))) for _ in range(n)]
        assert roots.input_root(batches) == input_root(batches)
        embs = [rng.standard_normal(D).astype(np.float32) for _ in range(n)]
        assert roots.output_root(embs) == output_root(embs)


class FakeChain:
    """Enough of agent.chain.Chain for the gateway: keys, balances, job posts and job state."""

    def __init__(self):
        self.h = 100
        self.jobs: dict[str, dict] = {}
        self.balances: dict[str, int] = {}
        self.n = 0

    def height(self):
        return self.h

    def key_info(self, path):
        acct = json.load(open(path))["acct"]
        return {"account": acct, "address": "rprl1p" + acct[:20], "script": "51"}

    def account(self, acct):
        return {"balance": self.balances.get(acct, 0)}

    def job(self, jid):
        return self.jobs.get(jid)

    def run(self, cmd, *args, stdin=None):
        if cmd == "keygen":
            self.n += 1
            json.dump({"acct": f"{self.n:064x}"}, open(args[0], "w"))
            return "ok"
        if cmd == "sign":
            return "ab" * 32 + ":" + "cd" * 64
        if cmd == "send":
            return {"txid": "00"}
        if cmd == "deposit":
            self.balances[args[2]] = self.balances.get(args[2], 0) + int(args[1])
            return {"txid": "00"}
        if cmd == "coins":
            return {"total": 0}
        if cmd in ("jobpost", "jobgen"):
            key, model, root, stages, fee, deadline, k, n, seq_len = args[:9]
            acct = json.load(open(key))["acct"]
            self.balances[acct] -= int(fee)
            jid = f"{len(self.jobs) + 1:064x}"
            self.jobs[jid] = {"id": jid, "status": "posted", "inputroot": root, "fee": int(fee), "stages": int(stages),
                              "seqlen": int(seq_len), "outputroot": "0" * 64, "client": acct}
            if cmd == "jobgen":
                self.jobs[jid].update(gen=True, tempmilli=int(args[9]))
            return {"job": jid}
        raise AssertionError(cmd)


@pytest.fixture()
def world(tmp_path):
    if not os.path.exists(TOKENIZER):
        pytest.skip("needs the Qwen3 tokenizer (UWC_TOKENIZER)")
    data_root = tmp_path / "data"
    dsrv = ThreadingHTTPServer(("127.0.0.1", 0), data_handler(str(data_root)))
    threading.Thread(target=dsrv.serve_forever, daemon=True).start()
    data_url = f"http://127.0.0.1:{dsrv.server_address[1]}"
    model = tmp_path / "m.json"
    json.dump({"layer_groups": 7, "cfg": {"MaxPos": 512, "D": D}, "gen": True}, open(model, "w"))
    json.dump({"model": "ee" * 32}, open(tmp_path / "m.reg.json", "w"))
    fake = FakeChain()
    gwmod.Chain = lambda net: fake
    a = argparse.Namespace(net="x", data=data_url, home=str(tmp_path / "gw"), model=str(model), model_name="qwen3-1.7b",
                           tokenizer=TOKENIZER, pad_token="<|endoftext|>", price_per_1k=1.5, min_fee=0.5, deadline=60,
                           k=2, n=2, max_batch=64, attempts=3, fee_bump=1.5, give_up=1800, sync_timeout=30, poll=0.2,
                           signup_code="letmein", keys_per_ip=3, spend_cap=0, faucet_key=str(tmp_path / "faucet.json"),
                           faucet_coins=100, faucet_fee_coins=2, cors_origin="*", max_new=24, temperature=0.7)
    gw = gwmod.Gateway(a)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), gwmod.handler(gw))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    yield gw, fake, url, data_url
    srv.shutdown()
    dsrv.shutdown()


def call(url, method, path, body=None, key=None):
    req = urllib.request.Request(url + path, method=method, data=json.dumps(body).encode() if body is not None else None)
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def provide(fake, data_url, jid, rng):
    """What a provider does: compute (here: random) embeddings, upload them, commit their root."""
    j = fake.jobs[jid]
    batches = json.loads(urllib.request.urlopen(f"{data_url}/inputs/{j['inputroot']}.json").read())["batches"]
    assert all(len(b) == j["seqlen"] for b in batches) and j["stages"] == 7 * len(batches)
    embs = rng.standard_normal((len(batches), D)).astype(np.float32)
    buf = io.BytesIO()
    np.save(buf, embs)
    urllib.request.urlopen(urllib.request.Request(f"{data_url}/outputs/{jid}.npy", data=buf.getvalue(), method="PUT")).read()
    j.update(status="verify-commit", outputroot=roots.output_root(list(embs)).hex())
    return batches, embs


def test_request_to_results(world):
    gw, fake, url, data_url = world
    assert call(url, "POST", "/v1/keys", {"name": "x", "signup_code": "wrong"})[0] == 403
    st, k = call(url, "POST", "/v1/keys", {"name": "friend", "signup_code": "letmein"})
    assert st == 201 and k["faucet_coins"] == 100 and k["api_key"].startswith("uwc_sk_")
    key = k["api_key"]
    assert call(url, "GET", "/v1/account")[0] == 401
    assert call(url, "GET", "/v1/account", key=key)[1]["jobs_balance_coins"] == 100
    assert call(url, "POST", "/v1/jobs", {"model": "gpt-4", "input": "hi"}, key)[0] == 404
    assert call(url, "POST", "/v1/jobs", {"input": "word " * 700}, key)[0] == 400

    texts = ["A short sentence.", "Another short one.", "A much longer text about GPUs. " * 12]
    st, r = call(url, "POST", "/v1/jobs", {"input": texts}, key)
    assert st == 202 and r["status"] == "pending"
    gw.tick()
    jobs = list(fake.jobs)
    assert len(jobs) == 2  # two length buckets
    assert sorted(fake.jobs[j]["seqlen"] for j in jobs) == [32, 128]
    rng = np.random.default_rng(0)
    out = {}
    for jid in jobs:
        batches, embs = provide(fake, data_url, jid, rng)
        out[jid] = (batches, embs)
    gw.tick()
    v = call(url, "GET", f"/v1/jobs/{r['id']}", key=key)[1]
    assert v["status"] == "done" and v["trust"] == "committed" and len(v["data"]) == 3
    # every input's embedding is its own job row, in the request's order
    for jid, (batches, embs) in out.items():
        idx = json.loads(gw.store.one("SELECT indexes FROM jobs WHERE job = ?", jid)["indexes"])
        for row, i in zip(embs, idx):
            assert np.allclose(v["data"][i]["embedding"], row)
    for jid in jobs:
        fake.jobs[jid]["status"] = "challenge"
    gw.tick()
    assert call(url, "GET", f"/v1/jobs/{r['id']}", key=key)[1]["trust"] == "verified"
    for jid in jobs:
        fake.jobs[jid]["status"] = "settled"
    gw.tick()
    assert call(url, "GET", f"/v1/jobs/{r['id']}", key=key)[1]["trust"] == "final"
    b64 = call(url, "GET", f"/v1/jobs/{r['id']}?encoding_format=base64", key=key)[1]["data"][0]["embedding"]
    assert np.allclose(np.frombuffer(base64.b64decode(b64), "<f4"), v["data"][0]["embedding"])


def test_expired_and_failed_jobs_are_posted_again(world):
    gw, fake, url, data_url = world
    key = call(url, "POST", "/v1/keys", {"name": "f", "signup_code": "letmein"})[1]["api_key"]
    rid = call(url, "POST", "/v1/jobs", {"input": "hello"}, key)[1]["id"]
    gw.tick()
    (j1,) = fake.jobs
    fee1 = fake.jobs[j1]["fee"]
    fake.jobs[j1]["status"] = "expired"
    fake.balances[fake.jobs[j1]["client"]] += fee1  # the chain refunds an expired job
    gw.tick()
    j2 = [j for j in fake.jobs if j != j1][0]
    assert fake.jobs[j2]["fee"] == int(fee1 * 1.5 + 0.999)
    provide(fake, data_url, j2, np.random.default_rng(2))
    gw.tick()
    assert call(url, "GET", f"/v1/jobs/{rid}", key=key)[1]["status"] == "done"
    # its provider loses a dispute: the job is posted again and the request waits for new results
    fake.jobs[j2]["status"] = "failed"
    fake.balances[fake.jobs[j2]["client"]] += fake.jobs[j2]["fee"]
    gw.tick()
    v = call(url, "GET", f"/v1/jobs/{rid}", key=key)[1]
    assert v["status"] == "posted" and v["trust"] == "none" and v["jobs"][0]["attempt"] == 3
    j3 = v["jobs"][0]["job"]
    provide(fake, data_url, j3, np.random.default_rng(3))
    gw.tick()
    assert call(url, "GET", f"/v1/jobs/{rid}", key=key)[1]["status"] == "done"


def test_sync_embeddings_openai_shape(world):
    gw, fake, url, data_url = world
    key = call(url, "POST", "/v1/keys", {"name": "s", "signup_code": "letmein"})[1]["api_key"]
    threading.Thread(target=gw.run_worker, daemon=True).start()
    stop = threading.Event()

    def provider():  # answers every new job
        done, rng = set(), np.random.default_rng(4)
        while not stop.is_set():
            for jid in list(fake.jobs):
                if jid not in done:
                    provide(fake, data_url, jid, rng)
                    done.add(jid)
            stop.wait(0.2)
    threading.Thread(target=provider, daemon=True).start()
    st, r = call(url, "POST", "/v1/embeddings", {"model": "qwen3-1.7b", "input": ["one", "two"]}, key)
    stop.set()
    assert st == 200 and r["object"] == "list" and len(r["data"]) == 2
    assert len(r["data"][0]["embedding"]) == D and r["usage"]["prompt_tokens"] > 0 and r["uwc"]["trust"] == "committed"


def provide_gen(fake, data_url, jid, gw, reply: str, stream_first: int | None = None):
    """A provider of a generation job: samples whose generated part is reply + <|im_end|>, the
    partial stream (if stream_first), and the commitment of the head-stage outputs."""
    from detgen.head import SENTINEL, gen_output_root, head_output_root
    j = fake.jobs[jid]
    row = json.loads(urllib.request.urlopen(f"{data_url}/inputs/{j['inputroot']}.json").read())["batches"][0]
    P, T = row.index(SENTINEL), j["seqlen"]
    assert j["stages"] == 8 and len(row) == T and j.get("gen")
    gen = gw.tok.encode(reply, add_special_tokens=False).ids + [gw.tok.token_to_id("<|im_end|>")]
    gen = (gen + [0] * T)[:T - P]
    if stream_first:
        urllib.request.urlopen(urllib.request.Request(f"{data_url}/stream/{jid}.json", method="PUT",
                                                      data=json.dumps({"tokens": [gen[:stream_first]]}).encode())).read()
    heads = np.zeros((1, T - 1, 2), dtype=np.int32)
    heads[0, P - 1:, 0] = gen
    buf = io.BytesIO()
    np.save(buf, heads)
    urllib.request.urlopen(urllib.request.Request(f"{data_url}/outputs/{jid}.npy", data=buf.getvalue(), method="PUT")).read()
    j.update(status="verify-commit", outputroot=gen_output_root([head_output_root(heads[0])]).hex())


def test_chat_completion(world):
    gw, fake, url, data_url = world
    key = call(url, "POST", "/v1/keys", {"name": "c", "signup_code": "letmein"})[1]["api_key"]
    assert call(url, "POST", "/v1/chat/completions", {"messages": "hi"}, key)[0] == 400
    assert call(url, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 900}, key)[0] == 400
    threading.Thread(target=gw.run_worker, daemon=True).start()
    stop = threading.Event()

    def provider():
        done = set()
        while not stop.is_set():
            for jid in list(fake.jobs):
                if jid not in done:
                    provide_gen(fake, data_url, jid, gw, "Paris is the capital of France.")
                    done.add(jid)
            stop.wait(0.2)
    threading.Thread(target=provider, daemon=True).start()
    st, r = call(url, "POST", "/v1/chat/completions",
                 {"model": "qwen3-1.7b", "messages": [{"role": "user", "content": "What is the capital of France?"}],
                  "temperature": 0.5}, key)
    stop.set()
    assert st == 200 and r["object"] == "chat.completion", r
    assert r["choices"][0]["message"]["content"] == "Paris is the capital of France."
    assert r["choices"][0]["finish_reason"] == "stop" and r["uwc"]["trust"] == "committed"
    (jid,) = fake.jobs
    assert fake.jobs[jid]["tempmilli"] == 500 and r["usage"]["completion_tokens"] == len(
        gw.tok.encode("Paris is the capital of France.", add_special_tokens=False).ids) + 1


def test_chat_stream(world):
    gw, fake, url, data_url = world
    key = call(url, "POST", "/v1/keys", {"name": "s", "signup_code": "letmein"})[1]["api_key"]
    threading.Thread(target=gw.run_worker, daemon=True).start()
    reply = "The quick brown fox jumps over the lazy dog."
    stop = threading.Event()

    def provider():
        while not stop.is_set() and not fake.jobs:
            stop.wait(0.1)
        if stop.is_set():
            return
        (jid,) = fake.jobs
        # a partial generation first, then the committed samples a moment later
        from detgen.head import SENTINEL  # noqa: F401
        toks = gw.tok.encode(reply, add_special_tokens=False).ids
        urllib.request.urlopen(urllib.request.Request(f"{data_url}/stream/{jid}.json", method="PUT",
                                                      data=json.dumps({"tokens": [toks[:4]]}).encode())).read()
        stop.wait(1.5)
        provide_gen(fake, data_url, jid, gw, reply)
    threading.Thread(target=provider, daemon=True).start()
    req = urllib.request.Request(url + "/v1/chat/completions", method="POST", data=json.dumps(
        {"messages": [{"role": "user", "content": "Say the fox sentence."}], "stream": True}).encode())
    req.add_header("Authorization", f"Bearer {key}")
    events = []
    with urllib.request.urlopen(req, timeout=60) as resp:
        assert resp.headers["Content-Type"] == "text/event-stream"
        for line in resp:
            line = line.decode().strip()
            if line.startswith("data: "):
                events.append(line[6:])
    stop.set()
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == reply and len([c for c in chunks if c["choices"][0]["delta"].get("content")]) >= 2  # it streamed
    last = chunks[-1]
    assert last["choices"][0]["finish_reason"] == "stop" and last["uwc"]["trust"] == "committed"

