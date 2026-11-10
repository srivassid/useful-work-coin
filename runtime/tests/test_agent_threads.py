"""Agent threading (no GPU needed): the provider hang of 2026-11-09, where its main loop and GPU
worker shared the job miner's gateway socket and the main loop stopped for three hours.

- JobMiner: concurrent refreshes over one real gateway client (pearl_gateway's JSONRPCClient on
  a Unix socket) all finish; a gateway that never answers times out and the next call reconnects.
- Agent.on_gpu: providers run GPU work on the worker and the main loop only polls for it.
- Agent.watchdog: a main loop that stops finishing steps is logged with every thread's stack.

Run: python -m pytest tests/test_agent_threads.py   (from runtime/, any Python with numpy+torch CPU)
"""
import concurrent.futures
import contextlib
import io
import json
import os
import socket
import sys
import tempfile
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "ext", "pearl", "miner", "pearl-gateway", "src"))


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules.setdefault(name, m)
    return sys.modules[name]


def load_mining():
    """agent.mining with its GPU and Rust dependencies stubbed, and MiningClient built on the
    real JSON-RPC socket client."""
    import logging
    _stub("miner_utils", get_logger=logging.getLogger)
    from pearl_gateway.comm.json_rpc_client import JSONRPCClient
    from pearl_gateway.config import MinerRpcConfig  # noqa: F401  (the real config)

    class MiningClient:  # miner_base.gateway_client.MiningClient without its pearl_mining types
        def __init__(self, cfg):
            self.client = JSONRPCClient(cfg)

        def get_mining_info(self):
            r = self.client.call("getMiningInfo")
            return types.SimpleNamespace(incomplete_header_bytes=r["header"], target=r["target"], cert_version=1)

        def submit_plain_proof(self, proof, job):
            self.client.call("submitPlainProof", {})

        def close(self):
            self.client.close()

    _stub("miner_base")
    _stub("miner_base.gateway_client", MiningClient=MiningClient)
    _stub("pearl_mining", IncompleteBlockHeader=types.SimpleNamespace(from_bytes=lambda b: b))
    _stub("uwc_ampere", ops=types.SimpleNamespace(), pearl_glue=types.SimpleNamespace())
    _stub("uwc_ampere.ops")
    _stub("uwc_ampere.pearl_glue")
    import detrt.qwen3  # noqa: F401  (real; JobMiner patches DetLinear.int_gemm)
    from agent import mining
    return mining


class Gateway:
    """A fake pearl gateway on a Unix socket: answers getMiningInfo after a short delay, or
    never (silent=True), the way a stuck gateway would."""

    def __init__(self, path):
        self.path, self.silent, self.calls = path, False, 0
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(path)
        self.srv.listen(16)
        threading.Thread(target=self.accept, daemon=True).start()

    def accept(self):
        while True:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self.serve, args=(c,), daemon=True).start()

    def serve(self, c):
        f = c.makefile("rw", encoding="utf-8")
        with contextlib.suppress(Exception):
            for line in f:
                req = json.loads(line)
                self.calls += 1
                if self.silent:
                    continue
                time.sleep(0.001)
                f.write(json.dumps({"jsonrpc": "2.0", "id": req["id"],
                                    "result": {"header": "ab" * 8, "target": 1}}) + "\n")
                f.flush()


def model():
    return types.SimpleNamespace(layers=[])


def test_miner_concurrent_refresh():
    mining = load_mining()
    with tempfile.TemporaryDirectory() as d:
        Gateway(os.path.join(d, "gw.sock"))
        jm = mining.JobMiner(model(), os.path.join(d, "gw.sock"), log=lambda *a: None)
        results = []

        def many():
            for _ in range(200):
                results.append(jm.refresh())
        ts = [threading.Thread(target=many, daemon=True) for _ in range(6)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(30)
        assert not any(t.is_alive() for t in ts), "a refresh hung"
        assert len(results) == 1200 and all(results)


def test_miner_silent_gateway_times_out_and_reconnects():
    mining = load_mining()
    mining.RPC_TIMEOUT = 0.5
    with tempfile.TemporaryDirectory() as d:
        gw = Gateway(os.path.join(d, "gw.sock"))
        logs = []
        jm = mining.JobMiner(model(), os.path.join(d, "gw.sock"), log=logs.append)
        gw.silent = True
        t = time.time()
        assert jm.refresh() is False
        assert time.time() - t < 5
        assert any("no template" in x for x in logs)
        gw.silent = False
        assert jm.refresh() is True  # a fresh connection after the timeout


def agent(cls, **args):
    from agent import roles
    a = object.__new__(cls)
    a.a = types.SimpleNamespace(stall=0.3, task_stall=0.3, **args)
    a.worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    a._gpu_tasks, a._task, a._beat = {}, None, time.time()
    a.WATCH_EVERY = 0.05
    a.roles = roles
    return a


def test_on_gpu_runs_on_worker_for_providers():
    from agent.roles import Provider, Verifier
    p = agent(Provider)
    main = threading.get_ident()
    ran_on = []
    gate = threading.Event()

    def fn():
        ran_on.append(threading.get_ident())
        gate.wait(5)
        return 42
    assert p.on_gpu(("trace", "j", 1), fn) is None  # queued, the caller does not wait
    assert p.on_gpu(("trace", "j", 1), fn) is None  # still running: not queued twice
    gate.set()
    for _ in range(100):
        r = p.on_gpu(("trace", "j", 1), fn)
        if r is not None:
            break
        time.sleep(0.02)
    assert r == 42 and ran_on and ran_on[0] != main and len(ran_on) == 1
    # a failed task is logged and retried on the next call
    assert p.on_gpu(("bad",), lambda: 1 / 0) is None
    time.sleep(0.1)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        assert p.on_gpu(("bad",), lambda: 1 / 0) is None
    assert ("bad",) not in p._gpu_tasks
    v = agent(Verifier)  # roles without GPU passes on the worker compute in place
    assert v.on_gpu(("x",), lambda: threading.get_ident()) == main


def test_watchdog_logs_stalled_loop_and_long_task():
    from agent.roles import Provider
    p = agent(Provider)
    p._beat = time.time() - 10
    hold = threading.Event()
    p.submit("generation job deadbeef", hold.wait, 5)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        stop = threading.Event()
        w = threading.Thread(target=p.watchdog, args=(stop,), name="watchdog", daemon=True)
        w.start()
        time.sleep(0.6)
        stop.set()
        w.join(2)
    hold.set()
    text = out.getvalue()
    assert "WATCHDOG: the main loop has not finished a step" in text
    assert "WATCHDOG: the GPU worker has run 'generation job deadbeef'" in text
    assert "--- thread MainThread" in text and "--- thread watchdog" not in text
