"""Agents with several models (Rules 4: Qwen3, Llama, Mistral on one testnet). Committees are drawn
from every staker whatever the job's model, so a verifier or watcher must use the right model for
each job and skip jobs of models it does not hold. Registrations of one checkpoint share a runtime.
No GPU: Runtime is replaced by a stand-in.

Run: python -m pytest tests/test_agent_models.py   (from runtime/)
"""
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def write_model(d, name, model_id, model_dir, groups, block=512, max_pos=1024):
    p = os.path.join(d, f"{name}.json")
    json.dump({"model_dir": model_dir, "rotation_block": block, "layer_groups": groups, "cfg": {"MaxPos": max_pos}}, open(p, "w"))
    json.dump({"model": model_id}, open(p.replace(".json", ".reg.json"), "w"))
    return p


def agent(model_arg, weights_arg=""):
    from agent import compute, roles
    built = []

    class FakeRuntime:
        def __init__(self, path, device, mine_socket=None, engine=None):
            built.append(path)
            self.path = path
    compute.Runtime = FakeRuntime
    a = object.__new__(roles.Verifier)
    a.a = types.SimpleNamespace(devnet="/nonexistent", device="cpu", engine="detgpu", mine=False)
    a.init_models(model_arg, weights_arg)
    return a, built


def test_models_switch_share_and_skip():
    with tempfile.TemporaryDirectory() as d:
        qwen = write_model(d, "qwen-r4", "aa" * 32, "/m/Qwen3-1.7B", 7)
        qwen_old = write_model(d, "qwen-old", "bb" * 32, "/m/Qwen3-1.7B", 7)       # same checkpoint, older registration
        llama = write_model(d, "llama", "cc" * 32, "/m/Llama-3.2-3B-Instruct", 7, block=128)
        mistral = write_model(d, "mistral", "dd" * 32, "/m/Mistral-7B", 8, block=128)
        a, built = agent(",".join([qwen, qwen_old, llama, mistral]), "/w/qwen,/w/qwen,/w/llama")
        assert a.model_id == "aa" * 32 and a.G == 7 and a.weights_path == "/w/qwen"  # the first is current
        assert a.use("dd" * 32) and a.G == 8 and a.model_path == mistral and a.weights_path == ""
        r_mistral = a.rt
        assert a.use("aa" * 32)
        r_qwen = a.rt
        assert a.use("bb" * 32) and a.rt is r_qwen          # one runtime for both Qwen registrations
        assert a.use("cc" * 32) and a.rt is not r_qwen and a.rt is not r_mistral
        assert len(built) == 3
        a.load_models()
        assert len(built) == 3                               # already loaded: nothing new
        assert not a.use("ee" * 32)                          # a model it does not hold
        assert a.model_id == "cc" * 32                        # ... leaves the current model as it was
        assert "ee" * 32 in a._skipped_models


def test_single_model_as_before():
    with tempfile.TemporaryDirectory() as d:
        qwen = write_model(d, "qwen", "aa" * 32, "/m/Qwen3-1.7B", 7)
        a, built = agent(qwen, "/w/qwen")
        assert list(a.models) == ["aa" * 32] and a.G == 7 and a.model_path == qwen
        a.load_models()
        assert built == [qwen]


def test_provider_claims_only_its_model():
    """Several models' pool categories are open (Rules 4): a provider claims units of its own model
    only, and never computes a job of a model it does not run."""
    from agent import roles
    with tempfile.TemporaryDirectory() as d:
        qwen = write_model(d, "qwen", "aa" * 32, "/m/Qwen3-1.7B", 7)
        p = object.__new__(roles.Provider)
        p.a = types.SimpleNamespace(devnet=d, device="cpu", engine="detgpu", mine=False, prefer="")
        p.init_models(qwen, "")
        doc = {"work_root": "77" * 32, "units": [{"input_root": "00" * 32, "batches": [[1, 2]]}] * 3}
        p.work_list = lambda root: doc
        p.lists = {doc["work_root"]: doc}
        p.can_bond = lambda bond, h: True
        p.st, p.path = {"jobs": {}}, os.path.join(d, "state.json")
        sent = []
        p.send = lambda ops, what: sent.append(ops) or {"job": "jj" * 32, "txid": "t"}
        p.chain = types.SimpleNamespace(jobs=lambda status=None: {})
        cat = lambda model: {"retired": False, "workroot": "77" * 32, "units": 3, "price": 2_000_000_000, "model": model}
        s = {"height": 10, "categories": {"c1" * 32: cat("cc" * 32), "c2" * 32: cat("dd" * 32)}}
        p.claim(s)
        assert sent == []                                   # no category of its model: nothing claimed
        s["categories"]["c3" * 32] = cat("aa" * 32)
        p.claim(s)
        assert len(sent) == 1 and sent[0][0]["job"] == "c3" * 32
        # a job of another model where this provider is the provider (an older version's claim)
        p.account = "pp" * 32
        p.chain = types.SimpleNamespace(job=lambda jid: {"id": jid, "provider": "pp" * 32, "model": "cc" * 32,
                                                         "status": "accepted", "pool": "c1" * 32, "unit": 0})
        p.ready, p.gen_ready = {}, {}
        p.advance("ff" * 32)
        assert p.ready == {} and p.st["jobs"]["ff" * 32]["foreign_model"] == "cc" * 32


def test_verifier_declares_its_models_on_rules5():
    """Rules 5 draws committees only from verifiers that declared the job's model: a verifier
    declares every model it loaded (sorted, as the chain wants), only once they are loaded, only
    on Rules 5, and again whenever the chain's list differs."""
    with tempfile.TemporaryDirectory() as d:
        a, _ = agent(",".join([write_model(d, "llama", "cc" * 32, "/m/L", 7), write_model(d, "qwen", "aa" * 32, "/m/Q", 7)]))
        sent, chain = [], types.SimpleNamespace(rules=5, h=100)
        chain.height = lambda: chain.h
        chain.state = lambda: {"params": {"rules": chain.rules}, "height": chain.h}
        a.chain, a.st = chain, {"jobs": {}}
        a.send = lambda ops, what: sent.append(ops) or {"txid": "t"}
        a.declare_models({"models": []})
        assert sent == []                                    # models not loaded yet
        a.models_loaded = True
        a.declare_models({"models": None})                   # a node from before Rules 5: no "models"
        assert sent == []
        a.declare_models({"models": []})
        assert sent == [[{"kind": "VERIFIER_MODELS", "roots": ["aa" * 32, "cc" * 32]}]]
        chain.h += 3
        a.declare_models({"models": []})                     # not mined yet: no resend within 10 blocks
        assert len(sent) == 1
        chain.h += 10
        a.declare_models({"models": ["aa" * 32, "cc" * 32]})  # recorded: nothing to do
        assert len(sent) == 1
        a.declare_models({"models": ["aa" * 32]})            # the chain's list differs: declare again
        assert len(sent) == 2
        b, _ = agent(write_model(d, "q2", "aa" * 32, "/m/Q", 7))
        b.chain, b.st, b.models_loaded = types.SimpleNamespace(height=lambda: 50, state=lambda: {"params": {"rules": 4}}), {"jobs": {}}, True
        b.send = lambda ops, what: sent.append(ops)
        b.declare_models({"models": []})
        assert len(sent) == 2                                # Rules 4: the op would fail; nothing sent


def test_provider_skips_models_without_verifiers_on_rules5():
    from agent import roles
    with tempfile.TemporaryDirectory() as d:
        qwen = write_model(d, "qwen", "aa" * 32, "/m/Qwen3-1.7B", 7)
        p = object.__new__(roles.Provider)
        p.a = types.SimpleNamespace(devnet=d, device="cpu", engine="detgpu", mine=False, prefer="")
        p.init_models(qwen, "")
        doc = {"work_root": "77" * 32, "units": [{"input_root": "00" * 32, "batches": [[1, 2]]}] * 3}
        p.work_list = lambda root: doc
        p.lists = {doc["work_root"]: doc}
        p.can_bond = lambda bond, h: True
        p.st, p.path = {"jobs": {}}, os.path.join(d, "state.json")
        sent = []
        p.send = lambda ops, what: sent.append(ops) or {"job": "jj" * 32, "txid": "t"}
        p.chain = types.SimpleNamespace(jobs=lambda status=None: {})
        cat = {"retired": False, "workroot": "77" * 32, "units": 3, "price": 2_000_000_000, "model": "aa" * 32}
        s = {"height": 10, "params": {"rules": 5}, "verifiers": {"cc" * 32: 2}, "categories": {"c1" * 32: cat}}
        p.claim(s)
        assert sent == []                                    # Rules 5, no verifier of its model
        s["verifiers"]["aa" * 32] = 1
        p.claim(s)
        assert len(sent) == 1
        s2 = {"height": 11, "params": {"rules": 4}, "categories": {"c2" * 32: dict(cat)}}
        p.claim(s2)                                          # Rules 4: anyone verifies; claimed as before
        assert len(sent) == 2
