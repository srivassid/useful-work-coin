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
