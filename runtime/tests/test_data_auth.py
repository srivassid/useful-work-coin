"""The --auth data service: signed uploads, rights from the chain (a fake chain here; signatures
with the real uwcctl, path in $UWCCTL, e.g. a build of ext/pearl/node/cmd/uwcctl)."""
import os
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from agent.data import Data, Policy, handler, put_digest

U = os.environ.get("UWCCTL", "")
pytestmark = pytest.mark.skipif(not os.path.exists(U), reason="needs UWCCTL (uwcctl with sign/verify)")
J = "ab" * 32
R = "cd" * 32
N = "3-" + "0" * 32 + ".json"


@pytest.fixture(scope="module")
def svc(tmp_path_factory):
    d = tmp_path_factory.mktemp("keys")
    keys = {}
    for n in ("prov", "verif", "rich", "poor", "other"):
        f = str(d / f"{n}.json")
        out = subprocess.run([U, "keygen", f], check=True, capture_output=True, text=True).stdout
        keys[n] = (f, out.split()[1])

    class FakeChain:
        bin, env = U, dict(os.environ)

        def job(self, jid):
            return {"provider": keys["prov"][1], "committee": [{"account": keys["verif"][1]}]} if jid == J else None

        def account(self, a):
            return {"balance": 5 * 10**8 if a == keys["rich"][1] else 0, "stake": 0}

    root = str(tmp_path_factory.mktemp("data"))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler(root, Policy(FakeChain(), 10**8), threading.BoundedSemaphore(8)))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"

    def client(n):
        f = keys[n][0]
        return Data(url, signer=lambda dg: subprocess.run([U, "sign", f, dg.hex()], capture_output=True, text=True,
                                                          check=True).stdout.strip())
    yield url, client, keys, root
    srv.shutdown()


def code(fn):
    try:
        fn()
        return 201
    except urllib.error.HTTPError as e:
        return e.code


def test_rights(svc):
    url, client, keys, root = svc
    cases = [
        (lambda: Data(url).put(f"/outputs/{J}.npy", b"x"), 401),
        (lambda: client("other").put(f"/outputs/{J}.npy", b"x"), 403),
        (lambda: client("prov").put(f"/outputs/{J}.npy", b"x"), 201),
        (lambda: client("prov").put(f"/outputs/{'ee' * 32}.npy", b"x"), 404),
        (lambda: client("poor").put(f"/inputs/{R}.json", b"{}"), 403),
        (lambda: client("rich").put(f"/inputs/{R}.json", b'{"batches":[]}'), 201),
        (lambda: client("rich").put(f"/inputs/{R}.json", b'{"batches":[]}'), 201),
        (lambda: client("rich").put(f"/inputs/{R}.json", b'{"batches":[[1]]}'), 409),
        (lambda: client("prov").put(f"/stagein/{J}/{N}", b"{}"), 404),
        (lambda: client("poor").put(f"/req/{J}/{N}", b"{}"), 403),
        (lambda: client("verif").put(f"/req/{J}/{N}", b"{}"), 201),
        (lambda: client("other").put(f"/stagein/{J}/{N}", b"{}"), 403),
        (lambda: client("prov").put(f"/stagein/{J}/{N}", b'{"rows":1}'), 201),
        (lambda: client("verif").put(f"/req/{J}/4-{'1' * 32}.json", b"x" * 2000), 413),
    ]
    got = [code(fn) for fn, _ in cases]
    assert got == [w for _, w in cases]
    assert open(os.path.join(root, "outputs", J + ".npy"), "rb").read() == b"x"


def test_replayed_signature(svc):
    url, client, keys, root = svc
    sig = subprocess.run([U, "sign", keys["prov"][0], put_digest(f"/outputs/{J}.npy", b"good").hex()],
                         capture_output=True, text=True).stdout.strip()
    r = urllib.request.Request(url + f"/outputs/{J}.npy", data=b"evil", method="PUT")
    r.add_header("X-UWC-Sig", sig)
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(r, timeout=10)
    assert e.value.code == 401
