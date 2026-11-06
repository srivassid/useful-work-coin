"""Forced-error commitments for the devnet council (spec/pool-and-watchers.md).

  python -m agent.fe commit --decoys decoy0 --reserve 60000000   # record, shares, PROPOSE + APPROVE
  python -m agent.fe reveal                                      # combine 2 shares, FE_REVEAL
Shares live in ~/.uwc/devnet/fe/<commitment>/share-<x>.json, one per council member.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "verifier"))
import forced_errors as fe  # noqa: E402

from .chain import Chain  # noqa: E402

KEYS = os.path.expanduser("~/.uwc/devnet/keys")
FE_DIR = os.path.expanduser("~/.uwc/devnet/fe")


def wait_mempool(chain: Chain):
    for _ in range(60):
        if not json.loads(subprocess_out(chain, "getrawmempool")):
            return
        time.sleep(10)


def subprocess_out(chain: Chain, *args) -> str:
    import subprocess
    p = subprocess.run([os.path.join(chain.dir, "bin", "prlctl"), "--configfile=/dev/null", "--rpcserver=127.0.0.1:18671",
                        f"--rpcuser={chain.env['RPC_USER']}", f"--rpcpass={chain.env['RPC_PASS']}", "--notls", *args],
                       capture_output=True, text=True)
    return p.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["commit", "reveal"])
    ap.add_argument("--devnet", default=os.path.expanduser("~/Documents/useful-work-coin/devnet"))
    ap.add_argument("--decoys", nargs="*", default=["decoy0"])
    ap.add_argument("--reserve", type=int, default=60_000_000)
    a = ap.parse_args()
    chain = Chain(a.devnet)
    if a.cmd == "commit":
        accts = [bytes.fromhex(chain.key_info(os.path.join(KEYS, k + ".json"))["account"]) for k in a.decoys]
        record, salt = fe.encode_record(accts), secrets.token_bytes(32)
        c = fe.fe_commitment(record, salt)
        d = os.path.join(FE_DIR, c.hex()[:16])
        os.makedirs(d, mode=0o700, exist_ok=True)
        for x, y in fe.split(salt + record, 2, 3):  # one share per council member
            p = os.path.join(d, f"share-{x}.json")
            json.dump({"x": x, "t": 2, "m": 3, "share": y.hex(), "commitment": c.hex()}, open(p, "w"))
            os.chmod(p, 0o600)
        r = chain.run("propose-fe", os.path.join(KEYS, "council0.json"), c.hex(), str(a.reserve))
        wait_mempool(chain)
        chain.run("approve", os.path.join(KEYS, "council1.json"), r["proposal"])
        wait_mempool(chain)
        fid = fe.tag_hash("fe", bytes.fromhex(r["proposal"])).hex()
        json.dump({"commitment": c.hex(), "fecommit": fid, "proposal": r["proposal"]}, open(os.path.join(d, "commit.json"), "w"))
        print(json.dumps({"commitment": c.hex(), "fecommit": fid, "shares": d}, indent=1))
    else:
        for d in sorted(os.listdir(FE_DIR)):
            info = json.load(open(os.path.join(FE_DIR, d, "commit.json")))
            if info.get("revealed"):
                continue
            shares = [json.load(open(os.path.join(FE_DIR, d, f"share-{x}.json"))) for x in (1, 3)]
            secret = fe.combine([(s["x"], bytes.fromhex(s["share"])) for s in shares])
            salt, record = secret[:32], secret[32:]
            assert fe.fe_commitment(record, salt).hex() == info["commitment"]
            r = chain.run("fe-reveal", os.path.join(KEYS, "council2.json"), info["fecommit"], salt.hex(), record.hex())
            info["revealed"] = r["txid"]
            json.dump(info, open(os.path.join(FE_DIR, d, "commit.json"), "w"))
            print(json.dumps({"fecommit": info["fecommit"], "tx": r["txid"]}))


if __name__ == "__main__":
    main()
