"""Keyholder and curator tools for build step 5 (spec/pool-and-watchers.md).

Forced errors: the council's keyholders run decoy jobs from fresh provider accounts. The list
of those accounts (the record) and a random salt are committed on chain with FE_COMMIT
(H("fe-record" || record || salt)). The record and salt are split t-of-m with Shamir's scheme
over GF(256), so no keyholder needs to keep the secret and any t of them can post FE_REVEAL
after the decoys end.

Curators: a pool category commits its work list (unit input roots) and, hidden among dummy
leaves, its canary units' expected stage roots. These helpers build both roots and the
CANARY_REVEAL proof exactly as node/jobstate does.

Usage:
  python verifier/forced_errors.py new --decoys 5 -t 3 -m 5 --out shares/
  python verifier/forced_errors.py reveal shares/share-1.json shares/share-3.json shares/share-4.json
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import secrets
import sys
from pathlib import Path

# ---------------------------------------------------------------- hashing (node/jobstate)


def tag_hash(tag: str, *parts: bytes) -> bytes:
    h = hashlib.sha256(tag.encode())
    for p in parts:
        h.update(p)
    return h.digest()


def u32(v: int) -> bytes:
    return v.to_bytes(4, "little")


def account_for_script(pk_script: bytes) -> bytes:
    return hashlib.sha256(pk_script).digest()


def _node(l: bytes, r: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + l + r).digest()


def _split(n: int) -> int:
    k = 1
    while k << 1 < n:
        k <<= 1
    return k


def merkle_root(leaves: list[bytes]) -> bytes:
    if not leaves:
        return hashlib.sha256(b"").digest()
    if len(leaves) == 1:
        return leaves[0]
    k = _split(len(leaves))
    return _node(merkle_root(leaves[:k]), merkle_root(leaves[k:]))


def merkle_proof(leaves: list[bytes], i: int) -> list[bytes]:
    if len(leaves) <= 1:
        return []
    k = _split(len(leaves))
    if i < k:
        return merkle_proof(leaves[:k], i) + [merkle_root(leaves[k:])]
    return merkle_proof(leaves[k:], i - k) + [merkle_root(leaves[:k])]


def unit_leaf(i: int, input_root: bytes) -> bytes:
    return tag_hash("unit", u32(i), input_root)


def canary_leaf(unit: int, expected_roots: list[bytes], salt: bytes) -> bytes:
    return tag_hash("canary", u32(unit), merkle_root(expected_roots), salt)


def canary_tree(canaries: dict[int, list[bytes]], dummies: int, salt: bytes) -> tuple[bytes, int, dict[int, tuple[int, list[bytes]]]]:
    """Leaves for the canary units, shuffled among `dummies` random leaves. Returns the root, the
    tree size and, per canary unit, (leaf index, proof) for CANARY_REVEAL."""
    leaves = [(u, canary_leaf(u, r, salt)) for u, r in canaries.items()] + [(None, secrets.token_bytes(32)) for _ in range(dummies)]
    order = sorted(range(len(leaves)), key=lambda _: secrets.randbits(64))
    leaves = [leaves[i] for i in order]
    hashes = [h for _, h in leaves]
    opens = {u: (i, merkle_proof(hashes, i)) for i, (u, _) in enumerate(leaves) if u is not None}
    return merkle_root(hashes), len(hashes), opens


# ---------------------------------------------------------------- forced-error record


def _uvarint(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def encode_record(decoys: list[bytes], stage_key: bytes | None = None) -> bytes:
    """jobstate.EncodeFERecord (version 1) or EncodeFERecordV2 (version 2, with the stage key
    that fixes each decoy job's wrong stage; needed from Rules 2)."""
    assert all(len(a) == 32 for a in decoys)
    if stage_key is None:
        return b"\x01" + _uvarint(len(decoys)) + b"".join(decoys)
    assert len(stage_key) == 32
    return b"\x02" + _uvarint(len(decoys)) + b"".join(decoys) + stage_key


def fe_stage(stage_key: bytes, job: bytes, stages: int) -> int:
    """jobstate.FEStage: the stage a decoy job carries its wrong root at (record version 2)."""
    x = tag_hash("fe-stage", stage_key, job)
    return int.from_bytes(x[:8], "big") % stages


def decode_record(b: bytes) -> list[bytes]:
    return decode_record_full(b)[0]


def decode_record_full(b: bytes) -> tuple[list[bytes], bytes | None]:
    """(decoy accounts, stage key or None for a version-1 record)."""
    if not b or b[0] not in (1, 2):
        raise ValueError("bad record version")
    key = None
    if b[0] == 2:
        if len(b) < 33:
            raise ValueError("bad record length")
        b, key = b"\x01" + b[1:-32], b[-32:]
    n, shift, i = 0, 0, 1
    while True:
        c = b[i]
        n |= (c & 0x7F) << shift
        i += 1
        if c < 0x80:
            break
        shift += 7
    body = b[i:]
    if len(body) != 32 * n:
        raise ValueError("bad record length")
    return [body[k * 32:(k + 1) * 32] for k in range(n)], key


def fe_commitment(record: bytes, salt: bytes) -> bytes:
    return tag_hash("fe-record", record, salt)


# ---------------------------------------------------------------- Shamir over GF(256)

_EXP = [0] * 512
_LOG = [0] * 256
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x ^= (_x << 1) ^ (0x11B if _x & 0x80 else 0)  # multiply by the generator 3
    _x &= 0xFF
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def _mul(a: int, b: int) -> int:
    return 0 if a == 0 or b == 0 else _EXP[_LOG[a] + _LOG[b]]


def _div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError
    return 0 if a == 0 else _EXP[(_LOG[a] - _LOG[b]) % 255]


def split(secret: bytes, t: int, m: int) -> list[tuple[int, bytes]]:
    """m shares (x = 1..m), any t of which reconstruct the secret."""
    if not 1 <= t <= m <= 255:
        raise ValueError("need 1 <= t <= m <= 255")
    coeffs = [[s] + [secrets.randbelow(256) for _ in range(t - 1)] for s in secret]
    shares = []
    for x in range(1, m + 1):
        ys = bytearray()
        for poly in coeffs:
            y = 0
            for c in reversed(poly):  # Horner
                y = _mul(y, x) ^ c
            ys.append(y)
        shares.append((x, bytes(ys)))
    return shares


def combine(shares: list[tuple[int, bytes]]) -> bytes:
    """Lagrange interpolation at 0. With fewer than t shares the result is unrelated to the secret."""
    xs = [x for x, _ in shares]
    if len(set(xs)) != len(xs):
        raise ValueError("duplicate share")
    out = bytearray()
    for k in range(len(shares[0][1])):
        acc = 0
        for i, (xi, yi) in enumerate(shares):
            num, den = 1, 1
            for j, xj in enumerate(xs):
                if j != i:
                    num = _mul(num, xj)
                    den = _mul(den, xi ^ xj)
            acc ^= _mul(yi[k], _div(num, den))
        out.append(acc)
    return bytes(out)


# ---------------------------------------------------------------- CLI


def cmd_new(a: argparse.Namespace) -> None:
    # fresh decoy provider accounts: in practice the hashes of fresh P2TR scripts the keyholders control
    decoys = [account_for_script(b"\x51\x20" + secrets.token_bytes(32)) for _ in range(a.decoys)]
    record, salt = encode_record(decoys), secrets.token_bytes(32)
    commitment = fe_commitment(record, salt)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for x, y in split(salt + record, a.t, a.m):
        p = out / f"share-{x}.json"
        p.write_text(json.dumps({"x": x, "t": a.t, "m": a.m, "share": y.hex(), "commitment": commitment.hex()}))
        os.chmod(p, 0o600)
    print(json.dumps({"commitment": commitment.hex(), "decoys": [d.hex() for d in decoys]}, indent=1))


def cmd_reveal(a: argparse.Namespace) -> None:
    sh = [json.loads(Path(p).read_text()) for p in a.shares]
    secret = combine([(s["x"], bytes.fromhex(s["share"])) for s in sh])
    salt, record = secret[:32], secret[32:]
    ok = fe_commitment(record, salt).hex() == sh[0]["commitment"]
    if not ok:
        sys.exit("shares do not open the commitment (too few shares, or from different commitments)")
    print(json.dumps({"salt": salt.hex(), "record": record.hex(), "decoys": [d.hex() for d in decode_record(record)]}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    n = sp.add_parser("new", help="make a decoy record, its commitment and t-of-m shares")
    n.add_argument("--decoys", type=int, default=5)
    n.add_argument("-t", type=int, required=True)
    n.add_argument("-m", type=int, required=True)
    n.add_argument("--out", required=True)
    r = sp.add_parser("reveal", help="combine t shares into the FE_REVEAL record and salt")
    r.add_argument("shares", nargs="+")
    a = ap.parse_args()
    {"new": cmd_new, "reveal": cmd_reveal}[a.cmd](a)


if __name__ == "__main__":
    main()
