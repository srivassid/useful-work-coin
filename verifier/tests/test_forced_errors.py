import itertools
import json
import secrets
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "verifier"))
import forced_errors as fe  # noqa: E402

VECTOR = ROOT / "ext/pearl/node/jobstate/testdata/step5_vectors.json"


def test_shamir_any_t_of_m():
    secret = secrets.token_bytes(32) + fe.encode_record([secrets.token_bytes(32) for _ in range(3)])
    shares = fe.split(secret, 3, 5)
    for sub in itertools.combinations(shares, 3):
        assert fe.combine(list(sub)) == secret
    for sub in itertools.combinations(shares, 4):
        assert fe.combine(list(sub)) == secret
    for sub in itertools.combinations(shares, 2):  # t - 1 shares give something else
        assert fe.combine(list(sub)) != secret


def test_t_minus_one_shares_are_uniform():
    # with t = 2, one share's byte is uniform whatever the secret: count values over many splits
    counts = [0] * 256
    for _ in range(25600):
        counts[fe.split(b"\x07", 2, 3)[0][1][0]] += 1
    assert min(counts) > 50 and max(counts) < 160


def test_record_round_trip():
    d = [secrets.token_bytes(32) for _ in range(200)]  # count needs a 2-byte varint
    assert fe.decode_record(fe.encode_record(d)) == d


def test_cli_round_trip(tmp_path):
    out = subprocess.run([sys.executable, str(ROOT / "verifier/forced_errors.py"), "new", "--decoys", "3", "-t", "2", "-m", "3",
                          "--out", str(tmp_path)], check=True, capture_output=True, text=True).stdout
    made = json.loads(out)
    rev = subprocess.run([sys.executable, str(ROOT / "verifier/forced_errors.py"), "reveal", str(tmp_path / "share-1.json"),
                          str(tmp_path / "share-3.json")], check=True, capture_output=True, text=True).stdout
    got = json.loads(rev)
    assert got["decoys"] == made["decoys"]
    assert fe.fe_commitment(bytes.fromhex(got["record"]), bytes.fromhex(got["salt"])).hex() == made["commitment"]
    bad = subprocess.run([sys.executable, str(ROOT / "verifier/forced_errors.py"), "reveal", str(tmp_path / "share-2.json")],
                         capture_output=True, text=True)
    assert bad.returncode != 0


def make_vectors():
    """Deterministic vectors that node/jobstate's TestStep5Vectors checks."""
    accts = [fe.account_for_script(bytes([0x51, i, 0])) for i in (1, 2, 3)]
    record = fe.encode_record(accts)
    salt = fe.tag_hash("salt")
    inputs = [fe.tag_hash("in", fe.u32(i)) for i in range(5)]
    wleaves = [fe.unit_leaf(i, r) for i, r in enumerate(inputs)]
    expected = [fe.tag_hash("root", fe.u32(s)) for s in range(4)]
    cleaves = [fe.tag_hash("dummy"), fe.canary_leaf(3, expected, salt), fe.tag_hash("dummy2")]
    return {
        "record": record.hex(), "salt": salt.hex(), "commitment": fe.fe_commitment(record, salt).hex(),
        "work_root": fe.merkle_root(wleaves).hex(), "unit3_proof": [h.hex() for h in fe.merkle_proof(wleaves, 3)],
        "canary_root": fe.merkle_root(cleaves).hex(), "canary_proof": [h.hex() for h in fe.merkle_proof(cleaves, 1)],
    }


def test_vectors_match_go_fixture():
    assert json.loads(VECTOR.read_text()) == make_vectors()


if __name__ == "__main__":
    VECTOR.write_text(json.dumps(make_vectors(), indent=1) + "\n")
    print("wrote", VECTOR)
