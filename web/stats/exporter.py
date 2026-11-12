#!/usr/bin/env python3
"""Publishes mainnet mining stats as a public JSON file, for the live miners page.

Reads the seed node's RPC (localhost only) every few seconds and writes OUT atomically. It
publishes only what the chain already makes public (block heights, times, coinbase addresses)
plus counts: peer IP addresses are never written.

Environment: RPC_URL (http://127.0.0.1:8691), RPC_USER, RPC_PASS_FILE, OUT, LABELS (a JSON file
mapping address -> label), INTERVAL seconds (10), KEEP blocks (10080, a week at a block a minute).
"""
import base64, json, os, sys, time, urllib.request

RPC_URL = os.environ.get("RPC_URL", "http://127.0.0.1:8691")
RPC_USER = os.environ.get("RPC_USER", "uwc")
RPC_PASS = open(os.environ.get("RPC_PASS_FILE", "/opt/uwc/main/rpcpass")).read().strip()
OUT = os.environ.get("OUT", "/var/www/uwc/stats.json")
LABELS_FILE = os.environ.get("LABELS", "/opt/uwc/stats/labels.json")
INTERVAL = float(os.environ.get("INTERVAL", "10"))
KEEP = int(os.environ.get("KEEP", "10080"))
RECHECK = 20  # tip blocks re-checked each pass, for reorgs

AUTH = "Basic " + base64.b64encode(f"{RPC_USER}:{RPC_PASS}".encode()).decode()


def rpc(method, *params):
    body = json.dumps({"jsonrpc": "1.0", "id": 1, "method": method, "params": list(params)}).encode()
    req = urllib.request.Request(RPC_URL, body, {"Authorization": AUTH, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        res = json.load(r)
    if res.get("error"):
        raise RuntimeError(f"{method}: {res['error']}")
    return res["result"]


def coinbase_address(block):
    for o in block["rawtx"][0]["vout"]:
        spk = o.get("scriptPubKey", {})
        if o.get("value", 0) > 0:
            return spk.get("address") or ("script:" + spk.get("hex", "")[:16])
    return "unknown"


blocks = {}  # height -> (hash, time, address)


def fetch(h):
    bh = rpc("getblockhash", h)
    b = rpc("getblock", bh, 2)
    blocks[h] = (bh, b["time"], coinbase_address(b))


def sync():
    tip = rpc("getblockcount")
    lo = max(1, tip - KEEP + 1)
    for h in [h for h in blocks if h < lo or h > tip]:
        del blocks[h]
    for h in range(max(lo, tip - RECHECK + 1), tip + 1):  # reorg check near the tip
        if h in blocks and rpc("getblockhash", h) != blocks[h][0]:
            del blocks[h]
    for h in range(lo, tip + 1):
        if h not in blocks:
            fetch(h)
    return tip


def labels():
    try:
        return json.load(open(LABELS_FILE))
    except (OSError, ValueError):
        return {}


def build(tip):
    now = int(time.time())
    lab = labels()
    windows = {"1h": 3600, "24h": 86400, "7d": 7 * 86400}
    miners = {}
    for h, (_, t, a) in blocks.items():
        m = miners.setdefault(a, {"address": a, "label": lab.get(a), "first_height": h, "last_height": h,
                                  "last_time": t, **{f"blocks_{k}": 0 for k in windows}})
        m["first_height"] = min(m["first_height"], h)
        if h > m["last_height"]:
            m["last_height"], m["last_time"] = h, t
        for k, s in windows.items():
            if t > now - s:
                m[f"blocks_{k}"] += 1
    summary = {}
    for k, s in windows.items():
        active = [m for m in miners.values() if m[f"blocks_{k}"] > 0]
        summary[k] = {"miners": len(active), "outside_miners": sum(1 for m in active if not m["label"]),
                      "blocks": sum(m[f"blocks_{k}"] for m in active)}
    # per hour for the last 24 hours: blocks and distinct miners
    hours = []
    top = now - now % 3600
    for i in range(23, -1, -1):
        start = top - i * 3600
        inside = [a for (_, t, a) in blocks.values() if start <= t < start + 3600]
        hours.append({"start": start, "blocks": len(inside), "miners": len(set(inside))})
    peers = rpc("getpeerinfo")
    mi = rpc("getmininginfo")
    tip_time = blocks[tip][1] if tip in blocks else None
    return {
        "network": "uwc-mainnet",
        "updated": now,
        "height": tip,
        "tip_time": tip_time,
        "difficulty": mi.get("difficulty"),
        "network_rate": mi.get("networkhashps"),
        "seed_peers": len(peers),
        "seed_peers_inbound": sum(1 for p in peers if p.get("inbound")),
        "summary": summary,
        "hours": hours,
        "miners": sorted(miners.values(), key=lambda m: (-m["blocks_24h"], -m["last_height"]))[:200],
        "recent": [{"height": h, "time": blocks[h][1], "address": blocks[h][2]}
                   for h in sorted(blocks, reverse=True)[:30]],
    }


def write(doc):
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    os.chmod(tmp, 0o644)
    os.replace(tmp, OUT)


def main():
    while True:
        try:
            write(build(sync()))
        except Exception as e:  # node restarting etc.: keep the last file, try again
            print("stats:", e, file=sys.stderr, flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
