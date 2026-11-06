"""Testnet metrics (design doc, "Testnet metrics") for the devnet, from the chain, the agents'
state files and the noise miners' logs.

  python -m agent.metrics [--json out.json]

- useful share: settled job compute / compute implied by difficulty (2^256 / target MACs a block)
- lambda: noise rates measured by the miners; jobs on the devnet run without mining for now
- verification overhead: verifier compute time / provider compute time, per job
- false-slash rate: failed jobs whose provider computed honestly
- client price: fee / (provider GPU time x what that GPU earns noise mining)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess

from .chain import Chain

DEVNET = os.path.expanduser("~/Documents/useful-work-coin/devnet")
GRAIN = 1e8


def block_info(chain: Chain, h: int) -> dict:
    out = subprocess.run([os.path.join(chain.dir, "bin", "prlctl"), "--configfile=/dev/null", f"--rpcserver=127.0.0.1:{chain.env.get('RPC_PORT', '18671')}",
                          f"--rpcuser={chain.env['RPC_USER']}", f"--rpcpass={chain.env['RPC_PASS']}", "--notls",
                          "getblockhash", str(h)], capture_output=True, text=True).stdout.strip()
    hdr = subprocess.run([os.path.join(chain.dir, "bin", "prlctl"), "--configfile=/dev/null", f"--rpcserver=127.0.0.1:{chain.env.get('RPC_PORT', '18671')}",
                          f"--rpcuser={chain.env['RPC_USER']}", f"--rpcpass={chain.env['RPC_PASS']}", "--notls",
                          "getblockheader", out], capture_output=True, text=True).stdout
    return json.loads(hdr)


def target_of(bits_hex: str) -> int:
    bits = int(bits_hex, 16)
    exp, mant = bits >> 24, bits & 0xFFFFFF
    return mant << (8 * (exp - 3)) if exp > 3 else mant >> (8 * (3 - exp))


def stage_macs(model: dict, seq_len: int) -> float:
    """Linear-layer MACs of one stage (layers/G layers over seq_len tokens): what a job computes."""
    c = model["cfg"]
    D, I, H, Hkv, hd = c["D"], c["I"], c["H"], c["Hkv"], c["Hd"]
    per_token_layer = D * (H + 2 * Hkv) * hd + H * hd * D + D * 2 * I + I * D
    attn = 2 * seq_len * H * hd / 2  # QK^T and PV, causal on average
    return (per_token_layer + attn) * seq_len * (c["L"] // model["layer_groups"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devnet", default=DEVNET)
    ap.add_argument("--window", type=int, default=60, help="blocks for difficulty and block time")
    ap.add_argument("--json", default=None)
    ap.add_argument("--extra", nargs="*", default=[], help="other boxes' devnet dirs (agent states, miner logs) copied here")
    a = ap.parse_args()
    chain = Chain(a.devnet)
    s = chain.state()
    h = s["height"]
    model = json.load(open(os.path.join(a.devnet, "models", "qwen3-1.7b.json")))

    # difficulty and block time over the window
    lo = max(1, h - a.window)
    b0, b1 = block_info(chain, lo), block_info(chain, h)
    secs_per_block = (b1["time"] - b0["time"]) / max(1, h - lo)
    hash_macs_per_block = 2 ** 256 / target_of(b1["bits"])
    network_macs_s = hash_macs_per_block / secs_per_block

    # jobs
    jobs = chain.jobs()
    settled_stages = s["stats"]["SettledStages"]
    per_stage = stage_macs(model, 128)
    useful_macs = settled_stages * per_stage
    hashed_total = hash_macs_per_block * h  # approximate: the current difficulty for every block

    # agents' own records
    dirs = [a.devnet] + a.extra

    def states(role):
        out = {}
        for d in dirs:
            for f in glob.glob(os.path.join(d, "agent", f"{role}-*.json")):
                out.update(json.load(open(f)).get("jobs", {}))
        return out
    prov, ver = states("provider"), states("verifier")
    decoys = states("decoy")
    overhead = []
    for jid, js in ver.items():
        p = prov.get(jid, {})
        if js.get("computed") and p.get("computed"):
            overhead.append(js["computed"] / p["computed"])

    # noise rates from the miners
    rates = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "logs", "noise-gpu*.jsonl")):
            recs = [json.loads(l) for l in open(f) if l.strip()][-20:]
            if recs:
                rates[os.path.basename(d.rstrip("/")) + ":" + os.path.basename(f)] = (
                    recs[-1]["device"], statistics.median(r["macs_per_s"] for r in recs))
    by_device: dict[str, list[float]] = {}
    for dev, r in rates.values():
        by_device.setdefault(dev, []).append(r)
    # live lambda: hashed MACs per second of a mining provider's job over its GPU class's noise rate
    live = []
    for jid, p in prov.items():
        if p.get("hashed_macs") and p.get("computed") and p.get("gpu"):
            noise = statistics.median(by_device.get(p["gpu"], [0])) or None
            if noise:
                live.append(p["hashed_macs"] / p["computed"] / noise)

    # client price against renting the same GPU at its mining income
    reward = 100 * GRAIN * (1 - 0.10)  # miner's share of the subsidy (approximately the current subsidy)
    price = []
    gpu_rate = statistics.median(r for _, r in rates.values()) if rates else 67e12
    income_per_gpu_s = reward / secs_per_block * gpu_rate / network_macs_s
    for jid, e in jobs.items():
        p = prov.get(jid, {})
        if not e.get("pool") and p.get("computed") and e["status"] == "settled":
            price.append(e["fee"] / (p["computed"] * income_per_gpu_s))

    failed = [jid for jid, e in jobs.items() if e["status"] == "failed"]
    false_slash = [jid for jid in failed if jid in prov and jid not in decoys]
    out = {
        "height": h,
        "seconds_per_block": round(secs_per_block, 1),
        "network_TMACs": round(network_macs_s / 1e12, 1),
        "stats": s["stats"],
        "jobs": s["jobs"],
        "useful_share": useful_macs / hashed_total if hashed_total else 0,
        "useful_share_note": f"{settled_stages} settled stages x {per_stage/1e9:.0f} GMAC vs {hashed_total/1e18:.1f} EMAC hashed",
        "noise_TMACs_by_gpu": {k: [d, round(r / 1e12, 1)] for k, (d, r) in rates.items()},
        "lambda_live_median": statistics.median(live) if live else None,
        "lambda_live_jobs": len(live),
        "verification_overhead_median": statistics.median(overhead) if overhead else None,
        "verification_overhead_jobs": len(overhead),
        "false_slashes": len(false_slash),
        "failed_jobs": len(failed),
        "client_price_vs_rent_median": statistics.median(price) if price else None,
        "client_jobs_priced": len(price),
        "pool_reserve": s["pool"], "treasury": s["treasury"],
        "categories": {cid[:8]: {k: c[k] for k in ("budget", "price", "done", "inflight", "units")} for cid, c in s["categories"].items()},
    }
    # per-job table for the page: kind, outcome, and the cause of each failure as the agents saw it
    causes = {}
    for f in glob.glob(os.path.join(a.devnet, "agent", "*-*.json")):
        pass
    table = []
    for jid, e in sorted(jobs.items(), key=lambda x: x[1]["posted"]):
        kind = "decoy" if jid in decoys else ("pool" if e.get("pool") else "client")
        table.append({"job": jid[:8], "kind": kind, "status": e["status"], "stages": e["stages"], "posted": e["posted"],
                      "fee": e["fee"], "provider": e["provider"][:8]})
    out["job_table"] = table
    out["blocks_window"] = [lo, h]
    print(json.dumps(out, indent=1, default=str))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
