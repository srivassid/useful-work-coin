"""Useful-Work Coin v3 -- economics and attack simulator.

Checks four claims in the design doc:
  A. Public-good pool equilibrium: share of GPUs on pool jobs u = rho / (1 - lambda),
     and effective hashrate falls by the factor (1 - rho).
  B. Any per-job premium is captured by self-dealing unless it is <= what the job burns.
  C. Withheld fee + a few random stage samples deter a provider that skips stages.
  D. Probability that a fully faked job passes, given colluding verifier stake.

Build step 6 extends it (parts E-I):
  E. Measured lambda: the step-1 benchmarks on RTX 3080 and RTX 5090, and what they imply for
     the client price and for which GPUs the pool attracts.
  F. A GPU count that responds to revenue (free entry and exit, two measured GPU classes with
     a cost curve), with the pool, client demand and a coin-price shock.
  G. Risk-averse providers: cheaters with concave utility, honest providers' risk premia
     (false slashing, bond capital, payout variance) and the bond size.
  H. Block withholding: a miner-provider re-rolls the verifier committee by discarding the
     seed block.
  I. Retune of rho, k, n and bonds from E-H.

Run: python3 sim.py [--parts ABCDEFGHI]   (stdlib only, deterministic seeds)
"""
import argparse
import json
import math
import os
import random


# ---------------------------------------------------------------- A. pool equilibrium
def pool_equilibrium(rho, lam, n=1000, rounds=2000, update_frac=0.02,
                     lam_spread=0.0, seed=1):
    """Agent-based best response. Each GPU noise-mines or serves pool jobs.

    Per-block revenue, block reward R = 1, ticket rate when noise-mining h = 1:
      noise GPU : (1 - rho) * 1 / H
      job GPU   : (1 - rho) * lam_i / H + rho / u_count   (pool budget split evenly)
    H = sum of ticket rates. GPUs revise with probability update_frac per round.
    """
    rng = random.Random(seed)
    lams = [min(1.0, max(0.0, lam + rng.uniform(-lam_spread, lam_spread)))
            for _ in range(n)]
    on_job = [False] * n
    k, H = 0, float(n)
    for _ in range(rounds):
        for i in range(n):
            if rng.random() >= update_frac:
                continue
            # revenue if GPU i chooses each option, holding others fixed
            k_job = k + (0 if on_job[i] else 1)
            H_job = H - (lams[i] if on_job[i] else 1.0) + lams[i]
            H_noise = H - (lams[i] if on_job[i] else 1.0) + 1.0
            r_job = (1 - rho) * lams[i] / H_job + rho / k_job
            r_noise = (1 - rho) * 1.0 / H_noise
            choose_job = r_job > r_noise
            if choose_job != on_job[i]:
                k += 1 if choose_job else -1
                H = H_job if choose_job else H_noise
                on_job[i] = choose_job
    u = sum(on_job) / n
    H = sum(lams[i] if on_job[i] else 1.0 for i in range(n)) / n
    return u, H


def part_a():
    print("A. Public-good pool equilibrium (homogeneous GPUs)")
    print(f"{'rho':>5} {'lambda':>7} {'u sim':>7} {'u formula':>10} {'H sim':>7} {'1-rho':>7}")
    rows = []
    for lam in (0.5, 0.7, 0.9):
        for rho in (0.05, 0.10, 0.20, 0.30):
            u, H = pool_equilibrium(rho, lam)
            f = min(1.0, rho / (1 - lam))
            rows.append((rho, lam, u, f, H))
            print(f"{rho:5.2f} {lam:7.2f} {u:7.3f} {f:10.3f} {H:7.3f} {1-rho:7.3f}")
    print("\nA2. Heterogeneous GPUs (lambda_i uniform +/- 0.2 around the mean)")
    for lam in (0.5, 0.7):
        for rho in (0.10, 0.20):
            u, H = pool_equilibrium(rho, lam, lam_spread=0.2)
            print(f"rho={rho:.2f} mean lambda={lam:.2f}: u sim={u:.3f} "
                  f"(homogeneous formula {min(1, rho/(1-lam)):.3f}), H={H:.3f}")
    return rows


# ---------------------------------------------------------------- B. self-dealing premium
def part_b(trials=200_000, seed=2):
    """A miner posts a job to itself. Premium tickets are M times easier.

    Per first-layer tile, reward per unbound ticket r = R / D. Self-dealer gains
    (M - 1) * r * T1 extra expected reward and pays burn b * fee.
    Compare fee at the floor  b * fee = (M - 1) * r * T1  and at half the floor.
    """
    rng = random.Random(seed)
    R, D, T1, M, b = 1.0, 1_000_000.0, 4096, 4.0, 0.10
    r = R / D
    floor_fee = (M - 1) * r * T1 / b
    print("\nB. Self-dealing a premium ticket job (M = 4, burn 10%)")
    for label, fee in (("at floor", floor_fee), ("half floor", 0.5 * floor_fee)):
        p_win = M * T1 / D  # chance the job's tiles win a block (small)
        gain = 0.0
        for _ in range(trials):
            if rng.random() < p_win:
                gain += R
        extra_reward = gain / trials - T1 * r      # minus what noise mining would earn
        net = extra_reward - b * fee
        print(f"  fee {label:>10}: extra reward {extra_reward:.5f}, burn {b*fee:.5f}, "
              f"net {net:+.5f} per job (expected net {(M-1)*r*T1 - b*fee:+.5f})")


# ---------------------------------------------------------------- C. sampled verification
def p_detect(S, m, n):
    """P(at least one of n distinct sampled stages is among m skipped of S)."""
    if n >= S - m + 1:
        return 1.0
    miss = 1.0
    for i in range(n):
        miss *= (S - m - i) / (S - i)
    return 1.0 - miss


def part_c():
    """Cheater skips m stages, saving m*c. If caught: loses withheld fee and bond.

    Honest profit = fee - S*c. Expected cheat minus honest = m*c - P(m)*(fee + bond).
    """
    S, c = 1000, 1.0
    print("\nC. Skipping stages (S = 1000 stages, cost 1 per stage)")
    print(f"{'fee/cost':>8} {'bond/cost':>9} {'samples n':>9} {'worst m':>8} "
          f"{'max gain':>9} {'overhead':>9}")
    out = []
    for fee_mult in (1.0, 1.2):
        for bond_mult in (0.0, 0.5):
            for n in (1, 2, 4):
                fee, bond = fee_mult * S * c, bond_mult * S * c
                best_m, best = 0, -1e18
                for m in range(1, S + 1):
                    g = m * c - p_detect(S, m, n) * (fee + bond)
                    if g > best:
                        best, best_m = g, m
                out.append((fee_mult, bond_mult, n, best_m, best))
                print(f"{fee_mult:8.1f} {bond_mult:9.1f} {n:9d} {best_m:8d} "
                      f"{best:9.3f} {n/S:9.2%}")
    # Monte Carlo spot check of p_detect
    rng = random.Random(3)
    S, m, n, T = 1000, 50, 3, 100_000
    hits = 0
    for _ in range(T):
        skipped = set(rng.sample(range(S), m))
        if any(s in skipped for s in rng.sample(range(S), n)):
            hits += 1
    print(f"  Monte Carlo check: S={S}, m={m}, n={n}: sim {hits/T:.4f} "
          f"vs exact {p_detect(S, m, n):.4f}")
    return out


# ---------------------------------------------------------------- D. collusion
def part_d(trials=200_000, seed=4):
    """Fully faked job passes only if every drawn verifier colludes and no watcher
    challenges. Verifiers drawn by stake; q = colluding stake share; w = chance a
    watcher rechecks a given job."""
    rng = random.Random(seed)
    print("\nD. Fully faked job passes (k verifiers, colluding stake q, watcher rate w)")
    print(f"{'q':>5} {'k':>3} {'w':>5} {'sim':>8} {'formula':>8}")
    for q in (0.10, 0.33, 0.50):
        for k in (2, 3, 5):
            for w in (0.0, 0.10):
                passed = 0
                for _ in range(trials):
                    if all(rng.random() < q for _ in range(k)) and rng.random() >= w:
                        passed += 1
                print(f"{q:5.2f} {k:3d} {w:5.2f} {passed/trials:8.4f} "
                      f"{q**k*(1-w):8.4f}")


# ================================================================ build step 6

# ---------------------------------------------------------------- E. measured lambda
# Step-1 benchmarks (results/, spec/step1-results.md): 8192-token batches of WikiText-2,
# torch.compile, GPU-time lambda. tok_* are tokens per GPU-second for the embed job: mined
# (the job as it runs on this chain), plain (same int8 kernels, no mining) and bf16 (stock).
# noise = sustained noise-mining rate in TMAC/s measured in the same run.
MEASURED = [
    # gpu,  model,        lam_embed, lam_score, tok_mined, tok_plain, tok_bf16, noise
    ("3080", "Qwen3-1.7B", 0.379, 0.369, 21903, 30510, 16245, 81.5),
    ("3080", "Qwen3-4B",   0.488, 0.468,  9583, 12781,  None, 71.4),  # bf16 4B does not fit 10 GB
    ("5090", "Qwen3-1.7B", 0.357, 0.334, 81943, 116019, 62915, 323.9),
    ("5090", "Qwen3-4B",   0.407, 0.394, 35895, 48443, 25725, 320.3),
    ("5090", "Qwen3-8B",   0.492, 0.481, 23016, 29939, 14238, 324.7),
]


def measured_from_results(root):
    """Recompute MEASURED from the raw benchmark files, if they are present."""
    dirs = {"3080": root, "5090": os.path.join(root, "rtx5090")}
    names = {"Qwen3-1.7B": "q17", "Qwen3-4B": "q4", "Qwen3-8B": "q8"}
    out = []
    for gpu, model, *_ in MEASURED:
        def load(mode):
            f = os.path.join(dirs[gpu], f"{names[model]}_{mode}_compiled_t8192.json")
            return json.load(open(f)) if os.path.exists(f) else None
        m, p, b = load("mined"), load("plain"), load("bf16")
        if m is None:
            return None
        lam = lambda job: m[job]["hashed_macs"] / m[job]["gpu_s"] / m["noise_macs_per_gpu_s"]
        out.append((gpu, model, round(lam("embed"), 3), round(lam("score"), 3),
                    round(m["embed"]["tokens_per_gpu_s"]), round(p["embed"]["tokens_per_gpu_s"]),
                    round(b["embed"]["tokens_per_gpu_s"]) if b else None,
                    round(m["noise_macs_per_gpu_s"] / 1e12, 1)))
    return out


def part_e(results_dir=None):
    print("\nE. Measured lambda (compiled runtime, 8192-token batches)")
    if results_dir:
        got = measured_from_results(results_dir)
        print(f"  table {'matches' if got == MEASURED else 'DIFFERS FROM'} {os.path.basename(results_dir)}/*.json" if got
              else f"  ({results_dir} not found; using the table)")
    print(f"{'GPU':>5} {'model':>11} {'lam emb':>8} {'lam scr':>8} {'mining +time':>12} "
          f"{'client/rent int8':>17} {'client/rent bf16':>17} {'GMAC given up/token':>20}")
    for gpu, model, le, ls, tm, tp, tb, noise in MEASURED:
        # Under free entry a GPU-second earns its cost c from mining; a client pays (1-lam)*c per
        # mined GPU-second, so per token it pays (1-lam)*c/tok_mined against c/tok_plain to rent.
        # the fee also carries the verifier and treasury shares (1% + 1% of the fee)
        vs_int8 = (1 - le) / 0.98 * tp / tm
        vs_bf16 = (1 - le) / 0.98 * tb / tm if tb else None
        given_up = (1 - le) * noise * 1e12 / tm / 1e9
        print(f"{gpu:>5} {model:>11} {le:8.3f} {ls:8.3f} {tp/tm-1:12.0%} {vs_int8:17.2f} "
              f"{'-' if vs_bf16 is None else f'{vs_bf16:.2f}':>17} {given_up:20.2f}")
    print("  client/rent < 1: buying a job costs less than renting the same GPU at cost to run it "
          "without mining.")
    print("  GMAC given up per token: hashes a GPU forgoes per pool token; the pool's open race "
          "goes to the lowest first.")


# ---------------------------------------------------------------- F. GPU count responding to revenue
GPU_CLASSES = {
    # measured: noise TMAC/s, lambda and tokens/s for the Qwen3-4B embed job (the pool category)
    # assumed: capex ($, amortized over 3 years) and board power (kW)
    "3080": dict(noise=71.4, lam=0.488, tok=9583, tok_plain=12781, capex=500.0, kw=0.25, count=6000),
    "5090": dict(noise=320.3, lam=0.407, tok=35895, tok_plain=48443, capex=2500.0, kw=0.65, count=1500),
}
HOURS_3Y = 3 * 365 * 24


def make_gpus(seed=11, elec=(0.03, 0.20)):
    """Potential GPUs: each class's count, with electricity prices spread uniformly ($/kWh)."""
    rng = random.Random(seed)
    gpus = []
    for cls, g in GPU_CLASSES.items():
        for _ in range(g["count"]):
            e = rng.uniform(*elec)
            gpus.append(dict(cls=cls, h=g["noise"], lam=g["lam"], tok=g["tok"],
                             cost=g["capex"] / HOURS_3Y + g["kw"] * e))
    return gpus


def entry_equilibrium(Y, rho, demand=0.0, gpus=None, rounds=1500, frac=0.03, seed=12,
                      pool_haircut=0.0, noise_haircut=0.0, client_haircut=0.0, start=None):
    """Free entry and exit. Y = block rewards per hour in dollars; rho = pool share;
    demand = client embed tokens per second (bought at a market price that clears).

    Each round a random 3% of potential GPUs pick the best of: off, noise, pool, client.
      noise : (1-rho) Y h / H                         (hashrate share of the miners' part)
      pool  : (1-rho) Y lam h / H + rho Y tok / T      (budget split by tokens delivered)
      client: (1-rho) Y lam h / H + p tok              (p = client price per token)
    minus the GPU's cost per hour when it is on. Haircuts model risk premia (part G).
    """
    rng = random.Random(seed)
    gpus = gpus or make_gpus()
    n = len(gpus)
    choice = list(start) if start else [0] * n  # 0 off, 1 noise, 2 pool, 3 client
    rate = lambda g, c: 0.0 if c == 0 else (g["h"] if c == 1 else g["lam"] * g["h"])
    H = sum(rate(g, c) for g, c in zip(gpus, choice)) or 1e-9
    T = sum(g["tok"] for g, c in zip(gpus, choice) if c == 2)
    C = sum(g["tok"] for g, c in zip(gpus, choice) if c == 3)
    p = 1.0 / 3600 / 1e4  # $ per token, adjusted below
    for r in range(rounds):
        for i in range(n):
            if rng.random() >= frac:
                continue
            g, c = gpus[i], choice[i]
            H0 = H - rate(g, c)
            T0 = T - (g["tok"] if c == 2 else 0)
            mined_lam = (1 - rho) * Y * g["lam"] * g["h"] / (H0 + g["lam"] * g["h"])
            inc = [0.0,
                   (1 - noise_haircut) * (1 - rho) * Y * g["h"] / (H0 + g["h"]) - g["cost"],
                   (1 - noise_haircut) * mined_lam + (1 - pool_haircut) * rho * Y * g["tok"] / (T0 + g["tok"]) - g["cost"]
                   if rho > 0 else -1e18,
                   (1 - noise_haircut) * mined_lam + (1 - client_haircut) * p * g["tok"] * 3600 - g["cost"]
                   if demand > 0 else -1e18]
            best = max(range(4), key=lambda k: inc[k])
            if inc[best] <= inc[c] + 1e-12:
                continue
            H = H0 + rate(g, best)
            T = T0 + (g["tok"] if best == 2 else 0)
            C += (g["tok"] if best == 3 else 0) - (g["tok"] if c == 3 else 0)
            choice[i] = best
        if demand > 0:  # the client price moves toward clearing the market
            p *= math.exp(0.05 * max(-1.0, min(1.0, (demand - C) / demand)))
    by = {cls: [0, 0, 0, 0] for cls in GPU_CLASSES}
    for g, c in zip(gpus, choice):
        by[g["cls"]][c] += 1
    active = sum(1 for c in choice if c)
    cap = sum(g["h"] for g, c in zip(gpus, choice) if c)
    u_hash = sum(g["h"] for g, c in zip(gpus, choice) if c == 2) / max(cap, 1e-9)
    return dict(H=H, active=active, by=by, pool_tok=T, client_tok=C, price=p, choice=choice, u_hash=u_hash,
                useful=sum(1 for c in choice if c >= 2) / max(1, active),
                cost=sum(g["cost"] for g, c in zip(gpus, choice) if c))


def rent_price_per_token(Y, rho, eq):
    """Renting a GPU costs at least what it earns noise mining. Per plain int8 token (no mining),
    the cheaper class to rent: (1-rho) Y h / H / (tok_plain * 3600)."""
    return min((1 - rho) * Y * g["noise"] / eq["H"] / (g["tok_plain"] * 3600) for g in GPU_CLASSES.values())


def part_f():
    print("\nF. GPU count responding to revenue (free entry; 6000 3080s + 1500 5090s, "
          "electricity $0.03-0.20/kWh)")
    gpus = make_gpus()
    Y = 300.0
    base = entry_equilibrium(Y, 0.0, gpus=gpus)
    print(f"  Y = ${Y:.0f}/h of block rewards. rho = 0: {base['active']} GPUs on "
          f"(3080 {base['by']['3080'][1]}, 5090 {base['by']['5090'][1]}), H = {base['H']/1e3:.1f} PMAC/s, "
          f"cost ${base['cost']:.0f}/h")
    print(f"{'rho':>5} {'demand':>7} {'on':>5} {'3080 noise/pool/cli':>20} {'5090 noise/pool/cli':>20} "
          f"{'u hash':>7} {'formula':>7} {'H kept':>7} {'fixed-N':>8} {'useful':>7} {'pool Mtok/h':>11} {'client/rent':>11}")
    rows = []
    for demand in (0.0, 1.5e6):
        for rho in (0.0, 0.05, 0.10, 0.20, 0.30):
            e = entry_equilibrium(Y, rho, demand=demand, gpus=gpus, start=base["choice"])
            # fixed GPU count: H/N = 1 - u(1-lam); with one lam per class use the 3080 lam (pool goes there)
            fixed = 1 - min(rho, 1 - GPU_CLASSES["3080"]["lam"])
            ratio = e["price"] / rent_price_per_token(Y, rho, e) if demand else float("nan")
            b3, b5 = e["by"]["3080"], e["by"]["5090"]
            rows.append(dict(rho=rho, demand=demand, H=e["H"] / base["H"], useful=e["useful"], on=e["active"],
                             pool_by=(b3[2], b5[2]), client_ratio=ratio, u_hash=e["u_hash"]))
            uf = min(1, rho / (1 - GPU_CLASSES["3080"]["lam"]))
            print(f"{rho:5.2f} {demand/1e6:6.1f}M {e['active']:5d} {b3[1]:>7d}/{b3[2]:>5d}/{b3[3]:>5d} "
                  f"{b5[1]:>7d}/{b5[2]:>5d}/{b5[3]:>5d} {e['u_hash']:7.1%} {uf:7.1%} {e['H']/base['H']:7.3f} {fixed:8.3f} "
                  f"{e['useful']:7.1%} {e['pool_tok']*3600/1e6:11.0f} "
                  f"{'' if demand == 0 else f'{ratio:.2f}':>11}")
    print("  u hash = pool GPUs' share of active noise-mining capacity (the formula's u, rho/(1-lam)); "
          "useful = share of GPUs by count on pool or client work")
    print("  demand in tokens/s of 4B embeddings; client/rent = equilibrium client price per token over")
    print("  renting a GPU for what it would earn noise mining and running the job in int8 without mining")
    # coin price shock: revenue halves, GPUs exit, the pool's share of GPUs holds
    print("  Coin price shock (rho = 0.10): block rewards fall from $300/h to $150/h, then recover")
    eq = entry_equilibrium(Y, 0.10, gpus=gpus, start=base["choice"])
    for Yt in (150.0, 300.0):
        eq2 = entry_equilibrium(Yt, 0.10, gpus=gpus, start=eq["choice"], rounds=800)
        print(f"    Y=${Yt:.0f}/h: {eq2['active']} GPUs on, H = {eq2['H']/base['H']:.3f} of the $300 baseline, "
              f"useful {eq2['useful']:.1%}, cost ${eq2['cost']:.0f}/h")
        eq = eq2
    return rows


# ---------------------------------------------------------------- G. risk-averse providers
def cara_ce(outcomes, a):
    """Certainty equivalent of [(prob, payoff)] under CARA utility with absolute risk aversion a."""
    if a == 0:
        return sum(p * x for p, x in outcomes)
    return -math.log(sum(p * math.exp(-a * x) for p, x in outcomes)) / a


def part_g():
    S, c = 1000, 1.0
    print("\nG1. Risk-averse cheaters (CARA, risk aversion a per unit of the job's cost S*c)")
    print(f"{'fee/cost':>8} {'bond/cost':>9} {'n':>3} {'a*S*c':>6} {'worst m':>8} {'best CE gain':>12}")
    for fee_mult, bond_mult, n in ((1.0, 0.0, 1), (1.0, 0.5, 1), (1.0, 0.5, 2)):
        for a_rel in (0.0, 0.5, 2.0):
            a = a_rel / (S * c)
            F, B = fee_mult * S * c, bond_mult * S * c
            best, bm = -1e18, 0
            for m in range(1, S + 1, 1 if S <= 1000 else 10):
                P = p_detect(S, m, n)
                ce = cara_ce([(1 - P, m * c), (P, m * c - F - B)], a)
                if ce > best:
                    best, bm = ce, m
            print(f"{fee_mult:8.1f} {bond_mult:9.1f} {n:3d} {a_rel:6.1f} {bm:8d} {round(best, 3) + 0.0:12.3f}")
    print("  Risk aversion only makes cheating worse, so bonds sized for a risk-neutral cheater "
          "are conservative.")

    print("\nG2. Honest providers' risk premium per job, as a share of the fee F")
    print("  (false slash with probability eps loses F+B; bond capital locked for job + challenge "
          "window ~1.5 days at 15%/year; CARA risk aversion a per unit of F)")
    print(f"{'bond/F':>7} {'eps':>7} {'a*F':>5} {'capital':>8} {'risk':>8} {'total':>8}")
    out = {}
    for bond in (0.25, 0.5, 1.0, 2.0):
        for eps in (0.0, 1e-4, 1e-3):
            for aF in (1.0,):
                capital = bond * 0.15 * 1.5 / 365
                F = 1.0
                # premium = expected payoff - certainty equivalent, payoff 0 normally, -(F+B) on a false slash
                ce = cara_ce([(1 - eps, 0.0), (eps, -(F + bond * F))], aF)
                risk = max(0.0, -ce)
                out[(bond, eps, aF)] = capital + risk
                if True:
                    print(f"{bond:7.2f} {eps:7.0e} {aF:5.1f} {capital:8.4%} {risk:8.4%} {capital+risk:8.4%}")
    print("  With a bit-exact runtime (eps ~ 0) a bond costs an honest provider only capital: "
          "0.06% of the fee at bond = F.")

    print("\nG3. Payout variance: noise mining is a lottery, pool units are paid every epoch")
    blocks_day = 1440
    for share in (1e-4, 1e-3):
        cv_noise = 1 / math.sqrt(share * blocks_day)  # solo miner, Poisson blocks per day
        print(f"  a GPU with {share:.2%} of hashrate: daily solo-mining income CV = {cv_noise:.2f}; "
              f"pool-unit income CV ~ 0 (equal split of each epoch's budget, 24 epochs/day)")
    print("  A pool GPU still mines its lam share (a lottery), but its pool income is a near-deterministic")
    print("  share of each epoch's budget. Solo miners pay mining pools (~1-2%) to remove variance; the")
    print("  same fee on the pool GPU's mined part is smaller, so risk aversion tilts slightly toward pool work:")
    gpus = make_gpus()
    print(f"{'scenario':>44} {'3080 pool':>9} {'5090 pool':>9} {'u hash':>7} {'H kept':>7}")
    for label, kw in (("risk-neutral", {}),
                      ("mining-pool fee 2% on mined income", dict(noise_haircut=0.02)),
                      ("+ false-slash premium 0.5% on pool income", dict(noise_haircut=0.02, pool_haircut=0.005)),
                      ("pool income discounted 10% (strong aversion)", dict(pool_haircut=0.10))):
        nk = {k: v for k, v in kw.items() if k == "noise_haircut"}
        base = entry_equilibrium(300.0, 0.0, gpus=gpus, **nk)  # same mining-pool fee, no pool
        e = entry_equilibrium(300.0, 0.10, gpus=gpus, start=base["choice"], **kw)
        print(f"{label:>44} {e['by']['3080'][2]:9d} {e['by']['5090'][2]:9d} {e['u_hash']:7.1%} {e['H']/base['H']:7.3f}")
    return out


# ---------------------------------------------------------------- H. block withholding
def reroll_value(x, alpha, V, L, R, withhold=True):
    """Expected gain of a provider who committed a fully faked job, relative to honest work.

    x = chance a fresh committee draw passes the fake (all seats colluding, no watcher catch),
    alpha = its hashrate share, V = gain if it passes, L = loss if caught (withheld fee + bond),
    R = block reward. When it mines the seed block and the committee is bad it may discard the
    block (losing R) and wait for a new seed: U = [(1-a)(xV-(1-x)L) + a x V - a(1-x)R] / (1-a(1-x)).
    """
    once = x * V - (1 - x) * L
    if not withhold:
        return once
    u = ((1 - alpha) * once + alpha * x * V - alpha * (1 - x) * R) / (1 - alpha * (1 - x))
    return max(once, u)


def reroll_pass_prob(x, alpha):
    return x / (1 - alpha * (1 - x))


def simulate_reroll(q, k, w, alpha, V, L, R, trials=100_000, seed=21):
    """Monte Carlo of the seed block race with the optimal withhold rule."""
    rng = random.Random(seed)
    x = q ** k * (1 - w)
    withhold = reroll_value(x, alpha, V, L, R) - R > -L  # discarding beats accepting a bad committee
    gain, passed, discarded = 0.0, 0, 0
    for _ in range(trials):
        while True:
            mine = rng.random() < alpha
            ok = all(rng.random() < q for _ in range(k)) and rng.random() >= w
            if ok:
                gain += V
                passed += 1
                break
            if mine and withhold:
                gain -= R
                discarded += 1
                continue
            gain -= L
            break
    return passed / trials, discarded / trials, gain / trials


def reroll_threshold(x, alpha, V, L):
    """Smallest job value, in block rewards (job cost / R), at which discarding pays."""
    for e in range(-40, 121):
        jr = 10 ** (e / 20)
        if reroll_value(x, alpha, V, L, 1 / jr) > reroll_value(x, alpha, V, L, 1 / jr, False) + 1e-12:
            return jr
    return None


def part_h():
    print("\nH. Block withholding: re-rolling the committee by discarding the seed block")
    print("  A provider that also mines (hashrate share alpha) sees the committee its own seed block would")
    print("  draw, and can discard the block when an honest verifier is in it. Fully faked job; loss if")
    print("  caught L = 1.5 job costs (fee + 50% bond); colluding stake q = 1/3; watchers check w = 10%.")
    print("  V = the attacker's gain if the fake passes, in job costs (1 = the compute it saved; more for")
    print("  tampering with chosen outputs). Job size in block rewards R.")
    print(f"{'k':>3} {'alpha':>6} {'V':>4} {'job/R':>6} {'pass':>7} {'reroll':>7} {'sim':>7} "
          f"{'discards':>8} {'gain/job':>9} {'sim':>8}")
    rows = []
    q, w, L = 1 / 3, 0.1, 1.5
    seed = 30
    for k in (2, 3, 5):
        for alpha in (0.3,):
            for V in (1.0, 10.0):
                for jobR in (1.0, 10.0, 100.0):
                    R = 1 / jobR
                    x = q ** k * (1 - w)
                    u = reroll_value(x, alpha, V, L, R)
                    rr = u > reroll_value(x, alpha, V, L, R, withhold=False) + 1e-12
                    p = reroll_pass_prob(x, alpha) if rr else x
                    seed += 1
                    ps, ds, gs = simulate_reroll(q, k, w, alpha, V, L, R, trials=200_000, seed=seed)
                    rows.append(dict(k=k, alpha=alpha, V=V, jobR=jobR, x=x, p=p, gain=u))
                    print(f"{k:3d} {alpha:6.2f} {V:4.0f} {jobR:6.0f} {x:7.4f} {p:7.4f} {ps:7.4f} "
                          f"{ds:8.3f} {u:+9.3f} {gs:+8.3f}")
    print("  pass = q^k (1-w) without re-rolls; reroll = x / (1 - alpha (1-x)) when discarding pays.")
    print("\n  Job size (in block rewards) from which re-rolling pays:")
    print(f"{'q':>5} {'k':>3} {'alpha':>6} {'V=1':>9} {'V=10':>9}")
    for qq in (1 / 3, 0.5):
        for k in (2, 3, 5):
            for alpha in (0.1, 0.3):
                x = qq ** k * 0.9
                t1, t10 = reroll_threshold(x, alpha, 1, L), reroll_threshold(x, alpha, 10, L)
                fmt = lambda t: f"{t:.1f} R" if t else "never"
                print(f"{qq:5.2f} {k:3d} {alpha:6.2f} {fmt(t1):>9} {fmt(t10):>9}")
    print("  Re-rolling multiplies the pass chance by at most 1/(1-alpha) and never makes a job that")
    print("  loses money on average (V = 1) profitable; it matters for tampering (V >> 1) on large jobs.")
    print("  A longer draw delay does not help: whoever mines the seed block can always discard it.")
    # private sortition: each staker learns privately (VRF) whether it is drawn; seats ~ Poisson(s).
    # The miner sees only its colluders' seats, which are independent of the honest ones, so a
    # discard cannot raise the pass chance exp(-s (1-q)).
    print("\n  Private (VRF) sortition is immune to re-rolls but its committee size varies; expected")
    print("  seats needed to match a fixed committee of k (no re-rolls):")
    rng = random.Random(40)
    for qq in (1 / 3, 0.5):
        for k in (2, 3, 5):
            s_eq = k * math.log(1 / qq) / (1 - qq)
            # Monte Carlo: honest seats ~ Poisson(s (1-q)); attacker discards on its own seat count only
            trials, passed = 100_000, 0
            for _ in range(trials):
                lam = s_eq * (1 - qq)
                # Poisson sample by inversion
                n_h, pr, u = 0, math.exp(-lam), rng.random()
                cum = pr
                while u > cum:
                    n_h += 1
                    pr *= lam / n_h
                    cum += pr
                passed += n_h == 0
            print(f"    q={qq:.2f} k={k}: {s_eq:4.1f} expected seats (+{s_eq/k-1:.0%} verifier pay); "
                  f"pass {passed/trials:.4f} vs q^k {qq**k:.4f}")
    return rows


# ---------------------------------------------------------------- I. retune
def min_k(job_R, q=1 / 3, alpha=0.3, V=10.0, L=1.5, w=0.1):
    """Smallest committee for which a fully faked job does not pay, even for a tampering
    attacker (gain V job costs) that mines a share alpha and re-rolls by withholding."""
    for k in range(2, 17):
        if reroll_value(q ** k * (1 - w), alpha, V, L, 1 / job_R) <= 0:
            return k
    return None


def part_i(f_rows, g_out, h_rows):
    print("\nI. Retune with measured lambda")
    lam4 = [r[2] for r in MEASURED if r[1] == "Qwen3-4B"]
    print(f"  lambda of the v1 pool job (Qwen3-4B embeddings): {min(lam4):.3f}-{max(lam4):.3f}")
    print("\n  rho (free entry, part F; u hash = pool share of mining capacity):")
    print(f"{'rho':>5} {'u formula':>9} {'u hash':>7} {'GPUs useful':>11} {'H kept':>7} {'1-rho':>6}")
    for r in f_rows:
        if r["demand"] or r["rho"] == 0:
            continue
        u_f = min(1, r["rho"] / (1 - GPU_CLASSES["3080"]["lam"]))
        print(f"{r['rho']:5.2f} {u_f:9.1%} {r['u_hash']:7.1%} {r['useful']:11.1%} {r['H']:7.3f} {1-r['rho']:6.3f}")
    print("\n  k: smallest committee that keeps a fully faked job unprofitable (colluding stake 1/3,")
    print("  attacker hashrate 0.3 re-rolling by withholding, watchers 10%, loss 1.5 job costs):")
    print(f"{'job size':>9} {'V=1':>4} {'V=10':>5} {'V=30':>5}")
    ks = {}
    for jr in (0.1, 0.5, 1.0, 3.0, 10.0, 100.0):
        ks[jr] = (min_k(jr, V=1), min_k(jr, V=10), min_k(jr, V=30))
        print(f"{jr:7.1f} R {ks[jr][0]:4d} {ks[jr][1]:5d} {ks[jr][2]:5d}")
    print("""
  Recommendations (spec/step6-results.md):
  rho  = 10%, unchanged. With measured lambda (~0.45) the pool takes rho/(1-lambda) ~ 20% of mining
         capacity (~40% of GPUs by count, the ones that give up the fewest hashes per token) at a
         hashrate cost of rho; free entry keeps that cost at rho until the cheap class saturates,
         then lowers it. Saturation is at rho ~ 0.5.
  k    = 2 minimum for jobs below one block reward; 3 from one block reward up (consensus rule
         JOB_POST: fee >= LargeJobFee needs k >= MinKLarge). Clients exposed to tampering worth
         more than ~10x the job still buy more (k = 5 covers 30x at 100 R).
  n    = 2, unchanged: deterrence of skipping needs one sample once F + B >= S c (part C), and
         risk aversion only strengthens it (G1).
  bond = 50% of the fee, unchanged: with a bit-exact runtime it costs an honest provider ~0.03% of
         the fee (G2); a 1e-3 false-slash rate would cost ~0.4%, so bonds are not the place to save.""")
    return ks


PARTS = {"A": part_a, "B": part_b, "C": part_c, "D": part_d}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="ABCDEFGHI")
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    a = ap.parse_args()
    f_rows = g_out = h_rows = None
    for part in a.parts.upper():
        if part in PARTS:
            PARTS[part]()
        elif part == "E":
            part_e(os.path.normpath(a.results) if os.path.isdir(a.results) else None)
        elif part == "F":
            f_rows = part_f()
        elif part == "G":
            g_out = part_g()
        elif part == "H":
            h_rows = part_h()
        elif part == "I":
            part_i(f_rows or part_f(), g_out, h_rows)
    print()
