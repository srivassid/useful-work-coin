"""Useful-Work Coin v3 -- economics and attack simulator.

Checks four claims in the design doc:
  A. Public-good pool equilibrium: share of GPUs on pool jobs u = rho / (1 - lambda),
     and effective hashrate falls by the factor (1 - rho).
  B. Any per-job premium is captured by self-dealing unless it is <= what the job burns.
  C. Withheld fee + a few random stage samples deter a provider that skips stages.
  D. Probability that a fully faked job passes, given colluding verifier stake.

Run: python3 sim.py   (stdlib only, deterministic seeds)
"""
import math
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


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()
