"""Checks of sim.py's step-6 parts against their formulas and the measured results."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def test_measured_table_matches_results():
    if not os.path.isdir(os.path.join(RESULTS, "rtx5090")):
        return  # sim.py is also shipped alone, without the raw results
    assert sim.measured_from_results(RESULTS) == sim.MEASURED


def test_free_entry_pool_share_is_the_formula_in_hash_terms():
    gpus = sim.make_gpus()
    base = sim.entry_equilibrium(300.0, 0.0, gpus=gpus)
    e = sim.entry_equilibrium(300.0, 0.10, gpus=gpus, start=base["choice"])
    lam = sim.GPU_CLASSES["3080"]["lam"]
    assert abs(e["u_hash"] - 0.10 / (1 - lam)) < 0.005
    assert abs(e["H"] / base["H"] - 0.90) < 0.005
    # the pool takes the GPUs that give up the fewest hashes per token: all 3080s, no 5090s
    assert e["by"]["5090"][2] == 0 and e["by"]["3080"][2] > 0


def test_risk_haircut_lowers_pool_share_proportionally():
    gpus = sim.make_gpus()
    base = sim.entry_equilibrium(300.0, 0.0, gpus=gpus)
    e = sim.entry_equilibrium(300.0, 0.10, gpus=gpus, start=base["choice"], pool_haircut=0.10)
    lam = sim.GPU_CLASSES["3080"]["lam"]
    assert abs(e["u_hash"] - 0.10 * 0.9 / (1 - lam)) < 0.005


def test_cara_reduces_to_expectation_and_penalizes_risk():
    out = [(0.3, 5.0), (0.7, -1.0)]
    assert abs(sim.cara_ce(out, 0.0) - (0.3 * 5 - 0.7)) < 1e-12
    assert sim.cara_ce(out, 0.5) < sim.cara_ce(out, 0.0)


def test_reroll_formula_matches_monte_carlo():
    q, k, w, alpha, V, L = 1 / 3, 2, 0.1, 0.3, 10.0, 1.5
    for jobR in (1.0, 10.0):
        R = 1 / jobR
        x = q ** k * (1 - w)
        u = sim.reroll_value(x, alpha, V, L, R)
        p = sim.reroll_pass_prob(x, alpha)
        ps, _, gs = sim.simulate_reroll(q, k, w, alpha, V, L, R, trials=200_000, seed=5)
        se = math.sqrt(p * (1 - p) / 200_000)
        assert abs(ps - p) < 4 * se
        assert abs(gs - u) < 0.03


def test_reroll_never_rescues_a_money_losing_fake():
    # V = 1: the fake gains only the compute it saved; with fee + bond at stake it never pays
    for k in (2, 3, 5):
        for jobR in (0.1, 1, 10, 1000):
            assert sim.reroll_value((1 / 3) ** k * 0.9, 0.45, 1.0, 1.5, 1 / jobR) < 0


def test_retuned_k():
    assert sim.min_k(1.0, V=10) == 2
    assert sim.min_k(10.0, V=10) == 3
    assert sim.min_k(100.0, V=30) == 3
