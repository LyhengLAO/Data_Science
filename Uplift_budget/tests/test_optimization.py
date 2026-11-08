import numpy as np
import pandas as pd

from src.optimization import (evaluate_policy_model_based, naive_top_uplift,
                                 net_gain_matrix, optimize_allocation, random_policy)

CTRL = "No E-Mail"
COSTS = {"A": 0.10, "B": 0.10}


def _uplift(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"A": rng.normal(0.004, 0.006, n), "B": rng.normal(0.004, 0.006, n)})


def test_budget_respected_and_one_action_per_client():
    U = _uplift()
    pol, s = optimize_allocation(U, 30.0, COSTS, budget=100.0, control=CTRL)
    assert s["status"] == "Optimal"
    assert s["total_cost"] <= 100.0 + 1e-6
    assert len(pol) == len(U) and set(pol.unique()) <= {CTRL, "A", "B"}


def test_only_positive_net_gain_selected():
    U = _uplift()
    pol, _ = optimize_allocation(U, 30.0, COSTS, budget=10_000.0, control=CTRL)
    gain = net_gain_matrix(U, 30.0, COSTS)
    for i, a in pol[pol != CTRL].items():
        assert gain.at[i, a] > 0


def test_optimal_beats_baselines():
    U = _uplift()
    opt, s_opt = optimize_allocation(U, 30.0, COSTS, 100.0, CTRL)
    naive = naive_top_uplift(U, COSTS, 100.0, CTRL)
    rnd = random_policy(U, COSTS, 100.0, CTRL, 0)
    g = lambda p: evaluate_policy_model_based(p, U, 30.0, COSTS, CTRL)["expected_net_gain"]  # noqa: E731
    assert g(opt) >= g(naive) - 1e-6
    assert g(opt) > g(rnd)


def test_arm_cap_constraint():
    U = pd.DataFrame({"A": np.full(500, 0.02), "B": np.full(500, 0.03)})  # B domine partout
    pol, s = optimize_allocation(U, 30.0, COSTS, 1000.0, CTRL, max_per_arm={"B": 100})
    assert (pol == "B").sum() <= 100
    assert (pol == "A").sum() > 0            # le surplus bascule sur A


def test_no_positive_gain_returns_empty():
    U = pd.DataFrame({"A": np.full(50, 0.0001), "B": np.full(50, 0.0001)})
    pol, s = optimize_allocation(U, 30.0, COSTS, 100.0, CTRL)
    assert (pol == CTRL).all() and s["n_contacted"] == 0
