import numpy as np
import pandas as pd

from src.ab_test import compare_policies, min_detectable_effect, policy_value, required_sample_size

CTRL = "No E-Mail"


def test_sample_size_monotonic():
    big = required_sample_size(0.01, 0.012)
    small = required_sample_size(0.01, 0.02)
    assert big > small > 0
    assert required_sample_size(0.01, 0.012, power=0.9) > big


def test_mde_consistent_with_sample_size():
    n = required_sample_size(0.01, 0.013)
    assert abs(min_detectable_effect(n, 0.01) - 0.003) < 5e-4


def _randomized_df(n=60_000, seed=0):
    """Effet de l'email = +4 pts si x=1, 0 sinon. Traitement randomisé 50/50."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, n)
    t = rng.choice([CTRL, "A"], n)
    y = (rng.random(n) < 0.05 + 0.04 * x * (t == "A")).astype(int)
    return pd.DataFrame({"x": x, "treatment": t, "y": y})


def test_ipw_detects_better_targeting_and_not_equal_policies():
    df = _randomized_df()
    costs = {"A": 0.1}
    smart = pd.Series(np.where(df.x == 1, "A", CTRL), index=df.index)
    rand = pd.Series(np.where(np.random.default_rng(1).random(len(df)) < 0.5, "A", CTRL), index=df.index)
    r = compare_policies(df, rand, smart, 30.0, costs, CTRL)
    assert r["profit_per_customer"]["diff_mean"] > 0 and r["profit_per_customer"]["significant"]
    same = compare_policies(df, smart, smart, 30.0, costs, CTRL)
    assert same["profit_per_customer"]["diff_mean"] == 0 and not same["profit_per_customer"]["significant"]


def test_ipw_unbiased_for_treat_all():
    df = _randomized_df()
    all_a = pd.Series("A", index=df.index)
    est = policy_value(df, all_a, 30.0, {"A": 0.1}, CTRL)["conv_rate_ipw"]
    truth = 0.05 + 0.04 * 0.5
    assert abs(est - truth) < 0.005
