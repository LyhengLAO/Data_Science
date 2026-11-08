import numpy as np

from src.evaluation import qini_coefficient, qini_curve, uplift_at_k, uplift_by_decile


def _toy(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, n)
    tau = rng.random(n)                       # effet individuel connu in [0,1]*0.2
    y = (rng.random(n) < 0.05 + 0.2 * tau * t).astype(int)
    return y, t, tau


def test_qini_endpoints():
    y, t, tau = _toy()
    x, q = qini_curve(y, t, tau)
    assert x[0] == 0 and x[-1] == 1 and q[0] == 0
    # q final = effet incrémental global estimé
    n_t, n_c = t.sum(), (1 - t).sum()
    assert np.isclose(q[-1], y[t == 1].sum() - y[t == 0].sum() * n_t / n_c)


def test_good_score_beats_random_and_inverted():
    y, t, tau = _toy()
    rng = np.random.default_rng(1)
    good, rand, inv = qini_coefficient(y, t, tau), qini_coefficient(y, t, rng.random(len(y))), qini_coefficient(y, t, -tau)
    assert good > 0 > inv
    assert good > rand


def test_uplift_at_k_and_deciles():
    y, t, tau = _toy()
    assert uplift_at_k(y, t, tau, 0.2) > uplift_at_k(y, t, -tau, 0.2)
    d = uplift_by_decile(y, t, tau)
    assert len(d) == 10 and d.observed_uplift.iloc[0] > d.observed_uplift.iloc[-1]
