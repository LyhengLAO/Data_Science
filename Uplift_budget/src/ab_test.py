"""Validation statistique : analyse de puissance + comparaison de politiques.

Comme le traitement est randomisé (propensity connu), on estime la valeur d'une politique
sans biais par pondération inverse (IPW) : pour une politique π, le taux de conversion
attendu est mean_i( 1[T_i = π(x_i)] / P(T = π(x_i)) * y_i ).
La comparaison A vs B est un test apparié sur les contributions par client.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def required_sample_size(p_baseline: float, p_expected: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Taille d'échantillon PAR GROUPE pour confirmer en live un écart de taux de conversion."""
    es = abs(proportion_effectsize(p_expected, p_baseline))
    return int(np.ceil(NormalIndPower().solve_power(es, alpha=alpha, power=power, ratio=1.0)))


def min_detectable_effect(n_per_group: int, p_baseline: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Plus petit écart de taux de conversion détectable avec n par groupe."""
    es = NormalIndPower().solve_power(nobs1=n_per_group, alpha=alpha, power=power, ratio=1.0)
    lo, hi = 0.0, 1.0 - p_baseline
    for _ in range(60):  # inversion numérique de l'effet de Cohen h
        mid = (lo + hi) / 2
        if proportion_effectsize(p_baseline + mid, p_baseline) < es:
            lo = mid
        else:
            hi = mid
    return float(hi)


def _ipw_terms(df: pd.DataFrame, policy: pd.Series, propensity: pd.Series) -> np.ndarray:
    pol = policy.reindex(df.index)
    match = (df.treatment.to_numpy() == pol.to_numpy()).astype(float)
    return match / pol.map(propensity).to_numpy() * df.y.to_numpy()


def policy_value(df, policy, value, costs, control) -> dict:
    """Valeur estimée (IPW) d'une politique : conversions et profit par client."""
    prop = df.treatment.value_counts(normalize=True)
    terms = _ipw_terms(df, policy, prop)
    cost = policy.reindex(df.index).map(lambda a: costs.get(a, 0.0)).to_numpy()
    return {
        "conv_rate_ipw": float(terms.mean()),
        "profit_per_customer_ipw": float((value * terms - cost).mean()),
        "total_profit_ipw": float((value * terms - cost).sum()),
        "total_cost": float(cost.sum()),
        "n_contacted": int((policy != control).sum()),
    }


def compare_policies(df, policy_a, policy_b, value, costs, control, alpha: float = 0.05) -> dict:
    """Test apparié B vs A sur le taux de conversion et sur le profit (IPW)."""
    prop = df.treatment.value_counts(normalize=True)
    ta, tb = _ipw_terms(df, policy_a, prop), _ipw_terms(df, policy_b, prop)
    ca = policy_a.reindex(df.index).map(lambda a: costs.get(a, 0.0)).to_numpy()
    cb = policy_b.reindex(df.index).map(lambda a: costs.get(a, 0.0)).to_numpy()

    def paired(d):
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        z = d.mean() / se if se > 0 else 0.0
        h = stats.norm.ppf(1 - alpha / 2) * se
        return {
            "diff_mean": float(d.mean()),
            "ci_low": float(d.mean() - h),
            "ci_high": float(d.mean() + h),
            "z": float(z),
            "p_value": float(2 * (1 - stats.norm.cdf(abs(z)))),
            "significant": bool(abs(z) > stats.norm.ppf(1 - alpha / 2)),
        }

    res = {"conversion_rate": paired(tb - ta), "profit_per_customer": paired((value * tb - cb) - (value * ta - ca))}
    pa, pb = float(ta.mean()), float(tb.mean())
    res["conv_rate_a"], res["conv_rate_b"] = pa, pb
    if pb != pa:
        res["n_per_group_live_test"] = required_sample_size(max(min(pa, pb), 1e-6), max(pa, pb))
    return res
