"""Optimisation de l'allocation budgétaire (multiple-choice knapsack, PuLP/CBC).

Pour chaque client i et chaque action k (email Hommes / Femmes) :
    gain net g_ik = uplift_ik * valeur_conversion - coût_k
Variables x_ik in {0,1}. On maximise sum g_ik x_ik sous :
    sum_k x_ik <= 1                (au plus une action par client)
    sum c_k x_ik <= budget         (budget total)
    sum_i x_ik <= cap_k            (optionnel : capacité par bras)
    sum x_ik <= max_contacts       (optionnel)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pulp


def net_gain_matrix(uplift: pd.DataFrame, value: float, costs: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({a: uplift[a] * value - costs[a] for a in uplift.columns})


def optimize_allocation(
    uplift: pd.DataFrame,
    value: float,
    costs: dict[str, float],
    budget: float,
    control: str,
    max_per_arm: dict[str, int] | None = None,
    max_contacts: int | None = None,
    time_limit: int = 60,
) -> tuple[pd.Series, dict]:
    """Retourne (policy: Series index=client -> action|control, summary)."""
    gain = net_gain_matrix(uplift, value, costs)
    arms = list(uplift.columns)
    policy = pd.Series(control, index=uplift.index, dtype=object)

    # Élagage : une action à gain net <= 0 n'est jamais choisie par un maximiseur.
    cand = [(i, a) for a in arms for i in gain.index[gain[a].to_numpy() > 0]]
    if not cand:
        return policy, _summary(policy, uplift, value, costs, control, "NoPositiveGain")

    prob = pulp.LpProblem("budget_allocation", pulp.LpMaximize)
    x = {(i, a): pulp.LpVariable(f"x_{n}", cat="Binary") for n, (i, a) in enumerate(cand)}
    prob += pulp.lpSum(gain.at[i, a] * v for (i, a), v in x.items())
    by_client: dict = {}
    for (i, a), v in x.items():
        by_client.setdefault(i, []).append(v)
    for vs in by_client.values():
        if len(vs) > 1:
            prob += pulp.lpSum(vs) <= 1
    prob += pulp.lpSum(costs[a] * v for (i, a), v in x.items()) <= budget
    for a, cap in (max_per_arm or {}).items():
        prob += pulp.lpSum(v for (i, b), v in x.items() if b == a) <= cap
    if max_contacts is not None:
        prob += pulp.lpSum(x.values()) <= max_contacts

    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, gapRel=1e-4))
    status = pulp.LpStatus[prob.status]
    for (i, a), v in x.items():
        if (v.value() or 0) > 0.5:
            policy.at[i] = a
    return policy, _summary(policy, uplift, value, costs, control, status)


def naive_top_uplift(uplift, costs, budget, control, value=None):
    """Baseline : meilleur bras par client, classement par uplift brut, on contacte
    jusqu'à épuisement du budget (ne tient pas compte du coût ni des contraintes)."""
    best = uplift.idxmax(axis=1)
    score = uplift.max(axis=1)
    policy = pd.Series(control, index=uplift.index, dtype=object)
    spent = 0.0
    for i in score[score > 0].sort_values(ascending=False).index:
        c = costs[best[i]]
        if spent + c > budget:
            break
        policy.at[i], spent = best[i], spent + c
    return policy


def random_policy(uplift, costs, budget, control, seed: int = 0):
    """Baseline : clients et bras tirés au hasard jusqu'à épuisement du budget."""
    rng = np.random.default_rng(seed)
    arms = list(uplift.columns)
    policy = pd.Series(control, index=uplift.index, dtype=object)
    spent = 0.0
    for i in rng.permutation(uplift.index.to_numpy()):
        a = arms[rng.integers(len(arms))]
        if spent + costs[a] > budget:
            break
        policy.at[i], spent = a, spent + costs[a]
    return policy


def _summary(policy, uplift, value, costs, control, status) -> dict:
    sel = policy[policy != control]
    exp_conv = float(sum(uplift.at[i, a] for i, a in sel.items()))
    cost = float(sum(costs[a] for a in sel))
    return {
        "status": status,
        "n_contacted": int(len(sel)),
        "contacts_per_arm": {k: int(v) for k, v in sel.value_counts().items()},
        "total_cost": round(cost, 2),
        "expected_incremental_conversions": round(exp_conv, 2),
        "expected_net_gain": round(exp_conv * value - cost, 2),
    } 


def evaluate_policy_model_based(policy, uplift, value, costs, control) -> dict:
    return _summary(policy, uplift, value, costs, control, "evaluated")
