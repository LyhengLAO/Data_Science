"""Optimisation budgétaire + validation A/B sur le jeu de test.

python -m uplift.run_optimization
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from ab_test import compare_policies, policy_value
from config import abspath, load_config
from optimization import evaluate_policy_model_based, naive_top_uplift, optimize_allocation, random_policy
from train import ensure_scores

log = logging.getLogger(__name__)


def main(config: str | None = None, budget: float | None = None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(config)
    arms, ctrl, biz = cfg["arms"], cfg["control_label"], cfg["business"]
    value, costs = biz["value_per_conversion"], biz["cost_per_contact"]
    budget = budget if budget is not None else biz["budget"]

    scores = ensure_scores(cfg)
    uplift = scores[[f"uplift_{a}" for a in arms]].rename(columns=lambda c: c.removeprefix("uplift_"))

    pol_opt, s_opt = optimize_allocation(
        uplift, value, costs, budget, ctrl,
        max_per_arm=biz.get("max_per_arm"), time_limit=biz.get("solver_time_limit", 60),
    )
    policies = {
        "optimized_lp": pol_opt,
        "naive_top_uplift": naive_top_uplift(uplift, costs, budget, ctrl),
        "random": random_policy(uplift, costs, budget, ctrl, cfg["seed"]),
    }

    report = {"budget": budget, "value_per_conversion": value, "policies": {}, "ab_tests": {}}
    for name, pol in policies.items():
        report["policies"][name] = {
            "model_based": evaluate_policy_model_based(pol, uplift, value, costs, ctrl),
            "ipw_unbiased": policy_value(scores, pol, value, costs, ctrl),
        }
        log.info("%-18s contacts=%6d | profit IPW=%9.2f", name,
                 report["policies"][name]["ipw_unbiased"]["n_contacted"],
                 report["policies"][name]["ipw_unbiased"]["total_profit_ipw"])
    for base in ("random", "naive_top_uplift"):
        report["ab_tests"][f"optimized_lp_vs_{base}"] = compare_policies(
            scores, policies[base], pol_opt, value, costs, ctrl)

    out = abspath("reports")
    out.mkdir(exist_ok=True)
    (out / "optimization.json").write_text(json.dumps(report, indent=2))
    alloc = scores[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]].copy()
    alloc["action"] = pol_opt
    for a in arms:
        alloc[f"uplift_{a}"] = uplift[a]
    alloc.to_parquet(abspath(cfg["data"]["processed_dir"]) / "allocation.parquet")
    return report


if __name__ == "__main__":
    main()
