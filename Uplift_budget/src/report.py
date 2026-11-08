"""Génère reports/RESULTS.md à partir des JSON produits (aucun chiffre écrit à la main).

python -m uplift.report
"""
import json

from config import abspath


def main() -> None:
    m = json.loads((abspath("reports") / "metrics.json").read_text())
    o = json.loads((abspath("reports") / "optimization.json").read_text())
    L = [f"# Résultats\n\n_Source des données : **{m['data_source']}** — train={m['n_train']}, test={m['n_test']}_\n"]
    L.append(f"## Modèles (test) — meilleur learner choisi sur validation : **{m['best_learner']}**\n")
    L.append("| Modèle | Qini moyen | Qini Hommes | Qini Femmes |\n|---|---|---|---|")
    for name, r in m["test"].items():
        arms = [k for k in r if k != "mean_qini_coef"]
        L.append(f"| {name} | {r['mean_qini_coef']:.6f} | " + " | ".join(f"{r[a]['qini_coef']:.6f}" for a in arms) + " |")
    L.append(f"\n## Politiques de ciblage (budget {o['budget']} EUR, IPW sans biais)\n")
    L.append("| Politique | Contacts | Coût | Profit total |\n|---|---|---|---|")
    for name, r in o["policies"].items():
        i = r["ipw_unbiased"]
        L.append(f"| {name} | {i['n_contacted']} | {i['total_cost']:.2f} | {i['total_profit_ipw']:.2f} |")
    L.append("\n## Tests A/B (politique optimisée vs baseline)\n")
    L.append("| Comparaison | Δ profit/client | IC 95% | p-value | Significatif |\n|---|---|---|---|---|")
    for name, r in o["ab_tests"].items():
        p = r["profit_per_customer"]
        L.append(f"| {name} | {p['diff_mean']:.5f} | [{p['ci_low']:.5f}, {p['ci_high']:.5f}] | {p['p_value']:.4f} | {p['significant']} |")
    (abspath("reports") / "RESULTS.md").write_text("\n".join(L) + "\n")
    print("reports/RESULTS.md généré")


if __name__ == "__main__":
    main()
