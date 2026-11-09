"""Dashboard Streamlit : modèles, segments, simulateur d'allocation, résultats A/B.

streamlit run streamlit_app/app.py        (après `make all`)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import load_config  # noqa: E402
from src.optimization import (evaluate_policy_model_based, naive_top_uplift,  # noqa: E402
                                 optimize_allocation, random_policy)

st.set_page_config(page_title="Uplift Budget Optimizer", page_icon="📈", layout="wide")
cfg = load_config()
ARMS, CTRL, BIZ = cfg["arms"], cfg["control_label"], cfg["business"]


@st.cache_data
def load_artifacts():
    scores = ROOT / cfg["data"]["processed_dir"] / "test_scores.parquet"
    metrics, opt = ROOT / "reports" / "metrics.json", ROOT / "reports" / "optimization.json"
    if not (scores.exists() and metrics.exists() and opt.exists()):
        return None
    return pd.read_parquet(scores), json.loads(metrics.read_text()), json.loads(opt.read_text())


@st.cache_data(show_spinner="Optimisation en cours…")
def run_lp(budget, value, c1, c2, cap2):
    scores = pd.read_parquet(ROOT / cfg["data"]["processed_dir"] / "test_scores.parquet")
    U = scores[[f"uplift_{a}" for a in ARMS]].rename(columns=lambda c: c.removeprefix("uplift_"))
    costs = {ARMS[0]: c1, ARMS[1]: c2}
    caps = {ARMS[1]: cap2} if cap2 else None
    pol, s = optimize_allocation(U, value, costs, budget, CTRL, max_per_arm=caps, time_limit=30)
    base = {
        "LP": s,
        "Naïf top-uplift": evaluate_policy_model_based(naive_top_uplift(U, costs, budget, CTRL), U, value, costs, CTRL),
        "Aléatoire": evaluate_policy_model_based(random_policy(U, costs, budget, CTRL, 0), U, value, costs, CTRL),
    }
    return pol, base


st.title("📈 Uplift Budget Optimizer")
st.caption("À qui envoyer la campagne pour maximiser le profit **incrémental** sous contrainte de budget ?")

art = load_artifacts()
if art is None:
    st.error("Artefacts introuvables. Lancer d'abord : `make all` (entraînement + optimisation).")
    st.stop()
scores, metrics, opt = art
if metrics["data_source"] != "hillstrom":
    st.warning(f"Données **{metrics['data_source']}** : résultats de démonstration uniquement.")

tab1, tab2, tab3, tab4 = st.tabs(["Vue d'ensemble", "Segments clients", "Simulateur d'allocation", "Résultats A/B"])

with tab1:
    best = metrics["best_learner"]
    t = metrics["test"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Meilleur learner (validation)", best)
    c2.metric("Qini moyen (test)", f"{t[best]['mean_qini_coef']:.5f}")
    resp = t[cfg["model"]["baselines"][0]]["mean_qini_coef"]
    c3.metric("Δ vs modèle de réponse", f"{t[best]['mean_qini_coef'] - resp:+.5f}")
    st.dataframe(pd.DataFrame({k: {"Qini moyen": v["mean_qini_coef"], **{f"Qini {a}": v[a]["qini_coef"] for a in ARMS}}
                               for k, v in t.items()}).T, width="stretch")
    cols = st.columns(len(ARMS))
    for col, a in zip(cols, ARMS):
        p = ROOT / "reports" / "figures" / f"qini_{a.split()[0].lower()}.png"
        if p.exists():
            col.image(str(p), caption=f"Courbes de Qini (test) — {a}")

with tab2:
    U = scores[[f"uplift_{a}" for a in ARMS]].rename(columns=lambda c: c.removeprefix("uplift_"))
    scores = scores.assign(best_arm=U.idxmax(axis=1), best_uplift=U.max(axis=1))
    lo = st.slider("Uplift minimal (pts de conversion)", float(scores.best_uplift.min()), float(scores.best_uplift.max()),
                   0.0, step=0.0005, format="%.4f")
    sel = scores[scores.best_uplift >= lo]
    st.write(f"**{len(sel)}** clients sur {len(scores)} ({len(sel) / len(scores):.0%})")
    st.plotly_chart(px.histogram(scores, x="best_uplift", color="best_arm", nbins=60,
                                 title="Distribution de l'uplift prédit (meilleur bras)"), width="stretch")
    dim = st.selectbox("Profil moyen par…", ["channel", "zip_code", "newbie", "mens", "womens"])
    st.plotly_chart(px.bar(scores.groupby(dim).best_uplift.mean().reset_index(), x=dim, y="best_uplift",
                           title=f"Uplift prédit moyen par {dim}"), width="stretch")
    st.dataframe(sel.sort_values("best_uplift", ascending=False).head(200), width="stretch")

with tab3:
    st.markdown("Ajustez les hypothèses business : l'allocation est **recalculée** par programmation linéaire.")
    a, b, c = st.columns(3)
    budget = a.slider("Budget (EUR)", 50, 3000, int(BIZ["budget"]), 50)
    value = b.slider("Marge par conversion (EUR)", 5, 150, int(BIZ["value_per_conversion"]), 5)
    use_cap = c.checkbox(f"Plafonner « {ARMS[1]} »", value=False)
    c1, c2, c3 = st.columns(3)
    cost1 = c1.number_input(f"Coût {ARMS[0]} (EUR)", 0.01, 5.0, float(BIZ["cost_per_contact"][ARMS[0]]), 0.01)
    cost2 = c2.number_input(f"Coût {ARMS[1]} (EUR)", 0.01, 5.0, float(BIZ["cost_per_contact"][ARMS[1]]), 0.01)
    cap2 = c3.number_input("Capacité max (clients)", 0, len(scores), len(scores) // 20) if use_cap else 0
    pol, base = run_lp(budget, value, cost1, cost2, int(cap2))
    s = base["LP"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clients contactés", s["n_contacted"])
    m2.metric("Coût", f"{s['total_cost']:.2f} €")
    m3.metric("Conversions incrémentales (modèle)", f"{s['expected_incremental_conversions']:.1f}")
    m4.metric("Gain net attendu (modèle)", f"{s['expected_net_gain']:.2f} €")
    st.plotly_chart(px.bar(pd.DataFrame({k: v["expected_net_gain"] for k, v in base.items()}.items(),
                                        columns=["politique", "gain net (EUR)"]), x="politique", y="gain net (EUR)",
                           title="Gain net attendu par politique (estimation du modèle)"), width="stretch")
    st.caption("Estimation du modèle : la validation non biaisée (IPW) est dans l'onglet « Résultats A/B ».")

with tab4:
    st.subheader(f"Politiques de ciblage — budget {opt['budget']} €, estimation IPW sans biais")
    st.dataframe(pd.DataFrame({k: v["ipw_unbiased"] for k, v in opt["policies"].items()}).T, width="stretch")
    for name, r in opt["ab_tests"].items():
        p = r["profit_per_customer"]
        verdict = " significatif" if p["significant"] else " non significatif"
        st.markdown(f"**{name}** — Δ profit/client = `{p['diff_mean']:+.5f}` € · IC95 % [{p['ci_low']:+.5f}, {p['ci_high']:+.5f}] · p = {p['p_value']:.4f} → {verdict}")
        if not p["significant"] and "n_per_group_live_test" in r:
            st.info(f"Non concluant avec cet échantillon. Pour confirmer en live l'écart de conversion observé "
                    f"({r['conv_rate_a']:.3%} → {r['conv_rate_b']:.3%}) : ~{r['n_per_group_live_test']:,} clients par groupe.")
