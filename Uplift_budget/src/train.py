"""Entraînement : sélection du learner sur validation, évaluation finale sur test.

python -m uplift.train
"""
from __future__ import annotations

import json
import logging

import joblib
import pandas as pd

from config import abspath, load_config
from data import FEATURES, load_raw, prepare, split
from evaluation import plot_qini, qini_coefficient, qini_curve, uplift_at_k
from features import build_preprocessor
from models import MultiArmUplift

log = logging.getLogger(__name__)

def _evaluate(model: MultiArmUplift, X, df, arms, control):
    """Qini / uplift@30% pour chaque bras (bras vs contrôle) + courbes."""
    U = model.predict_uplift(X)
    per_arm, curves = {}, {}
    for a in arms:
        mask = ((df.treatment == a) | (df.treatment == control)).to_numpy()
        y, t, s = df.y.to_numpy()[mask], (df.treatment.to_numpy()[mask] == a).astype(int), U[a].to_numpy()[mask]
        per_arm[a] = {
            "qini_coef": qini_coefficient(y, t, s),
            "uplift_at_30pct": uplift_at_k(y, t, s, 0.3),
            "n": int(mask.sum()),
        }
        curves[a] = qini_curve(y, t, s)
    per_arm["mean_qini_coef"] = float(pd.Series([per_arm[a]["qini_coef"] for a in arms]).mean())
    return U, per_arm, curves

def main(config: str | None = None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(config)
    seed, arms, ctrl = cfg["seed"], cfg["arms"], cfg["control_label"]

    raw, source = load_raw(cfg)
    df = prepare(raw, cfg)
    train, test = split(df, cfg["data"]["test_size"], seed)
    fit, val = split(train, cfg["data"]["val_size"], seed)
    log.info("source=%s | fit=%d val=%d test=%d", source, len(fit), len(val), len(test))

    # --- 1) sélection du learner sur la validation (pas sur le test) ---------------
    pre_sel = build_preprocessor().fit(fit[FEATURES])
    Xf, Xv = pre_sel.transform(fit[FEATURES]), pre_sel.transform(val[FEATURES])
    selection = {}
    for name in cfg["model"]["candidates"] + cfg["model"]["baselines"]:
        m = MultiArmUplift(name, arms, ctrl, seed).fit(Xf, fit.treatment, fit.y)
        _, per_arm, _ = _evaluate(m, Xv, val, arms, ctrl)
        selection[name] = per_arm
        log.info("[val] %-15s mean_qini=%.6f", name, per_arm["mean_qini_coef"])
    best = max(cfg["model"]["candidates"], key=lambda n: selection[n]["mean_qini_coef"])

    # --- 2) ré-entraînement sur tout le train, évaluation finale sur test ----------
    pre = build_preprocessor().fit(train[FEATURES])
    Xtr, Xte = pre.transform(train[FEATURES]), pre.transform(test[FEATURES])
    test_metrics, all_curves, best_U = {}, {}, None
    for name in cfg["model"]["candidates"] + cfg["model"]["baselines"]:
        m = MultiArmUplift(name, arms, ctrl, seed).fit(Xtr, train.treatment, train.y)
        U, per_arm, curves = _evaluate(m, Xte, test, arms, ctrl)
        test_metrics[name] = per_arm
        all_curves[name] = curves
        if name == best:
            best_model, best_U = m, U

    fig_dir = abspath("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    for a in arms:
        fig = plot_qini({n: c[a] for n, c in all_curves.items()}, title=f"Qini (test) — {a} vs contrôle")
        fig.savefig(fig_dir / f"qini_{a.split()[0].lower()}.png", dpi=130)

    # --- 3) artefacts ------------------------------------------------------------
    models_dir = abspath("models")
    models_dir.mkdir(exist_ok=True)
    joblib.dump({"pre": pre, "model": best_model, "learner": best, "arms": arms, "control": ctrl},
                models_dir / "best_model.joblib")

    scores = test.copy()
    for a in arms:
        scores[f"uplift_{a}"] = best_U[a].to_numpy()
    out_dir = abspath(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out_dir / "test_scores.parquet")

    metrics = {
        "data_source": source,
        "n_fit": len(fit), "n_val": len(val), "n_train": len(train), "n_test": len(test),
        "best_learner": best,
        "validation": selection,
        "test": test_metrics,
    }
    abspath("reports").mkdir(exist_ok=True)
    (abspath("reports") / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("best=%s | test mean_qini=%.6f (response_model=%.6f)", best,
             test_metrics[best]["mean_qini_coef"],
             test_metrics[cfg["model"]["baselines"][0]]["mean_qini_coef"])
    return metrics

def ensure_scores(cfg: dict | None = None) -> pd.DataFrame:
    """Charge test_scores.parquet, ou lance l'entraînement s'il n'existe pas (notebooks)."""
    cfg = cfg or load_config()
    path = abspath(cfg["data"]["processed_dir"]) / "test_scores.parquet"
    if not path.exists():
        main()
    return pd.read_parquet(path)


if __name__ == "__main__":
    main()

