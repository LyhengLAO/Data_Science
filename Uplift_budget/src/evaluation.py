"""Métriques d'uplift : courbe de Qini, coefficient de Qini, uplift@k, déciles."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_trapz = getattr(np, "trapezoid", None) or np.trapz


def qini_curve(y, t, score):
    """Qini(k) = Y_T(k) - Y_C(k) * N_T(k)/N_C(k), population triée par score décroissant.
    Retourne (x = fraction ciblée, q = conversions incrémentales cumulées)."""
    y, t, score = map(np.asarray, (y, t, score))
    order = np.argsort(-score, kind="stable")
    y, t = y[order], t[order].astype(int)
    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    y_t, y_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    ratio = np.divide(n_t, n_c, out=np.zeros(len(y)), where=n_c > 0)
    q = y_t - y_c * ratio
    x = np.arange(1, len(y) + 1) / len(y)
    return np.r_[0.0, x], np.r_[0.0, q]


def qini_coefficient(y, t, score) -> float:
    """Aire entre la courbe de Qini et la diagonale aléatoire, normalisée par n
    (unité : taux de conversion incrémental moyen). > 0 = mieux que le hasard."""
    x, q = qini_curve(y, t, score)
    return float(_trapz(q - q[-1] * x, x) / (len(x) - 1))


def uplift_at_k(y, t, score, k: float = 0.3) -> float:
    y, t, score = map(np.asarray, (y, t, score))
    top = np.argsort(-score, kind="stable")[: max(int(len(y) * k), 1)]
    yt, tt = y[top], t[top].astype(bool)
    if tt.all() or (~tt).all():
        return float("nan")
    return float(yt[tt].mean() - yt[~tt].mean())


def uplift_by_decile(y, t, score, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "t": np.asarray(t).astype(bool), "score": score})
    df["decile"] = pd.qcut(df.score.rank(method="first", ascending=False), bins, labels=range(1, bins + 1))
    g = df.groupby("decile", observed=True)
    out = pd.DataFrame(
        {
            "n": g.size(),
            "mean_pred_uplift": g.score.mean(),
            "rate_treated": g.apply(lambda d: d.y[d.t].mean(), include_groups=False),
            "rate_control": g.apply(lambda d: d.y[~d.t].mean(), include_groups=False),
        }
    )
    out["observed_uplift"] = out.rate_treated - out.rate_control
    return out.reset_index()


def plot_qini(curves: dict[str, tuple[np.ndarray, np.ndarray]], title: str = "Courbes de Qini"):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, (x, q) in curves.items():
        ax.plot(x, q, label=name)
    x0, q0 = next(iter(curves.values()))
    ax.plot([0, 1], [0, q0[-1]], "k--", label="aléatoire")
    ax.set(xlabel="Fraction de la population ciblée", ylabel="Conversions incrémentales", title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
