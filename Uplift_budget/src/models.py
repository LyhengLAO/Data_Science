"""Learners d'uplift (binaire traité vs contrôle) + wrapper multi-traitements.

Interface commune : fit(X, t, y) puis predict_uplift(X) = P(y=1|x,T=1) - P(y=1|x,T=0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

_PARAMS = dict(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=100, l2_regularization=1.0)


def _clf(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=seed, **_PARAMS)


def _reg(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(random_state=seed, **_PARAMS)

class SLearner:
    """Un seul modèle avec le traitement comme feature."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, X, t, y):
        self.m_ = _clf(self.seed).fit(np.column_stack([X, t]), y)
        return self

    def predict_uplift(self, X):
        n = len(X)
        p1 = self.m_.predict_proba(np.column_stack([X, np.ones(n)]))[:, 1]
        p0 = self.m_.predict_proba(np.column_stack([X, np.zeros(n)]))[:, 1]
        return p1 - p0

class TLearner:
    """Deux modèles indépendants (traités / contrôle)."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, X, t, y):
        t = np.asarray(t).astype(bool)
        self.m1_ = _clf(self.seed).fit(X[t], y[t])
        self.m0_ = _clf(self.seed).fit(X[~t], y[~t])
        return self

    def predict_uplift(self, X):
        return self.m1_.predict_proba(X)[:, 1] - self.m0_.predict_proba(X)[:, 1]

class XLearner:
    """Künzel et al. 2019 : effets imputés puis pondération par le propensity score."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, X, t, y):
        t = np.asarray(t).astype(bool)
        y = np.asarray(y)
        mu1 = _clf(self.seed).fit(X[t], y[t])
        mu0 = _clf(self.seed).fit(X[~t], y[~t])
        d1 = y[t] - mu0.predict_proba(X[t])[:, 1]       # effet imputé des traités
        d0 = mu1.predict_proba(X[~t])[:, 1] - y[~t]     # effet imputé des contrôles
        self.tau1_ = _reg(self.seed).fit(X[t], d1)
        self.tau0_ = _reg(self.seed).fit(X[~t], d0)
        self.g_ = LogisticRegression(max_iter=1000).fit(X, t)
        return self

    def predict_uplift(self, X):
        g = self.g_.predict_proba(X)[:, 1]
        return g * self.tau0_.predict(X) + (1 - g) * self.tau1_.predict(X)

class ResponseModel:
    """Baseline NON causale : P(conversion | traité). Cible ceux qui convertissent le plus,
    pas ceux que l'email fait convertir en plus. Sert à démontrer l'intérêt de l'uplift."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, X, t, y):
        t = np.asarray(t).astype(bool)
        self.m_ = _clf(self.seed).fit(X[t], y[t])
        return self

    def predict_uplift(self, X):
        return self.m_.predict_proba(X)[:, 1]


LEARNERS = {
    "s_learner": SLearner,
    "t_learner": TLearner,
    "x_learner": XLearner,
    "response_model": ResponseModel,
}


class MultiArmUplift:
    """Un learner par bras de traitement, chacun entraîné sur (bras vs contrôle)."""

    def __init__(self, learner: str, arms: list[str], control: str, seed: int = 42):
        self.learner, self.arms, self.control, self.seed = learner, list(arms), control, seed

    def fit(self, X, treatment, y):
        treatment, y = np.asarray(treatment), np.asarray(y)
        self.models_ = {}
        for arm in self.arms:
            mask = (treatment == arm) | (treatment == self.control)
            t = (treatment[mask] == arm).astype(int)
            self.models_[arm] = LEARNERS[self.learner](self.seed).fit(X[mask], t, y[mask])
        return self

    def predict_uplift(self, X) -> pd.DataFrame:
        return pd.DataFrame({a: self.models_[a].predict_uplift(X) for a in self.arms})
    