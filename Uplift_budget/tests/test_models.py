import numpy as np
import pytest

from src.data import FEATURES, split
from src.evaluation import qini_coefficient
from src.features import build_preprocessor
from src.models import LEARNERS, MultiArmUplift, ResponseModel, TLearner


@pytest.fixture(scope="module")
def with_truth(synth, prepared):
    """Données préparées + vérité terrain de l'uplift (alignées ligne à ligne)."""
    return prepared.join(synth[["_true_uplift_mens", "_true_uplift_womens"]])


@pytest.mark.parametrize("name", ["s_learner", "t_learner", "x_learner"])
def test_learners_recover_ground_truth(name, with_truth, cfg):
    """Sur données synthétiques à effets connus, l'uplift prédit corrèle avec la vérité."""
    train, test = split(with_truth, 0.3, 42)
    pre = build_preprocessor().fit(train[FEATURES])
    m = MultiArmUplift(name, cfg["arms"], cfg["control_label"], 0).fit(
        pre.transform(train[FEATURES]), train.treatment, train.y)
    U = m.predict_uplift(pre.transform(test[FEATURES]))
    assert U.shape == (len(test), 2) and np.isfinite(U.to_numpy()).all()
    r_m = np.corrcoef(U["Mens E-Mail"], test["_true_uplift_mens"])[0, 1]
    r_w = np.corrcoef(U["Womens E-Mail"], test["_true_uplift_womens"])[0, 1]
    assert r_m > 0.3 and r_w > 0.3, (name, r_m, r_w)


def test_t_learner_beats_random(splits, encoded, cfg):
    train, test = splits
    _, Xtr, Xte = encoded
    arm, ctrl = "Mens E-Mail", cfg["control_label"]
    mask = ((test.treatment == arm) | (test.treatment == ctrl)).to_numpy()
    y, t = test.y.to_numpy()[mask], (test.treatment.to_numpy()[mask] == arm).astype(int)
    m = MultiArmUplift("t_learner", [arm], ctrl, 0).fit(Xtr, train.treatment, train.y)
    assert qini_coefficient(y, t, m.predict_uplift(Xte)[arm].to_numpy()[mask]) > 0


def test_uplift_beats_response_model_when_responders_are_not_persuadable():
    """Cas d'école : x=0 convertit beaucoup SANS email (30 %), x=1 convertit peu (5 %) mais
    l'email ajoute +10 pts. Un modèle de réponse cible x=0 ; l'uplift cible x=1."""
    rng = np.random.default_rng(0)
    n = 40_000
    x = rng.integers(0, 2, n)
    t = rng.integers(0, 2, n)
    p = np.where(x == 0, 0.30, 0.05) + 0.10 * x * t
    y = (rng.random(n) < p).astype(int)
    X = np.column_stack([x, rng.normal(size=n)])
    tr, te = slice(0, 28_000), slice(28_000, n)
    up = TLearner(0).fit(X[tr], t[tr], y[tr]).predict_uplift(X[te])
    resp = ResponseModel(0).fit(X[tr], t[tr], y[tr]).predict_uplift(X[te])
    q_up, q_resp = qini_coefficient(y[te], t[te], up), qini_coefficient(y[te], t[te], resp)
    assert q_up > 0 > q_resp or q_up > 2 * q_resp


def test_registry():
    assert {"s_learner", "t_learner", "x_learner", "response_model"} <= set(LEARNERS)
