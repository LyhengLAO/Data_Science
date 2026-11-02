"""Ingestion, nettoyage et split des données (Hillstrom Email Marketing).

Le dataset est issu d'un test randomisé : 3 groupes (pas d'email / email Hommes /
email Femmes) à ~1/3 chacun. C'est ce qui rend l'estimation causale possible.
"""
from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import abspath

log = logging.getLogger(__name__)

NUM = ["recency", "history"]
BIN = ["mens", "womens", "newbie"]
CAT = ["zip_code", "channel"]
FEATURES = NUM + BIN + CAT


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Téléchargement %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)
    return dest

def make_synthetic(n: int = 30_000, seed: int = 0, effect_scale: float = 1.0) -> pd.DataFrame:
    """Données synthétiques au schéma Hillstrom, avec effets hétérogènes connus.

    Uniquement pour les tests / CI hors-ligne (UPLIFT_SYNTHETIC=1). Les colonnes
    `_true_uplift_*` donnent la vérité terrain pour valider les modèles.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "recency": rng.integers(1, 13, n),
            "history": np.round(rng.lognormal(5.0, 0.9, n), 2),
            "mens": rng.integers(0, 2, n),
            "womens": rng.integers(0, 2, n),
            "zip_code": rng.choice(["Surburban", "Rural", "Urban"], n, p=[0.45, 0.15, 0.40]),
            "newbie": rng.integers(0, 2, n),
            "channel": rng.choice(["Web", "Phone", "Multichannel"], n, p=[0.45, 0.40, 0.15]),
        }
    )
    sigmoid = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    base = -4.2 - 0.06 * df.recency + 0.0003 * df.history
    d_m = effect_scale * (0.2 + 1.0 * df.mens - 1.0 * ((df.recency > 9) & (df.newbie == 0)))
    d_w = effect_scale * (0.2 + 1.0 * df.womens - 0.6 * (df.channel == "Phone"))
    p0, pm, pw = sigmoid(base), sigmoid(base + d_m), sigmoid(base + d_w)
    df["_true_uplift_mens"], df["_true_uplift_womens"] = pm - p0, pw - p0

    seg = rng.choice(["No E-Mail", "Mens E-Mail", "Womens E-Mail"], n)
    p = np.where(seg == "Mens E-Mail", pm, np.where(seg == "Womens E-Mail", pw, p0))
    df["segment"] = seg
    df["conversion"] = (rng.random(n) < p).astype(int)
    df["spend"] = df.conversion * rng.gamma(2.0, 50.0, n)
    return df

def load_raw(cfg: dict) -> tuple[pd.DataFrame, str]:
    """Charge le CSV brut. Retourne (df, source) avec source in {hillstrom, synthetic}."""
    raw = abspath(cfg["data"]["raw_path"])
    if raw.exists():
        return pd.read_csv(raw), "hillstrom"
    if os.getenv("UPLIFT_SYNTHETIC") == "1":
        log.warning("UPLIFT_SYNTHETIC=1 : données SYNTHÉTIQUES (tests uniquement).")
        return make_synthetic(seed=cfg["seed"]), "synthetic"
    download(cfg["data"]["url"], raw)
    return pd.read_csv(raw), "hillstrom"

def prepare(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Colonnes finales : FEATURES + treatment (str) + y (0/1) ; validations de base."""
    out = df[FEATURES].copy()
    out["treatment"] = df[cfg["treatment_col"]].astype(str)
    out["y"] = df[cfg["outcome"]].astype(int)
    expected = {cfg["control_label"], *cfg["arms"]}
    unknown = set(out.treatment.unique()) - expected
    if unknown:
        raise ValueError(f"Groupes de traitement inattendus : {unknown}")
    if out[FEATURES].isna().any().any():
        out = out.dropna(subset=FEATURES)
    return out.reset_index(drop=True)


def split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split stratifié sur (traitement × outcome) : préserve le ratio T/C ET le taux de
    conversion dans chaque partie, indispensable en uplift (événement rare, effet faible)."""
    strat = df.treatment + "_" + df.y.astype(str)
    a, b = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
    return a.reset_index(drop=True), b.reset_index(drop=True)

if __name__ == "__main__":  # python -m data  -> télécharge le dataset
    from .config import load_config

    logging.basicConfig(level=logging.INFO)
    _cfg = load_config()
    _df, _src = load_raw(_cfg)
    print(f"{_src}: {_df.shape[0]} lignes, {_df.shape[1]} colonnes")