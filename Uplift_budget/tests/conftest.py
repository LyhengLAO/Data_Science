import os

import numpy as np
import pytest

os.environ.setdefault("UPLIFT_SYNTHETIC", "1")

from src.config import load_config  # noqa: E402
from src.data import FEATURES, make_synthetic, prepare, split  # noqa: E402
from src.features import build_preprocessor  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def synth():
    # effets marqués pour que la vérité terrain soit récupérable avec peu de données
    return make_synthetic(n=40_000, seed=1, effect_scale=2.0)


@pytest.fixture(scope="session")
def prepared(synth, cfg):
    return prepare(synth, cfg)


@pytest.fixture(scope="session")
def splits(prepared):
    return split(prepared, 0.3, 42)


@pytest.fixture(scope="session")
def encoded(splits):
    train, test = splits
    pre = build_preprocessor().fit(train[FEATURES])
    return pre, pre.transform(train[FEATURES]), pre.transform(test[FEATURES])


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)
