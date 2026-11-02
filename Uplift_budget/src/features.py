from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from data import BIN, CAT


def build_preprocessor() -> ColumnTransformer:
    """history est très asymétrique -> log1p ; recency standardisée ; catégories one-hot."""
    return ColumnTransformer(
        [
            ("history", Pipeline([("log", FunctionTransformer(np.log1p)), ("sc", StandardScaler())]), ["history"]),
            ("recency", StandardScaler(), ["recency"]),
            ("bin", "passthrough", BIN),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ]
    )
