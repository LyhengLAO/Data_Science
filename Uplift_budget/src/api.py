"""API FastAPI : scoring d'uplift et optimisation d'allocation.

uvicorn uplift.api:app --port 8000   (PYTHONPATH=src)
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .config import ROOT, load_config
from .data import FEATURES
from .optimization import optimize_allocation


class Customer(BaseModel):
    recency: int = Field(ge=1, le=12)
    history: float = Field(ge=0)
    mens: int = Field(ge=0, le=1)
    womens: int = Field(ge=0, le=1)
    zip_code: Literal["Surburban", "Rural", "Urban"]
    newbie: int = Field(ge=0, le=1)
    channel: Literal["Phone", "Web", "Multichannel"]


class ScoreRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=50_000)


class OptimizeRequest(ScoreRequest):
    budget: float = Field(gt=0)
    value_per_conversion: float | None = Field(default=None, gt=0)
    cost_per_contact: dict[str, float] | None = None
    max_per_arm: dict[str, int] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.getenv("MODEL_PATH", ROOT / "models" / "best_model.joblib"))
    app.state.artifact = joblib.load(path) if path.exists() else None
    app.state.cfg = load_config()
    yield


app = FastAPI(title="Uplift Budget Optimizer", version="1.0.0", lifespan=lifespan)


def _artifact(request: Request) -> dict:
    art = request.app.state.artifact
    if art is None:
        raise HTTPException(503, "Modèle indisponible : lancer `make train` d'abord.")
    return art


def _uplift(art: dict, customers: list[Customer]) -> pd.DataFrame:
    df = pd.DataFrame([c.model_dump() for c in customers])[FEATURES]
    return art["model"].predict_uplift(art["pre"].transform(df))


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": request.app.state.artifact is not None}


@app.get("/model_metrics")
def model_metrics():
    p = ROOT / "reports" / "metrics.json"
    if not p.exists():
        raise HTTPException(404, "metrics.json introuvable")
    return json.loads(p.read_text())


@app.post("/predict_uplift")
def predict_uplift(req: ScoreRequest, request: Request):
    art = _artifact(request)
    U = _uplift(art, req.customers)
    best = U.idxmax(axis=1)
    return [
        {"uplift": {a: float(U.at[i, a]) for a in U.columns},
         "best_action": best[i] if U.max(axis=1)[i] > 0 else art["control"],
         "best_uplift": float(U.max(axis=1)[i])}
        for i in U.index
    ]


@app.post("/optimize")
def optimize(req: OptimizeRequest, request: Request):
    art = _artifact(request)
    biz = request.app.state.cfg["business"]
    value = req.value_per_conversion or biz["value_per_conversion"]
    costs = req.cost_per_contact or biz["cost_per_contact"]
    U = _uplift(art, req.customers)
    missing = set(U.columns) - set(costs)
    if missing:
        raise HTTPException(422, f"Coût manquant pour : {sorted(missing)}")
    policy, summary = optimize_allocation(U, value, costs, req.budget, art["control"], req.max_per_arm)
    sel = policy[policy != art["control"]]
    return {
        "summary": summary,
        "allocation": [{"customer_index": int(i), "action": a, "expected_uplift": float(U.at[i, a])}
                       for i, a in sel.items()],
    }
