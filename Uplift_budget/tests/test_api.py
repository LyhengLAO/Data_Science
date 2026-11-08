import joblib
import pytest
from fastapi.testclient import TestClient

from src.data import FEATURES
from src.models import MultiArmUplift

CUSTOMER = dict(recency=3, history=250.0, mens=1, womens=0, zip_code="Urban", newbie=0, channel="Web")


@pytest.fixture(scope="module")
def client(tmp_path_factory, splits, encoded, cfg):
    train, _ = splits
    pre, Xtr, _ = encoded
    m = MultiArmUplift("t_learner", cfg["arms"], cfg["control_label"], 0).fit(Xtr, train.treatment, train.y)
    path = tmp_path_factory.mktemp("m") / "model.joblib"
    joblib.dump({"pre": pre, "model": m, "learner": "t_learner", "arms": cfg["arms"],
                 "control": cfg["control_label"]}, path)
    import os
    os.environ["MODEL_PATH"] = str(path)
    from uplift.api import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict(client):
    r = client.post("/predict_uplift", json={"customers": [CUSTOMER, {**CUSTOMER, "mens": 0, "womens": 1}]})
    assert r.status_code == 200 and len(r.json()) == 2
    assert set(r.json()[0]["uplift"]) == {"Mens E-Mail", "Womens E-Mail"}


def test_predict_validation_errors(client):
    assert client.post("/predict_uplift", json={"customers": [{**CUSTOMER, "recency": 99}]}).status_code == 422
    assert client.post("/predict_uplift", json={"customers": [{k: v for k, v in CUSTOMER.items() if k != "channel"}]}).status_code == 422
    assert client.post("/predict_uplift", json={"customers": []}).status_code == 422


def test_optimize(client):
    custs = [{**CUSTOMER, "recency": 1 + i % 12, "history": 50.0 + 10 * i} for i in range(300)]
    r = client.post("/optimize", json={"customers": custs, "budget": 5.0, "value_per_conversion": 50.0})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["total_cost"] <= 5.0 + 1e-6
    assert len(body["allocation"]) == body["summary"]["n_contacted"]
    assert client.post("/optimize", json={"customers": custs, "budget": -1}).status_code == 422
