from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

PAYLOAD = {
    "tenure_months": 24,
    "avg_monthly_spend_ngn": 5000.0,
    "recharge_frequency_per_month": 6,
    "payment_consistency_score": 0.8,
    "late_payment_count": 2,
    "operator": "MTN",
    "account_type": "prepaid",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict():
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in {"low", "medium", "high"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3


def test_predict_invalid():
    r = client.post("/predict", json={"tenure_months": -1})
    assert r.status_code == 422
