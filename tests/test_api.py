from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

PAYLOAD = {
    "age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 1, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_prediction_and_confidence():
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0


def test_predict_rejects_bad_input():
    bad = {**PAYLOAD, "age": "not-a-number"}
    assert client.post("/predict", json=bad).status_code == 422


def test_metrics_endpoint():
    client.post("/predict", json=PAYLOAD)
    assert "predictions_total" in client.get("/metrics/").text
