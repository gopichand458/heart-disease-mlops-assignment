"""FastAPI service: /predict, /health, /metrics."""
import logging
import os
import time
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel

from src.features import add_features

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("heart-api")

REQUESTS = Counter("api_requests_total", "HTTP requests",
                   ["path", "method", "status"])
LATENCY = Histogram("api_request_seconds", "Request latency", ["path"])
PREDICTIONS = Counter("predictions_total", "Predictions made", ["label"])

model = joblib.load(MODEL_PATH)
app = FastAPI(title="Heart Disease Risk API")
app.mount("/metrics", make_asgi_app())


class Patient(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: Optional[float] = None
    thal: Optional[float] = None


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    path = request.url.path
    if not path.startswith("/metrics"):
        REQUESTS.labels(path, request.method, response.status_code).inc()
        LATENCY.labels(path).observe(elapsed)
        log.info("%s %s -> %s in %.1f ms", request.method, path,
                 response.status_code, elapsed * 1000)
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: Patient):
    df = add_features(pd.DataFrame([patient.model_dump()]).astype(float))
    proba = float(model.predict_proba(df)[0, 1])
    pred = int(proba >= 0.5)
    PREDICTIONS.labels(str(pred)).inc()
    log.info("prediction=%d p_disease=%.3f", pred, proba)
    return {
        "prediction": pred,
        "label": "disease" if pred else "no disease",
        "confidence": round(proba if pred else 1 - proba, 4),
        "probability_disease": round(proba, 4),
    }
