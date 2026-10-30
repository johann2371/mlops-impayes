import json
import time
from datetime import datetime, timezone


import numpy as np
import pandas as pd
from fastapi import FastAPI

from src.api.schemas import PredictionRequest, PredictionResponse
from src.config import ROOT, params

app = FastAPI(title="API de prédiction du risque d'impayé", version="0.1.0")

MODEL_VERSION = params["model"]["version"]
LOG_PATH = ROOT / params["api"]["log_path"]


class FakeModel:
    """Modèle factice, remplacé par le vrai modèle une fois entraîné."""

    classes_ = ["low", "medium", "high"]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        risk = min(float(df.iloc[0]["late_payment_count"]) / 10, 1.0)
        return np.array([[0.8 * (1 - risk), 0.2, 0.8 * risk]])


def load_model():
    path = ROOT / params["model"]["path"]
    if path.exists():
        import joblib

        return joblib.load(path)
    return FakeModel()


model = load_model()


def log_prediction(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start = time.perf_counter()
    df = pd.DataFrame([request.model_dump()])
    proba = model.predict_proba(df)[0]
    classes = [str(c) for c in model.classes_]
    probabilities = {c: round(float(p), 4) for c, p in zip(classes, proba)}
    prediction = max(probabilities, key=probabilities.get)

    log_prediction(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
            "inputs": request.model_dump(),
            "prediction": prediction,
            "probabilities": probabilities,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }
    )
    return PredictionResponse(
        prediction=prediction,
        probabilities=probabilities,
        model_version=MODEL_VERSION,
    )
