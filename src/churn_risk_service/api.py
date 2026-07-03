"""FastAPI app exposing the churn model.

Run with:  uv run uvicorn churn_risk_service.api:app --reload
Docs at:   http://localhost:8000/docs
"""

import json
import time

from fastapi import FastAPI

from churn_risk_service.logging_config import get_logger
from churn_risk_service.schemas import CustomerFeatures, PredictionResponse
from churn_risk_service.service import get_artifact, score

logger = get_logger(__name__)

app = FastAPI(
    title="churn-risk-service",
    description="Predict customer churn risk. See /docs for the schema.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": get_artifact()["model_version"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures) -> PredictionResponse:
    start = time.perf_counter()
    response = score(features)
    # Structured log line — one JSON object per prediction, for online monitoring.
    logger.info(
        json.dumps(
            {
                "event": "predict",
                "model_version": response.model_version,
                "tenure": features.tenure,
                "contract": features.Contract,
                "monthly_charges": features.MonthlyCharges,
                "churn_probability": round(response.churn_probability, 4),
                "churn_prediction": response.churn_prediction,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            }
        )
    )
    return response
