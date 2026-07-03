"""FastAPI app exposing the churn model.

Run with:  uv run uvicorn churn_risk_service.api:app --reload
Docs at:   http://localhost:8000/docs
"""

from fastapi import FastAPI

from churn_risk_service.schemas import CustomerFeatures, PredictionResponse
from churn_risk_service.service import get_artifact, score

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
    return score(features)
