"""Model loading + scoring, decoupled from the web layer."""

from functools import lru_cache

import joblib
import pandas as pd

from churn_risk_service.logging_config import get_logger
from churn_risk_service.schemas import CustomerFeatures, PredictionResponse
from churn_risk_service.train import MODEL_PATH, train_and_persist

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_artifact() -> dict:
    """Load the serving artifact, training and persisting it if absent."""
    if MODEL_PATH.exists():
        logger.info("Loading model from %s", MODEL_PATH)
        return joblib.load(MODEL_PATH)
    logger.info("No persisted model; training one now")
    return train_and_persist()


def score(features: CustomerFeatures) -> PredictionResponse:
    artifact = get_artifact()
    row = pd.DataFrame([features.model_dump()])
    proba = float(artifact["pipeline"].predict_proba(row)[0, 1])
    threshold = float(artifact["threshold"])
    return PredictionResponse(
        churn_probability=proba,
        churn_prediction=proba >= threshold,
        threshold=threshold,
        model_version=artifact["model_version"],
    )
