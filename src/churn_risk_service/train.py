"""Train the final model and persist it as a serving artifact.

The deployed model (`logreg_feat`) is trained on train+val for more data; the test
split stays untouched for final evaluation. The chosen operating threshold (Day 10)
is stored alongside the pipeline.

Run with:  uv run python -m churn_risk_service.train
"""

from pathlib import Path

import joblib
import pandas as pd

from churn_risk_service.data import load_and_split
from churn_risk_service.logging_config import get_logger
from churn_risk_service.model import build_pipeline

logger = get_logger(__name__)

MODEL_PATH = Path("models/churn_model.joblib")
MODEL_VERSION = "v2-logreg_feat"
DEFAULT_THRESHOLD = 0.07  # cost-optimal on validation (FP=$50, FN=$500); see MODEL_CARD.md


def train_and_persist(path: Path = MODEL_PATH, threshold: float = DEFAULT_THRESHOLD) -> dict:
    s = load_and_split()
    x = pd.concat([s.X_train, s.X_val])
    y = pd.concat([s.y_train, s.y_val])
    pipeline = build_pipeline("logreg_feat").fit(x, y)
    artifact = {
        "pipeline": pipeline,
        "threshold": threshold,
        "model_version": MODEL_VERSION,
    }
    path.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, path)
    logger.info("Persisted %s (threshold=%.2f) -> %s", MODEL_VERSION, threshold, path)
    return artifact


def main() -> None:
    train_and_persist()


if __name__ == "__main__":
    main()
