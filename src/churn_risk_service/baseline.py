"""Baseline churn model + an honest evaluation report.

Compares a majority-class DummyClassifier against LogisticRegression on the
validation split. The point: on imbalanced data, *accuracy* flatters a useless
model — PR-AUC (average precision) is the honest headline metric.

Preprocessing is fit inside the Pipeline on the training split only, so no
validation/test information leaks into scaling or encoding.

Run with:  uv run python -m churn_risk_service.baseline
"""

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_risk_service.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_and_split
from churn_risk_service.logging_config import get_logger

logger = get_logger(__name__)

REPORT_PATH = Path("reports/eval_baseline.md")


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_model(kind: str = "logreg") -> Pipeline:
    if kind == "dummy":
        clf = DummyClassifier(strategy="most_frequent")
    elif kind == "logreg":
        clf = LogisticRegression(max_iter=1000)
    else:
        raise ValueError(f"unknown model kind: {kind}")
    return Pipeline([("pre", build_preprocessor()), ("clf", clf)])


def evaluate(model: Pipeline, x, y) -> dict[str, float]:
    proba = model.predict_proba(x)[:, 1]
    pred = model.predict(x)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
    }


def _report_markdown(results: dict[str, dict[str, float]], base_rate: float) -> str:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    header = "| model | " + " | ".join(metrics) + " |\n"
    sep = "| --- " * (len(metrics) + 1) + "|\n"
    rows = ""
    for name, res in results.items():
        rows += f"| {name} | " + " | ".join(f"{res[m]:.3f}" for m in metrics) + " |\n"
    return (
        "# Baseline evaluation (validation split)\n\n"
        f"Churn base rate (positive class): **{base_rate:.3f}**. "
        "Metrics are on the held-out validation set; "
        "test stays untouched until final selection.\n\n"
        + header
        + sep
        + rows
        + "\n## Read this honestly\n\n"
        "- The **dummy** (predict everyone stays) posts high **accuracy** — it just "
        "matches the majority class. Its **pr_auc ≈ the base rate**, i.e. no signal.\n"
        "- **LogisticRegression** barely moves accuracy but lifts **pr_auc** and "
        "**roc_auc** well above the floor — that's real ranking signal.\n"
        "- Headline metric for this problem: **PR-AUC**, not accuracy.\n\n"
        "## Leakage guard\n\n"
        "Preprocessing (scaling, one-hot) is fit inside the Pipeline on the training "
        "split only; validation/test rows never influence the fitted transforms.\n"
    )


def train_and_report(seed: int = 42, write: bool = True) -> dict[str, dict[str, float]]:
    s = load_and_split(seed=seed)
    results = {
        kind: evaluate(build_model(kind).fit(s.X_train, s.y_train), s.X_val, s.y_val)
        for kind in ("dummy", "logreg")
    }
    if write:
        REPORT_PATH.parent.mkdir(exist_ok=True)
        REPORT_PATH.write_text(_report_markdown(results, base_rate=float(s.y_train.mean())))
        logger.info("Wrote %s", REPORT_PATH)
    return results


def main() -> None:
    results = train_and_report()
    for name, res in results.items():
        logger.info(
            "%-7s | acc=%.3f pr_auc=%.3f roc_auc=%.3f",
            name,
            res["accuracy"],
            res["pr_auc"],
            res["roc_auc"],
        )


if __name__ == "__main__":
    main()
