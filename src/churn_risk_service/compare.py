"""Train all model variants and produce a comparison table on the validation split.

Run with:  uv run python -m churn_risk_service.compare
"""

from pathlib import Path

from churn_risk_service.baseline import evaluate
from churn_risk_service.data import load_and_split
from churn_risk_service.logging_config import get_logger
from churn_risk_service.model import KINDS, build_pipeline

logger = get_logger(__name__)

REPORT_PATH = Path("reports/comparison.md")
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def compare(seed: int = 42, write: bool = True) -> dict[str, dict[str, float]]:
    s = load_and_split(seed=seed)
    results = {
        kind: evaluate(build_pipeline(kind).fit(s.X_train, s.y_train), s.X_val, s.y_val)
        for kind in KINDS
    }
    if write:
        REPORT_PATH.parent.mkdir(exist_ok=True)
        REPORT_PATH.write_text(_markdown(results))
        logger.info("Wrote %s", REPORT_PATH)
    return results


def _markdown(results: dict[str, dict[str, float]]) -> str:
    header = "| model | " + " | ".join(METRICS) + " |\n"
    sep = "| --- " * (len(METRICS) + 1) + "|\n"
    rows = "".join(
        f"| {name} | " + " | ".join(f"{res[m]:.3f}" for m in METRICS) + " |\n"
        for name, res in results.items()
    )
    return (
        "# Model comparison (validation split)\n\n"
        "Headline metric: **pr_auc** (average precision) — the honest choice on a "
        "~26% base rate. Test split stays untouched.\n\n"
        + header
        + sep
        + rows
        + "\n- `logreg_base` — Day-8 baseline (raw features).\n"
        "- `logreg_feat` — same model + engineered features.\n"
        "- `hgb_feat` — gradient boosting + engineered features, `class_weight=balanced`.\n"
        "\nEngineered features are row-wise and fit inside the pipeline; no leakage.\n"
    )


def main() -> None:
    results = compare()
    for name, res in results.items():
        logger.info("%-12s | pr_auc=%.3f roc_auc=%.3f", name, res["pr_auc"], res["roc_auc"])


if __name__ == "__main__":
    main()
