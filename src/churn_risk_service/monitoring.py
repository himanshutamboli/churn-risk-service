"""Data-drift detection via Population Stability Index (PSI).

Why PSI, not a library: it's transparent, dependency-light, and testable. We build
a reference profile from the training features (quantile bins for numeric, category
frequencies for categorical), then score any current batch against it.

PSI interpretation (industry convention):
  < 0.10  no significant drift
  0.10 - 0.25  moderate drift (investigate)
  > 0.25  major drift (model likely degrading)

Run with:  uv run python -m churn_risk_service.monitoring
"""

from pathlib import Path

import numpy as np
import pandas as pd

from churn_risk_service.data import CATEGORICAL_FEATURES, load_and_split
from churn_risk_service.logging_config import get_logger

logger = get_logger(__name__)

REPORT_PATH = Path("reports/drift.md")

DRIFT_NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
DRIFT_CATEGORICAL = [*CATEGORICAL_FEATURES, "SeniorCitizen"]

MODERATE, MAJOR = 0.10, 0.25


def _psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-6) -> float:
    expected = np.clip(expected, eps, None)
    actual = np.clip(actual, eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _severity(psi: float) -> str:
    if psi >= MAJOR:
        return "major"
    if psi >= MODERATE:
        return "moderate"
    return "none"


def build_reference(x: pd.DataFrame, n_bins: int = 10) -> dict:
    """Reference distribution profile from training features."""
    numeric = {}
    for f in DRIFT_NUMERIC:
        edges = np.unique(np.quantile(x[f], np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(x[f], bins=edges)
        numeric[f] = {"edges": edges.tolist(), "expected": (counts / counts.sum()).tolist()}
    categorical = {
        f: {"expected": x[f].astype(str).value_counts(normalize=True).to_dict()}
        for f in DRIFT_CATEGORICAL
    }
    return {"numeric": numeric, "categorical": categorical}


def compute_drift(reference: dict, current: pd.DataFrame) -> list[dict]:
    """PSI per feature for a current batch vs the reference profile."""
    rows = []
    for f, ref in reference["numeric"].items():
        edges = np.array(ref["edges"])
        expected = np.array(ref["expected"])
        counts, _ = np.histogram(current[f], bins=edges)
        actual = counts / max(counts.sum(), 1)
        rows.append({"feature": f, "type": "numeric", "psi": _psi(expected, actual)})
    for f, ref in reference["categorical"].items():
        expected_map = ref["expected"]
        current_map = current[f].astype(str).value_counts(normalize=True).to_dict()
        cats = set(expected_map) | set(current_map)
        expected = np.array([expected_map.get(c, 0.0) for c in cats])
        actual = np.array([current_map.get(c, 0.0) for c in cats])
        rows.append({"feature": f, "type": "categorical", "psi": _psi(expected, actual)})
    for r in rows:
        r["severity"] = _severity(r["psi"])
    return sorted(rows, key=lambda r: r["psi"], reverse=True)


def _markdown(title: str, rows: list[dict]) -> str:
    header = "| feature | type | psi | severity |\n| --- | --- | --- | --- |\n"
    body = "".join(
        f"| {r['feature']} | {r['type']} | {r['psi']:.3f} | {r['severity']} |\n" for r in rows
    )
    return f"## {title}\n\n{header}{body}\n"


def main() -> None:
    s = load_and_split()
    reference = build_reference(s.X_train)

    # 1) test split vs train — same population, expect little drift.
    test_rows = compute_drift(reference, s.X_test)
    # 2) a synthetically shifted batch — expect major drift.
    shifted = s.X_test.copy()
    shifted["MonthlyCharges"] = shifted["MonthlyCharges"] + 60
    shifted["Contract"] = "Month-to-month"
    shifted_rows = compute_drift(reference, shifted)

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(
        "# Drift report (PSI)\n\n"
        + _markdown("Test split vs training reference", test_rows)
        + _markdown("Synthetically shifted batch (+$60 charges, all month-to-month)", shifted_rows)
    )
    logger.info("Wrote %s", REPORT_PATH)
    logger.info(
        "max PSI (test)=%.3f  max PSI (shifted)=%.3f", test_rows[0]["psi"], shifted_rows[0]["psi"]
    )


if __name__ == "__main__":
    main()
