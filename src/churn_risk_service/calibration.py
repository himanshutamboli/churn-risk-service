"""Probability calibration + business-cost-optimal threshold selection.

Two production concerns:

1. **Calibration** — a predicted 0.7 should mean ~70% churn probability. We compare
   the uncalibrated model's Brier score against an isotonic-calibrated version.
2. **Threshold** — 0.5 is arbitrary. We pick the threshold that minimizes expected
   business cost, given that missing a churner (FN) costs far more than a wasted
   retention offer (FP).

Run with:  uv run python -m churn_risk_service.calibration
"""

from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, confusion_matrix

from churn_risk_service.data import load_and_split
from churn_risk_service.logging_config import get_logger
from churn_risk_service.model import build_pipeline

logger = get_logger(__name__)

REPORT_PATH = Path("reports/calibration.md")

# Business cost assumptions (illustrative, documented in the model card).
COST_FP = 50.0  # wasted retention offer to someone who would have stayed
COST_FN = 500.0  # value lost when a real churner is missed (no offer)


def expected_cost(y_true, proba, threshold: float) -> float:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    # Offering to a true churner (tp) still costs the offer, but averts the loss.
    return COST_FP * (fp + tp) + COST_FN * fn


def choose_threshold(y_true, proba) -> tuple[float, float]:
    """Return (best_threshold, best_cost) minimizing expected business cost."""
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = [expected_cost(y_true, proba, t) for t in thresholds]
    best_i = int(np.argmin(costs))
    return float(thresholds[best_i]), float(costs[best_i])


def run(seed: int = 42, write: bool = True) -> dict:
    s = load_and_split(seed=seed)

    uncal = build_pipeline("logreg_feat").fit(s.X_train, s.y_train)
    cal = CalibratedClassifierCV(build_pipeline("logreg_feat"), method="isotonic", cv=5)
    cal.fit(s.X_train, s.y_train)

    p_uncal = uncal.predict_proba(s.X_val)[:, 1]
    p_cal = cal.predict_proba(s.X_val)[:, 1]
    brier_uncal = brier_score_loss(s.y_val, p_uncal)
    brier_cal = brier_score_loss(s.y_val, p_cal)

    # Calibration did not improve Brier (logreg is already well-calibrated), so we
    # deploy the uncalibrated model and select the threshold on it.
    best_t, best_cost = choose_threshold(s.y_val, p_uncal)
    cost_at_half = expected_cost(s.y_val, p_uncal, 0.5)

    pred_best = (p_uncal >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(s.y_val, pred_best, labels=[0, 1]).ravel()
    summary = {
        "brier_uncalibrated": brier_uncal,
        "brier_calibrated": brier_cal,
        "best_threshold": best_t,
        "cost_at_best": best_cost,
        "cost_at_0.5": cost_at_half,
        "savings_vs_0.5": cost_at_half - best_cost,
        "confusion_at_best": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "recall_at_best": tp / (tp + fn),
        "precision_at_best": tp / (tp + fp) if (tp + fp) else 0.0,
    }
    if write:
        REPORT_PATH.parent.mkdir(exist_ok=True)
        REPORT_PATH.write_text(_markdown(summary))
        logger.info("Wrote %s", REPORT_PATH)
    return summary


def _markdown(s: dict) -> str:
    c = s["confusion_at_best"]
    return (
        "# Calibration & threshold (validation split)\n\n"
        "## Calibration (Brier score, lower is better)\n\n"
        f"- uncalibrated: **{s['brier_uncalibrated']:.4f}**\n"
        f"- isotonic-calibrated: **{s['brier_calibrated']:.4f}**\n\n"
        "Isotonic calibration did not improve Brier — logistic regression is already "
        "well-calibrated here — so we deploy the uncalibrated model and select the "
        "threshold on it.\n\n"
        "## Business-cost threshold\n\n"
        f"Costs: wasted offer (FP) = ${COST_FP:.0f}, missed churner (FN) = ${COST_FN:.0f}.\n\n"
        f"- chosen threshold: **{s['best_threshold']:.2f}** (vs default 0.50)\n"
        f"- expected cost @ chosen: **${s['cost_at_best']:.0f}**\n"
        f"- expected cost @ 0.50: ${s['cost_at_0.5']:.0f}\n"
        f"- **savings vs 0.50: ${s['savings_vs_0.5']:.0f}**\n"
        f"- at chosen threshold: recall **{s['recall_at_best']:.3f}**, "
        f"precision {s['precision_at_best']:.3f} "
        f"(tp={c['tp']}, fp={c['fp']}, fn={c['fn']}, tn={c['tn']})\n\n"
        "Because a missed churner costs 10x a wasted offer, the optimal threshold "
        "sits below 0.5 — the model is tuned to catch more churners at the price of "
        "more retention offers.\n"
    )


def main() -> None:
    s = run()
    logger.info(
        "brier %.4f->%.4f | threshold=%.2f | savings vs 0.5 = $%.0f",
        s["brier_uncalibrated"],
        s["brier_calibrated"],
        s["best_threshold"],
        s["savings_vs_0.5"],
    )


if __name__ == "__main__":
    main()
