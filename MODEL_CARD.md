# Model Card — churn-risk-service

## Model details
- **Task:** binary classification — will a telecom customer churn?
- **Model:** logistic regression (`logreg_feat`) on scaled numeric + one-hot
  categorical features, plus row-wise engineered features, in a single
  scikit-learn `Pipeline`.
- **Version:** v2 (Day 9 selection). Selected over raw-feature logreg and gradient
  boosting on validation **PR-AUC**.

## Intended use
- **In scope:** rank customers by churn risk to prioritize retention outreach; a
  decision threshold converts risk into an "send retention offer?" action.
- **Out of scope:** individual-level automated decisions with adverse consequences,
  fairness-sensitive gating, or use on populations unlike the training data.

## Training data
- IBM sample **Telco Customer Churn**: 7,043 customers, 19 features, **26.5% churn**.
- Cleaning: `TotalCharges` blank strings (11 new customers, tenure 0) coerced to 0.
- Split: **stratified 60/20/20** train/val/test. Test held out until final selection.

## Features
- Raw: tenure, MonthlyCharges, TotalCharges, SeniorCitizen, contract, services, etc.
- Engineered (row-wise, leakage-safe): `num_addon_services`, `charges_per_tenure`,
  `has_internet`, `tenure_bucket`. Fit inside the pipeline; a test asserts they use
  no dataset-level statistics.

## Evaluation (validation split)
| metric | value |
| --- | --- |
| PR-AUC (headline) | 0.647 |
| ROC-AUC | 0.837 |
| accuracy | 0.804 |
| Brier score | 0.138 |

Accuracy is *not* the headline: a majority-class dummy scores 0.735 accuracy with
zero signal (PR-AUC ≈ base rate). See `reports/comparison.md`.

## Calibration
Checked isotonic calibration; Brier did **not** improve (0.1379 → 0.1384) — logistic
regression is already well-calibrated on this data. The **uncalibrated** model is
deployed. See `reports/calibration.md`.

## Decision threshold (business cost)
- Cost assumptions: wasted retention offer (FP) = **$50**; missed churner (FN) = **$500** (10×).
- Decision rule: offer when churn probability > FP/FN = **0.10**; empirical optimum
  on validation = **0.07**.
- At the chosen threshold: **recall 0.960**, precision 0.378 (tp=359, fp=591, fn=15).
- Expected validation cost: **$55,000** vs **$103,300** at the default 0.5 threshold —
  ~**$48k** cheaper. The asymmetry justifies an aggressive, recall-heavy operating point.

## Limitations
- Snapshot dataset; no temporal validation — real deployment needs time-based splits.
- Cost numbers are illustrative; recompute the threshold with the true offer cost and
  customer lifetime value before production use.
- Precision at the chosen threshold is low by design (many cheap FPs to avoid costly FNs).

## Ethical considerations
- The data includes `gender` and `SeniorCitizen`. This model has **not** undergone a
  fairness audit; before production, evaluate error-rate parity across these groups and
  consider excluding protected attributes.
- Retention offers driven by the model should not create discriminatory pricing.

## Maintenance
- Monitor input **drift** and score distribution over time (Day 13).
- Re-fit on recent data periodically; re-derive the threshold if costs change.
