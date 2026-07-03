# Calibration & threshold (validation split)

## Calibration (Brier score, lower is better)

- uncalibrated: **0.1379**
- isotonic-calibrated: **0.1384**

Isotonic calibration did not improve Brier — logistic regression is already well-calibrated here — so we deploy the uncalibrated model and select the threshold on it.

## Business-cost threshold

Costs: wasted offer (FP) = $50, missed churner (FN) = $500.

- chosen threshold: **0.07** (vs default 0.50)
- expected cost @ chosen: **$55000**
- expected cost @ 0.50: $103300
- **savings vs 0.50: $48300**
- at chosen threshold: recall **0.960**, precision 0.378 (tp=359, fp=591, fn=15, tn=444)

Because a missed churner costs 10x a wasted offer, the optimal threshold sits below 0.5 — the model is tuned to catch more churners at the price of more retention offers.
