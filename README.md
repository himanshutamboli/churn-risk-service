# churn-risk-service

[![CI](https://github.com/himanshutamboli/churn-risk-service/actions/workflows/ci.yml/badge.svg)](https://github.com/himanshutamboli/churn-risk-service/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/lint-ruff-orange.svg)](https://github.com/astral-sh/ruff)

> Customer churn prediction shipped as **software**, not a notebook: honest evaluation, a trained model behind an API, calibration, drift monitoring, Docker, and CI. Built on the IBM Telco Customer Churn dataset (7,043 customers, ~26.5% churn).

## Why this exists

Most churn "projects" report 80% accuracy and stop. On a 26% base rate, a model that predicts *"nobody churns"* also scores ~74% accuracy — and is worthless. This repo is built around honest evaluation and productionization instead.

## Baseline evaluation (validation split)

| model | accuracy | precision | recall | f1 | roc_auc | **pr_auc** |
| --- | --- | --- | --- | --- | --- | --- |
| dummy (majority) | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 | **0.265** |
| logistic regression | 0.803 | 0.658 | 0.535 | 0.590 | 0.836 | **0.642** |

**The lesson:** the dummy's 73.5% accuracy is pure base-rate illusion — its PR-AUC equals the 26.5% churn rate (no signal) and ROC-AUC is a coin flip. Logistic regression barely moves accuracy but roughly **doubles PR-AUC**. On imbalanced problems, **PR-AUC is the headline**, not accuracy. Full report: [`reports/eval_baseline.md`](reports/eval_baseline.md).

**Leakage guard:** preprocessing (scaling, one-hot) is fit inside the `Pipeline` on the *training* split only — validation/test never leak into the fitted transforms. Split is stratified 60/20/20; test is untouched until final model selection.

## Model iteration (Day 9)

Engineered features (add-on service count, charges-per-tenure, has-internet, tenure bucket) are **row-wise, fit inside the pipeline** — a test asserts they're computed per-row with no dataset-level statistics, so they can't leak.

| model | accuracy | precision | recall | f1 | roc_auc | **pr_auc** |
| --- | --- | --- | --- | --- | --- | --- |
| dummy | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 | 0.265 |
| logreg_base | 0.803 | 0.658 | 0.535 | 0.590 | 0.836 | 0.642 |
| **logreg_feat** | 0.804 | 0.666 | 0.527 | 0.588 | 0.837 | **0.647** |
| hgb_feat | 0.747 | 0.517 | 0.711 | 0.598 | 0.827 | 0.625 |

**Honest finding:** features gave logistic regression a small PR-AUC lift (0.642 → **0.647**). Gradient boosting (`class_weight=balanced`) did **not** improve PR-AUC — it shifted to a high-recall/low-precision operating point (0.71 recall), useful only if the business cost of a missed churner dominates. On this near-linear dataset, **tuned logistic regression is the model to beat**; not every problem needs a tree ensemble. Selected v2: `logreg_feat`. Full table: [`reports/comparison.md`](reports/comparison.md).

## Calibration & decision threshold (Day 10)

- **Calibration:** checked isotonic calibration; Brier didn't improve (0.1379 → 0.1384) — logistic regression is already well-calibrated, so we ship it uncalibrated and *document* the check.
- **Threshold by business cost:** with a wasted offer (FP) = $50 and a missed churner (FN) = $500, the decision rule is "offer when churn prob > 0.10"; empirical optimum **0.07**. That lifts recall to **0.96** and cuts expected validation cost from **$103k → $55k** vs the arbitrary 0.5 threshold.

Full write-up: [`reports/calibration.md`](reports/calibration.md) · complete [`MODEL_CARD.md`](MODEL_CARD.md).

## API (Day 11)

A **FastAPI** service serves the persisted model with typed pydantic validation
(auto-docs at `/docs`). It returns a probability *and* the business decision at the
Day-10 threshold. The service self-bootstraps: if no model file exists it trains and
persists one on first request.

```bash
uv run python -m churn_risk_service.train                    # persist models/churn_model.joblib
uv run uvicorn churn_risk_service.api:app --reload           # serve on :8000

curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":1,
  "PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic",
  "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
  "StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month",
  "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
  "MonthlyCharges":95.0,"TotalCharges":95.0}'
# -> {"churn_probability":0.630,"churn_prediction":true,"threshold":0.07,"model_version":"v2-logreg_feat"}
```

Invalid input (e.g. `tenure: -1`, unknown field) is rejected with `422`. Tested with `TestClient`.

## Docker (Day 12)

Multi-stage build: the builder installs deps and **bakes the model** (`train`) into the image; the runtime stage ships a slim image with just the venv + model.

```bash
docker build -t churn-risk-service .
docker run -p 8000:8000 churn-risk-service
curl -s localhost:8000/predict -H 'content-type: application/json' -d @examples/request.json
```

CI builds the image on every push and **smoke-tests the running container** (`/health` + `/predict`), so a broken Dockerfile fails the pipeline.

## Monitoring (Day 13)

- **Drift detection** via **PSI** (Population Stability Index) — a dependency-light, transparent check. A reference profile is built from the training features (quantile bins for numeric, frequencies for categorical); any batch is scored against it. Convention: PSI < 0.1 none · 0.1–0.25 moderate · > 0.25 major.
- **Validated:** test-split-vs-train reports ~0 PSI everywhere; a synthetically shifted batch (+$60 charges, all month-to-month) is flagged **major** (MonthlyCharges PSI 7.5, Contract 5.8). See [`reports/drift.md`](reports/drift.md).
- **Structured logging:** `/predict` emits one JSON line per request (features, probability, prediction, latency) — the raw material for online drift monitoring.

```bash
uv run python -m churn_risk_service.monitoring   # write reports/drift.md
```

## Quickstart

```bash
uv sync --dev
uv run python -m churn_risk_service.baseline   # train baseline + write reports/eval_baseline.md
uv run pytest                                   # data + baseline tests
```

## Project structure

```
churn-risk-service/
├── data/telco_churn.csv          # IBM Telco Customer Churn (committed, self-contained)
├── src/churn_risk_service/
│   ├── data.py                   # load, clean (TotalCharges quirk), stratified split
│   ├── baseline.py               # preprocessing pipeline, dummy vs logreg, honest eval
│   ├── features.py               # row-wise engineered features (leakage-safe)
│   ├── model.py                  # pipeline builder for all model variants
│   ├── compare.py                # train all variants -> comparison table
│   ├── train.py                  # fit final model + persist serving artifact
│   ├── schemas.py                # pydantic request/response
│   ├── service.py                # model load/train + scoring
│   ├── api.py                    # FastAPI app (/health, /predict) + structured logging
│   └── monitoring.py             # PSI drift detection
├── tests/                        # split/leakage guards, "accuracy lies", API (TestClient)
├── MODEL_CARD.md
└── reports/                      # generated eval_baseline.md + comparison.md + calibration.md
```

## Roadmap

| Day | Deliverable |
|---|---|
| 8 ✅ | Framing + baseline + honest eval (PR-AUC, leakage guard) |
| 9 ✅ | Feature engineering (no leakage) + model v2 + comparison table |
| 10 ✅ | Calibration + business-cost threshold + MODEL_CARD.md |
| 11 ✅ | FastAPI `/predict` + persisted model + pydantic validation |
| 12 ✅ | Multi-stage Dockerfile + container smoke-test in CI |
| 13 ✅ | PSI drift detection + structured request logging |
| 14 | Ship v1.0 |

## Data source

IBM sample "Telco Customer Churn" dataset, committed to the repo for reproducible, self-contained CI.

## License

MIT
