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
│   └── baseline.py               # preprocessing pipeline, dummy vs logreg, honest eval
├── tests/                        # cleaning/split invariants + "accuracy lies" guard
└── reports/eval_baseline.md      # generated evaluation report
```

## Roadmap

| Day | Deliverable |
|---|---|
| 8 ✅ | Framing + baseline + honest eval (PR-AUC, leakage guard) |
| 9 | Feature engineering (no future leakage) + model v2 |
| 10 | Calibration + threshold tied to business cost + model card |
| 11 | FastAPI `/predict` endpoint |
| 12 | Dockerfile + API tests in CI |
| 13 | Drift monitoring + structured logging |
| 14 | Ship v1.0 |

## Data source

IBM sample "Telco Customer Churn" dataset, committed to the repo for reproducible, self-contained CI.

## License

MIT
