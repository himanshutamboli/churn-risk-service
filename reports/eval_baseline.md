# Baseline evaluation (validation split)

Churn base rate (positive class): **0.265**. Metrics are on the held-out validation set; test stays untouched until final selection.

| model | accuracy | precision | recall | f1 | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- |
| dummy | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 | 0.265 |
| logreg | 0.803 | 0.658 | 0.535 | 0.590 | 0.836 | 0.642 |

## Read this honestly

- The **dummy** (predict everyone stays) posts high **accuracy** — it just matches the majority class. Its **pr_auc ≈ the base rate**, i.e. no signal.
- **LogisticRegression** barely moves accuracy but lifts **pr_auc** and **roc_auc** well above the floor — that's real ranking signal.
- Headline metric for this problem: **PR-AUC**, not accuracy.

## Leakage guard

Preprocessing (scaling, one-hot) is fit inside the Pipeline on the training split only; validation/test rows never influence the fitted transforms.
