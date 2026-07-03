# Model comparison (validation split)

Headline metric: **pr_auc** (average precision) — the honest choice on a ~26% base rate. Test split stays untouched.

| model | accuracy | precision | recall | f1 | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- |
| dummy | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 | 0.265 |
| logreg_base | 0.803 | 0.658 | 0.535 | 0.590 | 0.836 | 0.642 |
| logreg_feat | 0.804 | 0.666 | 0.527 | 0.588 | 0.837 | 0.647 |
| hgb_feat | 0.747 | 0.517 | 0.711 | 0.598 | 0.827 | 0.625 |

- `logreg_base` — Day-8 baseline (raw features).
- `logreg_feat` — same model + engineered features.
- `hgb_feat` — gradient boosting + engineered features, `class_weight=balanced`.

Engineered features are row-wise and fit inside the pipeline; no leakage.
