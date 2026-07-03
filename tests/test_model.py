from churn_risk_service.compare import compare


def test_features_help_and_models_have_signal():
    res = compare(write=False)
    # engineered features give logreg at least as good PR-AUC as the raw baseline
    assert res["logreg_feat"]["pr_auc"] >= res["logreg_base"]["pr_auc"]
    # every real model clears the base-rate floor by a wide margin
    for kind in ("logreg_base", "logreg_feat", "hgb_feat"):
        assert res[kind]["pr_auc"] > 0.55
        assert res[kind]["roc_auc"] > 0.80
