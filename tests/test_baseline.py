from churn_risk_service.baseline import train_and_report


def test_logreg_has_signal_and_accuracy_lies():
    res = train_and_report(write=False)
    # LogisticRegression shows real ranking signal above the floor
    assert res["logreg"]["pr_auc"] > res["dummy"]["pr_auc"]
    assert res["logreg"]["roc_auc"] > 0.75
    # The accuracy trap: dummy looks decent on accuracy but has no signal
    assert res["dummy"]["accuracy"] > 0.70
    assert res["dummy"]["roc_auc"] == 0.5
    assert res["dummy"]["pr_auc"] < 0.35
