from churn_risk_service.data import load_and_split
from churn_risk_service.monitoring import build_reference, compute_drift


def _psi_by_feature(rows):
    return {r["feature"]: r["psi"] for r in rows}


def test_no_drift_against_self():
    s = load_and_split()
    ref = build_reference(s.X_train)
    rows = compute_drift(ref, s.X_train)
    # a distribution compared to itself has ~zero PSI everywhere
    assert all(r["psi"] < 0.10 for r in rows)
    assert all(r["severity"] == "none" for r in rows)


def test_detects_numeric_drift():
    s = load_and_split()
    ref = build_reference(s.X_train)
    shifted = s.X_test.copy()
    shifted["MonthlyCharges"] = shifted["MonthlyCharges"] + 60
    psi = _psi_by_feature(compute_drift(ref, shifted))
    assert psi["MonthlyCharges"] > 0.25  # major drift


def test_detects_categorical_drift():
    s = load_and_split()
    ref = build_reference(s.X_train)
    shifted = s.X_test.copy()
    shifted["Contract"] = "Month-to-month"  # collapse the distribution
    psi = _psi_by_feature(compute_drift(ref, shifted))
    assert psi["Contract"] > 0.25
