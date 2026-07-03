import pandas as pd

from churn_risk_service.data import (
    CATEGORICAL_FEATURES,
    ID_COL,
    NUMERIC_FEATURES,
    TARGET,
    clean,
    load_and_split,
    load_raw,
)


def test_clean_fixes_quirks():
    x, y = clean(load_raw())
    # target binarized to 0/1
    assert set(y.unique()) == {0, 1}
    # id and target dropped from features
    assert ID_COL not in x.columns
    assert TARGET not in x.columns
    # TotalCharges is numeric with no missing (the 11 blanks are filled)
    assert pd.api.types.is_numeric_dtype(x["TotalCharges"])
    assert x["TotalCharges"].isna().sum() == 0
    # all declared features are present
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) <= set(x.columns)
    # known churn base rate
    assert round(y.mean(), 3) == 0.265


def test_split_is_stratified_and_disjoint():
    s = load_and_split()
    total = len(s.X_train) + len(s.X_val) + len(s.X_test)
    assert total == 7043
    # ~60/20/20
    assert abs(len(s.X_train) / total - 0.60) < 0.01
    assert abs(len(s.X_val) / total - 0.20) < 0.01
    assert abs(len(s.X_test) / total - 0.20) < 0.01
    # churn rate preserved across splits (stratification)
    for y in (s.y_train, s.y_val, s.y_test):
        assert abs(y.mean() - 0.265) < 0.01
    # no index overlap between splits
    idx = set(s.X_train.index) | set(s.X_val.index) | set(s.X_test.index)
    assert len(idx) == total
