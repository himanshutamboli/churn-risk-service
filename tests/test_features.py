from churn_risk_service.data import clean, load_raw
from churn_risk_service.features import ENG_CATEGORICAL, ENG_NUMERIC, add_features


def test_features_are_row_independent():
    """Engineering a subset of rows must equal engineering the full frame then
    slicing — proving no dataset-level (leaky) computation."""
    x, _ = clean(load_raw())
    full = add_features(x)
    idx = [0, 100, 500, 1234]
    part = add_features(x.iloc[idx])
    full_rows = full.iloc[idx]
    for col in ENG_NUMERIC + ENG_CATEGORICAL:
        assert list(part[col].values) == list(full_rows[col].values)


def test_engineered_columns_present():
    x, _ = clean(load_raw())
    out = add_features(x)
    assert set(ENG_NUMERIC + ENG_CATEGORICAL) <= set(out.columns)
    assert out["num_addon_services"].between(0, 6).all()
