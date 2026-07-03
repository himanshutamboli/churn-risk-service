"""Production-safe feature engineering.

Every feature is a *row-wise* transform of a single customer's attributes with
fixed rules — no target, no dataset-level statistics (e.g. no mean/target
encoding fit on the data). That guarantees: (a) no leakage from the label or from
other rows, and (b) identical behavior at training time and at single-row
inference time. A test asserts this row-independence.
"""

import pandas as pd

ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

ENG_NUMERIC = ["num_addon_services", "charges_per_tenure", "has_internet"]
ENG_CATEGORICAL = ["tenure_bucket"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered columns added (pure, row-wise)."""
    df = df.copy()
    df["num_addon_services"] = (df[ADDON_SERVICES] == "Yes").sum(axis=1)
    df["charges_per_tenure"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df["has_internet"] = (df["InternetService"] != "No").astype(int)
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 10_000],
        labels=["0-12", "12-24", "24-48", "48+"],
    ).astype(str)
    return df
