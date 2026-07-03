"""Load, clean, and split the Telco Customer Churn dataset.

Source: IBM sample "Telco Customer Churn" (7,043 customers, ~26.5% churn).
"""

from pathlib import Path
from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/telco_churn.csv")

TARGET = "Churn"
ID_COL = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


class Split(NamedTuple):
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV as-is."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (features, target). Drops the ID, fixes TotalCharges, binarizes target."""
    df = df.drop(columns=[ID_COL])
    # 11 new customers (tenure == 0) have a blank TotalCharges string -> 0 spend so far.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    y = (df[TARGET] == "Yes").astype(int)
    x = df.drop(columns=[TARGET])
    return x, y


def split(x: pd.DataFrame, y: pd.Series, seed: int = 42) -> Split:
    """Stratified 60/20/20 train/val/test split."""
    x_tmp, x_test, y_tmp, y_test = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=seed
    )
    # 0.25 of the remaining 0.80 -> 0.20 overall
    x_train, x_val, y_train, y_val = train_test_split(
        x_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=seed
    )
    return Split(x_train, x_val, x_test, y_train, y_val, y_test)


def load_and_split(path: Path = DATA_PATH, seed: int = 42) -> Split:
    """Convenience: load -> clean -> split."""
    x, y = clean(load_raw(path))
    return split(x, y, seed=seed)
