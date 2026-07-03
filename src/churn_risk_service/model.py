"""Model pipelines. Feature engineering lives inside the pipeline so training and
single-row inference apply identical logic."""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn_risk_service.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from churn_risk_service.features import ENG_CATEGORICAL, ENG_NUMERIC, add_features

# kind -> (uses engineered features?, classifier factory)
KINDS = ("dummy", "logreg_base", "logreg_feat", "hgb_feat")


def _build_preprocessor(use_features: bool) -> ColumnTransformer:
    numeric = NUMERIC_FEATURES + (ENG_NUMERIC if use_features else [])
    categorical = CATEGORICAL_FEATURES + (ENG_CATEGORICAL if use_features else [])
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def build_pipeline(kind: str) -> Pipeline:
    if kind not in KINDS:
        raise ValueError(f"unknown kind: {kind}")
    use_features = kind in ("logreg_feat", "hgb_feat")

    steps = []
    if use_features:
        steps.append(("feat", FunctionTransformer(add_features)))
    steps.append(("pre", _build_preprocessor(use_features)))

    if kind == "dummy":
        clf = DummyClassifier(strategy="most_frequent")
    elif kind == "hgb_feat":
        clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
    else:
        clf = LogisticRegression(max_iter=1000)
    steps.append(("clf", clf))
    return Pipeline(steps)
