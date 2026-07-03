import pytest
from fastapi.testclient import TestClient

from churn_risk_service.api import app

client = TestClient(app)

VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.0,
    "TotalCharges": 178.0,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"]


def test_predict_returns_valid_response():
    r = client.post("/predict", json=VALID_CUSTOMER)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_prediction"], bool)
    assert body["churn_prediction"] == (body["churn_probability"] >= body["threshold"])


def test_high_risk_profile_scores_higher_than_low_risk():
    # Month-to-month, low tenure, high charges = classic churn profile.
    high = client.post("/predict", json=VALID_CUSTOMER).json()["churn_probability"]
    low_customer = {
        **VALID_CUSTOMER,
        "Contract": "Two year",
        "tenure": 70,
        "InternetService": "No",
        "MonthlyCharges": 20.0,
        "TotalCharges": 1400.0,
    }
    low = client.post("/predict", json=low_customer).json()["churn_probability"]
    assert high > low


@pytest.mark.parametrize(
    "bad",
    [
        {**VALID_CUSTOMER, "tenure": -1},  # violates ge=0
        {**VALID_CUSTOMER, "SeniorCitizen": 5},  # violates le=1
        {k: v for k, v in VALID_CUSTOMER.items() if k != "gender"},  # missing field
    ],
)
def test_invalid_input_rejected(bad):
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
