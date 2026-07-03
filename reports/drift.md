# Drift report (PSI)

## Test split vs training reference

| feature | type | psi | severity |
| --- | --- | --- | --- |
| MonthlyCharges | numeric | 0.015 | none |
| tenure | numeric | 0.014 | none |
| TotalCharges | numeric | 0.009 | none |
| OnlineBackup | categorical | 0.005 | none |
| StreamingTV | categorical | 0.004 | none |
| MultipleLines | categorical | 0.004 | none |
| PhoneService | categorical | 0.002 | none |
| StreamingMovies | categorical | 0.001 | none |
| TechSupport | categorical | 0.001 | none |
| Contract | categorical | 0.000 | none |
| DeviceProtection | categorical | 0.000 | none |
| OnlineSecurity | categorical | 0.000 | none |
| Dependents | categorical | 0.000 | none |
| PaymentMethod | categorical | 0.000 | none |
| InternetService | categorical | 0.000 | none |
| SeniorCitizen | categorical | 0.000 | none |
| gender | categorical | 0.000 | none |
| Partner | categorical | 0.000 | none |
| PaperlessBilling | categorical | 0.000 | none |

## Synthetically shifted batch (+$60 charges, all month-to-month)

| feature | type | psi | severity |
| --- | --- | --- | --- |
| MonthlyCharges | numeric | 7.518 | major |
| Contract | categorical | 5.809 | major |
| tenure | numeric | 0.014 | none |
| TotalCharges | numeric | 0.009 | none |
| OnlineBackup | categorical | 0.005 | none |
| StreamingTV | categorical | 0.004 | none |
| MultipleLines | categorical | 0.004 | none |
| PhoneService | categorical | 0.002 | none |
| StreamingMovies | categorical | 0.001 | none |
| TechSupport | categorical | 0.001 | none |
| DeviceProtection | categorical | 0.000 | none |
| OnlineSecurity | categorical | 0.000 | none |
| Dependents | categorical | 0.000 | none |
| PaymentMethod | categorical | 0.000 | none |
| InternetService | categorical | 0.000 | none |
| SeniorCitizen | categorical | 0.000 | none |
| gender | categorical | 0.000 | none |
| Partner | categorical | 0.000 | none |
| PaperlessBilling | categorical | 0.000 | none |
