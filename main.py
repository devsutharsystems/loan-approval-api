import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from preprocessing_utils import encode_features, scale_features

app = FastAPI(
    title="Credit Risk Classification and Deployment Engine",
    description="A machine learning-powered REST API for classifying the credit risk of loan applicants.",
    version="1.0.0",
)

# Load trained artifacts once at startup
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoder_y.pkl", "rb") as f:
    label_encoder_y = pickle.load(f)


class CreditRiskRequest(BaseModel):
    gender: str = Field(..., examples=["Male"])
    married: str = Field(..., examples=["Yes"])
    dependents: str = Field(..., examples=["0"])
    education: str = Field(..., examples=["Graduate"])
    self_employed: str = Field(..., examples=["No"])
    loan_amount: float = Field(..., examples=[150.0])
    loan_amount_term: float = Field(..., examples=[360.0])
    credit_history: float = Field(..., examples=[1.0])


@app.get("/")
def root():
    return {
        "status": "Credit Risk Classification and Deployment Engine is running"
    }


@app.post("/predict-risk")
def predict(application: CreditRiskRequest):
    # Build a single-row DataFrame matching the training feature columns/order
    row = pd.DataFrame(
        [
            {
                "Gender": application.gender,
                "Married": application.married,
                "Dependents": application.dependents,
                "Education": application.education,
                "Self_Employed": application.self_employed,
                "LoanAmount": application.loan_amount,
                "Loan_Amount_Term": application.loan_amount_term,
                "Credit_History": application.credit_history,
            }
        ]
    )

    X = encode_features(row, encoders, fit=False)
    X = scale_features(X, scaler, fit=False)

    prediction = model.predict(X)[0]
    confidence = max(model.predict_proba(X)[0])

    status = label_encoder_y.inverse_transform([prediction])[0]

    return {
        "credit_risk": "Low Risk" if status == "Y" else "High Risk",
        "confidence": round(float(confidence), 3),
    }
