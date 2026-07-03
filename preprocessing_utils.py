"""
Shared preprocessing utilities for the Loan Approval Prediction project.
Used identically by train.py (training) and main.py (API inference) to
guarantee consistent encoding between training and prediction time.
"""

import numpy as np
import pandas as pd

# Columns used as model input, in the exact order the model expects
FEATURE_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

CATEGORICAL_COLUMNS = ["Gender", "Married", "Dependents", "Education", "Self_Employed"]


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using the same strategy for train and inference data."""
    df = df.copy()

    df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
    df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
    df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
    df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
    df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].mean())

    return df


def encode_features(df: pd.DataFrame, encoders: dict, fit: bool = False) -> np.ndarray:
    """
    Encode categorical columns using LabelEncoders.

    If fit=True, fits new encoders on this data (only used during training).
    If fit=False, uses the already-fitted encoders passed in (used at inference).
    This guarantees train and inference always use identical category mappings.
    """
    df = df.copy()

    for col in CATEGORICAL_COLUMNS:
        if fit:
            df[col] = encoders[col].fit_transform(df[col])
        else:
            df[col] = encoders[col].transform(df[col])

    return df[FEATURE_COLUMNS].values


def scale_features(X: np.ndarray, scaler, fit: bool = False) -> np.ndarray:
    """
    Scale numeric features using a StandardScaler.

    If fit=True, fits the scaler on this data (training only).
    If fit=False, only transforms using an already-fitted scaler (inference).
    """
    if fit:
        return scaler.fit_transform(X)
    return scaler.transform(X)
