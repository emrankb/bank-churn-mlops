"""Validate customer churn data and define leakage-safe preprocessing steps."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Exited"
IDENTIFIERS = ("CustomerId", "CustomerId.1", "Surname")
NUMERIC = (
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
)
CATEGORICAL = ("Geography", "Gender")
EXPECTED = (*IDENTIFIERS, *NUMERIC, *CATEGORICAL, TARGET)
BINARY = ("HasCrCard", "IsActiveMember", TARGET)


def load_churn_data(path: str | Path) -> pd.DataFrame:
    """Read a local CSV and check its schema before any model fitting."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    return validate_churn_data(frame)


def validate_churn_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject duplicate IDs, invalid target labels, and malformed feature data.

    Missing feature cells remain missing so the training-only preprocessing
    pipeline can impute them without leaking test data into training.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Churn data must be a nonempty DataFrame.")
    if not frame.columns.is_unique:
        raise ValueError("Column headers must be unique after CSV loading.")
    missing = set(EXPECTED) - set(frame.columns)
    unexpected = set(frame.columns) - set(EXPECTED)
    if missing or unexpected:
        raise ValueError(
            f"Unexpected churn schema: missing={sorted(missing)}, "
            f"extra={sorted(unexpected)}"
        )
    data = frame.loc[:, EXPECTED].copy()
    if data["CustomerId.1"].isna().any() or data["CustomerId.1"].duplicated().any():
        raise ValueError("Customer identifiers must be present and unique.")
    if data["CustomerId"].isna().any() or data["CustomerId"].duplicated().any():
        raise ValueError("Row identifiers must be present and unique.")
    for field in NUMERIC:
        raw = data[field]
        converted = pd.to_numeric(raw, errors="coerce")
        if (raw.notna() & converted.isna()).any():
            raise ValueError(f"Non-numeric entries in {field}.")
        data[field] = converted
    for field in BINARY:
        values = set(data[field].dropna().unique())
        if not values <= {0, 1} or (field == TARGET and data[field].isna().any()):
            raise ValueError(f"{field} must contain binary 0/1 values.")
    if data[TARGET].nunique() != 2:
        raise ValueError("Churn target must contain both classes.")
    for field in ("CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"):
        if (data[field].dropna() < 0).any():
            raise ValueError(f"{field} cannot contain negative values.")
    if (data["Age"].dropna() == 0).any():
        raise ValueError("Age must be greater than zero.")
    for field in CATEGORICAL:
        data[field] = data[field].astype("string").str.strip()
        if (data[field].dropna() == "").any():
            raise ValueError(f"{field} has empty category labels.")
    return data


def split_features_target(frame: pd.DataFrame):
    """Keep identifiers, names, and outcome labels out of training features."""
    data = validate_churn_data(frame)
    features = data.loc[:, [*NUMERIC, *CATEGORICAL]].copy()
    target = data[TARGET].astype(int)
    return features, target


def make_preprocessor() -> ColumnTransformer:
    """Return an unfitted transformer to train *only* on a training fold."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(NUMERIC)),
            ("categorical", categorical, list(CATEGORICAL)),
        ],
        remainder="drop",
    )
