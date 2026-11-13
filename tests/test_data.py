"""Synthetic unit tests; no private customer rows in Git."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from bank_churn.data import (
    CATEGORICAL, IDENTIFIERS, NUMERIC, TARGET,
    load_churn_data, make_preprocessor, split_features_target,
    validate_churn_data,
)
from bank_churn.eda import summarize_churn, write_eda


@pytest.fixture
def sample():
    rows = []
    for i in range(12):
        rows.append({
            "CustomerId": i + 1,
            "CustomerId.1": 15600000 + i,
            "Surname": f"Test{i}",
            "CreditScore": 600 + i,
            "Geography": ("France", "Germany", "Spain")[i % 3],
            "Gender": ("Female", "Male")[i % 2],
            "Age": 30 + i,
            "Tenure": i % 5,
            "Balance": 100.0 * i,
            "NumOfProducts": 1 + i % 2,
            "HasCrCard": i % 2,
            "IsActiveMember": (i + 1) % 2,
            "EstimatedSalary": 50000.0 + i,
            "Exited": i % 2,
        })
    return pd.DataFrame(rows)


def test_csv_roundtrip(sample, tmp_path):
    path = tmp_path / "Churn_Modeling.csv"
    sample.to_csv(path, index=False)
    data = load_churn_data(path)
    assert len(data) == 12
    assert int(data[TARGET].sum()) == 6


def test_feature_contract_excludes_sensitive_identifiers_and_target(sample):
    features, labels = split_features_target(sample)
    assert set(features.columns) == set(NUMERIC) | set(CATEGORICAL)
    assert not set(IDENTIFIERS) & set(features.columns)
    assert TARGET not in features
    assert labels.tolist() == sample[TARGET].tolist()


def test_duplicate_customer_id_fails(sample):
    sample.loc[1, "CustomerId.1"] = sample.loc[0, "CustomerId.1"]
    with pytest.raises(ValueError, match="identifiers"):
        validate_churn_data(sample)


@pytest.mark.parametrize("field", ["Geography", "Exited", "CreditScore"])
def test_required_column_fails(sample, field):
    with pytest.raises(ValueError, match="schema"):
        validate_churn_data(sample.drop(columns=[field]))


def test_invalid_target_fails(sample):
    sample.loc[0, "Exited"] = 4
    with pytest.raises(ValueError, match="binary"):
        validate_churn_data(sample)


def test_bad_numeric_value_fails(sample):
    sample["Balance"] = sample["Balance"].astype(object)
    sample.loc[0, "Balance"] = "not a balance"
    with pytest.raises(ValueError, match="Non-numeric"):
        validate_churn_data(sample)


def test_unknown_category_is_supported(sample):
    features, _ = split_features_target(sample)
    pipeline = make_preprocessor()
    train = features.iloc[:-1]
    test = features.iloc[-1:].copy()
    test["Geography"] = "Another Region"
    pipeline.fit(train)
    result = pipeline.transform(test)
    assert result.shape[0] == 1
    assert result.shape[1] > len(NUMERIC)


def test_missing_features_are_imputed(sample):
    sample.loc[0, "Age"] = None
    sample.loc[1, "Gender"] = None
    features, _ = split_features_target(sample)
    transformed = make_preprocessor().fit_transform(features)
    assert transformed.shape[0] == len(sample)
    assert not pd.isna(transformed).any()


def test_grouped_counts_and_pivot_equivalent(sample):
    report = summarize_churn(sample)
    assert report["rows"] == 12
    assert report["churned"] == 6
    assert report["churn_rate"] == 0.5
    assert sum(x["customers"] for x in report["grouped"]["Geography"]) == 12
    assert sum(x["churned"] for x in report["grouped"]["Gender"]) == 6
    assert "Age" in report["numeric_summary"]


def test_write_eda_has_no_customer_rows(sample, tmp_path):
    write_eda(sample, tmp_path)
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["rows"] == 12
    assert (tmp_path / "churn_by_geography.png").is_file()
    assert "Test0" not in (tmp_path / "summary.json").read_text()
    assert "15600000" not in (tmp_path / "summary.json").read_text()
