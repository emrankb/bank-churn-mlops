"""Synthetic model training tests with no real customer records."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from bank_churn.data import CATEGORICAL, IDENTIFIERS, NUMERIC
from bank_churn.training import (
    METRICS, model_candidates, save_training_artifacts, train_and_compare,
)


def synthetic_churn_data(n=90):
    rng = np.random.default_rng(21)
    rows = []
    for i in range(n):
        age = int(rng.integers(23, 73))
        inactive = int(i % 3 == 0)
        churn = int((i % 2 == 0))
        rows.append({
            "CustomerId": i + 1,
            "CustomerId.1": 15500000 + i,
            "Surname": f"Fictional{i}",
            "CreditScore": int(rng.integers(450, 850)),
            "Age": age,
            "Tenure": int(rng.integers(0, 11)),
            "Balance": float(rng.uniform(0, 120000)),
            "NumOfProducts": int(rng.integers(1, 4)),
            "HasCrCard": int(i % 2),
            "IsActiveMember": 1 - inactive,
            "EstimatedSalary": float(rng.uniform(20000, 150000)),
            "Geography": ["France", "Germany", "Spain"][i % 3],
            "Gender": ["Female", "Male"][i % 2],
            "Exited": churn,
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def trained():
    return train_and_compare(synthetic_churn_data())


def test_model_selection_is_train_cv_f1(trained):
    _model, report = trained
    expected = max(report.cross_validation, key=lambda name: report.cross_validation[name]["f1"])
    assert report.selected_model == expected
    assert report.selection_metric == "cross_validated_f1"
    assert set(report.cross_validation) == set(model_candidates())
    for scores in report.cross_validation.values():
        assert set(scores) == set(METRICS)
        assert all(0.0 <= value <= 1.0 for value in scores.values())


def test_heldout_split_counts_and_metrics(trained):
    _model, report = trained
    assert report.training_rows == 72
    assert report.test_rows == 18
    assert report.positive_train == 36
    assert report.positive_test == 9
    assert set(report.test_metrics) == set(METRICS)
    assert all(0.0 <= value <= 1.0 for value in report.test_metrics.values())


def test_pipeline_contains_preprocessing_and_classifier(trained):
    model, _report = trained
    assert isinstance(model, Pipeline)
    assert set(model.named_steps) == {"preprocess", "model"}
    original_features = synthetic_churn_data().loc[:, [*NUMERIC, *CATEGORICAL]]
    predictions = model.predict(original_features.iloc[:3])
    assert len(predictions) == 3
    assert set(predictions).issubset({0, 1})


def test_training_is_reproducible():
    one, first = train_and_compare(synthetic_churn_data(48))
    two, second = train_and_compare(synthetic_churn_data(48))
    assert first.as_dict() == second.as_dict()
    inputs = synthetic_churn_data(48).loc[:, [*NUMERIC, *CATEGORICAL]]
    assert np.array_equal(one.predict(inputs), two.predict(inputs))


def test_identifiers_and_surnames_do_not_affect_predictions(trained):
    model, _ = trained
    frame = synthetic_churn_data()
    features = frame.loc[:, [*NUMERIC, *CATEGORICAL]]
    before = model.predict(features)
    changed = frame.copy()
    for column in IDENTIFIERS:
        changed[column] = "different"
    after = model.predict(changed.loc[:, [*NUMERIC, *CATEGORICAL]])
    assert np.array_equal(before, after)


def test_too_few_positive_examples_are_rejected():
    frame = synthetic_churn_data(30)
    frame["Exited"] = [1] * 3 + [0] * 27
    with pytest.raises(ValueError, match="eight"):
        train_and_compare(frame)


def test_save_local_artifacts_with_no_customer_rows(trained, tmp_path):
    model, report = trained
    output = tmp_path / "model"
    metadata = save_training_artifacts(model, report, output)
    text = (output / "metrics.json").read_text()
    assert json.loads(text) == metadata
    assert (output / "model.joblib").is_file()
    assert "Fictional0" not in text
    assert "15500000" not in text
    assert set(metadata["input_features"]) == set(NUMERIC) | set(CATEGORICAL)
    with pytest.raises(FileExistsError):
        save_training_artifacts(model, report, output)
