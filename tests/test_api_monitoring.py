"""Offline FastAPI, drift, and retraining scheduling tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from bank_churn.api import create_app
from bank_churn.monitoring import monitor_model
from bank_churn.training import train_and_compare
from scripts.monitor_and_retrain import run_cycle
from tests.test_training import synthetic_churn_data


@pytest.fixture(scope="module")
def trained():
    return train_and_compare(synthetic_churn_data(48))


@pytest.fixture
def client(trained):
    return TestClient(create_app(model=trained[0]))


def sample_request():
    row = synthetic_churn_data(48).iloc[0]
    names = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
    ]
    return {name: (int(row[name]) if name in {
        "CreditScore", "Age", "Tenure", "NumOfProducts", "HasCrCard", "IsActiveMember"
    } else float(row[name]) if name in {"Balance", "EstimatedSalary"}
                   else row[name]) for name in names}


def test_health_and_prediction(client):
    assert client.get("/health").json() == {"status": "ready", "model_loaded": True}
    output = client.post("/predict", json=sample_request())
    assert output.status_code == 200
    response = output.json()
    assert response["churn_prediction"] in (0, 1)
    assert 0 <= response["churn_probability"] <= 1
    assert response["churn_prediction"] == int(response["churn_probability"] >= 0.5)


@pytest.mark.parametrize("change", [
    {"Surname": "private"}, {"CustomerId": 123},
    {"Exited": 1}, {"Age": -1}, {"Gender": ""},
    {"HasCrCard": 3}, {"Balance": -99.0},
])
def test_invalid_request_rejected(client, change):
    body = {**sample_request(), **change}
    response = client.post("/predict", json=body)
    assert response.status_code == 422


def test_missing_field_rejected(client):
    body = sample_request()
    del body["Geography"]
    assert client.post("/predict", json=body).status_code == 422


def test_missing_model_fails_fast(tmp_path):
    with pytest.raises(RuntimeError, match="Trained model not found"):
        create_app(model_path=tmp_path / "missing.joblib")


def test_local_saved_model_loads_and_serves(tmp_path, trained):
    path = tmp_path / "model.joblib"
    joblib.dump(trained[0], path)
    api = TestClient(create_app(model_path=path))
    assert api.post("/predict", json=sample_request()).status_code == 200


def test_no_accuracy_decline_and_no_drift(trained):
    frame = synthetic_churn_data(48)
    result = monitor_model(trained[0], frame, frame.copy(),
                           baseline_accuracy=trained[1].test_metrics["accuracy"])
    assert result["observed_rows"] == 48
    assert not result["drift_alert"]
    assert all(not item["alert"] for item in result["feature_drift"].values())
    assert not result["retraining_recommended"]


def test_simulated_accuracy_drop_and_drift(trained):
    frame = synthetic_churn_data(48)
    changed = frame.copy()
    changed["Exited"] = 1 - changed["Exited"]
    changed["Age"] += 25
    result = monitor_model(
        trained[0], frame, changed,
        baseline_accuracy=1.0, accuracy_drop=0.1,
    )
    assert result["retraining_recommended"]
    assert result["accuracy_drop"] >= 0.1
    assert result["drift_alert"]
    assert result["feature_drift"]["Age"]["alert"]


def test_monitor_needs_labeled_observations(trained):
    frame = synthetic_churn_data(48)
    with pytest.raises(ValueError, match="Insufficient"):
        monitor_model(trained[0], frame, frame.iloc[:8], baseline_accuracy=0.8)


def test_cycle_records_aggregate_without_retraining(tmp_path, trained):
    ref = synthetic_churn_data(48)
    observed = ref.copy()
    observed["Exited"] = 1 - observed["Exited"]
    reference_path = tmp_path / "reference.csv"
    observed_path = tmp_path / "observed.csv"
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    ref.to_csv(reference_path, index=False)
    observed.to_csv(observed_path, index=False)
    joblib.dump(trained[0], model_path)
    metrics_path.write_text(json.dumps({
        "test_metrics": {"accuracy": 1.0},
    }))
    report = run_cycle(
        reference_path=reference_path, observed_path=observed_path,
        model_path=model_path, metrics_path=metrics_path,
        report_dir=tmp_path / "reports",
    )
    assert report["retraining_recommended"]
    assert not report["retraining_executed"]
    assert report["retraining_skip_reason"] == "not_authorized"
    output = list((tmp_path / "reports").glob("monitor-*.json"))
    assert len(output) == 1
    text = output[0].read_text()
    assert "Fictional0" not in text
    assert "15500000" not in text


def test_retrain_trigger_and_cooldown(tmp_path, trained, monkeypatch):
    ref = synthetic_churn_data(48)
    observed = ref.copy()
    observed["Exited"] = 1 - observed["Exited"]
    reference_path = tmp_path / "reference.csv"
    observed_path = tmp_path / "observed.csv"
    retraining_path = tmp_path / "fresh_labeled.csv"
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    ref.to_csv(reference_path, index=False)
    observed.to_csv(observed_path, index=False)
    ref.to_csv(retraining_path, index=False)
    joblib.dump(trained[0], model_path)
    metrics_path.write_text(json.dumps({"test_metrics": {"accuracy": 1.0}}))
    from bank_churn import experiment_tracking, training

    tracked = []
    monkeypatch.setattr(experiment_tracking, "log_experiment",
                        lambda model, scores, tracking_uri: (
                            tracked.append(scores.selected_model) or
                            {"registered_version": "2"}
                        ))
    monkeypatch.setattr(training, "train_and_compare", lambda _frame: trained)
    start = datetime(2026, 11, 12, tzinfo=timezone.utc)
    kwargs = dict(
        reference_path=reference_path, observed_path=observed_path,
        model_path=model_path, metrics_path=metrics_path,
        report_dir=tmp_path / "reports", retraining_input=retraining_path,
        tracking_database=tmp_path / "mlflow.db",
        allow_retraining=True,
    )
    first = run_cycle(**kwargs, now=start)
    assert first["retraining_executed"]
    assert first["registered_version"] == "2"
    assert len(tracked) == 1
    assert (tmp_path / "reports" / "last_retrained.json").is_file()
    second = run_cycle(**kwargs, now=start + timedelta(hours=1))
    assert not second["retraining_executed"]
    assert second["retraining_skip_reason"] == "cooldown_active"
    assert len(tracked) == 1
