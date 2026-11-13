"""Real local MLflow integration, using only invented customer data."""
from __future__ import annotations

import json

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pytest

from bank_churn.experiment_tracking import (
    CHAMPION_ALIAS, EXPERIMENT_NAME, REGISTERED_MODEL_NAME,
    local_tracking_uri, log_experiment,
)
from bank_churn.training import train_and_compare
from tests.test_training import synthetic_churn_data


@pytest.fixture(scope="module")
def fitted():
    return train_and_compare(synthetic_churn_data(48))


def test_local_uri_requires_sqlite_db(tmp_path):
    path = tmp_path / "tracking.db"
    uri = local_tracking_uri(path)
    assert uri.startswith("sqlite:////")
    assert uri.endswith("tracking.db")
    with pytest.raises(ValueError, match=".db"):
        local_tracking_uri(tmp_path / "tracking.txt")


def test_tracking_runs_and_registry(fitted, tmp_path):
    model, result = fitted
    uri = local_tracking_uri(tmp_path / "mlflow.db")
    record = log_experiment(model, result, tracking_uri=uri)
    assert record["selected_model"] == result.selected_model
    assert record["selection_metric"] == "cross_validated_f1"
    assert len(record["candidate_run_ids"]) == 3
    assert record["registered_version"] == "1"
    assert record["model_alias"] == CHAMPION_ALIAS
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    assert experiment is not None
    for name, run_id in record["candidate_run_ids"].items():
        run = client.get_run(run_id)
        assert run.data.params["classifier"] == name
        assert run.data.tags["selected"] == str(name == result.selected_model).lower()
        assert run.data.metrics["cv_f1"] == pytest.approx(
            result.cross_validation[name]["f1"]
        )
        if name == result.selected_model:
            assert run.data.metrics["holdout_f1"] == pytest.approx(
                result.test_metrics["f1"]
            )
        else:
            assert "holdout_f1" not in run.data.metrics
        # No patient/customer values or row-level records in metrics/tags/params.
        contents = json.dumps({"params": run.data.params, "tags": run.data.tags})
        assert "Fictional0" not in contents
        assert "15500000" not in contents
    alias = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    assert str(alias.version) == "1"
    reloaded = mlflow.sklearn.load_model(record["registered_uri"])
    features = synthetic_churn_data(48).drop(
        columns=["CustomerId", "CustomerId.1", "Surname", "Exited"],
    )
    assert list(reloaded.predict(features.iloc[:4])) == list(
        model.predict(features.iloc[:4])
    )
    next_record = log_experiment(model, result, tracking_uri=uri)
    assert next_record["registered_version"] == "2"
    assert str(client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS).version) == "2"


def test_rejects_non_f1_winner_and_remote_tracking(fitted, tmp_path):
    model, result = fitted
    with pytest.raises(ValueError, match="SQLite"):
        log_experiment(model, result, tracking_uri="http://localhost:5000")
    result.selection_metric = "test_accuracy"
    with pytest.raises(ValueError, match="cross-validated F1"):
        log_experiment(
            model, result, tracking_uri=local_tracking_uri(tmp_path / "tracking.db"),
        )
