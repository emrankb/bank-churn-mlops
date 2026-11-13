"""Track churn model experiments and register the F1-selected pipeline.

Only aggregate scores and model parameters are logged. Do not record customer
rows, feature values, input examples, or test predictions in tracking logs.
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from .training import TrainingResult, model_candidates

EXPERIMENT_NAME = "bank-churn-classification"
REGISTERED_MODEL_NAME = "bank-churn-classifier"
CHAMPION_ALIAS = "champion"


def local_tracking_uri(database: str | Path) -> str:
    """Build a local SQLite URI, without using a remote tracking service."""
    path = Path(database).expanduser().resolve()
    if path.suffix != ".db":
        raise ValueError("MLflow SQLite database must have a .db extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + path.as_posix()


def log_experiment(
    model,
    result: TrainingResult,
    *,
    tracking_uri: str,
    experiment_name: str = EXPERIMENT_NAME,
    registered_model_name: str = REGISTERED_MODEL_NAME,
) -> dict:
    """Log three candidate runs, register the winner, and assign its alias.

    The held-out test metrics are associated only with the selected run.
    Candidate scores are mean cross-validated metrics from training folds.
    No model is refitted or selected from the test data in this function.
    """
    if not isinstance(tracking_uri, str) or not tracking_uri.startswith("sqlite:///"):
        raise ValueError("MLflow requires an explicitly configured local SQLite backend.")
    if not experiment_name.strip() or not registered_model_name.strip():
        raise ValueError("Experiment and registered model names are required.")
    candidates = model_candidates()
    if set(result.cross_validation) != set(candidates):
        raise ValueError("Tracked candidates differ from evaluated models.")
    if result.selected_model not in candidates:
        raise ValueError("Selected model was not evaluated.")
    best = max(result.cross_validation, key=lambda name: result.cross_validation[name]["f1"])
    if best != result.selected_model or result.selection_metric != "cross_validated_f1":
        raise ValueError("Only the cross-validated F1 winner may be registered.")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    run_ids = {}
    model_info = None
    for name, estimator in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            run_ids[name] = run.info.run_id
            mlflow.set_tags({
                "dataset": "private_local_csv",
                "evaluation": "stratified_cross_validation",
                "selection_metric": "f1",
                "selected": str(name == best).lower(),
            })
            mlflow.log_params({
                "classifier": name,
                "random_state": result.random_state,
                "cv_folds": result.cv_folds,
                "train_rows": result.training_rows,
                "test_rows": result.test_rows,
            })
            # Include a few estimator settings, not any customer-level data.
            mlflow.log_params({
                "estimator_" + key: str(value)
                for key, value in estimator.get_params(deep=False).items()
            })
            mlflow.log_metrics({
                "cv_" + metric: score
                for metric, score in result.cross_validation[name].items()
            })
            if name == best:
                mlflow.log_metrics({
                    "holdout_" + metric: score
                    for metric, score in result.test_metrics.items()
                })
                # No input_example: it would persist a private customer record.
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    name="churn_pipeline",
                    registered_model_name=registered_model_name,
                    await_registration_for=120,
                )

    if model_info is None:
        raise RuntimeError("No selected model was logged.")
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    matches = [v for v in versions if v.run_id == run_ids[best]]
    if len(matches) != 1:
        raise RuntimeError("Could not resolve the selected run's registry version.")
    version = matches[0].version
    client.set_registered_model_alias(
        registered_model_name, CHAMPION_ALIAS, version,
    )
    return {
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "registered_model_name": registered_model_name,
        "registered_version": str(version),
        "model_alias": CHAMPION_ALIAS,
        "selected_model": best,
        "selection_metric": "cross_validated_f1",
        "candidate_run_ids": run_ids,
        "selected_run_id": run_ids[best],
        "registered_uri": f"models:/{registered_model_name}/{version}",
        "alias_uri": f"models:/{registered_model_name}@{CHAMPION_ALIAS}",
    }
