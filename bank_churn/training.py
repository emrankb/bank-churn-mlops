"""Leakage-safe churn model comparison with a held-out final test set."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from .data import CATEGORICAL, NUMERIC, make_preprocessor, split_features_target

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
METRICS = ("accuracy", "precision", "recall", "f1")


@dataclass
class TrainingResult:
    """Aggregate model comparison and held-out score, never customer rows."""

    selected_model: str
    selection_metric: str
    cross_validation: dict
    test_metrics: dict
    training_rows: int
    test_rows: int
    positive_train: int
    positive_test: int
    random_state: int
    cv_folds: int

    def as_dict(self) -> dict:
        return asdict(self)


def model_candidates() -> dict:
    """Fixed, reproducible baseline and two trainable classifiers."""
    return {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=80, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=1,
        ),
    }


def _score(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def train_and_compare(frame: pd.DataFrame) -> tuple[Pipeline, TrainingResult]:
    """Select by cross-validated F1 on training rows; test only once afterwards.

    Preprocessing stays inside each estimator Pipeline, so cross-validation
    cannot fit the imputer, scaler, or encoder on its validation fold.
    """
    features, y = split_features_target(frame)
    counts = y.value_counts()
    if counts.min() < 8:
        raise ValueError("At least eight customers from each class are required.")
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    if y_train.value_counts().min() < CV_FOLDS:
        raise ValueError("Not enough training examples of each class for cross-validation.")
    folds = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE,
    )
    comparison = {}
    for name, estimator in model_candidates().items():
        pipeline = Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", estimator),
        ])
        observed = cross_validate(
            pipeline, X_train, y_train, cv=folds,
            scoring={metric: metric for metric in METRICS},
            error_score="raise", n_jobs=1,
        )
        comparison[name] = {
            metric: float(np.mean(observed["test_" + metric]))
            for metric in METRICS
        }

    # Stable tie-breaking: name order is explicit and not influenced by the test set.
    selected_name = max(comparison, key=lambda name: comparison[name]["f1"])
    best = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", model_candidates()[selected_name]),
    ])
    best.fit(X_train, y_train)
    predictions = best.predict(X_test)
    result = TrainingResult(
        selected_model=selected_name,
        selection_metric="cross_validated_f1",
        cross_validation=comparison,
        test_metrics=_score(y_test, predictions),
        training_rows=len(X_train),
        test_rows=len(X_test),
        positive_train=int(y_train.sum()),
        positive_test=int(y_test.sum()),
        random_state=RANDOM_STATE,
        cv_folds=CV_FOLDS,
    )
    return best, result


def save_training_artifacts(
    model: Pipeline, result: TrainingResult, output_dir: str | Path
) -> dict:
    """Write local model and metadata without overwriting existing artifacts.

    joblib uses pickle serialization. Only load an artifact you created and
    trust; untrusted joblib files can execute code when deserialized.
    """
    dest = Path(output_dir)
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite model directory: {dest}")
    dest.mkdir(parents=True)
    metadata = {
        "schema_version": 1,
        "input_features": [*NUMERIC, *CATEGORICAL],
        "selected_model": result.selected_model,
        "selection_metric": result.selection_metric,
        "random_state": result.random_state,
        "cv_folds": result.cv_folds,
        "training_rows": result.training_rows,
        "test_rows": result.test_rows,
        "positive_train": result.positive_train,
        "positive_test": result.positive_test,
        "cross_validation": result.cross_validation,
        "test_metrics": result.test_metrics,
    }
    (dest / "metrics.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    joblib.dump(model, dest / "model.joblib")
    return metadata
