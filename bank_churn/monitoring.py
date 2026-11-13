"""Batch monitoring for churn model accuracy and basic feature drift.

Only aggregate metrics are returned, never individual records or predictions.
Observed churn labels must represent outcomes that became known later.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .data import CATEGORICAL, NUMERIC, split_features_target


def monitor_model(
    model,
    reference: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    baseline_accuracy: float,
    accuracy_drop: float = 0.05,
    numeric_shift_threshold: float = 0.5,
    category_shift_threshold: float = 0.2,
    min_batch_size: int = 30,
) -> dict:
    """Compare a recent labeled batch to a reference and its original score.

    Drift checks are simple simulated indicators, not statistical hypothesis
    tests. A sustained accuracy decline can indicate a need for retraining;
    drift alone generates an alert but does not automatically retrain.
    """
    if (not 0 <= baseline_accuracy <= 1 or not 0 < accuracy_drop <= 1
            or numeric_shift_threshold <= 0 or category_shift_threshold <= 0):
        raise ValueError("Monitoring thresholds must be valid.")
    if min_batch_size < 1 or len(observed) < min_batch_size:
        raise ValueError("Insufficient labeled observations for monitoring.")
    reference_x, _ = split_features_target(reference)
    observed_x, observed_y = split_features_target(observed)
    predictions = model.predict(observed_x)
    accuracy = float(accuracy_score(observed_y, predictions))
    delta = baseline_accuracy - accuracy

    drift = {}
    for feature in NUMERIC:
        r = pd.to_numeric(reference_x[feature], errors="coerce").dropna()
        o = pd.to_numeric(observed_x[feature], errors="coerce").dropna()
        if r.empty or o.empty:
            drift[feature] = {"kind": "numeric", "shift": None, "alert": True}
            continue
        scale = max(float(r.std(ddof=0)), 1e-6)
        shift = abs(float(o.mean()) - float(r.mean())) / scale
        drift[feature] = {
            "kind": "numeric",
            "shift": round(shift, 6),
            "alert": shift >= numeric_shift_threshold,
        }
    for feature in CATEGORICAL:
        ref = reference_x[feature].fillna("<missing>").value_counts(normalize=True)
        obs = observed_x[feature].fillna("<missing>").value_counts(normalize=True)
        categories = ref.index.union(obs.index)
        shift = 0.5 * sum(
            abs(float(ref.get(key, 0)) - float(obs.get(key, 0)))
            for key in categories
        )
        drift[feature] = {
            "kind": "categorical",
            "shift": round(shift, 6),
            "alert": shift >= category_shift_threshold,
        }
    return {
        "evaluation_type": "labeled_batch_monitoring",
        "reference_rows": int(len(reference)),
        "observed_rows": int(len(observed)),
        "baseline_accuracy": round(float(baseline_accuracy), 6),
        "observed_accuracy": round(accuracy, 6),
        "accuracy_drop": round(delta, 6),
        "accuracy_drop_threshold": accuracy_drop,
        "retraining_recommended": bool(delta >= accuracy_drop),
        "drift_alert": any(value["alert"] for value in drift.values()),
        "feature_drift": drift,
    }
