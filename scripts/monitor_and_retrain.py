"""Run labeled-batch monitoring periodically and retrain when accuracy drops.

This scheduler runs only while this process is active. No private dataset is
sent to GitHub Actions. Every retraining event logs a new local MLflow version.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from bank_churn.api import load_trusted_model
from bank_churn.data import load_churn_data
from bank_churn.monitoring import monitor_model


def run_cycle(
    *,
    reference_path: Path,
    observed_path: Path,
    model_path: Path,
    metrics_path: Path,
    report_dir: Path,
    retraining_input: Path | None = None,
    tracking_database: Path | None = None,
    allow_retraining: bool = False,
    accuracy_drop: float = 0.05,
    now: datetime | None = None,
) -> dict:
    """One monitoring cycle; retrain only on a measured drop and explicit opt-in."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Monitoring time must have a timezone.")
    baseline = json.loads(metrics_path.read_text(encoding="utf-8"))
    baseline_accuracy = float(baseline["test_metrics"]["accuracy"])
    reference = load_churn_data(reference_path)
    observed = load_churn_data(observed_path)
    model = load_trusted_model(model_path)
    report = monitor_model(
        model, reference, observed, baseline_accuracy=baseline_accuracy,
        accuracy_drop=accuracy_drop,
    )
    report["checked_at_utc"] = now.astimezone(timezone.utc).isoformat()
    report["retraining_executed"] = False
    report["retraining_skip_reason"] = None

    if report["retraining_recommended"]:
        if not allow_retraining:
            report["retraining_skip_reason"] = "not_authorized"
        elif retraining_input is None or tracking_database is None:
            report["retraining_skip_reason"] = "missing_labeled_training_input_or_tracking"
        else:
            from bank_churn.experiment_tracking import local_tracking_uri, log_experiment
            from bank_churn.training import save_training_artifacts, train_and_compare

            # Labeled retraining input is supplied separately. Avoid repeatedly
            # training on the same monitoring batch or modifying the live model.
            latest = load_churn_data(retraining_input)
            trained, scores = train_and_compare(latest)
            uri = local_tracking_uri(tracking_database)
            tracked = log_experiment(trained, scores, tracking_uri=uri)
            dest = report_dir / "retrained" / now.strftime("%Y%m%dT%H%M%S%fZ")
            save_training_artifacts(trained, scores, dest)
            report["retraining_executed"] = True
            report["registered_version"] = tracked["registered_version"]
            report["retrained_model_path"] = str(dest / "model.joblib")

    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / ("monitor-" + now.strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("data/raw/Churn_Modeling.csv"))
    parser.add_argument("--observed", type=Path, required=True,
                        help="Separate recent customer records with known Exited outcomes")
    parser.add_argument("--model", type=Path, default=Path("artifacts/churn-model/model.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/churn-model/metrics.json"))
    parser.add_argument("--reports", type=Path, default=Path("reports/monitoring"))
    parser.add_argument("--retraining-input", type=Path,
                        help="Approved fresh labeled data for retraining, not monitoring predictions")
    parser.add_argument("--tracking-database", type=Path, default=Path("artifacts/mlflow.db"))
    parser.add_argument("--allow-retraining", action="store_true")
    parser.add_argument("--accuracy-drop", type=float, default=0.05)
    parser.add_argument("--interval-minutes", type=int, default=0,
                        help="0 runs once; positive values keep checking on a local schedule")
    args = parser.parse_args()

    if args.interval_minutes < 0:
        parser.error("Interval must be zero or greater.")
    if args.allow_retraining and args.retraining_input is None:
        parser.error("--retraining-input is required with --allow-retraining.")

    try:
        while True:
            report = run_cycle(
                reference_path=args.reference,
                observed_path=args.observed,
                model_path=args.model,
                metrics_path=args.metrics,
                report_dir=args.reports,
                retraining_input=args.retraining_input,
                tracking_database=args.tracking_database,
                allow_retraining=args.allow_retraining,
                accuracy_drop=args.accuracy_drop,
            )
            print(
                "Observed accuracy:", report["observed_accuracy"],
                "retraining recommended:", report["retraining_recommended"],
                "retraining executed:", report["retraining_executed"],
                flush=True,
            )
            if args.interval_minutes == 0:
                return 0
            time.sleep(args.interval_minutes * 60)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"Monitoring failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
