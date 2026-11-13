"""Train churn models and log experiments in a local MLflow SQLite registry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bank_churn.data import load_churn_data
from bank_churn.experiment_tracking import local_tracking_uri, log_experiment
from bank_churn.training import save_training_artifacts, train_and_compare


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/Churn_Modeling.csv"))
    parser.add_argument("--database", type=Path, default=Path("artifacts/mlflow.db"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/churn-model"))
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.exit(2, "Output model folder already exists; choose a new output path.\n")
    try:
        frame = load_churn_data(args.input)
        model, scores = train_and_compare(frame)
        uri = local_tracking_uri(args.database)
        tracking = log_experiment(model, scores, tracking_uri=uri)
        save_training_artifacts(model, scores, args.output_dir)
        (args.output_dir / "registry.json").write_text(
            json.dumps(tracking, indent=2) + "\n", encoding="utf-8",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"MLflow experiment failed: {exc}\n")
    print(f"Selected model: {scores.selected_model}")
    print(f"Cross-validated selection F1: {scores.cross_validation[scores.selected_model]['f1']:.4f}")
    print(f"Held-out test F1: {scores.test_metrics['f1']:.4f}")
    print(f"MLflow registry: {tracking['registered_uri']}")
    print(f"Model alias: {tracking['alias_uri']}")
    print(f"Tracking database: {args.database}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
