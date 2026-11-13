"""Train and compare churn models on a private local CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

from bank_churn.data import load_churn_data
from bank_churn.training import save_training_artifacts, train_and_compare


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/Churn_Modeling.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/churn-model"))
    args = parser.parse_args()
    try:
        frame = load_churn_data(args.input)
        model, result = train_and_compare(frame)
        save_training_artifacts(model, result, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Model training failed: {exc}\n")
    print(f"Selected model: {result.selected_model}")
    print(f"Selection: {result.selection_metric}")
    for name, scores in result.cross_validation.items():
        print(f"{name}: cross-validated F1={scores['f1']:.4f}")
    print(f"Held-out test F1: {result.test_metrics['f1']:.4f}")
    print(f"Saved local artifacts: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
