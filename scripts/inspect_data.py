"""Validate the customer CSV and create aggregate EDA outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

from bank_churn.data import load_churn_data
from bank_churn.eda import write_eda


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/Churn_Modeling.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/eda"))
    args = parser.parse_args()
    try:
        frame = load_churn_data(args.input)
        summary = write_eda(frame, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Data inspection failed: {exc}\n")
    print(f"Customers: {summary['rows']}")
    print(f"Churned: {summary['churned']}")
    print(f"Churn rate: {summary['churn_rate']:.2%}")
    print(f"Aggregate report: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
